"""Original guided MCTS with PUCT. Policy priors guide expansion,
value head replaces rollouts. Root visit distribution = training target.
"""
from __future__ import annotations

import chess
import numpy as np

from .game import encode_action, flip_action


class _Node:
    __slots__ = ("prior", "raw", "n", "w", "children", "expanded",
                 "terminal_z", "o", "m", "m_n", "context")

    def __init__(self, prior: float = 0.0, raw: float | None = None):
        self.prior = prior
        # raw network prior BEFORE Dirichlet mixing. Noise is re-mixed from
        # raw on every reuse (audit round 2: re-mixing into already-noised
        # priors decayed the network to 0.75^k after k reuses).
        self.raw = prior if raw is None else raw
        self.n = 0
        self.w = 0.0
        self.children: dict[int, "_Node"] = {}
        self.expanded = False
        self.terminal_z: float | None = None
        # B1 WU-style pending count (batched path only): concurrent descents
        # in one batch mark O+=1 so they diverge below the mark. Q stays
        # w/n (O never touches w). Initialized 0 here (v19f fix: the slot
        # existed but __init__ never set it, so every leaf_batch>1 search
        # raised AttributeError on first descent).
        self.o = 0
        # D3 in-tree MLH state: running mean of predicted plies-left
        # (perspective-invariant count, NO sign flip in backup), None until
        # the first ML backup lands. ml_fn=None or ml_slope==0.0 (defaults)
        # leaves every m None and selection bit-identical (bonus guarded).
        self.m = None
        self.m_n = 0
        self.context = None

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


def _select(child: _Node, total_n: int, c_puct: float) -> float:
    # audit-1.3: total_n is 0 on the first descent through a freshly
    # expanded node (all children n=0), which zeroed every score and picked
    # the lowest-index legal move instead of the policy's choice. max(1,..)
    # matches standard PUCT (sqrt of parent visits, >= 1 by then).
    u = c_puct * child.prior * np.sqrt(max(1, total_n)) / (1 + child.n)
    return child.q + u


def _select_wu(child: _Node, total_n: int, c_puct: float) -> float:
    """B1 WU-style selection (batched path ONLY; the legacy sequential loop
    keeps _select bit-exact). Pending descents count as visits in the
    exploration term: Nbar = N + O in both the sqrt and the denom, while Q
    stays w/n (O never touches w). With o == 0 this equals _select exactly.
    total_n here is the parent's Nbar (sum of children n + o)."""
    nbar = child.n + child.o
    u = c_puct * child.prior * np.sqrt(max(1, total_n)) / (1 + nbar)
    return child.q + u


def _select_fpu(child: _Node, parent_q: float, total_n: int,
                c_puct: float, reduction: float) -> float:
    """B2 first-play urgency (Lc0 T50 lite): unvisited children (n == 0)
    read the PARENT's Q minus the reduction instead of their own Q (0.0);
    visited children read q + u unchanged. The u term is _select's u
    exactly. Callers pass reduction 0.0 for the root's children (reduction
    off at root) and skip this helper entirely when fpu_reduction == 0.0
    (default = yesterday bit-exact)."""
    u = c_puct * child.prior * np.sqrt(max(1, total_n)) / (1 + child.n)
    if child.n == 0:
        return parent_q - reduction + u
    return child.q + u


def _ml_bonus(child_m, parent_m, parent_q: float,
              slope: float, cap: float, thr: float) -> float:
    """D3 in-tree MLH bonus (Lc0 moves-left lite): when the parent is
    decisive (|Q_parent| > thr), nudge toward lines that shorten wins /
    delay losses: bonus = slope * clamp(child_m - parent_m, -cap, +cap) *
    sign(-Q_parent). M is stored in NORMALIZED moves-left units (model head
    output, 0..1; cap 0.07 ~= 21 plies at move_cap 300), so per-ply diffs
    (~1/300) stay inside the clamp and the bonus is proportional, not
    bang-bang. Returns 0.0 unless both M exist and the parent is decisive.
    Never raises (selection-adjacent)."""
    try:
        if child_m is None or parent_m is None:
            return 0.0
        if abs(parent_q) <= thr:
            return 0.0
        d = float(child_m) - float(parent_m)
        if d > cap:
            d = cap
        elif d < -cap:
            d = -cap
        s = -1.0 if parent_q > 0.0 else 1.0
        b = slope * d * s
        return b if np.isfinite(b) else 0.0
    except Exception:
        return 0.0


def _o_remove(path) -> None:
    """Mirror of _vl_remove for WU pending counts (B1): undo the O+=1
    marks along a descent once its leaf is resolved."""
    for node in path:
        node.o -= 1


def _root_decided(root: _Node) -> bool:
    """B4 futile/proven stop probe: the root has a well-supported decisive
    child (n >= 3 and |Q| >= 0.999, i.e. mate/adjudicated-decided). Never
    raises; pure read, no mutation."""
    # Neural saturation and training adjudication are not game-theoretic proof.
    # Search until its requested budget/deadline; solved-tree propagation can
    # be added separately with explicit rule-terminal provenance.
    return False


def prune_tt(tt: dict | None, next_key) -> None:
    """Bound memory + kill history-merge hazard (audit §6.5/§6.6).

    After a move is chosen, keep ONLY the promoted child's subtree —
    that is all reuse actually needs. The old code kept every stored
    child subtree for the whole game (~1190 nodes/ply, ~350k nodes /
    ~90MB at ply 300, entry cap never firing). Pruning also removes
    the position-keyed history-merge risk: a stored node can no longer
    be served for a position reached by a different route with a
    different halfmove_clock / rep_count, because only the current
    line's child survives.
    """
    if tt is None:
        return
    keep = tt.pop(next_key, None)
    tt.clear()
    if keep is not None:
        tt[next_key] = keep


def tt_live_nodes(tt: dict | None) -> int:
    """Count live _Node objects reachable from tt (dedup by id)."""
    if not tt:
        return 0
    seen: set[int] = set()
    stack = list(tt.values())
    n = 0
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        n += 1
        stack.extend(node.children.values())
    return n


def search(state, evaluate_fn, n_sims: int, c_puct: float = 1.414,
           dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
           depths: list | None = None, tt: dict | None = None,
           history: list | None = None,
           stats: dict | None = None,
           contempt: float = 0.0,
           asymmetric_contempt: bool = False,
           quiescence_depth: int = 2,
            forcing_bonus: float = 0.25,
            contempt_edge_scale: float = 0.0,
             leaf_batch: int = 1, virtual_loss: float = 1.0,
             fpu_reduction: float = 0.0, prune_singletons: bool = False,
             eval_cache: dict | None = None,
             ml_fn=None, ml_slope: float = 0.0, ml_cap: float = 0.07,
             ml_thr: float = 0.8, futile_stop: bool = False,
             deadline=None, stop_event=None) -> np.ndarray:
    """Run MCTS from state. evaluate_fn([states]) -> (priors[B,4096], values[B]).
    Returns pi: normalized visit counts over 4096.
    tt: optional persistent transposition store across moves (subtree reuse).
    Expanded children are stored under their own position keys, so the next
    search promotes the picked child and reuses its subtree every move.
    Fresh Dirichlet noise is re-mixed from stored RAW network priors on
    every reuse (not from already-noised priors).
    history: rep-keys of positions played before this state (game prefix).
    Threefold repetition uses the full State history, like game termination.
    stats: optional dict filled with {"root_visits": int, "reused": int,
    "new_sims": int, "extensions": int, "breadth": int,
    "visit_entropy": float}.
    root_visits is the EFFECTIVE budget (carried + new sims); n_sims is
    the NOMINAL new sims (PDF 30-50 locks the nominal knob — see §6.1).
    Pass tt=None for a fixed-budget search (no carry) when measuring
    the reuse confound or generating fixed-sharpness policy targets.
    contempt/asymmetric_contempt (v9 search contempt): draw leaf lines —
    prospective repetitions AND in-search terminal draws — back up a
    contempt-shaped score instead of 0.0, computed from the mover's
    material edge (see selfplay.draw_leaf_value). Value TARGETS stay pure;
    this steers only move choice. contempt=0.0 reproduces yesterday's
    behaviour bit for bit (all draw backups are ±0.0 either way).
    quiescence_depth (v10, extended v11/v12): fresh expansions reached IN
    CHECK, VIA A CAPTURE, VIA PROMOTION, or with a PROMO THREAT on the
    board (either side pawn on 7th) are answered along the prior-best
    child instead of evaluated raw (horizon effect) — up to this many
    extra plies per sim. Costs extra forwards only on forcing lines; the
    nominal sim count is untouched. 0 disables (yesterday's behaviour).
    forcing_bonus (v10, extended v12): at expansion, checking/capturing/
    promoting / pawn-to-6th/7th children get
    priors scaled by (1+bonus) — forcing lines are searched first, quiet
    lines later, instead of spraying visits evenly over flat priors. This
    SHARPENS targets toward tactics (own it: same doctrine as the veto).
    0.0 disables.
    fpu_reduction (B2, default 0.0 = yesterday bit-exact): first-play
    urgency for unvisited children (n == 0 score parent_q - reduction + u;
    visited score q + u unchanged). Root's children use reduction 0.0
    (Lc0 T50: reduction off at root). Guarded by param != 0.0: old configs
    (no such field) and today's callers behave identically.
    prune_singletons is retained for configuration compatibility but ignored;
    raw visit targets preserve weakly explored legal alternatives.
    futile_stop is compatibility-only: neural saturation is not a proof.
    Deadlines/cancellation stop between batches and quiescence extensions.
    eval_cache (B6, default None = yesterday): optional per-GAME dict
    (owned by the caller) keyed by (rep_key, rep_count) — rep planes 14-15
    depend on the counts, so the count is part of the key. Hits reuse the
    stored (priors, value); misses forward once and store. Quiescence
    forwards consult it too. None calls through untouched.
    stats gains (all never-raise measurement): "early_stop" (B4, 1 when the
    futile/proven stop fired) and "kl" (B5, KL of the visit distribution
    against the root RAW priors renormalized over legal; 0.0 on div0).
    ml_fn (D3, default None = yesterday bit-exact): optional callable
    states -> array of predicted moves-left in NORMALIZED units (model
    moves_left head output, 0..1; terminal M=0). evaluate_fn's contract is
    untouched ((priors, values) only) — ML arrives on this side channel so
    other files' evaluate closures keep working. ml_slope (default 0.0 =
    off = bit-exact; V19 0.003 via cfg), ml_cap (0.07), ml_thr (0.8): the
    selection bonus of _ml_bonus, applied to visited children at every
    interior node (no root exception — the |Q|>thr gate closes naturally
    there) in BOTH the legacy and batched paths. Quiescence extensions
    skip the ML update (expansion leaf's M stands). selfplay builds ml_fn
    from its model (one extra forward per batch/leaf); server path
    (model None) passes None (bonus off, logged once).
    """
    import time as _clock
    def cancelled():
        return ((stop_event is not None and stop_event.is_set()) or
                (deadline is not None and _clock.monotonic() >= deadline))
    if history is not None and tuple(history) != state._hist:
        state = type(state)(state.board.copy(stack=False), _hist=tuple(history))
    key = state.key()
    root = tt.get(key) if tt is not None else None
    context = (state.board.halfmove_clock, state._hist)
    if root is not None and root.context is not None and root.context != context:
        root = None
    reused = 0
    if root is not None and root.expanded:
        reused = sum(c.n for c in root.children.values())

    # v9 search contempt scorer, bound once per search. Deferred import:
    # selfplay imports this module, so a top-level import would cycle.
    from .selfplay import draw_leaf_value as _dlv

    def _draw_leaf_value(board) -> float:
        return _dlv(board, contempt, asymmetric_contempt,
                     contempt_edge_scale)

    def _cache_key(s) -> tuple:
        try:
            return (s.rep_key(), s.board.halfmove_clock, s._hist)
        except Exception:
            try:
                return (s.key(), 0)
            except Exception:
                return ("unkeyed", 0)

    cache_stats = {"cache_hits": 0, "cache_misses": 0, "cache_evictions": 0}
    def _eval_batch(states):
        """evaluate_fn with per-state caching (B6). Miss states share one
        forward; hits reuse the stored (priors row, value). eval_cache=None
        calls through untouched (yesterday's behaviour, bit for bit). Never
        swallows forward errors (those propagate as today); only the dict
        bookkeeping is guarded."""
        if eval_cache is None:
            return evaluate_fn(states)
        n = len(states)
        rows_p: list = [None] * n
        rows_v: list = [None] * n
        miss_idx: list[int] = []
        for i, s in enumerate(states):
            try:
                hit = eval_cache.get(_cache_key(s))
            except Exception:
                hit = None
            if hit is not None:
                cache_stats["cache_hits"] += 1
                rows_p[i], rows_v[i] = hit
            else:
                cache_stats["cache_misses"] += 1
                miss_idx.append(i)
        if miss_idx:
            mp, mv = evaluate_fn([states[i] for i in miss_idx])
            for j, i in enumerate(miss_idx):
                _p = np.asarray(mp[j])
                _v = float(np.asarray(mv[j]).squeeze())
                rows_p[i], rows_v[i] = _p, _v
                try:
                    if len(eval_cache) >= 2048:
                        eval_cache.pop(next(iter(eval_cache)))
                        cache_stats["cache_evictions"] += 1
                    eval_cache[_cache_key(states[i])] = (_p.copy(), _v)
                except Exception:
                    pass
        return (np.stack([np.asarray(r) for r in rows_p]),
                np.asarray(rows_v, dtype=np.float64))

    early_fired = False  # B4 futile/proven stop (set below / by batched)
    early_box: list = [False]  # batched path reports through here

    # D3 master switch: ml_fn present AND slope nonzero. Off = every m
    # stays None, every bonus skipped, RNG untouched (bit-exact legacy).
    _ml_on = ml_fn is not None and ml_slope != 0.0
    _lm_term = 0.0 if _ml_on else None  # M for terminal/loop lines

    def _ml_single(s):
        """D3 leaf M for one state via ml_fn (normalized units). Non-finite
        reads return None (that sample is skipped, not averaged in). ml_fn
        errors propagate (loud broken head, same policy as eval_batch)."""
        _v = float(np.asarray(ml_fn([s])).flatten()[0])
        return _v if np.isfinite(_v) else None

    def _orient(action_abs: int, black_stm: bool) -> int:
        return flip_action(action_abs) if black_stm else action_abs

    def _boost_forcing(cur_state, pv) -> None:
        """In-place prior boost for checking/capturing children (v10
        forcing-first ordering). No board copies: gives_check/is_capture
        read the absolute move directly.
        v12: also boosts promotions and pawn pushes to 6th/7th rank
        (promotion blindness: 38...Ke8?? walked away from b8=Q — the
        search must LOOK at passers). Same shared bonus, renormalized."""
        if not forcing_bonus:
            return
        black = cur_state.board.turn == chess.BLACK
        boosted = False
        bd = cur_state.board
        for cm in cur_state._legal_chess_moves():
            # v14 Rust core: capture/check queries via bitboards when the
            # extension is present (falls back to python-chess).
            if cur_state.is_capture_fast(cm) or \
                    cur_state.gives_check_fast(cm):
                boost = True
            elif cm.promotion is not None:
                boost = True
            else:
                # pawn push to 6th/7th rank (promotion threat, not yet a
                # promotion): White to rank 6/7 (idx 5/6), Black to 3/2.
                pc = bd.piece_at(cm.from_square)
                if pc is None or pc.piece_type != chess.PAWN:
                    boost = False
                elif bd.turn == chess.WHITE:
                    boost = chess.square_rank(cm.to_square) in (5, 6)
                else:
                    boost = chess.square_rank(cm.to_square) in (2, 1)
            if boost:
                a = _orient(encode_action(cm.from_square, cm.to_square, cm.promotion),
                            black)
                if pv[a] > 0:
                    pv[a] *= (1.0 + forcing_bonus)
                    boosted = True
        if boosted:
            s = sum(float(pv[a]) for a in cur_state.legal_moves())
            if s > 0:
                pv /= s

    def _has_promo_threat(bd) -> bool:
        """True if either side has a pawn one step from queening (White on
        rank 7, Black on rank 2). Promotion-imminent positions need depth:
        raw eval cannot see the queening. Cheap: at most ~8 pawns/side."""
        for sq in bd.pieces(chess.PAWN, chess.WHITE):
            if chess.square_rank(sq) == 6:
                return True
        for sq in bd.pieces(chess.PAWN, chess.BLACK):
            if chess.square_rank(sq) == 1:
                return True
        return False
    extensions = 0
    legal = state.legal_moves()
    if not legal:
        return np.zeros(4096, dtype=np.float32)

    if root is None or not root.expanded:
        root = _Node()
        # expand root
        priors, _ = _eval_batch([state])
        p = np.asarray(priors[0], dtype=np.float64)
        mask = state.legal_mask()
        p[~mask] = 0.0
        s = sum(float(p[a]) for a in legal)
        p = p / s if s > 0 else mask.astype(np.float64) / mask.sum()
        # v10 forcing-first: boost checks/captures BEFORE noise mixing, so
        # the raw priors (remixed on every reuse) already encode it.
        _boost_forcing(state, p)
        # dirichlet noise on legal moves only (raw kept for re-mixing)
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            mixed = (1 - dirichlet_eps) * p[a] + dirichlet_eps * noise[i]
            root.children[a] = _Node(prior=float(mixed), raw=float(p[a]))
        root.expanded = True
    elif dirichlet_eps > 0:
        # reused subtree: remix fresh noise from RAW network priors.
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            if a in root.children:
                p0 = root.children[a].raw
                root.children[a].prior = (
                    (1 - dirichlet_eps) * p0 + dirichlet_eps * noise[i])
        # drop children that are no longer legal (rare: promotion filtering)
        for a in list(root.children):
            if a not in legal:
                del root.children[a]
    if tt is not None:
        if len(tt) > 20000:
            # oldest-first eviction (dicts are insertion-ordered)
            for k in list(tt)[:len(tt) - 20000]:
                del tt[k]
        root.context = context
        tt[key] = root

    history_set = set(history) if history else set()  # O(H) once
    if leaf_batch > 1:
        # v16.1 batched leaves (same math, shared forwards): the legacy
        # sequential loop below runs zero sims; all visits come from here.
        # B1: the batched path now uses WU pending counts (O) instead of
        # the -VL penalty; B2 FPU threads through when fpu_reduction != 0.
        try:
            extensions += _search_batched(
                root, state, _eval_batch, n_sims, c_puct, quiescence_depth,
                forcing_bonus, history, history_set, _draw_leaf_value,
                _orient, _boost_forcing, _has_promo_threat, depths,
                leaf_batch, virtual_loss, fpu_reduction, early_box,
                ml_fn, ml_slope, ml_cap, ml_thr, futile_stop, cancelled)
        finally:
            stack = [root]
            while stack:
                pending_node = stack.pop()
                pending_node.o = 0
                stack.extend(pending_node.children.values())
        early_fired = early_fired or bool(early_box[0])
    for _ in range(0 if leaf_batch > 1 else n_sims):
        if cancelled():
            early_fired = True
            break
        if futile_stop and _root_decided(root):
            # B4 futile/proven stop, MEASUREMENT ONLY (gate/arena/UCI):
            # a well-supported decisive child means further sims cannot
            # move the root pick — stop early (stats flag in the tail).
            # OFF by default: training needs the full visit distribution
            # as its target, and early exit would distort it. Pure read;
            # unfired searches are bit-identical either way.
            early_fired = True
            break
        node = root
        cur = state
        path = [node]
        actions = []
        path_keys = [state.rep_key()]
        path_set = {path_keys[0]}  # mirrors path_keys
        qext = 0  # v10 quiescence extensions used this sim (bounded)
        # v11 capture extension: True when the arrival move captured.
        # Captures change the piece count by exactly -1 (incl. en passant
        # and promotion-captures; quiet promotions/castling keep it) — a
        # ~µs proxy, no board copies, no move matching.
        # v12 promotion extension: quiet promotions keep the piece count
        # (pawn->queen swap) so the capture proxy misses them. Track the
        # mover's pawn count: a drop means the arrival move promoted.
        # Plus a position threat flag: either side with a pawn on 7th
        # (about to queen) needs depth even when the arrival was quiet.
        n_pieces = len(cur.board.piece_map())
        mover = cur.board.turn
        n_pawns_mover = len(cur.board.pieces(chess.PAWN, mover))
        arrived_by_capture = False
        arrived_by_promo = False
        # selection
        while node.expanded and node.children:
            total = sum(c.n for c in node.children.values())
            best_a, best_child, best_s = None, None, -1e18
            if fpu_reduction != 0.0:
                # B2 FPU (guarded: default 0.0 runs the exact block below).
                # Unvisited children read the parent's Q minus the
                # reduction; the root's children use reduction 0.0.
                _pq = -node.q
                _red = 0.0 if node is root else fpu_reduction
                for a, ch in node.children.items():
                    sc = _select_fpu(ch, _pq, total, c_puct, _red)
                    if _ml_on:  # D3 (skipped entirely when off)
                        sc += _ml_bonus(ch.m, node.m, _pq, ml_slope,
                                        ml_cap, ml_thr)
                    if sc > best_s + 1e-12:  # stable ties across NumPy/Rust floating arithmetic
                        best_s, best_a, best_child = sc, a, ch
            else:
                for a, ch in node.children.items():
                    sc = _select(ch, total, c_puct)
                    if _ml_on:  # D3 (skipped entirely when off)
                        sc += _ml_bonus(ch.m, node.m, -node.q, ml_slope,
                                        ml_cap, ml_thr)
                    if sc > best_s + 1e-12:  # stable ties across NumPy/Rust floating arithmetic
                        best_s, best_a, best_child = sc, a, ch
            actions.append(best_a)
            cur = cur.apply(best_a)
            new_n_pieces = len(cur.board.piece_map())
            arrived_by_capture = new_n_pieces < n_pieces
            n_pieces = new_n_pieces
            # v12: pawn-count drop of the mover = promotion (quiet or
            # capturing). Mover is the side that just moved = opposite of
            # the new side to move; n_pawns_mover tracked pre-move above,
            # refresh for the next ply (new mover = new side to move).
            just_moved = not cur.board.turn
            try:
                new_n_pawns = len(cur.board.pieces(chess.PAWN, just_moved))
            except Exception:
                new_n_pawns = n_pawns_mover
            arrived_by_promo = new_n_pawns < n_pawns_mover
            mover = cur.board.turn
            try:
                n_pawns_mover = len(cur.board.pieces(chess.PAWN, mover))
            except Exception:
                pass
            node = best_child
            path.append(node)
            rk = cur.rep_key()
            if cur.rep_count() >= 2:
                # Second occurrence: dead loop. v9 backs up the contempt-
                # shaped draw score (leaf view, negated to the mover-view
                # convention _backup uses — matching the -z below); at
                # contempt=0 this is -0.0 == yesterday's 0.0 exactly.
                _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                if depths is not None:
                    depths.append(len(actions))
                break
            path_keys.append(rk)
            path_set.add(rk)
            done, z = cur.is_terminal()
            if done:
                if z == 0.0:
                    # In-search terminal draw (rules or adjudicated): same
                    # contempt treatment as prospective loops. Mates and
                    # decisive adjudications (z=±1) are untouched.
                    _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                else:
                    # z is from cur side-to-move view; backup flips per ply
                    _backup(path, -z, _lm_term)
                if depths is not None:
                    depths.append(len(actions))
                break
        else:
            done, z = cur.is_terminal()
            if done:
                if z == 0.0:
                    _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                else:
                    _backup(path, -z, _lm_term)
                if depths is not None:
                    depths.append(len(actions))
                continue
            priors_b, values_b = _eval_batch([cur])
            pv = np.asarray(priors_b[0], dtype=np.float64)
            leaf_v = float(np.asarray(values_b[0]).squeeze())
            # D3: expansion leaf's M (quiescence below keeps this value).
            leaf_m = _ml_single(cur) if _ml_on else None
            m = cur.legal_mask()
            pv[~m] = 0.0
            tot = sum(float(pv[a]) for a in cur.legal_moves())
            pv = pv / tot if tot > 0 else m.astype(np.float64) / m.sum()
            _boost_forcing(cur, pv)
            for a in cur.legal_moves():
                node.children[a] = _Node(prior=float(pv[a]))
            node.expanded = True
            # v10 quiescence (checks only, bounded): a leaf with the side
            # to move IN CHECK is not evaluated raw (horizon effect).
            # Answer along the prior-best child instead, up to
            # quiescence_depth extra plies. Costs extra forwards only on
            # checking lines (rare); the nominal sim count is untouched,
            # and every extension is logged in stats for telemetry.
            # v11: arrival-via-capture extends too (tactical lines deserve
            # depth; quiet lines don't). Shared bounded pool.
            # v12: promotions and promo threats extend too (38...Ke8??
            # walked away from b8=Q — stopping a passer needs the search
            # to LOOK). Quiet promotions keep the piece count, so the
            # capture proxy misses them: arrived_by_promo (pawn-count drop)
            # plus _has_promo_threat (either side pawn on 7th).
            while (not cancelled() and quiescence_depth > 0 and qext < quiescence_depth
                     and (cur.in_check_fast() or arrived_by_capture
                         or arrived_by_promo
                         or _has_promo_threat(cur.board))
                    and node.children):
                qext += 1
                extensions += 1
                best_a = max(node.children,
                             key=lambda a: node.children[a].prior)
                actions.append(best_a)
                cur = cur.apply(best_a)
                new_n_pieces = len(cur.board.piece_map())
                arrived_by_capture = new_n_pieces < n_pieces
                n_pieces = new_n_pieces
                just_moved = not cur.board.turn
                try:
                    new_n_pawns = len(
                        cur.board.pieces(chess.PAWN, just_moved))
                except Exception:
                    new_n_pawns = n_pawns_mover
                arrived_by_promo = new_n_pawns < n_pawns_mover
                mover = cur.board.turn
                try:
                    n_pawns_mover = len(
                        cur.board.pieces(chess.PAWN, mover))
                except Exception:
                    pass
                node = node.children[best_a]
                path.append(node)
                rk = cur.rep_key()
                if cur.rep_count() >= 2:
                    _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                    leaf_v = None
                    break
                path_keys.append(rk)
                path_set.add(rk)
                done2, z2 = cur.is_terminal()
                if done2:
                    if z2 == 0.0:
                        _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                    else:
                        _backup(path, -z2, _lm_term)
                    leaf_v = None
                    break
                priors_q, values_q = _eval_batch([cur])
                pv = np.asarray(priors_q[0], dtype=np.float64)
                leaf_v = float(np.asarray(values_q[0]).squeeze())
                mq = cur.legal_mask()
                pv[~mq] = 0.0
                tot = sum(float(pv[a]) for a in cur.legal_moves())
                pv = pv / tot if tot > 0 else mq.astype(np.float64) / mq.sum()
                _boost_forcing(cur, pv)
                for a in cur.legal_moves():
                    node.children[a] = _Node(prior=float(pv[a]))
                node.expanded = True
            if leaf_v is not None:
                _backup(path, -leaf_v, leaf_m)
            if depths is not None:
                depths.append(len(actions))
            continue

    visits = np.zeros(4096, dtype=np.float32)
    for a, ch in root.children.items():
        visits[a] = ch.n
    if False:  # deprecated: singleton removal corrupts visit targets
        # B3 KataGo-lite decoupling: drop N<=1 singletons BEFORE normalize.
        # Default False = yesterday. Breadth below counts visits > 0 AFTER
        # pruning (documented shift when the flag is on). Degenerate case
        # (budget < breadth: every child a singleton) would zero the whole
        # vector — argmax would pin move 0 and temperature sampling would
        # crash — so fall back to the unpruned counts (documented; V19
        # budgets of 400 never hit it, tiny-budget tests do).
        visits[visits <= 1] = 0.0
        if visits.sum() == 0:
            for a, ch in root.children.items():
                visits[a] = ch.n
    tot = visits.sum()
    if stats is not None:
        stats.update(cache_stats)
        stats["root_visits"] = int(tot)
        stats["reused"] = int(reused)
        stats["new_sims"] = max(0, sum(c.n for c in root.children.values()) - reused)
        stats["extensions"] = int(extensions)
        stats["early_stop"] = 1 if early_fired else 0  # B4
        # v14 visit breadth (unconventional-move diagnostic): distinct
        # root moves visited + entropy of the visit distribution, measured
        # on RAW counts before normalization. A peaked prior starves deep
        # moves at low sims; breadth should RISE with sims if that trap
        # is opening. Callers average per game into stats["breadth"].
        _lv = visits[visits > 0]
        stats["breadth"] = int(_lv.shape[0])
        if tot > 0 and _lv.shape[0]:
            _pv = _lv.astype(np.float64) / float(tot)
            stats["visit_entropy"] = float(-(np.log(_pv) * _pv).sum())
        else:
            stats["visit_entropy"] = 0.0
        # B5 KL(visits || root_priors): how far the search moved from the
        # network's raw priors (renormalized over legal). Guard div0;
        # never raises (measurement code).
        try:
            _kl = 0.0
            if tot > 0:
                _raws = []
                _rsum = 0.0
                for a in legal:
                    _cc = root.children.get(a)
                    _r = float(_cc.raw) if _cc is not None else 0.0
                    _raws.append(_r)
                    _rsum += _r
                if _rsum > 0 and np.isfinite(_rsum):
                    for a, _r in zip(legal, _raws):
                        _v = float(visits[a]) / float(tot)
                        _p = _r / _rsum
                        if _v > 0.0 and _p > 0.0:
                            _kl += _v * float(np.log(_v / _p))
                    if not np.isfinite(_kl):
                        _kl = 0.0
        except Exception:
            _kl = 0.0
        stats["kl"] = float(_kl)
        # V20 D2 root_q (TD-target logging, Agent B): visit-weighted mean
        # child Q at the root (the search's value estimate for this
        # position). 0.0 when nothing visited. Additive dict entry only —
        # return signature unchanged (yesterday bit-exact otherwise).
        try:
            _rq = 0.0
            if tot > 0:
                _rq = float(sum(float(visits[a]) * float(
                    root.children[a].q) for a in root.children
                    if float(visits[a]) > 0.0) / float(tot))
            stats["root_q"] = float(_rq) if np.isfinite(_rq) else 0.0
        except Exception:
            stats["root_q"] = 0.0
    if tot > 0:
        visits /= tot
    else:
        visits[legal] = 1.0 / max(1, len(legal))
    if tt is not None:
        # audit round 2 §1.4: store expanded children under their own
        # position keys AFTER the sims (they only gain visits during the
        # loop). The next search promotes the picked child and reuses its
        # whole subtree every move, not just on loop-lines.
        # NOTE (§6.1): tot above is the EFFECTIVE budget (carried + new).
        # Callers prune to the promoted child AFTER choosing a move
        # (prune_tt) — without pruning the dict retains every line.
        for a, ch in root.children.items():
            if ch.expanded and ch.n > 0:
                try:
                    next_state = state.apply(a)
                    ch.context = (next_state.board.halfmove_clock, next_state._hist)
                    tt[next_state.key()] = ch
                except Exception:
                    pass
    return visits


def _vl_add(path, vl: float) -> None:
    """Child-only virtual loss (v16.1 leaf batching): the chosen child
    reads one virtual visit carrying a loss in its own view, so concurrent
    gathers in the same batch diverge below it. Ancestor totals derive
    from children sums (selection recomputes them), so only the child
    needs the mark — exact-once, no overcount. Removed by _vl_remove
    before the real backup. B1: retained for API stability; the batched
    path now uses WU pending counts (o) instead and no longer calls this."""
    for node in path:
        node.n += 1
        node.w -= vl


def _vl_remove(path, vl: float) -> None:
    for node in path:
        node.n -= 1
        node.w += vl


def _search_batched(root, state, evaluate_fn, n_sims: int, c_puct: float,
                    quiescence_depth: int, forcing_bonus: float,
                    history, history_set, dlv, orient_fn, boost_fn,
                    promo_threat_fn, depths, leaf_batch: int,
                     virtual_loss: float, fpu_reduction: float = 0.0,
                     early_box: list | None = None,
                     ml_fn=None, ml_slope: float = 0.0,
                     ml_cap: float = 0.07, ml_thr: float = 0.8,
                     futile_stop: bool = False, cancelled=lambda: False) -> int:
    """Batched-leaf MCTS (v16.1): same selection/expansion/quiescence math
    as the sequential loop, but up to leaf_batch leaves share ONE forward.
    B1: each gather descent marks O+=1 (WU pending count) so descents in a
    batch diverge; the batch evaluates together; each leaf then expands,
    runs the standard quiescence loop (single forwards, rare path), and
    backs up after O removal. The virtual_loss arg is accepted-but-ignored
    (API stability): O+=1 always, scaling nothing. Terminals/loops back up
    immediately (no forward needed). Pending-node collisions (forced
    single-reply lines) retry the descent, then fall back to an immediate
    single eval — correctness over batching, always. B2 FPU threads through
    when fpu_reduction != 0.0 (root's children: reduction 0.0). B4: breaks
    early on a decided root, reporting through early_box. D3: the batch's
    leaves share ONE ml_fn forward (plus solo calls for immediate-fallback
    leaves); terminals/loops back up M=0.0; quiescence keeps the expansion
    M. ml_slope==0.0 or ml_fn=None skips all of it (bit-exact). Returns
    quiescence extension count.
    """
    import chess as _ch
    extensions = 0
    sims_done = 0
    _fpu_on = fpu_reduction != 0.0
    _ml_on = ml_fn is not None and ml_slope != 0.0
    _lm_term = 0.0 if _ml_on else None
    while sims_done < n_sims:
        if cancelled():
            break
        if futile_stop and _root_decided(root):
            if early_box is not None:
                early_box[0] = True
            break
        batch_n = min(leaf_batch, n_sims - sims_done)
        pending: list = []
        pending_ids: set[int] = set()
        attempts = 0
        while len(pending) < batch_n and attempts < batch_n * 3 + 1:
            attempts += 1
            node = root
            cur = state
            path = [node]
            actions: list[int] = []
            path_keys = [state.rep_key()]
            path_set = {path_keys[0]}
            n_pieces = len(cur.board.piece_map())
            mover = cur.board.turn
            n_pawns_mover = len(cur.board.pieces(_ch.PAWN, mover))
            arrived_by_capture = False
            arrived_by_promo = False
            fell_back = False
            # selection with WU pending counts (B1): Nbar = N + O in the
            # exploration term (sqrt + denom), Q unchanged; FPU (B2) for
            # unvisited children when enabled (root's children: red 0.0).
            while node.expanded and node.children:
                total = sum(c.n + c.o for c in node.children.values())
                best_a, best_child, best_s = None, None, -1e18
                _pq = -node.q
                _red = 0.0 if (node is root or not _fpu_on) \
                    else fpu_reduction
                for a, ch in node.children.items():
                    if _fpu_on and ch.n == 0:
                        _nb = ch.n + ch.o
                        _u = c_puct * ch.prior * np.sqrt(max(1, total)) \
                            / (1 + _nb)
                        sc = _pq - _red + _u
                    else:
                        sc = _select_wu(ch, total, c_puct)
                    if _ml_on:  # D3 (skipped entirely when off)
                        sc += _ml_bonus(ch.m, node.m, _pq, ml_slope,
                                        ml_cap, ml_thr)
                    if sc > best_s + 1e-12:  # stable ties across NumPy/Rust floating arithmetic
                        best_s, best_a, best_child = sc, a, ch
                actions.append(best_a)
                path.append(best_child)
                best_child.o += 1
                cur = cur.apply(best_a)
                new_n_pieces = len(cur.board.piece_map())
                arrived_by_capture = new_n_pieces < n_pieces
                n_pieces = new_n_pieces
                just_moved = not cur.board.turn
                try:
                    new_n_pawns = len(
                        cur.board.pieces(_ch.PAWN, just_moved))
                except Exception:
                    new_n_pawns = n_pawns_mover
                arrived_by_promo = new_n_pawns < n_pawns_mover
                mover = cur.board.turn
                try:
                    n_pawns_mover = len(cur.board.pieces(_ch.PAWN, mover))
                except Exception:
                    pass
                node = best_child
                rk = cur.rep_key()
                if cur.rep_count() >= 2:
                    _o_remove(path[1:])
                    _backup(path, -dlv(cur.board))
                    if depths is not None:
                        depths.append(len(actions))
                    sims_done += 1
                    fell_back = True
                    break
                path_keys.append(rk)
                path_set.add(rk)
                done, z = cur.is_terminal()
                if done:
                    _o_remove(path[1:])
                    if z == 0.0:
                        _backup(path, -dlv(cur.board))
                    else:
                        _backup(path, -z, _lm_term)
                    if depths is not None:
                        depths.append(len(actions))
                    sims_done += 1
                    fell_back = True
                    break
            if fell_back:
                continue
            done, z = cur.is_terminal()
            if done:
                _o_remove(path[1:])
                if z == 0.0:
                    _backup(path, -dlv(cur.board))
                else:
                    _backup(path, -z, _lm_term)
                if depths is not None:
                    depths.append(len(actions))
                sims_done += 1
                continue
            if id(node) in pending_ids:
                # collision with a leaf stashed this batch: undo this
                # descent and retry (its O mark now diverts us). Bounded;
                # persistent collisions fall back to a single eval.
                _o_remove(path[1:])
                if attempts >= batch_n * 3:
                    # Flush the existing unique leaves, then descend again.
                    # Never enqueue an already released reservation or expand
                    # a leaf twice while another path owns it.
                    break
                continue
            pending.append((cur, node, path, actions, path_keys, path_set,
                            n_pieces, n_pawns_mover, arrived_by_capture,
                            arrived_by_promo, False))
            pending_ids.add(id(node))
            sims_done += 1
        # one shared forward for the batch (immediate leaves re-eval solo)
        batched = [p for p in pending if not p[10]]
        if batched:
            priors_b, values_b = evaluate_fn([p[0] for p in batched])
        # D3: the batch's leaves share ONE ml forward (normalized units).
        # ml_fn errors propagate (loud broken head, same policy as the
        # value forward above); only non-finite READS sanitize to None.
        ml_arr = None
        if _ml_on and batched:
            ml_arr = np.asarray(ml_fn([p[0] for p in batched]),
                                dtype=np.float64).flatten()
        bi = 0
        for (cur, node, path, actions, path_keys, path_set, n_pieces,
                n_pawns_mover, arrived_by_capture, arrived_by_promo,
                immediate) in pending:
            _o_remove(path[1:])
            leaf_m = None
            if immediate:
                priors_1, values_1 = evaluate_fn([cur])
                import numpy as _np1
                pv = _np1.asarray(priors_1[0], dtype=_np1.float64)
                leaf_v = float(_np1.asarray(values_1[0]).squeeze())
                if _ml_on:  # solo fallback leaf: solo ML read
                    _lm1 = float(np.asarray(ml_fn([cur]),
                                             dtype=np.float64)
                                 .flatten()[0])
                    leaf_m = _lm1 if np.isfinite(_lm1) else None
            else:
                import numpy as _np1
                pv = _np1.asarray(priors_b[bi], dtype=_np1.float64)
                leaf_v = float(_np1.asarray(values_b[bi]).squeeze())
                if _ml_on and ml_arr is not None \
                        and bi < len(ml_arr):
                    _lm2 = float(ml_arr[bi])
                    leaf_m = _lm2 if np.isfinite(_lm2) else None
                bi += 1
            m = cur.legal_mask()
            pv[~m] = 0.0
            tot = sum(float(pv[a]) for a in cur.legal_moves())
            pv = pv / tot if tot > 0 else m.astype(_np1.float64) / m.sum()
            boost_fn(cur, pv)
            for a in cur.legal_moves():
                node.children[a] = _Node(prior=float(pv[a]))
            node.expanded = True
            qext = 0
            while (not cancelled() and quiescence_depth > 0 and qext < quiescence_depth
                    and (cur.in_check_fast() or arrived_by_capture
                         or arrived_by_promo
                         or promo_threat_fn(cur.board))
                    and node.children):
                qext += 1
                extensions += 1
                best_a = max(node.children,
                             key=lambda a: node.children[a].prior)
                actions.append(best_a)
                cur = cur.apply(best_a)
                new_n_pieces = len(cur.board.piece_map())
                arrived_by_capture = new_n_pieces < n_pieces
                n_pieces = new_n_pieces
                just_moved = not cur.board.turn
                try:
                    new_n_pawns = len(cur.board.pieces(_ch.PAWN, just_moved))
                except Exception:
                    new_n_pawns = n_pawns_mover
                arrived_by_promo = new_n_pawns < n_pawns_mover
                mover = cur.board.turn
                try:
                    n_pawns_mover = len(cur.board.pieces(_ch.PAWN, mover))
                except Exception:
                    pass
                node = node.children[best_a]
                path.append(node)
                rk = cur.rep_key()
                if cur.rep_count() >= 2:
                    _backup(path, -dlv(cur.board))
                    leaf_v = None
                    break
                path_keys.append(rk)
                path_set.add(rk)
                done2, z2 = cur.is_terminal()
                if done2:
                    if z2 == 0.0:
                        _backup(path, -dlv(cur.board))
                    else:
                        _backup(path, -z2, _lm_term)
                    leaf_v = None
                    break
                priors_q, values_q = evaluate_fn([cur])
                import numpy as _np2
                pv = _np2.asarray(priors_q[0], dtype=_np2.float64)
                leaf_v = float(_np2.asarray(values_q[0]).squeeze())
                mq = cur.legal_mask()
                pv[~mq] = 0.0
                tot = sum(float(pv[a]) for a in cur.legal_moves())
                pv = pv / tot if tot > 0 else mq.astype(_np2.float64) / mq.sum()
                boost_fn(cur, pv)
                for a in cur.legal_moves():
                    node.children[a] = _Node(prior=float(pv[a]))
                node.expanded = True
            if leaf_v is not None:
                _backup(path, -leaf_v, leaf_m)
            if depths is not None:
                depths.append(len(actions))
    return extensions


def _backup(path: list[_Node], leaf_value: float, leaf_m=None):
    """Back up a leaf score along path, flipping sign each ply (zero-sum).
    Convention: callers pass the score NEGATED to mover-at-leaf view
    (parent view), so path[0] (root) accumulates opponent-view values;
    selection compares siblings (same view) so the offset is harmless.
    Do not "fix" the sign without re-deriving every caller.
    D3 M (plies-left, perspective-invariant count): averaged WITHOUT the
    sign flip (running mean m += (leaf_m - m)/n). leaf_m=None (ML off)
    leaves every m untouched (bit-exact); terminals/loops pass 0.0 (no
    plies remain); quiescence extensions keep the expansion leaf's M
    (callers simply don't recompute it)."""
    v = leaf_value
    for distance, node in enumerate(reversed(path)):
        node.n += 1
        node.w += v
        if leaf_m is not None:
            try:
                node.m_n += 1
                duration = leaf_m + distance / 300.0
                node.m = duration if node.m is None \
                    else node.m + (duration - node.m) / node.m_n
            except Exception:
                pass
        v = -v
