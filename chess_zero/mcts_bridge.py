"""V21 batch-D bridge: Python (MPS) evals feeding the MCTS tree.

BRIDGE CHOICE: batch-queue over the existing PyO3 extension (NOT C-ABI,
NOT per-leaf callback, NOT shared-memory). Rationale (see /tmp/v21_bridge.md):
 1. PyO3 transport reuses the proven chesscore toolchain (maturin, pinned
    pyo3 0.22.6) already building on this Mac; C-ABI adds dlopen/header
    maintenance for zero same-process gain.
 2. Batch-queue protocol amortizes the crossing: 400 sims / batch 8 =
    ~50 Python crossings per search instead of 400+ single-leaf callbacks
    (each paying GIL + tensor-convert cost). Quiescence singletons (rare
    path) bypass the queue via one direct call, same as batch C will do.
 3. The NN (MPS weights, torch forward) stays 100% Python-side: no Rust
    GPU code, no weight duplication, no device sync inside the tree.
    The bridge is device-agnostic (CPU/MPS/CUDA all verified: MPS repeat
    0.0, MPS-vs-CPU max|d| 3.9e-07 on best.pt 2026-09-25).
 4. Determinism: the bridge adds no RNG, no reordering, no math --
    batch order is preserved, vectors pass through untouched, so the
    tree sees bit-identical evals either side of the crossing.
REJECTED: C-ABI (unsafe surface, new build plumbing, same-process Mac
gains nothing); shared-memory queue (premature: unified-memory copies
are ~us; revisit only if crossings ever exceed 5% of search time);
per-leaf PyO3 callback (400+ GIL crossings/search, measured needless).

RUST BACKEND (V21.1 E1 landed: rust/mctscore ships a PyO3 `search` entry
point on the same pinned pyo3-0.22.6/maturin toolchain as chesscore).
RustTreeBackend prefers the real `mctscore.search`: Rust owns
selection/expansion/backup over its node arena while Python owns boards,
RNG/Dirichlet, and MPS evals; eval vectors cross ONLY via
MpsEvalBridge.evaluate_batch (one call per leaf-batch -- the batch-queue
protocol; NO NN in Rust, now or ever). StubMctTree below remains the
fallback when `mctscore` is not importable (same seam, same bridge).

RUST CONTRACT for the PyO3 entry point (stable):
  - Python seam:  TreeBackend.search(state, evaluate_fn, n_sims, **kw)
                  -> (pi, stats). RustTreeBackend prefers a real `mctscore`
                  module when importable (see RustTreeBackend.search).
  - Rust seam:    ONE Python callback: evaluate_batch(states) ->
                  (priors[B,4096] float64-masked, values[B] float64).
                  Rust collects up to max_batch leaves per crossing, calls
                  once, feeds vectors back in order. Positions cross as
                  (fen: str, rep_key: int, rep_count: int); Rust rebuilds
                  via the chess crate (no Python objects cross).
  - NO NN in Rust (spec): weights/forward/device stay behind the callback.

NO loop integration (V21.1 after training ends): this module is imported
only by tests/test_mcts_parity.py. It never touches loop/train/config/
checkpoints writes, never changes defaults, never launches training.

Timing (measured 2026-09-27, free box, MPS, 400 sims, REAL tree):
 pos0 python 4.94s vs rust 5.01s (1.012), pos1 4.39s vs 4.52s (1.030),
 game 10 plies 56.0s vs 57.9s (1.033). Bridge overhead ~= noise floor;
 tree work small vs NN forwards (win compounds with batching, not alone).
 TT-carryover fix (V21.1 audit F1-F9): multi-ply games persist ONE
 per-game `mctscore.GameStore` (Rust TtStore + eval-cache map) across
 searches — promote at entry, root/child refresh at exit (F1); every
 `prune_tt` site branches to `store.prune_to` when rust on (F2) and the
 Python `tt`/`eval_cache` dicts stay empty there; `clone_subtree` copies
 `o` verbatim (F3); Dirichlet draws iff fresh-expand OR eps>0, `None`
 otherwise (F4); u64 key domain documented here + bridge.rs `py_hash_u64`
 (F5); eval cache persists per game in the store (F6); `live_nodes`
 counts sum per entry vs Python id-dedup (F7, metric only). Gates (F8):
 50-pos exact+noisy zeros plus same-line game-leg (max|dPi|==0 every ply,
 ext + reused equal), independent-game (0 flips), contempt-0.3/asym + ML
 repeat. `--rust-tree` stays default OFF; best.pt read-only (F9).
Parity: exact (TT off + noise 0): max|dPi| == 0.0, visits identical,
 |dQ| == 0.0 over 50 fixed positions (sims=64, seeds 1000+i) -- see the
 harness report. Noisy (TT on + eps 0.25): all zeros too (< 0.02 bound).
U64 KEY DOMAIN (F5): every Rust TT/history/path key is
 `py_hash_u64(rep_key)` — `State.rep_key()` is python-chess
 `transposition_key()` (tuple of ints: board + turn + castling + EP, no
 clocks), hashed with Python `__hash__` and reinterpreted as u64
 (`h as u64`). Tuples of ints hash deterministically in-process, so a
 game-leg is self-consistent; hashes are NOT stable across processes
 (never persist). History/path/TT share the function, so loop detection
 and promotion agree; 64-bit collisions negligible (20k cap). Eval cache
 keys `(hash, rep_count)` (rep planes depend on the count); TT/history
 use the bare hash.
"""
from __future__ import annotations

import time

import chess
import numpy as np

from . import mcts as mcts_mod
from .mcts import (_Node, _backup, _ml_bonus, _root_decided, _select,
                   _select_fpu)
from .game import encode_action, flip_action

BRIDGE = "batch-queue-over-pyo3"  # the documented choice (see module doc)
BRIDGE_VERSION = 1  # bump when the Rust seam signature changes
RUST_MODULE = "mctscore"  # batch C provides rust/mctscore -> this import name


# ---------------------------------------------------------------------------
# The single crossing point: every eval the (stub or Rust) tree needs flows
# through MpsEvalBridge.evaluate_batch. Order-preserving, math-free.
# ---------------------------------------------------------------------------
class MpsEvalBridge:
    """Counts crossings/rows for the timing table; returns vectors untouched."""

    def __init__(self, evaluate_fn, max_batch: int = 8):
        self.inner = evaluate_fn
        self.max_batch = int(max_batch)
        self.crossings = 0
        self.rows = 0
        self.singletons = 0

    def evaluate_batch(self, states):
        states = list(states)
        self.crossings += 1
        self.rows += len(states)
        if len(states) == 1:
            self.singletons += 1
        priors, values = self.inner(states)
        return np.asarray(priors), np.asarray(values, dtype=np.float64)

    def reset_stats(self) -> None:
        self.crossings = 0
        self.rows = 0
        self.singletons = 0

    def stats_dict(self) -> dict:
        return {"crossings": int(self.crossings), "rows": int(self.rows),
                "singletons": int(self.singletons),
                "max_batch": int(self.max_batch)}


# ---------------------------------------------------------------------------
# TreeBackend seam (stable for batch C). Two implementations: pure-Python
# reference (today's mcts.search) and the Rust-tree path (stub now, real
# mctscore when batch C lands it).
# ---------------------------------------------------------------------------
class PythonTreeBackend:
    name = "python-tree"

    def search(self, state, evaluate_fn, n_sims: int, **kw):
        stats: dict = {}
        kw = dict(kw)
        kw["stats"] = stats
        pi = mcts_mod.search(state, evaluate_fn, int(n_sims), **kw)
        return np.asarray(pi, dtype=np.float32), stats


def rust_available() -> bool:
    """True once batch C lands an importable `mctscore` with a `search`
    entry point. The stub stays the fallback (never raises here)."""
    try:
        import importlib as _il
        _m = _il.import_module(RUST_MODULE)
        return bool(hasattr(_m, "search") and getattr(_m, "AUDIT_SEMANTICS", 0) >= 2)
    except Exception:
        return False


def new_game_store():
    """Create ONE per-game Rust-side store (F1/F2/F6).

    Holds the TT across the game's moves plus the per-game eval-cache map.
    Pass as `tt=` (and `eval_cache=`) to every `search` of that game;
    call `store.prune_to(next_rep_key)` after every applied move (mirrors
    `mcts.prune_tt`); `store.clear()` on game/agent reset. The Python
    `tt`/`eval_cache` dicts stay empty on this path (never written).
    Raises ImportError when `mctscore` (with `GameStore`) is missing so
    callers can fall back to plain dicts + the stub/Python tree.
    """
    import importlib as _il
    _m = _il.import_module(RUST_MODULE)
    if getattr(_m, "AUDIT_SEMANTICS", 0) < 2:
        raise ImportError("Rebuild mctscore for corrected search semantics")
    _gs = getattr(_m, "GameStore", None)
    if _gs is None:
        raise ImportError("mctscore.GameStore missing -- rebuild the wheel")
    return _gs()


def is_game_store(obj) -> bool:
    """True when `obj` is a Rust-side per-game `GameStore` (vs dict/None)."""
    try:
        import importlib as _il
        _m = _il.import_module(RUST_MODULE)
        if getattr(_m, "AUDIT_SEMANTICS", 0) < 2:
            return False
        _gs = getattr(_m, "GameStore", None)
        return _gs is not None and isinstance(obj, _gs)
    except Exception:
        return False


class RustTreeBackend:
    """Rust-tree path. Uses the real `mctscore.search` PyO3 entry point
    when importable (V21.1 E1); otherwise the faithful Python stub
    (same seam, same bridge)."""

    name = "rust-tree(stub)"

    def __init__(self, bridge: MpsEvalBridge | None = None,
                 max_batch: int = 8):
        self.bridge = bridge
        self.max_batch = int(max_batch)
        if rust_available():
            self.name = "rust-tree"

    def search(self, state, evaluate_fn, n_sims: int, **kw):
        stats: dict = {}
        kw = dict(kw)
        kw["stats"] = stats
        if rust_available():
            import importlib as _il
            _m = _il.import_module(RUST_MODULE)
            if getattr(_m, "AUDIT_SEMANTICS", 0) < 2:
                raise ImportError("Rebuild mctscore for corrected search semantics")
            # Contract: mctscore.search(state, evaluate_batch, n_sims,
            # **kw) -> (pi[4096], stats-dict) with evaluate_batch ==
            # bridge.evaluate_batch. The real crate owns
            # selection/expansion/backup; Python owns boards/RNG/NN.
            bridge = self.bridge or MpsEvalBridge(evaluate_fn,
                                                  self.max_batch)
            if kw.get("history") is not None and tuple(kw["history"]) != state._hist:
                state = type(state)(state.board.copy(stack=False), _hist=tuple(kw["history"]))
            res = _m.search(state, bridge.evaluate_batch, int(n_sims),
                            **kw)
            pi, rstats = res
            stats.update(dict(rstats))
            stats["bridge"] = bridge.stats_dict()
            return np.asarray(pi, dtype=np.float32), stats
        bridge = self.bridge or MpsEvalBridge(evaluate_fn, self.max_batch)
        pi = mcts_mod.search(state, bridge.evaluate_batch, int(n_sims), **kw)
        stats["bridge"] = bridge.stats_dict()
        return np.asarray(pi, dtype=np.float32), stats


# ---------------------------------------------------------------------------
# StubMctTree: mechanical port of the mcts.py SEQUENTIAL path (leaf_batch=1
# only). Selection helpers are IMPORTED (not copied) from mcts so the math
# is one source of truth; the traversal/expansion/quiescence/backup/tail
# below mirrors search() line-for-line with evals routed via the bridge.
# leaf_batch>1 delegates to proven mcts._search_batched through the same
# bridge (exact by construction; C ports the WU path to Rust in V21.1).
# ---------------------------------------------------------------------------
def _orient(action_abs: int, black_stm: bool) -> int:
    return flip_action(action_abs) if black_stm else action_abs


def _boost_forcing_impl(cur_state, pv, forcing_bonus: float) -> None:
    if not forcing_bonus:
        return
    black = cur_state.board.turn == chess.BLACK
    boosted = False
    bd = cur_state.board
    for cm in cur_state._legal_chess_moves():
        if cur_state.is_capture_fast(cm) or \
                cur_state.gives_check_fast(cm):
            boost = True
        elif cm.promotion is not None:
            boost = True
        else:
            pc = bd.piece_at(cm.from_square)
            if pc is None or pc.piece_type != chess.PAWN:
                boost = False
            elif bd.turn == chess.WHITE:
                boost = chess.square_rank(cm.to_square) in (5, 6)
            else:
                boost = chess.square_rank(cm.to_square) in (2, 1)
        if boost:
            a = _orient(encode_action(cm.from_square, cm.to_square, cm.promotion), black)
            if pv[a] > 0:
                pv[a] *= (1.0 + forcing_bonus)
                boosted = True
    if boosted:
        s = pv.sum()
        if s > 0:
            pv /= s


def _has_promo_threat_impl(bd) -> bool:
    for sq in bd.pieces(chess.PAWN, chess.WHITE):
        if chess.square_rank(sq) == 6:
            return True
    for sq in bd.pieces(chess.PAWN, chess.BLACK):
        if chess.square_rank(sq) == 1:
            return True
    return False


def bridge_search(state, bridge: MpsEvalBridge, n_sims: int,
                  c_puct: float = 1.414,
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
                  fpu_reduction: float = 0.0,
                  prune_singletons: bool = False,
                  eval_cache: dict | None = None,
                  ml_fn=None, ml_slope: float = 0.0,
                  ml_cap: float = 0.07, ml_thr: float = 0.8,
                  futile_stop: bool = False) -> np.ndarray:
    """Stub-tree search behind the bridge. Same contract as mcts.search
    (same defaults, same stats keys, same return). New code, offline use
    only -- never called by the live loop."""
    from .selfplay import draw_leaf_value as _dlv

    def _draw_leaf_value(board) -> float:
        return _dlv(board, contempt, asymmetric_contempt,
                     contempt_edge_scale)

    def _cache_key(s) -> tuple:
        try:
            return (s.rep_key(), s.rep_count())
        except Exception:
            try:
                return (s.key(), 0)
            except Exception:
                return ("unkeyed", 0)

    def _eval_batch(states):
        # Identical B6 caching to mcts.search; misses share ONE bridge
        # crossing (the batch-queue protocol C will implement in Rust).
        if eval_cache is None:
            return bridge.evaluate_batch(list(states))
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
                rows_p[i], rows_v[i] = hit
            else:
                miss_idx.append(i)
        if miss_idx:
            mp, mv = bridge.evaluate_batch([states[i] for i in miss_idx])
            for j, i in enumerate(miss_idx):
                _p = np.asarray(mp[j])
                _v = float(np.asarray(mv[j]).squeeze())
                rows_p[i], rows_v[i] = _p, _v
                try:
                    eval_cache[_cache_key(states[i])] = (_p, _v)
                except Exception:
                    pass
        return (np.stack([np.asarray(r) for r in rows_p]),
                np.asarray(rows_v, dtype=np.float64))

    # Batched-leaf path: proven implementation through bridge evals.
    # (Exact by construction; the stub natively ports leaf_batch=1.)
    _early_box: list = [False]
    if int(leaf_batch) > 1:
        key0 = state.key()
        root0 = tt.get(key0) if tt is not None else None
        reused0 = 0
        if root0 is not None and root0.expanded:
            reused0 = sum(c.n for c in root0.children.values())
        early_fired_b = False
        extensions_b = _search_batched_via_bridge(
            state, _eval_batch, root0, reused0, key0, tt, history,
            _draw_leaf_value, n_sims, c_puct, quiescence_depth,
            forcing_bonus, depths, int(leaf_batch), fpu_reduction,
            _early_box, ml_fn, ml_slope, ml_cap, ml_thr, futile_stop,
            dirichlet_alpha, dirichlet_eps, stats, eval_cache is not None,
            contempt, asymmetric_contempt, contempt_edge_scale,
            prune_singletons, virtual_loss, bridge)
        return extensions_b  # (pi array; tail stats filled inside)

    early_fired = False
    _ml_on = ml_fn is not None and ml_slope != 0.0
    _lm_term = 0.0 if _ml_on else None

    def _ml_single(s):
        _v = float(np.asarray(ml_fn([s])).flatten()[0])
        return _v if np.isfinite(_v) else None

    key = state.key()
    root = tt.get(key) if tt is not None else None
    reused = 0
    if root is not None and root.expanded:
        reused = sum(c.n for c in root.children.values())

    extensions = 0
    legal = state.legal_moves()
    if not legal:
        return np.zeros(4096, dtype=np.float32)

    if root is None or not root.expanded:
        root = _Node()
        priors, _ = _eval_batch([state])
        p = np.asarray(priors[0], dtype=np.float64)
        mask = state.legal_mask()
        p[~mask] = 0.0
        s = p.sum()
        p = p / s if s > 0 else mask.astype(np.float64) / mask.sum()
        _boost_forcing_impl(state, p, forcing_bonus)
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            mixed = (1 - dirichlet_eps) * p[a] + dirichlet_eps * noise[i]
            root.children[a] = _Node(prior=float(mixed), raw=float(p[a]))
        root.expanded = True
    elif dirichlet_eps > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            if a in root.children:
                p0 = root.children[a].raw
                root.children[a].prior = (
                    (1 - dirichlet_eps) * p0 + dirichlet_eps * noise[i])
        for a in list(root.children):
            if a not in legal:
                del root.children[a]
    if tt is not None:
        if len(tt) > 20000:
            for k in list(tt)[:len(tt) - 20000]:
                del tt[k]
        tt[key] = root

    history_set = set(history) if history else set()
    for _ in range(n_sims):
        if futile_stop and _root_decided(root):
            early_fired = True
            break
        node = root
        cur = state
        path = [node]
        actions = []
        path_keys = [state.rep_key()]
        path_set = {path_keys[0]}
        qext = 0
        n_pieces = len(cur.board.piece_map())
        mover = cur.board.turn
        n_pawns_mover = len(cur.board.pieces(chess.PAWN, mover))
        arrived_by_capture = False
        arrived_by_promo = False
        while node.expanded and node.children:
            total = sum(c.n for c in node.children.values())
            best_a, best_child, best_s = None, None, -1e18
            if fpu_reduction != 0.0:
                _pq = node.q
                _red = 0.0 if node is root else fpu_reduction
                for a, ch in node.children.items():
                    sc = _select_fpu(ch, _pq, total, c_puct, _red)
                    if _ml_on:
                        sc += _ml_bonus(ch.m, node.m, _pq, ml_slope,
                                        ml_cap, ml_thr)
                    if sc > best_s:
                        best_s, best_a, best_child = sc, a, ch
            else:
                for a, ch in node.children.items():
                    sc = _select(ch, total, c_puct)
                    if _ml_on:
                        sc += _ml_bonus(ch.m, node.m, node.q, ml_slope,
                                        ml_cap, ml_thr)
                    if sc > best_s:
                        best_s, best_a, best_child = sc, a, ch
            actions.append(best_a)
            cur = cur.apply(best_a)
            new_n_pieces = len(cur.board.piece_map())
            arrived_by_capture = new_n_pieces < n_pieces
            n_pieces = new_n_pieces
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
            if history and (rk in history_set or rk in path_set):
                _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                if depths is not None:
                    depths.append(len(actions))
                break
            path_keys.append(rk)
            path_set.add(rk)
            done, z = cur.is_terminal()
            if done:
                if z == 0.0:
                    _backup(path, -_draw_leaf_value(cur.board), _lm_term)
                else:
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
            leaf_m = _ml_single(cur) if _ml_on else None
            m = cur.legal_mask()
            pv[~m] = 0.0
            tot = pv.sum()
            pv = pv / tot if tot > 0 else m.astype(np.float64) / m.sum()
            _boost_forcing_impl(cur, pv, forcing_bonus)
            for a in cur.legal_moves():
                node.children[a] = _Node(prior=float(pv[a]))
            node.expanded = True
            while (quiescence_depth > 0 and qext < quiescence_depth
                    and (cur.in_check_fast() or arrived_by_capture
                         or arrived_by_promo
                         or _has_promo_threat_impl(cur.board))
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
                if history and (rk in history_set or rk in path_set):
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
                tot = pv.sum()
                pv = pv / tot if tot > 0 else mq.astype(np.float64) / mq.sum()
                _boost_forcing_impl(cur, pv, forcing_bonus)
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
    if prune_singletons:
        visits[visits <= 1] = 0.0
        if visits.sum() == 0:
            for a, ch in root.children.items():
                visits[a] = ch.n
    tot = visits.sum()
    if stats is not None:
        stats["root_visits"] = int(tot)
        stats["reused"] = int(reused)
        stats["new_sims"] = int(n_sims)
        stats["extensions"] = int(extensions)
        stats["early_stop"] = 1 if early_fired else 0
        _lv = visits[visits > 0]
        stats["breadth"] = int(_lv.shape[0])
        if tot > 0 and _lv.shape[0]:
            _pv = _lv.astype(np.float64) / float(tot)
            stats["visit_entropy"] = float(-(np.log(_pv) * _pv).sum())
        else:
            stats["visit_entropy"] = 0.0
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
    if tt is not None:
        for a, ch in root.children.items():
            if ch.expanded and ch.n > 0:
                try:
                    tt[state.apply(a).rep_key()] = ch
                except Exception:
                    pass
    return visits


def _search_batched_via_bridge(state, eval_batch_fn, root, reused, key,
                               tt, history, dlv, n_sims, c_puct,
                               quiescence_depth, forcing_bonus, depths,
                               leaf_batch, fpu_reduction, early_box,
                               ml_fn, ml_slope, ml_cap, ml_thr,
                               futile_stop, dirichlet_alpha,
                               dirichlet_eps, stats, _has_cache,
                               contempt, asymmetric_contempt,
                               contempt_edge_scale, prune_singletons,
                               virtual_loss, bridge):
    """leaf_batch>1 through the proven mcts._search_batched, with evals
    routed via the bridge. Root expand/reuse + tail mirror mcts.search
    exactly (same RNG calls, same TT writes, same stats keys)."""
    import numpy as _np
    legal = state.legal_moves()
    if not legal:
        if stats is not None:
            for _k, _v in (("root_visits", 0), ("reused", int(reused)),
                           ("new_sims", int(n_sims)), ("extensions", 0),
                           ("early_stop", 0), ("breadth", 0),
                           ("visit_entropy", 0.0), ("kl", 0.0),
                           ("root_q", 0.0)):
                stats[_k] = _v
        return _np.zeros(4096, dtype=_np.float32)
    if root is None or not root.expanded:
        root = _Node()
        priors, _ = eval_batch_fn([state])
        p = _np.asarray(priors[0], dtype=_np.float64)
        mask = state.legal_mask()
        p[~mask] = 0.0
        s = p.sum()
        p = p / s if s > 0 else mask.astype(_np.float64) / mask.sum()
        _boost_forcing_impl(state, p, forcing_bonus)
        noise = _np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            mixed = (1 - dirichlet_eps) * p[a] + dirichlet_eps * noise[i]
            root.children[a] = _Node(prior=float(mixed), raw=float(p[a]))
        root.expanded = True
    elif dirichlet_eps > 0:
        noise = _np.random.dirichlet([dirichlet_alpha] * len(legal))
        for i, a in enumerate(legal):
            if a in root.children:
                p0 = root.children[a].raw
                root.children[a].prior = (
                    (1 - dirichlet_eps) * p0 + dirichlet_eps * noise[i])
        for a in list(root.children):
            if a not in legal:
                del root.children[a]
    if tt is not None:
        if len(tt) > 20000:
            for k in list(tt)[:len(tt) - 20000]:
                del tt[k]
        tt[key] = root
    history_set = set(history) if history else set()
    # mcts._search_batched calls boost_fn(cur, pv) with 2 args (its own
    # closure captures the bonus); wrap ours identically (stub path only).
    _boost2 = lambda _cur, _pv, _fb=forcing_bonus: _boost_forcing_impl(
        _cur, _pv, _fb)  # noqa: E731
    extensions = mcts_mod._search_batched(
        root, state, eval_batch_fn, int(n_sims), c_puct, quiescence_depth,
        forcing_bonus, history, history_set, dlv, _orient,
        _boost2, _has_promo_threat_impl, depths,
        int(leaf_batch), virtual_loss, fpu_reduction, early_box,
        ml_fn, ml_slope, ml_cap, ml_thr, futile_stop)
    early_fired = bool(early_box[0])
    visits = _np.zeros(4096, dtype=_np.float32)
    for a, ch in root.children.items():
        visits[a] = ch.n
    if prune_singletons:
        visits[visits <= 1] = 0.0
        if visits.sum() == 0:
            for a, ch in root.children.items():
                visits[a] = ch.n
    tot = visits.sum()
    if stats is not None:
        stats["root_visits"] = int(tot)
        stats["reused"] = int(reused)
        stats["new_sims"] = int(n_sims)
        stats["extensions"] = int(extensions)
        stats["early_stop"] = 1 if early_fired else 0
        _lv = visits[visits > 0]
        stats["breadth"] = int(_lv.shape[0])
        if tot > 0 and _lv.shape[0]:
            _pv = _lv.astype(_np.float64) / float(tot)
            stats["visit_entropy"] = float(-(_np.log(_pv) * _pv).sum())
        else:
            stats["visit_entropy"] = 0.0
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
                if _rsum > 0 and _np.isfinite(_rsum):
                    for a, _r in zip(legal, _raws):
                        _v = float(visits[a]) / float(tot)
                        _p = _r / _rsum
                        if _v > 0.0 and _p > 0.0:
                            _kl += _v * float(_np.log(_v / _p))
                    if not _np.isfinite(_kl):
                        _kl = 0.0
        except Exception:
            _kl = 0.0
        stats["kl"] = float(_kl)
        try:
            _rq = 0.0
            if tot > 0:
                _rq = float(sum(float(visits[a]) * float(
                    root.children[a].q) for a in root.children
                    if float(visits[a]) > 0.0) / float(tot))
            stats["root_q"] = float(_rq) if _np.isfinite(_rq) else 0.0
        except Exception:
            stats["root_q"] = 0.0
    if tot > 0:
        visits /= tot
    if tt is not None:
        for a, ch in root.children.items():
            if ch.expanded and ch.n > 0:
                try:
                    tt[state.apply(a).rep_key()] = ch
                except Exception:
                    pass
    return visits


def timed_search(backend, state, evaluate_fn, n_sims: int, **kw):
    """One timed search: returns (pi, stats, wall_seconds). Offline only."""
    t0 = time.time()
    pi, stats = backend.search(state, evaluate_fn, int(n_sims), **kw)
    return pi, stats, time.time() - t0
