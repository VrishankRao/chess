"""Original self-play loop: MCTS + temperature -> (s, pi, z) triples."""
from __future__ import annotations

import numpy as np
import torch

from . import mcts as mcts_mod
from .game import State

# D3: server/evaluate_fn path (model None) with ml_slope != 0 logs the
# ML-off fallback exactly once per process (not once per game).
_ML_OFF_LOGGED = False


def _move_is_progress(state, action) -> bool:
    """D4 progress-move probe: did the move played from `state` reset the
    no-progress clock (pawn move or capture, python-chess truth via move
    matching — promotion-captures and en passant included)? Mate/terminal
    handling is the caller's (any terminal after the move ends the wait,
    flagged there). Never raises (target-side helper)."""
    try:
        import chess as _cc
        from .game import encode_action as _enc, flip_action as _flip
        _black = state.side_to_move == 1
        for _cm in state._legal_chess_moves():
            _a = _enc(_cm.from_square, _cm.to_square, _cm.promotion)
            if _black:
                _a = _flip(_a)
            if _a != int(action):
                continue
            try:
                _pc = state.board.piece_at(_cm.from_square)
                if _pc is not None and _pc.piece_type == _cc.PAWN:
                    return True
            except Exception:
                pass
            try:
                if state.board.is_capture(_cm):
                    return True
            except Exception:
                pass
            return False
    except Exception:
        pass
    return False


def _progress_targets(use, pflags, final_ply) -> list:
    """D4 plies-until-progress targets over the KEPT prefix: for position i,
    j = first index >= i whose recorded move reset the clock (pawn/capture;
    mate/any-terminal flags set by the caller end the wait too):
    P[i] = ply[j]-ply[i]+1 (the progress move itself costs one ply, so an
    immediate pawn push reads 1.0, not 0.0). No future progress ->
    P[i] = final_ply-ply[i] (censored remaining = ml_target*cap, D4).
    pflags None/length-mismatched (hand-built hists, legacy callers) ->
    all-censored. Always defined, never -1/ignore (D4)."""
    n = len(use)
    if not pflags or len(pflags) != n:
        pflags = [False] * n
    out = [0.0] * n
    _nxt = None  # ply of the first progress move at/after i
    for _i in range(n - 1, -1, -1):
        try:
            _ply = int(use[_i][4])
        except Exception:
            _ply = 0
        if pflags[_i]:
            _nxt = _ply
        if _nxt is None:
            try:
                out[_i] = float(max(0, int(final_ply) - _ply))
            except Exception:
                out[_i] = 0.0
        else:
            out[_i] = float(max(1, _nxt - _ply + 1))
    return out


def make_evaluate(model, device="cpu", jit=True):
    """Build a batched evaluate_fn(states) -> (priors, values) closure.

    jit (phase 1, v8.7+): freeze a scripted copy for CPU inference workers
    (~2.3x on batch-1: BN folding; measured IMPROVEMENTS.md §6.9). The
    passed-in model is NEVER mutated or frozen — only a local scripted
    copy is frozen, so the training model stays eager/trainable and
    train_step is unaffected. Env kill switch CHESS_ZERO_NO_JIT=1 forces
    eager (bisecting). Any freeze failure degrades to eager with a log
    line, never breaks a run. NOTE: frozen inference is mathematically
    equivalent but not bit-identical (~1e-6); MCTS amplifies ULP diffs
    into different games, so this is an environment change — do not
    compare pre/post arena numbers as same-engine evidence.
    """
    import os as _os
    model.to(device)
    model.eval()
    _ev_model = model
    if (jit and device == "cpu"
            and _os.environ.get("CHESS_ZERO_NO_JIT") != "1"):
        # Freeze AFTER .to()/.eval(): a frozen ScriptModule rejects .to().
        # Guard on cpu: loop.run_training builds the training model on GPU
        # and calls make_evaluate on it in the sequential path — freezing
        # that would break train_step's backward().
        try:
            _ev_model = torch.jit.freeze(torch.jit.script(model).eval(), preserved_attrs=["forward_inf"])
        except Exception as e:  # never break a run for speed
            print(f"[jit] freeze unavailable ({type(e).__name__}: "
                  f"{str(e)[:100]}); running eager", flush=True)

    _last = {}
    def fn(states: list[State]):
        import numpy as np
        import torch
        x = np.stack([s.encode(planes=model.trunk_in.in_channels) for s in states]).astype(np.float32)
        with torch.no_grad():
            # v16: index, don't unpack — old 7-head weights and new
            # 8-head nets must both serve (mid-run code/weight skew
            # would otherwise crash workers on the first new phase).
            # v18: out[1] is WDL logits — search reads Q = P(W)-P(L).
            out = _ev_model.forward_inf(torch.from_numpy(x).to(device)) if hasattr(_ev_model, "forward_inf") else _ev_model(torch.from_numpy(x).to(device))
            logits, wdl = out[0], out[1]
            _last.clear()
            _last.update({(s.key(), s._hist): float(out[3][i].detach().cpu()) for i, s in enumerate(states)})
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            if wdl.shape[1] == 1:
                # legacy scalar-value net served directly (never happens
                # through load_weights surgery, which grows v_fc2 1->3).
                vals = wdl.cpu().numpy().flatten()
            else:
                _pw = torch.softmax(wdl, dim=1).cpu().numpy()
                vals = (_pw[:, 0] - _pw[:, 2]).astype(np.float32)
        # mask illegal here as well (search re-masks; harmless)
        for i, s in enumerate(states):
            m = s.legal_mask()
            probs[i][~m] = 0.0
            tot = probs[i].sum()
            probs[i] = probs[i] / tot if tot > 0 else m.astype(np.float32) / m.sum()
        return probs, vals

    def ml_fn(states):
        if any((s.key(), s._hist) not in _last for s in states):
            fn(states)
        return np.array([_last[(s.key(), s._hist)] for s in states])
    fn.ml_fn = ml_fn
    model._audit_ml_fn = ml_fn
    return fn


def hangs_material(state, action, threshold: float = 0.09) -> bool:
    """True if `action` hangs material: the opponent has a reply whose NET
    gain (minus OUR best recapture — 2-ply, symmetric with tactical_action)
    is >= threshold (own fitted-table pawns). A pure 1-ply scan would cry
    wolf on every defended pawn (the poisoned-pawn trap in reverse), so a
    reply only counts if it survives our response, exactly as our captures
    must survive theirs. Mate-delivering moves never hang; moving into a
    terminal draw is not a material hang (contempt's department, not this).
    Allowing mate counts as the maximal hang. Early-exits on first proof.
    Threshold reuses tac_threshold by doctrine: we force captures gaining
    >= t and veto moves losing >= t — symmetric tactics."""
    import chess_zero.game as _g
    import chess as _cc
    _vals = _g.ADJUDICATE_VALUES

    def edge(board):
        d = sum(v * (len(board.pieces(pt, _cc.WHITE))
                     - len(board.pieces(pt, _cc.BLACK)))
                for pt, v in _vals.items())
        return -d / 10.0 if board.turn == _cc.BLACK else d / 10.0

    def reply_net(nxt_state):
        """Best net gain for the side to move in nxt_state (their view),
        mates included as infinite. Shared by both reply layers below."""
        rb = edge(nxt_state.board)
        best = 0.0
        for b in nxt_state.legal_moves():
            after = nxt_state.apply(b)
            rdone, rz = after.is_terminal()
            if rdone:
                if rz == -1.0:
                    return float("inf")  # they mate: unbounded gain
                continue
            g = -edge(after.board) - rb
            if g > best:
                best = g
        return best
    nxt = state.apply(action)
    done, _z = nxt.is_terminal()
    if done:
        # Mate delivered (opponent mated): keep unconditionally — the
        # terminal check below runs before any veto. Anything else
        # terminal here means the game already ended by our move.
        return False
    # Their reply must survive OUR recapture to count (2-ply net).
    for b in nxt.legal_moves():
        after = nxt.apply(b)
        rdone, rz = after.is_terminal()
        if rdone:
            if rz == -1.0:
                return True  # we allow mate: maximal hang, always veto
            continue  # their stalemating capture: not a material hang
        gain = -edge(after.board) - edge(nxt.board)
        if gain < threshold:
            continue
        ours = reply_net(after)
        if ours == float("inf"):
            continue  # we mate back: their "winning" capture is poisoned
        if gain - ours >= threshold:
            return True
    return False


def veto_blunder(state, choice: int, pi, threshold: float = 0.09):
    """Returns (action, vetoed). If MCTS's `choice` hangs material,
    replace it with the highest-visit move that doesn't (visits order =
    pi descending). If EVERYTHING hangs (lost anyway), keep the original
    — never force passivity. Quiet choices pass through after one cheap
    opponent scan; only suspected hangs pay for the full 2-ply check."""
    if not hangs_material(state, choice, threshold):
        return choice, False
    for a in sorted(state.legal_moves(), key=lambda x: -pi[x]):
        if a == choice:
            continue
        if not hangs_material(state, a, threshold):
            return a, True
    return choice, False


def apply_move_guards(state, choice: int, pi, tac, veto_on: bool,
                      veto_thr: float = 0.09):
    """Single choke point for training-identical move guards (v9 product
    doctrine: self-play, arena, gate and deployment all play the GUARDED
    policy, so films measure the product, not an unguarded shadow of it).
    - tac (forced tactic from tactical_action) wins outright, one-hot.
    - else veto_blunder replaces hanging picks (one-hot on replace).
    - else the choice stands with its visits distribution untouched.
    Returns (action, pi_out, vetoed). Callers sample/argmax FIRST (their
    temperature regimes differ), then pass the choice here. Pure function
    of its inputs — the unit-testable core of every move decision."""
    import numpy as _np
    if tac is not None:
        out = _np.zeros(4096, dtype=np.float32)
        out[int(tac)] = 1.0
        return int(tac), out, False
    if veto_on:
        new_choice, vetoed = veto_blunder(state, int(choice), pi, veto_thr)
        if vetoed:
            out = _np.zeros(4096, dtype=np.float32)
            out[int(new_choice)] = 1.0
            return int(new_choice), out, True
    return int(choice), pi, False


# ---------------------------------------------------------------------------
# V23 A3 (audit 20: guard targets; audit 22: fused ML callback).
# Play doctrine UNCHANGED (apply_move_guards still picks the guarded move;
# films measure the product). What changes is the TRAINING TARGET plus
# side-channel logging:
# - the played move stays the guard move (one-hot play policy);
# - the stored pi target is BLENDED: (1-w)*raw_visits + w*one_hot(guard),
#   with mate/legality (rules-derived) stronger than learned safety/ML
#   (learned heads get weak weight);
# - raw visits / played move / reason / confidence are kept separately in
#   the per-game stats side-channel (never in the replay tuple: arity
#   stays 18, arch FROZEN).
# Inference-fusion boundary with batch A4 (no duplication): A4 owns the
# fused forward (ONE forward per leaf batch: policy+WDL+safety+ML). A3
# CONSUMES a fused interface when present (model.forward_inf, or a future
# chess_zero.inference_fusion.make_ml_fn if A4 lands it) and otherwise
# wires the ML side-channel through the existing out[3] path. A3 never
# builds its own fused evaluate; search/evaluate keep their contracts.
# ---------------------------------------------------------------------------

# V23 A3 audit-20 blend weights (mass placed on the guard move; remainder
# stays on the raw visit distribution). Mate/tactical (forced mate or
# 2-ply sound capture, rules-derived) and veto (2-ply hanging-material,
# rules-derived) are STRONG; safety (learned king-safety head) and finish
# (learned moves-left head) are WEAK by doctrine: learned preferences must
# not overwrite search with maximal confidence. "none" = no guard fired.
GUARD_BLEND: dict = {
    "tactical": 1.0,
    "veto": 0.9,
    "safety": 0.25,
    "finish": 0.25,
    "none": 0.0,
}

# Reasons that count as rules-derived (strong) vs learned (weak). Mate
# delivery inside "tactical" is the strongest: it stays one-hot.
STRONG_GUARD_REASONS = ("tactical", "veto")


def blend_guard_target(raw_pi, guard_action: int, reason: str):
    """Blended corrective training target (audit 20).

    raw_pi: visit distribution BEFORE guards (sums to ~1). guard_action:
    the played (guarded) move. reason: one of tactical/veto/safety/
    finish/none. Returns a NEW float32[4096] distribution:
    (1-w)*raw + w*one_hot(guard) with w from GUARD_BLEND. w=1.0 (mate/
    tactical) is bit-identical to yesterday's one-hot; weak heads keep
    most of the search distribution. Never raises (error -> raw copy)."""
    import numpy as _np
    try:
        _raw = _np.asarray(raw_pi, dtype=_np.float64).flatten()
        if _raw.shape[0] != 4096:
            return _np.asarray(raw_pi, dtype=np.float32).copy()
        _w = float(GUARD_BLEND.get(str(reason), 0.0))
        _w = min(max(_w, 0.0), 1.0)
        if _w == 0.0:
            return _raw.astype(np.float32)
        _t = (1.0 - _w) * _raw
        _t[int(guard_action)] = float(_t[int(guard_action)]) + _w
        _s = float(_t.sum())
        if _s > 0 and np.isfinite(_s):
            _t = _t / _s
        return _t.astype(np.float32)
    except Exception:
        try:
            return np.asarray(raw_pi, dtype=np.float32).copy()
        except Exception:
            _z = np.zeros(4096, dtype=np.float32)
            try:
                _z[int(guard_action)] = 1.0
            except Exception:
                pass
            return _z


def make_guard_info(reason: str, raw_pi, played: int, n_visits: int = 0):
    """Raw guard record kept SEPARATELY from the blended target (audit 20):
    {reason, played, confidence (= blend weight), raw_top, raw_mass,
    n_visits}. raw_top = argmax of the pre-guard visits; raw_mass = raw
    probability the search itself gave the played move. Never raises."""
    import numpy as _np
    try:
        _raw = _np.asarray(raw_pi, dtype=np.float64).flatten()
        _top = int(_np.argmax(_raw)) if _raw.shape[0] == 4096 else int(played)
        try:
            _mass = float(_raw[int(played)]) if _raw.shape[0] == 4096 else 0.0
        except Exception:
            _mass = 0.0
    except Exception:
        _top, _mass = int(played), 0.0
    try:
        _conf = float(GUARD_BLEND.get(str(reason), 0.0))
    except Exception:
        _conf = 0.0
    return {"reason": str(reason), "played": int(played),
            "confidence": round(min(max(_conf, 0.0), 1.0), 4),
            "raw_top": int(_top), "raw_mass": round(float(_mass), 4),
            "n_visits": int(n_visits or 0)}


def apply_move_guards_ex(state, choice: int, pi, tac, veto_on: bool,
                         veto_thr: float = 0.09):
    """Guarded PLAY + blended TRAINING target + raw record (audit 20).

    Play decision is apply_move_guards bit-exact (guarded move wins for
    play; films measure the product). Additionally returns the blended
    training target (blend_guard_target over the PRE-guard visits) and a
    GuardInfo dict {reason, played, confidence, raw_top, raw_mass}.
    reason: tactical > veto > none (safety/finish compose downstream in
    the caller with their own weak reasons; this helper covers the
    tac/veto stage only). Returns (action, pi_play, pi_target, info).
    Never raises (error -> passthrough with reason 'none')."""
    import numpy as _np
    try:
        _raw = _np.asarray(pi, dtype=np.float32).copy()
    except Exception:
        _raw = pi
    try:
        _act, _play, _vetoed = apply_move_guards(
            state, int(choice), pi, tac, bool(veto_on), float(veto_thr))
    except Exception:
        return int(choice), pi, pi, make_guard_info("none", pi, int(choice))
    if tac is not None:
        _reason = "tactical"
    elif bool(_vetoed):
        _reason = "veto"
    else:
        _reason = "none"
    try:
        _n = int(_np.asarray(pi).sum() * 0 + 0)  # visits live in search stats
    except Exception:
        _n = 0
    _tgt = blend_guard_target(_raw, int(_act), _reason)
    return int(_act), _play, _tgt, make_guard_info(_reason, _raw, int(_act), _n)


def resolve_ml_fn(model, device="cpu"):
    """One shared ML side-channel constructor (audit 22, A3/A4 boundary).

    Returns states -> normalized moves-left predictions, or None when the
    model is None (server path: bonus off, logged once by the caller).
    Consumes the fused interface when present: model.forward_inf (ONE
    fused op for policy+WDL+safety+ML; here only the ML row is read, so
    no second full forward is spent on ML). Otherwise falls back to the
    existing full-forward out[3] path (yesterday bit-exact values:
    forward_inf uses the same modules in the same order). If batch A4
    later lands chess_zero.inference_fusion.make_ml_fn, that constructor
    wins when importable (no duplication: A3 never builds its own fused
    evaluate). Never raises (error -> None, bonus off)."""
    try:
        if model is None:
            return None
        if hasattr(model, "_audit_ml_fn"):
            return model._audit_ml_fn
        try:
            from .inference_fusion import make_ml_fn as _a4  # A4 owns this
            try:
                return _a4(model, device)
            except Exception:
                pass  # fall through to the local fused/single paths below
        except Exception:
            pass
        if hasattr(model, "forward_inf"):
            _m, _d = model, device

            def _fused_ml(states, _mm=_m, _dd=_d):
                import torch as _tm
                import numpy as _nm
                _x = _tm.stack(
                    [_tm.from_numpy(_nm.asarray(
                        s.encode(), dtype=_nm.float32))
                     for s in states]).to(_dd)
                with _tm.no_grad():
                    _out = _mm.forward_inf(_x)
                return _nm.asarray(_out[3].detach().cpu(),
                                   dtype=_nm.float64).flatten()
            return _fused_ml
        _m2, _d2 = model, device

        def _single_ml(states, _mm=_m2, _dd=_d2):
            import torch as _tm2
            import numpy as _nm2
            _x = _tm2.stack(
                [_tm2.from_numpy(_nm2.asarray(
                    s.encode(), dtype=_nm2.float32))
                 for s in states]).to(_dd)
            with _tm2.no_grad():
                _out = _mm(_x)
            return _nm2.asarray(_out[3].detach().cpu(),
                                dtype=_nm2.float64).flatten()
        return _single_ml
    except Exception:
        return None


def guardfree_tactics_score(raw_policy_fn, puzzles: list) -> dict:
    """Guard-FREE tactics metric (audit 20): accuracy of the RAW policy
    (no tac/veto/safety/finish) on tactical puzzles, measured alongside
    guarded playing strength so guard dependence is visible. puzzles:
    [(fen, expected_uci), ...]; a puzzle counts correct when the raw
    policy's move UCI equals expected (legality is checked: illegal raw
    picks count wrong, never raise). Returns {n, correct, accuracy,
    illegal}. Never raises (bad FEN -> counted wrong + noted)."""
    _n = _ok = _ill = 0
    _bad = 0
    try:
        from .game import State as _S
        import chess as _c
        for _fen, _exp in (puzzles or []):
            try:
                _st = _S(_c.Board(str(_fen)))
            except Exception:
                _n += 1
                _bad += 1
                continue
            _n += 1
            try:
                _a = int(raw_policy_fn(_st))
                _mv = _st.to_move(_a)
                _legal = _mv in _st.board.legal_moves
                if not _legal:
                    _ill += 1
                    continue
                if _mv.uci() == str(_exp):
                    _ok += 1
            except Exception:
                continue
    except Exception:
        pass
    _acc = round(float(_ok) / float(_n), 4) if _n else 0.0
    return {"n": int(_n), "correct": int(_ok), "accuracy": float(_acc),
            "illegal": int(_ill), "bad_fen": int(_bad)}


def tactical_action(state, threshold: float = 0.09):
    """2-ply tactical scan (v8.3): find our capture whose NET gain survives
    the opponent's best immediate reply. Returns the action or None.
    This is what teaches "recapture after a trade" into the policy weights:
    MCTS at 40 sims routinely never visits the recapture, so the target
    must carry the lesson instead. Pure 1-ply would force poisoned pawns;
    the reply scan rejects anything hanging more than it wins.
    Mate always fires; captures that draw on the spot (stalemate) never do.
    SCORING (audit round 2): net measured with the project's own
    ADJUDICATE_VALUES table (live: fitted from results, starts classical) —
    correct piece ranking with zero human table on this path. The unit-count
    version declined Q-for-pawn (every recapture cancels every capture);
    the classical table was the original documentation contradiction.
    Threshold ~= one pawn-equivalent of net gain."""
    import chess_zero.game as _g
    import chess as _cc
    _vals = _g.ADJUDICATE_VALUES

    def edge(board):
        # stm-relative edge for the side ABOUT TO MOVE, in pawns.
        d = sum(v * (len(board.pieces(pt, _cc.WHITE))
                     - len(board.pieces(pt, _cc.BLACK)))
                for pt, v in _vals.items())
        return -d / 10.0 if board.turn == _cc.BLACK else d / 10.0
    before = edge(state.board)
    best, best_net = None, threshold
    for a in state.legal_moves():
        nxt = state.apply(a)
        done, z = nxt.is_terminal()
        if done:
            if z == -1.0:  # side to move (opponent) is mated: always take
                return a
            continue  # stalemate-by-capture and friends: never force
        # nxt is the OPPONENT's turn: flip back to our view for the gain.
        gain = -edge(nxt.board) - before
        if gain < threshold:
            continue
        # opponent's best immediate reply gain (their stm view)
        reply_best = 0.0
        rbefore = edge(nxt.board)
        for b in nxt.legal_moves():
            after = nxt.apply(b)
            rdone, rz = after.is_terminal()
            if rdone:
                if rz == -1.0:
                    reply_best = float("inf")  # they mate us: poisoned
                    break
                continue
            rg = -edge(after.board) - rbefore
            if rg > reply_best:
                reply_best = rg
        if reply_best == float("inf"):
            continue
        if gain - reply_best > best_net:
            best, best_net = a, gain - reply_best
    return best


def draw_leaf_value(board, contempt: float = 0.0,
                    asymmetric: bool = False,
                    edge_scale: float = 0.0) -> float:
    """Draw score from the side-to-move LEAF view (v9 search contempt).

    Replaces label contempt (v5-v8 rewrote drawn-game value TARGETS with
    ±contempt, teaching the value head that draws equal wins/losses and
    collapsing it — §5.4). Targets now stay pure 0.0; contempt lives only
    in search backups, where it steers move choice without corrupting
    what the value head regresses on.
    - asymmetric (v9): ahead side reads a draw as failure (-c), behind
      side as achievement (+c), level as 0. Uses the live fitted table
      via adjudicated_edge_stm.
    - symmetric legacy: the mover pays -c (mover view), so the leaf view
      is +c unconditionally.
    - edge_scale (v11): magnitude scales with the material edge —
      mag = c * min(1, |edge|/scale), so level draws stay 0 and won
      positions hate draws at full strength. This fixes the v10-film
      failure mode: adaptive steering had relaxed contempt to its 0.1
      floor on globally-rare draws, leaving nothing to avoid a draw in a
      specifically WON endgame. 0.0 (default) = legacy fixed magnitude.
    contempt=0.0 returns exactly 0.0 (yesterday's behaviour, bit for bit).
    """
    if not contempt:
        return 0.0
    from chess_zero.game import adjudicated_edge_stm
    edge = adjudicated_edge_stm(board)
    mag = contempt
    if edge_scale and edge_scale > 0:
        mag = contempt * min(1.0, abs(edge) / edge_scale)
    if asymmetric:
        if edge > 0:
            return -mag
        if edge < 0:
            return +mag
        return 0.0
    return +mag


def ownership_map(final_board, stm: int):
    """(8,8) stm-relative final occupancy: +1 ours / -1 theirs / 0 empty
    (v10 dense supervision). Same end-state for every position of the game,
    flipped per position's own side to move — the trunk learns where pieces
    belong, not just what the result was."""
    import numpy as _np
    import chess as _cc
    ours = _cc.WHITE if stm == 0 else _cc.BLACK
    t = _np.zeros((8, 8), dtype=_np.float32)
    for sq, p in final_board.piece_map().items():
        r, c = 7 - _cc.square_rank(sq), _cc.square_file(sq)
        t[r, c] = 1.0 if p.color == ours else -1.0
    if stm == 1:
        # v9 flip / v12 fix: when Black is to move, State.encode rotates the
        # board tensor 180 degrees (t[:, ::-1, ::-1]) so "forward" is up.
        # The 1x1 conv ownership head is spatially aligned with that tensor,
        # so its spatial target must undergo the exact same 180-degree rotation!
        t = _np.ascontiguousarray(t[::-1, ::-1])
    return t


def final_margin(final_board, stm: int = 0) -> float:
    """Side-to-move relative final material in classical pawns/10 (v10, v12 fix).
    Positive = side to move has more final material. Preserves canonical orientation
    in alignment with value head, aux material head, and policy head (no white-vs-black
    polarity conflict in the shared trunk)."""
    import chess as _cc
    from chess_zero.game import PIECE_VALUES
    ours = _cc.WHITE if stm == 0 else _cc.BLACK
    theirs = _cc.BLACK if stm == 0 else _cc.WHITE
    diff = sum(v * (len(final_board.pieces(pt, ours))
                    - len(final_board.pieces(pt, theirs)))
               for pt, v in PIECE_VALUES.items())
    return diff / 10.0


def king_safety(final_board, stm: int = 0) -> float:
    """v16: stm-relative king-safety scalar in [0,1] (sigmoid target).
    1.0 = our king safe and theirs in danger; 0.0 = reverse. Danger of
    a king = enemy attacks on its zone (king square + neighbors) / 6,
    capped at 1. Rules-derived dense supervision for the 0/8 castling
    disease and back-rank mates (mated final boards read ~0.0 for the
    mated side). Same end-state teaching as ownership/margin: every
    position of a game learns from where the kings ended up."""
    import chess as _cc
    ours = _cc.WHITE if stm == 0 else _cc.BLACK
    theirs = _cc.BLACK if stm == 0 else _cc.WHITE

    def _danger(king_color, by_color):
        # Enemy attacks on the king zone, counting PIECES (not the
        # enemy king — proximity is opposition, not danger; including
        # it washed out KQK-style mates where kings stand close).
        ks = final_board.king(king_color)
        if ks is None:
            return 0.0
        zone = [ks] + [s for s in _cc.SQUARES
                       if _cc.square_distance(s, ks) == 1]
        hits = 0
        for s in zone:
            for a in final_board.attackers(by_color, s):
                if final_board.piece_at(a).piece_type != _cc.KING:
                    hits += 1
                    break
        return min(1.0, hits / 6.0)

    d_ours = _danger(ours, theirs)
    d_theirs = _danger(theirs, ours)
    return min(1.0, max(0.0, 0.5 + (d_theirs - d_ours) / 2.0))


# ---------------------------------------------------------------------------
# V20 batch-B builders (Agent B). Tuple arity is FROZEN at 18 (F9):
# shape_targets / build_resign_examples append D1/D2/D3 contract values
# with EXACT field names score_mean(14), score_stdev(15), root_q(16),
# av(17, None: Agent C labels offline) — matching replay.py TUPLE_ARITY
# and train.py's 16-tuple (KL LAST). Side-channel stats use the same
# EXACT names (root_q mean, score_mean/stdev, av_logits None).
# ---------------------------------------------------------------------------

def load_endgame_fens(path: str, partition="train") -> list:
    """D4: load endgame start FENs (one JSON {fen} per line, or bare FEN
    lines). Invalid lines skipped with a count (loud when empty). The
    halfmove-clock randomization (20% rows 0-60) lives in build_endgames
    output, not here. Never raises (missing file -> [] + log)."""
    out: list = []
    _bad = 0
    try:
        with open(path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _fen = _line
                if _line.startswith("{"):
                    try:
                        import json as _j
                        _fen = str(_j.loads(_line).get("fen", "") or "")
                    except Exception:
                        _bad += 1
                        continue
                if not _fen:
                    _bad += 1
                    continue
                try:
                    import chess as _c
                    board = _c.Board(_fen)
                    if not board.is_valid() or board.is_game_over():
                        raise ValueError("Invalid or terminal curriculum position")
                    import hashlib
                    # Ignore clocks and side/color mirroring for split identity.
                    keys = [" ".join(b.fen().split()[:4]) for b in (board, board.mirror())]
                    bucket = int(hashlib.sha256(min(keys).encode()).hexdigest()[:8], 16) % 5
                    if partition == "all" or ((bucket == 0) == (partition == "eval")):
                        out.append(_fen)
                except Exception:
                    _bad += 1
    except Exception as _e:
        print(f"[endgame] start list unreadable ({path}: "
              f"{type(_e).__name__}); starts OFF", flush=True)
        return []
    if not out:
        print(f"[endgame] no usable FENs in {path} (bad={_bad}); "
              f"starts OFF", flush=True)
    else:
        print(f"[endgame] {len(out)} start FENs from {path} "
              f"(bad={_bad})", flush=True)
    return out


def score_targets_for(final_board) -> dict:
    """D1 score-margin targets with EXACT names score_mean/score_stdev.
    score_mean = final white-relative piece-value diff in pawns (simple
    sum over PIECE_VALUES, white minus black). score_stdev = 0.0: a single
    terminal observation carries no dispersion (Agent A models noise /
    probe-gates the head; weight starts 0.0, NEVER in search/veto in V20).
    Never raises."""
    try:
        import chess as _cc
        from chess_zero.game import PIECE_VALUES as _PV
        _d = sum(float(v) * (len(final_board.pieces(_pt, _cc.WHITE))
                             - len(final_board.pieces(_pt, _cc.BLACK)))
                 for _pt, v in _PV.items())
        _mean = float(_d)
        if not np.isfinite(_mean):
            _mean = 0.0
    except Exception:
        _mean = 0.0
    return {"score_mean": round(_mean, 3), "score_stdev": 0.0}


def td_blend(z_final: float, root_q: float, td_lambda: float = 0.5) -> float:
    """D2 short-horizon TD blend: td_value = l*z_final + (1-l)*q_search.
    Pure helper (Agent A wires it into the WDL loss as 0.5*z + 0.5*td);
    loop/selfplay only LOG root_q. Never raises."""
    try:
        _l = min(max(float(td_lambda), 0.0), 1.0)
        _v = _l * float(z_final) + (1.0 - _l) * float(root_q)
        return float(_v) if np.isfinite(_v) else float(z_final)
    except Exception:
        try:
            return float(z_final)
        except Exception:
            return 0.0


def smart_resign_read(model, device, state):
    """D5 smart-resign sensor: one no-grad forward returning (W, ml_pred)
    with EXACT semantics W = P(win) from the WDL head (index 0), ml_pred =
    normalized moves-left head in [0,1]. (None, None) when model is None
    (server path: resign checks SKIP, game plays on) or on any error.
    Never raises."""
    try:
        if model is None:
            return None, None
        import torch as _t
        import numpy as _np
        _was = bool(model.training)
        model.eval()
        try:
            with _t.no_grad():
                _x = _t.from_numpy(_np.ascontiguousarray(
                    state.encode()[None])).to(device)
                _out = model(_x)
                _wdl = _t.softmax(_out[1][0], dim=0).float().cpu()
                _W = 1.0 - float(_wdl[2])  # non-loss probability, draws must not resign
                _ml = float(_out[3][0].detach().cpu())
        finally:
            if _was:
                try:
                    model.train()
                except Exception:
                    pass
        if not (np.isfinite(_W) and np.isfinite(_ml)):
            return None, None
        return _W, min(max(_ml, 0.0), 1.0)
    except Exception:
        return None, None


def apply_tail_downweight(examples: list, record_plys: list,
                          tail_ply: int = 120,
                          tail_weight: float = 0.25) -> int:
    """D5 tail downweight: policy weight (tuple index 10) *= tail_weight
    for positions with record ply > tail_ply, so dead-won shuffling tails
    don't drown middlegame signal. No-op when tail_weight == 1.0.
    record_plys aligns with examples (callers slice under drop_tail).
    Returns the downweighted count. Never raises."""
    try:
        if float(tail_weight) == 1.0 or not examples:
            return 0
        _n = 0
        for _i, _ex in enumerate(examples):
            try:
                _ply = int(record_plys[_i]) if _i < len(record_plys) else 0
            except Exception:
                _ply = 0
            if _ply > int(tail_ply):
                try:
                    _l = list(_ex)
                    _l[10] = float(_l[10]) * float(tail_weight)
                    examples[_i] = tuple(_l)
                    _n += 1
                except Exception:
                    continue
        return _n
    except Exception:
        return 0


def shape_targets(hist, w_result, final_ply, move_cap, final_board,
                    full: bool = True, drop_tail: int = 0,
                    prog_flags=None, played_out: bool = True):
    """v9: value targets are PURE game results. Draws keep z=0.0 for every
    position — the v5-v8 contempt overwrite (draws relabelled ±contempt by
    final material) is gone; it made the value head regress on our own
    shaping instead of on game results (§5.4, values.py circularity §2.7).
    Contempt now steers only search (draw_leaf_value). Appends moves-left
    targets (plies remaining / move_cap) plus v10 dense targets: ownership
    map, final margin, and recorded mobility. v16 appends king safety.
    v18 appends opp-reply from-square (absolute 64-way; -100 when the
    reply chain breaks on unrecorded plies) and the fast/full policy
    weight (0 = fast game: value+aux only, KataGo doctrine). hist entries
    carry the PLAYED action (7th) for reply chaining, the per-move
    search KL (8th, B5) and the per-move MCTS root Q (9th, V20 D2 —
    the TD blend source; legacy hists lacking it read root_q = z so
    the blend == z exactly). Returns 18-tuples (V20: yesterday's 14
    [encode, pi, z, material, moves_left, ownership, margin, mobility,
    safety, reply_from, full_weight, kl, P, ml_mask] + score_mean(14),
    score_stdev(15), root_q(16), av(17) appended at END, KL untouched).
    score_mean/score_stdev come from score_targets_for(final_board)
    (D1, white-relative pawns, stdev 0.0: single terminal observation);
    av is None here (Agent C labels it offline via build_av; replay
    reads None as zeros(4096), masked to zero loss). replay.sample
    IGNORES elements 12 (P) and 11 (kl) (not training targets there);
    ml_mask rides along as the 12th sample array.
    prog_flags (D4): list[bool] parallel to hist (True = that recorded move
    reset the no-progress clock: pawn/capture/mate-terminal). None or
    length-mismatched (legacy/hand-built hists) -> all P censored
    (remaining plies). Sliced with the kept prefix under drop_tail.
    played_out (D5): ml_mask = 1.0 unless the game truncated (the caller
    passes False for adjudicated/adjudicated-draw/cap terminals; mate,
    rules-draw and resign count as played-out).
    drop_tail (B8 deblunder-lite): drop the LAST N positions (poison tail)
    before building. Default 0 (yesterday); the resign path passes 4."""
    from .game import decode_action as _dec
    from .game import mirror_square as _mir
    # B8: slice first, then build exactly like the normal path (replies
    # chain within the kept prefix; the cut end reads -100 like any chain
    # break). drop_tail <= 0 keeps everything; over-drop yields [].
    use = hist[:-drop_tail] if drop_tail and drop_tail > 0 else hist
    # D4: flags align with the kept prefix (first len(use) entries).
    if prog_flags is not None and len(prog_flags) == len(hist):
        _pf = list(prog_flags[:len(use)])
    else:
        _pf = [False] * len(use)
    _P = _progress_targets(use, _pf, final_ply)
    _mask = 1.0 if played_out else 0.0
    # V20 D1: one score-margin target per GAME (final-board truth, same
    # teaching as ownership/margin: every position learns the ending).
    try:
        _sc = score_targets_for(final_board)
    except Exception:
        _sc = {"score_mean": 0.0, "score_stdev": 0.0}
    n = len(use)
    examples = []
    for i, h in enumerate(use):
        enc, pi, stm, m, ply, mob = h[:6]
        act = h[6] if len(h) > 6 else None
        try:
            kl = float(h[7]) if len(h) > 7 else 0.0
        except Exception:
            kl = 0.0
        if not np.isfinite(kl):
            kl = 0.0
        z = w_result if stm == 0 else -w_result
        # V20 D2 root_q (EXACT name): per-move MCTS root Q from the hist
        # 9th slot; legacy hists read z (TD blend == z exactly, yesterday).
        try:
            _rq = float(h[8]) if len(h) > 8 else float(z)
        except Exception:
            _rq = float(z)
        if not np.isfinite(_rq):
            _rq = float(z)
        ml = min(1.0, max(0.0, (final_ply - ply) / 300.0))
        margin = final_margin(final_board, stm)
        # reply = from-square of the NEXT position's played move (the
        # opponent's reply), absolute squares (documented v18 convention).
        # Chain breaks (unrecorded opening/sparring plies between recorded
        # positions) -> -100 (CE ignore_index).
        reply = -100
        if act is not None and i + 1 < n:
            h2 = use[i + 1]
            if len(h2) > 6 and h2[6] is not None and \
                    int(h2[4]) == int(ply) + 1:
                try:
                    _fr = int(_dec(int(h2[6]))[0])
                    # h2's action is oriented to h2's stm; absolutize
                    # (Black stm mirrors) for the documented convention.
                    if int(h2[2]) == 1:
                        _fr = int(_mir(_fr))
                    reply = _fr
                except Exception:
                    reply = -100
        examples.append((enc, pi, float(z), float(m), float(ml),
                         ownership_map(final_board, stm), float(margin),
                         float(mob), float(king_safety(final_board, stm)),
                         int(reply), 1.0 if full else 0.0, float(kl),
                         float(_P[i]), float(_mask),
                         float(_sc["score_mean"]) * (1.0 if stm == 0 else -1.0),
                         float(_sc["score_stdev"]), float(_rq), None))
    return examples


def build_resign_examples(hist, w_result, final_ply, move_cap, final_board,
                          is_full: bool, drop_tail: int = 0,
                          prog_flags=None, played_out: bool = True):
    """B9 resign target builder: same targets as the normal path (the
    resigned-from board is the final board), factored out of
    _play_game_impl so the playthrough window can fire from two sites.
    drop_tail (B8): poison-tail cut before building (resign path passes
    4; default 0 = yesterday). 18-tuples (V20, same layout as
    shape_targets: kl at 11, P at 12, ml_mask at 13, score_mean at 14,
    score_stdev at 15, root_q at 16 from the hist 9th slot defaulting
    to z, av None at 17). prog_flags/played_out behave exactly like
    shape_targets."""
    from .game import decode_action as _dec2
    from .game import mirror_square as _mir2
    use = hist[:-drop_tail] if drop_tail and drop_tail > 0 else hist
    if prog_flags is not None and len(prog_flags) == len(hist):
        _pf = list(prog_flags[:len(use)])
    else:
        _pf = [False] * len(use)
    _P = _progress_targets(use, _pf, final_ply)
    _mask = 1.0 if played_out else 0.0
    try:
        _sc2 = score_targets_for(final_board)
    except Exception:
        _sc2 = {"score_mean": 0.0, "score_stdev": 0.0}
    examples = []
    for _i, _h in enumerate(use):
        enc, p, stm, m, ply, mob = _h[:6]
        try:
            _kl = float(_h[7]) if len(_h) > 7 else 0.0
        except Exception:
            _kl = 0.0
        if not np.isfinite(_kl):
            _kl = 0.0
        _z = (w_result if stm == 0 else -w_result)
        try:
            _rq2 = float(_h[8]) if len(_h) > 8 else float(_z)
        except Exception:
            _rq2 = float(_z)
        if not np.isfinite(_rq2):
            _rq2 = float(_z)
        _reply = -100
        if len(_h) > 6 and _i + 1 < len(use):
            _h2 = use[_i + 1]
            if len(_h2) > 6 and _h2[6] is not None and \
                    int(_h2[4]) == int(ply) + 1:
                try:
                    _fr = int(_dec2(int(_h2[6]))[0])
                    if int(_h2[2]) == 1:
                        _fr = int(_mir2(_fr))
                    _reply = _fr
                except Exception:
                    _reply = -100
        examples.append((
            enc, p, (w_result if stm == 0 else -w_result), m,
            min(1.0, max(0.0, (final_ply - ply) / 300.0)),
            ownership_map(final_board, stm),
            final_margin(final_board, stm), mob,
            king_safety(final_board, stm), _reply,
            1.0 if is_full else 0.0, float(_kl),
            float(_P[_i]), float(_mask),
            float(_sc2["score_mean"]) * (1.0 if stm == 0 else -1.0), float(_sc2["score_stdev"]),
            float(_rq2), None))
    return examples


def finish_move(state, choice, pi, model, device, cfg, topk=4,
                pi_frac=0.25, win_v=0.5):
    """v15 ML-guided finishing: when clearly ahead, play the move
    minimizing predicted remaining game length (moves-left head).
    Pure RL: no external data — spends the ml head trained (loss-only)
    since v6. The old failure: mates and shuffles both score z=1.0, so
    nothing preferred mating; distance is the missing gradient.
    Only ever picks among near-best visited moves (topk by pi within
    pi_frac of max) PLUS the guarded choice (always included, so veto /
    tactics overrides survive). No-op when cfg.mate_finish is off,
    model is None (infer-server path), not winning (v_stm < win_v),
    or on any error. One batched forward per game-move (not per sim).
    """
    try:
        if not bool(getattr(cfg, "mate_finish", False)) or model is None:
            return choice
        import torch as _t
        import numpy as _np
        mx = float(_np.max(pi))
        order = _np.argsort(-pi)
        cands = [int(a) for a in order[:topk]
                 if float(pi[int(a)]) >= pi_frac * mx]
        if int(choice) not in cands:
            cands.append(int(choice))
        if len(cands) < 2:
            return choice
        states = [state] + [state.apply(a) for a in cands]
        x = _t.stack(
            [_t.from_numpy(_np.asarray(s.encode(), dtype=_np.float32))
             for s in states]).to(device)
        # Caller discipline keeps the net in eval here (make_evaluate
        # flips it); the helper never mutates mode, never trains.
        with _t.no_grad():
            out = model(x)
        # v18: out[1] is WDL logits — Q = P(W) - P(L).
        _wq = _t.softmax(out[1][0], dim=0)
        v_stm = float((_wq[0] - _wq[2]).detach().cpu())
        # Broken-head guard: NaN value/ML must hold the search pick,
        # never argmin into garbage (argmin over NaN is arbitrary).
        if not _np.isfinite(v_stm) or v_stm < win_v:
            return choice
        scored = []
        for i, a in enumerate(cands):
            _ml = float(out[3][1 + i].detach().cpu())
            if not _np.isfinite(_ml):
                return choice
            # Never prefer a move that draws on the spot (stalemate /
            # dead draw): mates read z=-1 for the mated side and stay
            # preferred at ml~=0. Adjudicated wins (z=+-1) also stay.
            _ch = states[1 + i]
            _done, _z = _ch.is_terminal()
            if _ch.board.is_checkmate():
                return int(a)
            if _done:
                continue
            _cp = _t.softmax(out[1][1+i], dim=0)
            _cq = float((_cp[2] - _cp[0]).detach().cpu())
            if _cq < max(win_v, v_stm - 0.10):
                continue
            if int(a) != int(choice) and hangs_material(state, int(a)):
                continue
            scored.append((_ml, int(a)))
        if not scored:
            return choice
        scored.sort()
        return int(scored[0][1])
    except Exception:
        return choice


def safety_guard(state, choice, pi, model, device, cfg, topk=4,
                   pi_frac=0.25, drop_thr=0.15):
    """v15.5 king-safety veto (TEETH for the safety head): the head
    predicts, this guard acts. Among near-best visited moves (topk by pi
    within pi_frac of max, PLUS the guarded choice always included), it
    rejects the pick when a sibling keeps our king clearly safer (own-
    safety gap > drop_thr) and plays the safest sibling instead.
    Returns (action, vetoed). Self-gating: an untrained head outputs ~0.5
    clustered, so gaps never reach the threshold and the veto sleeps
    until the head learns. One batched forward per game-move (finish_move
    doctrine). Never fires on forced tactics (caller runs it only when
    tac is None and no veto replaced). No-op on flag off / model None /
    NaN / any error. NOTE: arena/gate agents call this with a bare
    SimpleNamespace(safety_veto=True), so the DEFAULT drop_thr (0.15)
    is the live threshold there — keep config values aligned with it
    (V18 uses 0.15 exactly)."""
    try:
        if not bool(getattr(cfg, "safety_veto", False)) or model is None:
            return int(choice), False
        import torch as _t
        import numpy as _np
        thr = float(getattr(cfg, "safety_drop_thr", drop_thr) or drop_thr)
        mx = float(_np.max(pi))
        order = _np.argsort(-pi)
        cands = [int(a) for a in order[:topk]
                 if float(pi[int(a)]) >= pi_frac * mx]
        if int(choice) not in cands:
            cands.append(int(choice))
        if len(cands) < 2:
            return int(choice), False
        children = [state.apply(a) for a in cands]
        x = _t.stack(
            [_t.from_numpy(_np.asarray(s.encode(), dtype=_np.float32))
             for s in children]).to(device)
        with _t.no_grad():
            out = model(x)
        # safety head is stm-relative: child stm is the OPPONENT, so our
        # safety after our move = 1 - head(child).
        ours = []
        for i in range(len(cands)):
            _s = float(out[7][i].detach().cpu())
            if not _np.isfinite(_s):
                return int(choice), False
            ours.append(1.0 - _s)
        ci = cands.index(int(choice))
        bi = int(_np.argmax(ours))
        if bi != ci and ours[bi] - ours[ci] >= thr and not hangs_material(state, int(cands[bi])):
            return int(cands[bi]), True
        return int(choice), False
    except Exception:
        return int(choice), False


def _play_game_impl(model, cfg, device="cpu", temp_moves=20,
                    opening_random_moves=0,
                    resign_threshold=None, resign_moves=3, sparring=None,
                    spar_white=True, evaluate_fn=None,
                    stats: dict | None = None,
                    full_playout: bool = False,
                    fast: bool = False,
                    start_fen: str | None = None,
                    playthrough: bool = False) -> tuple[list, str]:
    """Sparring (v8.2): when sparring policy is set and the sparring color is
    to move, play its move with no MCTS and record nothing. Our moves keep
    full search + targets, so hanging material gets punished in training
    (greedy always takes free pieces) instead of only in the arena.
    evaluate_fn (phase 3, opt-in): prebuilt evaluate closure (e.g. server-
    backed). When given, model may be None — it is not touched.
    full_playout (v13): no resign, no adjudication (mate/rules/cap only).
    fast (v18, KataGo playout-cap doctrine): cheap game for VALUE data —
    fast_sims (no Dirichlet noise), normal rules. Its tuples carry
    policy weight 0 (value+aux only); policy trains on full games.
    Full-playout (mate-signal) wins over fast when both fire.
    Resign is per-side (v14): the mover resigns after resign_moves of
    THEIR moves read lost (shared streaks could never fire on
    stm-relative values). stats gains terminal/breadth/ventropy/plies/
    finishes/tactics/vetoes/safety/fast when provided."""
    from .game import unit_count_stm, material_stm
    import chess as _c0
    # v9: contempt steers SEARCH (draw_leaf_value), not labels. The same
    # steered value adaptive_contempt maintains now flows into every MCTS
    # call below instead of into shape_targets.
    contempt = float(getattr(cfg, "contempt", 0.0) or 0.0)
    asymmetric = bool(getattr(cfg, "asymmetric_contempt", False))
    edge_scale = float(getattr(cfg, "contempt_edge_scale", 0.0) or 0.0)
    aux_fn = unit_count_stm if getattr(cfg, "aux_unit_counts", False) \
        else material_stm
    evaluate = evaluate_fn if evaluate_fn is not None \
        else make_evaluate(model, device)
    if model is None and hasattr(evaluate, "model"):
        model = evaluate.model
    # B6 eval cache: one dict per GAME (created here, owned by this game),
    # passed to EVERY search call below. Key format is mcts._cache_key's —
    # (rep_key, rep_count) — constructed inside search; this dict stays
    # opaque here. evaluate_fn itself stays as-is (eval_cache is a
    # search() param only, never an evaluate_fn param).
    _is_full_game = not fast
    if sparring is not None and hasattr(sparring, "reset"):
        sparring.reset()
    eval_cache: dict = {}
    # V21.1 TT-carryover fix (F1/F2/F6): on the rust-tree path each game
    # owns ONE Rust-side GameStore (TT + per-game eval cache) instead of
    # the Python dicts. The dicts above stay empty there (never written);
    # every prune_tt site below branches to store.prune_to. None = Python
    # path (or missing wheel fallback to dicts + stub).
    _rust_on = bool(getattr(cfg, "rust_tree", False))
    _rust_store = None
    if _rust_on:
        try:
            from .mcts_bridge import new_game_store as _ngs
            _rust_store = _ngs()
        except Exception:
            _rust_store = None
    # D3 in-tree MLH wiring: search-side params thread from cfg with
    # yesterday defaults (slope 0.0 = off = bit-exact); V19 sets 0.003.
    # ml_fn is a side channel (NOT evaluate_fn's contract): one extra
    # model forward per batch/leaf reading the moves_left head (out[3],
    # normalized 0..1 — indices 0-8 are arch-stable per model.py). Server
    # path (model None) -> ml_fn None -> bonus off, logged once.
    # V23 A3 (audit 22): built via the shared resolve_ml_fn constructor
    # (consumes forward_inf when present, else the existing out[3] path;
    # defers to A4's inference_fusion.make_ml_fn when it lands). Values
    # are bit-identical either way (same modules, same order).
    ml_slope = float(getattr(cfg, "ml_slope", 0.0) or 0.0)
    ml_cap = float(getattr(cfg, "ml_cap", 0.07) or 0.07)
    ml_thr = float(getattr(cfg, "ml_thr", 0.8) or 0.8)
    ml_fn = None
    if ml_slope != 0.0:
        if model is not None:
            ml_fn = resolve_ml_fn(model, device)
        else:
            global _ML_OFF_LOGGED
            if not _ML_OFF_LOGGED:
                print("[ml] server/evaluate_fn path (model None): "
                      "in-tree ML bonus off", flush=True)
                _ML_OFF_LOGGED = True
    state = State.initial()
    tt: dict = {}  # subtree reuse across moves within this game
    # v14 fix: seed with the initial key. Starting empty missed loops
    # back to the startpos (k0) in-search. The current key is included
    # too — harmless, since loop checks run on children (never equal).
    rep_hist: list = [State.initial().rep_key()]
    # V20 D4 endgame starts: FEN start replaces the initial position
    # (uniform draw by the caller at endgame_frac). contempt is FORCED to
    # 0.0 on these games (no draw-shaping while learning technique) and
    # they are EXCLUDED from forced-book/curated-book/random opening
    # plies below (the start IS the diversity). Bad FEN -> initial +
    # loud log (never raises).
    _eg_start = bool(start_fen)
    if _eg_start:
        try:
            import chess as _c0b
            state = State(_c0b.Board(str(start_fen)))
            rep_hist = [state.rep_key()]
            contempt = 0.0
        except Exception as _e:
            print(f"[endgame] bad start FEN ({str(start_fen)[:60]}: "
                  f"{type(_e).__name__}); initial instead", flush=True)
            state = State.initial()
            rep_hist = [state.rep_key()]
            _eg_start = False
    # D6 alignment anchor: the TB replay starts here (None = startpos).
    _fen_out = str(start_fen) if _eg_start else None
    # hist entries: (encode, pi, stm, material_target, record_ply,
    # mobility, played_action, search_kl, root_q) — v18 7th element feeds
    # reply chaining; B5 8th element carries per-move search KL (stats["kl"],
    # 0.0 default) for B7 surprise weighting downstream; V20 9th element
    # carries per-move MCTS root Q (stats["root_q"]) for the D2 TD blend
    # (shape_targets reads it as tuple index 16; legacy hists read z).
    hist: list[tuple[np.ndarray, np.ndarray, int, float, int]] = []
    # V20 D6 TB-rescore move log: EVERY applied move as absolute UCI
    # (unrecorded opening plies, sparring replies and main moves alike),
    # so the loop can replay the game against data_tb/ without the
    # encoded tuples. Curated/empirical book openings bypass this log
    # (take_book_opening returns no moves) — those games set
    # stats["book_opening"]=True and the rescore path skips them loudly.
    _moves_uci: list = []
    _book_n = 0
    # D4 progress flags parallel to hist (True = that recorded move reset
    # the no-progress clock: pawn/capture/mate-terminal). Fed to the
    # builders for P targets; terminal-after-move upgrades set below.
    prog_flags: list[bool] = []
    resign_streak = {_c0.WHITE: 0, _c0.BLACK: 0}  # per-side (v14: a shared
    # streak could never fire — v_stm is stm-relative and alternates sign
    # every ply, so it never read lost 3x in a row. Tresign 0 for ~90
    # iters proved resign was dead; the mover resigns after resign_moves
    # of THEIR OWN moves read lost.
    # AZ-paper playout rule (v8.3): 10% of games ignore resign so bad
    # positions stay in the buffer (else the arena's lost positions are all
    # out-of-distribution and the model shuffles in them).
    import random as _rr0
    playout = _rr0.random() < float(
        getattr(cfg, "resign_playout_frac", 0.0) or 0.0)
    # V20 D5 playthrough (playthrough_frac coin lives in loop/parallel):
    # these games ignore resign ENTIRELY (termination only). F5 refusal:
    # this is 5%, never full no-resign (mate-finish module untouched).
    _no_resign_game = bool(playthrough) or bool(playout)
    # V20 D5 smart-resign state (R3 replaces the fixed resign rule when
    # cfg.smart_resign is on; old streak+B9 window path kept verbatim
    # when it is off). Checks run every resign_every plies; resign needs
    # resign_consec consecutive W<w_thr AND ML<ml_thr reads.
    _smart = bool(getattr(cfg, "smart_resign", False))
    _w_thr = float(getattr(cfg, "resign_w_thr", 0.02) or 0.02)
    _ml_thr = float(getattr(cfg, "resign_ml_thr", 0.3) or 0.3)
    _consec_need = max(1, int(getattr(cfg, "resign_consec", 3) or 3))
    _every = max(1, int(getattr(cfg, "resign_every", 8) or 8))
    _smart_consec = {_c0.WHITE: 0, _c0.BLACK: 0}
    _smart_next = {_c0.WHITE: 0, _c0.BLACK: 0}
    # V20 D2 root_q log: per-move MCTS root Q (stats["root_q"] per
    # search, mcts-side) averaged into stats at every return site.
    _root_qs: list = []
    # v16 curated openings: from a random ECO line (real positions)
    # instead of uniform-random plies when configured. Unrecorded like
    # opening_random_moves (no MCTS targets) but rep-seeded (truthful).
    # Replaces random junk, keeps diversity (16 lines x depths).
    from . import book as _bookmod
    _blines = [] if _eg_start else _bookmod.load_book(
        getattr(cfg, "book_path", "") or "")
    _bplies = int(getattr(cfg, "book_plies", 0) or 0)
    if _blines and _bplies > 0 and not _eg_start:
        state, _nbook = _bookmod.take_book_opening(
            state, rep_hist, _blines, _bplies)
        _book_n += int(_nbook or 0)
    else:
        # v19 D12 forced book: opening_book_frac of games open from the
        # empirical TRAIN split (unrecorded, rep-seeded, same rules).
        # book_empirical gates it (old configs: random plies as yesterday).
        _bfrac = float(getattr(cfg, "opening_book_frac", 0.0) or 0.0)
        if not _eg_start and bool(getattr(cfg, "book_empirical", False)) \
                and _bfrac > 0.0 and _rr0.random() < _bfrac:
            try:
                _elines = _bookmod.load_book(_bookmod._EMPIRICAL)
                _tlines, _ = _bookmod.split_book_lines(_elines, 60)
                _bpl = int(getattr(cfg, "book_plies", 0) or 6)
                if _tlines and _bpl > 0:
                    state, _nbook = _bookmod.take_book_opening(
                        state, rep_hist, _tlines, _bpl)
                    _book_n += int(_nbook or 0)
            except Exception:
                pass
    # v13: full-playout (mate-signal) games disable BOTH early-ending
    # paths: resign AND material adjudication (via game TRUNCATE_ENDINGS,
    # restored on every return below). Games end only by mate, rules
    # draws, or the ply cap (cap = draw).
    import chess_zero.game as _gg
    _truncate_saved = _gg.TRUNCATE_ENDINGS
    if full_playout:
        _gg.TRUNCATE_ENDINGS = False
        resign_threshold = None
    tac_on = bool(getattr(cfg, "tactical_override", False))
    tac_thr = float(getattr(cfg, "tac_threshold", 0.09))
    veto_on = bool(getattr(cfg, "blunder_veto", False))
    n_tac = n_veto = 0  # v10 veto telemetry (reported via stats)
    n_fin = 0  # v15 finishing telemetry
    n_safe = 0  # v15.5 safety-veto telemetry
    breadth_acc, vent_acc, b_n = 0.0, 0.0, 0  # v14 visit breadth
    # B9 resign-playthrough window: plies remaining once the resign streak
    # triggers (0 = no window open). The window opens instead of resigning
    # immediately; it cancels on recovery, else fires the resign.
    _pt_remaining = 0
    _pt_side = None

    def _resign_stats(_ex=None) -> None:
        # Shared tail for every resign-fire site (B9 fires from the
        # evaluated path or the tac-ply path): same telemetry as the
        # normal return, terminal "resign". Reads the current loop
        # locals at call time (per-iteration state/_is_full_game).
        if stats is not None:
            stats["tactics"] = stats.get("tactics", 0) + n_tac
            stats["vetoes"] = stats.get("vetoes", 0) + n_veto
            stats["finishes"] = stats.get("finishes", 0) + n_fin
            stats["safety"] = stats.get("safety", 0) + n_safe
            stats["fast"] = stats.get("fast", 0) + \
                (0 if _is_full_game else 1)
            stats["terminal"] = "resign"
            stats["breadth"] = breadth_acc / max(b_n, 1)
            stats["ventropy"] = vent_acc / max(b_n, 1)
            stats["plies"] = int(state.ply_count)
            _v20_note_stats(stats, state.board)
            _note_movelog(stats, _ex)

    def _v20_note_stats(_st: dict, _board) -> None:
        # V20 side-channel log (arity-safe): per-game mean root_q (D2),
        # score_mean/score_stdev (D1), av_logits placeholder (D3, Agent C
        # fills via build_av; training gradient only, never search), plus
        # endgame/playthrough/tail flags. Tuple promotion is Agent A/C's
        # (KL stays LAST); this dict never touches replay arity (F9).
        try:
            _st["root_q"] = round(float(sum(_root_qs) / len(_root_qs)), 4) \
                if _root_qs else 0.0
        except Exception:
            _st["root_q"] = 0.0
        try:
            _sc = score_targets_for(_board)
            _st["score_mean"] = _sc["score_mean"]
            _st["score_stdev"] = _sc["score_stdev"]
        except Exception:
            _st["score_mean"] = 0.0
            _st["score_stdev"] = 0.0
        _st["av_logits"] = None  # D3: Stockfish-labeled offline (Agent C)
        _st["endgame_start"] = 1 if _eg_start else 0
        _st["playthrough"] = 1 if bool(playthrough) else 0

    def _note_movelog(_st: dict, _ex=None) -> None:
        # V20 D6 move log for the loop's post-game TB-rescore pass:
        # full UCI list (replayable from startpos, or from start_fen for
        # endgame starts) + record plies parallel to _ex (the kept hist
        # prefix, so drop_tail slicing stays aligned) + book flag (book
        # openings bypass the log -> rescore skips those games loudly).
        # Never raises; logging must not break a game.
        try:
            _st["moves_uci"] = list(_moves_uci)
        except Exception:
            _st["moves_uci"] = []
        try:
            _n = len(_ex) if _ex is not None else 0
            _st["record_plys"] = [int(_h[4]) for _h in hist[:_n]]
        except Exception:
            _st["record_plys"] = []
        _st["start_fen"] = _fen_out
        _st["book_opening"] = bool(_book_n > 0)
    while True:
        done, _ = state.is_terminal()
        if done or state.ply_count >= cfg.move_cap:
            break
        if state.ply_count < opening_random_moves and not _eg_start:
            # unrecorded uniform-random opening ply: diversity without
            # teaching the policy junk (no MCTS target exists for it).
            import random as _r
            choice = int(_r.choice(state.legal_moves()))
            try:
                _moves_uci.append(state.to_uci(int(choice)))
            except Exception:
                pass
            state = state.apply(choice)
            # §6.5: unsearched move breaks the search line — drop cached
            # subtrees (they belong to a different route with different
            # clocks/history) instead of retaining them all game.
            # F2: rust path prunes the GameStore; the dict stays empty.
            if _rust_on and _rust_store is not None:
                try:
                    _rust_store.prune_to(state.key())
                except Exception:
                    pass
            else:
                mcts_mod.prune_tt(tt, state.key())
            rep_hist.append(state.rep_key())
            # v15: no streak reset on unsearched opening plies — the
            # streak counts OUR consecutive lost reads; opponent/random
            # moves must not clear it (same for sparring below).
            if state.ply_count >= cfg.move_cap:
                break
            continue
        spar_turn = sparring is not None and (
            (state.board.turn == _c0.WHITE) == spar_white)
        if spar_turn:
            choice = int(sparring(state))
            try:
                _moves_uci.append(state.to_uci(int(choice)))
            except Exception:
                pass
            state = state.apply(choice)
            if _rust_on and _rust_store is not None:
                try:
                    _rust_store.prune_to(state.key())
                except Exception:
                    pass
            else:
                mcts_mod.prune_tt(tt, state.key())
            rep_hist.append(state.rep_key())
            # v15: sparring moves don't clear our streak (see above).
            if state.ply_count >= cfg.move_cap:
                break
            continue
        _sst: dict = {}
        # v18 fast games: fast_sims, no Dirichlet noise (KataGo: fast
        # disables noise, maximizes strength-per-sim for value data).
        _fast_on = bool(fast) and not bool(full_playout)
        _sims_now = int(getattr(cfg, "fast_sims", 0) or cfg.sims) \
            if _fast_on else cfg.sims
        _eps_now = 0.0 if _fast_on else cfg.dirichlet_eps
        _is_full_game = not _fast_on
        # V21.1 E4 loop seam (flag only, default OFF): cfg knob stamped
        # by run_training (--rust-tree). False = mcts_mod.search exactly
        # as yesterday; True = identical kwargs via RustTreeBackend.
        # F1/F2/F6: rust path threads the per-game GameStore through
        # tt + eval_cache (dicts stay empty); otherwise the game dicts.
        _skw = dict(c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_eps=_eps_now,
                    tt=tt, history=list(state._hist),
                    stats=_sst,
                    contempt=contempt,
                    asymmetric_contempt=asymmetric,
                    contempt_edge_scale=edge_scale,
                    quiescence_depth=int(getattr(
                        cfg, "quiescence_depth", 0) or 0),
                    forcing_bonus=float(getattr(
                        cfg, "forcing_bonus", 0.0) or 0.0),
                    leaf_batch=int(getattr(
                        cfg, "leaf_batch", 1) or 1),
                    virtual_loss=float(getattr(
                        cfg, "virtual_loss", 1.0) or 1.0),
                    fpu_reduction=float(getattr(
                        cfg, "fpu_reduction", 0.0) or 0.0),
                    prune_singletons=bool(getattr(
                        cfg, "prune_singletons", False)),
                    ml_fn=ml_fn, ml_slope=ml_slope,
                    ml_cap=ml_cap, ml_thr=ml_thr,
                    eval_cache=eval_cache)
        if _rust_on and _rust_store is not None:
            _skw["tt"] = _rust_store
            _skw["eval_cache"] = _rust_store
        if bool(getattr(cfg, "rust_tree", False)):
            from .mcts_bridge import RustTreeBackend as _RTB
            pi = _RTB().search(state, evaluate, n_sims=_sims_now,
                               **_skw)[0]
        else:
            pi = mcts_mod.search(state, evaluate, n_sims=_sims_now,
                                 **_skw)
        # v14: accumulate per-search visit breadth (unconventional-move
        # diagnostic); averaged per game into stats below.
        breadth_acc += float(_sst.get("breadth", 0))
        vent_acc += float(_sst.get("visit_entropy", 0.0))
        b_n += 1
        # V20 D2: log this move's MCTS root Q for the TD blend (stats
        # mean at return sites; hist 9th slot -> tuple field root_q).
        _rq_now = 0.0
        try:
            _rq_now = float(_sst.get("root_q", 0.0))
            _root_qs.append(_rq_now if np.isfinite(_rq_now) else 0.0)
        except Exception:
            _root_qs.append(0.0)
            _rq_now = 0.0
        if not np.isfinite(_rq_now):
            _rq_now = 0.0
        enc = state.encode()
        tac = tactical_action(state, tac_thr) if tac_on else None
        # temperature (tactical overrides play deterministically: a
        # forced recapture is not a suggestion)
        if tac is not None:
            choice = int(tac)
        elif state.ply_count < temp_moves:
            legal = np.flatnonzero(pi)
            probs = pi[legal] / pi[legal].sum()
            choice = int(np.random.choice(legal, p=probs))
        else:
            choice = int(np.argmax(pi))
        # v9 guards (shared choke point — same call arena/gate/deployment
        # use): forced tactic one-hots for PLAY, veto replaces hangs for
        # PLAY, quiet choices keep their visits.
        # V23 A3 (audit 20): play stays guarded (product doctrine), but the
        # STORED target is blended: (1-w)*raw_visits + w*guard_one_hot with
        # mate/legality strong (tac 1.0, veto 0.9) and learned safety/finish
        # weak (0.25). Raw visits / played / reason / confidence ride the
        # per-game stats side-channel (arity stays 18). hist records the
        # BLENDED target so targets never teach maximal confidence for a
        # learned preference; the PLAYED action below is still the guard
        # move (replay chaining + films unchanged).
        if tac is not None:
            n_tac += 1
        _raw_pi = np.asarray(pi, dtype=np.float32).copy()
        choice, pi, _vetoed = apply_move_guards(
            state, choice, pi, tac, veto_on, tac_thr)
        if tac is not None:
            _guard_reason = "tactical"
        elif _vetoed:
            _guard_reason = "veto"
        else:
            _guard_reason = "none"
        if _vetoed:
            n_veto += 1
        # V15.5 safety veto AFTER blunder veto (soundness first): tac and
        # veto replacements are never second-guessed. Play one-hots on
        # replace (product doctrine); the training target blends weak.
        if tac is None and not _vetoed:
            _pre_s = int(choice)
            choice, _sved = safety_guard(
                state, choice, pi, model, device, cfg)
            if _sved:
                _oh = np.zeros(4096, dtype=np.float32)
                _oh[int(choice)] = 1.0
                pi = _oh
                n_safe += 1
                _guard_reason = "safety"
        # v15 finishing AFTER guards: soundness first — a forced tactic
        # or veto replacement is never second-guessed for speed. Play
        # one-hots on replace; the training target blends weak.
        if tac is None and not _vetoed:
            _pre = int(choice)
            choice = finish_move(state, choice, pi, model, device, cfg)
            if int(choice) != _pre:
                _oh = np.zeros(4096, dtype=np.float32)
                _oh[int(choice)] = 1.0
                pi = _oh
                n_fin += 1
                _guard_reason = "finish"
        # Blended training target + raw record (audit 20). "none" blends
        # to the raw visits exactly (bit-identical passthrough); tac 1.0
        # is bit-identical to yesterday's one-hot; veto/safety/finish now
        # carry corrective (not maximal) confidence.
        try:
            _pi_target = blend_guard_target(_raw_pi, int(choice),
                                            _guard_reason)
        except Exception:
            _pi_target = pi
        try:
            _gi = make_guard_info(_guard_reason, _raw_pi, int(choice))
            _gr = stats.get("guard_reasons", None) if stats is not None \
                else None
            if stats is not None:
                if not isinstance(_gr, dict):
                    stats["guard_reasons"] = {}
                    _gr = stats["guard_reasons"]
                _gr[_guard_reason] = int(_gr.get(_guard_reason, 0)) + 1
                stats["guard_conf_sum"] = float(
                    stats.get("guard_conf_sum", 0.0)) + float(
                        _gi.get("confidence", 0.0))
                stats["guard_moves"] = int(
                    stats.get("guard_moves", 0)) + 1
        except Exception:
            pass
        mob_now = min(1.0, len(state.legal_moves()) / 50.0)
        # v18: hist carries the PLAYED action (7th) for opp-reply
        # chaining in shape_targets (reply = next position's move).
        # B5: 8th element is this move's search KL (stats["kl"], never
        # raises; 0.0 default) for B7 surprise weighting downstream.
        try:
            _kl_now = float(_sst.get("kl", 0.0))
        except Exception:
            _kl_now = 0.0
        if not np.isfinite(_kl_now):
            _kl_now = 0.0
        hist.append((enc, pi, state.side_to_move,
                     aux_fn(state.board), state.ply_count, mob_now,
                     int(choice), float(_kl_now), float(_rq_now)))
        # D4: flag the recorded move (pawn/capture now; any terminal after
        # it upgrades below — the wait ends when the game ends).
        try:
            _prog_now = bool(_move_is_progress(state, int(choice)))
        except Exception:
            _prog_now = False
        prog_flags.append(_prog_now)
        # v15: never resign a position where tactics found a forced
        # sound move (tac can be mate/stalemate tricks the value head
        # misreads as lost — playing them beats resigning them).
        # V20 D5 smart resign (R3 replaces the fixed rule when
        # cfg.smart_resign is on; V20 sets resign_threshold=None): resign
        # only if W < resign_w_thr AND ML_pred < resign_ml_thr for
        # resign_consec consecutive checks every resign_every plies; else
        # play on. Playthrough/playout games skip checks entirely. When
        # smart_resign is off the B9 window path below runs verbatim
        # (yesterday bit-exact).
        _tail_w = float(getattr(cfg, "tail_weight", 1.0) or 1.0)
        _tail_ply = int(getattr(cfg, "tail_ply", 10 ** 9) or 10 ** 9)
        if _smart and tac is None and not _no_resign_game \
                and not full_playout:
            _side = state.board.turn
            if int(state.ply_count) >= _smart_next[_side]:
                _smart_next[_side] = int(state.ply_count) + _every
                _W_now, _ML_now = smart_resign_read(
                    model, device, state)
                if _W_now is None:
                    pass  # server path (model None): play on, never resign
                elif _W_now < _w_thr and _ML_now < _ml_thr:
                    _smart_consec[_side] += 1
                else:
                    _smart_consec[_side] = 0
                if _smart_consec[_side] >= _consec_need:
                    import chess as _c
                    w_result = -1.0 if state.board.turn == _c.WHITE \
                        else 1.0
                    examples = build_resign_examples(
                        hist, w_result, state.ply_count, cfg.move_cap,
                        state.board, _is_full_game, drop_tail=4,
                        prog_flags=prog_flags, played_out=True)
                    _td_n = apply_tail_downweight(
                        examples, [_h[4] for _h in hist][:len(examples)],
                        _tail_ply, _tail_w)
                    res = "1-0" if w_result == 1.0 else "0-1"
                    _resign_stats(examples)
                    if stats is not None:
                        stats["tail_downweighted"] = int(_td_n)
                    _gg.TRUNCATE_ENDINGS = _truncate_saved
                    return examples, res
        elif not _smart and resign_threshold is not None and tac is None:
            # one extra value eval per move (~2% overhead at 40 sims):
            # resign lost positions instead of shuffling to the cap.
            _, vs = evaluate([state])
            v_stm = float(np.asarray(vs).flatten()[0])
            _mover = state.board.turn
            if v_stm < resign_threshold:
                resign_streak[_mover] += 1
            else:
                resign_streak[_mover] = 0
            # B9 resign playthrough: when the streak triggers, do NOT resign
            # immediately — play up to 6 more plies; cancel if v_stm recovers
            # above threshold+0.15 in that window (keep playing normally after
            # cancel); else resign with the same targets (drop_tail=4 poison
            # cut). Bounded by the window counter; no behavior change when
            # threshold is None (window never opens).
            if _pt_remaining > 0:
                if (v_stm if _mover == _pt_side else -v_stm) > resign_threshold + 0.15:
                    # Recovery in the window: cancel the resign. Both
                    # streaks reset — the window's positions supersede the
                    # streak evidence; fresh reads rebuild it if still
                    # lost. Keep playing normally after cancel.
                    _pt_remaining = 0
                    resign_streak[_c0.WHITE] = 0
                    resign_streak[_c0.BLACK] = 0
                else:
                    _pt_remaining -= 1
                    if _pt_remaining <= 0:
                        # Window exhausted, still lost: resign with the
                        # same targets as the old immediate path.
                        import chess as _c
                        w_result = -1.0 if _pt_side == _c.WHITE else 1.0
                        examples = build_resign_examples(
                            hist, w_result, state.ply_count, cfg.move_cap,
                            state.board, _is_full_game, drop_tail=4,
                            prog_flags=prog_flags, played_out=True)
                        _td_n = apply_tail_downweight(
                            examples, [_h[4] for _h in hist][:len(examples)],
                            _tail_ply, _tail_w)
                        res = "1-0" if w_result == 1.0 else "0-1"
                        _resign_stats(examples)
                        if stats is not None:
                            stats["tail_downweighted"] = int(_td_n)
                        _gg.TRUNCATE_ENDINGS = _truncate_saved
                        return examples, res
            elif resign_streak[_mover] >= resign_moves \
                    and not _no_resign_game:
                # Streak triggered: open the 6-ply playthrough window
                # instead of resigning immediately; this move plays on
                # normally (falls through to state.apply below).
                _pt_remaining = 6
                _pt_side = _mover
        elif not _smart and _pt_remaining > 0:
            # A forced tactic is evidence worth playing, never a trigger
            # to resign without a fresh evaluation from the owner's view.
            _pt_remaining = 0
            resign_streak = {_c0.WHITE: 0, _c0.BLACK: 0}
        try:
            _moves_uci.append(state.to_uci(int(choice)))
        except Exception:
            pass
        state = state.apply(choice)
        # D4: any terminal after the move ends the progress wait (mate,
        # rules-draw, adjudication alike — P measures plies until the game
        # stops needing progress). Pure read of the fresh state's label
        # (the loop-top check re-derives it; idempotent).
        try:
            _pdone, _ = state.is_terminal()
        except Exception:
            _pdone = False
        if _pdone and prog_flags:
            prog_flags[-1] = True
        # §6.5/§6.6: keep ONLY the promoted child's subtree. Retaining
        # everything grows ~1190 nodes/ply with an entry cap that never
        # fires, and lets position-keyed hits merge different histories
        # (different halfmove_clock / rep_count).
        # F2: rust path prunes the GameStore; the dict stays empty.
        if _rust_on and _rust_store is not None:
            try:
                _rust_store.prune_to(state.key())
            except Exception:
                pass
        else:
            mcts_mod.prune_tt(tt, state.key())
        rep_hist.append(state.rep_key())
        if state.ply_count >= cfg.move_cap:
            break

    done, z_stm = state.is_terminal()
    # z_stm is from final side-to-move view; convert per position
    # result from White's view:
    if not done:
        w_result = 0.0
    elif z_stm == 0.0:
        w_result = 0.0
    else:
        # side to move got z_stm; white gets z_stm if stm is white else -z_stm
        import chess
        w_result = z_stm if state.board.turn == chess.WHITE else -z_stm
    examples = shape_targets(hist, w_result, state.ply_count, cfg.move_cap,
                              state.board, full=_is_full_game, drop_tail=0,
                              prog_flags=prog_flags,
                              played_out=(getattr(
                                  state, "last_terminal", None) or "cap")
                              not in ("adjudicated", "adjudicated-draw",
                                      "cap"))
    # V20 D5 tail downweight (dead-won shuffling tails weight 0.25 past
    # tail_ply; yesterday-off via tail_weight=1.0 default).
    _tail_n = apply_tail_downweight(
        examples, [_h[4] for _h in hist][:len(examples)],
        int(getattr(cfg, "tail_ply", 10 ** 9) or 10 ** 9),
        float(getattr(cfg, "tail_weight", 1.0) or 1.0))
    # outcome label
    if w_result == 1.0:
        res = "1-0"
    elif w_result == -1.0:
        res = "0-1"
    else:
        res = "1/2-1/2"
    if stats is not None:
        stats["tactics"] = stats.get("tactics", 0) + n_tac
        stats["vetoes"] = stats.get("vetoes", 0) + n_veto
        stats["finishes"] = stats.get("finishes", 0) + n_fin
        stats["safety"] = stats.get("safety", 0) + n_safe
        # v18 fast/full telemetry (value-data vs policy-data mix).
        stats["fast"] = stats.get("fast", 0) + (0 if _is_full_game else 1)
        # v13 honest compass: how this game ended (mate? adjudicated?
        # resign? rules-draw? cap?). Workers aggregate into T* counters.
        stats["terminal"] = getattr(state, "last_terminal", None) or "cap"
        # v14 visit breadth: mean distinct root moves visited per search.
        stats["breadth"] = breadth_acc / max(b_n, 1)
        stats["ventropy"] = vent_acc / max(b_n, 1)
        # v14 pace metric: total plies (training data per wall hour needs
        # games x plies, not just games — answers "is Rust faster" cleanly).
        stats["plies"] = int(state.ply_count)
        stats["tail_downweighted"] = int(_tail_n)
        _v20_note_stats(stats, state.board)
        _note_movelog(stats, examples)
    _gg.TRUNCATE_ENDINGS = _truncate_saved
    return examples, res



def play_game(model, cfg, device="cpu", temp_moves=20, opening_random_moves=0,
              resign_threshold=None, resign_moves=3, sparring=None,
              spar_white=True, evaluate_fn=None,
              stats: dict | None = None,
              full_playout: bool = False,
              fast: bool = False,
              start_fen: str | None = None,
              playthrough: bool = False) -> tuple[list, str]:
    """Thin wrapper owning the v13 TRUNCATE_ENDINGS kill-switch with
    try/finally: an exception mid-game must not leak adjudication-off
    into later games sharing the worker process (silent corruption).
    Identical signature to the old play_game; all callers unchanged.
    V20 adds start_fen (D4 endgame start: contempt forced 0, no book) and
    playthrough (D5: resign ignored entirely); both default off."""
    import chess_zero.game as _ggw
    _saved = _ggw.TRUNCATE_ENDINGS
    if full_playout:
        _ggw.TRUNCATE_ENDINGS = False
    try:
        return _play_game_impl(
            model, cfg, device=device, temp_moves=temp_moves,
            opening_random_moves=opening_random_moves,
            resign_threshold=resign_threshold, resign_moves=resign_moves,
            sparring=sparring, spar_white=spar_white,
            evaluate_fn=evaluate_fn, stats=stats,
            full_playout=full_playout, fast=fast,
            start_fen=start_fen, playthrough=playthrough)
    finally:
        _ggw.TRUNCATE_ENDINGS = _saved


if __name__ == "__main__":
    from .config import LOCAL_CONFIG
    from .model import AlphaZeroNet
    cfg = LOCAL_CONFIG
    net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels)
    ex, res = play_game(net, cfg, temp_moves=4)
    print(f"demo game: {len(ex)} positions, result {res}")

