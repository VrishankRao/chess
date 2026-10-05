"""Original evaluation: random + greedy-material baselines, win-rate, Elo diff.
Only win-rate vs baselines counts as improvement (falling loss does not).
"""
from __future__ import annotations

import math
import numpy as np
import chess

from .game import State, decode_action, PIECE_VALUES


def random_move(state: State) -> int:
    import random
    return random.choice(state.legal_moves())


def greedy_move(state: State) -> int:
    """1-ply material greedy from side-to-move view."""
    best, best_a = -1e18, None
    for a in state.legal_moves():
        nxt = state.apply(a)
        done, z = nxt.is_terminal()
        if done:
            # z is from opponent's view; our score is the negation
            score = -z * 10000.0
        else:
            score = _material_diff_for(nxt, state.board.turn)
        if score > best:
            best, best_a = score, a
    return best_a


def punisher_move(state: State) -> int:
    """v9 king-safety pressure: prefers checking moves (greedy score plus a
    sub-pawn check bonus, so checks win ties but never override material —
    unsound sacrifices would teach refutation, not fear). Greedy never
    checks an uncastled king, so nothing in sparring has ever punished king
    walks — this one does, whenever a check doesn't hang material.
    Explicit handcrafted deviation for training pressure only; never used
    in arena measurement. Mate delivery outranks everything (via the same
    terminal scoring greedy uses)."""
    best, best_a = -1e18, None
    for a in state.legal_moves():
        nxt = state.apply(a)
        done, z = nxt.is_terminal()
        if done:
            score = -z * 10000.0
        else:
            score = _material_diff_for(nxt, state.board.turn)
            if nxt.board.is_check():
                score += 0.5  # check-seeking tiebreak (< any pawn)
        if score > best:
            best, best_a = score, a
    return best_a


def sparring_for_game(i: int, kind: str | None = None):
    """Deterministic sparring mix (v9): even games greedy, odd punisher.
    Unit-testable; all three sparring loops use it so the mix is uniform
    regardless of worker chunking. kind=None = yesterday's even/odd
    exactly (frozen by test_full); kind in ("greedy","punisher","noisy",
    "tactics") selects one sparring policy directly (D14)."""
    if kind is None:
        return greedy_move if i % 2 == 0 else punisher_move
    if kind == "greedy":
        return greedy_move
    if kind == "punisher":
        return punisher_move
    if kind == "noisy":
        return noisy_move
    if kind == "tactics":
        return tactics_move
    raise ValueError(f"unknown sparring kind {kind!r}")


def noisy_move(state: State, p: float = 0.15) -> int:
    """D14 noisy sparring: uniform-random legal with prob p, else greedy.
    Temperature-free weakling for robustness coverage (training pressure
    only; never in arena measurement)."""
    import random as _r
    try:
        if _r.random() < float(p):
            return _r.choice(state.legal_moves())
    except Exception:
        pass
    return greedy_move(state)


def tactics_move(state: State) -> int:
    """D14 tactics-checker sparring: mate-in-1 if available, else the
    highest-MVV capture, else greedy. Punishes hanging material and
    rewards conversion (training pressure only). Never raises: any
    inspection failure falls back to greedy."""
    try:
        legal = state.legal_moves()
        for a in legal:
            try:
                nxt = state.apply(a)
                done, z = nxt.is_terminal()
            except Exception:
                continue
            if done and -z == 1.0:
                return a  # mate delivery outranks everything
        best, best_v = None, -1.0
        for a in legal:
            try:
                mv = state.to_move(a)
            except Exception:
                continue
            try:
                is_cap = bool(state.board.is_capture(mv))
            except Exception:
                is_cap = False
            if not is_cap:
                continue
            try:
                if bool(state.board.is_en_passant(mv)):
                    vv = float(PIECE_VALUES.get(chess.PAWN, 1.0))
                else:
                    _pc = state.board.piece_at(mv.to_square)
                    vv = float(PIECE_VALUES.get(
                        _pc.piece_type, 0.0)) if _pc is not None else 0.0
            except Exception:
                vv = 0.0
            if vv > best_v:
                best, best_v = a, vv
        if best is not None:
            return best
    except Exception:
        pass
    return greedy_move(state)


def sparring_kind_for_game(i: int) -> str:
    """D14 sparring rotation per iter: 60% punisher / 20% noisy /
    20% tactics, deterministic by game index (the v9 even/odd pattern
    generalized). Over 10 games: 6 punisher, 2 noisy, 2 tactics."""
    m = int(i) % 10
    if m < 6:
        return "punisher"
    if m < 8:
        return "noisy"
    return "tactics"


def eps_greedy_move(eps: float):
    """Tunable weakling: greedy with prob eps, else uniform random.
    eps=1.0 -> greedy, eps=0.0 -> random. Fills the gap between them."""
    def fn(state: State) -> int:
        import random as _r
        if _r.random() < eps:
            return greedy_move(state)
        return _r.choice(state.legal_moves())
    fn._spec = ("eps", eps)
    return fn


def _material(board_state: State) -> float:
    b = board_state.board
    tot = 0.0
    for pt, v in PIECE_VALUES.items():
        tot += v * len(b.pieces(pt, chess.WHITE))
        tot -= v * len(b.pieces(pt, chess.BLACK))
    return tot


def _material_diff_for(nxt: State, us) -> float:
    m = _material(nxt)
    # from `us` perspective (us=True white)
    if us == chess.WHITE:
        # after our move it's opponent to move; material diff white-black favors us
        return m
    return -m


def adjudicate_material(state: State, margin: float = 3.0) -> float:
    """Training-style adjudication from side-to-move view. Used when arena
    hits the move cap without a rules terminal so Elo signal stays decisive."""
    m = _material(state)
    us_white = state.board.turn == chess.WHITE
    diff = m if us_white else -m
    if abs(diff) >= margin:
        return 1.0 if diff > 0 else -1.0
    return 0.0


def _uci_queen(state, mv: int) -> str:
    """Action int -> absolute UCI (v9 flip: unmirrors Black-stm actions via
    State.to_uci). Takes the STATE, not the bare board — the turn decides
    the orientation. Queen-promoting pawns get the q suffix (queen-only
    codec has no underpromotions, so bare UCI won't replay)."""
    return state.to_uci(mv)


@__import__("chess_zero.game", fromlist=["standard_rules"]).standard_rules
def _play_ids(policy_a, policy_b, ids, cap=200, opening_moves=0,
              record=False, seed_base=1000, opening_lines=None) -> dict:
    """Core match loop over explicit game ids. A is White on even ids.
    opening_moves: per game (Sebastian's match-manager principle: never test
    from startpos only). PAIRED by id (audit-3.2): games 2k and 2k+1 share
    one seeded opening with colours reversed, removing most opening-luck
    variance at zero extra cost. seed_base varies the sample across
    iterations (audit round 2: a fixed base froze the same 10-20 openings
    for the whole run — an overfitting channel); callers pass an
    iter-derived base so pairs match within a match but vary across time.
    record: also return per-game (moves_uci, a_won_or_None, a_white) for
    PGN autopsy of losses."""
    wins = losses = draws = 0
    records = []
    for g in ids:
        for p in (policy_a, policy_b):
            r = getattr(p, "reset", None)
            if callable(r):
                r()
        s = State.initial()
        moves = []
        import random as _rr
        _rr = _rr.Random(seed_base + (g // 2))
        line = opening_lines[(g // 2) % len(opening_lines)] if opening_lines else None
        for j in range(len(line) if line is not None else opening_moves):
            if s.is_terminal()[0] or s.ply_count >= cap:
                break
            if line is None:
                om = _rr.choice(s.legal_moves())
            else:
                target = s.board.parse_uci(line[j])
                om = next(a for a in s.legal_moves() if s.to_move(a) == target)
            if record:
                moves.append(_uci_queen(s, om))
            s = s.apply(om)
        # v12: seed policy repetition histories with the opening prefix.
        # State._hist threads all played keys (openings included) for planes
        # 14-15 and threefold; policies' in-search loop scoring (history)
        # must see the same prefix or loop-lines repeating an opening slip
        # through as non-draws (self-play already appends openings to
        # rep_hist — arena measured a different engine until now).
        try:
            _prefix = list(s._hist)
            for _p in (policy_a, policy_b):
                _h = getattr(_p, "_hist", None)
                if isinstance(_h, list):
                    _h.extend(_prefix)
        except Exception:
            pass
        swap = (g % 2 == 1)
        while True:
            done, z = s.is_terminal()
            if done or s.ply_count >= cap:
                break
            white_to_move = s.board.turn == chess.WHITE
            # A is White on even games (swap=False)
            a_turn = (white_to_move and not swap) or ((not white_to_move) and swap)
            mv = policy_a(s) if a_turn else policy_b(s)
            if record:
                moves.append(_uci_queen(s, mv))
            s = s.apply(mv)
        done, z_stm = s.is_terminal()
        outcome = None  # from A's view: True win / False loss / None draw
        if not done:
            # arena adjudication at cap (keeps signal decisive)
            z_stm = adjudicate_material(s)
            if z_stm == 0.0:
                draws += 1
            else:
                stm_is_white = s.board.turn == chess.WHITE
                white_won = stm_is_white if z_stm == 1.0 else not stm_is_white
                a_won = (white_won and not swap) or ((not white_won) and swap)
                if a_won:
                    wins += 1
                    outcome = True
                else:
                    losses += 1
                    outcome = False
            if record:
                records.append((moves, outcome, not swap))
            continue
        if z_stm == 0.0:
            draws += 1
            if record:
                records.append((moves, None, not swap))
            continue
        # winner is the side that just moved = opposite of side-to-move,
        # but only when the terminal value is decisive (z_stm = -1: stm lost).
        stm_is_white = s.board.turn == chess.WHITE
        if z_stm == -1.0:
            white_won = not stm_is_white
        elif z_stm == 1.0:
            white_won = stm_is_white
        else:
            draws += 1
            if record:
                records.append((moves, None, not swap))
            continue
        a_won = (white_won and not swap) or ((not white_won) and swap)
        if a_won:
            wins += 1
            outcome = True
        else:
            losses += 1
            outcome = False
        if record:
            records.append((moves, outcome, not swap))
    out = {"wins": wins, "losses": losses, "draws": draws}
    return (out, records) if record else out


def play_match(policy_a, policy_b, games=4, cap=200, opening_moves=0,
               record=False, seed_base=1000, opening_lines=None) -> dict:
    """policy(state)->action. A plays White on even games."""
    return _play_ids(policy_a, policy_b, range(games), cap, opening_moves,
                     record, seed_base, opening_lines)


def _spec(p) -> tuple:
    if hasattr(p, "_spec"):
        return p._spec  # agent closures stamp this at construction
    return ("fn", p.__module__, p.__qualname__)


def _make_policy(spec, server_conn=None):
    # server_conn (phase 3, opt-in): pipe to a batched inference server.
    # Spec carries weights+dims for the eager local fallback (built here,
    # same as today). Default None = today's path exactly.
    kind = spec[0]
    if kind == "fn":
        import importlib
        mod = importlib.import_module(spec[1])
        fn = mod
        for part in spec[2].split("."):
            fn = getattr(fn, part)
        return fn
    if kind == "eps":
        return eps_greedy_move(spec[1])
    if kind == "agent":
        _, weights, blocks, channels, sims, c_puct, alpha = spec[:7]
        noise = spec[7] if len(spec) > 7 else 0.0
        temp_moves = spec[8] if len(spec) > 8 else 0
        planes = spec[9] if len(spec) > 9 else 13
        use_tt = spec[10] if len(spec) > 10 else True
        # v9 search contempt, stamped by loop.agent_policy/_gate/_arena.
        # Absent on old specs (pre-v9 checkpoints): pure 0.0 measurement.
        contempt = spec[11] if len(spec) > 11 else 0.0
        asymmetric = spec[12] if len(spec) > 12 else False
        # v9 product guards (tactical force + blunder veto), stamped the
        # same way. Absent = off = yesterday's unguarded measurement.
        tac_on = spec[13] if len(spec) > 13 else False
        veto_on = spec[14] if len(spec) > 14 else False
        tac_thr = spec[15] if len(spec) > 15 else 0.09
        qdepth = spec[16] if len(spec) > 16 else 0
        fbonus = spec[17] if len(spec) > 17 else 0.0
        edge_scale = spec[18] if len(spec) > 18 else 0.0
        # v15 finishing: stamped by loop.agent_policy/_gate/_arena.
        # Absent = off (yesterday's measurement). Server mode has no
        # local model, so finishing stays off there (documented).
        mate_on = spec[19] if len(spec) > 19 else False
        # v16.1 batched leaves. Absent = 1 = legacy sequential search.
        leaf_batch = spec[20] if len(spec) > 20 else 1
        vl = spec[21] if len(spec) > 21 else 1.0
        # v15.5 safety veto. Absent = off (yesterday's measurement).
        safety_on = spec[22] if len(spec) > 22 else False
        # v19: threshold travels with the flag (deploy/parallel parity —
        # bare namespaces used to silently pin 0.15).
        safety_thr = spec[26] if len(spec) > 26 else 0.15
        # V21.1 E4 rust-tree flag (spec[27], stamped by loop._gate/_arena/
        # _progress_check parallel specs; absent = off = yesterday).
        rust_tree = bool(spec[27]) if len(spec) > 27 else False
        import torch as _torch
        import numpy as _np
        from .model import AlphaZeroNet, load_weights
        from .selfplay import make_evaluate
        from . import mcts as _mcts
        if isinstance(weights, str):
            weights = _torch.load(weights, map_location="cpu",
                                  weights_only=False)
        # v18 SE trunk. Absent = plain blocks (yesterday's arch).
        # Old specs on new SE weights still work: infer from the LOADED
        # dict (infer on a path string always misses — load first).
        se_ratio = spec[23] if len(spec) > 23 else 0
        if not se_ratio:
            from .model import infer_se_ratio as _isr
            try:
                se_ratio = _isr(weights)
            except Exception:
                se_ratio = 0
        # v19 search knobs for parallel rebuilds (absent = yesterday).
        fpu = spec[24] if len(spec) > 24 else 0.0
        prune = spec[25] if len(spec) > 25 else False
        # Worker-fresh process: encode() must emit what this net eats.
        from . import game as _game_mod
        _game_mod.INPUT_PLANES = planes
        if server_conn is not None:
            # Server mode: no local inference model (the client factory
            # builds the lightweight fallback itself).
            from .infer_server import make_server_evaluate
            evaluate = make_server_evaluate(server_conn, weights, blocks,
                                            channels, planes, device="cpu")
            model = evaluate.model
        else:
            model = AlphaZeroNet(blocks=blocks, channels=channels,
                                 planes=planes, se_ratio=se_ratio)
            load_weights(model, weights, strict=True)
            evaluate = make_evaluate(model, device="cpu")

        def fn(state):
            fn._hist[:] = list(state._hist)
            from .selfplay import tactical_action as _tac
            from .selfplay import apply_move_guards as _guards
            from .selfplay import finish_move as _fin
            from .selfplay import safety_guard as _safe
            import types as _types
            tt = fn._tt if use_tt else None
            # V21.1 E4: identical params either side of the seam.
            # F1/F2: rust path threads the per-game GameStore (dict stays
            # empty); otherwise the worker dict.
            if rust_tree and use_tt and \
                    getattr(fn, "_rust_store", None) is not None:
                tt = fn._rust_store
            _kw = dict(c_puct=c_puct,
                       dirichlet_alpha=alpha, dirichlet_eps=noise,
                       tt=tt, history=fn._hist,
                       contempt=contempt,
                       asymmetric_contempt=asymmetric,
                       quiescence_depth=qdepth,
                       forcing_bonus=fbonus,
                       contempt_edge_scale=edge_scale,
                       leaf_batch=leaf_batch,
                       virtual_loss=vl,
                       fpu_reduction=fpu,
                       prune_singletons=prune,
                       futile_stop=False, ml_fn=getattr(evaluate, "ml_fn", None),
                       ml_slope=spec[28] if len(spec)>28 else 0.0,
                       ml_cap=spec[29] if len(spec)>29 else 0.07,
                       ml_thr=spec[30] if len(spec)>30 else 0.8)
            if rust_tree:
                from .mcts_bridge import RustTreeBackend as _RTB
                pi = _RTB().search(state, evaluate, n_sims=sims,
                                   **_kw)[0]
            else:
                pi = _mcts.search(state, evaluate, n_sims=sims, **_kw)
            fn._hist.append(state.rep_key())
            # plies (audit-1.8), matching self-play's temp_moves units.
            tac = _tac(state, tac_thr) if tac_on else None
            if tac is None and temp_moves and state.ply_count < temp_moves:
                legal = _np.flatnonzero(pi)
                tot = pi[legal].sum()
                if tot > 0:
                    probs = pi[legal] / tot
                    choice = int(_np.random.choice(legal, p=probs))
                    choice, _, _vet = _guards(state, choice, pi, None,
                                              veto_on, tac_thr)
                    # v15.5 safety veto (never second-guesses tac/veto).
                    if safety_on and tac is None and not _vet and \
                            model is not None:
                        choice, _sv = _safe(
                            state, choice, pi, model, "cpu",
                            _types.SimpleNamespace(
                                safety_veto=True,
                                safety_drop_thr=safety_thr))
                        if _sv:
                            pi = _np.zeros(4096, dtype=_np.float32)
                            pi[int(choice)] = 1.0
                    if mate_on and model is not None and not _vet:
                        choice = _fin(
                            state, choice, pi, model, "cpu",
                            _types.SimpleNamespace(mate_finish=True))
                    try:
                        _rs = getattr(fn, "_rust_store", None)
                        if rust_tree and use_tt and _rs is not None:
                            _rs.prune_to(state.apply(choice).key())
                        else:
                            _mcts.prune_tt(fn._tt,
                                           state.apply(choice).key())
                    except Exception:
                        pass
                    return choice
            choice, _, _vet = _guards(state, int(_np.argmax(pi)), pi, tac,
                                      veto_on, tac_thr)
            # v15.5 safety veto AFTER guards (soundness first; never
            # overrides a forced tactic or veto replacement). One-hots on
            # replace so finishing below composes (single candidate).
            if safety_on and tac is None and not _vet and \
                    model is not None:
                choice, _sv = _safe(
                    state, choice, pi, model, "cpu",
                    _types.SimpleNamespace(
                        safety_veto=True,
                        safety_drop_thr=safety_thr))
                if _sv:
                    pi = _np.zeros(4096, dtype=_np.float32)
                    pi[int(choice)] = 1.0
            # v15 finishing AFTER guards (soundness first; never
            # overrides a forced tactic or veto replacement).
            if mate_on and tac is None and not _vet and \
                    model is not None:
                import types as _types2
                choice = _fin(state, choice, pi, model, "cpu",
                              _types2.SimpleNamespace(mate_finish=True))
            try:
                _rs = getattr(fn, "_rust_store", None)
                if rust_tree and use_tt and _rs is not None:
                    _rs.prune_to(state.apply(choice).key())
                else:
                    _mcts.prune_tt(fn._tt, state.apply(choice).key())
            except Exception:
                pass
            return choice
        fn._tt = {}
        fn._hist = []
        # F1: one Rust-side GameStore per worker policy when the rust-tree
        # seam is on (cleared on reset); None = Python path.
        fn._rust_store = None
        if rust_tree:
            try:
                from .mcts_bridge import new_game_store as _ngs
                fn._rust_store = _ngs()
            except Exception:
                fn._rust_store = None

        def _reset():
            try:
                fn._tt.clear()
            except Exception:
                pass
            try:
                fn._hist.clear()
            except Exception:
                pass
            try:
                _rs = getattr(fn, "_rust_store", None)
                if _rs is not None:
                    _rs.clear()
            except Exception:
                pass
        fn.reset = _reset
        return fn
    raise ValueError(f"unknown policy spec {spec[0]}")


def elo_diff(wins: int, losses: int, draws: int) -> float:
    n = wins + losses + draws
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    score = min(max(score, 1e-6), 1 - 1e-6)
    return 400 * math.log10(score / (1 - score))


if __name__ == "__main__":
    r = play_match(random_move, greedy_move, games=4)
    print("random vs greedy:", r, "elo(A-random)=", round(elo_diff(**r), 1))
