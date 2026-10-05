"""UCI wrapper so the AlphaZero agent plugs into lichess-bot / GUIs.
Implements: uci, isready, ucinewgame, position [startpos|fen] [moves ...],
go [movetime|wtime...], quit. Replies bestmove <uci> (including underpromotions).
Time-managed: spends time/30 + 60% increment per move (capped at time/4),
calibrating sims from measured ms/sim. No clock -> fixed --sims.
Usage: python3 -m chess_zero.uci --ckpt live.pt --sims 100
"""
from __future__ import annotations

import argparse
import sys
import time as _t

import chess
import numpy as np


def parse_go(params: list[str], board, sim_cap: int):
    """Returns (sims, deadline_epoch or None)."""
    d = {}
    it = iter(params)
    for tok in it:
        if tok in ("wtime", "btime", "winc", "binc", "movestogo", "movetime",
                   "depth", "nodes"):
            try:
                d[tok] = int(next(it))
            except StopIteration:
                break
    if "nodes" in d:
        sim_cap = max(1, d["nodes"])
    if "infinite" in params:
        sim_cap = 2**31 - 1
    if "movetime" in d:
        return sim_cap, _t.time() + d["movetime"] / 1000.0
    ours = "wtime" if board.turn == chess.WHITE else "btime"
    inc = "winc" if board.turn == chess.WHITE else "binc"
    if ours not in d:
        return sim_cap, None
    ms_left = d[ours]
    # Spend a fair share of remaining time + 60% of increment per move,
    # capped at 25% of the clock and 8s (v14: 400-sim searches cost
    # ~3.3s on CPU; the cap must clear a full search with headroom for
    # human games). Honors movestogo when the GUI sends it (tournament
    # TC); falls back to ms/20 for sudden-death.
    mtg = d.get("movestogo")
    share = ms_left / max(1, mtg) if mtg else ms_left / 20.0
    budget_ms = min(share + 0.6 * d.get(inc, 0), ms_left / 4.0, 8000.0)
    return sim_cap, _t.time() + max(0.05, budget_ms / 1000.0)


def apply_uci_moves(start_board, moves):
    """Replay UCI moves through State.apply (NOT board.push) so the
    repetition history counter accumulates. Returns (state, prefix_keys).
    A FEN start has unknown history (zeros — the truthful value); a
    startpos replay carries exact counts, so planes 14-15 and threefold
    logic match training (audit round 2 §1.2: the old push-based path
    played with zeroed planes and no loop penalty — a different engine)."""
    from .game import State as _S, flip_action as _flip
    import chess as _c
    st = _S(start_board.copy(stack=False))
    keys = []
    for u in moves:
        m = st.board.parse_uci(u)
        keys.append(st.rep_key())
        a = __import__("chess_zero.game", fromlist=["encode_action"]).encode_action(m.from_square, m.to_square, m.promotion)
        if st.board.turn == _c.BLACK:
            # v9 flip: UCI is absolute, apply() takes stm-oriented.
            a = _flip(a)
        st = st.apply(a)
    return st, keys


def calibrate_q(q, elo: float = 0.0):
    """V20 D11 WDL rescale: per-net calibration in Elo (default 0 = off =
    yesterday bit-exact). An Elo edge E scales W/L odds by 10^(E/400); applied
    here as a logit-space shift E*ln(10)/400 on Q (atanh -> shift -> tanh),
    exact for decisive mass and first-order with draws in the denominator.
    Positive E trusts the net's wins more (corrects underconfidence),
    negative softens. Pure numpy, scalar or array, output clipped to [-1, 1].
    """
    import numpy as _np
    _e = float(elo or 0.0)
    _a = _np.asarray(q, dtype=_np.float64)
    if _e == 0.0:
        return _a
    _c = _np.clip(_a, -1.0 + 1e-9, 1.0 - 1e-9)
    _shift = _e * 2.302585092994046 / 400.0  # ln(10)/400 per Elo point
    return _np.tanh(0.5 * _np.log((1.0 + _c) / (1.0 - _c)) + _shift)


def _uci_ml_fn(eng):
    """v19 deploy ML bonus source: moves-left head of the loaded model
    (normalized units, same convention as training ml_fn). None when no
    model (server path) — bonus off."""
    _m = eng.get("model")
    if _m is None:
        return None
    import torch as _t
    import numpy as _np
    try:
        _dev = next(_m.parameters()).device
    except Exception:
        _dev = "cpu"

    def _fn(states):
        _x = _t.stack(
            [_t.from_numpy(_np.asarray(s.encode(), dtype=_np.float32))
             for s in states]).to(_dev)
        with _t.no_grad():
            _out = _m(_x)
        return _np.asarray(_out[3].detach().cpu(),
                           dtype=_np.float64).flatten()
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--planes", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--contempt", type=float, default=0.3,
                    help="v9 search contempt (draw lines score ±contempt by "
                         "mover's material edge); v20/D11 default = training "
                         "contempt 0.3 asymmetric")
    ap.add_argument("--wdl-elo", type=float, default=0.0,
                    help="v20/D11 WDL rescale: per-net calibration in Elo "
                         "(0 = off = yesterday; + sharpens, - softens)")
    ap.add_argument("--asymmetric-contempt", action="store_true",
                    help="ahead dislikes draws, behind likes them (v9)")
    ap.add_argument("--qdepth", type=int, default=2,
                    help="v10 quiescence: extra plies answering checks")
    ap.add_argument("--fbonus", type=float, default=0.25,
                    help="v10 forcing-first prior boost")
    ap.add_argument("--edge-scale", type=float, default=0.0,
                    help="v11 edge-scaled contempt magnitude (pawns for full "
                         "contempt; 0 = legacy fixed magnitude)")
    ap.add_argument("--mate-finish", action="store_true",
                    help="v15 ML-guided finishing: when clearly ahead, "
                         "prefer the searched move minimizing predicted "
                         "remaining length. Default OFF (old weights have "
                         "untested ml heads); the mac launcher enables it.")
    ap.add_argument("--book", default="",
                     help="v16 opening book (JSON lines file, or BUILTIN): "
                          "follow a random matching ECO line while on-book, "
                          "search on divergence. Startpos games only.")
    ap.add_argument("--leaf-batch", type=int, default=1,
                     help="v16.1 batched-leaf MCTS (shared forwards, "
                          "virtual loss); 1 = legacy sequential")
    ap.add_argument("--virtual-loss", type=float, default=1.0)
    ap.add_argument("--c-puct", type=float, default=1.414,
                     help="v19 PUCT exploration constant (training uses 1.6)")
    ap.add_argument("--fpu-reduction", type=float, default=0.0,
                     help="v19 first-play urgency reduction (0.0 = off, "
                          "training uses 0.5)")
    ap.add_argument("--prune-singletons", action="store_true",
                     help="v19 KataGo-lite: drop visits<=1 before normalize")
    ap.add_argument("--safety-veto", action="store_true",
                     help="v15.5 king-safety veto: reject picks that crater "
                          "predicted own-king safety (self-gating while the "
                          "head is untrained)")
    ap.add_argument("--safety-drop-thr", type=float, default=0.15)
    ap.add_argument("--ml-slope", type=float, default=0.0,
                      help="v19 in-tree moves-left bonus slope (0 = off; "
                           "V19 training uses 0.003)")
    ap.add_argument("--ml-thr", type=float, default=0.9,
                      help="v20/D11 MLH gate threshold (V19 0.8 -> V20 0.9; "
                           "bonus fires only when |parent Q| >= thr)")
    ap.add_argument("--ml-cap", type=float, default=0.07,
                      help="v19 in-tree moves-left bonus cap (V20 keeps 0.07)")
    ap.add_argument("--veto-thr", type=float, default=0.09,
                      help="v20/D11 blunder-veto hang threshold (pawns); "
                           "0.09 = pre-recal placeholder — recal_veto.py "
                           "sets the recalibrated default after D6 rescore "
                           "lands (R4), pass explicitly to override")
    ap.add_argument("--se-ratio", type=int, default=0,
                     help="v18 SE trunk ratio (0 = plain blocks); must match "
                          "the checkpoint (v18 nets use 4)")
    a = ap.parse_args()
    # Answer the handshake INSTANTLY (GUIs time out on slow torch imports),
    # then lazy-load the heavy stack on first use.
    out = sys.stdout
    eng = {}
    # v12: persistent transposition table across moves (subtree reuse).
    # Same discipline as training (loop.agent_policy): search promotes the
    # picked child, prune_tt keeps ONLY that subtree (bounds memory, kills
    # history-merge hazard). Opponent deviations naturally miss and clear.
    eng["tt"] = {}
    # GUI-debugging transcript: every command in, every reply out, with
    # timestamps. Never allowed to break the engine (wrapped try/except).
    import os as _os

    def _log(msg):
        try:
            with open(_os.path.expanduser("~/.chaturanga_uci.log"),
                      "a") as _f:
                _f.write(f"{_t.time():.2f} {msg}\n")
        except Exception:
            pass

    def ensure_loaded():
        if "ev" in eng:
            return
        import torch
        from .game import State as _State
        from . import game as _game_mod
        from .model import AlphaZeroNet, load_inference_weights, infer_se_ratio
        from . import mcts as _mcts_mod
        from .selfplay import make_evaluate
        _game_mod.INPUT_PLANES = a.planes
        eng["State"] = _State
        eng["mcts"] = _mcts_mod
        _wuci = torch.load(a.ckpt, map_location=a.device,
                           weights_only=False)
        # v19 arch-follows-weights (explicit --se-ratio wins when set).
        _se_uci = int(getattr(a, "se_ratio", 0) or 0) or \
            infer_se_ratio(_wuci)
        model = AlphaZeroNet(blocks=a.blocks, channels=a.channels,
                             planes=a.planes,
                             se_ratio=_se_uci).to(a.device)
        load_inference_weights(model, _wuci)
        model.eval()
        eng["model"] = model
        eng["ev"] = make_evaluate(model, device=a.device)

    board = chess.Board()
    prefix_keys: list = []  # rep keys before the current position
    book_allowed = True
    played: list = []  # startpos UCI moves (for book matching)
    ms_per_sim = [15.0]  # online calibration of eval speed
    book_lines = []
    if a.book:
        try:
            from .book import load_book as _load_book
            book_lines = _load_book(a.book)
            _log(f"book: {len(book_lines)} lines")
        except Exception as e:
            _log(f"book load failed ({type(e).__name__}); playing bookless")

    def bestmove(params):
        ensure_loaded()
        State = eng["State"]
        mcts_mod = eng["mcts"]
        ev = eng["ev"]
        if board.is_game_over(claim_draw=True):
            out.write("bestmove 0000\n")
            out.flush()
            return
        # v20/D11 WDL rescale: calibrate leaf Q in Elo (0 = off, wrapper
        # skipped = yesterday bit-exact).
        _elo = float(getattr(a, "wdl_elo", 0.0) or 0.0)
        if _elo != 0.0:
            _base = ev

            def ev(states, _b=_base, _e=_elo):  # noqa: F811
                _pr, _vv = _b(states)
                return _pr, calibrate_q(_vv, _e)
        # v16 book: instant reply while a matching ECO line continues
        # (startpos games only, move 1 included; divergence falls
        # through to search). Unseeded choice = varied repertoire.
        if book_lines and book_allowed:
            try:
                from .book import book_move as _bm
                _next = _bm(book_lines, played)
                if _next is not None and chess.Move.from_uci(_next) \
                        in board.legal_moves:
                    out.write(f"bestmove {_next}\n")
                    out.flush()
                    _log(f"OUT: bestmove {_next} (book)")
                    return
            except Exception:
                pass
        if "depth" in params:
            out.write("info string depth limits are unsupported; use nodes or movetime\n")
        sims, deadline = parse_go(params, board, a.sims)
        st = State(board.copy(stack=False))
        # re-attach this game's history so counts/loops match training
        st._hist = tuple(prefix_keys)
        t0 = _t.time()
        if deadline is not None:
            budget_ms = max(50.0, (deadline - t0) * 1000)
            sims = max(8, min(sims, int(budget_ms / ms_per_sim[0])))
        search_stats = {}
        pi = mcts_mod.search(st, ev, stats=search_stats, n_sims=sims, dirichlet_eps=0.0,
                             c_puct=float(getattr(a, "c_puct", 1.414)),
                             tt=eng.get("tt"),
                             history=list(prefix_keys),
                             contempt=a.contempt,
                             asymmetric_contempt=a.asymmetric_contempt,
                             quiescence_depth=a.qdepth,
                             forcing_bonus=a.fbonus,
                             contempt_edge_scale=a.edge_scale,
                             leaf_batch=int(getattr(
                                 a, "leaf_batch", 1) or 1),
                             virtual_loss=float(getattr(
                                 a, "virtual_loss", 1.0) or 1.0),
                             fpu_reduction=float(getattr(
                                 a, "fpu_reduction", 0.0) or 0.0),
                             prune_singletons=bool(getattr(
                                 a, "prune_singletons", False)),
                              ml_fn=getattr(eng["ev"], "ml_fn", None),
                              ml_slope=float(getattr(
                                  a, "ml_slope", 0.0) or 0.0),
                              ml_cap=float(getattr(a, "ml_cap", 0.07)
                                           if getattr(a, "ml_cap", 0.07)
                                           is not None else 0.07),
                              ml_thr=float(getattr(a, "ml_thr", 0.9)
                                           if getattr(a, "ml_thr", 0.9)
                                           is not None else 0.9),
                              futile_stop=False,
                              deadline=(_t.monotonic() + max(0.0, deadline-_t.time()) if deadline is not None else None),
                              stop_event=stop_event)
        dt_ms = max(1.0, (_t.time() - t0) * 1000)
        if search_stats.get("new_sims", 0) > 0:
            ms_per_sim[0] = 0.9 * ms_per_sim[0] + 0.1 * (dt_ms / search_stats["new_sims"])
        # v9 product guards (training-identical): forced tactics first,
        # then blunder veto on the argmax pick. Deployment plays the
        # GUARDED policy — films measure the product. (Deliberate doctrine
        # change; battery/calibration stay unguarded pure measurement.)
        from .selfplay import (tactical_action as _tac,
                               apply_move_guards as _guards,
                               finish_move as _fin,
                               safety_guard as _safe)
        _expired = stop_event.is_set() or (deadline is not None and _t.time() >= deadline)
        tac = None if _expired else _tac(st, float(getattr(a, "veto_thr", 0.09) or 0.09))
        act, _, _vetoed = _guards(st, int(np.argmax(pi)), pi, tac, not _expired,
                                  float(getattr(a, "veto_thr", 0.09)
                                        or 0.09))
        # v15.5 safety veto AFTER guards (never second-guesses tac/veto).
        # One-hots on replace so finishing below composes.
        if not _expired and tac is None and not _vetoed and a.safety_veto:
            import types as _types_s
            act, _sved = _safe(st, int(act), pi, eng.get("model"),
                               a.device,
                               _types_s.SimpleNamespace(
                                   safety_veto=True,
                                   safety_drop_thr=a.safety_drop_thr))
            if _sved:
                pi = np.zeros(4096, dtype=np.float32)
                pi[int(act)] = 1.0
        # v15 finishing AFTER guards (soundness first). Deployment
        # doctrine: always finish won games (flag-gated, mac launcher on).
        if not _expired and tac is None and not _vetoed and a.mate_finish:
            import types as _types
            act = _fin(st, int(act), pi, eng.get("model"), a.device,
                       _types.SimpleNamespace(mate_finish=True))
        final_act = int(act)
        mv = st.to_move(final_act)
        if mv not in board.legal_moves:
            # fallback: highest-pi legal action (never hang the bridge)
            for cand in np.argsort(-pi):
                m2 = st.to_move(int(cand))
                if m2 in board.legal_moves:
                    mv = m2
                    final_act = int(cand)
                    break
        # v12: keep ONLY the promoted child's subtree (training discipline).
        # Opponent moves arrive via the next `position`; a deviation misses
        # and prune clears — bounded, never stale.
        try:
            mcts_mod.prune_tt(eng.get("tt"), st.apply(final_act).key())
        except Exception:
            pass
        out.write(f"bestmove {mv.uci()}\n")
        out.flush()
        _log(f"OUT: bestmove {mv.uci()} (think {dt_ms:.0f} ms)")

    # Read protocol input while the main thread searches. The reader only
    # signals cancellation; all board/model mutations remain on this thread.
    import threading, queue
    stop_event = threading.Event()
    commands = queue.Queue()
    def read_commands():
        # A token belongs to a particular go command. A stop read before
        # the main thread dequeues go must still cancel that search.
        pending = None
        for line in sys.stdin:
            cmd = line.strip().split()[:1]
            if cmd == ["go"]:
                pending = threading.Event()
            elif cmd in (["stop"], ["quit"], ["position"], ["ucinewgame"]):
                if pending is not None:
                    pending.set()
            commands.put((line, pending))
        if pending is not None:
            pending.set()
        commands.put(None)
    threading.Thread(target=read_commands, daemon=True).start()
    from . import game as _rules
    _rules.TRUNCATE_ENDINGS = False
    for line, cancellation in iter(commands.get, None):
        p = line.strip().split()
        if not p:
            continue
        _log("IN: " + " ".join(p[:24]))
        if p[0] == "uci":
            out.write("id name ChaturangaZero-chess-v1\nid author team\n"
                      "uciok\n")
            out.flush()
        elif p[0] == "isready":
            ensure_loaded()
            out.write("readyok\n")
            out.flush()
        elif p[0] == "ucinewgame":
            book_allowed = True
            board = chess.Board()
            prefix_keys = []
            played = []
            try:
                eng.get("tt", {}).clear()
            except Exception:
                pass
        elif p[0] == "position":
            book_allowed = p[1] == "startpos"
            if p[1] == "startpos":
                board = chess.Board()
                i = 2
            else:  # fen ...
                board = chess.Board(" ".join(p[2:8]))
                i = 8
            if i < len(p) and p[i] == "moves":
                mvlist = p[i + 1:]
                if p[1] == "startpos":
                    st0, prefix_keys = apply_uci_moves(board, mvlist)
                    board = st0.board
                    played = list(mvlist)
                else:
                    st0, prefix_keys = apply_uci_moves(board, mvlist)
                    board = st0.board
                    played = []  # FEN games never touch the book
            else:
                prefix_keys = []  # bare position = fresh game
                played = []
                try:
                    eng.get("tt", {}).clear()
                except Exception:
                    pass
        elif p[0] == "go":
            stop_event = cancellation
            bestmove(p[1:])
        elif p[0] == "quit":
            break


if __name__ == "__main__":
    main()
