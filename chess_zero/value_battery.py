"""Per-checkpoint value/policy sanity battery.

Why this exists (audit round 3): the project's only strength instruments are
a 20-game gate with a ~36% null false-promotion rate and a 30-game greedy
arena on frozen openings. Neither can see a value head being erased. This
can: it asks the net about positions whose truth is not debatable, costs
~1 s per checkpoint, and gives a real per-iteration curve.

Two probes, reported separately:
  VALUE   9 positions with an unarguable result (up a queen, being mated
          next move, ...). Score = mean |value - truth| and sign accuracy.
          A functioning head scores well under 0.4; v8.5 iter1 scored 0.577
          against v8.4best's 0.275, which is how the iter-1 collapse was
          found.
  POLICY  4 positions with one free capture and no counterplay (P1 is the
          real 600-bot game position where v8.5 declined a hanging bishop).
          Reports whether the raw policy argmax and, with --sims, the search
          argmax find it. Stockfish depth 16 confirms every listed best move.

Usage:
  python3 -m chess_zero.value_battery --ckpt v85best.pt
  python3 -m chess_zero.value_battery --ckpt-dir checkpoints_v85 --sims 200
  python3 -m chess_zero.value_battery --ckpt-dir checkpoints_v85 --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import chess
import numpy as np
import torch

# (label, fen, truth from side-to-move view)
VALUE_POSITIONS = [
    ("startpos",            chess.STARTING_FEN,                            0.0),
    ("up a queen",          "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",             +1.0),
    ("down a queen",        "3qk3/8/8/8/8/8/8/4K3 w - - 0 1",             -1.0),
    ("up a rook (full)",
     "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          +1.0),
    ("down a rook (full)",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1",           -1.0),
    ("K+Q vs K, queen stm", "8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1",            +1.0),
    ("K+Q vs K, bare stm",  "8/8/8/4k3/8/8/4Q3/4K3 b - - 0 1",            -1.0),
    ("mate in 1 available", "6k1/5ppp/8/8/8/8/5PPP/6KQ w - - 0 1",        +1.0),
    ("being mated next",    "6k1/5ppp/8/8/8/8/5PPP/6KQ b - - 0 1",        -1.0),
]

# (label, fen, best move in UCI) — all confirmed by stockfish depth 16
POLICY_POSITIONS = [
    ("P1 free bishop (600-bot game, move 6)",
     "rnbqkb1r/1p2pppp/p2p1n2/1B6/3NP3/2N5/PPP2PPP/R1BQK2R b KQkq - 1 6",
     "a6b5"),
    ("P2 rook takes free queen",
     "4k3/8/8/8/8/8/4K3/q6R w - - 0 1", "h1a1"),
    ("P3 rook takes free queen",
     "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5"),
    ("P4 rook takes free knight",
     "4k3/8/8/4n3/8/8/K7/4R3 w - - 0 1", "e1e5"),
]


def _load(path, blocks, channels, planes, device):
    import chess_zero.game as _g
    _g.INPUT_PLANES = planes
    from .model import AlphaZeroNet, load_weights, infer_se_ratio
    from .selfplay import make_evaluate
    import torch as _t
    _w = _t.load(path, map_location="cpu", weights_only=False)
    m = AlphaZeroNet(blocks=blocks, channels=channels, planes=planes,
                     se_ratio=infer_se_ratio(_w))
    load_weights(m, _w, strict=True)
    m.eval()
    return make_evaluate(m, device)


def run_battery(evaluate, sims: int = 0, c_puct: float = 1.414) -> dict:
    """Returns {value_mae, value_sign_acc, policy_hits, search_hits, detail}."""
    from .game import State
    from . import mcts as mcts_mod
    detail = {"value": [], "policy": []}
    errs, signs = [], []
    for label, fen, truth in VALUE_POSITIONS:
        st = State(chess.Board(fen))
        _, v = evaluate([st])
        v = float(np.asarray(v).flatten()[0])
        errs.append(abs(v - truth))
        # truth 0.0 is unscored for sign (either sign is defensible)
        ok = True if truth == 0.0 else (np.sign(v) == np.sign(truth))
        signs.append(bool(ok))
        detail["value"].append({"pos": label, "truth": truth,
                                "value": round(v, 4), "sign_ok": bool(ok)})
    p_hits = s_hits = 0
    for label, fen, want_uci in POLICY_POSITIONS:
        board = chess.Board(fen)
        st = State(board.copy(stack=False))
        want = chess.Move.from_uci(want_uci)
        want_a = want.from_square * 64 + want.to_square
        if st.board.turn == chess.BLACK:
            # v9 flip: want_uci is absolute, policy space is stm-oriented.
            from .game import flip_action as _flip
            want_a = _flip(want_a)
        priors, _ = evaluate([st])
        legal = st.legal_moves()
        p_best = max(legal, key=lambda a: priors[0][a])
        p_ok = (p_best == want_a)
        p_hits += p_ok
        row = {"pos": label, "want": want_uci,
               "policy_move": st.to_uci(p_best),
               "policy_prior_on_best": round(float(priors[0][want_a]), 4),
               "policy_ok": bool(p_ok)}
        if sims:
            pi = mcts_mod.search(st, evaluate, n_sims=sims, c_puct=c_puct,
                                 dirichlet_eps=0.0)
            s_best = int(np.argmax(pi))
            s_ok = (s_best == want_a)
            s_hits += s_ok
            row["search_move"] = st.to_uci(s_best)
            row["search_ok"] = bool(s_ok)
        detail["policy"].append(row)
    return {"value_mae": round(float(np.mean(errs)), 4),
            "value_sign_acc": round(float(np.mean(signs)), 4),
            "policy_hits": p_hits, "policy_total": len(POLICY_POSITIONS),
            "search_hits": s_hits if sims else None,
            "detail": detail}


def _iter_num(path: str) -> int:
    m = re.search(r"iter(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[],
                    help="one or more checkpoint files")
    ap.add_argument("--ckpt-dir", default=None,
                    help="scan DIR/iter*.pt in numeric order")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--planes", type=int, default=21)
    ap.add_argument("--sims", type=int, default=0,
                    help="also run MCTS on the policy probes (0 = skip)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true",
                    help="print every position, not just the summary row")
    ap.add_argument("--json", default=None, help="write results to this file")
    a = ap.parse_args()

    paths = list(a.ckpt)
    if a.ckpt_dir:
        paths += sorted(glob.glob(os.path.join(a.ckpt_dir, "iter*.pt")),
                        key=_iter_num)
    if not paths:
        ap.error("pass --ckpt and/or --ckpt-dir")

    hdr = f"{'checkpoint':<28} {'val MAE':>8} {'sign':>6} {'policy':>7}"
    if a.sims:
        hdr += f" {'search':>7}"
    print(hdr)
    print("-" * len(hdr))
    out = []
    for p in paths:
        try:
            ev = _load(p, a.blocks, a.channels, a.planes, a.device)
        except Exception as e:
            print(f"{os.path.basename(p):<28} LOAD FAILED "
                  f"({type(e).__name__}: "
                  f"{' '.join(str(e).split())[:70]})")
            continue
        r = run_battery(ev, sims=a.sims)
        r["ckpt"] = p
        out.append(r)
        pol = "%d/%d" % (r["policy_hits"], r["policy_total"])
        line = ("%-28s %8.3f %5.0f%%  %6s"
                % (os.path.basename(p), r["value_mae"],
                   100 * r["value_sign_acc"], pol))
        if a.sims:
            line += "  %6s" % ("%d/%d" % (r["search_hits"],
                                          r["policy_total"]))
        print(line)
        if a.verbose:
            for d in r["detail"]["value"]:
                mark = "" if d["sign_ok"] else "   <-- WRONG SIGN"
                print(f"    {d['pos']:<24} truth {d['truth']:+.1f}  "
                      f"net {d['value']:+.3f}{mark}")
            for d in r["detail"]["policy"]:
                extra = f"  search {d['search_move']}" if a.sims else ""
                print(f"    {d['pos']:<40} want {d['want']}  "
                      f"policy {d['policy_move']} "
                      f"(prior on best {d['policy_prior_on_best']:.3f})"
                      f"{extra}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
