"""Greedy-hole diagnostics: play recorded games vs a baseline, report HOW we
lose (mated vs adjudicated, when material drops, game length) + dump PGNs.
Usage: python3 -m chess_zero.diagnose --ckpt checkpoints/iter13.pt --opp greedy --games 10 --sims 40
"""
from __future__ import annotations

import argparse
import json

import chess
import numpy as np
import torch

from .game import State, PIECE_VALUES
from .model import AlphaZeroNet
from . import mcts as mcts_mod
from .selfplay import make_evaluate
from .evaluate import random_move, greedy_move, adjudicate_material


def load_model(ckpt: str, blocks=4, channels=64, planes=21,
               device="cpu"):
    from .model import load_weights, infer_se_ratio
    import chess_zero.game as _g
    # v18: workers stamp INPUT_PLANES from cfg; this standalone loader
    # must too, or encode() emits the 13-plane default against a 21/30ch
    # net (shape crash).
    _g.INPUT_PLANES = planes
    _w = torch.load(ckpt, map_location=device, weights_only=False)
    m = AlphaZeroNet(blocks=blocks, channels=channels,
                     planes=planes, se_ratio=infer_se_ratio(_w)).to(device)
    load_weights(m, _w, strict=True)
    return m


def material_white(board: chess.Board) -> float:
    return sum(v * (len(board.pieces(pt, chess.WHITE))
                    - len(board.pieces(pt, chess.BLACK)))
               for pt, v in PIECE_VALUES.items())


def play_recorded(agent, opp, agent_white: bool, cap=200, sims_note=""):
    s = State.initial()
    sans: list[str] = []
    trace: list[float] = [material_white(s.board)]
    while True:
        done, _ = s.is_terminal()
        if done or s.ply_count >= cap:
            break
        white = s.board.turn == chess.WHITE
        mover_is_agent = (white == agent_white)
        if mover_is_agent:
            a = agent(s)
        else:
            a = opp(s)
        # record SAN before pushing (v9 flip: action is stm-oriented)
        mv = s.to_move(int(a))
        sans.append(s.board.san(mv))
        s = s.apply(int(a))
        trace.append(material_white(s.board))
    done, z_stm = s.is_terminal()
    if done and z_stm != 0.0:
        term = "checkmate"
        white_won = (s.board.turn == chess.BLACK)
    else:
        term = "adjudication/draw"
        z = adjudicate_material(s)
        white_won = z > 0 if s.board.turn == chess.WHITE else z < 0
        if z == 0.0:
            white_won = None
    agent_won = None if white_won is None else (white_won == agent_white)
    # material swing: min (most losing) white-relative material from agent view
    rel = [(m if agent_white else -m) for m in trace]
    return {"sans": sans, "plies": len(sans),
            "terminal": term, "agent_won": agent_won,
            "final_material_agent_view": rel[-1], "worst_material": min(rel),
            "agent_white": agent_white}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opp", default="greedy", choices=["greedy", "random"])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--planes", type=int, default=21)
    ap.add_argument("--out", default="diagnosis.json")
    a = ap.parse_args()
    model = load_model(a.ckpt, blocks=a.blocks, channels=a.channels,
                       planes=a.planes, device=a.device)
    ev = make_evaluate(model, device=a.device)

    def agent(st):
        pi = mcts_mod.search(st, ev, n_sims=a.sims, dirichlet_eps=0.0)
        return int(np.argmax(pi))

    opp = greedy_move if a.opp == "greedy" else random_move
    recs = [play_recorded(agent, opp, agent_white=(i % 2 == 0))
            for i in range(a.games)]
    mated = sum(1 for r in recs if r["terminal"] == "checkmate"
                and r["agent_won"] is False)
    adj_loss = sum(1 for r in recs if r["terminal"] != "checkmate"
                   and r["agent_won"] is False)
    wins = sum(1 for r in recs if r["agent_won"] is True)
    lens = [r["plies"] for r in recs]
    summ = {"ckpt": a.ckpt, "opp": a.opp, "sims": a.sims, "games": a.games,
            "wins": wins, "mated": mated, "adjudicated_losses": adj_loss,
            "avg_plies": sum(lens) / len(lens),
            "avg_worst_material": sum(r["worst_material"] for r in recs) / len(recs)}
    print(json.dumps(summ, indent=1))
    for i, r in enumerate(recs[:3]):
        print(f"--- game {i} white={r['agent_white']} {r['terminal']} "
              f"won={r['agent_won']} plies={r['plies']}")
        print(" ".join(f"{j // 2 + 1}.{m}" if j % 2 == 0 else m
                       for j, m in enumerate(r["sans"])))
    json.dump({"summary": summ, "games": recs}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
