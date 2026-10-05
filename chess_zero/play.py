"""Human-play CLI: ASCII board, human vs MCTS agent.
Usage: PYTHONPATH=. python3 -m chess_zero.play --ckpt live.pt --sims 100
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch

from .model import AlphaZeroNet
from .game import State
from . import mcts as mcts_mod
from .selfplay import make_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="live.pt")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--planes", type=int, default=21)
    a = ap.parse_args()
    import chess_zero.game as _g
    # v18: stamp what the net eats (see diagnose.load_model).
    _g.INPUT_PLANES = a.planes
    try:
        from .model import load_weights, infer_se_ratio
        _w = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        _w = None
        print(f"(no checkpoint [{e}] — untrained net, plays randomly)")
    model = AlphaZeroNet(blocks=a.blocks, channels=a.channels,
                         planes=a.planes,
                         se_ratio=infer_se_ratio(_w) if _w else 0)
    if _w is not None:
        from .model import load_weights
        load_weights(model, _w, strict=True)
        print(f"loaded {a.ckpt} ({a.sims} sims/move)")
    evaluate = make_evaluate(model)
    state = State.initial()
    human_white = input("Play as White? [Y/n]: ").strip().lower() != "n"
    while True:
        print(state.board)
        print()
        done, z = state.is_terminal()
        if done:
            print("game over, side-to-move value:", z)
            break
        is_human = (state.board.turn == chess.WHITE) == human_white
        if is_human:
            try:
                uci = input("your move (uci, e.g. e2e4): ").strip()
            except EOFError:
                print("\nbye.")
                break
            try:
                mv = state.board.parse_uci(uci)
                act = __import__("chess_zero.game", fromlist=["encode_action"]).encode_action(mv.from_square, mv.to_square, mv.promotion)
                if state.board.turn == chess.BLACK:
                    # v9 flip: human UCI is absolute, State is stm-oriented.
                    from .game import flip_action as _flip
                    act = _flip(act)
                if act not in state.legal_moves():
                    print("illegal move")
                    continue
                state = state.apply(act)
            except Exception as e:
                print("error:", e)
        else:
            import time
            from .selfplay import tactical_action as _tac
            from .selfplay import apply_move_guards as _guards
            t0 = time.time()
            pi = mcts_mod.search(state, evaluate, n_sims=a.sims)
            tac = _tac(state)
            act, _, _ = _guards(state, int(np.argmax(pi)), pi, tac,
                                True, 0.09)
            state = state.apply(act)
            print(f"agent played in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
