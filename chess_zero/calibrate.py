"""Stockfish calibration ladder: real absolute-Elo anchor.
Stockfish 14 caps UCI_Elo at >=1350, so the ladder is:
  depth1 (~800-1000?) -> depth2 -> elo1350 -> elo1500 -> elo1800.
Score at each rung pins our level via standard Elo math, even in defeat.
Usage: python3 -m chess_zero.calibrate --ckpt checkpoints/best.pt --sims 40
"""
from __future__ import annotations

import argparse
import math

import chess
import chess.engine
import numpy as np

from .game import State
from .model import AlphaZeroNet
from . import mcts as mcts_mod
from .selfplay import make_evaluate
from .evaluate import adjudicate_material


def stockfish_policy(path="/usr/games/stockfish", depth=None, elo=None,
                     movetime=0.2):
    eng = chess.engine.SimpleEngine.popen_uci(path)
    if elo is not None:
        eng.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    limit = chess.engine.Limit(depth=depth, time=movetime if depth is None else None)

    def fn(state: State) -> int:
        res = eng.play(state.board, limit)
        m = res.move
        a = __import__("chess_zero.game", fromlist=["encode_action"]).encode_action(m.from_square, m.to_square, m.promotion)
        if state.board.turn == chess.BLACK:
            # v9 flip: engine UCI is absolute, State is stm-oriented.
            from .game import flip_action as _flip
            a = _flip(a)
        return a
    fn._engine = eng
    return fn


def play_one(agent, opp, agent_white: bool, cap=200) -> float:
    """Agent score: 1/0.5/0."""
    s = State.initial()
    while True:
        done, _ = s.is_terminal()
        if done or s.ply_count >= cap:
            break
        white = s.board.turn == chess.WHITE
        s = s.apply(agent(s) if (white == agent_white) else opp(s))
    done, z_stm = s.is_terminal()
    if not done:
        z_stm = adjudicate_material(s)
        if z_stm == 0.0:
            return 0.5
        white_won = (s.board.turn == chess.WHITE) == (z_stm == 1.0)
        return 1.0 if (white_won == agent_white) else 0.0
    if z_stm == 0.0:
        return 0.5
    white_won = s.board.turn == chess.BLACK  # stm got mated
    return 1.0 if (white_won == agent_white) else 0.0


def elo_vs(score: float, opp_elo: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return opp_elo + 400 * math.log10(score / (1 - score))


RUNGS = {
    "sf-depth1": ("depth", 1, None),
    "sf-depth2": ("depth", 2, None),
    "sf-depth3": ("depth", 3, None),
    "sf-elo1350": ("elo", None, 1350),
    "sf-elo1500": ("elo", None, 1500),
    "sf-elo1800": ("elo", None, 1800),
}


def _calib_chunk(job) -> tuple[float, int]:
    weights_path, blocks, channels, planes, rung, game_ids, sims, sf, cap = \
        job
    import torch as _torch
    from .model import AlphaZeroNet as _Net, load_weights as _lw, \
        infer_se_ratio as _isr
    from .selfplay import make_evaluate as _me
    from . import mcts as _mcts
    _wp = _torch.load(weights_path, map_location="cpu", weights_only=False)
    model = _Net(blocks=blocks, channels=channels, planes=planes,
                 se_ratio=_isr(_wp))
    _lw(model, _wp, strict=True)
    from . import game as _game_mod
    _game_mod.INPUT_PLANES = planes
    # Deliberate: calibration keeps CLASSICAL adjudication values (workers
    # inherit import-time defaults; fitted values are never stamped here)
    # so rung scores stay comparable across versions.
    ev = _me(model, device="cpu")

    def agent(st):
        return int(np.argmax(_mcts.search(st, ev, n_sims=sims,
                                          dirichlet_eps=0.0)))

    kind, depth, elo = RUNGS[rung]
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    try:
        if elo is not None:
            eng.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        lim = chess.engine.Limit(depth=depth) if depth else \
            chess.engine.Limit(time=0.2)

        def opp(state: State) -> int:
            res = eng.play(state.board, lim)
            m = res.move
            a = __import__("chess_zero.game", fromlist=["encode_action"]).encode_action(m.from_square, m.to_square, m.promotion)
            if state.board.turn == chess.BLACK:
                # v9 flip: engine UCI is absolute, State is stm-oriented.
                from .game import flip_action as _flip
                a = _flip(a)
            return a

        sc = sum(play_one(agent, opp, agent_white=(i % 2 == 0), cap=cap)
                 for i in game_ids)
    finally:
        eng.quit()
    return sc, len(game_ids)


def calibrate_ckpt(weights_path: str, rung: str, games: int, sims: int,
                   blocks=5, channels=128, sf="/usr/games/stockfish",
                   cap=200, workers=8, input_planes=13) -> dict:
    """Parallel rung calibration. Returns {score, wins-est, games}."""
    import os as _os
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp
    workers = max(1, min(workers, games))
    if workers == 1:
        sc, n = _calib_chunk((weights_path, blocks, channels, input_planes,
                              rung, list(range(games)), sims, sf, cap))
    else:
        ids = list(range(games))
        slices = [ids[i::workers] for i in range(workers)]
        base = int(np.random.randint(0, 1_000_000_000))
        jobs = [(weights_path, blocks, channels, input_planes, rung, s,
                 sims, sf, cap)
                for i, s in enumerate(slices) if s]
        ctx = _mp.get_context("spawn")
        import torch as _t
        sc, n = 0.0, 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_t.set_num_threads,
                                 initargs=(1,)) as ex_:
            for s, c in ex_.map(_calib_chunk, jobs):
                sc += s
                n += c
    anchors = {"sf-elo1350": 1350, "sf-elo1500": 1500, "sf-elo1800": 1800}
    return {"rung": rung, "score": sc / n, "games": n,
            "est_elo": round(elo_vs(sc / n, anchors[rung]), 1)
            if rung in anchors else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sf", default="/usr/games/stockfish")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--planes", type=int, default=21)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rungs", default="sf-depth1,sf-elo1350",
                    help="comma-separated subset of: " + ",".join(RUNGS))
    a = ap.parse_args()
    import torch
    wp = a.ckpt if a.ckpt.endswith(".pt") and "iter" not in a.ckpt else None
    if wp is None:
        # extract raw weights file for worker broadcast
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        w = sd["weights"] if isinstance(sd, dict) and "weights" in sd else sd
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        try:
            torch.save(w, f.name)
            wp = f.name
            for name in a.rungs.split(","):
                r = calibrate_ckpt(wp, name.strip(), a.games, a.sims,
                                   a.blocks, a.channels, a.sf,
                                   workers=a.workers,
                                   input_planes=a.planes)
                print(f"{r['rung']}: score {r['score']:.3f} over "
                      f"{r['games']} games"
                      + (f" est {r['est_elo']}" if r["est_elo"] else ""),
                      flush=True)
        finally:
            try:
                import os as _os
                _os.unlink(f.name)
            except OSError:
                pass
    else:
        for name in a.rungs.split(","):
            r = calibrate_ckpt(wp, name.strip(), a.games, a.sims,
                               a.blocks, a.channels, a.sf,
                               workers=a.workers, input_planes=a.planes)
            print(f"{r['rung']}: score {r['score']:.3f} over "
                  f"{r['games']} games"
                  + (f" est {r['est_elo']}" if r["est_elo"] else ""),
                  flush=True)


if __name__ == "__main__":
    main()
