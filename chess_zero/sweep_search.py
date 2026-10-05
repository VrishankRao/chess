"""V20 D7 search sweep harness (Agent B): grid FPU x cPuct x MLH-thr at
FIXED sims on a middlegame suite. The coordinator runs the sweeps;
this file builds the harness + the literature defaults live in
V20_CONFIG (FPU 0.4, c_puct 1.2, ml_thr 0.9 — D7).

Design: each cell = a cfg copy with (fpu_reduction, c_puct, ml_thr)
overridden, playing `games` mirrored games vs the greedy baseline from
the embedded middlegame suite (round-robin FEN starts, colors swapped
per pair). Fixed sims => ~fixed NPS on one machine; measured NPS per
cell is reported so incomparable cells are visible, not hidden.
Output: printed table (sorted by score) + JSON {cells, suite, sims}.
CPU-viable smoke: --random (fresh net) --sims 2 --games 2 --tiny-grid.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

# Embedded middlegame suite (12 real middlegame positions, both colors
# to move; quiet + tactical mix). Starts, not labels — measurement only.
SUITE_FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
    "r2q1rk1/ppp1ppbp/2np1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQ1RK1 w - - 0 1",
    "2r2rk1/pp1bppbp/2np1np1/q1N1P3/3P4/2N2N2/PPP1BPPP/R1BQ1RK1 w - - 0 1",
    "r1bq1rk1/pp1pppbp/2n2np1/8/3PN3/2N2B2/PPP1BPPP/R2Q1RK1 b - - 0 1",
    "r1bqr1k1/ppp1ppbp/2np1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQR1K1 b - - 0 1",
    "rnbqk2r/ppp1ppbp/5np1/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    "2rq1rk1/pp1bpppp/2n2n2/2pp4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1 w - - 0 1",
    "r2qk2r/ppp1bppp/2n2n2/3p4/3P1B2/2N1PN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "rn1qk2r/p1ppbppp/bp2pn2/3P4/4P3/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 0 1",
]

DEFAULT_FPUS = (0.3, 0.4, 0.5)
DEFAULT_CPUCTS = (1.0, 1.2, 1.6)
DEFAULT_ML_THRS = (0.8, 0.9)


def wilson_lower(wins: float, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound (draws count half upstream)."""
    try:
        from .loop import wilson_lower as _w
        return _w(wins, n, z)
    except Exception:
        return 0.0


def run_cell(base_cfg, ckpt_or_net, fpu: float, cpuct: float, ml_thr: float,
             sims: int, games: int, device: str = "cpu",
             cap: int = 300, seed_base: int = 0) -> dict:
    """One grid cell: cfg copy with the three knobs overridden; mirrored
    suite games (challenger white then black per FEN) vs greedy at fixed
    sims. Returns {"fpu","cpuct","ml_thr","wld","score","wilson","nps",
    "secs"}. Never raises (total failure -> 0-0-0 + note)."""
    import chess as _c
    from .game import State as _S
    from .evaluate import greedy_move
    from .loop import agent_from_weights as _afw, _weights_dict as _wd
    _t0 = time.time()
    try:
        cfg = copy.copy(base_cfg)
        cfg.fpu_reduction = float(fpu)
        cfg.c_puct = float(cpuct)
        cfg.ml_thr = float(ml_thr)
        try:
            ch = _afw(_wd(ckpt_or_net), cfg, sims=int(sims),
                      device=device, noise=0.0)
        except Exception as _e:
            return {"fpu": fpu, "cpuct": cpuct, "ml_thr": ml_thr,
                    "wld": [0, 0, 0], "score": 0.0, "wilson": 0.0,
                    "nps": 0.0, "secs": 0.0,
                    "note": f"agent-build-{type(_e).__name__}"}
        _W = _L = _D = 0
        _moves = 0
        for _gi in range(int(games)):
            _fen = SUITE_FENS[(_gi // 2) % len(SUITE_FENS)]
            _ch_white = (_gi % 2 == 0)
            try:
                _st = _S(_c.Board(str(_fen)))
            except Exception:
                _D += 1
                continue
            for _pol in (ch,):
                try:
                    _rst = getattr(_pol, "reset", None)
                    if callable(_rst):
                        _rst()
                except Exception:
                    pass
            _n = 0
            while _n < int(cap):
                try:
                    _done, _ = _st.is_terminal()
                except Exception:
                    _done = False
                if _done or _st.ply_count >= int(cap):
                    break
                try:
                    _white = _st.board.turn == _c.WHITE
                    _pol = (ch if _white == _ch_white else greedy_move)
                    _st = _st.apply(int(_pol(_st)))
                except Exception:
                    break
                _n += 1
                _moves += 1
            try:
                _done, _z = _st.is_terminal()
            except Exception:
                _done, _z = False, 0.0
            if not _done or _z == 0.0:
                _D += 1
            else:
                import chess as _c2
                _wres = _z if _st.board.turn == _c2.WHITE else -_z
                _ch_res = _wres if _ch_white else -_wres
                if _ch_res == 1.0:
                    _W += 1
                elif _ch_res == -1.0:
                    _L += 1
                else:
                    _D += 1
        _secs = max(1e-9, time.time() - _t0)
        _n = _W + _L + _D
        _sc = ((_W + 0.5 * _D) / _n) if _n else 0.0
        return {"fpu": float(fpu), "cpuct": float(cpuct),
                "ml_thr": float(ml_thr), "wld": [int(_W), int(_L),
                                                 int(_D)],
                "score": round(_sc, 4),
                "wilson": round(wilson_lower(_W + 0.5 * _D, _n), 4),
                "nps": round(_moves * int(sims) / _secs, 1),
                "secs": round(_secs, 1), "sims": int(sims)}
    except Exception as _e:
        return {"fpu": fpu, "cpuct": cpuct, "ml_thr": ml_thr,
                "wld": [0, 0, 0], "score": 0.0, "wilson": 0.0,
                "nps": 0.0, "secs": round(time.time() - _t0, 1),
                "note": f"cell-error-{type(_e).__name__}"}


def run_sweep(base_cfg, ckpt_or_net, fpus, cpucts, ml_thrs,
              sims: int = 400, games: int = 4, device: str = "cpu",
              seed_base: int = 0) -> dict:
    """Full grid. Cells run sequentially (one process, fixed NPS context);
    prints a score-sorted table. Returns {"cells", "sims", "games",
    "suite_n"}. Never raises."""
    _cells = []
    for _f in fpus:
        for _c in cpucts:
            for _m in ml_thrs:
                _r = run_cell(base_cfg, ckpt_or_net, _f, _c, _m, sims,
                              games, device, seed_base=int(seed_base))
                _cells.append(_r)
                print(f"[sweep] fpu={_f} cpuct={_c} ml_thr={_m} @ "
                      f"{sims} sims: WLD={_r['wld']} score={_r['score']} "
                      f"wilson={_r['wilson']} nps={_r['nps']}",
                      flush=True)
    _cells.sort(key=lambda _d: (_d.get("score", 0.0),
                                _d.get("wilson", 0.0)), reverse=True)
    print("[sweep] ranking (score, wilson):", flush=True)
    for _r in _cells:
        print(f"  fpu={_r['fpu']} cpuct={_r['cpuct']} ml_thr={_r['ml_thr']}: "
              f"{_r['score']} ({_r['wilson']}) nps={_r['nps']}",
              flush=True)
    return {"cells": _cells, "sims": int(sims), "games": int(games),
            "suite_n": len(SUITE_FENS)}


def _load_ckpt_or_random(args):
    """Resolve the weights source: --ckpt path (tolerant load) or
    --random fresh net on the requested arch. Returns a weights-like
    (dict) for agent_from_weights."""
    import torch as _t
    from .model import AlphaZeroNet as _Net, load_weights as _lw, \
        infer_se_ratio as _isr
    if args.ckpt:
        return _t.load(args.ckpt, map_location="cpu",
                       weights_only=False)
    _m = _Net(blocks=args.blocks, channels=args.channels,
              planes=args.planes, se_ratio=args.se_ratio)
    return {k: v.cpu().clone() for k, v in _m.state_dict().items()}


def main() -> int:
    _p = argparse.ArgumentParser()
    _p.add_argument("--ckpt", default=None)
    _p.add_argument("--random", action="store_true")
    _p.add_argument("--blocks", type=int, default=6)
    _p.add_argument("--channels", type=int, default=64)
    _p.add_argument("--planes", type=int, default=30)
    _p.add_argument("--se-ratio", type=int, default=4)
    _p.add_argument("--sims", type=int, default=400)
    _p.add_argument("--games", type=int, default=4)
    _p.add_argument("--fpus", default="0.3,0.4,0.5")
    _p.add_argument("--cpucts", default="1.0,1.2,1.6")
    _p.add_argument("--ml-thrs", default="0.8,0.9")
    _p.add_argument("--tiny-grid", action="store_true")
    _p.add_argument("--device", default="cpu")
    _p.add_argument("--seed", type=int, default=0)
    _p.add_argument("--out", default=None)
    _a = _p.parse_args()
    if not _a.ckpt and not _a.random:
        print("need --ckpt or --random", flush=True)
        return 2
    if _a.tiny_grid:
        _fpus, _cp, _ml = (0.4,), (1.2,), (0.9,)
    else:
        def _nums(_s):
            return tuple(float(_x) for _x in str(_s).split(",") if _x)
        _fpus, _cp, _ml = _nums(_a.fpus), _nums(_a.cpucts), \
            _nums(_a.ml_thrs)
    from .config import V20_CONFIG
    import copy as _copy
    _cfg = _copy.copy(V20_CONFIG)
    _cfg.blocks = int(_a.blocks)
    _cfg.channels = int(_a.channels)
    _cfg.input_planes = int(_a.planes)
    _cfg.se_ratio = int(_a.se_ratio)
    _w = _load_ckpt_or_random(_a)
    _res = run_sweep(_cfg, _w, _fpus, _cp, _ml, _a.sims,
                     _a.games, _a.device, _a.seed)
    if _a.out:
        try:
            _d = os.path.dirname(_a.out) or "."
            os.makedirs(_d, exist_ok=True)
            with open(_a.out, "w") as _f:
                json.dump(_res, _f, indent=1)
            print(f"[sweep] wrote {_a.out}", flush=True)
        except Exception as _e:
            print(f"[sweep] write failed ({type(_e).__name__})",
                  flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
