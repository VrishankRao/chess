"""Endgame-panel parallel parity (surgical task).

Proves: on fixed FENs + fixed seed_base, the parallel fan-out
(workers=2, one job per mirrored pair, per-game explicit seeds
seed_base + pair_idx*2 + leg, worker-local agent rebuilds) returns
BIT-IDENTICAL WLD/score/pass to the sequential loop (workers=1).

Constraints: /tmp only (endgame FEN file in tempfile); no
best.pt/history writes (calls _endgame_panel_gate directly, never
_gate/run_training); no training launches. Tiny net (1x8, sims=2) so
the test is seconds, not minutes.

Run: PYTHONPATH=. python3 tests/test_endgame_parallel.py
     (or pytest tests/test_endgame_parallel.py).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

FIXED_FENS = [
    "8/8/4k3/8/8/3KQ3/8/8 w - - 0 1",  # KQvK
    "8/8/4k3/8/8/3KR3/8/8 w - - 0 1",  # KRvK
    "8/8/4k3/8/3P4/8/4K3/8 w - - 0 1",  # KPvK key square
]

SEED_BASE = 1234


def _tiny_weights(seed: int):
    from chess_zero.model import AlphaZeroNet
    torch.manual_seed(seed)
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    return {k: v.cpu().clone() for k, v in net.state_dict().items()}


def _cfg(fen_path: str, **kw):
    from chess_zero.config import Config
    d = dict(blocks=1, channels=8, input_planes=13, sims=2,
             move_cap=40, endgame_file=fen_path,
             endgame_panel_pairs=4, endgame_panel_sims=2,
             endgame_panel_min=0.55)
    d.update(kw)
    return Config(**d)


def _write_fens_tmp() -> str:
    fd, path = tempfile.mkstemp(prefix="eg_fens_", suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for fen in FIXED_FENS:
            f.write(json.dumps({"fen": fen}) + "\n")
    return path


def test_game_seed_derivation():
    from chess_zero.loop import _endgame_game_seed
    assert _endgame_game_seed(2000, 0, 0) == 2000
    assert _endgame_game_seed(2000, 0, 1) == 2001
    assert _endgame_game_seed(2000, 3, 0) == 2006
    assert _endgame_game_seed(2000, 3, 1) == 2007
    # distinct per (pair, leg)
    seen = {_endgame_game_seed(SEED_BASE, i, l)
            for i in range(8) for l in (0, 1)}
    assert len(seen) == 16


def test_default_off_is_sequential():
    from chess_zero.config import Config
    assert getattr(Config(), "endgame_panel_workers", 0) in (0, 1)


def test_parallel_wld_identical_to_sequential():
    from chess_zero.loop import _endgame_panel_gate
    path = _write_fens_tmp()
    try:
        cfg = _cfg(path)
        chall = _tiny_weights(11)
        incumb = _tiny_weights(22)
        seq = _endgame_panel_gate(chall, incumb, cfg, device="cpu",
                                  seed_base=SEED_BASE, workers=1)
        par = _endgame_panel_gate(chall, incumb, cfg, device="cpu",
                                  seed_base=SEED_BASE, workers=2)
        assert seq["wld"] == par["wld"], (seq["wld"], par["wld"])
        assert seq["score"] == par["score"], (seq["score"], par["score"])
        assert seq["pass"] == par["pass"], (seq["pass"], par["pass"])
        assert seq["pairs"] == par["pairs"] == 4
        assert seq["fens"] == par["fens"], "same FEN order, same seeds"
        # different fan-out (4 workers = 1 pair each) must also match
        par4 = _endgame_panel_gate(chall, incumb, cfg, device="cpu",
                                   seed_base=SEED_BASE, workers=4)
        assert seq["wld"] == par4["wld"], (seq["wld"], par4["wld"])
        # workers=0 is sequential fallback, identical too
        seq0 = _endgame_panel_gate(chall, incumb, cfg, device="cpu",
                                   seed_base=SEED_BASE, workers=0)
        assert seq["wld"] == seq0["wld"], (seq["wld"], seq0["wld"])
        print(f"[parity] seq={seq['wld']} par={par['wld']} "
              f"score={seq['score']} pass={seq['pass']}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_cfg_knob_enables_identical_parallel():
    from chess_zero.loop import _endgame_panel_gate
    path = _write_fens_tmp()
    try:
        cfg_seq = _cfg(path)
        cfg_par = _cfg(path, endgame_panel_workers=2)
        chall = _tiny_weights(11)
        incumb = _tiny_weights(22)
        seq = _endgame_panel_gate(chall, incumb, cfg_seq, device="cpu",
                                  seed_base=SEED_BASE)
        # workers=None -> reads cfg knob (=2) -> parallel, same result
        par = _endgame_panel_gate(chall, incumb, cfg_par, device="cpu",
                                  seed_base=SEED_BASE)
        assert seq["wld"] == par["wld"], (seq["wld"], par["wld"])
        assert seq["pass"] == par["pass"]
        # default cfg without workers arg stays sequential & identical
        assert seq["wld"] == _endgame_panel_gate(
            chall, incumb, cfg_seq, device="cpu",
            seed_base=SEED_BASE)["wld"]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    test_game_seed_derivation()
    print("seed derivation OK")
    test_default_off_is_sequential()
    print("default-off OK")
    test_parallel_wld_identical_to_sequential()
    print("parallel==sequential OK")
    test_cfg_knob_enables_identical_parallel()
    print("cfg-knob OK")
    print("all endgame-parallel tests passed")
