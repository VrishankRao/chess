"""Phase 3 tests: inference server protocol on CPU loopback (no GPU needed).
Live training paths are untouched (all new branches default off); the full
suite proving that is tests/test_full.py.
"""
import numpy as np
import torch

from chess_zero.game import State
from chess_zero.model import AlphaZeroNet
from chess_zero import infer_server as _S


def _net(blocks=2, channels=16, planes=13, seed=0):
    torch.manual_seed(seed)
    m = AlphaZeroNet(blocks=blocks, channels=channels, planes=planes)
    m.eval()
    return m


def _positions(n=8):
    import random
    out, s = [], State.initial()
    while len(out) < n:
        out.append(s)
        s = s.apply(random.choice(s.legal_moves()))
        if s.is_terminal()[0]:
            s = State.initial()
    return out


def _valid(probs, states):
    assert probs.shape == (len(states), 4096)
    for i, s in enumerate(states):
        m = s.legal_mask()
        assert (probs[i][~m] == 0).all()
        assert abs(probs[i].sum() - 1.0) < 1e-4


def test_server_matches_local():
    rep = _S.validate_against_local(
        _net().state_dict(), 2, 16, 13, _positions(), device="cpu")
    assert not rep["degraded"], rep
    assert rep["max_policy_abs_diff"] < 1e-6, rep
    assert rep["max_value_abs_diff"] < 1e-6, rep
    assert rep["argmax_agreement"] == rep["n"], rep


def test_reload_changes_answers():
    import multiprocessing as _mp
    states = _positions(4)
    ctx = _mp.get_context("spawn")
    pc, cc = ctx.Pipe(duplex=True)
    kp, kc = ctx.Pipe(duplex=True)
    stop = ctx.Event()
    proc = ctx.Process(target=_S.serve,
                       args=(_net(seed=0).state_dict(), 2, 16, 13,
                             [cc], kc, stop, 64, 1.0, "cpu"),
                       daemon=True)
    proc.start()
    cc.close()
    try:
        fn = _S.make_server_evaluate(pc, _net(seed=0).state_dict(),
                                     2, 16, 13, device="cpu")
        p0, _ = fn(states)
        kp.send(("RELOAD", _net(seed=7).state_dict()))
        assert kp.poll(60)
        assert kp.recv()[0] == "RELOADED"
        # New client (fresh pipe state) sees the new weights.
        p1, _ = fn(states)
        assert np.abs(p0 - p1).max() > 1e-4, "reload must change outputs"
        _valid(p1, states)
    finally:
        stop.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


def test_dead_server_falls_back():
    import multiprocessing as _mp
    states = _positions(4)
    ctx = _mp.get_context("spawn")
    pc, cc = ctx.Pipe(duplex=True)
    kp, kc = ctx.Pipe(duplex=True)
    stop = ctx.Event()
    proc = ctx.Process(target=_S.serve,
                       args=(_net().state_dict(), 2, 16, 13, [cc], kc,
                             stop, 64, 1.0, "cpu"),
                       daemon=True)
    proc.start()
    cc.close()
    # Generous first-call timeout: spawn + torch import in the child can
    # take a while on a loaded box; the fallback under test is about a
    # DEAD server, which we arrange explicitly below.
    fn = _S.make_server_evaluate(pc, _net().state_dict(), 2, 16, 13,
                                 device="cpu", timeout=90.0)
    p0, _ = fn(states)  # alive: served
    assert not fn.degraded()
    proc.terminate()
    proc.join(timeout=10)
    import pytest
    with pytest.raises((EOFError, BrokenPipeError, TimeoutError, ConnectionError, OSError)):
        fn(states)  # fail closed; stale fallback weights cannot be trusted


def test_kill_switch_disables():
    import os
    assert _S.server_enabled()
    os.environ[_S.KILL_SWITCH] = "1"
    try:
        assert not _S.server_enabled()
    finally:
        del os.environ[_S.KILL_SWITCH]


# ---------------------------------------------------------------------------
# V21 Batch A (MPS validation, default-off only). All tests skip gracefully
# where MPS is absent (CPU-only CI) and use tiny nets + workers<=2 so the
# live run is never disturbed. No best.pt/history writes, no training.
# Acceptance numbers live in chess_zero/infer_server.py (module docstring).
# ---------------------------------------------------------------------------

def _needs_mps() -> bool:
    """True when MPS is available; otherwise prints a skip line and returns
    False (CPU-only CI stays green without needing pytest)."""
    import torch as _t
    if _t.backends.mps.is_available():
        return True
    print("SKIP V21 Batch A MPS test: MPS unavailable on this box",
          flush=True)
    return False


def test_v21_mps_parity_tiny():
    if not _needs_mps():
        return
    rep = _S.validate_against_local(
        _net().state_dict(), 2, 16, 13, _positions(), device="mps")
    assert not rep["degraded"], rep
    assert rep["max_policy_abs_diff"] < 1e-4, rep
    assert rep["argmax_agreement"] == rep["n"], rep


def test_v21_mps_stability_200_forwards():
    if not _needs_mps():
        return
    import multiprocessing as _mp
    import os as _os
    import time as _t
    import psutil as _ps
    states = _positions(8)
    ctx = _mp.get_context("spawn")
    pc, cc = ctx.Pipe(duplex=True)
    kp, kc = ctx.Pipe(duplex=True)
    stop = ctx.Event()
    proc = ctx.Process(target=_S.serve,
                       args=(_net().state_dict(), 2, 16, 13,
                             [cc], kc, stop, 8, 1.0, "mps"),
                       daemon=True)
    t0 = _t.time()
    proc.start()
    cc.close()
    try:
        fn = _S.make_server_evaluate(pc, _net().state_dict(),
                                     2, 16, 13, device="cpu",
                                     timeout=120.0)
        p0, _ = fn(states[:4])  # warmup (spawn + import on a loaded box)
        assert not fn.degraded()
        me = _ps.Process(_os.getpid())
        r0 = me.memory_info().rss
        c0 = _ps.Process(proc.pid).memory_info().rss
        for _ in range(200):
            fn(states)
            assert not fn.degraded()
        assert proc.is_alive()
        assert _t.time() - t0 < 300, "MPS server hang suspected"
        r1 = me.memory_info().rss
        c1 = _ps.Process(proc.pid).memory_info().rss
        # No-leak bound: generous (swap/contention-safe); the real V21
        # numbers (best.pt, batch 20) showed RSS *dropping*.
        assert (r1 - r0) / 1e6 < 150, (r0, r1)
        assert (c1 - c0) / 1e6 < 200, (c0, c1)
    finally:
        stop.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


def test_v21_parallel_server_mps_opt_in():
    if not _needs_mps():
        return
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.parallel import (
        _apply_game_settings, play_games_parallel)
    cfg = dataclasses.replace(TEST_CONFIG, move_cap=30)
    _apply_game_settings(cfg)
    try:
        net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels,
                           planes=cfg.input_planes)
        ex, res = play_games_parallel(net, cfg, 2, temp_moves=0,
                                      workers=2, use_server=True,
                                      server_device="mps")
        assert sum(res[k] for k in ("1-0", "0-1", "1/2-1/2")) == 2, res
        assert sum(res[k] for k in ("Tmate", "Tresign", "Tadjudicated",
                                    "Tadjudicated-draw", "Trules-draw",
                                    "Tcap")) == 2, res
    finally:
        _apply_game_settings((3.0, 100, 0, 13))


def test_parallel_selfplay_server_cpu():
    # Real protocol through pool workers (CPU loopback server).
    # move_cap=60: random-net games shuffle to the cap; bounding plies
    # keeps this a protocol test, not an endurance run.
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.parallel import (
        _apply_game_settings, play_games_parallel)
    cfg = dataclasses.replace(TEST_CONFIG, move_cap=60)
    _apply_game_settings(cfg)
    try:
        net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels,
                           planes=cfg.input_planes)
        ex, res = play_games_parallel(net, cfg, 2, temp_moves=0,
                                      workers=2, use_server=True,
                                      server_device="cpu")
        assert sum(res[k] for k in ("1-0", "0-1", "1/2-1/2")) == 2, res
        # v13: exactly one terminal reason recorded per game.
        assert sum(res[k] for k in ("Tmate", "Tresign", "Tadjudicated",
                                    "Tadjudicated-draw", "Trules-draw",
                                    "Tcap")) == 2, res
        assert len(ex) > 0 and len(ex[0]) == 18  # V20: +score_mean/stdev/root_q/av
    finally:
        _apply_game_settings((3.0, 100, 0, 13))


def test_parallel_match_server_cpu():
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.loop import agent_policy, game_settings
    from chess_zero.evaluate import greedy_move
    from chess_zero.parallel import (
        _apply_game_settings, play_match_parallel)
    cfg = dataclasses.replace(TEST_CONFIG)
    _apply_game_settings(cfg)
    try:
        net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels,
                           planes=cfg.input_planes)
        ag = agent_policy(net, cfg, sims=2, device="cpu")
        r = play_match_parallel(ag, greedy_move, games=2, cap=30,
                                workers=2, opening_moves=2,
                                game_settings=game_settings(cfg),
                                use_server=True, server_device="cpu")
        assert sum(r.values()) == 2, r
    finally:
        _apply_game_settings((3.0, 100, 0, 13))


if __name__ == "__main__":
    test_server_matches_local()
    test_reload_changes_answers()
    test_dead_server_falls_back()
    test_kill_switch_disables()
    test_parallel_selfplay_server_cpu()
    test_parallel_match_server_cpu()
    test_v21_mps_parity_tiny()
    test_v21_mps_stability_200_forwards()
    test_v21_parallel_server_mps_opt_in()
    print("all infer-server tests passed")
