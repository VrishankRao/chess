"""V19 part-B remainder tests (Impl-B): search+data.

Covers (mcts.py itself is DONE and untouched here):
 1. selfplay per-GAME eval_cache dicts (mcts-level hit reuse identical
    (p,v) + key format, and selfplay wiring: one dict per game passed to
    every search call).
 2. shape_targets + resign builder drop_tail (removes 4).
 3. Resign playthrough: cancels on recovery, resigns when still lost.
 4. hist KL (8th element) carried to tuple index 11 (12-tuples).
 5. replay surprise duplication (high-kl rows duplicate, cap 3,
    legacy/zero-sum write-once).
 6. replay fp16 storage roundtrip + finite train step.
 7. parallel fast/safety aggregation.

Run: python3 tests/test_v19b.py  (or pytest tests/test_v19b.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


# ---------------------------------------------------------------- helpers

def _game_cfg(**kw):
    from chess_zero.config import Config
    base = dict(blocks=2, channels=32, sims=2, move_cap=14, temp_moves=0,
                resign_threshold=-0.5, resign_moves=1,
                resign_playout_frac=0.0, contempt=0.0)
    base.update(kw)
    return Config(**base)


def _play_kwargs():
    # NOTE: play_game's resign_threshold/resign_moves/temp_moves PARAMS
    # shadow cfg (parallel.py/loop.py pass them explicitly from cfg, so
    # mirror that here); cfg values alone do NOT reach _play_game_impl.
    return dict(resign_threshold=-0.5, resign_moves=1, temp_moves=0)


def _mock_search(kl_seq=None):
    """Fake mcts.search: one-hot first-legal pi, scripted stats["kl"]."""
    state = {"i": 0}

    def fake_search(s, evaluate_fn, *a, **k):
        i = state["i"]
        state["i"] += 1
        st = k.get("stats")
        if st is not None:
            kl = kl_seq[i] if kl_seq is not None and i < len(kl_seq) else 0.0
            st["kl"] = float(kl)
        pi = np.zeros(4096, dtype=np.float32)
        pi[s.legal_moves()[0]] = 1.0
        return pi

    return fake_search


def _script_values(seq):
    """Fake evaluate_fn: values pop from seq (last repeats); priors blank
    (only the resign reads use this fn when search is mocked)."""
    vals = list(seq)
    last = vals[-1]

    def fn(states):
        out = [vals.pop(0) if vals else last for _ in states]
        return (np.zeros((len(states), 4096), dtype=np.float32),
                np.array(out, dtype=np.float32))

    return fn


def _toy_example(z=0.0, kl=None):
    enc = np.zeros((4, 8, 8), dtype=np.float32)
    pi = np.zeros(8, dtype=np.float32)
    pi[0] = 1.0
    own = np.zeros((8, 8), dtype=np.float32)
    t = (enc, pi, float(z), 0.0, 0.0, own, 0.0, 0.0, 0.5, -100, 1.0)
    if kl is not None:
        t = t + (float(kl),)
    return t


# ---------------------------------------------------------------- 1. eval cache

def test_eval_cache_hit_reuse():
    from chess_zero.game import State
    from chess_zero import mcts as mcts_mod
    s0 = State.initial()
    calls = {"n": 0}

    def counting(states):
        calls["n"] += len(states)
        n = len(states)
        return (np.full((n, 4096), 1.0 / 4096, dtype=np.float64),
                np.zeros(n, dtype=np.float64))

    kw = dict(n_sims=6, c_puct=1.414, dirichlet_eps=0.0, tt=None,
              history=[], quiescence_depth=0, leaf_batch=1)
    cache: dict = {}
    pi1 = mcts_mod.search(s0, counting, **kw, eval_cache=cache)
    n1 = calls["n"]
    assert n1 > 0, "first search must forward"
    # Second identical search: every eval hits -> zero new forwards.
    pi2 = mcts_mod.search(s0, counting, **kw, eval_cache=cache)
    assert calls["n"] == n1, (calls["n"], n1)
    assert np.array_equal(pi1, pi2), "hit reuse must be identical (p,v)"
    # Key format matches mcts._cache_key: (rep_key, rep_count) 2-tuples.
    assert len(cache) > 0
    assert all(isinstance(k, tuple) and len(k) == 3 for k in cache), \
        list(cache)[:2]
    # Pre-filled dict alone serves a search with no forwards at all.
    def poison(states):
        raise AssertionError("cache miss on all-hit search")
    pi3 = mcts_mod.search(s0, poison, **kw, eval_cache=dict(cache))
    assert np.array_equal(pi1, pi3)


def test_eval_cache_wired_in_selfplay():
    from unittest import mock
    from chess_zero.config import Config
    from chess_zero import mcts as mcts_mod
    from chess_zero import selfplay as sp_mod
    cfg = Config(blocks=2, channels=32, sims=2, move_cap=4, temp_moves=0)
    seen = []
    orig = mcts_mod.search

    def rec(state, evaluate_fn, *a, **k):
        seen.append(k.get("eval_cache"))
        return orig(state, evaluate_fn, *a, **k)

    def fake_eval(states):
        n = len(states)
        priors = np.zeros((n, 4096), dtype=np.float64)
        for i, s in enumerate(states):
            for mv in s.legal_moves():
                priors[i, mv] = 1.0
            priors[i] /= priors[i].sum()
        return priors, np.zeros(n, dtype=np.float64)

    with mock.patch.object(mcts_mod, "search", rec):
        ex, _res = sp_mod.play_game(None, cfg, evaluate_fn=fake_eval)
    assert len(seen) >= 2, seen
    assert all(isinstance(c, dict) for c in seen)
    assert all(c is seen[0] for c in seen), "one dict per game"
    assert len(seen[0]) > 0, "game populated its cache"
    assert len(ex) > 0 and all(len(e) == 18 for e in ex)  # V20: +score_mean,
    # score_stdev, root_q, av (14+4); indices 0-13 unchanged (append-only)


# ---------------------------------------------------------------- 2. drop_tail

def test_drop_tail():
    from chess_zero.game import State
    from chess_zero import selfplay as sp_mod
    s0 = State.initial()
    enc = s0.encode()
    pi = np.zeros(4096, dtype=np.float32)
    pi[s0.legal_moves()[:4]] = 0.25
    hist = [(enc, pi, i % 2, 0.0, i, 0.5, s0.legal_moves()[0], 0.1 * i)
            for i in range(6)]
    full = sp_mod.shape_targets(hist, 1.0, 6, 300, s0.board, full=True)
    cut = sp_mod.shape_targets(hist, 1.0, 6, 300, s0.board, full=True,
                               drop_tail=4)
    assert len(full) == 6 and len(cut) == 2, (len(full), len(cut))
    assert all(len(e) == 18 for e in full + cut)  # V20: +4
    for a, b in zip(full[:2], cut):  # kept prefix identical
        assert a[2] == b[2] and a[4] == b[4]
    rc = sp_mod.build_resign_examples(hist, 1.0, 6, 300, s0.board, True,
                                      drop_tail=4)
    assert len(rc) == 2 and all(len(e) == 18 for e in rc)  # V20
    rc0 = sp_mod.build_resign_examples(hist, 1.0, 6, 300, s0.board, True)
    assert len(rc0) == 6  # default drop_tail=0


# ---------------------------------------------------------------- 3. playthrough

def test_playthrough_cancels_on_recovery():
    from unittest import mock
    from chess_zero import mcts as mcts_mod
    from chess_zero import selfplay as sp_mod
    cfg = _game_cfg()
    ev = _script_values([-0.9, -0.9, 0.9] + [0.9] * 30)
    stats: dict = {}
    with mock.patch.object(mcts_mod, "search", _mock_search()):
        ex, res = sp_mod.play_game(None, cfg, evaluate_fn=ev, stats=stats,
                                   **_play_kwargs())
    assert stats.get("terminal") == "cap", stats
    assert res == "1/2-1/2", res
    assert len(ex) == 14, len(ex)  # full game: cancel, no tail drop


def test_playthrough_resigns_when_still_lost():
    from unittest import mock
    from chess_zero import mcts as mcts_mod
    from chess_zero import selfplay as sp_mod
    cfg = _game_cfg()
    ev = _script_values([-0.9, 0.9] * 20)
    stats: dict = {}
    with mock.patch.object(mcts_mod, "search", _mock_search()):
        ex, res = sp_mod.play_game(None, cfg, evaluate_fn=ev, stats=stats,
                                   **_play_kwargs())
    assert stats.get("terminal") == "resign", stats
    assert res == "0-1", res  # ply-6 white to move resigns
    assert len(ex) == 3, len(ex)  # 7 recorded plies minus drop_tail=4
    assert all(len(e) == 18 for e in ex)  # V20: +score_mean/stdev/root_q/av


# ---------------------------------------------------------------- 4. hist KL

def test_hist_kl_carried_to_tuple():
    from unittest import mock
    from chess_zero import mcts as mcts_mod
    from chess_zero import selfplay as sp_mod
    cfg = _game_cfg(move_cap=6)
    ev = _script_values([0.9] * 20)
    with mock.patch.object(
            mcts_mod, "search",
            _mock_search(kl_seq=[0.5 + 0.1 * i for i in range(20)])):
        ex, _res = sp_mod.play_game(None, cfg, evaluate_fn=ev,
                                    **_play_kwargs())
    assert len(ex) == 6, len(ex)
    for i, e in enumerate(ex):
        assert len(e) == 18, len(e)  # V20: +score_mean/stdev/root_q/av
        assert abs(float(e[11]) - (0.5 + 0.1 * i)) < 1e-9, (i, e[11])


# ---------------------------------------------------------------- 5. surprise

def test_surprise_duplicates_high_kl():
    import random as _r
    from chess_zero.replay import ReplayBuffer
    st = _r.getstate()
    try:
        # legacy 11-tuples (no kl): write-once each, exact.
        buf = ReplayBuffer()
        buf.add_game([_toy_example(z=float(i)) for i in range(4)])
        assert len(buf) == 4, len(buf)
        # zero-sum kl: w=1 write-once, exact.
        buf = ReplayBuffer()
        buf.add_game([_toy_example(z=float(i), kl=0.0) for i in range(4)])
        assert len(buf) == 4, len(buf)
        # concentrated surprise: w0 = 0.5+0.5*4*10/10 = 2.5 -> >= 2 copies.
        _r.seed(0)
        buf = ReplayBuffer()
        buf.add_game([_toy_example(z=7.0, kl=10.0)] +
                     [_toy_example(z=float(i), kl=0.0) for i in range(1, 4)])
        n0 = sum(1 for e in buf.buf if float(e[2]) == 7.0)
        assert 2 <= n0 <= 3, n0
        # cap: w0 = 0.5+0.5*6*9/9 = 3.5 -> exactly 3, RNG-independent.
        _r.seed(1)
        buf = ReplayBuffer()
        buf.add_game([_toy_example(z=7.0, kl=9.0)] +
                     [_toy_example(z=float(i), kl=0.0) for i in range(1, 6)])
        n0 = sum(1 for e in buf.buf if float(e[2]) == 7.0)
        assert n0 == 3, n0
    finally:
        _r.setstate(st)


# ---------------------------------------------------------------- 6. fp16

def test_fp16_roundtrip_and_train_finite():
    from chess_zero.game import State
    from chess_zero.config import TEST_CONFIG
    from chess_zero.model import AlphaZeroNet
    from chess_zero.replay import ReplayBuffer
    from chess_zero.train import train_step
    s0 = State.initial()
    enc = s0.encode().astype(np.float32)
    assert enc.shape == (13, 8, 8)
    pi = np.zeros(4096, dtype=np.float32)
    for a in s0.legal_moves()[:4]:
        pi[a] = 0.25
    own = np.zeros((8, 8), dtype=np.float32)
    rows = [(enc.copy(), pi.copy(), 0.5, 0.1, 0.5, own.copy(), 0.2, 0.5,
             0.5, -100, 1.0, 0.0) for _ in range(8)]
    buf = ReplayBuffer()
    buf.add_game(rows)
    assert len(buf) == 8, len(buf)  # kl=0 -> w=1 write-once
    stored = buf.buf[0]
    assert len(stored) == 12  # kl survives storage (P/ml_mask are
    # built by shape_targets, not stored by add_game)
    assert stored[0].dtype == np.float16, stored[0].dtype
    assert stored[1].dtype == np.float16, stored[1].dtype
    assert stored[5].dtype == np.float16, stored[5].dtype
    s, pi_b, z, m, ml, own_b, margin, mob, safe, reply, pw, mm, _pp, \
        _sm, _ss, _rq, _av = buf.sample(8)
    assert s.dtype == np.float32 and pi_b.dtype == np.float32
    assert own_b.dtype == np.float32
    assert z.dtype == np.float32 and m.dtype == np.float32
    torch.manual_seed(0)
    net = AlphaZeroNet(blocks=TEST_CONFIG.blocks,
                       channels=TEST_CONFIG.channels)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ret = train_step(net, opt, torch.from_numpy(s), torch.from_numpy(pi_b),
                     torch.from_numpy(z), torch.from_numpy(m),
                     torch.from_numpy(ml), torch.from_numpy(own_b),
                     torch.from_numpy(margin), torch.from_numpy(mob),
                     torch.from_numpy(safe), torch.from_numpy(reply),
                     torch.from_numpy(pw))
    assert np.isfinite(float(ret[0])), ret[0]


# ---------------------------------------------------------------- 7. parallel

def test_parallel_fast_safety_aggregation():
    from unittest import mock
    from chess_zero.config import Config
    from chess_zero import parallel as par_mod
    cfg = Config(blocks=2, channels=32, fast_frac=0.5)

    def fake_play_game(model, cfg, **kw):
        st = kw.get("stats")
        if st is not None:
            st.update({"vetoes": 1, "tactics": 2, "finishes": 0,
                       "safety": 2, "fast": 1, "terminal": "mate",
                       "breadth": 3.0, "ventropy": 0.5, "plies": 40})
        return ([("dummy",)], "1-0")

    job = ("dummy_weights.pt", 2, 32, 13, cfg, 3, 0, 123)
    with mock.patch.object(par_mod, "_build_cpu_model",
                           lambda *a, **k: None), \
         mock.patch("torch.load", return_value={}), \
         mock.patch("chess_zero.selfplay.play_game", fake_play_game):
        examples, results = par_mod._selfplay_chunk(job)
    assert len(examples) == 3
    assert results["1-0"] == 3
    assert results["fast"] == 3, results  # fast aggregation exists
    assert results["safety"] == 6, results  # safety aggregation exists
    assert results["Tmate"] == 3 and results["plies"] == 120

    job2 = ("dummy_weights.pt", 2, 32, 13, cfg, 2, 0, 456, 0)
    with mock.patch.object(par_mod, "_build_cpu_model",
                           lambda *a, **k: None), \
         mock.patch("torch.load", return_value={}), \
         mock.patch("chess_zero.selfplay.play_game", fake_play_game):
        examples2, results2 = par_mod._sparring_chunk(job2)
    assert len(examples2) == 2
    assert results2["safety"] == 4, results2  # sparring safety aggregated


if __name__ == "__main__":
    test_eval_cache_hit_reuse()
    print("ok eval_cache_hit_reuse")
    test_eval_cache_wired_in_selfplay()
    print("ok eval_cache_wired_in_selfplay")
    test_drop_tail()
    print("ok drop_tail")
    test_playthrough_cancels_on_recovery()
    print("ok playthrough_cancels_on_recovery")
    test_playthrough_resigns_when_still_lost()
    print("ok playthrough_resigns_when_still_lost")
    test_hist_kl_carried_to_tuple()
    print("ok hist_kl_carried_to_tuple")
    test_surprise_duplicates_high_kl()
    print("ok surprise_duplicates_high_kl")
    test_fp16_roundtrip_and_train_finite()
    print("ok fp16_roundtrip_and_train_finite")
    test_parallel_fast_safety_aggregation()
    print("ok parallel_fast_safety_aggregation")
    print("all v19b tests passed")
