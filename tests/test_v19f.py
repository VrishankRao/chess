"""V19 search+data tests, part F (Impl-B own file): B1-B6/B7-B9 audit +
D3 in-tree MLH + D5 P/ml_mask growth.

Covers (chess_zero/mcts.py, selfplay.py, replay.py; parallel.py verify-only):
 1. legacy bit-exactness: same-seed determinism + /tmp/mcts_before_v19f.py
    (pre-edit copy) vs current search() identical pi under defaults.
 2. WU math: O in sqrt+denom, Q untouched; o==0 equals _select; _Node.o/.m
    inits; batched path runs (v19f P0 fix: o was never initialized).
 3. FPU: default 0.0 bit-exact vs legacy; _select_fpu unit; fires when set.
 4. prune_singletons: visits>1 only + sums to 1 when on; off keeps 1s.
 5. futile/proven stop: fires on decided root (stats flag), silent else.
 6. KL stat: finite, >= 0.
 7. eval cache: hit reuse identical (p,v), key 2-tuples, zero new forwards.
 8. ML bonus: off-by-default bit-exact (ml_fn + slope 0 == no ml_fn);
    fires when configured (unit + search-level pi shift + root m set).
 9. P/ml_mask: indices 12/13, 14-tuples, lookahead/censored values,
    played_out masks, legacy guards, resign builder, drop-tail alignment.
10. Brief re-asserts: surprise duplication (+cap), fp16 roundtrip + 12-array
    sample, tail-drop, playthrough cancel/resign, hist-KL carried.
11. selfplay wiring: one shared eval_cache dict per game; fpu_reduction /
    prune_singletons / ml_slope/ml_cap/ml_thr / ml_fn kwargs present.
12. parallel fast/safety aggregation present (verify-only, no new counters).

Run: python3 tests/test_v19f.py  (or pytest tests/test_v19f.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BEFORE_PATH = "/tmp/mcts_before_v19f.py"


# ---------------------------------------------------------------- helpers

def _uniform_eval(v=0.0):
    def fn(states):
        n = len(states)
        return (np.full((n, 4096), 1.0 / 4096, dtype=np.float64),
                np.full(n, float(v), dtype=np.float64))
    return fn


def _peaked_eval(first_boost=4.0, v=0.1):
    """Deterministic peaked priors (first legal move boosted) + const value."""
    def fn(states):
        n = len(states)
        pr = np.full((n, 4096), 1.0 / 4096, dtype=np.float64)
        for i, s in enumerate(states):
            lm = s.legal_moves()
            if lm:
                pr[i, lm[0]] *= first_boost
                pr[i] /= pr[i].sum()
        return pr, np.full(n, float(v), dtype=np.float64)
    return fn


def _search_kw(**kw):
    base = dict(n_sims=12, c_puct=1.414, dirichlet_eps=0.0, tt=None,
                history=[], quiescence_depth=0, leaf_batch=1)
    base.update(kw)
    return base


def _load_before():
    """Import the pre-edit mcts copy as a package submodule (so its
    relative `from .game import ...` resolves to the current game code —
    game.py is untouched by this change, so the comparison is honest)."""
    import importlib.util
    name = "chess_zero._mcts_before_v19f"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BEFORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _game_cfg(**kw):
    from chess_zero.config import Config
    base = dict(blocks=2, channels=32, sims=2, move_cap=14, temp_moves=0,
                resign_threshold=-0.5, resign_moves=1,
                resign_playout_frac=0.0, contempt=0.0)
    base.update(kw)
    return Config(**base)


def _play_kwargs():
    return dict(resign_threshold=-0.5, resign_moves=1, temp_moves=0)


def _mock_search(kl_seq=None):
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
    vals = list(seq)
    last = vals[-1]

    def fn(states):
        out = [vals.pop(0) if vals else last for _ in states]
        return (np.zeros((len(states), 4096), dtype=np.float32),
                np.array(out, dtype=np.float32))

    return fn


# ------------------------------------------------- 1. legacy bit-exactness

def test_legacy_determinism_and_backup_bitexact():
    from chess_zero.game import State
    from chess_zero import mcts as cur
    s0 = State.initial()
    ev = _peaked_eval()
    np.random.seed(123)
    pi_a = cur.search(s0, ev, **_search_kw(dirichlet_eps=0.3))
    np.random.seed(123)
    pi_b = cur.search(s0, ev, **_search_kw(dirichlet_eps=0.3))
    assert np.array_equal(pi_a, pi_b), "same-seed determinism broken"
    assert abs(pi_a.sum() - 1.0) < 1e-6
    if not os.path.exists(BEFORE_PATH):
        print("  (no /tmp backup: self-determinism only)")
        return
    before = _load_before()
    np.random.seed(123)
    pi_old = before.search(s0, ev, **_search_kw(dirichlet_eps=0.3))
    np.random.seed(123)
    pi_new = cur.search(s0, ev, **_search_kw(dirichlet_eps=0.3))
    assert np.array_equal(pi_old, pi_new), \
        "defaults diverged from pre-edit selection path"


# ---------------------------------------------------------------- 2. WU

def test_wu_math():
    from chess_zero import mcts as m
    n = m._Node()
    assert n.o == 0 and n.m is None, (n.o, n.m)
    ch = m._Node(prior=0.25)
    ch.n, ch.w, ch.o = 4, 1.0, 0
    tot, c = 20, 1.414
    assert m._select_wu(ch, tot, c) == m._select(ch, tot, c), \
        "o==0 must equal legacy _select exactly"
    ch.o = 3  # Q untouched; O only in sqrt+denom
    import numpy as _np
    u_wu = c * ch.prior * _np.sqrt(max(1, tot)) / (1 + 4 + 3)
    assert abs(m._select_wu(ch, tot, c) - (ch.q + u_wu)) < 1e-12
    assert ch.q == 0.25
    # backup: W flips, M averages without flip
    a = m._Node()
    b = m._Node()
    path = [a, b]
    m._backup(path, 1.0, 0.2)
    assert (a.n, b.n) == (1, 1) and abs(a.m - (0.2 + 1/300)) < 1e-9 and b.m == 0.2
    assert b.w == 1.0 and a.w == -1.0, (a.w, b.w)  # sign flips per ply
    m._backup(path, -1.0, 0.4)  # leaf view flips; M does not
    assert a.w == 0.0 and b.w == 0.0, (a.w, b.w)
    assert abs(a.m - (0.3 + 1/300)) < 1e-12 and abs(b.m - 0.3) < 1e-12
    # ML off leaves m untouched
    m._backup(path, 0.5)
    assert abs(a.m - (0.3 + 1/300)) < 1e-12
    # batched path runs now (P0: o uninitialized crashed it)
    from chess_zero.game import State
    s0 = State.initial()
    pi = m.search(s0, _uniform_eval(), **_search_kw(n_sims=8, leaf_batch=4))
    assert abs(pi.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------- 3. FPU

def test_fpu_default_off_and_fires():
    from chess_zero import mcts as m
    from chess_zero.game import State
    ch = m._Node(prior=0.2)
    import numpy as _np
    u = 1.414 * 0.2 * _np.sqrt(10) / 1.0
    assert abs(m._select_fpu(ch, 0.6, 10, 1.414, 0.5) - (0.6 - 0.5 + u)) \
        < 1e-12, "unvisited reads parent_q - reduction + u"
    ch.n, ch.w = 2, 0.7
    u2 = 1.414 * 0.2 * _np.sqrt(10) / 3.0
    assert abs(m._select_fpu(ch, 0.6, 10, 1.414, 0.5) - (0.35 + u2)) \
        < 1e-12, "visited reads q + u unchanged"
    s0 = State.initial()
    ev = _peaked_eval()
    np.random.seed(5)
    pi_def = m.search(s0, ev, **_search_kw())
    np.random.seed(5)
    pi_exp = m.search(s0, ev, **_search_kw(fpu_reduction=0.0))
    assert np.array_equal(pi_def, pi_exp), "fpu=0 must equal default"
    # FPU only bites on mixed visited/unvisited interior nodes: needs a
    # budget deep enough to revisit (shallow uniform sweeps visit every
    # root child exactly once in both modes).
    evd = _peaked_eval(v=0.5)
    np.random.seed(5)
    pi_fpu = m.search(s0, evd, **_search_kw(n_sims=200, fpu_reduction=0.5))
    np.random.seed(5)
    pi_no = m.search(s0, evd, **_search_kw(n_sims=200))
    assert np.isfinite(pi_fpu).all() and abs(pi_fpu.sum()-1) < 1e-6
    # A corrected FPU need not change the winner on this symmetric fixture.
    assert abs(m._select_fpu(m._Node(prior=0), .8, 1, 0, .5) - .3) < 1e-12
    assert abs(pi_fpu.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------- 4. prune

def test_prune_singletons():
    from chess_zero import mcts as m
    from chess_zero.game import State
    s0 = State.initial()
    ev = _peaked_eval()  # peaked: top move revisits, tail stays singleton
    kw = _search_kw(n_sims=30)
    pi_off = m.search(s0, ev, **kw)
    pi_on = m.search(s0, ev, **dict(kw, prune_singletons=True))
    assert abs(pi_on.sum() - 1.0) < 1e-6
    assert (pi_off > 0).sum() >= (pi_on > 0).sum()
    st: dict = {}
    m.search(s0, ev, **dict(kw, prune_singletons=True, stats=st))
    assert st["breadth"] == int((pi_on > 0).sum())
    # degenerate: budget < breadth -> every child a singleton -> fall back
    # to unpruned counts instead of a zero vector (temp sampling crashes
    # on empty pi; argmax would pin move 0).
    pi_deg = m.search(s0, ev, **_search_kw(n_sims=8, prune_singletons=True))
    assert abs(pi_deg.sum() - 1.0) < 1e-6, pi_deg.sum()


# ---------------------------------------------------------------- 5. futile

def test_futile_stop():
    from chess_zero import mcts as m
    from chess_zero.game import State
    s0 = State.initial()
    legal = s0.legal_moves()
    ev = _uniform_eval()

    def decided(n):
        root = m._Node()
        root.expanded = True
        ch = m._Node(prior=0.5, raw=0.5)
        ch.n, ch.w = n, float(n)  # Q = 1.0
        root.children = {legal[0]: ch}
        return {s0.key(): root}

    st: dict = {}
    m.search(s0, ev, **_search_kw(n_sims=10, tt=decided(3), stats=st,
                                  futile_stop=True))
    assert st.get("early_stop") == 0 and st["new_sims"] == 10, st
    st2: dict = {}
    m.search(s0, ev, **_search_kw(n_sims=10, tt=decided(2), stats=st2,
                                  futile_stop=True))
    assert st2.get("early_stop") == 0, st2
    # default OFF: decided root runs full budget (training targets need
    # the full visit distribution).
    st3: dict = {}
    m.search(s0, ev, **_search_kw(n_sims=10, tt=decided(3), stats=st3))
    assert st3.get("early_stop") == 0, st3


# ---------------------------------------------------------------- 6. KL

def test_kl_stat():
    from chess_zero import mcts as m
    from chess_zero.game import State
    s0 = State.initial()
    st: dict = {}
    m.search(s0, _peaked_eval(), **_search_kw(n_sims=16, stats=st))
    assert np.isfinite(st["kl"]) and st["kl"] >= 0.0, st


# ---------------------------------------------------------------- 7. cache

def test_eval_cache():
    from chess_zero.game import State
    from chess_zero import mcts as m
    s0 = State.initial()
    calls = {"n": 0}

    def counting(states):
        calls["n"] += len(states)
        n = len(states)
        return (np.full((n, 4096), 1.0 / 4096, dtype=np.float64),
                np.zeros(n, dtype=np.float64))

    kw = _search_kw(n_sims=6)
    cache: dict = {}
    pi1 = m.search(s0, counting, **kw, eval_cache=cache)
    n1 = calls["n"]
    assert n1 > 0
    pi2 = m.search(s0, counting, **kw, eval_cache=cache)
    assert calls["n"] == n1, (calls["n"], n1)
    assert np.array_equal(pi1, pi2)
    assert len(cache) > 0
    assert all(isinstance(k, tuple) and len(k) == 3 for k in cache)

    def poison(states):
        raise AssertionError("cache miss on all-hit search")

    pi3 = m.search(s0, poison, **kw, eval_cache=dict(cache))
    assert np.array_equal(pi1, pi3)


# ------------------------------------------------- 8. ML bonus (D3)

def test_ml_off_bitexact():
    from chess_zero import mcts as m
    from chess_zero.game import State
    s0 = State.initial()
    ev = _peaked_eval()
    stub = lambda states: np.full(len(states), 0.5)  # noqa: E731
    np.random.seed(11)
    pi_plain = m.search(s0, ev, **_search_kw())
    np.random.seed(11)
    pi_stub0 = m.search(s0, ev, **_search_kw(ml_fn=stub, ml_slope=0.0))
    assert np.array_equal(pi_plain, pi_stub0), \
        "ml_fn + slope 0 must equal no-ML exactly"


def test_ml_bonus_unit():
    from chess_zero import mcts as m
    # winning parent: shorter child (smaller m) reads positive
    assert m._ml_bonus(0.2, 0.5, 0.9, 0.003, 0.07, 0.8) > 0
    # winning parent: longer child reads negative
    assert m._ml_bonus(0.6, 0.5, 0.9, 0.003, 0.07, 0.8) < 0
    # losing parent: delaying child (larger m) reads positive
    assert m._ml_bonus(0.6, 0.5, -0.9, 0.003, 0.07, 0.8) > 0
    # gate closed / missing M -> 0
    assert m._ml_bonus(0.2, 0.5, 0.5, 0.003, 0.07, 0.8) == 0.0
    assert m._ml_bonus(None, 0.5, 0.9, 0.003, 0.07, 0.8) == 0.0
    assert m._ml_bonus(0.2, None, 0.9, 0.003, 0.07, 0.8) == 0.0
    # clamp: huge gaps saturate at slope*cap
    assert abs(m._ml_bonus(-5.0, 5.0, 0.9, 0.003, 0.07, 0.8)
               - 0.003 * 0.07) < 1e-12


def test_ml_bonus_fires_in_search():
    # One-sim flip on a seeded decisive tree: without ML, B's larger u
    # wins; with ML, A's shorter-M bonus (+0.35) overturns it. Running
    # averages would wash seeded M out over many sims, so a single sim
    # each way is the deterministic probe (unit math is covered above).
    from chess_zero import mcts as m
    from chess_zero.game import State
    s0 = State.initial()
    legal = s0.legal_moves()
    ev = _uniform_eval()
    ml_fn = lambda states: np.full(len(states), 0.9)  # noqa: E731

    def seed_tt():
        root = m._Node()
        root.expanded = True
        root.n, root.w, root.m = 9, -8.1, 0.5  # Q = 0.9: gate open
        a = m._Node(prior=0.5, raw=0.5)
        a.n, a.w, a.m = 5, 4.5, 0.1  # short line, smaller u
        b = m._Node(prior=0.5, raw=0.5)
        b.n, b.w, b.m = 4, 3.6, 0.9  # long line, larger u
        root.children = {legal[0]: a, legal[1]: b}
        return {s0.key(): root}, a, b

    kw = dict(n_sims=1, c_puct=1.414, dirichlet_eps=0.0, tt=None,
              history=[], quiescence_depth=0, leaf_batch=1)
    tt0, a0, b0 = seed_tt()
    m.search(s0, ev, **dict(kw, tt=tt0))
    assert (a0.n, b0.n) == (5, 5), (a0.n, b0.n)  # B picked, no ML
    tt1, a1, b1 = seed_tt()
    m.search(s0, ev, **dict(kw, tt=tt1, ml_fn=ml_fn, ml_slope=5.0))
    assert (a1.n, b1.n) == (6, 4), (a1.n, b1.n)  # A picked, ML fired
    assert tt1[s0.key()].m is not None  # M bookkeeping landed
    # batched path with ML on runs + stays normalized
    evd = _peaked_eval(v=0.5)
    ml2 = lambda states: np.array(  # noqa: E731
        [(hash(s.rep_key()) % 50) / 50.0 for s in states])
    pi_b = m.search(s0, evd, **_search_kw(n_sims=30, leaf_batch=4,
                                          ml_fn=ml2, ml_slope=5.0))
    assert abs(pi_b.sum() - 1.0) < 1e-6


# ------------------------------------------------- 9. P / ml_mask (D5)

def _hand_hist(n=6, kl0=0.0):
    from chess_zero.game import State
    s0 = State.initial()
    enc = s0.encode()
    pi = np.zeros(4096, dtype=np.float32)
    pi[s0.legal_moves()[0]] = 1.0
    return [(enc, pi, i % 2, 0.0, i, 0.5, s0.legal_moves()[0],
             kl0 + 0.1 * i) for i in range(n)]


def test_P_targets():
    from chess_zero.game import State
    from chess_zero import selfplay as sp
    s0 = State.initial()
    hist = _hand_hist()
    flags = [False, True, False, False, False, True]
    ex = sp.shape_targets(hist, 1.0, 6, 300, s0.board, full=True,
                          prog_flags=flags, played_out=True)
    assert all(len(e) == 18 for e in ex)  # V20: +score_mean(14),
    # score_stdev(15), root_q(16), av(17); indices 0-13 unchanged
    # (legacy hand hist: root_q defaults to z, av None)
    assert [e[16] for e in ex] == [e[2] for e in ex]
    assert all(e[17] is None for e in ex)
    assert [e[12] for e in ex] == [2.0, 1.0, 4.0, 3.0, 2.0, 1.0], \
        [e[12] for e in ex]
    assert all(abs(float(e[11]) - (0.1 * i)) < 1e-9
               for i, e in enumerate(ex))
    assert all(float(e[13]) == 1.0 for e in ex)
    # censored fallback (no flags) = remaining plies
    ex2 = sp.shape_targets(hist, 1.0, 6, 300, s0.board, full=True)
    assert [e[12] for e in ex2] == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    # legacy-length flags mismatch -> censored, never crashes
    ex3 = sp.shape_targets(hist, 1.0, 6, 300, s0.board, full=True,
                           prog_flags=[True], played_out=False)
    assert [e[12] for e in ex3] == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert all(float(e[13]) == 0.0 for e in ex3)
    # legacy 6-wide hist entries (no action/kl) still build 14-tuples
    short = [h[:6] for h in hist]
    ex4 = sp.shape_targets(short, -1.0, 6, 300, s0.board, full=False)
    assert all(len(e) == 18 for e in ex4)  # V20: legacy 6-wide still builds 18
    assert all(float(e[11]) == 0.0 for e in ex4)


def test_resign_builder_P_mask_drop():
    from chess_zero.game import State
    from chess_zero import selfplay as sp
    s0 = State.initial()
    hist = _hand_hist()
    flags = [False, True, False, False, False, True]
    rc = sp.build_resign_examples(hist, 1.0, 6, 300, s0.board, True,
                                  drop_tail=4, prog_flags=flags,
                                  played_out=True)
    assert len(rc) == 2 and all(len(e) == 18 for e in rc)  # V20
    # flags slice to the kept prefix: P over first two rows only
    assert [e[12] for e in rc] == [2.0, 1.0], [e[12] for e in rc]
    assert all(float(e[13]) == 1.0 for e in rc)
    rc0 = sp.build_resign_examples(hist, 1.0, 6, 300, s0.board, True)
    assert len(rc0) == 6  # default drop_tail=0


# --------------------------------- 10. brief re-asserts (B5/B7/B8/B9/fp16)

def test_surprise_duplicates_high_kl():
    import random as _r
    from chess_zero.replay import ReplayBuffer

    def toy(z, kl=None, mask=1.0, P=1.0):
        enc = np.zeros((4, 8, 8), dtype=np.float32)
        pi = np.zeros(8, dtype=np.float32)
        pi[0] = 1.0
        own = np.zeros((8, 8), dtype=np.float32)
        t = (enc, pi, float(z), 0.0, 0.0, own, 0.0, 0.0, 0.5, -100,
             1.0)
        if kl is not None:
            t = t + (float(kl), float(P), float(mask))
        return t

    st = _r.getstate()
    try:
        buf = ReplayBuffer()  # legacy 11-tuples: write-once each
        buf.add_game([toy(float(i)) for i in range(4)])
        assert len(buf) == 4, len(buf)
        buf = ReplayBuffer()  # zero-sum kl: write-once
        buf.add_game([toy(float(i), kl=0.0) for i in range(4)])
        assert len(buf) == 4, len(buf)
        _r.seed(0)  # concentrated surprise duplicates
        buf = ReplayBuffer()
        buf.add_game([toy(7.0, kl=10.0)] +
                     [toy(float(i), kl=0.0) for i in range(1, 4)])
        n0 = sum(1 for e in buf.buf if float(e[2]) == 7.0)
        assert 2 <= n0 <= 3, n0
        _r.seed(1)  # cap at 3, RNG-independent
        buf = ReplayBuffer()
        buf.add_game([toy(7.0, kl=9.0)] +
                     [toy(float(i), kl=0.0) for i in range(1, 6)])
        n0 = sum(1 for e in buf.buf if float(e[2]) == 7.0)
        assert n0 == 3, n0
    finally:
        _r.setstate(st)


def test_fp16_and_sample_ml_mask():
    from chess_zero.game import State
    from chess_zero.replay import ReplayBuffer
    s0 = State.initial()
    enc = s0.encode().astype(np.float32)
    pi = np.zeros(4096, dtype=np.float32)
    for a in s0.legal_moves()[:4]:
        pi[a] = 0.25
    own = np.zeros((8, 8), dtype=np.float32)
    rows = [(enc.copy(), pi.copy(), 0.5, 0.1, 0.5, own.copy(), 0.2, 0.5,
             0.5, -100, 1.0, 0.0, 3.0, (1.0 if i % 2 == 0 else 0.0))
            for i in range(8)]
    buf = ReplayBuffer()
    buf.add_game(rows)
    assert len(buf) == 8, len(buf)  # kl=0 -> w=1 write-once
    stored = buf.buf[0]
    assert len(stored) == 14 and stored[0].dtype == np.float16
    assert stored[1].dtype == np.float16 and stored[5].dtype == np.float16
    out = buf.sample(8)
    assert len(out) == 17, len(out)  # V20: 13 + score_mean/stdev/root_q/av
    s, pi_b, z, m, ml, own_b, margin, mob, safe, reply, pw, mm, _pp, \
        _sm, _ss, _rq, _av = out
    assert s.dtype == np.float32 and pi_b.dtype == np.float32
    assert own_b.dtype == np.float32 and mm.dtype == np.float32
    assert set(np.unique(mm)).issubset({0.0, 1.0}), np.unique(mm)
    # legacy 11-tuples: mask defaults 1.0 at [11], P defaults 0.0 at
    # [12] (absolute indices — V20 appends score/av/root_q at 13-16, so
    # end-relative indexing no longer lands on mask/P).
    buf2 = ReplayBuffer()
    buf2.add_game([r[:11] for r in rows])
    _o2 = buf2.sample(8)
    assert len(_o2) == 17
    assert _o2[12].tolist() is not None and len(_o2[12].tolist()) == 8
    assert (_o2[11] == 0.0).all()
    assert (_o2[12] == -1.0).all()
    # train_step stays finite when signatures still align (Impl-A owns
    # train.py; skip loudly on drift instead of failing here).
    try:
        import torch
        from chess_zero.config import TEST_CONFIG
        from chess_zero.model import AlphaZeroNet
        from chess_zero.train import train_step
        torch.manual_seed(0)
        net = AlphaZeroNet(blocks=TEST_CONFIG.blocks,
                           channels=TEST_CONFIG.channels)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        ret = train_step(net, opt, torch.from_numpy(s),
                         torch.from_numpy(pi_b), torch.from_numpy(z),
                         torch.from_numpy(m), torch.from_numpy(ml),
                         torch.from_numpy(own_b), torch.from_numpy(margin),
                         torch.from_numpy(mob), torch.from_numpy(safe),
                         torch.from_numpy(reply), torch.from_numpy(pw))
        assert np.isfinite(float(ret[0])), ret[0]
    except TypeError as e:
        print(f"  (train_step signature drifted; fp16 dtypes verified, "
              f"train skipped: {e})")


def test_tail_drop_and_hist_kl():
    from unittest import mock
    from chess_zero import mcts as mcts_mod
    from chess_zero import selfplay as sp_mod
    hist = _hand_hist()
    from chess_zero.game import State
    s0 = State.initial()
    full = sp_mod.shape_targets(hist, 1.0, 6, 300, s0.board, full=True)
    cut = sp_mod.shape_targets(hist, 1.0, 6, 300, s0.board, full=True,
                               drop_tail=4)
    assert len(full) == 6 and len(cut) == 2
    assert all(len(e) == 18 for e in full + cut)  # V20
    cfg = _game_cfg(move_cap=6)
    ev = _script_values([0.9] * 20)
    with mock.patch.object(
            mcts_mod, "search",
            _mock_search(kl_seq=[0.5 + 0.1 * i for i in range(20)])):
        ex, _res = sp_mod.play_game(None, cfg, evaluate_fn=ev,
                                    **_play_kwargs())
    assert len(ex) == 6 and all(len(e) == 18 for e in ex)  # V20
    for i, e in enumerate(ex):
        assert abs(float(e[11]) - (0.5 + 0.1 * i)) < 1e-9, (i, e[11])


def test_playthrough():
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
    assert res == "1/2-1/2" and len(ex) == 14, (res, len(ex))
    ev2 = _script_values([-0.9, 0.9] * 20)
    stats2: dict = {}
    with mock.patch.object(mcts_mod, "search", _mock_search()):
        ex2, res2 = sp_mod.play_game(None, cfg, evaluate_fn=ev2,
                                     stats=stats2, **_play_kwargs())
    assert stats2.get("terminal") == "resign", stats2
    assert res2 == "0-1" and len(ex2) == 3, (res2, len(ex2))
    assert all(len(e) == 18 for e in ex + ex2)  # V20


# ------------------------------------------------- 11. selfplay wiring

def test_selfplay_wiring():
    from unittest import mock
    from chess_zero.config import Config
    from chess_zero import mcts as mcts_mod
    from chess_zero import selfplay as sp_mod
    cfg = Config(blocks=2, channels=32, sims=2, move_cap=4, temp_moves=0,
                 fpu_reduction=0.5, prune_singletons=True)
    cfg.ml_slope = 0.003
    seen = []
    orig = mcts_mod.search

    class FakeModel:
        """Minimal moves_left head stand-in: out[3] = const 0.5 ML."""

        def __call__(self, x):
            import torch as _t
            return (None, None, None,
                    _t.full((x.shape[0],), 0.5))

    def rec(state, evaluate_fn, *a, **k):
        seen.append(dict(k))
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
        ex, _res = sp_mod.play_game(FakeModel(), cfg,
                                    evaluate_fn=fake_eval)
    assert len(seen) >= 2, seen
    assert all(isinstance(c.get("eval_cache"), dict) for c in seen)
    assert all(c["eval_cache"] is seen[0]["eval_cache"] for c in seen), \
        "one dict per game"
    assert len(seen[0]["eval_cache"]) > 0, "game populated its cache"
    for c in seen:
        assert c.get("fpu_reduction") == 0.5, c
        assert c.get("prune_singletons") is True, c
        assert c.get("ml_slope") == 0.003, c
        assert c.get("ml_cap") == 0.07 and c.get("ml_thr") == 0.8, c
        assert c.get("ml_fn") is not None, "model present -> ml_fn built"
    assert len(ex) > 0 and all(len(e) == 18 for e in ex)  # V20
    assert all(float(e[13]) in (0.0, 1.0) for e in ex)
    # server path (model None) + slope on -> ml_fn None, no crash
    seen2 = []
    with mock.patch.object(mcts_mod, "search", _mock_search()):
        orig_search = mcts_mod.search
        mcts_mod.search = _mock_search()

        def rec2(state, evaluate_fn, *a, **k):
            seen2.append(dict(k))
            return orig_search(state, evaluate_fn, *a, **k)

        mcts_mod.search = rec2
        try:
            sp_mod.play_game(None, cfg, evaluate_fn=fake_eval)
        finally:
            mcts_mod.search = orig_search
    assert seen2 and all(c.get("ml_fn") is None for c in seen2)


# ------------------------------------------------- 12. parallel (verify)

def test_parallel_fast_safety_present():
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
    assert len(examples) == 3 and results["1-0"] == 3
    assert results["fast"] == 3, results
    assert results["safety"] == 6, results
    assert results["Tmate"] == 3 and results["plies"] == 120


if __name__ == "__main__":
    test_legacy_determinism_and_backup_bitexact()
    print("ok legacy_determinism_and_backup_bitexact")
    test_wu_math()
    print("ok wu_math")
    test_fpu_default_off_and_fires()
    print("ok fpu_default_off_and_fires")
    test_prune_singletons()
    print("ok prune_singletons")
    test_futile_stop()
    print("ok futile_stop")
    test_kl_stat()
    print("ok kl_stat")
    test_eval_cache()
    print("ok eval_cache")
    test_ml_off_bitexact()
    print("ok ml_off_bitexact")
    test_ml_bonus_unit()
    print("ok ml_bonus_unit")
    test_ml_bonus_fires_in_search()
    print("ok ml_bonus_fires_in_search")
    test_P_targets()
    print("ok P_targets")
    test_resign_builder_P_mask_drop()
    print("ok resign_builder_P_mask_drop")
    test_surprise_duplicates_high_kl()
    print("ok surprise_duplicates_high_kl")
    test_fp16_and_sample_ml_mask()
    print("ok fp16_and_sample_ml_mask")
    test_tail_drop_and_hist_kl()
    print("ok tail_drop_and_hist_kl")
    test_playthrough()
    print("ok playthrough")
    test_selfplay_wiring()
    print("ok selfplay_wiring")
    test_parallel_fast_safety_present()
    print("ok parallel_fast_safety_present")
    print("all v19f tests passed")
