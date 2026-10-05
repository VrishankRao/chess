"""V21.1 F8 game-leg gates: TT-carryover parity across plies (REAL tree).

Read-only: best.pt via torch.load (no saves), no training launches, no
default flips (--rust-tree untouched; these tests drive both backends
explicitly). Env knobs: V21_GAMELEG_PLIES (default 6), V21_GAMELEG_SIMS
(default 400), V21_GAMELEG_SEED (default 5000), V21_DEVICE (auto).

Gates (all strict zeros):
- test_same_line_carryover: both backends search the SAME forced line
  (Python's pick applied to a shared state each ply). Per ply:
  max|dPi| == 0, ext identical, reused identical (>0 past ply 0),
  root_visits identical, argmax identical. TT + eval cache populate on
  both sides; post-prune stores hold <= 1 entry (F2 prune wiring).
- test_independent_game: each backend advances its OWN game from its own
  picks. Asserts 0 flips over the game (picks identical every ply) plus
  the same per-ply zeros (positions stay synced exactly because picks do).
- test_contempt_ml_carryover: same-line repeat with training contempt
  (0.3/asym) + in-tree ML on (slope 0.003, moves_left head) — exercises
  draw shaping + ML side channel + TT + eval cache + RNG together.

Base search params mirror the audit game-leg (V19 subset, 400 sims):
PUCT/FPU/prune/batch/quiescence on; contempt/ML off except where noted.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)

import numpy as np

import test_mcts_parity as H
from chess_zero.mcts_bridge import (
    MpsEvalBridge,
    PythonTreeBackend,
    RustTreeBackend,
    new_game_store,
)
from chess_zero import mcts as mcts_mod

SIMS = int(os.environ.get("V21_GAMELEG_SIMS", "400"))
PLIES = int(os.environ.get("V21_GAMELEG_PLIES", "6"))
ML_PLIES = int(os.environ.get("V21_GAMELEG_ML_PLIES", "4"))
SEED_BASE = int(os.environ.get("V21_GAMELEG_SEED", "5000"))

BASE_KW = dict(
    c_puct=1.6, dirichlet_alpha=0.3, dirichlet_eps=0.25,
    quiescence_depth=2, forcing_bonus=0.25, contempt=0.0,
    asymmetric_contempt=False, contempt_edge_scale=0.0,
    leaf_batch=8, virtual_loss=1.0, fpu_reduction=0.5,
    prune_singletons=True, ml_slope=0.0, futile_stop=False,
)
CONtempt_ML_KW = dict(
    c_puct=1.6, dirichlet_alpha=0.3, dirichlet_eps=0.25,
    quiescence_depth=2, forcing_bonus=0.25, contempt=0.3,
    asymmetric_contempt=True, contempt_edge_scale=0.0,
    leaf_batch=8, virtual_loss=1.0, fpu_reduction=0.5,
    prune_singletons=True, ml_slope=0.003, ml_cap=0.07, ml_thr=0.8,
    futile_stop=False,
)


def _ml_fn_from_model(model, device):
    def ml_fn(states):
        import torch as _t
        x = _t.stack(
            [_t.from_numpy(np.asarray(s.encode(), dtype=np.float32))
             for s in states]).to(device)
        with _t.no_grad():
            out = model(x)
        return np.asarray(out[3].detach().cpu(), dtype=np.float64).flatten()
    return ml_fn


def _search(py_backend, ru_backend, ru_bridge, state, ev, seed, n_sims,
            kw, py_tt, py_cache, ru_store, ml_fn=None):
    import random as _r
    k_py = dict(kw)
    k_py["tt"] = py_tt
    k_py["history"] = list(_search.hist)
    k_py["eval_cache"] = py_cache
    if ml_fn is not None:
        k_py["ml_fn"] = ml_fn
    np.random.seed(seed)
    _r.seed(seed + 999)
    pi_py, st_py = py_backend.search(state, ev, n_sims, **k_py)

    k_ru = dict(kw)
    k_ru["tt"] = ru_store
    k_ru["history"] = list(_search.hist)
    k_ru["eval_cache"] = ru_store
    if ml_fn is not None:
        k_ru["ml_fn"] = ml_fn
    np.random.seed(seed)
    _r.seed(seed + 999)
    if ru_bridge is not None:
        ru_bridge.reset_stats()
    pi_ru, st_ru = ru_backend.search(state, ev, n_sims, **k_ru)
    return (np.asarray(pi_py, dtype=np.float64),
            np.asarray(pi_ru, dtype=np.float64), st_py, st_ru)


_search.hist = []


def _assert_ply_zero(ply, pi_py, pi_ru, st_py, st_ru):
    dpi = float(np.abs(pi_py - pi_ru).max())
    assert dpi == 0.0, (ply, dpi)
    assert int(pi_py.argmax()) == int(pi_ru.argmax()), ply
    assert int(st_py.get("extensions", -1)) == \
        int(st_ru.get("extensions", -1)), (ply, st_py, st_ru)
    assert int(st_py.get("reused", -1)) == \
        int(st_ru.get("reused", -1)), (ply, st_py, st_ru)
    assert int(st_py.get("root_visits", -1)) == \
        int(st_ru.get("root_visits", -1)), (ply, st_py, st_ru)
    if ply > 0:
        # Carryover actually fired on both sides past the first move.
        assert int(st_py.get("reused", 0)) > 0, (ply, st_py)
        assert int(st_ru.get("reused", 0)) > 0, (ply, st_ru)
    return dpi


def _prune_both(py_tt, ru_store, state):
    mcts_mod.prune_tt(py_tt, state.rep_key())
    ru_store.prune_to(state.rep_key())
    assert len(py_tt) <= 1, len(py_tt)
    assert int(ru_store.tt_len()) <= 1, int(ru_store.tt_len())


def test_same_line_carryover():
    """Forced shared line: per-ply zeros + ext/reused equality."""
    import chess_zero.game as _g
    import chess_zero.game as G
    dev = H._device()
    ev, _model, old_planes = H.load_shared_evaluate(dev)
    try:
        py_b, ru_b = PythonTreeBackend(), RustTreeBackend()
        bridge = MpsEvalBridge(ev)
        ru_b = RustTreeBackend(bridge=bridge)
        bridge.inner = ev
        py_tt, py_cache = {}, {}
        ru_store = new_game_store()
        state = G.State.initial()
        _search.hist = [state.rep_key()]
        max_dpi = 0.0
        for ply in range(PLIES):
            seed = SEED_BASE + ply
            pi_py, pi_ru, st_py, st_ru = _search(
                py_b, ru_b, bridge, state, ev, seed, SIMS, BASE_KW,
                py_tt, py_cache, ru_store)
            max_dpi = max(max_dpi, _assert_ply_zero(
                ply, pi_py, pi_ru, st_py, st_ru))
            assert len(py_tt) > 0, ply
            assert int(ru_store.tt_len()) > 0, ply
            assert len(py_cache) > 0, ply
            assert int(ru_store.cache_len()) > 0, ply
            pick = int(pi_py.argmax())
            state = state.apply(pick)
            _prune_both(py_tt, ru_store, state)
            _search.hist.append(state.rep_key())
            done, _ = state.is_terminal()
            assert not done, ply
        print(f"[same-line] plies={PLIES} sims={SIMS} max|dPi|={max_dpi:.1e} "
              f"ext+reused equal every ply", flush=True)
    finally:
        _g.INPUT_PLANES = old_planes


def test_independent_game():
    """Each backend plays its own game: 0 flips, per-ply zeros."""
    import chess_zero.game as _g
    import chess_zero.game as G
    dev = H._device()
    ev, _model, old_planes = H.load_shared_evaluate(dev)
    try:
        py_b = PythonTreeBackend()
        bridge = MpsEvalBridge(ev)
        ru_b = RustTreeBackend(bridge=bridge)
        bridge.inner = ev
        py_tt, py_cache = {}, {}
        ru_store = new_game_store()
        st_py_state = G.State.initial()
        st_ru_state = G.State.initial()
        hist_py = [st_py_state.rep_key()]
        hist_ru = [st_ru_state.rep_key()]
        flips = 0
        max_dpi = 0.0
        for ply in range(PLIES):
            seed = SEED_BASE + 100 + ply
            import random as _r
            k_py = dict(BASE_KW, tt=py_tt, history=list(hist_py),
                        eval_cache=py_cache)
            np.random.seed(seed)
            _r.seed(seed + 999)
            pi_py, s_py = py_b.search(st_py_state, ev, SIMS, **k_py)
            pi_py = np.asarray(pi_py, dtype=np.float64)
            k_ru = dict(BASE_KW, tt=ru_store, history=list(hist_ru),
                        eval_cache=ru_store)
            np.random.seed(seed)
            _r.seed(seed + 999)
            bridge.reset_stats()
            pi_ru, s_ru = ru_b.search(st_ru_state, ev, SIMS, **k_ru)
            pi_ru = np.asarray(pi_ru, dtype=np.float64)
            dpi = float(np.abs(pi_py - pi_ru).max())
            max_dpi = max(max_dpi, dpi)
            assert dpi == 0.0, (ply, dpi)
            pk_py, pk_ru = int(pi_py.argmax()), int(pi_ru.argmax())
            if pk_py != pk_ru:
                flips += 1
            assert int(s_py.get("extensions", -1)) == \
                int(s_ru.get("extensions", -1)), ply
            assert int(s_py.get("reused", -1)) == \
                int(s_ru.get("reused", -1)), ply
            st_py_state = st_py_state.apply(pk_py)
            st_ru_state = st_ru_state.apply(pk_ru)
            mcts_mod.prune_tt(py_tt, st_py_state.rep_key())
            ru_store.prune_to(st_ru_state.rep_key())
            hist_py.append(st_py_state.rep_key())
            hist_ru.append(st_ru_state.rep_key())
        assert flips == 0, flips
        print(f"[independent] plies={PLIES} sims={SIMS} flips=0 "
              f"max|dPi|={max_dpi:.1e}", flush=True)
    finally:
        _g.INPUT_PLANES = old_planes


def test_contempt_ml_carryover():
    """Same-line repeat with contempt 0.3/asym + ML on."""
    import chess_zero.game as _g
    import chess_zero.game as G
    dev = H._device()
    ev, model, old_planes = H.load_shared_evaluate(dev)
    try:
        import torch as _t
        model.to(dev)
        model.eval()
        ml_fn = _ml_fn_from_model(model, dev)
        py_b = PythonTreeBackend()
        bridge = MpsEvalBridge(ev)
        ru_b = RustTreeBackend(bridge=bridge)
        bridge.inner = ev
        py_tt, py_cache = {}, {}
        ru_store = new_game_store()
        state = G.State.initial()
        _search.hist = [state.rep_key()]
        max_dpi = 0.0
        for ply in range(ML_PLIES):
            seed = SEED_BASE + 200 + ply
            pi_py, pi_ru, st_py, st_ru = _search(
                py_b, ru_b, bridge, state, ev, seed, SIMS,
                CONtempt_ML_KW, py_tt, py_cache, ru_store, ml_fn=ml_fn)
            max_dpi = max(max_dpi, _assert_ply_zero(
                ply, pi_py, pi_ru, st_py, st_ru))
            pick = int(pi_py.argmax())
            state = state.apply(pick)
            _prune_both(py_tt, ru_store, state)
            _search.hist.append(state.rep_key())
            done, _ = state.is_terminal()
            assert not done, ply
        print(f"[contempt-ml] plies={ML_PLIES} sims={SIMS} max|dPi|={max_dpi:.1e} "
              f"contempt=0.3/asym ml=on", flush=True)
    finally:
        _g.INPUT_PLANES = old_planes


if __name__ == "__main__":
    test_same_line_carryover()
    print("ok same_line_carryover")
    test_independent_game()
    print("ok independent_game")
    test_contempt_ml_carryover()
    print("ok contempt_ml_carryover")
    print("all game-leg tests passed")
