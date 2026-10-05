"""V21.1 E2 real-board parity: REAL mctscore tree vs Python tree.

Same net (checkpoints_mac/best.pt, READ-ONLY), same seeds, 50 fixed
positions (startpos + seeded plies, seed 20260925; per-position search
seed 1000+i reset on np+python RNGs) -> visit distributions + final
policy + Q.

Modes (mirroring tests/test_mcts_parity.py, which now exercises the same
real backend through the rust_available flip; this file additionally
GATES on the real entry point so a missing wheel fails loudly instead of
silently comparing the stub):
- EXACT: TT off + noise 0 -> visits identical, max|dPi| == 0, |dQ| == 0
  (hard asserts; closes verification P1-2 mock-only math).
- NOISY: TT on (fresh dict per side) + eps 0.25, same seeds ->
  divergence stats + loose bound 0.02 (measures, does not gate).

Code-only + offline: no training launch, no best.pt / history writes
(best.pt opened torch.load read-only; nothing saved), no loop
integration, no default changes. Env knobs inherited from
test_mcts_parity (V21_PARITY_NPOS default 50, V21_PARITY_SIMS default 64,
V21_DEVICE default mps-if-available).

Run: PYTHONPATH=. python3 tests/test_mcts_rust_parity.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)

import numpy as np

import test_mcts_parity as H
from chess_zero.mcts_bridge import (MpsEvalBridge, PythonTreeBackend,
                                    RustTreeBackend, rust_available)


def test_real_backend_present():
    """Gate: the PyO3 entry point must exist (else this file must fail,
    never silently fall back to the stub)."""
    assert rust_available(), "mctscore.search not importable -- build the wheel"
    import mctscore as _m
    assert hasattr(_m, "search"), "mctscore module has no search entry point"
    b = RustTreeBackend()
    assert b.name == "rust-tree", b.name
    print(f"[real-backend] mctscore {getattr(_m, '__version__', '?')} "
          f"backend='{b.name}'", flush=True)


def test_exact_parity_real_tree():
    """TT off + noise 0: bit-exact visits/policy/Q over NPOS positions."""
    import chess_zero.game as _g
    dev = H._device()
    ev, _model, old_planes = H.load_shared_evaluate(dev)
    try:
        positions = H.fixed_positions(H.NPOS)
        bridge = MpsEvalBridge(ev)
        max_dpi, sum_dpi, flips = 0.0, 0.0, 0
        max_dq, sum_dq = 0.0, 0.0
        for i, s in enumerate(positions):
            seed = H.RUN_SEED + i
            pi_a, v_a, q_a, _ = H.run_one(PythonTreeBackend, None, s, ev,
                                          seed, H.PARITY_SIMS, H.EXACT_KW)
            pi_b, v_b, q_b, st_b = H.run_one(RustTreeBackend, bridge, s, ev,
                                             seed, H.PARITY_SIMS, H.EXACT_KW)
            assert st_b.get("bridge", {}).get("rows", 0) > 0, \
                "real path produced no eval crossings"
            dpi = float(np.abs(pi_a - pi_b).max())
            dq = abs(float(q_a) - float(q_b))
            max_dpi = max(max_dpi, dpi)
            sum_dpi += dpi
            max_dq = max(max_dq, dq)
            sum_dq += dq
            if int(pi_a.argmax()) != int(pi_b.argmax()):
                flips += 1
            assert np.array_equal(np.round(v_a, 6), np.round(v_b, 6)), \
                (i, v_a[v_a > 0][:8], v_b[v_b > 0][:8])
            assert dpi == 0.0, (i, dpi)
            assert dq == 0.0, (i, dq)
        H._report("exact-real", max_dpi, sum_dpi / H.NPOS, flips, max_dq,
                  sum_dq / H.NPOS, H.NPOS)
    finally:
        _g.INPUT_PLANES = old_planes


def test_noisy_bound_real_tree():
    """TT on + noise 0.25, same seeds: divergence stats + loose bound."""
    import chess_zero.game as _g
    dev = H._device()
    ev, _model, old_planes = H.load_shared_evaluate(dev)
    try:
        positions = H.fixed_positions(H.NPOS)
        bridge = MpsEvalBridge(ev)
        max_dpi, sum_dpi, flips = 0.0, 0.0, 0
        max_dq, sum_dq = 0.0, 0.0
        for i, s in enumerate(positions):
            seed = H.RUN_SEED + i
            pi_a, _v_a, q_a, _ = H.run_one(PythonTreeBackend, None, s, ev,
                                           seed, H.PARITY_SIMS, H.NOISY_KW,
                                           tt={})
            pi_b, _v_b, q_b, _ = H.run_one(RustTreeBackend, bridge, s, ev,
                                           seed, H.PARITY_SIMS, H.NOISY_KW,
                                           tt={})
            dpi = float(np.abs(pi_a - pi_b).max())
            dq = abs(float(q_a) - float(q_b))
            max_dpi = max(max_dpi, dpi)
            sum_dpi += dpi
            max_dq = max(max_dq, dq)
            sum_dq += dq
            if int(pi_a.argmax()) != int(pi_b.argmax()):
                flips += 1
        H._report("noisy-real", max_dpi, sum_dpi / H.NPOS, flips, max_dq,
                  sum_dq / H.NPOS, H.NPOS)
        assert max_dpi < 0.02, ("noisy divergence blew past 0.02 -- "
                                f"bridge reorders evals? max={max_dpi}")
    finally:
        _g.INPUT_PLANES = old_planes


if __name__ == "__main__":
    test_real_backend_present()
    print("ok real_backend_present")
    test_exact_parity_real_tree()
    print("ok exact_parity_real_tree")
    test_noisy_bound_real_tree()
    print("ok noisy_bound_real_tree")
    print("all mcts-rust-parity tests passed")
