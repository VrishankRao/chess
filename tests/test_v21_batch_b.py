"""V21 Batch B tests (Agent: batch size + compile knobs).

Code-only: no training launch, no best.pt/history writes, no live-path
imports (loop/train never imported here). Offline, CPU tiny-net only
(workers<=1) so the live mac_run16 (7 workers) is undisturbed.

Run: PYTHONPATH=. python3 tests/test_v21_batch_b.py (or pytest).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def test_leaf_batch_default_is_8_and_frozen():
    from chess_zero import v21_batch_b as B
    from chess_zero.config import V20_CONFIG, Config
    assert B.LEAF_BATCH_DEFAULT == 8
    assert V20_CONFIG.leaf_batch == 8  # frozen live value, untouched
    assert Config().leaf_batch == 1  # base default untouched (no default change)
    assert B.leaf_batch_for() == 8
    assert B.leaf_batch_for(override=16) == 16
    assert B.leaf_batch_for(override=32) == 32
    assert B.leaf_batch_for(override=1) == 1
    assert B.leaf_batch_for(override=0) == 8  # invalid -> yesterday
    assert B.leaf_batch_for(override=999) == 8
    assert B.leaf_batch_for(override="bad") == 8
    # cfg passthrough without mutation
    from chess_zero.config import TEST_CONFIG
    before = TEST_CONFIG.leaf_batch
    assert B.leaf_batch_for(cfg=TEST_CONFIG) == int(before)
    assert TEST_CONFIG.leaf_batch == before  # never mutated


def test_compile_default_off_and_env_opt_in():
    from chess_zero import v21_batch_b as B
    assert B.USE_COMPILE_DEFAULT is False
    assert B.compile_enabled() is False  # default off, no env
    assert B.compile_enabled(flag=True) is True
    assert B.compile_enabled(flag=False) is False  # explicit wins over env
    os.environ[B.COMPILE_ENV] = "1"
    try:
        assert B.compile_enabled() is True
        assert B.compile_enabled(flag=False) is False  # explicit wins
    finally:
        del os.environ[B.COMPILE_ENV]
    assert B.compile_enabled() is False


def test_maybe_compile_default_returns_same_object():
    from chess_zero import v21_batch_b as B
    from chess_zero.model import AlphaZeroNet
    m = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    assert B.maybe_compile(m) is m  # default-off: identical object
    assert B.maybe_compile(m, enabled=False) is m


def test_maybe_compile_enabled_finite_and_close():
    from chess_zero import v21_batch_b as B
    from chess_zero.model import AlphaZeroNet
    torch.manual_seed(0)
    m = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    m.eval()
    c = B.maybe_compile(m, enabled=True)
    x = torch.randn(2, 13, 8, 8)
    with torch.no_grad():
        le, we = m.forward_pv(x)
        lc, wc = c.forward_pv(x) if c is not m else (le, we)
    assert bool(torch.isfinite(lc).all()) and bool(torch.isfinite(wc).all())
    pe = torch.softmax(le, dim=1).cpu().numpy()
    pc = torch.softmax(lc, dim=1).cpu().numpy()
    assert float(np.abs(pe - pc).max()) < 1e-4, float(np.abs(pe - pc).max())
    assert int((pe.argmax(1) == pc.argmax(1)).sum()) == 2


def test_numbers_table_in_comment_and_no_live_imports():
    import inspect
    from chess_zero import v21_batch_b as B
    src = inspect.getsource(B)
    for token in ("9.555", "2380", "0.1352", "5.53", "1.16x",
                  "LEAF_BATCH_DEFAULT", "USE_COMPILE_DEFAULT",
                  "CHESS_ZERO_COMPILE"):
        assert token in src, token
    assert "run_training" not in src  # no live-path coupling
    assert "V20_CONFIG" not in src or "V20_CONFIG.leaf_batch" in src


def test_v20_config_values_untouched():
    from chess_zero.config import V20_CONFIG, V19_CONFIG
    assert V20_CONFIG.leaf_batch == 8
    assert V20_CONFIG.sims == 400
    assert V20_CONFIG.c_puct == 1.2 and V20_CONFIG.fpu_reduction == 0.4
    assert V20_CONFIG.ml_thr == 0.9
    assert V19_CONFIG.ml_thr == 0.8  # V19 frozen


if __name__ == "__main__":
    test_leaf_batch_default_is_8_and_frozen()
    print("ok leaf_batch_default_frozen")
    test_compile_default_off_and_env_opt_in()
    print("ok compile_default_off_env")
    test_maybe_compile_default_returns_same_object()
    print("ok maybe_compile_passthrough")
    test_maybe_compile_enabled_finite_and_close()
    print("ok maybe_compile_finite_close")
    test_numbers_table_in_comment_and_no_live_imports()
    print("ok numbers_table_no_live_imports")
    test_v20_config_values_untouched()
    print("ok v20_frozen")
    print("all v21-batch-b tests passed")
