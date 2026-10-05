"""V21 batch-D parity harness: Python-tree vs Rust-tree (stub) via the bridge.

Spec: same net (checkpoints_mac/best.pt, READ-ONLY), same seeds, 50 fixed
positions -> visit distributions + final policy + Q.
Tolerance: EXACT match with TT off + noise 0 (max|dPi| == 0, visits
identical, |dQ| == 0); otherwise (TT on and/or noise > 0) report
divergence stats (max/mean|dPi|, argmax flips, Q diffs).
Timing: 400-sim search wall-clock, Rust-tree(stub) vs Python-tree.

Bridge choice: batch-queue over PyO3 (see chess_zero/mcts_bridge.py +
/tmp/v21_bridge.md). Batch C plugs a real `mctscore` module in behind
the same TreeBackend seam; this harness then runs unchanged (rust_available
flip is the only switch, no harness edits needed).

PARITY (measured 2026-09-25, best.pt meta iter27, device MPS, sims=64,
TT off, eps=0, seeds 1000+i, N=50): max|dPi|=0.000e+00, mean|dPi|=0.000e+00,
argmax_flips=0/50, max|dQ|=0.000e+00 -- EXACT, visits identical all 50.
NOISY (same 50, TT on, eps=0.25, same seeds): max|dPi|=0.000e+00,
flips=0/50, max|dQ|=0.000e+00 (< 0.02 bound). Bonus probe (not in-suite):
V20 live search params (c_puct 1.2, FPU 0.4, prune on, leaf_batch 8,
48 sims, 2 positions): max|dPi|=0.00e+00, dQ=0.00e+00, ext identical.
TIMING 400-sim (same box, live mac_run16 loaded, single-process harness):
pos0 python-tree 6.72s (59.5 evals/s, ext 58) vs rust-stub 6.62s
(60.4 evals/s, 459 crossings/459 rows) ratio 0.985; pos1 python 5.70s
(70.2 evals/s, ext 9) vs stub 5.98s (66.9 evals/s, 410/410) ratio 1.048.
Verdict: bridge overhead ~= noise floor (+-5% on a loaded box); true Rust
speedup lands with batch C (tree work is small vs NN forwards).

Code-only + offline microbenchmarks: no training launch, no best.pt /
history writes (best.pt opened torch.load read-only; nothing saved),
no loop integration, no default changes. Env knobs (CI quick mode):
V21_PARITY_NPOS (default 50), V21_PARITY_SIMS (default 64),
V21_TIMING_SIMS (default 400), V21_DEVICE (default mps-if-available),
V21_SKIP_TIMING=1 (parity only).

Run: PYTHONPATH=. python3 tests/test_mcts_parity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST = os.path.join(REPO, "checkpoints_mac", "best.pt")

NPOS = int(os.environ.get("V21_PARITY_NPOS", "50"))
PARITY_SIMS = int(os.environ.get("V21_PARITY_SIMS", "64"))
TIMING_SIMS = int(os.environ.get("V21_TIMING_SIMS", "400"))
SEED_BASE = 20260925
RUN_SEED = 1000

BLOCKS, CHANNELS, PLANES = 6, 64, 30

# Exact-mode search params (sequential stub path; ML/contempt off so the
# comparison isolates tree+bridge exactness, not head wiring).
EXACT_KW = dict(c_puct=1.414, dirichlet_alpha=0.3, dirichlet_eps=0.0,
                quiescence_depth=2, forcing_bonus=0.25, contempt=0.0,
                asymmetric_contempt=False, contempt_edge_scale=0.0,
                leaf_batch=1, virtual_loss=1.0, fpu_reduction=0.0,
                prune_singletons=False, ml_slope=0.0, futile_stop=False)
# Noisy-mode params (TT on + Dirichlet on): divergence STATS only.
NOISY_KW = dict(c_puct=1.414, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                quiescence_depth=2, forcing_bonus=0.25, contempt=0.0,
                asymmetric_contempt=False, contempt_edge_scale=0.0,
                leaf_batch=1, virtual_loss=1.0, fpu_reduction=0.0,
                prune_singletons=False, ml_slope=0.0, futile_stop=False)


def _device():
    want = os.environ.get("V21_DEVICE", "auto")
    if want != "auto":
        return want
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_shared_evaluate(device):
    """Load best.pt READ-ONLY once; return (evaluate_fn, model). One shared
    closure serves BOTH backends, so device numerics cannot separate them."""
    import torch
    import chess_zero.game as _g
    old_planes = _g.INPUT_PLANES
    _g.INPUT_PLANES = PLANES
    from chess_zero.game import State  # noqa: F401 (planes stamp check)
    from chess_zero.model import AlphaZeroNet, load_weights, infer_se_ratio
    ckpt = torch.load(BEST, map_location="cpu", weights_only=False)
    w = ckpt.get("weights", ckpt) if isinstance(ckpt, dict) else ckpt
    se = infer_se_ratio(w)
    model = AlphaZeroNet(blocks=BLOCKS, channels=CHANNELS, planes=PLANES,
                         se_ratio=se)
    load_weights(model, w)
    model.to(device)
    model.eval()

    def evaluate_fn(states):
        import torch as _t
        x = np.stack([s.encode() for s in states]).astype(np.float32)
        with _t.no_grad():
            out = model(_t.from_numpy(x).to(device))
            logits, wdl = out[0], out[1]
            probs = _t.softmax(logits, dim=1).cpu().numpy()
            if wdl.shape[1] == 1:
                vals = wdl.cpu().numpy().flatten().astype(np.float64)
            else:
                _pw = _t.softmax(wdl, dim=1).cpu().numpy()
                vals = (_pw[:, 0] - _pw[:, 2]).astype(np.float64)
        for i, s in enumerate(states):
            m = s.legal_mask()
            probs[i][~m] = 0.0
            tot = probs[i].sum()
            probs[i] = probs[i] / tot if tot > 0 else m.astype(
                np.float32) / m.sum()
        return probs.astype(np.float64), np.asarray(vals, dtype=np.float64)

    return evaluate_fn, model, old_planes


def fixed_positions(n):
    """50 FIXED positions: startpos + seeded random-ply positions (seed
    20260925). Deterministic: same list every run, no RNG at test time."""
    import random as _r
    from chess_zero.game import State
    rng = _r.Random(SEED_BASE)
    out = [State.initial()]
    guard = 0
    while len(out) < n and guard < n * 60:
        guard += 1
        s = State.initial()
        plies = rng.randint(0, 24)
        ok = True
        for _ in range(plies):
            legal = s.legal_moves()
            if not legal:
                ok = False
                break
            s = s.apply(rng.choice(legal))
            done, _ = s.is_terminal()
            if done:
                ok = False
                break
        if ok and not s.is_terminal()[0] and s.legal_moves():
            out.append(s)
    assert len(out) == n, (len(out), n)
    return out


def run_one(backend_cls, bridge, state, evaluate_fn, seed, n_sims, kw,
            tt=None):
    import random as _r
    from chess_zero.mcts_bridge import MpsEvalBridge
    np.random.seed(seed)
    _r.seed(seed + 999)
    b = backend_cls() if bridge is None else backend_cls(bridge=bridge)
    if bridge is not None:
        bridge.inner = evaluate_fn
        bridge.reset_stats()
    k2 = dict(kw)
    if tt is not None:
        k2["tt"] = tt
    pi, stats = b.search(state, evaluate_fn, int(n_sims), **k2)
    visits = np.asarray(pi, dtype=np.float64) * float(
        stats.get("root_visits", 0))
    return pi, visits, float(stats.get("root_q", 0.0)), stats


def _report(title, max_dpi, mean_dpi, flips, max_dq, mean_dq, n):
    print(f"[{title}] n={n} max|dPi|={max_dpi:.3e} "
          f"mean|dPi|={mean_dpi:.3e} argmax_flips={flips} "
          f"max|dQ|={max_dq:.3e} mean|dQ|={mean_dq:.3e}", flush=True)


def test_exact_parity():
    """TT off + noise 0: bit-exact visits/policy/Q over NPOS positions."""
    import chess_zero.game as _g
    from chess_zero.mcts_bridge import (MpsEvalBridge, PythonTreeBackend,
                                        RustTreeBackend)
    dev = _device()
    ev, _model, old_planes = load_shared_evaluate(dev)
    try:
        positions = fixed_positions(NPOS)
        bridge = MpsEvalBridge(ev)
        max_dpi, sum_dpi, flips = 0.0, 0.0, 0
        max_dq, sum_dq = 0.0, 0.0
        for i, s in enumerate(positions):
            seed = RUN_SEED + i
            pi_a, v_a, q_a, _ = run_one(PythonTreeBackend, None, s, ev,
                                       seed, PARITY_SIMS, EXACT_KW)
            pi_b, v_b, q_b, _ = run_one(RustTreeBackend, bridge, s, ev,
                                       seed, PARITY_SIMS, EXACT_KW)
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
        _report("exact", max_dpi, sum_dpi / NPOS, flips, max_dq,
                sum_dq / NPOS, NPOS)
    finally:
        _g.INPUT_PLANES = old_planes


def test_noisy_divergence_stats():
    """TT on + noise 0.25, same seeds: report divergence stats (loose
    bound only -- this mode measures, it does not gate)."""
    import chess_zero.game as _g
    from chess_zero.mcts_bridge import (MpsEvalBridge, PythonTreeBackend,
                                        RustTreeBackend)
    dev = _device()
    ev, _model, old_planes = load_shared_evaluate(dev)
    try:
        positions = fixed_positions(NPOS)
        bridge = MpsEvalBridge(ev)
        max_dpi, sum_dpi, flips = 0.0, 0.0, 0
        max_dq, sum_dq = 0.0, 0.0
        for i, s in enumerate(positions):
            seed = RUN_SEED + i
            pi_a, _v_a, q_a, _ = run_one(PythonTreeBackend, None, s, ev,
                                        seed, PARITY_SIMS, NOISY_KW, tt={})
            pi_b, _v_b, q_b, _ = run_one(RustTreeBackend, bridge, s, ev,
                                        seed, PARITY_SIMS, NOISY_KW, tt={})
            dpi = float(np.abs(pi_a - pi_b).max())
            dq = abs(float(q_a) - float(q_b))
            max_dpi = max(max_dpi, dpi)
            sum_dpi += dpi
            max_dq = max(max_dq, dq)
            sum_dq += dq
            if int(pi_a.argmax()) != int(pi_b.argmax()):
                flips += 1
        _report("noisy", max_dpi, sum_dpi / NPOS, flips, max_dq,
                sum_dq / NPOS, NPOS)
        assert max_dpi < 0.02, ("noisy divergence blew past 0.02 -- "
                                f"bridge reorders evals? max={max_dpi}")
    finally:
        _g.INPUT_PLANES = old_planes


def test_timing_400sim():
    """400-sim search wall-clock per backend on 2 positions + bridge
    crossing counts. No speedup assert (stub is Python; C brings speed)."""
    import time
    import chess_zero.game as _g
    from chess_zero.mcts_bridge import (MpsEvalBridge, PythonTreeBackend,
                                        RustTreeBackend, timed_search)
    if os.environ.get("V21_SKIP_TIMING", "0") == "1":
        print("[timing] skipped via V21_SKIP_TIMING=1", flush=True)
        return
    dev = _device()
    ev, _model, old_planes = load_shared_evaluate(dev)
    try:
        positions = fixed_positions(12)[5:7]  # two fixed middlegame spots
        rows = []
        for j, s in enumerate(positions):
            for cls in (PythonTreeBackend, RustTreeBackend):
                bridge = MpsEvalBridge(ev) if cls is RustTreeBackend else None
                b = cls() if bridge is None else cls(bridge=bridge)
                np.random.seed(777 + j)
                import random as _r
                _r.seed(888 + j)
                t0 = time.time()
                pi, stats = b.search(s, ev, TIMING_SIMS, **EXACT_KW)
                dt = time.time() - t0
                assert abs(float(pi.sum()) - 1.0) < 1e-5
                assert int(pi.argmax()) in s.legal_moves()
                bc = bridge.stats_dict() if bridge is not None else {}
                rows.append((cls().name if bridge is None else b.name,
                             j, dt, int(stats.get("root_visits", 0)),
                             int(stats.get("extensions", 0)),
                             bc.get("crossings", "-"), bc.get("rows", "-")))
        print(f"[timing] {TIMING_SIMS}-sim search, device={dev}, "
              f"box=live-loaded:", flush=True)
        for name, j, dt, rv, ext, cr, rw in rows:
            eps = rv / dt if dt > 0 else 0.0
            print(f"  pos{j} {name:16s} {dt:7.2f}s "
                  f"visits={rv} ext={ext} evals/s={eps:6.1f} "
                  f"crossings={cr} rows={rw}", flush=True)
        py = [r for r in rows if "python" in r[0]]
        ru = [r for r in rows if "rust" in r[0]]
        for (pn, pj, pdt, *_), (rn, rj, rdt, *_) in zip(py, ru):
            print(f"  pos{pj} ratio rust-stub/python = {rdt / pdt:.3f}",
                  flush=True)
    finally:
        _g.INPUT_PLANES = old_planes


def test_no_live_path_coupling():
    """Guardrails: bridge+harness never integrate the loop, flip defaults,
    or write checkpoints/history."""
    import inspect
    import chess_zero.mcts_bridge as _b
    src = inspect.getsource(_b)
    for mod in ("loop", "fullrun", "train", "parallel"):
        assert f"from .{mod} import" not in src, mod
        assert f"from chess_zero.{mod} import" not in src, mod
    assert "V20_CONFIG" not in src  # read-only live config, untouched
    assert "checkpoints_mac/best.pt" not in src  # harness owns the path
    assert "torch.save" not in src and "history.json" not in src
    assert os.path.exists(BEST)  # read-only fixture present
    import chess_zero.config as _c
    assert _c.V20_CONFIG.sims == 400 and _c.V20_CONFIG.v20 is True
    from chess_zero import mcts as _m
    assert _m.search.__defaults__ is not None  # live signature intact


if __name__ == "__main__":
    test_no_live_path_coupling()
    print("ok no_live_path_coupling")
    test_exact_parity()
    print("ok exact_parity")
    test_noisy_divergence_stats()
    print("ok noisy_divergence_stats")
    test_timing_400sim()
    print("ok timing_400sim")
    print("all mcts-parity tests passed")
