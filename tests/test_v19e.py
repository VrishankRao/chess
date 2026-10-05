"""V19e tests (Impl-A second sweep): D1 conv policy head, D2 forward_inf,
D4 progress head, D5 ml_mask consume, D24 code defaults, + A-item regression.

Scope: chess_zero/model.py, chess_zero/train.py ONLY (warmstart.py audited
unchanged: sample_rehearsal(path,k,seed) + encode_rehearsal_batch(records)
signatures exact, soft/check targets + smooth 0.05 present).

Run: PYTHONPATH=. python3 tests/test_v19e.py  (or pytest tests/test_v19e.py).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def _tiny_net(**kw):
    from chess_zero.model import AlphaZeroNet
    d = dict(blocks=1, channels=8, planes=13, se_ratio=0)
    d.update(kw)
    return AlphaZeroNet(**d)


def _batch(n=4, planes=13):
    torch.manual_seed(0)
    s = torch.randn(n, planes, 8, 8)
    pi = torch.zeros(n, 4096)
    pi[:, :8] = 1.0 / 8.0
    z = torch.zeros(n)
    m = torch.zeros(n)
    ml = torch.ones(n) * 0.5
    own = torch.zeros(n, 8, 8)
    sg = torch.zeros(n)
    return s, pi, z, m, ml, own, sg


# ------------------------------------------------------- D1 conv policy head

def test_conv_head_shapes_and_fewer_params():
    net = _tiny_net()
    # p_fc is now a Conv1x1 (not a flat Linear); stem names kept.
    assert isinstance(net.p_fc, nn.Conv2d), type(net.p_fc)
    assert isinstance(net.p_conv, nn.Conv2d) and isinstance(net.p_bn, nn.BatchNorm2d)
    assert isinstance(net.p_conv2, nn.Conv2d) and isinstance(net.p_bn2, nn.BatchNorm2d)
    assert tuple(net.p_fc.weight.shape) == (64, 8, 1, 1), net.p_fc.weight.shape
    x = torch.randn(2, 13, 8, 8)
    out = net(x)
    assert len(out) == 15, len(out)  # V20: 12 + score_mean/stdev/av LAST
    assert tuple(out[0].shape) == (2, 4096), out[0].shape
    # codec: logit[from*64+to] == out[from, to_row, to_col].
    flat = out[0][0]
    grid = out[0][0].reshape(64, 8, 8)
    for fr, to in ((0, 0), (7, 63), (63, 0), (28, 35)):
        assert float(flat[fr * 64 + to]) == float(grid[fr, to // 8, to % 8])
    # FEWER params than the flat FC it replaces (old: 512*4096+4096 FC
    # + 8ch 1x1 stem; new: same stem + 3x3 + 1x1 convs).
    new_n = sum(p.numel() for m in (net.p_conv, net.p_bn, net.p_conv2,
                                    net.p_bn2, net.p_fc) for p in m.parameters())
    old_n = 8 * 64 + (8 * 64) * 4096 + 4096  # stem + flat FC (ch=8 toy)
    assert new_n < old_n, (new_n, old_n)
    print(f"  conv head params {new_n} < flat FC {old_n}", flush=True)
    # soft head stays FC (train-only), A-spec shape untouched.
    assert isinstance(net.p_soft_fc, nn.Linear)
    assert tuple(net.p_soft_fc.weight.shape) == (4096, 8 * 64)


def test_conv_head_trains():
    from chess_zero.train import train_step
    torch.manual_seed(1)
    net = _tiny_net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    s, pi, z, m, ml, own, sg = _batch()
    t0, *_rest = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg)
    for _ in range(6):
        t1, *_rest = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg)
    assert t1 < t0, (t0, t1)


def test_surgery_p_fc_mismatch_zeros():
    from chess_zero.model import grow_state_dict, load_weights
    net = _tiny_net()
    own = {k: v for k, v in net.state_dict().items()}
    old = dict(own)
    # old Linear-era p_fc shapes (flat FC), everything else matching.
    old["p_fc.weight"] = torch.randn(4096, 8 * 64)
    old["p_fc.bias"] = torch.randn(4096)
    grown = grow_state_dict({k: v for k, v in own.items()}, old)
    assert tuple(grown["p_fc.weight"].shape) == (64, 8, 1, 1)
    assert bool((grown["p_fc.weight"] == 0).all())
    assert tuple(grown["p_fc.bias"].shape) == (64,)
    assert bool((grown["p_fc.bias"] == 0).all())
    # full load path (strict=False) must not crash on old checkpoints.
    load_weights(net, {"weights": old}, strict=False)
    x = torch.randn(1, 13, 8, 8)
    net.eval()
    with torch.no_grad():
        assert tuple(net(x)[0].shape) == (1, 4096)


# ---------------------------------------------------------------- D2 forward_inf

def test_forward_inf_parity_and_timing():
    net = _tiny_net()
    net.eval()
    x = torch.randn(4, 13, 8, 8)
    with torch.no_grad():
        full = net(x)
        inf = net.forward_inf(x)
    assert len(inf) == 4, len(inf)
    names = ("logits", "wdl", "safety", "moves_left")
    for i, (name, fidx) in enumerate((("logits", 0), ("wdl", 1),
                                     ("safety", 7), ("moves_left", 3))):
        a, b = full[fidx], inf[i]
        assert tuple(a.shape) == tuple(b.shape), (name, a.shape, b.shape)
        assert torch.equal(a, b), f"{name} not bitwise-equal"
    # timing report (informational; keep whichever is faster, default inf).
    with torch.no_grad():
        for _ in range(5):
            net(x)
            net.forward_inf(x)
        t0 = time.perf_counter()
        for _ in range(20):
            net(x)
        t_full = (time.perf_counter() - t0) / 20 * 1000
        t0 = time.perf_counter()
        for _ in range(20):
            net.forward_inf(x)
        t_inf = (time.perf_counter() - t0) / 20 * 1000
    print(f"  forward {t_full:.2f}ms vs forward_inf {t_inf:.2f}ms "
          f"(batch-4 CPU, toy net)", flush=True)


# ---------------------------------------------------------------- D4 progress

def test_progress_head_trains():
    from chess_zero.model import compute_loss
    from chess_zero.train import train_step
    torch.manual_seed(2)
    net = _tiny_net()
    assert tuple(net.pg_fc2.weight.shape) == (1, 32)  # FC32+FC1, linear out
    s, pi, z, m, ml, own, sg = _batch()
    prog = torch.rand(4) * 40.0  # raw plies target
    out = compute_loss(net, s, pi, z, m, ml, own, sg, sg, sg,
                       target_progress=prog)
    assert len(out) == 17, len(out)  # V20: 14 + score/av/td at END
    assert all(torch.isfinite(t).all() for t in out)
    assert float(out[-1]) >= 0.0
    # None targets -> graph-connected zero (reply precedent: the TOTAL
    # carries grad through the zero term; returns are detached by design).
    out0 = compute_loss(net, s, pi, z, m, ml, own, sg, sg, sg)
    assert float(out0[13]) == 0.0  # progress None -> graph-connected zero
    assert float(out0[14]) == 0.0 and float(out0[15]) == 0.0  # V20 score/av
    assert out0[0].grad_fn is not None
    # overfit: progress loss falls; train_step returns 16, KL last.
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    r0 = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg,
                    progress=prog)
    assert len(r0) == 16, len(r0)
    for _ in range(8):
        r1 = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg,
                        progress=prog)
    assert r1[11] < r0[11], (r0[11], r1[11])  # progress float at 11, KL last
    assert r1[-1] == 0.0  # no teacher -> KL zero


# ---------------------------------------------------------------- D5 ml_mask

def test_ml_mask_zeros_ml_term():
    from chess_zero.model import compute_loss
    torch.manual_seed(3)
    net = _tiny_net()
    net.eval()
    s, pi, z, m, ml, own, sg = _batch()
    ones = torch.ones(4)
    zeros = torch.zeros(4)
    a = compute_loss(net, s, pi, z, m, ml, own, sg, sg, sg, ml_mask=ones)
    b = compute_loss(net, s, pi, z, m, ml, own, sg, sg, sg, ml_mask=zeros)
    c = compute_loss(net, s, pi, z, m, ml, own, sg, sg, sg, ml_mask=None)
    assert len(a) == 17
    ml_w = 0.05
    # zeros mask kills exactly ml_w * ml_loss; None == all-ones.
    assert abs(float(b[4])) < 1e-9, float(b[4])
    assert abs((float(a[0]) - float(b[0])) - ml_w * float(a[4])) < 1e-5
    assert abs(float(c[0]) - float(a[0])) < 1e-9
    # partial mask stays finite.
    half = torch.tensor([1.0, 1.0, 0.0, 0.0])
    h = compute_loss(net, s, pi, z, m, ml, own, sg, sg, sg, ml_mask=half)
    assert torch.isfinite(h[0])


# ------------------------------------------------- A regression (brief) + D24

def test_a_regression_kl_augment_bn_defaults():
    from chess_zero.model import compute_loss
    from chess_zero.train import train_step, augment_batch
    import inspect
    torch.manual_seed(4)
    net = _tiny_net()
    s, pi, z, m, ml, own, sg = _batch()
    # D24 code defaults: soft 0.3, check 0.01, reply 0.05 (NOT 0.15 —
    # config sets 0.15), progress 0.05. value_w 1.0.
    for fn in (compute_loss, train_step):
        sig = inspect.signature(fn)
        assert sig.parameters["soft_w"].default == 0.3
        assert sig.parameters["check_w"].default == 0.01
        assert sig.parameters["reply_w"].default == 0.05
    assert inspect.signature(compute_loss).parameters["progress_w"].default == 0.05
    assert inspect.signature(compute_loss).parameters["value_w"].default == 1.0
    assert inspect.signature(train_step).parameters["kl_w"].default == 0.0
    # KL: identical teacher/student -> ~0 (tolerate trailing returns).
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.no_grad():
        teach = torch.softmax(net(s)[0].float(), dim=1)
    km = torch.ones(4, dtype=torch.bool)
    ret = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg,
                     teacher_probs=teach, kl_w=1.0, kl_mask=km)
    assert len(ret) == 16
    assert abs(ret[-1]) < 1e-4, ret[-1]
    # augment: flip twice == identity; castling planes swap on mirror.
    boards = torch.randn(2, 20, 8, 8)
    boards[:, 16] = 1.0
    boards[:, 17] = 2.0
    boards[:, 18] = 3.0
    boards[:, 19] = 4.0
    pis = torch.zeros(2, 4096)
    pis[:, 100] = 1.0
    owns = torch.randn(2, 8, 8)
    reply = torch.tensor([10, -100])
    b1, p1, o1, r1 = augment_batch(boards, pis, owns, reply)
    assert bool((b1[:, 16] == 1.0).all() and (b1[:, 17] == 2.0).all()) is False
    # after one mirror Ks<->Qs swapped (values 1<->2, 3<->4, then fliplr
    # keeps the uniform fills uniform so the swap is directly visible).
    assert bool((b1[:, 16] == 2.0).all() and (b1[:, 17] == 1.0).all())
    assert bool((b1[:, 18] == 4.0).all() and (b1[:, 19] == 3.0).all())
    b2, p2, o2, r2 = augment_batch(b1, p1, o1, r1)
    assert torch.equal(b2, boards) and torch.equal(p2, pis)
    assert torch.equal(o2, owns) and torch.equal(r2, reply)
    # BN momentum 0.02 on every BN.
    for mod in net.modules():
        if isinstance(mod, nn.BatchNorm2d):
            assert mod.momentum == 0.02, mod.momentum


def test_warmstart_signatures_exact():
    import inspect
    from chess_zero import warmstart as ws
    params = list(inspect.signature(ws.sample_rehearsal).parameters.values())
    assert [p.name for p in params] == ["games_path", "n_rows", "seed"], params
    assert params[2].default == 0  # sample_rehearsal(path, k, seed) positional
    params = list(inspect.signature(ws.encode_rehearsal_batch).parameters)
    assert params[0] == "records", params  # (records, ...) — called as (records)
    recs = ws.sample_rehearsal("/nonexistent.jsonl", 5, seed=1)
    assert recs == []  # never raises on missing file


if __name__ == "__main__":
    test_conv_head_shapes_and_fewer_params()
    print("ok conv_head_shapes_and_fewer_params")
    test_conv_head_trains()
    print("ok conv_head_trains")
    test_surgery_p_fc_mismatch_zeros()
    print("ok surgery_p_fc_mismatch_zeros")
    test_forward_inf_parity_and_timing()
    print("ok forward_inf_parity")
    test_progress_head_trains()
    print("ok progress_head_trains")
    test_ml_mask_zeros_ml_term()
    print("ok ml_mask_zeros_ml_term")
    test_a_regression_kl_augment_bn_defaults()
    print("ok a_regression_kl_augment_bn_defaults")
    test_warmstart_signatures_exact()
    print("ok warmstart_signatures_exact")
    print("all v19e tests passed")
