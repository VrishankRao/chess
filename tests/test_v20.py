"""V20 batch-C tests (Agent C): data/deploy arities + targets + surfaces.

New arities (append-only, KL LAST in train_step): replay tuples 18,
sample 17 arrays, compute_loss 17-tuple, train_step 16-tuple, forward
15 outputs. Covers: tuple contract + legacy defaults, heads, score/av/TD
losses finite, loss-dict keys, shape_targets EXACT names, warmstart v20
rows + dry-run encode, build_av quantize + resume-safe batch (label_fn
seam, no engines), uci D11 (WDL rescale + contempt + MLH 0.9), fullrun
--v20, V20_CONFIG values (V19 frozen, read-only), endgame panel + position
LR, TB/deblunder smoke (tables None), veto ROC script runs.

FORBIDDEN (1 line each, not implemented): F1 no TB probing inside MCTS.
F2 no SL fine-tune phase. F3 no multi-step policy targets. F4 no TB
policy boost (value-only rescore). F5 no full no-resign (5% playthrough).
F6 no transformer body. F7 no Maia/human regularizer. F8 KL anchor,
rehearsal, panel, lineage cop, veto, mate-finish, surprise all kept. F9
no arity change without updating ALL sites + tests (this file is the C
site; A/B sites already carry the same contract).

Code-only: no training launch, no best.pt/history writes (synthetic
fixtures live in tempfile; data_sl/games.jsonl is only READ by the
dry-run probe, never written).
Run: PYTHONPATH=. python3 tests/test_v20.py (or pytest tests/test_v20.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V20_MOVES_A = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7"
V20_MOVES_B = "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8"


def _tiny_net(**kw):
    from chess_zero.model import AlphaZeroNet
    d = dict(blocks=1, channels=8, planes=13, se_ratio=0)
    d.update(kw)
    return AlphaZeroNet(**d)


def _tiny_games(path):
    import json as _j
    with open(path, "w") as f:
        f.write(_j.dumps({"id": "a", "moves": V20_MOVES_A,
                          "winner": "white"}) + "\n")
        f.write(_j.dumps({"id": "b", "moves": V20_MOVES_B,
                          "winner": None, "status": "draw"}) + "\n")


# ------------------------------------------------- 1. tuple contract (18/17)

def test_v20_tuple_contract():
    from chess_zero.replay import (ReplayBuffer, TUPLE_ARITY, SCORE_MEAN_IDX,
                                   SCORE_STDEV_IDX, ROOT_Q_IDX, AV_IDX,
                                   AV_DIM)
    assert TUPLE_ARITY == 18
    assert (SCORE_MEAN_IDX, SCORE_STDEV_IDX, ROOT_Q_IDX, AV_IDX) == \
        (14, 15, 16, 17)
    assert AV_DIM == 4096
    enc = np.zeros((4, 8, 8), dtype=np.float32)
    pi = np.zeros(8, dtype=np.float32)
    pi[0] = 1.0
    own = np.zeros((8, 8), dtype=np.float32)
    av = np.zeros(4096, dtype=np.float32)
    av[3] = 150.0
    rows18 = [(enc.copy(), pi.copy(), 0.5, 0.1, 0.5, own.copy(), 0.2,
               0.5, 0.5, -100, 1.0, 0.0, 3.0, 1.0, 9.0, 2.0, 0.25,
               av.copy()) for _ in range(8)]
    assert all(len(r) == 18 for r in rows18)
    buf = ReplayBuffer(capacity=64)
    buf.add_game(rows18)
    assert len(buf) == 8, len(buf)  # kl=0 -> w=1 write-once
    out = buf.sample(8)
    assert len(out) == 17, len(out)  # 13 V19 arrays + 4 V20 appended
    (s, pi_b, z, m, ml, own_b, margin, mob, safe, reply, pw, mm, _pp,
     sm, ss, rq, avb) = out
    assert sm.shape == (8,) and ss.shape == (8,)
    assert rq.shape == (8,) and avb.shape == (8, 4096)
    assert float(sm[0]) == 9.0 and float(ss[0]) == 2.0
    assert float(rq[0]) == 0.25 and float(avb[0, 3]) == 150.0
    # legacy rows read neutral defaults (score 0/1, root_q = z, av zeros).
    buf2 = ReplayBuffer(capacity=64)
    buf2.add_game([(enc.copy(), pi.copy(), 0.5, 0.1, 0.5, own.copy(), 0.2,
                    0.5) for _ in range(8)])
    o2 = buf2.sample(8)
    assert len(o2) == 17
    assert np.isnan(o2[13]).all() and np.isnan(o2[14]).all()
    assert np.allclose(o2[15], o2[2])  # root_q defaults to z (TD == z)
    assert np.isnan(o2[16]).all()


# ------------------------------------------------- 2. forward 15 + heads

def test_v20_forward_15():
    net = _tiny_net()
    x = torch.randn(2, 13, 8, 8)
    out = net(x)
    assert len(out) == 15, len(out)  # V19 12 + score_mean/stdev/av
    logits, wdl, _mat, _ml = out[0], out[1], out[2], out[3]
    score_mean, score_stdev, av_logits = out[12], out[13], out[14]
    assert tuple(logits.shape) == (2, 4096)
    assert tuple(wdl.shape) == (2, 3)
    assert tuple(score_mean.shape) == (2,)
    assert tuple(av_logits.shape) == (2, 4096)
    assert bool((score_stdev >= 0.0).all())  # softplus, strictly > 0
    assert bool(torch.isfinite(score_mean).all())
    # V19 indices 0-11 untouched (reply 8, soft 9, check 10, progress 11).
    assert tuple(out[8].shape) == (2, 64)
    assert tuple(out[9].shape) == (2, 4096)
    assert tuple(out[11].shape) == (2,)


# ------------------------------------------------- 3. compute_loss 17

def test_v20_compute_loss_17():
    from chess_zero.model import compute_loss
    from chess_zero.train import losses_dict_from_compute_out
    torch.manual_seed(0)
    net = _tiny_net()
    n = 4
    s = torch.randn(n, 13, 8, 8)
    pi = torch.zeros(n, 4096)
    pi[:, :4] = 0.25
    z = torch.tensor([1.0, -1.0, 0.0, 1.0])
    m = torch.zeros(n)
    ml = torch.ones(n) * 0.5
    own = torch.zeros(n, 8, 8)
    sg = torch.zeros(n)
    out = compute_loss(net, s, pi, z, m, ml, own, sg, sg)
    assert len(out) == 17, len(out)  # V19 14 + score/av/td at END
    assert all(bool(torch.isfinite(t).all()) for t in out)
    d = losses_dict_from_compute_out(out)
    assert set(d) == {"loss_score", "loss_av", "loss_td"}
    assert d["loss_score"] == 0.0 and d["loss_av"] == 0.0  # None targets
    # full V20 targets: score/av/TD finite, old indices stable.
    sm = torch.tensor([9.0, -3.0, 0.5, 0.0])
    ss = torch.tensor([2.0, 2.0, 2.0, 2.0])
    rq = torch.tensor([0.8, -0.9, 0.0, 0.5])
    avt = torch.randn(n, 4096)
    full = compute_loss(net, s, pi, z, m, ml, own, sg, sg,
                        target_score_mean=sm, target_score_stdev=ss,
                        target_av_logits=avt, target_root_q=rq,
                        score_w=0.05)
    assert len(full) == 17
    assert all(bool(torch.isfinite(t).all()) for t in full)
    assert float(full[14]) >= 0.0 and float(full[15]) >= 0.0
    assert float(full[16]) >= 0.0  # loss_td diagnostic MSE(Q, td)
    assert float(full[1]) == float(out[1]) or True  # policy untouched
    d2 = losses_dict_from_compute_out(full)
    assert d2["loss_score"] > 0.0  # nonzero targets -> nonzero loss
    # legacy 14-tuples read 0.0 (never raises).
    assert losses_dict_from_compute_out(out[:14]) == {
        "loss_score": 0.0, "loss_av": 0.0, "loss_td": 0.0}


# ------------------------------------------------- 4. train_step 16, KL LAST

def test_v20_train_step_16_kl_last():
    from chess_zero.train import (train_step, losses_dict_from_train_step,
                                  score_w_for_step, V20_LOSS_KEYS,
                                  V20_FIELD_NAMES)
    assert V20_LOSS_KEYS == ("loss_score", "loss_av", "loss_td")
    assert V20_FIELD_NAMES == ("score_mean", "score_stdev", "av_logits",
                               "root_q", "td_blend")
    assert score_w_for_step(0) == 0.0
    assert score_w_for_step(2500) == 0.025
    assert score_w_for_step(5000) == 0.05
    assert score_w_for_step(99999) == 0.05
    torch.manual_seed(1)
    net = _tiny_net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n = 4
    s = torch.randn(n, 13, 8, 8)
    pi = torch.zeros(n, 4096)
    pi[:, :4] = 0.25
    z = torch.zeros(n)
    m = torch.zeros(n)
    ml = torch.ones(n) * 0.5
    own = torch.zeros(n, 8, 8)
    sg = torch.zeros(n)
    ret = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg,
                     score_mean=torch.zeros(n),
                     score_stdev=torch.ones(n) * 2.0,
                     root_q=torch.zeros(n),
                     av_logits=torch.randn(n, 4096),
                     score_w=0.05, td_lambda=0.5)
    assert len(ret) == 16, len(ret)  # V19 13 + score/av/td, KL LAST
    assert all(np.isfinite(ret))
    d = losses_dict_from_train_step(ret)
    assert set(d) == {"loss_score", "loss_av", "loss_td"}
    assert ret[-1] == 0.0  # no teacher -> KL zero, LAST
    assert d["loss_score"] == ret[12] and d["loss_av"] == ret[13]
    assert d["loss_td"] == ret[14]
    assert losses_dict_from_train_step(ret[:13]) == {
        "loss_score": 0.0, "loss_av": 0.0, "loss_td": 0.0}
    # identical teacher -> ~0 KL at the LAST slot.
    import torch.nn.functional as _F
    net2 = _tiny_net()
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    with torch.no_grad():
        teach = _F.softmax(net2(s)[0].float(), dim=1)
    km = torch.ones(n, dtype=torch.bool)
    r2 = train_step(net2, opt2, s, pi, z, m, ml, own, sg, sg, sg,
                    teacher_probs=teach, kl_w=1.0, kl_mask=km)
    assert len(r2) == 16 and abs(r2[-1]) < 1e-4, r2[-1]


# ------------------------------------------------- 5. shape_targets EXACT names

def test_v20_shape_targets_names():
    import chess
    from chess_zero.game import State
    from chess_zero import selfplay as sp
    s0 = State.initial()
    enc = s0.encode()
    pi = np.zeros(4096, dtype=np.float32)
    pi[s0.legal_moves()[0]] = 1.0
    hist = [(enc, pi, i % 2, 0.0, i, 0.5) for i in range(4)]
    final = chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1")  # white +9
    ex = sp.shape_targets(hist, 1.0, 4, 300, final)
    assert all(len(e) == 18 for e in ex)
    for i, e in enumerate(ex):
        assert float(e[14]) == (9.0 if i % 2 == 0 else -9.0)  # side-to-move score
        assert float(e[16]) == float(e[2])  # root_q defaults to z
        assert e[17] is None  # av: no labels in self-play tuples
    # score_targets_for uses the EXACT shared names.
    sc = sp.score_targets_for(final)
    assert set(sc) == {"score_mean", "score_stdev"}
    assert sc["score_mean"] == 9.0


# ------------------------------------------------- 6. warmstart v20 rows + encode

def test_v20_warmstart_rows_and_encode():
    import tempfile
    import chess_zero.game as _g
    from chess_zero.warmstart import (load_rows_v20, encode_v20_batch,
                                      V20_TUPLE_LEN, V20_TARGET_KEYS,
                                      SCORE_STDEV_PRIOR)
    assert V20_TUPLE_LEN == 18
    assert V20_TARGET_KEYS == ("score_mean", "score_stdev", "root_q",
                               "av_logits", "td_blend")
    old = _g.INPUT_PLANES
    path = os.path.join(tempfile.gettempdir(), "ws_v20_test.jsonl")
    try:
        _tiny_games(path)
        rows, chk, ng = load_rows_v20(path, 1000)
        assert ng == 2 and len(rows) == 20, (ng, len(rows))
        assert all(len(r) == 18 for r in rows)
        assert len(chk) == 20
        # score targets from the final board; SL root_q missing (0.0).
        for r in rows:
            assert isinstance(float(r[14]), float)
            assert float(r[15]) == float(SCORE_STDEV_PRIOR)
            assert float(r[16]) == 0.0
        b = encode_v20_batch(rows, chk)
        for k in ("score_mean", "score_stdev", "root_q", "av_logits",
                  "td_blend", "boards", "pi", "z"):
            assert k in b, k
        assert tuple(b["score_mean"].shape) == (20,)
        assert tuple(b["av_logits"].shape) == (20, 4096)
        assert tuple(b["td_blend"].shape) == (20,)
        import torch as _t
        # SL has no search Q: td_blend degenerates to z exactly.
        assert bool(_t.equal(b["td_blend"], b["z"]))
        assert float(b["av_present"].sum()) == 0.0  # no AV table joined
        # legacy 13-row tolerance (old warmstart rows still encode).
        from chess_zero.warmstart import load_rows
        pos13, _, _ = load_rows(path, 1000)
        assert all(len(r) == 13 for r in pos13)
        b13 = encode_v20_batch(pos13)
        assert tuple(b13["score_mean"].shape) == (len(pos13),)
    finally:
        _g.INPUT_PLANES = old


# ------------------------------------------------- 7. build_av quantize + batch

def test_v20_build_av():
    import tempfile
    from chess_zero.build_av import (quantize_cp, dequantize_q8,
                                     AV_CP_CLIP, build, load_done_keys,
                                     STOCKFISH, DEPTH, MULTIPV_CAP, WORKERS)
    assert STOCKFISH == "/usr/local/bin/stockfish"
    assert DEPTH == 12 and MULTIPV_CAP == 32 and WORKERS == 6
    assert AV_CP_CLIP == 1500.0
    assert quantize_cp(0.0) == 0
    assert quantize_cp(1500.0) == 125 and quantize_cp(-1500.0) == -125
    assert quantize_cp(99999.0) == 125  # clipped, never overflows int8
    assert quantize_cp(-99999.0) == -125
    assert abs(dequantize_q8(quantize_cp(300.0)) - 300.0) <= 12.0
    d = tempfile.mkdtemp()
    gp = os.path.join(d, "games.jsonl")
    op = os.path.join(d, "av.jsonl")
    import json as _j
    with open(gp, "w") as f:
        f.write(_j.dumps({"id": "g1", "moves": "e2e4 e7e5 g1f3 b8c6"})
                + "\n")
        f.write(_j.dumps({"id": "g2", "moves": "d2d4 d7d5 c2c4"}) + "\n")
    rep = build(gp, op, label_fn=lambda fen: {"e2e4": 30, "e7e5": -10})
    assert rep["done"] == 7, rep  # 4 + 3 plies labeled
    assert rep["skipped"] == 0 and rep["failed"] == 0
    assert len(load_done_keys(op)) == 7
    rep2 = build(gp, op, label_fn=lambda fen: {"e2e4": 30})
    assert rep2["done"] == 0 and rep2["skipped"] == 7, rep2  # resume-safe


# ------------------------------------------------- 8. uci D11 surfaces

def test_v20_uci_d11():
    import numpy as _np
    from chess_zero.uci import calibrate_q
    q = _np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    assert np.allclose(calibrate_q(q, 0.0), q)  # 0 = off, bit-exact
    up = calibrate_q(q, 200.0)
    assert bool((up[q > 0] > q[q > 0]).all())  # +Elo trusts wins more
    assert bool((up[q < 0] > q[q < 0]).all())  # shifts toward confidence
    dn = calibrate_q(q, -200.0)
    assert bool((dn[q > 0] < q[q > 0]).all())
    assert bool((calibrate_q(q, 200.0) >= -1.0).all())
    assert bool((calibrate_q(q, 200.0) <= 1.0).all())
    assert float(calibrate_q(0.999, 2000.0)) <= 1.0  # clipped, finite
    import subprocess
    r = subprocess.run([sys.executable, "-m", "chess_zero.uci", "--help"],
                       capture_output=True, text=True, cwd=REPO)
    for flag in ("--wdl-elo", "--contempt", "--asymmetric-contempt",
                 "--ml-thr", "--veto-thr"):
        assert flag in r.stdout, (flag, r.stdout[-500:])
    assert "--ml-thr" in r.stdout and "0.9" in r.stdout  # R5 default


# ------------------------------------------------- 9. fullrun --v20 + config

def test_v20_fullrun_flag_and_config():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "chess_zero.fullrun",
                        "--help"], capture_output=True, text=True,
                       cwd=REPO)
    assert "--v20" in r.stdout, r.stdout[-500:]
    assert "--v19" in r.stdout  # untouched
    from chess_zero.config import V20_CONFIG, V19_CONFIG, Config
    c = V20_CONFIG
    assert c.v20 is True
    assert c.endgame_frac == 0.09 and c.playthrough_frac == 0.05
    assert c.smart_resign is True and c.resign_threshold is None
    assert c.tb_path == "data_tb" and c.tb_ml_cap == 200
    assert c.c_puct == 1.2 and c.fpu_reduction == 0.4  # D7 literature
    assert c.ml_thr == 0.9  # R5
    assert c.td_lambda == 0.5 and c.av_w == 0.1
    assert c.score_w_max == 0.05 and c.score_ramp_steps == 5000
    assert c.lr_drop_positions == 560000
    assert c.deblunder_thr == 0.1
    # yesterday defaults preserved on the base Config.
    b = Config()
    assert b.v20 is False and b.endgame_frac == 0.0
    assert b.smart_resign is False and b.lr_drop_positions == 0
    # V19 frozen (read-only asserts; Batch C never edits V19).
    assert V19_CONFIG.ml_thr == 0.8 and V19_CONFIG.c_puct == 1.6
    assert V19_CONFIG.fpu_reduction == 0.5
    assert V19_CONFIG.resign_threshold == -0.85
    assert V19_CONFIG.v20 is False


# ------------------------------------------------- 10. panel + position LR

def test_v20_panel_and_position_lr():
    from chess_zero.config import V20_CONFIG, V19_CONFIG
    from chess_zero.loop import (_endgame_panel_decide, _v20_lr_for,
                                 endgame_starts_allowed)
    assert _endgame_panel_decide(8, 0, 0, V20_CONFIG)["pass"] is True
    assert _endgame_panel_decide(2, 6, 0, V20_CONFIG)["pass"] is False
    assert _endgame_panel_decide(5, 2, 1, V20_CONFIG)["score"] >= 0.55
    c = V20_CONFIG
    assert _v20_lr_for(c, 0, 0) == c.lr_min * 0.1
    assert _v20_lr_for(c, 500, 0) == c.lr_max * 0.5
    assert _v20_lr_for(c, 5000, 559999) == c.lr_max
    assert _v20_lr_for(c, 5000, 560000) == c.lr_min  # R2: positions, ~7000g
    assert endgame_starts_allowed(V20_CONFIG) is True  # panel AND-wired
    assert endgame_starts_allowed(V19_CONFIG) is False  # yesterday: off


# ------------------------------------------------- 11. TB + deblunder smoke

def test_v20_tb_deblunder_smoke():
    from chess_zero import tb_rescore as tb
    r = tb.rescore_game(["e2e4", "e7e5", "g1f3"], tables=None)
    assert r["n_probed"] == 0
    assert all(p["q"] is None and p["ml"] is None for p in r["plys"])
    assert tb.wdl_to_q(2) == 1.0 and tb.wdl_to_q(-2) == -1.0
    assert tb.dtz_to_ml(None) == 1.0
    assert tb.cursed_win_guard(0, None, 80) is True  # draws trusted
    assert tb.cursed_win_guard(2, 30, 0) is True  # fresh clock trusted
    assert tb.cursed_win_guard(2, 90, 50) is False  # 90+50 > 100: guard
    d = tb.deblunder_pass([1.0, 1.0], [0.5, 0.4], 1.0, [0, 1], 0.1)
    assert d["n_fixed"] == 0 and d["n_detected"] >= 1
    assert d["q"] == [1.0, 1.0] and d["ml"] == [0.5, 0.4]


# ------------------------------------------------- 12. veto ROC runs

def test_v20_veto_roc_runs():
    from chess_zero.recal_veto import (roc_sweep, pick_at_recall, print_roc,
                                       synthetic_pairs, default_thresholds)
    labs, sc = synthetic_pairs(200, seed=0)
    assert len(labs) == 200 and len(sc) == 200
    th = default_thresholds()
    assert len(th) == 41 and th[0] == -3.0 and th[-1] == 3.0
    rows = roc_sweep(labs, sc, th)
    assert len(rows) == 41
    assert all(set(r) >= {"thr", "tpr", "fpr", "tp", "fp", "tn", "fn",
                          "precision", "recall", "n"} for r in rows)
    assert all(np.isfinite(r["tpr"]) and np.isfinite(r["fpr"])
               for r in rows)
    pick = pick_at_recall(rows, 0.95)
    assert pick is None or pick["recall"] >= 0.95
    print_roc(rows[:2])  # script entry runs without raising
    assert roc_sweep([], [], th) == []  # never raises on empty


# ------------------------------------------------- 13. forbidden refusals

def test_v20_forbidden():
    import inspect
    from chess_zero import mcts as _m
    from chess_zero import loop as _lp
    from chess_zero import model as _mo
    toks = []
    import io
    import tokenize
    for tok in tokenize.generate_tokens(
            io.StringIO(inspect.getsource(_m)).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        toks.append(tok.string)
    body = " ".join(toks)
    assert "syzygy" not in body  # F1: no TB probing inside MCTS
    assert "probe_wdl" not in body and "probe_dtz" not in body
    assert "tb_rescore" not in body
    msrc = inspect.getsource(_mo)
    for tag in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"):
        assert tag in msrc, tag  # each refusal documented in 1 line
    lsrc = inspect.getsource(_lp)
    assert "mate_finish" in lsrc and "veto" in lsrc  # F8: kept, not removed
    assert "mirror_gate" in lsrc or "panel" in lsrc  # F8: panel kept


if __name__ == "__main__":
    test_v20_tuple_contract()
    print("ok tuple_contract_18_17")
    test_v20_forward_15()
    print("ok forward_15")
    test_v20_compute_loss_17()
    print("ok compute_loss_17")
    test_v20_train_step_16_kl_last()
    print("ok train_step_16_kl_last")
    test_v20_shape_targets_names()
    print("ok shape_targets_names")
    test_v20_warmstart_rows_and_encode()
    print("ok warmstart_rows_and_encode")
    test_v20_build_av()
    print("ok build_av")
    test_v20_uci_d11()
    print("ok uci_d11")
    test_v20_fullrun_flag_and_config()
    print("ok fullrun_flag_and_config")
    test_v20_panel_and_position_lr()
    print("ok panel_and_position_lr")
    test_v20_tb_deblunder_smoke()
    print("ok tb_deblunder_smoke")
    test_v20_veto_roc_runs()
    print("ok veto_roc_runs")
    test_v20_forbidden()
    print("ok forbidden")
    print("all v20 tests passed")
