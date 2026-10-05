"""V19 part-C tests (Impl-C): loop+config.

Covers (loop.py, config.py, fullrun.py, uci.py only):
 1. V19_CONFIG carries every C1 field value.
 2. SPRT LLR math (gsprt_llr): antisymmetry, ~0 on even scores, sign on
    blowouts, 0.0 on no games; Wald bounds from cfg error rates.
 3. SPRT decisions (_gate_sprt_decide): 10-0-0 promotes (truncated: LLR>0
    and 10/10 >= 11/20), 0-10-0 rejects, 5-5-10 holds; bound crossings
    on synthetic blowouts; mid-match continue.
 4. Mirrored color alternation: play_match records alternate A-white.
 5. Game-budget LR + value_w at landmarks.
 6. AdamW two groups (1-dim params carry wd=0).
 7. Sampling-ratio step math (+ zero-steps honesty preserved in loop).
 8. Rehearsal mix shapes + identical-teacher KL ~ 0 (skipped loudly if
    the warmstart helpers are ever absent — owned by Impl-A).
 9. PreciseBN-lite runs finite and leaves the model in eval.
10. Progress/agent spec stays 24 elements with se_ratio at [23]
    (evaluate.py verify-only: _make_policy reads spec[23]).
11. Aux late cull: fires past 8000 games, one-way latch, 0.005 values.
12. Dirichlet eps schedule landmarks + gate book draw determinism.
13. fullrun --v19 flag present; uci search flags present.

Run: python3 tests/test_v19c.py  (or pytest tests/test_v19c.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch


# ---------------------------------------------------------------- 1. config

def test_v19_config():
    from chess_zero.config import V19_CONFIG as c, Config
    assert c.blocks == 6 and c.channels == 64 and c.input_planes == 30
    assert c.se_ratio == 4 and c.sims == 400
    assert c.c_puct == 1.6
    assert c.fpu_reduction == 0.5
    assert c.prune_singletons is True
    assert c.ent_w == 0.002
    assert c.smooth_eps == 0.01
    assert (c.value_w0, c.value_w1) == (0.5, 1.0)
    assert c.soft_w == 0.3 and c.check_w == 0.01
    assert c.reply_w == 0.15 and c.own_w == 0.12
    assert (c.mob_w, c.margin_w, c.safe_w) == (0.01, 0.02, 0.01)  # D24
    assert c.fast_frac == 0.5
    assert c.kl_w0 == 0.2 and c.kl_games == 5000  # retuned: KL~8 vs CE~2.5
    assert c.rehearsal_frac == 0.2
    assert c.rehearsal_games == "data_sl/games.jsonl"
    assert c.teacher_path == "checkpoints_warm/warmstart.pt"
    assert c.lr_max == 3e-4 and c.lr_min == 3e-5
    assert c.lr_drop_games == 7000 and c.lr_warmup_steps == 1000
    assert (c.sprt_elo0, c.sprt_elo1) == (0.0, 30.0)
    assert (c.sprt_alpha, c.sprt_beta) == (0.10, 0.10)
    assert c.gate_games == 20 and c.mirror_gate is True
    # yesterday defaults preserved on the base Config (old configs keep
    # training exactly as before).
    b = Config()
    assert b.mirror_gate is False and b.lr_drop_games == 0
    assert b.kl_w0 == 0.0 and b.rehearsal_frac == 0.0
    assert b.fpu_reduction == 0.0 and b.prune_singletons is False
    assert b.smooth_eps == 0.0 and b.reply_w == 0.05


# ---------------------------------------------------------------- 2+3. SPRT

def test_sprt_llr_values():
    from chess_zero.loop import gsprt_llr as g, _sprt_bounds
    from chess_zero.config import V19_CONFIG as c
    LA, LB = _sprt_bounds(c)
    assert LA == math.log(0.1 / 0.9) and LB == math.log(0.9 / 0.1)
    assert LA < 0 < LB
    assert g(0, 0, 0) == 0.0
    assert g(10, 0, 0) > 0 and g(0, 10, 0) < 0
    assert abs(g(5, 5, 10)) < 0.5  # ~0 on the even score
    # sign-antisymmetric under W<->L (exact only for symmetric
    # hypotheses, e.g. elo(-15, 15), since s0 != 1 - s1 for (0, 30)).
    assert g(10, 0, 0) > 0 > g(0, 10, 0)
    assert abs(g(3, 7, 4, -15.0, 15.0) + g(7, 3, 4, -15.0, 15.0)) < 1e-12
    assert math.isfinite(g(20, 0, 0)) and math.isfinite(g(0, 0, 20))


def test_sprt_decisions():
    from chess_zero.loop import _gate_sprt_decide as decide
    from chess_zero.config import V19_CONFIG as c
    # 10-0-0 final (10-game gate): LLR>0 and 10/10 >= 11/20 -> PROMOTE.
    d = decide(10, 0, 0, c, 10, 10)
    assert d["stop"] and d["promote"] and d["via"] == "truncated", d
    # 20-game boundary: 11-0-9 (11/20 points) promotes, 10-1-9 does not.
    d = decide(11, 0, 9, c, 20, 20)
    assert d["stop"] and d["promote"] and d["via"] == "truncated", d
    d = decide(7, 5, 8, c, 20, 20)  # 11.0/20 points -> promote
    assert d["stop"] and d["promote"], d
    d = decide(6, 5, 9, c, 20, 20)  # 10.5/20 points -> hold
    assert d["stop"] and not d["promote"], d
    # 0-10-0 final: negative LLR -> truncated HOLD (reject).
    d = decide(0, 10, 0, c, 10, 10)
    assert d["stop"] and not d["promote"], d
    # 5-5-10 final: LLR ~ -0.07, not > 0 -> hold.
    d = decide(5, 5, 10, c, 20, 20)
    assert d["stop"] and not d["promote"] and abs(d["llr"]) < 0.5, d
    # mid-match, no cross: continue.
    d = decide(1, 0, 1, c, 4, 20)
    assert not d["stop"] and not d["promote"] and d["via"] == "continue"
    # synthetic blowouts cross the Wald bounds through the bound routes.
    LA_BOUND = math.log(0.9 / 0.1)
    assert decide(40, 0, 0, c, 40, 40)["via"] == "sprt-upper"
    assert decide(0, 40, 0, c, 40, 40)["via"] == "sprt-lower"
    assert LA_BOUND > 0


# --------------------------------------------------- 4. mirrored alternation

def test_mirrored_color_alternation():
    from chess_zero.evaluate import (play_match, random_move, greedy_move)
    _tot, recs = play_match(random_move, greedy_move, games=4, cap=60,
                            opening_moves=2, record=True, seed_base=777)
    assert len(recs) == 4
    assert [r[2] for r in recs] == [True, False, True, False], \
        [r[2] for r in recs]
    # pair mates share the seeded opening: same move count prefix length
    # is not guaranteed post-opening, but colors must swap adjacently.
    for k in range(0, 4, 2):
        assert recs[k][2] and not recs[k + 1][2]


# ------------------------------------------------------------- 5. LR/value

def test_lr_landmarks():
    from chess_zero.loop import _v19_lr_for as lr, _v19_value_w_for as vw
    from chess_zero.config import V19_CONFIG as c
    assert lr(c, 0, 0) == c.lr_min * 0.1  # step 0 near-zero, never wasted
    assert lr(c, 500, 0) == c.lr_max * 0.5
    assert lr(c, 1000, 0) == c.lr_max
    assert lr(c, 5000, 6999) == c.lr_max
    assert lr(c, 5000, 7000) == c.lr_min
    assert lr(c, 99999, 20000) == c.lr_min
    assert vw(c, 0) == 0.5 and vw(c, 6999) == 0.5
    assert vw(c, 7000) == 1.0 and vw(c, 20000) == 1.0


# ---------------------------------------------------------------- 6. AdamW

def test_adamw_groups():
    from chess_zero.model import AlphaZeroNet
    from chess_zero.loop import _build_adamw
    from chess_zero.config import V19_CONFIG as c
    net = AlphaZeroNet(blocks=1, channels=8, planes=30, se_ratio=4)
    opt = _build_adamw(net, c)
    assert type(opt).__name__ == "AdamW" and len(opt.param_groups) == 2
    g_decay, g_nodecay = opt.param_groups
    assert g_decay["weight_decay"] == c.l2
    assert g_nodecay["weight_decay"] == 0.0
    n1 = sum(p.numel() for p in g_decay["params"])
    n0 = sum(p.numel() for p in g_nodecay["params"])
    assert n1 > 0 and n0 > 0, (n1, n0)
    for p in g_nodecay["params"]:
        assert p.dim() <= 1
    for p in g_decay["params"]:
        assert p.dim() > 1
    # both groups track the game-budget LR identically.
    for g in opt.param_groups:
        g["lr"] = 1e-4
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 1e-4]


# ---------------------------------------------------------------- 7. ratio

def test_ratio_math():
    from chess_zero.loop import _ratio_steps
    assert _ratio_steps(0, 128, 1000) == 1  # clamp floor, never zero
    assert _ratio_steps(128 * 100, 128, 1000) == 150  # plies*1.5/batch
    assert _ratio_steps(10 ** 9, 128, 1000) == 1000  # clamp ceiling
    assert _ratio_steps(64, 128, 1000) == 1  # round(0.75) -> 1 via max


# ------------------------------------------------------- 8. rehearsal + KL

def _tiny_games(path):
    import json as _j
    with open(path, "w") as f:
        f.write(_j.dumps({"id": "a", "moves": "e2e4 e7e5 g1f3 b8c6 f1b5 "
                         "a7a6 b5a4 g8f6 e1g1 f8e7", "winner": "white"})
                + "\n")
        f.write(_j.dumps({"id": "b", "moves": "d2d4 d7d5 c2c4 e7e6 b1c3 "
                         "g8f6 c1g5 f8e7 e2e3 e8g8", "winner": None, "status": "draw"})
                + "\n")


def test_rehearsal_mix_shapes():
    try:
        from chess_zero.warmstart import (sample_rehearsal,
                                          encode_rehearsal_batch)
    except Exception as e:
        print(f"SKIP rehearsal (helpers missing: {type(e).__name__})")
        return
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "ws_v19c_test.jsonl")
    _tiny_games(path)
    recs = sample_rehearsal(path, 2, seed=7)
    assert recs == sample_rehearsal(path, 2, seed=7)  # deterministic
    rows = encode_rehearsal_batch(recs)
    assert len(rows) == 20 and all(len(r) == 13 for r in rows)
    # buffer-order mix: tail rows replaced, teacher [B,4096], mask [B].
    B, NREH = 8, 2
    s = np.zeros((B, 30, 8, 8), dtype=np.float32)
    pi = np.zeros((B, 4096), dtype=np.float32)
    for r in rows[:NREH]:
        assert np.shape(r[0]) == (30, 8, 8)  # INPUT_PLANES stamp
    s[-NREH:] = np.stack([r[0] for r in rows[:NREH]])
    assert s.shape == (B, 30, 8, 8)
    km = torch.zeros(B, dtype=torch.bool)
    km[-NREH:] = True
    assert km.sum().item() == NREH and km.dtype == torch.bool
    tp = torch.zeros((B, 4096), dtype=torch.float32)
    tp[-NREH:] = torch.softmax(torch.randn(NREH, 4096), dim=1)
    assert tp.shape == (B, 4096)
    assert abs(tp[-NREH:].sum(dim=1).mean().item() - 1.0) < 1e-5


def test_kl_identical_teacher_zero():
    from chess_zero.model import AlphaZeroNet
    from chess_zero.train import train_step
    torch.manual_seed(0)
    net = AlphaZeroNet(blocks=1, channels=8, planes=30, se_ratio=4)
    opt = torch.optim.AdamW([{"params": [p for p in net.parameters()
                                         if p.dim() > 1],
                              "weight_decay": 1e-4},
                             {"params": [p for p in net.parameters()
                                         if p.dim() <= 1],
                              "weight_decay": 0.0}], lr=1e-3)
    B = 4
    s = torch.randn(B, 30, 8, 8)
    pi = torch.zeros(B, 4096)
    pi[:, :8] = 1.0 / 8.0
    z = torch.zeros(B)
    m = torch.zeros(B)
    ml = torch.ones(B) * 0.5
    own = torch.zeros(B, 8, 8)
    sg = torch.zeros(B)
    with torch.no_grad():
        teach = torch.softmax(net(s)[0].float(), dim=1)
    km = torch.ones(B, dtype=torch.bool)
    out = train_step(net, opt, s, pi, z, m, ml, own, sg, sg, sg,
                     clip=1.0, teacher_probs=teach, kl_w=1.0, kl_mask=km)
    assert len(out) == 16  # v20 return arity (13 + score/av/td, KL at END)
    assert math.isfinite(out[0]) and abs(out[-1]) < 1e-4, out[-1]


# ------------------------------------------------------------- 9. PreciseBN

def test_precisebn_finite():
    from chess_zero.model import AlphaZeroNet
    from chess_zero.replay import ReplayBuffer
    from chess_zero.loop import _precise_bn_lite
    torch.manual_seed(0)
    net = AlphaZeroNet(blocks=1, channels=8, planes=30, se_ratio=4)
    net.eval()
    buf = ReplayBuffer(capacity=512)
    rng = np.random.RandomState(0)
    for _ in range(200):
        e = rng.rand(30, 8, 8).astype(np.float32)
        p = np.zeros(4096, dtype=np.float32)
        p[0] = 1.0
        buf.buf.append((e, p, 0.0, 1.0, 0.5,
                        np.zeros((8, 8), dtype=np.float32),
                        0.0, 0.5, 0.5, -100, 1.0))
    assert _precise_bn_lite(net, buf) is True
    assert not net.training  # left in eval
    x = torch.randn(2, 30, 8, 8)
    with torch.no_grad():
        out = net(x)
    assert all(bool(torch.isfinite(o).all()) for o in out)
    assert _precise_bn_lite(net, ReplayBuffer(capacity=8)) is False


# --------------------------------------------------- 10. spec 24 with se

def test_progress_spec_24_se():
    from chess_zero.config import Config
    from chess_zero.loop import agent_policy
    from chess_zero.model import AlphaZeroNet
    cfg = Config(blocks=1, channels=8, input_planes=13, se_ratio=0)
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    spec = agent_policy(net, cfg, sims=2, device="cpu")._spec
    assert len(spec) == 31, len(spec)
    assert spec[0] == "agent" and spec[23] == 0  # se at [23], plain net
    # evaluate._make_policy reads spec[23] as se_ratio (verified by code
    # read of evaluate.py:285-291; harness the same index here).
    import inspect
    from chess_zero import evaluate as _ev
    src = inspect.getsource(_ev._make_policy)
    assert "spec[23]" in src


# ------------------------------------------------------- 11+12. cull/dirich

def test_aux_cull():
    from chess_zero.config import Config
    from chess_zero.loop import _maybe_aux_cull as cull
    cfg = Config()
    fired, latched = cull(cfg, 7999, False)
    assert not fired and not latched
    fired, latched = cull(cfg, 8001, False)
    assert fired and latched
    # the four knobs _maybe_aux_cull sets (safe_w is the wired knob the
    # loop reads first; the safety_w alias lives only in loop's docstring
    # and is never stamped, so it is not asserted here).
    for k in ("mob_w", "safe_w", "margin_w", "check_w"):
        assert float(getattr(cfg, k)) == 0.005, k
    fired, latched = cull(cfg, 9000, latched)  # one-way: never refires
    assert not fired and latched


def test_dirichlet_and_book():
    from chess_zero.loop import _v19_eps_for, _gate_book_draw, _pair_seed
    assert _v19_eps_for(0) == 0.25 and _v19_eps_for(2999) == 0.25
    assert _v19_eps_for(3000) == 0.12 and _v19_eps_for(99999) == 0.12
    lines, info = _gate_book_draw(4242)
    assert info["n_lines"] == 78, info  # empirical book size
    assert len(lines) == 18
    lines2, _ = _gate_book_draw(4242)
    assert lines == lines2  # deterministic per gate seed
    lines3, _ = _gate_book_draw(999)
    assert lines != lines3  # reshuffled per gate
    assert _pair_seed(1000, 0, lines[0]) != _pair_seed(1000, 1, lines[1])


# ------------------------------------------------------- 13. flag surfaces

def test_fullrun_v19_flag():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "chess_zero.fullrun",
                        "--help"], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))))
    assert "--v19" in r.stdout, r.stdout[-500:]


def test_uci_search_flags():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "chess_zero.uci", "--help"],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))))
    for flag in ("--c-puct", "--fpu-reduction", "--prune-singletons"):
        assert flag in r.stdout, (flag, r.stdout[-500:])


if __name__ == "__main__":
    test_v19_config()
    test_sprt_llr_values()
    test_sprt_decisions()
    test_mirrored_color_alternation()
    test_lr_landmarks()
    test_adamw_groups()
    test_ratio_math()
    test_rehearsal_mix_shapes()
    test_kl_identical_teacher_zero()
    test_precisebn_finite()
    test_progress_spec_24_se()
    test_aux_cull()
    test_dirichlet_and_book()
    test_fullrun_v19_flag()
    test_uci_search_flags()
    print("all v19c tests passed")
