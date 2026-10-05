"""V19 second-sweep tests (Impl-D): loop+config+evaluate+tb_rescore.

Covers (loop.py, config.py, evaluate.py, tb_rescore.py only):
 D8  panel gate: Wilson math, _panel_decide (promote/hold cases),
     ancestor fallback, small live panel (monkeypatched 2/2/2/2).
 D9  PFSP kinds 12/5/3 of 20 (+ pool-file helper).
 D10 anti-RPS check shape + opening_top3 helper math.
 D11 book split 60/18 deterministic; gate draw uses holdout.
 D12 forced-book coin ~30% + opening_book_frac knob.
 D13 SF anchor skipped loudly without an engine (log-only).
 D14 sparring kinds (noisy/tactics legal; mate-in-1; MVV) + 60/20/20
     rotation; legacy even/odd default untouched.
 D15 game-id hashing ~5%; test-loss loud skip without sample_test.
 D16 atomic_save roundtrip + save_checkpoint meta; disk guard.
 D17 cum counters persist in best.pt meta + iter-file meta.
 D18 BN/plane report structure + drift alarm.
 D19 probe harness: mass sums to 1, top-3 legal, mate detection agrees
     with brute force (stub-boosted net passes thresholds; trained-net
     thresholds documented as needing trained weights).
 D20 panel agents built with noise=0.0 (spy on agent_from_weights).
 D21 contempt audit: absent from shape_targets body (labels pure),
     present in draw_leaf_value (sanctioned search shaping); dynamic
     tiny game with contempt>0 yields pure z in {-1,0,1}.
 D24 V19 aux weights (material 0.01, margin 0.02, mob/safe 0.01, ml
     0.05, own/reply/ent/smooth) + train_step kwarg surface.
 D26 tb_rescore: disk gate, WDL/DTZ maps, loud skip without tables.
 Flags: fullrun --v19, uci search flags (+ safety-drop-thr).

Run: python3 tests/test_v19d.py  (or pytest tests/test_v19d.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- D24 config

def test_v19_aux_weights_d24():
    from chess_zero.config import V19_CONFIG as c, Config
    assert c.aux_w == 0.01      # material head (was 0.1)
    assert c.ml_w == 0.05       # plies head
    assert c.margin_w == 0.02
    assert c.mob_w == 0.01 and c.safe_w == 0.01
    assert c.own_w == 0.12 and c.reply_w == 0.15
    assert c.soft_w == 0.3 and c.check_w == 0.01
    assert c.ent_w == 0.002 and c.smooth_eps == 0.01
    assert c.opening_book_frac == 0.3
    assert c.sf_anchor_games == 10 and c.sf_anchor_rung == "sf-elo1350"
    b = Config()  # yesterday defaults preserved for old configs
    assert (b.aux_w, b.margin_w, b.mob_w, b.safe_w) == (0.1, 0.05, 0.05,
                                                       0.05)
    assert b.opening_book_frac == 0.0 and b.sf_anchor_games == 0


def test_train_step_kwargs():
    import inspect
    from chess_zero.train import train_step
    params = set(inspect.signature(train_step).parameters)
    for k in ("value_w", "ent_reg", "smooth_eps", "teacher_probs",
              "kl_w", "kl_mask", "reply", "policy_w"):
        assert k in params, k


# ---------------------------------------------------------------- D8 panel

def test_wilson_lower():
    from chess_zero.loop import wilson_lower
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(20, 20) > 0.50          # perfect 20/20
    assert wilson_lower(12, 20) < 0.50          # 60% on 20: not enough
    assert wilson_lower(16, 20) > 0.50          # 80% on 20: enough
    assert wilson_lower(1, 2) < wilson_lower(2, 2)  # monotone


def test_panel_decide():
    from chess_zero.loop import _panel_decide
    from chess_zero.config import V19_CONFIG as c
    ok = _panel_decide({"incumbent": (5, 2, 1), "ancestor": (4, 2, 0),
                        "punisher": (4, 0, 0), "random": (2, 0, 0)}, c)
    assert not ok["promote"] and ok["score"] >= 0.60, ok  # insufficient incumbent-only evidence
    assert ok["incumbent"] >= 0.55 and ok["ancestor"] >= 0.55
    assert ok["wilson"] < 0.50 and math.isfinite(ok["llr"])
    # RPS hold: overall 80% but incumbent split 50%.
    rps = _panel_decide({"incumbent": (4, 4, 0), "ancestor": (6, 0, 0),
                         "punisher": (4, 0, 0), "random": (2, 0, 0)}, c)
    assert not rps["promote"], rps
    # weak overall hold (52.5%).
    weak = _panel_decide({"incumbent": (5, 2, 1), "ancestor": (4, 2, 0),
                          "punisher": (1, 3, 0), "random": (0, 2, 0)}, c)
    assert not weak["promote"] and weak["score"] == ok["score"], weak
    # empty scores never promote.
    assert not _panel_decide({}, c)["promote"]


def test_ancestor_fallback(tmpdir=None):
    import tempfile
    from chess_zero.loop import _ancestor_weights, _pool_files
    d = tempfile.mkdtemp()
    w, note = _ancestor_weights(d, {"fake": 1})
    assert w == {"fake": 1} and "fallback" in note  # <4 snapshots
    assert _pool_files(d) == []
    # 4 snapshots -> promo-3 = first of the four.
    import torch
    os.makedirs(os.path.join(d, "champions"))
    for s in (1, 2, 3, 4):
        torch.save({"s": s}, os.path.join(d, "champions",
                                          f"champ_seq{s}.pt"))
    w, note = _ancestor_weights(d, {"fake": 1})
    assert note == "champ_seq1" and str(w).endswith("champ_seq1.pt"), \
        (w, note)
    assert len(_pool_files(d)) == 4


def test_panel_gate_small():
    import torch
    import tempfile
    from chess_zero.config import Config
    from chess_zero.model import AlphaZeroNet
    from chess_zero import loop as L
    cfg = Config(blocks=1, channels=8, input_planes=13, sims=2,
                 gate_opening_moves=0, mirror_gate=True, gate_incumbent_games=2)
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    cw = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    d = tempfile.mkdtemp()
    old = L.PANEL_SPLITS
    L.PANEL_SPLITS = (("incumbent", 2), ("ancestor", 2), ("punisher", 2),
                      ("random", 2))
    try:
        g = L._panel_gate(cw, cw, cfg, 2, "cpu", 1, d, seed_base=31337)
    finally:
        L.PANEL_SPLITS = old
    assert g["via"] == "panel", g
    assert set(g["panel"]) == {"incumbent", "ancestor", "punisher",
                               "random"}
    assert sum(g["challenger_wld"]) == 2, g
    assert isinstance(g["sprt_pass"], bool) and math.isfinite(g["llr"])
    assert "fallback" in g["ancestor"]  # no snapshots in tmpdir


def test_gate_noise_zero():
    import torch
    from chess_zero.config import Config
    from chess_zero.model import AlphaZeroNet
    from chess_zero import loop as L
    cfg = Config(blocks=1, channels=8, input_planes=13, sims=2,
                 gate_opening_moves=0, mirror_gate=True, gate_incumbent_games=2)
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    cw = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    import tempfile
    d = tempfile.mkdtemp()
    seen = []
    real = L.agent_from_weights
    def spy(w, cfg_, sims=None, device="cpu", noise=0.0, **kwargs):
        seen.append(noise)
        return real(w, cfg_, sims=sims, device=device, noise=noise, **kwargs)
    L.agent_from_weights = spy
    old = L.PANEL_SPLITS
    L.PANEL_SPLITS = (("incumbent", 2), ("ancestor", 2), ("punisher", 2),
                      ("random", 2))
    try:
        L._panel_gate(cw, cw, cfg, 2, "cpu", 1, d, seed_base=777)
    finally:
        L.agent_from_weights = real
        L.PANEL_SPLITS = old
    assert seen and all(n == 0.0 for n in seen), seen  # D20


# ---------------------------------------------------------------- D9 PFSP

def test_pfsp_kinds():
    from chess_zero.loop import _pfsp_kinds as k
    kinds = k(20)
    assert len(kinds) == 20
    assert kinds.count("self") == 12
    assert kinds.count("pool") == 5
    assert kinds.count("punisher") == 3
    assert k(0) == [] and k(7).count("self") + k(7).count("pool") + \
        k(7).count("punisher") == 7
    assert k(20) == k(20)  # deterministic, no RNG


# ---------------------------------------------------------------- D10

def test_anti_rps_and_opening():
    import torch
    import tempfile
    from chess_zero.config import Config
    from chess_zero.model import AlphaZeroNet
    from chess_zero.loop import (_anti_rps_check, opening_top3_share)
    cfg = Config(blocks=1, channels=8, input_planes=13, sims=2,
                 gate_opening_moves=0, mirror_gate=True, gate_incumbent_games=2)
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    cw = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    r = _anti_rps_check(cw, tempfile.mkdtemp(), cfg, 2, "cpu",
                        seed_base=99)
    assert set(r) >= {"score", "wld", "hold", "note", "sims"}
    assert sum(r["wld"]) == 4 and isinstance(r["hold"], bool)
    o = opening_top3_share([("e2e4", "e7e5")] * 6 +
                           [(f"m{i}", "x") for i in range(4)])
    assert o == {"top3": 0.8, "n": 10}, o
    assert opening_top3_share([]) == {"top3": 0.0, "n": 0}
    assert opening_top3_share(None)["n"] == 0


# ---------------------------------------------------------------- D11/D12

def test_book_split():
    from chess_zero.book import load_book
    from chess_zero.loop import split_book_lines, _gate_book_draw
    lines = load_book(os.path.join(REPO, "chess_zero",
                                   "book_empirical.json"))
    assert len(lines) == 78
    tr, ho = split_book_lines(lines)
    assert len(tr) == 60 and len(ho) == 18
    assert split_book_lines(lines) == (tr, ho)  # deterministic
    assert {repr(l) for l in tr}.isdisjoint({repr(l) for l in ho})
    assert len({repr(l) for l in tr + ho}) == 78  # lossless
    # gate draws 10 from the HOLDOUT split (reshuffled per gate seed).
    g1, info = _gate_book_draw(4242)
    assert len(g1) == 18 and info["n_lines"] == 78
    assert info["n_holdout"] == 18 and info.get("split") == "holdout"
    assert all(list(l) in ho for l in g1)
    g2, _ = _gate_book_draw(4242)
    assert g1 == g2
    assert g1 != _gate_book_draw(999)[0]


def test_forced_book_coin():
    from chess_zero.loop import _roll_book_game
    from chess_zero.config import V19_CONFIG as c
    assert c.opening_book_frac == 0.3
    rng = random.Random(0)
    n = sum(_roll_book_game(rng, 0.3) for _ in range(2000))
    assert 500 < n < 700, n  # ~30%
    assert not _roll_book_game(random.Random(1), 0.0)
    assert _roll_book_game(random.Random(1), 1.0)


def test_forced_book_applies_train_split():
    # D12 application: train-split lines open real positions (6 plies).
    import random as _r
    from chess_zero.book import load_book, split_book_lines, \
        take_book_opening
    from chess_zero.game import State
    lines = load_book(os.path.join(REPO, "chess_zero",
                                   "book_empirical.json"))
    tr, _ho = split_book_lines(lines)
    s0 = State.initial()
    s, n = take_book_opening(s0, [s0.rep_key()], tr, 6,
                             rng=_r.Random(7))
    assert n == 6, n
    assert s.board.fen() != s0.board.fen()
    assert s.board.ply() == 6


# ---------------------------------------------------------------- D13

def test_sf_anchor_skipped():
    from chess_zero.config import Config, V19_CONFIG as c
    from chess_zero.loop import _sf_anchor
    assert c.sf_anchor_games == 10
    # no stockfish here -> loud skip dict, never raises.
    r = _sf_anchor("nonexistent.pt", c, 2, 1, ".", "/nonexistent/sf")
    assert "skipped" in r, r
    assert _sf_anchor("x", Config(), 2, 1, ".")["skipped"].startswith(
        "sf_anchor_games=0")


# ---------------------------------------------------------------- D14

def test_sparring_kinds():
    from chess_zero.game import State
    from chess_zero.evaluate import (
        sparring_for_game, sparring_kind_for_game, noisy_move,
        tactics_move, greedy_move, punisher_move)
    import chess
    # legacy default untouched (frozen by test_full).
    assert sparring_for_game(0) is greedy_move
    assert sparring_for_game(1) is punisher_move
    assert sparring_for_game(8) is greedy_move
    # kind selection.
    assert sparring_for_game(0, kind="noisy") is noisy_move
    assert sparring_for_game(0, kind="tactics") is tactics_move
    assert sparring_for_game(1, kind="punisher") is punisher_move
    try:
        sparring_for_game(0, kind="bogus")
        assert False, "must raise"
    except ValueError:
        pass
    # rotation 60/20/20 over 10.
    kinds = [sparring_kind_for_game(i) for i in range(10)]
    assert kinds.count("punisher") == 6
    assert kinds.count("noisy") == 2
    assert kinds.count("tactics") == 2
    assert [sparring_kind_for_game(i) for i in range(10)] == kinds
    # noisy/tactics return legal moves.
    st = State(chess.Board())
    assert noisy_move(st) in st.legal_moves()
    assert tactics_move(st) in st.legal_moves()
    # mate-in-1 found.
    m = State(chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/"
                          "RNB1K1NR w KQkq - 0 1"))
    assert m.to_uci(tactics_move(m)) == "f3f7"
    # MVV: Nxe6 (rook) over Nxc6 (pawn).
    v = State(chess.Board("4k3/8/2p1r3/8/3N4/8/5PPP/4K3 w - - 0 1"))
    assert v.to_uci(tactics_move(v)) == "d4e6"


# ---------------------------------------------------------------- D15

def test_game_id_hash():
    from chess_zero.loop import game_id_is_test
    ids = [f"seed-{i}" for i in range(2000)]
    n = sum(game_id_is_test(g) for g in ids)
    assert 40 < n < 220, n  # ~5%
    assert all(game_id_is_test(g) == game_id_is_test(g) for g in ids[:5])


def test_test_losses_skip():
    import torch
    from chess_zero.config import Config
    from chess_zero.model import AlphaZeroNet
    from chess_zero.replay import ReplayBuffer
    from chess_zero.loop import _test_losses
    assert not hasattr(ReplayBuffer(capacity=8), "sample_test")
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    r = _test_losses(net, ReplayBuffer(capacity=8), Config())
    assert r == {"test_loss": None, "test_loss_p": None,
                 "test_loss_v": None}, r


# ---------------------------------------------------------------- D16

def test_atomic_save_and_disk():
    import torch
    import tempfile
    from chess_zero.loop import (atomic_save, disk_ok, save_checkpoint)
    from chess_zero.config import Config
    from chess_zero.model import AlphaZeroNet
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sub", "w.pt")
    obj = {"a": torch.ones(3), "n": 7}
    assert atomic_save(obj, p) == p
    back = torch.load(p, map_location="cpu", weights_only=False)
    assert back["n"] == 7 and bool((back["a"] == 1).all())
    assert not any(f.startswith("w.pt.tmp") for f in os.listdir(
        os.path.join(d, "sub")))  # tmp replaced, not left behind
    ok, free = disk_ok(d, 0.000001)
    assert ok and free > 0
    ok, _ = disk_ok(d, 1e12)
    assert not ok
    # save_checkpoint persists meta (D17 iter-file side).
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    meta = {"iter": 3, "cum_steps": 111, "cum_games": 222}
    save_checkpoint(net, opt, os.path.join(d, "iter3.pt"), meta)
    m2 = torch.load(os.path.join(d, "iter3.pt"), map_location="cpu",
                    weights_only=False)["meta"]
    assert m2["cum_steps"] == 111 and m2["cum_games"] == 222


# ---------------------------------------------------------------- D18

def test_bn_plane_report():
    import torch
    from chess_zero.model import AlphaZeroNet
    from chess_zero.loop import bn_plane_report
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    net.eval()
    r = bn_plane_report(net)
    assert set(r) >= {"bn_first", "bn_last", "plane_top3_ratio",
                      "alarms"}
    assert math.isfinite(r["bn_first"]) and math.isfinite(r["bn_last"])
    assert r["plane_top3_ratio"] is not None and \
        r["plane_top3_ratio"] > 0
    assert r["alarms"] == []  # fresh net: no alarms
    # drift alarm: prev far from current zeros.
    r2 = bn_plane_report(net, {"bn_first": 5.0, "bn_last": 5.0})
    assert any("drift" in a for a in r2["alarms"]), r2


# ---------------------------------------------------------------- D19

def _stub_net(boost_uci=("f3", "f7"), fen=None):
    """Stub model: uniform logits + boost on one UCI (tests the probe
    harness thresholds, which need a non-random policy)."""
    import torch
    import chess as _c
    from chess_zero.game import State as _S, flip_action as _flip
    st = _S(_c.Board(fen))
    mv = _c.Move.from_uci(boost_uci[0] + boost_uci[1])
    a = mv.from_square * 64 + mv.to_square
    if st.board.turn == _c.BLACK:
        a = _flip(a)

    class Stub:
        training = False

        def __call__(self, x):
            lg = torch.zeros(x.shape[0], 4096)
            lg[:, a] = 5.0
            return (lg,)

        def eval(self):
            return self

        def train(self, m=True):
            return self

    return Stub()


def test_probe_harness():
    import torch
    from chess_zero.model import AlphaZeroNet
    from chess_zero.loop import probe_report
    mate_fen = ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR "
                "w KQkq - 0 1")
    stub = _stub_net(("f3", "f7"), mate_fen)
    rep = probe_report(stub, "cpu", fens=[mate_fen])
    assert len(rep) == 1 and "error" not in rep[0], rep
    assert rep[0]["mass_renorm"] == 1.0
    assert rep[0]["mates"] == ["f3f7"] and rep[0]["mate_in_top3"]
    assert rep[0]["top3"][0] == "f3f7"
    # real (random) net: harness integrity, no errors, legal top-3.
    net = AlphaZeroNet(blocks=1, channels=8, planes=13, se_ratio=0)
    rep2 = probe_report(net, "cpu")
    assert len(rep2) == 5 and all("error" not in r for r in rep2), rep2
    for r in rep2:
        assert r["mass_renorm"] == 1.0 and len(r["top3"]) == 3
        assert 0.0 <= r["promo_prior"] <= 1.0
    promos = [r for r in rep2 if r["fen"].startswith("8/2P5")]
    assert len(promos) == 1  # white-promo FEN present
    assert any(r["mates"] == ["f3f7"] for r in rep2)  # mate detected


# ---------------------------------------------------------------- D21

def _code_tokens(src):
    """Source minus comments/docstrings/strings (audit-grade: mentions in
    prose don't contaminate labels; live references do)."""
    import io
    import tokenize
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def test_contempt_audit():
    import inspect
    from chess_zero import selfplay as sp
    lab_src = inspect.getsource(sp.shape_targets)
    # prose may cite history ("v5-v8 contempt overwrite"); live code
    # must not reference contempt in the label builder.
    assert "contempt" not in _code_tokens(lab_src), \
        [l for l in lab_src.splitlines() if "contempt" in l]
    dlv_src = inspect.getsource(sp.draw_leaf_value)
    assert "contempt" in dlv_src  # sanctioned: search shaping only
    # dynamic: contempt>0 game still yields pure z in {-1,0,1}.
    import dataclasses
    import torch
    from chess_zero.config import TEST_CONFIG
    from chess_zero.model import AlphaZeroNet
    cfg = dataclasses.replace(TEST_CONFIG, contempt=0.3, sims=2,
                              temp_moves=2)
    net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels,
                       planes=cfg.input_planes)
    ex, _ = sp.play_game(net, cfg, device="cpu")
    assert len(ex) > 0
    for e in ex:
        assert float(e[2]) in (-1.0, 0.0, 1.0), e[2]


# ---------------------------------------------------------------- D26

def test_tb_rescore():
    from chess_zero import tb_rescore as tb
    assert tb.disk_free_gb(".") > 0
    # nonexistent path resolves to its parent volume (where a file
    # would be created); truly unreadable -> -1.0.
    assert tb.disk_free_gb("/nonexistent-xyz") > 0
    g = tb.ensure_tables("/nonexistent-tb-dir-xyz")
    assert g["ok"] is False and g["reason"] in ("disk-full",
                                                "no-tables",
                                                "disk unreadable"), g
    assert tb.wdl_to_q(2) == 1.0 and tb.wdl_to_q(-2) == -1.0
    assert tb.wdl_to_q(1) == 0.0 and tb.wdl_to_q(0) == 0.0
    assert tb.dtz_to_ml(None) == 1.0 and tb.dtz_to_ml(0) == 0.0
    assert 0.0 < tb.dtz_to_ml(50, 300) <= 1.0
    # no tables -> all-None plies, loud skip, never raises.
    r = tb.rescore_game(["e2e4", "e7e5", "g1f3"], tables=None)
    assert r["n_probed"] == 0 and all(p["q"] is None for p in r["plys"])
    s = tb.rescore_games_file("nonexistent.jsonl", "o.jsonl",
                              tables_dir="/nonexistent-tb-dir-xyz")
    assert s["games"] == 0 and s["skipped"], s
    # this machine clears the 3GB D26 disk gate.
    assert tb.disk_free_gb("/") >= 3.0


# ---------------------------------------------------------------- flags

def test_fullrun_v19_flag():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "chess_zero.fullrun",
                        "--help"], capture_output=True, text=True, cwd=REPO)
    assert "--v19" in r.stdout, r.stdout[-500:]


def test_uci_search_flags():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "chess_zero.uci", "--help"],
                       capture_output=True, text=True, cwd=REPO)
    for flag in ("--c-puct", "--fpu-reduction", "--prune-singletons",
                 "--safety-drop-thr"):
        assert flag in r.stdout, (flag, r.stdout[-500:])


if __name__ == "__main__":
    test_v19_aux_weights_d24()
    test_train_step_kwargs()
    test_wilson_lower()
    test_panel_decide()
    test_ancestor_fallback()
    test_panel_gate_small()
    test_gate_noise_zero()
    test_pfsp_kinds()
    test_anti_rps_and_opening()
    test_book_split()
    test_forced_book_coin()
    test_forced_book_applies_train_split()
    test_sf_anchor_skipped()
    test_sparring_kinds()
    test_game_id_hash()
    test_test_losses_skip()
    test_atomic_save_and_disk()
    test_bn_plane_report()
    test_probe_harness()
    test_contempt_audit()
    test_tb_rescore()
    test_fullrun_v19_flag()
    test_uci_search_flags()
    print("all v19d tests passed")
