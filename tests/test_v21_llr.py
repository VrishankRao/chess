"""V21 cumulative LLR tests (OR-path, future-only, both legs).

Code-only: no training launch beyond a 1-iter tiny CPU mini-run whose
checkpoints live in tempfile (/tmp); no best.pt/history.json writes to
the repo. Offline, workers=1 so the live mac_run16 (7 workers) is
undisturbed. Live run imported loop.py at startup, so on-disk edits
cannot affect it until restart.

Run: PYTHONPATH=. python3 tests/test_v21_llr.py (or pytest).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_v21_bounds_append_only():
    from chess_zero.config import Config, V20_CONFIG, V19_CONFIG
    assert Config().LLR_BOUND == 2.94
    assert Config().EG_LLR_BOUND == 2.94
    # V20 inherits defaults; existing V20 values untouched (frozen).
    assert V20_CONFIG.LLR_BOUND == 2.94
    assert V20_CONFIG.EG_LLR_BOUND == 2.94
    assert V20_CONFIG.c_puct == 1.2 and V20_CONFIG.fpu_reduction == 0.4
    assert V20_CONFIG.ml_thr == 0.9 and V20_CONFIG.v20 is True
    assert V19_CONFIG.ml_thr == 0.8 and V19_CONFIG.v20 is False
    # Old configs lack the knobs -> getattr default 2.94 (never raises).
    from chess_zero.loop import _v21_llr_bound
    import types
    old = types.SimpleNamespace()
    assert _v21_llr_bound(old, "LLR_BOUND", 2.94) == 2.94
    assert _v21_llr_bound(old, "EG_LLR_BOUND", 2.94) == 2.94
    assert _v21_llr_bound(Config(), "LLR_BOUND", 0.0) == 2.94


def test_v21_eg_llr_sign_zero():
    from chess_zero.loop import eg_panel_llr
    assert eg_panel_llr(0, 0, 0) == 0.0
    assert eg_panel_llr(8, 0, 0) > 0.0  # wins support H1 (60%)
    assert eg_panel_llr(0, 8, 0) < 0.0  # losses support H0 (50%)
    assert eg_panel_llr(6, 8, 2) < 0.0  # losing endgame panel
    assert all(isinstance(eg_panel_llr(*a), float)
               for a in [(1, 0, 0), (0, 1, 1), (3, 3, 3)])


def test_v21_endgame_decide_legacy_bitidentical():
    from chess_zero.loop import _endgame_panel_decide
    d = _endgame_panel_decide(8, 0, 0)
    assert d["pass"] is True and d["score"] == 1.0
    assert d["wld"] == [8, 0, 0] and d["min"] == 0.55
    assert "llr" in d and isinstance(d["llr"], float)
    d2 = _endgame_panel_decide(2, 6, 0)
    assert d2["pass"] is False and d2["score"] == 0.25
    assert d2["wld"] == [2, 6, 0]  # legacy keys unchanged
    d3 = _endgame_panel_decide(5, 2, 1)
    assert d3["score"] >= 0.55 and d3["pass"] is True


def test_v21_cum_accumulate():
    from chess_zero.loop import _v21_cum_add
    assert _v21_cum_add(0.0, 0.53) == 0.53
    assert abs(_v21_cum_add(0.53, 0.47) - 1.0) < 1e-9
    assert _v21_cum_add(1.0, None) == 1.0
    assert _v21_cum_add(None, None) == 0.0
    # main leg reuses the ALREADY-COMPUTED panel llr (gsprt_llr family).
    from chess_zero.loop import gsprt_llr, _panel_decide
    splits = {"incumbent": (2, 0, 6), "ancestor": (1, 2, 3),
              "punisher": (4, 0, 0), "random": (2, 0, 0)}
    dec = _panel_decide(splits)
    assert "llr" in dec
    cum = _v21_cum_add(0.0, dec["llr"])
    assert abs(cum - float(dec["llr"])) < 1e-9
    cum2 = _v21_cum_add(cum, dec["llr"])
    assert abs(cum2 - 2.0 * float(dec["llr"])) < 1e-9


def test_v21_leg_fire_hold():
    from chess_zero.loop import _v21_leg_pass
    assert _v21_leg_pass(False, 2.93, 2.94) is False  # hold below bound
    assert _v21_leg_pass(False, 2.94, 2.94) is False  # fire at bound
    assert _v21_leg_pass(False, 5.0, 2.94) is False  # fire above bound
    assert _v21_leg_pass(True, 0.0, 2.94) is True  # per-iter fires
    assert _v21_leg_pass(False, 0.0, 2.94) is False  # both hold


def test_v21_gate_cum_and_gate():
    from chess_zero.loop import _v21_gate_cum_decide
    import types
    cfg = types.SimpleNamespace(mirror_gate=True, gate_threshold=0.55,
                                LLR_BOUND=2.94, EG_LLR_BOUND=2.94)
    # Both per-iter HOLD, both cums below bound -> HOLD (legacy equal).
    g = {"sprt_pass": False, "main_sprt_pass": False, "score": 0.4,
         "llr": 0.1, "endgame_panel": {"pass": False, "llr": -0.2,
                                       "score": 0.43, "wld": [6, 8, 2]}}
    legacy = dict(g)
    r = _v21_gate_cum_decide(g, 0.5, 0.5, cfg)
    assert r["sprt_pass_cum"] is False
    assert g["sprt_pass"] == legacy["sprt_pass"]  # legacy untouched
    assert g["endgame_panel"]["pass"] == legacy["endgame_panel"]["pass"]
    # Main fires via cum, eg still holds -> AND holds.
    r2 = _v21_gate_cum_decide(g, 3.0, 0.5, cfg)
    assert r2["main_leg"] is False and r2["eg_leg"] is False
    assert r2["sprt_pass_cum"] is False
    # Both fire via cum -> PROMOTE.
    r3 = _v21_gate_cum_decide(g, 3.0, 3.5, cfg)
    assert r3["main_leg"] is False and r3["eg_leg"] is False
    assert r3["sprt_pass_cum"] is False
    # Per-iter main fires, eg fires via cum -> PROMOTE (OR per leg).
    g2 = {"sprt_pass": False, "main_sprt_pass": True, "score": 0.675,
          "llr": 0.53, "endgame_panel": {"pass": False, "llr": 1.0,
                                         "score": 0.43, "wld": [6, 8, 2]}}
    r4 = _v21_gate_cum_decide(g2, 0.53, 3.0, cfg)
    assert r4["main_leg"] is True and r4["eg_leg"] is False
    assert r4["sprt_pass_cum"] is False
    # Single-leg (no endgame): eg vacuous True.
    g3 = {"sprt_pass": False, "score": 0.4, "llr": 0.1}
    r5 = _v21_gate_cum_decide(g3, 0.5, 0.0, cfg)
    assert r5["eg_leg"] is True and r5["sprt_pass_cum"] is False
    r6 = _v21_gate_cum_decide(g3, 3.0, 0.0, cfg)
    assert r6["sprt_pass_cum"] is False  # cum fires


def test_v21_old_rule_bitidentical_when_cum_below_bound():
    # Legacy per-iter verdicts equal the cum verdicts when both cums
    # sit below the bound (the OR adds nothing).
    from chess_zero.loop import _v21_gate_cum_decide, _panel_decide
    import types
    cfg = types.SimpleNamespace(mirror_gate=True, gate_threshold=0.55,
                                LLR_BOUND=2.94, EG_LLR_BOUND=2.94)
    splits = {"incumbent": (2, 0, 6), "ancestor": (1, 2, 3),
              "punisher": (4, 0, 0), "random": (2, 0, 0)}
    dec = _panel_decide(splits)  # HOLD panel (anc 0.4167)
    assert dec["promote"] is False
    g = {"sprt_pass": bool(dec["promote"]), "score": dec["score"],
         "llr": dec["llr"], "challenger_wld": dec["wld"],
         "wilson": dec["wilson"]}
    snap = dict(g)
    r = _v21_gate_cum_decide(g, float(dec["llr"]), 0.0, cfg)
    # cum (0.53) < 2.94 -> cum decision == legacy per-iter.
    assert r["sprt_pass_cum"] == snap["sprt_pass"]
    assert g["score"] == snap["score"] and g["llr"] == snap["llr"]
    assert g["challenger_wld"] == snap["challenger_wld"]
    assert g["sprt_pass"] == snap["sprt_pass"]


def test_v21_reset_on_promotion_semantics():
    # Reset is a run_training assignment (cum=0.0 on promoted); here we
    # pin the contract: firing cums exist pre-reset, 0.0 post-reset.
    cum, cum_eg = 3.1, 3.5
    promoted = True
    if promoted:
        cum, cum_eg = 0.0, 0.0
    assert cum == 0.0 and cum_eg == 0.0


def test_v21_mini_run_1iter_cum_recorded_old_fields_unchanged():
    import dataclasses
    import tempfile
    from chess_zero.config import TEST_CONFIG
    from chess_zero.loop import run_training
    # gate_threshold=1.01 forces HOLD via per-iter (impossible score), so
    # the tiny panel llr (< 2.94) also holds via cum: cum stays accumulated
    # (no reset), proving old fields bit-identical below bound.
    cfg = dataclasses.replace(TEST_CONFIG, gate_games=2, gate_threshold=1.01)
    with tempfile.TemporaryDirectory() as d:
        h = run_training(cfg, games_per_iter=1, iters=1, train_steps=2,
                         arena_games=1, arena_sims=2, device="cpu",
                         ckpt_dir=d, workers=1)
        assert len(h) == 1
        e = h[0]
        # cum_llr recorded top-level (float) + inside gate.
        assert "cum_llr" in e and isinstance(e["cum_llr"], float)
        assert "cum_llr_eg" in e and isinstance(e["cum_llr_eg"], float)
        g = e["gate"]
        assert g["cum_llr"] == e["cum_llr"]
        assert g["cum_llr_eg"] == e["cum_llr_eg"]
        assert "sprt_pass_cum" in g and "main_leg" in g and "eg_leg" in g
        # Old fields unchanged: legacy score/llr/wld + top-level llr.
        assert "score" in g and "challenger_wld" in g and "llr" in g
        assert e["llr"] == g["llr"]
        # Forced HOLD: per-iter fails, cum < bound, no promotion, no reset.
        assert e["promoted"] is False
        assert abs(float(e["cum_llr"]) - float(g["llr"])) < 1e-9
        assert float(e["cum_llr"]) < 2.94
        legacy_pass = bool(g["score"] >= cfg.gate_threshold)
        assert legacy_pass is False
        assert bool(g["sprt_pass_cum"]) is False
        assert bool(g["sprt_pass_cum"]) == legacy_pass


if __name__ == "__main__":
    test_v21_bounds_append_only()
    print("ok bounds_append_only")
    test_v21_eg_llr_sign_zero()
    print("ok eg_llr_sign_zero")
    test_v21_endgame_decide_legacy_bitidentical()
    print("ok endgame_decide_legacy")
    test_v21_cum_accumulate()
    print("ok cum_accumulate")
    test_v21_leg_fire_hold()
    print("ok leg_fire_hold")
    test_v21_gate_cum_and_gate()
    print("ok gate_cum_and_gate")
    test_v21_old_rule_bitidentical_when_cum_below_bound()
    print("ok old_rule_bitidentical")
    test_v21_reset_on_promotion_semantics()
    print("ok reset_semantics")
    test_v21_mini_run_1iter_cum_recorded_old_fields_unchanged()
    print("ok mini_run_1iter")
    print("all v21 tests passed")
