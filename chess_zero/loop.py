"""Outer training loop: incumbent self-play -> challenger train -> arena +
gate -> promote. Only arena/gate win-rate counts as improvement.
v4: LR decay, arena gating with best-on-disk, SF arena at deploy sims.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch

from .config import Config
from .model import AlphaZeroNet
from .replay import ReplayBuffer
from .selfplay import play_game, make_evaluate
from .train import train_step
from . import mcts as mcts_mod
from .evaluate import random_move, greedy_move, play_match, elo_diff


def agent_policy(model, cfg: Config, sims: int | None = None, device="cpu",
                 noise: float = 0.0, temp_moves: int = 0, use_tt: bool = True,
                 rust_tree: bool = False):
    # V21.1 E4 loop seam (flag only, default OFF): explicit param OR the
    # cfg knob (run_training stamps cfg.rust_tree for worker pickling).
    # False = mcts_mod.search exactly as yesterday; True = identical
    # params through mcts_bridge.RustTreeBackend (stub until batch C
    # lands mctscore; parity-exact per test_mcts_parity).
    _rust_tree = bool(rust_tree or getattr(cfg, "rust_tree", False))
    evaluate = make_evaluate(model, device)
    n = sims if sims is not None else cfg.sims

    def fn(state):
        fn._hist[:] = list(state._hist)
        from .selfplay import (tactical_action as _tac,
                               apply_move_guards as _guards)
        tt = fn._tt if use_tt else None
        qdepth = int(getattr(cfg, "quiescence_depth", 0) or 0)
        fbonus = float(getattr(cfg, "forcing_bonus", 0.0) or 0.0)
        edge_scale = float(getattr(cfg, "contempt_edge_scale", 0.0)
                           or 0.0)
        _kw = dict(c_puct=cfg.c_puct,
                   dirichlet_alpha=cfg.dirichlet_alpha,
                   dirichlet_eps=noise,  # 0 in arena, eps in gate
                   tt=tt, history=fn._hist,
                   contempt=float(
                       getattr(cfg, "contempt", 0.0) or 0.0),
                   asymmetric_contempt=bool(getattr(
                       cfg, "asymmetric_contempt", False)),
                   quiescence_depth=qdepth,
                   forcing_bonus=fbonus,
                   contempt_edge_scale=edge_scale,
                   leaf_batch=int(getattr(
                       cfg, "leaf_batch", 1) or 1),
                   virtual_loss=float(getattr(
                       cfg, "virtual_loss", 1.0) or 1.0),
                   # v19 search knobs (mcts accepts both;
                   # 0.0/False = yesterday bit-exact).
                   fpu_reduction=float(getattr(
                       cfg, "fpu_reduction", 0.0) or 0.0),
                   prune_singletons=bool(getattr(
                       cfg, "prune_singletons", False)),
                   futile_stop=False, ml_fn=getattr(evaluate, "ml_fn", None),
                   ml_slope=float(getattr(cfg, "ml_slope", 0.0)),
                   ml_cap=float(getattr(cfg, "ml_cap", 0.07)),
                   ml_thr=float(getattr(cfg, "ml_thr", 0.8)))
        if _rust_tree:
            from .mcts_bridge import RustTreeBackend as _RTB
            # F1/F2: per-game GameStore when use_tt (dict stays empty).
            _tt_arg = fn._rust_store if (
                use_tt and getattr(fn, "_rust_store", None) is not None
            ) else tt
            _kw = dict(_kw, tt=_tt_arg)
            pi = _RTB().search(state, evaluate, n_sims=n, **_kw)[0]
        else:
            pi = mcts_mod.search(state, evaluate, n_sims=n, **_kw)
        fn._hist.append(state.rep_key())
        # audit-1.8: temp_moves are PLIES everywhere (self-play compares
        # state.ply_count; the old len(_hist) counted only this agent's own
        # moves, i.e. full moves in arena vs plies in self-play).
        tac_on = bool(getattr(cfg, "tactical_override", False))
        tac_thr = float(getattr(cfg, "tac_threshold", 0.09))
        veto_on = bool(getattr(cfg, "blunder_veto", False))
        tac = _tac(state, tac_thr) if tac_on else None
        if tac is None and temp_moves and state.ply_count < temp_moves:
            legal = np.flatnonzero(pi)
            tot = pi[legal].sum()
            if tot > 0:
                choice = int(np.random.choice(legal, p=pi[legal] / tot))
                choice, _, _vet = _guards(state, choice, pi, None,
                                          veto_on, tac_thr)
                if not _vet:
                    from .selfplay import finish_move as _fin, safety_guard as _safe
                    choice, _ = _safe(state, choice, pi, model, device, cfg)
                    choice = _fin(state, choice, pi, model, device, cfg)
                # §6.5: prune to the promoted child only (bounds memory,
                # kills history-merge hazard). No-op when use_tt=False.
                # F2: rust path prunes the GameStore; the dict stays empty.
                try:
                    _rs = getattr(fn, "_rust_store", None)
                    if _rust_tree and use_tt and _rs is not None:
                        _rs.prune_to(state.apply(choice).key())
                    else:
                        mcts_mod.prune_tt(fn._tt,
                                          state.apply(choice).key())
                except Exception:
                    pass
                return choice
        choice, _, _vet = _guards(state, int(np.argmax(pi)), pi, tac,
                                veto_on, tac_thr)
        # v15 finishing AFTER guards (soundness first; never overrides a
        # forced tactic or veto replacement).
        if tac is None and not _vet:
            from .selfplay import finish_move as _fin, safety_guard as _safe
            choice, _ = _safe(state, choice, pi, model, device, cfg)
            choice = _fin(state, choice, pi, model, device, cfg)
        # F2: rust path prunes the GameStore; the dict stays empty.
        try:
            _rs = getattr(fn, "_rust_store", None)
            if _rust_tree and use_tt and _rs is not None:
                _rs.prune_to(state.apply(choice).key())
            else:
                mcts_mod.prune_tt(fn._tt, state.apply(choice).key())
        except Exception:
            pass
        return choice
    fn._tt = {}
    fn._hist = []
    # F1/F6: one Rust-side GameStore per agent (cleared on reset alongside
    # the dicts) when the rust-tree seam is on; None = Python path.
    fn._rust_store = None
    if _rust_tree:
        try:
            from .mcts_bridge import new_game_store as _ngs
            fn._rust_store = _ngs()
        except Exception:
            fn._rust_store = None

    def _reset():
        try:
            fn._tt.clear()
        except Exception:
            pass
        try:
            fn._hist.clear()
        except Exception:
            pass
        try:
            _rs = getattr(fn, "_rust_store", None)
            if _rs is not None:
                _rs.clear()
        except Exception:
            pass
    fn.reset = _reset
    fn._rust_tree = bool(_rust_tree)
    # V21.1 E4: _spec stays 24 elements (test_v19c pins len 24). The
    # rust-tree flag travels via cfg pickling (sequential/selfplay) and
    # via the 28th manual-spec element in _gate/_arena parallel rebuilds
    # (evaluate._make_policy reads spec[27], default False).
    fn._spec = ("agent",
                {k: v.cpu() for k, v in model.state_dict().items()},
                cfg.blocks, cfg.channels, n, cfg.c_puct, cfg.dirichlet_alpha,
                noise, temp_moves, cfg.input_planes, use_tt,
                float(getattr(cfg, "contempt", 0.0) or 0.0),
                bool(getattr(cfg, "asymmetric_contempt", False)),
                bool(getattr(cfg, "tactical_override", False)),
                bool(getattr(cfg, "blunder_veto", False)),
                float(getattr(cfg, "tac_threshold", 0.09)),
                int(getattr(cfg, "quiescence_depth", 0) or 0),
                float(getattr(cfg, "forcing_bonus", 0.0) or 0.0),
                float(getattr(cfg, "contempt_edge_scale", 0.0) or 0.0),
                # v15 finishing flag for parallel arena/gate rebuilds.
                bool(getattr(cfg, "mate_finish", False)),
                # v16.1 batched leaves for parallel arena/gate rebuilds.
                int(getattr(cfg, "leaf_batch", 1) or 1),
                float(getattr(cfg, "virtual_loss", 1.0) or 1.0),
                # v15.5 safety veto for parallel arena/gate rebuilds.
                bool(getattr(cfg, "safety_veto", False)),
                # v18 SE trunk for parallel arena/gate rebuilds.
                int(getattr(cfg, "se_ratio", 0) or 0),
                float(getattr(cfg, "fpu_reduction", 0.0)),
                bool(getattr(cfg, "prune_singletons", False)),
                float(getattr(cfg, "safety_drop_thr", 0.15)),
                bool(_rust_tree),
                float(getattr(cfg, "ml_slope", 0.0)),
                float(getattr(cfg, "ml_cap", 0.07)),
                float(getattr(cfg, "ml_thr", 0.8)))
    return fn


def agent_from_weights(w, cfg: Config, sims: int | None = None,
                       device="cpu", noise: float = 0.0,
                       rust_tree: bool = False):
    # v18 arch-follows-weights: the model MUST match the checkpoint (SE
    # keys present -> SE trunk), else random SE distorts old weights or
    # plain blocks drop new ones. cfg only for fresh/ambiguous cases.
    from .model import AlphaZeroNet as _Net, load_weights as _lw, \
        infer_se_ratio as _isr
    try:
        _wd = w if isinstance(w, dict) else torch.load(
            w, map_location="cpu", weights_only=False)
        _se = _isr(_wd)
    except Exception:
        _se = 0
    if not _se:
        _se = int(getattr(cfg, "se_ratio", 0) or 0)
    m = _Net(blocks=cfg.blocks, channels=cfg.channels,
             planes=cfg.input_planes, se_ratio=_se).to(device)
    _lw(m, _wd, strict=True)
    return agent_policy(m, cfg, sims, device, noise=noise,
                        rust_tree=rust_tree)


def game_settings(cfg: Config) -> tuple:
    """Worker-stamped game tuning: every name in game.GAME_GLOBALS.
    Values prefer the cfg's live table (refreshed per iter), falling back
    to the parent module's current table."""
    import chess_zero.game as _g
    live = getattr(cfg, "adjudicate_values", None) or _g.ADJUDICATE_VALUES
    return (cfg.adjudicate_margin, cfg.no_progress_plies,
            cfg.adjudicate_min_ply, cfg.input_planes, dict(live))


def adapt_contempt(contempt: float, draw_share: float,
                   target: float = 0.4) -> float:
    """Lc0-style draw-rate steering: draws above target+0.1 push contempt up
    (more decisive games demanded), below target-0.1 relax it. Bounded
    [0.1, 0.5] so it can never explode or vanish silently."""
    if draw_share > target + 0.1:
        return min(0.5, contempt + 0.05)
    if draw_share < target - 0.1:
        return max(0.1, contempt - 0.02)
    return contempt


def save_checkpoint(model, optimizer, path, meta: dict, sched=None, replay=None, cfg=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"weights": model.state_dict(),
            "opt": optimizer.state_dict(), "meta": meta}
    if cfg is not None:
        from dataclasses import asdict
        ckpt["config"] = asdict(cfg)
    import hashlib
    from pathlib import Path
    root = Path(__file__).parent
    ckpt["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    ckpt["schema_version"] = 2
    ckpt["action_codec"] = "4096-v2-underpromotions"
    ckpt["input_planes"] = model.trunk_in.in_channels
    ckpt["rng"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                   "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        ckpt["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    if replay is not None:
        ckpt["replay"] = {"main": list(replay.buf), "endgame": list(replay.eg)}
    if sched is not None:
        ckpt["sched"] = sched.state_dict()
    atomic_save(ckpt, path)


def atomic_save(obj, path: str) -> str:
    """D16 atomic writes: tmp file + fsync + os.replace (readers never see
    a torn checkpoint, incl. parallel workers reading _w_*.pt mid-write)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    try:
        with open(tmp, "rb") as _f:
            _f.flush()
            os.fsync(_f.fileno())
    except Exception:
        pass
    os.replace(tmp, path)
    return path


def disk_ok(path: str, need_gb: float = 2.0) -> tuple:
    """D16 disk guard: (ok, free_gb). Never raises (stat failure = not-ok
    + loud log at the call site, never a crash)."""
    try:
        _u = shutil.disk_usage(path if os.path.exists(path) else
                               (os.path.dirname(path) or "."))
        _free = float(_u.free) / (1024.0 ** 3)
        return _free >= float(need_gb), round(_free, 2)
    except Exception as _e:
        print(f"[disk] guard unreadable ({type(_e).__name__}); "
              f"treating as FULL (no crash)", flush=True)
        return False, 0.0


def _lr_for(cfg: Config, it: int) -> float:
    # kept for reporting compat; the live schedule is MultiStepLR stepped
    # once per iter (see run_training) — or the v19 game-budget schedule.
    return cfg.lr * (cfg.lr_gamma ** sum(it >= m for m in cfg.lr_milestones))


# ---------------------------------------------------------------------------
# v19 loop helpers (Impl-C). All v19 loop behavior keys off _v19_loop(cfg)
# (mirror_gate, set only by V19_CONFIG); old configs keep yesterday's
# behavior exactly (milestone LR + fixed-threshold gate + no rehearsal,
# cull, or BN recal). New Config knobs all carry yesterday's defaults.
# ---------------------------------------------------------------------------

def _v19_loop(cfg: Config) -> bool:
    return bool(getattr(cfg, "mirror_gate", False))


def _v19_lr_for(cfg: Config, cum_steps: int, cum_games: int) -> float:
    """Game-budget LR (C3): linear warmup to lr_max, const to the drop
    point, lr_min after. Step 0 reads lr_min*0.1 (never a wasted zero
    step — warmup rises from near-zero, Transformer-style)."""
    lr_max = float(getattr(cfg, "lr_max", 3e-4))
    lr_min = float(getattr(cfg, "lr_min", 3e-5))
    warmup = max(1, int(getattr(cfg, "lr_warmup_steps", 1000) or 1000))
    drop = int(getattr(cfg, "lr_drop_games", 7000) or 7000)
    if cum_steps <= 0:
        return lr_min * 0.1
    if cum_steps < warmup:
        return lr_max * float(cum_steps) / float(warmup)
    if cum_games < drop:
        return lr_max
    return lr_min


def _v19_value_w_for(cfg: Config, cum_games: int) -> float:
    """Value-head weight schedule (C3): value_w0 before the drop point,
    value_w1 after."""
    drop = int(getattr(cfg, "lr_drop_games", 7000) or 7000)
    if cum_games < drop:
        return float(getattr(cfg, "value_w0", 0.5))
    return float(getattr(cfg, "value_w1", 1.0))


def _v19_eps_for(cum_games: int) -> float:
    """Dirichlet schedule (C4): 0.25 before 3000 games, 0.12 after."""
    return 0.25 if cum_games < 3000 else 0.12


# ---------------------------------------------------------------------------
# V20 batch-B loop helpers (Agent B). Panel-endgame BEFORE endgame-starts
# (D4 order fixed here: starts are allowed only when the v20 gate path —
# main panel AND endgame_panel — is wired, see endgame_starts_allowed).
# Forbidden one-liners: F1 no TB probing inside MCTS (search stays pure);
# F4 no TB policy boost (value-only rescore; KLD-clash precedent); F5 no
# full no-resign (playthrough is 5%, mate-finish untouched).
# ---------------------------------------------------------------------------

def _v20_loop(cfg: Config) -> bool:
    """V20 master switch (append-only knob cfg.v20; V19 configs read False
    and keep every yesterday behavior below)."""
    return bool(getattr(cfg, "v20", False))


def _v20_lr_for(cfg: Config, cum_steps: int, cum_positions: int) -> float:
    """Position-counted LR (D9/R2: game-counted LR replaced — shorter
    endgame-start games must not slow decay). Same shape as C3: linear
    warmup to lr_max, const to the drop point, lr_min after; the drop
    point is lr_drop_positions (V20: 560000 ~= 7000 games x ~80 plies,
    milestones kept equivalent). lr_drop_positions <= 0 falls back to
    the game-budget rule (never silently zero)."""
    lr_max = float(getattr(cfg, "lr_max", 3e-4))
    lr_min = float(getattr(cfg, "lr_min", 3e-5))
    warmup = max(1, int(getattr(cfg, "lr_warmup_steps", 1000) or 1000))
    drop = int(getattr(cfg, "lr_drop_positions", 0) or 0)
    if drop <= 0:
        try:
            _g = int(getattr(cfg, "lr_drop_games", 7000) or 7000)
        except Exception:
            _g = 7000
        drop = _g * 80
    if cum_steps <= 0:
        return lr_min * 0.1
    if cum_steps < warmup:
        return lr_max * float(cum_steps) / float(warmup)
    if cum_positions < drop:
        return lr_max
    return lr_min


def _roll_endgame(rng, frac: float) -> bool:
    """D4 endgame-start coin (uniform, testable). frac <= 0 never fires."""
    try:
        return float(rng.random()) < float(frac)
    except Exception:
        return False


def _roll_playthrough(rng, frac: float) -> bool:
    """D5 playthrough coin (5% games ignore resign entirely)."""
    try:
        return float(rng.random()) < float(frac)
    except Exception:
        return False


def endgame_starts_allowed(cfg: Config) -> bool:
    """D4 order gate: starts are enabled only on the v20 path (where the
    promotion rule is main-panel AND endgame_panel — panel first, then
    starts). Non-v20 or frac<=0 -> False (yesterday, no starts)."""
    try:
        return _v20_loop(cfg) and float(
            getattr(cfg, "endgame_frac", 0.0) or 0.0) > 0.0
    except Exception:
        return False


def _v20_rescore_game(cfg: Config, tb_mod, tb_handle, ex: list,
                      st: dict, res: str) -> dict:
    """V20 D6+D8 post-game pass for ONE self-play game (shared by the
    sequential loop and parallel workers): rewrite tuple z/ml in place
    from the TB probe where legal, then deblunder-walk. Returns counts
    {"probed","guarded","rewritten","deblundered","skipped",
    "skipped_book"}. No tables (None handle) or non-v20 -> all-skipped
    (never raises, never fabricates). Book-opening games skip loudly
    (their unrecorded plies break move-log alignment)."""
    _zero = {"probed": 0, "guarded": 0, "rewritten": 0, "deblundered": 0,
             "skipped": len(ex or []), "skipped_book": 0}
    try:
        if not _v20_loop(cfg) or tb_handle is None or not ex:
            return _zero
        if bool(st.get("book_opening", False)):
            _zero["skipped_book"] = 1
            return _zero
        _moves = st.get("moves_uci") or []
        _plys = st.get("record_plys") or []
        if not _moves or len(_plys) != len(ex):
            return _zero
        try:
            _ow = {"1-0": 1.0, "0-1": -1.0}.get(res, 0.0)
        except Exception:
            _ow = 0.0
        _s = tb_mod.rescore_examples(
            ex, _plys, _moves, tb_handle,
            int(getattr(cfg, "move_cap", 300) or 300),
            int(getattr(cfg, "tb_ml_cap", 200) or 200),
            st.get("start_fen"), _ow,
            float(getattr(cfg, "deblunder_thr", 0.1) or 0.1))
        _s["skipped_book"] = 0
        return _s
    except Exception:
        return _zero


def _build_adamw(model, cfg):
    """AdamW with two groups (C2): decay (conv/FC weights, wd=cfg.l2) and
    no-decay (all 1-dim params: biases, BN, SE). Empty groups dropped."""
    decay, no_decay = [], []
    for _n, _p in model.named_parameters():
        if _p.dim() <= 1:
            no_decay.append(_p)
        else:
            decay.append(_p)
    _groups = []
    if decay:
        _groups.append({"params": decay, "lr": cfg.lr,
                        "weight_decay": cfg.l2})
    if no_decay:
        _groups.append({"params": no_decay, "lr": cfg.lr,
                        "weight_decay": 0.0})
    for group in _groups:
        names = {id(p): name for name, p in model.named_parameters()}
        group["param_names"] = [names[id(p)] for p in group["params"]]
    return torch.optim.AdamW(_groups)


def _sprt_bounds(cfg):
    """Wald bounds from the cfg error rates: LA=log(b/(1-a)),
    LB=log((1-b)/a). With alpha=beta=0.10: -/+log(9) = -/+2.197."""
    a = float(getattr(cfg, "sprt_alpha", 0.10))
    b = float(getattr(cfg, "sprt_beta", 0.10))
    a = min(max(a, 1e-9), 1 - 1e-9)
    b = min(max(b, 1e-9), 1 - 1e-9)
    return math.log(b / (1 - a)), math.log((1 - b) / a)


def gsprt_llr(W, L, D, elo0=0.0, elo1=30.0) -> float:
    """Trinomial SPRT log-likelihood ratio, draws-split binomial (the
    textbook draws-aware SPRT used across engines): score hypotheses
    s0/s1 from the elo pair, W' = W + D/2, L' = L + D/2,
    LLR = W'*ln(s1/s0) + L'*ln((1-s1)/(1-s0)).
    Antisymmetric under W<->L; ~0 for even scores; 0.0 for no games.
    Scale note (honest): elo(0,30) hypotheses sit ~4pp apart, so a capped
    20-game match reaches |LLR| ~ 1 at most — real bound crossings need
    blowouts; the truncated fallback in _gate_sprt_decide does the routine
    work (by design, cf. C6)."""
    try:
        s0 = 1.0 / (1.0 + 10.0 ** (-float(elo0) / 400.0))
        s1 = 1.0 / (1.0 + 10.0 ** (-float(elo1) / 400.0))
    except Exception:
        return 0.0
    s0 = min(max(s0, 1e-9), 1 - 1e-9)
    s1 = min(max(s1, 1e-9), 1 - 1e-9)
    Wp = float(W) + 0.5 * float(D)
    Lp = float(L) + 0.5 * float(D)
    return Wp * math.log(s1 / s0) + Lp * math.log((1 - s1) / (1 - s0))


def _gate_sprt_decide(W, L, D, cfg, games_played, cap):
    """One SPRT accounting step (C6): running LLR vs LA/LB from the cfg
    error rates; truncated fallback (LLR>0 AND score>=11/20) once the cap
    is reached; otherwise continue. Returns
    {"llr","promote","stop","via"[, "score"]}."""
    LA, LB = _sprt_bounds(cfg)
    llr = gsprt_llr(W, L, D, float(getattr(cfg, "sprt_elo0", 0.0)),
                    float(getattr(cfg, "sprt_elo1", 30.0)))
    n = max(1, int(games_played))
    score = (float(W) + 0.5 * float(D)) / n
    if llr >= LB:
        return {"llr": llr, "promote": True, "stop": True,
                "via": "sprt-upper", "score": score}
    if llr <= LA:
        return {"llr": llr, "promote": False, "stop": True,
                "via": "sprt-lower", "score": score}
    if int(games_played) >= int(cap):
        if llr > 0 and score >= 11.0 / 20.0:
            return {"llr": llr, "promote": True, "stop": True,
                    "via": "truncated", "score": score}
        return {"llr": llr, "promote": False, "stop": True,
                "via": "truncated-hold", "score": score}
    return {"llr": llr, "promote": False, "stop": False,
            "via": "continue", "score": score}


# ---------------------------------------------------------------------------
# V21 cumulative LLR (OR-path, future-only, both legs). Bound 2.94 =
# log((1-beta)/alpha) with alpha=beta=0.05 (Fishtest convention; Wald
# SPRT upper bound). Main leg reuses the ALREADY-COMPUTED panel llr
# (_panel_decide "llr" via gsprt_llr); endgame leg uses eg_panel_llr
# below. Per-iter legacy rules are untouched; cum only ADDS an OR.
# ---------------------------------------------------------------------------

def eg_panel_llr(W, L, D, p0: float = 0.5, p1: float = 0.6) -> float:
    """Endgame WLD LLR, H0 score 50% vs H1 score 60%, trinomial approx.
    Share observed draws: d=D/n, pW0=pL0=(1-d)/2, pW1=p1-0.5*d, pL1=(1-p1)-0.5*d.
    LLR = W*ln(pW1/pW0) + L*ln(pL1/pL0) (draws cancel); 0.0 for n=0/degenerate."""
    try:
        _n = float(W) + float(L) + float(D)
        if _n <= 0:
            return 0.0
        _d = float(D) / _n
        _p0 = min(max(float(p0), 1e-9), 1 - 1e-9)
        _p1 = min(max(float(p1), 1e-9), 1 - 1e-9)
        _pw0 = max((1.0 - _d) / 2.0, 1e-9)
        _pl0 = max((1.0 - _d) / 2.0, 1e-9)
        _pw1 = max(float(_p1) - 0.5 * _d, 1e-9)
        _pl1 = max((1.0 - float(_p1)) - 0.5 * _d, 1e-9)
        return float(W) * math.log(_pw1 / _pw0) + float(L) * math.log(
            _pl1 / _pl0)
    except Exception:
        return 0.0


def _v21_llr_bound(cfg, key: str = "LLR_BOUND",
                   default: float = 2.94) -> float:
    """V21 bound read via getattr with default (old configs lack the knob).
    LLR_BOUND / EG_LLR_BOUND, default 2.94."""
    try:
        return float(getattr(cfg, key, default) or default)
    except Exception:
        return float(default)


def _v21_cum_add(cum: float, panel_llr) -> float:
    """V21 accumulate: cum + panel_llr (floats; None -> 0.0, never raises)."""
    try:
        return float(cum or 0.0) + float(panel_llr or 0.0)
    except Exception:
        try:
            return float(cum or 0.0)
        except Exception:
            return 0.0


def _v21_leg_pass(per_iter: bool, cum: float, bound: float) -> bool:
    """V21 OR-leg: per-iter rule OR (cum >= bound). Never raises."""
    # Every iteration is a different challenger. Historical evidence is
    # diagnostic only and must never bypass this candidate's gates.
    return bool(per_iter)


def _v21_gate_cum_decide(gate: dict, cum_main: float, cum_eg: float,
                         cfg=None) -> dict:
    """V21 pure OR-path decision from an ALREADY-COMPUTED gate dict plus
    UPDATED cums (caller already did cum += panel llr). Returns
    {"main_per","eg_per","main_leg","eg_leg","sprt_pass_cum"}. Legacy
    gate fields untouched; no-engines, never raises. Single-leg gates
    (no endgame_panel) treat eg_leg as True (vacuous, yesterday)."""
    if (gate or {}).get("via") in ("panel", "panel+endgame"):
        main = bool(gate.get("main_sprt_pass", gate.get("sprt_pass", False)))
        eg = bool((gate.get("endgame_panel") or {}).get("pass", True))
        return {"main_per": main, "eg_per": eg, "main_leg": main,
                "eg_leg": eg, "sprt_pass_cum": main and eg}
    try:
        _bmain = _v21_llr_bound(cfg, "LLR_BOUND", 2.94)
        _beg = _v21_llr_bound(cfg, "EG_LLR_BOUND", 2.94)
        _g = gate or {}
        _eg = _g.get("endgame_panel")
        if isinstance(_eg, dict) and "pass" in _eg:
            _mper = bool(_g.get("main_sprt_pass", _g.get("sprt_pass",
                                                       False)))
            _eper = bool(_eg.get("pass", False))
            _mleg = _v21_leg_pass(_mper, cum_main, _bmain)
            _eleg = _v21_leg_pass(_eper, cum_eg, _beg)
            return {"main_per": _mper, "eg_per": _eper,
                    "main_leg": bool(_mleg), "eg_leg": bool(_eleg),
                    "sprt_pass_cum": bool(_mleg and _eleg)}
        if bool(getattr(cfg, "mirror_gate", False)) if cfg is not None \
                else ("sprt_pass" in _g):
            _mper = bool(_g.get("sprt_pass", False))
        else:
            try:
                _thr = float(getattr(cfg, "gate_threshold", 0.55)
                             if cfg is not None else 0.55)
            except Exception:
                _thr = 0.55
            try:
                _mper = bool(float(_g.get("score", 0.0)) >= _thr)
            except Exception:
                _mper = bool(_g.get("sprt_pass", False))
        _mleg = _v21_leg_pass(_mper, cum_main, _bmain)
        return {"main_per": bool(_mper), "eg_per": True,
                "main_leg": bool(_mleg), "eg_leg": True,
                "sprt_pass_cum": bool(_mleg)}
    except Exception:
        try:
            return {"main_per": bool((gate or {}).get("sprt_pass",
                                                     False)),
                    "eg_per": True, "main_leg": bool(
                        (gate or {}).get("sprt_pass", False)),
                    "eg_leg": True,
                    "sprt_pass_cum": bool((gate or {}).get("sprt_pass",
                                                          False))}
        except Exception:
            return {"main_per": False, "eg_per": True,
                    "main_leg": False, "eg_leg": True,
                    "sprt_pass_cum": False}


def split_book_lines(lines: list, n_train: int = 60) -> tuple:
    """D11 empirical-book split — canonical implementation lives in
    book.split_book_lines (single home); this delegates so both callers
    share one rule."""
    from .book import split_book_lines as _split
    return _split(lines, n_train)


def _gate_book_draw(seed_base, cfg=None):
    """Reshuffle the empirical book's HOLDOUT split per gate (C7+D11).
    Returns (lines10, info): up to 10 holdout lines for the mirrored
    pairs (train split serves self-play forced-book + lineage). The drawn
    lines seed the per-pair seed_bases (observable, varying per gate)
    while the games use the paired random-plies path (2k/2k+1 share one
    seeded opening with colors swapped = mirrored). Book missing ->
    seed-only pairing + loud log. Return shape frozen (test_v19c)."""
    info = {"book": None, "n_lines": 0, "draw": [], "fallback": None}
    try:
        from .book import load_book as _lb
        _bp = os.path.join(os.path.dirname(__file__),
                           "book_empirical.json")
        _custom = getattr(cfg, "gate_book_file", "") if cfg is not None else ""
        if _custom: _bp = _custom
        _lines = _lb(_bp)
        info["n_lines"] = len(_lines)
        if not _lines:
            raise ValueError("empty book")
        _train, _hold = ([], _lines) if _custom else split_book_lines(_lines)
        info["n_train"] = len(_train)
        info["n_holdout"] = len(_hold)
        _pool = _hold if _hold else _train
        _rng = random.Random(int(seed_base))
        _idx = list(range(len(_pool)))
        _rng.shuffle(_idx)
        _limit = max(1, int(getattr(cfg, "gate_incumbent_games", 64)) // 2) if cfg is not None else 32
        _sel = [list(_pool[i]) for i in _idx[:_limit]]
        info["book"] = _bp
        info["draw"] = [len(_l) for _l in _sel]
        info["split"] = "holdout"
        return _sel, info
    except Exception as _e:
        info["fallback"] = f"paired random plies (book unreadable: " \
            f"{type(_e).__name__})"
        return [], info


def _pair_seed(seed_base, pair_idx, line) -> int:
    """Per-pair seed: gate base + pair stride + drawn-line hash (crc32 for
    cross-process stability; python hash() is salted)."""
    import zlib as _z
    try:
        _h = _z.crc32(repr(line).encode()) if line else 0
    except Exception:
        _h = 0
    return (int(seed_base) + int(pair_idx) * 7919 + int(_h)) % (2 ** 31 - 1)


def _maybe_aux_cull(cfg, cum_games, latched):
    """Late cull (C5): past 8000 games mob/safety/margin/check -> 0.005,
    one-way latch. Returns (fired_now, latched). safe_w is the wired knob
    (train_step kwarg); safety_w is set as a spec-literal alias."""
    if latched or cum_games <= 8000:
        return False, latched
    for _k in ("mob_w", "safe_w", "margin_w", "check_w"):
        try:
            setattr(cfg, _k, 0.005)
        except Exception:
            pass
    return True, True


def _load_teacher(cfg, device="cpu"):
    """KL-anchor teacher (C8): warmstart ckpt on the loop arch, frozen +
    eval (no_grad at every forward). Never raises: None + loud log."""
    try:
        _path = str(getattr(cfg, "teacher_path",
                            "checkpoints_warm/warmstart.pt"))
        if not os.path.exists(_path):
            print(f"[v19] KL anchor OFF: teacher file missing ({_path})",
                  flush=True)
            return None
        from .model import AlphaZeroNet as _Net, load_weights as _lw, \
            infer_se_ratio as _isr
        _w = torch.load(_path, map_location="cpu", weights_only=False)
        _se = 0
        try:
            _se = _isr(_w)
        except Exception:
            pass
        if not _se:
            _se = int(getattr(cfg, "se_ratio", 0) or 0)
        _t = _Net(blocks=cfg.blocks, channels=cfg.channels,
                  planes=cfg.input_planes, se_ratio=_se).to(device)
        _lw(_t, _w, strict=False)  # Teacher KL uses the unchanged policy path.
        _t.eval()
        for _p in _t.parameters():
            _p.requires_grad_(False)
        print(f"[v19] KL anchor teacher loaded ({_path})", flush=True)
        return _t
    except Exception as _e:
        print(f"[v19] KL anchor OFF: teacher load failed "
              f"({type(_e).__name__}: {str(_e)[:120]})", flush=True)
        return None


def _sample_rehearsal_pool_with_ids(cfg, n_games, seed):
    """V20.2 R1: sample + encode rehearsal rows WITH (game_id, ply) ids.

    Returns (rows, ids) where rows are 13-tuples IDENTICAL to
    _sample_rehearsal_pool and ids is a parallel list of (game_id, ply)
    (ply 0-based, same as build_av.iter_positions). Prefers the new
    warmstart with_ids helpers (share code); falls back to the legacy
    helpers with ids (None, ply-index) -> AV miss -> zeros (yesterday).
    Guards the warmstart import (loud skip, never raises)."""
    try:
        from . import warmstart as _ws
    except Exception as _e:
        print(f"[v19] rehearsal OFF: warmstart helpers missing "
              f"({type(_e).__name__})", flush=True)
        return [], []
    try:
        _path = str(getattr(cfg, "rehearsal_games", "data_sl/games.jsonl"))
        # Preferred: with_ids helpers (share code with the legacy path).
        _sr_ids = getattr(_ws, "sample_rehearsal_with_ids", None)
        _eb_ids = getattr(_ws, "encode_rehearsal_batch_with_ids", None)
        if callable(_sr_ids) and callable(_eb_ids):
            _recs = _sr_ids(_path, int(n_games), int(seed))
            if not _recs:
                print(f"[v19] rehearsal OFF: no records from {_path}",
                      flush=True)
                return [], []
            _rows, _ids = _eb_ids(_recs)
            print(f"[v19] rehearsal pool: {len(_rows)} rows from "
                  f"{len(_recs)} games ({_path})", flush=True)
            return _rows, _ids
        # Fallback: legacy helpers (no ids -> miss -> zeros, yesterday).
        _sr, _eb = _ws.sample_rehearsal, _ws.encode_rehearsal_batch
        _recs = _sr(_path, int(n_games), int(seed))
        if not _recs:
            print(f"[v19] rehearsal OFF: no records from {_path}",
                  flush=True)
            return [], []
        _rows = _eb(_recs)
        print(f"[v19] rehearsal pool: {len(_rows)} rows from "
              f"{len(_recs)} games ({_path})", flush=True)
        return list(_rows), [(None, int(_i)) for _i in range(len(_rows))]
    except Exception as _e:
        print(f"[v19] rehearsal OFF: pool build failed "
              f"({type(_e).__name__}: {str(_e)[:120]})", flush=True)
        return [], []


def _decode_av_cp_vector(av_table, game_id, ply):
    """V20.2 R3: AV-table lookup -> 4096 float32 centipawn vector.

    Hit: int8 q8 {uci: q8} dequantized (build_av.dequantize_q8, x12 cp)
    mapped through the stm-oriented 4096 codec (shared mirroring via
    warmstart.av_handle_from_row; duplicate-if-safer documented here:
    warmstart emits pawn handles/masks for SL batches, the loop tail needs
    replay-scale cp vectors, so the final vec build is loop-owned while the
    mirroring + scales are shared). Illegal moves stay 0. Miss/empty ->
    zeros(4096) (masked to zero loss in train.py: genuine SF vectors never
    have zero spread). Never raises (miss on any error)."""
    import numpy as _np
    try:
        if not av_table or game_id is None:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        try:
            _key = (str(game_id), int(ply))
        except Exception:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        _raw = av_table.get(_key)
        if not _raw:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        _avd = _raw.get("av") if isinstance(_raw, dict) else None
        if not _avd:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        _stm = _raw.get("stm", "w") if isinstance(_raw, dict) else "w"
        try:
            from .warmstart import av_handle_from_row as _hfr
            from .build_av import dequantize_q8 as _dq
        except Exception:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        try:
            _h = _hfr(_avd, _stm)
        except Exception:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        if not _h:
            return _np.full(4096, _np.nan, dtype=_np.float32)
        _v = _np.full(4096, _np.nan, dtype=_np.float32)
        for _a, _q in _h.items():
            try:
                _ai = int(_a)
                if 0 <= _ai < 4096:
                    _v[_ai] = float(_dq(int(_q)))
            except Exception:
                continue
        return _v
    except Exception:
        try:
            import numpy as _np2
            return _np2.zeros(4096, dtype=_np2.float32)
        except Exception:
            return None


def _sample_rehearsal_pool(cfg, n_games, seed):
    """Sample + encode rehearsal rows (C8) into buffer-order 11-tuples
    (s,pi,z,m,ml,own,margin,mob,safe,reply,pw). Guards the warmstart import
    (helpers verified present with exact signatures; skips loudly if the
    other implementer's half is ever absent)."""
    try:
        from . import warmstart as _ws
        _sr, _eb = _ws.sample_rehearsal, _ws.encode_rehearsal_batch
    except Exception as _e:
        print(f"[v19] rehearsal OFF: warmstart helpers missing "
              f"({type(_e).__name__})", flush=True)
        return []
    try:
        _path = str(getattr(cfg, "rehearsal_games", "data_sl/games.jsonl"))
        _recs = _sr(_path, int(n_games), int(seed))
        if not _recs:
            print(f"[v19] rehearsal OFF: no records from {_path}",
                  flush=True)
            return []
        _rows = _eb(_recs)
        print(f"[v19] rehearsal pool: {len(_rows)} rows from "
              f"{len(_recs)} games ({_path})", flush=True)
        return _rows
    except Exception as _e:
        print(f"[v19] rehearsal OFF: pool build failed "
              f"({type(_e).__name__}: {str(_e)[:120]})", flush=True)
        return []


def _precise_bn_lite(model, buf, batch=128, passes=4):
    """Refresh BN from representative replay, rolling back on failure."""
    if len(buf) < int(batch):
        return False
    layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [(m.momentum, m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone()) for m in layers]
    training = model.training
    ok = False
    try:
        dev = next(model.parameters()).device
        model.train()
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
        with torch.no_grad():
            for _ in range(max(16, int(passes))):
                states, *_ = buf.sample(int(batch))
                model(torch.from_numpy(np.asarray(states, dtype=np.float32)).to(dev))
        ok = True
        return True
    except Exception as exc:
        print(f"[BN] refresh failed; previous statistics restored: {exc}", flush=True)
        return False
    finally:
        for layer, (momentum, mean, var, count) in zip(layers, saved):
            layer.momentum = momentum
            if not ok:
                layer.running_mean.copy_(mean)
                layer.running_var.copy_(var)
                layer.num_batches_tracked.copy_(count)
        model.train(training)


def _ratio_steps(total_plies, batch_size, train_steps, sample_reuse=1.5) -> int:
    """Sampling-ratio target (C10): clamp(round(plies*1.5/batch),1,steps).
    total_plies approximates recorded positions (results["plies"]
    overestimates slightly by counting unrecorded opening plies)."""
    try:
        _r = int(round(float(total_plies) * float(sample_reuse) / max(1, int(batch_size))))
    except Exception:
        return int(train_steps)
    return max(1, min(int(train_steps), _r))


def opening_top3_share(prefixes) -> dict:
    """D10 opening-entropy telemetry: share of the top-3 2-ply prefixes.
    prefixes: iterable of 2-tuples (hashable). Returns {"top3": share,
    "n": count}. Empty -> share 0.0 (never raises)."""
    try:
        from collections import Counter as _C
        _ps = [tuple(p) for p in (prefixes or [])]
        if not _ps:
            return {"top3": 0.0, "n": 0}
        _c = _C(_ps)
        _top = sum(v for _, v in _c.most_common(3))
        return {"top3": round(_top / len(_ps), 4), "n": len(_ps)}
    except Exception:
        return {"top3": 0.0, "n": 0}


def _anti_rps_check(chall_w, ckpt_dir, cfg, sims, device, seed_base=0):
    """D10 anti-RPS veto: 4 games (mirrored pair ids) vs the promo-3
    ancestor snapshot at min(100, cfg.sims) sims, noise 0. Returns
    {"score", "wld", "hold", "note"}. Ancestor missing -> fallback
    incumbent; total failure -> {"hold": False} + loud log (a broken
    measurement must never veto a promotion by itself)."""
    from .evaluate import play_match
    _sims = max(1, min(100, int(sims or 100)))
    try:
        _anc_w, _note = _ancestor_weights(ckpt_dir, chall_w)
        ch = agent_from_weights(_weights_dict(chall_w), cfg, sims=_sims,
                                device=device, noise=0.0)
        anc = agent_from_weights(_weights_dict(_anc_w), cfg, sims=_sims,
                                 device=device, noise=0.0)
        _tot = play_match(ch, anc, games=4,
                          opening_moves=int(getattr(
                              cfg, "gate_opening_moves", 6) or 0),
                          seed_base=int(seed_base))
        _n = _tot["wins"] + _tot["losses"] + _tot["draws"]
        _sc = ((_tot["wins"] + 0.5 * _tot["draws"]) / _n) if _n else 0.0
        _hold = bool(_sc < 0.40)
        print(f"[anti-rps] 4 games vs {_note} @ {_sims} sims: "
              f"WLD=({_tot['wins']},{_tot['losses']},{_tot['draws']}) "
              f"score={_sc:.3f} -> {'HOLD' if _hold else 'pass'}",
              flush=True)
        return {"score": round(_sc, 4),
                "wld": [_tot["wins"], _tot["losses"], _tot["draws"]],
                "hold": _hold, "note": _note, "sims": _sims}
    except Exception as _e:
        print(f"[anti-rps] check failed ({type(_e).__name__}: "
              f"{str(_e)[:160]}); not vetoing", flush=True)
        return {"score": None, "wld": [0, 0, 0], "hold": False,
                "note": f"error-{type(_e).__name__}", "sims": _sims}


def game_id_is_test(game_id, mod: int = 20) -> bool:
    """D15 test split by game hash: test iff crc32(str(id)) % mod == 0
    (~5% at mod 20). crc32, never python hash() (salted per process)."""
    try:
        import zlib as _z
        return (_z.crc32(str(game_id).encode()) % int(mod)) == 0
    except Exception:
        return False


def _test_losses(model, buf, cfg, device="cpu"):
    """D15 test-loss report: one no-grad batch from buf.sample_test(n)
    (replay-owned; absent -> loud skip, None). Returns
    {"test_loss","test_loss_p","test_loss_v"} with None values when
    unavailable. Never raises."""
    _none = {"test_loss": None, "test_loss_p": None, "test_loss_v": None}
    try:
        _st = getattr(buf, "sample_test", None)
        if not callable(_st):
            print("[test] replay has no sample_test (test_buf is "
                  "Impl-B's file); test losses skipped loudly", flush=True)
            return _none
        _batch = _st(int(getattr(cfg, "batch_size", 256) or 256))
        if _batch is None:
            print("[test] sample_test returned None (empty test_buf?)",
                  flush=True)
            return _none
        from .train import compute_loss as _cl
        import torch as _t
        _dev = next(model.parameters()).device
        _was = bool(model.training)
        model.eval()
        try:
            with _t.no_grad():
                # Index (not unpack): replay-owned sample_test may return
                # 11 arrays (yesterday) or 12 (+ ml_mask, D28).
                s, pi, z, m, ml = _batch[0], _batch[1], _batch[2], \
                    _batch[3], _batch[4]
                own, margin, mob, safe = _batch[5], _batch[6], \
                    _batch[7], _batch[8]
                reply, pw = _batch[9], _batch[10]
                _out = _cl(model, _t.from_numpy(s).to(_dev),
                           _t.from_numpy(pi).to(_dev),
                           _t.from_numpy(z).to(_dev),
                           _t.from_numpy(m).to(_dev),
                           _t.from_numpy(ml).to(_dev),
                           _t.from_numpy(own).to(_dev),
                           _t.from_numpy(margin).to(_dev),
                           _t.from_numpy(mob).to(_dev),
                           _t.from_numpy(safe).to(_dev),
                           aux_w=float(getattr(cfg, "aux_w", 0.1)),
                           ml_w=float(getattr(cfg, "ml_w", 0.05)),
                           ent_w=float(getattr(cfg, "ent_w", 0.0)),
                           own_w=float(getattr(cfg, "own_w", 0.1)),
                           margin_w=float(getattr(cfg, "margin_w", 0.05)),
                           mob_w=float(getattr(cfg, "mob_w", 0.05)),
                           safe_w=float(getattr(cfg, "safe_w", 0.05)),
                           reply_w=float(getattr(cfg, "reply_w", 0.05)),
                           target_reply=_t.from_numpy(reply).to(_dev),
                           policy_w=_t.from_numpy(pw).to(_dev),
                           value_w=1.0,
                           smooth_eps=float(getattr(cfg, "smooth_eps",
                                                    0.0)),
                           soft_w=float(getattr(cfg, "soft_w", 0.3)),
                           check_w=float(getattr(cfg, "check_w", 0.01)))
        finally:
            if _was:
                model.train()
        _tot, _pl, _vl = (float(_out[0]), float(_out[1]), float(_out[2]))
        print(f"[test] test losses: total={_tot:.4f} policy={_pl:.4f} "
              f"value={_vl:.4f}", flush=True)
        return {"test_loss": round(_tot, 4),
                "test_loss_p": round(_pl, 4),
                "test_loss_v": round(_vl, 4)}
    except Exception as _e:
        print(f"[test] test-loss report failed "
              f"({type(_e).__name__}: {str(_e)[:120]})", flush=True)
        return _none


def bn_plane_report(model, prev: dict | None = None) -> dict:
    """D18 BN/plane alarms: mean|running_mean| of the first/last
    BatchNorm2d + per-input-plane weight-norm top-3 ratio (max/mean over
    trunk_in planes). Alarms: ratio > 5, or drift > 10% vs prev
    (prev = this dict from the prior iter). Never raises."""
    rep = {"bn_first": None, "bn_last": None, "plane_top3_ratio": None,
           "alarms": []}
    try:
        import torch.nn as _nn
        _bns = [_m for _m in model.modules()
                if isinstance(_m, _nn.BatchNorm2d)]
        if _bns:
            try:
                rep["bn_first"] = round(float(abs(
                    _bns[0].running_mean.detach().cpu()).mean()), 6)
                rep["bn_last"] = round(float(abs(
                    _bns[-1].running_mean.detach().cpu()).mean()), 6)
            except Exception:
                pass
        try:
            _w = None
            for _name, _p in model.named_parameters():
                if "trunk_in" in _name and _p.dim() == 4:
                    _w = _p.detach().cpu().float()
                    break
            if _w is None:  # fallback: first 4-dim conv weight
                for _name, _p in model.named_parameters():
                    if _p.dim() == 4:
                        _w = _p.detach().cpu().float()
                        break
            if _w is not None:
                _pn = _w.pow(2).sum(dim=(0, 2, 3)).sqrt()
                _mean = float(_pn.mean())
                if _mean > 0:
                    _top3 = float(_pn.topk(min(3, _pn.numel())
                                           ).values.mean())
                    rep["plane_top3_ratio"] = round(_top3 / _mean, 3)
        except Exception:
            pass
        if rep["plane_top3_ratio"] is not None and \
                rep["plane_top3_ratio"] > 5.0:
            rep["alarms"].append(
                f"plane-top3-ratio {rep['plane_top3_ratio']} > 5")
        if prev:
            for _k in ("bn_first", "bn_last"):
                try:
                    _a, _b = rep[_k], prev.get(_k)
                    if _a is not None and _b:
                        _drift = abs(_a - _b) / max(abs(_b), 1e-9)
                        if _drift > 0.10:
                            rep["alarms"].append(
                                f"BN drift {_k} {_drift:.1%} > 10%")
                except Exception:
                    pass
        for _a in rep["alarms"]:
            print(f"[alarm] D18 {_a}", flush=True)
        return rep
    except Exception as _e:
        print(f"[alarm] BN/plane report failed "
              f"({type(_e).__name__})", flush=True)
        return rep


def probe_report(model, device="cpu", fens: list | None = None) -> list:
    """D19 probe harness: 5 fixed FENs (white/black promotions,
    mate-in-1, hanging piece, startpos). Per FEN: renormalized legal
    policy mass, top-3 UCI, queen-promo prior, brute-force mate moves +
    whether the mate sits in the top-3. Trained-net thresholds (>80%
    mass concentration, promo >10%, mate top-3) need TRAINED weights;
    with a random net this harness verifies measurement integrity
    (mass sums to 1, top-3 legal, promo prior finite, mate detection
    agrees with brute force). Never raises (per-FEN error dicts)."""
    import chess as _c
    _FENS = fens or [
        "8/2P5/8/8/8/1k6/8/4K3 w - - 0 1",      # white promotion
        "4k3/8/8/8/8/1K6/2p5/8 b - - 0 1",      # black promotion
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR "
        "w KQkq - 0 1",                          # mate-in-1 (Qxf7#)
        "4k3/8/8/8/1n6/8/5PPP/4K3 w - - 0 1",   # hanging knight (b4)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ]
    from .game import State as _S, flip_action as _flip
    out = []
    for _fen in _FENS:
        try:
            _board = _c.Board(_fen)
            _st = _S(_board.copy(stack=False))
            _legal = _st.legal_moves()
            _b = torch.from_numpy(
                np.ascontiguousarray(_st.encode()[None])).to(device)
            _was = bool(model.training)
            model.eval()
            try:
                with torch.no_grad():
                    _o = model(_b)
                    _logits = _o[0].float().cpu().numpy()[0]
            finally:
                if _was:
                    model.train()
            _lp = _logits - _logits.max()
            _pr = np.exp(_lp)
            _mask = np.zeros_like(_pr)
            _mask[_legal] = 1.0
            _mass = float((_pr * _mask).sum())
            if _mass > 0:
                _pr = _pr * _mask / _mass
            _order = [int(a) for a in np.argsort(-_pr)
                      if a in set(_legal)][:3]
            _top3 = []
            for _a in _order:
                try:
                    _top3.append(_st.to_uci(_a))
                except Exception:
                    _top3.append(str(_a))
            # queen-promo prior: stm pawn moves to the last rank.
            _promo = 0.0
            try:
                for _a in _legal:
                    _mv = _st.to_move(_a)
                    _pc = _board.piece_at(_mv.from_square)
                    if _pc is not None and \
                            _pc.piece_type == _c.PAWN and \
                            _c.square_rank(_mv.to_square) in (0, 7):
                        _promo += float(_pr[_a])
            except Exception:
                pass
            # brute-force mate-in-1 (absolute, via board).
            _mates = set()
            try:
                for _mv in list(_board.legal_moves):
                    _board.push(_mv)
                    try:
                        if _board.is_checkmate():
                            _mates.add(_mv.uci())
                    finally:
                        _board.pop()
            except Exception:
                pass
            _mate_top3 = bool(_mates and any(
                str(_m)[:4] in [str(_t)[:4] for _t in _top3]
                for _m in _mates))
            out.append({"fen": _fen, "legal_mass": round(_mass, 6) if
                        _mass > 0 else 0.0,
                        "mass_renorm": 1.0 if _mass > 0 else 0.0,
                        "top3": _top3, "promo_prior": round(_promo, 4),
                        "mates": sorted(_mates),
                        "mate_in_top3": _mate_top3})
        except Exception as _e:
            out.append({"fen": _fen,
                        "error": f"{type(_e).__name__}: {str(_e)[:120]}"})
    return out


def _evaluation_schedule(cfg, iteration, final_iteration):
    """Use absolute iteration numbers so resume preserves evaluation cadence."""
    gate_every = max(1, int(getattr(cfg, "gate_every", 1)))
    weak_every = max(1, int(getattr(cfg, "weak_diagnostics_every", 1)))
    final = iteration == final_iteration
    return (final or iteration % gate_every == 0,
            final or iteration % weak_every == 0)


def run_training(cfg: Config, games_per_iter=8, iters=2, train_steps=60,
                 arena_games=6, arena_sims=12, device="cpu",
                 ckpt_dir="checkpoints", log_name="history.json", workers=1,
                 sf_games=0, sf_rung="sf-elo1350", sf_sims=None,
                 sf_path="/usr/games/stockfish", resume=None, start_iter=1,
                 final_ckpt="best.pt", use_server=False,
                 server_device="cuda", rust_tree: bool = False):
    """use_server (v8.8+): batched GPU inference for self-play/arena/gate
    workers via chess_zero.infer_server (one server per phase, built from
    that phase's weights file — staleness impossible by construction, no
    RELOAD needed). Default False = today's behaviour exactly.
    rust_tree (V21.1 E4, flag only, default OFF): when True, arena/gate
    agents and self-play search route through mcts_bridge.RustTreeBackend
    with identical params (stub until batch C lands mctscore). False =
    yesterday bit-exact (mcts.search). Stamped onto cfg.rust_tree so
    spawn workers (cfg pickling) see it; explicit param wins over cfg."""
    os.makedirs(ckpt_dir, exist_ok=True)
    if server_device == "cuda" and not torch.cuda.is_available():
        server_device = "mps" if torch.backends.mps.is_available() else "cpu"
    # V21.1 E4: resolve the flag once (explicit param OR cfg knob), stamp
    # it for spawn-worker pickling. Default OFF = untouched behaviour.
    _rust_tree = bool(rust_tree or getattr(cfg, "rust_tree", False))
    try:
        cfg.rust_tree = bool(_rust_tree)
    except Exception:
        pass
    if bool(_rust_tree):
        print("[rust-tree] ON: arena/gate/selfplay search via "
              "RustTreeBackend (identical params)", flush=True)
    torch.manual_seed(0)
    np.random.seed(0)
    import chess_zero.game as _game
    _game.ADJUDICATE_MARGIN = cfg.adjudicate_margin
    _game.NO_PROGRESS_PLIES = cfg.no_progress_plies
    _game.ADJUDICATE_MIN_PLY = cfg.adjudicate_min_ply
    _game.INPUT_PLANES = cfg.input_planes
    from . import warmstart as _warm_encoding
    _warm_encoding.ENCODING_PLANES = cfg.input_planes
    sf_sims = arena_sims if sf_sims is None else sf_sims
    from .model import AlphaZeroNet as _Net
    model = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels,
                          planes=cfg.input_planes,
                          se_ratio=int(getattr(cfg, "se_ratio", 0) or 0)
                          ).to(device)
    if resume:
        from .model import load_weights as _lw
        _resume_checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        _lw(model, _resume_checkpoint)
        print(f"weights from {resume} (saved replay and optimizer restored below)",
              flush=True)
    incumb_path = os.path.join(ckpt_dir, "incumbent.pt")
    if resume and os.path.exists(incumb_path):
        _inc_checkpoint = _weights_dict(incumb_path)
        incumb_w = _inc_checkpoint.get("weights", _inc_checkpoint)
    else:
        incumb_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        atomic_save(incumb_w, incumb_path)
    # Persistent challenger: the model trains CONTINUOUSLY across iters.
    # Re-initializing from the incumbent every iter discarded all cumulative
    # learning (challengers then lost 0-10 to the incumbent forever).
    # v19 C2: AdamW, decay (conv/FC, wd=cfg.l2) + no-decay (1-dim).
    opt = _build_adamw(model, cfg)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)
    if resume:
        # audit-3.4: cuts used to discard optimizer + LR schedule every
        # time, so no config ever completed milestones 25/40. Restore them
        # when the resume file carries them (iter files do; champion files
        # too since v8.5) — weights always load, optimizer state restores
        # opportunistically. Buffer still resets. v9: load_optimizer_state
        # pads momentum across plane growth (16->18ch), so arch-change cuts
        # keep the schedule instead of foreach-crashing on iter 1.
        try:
            _rc = _resume_checkpoint
            from .model import load_optimizer_state as _los
            print(_los(opt, _rc) + f" from {resume}", flush=True)
            if "sched" in _rc:
                sched.load_state_dict(_rc["sched"])
                print(f"resumed schedule from {resume}", flush=True)
        except Exception as e:
            print(f"fresh optimizer ({type(e).__name__}: {str(e)[:80]})",
                  flush=True)
    buf = ReplayBuffer(capacity=cfg.buffer_size,
                       eg_capacity=cfg.eg_capacity, eg_frac=cfg.eg_frac)
    # V20.2 R2: load the AV table ONCE at run start (shared loader with
    # warmstart.load_av_table; share-code documented in _decode_av_cp_vector).
    # Empty cfg.av_table ("", default) -> {} -> rehearsal tails read zeros
    # (yesterday bit-exact, loss_av == 0.0). Miss rows also read zeros.
    _av_table: dict = {}
    try:
        _av_path = str(getattr(cfg, "av_table", "") or "")
        if _av_path:
            from .warmstart import load_av_table as _lat
            _av_table = _lat(_av_path) or {}
            print(f"[v20] AV table: {len(_av_table)} keys from {_av_path} "
                  f"(rehearsal-tail distillation ON)", flush=True)
        else:
            print("[v20] AV table OFF (cfg.av_table empty; rehearsal tails "
                  "read zeros, yesterday)", flush=True)
    except Exception as _e:
        print(f"[v20] AV table OFF (load failed {type(_e).__name__})",
              flush=True)
        _av_table = {}
    history = []
    if resume:
        restored = _resume_checkpoint
        if "replay" in restored:
            buf.buf.extend(restored["replay"]["main"])
            buf.eg.extend(restored["replay"]["endgame"])
            restored.pop("replay")  # deque entries now own the retained arrays
        if "rng" in restored:
            random.setstate(restored["rng"]["python"])
            np.random.set_state(restored["rng"]["numpy"])
            torch.set_rng_state(restored["rng"]["torch"])
            if torch.cuda.is_available() and "cuda" in restored["rng"]:
                torch.cuda.set_rng_state_all(restored["rng"]["cuda"])
        history_path = os.path.join(ckpt_dir, log_name)
        if os.path.exists(history_path):
            with open(history_path) as previous:
                history = [row for row in json.load(previous) if int(row.get("iter", 0)) < start_iter]
    # v19 game-budget books (C3): cumulative optimizer steps + self-play
    # games. V20 D9 adds cum_positions (recorded positions; shorter
    # endgame-start games must not slow LR decay — R2 game-counted LR
    # replaced). On resume, restore from history.json when present (else
    # the schedule restarts at 0 — logged loudly).
    _v19 = _v19_loop(cfg)
    _v20 = _v20_loop(cfg)
    cum_steps, cum_games, cum_positions = 0, 0, 0
    # V21 cumulative LLR (OR-path, future-only, both legs): float, init 0.0.
    cum_llr, cum_llr_eg = 0.0, 0.0
    _culled = False
    _bn_prev = None
    if (_v19 or _v20) and start_iter > 1:
        # D17: prefer best.pt/final_ckpt meta over history.json recompute
        # (meta carries the exact counters, incl. iters whose history rows
        # were rotated away); history.json is the fallback.
        # V21.1: prefer the --resume file's meta first (exact stop point;
        # final_ckpt may be a stale champion from an older run — e.g.
        # resuming iter21.pt while best.pt is a previous version's).
        _restored = False
        if resume:
            try:
                _rm = _resume_checkpoint
                _rmm = (_rm.get("meta", {}) or {})
                if "cum_steps" in _rmm and "cum_games" in _rmm:
                    cum_steps, cum_games = int(_rmm["cum_steps"]), int(
                        _rmm["cum_games"])
                    cum_positions = int(_rmm.get("cum_positions", 0))
                    _restored = f"meta of resume {resume}"
            except Exception:
                pass
        if not _restored:
            try:
                _fm = torch.load(os.path.join(ckpt_dir, final_ckpt),
                                 map_location="cpu",
                                 weights_only=False)
                _mm = (_fm.get("meta", {}) or {})
                if "cum_steps" in _mm and "cum_games" in _mm:
                    cum_steps, cum_games = int(_mm["cum_steps"]), int(
                        _mm["cum_games"])
                    cum_positions = int(_mm.get("cum_positions", 0))
                    _restored = f"meta of {final_ckpt}"
            except Exception as _e:
                print(f"[v19] meta resume unreadable "
                      f"({type(_e).__name__}); trying {log_name}",
                      flush=True)
        if not _restored:
            try:
                with open(os.path.join(ckpt_dir, log_name)) as _hf:
                    _hh = json.load(_hf)
                _pg = sum(sum(int((_e.get("games", {}) or {}).get(_k, 0))
                              for _k in ("1-0", "0-1", "1/2-1/2"))
                          for _e in _hh)
                _ps = sum(int(_e.get("train_steps", 0) or 0) for _e in _hh)
                cum_games, cum_steps = int(_pg), int(_ps)
                cum_positions = sum(int(_e.get("positions", 0) or 0)
                                    for _e in _hh)
                _restored = log_name
            except Exception as _e:
                print(f"[v19] counters restart at 0 "
                      f"({type(_e).__name__}); LR warmup replays", flush=True)
        if _restored:
            print(f"[v19] resumed counters from {_restored}: "
                  f"cum_games={cum_games} cum_steps={cum_steps} "
                  f"cum_positions={cum_positions}", flush=True)
            # D17 LR continuity assert: schedule LR from the restored
            # counters vs the optimizer's current LR (loud on mismatch).
            # V20 reads the position-counted schedule (D9/R2).
            try:
                _lr_sched = _v20_lr_for(cfg, cum_steps, cum_positions) \
                    if _v20 else _v19_lr_for(cfg, cum_steps, cum_games)
                _lr_opts = [float(_g.get("lr", -1.0))
                            for _g in opt.param_groups]
                print(f"[v19] LR continuity: schedule={_lr_sched:.2e} "
                      f"opt={['%.2e' % _v for _v in _lr_opts]} "
                      f"(per-iter set() reconciles; >1% drift warns)",
                      flush=True)
                for _v in _lr_opts:
                    if _v > 0 and abs(_v - _lr_sched) / max(
                            _lr_sched, 1e-12) > 0.01:
                        print(f"[v19] WARNING: LR discontinuity "
                              f"(opt {_v:.2e} vs schedule "
                              f"{_lr_sched:.2e})", flush=True)
                        break
            except Exception as _e:
                print(f"[v19] LR continuity check skipped "
                      f"({type(_e).__name__})", flush=True)
    # V21 cumulative LLR restore (any cfg): --resume file first (exact
    # stop point, same run — NOT retroactive credit), then final_ckpt
    # meta, then last history entry; missing -> 0.0. Never raises; runs
    # on any resume (start_iter > 1).
    if start_iter > 1:
        _v21_restored = False
        if resume:
            try:
                _rm21 = _resume_checkpoint
                _rmm21 = (_rm21.get("meta", {}) or {})
                if "cum_llr" in _rmm21 or "cum_llr_eg" in _rmm21:
                    cum_llr = float(_rmm21.get("cum_llr", 0.0) or 0.0)
                    cum_llr_eg = float(_rmm21.get("cum_llr_eg", 0.0)
                                       or 0.0)
                    _v21_restored = f"meta of resume {resume}"
            except Exception:
                pass
        if not _v21_restored:
            try:
                _fm21 = torch.load(os.path.join(ckpt_dir, final_ckpt),
                                   map_location="cpu",
                                   weights_only=False)
                _mm21 = (_fm21.get("meta", {}) or {})
                if "cum_llr" in _mm21 or "cum_llr_eg" in _mm21:
                    cum_llr = float(_mm21.get("cum_llr", 0.0) or 0.0)
                    cum_llr_eg = float(_mm21.get("cum_llr_eg", 0.0)
                                       or 0.0)
                    _v21_restored = f"meta of {final_ckpt}"
            except Exception:
                pass
        if not _v21_restored:
            try:
                with open(os.path.join(ckpt_dir, log_name)) as _hf21:
                    _hh21 = json.load(_hf21)
                if _hh21:
                    _last21 = _hh21[-1] or {}
                    if "cum_llr" in _last21 or "cum_llr_eg" in _last21:
                        cum_llr = float(_last21.get("cum_llr", 0.0)
                                        or 0.0)
                        cum_llr_eg = float(_last21.get("cum_llr_eg", 0.0)
                                           or 0.0)
                        _v21_restored = log_name
            except Exception:
                pass
        if _v21_restored:
            print(f"[v21] resumed cum_llr={cum_llr:.4f} "
                  f"cum_llr_eg={cum_llr_eg:.4f} from {_v21_restored}",
                  flush=True)
    # lineage cop (user directive): promotion counter persists via
    # best.pt meta so the "every Nth promotion" progress check survives
    # restarts and history rotation.
    _promo_base = 0
    try:
        _b0 = torch.load(os.path.join(ckpt_dir, final_ckpt),
                         map_location="cpu", weights_only=False)
        _promo_base = int((_b0.get("meta", {}) or {}).get("promo_seq", 0))
    except Exception:
        pass
    _promos_this_run = 0

    import copy as _copy_baseline
    _baseline_model = _copy_baseline.deepcopy(model)
    _baseline_model.load_state_dict(incumb_w, strict=True)
    base = _arena(_baseline_model, cfg, arena_games, arena_sims, device, workers,
                  ckpt_dir, sf_games, sf_rung, sf_sims, sf_path, note="iter0",
                  seed_base=1000, use_server=use_server,
                  server_device=server_device, rust_tree=_rust_tree)
    del _baseline_model
    print(f"[iter 0/incumbent] {base}", flush=True)
    # §6.3: corroborated promotion needs an incumbent arena baseline.
    # Score = vs_greedy (wins+0.5*draws)/n. Promotions require gate pass
    # AND no arena regression beyond tolerance (both use paired openings
    # but DIFFERENT seed bases so the corroboration is independent).
    def _arena_score(a: dict) -> float:
        g = (a or {}).get("vs_greedy") or {}
        n = g.get("wins", 0) + g.get("losses", 0) + g.get("draws", 0)
        if n <= 0:
            return 0.0
        return (g.get("wins", 0) + 0.5 * g.get("draws", 0)) / n

    incumb_arena_score = _arena_score(base)

    if resume and "rng" in restored:
        random.setstate(restored["rng"]["python"])
        np.random.set_state(restored["rng"]["numpy"])
        torch.set_rng_state(restored["rng"]["torch"])
        if torch.cuda.is_available() and "cuda" in restored["rng"]:
            torch.cuda.set_rng_state_all(restored["rng"]["cuda"])
    if resume:
        for consumed in ("weights", "opt", "sched"):
            _resume_checkpoint.pop(consumed, None)
    for it in range(start_iter, start_iter + iters):
        t0 = time.time()
        if _v19:
            # D16 pre-iter disk guard (>2GB free else the iter aborts
            # loudly — history/ckpt stay intact, no crash).
            _dok, _dfree = disk_ok(ckpt_dir, 2.0)
            if not _dok:
                print(f"[iter {it}] DISK GUARD: only {_dfree}GB free "
                      f"(need 2.0GB) — iter ABORTED loudly, run intact",
                      flush=True)
                history.append({"iter": it, "aborted": "disk-full",
                                "free_gb": _dfree,
                                "cum_steps": cum_steps,
                                "cum_games": cum_games,
                                "cum_llr": float(cum_llr),
                                "cum_llr_eg": float(cum_llr_eg)})
                with open(os.path.join(ckpt_dir, log_name), "w") as f:
                    json.dump(history, f, indent=1)
                continue
        # live adjudication table for this iter's workers (audit round 2:
        # fitted values never reached self-play; the cfg object carries
        # them now, refreshed after each refit below).
        cfg.adjudicate_values = dict(_game.ADJUDICATE_VALUES)
        if _v19:
            # C4: dirichlet schedule before each self-play phase. Workers
            # read cfg per phase (parallel pickles it per call, local
            # reads it live) — propagation path verified.
            cfg.dirichlet_eps = _v19_eps_for(cum_games)
            print(f"[iter {it}] v19 dirichlet_eps={cfg.dirichlet_eps} "
                  f"(cum_games={cum_games})", flush=True)
        phase_start = time.perf_counter()
        phase_seconds = {}
        # ---- self-play with the frozen INCUMBENT ----
        if _v19:
            # D9 PFSP-lite: stamp last-4 champion snapshots into cfg for
            # workers (parallel pickles cfg per call — same propagation
            # path as the dirichlet schedule above). Empty = all self.
            try:
                cfg.pool_paths = tuple(_pool_files(ckpt_dir))
            except Exception:
                cfg.pool_paths = ()
        # V20 D4/D5/D6 per-iter self-play state (Agent B). Endgame FENs
        # load once per iter (panel-endgame is wired in _gate below, so
        # starts are allowed here — D4 order: panel FIRST, then starts).
        # TB tables resolve once per iter (None + loud skip when the
        # staging dir is incomplete — training continues as yesterday).
        _eg_fens: list = []
        _tb = None
        _tb_info: dict = {"skipped": "non-v20"}
        _rng20 = random.Random(8000 + int(it) * 7919)
        _tb_sum = {"probed": 0, "guarded": 0, "rewritten": 0,
                   "deblundered": 0, "skipped": 0, "skipped_book": 0}
        _n_eg = 0
        _n_pt = 0
        _iter_positions = 0
        if _v20:
            if endgame_starts_allowed(cfg):
                try:
                    from .selfplay import load_endgame_fens as _lef
                    _eg_fens = _lef(str(getattr(
                        cfg, "endgame_file",
                        "data_sl/endgames.jsonl")))
                except Exception as _e:
                    print(f"[iter {it}] endgame starts OFF "
                          f"({type(_e).__name__})", flush=True)
                    _eg_fens = []
            try:
                from . import tb_rescore as _tbm
                _tb, _tb_info = _tbm.resolve_tables(
                    str(getattr(cfg, "tb_path", "data_tb")))
            except Exception as _e:
                print(f"[iter {it}] TB rescore OFF "
                      f"({type(_e).__name__})", flush=True)
                _tb, _tb_info = None, {"skipped": type(_e).__name__}
        from .experiment_support import challenger_game_ids, collect_mixed_selfplay
        _actor_fraction = float(getattr(cfg, "challenger_selfplay_frac", 0.0))
        _challenger_ids = challenger_game_ids(games_per_iter, _actor_fraction, it)
        _challenger_w = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()} if _challenger_ids else None
        if workers > 1:
            from .parallel import play_games_parallel
            w_path = os.path.join(ckpt_dir, "_w_current.pt")
            atomic_save(incumb_w, w_path)
            # v14: no parent-side model build — workers load from the
            # weights FILE (the old incumb_cpu was constructed, loaded,
            # then discarded unused every iter).
            if _challenger_ids:
                _actor_path = os.path.join(ckpt_dir, "_w_selfplay_challenger.pt")
                atomic_save(_challenger_w, _actor_path)
                examples, results = collect_mixed_selfplay(
                    play_games_parallel, cfg, games_per_iter, cfg.temp_moves, workers,
                    w_path, _actor_path, _actor_fraction, use_server, server_device)
            else:
                examples, results = play_games_parallel(
                    None, cfg, games_per_iter, cfg.temp_moves, workers,
                    weights_path=w_path, use_server=use_server,
                    server_device=server_device)
                results["actor_champion_games"] = games_per_iter
            buf.add_game(examples)
            # V20 D9: positions counted here (parent sees the flat list;
            # per-game TB/endgame counters ride inside results via the
            # workers' union-merge). The parent TB handle (sequential
            # path only) closes here — workers open/close their own.
            _iter_positions = len(examples)
            try:
                if _tb is not None:
                    _tb.close()
            except Exception:
                pass
            _tb = None
        else:
            # audit-1.7: the old code loaded incumb_w INTO the persistent
            # challenger (model.load_state_dict), i.e. exactly the
            # re-init-every-iter behaviour §3 says was discarded. The
            # parallel path builds a separate incumb_cpu; mirror that here
            # so local/1-worker runs model the VM.
            from .model import AlphaZeroNet as _Net1
            from .model import load_weights as _lw1
            from .model import infer_se_ratio as _isr1
            from .evaluate import sparring_for_game as _spar_opp
            from .evaluate import sparring_kind_for_game as _spar_kind
            incumb_seq = _Net1(blocks=cfg.blocks, channels=cfg.channels,
                               planes=cfg.input_planes,
                               se_ratio=_isr1(incumb_w)).to(device)
            _lw1(incumb_seq, incumb_w)
            n_spar = min(games_per_iter,
                         int(round(games_per_iter * float(
                             getattr(cfg, "sparring_frac", 0.0) or 0.0))))
            # D9 PFSP-lite (v19 only): game-kind coin self/pool/punisher
            # (12/5/3 of 20) BEFORE the full/fast coin. Pool opponents are
            # agent_from_weights(last-4 champion files), noise 0 / temp 0,
            # delivered through the existing sparring= mechanism — pool
            # games record OUR moves only (existing sparring semantics,
            # asymmetric by design). Old configs keep sparring_frac.
            _pfsp = _pfsp_kinds(games_per_iter) if _v19 else []
            _pool_paths = _pool_files(ckpt_dir) if _v19 else []
            if _v19 and any(k == "pool" for k in _pfsp) and not _pool_paths:
                print(f"[iter {it}] PFSP pool empty (no champion "
                      f"snapshots yet); pool games run as self-play",
                      flush=True)
            results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "S1-0": 0,
                        "S0-1": 0, "S1/2-1/2": 0, "vetoes": 0, "tactics": 0,
                        "Tmate": 0, "Tresign": 0, "Tadjudicated": 0,
                        "Tadjudicated-draw": 0, "Trules-draw": 0, "Tcap": 0,
                        "breadth": 0.0, "ventropy": 0.0, "bgames": 0}
            _last_actor = None
            for i in range(games_per_iter):
                _actor = "challenger" if i in _challenger_ids else "champion"
                if _actor != _last_actor:
                    _lw1(incumb_seq, _challenger_w if _actor == "challenger" else incumb_w)
                    _last_actor = _actor
                results[f"actor_{_actor}_games"] = results.get(f"actor_{_actor}_games", 0) + 1
                st: dict = {}
                _kind = _pfsp[i] if i < len(_pfsp) else "self"
                if _kind == "pool" and not _pool_paths:
                    _kind = "self"
                _played = False
                if _v19 and _kind in ("pool", "punisher"):
                    if _kind == "pool":
                        _pp = random.Random(6000 + it * 131 + i).choice(
                            _pool_paths)
                        try:
                            _opp = agent_from_weights(
                                _weights_dict(_pp), cfg, sims=None,
                                device=device, noise=0.0)
                        except Exception as _e:
                            print(f"[iter {it}] pool opponent load "
                                  f"failed ({type(_e).__name__}); game "
                                  f"{i} aborted", flush=True)
                            raise RuntimeError("Historical opponent failed to load") from _e
                    else:
                        # D9+D14 merge: the "punisher" slots run the
                        # 60/20/20 sparring rotation (punisher/noisy/
                        # tactics by game index).
                        _opp = _spar_opp(i, kind=_spar_kind(i))
                    if _kind != "self":
                        ex, res = play_game(
                            incumb_seq, cfg, device=device,
                            temp_moves=cfg.temp_moves,
                            opening_random_moves=cfg.opening_random_moves,
                            resign_threshold=cfg.resign_threshold,
                            resign_moves=cfg.resign_moves,
                            sparring=_opp, spar_white=(i % 2 == 0),
                            stats=st)
                        if _v20:
                            try:
                                from . import tb_rescore as _tbm4
                                _rs = _v20_rescore_game(
                                    cfg, _tbm4, _tb, ex, st, res)
                                for _k in _tb_sum:
                                    _tb_sum[_k] = _tb_sum.get(_k, 0) + \
                                        int(_rs.get(_k, 0) or 0)
                            except Exception:
                                pass
                        buf.add_game(ex)
                        results[res] += 1
                        results["S" + res] += 1
                        results["pfsp_" + _kind] = \
                            results.get("pfsp_" + _kind, 0) + 1
                        _played = True
                        _iter_positions += len(ex)
                if _played:
                    pass  # PFSP pool/punisher game above; shared tail below
                elif (not _v19) and i < n_spar:
                    ex, res = play_game(
                        incumb_seq, cfg, device=device,
                        temp_moves=cfg.temp_moves,
                        opening_random_moves=cfg.opening_random_moves,
                        resign_threshold=cfg.resign_threshold,
                        resign_moves=cfg.resign_moves,
                        sparring=_spar_opp(i), spar_white=(i % 2 == 0),
                        stats=st)
                    if _v20:
                        try:
                            from . import tb_rescore as _tbm5
                            _rs = _v20_rescore_game(
                                cfg, _tbm5, _tb, ex, st, res)
                            for _k in _tb_sum:
                                _tb_sum[_k] = _tb_sum.get(_k, 0) + \
                                    int(_rs.get(_k, 0) or 0)
                        except Exception:
                            pass
                    buf.add_game(ex)
                    results[res] += 1
                    results["S" + res] += 1
                    _iter_positions += len(ex)
                else:
                    import random as _rr2
                    fp = float(getattr(cfg, "full_playout_frac", 0.0)
                               or 0.0)
                    full = bool(fp > 0.0 and _rr2.random() < fp)
                    ff = float(getattr(cfg, "fast_frac", 0.0) or 0.0)
                    fast = bool(not full and ff > 0.0
                                and _rr2.random() < ff)
                    # V20 D4 endgame starts (panel-endgame is wired in
                    # _gate, so starts are allowed here — order fixed):
                    # endgame_frac games open from the harvested list
                    # (uniform); contempt is forced 0 inside selfplay
                    # and book/random openings are skipped there.
                    # Sparring/pool/punisher games never start here
                    # (punishment/technique stay separate). D5: the
                    # playthrough coin (5%) ignores resign entirely
                    # (F5 refusal: never full no-resign; mate-finish
                    # untouched).
                    _fen = None
                    _pt = False
                    if _v20:
                        if _eg_fens and _roll_endgame(
                                _rng20, float(getattr(
                                    cfg, "endgame_frac", 0.0) or 0.0)):
                            _fen = _rng20.choice(_eg_fens)
                            _n_eg += 1
                        if _roll_playthrough(_rng20, float(getattr(
                                cfg, "playthrough_frac", 0.0) or 0.0)):
                            _pt = True
                            _n_pt += 1
                    ex, res = play_game(
                        incumb_seq, cfg, device=device,
                        temp_moves=cfg.temp_moves,
                        opening_random_moves=cfg.opening_random_moves,
                        resign_threshold=cfg.resign_threshold,
                        resign_moves=cfg.resign_moves, full_playout=full,
                        fast=fast, stats=st, start_fen=_fen,
                        playthrough=_pt)
                    # V20 D6+D8 post-game TB pass (value-only; tables
                    # missing -> loud skip inside resolve, counts zero).
                    if _v20:
                        try:
                            from . import tb_rescore as _tbm3
                            _rs = _v20_rescore_game(
                                cfg, _tbm3, _tb, ex, st, res)
                            for _k in _tb_sum:
                                _tb_sum[_k] = _tb_sum.get(_k, 0) + \
                                    int(_rs.get(_k, 0) or 0)
                        except Exception:
                            pass
                    buf.add_game(ex)
                    results[res] += 1
                    _iter_positions += len(ex)
                results["vetoes"] += st.get("vetoes", 0)
                results["tactics"] += st.get("tactics", 0)
                results["finishes"] = results.get("finishes", 0) + \
                    st.get("finishes", 0)
                results["safety"] = results.get("safety", 0) + \
                    st.get("safety", 0)
                results["fast"] = results.get("fast", 0) + \
                    st.get("fast", 0)
                tkey = "T" + str(st.get("terminal", "cap"))
                results[tkey] = results.get(tkey, 0) + 1
                # v14 visit breadth.
                results["breadth"] += float(st.get("breadth", 0.0))
                results["ventropy"] += float(st.get("ventropy", 0.0))
                results["bgames"] += 1
                results["plies"] = results.get("plies", 0) + \
                    int(st.get("plies", 0))
            # V20 sequential-path closeout: TB handle closes (workers
            # own theirs), per-iter D4/D5/D6 counters land in results.
            # ADDITIVE (workers>1 already merged theirs via union-merge;
            # never overwrite).
            if _v20:
                try:
                    if _tb is not None:
                        _tb.close()
                except Exception:
                    pass
                _tb = None
                results["endgame_starts"] = \
                    results.get("endgame_starts", 0) + int(_n_eg)
                results["playthroughs"] = \
                    results.get("playthroughs", 0) + int(_n_pt)
                results["positions"] = \
                    results.get("positions", 0) + int(_iter_positions)
                for _k, _v in _tb_sum.items():
                    results["tb_" + _k] = results.get("tb_" + _k, 0) + \
                        int(_v or 0)
                print(f"[iter {it}] v20 self-play: endgame_starts="
                      f"{results.get('endgame_starts', 0)} "
                      f"playthroughs={results.get('playthroughs', 0)} "
                      f"positions={results.get('positions', 0)} "
                      f"tb_probed={results.get('tb_probed', 0)} "
                      f"tb_guarded={results.get('tb_guarded', 0)} "
                      f"tb_rewritten={results.get('tb_rewritten', 0)} "
                      f"tb_deblundered={results.get('tb_deblundered', 0)}",
                      flush=True)
        phase_seconds["selfplay"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        # ---- train CHALLENGER (persistent: cumulative across iters) ----
        cum_games += int(games_per_iter)
        # V20 D9: positions accumulate every iter (both worker paths;
        # parallel counted at collection above).
        results["positions"] = int(_iter_positions)
        cum_positions += int(_iter_positions)
        if _v19:
            # C5: aux late cull before each train phase (one-way latch;
            # idempotent values: re-fire after resume past the point is a
            # single spurious log line, never a behavior change).
            _fired, _culled = _maybe_aux_cull(cfg, cum_games, _culled)
            if _fired:
                print(f"[iter {it}] V19 LATE CULL @ cum_games={cum_games}: "
                      f"mob_w/safety_w/margin_w/check_w -> 0.005 "
                      f"(one-way, never restored)", flush=True)
        if _v20:
            # D9/R2: position-counted LR on BOTH param groups (same
            # shape as C3; milestones kept equivalent via
            # lr_drop_positions ~= 7000 games x ~80 plies). Value
            # schedule unchanged (D9 touches LR only).
            lr = _v20_lr_for(cfg, cum_steps, cum_positions)
            for _g in opt.param_groups:
                _g["lr"] = lr
            value_w = _v19_value_w_for(cfg, cum_games)
        elif _v19:
            # C3: game-budget LR on BOTH param groups + value schedule.
            lr = _v19_lr_for(cfg, cum_steps, cum_games)
            for _g in opt.param_groups:
                _g["lr"] = lr
            value_w = _v19_value_w_for(cfg, cum_games)
        else:
            lr = opt.param_groups[0]["lr"]
            value_w = 1.0
        pl_acc, vl_acc, al_acc, ml_acc, en_acc, tot_acc, n = \
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        ol_acc, sl_acc, bl_acc = 0.0, 0.0, 0.0  # v10 dense heads
        kl_acc = 0.0  # v16 king safety
        rl_acc = 0.0  # v18 opp-reply aux
        tkl_acc = 0.0  # v19 KL-anchor teacher term
        prog_acc = 0.0  # v19 plies-progress aux
        sc_acc, avv_acc, td_acc = 0.0, 0.0, 0.0  # v20 score/av/td
        # §6.7: scale steps by buffer occupancy. 500 steps over one iter's
        # ~18k positions at full LR erased the value head (§5.3 — 7 epochs
        # over a fresh buffer). Scale linearly until the buffer fills;
        # always do at least 1 step so the loop still moves.
        eff_steps = train_steps
        if len(buf) >= cfg.batch_size:
            occ = min(1.0, len(buf) / float(cfg.buffer_size))
            eff_steps = max(1, int(round(train_steps * occ)))
            if eff_steps != train_steps:
                print(f"[iter {it}] buffer {len(buf)}/{cfg.buffer_size} "
                      f"-> steps {eff_steps}/{train_steps}", flush=True)
            if _v19:
                # C10: sampling-ratio ceiling (plies ~= recorded positions;
                # results["plies"] also counts unrecorded opening plies, so
                # it overestimates slightly — documented, acceptable).
                _plies = int(_iter_positions)
                if _plies > 0:
                    _ratio = _ratio_steps(_plies, cfg.batch_size,
                                          train_steps, getattr(cfg, "sample_reuse", 1.5))
                    if _ratio < eff_steps:
                        print(f"[iter {it}] ratio cap: plies={_plies} -> "
                              f"steps {eff_steps}->{_ratio}", flush=True)
                        eff_steps = _ratio
                else:
                    print(f"[iter {it}] ratio skipped (no plies reported; "
                          f"occupancy steps {eff_steps})", flush=True)
        else:
            # v14 honesty: tiny buffer means zero steps run (history used
            # to claim train_steps while running none).
            eff_steps = 0
        # C8: KL anchor + rehearsal mix. Teacher frozen/eval/no_grad;
        # forward on rehearsal rows only; KL masked to those rows with
        # kl_w decaying over kl_games. All skips are loud, never raising.
        teacher, reh_pool, kl_w = None, [], 0.0
        reh_ids: list = []
        n_reh, _mix = 0, False
        if (_v19 or _v20) and eff_steps > 0:
            kl_w = float(getattr(cfg, "kl_w0", 0.0) or 0.0) * max(
                0.0, 1.0 - cum_games / max(
                    1, int(getattr(cfg, "kl_games", 5000) or 5000)))
            _frac = float(getattr(cfg, "rehearsal_frac", 0.0) or 0.0)
            n_reh = int(round(cfg.batch_size * _frac))
            if n_reh > 0:
                teacher = _load_teacher(cfg, device)
                if n_reh > 0:
                    _need = eff_steps * n_reh
                    _kgames = min(500, max(20, int(_need // 30) + 1))
                    # V20.2 R1: rows + parallel (game_id, ply) ids (same
                    # order/len as the legacy pool; miss -> zeros).
                    try:
                        reh_pool, reh_ids = _sample_rehearsal_pool_with_ids(
                            cfg, _kgames, 9000 + it)
                    except Exception:
                        reh_pool = _sample_rehearsal_pool(cfg, _kgames,
                                                          9000 + it)
                        reh_ids = [(None, int(_i))
                                   for _i in range(len(reh_pool))]
                    _mix = bool(reh_pool)
                    if not _mix:
                        teacher = None
        _dev = next(model.parameters()).device
        for _ in range(eff_steps):
            if _v20 or _v19:
                lr = (_v20_lr_for(cfg, cum_steps + n, cum_positions) if _v20
                      else _v19_lr_for(cfg, cum_steps + n, cum_games))
                for group in opt.param_groups:
                    group["lr"] = lr
            # Index (not unpack): replay-owned sample() arity moves with
            # the tuple growth (11 yesterday, 12 +ml_mask, 13 +P per D28,
            # 17 with V20 score_mean/score_stdev/root_q/av appended).
            _samp = buf.sample(cfg.batch_size)
            s, pi, z, m, ml = _samp[0], _samp[1], _samp[2], _samp[3], \
                _samp[4]
            own, margin, mob, safe = _samp[5], _samp[6], _samp[7], \
                _samp[8]
            reply, pw = _samp[9], _samp[10]
            _mlm = _samp[11] if len(_samp) > 11 else None
            _prg = _samp[12] if len(_samp) > 12 else None
            # V20 D1/D2/D3 targets (EXACT names score_mean/score_stdev/
            # root_q/av_logits; None when the replay half predates them
            # -> train_step reads zero loss, yesterday bit-exact).
            _sm = _samp[13] if len(_samp) > 13 else None
            _ss = _samp[14] if len(_samp) > 14 else None
            _rq = _samp[15] if len(_samp) > 15 else None
            _av = _samp[16] if len(_samp) > 16 else None
            _tp, _km, _kw = None, None, 0.0
            if _mix and n_reh > 0 and reh_pool:
                # (1-frac) buffer rows + frac rehearsed rows: overwrite the
                # batch tail with rehearsal rows, mask the KL term to them.
                # V20.2 R1: sample (row, id) PAIRS so the (game_id, ply) ids
                # stay aligned with the rows; random.sample chooses indices
                # only, so the rows part is bit-identical to yesterday's
                # random.sample(reh_pool, ...) (same len, same RNG state).
                try:
                    try:
                        _rp = list(reh_pool)
                        _ri = list(reh_ids)
                        # Defensive: ids must parallel rows (same len);
                        # pad/truncate (miss -> zeros) so sampling len ==
                        # yesterday even on skew.
                        if len(_ri) < len(_rp):
                            _ri = _ri + [(None, -1)] * (len(_rp) - len(_ri))
                        elif len(_ri) > len(_rp):
                            _ri = _ri[:len(_rp)]
                        _paired = list(zip(_rp, _ri))
                    except Exception:
                        _paired = [(r, (None, -1)) for r in reh_pool]
                    _picked = random.sample(
                        _paired, min(n_reh, len(_paired)))
                    _rows = [_p[0] for _p in _picked]
                    _row_ids = [_p[1] if len(_p) > 1 else (None, -1)
                                for _p in _picked]
                    _nn = len(_rows)
                    _okshape = (_nn > 0 and
                                np.shape(_rows[0][0]) == np.shape(s)[1:])
                    if _okshape:
                        _B = int(s.shape[0])
                        s[-_nn:] = np.stack(
                            [_r[0] for _r in _rows]).astype(np.float32)
                        pi[-_nn:] = np.stack(
                            [_r[1] for _r in _rows]).astype(np.float32)
                        z[-_nn:] = np.array(
                            [_r[2] for _r in _rows], dtype=np.float32)
                        m[-_nn:] = np.array(
                            [_r[3] for _r in _rows], dtype=np.float32)
                        ml[-_nn:] = np.array(
                            [_r[4] for _r in _rows], dtype=np.float32)
                        own[-_nn:] = np.stack(
                            [_r[5] for _r in _rows]).astype(np.float32)
                        margin[-_nn:] = np.array(
                            [_r[6] for _r in _rows], dtype=np.float32)
                        mob[-_nn:] = np.array(
                            [_r[7] for _r in _rows], dtype=np.float32)
                        safe[-_nn:] = np.array(
                            [_r[8] for _r in _rows], dtype=np.float32)
                        reply[-_nn:] = np.array(
                            [_r[9] for _r in _rows], dtype=np.int64)
                        pw[-_nn:] = np.array(
                            [_r[10] if len(_r) > 10 else 1.0
                             for _r in _rows], dtype=np.float32)
                        # v19 rehearsal rows are 13-wide (11 + P + mask):
                        # overwrite progress/mask tails too, else stale
                        # buffer values would teach wrong P on SL rows.
                        try:
                            _prg = np.asarray(_prg)
                            _prg[-_nn:] = np.array(
                                [_r[11] if len(_r) > 11 else 0.0
                                 for _r in _rows], dtype=np.float32)
                        except Exception:
                            pass
                        try:
                            _mlm = np.asarray(_mlm)
                            _mlm[-_nn:] = np.array(
                                [_r[12] if len(_r) > 12 else 1.0
                                 for _r in _rows], dtype=np.float32)
                        except Exception:
                            pass
                        # V20 rehearsal tails (loop-owned; warmstart rows
                        # are 11/13-wide legacy): SL rows carry neutral
                        # V20 defaults — score 0/stdev 1 (no margin
                        # teaching off-distribution), root_q = z (TD ==
                        # z exactly, yesterday). V20.2 R3: av fills REAL
                        # vectors from the run-once AV table (hit -> int8
                        # q8 dequantized cp vector, miss -> zeros = today,
                        # masked in train.py since genuine SF vectors never
                        # have zero spread). Empty table = yesterday
                        # bit-exact bulk zeros. Else stale buffer values
                        # would teach wrong score/av on SL rows.
                        try:
                            if _sm is not None:
                                _sm = np.asarray(_sm)
                                _sm[-_nn:] = np.nan
                        except Exception:
                            pass
                        try:
                            if _ss is not None:
                                _ss = np.asarray(_ss)
                                _ss[-_nn:] = np.nan
                        except Exception:
                            pass
                        try:
                            if _rq is not None:
                                _rq = np.asarray(_rq)
                                _rq[-_nn:] = np.asarray(
                                    z[-_nn:], dtype=np.float32)
                        except Exception:
                            pass
                        try:
                            if _av is not None:
                                _av = np.asarray(_av)
                                if not _av_table:
                                    # Empty = yesterday bit-exact.
                                    _av[-_nn:] = np.nan
                                else:
                                    for _j, _rid in enumerate(_row_ids):
                                        try:
                                            _gid, _ply = _rid
                                        except Exception:
                                            _gid, _ply = None, -1
                                        _dst = int(_B) - int(_nn) + int(_j)
                                        try:
                                            _av[_dst] = _decode_av_cp_vector(
                                                _av_table, _gid, _ply)
                                        except Exception:
                                            try:
                                                _av[_dst] = np.nan
                                            except Exception:
                                                pass
                        except Exception:
                            pass
                        if teacher is not None and kl_w > 0:
                            _km = torch.zeros(_B, dtype=torch.bool)
                            _km[-_nn:] = True
                            with torch.no_grad():
                                _tout = teacher.forward_pv(torch.from_numpy(
                                    np.ascontiguousarray(s[-_nn:])).to(_dev))
                                _lp = torch.softmax(
                                    _tout[0].float(), dim=1).cpu().numpy()
                            _tp = torch.zeros((_B, pi.shape[1]),
                                              dtype=torch.float32)
                            _tp[-_nn:] = torch.from_numpy(
                                np.asarray(_lp, dtype=np.float32))
                            _kw = kl_w
                except Exception as _e:
                    print(f"[iter {it}] rehearsal step skipped "
                          f"({type(_e).__name__})", flush=True)
                    _tp, _km, _kw = None, None, 0.0
            # Return-arity tolerant (train.py is Impl-A's file: 12 with KL
            # at END, 13 with the D4 progress term inserted before it, 16
            # with V20 score/av/td inserted before it; KL stays LAST in
            # all — index from the ends, never unpack).
            _out = train_step(
                model, opt,
                torch.from_numpy(s), torch.from_numpy(pi),
                torch.from_numpy(z), torch.from_numpy(m),
                torch.from_numpy(ml), torch.from_numpy(own),
                torch.from_numpy(margin), torch.from_numpy(mob),
                torch.from_numpy(safe),
                torch.from_numpy(reply), torch.from_numpy(pw),
                clip=cfg.grad_clip, aux_w=cfg.aux_w,
                ml_w=cfg.ml_w, ent_w=cfg.ent_w,
                # v19: first-class aux weights (yesterday's defaults for
                # old configs, so passing them is behavior-identical).
                own_w=float(getattr(cfg, "own_w", 0.1)),
                margin_w=float(getattr(cfg, "margin_w", 0.05)),
                mob_w=float(getattr(cfg, "mob_w", 0.05)),
                safe_w=float(getattr(cfg, "safe_w",
                                     getattr(cfg, "safety_w", 0.05))),
                reply_w=float(getattr(cfg, "reply_w", 0.05)),
                soft_w=float(getattr(cfg, "soft_w", 0.3)),
                check_w=float(getattr(cfg, "check_w", 0.01)),
                value_w=value_w,
                augment=bool(getattr(cfg, "augment", False)),
                smooth_eps=float(getattr(cfg, "smooth_eps", 0.0)),
                progress=torch.from_numpy(_prg)
                if _prg is not None else None,
                ml_mask=torch.from_numpy(_mlm)
                if _mlm is not None else None,
                teacher_probs=_tp, kl_w=_kw, kl_mask=_km,
                # V20 D1/D2/D3 (EXACT names; None = legacy -> zero
                # loss). score_step = global optimizer steps (the ramp
                # counts steps, like lr warmup).
                score_mean=torch.from_numpy(_sm)
                if _sm is not None else None,
                score_stdev=None,  # no calibrated dispersion labels available
                av_logits=torch.from_numpy(_av)
                if _av is not None else None,
                root_q=torch.from_numpy(_rq)
                if _rq is not None else None,
                score_step=cum_steps + n,
                score_w_max=float(getattr(cfg, "score_w_max", 0.05)),
                score_ramp_steps=int(getattr(cfg, "score_ramp_steps",
                                             5000) or 5000),
                td_lambda=float(getattr(cfg, "td_lambda", 0.5)),
                av_w=float(getattr(cfg, "av_w", 0.1)),
                av_temp=float(getattr(cfg, "av_temp", 2.0)),
                av_scale=float(getattr(cfg, "av_scale", 100.0)))
            tot, pl, vl, al, mll, en = _out[0:6]
            ol, sl, bl, kl, rl = _out[6], _out[7], _out[8], _out[9], \
                _out[10]
            tkl = _out[-1]  # KL teacher term is LAST (12/13/16-tuple)
            prog = _out[11] if len(_out) > 12 else 0.0
            # V20 16-tuple: loss_score (12), loss_av (13), loss_td (14)
            # inserted BEFORE KL; legacy tuples read 0.0 (never raises).
            _sc = _out[12] if len(_out) > 14 else 0.0
            _avf = _out[13] if len(_out) > 14 else 0.0
            _td = _out[14] if len(_out) > 14 else 0.0
            tot_acc += tot; pl_acc += pl; vl_acc += vl; al_acc += al
            ml_acc += mll; en_acc += en; n += 1
            ol_acc += ol; sl_acc += sl; bl_acc += bl; kl_acc += kl
            rl_acc += rl; tkl_acc += tkl; prog_acc += prog
            sc_acc += _sc; avv_acc += _avf; td_acc += _td
        cum_steps += n
        phase_seconds["training"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        # ---- arena: measure the challenger ----
        # §6.3: arena and gate use DIFFERENT seed bases (offset 500k) so
        # the corroborating measurement runs on different openings than
        # the selection test. Same pair structure within each match.
        gate_due, weak_due = _evaluation_schedule(
            cfg, it, start_iter + iters - 1)
        cfg.panel_weak_diagnostics = weak_due
        if gate_due:
            arena_seed = 1000 + it * 7919
            gate_seed = arena_seed + 500000
            arena = _arena(model, cfg, arena_games, arena_sims, device, workers,
                           ckpt_dir, sf_games, sf_rung, sf_sims, sf_path,
                           note=f"iter{it}", seed_base=arena_seed,
                           use_server=use_server, server_device=server_device,
                           rust_tree=_rust_tree)
            if cfg.sf_depth_suite and sf_games > 0 and it % 5 == 0:
                # Rung-3 book: depth-graded SF ladder every 5 iters, same budget
                from .calibrate import calibrate_ckpt
                w_path = os.path.join(ckpt_dir, "_w_arena.pt")
                atomic_save({k: v.cpu() for k, v in model.state_dict().items()},
                           w_path)
                suite = {}
                for rung in ("sf-depth1", "sf-depth3", sf_rung):
                    try:
                        suite[rung] = calibrate_ckpt(
                            w_path, rung, sf_games, sf_sims, cfg.blocks,
                            cfg.channels, sf_path, workers=max(1, workers),
                            input_planes=cfg.input_planes)
                    except Exception as e:
                        suite[rung] = {"error": str(e)[:200]}
                arena["sf_suite"] = suite
            if it % 5 == 0:
                print(f"[iter {it}] endgame buffer: {buf.eg_report()}",
                      flush=True)
            phase_seconds["arena"] = time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            # ---- gate: challenger vs incumbent head-to-head ----
            # Noisy SELECTION (dirichlet on); arena above is deterministic
            # MEASUREMENT. Promotion requires BOTH (conjunction), not gate
            # alone — a 20-game gate at 0.55 false-promotes equals ~36%.
            if _v19 or _v20:
                # C9: PreciseBN-lite before gate (BN stat refresh, then eval).
                _precise_bn_lite(model, buf)
            gate = _gate(model, incumb_w, cfg, arena_sims, device, workers,
                         ckpt_dir, seed_base=gate_seed, use_server=use_server,
                         server_device=server_device, rust_tree=_rust_tree)
            # V21 cumulative LLR (OR-path, future-only, both legs): accumulate
            # the ALREADY-COMPUTED panel llrs (main "llr" + endgame "llr"),
            # then OR with the per-iter verdicts. Legacy "sprt_pass" untouched.
            try:
                _v21_panel_llr = float(gate.get("llr") or 0.0)
            except Exception:
                _v21_panel_llr = 0.0
            try:
                _v21_eg = gate.get("endgame_panel")
                _v21_eg_llr = float(_v21_eg.get("llr") or 0.0) \
                    if isinstance(_v21_eg, dict) else 0.0
            except Exception:
                _v21_eg_llr = 0.0
            cum_llr = _v21_panel_llr
            cum_llr_eg = _v21_eg_llr
            try:
                _v21dec = _v21_gate_cum_decide(gate, cum_llr, cum_llr_eg, cfg)
            except Exception:
                _v21dec = {"main_leg": bool(gate.get("sprt_pass", False)),
                           "eg_leg": True,
                           "sprt_pass_cum": bool(gate.get("sprt_pass", False))}
            try:
                gate["cum_llr"] = float(cum_llr)
                gate["cum_llr_eg"] = float(cum_llr_eg)
                gate["main_leg"] = bool(_v21dec.get("main_leg", False))
                gate["eg_leg"] = bool(_v21dec.get("eg_leg", True))
                gate["sprt_pass_cum"] = bool(_v21dec.get("sprt_pass_cum",
                                                        False))
            except Exception:
                pass
            if _v19 or _v20:
                # C6: SPRT promotion (pair-wise LLR + truncated fallback
                # decided inside _gate); V20: sprt_pass already ANDs the
                # main panel with the endgame panel (D4/D9); arena
                # corroboration conjunction below is UNCHANGED. V21: the
                # promotion gate reads the cum OR-path (sprt_pass_cum);
                # legacy sprt_pass is preserved bit-identical (equal when
                # both cums are below bound).
                gate_pass = bool(gate.get("sprt_pass_cum",
                                          gate.get("sprt_pass", False)))
            else:
                # Legacy threshold path: per-iter rule OR cum (V21). Legacy
                # score/threshold computation untouched; equal when cum low.
                try:
                    _legacy_pass = bool(gate["score"] >= cfg.gate_threshold)
                except Exception:
                    _legacy_pass = bool(gate.get("sprt_pass", False))
                gate_pass = bool(_v21dec.get("sprt_pass_cum", _legacy_pass))
            arena_score = _arena_score(arena)
            # Non-regression with tolerance for 30-game noise (±60 Elo ≈
            # ±0.08 score). Do not let one noisy arena veto a clear gate,
            # but do not promote on gate alone when the greedy line fell.
            arena_ok = arena_score >= incumb_arena_score - 0.05
            promoted = bool(gate_pass and arena_ok)
            _progress = None
            _anti_rps = None
            if promoted and _v19:
                # D10 anti-RPS veto: every promotion ALSO plays 4 games vs
                # the promo-3 ancestor; score < 40% HOLDS the promotion
                # (beats-incumbent-but-loses-to-grandparent = RPS cycle).
                # Runs BEFORE any incumb files are rewritten.
                _chall_w = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
                _anti_rps = _anti_rps_check(
                    _chall_w, ckpt_dir, cfg, cfg.sims, device,
                    seed_base=gate_seed + 777)
                if _anti_rps.get("hold"):
                    print(f"[iter {it}] panel+arena passed but anti-RPS "
                          f"vetoed (score {_anti_rps.get('score')} < 0.40) — "
                          f"HOLDING promotion", flush=True)
                    promoted = False
            gate["anti_rps"] = _anti_rps
            gate["pass"] = bool(gate_pass)
            gate["arena_score"] = round(arena_score, 4)
            gate["arena_ok"] = bool(arena_ok)
            gate["incumbent_arena"] = round(incumb_arena_score, 4)
            if gate_pass and not arena_ok:
                print(f"[iter {it}] gate passed ({gate['score']}) but arena "
                      f"regressed ({arena_score:.3f} vs incumb "
                      f"{incumb_arena_score:.3f}) — holding", flush=True)
            # V21: reset BOTH cums on any overall promotion (consume evidence).
            # History/meta below persist the reset 0.0; legs stay True (fired).
            if promoted:
                cum_llr, cum_llr_eg = 0.0, 0.0
                try:
                    gate["cum_llr"] = 0.0
                    gate["cum_llr_eg"] = 0.0
                except Exception:
                    pass
                print(f"[iter {it}] v21 cum LLR reset (promotion)", flush=True)
            if promoted:
                incumb_arena_score = arena_score
            if promoted:
                incumb_w = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
                # audit round 2 §1.8: champion files carry opt+sched so ANY
                # resume (watchdog included) restores the schedule instead of
                # restarting milestones. In-memory incumb_w stays a raw state
                # dict for the strict loaders; all FILE readers unwrap via
                # load_weights (tolerant) or copy bytes (eval_watch).
                atomic_save({"weights": incumb_w,
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict()}, incumb_path)
                # lineage cop: promo_seq counts promotions across restarts
                # (persisted in best.pt meta); every Nth promotion snapshots
                # and plays the champion from N promotions ago.
                _promos_this_run += 1
                _seq = _promo_base + _promos_this_run
                atomic_save({"weights": incumb_w, "meta": {"iter": it,
                            "promo_seq": _seq,
                            # D17: LR persist — resume prefers this meta
                            # (V20 adds cum_positions for the D9 schedule).
                            # V21 adds cum_llr/cum_llr_eg (reset 0.0 here).
                            "cum_steps": cum_steps, "cum_games": cum_games,
                            "cum_positions": cum_positions,
                            "cum_llr": float(cum_llr),
                            "cum_llr_eg": float(cum_llr_eg),
                            "lr": lr},
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict()},
                           os.path.join(ckpt_dir, "best.pt"))
                _champ_dir = os.path.join(ckpt_dir, "champions")
                try:
                    os.makedirs(_champ_dir, exist_ok=True)
                    atomic_save({"weights": incumb_w,
                                "meta": {"iter": it, "promo_seq": _seq}},
                               os.path.join(_champ_dir,
                                            f"champ_seq{_seq}.pt"))
                except Exception as e:
                    print(f"[iter {it}] champ snapshot failed "
                          f"({type(e).__name__})", flush=True)
                _every = int(getattr(cfg, "progress_check_every", 3) or 3)
                if _seq % _every == 0:
                    _old_snap = os.path.join(
                        _champ_dir, f"champ_seq{_seq - _every}.pt")
                    if os.path.exists(_old_snap):
                        try:
                            _old_meta = torch.load(
                                _old_snap, map_location="cpu",
                                weights_only=False).get("meta", {}) or {}
                            _old_iter = int(_old_meta.get("iter", 0))
                        except Exception:
                            _old_iter = 0
                        print(f"[iter {it}] lineage check: champ seq {_seq} "
                              f"(iter {it}) vs seq {_seq - _every} (iter "
                              f"{_old_iter})", flush=True)
                        _progress = _progress_check(
                            os.path.join(_champ_dir, f"champ_seq{_seq}.pt"),
                            _old_snap, cfg, cfg.sims, workers, ckpt_dir, it,
                            _old_iter, seed_base=71000 + _seq, sf_path=sf_path)
                        print(f"[iter {it}] lineage check: {_progress}",
                              flush=True)
                    else:
                        _progress = {"note": "no baseline snapshot yet"}
        else:
            # Skipping evaluation supplies no evidence and cannot promote.
            arena = {"evaluation_skipped": True}
            gate = {"skipped": True, "via": "scheduled-skip", "score": None,
                    "llr": None, "sprt_pass": False, "pass": False}
            promoted = False
            _progress = None
            phase_seconds["arena"] = 0.0
            phase_start = time.perf_counter()
            print(f"[iter {it}] promotion checks skipped (every "
                  f"{cfg.gate_every}); checkpoint will be saved", flush=True)
        if cfg.adaptive_contempt:
            # game keys only: vetoes/tactics are counters, not games.
            # v14 fix: plain keys count every game exactly once (sparring
            # games increment BOTH plain and S* keys). Summing S* too
            # double-counted sparring and steered contempt ~20% low.
            tot_g = sum(results.get(k, 0) for k in
                        ("1-0", "0-1", "1/2-1/2"))
            assert tot_g == games_per_iter, (tot_g, games_per_iter,
                                             dict(results))
            ds = results.get("1/2-1/2", 0) / max(tot_g, 1)
            new_c = adapt_contempt(float(cfg.contempt), ds)
            if new_c != float(cfg.contempt):
                print(f"[iter {it}] draw share {ds:.2f}: contempt "
                      f"{float(cfg.contempt):.2f} -> {new_c:.2f}", flush=True)
                cfg.contempt = new_c
        values_log = None
        if cfg.empirical_values and it >= 5 and it % cfg.values_every == 0 \
                and len(buf) >= 2000:
            from .values import fit_values, update_values, \
                decisive_samples
            import chess_zero.game as _g
            import random as _r
            sample = _r.sample(list(buf.buf),
                               min(20000, len(buf.buf)))
            # audit round 2 §2.7: fit on DECISIVE games only. Draw labels
            # are contempt-shaped by construction (ahead pays +c), so
            # regressing on them partly measures our own labels (the Q=4.7
            # collapse). Decisive labels (mate/adjudication/resign) are
            # unshaped.
            dec = decisive_samples(sample)
            fitted = fit_values(dec) if len(dec) >= 2000 else None
            new_vals = update_values(dict(_g.ADJUDICATE_VALUES), fitted)
            _g.ADJUDICATE_VALUES.update(new_vals)
            values_log = {{1: "P", 2: "N", 3: "B", 4: "R", 5: "Q"}.get(
                k, str(k)): round(v, 3) for k, v in new_vals.items()}
            print(f"[iter {it}] empirical values: {values_log}", flush=True)
        # v14 fix: self-play only. Sparring games (White often the fixed
        # greedy/punisher) polluted the color-balance read.
        _sw = results.get("1-0", 0) - results.get("S1-0", 0)
        _sb = results.get("0-1", 0) - results.get("S0-1", 0)
        _dec = _sw + _sb
        _wshare = round(_sw / max(_dec, 1), 3) if _dec else None
        # v13 honest compass: how self-play games ended. Mate rate,
        # conversion pressure and decisive length are the primary
        # strategy signals; greedy arena is secondary.
        _term = {k: int(results.get("T" + k, 0)) for k in
                 ("mate", "resign", "adjudicated", "adjudicated-draw",
                  "rules-draw", "cap")}
        _nterm = sum(_term.values())
        # v14 visit breadth: mean distinct root moves visited per search.
        # Rises with sims if the peaked-prior trap is opening; flat means
        # deep moves still starve (then: temp/dirichlet/c_puct).
        _nbg = max(int(results.get("bgames", 0)), 1)
        # D15 test losses (one no-grad batch/iter; None when the
        # replay-owned test_buf is absent) + D18 BN/plane alarms (drift
        # vs the previous iter) + D10 opening entropy (self-play move
        # lists are unavailable to the loop — play_game returns encoded
        # examples only — so this stays None until Impl-B captures
        # 2-ply prefixes; the helper + alarm rule are tested).
        _test_rep = _test_losses(model, buf, cfg, device) if _v19 else \
            {"test_loss": None, "test_loss_p": None,
             "test_loss_v": None}
        _bn_rep = bn_plane_report(model, _bn_prev) if _v19 else None
        if _v19:
            _bn_prev = _bn_rep
            if (results.get("pfsp_pool", 0) or
                    results.get("pfsp_punisher", 0)):
                print(f"[iter {it}] PFSP kinds: self="
                      f"{games_per_iter - results.get('pfsp_pool', 0) - results.get('pfsp_punisher', 0)} "
                      f"pool={results.get('pfsp_pool', 0)} "
                      f"punisher={results.get('pfsp_punisher', 0)}",
                      flush=True)
        _opening = {"top3": None, "n": 0,
                    "note": "self-play 2-ply prefixes unavailable in "
                            "loop (play_game returns examples, not moves; "
                            "Impl-B capture needed)"}
        entry = {"iter": it, "games": dict(results),
                  "vetoes": results.get("vetoes", 0),
                  "tactics": results.get("tactics", 0),
                  "finishes": int(results.get("finishes", 0)),
                  "safety": int(results.get("safety", 0)),
                  "progress": _progress,
                  "term": _term,
                  "mate_rate": round(_term["mate"] / max(_nterm, 1), 3),
                  "breadth": round(float(results.get("breadth", 0.0))
                                    / _nbg, 2),
                  "ventropy": round(float(results.get("ventropy", 0.0))
                                     / _nbg, 3),
                  # v14 pace metric: total self-play plies this iter (the
                  # throughput that compounds — games x plies per wall).
                  "plies": int(results.get("plies", 0)),
                 "buf": len(buf), "secs": round(time.time() - t0, 1),
                 "phase_seconds": {**phase_seconds, "gate_and_diagnostics": time.perf_counter() - phase_start},
                 "lr": lr, "contempt": round(float(cfg.contempt), 3),
                 "white_share": _wshare,
                 "train_steps": eff_steps,
                 "loss": round(tot_acc / max(n, 1), 4),
                 "loss_p": round(pl_acc / max(n, 1), 4),
                 "loss_v": round(vl_acc / max(n, 1), 4),
                 "loss_a": round(al_acc / max(n, 1), 4),
                 "loss_ml": round(ml_acc / max(n, 1), 4),
                 "loss_e": round(en_acc / max(n, 1), 4),
                 "loss_o": round(ol_acc / max(n, 1), 4),
                 "loss_s": round(sl_acc / max(n, 1), 4),
                  "loss_b": round(bl_acc / max(n, 1), 4),
                  "loss_k": round(kl_acc / max(n, 1), 4),
                  "loss_r": round(rl_acc / max(n, 1), 4),
                   "loss_tkl": round(tkl_acc / max(n, 1), 4),
                   "loss_prog": round(prog_acc / max(n, 1), 4),
                   # V20 D1/D2/D3 loss telemetry (0.0 until the ramp /
                   # AV labels / TD blend engage; KL stays LAST in the
                   # train_step tuple, these ride BEFORE it).
                   "loss_score": round(sc_acc / max(n, 1), 4),
                   "loss_av": round(avv_acc / max(n, 1), 4),
                    "loss_td": round(td_acc / max(n, 1), 4),
                    "cum_steps": cum_steps, "cum_games": cum_games,
                    "cum_positions": cum_positions,
                    # V21 cumulative LLR (OR-path, future-only): persisted
                    # top-level for resume (reset 0.0 on promotion above).
                    "cum_llr": float(cum_llr),
                    "cum_llr_eg": float(cum_llr_eg),
                    # V20 D4/D5/D6 self-play telemetry (resume source for
                    # cum_positions via "positions").
                   "positions": int(results.get("positions", 0)),
                   "endgame_starts": int(results.get("endgame_starts", 0)),
                   "playthroughs": int(results.get("playthroughs", 0)),
                   "tb": {k: int(results.get("tb_" + k, 0)) for k in
                          ("probed", "guarded", "rewritten",
                           "deblundered", "skipped", "skipped_book")},
                  "value_w": round(value_w, 4),
                  "gate": gate, "promoted": promoted,
                  # history contract: top-level LLR + opening entropy +
                  # test losses + BN/plane alarms (persist in best.pt
                  # meta + iter files for resume via cum_* above).
                  "llr": gate.get("llr"),
                  "opening_top3": _opening,
                  "test": _test_rep,
                  "bn": _bn_rep,
                  "values": values_log,
                  **arena}
        history.append(entry)
        print(f"[iter {it}] {entry}", flush=True)
        # v14: no gradient steps, no schedule burn (milestones count
        # real training, not iters). Neutral at LR floor; correct going up.
        # v19 C3: the game-budget LR is set directly on both groups above;
        # V20 D9: same for the position-counted LR. The milestone
        # scheduler object stays for ckpt save/load compat but is never
        # stepped. Non-v19/v20 keeps milestone stepping.
        if n > 0 and not _v19 and not _v20:
            sched.step()  # after optimizer steps: next iter uses decayed LR
        save_checkpoint(model, opt, os.path.join(ckpt_dir, f"iter{it}.pt"),
                        entry, sched=sched, replay=buf, cfg=cfg)
        with open(os.path.join(ckpt_dir, log_name), "w") as f:
            json.dump(history, f, indent=1)
    # The champion artifact keeps the metadata of the promotion that
    # produced its weights. Iteration files own optimizer/replay progress.
    champion_path = os.path.join(ckpt_dir, "incumbent.pt")
    champion = torch.load(champion_path, map_location="cpu", weights_only=False)
    if not isinstance(champion, dict) or "weights" not in champion:
        champion = {"weights": incumb_w, "meta": {"role": "champion"}}
    champion.pop("opt", None)
    champion.pop("sched", None)
    atomic_save(champion, os.path.join(ckpt_dir, final_ckpt))
    return history


# ---------------------------------------------------------------------------
# D8 panel gate + D9 PFSP pool + D10 anti-RPS (Impl-D second sweep).
# Panel (20 games): 8 vs incumbent + 6 vs ancestor (promo-3 snapshot,
# fallback incumbent) + 4 vs punisher + 2 vs random. Promote iff
# overall score >= 60% AND >= 55% vs EACH of incumbent/ancestor AND the
# Wilson lower-95 of the overall score > 50%. Arena corroboration
# conjunction (caller) is UNCHANGED. All agent games run noise=0.0 (D20:
# deterministic measurement; selection recall comes from the panel +
# Wilson design, not noise).
# ---------------------------------------------------------------------------

PANEL_SPLITS = (("incumbent", 8), ("ancestor", 6), ("punisher", 4),
                ("random", 2))


def wilson_lower(wins: float, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound for the overall gate score, draws counting
    half (score = (W + D/2) / n; the Bernoulli approximation is standard
    practice for game scores). n <= 0 -> 0.0 (never promotes on nothing)."""
    try:
        _n = int(n)
        if _n <= 0:
            return 0.0
        _s = min(max(float(wins) / _n, 0.0), 1.0)
        _z = float(z)
        _den = 1.0 + _z * _z / _n
        _mid = _s + _z * _z / (2.0 * _n)
        _rad = _z * math.sqrt(_s * (1.0 - _s) / _n +
                              _z * _z / (4.0 * _n * _n))
        return max(0.0, (_mid - _rad) / _den)
    except Exception:
        return 0.0


def _elo_los(W, L, D):
    """Logistic Elo + one-sided 95% lower bound + LOS (likelihood of
    superiority) via the normal approximation on game score (Fishtest
    stat_util family). N=0 or degenerate -> (0.0, 0.0, 0.5). Never raises.
    Eval-science: a 20-game point estimate (±140 at low draws) is NOT
    evidence; LOS>=95% AND Elo_low>0 is the promotion-grade bar."""
    import math as _m
    try:
        _W, _L, _D = float(W), float(L), float(D)
        _n = _W + _L + _D
        if _n <= 0:
            return 0.0, 0.0, 0.5
        _X = (_W + 0.5 * _D) / _n
        if _X <= 0.0 or _X >= 1.0:
            return (999.0 if _X >= 1.0 else -999.0), \
                (999.0 if _X >= 1.0 else -999.0), \
                (1.0 if _X >= 1.0 else 0.0)
        _var = ((_W / _n) + (_D / (4.0 * _n)) - _X * _X) / _n
        if _var <= 1e-12:
            return 0.0, 0.0, 0.5
        _se = _m.sqrt(_var)
        _elo = -400.0 * _m.log10(1.0 / _X - 1.0)
        _se_elo = 400.0 / _m.log(10.0) * _se / (_X * (1.0 - _X))
        _low = _elo - 1.645 * _se_elo
        _los = 0.5 * (1.0 + _m.erf((_X - 0.5) / (_se * _m.sqrt(2.0))))
        return round(_elo, 1), round(_low, 1), round(_los, 4)
    except Exception:
        return 0.0, 0.0, 0.5


def _panel_decide(splits: dict, cfg=None) -> dict:
    """D8 panel decision from per-split (W, L, D). Promote iff overall
    score >= 0.60 AND incumbent split >= 0.55 AND ancestor split >= 0.55
    AND Wilson lower-95 (overall) > 0.50 AND LOS >= 0.95 AND Elo
    lower-95 > 0 (eval-science: N=20 point estimates are ±140 noise;
    direction + magnitude confidence both required). Returns the decision
    dict (also carries the overall SPRT LLR for the history log)."""
    def _score(_wld):
        _w, _l, _d = (float(_wld[0]), float(_wld[1]), float(_wld[2]))
        _n = _w + _l + _d
        return ((_w + 0.5 * _d) / _n) if _n > 0 else 0.0
    # Only head-to-head incumbent games constitute superiority evidence.
    # Ancestor/weak-bot splits remain separate regression diagnostics.
    _W, _L, _D = map(float, splits.get("incumbent", (0, 0, 0)))
    _n = _W + _L + _D
    _ov = (_W + 0.5 * _D) / _n if _n > 0 else 0.0
    _inc = _score(splits.get("incumbent", (0, 0, 0)))
    _anc = _score(splits.get("ancestor", (0, 0, 0)))
    _wil = wilson_lower(_W + 0.5 * _D, int(_n))
    _elo, _elow, _los = _elo_los(_W, _L, _D)
    try:
        _llr = gsprt_llr(_W, _L, _D,
                         float(getattr(cfg, "sprt_elo0", 0.0)),
                         float(getattr(cfg, "sprt_elo1", 30.0)))
    except Exception:
        _llr = 0.0
    _ok = bool(_ov >= 0.60 and _inc >= 0.55 and _anc >= 0.55 and
               _wil > 0.50 and _los >= 0.95 and _elow > 0.0)
    return {"promote": _ok, "score": round(_ov, 4),
            "incumbent": round(_inc, 4), "ancestor": round(_anc, 4),
            "wilson": round(_wil, 4), "llr": round(_llr, 4),
            "elo": _elo, "elo_low": _elow, "los": _los,
            "wld": [int(_W), int(_L), int(_D)]}


def _champ_seq_files(ckpt_dir: str) -> list:
    """Sorted [(seq, path)] of champions/champ_seqN.pt. Never raises."""
    import re as _re
    out = []
    try:
        _d = os.path.join(ckpt_dir, "champions")
        for _f in os.listdir(_d):
            _m = _re.match(r"champ_seq(\d+)\.pt$", _f)
            if _m:
                out.append((int(_m.group(1)), os.path.join(_d, _f)))
    except Exception:
        pass
    return sorted(out)


def _ancestor_weights(ckpt_dir: str, incumb_w):
    """D8/D10 promo-3 ancestor: 3rd-from-latest champion snapshot
    (latest = current incumbent's seq N -> ancestor N-3). Fewer than 4
    snapshots -> fallback incumbent (loud log, never raises). Returns
    (weights_or_path, note)."""
    try:
        _files = _champ_seq_files(ckpt_dir)
        if len(_files) >= 4:
            _seq, _path = _files[-4]
            print(f"[panel] ancestor: champ_seq{_seq}.pt "
                  f"(promo-3 of latest seq {_files[-1][0]})", flush=True)
            return _path, f"champ_seq{_seq}"
    except Exception as _e:
        print(f"[panel] ancestor lookup failed "
              f"({type(_e).__name__}); fallback incumbent", flush=True)
        return incumb_w, "fallback-incumbent(lookup-error)"
    print(f"[panel] ancestor: only "
          f"{len(_champ_seq_files(ckpt_dir))} snapshots (< 4 needed); "
          f"fallback incumbent", flush=True)
    return incumb_w, "fallback-incumbent"


def _pool_files(ckpt_dir: str, k: int = 4) -> list:
    """D9 PFSP pool: paths of the last k champion snapshots (uniform
    sampling by the caller). Empty when no snapshots exist yet."""
    try:
        return [_p for _, _p in _champ_seq_files(ckpt_dir)[-int(k):]]
    except Exception:
        return []


def _pfsp_kinds(n_games: int) -> list:
    """D9 PFSP-lite game-kind rotation: 12 self / 5 pool / 3 punisher of
    every 20 games (scaled proportionally for other n). Deterministic by
    game index (no RNG): kinds[i] for i in range(n_games)."""
    try:
        _n = max(0, int(n_games))
    except Exception:
        return []
    if _n <= 0:
        return []
    _n_self = int(round(_n * 12.0 / 20.0))
    _n_pool = int(round(_n * 5.0 / 20.0))
    _n_self = min(_n_self, _n)
    _n_pool = min(_n_pool, _n - _n_self)
    _n_pun = _n - _n_self - _n_pool
    return (["self"] * _n_self + ["pool"] * _n_pool +
            ["punisher"] * _n_pun)


def _roll_book_game(rng, frac: float) -> bool:
    """D12 forced-book coin: True (open from the train-split book) with
    probability frac. Pure helper so the 30% rule is unit-testable; the
    selfplay/parallel application is Impl-B's file."""
    try:
        return float(rng.random()) < float(frac)
    except Exception:
        return False


def _weights_dict(w):
    """Resolve agent_from_weights input to a state-dict-like: dicts pass
    through, paths are torch.loaded (full ckpts {"weights":...} stay
    wrapped — model.load_weights unwraps, same as _progress_check)."""
    if isinstance(w, dict):
        return w
    return torch.load(w, map_location="cpu", weights_only=False)


def _panel_gate(chall_w, incumb_w, cfg, sims, device, workers, ckpt_dir,
                seed_base=1000, use_server=False, server_device="cuda",
                rust_tree: bool = False):
    """D8 panel match runner. Builds challenger/incumbent/ancestor agents
    with noise=0.0 (D20); punisher/random are the builtin policies. Each
    split runs as sequential ids (2k/2k+1 share one seeded opening with
    colors swapped = mirrored, colors alternate adjacently). Per-split
    seeds derive from the reshuffled holdout-book draw (D11; observable,
    varying per gate). Never raises: total failure -> threshold-style
    hold dict (loud log)."""
    from .evaluate import (play_match, random_move, punisher_move)
    opening = int(getattr(cfg, "gate_opening_moves", 6) or 0)
    lines, binfo = _gate_book_draw(seed_base, cfg)
    print(f"[panel] paired incumbent test plus ancestor/weak-bot diagnostics, noise=0.0, "
          f"book holdout lines={binfo.get('n_holdout', '?')} "
          f"draw={binfo['draw']} fallback={binfo['fallback']}", flush=True)
    if getattr(cfg, "gate_book_file", "") and not lines:
        return {"challenger_wld": [0,0,0], "score": 0.0, "llr": 0.0,
                "sprt_pass": False, "via": "panel-book-error", "panel": {}}
    _anc_w, _anc_note = _ancestor_weights(ckpt_dir, incumb_w)
    # V21.1 E4: explicit param OR cfg knob (run_training stamps cfg).
    _rt = bool(rust_tree or getattr(cfg, "rust_tree", False))
    try:
        ch = agent_from_weights(_weights_dict(chall_w), cfg, sims=sims,
                                device=device, noise=0.0, rust_tree=_rt)
        inc = agent_from_weights(_weights_dict(incumb_w), cfg, sims=sims,
                                 device=device, noise=0.0, rust_tree=_rt)
        anc = agent_from_weights(_weights_dict(_anc_w), cfg, sims=sims,
                                 device=device, noise=0.0, rust_tree=_rt)
    except Exception as _e:
        print(f"[panel] agent build failed ({type(_e).__name__}: "
              f"{str(_e)[:160]}); HOLD", flush=True)
        return {"challenger_wld": [0, 0, 0], "score": 0.0, "llr": 0.0,
                "sprt_pass": False, "via": "panel-build-error",
                "panel": {}, "wilson": 0.0, "ancestor": _anc_note,
                "book": binfo["draw"]}
    _opps = {"incumbent": inc, "ancestor": anc, "punisher": punisher_move,
             "random": random_move}
    splits = {}
    incumbent_pair_scores = []
    try:
        if workers > 1:
            from .parallel import play_match_parallel
        else:
            play_match_parallel = None
        for _si, (_name, _ng) in enumerate(PANEL_SPLITS):
            if _name in ("punisher", "random") and not getattr(cfg, "panel_weak_diagnostics", True):
                continue
            if _name == "incumbent":
                _ng = max(2, int(getattr(cfg, "gate_incumbent_games", 64)))
                _ng += _ng % 2
                if lines:
                    _ng = min(_ng, 2 * len(lines))
            _line = lines[_si % len(lines)] if lines else None
            _ss = _pair_seed(int(seed_base) + _si * 100003, 0, _line)
            _opp = _opps[_name]
            if play_match_parallel is not None:
                _tot, _recs = play_match_parallel(
                    ch, _opp, games=_ng,
                    workers=min(workers, _ng), opening_moves=opening,
                    game_settings=game_settings(cfg), record=True,
                    seed_base=_ss, opening_lines=(lines if _name == "incumbent" else [_line]) if _line else None, use_server=use_server,
                    server_device=server_device)
            else:
                _tot, _recs = play_match(
                    ch, _opp, games=_ng, opening_moves=opening,
                    record=True, seed_base=_ss, opening_lines=(lines if _name == "incumbent" else [_line]) if _line else None)
            if _name == "incumbent":
                scores = [0.5 if rec[1] is None else float(bool(rec[1])) for rec in _recs]
                incumbent_pair_scores = [(scores[i]+scores[i+1])/2 for i in range(0,len(scores)-1,2)]
            splits[_name] = (_tot["wins"], _tot["losses"],
                             _tot["draws"])
            _sc = (splits[_name][0] + 0.5 * splits[_name][2]) / max(
                1, sum(splits[_name]))
            print(f"[panel] vs {_name} ({_ng}g, seed {_ss}): "
                  f"WLD={splits[_name]} score={_sc:.3f}", flush=True)
    except Exception as _e:
        print(f"[panel] match failed ({type(_e).__name__}: "
              f"{str(_e)[:160]}); HOLD", flush=True)
        return {"challenger_wld": [0, 0, 0], "score": 0.0, "llr": 0.0,
                "sprt_pass": False, "via": "panel-match-error",
                "panel": {}, "wilson": 0.0, "ancestor": _anc_note,
                "book": binfo["draw"]}
    dec = _panel_decide(splits, cfg)
    from .experiment_support import paired_superiority
    paired = paired_superiority(incumbent_pair_scores, float(getattr(cfg, "gate_alpha", .05)))
    required = max(1, int(getattr(cfg, "gate_min_pairs", 1)))
    dec["promote"] = bool(paired["pass"] and paired["pairs"] >= required and dec["ancestor"] >= .5)
    dec["paired_lower"] = None  # legacy metadata, superseded by exact paired p-value
    print(f"[panel] fixed paired test: n={paired['pairs']}/{required} "
          f"p={paired['p_value']:.6g} alpha={paired.get('alpha', .05)}", flush=True)
    print(f"[panel] overall WLD={dec['wld']} score={dec['score']} "
          f"inc={dec['incumbent']} anc={dec['ancestor']} "
          f"wilson={dec['wilson']} llr={dec['llr']} -> "
          f"{'PROMOTE' if dec['promote'] else 'HOLD'}", flush=True)
    return {"challenger_wld": dec["wld"], "score": dec["score"],
            "llr": dec["llr"], "sprt_pass": bool(dec["promote"]),
            "via": "panel", "paired_test": paired,
            "panel": {k: {"wld": [int(v[0]), int(v[1]), int(v[2])],
                          "score": round((v[0] + 0.5 * v[2]) /
                                         max(1, v[0] + v[1] + v[2]), 4)}
                      for k, v in splits.items()},
            "wilson": dec["wilson"], "ancestor": _anc_note,
            "book": binfo["draw"],
            "paired_lower": dec.get("paired_lower"),
            "incumbent_pair_scores": incumbent_pair_scores}


# ---------------------------------------------------------------------------
# V20 D4/D9 endgame panel (Agent B): 8 mirrored pairs from the endgame
# list at endgame_panel_sims (400), ANDed with the main panel in the
# promotion rule. Mirrored = same FEN with colors swapped (challenger
# white, then challenger black). Pass bar: score >= endgame_panel_min
# (0.55, same family as the panel split bars; direction+technique, not
# a second full gate — LOS/Elo bars live on the main panel, unchanged).
# ---------------------------------------------------------------------------

ENDGAME_FALLBACK_FENS = (
    "8/8/4k3/8/8/3KQ3/8/8 w - - 0 1",  # KQvK
    "8/8/4k3/8/8/3KR3/8/8 w - - 0 1",  # KRvK
    "8/8/4k3/8/3P4/8/4K3/8 w - - 0 1",  # KPvK key square
    "1R6/8/5k2/8/8/5K2/8/8 w - - 0 1",  # Lucena seed
)


def _endgame_panel_decide(W, L, D, cfg=None) -> dict:
    """Pass iff overall score >= endgame_panel_min (default 0.55).
    V21 adds "llr" via eg_panel_llr (H0 50% vs H1 60%); legacy keys
    pass/score/wld/min are bit-identical (additive only)."""
    try:
        _min = float(getattr(cfg, "endgame_panel_min", 0.50)
                     if cfg is not None else 0.55)
    except Exception:
        _min = 0.55
    _n = float(W) + float(L) + float(D)
    _sc = ((float(W) + 0.5 * float(D)) / _n) if _n > 0 else 0.0
    try:
        _ellr = round(float(eg_panel_llr(W, L, D)), 4)
    except Exception:
        _ellr = 0.0
    return {"pass": bool(_n > 0 and _sc >= _min),
            "score": round(_sc, 4),
            "wld": [int(W), int(L), int(D)], "min": _min,
            "llr": _ellr}


@__import__("chess_zero.game", fromlist=["standard_rules"]).standard_rules
def _play_from_fen(white_pol, black_pol, fen: str, cap: int = 300):
    """Play one game from a FEN to termination/cap. Returns (w_result,
    n_plies): w_result in {1.0, -1.0, 0.0} white-relative (selfplay
    convention). Policies are state->action closures (reset when they
    carry one). Never raises (error -> draw)."""
    try:
        import chess as _c
        from .game import State as _S
        _st = _S(_c.Board(str(fen)))
        for _pol in (white_pol, black_pol):
            try:
                _rst = getattr(_pol, "reset", None)
                if callable(_rst):
                    _rst()
            except Exception:
                pass
            try:
                _pol._hist.append(_st.rep_key())
            except Exception:
                pass
        _n = 0
        while _n < int(cap):
            try:
                _done, _ = _st.is_terminal()
            except Exception:
                _done = False
            if _done:
                break
            try:
                _pol = white_pol if _st.board.turn == _c.WHITE \
                    else black_pol
                _st = _st.apply(int(_pol(_st)))
            except Exception as exc:
                raise RuntimeError("Endgame policy failed") from exc
            _n += 1
        try:
            _done, _z = _st.is_terminal()
        except Exception:
            return 0.0, _n
        if not _done or _z == 0.0:
            return 0.0, _n
        import chess as _c2
        return (_z if _st.board.turn == _c2.WHITE else -_z), _n
    except Exception as exc:
        raise RuntimeError("Invalid endgame measurement") from exc


def _endgame_game_seed(seed_base: int, pair_idx: int, leg: int) -> int:
    """Per-game explicit seed: seed_base + pair index + color leg.

    leg 0 = challenger white, leg 1 = challenger black. Linear (not
    hashed) so the derivation is auditable; sequential and parallel
    paths call this identically, hence BIT-IDENTICAL WLD on fixed FENs.
    With noise=0.0 deterministic policies the seed is a no-op for the
    game itself (MCTS uses no RNG at eps=0, choice is argmax) — it only
    pins any ambient RNG state per game so worker fan-out cannot leak
    ordering effects."""
    return int(seed_base) + int(pair_idx) * 2 + int(leg)


def _seed_endgame_rngs(seed: int) -> None:
    """Pin ambient RNGs (random/numpy/torch) to a per-game seed. Never
    raises; best-effort isolation so parallel workers and the sequential
    loop observe identical RNG state per game."""
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed) % (2 ** 32))
    except Exception:
        pass
    try:
        torch.manual_seed(int(seed) % (2 ** 32))
    except Exception:
        pass


def _endgame_panel_gate(chall_w, incumb_w, cfg, device="cpu",
                        seed_base=2000, rust_tree: bool = False,
                        workers: int | None = None) -> dict:
    """D4 endgame panel: endgame_panel_pairs mirrored FEN pairs at
    endgame_panel_sims sims, noise 0. FENs from cfg.endgame_file
    (uniform, seed-shuffled); file missing/empty -> embedded fallback
    quartet cycled (loud log, never blocks). Returns
    {"wld","score","pass","sims","pairs","fens","note"[, "llr"]}.
    V21 adds "llr" (eg_panel_llr, additive; legacy keys bit-identical).
    Parallel (default OFF): workers>1 fans out one job per mirrored
    pair across the existing ProcessPool (same spawn/_worker_init
    discipline as _panel_gate's play_match_parallel); each worker
    rebuilds its own agents (fresh TT/history, no shared mutable
    state) and seeds each leg via _endgame_game_seed(seed_base,
    pair_idx, leg), so WLD is BIT-IDENTICAL to sequential on fixed
    FENs. workers==1 (or None with cfg endgame_panel_workers 0/1) runs
    the sequential loop below (yesterday's code + per-game seeding,
    a no-op at noise=0.0 deterministic measurement). Any worker error
    falls back to sequential (loud log). Buffer/training untouched:
    returns result tallies only, never examples. Never raises (total
    failure -> pass False + loud log: a broken measurement must never
    promote by itself)."""
    try:
        _pairs = max(1, int(getattr(cfg, "endgame_panel_pairs", 8) or 8))
    except Exception:
        _pairs = 8
    try:
        _sims = max(1, int(getattr(cfg, "endgame_panel_sims", 400)
                           or 400))
    except Exception:
        _sims = 400
    try:
        from .selfplay import load_endgame_fens as _lef
        _fens = _lef(str(getattr(cfg, "endgame_file",
                                 "data_sl/endgames.jsonl")), partition="eval")
    except Exception as _e:
        print(f"[endgame-panel] list load failed "
              f"({type(_e).__name__}); fallback FENs", flush=True)
        _fens = []
    _note = "harvested"
    if not _fens:
        _fens = list(ENDGAME_FALLBACK_FENS)
        _note = "fallback-quartets (endgame file missing/empty)"
        print(f"[endgame-panel] {len(ENDGAME_FALLBACK_FENS)} fallback "
              f"FENs (no harvested list)", flush=True)
    try:
        _rng = random.Random(int(seed_base))
        _order = [_fens[_rng.randrange(len(_fens))]
                  for _ in range(_pairs)]
    except Exception:
        _order = [_fens[i % len(_fens)] for i in range(_pairs)]
    try:
        _rt = bool(rust_tree or getattr(cfg, "rust_tree", False))
        ch = agent_from_weights(_weights_dict(chall_w), cfg, sims=_sims,
                                device=device, noise=0.0, rust_tree=_rt)
        inc = agent_from_weights(_weights_dict(incumb_w), cfg, sims=_sims,
                                 device=device, noise=0.0, rust_tree=_rt)
    except Exception as _e:
        print(f"[endgame-panel] agent build failed "
              f"({type(_e).__name__}); HOLD", flush=True)
        return {"wld": [0, 0, 0], "score": 0.0, "pass": False,
                "sims": _sims, "pairs": _pairs, "fens": [],
                "note": f"build-error-{type(_e).__name__}", "llr": 0.0}
    try:
        _cap = int(getattr(cfg, "move_cap", 300) or 300)
    except Exception:
        _cap = 300
    # Effective fan-out: explicit workers wins (tests), else the cfg
    # knob (default 0 = OFF = yesterday sequential). <=1 = sequential.
    try:
        _eff = int(workers) if workers is not None else int(
            getattr(cfg, "endgame_panel_workers", 0) or 0)
    except Exception:
        _eff = 0
    if _eff > 1:
        try:
            from .parallel import play_endgame_pairs_parallel as _egpar
            _par = _egpar(chall_w, incumb_w, cfg, _order, sims=_sims,
                          device=device, rust_tree=_rt, seed_base=seed_base,
                          cap=_cap, workers=min(int(_eff), len(_order)))
            _W, _L, _D = (int(_par.get("W", 0)), int(_par.get("L", 0)),
                           int(_par.get("D", 0)))
            _used = [str(_f)[:48] for _f in _order]
            _dec = _endgame_panel_decide(_W, _L, _D, cfg)
            print(f"[endgame-panel] {_pairs} mirrored pairs @ {_sims} "
                  f"sims ({_note}): WLD={_dec['wld']} "
                  f"score={_dec['score']} (min {_dec['min']}) -> "
                  f"{'PASS' if _dec['pass'] else 'HOLD'} "
                  f"[parallel workers={min(int(_eff), len(_order))}]",
                  flush=True)
            return {"wld": _dec["wld"], "score": _dec["score"],
                    "pass": bool(_dec["pass"]), "sims": _sims,
                    "pairs": _pairs, "fens": _used, "note": _note,
                    "llr": float(_dec.get("llr", 0.0))}
        except Exception as _e:
            print(f"[endgame-panel] parallel failed "
                  f"({type(_e).__name__}: {str(_e)[:120]}); "
                  f"sequential fallback", flush=True)
            # fall through to the sequential loop below (today's code)
    _W = _L = _D = 0
    _used = []
    for _pi, _fen in enumerate(_order):
        _used.append(str(_fen)[:48])
        # mirrored pair: challenger white, then challenger black.
        # Per-game explicit seeds (no-op at noise=0.0; pins ambient
        # RNG so parallel workers observe identical state per game).
        _seed_endgame_rngs(_endgame_game_seed(seed_base, _pi, 0))
        _w1, _ = _play_from_fen(ch, inc, _fen, _cap)
        if _w1 == 1.0:
            _W += 1
        elif _w1 == -1.0:
            _L += 1
        else:
            _D += 1
        _seed_endgame_rngs(_endgame_game_seed(seed_base, _pi, 1))
        _w2, _ = _play_from_fen(inc, ch, _fen, _cap)
        if _w2 == -1.0:
            _W += 1
        elif _w2 == 1.0:
            _L += 1
        else:
            _D += 1
    _dec = _endgame_panel_decide(_W, _L, _D, cfg)
    print(f"[endgame-panel] {_pairs} mirrored pairs @ {_sims} sims "
          f"({_note}): WLD={_dec['wld']} score={_dec['score']} "
          f"(min {_dec['min']}) -> "
          f"{'PASS' if _dec['pass'] else 'HOLD'}", flush=True)
    return {"wld": _dec["wld"], "score": _dec["score"],
            "pass": bool(_dec["pass"]), "sims": _sims,
            "pairs": _pairs, "fens": _used, "note": _note,
            "llr": float(_dec.get("llr", 0.0))}


def _gate(challenger_model, incumb_w, cfg, sims, device, workers, ckpt_dir,
          seed_base=1000, use_server=False, server_device="cuda",
          rust_tree: bool = False):
    # Noisy SELECTION test (dirichlet noise ON): high recall for spotting a
    # better challenger, ~36% false-promotion at 0.55 for equals (computed).
    # Promotions are selection signals, confirmed (or not) by the
    # deterministic arena line - never standalone evidence (audit round 2).
    """Head-to-head: challenger (policy A) vs incumbent. Score in [0,1].
    Never moves challenger_model (its optimizer state lives on its device).
    v19: mirrored SPRT pairs via _gate_sprt_pairs (C6/C7); the returned
    dict gains "llr"/"sprt_pass" keys, "score"/"challenger_wld" kept.
    """
    if _v19_loop(cfg):
        # D8 panel gate (replaces the single-incumbent 20): challenger
        # weights to a file for the parallel rebuilds; ALL agent games
        # run noise=0.0 (D20 deterministic measurement). Promotion =
        # panel rule AND arena_ok (conjunction lives with the caller).
        # (The C6 SPRT pair runner _gate_sprt_pairs stays defined for
        # reference/history compat; D8 supersedes it for v19 gates. The
        # overall SPRT LLR is still logged inside the panel dict.)
        ch_path0 = os.path.join(ckpt_dir, "_w_challenger.pt")
        atomic_save({k: v.cpu() for k, v in
                     challenger_model.state_dict().items()}, ch_path0)
        if _v20_loop(cfg):
            # V20 D4/D9: main panel AND endgame panel (8 mirrored pairs
            # from the endgame list at endgame_panel_sims). The panel
            # was added BEFORE endgame starts were enabled (D4 order);
            # a broken endgame measurement HOLDS (never promotes).
            # V21.1 E4: rust_tree from explicit param OR cfg knob.
            _rt = bool(rust_tree or getattr(cfg, "rust_tree", False))
            main = _panel_gate(ch_path0, incumb_w, cfg, sims, device,
                               workers, ckpt_dir, seed_base=seed_base,
                               use_server=use_server,
                               server_device=server_device, rust_tree=_rt)
            eg_cfg_w = int(getattr(cfg, "endgame_panel_workers", 0)
                             or 0)
            # Default OFF (0/1) = yesterday sequential even when the main
            # panel fans out: the live run is unaffected even on re-import.
            # Opt-in (>1): cap at the main pool size (existing worker pool).
            eg_workers = min(int(workers), eg_cfg_w) \
                if eg_cfg_w > 1 else 1
            eg = _endgame_panel_gate(ch_path0, incumb_w, cfg, device,
                                     seed_base=int(seed_base) + 7919,
                                     rust_tree=_rt, workers=eg_workers)
            sprt = bool(main.get("sprt_pass", False)) and \
                bool(eg.get("pass", False))
            print(f"[gate] v20 panel+endgame: main "
                  f"{main.get('score')} "
                  f"{'PASS' if main.get('sprt_pass') else 'HOLD'} AND "
                  f"endgame {eg.get('score')} "
                  f"{'PASS' if eg.get('pass') else 'HOLD'} -> "
                  f"{'PROMOTE' if sprt else 'HOLD'}", flush=True)
            out = dict(main)
            out["sprt_pass"] = sprt
            out["via"] = "panel+endgame"
            out["endgame_panel"] = eg
            # V21: preserve the per-leg per-iter verdicts (legacy
            # "sprt_pass" stays the AND; OR-path reads these, additive).
            out["main_sprt_pass"] = bool(main.get("sprt_pass", False))
            return out
        _rt2 = bool(rust_tree or getattr(cfg, "rust_tree", False))
        return _panel_gate(ch_path0, incumb_w, cfg, sims, device,
                           workers, ckpt_dir, seed_base=seed_base,
                           use_server=use_server,
                           server_device=server_device, rust_tree=_rt2)
    from .model import AlphaZeroNet as _Net
    ch_path = os.path.join(ckpt_dir, "_w_challenger.pt")
    atomic_save({k: v.cpu() for k, v in
                challenger_model.state_dict().items()}, ch_path)
    inc_path = os.path.join(ckpt_dir, "incumbent.pt")
    if workers > 1:
        from .model import load_weights as _lwg
        # V21.1 E4: explicit param OR cfg knob; appended as spec[27].
        _rt = bool(rust_tree or getattr(cfg, "rust_tree", False))
        ch_cpu = _Net(blocks=cfg.blocks, channels=cfg.channels,
                      planes=cfg.input_planes)
        # v14: tolerant loader (growth surgery) like every other reader —
        # strict load_state_dict crashed arch-change warm-starts here.
        _lwg(ch_cpu, torch.load(ch_path, map_location="cpu",
                                weights_only=False))
        ch = agent_policy(ch_cpu, cfg, sims=sims, device="cpu",
                          noise=cfg.dirichlet_eps, rust_tree=_rt)
        ch._spec = ("agent", ch_path, cfg.blocks, cfg.channels, sims,
                    cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_eps,
                    0, cfg.input_planes, True,
                    float(getattr(cfg, "contempt", 0.0) or 0.0),
                    bool(getattr(cfg, "asymmetric_contempt", False)),
                    bool(getattr(cfg, "tactical_override", False)),
                    bool(getattr(cfg, "blunder_veto", False)),
                    float(getattr(cfg, "tac_threshold", 0.09)),
                    int(getattr(cfg, "quiescence_depth", 0) or 0),
                    float(getattr(cfg, "forcing_bonus", 0.0) or 0.0),
                     float(getattr(cfg, "contempt_edge_scale", 0.0)
                            or 0.0),
                     # v15 finishing flag for parallel arena/gate rebuilds.
                     bool(getattr(cfg, "mate_finish", False)),
                     # v16.1 batched leaves for parallel arena/gate rebuilds.
                     int(getattr(cfg, "leaf_batch", 1) or 1),
                     float(getattr(cfg, "virtual_loss", 1.0) or 1.0),
                     # v15.5 safety veto for parallel arena/gate rebuilds.
                     bool(getattr(cfg, "safety_veto", False)),
                     # v18 SE trunk for parallel arena/gate rebuilds.
                     int(getattr(cfg, "se_ratio", 0) or 0),
                     # v19 search knobs for parallel rebuilds (0.0/False =
                     # yesterday bit-exact when unset).
                     float(getattr(cfg, "fpu_reduction", 0.0) or 0.0),
                     bool(getattr(cfg, "prune_singletons", False)),
                     # v19 safety threshold travels with the flag (deploy
                     # parity — bare namespaces used to pin 0.15).
                     float(getattr(cfg, "safety_drop_thr", 0.15)
                           or 0.15),
                     # V21.1 E4 rust-tree flag (spec[27], default False).
                     bool(_rt),
                     float(getattr(cfg, "ml_slope", 0.0)),
                     float(getattr(cfg, "ml_cap", 0.07)),
                     float(getattr(cfg, "ml_thr", 0.8)))
        inc_cpu = _Net(blocks=cfg.blocks, channels=cfg.channels,
                       planes=cfg.input_planes)
        _lwg(inc_cpu, incumb_w)
        inc = agent_policy(inc_cpu, cfg, sims=sims, device="cpu",
                           noise=cfg.dirichlet_eps, rust_tree=_rt)
        inc._spec = ("agent", inc_path, cfg.blocks, cfg.channels, sims,
                     cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_eps,
                     0, cfg.input_planes, True,
                     float(getattr(cfg, "contempt", 0.0) or 0.0),
                     bool(getattr(cfg, "asymmetric_contempt", False)),
                     bool(getattr(cfg, "tactical_override", False)),
                     bool(getattr(cfg, "blunder_veto", False)),
                     float(getattr(cfg, "tac_threshold", 0.09)),
                     int(getattr(cfg, "quiescence_depth", 0) or 0),
                     float(getattr(cfg, "forcing_bonus", 0.0) or 0.0),
                      float(getattr(cfg, "contempt_edge_scale", 0.0)
                            or 0.0),
                     # v15 finishing flag for parallel arena/gate rebuilds.
                     bool(getattr(cfg, "mate_finish", False)),
                     # v16.1 batched leaves for parallel arena/gate rebuilds.
                     int(getattr(cfg, "leaf_batch", 1) or 1),
                     float(getattr(cfg, "virtual_loss", 1.0) or 1.0),
                     # v15.5 safety veto for parallel arena/gate rebuilds.
                     bool(getattr(cfg, "safety_veto", False)),
                     # v18 SE trunk for parallel arena/gate rebuilds.
                     int(getattr(cfg, "se_ratio", 0) or 0),
                     # v19 search knobs for parallel rebuilds (0.0/False =
                     # yesterday bit-exact when unset).
                     float(getattr(cfg, "fpu_reduction", 0.0) or 0.0),
                     bool(getattr(cfg, "prune_singletons", False)),
                     # v19 safety threshold travels with the flag (deploy
                     # parity — bare namespaces used to pin 0.15).
                     float(getattr(cfg, "safety_drop_thr", 0.15)
                           or 0.15),
                     # V21.1 E4 rust-tree flag (spec[27], default False).
                     bool(_rt),
                     float(getattr(cfg, "ml_slope", 0.0)),
                     float(getattr(cfg, "ml_cap", 0.07)),
                     float(getattr(cfg, "ml_thr", 0.8)))
        from .parallel import play_match_parallel
        r = play_match_parallel(ch, inc, games=cfg.gate_games,
                                workers=workers,
                                opening_moves=cfg.gate_opening_moves,
                                game_settings=game_settings(cfg),
                                seed_base=seed_base,
                                use_server=use_server,
                                server_device=server_device)
    else:
        # (audit: this branch had an undefined `noise` NameError; gate
        # noise is cfg.dirichlet_eps, matching the parallel path.)
        # V21.1 E4: explicit param OR cfg knob.
        _rt0 = bool(rust_tree or getattr(cfg, "rust_tree", False))
        ch = agent_policy(challenger_model, cfg, sims=sims, device=device,
                          noise=cfg.dirichlet_eps, rust_tree=_rt0)
        inc = agent_from_weights(incumb_w, cfg, sims=sims, device=device,
                                 noise=cfg.dirichlet_eps, rust_tree=_rt0)
        r = play_match(ch, inc, games=cfg.gate_games,
                       opening_moves=cfg.gate_opening_moves,
                       seed_base=seed_base)
    n = r["wins"] + r["losses"] + r["draws"]
    score = (r["wins"] + 0.5 * r["draws"]) / max(n, 1)
    # v19: history gate dicts carry the SPRT LLR even on the legacy path
    # (decision stays the fixed threshold there; "sprt_pass" None marks
    # "not an SPRT decision"). score/challenger_wld keys kept.
    _llr = gsprt_llr(r["wins"], r["losses"], r["draws"],
                     float(getattr(cfg, "sprt_elo0", 0.0)),
                     float(getattr(cfg, "sprt_elo1", 30.0)))
    return {"challenger_wld": [r["wins"], r["losses"], r["draws"]],
            "score": round(score, 4), "llr": round(_llr, 4),
            "sprt_pass": None}


def _gate_sprt_pairs(ch, inc, cfg, workers, ckpt_dir, seed_base=1000,
                     use_server=False, server_device="cuda"):
    """Mirrored SPRT gate (C6/C7): sequential 2-game pairs over ids [0,1]
    (A-white then A-black sharing one seeded opening = mirrored, colors
    alternate adjacently for pair-wise LLR). Per-pair seeds derive from
    the reshuffled empirical-book draw (C7). Running LLR after each pair
    with early stop; cap 20 games; truncated fallback (LLR>0 AND
    score>=11/20). Promotion = sprt_pass AND arena_ok (conjunction lives
    with the caller, unchanged). Never raises: total failure falls back
    to a threshold-style dict (loud log)."""
    games = max(2, int(getattr(cfg, "gate_games", 20) or 20))
    cap = min(games, 20)
    opening = int(getattr(cfg, "gate_opening_moves", 6) or 0)
    lines, binfo = _gate_book_draw(seed_base)
    print(f"[gate] v19 mirrored SPRT: cap {cap} games, book "
          f"lines={binfo['n_lines']} draw={binfo['draw']} "
          f"fallback={binfo['fallback']}", flush=True)
    if not binfo["book"]:
        print("[gate] book missing -> paired random-plies path "
              "(pairs still share one seed each; colors alternate)",
              flush=True)
    W = L = D = 0
    pairs = 0
    stop_info = None
    try:
        from .parallel import play_match_parallel
        from .evaluate import play_match
        k = 0
        while 2 * k < cap:
            _line = lines[k % len(lines)] if lines else None
            _ps = _pair_seed(seed_base, k, _line)
            _ng = min(2, cap - 2 * k)
            if workers > 1:
                _tot, _recs = play_match_parallel(
                    ch, inc, games=_ng, workers=min(workers, _ng),
                    opening_moves=opening,
                    game_settings=game_settings(cfg),
                    record=True, seed_base=_ps,
                    use_server=use_server, server_device=server_device)
            else:
                _tot, _recs = play_match(
                    ch, inc, games=_ng, opening_moves=opening,
                    record=True, seed_base=_ps)
            W += _tot["wins"]
            L += _tot["losses"]
            D += _tot["draws"]
            pairs += 1
            dec = _gate_sprt_decide(W, L, D, cfg, W + L + D, cap)
            print(f"[gate] pair {pairs}: cum WLD=({W},{L},{D}) "
                  f"LLR={dec['llr']:+.3f} via={dec['via']}", flush=True)
            if dec["stop"]:
                stop_info = dec
                break
            k += 1
        n = max(1, W + L + D)
        score = (W + 0.5 * D) / n
        if stop_info is None:
            stop_info = _gate_sprt_decide(W, L, D, cfg, n, cap)
        return {"challenger_wld": [W, L, D], "score": round(score, 4),
                "llr": round(stop_info["llr"], 4),
                "sprt_pass": bool(stop_info["promote"]),
                "via": stop_info["via"], "pairs": pairs,
                "stop_pair": pairs if stop_info["stop"] else None,
                "book": binfo["draw"]}
    except Exception as _e:
        print(f"[gate] SPRT pairs failed ({type(_e).__name__}: "
              f"{str(_e)[:160]}); threshold fallback", flush=True)
        n = max(1, W + L + D)
        score = (W + 0.5 * D) / n
        return {"challenger_wld": [W, L, D], "score": round(score, 4),
                "llr": None, "sprt_pass": bool(
                    score >= float(getattr(cfg, "gate_threshold", 0.55))),
                "via": "error-fallback", "pairs": pairs,
                "stop_pair": None, "book": binfo["draw"]}


def _progress_check(new_path, old_path, cfg, sims, workers, ckpt_dir,
                    new_iter: int, old_iter: int, seed_base=71000,
                    sf_path: str = "/usr/games/stockfish"):
    """Lineage cop (user directive): every Nth promotion, the new champion
    plays the champion from N promotions ago head-to-head (paired colors,
    deterministic measurement: noise 0, like arena — this is EVIDENCE,
    not selection). Full-PGN film saved for autopsy. D13 appends
    sf_anchor_games (V19: 10) vs the cheapest SF rung — log-only, never
    training use. Never raises: on any error returns {"error": ...} so a
    measurement can never break a training run (sf-rung doctrine).
    A = new champ."""
    try:
        from .model import AlphaZeroNet as _Net
        from .model import load_weights as _lwg
        from .model import infer_se_ratio as _isr
        games = int(getattr(cfg, "progress_check_games", 12) or 12)
        # V21.1 E4: cfg knob only (signature frozen); default OFF.
        _rt = bool(getattr(cfg, "rust_tree", False))
        _nw = torch.load(new_path, map_location="cpu", weights_only=False)
        new_cpu = _Net(blocks=cfg.blocks, channels=cfg.channels,
                       planes=cfg.input_planes, se_ratio=_isr(_nw))
        _lwg(new_cpu, _nw)
        new_ag = agent_policy(new_cpu, cfg, sims=sims, device="cpu",
                              noise=0.0, temp_moves=0, use_tt=True,
                              rust_tree=_rt)
        new_ag._spec = ("agent", new_path, cfg.blocks, cfg.channels,
                        sims, cfg.c_puct, cfg.dirichlet_alpha, 0.0,
                        0, cfg.input_planes, True,
                        float(getattr(cfg, "contempt", 0.0) or 0.0),
                        bool(getattr(cfg, "asymmetric_contempt", False)),
                        bool(getattr(cfg, "tactical_override", False)),
                        bool(getattr(cfg, "blunder_veto", False)),
                        float(getattr(cfg, "tac_threshold", 0.09)),
                        int(getattr(cfg, "quiescence_depth", 0) or 0),
                        float(getattr(cfg, "forcing_bonus", 0.0) or 0.0),
                        float(getattr(cfg, "contempt_edge_scale", 0.0)
                               or 0.0),
                        bool(getattr(cfg, "mate_finish", False)),
                        int(getattr(cfg, "leaf_batch", 1) or 1),
                        float(getattr(cfg, "virtual_loss", 1.0) or 1.0),
                        bool(getattr(cfg, "safety_veto", False)),
                        _isr(_nw),
                        float(getattr(cfg, "fpu_reduction", 0.0) or 0.0),
                        bool(getattr(cfg, "prune_singletons", False)),
                        float(getattr(cfg, "safety_drop_thr", 0.15)
                              or 0.15),
                        # V21.1 E4 rust-tree flag (spec[27], default False).
                        bool(_rt),
                     float(getattr(cfg, "ml_slope", 0.0)),
                     float(getattr(cfg, "ml_cap", 0.07)),
                     float(getattr(cfg, "ml_thr", 0.8)))
        _ow = torch.load(old_path, map_location="cpu", weights_only=False)
        old_cpu = _Net(blocks=cfg.blocks, channels=cfg.channels,
                       planes=cfg.input_planes, se_ratio=_isr(_ow))
        _lwg(old_cpu, _ow)
        old_ag = agent_policy(old_cpu, cfg, sims=sims, device="cpu",
                              noise=0.0, temp_moves=0, use_tt=True,
                              rust_tree=_rt)
        old_ag._spec = ("agent", old_path, cfg.blocks, cfg.channels,
                        sims, cfg.c_puct, cfg.dirichlet_alpha, 0.0,
                        0, cfg.input_planes, True,
                        float(getattr(cfg, "contempt", 0.0) or 0.0),
                        bool(getattr(cfg, "asymmetric_contempt", False)),
                        bool(getattr(cfg, "tactical_override", False)),
                        bool(getattr(cfg, "blunder_veto", False)),
                        float(getattr(cfg, "tac_threshold", 0.09)),
                        int(getattr(cfg, "quiescence_depth", 0) or 0),
                        float(getattr(cfg, "forcing_bonus", 0.0) or 0.0),
                        float(getattr(cfg, "contempt_edge_scale", 0.0)
                               or 0.0),
                        bool(getattr(cfg, "mate_finish", False)),
                        int(getattr(cfg, "leaf_batch", 1) or 1),
                        float(getattr(cfg, "virtual_loss", 1.0) or 1.0),
                        bool(getattr(cfg, "safety_veto", False)),
                        _isr(_ow),
                        float(getattr(cfg, "fpu_reduction", 0.0) or 0.0),
                        bool(getattr(cfg, "prune_singletons", False)),
                        float(getattr(cfg, "safety_drop_thr", 0.15)
                              or 0.15),
                        # V21.1 E4 rust-tree flag (spec[27], default False).
                        bool(_rt),
                     float(getattr(cfg, "ml_slope", 0.0)),
                     float(getattr(cfg, "ml_cap", 0.07)),
                     float(getattr(cfg, "ml_thr", 0.8)))
        from .parallel import play_match_parallel
        res, records = play_match_parallel(
            new_ag, old_ag, games=games, workers=workers,
            opening_moves=cfg.gate_opening_moves,
            game_settings=game_settings(cfg), record=True,
            seed_base=seed_base)
        n = res["wins"] + res["losses"] + res["draws"]
        score = (res["wins"] + 0.5 * res["draws"]) / max(n, 1)
        # full-PGN progress film (both sides; loss-only savers miss draws
        # and wins, and this film exists to show PROGRESS, not failure).
        import chess as _c
        import chess.pgn as _pgn
        pgn_path = os.path.join(
            "pgns", f"progress_iter{new_iter}_vs_iter{old_iter}.pgn")
        try:
            os.makedirs("pgns", exist_ok=True)
            with open(pgn_path, "w") as f:
                for moves, outcome, a_white in records:
                    try:
                        game = _pgn.Game()
                        game.headers["White"] = (
                            f"champ-{new_iter}"
                            if a_white else f"champ-{old_iter}")
                        game.headers["Black"] = (
                            f"champ-{old_iter}"
                            if a_white else f"champ-{new_iter}")
                        if outcome is None:
                            game.headers["Result"] = "1/2-1/2"
                        elif (outcome and a_white) or \
                                ((not outcome) and (not a_white)):
                            game.headers["Result"] = "1-0"
                        else:
                            game.headers["Result"] = "0-1"
                        node = game
                        board = _c.Board()
                        for u in moves:
                            node = node.add_variation(
                                board.parse_uci(u))
                            board.push_uci(u)
                        f.write(str(game) + "\n\n")
                    except Exception:
                        continue
        except Exception:
            pgn_path = ""
        return {"vs_iter": old_iter,
                "wld": [res["wins"], res["losses"], res["draws"]],
                "score": round(score, 4),
                "elo": round(elo_diff(**res), 1), "pgn": pgn_path,
                "sf_anchor": _sf_anchor(
                    new_path, cfg, sims, workers, ckpt_dir, sf_path)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}


def _sf_anchor(new_path, cfg, sims, workers, ckpt_dir,
               sf_path: str = "/usr/games/stockfish"):
    """D13 external anchor lite: sf_anchor_games (V19: 10, old cfgs 0 =
    off) vs the cheapest rung (sf_anchor_rung, default sf-elo1350) via
    calibrate_ckpt. Appended to LINEAGE checks only; log-only, NO
    training use. Never raises (missing engine -> loud skip dict)."""
    try:
        _n = int(getattr(cfg, "sf_anchor_games", 0) or 0)
    except Exception:
        _n = 0
    if _n <= 0:
        return {"skipped": "sf_anchor_games=0 (yesterday)"}
    _rung = str(getattr(cfg, "sf_anchor_rung", "sf-elo1350") or
                "sf-elo1350")
    try:
        from .calibrate import calibrate_ckpt
        _r = calibrate_ckpt(new_path, _rung, _n, sims, cfg.blocks,
                            cfg.channels, sf_path,
                            workers=max(1, min(int(workers or 1), _n)),
                            input_planes=cfg.input_planes)
        print(f"[lineage] SF anchor vs {_rung}: {_r}", flush=True)
        return _r
    except Exception as _e:
        print(f"[lineage] SF anchor skipped ({type(_e).__name__}: "
              f"{str(_e)[:120]})", flush=True)
        return {"skipped": f"{type(_e).__name__}: {str(_e)[:120]}"}


def _save_loss_pgn(records, ckpt_dir, note) -> str:
    """Write our (A) losses vs greedy to a PGN file for autopsy. Returns path.
    records: (moves_uci, a_outcome, a_white) from _play_ids(record=True)."""
    import chess as _c
    import chess.pgn as _pgn
    path = os.path.join(ckpt_dir, f"arena_losses_{note or 'x'}.pgn")
    n = 0
    with open(path, "w") as f:
        for moves, outcome, a_white in records:
            if outcome is not False:
                continue
            try:
                game = _pgn.Game()
                game.headers["White"] = "challenger" if a_white else "greedy"
                game.headers["Black"] = "greedy" if a_white else "challenger"
                # outcome False = A lost; winner is the other side
                game.headers["Result"] = "0-1" if a_white else "1-0"
                node = game
                board = _c.Board()
                for u in moves:
                    node = node.add_variation(board.parse_uci(u))
                    board.push_uci(u)
                f.write(str(game) + "\n\n")
                n += 1
            except Exception:
                continue
    return f"{path}:{n}"


def _arena(model, cfg, games, sims, device, workers=1, ckpt_dir="checkpoints",
           sf_games=0, sf_rung="sf-elo1350", sf_sims=None,
           sf_path="/usr/games/stockfish", note="", seed_base=1000,
           use_server=False, server_device="cuda",
           rust_tree: bool = False):
    # Deterministic MEASUREMENT (no noise/sampling; diverse paired
    # openings). The corroborating line for gate promotions.
    # V21.1 E4: explicit param OR cfg knob (run_training stamps cfg).
    _rt = bool(rust_tree or getattr(cfg, "rust_tree", False))
    sf_sims = sims if sf_sims is None else sf_sims
    ag = agent_policy(model, cfg, sims=sims, device=device,
                      noise=cfg.arena_noise,
                      temp_moves=cfg.arena_temp_moves, rust_tree=_rt)
    want_pgn = bool(getattr(cfg, "pgn_losses", False))
    if workers > 1:
        w_path = os.path.join(ckpt_dir, "_w_arena.pt")
        atomic_save({k: v.cpu() for k, v in model.state_dict().items()},
                   w_path)
        ag._spec = ("agent", w_path, cfg.blocks, cfg.channels, sims,
                    cfg.c_puct, cfg.dirichlet_alpha, cfg.arena_noise,
                    cfg.arena_temp_moves, cfg.input_planes, True,
                    float(getattr(cfg, "contempt", 0.0) or 0.0),
                    bool(getattr(cfg, "asymmetric_contempt", False)),
                    bool(getattr(cfg, "tactical_override", False)),
                    bool(getattr(cfg, "blunder_veto", False)),
                    float(getattr(cfg, "tac_threshold", 0.09)),
                    int(getattr(cfg, "quiescence_depth", 0) or 0),
                    float(getattr(cfg, "forcing_bonus", 0.0) or 0.0),
                     float(getattr(cfg, "contempt_edge_scale", 0.0)
                            or 0.0),
                     # v15 finishing flag for parallel arena/gate rebuilds.
                     bool(getattr(cfg, "mate_finish", False)),
                     # v16.1 batched leaves for parallel arena/gate rebuilds.
                     int(getattr(cfg, "leaf_batch", 1) or 1),
                     float(getattr(cfg, "virtual_loss", 1.0) or 1.0),
                     # v15.5 safety veto for parallel arena/gate rebuilds.
                     bool(getattr(cfg, "safety_veto", False)),
                     # v18 SE trunk for parallel arena/gate rebuilds.
                     int(getattr(cfg, "se_ratio", 0) or 0),
                     # v19 search knobs for parallel rebuilds (0.0/False =
                     # yesterday bit-exact when unset).
                     float(getattr(cfg, "fpu_reduction", 0.0) or 0.0),
                     bool(getattr(cfg, "prune_singletons", False)),
                     # v19 safety threshold travels with the flag (deploy
                     # parity — bare namespaces used to pin 0.15).
                     float(getattr(cfg, "safety_drop_thr", 0.15)
                           or 0.15),
                     # V21.1 E4 rust-tree flag (spec[27], default False).
                     bool(_rt),
                     float(getattr(cfg, "ml_slope", 0.0)),
                     float(getattr(cfg, "ml_cap", 0.07)),
                     float(getattr(cfg, "ml_thr", 0.8)))
        from .parallel import play_match_parallel
        vs_rand = None
        if getattr(cfg, "panel_weak_diagnostics", True):
            vs_rand = play_match_parallel(ag, random_move, games=games,
                                          workers=workers,
                                          opening_moves=cfg.arena_opening_moves,
                                          game_settings=game_settings(cfg),
                                          seed_base=seed_base,
                                          use_server=use_server,
                                          server_device=server_device)
        grd = play_match_parallel(ag, greedy_move, games=games,
                                  workers=workers,
                                  opening_moves=cfg.arena_opening_moves,
                                  game_settings=game_settings(cfg),
                                  record=want_pgn, seed_base=seed_base,
                                  use_server=use_server,
                                  server_device=server_device)
        vs_grd, grec = grd if want_pgn else (grd, [])
    else:
        vs_rand = None
        if getattr(cfg, "panel_weak_diagnostics", True):
            vs_rand = play_match(ag, random_move, games=games,
                                 opening_moves=cfg.arena_opening_moves,
                                 seed_base=seed_base)
        grd = play_match(ag, greedy_move, games=games,
                         opening_moves=cfg.arena_opening_moves,
                         record=want_pgn, seed_base=seed_base)
        vs_grd, grec = grd if want_pgn else (grd, [])
    out = {
        "vs_greedy": vs_grd,
        "elo_greedy": round(elo_diff(**vs_grd), 1),
    }
    if vs_rand is not None:
        out["vs_random"] = vs_rand
        out["elo_random"] = round(elo_diff(**vs_rand), 1)
    else:
        out["weak_diagnostics_skipped"] = True
    if want_pgn and grec:
        out["pgn_losses"] = _save_loss_pgn(grec, ckpt_dir, note)
    if sf_games > 0:
        try:
            from .calibrate import calibrate_ckpt
            w_path = os.path.join(ckpt_dir, "_w_arena.pt")
            atomic_save({k: v.cpu() for k, v in model.state_dict().items()},
                       w_path)
            sf = calibrate_ckpt(w_path, sf_rung, sf_games, sf_sims,
                                cfg.blocks, cfg.channels, sf_path,
                                workers=max(1, workers),
                                input_planes=cfg.input_planes)
            out["vs_sf"] = sf
        except Exception as e:
            out["vs_sf"] = {"error": str(e)[:200]}
    return out


if __name__ == "__main__":
    from .config import TEST_CONFIG
    run_training(TEST_CONFIG, games_per_iter=4, iters=2,
                 train_steps=30, arena_games=4, arena_sims=8)
