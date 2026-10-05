"""Veto ROC recalibration (V20 D11, Batch A train core).

Sweeps a veto-threshold over labeled (score, label) pairs, prints the ROC
table, and recommends the threshold at 95% recall of true blunders. Read-
only: never writes checkpoints/history/config (stdout only, code-only).

Modes:
  --synthetic (default): deterministic two-Gaussian smoke data (no model,
    no games) — proves the ROC math runs and is finite.
  --games P --ckpt C: label real plies (hangs_material on the played move
    = true blunder) and score them with the checkpoint's WDL head
    (score = Q_opponent(child): higher = more blunder-like). Best-effort
    with loud skips; never raises on bad rows.

D11 order (fixed): the veto DEFAULT is recalibrated AFTER the TB rescore
(D6) lands — this script only PRINTS the recommendation. Applying it to
config (R4) is explicitly deferred (outside R1 is forbidden for Batch A),
so this file must not edit any threshold default.
FORBIDDEN (1 line each, not implemented): F1 no TB in MCTS. F2 no SL fine-
tune. F3 no multi-step policy targets. F4 no TB policy boost. F5 no full
no-resign. F6 no transformer. F7 no Maia regularizer. F8 anchor/rehearsal/
panel/lineage/veto/mate-finish/surprise kept. F9 no arity change here.

Usage:
  PYTHONPATH=. python3 -m chess_zero.recal_veto --synthetic
  PYTHONPATH=. python3 -m chess_zero.recal_veto --games data_sl/games.jsonl \\
      --ckpt checkpoints_warm/warmstart.pt --max-positions 2000
"""
from __future__ import annotations

import argparse


def roc_sweep(labels, scores, thresholds) -> list:
    """ROC points over thresholds: predict blunder iff score >= thr.
    Returns list of dicts {thr,tpr,fpr,tp,fp,tn,fn,precision,recall,n}.
    Empty input -> [] (never raises; caller prints a loud skip)."""
    try:
        _labs = [int(bool(x)) for x in (labels or [])]
        _sc = [float(x) for x in (scores or [])]
        _th = [float(x) for x in (thresholds or [])]
    except Exception:
        return []
    if not _labs or len(_labs) != len(_sc) or not _th:
        return []
    _p = sum(_labs)
    _n = len(_labs) - _p
    rows = []
    for _t in _th:
        _tp = sum(1 for _l, _s in zip(_labs, _sc)
                  if _l == 1 and _s >= _t)
        _fp = sum(1 for _l, _s in zip(_labs, _sc)
                  if _l == 0 and _s >= _t)
        _fn = _p - _tp
        _tn = _n - _fp
        _tpr = (_tp / _p) if _p else 0.0
        _fpr = (_fp / _n) if _n else 0.0
        _prec = (_tp / (_tp + _fp)) if (_tp + _fp) else 1.0
        rows.append({"thr": _t, "tpr": _tpr, "fpr": _fpr, "tp": _tp,
                     "fp": _fp, "tn": _tn, "fn": _fn,
                     "precision": _prec, "recall": _tpr,
                     "n": len(_labs)})
    return rows


def pick_at_recall(rows, target: float = 0.95) -> dict | None:
    """Threshold with recall >= target minimizing FPR (ties: higher thr =
    fewer false alarms). None when no row reaches target (never raises)."""
    try:
        _t = float(target)
    except Exception:
        _t = 0.95
    _ok = [r for r in (rows or []) if r.get("recall", 0.0) >= _t]
    if not _ok:
        return None
    _ok.sort(key=lambda r: (r["fpr"], -r["thr"]))
    return _ok[0]


def print_roc(rows) -> None:
    """Print the ROC table (thr, tpr/recall, fpr, precision, counts)."""
    print("thr,tpr,fpr,precision,tp,fp,tn,fn", flush=True)
    for r in (rows or []):
        print(f"{r['thr']:.4f},{r['tpr']:.4f},{r['fpr']:.4f},"
              f"{r['precision']:.4f},"
              f"{r['tp']},{r['fp']},{r['tn']},{r['fn']}", flush=True)


def synthetic_pairs(n: int = 2000, seed: int = 0) -> tuple:
    """Deterministic smoke labels/scores: blunders ~ N(1.0, 1.0), clean ~
    N(-1.0, 1.0) (10% blunder rate). Fixed seed -> reproducible ROC."""
    import random as _r
    _rng = _r.Random(int(seed))
    labs, sc = [], []
    for _ in range(max(0, int(n))):
        _b = 1 if _rng.random() < 0.10 else 0
        labs.append(_b)
        sc.append(_rng.gauss(1.0 if _b else -1.0, 1.0))
    return labs, sc


def default_thresholds() -> list:
    """Sweep grid: 41 points over [-3, 3] (covers the synthetic smoke mass
    and Q-scale scores; games mode reuses the same grid)."""
    return [-3.0 + 0.15 * i for i in range(41)]


def _uci_to_stm_action(state, uci_str: str) -> int | None:
    """Absolute UCI -> stm-oriented 4096 action (v9 flip: unmirror on black
    stm). None when unparseable/illegal (caller skips loudly, never raises)."""
    try:
        import chess as _c
        from .game import flip_action as _flip
        _mv = state.board.parse_uci(uci_str)
        _a = _mv.from_square * 64 + _mv.to_square
        if state.board.turn == _c.BLACK:
            _a = _flip(_a)
        if _a not in state.legal_moves():
            return None
        return int(_a)
    except Exception:
        return None


def labeled_from_games(games_path: str, ckpt_path: str,
                       max_positions: int = 2000,
                       device: str = "cpu") -> tuple:
    """Label plies from a games.jsonl file: label = hangs_material(state,
    played_action) (rules truth, 1 = true blunder); score = Q_opponent of
    the child (WDL head: higher = better for the opponent = more blunder-
    like). Single batched forward per chunk (CPU-friendly). Returns
    (labels, scores, info). Never raises: empty + info note on any failure
    (missing files, bad rows, load errors all skip loudly)."""
    import os as _os
    info = {"games": 0, "positions": 0, "skipped": 0, "note": ""}
    if not games_path or not _os.path.exists(str(games_path)):
        info["note"] = f"games file missing ({games_path})"
        return [], [], info
    if not ckpt_path or not _os.path.exists(str(ckpt_path)):
        info["note"] = f"ckpt file missing ({ckpt_path})"
        return [], [], info
    try:
        import json as _j
        import numpy as _np
        import torch as _t
        from .game import State as _S
        from .model import AlphaZeroNet as _Net, load_weights as _lw, \
            infer_se_ratio as _isr
        from .selfplay import hangs_material as _hm
        from .config import V19_CONFIG as _cfg
        _w = _t.load(str(ckpt_path), map_location="cpu",
                     weights_only=False)
        try:
            _se = _isr(_w) or int(getattr(_cfg, "se_ratio", 0) or 0)
        except Exception:
            _se = int(getattr(_cfg, "se_ratio", 0) or 0)
        _net = _Net(blocks=_cfg.blocks, channels=_cfg.channels,
                    planes=_cfg.input_planes, se_ratio=_se)
        try:
            _lw(_net, _w, strict=True)
        except Exception as _e:
            info["note"] = f"ckpt load failed ({type(_e).__name__})"
            return [], [], info
        _net.to(device)
        _net.eval()
        import chess_zero.game as _g
        _g.INPUT_PLANES = _cfg.input_planes
        _labels, _kids = [], []
        _ng = 0
        with open(str(games_path)) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _gm = _j.loads(_line)
                except Exception:
                    info["skipped"] += 1
                    continue
                _moves = [m for m in
                          (str(_gm.get("moves", "") or "").split(" ")) if m]
                if len(_moves) < 4:
                    continue
                _ng += 1
                _st = _S.initial()
                for _u in _moves:
                    try:
                        _a = _uci_to_stm_action(_st, _u)
                        if _a is None:
                            _st = None
                            break
                        try:
                            _lab = 1 if bool(_hm(_st, int(_a))) else 0
                        except Exception:
                            info["skipped"] += 1
                            _st = _st.apply(int(_a))
                            continue
                        _labels.append(_lab)
                        _kids.append(_st.apply(int(_a)))
                        _st = _kids[-1]
                        if len(_kids) >= max(1, int(max_positions)):
                            break
                    except Exception:
                        info["skipped"] += 1
                        break
                if len(_kids) >= max(1, int(max_positions)):
                    break
        info["games"] = _ng
        info["positions"] = len(_kids)
        if not _kids:
            info["note"] = "no labelable positions (parse/legality skips)"
            return [], [], info
        _scores = []
        _B = 128
        with _t.no_grad():
            for _i in range(0, len(_kids), _B):
                _ch = _kids[_i:_i + _B]
                try:
                    _x = _t.stack(
                        [_t.from_numpy(_np.asarray(
                            _s.encode(), dtype=_np.float32))
                         for _s in _ch]).to(device)
                    _o = _net(_x)
                    _wdl = _o[1].float().cpu()
                    _q_opp = (_wdl.softmax(dim=1)[:, 0] -
                              _wdl.softmax(dim=1)[:, 2]).tolist()
                    _scores.extend([float(v) for v in _q_opp])
                except Exception as _e:
                    info["skipped"] += len(_ch)
                    info["note"] = f"forward chunk failed " \
                        f"({type(_e).__name__})"
                    continue
        _m = min(len(_labels), len(_scores))
        return _labels[:_m], _scores[:_m], info
    except Exception as _e:
        info["note"] = f"recal failed ({type(_e).__name__}: " \
            f"{str(_e)[:120]})"
        return [], [], info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true", default=False)
    ap.add_argument("--games", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--max-positions", type=int, default=2000)
    ap.add_argument("--recall", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.games and a.ckpt:
        labs, sc, info = labeled_from_games(
            a.games, a.ckpt, a.max_positions, a.device)
        print(f"[recal_veto] games={info.get('games')} "
              f"positions={info.get('positions')} "
              f"skipped={info.get('skipped')} "
              f"note={info.get('note')}", flush=True)
        if not labs:
            print("[recal_veto] no labeled data; "
                  "falling back to --synthetic", flush=True)
            labs, sc = synthetic_pairs()
    else:
        labs, sc = synthetic_pairs()
        print(f"[recal_veto] synthetic n={len(labs)} "
              f"(blunders={sum(labs)})", flush=True)
    ths = default_thresholds()
    rows = roc_sweep(labs, sc, ths)
    if not rows:
        print("[recal_veto] empty ROC (no data); nothing to recommend",
              flush=True)
        return
    print_roc(rows)
    best = pick_at_recall(rows, a.recall)
    if best is None:
        print(f"[recal_veto] no threshold reaches recall>={a.recall:.2f} "
              f"(max recall={max(r['recall'] for r in rows):.4f}); "
              f"leaving default UNCHANGED (R4 deferred until post-D6)",
              flush=True)
    else:
        print(f"[recal_veto] recommended threshold@{a.recall:.2f}-recall: "
              f"thr={best['thr']:.4f} tpr={best['tpr']:.4f} "
              f"fpr={best['fpr']:.4f} precision={best['precision']:.4f} "
              f"(NOT applied: R4 deferred until post-D6 rescore)",
              flush=True)


if __name__ == "__main__":
    main()
