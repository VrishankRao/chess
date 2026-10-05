"""Full-scale run entrypoint for the GPU VM. Resumable: picks up from the
latest checkpoints/iterN.pt. Usage:
  python3 -m chess_zero.fullrun [--games 200 --iters 100 --steps 1000 --arena 40]
Defaults = PDF-scale training config (5x128, 40 sims).
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import torch

from .config import TRAIN_CONFIG
from .loop import run_training


def latest_iter(ckpt_dir="checkpoints") -> int:
    best = 0
    for p in glob.glob(os.path.join(ckpt_dir, "iter*.pt")):
        m = re.search(r"iter(\d+)\.pt$", p)
        if m:
            best = max(best, int(m.group(1)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--arena", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--sf_games", type=int, default=0,
                    help="stockfish games per arena (0=off)")
    ap.add_argument("--sf_rung", default="sf-elo1350")
    ap.add_argument("--sf_path", default="/usr/games/stockfish")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--start_iter", type=int, default=1)
    ap.add_argument("--final_ckpt", default="best.pt")
    ap.add_argument("--v3", action="store_true",
                    help="anti-draw/opening-diversity preset")
    ap.add_argument("--v5", action="store_true",
                    help="v3 + material head, contempt, no-progress rule")
    ap.add_argument("--v6", action="store_true",
                    help="v5 with pure unit-count aux + empirical values")
    ap.add_argument("--v7", action="store_true",
                    help="v6 + asymmetric contempt, tight screws, entropy, "
                         "phase plane, open arena, depth suite")
    ap.add_argument("--v8", action="store_true",
                    help="v7 + repetition planes, temp 30, adaptive contempt")
    ap.add_argument("--v8_1", action="store_true",
                    help="v8 training + deterministic-diverse arena")
    ap.add_argument("--v8_2", action="store_true",
                    help="v8.1 + greedy sparring + PGN loss autopsy")
    ap.add_argument("--v8_3", action="store_true",
                    help="v8.2 + tactical override + resign playout")
    ap.add_argument("--v8_4", action="store_true",
                    help="v8.3 + environment-integrity fixes")
    ap.add_argument("--v8_5", action="store_true",
                    help="v8.4 + round-2 audit fixes")
    ap.add_argument("--v8_6", action="store_true",
                    help="v8.5 + round-4 audit fixes (prune, corroboration, weighted refit, scaled steps)")
    ap.add_argument("--v8_7", action="store_true",
                    help="v8.6 + frozen worker inference (phase 1 speedup)")
    ap.add_argument("--v8_8", action="store_true",
                    help="v8.7 + batched GPU inference server (phase 3)")
    ap.add_argument("--v9", action="store_true",
                    help="search contempt + veto + punisher + flip (v9)")
    ap.add_argument("--v10", action="store_true",
                    help="v9 + planning package (ownership/quiescence)")
    ap.add_argument("--v11", action="store_true",
                    help="v10 + edge-scaled contempt")
    ap.add_argument("--v12", action="store_true",
                    help="v11 + capture extensions + pawn-anchored values + fixed ownership")
    ap.add_argument("--v13", action="store_true",
                    help="v13 finishing school on M1 footprint: full-playout "
                         "mate-signal games + margin 3.0 + terminal metrics")
    ap.add_argument("--v14", action="store_true",
                    help="v13 + 400 sims (breaks PDF 30-50 range, "
                         "user-directed): strength over knob-faithfulness")
    ap.add_argument("--v15", action="store_true",
                    help="v14 + ML-guided finishing (mate_finish): when "
                         "ahead, minimize predicted remaining length")
    ap.add_argument("--v16", action="store_true",
                    help="v16 SF-films cut: 6x128 (surgery warm-start), "
                         "safety head, ECO book, no decisive adjudication")
    ap.add_argument("--v15_5", action="store_true",
                    help="v15.5 rollback cut: 4x64 + leaf batching + "
                         "safety veto with teeth, no training book")
    ap.add_argument("--v18", action="store_true",
                    help="v18 final-bot cut: 6x64 SE + 30 tactical planes + "
                         "WDL + reply aux + fast/full split (SL warm-start)")
    ap.add_argument("--v19", action="store_true",
                    help="v19 final-bot cut: v18 + FPU/singleton search, "
                         "game-budget LR, SPRT mirrored gate, KL anchor + "
                         "rehearsal, aux late cull (SL warm-start first)")
    ap.add_argument("--audited", action="store_true", help="Corrected V20 baseline with a halfmove-clock plane")
    ap.add_argument("--eval-sims", type=int, default=None)
    ap.add_argument("--gate-every", type=int, default=5,
                    help="full promotion checks every N iterations (and final)")
    ap.add_argument("--weak-diagnostics-every", type=int, default=10,
                    help="additional weak-bot diagnostics every N iterations")
    ap.add_argument("--sample-reuse", type=float, default=None)
    ap.add_argument("--challenger-selfplay-frac", type=float, default=0.0)
    ap.add_argument("--lr-warmup-steps", type=int, default=None)
    ap.add_argument("--gate-games", type=int, default=None)
    ap.add_argument("--gate-alpha", type=float, default=0.05)
    ap.add_argument("--gate-book-file", default="")
    ap.add_argument("--gate-min-pairs", type=int, default=1)
    ap.add_argument("--v20", action="store_true",
                    help="v20 strength-first middlegame+endgame upgrade: "
                         "v19 + score/TD/AV heads, endgame starts+panel, "
                         "smart resign, TB rescore, position LR "
                         "(SL warm-start first)")
    ap.add_argument("--mac", "--m1", dest="mac", action="store_true",
                    help="Apple Silicon M1 8GB RAM preset (4x64, MPS, safe buffer/workers)")
    ap.add_argument("--use-server", action="store_true",
                      help="force infer-server workers (default on for v8.8)")
    ap.add_argument("--infer-server", dest="infer_server",
                      action="store_true",
                      help="V21 Batch A opt-in: batched MPS/CUDA inference "
                           "server for workers (default OFF; live run "
                           "untouched; alias for --use-server)")
    ap.add_argument("--rust-tree", dest="rust_tree", action="store_true",
                    default=False,
                    help="V21.1 E4 opt-in: route search through "
                         "mcts_bridge.RustTreeBackend with identical params "
                         "(default OFF; off = yesterday bit-exact)")
    ap.add_argument("--av-table", dest="av_table", default="",
                    help="V20.2 R2 opt-in: build_av.py output jsonl for "
                         "rehearsal-tail AV distillation (default OFF = "
                         "empty, yesterday bit-exact; miss rows read zeros)")
    ap.add_argument("--endgame-workers", dest="endgame_workers", type=int,
                    default=0,
                    help="parallel endgame-panel fan-out (default 0 = "
                         "sequential yesterday bit-exact; >1 = parallel, "
                         "bit-identical WLD on fixed seeds)")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--sf_sims", type=int, default=None)
    a = ap.parse_args()
    import sys
    import os as _os
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    is_mac = sys.platform == "darwin" or ram_gb <= 8.5

    if a.device == "cuda" and not torch.cuda.is_available():
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = a.device

    server_dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if a.workers:
        workers = a.workers
    elif a.mac or is_mac:
        # 8GB unified memory on Mac: 3-4 workers maximize M1 performance cores
        # while keeping total RAM under ~2GB to prevent memory compression and swap thrashing.
        workers = min(4, max(1, (_os.cpu_count() or 2) // 2))
    else:
        workers = max(1, (_os.cpu_count() or 2) - 1)

    print(f"device={dev} cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else ('Apple Silicon MPS' if dev == 'mps' else 'none')} "
          f"workers={workers} cpus={_os.cpu_count()} ram_gb={ram_gb:.1f}", flush=True)
    start = latest_iter(a.ckpt_dir)
    if start and not a.resume:
        print(f"note: {start} checkpoints present; pass --resume to warm-start",
              flush=True)
    from .config import TRAIN_CONFIG, V3_CONFIG, V5_CONFIG, V6_CONFIG, \
        V7_CONFIG, V8_CONFIG, V8_1_CONFIG, V8_2_CONFIG, V8_3_CONFIG, \
        V8_4_CONFIG, V8_5_CONFIG, V8_6_CONFIG, V8_7_CONFIG, V8_8_CONFIG, \
        V9_CONFIG, V10_CONFIG, V11_CONFIG, V12_CONFIG, V13_CONFIG, \
        V14_CONFIG, V15_CONFIG, V16_CONFIG, V15_5_CONFIG, V18_CONFIG, \
        V19_CONFIG, V20_CONFIG, MAC_CONFIG, AUDITED_CONFIG
    selected = [key for key, value in vars(a).items() if value is True and (key in ("mac", "audited") or (key.startswith("v") and key[1:2].isdigit()))]
    if len(selected) > 1:
        ap.error("Select one training preset: " + ", ".join(selected))
    if start and not a.resume and a.start_iter <= start:
        ap.error("Checkpoint directory already contains this run; choose --resume or a new --ckpt_dir")
    cfg = TRAIN_CONFIG
    if a.mac and (a.v13 or a.v14 or a.v15 or a.v16 or a.v15_5 or a.v18
                  or a.v19 or a.v20):
        # v15: --mac silently shadows version flags (MAC_CONFIG has no
        # finishing/full-playout). Loud, not silent.
        print("WARNING: --mac overrides --v13/--v14/--v15/--v16/--v15_5/"
              "--v18/--v19/--v20 "
                "(MAC_CONFIG: sims 40, no finishing). "
                "Drop --mac to train the version.", flush=True)
    if a.audited:
        cfg = AUDITED_CONFIG
    elif a.mac:
        cfg = MAC_CONFIG
    elif a.v20:
        cfg = V20_CONFIG
    elif a.v19:
        cfg = V19_CONFIG
    elif a.v18:
        cfg = V18_CONFIG
    elif a.v15_5:
        cfg = V15_5_CONFIG
    elif a.v16:
        cfg = V16_CONFIG
    elif a.v15:
        cfg = V15_CONFIG
    elif a.v14:
        cfg = V14_CONFIG
    elif a.v13:
        cfg = V13_CONFIG
    elif a.v12:
        cfg = V12_CONFIG
    elif a.v11:
        cfg = V11_CONFIG
    elif a.v10:
        cfg = V10_CONFIG
    elif a.v9:
        cfg = V9_CONFIG
    elif a.v8_8:
        cfg = V8_8_CONFIG
    elif a.v8_7:
        cfg = V8_7_CONFIG
    elif a.v8_6:
        cfg = V8_6_CONFIG
    elif a.v8_5:
        cfg = V8_5_CONFIG
    elif a.v8_4:
        cfg = V8_4_CONFIG
    elif a.v8_3:
        cfg = V8_3_CONFIG
    elif a.v8_2:
        cfg = V8_2_CONFIG
    elif a.v8_1:
        cfg = V8_1_CONFIG
    elif a.v8:
        cfg = V8_CONFIG
    elif a.v7:
        cfg = V7_CONFIG
    elif a.v6:
        cfg = V6_CONFIG
    elif a.v5:
        cfg = V5_CONFIG
    elif a.v3:
        cfg = V3_CONFIG
    # V20.2 R2 (append/flag only, empty default): thread the AV table path
    # onto the cfg (loop loads it once at run start; "" = OFF = yesterday).
    try:
        cfg.av_table = str(a.av_table or "")
    except Exception:
        pass
    if str(getattr(cfg, "av_table", "") or ""):
        print(f"av-table: {cfg.av_table} (rehearsal-tail AV ON)",
              flush=True)
    try:
        cfg.endgame_panel_workers = int(a.endgame_workers or 0)
    except Exception:
        pass
    if int(getattr(cfg, "endgame_panel_workers", 0) or 0) > 1:
        print(f"endgame-panel: parallel workers="
              f"{cfg.endgame_panel_workers} (bit-identical WLD)",
              flush=True)
    use_server = bool(a.v8_8 or a.v9 or a.v10 or a.v11 or a.v12
                      or a.use_server or a.infer_server)
    if a.mac:
        print("mac: Apple Silicon M1 (8GB RAM) preset", flush=True)
    elif a.v20:
        print("v20: v19 + score/TD/AV heads + endgame starts/panel + "
              "smart resign + TB rescore + position LR "
              "(SL warm-start first)", flush=True)
    elif a.v19:
        print("v19: 6x64 SE + WDL + FPU/singleton search + game-budget LR "
              "+ SPRT mirrored gate + KL anchor/rehearsal "
              "(SL warm-start first)", flush=True)
    elif a.v18:
        print("v18: 6x64 SE + 30 tactical planes + WDL + fast/full "
              "(SL warm-start first)", flush=True)
    elif a.v15_5:
        print("v15.5: 4x64 rollback + leaf batching + safety veto, "
              "no training book", flush=True)
    elif a.v16:
        print("v16: 6x128 surgery warm-start + safety + book + "
              "no-decisive-adjudication", flush=True)
    elif a.v15:
        print("v15: v14 + ML-guided finishing (mate_finish)", flush=True)
    elif a.v14:
        print("v14: v13 finishing school at 400 sims (PDF range broken, "
              "user-directed)", flush=True)
    elif a.v13:
        print("v13: finishing school (full-playout mate games + margin "
              "3.0 + terminal metrics)", flush=True)
    elif a.v12:
        print("v12: capture extensions + pawn-anchored values + fixed ownership", flush=True)
    elif a.v11:
        print("v11: edge-scaled contempt", flush=True)
    elif a.v10:
        print("v10: planning package (ownership + quiescence)", flush=True)
    elif a.v9:
        print("v9: search contempt + veto + flip", flush=True)
    elif a.v8_8:
        print("v8.8: v8.7 + batched GPU inference server", flush=True)
    elif a.v8_7:
        print("v8.7: v8.6 + frozen worker inference", flush=True)
    elif a.v8_6:
        print("v8.6: v8.5 + round-4 audit fixes", flush=True)
    elif a.v8_5:
        print("v8.5: v8.4 + round-2 audit fixes", flush=True)
    elif a.v8_4:
        print("v8.4: v8.3 + environment-integrity fixes", flush=True)
    elif a.v8_3:
        print("v8.3: v8.2 + tactical override + resign playout", flush=True)
    elif a.v8_2:
        print("v8.2: v8.1 + greedy sparring + PGN loss autopsy", flush=True)
    elif a.v8_1:
        print("v8.1: v8 training + deterministic-diverse arena", flush=True)
    elif a.v8:
        print("v8: v7 + repetition planes + temp30 + adaptive contempt",
              flush=True)
    elif a.v7:
        print("v7: asymmetric contempt + tight screws + entropy + phase",
              flush=True)
    elif a.v6:
        print("v6: unit-count aux + empirical values", flush=True)
    elif a.v5:
        print("v5: material head + contempt + no-progress rule", flush=True)
    import dataclasses
    cfg = dataclasses.replace(cfg)
    if a.gate_every < 1 or a.weak_diagnostics_every < 1:
        ap.error("evaluation intervals must be positive")
    if a.weak_diagnostics_every % a.gate_every:
        ap.error("weak diagnostics interval must be a multiple of gate interval")
    if not 0 <= a.challenger_selfplay_frac <= 1 or not 0 < a.gate_alpha < 1:
        ap.error("invalid challenger fraction or gate alpha")
    if a.lr_warmup_steps is not None and a.lr_warmup_steps < 1:
        ap.error("warmup steps must be positive")
    if a.gate_min_pairs < 1 or (a.gate_games is not None and (a.gate_games < 2 or a.gate_games % 2)):
        ap.error("gate games must be positive and even; minimum pairs positive")
    cfg.challenger_selfplay_frac = a.challenger_selfplay_frac
    cfg.gate_alpha = a.gate_alpha
    cfg.gate_book_file = a.gate_book_file
    cfg.gate_min_pairs = a.gate_min_pairs
    if a.lr_warmup_steps is not None: cfg.lr_warmup_steps = a.lr_warmup_steps
    if a.gate_games is not None: cfg.gate_incumbent_games = a.gate_games
    cfg.gate_every = a.gate_every
    cfg.weak_diagnostics_every = a.weak_diagnostics_every
    print(f"Evaluation: full gate every {cfg.gate_every} iterations; "
          f"weak diagnostics every {cfg.weak_diagnostics_every}; final gate ON", flush=True)
    if a.sample_reuse is not None:
        cfg.sample_reuse = max(0.0, a.sample_reuse)
    if a.resume and a.start_iter == 1:
        resume_meta = torch.load(a.resume, map_location="cpu", weights_only=False).get("meta", {})
        a.start_iter = int(resume_meta.get("iter", 0)) + 1
    h = run_training(cfg, games_per_iter=a.games, iters=a.iters,
                     train_steps=a.steps, arena_games=a.arena,
                     arena_sims=a.eval_sims or cfg.sims, device=dev, ckpt_dir=a.ckpt_dir,
                     workers=workers, sf_games=a.sf_games, sf_rung=a.sf_rung,
                     sf_sims=a.sf_sims, sf_path=a.sf_path, resume=a.resume,
                     start_iter=a.start_iter, final_ckpt=a.final_ckpt,
                     use_server=use_server, server_device=server_dev,
                     rust_tree=bool(a.rust_tree))
    print("finished", len(h), "iters", flush=True)


if __name__ == "__main__":
    main()
