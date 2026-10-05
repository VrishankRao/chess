"""One-off: re-score an old weights file under CURRENT rules (audit round 2
§1.10 - separate the rules change from the model). Usage:
  python3 -m chess_zero.rescore --weights checkpoints_v83/best.pt --games 12
Runs a deterministic paired-opening arena vs greedy and prints WLD + Elo.
Does NOT touch training state.
§6.2 (corrected round-4 re-audit): this runs old WEIGHTS under the current
CODE (terminal order + PUCT fix) — it separates weights from those two,
NOT rules from model. Reuse is not a confound here: alternating arena
play is always fresh at exactly n_sims (exec-verified), so --no-reuse
must match the default run. For full attribution toggle one thing at a
time across two runs (rules only, rules+PUCT).
"""
from __future__ import annotations

import argparse
import os

import torch

import chess_zero.game as _game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--planes", type=int, default=21)
    ap.add_argument("--seed_base", type=int, default=1000)
    ap.add_argument("--no-reuse", action="store_true",
                    help="fixed-budget search (tt=None): must match the "
                         "default run in arena (alternating play is always "
                         "fresh — a mismatch would disprove that)")
    a = ap.parse_args()
    _game.ADJUDICATE_MARGIN = 1.0
    _game.NO_PROGRESS_PLIES = 60
    _game.ADJUDICATE_MIN_PLY = 16
    _game.INPUT_PLANES = a.planes
    from .config import Config
    from .model import AlphaZeroNet, load_weights, infer_se_ratio
    from .loop import agent_policy, game_settings
    from .evaluate import greedy_move
    from .parallel import play_match_parallel
    from .evaluate import elo_diff
    cfg = Config(blocks=a.blocks, channels=a.channels,
                 input_planes=a.planes, sims=a.sims, arena_noise=0.0,
                 arena_temp_moves=0, arena_opening_moves=6)
    cfg.adjudicate_values = dict(_game.ADJUDICATE_VALUES)
    model = AlphaZeroNet(blocks=a.blocks, channels=a.channels,
                         planes=a.planes,
                         se_ratio=infer_se_ratio(torch.load(
                             a.weights, map_location="cpu",
                             weights_only=False)))
    load_weights(model, torch.load(a.weights, map_location="cpu",
                                   weights_only=False), strict=True)
    model.eval()
    ag = agent_policy(model, cfg, sims=a.sims, device="cpu",
                      use_tt=not a.no_reuse)
    r = play_match_parallel(ag, greedy_move, games=a.games, cap=200,
                            workers=a.workers, opening_moves=6,
                            game_settings=game_settings(cfg),
                            seed_base=a.seed_base)
    tag = "no-reuse" if a.no_reuse else "reuse"
    print(f"RESCORE [{tag}] {a.weights}: {r} elo={round(elo_diff(**r), 1)}",
          flush=True)


if __name__ == "__main__":
    main()
