"""V21 Batch B — leaf-batch + torch.compile knobs (efficiency only).

Prime directive: NO search-behavior change. Same sims, same params, same
policies (within measured tolerance). Anything that changes move selection
goes to the V22 list, NOT this version.

This module is NEW (V21) and default-off: importing it changes nothing.
The live run (mac_run16, V20_CONFIG, workers=7) never imports it; loop.py /
train.py / mcts.py / model.py / config.py are UNTOUCHED (read-only). No
training launch, no best.pt / history writes, no V20_CONFIG value edits.

V21 Batch B measured numbers (offline microbenchmarks, workers=1, read-only
weights checkpoints_warm/warmstart_v20.pt, torch 2.13.0, Apple Silicon MPS;
live mac_run16 with 7 workers was RUNNING during measurement, so wall-clock
is contention-noisy — forward throughput ratios are the stable signal):

  Table 1 — MPS forward timing, 6x64/SE4/30-plane net (forward_pv, fp32):
    batch  8:   9.555 ms/fwd    837.3 pos/s   (full forward: 15.605 ms)
    batch 16:  10.999 ms/fwd   1454.6 pos/s   (full forward: 16.917 ms)
    batch 32:  13.441 ms/fwd   2380.8 pos/s   (full forward: 19.503 ms)
    => per-forward cost rises sublinearly; per-position cost falls
       1.19 -> 0.69 -> 0.42 ms/pos (~2.8x positions/sec at batch 32).

  Table 2 — 400-sim search policy divergence, leaf_batch 8 vs 16 vs 32,
  20 fixed positions (warmstart_v20 net, MPS eager, V20 search knobs
  c_puct=1.2/FPU=0.4/singletons-pruned/quiescence=2/forcing=0.25,
  dirichlet_eps=0.0 seeded per position, tt=None, no MLH):
    max|dpi| 8v16 = 0.1250   flips/20 = 1   (5%)
    max|dpi| 8v32 = 0.1352   flips/20 = 2   (10%)
    max|dpi| 16v32 = 0.1250  flips/20 = 1   (5%)
    e.g. pos03 argmax 82 vs 731 (flip), pos17 334 vs 339 (flip).
    => NOT parity-clean (>> 1e-4 bar). Batch size CHANGES move selection
       via WU-pending divergence, so larger batches are a V22 strength
       question, NOT a V21 efficiency win. Default stays 8 = yesterday.

  Table 3 — 400-sim search wall-clock (same 400-sim setup, 5 positions):
    leaf_batch  8:  5.53 s/game
    leaf_batch 16:  6.91 s/game
    leaf_batch 32:  6.85 s/game
    => NO search-level win under live-run contention despite Table 1
       forward throughput. Reinforces: keep default 8.

  Table 4 — torch.compile re-test (current torch 2.13.0, MPS, forward_pv):
    run 1: eager b8 7.821ms -> compiled 6.735ms (1.16x);
           eager b32 11.765ms -> compiled 10.114ms (1.16x).
    run 2 (repeat 3x, contended): b8 speedups 1.05x/0.92x/0.95x;
           b32 speedups 1.08x/0.98x/0.97x.
    parity: max|dprob| = 0.00e+00, max|dQ| = 0.00e+00, argmax 40/40,
           finite True (bit-identical on this net shape).
    => NO RELIABLE win on MPS for the 6x64 shape (0.92-1.16x noise band
       with the live run contending). Opt-in flag ships default-OFF for
       future re-validation on idle hardware; never on by default.

Decision (acceptance: no default changes):
  - leaf_batch default stays 8 (yesterday / V20_CONFIG value). This module
    exposes LEAF_BATCH_DEFAULT = 8 + leaf_batch_for() validation helper
    only; it never mutates any Config.
  - torch.compile ships as opt-in only (USE_COMPILE_DEFAULT = False,
    env CHESS_ZERO_COMPILE=1 or explicit flag). maybe_compile() falls
    back to eager loudly and never raises.
"""
from __future__ import annotations

import os

# Yesterday's leaf batch (V20_CONFIG.leaf_batch). Duplicated here as a
# literal so this module never imports Config (no live-path coupling) and
# never mutates it. Changing this constant changes nothing live — callers
# must explicitly opt in via leaf_batch_for(override=...).
LEAF_BATCH_DEFAULT: int = 8

# Explicitly validated batch choices. 1 = legacy sequential path in mcts.
LEAF_BATCH_VALID: tuple = (1, 8, 16, 32)

# torch.compile opt-in. Default OFF. Env name doubles as the kill switch
# in reverse (unset/0 = eager = yesterday).
USE_COMPILE_DEFAULT: bool = False
COMPILE_ENV: str = "CHESS_ZERO_COMPILE"
COMPILE_MODE_DEFAULT: str = "default"


def leaf_batch_for(cfg=None, override=None) -> int:
    """Resolve the leaf batch for an offline/future caller (default 8).

    Precedence: explicit override (int) > cfg.leaf_batch (when cfg carries
    one) > LEAF_BATCH_DEFAULT. Invalid values (non-int, <= 0, > 32) fall
    back to LEAF_BATCH_DEFAULT. Never raises, never mutates cfg.
    """
    if override is not None:
        try:
            v = int(override)
            if v >= 1 and v <= 32:
                return v
        except Exception:
            pass
        return int(LEAF_BATCH_DEFAULT)
    if cfg is not None:
        try:
            v = int(getattr(cfg, "leaf_batch", LEAF_BATCH_DEFAULT))
            if v >= 1 and v <= 32:
                return v
        except Exception:
            pass
    return int(LEAF_BATCH_DEFAULT)


def compile_enabled(flag=None) -> bool:
    """Whether torch.compile is requested (default OFF).

    Precedence: explicit flag (bool) > env CHESS_ZERO_COMPILE ("1"/"true"
    (case-insensitive)/"yes" = on) > USE_COMPILE_DEFAULT (False).
    Never raises.
    """
    try:
        if flag is not None:
            return bool(flag)
        v = os.environ.get(COMPILE_ENV, "")
        if isinstance(v, str):
            if v.strip().lower() in ("1", "true", "yes", "on"):
                return True
            return bool(USE_COMPILE_DEFAULT)
        return bool(USE_COMPILE_DEFAULT)
    except Exception:
        return bool(USE_COMPILE_DEFAULT)


def maybe_compile(model, enabled: bool = False, mode: str | None = None):
    """Return a compiled model when explicitly enabled, else `model`.

    Default-off: enabled=False (the default) returns `model` unchanged
    (same object). enabled=True attempts torch.compile(mode or
    COMPILE_MODE_DEFAULT); ANY failure (old torch, MPS backend gap,
    compile error) logs loudly and returns the eager model. Never raises,
    never mutates the caller's training model in place beyond what
    torch.compile itself returns (callers should treat the return as the
    inference-only handle, exactly like the jit-freeze precedent).
    """
    if not enabled:
        return model
    try:
        import torch as _t
        _mode = mode or COMPILE_MODE_DEFAULT
        return _t.compile(model, mode=_mode)
    except Exception as _e:
        try:
            print(f"[v21] torch.compile unavailable "
                  f"({type(_e).__name__}: {str(_e)[:120]}); running eager",
                  flush=True)
        except Exception:
            pass
        return model
