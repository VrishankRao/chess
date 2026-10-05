"""ResNet with policy + value heads (PDF) plus an auxiliary material head.
The aux head predicts side-to-move material difference in pawns/10 from the
trunk — self-supervised from the board itself (zero human games). It forces
the trunk to represent "what is hanging" before policy/value ever see it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    __constants__ = ["se_ratio"]
    def __init__(self, c: int, se_ratio: int = 0):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        # v18 SE (Lc0/KataGo doctrine, ~3% cost): channel attention lets
        # the trunk condition on global context (winning->simple). 0 = off
        # (yesterday's block exactly). New keys random-init on old
        # checkpoints (strict=False); v18 trains from SL warm-start anyway.
        self.se_ratio = se_ratio
        if se_ratio > 0:
            se_c = max(8, c // se_ratio)
            self.se_fc1 = nn.Linear(c, se_c)
            self.se_fc2 = nn.Linear(se_c, c)
        else:
            self.se_fc1 = nn.Identity()
            self.se_fc2 = nn.Identity()

    def forward(self, x):
        r = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        if self.se_ratio > 0:
            s = x.mean(dim=(2, 3))
            s = F.relu(self.se_fc1(s))
            s = torch.sigmoid(self.se_fc2(s)).unsqueeze(-1).unsqueeze(-1)
            x = x * s
        return F.relu(x + r)


def wdl_q(wdl_logits):
    """Expected score Q = P(win) - P(loss) from WDL logits (Lc0: Q=W-L).
    The scalar value head is GONE (v18): everything that read values
    (search backups, resign, finishing) reads Q through this helper."""
    _p = F.softmax(wdl_logits, dim=1)
    return _p[:, 0] - _p[:, 2]


def infer_se_ratio(w) -> int:
    """v18 weight-adaptive SE: returns 4 when the checkpoint carries SE
    keys, else 0 (plain blocks). Lets measurement tools serve any lineage
    without flags; explicit cfg/arch args still win where given."""
    try:
        inner = w.get("weights", w) if isinstance(w, dict) else {}
        if isinstance(inner, dict) and \
                any("se_fc" in k for k in inner):
            return 4
    except Exception:
        pass
    return 0


class AlphaZeroNet(nn.Module):
    def __init__(self, blocks=4, channels=64, planes=13, actions=4096,
                 policy_channels=8, value_channels=8, aux_channels=4,
                 se_ratio: int = 0):
        super().__init__()
        self.trunk_in = nn.Conv2d(planes, channels, 3, padding=1, bias=False)
        self.trunk_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResidualBlock(channels, se_ratio)
                                      for _ in range(blocks)])
        # policy head: 1x1 conv then FC to 4096
        self.p_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(policy_channels)
        # v19 D1 conv policy head (arch thread #1, FEWER params than the
        # flat FC): the flat p_fc Linear(policy_channels*64 -> 4096) is
        # DELETED as a Linear and replaced by Conv3x3(ch,ch)+BN+ReLU +
        # Conv1x1(ch,64) with ch = policy_channels. Kept p_conv/p_bn stem
        # (1x1 channels->policy_channels) feeds the new stack; the final
        # 1x1 REUSES the name p_fc (now a Conv2d) so old checkpoints carrying
        # Linear p_fc keys hit the grow_state_dict fresh-head special case
        # (zeros, not a rank-change crash). flatten [B,64,8,8] -> [B,4096]
        # gives logit[from*64+to] = out[from, to_row, to_col] (stm-oriented,
        # matches the 4096 codec; promotion suffix irrelevant to codec).
        self.p_conv2 = nn.Conv2d(policy_channels, policy_channels, 3,
                                 padding=1, bias=False)
        self.p_bn2 = nn.BatchNorm2d(policy_channels)
        self.p_fc = nn.Conv2d(policy_channels, 64, 1, bias=True)
        # v18 WDL value head (Lc0 v0.21 doctrine, replaces the scalar
        # tanh): 3-way win/draw/loss; search reads Q=W-L via wdl_q().
        # v_fc2 grows 1->3 on old checkpoints (zero-padded rows = uniform
        # WDL = Q 0 start, graceful). Scalar value is gone everywhere.
        self.v_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(value_channels)
        self.v_fc1 = nn.Linear(value_channels * 64, 64)
        self.v_fc2 = nn.Linear(64, 3)
        # aux material head: predicts stm material diff (pawns/10), linear out
        self.a_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.a_bn = nn.BatchNorm2d(aux_channels)
        self.a_fc1 = nn.Linear(aux_channels * 64, 32)
        self.a_fc2 = nn.Linear(32, 1)
        # moves-left head: plies remaining / move_cap, sigmoid out.
        # Pure self-supervision from game lengths (Leela-style): teaches
        # technique (fast wins, slow losses) and enables real time management.
        self.m_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.m_bn = nn.BatchNorm2d(aux_channels)
        self.m_fc1 = nn.Linear(aux_channels * 64, 32)
        self.m_fc2 = nn.Linear(32, 1)
        # v10 ownership head: per-square final occupancy, stm-relative
        # (+1 ours / -1 theirs / 0 empty), sigmoid out. Fully convolutional
        # (no FC): spatially aligned dense supervision from the final board
        # — the first head that tells the trunk WHERE pieces belong. KataGo
        # recipe; rules-derived, zero human games. Missing keys in old
        # checkpoints random-init via strict=False (moves-left precedent).
        self.o_conv = nn.Conv2d(channels, 1, 1, bias=False)
        # v10 margin head: final stm-relative material (pawns/10), linear.
        # Same teacher as aux but end-state instead of current — the value
        # head's coarse results can't credit quiet moves; these can.
        # v12: stm-relative (not white-relative) to preserve canonical
        # orientation with value/aux/mobility heads (no polarity conflict).
        self.s_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.s_bn = nn.BatchNorm2d(aux_channels)
        self.s_fc1 = nn.Linear(aux_channels * 64, 32)
        self.s_fc2 = nn.Linear(32, 1)
        # v10 mobility head: own legal-move count squashed to [0,1].
        # Trivially computable self-supervision that rewards piece activity
        # in the trunk (bundle with ownership; near-free).
        self.b_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.b_bn = nn.BatchNorm2d(aux_channels)
        self.b_fc1 = nn.Linear(aux_channels * 64, 32)
        self.b_fc2 = nn.Linear(32, 1)
        # v16 king-safety head: P(own king safer than theirs), sigmoid.
        # Rules-derived dense supervision for the 0/8 castling disease
        # and back-rank mates (mated final boards read ~0.0). Same
        # bundle pattern as mobility. New keys random-init on old
        # checkpoints (strict=False skips ABSENT keys; shape mismatches
        # still raise — only trunk_in growth is patched).
        self.k_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.k_bn = nn.BatchNorm2d(aux_channels)
        self.k_fc1 = nn.Linear(aux_channels * 64, 32)
        self.k_fc2 = nn.Linear(32, 1)
        # v18 opp-reply aux head (KataGo opp-policy doctrine, lite):
        # 64-way from-square of the OPPONENT's reply (absolute squares,
        # documented). Trains opponent modeling into the trunk; never
        # read at inference (zero behavior change, pure representation
        # pressure). -100 target = chain broken (unrecorded plies), ignore.
        self.r_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.r_bn = nn.BatchNorm2d(aux_channels)
        self.r_fc1 = nn.Linear(aux_channels * 64, 32)
        self.r_fc2 = nn.Linear(32, 64)
        # v19 soft-policy aux head (train-only distillation signal, never
        # read at inference): full-size FC like p_fc from the same policy
        # trunk features. Target = pi^(1/4) normalized (T=4). New keys
        # random-init via strict=False (reply precedent).
        self.p_soft_fc = nn.Linear(policy_channels * 64, actions)
        # v19 check head: P(side-to-move in check), sigmoid, BCE. 1x1 conv
        # (aux_ch) + FC 256->32->1 (256 = aux_channels*64 when aux=4).
        # Target = board.is_check() of the position. New keys random-init.
        self.c_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.c_bn = nn.BatchNorm2d(aux_channels)
        self.c_fc1 = nn.Linear(aux_channels * 64, 32)
        self.c_fc2 = nn.Linear(32, 1)
        # v19 D4 plies-until-progress head: scalar like moves-left (1x1
        # conv + FC32 + FC1, LINEAR out, no sigmoid). Target = plies to the
        # next pawn-move/capture/mate from the position (loop supplies;
        # censored rows carry ml_target*cap, never -1/ignore). Huber loss
        # (delta 10 / scale 20, Lc0 huber_delta=10 plies), weight
        # progress_w default 0.05. Appended LAST in forward (index 11).
        # New keys random-init via strict=False (reply precedent).
        self.pg_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.pg_bn = nn.BatchNorm2d(aux_channels)
        self.pg_fc1 = nn.Linear(aux_channels * 64, 32)
        self.pg_fc2 = nn.Linear(32, 1)
        # V20 D1+D3 (Batch A, train core): score-margin heads + action-value
        # head, APPENDED at END (forward indices 12/13/14; 0-11 identical).
        # - score_mean: scalar pawns, LINEAR out (Huber d=1.0 in loss).
        #   Target = final-board piece-value diff, white-relative, in pawns
        #   (Agent B shape_targets; Batch A consumes only).
        # - score_stdev: scalar, softplus(raw) > 0 (Huber d=1.0 in loss).
        # - av_logits ("av"): full 4096 FC from the policy trunk features
        #   (p), like the soft head (tiny, train-only). Target = Stockfish
        #   top-Q per move, float32 centipawn clip +-1500 (Agent C
        #   build_av.py; "/100 -> pawns" scale lives in compute_loss).
        # Training gradient ONLY in V20: NEVER wired into search/veto
        # (spec D1). New keys random-init via strict=False (reply prec).
        # Probe gate (D1): linear-probe R2>0.3 on a buffer sample required
        # before unfreezing the backbone with score_w>0; until the probe
        # runs, production must keep score_w=0 (ramp helper defaults to 0
        # when no step is passed). Result: probe NOT yet run (Batch A).
        # FORBIDDEN (1 line each, not implemented here): F1 no TB in MCTS.
        # F2 no SL fine-tune phase. F3 no multi-step policy targets. F4 no
        # TB policy boost. F5 no full no-resign. F6 no transformer body. F7
        # no Maia/human regularizer. F8 KL/rehearsal/panel/lineage/veto/
        # mate-finish/surprise all kept (modify ok, removal forbidden). F9
        # arity 14->18 updates replay+train here; loop/warmstart/tests are
        # Agents B/C (exact names score_mean/score_stdev/root_q/av_logits).
        self.sm_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.sm_bn = nn.BatchNorm2d(aux_channels)
        self.sm_fc1 = nn.Linear(aux_channels * 64, 32)
        self.sm_fc2 = nn.Linear(32, 1)
        self.ss_conv = nn.Conv2d(channels, aux_channels, 1, bias=False)
        self.ss_bn = nn.BatchNorm2d(aux_channels)
        self.ss_fc1 = nn.Linear(aux_channels * 64, 32)
        self.ss_fc2 = nn.Linear(32, 1)
        self.av_fc = nn.Linear(policy_channels * 64, actions)
        # v19 BN momentum 0.02 (future updates only; old checkpoints'
        # running stats still load; momentum affects future updates only).
        self.apply(lambda m: setattr(m, "momentum", 0.02)
                   if isinstance(m, nn.BatchNorm2d) else None)

    def forward(self, x):
        x = F.relu(self.trunk_bn(self.trunk_in(x)))
        x = self.blocks(x)
        # v19 D1: conv policy head (stm-oriented codec: flatten of
        # [B,64,8,8] is logit[from*64+to]).
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = F.relu(self.p_bn2(self.p_conv2(p)))
        logits = self.p_fc(p).flatten(1)
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        wdl = self.v_fc2(v)
        a = F.relu(self.a_bn(self.a_conv(x)))
        a = F.relu(self.a_fc1(a.flatten(1)))
        material = self.a_fc2(a).squeeze(1)
        m = F.relu(self.m_bn(self.m_conv(x)))
        m = F.relu(self.m_fc1(m.flatten(1)))
        moves_left = torch.sigmoid(self.m_fc2(m)).squeeze(1)
        ownership = torch.sigmoid(self.o_conv(x)).squeeze(1)
        sc = F.relu(self.s_bn(self.s_conv(x)))
        sc = F.relu(self.s_fc1(sc.flatten(1)))
        margin = self.s_fc2(sc).squeeze(1)
        mb = F.relu(self.b_bn(self.b_conv(x)))
        mb = F.relu(self.b_fc1(mb.flatten(1)))
        mobility = torch.sigmoid(self.b_fc2(mb)).squeeze(1)
        kb = F.relu(self.k_bn(self.k_conv(x)))
        kb = F.relu(self.k_fc1(kb.flatten(1)))
        safety = torch.sigmoid(self.k_fc2(kb)).squeeze(1)
        rb = F.relu(self.r_bn(self.r_conv(x)))
        rb = F.relu(self.r_fc1(rb.flatten(1)))
        reply = self.r_fc2(rb)
        # v19: APPEND new heads at END (keep indices 0-8 identical).
        # soft reuses the policy trunk features (p); check is aux-style.
        soft = self.p_soft_fc(p.flatten(1))
        cb = F.relu(self.c_bn(self.c_conv(x)))
        cb = F.relu(self.c_fc1(cb.flatten(1)))
        check = torch.sigmoid(self.c_fc2(cb)).squeeze(1)
        # v19 D4: progress was LAST in V19 (index 11). Linear out (raw plies).
        gb = F.relu(self.pg_bn(self.pg_conv(x)))
        gb = F.relu(self.pg_fc1(gb.flatten(1)))
        progress = self.pg_fc2(gb).squeeze(1)
        # V20 D1+D3: APPEND score/av heads at END (indices 12/13/14; 0-11
        # identical to V19). score_mean linear (pawns); score_stdev =
        # softplus(raw) > 0; av_logits FC from policy trunk (train-only).
        sm = F.relu(self.sm_bn(self.sm_conv(x)))
        sm = F.relu(self.sm_fc1(sm.flatten(1)))
        score_mean = self.sm_fc2(sm).squeeze(1)
        ss = F.relu(self.ss_bn(self.ss_conv(x)))
        ss = F.relu(self.ss_fc1(ss.flatten(1)))
        score_stdev = F.softplus(self.ss_fc2(ss)).squeeze(1)
        av_logits = self.av_fc(p.flatten(1))
        return logits, wdl, material, moves_left, ownership, margin, \
            mobility, safety, reply, soft, check, progress, score_mean, \
            score_stdev, av_logits

    @torch.jit.export
    def forward_inf(self, x):
        """v19 D2 native inference forward: (logits, wdl, safety,
        moves_left) only — the 4 outputs search/evaluate need. Bitwise
        identical ops to forward()'s corresponding heads (same modules,
        same order); skips policy-soft/check + 5 scalar aux heads.
        Scripted directly by callers (no wrapper module); falls back to
        full forward elsewhere (not here)."""
        x = F.relu(self.trunk_bn(self.trunk_in(x)))
        x = self.blocks(x)
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = F.relu(self.p_bn2(self.p_conv2(p)))
        logits = self.p_fc(p).flatten(1)
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        wdl = self.v_fc2(v)
        kb = F.relu(self.k_bn(self.k_conv(x)))
        kb = F.relu(self.k_fc1(kb.flatten(1)))
        safety = torch.sigmoid(self.k_fc2(kb)).squeeze(1)
        m = F.relu(self.m_bn(self.m_conv(x)))
        m = F.relu(self.m_fc1(m.flatten(1)))
        moves_left = torch.sigmoid(self.m_fc2(m)).squeeze(1)
        return logits, wdl, safety, moves_left

    @torch.jit.export
    def forward_pv(self, x):
        """Inference-only forward: policy logits + Q value ONLY (v16.1,
        v18 WDL). Bitwise-identical ops to forward()'s policy + WDL
        outputs; skips the 7 aux heads that search never queries."""
        x = F.relu(self.trunk_bn(self.trunk_in(x)))
        x = self.blocks(x)
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = F.relu(self.p_bn2(self.p_conv2(p)))
        logits = self.p_fc(p).flatten(1)
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        return logits, wdl_q(self.v_fc2(v))

    def masked_log_probs(self, x, mask: torch.Tensor):
        out = self.forward(x)
        logits = out[0].masked_fill(~mask, -1e9)
        return F.log_softmax(logits, dim=1), wdl_q(out[1]), out[2], out[3]


def _pad_to(tensor, shape, fill):
    """Zero-/fill-pad a tensor up to shape (dims beyond tensor rank must
    already match). Only grows; never shrinks (caller guarantees)."""
    import torch as _t
    out = _t.full(shape, fill, dtype=tensor.dtype)
    sl = tuple(slice(0, s) for s in tensor.shape)
    out[sl] = tensor.cpu()
    return out


def grow_state_dict(own, old):
    """Growth surgery: fit an old checkpoint into a BIGGER architecture
    with behavior preserved (new capacity starts neutral, not random).
    - plane growth (trunk_in in-channels, v7/v9/v12 precedent): zero-pad.
    - channel growth (conv out/in-channels, v16 64->128): zero-pad rows
      (new filters dead) and cols (old outputs untouched).
    - depth growth (new residual blocks, v16 4->6): ALL-ZERO weights +
      zero BN (weight 0? NO — weight 1/bias 0 with zero conv output is
      still 0; but zero conv outputs need no BN help). Zero convs make
      the block output 0, so relu(0 + r) = r on non-negative trunk
      activations: EXACT identity in eval AND train mode.
    - new heads (k_* safety pattern): absent keys stay absent (random
      init via strict=False) — small heads train fast; zeroing them
      would bias the trunk through shared gradients... (random keeps
      them symmetric).
    - BatchNorm: weight->1.0, bias->0.0, running_mean->0.0,
      running_var->1.0 pads (standard init for new channels).
    - anything SHRINKING: ValueError (refuse silent shrink).
    Returns a grown dict with every own-key present (strictloadable)
    except absent new-head keys. v7/v9/v12 plane behavior unchanged.
    """
    import re as _re
    import torch as _t
    old_blocks = {int(m.group(1)) for k in old
                  for m in [_re.match(r"blocks\.(\d+)\.", k)] if m}
    max_old = max(old_blocks) if old_blocks else -1
    grown = {}
    for k, new_t in own.items():
        if k not in old:
            m = _re.match(r"blocks\.(\d+)\.", k)
            if m and int(m.group(1)) > max_old:
                # New residual block: all zeros (weights, BN, biases,
                # running stats get neutral running values below).
                # A random first transform with a zero final BN scale is
                # an identity initially, but can learn after the first step.
                if ".b2.weight" in k or ".b2.bias" in k:
                    grown[k] = _t.zeros_like(new_t)
                else:
                    grown[k] = new_t.clone()
                continue
            continue  # new head: leave random (strict=False skips)
        v = old[k]
        if tuple(v.shape) == tuple(new_t.shape):
            grown[k] = v
            continue
        if k in ("p_fc.weight", "p_fc.bias"):
            # v19 D1 fresh conv-head doctrine: p_fc was a flat Linear
            # (2-D [4096, policy*64] / [4096]) and is now a Conv1x1
            # (4-D [64, policy, 1, 1] / [64]) — incompatible by design
            # (V19 trains fresh from a new warm-start). Zero it (neutral
            # fresh head) instead of crashing on the rank change or
            # silently shrinking, so old checkpoints load for measurement.
            grown[k] = _t.zeros_like(new_t)
            continue
        if len(new_t.shape) != len(v.shape):
            raise ValueError(f"{k}: rank change, refusing surgery")
        if any(n < o for n, o in zip(new_t.shape, v.shape)):
            raise ValueError(
                f"{k}: new shape {tuple(new_t.shape)} shrinks {tuple(v.shape)}"
                " — refusing silent shrink")
        if "running_var" in k:
            grown[k] = _pad_to(v, tuple(new_t.shape), 1.0)
        elif k.endswith(".weight") and v.dim() == 1:
            grown[k] = _pad_to(v, tuple(new_t.shape), 1.0)  # BN weight
        elif k in ("v_fc2.weight", "v_fc2.bias") and tuple(v.shape) != \
                tuple(new_t.shape):
            # v18 scalar->WDL cut: a scalar-era row padded to [old,0,0]
            # would read a BIASED Q (e.g. old<<0 -> Q~-0.5, not -1).
            # Zero the whole head instead: uniform WDL, Q=0, neutral start
            # (honest for measurement, safe for training).
            grown[k] = _t.zeros_like(new_t)
        elif v.dim() >= 2 and new_t.shape[0] > v.shape[0]:
            # New output features must activate so their outgoing zero
            # connections can receive a gradient. Old outputs stay exact.
            padded = new_t.clone()
            padded[:v.shape[0]] = 0
            padded[tuple(slice(0, n) for n in v.shape)] = v
            grown[k] = padded
        else:
            grown[k] = _pad_to(v, tuple(new_t.shape), 0.0)
    return grown


def load_weights(model, w, strict=False):
    """Tolerant loader: unwraps training dicts; older checkpoints load with
    strict=False (new heads stay random-init). GROWTH SURGERY via
    grow_state_dict (v7/v9/v12 plane growth, v16 channel+depth growth):
    transfer survives wider/deeper nets with behavior preserved."""
    import torch as _t
    from collections.abc import Mapping
    if isinstance(w, (str, __import__("os").PathLike)):
        w = _t.load(w, map_location="cpu", weights_only=False)
    if isinstance(w, dict) and "weights" in w:
        w = w["weights"]
    if not isinstance(w, Mapping) or not w or not all(_t.is_tensor(v) for v in w.values()):
        raise ValueError("Expected a nonempty model state dictionary")
    if not set(w).intersection(model.state_dict()):
        raise ValueError("Checkpoint contains no recognized model parameters")
    if not strict and isinstance(w, dict):
        own = model.state_dict()
        # grow_state_dict returns every own-key grown/padded/zeroed
        # except absent new-head keys (left random via strict=False).
        # Shrink attempts raise ValueError (refuse silent shrink).
        w = grow_state_dict(
            {k: v for k, v in own.items()}, dict(w))
    model.load_state_dict(w, strict=strict)
    return model


def load_inference_weights(model, checkpoint):
    """Exact inference load; allow V19 to omit unused V20 training heads.

    No parameter growth/padding or policy/value substitution. Missing or
    mismatched inference tensors still fail. forward_inf does not consume
    score-mean, score-stdev, or AV heads.
    """
    from collections.abc import Mapping
    if isinstance(checkpoint, (str, __import__("os").PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("weights", checkpoint)
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Expected model weights")
    own = model.state_dict()
    optional = ("sm_conv.", "sm_bn.", "sm_fc1.", "sm_fc2.",
                "ss_conv.", "ss_bn.", "ss_fc1.", "ss_fc2.", "av_fc.")
    merged = dict(weights)
    for name, value in own.items():
        if name not in merged and name.startswith(optional):
            merged[name] = value
    model.load_state_dict(merged, strict=True)
    return model


def load_optimizer_state(opt, ckpt: dict) -> str:
    """Restore matching optimizer groups and state through PyTorch's API.

    Architecture or parameter-name changes require a fresh optimizer or
    an explicit named migration; positional padding is not safe.
    """
    import torch as _t
    saved = ckpt.get("opt", None)
    if saved is None:
        raise ValueError("checkpoint carries no optimizer state")
    # Positional restoration is safe only when the complete group layout
    # and tensor shapes match. Reject architecture migrations rather than
    # attach a same-shaped moment to an unrelated parameter.
    current = opt.state_dict()
    groups = saved.get("param_groups", [])
    if len(groups) != len(opt.param_groups):
        raise ValueError("Optimizer group mismatch; explicit named migration required")
    for live, old in zip(opt.param_groups, groups):
        if len(live["params"]) != len(old["params"]):
            raise ValueError("Optimizer architecture changed; use fresh optimizer")
        if "param_names" in old and live.get("param_names") != old["param_names"]:
            raise ValueError("Optimizer parameter names changed; explicit migration required")
        for parameter, pid in zip(live["params"], old["params"]):
            for name, value in saved.get("state", {}).get(pid, {}).items():
                if _t.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                    raise ValueError(f"Optimizer shape mismatch for {name}; use fresh optimizer")
    opt.load_state_dict(saved)
    return "optimizer restored (validated groups and state shapes)"


def compute_loss(model, boards, target_pi, target_z, target_m, target_ml,
                 target_own, target_margin, target_mob, target_safe=None,
                 aux_w=0.1, ml_w=0.05, ent_w=0.0,
                 own_w=0.1, margin_w=0.05, mob_w=0.05, safe_w=0.05,
                 reply_w=0.05, target_reply=None, policy_w=None,
                 value_w=1.0, ent_reg=0.0, smooth_eps=0.0,
                 soft_w=0.3, check_w=0.01, target_check=None,
                 progress_w=0.05, target_progress=None, ml_mask=None,
                 target_score_mean=None, target_score_stdev=None,
                 target_av_logits=None, target_av=None,
                 target_root_q=None, root_q=None,
                 score_w=0.0, av_w=0.1, av_temp=2.0, av_scale=100.0,
                 td_lambda=0.5, av_mask=None, legal_mask=None, forward_out=None):
    # NOTE (audit §2): there is deliberately NO explicit L2 term here —
    # an `l2` parameter used to be threaded through but was never used in
    # the total. Regularization comes from Adam's weight_decay only.
    # v19: forward returns 11 outputs (..., reply, soft, check appended;
    # indices 0-8 identical to v18). Return is 13-tuple (11 + soft+check).
    # v19e D4: forward returns 12 (progress LAST, index 11); return is
    # 14-tuple (13 + progress loss at END). D24 code defaults: soft 0.3,
    # check 0.01, reply 0.05 (config, not code, sets reply 0.15),
    # progress 0.05. D5: ml_mask (array-like, None = all ones) multiplies
    # the ML term (played-out rows train M; truncated rows contribute 0
    # but keep the batch denominator, so yesterday == all-ones exactly).
    # V20 (Batch A): forward returns 15 (score_mean 12, score_stdev 13,
    # av_logits 14 appended at END; 0-11 identical). Return is 17-tuple
    # (old 14 + loss_score 14 + loss_av 15 + loss_td 16, all appended at
    # END so old indices 0-13 are stable). Code defaults: score_w 0.0
    # (yesterday = no score head; caller passes the ramp value), av_w 0.1,
    # av_temp (T) 2.0, td_lambda 0.5. Exact shared names: score_mean,
    # score_stdev, av_logits, root_q; losses-dict keys loss_score/loss_av/
    # loss_td (see train.losses_dict_from_compute_out).
    out = model(boards) if forward_out is None else forward_out
    logits, wdl, material, moves_left = out[0], out[1], out[2], out[3]
    own, margin, mob, safe, reply = out[4], out[5], out[6], out[7], out[8]
    # v19: new heads at END (strict-load: old 9-output mocks in
    # finish/safety paths index only 0-8, unaffected). Tolerate 9-output
    # models (old mocks) with graph-connected zeros for the new terms.
    # v19e: tolerate 11-output models (pre-progress) the same way.
    # V20: tolerate 12-output models (pre-score/av) the same way; full
    # V20 forward has 15 (progress 11, score_mean 12, score_stdev 13,
    # av_logits 14).
    if len(out) >= 15:
        soft_logits, check_pred, progress_pred = out[9], out[10], out[11]
        score_mean_pred, score_stdev_pred, av_pred = \
            out[12], out[13], out[14]
    elif len(out) >= 12:
        soft_logits, check_pred, progress_pred = out[9], out[10], out[11]
        score_mean_pred = material * 0.0 + 0.0
        score_stdev_pred = material * 0.0 + 0.0
        av_pred = logits * 0.0
    elif len(out) >= 11:
        soft_logits, check_pred = out[9], out[10]
        progress_pred = material * 0.0  # detached below via zero term
        score_mean_pred = material * 0.0 + 0.0
        score_stdev_pred = material * 0.0 + 0.0
        av_pred = logits * 0.0
    else:  # pragma: no cover - old mocks only
        soft_logits = logits
        check_pred = torch.sigmoid(material)
        progress_pred = material * 0.0
        score_mean_pred = material * 0.0 + 0.0
        score_stdev_pred = material * 0.0 + 0.0
        av_pred = logits * 0.0
    # v19 label smoothing: pi_smooth = (1-eps)*pi + eps*uniform_legal,
    # uniform over legal move count. Legal = target_pi>0 rows (MCTS pi
    # has zeros on illegal; one-hot warm-start has 1 nonzero). When
    # smooth_eps==0 (yesterday) pi_for_policy is target_pi exactly.
    if smooth_eps != 0.0 and legal_mask is not None:
        _legal = legal_mask.to(device=target_pi.device, dtype=torch.bool)
        _nlegal = _legal.sum(dim=1).clamp(min=1)
        _uniform = _legal.to(target_pi.dtype) / \
            _nlegal.unsqueeze(1).to(target_pi.dtype)
        pi_for_policy = (1.0 - smooth_eps) * target_pi + \
            smooth_eps * _uniform
    else:
        pi_for_policy = target_pi
    logp = F.log_softmax(logits, dim=1)
    # v18 fast/full split: policy_w rows (1 = full game, 0 = fast game).
    # Fast rows train value+aux only (KataGo: value wants many cheap
    # games, policy wants good games). None = all rows full (yesterday).
    if policy_w is None:
        policy_loss = -(pi_for_policy * logp).sum(dim=1).mean()
        _pw = None
    else:
        pw = policy_w.to(logp.dtype)
        policy_loss = -(pi_for_policy * logp * pw.unsqueeze(1)).sum() / \
            pw.sum().clamp(min=1.0)
        _pw = pw
    # v18 WDL value (Lc0 v0.21): scalar tanh replaced by 3-way
    # win/draw/loss CE. z>0 -> win, z<0 -> loss, else draw.
    # v19: multiplied by value_w (default 1.0 = yesterday).
    # V20 D2 TD blend (SAFE): when target_root_q (MCTS root Q, Agent B
    # shape_targets) is given, train WDL on z_blend = 0.5*z + 0.5*td with
    # td = td_lambda*z + (1-td_lambda)*root_q (td_lambda default 0.5, so
    # z_blend = 0.75*z + 0.25*root_q: final-z dominant by construction).
    # None (old callers/buffers, yesterday) -> pure z, bit-exact.
    # Aliases: root_q == target_root_q (single source of truth below).
    _rq = target_root_q if target_root_q is not None else root_q
    if _rq is None:
        z_for_wdl = target_z
        _td_cont = target_z.to(dtype=torch.float32).reshape(-1)
    else:
        _lam = float(td_lambda)
        _rqt = _rq.to(device=target_z.device,
                      dtype=torch.float32).reshape(-1)
        _zt = target_z.to(dtype=torch.float32).reshape(-1)
        _td = _lam * _zt + (1.0 - _lam) * _rqt
        z_for_wdl = 0.5 * _zt + 0.5 * _td
        _td_cont = (0.5 * _zt + 0.5 * _td).detach()
    # Scalar search Q cannot identify draw probability. Preserve outcome labels.
    z_for_wdl = target_z
    wdl_idx = torch.where(z_for_wdl > 0, torch.zeros_like(z_for_wdl),
                          torch.where(z_for_wdl < 0,
                                      torch.full_like(z_for_wdl, 2),
                                      torch.ones_like(z_for_wdl))).long()
    value_loss = F.cross_entropy(wdl, wdl_idx)
    aux_loss = F.mse_loss(material, target_m)
    # v19e D5: ml_mask (None = all ones = yesterday bit-exact). Per-row
    # squared error times mask, mean over the batch (masked rows add 0;
    # denominator stays B, so all-ones mask == plain MSE exactly).
    _ml_se = (moves_left - target_ml) ** 2
    if ml_mask is None:
        ml_loss = _ml_se.mean()
    else:
        _mm = ml_mask.to(device=_ml_se.device,
                         dtype=_ml_se.dtype).reshape(-1)
        ml_loss = (_ml_se * _mm.unsqueeze(-1)
                   if _ml_se.dim() > 1 else _ml_se * _mm).mean()
    # v10 dense spatial supervision: ownership is {-1,0,1} per square,
    # trained through the sigmoid with MSE (codebase-consistent); margin
    # and mobility are scalars like aux.
    own_loss = F.mse_loss(own, (target_own + 1.0) / 2.0)
    margin_loss = F.mse_loss(margin, target_margin)
    mob_loss = F.mse_loss(mob, target_mob)
    # v16 king safety: scalar sigmoid like mobility. None = pre-v16
    # caller: neutral 0.5 target (no gradient direction either way).
    if target_safe is None:
        target_safe = torch.full_like(mob, 0.5)
    safe_loss = F.mse_loss(safe, target_safe)
    # v18 opp-reply aux (64-way from-square, -100 = chain broken).
    # All-ignored batches yield zero — kept GRAPH-CONNECTED (reply*0)
    # so the head's params still draw optimizer state (state continuity
    # across growth surgery; a detached zero would leave 7 stateless
    # params and break the surgical-restore invariant).
    if target_reply is None:
        reply_loss = (reply * 0.0).sum()
    else:
        valid = (target_reply != -100)
        if bool(valid.any()):
            reply_loss = F.cross_entropy(reply[valid],
                                         target_reply[valid].long())
        else:
            reply_loss = (reply * 0.0).sum()
    # v19 soft-policy aux (T=4): target = (pi_for_policy)^(1/4)
    # normalized per row. Derived from the SMOOTHED pi so warmstart's
    # smooth_eps=0.05 flows to both heads consistently. Same policy_w
    # masking as the main policy (fast rows exclude policy-like signals;
    # value+scalar aux still train). Graph-connected even when masked.
    _st_unnorm = torch.pow(pi_for_policy.clamp(min=0.0), 0.25)
    _st_sum = _st_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-12)
    soft_target = _st_unnorm / _st_sum
    soft_logp = F.log_softmax(soft_logits, dim=1)
    if _pw is None:
        # rows with all-zero pi (e.g. fast-mask test with zeros) yield
        # zero target -> zero loss, still graph-connected via soft_logp.
        soft_loss = -(soft_target * soft_logp).sum(dim=1).mean()
    else:
        soft_loss = -(soft_target * soft_logp *
                      _pw.unsqueeze(1)).sum() / _pw.sum().clamp(min=1.0)
    # v19 check head: P(stm in check), BCE. None = no targets (old
    # callers/buffers) -> graph-connected zero (reply precedent).
    if target_check is None:
        check_loss = (check_pred * 0.0).sum()
    else:
        _tc = target_check.to(device=check_pred.device,
                              dtype=check_pred.dtype)
        check_loss = F.binary_cross_entropy(check_pred, _tc)
    # v19e D4 progress head: plies until next pawn-move/capture/mate
    # (raw plies, linear out). Huber delta 10 / scale 20 (Lc0
    # huber_delta=10 plies): huber/20 keeps the term O(1)-ish next to
    # the other aux terms; weighted by progress_w (default 0.05).
    # None = no targets (SL warm-start, old callers) -> graph-connected
    # zero (reply precedent: progress_pred*0 keeps optimizer state).
    if target_progress is None:
        progress_loss = (progress_pred * 0.0).sum()
    else:
        _tp = target_progress.to(device=progress_pred.device,
                                 dtype=progress_pred.dtype).reshape(-1)
        _pp = progress_pred.reshape(-1)
        valid = torch.isfinite(_tp) & (_tp >= 0)
        progress_loss = F.huber_loss(_pp[valid], _tp[valid], delta=10.0) / 20.0 if valid.any() else (_pp * 0.0).sum()
    # V20 D1 score-margin head (CAREFUL, probe-first): Huber d=1.0 on both
    # scalars; loss_score = (huber(mean) + huber(stdev)) / 2 keeps the term
    # O(1)-ish next to the other aux terms; weighted by score_w (ramp
    # 0->0.05/5000 steps in train.py; default 0.0 here = yesterday). None
    # targets (old callers/buffers) -> graph-connected zero (reply prec:
    # score_mean_pred*0 + score_stdev_pred*0 keeps optimizer state on BOTH
    # heads — stdev params must also draw grad, else the surgical-restore
    # invariant (no stateless params) breaks; never a detached zero).
    # NO search/veto read in V20 (training gradient only, spec D1).
    if target_score_mean is None and target_score_stdev is None:
        score_loss = (score_mean_pred * 0.0).sum() + \
            (score_stdev_pred * 0.0).sum()
    else:
        _smp = score_mean_pred.reshape(-1)
        _ssp = score_stdev_pred.reshape(-1)
        _terms = []
        # Missing sides contribute graph-connected zeros OUTSIDE the mean
        # (same continuity invariant as above; adding 0.0 keeps the loss
        # scale of the present terms exactly).
        _zero_terms = []
        if target_score_mean is not None:
            _tsm = target_score_mean.to(
                device=_smp.device, dtype=_smp.dtype).reshape(-1)
            _valid = torch.isfinite(_tsm)
            _terms.append(F.huber_loss(_smp[_valid], _tsm[_valid], delta=1.0) if bool(_valid.any()) else _smp.sum()*0.0)
        else:
            _zero_terms.append((score_mean_pred * 0.0).sum())
        if target_score_stdev is not None:
            _tss = target_score_stdev.to(
                device=_ssp.device, dtype=_ssp.dtype).reshape(-1)
            _valid = torch.isfinite(_tss)
            _terms.append(F.huber_loss(_ssp[_valid], _tss[_valid], delta=1.0) if bool(_valid.any()) else _ssp.sum()*0.0)
        else:
            _zero_terms.append((score_stdev_pred * 0.0).sum())
        score_loss = sum(_terms) / max(1, len(_terms)) + sum(_zero_terms)
    # V20 D3 action-value distillation (CAREFUL, offline batch): av_w *
    # MSE(softmax(net_av/T) || softmax(sf_av/T)), T=av_temp default 2.0.
    # Teacher stored as float32 centipawn clip +-1500 (Agent C build_av);
    # av_scale=100 converts cp -> pawns before softmax (single knob; pass
    # av_scale=1.0 if Agent C stores normalized Q). Softmax is shift-
    # invariant so +-15 pawns never overflows (stable max-subtraction).
    # Legal-moves masking: illegal teacher rows read -1500 (clip floor) ->
    # ~0 mass, i.e. masked in effect; student softmax runs over full 4096
    # (same support, MSE well-defined). Aliases: target_av ==
    # target_av_logits. None (old callers; rows without AV) ->
    # graph-connected zero. All-zero teacher rows (replay legacy default)
    # also read zero (masked: genuine SF vectors never have zero spread).
    _avt = target_av_logits if target_av_logits is not None else target_av
    if _avt is None:
        av_loss = (av_pred * 0.0).sum()
    else:
        _avt_t = _avt.to(device=av_pred.device,
                         dtype=av_pred.dtype)
        # NaN means unlabelled, including a completely unlabelled row.
        # Explicit masks preserve genuinely labelled zero-centipawn moves.
        _mask = torch.isfinite(_avt_t) if av_mask is None else av_mask.to(
            device=av_pred.device, dtype=torch.bool) & torch.isfinite(_avt_t)
        if av_mask is None:
            # Legacy all-zero rows carried no labels.
            _legacy_missing = torch.isfinite(_avt_t).all(1, keepdim=True) & (torch.nan_to_num(_avt_t).abs().sum(1, keepdim=True) == 0)
            _mask &= ~_legacy_missing
        _valid = _mask.any(dim=1)
        _T = max(1e-6, float(av_temp))
        if bool(_valid.any()):
            _mask = _mask[_valid]
            _student = (av_pred[_valid] / _T).masked_fill(~_mask, -1e9)
            _teacher = (torch.nan_to_num(_avt_t[_valid]) / float(av_scale) / _T).masked_fill(~_mask, -1e9)
            _prob = F.softmax(_teacher, dim=1)
            av_loss = F.kl_div(F.log_softmax(_student, dim=1), _prob,
                               reduction="batchmean") * (_T * _T)
        else:
            av_loss = (av_pred * 0.0).sum()
    # V20 D2 loss_td diagnostic: MSE(Q_pred, td_target_continuous) where
    # Q_pred = P(W)-P(L) from the WDL head and td_target = z_blend above
    # (== z when root_q is None, so yesterday == MSE(Q, z) exactly).
    # Diagnostic ONLY (not added to total; the TD gradient flows through
    # value_loss on the blended WDL target). Detached in the return like
    # every other term; always finite when the forward is finite.
    try:
        _q_pred = (F.softmax(wdl, dim=1)[:, 0] -
                   F.softmax(wdl, dim=1)[:, 2]).reshape(-1)
        _td_t = _td_cont.reshape(-1).to(device=_q_pred.device,
                                        dtype=_q_pred.dtype)
        td_loss = F.mse_loss(_q_pred, _td_t)
    except Exception:
        td_loss = (material * 0.0).sum() + 0.0 * value_loss
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * logp).sum(dim=1).mean()
    # v19 legal-only entropy bonus: -ent_reg * entropy over LEGAL moves
    # only (masked softmax over target_pi>0 rows). ent_reg==0 (yesterday)
    # contributes exactly 0 (no extra forward cost beyond this branch).
    if ent_reg != 0.0 and legal_mask is not None:
        _lm = legal_mask.to(device=target_pi.device, dtype=torch.bool)
        # rows with no legal (zeros pi) contribute 0 entropy.
        _has = _lm.sum(dim=1) > 0
        _logits_m = logits.masked_fill(~_lm, -1e9)
        _logp_m = F.log_softmax(_logits_m, dim=1)
        _p_m = _logp_m.exp()
        _ent_legal_rows = -(_p_m * _logp_m).sum(dim=1)
        _ent_legal_rows = torch.where(_has, _ent_legal_rows,
                                      torch.zeros_like(_ent_legal_rows))
        legal_entropy = _ent_legal_rows.mean()
    else:
        legal_entropy = (logits * 0.0).sum(dim=1).mean()
    total = policy_loss + value_w * value_loss + aux_w * aux_loss + \
        ml_w * ml_loss + own_w * own_loss + margin_w * margin_loss + \
        mob_w * mob_loss + safe_w * safe_loss + reply_w * reply_loss + \
        soft_w * soft_loss + check_w * check_loss + \
        progress_w * progress_loss - ent_w * entropy - \
        ent_reg * legal_entropy + score_w * score_loss + \
        av_w * av_loss
    # V20: return is 17-tuple (old 14 + score/av/td appended at END; old
    # indices 0-13 stable). loss_td is a DIAGNOSTIC MSE(Q, td_target): the
    # TD gradient flows through value_loss on the blended WDL target (D2),
    # not through a separate weighted term (avoids double-counting value;
    # keeps final-z dominant). All three new terms are finite whenever the
    # forward is finite (None targets -> graph-connected zero, reply prec).
    return (total, policy_loss.detach(), value_loss.detach(),
            aux_loss.detach(), ml_loss.detach(), entropy.detach(),
            own_loss.detach(), margin_loss.detach(), mob_loss.detach(),
            safe_loss.detach(), reply_loss.detach(),
            soft_loss.detach(), check_loss.detach(),
            progress_loss.detach(), score_loss.detach(),
            av_loss.detach(), td_loss.detach())
