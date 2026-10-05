"""Training step + short demo.
Loss: MSE(value)+CE(policy)+0.1*MSE(aux)+0.05*MSE(movesleft)-ent*entropy.
L2 lives in Adam's weight_decay, not in the loss (audit §2)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LOCAL_CONFIG
from .model import AlphaZeroNet, compute_loss
from .replay import ReplayBuffer
from .selfplay import play_game


# V20 loss-dict keys shared with Agent C tests (EXACT): loss_score (D1
# score-margin Huber), loss_av (D3 Stockfish distillation), loss_td (D2 TD
# diagnostic MSE(Q, td_target)). train_step return tuple carries them at
# indices -4/-3/-2 with KL LAST at -1; use losses_dict_from_train_step().
V20_LOSS_KEYS = ("loss_score", "loss_av", "loss_td")

# V20 shared field names (EXACT, Agent A<->C contract): score_mean,
# score_stdev, av_logits, root_q, td_blend. td_blend = the continuous TD
# target z_blend = 0.5*z + 0.5*(td_lambda*z + (1-td_lambda)*root_q).
V20_FIELD_NAMES = ("score_mean", "score_stdev", "av_logits", "root_q",
                   "td_blend")


def score_w_for_step(step, w_max: float = 0.05,
                     ramp_steps: int = 5000) -> float:
    """V20 D1 score-weight ramp: 0 -> w_max over the first ramp_steps
    optimizer steps (linear). Step 0 reads 0.0 (yesterday bit-exact until
    the probe gate passes and the caller starts passing score_step)."""
    try:
        _s = float(step)
    except Exception:
        return 0.0
    if not _s > 0:
        return 0.0
    try:
        _r = max(1, int(ramp_steps))
    except Exception:
        _r = 5000
    return float(w_max) * min(1.0, _s / float(_r))


def losses_dict_from_train_step(out) -> dict:
    """Map a V20 train_step 16-tuple to {"loss_score","loss_av","loss_td"}
    (floats; KL LAST at out[-1] is NOT in the dict). Legacy 13-tuples
    (pre-V20 callers) read 0.0 for all three (never raises)."""
    try:
        _n = len(out)
    except Exception:
        return {k: 0.0 for k in V20_LOSS_KEYS}
    if _n >= 16:
        # V20 order: ..., prog(11), score(12), av(13), td(14), kl(15 LAST).
        return {"loss_score": float(out[12]), "loss_av": float(out[13]),
                "loss_td": float(out[14])}
    return {k: 0.0 for k in V20_LOSS_KEYS}


def losses_dict_from_compute_out(out) -> dict:
    """Map a V20 compute_loss 17-tuple to the same dict (score 14, av 15,
    td 16 appended at END). Legacy 14-tuples read 0.0 (never raises)."""
    try:
        _n = len(out)
    except Exception:
        return {k: 0.0 for k in V20_LOSS_KEYS}
    if _n >= 17:
        return {"loss_score": float(out[14]), "loss_av": float(out[15]),
                "loss_td": float(out[16])}
    return {k: 0.0 for k in V20_LOSS_KEYS}


# v19 a-h mirror permutation (file flip on both squares):
# (fr,to) -> (fr^7,to^7). Self-inverse involution, so gather and scatter
# coincide. Computed once on CPU; moved to the batch device on use.
_AUG_PERM_CPU = None


def _aug_perm_cpu():
    global _AUG_PERM_CPU
    if _AUG_PERM_CPU is None:
        import torch as _t
        _perm = _t.empty(4096, dtype=_t.long)
        for _a in range(4096):
            from .game import decode_action, encode_action, action_promotion, _PROMOTION_DECODE
            _fr, _to = decode_action(_a)
            dx, dy = abs((_fr%8)-(_to%8)), abs((_fr//8)-(_to//8))
            impossible = dx and dy and dx != dy and sorted((dx,dy)) != [1,2]
            _perm[_a] = _a if impossible and _a not in _PROMOTION_DECODE else encode_action(_fr ^ 7, _to ^ 7, action_promotion(_a))
        _AUG_PERM_CPU = _perm
    return _AUG_PERM_CPU


def augment_batch(boards, pis, owns, reply, teacher_probs=None):
    """v19 a-h mirror augmentation (deterministic flip; caller gates p=0.5).

    - flip planes left-right (boards last dim);
    - SWAP castling planes 16<->17 and 18<->19 AFTER flip (mirror swaps
      kingside/queenside);
    - permute pi actions (fr,to)->(fr^7,to^7);
    - fliplr ownership (8,8);
    - reply from-squares file-flip (sq -> sq ^ 7), keep -100 as -100;
    - scalars (z/m/ml/margin/mob/safe) and WDL idx invariant (untouched);
    - policy_w per-row scalars invariant under whole-batch flip (never
      reordered, so no realignment needed);
    - teacher_probs (if given) permuted exactly like pi (KL stays aligned);
    - kl_mask rows (if any) are per-row scalars, order preserved (no-op).
    Never reorders rows: spatial tensors + matching pi/ownership/reply
    (+teacher) stay aligned. Flip twice == identity.
    Returns (boards, pis, owns, reply) or (..., teacher) when teacher given.
    """
    import torch as _t
    b = _t.flip(boards, dims=[-1])
    _P = b.shape[1]
    if _P > 28:
        b[:, 28] = boards[:, 28]  # fixed coordinates, not a board feature
    if _P > 17:
        _tmp = b[:, 16].clone()
        b[:, 16] = b[:, 17]
        b[:, 17] = _tmp
    if _P > 19:
        _tmp = b[:, 18].clone()
        b[:, 18] = b[:, 19]
        b[:, 19] = _tmp
    _perm = _aug_perm_cpu().to(pis.device)
    pi2 = pis[:, _perm]
    o2 = _t.flip(owns, dims=[-1])
    r2 = reply.clone()
    _m = (reply != -100)
    if bool(_m.any()):
        r2[_m] = reply[_m] ^ 7
    if teacher_probs is None:
        return b, pi2, o2, r2
    t2 = teacher_probs[:, _perm.to(teacher_probs.device)]
    return b, pi2, o2, r2, t2


def train_step(model, optimizer, boards, pis, zs, ms, mls, owns, margins,
                 mobs, safes=None, reply=None, policy_w=None, clip=1.0,
                 aux_w=0.1, ml_w=0.05, ent_w=0.0,
                 own_w=0.1, margin_w=0.05, mob_w=0.05, safe_w=0.05,
                 reply_w=0.05,
                 soft_w=0.3, check_w=0.01, value_w=1.0, ent_reg=0.0,
                 smooth_eps=0.0, checks=None,
                 teacher_probs=None, kl_w=0.0, kl_mask=None,
                 augment=False, progress=None, progress_w=0.05,
                 ml_mask=None,
                 score_mean=None, score_stdev=None, av_logits=None,
                 av=None, root_q=None, target_root_q=None,
                 score_w=None, score_step=None, score_w_max=0.05,
                 score_ramp_steps=5000, td_lambda=0.5, av_w=0.1,
                 av_temp=2.0, av_scale=100.0, av_mask=None, legal_mask=None):
    # V20 (Batch A): new tensor kwargs use the EXACT shared names
    # (score_mean/score_stdev/av_logits/root_q; av and target_root_q are
    # accepted aliases). score_w ramp 0->score_w_max over score_ramp_steps
    # (default 0.05/5000): explicit score_w wins; elif score_step is not
    # None the ramp computes it; else 0.0 (yesterday bit-exact). td blend
    # via td_lambda (0.5) + root_q (None = pure z, yesterday). av_w 0.1,
    # T=av_temp 2.0. Return is 16-tuple (old 13 + score/av/td inserted
    # BEFORE KL; KL stays LAST at index -1/15). FORBIDDEN (1 line each, not
    # implemented): F1 no TB in MCTS. F2 no SL fine-tune. F3 no multi-step
    # policy targets. F4 no TB policy boost. F5 no full no-resign (D5 is
    # Agent B). F6 no transformer. F7 no Maia regularizer. F8 anchor/
    # rehearsal/panel/lineage/veto/mate-finish/surprise kept. F9 arity
    # 14->18 with legacy tolerance (None targets -> zero loss); loop/
    # warmstart/tests moves are Agents B/C. REMOVED (R1 only): stratified
    # sampling lives in replay.py; R2 game-LR, R3 resign, R4 veto default,
    # R5 MLH thr are NOT touched here (outside R1 is forbidden).
    # Derive full legal support and check labels from rule-complete inputs.
    # Never substitute positive visit support for legality.
    if boards.shape[1] >= 21 and (legal_mask is None or checks is None):
        from .game import state_from_encoding
        _states = [state_from_encoding(x) for x in boards.detach().cpu().numpy()]
        if legal_mask is None:
            import numpy as np
            legal_mask = torch.from_numpy(np.stack([s.legal_mask() for s in _states]))
        if checks is None:
            checks = torch.tensor([float(s.board.is_check()) for s in _states])
    dev = next(model.parameters()).device
    boards = boards.to(dev)
    pis, zs, ms, mls = pis.to(dev), zs.to(dev), ms.to(dev), mls.to(dev)
    owns, margins, mobs = owns.to(dev), margins.to(dev), mobs.to(dev)
    if safes is None:
        # v16: old callers/buffers without safety default neutral (0.5).
        safes = torch.full_like(mobs, 0.5)
    else:
        safes = safes.to(dev)
    if reply is None:
        # v18: no reply targets (old callers) -> all-ignored -> zero loss.
        reply = torch.full_like(mobs, -100, dtype=torch.long)
    else:
        reply = reply.to(dev)
    if policy_w is None:
        # v18: all rows full (old callers) -> unmasked policy.
        policy_w = torch.ones_like(mobs)
    else:
        policy_w = policy_w.to(dev)
    if checks is None:
        checks_t = None
    else:
        checks_t = checks.to(dev)
    # v19e D4/D5: progress targets (plies to next pawn-move/capture/mate;
    # None = no targets -> graph-connected zero in compute_loss) and
    # ml_mask (None = all ones = yesterday). Tuple building lives in
    # selfplay/replay (another agent); train_step only consumes.
    if progress is None:
        progress_t = None
    else:
        progress_t = progress.to(dev)
    if ml_mask is None:
        ml_mask_t = None
    else:
        import torch as _t2
        ml_mask_t = _t2.as_tensor(ml_mask).to(dev)
    if teacher_probs is None:
        teacher_t = None
    else:
        teacher_t = teacher_probs.to(dev)
    if kl_mask is None:
        kl_mask_t = None
    else:
        import torch as _t
        kl_mask_t = _t.as_tensor(kl_mask).to(dev)
        if kl_mask_t.dtype != _t.bool:
            kl_mask_t = kl_mask_t.bool()
    # V20 new-target plumbing (None = legacy caller -> zero loss in
    # compute_loss, yesterday bit-exact). av_logits wins over av alias;
    # target_root_q wins over root_q alias. Defined BEFORE augment so the
    # av mirror permutes the device tensor (same 4096 codec as pi).
    _av_in = av_logits if av_logits is not None else av
    _rq_in = target_root_q if target_root_q is not None else root_q
    _sm_t = None if score_mean is None else score_mean.to(dev)
    _ss_t = None if score_stdev is None else score_stdev.to(dev)
    _av_t = None if _av_in is None else _av_in.to(dev)
    _rq_t = None if _rq_in is None else _rq_in.to(dev)
    if score_w is not None:
        _score_w = float(score_w)
    elif score_step is not None:
        _score_w = score_w_for_step(score_step, float(score_w_max),
                                    int(score_ramp_steps))
    else:
        _score_w = 0.0
    # v19 a-h mirror INSIDE train_step (whole-batch flip, p=0.5 when
    # enabled). Default False = yesterday bit-exact (global rule).
    # V20: av_logits shares pi's 4096 codec, so it permutes with pi under
    # the mirror (score/root_q/checks/policy_w/kl_mask rows are per-row
    # scalars, order preserved — no-op, like checks).
    if augment and boards.shape[1] >= 20 and not bool(boards[:, 16:20].any()):
        import random as _r
        if _r.random() < 0.5:
            if teacher_t is None:
                boards, pis, owns, reply = augment_batch(
                    boards, pis, owns, reply)
            else:
                boards, pis, owns, reply, teacher_t = augment_batch(
                    boards, pis, owns, reply, teacher_t)
            # checks/policy_w/kl_mask rows invariant under whole-batch
            # flip (per-row scalars, never reordered) — no-op.
            permutation = _aug_perm_cpu()
            if legal_mask is not None:
                legal_mask = legal_mask[:, permutation.to(legal_mask.device)]
            if av_mask is not None:
                av_mask = av_mask[:, permutation.to(av_mask.device)]
            if _av_t is not None:
                try:
                    _perm_av = _aug_perm_cpu().to(_av_t.device)
                    _av_t = _av_t[:, _perm_av]
                except Exception:
                    pass
    model.train()
    optimizer.zero_grad()
    _forward = model(boards)
    out = compute_loss(model, boards, pis, zs, ms, mls, owns, margins,
                       mobs, safes, aux_w=aux_w, ml_w=ml_w, ent_w=ent_w,
                       own_w=own_w, margin_w=margin_w, mob_w=mob_w,
                       safe_w=safe_w, reply_w=reply_w,
                       target_reply=reply, policy_w=policy_w,
                       value_w=value_w, ent_reg=ent_reg,
                       smooth_eps=smooth_eps, soft_w=soft_w,
                       check_w=check_w, target_check=checks_t,
                       progress_w=progress_w, target_progress=progress_t,
                       ml_mask=ml_mask_t,
                       target_score_mean=_sm_t,
                       target_score_stdev=_ss_t,
                       target_av_logits=_av_t,
                       target_root_q=_rq_t,
                       score_w=_score_w, av_w=float(av_w),
                       av_temp=float(av_temp), av_scale=float(av_scale),
                       td_lambda=float(td_lambda), av_mask=av_mask, legal_mask=legal_mask,
                       forward_out=_forward)
    total, pl, vl, al, ml, ent, ol, sl, bl, kl_safe, rl = out[:11]
    # out has 17 (V20: old 14 + score/av/td appended at END); train_step
    # returns 16 (old 13 terms with progress float + score/av/td floats
    # inserted BEFORE KL). KL stays at END (A4 invariant, now index -1).
    # soft/check live in total only (A-spec precedent).
    prog = out[13]
    _score_f = out[14]
    _av_f = out[15]
    _td_f = out[16]
    if teacher_t is None:
        # No teacher: graph-connected zero via total (policy head already
        # draws grad from policy_loss, so no Adam-state gap).
        kl_tensor = total * 0.0
        kl_float = 0.0
        total_kl = total
    else:
        # Student forward for KL (extra forward; batch is small).
        # KL = (teacher * (log teacher - logp_student)) masked to
        # kl_mask rows (None = all rows), term = kl_w * mean.
        # Graph-connected zero when kl_w==0 (0 * KL still draws grad).
        _s_out = _forward
        _s_logits = _s_out[0]
        _logp_s = F.log_softmax(_s_logits, dim=1)
        _teach = teacher_t.to(dtype=_s_logits.dtype)
        _log_t = torch.log(_teach.clamp(min=1e-12))
        _kl_rows = (_teach * (_log_t - _logp_s)).sum(dim=1)
        if kl_mask_t is None:
            _kl_mean = _kl_rows.mean()
        else:
            if bool((kl_mask_t).any()):
                _kl_mean = _kl_rows[kl_mask_t].mean()
            else:
                _kl_mean = (_s_logits * 0.0).sum() * 0.0
        kl_tensor = kl_w * _kl_mean
        kl_float = float(kl_tensor.detach())
        total_kl = total + kl_tensor
    total_kl.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    # V20 16-tuple: old 13 order preserved through prog (index 11), then
    # loss_score (12), loss_av (13), loss_td (14), KL LAST (15). Loop reads
    # _out[0:11] positionally + prog at [11] + KL at [-1], so old indexing
    # is undisturbed; use losses_dict_from_train_step(out) for the dict.
    return (float(total_kl.detach()), float(pl.detach()),
            float(vl.detach()), float(al.detach()), float(ml.detach()),
            float(ent.detach()), float(ol.detach()), float(sl.detach()),
            float(bl.detach()), float(kl_safe.detach()),
            float(rl.detach()), float(prog.detach()),
            float(_score_f.detach()), float(_av_f.detach()),
            float(_td_f.detach()), float(kl_float))


def demo_overfit(steps: int = 30):
    """Prove the net can learn: overfit one self-play game, loss must fall."""
    cfg = LOCAL_CONFIG
    torch.manual_seed(0)
    model = AlphaZeroNet(blocks=2, channels=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg.l2)
    ex, res = play_game(model, cfg, temp_moves=2)
    import numpy as np
    s = torch.from_numpy(np.stack([e[0] for e in ex]))
    pi = torch.from_numpy(np.stack([e[1] for e in ex]))
    z = torch.tensor([e[2] for e in ex], dtype=torch.float32)
    m = torch.tensor([e[3] for e in ex], dtype=torch.float32)
    ml = torch.tensor([e[4] for e in ex], dtype=torch.float32)
    n = len(ex)
    own = torch.zeros(n, 8, 8)
    margin = torch.zeros(n)
    mob = torch.zeros(n)
    safe = torch.zeros(n)
    losses = []
    for _ in range(steps):
        tot, *_ = train_step(
            model, opt, s, pi, z, m, ml, own, margin, mob, safe)
        losses.append(tot)
    print(f"overfit {len(ex)} positions {res}: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "training did not reduce loss"
    return losses


if __name__ == "__main__":
    demo_overfit()
