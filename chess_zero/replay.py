"""Replay buffer: stratified phase sampling with endgame guarantee (V20 D10).
Positions with <=10 total pieces route to a second deque at insert; batches
mix an endgame share from it, floored at 15% whenever endgame rows exist
(R1 replaces unstratified sampling; phase by piece count: endgame <=10,
else opening/middlegame in the main deque at their natural ratio).
Examples are 18-tuples (V20: existing 14 [encode, pi, z, material,
moves_left, ownership, margin, mobility, safety, reply_from, full_weight,
kl, P, ml_mask] + score_mean(14), score_stdev(15), root_q(16),
av_logits(17)). "KL last" (spec) refers to the train_step RETURN (KL stays
at the END there); in the stored tuple KL stays at index 11 (write-time
surprise weight, IGNORED by sample, exactly as V19). Older 8/9/11/12/14-
tuples in flight read neutral defaults (safe 0.5, reply -100, weight 1.0,
kl 0.0, P 0.0, mask 1.0, score_mean 0.0, score_stdev 1.0, root_q = z,
av = zeros(4096)) so mixed buffers never crash. sample() IGNORES kl (write-
time weight only) and returns 17 arrays (the 13 V19 training arrays +
score_mean, score_stdev, root_q, av_logits appended at END; old indices
0-12 stable). Surprise (B7) is per-head NORMALIZED on the kl head only:
w = 0.5 + 0.5*N*kl/sum(kl) (mean 1.0); the new score/av heads NEVER enter
the weighting (they must not hijack sampling, D10).
Storage (C11): planes/pi/ownership/av are stored as float16 (astype copy)
to halve buffer RAM; sample() casts back to float32. Scalars stay float32.
av stored as float16 [4096] centipawn (or int8 quantized, converted on
read); None (rows without AV) reads zeros(4096), which compute_loss masks
to zero loss (genuine SF vectors never have zero spread).
FORBIDDEN (1 line each, not implemented): F1 no TB in MCTS. F2 no SL fine-
tune. F3 no multi-step policy targets. F4 no TB policy boost. F5 no full
no-resign. F6 no transformer. F7 no Maia regularizer. F8 anchor/rehearsal/
panel/lineage/veto/mate-finish/surprise kept. F9 arity 14->18 with legacy
tolerance here; loop/warmstart/tests moves are Agents B/C (exact names
score_mean/score_stdev/root_q/av_logits). REMOVED (R1 only): unstratified
sampling is gone; R2 game-LR, R3 resign, R4 veto default, R5 MLH thr are
NOT touched here (outside R1 is forbidden).
"""
from __future__ import annotations

from collections import deque
import random
import numpy as np

EG_PIECES = 10

# V20 tuple contract (EXACT order; Agents A<->C share these names).
# 0-13 = V19 (kl at 11, P progress at 12, ml_mask at 13); 14-17 appended.
TUPLE_ARITY = 18
SCORE_MEAN_IDX = 14
SCORE_STDEV_IDX = 15
ROOT_Q_IDX = 16
AV_IDX = 17
AV_DIM = 4096
# int8-quantized AV (Agent C build_av, clip +-1500cp) -> float32 cp scale.
AV_INT8_SCALE = 12.0
# D10: endgame share floor per batch whenever endgame rows exist.
MIN_EG_FRAC = 0.15


def _phase_is_endgame(enc_row) -> bool:
    """Phase by piece count (first 12 planes are piece planes): <=10 men
    is endgame (routes to the eg deque at insert)."""
    try:
        return float(np.asarray(enc_row)[:12].sum()) <= EG_PIECES
    except Exception:
        return False


def _kl_surprise_weights(examples) -> list:
    """Per-head NORMALIZED surprise (D10): kl head only (tuple index 11),
    w = 0.5 + 0.5*N*kl/sum(kl), mean 1.0 so the expected write count is
    unchanged. Score/av heads NEVER enter (must not hijack sampling).
    Missing/legacy/non-finite/negative kl and zero-sum games read w=1."""
    n = len(examples)
    kls: list[float] = []
    for e in examples:
        try:
            _k = float(e[11]) if len(e) > 11 else 0.0
        except Exception:
            _k = 0.0
        if not np.isfinite(_k) or _k < 0.0:
            _k = 0.0
        kls.append(_k)
    _tot = float(sum(kls))
    if not np.isfinite(_tot) or _tot <= 0.0:
        return [1.0] * n
    return [0.5 + 0.5 * n * _k / _tot for _k in kls]


class SparsePolicy:
    """Lossless sparse storage of the existing float16 replay policy."""
    def __init__(self, dense):
        dense = np.asarray(dense, dtype=np.float16)
        self.indices = np.flatnonzero(dense).astype(np.uint16)
        self.values = dense[self.indices]
        self.shape = dense.shape
        self.dtype = self.values.dtype

    def __array__(self, dtype=None, copy=None):
        dense = np.zeros(self.shape, dtype=dtype or self.dtype)
        dense[self.indices] = self.values
        return dense

    @property
    def nbytes(self):
        return self.indices.nbytes + self.values.nbytes


class ReplayBuffer:
    def __init__(self, capacity: int = 50000, eg_capacity: int = 20000,
                 eg_frac: float = 0.0):
        if capacity < 1 or not 0 <= eg_frac <= 1 or eg_capacity < 0:
            raise ValueError("Invalid replay capacity or endgame fraction")
        end_capacity = min(eg_capacity, capacity, max(0, int(capacity * eg_frac)))
        self.buf: deque = deque(maxlen=capacity - end_capacity)
        self.eg: deque = deque(maxlen=end_capacity)
        self.eg_frac = eg_frac

    def __len__(self):
        return len(self.buf) + len(self.eg)

    def eg_len(self):
        return len(self.eg)

    def add_game(self, examples: list[tuple[np.ndarray, np.ndarray, float,
                                            float, float]]):
        # B7 surprise oversampling at write: per-example weight from the
        # GAME's kl distribution via _kl_surprise_weights (per-head
        # normalized, D10: score/av heads excluded by construction).
        # Missing/legacy/non-finite/negative kl and zero-sum games read
        # w=1 (write once). Copies = int(w) + (random.random() < frac),
        # capped at 3 per position (boring rows may draw 0 copies — that
        # IS the oversampling). Uses random.random (no seed plumbing).
        # C11 fp16 storage: planes/pi/ownership/av stored as float16
        # (astype copy); scalars stay float32; None av stays None.
        # sample() casts back to float32. V20 18-tuples pass through with
        # new scalars untouched (legacy rows keep working: sample fills
        # neutral defaults).
        _ws = _kl_surprise_weights(examples)
        for e, _w in zip(examples, _ws):
            try:
                _l = list(e)
                _l[0] = np.asarray(_l[0]).astype(np.float16)
                _l[1] = SparsePolicy(_l[1])
                _l[5] = np.asarray(_l[5]).astype(np.float16)
                if len(_l) > AV_IDX and _l[AV_IDX] is not None:
                    try:
                        _av = np.asarray(_l[AV_IDX])
                        if _av.dtype != np.int8:
                            _l[AV_IDX] = _av.astype(np.float16)
                    except Exception:
                        pass
                _e = tuple(_l)
            except Exception:
                _e = e
            _iv = int(_w)
            _copies = _iv + (1 if random.random() < (_w - _iv) else 0)
            if _copies > 3:
                _copies = 3
            for _ in range(_copies):
                if self.eg.maxlen and (not self.buf.maxlen or float(_e[0][:12].sum()) <= EG_PIECES):
                    self.eg.append(_e)
                else:
                    self.buf.append(_e)

    def _draw(self, source, n):
        return random.sample(source, n)

    def sample(self, batch_size: int):
        # V20 D10 stratification (R1): endgame share floored at MIN_EG_FRAC
        # (0.15) whenever the eg deque holds rows — eff = max(eg_frac,
        # 0.15). V19 configs (eg_frac 0.25) already exceed it; small-but-
        # nonzero settings are lifted to the floor; eg_frac=0 keeps
        # yesterday storage (no eg rows routed, nothing to draw -> main
        # only, bit-exact). Endgame = piece count <= EG_PIECES (first 12
        # planes). Guarantee binds "when available": clamped to what each
        # deque actually holds (v14 resume-skew clamp kept).
        _eff = self.eg_frac
        if len(self.eg) > 0:
            try:
                _eff = max(float(self.eg_frac), float(MIN_EG_FRAC))
            except Exception:
                _eff = float(MIN_EG_FRAC)
        n_eg = 0
        if _eff > 0 and len(self.eg) >= 1:
            import math
            n_eg = math.ceil(batch_size * _eff)
            n_eg = min(n_eg, batch_size - 1, len(self.eg))
        # Fill the remainder from the other stratum if one is undersized.
        n_eg = min(len(self.eg), max(n_eg, batch_size - len(self.buf)))
        # v14: clamp to what each deque actually holds. Right after a
        # resume the eg deque can outrun main (total >= batch while
        # main < n_main) — random.sample would raise ValueError mid-run.
        n_eg = min(n_eg, len(self.eg), batch_size)
        n_main = min(batch_size - n_eg, len(self.buf))
        if n_main + n_eg == 0:
            raise ValueError("Cannot sample empty replay")
        batch = self._draw(self.buf, n_main) if n_main else []
        if n_eg:
            batch = batch + self._draw(self.eg, n_eg)
        random.shuffle(batch)
        # C11: stored planes/pi/ownership/av are float16; the astype calls
        # below cast back to float32 (train numerics unaffected). Tuple
        # element 11 (kl) is IGNORED here (write-time weight only).
        # Element 12 (P progress) + 13 (ml_mask) ride along (V19 order in
        # sample is mask(11th array) then prog(12th) — kept stable below).
        # V20 appends score_mean(14)/score_stdev(15)/root_q(16)/av(17) as
        # the 14th-17th sample arrays (old indices 0-12 stable).
        # Legacy rows read neutral defaults (score_mean 0.0, score_stdev
        # 1.0, root_q = z so the TD blend == z exactly, av = zeros(4096)
        # which compute_loss masks to zero loss).
        s = np.stack([e[0] for e in batch]).astype(np.float32)
        pi = np.stack([e[1] for e in batch]).astype(np.float32)
        z = np.array([e[2] for e in batch], dtype=np.float32)
        m = np.array([e[3] for e in batch], dtype=np.float32)
        ml = np.array([e[4] for e in batch], dtype=np.float32)
        own = np.stack([e[5] for e in batch]).astype(np.float32)
        margin = np.array([e[6] for e in batch], dtype=np.float32)
        mob = np.array([e[7] for e in batch], dtype=np.float32)
        # v16: 9th element (king safety) if present; old 8-tuples in
        # flight read NEUTRAL 0.5 (0.0 would teach danger — same default
        # as train_step's missing-safes path).
        safe = np.array([e[8] if len(e) > 8 else 0.5 for e in batch],
                        dtype=np.float32)
        # v18: 10th (opp-reply from-square, -100 = chain broken) and
        # 11th (fast/full policy weight) with the same mixed-tuple
        # tolerance (old rows read -100 / 1.0 = full).
        reply = np.array([e[9] if len(e) > 9 else -100 for e in batch],
                         dtype=np.int64)
        pw = np.array([e[10] if len(e) > 10 else 1.0 for e in batch],
                      dtype=np.float32)
        # D5: 14th element (moves-left mask: 0.0 truncated, 1.0 played-out).
        ml_mask = np.array([e[13] if len(e) > 13 else 0.0 for e in batch],
                           dtype=np.float32)
        # D4: 13th element (plies-until-progress target). Legacy rows
        # (len <= 12) read censored-neutral 0.0 with... no: P is trained
        # with Huber against the target, so legacy default must not bias:
        # read -1.0 and let compute_loss mask negatives to zero loss?
        # Simplest honest: legacy default 0.0 (no-progressdone positions
        # are rare in legacy buffers; new rows always carry P).
        prog = np.array([e[12] if len(e) > 12 else -1.0 for e in batch],
                        dtype=np.float32)
        # V20 new fields (EXACT names score_mean/score_stdev/root_q/av).
        score_mean = np.array(
            [e[SCORE_MEAN_IDX] if len(e) > SCORE_MEAN_IDX else np.nan
             for e in batch], dtype=np.float32)
        score_stdev = np.array(
            [e[SCORE_STDEV_IDX] if len(e) > SCORE_STDEV_IDX else np.nan
             for e in batch], dtype=np.float32)
        root_q = np.array(
            [e[ROOT_Q_IDX] if len(e) > ROOT_Q_IDX else float(e[2])
             for e in batch], dtype=np.float32)
        _av_rows = []
        for e in batch:
            _a = e[AV_IDX] if len(e) > AV_IDX else None
            if _a is None:
                _av_rows.append(np.full(AV_DIM, np.nan, dtype=np.float32))
                continue
            try:
                _aa = np.asarray(_a)
                if _aa.dtype == np.int8:
                    # Agent C int8-quantized centipawn -> float32 cp.
                    _missing = _aa == -128
                    _aa = (_aa.astype(np.float32) * AV_INT8_SCALE)
                    _aa[_missing] = np.nan
                else:
                    _aa = _aa.astype(np.float32)
                if _aa.shape != (AV_DIM,):
                    _aa = np.asarray(_aa, dtype=np.float32).reshape(-1)
                    if _aa.shape[0] > AV_DIM:
                        _aa = _aa[:AV_DIM]
                    elif _aa.shape[0] < AV_DIM:
                        _pad = np.zeros(AV_DIM - _aa.shape[0],
                                        dtype=np.float32)
                        _aa = np.concatenate([_aa, _pad])
                _av_rows.append(_aa)
            except Exception:
                _av_rows.append(np.full(AV_DIM, np.nan, dtype=np.float32))
        av_logits = np.stack(_av_rows).astype(np.float32)
        return s, pi, z, m, ml, own, margin, mob, safe, reply, pw, \
            ml_mask, prog, score_mean, score_stdev, root_q, av_logits

    def eg_report(self, n: int = 2000) -> dict:
        """What the endgame deque actually holds: histogram of moves-left
        targets (proxy for distance-to-end) + mean |z|. Answers whether
        eg_frac feeds technique or more shuffling draws."""
        if not self.eg:
            return {"n_eg": 0, "n_main": len(self.buf)}
        samp = random.sample(self.eg, min(n, len(self.eg)))
        edges = [0.0, 0.1, 0.25, 0.5, 0.75, 1.01]
        hist = [0] * (len(edges) - 1)
        zabs = 0.0
        for e in samp:
            ml = float(e[4])
            zabs += abs(float(e[2]))
            for i in range(len(hist)):
                if edges[i] <= ml < edges[i + 1]:
                    hist[i] += 1
                    break
        return {"n_eg": len(self.eg), "n_main": len(self.buf),
                "ml_hist": hist, "mean_abs_z": round(zabs / len(samp), 3)}
