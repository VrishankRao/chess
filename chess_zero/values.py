"""Empirical piece values: the model updates what pieces are worth from GAME
RESULTS, not human tables. Every few iters, ridge-regress game outcomes on
stm-relative holdings reconstructed straight from encoding planes
(planes 0-4 minus 6-10 in canonical orientation = exact count diffs).
EMA-smoothed, clamped near classical, decisive-game guarded. Used for
TRAINING adjudication only; arena stays classical for comparability.
"""
from __future__ import annotations

import numpy as np

import chess

ORDER_NOKING = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
                chess.QUEEN]
CLASSICAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
             chess.ROOK: 5.0, chess.QUEEN: 9.0}


def holdings_from_enc(enc: np.ndarray) -> np.ndarray:
    """stm-relative count diffs [P,N,B,R,Q] from a (P,8,8) encoding
    (P>=13; uses planes 0-4 minus 6-10, rotation-invariant counts)."""
    return np.array([enc[i].sum() - enc[6 + i].sum() for i in range(5)],
                    dtype=np.float64)


def decisive_samples(sample: list) -> list[tuple[np.ndarray, float, float]]:
    """Keep exact ±1.0 labels only (audit round 2 §2.7). Draw labels are
    contempt-shaped by construction (ahead pays +c for the draw), so
    regressing on them measures our own shaping. Decisive labels
    (mate/adjudication/resign) are unshaped.
    Returns (enc, z, ml) where ml = moves-left target (plies remaining /
    move_cap, 0 = at the terminal). ml drives distance weighting in
    fit_values (§6.4): endgame positions (small ml) are table-clean,
    opening positions (large ml) may see material change before the end.
    Accepts 5-tuples (enc, pi, z, m, ml) from the buffer or legacy
    3-tuples (enc, _, z); missing ml defaults to 0.0 (full weight)."""
    out = []
    for e in sample:
        try:
            z = float(e[2])
        except Exception:
            continue
        if abs(z) < 1.0:
            continue
        ml = 0.0
        try:
            if len(e) >= 5:
                ml = float(e[4])
        except Exception:
            ml = 0.0
        out.append((e[0], z, ml))
    return out


def fit_values(samples: list[tuple[np.ndarray, float, float]] | list[tuple[np.ndarray, float]],
               ridge: float = 100.0) -> dict | None:
    """Least-squares values from (encoding, stm-outcome[, ml]) pairs.
    §6.4: distance-weighted — adjudicated labels are sign(final material)
    scored by the CURRENT table, so ply-t positions far from the end are
    pulled toward the incumbent table (circularity). Weight w = 1/(1+2*ml)
    down-weights far-from-end samples (ml=0 → 1.0, ml=1 → 0.33) while
    keeping every decisive game in the fit. Mates (unshaped,
    table-independent) keep full influence via small ml.
    Returns None when data is too thin/uniform to trust."""
    if len(samples) < 2000:
        return None
    encs = []
    zs = []
    ws = []
    for e in samples:
        if len(e) >= 3:
            enc, z, ml = e[0], float(e[1]), float(e[2])
        else:
            enc, z, ml = e[0], float(e[1]), 0.0
        # clamp ml to [0,1] (buffer guarantees it, tests may not)
        ml = min(max(ml, 0.0), 1.0)
        encs.append(enc)
        zs.append(z)
        ws.append(1.0 / (1.0 + 2.0 * ml))
    X = np.stack([holdings_from_enc(e) for e in encs])
    z = np.array(zs, dtype=np.float64)
    w = np.array(ws, dtype=np.float64)
    # Safety net for direct calls on mixed/unshaped data (all-draws test).
    # On the decisive-only path every z is ±1 so this is always 1.0 and
    # never fires — the real draw filter is decisive_samples above.
    if (np.abs(z) > 0.01).mean() < 0.15:
        return None  # not enough decisive games
    if (X.var(axis=0) < 1e-6).any():
        return None  # a piece type never varies
    A = X.T @ (w[:, None] * X) + ridge * np.eye(5)
    vals = np.linalg.solve(A, X.T @ (w * z))
    # v12 fix: pawn-numeraire anchor kills bounded-target deflation.
    # Unconstrained regression of saturated ±1.0 on unbounded diffs forces
    # all coeffs toward 0 as games get lopsided (iter-130 collapse to the
    # 0.5x floor). Post-hoc P-normalization preserves exchange ratios when
    # P is positive (the common proportional-collapse case).
    # Fallback: when P is non-positive (collinear synthetic / queen-perfect
    # predictor), solve the CONSTRAINED fit with P fixed to 1.0:
    #   (Xr' W Xr + rI) vr = Xr' W (z - xp)
    # so a strong queen signal still returns Q>0 instead of None. If even
    # the constrained queen is non-positive, the data is untrustworthy.
    if vals[0] > 1e-4:
        vals = vals / vals[0]
        return {pt: float(v) for pt, v in zip(ORDER_NOKING, vals)}
    xp = X[:, 0]
    Xr = X[:, 1:]
    try:
        Ac = Xr.T @ (w[:, None] * Xr) + ridge * np.eye(4)
        vr = np.linalg.solve(Ac, Xr.T @ (w * (z - xp)))
    except np.linalg.LinAlgError:
        return None
    if vr[3] <= 0:
        return None  # queen must correlate with winning
    full = np.concatenate([[1.0], vr])
    return {pt: float(v) for pt, v in zip(ORDER_NOKING, full)}


def update_values(current: dict, fitted: dict | None, alpha: float = 0.2,
                  band: tuple[float, float] = (0.5, 2.0)) -> dict:
    """EMA toward fitted values, clamped to band x classical. None = hold."""
    if fitted is None:
        return dict(current)
    out = {}
    for pt in ORDER_NOKING:
        v = (1 - alpha) * current.get(pt, CLASSICAL[pt]) + alpha * fitted[pt]
        lo, hi = band[0] * CLASSICAL[pt], band[1] * CLASSICAL[pt]
        out[pt] = min(max(v, lo), hi)
    return out
