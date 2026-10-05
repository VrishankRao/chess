"""SL warm-start (v18): supervised training on Lichess elite games, then STOP.
Trains policy (played moves) + WDL (results) + ALL aux heads (rules-derived
from boards: material/ownership/margin/mobility/safety/moves-left/reply) on
a fresh V18 arch (6x64 SE, 30 planes). 3-4 epochs max (overfit cliff is
documented) — then pure self-play takes over with MCTS distillation.
Scaffolding, removed after use (ChessCoach lesson: sequential, not blended).

Usage (DO NOT RUN without user order — training gate):
  PYTHONPATH=. python3 -m chess_zero.warmstart --games data_sl/games.jsonl \\
      --epochs 3 --batch 256 --out checkpoints_warm/warmstart.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch


ENCODING_PLANES = None

def valid_result_record(record):
    """Accept declared results; never label aborted/unknown games draws."""
    status = record.get("status")
    if status is not None and status not in ("mate", "resign", "stalemate", "timeout", "outoftime", "draw", "variantEnd"):
        return False
    winner = record.get("winner")
    if winner not in (None, "white", "black"):
        return False
    return winner is not None or status in ("draw", "stalemate") or record.get("result") == "1/2-1/2"


def _encode_game(moves_tokens, winner_str, move_cap: int = 300):
    """v19 shared helper (no duplication): replay one game into 11-tuples
    + check targets + P (plies-until-progress) + ml_mask. moves_tokens =
    list of SAN/UCI strings as in the file; winner_str = "white"/"black"/
    None. Returns (positions_11, checks, P_list, mask_list).
    P lookahead: plies until next pawn move / capture / mate (censored =
    remaining plies); mask all-ones here (SL games complete). Caller stamps
    INPUT_PLANES (V18 30) before calling. Reuses the exact load_positions
    teaching (one-hot pi, WDL z, rules-derived aux, absolute reply chain).
    Check = stm in check of the POSITION (board.is_check()).
    """
    import chess
    from chess_zero.game import State, mirror_square, encode_action, \
        material_stm, unit_count_stm
    from chess_zero.selfplay import ownership_map, final_margin, \
        king_safety
    from chess_zero.config import V18_CONFIG as _cfg
    _aux = unit_count_stm if _cfg.aux_unit_counts else material_stm
    if winner_str not in ("white", "black", None):
        raise ValueError("Unknown game result")
    w_result = 1.0 if winner_str == "white" else (
        -1.0 if winner_str == "black" else 0.0)
    board = chess.Board()
    states = []
    history = []
    acts = []  # absolute (fr, to) per ply
    for mv in moves_tokens:
        if not mv:
            continue
        try:
            m = board.parse_uci(mv)
        except Exception:
            m = board.parse_san(mv)
        acts.append((m.from_square, m.to_square, m.promotion))
        st = State(board.copy(stack=False), _hist=history)
        states.append(st)
        history.append(st.rep_key())
        board.push(m)
    final = board
    final_ply = len(acts)
    positions = []
    checks = []
    prog = []
    masks = []
    for i, st in enumerate(states):
        stm = 0 if st.board.turn == chess.WHITE else 1
        fr, to, promotion = acts[i]
        if stm == 1:
            fr, to = mirror_square(fr), mirror_square(to)
        a = encode_action(fr, to, promotion)
        pi = np.zeros(4096, dtype=np.float32)
        pi[a] = 1.0
        z = w_result if stm == 0 else -w_result
        ml = min(1.0, max(0.0, (final_ply - i) / 300.0))
        if i + 1 < len(acts):
            reply = int(acts[i + 1][0])  # absolute from-square
        else:
            reply = -100
        positions.append((
            st.encode(), pi, float(z),
            float(_aux(st.board)), float(ml),
            ownership_map(final, stm),
            float(final_margin(final, stm)),
            min(1.0, len(st.legal_moves()) / 50.0),
            float(king_safety(final, stm)),
            int(reply), 1.0))
        try:
            checks.append(1.0 if st.board.is_check() else 0.0)
        except Exception:
            checks.append(0.0)
        masks.append(1.0 if final.is_checkmate() or final.is_stalemate() or final.is_insufficient_material() else 0.0)
    # One reverse pass rather than a forward scan from every position.
    next_progress = final_ply
    prog = [0.0] * len(acts)
    for i in range(len(acts)-1, -1, -1):
        move = chess.Move(*acts[i][:2], promotion=acts[i][2])
        before = states[i].board
        piece = before.piece_at(move.from_square)
        if (piece and piece.piece_type == chess.PAWN) or before.is_capture(move):
            next_progress = i+1
        prog[i] = float(next_progress-i)
    return positions, checks, prog, masks


def load_positions(path: str, max_positions: int, move_cap: int = 300):
    """Replay games into 11-tuples (same contract as shape_targets)."""
    import chess_zero.game as _g
    from chess_zero.config import V18_CONFIG as _cfg
    _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
    import json as _j
    positions = []
    ngames = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = _j.loads(line)
                if not valid_result_record(g):
                    continue
            except Exception:
                continue
            moves = (g.get("moves", "") or "").split(" ")
            if len(moves) < 10:
                continue
            try:
                _pos, _chk, _prg, _msk = _encode_game(
                    moves, g.get("winner"), move_cap)
            except Exception:
                continue
            if not _pos:
                continue
            ngames += 1
            for _p in _pos:
                positions.append(_p)
                if len(positions) >= max_positions:
                    return positions, ngames
    return positions, ngames


def load_positions_with_checks(path: str, max_positions: int,
                               move_cap: int = 300):
    """v19: like load_positions but also returns check targets aligned
    with positions (stm in check, 1/0). Returns (positions_11, checks, ng).
    load_positions wraps the same helper (no duplication) for compat."""
    import chess_zero.game as _g
    from chess_zero.config import V18_CONFIG as _cfg
    _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
    import json as _j
    positions = []
    checks_all: list = []
    ngames = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = _j.loads(line)
                if not valid_result_record(g):
                    continue
            except Exception:
                continue
            moves = (g.get("moves", "") or "").split(" ")
            if len(moves) < 10:
                continue
            try:
                _pos, _chk, _prg, _msk = _encode_game(
                    moves, g.get("winner"), move_cap)
            except Exception:
                continue
            if not _pos:
                continue
            ngames += 1
            for _p, _c in zip(_pos, _chk):
                positions.append(_p)
                checks_all.append(_c)
                if len(positions) >= max_positions:
                    return positions, checks_all, ngames
    return positions, checks_all, ngames


def load_rows(path: str, max_positions: int, move_cap: int = 300):
    """v19 full loader: (rows13, checks, ngames). rows13 = 11-tuple +
    P progress + ml_mask(1.0, SL games complete). main() trains every
    head including progress from these."""
    import chess_zero.game as _g
    from chess_zero.config import V18_CONFIG as _cfg
    _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
    import json as _j
    rows = []
    checks_all: list = []
    ngames = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = _j.loads(line)
                if not valid_result_record(g):
                    continue
            except Exception:
                continue
            moves = (g.get("moves", "") or "").split(" ")
            if len(moves) < 10:
                continue
            try:
                _pos, _chk, _prg, _msk = _encode_game(
                    moves, g.get("winner"), move_cap)
            except Exception:
                continue
            if not _pos:
                continue
            ngames += 1
            for _p, _c, _g2, _m in zip(_pos, _chk, _prg, _msk):
                rows.append(_p + (float(_g2), float(_m)))
                checks_all.append(_c)
                if len(rows) >= max_positions:
                    return rows, checks_all, ngames
    return rows, checks_all, ngames


def sample_rehearsal_with_ids(games_path: str, n_rows: int,
                                seed: int = 0):
    """V20.2 R1: like sample_rehearsal but threads game_id.

    Returns list of (moves_uci_list, winner, game_id). game_id = row "id"
    when present else "line{lineno}" (stable across runs, SAME key as
    build_av.iter_positions, so the loop's AV-table join hits). New helper;
    sample_rehearsal delegates to this (share code, yesterday bit-identical
    after stripping ids). Deterministic under seed."""
    import json as _j
    records = []
    try:
        with open(games_path) as f:
            for _lineno, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    g = _j.loads(line)
                    if not valid_result_record(g):
                        continue
                except Exception:
                    continue
                moves = (g.get("moves", "") or "").split(" ")
                moves = [m for m in moves if m]
                if len(moves) < 10:
                    continue
                _gid = str(g.get("id") or f"line{_lineno}")
                records.append((list(moves), g.get("winner"), _gid))
    except Exception:
        return []
    _rng = random.Random(int(seed))
    _rng.shuffle(records)
    try:
        _k = int(n_rows)
    except Exception:
        _k = len(records)
    if _k < 0:
        _k = 0
    return records[:_k]


def sample_rehearsal(games_path: str, n_rows: int, seed: int = 0):
    """v19 KL-anchor sampler: lightweight game records only (encoding done
    on the fly in loop via encode_rehearsal_batch). Returns list of
    (moves_uci_list, winner). Deterministic under seed. n_rows = max game
    records to return (positions approximate n_rows * avg_len; the loop
    encodes then slices to its rehearsal_frac). V20.2 R1: delegates to
    sample_rehearsal_with_ids (share code) and strips ids; yesterday
    bit-identical."""
    try:
        _recs = sample_rehearsal_with_ids(games_path, n_rows, seed)
        return [(m, w) for m, w, _ in _recs]
    except Exception:
        # Fallback: never raises on missing file (legacy contract).
        return []


def encode_rehearsal_batch_with_ids(records, move_cap: int = 300):
    """V20.2 R1: encode records into 13-tuples + parallel ids.

    Returns (rows, ids) where rows are 13-tuples IDENTICAL to
    encode_rehearsal_batch and ids is a parallel list of (game_id, ply)
    with ply = 0-based position index within the game (SAME ply as
    build_av.iter_positions, so the loop's AV-table join hits). Records
    may be 2-tuples (moves, winner) -> game_id None (miss -> zeros,
    yesterday) or 3-tuples (moves, winner, game_id). New helper;
    encode_rehearsal_batch delegates (share code, yesterday bit-identical
    rows)."""
    import chess_zero.game as _g
    from chess_zero.config import V18_CONFIG as _cfg
    _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
    out = []
    ids: list = []
    for _rec in (records or []):
        try:
            if isinstance(_rec, (list, tuple)) and len(_rec) >= 2:
                _moves, _win = _rec[0], _rec[1]
                _gid = _rec[2] if len(_rec) >= 3 else None
            else:
                continue
            if isinstance(_moves, str):
                _moves = _moves.split(" ")
            _pos, _chk, _prg, _msk = _encode_game(
                list(_moves), _win, move_cap)
            for _i, (_p, _gg, _m) in enumerate(zip(_pos, _prg, _msk)):
                out.append(_p + (float(_gg), float(_m)))
                try:
                    _gid_s = str(_gid) if _gid is not None else None
                except Exception:
                    _gid_s = None
                ids.append((_gid_s, int(_i)))
        except Exception:
            continue
    return out, ids


def encode_rehearsal_batch(records, move_cap: int = 300):
    """v19: encode sample_rehearsal records into 13-tuples (11 + P
    progress + ml_mask) reusing the load_positions internals (_encode_game,
    no duplication). Stamps INPUT_PLANES (V18 30) like load_positions.
    Loop tails read [11]/[12] for progress/mask targets. V20.2 R1:
    delegates to encode_rehearsal_batch_with_ids (share code) and returns
    rows only; yesterday bit-identical."""
    try:
        _rows, _ = encode_rehearsal_batch_with_ids(records, move_cap)
        return _rows
    except Exception:
        # Fallback: legacy inline path (never raises).
        import chess_zero.game as _g
        from chess_zero.config import V18_CONFIG as _cfg
        _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
        out = []
        for _rec in (records or []):
            try:
                if isinstance(_rec, (list, tuple)) and len(_rec) >= 2:
                    _moves, _win = _rec[0], _rec[1]
                else:
                    continue
                if isinstance(_moves, str):
                    _moves = _moves.split(" ")
                _pos, _chk, _prg, _msk = _encode_game(
                    list(_moves), _win, move_cap)
                for _p, _g, _m in zip(_pos, _prg, _msk):
                    out.append(_p + (float(_g), float(_m)))
            except Exception:
                continue
        return out


# ---------------------------------------------------------------- V20 (D1/D2/D3)
# Tuple contract (shared with Agent A replay.py/train.py/model.py and Agent B
# loop.py — EXACT names; F9: any change updates ALL sites + tests):
#   replay 14-tuple: 0 enc, 1 pi, 2 z, 3 m, 4 ml, 5 own, 6 margin, 7 mob,
#     8 safe, 9 reply, 10 pw, 11 kl, 12 P, 13 ml_mask.
#   V20 appends D1 (+2: score_mean, score_stdev) -> 16, D2 (+1: root_q) -> 17,
#   D3 (+1: av handle) -> 18. Old indices NEVER move; KL stays LAST in the
#   train_step RETURN (not the tuple). Legacy 11/13/14-tuples read neutral
#   defaults (kl 0.0, P censored, mask 1.0, score 0.0/2.0, root_q 0.0/missing,
#   av None) so mixed buffers never crash (repo doctrine).
V20_TUPLE_LEN = 18
I_KL = 11
I_P = 12
I_ML_MASK = 13
I_SCORE_MEAN = 14
I_SCORE_STDEV = 15
I_ROOT_Q = 16
I_AV = 17
# EXACT train-target names (batch-C/A agreement): score_mean, score_stdev,
# root_q (logged search Q; 0.0 = missing in SL), av_logits (decoded 4096
# pawn-vector; tuple stores the int8 HANDLE under I_AV), td_blend (D2 WDL
# target 0.5*z + 0.5*(td_lambda*z + (1-td_lambda)*root_q)).
V20_TARGET_KEYS = ("score_mean", "score_stdev", "root_q", "av_logits",
                   "td_blend")
SCORE_STDEV_PRIOR = 2.0  # SL single outcomes carry no variance info; see below
ROOT_Q_MISSING = 0.0  # sentinel: SL warm-start logs no search Q (measure-zero
# collision with genuine 0.0 search Qs; those degenerate toward z, which is
# within D2's final-z-dominant doctrine)


def score_targets_for_game(final_board):
    """D1 score-margin targets: (score_mean, score_stdev).

    score_mean = white-relative final material in pawns (spec: white-relative;
    reuse final_margin(final, 0) which returns pawns/10, x10 -> pawns).
    score_stdev = SCORE_STDEV_PRIOR constant: one SL game outcome carries NO
    variance information, so a constant prior keeps the head near typical
    middlegame uncertainty instead of hallucinating confidence. Training
    gradient only in V20 (never search/veto); the D1 linear-probe gate
    (R2 > 0.3 on a buffer sample, else score_w stays 0) governs unfreezing.
    """
    from chess_zero.selfplay import final_margin as _fm
    try:
        _mean = float(_fm(final_board, 0)) * 10.0
    except Exception:
        _mean = 0.0
    if _mean != _mean:  # NaN guard
        _mean = 0.0
    return _mean, float(SCORE_STDEV_PRIOR)


def td_blend_target(z, root_q=0.0, td_lambda=0.5, has_q=True):
    """D2 short-horizon TD value target: 0.5*z + 0.5*td with
    td = td_lambda*z + (1-td_lambda)*root_q (td_lambda default 0.5 ->
    effective WDL target 0.75*z + 0.25*root_q: final-z dominant per D2).
    has_q False (SL warm-start: no search Q logged) -> degenerates to z
    (pure-results WDL, exactly yesterday)."""
    try:
        _z = float(z)
    except Exception:
        _z = 0.0
    if not has_q:
        return _z
    try:
        _l = float(td_lambda)
    except Exception:
        _l = 0.5
    _l = max(0.0, min(1.0, _l))
    try:
        _q = float(root_q)
    except Exception:
        return _z
    if _q != _q:  # NaN search Q -> fall back to z
        return _z
    _td = _l * _z + (1.0 - _l) * _q
    return 0.5 * _z + 0.5 * _td


def score_w_for_step(step, ramp_steps=5000, w_max=0.05):
    """D1 loss-weight ramp: score_w 0 -> w_max over the first ramp_steps
    training steps (V20_CONFIG: score_w_max 0.05, score_ramp_steps 5000).
    Linear; step <= 0 reads 0.0 (probe gate holds the backbone frozen)."""
    try:
        _s = float(step)
    except Exception:
        return 0.0
    try:
        _r = max(1.0, float(ramp_steps))
        _w = float(w_max)
    except Exception:
        return 0.0
    if _s <= 0.0:
        return 0.0
    return _w * min(1.0, _s / _r)


def linear_probe_r2(features, targets) -> float:
    """D1 probe gate: least-squares linear probe R^2 of features -> targets
    (bias included). Gate: R2 > 0.3 on a buffer sample before unfreezing the
    backbone for the score head, else score_w stays 0. Pure function (no
    model); < 3 rows or zero target variance reads 0.0."""
    import torch as _t
    try:
        _X = _t.as_tensor(np.asarray(features), dtype=_t.float32)
        _y = _t.as_tensor(np.asarray(targets), dtype=_t.float32).reshape(-1)
    except Exception:
        return 0.0
    if _X.dim() == 1:
        _X = _X.unsqueeze(1)
    _n = _X.shape[0]
    if _n < 3 or _n != _y.shape[0]:
        return 0.0
    try:
        _X1 = _t.cat([_X, _t.ones(_n, 1)], dim=1)
        _pred = _X1 @ _t.linalg.lstsq(_X1, _y).solution
        _ss_res = float(((_y - _pred) ** 2).sum())
        _ss_tot = float(((_y - _y.mean()) ** 2).sum())
    except Exception:
        return 0.0
    if not np.isfinite(_ss_res) or _ss_tot <= 0.0:
        return 0.0
    return max(0.0, 1.0 - _ss_res / _ss_tot)


def load_av_table(av_path: str) -> dict:
    """Load build_av.py output into {(game_id, ply): {"stm", "av": {uci: q8}}}.
    Missing/unreadable path -> {} (rows read av None = aux skipped,
    graph-connected zero by the reply-head precedent)."""
    _tab: dict = {}
    if not av_path:
        return _tab
    try:
        with open(av_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _r = json.loads(_line)
                    _tab[(str(_r.get("id")), int(_r.get("ply")))] = {
                        "stm": _r.get("stm", "w"), "av": _r.get("av") or {}}
                except Exception:
                    continue
    except Exception:
        pass
    return _tab


def av_handle_from_row(av_uci_dict, stm_str: str = "w"):
    """build_av {uci: q8} -> tuple handle {stm-oriented action int: q8}.

    The 4096 codec is stm-oriented (Black mirrors); mapping needs the
    position side, stored per build_av row. None/empty -> None (missing).
    """
    if not av_uci_dict:
        return None
    import chess as _ch
    from chess_zero.game import mirror_square as _mir, \
        encode_action as _enc
    _stm = 1 if str(stm_str) == "b" else 0
    _h: dict = {}
    try:
        _items = av_uci_dict.items()
    except Exception:
        return None
    for _u, _q in _items:
        try:
            _m = _ch.Move.from_uci(str(_u))
            _fr, _to = _m.from_square, _m.to_square
            if _stm == 1:
                _fr, _to = _mir(_fr), _mir(_to)
            _h[int(_enc(_fr, _to, _m.promotion))] = int(_q)
        except Exception:
            continue
    return _h or None


def decode_av_logits(handle):
    """Tuple handle -> (vec4096 float32 pawns, mask4096 bool).

    Unit contract (D3, shared with Agent A): av_logits in PAWNS (cp/100;
    dequantize q8 -> cp first), stm-relative, over LEGAL moves only (mask
    marks them). Distillation T=2.0 is in pawns on both sides. Missing/empty
    handle -> (zeros, all-False) so the loss can skip the row graph-connected.
    """
    import numpy as _np
    from chess_zero.build_av import dequantize_q8 as _dq
    _v = _np.full(4096, _np.nan, dtype=_np.float32)
    _m = _np.zeros(4096, dtype=bool)
    if not handle:
        return _v, _m
    try:
        _items = handle.items()
    except Exception:
        return _v, _m
    for _a, _q in _items:
        try:
            _ai = int(_a)
            if 0 <= _ai < 4096:
                _v[_ai] = float(_dq(_q))
                _m[_ai] = True
        except Exception:
            continue
    return _v, _m


def _v20_extend(moves_tokens, winner_str, move_cap: int = 300,
                game_id=None, av_table=None):
    """Shared V20 row builder: _encode_game (11 + checks + P + mask) then
    kl=0.0 (SL logs no search KL) at 11, P at 12, mask at 13 (replay-aligned),
    score_mean/score_stdev (D1, final board) at 14/15, root_q=0.0 = missing
    (D2, SL has no search Q) at 16, av handle (D3, None when no label) at 17.
    stm parity (white moves even plies) maps av rows without re-parsing."""
    import chess as _ch
    _pos11, _checks, _prog, _masks = _encode_game(
        moves_tokens, winner_str, move_cap)
    _bd = _ch.Board()
    for _mv in moves_tokens:
        if not _mv:
            continue
        try:
            _m = _bd.parse_uci(_mv)
        except Exception:
            _m = _bd.parse_san(_mv)
        _bd.push(_m)
    _smean, _sstdev = score_targets_for_game(_bd)
    _rows = []
    for _i, _p in enumerate(_pos11):
        _stm = "b" if (_i % 2 == 1) else "w"
        _av = None
        if av_table is not None and game_id is not None:
            _raw = av_table.get((str(game_id), int(_i)))
            if _raw is not None:
                _av = av_handle_from_row(_raw.get("av"),
                                         _raw.get("stm", _stm))
        _rows.append(tuple(_p) + (0.0, float(_prog[_i]), float(_masks[_i]),
                                  float(_smean) * (1.0 if _stm == "w" else -1.0), float(_sstdev),
                                  float(ROOT_Q_MISSING), _av))
    return _rows, _checks


def load_rows_v20(path: str, max_positions: int, move_cap: int = 300,
                  av_table=None):
    """V20 full loader: (rows18, checks, ngames). rows18 = 11-tuple + kl(0.0)
    + P + ml_mask + score_mean + score_stdev + root_q(0.0) + av handle.
    av_table (load_av_table) joins build_av labels by (game id, ply); rows
    without "id" or without labels read av None. Legacy load_rows* above are
    UNTOUCHED (yesterday bit-exact)."""
    import chess_zero.game as _g
    from chess_zero.config import V18_CONFIG as _cfg
    _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
    import json as _j
    rows = []
    checks_all: list = []
    ngames = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = _j.loads(line)
                if not valid_result_record(g):
                    continue
            except Exception:
                continue
            moves = (g.get("moves", "") or "").split(" ")
            if len(moves) < 10:
                continue
            try:
                _r18, _chk = _v20_extend(moves, g.get("winner"), move_cap,
                                         g.get("id"), av_table)
            except Exception:
                continue
            if not _r18:
                continue
            ngames += 1
            for _p, _c in zip(_r18, _chk):
                rows.append(_p)
                checks_all.append(_c)
                if len(rows) >= max_positions:
                    return rows, checks_all, ngames
    return rows, checks_all, ngames


def encode_rehearsal_batch_v20(records, move_cap: int = 300, av_table=None):
    """V20 rehearsal encoder: sample_rehearsal records -> rows18 (same layout
    as load_rows_v20; 2-tuple records carry no game ids so av reads None,
    kl/root_q read 0.0). V20.2 R1: 3-tuple records (moves, winner, game_id)
    join av_table by (game_id, ply) via _v20_extend (2-tuples unchanged =
    yesterday). Loop tails read [14]/[15]/[16]/[17] for the new targets."""
    import chess_zero.game as _g
    from chess_zero.config import V18_CONFIG as _cfg
    _g.INPUT_PLANES = ENCODING_PLANES or _cfg.input_planes
    out = []
    for _rec in (records or []):
        try:
            if isinstance(_rec, (list, tuple)) and len(_rec) >= 2:
                _moves, _win = _rec[0], _rec[1]
                _gid = _rec[2] if len(_rec) >= 3 else None
            else:
                continue
            if isinstance(_moves, str):
                _moves = _moves.split(" ")
            _r18, _chk = _v20_extend(list(_moves), _win, move_cap, _gid,
                                     av_table)
            out.extend(_r18)
        except Exception:
            continue
    return out


def encode_v20_batch(rows, checks=None, td_lambda: float = 0.5):
    """Rows (18, tolerant 11/13/14 legacy) -> dict of torch tensors keyed by
    legacy names + V20_TARGET_KEYS. The dry-run/acceptance entry point:
    no model needed, pure encoding. Legacy lengths: 13 = old warmstart
    (11 + P[11] + mask[12], kl reads 0.0); 14 = replay (kl[11], P[12],
    mask[13]); < 18 reads score 0.0/2.0, root_q missing, av None.
    td_blend computed per-row (D2; missing root_q -> z). av_logits stacked
    4096 pawns + av_mask bool + av_present float."""
    import torch as _t
    _n = len(rows)
    _s = np.stack([np.asarray(e[0], dtype=np.float32) for e in rows])
    _pi = np.stack([np.asarray(e[1], dtype=np.float32) for e in rows])
    _z = np.array([float(e[2]) for e in rows], dtype=np.float32)
    _m = np.array([float(e[3]) for e in rows], dtype=np.float32)
    _ml = np.array([float(e[4]) for e in rows], dtype=np.float32)
    _own = np.stack([np.asarray(e[5], dtype=np.float32) for e in rows])
    _margin = np.array([float(e[6]) for e in rows], dtype=np.float32)
    _mob = np.array([float(e[7]) for e in rows], dtype=np.float32)
    _safe = np.array([float(e[8]) if len(e) > 8 else 0.5 for e in rows],
                     dtype=np.float32)
    _reply = np.array([int(e[9]) if len(e) > 9 else -100 for e in rows],
                      dtype=np.int64)
    _pw = np.array([float(e[10]) if len(e) > 10 else 1.0 for e in rows],
                   dtype=np.float32)
    _kl, _prog, _mmask = [], [], []
    _sm, _ss, _rq, _td = [], [], [], []
    _avl, _avm, _avp = [], [], []
    for e in rows:
        _L = len(e)
        if _L == 13:  # legacy warmstart rows: P[11], mask[12], no kl
            _kl.append(0.0)
            _prog.append(float(e[11]))
            _mmask.append(float(e[12]))
        else:  # replay 14 / v20 18 (shorter reads neutral defaults)
            _kl.append(float(e[11]) if _L > 11 else 0.0)
            _prog.append(float(e[12]) if _L > 12 else 0.0)
            _mmask.append(float(e[13]) if _L > 13 else 1.0)
        _sm.append(float(e[14]) if _L > 14 else 0.0)
        _ss.append(float(e[15]) if _L > 15 else float(SCORE_STDEV_PRIOR))
        _rqv = float(e[16]) if _L > 16 else float(ROOT_Q_MISSING)
        _rq.append(_rqv)
        _td.append(td_blend_target(float(e[2]), _rqv, td_lambda,
                                   has_q=(_rqv != float(ROOT_Q_MISSING))))
        _lv, _lm = decode_av_logits(e[17] if _L > 17 else None)
        _avl.append(_lv)
        _avm.append(_lm)
        _avp.append(1.0 if bool(_lm.any()) else 0.0)
    _b = {
        "boards": _t.from_numpy(_s), "pi": _t.from_numpy(_pi),
        "z": _t.tensor(_z), "m": _t.tensor(_m), "ml": _t.tensor(_ml),
        "own": _t.from_numpy(_own), "margin": _t.tensor(_margin),
        "mob": _t.tensor(_mob), "safe": _t.tensor(_safe),
        "reply": _t.from_numpy(_reply), "pw": _t.tensor(_pw),
        "kl": _t.tensor(np.array(_kl, dtype=np.float32)),
        "progress": _t.tensor(np.array(_prog, dtype=np.float32)),
        "ml_mask": _t.tensor(np.array(_mmask, dtype=np.float32)),
        "score_mean": _t.tensor(np.array(_sm, dtype=np.float32)),
        "score_stdev": _t.tensor(np.array(_ss, dtype=np.float32)),
        "root_q": _t.tensor(np.array(_rq, dtype=np.float32)),
        "td_blend": _t.tensor(np.array(_td, dtype=np.float32)),
        "av_logits": _t.from_numpy(np.stack(_avl)),
        "av_mask": _t.from_numpy(np.stack(_avm)),
        "av_present": _t.tensor(np.array(_avp, dtype=np.float32)),
    }
    if checks is not None:
        _b["checks"] = _t.tensor(np.asarray(checks[:_n], dtype=np.float32))
    return _b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data_sl/games.jsonl")
    ap.add_argument("--planes", type=int, default=None, help="Explicit input-plane count; audited baseline uses 31")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="checkpoints_warm/warmstart.pt")
    ap.add_argument("--max-positions", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--v20", action="store_true",
                    help="V20 rows18 (score/root_q/av + td_blend WDL); "
                         "new train_step kwargs passed only when supported "
                         "(signature-guarded for Agent-A landing order)")
    ap.add_argument("--av-table", default="",
                    help="build_av.py output jsonl for D3 SL distillation "
                         "(missing rows read av None = aux skipped)")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="encode N rows and print batch shapes, then exit "
                         "(no training, no writes; acceptance probe)")
    a = ap.parse_args()
    global ENCODING_PLANES
    ENCODING_PLANES = a.planes

    from chess_zero.config import V18_CONFIG as cfg
    from chess_zero.model import AlphaZeroNet
    from chess_zero.train import train_step
    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"warm-start: loading {a.games}", flush=True)
    _avt = load_av_table(a.av_table) if (a.v20 and a.av_table) else None
    if a.v20:
        pos, chk, ng = load_rows_v20(a.games, a.max_positions,
                                     av_table=_avt)
    else:
        pos, chk, ng = load_rows(a.games, a.max_positions)
    print(f"warm-start: {len(pos)} positions from {ng} games", flush=True)
    if not pos:
        raise SystemExit("no positions — download the dataset first")
    if a.dry_run:
        _k = max(1, min(int(a.dry_run), len(pos)))
        _tdl = float(getattr(cfg, "td_lambda", 0.5))
        _b = encode_v20_batch(pos[:_k], chk[:_k], td_lambda=_tdl)
        for _kk, _vv in _b.items():
            print(f"dry-run {_kk}: {tuple(_vv.shape)} {_vv.dtype}",
                  flush=True)
        import torch as _td
        print(f"dry-run score_mean range "
              f"[{float(_b['score_mean'].min()):.2f}, "
              f"{float(_b['score_mean'].max()):.2f}] stdev "
              f"{float(_b['score_stdev'].mean()):.2f} "
              f"td==z {bool(_td.equal(_b['td_blend'], _b['z']))} "
              f"av_present {float(_b['av_present'].sum()):.0f}/{_k}",
              flush=True)
        return
    if a.planes is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, input_planes=a.planes)
    net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels,
                       planes=cfg.input_planes,
                       se_ratio=cfg.se_ratio).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr,
                           weight_decay=cfg.l2)
    # v19: new-head weights from cfg with yesterday-safe getattr defaults
    # (old configs keep training; V19 sets 0.3/0.01). Check targets are
    # stm-in-check of the POSITION; soft uses label-smoothed one-hot
    # (smooth 0.05) via smooth_eps so both policy heads see pi_smooth.
    _soft_w = float(getattr(cfg, "soft_w", 0.3))
    _check_w = float(getattr(cfg, "check_w", 0.01))
    _prog_w = float(getattr(cfg, "progress_w", 0.05))
    paired = list(zip(pos, chk))
    n = len(paired)
    # v20: new-kw signature guard (Agent-A landing order: pass score/av/TD
    # targets only when this checkout's train_step supports them; else the
    # rows still encode (acceptance) and WDL trains on the td blend, which
    # degenerates to z without search Q — yesterday exactly).
    import inspect as _insp
    _ts_params = set(_insp.signature(train_step).parameters)
    _tdl = float(getattr(cfg, "td_lambda", 0.5))
    _av_w = float(getattr(cfg, "av_w", 0.1))
    _ramp = int(getattr(cfg, "score_ramp_steps", 5000))
    _swmax = float(getattr(cfg, "score_w_max", 0.05))
    _v20_live = sorted({"score_mean", "score_stdev", "root_q", "av_logits",
                        "td_blend", "score_w", "av_w"} & _ts_params)
    if a.v20:
        print(f"warm-start v20: rows18 td_lambda={_tdl} av_w={_av_w} "
              f"score ramp 0->{_swmax}/{_ramp} steps; train_step takes "
              f"{_v20_live or 'none (held at 0)'}", flush=True)
    _gstep = 0
    for ep in range(a.epochs):
        random.shuffle(paired)
        tot = 0.0
        steps = 0
        for i in range(0, n, a.batch):
            b = paired[i:i + a.batch]
            if len(b) < 8:
                continue
            if a.v20:
                _bpos = [e[0] for e in b]
                _bchk = [e[1] for e in b]
                _eb = encode_v20_batch(_bpos, _bchk, td_lambda=_tdl)
                _kw = {}
                if "score_mean" in _ts_params:
                    _kw["score_mean"] = _eb["score_mean"]
                if "score_stdev" in _ts_params:
                    _kw["score_stdev"] = None
                if "root_q" in _ts_params:
                    _kw["root_q"] = _eb["root_q"]
                if "av_logits" in _ts_params:
                    _kw["av_logits"] = _eb["av_logits"]
                    _kw["av_mask"] = _eb["av_mask"]
                if "td_blend" in _ts_params:
                    _kw["td_blend"] = _eb["td_blend"]
                if "score_w" in _ts_params:
                    _kw["score_w"] = score_w_for_step(
                        _gstep, _ramp, _swmax)
                if "av_w" in _ts_params:
                    _kw["av_w"] = _av_w
                # D2: WDL head trains on the blend (== z while SL root_q
                # missing, so pre-Agent-A checkouts train yesterday).
                t, *_ = train_step(
                    net, opt, _eb["boards"], _eb["pi"], _eb["td_blend"],
                    _eb["m"], _eb["ml"], _eb["own"], _eb["margin"],
                    _eb["mob"], _eb["safe"], _eb["reply"], _eb["pw"],
                    reply_w=cfg.reply_w, soft_w=_soft_w,
                    check_w=_check_w, smooth_eps=0.05,
                    checks=_eb.get("checks"), progress=_eb["progress"],
                    ml_mask=_eb["ml_mask"], progress_w=_prog_w, **_kw)
                _gstep += 1
                tot += t
                steps += 1
                continue
            _bpos = [e[0] for e in b]
            _bchk = [e[1] for e in b]
            s = torch.from_numpy(np.stack([e[0] for e in _bpos]))
            pi = torch.from_numpy(np.stack([e[1] for e in _bpos]))
            z = torch.tensor([e[2] for e in _bpos], dtype=torch.float32)
            m = torch.tensor([e[3] for e in _bpos], dtype=torch.float32)
            ml = torch.tensor([e[4] for e in _bpos], dtype=torch.float32)
            own = torch.stack([torch.from_numpy(e[5]) for e in _bpos])
            margin = torch.tensor([e[6] for e in _bpos], dtype=torch.float32)
            mob = torch.tensor([e[7] for e in _bpos], dtype=torch.float32)
            safe = torch.tensor([e[8] for e in _bpos], dtype=torch.float32)
            reply = torch.tensor([e[9] for e in _bpos], dtype=torch.long)
            pw = torch.tensor([e[10] for e in _bpos], dtype=torch.float32)
            checks = torch.tensor(_bchk, dtype=torch.float32)
            # v19: rows13 carry P (index 11) + ml_mask (index 12); SL
            # games are complete so mask reads 1.0.
            prog = torch.tensor([e[11] if len(e) > 11 else 0.0
                                 for e in _bpos], dtype=torch.float32)
            mmask = torch.tensor([e[12] if len(e) > 12 else 1.0
                                  for e in _bpos], dtype=torch.float32)
            t, *_ = train_step(net, opt, s, pi, z, m, ml, own, margin,
                               mob, safe, reply, pw, reply_w=cfg.reply_w,
                               soft_w=_soft_w, check_w=_check_w,
                               smooth_eps=0.05, checks=checks,
                               progress=prog, ml_mask=mmask,
                               progress_w=_prog_w)
            tot += t
            steps += 1
        print(f"warm-start epoch {ep + 1}/{a.epochs}: loss "
              f"{tot / max(steps, 1):.4f} ({steps} steps)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from .loop import atomic_save
    atomic_save({"weights": {k: v.cpu() for k, v in
                            net.state_dict().items()},
                "meta": {"warm": True, "games": ng,
                         "positions": n, "epochs": a.epochs}}, a.out)
    print(f"warm-start: saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
