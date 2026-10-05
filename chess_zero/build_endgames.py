"""V20 D4 endgame-start harvest (Agent B): collect <=8-man endgame
positions from pgns/*.pgn (all our games) + 40 hardcoded technical
opens (KQvK, KRvK, KPvK key squares, Lucena, Philidor, QvR, R+BvR).
Output data_sl/endgames.jsonl (one {"fen"} per line; FEN + side to
move; fullmove normalized to 1 so the loop's TB-rescore alignment
(record ply == replay index) holds; halfmove clock randomized 0-60
for 20% of rows for 50-move-pressure diversity).

Loop contract (D4 order fixed): the promotion rule wires main-panel
AND endgame_panel FIRST (loop._endgame_panel_gate reads this file);
only then do endgame_frac starts draw from it (contempt forced 0,
excluded from forced-book — both in selfplay).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

# 40 hardcoded technical opens (kings kept non-adjacent; all pass
# chess.Board.is_valid — verified by the acceptance mini-run).
TECHNICAL_OPENS = [
    # KQvK x5.
    "8/8/4k3/8/8/2KQ4/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/5QK1/8 w - - 0 1",
    "7k/8/8/8/8/8/6Q1/6K1 w - - 0 1",
    "8/8/8/3k4/8/8/2Q5/3K4 w - - 0 1",
    "8/8/8/8/1k6/8/2Q5/1K6 w - - 0 1",
    # KRvK x5.
    "8/8/4k3/8/8/3K2R1/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/5RK1/8 w - - 0 1",
    "7k/8/8/8/8/8/1R6/1K6 w - - 0 1",
    "8/8/8/3k4/8/8/4R3/3K4 w - - 0 1",
    "8/1K6/8/8/7R/1k6/8/8 w - - 0 1",
    # KPvK key squares / opposition / rook-pawn x10.
    "8/8/2k5/4P3/4K3/8/8/8 w - - 0 1",
    "8/8/8/4P1k1/4K3/8/8/8 w - - 0 1",
    "8/8/8/3kP3/8/4K3/8/8 w - - 0 1",
    "8/8/8/2kP4/4K3/8/8/8 w - - 0 1",
    "8/8/8/8/P6k/8/7K/8 w - - 0 1",
    "8/8/8/8/6kP/8/6K1/8 w - - 0 1",
    "8/8/2k5/4P3/4K3/8/8/8 b - - 0 1",
    "8/8/3k4/8/8/4K3/4P3/8 w - - 0 1",
    "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1",
    "8/8/8/8/8/2k5/2P5/2K5 w - - 0 1",
    # R+P versus bare king conversion positions x4 (not Lucena).
    "8/8/8/8/8/1k6/1P6/1KR5 w - - 0 1",
    "8/8/8/8/8/1k6/1P6/1KR5 b - - 0 1",
    "8/8/8/8/8/5k2/5P2/5KR1 w - - 0 1",
    "8/8/8/8/8/5k2/5P2/5KR1 b - - 0 1",
    # R+P versus R positions x4 (theoretical result unlabelled).
    "4k3/8/8/8/4P3/8/4R1K1/r7 w - - 0 1",
    "4k3/8/8/8/4P3/8/4R1K1/r7 b - - 0 1",
    "3k4/8/8/8/3P4/8/3R2K1/r7 w - - 0 1",
    "3k4/8/8/8/3P4/8/3R2K1/r7 b - - 0 1",
    # QvR x6.
    "8/8/8/8/8/1k6/3Q3r/1K6 w - - 0 1",
    "8/8/8/8/8/1k6/3Q3r/1K6 b - - 0 1",
    "7k/8/8/8/8/8/5QK1/5r2 w - - 0 1",
    "7k/8/8/8/8/8/5QK1/5r2 b - - 0 1",
    "8/8/8/3k4/8/8/5QK1/r7 w - - 0 1",
    "8/8/8/3k4/8/8/5QK1/r7 b - - 0 1",
    # R+B versus bare king conversion positions x6.
    "8/8/8/3k4/8/8/3BR3/3K4 w - - 0 1",
    "8/8/8/3k4/8/8/3BR3/3K4 b - - 0 1",
    "7k/8/8/8/8/8/1R3BK1/8 w - - 0 1",
    "7k/8/8/8/8/8/1R3BK1/8 b - - 0 1",
    "8/8/8/8/1k6/8/2R1B3/1K6 w - - 0 1",
    "8/8/8/8/1k6/8/2R1B3/1K6 b - - 0 1",
]

assert len(TECHNICAL_OPENS) == 40, len(TECHNICAL_OPENS)

MAX_MEN = 8


def normalize_fen(fen: str) -> str | None:
    """Normalize to fullmove 1 (loop TB alignment: record ply == replay
    index); keep placement/turn/castling/ep/halfmove. None when the FEN
    is illegal (kings adjacent, pawns on back rank, ...). Never raises."""
    try:
        import chess as _c
        _b = _c.Board(str(fen))
        if not _b.is_valid():
            return None
        _parts = _b.fen().split(" ")
        _parts[5] = "1"
        return " ".join(_parts)
    except Exception:
        return None


def harvest_pgns(pgn_glob: str = "pgns/*.pgn",
                 max_positions: int = 2000) -> tuple:
    """Walk mainlines of pgns/*.pgn; collect non-terminal positions with
    2..MAX_MEN men, deduped, normalized. Returns (fens, stats). Never
    raises (missing/unreadable PGNs -> empty + loud log)."""
    import chess as _c
    import chess.pgn as _pgn
    _seen: dict = {}
    _stats = {"games": 0, "skipped_games": 0, "positions": 0}
    try:
        _files = sorted(glob.glob(pgn_glob))
    except Exception:
        _files = []
    if not _files:
        print(f"[endgames] no PGN files at '{pgn_glob}'; harvest empty",
              flush=True)
        return [], _stats
    for _fp in _files:
        try:
            _fh = open(_fp)
        except Exception:
            _stats["skipped_games"] += 1
            continue
        with _fh:
            while len(_seen) < int(max_positions):
                try:
                    _g = _pgn.read_game(_fh)
                except Exception:
                    _stats["skipped_games"] += 1
                    break
                if _g is None:
                    break
                _stats["games"] += 1
                try:
                    _b = _g.board()
                    for _mv in _g.mainline_moves():
                        _b.push(_mv)
                        if _b.is_game_over():
                            continue
                        if not (2 <= len(_b.piece_map()) <= MAX_MEN):
                            continue
                        _fen = normalize_fen(_b.fen())
                        if _fen and _fen not in _seen:
                            _seen[_fen] = True
                            _stats["positions"] += 1
                            if len(_seen) >= int(max_positions):
                                break
                except Exception:
                    _stats["skipped_games"] += 1
                    continue
    print(f"[endgames] harvested {len(_seen)} <=8-man positions from "
          f"{_stats['games']} games ({_stats['skipped_games']} skipped)",
          flush=True)
    return list(_seen.keys()), _stats


def build_endgames(pgn_glob: str = "pgns/*.pgn", out: str =
                   "data_sl/endgames.jsonl", max_harvest: int = 2000,
                   seed: int = 20) -> dict:
    """Harvest + 40 technical opens -> out (one {"fen"} per line).
    20% of rows get halfmove randomized 0-60 (seeded). Technical opens
    that fail validation are DROPPED loudly (never written). Returns
    summary. Never raises."""
    _rng = random.Random(int(seed))
    _harvest, _hstats = harvest_pgns(pgn_glob, max_harvest)
    _tech, _bad = [], 0
    for _fen in TECHNICAL_OPENS:
        _n = normalize_fen(_fen)
        if _n is None:
            _bad += 1
            print(f"[endgames] technical FEN invalid, dropped: "
                  f"{str(_fen)[:48]}", flush=True)
        else:
            _tech.append(_n)
    _all = list(_harvest) + _tech
    _rand = 0
    _rows = []
    for _fen in _all:
        try:
            if _rng.random() < 0.20:
                _p = _fen.split(" ")
                _p[4] = str(_rng.randint(0, 60))
                _fen = " ".join(_p)
                _rand += 1
            _rows.append({"fen": _fen})
        except Exception:
            _rows.append({"fen": _fen})
    try:
        _d = os.path.dirname(out) or "."
        os.makedirs(_d, exist_ok=True)
        with open(out, "w") as _f:
            for _r in _rows:
                _f.write(json.dumps(_r) + "\n")
    except Exception as _e:
        print(f"[endgames] write failed ({type(_e).__name__}: "
              f"{str(_e)[:120]})", flush=True)
        return {"out": None, "error": type(_e).__name__}
    print(f"[endgames] wrote {len(_rows)} rows "
          f"({len(_harvest)} harvested + {len(_tech)} technical, "
          f"{_bad} invalid, {_rand} clock-randomized) -> {out}",
          flush=True)
    return {"out": out, "rows": len(_rows), "harvested": len(_harvest),
            "technical": len(_tech), "invalid": _bad,
            "randomized": _rand, "pgn_stats": _hstats}


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--pgns", default="pgns/*.pgn")
    _p.add_argument("--out", default="data_sl/endgames.jsonl")
    _p.add_argument("--max-harvest", type=int, default=2000)
    _p.add_argument("--seed", type=int, default=20)
    _a = _p.parse_args()
    print(build_endgames(_a.pgns, _a.out, _a.max_harvest, _a.seed),
          flush=True)
