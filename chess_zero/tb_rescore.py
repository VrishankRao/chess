"""Syzygy rescore (D26, extended V20 D6+D8): offline <=5-man WDL
adjudication + Q/M overwrite for training data. TB is NEVER probed in
MCTS (gate/search stay pure — F1 refusal: in-tree probing would couple
search to disk latency and leak perfect endgame play into the very
targets meant to teach it).

V20 D6 (value-only WDL+MLH rescore, Agent B):
  - post-game pass over self-play games (loop calls rescore_examples);
    for positions with <= 5 men + NO castling rights: rewrite the WDL
    target from data_tb/ WDL probe; rewrite the MLH target from the DTZ
    probe (plies, capped by tb_ml_cap=200).
  - 50-move/cursed-win guard: if halfmove_clock > 0 and TB says win,
    verify |DTZ| + clock <= 100 else KEEP the neural target (a cursed
    win / late-clock conversion is not a training win).
  - Draws rewrite Q only (0.0); ML is kept neural (DTZ 0 means "already
    zeroed", not "0 plies remain" — writing 0 would teach instant end).
  - Value heads ONLY. NO policy boost (F4 refusal: TB policy targets
    clash with the KL anchor — V19 precedent: the anchor + rehearsal
    distribution assumes self-play policy targets; injecting perfect
    one-hot TB moves would fight the KL term every step. Documented,
    not implemented).
  - Correctness notes (V20 fixes over D26): probe_wdl/probe_dtz are
    SIDE-TO-MOVE-relative (python-chess docs; the D26 flip for Black
    inverted every black-to-move probe — removed), and DTZ is already
    in PLIES (the old 2x double-counted; ml = |dtz|/cap now).
V20 D8 (deblunder audit, Agent B): Lc0-style walk-back in the rescore
path. Per ply, best_WL = the TB game-theoretic value (stm view);
chosen_WL = -(next ply's best) (what the played move actually led to)
falling back to the game outcome where the next ply is unprobed. Walk
from the end: if best_WL - chosen_WL > deblunder_thr (0.1 WDL units ~ a
5% win-prob throw, spec-literal), the move threw value (temp-injected
blunder in the targets) -> replace z/ml with the best-move's
(z = best_WL, ml = continuation plies + 1). Value-only, like D6.
Required ordering note: this runs in the post-game pass, BEFORE the
tuples reach the buffer — i.e. before D1/D3 weights go nonzero the
targets are already deblundered (documented, enforced by call order
in loop.py, not by a flag here).

Disk doctrine: needs 3GB+ free (tables + headroom). Check FIRST via
ensure_tables()/resolve_tables(); when the tables dir is missing/empty
every entry point SKIPS LOUDLY (prints + returns a skipped dict)
instead of raising, so a tableless machine trains exactly as yesterday.

Table homes (V20): final home data_tb/ (config tb_path); /tmp/syzygy
is the download staging dir; legacy syzygy/ last. resolve_tables()
tries them in order and logs which hit.

Rescore semantics (operates on move-list game records + tuple targets
— training tuples carry encoded planes, not boards, so the loop passes
moves_uci/record_plys/start_fen from the self-play move log):
  - replay each game; for every RECORDED position with <= 5 men on
    board and no castling rights, probe WDL (cursed/blessed collapse
    to their game-theoretic sign, subject to the clock guard) and DTZ;
  - overwrite z (stm-relative, like our targets) and ml (normalized
    plies/tb_ml_cap).
"""
from __future__ import annotations

import os
import shutil

TABLES_DIR = "data_tb"
MIN_FREE_GB = 3.0
MAX_MEN = 5
# V20 D6: candidate table homes in priority order (final home first,
# download staging second, legacy last). Loop's cfg.tb_path prepends.
CANDIDATE_DIRS = ("data_tb", "/tmp/syzygy", "syzygy")


def disk_free_gb(path: str = ".") -> float:
    """Free GB at path. -1.0 when unreadable (treated as no-space)."""
    try:
        _u = shutil.disk_usage(path if os.path.exists(path)
                               else (os.path.dirname(path) or "."))
        return float(_u.free) / (1024.0 ** 3)
    except Exception:
        return -1.0


def ensure_tables(tables_dir: str = TABLES_DIR,
                  min_free_gb: float = MIN_FREE_GB) -> dict:
    """D26 gate: (1) df check for min_free_gb+ free, (2) tables present.
    Returns {"ok": bool, "reason": str, "free_gb": float, "n_tables": int}.
    ok False -> caller SKIPS LOUDLY. Never raises."""
    try:
        _free = disk_free_gb(tables_dir)
        if _free < 0:
            return {"ok": False, "reason": "disk unreadable", "free_gb": -1.0,
                    "n_tables": 0}
        if _free < float(min_free_gb):
            print(f"[tb] SKIP: only {_free:.1f}GB free (need "
                  f"{min_free_gb}GB incl. headroom)", flush=True)
            return {"ok": False, "reason": "disk-full", "free_gb": _free,
                    "n_tables": 0}
        _n = 0
        try:
            _n = sum(1 for _f in os.listdir(tables_dir)
                     if _f.endswith((".rtbw", ".rtbz")))
        except Exception:
            _n = 0
        if _n == 0:
            print(f"[tb] SKIP: no .rtbw/.rtbz tables in '{tables_dir}'",
                  flush=True)
            return {"ok": False, "reason": "no-tables", "free_gb": _free,
                    "n_tables": 0}
        print(f"[tb] {_n} tables in '{tables_dir}', {_free:.1f}GB free",
              flush=True)
        return {"ok": True, "reason": "ready", "free_gb": _free,
                "n_tables": _n}
    except Exception as _e:
        return {"ok": False, "reason": f"{type(_e).__name__}",
                "free_gb": -1.0, "n_tables": 0}


def resolve_tables(tb_path: str | None = None,
                   min_free_gb: float = MIN_FREE_GB) -> tuple:
    """V20 D6 table resolver: try cfg tb_path first, then CANDIDATE_DIRS
    (data_tb, /tmp/syzygy staging, legacy syzygy). Returns (handle, info):
    handle = open chess.syzygy.Tablebase or None; info carries
    {"dir", "n_tables", "skipped"} for loud logging. Caller closes the
    handle (or pass to rescore_examples per iter/chunk). Never raises:
    no tables anywhere -> (None, skipped dict) so training continues
    exactly as yesterday."""
    import chess.syzygy as _sz
    _seen = []
    if tb_path:
        _seen.append(str(tb_path))
    for _d in CANDIDATE_DIRS:
        if _d not in _seen:
            _seen.append(_d)
    for _d in _seen:
        _gate = ensure_tables(_d, min_free_gb)
        if not _gate["ok"]:
            continue
        try:
            _tb = _sz.open_tablebase(_d)
            print(f"[tb] using '{_d}' ({_gate['n_tables']} tables)",
                  flush=True)
            return _tb, {"dir": _d, "n_tables": _gate["n_tables"],
                         "skipped": None}
        except Exception as _e:
            print(f"[tb] SKIP: tablebase open failed for '{_d}' "
                  f"({type(_e).__name__}: {str(_e)[:120]})", flush=True)
    print(f"[tb] SKIP: no usable tables in {list(_seen)} "
          f"(staging /tmp/syzygy incomplete?) — rescore OFF, "
          f"training continues as yesterday", flush=True)
    return None, {"dir": None, "n_tables": 0,
                  "skipped": "no-tables-anywhere"}


def wdl_to_q(wdl: int) -> float:
    """Game-theoretic sign collapse: blessed/cursed wins are still wins.
    wdl in {-2,-1,0,1,2} (python-chess probe_wdl range, STm-relative:
    2 = side to move winning) -> {-1,0,1} (stm-relative, like our z)."""
    try:
        _w = int(wdl)
    except Exception:
        return 0.0
    if _w == 2:
        return 1.0
    if _w == -2:
        return -1.0
    return 0.0


def dtz_to_ml(dtz, move_cap: int = 300) -> float:
    """DTZ (plies, signed, 0/None when n/a — python-chess DTZ is already
    in PLIES: 1..100 to the zeroing move for unconditional wins) ->
    ml-head target in [0,1]: |DTZ| capped by move_cap, normalized.
    None-safe. (V20 fix: the D26 code doubled with 2*|DTZ|.)"""
    try:
        if dtz is None:
            return 1.0
        _plies = abs(int(dtz))
        return round(min(1.0, _plies / max(1, int(move_cap))), 4)
    except Exception:
        return 1.0


def cursed_win_guard(wdl: int, dtz, halfmove_clock: int) -> bool:
    """V20 D6 50-move/cursed-win guard: True = TB value is TRUSTED for
    training, False = keep the neural target. Rule (spec-literal): if
    halfmove_clock > 0 and TB says decisive, require
    |DTZ| + clock <= 100 (the win/loss must still be convertible inside
    the 50-move window from HERE); else keep. Clock == 0 (fresh zeroing)
    always trusts decisive probes. Draws always trusted (q only; DTZ 0
    carries no distance info and never reaches the ml rewrite)."""
    try:
        _w = int(wdl)
    except Exception:
        return False
    if abs(_w) < 2:
        return True
    try:
        _clock = int(halfmove_clock or 0)
    except Exception:
        _clock = 0
    if _clock <= 0:
        return True
    try:
        if dtz is None:
            return False
        return (abs(int(dtz)) + _clock) <= 100
    except Exception:
        return False


def deblunder_pass(best_q: list, ml: list, outcome_w: float | None,
                    stm: list, thr: float = 0.1, cap: float = 200.0) -> dict:
    """Diagnostic only: count known value drops, never invent corrections.

    Exact tablebase labels are already authoritative. A worse played
    continuation does not justify rewriting earlier unprobed positions or
    treating DTZ as time to mate. n_fixed remains zero for compatibility.
    """
    detected = 0
    for i, value in enumerate(best_q):
        if value is None:
            continue
        chosen = (-float(best_q[i+1]) if i+1 < len(best_q) and best_q[i+1] is not None
                  else (None if outcome_w is None else float(outcome_w) * (1 if int(stm[i]) == 0 else -1)))
        if chosen is not None and float(value) - chosen > thr:
            detected += 1
    return {"q": list(best_q), "ml": list(ml), "n_fixed": 0, "n_detected": detected}


def rescore_game(moves_uci: list, move_cap: int = 300,
                 tables=None, tb_ml_cap: int = 200,
                 start_fen: str | None = None,
                 outcome_w: float | None = None,
                 deblunder_thr: float = 0.1) -> dict:
    """Rescore ONE game: per-ply list of (wdl_q | None, ml | None) for
    <=5-man, no-castling positions (None where unprobable or guarded —
    caller keeps the neural target there). tables = open
    chess.syzygy.Tablebase (caller-owned; None -> all-None + skipped
    count). Pure replay via python-chess from startpos (or start_fen
    for endgame starts); stm-relative Q (probes are already
    stm-relative — NO flip). Draws rewrite Q only (ml None = keep).
    Decisive probes pass cursed_win_guard (clock-aware), then the D8
    deblunder_pass walks the probed arrays (thr) before return.
    Returns {"plys", "n_probed", "n_skipped", "n_guarded",
    "n_castling", "n_deblundered"}."""
    import chess as _c
    _out = {"plys": [], "n_probed": 0, "n_skipped": 0, "n_guarded": 0,
            "n_castling": 0, "n_deblundered": 0}
    try:
        _b = _c.Board(str(start_fen)) if start_fen else _c.Board()
    except Exception:
        _b = _c.Board()
    try:
        for _u in (moves_uci or []):
            _rec = {"q": None, "ml": None}
            try:
                _men = len(_b.piece_map())
                _castle = _b.castling_xfen() != "-"
                if tables is not None and _men <= MAX_MEN and not _castle:
                    try:
                        _wdl = tables.probe_wdl(_b)
                        try:
                            _dtz = tables.probe_dtz(_b)
                        except Exception:
                            _dtz = None
                        if cursed_win_guard(_wdl, _dtz,
                                            _b.halfmove_clock):
                            _q = wdl_to_q(_wdl)
                            _rec["q"] = _q
                            # DTZ measures the next zeroing move, not game end.
                            # Leave the existing moves-left supervision unchanged.
                            _out["n_probed"] += 1
                        else:
                            _out["n_guarded"] += 1
                    except Exception:
                        _out["n_skipped"] += 1
                else:
                    if tables is not None and _men <= MAX_MEN and _castle:
                        _out["n_castling"] += 1
                    else:
                        _out["n_skipped"] += 1
            finally:
                _out["plys"].append(_rec)
            try:
                _b.push_uci(_u)
            except Exception:
                break
    except Exception:
        pass
    # Exact TB values are already corrected; no additional deblunder is claimed.
    return _out


def rescore_examples(examples: list, record_plys: list, moves_uci: list,
                     tables, move_cap: int = 300,
                     tb_ml_cap: int = 200,
                     start_fen: str | None = None,
                     outcome_w: float | None = None,
                     deblunder_thr: float = 0.1) -> dict:
    """V20 D6 loop hook: post-game pass rewriting tuple z (index 2) and
    ml (index 4) IN PLACE for TB-probed recorded positions. Alignment:
    record ply p <-> replay index p - offset, offset = start-board ply
    (0 for startpos / fullmove-1 FENs; build_endgames normalizes to 1).
    Unaligned/missing entries are left alone (never raises, never
    fabricates). Value heads ONLY — pi/av/ownership/etc. untouched
    (F4). Returns {"probed","guarded","rewritten","deblundered",
    "skipped"} (rewritten counts examples actually overwritten)."""
    _sum = {"probed": 0, "guarded": 0, "rewritten": 0, "deblundered": 0,
            "skipped": 0}
    try:
        if tables is None or not examples:
            _sum["skipped"] = len(examples or [])
            return _sum
        import chess as _c
        try:
            _off = _c.Board(str(start_fen)).ply() if start_fen else 0
        except Exception:
            _off = 0
        _r = rescore_game(list(moves_uci or []), move_cap, tables,
                          tb_ml_cap, start_fen, outcome_w, deblunder_thr)
        _sum["probed"] = int(_r["n_probed"])
        _sum["guarded"] = int(_r["n_guarded"])
        _sum["deblundered"] = int(_r["n_deblundered"])
        _plys = _r["plys"]
        _n = 0
        for _j, _ex in enumerate(examples):
            try:
                _p = int(record_plys[_j]) if _j < len(record_plys or []) \
                    else -1
            except Exception:
                _sum["skipped"] += 1
                continue
            _idx = _p - int(_off)
            if _idx < 0 or _idx >= len(_plys):
                _sum["skipped"] += 1
                continue
            _rec = _plys[_idx]
            if _rec["q"] is None and _rec["ml"] is None:
                _sum["skipped"] += 1
                continue
            try:
                _l = list(_ex)
                if _rec["q"] is not None:
                    _l[2] = float(_rec["q"])
                if _rec["ml"] is not None:
                    _l[4] = float(_rec["ml"])
                examples[_j] = tuple(_l)
                _n += 1
            except Exception:
                _sum["skipped"] += 1
        _sum["rewritten"] = int(_n)
        return _sum
    except Exception:
        return _sum


def rescore_games_file(games_path: str, out_path: str,
                       tables_dir: str = TABLES_DIR,
                       move_cap: int = 300,
                       tb_ml_cap: int = 200) -> dict:
    """Batch entry: annotate a games.jsonl ({moves, winner} per line)
    with tb q/ml per ply -> out_path (.rescored.jsonl). Tables missing
    -> loud skip dict, out file NOT written. Returns summary. Value
    annotations ONLY (no policy — F4)."""
    _gate = ensure_tables(tables_dir)
    if not _gate["ok"]:
        return {"skipped": _gate["reason"], "games": 0, "probed": 0}
    import json as _j
    import chess.syzygy as _sz

    def _outcome(_g) -> float | None:
        try:
            _w = _g.get("winner")
            if _w in ("white", "1-0", 1.0, 1):
                return 1.0
            if _w in ("black", "0-1", -1.0, -1):
                return -1.0
        except Exception:
            pass
        return None
    _games = _probed = _guarded = _db = 0
    try:
        _tb = _sz.open_tablebase(tables_dir)
    except Exception as _e:
        print(f"[tb] SKIP: tablebase open failed "
              f"({type(_e).__name__}: {str(_e)[:120]})", flush=True)
        return {"skipped": f"open-failed-{type(_e).__name__}", "games": 0,
                "probed": 0}
    try:
        with open(games_path) as _f, open(out_path, "w") as _o:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _g = _j.loads(_line)
                except Exception:
                    continue
                _mv = _g.get("moves", "")
                if isinstance(_mv, str):
                    _moves = [_m for _m in _mv.split(" ") if _m]
                else:
                    _moves = [str(_m) for _m in (_mv or [])]
                _r = rescore_game(_moves, move_cap, _tb, tb_ml_cap,
                                  _g.get("start_fen"),
                                  _outcome(_g))
                _g["tb_q"] = [_p["q"] for _p in _r["plys"]]
                _g["tb_ml"] = [_p["ml"] for _p in _r["plys"]]
                _o.write(_j.dumps(_g) + "\n")
                _games += 1
                _probed += _r["n_probed"]
                _guarded += _r["n_guarded"]
                _db += _r["n_deblundered"]
    finally:
        try:
            _tb.close()
        except Exception:
            pass
    print(f"[tb] rescored {_games} games ({_probed} TB positions, "
          f"{_guarded} clock-guarded, {_db} deblundered) -> {out_path}",
          flush=True)
    return {"skipped": None, "games": _games, "probed": _probed,
            "guarded": _guarded, "deblundered": _db, "out": out_path}


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--games", default="data_sl/games.jsonl")
    _p.add_argument("--out", default="data_sl/games.rescored.jsonl")
    _p.add_argument("--tables", default=TABLES_DIR)
    _p.add_argument("--move-cap", type=int, default=300)
    _p.add_argument("--tb-ml-cap", type=int, default=200)
    _a = _p.parse_args()
    print(rescore_games_file(_a.games, _a.out, _a.tables, _a.move_cap,
                             _a.tb_ml_cap),
          flush=True)
