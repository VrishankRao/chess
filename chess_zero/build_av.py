"""V20 D3: offline Stockfish action-value (AV) distillation labels.

For positions in data_sl/games.jsonl (+ harvested endgames), query Stockfish
(/usr/local/bin/stockfish, depth 12, multipv = all-legal capped at 32) and
store the top-Q per legal move as an int8-centipawn vector for the V20 `av`
aux head (training gradient only — NEVER wired into search/veto).

Conventions (shared with warmstart.py / train.py Agent-A head):
  - scores are SIDE-TO-MOVE-relative centipawns (match the stm-oriented pi
    codec), clipped to +-AV_CP_CLIP, quantized int8 with quantum
    AV_CP_SCALE (q8 = round(cp/12), cp = q8*12; +-1500/12 = +-125 fits int8).
  - the distillation loss (Agent A) is av_w=0.1 * MSE(softmax(net_av/T) ||
    softmax(sf_av/T)) with T=2.0 over LEGAL moves; decode_av_logits()
    emits pawns (cp/100) so T is in pawns on both sides.
  - output rows are JSON lines: {"id", "ply", "fen", "stm", "av": {uci: q8},
    "depth"}. Resume-safe: existing output keys {(id, ply)} are skipped, so
    re-running over data_sl/games.jsonl only labels new rows.

Usage (DO NOT RUN without user order — offline batch, spawns engines):
  PYTHONPATH=. python3 -m chess_zero.build_av --games data_sl/games.jsonl \\
      --out data_sl/av.jsonl --depth 12 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os

STOCKFISH = "/usr/local/bin/stockfish"
DEPTH = 12
MULTIPV_CAP = 32
WORKERS = 6
AV_CP_CLIP = 1500.0
AV_CP_SCALE = 12.0  # int8 quantum: q8 = clip(round(cp/12), -127, 127)


def quantize_cp(cp: float) -> int:
    """Centipawns -> int8 quantum (documented scale above)."""
    try:
        _c = float(cp)
    except Exception:
        return 0
    import math as _m
    if not _m.isfinite(_c):
        return 0
    _c = max(-AV_CP_CLIP, min(AV_CP_CLIP, _c))
    return max(-127, min(127, int(round(_c / AV_CP_SCALE))))


def dequantize_q8(q: int) -> float:
    """Int8 quantum -> centipawns (inverse of quantize_cp)."""
    try:
        return float(int(q)) * AV_CP_SCALE
    except Exception:
        return 0.0


def label_position(stockfish_path: str, fen: str, depth: int = DEPTH,
                   multipv_cap: int = MULTIPV_CAP,
                   engine=None) -> dict | None:
    """Label ONE position: {uci: q8} over all legal moves (stm-relative cp).

    engine (optional): an open chess.engine.SimpleEngine (worker processes
    keep one engine open; None = open/close per call). Returns None for
    terminal positions (no legal moves) or engine failures (row skipped,
    never crash the batch).
    """
    import chess as _ch
    try:
        _bd = _ch.Board(fen)
    except Exception:
        return None
    try:
        _legal = list(_bd.legal_moves)
    except Exception:
        return None
    if not _legal:
        return None
    _n = max(1, min(len(_legal), int(multipv_cap)))
    _own = False
    try:
        if engine is None:
            import chess.engine as _ce
            engine = _ce.SimpleEngine.popen_uci(stockfish_path)
            _own = True
        try:
            engine.configure({"Threads": 1, "Hash": 64})
        except Exception:
            pass
        import chess.engine as _ce2
        _res = engine.analyse(_bd, _ce2.Limit(depth=int(depth)),
                              multipv=_n)
        if isinstance(_res, dict):
            _res = [_res]
        _out: dict = {}
        for _info in (_res or []):
            try:
                _pv = _info.get("pv")
                if not _pv:
                    continue
                _u = _pv[0].uci()
                # stm-relative: pov(side to move), mate mapped to +-clip
                # then clipped again inside quantize_cp.
                _cp = _info["score"].pov(_bd.turn).score(
                    mate_score=int(AV_CP_CLIP))
                _out[str(_u)] = quantize_cp(_cp)
            except Exception:
                continue
        return _out or None
    except Exception:
        return None
    finally:
        if _own:
            try:
                engine.quit()
            except Exception:
                pass


# Worker-process engine (one Stockfish per pool worker; fork-safe because
# the engine is opened in the initializer, after forking).
_W = {}


def _worker_init(stockfish_path: str):
    import chess.engine as _ce
    _W["engine"] = _ce.SimpleEngine.popen_uci(stockfish_path)
    try:
        _W["engine"].configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass


def _worker_label(job):
    _fen, _depth, _cap = job
    try:
        return label_position("", _fen, _depth, _cap,
                              engine=_W.get("engine"))
    except Exception:
        return None


def iter_positions(games_path: str, max_positions: int = 0):
    """Yield (key, fen, stm) with key = (game_id, ply).

    game_id = row "id" when present else "line{lineno}" (stable across runs,
    so resume-skip joins correctly). Positions replayed from the move list;
    unparseable games/lines skipped. max_positions <= 0 = all.
    """
    import chess as _ch
    _n = 0
    with open(games_path) as _f:
        for _lineno, _line in enumerate(_f):
            _line = _line.strip()
            if not _line:
                continue
            try:
                _g = json.loads(_line)
            except Exception:
                continue
            _moves = (_g.get("moves", "") or "").split(" ")
            _moves = [m for m in _moves if m]
            if len(_moves) < 2:
                continue
            _gid = str(_g.get("id") or f"line{_lineno}")
            _bd = _ch.Board()
            for _ply, _tok in enumerate(_moves):
                try:
                    try:
                        _m = _bd.parse_uci(_tok)
                    except Exception:
                        _m = _bd.parse_san(_tok)
                except Exception:
                    break
                _fen = _bd.fen()
                _stm = "w" if _bd.turn == _ch.WHITE else "b"
                yield (_gid, _ply), _fen, _stm
                _bd.push(_m)
                _n += 1
                if max_positions and max_positions > 0 and _n >= max_positions:
                    return


def load_done_keys(out_path: str) -> set:
    """Resume-safe: keys {(id, ply)} already labeled in out_path."""
    _done: set = set()
    if not out_path or not os.path.exists(out_path):
        return _done
    try:
        with open(out_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _r = json.loads(_line)
                    _done.add((str(_r.get("id")), int(_r.get("ply"))))
                except Exception:
                    continue
    except Exception:
        pass
    return _done


def build(games_path: str, out_path: str, depth: int = DEPTH,
          workers: int = WORKERS, stockfish_path: str = STOCKFISH,
          max_positions: int = 0, multipv_cap: int = MULTIPV_CAP,
          label_fn=None) -> dict:
    """Offline batch: label positions, append rows to out_path (resume-safe).

    label_fn (test seam): fn(fen) -> {uci: q8} | None, bypasses engines.
    Returns {"done", "skipped", "failed", "out"} counts. Never raises on
    engine/parse failures (rows skipped, counted).
    """
    _done_keys = load_done_keys(out_path)
    _jobs = []  # (key, fen, stm)
    for _key, _fen, _stm in iter_positions(games_path, max_positions):
        if _key in _done_keys:
            continue
        _jobs.append((_key, _fen, _stm))
    _skipped = len(_done_keys)
    _ok = _fail = 0
    _od = os.path.dirname(out_path) or "."
    try:
        os.makedirs(_od, exist_ok=True)
    except Exception:
        pass
    _f = open(out_path, "a")
    try:
        if label_fn is not None:
            for (_gid, _ply), _fen, _stm in _jobs:
                try:
                    _av = label_fn(_fen)
                except Exception:
                    _av = None
                if not _av:
                    _fail += 1
                    continue
                _f.write(json.dumps({"id": _gid, "ply": _ply, "fen": _fen,
                                     "stm": _stm, "av": _av,
                                     "depth": int(depth)}) + "\n")
                _ok += 1
            _f.flush()
            return {"done": _ok, "skipped": _skipped, "failed": _fail,
                    "out": out_path}
        from concurrent.futures import ProcessPoolExecutor as _Pool
        _pool_jobs = [(_fen, int(depth), int(multipv_cap))
                      for _, _fen, _ in _jobs]
        with _Pool(max_workers=max(1, int(workers)),
                   initializer=_worker_init,
                   initargs=(stockfish_path,)) as _ex:
            for (_job, _av) in zip(_jobs, _ex.map(_worker_label,
                                                  _pool_jobs)):
                (_gid, _ply), _fen, _stm = _job
                if not _av:
                    _fail += 1
                    continue
                _f.write(json.dumps({"id": _gid, "ply": _ply, "fen": _fen,
                                     "stm": _stm, "av": _av,
                                     "depth": int(depth)}) + "\n")
                _ok += 1
                if _ok % 50 == 0:
                    _f.flush()
        _f.flush()
    finally:
        try:
            _f.close()
        except Exception:
            pass
    return {"done": _ok, "skipped": _skipped, "failed": _fail,
            "out": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data_sl/games.jsonl")
    ap.add_argument("--endgames", default="",
                    help="optional extra games.jsonl (harvested endgames); "
                         "labeled into the same output")
    ap.add_argument("--out", default="data_sl/av.jsonl")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--stockfish", default=STOCKFISH)
    ap.add_argument("--max-positions", type=int, default=0,
                    help="cap labeled positions (0 = all)")
    ap.add_argument("--multipv-cap", type=int, default=MULTIPV_CAP)
    a = ap.parse_args()
    print(f"build_av: {a.games} -> {a.out} (depth {a.depth}, "
          f"workers {a.workers})", flush=True)
    _rep = build(a.games, a.out, a.depth, a.workers, a.stockfish,
                 a.max_positions, a.multipv_cap)
    print(f"build_av: labeled {_rep['done']} (skipped {_rep['skipped']} "
          f"resume, failed {_rep['failed']})", flush=True)
    if a.endgames:
        _rep2 = build(a.endgames, a.out, a.depth, a.workers, a.stockfish,
                      a.max_positions, a.multipv_cap)
        print(f"build_av endgames: labeled {_rep2['done']} (skipped "
              f"{_rep2['skipped']}, failed {_rep2['failed']})", flush=True)


if __name__ == "__main__":
    main()
