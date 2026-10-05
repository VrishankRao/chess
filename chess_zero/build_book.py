"""Empirical opening book builder (v18 adjunct): most-played lines from the
SL dataset become the wide book (200+ lines). Same {"lines": [...]} format
as book.json (load_book compatible). Diversity cap per first move so one
popular line can't eat the book. Training book stays OFF by default
(book_path "") — enable by pointing at the emitted file.

Usage: PYTHONPATH=. python3 -m chess_zero.build_book --games data_sl/games.jsonl \\
    --out chess_zero/book_empirical.json --plies 8 --lines 200
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data_sl/games.jsonl")
    ap.add_argument("--out", default="chess_zero/book_empirical.json")
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--lines", type=int, default=200)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--per-first", type=int, default=12)
    a = ap.parse_args()
    cnt: Counter = Counter()
    import chess
    with open(a.games) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue
            mv = (g.get("moves", "") or "").split(" ")
            if len(mv) < a.plies:
                continue
            # SAN from the API (see warmstart): replay to UCI for the
            # book format (absolute UCI strings).
            try:
                board = chess.Board()
                seq = []
                for s in mv[:a.plies]:
                    try:
                        m = board.parse_uci(s)
                    except Exception:
                        m = board.parse_san(s)
                    seq.append(m.uci())
                    board.push(m)
            except Exception:
                continue
            cnt[tuple(seq)] += 1
    ranked = [(c, seq) for seq, c in cnt.items() if c >= a.min_count]
    ranked.sort(reverse=True)
    picked, per_first = [], Counter()
    for c, seq in ranked:
        if per_first[seq[0]] >= a.per_first:
            continue
        per_first[seq[0]] += 1
        picked.append(list(seq))
        if len(picked) >= a.lines:
            break
    with open(a.out, "w") as f:
        json.dump({"lines": picked}, f)
    print(f"book: {len(picked)} lines from {len(cnt)} distinct "
          f"(min count {a.min_count}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
