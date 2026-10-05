"""Tactical puzzle set builder + evaluator (v17 metric): mate-in-1 and
hang-take positions mined from the SL dataset. The representation metric
— hang recall should jump when tactical planes land, loss curves won't
show it. No download beyond the SL games (self-built, not Lichess puzzles).

Usage:
  PYTHONPATH=. python3 -m chess_zero.build_puzzles --games data_sl/games.jsonl \\
      --out data_sl/puzzles.jsonl
  PYTHONPATH=. python3 -m chess_zero.build_puzzles --eval --ckpt <w> \\
      --puzzles data_sl/puzzles.jsonl --blocks 6 --channels 64 --planes 30
"""
from __future__ import annotations

import argparse
import json

VALS = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def build(games_path: str, out_path: str, cap_each: int = 1500,
            min_gain: int = 1):
    import chess
    mates, hangs = [], []
    with open(games_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue
            moves = (g.get("moves", "") or "").split(" ")
            try:
                board = chess.Board()
                for mv in moves:
                    # SAN from the API (see warmstart): parse in context.
                    try:
                        _pm = board.parse_uci(mv)
                    except Exception:
                        _pm = board.parse_san(mv)
                    # scan BEFORE playing: stm to move in this position.
                    fen = board.fen()
                    if len(mates) < cap_each:
                        solutions = []
                        for lm in board.legal_moves:
                            board.push(lm)
                            if board.is_checkmate():
                                solutions.append(lm.uci())
                            board.pop()
                        if solutions:
                            mates.append({"fen": fen, "type": "mate", "targets": solutions})
                    if len(hangs) < cap_each:
                        captures = []
                        for lm in board.legal_moves:
                            if board.is_en_passant(lm):
                                # EP victim: the passed pawn (no piece on
                                # the target square).
                                vic = board.piece_at(
                                    chess.square(
                                        chess.square_file(lm.to_square),
                                        chess.square_rank(lm.from_square)))
                                gain = 1 if vic is not None else 0
                            elif not board.is_capture(lm):
                                continue
                            else:
                                vic = board.piece_at(lm.to_square)
                                if vic is None:
                                    continue
                                gain = VALS.get(vic.symbol().lower(), 0)
                            if gain < min_gain:
                                continue
                            # hanging = the VICTIM is undefended: victim's
                            # side (not turn) does not attack the square.
                            if not board.is_attacked_by(
                                    not board.turn, lm.to_square):
                                captures.append(lm.uci())
                        if captures:
                            hangs.append({"fen": fen, "type": "hang",
                                "label_semantics": "all undefended captures; not engine-proven best moves",
                                "targets": captures})
                    board.push(_pm)
                    if len(mates) >= cap_each and len(hangs) >= cap_each:
                        break
            except Exception:
                continue
            if len(mates) >= cap_each and len(hangs) >= cap_each:
                break
    # FENs are pre-move (side to move faces the tactic).
    mates = mates[:cap_each]
    hangs = hangs[:cap_each]
    with open(out_path, "w") as f:
        for p in mates + hangs:
            f.write(json.dumps(p) + "\n")
    print(f"puzzles: {len(mates)} mates + {len(hangs)} hangs -> {out_path}",
          flush=True)


def evaluate(ckpt: str, puzzles_path: str, blocks: int, channels: int,
             planes: int, se_ratio=None):
    # Eval-science: deterministic 2000/828 live/holdout split by crc32 of
    # the FEN (never tune on holdout; veto if holdout ever drops). Report
    # both splits + Wilson-ish n (paired deltas via McNemar are the
    # promotion-grade comparison, not absolute %).
    import chess
    import numpy as np
    import torch
    import chess_zero.game as _g
    from chess_zero.game import State
    from chess_zero.model import AlphaZeroNet, load_weights, infer_se_ratio
    _g.INPUT_PLANES = planes
    w = torch.load(ckpt, map_location="cpu", weights_only=False)
    if se_ratio is None:
        se_ratio = infer_se_ratio(w)
    m = AlphaZeroNet(blocks=blocks, channels=channels, planes=planes,
                     se_ratio=se_ratio)
    load_weights(m, w, strict=True)
    m.eval()
    import zlib as _z
    stats = {"mate": [0, 0, 0], "hang": [0, 0, 0]}  # top1, top3, n
    hold = {"mate": [0, 0, 0], "hang": [0, 0, 0]}
    with open(puzzles_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            _is_hold = (_z.crc32(p["fen"].encode()) % 2828) >= 2000
            _st = hold if _is_hold else stats
            board = chess.Board(p["fen"])
            st = State(board)
            legal = st.legal_moves()
            x = torch.from_numpy(
                np.stack([st.encode()]).astype(np.float32))
            with torch.no_grad():
                logits = m(x)[0].numpy()[0]
            order = sorted(legal, key=lambda a: -logits[a])
            want = set()
            for t in p["targets"]:
                mv = chess.Move.from_uci(t)
                fr, to = mv.from_square, mv.to_square
                if board.turn == chess.BLACK:
                    from chess_zero.game import mirror_square as _mir
                    fr, to = _mir(fr), _mir(to)
                from chess_zero.game import encode_action
                want.add(encode_action(fr, to, mv.promotion))
            s = _st[p["type"]]
            s[2] += 1
            if order and order[0] in want:
                s[0] += 1
            if any(a in want for a in order[:3]):
                s[1] += 1
    for k, (t1, t3, n) in stats.items():
        print(f"{k}: top1 {t1}/{n}={t1 / max(n, 1):.3f} "
              f"top3 {t3}/{n}={t3 / max(n, 1):.3f}", flush=True)
    for k, (t1, t3, n) in hold.items():
        print(f"{k}-HOLDOUT: top1 {t1}/{n}={t1 / max(n, 1):.3f} "
              f"top3 {t3}/{n}={t3 / max(n, 1):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data_sl/games.jsonl")
    ap.add_argument("--out", default="data_sl/puzzles.jsonl")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--puzzles", default="data_sl/puzzles.jsonl")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--planes", type=int, default=30)
    ap.add_argument("--se-ratio", type=int, default=None,
                     help="SE trunk ratio (default: infer from checkpoint)")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.ckpt, a.puzzles, a.blocks, a.channels, a.planes,
                 a.se_ratio)
    else:
        build(a.games, a.out)


if __name__ == "__main__":
    main()
