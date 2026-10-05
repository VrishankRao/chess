"""Opening book: curated ECO lines (deployment + self-play).
Deployment (UCI): follow a random matching line while the played moves
stay on-book; diverge -> normal search. Self-play: open from a random
line instead of uniform-random plies (real positions, still unrecorded).
Pure-RL note: the book shapes WHERE games start / what deployment plays,
never the training targets (no supervised labels, no SF data). AZ-pure
would use scale instead; this substitutes knowledge for the compute a
5k-TPU run has and an M1 doesn't. Documented deviation, user-approved.
"""
from __future__ import annotations

import json
import os
import random as _random

_BUILTIN = os.path.join(os.path.dirname(__file__), "book.json")
_EMPIRICAL = os.path.join(os.path.dirname(__file__), "book_empirical.json")
_cache: dict = {}


def load_book(path: str) -> list:
    """Load lines (lists of UCI strings from startpos). BUILTIN resolves
    to the packaged file; results cached per path (workers reimport)."""
    if not path or path == "OFF":
        return []
    if path == "BUILTIN":
        path = _BUILTIN
    if path not in _cache:
        with open(path) as f:
            d = json.load(f)
        lines = d["lines"] if isinstance(d, dict) else d
        _cache[path] = [list(l) for l in lines]
    return _cache[path]


def matching_lines(lines: list, played: list) -> list:
    """Lines whose first len(played) moves equal played (startpos only)."""
    n = len(played)
    return [l for l in lines if len(l) > n and l[:n] == played]


def book_move(lines: list, played: list, rng=None) -> str | None:
    """Next book move (random among matching lines) or None if off-book."""
    m = matching_lines(lines, played)
    if not m:
        return None
    r = rng or _random
    return r.choice(m)[len(played)]


def take_book_opening(state, rep_hist: list, lines: list, plies: int,
                      rng=None):
    """Play up to `plies` book moves from a random line. Returns
    (state, n_taken). Moves are unrecorded (no MCTS targets — same rule
    as opening_random_moves) but DO seed rep_hist (real positions).
    Throws nothing: empty lines or short lines just take fewer."""
    import chess as _c
    from .game import State as _S
    if not lines or plies <= 0:
        return state, 0
    r = rng or _random
    line = r.choice(lines)
    n = 0
    for u in line[:plies]:
        try:
            mv = state.board.parse_uci(u)
        except Exception:
            break
        if mv not in state.board.legal_moves:
            break
        from .game import flip_action as _flip
        a = mv.from_square * 64 + mv.to_square
        if state.board.turn == _c.BLACK:
            a = _flip(a)
        state = state.apply(a)
        rep_hist.append(state.rep_key())
        n += 1
    return state, n


def split_book_lines(lines: list, n_train: int = 60) -> tuple:
    """Canonical train/holdout split (single home for the rule; loop.py
    delegates to this): first n_train lines by crc32 hash-order go TRAIN
    (self-play forced book), rest HOLDOUT (gate pairs). Deterministic
    (python hash() is salted per process — never use it here)."""
    try:
        import zlib as _z
        _keyed = sorted(((_z.crc32(repr(list(_l)).encode()), _l)
                         for _l in (lines or [])))
        _ordered = [_l for _, _l in _keyed]
    except Exception:
        _ordered = [list(_l) for _l in (lines or [])]
    _n = max(0, min(int(n_train), len(_ordered)))
    return _ordered[:_n], _ordered[_n:]
