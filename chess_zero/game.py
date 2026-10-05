"""Chess State: rules, 4096 codec, plane encoding (13 PDF baseline, 14 v7
phase, 16 v8 repetition, 18 v9 castling+flip, 21 v12 opp-castling+EP,
30 v18 tactical: attacks/defends/hang/checkers/king-danger/material/
opp-bishops/checkerboard/legal-from).
Uses python-chess only as the move generator (no local code reused).
PDF-faithful: from-to 4096 mapping, mask-before-softmax, binary planes + turn.
"""
from __future__ import annotations

import chess
import numpy as np

ACTION_SIZE = 4096
PLANES = 13  # PDF baseline; INPUT_PLANES may grow (v7 phase plane)
INPUT_PLANES = 13

# Every module global the game rules read at runtime. _apply_game_settings
# stamps ALL of these into workers (audit round 2: partial stamps recurred
# twice — the stamp is structural now, not a hand-kept list beside it).
GAME_GLOBALS = ("ADJUDICATE_MARGIN", "NO_PROGRESS_PLIES",
                "ADJUDICATE_MIN_PLY", "INPUT_PLANES", "ADJUDICATE_VALUES")

# Training-only adjudication margin (material pawns). Lower = more decisive
# capped games = stronger win/loss signal, fewer shuffling draws.
# Arena/eval keep their own margin via evaluate.adjudicate_material.
ADJUDICATE_MARGIN = 3.0

# Training-only no-progress rule (plies without pawn move/capture, mirrors
# real chess's 50-move rule at training timescale). Dead shuffling loops
# terminate here instead of drifting to the 300-ply cap.
NO_PROGRESS_PLIES = 100

# Book-blind floor: no-progress adjudication never fires before this ply
# (v7=16). Early middlegame shuffles are theory, not dead draws.
ADJUDICATE_MIN_PLY = 0

# v13: per-game kill-switch for ending truncation. Full-playout games
# (mate-signal) run with this False: adjudication never fires, so games
# end only by mate, rules draws, or the ply cap. Default True = every
# existing caller (arena, gate, tests, UCI, sparring) unchanged. NOT in
# GAME_GLOBALS: it is a per-game override set by play_game around one
# game, never a stamped worker setting.
TRUNCATE_ENDINGS = True


def material_stm(board: chess.Board) -> float:
    """Side-to-move material difference, in pawns/10 (aux-head target)."""
    diff = sum(v * (len(board.pieces(pt, chess.WHITE))
                    - len(board.pieces(pt, chess.BLACK)))
               for pt, v in PIECE_VALUES.items())
    if board.turn == chess.BLACK:
        diff = -diff
    return diff / 10.0


def unit_count_stm(board: chess.Board) -> float:
    """PURE counting target: every non-king piece counts as 1, stm-relative,
    scaled /10 like material_stm. Zero chess knowledge — a hang reads as -1
    unit no matter which piece. The value head re-learns relative worth."""
    diff = sum(len(board.pieces(pt, chess.WHITE))
               - len(board.pieces(pt, chess.BLACK))
               for pt in ORDER if pt != chess.KING)
    if board.turn == chess.BLACK:
        diff = -diff
    return diff / 10.0

# plane order within each color block
ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
         chess.ROOK, chess.QUEEN, chess.KING]

PIECE_VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
                chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}

# Training adjudication values (material pawns). Starts classical; the
# empirical tracker (values.py) slowly rewrites these from game results.
# Arena/eval keep classical via evaluate.adjudicate_material (comparability).
ADJUDICATE_VALUES: dict = dict(PIECE_VALUES)


def adjudicated_edge_stm(board: chess.Board) -> float:
    """Side-to-move material edge in pawns, scored with the LIVE
    adjudication table (starts classical, refit from results). Positive =
    side to move is ahead. Used by v9 search contempt (draw_leaf_value):
    the ahead side learns draws are failure, the behind side achievement.
    Pure counting of what's on the board — no chess knowledge beyond the
    project's own fitted table."""
    diff = sum(v * (len(board.pieces(pt, chess.WHITE))
                    - len(board.pieces(pt, chess.BLACK)))
               for pt, v in ADJUDICATE_VALUES.items())
    if board.turn == chess.BLACK:
        diff = -diff
    return diff


# Codec v2 reserves geometrically impossible from/to slots for underpromotions.
# Ordinary moves and queen promotions retain all legacy action IDs and the
# 4096-wide checkpoint head. No legal king/queen/rook/bishop/knight/pawn move
# can occupy a reserved slot. Paired slots preserve 180-degree orientation.
_PROMOTION_ENCODE = {}
_PROMOTION_DECODE = {}
_candidates = []
for _a in range(4096):
    _f, _t = divmod(_a, 64)
    _dx, _dy = abs((_f % 8)-(_t % 8)), abs((_f // 8)-(_t // 8))
    if _dx and _dy and _dx != _dy and sorted((_dx, _dy)) != [1, 2] and _a < 4095-_a:
        _candidates.append(_a)
_ci = 0
for _file in range(8):
    for _dest in range(max(0, _file-1), min(8, _file+2)):
        for _piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            _fr, _to = 48+_file, 56+_dest
            _code = _candidates[_ci]
            _ci += 1
            for _ff, _tt, _aa in ((_fr, _to, _code), (63-_fr, 63-_to, 4095-_code)):
                _PROMOTION_ENCODE[(_ff, _tt, _piece)] = _aa
                _PROMOTION_DECODE[_aa] = (_ff, _tt, _piece)


def encode_action(from_sq: int, to_sq: int, promotion=None) -> int:
    if promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        return _PROMOTION_ENCODE[(from_sq, to_sq, promotion)]
    return from_sq * 64 + to_sq


def decode_action(a: int) -> tuple[int, int]:
    if int(a) in _PROMOTION_DECODE:
        return _PROMOTION_DECODE[int(a)][:2]
    return divmod(a, 64)


def action_promotion(a):
    return _PROMOTION_DECODE.get(int(a), (None, None, None))[2]


def _queen_only(moves: list[chess.Move]) -> list[chess.Move]:
    """Keep queen promotions only so (from,to) stays unique (4096-compatible)."""
    out = []
    for m in moves:
        if m.promotion is not None and m.promotion != chess.QUEEN:
            continue
        out.append(m)
    return out


def mirror_square(sq: int) -> int:
    """180-degree board rotation (audit item 8 / v9 flip). Maps a square to
    the square the side-to-move orientation sees it from."""
    return chess.square(7 - chess.square_file(sq), 7 - chess.square_rank(sq))


def flip_action(action: int) -> int:
    """Mirror a from×to action through 180 degrees. Self-inverse."""
    fr, to = decode_action(int(action))
    return encode_action(mirror_square(fr), mirror_square(to), action_promotion(action))


class State:
    def __init__(self, board: chess.Board | None = None, _hist=()):
        self.board = chess.Board() if board is None else board
        # Ancestor repetition keys (oldest first, EXCLUDING self). Threaded
        # by apply(); fresh constructions (eval, UCI, tests) start empty =
        # "no known history", which is the truthful value there.
        # audit-1.2: board.copy(stack=False) truncated the move stack, so
        # python-chess repetition detection could never fire. This counter
        # replaces the stack (O(n) ints per state, not O(n) boards).
        self._hist = tuple(_hist)
        # v14: memoized queen-only legal list. The board is NEVER mutated
        # in place (apply() copies), so one generation per State is exact.
        # Callers are read-only: do not mutate the returned list.
        self._legals: list[chess.Move] | None = None
        # v14 Rust core: lazy chesscore.Position snapshot for hot queries.
        # Built once per State from FEN (~20us) then µs queries; None when
        # CHESS_ZERO_NO_CPP=1 or the extension is missing (graceful).
        # Same no-mutation contract as _legals (snapshot goes stale on
        # board mutation — no current path mutates).
        self._rpos = None

    @classmethod
    def initial(cls) -> "State":
        return cls(chess.Board())

    @property
    def side_to_move(self) -> int:
        return 0 if self.board.turn == chess.WHITE else 1

    @property
    def ply_count(self) -> int:
        return self.board.ply()

    def _legal_chess_moves(self) -> list[chess.Move]:
        if self._legals is None:
            self._legals = sorted(self.board.legal_moves,
                key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
        return self._legals

    def _rust_pos(self):
        """Lazy Rust snapshot (None when disabled/missing/failed)."""
        if self._rpos is None:
            import os as _os
            if _os.environ.get("CHESS_ZERO_NO_CPP") != "1":
                try:
                    import chesscore as _cc
                    self._rpos = _cc.Position(self.board.fen())
                    # v14: log backend + extension version once per
                    # process (stale-wheel visibility: a forgotten
                    # reinstall would otherwise serve old movegen
                    # silently).
                    global _CPP_LOGGED
                    if not globals().get("_CPP_LOGGED", False):
                        _CPP_LOGGED = True
                        # stderr: stdout is the UCI channel — a print
                        # there makes strict GUIs flag unexpected output
                        # (caught live on the Lichess bridge).
                        import sys as _sys
                        print(f"[rust] chesscore "
                              f"{getattr(_cc, '__version__', '?')} "
                              f"backend active", file=_sys.stderr,
                              flush=True)
                except Exception:
                    self._rpos = False
            else:
                self._rpos = False
        return self._rpos or None

    def is_capture_fast(self, mv: chess.Move) -> bool:
        """is_capture via Rust when available (hot: _boost_forcing)."""
        rp = self._rust_pos()
        if rp is not None:
            try:
                return bool(rp.is_capture(mv.from_square, mv.to_square))
            except Exception:
                pass
        return self.board.is_capture(mv)

    def gives_check_fast(self, mv: chess.Move) -> bool:
        """gives_check via Rust when available (hot: _boost_forcing)."""
        rp = self._rust_pos()
        if rp is not None:
            try:
                pro = mv.promotion
                p = {None: 0, chess.KNIGHT: 2, chess.BISHOP: 3,
                     chess.ROOK: 4, chess.QUEEN: 5}[pro]
                return bool(rp.gives_check(mv.from_square, mv.to_square,
                                           p))
            except Exception:
                pass
        return self.board.gives_check(mv)

    def in_check_fast(self) -> bool:
        """is_check via Rust when available (hot: quiescence loop)."""
        rp = self._rust_pos()
        if rp is not None:
            try:
                return bool(rp.in_check())
            except Exception:
                pass
        return self.board.is_check()

    def legal_moves(self) -> list[int]:
        """Actions in SIDE-TO-MOVE orientation (v9 flip): for Black, every
        square is 180-degree mirrored, so "forward" is always up-tensor and
        one policy covers both colors. apply() unmirrors transparently, so
        all in-State-space logic (MCTS, self-play, evaluator) is unchanged.
        Anything converting to ABSOLUTE squares (UCI output, PGN record)
        must unmirror for Black stm — see to_uci()."""
        black = self.board.turn == chess.BLACK
        out = []
        for m in self._legal_chess_moves():
            fr, to = m.from_square, m.to_square
            if black:
                fr, to = mirror_square(fr), mirror_square(to)
            out.append(encode_action(fr, to, m.promotion))
        return out

    def to_uci(self, action: int) -> str:
        """Action int -> absolute UCI string (queen-promoting pawns get the
        q suffix so saved games replay). Unmirrors Black-stm actions."""
        return self.to_move(action).uci()

    def to_move(self, action: int) -> chess.Move:
        """Action int -> absolute legal chess.Move (unmirrors Black-stm
        actions; resolves queen promotion from the position). Single choke
        point for everything leaving State-space (UCI output, PGN, SAN)."""
        fr, to = decode_action(int(action))
        if self.board.turn == chess.BLACK:
            fr, to = mirror_square(fr), mirror_square(to)
        for m in self._legal_chess_moves():
            if m.from_square == fr and m.to_square == to and m.promotion == (action_promotion(action) or (chess.QUEEN if m.promotion is not None else None)):
                return m
        pc = self.board.piece_at(fr)
        pro = chess.QUEEN if (pc and pc.piece_type == chess.PAWN
                              and chess.square_rank(to) in (0, 7)) else None
        return chess.Move(fr, to, promotion=pro)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        for a in self.legal_moves():
            mask[a] = True
        return mask

    def apply(self, action: int) -> "State":
        fr, to = decode_action(int(action))
        if self.board.turn == chess.BLACK:
            # v9 flip: incoming actions are stm-oriented; unmirror first.
            fr, to = mirror_square(fr), mirror_square(to)
        # find matching legal move (handles queen promotion automatically)
        chosen = None
        for m in self._legal_chess_moves():
            if m.from_square == fr and m.to_square == to and m.promotion == (action_promotion(action) or (chess.QUEEN if m.promotion is not None else None)):
                chosen = m
                break
        if chosen is None:
            raise ValueError(f"illegal action {action} "
                             f"({chess.square_name(fr)}->{chess.square_name(to)}) "
                             f"for fen {self.board.fen()}")
        nb = self.board.copy(stack=False)
        nb.push(chosen)
        child = State(nb, _hist=self._hist + (self.rep_key(),))
        # v14: inherit Rust snapshot via push (no FEN tax); lazy on miss.
        try:
            if self._rpos:
                p = {None: 0, chess.KNIGHT: 2, chess.BISHOP: 3,
                     chess.ROOK: 4, chess.QUEEN: 5}[chosen.promotion]
                child._rpos = self._rpos.pushed(fr, to, p) or None
        except Exception:
            child._rpos = None
        return child

    def rep_count(self) -> int:
        """Prior occurrences of the CURRENT position this game."""
        return self._hist.count(self.rep_key())

    def is_terminal(self) -> tuple[bool, float]:
        """Returns (done, z from side-to-move perspective).
        Side-channel: sets self.last_terminal to mate | rules-draw |
        adjudicated | adjudicated-draw | None (no contempt logic here —
        targets are shaped in selfplay from this label)."""
        b = self.board
        self.last_terminal = None
        if b.is_checkmate():
            self.last_terminal = "mate"
            return True, -1.0  # side to move is mated
        # audit-1.1: rules draws BEFORE adjudication. Adjudicating first
        # labelled stalemates and dead endgames as wins/losses (verified:
        # stalemate+clock80 taught as -1.0). A drawn position is drawn no
        # matter what the clock says.
        if (b.is_stalemate() or b.is_insufficient_material()
                or b.is_fifty_moves() or b.is_seventyfive_moves()
                or self.rep_count() >= 2):
            # rep_count excludes self: 2 priors + this = threefold.
            # Auto-claimed as a draw (training simplification of FIDE's
            # claim rule; fivefold-equivalent in outcome).
            self.last_terminal = "rules-draw"
            return True, 0.0
        if TRUNCATE_ENDINGS and b.halfmove_clock >= NO_PROGRESS_PLIES \
                and b.ply() >= ADJUDICATE_MIN_PLY:
            self.last_terminal = "adjudicated"
            z = self._adjudicate()
            if z != 0.0:
                return True, z
            self.last_terminal = "adjudicated-draw"
            return True, 0.0
        if TRUNCATE_ENDINGS and b.ply() >= 300:
            self.last_terminal = "adjudicated"
            z = self._adjudicate()
            if z != 0.0:
                return True, z
            self.last_terminal = "adjudicated-draw"
            return True, 0.0
        return False, 0.0

    def _adjudicate(self) -> float:
        """Training-only material adjudication, stm view."""
        b = self.board
        diff = 0.0
        for pt, v in ADJUDICATE_VALUES.items():
            diff += v * (len(b.pieces(pt, chess.WHITE))
                         - len(b.pieces(pt, chess.BLACK)))
        if b.turn == chess.BLACK:
            diff = -diff
        if abs(diff) >= ADJUDICATE_MARGIN:
            return 1.0 if diff > 0 else -1.0
        return 0.0

    def encode(self, planes=None) -> np.ndarray:
        """(planes,8,8) float32. Planes 0-5 us, 6-11 them, 12 turn.
        Plane 13 (v7+): game phase = men on board / 32 (pure counting, no
        chess knowledge). Planes 14-15 (v8+): current position seen once /
        twice before THIS game (from the state's own history counter, so
        training and inference see identical features — audit-1.5: the old
        play_game-only fill left them zero inside every search).
        Planes 16-17 (v9+): stm castling rights, kingside then queenside
        (all-ones when held). The net has never SEEN castling rights, which
        is part of why it walks its king instead of castling — pure rules
        visibility, not knowledge.
        Planes 18-19 (v12+): OPPONENT castling rights, kingside then
        queenside (all-ones when held). Without these the net cannot see
        whether the defender can castle — king safety and attack prospects
        depend on it directly. Uniform fills, rotation-invariant like 16-17.
        Plane 20 (v12+): en-passant target square (single 1.0, zeros when
        none). Spatial like piece planes: set in absolute coordinates before
        the Black rotation so it lands stm-oriented with everything else.
        v9 FLIP (audit item 8, exec-confirmed MAE 0.69 asymmetry): for Black
        stm the whole tensor is rotated 180 degrees, so "forward" is always
        up-tensor and one policy covers both colors. Uniform planes (12-19)
        are rotation-invariant, so only piece planes + EP move. Actions are
        stm-oriented to match (see legal_moves/apply/to_uci).
        Row 0 = rank 8 (white at bottom) before the Black rotation.
        """
        planes = INPUT_PLANES if planes is None else int(planes)
        if planes < 13:
            raise ValueError("Chess encoding requires at least 13 planes")
        t = np.zeros((max(30, planes), 8, 8), dtype=np.float32)
        us = self.board.turn
        them = not us
        for i, pt in enumerate(ORDER):
            for sq in self.board.pieces(pt, us):
                r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                t[i, r, c] = 1.0
            for sq in self.board.pieces(pt, them):
                r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                t[6 + i, r, c] = 1.0
        if self.board.turn == chess.WHITE:
            t[12, :, :] = 1.0
        if planes > 13:
            men = sum(len(self.board.pieces(pt, c))
                      for pt in ORDER for c in (chess.WHITE, chess.BLACK))
            t[13, :, :] = men / 32.0
        if planes > 15:
            nrep = self.rep_count()
            if nrep >= 1:
                t[14, :, :] = 1.0
            if nrep >= 2:
                t[15, :, :] = 1.0
        if planes > 17:
            if self.board.has_kingside_castling_rights(us):
                t[16, :, :] = 1.0
            if self.board.has_queenside_castling_rights(us):
                t[17, :, :] = 1.0
        if planes > 19:
            # v12: opponent castling rights (them KS then QS). Mirrors 16-17
            # but for the defender — uniform fills, rotation-invariant.
            if self.board.has_kingside_castling_rights(them):
                t[18, :, :] = 1.0
            if self.board.has_queenside_castling_rights(them):
                t[19, :, :] = 1.0
        if planes > 20:
            # v12: en-passant target square. Absolute (r,c) here; the Black
            # rotation below carries it stm-oriented with the pieces.
            # v14 fix: only when an EP capture is actually legal — an
            # irrelevant ep_square (no legal en passant) lit the plane.
            ep = self.board.ep_square
            if ep is not None and self.board.has_legal_en_passant():
                r, c = 7 - chess.square_rank(ep), chess.square_file(ep)
                t[20, r, c] = 1.0
        if planes > 21:
            # v18 tactical planes (all absolute pre-rotation; the Black
            # rotation below carries them stm-oriented with the pieces).
            # Rules-derived visibility, zero human games (Giraffe/FX
            # doctrine): the trunk no longer has to rediscover rays to
            # see hangs. Cost ~150 is_attacked_by calls (~0.3ms) vs a
            # 3-10ms CPU forward.
            b = self.board
            us = b.turn
            them = not us
            grip = b.piece_map()
            # 21: squares attacked by THEM. 22: OUR pieces defended by us.
            # 23: OUR pieces en prise (attacked, undefended — SEE0).
            for sq, p in grip.items():
                if p.color != us:
                    continue
                r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                attacked = b.is_attacked_by(them, sq)
                defended = b.is_attacked_by(us, sq)
                if defended:
                    t[22, r, c] = 1.0
                if attacked and not defended:
                    t[23, r, c] = 1.0
            # 21 covers EVERY attacked square (occupied or empty: pins
            # and x-ray lanes light up, not just hanging pieces).
            for sq in range(64):
                if b.is_attacked_by(them, sq):
                    r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                    t[21, r, c] = 1.0
            # 24: checkers (pieces giving check, FX exact).
            for sq in b.checkers():
                r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                t[24, r, c] = 1.0
            # 25: king danger — enemy attacks on the own-king 3x3 zone,
            # clipped 0-3, uniform fill over the zone.
            ks = b.king(us)
            if ks is not None:
                zone = [ks] + [s for s in chess.SQUARES
                               if chess.square_distance(s, ks) == 1]
                hits = sum(1 for s in zone
                           if b.is_attacked_by(them, s))
                v = min(1.0, hits / 3.0)
                for s in zone:
                    r, c = 7 - chess.square_rank(s), chess.square_file(s)
                    t[25, r, c] = v
            # 26: material diff broadcast (classical/39).
            diff = sum(v * (len(b.pieces(pt, chess.WHITE))
                            - len(b.pieces(pt, chess.BLACK)))
                       for pt, v in PIECE_VALUES.items())
            if us == chess.BLACK:
                diff = -diff
            t[26, :, :] = diff / 39.0
            # 27: opposite-color bishops flag (FX: exactly 2 enemy
            # bishops on opposite colors).
            _tb = list(b.pieces(chess.BISHOP, them))
            if len(_tb) == 2 and ((chess.square_file(_tb[0])
                                   + chess.square_rank(_tb[0])) % 2
                                  != (chess.square_file(_tb[1])
                                      + chess.square_rank(_tb[1])) % 2):
                t[27, :, :] = 1.0
            # 28: checkerboard (coordinate anchor; rotation-invariant:
            # 180-degree rotation preserves square color).
            for rr in range(8):
                for cc in range(8):
                    t[28, rr, cc] = float((rr + cc) % 2)
            # 29: own pieces with >=1 legal move (uses memoized legals).
            _froms = {m.from_square for m in self._legal_chess_moves()}
            for sq in _froms:
                r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                t[29, r, c] = 1.0
        if planes >= 31:
            t[30, :, :] = min(1.0, b.halfmove_clock / 100.0)
        if us == chess.BLACK:
            t = np.ascontiguousarray(t[:, ::-1, ::-1])
        return t[:planes]

    def key(self):
        """Search identity includes clock and history; repetition identity does not."""
        return (self.rep_key(), self.board.halfmove_clock, self._hist)

    def rep_key(self):
        """Repetition key: board + turn + castling + ep (no clocks).
        Same position at different times shares a key (unlike key())."""
        b = self.board
        f = getattr(b, "transposition_key", None)
        if callable(f):
            return f()
        return b._transposition_key()

    def __str__(self):
        return str(self.board)


def state_from_encoding(encoded):
    """Reconstruct rule state for legal/check labels (requires castling+EP planes)."""
    x = np.asarray(encoded)
    if x.shape[0] < 21:
        raise ValueError('Legal masks require castling and en-passant planes')
    turn = bool(x[12, 0, 0] > .5)
    absolute = x if turn else x[:, ::-1, ::-1]
    board = chess.Board(None)
    board.turn = turn
    for side, offset in ((turn, 0), (not turn, 6)):
        for j, piece in enumerate(ORDER):
            for row, col in np.argwhere(absolute[offset+j] > .5):
                board.set_piece_at(chess.square(int(col), 7-int(row)), chess.Piece(piece, side))
    for plane, side, square in ((16,turn,7),(17,turn,0),(18,not turn,7),(19,not turn,0)):
        if absolute[plane,0,0] > .5:
            board.castling_rights |= chess.BB_SQUARES[square + (0 if side else 56)]
    ep = np.argwhere(absolute[20] > .5)
    if len(ep):
        row,col = ep[0];board.ep_square = chess.square(int(col),7-int(row))
    if x.shape[0] >= 31:
        board.halfmove_clock = round(float(x[30,0,0])*100)
    return State(board)


def standard_rules(fn):
    """Run a complete evaluation game with training adjudication disabled."""
    from functools import wraps
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global TRUNCATE_ENDINGS
        before = TRUNCATE_ENDINGS
        TRUNCATE_ENDINGS = False
        try:
            return fn(*args, **kwargs)
        finally:
            TRUNCATE_ENDINGS = before
    return wrapped
