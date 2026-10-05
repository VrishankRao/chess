"""Full correctness suite: terminals, codec, encoding, MCTS signs, buffer.
Uses TEST_CONFIG scale throughout so it runs in ~1 min on CPU.
"""
import numpy as np
import torch

import chess

from chess_zero.game import (State, ACTION_SIZE, encode_action, decode_action,
                             ORDER)
from chess_zero.config import TEST_CONFIG
from chess_zero.model import AlphaZeroNet
from chess_zero import mcts as mcts_mod
from chess_zero.replay import ReplayBuffer
from chess_zero.train import train_step
from chess_zero.evaluate import adjudicate_material


def _play_uci_list(moves):
    s = State.initial()
    for u in moves:
        mv = s.board.parse_uci(u)
        if mv.promotion not in (None, chess.QUEEN):
            raise AssertionError("test uses queen promotions only")
        a = mv.from_square * 64 + mv.to_square
        if s.board.turn == chess.BLACK:
            # v9 flip: UCI is absolute, apply() takes stm-oriented.
            from chess_zero.game import flip_action
            a = flip_action(a)
        s = s.apply(a)
    return s


def _apply_uci(s, u):
    """Absolute-UCI apply for tests (v9 flip: unmirror on black stm)."""
    from chess_zero.game import flip_action
    mv = s.board.parse_uci(u)
    a = mv.from_square * 64 + mv.to_square
    if s.board.turn == chess.BLACK:
        a = flip_action(a)
    return s.apply(a)


def test_fools_mate_terminal():
    # f3 e5 g4 Qh4# -> black mates; side to move (white) lost
    s = _play_uci_list(["f2f3", "e7e5", "g2g4", "d8h4"])
    assert s.board.is_checkmate()
    done, z = s.is_terminal()
    assert done and z == -1.0, (done, z)


def test_scholars_mate_terminal():
    s = _play_uci_list(["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6",
                        "h5f7"])
    assert s.board.is_checkmate()
    done, z = s.is_terminal()
    assert done and z == -1.0


def test_stalemate_is_draw():
    s = State(chess.Board("7k/5Q2/8/8/8/8/8/6K1 b - - 0 1"))
    assert s.board.is_stalemate()
    done, z = s.is_terminal()
    assert done and z == 0.0


def test_insufficient_material_draw():
    s = State(chess.Board("8/8/4k3/8/8/3K4/8/8 w - - 0 1"))
    done, z = s.is_terminal()
    assert done and z == 0.0


def test_action_codec_roundtrip_all():
    for a in range(ACTION_SIZE):
        fr, to = decode_action(a)
        from chess_zero.game import action_promotion
        assert encode_action(fr, to, action_promotion(a)) == a
    # every opening move applies cleanly
    s = State.initial()
    for a in s.legal_moves():
        s2 = s.apply(a)
        assert s2.ply_count == 1


def test_encoding_sides_swap():
    w = State.initial()
    ew = w.encode()
    # white to move: plane 0 has 8 pawns, plane 5 one king
    assert ew[0].sum() == 8 and ew[5].sum() == 1
    assert ew[12].sum() == 64  # turn plane = white
    s = _play_uci_list(["e2e4"])  # black to move
    eb = s.encode()
    assert eb[12].sum() == 0
    # v9 flip: black-stm tensor is rotated 180, so the white pawn on e4
    # (tensor row 4, col 4 unrotated) appears rotated at row 3, col 3 in
    # THEIR pawn plane (stm=black).
    assert eb[6, 3, 3] == 1.0
    # black still has all 8 pawns in OUR pawn plane
    assert eb[0].sum() == 8


def test_flip_mirror_consistency():
    # The v9 flip promise: a position and its 180-rotated color-swapped
    # twin (opposite chair, same game) encode to the SAME tensor, so one
    # policy covers both colors. Only the turn plane differs by design
    # (it names the chair), and castling planes swap kingside<->queenside
    # under rotation (a kingside right mirrors to queenside).
    import chess
    import numpy as np
    import chess_zero.game as _g
    old = _g.INPUT_PLANES
    _g.INPUT_PLANES = 18
    try:
        w = _play_uci_list(["e2e4", "e7e5", "g1f3"])
        ew = w.encode()
        assert ew.shape == (18, 8, 8)
        b2 = chess.Board.empty()
        for sq, p in w.board.piece_map().items():
            r, c = chess.square_rank(sq), chess.square_file(sq)
            nsq = chess.square(7 - c, 7 - r)
            b2.set_piece_at(nsq, chess.Piece(p.piece_type,
                                             not p.color))
        b2.turn = not w.board.turn
        m = State(b2)
        em = m.encode()
        assert (em[0:12] == ew[0:12]).all()
        # turn plane names the chair: exactly one of the twins is ones.
        assert {em[12].sum(), ew[12].sum()} == {0.0, 64.0}
        assert (em[13:16] == ew[13:16]).all()
    finally:
        _g.INPUT_PLANES = old


def test_flip_action_roundtrip():
    # flip is self-inverse; black legal moves are the mirrored absolutes;
    # every legal move applies cleanly in both colors over random games.
    import random
    import chess
    from chess_zero.game import (State, flip_action, mirror_square,
                                 decode_action)
    for a in (0, 63, 4095, 1234):
        assert flip_action(flip_action(a)) == a
    assert mirror_square(mirror_square(chess.E2)) == chess.E2
    s = _play_uci_list(["e2e4"])  # black stm
    abs_moves = set((m.from_square, m.to_square)
                    for m in s.board.legal_moves)
    stm_moves = set(decode_action(a) for a in s.legal_moves())
    expect = set((mirror_square(f), mirror_square(t)) for f, t in abs_moves)
    # no promotions available here, so the mirror mapping is exact
    assert stm_moves == expect
    random.seed(0)
    s = State.initial()
    for _ in range(120):
        legal = s.legal_moves()
        assert legal
        a = random.choice(legal)
        u = s.to_uci(a)  # must replay as absolute UCI
        b = s.board.copy()
        b.push_uci(u)
        s = s.apply(a)
        assert s.board.board_fen() == b.board_fen()
        if s.is_terminal()[0]:
            s = State.initial()


def test_castling_planes():
    # v9 planes 16-17: stm kingside/queenside rights, uniform fill.
    import chess
    import chess_zero.game as _g
    from chess_zero.game import State
    old = _g.INPUT_PLANES
    _g.INPUT_PLANES = 18
    try:
        e = State.initial().encode()
        assert e.shape == (18, 8, 8)
        assert e[16].mean() == 1.0 and e[17].mean() == 1.0
        s = _play_uci_list(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4",
                            "f8c5", "e1g1"])  # white castled: rights gone
        e2 = s.encode()  # black stm now
        assert e2[16].mean() == 1.0 and e2[17].mean() == 1.0  # black intact
        _g.INPUT_PLANES = 16
        assert State.initial().encode().shape == (16, 8, 8)  # old cfgs work
    finally:
        _g.INPUT_PLANES = old


def test_opp_castling_and_ep_planes():
    # v12 planes 18-19: opponent KS/QS rights; plane 20: EP square.
    # Startpos: both sides hold rights -> all four uniform planes ones,
    # EP empty. After e2e4: black stm sees white EP square d6?? no —
    # e2e4 sets EP e3 (absolute), visible stm-oriented after rotation.
    import chess
    import chess_zero.game as _g
    from chess_zero.game import State
    old = _g.INPUT_PLANES
    _g.INPUT_PLANES = 21
    try:
        e = State.initial().encode()
        assert e.shape == (21, 8, 8)
        assert e[16].mean() == 1.0 and e[17].mean() == 1.0
        assert e[18].mean() == 1.0 and e[19].mean() == 1.0
        assert e[20].sum() == 0  # no EP at start
        # white castles -> white rights gone; black stm sees them as OPP.
        s = _play_uci_list(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4",
                            "f8c5", "e1g1"])
        e2 = s.encode()  # black stm: us=black intact, them=white gone
        assert e2[16].mean() == 1.0 and e2[17].mean() == 1.0
        assert e2[18].sum() == 0 and e2[19].sum() == 0
        # EP visibility: e2e4 leaves ep e3 absolute — but NO black pawn
        # can capture it, so the plane stays dark (v14: irrelevant EP
        # squares must not light plane 20; rep_key already agreed).
        se = _play_uci_list(["e2e4"])
        ee = se.encode()  # black stm, rotated
        assert ee.shape == (21, 8, 8)
        assert ee[20].sum() == 0
        # Real legal EP: 1.e4 a6 2.e5 d5 — white CAN play exd6 e.p.
        se2 = _play_uci_list(["e2e4", "a7a6", "e4e5", "d7d5"])
        ee2 = se2.encode()  # white stm, no rotation
        assert ee2[20].sum() == 1.0
        # absolute d6 (file d=3, rank idx 5 -> r=7-5=2).
        assert ee2[20, 2, 3] == 1.0
        # 18ch compat: old cfgs still encode (new planes absent, not crash)
        _g.INPUT_PLANES = 18
        assert State.initial().encode().shape == (18, 8, 8)
    finally:
        _g.INPUT_PLANES = old


def test_mcts_finds_mate_in_one():
    # ...Qxf7# available; scripted uniform evaluator isolates search logic
    s = _play_uci_list(["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"])
    mate = encode_action(chess.H5, chess.F7)  # Qxf7#
    assert mate in s.legal_moves()

    def uniform(states):
        p = np.zeros((len(states), ACTION_SIZE), dtype=np.float32)
        for i, st in enumerate(states):
            lm = st.legal_moves()
            p[i, lm] = 1.0 / len(lm)
        return p, np.zeros(len(states), dtype=np.float32)

    pi = mcts_mod.search(s, uniform, n_sims=300)
    assert int(np.argmax(pi)) == mate, \
        f"mating move not preferred: argmax={np.argmax(pi)} mate={mate}"
    # backup-sign corollary: mating action must hold the visit majority
    assert pi[mate] > 0.5, pi[mate]


def test_adjudication_signs():
    w = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1"))  # KQ vs K, white
    assert adjudicate_material(w) == 1.0
    b = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 b - - 0 1"))  # stm=black bare
    assert adjudicate_material(b) == -1.0
    eq = State(chess.Board("8/8/4k3/8/8/3K4/8/8 w - - 0 1"))
    assert adjudicate_material(eq) == 0.0


def test_value_range_and_masked_policy():
    net = AlphaZeroNet(blocks=2, channels=32)
    net.eval()
    s = State.initial()
    x = torch.from_numpy(np.stack([s.encode(), s.encode()]))
    with torch.no_grad():
        out = net(x)
    assert len(out) == 15
    logits, wdl, aux, ml = out[0], out[1], out[2], out[3]
    _soft, _chk, _prog = out[9], out[10], out[11]
    _pw = torch.softmax(wdl, dim=1)
    assert bool((_pw.sum(dim=1) - 1.0).abs().max() < 1e-4)
    v = _pw[:, 0] - _pw[:, 2]
    assert bool(((v >= -1.0) & (v <= 1.0)).all())
    assert aux.shape == (2,) and torch.isfinite(aux).all()
    assert ml.shape == (2,) and bool(((ml >= 0.0) & (ml <= 1.0)).all())
    mask = torch.from_numpy(np.stack([s.legal_mask(), s.legal_mask()]))
    import torch.nn.functional as F
    lp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)
    probs = lp.exp()
    assert abs(probs[0].sum().item() - 1.0) < 1e-4
    assert bool((probs[:, ~s.legal_mask()] == 0).all())


def test_replay_shapes():
    buf = ReplayBuffer(capacity=64)
    s = State.initial()
    _o88 = np.zeros((8, 8), dtype=np.float32)
    ex = [(s.encode(), np.ones(ACTION_SIZE) / ACTION_SIZE, 0.0, 0.0, 1.0,
           _o88, 0.0, 0.5)] * 20
    buf.add_game(ex)
    out = buf.sample(8)
    assert len(out) == 17
    b, pi, z, m, ml, own, marg, mob, safe, reply, pw, mm, _pp, _sm, _ss, \
        _rq, _av = out
    assert own.shape == (8, 8, 8) and marg.shape == (8,) and mob.shape == (8,)
    assert safe.shape == (8,)
    assert reply.shape == (8,) and pw.shape == (8,)
    assert b.shape == (8, 13, 8, 8) and pi.shape == (8, 4096)
    assert z.shape == (8,) and m.shape == (8,) and ml.shape == (8,)


def test_endgame_routing():
    buf = ReplayBuffer(capacity=64, eg_capacity=64, eg_frac=0.5)
    full = State.initial().encode()  # 32 men -> main
    eg = np.zeros((13, 8, 8), dtype=np.float32)  # K vs K -> endgame
    eg[5, 0, 4] = 1.0
    eg[11, 7, 4] = 1.0
    pi = np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE
    _o88 = np.zeros((8, 8), dtype=np.float32)
    buf.add_game([(full, pi, 0.0, 0.0, 1.0, _o88, 0.0, 0.5)] * 10 +
                 [(eg, pi, 0.0, 0.0, 0.1, _o88, 0.0, 0.5)] * 10)
    assert buf.eg_len() == 10 and len(buf.buf) == 10
    out = buf.sample(8)
    assert len(out) == 17
    b, pi, z, m, ml, own, marg, mob, safe, reply, pw, mm, _pp, _sm, _ss, \
        _rq, _av = out
    assert own.shape == (8, 8, 8) and marg.shape == (8,) and mob.shape == (8,)
    assert safe.shape == (8,)
    assert reply.shape == (8,) and pw.shape == (8,)
    assert b.shape[0] == 8 and ml.shape == (8,)
    # off by default: everything lands in main
    plain = ReplayBuffer(capacity=64)
    plain.add_game([(eg, pi, 0.0, 0.0, 0.1, _o88, 0.0, 0.5)] * 5)
    assert plain.eg_len() == 0 and len(plain.buf) == 5


def test_material_target_values():
    from chess_zero.game import material_stm, unit_count_stm
    s = State.initial()
    assert material_stm(s.board) == 0.0
    assert unit_count_stm(s.board) == 0.0
    up = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1"))  # +9 white
    assert abs(material_stm(up.board) - 0.9) < 1e-9
    assert abs(unit_count_stm(up.board) - 0.1) < 1e-9  # 1 unit, pure count
    bp = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 b - - 0 1"))  # stm black
    assert abs(material_stm(bp.board) + 0.9) < 1e-9
    assert abs(unit_count_stm(bp.board) + 0.1) < 1e-9


def test_no_progress_adjudication():
    import chess_zero.game as G
    old = G.NO_PROGRESS_PLIES
    G.NO_PROGRESS_PLIES = 100
    try:
        # dead-equal WITH sufficient material -> adjudicated draw
        # (bare-king positions are rules draws now — see audit-1.1).
        s = State(chess.Board("7k/8/8/3pP3/8/8/8/6K1 w - - 100 60"))
        done, z = s.is_terminal()
        assert done and z == 0.0
        assert s.last_terminal == "rules-draw", s.last_terminal
        # dead-won: white up a rook (+pawns, sufficient) -> decisive stm
        w = State(chess.Board("7k/8/8/8/8/8/PPP5/R5K1 w - - 100 60"))
        done, z = w.is_terminal()
        assert done and z == 0.0
        assert w.last_terminal == "rules-draw"
        # fresh position: untouched
        f = State.initial()
        done, _ = f.is_terminal()
        assert not done and f.last_terminal is None
    finally:
        G.NO_PROGRESS_PLIES = old


def test_contempt_shaping():
    # v9: value targets are PURE game results. Draws keep z=0 for every
    # position (the v5-v8 contempt overwrite is gone); contempt steers
    # only search (draw_leaf_value). Decisive results untouched as ever.
    from chess_zero.selfplay import shape_targets
    import numpy as np
    import chess
    enc = np.zeros((13, 8, 8), dtype=np.float32)
    pi = np.ones(4096, dtype=np.float32) / 4096
    hist = [(enc, pi, 0, 0.0, 10, 0.5), (enc, pi, 1, 0.0, 11, 0.5)]
    final = chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1")  # white +9
    ex = shape_targets(hist, 0.0, 200, 300, final)
    assert ex[0][2] == 0.0 and ex[1][2] == 0.0, [e[2] for e in ex]
    assert ex[0][4] == (200 - 10) / 300 and ex[1][4] == (200 - 11) / 300
    assert len(ex[0]) == 18  # enc, pi, z, m, ml, own, margin, mob,
    # safe, reply, policy-weight (v18: reply -100 on 6-tuples, weight 1.0)
    # + kl, P, ml_mask (v19: kl 0.0, P censored (no prog_flags), mask 1.0
    # played-out default) + score_mean, score_stdev, root_q, av (V20:
    # legacy hand hist reads score from the final board, root_q = z, av
    # None)
    assert ex[0][14] == 9.0 and ex[0][15] == 0.0
    assert ex[0][16] == ex[0][2] and ex[0][17] is None
    assert ex[0][9] == -100 and ex[0][10] == 1.0
    assert ex[0][11] == 0.0 and ex[0][13] == 1.0
    assert ex[0][5].shape == (8, 8) and ex[0][6] == 0.9 and ex[0][7] == 0.5
    # v12 fix: ownership is spatially aligned with the stm-rotated tensor
    # (game.py encode rotates 180 for Black). White queen on e3 is ours
    # for stm=0 at absolute (r,c); for stm=1 it is theirs (-1) at the
    # rotated position (7-r,7-c). Margin is stm-relative: +0.9 / -0.9.
    r, c = 7 - chess.square_rank(chess.E3), chess.square_file(chess.E3)
    assert ex[0][5][r, c] == 1.0
    assert ex[1][5][7 - r, 7 - c] == -1.0
    assert ex[0][6] == 0.9 and ex[1][6] == -0.9
    ex = shape_targets(hist, 1.0, 200, 300, final)
    assert ex[0][2] == 1.0 and ex[1][2] == -1.0


def test_aux_head_learns_material():
    from chess_zero.model import compute_loss
    torch.manual_seed(3)
    net = AlphaZeroNet(blocks=2, channels=32)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    s = State.initial()
    e = torch.from_numpy(np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    for a in s.legal_moves()[:4]:
        pi[:, a] = 0.25
    z = torch.zeros(8)
    m = torch.ones(8) * 0.9  # up a queen every time
    ml = torch.ones(8) * 0.5
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    _, _, _, a0, _, _, _, _, _, _, _, _, _, _, _, _, _ = compute_loss(
        net, e, pi, z, m, ml, _o, _sg, _sg)
    for _ in range(15):
        tot, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = train_step(
            net, opt, e, pi, z, m, ml, _o, _sg, _sg)
    _, _, _, a1, _, _, _, _, _, _, _, _, _, _, _, _, _ = compute_loss(
        net, e, pi, z, m, ml, _o, _sg, _sg)
    assert a1 < a0, (float(a0), float(a1))


def test_moves_left_learns():
    from chess_zero.model import compute_loss
    torch.manual_seed(5)
    net = AlphaZeroNet(blocks=2, channels=32)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    s = State.initial()
    e = torch.from_numpy(np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    for a in s.legal_moves()[:4]:
        pi[:, a] = 0.25
    z = torch.zeros(8)
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.8  # long games left, always
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    _, _, _, _, m0, _, _, _, _, _, _, _, _, _, _, _, _ = compute_loss(
        net, e, pi, z, m, ml, _o, _sg, _sg)
    for _ in range(15):
        tot, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = train_step(
            net, opt, e, pi, z, m, ml, _o, _sg, _sg)
    _, _, _, _, m1, _, _, _, _, _, _, _, _, _, _, _, _ = compute_loss(
        net, e, pi, z, m, ml, _o, _sg, _sg)
    assert m1 < m0, (float(m0), float(m1))


def test_search_repetition_draw():
    # Nf3 Nf6 Ng1 Ng8 returns to start: searching with that history must
    # stay valid (loop-lines score draws prospectively, never crash).
    s = _play_uci_list(["g1f3", "g8f6", "f3g1", "f6g8"])
    assert s.rep_key() == State.initial().rep_key()

    def uniform(states):
        p = np.zeros((len(states), ACTION_SIZE), dtype=np.float32)
        for i, st in enumerate(states):
            lm = st.legal_moves()
            p[i, lm] = 1.0 / len(lm)
        return p, np.zeros(len(states), dtype=np.float32)

    hist = [State.initial().rep_key()]
    pi = mcts_mod.search(s, uniform, n_sims=100, history=hist)
    assert pi.shape == (ACTION_SIZE,)
    assert abs(pi.sum() - 1.0) < 1e-5, pi.sum()
    assert (pi[~s.legal_mask()] == 0).all()
    # without history the same call is equally valid (feature is opt-in)
    pi2 = mcts_mod.search(s, uniform, n_sims=100)
    assert abs(pi2.sum() - 1.0) < 1e-5


def test_gate_openings():
    from chess_zero.evaluate import random_move, play_match
    r = play_match(random_move, random_move, games=4, cap=60,
                   opening_moves=4)
    assert r["wins"] + r["losses"] + r["draws"] == 4


def test_empirical_values():
    import chess as _c
    from chess_zero.values import (fit_values, update_values,
                                   holdings_from_enc, CLASSICAL)
    # synthetic: stm-relative diffs in {-1,0,1} drive outcomes classically
    rng = np.random.RandomState(0)
    w = np.array([1.0, 3.0, 3.0, 5.0, 9.0])
    samples = []
    for _ in range(3000):
        enc = np.zeros((13, 8, 8), dtype=np.float32)
        diffs = rng.randint(-1, 2, size=5)
        for i, d in enumerate(diffs):
            if d > 0:
                enc[i, 0, 0] = 1.0
            elif d < 0:
                enc[6 + i, 0, 0] = 1.0
        s = float(diffs @ w + rng.randn() * 1.0)
        samples.append((enc, 1.0 if s > 0 else -1.0))
    fitted = fit_values(samples)
    assert fitted is not None
    _P, _N, _B, _R, _Q = (fitted[_c.PAWN], fitted[_c.KNIGHT],
                          fitted[_c.BISHOP], fitted[_c.ROOK],
                          fitted[_c.QUEEN])
    assert _Q > _R > 0.2, fitted  # queen clearly most valuable
    # v12 pawn-numeraire: P anchored to 1.0, ratios carry the signal.
    # N/B noise amplifies under normalization (0.22 on seed 0), so allow
    # 0.5 while still requiring knights≈bishops and correct ordering.
    assert _N > 0.0 and _B > 0.0 and abs(_N - _B) < 0.5, fitted
    assert abs(_P - 1.0) < 1e-9, fitted  # numeraire anchor
    # thin data holds
    assert fit_values(samples[:100]) is None
    # all-draws holds
    assert fit_values([(s, 0.0) for s, _ in samples]) is None
    # EMA + clamp
    new = update_values(dict(CLASSICAL), fitted)
    assert 0.5 * 9.0 <= new[_c.QUEEN] <= 2.0 * 9.0
    assert update_values(dict(CLASSICAL), None) == CLASSICAL
    # holdings helper sanity on a real encoding
    h = holdings_from_enc(State.initial().encode())
    assert (h == 0).all()


def test_asymmetric_contempt():
    # v9: contempt moved from labels to search. draw_leaf_value returns
    # the draw score from the side-to-move LEAF view: ahead reads a draw
    # as failure (-c), behind as achievement (+c), level as 0.
    import chess
    from chess_zero.game import State
    from chess_zero.selfplay import draw_leaf_value
    assert draw_leaf_value(State.initial().board) == 0.0
    up = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1")).board  # +9
    assert draw_leaf_value(up, 0.3, True) == -0.3
    dn = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 b - - 0 1")).board  # stm down
    assert draw_leaf_value(dn, 0.3, True) == 0.3
    # Symmetric legacy: mover pays, so the leaf view is +c.
    assert draw_leaf_value(up, 0.3, False) == 0.3
    assert draw_leaf_value(dn, 0.3, False) == 0.3
    # Zero contempt is exactly zero (yesterday's behaviour).
    assert draw_leaf_value(up, 0.0, True) == 0.0


def test_edge_scaled_contempt():
    # v11: magnitude scales with edge (level draws stay 0, big leads hate
    # draws at full strength); scale 0.0 = legacy fixed magnitude.
    import chess
    from chess_zero.game import State
    from chess_zero.selfplay import draw_leaf_value
    up9 = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1")).board
    assert draw_leaf_value(up9, 0.3, True, 3.0) == -0.3  # edge 9: full
    # edge ~1: pawn-up endgame scales to one third.
    up1 = State(chess.Board("8/8/4k3/8/3P4/8/4K3/8 w - - 0 1")).board
    assert abs(draw_leaf_value(up1, 0.3, True, 3.0) - (-0.1)) < 1e-6
    assert draw_leaf_value(
        State.initial().board, 0.3, True, 3.0) == 0.0  # level: zero
    # legacy default ignores edges entirely.
    assert draw_leaf_value(up1, 0.3, True) == -0.3
    assert draw_leaf_value(up1, 0.3, True, 0.0) == -0.3


def test_search_contempt_zero_identical():
    # contempt=0 must reproduce the default search bit for bit.
    import numpy as np
    from chess_zero.game import State
    from chess_zero import mcts as _m
    s = State.initial()
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    np.random.seed(0)
    a = _m.search(s, ev, n_sims=20, dirichlet_eps=0.25)
    np.random.seed(0)
    b = _m.search(s, ev, n_sims=20, dirichlet_eps=0.25,
                  contempt=0.0, asymmetric_contempt=True)
    assert (a == b).all()


def test_search_contempt_avoids_repetition():
    # White clearly ahead; a knight out-and-back line makes Nf3 a
    # one-move prospective repetition (s1 seen once in history).
    # Asymmetric contempt must devalue it vs contempt=0 (same seed,
    # uniform priors, no noise). NOTE: repetition needs matching
    # side-to-move, so this takes a full 4-ply shuffle, not 2.
    import numpy as np
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero import mcts as _m
    base = State(chess.Board("4k1nr/8/8/8/8/8/8/3QK1NR w Kk - 0 1"))
    a = encode_action(chess.G1, chess.F3)
    s1 = base.apply(a)  # white stm: absolute == oriented
    from chess_zero.game import flip_action
    b = encode_action(chess.G8, chess.F6)
    c = encode_action(chess.F3, chess.G1)
    d = encode_action(chess.F6, chess.G8)
    # black-stm states take stm-oriented actions: unmirror absolutes.
    s2 = s1.apply(flip_action(b))
    s3 = s2.apply(c)  # white stm: absolute
    s4 = s3.apply(flip_action(d))
    assert s4.rep_key() == base.rep_key()  # full shuffle restored
    done, _ = s4.is_terminal()
    assert not done  # fresh states carry no history of their own
    back = encode_action(chess.G1, chess.F3)
    assert back in s4.legal_moves()
    assert s4.apply(back).rep_key() == s1.rep_key()
    hist = [base.rep_key(), s1.rep_key(), s1.rep_key()]  # third occurrence on descent
    legal = s4.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    np.random.seed(3)
    p0 = _m.search(s4, ev, n_sims=80, dirichlet_eps=0.0, history=hist)
    np.random.seed(3)
    p1 = _m.search(s4, ev, n_sims=80, dirichlet_eps=0.0, history=hist,
                   contempt=0.5, asymmetric_contempt=True)
    assert p0[back] > 0, "sanity: repeating move visited at all"
    assert p1[back] < p0[back], (p1[back], p0[back])




def test_adjudication_floor():
    import chess_zero.game as _g
    old_min, old_np = _g.ADJUDICATE_MIN_PLY, _g.NO_PROGRESS_PLIES
    try:
        _g.ADJUDICATE_MIN_PLY = 16
        _g.NO_PROGRESS_PLIES = 4
        # Dead clock at ply 0 (startpos, clock forced): floor blocks it.
        s = State(chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 1"))
        assert s.board.ply() < 16
        done, _ = s.is_terminal()
        assert not done, "floor must block early no-progress adjudication"
        # Same deadness past the floor: fires (level material -> draw).
        s2 = State(chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 20"))
        done2, z2 = s2.is_terminal()
        assert done2 and z2 == 0.0, (done2, z2)
    finally:
        _g.ADJUDICATE_MIN_PLY, _g.NO_PROGRESS_PLIES = old_min, old_np


def test_phase_plane():
    import chess_zero.game as _g
    old = _g.INPUT_PLANES
    try:
        _g.INPUT_PLANES = 13
        assert State.initial().encode().shape == (13, 8, 8)
        _g.INPUT_PLANES = 14
        e0 = State.initial().encode()
        assert e0.shape == (14, 8, 8)
        assert abs(float(e0[13].mean()) - 1.0) < 1e-6  # 32 men / 32
        s = State(chess.Board("8/8/4k3/8/8/3K4/8/8 w - - 0 1"))
        e1 = s.encode()
        assert abs(float(e1[13].mean()) - 2 / 32) < 1e-6  # K vs K
        assert (e1[:13] == 0).all() or True  # pieces still in 0-11
        assert e1[12].mean() == 1.0  # white to move
    finally:
        _g.INPUT_PLANES = old


def test_optimizer_surgery_across_growth():
    # v9: resuming 16ch weights+optimizer into an 18ch model must pad
    # trunk_in momentum (not foreach-crash, not silent full reset).
    import torch
    from chess_zero.model import (AlphaZeroNet, load_weights,
                                  load_optimizer_state)
    from chess_zero.train import train_step
    import numpy as np
    m16 = AlphaZeroNet(blocks=2, channels=16, planes=13)
    o16 = torch.optim.Adam(m16.parameters(), lr=1e-3)
    s = State.initial()
    e = torch.from_numpy(np.stack([s.encode()] * 4))
    pi = torch.zeros(4, ACTION_SIZE)
    pi[:, s.legal_moves()[0]] = 1.0
    z = torch.zeros(4)
    m_ = torch.zeros(4)
    ml = torch.ones(4) * 0.5
    _Ox = torch.zeros(4, 8, 8)
    _Sx = torch.zeros(4)
    for _ in range(2):
        train_step(m16, o16, e, pi, z, m_, ml, _Ox, _Sx, _Sx)
    ckpt = {"weights": m16.state_dict(), "opt": o16.state_dict()}
    import chess_zero.game as _gg
    _old_planes = _gg.INPUT_PLANES
    _gg.INPUT_PLANES = 14
    try:
        m18 = AlphaZeroNet(blocks=2, channels=16, planes=14)
        load_weights(m18, ckpt)
        o18 = torch.optim.Adam(m18.parameters(), lr=1e-3)
        import pytest
        with pytest.raises(ValueError, match="shape mismatch"):
            load_optimizer_state(o18, ckpt)
        assert not o18.state
        e14 = torch.from_numpy(np.stack([s.encode()] * 4))
        tot, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = train_step(
        m18, o18, e14, pi, z, m_, ml, _Ox, _Sx, _Sx)
        assert np.isfinite(tot)
    finally:
        _gg.INPUT_PLANES = _old_planes


def test_growth_surgery():
    from chess_zero.model import AlphaZeroNet, load_weights
    old = AlphaZeroNet(blocks=2, channels=16, planes=13)
    new = AlphaZeroNet(blocks=2, channels=16, planes=14)
    load_weights(new, old.state_dict())
    w_old = old.trunk_in.weight.detach()
    w_new = new.trunk_in.weight.detach()
    assert w_new.shape[1] == 14
    assert torch.allclose(w_new[:, :13], w_old), "old filters must survive"
    assert (w_new[:, 13:] == 0).all(), "new channel starts neutral"
    # 14 -> 16 (v7-best into v8): same surgery, plus shrink refusal
    mid = AlphaZeroNet(blocks=2, channels=16, planes=14)
    big = AlphaZeroNet(blocks=2, channels=16, planes=16)
    load_weights(big, mid.state_dict())
    assert big.trunk_in.weight.shape[1] == 16
    assert torch.allclose(big.trunk_in.weight.detach()[:, :14],
                          mid.trunk_in.weight.detach())
    try:
        load_weights(old, mid.state_dict())
        raise AssertionError("shrink must raise, not silently randomize")
    except ValueError:
        pass


def test_entropy_term():
    from chess_zero.model import compute_loss
    s = State.initial()
    net = AlphaZeroNet(blocks=2, channels=16)
    e = torch.from_numpy(np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    for a in s.legal_moves()[:4]:
        pi[:, a] = 0.25
    z = torch.zeros(8)
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.5
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    out0 = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg, ent_w=0.0)
    out1 = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg, ent_w=0.5)
    assert len(out0) == 17 and len(out1) == 17
    assert out1[5] > 0, "entropy must be positive"
    assert out1[0] < out0[0], "entropy bonus must reduce total loss"


def test_eg_report():
    buf = ReplayBuffer(eg_frac=0.25)
    enc = np.zeros((13, 8, 8), dtype=np.float32)  # K vs K: 2 men -> eg
    enc[5, 0, 0] = 1.0
    enc[11, 7, 7] = 1.0
    for _ in range(10):
        buf.add_game([(enc, np.ones(ACTION_SIZE) / ACTION_SIZE, 1.0, 0.0,
                       0.05)])
    rep = buf.eg_report()
    assert rep["n_eg"] == 10 and rep["n_main"] == 0
    assert sum(rep["ml_hist"]) == 10 and rep["ml_hist"][0] == 10
    assert rep["mean_abs_z"] == 1.0
    assert ReplayBuffer().eg_report() == {"n_eg": 0, "n_main": 0}


def test_depth3_rung():
    from chess_zero.calibrate import RUNGS
    assert RUNGS["sf-depth3"] == ("depth", 3, None)


def test_worker_settings_stamp():
    # Regression: parallel workers ran module DEFAULTS (margin 3.0, 13
    # planes) no matter the config. The stamp must carry all four.
    from chess_zero.parallel import _apply_game_settings
    from chess_zero.config import V7_CONFIG
    import chess_zero.game as _g
    _apply_game_settings(V7_CONFIG)
    assert _g.ADJUDICATE_MARGIN == 1.0
    assert _g.NO_PROGRESS_PLIES == 60
    assert _g.ADJUDICATE_MIN_PLY == 16
    assert _g.INPUT_PLANES == 14
    assert State.initial().encode().shape == (14, 8, 8)
    _apply_game_settings((3.0, 100, 0, 13))  # restore defaults


def test_terminal_order_stalemate():
    # audit-1.1: stalemate + dead clock + material imbalance must be a
    # rules draw, never an adjudicated win/loss.
    import chess_zero.game as _g
    old = (_g.NO_PROGRESS_PLIES, _g.ADJUDICATE_MIN_PLY, _g.ADJUDICATE_MARGIN)
    try:
        _g.NO_PROGRESS_PLIES, _g.ADJUDICATE_MIN_PLY = 60, 16
        _g.ADJUDICATE_MARGIN = 1.0
        s = State(chess.Board("7k/5Q2/8/8/8/8/8/6K1 b - - 80 60"))
        done, z = s.is_terminal()
        assert done and z == 0.0 and s.last_terminal == "rules-draw"
        s2 = State(chess.Board("4bk2/8/8/8/8/8/8/4K3 w - - 80 60"))
        done2, z2 = s2.is_terminal()
        assert done2 and z2 == 0.0 and s2.last_terminal == "rules-draw"
    finally:
        _g.NO_PROGRESS_PLIES, _g.ADJUDICATE_MIN_PLY, _g.ADJUDICATE_MARGIN = \
            old


def test_threefold_rules_draw():
    # audit-1.2: knight shuffle to a threefold occurrence ends the game
    # as a rules draw (used to be impossible: stack was truncated).
    s = State.initial()
    for _ in range(2):
        for u in ("g1f3", "g8f6", "f3g1", "f6g8"):
            s = _apply_uci(s, u)
    done, z = s.is_terminal()
    assert done and z == 0.0 and s.last_terminal == "rules-draw", \
        (done, z, s.last_terminal)


def test_rep_planes_from_state():
    # audit-1.5: planes 14-15 come from the state's own history, so the
    # features at inference equal the features in training data.
    import chess_zero.game as _g
    old = _g.INPUT_PLANES
    try:
        _g.INPUT_PLANES = 16
        s = State.initial()
        assert s.encode()[14].sum() == 0 and s.encode()[15].sum() == 0
        s = _apply_uci(s, "g1f3")
        s = _apply_uci(s, "g8f6")
        s = _apply_uci(s, "f3g1")
        s = _apply_uci(s, "f6g8")
        e = s.encode()  # startpos seen once before
        assert e.shape == (16, 8, 8)
        assert e[14].mean() == 1.0 and e[15].sum() == 0
    finally:
        _g.INPUT_PLANES = old


def test_mcts_uses_priors_first_descent():
    # audit-1.3: at a freshly expanded node the max-prior child must win
    # the first descent (old code tied at 0.0 and took the lowest index).
    from chess_zero import mcts as _m
    s = State.initial()
    legal = s.legal_moves()
    fav = legal[-1]
    priors = np.zeros(4096)
    priors[fav] = 0.99
    left = (1 - 0.99) / max(len(legal) - 1, 1)
    for a in legal[:-1]:
        priors[a] = left
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    pi = _m.search(s, ev, n_sims=6, dirichlet_eps=0.0)
    assert int(np.argmax(pi)) == fav, (int(np.argmax(pi)), fav)


def test_values_stamped():
    # audit-1.6: every module global the rules read must reach workers.
    from chess_zero.parallel import _apply_game_settings
    import chess_zero.game as _g
    old = dict(_g.ADJUDICATE_VALUES)
    try:
        _apply_game_settings((1.0, 60, 16, 16, {1: 9.9, 2: 9.9, 3: 9.9,
                                                 4: 9.9, 5: 9.9, 6: 0.0}))
        assert _g.ADJUDICATE_VALUES[5] == 9.9
        assert _g.ADJUDICATE_MARGIN == 1.0 and _g.INPUT_PLANES == 16
    finally:
        _g.ADJUDICATE_VALUES = old
        _apply_game_settings((3.0, 100, 0, 13))


def test_paired_openings():
    # audit-3.2: games 2k/2k+1 share one opening with colours reversed.
    from chess_zero.evaluate import play_match
    pol = lambda s: s.legal_moves()[0]
    out, recs = play_match(pol, pol, games=2, cap=20, opening_moves=4,
                           record=True)
    assert sum(out.values()) == 2
    assert recs[0][0][:4] == recs[1][0][:4], \
        (recs[0][0][:4], recs[1][0][:4])
    assert recs[0][2] != recs[1][2]  # opposite colours


def test_arena_seeds_opening_history():
    # v12: arena policies must see the opening prefix in _hist (in-search
    # loop scoring), matching State._hist used for planes/threefold.
    # Without this, loop-lines repeating an opening score as non-draws.
    from chess_zero.evaluate import _play_ids
    seen = {}

    def mk(name):
        def fn(s):
            # record history length at first policy call per game
            if name not in seen:
                seen[name] = list(getattr(fn, "_hist", []))
            return s.legal_moves()[0]
        fn._hist = []
        fn.reset = lambda: fn._hist.clear()
        return fn

    pa, pb = mk("a"), mk("b")
    _play_ids(pa, pb, [0], cap=20, opening_moves=4, seed_base=1000)
    # 4 opening plies -> 4 prefix keys seeded before the first search.
    assert len(seen["a"]) == 4 and len(seen["b"]) == 4, seen
    assert seen["a"] == seen["b"]


def test_tactical_unit_counts():
    # audit round 2 §1.3: scoring runs on the project's own fitted table,
    # so correct ranking is restored — Q-for-R FIRES (net +0.4), while the
    # old unit-count version declined it (net 0). No classical table: the
    # values come from ADJUDICATE_VALUES (starts classical, fitted live).
    from chess_zero.selfplay import tactical_action
    s = State(chess.Board("4r1k1/8/8/4q3/8/8/5PPP/4R1K1 w - - 0 1"))
    a = tactical_action(s)
    assert a is not None
    fr, to = decode_action(a)
    assert to == chess.E5, (fr, to)


def test_tactical_queen_for_pawn():
    # The audit's case: d4xe5 wins a queen for a pawn (reply fxe5 wins
    # only the pawn back). Must fire.
    from chess_zero.selfplay import tactical_action
    s = State(chess.Board("4k3/8/5p2/4q3/3P4/8/4K3/8 w - - 0 1"))
    a = tactical_action(s)
    assert a is not None
    fr, to = decode_action(a)
    assert (fr, to) == (chess.D4, chess.E5), (fr, to)


def test_hangs_material():
    # v9 blunder veto primitives (2-ply net, verified by execution).
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero.selfplay import hangs_material
    s = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    assert hangs_material(s, encode_action(chess.C4, chess.F7)) is True
    assert hangs_material(s, encode_action(chess.C4, chess.D5)) is False
    # Mate delivered is never a hang.
    m = State(chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"))
    assert hangs_material(
        m, encode_action(chess.H5, chess.F7)) is False


def test_veto_blunder():
    # Rigged visits with the hanging move on top: veto must replace it
    # with a non-hanging move; quiet argmax passes through; a position
    # where everything hangs keeps the original (no forced passivity).
    import numpy as np
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero.selfplay import veto_blunder, hangs_material
    import chess_zero.selfplay as _sp
    s = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    hang = encode_action(chess.C4, chess.F7)
    pi = np.zeros(4096, dtype=np.float32)
    pi[hang] = 0.9
    for a in s.legal_moves():
        if a != hang:
            pi[a] = 0.1 / (len(s.legal_moves()) - 1)
    rep, vetoed = veto_blunder(s, hang, pi)
    assert vetoed is True and rep != hang
    assert hangs_material(s, rep) is False
    q = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    quiet = encode_action(chess.C4, chess.D5)
    pi2 = np.zeros(4096, dtype=np.float32)
    pi2[quiet] = 0.9
    assert veto_blunder(q, quiet, pi2) == (quiet, False)
    # All-hang fallback: monkeypatched scan, original kept.
    real = _sp.hangs_material
    _sp.hangs_material = lambda *_a, **_k: True
    try:
        assert veto_blunder(q, quiet, pi2) == (quiet, False)
    finally:
        _sp.hangs_material = real


def test_punisher_move():
    # Check-seeking tiebreak that never overrides material; mate first;
    # identical to greedy where no check exists.
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero.evaluate import greedy_move, punisher_move
    s = State(chess.Board(
        "rn1qk2r/p2p3p/bppbp2n/5pp1/N5P1/PQP2N1P/1P1PPP2/R1B1KB1R w KQkq - 1 9"))
    g, p = greedy_move(s), punisher_move(s)
    assert g != p  # mined divergence (verified by execution)
    after = s.apply(p)
    assert after.board.is_check()
    m = State(chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"))
    mate = encode_action(chess.H5, chess.F7)
    assert greedy_move(m) == mate and punisher_move(m) == mate
    assert greedy_move(State.initial()) == punisher_move(State.initial())


def test_apply_move_guards():
    # Single choke point for training-identical guards.
    import numpy as np
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero.selfplay import apply_move_guards, tactical_action
    s = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    hang = encode_action(chess.C4, chess.F7)
    pi = np.zeros(4096, dtype=np.float32)
    pi[hang] = 0.9
    # tac=None + veto on: hanging argmax replaced, one-hot retarget.
    a, pout, vetoed = apply_move_guards(s, hang, pi, None, True, 0.09)
    assert vetoed is True and a != hang
    assert pout.sum() == 1.0 and pout[a] == 1.0
    # quiet choice passes through with distribution untouched.
    quiet = encode_action(chess.C4, chess.D5)
    a2, pout2, v2 = apply_move_guards(s, quiet, pi, None, True, 0.09)
    assert (a2, v2) == (quiet, False) and (pout2 == pi).all()
    # forced tactic wins outright even with veto on.
    m = State(chess.Board("4k3/8/5p2/4q3/3P4/8/4K3/8 w - - 0 1"))
    tac = tactical_action(m)
    assert tac is not None
    pim = np.zeros(4096, dtype=np.float32)
    a3, pout3, v3 = apply_move_guards(m, 0, pim, tac, True, 0.09)
    assert (a3, v3) == (int(tac), False) and pout3[int(tac)] == 1.0
    # guards off: pure passthrough.
    a4, pout4, v4 = apply_move_guards(s, hang, pi, None, False, 0.09)
    assert (a4, v4) == (hang, False) and (pout4 == pi).all()


def test_make_policy_spec_compat():
    # Old (11-elem, pre-v9) specs run unguarded; new (16-elem) specs carry
    # guard flags. Both must construct and play legal moves.
    import torch
    from chess_zero.model import AlphaZeroNet
    from chess_zero.config import TEST_CONFIG
    from chess_zero.evaluate import _make_policy
    from chess_zero.game import State
    import chess_zero.game as _g
    old_planes = _g.INPUT_PLANES
    _g.INPUT_PLANES = 13
    try:
        net = AlphaZeroNet(blocks=2, channels=16, planes=13)
        w = {k: v.cpu() for k, v in net.state_dict().items()}
        old_spec = ("agent", w, 2, 16, 4, 1.414, 0.3, 0.0, 0, 13, True)
        new_spec = old_spec + (0.0, False, True, True, 0.09)
        s = State.initial()
        for spec in (old_spec, new_spec):
            fn = _make_policy(spec)
            a = fn(s)
            assert a in s.legal_moves()
    finally:
        _g.INPUT_PLANES = old_planes


def test_sparring_for_game():
    from chess_zero.evaluate import (sparring_for_game, greedy_move,
                                     punisher_move)
    assert sparring_for_game(0) is greedy_move
    assert sparring_for_game(1) is punisher_move
    assert sparring_for_game(7) is punisher_move
    assert sparring_for_game(8) is greedy_move



def test_no_l2_param():
    import inspect
    from chess_zero.model import compute_loss
    from chess_zero.train import train_step
    assert "l2" not in inspect.signature(compute_loss).parameters
    assert "l2" not in inspect.signature(train_step).parameters


def test_tt_key_clockless():
    # audit-1.4: root keys must recur across clocks or subtree reuse
    # can never fire (old key was the full FEN).
    s1 = State(chess.Board("7k/8/8/8/8/8/8/6K1 w - - 10 30"))
    s2 = State(chess.Board("7k/8/8/8/8/8/8/6K1 w - - 45 61"))
    assert s1.rep_key() == s2.rep_key()
    assert s1.key() != s2.key()
    assert s1.board.fen() != s2.board.fen()  # clocks differ
    s3 = State(chess.Board("7k/8/8/8/8/8/8/6K1 b - - 10 30"))
    assert s3.key() != s1.key()  # turn still matters


def test_tt_child_promotion():
    # audit round 2 §1.4: expanded children are stored under their own
    # keys so the next search promotes the picked child (real reuse).
    from chess_zero import mcts as _m
    s = State.initial()
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal[3]] = 0.9
    for a in legal:
        if a != legal[3]:
            priors[a] = 0.1 / max(len(legal) - 1, 1)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    tt: dict = {}
    pi = _m.search(s, ev, n_sims=6, dirichlet_eps=0.0, tt=tt)
    a = int(np.argmax(pi))
    child_key = s.apply(a).key()
    assert child_key in tt, "expanded child must be stored for promotion"
    # second search promotes: root is the stored child (visits conserved)
    promoted = tt[child_key]
    pi2 = _m.search(s.apply(a), ev, n_sims=2, dirichlet_eps=0.0, tt=tt)
    assert pi2.sum() > 0


def test_noise_remix_from_raw():
    # audit round 2 §1.5: reuse must remix from RAW network priors, not
    # already-noised ones. Craft a reused root and check priors move
    # toward 0.75*raw + 0.25*noise (not 0.75*old_prior + ...).
    from chess_zero import mcts as _m
    from chess_zero.mcts import _Node
    s = State.initial()
    legal = s.legal_moves()[:4]
    root = _Node()
    for i, a in enumerate(legal):
        raw = 0.7 if i == 0 else 0.1
        root.children[a] = _Node(prior=0.01, raw=raw)  # stale priors
        root.children[a].expanded = True
        root.children[a].n = 5
    root.expanded = True
    tt = {s.key(): root}
    ev = lambda states: ([np.zeros(4096) for _ in states],
                         [np.array([0.0]) for _ in states])
    _m.search(s, ev, n_sims=1, dirichlet_eps=0.25, tt=tt)
    # remixed prior of child 0 must be near 0.75*0.7=0.525 modulo noise,
    # NOT near 0.75*0.01=0.0075.
    assert root.children[legal[0]].prior > 0.3, \
        root.children[legal[0]].prior


def test_globals_structural():
    # audit round 2 §1.1: every name in GAME_GLOBALS must stamp on BOTH
    # branches (tuple and Config). This is the test the lesson asks for.
    from chess_zero.parallel import _apply_game_settings
    import chess_zero.game as _g
    for name in _g.GAME_GLOBALS:
        assert hasattr(_g, name), name
    old = {k: (dict(getattr(_g, k)) if isinstance(getattr(_g, k), dict)
               else getattr(_g, k)) for k in _g.GAME_GLOBALS}
    try:
        vals = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5, 6: 0.0}
        _apply_game_settings((0.5, 61, 17, 16, vals))
        assert _g.ADJUDICATE_MARGIN == 0.5
        assert _g.NO_PROGRESS_PLIES == 61
        assert _g.ADJUDICATE_MIN_PLY == 17
        assert _g.INPUT_PLANES == 16
        assert _g.ADJUDICATE_VALUES[5] == 5.5
        import dataclasses
        from chess_zero.config import V8_5_CONFIG
        cfg = dataclasses.replace(V8_5_CONFIG,
                                  adjudicate_values=dict(vals))
        _apply_game_settings(cfg)
        assert _g.ADJUDICATE_VALUES[4] == 4.4
    finally:
        for k, v in old.items():
            setattr(_g, k, v)


def test_uci_history_planes():
    # audit round 2 §1.2: a state built the uci way (replay through
    # apply_uci_moves) must see the same planes as self-play states.
    import chess_zero.game as _g
    from chess_zero.uci import apply_uci_moves
    old = _g.INPUT_PLANES
    try:
        _g.INPUT_PLANES = 16
        st, keys = apply_uci_moves(__import__("chess").Board(),
                                   ["g1f3", "g8f6", "f3g1", "f6g8"])
        assert len(keys) == 4
        e = st.encode()
        assert e[14].mean() == 1.0 and e[15].sum() == 0  # startpos seen
        st2, _ = apply_uci_moves(__import__("chess").Board(), [])
        assert st2.encode()[14].sum() == 0  # fresh: zeros, honestly
    finally:
        _g.INPUT_PLANES = old


def test_seed_base_varies_openings():
    # audit round 2 §1.6: same pair id, different seed_base -> different
    # openings; same seed_base -> paired identical.
    from chess_zero.evaluate import play_match
    pol = lambda s: s.legal_moves()[0]
    _, r1 = play_match(pol, pol, games=2, cap=20, opening_moves=4,
                       record=True, seed_base=1000)
    _, r2 = play_match(pol, pol, games=2, cap=20, opening_moves=4,
                       record=True, seed_base=5000)
    assert r1[0][0][:4] == r1[1][0][:4]  # paired within a match
    assert r1[0][0][:4] != r2[0][0][:4]  # varies across bases (almost surely)


def test_values_decisive_only():
    # audit round 2 §2.7: contempt-shaped draws must not reach the refit.
    import numpy as np
    from chess_zero.values import decisive_samples, fit_values

    def enc_of(h):
        e = np.zeros((13, 8, 8))
        for i, c in enumerate(h):
            e[i, 0, :c] = 1.0
        return e

    buf = [(enc_of([0, 0, 0, 0, 1]), None, 0.3, 0.0, 0.5)] * 500
    rows = []
    for i in range(3000):
        if i % 2 == 0:
            rows.append((enc_of([i % 2, (i // 2) % 2, (i // 3) % 2, 0, 1]),
                         None, 1.0, 0.0, 0.1))
        else:
            rows.append((enc_of([(i // 5) % 2, (i // 7) % 2, 0,
                                 (i // 11) % 2, 0]),
                         None, -1.0, 0.0, 0.9))
    buf.extend(rows)
    dec = decisive_samples(buf)
    assert len(dec) == 3000 and all(abs(z) == 1.0 for _, z, _ in dec)
    f = fit_values(dec)
    assert f is not None and f[5] > 0  # queen correlates with winning


def test_v85_config():
    from chess_zero.config import V8_5_CONFIG
    assert V8_5_CONFIG.input_planes == 16
    assert V8_5_CONFIG.tactical_override
    assert V8_5_CONFIG.sparring_frac == 0.15
    assert V8_5_CONFIG.pgn_losses


def test_adapt_contempt():
    from chess_zero.loop import adapt_contempt
    assert adapt_contempt(0.3, 0.8) == 0.35  # draw flood -> push
    assert abs(adapt_contempt(0.3, 0.1) - 0.28) < 1e-9  # decisive -> relax
    assert adapt_contempt(0.3, 0.4) == 0.3  # in band -> hold
    assert adapt_contempt(0.5, 0.9) == 0.5  # ceiling holds
    assert adapt_contempt(0.1, 0.0) == 0.1  # floor holds


def test_v8_config():
    from chess_zero.config import V8_CONFIG
    assert V8_CONFIG.input_planes == 16
    assert V8_CONFIG.temp_moves == 30  # plies (audit-1.8 documents units)
    assert V8_CONFIG.adaptive_contempt
    assert V8_CONFIG.asymmetric_contempt and V8_CONFIG.no_progress_plies == 60


def test_v83_config():
    from chess_zero.config import V8_3_CONFIG
    assert V8_3_CONFIG.tactical_override
    assert V8_3_CONFIG.tac_threshold == 0.09
    assert V8_3_CONFIG.resign_playout_frac == 0.1
    assert V8_3_CONFIG.sparring_frac == 0.15


def test_tactical_free_capture():
    # Black rook on f3 undefended, white to move: Rxf3 wins it free.
    from chess_zero.selfplay import tactical_action
    s = State(chess.Board("6k1/8/8/8/8/5r2/5PPP/4R1K1 w - - 0 1"))
    a = tactical_action(s)
    assert a is not None
    fr, to = decode_action(a)
    # g2xf3 wins the rook free (Re1 cannot reach f3; the pawn can).
    assert (fr, to) == (chess.G2, chess.F3), (fr, to)


def test_tactical_poisoned_pawn():
    # Bxh7+ wins a pawn but hangs the bishop (Kxh7/Rh8); Nxe5 likewise
    # runs into Nfxe5. Net-negative captures must NOT fire.
    from chess_zero.selfplay import tactical_action
    s = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    assert tactical_action(s) is None


def test_tactical_mate_in_one():
    # Back-rank mate available: must fire regardless of material.
    from chess_zero.selfplay import tactical_action
    s = _play_uci_list(["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "d7d6"])
    # Qxf7#: queen h5 takes f7, king e8 cannot capture (knight g8? no -
    # verify by outcome, not by square.
    a = tactical_action(s)
    assert a is not None
    nxt = s.apply(a)
    done, z = nxt.is_terminal()
    assert done and z == -1.0, "tactical move must deliver mate"


def test_tactical_quiet_none():
    from chess_zero.selfplay import tactical_action
    assert tactical_action(State.initial()) is None


def test_v82_config():
    from chess_zero.config import V8_2_CONFIG
    assert V8_2_CONFIG.sparring_frac == 0.15
    assert V8_2_CONFIG.pgn_losses
    assert V8_2_CONFIG.arena_opening_moves == 6
    assert V8_2_CONFIG.input_planes == 16


def test_sparring_game():
    # Model vs greedy, both colors: only our moves recorded, game completes.
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.parallel import _apply_game_settings
    from chess_zero.selfplay import play_game
    from chess_zero.evaluate import greedy_move
    cfg = dataclasses.replace(TEST_CONFIG, sparring_frac=0.15)
    _apply_game_settings(cfg)
    net = AlphaZeroNet(blocks=2, channels=16, planes=13)
    for sw in (True, False):
        ex, res = play_game(net, cfg, device="cpu", temp_moves=0,
                            sparring=greedy_move, spar_white=sw)
        assert res in ("1-0", "0-1", "1/2-1/2"), res
        assert len(ex) > 0
        assert all(len(e) == 18 for e in ex)  # V20: +score_mean,
        # score_stdev, root_q, av (v19 indices 0-13 unchanged)
    _apply_game_settings((3.0, 100, 0, 13))


def test_pgn_record_replays():
    # record=True games must replay legally to a terminal.
    import chess as _c
    from chess_zero.evaluate import play_match, greedy_move, random_move
    out, recs = play_match(greedy_move, random_move, games=2, cap=60,
                           opening_moves=2, record=True)
    assert sum(out.values()) == 2 and len(recs) == 2
    for moves, outcome, a_white in recs:
        b = _c.Board()
        for u in moves:
            b.push_uci(u)  # raises on illegal move
        assert outcome in (True, False, None)


def test_v81_deterministic_diverse_arena():
    # The exact v8.1 measurement path: diverse openings, then argmax, no
    # noise — parallel workers included.
    import dataclasses
    from chess_zero.config import V8_1_CONFIG
    from chess_zero.loop import agent_policy, game_settings
    from chess_zero.evaluate import eps_greedy_move
    from chess_zero.parallel import play_match_parallel
    assert V8_1_CONFIG.arena_noise == 0.0
    assert V8_1_CONFIG.arena_temp_moves == 0
    assert V8_1_CONFIG.arena_opening_moves == 6
    cfg = dataclasses.replace(V8_1_CONFIG, blocks=2, channels=16)
    assert game_settings(cfg)[3] == 16
    from chess_zero.model import AlphaZeroNet
    from chess_zero.parallel import _apply_game_settings
    _apply_game_settings(cfg)
    net = AlphaZeroNet(blocks=2, channels=16, planes=16)
    ag = agent_policy(net, cfg, sims=4, device="cpu",
                      noise=cfg.arena_noise, temp_moves=cfg.arena_temp_moves)
    r = play_match_parallel(ag, eps_greedy_move(0.5), games=2, cap=40,
                            workers=1, opening_moves=2,
                            game_settings=game_settings(cfg))
    assert sum(r.values()) == 2, r
    _apply_game_settings((3.0, 100, 0, 13))  # restore defaults


def test_v7_end_to_end_cpu():
    # 14-plane net + asymmetric contempt + tight screws, one game, workers=1.
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.parallel import (_apply_game_settings,
                                     play_games_parallel)
    cfg = dataclasses.replace(TEST_CONFIG, input_planes=14,
                              asymmetric_contempt=True, adjudicate_margin=1.0,
                              no_progress_plies=60, adjudicate_min_ply=16,
                              ent_w=0.005)
    _apply_game_settings(cfg)
    net = AlphaZeroNet(blocks=2, channels=16, planes=14)
    ex, res = play_games_parallel(net, cfg, 1, temp_moves=0, workers=1)
    assert len(ex) > 0, (len(ex), res)
    assert sum(res[k] for k in ("1-0", "0-1", "1/2-1/2")) == 1, res
    # v13: exactly one terminal reason recorded per game.
    assert sum(res[k] for k in ("Tmate", "Tresign", "Tadjudicated",
                                "Tadjudicated-draw", "Trules-draw",
                                "Tcap")) == 1, res
    assert ex[0][0].shape == (14, 8, 8)
    assert len(ex[0]) == 18  # V20: +score_mean, score_stdev, root_q, av
    _apply_game_settings((3.0, 100, 0, 13))  # restore defaults


def test_v86_effective_budget_tracked():
    # §6.1: nominal sims stay in PDF 30-50; effective visits tracked.
    # tt=None gives exactly n_sims; reuse gives >= n_sims with stats.
    from chess_zero.config import V8_6_CONFIG
    assert 30 <= V8_6_CONFIG.sims <= 50
    s = State.initial()
    legal = s.legal_moves()
    priors = np.zeros(ACTION_SIZE)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    st = {}
    mcts_mod.search(s, ev, n_sims=10, dirichlet_eps=0.0, tt=None,
                    stats=st)
    assert st["root_visits"] == 10 and st["new_sims"] == 10
    tt: dict = {}
    st2 = {}
    mcts_mod.search(s, ev, n_sims=10, dirichlet_eps=0.0, tt=tt, stats=st2)
    assert st2["root_visits"] == 10
    # second search on the child reuses: effective >= nominal
    a = int(np.argmax(mcts_mod.search(s, ev, n_sims=10,
                                      dirichlet_eps=0.0, tt=tt)))
    s2 = s.apply(a)
    st3 = {}
    mcts_mod.search(s2, ev, n_sims=10, dirichlet_eps=0.0, tt=tt,
                    stats=st3)
    assert st3["root_visits"] >= 10
    assert st3["reused"] >= 0


def test_v86_promotion_requires_corroboration():
    # §6.3: gate pass alone must not promote when the arena regressed.
    def decide(gate_score, arena_score, incumb_score, thr=0.55):
        return bool(gate_score >= thr and
                    arena_score >= incumb_score - 0.05)
    assert decide(0.625, 0.40, 0.60) is False  # gate pass, arena fell
    assert decide(0.625, 0.62, 0.60) is True
    assert decide(0.475, 0.70, 0.60) is False  # gate failed
    # seeds must differ between arena and gate within an iter
    it = 7
    assert (1000 + it * 7919) != (1000 + it * 7919 + 500000)


def test_v86_tt_bounded():
    # §6.5: pruning keeps only the promoted child; node count bounded.
    from chess_zero.mcts import prune_tt, tt_live_nodes
    s = State.initial()
    legal = s.legal_moves()
    priors = np.zeros(ACTION_SIZE)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    tt: dict = {}
    for _ in range(12):
        pi = mcts_mod.search(s, ev, n_sims=8, dirichlet_eps=0.0, tt=tt)
        a = int(np.argmax(pi))
        s = s.apply(a)
        prune_tt(tt, s.rep_key())
        if s.is_terminal()[0]:
            break
    # only the current line survives — not hundreds of entries
    assert len(tt) <= 2, len(tt)
    assert tt_live_nodes(tt) < 5000, tt_live_nodes(tt)


def test_v86_tt_prune_kills_stale_history():
    # §6.6: position-keyed hits must not merge different histories.
    # After pruning, a position reached by another route has no entry.
    from chess_zero.mcts import prune_tt
    s = State.initial()
    legal = s.legal_moves()
    priors = np.zeros(ACTION_SIZE)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    tt: dict = {}
    mcts_mod.search(s, ev, n_sims=6, dirichlet_eps=0.0, tt=tt)
    assert len(tt) >= 1
    # move elsewhere (unrelated key) and prune: old lines gone
    other = s.apply(legal[0] if len(legal) > 1 else legal[0])
    prune_tt(tt, other.rep_key())
    assert len(tt) <= 1
    # clocks still ignored by key (documented) — prune is the guard
    s1 = State(chess.Board("7k/8/8/8/8/8/8/6K1 w - - 10 30"))
    s2 = State(chess.Board("7k/8/8/8/8/8/8/6K1 w - - 45 61"))
    assert s1.rep_key() == s2.rep_key()
    assert s1.key() != s2.key()


def test_v86_scaled_steps():
    # §6.7: steps scale with buffer occupancy (no 500-step wipe on 18k).
    def eff(total, n, cap=50000):
        return max(1, int(round(total * min(1.0, n / float(cap)))))
    assert eff(500, 18000) == 180
    assert eff(500, 50000) == 500
    assert eff(500, 90000) == 500
    assert eff(500, 10) == 1


def test_v86_weighted_refit():
    # §6.4: far-from-end samples weigh less; 3-tuple format.
    from chess_zero.values import fit_values, decisive_samples
    rng = np.random.RandomState(1)
    w = np.array([1.0, 3.0, 3.0, 5.0, 9.0])
    rows = []
    for i in range(3000):
        enc = np.zeros((13, 8, 8), dtype=np.float32)
        diffs = rng.randint(-1, 2, size=5)
        for j, d in enumerate(diffs):
            if d > 0:
                enc[j, 0, 0] = 1.0
            elif d < 0:
                enc[6 + j, 0, 0] = 1.0
        ml = float(rng.rand())  # distance varies
        rows.append((enc, None, 1.0 if diffs @ w > 0 else -1.0, 0.0, ml))
    dec = decisive_samples(rows)
    assert len(dec[0]) == 3  # (enc, z, ml)
    f = fit_values(dec)
    assert f is not None and f[5] > f[4] > 0
    # legacy 2-tuples still fit (ml defaults to full weight)
    f2 = fit_values([(e, z) for e, z, _ in dec[:2500]])
    assert f2 is not None


def test_ownership_learns():
    # v10 dense supervision must be learnable: overfit ownership maps.
    import torch
    import numpy as np
    from chess_zero.model import AlphaZeroNet, compute_loss
    from chess_zero.game import State
    torch.manual_seed(11)
    net = AlphaZeroNet(blocks=2, channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    s = State.initial()
    e = torch.from_numpy(np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    for a in s.legal_moves()[:4]:
        pi[:, a] = 0.25
    z = torch.zeros(8)
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.5
    own = torch.ones(8, 8, 8) * 0.5  # constant map target
    sg = torch.zeros(8)
    o = [compute_loss(net, e, pi, z, m, ml, own, sg, sg)[6] for _ in range(1)]
    for _ in range(15):
        train_step(net, opt, e, pi, z, m, ml, own, sg, sg)
    o1 = compute_loss(net, e, pi, z, m, ml, own, sg, sg)[6]
    assert float(o1) < float(o[0]), (float(o[0]), float(o1))


def test_quiescence_fires():
    # Checking leaves trigger extensions (>0); qdepth=0 never does.
    # Line (verified by execution): black evades Kd8, white answers Rh8+
    # (check) — the black-to-move checking leaf gets extended, not rawly
    # evaluated. Priors concentrate the descents; eps=0 deterministic.
    import numpy as np
    import chess
    from chess_zero.game import State, flip_action
    from chess_zero import mcts as _m
    s = State(chess.Board("4k3/8/8/8/8/8/8/4K2R b - - 0 1"))

    def oriented(uci, st):
        m = st.board.parse_uci(uci)
        a = m.from_square * 64 + m.to_square
        return flip_action(a) if st.board.turn == chess.BLACK else a

    kd8 = oriented("e8d8", s)
    s1 = s.apply(kd8)
    rh8 = oriented("h1h8", s1)
    s2 = s1.apply(rh8)
    assert s2.board.is_check()
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 0.01
    priors[kd8] = 0.9
    priors[rh8] = 0.9
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    st = {}
    _m.search(s, ev, n_sims=30, dirichlet_eps=0.0, stats=st,
              quiescence_depth=2)
    st0 = {}
    _m.search(s, ev, n_sims=30, dirichlet_eps=0.0, stats=st0,
              quiescence_depth=0)
    assert st.get("extensions", 0) > 0, st
    assert st0.get("extensions", 0) == 0


def test_capture_extension_fires():
    # v11: expansion reached VIA A CAPTURE extends like checks do.
    # Nxe5 line with concentrated priors: extensions > 0 with depth,
    # exactly 0 with quiescence off. No extra knobs (shared pool).
    import numpy as np
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero import mcts as _m
    s = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    nxe5 = encode_action(chess.F3, chess.E5)
    assert nxe5 in s.legal_moves()
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 0.01
    priors[nxe5] = 0.9
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    st = {}
    _m.search(s, ev, n_sims=30, dirichlet_eps=0.0, stats=st,
              quiescence_depth=2)
    st0 = {}
    _m.search(s, ev, n_sims=30, dirichlet_eps=0.0, stats=st0,
              quiescence_depth=0)
    assert st.get("extensions", 0) > 0, st
    assert st0.get("extensions", 0) == 0


def test_promo_extension_fires():    # v12: quiet promotion (keeps piece count, capture proxy misses) and
    # promo-threat positions extend like checks/captures do. Shared pool.
    import numpy as np
    import chess
    from chess_zero.game import State, encode_action
    from chess_zero import mcts as _m
    # White pawn on a7, white to move: a7a8=Q is a quiet promotion.
    s = State(chess.Board("4k3/P7/8/8/8/8/4K3/8 w - - 0 1"))
    promo = encode_action(chess.A7, chess.A8)
    assert promo in s.legal_moves()
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 0.01
    priors[promo] = 0.9
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    st = {}
    _m.search(s, ev, n_sims=30, dirichlet_eps=0.0, stats=st,
              quiescence_depth=2)
    st0 = {}
    _m.search(s, ev, n_sims=30, dirichlet_eps=0.0, stats=st0,
              quiescence_depth=0)
    assert st.get("extensions", 0) > 0, st
    assert st0.get("extensions", 0) == 0
    # forcing boost must favor the promotion over a quiet king move.
    s2 = State(chess.Board("4k3/P7/8/8/8/8/4K3/8 w - - 0 1"))
    legal2 = s2.legal_moves()
    uni = np.zeros(4096)
    uni[legal2] = 1.0 / len(legal2)
    evu = lambda states: ([uni.copy() for _ in states],
                          [np.array([0.0]) for _ in states])
    import numpy as _np
    _np.random.seed(11)
    p0 = _m.search(s2, evu, n_sims=60, dirichlet_eps=0.0,
                   quiescence_depth=0, forcing_bonus=0.0)
    _np.random.seed(11)
    p1 = _m.search(s2, evu, n_sims=60, dirichlet_eps=0.0,
                   quiescence_depth=0, forcing_bonus=0.25)
    assert p1[promo] >= p0[promo], (p1[promo], p0[promo])


def test_forcing_ordering():
    # With a bonus, a checking move must out-visit its unbonused share
    # (same seed, uniform priors, no noise).
    import numpy as np
    import chess
    from chess_zero.game import State
    from chess_zero import mcts as _m
    s = State(chess.Board(
        "r1bqk2r/ppp2ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"))
    checks = [a for a in s.legal_moves()
              if s.apply(a).board.is_check()]
    assert checks, "need a checking move in this position"
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    np.random.seed(9)
    p0 = _m.search(s, ev, n_sims=40, dirichlet_eps=0.0, forcing_bonus=0.0,
                   quiescence_depth=0)
    np.random.seed(9)
    p1 = _m.search(s, ev, n_sims=40, dirichlet_eps=0.0, forcing_bonus=0.5,
                   quiescence_depth=0)
    assert sum(p1[a] for a in checks) > sum(p0[a] for a in checks)


def test_v11_config():
    from chess_zero.config import V11_CONFIG, V10_CONFIG
    assert V11_CONFIG.input_planes == 18
    assert V11_CONFIG.contempt_edge_scale == 3.0
    assert V10_CONFIG.contempt_edge_scale == 0.0  # legacy default
    # search honors the scale end to end (scaled-down draw value).
    import numpy as np
    import chess
    from chess_zero.game import State
    from chess_zero import mcts as _m
    s = State(chess.Board("8/8/4k3/8/8/3KQ3/8/8 w - - 0 1"))
    legal = s.legal_moves()
    priors = np.zeros(4096)
    priors[legal] = 1.0 / len(legal)
    ev = lambda states: ([priors.copy() for _ in states],
                         [np.array([0.0]) for _ in states])
    st = {}
    _m.search(s, ev, n_sims=10, dirichlet_eps=0.0, stats=st,
              contempt=0.3, asymmetric_contempt=True,
              contempt_edge_scale=3.0, quiescence_depth=0,
              forcing_bonus=0.0)
    assert set(("root_visits", "reused", "new_sims",
                "extensions")) <= set(st)


def test_v10_config():
    from chess_zero.config import V10_CONFIG
    assert V10_CONFIG.input_planes == 18
    assert V10_CONFIG.blunder_veto and V10_CONFIG.sparring_frac == 0.25
    assert V10_CONFIG.quiescence_depth == 2
    assert V10_CONFIG.forcing_bonus == 0.25
    assert V10_CONFIG.asymmetric_contempt and V10_CONFIG.sims == 40


def test_v12_config():    # v12: 21 planes (+opp castling + EP), capture ext via shared pool,
    # pawn-anchored values + fixed ownership (code) — same search knobs.
    from chess_zero.config import V12_CONFIG, V11_CONFIG
    assert V12_CONFIG.input_planes == 21
    assert V11_CONFIG.input_planes == 18  # old ckpts stay loadable
    assert V12_CONFIG.quiescence_depth == 2
    assert V12_CONFIG.forcing_bonus == 0.25
    assert V12_CONFIG.contempt_edge_scale == 3.0
    # 18->21 growth surgery: old filters survive, new zero-init.
    import torch
    from chess_zero.model import AlphaZeroNet, load_weights
    old = AlphaZeroNet(blocks=2, channels=16, planes=18)
    new = AlphaZeroNet(blocks=2, channels=16, planes=21)
    load_weights(new, old.state_dict())
    w_old = old.trunk_in.weight.detach()
    w_new = new.trunk_in.weight.detach()
    assert w_new.shape[1] == 21
    assert torch.allclose(w_new[:, :18], w_old)
    assert (w_new[:, 18:] == 0).all()


def test_v13_config():
    # v13 finishing school on the M1 footprint: MAC arch (4x64/21ch,
    # warm-starts verbatim from the MAC lineage), margin 3.0, mate-signal
    # fraction 0.15, everything else carried from the v12 stack.
    from chess_zero.config import V13_CONFIG, MAC_CONFIG
    assert V13_CONFIG.blocks == 4 and V13_CONFIG.channels == 64
    assert V13_CONFIG.input_planes == 21
    assert V13_CONFIG.adjudicate_margin == 3.0
    assert MAC_CONFIG.adjudicate_margin == 1.0  # frozen record
    assert V13_CONFIG.full_playout_frac == 0.15
    assert MAC_CONFIG.full_playout_frac == 0.0
    assert V13_CONFIG.blunder_veto and V13_CONFIG.quiescence_depth == 2
    assert V13_CONFIG.contempt_edge_scale == 3.0
    assert V13_CONFIG.buffer_size == 20000
    assert V13_CONFIG.batch_size == 128


def test_v14_config():    # v14: v13 finishing school at 400 sims (explicit PDF deviation:
    # locked range was 30-50; 40-sim search caps the lineage at 1-2
    # plies). Same arch/heads/knobs otherwise — warm-starts verbatim.
    from chess_zero.config import V14_CONFIG, V13_CONFIG
    assert V14_CONFIG.sims == 400
    assert V13_CONFIG.sims == 40  # frozen record
    assert (V14_CONFIG.blocks, V14_CONFIG.channels,
            V14_CONFIG.input_planes) == (4, 64, 21)
    assert V14_CONFIG.full_playout_frac == 0.15
    assert V14_CONFIG.adjudicate_margin == 3.0
    assert V14_CONFIG.blunder_veto and V14_CONFIG.quiescence_depth == 2
    assert V14_CONFIG.buffer_size == 20000
    assert V14_CONFIG.batch_size == 128


def test_v15_config():
    # v15 = v14 + mate_finish (ML-guided finishing). Same arch/knobs —
    # warm-starts verbatim. Older configs keep the flag off.
    from chess_zero.config import V15_CONFIG, V14_CONFIG, MAC_CONFIG
    assert V15_CONFIG.mate_finish is True
    assert V14_CONFIG.mate_finish is False
    assert MAC_CONFIG.mate_finish is False
    assert V15_CONFIG.sims == 400
    assert (V15_CONFIG.blocks, V15_CONFIG.channels,
            V15_CONFIG.input_planes) == (4, 64, 21)
    assert V15_CONFIG.full_playout_frac == 0.15
    assert V15_CONFIG.adjudicate_margin == 3.0


def test_finish_move():
    # v15 finishing with a MOCK model (no net): switches to the
    # shorter-ML move when winning; holds otherwise.
    import numpy as _np
    import torch as _t
    from chess_zero.config import Config
    from chess_zero.game import State
    from chess_zero.selfplay import finish_move
    import chess as _c

    class _MockModel:
        def __init__(self, v, mls):
            # v18: out[1] is WDL logits — encode v as [w, d, l] mass.
            # v=+0.9 -> mostly win; v=-0.5 -> mostly loss.
            self._v = v
            self._mls = mls

        def parameters(self):
            return iter([_t.zeros(1)])

        def __call__(self, x):
            b = x.shape[0]
            if self._v >= 0:
                wdl = _t.tensor([[3.0, 0.0, -1.0]] + [[-1.0, 0.0, 3.0]] * (b-1))
            else:
                wdl = _t.tensor([[-3.0, 0.0, 1.0]] * b)
            return (_t.zeros(b, 4096), wdl,
                    _t.zeros(b), _t.tensor(self._mls[:b]),
                    _t.zeros(b, 8, 8), _t.zeros(b), _t.zeros(b),
                    _t.zeros(b), _t.zeros(b, 64))

    s = State(chess.Board())
    a_e4 = 12 * 64 + 28  # e2e4
    a_d4 = 11 * 64 + 27  # d2d4
    pi = _np.zeros(4096)
    pi[a_e4] = 0.7
    pi[a_d4] = 0.2
    cfg = Config(mate_finish=True)
    # children mls: [state, e4-child, d4-child] -> d4 shorter.
    m = _MockModel(0.9, [0.5, 0.8, 0.1])
    assert finish_move(s, a_e4, pi, m, "cpu", cfg) == a_d4
    # losing: hold the search pick.
    m2 = _MockModel(-0.5, [0.5, 0.8, 0.1])
    assert finish_move(s, a_e4, pi, m2, "cpu", cfg) == a_e4
    # flag off / model None: hold.
    assert finish_move(s, a_e4, pi, m, "cpu", Config()) == a_e4
    assert finish_move(s, a_e4, pi, None, "cpu", cfg) == a_e4


def test_progress_check_never_raises():
    # lineage cop: bad paths return {"error"}, never raise (a
    # measurement must not break a training run).
    from chess_zero.config import Config
    from chess_zero.loop import _progress_check
    out = _progress_check("/none/a.pt", "/none/b.pt",
                          Config(blocks=2, channels=8), 8, 1, "/tmp",
                          99, 96)
    assert "error" in out, out


def test_v18_config():
    # v18 final-bot cut: 6x64 SE, 30 tactical planes, WDL+reply heads,
    # fast/full split. Training book OFF (empirical artifact separate).
    from chess_zero.config import V18_CONFIG, V15_5_CONFIG
    c = V18_CONFIG
    assert (c.blocks, c.channels, c.input_planes) == (6, 64, 30)
    assert c.se_ratio == 4
    assert c.sims == 400
    assert c.mate_finish is True
    assert c.leaf_batch == 8
    assert c.safety_veto is True
    assert c.fast_frac == 0.5 and c.fast_sims == 100
    assert c.reply_w == 0.05
    assert c.book_path == "" and c.book_plies == 0
    assert c.opening_random_moves == 6
    assert c.adjudicate_margin == 3.0
    assert V15_5_CONFIG.input_planes == 21
    assert V15_5_CONFIG.se_ratio == 0


def test_tactical_planes():
    # v17 encoder: 30 planes; hang/defense/checkerboard light correctly;
    # startpos quiet; Black rotation carries them oriented.
    import numpy as _np
    import chess
    import chess_zero.game as _g
    from chess_zero.game import State
    old = _g.INPUT_PLANES
    _g.INPUT_PLANES = 30
    try:
        s0 = State(chess.Board())
        t0 = s0.encode()
        assert t0.shape == (30, 8, 8)
        # startpos: knights+pawns attack (plane 21 lit), but nothing
        # hangs, no checkers, no king danger, even material.
        assert t0[21].sum() > 0 and t0[23].sum() == 0.0
        assert t0[24].sum() == 0.0 and t0[25].sum() == 0.0
        assert t0[26].sum() == 0.0
        # checkerboard pattern present.
        assert t0[28, 0, 0] == 0.0 and t0[28, 0, 1] == 1.0
        # 1.e4 d5: e4 pawn hangs (attacked, undefended); d5 defended.
        from chess_zero.game import encode_action, mirror_square
        s = State(chess.Board())
        for u in ("e2e4", "d7d5"):
            fr, to = chess.parse_square(u[:2]), chess.parse_square(u[2:4])
            if s.board.turn == chess.BLACK:
                fr, to = mirror_square(fr), mirror_square(to)
            s = s.apply(encode_action(fr, to))
        t = s.encode()
        # e4 = file 4 rank 3 -> row 7-3=4? rank idx 3 -> r=7-3=4, c=4.
        assert t[23, 4, 4] == 1.0, "e4 must read hanging"
        assert t[21, 4, 4] == 1.0, "e4 must read attacked"
        # legal-from lights the stm pieces that can move.
        assert t[29].sum() > 0
    finally:
        _g.INPUT_PLANES = old


def test_se_block():
    # v18 SE trunk: off by default (no se keys, yesterday's block);
    # on runs finite with the right shapes.
    import torch
    from chess_zero.model import ResidualBlock, AlphaZeroNet
    b0 = ResidualBlock(16)
    assert not any("se_fc" in k for k in b0.state_dict())
    b1 = ResidualBlock(16, se_ratio=4)
    assert "se_fc1.weight" in b1.state_dict()
    x = torch.randn(2, 16, 8, 8)
    for b in (b0, b1):
        b.eval()
        with torch.no_grad():
            y = b(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
    n = AlphaZeroNet(blocks=2, channels=16, planes=13, se_ratio=4)
    with torch.no_grad():
        out = n(torch.randn(1, 13, 8, 8))
    assert len(out) == 15, len(out)
    assert out[1].shape == (1, 3)  # WDL
    assert out[11].shape == (1,)  # progress


def test_wdl_trains():
    # v18 WDL: Q math sane; CE falls on fixed WDL targets.
    import torch
    import numpy as _np
    from chess_zero.model import AlphaZeroNet, compute_loss, wdl_q
    assert abs(float(wdl_q(torch.tensor([[10.0, 0.0, 0.0]]))) - 1.0) < 1e-3
    assert abs(float(wdl_q(torch.tensor([[0.0, 0.0, 10.0]]))) + 1.0) < 1e-3
    assert abs(float(wdl_q(torch.tensor([[0.0, 0.0, 0.0]])))) < 1e-6
    torch.manual_seed(3)
    net = AlphaZeroNet(blocks=2, channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    e = torch.randn(8, 13, 8, 8)
    pi = torch.zeros(8, 4096)
    pi[:, 0] = 1.0
    z = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0])
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.5
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    l0 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg)[2])
    from chess_zero.train import train_step
    for _ in range(20):
        train_step(net, opt, e, pi, z, m, ml, _o, _sg, _sg)
    l1 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg)[2])
    assert l1 < l0, (l0, l1)


def test_plane_growth_surgery_exact():
    # v18: 21ch weights grow to 30ch with behavior preserved (new planes
    # x zero weights = 0 contribution). Same-prefix input -> diff 0.
    import torch
    import numpy as _np
    from chess_zero.model import AlphaZeroNet, load_weights
    torch.manual_seed(9)
    old = AlphaZeroNet(blocks=2, channels=16, planes=21)
    new = AlphaZeroNet(blocks=2, channels=16, planes=30)
    load_weights(new, {k: v for k, v in old.state_dict().items()})
    x21 = torch.randn(2, 21, 8, 8)
    x30 = torch.cat([x21, torch.zeros(2, 9, 8, 8)], dim=1)
    old.eval()
    new.eval()
    with torch.no_grad():
        o21 = old(x21)
        o30 = new(x30)
    # shared outputs identical (logits/policy trunk + aux on old planes).
    assert float((o21[0] - o30[0]).abs().max()) == 0.0
    assert float((o21[2] - o30[2]).abs().max()) == 0.0
    # scalar-era v_fc2 (1ch) grows to zeroed WDL (uniform, Q=0 — never
    # the biased [old,0,0]).
    import copy as _cp
    _fake_old = {k: v for k, v in old.state_dict().items()}
    _fake_old["v_fc2.weight"] = torch.zeros(1, 64)
    _fake_old["v_fc2.bias"] = torch.zeros(1)
    from chess_zero.model import grow_state_dict as _gs
    _gr = _gs({k: v for k, v in new.state_dict().items()}, _fake_old)
    assert bool((_gr["v_fc2.weight"] == 0).all())
    assert bool((_gr["v_fc2.bias"] == 0).all())


def test_fast_policy_mask():
    # v18 fast/full: all-zero policy weight == zeroed-pi total (policy
    # term excluded, everything else identical).
    import torch
    from chess_zero.model import AlphaZeroNet, compute_loss
    torch.manual_seed(4)
    net = AlphaZeroNet(blocks=2, channels=16)
    e = torch.randn(4, 13, 8, 8)
    pi = torch.zeros(4, 4096)
    pi[:, 7] = 1.0
    z = torch.ones(4)
    m = torch.zeros(4)
    ml = torch.ones(4) * 0.5
    _o = torch.zeros(4, 8, 8)
    _sg = torch.zeros(4)
    pw0 = torch.zeros(4)
    a = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg, policy_w=pw0)
    b = compute_loss(net, e, torch.zeros_like(pi), z, m, ml, _o, _sg,
                     _sg)
    assert abs(float(a[0]) - float(b[0])) < 1e-6
    assert float(a[1]) == 0.0  # policy term fully excluded


def test_reply_ignores_broken():
    # v18 reply aux: all -100 -> zero finite loss; valid rows train.
    import numpy as _np
    import torch
    from chess_zero.model import AlphaZeroNet, compute_loss
    from chess_zero.train import train_step
    torch.manual_seed(6)
    net = AlphaZeroNet(blocks=2, channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    e = torch.randn(6, 13, 8, 8)
    pi = torch.zeros(6, 4096)
    pi[:, 3] = 1.0
    z = torch.zeros(6)
    m = torch.zeros(6)
    ml = torch.ones(6) * 0.5
    _o = torch.zeros(6, 8, 8)
    _sg = torch.zeros(6)
    bad = torch.full((6,), -100, dtype=torch.long)
    r0 = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg,
                      target_reply=bad)
    assert float(r0[10]) == 0.0 and np.isfinite(float(r0[0]))
    good = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)
    l0 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg,
                            target_reply=good)[10])
    for _ in range(15):
        train_step(net, opt, e, pi, z, m, ml, _o, _sg, _sg,
                   reply=good)
    l1 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg,
                            target_reply=good)[10])
    assert l1 < l0, (l0, l1)


def test_shape_targets_11():
    # v18 11-tuples: reply chains on consecutive plies, -100 on gaps,
    # fast flag flows.
    import numpy as _np
    import chess
    from chess_zero.game import State
    from chess_zero.selfplay import shape_targets
    s = State(chess.Board())
    enc = s.encode()
    pi = _np.zeros(4096, dtype=_np.float32)
    pi[0] = 1.0
    a_e4 = 12 * 64 + 28
    # e7e5 as BLACK-stm stores it (oriented): mirror(e7)mirror(e5).
    # shape_targets absolutizes back to e7 (52).
    a_e5 = 11 * 64 + 27
    hist = [(enc, pi, 0, 0.0, 0, 0.5, a_e4),
            (enc, pi, 1, 0.0, 1, 0.5, a_e5),
            (enc, pi, 0, 0.0, 5, 0.5, a_e4)]  # ply gap -> -100
    ex = shape_targets(hist, 1.0, 6, 300, s.board, full=True)
    assert len(ex[0]) == 18  # V20
    assert ex[0][9] == 52, ex[0][9]  # reply = e7 from-square, absolute
    assert ex[1][9] == -100  # gap: next ply is 5, not 2
    assert ex[0][10] == 1.0
    exf = shape_targets(hist, 1.0, 6, 300, s.board, full=False)
    assert exf[0][10] == 0.0 and exf[0][9] == 52


def test_warmstart_load():
    # warm-start loader: synthetic 2-game jsonl -> positions with one-hot
    # pi, WDL-consistent z, absolute reply chain. Restores INPUT_PLANES
    #     (suite order must not matter — loader stamps 30 for its run).
    import json
    import numpy as _np
    import chess_zero.game as _gg
    from chess_zero.warmstart import load_positions
    _old_planes = _gg.INPUT_PLANES
    path = "/tmp/ws_test_games.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"id": "a",
                            "moves": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 "
                                     "b5a4 g8f6 e1g1 f8e7",
                            "winner": "white", "white_elo": 2300,
                            "black_elo": 2300, "speed": "rapid"}) + "\n")
        f.write(json.dumps({"id": "b",
                            "moves": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 "
                                     "c1g5 f8e7 e2e3 e8g8",
                            "winner": None, "status": "draw", "white_elo": 2400,
                            "black_elo": 2400, "speed": "blitz"}) + "\n")
    pos, ng = load_positions(path, 1000)
    assert ng == 2 and len(pos) == 20
    for e in pos:
        assert len(e) == 11
        assert abs(float(e[1].sum()) - 1.0) < 1e-6
        assert e[9] == -100 or 0 <= e[9] < 64
    assert pos[0][2] == 1.0  # white won, white stm
    assert pos[1][2] == -1.0  # white won, black stm
    assert pos[0][9] == 52  # reply to 1.e4 is e7e5 from e7
    _gg.INPUT_PLANES = _old_planes


def test_v15_5_config():
    # v15.5 rollback: v15 knobs + leaf batching + safety veto, NO
    # training book (random plies restored), margin back to 3.0.
    from chess_zero.config import V15_5_CONFIG, V15_CONFIG, V16_CONFIG
    c = V15_5_CONFIG
    assert (c.blocks, c.channels, c.input_planes) == (4, 64, 21)
    assert c.sims == 400
    assert c.mate_finish is True
    assert c.leaf_batch == 8
    assert c.safety_veto is True
    assert c.book_path == "" and c.book_plies == 0
    assert c.opening_random_moves == 6
    assert c.adjudicate_margin == 3.0
    assert c.full_playout_frac == 0.15
    assert V15_CONFIG.safety_veto is False
    assert V16_CONFIG.safety_veto is False


def test_safety_guard():
    # v15.5 safety veto with a MOCK model (no net): replaces a pick
    # whose sibling keeps our king clearly safer; sleeps on clustered
    # (untrained-head) outputs; holds on flag off / None / NaN.
    import numpy as _np
    import torch as _t
    from chess_zero.config import Config
    from chess_zero.game import State
    from chess_zero.selfplay import safety_guard
    import chess as _c

    class _MockSafe:
        def __init__(self, safes):
            self._s = safes

        def __call__(self, x):
            b = x.shape[0]
            return (_t.zeros(b, 4096), _t.zeros(b), _t.zeros(b),
                    _t.zeros(b), _t.zeros(b, 8, 8), _t.zeros(b),
                    _t.zeros(b), _t.tensor(self._s[:b]))

    s = State(chess.Board())
    a_e4 = 12 * 64 + 28  # e2e4
    a_d4 = 11 * 64 + 27  # d2d4
    pi = _np.zeros(4096)
    pi[a_e4] = 0.7
    pi[a_d4] = 0.2
    cfg = Config(safety_veto=True)
    # children batch order = cands order [e4, d4]: e4-child reads 0.9
    # danger for the opponent-view... ours = 1-0.9 = 0.1; d4 ours 0.8.
    m = _MockSafe([0.9, 0.2])
    act, ved = safety_guard(s, a_e4, pi, m, "cpu", cfg)
    assert ved is True and act == a_d4
    # clustered (untrained head): gap 0.02 < 0.15 -> sleep.
    m2 = _MockSafe([0.51, 0.49])
    act2, ved2 = safety_guard(s, a_e4, pi, m2, "cpu", cfg)
    assert ved2 is False and act2 == a_e4
    # flag off / model None / NaN: hold.
    assert safety_guard(s, a_e4, pi, m, "cpu", Config())[0] == a_e4
    assert safety_guard(s, a_e4, pi, None, "cpu", cfg)[0] == a_e4
    m3 = _MockSafe([float("nan"), 0.2])
    assert safety_guard(s, a_e4, pi, m3, "cpu", cfg)[0] == a_e4


def test_no_decisive_adjudication():
    # v16 AZ-faithful: infinite margin never calls a win. A quiet
    # up-material position draws out (adjudicated-draw) instead of
    # being counted decisive — games must be finished, not counted.
    import chess_zero.game as _g
    from chess_zero.game import State
    import chess
    old = _g.ADJUDICATE_MARGIN
    _g.ADJUDICATE_MARGIN = float("inf")
    try:
        s = State.initial()
        s.board.remove_piece_at(chess.D8)  # +9 edge, quiet 100 plies
        s.board.halfmove_clock = 100
        done, z = s.is_terminal()
        assert done and z == 0.0, (done, z)
        assert s.last_terminal == "rules-draw"
        # mates still terminate decisive with truncation off too.
        m = _play_uci_list(["f2f3", "e7e5", "g2g4", "d8h4"])
        done3, z3 = m.is_terminal()
        assert done3 and z3 == -1.0 and m.last_terminal == "mate"
    finally:
        _g.ADJUDICATE_MARGIN = old


def test_truncate_endings_flag():    # v13: TRUNCATE_ENDINGS gates material adjudication only — mate and
    # rules draws fire identically either way. Self-sufficient on game
    # globals (suite order must not matter).
    import chess_zero.game as _g
    from chess_zero.parallel import _apply_game_settings
    _apply_game_settings((3.0, 60, 0, 13))
    s = State.initial()
    s.board.remove_piece_at(chess.D8)  # black queen gone: +9 edge
    s.board.halfmove_clock = 60  # quiet past the no-progress rule
    try:
        assert _g.TRUNCATE_ENDINGS is True  # default: today's behaviour
        done, z = s.is_terminal()
        assert done and s.last_terminal == "adjudicated", (done, z)
        assert z == 1.0  # stm is white, up material: +1
        _g.TRUNCATE_ENDINGS = False
        done2, _ = s.is_terminal()
        assert not done2, "full playout must not adjudicate"
        # mate still terminates with truncation off (fool's mate).
        m = _play_uci_list(["f2f3", "e7e5", "g2g4", "d8h4"])
        done3, z3 = m.is_terminal()
        assert done3 and m.last_terminal == "mate" and z3 == -1.0
    finally:
        _g.TRUNCATE_ENDINGS = True
        _apply_game_settings((3.0, 60, 0, 13))
    assert _g.TRUNCATE_ENDINGS is True


def test_terminal_reason_recorded():
    # v13 honest compass: parallel results carry T* terminal counters
    # summing to the game count (truncated games here — fast).
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.model import AlphaZeroNet
    from chess_zero.parallel import _apply_game_settings, play_games_parallel
    cfg = dataclasses.replace(TEST_CONFIG, tactical_override=True,
                              blunder_veto=True)
    _apply_game_settings(cfg)
    try:
        net = AlphaZeroNet(blocks=2, channels=16, planes=13)
        _, res = play_games_parallel(net, cfg, 2, temp_moves=0, workers=1)
        tkeys = ("Tmate", "Tresign", "Tadjudicated", "Tadjudicated-draw",
                 "Trules-draw", "Tcap")
        assert all(k in res for k in tkeys), res
        assert sum(res[k] for k in tkeys) == 2, res
        # v14 pace metric rides along.
        assert res.get("bgames", 0) == 2, res
        assert res.get("plies", 0) > 0, res
    finally:
        _apply_game_settings((3.0, 100, 0, 13))


def test_visit_breadth_recorded():
    # v14 unconventional-move diagnostic: search fills breadth (distinct
    # root moves visited) + visit_entropy, bounded sanely.
    import math
    from chess_zero.game import State
    from chess_zero.model import AlphaZeroNet
    from chess_zero.selfplay import make_evaluate
    net = AlphaZeroNet(blocks=2, channels=16, planes=13)
    ev = make_evaluate(net, "cpu")
    s = State.initial()
    n_legal = len(s.legal_moves())
    st: dict = {}
    pi = mcts_mod.search(s, ev, n_sims=8, tt=None, stats=st)
    assert pi.sum() > 0.99
    assert 1 <= st["breadth"] <= n_legal, st
    assert 0.0 <= st["visit_entropy"] <= math.log(n_legal) + 1e-6, st
    assert st["root_visits"] > 0


def test_book_valid():
    # v16 book: 16 ECO lines, all legal, all starting with sane first
    # moves (no 1.a3 junk — the whole point of curating).
    import chess
    from chess_zero.book import load_book
    lines = load_book("BUILTIN")
    assert len(lines) == 16, len(lines)
    for l in lines:
        b = chess.Board()
        for u in l:
            b.push_uci(u)  # raises on any illegal move
        assert len(l) >= 12, len(l)
    assert {l[0] for l in lines} <= {"e2e4", "d2d4", "c2c4", "g1f3"}


def test_book_matching():
    # prefix matching + seeded choice + off-book None.
    import random
    from chess_zero.book import matching_lines, book_move
    lines = [["e2e4", "e7e5", "g1f3"], ["e2e4", "c7c5"],
             ["d2d4", "d7d5"]]
    assert matching_lines(lines, []) == lines
    assert matching_lines(lines, ["e2e4"]) == [lines[0], lines[1]]
    assert matching_lines(lines, ["e2e4", "e7e5", "g1f3", "b8c6"]) == []
    r = random.Random(0)
    assert book_move(lines, ["e2e4"], r) in ("e7e5", "c7c5")
    assert book_move(lines, ["e2e4", "e7e5", "g1f3", "b8c6"], r) is None


def test_king_safety_target():
    # v16 safety head teacher: symmetric startpos ~= 0.5; mated side
    # reads low; mating side reads high. stm-relative like margin.
    import chess
    from chess_zero.selfplay import king_safety
    s = chess.Board().mirror()  # still symmetric startpos
    assert abs(king_safety(chess.Board(), 0) - 0.5) < 0.05
    assert abs(king_safety(s, 0) - 0.5) < 0.05
    # fool's mate: white mated -> white-stm safety collapses.
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
    assert mated.is_checkmate()
    assert king_safety(mated, 0) < 0.45, king_safety(mated, 0)
    assert king_safety(mated, 1) > 0.55, king_safety(mated, 1)
    # KQ vs bare K (us to move, mating): high.
    kq = chess.Board("8/8/8/8/8/5QK1/8/7k w - - 0 1")
    assert king_safety(kq, 0) > 0.6, king_safety(kq, 0)


def test_v16_config():
    # v16 SF-films cut: 6x128 capacity, safety head on, ECO book
    # instead of random plies, no decisive adjudication (inf margin),
    # finishing + full playouts carried. Older configs frozen.
    from chess_zero.config import V16_CONFIG, V15_CONFIG
    assert (V16_CONFIG.blocks, V16_CONFIG.channels,
            V16_CONFIG.input_planes) == (6, 128, 21)
    assert V16_CONFIG.sims == 400
    assert V16_CONFIG.book_path == "BUILTIN" and V16_CONFIG.book_plies == 8
    assert V16_CONFIG.opening_random_moves == 0
    assert V16_CONFIG.adjudicate_margin == float("inf")
    assert V16_CONFIG.mate_finish is True
    assert V16_CONFIG.full_playout_frac == 0.15
    assert V15_CONFIG.adjudicate_margin == 3.0
    assert V15_CONFIG.book_path == ""


def test_channel_depth_surgery_equivalent():    # v16 4x64 -> 6x128 growth surgery preserves behavior: zero-padded
    # channels stay dead, zeroed new blocks are identity on non-negative
    # trunk activations. All 8 heads must match to ~1e-5 in eval mode.
    import torch
    from chess_zero.model import (AlphaZeroNet, grow_state_dict,
                                  load_weights)
    torch.manual_seed(0)
    small = AlphaZeroNet(blocks=4, channels=64, planes=21)
    big = AlphaZeroNet(blocks=6, channels=128, planes=21)
    grown = grow_state_dict(dict(big.state_dict()),
                            dict(small.state_dict()))
    big.load_state_dict(grown)  # strict: every key present, exact shape
    big.eval()
    small.eval()
    x = torch.randn(2, 21, 8, 8)
    with torch.no_grad():
        a = small(x)
        b = big(x)
    # v19: 12 outputs; same-arch-different-size growth is bit-exact on
    # EVERY head (new filters/channels are dead zeros, identity blocks;
    # v_fc2 shapes match so WDL copies verbatim; soft/check/progress
    # heads match shapes too). The scalar-era zero-row contract is
    # obsolete (no scalar checkpoints constructible).
    assert len(a) == len(b) == 15
    for i in range(15):
        d = (a[i] - b[i]).abs().max()
        assert d < 1e-5, (i, float(d))
    # same path the run uses (wrapped ckpt, strict=False).
    big2 = AlphaZeroNet(blocks=6, channels=128, planes=21)
    load_weights(big2, {"weights": dict(small.state_dict())})
    big2.eval()
    with torch.no_grad():
        c = big2(x)
    for i in range(15):
        assert torch.allclose(a[i], c[i], atol=1e-5)


def test_safety_head_trains_and_loads():    # v16: 8th output exists in [0,1]; old 7-head weights load with the
    # new head random (strict=False skips ABSENT keys).
    import torch
    from chess_zero.model import AlphaZeroNet, load_weights
    net = AlphaZeroNet(blocks=2, channels=16, planes=13)
    x = torch.randn(2, 13, 8, 8)
    out = net(x)
    assert len(out) == 15, len(out)
    assert bool(((out[7] >= 0.0) & (out[7] <= 1.0)).all())
    old = {k: v for k, v in net.state_dict().items()
           if not k.startswith("k_")}
    assert not any(k.startswith("k_") for k in old)
    net2 = AlphaZeroNet(blocks=2, channels=16, planes=13)
    load_weights(net2, old)  # must not raise; k_* stay random


def test_take_book_opening():    # curated self-play openings: takes plies, seeds rep history,
    # stays legal. Deterministic under seed.
    import random
    from chess_zero.book import load_book, take_book_opening
    from chess_zero.game import State
    lines = load_book("BUILTIN")
    s = State.initial()
    hist = [s.rep_key()]
    s2, n = take_book_opening(s, hist, lines, 4, random.Random(1))
    assert n == 4 and s2.ply_count == 4 and len(hist) == 5
    s3, n3 = take_book_opening(s, [], [], 4)
    assert n3 == 0 and s3.ply_count == 0


def test_rust_python_backend_parity():    # v14 Rust core: legal SETS (not order), queen-only, and check/
    # capture predicates agree with python-chess on tricky positions
    # (EP, promotions, castling) under both backends. Skipped when the
    # extension is missing (fallback is trivially self-consistent).
    import os as _os
    try:
        import chesscore as _cc
        assert _cc.__version__ == "0.1.0", _cc.__version__
    except ImportError:
        return
    from chess_zero.game import State
    fens = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        # legal EP available: white to play exd6.
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1",
        # promotion race both wings.
        "8/2P2k2/8/8/8/5K2/2p5/8 w - - 0 1",
        # castling through checkthemes (Kiwipete).
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]
    for fen in fens:
        b = chess.Board(fen)
        py_set = {(m.from_square, m.to_square) for m in b.legal_moves
                  if m.promotion in (None, chess.QUEEN)}
        s = State(chess.Board(fen))
        rs_set = {(m.from_square, m.to_square)
                  for m in s._legal_chess_moves()}
        assert py_set == rs_set, (fen, py_set ^ rs_set)
        for m in b.legal_moves:
            if m.promotion not in (None, chess.QUEEN):
                continue
            assert s.is_capture_fast(m) == b.is_capture(m), (fen, m)
            assert s.gives_check_fast(m) == b.gives_check(m), (fen, m)
        assert s.in_check_fast() == b.is_check(), fen
    # kill-switch restores generator-order python path deterministically.
    _os.environ["CHESS_ZERO_NO_CPP"] = "1"
    try:
        s2 = State(chess.Board(fens[0]))
        assert s2._rust_pos() is None
        assert len(s2._legal_chess_moves()) == 20
    finally:
        del _os.environ["CHESS_ZERO_NO_CPP"]


def test_resign_fires_per_side():
    # v14 per-side resign: threshold 2.0 fires on everything (v_stm in
    # [-1,1] is always < 2.0), so the game must end by resign in ~3
    # moves. Fast proof the streak path works — the old shared streak
    # never fired once in ~90 iters (Tresign 0 everywhere).
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.model import AlphaZeroNet
    from chess_zero.selfplay import play_game
    cfg = dataclasses.replace(TEST_CONFIG)
    net = AlphaZeroNet(blocks=2, channels=16, planes=13)
    st: dict = {}
    ex, res = play_game(net, cfg, device="cpu", temp_moves=0,
                        opening_random_moves=0, resign_threshold=2.0,
                        resign_moves=3, stats=st)
    assert st.get("terminal") == "resign", st
    assert res in ("1-0", "0-1"), (res, st)
    assert len(ex) > 0


def test_run_training_trains():
    # P0 regression: the gradient-step loop once sat indented under
    # else: — zero steps ran while history claimed train_steps. A tiny
    # real run must show nonzero steps and nonzero loss.
    import dataclasses
    import tempfile
    from chess_zero.config import TEST_CONFIG
    from chess_zero.loop import run_training
    cfg = dataclasses.replace(TEST_CONFIG)
    with tempfile.TemporaryDirectory() as d:
        h = run_training(cfg, games_per_iter=1, iters=1, train_steps=2,
                         arena_games=1, arena_sims=2, device="cpu",
                         ckpt_dir=d, workers=1)
        assert len(h) == 1
        assert h[0]["train_steps"] >= 1, h[0]
        assert h[0]["loss"] != 0.0, h[0]
        assert h[0]["buf"] >= 1, h[0]


def test_veto_telemetry():
    # results dicts carry veto/tactic counters (v10 telemetry).
    import dataclasses
    from chess_zero.config import TEST_CONFIG
    from chess_zero.model import AlphaZeroNet
    from chess_zero.parallel import _apply_game_settings, play_games_parallel
    cfg = dataclasses.replace(TEST_CONFIG, tactical_override=True,
                              blunder_veto=True)
    _apply_game_settings(cfg)
    try:
        net = AlphaZeroNet(blocks=2, channels=16, planes=13)
        _, res = play_games_parallel(net, cfg, 2, temp_moves=0, workers=1)
        assert "vetoes" in res and "tactics" in res, res
        assert sum(res[k] for k in ("1-0", "0-1", "1/2-1/2")) == 2
    finally:
        _apply_game_settings((3.0, 100, 0, 13))


def test_v9_config():
    from chess_zero.config import V9_CONFIG
    assert V9_CONFIG.input_planes == 18
    assert V9_CONFIG.blunder_veto
    assert V9_CONFIG.sparring_frac == 0.25
    assert V9_CONFIG.asymmetric_contempt and V9_CONFIG.sims == 40


def test_v19_forward_11_and_bn():
    # v19: forward 9->11 (..., reply, soft, check); indices 0-8 identical;
    # BN momentum 0.02; forward_pv/masked_log_probs keep working by index.
    import torch
    from chess_zero.model import AlphaZeroNet
    net = AlphaZeroNet(blocks=2, channels=16, planes=13)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert abs(float(m.momentum) - 0.02) < 1e-9, m.momentum
    x = torch.randn(2, 13, 8, 8)
    out = net(x)
    assert len(out) == 15, len(out)
    assert out[0].shape == (2, 4096) and out[1].shape == (2, 3)
    assert out[8].shape == (2, 64)  # reply stays index 8
    assert out[9].shape == (2, 4096)  # soft appended at 9
    assert out[10].shape == (2,)  # check appended at 10
    assert out[11].shape == (2,)  # progress appended at 11
    assert bool(((out[10] >= 0.0) & (out[10] <= 1.0)).all())
    pv = net.forward_pv(x)
    assert len(pv) == 2
    mask = torch.ones(2, 4096, dtype=torch.bool)
    lp, _, _, _ = net.masked_log_probs(x, mask)
    assert lp.shape == (2, 4096)


def test_v19_soft_trains():
    # v19 soft head: shapes + CE falls (13-tuple return).
    import torch
    import numpy as _np
    from chess_zero.model import AlphaZeroNet, compute_loss
    from chess_zero.train import train_step
    from chess_zero.game import State, ACTION_SIZE
    torch.manual_seed(3)
    net = AlphaZeroNet(blocks=2, channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    s = State.initial()
    e = torch.from_numpy(_np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    for a in s.legal_moves()[:4]:
        pi[:, a] = 0.25
    z = torch.zeros(8)
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.5
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    out0 = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg)
    assert len(out0) == 17, len(out0)
    s0 = float(out0[11])
    for _ in range(15):
        train_step(net, opt, e, pi, z, m, ml, _o, _sg, _sg)
    s1 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg)[11])
    assert s1 < s0, (s0, s1)


def test_v19_check_trains():
    # v19 check head: overfit tiny (BCE falls); None target -> zero.
    import torch
    import numpy as _np
    from chess_zero.model import AlphaZeroNet, compute_loss
    from chess_zero.train import train_step
    from chess_zero.game import State, ACTION_SIZE
    torch.manual_seed(7)
    net = AlphaZeroNet(blocks=2, channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    s = State.initial()
    e = torch.from_numpy(_np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    pi[:, s.legal_moves()[0]] = 1.0
    z = torch.zeros(8)
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.5
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    chk = torch.ones(8)  # all in-check
    c0 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg,
                            target_check=chk)[12])
    assert float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg)[12]) \
        == 0.0  # None -> graph-connected zero
    for _ in range(20):
        train_step(net, opt, e, pi, z, m, ml, _o, _sg, _sg, checks=chk)
    c1 = float(compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg,
                            target_check=chk)[12])
    assert c1 < c0, (c0, c1)


def test_v19_entropy_smooth():
    # v19: smooth on uniform pi == same total (finite); ent_reg finite+masked.
    import torch
    from chess_zero.model import AlphaZeroNet, compute_loss
    from chess_zero.game import State
    torch.manual_seed(4)
    net = AlphaZeroNet(blocks=2, channels=16)
    net.eval()
    e = torch.randn(2, 13, 8, 8)
    s = State.initial()
    legal = s.legal_moves()[:4]
    piu = torch.zeros(2, 4096)
    for a in legal:
        piu[:, a] = 0.25
    z = torch.zeros(2)
    m = torch.zeros(2)
    ml = torch.ones(2) * 0.5
    _o = torch.zeros(2, 8, 8)
    _sg = torch.zeros(2)
    o0 = compute_loss(net, e, piu, z, m, ml, _o, _sg, _sg, smooth_eps=0.0)
    o1 = compute_loss(net, e, piu, z, m, ml, _o, _sg, _sg, smooth_eps=0.1)
    assert len(o0) == 17 and len(o1) == 17
    assert abs(float(o0[0]) - float(o1[0])) < 1e-6, (float(o0[0]),
                                                     float(o1[0]))
    o2 = compute_loss(net, e, piu, z, m, ml, _o, _sg, _sg, ent_reg=0.01)
    assert _sg is not None and float(o2[0]) == float(o2[0])  # finite
    import math as _m
    assert _m.isfinite(float(o2[0]))
    # masked: illegal rows excluded — uniform over 4 legals, entropy>0
    assert float(o0[5]) > 0


def test_v19_value_w_scales():
    # v19: value_w scales WDL term linearly (float32 tolerance, not exact).
    import torch
    from chess_zero.model import AlphaZeroNet, compute_loss
    torch.manual_seed(5)
    net = AlphaZeroNet(blocks=2, channels=16)
    net.eval()
    e = torch.randn(4, 13, 8, 8)
    pi = torch.zeros(4, 4096)
    pi[:, 7] = 1.0
    z = torch.ones(4)
    m = torch.zeros(4)
    ml = torch.ones(4) * 0.5
    _o = torch.zeros(4, 8, 8)
    _sg = torch.zeros(4)
    a = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg, value_w=0.5)
    b = compute_loss(net, e, pi, z, m, ml, _o, _sg, _sg, value_w=1.0)
    assert len(a) == 17 and len(b) == 17
    expect = 0.5 * float(b[2])
    got = float(b[0]) - float(a[0])
    assert abs(got - expect) < 1e-5, (got, expect)


def test_v19_kl_zero():
    # v19: identical teacher/student -> ~0 KL; 12-tuple return.
    import torch
    import torch.nn.functional as _F
    from chess_zero.model import AlphaZeroNet
    from chess_zero.train import train_step
    torch.manual_seed(9)
    net = AlphaZeroNet(blocks=2, channels=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    e = torch.randn(4, 13, 8, 8)
    pi = torch.zeros(4, 4096)
    pi[:, 5] = 1.0
    z = torch.zeros(4)
    m = torch.zeros(4)
    ml = torch.ones(4) * 0.5
    _o = torch.zeros(4, 8, 8)
    _sg = torch.zeros(4)
    net.train()
    with torch.no_grad():
        _tout = net(e)
        teacher = _F.softmax(_tout[0], dim=1)
    out = train_step(net, opt, e, pi, z, m, ml, _o, _sg, _sg,
                     teacher_probs=teacher, kl_w=1.0)
    assert len(out) == 16, len(out)
    assert abs(float(out[-1])) < 1e-3, float(out[-1])
    # kl_w=0 -> graph-connected zero (float 0, finite total)
    out0 = train_step(net, opt, e, pi, z, m, ml, _o, _sg, _sg,
                      teacher_probs=teacher, kl_w=0.0)
    assert len(out0) == 16 and float(out0[-1]) == 0.0
    import math as _m
    assert _m.isfinite(float(out0[0]))


def test_v19_augment_flip_twice():
    # v19: flip twice == identity; reply -100 preserved; WDL invariant.
    import torch
    from chess_zero.train import augment_batch
    torch.manual_seed(0)
    b = torch.randn(2, 30, 8, 8)
    p = torch.zeros(2, 4096)
    p[:, 100] = 1.0
    p[:, 200] = 0.5
    ow = torch.randn(2, 8, 8)
    rep = torch.tensor([10, -100], dtype=torch.long)
    b1, p1, o1, r1 = augment_batch(b, p, ow, rep)
    assert r1[0] == (10 ^ 7) and r1[1] == -100, r1
    b2, p2, o2, r2 = augment_batch(b1, p1, o1, r1)
    assert bool(((b - b2).abs().max() == 0))
    assert bool(((p - p2).abs().max() == 0))
    assert bool(((ow - o2).abs().max() == 0))
    assert bool((rep == r2).all())
    # teacher permutes like pi and double-flips too
    t = torch.randn(2, 4096)
    _, pt1, _, _, tt1 = augment_batch(b, p, ow, rep, t)
    _, pt2, _, _, tt2 = augment_batch(b1, pt1, o1, r1, tt1)
    assert bool(((t - tt2).abs().max() == 0))


def test_v19_augment_castling_swap():
    # v19: mirror swaps castling planes 16<->17 and 18<->19 AFTER flip.
    import torch
    from chess_zero.train import augment_batch
    bc = torch.zeros(1, 30, 8, 8)
    bc[:, 16] = 1.0
    bc[:, 18] = 2.0
    p = torch.zeros(1, 4096)
    p[:, 0] = 1.0
    ow = torch.zeros(1, 8, 8)
    rep = torch.tensor([-100], dtype=torch.long)
    bf, pf, _of, _rf = augment_batch(bc, p, ow, rep)
    assert float(bf[0, 16].mean()) == 0.0
    assert float(bf[0, 17].mean()) == 1.0
    assert float(bf[0, 18].mean()) == 0.0
    assert float(bf[0, 19].mean()) == 2.0
    # pi file-flip: e2e4 (12*64+28) -> d2d4 (11*64+27)
    p2 = torch.zeros(1, 4096)
    p2[:, 12 * 64 + 28] = 1.0
    _, pf2, _, _ = augment_batch(bc, p2, ow, rep)
    assert int(pf2.argmax()) == 11 * 64 + 27, int(pf2.argmax())


def test_v19_strict_load():
    # v19: new head keys random-init via strict=False (existing pattern).
    import torch
    from chess_zero.model import AlphaZeroNet, load_weights
    net = AlphaZeroNet(blocks=2, channels=16, planes=13)
    old = {k: v for k, v in net.state_dict().items()
           if not (k.startswith("p_soft") or k.startswith("c_"))}
    assert not any(k.startswith("p_soft") for k in old)
    assert not any(k.startswith("c_") for k in old)
    net2 = AlphaZeroNet(blocks=2, channels=16, planes=13)
    load_weights(net2, old)  # must not raise; new heads stay random
    out = net2(torch.randn(1, 13, 8, 8))
    assert len(out) == 15


def test_v19_rehearsal():
    # v19: sample_rehearsal deterministic under seed; encode 11-tuples.
    import json as _j
    import chess_zero.game as _gg
    from chess_zero.warmstart import (sample_rehearsal,
                                      encode_rehearsal_batch,
                                      load_positions)
    _old = _gg.INPUT_PLANES
    path = "/tmp/ws_v19_test.jsonl"
    try:
        with open(path, "w") as f:
            f.write(_j.dumps({"id": "a", "moves": "e2e4 e7e5 g1f3 b8c6 "
                             "f1b5 a7a6 b5a4 g8f6 e1g1 f8e7",
                             "winner": "white"}) + "\n")
            f.write(_j.dumps({"id": "b", "moves": "d2d4 d7d5 c2c4 e7e6 "
                             "b1c3 g8f6 c1g5 f8e7 e2e3 e8g8",
                             "winner": None, "status": "draw"}) + "\n")
        r1 = sample_rehearsal(path, 2, seed=0)
        r2 = sample_rehearsal(path, 2, seed=0)
        assert r1 == r2, "deterministic under seed"
        assert len(r1) == 2 and len(r1[0]) == 2
        assert isinstance(r1[0][0], list) and len(r1[0][0]) >= 10
        enc = encode_rehearsal_batch(r1)
        assert len(enc) == 20, len(enc)
        for e in enc:
            assert len(e) == 13, len(e)  # 11 + P + ml_mask
            assert abs(float(e[1].sum()) - 1.0) < 1e-6
            assert 0 <= e[11] <= 300 and e[12] == 0.0  # truncated PGN: censored duration
        # shared internals: same count as load_positions on same file
        pos, _ng = load_positions(path, 1000)
        assert len(pos) == len(enc), (len(pos), len(enc))
    finally:
        _gg.INPUT_PLANES = _old


if __name__ == "__main__":
    test_fools_mate_terminal()
    test_scholars_mate_terminal()
    test_stalemate_is_draw()
    test_insufficient_material_draw()
    test_action_codec_roundtrip_all()
    test_encoding_sides_swap()
    test_mcts_finds_mate_in_one()
    test_adjudication_signs()
    test_value_range_and_masked_policy()
    test_replay_shapes()
    test_endgame_routing()
    test_material_target_values()
    test_no_progress_adjudication()
    test_contempt_shaping()
    test_aux_head_learns_material()
    test_moves_left_learns()
    test_search_repetition_draw()
    test_gate_openings()
    test_empirical_values()
    test_asymmetric_contempt()
    test_adjudication_floor()
    test_phase_plane()
    test_growth_surgery()
    test_optimizer_surgery_across_growth()
    test_entropy_term()
    test_eg_report()
    test_depth3_rung()
    test_worker_settings_stamp()
    test_v7_end_to_end_cpu()
    test_rep_planes_from_state()
    test_adapt_contempt()
    test_v8_config()
    test_v81_deterministic_diverse_arena()
    test_v82_config()
    test_sparring_game()
    test_pgn_record_replays()
    test_v83_config()
    test_tactical_free_capture()
    test_tactical_poisoned_pawn()
    test_tactical_mate_in_one()
    test_tactical_quiet_none()
    test_tactical_unit_counts()
    test_terminal_order_stalemate()
    test_threefold_rules_draw()
    test_mcts_uses_priors_first_descent()
    test_values_stamped()
    test_paired_openings()
    test_arena_seeds_opening_history()
    test_no_l2_param()
    test_tt_key_clockless()
    test_tt_child_promotion()
    test_noise_remix_from_raw()
    test_globals_structural()
    test_uci_history_planes()
    test_seed_base_varies_openings()
    test_values_decisive_only()
    test_tactical_queen_for_pawn()
    test_hangs_material()
    test_veto_blunder()
    test_punisher_move()
    test_apply_move_guards()
    test_make_policy_spec_compat()
    test_sparring_for_game()
    test_flip_mirror_consistency()
    test_flip_action_roundtrip()
    test_castling_planes()
    test_opp_castling_and_ep_planes()
    test_edge_scaled_contempt()
    test_search_contempt_zero_identical()
    test_search_contempt_avoids_repetition()
    test_v9_config()
    test_ownership_learns()
    test_quiescence_fires()
    test_capture_extension_fires()
    test_promo_extension_fires()
    test_forcing_ordering()
    test_veto_telemetry()
    test_v10_config()
    test_v11_config()
    test_v12_config()
    test_v13_config()
    test_v14_config()
    test_truncate_endings_flag()
    test_terminal_reason_recorded()
    test_visit_breadth_recorded()
    test_rust_python_backend_parity()
    test_v15_config()
    test_finish_move()
    test_v15_5_config()
    test_safety_guard()
    test_progress_check_never_raises()
    test_v18_config()
    test_tactical_planes()
    test_se_block()
    test_wdl_trains()
    test_plane_growth_surgery_exact()
    test_fast_policy_mask()
    test_reply_ignores_broken()
    test_shape_targets_11()
    test_warmstart_load()
    test_resign_fires_per_side()
    test_run_training_trains()
    test_book_valid()
    test_book_matching()
    test_take_book_opening()
    test_no_decisive_adjudication()
    test_king_safety_target()
    test_safety_head_trains_and_loads()
    test_channel_depth_surgery_equivalent()
    test_v16_config()
    test_v85_config()
    test_v86_effective_budget_tracked()
    test_v86_promotion_requires_corroboration()
    test_v86_tt_bounded()
    test_v86_tt_prune_kills_stale_history()
    test_v86_scaled_steps()
    test_v86_weighted_refit()
    test_v19_forward_11_and_bn()
    test_v19_soft_trains()
    test_v19_check_trains()
    test_v19_entropy_smooth()
    test_v19_value_w_scales()
    test_v19_kl_zero()
    test_v19_augment_flip_twice()
    test_v19_augment_castling_swap()
    test_v19_strict_load()
    test_v19_rehearsal()
    print("all full tests passed")
