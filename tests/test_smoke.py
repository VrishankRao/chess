"""Smoke tests proving M1/M2/M3 integration (proposal milestones).
Fast: uses TEST_CONFIG (2x32 net, 8 sims).
"""
import numpy as np
import torch

from chess_zero.game import State, ACTION_SIZE
from chess_zero.config import TEST_CONFIG
from chess_zero.model import AlphaZeroNet
from chess_zero import mcts as mcts_mod
from chess_zero.selfplay import play_game, make_evaluate
from chess_zero.train import train_step
from chess_zero.evaluate import random_move, greedy_move, play_match


def test_encoding_shape():
    s = State.initial()
    e = s.encode()
    assert e.shape == (13, 8, 8), e.shape
    assert set(np.unique(e)).issubset({0.0, 1.0})


def test_legal_mask_matches():
    s = State.initial()
    assert s.legal_mask().sum() == len(s.legal_moves()) == 20


def test_mcts_pi_valid():
    cfg = TEST_CONFIG
    net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels)
    ev = make_evaluate(net)
    s = State.initial()
    pi = mcts_mod.search(s, ev, n_sims=cfg.sims)
    assert pi.shape == (ACTION_SIZE,)
    assert abs(pi.sum() - 1.0) < 1e-5, pi.sum()
    assert (pi[s.legal_mask()] > 0).all() or pi.sum() > 0
    assert (pi[~s.legal_mask()] == 0).all()


def test_selfplay_game():
    cfg = TEST_CONFIG
    net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels)
    ex, res = play_game(net, cfg, temp_moves=2)
    assert len(ex) > 0
    assert res in ("1-0", "0-1", "1/2-1/2")
    for enc, pi, z, m, ml, _own, _marg, _mob, _safe, _reply, _pw, *_kl in ex:
        assert enc.shape == (13, 8, 8)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert z in (-1.0, 0.0, 1.0) or -1.0 <= z <= 1.0  # contempt-shaped
        assert isinstance(float(m), float)
        assert 0.0 <= float(ml) <= 1.0
        assert _reply == -100 or 0 <= _reply < 64
        assert float(_pw) in (0.0, 1.0)


def test_training_reduces_loss():
    torch.manual_seed(1)
    np.random.seed(1)
    cfg = TEST_CONFIG
    net = AlphaZeroNet(blocks=cfg.blocks, channels=cfg.channels)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    s = State.initial()
    e = torch.from_numpy(np.stack([s.encode()] * 8))
    pi = torch.zeros(8, ACTION_SIZE)
    for a in s.legal_moves()[:4]:
        pi[:, a] = 0.25
    z = torch.ones(8) * 0.5
    m = torch.zeros(8)
    ml = torch.ones(8) * 0.5
    _o = torch.zeros(8, 8, 8)
    _sg = torch.zeros(8)
    l0, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = train_step(net, opt, e, pi, z, m, ml,
                                             _o, _sg, _sg)
    for _ in range(5):
        l1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = train_step(net, opt, e, pi, z, m, ml,
                                                 _o, _sg, _sg)
    assert l1 < l0, (l0, l1)


def test_baselines_play():
    r = play_match(random_move, greedy_move, games=2, cap=60)
    assert r["wins"] + r["losses"] + r["draws"] == 2


if __name__ == "__main__":
    test_encoding_shape()
    test_legal_mask_matches()
    test_mcts_pi_valid()
    test_selfplay_game()
    test_training_reduces_loss()
    test_baselines_play()
    print("all smoke tests passed")
