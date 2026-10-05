"""Small native differential checks; require the rebuilt audited extension."""
import numpy as np
import pytest
import chess
from chess_zero.game import State
from chess_zero.mcts_bridge import PythonTreeBackend, RustTreeBackend, new_game_store, rust_available

pytestmark=pytest.mark.skipif(not rust_available(), reason='audited native extension not installed')

def evaluate(states):
    p=np.zeros((len(states),4096))
    for i,s in enumerate(states):
        legal=s.legal_moves();p[i,legal]=np.arange(1,len(legal)+1,dtype=float)
    p/=p.sum(1,keepdims=True)
    return p,np.zeros(len(states))

@pytest.mark.parametrize("flat", [False, True])
def test_native_game_reuse_and_promotions(flat):
    py=PythonTreeBackend();rs=RustTreeBackend()
    def ev(states):
        if not flat:return evaluate(states)
        p=np.stack([s.legal_mask().astype(float) for s in states]);p/=p.sum(1,keepdims=True)
        return p,np.zeros(len(states))
    fens=[chess.STARTING_FEN,'rn1r3k/4bp2/b5P1/2p4P/1pp1P3/5P2/1q6/2QRKB1R w K - 3 28',
          '4k3/P7/8/8/8/8/4K3/8 w - - 0 1','4k3/8/8/8/8/8/p3K3/8 b - - 0 1']
    for fen in fens:
        st=State(chess.Board(fen));pt={};rt=new_game_store()
        for ply in range(6):
            if st.is_terminal()[0]:break
            kw=dict(dirichlet_eps=0.,leaf_batch=4,quiescence_depth=0,fpu_reduction=.5,history=list(st._hist))
            p,a=py.search(st,ev,24,tt=pt,**kw);q,b=rs.search(st,ev,24,tt=rt,**kw)
            np.testing.assert_array_equal(p,q, err_msg=f"fen={fen}; ply={ply}; state={st.board.fen()}; python={a}; rust={b}")
            assert a['reused']==b['reused']
            st=st.apply(int(p.argmax()))
            from chess_zero.mcts import prune_tt
            prune_tt(pt,st.key());rt.prune_to(st.key())

def test_native_errors_are_pickleable():
    import pickle
    def broken(states):raise RuntimeError('intentional inference failure')
    with pytest.raises(RuntimeError) as raised:
        RustTreeBackend().search(State.initial(),broken,4,dirichlet_eps=0.)
    assert str(pickle.loads(pickle.dumps(raised.value)))
