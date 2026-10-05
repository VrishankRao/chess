"""Semantic regression tests for the September 2026 audit (CPU, no data jobs)."""
import unittest
import tempfile
import time
import copy
import numpy as np
import torch
import chess
from chess_zero import mcts
from chess_zero.model import AlphaZeroNet, load_weights, load_optimizer_state
from chess_zero.game import State
from chess_zero.config import Config

torch.set_num_threads(1)

class AuditRegressions(unittest.TestCase):
    def test_optimizer_resume_next_step_matches(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.01)
        p.square().sum().backward();opt.step();opt.zero_grad()
        q=torch.nn.Parameter(p.detach().clone());other=torch.optim.AdamW([q],lr=.01)
        load_optimizer_state(other,{'opt':copy.deepcopy(opt.state_dict())})
        for param,optimizer in [(p,opt),(q,other)]:
            param.square().sum().backward();optimizer.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertEqual(other.state[q]['step'].item(),2)

    def test_growth_filters_learn(self):
        old=AlphaZeroNet(blocks=1,channels=4);new=AlphaZeroNet(blocks=2,channels=4)
        load_weights(new,old.state_dict());opt=torch.optim.AdamW(new.parameters(),lr=.01)
        x=torch.randn(4,13,8,8)
        for _ in range(3):
            opt.zero_grad();new(x)[1].square().mean().backward();opt.step()
        self.assertGreater(new.blocks[1].c1.weight.grad.abs().max().item(),0)

    def test_fpu_uses_selecting_player(self):
        # Force a winning visited move and compare an unvisited sibling.
        root=mcts._Node();child=mcts._Node();mcts._backup([root,child],.8)
        self.assertAlmostEqual(mcts._select_fpu(mcts._Node(),-root.q,1,0,.4),.4)
        self.assertLess(mcts._ml_bonus(.8,.5,-root.q,.003,.07,.5),0)

    def test_saturated_loss_does_not_stop_search(self):
        root=mcts._Node();root.children[1]=mcts._Node();root.children[1].n=3;root.children[1].w=-3
        self.assertFalse(mcts._root_decided(root))

    def test_collision_reservations_and_exception_cleanup(self):
        st=State.initial();tt={}
        def ev(states):
            p=np.zeros((len(states),4096))
            for i,s in enumerate(states):p[i,s.legal_moves()[0]]=1
            return p,np.zeros(len(states))
        pi=mcts.search(st,ev,8,leaf_batch=8,dirichlet_eps=0,quiescence_depth=0,tt=tt)
        stack=list(tt.values())
        while stack:
            node=stack.pop();self.assertEqual(node.o,0);stack.extend(node.children.values())
        calls=0
        def bad(states):
            nonlocal calls
            calls+=1
            if calls>1:raise RuntimeError('intentional')
            return ev(states)
        tt={}
        with self.assertRaises(RuntimeError):mcts.search(st,bad,8,leaf_batch=8,quiescence_depth=0,tt=tt)
        stack=list(tt.values())
        while stack:
            node=stack.pop();self.assertEqual(node.o,0);stack.extend(node.children.values())

    def test_terminal_start(self):
        from chess_zero.selfplay import play_game
        rows,result=play_game(None,Config(),evaluate_fn=lambda s:None,start_fen='7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
        self.assertEqual(rows,[]);self.assertEqual(result,'1-0')

    def test_pool_path_loader(self):
        from chess_zero.loop import agent_from_weights
        cfg=Config(blocks=1,channels=4,input_planes=13)
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            torch.save(AlphaZeroNet(blocks=1,channels=4).state_dict(),f.name)
            agent=agent_from_weights(f.name,cfg,sims=1)
            self.assertEqual(len(agent._spec),31)

    def test_uci_deadline_returns_legal_policy(self):
        def ev(states):return np.stack([s.legal_mask().astype(float) for s in states]),np.zeros(len(states))
        pi=mcts.search(State.initial(),ev,100000,deadline=time.monotonic()-1,dirichlet_eps=0)
        self.assertTrue(np.isfinite(pi).all());self.assertAlmostEqual(float(pi.sum()),1,places=5)

    def test_book_line_reaches_match(self):
        from chess_zero.evaluate import play_match
        seen=[]
        def policy(s):seen.append(s.board.fen());return s.legal_moves()[0]
        play_match(policy,policy,games=2,cap=3,opening_lines=[['e2e4','e7e5']])
        board=chess.Board();board.push_uci('e2e4');board.push_uci('e7e5')
        self.assertEqual(seen,[board.fen(),board.fen()])

    def test_weak_opponents_never_bypass_incumbent(self):
        from chess_zero.loop import _panel_decide,_v21_leg_pass
        self.assertFalse(_panel_decide({'incumbent':(3,3,2),'ancestor':(2,4,0),'punisher':(400,0,0)})['promote'])
        self.assertFalse(_v21_leg_pass(False,100,2.94))

    def test_cursed_wdl_is_draw(self):
        from chess_zero.tb_rescore import wdl_to_q
        self.assertEqual([wdl_to_q(i) for i in [-2,-1,0,1,2]],[-1,0,0,0,1])

    def test_fused_eval_reuses_forward(self):
        from chess_zero.selfplay import make_evaluate,resolve_ml_fn
        net=AlphaZeroNet(blocks=1,channels=4);count=[0];original=net.forward_inf
        def counted(x):count[0]+=1;return original(x)
        net.forward_inf=counted;ev=make_evaluate(net,jit=False);states=[State.initial()]
        ev(states);resolve_ml_fn(net)(states);self.assertEqual(count[0],1)

if __name__=='__main__':unittest.main()

class AuditAdditionalRegressions(unittest.TestCase):
    def test_underpromotions_both_colors(self):
        for fen in ['4k3/P7/8/8/8/8/4K3/8 w - - 0 1', '4k3/8/8/8/8/8/p3K3/8 b - - 0 1']:
            state=State(chess.Board(fen))
            actions=state.legal_moves()
            self.assertEqual(len(set(actions)),len(list(state.board.legal_moves)))
            self.assertEqual({state.to_move(a) for a in actions},set(state.board.legal_moves))
            self.assertEqual({state.to_move(a).promotion for a in actions if state.to_move(a).promotion},{chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN})
            for a in actions:state.apply(a)

    def test_encoding_roundtrip_legal_mask(self):
        from chess_zero.game import state_from_encoding
        import random
        rng=random.Random(19);state=State.initial()
        for _ in range(35):
            rebuilt=state_from_encoding(state.encode(planes=31))
            np.testing.assert_array_equal(state.legal_mask(),rebuilt.legal_mask())
            state=state.apply(rng.choice(state.legal_moves()))

    def test_twofold_is_not_terminal(self):
        state=State.initial();key=state.rep_key()
        once=State(state.board.copy(),_hist=[key])
        twice=State(state.board.copy(),_hist=[key,key])
        self.assertFalse(once.is_terminal()[0]);self.assertTrue(twice.is_terminal()[0])
        self.assertNotEqual(once.key(),twice.key())

    def test_bn_refresh_failure_restores_buffers(self):
        from chess_zero.loop import _precise_bn_lite
        net=AlphaZeroNet(blocks=1,channels=4);before=copy.deepcopy(net.state_dict())
        class BrokenReplay:
            def __len__(self):return 10
            def sample(self,n):raise RuntimeError('fixture')
        self.assertFalse(_precise_bn_lite(net,BrokenReplay(),batch=2))
        self.assertTrue(net.training)
        for k,v in before.items():torch.testing.assert_close(net.state_dict()[k],v,rtol=0,atol=0)

    def test_sparse_replay_and_capacity(self):
        from chess_zero.replay import ReplayBuffer,SparsePolicy
        dense=np.zeros(4096);dense[[10,100]]= [.3,.7]
        sparse=SparsePolicy(dense)
        self.assertLess(sparse.nbytes,32)
        np.testing.assert_array_equal(np.asarray(sparse),dense.astype(np.float16))
        with tempfile.NamedTemporaryFile() as f:
            torch.save(sparse,f.name)
            np.testing.assert_array_equal(np.asarray(torch.load(f.name,weights_only=False)),np.asarray(sparse))
        buf=ReplayBuffer(capacity=20,eg_capacity=20,eg_frac=.25)
        self.assertEqual(buf.buf.maxlen+buf.eg.maxlen,20)

    def test_no_second_batchnorm_update_for_kl(self):
        from chess_zero.train import train_step
        net=AlphaZeroNet(blocks=1,channels=4);opt=torch.optim.AdamW(net.parameters())
        st=State.initial();x=torch.tensor(np.stack([st.encode()]*2));pi=torch.tensor(np.stack([st.legal_mask()]*2),dtype=torch.float32);pi/=pi.sum(1,keepdim=True)
        z=torch.zeros(2);own=torch.zeros(2,8,8)
        train_step(net,opt,x,pi,z,z,z,own,z,z,teacher_probs=pi,kl_w=.1)
        self.assertEqual(int(net.trunk_bn.num_batches_tracked),1)

    def test_uci_prefetched_stop_is_not_lost(self):
        import subprocess,sys
        with tempfile.NamedTemporaryFile(suffix='.pt') as ckpt:
            torch.save(AlphaZeroNet(blocks=1,channels=4).state_dict(),ckpt.name)
            proc=subprocess.run([sys.executable,'-m','chess_zero.uci','--ckpt',ckpt.name,'--blocks','1','--channels','4','--qdepth','0'],input='uci\nisready\nposition startpos\ngo infinite\nstop\nquit\n',text=True,capture_output=True,timeout=30)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertIn('readyok',proc.stdout)
            best=[line.split()[1] for line in proc.stdout.splitlines() if line.startswith('bestmove ')]
            self.assertEqual(len(best),1);self.assertIn(chess.Move.from_uci(best[0]),chess.Board().legal_moves)

    def test_unknown_result_is_rejected(self):
        from chess_zero.warmstart import valid_result_record
        self.assertFalse(valid_result_record({'status':'aborted'}))
        self.assertFalse(valid_result_record({}))
        self.assertTrue(valid_result_record({'status':'draw'}))
        self.assertTrue(valid_result_record({'winner':'white'}))

class EncodingSymmetryRegression(unittest.TestCase):
    def test_mirror_matches_fresh_encoding_without_castling(self):
        from chess_zero.train import augment_batch
        st=State(chess.Board('4k3/8/2p5/3P4/8/2N5/8/4K3 w - - 7 20'))
        x=torch.tensor(st.encode(31))[None];pi=torch.tensor(st.legal_mask().astype(float))[None]
        mirrored,*_=augment_batch(x,pi,torch.zeros(1,8,8),torch.tensor([-100]))
        fresh=State(st.board.transform(chess.flip_horizontal)).encode(31)
        np.testing.assert_array_equal(mirrored[0].numpy(),fresh)

class CheckpointRolesRegression(unittest.TestCase):
    def test_resume_preserves_existing_champion_metadata(self):
        from unittest.mock import patch
        from pathlib import Path
        from chess_zero.loop import run_training,_build_adamw,save_checkpoint
        cfg=Config(blocks=1,channels=4,input_planes=13)
        champion=AlphaZeroNet(blocks=1,channels=4)
        challenger=AlphaZeroNet(blocks=1,channels=4)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);resume=root/'iter4.pt'
            save_checkpoint(challenger,_build_adamw(challenger,cfg),str(resume),{'iter':4})
            torch.save({'weights':champion.state_dict(),'meta':{'iter':2,'promo_seq':1}},root/'incumbent.pt')
            with patch('chess_zero.loop._arena',return_value={}):
                run_training(cfg,iters=0,ckpt_dir=directory,resume=str(resume),start_iter=5,device='cpu')
            best=torch.load(root/'best.pt',weights_only=False)
            self.assertEqual(best['meta'],{'iter':2,'promo_seq':1})
            self.assertNotIn('opt',best)
            for name,value in champion.state_dict().items():torch.testing.assert_close(best['weights'][name],value,rtol=0,atol=0)
