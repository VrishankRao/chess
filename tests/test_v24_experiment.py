import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import dataclasses
from chess_zero.config import Config, TEST_CONFIG
from chess_zero.experiment_support import paired_superiority, challenger_game_ids, collect_mixed_selfplay
from chess_zero.loop import _v21_gate_cum_decide, run_training

class V24Tests(unittest.TestCase):
    def test_exact_test_matches_exhaustive_null(self):
        for scores in ((1,.75,.5,0),(.75,.75,.5,.25),(1,1,1,1),(.5,)*8):
            observed=sum(s-.5 for s in scores)
            diffs=[s-.5 for s in scores]
            null=[sum(a*b for a,b in zip(signs,diffs)) for signs in itertools.product((-1,1),repeat=len(diffs))]
            self.assertEqual(paired_superiority(scores)['p_value'],sum(x>=observed for x in null)/len(null))
        self.assertFalse(paired_superiority([])['pass'])
        self.assertFalse(paired_superiority([.5]*64)['pass'])
        self.assertFalse(paired_superiority([0]*64)['pass'])
        self.assertTrue(paired_superiority([1]*64)['pass'])

    def test_null_false_positive_bound(self):
        for alpha in (.05, .00625):
            rejections = 0
            for signs in itertools.product((-1,1),repeat=8):
                scores=[.5+.25*sign for sign in signs]
                rejections += paired_superiority(scores,alpha)['pass']
            self.assertLessEqual(rejections/256,alpha)

    def test_no_llr_bypass(self):
        for main,eg in ((False,True),(True,False),(True,True)):
            g={'via':'panel+endgame','main_sprt_pass':main,'endgame_panel':{'pass':eg}}
            self.assertEqual(_v21_gate_cum_decide(g,100,100,Config())['sprt_pass_cum'],main and eg)

    def test_actor_counts_and_merge(self):
        self.assertEqual(len(challenger_game_ids(20,.5,1)),10)
        calls=[]
        def collect(model,cfg,count,temp,workers,**kw):
            calls.append((count,kw['weights_path']))
            return [kw['weights_path']]*count,{'1/2-1/2':count,'plies':count*10,'breadth':count*.5}
        ex,stats=collect_mixed_selfplay(collect,Config(),20,30,7,'champ','chall',.5)
        self.assertEqual(calls,[(10,'chall'),(10,'champ')])
        self.assertEqual(stats['actor_challenger_games'],10)
        self.assertEqual(stats['actor_champion_games'],10)
        self.assertEqual(stats['plies'],200)
        self.assertEqual(len(ex),20)

    def test_sequential_actor_switch_and_saving(self):
        cfg=dataclasses.replace(TEST_CONFIG,challenger_selfplay_frac=.5)
        real_load=__import__('chess_zero.model',fromlist=['load_weights']).load_weights
        arena={'vs_greedy':{'wins':1,'losses':0,'draws':0}}
        with tempfile.TemporaryDirectory() as d,patch('chess_zero.model.load_weights',wraps=real_load),patch('chess_zero.loop.play_game',return_value=([],'1/2-1/2')),patch('chess_zero.loop._arena',return_value=arena),patch('chess_zero.loop._gate',return_value={'score':0,'llr':0,'sprt_pass':False}):
            h=run_training(cfg,games_per_iter=4,iters=1,train_steps=0,arena_games=1,arena_sims=2,device='cpu',ckpt_dir=d,workers=1)
            self.assertEqual(h[0]['games']['actor_challenger_games'],2)
            self.assertEqual(h[0]['games']['actor_champion_games'],2)
            self.assertTrue(Path(d,'iter1.pt').exists())

if __name__=='__main__':unittest.main()
