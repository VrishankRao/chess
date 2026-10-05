import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from chess_zero.config import Config, TEST_CONFIG
from chess_zero.loop import _arena, _evaluation_schedule, run_training

class EvaluationCadenceTests(unittest.TestCase):
    def test_absolute_cadence_and_final(self):
        cfg=Config(gate_every=5, weak_diagnostics_every=10)
        self.assertEqual([_evaluation_schedule(cfg,i,40) for i in (4,5,6,10,40)],
                         [(False,False),(True,False),(False,False),(True,True),(True,True)])
        self.assertEqual(_evaluation_schedule(cfg,37,37),(True,True))

    def test_greedy_corroboration_survives_diagnostic_skip(self):
        cfg=Config()
        result={'wins':1,'losses':0,'draws':0}
        for weak,count in ((False,1),(True,2)):
            cfg.panel_weak_diagnostics=weak
            with patch('chess_zero.loop.agent_policy',return_value=lambda s: None), patch('chess_zero.loop.play_match',return_value=result) as match:
                out=_arena(None,cfg,1,1,'cpu')
                self.assertEqual(match.call_count,count)
                self.assertEqual('vs_random' in out,weak)
                self.assertEqual(out['vs_greedy'],result)

    def test_skipped_iteration_saves_and_never_promotes(self):
        cfg=dataclasses.replace(TEST_CONFIG, gate_every=5, weak_diagnostics_every=10)
        arena={'vs_greedy':{'wins':1,'losses':0,'draws':0}}
        gate={'score':0.0,'llr':0.0,'sprt_pass':False}
        with tempfile.TemporaryDirectory() as d, patch('chess_zero.loop.play_game',return_value=([], '1/2-1/2')), patch('chess_zero.loop._arena',return_value=arena) as a, patch('chess_zero.loop._gate',return_value=gate) as g:
            h=run_training(cfg,games_per_iter=1,iters=2,train_steps=0,arena_games=1,arena_sims=2,device='cpu',ckpt_dir=d,workers=1,start_iter=6)
            self.assertTrue(h[0]['gate']['skipped'])
            self.assertFalse(h[0]['promoted'])
            self.assertIsNone(h[0]['llr'])
            self.assertTrue(Path(d,'iter6.pt').exists())
            self.assertTrue(Path(d,'iter7.pt').exists())
            self.assertEqual(g.call_count,1)
            self.assertEqual(a.call_count,2) # incumbent baseline + final iteration

if __name__=='__main__': unittest.main()
