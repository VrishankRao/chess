"""Repeatable local throughput probe. Does not train, promote, or save weights.

Example: python -m chess_zero.audit_benchmark --blocks 6 --channels 64 \
  --planes 30 --checkpoint checkpoints_v21/incumbent.pt --out benchmark.json
Use explicit matching architecture flags for a checkpoint. Results describe
this machine and workload, not Elo or expected gains from a training run.
"""
from __future__ import annotations
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from .game import State
from .model import AlphaZeroNet, load_weights, infer_se_ratio
from .selfplay import make_evaluate
from . import mcts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint')
    ap.add_argument('--blocks',type=int,default=1)
    ap.add_argument('--channels',type=int,default=8)
    ap.add_argument('--planes',type=int,default=31)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--sims',type=int,default=32)
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--out',default='audit-benchmark.json')
    args=ap.parse_args()
    if args.repeats < 1 or args.sims < 1:ap.error('repeats and sims must be positive')
    torch.set_num_threads(args.threads);torch.manual_seed(7);np.random.seed(7)
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    model=AlphaZeroNet(blocks=args.blocks,channels=args.channels,planes=args.planes,se_ratio=infer_se_ratio(weights)).to(args.device).eval()
    if weights is not None:load_weights(model,weights,strict=True)
    rng=random.Random(7);state=State.initial();states=[]
    for _ in range(32):
        states.append(state)
        state=state.apply(rng.choice(state.legal_moves()))
        if state.is_terminal()[0]:state=State.initial()
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        elif args.device=='mps':torch.mps.synchronize()
    def measure(fn):
        fn();sync();times=[]
        for _ in range(args.repeats):
            sync();start=time.perf_counter();fn();sync();times.append(time.perf_counter()-start)
        return {'median_seconds':statistics.median(times),'max_seconds':max(times),'samples':times}
    report={'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'settings':vars(args),'inference':{},'search':{}}
    with torch.inference_mode():
        for batch in (1,4,8,16,32):
            x=torch.from_numpy(np.stack([s.encode(args.planes) for s in states[:batch]])).to(args.device)
            full=measure(lambda:model(x));fused=measure(lambda:model.forward_inf(x))
            full['positions_per_second']=batch/full['median_seconds'];fused['positions_per_second']=batch/fused['median_seconds']
            report['inference'][str(batch)]={'full':full,'fused':fused}
        evaluate=make_evaluate(model,args.device)
        for batch in (1,4,8):
            def search():
                np.random.seed(7)
                return mcts.search(states[12],evaluate,args.sims,leaf_batch=batch,dirichlet_eps=0.,quiescence_depth=0)
            report['search'][str(batch)]=measure(search)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(f'Benchmark written to {args.out}; no weights changed.')

if __name__=='__main__':main()
