"""Fixed-sample paired evaluation and self-play experiment helpers."""
from collections import defaultdict
import math
import random


def paired_superiority(scores, alpha=0.05):
    """Exact one-sided sign-flip test of pair score minus 0.5.

    One observation per independent opening, with both engine colors.
    Null assumes engine labels are exchangeable within each opening pair.
    Quarter-point scores permit an exact integer dynamic program, no SciPy.
    Fixed sample only: no optional stopping or pooling changing checkpoints.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    values = list(scores)
    if not values:
        return {'pass': False, 'p_value': 1.0, 'score': 0.0, 'pairs': 0}
    if any(not math.isfinite(s) or s not in (0,.25,.5,.75,1) for s in values):
        raise ValueError('pair scores must be quarter points in [0,1]')
    signed = [int(round(4*(s-.5))) for s in values]
    distribution = {0: 1}
    for weight in (abs(x) for x in signed if x):
        nxt = defaultdict(int)
        for total, count in distribution.items():
            nxt[total+weight] += count
            nxt[total-weight] += count
        distribution = nxt
    observed = sum(signed)
    p = sum(count for total,count in distribution.items() if total >= observed)/sum(distribution.values())
    score = sum(values)/len(values)
    return {'pass': bool(score > .5 and p <= alpha), 'p_value': p,
            'score': score, 'pairs': len(values), 'alpha': alpha}


def challenger_game_ids(n_games, fraction, iteration):
    if not 0 <= fraction <= 1:
        raise ValueError('challenger fraction must be in [0,1]')
    ids=list(range(n_games))
    random.Random(24000+iteration).shuffle(ids)
    return set(ids[:int(round(n_games*fraction))])


def collect_mixed_selfplay(collector, cfg, n_games, temp_moves, workers,
                           champion_path, challenger_path, fraction,
                           use_server=False, server_device='cpu'):
    """Two frozen actors, proportionate sparring/PFSP in each sub-pool."""
    n_chall=int(round(n_games*fraction))
    examples=[]; results={}
    for count,path,role in ((n_chall,challenger_path,'challenger'),
                            (n_games-n_chall,champion_path,'champion')):
        if not count: continue
        ex,stats=collector(None,cfg,count,temp_moves,min(workers,count),
                           weights_path=path,use_server=use_server,
                           server_device=server_device)
        examples.extend(ex)
        for key,value in stats.items():
            if not isinstance(value,(int,float)):
                raise TypeError(f'non-numeric self-play counter: {key}')
            results[key]=results.get(key,0)+value
        results[f'actor_{role}_games']=count
    return examples,results
