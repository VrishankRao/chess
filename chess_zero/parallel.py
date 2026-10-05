"""Process-parallel self-play + arena across CPU cores.

Zero quality change by construction: every game runs the IDENTICAL algorithm
(same MCTS, sims, PUCT, noise, temperature, net weights) as sequential play.
The only difference is independent RNG seeds per worker (more diversity).
Workers run CPU inference (1 torch thread each); the GPU stays reserved for
batched training, which is what it is good at.
"""
from __future__ import annotations

import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

# Spawn (never fork): the parent holds a CUDA model, and forking with a live
# CUDA context kills children. Spawn re-imports cleanly in each worker.
_CTX = mp.get_context("spawn")


def _worker_init():
    torch.set_num_threads(1)
    try:
        # Training workers yield CPU priority so a live lobby engine sharing
        # the box never starves (a starved engine flags — see Sep-6 blitz).
        os.nice(10)
    except Exception:
        pass


def _build_cpu_model(weights, blocks, channels, planes=13, se_ratio=None):
    from .model import AlphaZeroNet, load_weights, infer_se_ratio
    if se_ratio is None:
        se_ratio = infer_se_ratio(weights)
    m = AlphaZeroNet(blocks=blocks, channels=channels, planes=planes,
                     se_ratio=se_ratio)
    load_weights(m, weights, strict=True)
    return m


def _apply_game_settings(cfg_or_tuple):
    """Stamp every name in game.GAME_GLOBALS into this (worker-fresh)
    process. Accepts the 5-tuple or a Config. A multiprocess worker starts
    from import-time defaults, so anything the rules read and this misses
    silently runs defaults (audit round 2: this recurred twice).
    None/short entries mean "inherit the module default" (legacy callers);
    run_training always provides all five, pinned by test."""
    import chess_zero.game as _g
    if isinstance(cfg_or_tuple, tuple):
        vals = dict(zip(_g.GAME_GLOBALS, cfg_or_tuple))
    else:
        c = cfg_or_tuple
        vals = {"ADJUDICATE_MARGIN": getattr(c, "adjudicate_margin", 3.0),
                "NO_PROGRESS_PLIES": getattr(c, "no_progress_plies", 100),
                "ADJUDICATE_MIN_PLY": getattr(c, "adjudicate_min_ply", 0),
                "INPUT_PLANES": getattr(c, "input_planes", 13),
                "ADJUDICATE_VALUES": getattr(c, "adjudicate_values", None)}
    for k in _g.GAME_GLOBALS:
        if k in vals and vals[k] is not None:
            v = vals[k]
            setattr(_g, k, dict(v) if isinstance(v, dict) else v)


def _selfplay_chunk(job) -> tuple[list, dict]:
    weights_path, blocks, channels, planes, cfg, n_games, temp_moves, seed = \
        job[:8]
    # Phase 3 opt-in: 9th tuple element carries a server pipe. When present
    # the worker builds NO local model (saves load time + ~600 MB) and all
    # inference goes through the server with local-CPU fallback.
    server_conn = job[8] if len(job) > 8 else None
    _apply_game_settings(cfg)
    import random as _r
    np.random.seed(seed)
    _r.seed(seed + 1)
    torch.manual_seed(seed + 2)
    from .selfplay import play_game
    if server_conn is not None:
        from .infer_server import make_server_evaluate
        evaluate_fn = make_server_evaluate(server_conn, weights_path,
                                           blocks, channels, planes,
                                           device="cpu")
        model = None
    else:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        # v19 arch-follows-weights: explicit cfg wins when set, else
        # infer from the checkpoint (None -> _build_cpu_model infers).
        model = _build_cpu_model(weights, blocks, channels, planes,
                                 int(getattr(cfg, "se_ratio", 0) or 0)
                                 or None)
        evaluate_fn = None
    examples: list = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "vetoes": 0, "tactics": 0,
               "Tmate": 0, "Tresign": 0, "Tadjudicated": 0,
               "Tadjudicated-draw": 0, "Trules-draw": 0, "Tcap": 0,
               "breadth": 0.0, "ventropy": 0.0, "bgames": 0}
    orm = getattr(cfg, "opening_random_moves", 0)
    rt = getattr(cfg, "resign_threshold", None)
    rm = getattr(cfg, "resign_moves", 3)
    # v13: full-playout (mate-signal) coin per game, drawn from the
    # chunk-seeded RNG (deterministic per seed). Sparring chunks stay
    # truncated — punishment data, not finishing school.
    # v18: fast-game coin (KataGo playout-cap doctrine): cheap value-data
    # games at fast_sims/no-noise. Full-playout wins ties (mate signal
    # outranks speed); sparring never fast (punishment needs teeth).
    fp_frac = float(getattr(cfg, "full_playout_frac", 0.0) or 0.0)
    fast_frac = float(getattr(cfg, "fast_frac", 0.0) or 0.0)
    # v19 D9 PFSP-lite: pool_frac of chunk games vs a frozen champion
    # snapshot (max-min robustness) via the sparring= mechanism (our
    # moves recorded, pool moves free — existing sparring semantics).
    # pool_paths stamped per iter by the loop; empty = all self.
    pool_frac = float(getattr(cfg, "pool_frac", 0.0) or 0.0)
    pool_paths = list(getattr(cfg, "pool_paths", None) or [])
    _pool_opp = None
    if pool_frac > 0.0 and pool_paths:
        try:
            from .loop import agent_from_weights as _afw
            _pp = _r.choice(pool_paths)
            _pool_opp = _afw(_pp, cfg, sims=cfg.sims, device="cpu",
                             noise=0.0)
        except Exception:
            print(f"[pool] checkpoint load failed: {_pp}", flush=True)
            raise RuntimeError("Requested historical opponent failed to load")
    # V20 D4/D5/D6 per-chunk state (Agent B; sparring/punisher chunks
    # skip endgame starts — punishment data, not technique). FENs load
    # once per chunk; TB tables resolve once per chunk (None + loud
    # skip when staging is incomplete). All defaults-off for old cfgs.
    from .loop import _v20_loop as _v20q, endgame_starts_allowed as _ega, \
        _roll_endgame as _re, _roll_playthrough as _rp, \
        _v20_rescore_game as _vrg
    _v20 = bool(_v20q(cfg))
    _eg_fens: list = []
    _tb = None
    _tbm = None
    _n_eg = 0
    _n_pt = 0
    _tb_sum = {"probed": 0, "guarded": 0, "rewritten": 0,
               "deblundered": 0, "skipped": 0, "skipped_book": 0}
    if _v20:
        if bool(_ega(cfg)):
            try:
                from .selfplay import load_endgame_fens as _lef
                _eg_fens = _lef(str(getattr(
                    cfg, "endgame_file", "data_sl/endgames.jsonl")))
            except Exception:
                _eg_fens = []
        try:
            from . import tb_rescore as _tbm
            _tb, _ = _tbm.resolve_tables(
                str(getattr(cfg, "tb_path", "data_tb")))
        except Exception:
            _tb, _tbm = None, None
    for _gi in range(n_games):
        st: dict = {}
        full = bool(fp_frac > 0.0 and _r.random() < fp_frac)
        fast = bool(not full and fast_frac > 0.0
                    and _r.random() < fast_frac)
        # v19 D9 PFSP-lite: pool games run the frozen snapshot as the
        # (unrecorded) sparring side, alternating colors per game.
        _pool_now = bool(_pool_opp is not None
                         and _r.random() < pool_frac)
        _fen = None
        _pt = False
        if _v20 and not _pool_now:
            if _eg_fens and bool(_re(_r, float(getattr(
                    cfg, "endgame_frac", 0.0) or 0.0))):
                _fen = _r.choice(_eg_fens)
                _n_eg += 1
            if bool(_rp(_r, float(getattr(
                    cfg, "playthrough_frac", 0.0) or 0.0))):
                _pt = True
                _n_pt += 1
        if _pool_now:
            ex, res = play_game(
                model, cfg, device="cpu", temp_moves=temp_moves,
                opening_random_moves=orm, resign_threshold=rt,
                resign_moves=rm, evaluate_fn=evaluate_fn,
                sparring=_pool_opp, spar_white=((seed + _gi) % 2 == 0),
                full_playout=full, fast=fast, stats=st)
        else:
            ex, res = play_game(
                model, cfg, device="cpu", temp_moves=temp_moves,
                opening_random_moves=orm, resign_threshold=rt,
                resign_moves=rm, evaluate_fn=evaluate_fn,
                full_playout=full, fast=fast, stats=st,
                start_fen=_fen, playthrough=_pt)
        if _v20 and _tbm is not None:
            try:
                _rs = _vrg(cfg, _tbm, _tb, ex, st, res)
                for _k in _tb_sum:
                    _tb_sum[_k] = _tb_sum.get(_k, 0) + \
                        int(_rs.get(_k, 0) or 0)
            except Exception:
                pass
        examples.extend(ex)
        results[res] += 1
        if _pool_now:
            # sparring-involved: S-keys (punishment data, consistent with
            # sparring chunks; keeps white_share math exact).
            results["S" + res] = results.get("S" + res, 0) + 1
            results["pfsp_pool"] = results.get("pfsp_pool", 0) + 1
        results["vetoes"] += st.get("vetoes", 0)
        results["tactics"] += st.get("tactics", 0)
        results["finishes"] = results.get("finishes", 0) + \
            st.get("finishes", 0)
        results["safety"] = results.get("safety", 0) + \
            st.get("safety", 0)
        results["fast"] = results.get("fast", 0) + \
            st.get("fast", 0)
        tkey = "T" + str(st.get("terminal", "cap"))
        results[tkey] = results.get(tkey, 0) + 1
        # v14 visit breadth: sum of per-game means, averaged at history.
        results["breadth"] += float(st.get("breadth", 0.0))
        results["ventropy"] += float(st.get("ventropy", 0.0))
        results["bgames"] += 1
        # v14 pace metric: total plies per game.
        results["plies"] = results.get("plies", 0) + int(st.get("plies", 0))
    # V20 chunk closeout: TB handle closes; D4/D5/D6 counters merge up
    # (parent union-merges; "positions" is counted by the PARENT from
    # the flat examples list, never here — no double count).
    if _v20:
        try:
            if _tb is not None:
                _tb.close()
        except Exception:
            pass
        results["endgame_starts"] = \
            results.get("endgame_starts", 0) + int(_n_eg)
        results["playthroughs"] = \
            results.get("playthroughs", 0) + int(_n_pt)
        for _k, _v in _tb_sum.items():
            results["tb_" + _k] = results.get("tb_" + _k, 0) + \
                int(_v or 0)
    return examples, results


def _sparring_chunk(job) -> tuple[list, dict]:
    """v8.2: our model (one color) vs a fixed sparring policy (greedy).
    Only our moves are recorded; hanging material gets punished in the
    training data itself. alternates color by spar_white."""
    weights_path, blocks, channels, planes, cfg, n_games, temp_moves, seed, \
        s_off = job[:9]
    server_conn = job[9] if len(job) > 9 else None
    _apply_game_settings(cfg)
    import random as _r
    np.random.seed(seed)
    _r.seed(seed + 1)
    torch.manual_seed(seed + 2)
    from .selfplay import play_game
    from .evaluate import sparring_for_game
    if server_conn is not None:
        from .infer_server import make_server_evaluate
        _ev = make_server_evaluate(server_conn, weights_path,
                                   blocks, channels, planes, device="cpu")
        model = None
    else:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        # v19 arch-follows-weights: explicit cfg wins when set, else
        # infer from the checkpoint (None -> _build_cpu_model infers).
        model = _build_cpu_model(weights, blocks, channels, planes,
                                 int(getattr(cfg, "se_ratio", 0) or 0)
                                 or None)
        _ev = None
    examples: list = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "S1-0": 0, "S0-1": 0,
               "S1/2-1/2": 0, "vetoes": 0, "tactics": 0,
               "Tmate": 0, "Tresign": 0, "Tadjudicated": 0,
               "Tadjudicated-draw": 0, "Trules-draw": 0, "Tcap": 0,
               "breadth": 0.0, "ventropy": 0.0, "bgames": 0}
    orm = getattr(cfg, "opening_random_moves", 0)
    rt = getattr(cfg, "resign_threshold", None)
    rm = getattr(cfg, "resign_moves", 3)
    for k in range(n_games):
        # v9 sparring mix (deterministic by GLOBAL game index): even
        # greedy, odd punisher (check-seeking king pressure). v14 fix:
        # chunk-local k made every 1-game chunk greedy (k=0) — punisher
        # never ran — and broke color alternation. Global ids restore
        # the mix (matches the sequential path).
        # v13 honest compass: sparring games record T* too (mate_rate
        # covers ALL games, not just self-play).
        gidx = s_off + k
        st: dict = {}
        ex, res = play_game(model, cfg, device="cpu", temp_moves=temp_moves,
                            opening_random_moves=orm, resign_threshold=rt,
                            resign_moves=rm, sparring=sparring_for_game(gidx),
                            spar_white=(gidx % 2 == 0), evaluate_fn=_ev,
                            stats=st)
        examples.extend(ex)
        results[res] += 1
        results["S" + res] += 1
        results["vetoes"] += st.get("vetoes", 0)
        results["tactics"] += st.get("tactics", 0)
        results["finishes"] = results.get("finishes", 0) + \
            st.get("finishes", 0)
        results["safety"] = results.get("safety", 0) + \
            st.get("safety", 0)
        tkey = "T" + str(st.get("terminal", "cap"))
        results[tkey] = results.get(tkey, 0) + 1
        # v14 visit breadth (sparring games search our moves too).
        results["breadth"] += float(st.get("breadth", 0.0))
        results["ventropy"] += float(st.get("ventropy", 0.0))
        results["bgames"] += 1
        results["plies"] = results.get("plies", 0) + int(st.get("plies", 0))
    return examples, results


def play_games_parallel(model, cfg, n_games: int, temp_moves: int,
                        workers: int | None = None,
                        weights_path: str | None = None,
                        use_server: bool = False,
                        server_device: str = "cuda") -> tuple[list, dict]:
    """Returns (all_examples, merged_results). Weights broadcast via a FILE
    (not IPC) to avoid torch fd-sharing blowing the fd limit at 31 workers.
    v8.2: the first sparring_frac of games are sparring (model vs greedy,
    alternating colors) via _sparring_chunk; the rest pure self-play.
    use_server (phase 3, opt-in): serve batched GPU inference from one
    server process instead of per-worker CPU models. Default False =
    today's behaviour exactly. workers==1 ignores the flag (log line)."""
    import tempfile
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, n_games))
    spar_frac = float(getattr(cfg, "sparring_frac", 0.0) or 0.0)
    n_spar = min(n_games, int(round(n_games * spar_frac)))
    orm = getattr(cfg, "opening_random_moves", 0)
    rt = getattr(cfg, "resign_threshold", None)
    rm = getattr(cfg, "resign_moves", 3)
    if workers == 1:
        if use_server:
            print("[infer-server] workers==1 ignores use_server "
                  "(no fan-out to batch); running local CPU", flush=True)
        from .selfplay import play_game
        from .evaluate import sparring_for_game
        examples: list = []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "S1-0": 0, "S0-1": 0,
                   "S1/2-1/2": 0, "vetoes": 0, "tactics": 0,
                   "Tmate": 0, "Tresign": 0, "Tadjudicated": 0,
                   "Tadjudicated-draw": 0, "Trules-draw": 0, "Tcap": 0,
                   "breadth": 0.0, "ventropy": 0.0, "bgames": 0}
        fp_frac = float(getattr(cfg, "full_playout_frac", 0.0) or 0.0)
        fast_frac = float(getattr(cfg, "fast_frac", 0.0) or 0.0)
        # v19 D9 PFSP-lite (workers==1 mirror of _selfplay_chunk).
        import random as _rr0
        pool_frac = float(getattr(cfg, "pool_frac", 0.0) or 0.0)
        pool_paths = list(getattr(cfg, "pool_paths", None) or [])
        _pool_opp = None
        if pool_frac > 0.0 and pool_paths:
            try:
                from .loop import agent_from_weights as _afw0
                _pool_opp = _afw0(_rr0.choice(pool_paths),
                                  cfg, sims=cfg.sims, device="cpu",
                                  noise=0.0)
            except Exception:
                _pool_opp = None
        # V20 D4/D5/D6 mirror of _selfplay_chunk (same rules: no
        # endgame starts on sparring/pool games; rescore every game
        # with complete move logs; counters merge up union-style).
        from .loop import _v20_loop as _v20q1, \
            endgame_starts_allowed as _ega1, _roll_endgame as _re1, \
            _roll_playthrough as _rp1, _v20_rescore_game as _vrg1
        _v20 = bool(_v20q1(cfg))
        _eg_fens: list = []
        _tb = None
        _tbm = None
        if _v20:
            if bool(_ega1(cfg)):
                try:
                    from .selfplay import load_endgame_fens as _lef1
                    _eg_fens = _lef1(str(getattr(
                        cfg, "endgame_file", "data_sl/endgames.jsonl")))
                except Exception:
                    _eg_fens = []
            try:
                from . import tb_rescore as _tbm
                _tb, _ = _tbm.resolve_tables(
                    str(getattr(cfg, "tb_path", "data_tb")))
            except Exception:
                _tb, _tbm = None, None
        for i in range(n_games):
            st: dict = {}
            if i < n_spar:
                ex, res = play_game(
                    model, cfg, device="cpu", temp_moves=temp_moves,
                    opening_random_moves=orm, resign_threshold=rt,
                    resign_moves=rm, sparring=sparring_for_game(i),
                    spar_white=(i % 2 == 0), stats=st)
                examples.extend(ex)
                results[res] += 1
                results["S" + res] += 1
            else:
                import random as _rr1
                full = bool(fp_frac > 0.0 and _rr1.random() < fp_frac)
                fast = bool(not full and fast_frac > 0.0
                            and _rr1.random() < fast_frac)
                _pool_now = bool(_pool_opp is not None
                                 and _rr1.random() < pool_frac)
                _fen = None
                _pt = False
                if _v20 and not _pool_now:
                    if _eg_fens and bool(_re1(_rr1, float(getattr(
                            cfg, "endgame_frac", 0.0) or 0.0))):
                        _fen = _rr1.choice(_eg_fens)
                        results["endgame_starts"] = \
                            results.get("endgame_starts", 0) + 1
                    if bool(_rp1(_rr1, float(getattr(
                            cfg, "playthrough_frac", 0.0) or 0.0))):
                        _pt = True
                        results["playthroughs"] = \
                            results.get("playthroughs", 0) + 1
                if _pool_now:
                    ex, res = play_game(
                        model, cfg, device="cpu", temp_moves=temp_moves,
                        opening_random_moves=orm, resign_threshold=rt,
                        resign_moves=rm, sparring=_pool_opp,
                        spar_white=(i % 2 == 0),
                        full_playout=full, fast=fast, stats=st)
                else:
                    ex, res = play_game(
                        model, cfg, device="cpu", temp_moves=temp_moves,
                        opening_random_moves=orm, resign_threshold=rt,
                        resign_moves=rm,
                        full_playout=full, fast=fast, stats=st,
                        start_fen=_fen, playthrough=_pt)
                if _v20 and _tbm is not None:
                    try:
                        _rs = _vrg1(cfg, _tbm, _tb, ex, st, res)
                        for _k, _v in _rs.items():
                            if _k == "skipped_book":
                                results["tb_skipped_book"] = \
                                    results.get("tb_skipped_book", 0) + \
                                    int(_v or 0)
                            elif _k in ("probed", "guarded", "rewritten",
                                        "deblundered", "skipped"):
                                results["tb_" + _k] = \
                                    results.get("tb_" + _k, 0) + \
                                    int(_v or 0)
                    except Exception:
                        pass
                examples.extend(ex)
                results[res] += 1
                if _pool_now:
                    results["S" + res] += 1
                    results["pfsp_pool"] = \
                        results.get("pfsp_pool", 0) + 1
            results["vetoes"] += st.get("vetoes", 0)
            results["tactics"] += st.get("tactics", 0)
            results["finishes"] = results.get("finishes", 0) + \
                st.get("finishes", 0)
            results["safety"] = results.get("safety", 0) + \
                st.get("safety", 0)
            results["fast"] = results.get("fast", 0) + \
                st.get("fast", 0)
            tkey = "T" + str(st.get("terminal", "cap"))
            results[tkey] = results.get(tkey, 0) + 1
            # v14 visit breadth.
            results["breadth"] += float(st.get("breadth", 0.0))
            results["ventropy"] += float(st.get("ventropy", 0.0))
            results["bgames"] += 1
            results["plies"] = results.get("plies", 0) + \
                int(st.get("plies", 0))
        if _v20:
            try:
                if _tb is not None:
                    _tb.close()
            except Exception:
                pass
        return examples, results
    _tmp = None
    if weights_path is None:
        _tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        weights_path = _tmp.name
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                    weights_path)
    base = int(np.random.randint(0, 1_000_000_000))
    # sparring game indices first (alternate colors), then self-play
    spar_ids = list(range(n_spar))
    self_ids = list(range(n_spar, n_games))
    chunks_s, rem_s = divmod(len(spar_ids), workers)
    chunks_p, rem_p = divmod(len(self_ids), workers)
    jobs = []
    # v14 fix: sparring chunks carry their GLOBAL start index. The old
    # per-chunk (k, color) made every 1-game chunk greedy (k=0) — the
    # punisher never ran at 20 games x6 workers — and broke color
    # alternation. Global ids restore the documented 50/50 mix.
    s_off = 0
    for w in range(workers):
        ns = chunks_s + (1 if w < rem_s else 0)
        if ns:
            jobs.append(("spar", weights_path, cfg.blocks, cfg.channels,
                         cfg.input_planes, cfg, ns, temp_moves,
                         base + w * 100003, s_off))
            s_off += ns
        n = chunks_p + (1 if w < rem_p else 0)
        if n:
            jobs.append(("self", weights_path, cfg.blocks, cfg.channels,
                         cfg.input_planes, cfg, n, temp_moves,
                         base + 7 + w * 100003))
    examples_all: list = []
    merged = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "S1-0": 0, "S0-1": 0,
              "S1/2-1/2": 0, "vetoes": 0, "tactics": 0,
              "Tmate": 0, "Tresign": 0, "Tadjudicated": 0,
              "Tadjudicated-draw": 0, "Trules-draw": 0, "Tcap": 0,
              "breadth": 0.0, "ventropy": 0.0, "bgames": 0}
    _server = None
    if use_server:
        # One pipe per JOB (a job = one worker chunk); the server batches
        # across all of them. Created here, torn down in the finally.
        from .infer_server import ServerHandle, server_enabled
        import torch as _t
        if not server_enabled():
            print("[infer-server] kill switch set; running local CPU",
                  flush=True)
        elif server_device == "cuda" and not _t.cuda.is_available():
            print("[infer-server] no CUDA here; running local CPU",
                  flush=True)
        elif server_device == "mps" and not _t.backends.mps.is_available():
            # V21 Batch A (default-off): allow the MPS server path on
            # Apple Silicon. CUDA-default callers still fall back above,
            # so the live run (use_server=False) is untouched.
            print("[infer-server] no MPS here; running local CPU",
                  flush=True)
        else:
            _server = ServerHandle(weights_path, cfg.blocks,
                                   cfg.channels, cfg.input_planes,
                                   n_clients=len(jobs), device=server_device)
            jobs = [j + (_server.conns[i],) for i, j in enumerate(jobs)]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_CTX,
                                 initializer=_worker_init) as ex_:
            for examples, results in ex_.map(_mixed_chunk, jobs):
                examples_all.extend(examples)
                # v14 union-merge: future terminal labels ride along
                # instead of vanishing on fixed keys.
                for k, v in results.items():
                    merged[k] = merged.get(k, 0) + v
    finally:
        if _server is not None:
            _server.stop()
        # v14: remove our own temp weights file (caller-provided paths
        # are never ours to delete).
        if _tmp is not None:
            try:
                os.unlink(_tmp.name)
            except OSError:
                pass
    return examples_all, merged


def _mixed_chunk(job) -> tuple[list, dict]:
    kind = job[0]
    # Phase 3: an 11th/12th trailing element carries the server pipe;
    # forwarded so _selfplay/_sparring_chunk see it (they slice job[:8/9]).
    extra = job[10:] if kind == "spar" else job[9:]
    if kind == "spar":
        _, wp, bl, ch, pl, cfg, n, tm, seed, s_off = job[:10]
        return _sparring_chunk(
            (wp, bl, ch, pl, cfg, n, tm, seed, s_off) + tuple(extra))
    _, wp, bl, ch, pl, cfg, n, tm, seed = job[:9]
    return _selfplay_chunk(
        (wp, bl, ch, pl, cfg, n, tm, seed) + tuple(extra))


def _arena_chunk(job) -> dict:
    spec_a, spec_b, game_ids, cap, seed, opening_moves, settings = job[:7]
    record = job[7] if len(job) > 7 else False
    seed_base = job[8] if len(job) > 8 else 1000
    # Phase 3 opt-in: (conn_for_A, conn_for_B), None where the side is not
    # a net policy. One conn per agent closure — pipes are point-to-point,
    # sharing one between two agents would unroute replies.
    conns = job[9] if len(job) > 9 else (None, None)
    _apply_game_settings(settings)
    import random as _r
    np.random.seed(seed)
    _r.seed(seed + 1)
    from .evaluate import _play_ids, _make_policy
    pa = _make_policy(spec_a, server_conn=conns[0])
    pb = _make_policy(spec_b, server_conn=conns[1])
    return _play_ids(pa, pb, game_ids, cap, opening_moves, record,
                     seed_base, job[10] if len(job) > 10 else None)


def play_match_parallel(policy_a, policy_b, games=8, cap=200,
                        workers: int | None = None,
                        opening_moves: int = 0,
                        game_settings: tuple | None = None,
                        record: bool = False,
                        seed_base: int = 1000,
                        use_server: bool = False,
                        server_device: str = "cuda", opening_lines=None) -> dict:
    """game_settings: (margin, no_progress_plies, min_ply, planes, values)
    stamped into each worker (None = that worker's module defaults).
    record: return (totals, game_records) for PGN autopsy.
    seed_base: opening sample id (iter-derived in training so the test set
    varies across iterations while pairs match within a match).
    use_server (phase 3, opt-in): one batched-GPU server per DISTINCT net
    policy in the match (gate = two: challenger + incumbent). Default
    False = today's behaviour exactly."""
    from .evaluate import play_match
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, games))
    if workers == 1:
        if game_settings is not None:
            _apply_game_settings(game_settings)
        return play_match(policy_a, policy_b, games=games, cap=cap,
                          opening_moves=opening_moves, record=record,
                          seed_base=seed_base, opening_lines=opening_lines)
    from .evaluate import _spec
    sa, sb = _spec(policy_a), _spec(policy_b)
    base = int(np.random.randint(0, 1_000_000_000))
    ids = list(range(games))
    slices = [ids[i::workers] for i in range(workers)]
    jobs = [(sa, sb, s, cap, base + i * 7919, opening_moves,
             game_settings, record, seed_base)
            for i, s in enumerate(slices) if s]
    total = {"wins": 0, "losses": 0, "draws": 0}
    records = {}
    _servers: list = []
    if use_server:
        from .infer_server import ServerHandle, server_enabled
        import torch as _t
        need_a, need_b = sa[0] == "agent", sb[0] == "agent"
        if not (need_a or need_b):
            print("[infer-server] no net policy in match; local CPU",
                  flush=True)
        elif not server_enabled():
            print("[infer-server] kill switch set; local CPU", flush=True)
        elif server_device == "cuda" and not _t.cuda.is_available():
            print("[infer-server] no CUDA here; local CPU", flush=True)
        elif server_device == "mps" and not _t.backends.mps.is_available():
            # V21 Batch A (default-off): MPS match path, same rule as
            # self-play above. Live run (use_server=False) untouched.
            print("[infer-server] no MPS here; local CPU", flush=True)
        else:
            # One server per distinct net (gate serves two weight sets).
            handles: dict[int, object] = {}
            for side, spec in (("a", sa), ("b", sb)):
                if spec[0] == "agent":
                    key = id(spec[1]) if isinstance(spec[1], dict) \
                        else spec[1]
                    if key not in handles:
                        handles[key] = ServerHandle(
                            spec[1], spec[2], spec[3], spec[9]
                            if len(spec) > 9 else 13,
                            n_clients=len(jobs), device=server_device)
            _servers = list(handles.values())
            # Align conns per job: (connA_or_None, connB_or_None).
            per_side: dict[str, object] = {}
            for side, spec in (("a", sa), ("b", sb)):
                if spec[0] == "agent":
                    key = id(spec[1]) if isinstance(spec[1], dict) \
                        else spec[1]
                    per_side[side] = handles[key]
            ca = per_side.get("a")
            cb = per_side.get("b")
            jobs = [j + ((ca.conns[i] if ca else None,
                          cb.conns[i] if cb else None),)
                    for i, j in enumerate(jobs)]
    jobs = [j + ((None, None),) if len(j) == 9 else j for j in jobs]
    jobs = [j + (opening_lines,) for j in jobs]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_CTX,
                                 initializer=_worker_init) as ex_:
            for job, r in zip(jobs, ex_.map(_arena_chunk, jobs)):
                if record:
                    rr, recs = r
                    records.update(zip(job[2], recs))
                else:
                    rr = r
                for k in total:
                    total[k] += rr[k]
    finally:
        for _s in _servers:
            _s.stop()
    return (total, [records[g] for g in sorted(records)]) if record else total


def _endgame_pair_job(job) -> dict:
    """One mirrored endgame pair (challenger white, then challenger
    black) from a fixed FEN. Each worker builds its OWN agents (fresh
    TT/history isolation — _play_from_fen resets per game on top; no
    shared mutable state — weights arrive by pickle/file copy) and
    seeds each leg via loop._endgame_game_seed(seed_base, pair_idx,
    leg), identical to the sequential loop. Returns {"W","L","D"}
    challenger-relative tallies for the 2 games. Never returns
    examples (buffer/training untouched). Top-level so spawn can
    pickle it."""
    (chall_w, incumb_w, cfg, sims, rust_tree, fen, pair_idx, seed_base,
        cap) = job[:9]
    _dev = str(job[9]) if len(job) > 9 else "cpu"
    _apply_game_settings(cfg)
    import torch as _t
    _t.set_num_threads(1)
    from .loop import (agent_from_weights as _afw, _play_from_fen as _pff,
                       _weights_dict as _wd,
                       _endgame_game_seed as _gs,
                       _seed_endgame_rngs as _sr)
    _rt = bool(rust_tree or getattr(cfg, "rust_tree", False))
    ch = _afw(_wd(chall_w), cfg, sims=int(sims), device=_dev,
              noise=0.0, rust_tree=_rt)
    inc = _afw(_wd(incumb_w), cfg, sims=int(sims), device=_dev,
               noise=0.0, rust_tree=_rt)
    _W = _L = _D = 0
    _sr(_gs(seed_base, pair_idx, 0))
    _w1, _ = _pff(ch, inc, str(fen), int(cap))
    if _w1 == 1.0:
        _W += 1
    elif _w1 == -1.0:
        _L += 1
    else:
        _D += 1
    _sr(_gs(seed_base, pair_idx, 1))
    _w2, _ = _pff(inc, ch, str(fen), int(cap))
    if _w2 == -1.0:
        _W += 1
    elif _w2 == 1.0:
        _L += 1
    else:
        _D += 1
    return {"W": int(_W), "L": int(_L), "D": int(_D)}


def play_endgame_pairs_parallel(chall_w, incumb_w, cfg, fens: list,
                                sims: int = 400,
                                device: str = "cpu",
                                rust_tree: bool = False,
                                seed_base: int = 2000, cap: int = 300,
                                workers: int | None = None) -> dict:
    """Fan out mirrored endgame pairs, one job per FEN, across the
    existing spawn ProcessPool (same _CTX/_worker_init discipline as
    play_match_parallel). Per-game explicit seeds (seed_base + pair
    index + color leg) make WLD BIT-IDENTICAL to the sequential loop
    on fixed FENs. Workers rebuild agents on the CALLER's device
    (threaded through the job; same as the sequential parent) — CPU
    in all tests and the live run. Returns {"W","L","D"} challenger-relative.
    Raises on any failure so the caller can fall back to sequential
    (never silently degrades a promotion measurement)."""
    _fens = list(fens or [])
    if not _fens:
        return {"W": 0, "L": 0, "D": 0}
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(int(workers), len(_fens)))
    jobs = [(chall_w, incumb_w, cfg, int(sims), bool(rust_tree),
             str(_f), int(_pi), int(seed_base), int(cap), str(device))
            for _pi, _f in enumerate(_fens)]
    if workers == 1:
        # Inline (no pool): same per-pair jobs, same seeds — parity path
        # for tests that want the parallel code without spawning.
        _W = _L = _D = 0
        for _j in jobs:
            _r = _endgame_pair_job(_j)
            _W += int(_r.get("W", 0))
            _L += int(_r.get("L", 0))
            _D += int(_r.get("D", 0))
        return {"W": _W, "L": _L, "D": _D}
    _W = _L = _D = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=_CTX,
                             initializer=_worker_init) as ex_:
        for _r in ex_.map(_endgame_pair_job, jobs):
            _W += int(_r.get("W", 0))
            _L += int(_r.get("L", 0))
            _D += int(_r.get("D", 0))
    return {"W": _W, "L": _L, "D": _D}
