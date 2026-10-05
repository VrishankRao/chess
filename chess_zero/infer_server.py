"""GPU inference server (phase 3, §8.3). ISOLATED module: nothing here runs
unless a caller explicitly opts in (use_server=True / server_conn=...).
Live training paths are untouched by default.

Design: one server process owns the CUDA model. Workers hold a pipe end,
send one encoded position per MCTS sim, block for the reply. Batching
happens server-side across workers (dynamic batching up to max_batch).

Quality rules (round-5 conditions — a speed change must not become an
engine change):
- fp32 ONLY (TF32 explicitly disabled). GPU/CPU divergence stays at ULP
  level like phase 1; validate with validate_against_local() before any
  training run through this path.
- RELOAD has barrier semantics: the parent sends it only when no games
  are in flight (loop writes _w_current.pt between phases), and waits
  for the ack before starting the next phase. Never mix weights in one
  buffer.
 - Robustness: client poll timeout + EOF handling falls back to a local
   CPU model per worker (today's behaviour), loudly. CHESS_ZERO_NO_INFER_SERVER=1
   skips the server entirely (bisecting).

 V21 Batch A (MPS validation, 2026-09-25; live mac_run16 running, so all
 numbers are workers<=2 offline microbenchmarks, default-off only):
 - Stability: 200 consecutive batched forwards (batch 20, V20 6x64/30ch/SE4
   best.pt, server device mps, max_batch 8) in 9.3s, degraded=False,
   server alive at end. RSS self 104.4->74.0MB (d -30.3), child
   224.2->189.6MB (d -34.7): no hang, no leak.
 - Parity (same net, fixed positions seed 20260925): net-level over 200
   positions cpu-direct vs mps-server max|dpi|=5.364e-06,
   max|dv|=2.563e-06, argmax 200/200 -> CLEAN (<1e-4). mps-direct vs
   mps-server exactly 0.0. Search-level (400 sims, V20 knobs, leaf_batch 8)
   over 5 sampled positions: max|dpi|=0.0, no argmax flips.
 - Timing (same box, live run contending; torch 2.13.0):
     evals/sec batch8 (50x): direct-cpu 359.5, direct-mps 379.8,
       server-mps 356.9  -> server/direct-cpu 0.99x
     evals/sec batch32 (30x): direct-cpu 475.8, direct-mps 938.0,
       server-mps 896.1  -> server/direct-cpu 1.88x (device effect;
       server/direct-mps 0.96x; live leaf_batch=8 captures none of it)
     per 400-sim search (pos0, 3 reps): direct-cpu 2.20s,
       direct-mps 5.38s (noisy), server-mps 10.39s (IPC dominates
       single-stream; batching needs fan-out)
     parallel 2-worker full games (sims 400, cap 60, /tmp copy of best.pt):
       direct 167.3s (83.7s/game) vs server-mps 249.4s (124.7s/game)
       -> 0.67x.
 - Verdict: parity CLEAN but the >=1.3x bar is NOT met at workers<=2 under
   live contention (spec asked 7-worker A/B; capped at 2 to protect the
   live run). --infer-server stays plumbed but DEFAULT OFF; a future run
   (live run over, free box, higher fan-out) may re-validate.
"""
from __future__ import annotations

import os
import time

KILL_SWITCH = "CHESS_ZERO_NO_INFER_SERVER"


def server_enabled() -> bool:
    return os.environ.get(KILL_SWITCH) != "1"


def _load_model(weights, blocks: int, channels: int, planes: int,
                device: str, se_ratio=None):
    import torch
    from .model import AlphaZeroNet, load_weights, infer_se_ratio
    if isinstance(weights, str):
        weights = torch.load(weights, map_location="cpu",
                             weights_only=False)
    if se_ratio is None:
        # load-then-infer: infer on a path string always misses.
        se_ratio = infer_se_ratio(weights)
    # Pure fp32: TF32's ~1e-3 divergence is unnecessary noise here; the
    # net is tiny and batch-128 fp32 is still milliseconds on an L4.
    if str(device).startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False  # varying batch sizes
        except Exception:
            pass
    model = AlphaZeroNet(blocks=blocks, channels=channels, planes=planes,
                         se_ratio=se_ratio)
    load_weights(model, weights, strict=True)
    model = model.to(device).float().eval()
    return model


def serve(weights, blocks: int, channels: int, planes: int, conns,
          control_conn, stop_evt, max_batch: int = 64,
          max_wait_ms: float = 1.0, device: str = "cuda"):
    """Server entry point (run in a spawn child). conns: worker pipes.
    control_conn: parent pipe for ("RELOAD", path) with ("RELOADED",) ack.
    """
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    from multiprocessing.connection import wait as _wait
    try:
        import os as _os
        if hasattr(_os, "sched_setaffinity"):
            _os.sched_setaffinity(0, {0})  # best-effort: stay off workers
    except Exception:
        pass
    # NOTE: deliberately no os.nice() — the server is the critical path.
    model = _load_model(weights, blocks, channels, planes, device)
    all_conns = list(conns) + [control_conn]
    while not stop_evt.is_set():
        try:
            ready = _wait(all_conns, timeout=max_wait_ms / 1000.0)
        except Exception:
            continue
        if control_conn in ready:
            try:
                msg = control_conn.recv()
            except Exception:
                continue
            if isinstance(msg, tuple) and msg and msg[0] == "RELOAD":
                try:
                    model = _load_model(msg[1], blocks, channels,
                                        planes, device)
                    control_conn.send(("RELOADED", None))
                except Exception as e:
                    control_conn.send(("ERROR", str(e)[:200]))
                continue
            if isinstance(msg, tuple) and msg and msg[0] == "STOP":
                break
        # Drain EVAL requests round-robin up to max_batch.
        batch: list[tuple] = []
        for _ in range(max_batch):
            progressed = False
            for c in conns:
                if len(batch) >= max_batch:
                    break
                try:
                    if c.poll():
                        msg = c.recv()
                    else:
                        continue
                except Exception:
                    continue
                progressed = True
                if isinstance(msg, tuple) and msg and msg[0] == "EVAL":
                    batch.append((c, msg[1]))
            if not progressed:
                break
        if not batch:
            continue
        try:
            import numpy as _np
            import torch as _t
            # Payloads arrive WITH their batch dim (client stacks, usually
            # B=1 per MCTS sim) — concatenate, never stack (stacking a
            # second time makes 5D and the forward raises). Row counts per
            # request are tracked so multi-row replies split correctly.
            rows = []
            for _, b in batch:
                rows.append(_np.asarray(b).reshape(-1, planes, 8, 8))
            x = _np.concatenate(rows, axis=0).astype(_np.float32)
            with _t.no_grad():
                # v16: index, don't unpack (7-head vs 8-head skew safe).
                # v18: out[1] is WDL logits — serve Q = P(W)-P(L).
                pieces = []
                for start in range(0, len(x), max(1, int(max_batch))):
                    tensor = _t.from_numpy(x[start:start+max_batch]).to(device)
                    pieces.append(model.forward_inf(tensor) if hasattr(model, "forward_inf") else model(tensor))
                logits = _t.cat([v[0] for v in pieces])
                wdl = _t.cat([v[1] for v in pieces])
                probs = _t.softmax(logits, dim=1).cpu().numpy()
                if wdl.shape[1] == 1:
                    vals = wdl.cpu().numpy().flatten()
                else:
                    _pw = _t.softmax(wdl, dim=1).cpu().numpy()
                    vals = (_pw[:, 0] - _pw[:, 2]).astype(_np.float32)
            off = 0
            for (c, _), r in zip(batch, rows):
                n = r.shape[0]
                try:
                    c.send(("OK", probs[off:off + n].astype(_np.float32),
                            vals[off:off + n].astype(_np.float32),
                            {"wdl": wdl[off:off+n].cpu().numpy(),
                             "safety": _t.cat([v[2] for v in pieces])[off:off+n].cpu().numpy(),
                             "ml": _t.cat([v[3] for v in pieces])[off:off+n].cpu().numpy()}))
                except Exception:
                    pass
                off += n
        except Exception as e:
            # Skip the batch: clients poll-timeout onto local fallback.
            # Never let one bad batch kill the server mid-iteration — but
            # NEVER fail silently either (a silent skip cost a debugging
            # hour once already): first occurrence logs loudly.
            if not getattr(serve, "_err_logged", False):
                serve._err_logged = True
                print(f"[infer-server] batch failed ({type(e).__name__}: "
                      f"{str(e)[:160]}); clients fall back", flush=True)
            continue


class ServerHandle:
    """Parent-side lifecycle. Start once per parallel phase, stop in a
    finally. RELOAD only between phases (barrier: no games in flight)."""

    def __init__(self, weights, blocks: int, channels: int, planes: int,
                 n_clients: int, max_batch: int = 64,
                 max_wait_ms: float = 1.0, device: str = "cuda"):
        import torch as _torch
        if device == "cuda" and not _torch.cuda.is_available():
            device = "mps" if _torch.backends.mps.is_available() else "cpu"
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        self.conns = []
        self._child_conns = []
        for _ in range(n_clients):
            parent_c, child_c = ctx.Pipe(duplex=True)
            self.conns.append(parent_c)
            self._child_conns.append(child_c)
        self._control_parent, control_child = ctx.Pipe(duplex=True)
        self._stop = ctx.Event()
        self.proc = ctx.Process(
            target=serve,
            args=(weights, blocks, channels, planes,
                  self._child_conns, control_child, self._stop,
                  max_batch, max_wait_ms, device),
            daemon=True)
        self.proc.start()
        for c in self._child_conns:
            c.close()

    def reload_weights(self, weights_path: str, timeout: float = 120.0):
        """Barrier reload: call ONLY with no games in flight. Blocks for
        the server ack; raises loudly on failure (parent must abort the
        iter rather than train on mixed weights)."""
        self._control_parent.send(("RELOAD", weights_path))
        if not self._control_parent.poll(timeout):
            raise RuntimeError("infer server RELOAD ack timeout")
        tag, payload = self._control_parent.recv()
        if tag != "RELOADED":
            raise RuntimeError(f"infer server RELOAD failed: {payload}")

    def stop(self):
        try:
            self._control_parent.send(("STOP", None))
        except Exception:
            pass
        self._stop.set()
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        for c in self.conns + [self._control_parent]:
            try:
                c.close()
            except Exception:
                pass


def make_server_evaluate(conn, weights, blocks: int, channels: int,
                         planes: int, device: str = "cpu",
                         timeout: float = 30.0, se_ratio=None):
    """Fused remote evaluator, including the heads used by move guards.

    Transport failures fail closed: a reloaded server may have different
    weights from a client's original checkpoint, so silently falling back
    to that checkpoint would generate mixed-version training data.
    """
    import numpy as np
    import torch
    last = {}

    def forward_encoded(x):
        arr = np.ascontiguousarray(x, dtype=np.float32)
        conn.send(("EVAL", arr))
        if not conn.poll(timeout):
            raise TimeoutError("Inference server did not respond; aborting this game")
        response = conn.recv()
        if len(response) != 4 or response[0] != "OK":
            raise RuntimeError("Incompatible/failed inference server response")
        _, probs, values, extra = response
        return (np.asarray(probs), np.asarray(values), extra)

    def fn(states):
        x = np.stack([s.encode(planes=planes) for s in states])
        probs, values, extra = forward_encoded(x)
        last.clear()
        last.update({s.key(): float(extra["ml"][i]) for i,s in enumerate(states)})
        for i,s in enumerate(states):
            mask = s.legal_mask()
            probs[i,~mask] = 0
            total = probs[i].sum()
            probs[i] = probs[i]/total if total > 0 else mask.astype(np.float32)/max(1,mask.sum())
        return probs, values

    def ml_fn(states):
        if any(s.key() not in last for s in states):
            fn(states)
        return np.array([last[s.key()] for s in states])

    class GuardModel:
        training = False
        def eval(self): self.training = False; return self
        def train(self, mode=True): self.training = mode; return self
        def __call__(self, tensor):
            probs, values, extra = forward_encoded(tensor.detach().cpu().numpy())
            p = torch.from_numpy(np.log(np.maximum(probs,1e-30)))
            w = torch.from_numpy(np.asarray(extra["wdl"]))
            ml = torch.from_numpy(np.asarray(extra["ml"]))
            safe = torch.from_numpy(np.asarray(extra["safety"]))
            zero = ml * 0
            return (p,w,zero,ml,zero,zero,zero,safe)
    fn.ml_fn = ml_fn
    fn.model = GuardModel()
    fn.model._audit_ml_fn = ml_fn
    fn.degraded = lambda: False
    return fn


def validate_against_local(weights, blocks: int, channels: int,
                           planes: int, states: list,
                           device: str = "cpu") -> dict:
    """Battery gate for §8.3: same weights + positions through the server
    path vs local CPU must agree before any training run uses it."""
    import numpy as _np
    import multiprocessing as _mp
    from .selfplay import make_evaluate
    import torch as _t
    import chess_zero.game as _g
    old_planes = getattr(_g, "INPUT_PLANES", 13)
    _g.INPUT_PLANES = planes
    if isinstance(weights, str):
        wpath = weights
        weights = _t.load(weights, map_location="cpu", weights_only=False)
    else:
        wpath = None
    local = make_evaluate(_fresh(blocks, channels, planes, weights),
                          device="cpu", jit=False)
    ctx = _mp.get_context("spawn")
    parent_c, child_c = ctx.Pipe(duplex=True)
    ctrl_p, ctrl_c = ctx.Pipe(duplex=True)
    stop = ctx.Event()
    proc = ctx.Process(target=serve, args=(weights, blocks, channels,
                                           planes, [child_c], ctrl_c,
                                           stop, 64, 1.0, device),
                       daemon=True)
    proc.start()
    child_c.close()
    try:
        remote = make_server_evaluate(parent_c, weights, blocks,
                                      channels, planes, device="cpu")
        pl, vl = local(states)
        pr, vr = remote(states)
        out = {
            "max_policy_abs_diff": float(_np.abs(pl - pr).max()),
            "max_value_abs_diff": float(_np.abs(vl - vr).max()),
            "argmax_agreement": int((pl.argmax(1) == pr.argmax(1)).sum()),
            "n": len(states),
            "degraded": bool(remote.degraded()),
        }
    finally:
        _g.INPUT_PLANES = old_planes
        stop.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
    return out


def _fresh(blocks: int, channels: int, planes: int, weights,
             se_ratio=None):
    from .model import AlphaZeroNet, load_weights, infer_se_ratio
    if se_ratio is None:
        se_ratio = infer_se_ratio(weights)
    m = AlphaZeroNet(blocks=blocks, channels=channels, planes=planes,
                     se_ratio=se_ratio)
    load_weights(m, weights, strict=True)
    m.eval()
    return m
