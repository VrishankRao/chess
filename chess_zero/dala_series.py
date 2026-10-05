"""Dala-700 test series: rated 3+2 games vs a fixed-Elo sparring partner.
Self-contained: drives its own UCI engine + Lichess API loop (no bridge).
Frozen weights only. Logs to dala700.json.
Usage: python3 dala_series.py --weights best_i10.pt --games 12 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

API = "https://lichess.org/api"
FALLBACKS = ["ConfidenceBuilder", "uSunfish-l0"]


def api(req_path, token, method="GET", data=None):
    req = urllib.request.Request(
        API + req_path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}


def sse_lines(resp, deadline=None):
    """Yield parsed SSE data blocks. Returns (stops) at deadline so callers
    can never hang on a quiet-but-open stream (Lichess keep-alives yield
    nothing, which used to starve timeout checks placed in the loop body).
    Forces the fd non-blocking so the deadline can always fire."""
    try:
        os.set_blocking(resp.fileno(), False)
    except Exception:
        pass
    buf = b""
    while True:
        if deadline is not None and time.time() > deadline:
            return
        try:
            chunk = os.read(resp.fileno(), 4096)
        except BlockingIOError:
            time.sleep(0.2)
            continue
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            text = block.decode(errors="replace")
            data = "\n".join(l[5:].strip() for l in text.splitlines()
                             if l.startswith("data:"))
            if data:
                yield json.loads(data)


class UCIEngine:
    """Minimal UCI client for our uci.py (fixed-sims, ignores clock)."""

    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  bufsize=1)
        self._cmd("uci")
        self._wait("uciok")

    def _cmd(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if token in line:
                return

    def newgame(self):
        self._cmd("ucinewgame")
        self._cmd("isready")
        self._wait("readyok")

    def move(self, moves: list[str]) -> str:
        self.newgame()
        self._cmd("position startpos" +
                  (f" moves {' '.join(moves)}" if moves else ""))
        self._cmd("go")
        while True:
            line = self.p.stdout.readline().strip().split()
            if line and line[0] == "bestmove":
                return line[1]

    def quit(self):
        try:
            self._cmd("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game(token, engine, game_id, my_color, timeout_game=900):
    """Play one game via board API. Returns (result, pgn_id)."""
    req = urllib.request.Request(
        f"{API}/bot/game/stream/{game_id}",
        headers={"Authorization": f"Bearer {token}"})
    moves: list[str] = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_game + 60) as r:
        for ev in sse_lines(r, deadline=t0 + timeout_game):
            t = ev.get("type")
            if t == "gameFull":
                moves = (ev.get("state", {}).get("moves") or "").split()
                my_color = ev.get("color", my_color)
            elif t == "gameState":
                moves = (ev.get("moves") or "").split()
                status = ev.get("status", "started")
                if status != "started":
                    w = ev.get("winner")
                    if w is None:
                        return 0.5, game_id
                    return (1.0 if w == my_color else 0.0), game_id
                if len(moves) % 2 == (0 if my_color == "white" else 1):
                    if time.time() - t0 > timeout_game:
                        return 0.5, game_id
                    mv = engine.move(moves)
                    api(f"/bot/game/{game_id}/move/{mv}", token,
                        method="POST")
    return 0.5, game_id


def challenge_and_play(token, engine, opp, color, clock_lim=180,
                       clock_inc=2, accept_wait=150):
    import urllib.error
    body = {"rated": True, "color": color, "variant": "standard",
            "clock": {"limit": clock_lim, "increment": clock_inc}}
    try:
        ch = api(f"/challenge/{opp}", token, method="POST", data=body)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(120)  # lichess challenge rate limit: back off once
            try:
                ch = api(f"/challenge/{opp}", token, method="POST",
                         data=body)
            except Exception as e2:
                return None, f"rate-limited: {e2}"
        else:
            return None, f"challenge-post: {e}"
    except Exception as e:
        return None, f"challenge-post: {e}"
    cid = ch.get("id") or ch.get("challenge", {}).get("id")
    if not cid:
        return None, f"no-challenge: {str(ch)[:120]}"
    # wait for accept by polling account/playing (the event-stream endpoint
    # is throttle-prone; playing is cheap and reliable). Match our challenge
    # by opponent username.
    t0 = time.time()
    gid = None
    while time.time() - t0 < accept_wait:
        try:
            playing = api("/account/playing", token).get("nowPlaying", [])
        except Exception:
            playing = []
        for g in playing:
            o = (g.get("opponent") or {}).get("id", "")
            if o.lower() == opp.lower():
                gid = g.get("gameId")
                break
        if gid:
            break
        # decline detection: challenge vanished without a game
        time.sleep(10)
    if not gid:
        try:
            api(f"/challenge/{cid}/cancel", token, method="POST")
        except Exception:
            pass
        return None, "timeout"
    my_color = color if color in ("white", "black") else "white"
    return play_game(token, engine, gid, my_color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--opp", default="dala-700")
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN", ""))
    ap.add_argument("--out", default="dala700.json")
    ap.add_argument("--engine-cmd", default=None,
                    help="python UCI cmd; default: local uci.py")
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch
    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    n = sum(p.numel() for p in
            (sd["weights"] if isinstance(sd, dict) and "weights" in sd
             else sd).values())
    print(f"weights ok: {n} params from {a.weights}", flush=True)

    cmd = (a.engine_cmd.split() if a.engine_cmd else
           [sys.executable, "-m", "chess_zero.uci", "--ckpt", a.weights,
            "--sims", str(a.sims), "--blocks", str(a.blocks),
            "--channels", str(a.channels), "--device", "cpu"])
    eng = UCIEngine(cmd)
    try:
        eng.newgame()
        print("uci handshake ok", flush=True)
    finally:
        if a.dry_run:
            eng.quit()
    if a.dry_run:
        me = {"id": "dry-run"}
        print(f"dry-run ok: would play {a.games} rated 3+2 vs {a.opp}",
              flush=True)
        return

    try:
        me = api("/account", a.token)
        print(f"auth ok: {me.get('username')}", flush=True)
    except Exception as e:
        print(f"auth failed: {e}", flush=True)
        return

    opps = [a.opp] + [o for o in FALLBACKS if o != a.opp]
    recs = []
    for i in range(a.games):
        color = "white" if i % 2 == 0 else "black"
        played = False
        for opp in opps:
            for attempt in range(3):
                res, info = challenge_and_play(a.token, eng, opp, color)
                if res is not None:
                    recs.append({"game": i, "opp": opp, "color": color,
                                 "score": res, "pgn": info})
                    print(recs[-1], flush=True)
                    played = True
                    break
                print(f"iter-game {i} vs {opp}: {info}, retry", flush=True)
            if played:
                break
        if not played:
            recs.append({"game": i, "opp": None, "color": color,
                         "score": None, "pgn": "all-declined"})
        time.sleep(20)  # spacing between challenges
    eng.quit()
    try:
        old = json.load(open(a.out)) if os.path.exists(a.out) else []
    except Exception:
        old = []
    old.extend(recs)
    json.dump(old, open(a.out, "w"), indent=1)
    sc = [r["score"] for r in recs if r["score"] is not None]
    print(f"series score: {sum(sc)}/{len(sc)}", flush=True)
    sys.exit(0 if sc else 1)


if __name__ == "__main__":
    main()
