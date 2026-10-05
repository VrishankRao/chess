"""Evaluation-calendar watcher: milestone freezes, dala series, live.pt
cutover, matchmaking windows. Polls the training ckpt dir; never touches
training itself. State in eval_state.json (no double-firing across restarts).
Usage: python3 eval_watch.py --ckpt-dir checkpoints_v4 [--once]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def log(*a):
    print(*a, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def lichess(path, token, method="GET", data=None, timeout=20):
    req = urllib.request.Request(
        "https://lichess.org/api" + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}


def bridge_idle(token) -> bool:
    try:
        d = lichess("/account/playing", token)
        return len(d.get("nowPlaying", [])) == 0
    except Exception:
        return False


def bridge_running(bridge_dir) -> bool:
    pid_file = os.path.join(bridge_dir, "bridge.pid")
    if not os.path.exists(pid_file):
        return False
    try:
        os.kill(int(open(pid_file).read().strip()), 0)
        return True
    except Exception:
        return False


def bridge_stop(bridge_dir):
    pid_file = os.path.join(bridge_dir, "bridge.pid")
    if os.path.exists(pid_file):
        try:
            os.kill(int(open(pid_file).read().strip()), 15)
        except Exception:
            pass
        try:
            os.remove(pid_file)
        except Exception:
            pass


def bridge_start(bridge_dir, venv):
    env = dict(os.environ)
    cmd = ("source " + os.path.join(venv, "bin/activate") +
           f" && cd {bridge_dir} && setsid nohup python3 lichess-bot.py"
           " > bot.log 2>&1 < /dev/null & echo $!")
    pid = subprocess.check_output(cmd, shell=True, executable="/bin/bash",
                                  cwd=bridge_dir).decode().strip().split()[-1]
    open(os.path.join(bridge_dir, "bridge.pid"), "w").write(pid + "\n")
    return pid


def set_matchmaking(bridge_dir, on: bool):
    import yaml
    p = os.path.join(bridge_dir, "config.yml")
    c = yaml.safe_load(open(p))
    c.setdefault("matchmaking", {})["allow_matchmaking"] = bool(on)
    yaml.safe_dump(c, open(p, "w"))


def completed_iters(ckpt_dir):
    import glob
    import re
    out = set()
    for p in glob.glob(os.path.join(ckpt_dir, "history.json")):
        try:
            for e in json.load(open(p)):
                out.add(int(e["iter"]))
        except Exception:
            pass
    for f in glob.glob(os.path.join(ckpt_dir, "iter*.pt")):
        m = re.search(r"iter(\d+)\.pt$", f)
        if m:
            out.add(int(m.group(1)))
    return out


def run_dala(weights, games, opp, token, out, sims=40):
    cmd = [sys.executable, os.path.join(HERE, "dala_series.py"),
           "--weights", weights, "--games", str(games), "--opp", opp,
           "--token", token, "--out", out, "--sims", str(sims)]
    log("SPAWN:", " ".join(cmd))
    return subprocess.call(cmd, cwd=BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--state", default="eval_state.json")
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN", ""))
    ap.add_argument("--opp", default="dala-700")
    ap.add_argument("--dala-games", type=int, default=12)
    ap.add_argument("--bridge-dir", default=os.path.expanduser("~/lichess-bot"))
    ap.add_argument("--venv", default=os.path.expanduser("~/venv312"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--live", default="live.pt",
                    help="stable engine weights for the lobby bridge")
    a = ap.parse_args()

    ckpt = (a.ckpt_dir if os.path.isabs(a.ckpt_dir)
            else os.path.join(BASE, a.ckpt_dir))
    state_p = (a.state if os.path.isabs(a.state)
               else os.path.join(BASE, a.state))
    live_p = (a.live if os.path.isabs(a.live)
              else os.path.join(BASE, a.live))
    try:
        st = json.load(open(state_p))
    except Exception:
        st = {"dalas": [], "windows": [], "live_sha": None,
              "cutover": False, "v1best": None}
    st.setdefault("dalas", [])
    st.setdefault("windows", [])
    st.setdefault("dala_fail_ts", {})
    st.setdefault("battery", [])

    # init live.pt from v1 benchmark if missing (gated cutover later)
    if not os.path.exists(live_p):
        v1 = os.path.join(BASE, "checkpoints", "best.pt")
        if os.path.exists(v1):
            shutil.copy(v1, live_p)
            st["v1best"] = sha(v1)
            log("live.pt init from v1 best.pt")

    def save():
        json.dump(st, open(state_p, "w"), indent=1)

    while True:
        done = completed_iters(ckpt)
        best = os.path.join(ckpt, "best.pt")
        # promotion watch: best.pt hash change -> refresh live.pt, restart
        # bridge while idle (lobby always deploys the proven best).
        if st.get("cutover") and os.path.exists(best):
            h = sha(best)
            if st.get("live_best_sha") != h:
                shutil.copy(best, live_p)
                st["live_best_sha"] = h
                save()
                log("live.pt refreshed to new best")
                if bridge_running(a.bridge_dir):
                    if bridge_idle(a.token):
                        bridge_stop(a.bridge_dir)
                        bridge_start(a.bridge_dir, a.venv)
                        log("bridge restarted on new weights")
                    else:
                        log("bridge busy: restart deferred to next poll")
        # live.pt cutover: v4 best takes over only after proving >= v1
        # (checked via ladder at promotions; until then v1 holds the lobby)
        for it in sorted(done):
            key = f"{os.path.basename(ckpt)}:{it}"
            # value battery (audit round 3 §5.7): per-iter MAE/sign/policy
            # curve so a value-head collapse is caught the iter it happens.
            # Logging only - never gates anything. ~1-2 min CPU.
            if key not in st["battery"]:
                fr_iter = os.path.join(ckpt, f"iter{it}.pt")
                if os.path.exists(fr_iter):
                    try:
                        from .value_battery import _load, run_battery
                        # v14: read arch from the checkpoint itself
                        # (hardcoded 5x128/16ch broke on the 21ch
                        # lineage) — trunk_in is (channels, planes).
                        _sd = torch.load(fr_iter, map_location="cpu",
                                         weights_only=False)
                        _w = _sd.get("weights", _sd)
                        _tw = _w["trunk_in.weight"]
                        _ch, _pl = int(_tw.shape[0]), int(_tw.shape[1])
                        _bl = len({k.split(".")[1] for k in _w
                                   if k.startswith("blocks.")})
                        ev = _load(fr_iter, _bl, _ch, _pl, "cpu")
                        r = run_battery(ev, sims=40)
                        r["iter"] = it
                        with open(os.path.join(ckpt, "vbattery.jsonl"),
                                  "a") as bf:
                            bf.write(json.dumps(r) + "\n")
                        log(f"battery iter {it}: MAE {r['value_mae']} "
                            f"sign {r['value_sign_acc']} "
                            f"policy {r['policy_hits']}/"
                            f"{r['policy_total']}")
                    except Exception as e:
                        log(f"battery iter {it} failed: {e}")
                    st["battery"].append(key)
                    save()
            if it % 10 == 0 and key not in st["dalas"]:
                import time as _t
                last_fail = st["dala_fail_ts"].get(key, 0)
                if _t.time() - last_fail < 6 * 3600:
                    continue  # failed recently: cool down, don't spam lobby
                fr = os.path.join(ckpt, f"best_i{it}.pt")
                if os.path.exists(best):
                    shutil.copy(best, fr)
                    log(f"milestone freeze -> {fr}")
                    rc = run_dala(fr, a.dala_games, a.opp, a.token,
                                  os.path.join(BASE, "dala700.json"))
                    if rc == 0:
                        st["dalas"].append(key)
                        save()
                    else:
                        import time as _t2
                        st["dala_fail_ts"][key] = _t2.time()
                        save()
                    log(f"dala series iter {it} exit={rc}")
                    if rc != 0:
                        continue
                    # cutover check: milestone best vs v1 best (6 games)
                    if not st["cutover"]:
                        v1 = os.path.join(BASE, "checkpoints", "best.pt")
                        if os.path.exists(v1):
                            try:
                                out = subprocess.check_output(
                                    [sys.executable,
                                     os.path.join(BASE, "ladder.py"), fr,
                                     "6", v1], cwd=BASE,
                                    timeout=1800).decode()
                                log("probe:", out.strip().splitlines()[-1])
                                import re as _re
                                m = _re.search(r"score=([0-9.]+)", out)
                                if m and float(m.group(1)) >= 0.5:
                                    shutil.copy(fr, live_p)
                                    st["cutover"] = key
                                    log(f"CUTOVER: lobby now plays {fr}")
                            except Exception as e:
                                log(f"cutover probe failed: {e}")
                            st["cutover_check"] = key
                            save()
            if it % 25 == 0 and key not in st["windows"]:
                st["windows"].append(key)
                save()
                log(f"window opens at iter {it}: matchmaking on")
                set_matchmaking(a.bridge_dir, True)
                if not bridge_running(a.bridge_dir) and bridge_idle(a.token):
                    bridge_start(a.bridge_dir, a.venv)
        save()
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
