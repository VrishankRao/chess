"""Original learning-curve plot from checkpoints/history.json (M4 deliverable)."""
from __future__ import annotations

import json
import os


def main(path="checkpoints/history.json", out="checkpoints/curves.png"):
    if not os.path.exists(path):
        print(f"no {path} yet — run chess_zero.loop first")
        return
    hist = json.load(open(path))
    it = [h["iter"] for h in hist]
    loss = [h["loss"] for h in hist]
    elo_r = [h["elo_random"] for h in hist]
    elo_g = [h["elo_greedy"] for h in hist]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(it, loss, marker="o")
    ax[0].set_title("total loss per iter")
    ax[0].set_xlabel("iter")
    ax[1].plot(it, elo_r, marker="o", label="vs random")
    ax[1].plot(it, elo_g, marker="s", label="vs greedy")
    ax[1].set_title("arena Elo diff")
    ax[1].set_xlabel("iter")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    print("iter loss eloR eloG")
    for h in hist:
        print(h["iter"], h["loss"], h["elo_random"], h["elo_greedy"])


if __name__ == "__main__":
    main()
