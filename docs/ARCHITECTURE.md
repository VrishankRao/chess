# Architecture and training flow

`game.py` wraps chess rules and encodes boards from the side-to-move perspective. The current champion uses 30 planes, including tactical attack/defense features. `model.py` supplies a 6×64 residual network with squeeze-and-excitation, a convolutional policy head over 4,096 from/to actions, a WDL value head, and auxiliary heads.

`mcts.py` performs PUCT search, optional leaf batching, history-aware repetition handling, and tree/cache reuse. Search priors are masked to legal moves. Tactical, king-safety and finishing guards affect the deployed engine's choices. The system's strength therefore reflects the complete guarded search engine, not just policy top-1 accuracy.

`selfplay.py` generates position targets from games. `parallel.py` distributes CPU search across spawned workers. `replay.py` stores examples; `train.py` computes losses and performs updates; `loop.py` keeps the challenger persistent while the incumbent remains the selection baseline. An opt-in actor mixture can generate games from both frozen-per-phase snapshots.

KL anchoring/rehearsal use the included supervised teacher and human-game records. V20 adds score/TD/AV/endgame features, but those additions are experimental and did not establish stronger play. The runtime model class contains V20 heads even for v19; `load_inference_weights` allows missing score/AV heads because `forward_inf` does not use them. It never pads or substitutes policy/value/search weights.

Promotion tests must compare a frozen challenger with a frozen incumbent using distinct paired openings. They must not combine evidence from changing model weights. The paired test is a one-sided exact sign-flip test under exchangeable engine labels within independent opening pairs. A successful test does not automatically establish an absolute Elo or protect against repeated-test selection effects.
