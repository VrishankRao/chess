# Chess Zero

A chess-only neural engine and training project. It combines a residual policy/value network, Monte Carlo tree search (MCTS), supervised warm-start data, self-play, replay training, and a UCI interface for chess GUIs.

## Current model

`weights/champion.pt` contains the v19 champion's exact learned weights: 6 residual blocks, 64 channels, squeeze-and-excitation, 30 input planes, and a WDL value head. It is a weights-only export, not a full optimizer/replay resume checkpoint. `ASSETS.json` records file hashes and architecture details.

This is the strongest demonstrated champion available in this project. Later v23 iteration 25 and 35 checkpoints scored 46.1% and 40.6% against it in 64-game comparisons using matched search settings. They are not included as deployment weights. These results do not establish an absolute Elo rating.

## Setup

Python 3.12 or newer is required. Create an environment from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`requirements.txt` is an alternative to editable installation. `requirements-tested.txt` records the exact versions used for export verification. Install PyTorch appropriate to your CUDA system if needed. Apple Silicon can use `mps` for gradient training; CPU inference is the default for parallel search. CPU-only training also works but is slower.

The Python implementation works without Rust. Optional bitboard move generation:

```sh
python -m pip install 'maturin>=1,<2'
maturin develop --manifest-path rust/chesscore/Cargo.toml --release
```

This requires a Rust toolchain and, on macOS, command-line build tools. The `mctscore` Rust tree is experimental and remains disabled by default; a prior training-scale panic is unresolved. Do not enable `--rust-tree` as a default training setting.

## Play the champion

```sh
./scripts/run_engine.sh
```

This starts a UCI engine, not an interactive board. To check the protocol, enter `uci` and `isready`; to exit, enter `quit`. The first search includes model loading. For a terminal chess game, use a UCI-capable GUI or client.

Banksia expects a native executable on macOS. Build the portable launcher:

```sh
mkdir -p build
cc scripts/engine_launcher.c -o build/chess-zero-engine
```

Add the absolute path to `build/chess-zero-engine` in your GUI. Keep this executable inside the repository's `build` directory so it can locate the script and weights. It automatically uses `.venv/bin/python`; override with `CHESS_ZERO_PYTHON` if necessary. No machine-specific absolute path is embedded in the source. On Windows, configure the GUI to run the equivalent `python -m chess_zero.uci` command or use a platform-specific launcher.

The champion launcher uses CPU search, 400 simulations, leaf batch 8, opening book, tactical guards, and matched v19 search settings. Search budgets respond to UCI time controls. The inference loader permits only absent V20 training heads; it requires exact policy, value, safety, moves-left and shared tensors. Training and deployment checkpoints are not interchangeable across arbitrary architectures.

## Train a new run

Run commands from the repository root. This example starts a new v19-style run from champion weights. It creates a fresh optimizer and replay buffer; it does **not** resume the champion's historical optimizer state:

```sh
python -u -m chess_zero.fullrun \
  --v19 --device mps --workers 4 --games 20 --iters 40 --steps 300 \
  --arena 12 --eval-sims 400 --gate-every 5 --weak-diagnostics-every 10 \
  --resume weights/champion.pt --start_iter 1 --ckpt_dir runs/v19-control
```

Use `--device cpu` on CPU machines or `--device cuda` with a CUDA PyTorch installation. Adjust worker count to RAM and CPU capacity; more workers are not always faster. Do not run heavy training while expecting responsive GUI play on a small machine.

`--steps 300` is an upper limit. Replay occupancy and sample-reuse caps can reduce the actual number of updates substantially. Watch `train_steps`, `cum_steps`, `lr`, phase timings, and match results in `runs/v19-control/history.json`. Falling loss alone is not proof of stronger play.

Full promotion evaluations run every five iterations and on the final iteration; extra weak-bot diagnostics run every ten. Evaluation uses exact paired tests and separate endgame/ancestor checks where configured. The built-in book provides only 18 held-out opening pairs, so a larger requested game count alone does not produce more independent openings. For a larger reserved evaluation suite, supply a JSON file with `{"lines": [["e2e4", "e7e5", "g1f3"], ...]}` and configure `--gate-book-file`, `--gate-games`, and `--gate-min-pairs`. Each pair uses one opening with engines playing both colors. `--gate-alpha` is per test; account for repeated tests when designing an experiment.

Experimental controls are available but are not established strength improvements:

```sh
# Add to a separately named controlled run, changing one variable at a time:
--challenger-selfplay-frac 0.5
--lr-warmup-steps 200
```

The default actor is champion-only and default warmup is 1,000 updates. The 50/50 actor setting generates half the games from the current challenger and half from the incumbent. The champion remains frozen until a valid promotion. Keep the supplied champion unchanged and save all new runs under `runs/`.

### Resume your own run

```sh
python -u -m chess_zero.fullrun \
  --v19 --device mps --workers 4 --games 20 --iters 30 --steps 300 \
  --arena 12 --eval-sims 400 --gate-every 5 --weak-diagnostics-every 10 \
  --resume runs/v19-control/iter10.pt --start_iter 11 \
  --ckpt_dir runs/v19-control
```

This runs iterations 11–40. Resume from your own `iterN.pt` to restore optimizer, replay, RNG and progress counters, using the same architecture and experiment settings. Do not replace `--resume` with the supplied weights-only champion and expect exact continuation.

### Warm-start from supervised data

The teacher checkpoint is already included at `checkpoints_warm/warmstart.pt`. To build another warm-start rather than overwrite it:

```sh
python -m chess_zero.warmstart \
  --games data_sl/games.jsonl --epochs 3 --batch 256 \
  --max-positions 200000 --device mps --out runs/new-warmstart.pt
```

`--v20` enables the experimental V20 recipe; V20/V23 did not demonstrate superiority to the v19 champion. Optional AV labels are provided in `data_sl/av_15k.jsonl` for `--av-table`. Syzygy tablebases and a Stockfish executable are optional external assets and are not included. V20 tablebase rescore skips when tables are missing. Inspect startup logs rather than assuming an optional feature is active.

## Included files

- `chess_zero/`: model, board encoding, search, replay, training, evaluation, UCI, data builders and experimental helpers.
- `weights/champion.pt`: current deployment champion, weights only.
- `checkpoints_warm/warmstart.pt`: supervised teacher/warm-start checkpoint.
- `data_sl/`: human-game records, puzzles, endgame positions and AV labels used by this project.
- `rust/`: optional move-generation core and experimental tree sources, including dependency lockfiles.
- `tests/`: regression, search parity, and integration checks.
- `scripts/`: portable UCI launchers.
- `docs/`: architecture and data notes.

Historical handoffs, credentials, Lichess bridge configuration, personal paths, logs, compiled binaries, old checkpoints, tablebases, and local-only experiment orchestration are excluded. No training automatically starts on installation.

## Checks

```sh
python -m pip install 'pytest>=8'
python -m pytest -q tests/test_v19_inference_loader.py tests/test_evaluation_cadence.py tests/test_v24_experiment.py
python tests/test_smoke.py
```

Some full-suite cases require optional Rust extensions or historical checkpoints. Use the targeted checks above for the portable export; do not interpret unavailable optional fixtures as a measured engine-strength result.
