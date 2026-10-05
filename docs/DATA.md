# Data and weights

The JSONL files under `data_sl` were collected or derived from public chess games for this project's supervised/rehearsal and endgame experiments. Human-game records include game identifiers, moves, ratings, speeds and results where available. Puzzle and endgame records were derived from game positions. The AV file contains generated move-value labels.

These assets reproduce the local project inputs; they are not claimed to be independent held-out test data. Unknown/aborted outcomes must not be relabeled as draws. Dataset availability does not imply a validated strength gain from every objective.

`ASSETS.json` records SHA-256 checksums for exported weights and data. The champion's source lineage is v19, iteration 27; its optimizer, schedule and replay are omitted from this export. The teacher is the existing supervised warm-start. No token, Lichess account configuration or personal handoff is included.

External tools, datasets, and dependency crates retain their respective terms. No new open-source license has been assigned to the author's project automatically.
