#!/bin/sh
set -eu
chess_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$chess_root"
if [ -n "${CHESS_ZERO_PYTHON:-}" ]; then
  chess_python=$CHESS_ZERO_PYTHON
elif [ -x .venv/bin/python ]; then
  chess_python="$chess_root/.venv/bin/python"
else
  chess_python=python3
fi
exec "$chess_python" -m chess_zero.uci \
  --ckpt weights/champion.pt --blocks 6 --channels 64 --planes 30 \
  --se-ratio 4 --device cpu --sims 400 \
  --c-puct 1.6 --fpu-reduction 0.5 --prune-singletons \
  --contempt 0.3 --asymmetric-contempt --edge-scale 3.0 \
  --mate-finish --book BUILTIN --leaf-batch 8 --safety-veto \
  --ml-slope 0.003 --ml-thr 0.9 "$@"
