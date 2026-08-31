#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MS2MOL_CKPT:?Set MS2MOL_CKPT to the separately downloaded NPLIB1 checkpoint}"
: "${ALIGN_CKPT:?Set ALIGN_CKPT to the separately downloaded NPLIB1 align checkpoint}"
PYTHON="${PYTHON:-python}"

test -f "$MS2MOL_CKPT"
test -f "$ALIGN_CKPT"

exec "$PYTHON" scripts/sample.py \
  --config examples/nplib1_10/configs/sample.yml \
  --ms2mol_ckpt "$MS2MOL_CKPT" \
  --align_ckpt "$ALIGN_CKPT" \
  --device cuda \
  --batch_size 10 \
  --n_samples 100 \
  --n_timesteps 20 \
  --output_dir examples/nplib1_10/outputs \
  --save_candidate_cache
