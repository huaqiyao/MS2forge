#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/generated_80k}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$OUTPUT_DIR"

exec "$TORCHRUN" --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/sample.py \
  --config configs/sample.yml \
  --output_dir "$OUTPUT_DIR" \
  "$@"
