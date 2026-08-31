#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29544}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p outputs/checkpoints outputs/logs
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  scripts/train.py \
  --config configs/train_ms2mol.yml \
  --pretrained_ckpt checkpoints/graph2mol_iter1460000.pt \
  --align_ckpt checkpoints/align.pt \
  --ckptdir outputs/checkpoints \
  --logdir outputs/logs \
  "$@"
