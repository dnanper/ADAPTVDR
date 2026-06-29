#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/cmdir/home/test01/.conda/envs/graduation_thesis/bin/python}"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train_config_colpali.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/vidore_train/datasets--vidore--colpali_train_set/snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -u scripts/validate_retrieval_parquet.py \
  --data-path "$DATA_PATH" \
  --split train \
  --num-shards 5 \
  --max-image-checks 8

TRAIN_CMD=(
  "$PYTHON_BIN" -u src/train/train_colpali.py
  --config "$CONFIG_PATH"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  TRAIN_CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi

"${TRAIN_CMD[@]}"
