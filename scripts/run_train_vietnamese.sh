#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/cmdir/home/test01/.conda/envs/graduation_thesis/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train_config_vietnamese.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/colqwen3_5_lora-0.8b/ColQwen3.5-0.8B-Embedding-v2}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/datasets--5CD-AI--Viet-colpali_train_set-Gemini/snapshots/a950aa56995f6e740c0bafc60e05b4f57b5132bb/data/data}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -u scripts/validate_retrieval_parquet.py \
  --data-path "$DATA_PATH" \
  --split train \
  --num-shards 5 \
  --max-image-checks 8

"$PYTHON_BIN" -u src/train/train.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH"
