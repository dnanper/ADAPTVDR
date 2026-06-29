#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/cmdir/home/test01/.conda/envs/graduation_thesis/bin/python}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT_DIR/dataset/vidore_train/datasets--vidore--colpali_train_set/snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/dataset/attn_cache_query_qwen08_grid_no_system}"
TEACHER_MODEL="${TEACHER_MODEL:-/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3-VL-8B-Instruct}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -u scripts/precompute_teacher_attn.py \
  --teacher-model "$TEACHER_MODEL" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --output-path "$OUTPUT_PATH" \
  --split train \
  --prompt-mode query_image \
  --source-mode query \
  --layer-index -1 \
  --batch-size 16 \
  --save-every 1 \
  --min-pixels 4096 \
  --max-pixels 1048576 \
  --no-system-instruction \
  --resume
