#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/cmdir/home/test01/.conda/envs/graduation_thesis/bin/python}"

TEACHER_MODEL="${TEACHER_MODEL:-/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT_DIR/dataset/vidore_train/datasets--vidore--colpali_train_set/snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/dataset/attn_cache_prior_qwen08}"

SPLIT="${SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-1}"
LAYER_INDEX="${LAYER_INDEX:--1}"
MIN_PIXELS="${MIN_PIXELS:-4096}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
INSTRUCTION="${INSTRUCTION-}"
if [[ -z "${INSTRUCTION}" ]]; then
  INSTRUCTION="Represent the user's input."
fi
PROMPT_MODE="${PROMPT_MODE:-image_only}"
SOURCE_MODE="${SOURCE_MODE:-instruction}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -u scripts/precompute_teacher_attn.py \
  --teacher-model "$TEACHER_MODEL" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --output-path "$OUTPUT_PATH" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --save-every "$SAVE_EVERY" \
  --layer-index "$LAYER_INDEX" \
  --instruction "$INSTRUCTION" \
  --min-pixels "$MIN_PIXELS" \
  --max-pixels "$MAX_PIXELS" \
  --prompt-mode "$PROMPT_MODE" \
  --source-mode "$SOURCE_MODE" \
  --resume \
  "$@"
