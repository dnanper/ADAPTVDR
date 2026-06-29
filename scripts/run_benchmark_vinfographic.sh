#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/cmdir/home/test01/.conda/envs/graduation_thesis/bin/python}"

MODEL_PATH="${MODEL_PATH:-/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/colqwen3_5_lora-0.8b/ColQwen3.5-0.8B-Embedding-Vietnamese/final}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/dataset/vinfographic}"

SUMMARY_CSV="${SUMMARY_CSV:-$ROOT_DIR/results/vinfographic_colqwen3_5.csv}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-$ROOT_DIR/results/vinfographic_colqwen3_5_predictions.csv}"

IMG_BATCH="${IMG_BATCH:-2}"
QUERY_BATCH="${QUERY_BATCH:-8}"
TOP_K="${TOP_K:-10}"
MODE="${MODE:-multivec_mrl}"
DIMS="${DIMS:-128 256 512 1024 2048}"
SPLITS="${SPLITS:-single_test multi_test}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -u evaluate/evaluate_vinfographic_colqwen3_5.py \
  --dataset-root "$DATASET_ROOT" \
  --model-path "$MODEL_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --mode "$MODE" \
  --img-batch "$IMG_BATCH" \
  --query-batch "$QUERY_BATCH" \
  --top-k "$TOP_K" \
  --summary-csv "$SUMMARY_CSV" \
  --predictions-csv "$PREDICTIONS_CSV" \
  --splits $SPLITS \
  --dims $DIMS \
  "$@"
