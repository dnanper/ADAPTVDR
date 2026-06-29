#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/cmdir/home/test01/.conda/envs/graduation_thesis/bin/python}"

TRAIN_MODE="${TRAIN_MODE:-query}"
ATTN_VARIANT="${ATTN_VARIANT:-grid}"

if [[ -z "${CONFIG_PATH:-}" ]]; then
  case "${TRAIN_MODE}:${ATTN_VARIANT}" in
    query:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_query.yaml"
      ;;
    query_mean_cosine:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_query_mean_cosine.yaml"
      ;;
    query_mean_cosine_dist:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_query_mean_cosine.yaml"
      ;;
    query_paper_raw_cosine:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_query_paper_prompt_raw_cosine.yaml"
      ;;
    query_oldprompt_raw_cosine:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_query_oldprompt_raw_cosine.yaml"
      ;;
    prior:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_prior.yaml"
      ;;
    both:grid)
      CONFIG_PATH="$ROOT_DIR/configs/train_config_colqwen35_08b_colpali_grid_both.yaml"
      ;;
    *)
      echo "Unsupported TRAIN_MODE/ATTN_VARIANT: ${TRAIN_MODE}/${ATTN_VARIANT}" >&2
      exit 1
  esac
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/vidore_train/datasets--vidore--colpali_train_set/snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data}"

cd "$ROOT_DIR"

echo "Using TRAIN_MODE=${TRAIN_MODE} ATTN_VARIANT=${ATTN_VARIANT}"
echo "Using config: $CONFIG_PATH"

"$PYTHON_BIN" -u scripts/validate_retrieval_parquet.py \
  --data-path "$DATA_PATH" \
  --split train \
  --num-shards 5 \
  --max-image-checks 8

TRAIN_CMD=(
  "$PYTHON_BIN" -u src/train/train.py
  --config "$CONFIG_PATH"
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  TRAIN_CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi

"${TRAIN_CMD[@]}"
