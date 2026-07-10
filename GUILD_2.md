## 1. Clone Repo

```bash

git clone https://github.com/dnanper/vidore-thesis.git -b phi3
cd /workspace/vidore-thesis
```

## 2.

```bash
mkdir -p logs models dataset checkpoints
pip install -U "transformers>=4.51.0" "accelerate>=0.26.0" "qwen-vl-utils>=0.0.8" \
  "huggingface_hub" "safetensors" "pyarrow" "pandas" "polars" "pillow" "tqdm" "pyyaml" \
  "peft" "bitsandbytes"
```

## 3.

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

## 4.

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-10p \
  --output-path dataset/attn_cache_mmdocir_phi3_prior_10p \
  --prompt-mode image_only --source-mode instruction --layer-index -1 \
  --batch-size 1 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_prior_10p.log
```

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-5p \
  --output-path dataset/attn_cache_mmdocir_phi3_query_5p \
  --prompt-mode query_image --source-mode query --layer-index -1 \
  --batch-size 42 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_query_5p.log
```

## 5.

```bash
python scripts/make_phi3_runtime_config.py  --src configs/train_config_phi3_mmdocir.yaml  --dst configs/runtime/train_config_phi3_full.yaml  --model models/Phi-3-vision-128k-instruct  --dtype bfloat16  --output-dir checkpoints/colphi3_full  --train-data-path dataset/mmdocir-triplets-k1-full  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_full  --query-cache dataset/attn_cache_mmdocir_phi3_query_full
```

## 6. Train Full

```bash
python -u src/train/train_phi3_mmdocir.py \
  --config configs/runtime/train_config_phi3_full.yaml \
  2>&1 | tee logs/train_phi3_full.log
```

Metrics:

```bash
tail -n 20 checkpoints/colphi3_full/train_metrics.csv
```

## 7. Eval

```bash
python evaluate/evaluate_mmdocir_phi3.py \
  --model models/Phi-3-vision-128k-instruct \
  --checkpoint checkpoints/colphi3_full/final \
  --eval-root dataset/MMDocIR_Evaluation_Dataset \
  --batch-size 16
```

Eval with adaptive pruning:

```bash
python evaluate/evaluate_mmdocir_phi3.py \
  --model models/Phi-3-vision-128k-instruct \
  --checkpoint checkpoints/colphi3_full/final \
  --eval-root dataset/MMDocIR_Evaluation_Dataset \
  --batch-size 16 \
  --prune-docs \
  --prune-r-min 0.3 \
  --prune-r-max 0.9 \
  --prune-mode linear
```
