# Phi3 Vast Smoke32

## 1. Setup

```bash
cd /workspace/vidore-thesis
mkdir -p logs models dataset checkpoints
pip install -U "transformers>=4.51.0" "accelerate>=0.26.0" "qwen-vl-utils>=0.0.8" \
  "huggingface_hub" "safetensors" "pyarrow" "pandas" "polars" "pillow" "tqdm" "pyyaml" \
  "peft" "bitsandbytes"
```

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
hf download microsoft/Phi-3-vision-128k-instruct --local-dir models/Phi-3-vision-128k-instruct
```

## 2. Prepare Smoke32 Data

Skip this if `dataset/mmdocir-triplets-k1-smoke32` already exists.

```bash
python scripts/make_smoke32_triplets.py \
  --src-dir dataset/mmdocir-triplets-k1-10p \
  --dst-dir dataset/mmdocir-triplets-k1-smoke32 \
  --num-samples 32
```

## 3. Teacher Cache

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-smoke32 \
  --output-path dataset/attn_cache_mmdocir_phi3_prior_smoke32 \
  --prompt-mode image_only --source-mode instruction --layer-index -1 \
  --batch-size 1 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_prior_smoke32.log
```

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-smoke32 \
  --output-path dataset/attn_cache_mmdocir_phi3_query_smoke32 \
  --prompt-mode query_image --source-mode query --layer-index -1 \
  --batch-size 1 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_query_smoke32.log
```

## 4. Integration Smoke Test

```bash
python scripts/smoke_phi3_vast.py --test all \
  --model models/Phi-3-vision-128k-instruct \
  --triplet-dir dataset/mmdocir-triplets-k1-smoke32 \
  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_smoke32 \
  --query-cache dataset/attn_cache_mmdocir_phi3_query_smoke32
```

## 5. Create Smoke32 Train Config

```bash
python scripts/make_phi3_runtime_config.py \
  --src configs/train_config_phi3_smoke32.yaml \
  --dst configs/runtime/train_config_phi3_smoke32.yaml \
  --model models/Phi-3-vision-128k-instruct \
  --dtype float16 \
  --output-dir checkpoints/colphi3_smoke32 \
  --train-data-path dataset/mmdocir-triplets-k1-smoke32 \
  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_smoke32 \
  --query-cache dataset/attn_cache_mmdocir_phi3_query_smoke32
```

## 6. Train Smoke32

```bash
python -u src/train/train_phi3_mmdocir.py \
  --config configs/runtime/train_config_phi3_smoke32.yaml \
  2>&1 | tee logs/train_phi3_smoke32.log
```

Metrics:

```bash
tail -n 20 checkpoints/colphi3_smoke32/train_metrics.csv
```
