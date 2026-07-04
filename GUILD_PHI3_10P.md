# Phi3 Vast 10 Percent

## 1. Setup

```bash
cd /workspace/vidore-thesis
mkdir -p logs models dataset checkpoints configs/runtime
pip install -U "transformers>=4.51.0" "accelerate>=0.26.0" "qwen-vl-utils>=0.0.8" \
  "huggingface_hub" "safetensors" "pyarrow" "pandas" "pillow" "tqdm" "pyyaml" \
  "peft" "bitsandbytes"
```

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
hf download microsoft/Phi-3-vision-128k-instruct --local-dir models/Phi-3-vision-128k-instruct
```

## 2. Build 10% Triplets

Skip if `dataset/mmdocir-triplets-k1-10p` already exists.

```bash
python scripts/build_mmdocir_triplets.py \
  --dataset_root dataset/MMDocIR_Train_Dataset \
  --hard_neg_k 1 --sample-fraction 0.1 --sample-seed 42 \
  --output_dir dataset/mmdocir-triplets-k1-10p
```

## 3. Teacher Cache

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
  --train-data-path dataset/mmdocir-triplets-k1-10p \
  --output-path dataset/attn_cache_mmdocir_phi3_query_10p \
  --prompt-mode query_image --source-mode query --layer-index -1 \
  --batch-size 1 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_query_10p.log
```

## 4. Integration Smoke Test

```bash
python scripts/smoke_phi3_vast.py --test all \
  --model models/Phi-3-vision-128k-instruct \
  --triplet-dir dataset/mmdocir-triplets-k1-10p \
  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_10p \
  --query-cache dataset/attn_cache_mmdocir_phi3_query_10p
```

## 5. Create 10% Train Config

```bash
python scripts/make_phi3_runtime_config.py \
  --src configs/train_config_phi3_mmdocir.yaml \
  --dst configs/runtime/train_config_phi3_10p.yaml \
  --model models/Phi-3-vision-128k-instruct \
  --dtype bfloat16 \
  --output-dir checkpoints/colphi3_10p \
  --train-data-path dataset/mmdocir-triplets-k1-10p \
  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_10p \
  --query-cache dataset/attn_cache_mmdocir_phi3_query_10p
```

## 6. Train 10%

```bash
python -u src/train/train_phi3_mmdocir.py \
  --config configs/runtime/train_config_phi3_10p.yaml \
  2>&1 | tee logs/train_phi3_10p.log
```

Metrics:

```bash
tail -n 20 checkpoints/colphi3_10p/train_metrics.csv
```
