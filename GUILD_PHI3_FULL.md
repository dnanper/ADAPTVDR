# Phi3 Vast Full

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

## 2. Build Full Triplets

Skip if `dataset/mmdocir-triplets-k1-full` already exists.

```bash
python scripts/build_mmdocir_triplets.py \
  --dataset_root dataset/MMDocIR_Train_Dataset \
  --hard_neg_k 1 --sample-fraction 1.0 --sample-seed 42 \
  --output_dir dataset/mmdocir-triplets-k1-full
```

## 3. Teacher Cache

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-full \
  --output-path dataset/attn_cache_mmdocir_phi3_prior_full \
  --prompt-mode image_only --source-mode instruction --layer-index -1 \
  --batch-size 1 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_prior_full.log
```

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-full \
  --output-path dataset/attn_cache_mmdocir_phi3_query_full \
  --prompt-mode query_image --source-mode query --layer-index -1 \
  --batch-size 1 --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 --max-pixels 1048576 --resume \
  2>&1 | tee logs/precompute_query_full.log
```

## 4. Integration Smoke Test

```bash
python scripts/smoke_phi3_vast.py --test all \
  --model models/Phi-3-vision-128k-instruct \
  --triplet-dir dataset/mmdocir-triplets-k1-full \
  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_full \
  --query-cache dataset/attn_cache_mmdocir_phi3_query_full
```

## 5. Create Full Train Config

```bash
python scripts/make_phi3_runtime_config.py \
  --src configs/train_config_phi3_mmdocir.yaml \
  --dst configs/runtime/train_config_phi3_full.yaml \
  --model models/Phi-3-vision-128k-instruct \
  --dtype bfloat16 \
  --output-dir checkpoints/colphi3_full \
  --train-data-path dataset/mmdocir-triplets-k1-full \
  --prior-cache dataset/attn_cache_mmdocir_phi3_prior_full \
  --query-cache dataset/attn_cache_mmdocir_phi3_query_full
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
