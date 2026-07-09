## 1. Clone Repo

```bash
cd /workspace
git clone https://github.com/dnanper/vidore-thesis.git -b phi3
cd /workspace/vidore-thesis
```

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

## 2. Create Python Env

```bash
pip install -U \
  "transformers>=4.51.0" \
  "accelerate>=0.26.0" \
  "qwen-vl-utils>=0.0.8" \
  "huggingface_hub" \
  "safetensors" \
  "pyarrow" \
  "pandas" \
  "pillow" \
  "tqdm" \
  "pyyaml"
```

## 3. Hugging Face Login + Cache

```bash
hf auth login
```

## 4. Download MMDocIR Data

```bash
hf download MMDocIR/MMDocIR_Train_Dataset `
    --repo-type dataset `
    --local-dir dataset/MMDocIR_Train_Dataset
```

```bash
hf download MMDocIR/MMDocIR_Evaluation_Dataset  --repo-type dataset  --local-dir dataset/MMDocIR_Evaluation_Dataset
```

## 5. Download Teacher Model

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
```

## 6. Build MMDocIR Triplets

100% training set:

```bash
python scripts/build_mmdocir_triplets.py --dataset_root dataset/MMDocIR_Train_Dataset --hard_neg_k 1 --sample-fraction 1.0 --sample-seed 42 --output_dir dataset/mmdocir-triplets-k1-full
```

Quick inspect:

```bash
python - <<'PY'
import glob, pandas as pd
files = sorted(glob.glob("dataset/mmdocir-triplets-k1-10p/*.parquet"))
print("shards", len(files))
df = pd.read_parquet(files[0])
print(df.columns.tolist())
print(df[["sample_id", "query", "positive_id", "negative_ids"]].head(2))
print("rows first shard", len(df))
PY
```

## 7. Smoke Precompute on 32 Samples

Run this before the 10% job to catch environment/model/data issues.

```bash
mkdir -p logs
```

Prior teacher cache:

```bash
python -u scripts/precompute_teacher_attn.py \
    --teacher-model models/Qwen3-VL-8B-Instruct \
    --train-data-path dataset/mmdocir-triplets-k1-smoke32 \
    --output-path dataset/attn_cache_mmdocir_phi3_prior_smoke32 \
    --prompt-mode image_only \
    --source-mode instruction \
    --layer-index -1 \
    --batch-size 1 \
    --save-every 1 \
    --instruction "Represent the user's input." \
    --min-pixels 4096 \
    --max-pixels 1048576 \
    --resume
```

Query-conditioned teacher cache:

```bash
python -u scripts/precompute_teacher_attn.py \
    --teacher-model models/Qwen3-VL-8B-Instruct \
    --train-data-path dataset/mmdocir-triplets-k1-smoke32 \
    --output-path dataset/attn_cache_mmdocir_phi3_query_smoke32 \
    --prompt-mode query_image \
    --source-mode query \
    --layer-index -1 \
    --batch-size 1 \
    --save-every 1 \
    --instruction "Represent the user's input." \
    --min-pixels 4096 \
    --max-pixels 1048576 \
    --resume
```

Check outputs:

```bash
find dataset/attn_cache_mmdocir_phi3_prior_smoke32 -maxdepth 1 -type f | head
find dataset/attn_cache_mmdocir_phi3_query_smoke32 -maxdepth 1 -type f | head
```

## 8. Full 100% Precompute

Prior:

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-full \
  --output-path dataset/attn_cache_mmdocir_phi3_prior_full \
  --prompt-mode image_only \
  --source-mode instruction \
  --layer-index -1 \
  --batch-size 1 \
  --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 \
  --max-pixels 1048576 \
  --resume 2>&1 | tee logs/precompute_prior_full.log
```

Query:

```bash
python -u scripts/precompute_teacher_attn.py \
  --teacher-model models/Qwen3-VL-8B-Instruct \
  --train-data-path dataset/mmdocir-triplets-k1-full \
  --output-path dataset/attn_cache_mmdocir_phi3_query_full \
  --prompt-mode query_image \
  --source-mode query \
  --layer-index -1 \
  --batch-size 1 \
  --save-every 1 \
  --instruction "Represent the user's input." \
  --min-pixels 4096 \
  --max-pixels 1048576 \
  --resume 2>&1 | tee logs/precompute_query_full.log
```

## 10. Verify Cache Metadata

```bash
python - <<'PY'
import torch
for path in [
    "dataset/attn_cache_mmdocir_phi3_prior_10p/metadata.pt",
    "dataset/attn_cache_mmdocir_phi3_query_10p/metadata.pt",
    "dataset/attn_cache_mmdocir_phi3_prior_full/metadata.pt",
    "dataset/attn_cache_mmdocir_phi3_query_full/metadata.pt",
]:
    meta = torch.load(path, map_location="cpu", weights_only=True)
    print(path)
    for k in ["prompt_mode", "source_mode", "num_saved_samples", "num_saved_batches", "teacher_model"]:
        print(" ", k, meta.get(k))
PY
```

Expected:

- prior: `prompt_mode=image_only`, `source_mode=instruction`
- query: `prompt_mode=query_image`, `source_mode=query`
- `num_saved_samples` near 10% or 100% of built triplets, depending on cache path

## 11. Enable Caches for Later Training

Edit `configs/train_config_phi3_mmdocir.yaml`:

```yaml
data:
  train_data_path: "dataset/mmdocir-triplets-k1-10p"
  image_size: null
  min_pixels: 4096
  max_pixels: 1048576
  attn_cache_path_prior: "dataset/attn_cache_mmdocir_phi3_prior_10p"
  attn_cache_path_query: "dataset/attn_cache_mmdocir_phi3_query_10p"

pruning:
  k_min: 0.3
  k_max: 0.9
  mode: "linear"
```

For full-data training, switch all three paths to `mmdocir-triplets-k1-full`, `attn_cache_mmdocir_phi3_prior_full`, and `attn_cache_mmdocir_phi3_query_full`.

Training command later:

```bash
python src/train/train_phi3_mmdocir.py --config configs/train_config_phi3_smoke32.yaml
```

```bash
python src/train/train_phi3_mmdocir.py --config configs/train_config_phi3_mmdocir.yaml
```

## 12. Evaluate Phi3 With Patch Pruning

Baseline MMDocIR page-level eval:

```bash
python evaluate/evaluate_mmdocir_phi3.py \
  --model models/Phi-3-vision-128k-instruct \
  --checkpoint checkpoints/colphi3_mmdocir_thesis/final \
  --eval-root dataset/MMDocIR_Evaluation_Dataset \
  --batch-size 4
```

Eval with adaptive pruning:

```bash
python evaluate/evaluate_mmdocir_phi3.py \
  --model models/Phi-3-vision-128k-instruct \
  --checkpoint checkpoints/colphi3_mmdocir_thesis/final \
  --eval-root dataset/MMDocIR_Evaluation_Dataset \
  --batch-size 4 \
  --prune-docs \
  --prune-r-min 0.3 \
  --prune-r-max 0.9 \
  --prune-mode linear
```

Use `--image-size 1344` only when you intentionally want the old fixed-square preprocessing. Omit it for dynamic resolution.

## 13. Difference vs Thesis Main ColPali Data

Thesis main Table 4.10 uses `vidore/colpali_train_set`:

- each row is already a simple `(query, image)` pair
- no explicit hard negative in the row
- teacher precompute reads `query` + `image`
- retrieval loss mainly uses in-batch negatives
- sample id is generated from shard path, row index, filename, query

MMDocIR train data:

- official train data is split into page parquet + annotation jsonl
- annotation has `positive_passages` and `negative_passages`
- preprocessing must resolve `(doc_name, page_id)` to actual page image
- output becomes local triplet parquet: `query`, `positive`, `hard_negatives`
- teacher precompute still uses only `(query, positive)` for alignment
- hard negative pages are used later by retrieval loss, not by teacher precompute
- sample id is explicit and stable: derived from positive page id + query

So MMDocIR needs an extra preprocessing bridge:

```text
MMDocIR pages + annotations -> thesis triplet parquet -> teacher cache on positive pages
```

ColPali thesis data is already close to:

```text
query + positive image -> teacher cache
```

## 14. Phi3 Code Path Map

```text
scripts/build_mmdocir_triplets.py
  -> dataset/mmdocir-triplets-k1-10p or k1-full

scripts/precompute_teacher_attn.py
  -> prior/query teacher patch-importance caches over positive pages

src/train/phi3_collator.py
  -> dynamic page resize + query/doc token masks

scripts/colphi3_embedding.py
  -> Phi3 hidden states projected to 128-dim token embeddings

src/train/train_phi3_mmdocir.py
  -> LoRA + Matryoshka MaxSim + teacher alignment

evaluate/evaluate_mmdocir_phi3.py
  -> MMDocIR page-level eval, optional adaptive pruning
```

## 15. Artifacts To Keep

Copy or persist these before shutting down Vast:

```text
dataset/mmdocir-triplets-k1-10p/
dataset/mmdocir-triplets-k1-full/
dataset/attn_cache_mmdocir_phi3_prior_10p/
dataset/attn_cache_mmdocir_phi3_query_10p/
dataset/attn_cache_mmdocir_phi3_prior_full/
dataset/attn_cache_mmdocir_phi3_query_full/
checkpoints/colphi3_mmdocir_thesis/
models/Qwen3-VL-8B-Instruct/   optional if persistent cache not kept
```

1. Smoke eval trước, xem checkpoint có sống không

python -u evaluate/evaluate_mmdocir_phi3.py \
 --model models/Phi-3-vision-128k-instruct \
 --checkpoint checkpoints/colphi3_5p/final \
 --eval-root dataset/MMDocIR_Evaluation_Dataset \
 --batch-size 4 \
 --max-queries 10 \
 2>&1 | tee logs/eval_phi3_5p_10q.log

2. Full MMDocIR paper-style eval

python -u evaluate/evaluate_mmdocir_phi3.py \
 --model models/Phi-3-vision-128k-instruct \
 --checkpoint checkpoints/colphi3_5p/final \
 --eval-root dataset/MMDocIR_Evaluation_Dataset \
 --batch-size 4 \
 2>&1 | tee logs/eval_phi3_5p_full.log

Script này đang đúng hướng paper MMDocIR: dùng MMDocIR_pages.parquet +
MMDocIR_annotations.jsonl, rank page trong cùng document, in Recall@1, Recall@5,
Recall@10, nDCG@5.

3. Eval phần giải pháp thesis: adaptive pruning

python -u evaluate/evaluate_mmdocir_phi3.py \
 --model models/Phi-3-vision-128k-instruct \
 --checkpoint checkpoints/colphi3_5p/final \
 --eval-root dataset/MMDocIR_Evaluation_Dataset \
 --batch-size 4 \
 --prune-docs \
 --prune-r-min 0.3 \
 --prune-r-max 0.9 \
 --prune-mode linear \
 2>&1 | tee logs/eval_phi3_5p_full_pruned.log
