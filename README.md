# ADAPTVDR

Research code for **ADAPTVDR: Efficient Multi-Vector Visual Document Retrieval with Adaptive Resolution and Index Compression**.

ADAPTVDR is a ColPali-style visual document retriever. It preserves page aspect ratio within bounded pixel budgets, uses frozen Teacher attention maps to supervise local Student evidence during training, and prunes Student page vectors for a smaller MaxSim index. The Teacher is used only offline; the Student alone indexes and retrieves pages.

## Paper results

- **ViDoRe v1:** PaliGemma2-3B reaches **86.88 mean nDCG@5**; the published ColPali reference is 81.25.
- **Index compression:** linear entropy pruning saves **24.15%** of raw vector payload for a **0.43-point nDCG@5** decrease.
- **MMDocIR:** pruned Phi-3-Vision-128K-Instruct reaches **82.38 macro** and **83.25 micro Recall@5**.

See the paper for complete experimental settings, ablations, and comparison scope.

## Method at a glance

1. Encode query tokens and page patches with a Student multi-vector retriever; rank pages with MaxSim.
2. Preserve page geometry with bounded dynamic resolution (`min_pixels` to `max_pixels`).
3. Cache query-conditioned and image-prior Teacher maps for positive training pages; align them to Student patch grids.
4. At indexing, retain Student page patches according to attention importance and entropy-adaptive keep ratios. MaxSim serving is unchanged.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/train/` | datasets, collators, losses, Teacher-target loading, and training entry points |
| `scripts/` | backbone embedders, Teacher-cache precomputation, MMDocIR triplet construction, and pruning |
| `configs/` | ViDoRe, ColPali/ColQwen, and Phi-3/MMDocIR experiment settings |
| `evaluate/` | ViDoRe and MMDocIR retrieval evaluation |
| `GUILD.md` | full MMDocIR/Phi-3 data, cache, training, and evaluation runbook |

## Setup

Python, PyTorch with CUDA, and a bf16-capable NVIDIA GPU are required for the reported runs. Install the PyTorch build matching your CUDA environment first, then:

```bash
git clone https://github.com/dnanper/vidore-thesis.git
cd vidore-thesis
pip install -r requirements.txt
pip install accelerate bitsandbytes pyyaml
```

Datasets, model weights, Teacher caches, and checkpoints are not included. The checked-in YAML files contain machine-local paths; update `model.name_or_path`, data paths, cache paths, and `training.output_dir` before running.

## Reproduce a pipeline

### ViDoRe training

Precompute the prior and query Teacher caches for the `vidore/colpali_train_set`, then train a Student configuration:

```bash
python src/train/train.py \
  --config configs/train_config_colqwen35_08b_colpali_grid_both.yaml
```

Use `scripts/precompute_teacher_attn.py` for cache generation. The corresponding configuration specifies the expected cache locations and dynamic-resolution bounds.

### MMDocIR / Phi-3

Build page-level triplets, cache Teacher maps over positive pages, then run the Phi-3 training entry point:

```bash
python scripts/build_mmdocir_triplets.py \
  --dataset_root dataset/MMDocIR_Train_Dataset \
  --hard_neg_k 1 --sample-fraction 0.1 --sample-seed 42 \
  --output_dir dataset/mmdocir-triplets-k1-10p

python src/train/train_phi3_mmdocir.py \
  --config configs/train_config_phi3_mmdocir.yaml
```

`GUILD.md` gives exact download, precompute, smoke-test, and full-run commands. Start with `configs/train_config_phi3_smoke32.yaml` to validate the data-to-cache-to-training path.

### Evaluate MMDocIR with pruning

```bash
python evaluate/evaluate_mmdocir_phi3.py \
  --checkpoint checkpoints/colphi3_mmdocir_thesis/final \
  --eval-root dataset/MMDocIR_Evaluation_Dataset \
  --prune-docs --prune-r-min 0.3 --prune-r-max 0.9 --prune-mode linear
```

Omit `--prune-docs` for an unpruned evaluation. Omit `--image-size` to retain dynamic-resolution preprocessing.

## Citation

```bibtex
@inproceedings{hoang2026adaptvdr,
  title     = {ADAPTVDR: Efficient Multi-Vector Visual Document Retrieval with Adaptive Resolution and Index Compression},
  author    = {Hoang, Bao-Long and Phan, Tat-An and Nguyen, Thi-Hau},
  booktitle = {Knowledge and Systems Engineering},
  year      = {2026}
}
```
