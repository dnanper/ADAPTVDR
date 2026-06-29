---
language:
- vi
- en
task_categories:
- visual-question-answering
- question-answering
tags:
- infographic
- vietnamese
- vqa
- document-understanding
size_categories:
- 10K<n<100K
---

# ViInfographicVQA

## Overview

**ViInfographicVQA** is a Vietnamese **Visual Question Answering (VQA)** benchmark for **infographic understanding**.  
It evaluates models’ ability to **read, reason, and synthesize information** from data-rich, layout-heavy visuals that mix **text, charts, maps, and design elements**.

Two settings are provided:
- **Single-image VQA** – questions answered from one infographic.
- **Multi-image VQA** – questions requiring reasoning across multiple, semantically related infographics.

---

## 📊 Dataset Summary

| Split                 | #Images | #QAs  | Description                               |
|----------------------|--------:|------:|-------------------------------------------|
| Single-image (train) | 1,787   | 12,521| VQA on individual infographics             |
| Single-image (test)  | 193     | 1,374 | Held-out evaluation                        |
| Multi-image (train)  | 5,861   | 5,878 | Cross-image reasoning (training)           |
| Multi-image (test)   | 653     | 636   | Cross-image reasoning (test)               |
| **Total**            | **6,747** | **20,409** | Across all splits                     |

- **Language:** Vietnamese  
- **Domains:** Economy, Healthcare, Education, Society & Culture, Disasters & Accidents, Sports & Arts, Weather, etc.


## 🗂️ Repository Layout

```

ViInfographicVQA/
├── images/                # all image files (referenced by filename)
├── <parquet files>        # four splits stored as parquet shards on the Hub
└── README.md

````


## 🚀 Quickstart

```python
from datasets import load_dataset

# Load all splits (parquet)
ds = load_dataset("VLAI-AIVN/ViInfographicVQA")

single_train = ds["single_train"]
multi_train  = ds["multi_train"]

# Each sample:
# - images_paths: list of filenames (relative to `images/`)
# - image: preview Image() (the first file)
ex = multi_train[0]
print(ex["images_paths"])  # e.g. ["13321.jpg", "13028.jpg", "13458.jpg"]
preview = ex["image"]      # PIL.Image preview (for quick visualization)
````

### Read **all images** for multi-image samples (no local download)

Use Hub file URIs, then cast to `Image()`:

```python
from datasets import Image, Sequence, load_dataset

ds = load_dataset("VLAI-AIVN/ViInfographicVQA")
repo_base = "hf://datasets/VLAI-AIVN/ViInfographicVQA/images"

def add_full_paths(example):
    example["images_full"] = [f"{repo_base}/{fn}" for fn in example["images_paths"]]
    return example

multi = ds["multi_train"].map(add_full_paths, remove_columns=[])
multi = multi.cast_column("images_full", Sequence(Image()))

all_imgs = multi[0]["images_full"]   # list[PIL.Image] — all referenced images
```

### Streaming (large-scale training)

```python
from datasets import load_dataset, Image, Sequence

ds = load_dataset("VLAI-AIVN/ViInfographicVQA", streaming=True)
repo_base = "hf://datasets/VLAI-AIVN/ViInfographicVQA/images"

def add_full_paths(example):
    example["images_full"] = [f"{repo_base}/{fn}" for fn in example["images_paths"]]
    return example

multi_stream = ds["multi_train"].map(add_full_paths)
multi_stream = multi_stream.cast_column("images_full", Sequence(Image()))

ex = next(iter(multi_stream))
imgs = ex["images_full"]  # list of PIL.Image (lazy/streamed)
```

### Local download (offline use)

```python
from huggingface_hub import snapshot_download
from datasets import load_dataset

# Download the entire dataset repo locally (parquet + images)
local_dir = snapshot_download(repo_id="VLAI-AIVN/ViInfographicVQA", repo_type="dataset")

# Load from disk
ds = load_dataset(local_dir)

# Reconstruct absolute paths to images on disk if needed:
import os
images_root = os.path.join(local_dir, "images")
def to_abs(example):
    example["images_abs"] = [os.path.join(images_root, fn) for fn in example["images_paths"]]
    return example

multi_local = ds["multi_train"].map(to_abs)
print(multi_local[0]["images_abs"][:3])  # ['/.../images/13321.jpg', ...]
```

> **Speed tip:** set `HF_HUB_ENABLE_HF_TRANSFER=1` to accelerate uploads/downloads.


## 🔍 Research Applications

* Multimodal reasoning on charts, tables, and dense text
* Cross-image synthesis and comparison
* Low-resource VQA in Vietnamese
* Evaluation of OCR, layout parsing, and numerical reasoning


## 🧮 Evaluation

We use **Average Normalized Levenshtein Similarity (ANLS)** for string-based answer evaluation, which tolerates minor textual variations while penalizing semantic errors.


## 📚 Citation

If you use this dataset, please cite:

```bibtex
@article{van2025viinfographicvqa,
  title={ViInfographicVQA: A Benchmark for Single and Multi-image Visual Question Answering on Vietnamese Infographics},
  author={Van-Dinh, Tue-Thu and Tran, Hoang-Duy and Duong, Truong-Binh and Pham, Mai-Hanh and Le-Nguyen, Binh-Nam and Nguyen, Quoc-Thai},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
