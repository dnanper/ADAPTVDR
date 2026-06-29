#!/usr/bin/env python3
"""
Baseline evaluation of ColPali (original paper) on ViDoRe benchmark (fully offline).

Usage:
    python evaluate_colpali.py
    python evaluate_colpali.py --subsets arxivqa docvqa   # run specific subsets only
    python evaluate_colpali.py --img-batch 1              # reduce if VRAM OOM

"""

import os, argparse, time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import List

from colpali_engine.models import ColPali, ColPaliProcessor

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "/data2/cmdir/home/test01/longvnu/stable_diff/models/vidore/colpali"
DATA_BASE   = "/data2/cmdir/home/test01/longvnu/graduation_thesis/dataset/vidore"
OUT_CSV     = "/data2/cmdir/home/test01/longvnu/graduation_thesis/results_colpali.csv"

SUBSETS = {
    "arxivqa":       "datasets--vidore--arxivqa_test_subsampled/snapshots/b8a106812c8682bab08935cf5d1b4566c82562de",
    "docvqa":        "datasets--vidore--docvqa_test_subsampled/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6",
    "infovqa":       "datasets--vidore--infovqa_test_subsampled/snapshots/f793e830aaeae1ceb8a2df626fc555b3fd04d3db",
    "shiftproject":  "datasets--vidore--shiftproject_test/snapshots/6e6223b3839a3f4e62d676e70a1b715ee520cc66",
    "synth_ai":      "datasets--vidore--syntheticDocQA_artificial_intelligence_test/snapshots/5694fec64c57f7380f918435f9234b96d817b82a",
    "synth_energy":  "datasets--vidore--syntheticDocQA_energy_test/snapshots/a2f2f358463b03b84506849460ab5094a358526c",
    "synth_gov":     "datasets--vidore--syntheticDocQA_government_reports_test/snapshots/91cf66572d89c9cccac0661de227acaf04b44f64",
    "synth_health":  "datasets--vidore--syntheticDocQA_healthcare_industry_test/snapshots/d973f3da3e60c8eaf566efa2f7d2a1515cba40ed",
    "tabfquad":      "datasets--vidore--tabfquad_test_subsampled/snapshots/16c8e633612fbda7400bfcbbc31d61a7534f580f",
    "tatdqa":        "datasets--vidore--tatdqa_test/snapshots/b46fea43695e14697510104a3331d9e88683a416",
}

# ─── Metrics ──────────────────────────────────────────────────────────────────
def ndcg_at_k(scores: np.ndarray, relevant_idx: int, k: int = 5) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    if relevant_idx not in top_k:
        return 0.0
    rank = int(np.where(top_k == relevant_idx)[0][0]) + 1
    return (1.0 / np.log2(rank + 1)) / (1.0 / np.log2(2))

def recall_at_k(scores: np.ndarray, relevant_idx: int, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return 1.0 if relevant_idx in top_k else 0.0

# ─── MaxSim ───────────────────────────────────────────────────────────────────
def compute_scores_matrix(
    query_embs: List[torch.Tensor],
    doc_embs: List[torch.Tensor],
) -> np.ndarray:
    """[n_queries, n_docs] MaxSim matrix."""
    device = query_embs[0].device if query_embs[0].is_cuda else torch.device("cpu")
    scores = np.zeros((len(query_embs), len(doc_embs)), dtype=np.float32)

    for q_idx, q_emb in enumerate(tqdm(query_embs, desc="    MaxSim", leave=False)):
        q = q_emb.float().to(device)
        for d_idx, d_emb in enumerate(doc_embs):
            d = d_emb.float().to(device)
            sim = torch.einsum("qd,pd->qp", q, d)
            scores[q_idx, d_idx] = sim.max(dim=1).values.sum().item()

    return scores

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_subset(subset_path: str) -> pd.DataFrame:
    data_dir = os.path.join(subset_path, "data")
    parquets = sorted(f for f in os.listdir(data_dir) if f.startswith("test-") and f.endswith(".parquet"))
    if not parquets:
        raise FileNotFoundError(f"No test parquet files found in {data_dir}")
    frames = [pd.read_parquet(os.path.join(data_dir, f)) for f in parquets]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

def decode_image(img_field) -> Image.Image:
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img_field)}")

# ─── Embed helpers ────────────────────────────────────────────────────────────
def encode_images(
    model: ColPali,
    processor: ColPaliProcessor,
    images: List[Image.Image],
    batch_size: int,
) -> List[torch.Tensor]:
    all_embs = []
    for i in tqdm(range(0, len(images), batch_size), desc="    Images", leave=False):
        batch_imgs = images[i : i + batch_size]
        inputs = processor.process_images(batch_imgs).to(model.device)
        with torch.no_grad():
            embs = model(**inputs)  # [B, n_patches, 128]
        # unbind → list of [n_patches, 128]
        all_embs.extend(list(embs.unbind(0)))
    return all_embs

def encode_queries(
    model: ColPali,
    processor: ColPaliProcessor,
    queries: List[str],
    batch_size: int,
) -> List[torch.Tensor]:
    all_embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="    Queries", leave=False):
        batch_q = queries[i : i + batch_size]
        inputs = processor.process_queries(batch_q).to(model.device)
        with torch.no_grad():
            embs = model(**inputs)  # [B, n_tokens, 128]
        all_embs.extend(list(embs.unbind(0)))
    return all_embs

# ─── Evaluate one subset ──────────────────────────────────────────────────────
def evaluate_subset(
    model: ColPali,
    processor: ColPaliProcessor,
    df: pd.DataFrame,
    subset_name: str,
    img_batch: int,
    query_batch: int,
) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Subset : {subset_name}  ({len(df)} rows)")

    filenames     = df["image_filename"].tolist()
    unique_fnames = list(dict.fromkeys(filenames))
    fname_to_idx  = {fn: i for i, fn in enumerate(unique_fnames)}

    fname_to_img = {}
    for _, row in df.iterrows():
        fn = row["image_filename"]
        if fn not in fname_to_img:
            fname_to_img[fn] = decode_image(row["image"])
    unique_images = [fname_to_img[fn] for fn in unique_fnames]
    valid_queries, relevant_idxs = [], []
    for _, row in df.iterrows():
        q = str(row.get("query", "")).strip()
        if q.lower() == "none" or len(q) < 5:
            continue
        valid_queries.append(q)
        relevant_idxs.append(fname_to_idx[row["image_filename"]])
    queries = valid_queries

    print(f"  Corpus : {len(unique_images)} unique images")
    print(f"  Queries: {len(queries)}")

    print("  Encoding images ...")
    image_embs = encode_images(model, processor, unique_images, img_batch)

    print("  Encoding queries ...")
    query_embs = encode_queries(model, processor, queries, query_batch)

    print("  Computing MaxSim scores ...")
    scores_matrix = compute_scores_matrix(query_embs, image_embs)

    ndcg5, r1, r5 = [], [], []
    for q_idx, rel_idx in enumerate(relevant_idxs):
        s = scores_matrix[q_idx]
        ndcg5.append(ndcg_at_k(s, rel_idx, k=5))
        r1.append(recall_at_k(s, rel_idx, k=1))
        r5.append(recall_at_k(s, rel_idx, k=5))

    result = {
        "subset":    subset_name,
        "n_queries": len(queries),
        "n_docs":    len(unique_images),
        "ndcg@5":    round(float(np.mean(ndcg5)), 4),
        "recall@1":  round(float(np.mean(r1)), 4),
        "recall@5":  round(float(np.mean(r5)), 4),
    }
    print(f"  → nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  Recall@5={result['recall@5']:.4f}")
    return result

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets",     nargs="*", default=None, help="Run only these subsets (default: all)")
    parser.add_argument("--img-batch",   type=int,  default=2,    help="Batch size for image encoding")
    parser.add_argument("--query-batch", type=int,  default=8,    help="Batch size for query encoding")
    args = parser.parse_args()

    subsets_to_run = {
        k: v for k, v in SUBSETS.items()
        if args.subsets is None or k in args.subsets
    }

    print("=" * 55)
    print("  ColPali Baseline Evaluation on ViDoRe")
    print("=" * 55)
    print(f"  Model : {MODEL_PATH}")
    print(f"  Output: {OUT_CSV}")
    print(f"  Subsets ({len(subsets_to_run)}): {', '.join(subsets_to_run.keys())}")

    print("\nLoading model ...")
    model = ColPali.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded.\n")

    results = []
    t_start = time.time()

    for subset_name, rel_path in subsets_to_run.items():
        full_path = os.path.join(DATA_BASE, rel_path)
        df        = load_subset(full_path)
        result    = evaluate_subset(model, processor, df, subset_name, args.img_batch, args.query_batch)
        results.append(result)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t_start
    df_res  = pd.DataFrame(results)
    avg     = df_res[["ndcg@5", "recall@1", "recall@5"]].mean()

    print(f"\n{'='*55}")
    print("  FINAL RESULTS — ColPali on ViDoRe")
    print(f"{'='*55}")
    print(df_res[["subset", "ndcg@5", "recall@1", "recall@5"]].to_string(index=False))
    print(f"\n  Average  nDCG@5={avg['ndcg@5']:.4f}  Recall@1={avg['recall@1']:.4f}  Recall@5={avg['recall@5']:.4f}")
    print(f"  Total time: {elapsed/60:.1f} min")

    results.append({
        "subset": "AVERAGE", "n_queries": "", "n_docs": "",
        "ndcg@5":   round(avg["ndcg@5"],   4),
        "recall@1": round(avg["recall@1"], 4),
        "recall@5": round(avg["recall@5"], 4),
    })
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n  Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
