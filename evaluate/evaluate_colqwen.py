#!/usr/bin/env python3
"""
Baseline evaluation of ColQwen3 on ViDoRe benchmark (fully offline).

Usage:
    python evaluate_colqwen.py
    python evaluate_colqwen.py --subsets arxivqa docvqa   # run specific subsets only
    python evaluate_colqwen.py --img-batch 1              # reduce if VRAM OOM
"""

import os, sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import List

from scripts.colqwen3_vl_embedding_4b import OpsColQwen3Embedder


# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "/data2/cmdir/home/test01/longvnu/stable_diff/models/OpenSearchAI/ColQwen3-VL"
DATA_BASE   = "/data2/cmdir/home/test01/longvnu/graduation_thesis/dataset/vidore"
OUT_CSV     = "/data2/cmdir/home/test01/longvnu/graduation_thesis/results_colqwen3.csv"

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
    rank = int(np.where(top_k == relevant_idx)[0][0]) + 1  # 1-indexed
    return (1.0 / np.log2(rank + 1)) / (1.0 / np.log2(2))  # DCG / IDCG

def recall_at_k(scores: np.ndarray, relevant_idx: int, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return 1.0 if relevant_idx in top_k else 0.0

# ─── MaxSim ───────────────────────────────────────────────────────────────────
def compute_scores_matrix(
    query_embs: List[torch.Tensor],
    doc_embs: List[torch.Tensor],
) -> np.ndarray:
    """
    Compute [n_queries, n_docs] MaxSim matrix.
    MaxSim(q, d) = sum_over_query_tokens( max_over_doc_patches( q_i · d_j ) )
    """
    device = query_embs[0].device if query_embs[0].is_cuda else torch.device("cpu")
    n_q, n_d = len(query_embs), len(doc_embs)
    scores = np.zeros((n_q, n_d), dtype=np.float32)

    for q_idx, q_emb in enumerate(tqdm(query_embs, desc="    MaxSim", leave=False)):
        q = q_emb.float().to(device)  # [n_q_tokens, dim]
        for d_idx, d_emb in enumerate(doc_embs):
            d = d_emb.float().to(device)  # [n_d_patches, dim]
            sim = torch.einsum("qd,pd->qp", q, d)  # [n_q_tokens, n_d_patches]
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

# ─── Evaluate one subset ──────────────────────────────────────────────────────
def evaluate_subset(
    embedder: OpsColQwen3Embedder,
    df: pd.DataFrame,
    subset_name: str,
    img_batch: int,
    query_batch: int,
) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Subset : {subset_name}  ({len(df)} rows)")

    # 1. TẠO KHO TÀI LIỆU (CORPUS): Giữ nguyên toàn bộ ảnh, không được lọc bỏ!
    filenames     = df["image_filename"].tolist()
    unique_fnames = list(dict.fromkeys(filenames))
    fname_to_idx  = {fn: i for i, fn in enumerate(unique_fnames)}

    fname_to_img = {}
    for _, row in df.iterrows():
        fn = row["image_filename"]
        if fn not in fname_to_img:
            fname_to_img[fn] = decode_image(row["image"])
    unique_images = [fname_to_img[fn] for fn in unique_fnames]

    # 2. LỌC ĐỀ BÀI (QUERIES): Chỉ lấy những câu hỏi có nghĩa và ID đáp án tương ứng
    valid_queries = []
    valid_relevant_idxs = []

    for _, row in df.iterrows():
        q_text = str(row.get("query", "")).strip()
        
        # Bộ lọc ma thuật: Đá văng các thể loại rác, None, hoặc câu quá ngắn
        if q_text.lower() == "none" or len(q_text) < 5:
            continue
            
        valid_queries.append(q_text)
        # Lưu lại vị trí (index) của tài liệu đúng trong cái kho 1000 ảnh kia
        valid_relevant_idxs.append(fname_to_idx[row["image_filename"]])

    queries = valid_queries
    relevant_idxs = valid_relevant_idxs

    print(f"  Corpus : {len(unique_images)} unique images")
    print(f"  Queries: {len(queries)}")

    print(f"  Encoding images (batch={img_batch}) ...")
    image_embs: List[torch.Tensor] = []
    for i in tqdm(range(0, len(unique_images), img_batch), desc="    img-batch", unit="batch"):
        image_embs.extend(embedder.encode_images(unique_images[i : i + img_batch]))

    print(f"  Encoding queries (batch={query_batch}) ...")
    query_embs: List[torch.Tensor] = []
    for i in tqdm(range(0, len(queries), query_batch), desc="    qry-batch", unit="batch"):
        query_embs.extend(embedder.encode_queries(queries[i : i + query_batch]))

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
    parser.add_argument("--subsets",    nargs="*", default=None, help="Run only these subsets (default: all)")
    parser.add_argument("--img-batch",  type=int,  default=2,    help="Batch size for image encoding")
    parser.add_argument("--query-batch",type=int,  default=8,    help="Batch size for query encoding")
    args = parser.parse_args()

    subsets_to_run = {
        k: v for k, v in SUBSETS.items()
        if args.subsets is None or k in args.subsets
    }

    print("=" * 55)
    print("  ColQwen3 Baseline Evaluation on ViDoRe")
    print("=" * 55)
    print(f"  Model : {MODEL_PATH}")
    print(f"  Output: {OUT_CSV}")
    print(f"  Subsets ({len(subsets_to_run)}): {', '.join(subsets_to_run.keys())}")

    print("\nLoading model ...")
    embedder = OpsColQwen3Embedder(model_name=MODEL_PATH, dims=128)
    print("Model loaded.\n")

    results = []
    t_start = time.time()

    for subset_name, rel_path in subsets_to_run.items():
        full_path = os.path.join(DATA_BASE, rel_path)
        df        = load_subset(full_path)
        result    = evaluate_subset(embedder, df, subset_name, args.img_batch, args.query_batch)
        results.append(result)

        # Save incrementally after each subset
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    # Summary
    elapsed = time.time() - t_start
    df_res  = pd.DataFrame(results)
    avg     = df_res[["ndcg@5", "recall@1", "recall@5"]].mean()

    print(f"\n{'='*55}")
    print("  FINAL RESULTS — ColQwen3 on ViDoRe")
    print(f"{'='*55}")
    print(df_res[["subset", "ndcg@5", "recall@1", "recall@5"]].to_string(index=False))
    print(f"\n  Average  nDCG@5={avg['ndcg@5']:.4f}  Recall@1={avg['recall@1']:.4f}  Recall@5={avg['recall@5']:.4f}")
    print(f"  Total time: {elapsed/60:.1f} min")

    # Append average row to CSV
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
