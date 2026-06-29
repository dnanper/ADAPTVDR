"""
Quick inference test with trained model (merged or base).

Usage:
    python infer.py
    python infer.py --model model/Qwen/Qwen3.5-0.8B-Embedding
    python infer.py --n_docs 30 --n_queries 10
"""
import argparse
import io
import math

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from scripts.colqwen3_5_embedding import ColQwen3_5Embedder

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "/data2/cmdir/home/test01/longvnu/graduation_thesis/model/ColQwen3.5-0.8B-Embedding"
EVAL_PARQUET  = (
    "dataset/vidore_train/"
    "datasets--vidore--colpali_train_set/"
    "snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data/"
    "test-00000-of-00001.parquet"
)
DOC_BATCH_SIZE   = 4
QUERY_BATCH_SIZE = 8
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pil(img_data) -> Image.Image:
    if isinstance(img_data, dict):
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    return img_data.convert("RGB")


def encode_docs(images, embedder: ColQwen3_5Embedder, batch_size: int = DOC_BATCH_SIZE):
    """Encode a list of PIL images → list of [B, N, D] tensors (GPU)."""
    embs = []
    for i in tqdm(range(0, len(images), batch_size), desc="encoding docs", leave=False):
        batch = [
            {
                "image": img,
                "instruction": "Represent this document image to support information retrieval based on its text and layout."
            }
        for img in images[i : i + batch_size]]
        emb, _ = embedder.process(batch, normalize=True, pooling=False)
        embs.append(emb)  # keep on GPU
    return embs  # list of [B, N, D]


def encode_queries(queries, embedder: ColQwen3_5Embedder, batch_size: int = QUERY_BATCH_SIZE):
    """Encode a list of text queries → list of [B, N, D] tensors (GPU)."""
    embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="encoding queries", leave=False):
        batch = [
            {
                "text": q, 
                "instruction": "Represent the following query for searching relevant document pages."
            } 
        for q in queries[i : i + batch_size]]
        emb, _ = embedder.process(batch, normalize=True, pooling=False)
        embs.append(emb)  # keep on GPU
    return embs  # list of [B, N, D]


def maxsim_scores(q_embs, d_embs) -> np.ndarray:
    n_q = sum(e.shape[0] for e in q_embs)
    n_d = sum(e.shape[0] for e in d_embs)
    scores = np.zeros((n_q, n_d), dtype=np.float32)
    q_off = 0
    for qb in q_embs:
        d_off = 0
        for db in d_embs:
            sim = torch.einsum("bqd,cnd->bcqn", qb.float(), db.float())  # GPU
            ms  = sim.max(dim=-1).values.sum(dim=-1)                      # GPU
            scores[q_off : q_off + qb.shape[0],
                   d_off : d_off + db.shape[0]] = ms.cpu().numpy()        # pull only scores
            d_off += db.shape[0]
        q_off += qb.shape[0]
    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=DEFAULT_MODEL,  help="path to merged model dir")
    parser.add_argument("--n_docs",    type=int, default=9999,  help="# unique docs in corpus (default: all)")
    parser.add_argument("--n_queries", type=int, default=9999,  help="# queries to test (default: all)")
    parser.add_argument("--embed_dim", type=int, default=None,  help="truncate to this dim (Matryoshka)")
    args = parser.parse_args()

    # ── Load embedder ──────────────────────────────────────────────────────────
    print(f"Loading embedder from: {args.model}")
    embedder = ColQwen3_5Embedder(
        model_name_or_path=args.model,
        embed_dim=args.embed_dim,
    )
    print(f"Model loaded. embed_dim={args.embed_dim or 'full'}")

    # ── Load eval data ─────────────────────────────────────────────────────────
    print(f"Loading eval parquet: {EVAL_PARQUET}")
    df = pd.read_parquet(EVAL_PARQUET)

    df["_key"] = df["image_filename"].astype(str)
    unique_keys = list(dict.fromkeys(df["_key"]))[:args.n_docs]
    key_to_idx  = {k: i for i, k in enumerate(unique_keys)}

    key_to_img: dict = {}
    for _, row in df.iterrows():
        k = row["_key"]
        if k in unique_keys and k not in key_to_img:
            key_to_img[k] = load_pil(row["image"])

    corpus_imgs = [key_to_img[k] for k in unique_keys]

    # Build query list with ground-truth doc index
    query_rows = df[df["_key"].isin(unique_keys)].head(args.n_queries)
    queries      = [str(r["query"]) for _, r in query_rows.iterrows()]
    gt_doc_idxs  = [key_to_idx[r["_key"]] for _, r in query_rows.iterrows()]

    print(f"Corpus : {len(corpus_imgs)} docs")
    print(f"Queries: {len(queries)}")

    # ── Encode ─────────────────────────────────────────────────────────────────
    d_embs = encode_docs(corpus_imgs, embedder)
    q_embs = encode_queries(queries,  embedder)

    # ── MaxSim retrieval ───────────────────────────────────────────────────────
    scores = maxsim_scores(q_embs, d_embs)   # [N_q, N_d]
    ranked = np.argsort(-scores, axis=1)     # [N_q, N_d] descending

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Retrieval results (top-3 per query)")
    print(f"{'='*70}")

    hits_1 = hits_5 = 0
    ndcg_sum = 0.0
    for i, (query, gt) in enumerate(zip(queries, gt_doc_idxs)):
        top5  = ranked[i, :5].tolist()
        rank  = (ranked[i] == gt).nonzero()[0][0] + 1  # 1-indexed rank
        tag   = "✓" if rank == 1 else ("~" if rank <= 5 else "✗")

        print(f"\n  Q{i+1}: {query[:80]}")
        print(f"       GT doc_idx={gt} | rank={rank} {tag}")
        for j, doc_idx in enumerate(top5):
            marker = ">> " if doc_idx == gt else "   "
            print(f"       {marker}top{j+1}: doc_{doc_idx}  score={scores[i, doc_idx]:.4f}")

        hits_1 += (rank == 1)
        hits_5 += (rank <= 5)
        ndcg_sum += (1.0 / math.log2(rank + 1)) if rank <= 5 else 0.0

    n = len(queries)
    print(f"\n{'='*70}")
    print(f"  Recall@1 : {hits_1}/{n} = {hits_1/n*100:.1f}%")
    print(f"  Recall@5 : {hits_5}/{n} = {hits_5/n*100:.1f}%")
    print(f"  NDCG@5   : {ndcg_sum/n:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
