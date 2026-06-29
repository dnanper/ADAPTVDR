"""
Inference test dùng LoRA adapter trực tiếp (không merge vào base model).

Usage:
    # ColQwen3.5 — multi-vector MRL (no projection head)
    python infer_lora.py --mode multivec_mrl \\
        --lora checkpoints/colqwen3_5_lora-2b/final

    # ColQwen3.5 — multi-vector with projection head (1024→128)
    python infer_lora.py --mode multivec_proj --proj_dim 128 \\
        --lora checkpoints/colqwen3_5_0.8B_proj128/final

    # ColQwen3.5 — dense pooled embedding
    python infer_lora.py --mode dense --proj_dim 128 \\
        --lora checkpoints/colqwen3_5_lora_dense/final

    # ColPali
    python infer_lora.py \\
        --base  /data2/.../vidore/colpaligemma-3b-mix-448-base \\
        --lora  /data2/.../vidore/colpali \\
        --model_type colpali
"""
import argparse
import io
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
EVAL_PARQUET = (
    "dataset/vidore_train/"
    "datasets--vidore--colpali_train_set/"
    "snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data/"
    "test-00000-of-00001.parquet"
)
DOC_BATCH_SIZE   = 4
QUERY_BATCH_SIZE = 8

DEFAULT_MODEL_TYPE = "colqwen"  # or "colpali"
DEFAULT_BASE_MODEL = "/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B"
DEFAULT_LORA_MODEL = "/data2/cmdir/home/test01/longvnu/graduation_thesis/checkpoints/colqwen3_5_lora/checkpoint-1000"
# ─────────────────────────────────────────────────────────────────────────────


def load_pil(img_data) -> Image.Image:
    if isinstance(img_data, dict):
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    return img_data.convert("RGB")


def maxsim_scores(q_embs, d_embs) -> np.ndarray:
    """q_embs / d_embs: list of (B, L, D) tensors."""
    n_q = sum(e.shape[0] for e in q_embs)
    n_d = sum(e.shape[0] for e in d_embs)
    scores = np.zeros((n_q, n_d), dtype=np.float32)
    q_off = 0
    for qb in q_embs:
        d_off = 0
        for db in d_embs:
            sim = torch.einsum("bqd,cnd->bcqn", qb.float(), db.float())
            ms  = sim.max(dim=-1).values.sum(dim=-1)
            scores[q_off : q_off + qb.shape[0],
                   d_off : d_off + db.shape[0]] = ms.cpu().numpy()
            d_off += db.shape[0]
        q_off += qb.shape[0]
    return scores


def dot_scores(q_embs: torch.Tensor, d_embs: torch.Tensor) -> np.ndarray:
    """q_embs: (Q, D), d_embs: (N, D) — pooled dense embeddings."""
    return (q_embs.float() @ d_embs.float().T).cpu().numpy()


# ── ColQwen3.5 batched encode ─────────────────────────────────────────────────

def encode_colqwen_docs(images, embedder, batch_size=DOC_BATCH_SIZE):
    """Encode docs. Returns list of (B, Ld, D) for multivec modes, or cats to (N, D) for dense."""
    embs = []
    for i in tqdm(range(0, len(images), batch_size), desc="docs", leave=False):
        embs.append(embedder.encode_docs(images=images[i : i + batch_size]))
    if embedder.mode == "dense":
        return torch.cat(embs, dim=0)   # (N, D)
    return embs                         # list of (B, Ld, D)


def encode_colqwen_queries(queries, embedder, batch_size=QUERY_BATCH_SIZE):
    """Encode queries. Returns list of (B, Lq, D) for multivec modes, or cats to (Q, D) for dense."""
    embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="queries", leave=False):
        embs.append(embedder.encode_queries(queries[i : i + batch_size]))
    if embedder.mode == "dense":
        return torch.cat(embs, dim=0)   # (Q, D)
    return embs                         # list of (B, Lq, D)


# ── ColPali encode helpers ────────────────────────────────────────────────────

def encode_colpali_docs(images, model, processor, device, batch_size=DOC_BATCH_SIZE):
    embs = []
    for i in tqdm(range(0, len(images), batch_size), desc="docs", leave=False):
        batch  = images[i : i + batch_size]
        inputs = {k: v.to(device) for k, v in
                  processor.process_images(batch).items()}
        outputs = model(**inputs, output_hidden_states=True)
        last_h  = outputs.hidden_states[-1].float()
        emb     = model.custom_text_proj(last_h)
        emb     = F.normalize(emb, p=2, dim=-1)
        mask    = inputs["attention_mask"]
        emb     = emb * mask.unsqueeze(-1).float()
        embs.append(emb)
    return embs


def encode_colpali_queries(queries, model, processor, device, batch_size=QUERY_BATCH_SIZE):
    embs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="queries", leave=False):
        batch  = queries[i : i + batch_size]
        inputs = {k: v.to(device) for k, v in
                  processor.process_queries(batch).items()}
        outputs = model(**inputs, output_hidden_states=True)
        last_h  = outputs.hidden_states[-1].float()
        emb     = model.custom_text_proj(last_h)
        emb     = F.normalize(emb, p=2, dim=-1)
        mask    = inputs["attention_mask"]
        emb     = emb * mask.unsqueeze(-1).float()
        embs.append(emb)
    return embs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL,  help="path to base model dir")
    parser.add_argument("--lora", default=DEFAULT_LORA_MODEL,  help="path to LoRA adapter dir")
    parser.add_argument("--model_type", default="colqwen",
                        choices=["colqwen", "colpali"], help="which model family")
    parser.add_argument("--mode", default="multivec_mrl",
                        choices=["dense", "multivec_proj", "multivec_mrl"],
                        help="embedding mode: dense | multivec_proj | multivec_mrl")
    parser.add_argument("--proj_dim",  type=int, default=None,
                        help="projection head dim (required for dense / multivec_proj)")
    parser.add_argument("--n_docs",    type=int, default=9999)
    parser.add_argument("--n_queries", type=int, default=9999)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Model type : {args.model_type}")
    print(f"Mode       : {args.mode}")
    print(f"Base       : {args.base}")
    print(f"LoRA       : {args.lora}")

    # ── Load model + adapter ───────────────────────────────────────────────────
    if args.model_type == "colqwen":
        from scripts.colqwen3_5_embedding import ColQwen3_5Embedder

        embedder = ColQwen3_5Embedder(
            model_name_or_path=args.base,
            lora_checkpoint=args.lora,
            mode=args.mode,
            projection_dim=args.proj_dim,
        )
        processor = embedder.processor

    else:  # colpali
        from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
        from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

        base_model = ColPali.from_pretrained(
            args.base, torch_dtype=torch.bfloat16,
        ).to(device)
        embedder  = PeftModel.from_pretrained(base_model, args.lora)
        embedder.eval()
        processor = ColPaliProcessor.from_pretrained(args.base)

    print("Model + LoRA loaded (not merged).")

    # ── Load eval data ─────────────────────────────────────────────────────────
    df = pd.read_parquet(EVAL_PARQUET)
    df["_key"]   = df["image_filename"].astype(str)
    unique_keys  = list(dict.fromkeys(df["_key"]))[:args.n_docs]
    key_to_idx   = {k: i for i, k in enumerate(unique_keys)}

    key_to_img: dict = {}
    for _, row in df.iterrows():
        k = row["_key"]
        if k in unique_keys and k not in key_to_img:
            key_to_img[k] = load_pil(row["image"])
    corpus_imgs = [key_to_img[k] for k in unique_keys]

    query_rows  = df[df["_key"].isin(unique_keys)].head(args.n_queries)
    queries     = [str(r["query"]) for _, r in query_rows.iterrows()]
    gt_doc_idxs = [key_to_idx[r["_key"]] for _, r in query_rows.iterrows()]

    print(f"Corpus : {len(corpus_imgs)} docs — Queries: {len(queries)}")

    # ── Encode ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        if args.model_type == "colqwen":
            d_embs = encode_colqwen_docs(corpus_imgs, embedder)   # (N, D) or list
            q_embs = encode_colqwen_queries(queries,  embedder)   # (Q, D) or list
        else:
            d_embs = encode_colpali_docs(corpus_imgs, embedder, processor, device)
            q_embs = encode_colpali_queries(queries,  embedder, processor, device)

    # ── Score & metrics ───────────────────────────────────────────────────────
    if args.mode == "dense" and args.model_type == "colqwen":
        scores = dot_scores(q_embs, d_embs)
    else:
        scores = maxsim_scores(q_embs, d_embs)

    ranked = np.argsort(-scores, axis=1)

    hits_1 = hits_5 = 0
    ndcg_sum = 0.0
    for i, (query, gt) in enumerate(zip(queries, gt_doc_idxs)):
        top5 = ranked[i, :5].tolist()
        rank = (ranked[i] == gt).nonzero()[0][0] + 1
        tag  = "✓" if rank == 1 else ("~" if rank <= 5 else "✗")

        print(f"\n  Q{i+1}: {query[:80]}")
        print(f"       GT doc_idx={gt} | rank={rank} {tag}")
        for j, doc_idx in enumerate(top5):
            marker = ">> " if doc_idx == gt else "   "
            print(f"       {marker}top{j+1}: doc_{doc_idx}  score={scores[i, doc_idx]:.4f}")

        hits_1 += (rank == 1)
        hits_5 += (rank <= 5)
        ndcg_sum += (1.0 / math.log2(rank + 1)) if rank <= 5 else 0.0

    n = len(queries)
    print(f"\n{'='*60}")
    print(f"  Mode     : {args.mode}")
    print(f"  Recall@1 : {hits_1}/{n} = {hits_1/n*100:.1f}%")
    print(f"  Recall@5 : {hits_5}/{n} = {hits_5/n*100:.1f}%")
    print(f"  NDCG@5   : {ndcg_sum/n:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
