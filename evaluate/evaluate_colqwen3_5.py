#!/usr/bin/env python3
"""
Evaluate trained ColQwen3.5-0.8B (LoRA) on ViDoRe benchmark (fully offline).

Usage:
    python evaluate/evaluate_colqwen3_5.py
    python evaluate/evaluate_colqwen3_5.py --no-lora               # base model only
    python evaluate/evaluate_colqwen3_5.py --checkpoint checkpoints/colqwen3_5_lora/final
    python evaluate/evaluate_colqwen3_5.py --subsets arxivqa docvqa
    python evaluate/evaluate_colqwen3_5.py --img-batch 2 --query-batch 8
    python evaluate/evaluate_colqwen3_5.py --v2                    # ViDoRe v2 (BEIR-style)
    # proj128 model (linear projection head 2048→128):
    python evaluate/evaluate_colqwen3_5.py \
        --model-path /data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B \
        --checkpoint checkpoints/colqwen3_5_0.8B_proj128/final \
        --mode multivec_proj \
        --proj-dim 128 \
        --out-csv results/vidore_proj128.csv
"""

import os, sys, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import List

from peft import PeftModel
from colqwen3_5_embedding import ColQwen3_5ForEmbedding
from train.collator import ColPaliCollator


# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B"
DATA_BASE    = "dataset/vidore"
DATA_BASE_V2 = "dataset/vidore-v2"
DEFAULT_CKPT = "/data2/cmdir/home/test01/longvnu/graduation_thesis/checkpoints/colqwen3_5_triplet-en-0.8b/final"

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

SUBSETS_V2 = {
    "biomedical_v2":  "datasets--vidore--biomedical_lectures_v2/snapshots/c4754665734e38742b191f0c28d504e8558d0462",
    "economics_v2":   "datasets--vidore--economics_reports_v2/snapshots/76fe40166ba07b1bf50457f5c6057cacdd045f10",
    "esg_human_v2":   "datasets--vidore--esg_reports_human_labeled_v2/snapshots/5a338c329bf1608ac46ac2808060d44bcd92d521",
    "esg_v2":         "datasets--vidore--esg_reports_v2/snapshots/87538b12b20b67a2b4326638921301f87f0cbaf0",
}


# ─── Embedder ─────────────────────────────────────────────────────────────────

class ColQwen3_5Embedder:
    """Thin wrapper around trained ColQwen3_5ForEmbedding + LoRA."""

    def __init__(
        self, checkpoint: str | None, device: torch.device,
        mode: str = "multivec_mrl",
        projection_dim: int | None = None,
        model_path: str = MODEL_PATH,
        use_query_system_instruction: bool = True,
        use_doc_system_instruction: bool = True,
    ):
        self.device = device
        self.mode = mode
        self.projection_dim = projection_dim
        print(f"  Loading base model ({mode}) ...")

        if projection_dim:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path)
            cfg.embedding_dim = projection_dim
            self.model = ColQwen3_5ForEmbedding.from_pretrained(
                model_path, config=cfg
            ).to(device)
            print(f"  Projection head: {cfg.text_config.hidden_size} → {projection_dim}")
        else:
            self.model = ColQwen3_5ForEmbedding.from_pretrained(
                model_path
            ).to(device)

        if checkpoint:
            print(f"  Applying LoRA from {checkpoint} ...")
            self.model = PeftModel.from_pretrained(self.model, checkpoint)
            print(f"  LoRA active (unmerged).")

        self.model.eval()
        self.collator = ColPaliCollator(
            model_path=model_path,
            min_pixels=4096,
            max_pixels=1048576,
            query_instruction="" if not use_query_system_instruction else "Represent the user's input.",
            doc_instruction="" if not use_doc_system_instruction else "Represent the user's input.",
            use_query_system_instruction=use_query_system_instruction,
            use_doc_system_instruction=use_doc_system_instruction,
        )

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Returns list of tensors:
        - dense:         [proj_dim] per image (1D)
        - multivec_proj: [N_tokens, proj_dim] per image
        - multivec_mrl:  [N_tokens, D] per image
        """
        proc = self.collator.processor
        d_convs = [self.collator._make_doc_conv(img) for img in images]
        d_texts = proc.apply_chat_template(d_convs, add_generation_prompt=True, tokenize=False)
        inputs  = {k: v.to(self.device) for k, v in proc(
            text=d_texts, images=images, padding=True,
            truncation=True, max_length=2048, return_tensors="pt",
        ).items()}

        out  = self.model(**inputs)
        mask = inputs["attention_mask"].bool()
        h    = out.hidden_states          # keep in model dtype (bf16)
        linear_head = getattr(self.model, 'linear_head', None)

        if self.mode == "dense":
            # Pool to last real token, apply linear_head
            seq_len  = mask.sum(dim=1) - 1  # index of last real token
            pooled   = h[torch.arange(h.shape[0], device=self.device), seq_len]
            if linear_head is not None:
                pooled = linear_head(pooled)
            emb = F.normalize(pooled.float(), p=2, dim=-1)
            return [emb[i].cpu() for i in range(emb.shape[0])]
        else:
            if linear_head is not None:
                h = linear_head(h)
            emb = F.normalize(h.float(), p=2, dim=-1) * mask.unsqueeze(-1).float()
            return [emb[i][mask[i]].cpu() for i in range(emb.shape[0])]

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> List[torch.Tensor]:
        """Returns list of tensors:
        - dense:         [proj_dim] per query (1D)
        - multivec_proj: [N_tokens, proj_dim] per query
        - multivec_mrl:  [N_tokens, D] per query
        """
        proc = self.collator.processor
        q_convs = [self.collator._make_query_conv(q) for q in queries]
        q_texts = proc.apply_chat_template(q_convs, add_generation_prompt=True, tokenize=False)
        inputs  = {k: v.to(self.device) for k, v in proc(
            text=q_texts, padding=True,
            truncation=True, max_length=2048, return_tensors="pt",
        ).items()}

        # Text-only: provide explicit position_ids to skip compute_3d_position_ids
        b, seq = inputs["input_ids"].shape
        inputs["position_ids"] = (
            torch.arange(seq, device=self.device).unsqueeze(0).expand(b, -1).contiguous()
        )

        out  = self.model(**inputs)
        mask = inputs["attention_mask"].bool()
        h    = out.hidden_states          # keep in model dtype (bf16)
        linear_head = getattr(self.model, 'linear_head', None)

        if self.mode == "dense":
            seq_len = mask.sum(dim=1) - 1
            pooled  = h[torch.arange(h.shape[0], device=self.device), seq_len]
            if linear_head is not None:
                pooled = linear_head(pooled)
            emb = F.normalize(pooled.float(), p=2, dim=-1)
            return [emb[i].cpu() for i in range(emb.shape[0])]
        else:
            if linear_head is not None:
                h = linear_head(h)
            emb = F.normalize(h.float(), p=2, dim=-1) * mask.unsqueeze(-1).float()
            return [emb[i][mask[i]].cpu() for i in range(emb.shape[0])]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def ndcg_at_k(scores: np.ndarray, relevant_idx: int, k: int = 5) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    if relevant_idx not in top_k:
        return 0.0
    rank = int(np.where(top_k == relevant_idx)[0][0]) + 1
    return 1.0 / np.log2(rank + 1)   # IDCG=1 for single relevant doc

def recall_at_k(scores: np.ndarray, relevant_idx: int, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return 1.0 if relevant_idx in top_k else 0.0

# ── Multi-relevant variants (for ViDoRe v2 / BEIR qrels) ──────────────────────

def ndcg_at_k_multi(scores: np.ndarray, relevant_set: set, k: int = 5) -> float:
    """nDCG@k with multiple relevant docs (binary relevance, score ≥ 1 = relevant)."""
    if not relevant_set:
        return 0.0
    top_k = np.argsort(scores)[::-1][:k]
    dcg   = sum(1.0 / np.log2(rank + 2) for rank, idx in enumerate(top_k) if idx in relevant_set)
    idcg  = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k_multi(scores: np.ndarray, relevant_set: set, k: int) -> float:
    """Recall@k with multiple relevant docs."""
    if not relevant_set:
        return 0.0
    top_k = set(np.argsort(scores)[::-1][:k].tolist())
    return len(top_k & relevant_set) / len(relevant_set)


# ─── MaxSim (batched on GPU) ──────────────────────────────────────────────────

def truncate_and_renorm(embs: List[torch.Tensor], dim: int) -> List[torch.Tensor]:
    """Matryoshka truncation: slice to `dim` dims then L2-renorm. No projection needed."""
    return [F.normalize(e[..., :dim].float(), p=2, dim=-1) for e in embs]


def compute_scores_matrix(
    query_embs: List[torch.Tensor],
    doc_embs:   List[torch.Tensor],
    device:     torch.device,
) -> np.ndarray:
    """[N_q, N_d] MaxSim matrix — stacks per-batch on GPU then moves to CPU."""
    n_q = len(query_embs)
    n_d = len(doc_embs)
    scores = np.zeros((n_q, n_d), dtype=np.float32)

    # Stack docs once for efficiency
    max_d_len = max(e.shape[0] for e in doc_embs)
    dim       = doc_embs[0].shape[1]
    d_pad  = torch.zeros(n_d, max_d_len, dim)
    d_mask = torch.zeros(n_d, max_d_len, dtype=torch.bool)
    for i, e in enumerate(doc_embs):
        d_pad[i, :e.shape[0]] = e
        d_mask[i, :e.shape[0]] = True
    d_pad  = d_pad.to(device)
    d_mask = d_mask.to(device)

    BATCH_Q = 16
    for qi in tqdm(range(0, n_q, BATCH_Q), desc="    MaxSim", leave=False):
        q_batch = query_embs[qi : qi + BATCH_Q]
        max_q_len = max(e.shape[0] for e in q_batch)
        bq = len(q_batch)
        q_pad  = torch.zeros(bq, max_q_len, dim).to(device)
        for j, e in enumerate(q_batch):
            q_pad[j, :e.shape[0]] = e.to(device)

        # [bq, n_d, max_q_len, max_d_len] → max over d → sum over q
        sim = torch.einsum("bqd,cnd->bcqn", q_pad, d_pad)
        ms  = sim.max(dim=-1).values.sum(dim=-1)  # [bq, n_d]
        scores[qi : qi + bq] = ms.cpu().numpy()

    return scores


def compute_dense_scores_matrix(
    query_embs: List[torch.Tensor],
    doc_embs:   List[torch.Tensor],
    device:     torch.device,
) -> np.ndarray:
    """[N_q, N_d] dot-product matrix for dense 1D embeddings."""
    q_mat = torch.stack(query_embs).to(device)   # [N_q, D]
    d_mat = torch.stack(doc_embs).to(device)     # [N_d, D]
    scores = (q_mat @ d_mat.T).cpu().numpy()     # [N_q, N_d]
    return scores


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_subset(subset_path: str) -> pd.DataFrame:
    data_dir = os.path.join(subset_path, "data")
    parquets = sorted(f for f in os.listdir(data_dir) if f.startswith("test-") and f.endswith(".parquet"))
    if not parquets:
        raise FileNotFoundError(f"No test parquet files: {data_dir}")
    frames = [pd.read_parquet(os.path.join(data_dir, f)) for f in parquets]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

def decode_image(img_field) -> Image.Image:
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img_field)}")


# ─── Evaluate one v2 subset (BEIR-style: queries/ corpus/ qrels/) ─────────────

def evaluate_subset_v2(
    embedder: ColQwen3_5Embedder,
    subset_path: str,
    subset_name: str,
    img_batch: int,
    query_batch: int,
    dims: List[int],
) -> List[dict]:
    """Encode once at full dim, evaluate at each dim. Uses BEIR qrels (multi-relevant)."""
    print(f"\n{'─'*60}")
    print(f"  Subset (v2) : {subset_name}")

    # ── Load queries ──────────────────────────────────────────────────────────
    q_dir = os.path.join(subset_path, "queries")
    q_files = sorted(f for f in os.listdir(q_dir) if f.endswith(".parquet"))
    q_df = pd.concat([pd.read_parquet(os.path.join(q_dir, f)) for f in q_files], ignore_index=True)

    # ── Load corpus ───────────────────────────────────────────────────────────
    c_dir = os.path.join(subset_path, "corpus")
    c_files = sorted(f for f in os.listdir(c_dir) if f.endswith(".parquet"))
    c_df = pd.concat([pd.read_parquet(os.path.join(c_dir, f)) for f in c_files], ignore_index=True)

    # ── Load qrels ────────────────────────────────────────────────────────────
    ql_dir = os.path.join(subset_path, "qrels")
    ql_files = sorted(f for f in os.listdir(ql_dir) if f.endswith(".parquet"))
    ql_df = pd.concat([pd.read_parquet(os.path.join(ql_dir, f)) for f in ql_files], ignore_index=True)
    # Keep only answerable, score >= 1
    ql_df = ql_df[ql_df["score"] >= 1].copy()

    print(f"  Queries : {len(q_df)}  |  Corpus : {len(c_df)}  |  Qrels : {len(ql_df)}")

    # ── Build corpus index ────────────────────────────────────────────────────
    corpus_ids   = c_df["corpus-id"].tolist()
    cid_to_idx   = {cid: i for i, cid in enumerate(corpus_ids)}
    corpus_images = [decode_image(row["image"]) for _, row in c_df.iterrows()]

    # ── Build query list + relevant sets ─────────────────────────────────────
    qid_to_rel: dict = {}
    for _, row in ql_df.iterrows():
        qid = row["query-id"]
        cid = row["corpus-id"]
        if cid in cid_to_idx:
            qid_to_rel.setdefault(qid, set()).add(cid_to_idx[cid])

    valid_queries, valid_rel_sets = [], []
    for _, row in q_df.iterrows():
        qid = row["query-id"]
        q   = str(row.get("query", "")).strip()
        rel = qid_to_rel.get(qid, set())
        if len(q) < 5 or not rel:
            continue
        valid_queries.append(q)
        valid_rel_sets.append(rel)

    print(f"  Valid queries (with qrels): {len(valid_queries)}")

    # ── Encode ────────────────────────────────────────────────────────────────
    print(f"  Encoding corpus  (batch={img_batch}) ...")
    doc_embs_full: List[torch.Tensor] = []
    for i in tqdm(range(0, len(corpus_images), img_batch), desc="    img-batch", leave=False):
        doc_embs_full.extend(embedder.encode_images(corpus_images[i : i + img_batch]))

    print(f"  Encoding queries (batch={query_batch}) ...")
    q_embs_full: List[torch.Tensor] = []
    for i in tqdm(range(0, len(valid_queries), query_batch), desc="    qry-batch", leave=False):
        q_embs_full.extend(embedder.encode_queries(valid_queries[i : i + query_batch]))

    full_dim = doc_embs_full[0].shape[-1]

    # ── Dense mode: single-dim dot product ────────────────────────────────────
    if embedder.mode == "dense":
        print(f"  Computing dense dot-product scores (dim={full_dim}) ...")
        scores_matrix = compute_dense_scores_matrix(q_embs_full, doc_embs_full, embedder.device)
        ndcg5, r1, r5 = [], [], []
        for q_idx, rel_set in enumerate(valid_rel_sets):
            s = scores_matrix[q_idx]
            ndcg5.append(ndcg_at_k_multi(s, rel_set, k=5))
            r1.append(recall_at_k_multi(s, rel_set, k=1))
            r5.append(recall_at_k_multi(s, rel_set, k=5))
        result = {
            "subset": subset_name, "dims": full_dim,
            "n_queries": len(valid_queries), "n_docs": len(corpus_ids),
            "ndcg@5":   round(float(np.mean(ndcg5)), 4),
            "recall@1": round(float(np.mean(r1)),    4),
            "recall@5": round(float(np.mean(r5)),    4),
        }
        print(f"  [dim={full_dim:4d}] nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  Recall@5={result['recall@5']:.4f}")
        return [result]

    # ── Multivec modes: MaxSim over Matryoshka dims ────────────────────────────
    results = []
    for dim in dims:
        if dim > full_dim:
            print(f"  [skip] dim={dim} > model hidden_dim={full_dim}")
            continue

        print(f"  Computing MaxSim scores at dim={dim} ...")
        doc_embs = truncate_and_renorm(doc_embs_full, dim)
        q_embs   = truncate_and_renorm(q_embs_full,   dim)
        scores_matrix = compute_scores_matrix(q_embs, doc_embs, embedder.device)

        ndcg5, r1, r5 = [], [], []
        for q_idx, rel_set in enumerate(valid_rel_sets):
            s = scores_matrix[q_idx]
            ndcg5.append(ndcg_at_k_multi(s, rel_set, k=5))
            r1.append(recall_at_k_multi(s, rel_set, k=1))
            r5.append(recall_at_k_multi(s, rel_set, k=5))

        result = {
            "subset":    subset_name,
            "dims":      dim,
            "n_queries": len(valid_queries),
            "n_docs":    len(corpus_ids),
            "ndcg@5":    round(float(np.mean(ndcg5)), 4),
            "recall@1":  round(float(np.mean(r1)),    4),
            "recall@5":  round(float(np.mean(r5)),    4),
        }
        print(f"  [dim={dim:4d}] nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  Recall@5={result['recall@5']:.4f}")
        results.append(result)

    return results


# ─── Evaluate one subset ──────────────────────────────────────────────────────

def evaluate_subset(
    embedder: ColQwen3_5Embedder,
    df: pd.DataFrame,
    subset_name: str,
    img_batch: int,
    query_batch: int,
    dims: List[int],
) -> List[dict]:
    """Encode once at full dim, then evaluate at each requested dim via Matryoshka truncation."""
    print(f"\n{'─'*60}")
    print(f"  Subset : {subset_name}  ({len(df)} rows)")

    # Build corpus
    filenames     = df["image_filename"].tolist()
    unique_fnames = list(dict.fromkeys(filenames))
    fname_to_idx  = {fn: i for i, fn in enumerate(unique_fnames)}

    fname_to_img = {}
    for _, row in df.iterrows():
        fn = row["image_filename"]
        if fn not in fname_to_img:
            fname_to_img[fn] = decode_image(row["image"])
    unique_images = [fname_to_img[fn] for fn in unique_fnames]

    # Build queries
    valid_queries, valid_rel_idxs = [], []
    for _, row in df.iterrows():
        q = str(row.get("query", "")).strip()
        if q.lower() == "none" or len(q) < 5:
            continue
        valid_queries.append(q)
        valid_rel_idxs.append(fname_to_idx[row["image_filename"]])

    print(f"  Corpus : {len(unique_images)} unique images")
    print(f"  Queries: {len(valid_queries)}")

    # Encode ONCE at full dim — truncation happens per-dim below
    print(f"  Encoding images (batch={img_batch}) ...")
    doc_embs_full: List[torch.Tensor] = []
    for i in tqdm(range(0, len(unique_images), img_batch), desc="    img-batch", leave=False):
        doc_embs_full.extend(embedder.encode_images(unique_images[i : i + img_batch]))

    print(f"  Encoding queries (batch={query_batch}) ...")
    q_embs_full: List[torch.Tensor] = []
    for i in tqdm(range(0, len(valid_queries), query_batch), desc="    qry-batch", leave=False):
        q_embs_full.extend(embedder.encode_queries(valid_queries[i : i + query_batch]))

    full_dim = doc_embs_full[0].shape[-1]

    # ── Dense mode: single-dim dot product (no Matryoshka loop) ──────────────
    if embedder.mode == "dense":
        print(f"  Computing dense dot-product scores (dim={full_dim}) ...")
        scores_matrix = compute_dense_scores_matrix(q_embs_full, doc_embs_full, embedder.device)
        ndcg5, r1, r5 = [], [], []
        for q_idx, rel_idx in enumerate(valid_rel_idxs):
            s = scores_matrix[q_idx]
            ndcg5.append(ndcg_at_k(s, rel_idx, k=5))
            r1.append(recall_at_k(s, rel_idx, k=1))
            r5.append(recall_at_k(s, rel_idx, k=5))
        result = {
            "subset": subset_name, "dims": full_dim,
            "n_queries": len(valid_queries), "n_docs": len(unique_images),
            "ndcg@5":   round(float(np.mean(ndcg5)), 4),
            "recall@1": round(float(np.mean(r1)),    4),
            "recall@5": round(float(np.mean(r5)),    4),
        }
        print(f"  [dim={full_dim:4d}] nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  Recall@5={result['recall@5']:.4f}")
        return [result]

    # ── Multivec modes: MaxSim over Matryoshka dims ────────────────────────────
    results = []
    for dim in dims:
        if dim > full_dim:
            print(f"  [skip] dim={dim} > model hidden_dim={full_dim}")
            continue

        print(f"  Computing MaxSim scores at dim={dim} ...")
        doc_embs = truncate_and_renorm(doc_embs_full, dim)
        q_embs   = truncate_and_renorm(q_embs_full,   dim)
        scores_matrix = compute_scores_matrix(q_embs, doc_embs, embedder.device)

        ndcg5, r1, r5 = [], [], []
        for q_idx, rel_idx in enumerate(valid_rel_idxs):
            s = scores_matrix[q_idx]
            ndcg5.append(ndcg_at_k(s, rel_idx, k=5))
            r1.append(recall_at_k(s, rel_idx, k=1))
            r5.append(recall_at_k(s, rel_idx, k=5))

        result = {
            "subset":    subset_name,
            "dims":      dim,
            "n_queries": len(valid_queries),
            "n_docs":    len(unique_images),
            "ndcg@5":    round(float(np.mean(ndcg5)), 4),
            "recall@1":  round(float(np.mean(r1)), 4),
            "recall@5":  round(float(np.mean(r5)), 4),
        }
        print(f"  [dim={dim:4d}] nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  Recall@5={result['recall@5']:.4f}")
        results.append(result)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default=DEFAULT_CKPT, help="LoRA checkpoint path")
    parser.add_argument("--model-path",   default=MODEL_PATH,   help="Base model path (override MODEL_PATH)")
    parser.add_argument("--no-lora",      action="store_true",  help="Skip LoRA (base model only)")
    parser.add_argument("--mode",         default="multivec_mrl",
                        choices=["dense", "multivec_proj", "multivec_mrl"],
                        help="Embedding mode: dense | multivec_proj | multivec_mrl")
    parser.add_argument("--subsets",      nargs="*", default=None)
    parser.add_argument("--v2",           action="store_true", help="Evaluate on ViDoRe v2 (BEIR-style, multi-relevant)")
    parser.add_argument("--img-batch",    type=int,  default=2)
    parser.add_argument("--query-batch",  type=int,  default=8)
    parser.add_argument("--dims",         nargs="+", type=int,  default=[128, 256, 512, 1024, 2048],
                        help="Matryoshka dims to evaluate (e.g. --dims 128 256 512 1024 2048). "
                             "Encodes once at full dim then truncates per dim.")
    parser.add_argument("--proj-dim",     type=int,  default=None,
                        help="Projection head dim (e.g. 128). Only for proj-head-trained models.")
    parser.add_argument("--out-csv",      default="results/vidore_colqwen3_5.csv")
    parser.add_argument("--no-query-system-instruction", action="store_true")
    parser.add_argument("--no-doc-system-instruction", action="store_true")
    args = parser.parse_args()

    ckpt = None if args.no_lora else args.checkpoint
    tag  = "base" if args.no_lora else Path(args.checkpoint).name

    subsets_to_run = {
        k: v for k, v in SUBSETS.items()
        if args.subsets is None or k in args.subsets
    }

    subsets_v2_to_run = {
        k: v for k, v in SUBSETS_V2.items()
        if args.subsets is None or k in args.subsets
    } if args.v2 else {}

    print("=" * 60)
    print("  ColQwen3.5 Evaluation on ViDoRe benchmark")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt or 'none (base model)'}")
    print(f"  Mode       : {args.mode}")
    if subsets_to_run:
        print(f"  V1 subsets : {', '.join(subsets_to_run.keys())}")
    if subsets_v2_to_run:
        print(f"  V2 subsets : {', '.join(subsets_v2_to_run.keys())}")
    print(f"  Dims       : {args.dims}")
    if args.proj_dim:
        print(f"  Proj dim   : {args.proj_dim}")
    print(f"  Output     : {args.out_csv}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = ColQwen3_5Embedder(
        checkpoint=ckpt, device=device,
        mode=args.mode,
        projection_dim=args.proj_dim,
        model_path=args.model_path,
        use_query_system_instruction=not args.no_query_system_instruction,
        use_doc_system_instruction=not args.no_doc_system_instruction,
    )

    os.makedirs("results", exist_ok=True)
    results = []
    t_start = time.time()

    for subset_name, rel_path in subsets_to_run.items():
        full_path    = os.path.join(DATA_BASE, rel_path)
        df           = load_subset(full_path)
        sub_results  = evaluate_subset(
            embedder, df, subset_name, args.img_batch, args.query_batch, args.dims
        )
        for r in sub_results:
            r["checkpoint"] = tag
        results.extend(sub_results)
        pd.DataFrame(results).to_csv(args.out_csv, index=False)

    for subset_name, rel_path in subsets_v2_to_run.items():
        full_path   = os.path.join(DATA_BASE_V2, rel_path)
        sub_results = evaluate_subset_v2(
            embedder, full_path, subset_name, args.img_batch, args.query_batch, args.dims
        )
        for r in sub_results:
            r["checkpoint"] = tag
        results.extend(sub_results)
        pd.DataFrame(results).to_csv(args.out_csv, index=False)

    elapsed = time.time() - t_start
    df_res  = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  FINAL — ColQwen3.5-0.8B [{tag}] on ViDoRe")
    print(f"{'='*60}")

    for dim in args.dims:
        df_dim = df_res[df_res["dims"] == dim]
        if df_dim.empty:
            continue
        avg = df_dim[["ndcg@5", "recall@1", "recall@5"]].mean()
        print(f"\n  [dim={dim}]")
        print(df_dim[["subset", "n_queries", "n_docs", "ndcg@5", "recall@1", "recall@5"]].to_string(index=False))
        print(f"  Average  nDCG@5={avg['ndcg@5']:.4f}  Recall@1={avg['recall@1']:.4f}  Recall@5={avg['recall@5']:.4f}")

    print(f"  Total time: {elapsed/60:.1f} min")

    # Append per-dim AVERAGE rows
    avg_rows = []
    for dim in args.dims:
        df_dim = df_res[df_res["dims"] == dim]
        if df_dim.empty:
            continue
        avg = df_dim[["ndcg@5", "recall@1", "recall@5"]].mean()
        avg_rows.append({"subset": "AVERAGE", "dims": dim, "checkpoint": tag,
                         "ndcg@5": round(avg["ndcg@5"], 4),
                         "recall@1": round(avg["recall@1"], 4),
                         "recall@5": round(avg["recall@5"], 4)})
    pd.DataFrame(results + avg_rows).to_csv(args.out_csv, index=False)
    print(f"\n  Saved → {args.out_csv}")


if __name__ == "__main__":
    main()
