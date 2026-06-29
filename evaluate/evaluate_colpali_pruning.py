#!/usr/bin/env python3
"""Evaluate the ColPali PaliGemma LoRA adapter with pruning on ViDoRe v1/v2.

Variants:
    - adapter                  : no pruning
    - linear_pruning           : entropy-based linear keep ratio
    - perplexity_tau{1,2,3}    : perplexity keep ratio with temperature tau

All retrieval scores are computed with Matryoshka-truncated 128-d token vectors.

Usage:
    python evaluate/evaluate_colpali_pruning.py
    python evaluate/evaluate_colpali_pruning.py --subsets arxivqa economics_v2
    python evaluate/evaluate_colpali_pruning.py --adapter-path checkpoints/colpali_vidore_mrl-3b/checkpoint-3000
"""

import argparse
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from adaptive_pruning import AdaptivePruner, IMAGE_TOKEN_ID_COLPALI, PruningStats
from colpali_mrl_embedding import ColPaliMRLEmbedder


DEFAULT_ADAPTER_PATH = "checkpoints/colpali_vidore_mrl-3b/checkpoint-3000"
DATA_BASE_V1 = "dataset/vidore"
DATA_BASE_V2 = "dataset/vidore-v2"
OUT_CSV = "results/vidore_colpali_adapter_pruning_eval.csv"

SUBSETS_V1 = {
    "arxivqa": "datasets--vidore--arxivqa_test_subsampled/snapshots/b8a106812c8682bab08935cf5d1b4566c82562de",
    "docvqa": "datasets--vidore--docvqa_test_subsampled/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6",
    "infovqa": "datasets--vidore--infovqa_test_subsampled/snapshots/f793e830aaeae1ceb8a2df626fc555b3fd04d3db",
    "shiftproject": "datasets--vidore--shiftproject_test/snapshots/6e6223b3839a3f4e62d676e70a1b715ee520cc66",
    "synth_ai": "datasets--vidore--syntheticDocQA_artificial_intelligence_test/snapshots/5694fec64c57f7380f918435f9234b96d817b82a",
    "synth_energy": "datasets--vidore--syntheticDocQA_energy_test/snapshots/a2f2f358463b03b84506849460ab5094a358526c",
    "synth_gov": "datasets--vidore--syntheticDocQA_government_reports_test/snapshots/91cf66572d89c9cccac0661de227acaf04b44f64",
    "synth_health": "datasets--vidore--syntheticDocQA_healthcare_industry_test/snapshots/d973f3da3e60c8eaf566efa2f7d2a1515cba40ed",
    "tabfquad": "datasets--vidore--tabfquad_test_subsampled/snapshots/16c8e633612fbda7400bfcbbc31d61a7534f580f",
    "tatdqa": "datasets--vidore--tatdqa_test/snapshots/b46fea43695e14697510104a3331d9e88683a416",
}

SUBSETS_V2 = {
    "biomedical_v2": "datasets--vidore--biomedical_lectures_v2/snapshots/c4754665734e38742b191f0c28d504e8558d0462",
    "economics_v2": "datasets--vidore--economics_reports_v2/snapshots/76fe40166ba07b1bf50457f5c6057cacdd045f10",
    "esg_human_v2": "datasets--vidore--esg_reports_human_labeled_v2/snapshots/5a338c329bf1608ac46ac2808060d44bcd92d521",
    "esg_v2": "datasets--vidore--esg_reports_v2/snapshots/87538b12b20b67a2b4326638921301f87f0cbaf0",
}


def ndcg_at_k(scores: np.ndarray, relevant_idx: int, k: int = 5) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    if relevant_idx not in top_k:
        return 0.0
    rank = int(np.where(top_k == relevant_idx)[0][0]) + 1
    return 1.0 / np.log2(rank + 1)


def recall_at_k(scores: np.ndarray, relevant_idx: int, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return 1.0 if relevant_idx in top_k else 0.0


def ndcg_at_k_multi(scores: np.ndarray, relevant_set: set, k: int = 5) -> float:
    if not relevant_set:
        return 0.0
    top_k = np.argsort(scores)[::-1][:k]
    dcg = sum(1.0 / np.log2(rank + 2) for rank, idx in enumerate(top_k) if idx in relevant_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k_multi(scores: np.ndarray, relevant_set: set, k: int) -> float:
    if not relevant_set:
        return 0.0
    top_k = set(np.argsort(scores)[::-1][:k].tolist())
    return len(top_k & relevant_set) / len(relevant_set)


def load_subset_v1(subset_path: str) -> pd.DataFrame:
    data_dir = os.path.join(subset_path, "data")
    parquets = sorted(f for f in os.listdir(data_dir) if f.startswith("test-") and f.endswith(".parquet"))
    if not parquets:
        raise FileNotFoundError(f"No test parquet files found in {data_dir}")
    frames = [pd.read_parquet(os.path.join(data_dir, f)) for f in parquets]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_parquet_dir(dir_path: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(dir_path) if f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dir_path}")
    frames = [pd.read_parquet(os.path.join(dir_path, file_name)) for file_name in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def decode_image(img_field) -> Image.Image:
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img_field)}")


def tensor_kb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 1024.0


def aggregate_pruning_stats(stats_list: Sequence[PruningStats]) -> PruningStats:
    merged = PruningStats()
    for stats in stats_list:
        merged.original_patches.extend(stats.original_patches)
        merged.kept_patches.extend(stats.kept_patches)
        merged.keep_ratios.extend(stats.keep_ratios)
    return merged


def summarize_docs(doc_embs: Sequence[torch.Tensor]) -> Dict[str, float]:
    tokens = [int(doc.shape[0]) for doc in doc_embs]
    doc_kb = [tensor_kb(doc) for doc in doc_embs]
    return {
        "avg_doc_tokens": float(np.mean(tokens)) if tokens else 0.0,
        "avg_doc_kb": float(np.mean(doc_kb)) if doc_kb else 0.0,
        "total_doc_kb": float(np.sum(doc_kb)) if doc_kb else 0.0,
    }


@torch.no_grad()
def encode_queries(
    embedder: ColPaliMRLEmbedder,
    queries: Sequence[str],
    batch_size: int,
) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    for start in tqdm(range(0, len(queries), batch_size), desc="    qry-batch", leave=False):
        batch_inputs = embedder.process_queries(list(queries[start : start + batch_size]))
        batch_out = embedder._forward(batch_inputs, output_attentions=False)
        batch_mask = batch_out.retrieval_mask.bool()
        for batch_idx in range(batch_mask.shape[0]):
            outputs.append(batch_out.embeddings[batch_idx][batch_mask[batch_idx]].cpu())
    return outputs


@torch.no_grad()
def encode_images_baseline(
    embedder: ColPaliMRLEmbedder,
    images: Sequence[Image.Image],
    batch_size: int,
) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    for start in tqdm(range(0, len(images), batch_size), desc="    img-base", leave=False):
        batch_inputs = embedder.process_images(list(images[start : start + batch_size]))
        batch_out = embedder._forward(batch_inputs, output_attentions=False)
        batch_mask = batch_out.retrieval_mask.bool()
        for batch_idx in range(batch_mask.shape[0]):
            outputs.append(batch_out.embeddings[batch_idx][batch_mask[batch_idx]].cpu())
    return outputs


@torch.no_grad()
def encode_images_pruned(
    embedder: ColPaliMRLEmbedder,
    images: Sequence[Image.Image],
    batch_size: int,
    pruner: AdaptivePruner,
) -> Tuple[List[torch.Tensor], PruningStats]:
    outputs: List[torch.Tensor] = []
    stats_list: List[PruningStats] = []
    for start in tqdm(range(0, len(images), batch_size), desc="    img-prune", leave=False):
        batch_inputs = embedder.process_images(list(images[start : start + batch_size]))
        batch_out = embedder._forward(batch_inputs, output_attentions=True)
        pruned_batch, stats = pruner.prune_doc(
            hidden_states=batch_out.embeddings,
            attentions=batch_out.attentions,
            input_ids=batch_out.input_ids,
            attention_mask=batch_out.attention_mask,
        )
        outputs.extend([doc.cpu() for doc in pruned_batch])
        stats_list.append(stats)
    return outputs, aggregate_pruning_stats(stats_list)


def compute_scores_matrix(
    query_embs: Sequence[torch.Tensor],
    doc_embs: Sequence[torch.Tensor],
    device: torch.device,
    query_chunk: int,
    doc_chunk: int,
) -> np.ndarray:
    n_q = len(query_embs)
    n_d = len(doc_embs)
    scores = np.zeros((n_q, n_d), dtype=np.float32)
    dim = int(doc_embs[0].shape[-1])

    for q_start in tqdm(range(0, n_q, query_chunk), desc="    MaxSim", leave=False):
        q_batch = query_embs[q_start : q_start + query_chunk]
        q_max_len = max(int(q.shape[0]) for q in q_batch)
        q_pad = torch.zeros(len(q_batch), q_max_len, dim, dtype=torch.float32, device=device)
        for q_idx, q_emb in enumerate(q_batch):
            q_pad[q_idx, : q_emb.shape[0]] = q_emb.float().to(device)

        for d_start in range(0, n_d, doc_chunk):
            d_batch = doc_embs[d_start : d_start + doc_chunk]
            d_max_len = max(int(d.shape[0]) for d in d_batch)
            d_pad = torch.zeros(len(d_batch), d_max_len, dim, dtype=torch.float32, device=device)
            for d_idx, d_emb in enumerate(d_batch):
                d_pad[d_idx, : d_emb.shape[0]] = d_emb.float().to(device)

            sim = torch.einsum("bqd,cnd->bcqn", q_pad, d_pad)
            maxsim = sim.max(dim=-1).values.sum(dim=-1)
            scores[q_start : q_start + len(q_batch), d_start : d_start + len(d_batch)] = maxsim.cpu().numpy()

    return scores


def evaluate_scores_v1(scores_matrix: np.ndarray, relevant_idxs: Sequence[int]) -> Dict[str, float]:
    ndcg5, r1, r5 = [], [], []
    for q_idx, rel_idx in enumerate(relevant_idxs):
        scores = scores_matrix[q_idx]
        ndcg5.append(ndcg_at_k(scores, rel_idx, k=5))
        r1.append(recall_at_k(scores, rel_idx, k=1))
        r5.append(recall_at_k(scores, rel_idx, k=5))
    return {
        "ndcg@5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "recall@1": float(np.mean(r1)) if r1 else 0.0,
        "recall@5": float(np.mean(r5)) if r5 else 0.0,
    }


def evaluate_scores_v2(scores_matrix: np.ndarray, relevant_sets: Sequence[set]) -> Dict[str, float]:
    ndcg5, r1, r5 = [], [], []
    for q_idx, relevant_set in enumerate(relevant_sets):
        scores = scores_matrix[q_idx]
        ndcg5.append(ndcg_at_k_multi(scores, relevant_set, k=5))
        r1.append(recall_at_k_multi(scores, relevant_set, k=1))
        r5.append(recall_at_k_multi(scores, relevant_set, k=5))
    return {
        "ndcg@5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "recall@1": float(np.mean(r1)) if r1 else 0.0,
        "recall@5": float(np.mean(r5)) if r5 else 0.0,
    }


def build_v1_payload(df: pd.DataFrame) -> Tuple[List[Image.Image], List[str], List[int]]:
    filenames = df["image_filename"].tolist()
    unique_fnames = list(dict.fromkeys(filenames))
    fname_to_idx = {file_name: idx for idx, file_name in enumerate(unique_fnames)}

    fname_to_img = {}
    for _, row in df.iterrows():
        file_name = row["image_filename"]
        if file_name not in fname_to_img:
            fname_to_img[file_name] = decode_image(row["image"])
    images = [fname_to_img[file_name] for file_name in unique_fnames]

    queries, relevant_idxs = [], []
    for _, row in df.iterrows():
        query = str(row.get("query", "")).strip()
        if query.lower() == "none" or len(query) < 5:
            continue
        queries.append(query)
        relevant_idxs.append(fname_to_idx[row["image_filename"]])

    return images, queries, relevant_idxs


def build_v2_payload(subset_path: str) -> Tuple[List[Image.Image], List[str], List[set]]:
    queries_df = load_parquet_dir(os.path.join(subset_path, "queries"))
    corpus_dir = os.path.join(subset_path, "corpus")
    if not os.path.isdir(corpus_dir):
        corpus_dir = os.path.join(subset_path, "docs")
    corpus_df = load_parquet_dir(corpus_dir)
    qrels_df = load_parquet_dir(os.path.join(subset_path, "qrels"))
    qrels_df = qrels_df[qrels_df["score"] >= 1].copy()

    corpus_ids = corpus_df["corpus-id"].tolist()
    corpus_id_to_idx = {corpus_id: idx for idx, corpus_id in enumerate(corpus_ids)}
    images = [decode_image(row["image"]) for _, row in corpus_df.iterrows()]

    query_id_to_rel = {}
    for _, row in qrels_df.iterrows():
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        if corpus_id in corpus_id_to_idx:
            query_id_to_rel.setdefault(query_id, set()).add(corpus_id_to_idx[corpus_id])

    queries, relevant_sets = [], []
    for _, row in queries_df.iterrows():
        query_id = row["query-id"]
        query = str(row.get("query", "")).strip()
        relevant_set = query_id_to_rel.get(query_id, set())
        if len(query) < 5 or not relevant_set:
            continue
        queries.append(query)
        relevant_sets.append(relevant_set)

    return images, queries, relevant_sets


def format_tau(tau: float) -> str:
    tau_float = float(tau)
    return str(int(tau_float)) if tau_float.is_integer() else str(tau_float).replace(".", "_")


def build_variant_specs(perplexity_taus: Sequence[float]) -> List[dict]:
    variants = [
        {
            "variant": "adapter",
            "pruning_mode": "none",
            "tau": None,
        },
        {
            "variant": "linear_pruning",
            "pruning_mode": "linear",
            "tau": None,
        },
    ]
    for tau in perplexity_taus:
        variants.append(
            {
                "variant": f"perplexity_tau{format_tau(float(tau))}",
                "pruning_mode": "perplexity",
                "tau": float(tau),
            }
        )
    return variants


def compare_variants(
    subset_name: str,
    dataset_version: str,
    images: Sequence[Image.Image],
    queries: Sequence[str],
    relevant_targets: Sequence,
    embedder: ColPaliMRLEmbedder,
    variant_specs: Sequence[dict],
    img_batch: int,
    query_batch: int,
    query_chunk: int,
    doc_chunk: int,
    r_min: float,
    r_max: float,
) -> List[dict]:
    print(f"\n{'─' * 68}")
    print(f"  Subset [{dataset_version}] : {subset_name}")
    print(f"  Corpus : {len(images)} docs")
    print(f"  Queries: {len(queries)}")

    print(f"  Encoding queries (bf16 storage, batch={query_batch}) ...")
    query_embs = encode_queries(embedder, queries, query_batch)

    print(f"  Encoding adapter docs (bf16 storage, batch={img_batch}) ...")
    baseline_doc_embs = encode_images_baseline(embedder, images, img_batch)

    baseline_summary = summarize_docs(baseline_doc_embs)
    results = []
    for spec in variant_specs:
        variant_name = spec["variant"]
        if spec["pruning_mode"] == "none":
            doc_embs = baseline_doc_embs
            variant_stats = None
        else:
            pruner = AdaptivePruner(
                r_min=r_min,
                r_max=r_max,
                mode=spec["pruning_mode"],
                tau=spec["tau"] if spec["tau"] is not None else 2.0,
                image_token_id=IMAGE_TOKEN_ID_COLPALI,
                layer_idx=-1,
                normalize=True,
                keep_text_tokens=False,
            )
            print(f"  Encoding {variant_name} docs (bf16 storage, batch={img_batch}) ...")
            doc_embs, variant_stats = encode_images_pruned(embedder, images, img_batch, pruner)

        print(f"  Computing {variant_name} MaxSim scores ...")
        scores_matrix = compute_scores_matrix(query_embs, doc_embs, embedder.device, query_chunk, doc_chunk)
        metrics = evaluate_scores_v1(scores_matrix, relevant_targets) if dataset_version == "v1" else evaluate_scores_v2(scores_matrix, relevant_targets)
        summary = summarize_docs(doc_embs)
        keep_ratio = variant_stats.mean_keep_ratio if variant_stats is not None else 1.0
        result = {
            "subset": subset_name,
            "dataset": dataset_version,
            "variant": variant_name,
            "pruning_mode": spec["pruning_mode"],
            "tau": spec["tau"],
            "embedding_dim": int(doc_embs[0].shape[-1]),
            "dtype": str(doc_embs[0].dtype),
            "n_queries": len(queries),
            "n_docs": len(images),
            "ndcg@5": round(metrics["ndcg@5"], 4),
            "recall@1": round(metrics["recall@1"], 4),
            "recall@5": round(metrics["recall@5"], 4),
            "avg_doc_tokens": round(summary["avg_doc_tokens"], 2),
            "avg_doc_kb": round(summary["avg_doc_kb"], 4),
            "total_doc_kb": round(summary["total_doc_kb"], 2),
            "mean_keep_ratio": round(keep_ratio, 4),
            "storage_saving_pct": round(
                100.0 * (1.0 - (summary["avg_doc_kb"] / baseline_summary["avg_doc_kb"])),
                2,
            ) if baseline_summary["avg_doc_kb"] > 0 else 0.0,
        }
        print(
            f"  [{variant_name}] nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  "
            f"Recall@5={result['recall@5']:.4f}  avg_kb/page={result['avg_doc_kb']:.4f}"
        )
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--subsets", nargs="*", default=None, help="Subset names across v1/v2")
    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--img-batch", type=int, default=2)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--doc-chunk", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--perplexity-taus", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--r-min", type=float, default=0.3)
    parser.add_argument("--r-max", type=float, default=0.99)
    parser.add_argument("--out-csv", default=OUT_CSV)
    args = parser.parse_args()

    subsets_v1 = {
        key: value for key, value in SUBSETS_V1.items()
        if not args.skip_v1 and (args.subsets is None or key in args.subsets)
    }
    subsets_v2 = {
        key: value for key, value in SUBSETS_V2.items()
        if not args.skip_v2 and (args.subsets is None or key in args.subsets)
    }

    if not subsets_v1 and not subsets_v2:
        raise ValueError("No subsets selected. Check --subsets / --skip-v1 / --skip-v2.")

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    print("=" * 68)
    print("  ColPali Adapter Benchmark: adapter vs pruning variants")
    print("=" * 68)
    print(f"  Adapter path: {args.adapter_path}")
    if args.base_model_path:
        print(f"  Base model  : {args.base_model_path}")
    if subsets_v1:
        print(f"  V1 subsets  : {', '.join(subsets_v1.keys())}")
    if subsets_v2:
        print(f"  V2 subsets  : {', '.join(subsets_v2.keys())}")
    print(f"  Embed dim   : {args.embed_dim}")
    print(f"  Variants    : {', '.join(spec['variant'] for spec in build_variant_specs(args.perplexity_taus))}")
    print(f"  r_min/r_max : {args.r_min} / {args.r_max}")
    print(f"  Output CSV  : {args.out_csv}")

    variant_specs = build_variant_specs(args.perplexity_taus)
    embedder = ColPaliMRLEmbedder(
        base_model_name_or_path=args.base_model_path,
        lora_checkpoint=args.adapter_path,
        embed_dim=args.embed_dim,
        attn_implementation="eager",
    )

    results: List[dict] = []
    started_at = time.time()

    for subset_name, rel_path in subsets_v1.items():
        df = load_subset_v1(os.path.join(DATA_BASE_V1, rel_path))
        images, queries, relevant_idxs = build_v1_payload(df)
        results.extend(
            compare_variants(
                subset_name=subset_name,
                dataset_version="v1",
                images=images,
                queries=queries,
                relevant_targets=relevant_idxs,
                embedder=embedder,
                variant_specs=variant_specs,
                img_batch=args.img_batch,
                query_batch=args.query_batch,
                query_chunk=args.query_chunk,
                doc_chunk=args.doc_chunk,
                r_min=args.r_min,
                r_max=args.r_max,
            )
        )
        pd.DataFrame(results).to_csv(args.out_csv, index=False)

    for subset_name, rel_path in subsets_v2.items():
        images, queries, relevant_sets = build_v2_payload(os.path.join(DATA_BASE_V2, rel_path))
        results.extend(
            compare_variants(
                subset_name=subset_name,
                dataset_version="v2",
                images=images,
                queries=queries,
                relevant_targets=relevant_sets,
                embedder=embedder,
                variant_specs=variant_specs,
                img_batch=args.img_batch,
                query_batch=args.query_batch,
                query_chunk=args.query_chunk,
                doc_chunk=args.doc_chunk,
                r_min=args.r_min,
                r_max=args.r_max,
            )
        )
        pd.DataFrame(results).to_csv(args.out_csv, index=False)

    elapsed_min = (time.time() - started_at) / 60.0
    df_results = pd.DataFrame(results)

    avg_rows = []
    for dataset_name in sorted(df_results["dataset"].unique().tolist()):
        for variant_name in [spec["variant"] for spec in variant_specs]:
            df_slice = df_results[(df_results["dataset"] == dataset_name) & (df_results["variant"] == variant_name)]
            if df_slice.empty:
                continue
            avg_rows.append({
                "subset": "AVERAGE",
                "dataset": dataset_name,
                "variant": variant_name,
                "pruning_mode": str(df_slice["pruning_mode"].iloc[0]),
                "tau": df_slice["tau"].iloc[0],
                "embedding_dim": int(df_slice["embedding_dim"].iloc[0]),
                "dtype": "torch.bfloat16",
                "n_queries": int(df_slice["n_queries"].sum()),
                "n_docs": int(df_slice["n_docs"].sum()),
                "ndcg@5": round(float(df_slice["ndcg@5"].mean()), 4),
                "recall@1": round(float(df_slice["recall@1"].mean()), 4),
                "recall@5": round(float(df_slice["recall@5"].mean()), 4),
                "avg_doc_tokens": round(float(df_slice["avg_doc_tokens"].mean()), 2),
                "avg_doc_kb": round(float(df_slice["avg_doc_kb"].mean()), 4),
                "total_doc_kb": round(float(df_slice["total_doc_kb"].sum()), 2),
                "mean_keep_ratio": round(float(df_slice["mean_keep_ratio"].mean()), 4),
                "storage_saving_pct": round(float(df_slice["storage_saving_pct"].mean()), 2),
            })

    final_rows = results + avg_rows
    pd.DataFrame(final_rows).to_csv(args.out_csv, index=False)

    print(f"\n{'=' * 68}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 68}")
    for dataset_name in ["v1", "v2"]:
        dataset_rows = [row for row in avg_rows if row["dataset"] == dataset_name]
        if not dataset_rows:
            continue
        print(f"\n  [{dataset_name}]")
        print(pd.DataFrame(dataset_rows)[[
            "variant", "ndcg@5", "recall@1", "recall@5", "avg_doc_tokens", "avg_doc_kb", "storage_saving_pct"
        ]].to_string(index=False))
    print(f"\n  Total time: {elapsed_min:.1f} min")
    print(f"  Saved → {args.out_csv}")


if __name__ == "__main__":
    main()
