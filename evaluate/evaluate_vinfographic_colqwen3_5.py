#!/usr/bin/env python3
"""
Benchmark ColQwen3.5 retrieval on ViInfographic.

Retrieval protocol:
- Query: question text
- Documents: all unique images within each split pool
- Relevant docs:
  - single_test: the sample's `image_path`
  - multi_test: all images in the sample's `image_paths`

Outputs:
- summary CSV with split-level metrics
- predictions CSV with top-k retrieved docs per query
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

MODEL_PATH = "/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B"


DEFAULT_DATASET_ROOT = ROOT / "dataset" / "vinfographic"
DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "colqwen3_5_lora-0.8b"
    / "ColQwen3.5-0.8B-Embedding-Vietnamese"
    / "final"
)
DEFAULT_SPLITS = ["single_test", "multi_test"]


def get_runtime_components():
    try:
        from evaluate.evaluate_colqwen3_5 import (
            ColQwen3_5Embedder,
            compute_dense_scores_matrix,
            compute_scores_matrix,
            truncate_and_renorm,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing runtime dependency while importing ColQwen3.5 evaluation stack. "
            "Install the project's eval dependencies first, especially `peft`."
        ) from exc

    return {
        "ColQwen3_5Embedder": ColQwen3_5Embedder,
        "compute_dense_scores_matrix": compute_dense_scores_matrix,
        "compute_scores_matrix": compute_scores_matrix,
        "truncate_and_renorm": truncate_and_renorm,
    }


def ndcg_at_k_multi(scores: np.ndarray, relevant_set: Set[int], k: int = 5) -> float:
    if not relevant_set:
        return 0.0
    top_k = np.argsort(scores)[::-1][:k]
    dcg = sum(1.0 / np.log2(rank + 2) for rank, idx in enumerate(top_k) if int(idx) in relevant_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k_multi(scores: np.ndarray, relevant_set: Set[int], k: int) -> float:
    if not relevant_set:
        return 0.0
    top_k = set(int(idx) for idx in np.argsort(scores)[::-1][:k].tolist())
    return len(top_k & relevant_set) / len(relevant_set)


def hard_hit_at_k(scores: np.ndarray, relevant_set: Set[int], k: int) -> float:
    if not relevant_set:
        return 0.0
    top_k = set(int(idx) for idx in np.argsort(scores)[::-1][:k].tolist())
    return 1.0 if relevant_set.issubset(top_k) else 0.0


def ordered_hard_hit_at_k(scores: np.ndarray, relevant_order: List[int], k: int) -> float:
    if not relevant_order:
        return 0.0
    ranked = [int(idx) for idx in np.argsort(scores)[::-1][:k].tolist()]
    retrieved_relevant_in_order = [idx for idx in ranked if idx in set(relevant_order)]
    if len(retrieved_relevant_in_order) < len(relevant_order):
        return 0.0
    return 1.0 if retrieved_relevant_in_order[: len(relevant_order)] == relevant_order else 0.0


def reciprocal_rank(scores: np.ndarray, relevant_set: Set[int]) -> float:
    if not relevant_set:
        return 0.0
    ranked = np.argsort(scores)[::-1]
    for rank, doc_idx in enumerate(ranked, start=1):
        if int(doc_idx) in relevant_set:
            return 1.0 / rank
    return 0.0


def load_vinfographic_split(dataset_root: Union[str, Path], split: str) -> dict:
    dataset_root = Path(dataset_root)
    split_path = dataset_root / "data" / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    with split_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    doc_paths: List[str] = []
    doc_index: Dict[str, int] = {}
    queries = []

    for sample in samples:
        question = str(sample.get("question", "")).strip()
        if len(question) < 3:
            continue

        if "image_path" in sample:
            positive_paths = [sample["image_path"]]
        elif "image_paths" in sample:
            positive_paths = list(sample["image_paths"])
        else:
            raise KeyError(f"Sample {sample.get('question_id')} has no image_path/image_paths")

        dedup_positive_paths = list(dict.fromkeys(positive_paths))
        positive_doc_indices = set()
        for rel_path in dedup_positive_paths:
            if rel_path not in doc_index:
                doc_index[rel_path] = len(doc_paths)
                doc_paths.append(rel_path)
            positive_doc_indices.add(doc_index[rel_path])

        queries.append(
            {
                "question_id": str(sample.get("question_id", "")),
                "question": question,
                "positive_doc_indices": positive_doc_indices,
                "positive_doc_paths": dedup_positive_paths,
                "answer": str(sample.get("answer", "")),
                "answer_source": str(sample.get("answer_source", "")),
                "image_type": str(sample.get("image_type", "")),
                "element": str(sample.get("element", "")),
            }
        )

    return {
        "split": split,
        "split_path": split_path,
        "doc_paths": doc_paths,
        "doc_abs_paths": [str(dataset_root / rel_path) for rel_path in doc_paths],
        "queries": queries,
    }


def compute_split_metrics(
    scores_matrix: np.ndarray,
    positive_sets: List[Set[int]],
    positive_orders: List[List[int]],
    ks: Iterable[int] = (1, 5, 10),
    ndcg_k: int = 5,
) -> dict:
    metrics = {f"recall@{k}": [] for k in ks}
    hard_metrics = {f"hard_recall@{k}": [] for k in ks}
    ordered_hard_metrics = {f"ordered_hard_recall@{k}": [] for k in ks}
    rr_scores = []
    ndcg_scores = []

    for q_idx, relevant_set in enumerate(positive_sets):
        scores = scores_matrix[q_idx]
        relevant_order = positive_orders[q_idx]
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k_multi(scores, relevant_set, k))
            hard_metrics[f"hard_recall@{k}"].append(hard_hit_at_k(scores, relevant_set, k))
            ordered_hard_metrics[f"ordered_hard_recall@{k}"].append(
                ordered_hard_hit_at_k(scores, relevant_order, k)
            )
        rr_scores.append(reciprocal_rank(scores, relevant_set))
        ndcg_scores.append(ndcg_at_k_multi(scores, relevant_set, k=ndcg_k))

    summary = {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}
    summary.update(
        {name: float(np.mean(values)) if values else 0.0 for name, values in hard_metrics.items()}
    )
    summary.update(
        {
            name: float(np.mean(values)) if values else 0.0
            for name, values in ordered_hard_metrics.items()
        }
    )
    summary["mrr"] = float(np.mean(rr_scores)) if rr_scores else 0.0
    summary[f"ndcg@{ndcg_k}"] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return summary


def build_prediction_rows(split_bundle: dict, scores_matrix: np.ndarray, dim: int, top_k: int) -> List[dict]:
    rows = []
    doc_paths = split_bundle["doc_paths"]
    for q_idx, query in enumerate(split_bundle["queries"]):
        ranked = np.argsort(scores_matrix[q_idx])[::-1][:top_k]
        positive_paths = [doc_paths[idx] for idx in sorted(query["positive_doc_indices"])]
        for rank, doc_idx in enumerate(ranked, start=1):
            rows.append(
                {
                    "split": split_bundle["split"],
                    "dims": dim,
                    "question_id": query["question_id"],
                    "question": query["question"],
                    "answer": query.get("answer", ""),
                    "answer_source": query.get("answer_source", ""),
                    "image_type": query.get("image_type", ""),
                    "element": query.get("element", ""),
                    "rank": rank,
                    "retrieved_doc_path": doc_paths[int(doc_idx)],
                    "score": float(scores_matrix[q_idx, int(doc_idx)]),
                    "is_relevant": int(doc_idx) in query["positive_doc_indices"],
                    "positive_doc_paths": "|".join(positive_paths),
                }
            )
    return rows


def load_rgb_image(path: Union[str, Path]) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def encode_images(
    embedder,
    doc_paths: List[str],
    img_batch: int,
) -> List[torch.Tensor]:
    embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(doc_paths), img_batch), desc="    img-batch", leave=False):
        batch_paths = doc_paths[start : start + img_batch]
        images = [load_rgb_image(path) for path in batch_paths]
        embeddings.extend(embedder.encode_images(images))
    return embeddings


def encode_queries(
    embedder,
    queries: List[str],
    query_batch: int,
) -> List[torch.Tensor]:
    embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(queries), query_batch), desc="    qry-batch", leave=False):
        embeddings.extend(embedder.encode_queries(queries[start : start + query_batch]))
    return embeddings


def evaluate_split(
    embedder,
    split_bundle: dict,
    img_batch: int,
    query_batch: int,
    dims: List[int],
    top_k: int,
    recall_ks: Iterable[int],
    runtime: dict,
) -> Tuple[List[dict], List[dict]]:
    split_name = split_bundle["split"]
    doc_abs_paths = split_bundle["doc_abs_paths"]
    queries = split_bundle["queries"]
    query_texts = [query["question"] for query in queries]
    positive_sets = [query["positive_doc_indices"] for query in queries]
    doc_idx_by_path = {path: idx for idx, path in enumerate(split_bundle["doc_paths"])}
    positive_orders = [
        [doc_idx_by_path[path] for path in query["positive_doc_paths"]]
        for query in queries
    ]

    print(f"\n{'=' * 60}")
    print(f"  Split      : {split_name}")
    print(f"  Documents  : {len(doc_abs_paths)}")
    print(f"  Queries    : {len(query_texts)}")

    print(f"  Encoding images (batch={img_batch}) ...")
    doc_embs_full = encode_images(embedder, doc_abs_paths, img_batch)

    print(f"  Encoding queries (batch={query_batch}) ...")
    query_embs_full = encode_queries(embedder, query_texts, query_batch)

    full_dim = doc_embs_full[0].shape[-1]
    eval_dims = [full_dim] if embedder.mode == "dense" else [dim for dim in dims if dim <= full_dim]
    if not eval_dims:
        raise ValueError(f"No valid dims to evaluate. Requested={dims}, available={full_dim}")

    summary_rows = []
    prediction_rows = []

    for dim in eval_dims:
        if embedder.mode == "dense":
            print(f"  Computing dense scores (dim={dim}) ...")
            scores_matrix = runtime["compute_dense_scores_matrix"](
                query_embs_full, doc_embs_full, embedder.device
            )
        else:
            print(f"  Computing MaxSim scores (dim={dim}) ...")
            doc_embs = runtime["truncate_and_renorm"](doc_embs_full, dim)
            query_embs = runtime["truncate_and_renorm"](query_embs_full, dim)
            scores_matrix = runtime["compute_scores_matrix"](query_embs, doc_embs, embedder.device)

        metric_values = compute_split_metrics(
            scores_matrix,
            positive_sets,
            positive_orders,
            ks=recall_ks,
            ndcg_k=5,
        )
        row = {
            "split": split_name,
            "dims": dim,
            "n_queries": len(query_texts),
            "n_docs": len(doc_abs_paths),
        }
        row.update({name: round(value, 4) for name, value in metric_values.items()})
        summary_rows.append(row)

        recall_msg = "  ".join(
            f"Recall@{k}={metric_values[f'recall@{k}']:.4f}" for k in recall_ks
        )
        hard_msg = "  ".join(
            f"Hard@{k}={metric_values[f'hard_recall@{k}']:.4f}" for k in recall_ks
        )
        ordered_msg = "  ".join(
            f"OrdHard@{k}={metric_values[f'ordered_hard_recall@{k}']:.4f}" for k in recall_ks
        )
        print(
            f"  [dim={dim:4d}] {recall_msg}  {hard_msg}  {ordered_msg}  "
            f"MRR={metric_values['mrr']:.4f}  nDCG@5={metric_values['ndcg@5']:.4f}"
        )

        prediction_rows.extend(
            build_prediction_rows(
                split_bundle=split_bundle,
                scores_matrix=scores_matrix,
                dim=dim,
                top_k=min(top_k, len(doc_abs_paths)),
            )
        )

    return summary_rows, prediction_rows


def aggregate_summary_rows(summary_rows: List[dict], recall_ks: Iterable[int]) -> List[dict]:
    aggregate_rows = []
    if not summary_rows:
        return aggregate_rows

    df = pd.DataFrame(summary_rows)
    metric_columns = (
        [f"recall@{k}" for k in recall_ks]
        + [f"hard_recall@{k}" for k in recall_ks]
        + [f"ordered_hard_recall@{k}" for k in recall_ks]
        + ["mrr", "ndcg@5"]
    )
    for dim, df_dim in df.groupby("dims", sort=True):
        weights = df_dim["n_queries"].to_numpy(dtype=np.float64)
        weight_sum = float(weights.sum())
        row = {
            "split": "ALL",
            "dims": int(dim),
            "n_queries": int(df_dim["n_queries"].sum()),
            "n_docs": int(df_dim["n_docs"].sum()),
        }
        for metric in metric_columns:
            values = df_dim[metric].to_numpy(dtype=np.float64)
            row[metric] = round(float(np.average(values, weights=weights)) if weight_sum else 0.0, 4)
        aggregate_rows.append(row)
    return aggregate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Path to dataset/vinfographic")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="LoRA checkpoint path")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Base model path")
    parser.add_argument("--no-lora", action="store_true", help="Evaluate the base model without LoRA")
    parser.add_argument(
        "--mode",
        default="multivec_mrl",
        choices=["dense", "multivec_proj", "multivec_mrl"],
        help="Embedding mode",
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, help="Splits to evaluate")
    parser.add_argument("--img-batch", type=int, default=2)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--dims", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--proj-dim", type=int, default=None, help="Projection dim for multivec_proj/dense")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k predictions to export per query")
    parser.add_argument("--summary-csv", default="results/vinfographic_colqwen3_5.csv")
    parser.add_argument("--predictions-csv", default="results/vinfographic_colqwen3_5_predictions.csv")
    args = parser.parse_args()

    recall_ks = (1, 5, 10)
    checkpoint = None if args.no_lora else args.checkpoint
    tag = "base" if args.no_lora else Path(args.checkpoint).name

    print("=" * 60)
    print("  ColQwen3.5 Retrieval Benchmark on ViInfographic")
    print("=" * 60)
    print(f"  Dataset    : {args.dataset_root}")
    print(f"  Splits     : {', '.join(args.splits)}")
    print(f"  Checkpoint : {checkpoint or 'none (base model)'}")
    print(f"  Mode       : {args.mode}")
    print(f"  Dims       : {args.dims}")
    print(f"  Summary    : {args.summary_csv}")
    print(f"  Predictions: {args.predictions_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = get_runtime_components()
    embedder = runtime["ColQwen3_5Embedder"](
        checkpoint=checkpoint,
        device=device,
        mode=args.mode,
        projection_dim=args.proj_dim,
        model_path=args.model_path,
    )

    summary_rows = []
    prediction_rows = []
    started_at = time.time()

    for split in args.splits:
        split_bundle = load_vinfographic_split(args.dataset_root, split)
        split_summary, split_predictions = evaluate_split(
            embedder=embedder,
            split_bundle=split_bundle,
            img_batch=args.img_batch,
            query_batch=args.query_batch,
            dims=args.dims,
            top_k=args.top_k,
            recall_ks=recall_ks,
            runtime=runtime,
        )
        for row in split_summary:
            row["checkpoint"] = tag
        for row in split_predictions:
            row["checkpoint"] = tag
        summary_rows.extend(split_summary)
        prediction_rows.extend(split_predictions)

    summary_rows.extend(aggregate_summary_rows(summary_rows, recall_ks))
    for row in summary_rows:
        row.setdefault("checkpoint", tag)

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    os.makedirs(Path(args.summary_csv).parent, exist_ok=True)
    os.makedirs(Path(args.predictions_csv).parent, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    predictions_df.to_csv(args.predictions_csv, index=False)

    elapsed = time.time() - started_at
    print(f"\n{'=' * 60}")
    print(f"  FINAL — ViInfographic Retrieval [{tag}]")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))
    print(f"\n  Total time: {elapsed / 60:.1f} min")
    print(f"  Saved summary     -> {args.summary_csv}")
    print(f"  Saved predictions -> {args.predictions_csv}")


if __name__ == "__main__":
    main()
