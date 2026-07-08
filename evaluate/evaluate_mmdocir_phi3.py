"""Evaluate Phi3 late-interaction retriever on MMDocIR page-level eval.

Official local layout:
  MMDocIR_pages.parquet       columns: file_name, page, image, ...
  MMDocIR_annotations.jsonl   each row has query + positive_passages
"""

import argparse
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.colphi3_embedding import ColPhi3ForEmbedding
from scripts.adaptive_pruning import AdaptivePruner
from train.phi3_collator import Phi3MMDocIRCollator


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, dict):
        value = value["bytes"]
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value)!r}")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def passage_key(passage: Dict[str, Any]) -> Tuple[str, int]:
    return str(passage["doc_name"]), int(passage["page_id"])


def annotation_query(rec: Dict[str, Any]) -> str:
    for key in ("query", "question", "generated_query"):
        if rec.get(key):
            return str(rec[key])
    raise ValueError(f"Annotation row has no query-like field: {sorted(rec.keys())}")


def maxsim_one(q: torch.Tensor, d: torch.Tensor) -> float:
    return float((q @ d.T).max(dim=1).values.sum().item())


@torch.no_grad()
def encode_queries(model, collator, queries: List[str], batch_size: int, device) -> List[torch.Tensor]:
    out = []
    for i in tqdm(range(0, len(queries), batch_size), desc="queries"):
        batch = collator._process_queries(queries[i : i + batch_size])
        query_mask = collator.make_query_token_mask(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        encoded = model(**batch)
        for emb, mask in zip(encoded.hidden_states, query_mask.to(device)):
            out.append(emb[mask].cpu())
    return out


@torch.no_grad()
def encode_docs(
    model,
    collator,
    images: List[Image.Image],
    batch_size: int,
    device,
    pruner: AdaptivePruner | None = None,
) -> Tuple[List[torch.Tensor], List[float]]:
    out = []
    keep_ratios = []
    for i in tqdm(range(0, len(images), batch_size), desc="pages"):
        batch = collator._process_docs(images[i : i + batch_size])
        doc_mask = collator.make_doc_token_mask(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        encoded = model(**batch, output_attentions=pruner is not None)
        if pruner is not None:
            pruned, stats = pruner.prune_doc(
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            out.extend([emb.cpu() for emb in pruned])
            keep_ratios.extend(stats.keep_ratios)
        else:
            for emb, mask in zip(encoded.hidden_states, doc_mask.to(device)):
                out.append(emb[mask].cpu())
    return out, keep_ratios


def ndcg_at_k(ranked_pages: List[int], relevant_pages: set, k: int) -> float:
    dcg = 0.0
    for rank, page in enumerate(ranked_pages[:k], start=1):
        if page in relevant_pages:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(relevant_pages), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_pages: List[int], relevant_pages: set, k: int) -> float:
    return 1.0 if relevant_pages.intersection(ranked_pages[:k]) else 0.0


def load_pages(path: Path):
    df = pd.read_parquet(path, columns=["file_name", "page", "image"])
    grouped = defaultdict(list)
    for row in df.itertuples(index=False):
        grouped[str(row.file_name)].append((int(row.page), decode_image(row.image)))
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-root", default=None, help="Directory containing MMDocIR_pages.parquet and MMDocIR_annotations.jsonl.")
    parser.add_argument("--pages", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=None, help="Evaluate only the first N annotation rows.")
    parser.add_argument("--image-size", type=int, default=None, help="Fixed square resize. Omit for dynamic pixel bounds.")
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--prune-docs", action="store_true", help="Apply adaptive patch pruning to document embeddings.")
    parser.add_argument("--prune-r-min", type=float, default=0.3)
    parser.add_argument("--prune-r-max", type=float, default=0.9)
    parser.add_argument("--prune-mode", choices=["linear", "perplexity"], default="linear")
    parser.add_argument("--prune-tau", type=float, default=2.0)
    args = parser.parse_args()

    if not args.eval_root and (not args.pages or not args.annotations):
        raise ValueError("Provide --eval-root or both --pages and --annotations")
    eval_root = Path(args.eval_root) if args.eval_root else None
    pages_path = Path(args.pages) if args.pages else eval_root / "MMDocIR_pages.parquet"
    ann_path = Path(args.annotations) if args.annotations else eval_root / "MMDocIR_annotations.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColPhi3ForEmbedding(args.model, projection_dim=128).to(device)
    model = PeftModel.from_pretrained(model, args.checkpoint).to(device)
    model.eval()
    collator = Phi3MMDocIRCollator(
        args.model,
        image_size=args.image_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    pruner = None
    if args.prune_docs:
        pruner = AdaptivePruner(
            r_min=args.prune_r_min,
            r_max=args.prune_r_max,
            mode=args.prune_mode,
            tau=args.prune_tau,
            image_token_id=collator.image_token_id,
            keep_text_tokens=False,
        )

    pages_by_doc = load_pages(pages_path)
    annotations = list(read_jsonl(ann_path))
    if args.max_queries is not None:
        annotations = annotations[: max(0, int(args.max_queries))]
    queries = [annotation_query(row) for row in annotations]
    query_embs = encode_queries(model, collator, queries, args.batch_size, device)

    doc_cache: Dict[str, Tuple[List[int], List[torch.Tensor]]] = {}
    all_keep_ratios: List[float] = []
    metrics = {"r1": [], "r5": [], "r10": [], "ndcg5": []}
    skipped = 0

    for rec, q_emb in tqdm(list(zip(annotations, query_embs)), desc="score"):
        positives = list(rec.get("positive_passages") or [])
        if not positives:
            skipped += 1
            continue
        doc_name, _ = passage_key(positives[0])
        relevant = {page for doc, page in map(passage_key, positives) if doc == doc_name}
        if doc_name not in pages_by_doc:
            skipped += 1
            continue
        if doc_name not in doc_cache:
            page_ids, images = zip(*sorted(pages_by_doc[doc_name], key=lambda x: x[0]))
            doc_embs, keep_ratios = encode_docs(model, collator, list(images), args.batch_size, device, pruner=pruner)
            all_keep_ratios.extend(keep_ratios)
            doc_cache[doc_name] = (
                list(page_ids),
                doc_embs,
            )

        page_ids, doc_embs = doc_cache[doc_name]
        page_scores = torch.tensor([maxsim_one(q_emb, d_emb) for d_emb in doc_embs])
        ranked_idx = torch.argsort(page_scores, descending=True).tolist()
        ranked_pages = [page_ids[i] for i in ranked_idx]
        metrics["r1"].append(recall_at_k(ranked_pages, relevant, 1))
        metrics["r5"].append(recall_at_k(ranked_pages, relevant, 5))
        metrics["r10"].append(recall_at_k(ranked_pages, relevant, 10))
        metrics["ndcg5"].append(ndcg_at_k(ranked_pages, relevant, 5))

    denom = max(len(metrics["r1"]), 1)
    print(f"queries: {denom}  skipped: {skipped}")
    print(f"Recall@1: {sum(metrics['r1']) / denom:.4f}")
    print(f"Recall@5: {sum(metrics['r5']) / denom:.4f}")
    print(f"Recall@10: {sum(metrics['r10']) / denom:.4f}")
    print(f"nDCG@5: {sum(metrics['ndcg5']) / denom:.4f}")
    if all_keep_ratios:
        print(f"Pruning mean keep ratio: {sum(all_keep_ratios) / len(all_keep_ratios):.4f}")


if __name__ == "__main__":
    main()
