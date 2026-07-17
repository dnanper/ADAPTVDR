"""Evaluate Phi3 late-interaction retriever on MMDocIR page-level eval.

Official local layout:
  MMDocIR_pages.parquet       columns: doc_name/passage_id/image_binary or file_name/page/image
  MMDocIR_annotations.jsonl   rows have query/positive_passages or questions/page_indices
"""

import argparse
import io
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
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


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _page_ids_for_question(page_indices: List[Any], question_idx: int, question_count: int) -> List[int]:
    if not page_indices:
        return []
    if question_count == 1:
        return [_page_id(page) for page in page_indices]
    if len(page_indices) == question_count:
        value = page_indices[question_idx]
        return [_page_id(page) for page in _as_list(value)]
    return [_page_id(page) for page in page_indices]


def normalize_annotations(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for rec in rows:
        if rec.get("positive_passages"):
            normalized.append(rec)
            continue

        questions = _as_list(rec.get("questions"))
        if not questions:
            normalized.append(rec)
            continue

        doc_name = str(rec["doc_name"])
        page_indices = _as_list(rec.get("page_indices"))
        for idx, item in enumerate(questions):
            if isinstance(item, dict):
                query = str(item.get("Q") or item.get("query") or item.get("question") or "").strip()
                page_ids = [_page_id(page) for page in _as_list(item.get("page_id"))]
            else:
                query = str(item).strip()
                page_ids = _page_ids_for_question(page_indices, idx, len(questions))
            if not query:
                continue
            normalized.append(
                {
                    "query": query,
                    "positive_passages": [
                        {"doc_name": doc_name, "page_id": page_id}
                        for page_id in page_ids
                    ],
                }
            )
    return normalized


def maxsim_one(q: torch.Tensor, d: torch.Tensor) -> float:
    return float((q @ d.T).max(dim=1).values.sum().item())


def clone_tensor_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def encode_queries(model, collator, queries: List[str], batch_size: int, device) -> List[torch.Tensor]:
    out = []
    for i in tqdm(range(0, len(queries), batch_size), desc="queries"):
        batch = collator._process_queries(queries[i : i + batch_size])
        query_mask = collator.make_query_token_mask(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        encoded = model(**clone_tensor_inputs(batch))
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
        encoded = model(**clone_tensor_inputs(batch), output_attentions=pruner is not None)
        if pruner is not None:
            pruned, stats = pruner.prune_doc(
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                patch_mask=doc_mask,
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
    return len(relevant_pages.intersection(ranked_pages[:k])) / len(relevant_pages) if relevant_pages else 0.0


def _first_existing(columns: Iterable[str], candidates: Tuple[str, ...]) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"None of columns {candidates!r} found in parquet schema {sorted(available)!r}")


def _page_id(value: Any) -> int:
    text = str(value)
    try:
        return int(text)
    except ValueError:
        pass
    for sep in (":", "_", "-", "/"):
        tail = text.rsplit(sep, 1)[-1]
        if tail.isdigit():
            return int(tail)
    raise ValueError(f"Cannot parse page id from {value!r}")


def load_pages(path: Path):
    columns = pq.read_schema(path).names
    doc_col = _first_existing(columns, ("file_name", "doc_name"))
    page_col = _first_existing(columns, ("page", "page_id", "passage_id"))
    image_col = _first_existing(columns, ("image", "image_binary", "bytes"))
    df = pd.read_parquet(path, columns=[doc_col, page_col, image_col])
    grouped = defaultdict(list)
    for row in df.itertuples(index=False):
        grouped[str(getattr(row, doc_col))].append(
            (_page_id(getattr(row, page_col)), decode_image(getattr(row, image_col)))
        )
    return dict(grouped)


def _doc_aliases(doc_name: str) -> set[str]:
    normalized = doc_name.replace("\\", "/")
    base = normalized.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    return {doc_name, normalized, base, stem}


def build_doc_alias_map(doc_names: Iterable[str]) -> Dict[str, str]:
    aliases = defaultdict(set)
    for doc_name in doc_names:
        for alias in _doc_aliases(doc_name):
            aliases[alias].add(doc_name)
    return {alias: next(iter(matches)) for alias, matches in aliases.items() if len(matches) == 1}


def resolve_doc_name(doc_name: str, pages_by_doc: Dict[str, Any], aliases: Dict[str, str]) -> str | None:
    if doc_name in pages_by_doc:
        return doc_name
    for alias in _doc_aliases(doc_name):
        if alias in aliases:
            return aliases[alias]
    return None


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
    parser.add_argument("--sample-log", default=None, help="Write per-query ranking samples as JSONL.")
    parser.add_argument("--top-k-log", type=int, default=10, help="Number of ranked pages to store per sample.")
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
    doc_aliases = build_doc_alias_map(pages_by_doc.keys())
    annotations = normalize_annotations(read_jsonl(ann_path))
    if args.max_queries is not None:
        annotations = annotations[: max(0, int(args.max_queries))]
    queries = [annotation_query(row) for row in annotations]
    query_embs = encode_queries(model, collator, queries, args.batch_size, device)

    doc_cache: Dict[str, Tuple[List[int], List[torch.Tensor]]] = {}
    all_keep_ratios: List[float] = []
    metrics = {"r1": [], "r5": [], "r10": [], "ndcg5": []}
    skipped = 0
    skip_reasons = Counter()
    sample_log = Path(args.sample_log) if args.sample_log else None
    if sample_log is not None:
        sample_log.parent.mkdir(parents=True, exist_ok=True)

    with sample_log.open("w", encoding="utf-8") if sample_log is not None else nullcontext() as log_f:
        for rec, q_emb in tqdm(list(zip(annotations, query_embs)), desc="score"):
            query = annotation_query(rec)
            positives = list(rec.get("positive_passages") or [])
            if not positives:
                skipped += 1
                skip_reasons["no_positive_passages"] += 1
                if log_f:
                    log_f.write(json.dumps({"query": query, "skip_reason": "no_positive_passages"}, ensure_ascii=False) + "\n")
                continue

            requested_doc, _ = passage_key(positives[0])
            doc_name = resolve_doc_name(requested_doc, pages_by_doc, doc_aliases)
            if doc_name is None:
                skipped += 1
                skip_reasons["doc_not_found"] += 1
                if log_f:
                    log_f.write(
                        json.dumps(
                            {
                                "query": query,
                                "doc_name": requested_doc,
                                "relevant_pages": [page for _, page in map(passage_key, positives)],
                                "skip_reason": "doc_not_found",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            relevant = {page for doc, page in map(passage_key, positives) if resolve_doc_name(doc, pages_by_doc, doc_aliases) == doc_name}
            if not relevant:
                skipped += 1
                skip_reasons["no_relevant_pages"] += 1
                if log_f:
                    log_f.write(
                        json.dumps(
                            {"query": query, "doc_name": requested_doc, "resolved_doc_name": doc_name, "skip_reason": "no_relevant_pages"},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
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

            if log_f:
                top = [
                    {"rank": rank + 1, "page": page_ids[i], "score": float(page_scores[i]), "relevant": page_ids[i] in relevant}
                    for rank, i in enumerate(ranked_idx[: max(1, args.top_k_log)])
                ]
                log_f.write(
                    json.dumps(
                        {
                            "query": query,
                            "doc_name": requested_doc,
                            "resolved_doc_name": doc_name,
                            "relevant_pages": sorted(relevant),
                            "top_pages": top,
                            "hit@1": bool(recall_at_k(ranked_pages, relevant, 1)),
                            "hit@5": bool(recall_at_k(ranked_pages, relevant, 5)),
                            "hit@10": bool(recall_at_k(ranked_pages, relevant, 10)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    denom = len(metrics["r1"])
    print(f"queries: {denom}  skipped: {skipped}")
    if skip_reasons:
        print(f"skip reasons: {dict(skip_reasons)}")
    print(f"Recall@1: {sum(metrics['r1']) / denom if denom else 0.0:.4f}")
    print(f"Recall@5: {sum(metrics['r5']) / denom if denom else 0.0:.4f}")
    print(f"Recall@10: {sum(metrics['r10']) / denom if denom else 0.0:.4f}")
    print(f"nDCG@5: {sum(metrics['ndcg5']) / denom if denom else 0.0:.4f}")
    if sample_log is not None:
        print(f"sample log: {sample_log}")
    if all_keep_ratios:
        print(f"Pruning mean keep ratio: {sum(all_keep_ratios) / len(all_keep_ratios):.4f}")


if __name__ == "__main__":
    main()
