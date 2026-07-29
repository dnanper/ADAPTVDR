"""Smoke-test the Phi3/MMDocIR integration on a GPU machine.

Examples:
  python scripts/smoke_phi3_vast.py --test all
  python scripts/smoke_phi3_vast.py --test cache
  python scripts/smoke_phi3_vast.py --test forward --model models/Phi-3-vision-128k-instruct
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from scripts.colphi3_embedding import ColPhi3ForEmbedding
from train.dataset import TripletDataset
from train.loss import MatryoshkaAugmentedMaxSimLoss
from train.phi3_collator import Phi3MMDocIRCollator
from train.teacher_attention import (
    TeacherAttentionCache,
    attention_alignment_loss,
    extract_prior_patch_scores,
    extract_query_patch_scores_from_similarity,
)


def _load_first_row(triplet_dir: str) -> dict[str, Any]:
    files = sorted(glob.glob(str(Path(triplet_dir) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {triplet_dir}")
    df = pd.read_parquet(files[0])
    if df.empty:
        raise ValueError(f"Empty parquet shard: {files[0]}")
    row = df.iloc[0].to_dict()
    print(f"[data] shard={files[0]} rows={len(df)}")
    print(f"[data] sample_id={row.get('sample_id')}")
    print(f"[data] positive_id={row.get('positive_id')}")
    print(f"[data] positive_bytes={len(row['positive'])}")
    print(f"[data] hard_negatives={len(row.get('hard_negatives') or [])}")
    return row


def _load_first_item(triplet_dir: str) -> dict[str, Any]:
    _load_first_row(triplet_dir)
    dataset = TripletDataset(triplet_dir, hard_neg_k=1)
    item = dataset[0]
    print(f"[dataset] sample_id={item.get('sample_id')}")
    print(f"[dataset] image_size={item['image'].size}")
    print(f"[dataset] hard_neg_images={len(item.get('hard_neg_images') or [])}")
    return item


def inspect_cache(cache_dir: str) -> dict[str, Any]:
    root = Path(cache_dir)
    meta_path = root / "metadata.pt"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    batch_files = sorted(root.glob("batch-*.pt"))
    if not batch_files:
        raise FileNotFoundError(f"No batch shards in {cache_dir}")
    batch = torch.load(batch_files[0], map_location="cpu", weights_only=False)
    score_lens = [int(len(score)) for score in batch["scores"][:3]]
    if not score_lens or min(score_lens) <= 0:
        raise AssertionError(f"Invalid teacher score lengths: {score_lens}")
    info = {
        "path": cache_dir,
        "prompt_mode": meta.get("prompt_mode"),
        "source_mode": meta.get("source_mode"),
        "num_saved_samples": meta.get("num_saved_samples"),
        "num_saved_batches": meta.get("num_saved_batches"),
        "cache_format": meta.get("cache_format"),
        "teacher_model": meta.get("teacher_model"),
        "score_lens_head": score_lens,
        "first_sample_ids": batch["sample_ids"][:3],
        "first_grid": str((batch.get("grids") or [None])[0]),
    }
    print("[cache]", json.dumps(info, indent=2, ensure_ascii=False))
    return info


def build_batch(args: argparse.Namespace):
    row = _load_first_item(args.triplet_dir)
    collator = Phi3MMDocIRCollator(
        args.model,
        image_size=args.image_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    batch = collator([row])
    q_tokens = int(batch["query_token_mask"].sum())
    d_tokens = int(batch["doc_token_mask"].sum())
    print(f"[collator] image_token_id={collator.image_token_id}")
    print(f"[collator] query_input_ids={tuple(batch['query_inputs']['input_ids'].shape)} query_tokens={q_tokens}")
    print(f"[collator] doc_input_ids={tuple(batch['doc_inputs']['input_ids'].shape)} doc_image_tokens={d_tokens}")
    if "doc_image_grid_thw" in batch:
        print(f"[collator] doc_image_grid_thw={batch['doc_image_grid_thw'].tolist()}")
    if q_tokens <= 0:
        raise AssertionError("query_token_mask is empty")
    if d_tokens <= 0:
        raise AssertionError("doc_token_mask is empty; Phi3 image placeholder detection failed")
    return row, collator, batch


def _to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _clone_tensor_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def forward_once(args: argparse.Namespace, *, output_attentions: bool = True):
    row, collator, batch = build_batch(args)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = ColPhi3ForEmbedding(args.model, projection_dim=args.projection_dim, torch_dtype=dtype).to(device).eval()
    q_inputs = _to_device(batch["query_inputs"], device)
    d_inputs = _to_device(batch["doc_inputs"], device)
    with torch.no_grad():
        q_out = model(**_clone_tensor_inputs(q_inputs))
        d_out = model(**_clone_tensor_inputs(d_inputs), output_attentions=output_attentions)
    q = q_out.hidden_states[batch["query_token_mask"].to(device)]
    d = d_out.hidden_states[batch["doc_token_mask"].to(device)]
    print(f"[forward] q_out={tuple(q_out.hidden_states.shape)} d_out={tuple(d_out.hidden_states.shape)}")
    print(f"[forward] q_selected={tuple(q.shape)} d_selected={tuple(d.shape)}")
    print(f"[forward] q_norm={q.float().norm(dim=-1).mean().item():.4f} d_norm={d.float().norm(dim=-1).mean().item():.4f}")
    print(f"[forward] attentions={d_out.attentions is not None}")
    if q.shape[-1] != args.projection_dim or d.shape[-1] != args.projection_dim:
        raise AssertionError("Projection dimension mismatch")
    if q.shape[0] <= 0 or d.shape[0] <= 0:
        raise AssertionError("No selected retrieval tokens")
    return row, collator, batch, q_out, d_out, q_inputs, d_inputs


def _check_loss(args: argparse.Namespace, batch: dict[str, Any], q_out: Any, d_out: Any) -> None:
    device = torch.device(args.device)
    loss_fn = MatryoshkaAugmentedMaxSimLoss(dims=[32, 64, args.projection_dim], temperature=1.0)
    loss = loss_fn(
        q_out.hidden_states,
        d_out.hidden_states,
        batch["query_token_mask"].to(device),
        batch["doc_token_mask"].to(device),
        pos_count=int(batch["pos_count"]),
    )
    print(f"[loss] value={float(loss):.6f}")
    if not torch.isfinite(loss):
        raise AssertionError("Loss is not finite")


def loss_smoke(args: argparse.Namespace) -> None:
    _, _, batch, q_out, d_out, _, _ = forward_once(args, output_attentions=False)
    _check_loss(args, batch, q_out, d_out)


def _check_alignment(
    args: argparse.Namespace,
    row: dict[str, Any],
    collator: Phi3MMDocIRCollator,
    batch: dict[str, Any],
    q_out: Any,
    d_out: Any,
    q_inputs: dict[str, Any],
    d_inputs: dict[str, Any],
) -> None:
    sid = row["sample_id"]
    query_cache = TeacherAttentionCache(args.query_cache)
    doc_mask = batch["doc_token_mask"][:1].to(d_inputs["input_ids"].device)
    student_grids = list(batch["doc_image_grid_thw"][:1]) if "doc_image_grid_thw" in batch else None

    student_query_scores = extract_query_patch_scores_from_similarity(
        q_emb=q_out.hidden_states,
        d_emb=d_out.hidden_states[:1],
        q_input_ids=q_inputs["input_ids"],
        q_attention_mask=q_inputs["attention_mask"],
        d_input_ids=d_inputs["input_ids"][:1],
        d_attention_mask=d_inputs["attention_mask"][:1],
        queries=[row["query"]],
        tokenizer=collator.processor.tokenizer,
        image_token_id=collator.image_token_id,
        d_image_mask=doc_mask,
    )
    teacher_query = query_cache.get_many([sid])
    loss_q, matched_q = attention_alignment_loss(
        student_scores=student_query_scores,
        teacher_scores=teacher_query,
        student_grids=student_grids,
        teacher_grids=query_cache.get_many_grids([sid]),
        loss_type="kl",
    )

    print(
        "[align] query",
        f"loss={float(loss_q):.6f}",
        f"matched={matched_q}",
        f"student_len={len(student_query_scores[0]) if student_query_scores else 0}",
        f"teacher_len={len(teacher_query[0]) if teacher_query[0] is not None else 0}",
    )
    if matched_q <= 0:
        raise AssertionError("Query teacher alignment matched zero samples")

    if not args.include_prior_attn:
        print("[align] prior skipped; pass --include-prior-attn to test attention-based prior alignment")
        return

    prior_cache = TeacherAttentionCache(args.prior_cache)
    prior_instruction_token_ids = collator.processor.tokenizer.encode(collator.doc_instruction, add_special_tokens=False)
    student_prior_scores = extract_prior_patch_scores(
        attentions=d_out.attentions,
        input_ids=d_inputs["input_ids"][:1],
        attention_mask=d_inputs["attention_mask"][:1],
        image_token_id=collator.image_token_id,
        source_mode=prior_cache.source_mode,
        instruction_token_ids=prior_instruction_token_ids,
        image_token_mask=doc_mask,
    )
    teacher_prior = prior_cache.get_many([sid])
    loss_p, matched_p = attention_alignment_loss(
        student_scores=student_prior_scores,
        teacher_scores=teacher_prior,
        student_grids=student_grids,
        teacher_grids=prior_cache.get_many_grids([sid]),
        loss_type="kl",
    )
    print(
        "[align] prior",
        f"loss={float(loss_p):.6f}",
        f"matched={matched_p}",
        f"student_len={len(student_prior_scores[0]) if student_prior_scores else 0}",
        f"teacher_len={len(teacher_prior[0]) if teacher_prior[0] is not None else 0}",
    )
    if matched_p <= 0:
        raise AssertionError("Prior teacher alignment matched zero samples")


def alignment_smoke(args: argparse.Namespace) -> None:
    row, collator, batch, q_out, d_out, q_inputs, d_inputs = forward_once(
        args,
        output_attentions=args.include_prior_attn,
    )
    _check_alignment(args, row, collator, batch, q_out, d_out, q_inputs, d_inputs)


def full_smoke(args: argparse.Namespace) -> None:
    inspect_cache(args.prior_cache)
    inspect_cache(args.query_cache)
    row, collator, batch, q_out, d_out, q_inputs, d_inputs = forward_once(
        args,
        output_attentions=args.include_prior_attn,
    )
    _check_loss(args, batch, q_out, d_out)
    _check_alignment(args, row, collator, batch, q_out, d_out, q_inputs, d_inputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["cache", "collator", "forward", "loss", "align", "all"], default="all")
    parser.add_argument("--triplet-dir", default="dataset/mmdocir-triplets-k1-smoke32")
    parser.add_argument("--model", default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--prior-cache", default="dataset/attn_cache_mmdocir_phi3_prior_smoke32")
    parser.add_argument("--query-cache", default="dataset/attn_cache_mmdocir_phi3_query_smoke32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--include-prior-attn", action="store_true")
    args = parser.parse_args()

    if args.test == "all":
        full_smoke(args)
        print(f"[ok] smoke test passed: {args.test}")
        return
    if args.test == "cache":
        inspect_cache(args.prior_cache)
        inspect_cache(args.query_cache)
    if args.test == "collator":
        build_batch(args)
    if args.test == "forward":
        forward_once(args, output_attentions=args.include_prior_attn)
    if args.test == "loss":
        loss_smoke(args)
    if args.test == "align":
        alignment_smoke(args)
    print(f"[ok] smoke test passed: {args.test}")


if __name__ == "__main__":
    main()
