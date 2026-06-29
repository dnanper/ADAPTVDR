"""Precompute teacher attention maps for AGREE local alignment.

This script runs a Qwen3-VL teacher over training pairs `(query, doc_image)` and
caches one attention vector per sample:

    a_bar[j] = mean_heads(mean_query_tokens(attn[q_i -> image_patch_j]))

Output format:
    output_path/
      metadata.pt
      batch-000000.pt
      batch-000001.pt
      ...

Each batch shard stores one batch payload:

    {
      "sample_ids": List[str],
      "scores": List[fp16 Tensor[num_image_patches]],
      "grids": List[int64 Tensor[3]],  # image_grid_thw per sample, when available
      "num_samples": int,
    }

The script flushes every processed batch to disk so we do not retain a growing
cache in memory, and GPU tensors are released after each batch.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.append(str(root))

import argparse
import glob
import hashlib
import io
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_QUERY_INSTRUCTION = "Represent the user's input."
PROMPT_MODE_QUERY_IMAGE = "query_image"
PROMPT_MODE_IMAGE_ONLY = "image_only"
SOURCE_MODE_QUERY = "query"
SOURCE_MODE_INSTRUCTION = "instruction"
SOURCE_MODE_ALL_NON_IMAGE = "all_non_image"
REQUIRED_COLUMNS = ("query", "image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-model",
        default="/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3-VL-8B-Instruct",
        help="Path to the Qwen3-VL teacher model.",
    )
    parser.add_argument(
        "--train-data-path",
        required=True,
        help="Directory containing parquet shards such as train-00000-of-00082.parquet.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination .pt cache path.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Attention layer to use. -1 means the final attention-producing layer.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist every N processed batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to process per teacher forward pass.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing batch shards if present.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_QUERY_INSTRUCTION,
        help="System instruction used in the teacher prompt.",
    )
    parser.add_argument(
        "--no-system-instruction",
        action="store_true",
        help="Use a paper-style query-token prompt with only user image+query content.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Override teacher image min_pixels to match student preprocessing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override teacher image max_pixels to match student preprocessing.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[PROMPT_MODE_QUERY_IMAGE, PROMPT_MODE_IMAGE_ONLY],
        default=PROMPT_MODE_QUERY_IMAGE,
        help="Whether the teacher prompt includes both query+image or image only.",
    )
    parser.add_argument(
        "--source-mode",
        choices=[SOURCE_MODE_QUERY, SOURCE_MODE_INSTRUCTION, SOURCE_MODE_ALL_NON_IMAGE],
        default=SOURCE_MODE_QUERY,
        help="Which token rows to aggregate when producing per-patch attention scores.",
    )
    return parser.parse_args()


def normalize_instruction_text(instruction: Optional[str]) -> str:
    instruction = (instruction or DEFAULT_QUERY_INSTRUCTION).strip()
    if instruction and instruction[-1] not in ".!?":
        instruction = instruction + "."
    return instruction


def _decode_image(img_data) -> Image.Image:
    if isinstance(img_data, dict):
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img_data)!r}")


def _parquet_files(data_path: str, split: str, num_shards: Optional[int]) -> List[str]:
    pattern = os.path.join(data_path, f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if num_shards is not None:
        files = files[:num_shards]
    if not files:
        raise FileNotFoundError(f"No parquet files found: {pattern}")
    return files


def _validate_required_columns(df: pd.DataFrame, shard_file: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{shard_file} is missing required columns: {missing}")


def stable_sample_id(
    shard_path: str,
    row_idx: int,
    image_filename: Optional[str],
    query: str,
) -> str:
    payload = "||".join(
        [
            os.path.basename(shard_path),
            str(row_idx),
            image_filename or "",
            query.strip(),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def find_subsequence_positions(sequence: Sequence[int], subsequence: Sequence[int]) -> List[int]:
    if not subsequence:
        raise ValueError("subsequence must not be empty")
    if len(subsequence) > len(sequence):
        raise ValueError("subsequence is longer than sequence")

    match: Optional[List[int]] = None
    last_start = len(sequence) - len(subsequence) + 1
    for start in range(last_start):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            match = list(range(start, start + len(subsequence)))
    if match is None:
        raise ValueError("subsequence not found in sequence")
    return match


def get_image_token_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    valid = attention_mask.bool()
    image_mask = (input_ids == image_token_id) & valid
    positions = image_mask.nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        raise ValueError("No image patch tokens found in the encoded input")
    return positions


def get_query_token_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    query_token_ids: Sequence[int],
) -> torch.Tensor:
    valid_ids = input_ids[attention_mask.bool()].tolist()
    positions = find_subsequence_positions(valid_ids, query_token_ids)
    return torch.tensor(positions, dtype=torch.long, device=input_ids.device)


def get_instruction_token_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    instruction_token_ids: Sequence[int],
) -> torch.Tensor:
    valid_ids = input_ids[attention_mask.bool()].tolist()
    positions = find_subsequence_positions(valid_ids, instruction_token_ids)
    return torch.tensor(positions, dtype=torch.long, device=input_ids.device)


def get_all_non_image_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    valid = attention_mask.bool()
    source_mask = (input_ids != image_token_id) & valid
    positions = source_mask.nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        raise ValueError("No non-image source tokens found in the encoded input")
    return positions


def aggregate_source_to_image_attention(
    attn_layer: torch.Tensor,
    source_positions: torch.Tensor,
    image_positions: torch.Tensor,
) -> torch.Tensor:
    if attn_layer.dim() == 4:
        if attn_layer.shape[0] != 1:
            raise ValueError("Expected batch size 1 for cached teacher attention")
        attn_layer = attn_layer[0]
    if attn_layer.dim() != 3:
        raise ValueError(f"Expected [heads, seq, seq], got {tuple(attn_layer.shape)}")

    attn_map = attn_layer[:, source_positions, :]
    attn_map = attn_map[:, :, image_positions]
    return attn_map.mean(dim=0).mean(dim=0)


def aggregate_query_to_image_attention(
    attn_layer: torch.Tensor,
    query_positions: torch.Tensor,
    image_positions: torch.Tensor,
) -> torch.Tensor:
    return aggregate_source_to_image_attention(
        attn_layer=attn_layer,
        source_positions=query_positions,
        image_positions=image_positions,
    )


def select_attention_layer(attentions: Sequence[Optional[torch.Tensor]], layer_index: int) -> torch.Tensor:
    if not attentions:
        raise ValueError("Teacher model did not return attentions")

    if layer_index == -1:
        for attn_layer in reversed(attentions):
            if attn_layer is not None:
                return attn_layer
        raise ValueError("Teacher model returned attentions, but every layer is None")

    attn_layer = attentions[layer_index]
    if attn_layer is None:
        raise ValueError(f"Selected attention layer {layer_index} is None")
    return attn_layer


def _prepare_output_dir(output_path: Path) -> Path:
    if output_path.suffix == ".pt":
        batch_dir = output_path.with_suffix("")
    else:
        batch_dir = output_path
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def _metadata_path(batch_dir: Path) -> Path:
    return batch_dir / "metadata.pt"


def _batch_shard_path(batch_dir: Path, batch_index: int) -> Path:
    return batch_dir / f"batch-{batch_index:06d}.pt"


def _save_metadata(batch_dir: Path, metadata: Dict[str, object]) -> None:
    path = _metadata_path(batch_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(metadata, tmp_path)
    tmp_path.replace(path)


def _load_metadata(batch_dir: Path) -> Dict[str, object]:
    path = _metadata_path(batch_dir)
    if not path.exists():
        return {}
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected dict metadata at {path}, got {type(loaded)!r}")
    return loaded


def _list_batch_shards(batch_dir: Path) -> List[Path]:
    return sorted(batch_dir.glob("batch-*.pt"))


def _load_seen_ids(batch_dir: Path) -> Set[str]:
    seen_ids: Set[str] = set()
    for shard_path in _list_batch_shards(batch_dir):
        loaded = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected dict cache shard at {shard_path}, got {type(loaded)!r}")
        if "sample_ids" in loaded:
            seen_ids.update(str(sample_id) for sample_id in loaded["sample_ids"])
        else:
            seen_ids.update(str(sample_id) for sample_id in loaded.keys())
    return seen_ids


def _save_batch_shard(
    batch_cache: Dict[str, torch.Tensor],
    batch_dir: Path,
    batch_index: int,
    batch_grids: Optional[Dict[str, torch.Tensor]] = None,
) -> Path:
    shard_path = _batch_shard_path(batch_dir, batch_index)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    sample_ids = list(batch_cache.keys())
    payload = {
        "sample_ids": sample_ids,
        "scores": [batch_cache[sample_id] for sample_id in sample_ids],
        "num_samples": len(batch_cache),
    }
    if batch_grids is not None:
        payload["grids"] = [batch_grids.get(sample_id) for sample_id in sample_ids]
    torch.save(payload, tmp_path)
    tmp_path.replace(shard_path)
    return shard_path


def _iter_rows(
    parquet_files: Sequence[str],
    max_samples: Optional[int],
) -> Iterator[Tuple[str, int, pd.Series]]:
    yielded = 0
    for shard_file in parquet_files:
        try:
            df = pd.read_parquet(shard_file, columns=["query", "image", "image_filename"])
        except Exception:
            df = pd.read_parquet(shard_file, columns=["query", "image"])
            df["image_filename"] = None
        _validate_required_columns(df, shard_file)
        for row_idx, row in df.iterrows():
            yield shard_file, row_idx, row
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                return


def _load_teacher(
    model_name_or_path: str,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
):
    # Lazy import so unit tests can import this file without requiring Qwen3-VL.
    from qwen3_vl_embedding import Qwen3VLEmbedder

    kwargs = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels

    return Qwen3VLEmbedder(
        model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        **kwargs,
    )


def _batched_rows(
    rows: Iterator[Tuple[str, int, pd.Series]],
    batch_size: int,
) -> Iterator[List[Tuple[str, int, pd.Series]]]:
    batch: List[Tuple[str, int, pd.Series]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    parquet_files = _parquet_files(args.train_data_path, args.split, args.num_shards)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    output_path = Path(args.output_path)
    batch_dir = _prepare_output_dir(output_path)
    metadata = _load_metadata(batch_dir)
    seen_ids: Set[str] = set()
    next_batch_index = len(_list_batch_shards(batch_dir))
    if args.resume:
        seen_ids = _load_seen_ids(batch_dir)
        print(f"[resume] loaded {len(seen_ids)} cached attention maps from {batch_dir}")
    elif next_batch_index > 0:
        raise FileExistsError(
            f"{batch_dir} already contains batch shards. Use --resume to continue safely."
        )

    teacher = _load_teacher(
        args.teacher_model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor = teacher.processor
    tokenizer = processor.tokenizer
    model_device = next(teacher.model.parameters()).device
    normalized_instruction = "" if args.no_system_instruction else normalize_instruction_text(args.instruction)
    instruction_token_ids = tokenizer.encode(normalized_instruction, add_special_tokens=False)
    if args.no_system_instruction and args.source_mode == SOURCE_MODE_INSTRUCTION:
        raise ValueError("--source-mode instruction requires a system instruction")
    if args.source_mode == SOURCE_MODE_INSTRUCTION and not instruction_token_ids:
        raise ValueError("Instruction tokenization is empty; cannot use --source-mode instruction")
    if args.prompt_mode == PROMPT_MODE_IMAGE_ONLY and args.source_mode == SOURCE_MODE_QUERY:
        raise ValueError("--source-mode query requires --prompt-mode query_image")
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    print(f"Teacher model : {args.teacher_model}")
    print(f"Shards        : {len(parquet_files)}")
    print(f"Output path   : {batch_dir}")
    print(f"Image token   : {image_token} ({image_token_id})")
    print(f"Batch size    : {args.batch_size}")
    print(f"Prompt mode   : {args.prompt_mode}")
    print(f"Source mode   : {args.source_mode}")
    print(f"System prompt : {not args.no_system_instruction}")
    print(f"Instruction   : {normalized_instruction!r}")
    print(f"Min pixels    : {teacher.min_pixels}")
    print(f"Max pixels    : {teacher.max_pixels}")

    processed = 0
    skipped = 0
    saved_batches = next_batch_index
    rows = _iter_rows(parquet_files, args.max_samples)
    progress = tqdm(_batched_rows(rows, args.batch_size), desc="teacher-attn")

    for raw_batch in progress:
        batch_examples = []
        batch_conversations = []

        for shard_file, row_idx, row in raw_batch:
            query = str(row["query"]).strip()
            needs_query_text = args.prompt_mode == PROMPT_MODE_QUERY_IMAGE or args.source_mode == SOURCE_MODE_QUERY
            if needs_query_text and (not query or query.lower() == "none"):
                skipped += 1
                continue

            sample_id = stable_sample_id(
                shard_path=shard_file,
                row_idx=row_idx,
                image_filename=row.get("image_filename"),
                query=query,
            )
            if sample_id in seen_ids:
                continue

            try:
                image = _decode_image(row["image"])
            except Exception as exc:
                skipped += 1
                print(f"[skip] failed to decode image for sample {sample_id}: {exc}")
                continue

            query_token_ids = None
            if args.source_mode == SOURCE_MODE_QUERY:
                query_token_ids = tokenizer.encode(query, add_special_tokens=False)
                if not query_token_ids:
                    skipped += 1
                    print(f"[skip] empty tokenized query for sample {sample_id}")
                    continue

            batch_examples.append((sample_id, query_token_ids))
            batch_conversations.append(
                teacher.format_model_input(
                    text=query if args.prompt_mode == PROMPT_MODE_QUERY_IMAGE else None,
                    image=image,
                    instruction=normalized_instruction,
                    use_system_instruction=not args.no_system_instruction,
                )
            )

        if not batch_examples:
            progress.set_postfix(processed=processed, skipped=skipped, batches=saved_batches)
            continue

        model_inputs = teacher._preprocess_inputs(batch_conversations)
        model_inputs = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }

        with torch.no_grad():
            outputs = teacher.model(
                **model_inputs,
                output_attentions=True,
                attention_layer_index=args.layer_index,
                use_cache=False,
            )

        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            raise ValueError("Teacher model did not return attentions")
        attn_layer = select_attention_layer(attentions, args.layer_index)

        batch_cache: Dict[str, torch.Tensor] = {}
        batch_grids: Dict[str, torch.Tensor] = {}
        for batch_idx, (sample_id, query_token_ids) in enumerate(batch_examples):
            input_ids = model_inputs["input_ids"][batch_idx]
            attention_mask = model_inputs["attention_mask"][batch_idx]
            image_positions = get_image_token_positions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_token_id=image_token_id,
            )
            if args.source_mode == SOURCE_MODE_QUERY:
                assert query_token_ids is not None
                source_positions = get_query_token_positions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    query_token_ids=query_token_ids,
                )
            elif args.source_mode == SOURCE_MODE_INSTRUCTION:
                source_positions = get_instruction_token_positions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    instruction_token_ids=instruction_token_ids,
                )
            else:
                source_positions = get_all_non_image_positions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_token_id=image_token_id,
                )

            attn_map = aggregate_source_to_image_attention(
                attn_layer=attn_layer[batch_idx],
                source_positions=source_positions,
                image_positions=image_positions,
            )
            batch_cache[sample_id] = attn_map.detach().cpu().to(dtype=torch.float16)
            if "image_grid_thw" in model_inputs:
                batch_grids[sample_id] = (
                    model_inputs["image_grid_thw"][batch_idx]
                    .detach()
                    .cpu()
                    .to(dtype=torch.long)
                )
            seen_ids.add(sample_id)
            processed += 1

        shard_path = _save_batch_shard(
            batch_cache,
            batch_dir,
            next_batch_index,
            batch_grids=batch_grids if batch_grids else None,
        )
        next_batch_index += 1
        saved_batches += 1

        metadata.update(
            {
                "teacher_model": args.teacher_model,
                "train_data_path": args.train_data_path,
                "split": args.split,
                "layer_index": args.layer_index,
                "layer_policy": "last_non_none" if args.layer_index == -1 else "explicit",
                "instruction": normalized_instruction,
                "use_system_instruction": not args.no_system_instruction,
                "min_pixels": teacher.min_pixels,
                "max_pixels": teacher.max_pixels,
                "prompt_mode": args.prompt_mode,
                "source_mode": args.source_mode,
                "cache_format": "scores_with_image_grid_thw",
                "batch_size": args.batch_size,
                "num_saved_batches": saved_batches,
                "num_saved_samples": len(seen_ids),
                "last_batch_path": str(shard_path),
            }
        )
        if saved_batches % args.save_every == 0:
            _save_metadata(batch_dir, metadata)

        del batch_cache
        del attn_layer
        del outputs
        del model_inputs
        _release_cuda_memory()

        progress.set_postfix(processed=processed, skipped=skipped, batches=saved_batches)

    metadata.update(
        {
            "teacher_model": args.teacher_model,
            "train_data_path": args.train_data_path,
            "split": args.split,
            "layer_index": args.layer_index,
            "layer_policy": "last_non_none" if args.layer_index == -1 else "explicit",
            "instruction": normalized_instruction,
            "use_system_instruction": not args.no_system_instruction,
            "min_pixels": teacher.min_pixels,
            "max_pixels": teacher.max_pixels,
            "prompt_mode": args.prompt_mode,
            "source_mode": args.source_mode,
            "cache_format": "scores_with_image_grid_thw",
            "batch_size": args.batch_size,
            "num_saved_batches": saved_batches,
            "num_saved_samples": len(seen_ids),
        }
    )
    _save_metadata(batch_dir, metadata)
    print(f"[done] saved {len(seen_ids)} attention maps across {saved_batches} batch shards in {batch_dir}")
    print(f"[stats] processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
