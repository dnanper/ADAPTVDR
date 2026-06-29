"""Sanity-check whether teacher attention supervision is affected by causal masking.

This script runs the Qwen3-VL teacher on one or more samples and reports:
  - relative token ordering of instruction/query/image patch spans
  - aggregate attention mass from instruction -> image patches
  - aggregate attention mass from query -> image patches

It is useful for validating whether a proposed teacher-attention source span
can actually attend to image patches under a causal attention regime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.precompute_teacher_attn import (  # noqa: E402
    _decode_image,
    _parquet_files,
    _validate_required_columns,
    aggregate_source_to_image_attention,
    get_image_token_positions,
    get_instruction_token_positions,
    get_query_token_positions,
    normalize_instruction_text,
    select_attention_layer,
)
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402


DEFAULT_TEACHER_MODEL = "/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TRAIN_DATA = (
    "/data2/cmdir/home/test01/longvnu/graduation_thesis/"
    "dataset/vidore_train/datasets--vidore--colpali_train_set/"
    "snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--train-data-path", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--instruction", default="Represent the user's input.")
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    return parser.parse_args()


def _find_sample_row(data_path: str, split: str, num_shards: int, sample_index: int) -> Tuple[str, int, pd.Series]:
    shard_files = _parquet_files(data_path, split, num_shards)
    remaining = sample_index
    for shard_file in shard_files:
        df = pd.read_parquet(shard_file)
        _validate_required_columns(df, shard_file)
        if remaining < len(df):
            return shard_file, remaining, df.iloc[remaining]
        remaining -= len(df)
    raise IndexError(f"sample_index={sample_index} exceeds available rows in the selected shards")


def _span_summary(name: str, positions: Sequence[int]) -> str:
    if not positions:
        return f"{name}: []"
    return f"{name}: [{positions[0]}..{positions[-1]}] (n={len(positions)})"


def _relative_order(left_name: str, left: Sequence[int], right_name: str, right: Sequence[int]) -> str:
    if not left or not right:
        return f"{left_name} vs {right_name}: unavailable"
    if max(left) < min(right):
        return f"{left_name} is entirely BEFORE {right_name}"
    if max(right) < min(left):
        return f"{left_name} is entirely AFTER {right_name}"
    return f"{left_name} and {right_name} overlap/interleave"


def _stats(vec: torch.Tensor) -> str:
    vec = vec.float()
    return (
        f"sum={float(vec.sum()):.8f} "
        f"max={float(vec.max()):.8f} "
        f"min={float(vec.min()):.8f} "
        f"mean={float(vec.mean()):.8f}"
    )


def _decode_tokens(tokenizer, input_ids: torch.Tensor, positions: Sequence[int], limit: int = 12) -> List[str]:
    subset = [int(input_ids[pos]) for pos in list(positions)[:limit]]
    tokens = tokenizer.convert_ids_to_tokens(subset)
    if len(positions) > limit:
        tokens.append("...")
    return tokens


def main() -> None:
    args = parse_args()
    normalized_instruction = normalize_instruction_text(args.instruction)

    teacher = Qwen3VLEmbedder(
        model_name_or_path=args.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        default_instruction=normalized_instruction,
    )

    shard_file, row_idx, row = _find_sample_row(
        data_path=args.train_data_path,
        split=args.split,
        num_shards=args.num_shards,
        sample_index=args.sample_index,
    )

    query = str(row["query"]).strip()
    image = _decode_image(row["image"])
    conversation = teacher.format_model_input(
        text=query,
        image=image,
        instruction=normalized_instruction,
    )
    model_inputs = teacher._preprocess_inputs([conversation])
    model_inputs = {
        key: value.to(teacher.model.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_inputs.items()
    }

    with torch.no_grad():
        outputs = teacher.model(
            **model_inputs,
            output_attentions=True,
            attention_layer_index=args.layer_index,
            use_cache=False,
        )

    attn_layer = select_attention_layer(outputs.attentions, args.layer_index)[0]
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]
    tokenizer = teacher.processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    instruction_token_ids = tokenizer.encode(normalized_instruction, add_special_tokens=False)
    query_token_ids = tokenizer.encode(query, add_special_tokens=False)

    image_positions = get_image_token_positions(input_ids, attention_mask, image_token_id).tolist()
    instruction_positions = get_instruction_token_positions(
        input_ids, attention_mask, instruction_token_ids
    ).tolist()
    query_positions = get_query_token_positions(input_ids, attention_mask, query_token_ids).tolist()

    prior_scores = aggregate_source_to_image_attention(
        attn_layer=attn_layer,
        source_positions=torch.tensor(instruction_positions, device=attn_layer.device, dtype=torch.long),
        image_positions=torch.tensor(image_positions, device=attn_layer.device, dtype=torch.long),
    )
    query_scores = aggregate_source_to_image_attention(
        attn_layer=attn_layer,
        source_positions=torch.tensor(query_positions, device=attn_layer.device, dtype=torch.long),
        image_positions=torch.tensor(image_positions, device=attn_layer.device, dtype=torch.long),
    )

    valid_ids = input_ids[attention_mask.bool()]
    print(f"Shard file      : {os.path.basename(shard_file)}")
    print(f"Row index       : {row_idx}")
    print(f"Layer index     : {args.layer_index}")
    print(f"Valid seq len   : {valid_ids.numel()}")
    print(_span_summary("instruction", instruction_positions))
    print(_span_summary("image", image_positions))
    print(_span_summary("query", query_positions))
    print(_relative_order("instruction", instruction_positions, "image", image_positions))
    print(_relative_order("query", query_positions, "image", image_positions))
    print()
    print("Instruction tokens:", _decode_tokens(tokenizer, input_ids, instruction_positions))
    print("Query tokens      :", _decode_tokens(tokenizer, input_ids, query_positions))
    print("Image tokens      :", _decode_tokens(tokenizer, input_ids, image_positions))
    print()
    print("instruction -> image :", _stats(prior_scores))
    print("query -> image       :", _stats(query_scores))

    prior_nonzero = int((prior_scores > 0).sum().item())
    query_nonzero = int((query_scores > 0).sum().item())
    print()
    print(f"prior nonzero patches : {prior_nonzero}/{prior_scores.numel()}")
    print(f"query nonzero patches : {query_nonzero}/{query_scores.numel()}")


if __name__ == "__main__":
    main()
