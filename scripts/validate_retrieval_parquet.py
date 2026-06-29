"""Validate a ColPali-style parquet dataset before fine-tuning.

Checks:
  - required columns exist
  - query text is non-empty and long enough
  - sample images can be decoded
  - reports shard/row counts and basic query length stats
"""

from __future__ import annotations

import argparse
import glob
import io
import os
from statistics import mean

import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = ("query", "image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=32,
        help="How many images to decode per shard.",
    )
    return parser.parse_args()


def decode_image(img_data) -> Image.Image:
    if isinstance(img_data, dict):
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    raise TypeError(f"Unsupported image payload: {type(img_data)!r}")


def parquet_files(data_path: str, split: str, num_shards: int | None) -> list[str]:
    pattern = os.path.join(data_path, f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if num_shards is not None:
        files = files[:num_shards]
    if not files:
        raise FileNotFoundError(f"No parquet files found: {pattern}")
    return files


def main() -> None:
    args = parse_args()
    files = parquet_files(args.data_path, args.split, args.num_shards)

    total_rows = 0
    empty_queries = 0
    none_queries = 0
    short_queries = 0
    query_lengths: list[int] = []
    decoded_images = 0
    decode_errors = 0

    print(f"Validating {len(files)} shard(s) from: {args.data_path}")

    for shard_file in files:
        df = pd.read_parquet(shard_file)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{shard_file} is missing required columns: {missing}")

        total_rows += len(df)

        for q in df["query"].tolist():
            text = "" if q is None else str(q).strip()
            if not text:
                empty_queries += 1
                continue
            if text.lower() == "none":
                none_queries += 1
                continue
            if len(text) < 5:
                short_queries += 1
                continue
            query_lengths.append(len(text))

        for img_data in df["image"].tolist()[: args.max_image_checks]:
            try:
                decode_image(img_data)
                decoded_images += 1
            except Exception:
                decode_errors += 1

    valid_queries = len(query_lengths)
    print(f"Shards          : {len(files)}")
    print(f"Rows            : {total_rows}")
    print(f"Valid queries   : {valid_queries}")
    print(f"Empty queries   : {empty_queries}")
    print(f"'none' queries  : {none_queries}")
    print(f"Short queries   : {short_queries}")
    if query_lengths:
        print(
            "Query length    : "
            f"min={min(query_lengths)} mean={mean(query_lengths):.1f} max={max(query_lengths)}"
        )
    print(f"Decoded images  : {decoded_images}")
    print(f"Decode errors   : {decode_errors}")

    if empty_queries or none_queries or short_queries or decode_errors:
        raise SystemExit(1)

    print("Validation OK")


if __name__ == "__main__":
    main()
