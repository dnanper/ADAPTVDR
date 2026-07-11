from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Set

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.precompute_teacher_attn import _parquet_files, stable_sample_id
from train.teacher_attention import TeacherAttentionCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check teacher-attention cache coverage against a train parquet set."
    )
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--prior-cache", default=None)
    parser.add_argument("--query-cache", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--show-missing", type=int, default=20)
    return parser.parse_args()


def iter_train_sample_ids(
    train_data_path: str,
    split: str,
    num_shards: Optional[int],
) -> Iterable[str]:
    for shard_file in _parquet_files(train_data_path, split, num_shards):
        columns = pq.ParquetFile(shard_file).schema.names
        read_cols = [col for col in ("sample_id", "query", "image_filename") if col in columns]
        if "query" not in read_cols:
            raise ValueError(f"{shard_file} is missing required column: query")
        df = pd.read_parquet(shard_file, columns=read_cols)
        for row_idx, row in df.iterrows():
            sample_id = row.get("sample_id")
            if sample_id is None or str(sample_id).strip() == "":
                sample_id = stable_sample_id(
                    shard_path=shard_file,
                    row_idx=int(row_idx),
                    image_filename=row.get("image_filename"),
                    query=str(row["query"]),
                )
            yield str(sample_id)


def check_cache(label: str, cache_path: str, train_ids: Set[str], show_missing: int) -> None:
    cache = TeacherAttentionCache(cache_path)
    cache_ids = set(cache.vectors)
    missing = sorted(train_ids - cache_ids)
    extra = sorted(cache_ids - train_ids)

    print(f"\n[{label}] {cache_path}")
    print(f"  train ids : {len(train_ids)}")
    print(f"  cache ids : {len(cache_ids)}")
    print(f"  covered   : {len(train_ids & cache_ids)}")
    print(f"  missing   : {len(missing)}")
    print(f"  extra     : {len(extra)}")
    if missing and show_missing > 0:
        print("  missing examples:")
        for sample_id in missing[:show_missing]:
            print(f"    {sample_id}")


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_data_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    train_ids = set(iter_train_sample_ids(args.train_data_path, args.split, args.num_shards))
    print(f"train unique ids: {len(train_ids)}")

    if not args.prior_cache and not args.query_cache:
        raise ValueError("Pass --prior-cache and/or --query-cache")
    if args.prior_cache:
        check_cache("prior", args.prior_cache, train_ids, args.show_missing)
    if args.query_cache:
        check_cache("query", args.query_cache, train_ids, args.show_missing)


if __name__ == "__main__":
    main()
