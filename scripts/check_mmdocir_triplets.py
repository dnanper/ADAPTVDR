from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = ("query", "positive", "sample_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MMDocIR triplet parquet dataset integrity.")
    parser.add_argument("--data-path", default="dataset/mmdocir-triplets-k1-full")
    parser.add_argument("--expected-rows", type=int, default=63169)
    parser.add_argument("--show-files", action="store_true")
    return parser.parse_args()


def shard_num(path: Path) -> int | None:
    match = re.search(r"train-(\d+)\.parquet$", path.name)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    files = sorted(data_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")

    nums = [n for n in (shard_num(path) for path in files) if n is not None]
    print(f"data path: {data_path}")
    print(f"parquet files: {len(files)}")
    if nums:
        missing_nums = [i for i in range(min(nums), max(nums) + 1) if i not in set(nums)]
        print(f"shard min/max: {min(nums)} / {max(nums)}")
        print(f"missing shard nums: {missing_nums if missing_nums else 'none'}")
    else:
        print("shard min/max: unavailable")
        print("missing shard nums: unavailable")

    total_rows = 0
    sample_ids: set[str] = set()
    duplicate_count = 0
    bad_schema = []
    bad_read = []

    for path in files:
        try:
            columns = pq.ParquetFile(path).schema.names
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in columns]
            if missing_cols:
                bad_schema.append((path.name, missing_cols, columns))
                continue

            df = pd.read_parquet(path, columns=["sample_id"])
            row_count = len(df)
            total_rows += row_count
            ids = [str(value) for value in df["sample_id"].tolist()]
            before = len(sample_ids)
            sample_ids.update(ids)
            duplicate_count += row_count - (len(sample_ids) - before)
            if args.show_files:
                print(f"{path.name}: rows={row_count}")
        except Exception as exc:
            bad_read.append((path.name, repr(exc)))

    print(f"total rows: {total_rows}")
    print(f"unique sample_ids: {len(sample_ids)}")
    print(f"duplicate sample_ids: {duplicate_count}")
    print(f"expected rows: {args.expected_rows}")
    print(f"row delta: {total_rows - args.expected_rows}")
    print(f"bad schema files: {len(bad_schema)}")
    for name, missing_cols, columns in bad_schema[:20]:
        print(f"  BAD_SCHEMA {name}: missing={missing_cols} cols={columns}")
    print(f"bad read files: {len(bad_read)}")
    for name, error in bad_read[:20]:
        print(f"  BAD_READ {name}: {error}")


if __name__ == "__main__":
    main()
