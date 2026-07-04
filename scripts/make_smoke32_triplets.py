from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="dataset/mmdocir-triplets-k1-10p")
    parser.add_argument("--dst-dir", default="dataset/mmdocir-triplets-k1-smoke32")
    parser.add_argument("--num-samples", type=int, default=32)
    args = parser.parse_args()

    files = sorted(glob.glob(str(Path(args.src_dir) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"Missing parquet shards in {args.src_dir}")

    df = pd.read_parquet(files[0]).head(args.num_samples)
    Path(args.dst_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.dst_dir) / "mmdocir-train-00000.parquet"
    df.to_parquet(out, index=False)
    print(f"{out} rows={len(df)}")


if __name__ == "__main__":
    main()
