"""Offline pre-processor: flatten LlamaIndex shards into self-contained triplet parquets.

Each output row:
    query          : Utf8
    positive       : Binary   (raw JPEG/PNG bytes of the positive document image)
    hard_negatives : List(Binary)  (up to --hard_neg_k resolved negative images)

Benefits over original LlamaIndex shards:
    - Zero cross-shard lookups needed at train time
    - Polars (Rust core) writes List(Binary) natively — no PyArrow nested-type limits
    - Fork-safe → any DataLoader num_workers
    - Deterministic: hard negatives resolved ONCE offline, not per-epoch

Memory during build:
    All source image bytes loaded into a single id→bytes dict.
    en+fr  ≈  30 GB   (200k images × ~150 KB average JPEG)
    5 langs ≈  80 GB

Usage:
    python scripts/build_triplet_dataset.py \\
        --data_root  dataset/llamaindex-multilingual/6b92b5cae23d44509f1e05d7062befe5ec77f7c9 \\
        --languages  en fr \\
        --hard_neg_k 10 \\
        --output_dir dataset/triplets-en-fr \\
        --rows_per_shard 10000
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

import polars as pl
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_shards(data_root: str, languages: List[str]) -> List[str]:
    all_files: List[str] = []
    for lang in languages:
        pattern = os.path.join(data_root, lang, "train-*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files found: {pattern}")
        all_files.extend(files)
    print(f"Found {len(all_files)} source shards for languages {languages}")
    return all_files


def _flush_shard(
    output_dir: str,
    idx: int,
    queries: List[str],
    positives: List[bytes],
    hard_negs: List[List[bytes]],
    prefix: str = "",
):
    df = pl.DataFrame({
        "query":          pl.Series(queries,    dtype=pl.Utf8),
        "positive":       pl.Series(positives,  dtype=pl.Binary),
        "hard_negatives": pl.Series(hard_negs,  dtype=pl.List(pl.Binary)),
    })
    path = os.path.join(output_dir, f"{prefix}train-{idx:05d}-of-tmp.parquet")
    df.write_parquet(path, compression="zstd")


# ── Main builder ──────────────────────────────────────────────────────────────

def build(
    data_root:      str,
    languages:      List[str],
    hard_neg_k:     int,
    output_dir:     str,
    rows_per_shard: int,
    lang_prefix:    str = "",
):
    all_files = _discover_shards(data_root, languages)

    # ── Phase 1: Load id → raw_bytes + collect valid-query records ────────
    id_to_bytes: Dict[str, Optional[bytes]] = {}
    records: List[dict] = []

    print("\nPhase 1/2 — Loading image bytes + metadata into RAM ...")
    for shard in tqdm(all_files, desc="Scanning"):
        df = pl.read_parquet(shard, columns=["id", "query", "negatives", "image"])

        ids      = df["id"].to_list()
        queries  = df["query"].to_list()
        negs_col = df["negatives"].to_list()

        # Image column: Struct{bytes: Binary, path: Utf8}  OR  plain Binary
        img_col = df["image"]
        if img_col.dtype == pl.Struct:
            img_bytes_list = img_col.struct.field("bytes").to_list()
        else:
            img_bytes_list = img_col.to_list()

        for img_id, q, neg, raw in zip(ids, queries, negs_col, img_bytes_list):
            id_to_bytes[img_id] = raw if isinstance(raw, bytes) else None

            q_str = str(q) if q is not None else ""
            if len(q_str.strip()) >= 5 and q_str.lower() != "none":
                records.append({
                    "id":        img_id,
                    "query":     q_str,
                    "negatives": list(neg) if neg is not None else [],
                })

    print(
        f"  Total IDs indexed : {len(id_to_bytes)}\n"
        f"  Valid queries     : {len(records)}"
    )

    # ── Phase 2: Build & write triplet parquets ───────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    shard_idx = 0
    buf_q:   List[str]         = []
    buf_pos: List[bytes]       = []
    buf_hn:  List[List[bytes]] = []

    n_written  = 0
    n_skip_pos = 0
    n_skip_neg = 0

    print("\nPhase 2/2 — Building triplets ...")
    for rec in tqdm(records, desc="Triplets"):
        pos_bytes = id_to_bytes.get(rec["id"])
        if not pos_bytes:
            n_skip_pos += 1
            continue

        hn_bytes: List[bytes] = []
        for neg_id in rec["negatives"][:hard_neg_k]:
            nb = id_to_bytes.get(neg_id)
            if not nb:
                n_skip_neg += 1
                continue
            hn_bytes.append(nb)

        buf_q.append(rec["query"])
        buf_pos.append(pos_bytes)
        buf_hn.append(hn_bytes)

        if len(buf_q) >= rows_per_shard:
            _flush_shard(output_dir, shard_idx, buf_q, buf_pos, buf_hn, prefix=lang_prefix)
            n_written  += len(buf_q)
            shard_idx  += 1
            buf_q, buf_pos, buf_hn = [], [], []

    if buf_q:
        _flush_shard(output_dir, shard_idx, buf_q, buf_pos, buf_hn, prefix=lang_prefix)
        n_written += len(buf_q)
        shard_idx += 1

    # ── Rename with correct final count ───────────────────────────────────
    total = shard_idx
    for i in range(total):
        old = os.path.join(output_dir, f"{lang_prefix}train-{i:05d}-of-tmp.parquet")
        new = os.path.join(output_dir, f"{lang_prefix}train-{i:05d}-of-{total:05d}.parquet")
        os.rename(old, new)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Output dir       : {output_dir}")
    print(f"  Shards written   : {total}  ({rows_per_shard} rows/shard)")
    print(f"  Total triplets   : {n_written}")
    if n_skip_pos:
        print(f"  Skipped (bad pos): {n_skip_pos} records")
    if n_skip_neg:
        print(f"  Skipped (bad neg): {n_skip_neg} individual hard negatives")
    print(f"  Hard neg coverage: ~{(n_written * hard_neg_k - n_skip_neg) / max(n_written * hard_neg_k, 1):.1%}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build self-contained triplet parquets from LlamaIndex multilingual shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--languages",      nargs="+", default=["en", "fr"])
    parser.add_argument("--hard_neg_k",     type=int,  default=10)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--rows_per_shard", type=int,  default=10_000)
    parser.add_argument(
        "--lang_prefix", type=str, default="",
        help="Prefix for output filenames, e.g. 'en-' → en-train-00000-of-00006.parquet.",
    )
    args = parser.parse_args()

    build(
        data_root=args.data_root,
        languages=args.languages,
        hard_neg_k=args.hard_neg_k,
        output_dir=args.output_dir,
        rows_per_shard=args.rows_per_shard,
        lang_prefix=args.lang_prefix,
    )
