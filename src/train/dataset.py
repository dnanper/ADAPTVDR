"""Dataset loaders for ColPali training.

ViDoReDataset     — original ViDoRe parquet format (query + image)
LlamaIndexDataset — LlamaIndex Multilingual format (query + image + hard negatives)
                    Supports two memory modes:
                    - preload_all=True  : decode all images into RAM at init (~70GB for 5 langs)
                    - preload_all=False : lazy ShardLRUCache — keep max_shards shards in RAM
                      IMPORTANT: requires num_workers=0 (cache is not fork-safe)
TripletDataset    — Pre-built triplet parquets (query + positive_bytes + hard_neg_bytes)
                    Built offline by scripts/build_triplet_dataset.py.
                    Loads all shards via Polars collect() — en k=3 ≈ 45 GB.
                    Fork-safe → any DataLoader num_workers works.
"""
import io
import glob
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import pandas as pd
import polars as pl
from PIL import Image
from torch.utils.data import Dataset

from scripts.precompute_teacher_attn import stable_sample_id


class ViDoReDataset(Dataset):
    """Loads ViDoRe parquet shards and exposes (query, image) pairs."""

    def __init__(self, data_path: str, split: str = "train", num_shards: Optional[int] = None):
        pattern = os.path.join(data_path, f"{split}-*.parquet")
        parquet_files = sorted(glob.glob(pattern))

        if num_shards is not None:
            parquet_files = parquet_files[:num_shards]

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found: {pattern}")

        frames = []
        for shard_file in parquet_files:
            try:
                frame = pd.read_parquet(shard_file, columns=["query", "image", "image_filename"])
            except Exception:
                frame = pd.read_parquet(shard_file, columns=["query", "image"])
                frame["image_filename"] = None
            frame["__shard_path"] = shard_file
            frame["__row_idx"] = range(len(frame))
            frames.append(frame)

        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        query = str(row["query"])

        img_data = row["image"]
        if isinstance(img_data, dict):
            image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        else:
            raise ValueError(f"Unknown image type: {type(img_data)}")

        sample_id = stable_sample_id(
            shard_path=row["__shard_path"],
            row_idx=int(row["__row_idx"]),
            image_filename=row.get("image_filename"),
            query=query,
        )

        return {"query": query, "image": image, "sample_id": sample_id}


def _decode_image(img_data) -> Image.Image:
    if isinstance(img_data, dict):
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    raise ValueError(f"Unknown image type: {type(img_data)}")


def _pdf_hash(doc_id: str) -> str:
    """Extract PDF hash from LlamaIndex ID format: '{lang}_{page}_{hash}'."""
    parts = doc_id.split("_", 2)
    return parts[2] if len(parts) == 3 else doc_id


class ShardLRUCache:
    """LRU cache of decoded image shards for lazy cross-shard hard negative lookup.

    Keeps at most `max_shards` parquet shards decoded in RAM simultaneously.
    When a new shard is needed and the cache is full, the least-recently-used
    shard is evicted.

    Each shard is stored as {img_id: PIL.Image} — O(1) lookup within a shard.

    Args:
        max_shards: Maximum number of shards to keep decoded in RAM.
                    Each shard ≈ 1.5 GB. Recommended: 10–15 for 32 GB machines.

    WARNING: Not fork-safe. Use with DataLoader num_workers=0 only.
    """

    def __init__(self, max_shards: int = 12):
        self.max_shards = max_shards
        self._cache: OrderedDict = OrderedDict()  # shard_path → {img_id: Image}
        self._hits   = 0
        self._misses = 0

    def get_image(self, shard_file: str, img_id: str) -> Optional[Image.Image]:
        """Return decoded image for img_id, loading its shard if not cached."""
        if shard_file not in self._cache:
            self._load(shard_file)
            self._misses += 1
        else:
            self._hits += 1
        self._cache.move_to_end(shard_file)
        return self._cache[shard_file].get(img_id)

    def _load(self, shard_file: str):
        if len(self._cache) >= self.max_shards:
            evicted, _ = self._cache.popitem(last=False)  # remove LRU
        df = pd.read_parquet(shard_file, columns=["id", "image"])
        shard_dict = {}
        n_bad = 0
        for row_id, img_data in zip(df["id"].tolist(), df["image"].tolist()):
            try:
                shard_dict[row_id] = _decode_image(img_data)
            except Exception:
                shard_dict[row_id] = None   # corrupted — caller handles None
                n_bad += 1
        if n_bad:
            print(f"  [WARN] {shard_file}: {n_bad} corrupted images skipped")
        self._cache[shard_file] = shard_dict

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> str:
        return (
            f"ShardLRUCache(loaded={len(self._cache)}/{self.max_shards}, "
            f"hits={self._hits}, misses={self._misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class LlamaIndexDataset(Dataset):
    """LlamaIndex Multilingual Visual Document Retrieval dataset.

    Supports two memory modes controlled by `preload_all`:

    preload_all=True (default, for machines with ≥ 64 GB RAM):
        Decodes every image at init into id_to_image dict.
        __getitem__ is O(1) with zero disk I/O during training.

    preload_all=False (for machines with ~32 GB RAM):
        Builds only a lightweight id → shard_file index at init (~10 MB).
        Images are loaded on-demand via ShardLRUCache (max_shards shards in RAM).
        IMPORTANT: set DataLoader num_workers=0 — cache is not fork-safe.

    Each item returns:
        query           : str
        image           : PIL.Image   (positive doc)
        doc_id          : str         (full ID, e.g. "en_13_fff5...")
        pdf_hash        : str         (PDF-level collision key for sampler)
        hard_neg_images : List[PIL.Image]  (top-K hard negs from voyage-3 index)
    """

    def __init__(
        self,
        data_root:    str,
        languages:    List[str] = ("en",),
        hard_neg_k:   int  = 3,
        preload_all:  bool = False,
        max_shards:   int  = 12,        # only used when preload_all=False
    ):
        self.hard_neg_k  = hard_neg_k
        self.preload_all = preload_all

        print(f"[LlamaIndexDataset] languages={list(languages)}, hard_neg_k={hard_neg_k}, "
              f"preload_all={preload_all}" + (f", max_shards={max_shards}" if not preload_all else ""))

        # ── 1. Scan all parquet files ────────────────────────────────────────
        all_files: List[str] = []
        for lang in languages:
            pattern = os.path.join(data_root, lang, "train-*.parquet")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No parquet files found: {pattern}")
            all_files.extend(files)

        print(f"  Found {len(all_files)} shards across {list(languages)}")

        # ── 2. Build id_to_loc + records (no image bytes read yet) ───────────
        # id_to_loc: every id (incl. null-query rows) → shard_file
        # records:   only valid-query rows
        self.id_to_loc: Dict[str, str] = {}   # img_id → shard_file path
        records_raw: List[dict]        = []

        print(f"  Scanning metadata (no image bytes) ...")
        for shard_file in all_files:
            df = pd.read_parquet(shard_file, columns=["id", "query", "negatives"])
            ids     = df["id"].tolist()
            queries = df["query"].tolist()
            negs    = df["negatives"].tolist()

            for img_id, q, neg in zip(ids, queries, negs):
                self.id_to_loc[img_id] = shard_file   # all rows (incl. null-query)
                q_str = str(q) if q is not None else ""
                if len(q_str.strip()) >= 5 and q_str.lower() != "none":
                    records_raw.append({
                        "id":        img_id,
                        "query":     q_str,
                        "negatives": list(neg) if neg is not None else [],
                        "pdf_hash":  _pdf_hash(img_id),
                    })

        n_total   = len(self.id_to_loc)
        n_valid   = len(records_raw)
        print(f"  Total IDs indexed: {n_total}  |  Valid queries: {n_valid}  "
              f"(filtered {n_total - n_valid} null-query rows)")

        self.records = records_raw

        # ── 3a. Preload mode: decode all images now ───────────────────────────
        if preload_all:
            print(f"  Preloading all {n_total} images into RAM ...")
            self.id_to_image: Dict[str, Image.Image] = {}
            n_bad_total = 0
            for shard_file in all_files:
                df_img = pd.read_parquet(shard_file, columns=["id", "image"])
                n_bad = 0
                for img_id, img_data in zip(df_img["id"].tolist(), df_img["image"].tolist()):
                    try:
                        self.id_to_image[img_id] = _decode_image(img_data)
                    except Exception:
                        self.id_to_image[img_id] = None  # corrupted — _get_image returns None
                        n_bad += 1
                if n_bad:
                    print(f"  [WARN] {shard_file}: {n_bad} corrupted images skipped")
                    n_bad_total += n_bad
            if n_bad_total:
                print(f"  Preload done. Total corrupted: {n_bad_total}")
            else:
                print(f"  Preload done.")
        # ── 3b. Lazy mode: set up ShardLRUCache ───────────────────────────────
        else:
            self.shard_cache = ShardLRUCache(max_shards=max_shards)
            print(f"  Lazy mode: ShardLRUCache(max_shards={max_shards}) ready. "
                  f"num_workers MUST be 0.")

        print(f"  Dataset ready: {len(self.records)} samples.")

    def _get_image(self, img_id: str) -> Optional[Image.Image]:
        if self.preload_all:
            return self.id_to_image.get(img_id)
        shard_file = self.id_to_loc.get(img_id)
        if shard_file is None:
            return None
        return self.shard_cache.get_image(shard_file, img_id)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        pos_image = self._get_image(rec["id"])
        if pos_image is None:
            # Corrupted positive — use 1×1 white placeholder to avoid collator crash.
            # Signal is wrong but sample count is tiny; logged on shard load.
            pos_image = Image.new("RGB", (1, 1), (255, 255, 255))

        hard_neg_images: List[Image.Image] = []
        for neg_id in rec["negatives"][:self.hard_neg_k]:
            img = self._get_image(neg_id)
            if img is not None:
                hard_neg_images.append(img)

        return {
            "query":           rec["query"],
            "image":           pos_image,
            "doc_id":          rec["id"],
            "pdf_hash":        rec["pdf_hash"],
            "hard_neg_images": hard_neg_images,
            "sample_id":       None,
        }


class TripletDataset(Dataset):
    """Loads pre-built triplet parquets produced by scripts/build_triplet_dataset.py.

    Each parquet row: query (str), positive (bytes), hard_negatives (list[bytes]).
    All shards loaded into RAM via Polars collect(). en k=3 ≈ 45 GB — fits on 60 GB machine.
    Fork-safe → any DataLoader num_workers works.
    """

    def __init__(self, data_path: str, hard_neg_k: int = 3):
        pattern = os.path.join(data_path, "*train-*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No triplet parquets found in: {data_path}")

        self.df = pl.scan_parquet(pattern).collect()
        self.hard_neg_k = hard_neg_k
        print(f"[TripletDataset] {len(files)} shards  {len(self.df)} triplets  hard_neg_k={hard_neg_k}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)

        positive = Image.open(io.BytesIO(row["positive"])).convert("RGB")

        hard_neg_images: List[Image.Image] = []
        for b in (row["hard_negatives"] or [])[: self.hard_neg_k]:
            try:
                hard_neg_images.append(Image.open(io.BytesIO(b)).convert("RGB"))
            except Exception:
                pass

        return {
            "query":           row["query"],
            "image":           positive,
            "hard_neg_images": hard_neg_images,
            "sample_id":       row.get("sample_id"),
        }
