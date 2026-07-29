"""Build local MMDocIR triplet parquets for Phi3/thesis training.

Expected output columns:
  sample_id: str
  query: str
  positive: bytes
  hard_negatives: list[bytes]
  positive_id: str
  negative_ids: list[str]
"""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _passage_id(passage: Any) -> str:
    if isinstance(passage, dict):
        for key in ("passage_id", "id", "docid", "doc_id", "page_id", "pid"):
            if passage.get(key) is not None:
                return str(passage[key])
    return str(passage)


def _image_bytes_from_passage(passage: Any) -> Optional[bytes]:
    if not isinstance(passage, dict):
        return None
    for key in ("image", "page_image", "screenshot", "bytes"):
        value = passage.get(key)
        if isinstance(value, bytes):
            return value
        if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
            return value["bytes"]
        if isinstance(value, Image.Image):
            buf = io.BytesIO()
            value.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
    return None


def _load_image_bytes_from_dir(image_root: Path, passage: Any) -> Optional[bytes]:
    pid = _passage_id(passage)
    candidates = [
        image_root / pid,
        image_root / f"{pid}.png",
        image_root / f"{pid}.jpg",
        image_root / f"{pid}.jpeg",
    ]
    if isinstance(passage, dict):
        for key in ("image_path", "path", "file", "filename"):
            if passage.get(key):
                candidates.append(image_root / str(passage[key]))
    for path in candidates:
        if path.exists() and path.is_file():
            return path.read_bytes()
    return None


def _resolve_image_bytes(passage: Any, image_root: Optional[Path]) -> bytes:
    data = _image_bytes_from_passage(passage)
    if data is None and image_root is not None:
        data = _load_image_bytes_from_dir(image_root, passage)
    if data is None:
        raise ValueError(
            f"Cannot resolve image bytes for passage {_passage_id(passage)!r}"
        )
    Image.open(io.BytesIO(data)).convert("RGB")
    return data


def _sample_id(query: str, positive_id: str) -> str:
    digest = hashlib.sha1(f"{positive_id}\n{query}".encode("utf-8")).hexdigest()[:16]
    return f"mmdocir:{positive_id}:{digest}"


DEFAULT_DOMAINS = [
    "ArxivQA",
    "DUDE",
    "MP-DocVQA",
    "SciQAG",
    "SlideVQA",
    "TAT-DQA",
    "Wiki-ss",
]


def _image_to_png_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, bytes):
        Image.open(io.BytesIO(value)).convert("RGB")
        return value
    if isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    raise ValueError(f"Unsupported image value: {type(value)!r}")


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_page_index_for_keys(
    dataset_root: Path,
    domain: str,
    wanted_keys: set[Tuple[str, int]],
) -> Dict[Tuple[str, int], bytes]:
    index: Dict[Tuple[str, int], bytes] = {}
    parquet_path = dataset_root / "parquet" / f"{domain}_filter.parquet"

    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(
        batch_size=256,
        columns=["file_name", "page", "image"],
    ):
        df = batch.to_pandas()
        for row in df.itertuples(index=False):
            key = (str(row.file_name), int(row.page))
            if key in wanted_keys:
                index[key] = _image_to_png_bytes(row.image)

        if len(index) >= len(wanted_keys):
            break

    return index


def _official_passage_key(passage: Dict[str, Any]) -> Tuple[str, int]:
    return str(passage["doc_name"]), int(passage["page_id"])


def build_official_local(
    dataset_root: str,
    domains: List[str],
    output_dir: str,
    hard_neg_k: int,
    shard_size: int,
    sample_fraction: float,
    sample_seed: int,
    max_samples: Optional[int],
) -> None:
    if not 0 < sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in (0, 1]")
    root = Path(dataset_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: Dict[str, List[Any]] = {
        "sample_id": [],
        "query": [],
        "positive": [],
        "hard_negatives": [],
        "positive_id": [],
        "negative_ids": [],
    }
    shard_idx = 0
    written = 0
    rng = random.Random(sample_seed)

    def flush() -> None:
        nonlocal shard_idx, written, rows
        if not rows["query"]:
            return
        path = out / f"mmdocir-train-{shard_idx:05d}.parquet"
        pq.write_table(pa.table(rows), path)
        written += len(rows["query"])
        print(f"wrote {path} ({len(rows['query'])} rows)")
        shard_idx += 1
        rows = {k: [] for k in rows}

    print(f"building local MMDocIR triplets from {dataset_root} -> {output_dir}")
    for domain in domains:
        ann_path = root / "annotations_top1_negative" / f"{domain}_train.jsonl"
        domain_rows = list(_load_jsonl(ann_path))

        if sample_fraction < 1:
            keep = max(1, int(round(len(domain_rows) * sample_fraction)))
            domain_rows = rng.sample(domain_rows, keep)

        wanted_keys = set()

        for rec in domain_rows:
            positives = list(rec.get("positive_passages") or [])
            negatives = list(rec.get("negative_passages") or [])[:hard_neg_k]

            if positives:
                wanted_keys.add(_official_passage_key(positives[0]))

            for neg in negatives:
                wanted_keys.add(_official_passage_key(neg))

        print(f"loading {domain}: need {len(wanted_keys)} pages")
        page_index = _load_page_index_for_keys(root, domain, wanted_keys)
        print(f"loaded {domain}: {len(page_index)} pages")

        for rec in domain_rows:
            if max_samples is not None and written + len(rows["query"]) >= max_samples:
                break

            query = str(rec.get("query", "")).strip()
            positives = list(rec.get("positive_passages") or [])
            negatives = list(rec.get("negative_passages") or [])[:hard_neg_k]

            if not query or not positives or not negatives:
                continue

            try:
                pos_key = _official_passage_key(positives[0])
                neg_keys = [_official_passage_key(neg) for neg in negatives]

                positive = page_index[pos_key]
                hard_negatives = [page_index[key] for key in neg_keys]
            except Exception as exc:
                print(f"[skip] {domain}: {exc}")
                continue

            pos_id = f"{pos_key[0]}:{pos_key[1]}"
            neg_ids = [f"{doc}:{page}" for doc, page in neg_keys]

            rows["sample_id"].append(_sample_id(query, pos_id))
            rows["query"].append(query)
            rows["positive"].append(positive)
            rows["hard_negatives"].append(hard_negatives)
            rows["positive_id"].append(pos_id)
            rows["negative_ids"].append(neg_ids)

            if len(rows["query"]) >= shard_size:
                flush()

        del page_index

        if max_samples is not None and written + len(rows["query"]) >= max_samples:
            break

    flush()
    print(f"done: {written} rows -> {output_dir}")


def _iter_rows(
    dataset_name: str, split: str, local_files_only: bool
) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name, split=split, download_mode="reuse_dataset_if_exists"
    )
    if local_files_only:
        # datasets has no reliable global local-only switch for all builders; this flag is
        # kept as CLI documentation for cloud/local workflow symmetry.
        print(
            "[WARN] --local-files-only requested; Hugging Face cache must already satisfy load_dataset()."
        )
    yield from ds


def build(
    dataset_name: str,
    split: str,
    output_dir: str,
    image_root: Optional[str],
    hard_neg_k: int,
    shard_size: int,
    local_files_only: bool,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(image_root) if image_root else None

    rows: Dict[str, List[Any]] = {
        "sample_id": [],
        "query": [],
        "positive": [],
        "hard_negatives": [],
        "positive_id": [],
        "negative_ids": [],
    }
    shard_idx = 0
    written = 0

    def flush() -> None:
        nonlocal shard_idx, written, rows
        if not rows["query"]:
            return
        table = pa.table(rows)
        path = out / f"mmdocir-train-{shard_idx:05d}.parquet"
        pq.write_table(table, path)
        written += len(rows["query"])
        print(f"wrote {path} ({len(rows['query'])} rows)")
        shard_idx += 1
        rows = {k: [] for k in rows}

    for rec in _iter_rows(dataset_name, split, local_files_only):
        query = str(rec.get("query", "")).strip()
        pos = _first(rec.get("positive_passages"))
        negs = list(rec.get("negative_passages") or [])[:hard_neg_k]
        if not query or pos is None or not negs:
            continue
        try:
            positive = _resolve_image_bytes(pos, root)
            hard_negatives = [_resolve_image_bytes(neg, root) for neg in negs]
        except Exception as exc:
            print(f"[skip] {exc}")
            continue

        pos_id = _passage_id(pos)
        neg_ids = [_passage_id(neg) for neg in negs]
        rows["sample_id"].append(_sample_id(query, pos_id))
        rows["query"].append(query)
        rows["positive"].append(positive)
        rows["hard_negatives"].append(hard_negatives)
        rows["positive_id"].append(pos_id)
        rows["negative_ids"].append(neg_ids)

        if len(rows["query"]) >= shard_size:
            flush()

    flush()
    print(f"done: {written} rows -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Local MMDocIR_Train_Dataset root with parquet/ and annotations_top1_negative/.",
    )
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--dataset_name", default="MMDocIR/MMDocIR_Train_Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="dataset/mmdocir-triplets-k1")
    parser.add_argument(
        "--image_root", default=None, help="Optional local passage image directory."
    )
    parser.add_argument("--hard_neg_k", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Stratified fraction of annotations per domain, e.g. 0.1 for 10%.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.dataset_root:
        build_official_local(
            dataset_root=args.dataset_root,
            domains=list(args.domains),
            output_dir=args.output_dir,
            hard_neg_k=args.hard_neg_k,
            shard_size=args.shard_size,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            max_samples=args.max_samples,
        )
    else:
        build(
            dataset_name=args.dataset_name,
            split=args.split,
            output_dir=args.output_dir,
            image_root=args.image_root,
            hard_neg_k=args.hard_neg_k,
            shard_size=args.shard_size,
            local_files_only=args.local_files_only,
        )


if __name__ == "__main__":
    main()
