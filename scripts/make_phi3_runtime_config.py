from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--model", default="models/Phi-3-vision-128k-instruct")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--prior-cache", required=True)
    parser.add_argument("--query-cache", required=True)
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["name_or_path"] = args.model
    if args.dtype:
        cfg["model"]["torch_dtype"] = args.dtype
        cfg["training"]["bf16"] = args.dtype == "bfloat16"
        cfg["training"]["fp16"] = args.dtype == "float16"
    cfg["training"]["output_dir"] = args.output_dir
    cfg["data"]["train_data_path"] = args.train_data_path
    cfg["data"]["attn_cache_path_prior"] = args.prior_cache
    cfg["data"]["attn_cache_path_query"] = args.query_cache

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(args.dst)


if __name__ == "__main__":
    main()
