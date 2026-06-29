import os, json, argparse, logging, csv
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence
from huggingface_hub import login, HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("viinfographicvqa")

# ----------------- helpers -----------------
def read_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, list), f"{path} phải là list các object"
        return data

def ensure_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def exists(p: str) -> bool:
    return bool(p) and os.path.exists(p)

def write_missing(rows: List[Dict[str, Any]], out_path: str):
    if not rows: return
    keys = rows[0].keys()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    logger.warning(f"⚠️  Báo cáo ảnh thiếu: {out_path} ({len(rows)} dòng)")

# ----------------- builders -----------------
def build_single_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ex in items:
        img = ex.get("image_path")
        basename = os.path.basename(img) if img else None
        rows.append({
            "question_id"  : str(ex.get("question_id")),
            "images_paths" : [basename] if basename else [],   # chỉ tên file
            "image_type"   : ex.get("image_type"),
            "answer_source": ex.get("answer_source"),
            "element"      : ensure_list(ex.get("element")),
            "question"     : ex.get("question"),
            "answer"       : ex.get("answer"),
        })
    return pd.DataFrame(rows)

def build_multi_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ex in items:
        basenames = [os.path.basename(p) for p in ex.get("image_paths", [])]
        rows.append({
            "question_id"  : str(ex.get("question_id")),
            "images_paths" : basenames,                        # chỉ tên file
            "image_type"   : ex.get("image_type"),
            "answer_source": ex.get("answer_source"),
            "element"      : ensure_list(ex.get("element")) if "element" in ex else [],
            "question"     : ex.get("question"),
            "answer"       : ex.get("answer"),
        })
    return pd.DataFrame(rows)

def make_unified_dataset(df: pd.DataFrame, base_dir: str, split_name: str, images_dirname: str="images") -> Dataset:
    """images_paths = list[str] (filenames only). preview image read from images/{filename}."""
    df = df.copy()
    # preview path: lấy ảnh đầu tiên
    df["preview_path"] = df["images_paths"].map(lambda lst: os.path.join(base_dir, images_dirname, lst[0]) if (lst and lst[0]) else None)

    # validate: mọi filename phải tồn tại trong images/
    missing = []
    keep = []
    for i, filenames in enumerate(df["images_paths"]):
        # full paths
        fulls = [os.path.join(base_dir, images_dirname, fn) for fn in (filenames or [])]
        ok = bool(fulls) and all(exists(p) for p in fulls)
        if not ok:
            miss = [p for p in fulls if not exists(p)]
            missing.append({
                "split": split_name,
                "row_index": i,
                "question_id": df.loc[i, "question_id"],
                "missing": ";".join(miss) if miss else "(no images)"
            })
        keep.append(ok)
    if any(not k for k in keep):
        logger.warning(f"[{split_name}] Bỏ {sum(1 for k in keep if not k)} mẫu thiếu ảnh")
    df = df[keep].reset_index(drop=True)

    # unified features cho tất cả split
    features = Features({
        "question_id"  : Value("string"),
        "images_paths" : Sequence(Value("string")),  # chỉ tên file
        "image"        : Image(),                    # preview (nhúng bytes vào parquet)
        "image_type"   : Value("string"),
        "answer_source": Value("string"),
        "element"      : Sequence(Value("string")),
        "question"     : Value("string"),
        "answer"       : Value("string"),
    })

    # tạo dataset: rename preview_path -> image rồi cast
    ds = Dataset.from_pandas(
        df[["question_id","images_paths","preview_path","image_type","answer_source","element","question","answer"]],
        preserve_index=False
    )
    ds = ds.rename_column("preview_path", "image").cast(features)
    return ds, missing

# ----------------- uploader -----------------
def upload_images_folder(repo_id: str, images_dir: str, path_in_repo: str="images", repo_type: str="dataset"):
    """Upload toàn bộ thư mục images/ lên repo dataset.
    LƯU Ý: nhiều file nhỏ => thời gian lâu. Đã khuyến nghị bật HF_TRANSFER.
    """
    logger.info(f"Uploading folder '{images_dir}' to '{repo_id}/{path_in_repo}' ...")
    api = HfApi()
    api.upload_folder(
        folder_path=images_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message="Upload raw images folder",
        allow_patterns=None,  # hoặc ví dụ ["*.jpg","*.png"]
        ignore_patterns=None,
    )
    logger.info("✅ Upload images folder done.")

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Push ViInfographicVQA (unified schema, filenames only) to HF Hub")
    ap.add_argument("--repo_id", required=True, help="VD: VLAI-AIVN/ViInfographicVQA")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--base_dir", default=".", help="Thư mục chứa /images và /data")
    ap.add_argument("--images_subdir", default="images", help="Tên thư mục ảnh (mặc định: images)")
    ap.add_argument("--max_shard_size", default="4GB")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--branch", default=None)  # ví dụ: parquet-v1
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--upload_images_folder", action="store_true", help="Sau khi push parquet, upload cả thư mục images/ lên repo")
    args = ap.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    base = os.path.abspath(args.base_dir)
    data_dir = os.path.join(base, "data")
    images_dir = os.path.join(base, args.images_subdir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Không thấy {args.images_subdir}/: {images_dir}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Không thấy data/: {data_dir}")

    logger.info(f"Base: {base}")
    logger.info(f"Images: {len(os.listdir(images_dir))} files")

    # đọc annotations
    def read_if(path):
        return read_json_list(path) if os.path.exists(path) else None

    st = read_if(os.path.join(data_dir, "single_train.json"))
    sv = read_if(os.path.join(data_dir, "single_test.json"))
    mt = read_if(os.path.join(data_dir, "multi_train.json"))
    mv = read_if(os.path.join(data_dir, "multi_test.json"))

    ddict = {}
    missing_all = []

    if st:
        from_df, miss = make_unified_dataset(build_single_df(st), base, "single_train", args.images_subdir)
        ddict["single_train"] = from_df; missing_all += miss
        logger.info(f"single_train: {len(from_df)} samples")

    if sv:
        from_df, miss = make_unified_dataset(build_single_df(sv), base, "single_test", args.images_subdir)
        ddict["single_test"] = from_df; missing_all += miss
        logger.info(f"single_test : {len(from_df)} samples")

    if mt:
        from_df, miss = make_unified_dataset(build_multi_df(mt), base, "multi_train", args.images_subdir)
        ddict["multi_train"] = from_df; missing_all += miss
        logger.info(f"multi_train : {len(from_df)} samples")

    if mv:
        from_df, miss = make_unified_dataset(build_multi_df(mv), base, "multi_test", args.images_subdir)
        ddict["multi_test"] = from_df; missing_all += miss
        logger.info(f"multi_test  : {len(from_df)} samples")

    if not ddict:
        raise SystemExit("Không có split nào để push (thiếu file json?).")

    # báo cáo ảnh thiếu
    write_missing(missing_all, os.path.join(base, "missing_images_report.csv"))

    if args.dry_run:
        logger.info("Dry run OK, không push parquet / images."); return

    # push 4 split cùng lúc (đảm bảo schema đồng nhất)
    push_kwargs = dict(
        max_shard_size=args.max_shard_size,
        private=args.private,
        commit_message="Initial upload: unified schema (images_paths=filenames, preview image) → Parquet shards",
    )
    if args.branch:
        push_kwargs["revision"] = args.branch

    ds_all = DatasetDict(ddict)
    logger.info(f"Pushing parquet shards to {args.repo_id} ...")
    ds_all.push_to_hub(args.repo_id, **push_kwargs)
    logger.info("✅ Parquet splits uploaded.")

    # (khuyến nghị) upload thư mục images/ để người dùng tải đủ ảnh
    if args.upload_images_folder:
        upload_images_folder(args.repo_id, images_dir, path_in_repo=args.images_subdir, repo_type="dataset")

    logger.info("🎉 Hoàn tất.")

if __name__ == "__main__":
    main()
