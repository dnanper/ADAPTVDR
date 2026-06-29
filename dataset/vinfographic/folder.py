import os
import argparse
import logging
from huggingface_hub import HfApi, login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-uploader")

def main():
    p = argparse.ArgumentParser(description="Upload images/ folder and README.md to a HF dataset repo")
    p.add_argument("--repo_id", required=True, help="e.g. VLAI-AIVN/ViInfographicVQA")
    p.add_argument("--hf_token", default=None, help="Optional; or use `huggingface-cli login`")
    p.add_argument("--images_dir", default="images", help="Local images folder to upload")
    p.add_argument("--readme", default="README.md", help="Local README.md to upload")
    p.add_argument("--branch", default=None, help="Optional target revision/branch (e.g., parquet-v1)")
    p.add_argument("--commit_message", default="Upload images folder and README", help="Commit message")
    p.add_argument("--skip_images", action="store_true", help="Skip uploading images folder")
    p.add_argument("--skip_readme", action="store_true", help="Skip uploading README.md")
    args = p.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    api = HfApi()

    # Upload images folder
    if not args.skip_images:
        if not os.path.isdir(args.images_dir):
            raise FileNotFoundError(f"images_dir not found: {args.images_dir}")
        logger.info(f"Uploading folder '{args.images_dir}' -> {args.repo_id}/images (repo_type=dataset)")
        api.upload_folder(
            folder_path=args.images_dir,
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo="images",
            commit_message=args.commit_message,
            revision=args.branch,
            allow_patterns=None,   # or e.g. ["*.jpg","*.png"]
            ignore_patterns=None,
        )
        logger.info("✅ Images uploaded")

    # Upload README.md
    if not args.skip_readme:
        if not os.path.isfile(args.readme):
            raise FileNotFoundError(f"README not found: {args.readme}")
        logger.info(f"Uploading README.md -> {args.repo_id}/README.md (repo_type=dataset)")
        api.upload_file(
            path_or_fileobj=args.readme,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
            revision=args.branch,
        )
        logger.info("✅ README uploaded")

    logger.info("🎉 Done.")

if __name__ == "__main__":
    # Speed tip for large uploads:
    # export HF_HUB_ENABLE_HF_TRANSFER=1
    main()