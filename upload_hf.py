"""
Upload LoRA adapter to HuggingFace Hub.

Usage:
    python upload_hf.py --token hf_xxxx
    python upload_hf.py --token hf_xxxx --repo leo-vnuuet/ColQwen3.5-2B-Embedding
    python upload_hf.py --token hf_xxxx --private
"""
import argparse
import os
from huggingface_hub import HfApi

ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/colqwen3_5_lora-2b/ColQwen3.5-2B-Embedding")
DEFAULT_REPO = "leo-vnuuet/ColQwen3.5-2B-Embedding"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace write token. Defaults to HF_TOKEN.",
    )
    parser.add_argument("--repo",    default=DEFAULT_REPO, help="HF repo id (user/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Missing HuggingFace token. Pass --token or set HF_TOKEN.")

    api = HfApi(token=args.token)

    print(f"[1/2] Creating repo (if not exists): {args.repo}")
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"[2/2] Uploading adapter from: {ADAPTER_PATH}")
    api.upload_folder(
        folder_path=ADAPTER_PATH,
        repo_id=args.repo,
        repo_type="model",
        commit_message="update README, usage implementation",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
