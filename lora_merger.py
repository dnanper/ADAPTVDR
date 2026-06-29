"""
lora_merger.py — Merge LoRA adapter vào base model và save ra disk.

Usage:
    python lora_merger.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from peft import PeftModel

from scripts.colqwen3_5_embedding import ColQwen3_5ForEmbedding
from transformers import AutoProcessor

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH  = "/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B"
ADAPTER_PATH     = "/data2/cmdir/home/test01/longvnu/graduation_thesis/checkpoints/colqwen3_5_lora/final"
OUTPUT_PATH      = "model/ColQwen3.5-0.8B-Embedding"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Load in fp32 so the merge W = W_base + B@A*scale is precise.
    # bfloat16 merge accumulates rounding errors (esp. at r=32) → worse than live LoRA.
    # Cast back to bfloat16 only after merge before saving.
    print(f"[1/4] Loading base model from: {BASE_MODEL_PATH} (fp32 for precise merge)")
    base_model = ColQwen3_5ForEmbedding.from_pretrained(
        BASE_MODEL_PATH,
    )
    base_model.eval()

    print(f"[2/4] Loading LoRA adapter from: {ADAPTER_PATH}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    print("[3/4] Merging adapter weights into base model (fp32) ...")
    merged_model = peft_model.merge_and_unload()
    # print("      Merge done! Casting to bfloat16 ...")
    # merged_model = merged_model.to(torch.bfloat16)

    print(f"[4/4] Saving merged model to: {OUTPUT_PATH}")
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH)

    # Copy processor/tokenizer from base model
    print("      Copying processor/tokenizer ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    processor.save_pretrained(OUTPUT_PATH)

    print(f"\nDone! Merged model saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
