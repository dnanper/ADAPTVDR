"""ColPali collator for PaliGemma-3B training.

Asymmetric encoding (same as original ColPali paper):
  - Query  : text tokens only  → query_token_mask = non-image tokens & attention_mask
  - Document: image patch tokens only → doc_token_mask = (input_ids == image_token_id)

PaliGemma 448: SigLIP ViT 448×448 with patch_size=14 → 32×32 = 1024 patches.
After the PaliGemma linear projection these become 256 image tokens in the sequence.

NOTE: PaliGemmaProcessor always requires `images`. For queries we pass a dummy white
image but mask out those image tokens so they don't contribute to the query embedding.
"""
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor

# Dummy white image for query processing (PaliGemmaProcessor requires images)
_DUMMY_IMAGE = Image.new("RGB", (448, 448), color=(255, 255, 255))


class ColPaliCollator:
    """Collates (query, image) pairs for ColPali-3B training.

    Args:
        processor_path: Path to the PaliGemma processor (same dir as weights).
        max_query_len:  Max tokens for query text (default 50, original ColPali).
    """

    def __init__(self, processor_path: str, max_query_len: int = 50):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.max_query_len = max_query_len

        # Image token id: used to build the doc mask
        # PaliGemma uses a special <image> token in the sequence for each patch
        tok = self.processor.tokenizer
        img_tok = getattr(self.processor, "image_token", "<image>")
        self.image_token_id: int = tok.convert_tokens_to_ids(img_tok)

    def __call__(self, batch: List[dict]) -> dict:
        queries = [item["query"] for item in batch]
        images  = [item["image"] for item in batch]         # PIL Images
        sample_ids = [item.get("sample_id") for item in batch]

        # ── Query inputs (text only, dummy image required by processor) ───────
        q_texts = [f"<image>Question: {q}\nAnswer:" for q in queries]
        query_inputs = self.processor(
            text=q_texts,
            images=[_DUMMY_IMAGE] * len(queries),
            return_tensors="pt",
            padding="longest",
        )
        # Mask: only non-image tokens (exclude dummy image patches)
        query_token_mask = (
            (query_inputs["input_ids"] != self.image_token_id)
            & query_inputs["attention_mask"].bool()
        )

        # ── Document inputs (image + minimal text) ────────────────────────
        doc_inputs = self.processor(
            text=["<image>\n"] * len(images),   # minimal suffix — ColPali standard
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

        # Document mask: True only for image-patch tokens (not text / padding)
        doc_token_mask = (doc_inputs["input_ids"] == self.image_token_id)

        return {
            "query_inputs": {
                "input_ids":        query_inputs["input_ids"],
                "attention_mask":   query_inputs["attention_mask"],
                "pixel_values":     query_inputs["pixel_values"],
                "token_type_ids":   query_inputs.get("token_type_ids"),
            },
            "doc_inputs": {
                "input_ids":        doc_inputs["input_ids"],
                "attention_mask":   doc_inputs["attention_mask"],
                "pixel_values":     doc_inputs["pixel_values"],
                "token_type_ids":   doc_inputs.get("token_type_ids"),
            },
            "query_token_mask": query_token_mask,
            "doc_token_mask":   doc_token_mask,
            "sample_ids":       sample_ids,
            "queries":          queries,
        }
