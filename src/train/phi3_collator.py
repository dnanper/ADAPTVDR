import math
from typing import Any, Dict, List, Optional

from PIL import Image


class Phi3MMDocIRCollator:
    """Build query/doc batches for Phi3-Vision late-interaction training."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        image_size: Optional[int] = 1344,
        min_pixels: int = 4096,
        max_pixels: int = 1048576,
        query_instruction: str = "Represent the user's input.",
        doc_instruction: str = "Represent the user's input.",
        query_template: str = "<|user|>\n{query_instruction}\nquery: {query}<|end|>\n<|assistant|>\n",
        doc_template: str = "<|user|>\n{doc_instruction}\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n",
    ):
        from scripts.phi3_compat import patch_phi3_auto_image_processor_register
        from transformers import AutoProcessor

        patch_phi3_auto_image_processor_register()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        self.max_length = max_length
        self.image_size = image_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self.query_template = query_template
        self.doc_template = doc_template
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.image_token_id = self._resolve_image_token_id(tokenizer)

    def _resolve_image_token_id(self, tokenizer) -> int:
        for owner in (self.processor, tokenizer):
            token_id = getattr(owner, "image_token_id", None)
            if token_id is not None:
                return int(token_id)
        for token in ("<|image_1|>", "<|image|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                return int(token_id)
        return int(tokenizer.convert_tokens_to_ids("<|image_1|>"))

    def _resize_page(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.image_size is not None:
            return image.resize((int(self.image_size), int(self.image_size)))

        width, height = image.size
        pixels = max(1, width * height)
        scale = 1.0
        if pixels > self.max_pixels:
            scale = math.sqrt(self.max_pixels / pixels)
        elif pixels < self.min_pixels:
            scale = math.sqrt(self.min_pixels / pixels)

        if scale == 1.0:
            return image
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(new_size)

    def make_query_token_mask(self, inputs: Dict[str, Any]):
        input_ids = inputs["input_ids"]
        return (input_ids != self.image_token_id) & inputs["attention_mask"].bool()

    def make_doc_token_mask(self, inputs: Dict[str, Any]):
        return (inputs["input_ids"] == self.image_token_id) & inputs["attention_mask"].bool()

    def _process_queries(self, queries: List[str]) -> Dict[str, Any]:
        prompts = [
            self.query_template.format(query=q, query_instruction=self.query_instruction)
            for q in queries
        ]
        return dict(
            self.processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        )

    def _process_docs(self, images: List[Image.Image]) -> Dict[str, Any]:
        prompts = [self.doc_template.format(doc_instruction=self.doc_instruction) for _ in images]
        resized = [self._resize_page(img) for img in images]
        return dict(
            self.processor(
                text=prompts,
                images=resized,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        sample_ids = [item.get("sample_id") for item in batch]
        all_doc_images = [item["image"] for item in batch]
        for item in batch:
            all_doc_images.extend(item.get("hard_neg_images", []))

        query_inputs = self._process_queries(queries)
        doc_inputs = self._process_docs(all_doc_images)

        return {
            "query_inputs": query_inputs,
            "doc_inputs": doc_inputs,
            "query_token_mask": self.make_query_token_mask(query_inputs),
            "doc_token_mask": self.make_doc_token_mask(doc_inputs),
            "pos_count": len(batch),
            "sample_ids": sample_ids,
            "queries": queries,
        }
