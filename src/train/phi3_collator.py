import math
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
import torch.nn.functional as F


class Phi3MMDocIRCollator:
    """Build query/doc batches for Phi3-Vision late-interaction training."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        image_size: Optional[int] = None,
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
        spatial_mask, _ = self.make_doc_token_mask_and_grid(inputs)
        return spatial_mask

    def make_doc_token_mask_and_grid(self, inputs: Dict[str, Any]):
        if "image_sizes" in inputs:
            return self._make_phi3_local_spatial_mask(inputs)

        valid = inputs["attention_mask"].bool()
        image_mask = (inputs["input_ids"] == self.image_token_id) & valid
        if bool(image_mask.any()):
            return image_mask, None
        return (inputs["input_ids"] < 0) & valid, None

    def _normalise_image_sizes(self, image_sizes: Any) -> List[List[int]]:
        if isinstance(image_sizes, torch.Tensor):
            values = image_sizes.detach().cpu()
            if values.dim() == 1:
                values = values.unsqueeze(0)
            return [[int(v) for v in row[:2].tolist()] for row in values]

        normalised = []
        for item in image_sizes:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().flatten().tolist()
            normalised.append([int(item[0]), int(item[1])])
        return normalised

    def _make_phi3_local_spatial_mask(self, inputs: Dict[str, Any]):
        input_ids = inputs["input_ids"]
        valid = inputs["attention_mask"].bool()
        image_sizes = self._normalise_image_sizes(inputs["image_sizes"])
        spatial_mask = torch.zeros_like(valid)
        grids = []

        for batch_idx, image_size in enumerate(image_sizes):
            height, width = image_size
            h_crop = max(1, height // 336)
            w_crop = max(1, width // 336)
            local_h = h_crop * 12
            local_w = w_crop * 12
            image_positions = ((input_ids[batch_idx] < 0) & valid[batch_idx]).nonzero(as_tuple=False).squeeze(-1)
            expected_image_tokens = local_h * (local_w + 1) + 1 + 12 * 13
            if image_positions.numel() < expected_image_tokens:
                raise ValueError(
                    "Phi3 image tokens were truncated before the local/global HD layout was complete: "
                    f"got {int(image_positions.numel())}, expected at least {expected_image_tokens}. "
                    "Increase max_length or reduce max_pixels."
                )

            cursor = 0
            for _ in range(local_h):
                row_positions = image_positions[cursor: cursor + local_w]
                spatial_mask[batch_idx, row_positions] = True
                cursor += local_w + 1
            grids.append(torch.tensor([1, local_h, local_w], dtype=torch.long))

        grid_tensor = torch.stack(grids, dim=0)
        mask_counts = spatial_mask.sum(dim=1).cpu()
        grid_counts = (grid_tensor[:, 1] * grid_tensor[:, 2]).cpu()
        if not torch.equal(mask_counts, grid_counts):
            raise ValueError(
                "Phi3 local spatial mask/grid mismatch: "
                f"mask={mask_counts.tolist()}, grid={grid_counts.tolist()}"
            )

        return spatial_mask, grid_tensor

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
        prompt = self.doc_template.format(doc_instruction=self.doc_instruction)
        outputs = []
        for image in images:
            outputs.append(dict(
                self.processor(
                    text=prompt,
                    images=[self._resize_page(image)],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            ))
        return self._merge_processor_outputs(outputs)

    def _merge_processor_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        keys = outputs[0].keys()
        for key in keys:
            values = [out[key] for out in outputs]
            if not all(isinstance(v, torch.Tensor) for v in values):
                merged[key] = values
                continue

            if key in {"input_ids", "attention_mask"}:
                pad_value = 0
                if key == "input_ids":
                    pad_value = int(getattr(self.processor.tokenizer, "pad_token_id", 0) or 0)
                merged[key] = torch.nn.utils.rnn.pad_sequence(
                    [v.squeeze(0) for v in values],
                    batch_first=True,
                    padding_value=pad_value,
                )
                continue

            shapes = [tuple(v.shape[1:]) for v in values]
            if len(set(shapes)) == 1:
                merged[key] = torch.cat(values, dim=0)
                continue

            if values[0].dim() >= 3 and all(v.shape[0] == 1 for v in values):
                max_len = max(v.shape[1] for v in values)
                padded = []
                for v in values:
                    pad_len = max_len - v.shape[1]
                    pad = [0, 0] * (v.dim() - 2) + [0, pad_len]
                    padded.append(F.pad(v, pad))
                merged[key] = torch.cat(padded, dim=0)
                continue

            merged[key] = values
        return merged

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        sample_ids = [item.get("sample_id") for item in batch]
        all_doc_images = [item["image"] for item in batch]
        for item in batch:
            all_doc_images.extend(item.get("hard_neg_images", []))

        query_inputs = self._process_queries(queries)
        doc_inputs = self._process_docs(all_doc_images)
        doc_token_mask, doc_image_grid_thw = self.make_doc_token_mask_and_grid(doc_inputs)

        result = {
            "query_inputs": query_inputs,
            "doc_inputs": doc_inputs,
            "query_token_mask": self.make_query_token_mask(query_inputs),
            "doc_token_mask": doc_token_mask,
            "pos_count": len(batch),
            "sample_ids": sample_ids,
            "queries": queries,
        }
        if doc_image_grid_thw is not None:
            result["doc_image_grid_thw"] = doc_image_grid_thw
        return result
