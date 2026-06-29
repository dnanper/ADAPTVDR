from typing import Any, Dict, List

from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils.vision_process import process_vision_info

QUERY_INSTRUCTION = "Represent the following query for searching relevant document pages."
DOC_INSTRUCTION = "Represent this document image to support information retrieval based on its text and layout."

DEFAULT_INSTRUCTION = "Represent the user's input."


def _load_multimodal_processor(model_path: str):
    qwen3_vl_processor_cls = None
    try:
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        qwen3_vl_processor_cls = Qwen3VLProcessor
    except Exception:
        qwen3_vl_processor_cls = None

    if qwen3_vl_processor_cls is not None:
        try:
            return qwen3_vl_processor_cls.from_pretrained(model_path, padding_side="right")
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="right",
        trust_remote_code=True,
    )
    if not hasattr(processor, "image_processor") or not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            "Failed to load a multimodal processor for "
            f"{model_path}. Current transformers build returned {type(processor).__name__} "
            "instead of a vision-language processor. Upgrade transformers to a version that "
            "supports Qwen3/Qwen3.5-VL or use a compatible environment."
        )
    return processor


class ColPaliCollator:
    """Collates (query, image) pairs into model-ready query_inputs and doc_inputs."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        query_instruction: str = DEFAULT_INSTRUCTION,
        doc_instruction: str = DEFAULT_INSTRUCTION,
        use_query_system_instruction: bool = True,
        use_doc_system_instruction: bool = True,
    ):
        self.processor = _load_multimodal_processor(model_path)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # Bound image size so token counts stay within max_length
        self.processor.image_processor.min_pixels = min_pixels
        self.processor.image_processor.max_pixels = max_pixels
        self.max_length = max_length
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self.use_query_system_instruction = use_query_system_instruction
        self.use_doc_system_instruction = use_doc_system_instruction
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)

    def _make_query_conv(self, query: str) -> List[Dict]:
        conversation = []
        if self.use_query_system_instruction:
            conversation.append({
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        # "text": QUERY_INSTRUCTION
                        "text": self.query_instruction
                    }
                ]
            })
        conversation.append(
            {
                "role": "user",   
                "content": [
                    {
                        "type": "text", 
                        "text": query
                    }
                ]
            },
        )
        return conversation

    def _make_doc_conv(self, image: Image.Image) -> List[Dict]:
        conversation = []
        if self.use_doc_system_instruction:
            conversation.append({
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        # "text": DOC_INSTRUCTION
                        "text": self.doc_instruction
                    }
                ]
            })
        conversation.append(
            {
                "role": "user",   
                "content": [
                    {
                        "type": "image", 
                        "image": image,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ]
            },
        )
        return conversation

    def _preprocess_vision_convs(self, conversations: List[List[Dict]], *, include_images: bool) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        processor_kwargs: Dict[str, Any] = {
            "text": text,
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }

        if include_images:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            if video_inputs is not None:
                videos, video_metadata = zip(*video_inputs)
                processor_kwargs["videos"] = list(videos)
                processor_kwargs["video_metadata"] = list(video_metadata)
            else:
                processor_kwargs["videos"] = None
                processor_kwargs["video_metadata"] = None
            processor_kwargs["images"] = images
            processor_kwargs["do_resize"] = False
            processor_kwargs.update(video_kwargs)

        return dict(self.processor(**processor_kwargs))

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries    = [item["query"] for item in batch]
        pos_images = [item["image"] for item in batch]
        sample_ids = [item.get("sample_id") for item in batch]

        # ── Augmented doc list: [B positives | hard negatives per sample] ─────
        # Layout: pos_0, pos_1, ..., pos_{B-1}, neg_0_0, neg_0_1, ..., neg_{B-1}_{K-1}
        # pos_count = B lets the loss know where positives end.
        all_doc_images = list(pos_images)
        for item in batch:
            all_doc_images.extend(item.get("hard_neg_images", []))
        pos_count = len(batch)

        # ── Query inputs (text only) ──────────────────────────────────────────
        q_convs  = [self._make_query_conv(q) for q in queries]
        query_inputs = self._preprocess_vision_convs(q_convs, include_images=False)

        # ── Document inputs (positives + hard negatives) ──────────────────────
        d_convs  = [self._make_doc_conv(img) for img in all_doc_images]
        doc_inputs = self._preprocess_vision_convs(d_convs, include_images=True)

        return {
            "query_inputs": query_inputs,
            "doc_inputs":   doc_inputs,
            "pos_count":    pos_count,   # B — positives are doc_inputs[0:pos_count]
            "sample_ids":   sample_ids,
            "queries":      queries,
        }
