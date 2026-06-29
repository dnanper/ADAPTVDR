from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor

from scripts.adaptive_pruning import AdaptivePruner, IMAGE_TOKEN_ID_COLPALI, PruningStats
from scripts.colpali_paligemma_embedding import ColPaliForEmbedding


_DUMMY_IMAGE = Image.new("RGB", (448, 448), color=(255, 255, 255))


@dataclass
class ColPaliMRLEmbedderOutput:
    embeddings: torch.FloatTensor
    retrieval_mask: torch.Tensor
    attention_mask: torch.Tensor
    attentions: Optional[tuple] = None
    input_ids: Optional[torch.Tensor] = None


class ColPaliMRLEmbedder:
    """Embedder for the PaliGemma + LoRA Matryoshka ColPali training path."""

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        lora_checkpoint: Optional[str] = None,
        embed_dim: Optional[int] = 128,
        max_query_len: int = 50,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.max_query_len = max_query_len

        self.lora_checkpoint = str(lora_checkpoint) if lora_checkpoint is not None else None
        self.base_model_name_or_path = self._resolve_base_model_path(
            base_model_name_or_path=base_model_name_or_path,
            lora_checkpoint=self.lora_checkpoint,
        )

        model = ColPaliForEmbedding.from_pretrained(
            self.base_model_name_or_path,
            torch_dtype=torch_dtype,
            **kwargs,
        ).to(self.device)
        if self.lora_checkpoint is not None:
            model = PeftModel.from_pretrained(model, self.lora_checkpoint, is_trainable=False)
        self.model = model.eval()

        self.processor = AutoProcessor.from_pretrained(self.base_model_name_or_path)
        tok = self.processor.tokenizer
        img_tok = getattr(self.processor, "image_token", "<image>")
        self.image_token_id = tok.convert_tokens_to_ids(img_tok)

    @staticmethod
    def _resolve_base_model_path(
        base_model_name_or_path: Optional[str],
        lora_checkpoint: Optional[str],
    ) -> str:
        if base_model_name_or_path:
            return str(base_model_name_or_path)

        if not lora_checkpoint:
            raise ValueError("Either base_model_name_or_path or lora_checkpoint must be provided.")

        adapter_config_path = Path(lora_checkpoint) / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"Could not infer base model path because {adapter_config_path} does not exist."
            )

        with adapter_config_path.open("r", encoding="utf-8") as fh:
            adapter_config = json.load(fh)

        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"{adapter_config_path} is missing base_model_name_or_path.")
        return str(base_model)

    def process_images(self, images: List[Union[str, Image.Image]]) -> Dict[str, torch.Tensor]:
        pil_images = [
            Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
            for img in images
        ]
        batch = self.processor(
            text=["<image>\n"] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        batch["_retrieval_mask"] = batch["input_ids"] == self.image_token_id
        return batch

    def process_queries(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        batch = self.processor(
            text=[f"<image>Question: {query}\nAnswer:" for query in queries],
            images=[_DUMMY_IMAGE] * len(queries),
            return_tensors="pt",
            padding="longest",
        )
        batch["_retrieval_mask"] = (
            (batch["input_ids"] != self.image_token_id)
            & batch["attention_mask"].bool()
        )
        return batch

    @torch.no_grad()
    def _forward(
        self,
        inputs: Dict[str, torch.Tensor],
        output_attentions: bool = False,
    ) -> ColPaliMRLEmbedderOutput:
        inputs = dict(inputs)
        retrieval_mask = inputs.pop("_retrieval_mask")
        retrieval_mask = retrieval_mask.to(self.device)

        tensor_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        pixel_values = tensor_inputs.get("pixel_values")
        if pixel_values is not None:
            tensor_inputs["pixel_values"] = pixel_values.to(dtype=self.model.dtype)

        outputs = self.model(
            **tensor_inputs,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

        embeddings = outputs.last_hidden_state.float()
        if self.embed_dim is not None and embeddings.shape[-1] > self.embed_dim:
            embeddings = embeddings[..., : self.embed_dim]
        embeddings = F.normalize(embeddings, p=2, dim=-1).to(dtype=outputs.last_hidden_state.dtype)

        return ColPaliMRLEmbedderOutput(
            embeddings=embeddings,
            retrieval_mask=retrieval_mask,
            attention_mask=tensor_inputs["attention_mask"],
            attentions=outputs.attentions if output_attentions else None,
            input_ids=tensor_inputs.get("input_ids"),
        )

    def embed_images_pruned(
        self,
        images: List[Union[str, Image.Image]],
        pruner: Optional[AdaptivePruner] = None,
        r_min: float = 0.3,
        r_max: float = 0.99,
    ) -> Tuple[List[torch.Tensor], PruningStats]:
        if pruner is None:
            pruner = AdaptivePruner(
                r_min=r_min,
                r_max=r_max,
                image_token_id=IMAGE_TOKEN_ID_COLPALI,
                keep_text_tokens=False,
            )

        batch = self.process_images(images)
        out = self._forward(batch, output_attentions=True)
        return pruner.prune_doc(
            hidden_states=out.embeddings,
            attentions=out.attentions,
            input_ids=out.input_ids,
            attention_mask=out.attention_mask,
        )
