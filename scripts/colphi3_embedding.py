from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput


@dataclass
class ColPhi3Output(ModelOutput):
    hidden_states: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None
    attentions: Optional[tuple] = None


class ColPhi3ForEmbedding(nn.Module):
    """Phi3-Vision late-interaction encoder with a Col-Phi3 style projection."""

    def __init__(self, model_name_or_path: str, projection_dim: int = 128, torch_dtype=torch.bfloat16):
        super().__init__()
        from scripts.phi3_compat import patch_phi3_auto_image_processor_register

        patch_phi3_auto_image_processor_register()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self.backbone.config, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer Phi3 hidden size from config")

        self.projection = nn.Linear(hidden_size, projection_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def enable_input_require_grads(self):
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def gradient_checkpointing_enable(self):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, **kwargs) -> ColPhi3Output:
        attention_mask = kwargs.get("attention_mask")
        output_attentions = bool(kwargs.pop("output_attentions", False))
        kwargs.pop("labels", None)

        outputs = self.backbone(
            **kwargs,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
        projected = self.projection(last_hidden)
        if attention_mask is not None:
            projected = projected * attention_mask.unsqueeze(-1).to(projected.dtype)
        projected = F.normalize(projected.float(), p=2, dim=-1).to(projected.dtype)

        return ColPhi3Output(
            hidden_states=projected,
            attention_mask=attention_mask,
            attentions=outputs.attentions if output_attentions else None,
        )
