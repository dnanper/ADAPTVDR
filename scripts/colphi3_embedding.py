from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
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
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        ensure_phi3_img_projection_bias(self.backbone)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self.backbone.config, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer Phi3 hidden size from config")

        self.linear_head = nn.Linear(hidden_size, projection_dim, bias=False)
        nn.init.orthogonal_(self.linear_head.weight)

    def enable_input_require_grads(self):
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def gradient_checkpointing_enable(self):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, **kwargs) -> ColPhi3Output:
        attention_mask = kwargs.get("attention_mask")
        output_attentions = bool(kwargs.pop("output_attentions", False))
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("labels", None)
        kwargs["use_cache"] = False

        outputs = self.backbone(
            **kwargs,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
        if self.linear_head.weight.dtype != last_hidden.dtype:
            self.linear_head.to(dtype=last_hidden.dtype)
        projected = self.linear_head(last_hidden)
        if attention_mask is not None:
            projected = projected * attention_mask.unsqueeze(-1).to(projected.dtype)
        projected = F.normalize(projected.float(), p=2, dim=-1).to(projected.dtype)

        return ColPhi3Output(
            hidden_states=projected,
            attention_mask=attention_mask,
            attentions=outputs.attentions if output_attentions else None,
        )


def ensure_phi3_img_projection_bias(module: nn.Module) -> None:
    """Phi3 remote code reads img_projection.bias even when it is a Sequential."""
    for name, child in module.named_modules():
        if not name.endswith("img_projection"):
            continue
        _attach_projection_bias(child)
        for wrapped_name in ("original_module",):
            wrapped = getattr(child, wrapped_name, None)
            if isinstance(wrapped, nn.Module):
                _attach_projection_bias(wrapped)
        modules_to_save = getattr(child, "modules_to_save", None)
        if isinstance(modules_to_save, nn.ModuleDict):
            for wrapped in modules_to_save.values():
                _attach_projection_bias(wrapped)


def _attach_projection_bias(module: nn.Module) -> None:
    if getattr(module, "bias", None) is not None:
        return
    if isinstance(module, nn.Sequential):
        for child in reversed(module):
            bias = getattr(child, "bias", None)
            if bias is not None:
                module.bias = bias
                return
