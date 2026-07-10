from types import SimpleNamespace
from typing import Any, Callable

import torch


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def clone_tensor_inputs(batch: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def slice_batch(batch: dict, start: int, end: int) -> dict:
    sliced = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] >= end:
            sliced[key] = value[start:end]
        elif isinstance(value, list):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def forward_model_in_chunks(
    model: Callable[..., Any],
    batch: dict,
    device: torch.device,
    *,
    microbatch_size: int,
    output_attentions: bool = False,
) -> SimpleNamespace:
    batch_size = int(batch["input_ids"].shape[0])
    if microbatch_size <= 0 or microbatch_size >= batch_size:
        inputs = move_to_device(batch, device)
        out = model(**clone_tensor_inputs(inputs), output_attentions=output_attentions)
        return SimpleNamespace(
            hidden_states=out.hidden_states,
            attention_mask=out.attention_mask,
            attentions=out.attentions if output_attentions else None,
        )

    hidden_states = []
    attention_masks = []
    attentions = [] if output_attentions else None
    for start in range(0, batch_size, microbatch_size):
        end = min(batch_size, start + microbatch_size)
        inputs = move_to_device(slice_batch(batch, start, end), device)
        out = model(**clone_tensor_inputs(inputs), output_attentions=output_attentions)
        hidden_states.append(out.hidden_states)
        attention_masks.append(out.attention_mask)
        if output_attentions:
            attentions.append(out.attentions)

    return SimpleNamespace(
        hidden_states=torch.cat(hidden_states, dim=0),
        attention_mask=torch.cat(attention_masks, dim=0),
        attentions=tuple(attentions) if output_attentions else None,
    )
