from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from scripts.precompute_teacher_attn import (
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODE_QUERY_IMAGE,
    SOURCE_MODE_ALL_NON_IMAGE,
    SOURCE_MODE_INSTRUCTION,
    SOURCE_MODE_QUERY,
    find_subsequence_positions,
)


class TeacherAttentionCache:
    """Loads precomputed teacher patch-importance vectors into memory."""

    def __init__(self, cache_path: str):
        batch_dir = Path(cache_path)
        if batch_dir.suffix == ".pt":
            batch_dir = batch_dir.with_suffix("")
        metadata_path = batch_dir / "metadata.pt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing teacher-attention metadata: {metadata_path}")

        self.batch_dir = batch_dir
        self.metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        if not isinstance(self.metadata, dict):
            raise ValueError(f"Expected dict metadata at {metadata_path}, got {type(self.metadata)!r}")

        self.vectors: Dict[str, torch.Tensor] = {}
        self.grids: Dict[str, torch.Tensor] = {}
        for shard_path in sorted(batch_dir.glob("batch-*.pt")):
            loaded = torch.load(shard_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected dict shard at {shard_path}, got {type(loaded)!r}")
            if "sample_ids" in loaded:
                sample_ids = loaded.get("sample_ids", [])
                scores = loaded.get("scores", [])
                grids = loaded.get("grids", None)
                if len(sample_ids) != len(scores):
                    raise ValueError(
                        f"Mismatched sample_ids/scores lengths in {shard_path}: "
                        f"{len(sample_ids)} vs {len(scores)}"
                    )
                if grids is not None and len(sample_ids) != len(grids):
                    raise ValueError(
                        f"Mismatched sample_ids/grids lengths in {shard_path}: "
                        f"{len(sample_ids)} vs {len(grids)}"
                    )
                for idx, (sample_id, tensor) in enumerate(zip(sample_ids, scores)):
                    sample_key = str(sample_id)
                    self.vectors[sample_key] = tensor.float()
                    if grids is not None and grids[idx] is not None:
                        self.grids[sample_key] = torch.as_tensor(grids[idx], dtype=torch.long)
            else:
                for sample_id, tensor in loaded.items():
                    self.vectors[str(sample_id)] = tensor.float()

    def get_many(self, sample_ids: Sequence[Optional[str]]) -> List[Optional[torch.Tensor]]:
        return [self.vectors.get(sample_id) if sample_id is not None else None for sample_id in sample_ids]

    def get_many_grids(self, sample_ids: Sequence[Optional[str]]) -> List[Optional[torch.Tensor]]:
        return [self.grids.get(sample_id) if sample_id is not None else None for sample_id in sample_ids]

    @property
    def prompt_mode(self) -> str:
        return str(self.metadata.get("prompt_mode", PROMPT_MODE_QUERY_IMAGE))

    @property
    def source_mode(self) -> str:
        return str(self.metadata.get("source_mode", SOURCE_MODE_QUERY))


def _last_non_none_attention(attentions: tuple) -> Optional[torch.Tensor]:
    for attn in reversed(attentions):
        if attn is not None:
            return attn
    return None


def _instruction_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    instruction_token_ids: Sequence[int],
) -> torch.Tensor:
    valid_ids = input_ids[attention_mask.bool()].tolist()
    positions = find_subsequence_positions(valid_ids, instruction_token_ids)
    return torch.tensor(positions, device=input_ids.device, dtype=torch.long)


def _all_non_image_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    valid = attention_mask.bool()
    positions = ((input_ids != image_token_id) & valid).nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        raise ValueError("No non-image source tokens found")
    return positions


def _query_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    query_token_ids: Sequence[int],
) -> torch.Tensor:
    valid_ids = input_ids[attention_mask.bool()].tolist()
    positions = find_subsequence_positions(valid_ids, query_token_ids)
    return torch.tensor(positions, device=input_ids.device, dtype=torch.long)


def extract_prior_patch_scores(
    attentions: tuple,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    source_mode: str,
    instruction_token_ids: Optional[Sequence[int]] = None,
) -> List[torch.Tensor]:
    """Extract document-only patch scores from student attentions.

    Supports only image-only prompt supervision. Query-conditioned attention
    alignment remains a separate future path.
    """

    attn_layer = _last_non_none_attention(attentions)
    if attn_layer is None:
        return [torch.zeros(0, device=input_ids.device)] * input_ids.shape[0]

    attn_avg = attn_layer.float().mean(dim=1)
    scores: List[torch.Tensor] = []

    for batch_idx in range(input_ids.shape[0]):
        input_ids_b = input_ids[batch_idx]
        attention_mask_b = attention_mask[batch_idx]
        image_positions = ((input_ids_b == image_token_id) & attention_mask_b.bool()).nonzero(as_tuple=False).squeeze(-1)
        if image_positions.numel() == 0:
            scores.append(torch.zeros(0, device=attn_avg.device))
            continue

        if source_mode == SOURCE_MODE_INSTRUCTION:
            if not instruction_token_ids:
                raise ValueError("instruction_token_ids are required for source_mode='instruction'")
            source_positions = _instruction_positions(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                instruction_token_ids=instruction_token_ids,
            )
        elif source_mode == SOURCE_MODE_ALL_NON_IMAGE:
            source_positions = _all_non_image_positions(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                image_token_id=image_token_id,
            )
        elif source_mode == SOURCE_MODE_QUERY:
            raise ValueError("Query-conditioned attention cache is not wired into training yet")
        else:
            raise ValueError(f"Unsupported source_mode={source_mode!r}")

        patch_scores = attn_avg[batch_idx][source_positions][:, image_positions].mean(dim=0)
        scores.append(patch_scores)

    return scores


def extract_query_patch_scores_from_similarity(
    *,
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    q_input_ids: torch.Tensor,
    q_attention_mask: torch.Tensor,
    d_input_ids: torch.Tensor,
    d_attention_mask: torch.Tensor,
    queries: Sequence[str],
    tokenizer,
    image_token_id: int,
    mode: str = "softmax_sum",
) -> List[torch.Tensor]:
    """Build student patch scores from query-page interaction.

    For each positive (query, doc) pair:
    1. Find the true query-text token span inside the query prompt.
    2. Select only image-patch tokens on the document side.
    3. Turn query-token × patch similarities into one vector over patches.
    """

    q_norm = F.normalize(q_emb.float(), p=2, dim=-1)
    d_norm = F.normalize(d_emb.float(), p=2, dim=-1)
    scores: List[torch.Tensor] = []

    for idx, query_text in enumerate(queries):
        query_token_ids = tokenizer.encode(str(query_text).strip(), add_special_tokens=False)
        if not query_token_ids:
            scores.append(torch.zeros(0, device=q_emb.device))
            continue

        try:
            query_positions = _query_positions(
                input_ids=q_input_ids[idx],
                attention_mask=q_attention_mask[idx],
                query_token_ids=query_token_ids,
            )
        except ValueError:
            scores.append(torch.zeros(0, device=q_emb.device))
            continue
        image_positions = (
            (d_input_ids[idx] == image_token_id) & d_attention_mask[idx].bool()
        ).nonzero(as_tuple=False).squeeze(-1)
        if image_positions.numel() == 0:
            scores.append(torch.zeros(0, device=q_emb.device))
            continue

        sim = q_norm[idx][query_positions] @ d_norm[idx][image_positions].T
        if mode == "sum":
            patch_scores = sim.sum(dim=0)
        elif mode == "mean":
            patch_scores = sim.mean(dim=0)
        else:
            patch_scores = sim.softmax(dim=-1).sum(dim=0)
        scores.append(patch_scores)

    return scores


def attention_alignment_loss(
    student_scores: Sequence[torch.Tensor],
    teacher_scores: Sequence[Optional[torch.Tensor]],
    *,
    loss_type: str = "kl",
    student_grids: Optional[Sequence[Optional[torch.Tensor]]] = None,
    teacher_grids: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, int]:
    """Align variable-length patch score vectors.

    Scores are normalized over patches before comparison so teacher/student
    scale mismatches do not dominate the loss.
    """

    losses: List[torch.Tensor] = []
    for idx, (student, teacher) in enumerate(zip(student_scores, teacher_scores)):
        if teacher is None or student.numel() == 0:
            continue
        teacher = teacher.to(device=student.device, dtype=student.dtype)
        if teacher.numel() != student.numel():
            student_grid = student_grids[idx] if student_grids is not None and idx < len(student_grids) else None
            teacher_grid = teacher_grids[idx] if teacher_grids is not None and idx < len(teacher_grids) else None
            teacher = _resize_teacher_scores_to_student_grid(
                teacher=teacher,
                teacher_grid=teacher_grid,
                student_grid=student_grid,
                student_numel=student.numel(),
            )
            if teacher is None:
                continue

        if loss_type == "raw_cosine":
            losses.append(
                1.0 - F.cosine_similarity(
                    student.float().unsqueeze(0),
                    teacher.float().unsqueeze(0),
                    dim=-1,
                ).mean()
            )
            continue

        student_prob = student.clamp(min=0)
        teacher_prob = teacher.clamp(min=0)
        student_prob = student_prob / (student_prob.sum() + 1e-9)
        teacher_prob = teacher_prob / (teacher_prob.sum() + 1e-9)

        if loss_type == "mse":
            losses.append(F.mse_loss(student_prob, teacher_prob))
        elif loss_type == "cosine":
            losses.append(
                1.0 - F.cosine_similarity(
                    student_prob.unsqueeze(0),
                    teacher_prob.unsqueeze(0),
                    dim=-1,
                ).mean()
            )
        else:
            losses.append(
                F.kl_div(
                    student_prob.add(1e-9).log(),
                    teacher_prob,
                    reduction="batchmean",
                )
            )

    if not losses:
        return torch.tensor(0.0), 0

    return torch.stack(losses).mean(), len(losses)


def _grid_hw_for_length(grid: Optional[torch.Tensor], length: int) -> Optional[Tuple[int, int]]:
    if grid is None:
        return None
    values = torch.as_tensor(grid).detach().cpu().flatten().tolist()
    values = [int(v) for v in values if int(v) > 0]
    if len(values) >= 3:
        t, h, w = values[-3], values[-2], values[-1]
        if h * w == length:
            return h, w
        if t * h * w == length:
            return t * h, w
    if len(values) >= 2:
        h, w = values[-2], values[-1]
        if h * w == length:
            return h, w
    return None


def _resize_teacher_scores_to_student_grid(
    *,
    teacher: torch.Tensor,
    teacher_grid: Optional[torch.Tensor],
    student_grid: Optional[torch.Tensor],
    student_numel: int,
) -> Optional[torch.Tensor]:
    teacher_hw = _grid_hw_for_length(teacher_grid, teacher.numel())
    student_hw = _grid_hw_for_length(student_grid, student_numel)
    if teacher_hw is None or student_hw is None:
        return None

    teacher_map = teacher.reshape(1, 1, teacher_hw[0], teacher_hw[1])
    resized = F.adaptive_max_pool2d(teacher_map, output_size=student_hw).flatten()
    if resized.numel() != student_numel:
        return None
    return resized


def ensure_prior_cache_compatible(metadata: Dict[str, object]) -> None:
    prompt_mode = metadata.get("prompt_mode", PROMPT_MODE_QUERY_IMAGE)
    if prompt_mode != PROMPT_MODE_IMAGE_ONLY:
        raise ValueError(
            "Training alignment currently supports only image-only teacher attention caches. "
            f"Received prompt_mode={prompt_mode!r}."
        )


def ensure_cache_prompt_mode(metadata: Dict[str, object], expected_prompt_mode: str) -> None:
    prompt_mode = str(metadata.get("prompt_mode", PROMPT_MODE_QUERY_IMAGE))
    if prompt_mode != expected_prompt_mode:
        raise ValueError(
            f"Expected teacher attention cache with prompt_mode={expected_prompt_mode!r}, "
            f"received {prompt_mode!r}."
        )
