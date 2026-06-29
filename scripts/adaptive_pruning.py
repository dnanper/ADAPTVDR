"""adaptive_pruning.py — Entropy-based adaptive patch pruning for ColPali/ColQwen3.5/ColQwen3.

Novelty: Instead of a fixed keep-ratio (e.g. HPC-ColPali keeps top-60% patches always),
we use the Shannon entropy of the last-layer attention map to set a dynamic threshold:

    keep_ratio = r_min + (r_max - r_min) * (H / H_max)

where H = entropy of per-patch attention scores, H_max = log2(N) for N patches.

* Dense text pages  → high entropy (distributed attention)  → keep MORE patches (→ r_max)
* Whitespace-heavy  → low entropy (concentrated attention)  → keep FEWER patches (→ r_min)

Supports three backbone families:
  - ColPali     (PaliGemma-3B, standard transformer):   image_token_id=257152
  - ColQwen3.5  (Qwen3.5-0.8B, hybrid DeltaNet+GQA):  image_token_id=248056
  - ColQwen3    (Qwen3VL 2B/4B, standard transformer):  image_token_id=151655

Usage:
    from scripts.adaptive_pruning import AdaptivePruner, IMAGE_TOKEN_ID_COLPALI

    pruner = AdaptivePruner(r_min=0.3, r_max=0.9, image_token_id=IMAGE_TOKEN_ID_COLPALI)
    pruned_list, stats = pruner.prune_doc(
        hidden_states=hidden,      # [B, N, D]  — model projected + normalized embeddings
        attentions=attns,          # tuple of per-layer attentions from forward(output_attentions=True)
        input_ids=input_ids,       # [B, N]
        attention_mask=attn_mask,  # [B, N]
    )
    # pruned_list: List[Tensor[k_b, D]] — variable length per sample (text + pruned patches)
    # stats.mean_keep_ratio: float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_TOKEN_ID_COLPALI: int = 257152   # PaliGemma-3B  (<image> token)
IMAGE_TOKEN_ID_QWEN35:  int = 248056   # Qwen3.5-0.8B  (Qwen3_5Tokenizer)
IMAGE_TOKEN_ID_QWEN3VL: int = 151655   # Qwen3VL 2B/4B (Qwen2VLProcessor)

# Use last layer by default (always full-attention in both architectures)
FULL_ATTENTION_LAYER_IDX: int = -1


# ── Entropy core ──────────────────────────────────────────────────────────────

def compute_entropy(attn_scores: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a 1-D attention-score vector (bits).

    We L1-normalize the scores rather than re-applying softmax because the
    model already outputs post-softmax weights; re-softmax would suppress the
    distribution and under-estimate entropy for concentrated maps.

    Args:
        attn_scores: [N]  per-patch importance (≥0, any scale).
    Returns:
        Scalar entropy H ∈ [0, log2(N)] bits.
    """
    scores = attn_scores.float().clamp(min=0)
    probs  = scores / (scores.sum() + 1e-9)          # L1-normalize
    return -(probs * torch.log2(probs + 1e-9)).sum()  # bits


def compute_keep_ratio(
    attn_scores: torch.Tensor,
    r_min: float = 0.3,
    r_max: float = 0.9,
) -> float:
    """Map per-patch entropy to a keep_ratio ∈ [r_min, r_max].

    Formula:  keep = r_min + (r_max − r_min) * (H / H_max)
    """
    N     = attn_scores.shape[0]
    H     = compute_entropy(attn_scores)
    H_max = torch.log2(torch.tensor(float(N), device=attn_scores.device))
    keep  = r_min + (r_max - r_min) * (H / (H_max + 1e-9)).clamp(0.0, 1.0)
    return float(keep)


def compute_keep_ratio_perplexity(
    attn_scores: torch.Tensor,
    tau: float = 2.0,
    r_min: float = 0.0,
) -> float:
    """Perplexity-based keep ratio — derived purely from information theory.

    Perplexity(p) = 2^H = effective number of patches carrying information.
    Normalized by N = 2^H_max gives the fraction of patches to keep.

    Formula:  keep = max(r_min, 2^((H − H_max) / τ))

    τ = temperature:
        τ = 1  →  keep = perplexity / N  (aggressive, 0 extra hyperparams)
        τ = 2  →  softer (recommended default)
        τ → ∞  →  keep → 1.0 (no pruning)

    Range: [r_min, 1] — bounded below for retrieval safety.
    """
    N     = attn_scores.shape[0]
    H     = compute_entropy(attn_scores)
    H_max = torch.log2(torch.tensor(float(N), device=attn_scores.device))
    keep  = 2.0 ** (((H - H_max) / tau).clamp(-20.0, 0.0))
    keep  = keep.clamp(min=r_min, max=1.0)
    return float(keep)


def adaptive_prune(
    embeddings:  torch.Tensor,
    attn_scores: torch.Tensor,
    r_min: float = 0.3,
    r_max: float = 0.9,
    mode:  str   = "linear",
    tau:   float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prune patch embeddings, keeping the top-k by attention score.

    Spatial order is preserved (indices are sorted ascending after top-k).

    Args:
        embeddings:  [N, D]  per-patch embeddings.
        attn_scores: [N]     per-patch importance score.
        r_min, r_max:        keep-ratio bounds (used when mode="linear").
        mode:                "linear" (default) or "perplexity".
        tau:                 temperature for perplexity mode.
    Returns:
        (pruned_emb [k, D],  kept_indices [k])
    """
    N = embeddings.shape[0]
    if mode == "perplexity":
        keep_ratio = compute_keep_ratio_perplexity(attn_scores, tau, r_min)
    else:
        keep_ratio = compute_keep_ratio(attn_scores, r_min, r_max)
    k = max(1, int(keep_ratio * N))
    _, top_indices  = attn_scores.topk(k)
    top_indices, _  = top_indices.sort()          # restore spatial order
    return embeddings[top_indices], top_indices


# ── Attention extraction ───────────────────────────────────────────────────────

def extract_image_patch_scores(
    attentions:      tuple,
    input_ids:       torch.Tensor,
    attention_mask:  Optional[torch.Tensor] = None,
    image_token_id:  int = IMAGE_TOKEN_ID_QWEN35,
    layer_idx:       int = FULL_ATTENTION_LAYER_IDX,
) -> Optional[List[torch.Tensor]]:
    """Extract per-image-patch importance from model attention weights.

    Strategy:
        1. Walk to the target layer (default: last). For Qwen3.5 hybrid, most
           layers are DeltaNet (attentions[i] = None); we skip those and use
           the last non-None entry — always the last GQA layer (layer 23).
        2. Average over attention heads to get [N_src, N_tgt] per sample.
        3. Compute text→image attention: average over text-token rows,
           for the columns that correspond to image-patch tokens.
           This captures "which patches was the model looking at from text?"

    Args:
        attentions:     Tuple, one entry per layer.  Each is either None
                        (linear/DeltaNet layer) or Tensor [B, H, N, N].
        input_ids:      [B, N]  token ids.
        attention_mask: [B, N]  1 = real token, 0 = padding. When provided,
                padding tokens are excluded from text/image selection.
        image_token_id: ID identifying image-patch tokens.
        layer_idx:      Layer to use. -1 = last non-None layer.
    Returns:
        List[Tensor[n_img_patches_b]] — one importance vector per batch item,
        or None if no valid attention layer was found.
    """
    if attentions is None or len(attentions) == 0:
        return None

    # Find the target layer, searching backwards for first non-None
    if layer_idx == -1:
        search_order = range(len(attentions) - 1, -1, -1)
    else:
        search_order = list(range(layer_idx, -1, -1))

    attn_layer = None
    for idx in search_order:
        if attentions[idx] is not None:
            attn_layer = attentions[idx]   # [B, H, N, N]
            break

    if attn_layer is None:
        return None

    B, H, N, _ = attn_layer.shape

    if attention_mask is None:
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid_mask = attention_mask.bool()

    img_mask  = (input_ids == image_token_id) & valid_mask   # [B, N] True = image patch
    text_mask = (~img_mask) & valid_mask                     # [B, N] True = non-pad text token

    if not img_mask.any():
        return None

    # Average over heads → [B, N_src, N_tgt]
    attn_avg = attn_layer.float().mean(dim=1)

    scores_list: List[torch.Tensor] = []
    for b in range(B):
        img_positions  = img_mask[b].nonzero(as_tuple=True)[0]
        text_positions = text_mask[b].nonzero(as_tuple=True)[0]

        if len(img_positions) == 0:
            scores_list.append(torch.zeros(0, device=attn_layer.device))
            continue

        if len(text_positions) == 0:
            # No text tokens — use image-to-image self-attention
            patch_scores = attn_avg[b][img_positions][:, img_positions].mean(dim=0)
        else:
            # Text-to-image attention: rows=text, cols=image → mean over text rows
            patch_scores = attn_avg[b][text_positions][:, img_positions].mean(dim=0)

        scores_list.append(patch_scores)   # [n_img_patches_b]

    return scores_list


# ── Stats dataclass ────────────────────────────────────────────────────────────

@dataclass
class PruningStats:
    """Per-batch statistics from one adaptive pruning pass."""
    original_patches: List[int]   = field(default_factory=list)
    kept_patches:     List[int]   = field(default_factory=list)
    keep_ratios:      List[float] = field(default_factory=list)

    @property
    def mean_keep_ratio(self) -> float:
        return sum(self.keep_ratios) / len(self.keep_ratios) if self.keep_ratios else 0.0

    def __repr__(self) -> str:
        avg = self.mean_keep_ratio
        total_orig = sum(self.original_patches)
        total_kept = sum(self.kept_patches)
        return (
            f"PruningStats(batch={len(self.keep_ratios)}, "
            f"total_patches={total_orig}→{total_kept}, "
            f"mean_keep={avg:.2%})"
        )


# ── High-level pruner ─────────────────────────────────────────────────────────

class AdaptivePruner:
    """Entropy-based adaptive pruner for ColPali-style document embeddings.

    Works with:
      - ColQwen3.5  (Qwen3.5-0.8B hybrid)  — image_token_id=248056
      - ColQwen3    (Qwen3VL 2B/4B)         — image_token_id=151655

    The pruner keeps ALL text tokens and prunes only image-patch tokens.
    Pruned patches are those with the lowest text→image attention score.

    After pruning, text tokens come first (in original order), then the
    surviving image patches (in original spatial order).

    Args:
        r_min:          Minimum keep ratio (applied to whitespace-heavy pages). Default 0.3.
        r_max:          Maximum keep ratio (applied to dense text pages).       Default 0.9.
        image_token_id: Token ID for image patches.
        layer_idx:      Which attention layer to use. -1 = last (recommended).
        normalize:      L2-normalize pruned embeddings after pruning.
    """

    def __init__(
        self,
        r_min:          float = 0.3,
        r_max:          float = 0.9,
        mode:           str   = "linear",   # "linear" | "perplexity"
        tau:            float = 2.0,        # temperature for perplexity mode
        image_token_id: int   = IMAGE_TOKEN_ID_QWEN35,
        layer_idx:      int   = FULL_ATTENTION_LAYER_IDX,
        keep_text_tokens: bool = True,
        normalize:      bool  = True,
    ):
        self.r_min          = r_min
        self.r_max          = r_max
        self.mode           = mode
        self.tau            = tau
        self.image_token_id = image_token_id
        self.layer_idx      = layer_idx
        self.keep_text_tokens = keep_text_tokens
        self.normalize      = normalize

    def _prune_from_patch_scores(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_scores_list: Optional[List[torch.Tensor]],
        normalize: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], PruningStats]:
        do_norm = normalize if normalize is not None else self.normalize
        B = hidden_states.shape[0]

        stats        = PruningStats()
        pruned_list: List[torch.Tensor] = []

        for b in range(B):
            mask_b = attention_mask[b].bool()
            emb_b  = hidden_states[b][mask_b]
            ids_b  = input_ids[b][mask_b]

            img_mask_b  = (ids_b == self.image_token_id)
            img_indices = img_mask_b.nonzero(as_tuple=True)[0]

            no_scores = (
                patch_scores_list is None
                or b >= len(patch_scores_list)
                or len(patch_scores_list[b]) == 0
            )
            if no_scores or len(img_indices) == 0:
                fallback_emb = emb_b if self.keep_text_tokens else emb_b[img_indices]
                final_emb = F.normalize(fallback_emb.float(), p=2, dim=-1).to(fallback_emb.dtype) if do_norm else fallback_emb
                pruned_list.append(final_emb)
                n = len(img_indices) if len(img_indices) > 0 else emb_b.shape[0]
                stats.original_patches.append(n)
                stats.kept_patches.append(n)
                stats.keep_ratios.append(1.0)
                continue

            scores_b = patch_scores_list[b]
            if len(scores_b) != len(img_indices):
                fallback_emb = emb_b if self.keep_text_tokens else emb_b[img_indices]
                final_emb = F.normalize(fallback_emb.float(), p=2, dim=-1).to(fallback_emb.dtype) if do_norm else fallback_emb
                pruned_list.append(final_emb)
                stats.original_patches.append(len(img_indices))
                stats.kept_patches.append(len(img_indices))
                stats.keep_ratios.append(1.0)
                continue

            img_emb_b = emb_b[img_indices]
            pruned_patches, _ = adaptive_prune(
                img_emb_b, scores_b, self.r_min, self.r_max,
                mode=self.mode, tau=self.tau,
            )

            if self.keep_text_tokens:
                text_emb_b = emb_b[~img_mask_b]
                final_emb = torch.cat([text_emb_b, pruned_patches], dim=0)
            else:
                final_emb = pruned_patches

            if do_norm:
                final_emb = F.normalize(final_emb.float(), p=2, dim=-1).to(emb_b.dtype)

            pruned_list.append(final_emb)
            n_orig = len(img_indices)
            n_kept = pruned_patches.shape[0]
            stats.original_patches.append(n_orig)
            stats.kept_patches.append(n_kept)
            stats.keep_ratios.append(n_kept / max(1, n_orig))

        return pruned_list, stats

    def prune_doc_with_patch_scores(
        self,
        hidden_states: torch.Tensor,
        patch_scores_list: List[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], PruningStats]:
        """Prune document embeddings from externally supplied patch scores.

        This is the hook used when patch importance comes from query-document
        interaction rather than document self-attention.
        """
        return self._prune_from_patch_scores(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_scores_list=patch_scores_list,
            normalize=normalize,
        )

    def prune_doc(
        self,
        hidden_states:  torch.Tensor,
        attentions:     tuple,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        normalize:      Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], PruningStats]:
        """Prune document patch embeddings for a batch of documents.

        Flow per sample:
            1. Mask out padding tokens (via attention_mask).
            2. Extract per-patch attention scores from the chosen layer.
            3. Compute adaptive keep_ratio from patch entropy.
            4. Prune image patches to top-k; keep all text tokens.
            5. (Optionally) L2-normalize the final embeddings.

        Args:
            hidden_states:  [B, N, D]  embeddings from model.forward()
                            (may already be projected, e.g. from ColQwen3).
            attentions:     Tuple of per-layer attention tensors (from
                            model.forward(output_attentions=True)).
            input_ids:      [B, N]  integer token ids.
            attention_mask: [B, N]  1 = real, 0 = pad.
            normalize:      Override instance normalize flag if not None.
        Returns:
            (pruned_list, stats)
              pruned_list: List[Tensor[k_b, D]] — one tensor per batch item
              stats:       PruningStats
        """
        patch_scores_list = extract_image_patch_scores(
            attentions=attentions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_id=self.image_token_id,
            layer_idx=self.layer_idx,
        )
        return self._prune_from_patch_scores(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_scores_list=patch_scores_list,
            normalize=normalize,
        )
