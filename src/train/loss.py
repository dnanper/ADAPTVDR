"""Loss functions for ColPali training.

InfoNCELoss                    — symmetric InfoNCE for single-vector embeddings.
MaxSimLoss                     — ColBERT-style in-batch contrastive loss for multi-vector embeddings.
MatryoshkaMaxSimLoss           — Matryoshka MRL variant: MaxSim loss summed over multiple dim prefixes.
AugmentedMaxSimLoss            — MaxSim with hard-negative augmented doc batch (forward-only CE).
MatryoshkaAugmentedMaxSimLoss  — MRL + hard negatives combined (multivec_mrl + hard negs mode).
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE contrastive loss.

    Both (q→d) and (d→q) directions are averaged, which also makes the
    loss invariant to swapping query and document arguments.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Query embeddings    [B, D]
            d: Document embeddings [B, D]
        Returns:
            Scalar loss
        """
        q = F.normalize(q, p=2, dim=-1)
        d = F.normalize(d, p=2, dim=-1)
        sim = torch.matmul(q, d.T) / self.temperature          # [B, B]
        labels = torch.arange(sim.shape[0], device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
                                                                            

class MaxSimLoss(nn.Module):
    """ColBERT-style in-batch contrastive loss using MaxSim scoring.

    score(q_i, d_j) = Σ_t  max_s  (q_i[t] · d_j[s])

    The loss is symmetric cross-entropy over the [B, B] score matrix.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _compute_scores(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Compute the full [B, B] MaxSim score matrix.

        Args:
            q: [B, Nq, D]   — query multi-vectors
            d: [B, Nd, D]   — document multi-vectors
        Returns:
            scores: [B, B]  scores[i, j] = MaxSim(q_i, d_j)
        """
        # All-pairs dot products: [Bq, Bd, Nq, Nd]
        sim = torch.einsum("bqd,cnd->bcqn", q, d)
        # Max over document tokens per query token → [Bq, Bd, Nq]
        max_sim = sim.max(dim=-1).values
        # Sum over query tokens → [Bq, Bd]
        return max_sim.sum(dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        d: torch.Tensor,
        q_mask: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q:      [B, Nq, D]  query multi-vectors
            d:      [B, Nd, D]  document multi-vectors
            q_mask: [B, Nq]     bool, True = real token
            d_mask: [B, Nd]     bool, True = real token
        Returns:
            Scalar loss
        """
        # L2-normalize token embeddings (ColBERT/ColPali style: cosine similarity per token)
        # Normalize first, then zero out padding so pad tokens don't contribute to MaxSim
        q = F.normalize(q.float(), p=2, dim=-1) * q_mask.unsqueeze(-1).float()
        d = F.normalize(d.float(), p=2, dim=-1) * d_mask.unsqueeze(-1).float()

        scores = self._compute_scores(q, d) / self.temperature     # [B, B]
        labels = torch.arange(scores.shape[0], device=scores.device)
        return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


class MatryoshkaMaxSimLoss(MaxSimLoss):
    """Matryoshka Representation Learning variant of MaxSimLoss.

    Subclasses MaxSimLoss and reuses its ``_compute_scores`` static method.
    At each scale ``d`` in ``dims``, token embeddings are truncated to their
    first ``d`` features, L2-normalised, and a symmetric MaxSim contrastive
    loss is computed via the parent's scorer.  The final loss is the
    (optionally weighted) mean across all scales.

    Reference: Kusupati et al., "Matryoshka Representation Learning"
               (NeurIPS 2022). https://arxiv.org/abs/2205.13147
    """

    def __init__(
        self,
        dims: Optional[List[int]] = None,
        temperature: float = 1.0,
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            dims:        Embedding prefix dimensions to train at, e.g.
                         [64, 128, 256, 512, 1024].  Sorted ascending
                         internally.  Defaults to [64, 128, 256, 512, 1024].
            temperature: Softmax temperature (same as MaxSimLoss).
            weights:     Per-dim loss weights (auto-normalised to sum 1).
                         None → uniform weighting.
        """
        super().__init__(temperature=temperature)
        self.dims = sorted(dims) if dims is not None else [64, 128, 256, 512, 1024]

        if weights is not None:
            if len(weights) != len(self.dims):
                raise ValueError(
                    f"len(weights)={len(weights)} must equal len(dims)={len(self.dims)}"
                )
            total = sum(weights)
            self.weights: Optional[List[float]] = [w / total for w in weights]
        else:
            self.weights = None  # uniform

    def forward(
        self,
        q: torch.Tensor,
        d: torch.Tensor,
        q_mask: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q:      [B, Nq, D]  query multi-vectors
            d:      [B, Nd, D]  document multi-vectors
            q_mask: [B, Nq]     bool, True = real token
            d_mask: [B, Nd]     bool, True = real token
        Returns:
            Scalar Matryoshka loss (mean over all valid dim scales)
        """
        D = q.shape[-1]
        q_f = q.float()
        d_f = d.float()
        mask_q = q_mask.unsqueeze(-1).float()  # [B, Nq, 1]
        mask_d = d_mask.unsqueeze(-1).float()  # [B, Nd, 1]

        losses: List[torch.Tensor] = []
        for dim in self.dims:
            if dim > D:
                break  # skip any dim larger than the actual embedding size
            q_trunc = F.normalize(q_f[..., :dim], p=2, dim=-1) * mask_q
            d_trunc = F.normalize(d_f[..., :dim], p=2, dim=-1) * mask_d
            # Reuse parent's _compute_scores (einsum MaxSim) directly
            scores = self._compute_scores(q_trunc, d_trunc) / self.temperature
            labels = torch.arange(scores.shape[0], device=scores.device)
            losses.append(
                (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
            )

        if not losses:
            raise ValueError(
                f"No valid dims ≤ {D} found in self.dims={self.dims}. "
                "Check the 'dims' argument."
            )

        if self.weights is not None:
            n = len(losses)
            w = self.weights[:n]
            total_w = sum(w)
            return sum(loss * (wt / total_w) for loss, wt in zip(losses, w))  # type: ignore[return-value]

        return torch.stack(losses).mean()


class AugmentedMaxSimLoss(MaxSimLoss):
    """MaxSim loss with hard-negative augmented doc batch.

    Doc batch layout: [B positives | hard negatives]
    where B = pos_count and hard negatives immediately follow.

    Unlike MaxSimLoss (symmetric), this is forward-only cross-entropy:
        loss = CE(scores[:B], labels)
    because hard negatives have no associated query — the reverse direction
    (d→q) is undefined for negatives.

    The forward scores matrix is [B, B + n_hard_negs], which automatically
    includes all in-batch negatives AND the hard negatives as wrong choices.
    """

    def forward(
        self,
        q:         torch.Tensor,
        d:         torch.Tensor,
        q_mask:    torch.Tensor,
        d_mask:    torch.Tensor,
        pos_count: int,
    ) -> torch.Tensor:
        """
        Args:
            q:         [B, Nq, D]           query multi-vectors
            d:         [B + n_neg, Nd, D]   augmented doc multi-vectors
            q_mask:    [B, Nq]              bool
            d_mask:    [B + n_neg, Nd]      bool
            pos_count: int                  = B (positives are d[0:pos_count])
        Returns:
            Scalar loss
        """
        q = F.normalize(q.float(), p=2, dim=-1) * q_mask.unsqueeze(-1).float()
        d = F.normalize(d.float(), p=2, dim=-1) * d_mask.unsqueeze(-1).float()

        # scores: [B, B + n_neg]
        scores = self._compute_scores(q, d) / self.temperature
        labels = torch.arange(pos_count, device=scores.device)

        # Forward-only: query finds its positive among all docs (including hard negs)
        return F.cross_entropy(scores[:pos_count], labels)


class MatryoshkaAugmentedMaxSimLoss(MaxSimLoss):
    """Matryoshka MRL + hard-negative augmented doc batch.

    Combines MatryoshkaMaxSimLoss and AugmentedMaxSimLoss:
      - Hard negatives: doc batch = [B positives | hard negatives]
      - Forward-only CE at each MRL dim prefix
      - Final loss = mean over all dim scales

    Use for multivec_mrl mode with hard negatives.
    """

    def __init__(
        self,
        dims: Optional[List[int]] = None,
        temperature: float = 1.0,
        weights: Optional[List[float]] = None,
    ):
        super().__init__(temperature=temperature)
        self.dims = sorted(dims) if dims is not None else [128, 256, 512, 1024]
        if weights is not None:
            if len(weights) != len(self.dims):
                raise ValueError(f"len(weights)={len(weights)} != len(dims)={len(self.dims)}")
            total = sum(weights)
            self.weights: Optional[List[float]] = [w / total for w in weights]
        else:
            self.weights = None

    def forward(
        self,
        q:         torch.Tensor,
        d:         torch.Tensor,
        q_mask:    torch.Tensor,
        d_mask:    torch.Tensor,
        pos_count: int,
    ) -> torch.Tensor:
        """
        Args:
            q:         [B, Nq, D]
            d:         [B + n_neg, Nd, D]   positives first, then hard negs
            q_mask:    [B, Nq]
            d_mask:    [B + n_neg, Nd]
            pos_count: B (number of positives = number of queries)
        """
        D = q.shape[-1]
        q_f    = q.float()
        d_f    = d.float()
        mask_q = q_mask.unsqueeze(-1).float()
        mask_d = d_mask.unsqueeze(-1).float()
        labels = torch.arange(pos_count, device=q.device)

        losses: List[torch.Tensor] = []
        for dim in self.dims:
            if dim > D:
                break
            q_trunc = F.normalize(q_f[..., :dim], p=2, dim=-1) * mask_q
            d_trunc = F.normalize(d_f[..., :dim], p=2, dim=-1) * mask_d
            scores  = self._compute_scores(q_trunc, d_trunc) / self.temperature  # [B, B+n_neg]
            losses.append(F.cross_entropy(scores[:pos_count], labels))

        if not losses:
            raise ValueError(f"No valid dims ≤ {D} in self.dims={self.dims}")

        if self.weights is not None:
            n = len(losses)
            w = self.weights[:n]
            total_w = sum(w)
            return sum(loss * (wt / total_w) for loss, wt in zip(losses, w))  # type: ignore[return-value]

        return torch.stack(losses).mean()
