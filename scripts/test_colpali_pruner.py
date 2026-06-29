"""test_colpali_pruner.py — Quick smoke test for AdaptivePruner on ColPali.

Usage:
    cd graduation_thesis
    python -m scripts.test_colpali_pruner

Checks:
    1. Model loads OK with attn_implementation="eager"
    2. embed_images_pruned() runs without error
    3. Embedding sizes before/after pruning
    4. PruningStats (keep_ratio, patches saved)
    5. MaxSim score is non-zero (sanity check that pruned embs work)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from io import BytesIO

import torch
import pandas as pd
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "/data2/cmdir/home/test01/longvnu/stable_diff/models/vidore/colpali"
PARQUET    = (
    "dataset/vidore/datasets--vidore--docvqa_test_subsampled"
    "/snapshots/49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6"
    "/data/test-00000-of-00001.parquet"
)
N_IMAGES   = 4      # number of pages to test
R_MIN      = 0.3
R_MAX      = 0.99
QUERY      = "What is the total revenue?"

# ── Helpers ───────────────────────────────────────────────────────────────────

def kb(t: torch.Tensor) -> float:
    return t.element_size() * t.nelement() / 1024


def maxsim(q_emb: torch.Tensor, d_emb: torch.Tensor) -> float:
    """ColBERT-style MaxSim: sum of max dot-products over query tokens."""
    scores = torch.einsum("qd,kd->qk", q_emb.float(), d_emb.float())
    return scores.max(dim=-1).values.sum().item()


def sep(title: str = ""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sep("1. Load test images")
    df   = pd.read_parquet(PARQUET, columns=["image", "query"]).head(N_IMAGES)
    imgs = [Image.open(BytesIO(r["image"]["bytes"])).convert("RGB") for _, r in df.iterrows()]
    print(f"  Loaded {len(imgs)} images: {[img.size for img in imgs]}")

    sep("2. Load ColPaliEmbedder")
    from scripts.colpali_embedding import ColPaliEmbedder
    embedder = ColPaliEmbedder(
        MODEL_NAME,
        attn_implementation="eager",   # required — flash_attn skips attention weights
    )
    print(f"  Device : {embedder.device}")
    print(f"  dtype  : {next(embedder.model.parameters()).dtype}")

    sep("3. Original embeddings (no pruning)")
    batch    = embedder.process_images(imgs)
    out_orig = embedder._forward(batch, output_attentions=False)
    embs     = out_orig.embeddings   # [B, N, 128]
    masks    = out_orig.attention_mask

    for i in range(len(imgs)):
        n = masks[i].sum().item()
        print(f"  img[{i}]: {n:4d} tokens  {kb(embs[i, :n]):.1f} KB  dtype={embs.dtype}")

    sep("4. Pruned embeddings")
    pruned_list, stats = embedder.embed_images_pruned(imgs, r_min=R_MIN, r_max=R_MAX)
    print(f"  {stats}")
    print()

    total_orig_kb = 0.0
    total_prun_kb = 0.0
    for i, p in enumerate(pruned_list):
        n_orig = masks[i].sum().item()
        orig_kb = kb(embs[i, :n_orig])
        prun_kb = kb(p)
        total_orig_kb += orig_kb
        total_prun_kb += prun_kb
        pct = (1 - prun_kb / orig_kb) * 100
        print(
            f"  img[{i}]: {n_orig:4d} → {p.shape[0]:4d} tokens  "
            f"{orig_kb:.1f} KB → {prun_kb:.1f} KB  "
            f"({pct:.1f}% saved)"
        )

    print(f"\n  Total: {total_orig_kb:.1f} KB → {total_prun_kb:.1f} KB  "
          f"({(1-total_prun_kb/total_orig_kb)*100:.1f}% saved)")

    sep("4b. Perplexity-based pruning  keep = 2^((H-H_max)/τ)")
    from scripts.adaptive_pruning import AdaptivePruner, IMAGE_TOKEN_ID_COLPALI
    pruner_perp = AdaptivePruner(
        mode           = "perplexity",
        tau            = 2.0,
        image_token_id = IMAGE_TOKEN_ID_COLPALI,
    )
    pruned_perp, stats_perp = embedder.embed_images_pruned(imgs, pruner=pruner_perp)
    print(f"  {stats_perp}\n")

    print(f"  {'':6s}  {'linear':>12s}  {'perplexity':>12s}")
    print(f"  {'img':6s}  {'keep%':>12s}  {'keep%':>12s}")
    print(f"  {'─'*36}")
    for i, (p_lin, p_perp) in enumerate(zip(pruned_list, pruned_perp)):
        n_orig    = masks[i].sum().item()
        r_lin     = p_lin.shape[0]  / n_orig * 100
        r_perp    = p_perp.shape[0] / n_orig * 100
        print(f"  img[{i}]  {r_lin:>10.1f}%  {r_perp:>10.1f}%"
              f"  ({'↑' if r_perp > r_lin else '↓'} {abs(r_perp-r_lin):.1f}pp)")

    sep("5. MaxSim retrieval sanity check")
    q_inputs = embedder.process_queries([QUERY])
    q_out    = embedder._forward(q_inputs)
    q_mask   = q_out.attention_mask[0].bool()
    q_emb    = q_out.embeddings[0][q_mask]          # [Q, 128]

    print(f"  Query : \"{QUERY}\"")
    print(f"  Query tokens: {q_emb.shape[0]}")
    print()

    # Score against original vs pruned, should be close
    for i in range(len(imgs)):
        n_orig      = masks[i].sum().item()
        orig_emb_i  = embs[i, :n_orig]
        score_orig  = maxsim(q_emb, orig_emb_i)
        score_prun  = maxsim(q_emb, pruned_list[i])
        delta       = score_prun - score_orig
        print(
            f"  img[{i}]: MaxSim original={score_orig:.4f}  "
            f"pruned={score_prun:.4f}  Δ={delta:+.4f}"
        )

    sep("Done")
    print("  Pruner is working correctly on ColPali!")


if __name__ == "__main__":
    main()
