"""visualize_attention.py — Overlay attention heatmap on document images.

Usage:
    python scripts/visualize_attention.py

Saves results to results/attention_heatmaps/
"""

import os
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from io import BytesIO
import pandas as pd

from scripts.colpali_embedding import ColPaliEmbedder
from scripts.adaptive_pruning import extract_image_patch_scores, IMAGE_TOKEN_ID_COLPALI

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "/data2/cmdir/home/test01/longvnu/stable_diff/models/vidore/colpali"
PARQUET     = (
    "dataset/vidore/datasets--vidore--docvqa_test_subsampled/snapshots/"
    "49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6/data/test-00000-of-00001.parquet"
)
N_IMAGES    = 4
OUT_DIR     = "results/attention_heatmaps"
ALPHA       = 0.55       # heatmap opacity
PATCH_GRID  = 32         # SigLIP-So400m/14: 448/14 = 32 patches per side
COLORMAP    = "inferno"  # "hot", "jet", "viridis", "inferno"

# ── Helpers ───────────────────────────────────────────────────────────────────

def scores_to_heatmap(scores: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    """Reshape [1024] patch scores → [H, W] numpy heatmap at image resolution."""
    n = scores.shape[0]
    grid = int(n ** 0.5)
    assert grid * grid == n, f"Expected square patch grid, got {n} patches"

    heat = scores.float().cpu().numpy().reshape(grid, grid)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)  # normalize 0–1

    # Upsample to image size using PIL
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (img_w, img_h), resample=Image.BILINEAR
    )
    return np.array(heat_img) / 255.0   # [H, W] float 0–1


def overlay_heatmap(
    orig_img:  Image.Image,
    scores:    torch.Tensor,
    alpha:     float = ALPHA,
    cmap_name: str   = COLORMAP,
) -> Image.Image:
    """Blend attention heatmap on top of original image."""
    orig_np = np.array(orig_img.convert("RGB")).astype(float) / 255.0
    H, W    = orig_np.shape[:2]

    heat = scores_to_heatmap(scores, W, H)              # [H, W] 0–1
    cmap = cm.get_cmap(cmap_name)
    heat_rgba = cmap(heat)                              # [H, W, 4]  RGBA
    heat_rgb  = heat_rgba[:, :, :3]                     # [H, W, 3]

    blended = (1 - alpha) * orig_np + alpha * heat_rgb  # [H, W, 3]
    blended = (blended * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def make_figure(
    orig_img:    Image.Image,
    overlay_img: Image.Image,
    scores:      torch.Tensor,
    idx:         int,
    keep_ratio:  float,
    n_keep:      int,
    n_total:     int,
) -> plt.Figure:
    """3-panel figure: original | heatmap overlay | patch grid bar."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"Image [{idx}]  —  keep_ratio={keep_ratio:.1%}  "
        f"({n_keep}/{n_total} patches)",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: original
    axes[0].imshow(orig_img)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    # Panel 2: heatmap overlay
    axes[1].imshow(overlay_img)
    axes[1].set_title(f"Attention Heatmap (α={ALPHA})", fontsize=11)
    axes[1].axis("off")

    # Panel 3: patch importance bar (sorted)
    grid = int(scores.shape[0] ** 0.5)
    heat = scores.float().cpu().numpy().reshape(grid, grid)
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)

    # Show patch grid as image with colorbar
    im = axes[2].imshow(heat_norm, cmap=COLORMAP, vmin=0, vmax=1)
    axes[2].set_title(f"Patch Importance Grid ({grid}×{grid})", fontsize=11)
    axes[2].set_xlabel("Patch column")
    axes[2].set_ylabel("Patch row")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Importance")

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load images
    print("Loading test images...")
    df   = pd.read_parquet(PARQUET, columns=["image"]).head(N_IMAGES)
    imgs = [Image.open(BytesIO(r["image"]["bytes"])).convert("RGB")
            for _, r in df.iterrows()]
    print(f"  {len(imgs)} images: {[img.size for img in imgs]}")

    # 2. Load embedder
    print("\nLoading ColPaliEmbedder...")
    embedder = ColPaliEmbedder(MODEL_NAME, attn_implementation="eager")
    print(f"  Device: {embedder.device}")

    # 3. Forward pass with attentions
    print("\nRunning forward pass (output_attentions=True)...")
    batch  = embedder.process_images(imgs)
    out    = embedder._forward(batch, output_attentions=True)

    embs        = out.embeddings      # [B, N, 128]
    input_ids   = out.input_ids       # [B, N]
    attn_mask   = out.attention_mask  # [B, N]
    attentions  = out.attentions      # tuple of [B, H, N, N]

    print(f"  Sequence length : {embs.shape[1]}")
    print(f"  Attention layers: {len(attentions)}")

    # 4. Extract per-patch scores
    patch_scores = extract_image_patch_scores(
        attentions=attentions,
        input_ids=input_ids,
        attention_mask=attn_mask,
        image_token_id=IMAGE_TOKEN_ID_COLPALI,
        layer_idx=-1,
    )

    # 5. Generate heatmaps
    print(f"\nGenerating heatmaps → {OUT_DIR}/")
    for i, (img, scores) in enumerate(zip(imgs, patch_scores)):
        n_patches = scores.shape[0]
        assert n_patches == PATCH_GRID ** 2, (
            f"Expected {PATCH_GRID**2} patches, got {n_patches}. "
            f"Check PATCH_GRID setting."
        )

        # Compute keep stats (r_min=0.3, r_max=0.9 defaults)
        from scripts.adaptive_pruning import compute_keep_ratio
        keep_ratio = compute_keep_ratio(scores, r_min=0.3, r_max=0.99)
        n_keep     = max(1, int(keep_ratio * n_patches))

        # Top-k patch mask for overlay
        topk_vals, topk_idx = scores.topk(n_keep)
        binary_mask = torch.zeros(n_patches)
        binary_mask[topk_idx] = 1.0

        # Generate overlay images
        overlay     = overlay_heatmap(img, scores,       alpha=ALPHA)
        overlay_bin = overlay_heatmap(img, binary_mask,  alpha=0.45, cmap_name="RdYlGn")

        # Save 3-panel figure (continuous heatmap)
        fig = make_figure(img, overlay, scores, i, keep_ratio, n_keep, n_patches)
        out_path = os.path.join(OUT_DIR, f"img_{i:02d}_heatmap.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  img[{i}]: keep={keep_ratio:.1%} ({n_keep}/{n_patches})  → {out_path}")

        # Save binary keep/drop overlay
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 7))
        fig2.suptitle(
            f"Image [{i}]  —  Kept patches (green) vs Pruned (red)  "
            f"keep={keep_ratio:.1%} ({n_keep}/{n_patches})",
            fontsize=12, fontweight="bold",
        )
        axes2[0].imshow(img);          axes2[0].set_title("Original"); axes2[0].axis("off")
        axes2[1].imshow(overlay_bin);  axes2[1].set_title("Keep (green) / Drop (red)"); axes2[1].axis("off")
        plt.tight_layout()
        bin_path = os.path.join(OUT_DIR, f"img_{i:02d}_binary.png")
        fig2.savefig(bin_path, dpi=120, bbox_inches="tight")
        plt.close(fig2)

    print(f"\nDone! Saved {N_IMAGES * 2} figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
