## Contributing & Future Directions

This repository accompanies the thesis *"Optimization of ColPali for Document Retrieval"* and is open for further research and experimentation. Below are concrete directions worth exploring.

### Pruning Strategies

The current `AdaptivePruner` (`scripts/adaptive_pruning.py`) supports two keep-ratio modes:

| Mode | Formula | Hyperparams |
|---|---|---|
| `linear` | `r_min + (r_max − r_min) · H/H_max` | `r_min`, `r_max` |
| `perplexity` | `2^((H − H_max) / τ)` | `τ` |

Potential contributions:
- **New scoring functions** — instead of CLS/text→image attention, try L2-norm of 128-dim embeddings or CLIP-based saliency scores. See `extract_image_patch_scores()` in `adaptive_pruning.py`.
- **Learned threshold** — replace `τ` with a small MLP trained via distillation from full-embedding retrieval scores.
- **Layer ablation** — `FULL_ATTENTION_LAYER_IDX = -1` by default. A layer-index vs nDCG@5 curve would justify (or challenge) this choice.

### Backbone Extensions

The pruner is backbone-agnostic. Each backbone requires only:
1. An `image_token_id` constant (see `IMAGE_TOKEN_ID_*` in `adaptive_pruning.py`)
2. A corresponding embedder in `scripts/`

| Backbone | Status | Token ID |
|---|---|---|
| ColPali (PaliGemma-3B) | ✅ working | 257152 |
| ColQwen3.5-0.8B | ✅ working | 248056 |
| ColQwen3VL 2B/4B | ✅ working | 151655 |
| Vintern-Embedding-1B | contribution welcome | — |

### Evaluation

Run the full ViDoRe benchmark with pruning:

```bash
# Edit keep-ratio mode/tau in configs/train_config.yaml, then:
python evaluate/run_vidore.py --pruner perplexity --tau 2.0
```

Key metrics to report: **nDCG@5**, **Recall@1**, **Recall@5**, **KB/page**.

### LOO Rank-Correlation Experiment

To validate that attention scores from the 1024-dim last layer are good proxies for MaxSim patch contribution (no extra forward passes needed):

```python
# For each (doc, query) pair:
# 1. S = Q @ E.T                                  # [Q_len, N]
# 2. LOO[p] = maxsim_full - maxsim(S[:, p!=i])   # contribution of patch p
# 3. Spearman ρ(attention_scores, LOO)            # > 0.6 = proxy is valid
```

### Citation

If you use or extend this work, please cite the original ColPali paper:

```bibtex
@article{faysse2024colpali,
  title   = {ColPali: Efficient Document Retrieval with Vision Language Models},
  author  = {Faysse, Manuel and others},
  journal = {arXiv preprint arXiv:2407.01449},
  year    = {2024}
}
```
