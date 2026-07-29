# MMDocIR Phi3 5% Pruned: Ranking Error Patterns

Source: final ranking JSONL, cross-checked with checkpoint-250. Both have 1,658 queries.

## Rank Buckets

| Bucket | Queries | Share |
| --- | ---: | ---: |
| Hit@1 | 988 | 59.6% |
| Hit@2-3 only | 355 | 21.4% |
| Hit@4-5 only | 99 | 6.0% |
| Miss@5 | 216 | 13.0% |

There are 454 late-hit queries (ranks 2-5), versus 216 miss@5 queries. The main weakness is
top-rank discrimination, not only complete retrieval failure. Across the two distinct checkpoints,
435 queries remain late hits and 209 remain miss@5.

## Question Type Breakdown

| Type | Queries | R@1 | R@3 | R@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| text-only | 383 | 68.93 | 87.47 | 91.91 | 81.75 |
| multimodal-t | 240 | 56.25 | 82.50 | 90.42 | 75.00 |
| Figure | 170 | 45.11 | 68.93 | 76.63 | 67.30 |
| meta-data | 153 | 33.33 | 56.64 | 65.09 | 51.10 |
| Table | 140 | 39.88 | 61.85 | 74.40 | 62.38 |
| Chart | 101 | 62.57 | 85.45 | 89.36 | 82.31 |
| Generalized-text/Layout | 30 | 37.61 | 58.67 | 61.67 | 56.79 |

## Multi-Page Ground Truth

| Relevant pages | Queries | R@1 | R@3 | R@5 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,346 | 59.58 | 80.24 | 86.18 |
| 2 | 230 | 31.30 | 61.96 | 74.78 |
| 3 | 43 | 17.83 | 50.39 | 63.57 |
| 4+ | 39 | 10.85 | 30.77 | 49.62 |

## Near-Miss Evidence

| Error bucket | Queries | Top-1 adjacent to GT | Top-1 within 3 pages of GT |
| --- | ---: | ---: | ---: |
| Hit@2-3 only | 355 | 32.4% | 53.8% |
| Hit@4-5 only | 99 | 16.2% | 37.4% |
| Miss@5 | 216 | 11.6% | 27.8% |

Late hits are frequently neighboring pages from the same section or continuation. Their mean score
gap between incorrect top-1 and best relevant page is 0.55 (hit@2-3) and 0.98 (hit@4-5); it is
1.37 for misses whose relevant page still occurs in top-10.

## Actionable Insight

Prioritize same-document and adjacent-page hard negatives, then compare pruning against no-pruning
on the 454 late-hit queries. The strongest failures are metadata, tables, figures, and layout
questions, which need fine local visual grounding rather than generic topical matching.
