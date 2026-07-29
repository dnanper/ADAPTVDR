# MMDocIR Phi3 Pruning: Five-Run Aggregate

Runs: 5; 1,658 queries per run. Values are mean +/- sample SD across runs.
95% CI of the mean is available from the underlying calculation but is not the +/- value in this report.

## Inputs

- `mmdocir_phi3_5p_pruned_250_samples.jsonl`
- `mmdocir_phi3_5p_pruned_250_samples_3.jsonl`
- `mmdocir_phi3_5p_pruned_2_samples.jsonl`
- `mmdocir_phi3_5p_pruned_250_samples_2.jsonl`
- `mmdocir_phi3_final_pruned_samples.jsonl`

## Recall@1

| Domain | Paper Col-Phi3 | Phi3 mean +/- SD | Delta |
| --- | ---: | ---: | ---: |
| Research Report | 56.7 | 56.74 +/- 0.42 | +0.04 |
| Administration & Industry | 50.4 | 47.68 +/- 0.49 | -2.72 |
| Tutorial & Workshop | 56.9 | 55.93 +/- 0.26 | -0.97 |
| Academic Paper | 61.3 | 49.35 +/- 0.28 | -11.95 |
| Brochure | 54.8 | 49.87 +/- 0.72 | -4.93 |
| Financial Report | 50.7 | 47.46 +/- 0.40 | -3.24 |
| Guidebook | 60.8 | 51.20 +/- 0.71 | -9.60 |
| Government | 61.3 | 64.08 +/- 0.99 | +2.78 |
| Laws | 63.6 | 65.98 +/- 1.66 | +2.38 |
| News | 54.0 | 56.50 +/- 0.40 | +2.50 |
| Average Macro | 57.0 | 54.48 +/- 0.35 | -2.52 |
| Average Micro | 57.1 | 53.23 +/- 0.18 | -3.87 |

## Recall@3

| Domain | Paper Col-Phi3 | Phi3 mean +/- SD | Delta |
| --- | ---: | ---: | ---: |
| Research Report | 80.2 | 76.33 +/- 0.28 | -3.87 |
| Administration & Industry | 74.1 | 72.08 +/- 0.24 | -2.02 |
| Tutorial & Workshop | 77.4 | 76.04 +/- 0.35 | -1.36 |
| Academic Paper | 84.8 | 80.98 +/- 0.23 | -3.82 |
| Brochure | 69.1 | 70.96 +/- 0.18 | +1.86 |
| Financial Report | 67.7 | 67.08 +/- 0.40 | -0.62 |
| Guidebook | 78.7 | 74.99 +/- 0.00 | -3.71 |
| Government | 79.5 | 76.26 +/- 0.49 | -3.24 |
| Laws | 81.8 | 90.61 +/- 0.41 | +8.81 |
| News | 69.3 | 70.07 +/- 0.00 | +0.77 |
| Average Macro | 76.3 | 75.54 +/- 0.06 | -0.76 |
| Average Micro | 76.8 | 75.61 +/- 0.15 | -1.19 |

## Recall@5

| Domain | Paper Col-Phi3 | Phi3 mean +/- SD | Delta |
| --- | ---: | ---: | ---: |
| Research Report | 86.3 | 83.88 +/- 0.15 | -2.42 |
| Administration & Industry | 78.8 | 77.95 +/- 0.49 | -0.85 |
| Tutorial & Workshop | 81.2 | 81.41 +/- 0.26 | +0.21 |
| Academic Paper | 92.4 | 89.84 +/- 0.03 | -2.56 |
| Brochure | 79.0 | 80.18 +/- 0.18 | +1.18 |
| Financial Report | 73.8 | 77.48 +/- 0.08 | +3.68 |
| Guidebook | 85.3 | 80.16 +/- 0.00 | -5.14 |
| Government | 85.1 | 81.85 +/- 0.49 | -3.25 |
| Laws | 87.1 | 94.39 +/- 0.41 | +7.29 |
| News | 73.0 | 75.91 +/- 0.00 | +2.91 |
| Average Macro | 82.2 | 82.30 +/- 0.10 | +0.10 |
| Average Micro | 83.0 | 83.21 +/- 0.05 | +0.21 |

## nDCG@5

Table 5 of the MMDocIR paper does not report nDCG@5.

| Domain | Phi3 mean +/- SD nDCG@5 |
| --- | ---: |
| Research Report | 78.12 +/- 0.09 |
| Administration & Industry | 68.14 +/- 0.30 |
| Tutorial & Workshop | 76.42 +/- 0.08 |
| Academic Paper | 73.40 +/- 0.07 |
| Brochure | 70.63 +/- 0.26 |
| Financial Report | 64.61 +/- 0.05 |
| Guidebook | 72.02 +/- 0.26 |
| Government | 73.98 +/- 0.23 |
| Laws | 82.11 +/- 0.41 |
| News | 66.83 +/- 0.13 |
| Average Macro | 72.63 +/- 0.09 |
| Average Micro | 72.11 +/- 0.06 |

## Significance Boundary

A p-value against the paper cannot be computed from Table 5 alone because it lacks per-query paired outcomes.
