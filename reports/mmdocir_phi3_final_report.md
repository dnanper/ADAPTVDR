# MMDocIR Phi3 Final Evaluation

Source logs: `mmdocir_phi3_final.jsonl` and `mmdocir_phi3_final_pruned_samples.jsonl`.

Protocol: page-level, within-document retrieval; 1,658 queries, no skipped queries. Recall@k is `|top-k intersect ground-truth pages| / |ground-truth pages|`, matching the official MMDocIR evaluator. Macro is the unweighted mean over 10 domains; micro is the query-weighted mean. Values are percentages.

## No Pruning

| Domain | Queries | Recall@1 | Recall@3 | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Research Report | 194 | 55.40 | 76.90 | 84.03 | 77.78 |
| Administration & Industry | 56 | 45.36 | 69.49 | 77.41 | 66.86 |
| Tutorial & Workshop | 104 | 54.44 | 76.14 | 80.66 | 75.18 |
| Academic Paper | 389 | 48.92 | 81.31 | 90.12 | 73.28 |
| Brochure | 76 | 50.00 | 68.86 | 80.37 | 70.29 |
| Financial Report | 344 | 48.76 | 67.37 | 77.18 | 64.94 |
| Guidebook | 115 | 49.81 | 75.42 | 80.42 | 71.40 |
| Government | 111 | 64.26 | 75.00 | 78.60 | 72.41 |
| Laws | 132 | 70.08 | 89.39 | 93.94 | 83.54 |
| News | 137 | 56.93 | 70.80 | 74.45 | 66.36 |
| Average Macro | - | 54.40 | 75.07 | 81.72 | 72.20 |
| Average Micro | 1,658 | 53.36 | 75.55 | 82.82 | 71.90 |

## Adaptive Pruning

Pruning configuration: linear, `r_min=0.3`, `r_max=0.9`. Mean keep ratio: 86.19%.

| Domain | Queries | Recall@1 | Recall@3 | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Research Report | 194 | 57.21 | 76.64 | 83.71 | 78.22 |
| Administration & Industry | 56 | 47.14 | 71.82 | 77.41 | 67.81 |
| Tutorial & Workshop | 104 | 55.64 | 75.66 | 81.70 | 76.32 |
| Academic Paper | 389 | 49.04 | 81.23 | 89.88 | 73.33 |
| Brochure | 76 | 50.66 | 71.16 | 80.37 | 70.91 |
| Financial Report | 344 | 47.02 | 67.51 | 77.57 | 64.55 |
| Guidebook | 115 | 51.99 | 74.99 | 80.16 | 72.31 |
| Government | 111 | 65.17 | 76.80 | 81.31 | 74.23 |
| Laws | 132 | 67.80 | 90.15 | 93.94 | 82.56 |
| News | 137 | 56.93 | 70.07 | 75.91 | 66.97 |
| Average Macro | - | 54.86 | 75.60 | 82.20 | 72.72 |
| Average Micro | 1,658 | 53.43 | 75.78 | 83.15 | 72.17 |

## Pruning Delta

| Aggregate | Delta Recall@1 | Delta Recall@3 | Delta Recall@5 | Delta nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Macro | +0.46 | +0.53 | +0.48 | +0.52 |
| Micro | +0.07 | +0.23 | +0.33 | +0.27 |

Adaptive pruning retains 86.19% of patches and improves all aggregate metrics in this run. The comparison is against the no-pruning sample log generated from the same trained Phi3 adapter and evaluation data.
