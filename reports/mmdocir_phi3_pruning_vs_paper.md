# MMDocIR Phi3 Pruning vs Paper

Queries: 1658. Metrics are recomputed from saved top-10 rankings with official Recall@k.

## Recall@1

| Domain | Paper Col-Phi3 | Phi3 pruning | Delta |
| --- | ---: | ---: | ---: |
| Research Report | 56.7 | 57.21 | +0.51 |
| Administration & Industry | 50.4 | 47.14 | -3.26 |
| Tutorial & Workshop | 56.9 | 55.64 | -1.26 |
| Academic Paper | 61.3 | 49.04 | -12.26 |
| Brochure | 54.8 | 50.66 | -4.14 |
| Financial Report | 50.7 | 47.02 | -3.68 |
| Guidebook | 60.8 | 51.99 | -8.81 |
| Government | 61.3 | 65.17 | +3.87 |
| Laws | 63.6 | 67.80 | +4.20 |
| News | 54.0 | 56.93 | +2.93 |
| Average Macro | 57.0 | 54.86 | -2.14 |
| Average Micro | 57.1 | 53.43 | -3.67 |

## Recall@3

| Domain | Paper Col-Phi3 | Phi3 pruning | Delta |
| --- | ---: | ---: | ---: |
| Research Report | 80.2 | 76.64 | -3.56 |
| Administration & Industry | 74.1 | 71.82 | -2.28 |
| Tutorial & Workshop | 77.4 | 75.66 | -1.74 |
| Academic Paper | 84.8 | 81.23 | -3.57 |
| Brochure | 69.1 | 71.16 | +2.06 |
| Financial Report | 67.7 | 67.51 | -0.19 |
| Guidebook | 78.7 | 74.99 | -3.71 |
| Government | 79.5 | 76.80 | -2.70 |
| Laws | 81.8 | 90.15 | +8.35 |
| News | 69.3 | 70.07 | +0.77 |
| Average Macro | 76.3 | 75.60 | -0.70 |
| Average Micro | 76.8 | 75.78 | -1.02 |

## Recall@5

| Domain | Paper Col-Phi3 | Phi3 pruning | Delta |
| --- | ---: | ---: | ---: |
| Research Report | 86.3 | 83.71 | -2.59 |
| Administration & Industry | 78.8 | 77.41 | -1.39 |
| Tutorial & Workshop | 81.2 | 81.70 | +0.50 |
| Academic Paper | 92.4 | 89.88 | -2.52 |
| Brochure | 79.0 | 80.37 | +1.37 |
| Financial Report | 73.8 | 77.57 | +3.77 |
| Guidebook | 85.3 | 80.16 | -5.14 |
| Government | 85.1 | 81.31 | -3.79 |
| Laws | 87.1 | 93.94 | +6.84 |
| News | 73.0 | 75.91 | +2.91 |
| Average Macro | 82.2 | 82.20 | -0.00 |
| Average Micro | 83.0 | 83.15 | +0.15 |
