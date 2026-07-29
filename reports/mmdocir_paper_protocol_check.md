# MMDocIR Paper Protocol Check

Primary sources: [MMDocIR paper](https://aclanthology.org/2025.emnlp-main.1576.pdf) and the authors' [official evaluator](https://github.com/MMDocRAG/MMDocIR/blob/main/metric_eval.py).

## What Table 5 Measures

- Page-level evaluation has 1,658 queries from 313 documents across 10 domains. The paper figure states these counts; the current local annotations also normalize to 1,658 questions.
- Each query ranks only pages from its source document. The official `search.py` slices page embeddings using that query's document page range before scoring.
- Official page Recall@k is `|top-k pages intersect ground-truth pages| / |ground-truth pages|`, not a binary hit rate. See `recall()` in `metric_eval.py`.
- Macro is the mean of the ten per-domain scores. Micro is the mean across all queries. See `evaluate_page()` in `metric_eval.py`.

## Current Phi3 Evaluator

- Candidate scope matches the official page task: it resolves the query document then ranks all pages in that document.
- The data count matches: 1,658 query records and ten domains.
- Before 2026-07-15, `evaluate_mmdocir_phi3.py` used a binary any-positive hit rate. Its console Recall values therefore could not be compared to Table 5.
- The evaluator now uses the official recall fraction. Existing JSONL top-10 rankings were sufficient to recompute Recall@1/3/5 without re-running inference.

## Comparison Boundary

The corrected Phi3 numbers may be compared to Table 5 for the page-level metric and aggregation protocol. They remain a separate experimental result: Phi3 base model, training recipe, image preprocessing, and adaptive pruning differ from the paper's published Col-Phi3 configuration.
