"""Re-score saved MMDocIR rankings with the official page-level metrics."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DOMAIN_ORDER = (
    "Research report / Introduction",
    "Administration/Industry file",
    "Tutorial/Workshop",
    "Academic paper",
    "Brochure",
    "Financial report",
    "Guidebook",
    "Government",
    "Laws",
    "News",
)
PAPER_LABELS = (
    "Research Report",
    "Administration & Industry",
    "Tutorial & Workshop",
    "Academic Paper",
    "Brochure",
    "Financial Report",
    "Guidebook",
    "Government",
    "Laws",
    "News",
)
PAPER_COLPHI3 = {
    "r1": [56.7, 50.4, 56.9, 61.3, 54.8, 50.7, 60.8, 61.3, 63.6, 54.0, 57.0, 57.1],
    "r3": [80.2, 74.1, 77.4, 84.8, 69.1, 67.7, 78.7, 79.5, 81.8, 69.3, 76.3, 76.8],
    "r5": [86.3, 78.8, 81.2, 92.4, 79.0, 73.8, 85.3, 85.1, 87.1, 73.0, 82.2, 83.0],
}


def recall_at_k(ranked_pages, relevant_pages, k):
    return len(set(ranked_pages[:k]) & set(relevant_pages)) / len(relevant_pages) if relevant_pages else 0.0


def ndcg_at_k(ranked_pages, relevant_pages, k):
    dcg = sum(1 / math.log2(rank + 1) for rank, page in enumerate(ranked_pages[:k], 1) if page in relevant_pages)
    ideal = min(len(relevant_pages), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def summarize_rows(rows, ks=(1, 3, 5)):
    by_domain = defaultdict(lambda: defaultdict(list))
    all_metrics = defaultdict(list)
    for row in rows:
        relevant = set(row["relevant_pages"])
        ranked = [item["page"] for item in row["top_pages"]]
        values = {f"r{k}": recall_at_k(ranked, relevant, k) for k in ks}
        values["ndcg5"] = ndcg_at_k(ranked, relevant, 5)
        for metric, value in values.items():
            by_domain[row["domain"]][metric].append(value)
            all_metrics[metric].append(value)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    domain_summary = {
        domain: {metric: mean(values) for metric, values in metrics.items()} | {"queries": len(next(iter(metrics.values())))}
        for domain, metrics in by_domain.items()
    }
    metric_names = tuple(all_metrics)
    return {
        "by_domain": domain_summary,
        "macro": {metric: mean([domain_summary[domain][metric] for domain in domain_summary]) for metric in metric_names},
        "micro": {metric: mean(values) for metric, values in all_metrics.items()},
        "queries": len(rows),
    }


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_query(query):
    return " ".join(query.split())


def annotation_index(annotations):
    index = {}
    for record in annotations:
        for question in record["questions"]:
            key = (normalized_query(question["Q"]), record["doc_name"], tuple(sorted(question["page_id"])))
            index[key] = record["domain"]
    return index


def enrich_rows(sample_rows, annotations):
    index = annotation_index(annotations)
    rows = []
    missing = []
    for row in sample_rows:
        key = (normalized_query(row["query"]), row["doc_name"], tuple(sorted(row["relevant_pages"])))
        domain = index.get(key)
        if domain is None:
            missing.append(key)
            continue
        rows.append(row | {"domain": domain})
    if missing:
        raise ValueError(f"Could not match {len(missing)} sample-log rows to annotations")
    return rows


def table_for_metric(metric, report):
    paper = PAPER_COLPHI3[metric]
    lines = [f"| Domain | Paper Col-Phi3 | Phi3 pruning | Delta |", "| --- | ---: | ---: | ---: |"]
    for index, (domain, label) in enumerate(zip(DOMAIN_ORDER, PAPER_LABELS)):
        value = report["by_domain"][domain][metric] * 100
        lines.append(f"| {label} | {paper[index]:.1f} | {value:.2f} | {value - paper[index]:+.2f} |")
    for index, name in ((10, "Average Macro"), (11, "Average Micro")):
        value = report["macro" if index == 10 else "micro"][metric] * 100
        lines.append(f"| {name} | {paper[index]:.1f} | {value:.2f} | {value - paper[index]:+.2f} |")
    return "\n".join(lines)


def markdown_report(report):
    parts = [
        "# MMDocIR Phi3 Pruning vs Paper",
        "",
        f"Queries: {report['queries']}. Metrics are recomputed from saved top-10 rankings with official Recall@k.",
    ]
    for metric, title in (("r1", "Recall@1"), ("r3", "Recall@3"), ("r5", "Recall@5")):
        parts.extend(["", f"## {title}", "", table_for_metric(metric, report)])
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-log", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = summarize_rows(enrich_rows(read_jsonl(args.sample_log), read_jsonl(args.annotations)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), end="")


if __name__ == "__main__":
    main()
