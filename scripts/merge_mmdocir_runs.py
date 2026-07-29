"""Aggregate repeated MMDocIR JSONL evaluations into one paper-style report."""

import argparse
import math
import statistics
from pathlib import Path

try:
    from scripts.summarize_mmdocir_jsonl import (
        DOMAIN_ORDER,
        PAPER_COLPHI3,
        PAPER_LABEL_BY_DOMAIN,
        enrich_rows,
        read_jsonl,
        summarize_rows,
    )
except ModuleNotFoundError:
    from summarize_mmdocir_jsonl import (
        DOMAIN_ORDER,
        PAPER_COLPHI3,
        PAPER_LABEL_BY_DOMAIN,
        enrich_rows,
        read_jsonl,
        summarize_rows,
    )


def _summary(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    t95 = 2.776 if len(values) == 5 else 1.96
    return {"mean": mean, "sd": sd, "ci95": t95 * sd / math.sqrt(len(values))}


def merge_reports(reports):
    metrics = tuple(reports[0]["micro"])
    domains = tuple(reports[0]["by_domain"])
    return {
        "runs": len(reports),
        "by_domain": {
            domain: {metric: _summary([report["by_domain"][domain][metric] for report in reports]) for metric in metrics}
            for domain in domains
        },
        "macro": {metric: _summary([report["macro"][metric] for report in reports]) for metric in metrics},
        "micro": {metric: _summary([report["micro"][metric] for report in reports]) for metric in metrics},
    }


def _value(summary):
    return f"{summary['mean'] * 100:.2f} +/- {summary['sd'] * 100:.2f}"


def _recall_table(metric, report):
    paper = PAPER_COLPHI3[metric]
    lines = ["| Domain | Paper Col-Phi3 | Phi3 mean +/- SD | Delta |", "| --- | ---: | ---: | ---: |"]
    for index, domain in enumerate(DOMAIN_ORDER):
        summary = report["by_domain"][domain][metric]
        lines.append(
            f"| {PAPER_LABEL_BY_DOMAIN[domain]} | {paper[index]:.1f} | {_value(summary)} | {summary['mean'] * 100 - paper[index]:+.2f} |"
        )
    for index, aggregate in ((10, "macro"), (11, "micro")):
        summary = report[aggregate][metric]
        label = "Average Macro" if aggregate == "macro" else "Average Micro"
        lines.append(f"| {label} | {paper[index]:.1f} | {_value(summary)} | {summary['mean'] * 100 - paper[index]:+.2f} |")
    return "\n".join(lines)


def _ndcg_table(report):
    lines = ["| Domain | Phi3 mean +/- SD nDCG@5 |", "| --- | ---: |"]
    for domain in DOMAIN_ORDER:
        lines.append(f"| {PAPER_LABEL_BY_DOMAIN[domain]} | {_value(report['by_domain'][domain]['ndcg5'])} |")
    lines.append(f"| Average Macro | {_value(report['macro']['ndcg5'])} |")
    lines.append(f"| Average Micro | {_value(report['micro']['ndcg5'])} |")
    return "\n".join(lines)


def markdown_report(report, paths):
    parts = [
        "# MMDocIR Phi3 Pruning: Five-Run Aggregate",
        "",
        f"Runs: {report['runs']}; 1,658 queries per run. Values are mean +/- sample SD across runs.",
        "95% CI of the mean is available from the underlying calculation but is not the +/- value in this report.",
        "",
        "## Inputs",
        "",
        *[f"- `{path}`" for path in paths],
    ]
    for metric, title in (("r1", "Recall@1"), ("r3", "Recall@3"), ("r5", "Recall@5")):
        parts.extend(["", f"## {title}", "", _recall_table(metric, report)])
    parts.extend(["", "## nDCG@5", "", "Table 5 of the MMDocIR paper does not report nDCG@5.", "", _ndcg_table(report)])
    parts.extend(
        [
            "",
            "## Significance Boundary",
            "",
            "A p-value against the paper cannot be computed from Table 5 alone because it lacks per-query paired outcomes.",
        ]
    )
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-log", action="append", required=True, help="Repeat once per independent run.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    annotations = read_jsonl(args.annotations)
    reports = [summarize_rows(enrich_rows(read_jsonl(path), annotations)) for path in args.sample_log]
    merged = merge_reports(reports)
    text = markdown_report(merged, args.sample_log)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
