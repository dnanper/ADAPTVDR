import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_mmdocir_jsonl import ndcg_table, summarize_rows


class TestSummarizeMMDocIRJsonl(unittest.TestCase):
    def test_uses_official_page_recall_and_domain_macro(self):
        rows = [
            {
                "domain": "Research report / Introduction",
                "relevant_pages": [1, 2],
                "top_pages": [{"page": 1}, {"page": 9}, {"page": 8}],
            },
            {
                "domain": "News",
                "relevant_pages": [7],
                "top_pages": [{"page": 3}, {"page": 7}],
            },
        ]

        report = summarize_rows(rows, ks=(1, 3))

        self.assertEqual(report["by_domain"]["Research report / Introduction"]["r1"], 0.5)
        self.assertEqual(report["by_domain"]["News"]["r1"], 0.0)
        self.assertEqual(report["micro"]["r3"], 0.75)
        self.assertEqual(report["macro"]["r1"], 0.25)
        self.assertIn("nDCG@5", ndcg_table(report))
