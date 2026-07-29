import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_mmdocir_runs import merge_reports


class TestMergeMMDocIRRuns(unittest.TestCase):
    def test_uses_every_run_when_averaging(self):
        reports = [
            {"by_domain": {"News": {"r1": 0.4, "ndcg5": 0.5}}, "macro": {"r1": 0.4, "ndcg5": 0.5}, "micro": {"r1": 0.4, "ndcg5": 0.5}},
            {"by_domain": {"News": {"r1": 0.6, "ndcg5": 0.7}}, "macro": {"r1": 0.6, "ndcg5": 0.7}, "micro": {"r1": 0.6, "ndcg5": 0.7}},
            {"by_domain": {"News": {"r1": 0.4, "ndcg5": 0.5}}, "macro": {"r1": 0.4, "ndcg5": 0.5}, "micro": {"r1": 0.4, "ndcg5": 0.5}},
        ]

        merged = merge_reports(reports)

        self.assertAlmostEqual(merged["micro"]["r1"]["mean"], 0.4666666667)
        self.assertGreater(merged["by_domain"]["News"]["r1"]["sd"], 0.0)
