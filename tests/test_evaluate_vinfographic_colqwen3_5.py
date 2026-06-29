import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate.evaluate_vinfographic_colqwen3_5 import (
    build_prediction_rows,
    compute_split_metrics,
    load_vinfographic_split,
)


class TestLoadViInfographicSplit(unittest.TestCase):
    def test_loads_single_and_multi_into_shared_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            single_samples = [
                {
                    "question_id": "s1",
                    "question": "single question",
                    "image_path": "images/1.jpg",
                }
            ]
            multi_samples = [
                {
                    "question_id": "m1",
                    "question": "multi question",
                    "image_paths": ["images/1.jpg", "images/2.jpg"],
                }
            ]

            (data_dir / "single_test.json").write_text(json.dumps(single_samples), encoding="utf-8")
            (data_dir / "multi_test.json").write_text(json.dumps(multi_samples), encoding="utf-8")

            single = load_vinfographic_split(root, "single_test")
            multi = load_vinfographic_split(root, "multi_test")

            self.assertEqual(single["split"], "single_test")
            self.assertEqual(single["doc_paths"], ["images/1.jpg"])
            self.assertEqual(single["queries"][0]["positive_doc_indices"], {0})

            self.assertEqual(multi["split"], "multi_test")
            self.assertEqual(multi["doc_paths"], ["images/1.jpg", "images/2.jpg"])
            self.assertEqual(multi["queries"][0]["positive_doc_indices"], {0, 1})


class TestComputeSplitMetrics(unittest.TestCase):
    def test_multi_relevant_metrics_are_computed_correctly(self):
        scores = np.array(
            [
                [0.2, 0.9, 0.8],
                [0.7, 0.6, 0.5],
            ],
            dtype=np.float32,
        )
        positive_sets = [{1, 2}, {2}]
        positive_orders = [[1, 2], [2]]

        metrics = compute_split_metrics(scores, positive_sets, positive_orders, ks=(1, 2), ndcg_k=2)

        self.assertAlmostEqual(metrics["recall@1"], 0.25, places=6)
        self.assertAlmostEqual(metrics["recall@2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["hard_recall@1"], 0.0, places=6)
        self.assertAlmostEqual(metrics["hard_recall@2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["ordered_hard_recall@1"], 0.0, places=6)
        self.assertAlmostEqual(metrics["ordered_hard_recall@2"], 0.5, places=6)
        self.assertAlmostEqual(metrics["mrr"], 0.6666666667, places=6)
        self.assertAlmostEqual(metrics["ndcg@2"], 0.5, places=6)

    def test_ordered_hard_requires_correct_relevant_order(self):
        scores = np.array([[0.1, 0.8, 0.9]], dtype=np.float32)
        metrics = compute_split_metrics(scores, [{1, 2}], [[1, 2]], ks=(2,), ndcg_k=2)

        self.assertAlmostEqual(metrics["recall@2"], 1.0, places=6)
        self.assertAlmostEqual(metrics["hard_recall@2"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ordered_hard_recall@2"], 0.0, places=6)


class TestBuildPredictionRows(unittest.TestCase):
    def test_prediction_rows_include_ranks_and_hits(self):
        split_bundle = {
            "split": "multi_test",
            "doc_paths": ["images/1.jpg", "images/2.jpg", "images/3.jpg"],
            "queries": [
                {
                    "question_id": "q1",
                    "question": "question 1",
                    "positive_doc_indices": {1, 2},
                    "answer": "",
                    "answer_source": "",
                    "image_type": "",
                    "element": "",
                }
            ],
        }
        scores = np.array([[0.1, 0.9, 0.8]], dtype=np.float32)

        rows = build_prediction_rows(split_bundle, scores, dim=128, top_k=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question_id"], "q1")
        self.assertEqual(rows[0]["retrieved_doc_path"], "images/2.jpg")
        self.assertEqual(rows[0]["rank"], 1)
        self.assertTrue(rows[0]["is_relevant"])
        self.assertEqual(rows[0]["positive_doc_paths"], "images/2.jpg|images/3.jpg")
        self.assertEqual(rows[1]["retrieved_doc_path"], "images/3.jpg")
        self.assertTrue(rows[1]["is_relevant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
