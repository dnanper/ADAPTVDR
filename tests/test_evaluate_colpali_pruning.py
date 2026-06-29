import unittest

from evaluate.evaluate_colpali_pruning import build_variant_specs


class TestEvaluateColPaliPruning(unittest.TestCase):
    def test_build_variant_specs_matches_requested_matrix(self):
        specs = build_variant_specs([1, 2, 3])

        self.assertEqual(
            [spec["variant"] for spec in specs],
            [
                "adapter",
                "linear_pruning",
                "perplexity_tau1",
                "perplexity_tau2",
                "perplexity_tau3",
            ],
        )
        self.assertEqual(
            [spec["pruning_mode"] for spec in specs],
            ["none", "linear", "perplexity", "perplexity", "perplexity"],
        )
        self.assertEqual(
            [spec["tau"] for spec in specs],
            [None, None, 1.0, 2.0, 3.0],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
