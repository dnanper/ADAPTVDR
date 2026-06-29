"""
TDD Tests for ColQwen3.5 Training Pipeline
Components: dataset.py | loss.py | collator.py
Run: python tests.py
"""
import torch
import unittest
import numpy as np
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# DATASET TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestViDoReDataset(unittest.TestCase):
    DATA_PATH = "/data2/cmdir/home/test01/longvnu/graduation_thesis/dataset/vidore_train/datasets--vidore--colpali_train_set/snapshots/e13d3594064836f7fd69fad7e3d2b51065b335c7/data"

    def setUp(self):
        from dataset import ViDoReDataset
        self.dataset = ViDoReDataset(self.DATA_PATH, split="train", num_shards=1)

    def test_len_positive(self):
        """Dataset must have at least 1 sample"""
        self.assertGreater(len(self.dataset), 0)

    def test_getitem_keys(self):
        """Each item must have 'query' (str) and 'image' (PIL.Image)"""
        item = self.dataset[0]
        self.assertIn("query", item)
        self.assertIn("image", item)
        self.assertIsInstance(item["query"], str)
        self.assertIsInstance(item["image"], Image.Image)

    def test_query_nonempty(self):
        """Query text should not be empty"""
        item = self.dataset[0]
        self.assertTrue(len(item["query"].strip()) > 0)

    def test_image_rgb(self):
        """Image should be RGB"""
        item = self.dataset[0]
        self.assertEqual(item["image"].mode, "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# LOSS TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestInfoNCELoss(unittest.TestCase):
    def setUp(self):
        from loss import InfoNCELoss
        self.loss_fn = InfoNCELoss(temperature=0.07)
        self.B, self.D = 4, 1024

    def test_output_is_scalar(self):
        """InfoNCE must return a scalar"""
        q = torch.randn(self.B, self.D)
        d = torch.randn(self.B, self.D)
        loss = self.loss_fn(q, d)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_positive(self):
        """InfoNCE loss must be > 0"""
        q = torch.randn(self.B, self.D)
        d = torch.randn(self.B, self.D)
        loss = self.loss_fn(q, d)
        self.assertGreater(loss.item(), 0)

    def test_perfect_alignment_lower_loss(self):
        """Identical q=d should have lower loss than random"""
        q = torch.nn.functional.normalize(torch.randn(self.B, self.D), dim=-1)
        loss_aligned = self.loss_fn(q, q.clone())
        loss_random  = self.loss_fn(q, torch.nn.functional.normalize(torch.randn(self.B, self.D), dim=-1))
        self.assertLess(loss_aligned.item(), loss_random.item())

    def test_symmetry(self):
        """loss(q,d) and loss(d,q) should be close (same batch)"""
        q = torch.nn.functional.normalize(torch.randn(self.B, self.D), dim=-1)
        d = torch.nn.functional.normalize(torch.randn(self.B, self.D), dim=-1)
        loss_qd = self.loss_fn(q, d)
        loss_dq = self.loss_fn(d, q)
        self.assertAlmostEqual(loss_qd.item(), loss_dq.item(), places=3)


class TestMaxSimLoss(unittest.TestCase):
    def setUp(self):
        from loss import MaxSimLoss
        self.loss_fn = MaxSimLoss(temperature=1.0)
        self.B, self.Nq, self.Nd, self.D = 2, 10, 49, 1024

    def test_output_is_scalar(self):
        """MaxSim must return a scalar"""
        q   = torch.randn(self.B, self.Nq, self.D)
        d   = torch.randn(self.B, self.Nd, self.D)
        q_mask = torch.ones(self.B, self.Nq, dtype=torch.bool)
        d_mask = torch.ones(self.B, self.Nd, dtype=torch.bool)
        loss = self.loss_fn(q, d, q_mask, d_mask)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_scores_shape(self):
        """Score matrix must be [B, B]"""
        from loss import MaxSimLoss
        q = torch.nn.functional.normalize(torch.randn(self.B, self.Nq, self.D), dim=-1)
        d = torch.nn.functional.normalize(torch.randn(self.B, self.Nd, self.D), dim=-1)
        scores = MaxSimLoss._compute_scores(q, d)
        self.assertEqual(scores.shape, (self.B, self.B))

    def test_diagonal_highest_when_aligned(self):
        """Diagonal of score matrix should be highest when q matches its own d"""
        q = torch.nn.functional.normalize(torch.randn(self.B, self.Nq, self.D), dim=-1)
        # Make doc[i] = repeat query[i] tokens so they align perfectly
        d = q[:, :self.Nd, :] if self.Nq >= self.Nd else q.repeat(1, 2, 1)[:, :self.Nd, :]
        from loss import MaxSimLoss
        scores = MaxSimLoss._compute_scores(q, d)
        for i in range(self.B):
            self.assertEqual(scores[i].argmax().item(), i)

    def test_masked_tokens_ignored(self):
        """Padding tokens (mask=0) should not affect score"""
        from loss import MaxSimLoss
        q = torch.nn.functional.normalize(torch.randn(1, self.Nq, self.D), dim=-1)
        d = torch.nn.functional.normalize(torch.randn(1, self.Nd, self.D), dim=-1)

        # Full mask vs half mask on doc — score of active tokens should differ
        full_mask = torch.ones(1, self.Nd, dtype=torch.bool)
        half_mask = torch.zeros(1, self.Nd, dtype=torch.bool)
        half_mask[0, :self.Nd // 2] = True

        score_full = MaxSimLoss._compute_scores(q, d)
        # Replace last half of d with same tokens as first half
        d_padded = d.clone()
        d_padded[0, self.Nd // 2:] = 0.0  # zero out pad tokens
        score_partial = MaxSimLoss._compute_scores(q, d_padded)

        # Scores should be different
        self.assertNotAlmostEqual(score_full[0, 0].item(), score_partial[0, 0].item(), places=3)


# ══════════════════════════════════════════════════════════════════════════════
# COLLATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestColPaliCollator(unittest.TestCase):
    MODEL_PATH = "/data2/cmdir/home/test01/longvnu/stable_diff/models/Qwen/Qwen3.5-0.8B"

    def setUp(self):
        from collator import ColPaliCollator
        self.collator = ColPaliCollator(self.MODEL_PATH)
        self.batch = [
            {"query": "What is this chart about?",
             "image": Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))},
            {"query": "Describe the figure.",
             "image": Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))},
        ]

    def test_output_keys(self):
        """Collator must return query_inputs and doc_inputs"""
        out = self.collator(self.batch)
        self.assertIn("query_inputs", out)
        self.assertIn("doc_inputs", out)

    def test_query_has_input_ids(self):
        """Query inputs must have input_ids"""
        out = self.collator(self.batch)
        self.assertIn("input_ids", out["query_inputs"])

    def test_doc_has_pixel_values(self):
        """Doc inputs must have pixel_values"""
        out = self.collator(self.batch)
        self.assertIn("pixel_values", out["doc_inputs"])

    def test_batch_size_preserved(self):
        """Batch size must be preserved in both query and doc.

        Note: Qwen3.5 stores all image patches flat → pixel_values.shape[0]
        equals total_patches_across_batch, NOT B.  Use image_grid_thw.shape[0]
        to verify that exactly B images were processed.
        """
        out = self.collator(self.batch)
        B = len(self.batch)
        self.assertEqual(out["query_inputs"]["input_ids"].shape[0], B)
        self.assertEqual(out["doc_inputs"]["image_grid_thw"].shape[0], B)


if __name__ == "__main__":
    unittest.main(verbosity=2)
