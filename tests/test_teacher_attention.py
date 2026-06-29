import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.precompute_teacher_attn import _prepare_output_dir, _save_batch_shard
from train.teacher_attention import TeacherAttentionCache, attention_alignment_loss


class TestTeacherAttention(unittest.TestCase):
    def test_cache_loads_optional_image_grids(self):
        with TemporaryDirectory() as tmpdir:
            batch_dir = _prepare_output_dir(Path(tmpdir) / "attn_cache.pt")
            torch.save({"prompt_mode": "query_image"}, batch_dir / "metadata.pt")
            _save_batch_shard(
                {"id-1": torch.tensor([1.0, 2.0], dtype=torch.float16)},
                batch_dir=batch_dir,
                batch_index=0,
                batch_grids={"id-1": torch.tensor([1, 1, 2])},
            )

            cache = TeacherAttentionCache(str(batch_dir))

            self.assertTrue(torch.equal(cache.get_many(["id-1"])[0], torch.tensor([1.0, 2.0])))
            self.assertTrue(torch.equal(cache.get_many_grids(["id-1"])[0], torch.tensor([1, 1, 2])))

    def test_alignment_downsamples_teacher_grid_to_student_grid(self):
        teacher = torch.arange(1, 17, dtype=torch.float32)
        student = torch.tensor([6.0, 8.0, 14.0, 16.0], dtype=torch.float32)

        loss, matched = attention_alignment_loss(
            student_scores=[student],
            teacher_scores=[teacher],
            student_grids=[torch.tensor([1, 2, 2])],
            teacher_grids=[torch.tensor([1, 4, 4])],
            loss_type="kl",
        )

        self.assertEqual(matched, 1)
        self.assertLess(float(loss), 1e-6)

    def test_alignment_skips_mismatched_scores_without_grids(self):
        loss, matched = attention_alignment_loss(
            student_scores=[torch.ones(4)],
            teacher_scores=[torch.ones(16)],
            loss_type="kl",
        )

        self.assertEqual(matched, 0)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
