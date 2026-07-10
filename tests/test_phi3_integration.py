import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train.phi3_collator import Phi3MMDocIRCollator
from train.teacher_attention import attention_alignment_loss


class _FakeTokenizer:
    image_token_id = 99

    def convert_tokens_to_ids(self, token):
        return 99 if token in {"<|image_1|>", "<|image|>"} else 1

    def encode(self, text, add_special_tokens=False):
        return [7, 8]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_token_id = 99
        self.images_seen = []
        self.use_phi3_hd_layout = False

    def __call__(self, *, text, images=None, padding=True, truncation=True, max_length=2048, return_tensors="pt"):
        batch_size = 1 if isinstance(text, str) else len(text)
        if images is None:
            input_ids = torch.tensor([[10, 20, 21, 0]] * batch_size)
            attention_mask = torch.tensor([[1, 1, 1, 0]] * batch_size)
        else:
            self.images_seen.extend(images)
            if self.use_phi3_hd_layout:
                image_tokens = [-1] * 313
                ids = [10] + image_tokens + [11, 0]
                input_ids = torch.tensor([ids] * batch_size)
                attention_mask = torch.tensor([[1] * (len(ids) - 1) + [0]] * batch_size)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "image_sizes": torch.tensor([[336, 336]] * batch_size),
                }
            input_ids = torch.tensor([[10, 99, 99, 11, 0]] * batch_size)
            attention_mask = torch.tensor([[1, 1, 1, 1, 0]] * batch_size)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestPhi3Integration(unittest.TestCase):
    def test_collator_returns_colpali_style_masks_and_dynamic_resizes_docs(self):
        fake_processor = _FakeProcessor()
        fake_transformers = MagicMock()
        fake_transformers.AutoProcessor.from_pretrained.return_value = fake_processor
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            collator = Phi3MMDocIRCollator(
                "fake-phi3",
                image_size=None,
                min_pixels=4096,
                max_pixels=1_000_000,
            )
            batch = collator([
                {
                    "query": "find revenue",
                    "image": Image.new("RGB", (2000, 1000)),
                    "sample_id": "s1",
                }
            ])

        self.assertTrue(torch.equal(batch["query_token_mask"], torch.tensor([[True, True, True, False]])))
        self.assertTrue(torch.equal(batch["doc_token_mask"], torch.tensor([[False, True, True, False, False]])))
        self.assertLessEqual(fake_processor.images_seen[0].width * fake_processor.images_seen[0].height, 1_000_000)
        self.assertEqual(batch["pos_count"], 1)

    def test_collator_uses_negative_phi3_image_placeholders(self):
        fake_processor = _FakeProcessor()
        fake_processor.use_phi3_hd_layout = True
        fake_transformers = MagicMock()
        fake_transformers.AutoProcessor.from_pretrained.return_value = fake_processor
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            collator = Phi3MMDocIRCollator("fake-phi3", image_size=None)
            batch = collator([
                {
                    "query": "find revenue",
                    "image": Image.new("RGB", (32, 32)),
                    "sample_id": "s1",
                }
            ])

        self.assertEqual(int(batch["doc_token_mask"].sum()), 144)
        self.assertEqual(batch["doc_image_grid_thw"].tolist(), [[1, 12, 12]])

    def test_collator_keeps_positive_doc_inputs_separate_from_hard_negatives(self):
        fake_processor = _FakeProcessor()
        fake_transformers = MagicMock()
        fake_transformers.AutoProcessor.from_pretrained.return_value = fake_processor
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            collator = Phi3MMDocIRCollator("fake-phi3", image_size=None)
            batch = collator([
                {
                    "query": "find revenue",
                    "image": Image.new("RGB", (32, 32)),
                    "hard_neg_images": [
                        Image.new("RGB", (64, 64)),
                        Image.new("RGB", (96, 96)),
                    ],
                    "sample_id": "s1",
                }
            ])

        self.assertEqual(batch["pos_count"], 1)
        self.assertEqual(batch["doc_inputs"]["input_ids"].shape[0], 3)
        self.assertEqual(batch["positive_doc_inputs"]["input_ids"].shape[0], 1)
        self.assertTrue(torch.equal(batch["positive_doc_token_mask"], batch["doc_token_mask"][:1]))

    def test_teacher_alignment_requires_grid_unless_explicit_1d_fallback(self):
        student = [torch.ones(144)]
        teacher = [torch.ones(36)]

        _, matched_without_grid = attention_alignment_loss(student, teacher)
        self.assertEqual(matched_without_grid, 0)

        loss, matched_with_grid = attention_alignment_loss(
            student,
            teacher,
            student_grids=[torch.tensor([1, 12, 12])],
            teacher_grids=[torch.tensor([1, 12, 12])],
        )
        self.assertEqual(matched_with_grid, 1)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
