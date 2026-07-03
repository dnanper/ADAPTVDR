import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from train.phi3_collator import Phi3MMDocIRCollator


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

    def __call__(self, *, text, images=None, padding=True, truncation=True, max_length=2048, return_tensors="pt"):
        batch_size = len(text)
        if images is None:
            input_ids = torch.tensor([[10, 20, 21, 0]] * batch_size)
            attention_mask = torch.tensor([[1, 1, 1, 0]] * batch_size)
        else:
            self.images_seen.extend(images)
            input_ids = torch.tensor([[10, 99, 99, 11, 0]] * batch_size)
            attention_mask = torch.tensor([[1, 1, 1, 1, 0]] * batch_size)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestPhi3Integration(unittest.TestCase):
    def test_collator_returns_colpali_style_masks_and_dynamic_resizes_docs(self):
        fake_processor = _FakeProcessor()
        with patch("train.phi3_collator.AutoProcessor.from_pretrained", return_value=fake_processor):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
