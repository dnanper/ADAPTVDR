import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except ModuleNotFoundError:
    LoraConfig = None
    get_peft_model = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.colphi3_embedding import ColPhi3ForEmbedding, ensure_phi3_img_projection_bias


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        hidden = torch.ones(1, 3, 4)
        return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden, attentions=("attn",))


class _FakeTrainBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_projection = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.q_proj = nn.Linear(4, 4)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = torch.ones(1, 3, 4)
        return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden, attentions=None)


class _FakeSaveWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_module = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.modules_to_save = nn.ModuleDict({"default": nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))})


class TestColPhi3Embedding(unittest.TestCase):
    def test_forward_normalizes_peft_control_kwargs_before_backbone_call(self):
        model = ColPhi3ForEmbedding.__new__(ColPhi3ForEmbedding)
        nn.Module.__init__(model)
        model.backbone = _FakeBackbone()
        model.linear_head = nn.Linear(4, 2, bias=False)

        out = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 0]]),
            output_hidden_states=False,
            output_attentions=True,
            return_dict=False,
        )

        self.assertTrue(torch.equal(out.hidden_states[:, 2], torch.zeros(1, 2)))
        self.assertEqual(out.attentions, ("attn",))
        self.assertTrue(model.backbone.kwargs["output_hidden_states"])
        self.assertTrue(model.backbone.kwargs["return_dict"])

    def test_lora_modules_to_save_does_not_wrap_phi3_img_projection(self):
        if LoraConfig is None or get_peft_model is None:
            self.skipTest("peft is not installed")
        model = ColPhi3ForEmbedding.__new__(ColPhi3ForEmbedding)
        nn.Module.__init__(model)
        model.backbone = _FakeTrainBackbone()
        model.linear_head = nn.Linear(4, 2, bias=False)

        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj"],
                task_type="FEATURE_EXTRACTION",
                modules_to_save=["linear_head"],
            ),
        )

        self.assertIsInstance(peft_model.base_model.model.backbone.img_projection, nn.Sequential)
        self.assertEqual(peft_model.base_model.model.backbone.img_projection[0].bias.shape, torch.Size([4]))

    def test_phi3_img_projection_bias_alias_is_added_to_wrapped_sequential(self):
        root = nn.Module()
        root.vision_embed_tokens = nn.Module()
        root.vision_embed_tokens.img_projection = _FakeSaveWrapper()

        ensure_phi3_img_projection_bias(root)

        wrapped = root.vision_embed_tokens.img_projection
        self.assertEqual(wrapped.original_module.bias.shape, torch.Size([4]))
        self.assertEqual(wrapped.modules_to_save["default"].bias.shape, torch.Size([4]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
