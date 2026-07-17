import math
import unittest

import torch

from scripts.adaptive_pruning import (
    AdaptivePruner,
    compute_keep_ratio_perplexity,
    extract_image_patch_scores,
)


class TestAdaptivePruning(unittest.TestCase):
    def test_extract_image_patch_scores_excludes_padding_tokens(self):
        image_token_id = 99
        input_ids = torch.tensor([[1, image_token_id, image_token_id, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        attn = torch.zeros(1, 1, 4, 4)
        attn[0, 0, 0, 1] = 0.1
        attn[0, 0, 0, 2] = 0.2
        attn[0, 0, 3, 1] = 0.9
        attn[0, 0, 3, 2] = 0.8

        scores = extract_image_patch_scores(
            attentions=(attn,),
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_id=image_token_id,
        )

        self.assertIsNotNone(scores)
        self.assertTrue(torch.allclose(scores[0], torch.tensor([0.1, 0.2])))

    def test_perplexity_keep_ratio_respects_r_min(self):
        attn_scores = torch.tensor([1.0] + [0.0] * 575)

        keep = compute_keep_ratio_perplexity(attn_scores, tau=2.0, r_min=0.3)

        self.assertTrue(math.isclose(keep, 0.3, rel_tol=0.0, abs_tol=1e-6))

    def test_prune_doc_can_return_image_tokens_only(self):
        hidden_states = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([[7, 99, 99]])
        attention_mask = torch.tensor([[1, 1, 1]])

        pruner = AdaptivePruner(
            image_token_id=99,
            keep_text_tokens=False,
            normalize=False,
        )

        pruned_list, stats = pruner.prune_doc(
            hidden_states=hidden_states,
            attentions=(None,),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        self.assertEqual(len(pruned_list), 1)
        self.assertTrue(torch.equal(pruned_list[0], hidden_states[0, 1:]))
        self.assertEqual(stats.original_patches, [2])
        self.assertEqual(stats.kept_patches, [2])

    def test_prune_doc_with_patch_scores_uses_external_scores(self):
        hidden_states = torch.tensor(
            [[[5.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([[7, 99, 99, 99]])
        attention_mask = torch.tensor([[1, 1, 1, 1]])
        patch_scores = [torch.tensor([0.9, 0.1, 0.2], dtype=torch.float32)]

        pruner = AdaptivePruner(
            image_token_id=99,
            r_min=0.5,
            r_max=0.5,
            normalize=False,
        )

        pruned_list, stats = pruner.prune_doc_with_patch_scores(
            hidden_states=hidden_states,
            patch_scores_list=patch_scores,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        self.assertEqual(stats.original_patches, [3])
        self.assertEqual(stats.kept_patches, [1])
        self.assertTrue(torch.equal(pruned_list[0], torch.tensor([[5.0, 0.0], [1.0, 0.0]])))

    def test_prune_doc_uses_phi3_spatial_patch_mask(self):
        hidden_states = torch.tensor([[[5.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]])
        input_ids = torch.tensor([[7, -1, -1, -1]])
        attention_mask = torch.ones(1, 4, dtype=torch.long)
        spatial_patch_mask = torch.tensor([[False, True, True, False]])
        attention = torch.zeros(1, 1, 4, 4)
        attention[0, 0, 0, 1] = 0.9
        attention[0, 0, 0, 2] = 0.1

        pruner = AdaptivePruner(
            image_token_id=99,
            r_min=0.5,
            r_max=0.5,
            keep_text_tokens=False,
            normalize=False,
        )

        pruned_list, stats = pruner.prune_doc(
            hidden_states=hidden_states,
            attentions=(attention,),
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_mask=spatial_patch_mask,
        )

        self.assertTrue(torch.equal(pruned_list[0], torch.tensor([[1.0, 0.0]])))
        self.assertEqual(stats.original_patches, [2])
        self.assertEqual(stats.kept_patches, [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
