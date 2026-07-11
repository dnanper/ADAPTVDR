import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.precompute_teacher_attn import (
    _batched_rows,
    _load_seen_ids,
    _prepare_output_dir,
    _save_batch_shard,
    select_attention_layer,
    aggregate_source_to_image_attention,
    aggregate_source_to_image_attention_with_fallback,
    aggregate_query_to_image_attention,
    find_subsequence_positions,
    get_all_non_image_positions,
    get_instruction_token_positions,
    normalize_instruction_text,
    stable_sample_id,
)


class TestPrecomputeTeacherAttention(unittest.TestCase):
    def test_stable_sample_id_is_deterministic(self):
        kwargs = {
            "shard_path": "train-00000-of-00082.parquet",
            "row_idx": 17,
            "image_filename": "page_9.jpg",
            "query": "What is the duration of the course?",
        }
        self.assertEqual(stable_sample_id(**kwargs), stable_sample_id(**kwargs))

    def test_find_subsequence_positions_returns_contiguous_match(self):
        sequence = [10, 20, 30, 40, 50, 60]
        subsequence = [30, 40, 50]
        self.assertEqual(find_subsequence_positions(sequence, subsequence), [2, 3, 4])

    def test_aggregate_query_to_image_attention_averages_heads_and_queries(self):
        attn_layer = torch.tensor(
            [
                [
                    [0.0, 0.1, 0.2, 0.3],
                    [0.0, 0.4, 0.5, 0.6],
                    [0.0, 0.7, 0.8, 0.9],
                    [0.0, 1.0, 1.1, 1.2],
                ],
                [
                    [0.0, 1.1, 1.2, 1.3],
                    [0.0, 1.4, 1.5, 1.6],
                    [0.0, 1.7, 1.8, 1.9],
                    [0.0, 2.0, 2.1, 2.2],
                ],
            ],
            dtype=torch.float32,
        )
        query_positions = torch.tensor([0, 1])
        image_positions = torch.tensor([2, 3])

        out = aggregate_query_to_image_attention(
            attn_layer=attn_layer,
            query_positions=query_positions,
            image_positions=image_positions,
        )

        expected = torch.tensor([0.85, 0.95], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected))

    def test_aggregate_source_to_image_attention_matches_generic_case(self):
        attn_layer = torch.tensor(
            [
                [
                    [0.0, 0.1, 0.2],
                    [0.3, 0.4, 0.5],
                    [0.6, 0.7, 0.8],
                ]
            ],
            dtype=torch.float32,
        )
        out = aggregate_source_to_image_attention(
            attn_layer=attn_layer,
            source_positions=torch.tensor([0, 1]),
            image_positions=torch.tensor([2]),
        )
        self.assertTrue(torch.allclose(out, torch.tensor([0.35], dtype=torch.float32)))

    def test_aggregate_source_to_image_attention_falls_back_when_causal_source_is_zero(self):
        attn_layer = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.7, 0.0],
                    [0.2, 0.1, 0.4, 0.3],
                    [0.3, 0.1, 0.2, 0.4],
                ]
            ],
            dtype=torch.float32,
        )

        out = aggregate_source_to_image_attention_with_fallback(
            attn_layer=attn_layer,
            source_positions=torch.tensor([0]),
            image_positions=torch.tensor([2, 3]),
        )

        self.assertTrue(torch.allclose(out, torch.tensor([0.3, 0.35], dtype=torch.float32)))

    def test_get_instruction_token_positions_finds_instruction_span(self):
        input_ids = torch.tensor([101, 11, 12, 13, 201, 202])
        attention_mask = torch.tensor([1, 1, 1, 1, 1, 1])

        out = get_instruction_token_positions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            instruction_token_ids=[11, 12, 13],
        )

        self.assertTrue(torch.equal(out, torch.tensor([1, 2, 3])))

    def test_get_all_non_image_positions_excludes_image_tokens(self):
        input_ids = torch.tensor([9, 99, 10, 99, 11])
        attention_mask = torch.tensor([1, 1, 1, 1, 0])

        out = get_all_non_image_positions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_id=99,
        )

        self.assertTrue(torch.equal(out, torch.tensor([0, 2])))

    def test_normalize_instruction_text_appends_terminal_punctuation(self):
        self.assertEqual(normalize_instruction_text("Represent the user's input"), "Represent the user's input.")

    def test_batched_rows_groups_items_by_batch_size(self):
        rows = iter(
            [
                ("a", 0, "row0"),
                ("a", 1, "row1"),
                ("b", 0, "row2"),
            ]
        )

        out = list(_batched_rows(rows, batch_size=2))

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [("a", 0, "row0"), ("a", 1, "row1")])
        self.assertEqual(out[1], [("b", 0, "row2")])

    def test_batch_shards_are_saved_and_reloadable_for_resume(self):
        with TemporaryDirectory() as tmpdir:
            batch_dir = _prepare_output_dir(Path(tmpdir) / "attn_cache.pt")
            _save_batch_shard(
                {"id-1": torch.tensor([1.0], dtype=torch.float16)},
                batch_dir=batch_dir,
                batch_index=0,
            )
            _save_batch_shard(
                {"id-2": torch.tensor([2.0, 3.0], dtype=torch.float16)},
                batch_dir=batch_dir,
                batch_index=1,
            )

            seen_ids = _load_seen_ids(batch_dir)

            self.assertEqual(seen_ids, {"id-1", "id-2"})

    def test_select_attention_layer_uses_last_non_none_for_default_mode(self):
        attn0 = torch.ones(1, 1, 2, 2)
        attn1 = None
        attn2 = torch.full((1, 1, 2, 2), 2.0)

        out = select_attention_layer((attn0, attn1, attn2), layer_index=-1)

        self.assertIs(out, attn2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
