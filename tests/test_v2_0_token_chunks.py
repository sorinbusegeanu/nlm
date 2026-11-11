import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.v2_0_token_chunks import (
    DATASET_SPECS,
    _coverages,
    _window_starts,
    build_dataset,
    chunk_response,
)


class CharacterTokenizer:
    """Small deterministic tokenizer stand-in; each character is one token."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, *, add_special_tokens=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[ord(char) for char in str(text)] for text in texts]
        if add_special_tokens:
            rows = [[1, *row, 2] for row in rows]
        return {"input_ids": rows}

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)


class V20TokenChunksTests(unittest.TestCase):
    def test_window_boundaries_and_complete_coverage(self):
        windows = _window_starts(1001, capacity=10, stride=5)
        self.assertEqual(windows[0], (0, 10))
        self.assertEqual(windows[-1][1], 1001)
        self.assertEqual(_coverages(1001, windows), 1001)
        self.assertEqual(len(windows), len(set(windows)))
        self.assertEqual(_window_starts(8, capacity=10, stride=5), [(0, 8)])

    def test_chunking_obeys_special_token_limit_and_is_deterministic(self):
        tokenizer = CharacterTokenizer()
        text = "abcdefghijklmnopqrstuvwxyz" * 20
        first, original_count, capacity = chunk_response(
            tokenizer, text, max_encoded_length=12, stride=5)
        second, _, _ = chunk_response(tokenizer, text, max_encoded_length=12, stride=5)
        self.assertEqual(first, second)
        self.assertEqual(original_count, len(text))
        self.assertEqual(capacity, 10)
        self.assertEqual(_coverages(original_count, [
            (row["token_start"], row["token_end"]) for row in first]), original_count)
        self.assertTrue(all(len(tokenizer(row["text"])["input_ids"][0]) <= 12 for row in first))
        short, _, _ = chunk_response(tokenizer, "short", max_encoded_length=12, stride=5)
        self.assertEqual(len(short), 1)

    def test_generated_rows_are_independent_traceable_and_unbalanced_by_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data/processed"
            config_dir = root / "configs/models"
            data_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            (config_dir / "ultrabert380.yaml").write_text(
                "batch_size: 8\ngradient_accumulation_steps: 2\n")
            source = pd.DataFrame([
                {"id": "orig-a", "text": "a" * 650, "model": "label-a", "prompt_id": "pa",
                 "source_dataset": "raw-a", "source_file": "a.parquet", "license": "test"},
                {"id": "orig-b", "text": "b" * 80, "model": "label-b", "prompt_id": "pb",
                 "source_dataset": "raw-b", "source_file": "b.parquet", "license": "test"},
            ])
            source.to_parquet(data_dir / "dataset_v2.0.parquet", index=False)
            (data_dir / "dataset_profile_v2.0.json").write_text(json.dumps({
                "rows": 2, "classes": ["label-a", "label-b"]
            }))
            pd.DataFrame([
                {"id": "v0-a", "text": "unique validation alpha", "model": "label-a", "prompt_id": "vpa"},
                {"id": "v0-b", "text": "unique validation beta", "model": "label-b", "prompt_id": "vpb"},
            ]).to_parquet(data_dir / "dataset_v0.parquet", index=False)

            dataset = "v2.0.chunks.300TokBalanced"
            rows, profile = build_dataset(dataset, root=root, tokenizer=CharacterTokenizer(), batch_size=1)
            self.assertEqual(len(rows), 5)
            self.assertNotIn("view_type", rows.columns)
            self.assertEqual(rows.id.nunique(), len(rows))
            self.assertEqual(rows.original_response_id.nunique(), len(source))
            self.assertEqual(rows.loc[rows.model.eq("label-a"), "source_dataset"].unique().tolist(), ["raw-a"])
            self.assertEqual(rows["label"].tolist(), rows.model.tolist())
            self.assertEqual(profile["max_encoded_length"], 300)
            self.assertEqual(profile["v0_normalized_text_overlap"]["generated_chunks"], 0)
            self.assertEqual(profile["v0_prompt_id_overlap_rows"], 0)
            self.assertEqual(profile["original_token_coverage"]["fraction"], 1.0)
            self.assertEqual(profile["original_response_coverage"]["fraction"], 1.0)
            self.assertEqual(profile["chunking_strategy"], "ultrabert_token_windows_v1")

    def test_requested_specs_use_50_percent_stride(self):
        self.assertEqual(DATASET_SPECS["v2.0.chunks.300TokBalanced"],
                         {"max_encoded_length": 300, "stride": 150})
        self.assertEqual(DATASET_SPECS["v2.0.chunks.200TokBalanced"],
                         {"max_encoded_length": 200, "stride": 100})


if __name__ == "__main__":
    unittest.main()
