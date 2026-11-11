import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline import _original_response_rows, _partition_sha256, select_single_training_views
from src.v2_0_chunks800 import build_dataset, chunk_starts
from src.model_factory import select_epoch_view_indices


class V20Chunks800Tests(unittest.TestCase):
    def test_chunk_windows_are_reproducible_and_bounded(self):
        text = " ".join(f"word{i}" for i in range(500))
        a = chunk_starts(text, seed=2026, original_id="row-1")
        b = chunk_starts(text, seed=2026, original_id="row-1")
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 3)
        self.assertGreater(len(a), 1)
        self.assertEqual(len({start for start, _ in a}), len(a))
        for start, end in a:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end - start, 801)
            self.assertTrue(text[start].isalnum())
        self.assertEqual(chunk_starts("short response", original_id="short"), [])

    def test_generated_artifact_preserves_full_response_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data/processed"
            data_dir.mkdir(parents=True)
            text = " ".join(f"token{i}" for i in range(300))
            base = pd.DataFrame([
                {"id": "orig-long", "text": text, "model": "label-a", "prompt_id": "p1",
                 "source_dataset": "s", "source_file": "f", "source_license": "l"},
                {"id": "orig-short", "text": "short " * 50, "model": "label-b", "prompt_id": "p2",
                 "source_dataset": "s", "source_file": "f", "source_license": "l"},
            ])
            base.to_parquet(data_dir / "dataset_v2.0.parquet", index=False)
            (data_dir / "dataset_profile_v2.0.json").write_text('{"classes":["label-a","label-b"]}')
            views, profile = build_dataset(root=root)
            long = views[views.original_response_id.eq("orig-long")]
            short = views[views.original_response_id.eq("orig-short")]
            self.assertEqual(len(long[long.view_type.eq("full_response")]), 1)
            self.assertEqual(long.loc[long.view_type.eq("full_response"), "text"].iloc[0], text)
            self.assertLessEqual(len(long[long.view_type.eq("chunk")]), 3)
            self.assertEqual(len(short), 1)
            self.assertEqual(set(long.model), {"label-a"})
            self.assertEqual(profile["original_response_count"], 2)

    def test_partition_collapse_and_epoch_selection_keep_one_per_original(self):
        frame = pd.DataFrame([
            {"id": "v-full", "original_response_id": "o1", "view_type": "full_response",
             "text": "the original full response", "model": "a", "prompt_id": "p1",
             "chunk_id": "full", "char_start": 0},
            {"id": "v-chunk", "original_response_id": "o1", "view_type": "chunk",
             "text": "a response chunk", "model": "a", "prompt_id": "p1",
             "chunk_id": "chunk-01", "char_start": 15},
        ])
        collapsed = _original_response_rows(frame, training=True)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed.id.iloc[0], "o1")
        self.assertEqual(len(collapsed.training_views.iloc[0]), 2)
        self.assertNotEqual(_partition_sha256(collapsed), _partition_sha256(collapsed.assign(
            training_views=[["the original full response"]])))
        first, counts = select_epoch_view_indices([2] * 100, seed=2026, epoch=0)
        same, _ = select_epoch_view_indices([2] * 100, seed=2026, epoch=0)
        next_epoch, _ = select_epoch_view_indices([2] * 100, seed=2026, epoch=1)
        self.assertEqual(first, same)
        self.assertEqual(len(first), 100)
        self.assertEqual(sum(counts.values()), 100)
        self.assertNotEqual(first, next_epoch)
        # Pipeline's single-fit classifier path uses the same reproducible rule.
        selected_a, selected_counts = select_single_training_views(
            [["full", "chunk"]] * 100, seed=2026)
        selected_b, _ = select_single_training_views([["full", "chunk"]] * 100, seed=2026)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(sum(selected_counts.values()), 100)

    def test_text_dataset_switches_views_by_epoch_without_changing_row_count(self):
        from src.model_factory import _TextDataset

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                ids = [[ord(char) for char in str(text)] or [0] for text in texts]
                return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

        originals = [f"full response {i}" for i in range(64)]
        views = [[text, f"chunk variant {i}"] for i, text in enumerate(originals)]
        view_ids = [["full", "chunk-01"] for _ in originals]
        dataset = _TextDataset(originals, [0] * len(originals), Tokenizer(), 64,
                               training_views=views, training_view_ids=view_ids, seed=2026)
        before = [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]
        dataset.set_epoch(1)
        after = [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]
        self.assertEqual(len(before), len(after))
        self.assertNotEqual(before, after)
        self.assertEqual(dataset.epoch_records[-1]["original_responses"], len(originals))
        self.assertEqual(sum(dataset.epoch_records[-1]["selected_view_counts"].values()), len(originals))


if __name__ == "__main__":
    unittest.main()
