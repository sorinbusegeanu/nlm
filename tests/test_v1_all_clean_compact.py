import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_factory import curate_v1_all_clean_compact
from src.pipeline import _load_dataset
from tests.test_v1_all_balanced_deduped import LABELS, make_fixture


class V1AllCleanCompactTests(unittest.TestCase):
    def test_defect_cleaning_near_duplicate_compaction_and_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, processed = make_fixture(root)
            original = pd.read_parquet(source)
            additions = [
                {"id": "bad-envelope", "text": "<|im_start|>assistant\nserialized payload", "model": LABELS[0],
                 "prompt_id": "bad-prompt-1", "prompt_canon": "bad prompt one", "source_dataset": "source-x",
                 "source_file": "fixture.parquet", "source_license": "CC0", "origin_hint": "test"},
                {"id": "bad-truncated", "text": "Incomplete answer [truncated]", "model": LABELS[1],
                 "prompt_id": "bad-prompt-2", "prompt_canon": "bad prompt two", "source_dataset": "source-x",
                 "source_file": "fixture.parquet", "source_license": "CC0", "origin_hint": "test"},
            ]
            original = pd.concat([original, pd.DataFrame(additions)], ignore_index=True)
            original.to_parquet(source, index=False)

            result = curate_v1_all_clean_compact(source, processed,
                compact_char_similarity=.92, compact_word_similarity=.85, tfidf_diagnostic_sample=0)
            clean = pd.read_parquet(processed / "dataset_v1-all_clean.parquet")
            compact = pd.read_parquet(processed / "dataset_v1-all_compact.parquet")
            clean_audit = pd.read_csv(processed / "v1-all_clean_removals.csv")
            compact_audit = pd.read_csv(processed / "v1-all_compact_removals.csv")

            self.assertIn("serialized_envelope_or_extraction_artifact",
                          set(clean_audit.removal_reason))
            self.assertIn("explicit_truncation_or_failure_marker", set(clean_audit.removal_reason))
            self.assertEqual(set(clean.model.astype(str)), set(LABELS))
            self.assertEqual(set(compact.id.astype(str)), set(clean.id.astype(str)) - set(compact_audit.id.astype(str)))
            self.assertTrue(set(compact.id.astype(str)).issubset(set(clean.id.astype(str))))
            self.assertTrue(compact_audit.representative_id.notna().all())
            self.assertGreater(len(compact_audit), 0)
            self.assertTrue(compact_audit.removal_reason.eq("verified_same_label_near_duplicate").all())
            self.assertEqual(result["clean"]["validation_split_sha256"], "frozen-test-split")
            self.assertEqual(result["clean"]["validation_id_overlap"], 0)
            self.assertEqual(result["clean"]["validation_prompt_overlap"], 0)
            combined, _ = _load_dataset("v1-all_clean", processed)
            self.assertEqual(int(combined.split.eq("validation").sum()), len(LABELS))

    def test_same_source_and_seed_are_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, processed = make_fixture(root)
            first_dir = processed / "first"
            first_dir.mkdir()
            for name in ("dataset_v0.parquet", "splits_v0.parquet", "dataset_profile.json"):
                (first_dir / name).write_bytes((processed / name).read_bytes())
            first = curate_v1_all_clean_compact(source, first_dir, tfidf_diagnostic_sample=0)

            second_dir = processed / "second"
            second_dir.mkdir()
            for name in ("dataset_v0.parquet", "splits_v0.parquet", "dataset_profile.json"):
                (second_dir / name).write_bytes((processed / name).read_bytes())
            second = curate_v1_all_clean_compact(source, second_dir, tfidf_diagnostic_sample=0)
            for version in ("v1-all_clean", "v1-all_compact"):
                a = pd.read_parquet(first_dir / f"dataset_{version}.parquet")
                b = pd.read_parquet(second_dir / f"dataset_{version}.parquet")
                pd.testing.assert_frame_equal(a, b)
            self.assertEqual(first["clean"]["dataset_sha256"], second["clean"]["dataset_sha256"])
            self.assertEqual(first["compact"]["dataset_sha256"], second["compact"]["dataset_sha256"])


if __name__ == "__main__":
    unittest.main()
