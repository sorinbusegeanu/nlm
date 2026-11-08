import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_factory import curate_v1_all_balanced_deduped
from src.pipeline import _load_dataset


LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]


def make_fixture(root: Path):
    processed = root / "processed"
    processed.mkdir()
    rows = []
    val_rows, assignments = [], []
    for label in LABELS:
        for n in range(4):
            text = (f"A complete response about {label} and a distinct topic number {n}. "
                    + (f"topic{n} detail " * 20) + "This is ordinary complete prose.")
            if label == LABELS[0] and n == 1:
                text = rows[-1]["text"][:-1] + "!"
            rows.append({"id": f"{label}-pool-{n}", "text": text, "model": label,
                         "prompt_id": f"pool-prompt-{label}-{n}", "prompt_canon": f"unique pool prompt {label} {n}",
                         "source_dataset": f"source-{n % 2}", "source_file": "fixture.parquet",
                         "source_license": "CC0", "origin_hint": "test capture"})
            if n == 0:
                for split in ("train", "validation"):
                    row_id = f"v0-{label}-{split}"
                    val_rows.append({"id": row_id, "text": f"Frozen official V0 row for {label} ({split}).",
                                     "model": label, "prompt_id": f"v0-prompt-{label}-{split}"})
                    assignments.append({"id": row_id, "prompt_id": f"v0-prompt-{label}-{split}",
                                        "model": label, "split": split})
    source = root / "v1_all.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    pd.DataFrame(val_rows).to_parquet(processed / "dataset_v0.parquet", index=False)
    pd.DataFrame(assignments).to_parquet(processed / "splits_v0.parquet", index=False)
    val_ids = sorted(x["id"] for x in assignments if x["split"] == "validation")
    (processed / "dataset_profile.json").write_text(json.dumps({
        "dataset_version": "v0", "split_sha256": "frozen-test-split",
        "validation_rows": len(val_ids),
    }))
    return source, processed


class V1AllCurationTests(unittest.TestCase):
    def test_deduplicates_deterministically_preserves_classes_and_v0_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, processed = make_fixture(root)
            first = curate_v1_all_balanced_deduped(source, processed, near_duplicate_threshold=.9)
            first_data = pd.read_parquet(processed / "dataset_v1-all_balanced_deduped.parquet")
            self.assertEqual(set(first_data.model), set(LABELS))
            self.assertEqual(first["removals_by_reason"]["near_duplicate_response"], 1)
            self.assertEqual(first["validation_id_overlap"], 0)
            self.assertEqual(first["validation_prompt_overlap"], 0)
            combined, _ = _load_dataset("v1-all_balanced_deduped", processed)
            self.assertEqual(int(combined.split.eq("validation").sum()), len(LABELS))
            self.assertTrue(set(first_data.id).isdisjoint(
                set(combined.loc[combined.split.eq("validation"), "id"])))

            second_dir = root / "repeat"
            second_dir.mkdir()
            for name in ("dataset_v0.parquet", "splits_v0.parquet", "dataset_profile.json"):
                (second_dir / name).write_bytes((processed / name).read_bytes())
            second = curate_v1_all_balanced_deduped(source, second_dir, near_duplicate_threshold=.9)
            second_data = pd.read_parquet(second_dir / "dataset_v1-all_balanced_deduped.parquet")
            pd.testing.assert_frame_equal(first_data, second_data)
            self.assertEqual(first["removal_manifest_sha256"], second["removal_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
