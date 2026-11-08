import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_factory import OPENPOOL_VERTEX_G37_SOURCES, curate_openpool
from src.pipeline import _effective_profile, _load_dataset


class OpenPoolCurationTests(unittest.TestCase):
    def test_curated_training_rows_reuse_v0_validation(self):
        labels = [f"model-{i:02d}" for i in range(17)] + ["gemini-3.7-flash"]
        rows = []
        for label in labels:
            for i in range(4):
                source = (sorted(OPENPOOL_VERTEX_G37_SOURCES)[i % 2]
                          if label == "gemini-3.7-flash" else "source/example")
                rows.append({
                    "id": f"{label}-{i}",
                    "text": f"A unique English response for {label}, sample number {i}.",
                    "model": label,
                    "prompt_canon": f"prompt-{i}",
                    "source_dataset": source,
                    "source_file": "responses.parquet",
                    "source_license": "CC0-1.0",
                    "origin_hint": "" if label == "gemini-3.7-flash" else "captured",
                })
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_path = root / "pool.parquet"
            pd.DataFrame(rows).to_parquet(source_path, index=False)
            processed = root / "processed"
            processed.mkdir()
            v0_rows = []
            v0_splits = []
            for label in labels:
                for i, split in enumerate(("train", "validation")):
                    row_id = f"v0-{label}-{i}"
                    v0_rows.append({"id": row_id, "prompt_id": f"v0-prompt-{i}",
                                    "text": f"Official V0 example for {label} {i}.", "model": label})
                    v0_splits.append({"id": row_id, "prompt_id": f"v0-prompt-{i}",
                                      "model": label, "split": split})
            pd.DataFrame(v0_rows).to_parquet(processed / "dataset_v0.parquet", index=False)
            pd.DataFrame(v0_splits).to_parquet(processed / "splits_v0.parquet", index=False)
            (processed / "dataset_profile.json").write_text(json.dumps({
                "dataset_version": "v0", "dataset_sha256": "base-data", "split_sha256": "frozen-v0-split",
                "validation_rows": 18,
            }))
            # Quota sampling is exercised separately from the full-quota contract.
            for row in rows:
                row["text"] += " Additional distinct wording to maintain unique normalized responses."
            pd.DataFrame(rows).to_parquet(source_path, index=False)
            with patch("src.data_factory._language", return_value="en"):
                profile = curate_openpool(
                    source_path, processed, dataset_version="v1",
                    rows_per_class=3,
                )
            data = pd.read_parquet(processed / "dataset_v1.parquet")
            self.assertEqual(len(data), 54)
            counts = data.model.value_counts().to_dict()
            self.assertEqual(counts["gemini-3.7-flash"], 3)
            self.assertTrue(all(counts[label] == 3 for label in labels[:-1]))
            self.assertEqual(set(data.loc[data.model.eq("gemini-3.7-flash"), "source_dataset"])
                             <= OPENPOOL_VERTEX_G37_SOURCES, True)
            self.assertTrue(data.id.is_unique)
            self.assertFalse((processed / "splits_v1.parquet").exists())
            self.assertEqual(profile["dataset_version"], "v1")
            self.assertEqual(profile["selected_rows_by_class"]["gemini-3.7-flash"], 3)
            self.assertTrue((processed / "openpool_v1_removals.csv").exists())
            self.assertEqual(profile["removals_by_reason"]["quota_source_stratified_downsample"], 18)
            self.assertEqual(profile["validation_dataset_version"], "v0")
            self.assertEqual(profile["validation_rows"], 18)
            combined, _ = _load_dataset("v1", processed)
            self.assertEqual(int(combined.split.eq("train").sum()), 54)
            self.assertEqual(int(combined.split.eq("validation").sum()), 18)
            self.assertEqual(set(combined.loc[combined.split.eq("train"), "id"]),
                             set(data.id))
            self.assertTrue(set(combined.loc[combined.split.eq("validation"), "id"])
                            .isdisjoint(set(combined.loc[combined.split.eq("train"), "id"])))
            effective = _effective_profile(processed, "v1", combined)
            self.assertEqual(effective["dataset_sha256"], profile["dataset_sha256"])
            self.assertEqual(effective["training_rows"], 54)
            self.assertEqual(effective["validation_rows"], 18)
            self.assertEqual(effective["effective_dataset_components"]["v0_validation_split_sha256"],
                             "frozen-v0-split")


if __name__ == "__main__":
    unittest.main()
