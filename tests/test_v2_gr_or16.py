from __future__ import annotations

import unittest

import pandas as pd

from src.v2_gr_or16 import LABELS, build_dataset, prompt_key, validate_artifact
from src.dataset_catalog import load_dataset_catalog


def fixture() -> pd.DataFrame:
    labels = sorted(LABELS)
    rows = []
    # Two canonical groups and one fallback group; repeated labels have distinct outputs.
    for i, label in enumerate(labels):
        rows.append({"id": f"r{i:02d}", "text": f"unique response {i}", "model": label,
                     "prompt_canon": " Canon A " if i < 2 else ("" if i < 4 else f"canon-{i}"),
                     "prompt_group": "pg-a" if i < 2 else "pg-fallback" if i < 4 else f"pg-{i}",
                     "source_dataset": "src", "source_file": "file2.csv" if i == 3 else "file.csv"})
    rows.append({"id": "r18", "text": "third canonical A response", "model": labels[0],
                 "prompt_canon": "canon a", "prompt_group": "else", "source_dataset": "other", "source_file": "other.csv"})
    return pd.DataFrame(rows)


class V2GROr16Tests(unittest.TestCase):
    def test_namespaces_and_fallback_tuple(self):
        df = fixture()
        data, _ = build_dataset(df)
        self.assertTrue(data.loc[0, "prompt_id"].startswith("canon:"))
        self.assertTrue(data.loc[2, "prompt_id"].startswith("fallback:"))
        self.assertNotEqual(data.loc[0, "prompt_id"], data.loc[2, "prompt_id"])
        # Canonical key joins rows from different sources; fallback key keeps all tuple fields.
        self.assertEqual(data.loc[0, "prompt_id"], data.loc[1, "prompt_id"])
        self.assertNotEqual(data.loc[2, "prompt_id"], data.loc[3, "prompt_id"])

    def test_group_stats_eligibility_and_rotation(self):
        data, profile = build_dataset(fixture())
        validate_artifact(data, profile)
        canonical = data[data.prompt_id.eq(data.loc[0, "prompt_id"])]
        self.assertEqual(canonical.group_size.unique().tolist(), [3])
        self.assertEqual(canonical.group_label_count.unique().tolist(), [2])
        same_label = canonical[canonical.model.eq(canonical.model.iloc[0])].sort_values("id")
        self.assertEqual(same_label.ranking_rotation_index.tolist(), [0, 1])
        self.assertEqual(same_label.group_model_count.unique().tolist(), [2])
        self.assertTrue(canonical.ranking_eligible.all())

    def test_v0_overlap_rejected(self):
        data = fixture()
        v0 = pd.DataFrame({"id": ["v0"], "text": [" UNIQUE   RESPONSE 0 "], "prompt_id": ["v0p"]})
        with self.assertRaisesRegex(ValueError, "overlaps V0"):
            build_dataset(data, v0=v0)

    def test_normalized_duplicates_rejected(self):
        data = fixture()
        data.loc[1, "text"] = " UNIQUE  RESPONSE 0 "
        with self.assertRaisesRegex(ValueError, "normalized response duplicates"):
            build_dataset(data)

    def test_label_counts_and_singleton_classification_retention(self):
        data, profile = build_dataset(fixture())
        self.assertEqual(len(data), 19)
        self.assertEqual(data.model.nunique(), 18)
        self.assertEqual(int((~data.ranking_eligible).sum()), 16)
        self.assertEqual(profile["classification_rows"], len(data))

    def test_catalog_reserves_v0_validation(self):
        catalog = load_dataset_catalog()
        entry = next(item for item in catalog["datasets"] if item["id"] == "v2.GR.OR16")
        self.assertTrue(entry["external_only"])
        self.assertEqual(entry["validation_dataset"], "v0")
        self.assertEqual(entry["validation_rows"], 2699)


if __name__ == "__main__":
    unittest.main()
