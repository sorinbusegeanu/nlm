import unittest

import pandas as pd

from src.data_factory import build_full_v0_external_scope


def _rows(prefix, count, labels=("alpha", "beta")):
    rows = []
    for index in range(count):
        label = labels[index % len(labels)]
        rows.append({
            "id": f"{prefix}-{index:03d}",
            "prompt_id": f"{prefix}-prompt-{index:03d}",
            "prompt_canon": f"A distinct question {prefix} {index}.",
            "text": f"A unique answer by {label} number {prefix} {index}.",
            "model": label,
        })
    return pd.DataFrame(rows)


class FullV0ValidationScopeTests(unittest.TestCase):
    def setUp(self):
        self.external = _rows("ext", 40)
        self.v0 = _rows("v0", 8)

    def test_uses_all_v0_for_final_and_external_only_for_selection(self):
        rows, assignments, profile = build_full_v0_external_scope(
            self.external, self.v0, expected_classes=2, expected_v0_rows=8)

        training = rows[rows.split.eq("train")]
        validation = rows[rows.split.eq("validation")]
        self.assertEqual(len(training), len(self.external))
        self.assertEqual(set(training.id), set(self.external.id))
        self.assertEqual(len(validation), 8)
        self.assertEqual(set(validation.id), set(self.v0.id))
        self.assertFalse(set(training.id) & set(validation.id))
        self.assertFalse(set(training.text.map(str.casefold)) & set(validation.text.map(str.casefold)))
        self.assertFalse(set(training.prompt_id) & set(validation.prompt_id))
        self.assertEqual(profile["validation_rows"], len(validation))
        self.assertEqual(profile["training_rows"], len(training))
        self.assertEqual(profile["isolation_overlap_counts"], {
            "id": 0, "normalized_response_text": 0,
            "prompt_id": 0, "normalized_prompt_text": 0,
        })
        self.assertEqual(set(assignments.split), {"train", "validation"})

    def test_scope_is_deterministic(self):
        first_rows, first_assignments, first_profile = build_full_v0_external_scope(
            self.external, self.v0, expected_classes=2, expected_v0_rows=8)
        second_rows, second_assignments, second_profile = build_full_v0_external_scope(
            self.external, self.v0, expected_classes=2, expected_v0_rows=8)
        self.assertEqual(first_profile["split_sha256"], second_profile["split_sha256"])
        pd.testing.assert_frame_equal(first_assignments, second_assignments)
        pd.testing.assert_frame_equal(first_rows, second_rows)

    def test_rejects_external_v0_text_and_prompt_overlap(self):
        external = self.external.copy()
        external.loc[0, "text"] = self.v0.loc[0, "text"]
        with self.assertRaisesRegex(ValueError, "overlaps full V0"):
            build_full_v0_external_scope(external, self.v0, expected_classes=2,
                                         expected_v0_rows=8)

        external = self.external.copy()
        external.loc[0, "prompt_canon"] = self.v0.loc[0, "prompt_canon"]
        with self.assertRaisesRegex(ValueError, "overlaps full V0"):
            build_full_v0_external_scope(external, self.v0, expected_classes=2,
                                         expected_v0_rows=8)

    def test_rejects_wrong_v0_size(self):
        with self.assertRaisesRegex(ValueError, "requires 9 official rows"):
            build_full_v0_external_scope(self.external, self.v0, expected_classes=2,
                                         expected_v0_rows=9)


if __name__ == "__main__":
    unittest.main()
