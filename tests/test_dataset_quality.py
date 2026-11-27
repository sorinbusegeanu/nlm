import unittest

import pandas as pd

from src.dataset_quality import filter_dataset


class DatasetQualityTests(unittest.TestCase):
    def test_english_length_and_duplicate_rules_are_deterministic(self):
        rows = pd.DataFrame([
            {"id": "b", "source_row_id": "b", "prompt_id": "p1", "text": "A" * 60,
             "model": "glm-5.3-flash", "source_dataset": "source", "language": "en"},
            {"id": "a", "source_row_id": "a", "prompt_id": "p2", "text": "  " + "A" * 60 + "  ",
             "model": "glm-5.3-flash", "source_dataset": "source", "language": "en"},
            {"id": "zh", "source_row_id": "zh", "prompt_id": "p3", "text": "B" * 60,
             "model": "glm-5.3-flash", "source_dataset": "source", "language": "zh"},
            {"id": "short", "source_row_id": "short", "prompt_id": "p4", "text": "tiny",
             "model": "glm-5.3-flash", "source_dataset": "source", "language": "en"},
        ])
        kept, audit, stats = filter_dataset(rows, min_chars=50, max_chars=100)
        self.assertEqual(kept.id.tolist(), ["a"])
        self.assertEqual(stats["retained_rows"], 1)
        self.assertEqual(audit.removal_reason.value_counts().to_dict(), {
            "normalized_duplicate_response": 1,
            "not_explicitly_english": 1,
            "response_too_short": 1,
        })

    def test_cross_label_identical_text_excludes_both_labels(self):
        rows = pd.DataFrame([
            {"id": "a", "source_row_id": "a", "prompt_id": "p1", "text": "X" * 60,
             "model": "glm-5.3-flash", "source_dataset": "source", "language": "en"},
            {"id": "b", "source_row_id": "b", "prompt_id": "p2", "text": " " + "X" * 60 + "  ",
             "model": "glm-5.3", "source_dataset": "source", "language": "en"},
        ])
        kept, audit, _ = filter_dataset(rows, min_chars=50, max_chars=100)
        self.assertTrue(kept.empty)
        self.assertEqual(len(audit), 2)
        self.assertTrue(audit.removal_reason.eq("conflicting_label_duplicate").all())


if __name__ == "__main__":
    unittest.main()
