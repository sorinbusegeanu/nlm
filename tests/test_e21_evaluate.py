from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e21_evaluate import evaluate_epoch_scores, hungarian_predictions


class E21EvaluationTests(unittest.TestCase):
    def test_hungarian_assignment_uses_logits_and_unique_labels(self):
        classes = ["a", "b", "c"]
        scores = np.asarray([[10, 9, 0], [9, 0, 0], [0, 0, 5]], dtype=float)
        result = hungarian_predictions(scores, ["g", "g", "h"], classes)
        self.assertEqual(set(result[:2]), {"a", "b"})
        self.assertEqual(result[2], "c")

    def test_rejects_groups_larger_than_label_roster(self):
        with self.assertRaisesRegex(ValueError, "unique labels"):
            hungarian_predictions(np.zeros((3, 2)), ["g"] * 3, ["a", "b"])

    def test_all_methods_and_corrections_are_reported(self):
        classes = ["a", "b", "c"]
        y_true = ["a", "b", "c"]
        scores = np.asarray([[8, 4, 0], [5, 4, 0], [0, 0, 5]], dtype=float)
        groups = {"e19_d": ["g", "g", "g"], "oracle_true_prompt": ["g", "g", "g"]}
        rows, labels, predictions = evaluate_epoch_scores(scores, y_true, classes, groups, epoch=1)
        independent = next(row for row in rows if row["method"] == "independent_argmax")
        grouped = next(row for row in rows if row["method"] == "e19_d")
        self.assertEqual(independent["top3_accuracy"], 1.0)
        self.assertEqual(grouped["corrected"], 1)
        self.assertEqual(grouped["degraded"], 0)
        self.assertEqual(len(labels), 3 * 3)
        self.assertEqual(predictions["oracle_true_prompt"].tolist(), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
