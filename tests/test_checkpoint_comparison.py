import unittest

import numpy as np
import pandas as pd

from src.checkpoint_comparison import (
    align_checkpoint_predictions,
    checkpoint_metrics,
    compare_probability_matrices,
)


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["a", "b", "c", "d"]
        self.validation = pd.DataFrame({
            "id": ["r1", "r2", "r3", "r4"],
            "prompt_id": ["p1", "p2", "p3", "p4"],
            "model": ["a", "b", "c", "d"],
        })

    def frame(self, probabilities, order=None):
        order = list(range(len(self.validation))) if order is None else order
        frame = self.validation.iloc[order].copy().rename(columns={"model": "y_true"})
        values = np.asarray(probabilities)[order]
        for index, label in enumerate(self.classes):
            frame[f"prob::{label}"] = values[:, index]
        return frame

    def test_rank_metrics_top3_mrr_and_per_class(self):
        probs = np.array([
            [.6, .2, .1, .1],
            [.1, .5, .3, .1],
            [.1, .2, .4, .3],
            [.1, .1, .2, .6],
        ])
        result = checkpoint_metrics(self.validation.model, probs, self.classes)
        self.assertAlmostEqual(result["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(result["top3_accuracy"], 1.0)
        self.assertAlmostEqual(result["mrr"], 1.0)
        self.assertAlmostEqual(result["per_class"]["c"]["top1_recall"], 1.0)
        self.assertAlmostEqual(result["per_class"]["c"]["top3_accuracy"], 1.0)

    def test_alignment_joins_by_id_and_rejects_mismatched_metadata(self):
        p = np.eye(4) * .7 + .075
        aligned, previous, new = align_checkpoint_predictions(
            self.frame(p, [2, 0, 3, 1]), self.frame(p, [1, 3, 0, 2]),
            self.classes, self.validation)
        self.assertEqual(aligned.id.tolist(), self.validation.id.tolist())
        np.testing.assert_allclose(previous, new)
        bad = self.frame(p)
        bad.loc[bad.id == "r1", "y_true"] = "b"
        with self.assertRaisesRegex(ValueError, "y_true differs"):
            align_checkpoint_predictions(self.frame(p), bad, self.classes, self.validation)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            align_checkpoint_predictions(self.frame(p), self.frame(p).iloc[:3],
                                         self.classes, self.validation)

    def test_metric_deltas_and_prediction_movements(self):
        previous = np.array([
            [.7, .1, .1, .1],
            [.5, .3, .1, .1],
            [.1, .5, .3, .1],
            [.1, .1, .1, .7],
        ])
        new = np.array([
            [.2, .6, .1, .1],
            [.5, .3, .1, .1],
            [.1, .2, .6, .1],
            [.4, .2, .2, .2],
        ])
        result = compare_probability_matrices(self.validation.model, previous, new, self.classes)
        self.assertEqual(result["corrected_top1"], 1)
        self.assertEqual(result["regressed_top1"], 2)
        self.assertEqual(result["rank_up"], 1)
        self.assertEqual(result["rank_down"], 2)
        self.assertEqual(result["rank_same"], 1)
        expected_delta = np.mean([.2 - .7, .3 - .3, .6 - .3, .2 - .7])
        self.assertAlmostEqual(result["mean_true_probability_change"], expected_delta)
        self.assertAlmostEqual(result["previous"]["mrr"], .75)
        self.assertAlmostEqual(result["new"]["mrr"], .5625)


if __name__ == "__main__":
    unittest.main()
