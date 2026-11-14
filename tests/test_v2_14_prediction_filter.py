import unittest

import numpy as np
import pandas as pd

from src.v2_14_prediction_filter import (
    CATEGORY_NAMES,
    RETENTION_80PCT,
    select_rows,
)


class V214PredictionFilterTests(unittest.TestCase):
    def test_80pct_retention_configuration(self):
        self.assertEqual(RETENTION_80PCT.tolist(), [0.60, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(len(CATEGORY_NAMES), 6)

    def test_custom_retention_is_deterministic_and_respects_full_keep_categories(self):
        frame = pd.DataFrame({
            "id": [f"r{i:03d}" for i in range(120)],
            "text": [f"response number {i} with enough words" for i in range(120)],
            "model": ["label-a"] * 120,
            "source_dataset": ["source-a"] * 120,
        })
        category = np.repeat(np.arange(6, dtype=np.int8), 20)
        rates = np.array([0.60, 1, 1, 1, 1, 1], dtype=float)
        first = select_rows(frame, category, seed=2026, retention=rates)
        second = select_rows(frame, category, seed=2026, retention=rates)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first[category == 0].sum()), 12)
        self.assertTrue(first[category != 0].all())

    def test_invalid_retention_is_rejected(self):
        frame = pd.DataFrame({"id": ["a"], "text": ["a b"], "model": ["label"], "source_dataset": ["src"]})
        with self.assertRaises(ValueError):
            select_rows(frame, np.array([0]), retention=[0.5])


if __name__ == "__main__":
    unittest.main()
