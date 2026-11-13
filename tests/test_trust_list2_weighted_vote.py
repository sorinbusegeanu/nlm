import unittest

import numpy as np
import pandas as pd

from src.trust_list2_weighted_vote import _normalized_vote_scores
from src.v2_checkpoint_trust import LABELS


class TrustList2VotingComparisonTests(unittest.TestCase):
    def test_equal_and_precision_weights_share_family_normalization(self):
        label_a, label_b = LABELS[:2]
        eligible = pd.DataFrame([
            {"experiment_id": "a1", "target_label": label_a, "precision": 1.0, "prediction_count": 30},
            {"experiment_id": "a2", "target_label": label_a, "precision": 0.5, "prediction_count": 30},
            {"experiment_id": "b1", "target_label": label_a, "precision": 0.8, "prediction_count": 30},
            {"experiment_id": "a1", "target_label": label_b, "precision": 0.7, "prediction_count": 30},
            {"experiment_id": "a2", "target_label": label_b, "precision": 0.9, "prediction_count": 30},
            {"experiment_id": "b1", "target_label": label_b, "precision": 0.6, "prediction_count": 30},
        ])
        checkpoints = pd.DataFrame({"experiment_id": ["a1", "a2", "b1"]})
        families = {"a1": "family-a", "a2": "family-a", "b1": "family-b"}
        counts = {"family-a": 2, "family-b": 1}
        keyed = eligible.set_index(["experiment_id", "target_label"])
        predictions = {
            "a1": np.asarray([label_a, label_b], dtype=object),
            "a2": np.asarray([label_a, label_b], dtype=object),
            "b1": np.asarray([label_a, label_b], dtype=object),
        }
        equal = _normalized_vote_scores(eligible, checkpoints, families, counts, keyed,
                                        predictions, weighted=False)
        weighted = _normalized_vote_scores(eligible, checkpoints, families, counts, keyed,
                                           predictions, weighted=True)
        self.assertTrue(np.allclose(equal[:, :2], [[2.0, 0.0], [0.0, 2.0]]))
        self.assertAlmostEqual(weighted[0, 0], 0.775)
        self.assertAlmostEqual(weighted[1, 1], 0.7)


if __name__ == "__main__":
    unittest.main()
