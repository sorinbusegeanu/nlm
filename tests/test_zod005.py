import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.zod005 import _apply_rules, _load_predictions, _sort_rules


def row(seq, candidate, label, precision, support):
    return {
        "SeqID": seq, "sequence_id": f"N{seq:04d}", "experiment_id": candidate,
        "model": "fixture", "training_dataset": "v1", "target_label": label,
        "precision": precision, "prediction_count": support,
    }


class ZOD005Test(unittest.TestCase):
    def test_eligibility_and_precision_support_order(self):
        ranked = _sort_rules(pd.DataFrame([
            row(4, "support", "a", .95, 32),
            row(3, "precision", "b", .96, 30),
            row(2, "below_precision", "a", .899, 100),
            row(1, "below_support", "b", .99, 29),
            row(5, "tie", "c", .96, 30),
        ]))
        self.assertEqual(ranked.experiment_id.tolist(), ["precision", "tie", "support"])

    def test_rules_claim_only_unassigned_rows_and_fallback_fills_rest(self):
        classes = ["a", "b"]
        rules = _sort_rules(pd.DataFrame([
            row(1, "first", "a", .99, 50),
            row(2, "second", "b", .95, 40),
        ]))
        prediction, source, assigned, trace, fallback = _apply_rules(
            np.array(["a", "a", "b", "b"]), classes,
            {"first": np.array(["a", "a", "a", "b"]),
             "second": np.array(["b", "b", "b", "b"]),
             "fallback": np.array(["b", "a", "a", "a"])},
            rules, "fallback")
        self.assertEqual(prediction.tolist(), ["a", "a", "a", "b"])
        self.assertEqual(source.tolist(), ["first", "first", "first", "second"])
        self.assertEqual(assigned.tolist(), [True, True, True, True])
        self.assertEqual(fallback["assigned"], 0)
        self.assertEqual([item["assigned"] for item in trace], [3, 1])

    def test_rules_can_leave_rows_for_fallback(self):
        rules = _sort_rules(pd.DataFrame([row(1, "specialist", "a", .91, 30)]))
        prediction, source, assigned, _, fallback = _apply_rules(
            np.array(["a", "b"]), ["a", "b"],
            {"specialist": np.array(["b", "b"]), "fallback": np.array(["a", "b"])},
            rules, "fallback")
        self.assertEqual(prediction.tolist(), ["a", "b"])
        self.assertEqual(source.tolist(), ["fallback", "fallback"])
        self.assertEqual(assigned.tolist(), [False, False])
        self.assertEqual(fallback["correct"], 2)

    def test_prediction_cache_is_joined_by_id_and_checks_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            cache_id = "checkpoint-a"
            digest = "v0-hash"
            trust = pd.DataFrame([{"experiment_id": cache_id,
                                   "validation_split_sha256": digest}])
            validation = pd.DataFrame({"id": ["r1", "r2"], "y_true": ["a", "b"]})
            cache_path = cache_dir / f"{cache_id}.parquet"
            pd.DataFrame({"id": ["r2", "r1"], "y_true": ["b", "a"],
                          "y_pred": ["a", "b"]}).to_parquet(cache_path, index=False)
            cache_path.with_suffix(".json").write_text(json.dumps({
                "cache_key": {"experiment_id": cache_id, "v0_sha256": digest}, "rows": 2,
            }))
            predictions, status, _ = _load_predictions(trust, validation, cache_dir,
                                                       infer_missing=False)
            self.assertEqual(predictions[cache_id].tolist(), ["b", "a"])
            self.assertEqual(status[cache_id], "reused")
            pd.DataFrame({"id": ["r2", "r1"], "y_true": ["a", "b"],
                          "y_pred": ["a", "b"]}).to_parquet(cache_path, index=False)
            with self.assertRaisesRegex(ValueError, "label alignment mismatch"):
                _load_predictions(trust, validation, cache_dir, infer_missing=False)


if __name__ == "__main__":
    unittest.main()
