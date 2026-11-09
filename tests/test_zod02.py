import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.zod02 import _select_marginal_rules, run_zod02


def _candidate(candidate_id, labels, classes):
    probs = np.zeros((len(labels), len(classes)), dtype=float)
    lookup = {label: index for index, label in enumerate(classes)}
    probs[np.arange(len(labels)), [lookup[label] for label in labels]] = 1.0
    return {"candidate_id": candidate_id, "probabilities": probs}


class Zod02SelectionTests(unittest.TestCase):
    def test_multiple_specialists_for_one_label_can_expand_recall(self):
        classes = ["a", "b"]
        y = np.asarray(["a", "a", "b", "b"])
        fallback = _candidate("fallback", ["b", "b", "b", "b"], classes)
        first = _candidate("first", ["a", "b", "b", "b"], classes)
        second = _candidate("second", ["b", "a", "b", "b"], classes)
        selected = _select_marginal_rules([fallback, first, second], y, classes,
            fallback_id="fallback", candidate_order=["fallback", "first", "second"],
            minimum_support=1)
        a_rules = [rule for rule in selected["selected_rules"] if rule["label"] == "a"]
        self.assertEqual([rule["candidate_id"] for rule in a_rules], ["first", "second"])
        self.assertEqual(selected["fit_metrics"]["per_label"]["a"]["recall"], 1.0)
        self.assertEqual(selected["fit_metrics"]["balanced_accuracy"], 1.0)

    def test_redundant_rule_with_no_new_rows_is_rejected(self):
        classes = ["a", "b"]
        y = np.asarray(["a", "a", "b", "b"])
        fallback = _candidate("fallback", ["b", "b", "b", "b"], classes)
        first = _candidate("first", ["a", "b", "b", "b"], classes)
        duplicate = _candidate("duplicate", ["a", "b", "b", "b"], classes)
        selected = _select_marginal_rules([fallback, first, duplicate], y, classes,
            fallback_id="fallback", candidate_order=["fallback", "first", "duplicate"],
            minimum_support=1)
        duplicate_a = next(row for row in selected["audit"]
                           if row["candidate_id"] == "duplicate" and row["label"] == "a")
        self.assertFalse(duplicate_a["accepted"])
        self.assertEqual(duplicate_a["decision_reason"], "no_unassigned_matches")

    def test_grouped_oof_and_full_fit_artifacts_are_written(self):
        classes = ["a", "b"]
        groups = np.repeat([f"p{i}" for i in range(5)], 2)
        y = np.tile(["a", "b"], 5)
        fallback = _candidate("fallback", ["b", "b"] * 5, classes)
        first = _candidate("first", ["a", "b"] * 5, classes)
        second = _candidate("second", ["b", "b"] * 5, classes)
        candidates = [fallback, first, second]
        for candidate in candidates:
            candidate["classes"] = classes
        with tempfile.TemporaryDirectory() as temp:
            result = run_zod02(candidates=candidates, y_true=y, groups=groups,
                classes=classes, fallback_id="fallback",
                zod01_rules=[("a", "first", "first")],
                validation_ids=[f"r{i}" for i in range(len(y))], candidate_order=[
                    "fallback", "first", "second"], folds=5, minimum_support=1,
                output_dir=temp, report_path=str(Path(temp) / "report.md"),
                validation_split_sha256="split-hash")
            metrics = result["metrics"]
            self.assertEqual(metrics["validation_rows"], 10)
            self.assertEqual(len(metrics["folds_summary"]), 5)
            self.assertTrue((Path(temp) / "oof_predictions.parquet").is_file())
            self.assertTrue((Path(temp) / "selected_rules_full_v0.csv").is_file())
            self.assertTrue((Path(temp) / "corrected_regressed.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
