import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ensemble_registry import (read_active_roster, read_ensemble_registry,
                                   refresh_prediction_factory, write_active_roster,
                                   write_ensemble_registry)
from src.inference import package_prediction_bundle
from src.prediction_factory import evaluate_prediction_strategies


class EnsembleRefreshRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pred = self.root / "predictions"
        self.pred.mkdir()
        self.roster = self.pred / "active_model_roster.json"
        self.registry = self.root / "ensembles.md"
        self._candidate("old", "svc", .99, .7)
        self._candidate("new", "svc", .55, .8)
        self._candidate("other", "catboost_stylometry", .001, .1)
        self.champion = {"version": "v1", "candidate_roster_hash": "old",
            "selected_strategy": "optimized_probability_blend",
            "selected_members": [{"candidate_id": "old"}, {"candidate_id": "other"}],
            "selected_weights": {"old": .9, "other": .1}, "submitted_balanced_accuracy": .55}
        write_ensemble_registry([self.champion], self.registry,
            metadata={"last_recorded_active_roster_hash": "old", "optimizer_protocol": "forward_group_cv_weighted_v1"})
        write_active_roster({"roster_hash": "old", "candidates": [
            {"model": "svc", "candidate_id": "old"},
            {"model": "catboost_stylometry", "candidate_id": "other"}]}, self.roster)

    def _candidate(self, name, model, true_probability, score):
        labels = np.array(["a", "b"] * 4)
        frame = pd.DataFrame({"id": [str(i) for i in range(8)],
            "prompt_id": [f"p{i//2}" for i in range(8)], "y_true": labels,
            "prob::a": np.where(labels == "a", true_probability, 1-true_probability),
            "prob::b": np.where(labels == "b", true_probability, 1-true_probability)})
        prediction = self.pred / f"{name}.parquet"
        frame.to_parquet(prediction, index=False)
        checkpoint = self.root / f"{name}.bin"
        checkpoint.write_bytes(b"fake checkpoint")
        manifest = {"model": model, "dataset": "v0", "split": "validation",
            "experiment_id": name, "prediction_path": str(prediction),
            "checkpoint_path": str(checkpoint), "metrics": {"balanced_accuracy": score}}
        (self.pred / f"{name}_manifest.json").write_text(json.dumps(manifest))

    def _evaluate(self, ids, champion=None):
        return evaluate_prediction_strategies(self.root, candidate_ids=ids,
                                              previous_champion=champion, group_folds=2)

    def test_real_packaging_equal_and_weighted_optimizer_reports(self):
        equal = self._evaluate(["new"])
        weighted = self._evaluate(["new", "other"], self.champion)
        self.assertEqual(equal["selected_strategy"], "equal_probability_blend")
        self.assertNotIn("previous_champion", equal["strategy_metrics"])
        self.assertEqual(weighted["selected_strategy"], "optimized_probability_blend")
        for report in (equal, weighted):
            self.assertNotIn("reliability_rule", report)
            target = self.root / (report["selected_strategy"] + ".zip")
            package_prediction_bundle(report["manifest_path"], target)
            with zipfile.ZipFile(target) as archive:
                bundle = json.loads(archive.read("prediction_bundle.json"))
            self.assertEqual(bundle["selected_strategy"], report["selected_strategy"])
            self.assertEqual(bundle["selected_weights"], report["selected_weights"])

    def test_weighted_champion_and_report_round_trip(self):
        report = self._evaluate(["new", "other"], self.champion)
        all_weighted = report["strategy_metrics"]["all_candidate_weighted_blend"]
        self.assertEqual(all_weighted["ensemble_size"], 2)
        self.assertEqual(set(report["all_candidate_weights"]), {"new", "other"})
        self.assertAlmostEqual(sum(report["all_candidate_weights"].values()), 1.0, places=6)
        self.assertIn("all_candidate_weighted_blend", report["full_data_strategy_metrics_exploratory"])
        self.assertEqual(report["selected_comparison"], "previous_champion")
        self.assertEqual(report["selected_members"], ["old", "other"])
        self.assertEqual(report["selected_candidate_ids"], report["selected_members"])
        self.assertEqual(report["selected_weights"], {"old": .9, "other": .1})
        self.assertEqual(report["selected_validation_metric_scope"], "prompt_group_out_of_fold")
        self.assertEqual(report["selected_validation_accuracy"],
                         report["strategy_metrics"][report["selected_comparison"]]["accuracy"])
        self.assertNotEqual(report["forward_selected_candidate_ids"], report["selected_members"])
        self.assertEqual(report["full_data_strategy_metrics_exploratory"]["previous_champion"]["accuracy"], 1.0)
        self.assertLess((.99 + .001) / 2, .5)  # Equal averaging predicts the wrong class.
        again = self._evaluate(["new", "other"], report)
        self.assertEqual(again["selected_members"], report["selected_members"])
        self.assertEqual(again["selected_weights"], report["selected_weights"])
        changed = self._evaluate(["new", "other"], {**self.champion, "selected_weights": {"old": .8, "other": .2}})
        self.assertNotEqual(changed["manifest_path"], report["manifest_path"])
        invalid = {**self.champion, "selected_weights": {"old": .9}}
        with self.assertRaisesRegex(ValueError, "requires a weight"):
            self._evaluate(["new", "other"], invalid)
        with self.assertRaisesRegex(ValueError, "invalid or unnormalized"):
            self._evaluate(["new", "other"], {**self.champion, "selected_weights": {"old": .9, "other": .2}})
        with self.assertRaisesRegex(ValueError, "lacks its executable strategy"):
            self._evaluate(["new", "other"], {**self.champion, "selected_strategy": "previous_champion"})

    def test_refresh_replaces_active_checkpoint_but_keeps_champion(self):
        result = refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry,
                                            roster_path=self.roster)
        self.assertEqual(result["selected_members"], ["old", "other"])
        active = read_active_roster(self.roster)
        self.assertEqual({row["candidate_id"] for row in active["candidates"]}, {"new", "other"})
        history = read_ensemble_registry(self.registry)["ensembles"]
        self.assertEqual(history[0]["submitted_balanced_accuracy"], .55)
        self.assertEqual([row["candidate_id"] for row in history[-1]["selected_members"]], ["old", "other"])
        refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry, roster_path=self.roster)
        self.assertEqual(len(read_ensemble_registry(self.registry)["ensembles"]), len(history))

    def test_inconsistent_prompt_groups_are_rejected(self):
        path = self.pred / "other.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "prompt_id"] = "different"
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "prompt groups"):
            self._evaluate(["new", "other"])

    def test_unrelated_splits_and_classes_do_not_replace_active_reference(self):
        self._candidate("unrelated", "aaa", .6, .2)
        path = self.pred / "unrelated.parquet"
        frame = pd.read_parquet(path)
        frame["id"] = "other-" + frame.id
        frame["prob::c"] = 0.0
        frame.to_parquet(path, index=False)
        result = refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry,
                                            roster_path=self.roster)
        self.assertEqual(result["selected_members"], ["old", "other"])
        self.assertIn("unrelated", [item["candidate_id"] for item in result["excluded_candidates"]])

    def test_validation_labels_change_report_identity(self):
        before = self._evaluate(["new"])
        for path in self.pred.glob("*.parquet"):
            frame = pd.read_parquet(path)
            frame["y_true"] = frame.y_true.map({"a": "b", "b": "a"})
            frame.to_parquet(path, index=False)
        after = self._evaluate(["new"])
        self.assertNotEqual(before["manifest_path"], after["manifest_path"])
        self.assertNotEqual(before["selected_validation_accuracy"], after["selected_validation_accuracy"])

    def test_failed_refresh_does_not_write_roster(self):
        before = self.roster.read_bytes()
        with patch("src.prediction_factory.evaluate_prediction_strategies", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry,
                                           roster_path=self.roster)
        self.assertEqual(self.roster.read_bytes(), before)

    def test_failed_roster_write_restores_registry(self):
        before_registry = self.registry.read_bytes()
        before_roster = self.roster.read_bytes()
        with patch("src.ensemble_registry.write_active_roster", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry,
                                           roster_path=self.roster)
        self.assertEqual(self.registry.read_bytes(), before_registry)
        self.assertEqual(self.roster.read_bytes(), before_roster)

    def test_changed_winner_creates_version_without_rewriting_history(self):
        refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry,
                                   roster_path=self.roster)
        before = read_ensemble_registry(self.registry)["ensembles"]
        replacement = self._evaluate(["new"])
        with patch("src.prediction_factory.evaluate_prediction_strategies", return_value=replacement):
            result = refresh_prediction_factory(artifact_dir=self.root, registry_path=self.registry,
                                                roster_path=self.roster)
        after = read_ensemble_registry(self.registry)["ensembles"]
        self.assertEqual(result["ensemble_version"], "v3")
        self.assertEqual(after[:2], before)
        self.assertEqual(after[0]["submitted_balanced_accuracy"], .55)
        self.assertEqual(after[-1]["selected_members"][0]["candidate_id"], "new")


if __name__ == "__main__":
    unittest.main()
