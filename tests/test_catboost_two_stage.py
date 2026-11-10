import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.catboost_two_stage import (
    EXPERIMENTS,
    _aggregate_feature_importance,
    _pruned_recipe,
    _require_same_validation,
    _select_matrix,
    _stage_artifact_dir,
    _stage_id,
)
from src.feature_factory import BASE_STYLOMETRY_GROUPS, STYLOMETRY_GROUPS, stylometry
from src.inference import predict_catboost_stylometry
from src.model_factory import train_catboost_stylometry


class CatBoostTwoStageTests(unittest.TestCase):
    def test_catboost_init_model_appends_trees_and_preserves_parent_prefix(self):
        classes = ["a", "b", "c"]
        stage1_text, stage1_labels, stage2_text, stage2_labels = [], [], [], []
        for label, token in zip(classes, ("amber", "birch", "copper")):
            for i in range(9):
                stage1_text.append(f"{token} style sample with punctuation, number {i}.")
                stage1_labels.append(label)
                stage2_text.append(f"{token} second stage response; detail {i}!")
                stage2_labels.append(label)
        valid_text = ["amber validation response.", "birch validation response.", "copper validation response."]
        recipe = {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": ["P", "F"]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = train_catboost_stylometry(
                stage1_text, stage1_labels, valid_text, classes,
                output_dir=root / "stage1", iterations=3, depth=2,
                learning_rate=.05, l2_leaf_reg=5, thread_count=1, seed=2026,
                feature_recipe=recipe, validation_labels=None)
            second = train_catboost_stylometry(
                stage2_text, stage2_labels, valid_text, classes,
                output_dir=root / "stage2", iterations=2, depth=2,
                learning_rate=.05, l2_leaf_reg=5, thread_count=1, seed=2026,
                feature_recipe=recipe, validation_labels=None,
                init_model=root / "stage1" / "model.cbm")
            self.assertEqual(first["tree_count"], 3)
            self.assertEqual(second["tree_count"], 5)
            prefix1 = first["model"].predict(
                stylometry(stage1_text, recipe), prediction_type="RawFormulaVal")
            prefix2 = second["model"].predict(
                stylometry(stage1_text, recipe), prediction_type="RawFormulaVal", ntree_end=3)
            np.testing.assert_allclose(prefix2, prefix1, atol=1e-8, rtol=1e-8)
            reloaded, _ = predict_catboost_stylometry(
                root / "stage2", valid_text, classes,
                temperature=second["calibration_temperature"])
            np.testing.assert_allclose(reloaded, second["probabilities"], atol=1e-7, rtol=1e-6)

    def test_stage_recipes_have_matching_feature_schemas(self):
        full_recipe = {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": list(STYLOMETRY_GROUPS)}
        texts = ["A sample response, with a sentence.", "Another sample with style!"]
        full, all_names = stylometry(texts, full_recipe, return_feature_names=True)
        recipes = {
            "A": EXPERIMENTS["A"]["recipe"],
            "B": EXPERIMENTS["B"]["recipe"],
            "C": {"baseline": list(BASE_STYLOMETRY_GROUPS),
                  "groups": [g for g in STYLOMETRY_GROUPS if g not in {"L", "V"}]},
        }
        for recipe in recipes.values():
            _, names = _select_matrix(full, all_names, recipe)
            _, direct_names = stylometry(texts, recipe, return_feature_names=True)
            self.assertEqual(names, direct_names)
            self.assertEqual(len(names), len(set(names)))

    def test_c_recipe_removes_lowest_eligible_means_but_keeps_p_and_f(self):
        importance = {group: {"feature_count": 1, "mean_importance": float(i),
                              "total_importance": float(i)}
                      for i, group in enumerate(STYLOMETRY_GROUPS)}
        importance["V"]["mean_importance"] = importance["L"]["mean_importance"]
        recipe, removed = _pruned_recipe(importance)
        self.assertEqual(removed, ["L", "V"])
        self.assertIn("P", recipe["groups"])
        self.assertIn("F", recipe["groups"])
        self.assertEqual(len(recipe["groups"]), len(STYLOMETRY_GROUPS) - 2)

    def test_importance_aggregates_mean_by_feature_group(self):
        names = ["char_count", "word_count", "P_char_00", "P_char_01", "F_as"]
        result = _aggregate_feature_importance(names, [2, 4, 1, 3, 5])
        self.assertEqual(result["length"]["feature_count"], 2)
        self.assertEqual(result["length"]["mean_importance"], 3)
        self.assertEqual(result["P"]["mean_importance"], 2)
        self.assertEqual(result["F"]["mean_importance"], 5)

    def test_same_v0_validation_contract_and_six_independent_artifact_paths(self):
        one = pd.DataFrame({"id": ["a", "b"], "model": ["x", "y"]})
        two = one.iloc[::-1].reset_index(drop=True)
        _require_same_validation(one, two, ["x", "y"], ["x", "y"])
        with self.assertRaises(ValueError):
            _require_same_validation(one, one.assign(id=["a", "c"]), ["x", "y"], ["x", "y"])
        paths = [_stage_artifact_dir("artifacts/runs/run", exp, stage)
                 for exp in ("A", "B", "C") for stage in ("stage1", "stage2")]
        ids = [_stage_id("run", exp, stage)
               for exp in ("A", "B", "C") for stage in ("stage1", "stage2")]
        self.assertEqual(len(set(paths)), 6)
        self.assertEqual(len(set(ids)), 6)

    def test_two_stage_fit_uses_validation_labels_for_balanced_early_stopping(self):
        from src.catboost_two_stage import _save_stage

        classes = ["a", "b"]
        recipe = EXPERIMENTS["A"]["recipe"]
        train_text = ["amber alpha", "birch beta"]
        valid = pd.DataFrame({"id": ["v1", "v2"], "prompt_id": ["p1", "p2"],
                              "text": ["amber validation", "birch validation"],
                              "model": classes})
        train = pd.DataFrame({"id": ["t1", "t2"], "prompt_id": ["tp1", "tp2"], "text": train_text, "model": classes})
        full, names = stylometry(train_text + valid.text.tolist(),
            {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": list(STYLOMETRY_GROUPS)},
            return_feature_names=True)
        result = {"scores": np.array([[2., 0.], [0., 2.]]),
                  "probabilities": np.array([[.9, .1], [.1, .9]]),
                  "model": None, "fit_seconds": 0.1, "inference_seconds": 0.01,
                  "calibration_temperature": 1.0, "calibration_method": "train-only scalar temperature",
                  "feature_names": _select_matrix(full, names, recipe)[1], "tree_count": 1}

        class FakeModel:
            tree_count_ = 1

        result["model"] = FakeModel()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "stage1" / "checkpoint"
            def fake_train(*args, **kwargs):
                self.assertEqual(kwargs["validation_labels"], classes)
                self.assertEqual(kwargs["checkpoint_selection_metric"], "balanced_accuracy")
                checkpoint.mkdir(parents=True, exist_ok=True)
                (checkpoint / "model.cbm").write_bytes(b"fake")
                (checkpoint / "stylometry.json").write_text('{"tree_count":1}\n')
                return result
            with patch("src.catboost_two_stage.train_catboost_stylometry", side_effect=fake_train), \
                 patch("src.catboost_two_stage.Registry"):
                _save_stage(experiment="A", experiment_name="P+F", stage="stage1",
                    dataset_name="v1_all", train=train, valid=valid, classes=classes,
                    recipe=recipe, iterations=1,
                    cfg={"depth": 2, "learning_rate": .05, "l2_leaf_reg": 5,
                         "thread_count": 1, "temperature_calibration": True}, seed=2026,
                    checkpoint=checkpoint, parent_checkpoint=None, parent_sha256=None,
                    matrices={"full": full, "all_names": names,
                              "offsets": {"v1_all": (0, len(train)), "v0": (len(train), len(train)),
                                          "val": (len(train), len(train) + len(valid))}},
                    report_dir=root / "run", data_profile={"dataset_sha256": "data", "split_sha256": "split"},
                    registry_path=root / "registry.sqlite", code_revision="test",
                    validation_dataset="val")


if __name__ == "__main__":
    unittest.main()
