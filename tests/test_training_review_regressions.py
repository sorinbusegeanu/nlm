"""Regression tests for training review findings using only temporary fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import pipeline


class TrainingReviewRegressionTests(unittest.TestCase):
    def test_restart_prevalidates_then_uses_disjoint_historical_v0_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "data").mkdir()
            (root / "configs/models").mkdir(parents=True)
            (root / "configs/features").mkdir(parents=True)
            (root / "configs/data").mkdir(parents=True)
            rows = []
            for label, prompt_prefix in (("a", "pa"), ("b", "pb")):
                for index in range(10):
                    rows.append({"id": f"{label}{index}", "prompt_id": f"{prompt_prefix}{index}",
                                 "text": f"response {label} {index}", "model": label})
            pd.DataFrame(rows).to_csv(root / "data/train.csv", index=False)
            (root / "configs/data/v0.yaml").write_text(
                "train_path: data/train.csv\noutput_dir: data/processed\n"
                "validation_fraction: 0.2\nsplit_candidates: 50\nexpected_classes: 2\n")
            (root / "configs/models/svc.yaml").write_text(
                "model: svc\nC: 1.0\ncalibration_cv: 2\nepochs: 1\n")
            (root / "configs/features/default.yaml").write_text("views: [char]\n")
            (root / "data/processed").mkdir(parents=True)
            (root / "data/processed/old.txt").write_text("old generated state")
            (root / "artifacts").mkdir()
            (root / "artifacts/old.txt").write_text("old artifact state")
            observed = {}

            def build_features(dataset, data_dir, config_path, cache_dir, *, validation_dataset):
                observed["features"] = pipeline._training_partitions(
                    dataset, validation_dataset, data_dir)
                return {"ok": True}

            def train_models(dataset, model, data_dir, artifact_dir, registry, **kwargs):
                observed["training"] = pipeline._training_partitions(
                    dataset, kwargs["validation_dataset"], data_dir)
                return {"fit_stubbed": True}

            with patch.object(pipeline, "__file__", str(root / "src/pipeline.py")), \
                 patch.dict(pipeline.MODEL_CONFIGS, {"svc": "configs/models/svc.yaml"}), \
                 patch.object(pipeline, "build_features", side_effect=build_features), \
                 patch.object(pipeline, "train_models", side_effect=train_models), \
                 patch.object(pipeline, "evaluate_ensembles", return_value={"ok": True}):
                result = pipeline.restart_mvp(models=["svc"], seed=27,
                    bootstrap_replicates=1, data_config="configs/data/v0.yaml",
                    feature_config="configs/features/default.yaml")

            self.assertTrue(result["models"][0]["fit_stubbed"])
            train, valid = observed["training"]
            self.assertEqual(set(train.id) | set(valid.id), {row["id"] for row in rows})
            self.assertFalse(set(train.id) & set(valid.id))
            self.assertFalse(set(train.prompt_id) & set(valid.prompt_id))
            self.assertEqual(set(train.id), set(observed["features"][0].id))
            self.assertEqual(set(valid.id), set(observed["features"][1].id))


if __name__ == "__main__":
    unittest.main()
