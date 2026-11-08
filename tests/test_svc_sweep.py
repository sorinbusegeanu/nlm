import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.svc_sweep import evaluate_svc_sweep, run_svc_sweep


class SVCSweepTests(unittest.TestCase):
    def _candidate(self, root, candidate_id, model, dataset, labels, groups, probs, c=None):
        prediction_path = root / f"{candidate_id}_validation.parquet"
        frame = pd.DataFrame({"id": [f"row-{i:04d}" for i in range(len(labels))],
                              "prompt_id": groups, "y_true": labels})
        for index, label in enumerate(sorted(set(labels))):
            frame[f"prob::{label}"] = probs[:, index]
        frame.to_parquet(prediction_path, index=False)
        config = {"features": {"views": ["char", "word"]}, "calibration": "sigmoid",
                  "calibration_cv": 3}
        if c is not None:
            config["C"] = c
        manifest = {"experiment_id": candidate_id, "model": model, "dataset": dataset,
                    "seed": 2026, "split": "validation", "prediction_path": str(prediction_path),
                    "config": config, "metrics": {"balanced_accuracy": 0.4}}
        manifest_path = root / f"{candidate_id}_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

    def test_variants_replace_only_champion_svc_and_write_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / "artifacts"
            prediction_dir = artifact_dir / "predictions"
            prediction_dir.mkdir(parents=True)
            registry_path = root / "ensembles.md"
            labels = np.repeat([f"class-{i:02d}" for i in range(18)], 30)
            groups = np.repeat([f"prompt-{i:02d}" for i in range(30)], 18)
            n, k = len(labels), len(set(labels))
            y = np.asarray([int(value[-2:]) for value in labels])
            probabilities = np.full((n, k), 0.01 / (k - 1), dtype=float)
            probabilities[np.arange(n), y] = 0.99
            member_ids = [f"model{i}_v0" for i in range(8)] + ["svc_v1"]
            for candidate_id in member_ids[:8]:
                self._candidate(prediction_dir, candidate_id, candidate_id.split("_")[0],
                                "v0", labels, groups, probabilities)
            self._candidate(prediction_dir, "svc_v1", "svc", "v1", labels, groups,
                            probabilities, c=0.5)
            variant_ids = []
            for c in (0.1, 2.0, 5.0):
                candidate_id = f"svc_v1_1_c{str(c).replace('.', '_')}"
                variant_ids.append(candidate_id)
                variant_probabilities = probabilities.copy()
                # Each variant has a deterministic, slightly different set of mistakes.
                mask = np.arange(n) % (int(c * 10) + 2) == 0
                variant_probabilities[mask] = 1.0 / k
                self._candidate(prediction_dir, candidate_id, "svc", "v1.1", labels, groups,
                                variant_probabilities, c=c)
            entries = [{"version": "v6", "prediction_experiment_id": "prediction-v6",
                        "selected_strategy": "equal_probability_blend",
                        "selected_members": [{"candidate_id": candidate_id} for candidate_id in member_ids],
                        "selected_weights": {candidate_id: 1.0 / 9 for candidate_id in member_ids}}]
            from src.ensemble_registry import write_ensemble_registry
            write_ensemble_registry(entries, registry_path)

            report = evaluate_svc_sweep(artifact_dir, registry_path=registry_path,
                                        output_dir=root / "sweep", variant_ids=variant_ids,
                                        baseline_id=variant_ids[0])
            self.assertEqual(report["validation_rows"], 540)
            self.assertEqual(report["champion_metrics"]["accuracy"], 1.0)
            self.assertEqual(report["champion_members"], member_ids)
            self.assertEqual(len(report["candidates"]), 4)
            self.assertGreaterEqual(report["v1_1_c_0_5_reference"]["standalone_metrics"]["accuracy"], 0.0)
            self.assertEqual(sum(bool(row["is_reference_baseline"]) for row in report["candidates"]), 1)
            self.assertTrue((root / "sweep" / "results.csv").is_file())
            self.assertTrue((root / "sweep" / "validation_predictions.parquet").is_file())

    def test_runner_disables_automatic_refresh_for_each_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "sweep"
            manifests = Path(directory) / "predictions"
            manifests.mkdir()
            (manifests / "svc_v1_1_c0_5_manifest.json").write_text(json.dumps({
                "experiment_id": "svc_v1_1_c0_5", "model": "svc", "dataset": "v1.1",
                "seed": 2026, "split": "validation", "config": {"C": 0.5,
                    "features": {"views": ["char", "word"]}, "calibration": "sigmoid",
                    "calibration_cv": 3}}))
            with patch("src.svc_sweep.train_models", side_effect=[
                     {"experiment_id": "svc-c1", "config": {"C": 0.1}, "dataset": "v1.1", "seed": 2026, "split": "validation"},
                     {"experiment_id": "svc-c2", "config": {"C": 2.0}, "dataset": "v1.1", "seed": 2026, "split": "validation"},
                     {"experiment_id": "svc-c3", "config": {"C": 5.0}, "dataset": "v1.1", "seed": 2026, "split": "validation"},
                 ]) as train, \
                 patch("src.svc_sweep.evaluate_svc_sweep", return_value={"experiment": "test"}):
                run_svc_sweep(artifact_dir=directory, registry_path=str(Path(directory) / "registry.sqlite"),
                              ensemble_registry=str(Path(directory) / "ensembles.md"),
                              output_dir=output_dir)
            self.assertEqual(train.call_count, 3)
            self.assertTrue(all(call.kwargs["refresh_prediction"] is False
                                for call in train.call_args_list))


if __name__ == "__main__":
    unittest.main()
