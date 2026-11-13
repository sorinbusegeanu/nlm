import json
import tempfile
import unittest
from pathlib import Path

from dashboard.training_lineage import training_lineage, trainer_completion
from dashboard.training_config import live_training_config


class TrainingLineageTests(unittest.TestCase):
    def test_live_continuation_uses_config_file_unless_cli_overrides_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "stage2.yaml"
            config_path.write_text(
                "model: ultrabert380\nepochs: 20\nlearning_rate: 8.0e-6\n"
                "learning_rate_decay: null\nlr_scheduler_type: linear\n")
            options = {"--config": str(config_path), "--dataset": "v2.0",
                       "--validation-dataset": "v0"}
            config = live_training_config(root, "continue-encoder", "ultrabert380", options)
            self.assertEqual(config["learning_rate"], 8e-6)
            self.assertIsNone(config["learning_rate_decay"])
            self.assertEqual(config["epochs"], 20)
            self.assertEqual(config["lr_scheduler_type"], "linear")
            self.assertEqual(config["training_data"], "v2.0")

            options.update({"--learning-rate": "1e-5", "--epochs": "6"})
            overridden = live_training_config(root, "continue-encoder", "ultrabert380", options)
            self.assertEqual(overridden["learning_rate"], 1e-5)
            self.assertEqual(overridden["epochs"], 6)

    def test_training_tables_show_raw_loss_without_calibration_columns(self):
        dashboard = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        source = dashboard.read_text()
        self.assertIn('"Validation loss (raw)"', source)
        self.assertNotIn("Validation loss (calibrated)", source)
        self.assertNotIn("Calibration temperature", source)

    def test_continuation_shows_source_then_target_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "source_manifest_manifest.json").write_text(json.dumps({
                "experiment_id": "source_manifest", "dataset": "v1_all",
            }))
            manifest = {
                "experiment_id": "continued_run", "dataset": "v0",
                "continued_from_experiment": "source_manifest",
            }
            self.assertEqual(training_lineage(manifest, tmp), "V1-all → V0")

    def test_regular_run_shows_its_dataset(self):
        self.assertEqual(training_lineage({"dataset": "v1.1"}, "."), "V1.1")

    def test_explicit_staged_sequence_takes_precedence(self):
        manifest = {
            "dataset": "v0", "continued_from_experiment": "ignored_source",
            "training_sequence": ["v1_all", "v0"],
        }
        self.assertEqual(training_lineage(manifest, "."), "V1-all → V0")

    def test_missing_source_manifest_is_not_misreported_as_target_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"dataset": "v0", "continued_from_experiment": "missing_source"}
            self.assertEqual(training_lineage(manifest, tmp), "continued → V0")

    def test_recursive_continuation_keeps_full_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "first_manifest.json").write_text(json.dumps({
                "experiment_id": "first", "dataset": "v1_all",
            }))
            Path(tmp, "second_manifest.json").write_text(json.dumps({
                "experiment_id": "second", "dataset": "v0",
                "continued_from_experiment": "first",
            }))
            current = {"experiment_id": "third", "dataset": "v0",
                       "continued_from_experiment": "second"}
            self.assertEqual(training_lineage(current, tmp), "V1-all → V0 → V0")

    def test_trainer_completion_distinguishes_postprocessing_from_next_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp, "training_progress.json")
            progress.write_text("{}")
            log = Path(tmp, "training_log.json")
            log.write_text(json.dumps([
                {"eval_accuracy": 0.5, "epoch": 6.0},
                {"train_runtime": 26986.8, "epoch": 6.0},
            ]))
            self.assertEqual(trainer_completion(progress), {"finished": True, "epoch": 6})

    def test_active_training_log_without_runtime_is_not_marked_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp, "training_progress.json")
            progress.write_text("{}")
            Path(tmp, "training_log.json").write_text(json.dumps([
                {"eval_accuracy": 0.5, "epoch": 6.0},
            ]))
            self.assertEqual(trainer_completion(progress), {"finished": False, "epoch": None})


if __name__ == "__main__":
    unittest.main()
