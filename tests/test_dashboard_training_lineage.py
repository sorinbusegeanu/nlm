import json
import tempfile
import unittest
from pathlib import Path

from dashboard.training_lineage import training_lineage


class TrainingLineageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
