import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import v2_checkpoint_trust as trust


class V2CheckpointTrustTests(unittest.TestCase):
    def test_publish_versions_atomically_and_latest_resolves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {column: "" for column in trust.TRUST_COLUMNS}
            row.update({"sequence_id": "N0001", "experiment_id": "exp", "target_label": trust.LABELS[0],
                        "precision": 1.0, "recall": 1.0, "prediction_count": 1,
                        "true_positive": 1, "false_positive": 0, "false_negative": 0,
                        "evaluation_eligible": True, "validation_training_overlap": 0,
                        "checkpoint_available": True})
            frame = pd.DataFrame([row]); frame.attrs["trust_list_version"] = "trust-list-2-v001"
            metadata = {"generated_at_utc": trust._utc_now()}
            trust._publish_version(root, frame, metadata, "trust-list-2-v001")
            first_csv = root / "docs/trust-list-2-v001.csv"
            before = first_csv.read_bytes()
            trust._publish_version(root, frame, {"generated_at_utc": trust._utc_now()}, "trust-list-2-v002")
            self.assertEqual(first_csv.read_bytes(), before)
            csv_path, metadata_path, loaded = trust.latest_trust_paths(root)
            self.assertEqual(csv_path.name, "trust-list-2-v002.csv")
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(loaded["version"], "trust-list-2-v002")
            self.assertEqual(pd.read_csv(csv_path).trust_list_version.unique().tolist(), ["trust-list-2-v002"])

    def test_append_training_run_creates_new_version_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "docs").mkdir()
            roster = pd.DataFrame({"model": ["qwen06"]})
            roster.to_csv(root / trust.ROSTER_PATH, index=False)
            row = {column: "" for column in trust.TRUST_COLUMNS}
            row.update({"sequence_id": "N0001", "experiment_id": "existing", "target_label": trust.LABELS[0],
                        "precision": 1.0, "recall": 1.0, "prediction_count": 1,
                        "true_positive": 1, "false_positive": 0, "false_negative": 0,
                        "evaluation_eligible": True, "validation_training_overlap": 0,
                        "checkpoint_available": True})
            initial = pd.DataFrame([row]); initial.attrs["trust_list_version"] = "trust-list-2-v001"
            trust._publish_version(root, initial, {"generated_at_utc": trust._utc_now()}, "trust-list-2-v001")
            manifest_path = root / "artifacts/predictions/new_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}")
            manifest = {"model": "qwen06", "dataset": "v1", "validation_dataset": "v0",
                        "experiment_id": "new", "checkpoint_path": "missing", "validation_rows": 2699}
            fake_rows = []
            for label in trust.LABELS:
                r = dict(row); r.update({"sequence_id": "N0002", "experiment_id": "new", "target_label": label})
                fake_rows.append(r)
            with patch.object(trust, "_rows_from_training_manifest", return_value=fake_rows):
                result = trust.append_training_run(manifest, manifest_path, root=root,
                                                   roster_path=root / trust.ROSTER_PATH)
                self.assertEqual(result["status"], "added")
                self.assertEqual(result["version"], "trust-list-2-v002")
                duplicate = trust.append_training_run(manifest, manifest_path, root=root,
                                                      roster_path=root / trust.ROSTER_PATH)
            self.assertEqual(duplicate["status"], "already_recorded")
            self.assertEqual(trust.latest_trust_paths(root)[2]["version"], "trust-list-2-v002")
            self.assertEqual(pd.read_csv(root / "docs/trust-list-2-v001.csv").experiment_id.unique().tolist(), ["existing"])

    def test_future_updates_wait_for_initial_roster_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / trust.ROSTER_PATH).parent.mkdir(parents=True)
            pd.DataFrame({"model": ["qwen06"]}).to_csv(root / trust.ROSTER_PATH, index=False)
            result = trust.append_training_run({"model": "qwen06", "dataset": "v1"}, "run.json",
                                               root=root, roster_path=root / trust.ROSTER_PATH)
            self.assertEqual(result["status"], "deferred")


if __name__ == "__main__":
    unittest.main()
