import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from src import v2_checkpoint_trust as trust
from src.pipeline import _partition_sha256


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

    def test_rank_k_metrics_and_saved_probability_csv_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v0_dir = root / "data/processed"
            cache_dir = root / trust.PREDICTION_DIR
            v0_dir.mkdir(parents=True)
            cache_dir.mkdir(parents=True)
            ids = [f"{index:04d}" for index in range(trust.V0_ROWS)]
            labels = [trust.LABELS[index % len(trust.LABELS)] for index in range(trust.V0_ROWS)]
            v0 = pd.DataFrame({"id": ids, "prompt_id": [f"p{i}" for i in ids],
                               "text": [f"response {i}" for i in ids], "model": labels})
            v0.to_parquet(v0_dir / "dataset_v0.parquet", index=False)
            probabilities = np.full((trust.V0_ROWS, len(trust.LABELS)), 0.001, dtype=float)
            for row_index, label in enumerate(labels):
                correct = trust.LABELS.index(label)
                other1 = (correct + 1) % len(trust.LABELS)
                other2 = (correct + 2) % len(trust.LABELS)
                if row_index % 3 == 0:
                    order = [correct, other1, other2]
                elif row_index % 3 == 1:
                    order = [other1, correct, other2]
                else:
                    order = [other1, other2, correct]
                probabilities[row_index, order] = [0.70, 0.20, 0.08]
                probabilities[row_index] /= probabilities[row_index].sum()
            experiment_id = "cached-exp"
            cached = pd.DataFrame({"id": ids, "y_true": labels})
            for index, label in enumerate(trust.LABELS):
                cached[f"prob::{label}"] = probabilities[:, index]
            cached.to_parquet(cache_dir / f"{experiment_id}.parquet", index=False)

            rank1 = trust._rank_k_metrics(labels, probabilities, 1)
            rows = []
            for label in trust.LABELS:
                metric = rank1[label]
                rows.append({
                    "sequence_id": "N0001", "experiment_id": experiment_id,
                    "model": "fixture", "model_title": "Fixture", "target_label": label,
                    "training_dataset": "v1", "dataset_lineage": "v1", "checkpoint": "model",
                    "checkpoint_path": "artifacts/checkpoint", "checkpoint_created_at_utc": trust._utc_now(),
                    "checkpoint_date": "11.13", **metric, "evaluation_eligible": True,
                    "validation_training_overlap": 0, "evaluation_note": "Full V0 fixture",
                    "checkpoint_available": True, "validation_split_sha256": "fixture",
                    "catalog_source": "fixture", "smoothed_precision": 0.0,
                    "rule_eligible": True, "overall_accuracy": rank1["_overall"]["overall_accuracy"],
                    "balanced_accuracy": rank1["_overall"]["balanced_accuracy"],
                    "validation_rows": trust.V0_ROWS,
                })
            metadata = {"generated_at_utc": trust._utc_now(),
                        "validation_sha256": _partition_sha256(v0), "validation_rows": trust.V0_ROWS}
            frame = pd.DataFrame(rows)
            frame.attrs["trust_list_version"] = "trust-list-2-v001"
            trust._publish_version(root, frame, metadata, "trust-list-2-v001")

            report = trust.generate_topk_trust_lists(root=root)
            top2 = pd.read_csv(root / "docs/trust-list-2-top2.csv")
            top3 = pd.read_csv(root / "docs/trust-list-2-top3.csv")
            self.assertEqual(list(top2.columns), trust.TRUST_COLUMNS)
            self.assertEqual(list(top3.columns), trust.TRUST_COLUMNS)
            self.assertEqual(len(top2), len(trust.LABELS))
            self.assertEqual(len(top3), len(trust.LABELS))
            self.assertEqual(int(top2.true_positive.sum()), sum(i % 3 == 1 for i in range(trust.V0_ROWS)))
            self.assertEqual(int(top3.true_positive.sum()), sum(i % 3 == 2 for i in range(trust.V0_ROWS)))
            self.assertFalse(report["rank1_artifact_mismatches"])
            self.assertEqual(report["source_trust_list_version"], "trust-list-2-v001")


if __name__ == "__main__":
    unittest.main()
