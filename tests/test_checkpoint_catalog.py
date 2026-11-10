import json
import ast
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from src.checkpoint_catalog import (
    CATALOG_COLUMNS,
    _align_v0_predictions,
    _display_date,
    _next_sequence,
    _sequence_id,
    append_manifest,
    load_catalog,
    refresh_trust_list,
    seed_from_specialist_snapshot,
    sort_catalog_for_dashboard,
)


class CheckpointCatalogTests(unittest.TestCase):
    def test_versioned_trust_list_has_complete_order_and_bumps_on_source_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog_path = root / "catalog.csv"
            trust_path = root / "trust.csv"
            metadata_path = root / "trust.json"
            rows = []
            for number, precision in ((1, .5), (2, .9)):
                row = {column: "" for column in CATALOG_COLUMNS}
                row.update({"sequence_id": f"N{number:04d}", "experiment_id": f"exp-{number}",
                    "target_label": "label", "precision": precision, "recall": precision,
                    "prediction_count": 10, "true_positive": int(precision * 10),
                    "false_positive": 10 - int(precision * 10), "false_negative": 0,
                    "evaluation_eligible": True, "validation_training_overlap": 0,
                    "checkpoint_available": True})
                rows.append(row)
            pd.DataFrame(rows, columns=CATALOG_COLUMNS).to_csv(catalog_path, index=False)
            first = refresh_trust_list(catalog_path=catalog_path, trust_list_path=trust_path,
                                       metadata_path=metadata_path)
            self.assertEqual(first["version"], "trust-v001")
            listed = pd.read_csv(trust_path)
            self.assertEqual(listed.SeqID.tolist(), [1, 2])
            self.assertEqual(listed.experiment_id.tolist(), ["exp-2", "exp-1"])
            self.assertFalse(listed.SeqID.isna().any())
            self.assertEqual(refresh_trust_list(catalog_path=catalog_path, trust_list_path=trust_path,
                metadata_path=metadata_path)["status"], "unchanged")
            listed.loc[0, "precision"] = .4
            listed.to_csv(catalog_path, index=False)
            second = refresh_trust_list(catalog_path=catalog_path, trust_list_path=trust_path,
                                        metadata_path=metadata_path)
            self.assertEqual(second["version"], "trust-v002")

    def test_sequence_ids_are_stable_and_monotonic(self):
        frame = pd.DataFrame({"sequence_id": ["N0001", "N0072", "N10000"]})
        self.assertEqual(_next_sequence(frame), 10001)
        self.assertEqual(_sequence_id(72), "N0072")
        self.assertEqual(_sequence_id(10000), "N10000")

    def test_dashboard_sort_is_precision_then_recall_with_missing_last(self):
        frame = pd.DataFrame([
            {"sequence_id": "N0002", "target_label": "b", "precision": .8, "recall": .6},
            {"sequence_id": "N0001", "target_label": "a", "precision": .8, "recall": .7},
            {"sequence_id": "N0003", "target_label": "c", "precision": .95, "recall": .2},
            {"sequence_id": "N0004", "target_label": "d", "precision": np.nan, "recall": 1.0},
        ])
        self.assertEqual(sort_catalog_for_dashboard(frame).sequence_id.tolist(),
                         ["N0003", "N0001", "N0002", "N0004"])
        self.assertEqual(_display_date(datetime(2026, 11, 8, 23, tzinfo=timezone.utc)), "11.09")

    def test_prediction_factory_dashboard_no_longer_reads_factory_or_roster_reports(self):
        app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
        tree = ast.parse(app_path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "render_prediction_factory")
        source = ast.get_source_segment(app_path.read_text(), function)
        self.assertIn("checkpoint_label_trust_list.csv", source)
        for removed in ("read_ensemble_registry", "active_model_roster.json", "prediction_v0_", "strategy_metrics"):
            self.assertNotIn(removed, source)

    def test_snapshot_seed_preserves_existing_ids_and_assigns_legacy_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory_path = root / "inventory.csv"
            rules_path = root / "rules.csv"
            catalog_path = root / "catalog.csv"
            pd.DataFrame([
                {"candidate_key": "N0001", "experiment_id": "exp-1", "model": "svc",
                 "model_title": "SVC", "training_dataset": "v1", "checkpoint_available": True,
                 "manifest_path": "", "checkpoint_path": "", "checkpoint_basename": "svc.pkl",
                 "validation_training_overlap": 0},
                {"candidate_key": "A", "experiment_id": "exp-a", "model": "ultrabert380",
                 "model_title": "UltraBERT-380M", "training_dataset": "v1", "checkpoint_available": True,
                 "manifest_path": "", "checkpoint_path": "", "checkpoint_basename": "ultra",
                 "validation_training_overlap": 0},
            ]).to_csv(inventory_path, index=False)
            pd.DataFrame([
                {"candidate_key": "N0001", "candidate_id": "exp-1", "model": "svc",
                 "model_title": "SVC", "dataset": "v1", "label": "label-a",
                 "precision": 0.75, "recall": 0.5, "prediction_count": 12,
                 "true_positive": 9, "false_positive": 3, "false_negative": 9,
                 "evaluation_eligible": True, "evaluation_exclusion_reason": ""},
                {"candidate_key": "A", "candidate_id": "exp-a", "model": "ultrabert380",
                 "model_title": "UltraBERT-380M", "dataset": "v1", "label": "label-a",
                 "precision": 0.8, "recall": 0.6, "prediction_count": 10,
                 "true_positive": 8, "false_positive": 2, "false_negative": 8,
                 "evaluation_eligible": False, "evaluation_exclusion_reason": "overlap"},
            ]).to_csv(rules_path, index=False)

            seeded = seed_from_specialist_snapshot(catalog_path=catalog_path,
                inventory_path=inventory_path, rules_path=rules_path)
            ids = dict(zip(seeded.experiment_id, seeded.sequence_id))
            self.assertEqual(ids, {"exp-1": "N0001", "exp-a": "N0002"})
            alias = seeded.loc[seeded.experiment_id.eq("exp-a")].iloc[0]
            self.assertEqual(alias.legacy_candidate_key, "A")
            self.assertFalse(bool(alias.evaluation_eligible))

            again = seed_from_specialist_snapshot(catalog_path=catalog_path,
                inventory_path=inventory_path, rules_path=rules_path)
            self.assertEqual(again.sequence_id.tolist(), seeded.sequence_id.tolist())

    def test_append_computes_exact_per_label_metrics_on_frozen_v0(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            data_dir.mkdir()
            labels = [f"label-{index:02d}" for index in range(18)]
            validation_rows = []
            for prompt_index in range(30):
                for label in labels:
                    row_id = f"v-{prompt_index:02d}-{label}"
                    validation_rows.append({"id": row_id, "prompt_id": f"p-{prompt_index:02d}",
                                            "model": label, "text": f"response {row_id}"})
            train_rows = [{"id": f"train-{index}", "prompt_id": f"train-p-{index}",
                           "model": labels[index % 18], "text": "train response"}
                          for index in range(10)]
            rows = pd.DataFrame(validation_rows + train_rows)
            split = pd.DataFrame({"id": rows.id,
                                  "split": ["validation"] * 540 + ["train"] * 10})
            rows.to_parquet(data_dir / "dataset_v0.parquet", index=False)
            split.to_parquet(data_dir / "splits_v0.parquet", index=False)
            (data_dir / "dataset_profile.json").write_text(json.dumps({"split_sha256": "v0-split-hash"}))

            catalog_path = root / "catalog.csv"
            prior = {column: "" for column in CATALOG_COLUMNS}
            prior.update({"sequence_id": "N0001", "experiment_id": "old-run", "target_label": labels[0]})
            pd.DataFrame([prior], columns=CATALOG_COLUMNS).to_csv(catalog_path, index=False)

            prediction_path = root / "predictions.parquet"
            pred_frame = pd.DataFrame(validation_rows)[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"})
            for label in labels:
                pred_frame[f"prob::{label}"] = (pred_frame.y_true.eq(label)).astype(float)
            pred_frame.to_parquet(prediction_path, index=False)
            checkpoint_path = root / "checkpoint"
            checkpoint_path.mkdir()
            manifest_path = root / "new_manifest.json"
            manifest = {"experiment_id": "new-run", "model": "catboost_stylometry",
                "dataset": "v1_all", "checkpoint_path": str(checkpoint_path),
                "prediction_path": str(prediction_path),
                "created_at_utc": "2026-11-09T01:00:00+00:00"}
            manifest_path.write_text(json.dumps(manifest))

            outcome = append_manifest(manifest=manifest, manifest_path=manifest_path,
                                      catalog_path=catalog_path, data_dir=data_dir)
            self.assertEqual(outcome, {"status": "added", "sequence_id": "N0002", "labels": 18})
            catalog = load_catalog(catalog_path)
            added = catalog.loc[catalog.experiment_id.eq("new-run")]
            self.assertEqual(len(added), 18)
            self.assertEqual(set(added.sequence_id), {"N0002"})
            self.assertTrue(np.allclose(added.precision.astype(float), 1.0))
            self.assertTrue(np.allclose(added.recall.astype(float), 1.0))
            self.assertTrue(np.all(added.prediction_count.astype(int) == 30))
            self.assertTrue((added.validation_split_sha256 == "v0-split-hash").all())
            self.assertEqual(append_manifest(manifest=manifest, manifest_path=manifest_path,
                catalog_path=catalog_path, data_dir=data_dir)["status"], "already_recorded")

    def test_v0_alignment_rejects_wrong_ids_and_prompt_groups(self):
        validation = pd.DataFrame({"id": ["1", "2"], "prompt_id": ["p1", "p2"],
                                   "model": ["a", "b"]})
        wrong_ids = pd.DataFrame({"id": ["1", "other"], "prompt_id": ["p1", "p2"],
                                  "y_true": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "IDs do not exactly match"):
            _align_v0_predictions(wrong_ids, validation, "bad-ids")
        wrong_groups = pd.DataFrame({"id": ["1", "2"], "prompt_id": ["p1", "wrong"],
                                     "y_true": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "prompt IDs do not match"):
            _align_v0_predictions(wrong_groups, validation, "bad-groups")

    def test_missing_prediction_cache_does_not_add_catalog_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog_path = root / "catalog.csv"
            prior = {column: "" for column in CATALOG_COLUMNS}
            prior.update({"sequence_id": "N0001", "experiment_id": "old-run"})
            pd.DataFrame([prior], columns=CATALOG_COLUMNS).to_csv(catalog_path, index=False)
            manifest_path = root / "run_manifest.json"
            manifest = {"experiment_id": "missing-cache", "dataset": "v1_all",
                        "checkpoint_path": str(root / "checkpoint"),
                        "prediction_path": str(root / "missing.parquet")}
            outcome = append_manifest(manifest=manifest, manifest_path=manifest_path,
                                      catalog_path=catalog_path)
            self.assertEqual(outcome["status"], "skipped")
            self.assertEqual(len(load_catalog(catalog_path)), 1)

    def test_training_completion_refreshes_catalog_after_prediction_factory(self):
        from src.pipeline import _refresh_ensemble_after_training
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "run_manifest.json"
            manifest = {"experiment_id": "run-1", "dataset": "v1_all"}
            manifest_path.write_text(json.dumps(manifest))
            with patch("src.ensemble_registry.refresh_prediction_factory",
                       return_value={"status": "evaluated"}), \
                 patch("src.checkpoint_catalog.refresh_catalog",
                       return_value={"status": "refreshed", "checkpoint_count": 1}):
                _refresh_ensemble_after_training(manifest, manifest_path,
                    artifact_dir="artifacts", data_dir="data/processed")
            stored = json.loads(manifest_path.read_text())
            self.assertEqual(stored["prediction_factory_update"]["status"], "evaluated")
            self.assertEqual(stored["checkpoint_catalog_update"]["checkpoint_count"], 1)


if __name__ == "__main__":
    unittest.main()
