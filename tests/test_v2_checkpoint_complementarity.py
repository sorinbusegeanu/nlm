import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2_checkpoint_complementarity import (
    _greedy_replay,
    _replay_fixed_order,
    run_v2_checkpoint_complementarity,
)


def _rule(*, rank, sequence_id, experiment_id, label, prediction,
          support, true_positive, precision, eligible=True):
    return {
        "SeqID": rank,
        "sequence_id": sequence_id,
        "experiment_id": experiment_id,
        "model": "fake_model",
        "training_dataset": "fake_dataset",
        "target_label": label,
        "rule_eligible": eligible,
        "evaluation_eligible": True,
        "validation_training_overlap": 0,
        "prediction_count": support,
        "true_positive": true_positive,
        "false_positive": support - true_positive,
        "smoothed_precision": (true_positive + 1) / (support + 2),
        "precision": precision,
        "prediction": np.asarray(prediction, dtype=object),
    }


class V2CheckpointComplementarityTests(unittest.TestCase):
    def test_fixed_order_reports_precision_on_only_unclaimed_rows(self):
        y_true = np.asarray(["A", "B"] * 10, dtype=object)
        pred_a = np.asarray(["A"] * 10 + ["B"] * 10, dtype=object)
        pred_b = np.asarray(["A"] + ["B"] * 19, dtype=object)
        trust = pd.DataFrame([
            _rule(rank=1, sequence_id="N0001", experiment_id="a", label="A",
                  prediction=pred_a, support=10, true_positive=5, precision=0.5),
            _rule(rank=2, sequence_id="N0002", experiment_id="b", label="B",
                  prediction=pred_b, support=19, true_positive=10, precision=10 / 19),
        ])

        audit, summary, _ = _replay_fixed_order(trust, {"a": pred_a, "b": pred_b}, y_true)

        second = audit.iloc[1]
        self.assertEqual(int(second.incremental_assigned), 10)
        self.assertEqual(int(second.incremental_true_positive), 5)
        self.assertEqual(int(second.incremental_false_positive), 5)
        self.assertAlmostEqual(float(second.global_precision), 10 / 19)
        self.assertAlmostEqual(float(second.incremental_precision), 0.5)
        self.assertEqual(summary["claimed_rows"], 20)
        self.assertEqual(summary["correct_assignments"], 10)
        self.assertEqual(summary["incorrect_assignments"], 10)

    def test_greedy_order_is_deterministic_and_ties_use_seqid(self):
        y_true = np.asarray(["A", "B"] * 10, dtype=object)
        pred_a = np.asarray(["A" if i % 2 == 0 else "B" for i in range(20)], dtype=object)
        pred_a_copy = pred_a.copy()
        pred_b = np.asarray(["B" if i % 2 else "A" for i in range(20)], dtype=object)
        trust = pd.DataFrame([
            _rule(rank=1, sequence_id="N0001", experiment_id="a", label="A",
                  prediction=pred_a, support=10, true_positive=10, precision=1.0),
            _rule(rank=2, sequence_id="N0002", experiment_id="a_copy", label="A",
                  prediction=pred_a_copy, support=10, true_positive=10, precision=1.0),
            _rule(rank=3, sequence_id="N0003", experiment_id="b", label="B",
                  prediction=pred_b, support=10, true_positive=10, precision=1.0),
        ])
        predictions = {"a": pred_a, "a_copy": pred_a_copy, "b": pred_b}

        first, summary = _greedy_replay(trust, predictions, y_true,
            objective="precision", fallback_prediction=np.full(20, "A"),
            minimum_incremental_support=10)
        second, repeated_summary = _greedy_replay(trust, predictions, y_true,
            objective="precision", fallback_prediction=np.full(20, "A"),
            minimum_incremental_support=10)

        self.assertEqual(first.to_dict("records"), second.to_dict("records"))
        self.assertEqual(first.sequence_id.tolist(), ["N0001", "N0003"])
        self.assertEqual(summary, repeated_summary)
        self.assertEqual(summary["fallback_rows"], 0)
        self.assertEqual(summary["metrics_with_fallback"]["correct"], 20)

    def test_net_gain_stops_before_zero_or_negative_incremental_rules(self):
        y_true = np.asarray(["A", "B"] * 20, dtype=object)
        pred_a = np.asarray(["A" if i in range(0, 20, 2) else "B" for i in range(40)], dtype=object)
        pred_b = np.asarray(["B" if i in range(1, 20, 2) else "A" for i in range(40)], dtype=object)
        pred_bad_a = np.asarray(["A" if i in range(21, 40, 2) else "B" for i in range(40)], dtype=object)
        trust = pd.DataFrame([
            _rule(rank=1, sequence_id="N0001", experiment_id="a", label="A",
                  prediction=pred_a, support=10, true_positive=10, precision=1.0),
            _rule(rank=2, sequence_id="N0002", experiment_id="b", label="B",
                  prediction=pred_b, support=10, true_positive=10, precision=1.0),
            _rule(rank=3, sequence_id="N0003", experiment_id="bad", label="A",
                  prediction=pred_bad_a, support=10, true_positive=0, precision=0.0),
        ])

        trace, summary = _greedy_replay(trust,
            {"a": pred_a, "b": pred_b, "bad": pred_bad_a}, y_true,
            objective="net_gain", fallback_prediction=y_true,
            minimum_incremental_support=10)

        self.assertEqual(trace.sequence_id.tolist(), ["N0001", "N0002"])
        self.assertEqual(summary["fallback_rows"], 20)
        self.assertEqual(summary["fallback_correct"], 20)
        self.assertEqual(summary["metrics_with_fallback"]["correct"], 40)

    def test_real_packaging_path_reads_predictions_and_preserves_trust_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("docs", "data/processed", "cache", "predictions", "analysis"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            y = np.asarray(["A", "B"] * 10, dtype=object)
            ids = [f"id-{i:02d}" for i in range(20)]
            v0 = pd.DataFrame({"id": ids, "prompt_id": [f"p{i:02d}" for i in range(20)],
                "text": [f"response {i}" for i in range(20)], "model": y})
            v0.to_parquet(root / "data/processed/dataset_v0.parquet", index=False)
            digest_frame = v0[["id", "prompt_id", "text", "model"]].astype(str).sort_values("id", kind="stable")
            validation_hash = hashlib.sha256(digest_frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
            pred_a = np.asarray(["A"] * 10 + ["B"] * 10, dtype=object)
            pred_b = np.asarray(["A"] + ["B"] * 19, dtype=object)
            trust_rows = []
            for rank, seq, exp, label, prediction in (
                    (1, "N0001", "a", "A", pred_a), (2, "N0002", "b", "B", pred_b)):
                mask = prediction == label
                support = int(mask.sum())
                tp = int(np.sum(mask & (y == label)))
                trust_rows.append({
                    "SeqID": rank, "trust_list_version": "trust-list-2-v001",
                    "sequence_id": seq, "experiment_id": exp, "model": "fake_model",
                    "training_dataset": "external", "target_label": label,
                    "precision": tp / support, "recall": 0.5,
                    "prediction_count": support, "true_positive": tp,
                    "false_positive": support - tp, "false_negative": 10 - tp,
                    "evaluation_eligible": True, "validation_training_overlap": 0,
                    "checkpoint_available": True, "smoothed_precision": (tp + 1) / (support + 2),
                    "rule_eligible": True, "overall_accuracy": 0.5,
                })
                pd.DataFrame({"id": ids, "y_true": y, "y_pred": prediction}).to_parquet(
                    root / f"cache/{exp}.parquet", index=False)
            trust_path = root / "docs/trust.csv"
            pd.DataFrame(trust_rows).to_csv(trust_path, index=False)
            before = trust_path.read_bytes()
            metadata_path = root / "docs/trust.json"
            metadata_path.write_text(json.dumps({"version": "trust-list-2-v001",
                "generated_at_utc": "2026-01-01T00:00:00Z", "validation_rows": 20,
                "validation_sha256": validation_hash, "checkpoints": 2}))

            result = run_v2_checkpoint_complementarity(
                trust_list=trust_path, trust_metadata=metadata_path,
                cache_dir=root / "cache", artifact_dir=root / "predictions",
                data_dir=root / "data/processed", output_dir=root / "analysis",
                report_path=root / "docs/report.md", fallback_sequence_id="N0002")

            self.assertEqual(trust_path.read_bytes(), before)
            self.assertTrue((root / "analysis/incremental_rule_audit.csv").is_file())
            self.assertTrue((root / "analysis/greedy_precision_selection.csv").is_file())
            self.assertTrue((root / "analysis/greedy_net_gain_selection.csv").is_file())
            self.assertTrue((root / "analysis/complementarity_metrics.json").is_file())
            self.assertTrue((root / "docs/report.md").is_file())
            self.assertEqual(result["fixed_order"]["claimed_rows"], 20)


if __name__ == "__main__":
    unittest.main()
