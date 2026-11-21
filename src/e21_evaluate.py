"""Evaluate saved E21 epoch logits with independent and grouped Hungarian decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from .data_factory import _normal_text

ROOT = Path(__file__).resolve().parents[1]
E19F_ASSIGNMENTS = ROOT / "artifacts/analysis/e19f/predicted_group_assignments.parquet"
E19D_ASSIGNMENTS = ROOT / "artifacts/analysis/e19d/oof_group_predictions.parquet"


def hungarian_predictions(scores: np.ndarray, groups: list[str], classes: list[str]) -> np.ndarray:
    """Maximize logits independently in each reconstructed group with unique labels."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(classes) or len(values) != len(groups):
        raise ValueError("Hungarian score matrix, labels and group IDs are not aligned")
    if not np.isfinite(values).all():
        raise ValueError("Hungarian assignment requires finite logits")
    group_array = np.asarray(list(map(str, groups)), dtype=object)
    result = np.empty(len(values), dtype=object)
    for group in sorted(set(group_array)):
        rows = np.flatnonzero(group_array == group)
        if len(rows) > len(classes):
            raise ValueError(f"group {group!r} has {len(rows)} rows for {len(classes)} unique labels")
        assigned_rows, assigned_classes = linear_sum_assignment(values[rows], maximize=True)
        if len(assigned_rows) != len(rows) or len(set(assigned_classes.tolist())) != len(rows):
            raise RuntimeError(f"Hungarian assignment did not assign each row uniquely in group {group!r}")
        result[rows[assigned_rows]] = np.asarray(classes, dtype=object)[assigned_classes]
    return result


def _align_groups(frame: pd.DataFrame, v0: pd.DataFrame, *, id_column: str, group_column: str,
                  name: str) -> list[str]:
    required = {id_column, group_column}
    if required - set(frame):
        raise ValueError(f"{name} grouping artifact missing {sorted(required - set(frame))}")
    selected = frame[[id_column, group_column]].copy()
    selected[id_column] = selected[id_column].astype(str)
    if selected[id_column].duplicated().any() or selected[group_column].isna().any():
        raise ValueError(f"{name} grouping artifact has duplicate IDs or missing groups")
    mapping = selected.set_index(id_column)[group_column].astype(str)
    v0_ids = v0.id.astype(str).tolist()
    if set(mapping.index) != set(v0_ids) or len(mapping) != len(v0_ids):
        raise ValueError(f"{name} group IDs do not cover exactly the same 2,699 V0 IDs")
    return mapping.reindex(v0_ids).tolist()


def load_reconstructed_groups(v0: pd.DataFrame, *, e19f_path: Path = E19F_ASSIGNMENTS,
                              e19d_path: Path = E19D_ASSIGNMENTS) -> dict[str, list[str]]:
    e19f = pd.read_parquet(e19f_path)
    e19f = e19f.loc[e19f.method.astype(str).eq("E19-F-A")
                    & e19f.evaluation.astype(str).eq("complete-held-out")]
    e19d = pd.read_parquet(e19d_path)
    return {
        "e19_f_a": _align_groups(e19f, v0, id_column="id", group_column="predicted_group_id", name="E19-F-A"),
        "e19_d": _align_groups(e19d, v0, id_column="id", group_column="predicted_group_id", name="E19-D"),
        "oracle_true_prompt": v0.prompt_id.astype(str).tolist(),
    }


def _prediction_metrics(y_true: list[str], y_pred: np.ndarray, classes: list[str],
                        base_prediction: np.ndarray | None = None) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    correct = pred == truth
    metrics: dict[str, Any] = {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
    }
    if base_prediction is not None:
        base = np.asarray(base_prediction, dtype=object)
        base_correct = base == truth
        metrics.update({"changed": int((pred != base).sum()),
                        "corrected": int((~base_correct & correct).sum()),
                        "degraded": int((base_correct & ~correct).sum())})
    return metrics


def _per_label_rows(y_true, y_pred, classes, *, method, epoch):
    truth = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    precision, recall, _, support = precision_recall_fscore_support(
        truth, pred, labels=classes, zero_division=0)
    output = []
    for index, label in enumerate(classes):
        positive_true = truth == label
        positive_pred = pred == label
        accuracy = float(np.mean(positive_true == positive_pred))
        output.append({"epoch": epoch, "method": method, "label": label,
                       "precision": float(precision[index]), "recall": float(recall[index]),
                       "accuracy": accuracy, "support": int(support[index])})
    return output


def evaluate_epoch_scores(scores: np.ndarray, y_true: list[str], classes: list[str],
                          groups: dict[str, list[str]], *, epoch: int) -> tuple[list[dict], list[dict], dict]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(y_true), len(classes)):
        raise ValueError("epoch logits are not aligned to validation rows and classes")
    if not np.isfinite(scores).all():
        raise ValueError("epoch logits contain non-finite values")
    independent = np.asarray(classes, dtype=object)[scores.argmax(axis=1)]
    top3_indices = np.argsort(-scores, axis=1, kind="stable")[:, :min(3, len(classes))]
    label_to_index = {label: index for index, label in enumerate(classes)}
    top3 = np.asarray([y_true[i] in {classes[j] for j in top3_indices[i]}
                       for i in range(len(y_true))], dtype=bool)
    row_metrics = []
    per_label = []
    methods = {"independent_argmax": independent}
    for method, group_ids in groups.items():
        methods[method] = hungarian_predictions(scores, group_ids, classes)
    for method, prediction in methods.items():
        entry = {"epoch": int(epoch), "method": method,
                 **_prediction_metrics(y_true, prediction, classes,
                                       None if method == "independent_argmax" else independent)}
        if method == "independent_argmax":
            entry["top3_accuracy"] = float(top3.mean())
            shifted = scores - scores.max(axis=1, keepdims=True)
            log_normalizer = np.log(np.exp(shifted).sum(axis=1))
            true_scores = np.asarray([scores[i, label_to_index[label]] for i, label in enumerate(y_true)])
            entry["v0_cross_entropy"] = float(np.mean(log_normalizer - (true_scores - scores.max(axis=1))))
        row_metrics.append(entry)
        per_label.extend(_per_label_rows(y_true, prediction, classes, method=method, epoch=epoch))
    return row_metrics, per_label, {method: pred for method, pred in methods.items()}


def evaluate_manifest(manifest_path: Path, *, output_dir: Path | None = None,
                      v0_path: Path = ROOT / "data/processed/dataset_v0.parquet",
                      e19f_path: Path = E19F_ASSIGNMENTS, e19d_path: Path = E19D_ASSIGNMENTS) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_dataset") != "v2.GR.OR16" or manifest.get("validation_dataset") != "v0":
        raise ValueError("manifest is not a V2.GR.OR16 run validated on complete V0")
    v0 = pd.read_parquet(v0_path).sort_values("id", kind="stable").reset_index(drop=True)
    if len(v0) != 2699:
        raise ValueError(f"complete V0 must contain 2699 rows, found {len(v0)}")
    classes = list(map(str, manifest.get("classes", [])))
    if len(classes) != 18 or set(classes) != set(v0.model.astype(str)):
        raise ValueError("manifest class order must cover the canonical 18-label V0 roster")
    groups = load_reconstructed_groups(v0, e19f_path=e19f_path, e19d_path=e19d_path)
    records = manifest.get("validation_predictions_by_epoch") or []
    if not records:
        raise ValueError("manifest has no saved per-epoch V0 predictions")
    output_dir = output_dir or (ROOT / "artifacts/analysis/e21" / manifest["experiment_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    score_metrics, label_metrics = [], []
    all_predictions = []
    y_true = v0.model.astype(str).tolist()
    id_order = v0.id.astype(str).tolist()
    for record in records:
        epoch = int(record["epoch"])
        frame = pd.read_parquet(record["path"])
        if "id" not in frame or frame.id.astype(str).duplicated().any():
            raise ValueError(f"epoch {epoch} predictions have missing or duplicate row IDs")
        if set(frame.id.astype(str)) != set(id_order) or len(frame) != len(id_order):
            raise ValueError(f"epoch {epoch} predictions do not cover exactly all V0 row IDs")
        frame["id"] = frame.id.astype(str)
        frame = frame.set_index("id").reindex(id_order).reset_index()
        score_columns = [f"score::{label}" for label in classes]
        if set(score_columns) - set(frame.columns):
            raise ValueError(f"epoch {epoch} predictions are missing canonical class score columns")
        scores = frame[score_columns].to_numpy(dtype=np.float64)
        metrics, per_label, predictions = evaluate_epoch_scores(scores, y_true, classes, groups, epoch=epoch)
        training_log = next((item for item in manifest.get("training_loss_components_by_epoch", [])
                             if item.get("epoch") is not None
                             and int(round(float(item["epoch"]))) == epoch), {})
        for metric in metrics:
            metric["training_classification_loss"] = training_log.get("train_classification_loss")
            metric["training_group_ranking_loss"] = training_log.get("train_group_ranking_loss")
            metric["fit_seconds_total"] = manifest.get("fit_seconds")
            metric["training_seconds_excluding_validation"] = (manifest.get("runtime_breakdown") or {}).get(
                "training_excluding_validation_seconds")
        score_metrics.extend(metrics); label_metrics.extend(per_label)
        output = pd.DataFrame({"id": id_order, "y_true": y_true, "epoch": epoch})
        for method, prediction in predictions.items():
            output[f"prediction::{method}"] = prediction
        all_predictions.append(output)
    summary = {
        "experiment_id": manifest["experiment_id"], "experiment_name": manifest.get("experiment_name"),
        "training_dataset": manifest.get("training_dataset"), "training_rows": manifest.get("train_rows"),
        "validation_dataset": "v0", "validation_rows": len(v0), "classes": classes,
        "checkpoint_selection_metric": manifest.get("checkpoint_selection_metric"),
        "selected_epoch": manifest.get("selected_epoch"),
        "selected_independent_balanced_accuracy": manifest.get("selected_validation_balanced_accuracy"),
        "fit_seconds": manifest.get("fit_seconds"), "group_aware_sampling": manifest.get("group_aware_sampling"),
        "loss_config": manifest.get("loss_config"),
        "metrics_by_epoch": score_metrics,
        "prediction_methods": ["independent_argmax", "e19_f_a", "e19_d", "oracle_true_prompt"],
        "assignment_score": "raw classification logits",
    }
    pd.DataFrame(score_metrics).to_csv(output_dir / "metrics_by_epoch.csv", index=False)
    pd.DataFrame(label_metrics).to_csv(output_dir / "per_label_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(output_dir / "predictions_by_epoch.parquet", index=False)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--v0", type=Path, default=ROOT / "data/processed/dataset_v0.parquet")
    parser.add_argument("--e19f", type=Path, default=E19F_ASSIGNMENTS)
    parser.add_argument("--e19d", type=Path, default=E19D_ASSIGNMENTS)
    args = parser.parse_args(argv)
    summary = evaluate_manifest(args.manifest, output_dir=args.output_dir, v0_path=args.v0,
                                e19f_path=args.e19f, e19d_path=args.e19d)
    print(json.dumps({"experiment_id": summary["experiment_id"],
                      "selected_epoch": summary["selected_epoch"],
                      "metrics": summary["metrics_by_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
