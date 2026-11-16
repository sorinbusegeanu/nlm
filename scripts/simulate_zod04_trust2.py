"""Replay the ZOD04 first-eligible-rule cascade against corrected Trust List 2."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from scripts.package_zod04_kaggle import _selected_rules
from src.zod005 import _load_v0
from src.pipeline import _load_dataset
from src.v2_checkpoint_complementarity import _sha256_v0


ROOT = Path(__file__).resolve().parents[1]
LATEST_TRUST = ROOT / "docs/trust-list-2-latest.json"
PREDICTIONS = ROOT / "artifacts/analysis/trust_list_2_v0"
OUTPUT = ROOT / "artifacts/analysis/zod04_trust2_v2_fallback_simulation"
PREVIOUS_SIMULATION = ROOT / "artifacts/analysis/zod04_trust2_simulation"
REMOVED_IDS = {"N0081", "N0086"}
REMOVED_EXPERIMENTS = {
    "svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0",
    "ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0",
}
FALLBACK_EXPERIMENT_ID = (
    "ultrabert380_v2.14.balanced_valv0_datacd67b24a_split727ebb7e_"
    "seed2026_cfg5044b89f_codeab9f6d2"
)


def _load_cache(experiment_id: str, validation: pd.DataFrame, classes: list[str],
                trust_meta: dict) -> tuple[np.ndarray, dict]:
    if experiment_id == FALLBACK_EXPERIMENT_ID:
        manifest_path = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        provenance = manifest.get("validation_dataset_provenance", {})
        if (manifest.get("experiment_id") != experiment_id or manifest.get("model") != "ultrabert380"
                or provenance.get("name") != "v0" or int(provenance.get("rows", 0)) != len(validation)
                or provenance.get("sha256") != trust_meta.get("validation_sha256")
                or manifest.get("classes") != classes
                or manifest.get("training_dataset") != "v2.14.balanced"
                or int(manifest.get("training_dataset_rows", 0)) != 63702
                or manifest.get("checkpoint_selection_scope") != "v0"
                or int(manifest.get("checkpoint_selection_rows", 0)) != len(validation)):
            raise ValueError("V2.14 fallback manifest does not match UltraBERT and the full V0 snapshot")
        path = ROOT / manifest["prediction_path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("prediction_sha256"):
            raise ValueError("V2.14 full-V0 prediction cache is missing or its manifest hash differs")
        frame = pd.read_parquet(path)
        if not {"id", "y_true"}.issubset(frame.columns):
            raise ValueError("V2.14 prediction cache lacks ID or truth columns")
        frame["id"] = frame.id.astype(str)
        if len(frame) != len(validation) or frame.id.duplicated().any() or set(frame.id) != set(validation.id):
            raise ValueError("V2.14 prediction IDs do not exactly match complete V0")
        aligned = frame.set_index("id").loc[validation.id]
        if not np.array_equal(aligned.y_true.astype(str).to_numpy(), validation.y_true.to_numpy()):
            raise ValueError("V2.14 cache truth labels do not align with complete V0")
        if "y_pred" in aligned:
            prediction = aligned.y_pred.astype(str).to_numpy()
        else:
            columns = [f"prob::{label}" for label in classes]
            if not set(columns).issubset(aligned.columns):
                raise ValueError("V2.14 cache lacks top-1 predictions and ordered probability columns")
            prediction = np.asarray(classes, dtype=object)[aligned[columns].to_numpy(dtype=float).argmax(axis=1)]
        checkpoint_path = ROOT / manifest["checkpoint_path"]
        selection_path = checkpoint_path / "checkpoint_selection.json"
        if not checkpoint_path.is_dir() or not selection_path.is_file():
            raise FileNotFoundError("V2.14 selected checkpoint or checkpoint selection record is unavailable")
        selection = json.loads(selection_path.read_text())
        if (selection.get("selected_epoch") != manifest.get("selected_epoch")
                or selection.get("metric") != manifest.get("checkpoint_selection_metric")):
            raise ValueError("V2.14 checkpoint selection record differs from its training manifest")
        return prediction, {"path": str(path.relative_to(ROOT)),
            "prediction_sha256": manifest["prediction_sha256"],
            "experiment_name": manifest.get("experiment_name"),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "selected_epoch": manifest.get("selected_epoch"),
            "selection_metric": manifest.get("checkpoint_selection_metric"),
            "validation_sha256": provenance["sha256"], "rows": len(frame),
            "checkpoint_selection": selection}
    path = PREDICTIONS / f"{experiment_id}.parquet"
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"aligned full-V0 prediction cache and provenance are required: {path}")
    details = json.loads(sidecar.read_text())
    key = details.get("cache_key", {})
    expected_hash = str(trust_meta["validation_sha256"])
    if key.get("experiment_id") != experiment_id or key.get("v0_sha256") != expected_hash:
        raise ValueError(f"prediction cache provenance mismatch for {experiment_id}: {sidecar}")
    if int(key.get("rows", -1)) != len(validation) or key.get("labels") != classes:
        raise ValueError(f"prediction cache row count or label order mismatch for {experiment_id}")
    registry_run = next((item for item in trust_meta.get("inference_runs", [])
                         if item.get("experiment_id") == experiment_id), None)
    if registry_run is None or registry_run.get("checkpoint_sha256") != details.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint hash is not tied to Trust List 2 metadata for {experiment_id}")
    frame = pd.read_parquet(path)
    if not {"id", "y_true", "y_pred"}.issubset(frame.columns):
        raise ValueError(f"cache missing ID, truth, or top-1 prediction: {path}")
    frame["id"] = frame.id.astype(str)
    if len(frame) != len(validation) or frame.id.duplicated().any() or set(frame.id) != set(validation.id):
        raise ValueError(f"prediction IDs do not exactly match full V0 for {experiment_id}")
    aligned = frame.set_index("id").loc[validation.id]
    if not np.array_equal(aligned.y_true.astype(str).to_numpy(), validation.y_true.to_numpy()):
        raise ValueError(f"cached truth labels do not align with full V0 for {experiment_id}")
    prediction = aligned.y_pred.astype(str).to_numpy()
    if not set(prediction).issubset(set(classes)):
        raise ValueError(f"unknown predicted label in cache for {experiment_id}")
    return prediction, {"path": str(path.relative_to(ROOT)), "checkpoint_sha256": details["checkpoint_sha256"],
                        "v0_sha256": key["v0_sha256"], "rows": len(frame)}


def _metrics(y_true: np.ndarray, prediction: np.ndarray, classes: list[str]) -> dict:
    matrix = confusion_matrix(y_true, prediction, labels=classes)
    labels = {}
    for index, label in enumerate(classes):
        tp = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        labels[label] = {"precision": tp / predicted if predicted else 0.0,
                         "recall": tp / support if support else 0.0,
                         "support": support, "predicted": predicted, "correct": tp}
    return {"rows": len(y_true), "correct": int(np.sum(y_true == prediction)),
            "accuracy": float(accuracy_score(y_true, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
            "per_label": labels, "confusion_matrix": matrix.tolist()}


def simulate() -> dict:
    validation = _load_v0("data/processed")
    validation["id"] = validation.id.astype(str)
    validation["y_true"] = validation.y_true.astype(str)
    latest = json.loads(LATEST_TRUST.read_text())
    trust_csv, trust_json = ROOT / latest["csv_path"], ROOT / latest["metadata_path"]
    trust = pd.read_csv(trust_csv, dtype=str, keep_default_na=False)
    meta = json.loads(trust_json.read_text())
    if meta.get("version") != latest.get("version") or meta.get("validation_rows") != len(validation):
        raise ValueError("latest Trust List 2 metadata does not match full V0")
    raw_v0, _ = _load_dataset("v0", "data/processed")
    v0_sha = _sha256_v0(raw_v0)
    if v0_sha != meta.get("validation_sha256"):
        raise ValueError("V0 truth rows do not match Trust List 2 V0 hash")
    classes = sorted(validation.y_true.unique().tolist())
    if len(validation) != 2699 or len(classes) != 18:
        raise ValueError("expected the complete 2,699-row, 18-label V0 dataset")

    # Apply the corrected list's exclusions and the same eligibility/overlap gates
    # before delegating rule ordering and first-rule-per-label selection to ZOD04.
    eligible = trust.loc[
        ~trust.sequence_id.isin(REMOVED_IDS)
        & ~trust.experiment_id.isin(REMOVED_EXPERIMENTS)
        & trust.evaluation_eligible.str.lower().isin({"true", "1"})
        & trust.rule_eligible.str.lower().isin({"true", "1"})
        & trust.checkpoint_available.str.lower().isin({"true", "1"})
    ].copy()
    eligible["validation_training_overlap"] = pd.to_numeric(
        eligible.validation_training_overlap, errors="raise")
    eligible = eligible.loc[eligible.validation_training_overlap.eq(0)].copy()
    eligible["trust_list_version"] = meta["version"]
    eligible["SeqID"] = pd.to_numeric(eligible.SeqID, errors="raise")

    rules, _, _ = _selected_rules(eligible, expected_trust_version=meta["version"])
    fallback_id = FALLBACK_EXPERIMENT_ID
    candidate_ids = list(dict.fromkeys([r["candidate_id"] for r in rules] + [fallback_id]))
    if set(record["label"] for record in rules) != set(classes):
        raise ValueError("ZOD04 selection did not produce exactly one rule for each V0 label")
    if (set(record["sequence_id"] for record in rules) & REMOVED_IDS
            or set(record["candidate_id"] for record in rules) & REMOVED_EXPERIMENTS):
        raise ValueError("removed checkpoint appeared in the selected rules or fallback")

    top1, cache_provenance = {}, {}
    for experiment_id in candidate_ids:
        top1[experiment_id], cache_provenance[experiment_id] = _load_cache(
            experiment_id, validation, classes, meta)

    y_true = validation.y_true.to_numpy()
    fallback_prediction = top1[fallback_id]
    final_prediction = fallback_prediction.copy()
    selected_source = np.full(len(validation), "V2.14", dtype=object)
    pending = np.ones(len(validation), dtype=bool)
    rule_rows = []
    for rule in rules:
        candidate_id, label = rule["candidate_id"], rule["label"]
        claimed = pending & (top1[candidate_id] == label)
        assigned = int(claimed.sum())
        correct = int(np.sum(claimed & (y_true == label)))
        final_prediction[claimed] = label
        selected_source[claimed] = rule["sequence_id"]
        pending[claimed] = False
        cumulative = _metrics(y_true, final_prediction, classes)
        rule_rows.append({"rule_order": rule["order"], "label": label,
                          "checkpoint": rule["sequence_id"], "experiment_id": candidate_id,
                          "trust_precision": rule["precision_v0"], "assigned": assigned,
                          "correct": correct, "incorrect": assigned - correct,
                          "incremental_precision": correct / assigned if assigned else 0.0,
                          "remaining_for_fallback": int(pending.sum()),
                          "cumulative_correct": cumulative["correct"],
                          "cumulative_accuracy": cumulative["accuracy"]})

    fallback_assigned = int(pending.sum())
    fallback_correct = int(np.sum(pending & (fallback_prediction == y_true)))
    fallback_metrics = _metrics(y_true, fallback_prediction, classes)
    cascade_metrics = _metrics(y_true, final_prediction, classes)
    decisions = validation[["id", "y_true"]].copy()
    decisions["prediction"] = final_prediction
    decisions["selected_sequence_id"] = selected_source
    decisions["correct"] = final_prediction == y_true
    decisions["used_fallback"] = selected_source == "V2.14"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rules_frame = pd.DataFrame(rules).rename(columns={"precision_v0": "trust_precision"})
    rules_frame.to_csv(OUTPUT / "selected_rules.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(OUTPUT / "per_rule_metrics.csv", index=False)
    decisions.to_parquet(OUTPUT / "per_response_decisions.parquet", index=False)
    pd.DataFrame(cascade_metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        OUTPUT / "confusion_matrix.csv", index_label="actual")
    pd.DataFrame([{"label": label, **scores} for label, scores in cascade_metrics["per_label"].items()]).to_csv(
        OUTPUT / "per_label_metrics.csv", index=False)
    (OUTPUT / "cache_provenance.json").write_text(json.dumps(cache_provenance, indent=2) + "\n")

    previous_path = PREVIOUS_SIMULATION / "summary.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    previous_metrics = previous.get("cascade_metrics", {})
    previous_fallback_metrics = previous.get("fallback_alone_metrics", {})
    previous_rules = previous.get("selected_rules", [])
    same_rules = len(previous_rules) == len(rules) and all(
        (old.get("order"), old.get("candidate_id"), old.get("label"))
        == (new.get("order"), new.get("candidate_id"), new.get("label"))
        for old, new in zip(previous_rules, rules))
    z005_dir = ROOT / "artifacts/analysis/zod005/v001"
    z005_config = json.loads((z005_dir / "cascade.json").read_text())
    z005_metrics = json.loads((z005_dir / "metrics.json").read_text())
    z005_removed = sorted(set(z005_config.get("selected_sequences", [])) & REMOVED_IDS)
    comparison = {
        "fallback_alone": {k: v for k, v in fallback_metrics.items() if k != "confusion_matrix"},
        "zod04_corrected_trust_list": {k: v for k, v in cascade_metrics.items() if k != "confusion_matrix"},
        "previous_n0096_fallback_simulation": {
            "trust_list_version": previous.get("trust_list_version"),
            "fallback_experiment_id": previous.get("fallback", {}).get("experiment_id"),
            "correct": previous_metrics.get("correct"),
            "accuracy": previous_metrics.get("accuracy"),
            "balanced_accuracy": previous_metrics.get("balanced_accuracy"),
        },
        "zod005_existing": {
            "status": "incompatible_with_corrected_trust_list" if z005_removed else "compatible",
            "trust_list_version": z005_config.get("trust_list_version"),
            "rows": z005_metrics.get("rows"), "correct": z005_metrics.get("correct"),
            "accuracy": z005_metrics.get("accuracy"), "balanced_accuracy": z005_metrics.get("balanced_accuracy"),
            "removed_checkpoints_used": z005_removed,
            "note": "Existing ZOD005 used removed checkpoints; shown for context only, not a corrected-list comparison."
        },
    }
    summary = {
        "name": "zod04_trust2_simulation", "algorithm": "first_eligible_precision_ranked_rule_per_label_then_fallback",
        "trust_list_source": str(trust_csv.relative_to(ROOT)), "trust_list_version": meta["version"],
        "removed_sequence_ids": sorted(REMOVED_IDS), "v0_rows": len(validation), "labels": classes,
        "v0_sha256": meta["validation_sha256"], "rule_count": len(rules),
        "selected_rules": rules, "fallback": {"sequence_id": "V2.14",
            "experiment_id": fallback_id,
            "experiment_name": cache_provenance[fallback_id].get("experiment_name"),
            "model": "ultrabert380",
            "training_dataset": "v2.14.balanced (63,702 rows)",
            "checkpoint_selection_dataset": "V0 (2,699 rows; balanced accuracy)",
            "standalone_accuracy": fallback_metrics["accuracy"],
            "standalone_balanced_accuracy": fallback_metrics["balanced_accuracy"]},
        "fallback_coverage": {"rows": fallback_assigned, "fraction": fallback_assigned / len(validation),
            "correct": fallback_correct, "incorrect": fallback_assigned - fallback_correct,
            "accuracy_on_assigned": fallback_correct / fallback_assigned if fallback_assigned else 0.0},
        "fallback_alone_metrics": fallback_metrics,
        "cascade_metrics": cascade_metrics,
        "accuracy_delta_vs_fallback": cascade_metrics["accuracy"] - fallback_metrics["accuracy"],
        "correct_delta_vs_fallback": cascade_metrics["correct"] - fallback_metrics["correct"],
        "difference_vs_previous_n0096_simulation": {
            "correct_delta": cascade_metrics["correct"] - int(previous_metrics.get("correct", 0)),
            "accuracy_delta": cascade_metrics["accuracy"] - float(previous_metrics.get("accuracy", 0.0)),
            "balanced_accuracy_delta": cascade_metrics["balanced_accuracy"] - float(previous_metrics.get("balanced_accuracy", 0.0)),
            "previous_trust_list_version": previous.get("trust_list_version"),
            "rules_match_previous": same_rules,
            "fallback_correct_delta": fallback_metrics["correct"] - int(previous_fallback_metrics.get("correct", 0)),
            "fallback_accuracy_delta": fallback_metrics["accuracy"] - float(previous_fallback_metrics.get("accuracy", 0.0)),
            "fallback_balanced_accuracy_delta": fallback_metrics["balanced_accuracy"] - float(previous_fallback_metrics.get("balanced_accuracy", 0.0)),
        },
        "comparison": comparison, "cache_provenance": cache_provenance,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary, rule_rows)
    return summary


def _write_report(summary: dict, rule_rows: list[dict]) -> None:
    fallback = summary["fallback"]
    base, cascade = summary["fallback_alone_metrics"], summary["cascade_metrics"]
    z005 = summary["comparison"]["zod005_existing"]
    lines = ["# ZOD04 replay using corrected Trust List 2", "",
             f"V0 snapshot: {summary['v0_rows']} rows; SHA-256 `{summary['v0_sha256']}`. "
             f"Rules selected from `{summary['trust_list_source']}` in SeqID order after excluding "
             "N0081 and N0086, applying eligibility and zero-overlap filters.", "",
             f"Fallback: {fallback['sequence_id']} `{fallback['experiment_id']}` "
             f"({fallback['training_dataset']}); standalone {base['correct']}/{base['rows']} "
             f"({base['accuracy']:.4%}; balanced {base['balanced_accuracy']:.4%}).", "",
             "## Results", "",
             "| Evaluation | Correct / rows | Accuracy | Balanced accuracy |",
             "|---|---:|---:|---:|",
             f"| Fallback alone | {base['correct']} / {base['rows']} | {base['accuracy']:.4%} | {base['balanced_accuracy']:.4%} |",
             f"| ZOD04 corrected Trust List 2 | {cascade['correct']} / {cascade['rows']} | {cascade['accuracy']:.4%} | {cascade['balanced_accuracy']:.4%} |",
             f"| Existing ZOD005 (context only) | {z005['correct']} / {z005['rows']} | {z005['accuracy']:.4%} | {z005['balanced_accuracy']:.4%} |",
             "", f"ZOD04 changes versus fallback: {summary['correct_delta_vs_fallback']:+d} correct; "
             f"{summary['accuracy_delta_vs_fallback']:+.4%} accuracy.",
             f"Fallback receives {summary['fallback_coverage']['rows']} rows "
             f"({summary['fallback_coverage']['fraction']:.2%} coverage) after rules and gets "
             f"{summary['fallback_coverage']['correct']}/{summary['fallback_coverage']['rows']} "
             f"({summary['fallback_coverage']['accuracy_on_assigned']:.2%}) on those rows.", "",
             f"Compared with the previous N0096-fallback simulation: "
             f"{summary['difference_vs_previous_n0096_simulation']['correct_delta']:+d} correct, "
             f"{summary['difference_vs_previous_n0096_simulation']['accuracy_delta']:+.4%} accuracy, "
             f"and {summary['difference_vs_previous_n0096_simulation']['balanced_accuracy_delta']:+.4%} "
             f"balanced accuracy. Fallback-alone changed by "
             f"{summary['difference_vs_previous_n0096_simulation']['fallback_correct_delta']:+d} correct, "
             f"{summary['difference_vs_previous_n0096_simulation']['fallback_accuracy_delta']:+.4%} "
             f"accuracy, and "
             f"{summary['difference_vs_previous_n0096_simulation']['fallback_balanced_accuracy_delta']:+.4%} "
             f"balanced accuracy. Selected rules by order, checkpoint, and label match the previous run: "
             f"{summary['difference_vs_previous_n0096_simulation']['rules_match_previous']}.", "",
             "The fallback trained on 63,702 v2.14.balanced rows. V0's 2,699 rows were used for "
             "validation and checkpoint selection by balanced accuracy, so these V0 results are "
             "selection-biased and are not an independent evaluation.", "",
             "Existing ZOD005 is not compatible with the corrected list: its frozen cascade uses "
             f"removed checkpoint(s) {', '.join(z005['removed_checkpoints_used'])}. Its score above is "
             "reported only as context. The original ZOD04 Kaggle score of 0.550 is a separate "
             "Kaggle submission result and is not a full-V0 local simulation.", "",
             "## Rule-by-rule execution", "",
             "| Order | Label | Checkpoint | Trust precision | Assigned | Correct | Incorrect | Incremental precision | Cumulative accuracy |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rule_rows:
        lines.append(f"| {row['rule_order']} | {row['label']} | {row['checkpoint']} | "
                     f"{row['trust_precision']:.4%} | {row['assigned']} | {row['correct']} | "
                     f"{row['incorrect']} | {row['incremental_precision']:.4%} | "
                     f"{row['cumulative_accuracy']:.4%} |")
    lines.extend(["", "## Per-label precision and recall", "",
                  "| Label | Precision | Recall | Support | Predicted | Correct |",
                  "|---|---:|---:|---:|---:|---:|"])
    for label, values in cascade["per_label"].items():
        lines.append(f"| {label} | {values['precision']:.4%} | {values['recall']:.4%} | "
                     f"{values['support']} | {values['predicted']} | {values['correct']} |")
    lines.extend(["", f"Artifacts: `{OUTPUT.relative_to(ROOT)}`. No model inference was launched; all "
                  "predictions came from full-V0 caches with matching row IDs, label ordering, V0 hash, "
                  "and checkpoint or prediction-cache provenance.", ""])
    (OUTPUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    print(json.dumps(simulate(), indent=2))
