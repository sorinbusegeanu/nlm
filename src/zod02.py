"""ZOD02: cross-fitted multi-specialist cascade with marginal BAcc selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from .trusted_checkpoint_ensemble import ULTRABERT_CHECKPOINTS
from .zod01 import DEFAULT_MINIMUM_SUPPORT, DEFAULT_PRIOR_ALPHA, DEFAULT_PRIOR_BETA, _label_stats, _learn_rules
from .zod01_fixed_cascade import RULES as ZOD01_RULES, _load_prediction, _resolve_fallback


def _validation_frame(data_dir="data/processed"):
    from .pipeline import _load_dataset

    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.astype(str).eq("validation"),
                          ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["y_true"] = validation.pop("model").astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540 or validation.prompt_id.nunique() != 30:
        raise ValueError("ZOD02 requires frozen V0 validation (540 rows, 30 prompt groups)")
    return validation


def _load_cached_candidates(*, validation, inventory_path, ranked_rules_path,
                            predictions_dir, fallback_id):
    inventory = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    ranked = pd.read_csv(ranked_rules_path, dtype=str, keep_default_na=False)
    if not {"candidate_key", "experiment_id", "checkpoint_available",
            "evaluation_eligible", "validation_training_overlap", "prediction_path"}.issubset(inventory.columns):
        raise ValueError("checkpoint inventory does not have the expected candidate metadata")
    inv_by_id = {str(row.experiment_id): row for row in inventory.itertuples(index=False)}
    inv_by_key = {str(row.candidate_key): row for row in inventory.itertuples(index=False)}
    if fallback_id not in inv_by_id:
        raise ValueError(f"V1-all → V0 fallback is missing from the checkpoint inventory: {fallback_id}")

    eligible_ids = set()
    exclusion_counts = {"training_validation_overlap": 0, "checkpoint_unavailable": 0,
                        "not_evaluation_eligible": 0}
    for row in inventory.itertuples(index=False):
        candidate_id = str(row.experiment_id)
        overlap = int(float(getattr(row, "validation_training_overlap", 0) or 0))
        if overlap:
            exclusion_counts["training_validation_overlap"] += 1
            continue
        if str(getattr(row, "evaluation_eligible", "true")).lower() not in {"true", "1"}:
            exclusion_counts["not_evaluation_eligible"] += 1
            continue
        if str(getattr(row, "checkpoint_available", "false")).lower() not in {"true", "1"}:
            exclusion_counts["checkpoint_unavailable"] += 1
            continue
        checkpoint_path = Path(str(getattr(row, "checkpoint_path", "") or ""))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not str(getattr(row, "checkpoint_path", "") or "") or not checkpoint_path.exists():
            exclusion_counts["checkpoint_unavailable"] += 1
            continue
        eligible_ids.add(candidate_id)
    if fallback_id not in eligible_ids:
        raise ValueError("required ZOD01 fallback is not eligible and available")

    ranked_ids = set(ranked.loc[ranked.evaluation_eligible.astype(str).str.lower().isin(["true", "1"]),
                                "candidate_id"].astype(str))
    eligible_ids &= ranked_ids
    eligible_ids.add(fallback_id)
    classes = sorted(validation.y_true.unique().tolist())
    prediction_root = Path(predictions_dir)
    top1_by_id = {}
    metadata_by_id = {}
    for candidate_id in sorted(eligible_ids):
        inv = inv_by_id[candidate_id]
        cached_path_value = str(getattr(inv, "prediction_path", "") or "")
        cached_path = Path(cached_path_value) if cached_path_value else Path()
        if not cached_path.is_absolute():
            cached_path = Path.cwd() / cached_path
        if not cached_path.is_file():
            fallback_path = prediction_root / f"{candidate_id}_validation.parquet"
            cached_path = fallback_path if fallback_path.is_file() else cached_path
        if not cached_path.is_file():
            raise FileNotFoundError(f"cached V0 predictions missing for {candidate_id}")
        top1_by_id[candidate_id] = _load_prediction(cached_path, validation, classes, candidate_id)
        metadata_by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_key": str(getattr(inv, "candidate_key", "")),
            "model": str(getattr(inv, "model", "")),
            "model_title": str(getattr(inv, "model_title", "")),
            "dataset": str(getattr(inv, "training_dataset", "")),
            "checkpoint_path": str(getattr(inv, "checkpoint_path", "")),
            "manifest_path": str(getattr(inv, "manifest_path", "")),
            "prediction_path": str(cached_path),
        }
    missing_baseline = {key for _, key in ZOD01_RULES} - set(inv_by_key)
    if missing_baseline:
        raise ValueError(f"fixed ZOD01 rules reference missing candidate keys: {sorted(missing_baseline)}")
    zod01_rule_ids = []
    for label, key in ZOD01_RULES:
        candidate_id = (ULTRABERT_CHECKPOINTS[key] if key in ULTRABERT_CHECKPOINTS
                        else str(inv_by_key[key].experiment_id))
        if candidate_id not in top1_by_id:
            raise ValueError(f"fixed ZOD01 rule candidate excluded from ZOD02 input: {key}/{candidate_id}")
        zod01_rule_ids.append((str(label), candidate_id, key))
    return classes, top1_by_id, metadata_by_id, zod01_rule_ids, exclusion_counts, ranked


def _top1_candidates(top1_by_id, ids, classes):
    class_to_index = {label: i for i, label in enumerate(classes)}
    candidates = []
    for candidate_id in ids:
        labels = np.asarray(top1_by_id[candidate_id], dtype=object)
        probs = np.zeros((len(labels), len(classes)), dtype=np.float64)
        probs[np.arange(len(labels)), [class_to_index[x] for x in labels]] = 1.0
        candidates.append({"candidate_id": candidate_id, "probabilities": probs})
    return candidates


def _metrics(y_true, prediction, classes):
    y = np.asarray(y_true, dtype=str)
    pred = np.asarray(prediction, dtype=str)
    labels = _label_stats(y, pred, classes)
    return {"rows": len(y), "correct": int((y == pred).sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "per_label": labels,
            "recall_ge_80": sum(v["recall"] >= .8 for v in labels.values()),
            "recall_ge_90": sum(v["recall"] >= .9 for v in labels.values())}


def _select_marginal_rules(candidates, y_true, classes, *, fallback_id,
                           candidate_order=None, minimum_support=DEFAULT_MINIMUM_SUPPORT,
                           prior_alpha=DEFAULT_PRIOR_ALPHA, prior_beta=DEFAULT_PRIOR_BETA):
    """Greedily accept ranked rules only when they raise fit-fold balanced accuracy."""
    y = np.asarray(y_true, dtype=str)
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    by_id = {item["candidate_id"]: item for item in candidates}
    if fallback_id not in by_id:
        raise ValueError(f"fallback candidate {fallback_id} is missing")
    rules = _learn_rules(candidates, y, classes, minimum_support=minimum_support,
                         prior_alpha=prior_alpha, prior_beta=prior_beta,
                         candidate_order=order)
    top1 = {candidate_id: np.asarray(classes, dtype=object)[
        np.asarray(candidate["probabilities"]).argmax(axis=1)]
        for candidate_id, candidate in by_id.items()}
    assigned = np.zeros(len(y), dtype=bool)
    current = top1[fallback_id].copy()
    current_score = float(balanced_accuracy_score(y, current))
    selected, audit = [], []
    for rule in rules:
        candidate_id, label = rule["candidate_id"], rule["label"]
        remaining_match = (~assigned) & (top1[candidate_id] == label)
        indices = np.flatnonzero(remaining_match)
        before_score = current_score
        before_recall = _label_stats(y, current, classes)[label]["recall"]
        accepted = False
        reason = ""
        after_score = current_score
        incremental_tp = int(np.sum(y[indices] == label))
        incremental_fp = int(len(indices) - incremental_tp)
        if not bool(rule["eligible"]):
            reason = "support_below_minimum"
        elif len(indices) == 0:
            reason = "no_unassigned_matches"
        else:
            trial = current.copy()
            trial[indices] = label
            after_score = float(balanced_accuracy_score(y, trial))
            if after_score > before_score + 1e-12:
                accepted = True
                reason = "positive_marginal_balanced_accuracy"
                assigned[indices] = True
                current = trial
                current_score = after_score
                selected.append({**rule, "selected_order": len(selected) + 1})
            else:
                reason = "nonpositive_marginal_balanced_accuracy"
        after_recall = _label_stats(y, current, classes)[label]["recall"]
        audit.append({**rule, "accepted": accepted, "decision_reason": reason,
            "fit_assigned_if_accepted": int(len(indices)),
            "incremental_true_positives": incremental_tp,
            "incremental_false_positives": incremental_fp,
            "balanced_accuracy_before": before_score,
            "balanced_accuracy_after_if_applied": after_score,
            "marginal_balanced_accuracy": after_score - before_score if len(indices) and rule["eligible"] else 0.0,
            "per_label_recall_before": before_recall,
            "per_label_recall_after": after_recall,
            "remaining_fit_rows": int((~assigned).sum())})
    result = {"selected_rules": selected, "audit": audit,
              "fit_prediction": current, "fit_balanced_accuracy": current_score,
              "fit_metrics": _metrics(y, current, classes),
              "assigned_fit_rows": int(assigned.sum()),
              "fallback_fit_rows": int((~assigned).sum())}
    return result


def _apply_selected_rules(top1_by_id, y_true, classes, rules, *, fallback_id,
                          candidate_order=None):
    n = len(y_true)
    assigned = np.zeros(n, dtype=bool)
    output = np.empty(n, dtype=object)
    source = np.empty(n, dtype=object)
    rule_order = np.empty(n, dtype=object)
    trace = []
    for index, rule in enumerate(rules, start=1):
        candidate_id, label = rule["candidate_id"], rule["label"]
        take = (~assigned) & (np.asarray(top1_by_id[candidate_id], dtype=object) == label)
        rows = np.flatnonzero(take)
        output[rows], source[rows], rule_order[rows] = label, candidate_id, index
        assigned[rows] = True
        correct = int(np.sum(np.asarray(y_true, dtype=str)[rows] == label))
        trace.append({**rule, "selected_order": index, "assigned_examples": int(len(rows)),
            "incremental_true_positives": correct, "incremental_false_positives": int(len(rows) - correct),
            "remaining_rows": int((~assigned).sum())})
    remaining = np.flatnonzero(~assigned)
    fallback_predictions = np.asarray(top1_by_id[fallback_id], dtype=object)[remaining]
    output[remaining] = fallback_predictions
    source[remaining] = fallback_id
    rule_order[remaining] = "fallback"
    y = np.asarray(y_true, dtype=str)
    fallback_correct = int(np.sum(fallback_predictions == y[remaining]))
    trace.append({"selected_order": "fallback", "candidate_id": fallback_id,
        "label": "checkpoint top-1", "assigned_examples": len(remaining),
        "incremental_true_positives": fallback_correct,
        "incremental_false_positives": int(len(remaining) - fallback_correct),
        "remaining_rows": 0})
    decisions = pd.DataFrame({"y_true": y, "prediction": output,
        "selected_candidate_id": source, "selected_rule_order": rule_order})
    return output, decisions, trace


def _apply_fixed_zod01(top1_by_id, y_true, rules, fallback_id):
    n = len(y_true)
    assigned = np.zeros(n, dtype=bool)
    output = np.empty(n, dtype=object)
    source = np.empty(n, dtype=object)
    for label, candidate_id, _key in rules:
        take = (~assigned) & (np.asarray(top1_by_id[candidate_id], dtype=object) == label)
        rows = np.flatnonzero(take)
        output[rows] = label
        source[rows] = candidate_id
        assigned[rows] = True
    remaining = np.flatnonzero(~assigned)
    output[remaining] = np.asarray(top1_by_id[fallback_id], dtype=object)[remaining]
    source[remaining] = fallback_id
    return output, source


def run_zod02(*, candidates, y_true, groups, classes, fallback_id, zod01_rules,
              validation_ids=None, candidate_order=None, folds=5,
              minimum_support=DEFAULT_MINIMUM_SUPPORT, prior_alpha=DEFAULT_PRIOR_ALPHA,
              prior_beta=DEFAULT_PRIOR_BETA, output_dir="artifacts/analysis/zod02",
              report_path="docs/zod02_report.md", validation_split_sha256=""):
    """Cross-fit ZOD02 rule acceptance and compare against the fixed ZOD01 cascade."""
    y, group_ids = np.asarray(y_true, dtype=str), np.asarray(groups, dtype=str)
    classes = list(map(str, classes))
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    by_id = {item["candidate_id"]: item for item in candidates}
    if len(by_id) != len(candidates) or set(order) != set(by_id):
        raise ValueError("candidate_order must list each checkpoint candidate exactly once")
    if fallback_id not in by_id or len(y) != len(group_ids):
        raise ValueError("fallback and aligned labels/groups are required")
    if set(y) != set(classes) or len(np.unique(group_ids)) < int(folds):
        raise ValueError("classes or prompt groups do not match the evaluation requirements")
    if validation_ids is not None and len(validation_ids) != len(y):
        raise ValueError("validation IDs do not align")

    splitter = GroupKFold(n_splits=int(folds))
    zod02_oof = np.empty(len(y), dtype=object)
    zod01_oof = np.empty(len(y), dtype=object)
    zod02_source = np.empty(len(y), dtype=object)
    zod01_source = np.empty(len(y), dtype=object)
    fold_values = np.zeros(len(y), dtype=int)
    oof_audit, oof_trace = [], []
    fold_metrics = []
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(np.zeros(len(y)), y, group_ids), start=1):
        fit_candidates = [{**by_id[candidate_id],
            "probabilities": np.asarray(by_id[candidate_id]["probabilities"])[fit_idx]}
            for candidate_id in order]
        selected = _select_marginal_rules(fit_candidates, y[fit_idx], classes,
            fallback_id=fallback_id, candidate_order=order, minimum_support=minimum_support,
            prior_alpha=prior_alpha, prior_beta=prior_beta)
        for row in selected["audit"]:
            oof_audit.append({"fold": fold, **row})

        held_predictions_by_id = {candidate_id: np.asarray(by_id[candidate_id]["probabilities"])[held_idx]
                                  for candidate_id in order}
        held_top1 = {candidate_id: np.asarray(classes, dtype=object)[p.argmax(axis=1)]
                     for candidate_id, p in held_predictions_by_id.items()}
        fold_rules = [{**rule, "rule_rank": rank, "prediction_count": rule["prediction_count"]}
                      for rank, rule in enumerate(selected["selected_rules"], start=1)]
        z2_pred, z2_decisions, z2_trace = _apply_selected_rules(
            held_top1, y[held_idx], classes, fold_rules, fallback_id=fallback_id)
        z1_pred, z1_source = _apply_fixed_zod01(held_top1, y[held_idx], zod01_rules, fallback_id)
        zod02_oof[held_idx], zod01_oof[held_idx] = z2_pred, z1_pred
        zod02_source[held_idx] = z2_decisions.selected_candidate_id.to_numpy()
        zod01_source[held_idx] = z1_source
        fold_values[held_idx] = fold
        ids = (np.asarray(validation_ids, dtype=str)[held_idx] if validation_ids is not None
               else np.asarray([str(i) for i in held_idx]))
        for trace_row in z2_trace:
            oof_trace.append({"fold": fold, **trace_row})
        fold_metrics.append({"fold": fold, "fit_rows": len(fit_idx),
            "held_out_rows": len(held_idx), "fit_selected_rule_count": len(selected["selected_rules"]),
            "fit_balanced_accuracy": selected["fit_balanced_accuracy"],
            "zod02": _metrics(y[held_idx], z2_pred, classes),
            "zod01": _metrics(y[held_idx], z1_pred, classes)})

    zod02_metrics = _metrics(y, zod02_oof, classes)
    zod01_metrics = _metrics(y, zod01_oof, classes)
    correction_mask = (zod02_oof == y) & (zod01_oof != y)
    regression_mask = (zod02_oof != y) & (zod01_oof == y)
    final_fit_candidates = [{**by_id[candidate_id], "probabilities": by_id[candidate_id]["probabilities"]}
                            for candidate_id in order]
    final_selection = _select_marginal_rules(final_fit_candidates, y, classes,
        fallback_id=fallback_id, candidate_order=order, minimum_support=minimum_support,
        prior_alpha=prior_alpha, prior_beta=prior_beta)
    final_rules = final_selection["selected_rules"]

    metrics = {"approach": "zod02_multi_specialist_cascade", "validation_rows": len(y),
        "prompt_groups": int(len(np.unique(group_ids))), "folds": int(folds),
        "validation_split_sha256": validation_split_sha256,
        "validation_ids_sha256": hashlib.sha256("\n".join(map(str, validation_ids if validation_ids is not None else range(len(y)))).encode()).hexdigest(),
        "candidate_count": len(candidates), "fallback_candidate_id": fallback_id,
        "minimum_support": int(minimum_support),
        "precision_smoothing": {"alpha": prior_alpha, "beta": prior_beta,
            "formula": "(TP + alpha) / (prediction_count + alpha + beta)"},
        "rule_acceptance": "Accept only if replacing the current fallback/cascade prediction on remaining matching rows increases fitting-fold multiclass balanced accuracy.",
        "zod02_oof": zod02_metrics, "zod01_oof": zod01_metrics,
        "oof_delta": {"accuracy": zod02_metrics["accuracy"] - zod01_metrics["accuracy"],
            "balanced_accuracy": zod02_metrics["balanced_accuracy"] - zod01_metrics["balanced_accuracy"],
            "corrected_vs_zod01": int(correction_mask.sum()),
            "regressed_vs_zod01": int(regression_mask.sum())},
        "final_fit_metrics_exploratory": final_selection["fit_metrics"],
        "final_selected_rule_count": len(final_rules),
        "final_rules_exploratory": final_rules,
        "folds_summary": fold_metrics,
        "development_set_caveat": "V0 was used earlier to select the checkpoints and ZOD01 specialists; results are development diagnostics, not an independent generalization estimate."}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_all = (list(map(str, validation_ids)) if validation_ids is not None
               else [str(i) for i in range(len(y))])
    predictions = pd.DataFrame({"id": ids_all, "prompt_id": group_ids, "fold": fold_values,
        "y_true": y, "zod01_prediction": zod01_oof, "zod02_prediction": zod02_oof,
        "zod01_correct": zod01_oof == y, "zod02_correct": zod02_oof == y,
        "selected_zod01_checkpoint": zod01_source,
        "selected_zod02_checkpoint": zod02_source,
        "corrected_by_zod02": correction_mask, "regressed_by_zod02": regression_mask})
    predictions.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(oof_audit).to_csv(out_dir / "rule_selection_oof.csv", index=False)
    pd.DataFrame(oof_trace).to_csv(out_dir / "rule_application_oof.csv", index=False)
    pd.DataFrame(final_selection["audit"]).to_csv(out_dir / "rule_selection_full_v0.csv", index=False)
    full_audit = pd.DataFrame(final_selection["audit"])
    selected_audit = full_audit.loc[full_audit.accepted.astype(bool)].copy()
    full_rules_frame = pd.DataFrame(final_rules).merge(
        selected_audit[["candidate_id", "label", "fit_assigned_if_accepted",
            "incremental_true_positives", "incremental_false_positives",
            "marginal_balanced_accuracy", "per_label_recall_after"]],
        on=["candidate_id", "label"], how="left", validate="one_to_one")
    full_rules_frame.to_csv(out_dir / "selected_rules_full_v0.csv", index=False)
    predictions.loc[correction_mask | regression_mask].to_parquet(out_dir / "corrected_regressed.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    _write_report(Path(report_path), metrics, full_rules_frame.to_dict("records"), oof_audit, predictions)
    return {"metrics": metrics, "artifact_dir": str(out_dir), "report_path": str(report_path),
            "predictions": predictions, "final_rules": final_rules}


def run_zod02_analysis(*, data_dir="data/processed",
        inventory_path="artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv",
        ranked_rules_path="artifacts/analysis/zod01_specialist_discovery/ranked_checkpoint_label_rules.csv",
        predictions_dir="artifacts/analysis/zod01_specialist_discovery/predictions",
        registry_path="experiments/registry.sqlite", zod01_metrics_path="artifacts/analysis/zod01_18label_cascade/metrics.json",
        output_dir="artifacts/analysis/zod02", report_path="docs/zod02_report.md"):
    validation = _validation_frame(data_dir)
    classes = sorted(validation.y_true.unique().tolist())
    fallback = _resolve_fallback(registry_path)
    fallback_id = str(fallback["experiment_id"])
    prior_zod01 = json.loads(Path(zod01_metrics_path).read_text())
    if str(prior_zod01.get("fallback_candidate_id")) != fallback_id:
        raise ValueError("ZOD02 fallback differs from the recorded ZOD01 V1-all → V0 fallback")
    (classes, top1, metadata, zod01_rules, exclusions, ranked) = _load_cached_candidates(
        validation=validation, inventory_path=inventory_path, ranked_rules_path=ranked_rules_path,
        predictions_dir=predictions_dir, fallback_id=fallback_id)
    candidate_order = sorted(top1)
    candidates = _top1_candidates(top1, candidate_order, classes)
    profile_path = Path(data_dir) / "dataset_profile.json"
    profile = json.loads(profile_path.read_text())
    result = run_zod02(candidates=candidates, y_true=validation.y_true.to_numpy(),
        groups=validation.prompt_id.to_numpy(), classes=classes, fallback_id=fallback_id,
        zod01_rules=zod01_rules, validation_ids=validation.id.to_numpy(),
        candidate_order=candidate_order, minimum_support=DEFAULT_MINIMUM_SUPPORT,
        prior_alpha=DEFAULT_PRIOR_ALPHA, prior_beta=DEFAULT_PRIOR_BETA,
        output_dir=output_dir, report_path=report_path,
        validation_split_sha256=str(profile.get("split_sha256", "")))
    result["metrics"]["excluded_candidates"] = exclusions
    result["metrics"]["zod01_fixed_rule_count"] = len(zod01_rules)
    result["metrics"]["candidate_models"] = sorted({metadata[cid]["model"] for cid in candidate_order})
    result["metrics"]["source_ranked_rule_rows"] = int(len(ranked))
    Path(output_dir, "metrics.json").write_text(json.dumps(result["metrics"], indent=2, default=str) + "\n")
    return result


def _write_report(path, metrics, final_rules, oof_audit, predictions):
    z2, z1 = metrics["zod02_oof"], metrics["zod01_oof"]
    lines = ["# ZOD02 — Multi-Specialist Cascade", "",
        f"Five-fold prompt-group evaluation on frozen V0 ({metrics['validation_rows']} rows, {metrics['prompt_groups']} groups, {len(z2['per_label'])} labels). Existing ranked checkpoint-label candidates are re-ranked within each fitting fold with Beta({metrics['precision_smoothing']['alpha']:g},{metrics['precision_smoothing']['beta']:g}) precision smoothing and minimum support {metrics['minimum_support']}. Overlap checkpoints are excluded.", "",
        "Each candidate rule can assign only still-unassigned rows whose checkpoint top-1 predicts that label. A rule is retained only when applying it improves fitting-fold balanced accuracy relative to the current cascade with the V1-all → V0 UltraBERT fallback on unassigned rows. The frozen ZOD01 comparison applies its existing 18 fixed rules on the same held-out folds.", "",
        metrics["development_set_caveat"], "",
        "## Five-fold out-of-fold comparison", "",
        "| Cascade | Correct | Accuracy | Balanced accuracy | Δ balanced accuracy vs ZOD01 |", "|---|---:|---:|---:|---:|",
        f"| ZOD01 | {z1['correct']}/{z1['rows']} | {z1['accuracy']:.2%} | {z1['balanced_accuracy']:.2%} | — |",
        f"| ZOD02 | {z2['correct']}/{z2['rows']} | {z2['accuracy']:.2%} | {z2['balanced_accuracy']:.2%} | {metrics['oof_delta']['balanced_accuracy']:+.2%} |", "",
        f"ZOD02 corrected {metrics['oof_delta']['corrected_vs_zod01']} examples ZOD01 missed and regressed {metrics['oof_delta']['regressed_vs_zod01']} examples ZOD01 got right.", "",
        "## Per-label OOF recall", "",
        "| Label | ZOD01 recall | ZOD02 recall | Δ |", "|---|---:|---:|---:|"]
    for label in z1["per_label"]:
        a, b = z1["per_label"][label]["recall"], z2["per_label"][label]["recall"]
        lines.append(f"| {label} | {a:.1%} | {b:.1%} | {b-a:+.1%} |")
    lines.extend(["", "## Full-V0 deployment rule list", "",
        f"The following {len(final_rules)} rules were selected by running the fitting procedure on all 540 V0 examples. This list is used for the Kaggle package and is exploratory; OOF metrics above use fold-specific rule lists.", "",
        "| Order | Label | Checkpoint | Smoothed precision | Fit support | Assigned on full V0 | Incremental TP | Incremental FP | Δ fit balanced accuracy | Recall after rule |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for order, rule in enumerate(final_rules, start=1):
        lines.append(f"| {order} | {rule['label']} | `{rule['candidate_id']}` | {rule['smoothed_precision']:.1%} | {rule['prediction_count']} | {int(rule.get('fit_assigned_if_accepted', 0))} | {int(rule.get('incremental_true_positives', 0))} | {int(rule.get('incremental_false_positives', 0))} | {float(rule.get('marginal_balanced_accuracy', 0)):+.3%} | {float(rule.get('per_label_recall_after', 0)):.1%} |")
    lines.extend(["", f"Full-V0 fit balanced accuracy: {metrics['final_fit_metrics_exploratory']['balanced_accuracy']:.2%}; this is a selection score, not an independent estimate.",
        f"Validation split SHA-256: `{metrics['validation_split_sha256']}`. Artifacts are under `artifacts/analysis/zod02/`.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run ZOD02 with cached checkpoint predictions")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--inventory", default="artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv")
    parser.add_argument("--ranked-rules", default="artifacts/analysis/zod01_specialist_discovery/ranked_checkpoint_label_rules.csv")
    parser.add_argument("--predictions-dir", default="artifacts/analysis/zod01_specialist_discovery/predictions")
    parser.add_argument("--registry", default="experiments/registry.sqlite")
    parser.add_argument("--output-dir", default="artifacts/analysis/zod02")
    parser.add_argument("--report", default="docs/zod02_report.md")
    args = parser.parse_args(argv)
    result = run_zod02_analysis(data_dir=args.data_dir, inventory_path=args.inventory,
        ranked_rules_path=args.ranked_rules, predictions_dir=args.predictions_dir,
        registry_path=args.registry, output_dir=args.output_dir, report_path=args.report)
    print(json.dumps({"report_path": result["report_path"],
        "balanced_accuracy": result["metrics"]["zod02_oof"]["balanced_accuracy"],
        "zod01_balanced_accuracy": result["metrics"]["zod01_oof"]["balanced_accuracy"],
        "selected_rule_count": result["metrics"]["final_selected_rule_count"]}, indent=2))


if __name__ == "__main__":
    main()
