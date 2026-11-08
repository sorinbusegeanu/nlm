"""Sequential precision-ranked checkpoint rules with prompt-group OOF evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .checkpoint_comparison import _validated_probabilities
from .prediction_factory import _align_candidates, _read_candidates
from .trusted_checkpoint_ensemble import ULTRABERT_CHECKPOINTS

DEFAULT_MINIMUM_SUPPORT = 10
DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0


def _top1(probabilities, classes):
    p = np.asarray(probabilities, dtype=np.float64)
    return np.asarray(classes, dtype=object)[p.argmax(axis=1)]


def _label_stats(y_true, predictions, classes):
    y = np.asarray(y_true, dtype=str)
    pred = np.asarray(predictions, dtype=str)
    result = {}
    for label in classes:
        tp = int(np.sum((y == label) & (pred == label)))
        fp = int(np.sum((y != label) & (pred == label)))
        fn = int(np.sum((y == label) & (pred != label)))
        predicted_count = tp + fp
        support = tp + fn
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        result[label] = {"true_positive": tp, "false_positive": fp,
                         "false_negative": fn, "support": support,
                         "prediction_count": predicted_count,
                         "precision": float(precision), "recall": float(recall)}
    return result


def _learn_rules(candidates, y_true, classes, *, minimum_support=DEFAULT_MINIMUM_SUPPORT,
                 prior_alpha=DEFAULT_PRIOR_ALPHA, prior_beta=DEFAULT_PRIOR_BETA,
                 candidate_order=None):
    """Build all candidate-label rules and rank by smoothed precision.

    Rules below minimum support remain in the ordered audit list but are skipped
    during assignment. Ties use raw precision, support, candidate order and class
    order, making the sequence deterministic.
    """
    y = np.asarray(y_true, dtype=str)
    classes = list(map(str, classes))
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    by_id = {item["candidate_id"]: item for item in candidates}
    if len(by_id) != len(candidates) or set(order) != set(by_id):
        raise ValueError("candidate_order must list every candidate ID exactly once")
    class_order = {label: index for index, label in enumerate(classes)}
    candidate_rank = {candidate_id: index for index, candidate_id in enumerate(order)}
    rules = []
    for candidate_id in order:
        candidate = by_id[candidate_id]
        predicted = _top1(candidate["probabilities"], classes)
        for label in classes:
            mask = predicted == label
            count = int(mask.sum())
            tp = int(np.sum(mask & (y == label)))
            fp = count - tp
            fn = int(np.sum((y == label) & ~mask))
            actual = int(np.sum(y == label))
            raw_precision = tp / count if count else 0.0
            recall = tp / actual if actual else 0.0
            smoothed = (tp + prior_alpha) / (count + prior_alpha + prior_beta)
            rules.append({"candidate_id": candidate_id, "label": label,
                          "true_positive": tp, "false_positive": fp,
                          "false_negative": fn, "prediction_count": count,
                          "actual_count": actual, "precision": float(raw_precision),
                          "smoothed_precision": float(smoothed), "recall": float(recall),
                          "eligible": count >= minimum_support})
    rules.sort(key=lambda rule: (-rule["smoothed_precision"], -rule["precision"],
        -rule["prediction_count"], candidate_rank[rule["candidate_id"]],
        class_order[rule["label"]]))
    for rank, rule in enumerate(rules, start=1):
        rule["rule_rank"] = rank
    return rules


def _apply_rules(candidates, classes, rules, *, minimum_support=DEFAULT_MINIMUM_SUPPORT,
                 fallback_candidate_id="D", row_ids=None, prompt_ids=None,
                 y_true=None, fold=None):
    """Apply ordered eligible rules to unassigned rows, then use D on leftovers."""
    classes = list(map(str, classes))
    n_rows = len(candidates[0]["probabilities"])
    candidate_predictions = {item["candidate_id"]: _top1(item["probabilities"], classes)
                             for item in candidates}
    if fallback_candidate_id not in candidate_predictions:
        fallback_candidate_id = candidates[-1]["candidate_id"]
    assigned = np.zeros(n_rows, dtype=bool)
    final_predictions = np.empty(n_rows, dtype=object)
    selected_candidate = np.empty(n_rows, dtype=object)
    selected_rule = np.empty(n_rows, dtype=object)
    trace = []
    y = np.asarray(y_true, dtype=str) if y_true is not None else None
    for rule in rules:
        before = int((~assigned).sum())
        eligible = bool(rule["prediction_count"] >= minimum_support)
        if eligible:
            take = (~assigned) & (candidate_predictions[rule["candidate_id"]] == rule["label"])
            indices = np.flatnonzero(take)
            assigned[indices] = True
            final_predictions[indices] = rule["label"]
            selected_candidate[indices] = rule["candidate_id"]
            selected_rule[indices] = int(rule["rule_rank"])
        else:
            indices = np.empty(0, dtype=int)
        after = int((~assigned).sum())
        correct = int(np.sum(y[indices] == rule["label"])) if y is not None else None
        trace.append({**rule, "fold": fold, "applied": eligible,
            "assigned_examples": int(len(indices)), "correct": correct,
            "incorrect": int(len(indices) - correct) if correct is not None else None,
            "remaining_before": before, "remaining_after": after})
    remaining = np.flatnonzero(~assigned)
    fallback_preds = candidate_predictions[fallback_candidate_id][remaining]
    final_predictions[remaining] = fallback_preds
    selected_candidate[remaining] = fallback_candidate_id
    selected_rule[remaining] = "D_fallback"
    fallback_correct = int(np.sum(fallback_preds == y[remaining])) if y is not None else None
    trace.append({"fold": fold, "rule_rank": "fallback", "candidate_id": fallback_candidate_id,
        "label": "checkpoint top-1", "precision": None, "smoothed_precision": None,
        "prediction_count": None, "eligible": True, "applied": True,
        "assigned_examples": int(len(remaining)), "correct": fallback_correct,
        "incorrect": int(len(remaining) - fallback_correct) if fallback_correct is not None else None,
        "remaining_before": int(len(remaining)), "remaining_after": 0})
    decisions = pd.DataFrame({"row_index": np.arange(n_rows),
        "id": list(map(str, row_ids)) if row_ids is not None else list(map(str, range(n_rows))),
        "prompt_id": list(map(str, prompt_ids)) if prompt_ids is not None else [""] * n_rows,
        "fold": fold, "y_true": y if y is not None else [None] * n_rows,
        "prediction": final_predictions, "selected_candidate_id": selected_candidate,
        "selected_rule_rank": selected_rule})
    return final_predictions, decisions, trace


def _metrics(y_true, predictions, classes):
    y = np.asarray(y_true, dtype=str)
    pred = np.asarray(predictions, dtype=str)
    per_label = _label_stats(y, pred, classes)
    return {"rows": int(len(y)), "correct": int(np.sum(y == pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            "per_label": per_label,
            "recall_ge_80": int(sum(v["recall"] >= .80 for v in per_label.values())),
            "recall_ge_90": int(sum(v["recall"] >= .90 for v in per_label.values()))}


def run_zod01(*, candidates, y_true, groups, classes, validation_ids=None,
              candidate_order=None, folds=5, minimum_support=DEFAULT_MINIMUM_SUPPORT,
              prior_alpha=DEFAULT_PRIOR_ALPHA, prior_beta=DEFAULT_PRIOR_BETA,
              output_dir="artifacts/analysis/zod01", report_path="docs/zod01_ultrabert.md",
              validation_split_sha256=None):
    """Fit rule order on each GroupKFold training partition and pool held-out rows."""
    if not candidates:
        raise ValueError("no candidates supplied")
    classes = list(map(str, classes))
    y, group_ids = np.asarray(y_true, dtype=str), np.asarray(groups, dtype=str)
    order = list(candidate_order or [item["candidate_id"] for item in candidates])
    by_id = {item["candidate_id"]: item for item in candidates}
    if len(by_id) != len(candidates) or set(order) != set(by_id):
        raise ValueError("candidate_order must list every candidate ID exactly once")
    candidates = [by_id[candidate_id] for candidate_id in order]
    if len(y) != len(group_ids) or len(np.unique(group_ids)) < int(folds):
        raise ValueError("truth and prompt groups must align and contain enough groups")
    if set(y) != set(classes):
        raise ValueError("classes must match the validation labels")
    for candidate in candidates:
        if list(map(str, candidate.get("classes", classes))) != classes:
            raise ValueError(f"{candidate['candidate_id']} class order mismatch")
        _validated_probabilities(candidate["probabilities"], len(y), len(classes), candidate["candidate_id"])
        if "ids" in candidate and validation_ids is not None and list(map(str, candidate["ids"])) != list(map(str, validation_ids)):
            raise ValueError(f"{candidate['candidate_id']} validation IDs mismatch")
        if "prompt_ids" in candidate and not np.array_equal(np.asarray(candidate["prompt_ids"], dtype=str), group_ids):
            raise ValueError(f"{candidate['candidate_id']} prompt IDs mismatch")

    predictions = np.empty(len(y), dtype=object)
    selected_candidate = np.empty(len(y), dtype=object)
    selected_rule = np.empty(len(y), dtype=object)
    folds_out = np.full(len(y), -1, dtype=int)
    traces, decision_frames, fold_stats = [], [], []
    reliability_by_fold = {}
    splitter = GroupKFold(n_splits=int(folds))
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(np.zeros(len(y)), y, group_ids), start=1):
        fit_candidates = [{**candidate, "probabilities": np.asarray(candidate["probabilities"])[fit_idx]}
                          for candidate in candidates]
        rules = _learn_rules(fit_candidates, y[fit_idx], classes,
            minimum_support=minimum_support, prior_alpha=prior_alpha, prior_beta=prior_beta,
            candidate_order=order)
        reliability_by_fold[str(fold)] = {"fit_rows": int(len(fit_idx)),
            "held_out_rows": int(len(held_idx)), "minimum_support": minimum_support,
            "prior_alpha": prior_alpha, "prior_beta": prior_beta,
            "rules": rules}
        held_candidates = [{**candidate, "probabilities": np.asarray(candidate["probabilities"])[held_idx]}
                           for candidate in candidates]
        held_predictions, decisions, trace = _apply_rules(held_candidates, classes, rules,
            minimum_support=minimum_support, fallback_candidate_id="D",
            row_ids=np.asarray(validation_ids)[held_idx] if validation_ids is not None else None,
            prompt_ids=group_ids[held_idx], y_true=y[held_idx], fold=fold)
        predictions[held_idx] = held_predictions
        selected_candidate[held_idx] = decisions.selected_candidate_id.to_numpy()
        selected_rule[held_idx] = decisions.selected_rule_rank.to_numpy()
        folds_out[held_idx] = fold
        decisions["row_index"] = held_idx
        decision_frames.append(decisions)
        traces.extend(trace)
        fold_stats.append({"fold": fold, "fit_rows": int(len(fit_idx)),
            "held_out_rows": int(len(held_idx)), "fit_prompt_groups": int(len(set(group_ids[fit_idx]))),
            "held_out_prompt_groups": int(len(set(group_ids[held_idx]))),
            "eligible_rules": int(sum(rule["eligible"] for rule in rules)),
            "fallback_rows": int(sum(decisions.selected_rule_rank == "D_fallback"))})

    metrics = _metrics(y, predictions, classes)
    metrics.update({"folds": fold_stats, "minimum_support": minimum_support,
        "precision_smoothing": {"alpha": prior_alpha, "beta": prior_beta,
            "formula": "(TP + alpha) / (prediction_count + alpha + beta)"},
        "validation_rows": int(len(y)), "prompt_groups": int(len(set(group_ids))),
        "validation_split_sha256": validation_split_sha256,
        "validation_ids_sha256": hashlib.sha256("\n".join(map(str, validation_ids or range(len(y)))).encode()).hexdigest(),
        "candidate_order": order})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({"id": list(map(str, validation_ids)) if validation_ids is not None else list(map(str, range(len(y)))),
        "prompt_id": group_ids, "fold": folds_out, "y_true": y, "prediction": predictions,
        "selected_candidate_id": selected_candidate, "selected_rule_rank": selected_rule})
    for candidate in candidates:
        cid = candidate["candidate_id"]
        oof[f"prediction::{cid}"] = _top1(candidate["probabilities"], classes)
    rules_frame = pd.DataFrame(traces)
    rules_frame["rule_rank"] = rules_frame["rule_rank"].astype(str)
    decision_frame = pd.concat(decision_frames, ignore_index=True).sort_values("row_index", kind="stable")
    paths = {"oof_predictions": out_dir / "oof_predictions.parquet",
        "routing_decisions": out_dir / "routing_decisions.parquet",
        "rule_trace": out_dir / "rule_trace.parquet",
        "rule_reliability": out_dir / "rule_reliability.json",
        "metrics": out_dir / "metrics.json"}
    oof.to_parquet(paths["oof_predictions"], index=False)
    decision_frame.to_parquet(paths["routing_decisions"], index=False)
    rules_frame.to_parquet(paths["rule_trace"], index=False)
    paths["rule_reliability"].write_text(json.dumps(reliability_by_fold, indent=2) + "\n", encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_report(Path(report_path), metrics, rules_frame, order)
    return {"report_path": str(report_path), "artifact_paths": {k: str(v) for k, v in paths.items()},
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "recall_ge_80": metrics["recall_ge_80"], "recall_ge_90": metrics["recall_ge_90"]}


def run_ultrabert_zod01(*, artifact_dir="artifacts/predictions", data_dir="data/processed",
                        output_dir="artifacts/analysis/zod01_ultrabert",
                        report_path="docs/zod01_ultrabert.md", minimum_support=DEFAULT_MINIMUM_SUPPORT,
                        prior_alpha=DEFAULT_PRIOR_ALPHA, prior_beta=DEFAULT_PRIOR_BETA):
    from .pipeline import _load_dataset
    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.eq("validation"), ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["y_true"] = validation.pop("model").astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540 or validation.prompt_id.nunique() != 30:
        raise ValueError("expected frozen V0 validation with 540 examples and 30 prompt groups")
    classes = sorted(validation.y_true.unique().tolist())
    candidates, problems = _read_candidates(Path(artifact_dir), min_validation_rows=540,
        candidate_ids=ULTRABERT_CHECKPOINTS.values(), deduplicate=False)
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    missing = set(ULTRABERT_CHECKPOINTS.values()) - set(by_id)
    if missing:
        raise FileNotFoundError(f"missing checkpoint predictions {sorted(missing)}; issues={problems}")
    _, y, groups, aligned, excluded, ids = _align_candidates(candidates,
        reference_dataset="v0", reference_frame=validation)
    aligned_by_id = {candidate["candidate_id"]: candidate for candidate in aligned}
    ordered = []
    for alias, experiment_id in ULTRABERT_CHECKPOINTS.items():
        candidate = dict(aligned_by_id[experiment_id])
        if candidate["model"] != "ultrabert380":
            raise ValueError(f"candidate {alias} is not UltraBERT")
        candidate.update({"candidate_id": alias, "classes": classes, "ids": ids, "prompt_ids": groups,
                          "experiment_id": experiment_id})
        ordered.append(candidate)
    profile = json.loads((Path(data_dir) / "dataset_profile.json").read_text())
    result = run_zod01(candidates=ordered, y_true=y, groups=groups, classes=classes,
        validation_ids=ids, candidate_order=list(ULTRABERT_CHECKPOINTS), folds=5,
        minimum_support=minimum_support, prior_alpha=prior_alpha, prior_beta=prior_beta,
        output_dir=output_dir, report_path=report_path,
        validation_split_sha256=profile.get("split_sha256"))
    metrics_path = Path(result["artifact_paths"]["metrics"])
    metrics = json.loads(metrics_path.read_text())
    metrics["candidate_artifacts"] = {alias: {"experiment_id": experiment_id,
        "manifest_path": aligned_by_id[experiment_id]["manifest_path"],
        "prediction_path": str((Path(artifact_dir) / f"{experiment_id}_validation.parquet").resolve())}
        for alias, experiment_id in ULTRABERT_CHECKPOINTS.items()}
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return result


def _write_report(path, metrics, rule_trace, order):
    per_label = metrics["per_label"]
    lines = ["# zod01 — Sequential Precision-Ranked UltraBERT Ensemble", "",
        "## Method and validation", "",
        f"Five-fold GroupKFold by prompt ID evaluates {metrics['validation_rows']} frozen V0 examples ({metrics['prompt_groups']} prompt groups, 18 labels). Each fold learns its rule statistics and ordering from the other four folds only. Checkpoints were previously selected using V0, so these are development diagnostics, not an independent generalization estimate.", "",
        f"All 72 checkpoint-label rules are ranked by Beta-smoothed precision `(TP+{metrics['precision_smoothing']['alpha']:g})/(prediction_count+{metrics['precision_smoothing']['alpha']:g}+{metrics['precision_smoothing']['beta']:g})`; tie-breaks are raw precision, prediction support, checkpoint order `{', '.join(order)}`, and canonical label order. A rule is applied only when its fitting-fold prediction count is at least {metrics['minimum_support']}; unsupported rules remain in the audit sequence but are skipped. Remaining held-out rows receive checkpoint D's top-1 prediction.", "",
        f"Split SHA-256: `{metrics.get('validation_split_sha256')}`; validation ID SHA-256: `{metrics.get('validation_ids_sha256')}`.", "",
        "## Overall result", "",
        "| Rows | Correct | Accuracy | Balanced accuracy | Macro F1 | Labels recall ≥80% | Labels recall ≥90% |", "|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| {metrics['rows']} | {metrics['correct']} | {metrics['accuracy']:.1%} | {metrics['balanced_accuracy']:.1%} | {metrics['macro_f1']:.1%} | {metrics['recall_ge_80']} | {metrics['recall_ge_90']} |")
    lines.extend(["", "## Per-label precision and recall", "",
        "| Label | TP | FP | FN | Predicted | Precision | Recall |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label, row in per_label.items():
        mark = " **≥90%**" if row["recall"] >= .90 else " **≥80%**" if row["recall"] >= .80 else ""
        lines.append(f"| {label} | {row['true_positive']} | {row['false_positive']} | {row['false_negative']} | {row['prediction_count']} | {row['precision']:.1%} | {row['recall']:.1%}{mark} |")
    lines.extend(["", "## Sequential rule trace", "",
        "Counts are from each rule applied to that fold's held-out rows; 'remaining' is after the rule. Support and precision used for ordering come only from the fitting portion. Rule ranks can differ by fold, so sequences are shown separately.", ""])
    for fold in sorted(int(value) for value in rule_trace.fold.dropna().unique()):
        lines.extend([f"### Fold {fold}", "",
            "| Rank | Rule | Fit support | Smoothed precision | Assigned | Correct | Incorrect | Remaining |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
        current = rule_trace.loc[rule_trace.fold.eq(fold)].sort_values("rule_rank", key=lambda col: col.map(lambda value: 10**9 if value == "fallback" else int(value)))
        for _, row in current.iterrows():
            rank = str(row["rule_rank"])
            rule_name = "D fallback" if rank == "fallback" else f"{row['candidate_id']} → {row['label']}" + (" (skipped: support < minimum)" if not bool(row["eligible"]) else "")
            smoothed = "—" if pd.isna(row.get("smoothed_precision")) else f"{float(row['smoothed_precision']):.1%}"
            support = "—" if pd.isna(row.get("prediction_count")) else str(int(row["prediction_count"]))
            lines.append(f"| {rank} | {rule_name} | {support} | {smoothed} | {int(row['assigned_examples'])} | {int(row['correct'])} | {int(row['incorrect'])} | {int(row['remaining_after'])} |")
        lines.append("")
    lines.extend(["Artifacts: `oof_predictions.parquet`, `routing_decisions.parquet`, `rule_trace.parquet`, `rule_reliability.json`, and `metrics.json`.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
