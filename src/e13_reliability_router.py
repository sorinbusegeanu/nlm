"""CPU CatBoost routing by fold-local checkpoint correctness estimates."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

from . import ensemble_catalog as catalog
from .confusion_ensemble import error_correlations
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
CLASSES = np.asarray(LABELS, dtype=object)
CONFIGS = {"E13-A": ("checkpoint", 15), "E13-B": ("family_max", 15),
           "E13-C": ("family_max", 20)}
SEED = 2026
SMOOTHING = 1.0
CATBOOST_PARAMS = {
    "loss_function": "Logloss", "eval_metric": "Logloss", "task_type": "CPU",
    "depth": 7, "learning_rate": 0.04, "iterations": 1500,
    "l2_leaf_reg": 5.0, "random_seed": SEED, "thread_count": 20,
    "od_type": "Iter", "od_wait": 120, "verbose": False,
    "allow_writing_files": False, "random_strength": 0.5,
}
FEATURE_COLUMNS = (
    "checkpoint_id", "model_family", "predicted_label", "competing_label",
    "top1_probability", "top2_probability", "margin", "normalized_entropy",
    "label_precision", "label_recall", "pred_to_competitor_rate",
    "competitor_to_pred_rate", "agreement_count", "family_agreement_count",
    "ensemble_pred_probability", "ensemble_competitor_probability",
    "checkpoint_ensemble_pred_delta", "checkpoint_ensemble_competitor_delta",
)
CATEGORICAL = ("checkpoint_id", "model_family", "predicted_label", "competing_label")


def _sorted_ids(probabilities):
    return sorted(probabilities, key=lambda sid: int(str(sid).lstrip("N")))


def select_roster(hard, truth, families, ids, limit, exclude_ids=()):
    """Greedily select by fitting-fold correct coverage, with deterministic ties."""
    hard, truth = np.asarray(hard, dtype=np.int64), np.asarray(truth, dtype=np.int64)
    support = np.bincount(truth, minlength=len(LABELS))
    if hard.ndim != 2 or hard.shape[1] != len(truth) or not support.all():
        raise ValueError("Roster fitting data must cover all canonical labels and align")
    if not (len(hard) == len(families) == len(ids)):
        raise ValueError("Checkpoint metadata does not align with prediction rows")
    hits = hard == truth
    correct_by_label = np.stack([
        np.bincount(truth[hit], minlength=len(LABELS)) for hit in hits
    ])
    balanced = (correct_by_label / support[None, :]).mean(axis=1)
    accuracy = hits.mean(axis=1)
    ranked = sorted(range(len(ids)), key=lambda i: (-balanced[i], -accuracy[i], str(ids[i])))
    selected = [ranked[0]]
    covered = hits[ranked[0]].copy()
    corr = error_correlations(hard, truth)
    steps = [{"checkpoint_id": str(ids[ranked[0]]),
              "newly_covered": int(covered.sum()),
              "balanced_accuracy": float(balanced[ranked[0]]),
              "accuracy": float(accuracy[ranked[0]]), "new_family": True}]
    while len(selected) < min(int(limit), len(ids)):
        existing_families = {families[i] for i in selected}
        remaining = [i for i in ranked if i not in selected]
        gain = (hits & ~covered[None, :]).sum(axis=1)
        chosen = min(remaining, key=lambda i: (
            -int(gain[i]), float(corr[i, selected].mean()),
            families[i] in existing_families, -balanced[i], -accuracy[i], str(ids[i])))
        selected.append(chosen)
        covered |= hits[chosen]
        steps.append({"checkpoint_id": str(ids[chosen]),
                      "newly_covered": int(gain[chosen]),
                      "mean_error_correlation": float(corr[chosen, selected[:-1]].mean()),
                      "balanced_accuracy": float(balanced[chosen]),
                      "accuracy": float(accuracy[chosen]),
                      "new_family": families[chosen] not in existing_families})
    excluded = set(map(str, exclude_ids))
    retained = [i for i in selected if str(ids[i]) not in excluded]
    retained_steps = [step for step in steps if step["checkpoint_id"] not in excluded]
    if not retained:
        raise ValueError("Roster exclusion removed every selected checkpoint")
    return retained, retained_steps


def reliability_statistics(hard, truth, alpha=SMOOTHING, sample_weights=None):
    """Fold-local smoothed precision, recall and directional confusion."""
    hard, truth = np.asarray(hard, dtype=np.int64), np.asarray(truth, dtype=np.int64)
    k = len(LABELS)
    if hard.ndim != 2 or hard.shape[1] != len(truth) or alpha <= 0:
        raise ValueError("Invalid reliability-statistic inputs")
    weights = (np.ones(len(truth), dtype=np.float64) if sample_weights is None
               else np.asarray(sample_weights, dtype=np.float64))
    if weights.shape != (len(truth),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Reliability sample weights must be finite positive row weights")
    counts = np.stack([
        np.bincount(truth * k + row, weights=weights, minlength=k * k).reshape(k, k)
        for row in hard
    ])
    support = np.bincount(truth, weights=weights, minlength=k)
    tp = np.diagonal(counts, axis1=1, axis2=2)
    predicted_support = counts.sum(axis=1)
    precision = (tp + alpha) / (predicted_support + 2 * alpha)
    recall = (tp + alpha) / (support[None, :] + 2 * alpha)
    directional = (counts + alpha) / (support[None, :, None] + k * alpha)
    return {"counts": counts, "support": support, "precision": precision,
            "recall": recall, "directional": directional}


def family_balanced_probabilities(probabilities, families):
    """Mean checkpoints within each family, then give each family equal weight."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    names = sorted(set(families))
    if probabilities.ndim != 3 or probabilities.shape[0] != len(families) or not names:
        raise ValueError("Family probability inputs do not align")
    by_family = [probabilities[np.asarray(families) == name].mean(axis=0) for name in names]
    result = np.mean(by_family, axis=0)
    sums = result.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or np.any(sums <= 0):
        raise ValueError("Invalid family-balanced probabilities")
    return result / sums


def build_feature_frame(probabilities, ids, families, truth_fit, fit_rows, target_rows,
                        *, alpha=SMOOTHING, fit_sample_weights=None):
    """Build one row per selected-checkpoint/target-example pair."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != len(ids) or p.shape[0] != len(families):
        raise ValueError("Probability tensor and checkpoint metadata do not align")
    if p.shape[2] != len(LABELS) or not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite and use canonical labels")
    if not np.allclose(p.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError("Checkpoint probabilities must be normalized")
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    if np.intersect1d(fit_rows, target_rows).size:
        raise ValueError("Reliability fit and feature target rows must be disjoint")
    truth_fit = np.asarray(truth_fit, dtype=np.int64)
    if len(truth_fit) != len(fit_rows):
        raise ValueError("Training labels and fitting rows do not align")

    fit_hard = p[:, fit_rows].argmax(axis=2)
    stats = reliability_statistics(fit_hard, truth_fit, alpha, fit_sample_weights)
    target_prob = p[:, target_rows]
    n_checkpoints, n_rows, k = target_prob.shape
    top = target_prob.argmax(axis=2)
    sorted_prob = np.sort(target_prob, axis=2)
    top1 = sorted_prob[:, :, -1]
    top2 = sorted_prob[:, :, -2]
    margin = top1 - top2
    entropy = -(np.clip(target_prob, 1e-12, 1.0) *
                np.log(np.clip(target_prob, 1e-12, 1.0))).sum(axis=2) / np.log(k)
    ensemble = family_balanced_probabilities(target_prob, families)
    ensemble_rank = np.argsort(-ensemble, axis=1, kind="stable")
    row = np.arange(n_rows)
    ensemble_pred = ensemble.argmax(axis=1)
    competing = np.where(ensemble_pred[None, :] == top,
                         ensemble_rank[:, 1][None, :], ensemble_pred[None, :])
    # If the global leader is a checkpoint's own label, use the next best alternative.
    for checkpoint in range(n_checkpoints):
        same = top[checkpoint] == ensemble_pred
        if same.any():
            competing[checkpoint, same] = ensemble_rank[same, 1]
        # A checkpoint whose prediction is not the ensemble leader competes with that leader.
    own_ensemble = ensemble[row[None, :], top]
    alt_ensemble = ensemble[row[None, :], competing]
    own_checkpoint = target_prob[np.arange(n_checkpoints)[:, None], row[None, :], top]
    alt_checkpoint = target_prob[np.arange(n_checkpoints)[:, None], row[None, :], competing]

    agreements = (top[:, None, :] == top[None, :, :]).sum(axis=1) - 1
    family_array = np.asarray(families, dtype=object)
    family_agreement = np.zeros((n_checkpoints, n_rows), dtype=np.float64)
    for checkpoint in range(n_checkpoints):
        family_agreement[checkpoint] = sum(
            np.any(top[family_array == family] == top[checkpoint], axis=0)
            for family in sorted(set(families)) if family != families[checkpoint]
        )

    frame = pd.DataFrame({
        "checkpoint_id": np.tile(np.asarray(ids, dtype=str), n_rows),
        "model_family": np.tile(family_array.astype(str), n_rows),
        "predicted_label": CLASSES[top].T.reshape(-1),
        "competing_label": CLASSES[competing].T.reshape(-1),
        "top1_probability": top1.T.reshape(-1),
        "top2_probability": top2.T.reshape(-1), "margin": margin.T.reshape(-1),
        "normalized_entropy": entropy.T.reshape(-1),
        "label_precision": stats["precision"][np.arange(n_checkpoints)[:, None], top].T.reshape(-1),
        "label_recall": stats["recall"][np.arange(n_checkpoints)[:, None], top].T.reshape(-1),
        "pred_to_competitor_rate": stats["directional"][
            np.arange(n_checkpoints)[:, None], top, competing].T.reshape(-1),
        "competitor_to_pred_rate": stats["directional"][
            np.arange(n_checkpoints)[:, None], competing, top].T.reshape(-1),
        "agreement_count": agreements.T.reshape(-1),
        "family_agreement_count": family_agreement.T.reshape(-1),
        "ensemble_pred_probability": own_ensemble.T.reshape(-1),
        "ensemble_competitor_probability": alt_ensemble.T.reshape(-1),
        "checkpoint_ensemble_pred_delta": (own_checkpoint - own_ensemble).T.reshape(-1),
        "checkpoint_ensemble_competitor_delta": (alt_checkpoint - alt_ensemble).T.reshape(-1),
    })
    for col in CATEGORICAL:
        frame[col] = frame[col].fillna("").astype(str)
    numeric = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite E13 features")
    return frame.loc[:, FEATURE_COLUMNS], top


def _targets(hard, truth):
    return (np.asarray(hard, dtype=np.int64).T == np.asarray(truth, dtype=np.int64)[:, None]).astype(np.int8).reshape(-1)


def _cat_frame(frame):
    result = frame.copy()
    for col in CATEGORICAL:
        result[col] = result[col].astype(str)
    return result


def _new_catboost(iterations=None):
    params = dict(CATBOOST_PARAMS)
    if iterations is not None:
        params["iterations"] = max(1, int(iterations))
        params.pop("od_type", None)
        params.pop("od_wait", None)
    return CatBoostClassifier(**params)


def _score_correctness(model, frame):
    return np.asarray(model.predict_proba(_cat_frame(frame))[:, 1], dtype=np.float64)


def route_predictions(method, hard, reliability, ensemble_probs, families):
    """Route checkpoint correctness probabilities to a label prediction."""
    hard = np.asarray(hard, dtype=np.int64)
    reliability = np.asarray(reliability, dtype=np.float64)
    if hard.shape != reliability.shape or hard.ndim != 2:
        raise ValueError("Hard predictions and reliability scores must be [checkpoints, rows]")
    n_checkpoints, n_rows = hard.shape
    if method == "checkpoint":
        pred = np.empty(n_rows, dtype=np.int64)
        selected = np.empty(n_rows, dtype=np.int64)
        for row in range(n_rows):
            best_score = reliability[:, row].max()
            tied = np.flatnonzero(np.isclose(reliability[:, row], best_score, rtol=0, atol=1e-12))
            chosen = min(tied, key=lambda i: (-ensemble_probs[row, hard[i, row]], i))
            selected[row], pred[row] = chosen, hard[chosen, row]
        return pred, selected
    if method != "family_max":
        raise ValueError(f"Unknown E13 routing method: {method}")
    label_scores = np.full((n_rows, len(LABELS)), -np.inf, dtype=np.float64)
    for row in range(n_rows):
        for label in range(len(LABELS)):
            voters = np.flatnonzero(hard[:, row] == label)
            if not len(voters):
                continue
            by_family = {}
            for idx in voters:
                family = families[idx]
                by_family[family] = max(by_family.get(family, -np.inf), reliability[idx, row])
            label_scores[row, label] = float(np.mean(list(by_family.values())))
    pred = np.empty(n_rows, dtype=np.int64)
    selected = np.full(n_rows, -1, dtype=np.int64)
    for row in range(n_rows):
        best = np.max(label_scores[row])
        tied = np.flatnonzero(np.isclose(label_scores[row], best, rtol=0, atol=1e-12))
        pred[row] = min(tied, key=lambda label: (-ensemble_probs[row, label], label))
        voters = np.flatnonzero(hard[:, row] == pred[row])
        selected[row] = min(voters, key=lambda i: (-reliability[i, row], i))
    return pred, selected


def _metrics(truth, pred, chosen, correctness_truth, correctness_probability, hard, baseline):
    cm = confusion_matrix(truth, pred, labels=np.arange(len(LABELS)))
    precision, recall, _, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "corrected_vs_H09": int(((baseline != truth) & (pred == truth)).sum()),
        "introduced_vs_H09": int(((baseline == truth) & (pred != truth)).sum()),
        "checkpoint_correctness_accuracy": float(accuracy_score(
            correctness_truth, correctness_probability >= 0.5)),
        "checkpoint_correctness_logloss": float(-np.mean(
            correctness_truth * np.log(np.clip(correctness_probability, 1e-12, 1)) +
            (1 - correctness_truth) * np.log(np.clip(1 - correctness_probability, 1e-12, 1)))),
        "reliability_selected_checkpoint_accuracy": float(np.mean(hard[chosen, np.arange(len(truth))] == truth)),
        "oracle_coverage": float((hard == truth[None, :]).any(axis=0).mean()),
        "confusion_matrix": cm.tolist(),
        "per_label": {str(LABELS[i]): {"support": int(support[i]),
                                      "precision": float(precision[i]),
                                      "recall": float(recall[i]),
                                      "true_positive": int(cm[i, i]),
                                      "false_positive": int(cm[:, i].sum() - cm[i, i]),
                                      "false_negative": int(cm[i].sum() - cm[i, i])}
                       for i in range(len(LABELS))},
    }


def _fit_router(probabilities, ids, families, truth, groups, fit_rows, held_rows,
                limit, method, baseline, row_ids, model_path=None, exclude_ids=()):
    """Fit one outer router, with isolated early stopping and cross-fit records."""
    started = time.monotonic()
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    held_rows = np.asarray(held_rows, dtype=np.int64)
    early_split = list(GroupKFold(n_splits=4).split(
        fit_rows, truth[fit_rows], groups[fit_rows]))[-1]
    local_train, local_stop = early_split
    router_fit_rows = fit_rows[local_train]
    stop_rows = fit_rows[local_stop]

    def records_for(rows, inner_folds=4):
        xs, ys, logs = [], [], []
        for local_fit, local_valid in GroupKFold(n_splits=inner_folds).split(
                rows, truth[rows], groups[rows]):
            stats_fit, valid = rows[local_fit], rows[local_valid]
            roster, steps = select_roster(
                probabilities[:, stats_fit].argmax(axis=2), truth[stats_fit],
                families, ids, limit, exclude_ids)
            selected_ids = [ids[i] for i in roster]
            selected_families = [families[i] for i in roster]
            frame, top = build_feature_frame(
                probabilities[roster], selected_ids, selected_families,
                truth[stats_fit], stats_fit, valid)
            hard = probabilities[roster][:, valid].argmax(axis=2)
            ys.append(_targets(hard, truth[valid]))
            xs.append(frame)
            logs.append({"fit_ids": [str(row_ids[i]) for i in stats_fit],
                         "held_ids": [str(row_ids[i]) for i in valid],
                         "checkpoint_ids": selected_ids,
                         "selection_steps": steps,
                         "record_count": int(len(frame))})
        return pd.concat(xs, ignore_index=True), np.concatenate(ys), logs

    train_early_X, train_early_y, early_records = records_for(router_fit_rows)
    # Stopping features use statistics and roster fit only on router-fit groups.
    stop_roster, stop_steps = select_roster(
        probabilities[:, router_fit_rows].argmax(axis=2), truth[router_fit_rows],
        families, ids, limit, exclude_ids)
    stop_ids = [ids[i] for i in stop_roster]
    stop_families = [families[i] for i in stop_roster]
    stop_X, _ = build_feature_frame(
        probabilities[stop_roster], stop_ids, stop_families,
        truth[router_fit_rows], router_fit_rows, stop_rows)
    stop_hard = probabilities[stop_roster][:, stop_rows].argmax(axis=2)
    stop_y = _targets(stop_hard, truth[stop_rows])
    early = _new_catboost()
    early.fit(_cat_frame(train_early_X), train_early_y,
              cat_features=list(CATEGORICAL),
              eval_set=(_cat_frame(stop_X), stop_y),
              early_stopping_rounds=CATBOOST_PARAMS["od_wait"],
              verbose=False, use_best_model=True)
    best_iterations = max(1, int(early.get_best_iteration()) + 1)

    train_X, train_y, train_records = records_for(fit_rows)
    model = _new_catboost(iterations=best_iterations)
    model.fit(_cat_frame(train_X), train_y, cat_features=list(CATEGORICAL), verbose=False)
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))

    outer_roster, outer_steps = select_roster(
        probabilities[:, fit_rows].argmax(axis=2), truth[fit_rows], families, ids, limit,
        exclude_ids)
    outer_ids = [ids[i] for i in outer_roster]
    outer_families = [families[i] for i in outer_roster]
    held_X, _ = build_feature_frame(
        probabilities[outer_roster], outer_ids, outer_families,
        truth[fit_rows], fit_rows, held_rows)
    reliability_flat = _score_correctness(model, held_X)
    reliability = reliability_flat.reshape(len(held_rows), len(outer_roster)).T
    hard = probabilities[outer_roster][:, held_rows].argmax(axis=2)
    ensemble = family_balanced_probabilities(probabilities[outer_roster][:, held_rows], outer_families)
    pred, selected_local = route_predictions(method, hard, reliability, ensemble, outer_families)
    correctness = _targets(hard, truth[held_rows])
    metrics = _metrics(truth[held_rows], pred, selected_local, correctness,
                       reliability.T.reshape(-1), hard, baseline[held_rows])
    selected_ids = outer_ids
    chosen_counts = {sid: int(np.count_nonzero(selected_local == i))
                     for i, sid in enumerate(selected_ids)}
    metrics.update({"checkpoint_ids": selected_ids, "selection_steps": outer_steps,
                    "chosen_checkpoint_counts": chosen_counts,
                    "held_ids": [str(row_ids[i]) for i in held_rows],
                    "fit_ids": [str(row_ids[i]) for i in fit_rows],
                    "training_record_count": int(len(train_X)),
                    "early_stopping_record_count": int(len(stop_X)),
                    "best_iterations": best_iterations,
                    "early_stopping_best_iteration": int(early.get_best_iteration()),
                    "inner_fit_folds": train_records,
                    "early_stopping_feature_folds": early_records,
                    "early_stopping_roster": stop_ids,
                    "early_stopping_selection_steps": stop_steps,
                    "model_path": str(model_path) if model_path else None,
                    "held_row_count": int(len(held_rows)),
                    "correctness_record_count": int(len(correctness)),
                    "training_seconds": float(time.monotonic() - started),
                    "process_peak_memory_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)})
    return pred, metrics, model, (reliability, hard, outer_roster)


def _load_references(root, v0):
    row_ids = v0.id.astype(str).tolist()
    labels = v0.model.astype(str).to_numpy()
    h09_frame = pd.read_parquet(root / "artifacts/analysis/ensemble_hypotheses_nov15/predictions.parquet")
    oracle_frame = pd.read_parquet(root / "artifacts/analysis/oracle04/row_predictions.parquet")
    for frame in (h09_frame, oracle_frame):
        if frame.id.astype(str).tolist() != row_ids or frame.y_true.astype(str).tolist() != labels.tolist():
            raise ValueError("Historical reference rows do not align with V0")
    h09 = pd.Categorical(h09_frame["pred::H09-EQUAL"].astype(str), categories=LABELS).codes
    oracle = pd.Categorical(oracle_frame["pred::combined_logistic_core"].astype(str), categories=LABELS).codes
    if (h09 < 0).any() or (oracle < 0).any():
        raise ValueError("Historical reference contains non-canonical labels")
    return h09.astype(np.int64), oracle.astype(np.int64)


def _write_report(root, output, run_result, v0, h09, oracle, report_path=None):
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes
    lines = ["# E13 — Learned Checkpoint Reliability Routing", "",
             f"Trust List: **{run_result['trust_list']['version']}** · "
             f"V0 rows: **{run_result['validation_rows']}** · "
             "five outer prompt-group folds · CPU CatBoost · cached checkpoint probabilities only.", "",
             "Each fold selects its roster using fitting rows only. Router training features are four-fold "
             "prompt-group cross-fitted; early-stopping rows use roster and reliability statistics fitted "
             "on disjoint prompt groups. These are V0 development results because V0 is also used for "
             "checkpoint selection and router fitting.", "",
             "## Comparison", "",
             "| Ensemble | Accuracy | Balanced accuracy | Corrected vs H09 | Introduced vs H09 | "
             "Checkpoint-correctness accuracy | Selected-checkpoint accuracy | Oracle coverage |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    h09_acc = accuracy_score(truth, h09)
    h09_ba = balanced_accuracy_score(truth, h09)
    oracle_acc = accuracy_score(truth, oracle)
    oracle_ba = balanced_accuracy_score(truth, oracle)
    lines.append(f"| H09 EQUAL | {h09_acc:.4%} | {h09_ba:.4%} | — | — | — | — | — |")
    lines.append(f"| ORACLE04 combined logistic | {oracle_acc:.4%} | {oracle_ba:.4%} | — | — | — | — | — |")
    for item in run_result["results"]:
        lines.append(
            f"| {item['ensemble']} | {item['accuracy']:.4%} | {item['balanced_accuracy']:.4%} | "
            f"{item['corrected_vs_H09']} | {item['introduced_vs_H09']} | "
            f"{item['checkpoint_correctness_accuracy']:.4%} | "
            f"{item['reliability_selected_checkpoint_accuracy']:.4%} | "
            f"{item['oracle_coverage']:.4%} |")
    lines += ["", "ORACLE04 comparison uses the saved five-fold prompt-group cross-fitted `combined_logistic_core` "
              "predictions. ORACLE04 has its own nested cross-fitting and strategy selection, so the figure "
              "is a historical reference rather than a model with identical features or tuning.", "",
              "## Fold results", "",
              "| Experiment | Fold | Rows | Accuracy | Balanced accuracy | Correctness accuracy | "
              "Selected-checkpoint accuracy | Coverage | Trees | Seconds | Peak RSS (MB) | Roster |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in run_result["results"]:
        for fold in item["folds"]:
            lines.append(
                f"| {item['ensemble']} | {fold['fold']} | {fold['held_row_count']} | "
                f"{fold['accuracy']:.4%} | {fold['balanced_accuracy']:.4%} | "
                f"{fold['checkpoint_correctness_accuracy']:.4%} | "
                f"{fold['reliability_selected_checkpoint_accuracy']:.4%} | "
                f"{fold['oracle_coverage']:.4%} | {fold['best_iterations']} | "
                f"{fold['training_seconds']:.1f} | {fold['process_peak_memory_mb']:.0f} | "
                f"{', '.join(fold['checkpoint_ids'])} |")
    lines += ["", "## Per-label precision and recall", ""]
    for item in run_result["results"]:
        lines += [f"### {item['ensemble']}", "",
                  "| Label | Support | Precision | Recall | TP | FP | FN |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label in LABELS:
            m = item["per_label"][label]
            lines.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} | "
                         f"{m['true_positive']} | {m['false_positive']} | {m['false_negative']} |")
        lines += ["", "Confusion matrix (true labels as rows, predicted labels as columns):", "",
                  "```text", "\t" + "\t".join(LABELS)]
        for label, row in zip(LABELS, item["confusion_matrix"]):
            lines.append(label + "\t" + "\t".join(str(value) for value in row))
        lines += ["```", "", "Selected checkpoint usage:", ""]
        lines += [f"- {sid}: routed {count} examples; rostered in {item['roster_fold_frequency'].get(sid, 0)} folds"
                  for sid, count in sorted(item["selected_checkpoint_frequencies"].items(),
                                           key=lambda pair: (-pair[1], int(pair[0][1:])))]
        lines.append("")
    lines += ["## Artifacts", "",
              f"- Run directory: `{output}`",
              "- `predictions.parquet`: one held-out prediction per V0 row and experiment.",
              "- `results.json`: fold rosters, fold metrics, selection steps, per-label metrics, and confusion matrices.",
              "- `config.json`: Trust List snapshot, seed, CatBoost settings, smoothing, and routing rules.",
              "- `E13-*_fold*.cbm`: CatBoost models trained for the outer folds.", ""]
    excluded = run_result.get("excluded_checkpoints", [])
    if excluded:
        lines[1] += f" · Excluded without replacement: {', '.join(excluded)}"
    report = "\n".join(lines)
    (output / "e13.md").write_text(report)
    (report_path or root / "docs/e13.md").write_text(report)


def _run_configuration(code, method, limit, v0, ids, members, probabilities,
                       baseline, folds, output_dir=None, *, reuse_cache=None,
                       return_cache=False, exclude_ids=()):
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    groups = v0.prompt_id.astype(str).to_numpy()
    matrices = np.stack([probabilities[sid] for sid in ids])
    families = [_model_family(members[sid]["model"]) for sid in ids]
    predictions = np.full(len(v0), -1, dtype=np.int64)
    assignment_counts = np.zeros(len(v0), dtype=np.int8)
    fold_results, saved_paths, new_cache = [], [], []
    row_ids = v0.id.astype(str).to_numpy()
    for number, (fit, held) in enumerate(GroupKFold(n_splits=folds).split(v0, truth, groups), 1):
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Outer prompt groups overlap")
        if reuse_cache is None:
            model_path = output_dir / f"{code}_fold{number}.cbm" if output_dir else None
            pred, metrics, model, aux = _fit_router(
                matrices, ids, families, truth, groups, fit, held, limit, method,
                baseline, row_ids, model_path, exclude_ids)
            new_cache.append({"aux": aux, "metrics": metrics,
                              "model_path": str(model_path) if model_path else None})
        else:
            cached = reuse_cache[number - 1]
            reliability, hard, roster = cached["aux"]
            selected_ids = [ids[i] for i in roster]
            selected_families = [families[i] for i in roster]
            ensemble = family_balanced_probabilities(matrices[roster][:, held], selected_families)
            pred, selected_local = route_predictions(method, hard, reliability, ensemble,
                                                     selected_families)
            metrics = _metrics(truth[held], pred, selected_local, _targets(hard, truth[held]),
                               reliability.T.reshape(-1), hard, baseline[held])
            common_keys = ("checkpoint_ids", "selection_steps", "fit_ids", "held_ids",
                           "training_record_count", "early_stopping_record_count", "best_iterations",
                           "early_stopping_best_iteration", "inner_fit_folds",
                           "early_stopping_feature_folds", "early_stopping_roster",
                           "early_stopping_selection_steps", "model_path", "held_row_count",
                           "correctness_record_count", "training_seconds", "process_peak_memory_mb")
            metrics.update({key: cached["metrics"][key] for key in common_keys})
            metrics["chosen_checkpoint_counts"] = {
                sid: int(np.count_nonzero(selected_local == i))
                for i, sid in enumerate(selected_ids)}
            model_path = Path(cached["model_path"]) if cached["model_path"] else None
            new_cache.append(cached)
        predictions[held] = pred
        assignment_counts[held] += 1
        metrics["fold"] = number
        fold_results.append(metrics)
        if model_path:
            saved_paths.append(str(model_path))
        print(f"{code} fold {number}/{folds}: accuracy={metrics['accuracy']:.4%}, "
              f"balanced_accuracy={metrics['balanced_accuracy']:.4%}, "
              f"correctness_acc={metrics['checkpoint_correctness_accuracy']:.4%}", flush=True)
    if (predictions < 0).any() or not np.all(assignment_counts == 1):
        raise ValueError("Outer folds must cover every V0 row exactly once")
    # Fold results contain precisely the held rows; their truths are reconstructed for aggregation.
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    all_correctness, all_reliability = [], []
    chosen_frequency = {}
    roster_frequency = {}
    for result in fold_results:
        chosen_frequency.update({key: chosen_frequency.get(key, 0) + value
                                 for key, value in result["chosen_checkpoint_counts"].items()})
        for sid in result["checkpoint_ids"]:
            roster_frequency[sid] = roster_frequency.get(sid, 0) + 1
    # Compute report-level metrics from the complete held-out predictions.
    all_truth = truth
    accuracy = float(accuracy_score(all_truth, predictions))
    balanced = float(balanced_accuracy_score(all_truth, predictions))
    cm = confusion_matrix(all_truth, predictions, labels=np.arange(len(LABELS)))
    precision, recall, _, support = precision_recall_fscore_support(
        all_truth, predictions, labels=np.arange(len(LABELS)), zero_division=0)
    per_label = {str(LABELS[i]): {"support": int(support[i]), "precision": float(precision[i]),
                                  "recall": float(recall[i]), "true_positive": int(cm[i, i]),
                                  "false_positive": int(cm[:, i].sum() - cm[i, i]),
                                  "false_negative": int(cm[i].sum() - cm[i, i])}
                 for i in range(len(LABELS))}
    total_correctness_records = sum(item["correctness_record_count"] for item in fold_results)
    correctness_accuracy = sum(item["checkpoint_correctness_accuracy"] * item["correctness_record_count"]
                               for item in fold_results) / total_correctness_records
    correctness_logloss = sum(item["checkpoint_correctness_logloss"] * item["correctness_record_count"]
                              for item in fold_results) / total_correctness_records
    total_held = sum(item["held_row_count"] for item in fold_results)
    selected_acc = sum(item["reliability_selected_checkpoint_accuracy"] * item["held_row_count"]
                       for item in fold_results) / total_held
    oracle_coverage = sum(item["oracle_coverage"] * item["held_row_count"]
                          for item in fold_results) / total_held
    duration = sum(item["training_seconds"] for item in fold_results)
    peak_memory = max(item["process_peak_memory_mb"] for item in fold_results)
    result = {
        "ensemble": code, "routing_method": method, "checkpoints_per_fold": limit,
        "accuracy": accuracy, "balanced_accuracy": balanced,
        "corrected_vs_H09": int(((baseline != truth) & (predictions == truth)).sum()),
        "introduced_vs_H09": int(((baseline == truth) & (predictions != truth)).sum()),
        "checkpoint_correctness_accuracy": float(correctness_accuracy),
        "checkpoint_correctness_logloss": float(correctness_logloss),
        "reliability_selected_checkpoint_accuracy": float(selected_acc),
        "oracle_coverage": float(oracle_coverage),
        "training_duration_seconds": float(duration),
        "process_peak_memory_mb": float(peak_memory),
        "folds": fold_results, "per_label": per_label,
        "confusion_matrix": cm.tolist(), "selected_checkpoint_frequencies": chosen_frequency,
        "roster_fold_frequency": roster_frequency,
        "checkpoint_ids": sorted(roster_frequency, key=lambda sid: int(sid[1:])),
        "model_paths": saved_paths,
    }
    if return_cache:
        return predictions, result, new_cache
    return predictions, result


def evaluate_catalog_strategy(strategy, folds, v0, probabilities, members):
    code = strategy["id"]
    if code not in CONFIGS:
        raise ValueError(f"Unknown E13 strategy: {code}")
    method, limit = CONFIGS[code]
    ids = _sorted_ids(probabilities)
    if len(ids) < limit:
        raise ValueError(f"E13 {code} requires {limit} eligible cached checkpoints")
    h09, oracle = _load_references(ROOT, v0)
    pred, details = _run_configuration(code, method, limit, v0, ids, members,
                                       probabilities, h09, folds)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes
    details["comparison_vs_ORACLE04"] = {
        "accuracy": float(accuracy_score(truth, oracle)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, oracle)),
        "corrected": int(((oracle != truth) & (pred == truth)).sum()),
        "introduced": int(((oracle == truth) & (pred != truth)).sum()),
        "protocol": "Existing five prompt-group cross-fitted V0 predictions; ORACLE04 uses nested cross-fitting and its own selected strategy.",
    }
    return CLASSES[pred], details


def run(root=ROOT, output=None, *, strategies=None, exclude_ids=()):
    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output) if output else root / "artifacts/analysis/e13" / stamp
    if output.exists():
        raise FileExistsError(output)
    v0 = catalog.cascades._v0(root)
    _, members, probabilities, excluded, snapshot = catalog._load_inputs(root, v0)
    ids = _sorted_ids(probabilities)
    if len(ids) < 20:
        raise ValueError(f"E13-C needs 20 eligible cached checkpoints; found {len(ids)}")
    unknown_exclusions = set(map(str, exclude_ids)) - set(ids)
    if unknown_exclusions:
        raise ValueError(f"Unknown checkpoint exclusions: {sorted(unknown_exclusions)}")
    h09, oracle = _load_references(root, v0)
    selected = strategies or list(CONFIGS)
    if set(selected) - set(CONFIGS):
        raise ValueError(f"Unknown E13 strategy: {sorted(set(selected) - set(CONFIGS))}")
    output.mkdir(parents=True)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes
    predictions = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    results = []
    shared_15_cache = None
    for code in selected:
        method, limit = CONFIGS[code]
        # A and B share the same 15-checkpoint correctness models and differ only in routing.
        if code == "E13-B" and shared_15_cache is not None:
            pred, result = _run_configuration(
                code, method, limit, v0, ids, members, probabilities, h09,
                5, output_dir=output, reuse_cache=shared_15_cache,
                exclude_ids=exclude_ids)
        elif code == "E13-A":
            pred, result, shared_15_cache = _run_configuration(
                code, method, limit, v0, ids, members, probabilities, h09,
                5, output_dir=output, return_cache=True, exclude_ids=exclude_ids)
        else:
            pred, result = _run_configuration(
                code, method, limit, v0, ids, members, probabilities, h09,
                5, output_dir=output, exclude_ids=exclude_ids)
        predictions[code] = CLASSES[pred]
        result["comparison_vs_ORACLE04"] = {
            "accuracy": float(accuracy_score(truth, oracle)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, oracle)),
            "corrected": int(((oracle != truth) & (pred == truth)).sum()),
            "introduced": int(((oracle == truth) & (pred != truth)).sum()),
            "protocol": "Existing five prompt-group cross-fitted V0 predictions; ORACLE04 uses nested cross-fitting and its own selected strategy.",
        }
        results.append(result)
        print(f"{code}: accuracy={result['accuracy']:.4%}, "
              f"balanced_accuracy={result['balanced_accuracy']:.4%}, "
              f"corrected/introduced vs H09={result['corrected_vs_H09']}/"
              f"{result['introduced_vs_H09']}", flush=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "config.json").write_text(json.dumps({
        "trust_list": snapshot, "folds": 5, "seed": SEED,
        "catboost": CATBOOST_PARAMS, "smoothing_alpha": SMOOTHING,
        "inner_feature_folds": 4, "roster_selection": "BA-first greedy new-correct coverage; ties by error correlation, family diversity, BA, accuracy, checkpoint ID",
        "excluded_checkpoints_removed_without_replacement": sorted(map(str, exclude_ids)),
        "family_probability_aggregation": "mean checkpoints within family, then equal mean across families",
        "routing": {code: {"method": CONFIGS[code][0], "checkpoints": CONFIGS[code][1]}
                    for code in selected},
        "ids": ids, "families": {sid: _model_family(members[sid]["model"]) for sid in ids},
        "excluded": excluded,
    }, indent=2) + "\n")
    run_result = {"trust_list": snapshot, "validation_rows": len(v0),
                  "labels": list(LABELS), "row_ids": v0.id.astype(str).tolist(),
                  "excluded_checkpoints": sorted(map(str, exclude_ids)),
                  "results": results}
    (output / "results.json").write_text(json.dumps(run_result, indent=2, allow_nan=False) + "\n")
    if exclude_ids:
        exclusion_name = "_".join(sorted(map(str, exclude_ids)))
        report_path = root / "docs" / f"e13_without_{exclusion_name}.md"
        _write_report(root, output, run_result, v0, h09, oracle, report_path=report_path)
    else:
        _write_report(root, output, run_result, v0, h09, oracle)
    if not exclude_ids:
        active_catalog = catalog.load_catalog()
        catalog_entries = {entry["id"]: entry for entry in active_catalog["strategies"]}
        catalog_results = []
        for result in results:
            item = dict(result)
            item.update({"status": "ok", "strategy": catalog_entries[result["ensemble"]]})
            catalog_results.append(item)
        (output / "catalog_results.json").write_text(json.dumps({
            "catalog_version": active_catalog["version"],
            "catalog_sha256": catalog.cascades._sha256(catalog.CATALOG),
            "trust_list": snapshot, "results": catalog_results,
        }, indent=2, allow_nan=False) + "\n")
    print("| Experiment | Accuracy | Balanced accuracy | Corrected | Introduced |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for item in results:
        print(f"| {item['ensemble']} | {item['accuracy']:.4%} | {item['balanced_accuracy']:.4%} | "
              f"{item['corrected_vs_H09']} | {item['introduced_vs_H09']} |", flush=True)
    print(f"Saved E13 models, predictions, configuration, and results to {output}", flush=True)
    return run_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=str.upper, choices=CONFIGS,
                        help="Run one E13 variant; default runs all three")
    parser.add_argument("--exclude-checkpoint", action="append", default=[],
                        help="Remove an ID from each selected roster without replacing it; repeat as needed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root, args.output, strategies=[args.strategy] if args.strategy else None,
        exclude_ids=args.exclude_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
