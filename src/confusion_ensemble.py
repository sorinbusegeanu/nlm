"""CPU evidence aggregation from checkpoint confusion matrices and error correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from .v2_checkpoint_trust import LABELS


def select_coverage_roster(hard, truth, class_count, limit):
    """Select complementary top-1 coverage using fitting labels only.

    Callers order checkpoint rows by sequence ID to make final ties deterministic.
    """
    support = np.bincount(truth, minlength=class_count)
    if not np.all(support > 0):
        raise ValueError("The fitting partition must contain every canonical label")
    hits = hard == truth
    balanced = (hits / support[truth]).sum(axis=1) / class_count
    correct = hits.sum(axis=1)
    ranked = sorted(range(len(hard)), key=lambda i: (-balanced[i], -correct[i], i))
    roster = ranked[:1]
    covered = hits[roster[0]].copy()
    steps = [{"checkpoint_index": roster[0], "newly_covered": int(covered.sum())}]
    while len(roster) < min(limit, len(hard)):
        gains = (hits & ~covered).sum(axis=1)
        chosen = max((i for i in ranked if i not in roster), key=gains.__getitem__)
        if gains[chosen] == 0:
            break
        roster.append(chosen)
        steps.append({"checkpoint_index": chosen, "newly_covered": int(gains[chosen])})
        covered |= hits[chosen]
    return roster, steps


def error_correlations(hard, truth):
    """Positive Pearson error correlations, with a zero diagonal and constant rows."""
    errors = (hard != truth).astype(float)
    centered = errors - errors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = norms[:, None] * norms[None, :]
    correlations = np.divide(centered @ centered.T, denominator,
                             out=np.zeros_like(denominator), where=denominator > 0)
    correlations = np.clip(correlations, 0.0, 1.0)
    np.fill_diagonal(correlations, 0.0)
    return correlations


def _fit_statistics(hard, truth, class_count):
    # Rows are true labels and columns are checkpoint predictions.
    truth = np.asarray(truth, dtype=np.int64)
    counts = np.stack([
        np.bincount(truth * class_count + row, minlength=class_count ** 2)
        .reshape(class_count, class_count)
        for row in hard
    ])
    return counts, error_correlations(hard, truth)


def _parameters(counts, correlations, alpha, gamma):
    likelihoods = (counts + alpha) / (counts.sum(axis=2, keepdims=True) + counts.shape[2] * alpha)
    weights = 1.0 / (1.0 + gamma * correlations.sum(axis=1))
    return likelihoods, weights


def _predict_codes(hard, likelihoods, weights, prior):
    scores = np.broadcast_to(np.log(prior), (hard.shape[1], len(prior))).copy()
    for row, matrix, weight in zip(hard, likelihoods, weights):
        scores += weight * np.log(matrix[:, row]).T
    return scores.argmax(axis=1)


def _tune(hard, truth, groups, row_ids, checkpoint_ids, strategy, class_count):
    combinations = [(alpha, gamma) for alpha in strategy["alpha_grid"]
                    for gamma in strategy["gamma_grid"]]
    predictions = np.full((len(combinations), len(truth)), -1, dtype=int)
    audit = []
    prior = np.full(class_count, 1.0 / class_count)
    for fit, held in GroupKFold(n_splits=strategy["inner_folds"]).split(hard.T, truth, groups):
        roster, _ = select_coverage_roster(hard[:, fit], truth[fit], class_count,
                                          strategy["max_checkpoints"])
        counts, correlations = _fit_statistics(hard[roster][:, fit], truth[fit], class_count)
        for index, (alpha, gamma) in enumerate(combinations):
            likelihoods, weights = _parameters(counts, correlations, alpha, gamma)
            predictions[index, held] = _predict_codes(hard[roster][:, held], likelihoods, weights, prior)
        audit.append({"fit_ids": row_ids[fit].tolist(), "held_ids": row_ids[held].tolist(),
                      "checkpoint_ids": [checkpoint_ids[i] for i in roster]})
    scores = [{"alpha": alpha, "gamma": gamma,
               "accuracy": float(np.mean(predictions[index] == truth)),
               "balanced_accuracy": float(balanced_accuracy_score(truth, predictions[index]))}
              for index, (alpha, gamma) in enumerate(combinations)]
    best = max(scores, key=lambda row: (row["accuracy"], row["balanced_accuracy"],
                                       row["alpha"], row["gamma"]))
    return best, scores, audit


def fit_confusion_ensemble(hard, truth, groups, row_ids, checkpoint_ids, strategy, labels=LABELS):
    """Fit roster, nested hyperparameters and evidence tables on this partition."""
    if not len(hard) or strategy["max_checkpoints"] < 1:
        raise ValueError("At least one eligible checkpoint is required")
    if (not strategy["alpha_grid"] or not strategy["gamma_grid"]
            or any(not np.isfinite(a) or a <= 0 for a in strategy["alpha_grid"])
            or any(not np.isfinite(g) or g < 0 for g in strategy["gamma_grid"])):
        raise ValueError("Smoothing must be positive and correlation penalties nonnegative")
    class_count = len(labels)
    best, scores, audit = _tune(hard, truth, groups, row_ids, checkpoint_ids, strategy, class_count)
    roster, steps = select_coverage_roster(hard, truth, class_count, strategy["max_checkpoints"])
    counts, correlations = _fit_statistics(hard[roster], truth, class_count)
    likelihoods, weights = _parameters(counts, correlations, best["alpha"], best["gamma"])
    coverage = (hard[roster] == truth).any(axis=0)
    return {
        "checkpoint_ids": [checkpoint_ids[i] for i in roster], "labels": list(labels),
        "prior": [1.0 / class_count] * class_count,
        "alpha": best["alpha"], "gamma": best["gamma"],
        "confusion_orientation": "true_label,predicted_label",
        "confusion_matrices": likelihoods.tolist(), "confusion_counts": counts.tolist(),
        "error_correlations": correlations.tolist(), "weights": weights.tolist(),
        "selection_steps": [{"checkpoint_id": checkpoint_ids[step["checkpoint_index"]],
                             "newly_covered": step["newly_covered"]} for step in steps],
        "fit_ids": row_ids.tolist(), "fit_oracle_correct": int(coverage.sum()),
        "fit_oracle_accuracy": float(coverage.mean()),
        "inner_scores": scores, "inner_folds": audit,
    }


def predict_confusion_ensemble(model, probabilities, labels=LABELS):
    """Predict from a saved model and probability matrices in the supplied label order."""
    model_labels = model["labels"]
    if len(set(labels)) != len(model_labels) or set(labels) != set(model_labels):
        raise ValueError("Prediction labels must match the fitted canonical labels")
    order = [list(labels).index(label) for label in model_labels]
    hard = []
    for checkpoint_id in model["checkpoint_ids"]:
        matrix = np.asarray(probabilities[checkpoint_id])
        if (matrix.ndim != 2 or matrix.shape[1] != len(order)
                or not np.isfinite(matrix).all()):
            raise ValueError(f"Invalid probability matrix for {checkpoint_id}")
        hard.append(matrix[:, order].argmax(axis=1))
    predicted = _predict_codes(np.stack(hard), np.asarray(model["confusion_matrices"]),
                               np.asarray(model["weights"]), np.asarray(model["prior"]))
    return np.asarray(model_labels, dtype=object)[predicted]


def evaluate_confusion_ensemble(strategy, folds, v0, probabilities):
    """Run five outer folds and a separate full-V0 deployment fit."""
    checkpoint_ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    if not checkpoint_ids:
        raise ValueError("No eligible cached checkpoints")
    hard = np.stack([probabilities[sid].argmax(axis=1) for sid in checkpoint_ids])
    truth = pd.Categorical(v0.model, categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any():
        raise ValueError("V0 includes unknown labels")
    row_ids = v0.id.to_numpy()
    groups = v0.prompt_id.to_numpy()
    predicted = np.empty(len(v0), dtype=object)
    oracle = np.zeros(len(v0), dtype=bool)
    records = []
    for fit, held in GroupKFold(n_splits=folds).split(v0, truth, groups):
        model = fit_confusion_ensemble(hard[:, fit], truth[fit], groups[fit], row_ids[fit],
                                       checkpoint_ids, strategy)
        predicted[held] = predict_confusion_ensemble(
            model, {sid: probabilities[sid][held] for sid in model["checkpoint_ids"]})
        roster = [checkpoint_ids.index(sid) for sid in model["checkpoint_ids"]]
        oracle[held] = (hard[roster][:, held] == truth[held]).any(axis=0)
        records.append({**model, "held_ids": row_ids[held].tolist(),
                        "oracle_correct": int(oracle[held].sum()),
                        "oracle_accuracy": float(oracle[held].mean())})

    deployment = fit_confusion_ensemble(hard, truth, groups, row_ids, checkpoint_ids, strategy)
    deployment_predicted = predict_confusion_ensemble(deployment, probabilities)
    deployment["fit_accuracy"] = float(np.mean(deployment_predicted == v0.model.to_numpy()))
    deployment["fit_balanced_accuracy"] = float(balanced_accuracy_score(v0.model, deployment_predicted))
    confusion = confusion_matrix(v0.model, predicted, labels=LABELS)
    per_label = {}
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index].sum()) - tp
        per_label[label] = {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                            "support": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0}
    return predicted, {
        "deployment": deployment, "folds": records, "per_label": per_label,
        "oracle_correct": int(oracle.sum()), "oracle_accuracy": float(oracle.mean()),
        "evaluation_checkpoint_ids": sorted(
            {sid for model in records for sid in model["checkpoint_ids"]}, key=lambda sid: int(sid[1:])),
    }
