"""Classification metrics and OOF prediction alignment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)


def evaluate(y_true, scores, classes, probabilities=None):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    classes = list(classes)
    if scores.shape != (len(y_true), len(classes)):
        raise ValueError("scores shape must be [rows, classes]")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    pred = np.asarray(classes)[scores.argmax(axis=1)]
    matrix = confusion_matrix(y_true, pred, labels=classes)
    true_positives = np.diag(matrix)
    false_positives = matrix.sum(axis=0) - true_positives
    result = {
        "accuracy": float(np.mean(y_true == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=classes, average="macro", zero_division=0)),
        "per_class_precision": {str(c): float(p) for c, p in zip(
            classes, precision_score(y_true, pred, labels=classes, average=None, zero_division=0))},
        "per_class_recall": {str(c): float(r) for c, r in zip(classes, recall_score(y_true, pred, labels=classes, average=None, zero_division=0))},
        "confusion_matrix": matrix.tolist(),
        "per_class_true_positives": {str(c): int(v) for c, v in zip(classes, true_positives)},
        "per_class_false_positives": {str(c): int(v) for c, v in zip(classes, false_positives)},
        "classes": classes,
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape != scores.shape or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("invalid probability matrix")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("probability rows must sum to one")
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=classes))
        result["multiclass_brier"] = float(np.mean(np.sum((probabilities - np.eye(len(classes))[[classes.index(y) for y in y_true]]) ** 2, axis=1)))
        calibrated_pred = np.asarray(classes)[probabilities.argmax(axis=1)]
        result["calibrated_balanced_accuracy"] = float(balanced_accuracy_score(y_true, calibrated_pred))
        result["calibrated_macro_f1"] = float(f1_score(y_true, calibrated_pred, labels=classes, average="macro", zero_division=0))
        result["calibrated_confusion_matrix"] = confusion_matrix(y_true, calibrated_pred, labels=classes).tolist()
        result["calibrated_per_class_recall"] = {str(c): float(r) for c, r in zip(classes, recall_score(y_true, calibrated_pred, labels=classes, average=None, zero_division=0))}
    return result


def align_predictions(frames, classes):
    """Join model tables by id and enforce complete, unambiguous aligned rows."""
    if not frames:
        raise ValueError("at least one prediction frame is required")
    merged = None
    for name, frame in frames.items():
        if "id" not in frame or frame.id.duplicated().any():
            raise ValueError(f"{name}: id must be present and unique")
        cols = [f"{c}" for c in classes]
        if set(cols) - set(frame.columns):
            raise ValueError(f"{name}: missing class columns")
        current = frame[["id", *cols]].rename(columns={c: f"{name}::{c}" for c in cols})
        merged = current if merged is None else merged.merge(current, on="id", how="inner", validate="one_to_one")
    if any(len(x) != len(merged) for x in frames.values()):
        raise ValueError("prediction files contain different id sets")
    return merged.sort_values("id", kind="stable").reset_index(drop=True)


def ensemble_probabilities(probability_matrices):
    arrays = [np.asarray(p, dtype=float) for p in probability_matrices]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("probability matrices must be non-empty and have matching shapes")
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in arrays):
        raise ValueError("probabilities must be finite and nonnegative")
    mean = np.mean(arrays, axis=0)
    sums = mean.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("probability rows must have positive mass")
    return mean / sums


def complementarity(y_true, predictions):
    """Pairwise errors, disagreement, unique correct counts and oracle accuracy."""
    names = list(predictions)
    pred = {k: np.asarray(v) for k, v in predictions.items()}
    y = np.asarray(y_true)
    result = {}
    for i, a in enumerate(names):
        ca = pred[a] == y
        for b in names[i + 1:]:
            cb = pred[b] == y
            ea, eb = (~ca).astype(float), (~cb).astype(float)
            error_corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else None
            result[f"{a}__{b}"] = {
                "disagreement": float(np.mean(pred[a] != pred[b])),
                "error_correlation": error_corr,
                "error_overlap": int(np.sum(~ca & ~cb)),
                "a_unique_correct": int(np.sum(ca & ~cb)),
                "b_unique_correct": int(np.sum(cb & ~ca)),
                "oracle_accuracy": float(np.mean(ca | cb)),
            }
    return result


def paired_group_bootstrap(y_true, pred_a, pred_b, groups, *, n_bootstrap=2000, seed=2026, confidence=.95):
    """Paired percentile interval for balanced-accuracy delta, resampling groups."""
    y, a, b, g = map(np.asarray, (y_true, pred_a, pred_b, groups))
    if not (len(y) == len(a) == len(b) == len(g)):
        raise ValueError("truth, predictions, and groups must have equal lengths")
    unique = np.unique(g)
    if len(unique) < 2 or n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap requires >=2 groups, positive replicates, and confidence in (0,1)")
    rng = np.random.default_rng(seed)
    by_group = {k: np.flatnonzero(g == k) for k in unique}
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[k] for k in sampled])
        deltas[i] = balanced_accuracy_score(y[idx], b[idx]) - balanced_accuracy_score(y[idx], a[idx])
    alpha = (1 - confidence) / 2
    return {"delta_balanced_accuracy": float(balanced_accuracy_score(y, b) - balanced_accuracy_score(y, a)),
            "confidence": confidence, "lower": float(np.quantile(deltas, alpha)),
            "upper": float(np.quantile(deltas, 1 - alpha)), "bootstrap_replicates": n_bootstrap,
            "resampling_unit": "prompt_id"}
