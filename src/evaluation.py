"""Classification metrics and OOF prediction alignment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, recall_score


def evaluate(y_true, scores, classes, probabilities=None):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    classes = list(classes)
    if scores.shape != (len(y_true), len(classes)):
        raise ValueError("scores shape must be [rows, classes]")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    pred = np.asarray(classes)[scores.argmax(axis=1)]
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=classes, average="macro", zero_division=0)),
        "per_class_recall": {str(c): float(r) for c, r in zip(classes, recall_score(y_true, pred, labels=classes, average=None, zero_division=0))},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=classes).tolist(),
        "classes": classes,
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape != scores.shape or not np.isfinite(probabilities).all():
            raise ValueError("invalid probability matrix")
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=classes))
        result["multiclass_brier"] = float(np.mean(np.sum((probabilities - np.eye(len(classes))[[classes.index(y) for y in y_true]]) ** 2, axis=1)))
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
            result[f"{a}__{b}"] = {
                "disagreement": float(np.mean(pred[a] != pred[b])),
                "error_overlap": int(np.sum(~ca & ~cb)),
                "a_unique_correct": int(np.sum(ca & ~cb)),
                "b_unique_correct": int(np.sum(cb & ~ca)),
                "oracle_accuracy": float(np.mean(ca | cb)),
            }
    return result
