"""Shared prompt-group and saved E13-B context helpers for E16-E18."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

from . import e15
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
SHARED_CACHE = ROOT / "artifacts/analysis/e16_e17_e18_shared"


def load_context(root=ROOT):
    root = Path(root).resolve()
    v0, y, ids, groups, checkpoint_ids, families, matrices, snapshot, excluded, cache_hashes = e15._load_inputs(root)
    if not {"id", "prompt_id", "text", "model"}.issubset(v0.columns):
        raise ValueError("Full V0 must contain row IDs, prompt IDs, raw text, and labels")
    baseline = pd.read_parquet(e15.BASELINE_PREDICTIONS)
    if baseline.id.astype(str).tolist() != ids.tolist():
        raise ValueError("Saved E13-B row order differs from V0")
    base = e15._label_codes(baseline["E13-B"])
    results = json.loads(e15.BASELINE_RESULTS.read_text())
    baseline_result = next(x for x in results["results"] if x["ensemble"] == "E13-B")
    contexts = []
    for details in baseline_result["folds"]:
        fold_no = int(details["fold"])
        state = e15._predict_saved_e13_outer(
            matrices, checkpoint_ids, families, y, ids, v0, details)
        rows = state["rows"]
        if not np.array_equal(state["base"], base[rows]):
            raise ValueError(f"E13-B saved fold {fold_no} fails exact reproduction")
        if set(groups[details_to_indices(ids, details["fit_ids"])]) & set(groups[rows]):
            raise ValueError(f"E13-B fold {fold_no} shares prompt groups")
        contexts.append({"fold": fold_no, "fit_rows": details_to_indices(ids, details["fit_ids"]),
                         "held_rows": rows, "baseline": state["base"],
                         "alternative": state["alternative"],
                         "label_scores": state["label_scores"],
                         "roster": state["roster"], "details": details})
    contexts.sort(key=lambda x: x["fold"])
    if not np.array_equal(np.sort(np.concatenate([x["held_rows"] for x in contexts])), np.arange(len(v0))):
        raise ValueError("Saved E13-B folds do not cover V0 exactly once")
    return {"root": root, "v0": v0, "y": y, "row_ids": ids,
            "groups": groups, "texts": v0.text.fillna("").astype(str).to_numpy(),
            "checkpoint_ids": checkpoint_ids, "families": families,
            "matrices": matrices, "snapshot": snapshot,
            "cache_hashes": cache_hashes, "excluded": excluded,
            "baseline": base, "outer_folds": contexts}


def details_to_indices(row_ids, values):
    row_map = {str(value): i for i, value in enumerate(row_ids)}
    return np.asarray([row_map[str(value)] for value in values], dtype=np.int64)


def _cache_key(ctx, fit_rows, target_rows):
    h = hashlib.sha256()
    h.update(ctx["snapshot"]["sha256"].encode())
    h.update("\n".join(ctx["row_ids"].astype(str)).encode())
    h.update(np.asarray(fit_rows, dtype=np.int64).tobytes())
    h.update(np.asarray(target_rows, dtype=np.int64).tobytes())
    for sid in ctx["checkpoint_ids"]:
        h.update(ctx["cache_hashes"][sid].encode())
    return h.hexdigest()


def fit_e13_predict(ctx, fit_rows, target_rows, *, cache=True, output_dir=None):
    """Fit E13-B router using fit_rows and return label scores for target_rows."""
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    if np.intersect1d(fit_rows, target_rows).size:
        raise ValueError("E13 fit and target rows overlap")
    if set(ctx["groups"][fit_rows]) & set(ctx["groups"][target_rows]):
        raise ValueError("E13 fit and target prompt groups overlap")
    key = _cache_key(ctx, fit_rows, target_rows)
    path = SHARED_CACHE / f"{key}.joblib" if cache else None
    if path and path.is_file():
        cached = joblib.load(path)
        if cached["key"] != key or not np.array_equal(cached["target_rows"], target_rows):
            raise ValueError("E13 shared cache metadata mismatch")
        return cached
    model_path = Path(output_dir) / f"e13_fit_{key[:16]}.cbm" if output_dir else None
    result = e15._fit_e13_predict(
        ctx["matrices"], ctx["checkpoint_ids"], ctx["families"], ctx["y"],
        ctx["groups"], fit_rows, target_rows, ctx["row_ids"], model_path=model_path)
    payload = {"key": key, "target_rows": target_rows,
               "label_scores": result["label_scores"].astype(np.float32),
               "baseline": result["base"].astype(np.int64),
               "alternative": result["alternative"].astype(np.int64),
               "roster": result["roster_ids"],
               "router_best_iterations": int(result["metrics"]["best_iterations"]),
               "router_training_seconds": float(result["metrics"]["training_seconds"])}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump(payload, tmp, compress=3)
        tmp.replace(path)
    return payload


def oof_e13(ctx, rows, *, n_splits=4, cache=True, output_dir=None, tag="oof"):
    """Return E13-B scores/predictions for rows from group-disjoint router fits."""
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.full((len(rows), len(LABELS)), np.nan, dtype=np.float32)
    pred = np.full(len(rows), -1, dtype=np.int64)
    alt = np.full(len(rows), -1, dtype=np.int64)
    assigned = np.zeros(len(rows), dtype=np.int8)
    pos = {int(row): i for i, row in enumerate(rows)}
    splits = GroupKFold(n_splits=n_splits).split(
        rows, ctx["y"][rows], ctx["groups"][rows])
    for fold_no, (fit_local, held_local) in enumerate(splits, 1):
        fit_rows, held_rows = rows[fit_local], rows[held_local]
        state = fit_e13_predict(ctx, fit_rows, held_rows, cache=cache, output_dir=output_dir)
        p = np.asarray([pos[int(row)] for row in held_rows], dtype=np.int64)
        scores[p] = state["label_scores"]
        pred[p] = state["baseline"]
        alt[p] = state["alternative"]
        assigned[p] += 1
        print(f"{tag}: E13 group fold {fold_no}/{n_splits}, fit={len(fit_rows)} held={len(held_rows)}",
              flush=True)
    if not np.all(assigned == 1) or (pred < 0).any() or (alt < 0).any():
        raise ValueError("Nested E13 OOF did not cover every requested row exactly once")
    return {"rows": rows, "label_scores": scores, "baseline": pred, "alternative": alt}


def finite_scores(scores: np.ndarray) -> np.ndarray:
    """Replace unsupported E13 labels with a low but finite utility."""
    scores = np.asarray(scores, dtype=np.float64)
    out = scores.copy()
    for i in range(len(out)):
        supported = np.isfinite(out[i])
        if not supported.any():
            raise ValueError("E13 row has no supported label")
        out[i, ~supported] = float(out[i, supported].min() - 0.5)
    return out


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {"rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix": cm.tolist(),
            "per_label": {LABELS[i]: {"support": int(support[i]),
                                      "precision": float(precision[i]),
                                      "recall": float(recall[i]), "f1": float(f1[i]),
                                      "tp": int(cm[i, i]), "fp": int(cm[:, i].sum()-cm[i, i]),
                                      "fn": int(cm[i].sum()-cm[i, i])}
                          for i in range(len(LABELS))}}


def comparison(y, baseline, pred):
    y, baseline, pred = map(lambda x: np.asarray(x, dtype=np.int64), (y, baseline, pred))
    corrected = (baseline != y) & (pred == y)
    introduced = (baseline == y) & (pred != y)
    return {"baseline_accuracy": float(accuracy_score(y, baseline)),
            "baseline_balanced_accuracy": float(balanced_accuracy_score(y, baseline)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "delta_accuracy": float(accuracy_score(y, pred)-accuracy_score(y, baseline)),
            "delta_balanced_accuracy": float(balanced_accuracy_score(y, pred)-balanced_accuracy_score(y, baseline)),
            "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
            "net_gains": int(corrected.sum()-introduced.sum()),
            "switches": int(np.count_nonzero(pred != baseline))}
