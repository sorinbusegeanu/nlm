"""E15: leakage-controlled logistic corrections to E13-B predictions.

Only cached Trust List 2 v037 probabilities are used.  The E13 reliability
router is fitted inside each E15 fitting partition; its scores for E15
training rows are themselves prompt-group out-of-fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import ensemble_catalog as catalog
from . import e13_reliability_router as e13
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
E13_RUN = ROOT / "artifacts/analysis/e13/20261120T093318893046Z"
BASELINE_PREDICTIONS = E13_RUN / "predictions.parquet"
BASELINE_RESULTS = E13_RUN / "results.json"
BASELINE_CONFIG = E13_RUN / "config.json"
OUTER_FOLDS = 5
TUNING_FOLDS = 3
FEATURE_OOF_FOLDS = 4
THREADS = 24
SEED = 2026
CS = (0.01, 0.1, 1.0)
MAX_ITER = 2000
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def _label_codes(values) -> np.ndarray:
    codes = pd.Categorical(pd.Series(values).astype(str), categories=LABELS).codes
    if (codes < 0).any():
        raise ValueError("V0 contains labels outside the canonical 18-label order")
    return codes.astype(np.int64)


def _pair_scores(hard: np.ndarray, reliability: np.ndarray,
                 ensemble: np.ndarray, families: list[str]) -> np.ndarray:
    """E13-B family-max scores, in canonical label order."""
    hard = np.asarray(hard, dtype=np.int64)
    reliability = np.asarray(reliability, dtype=np.float64)
    if hard.ndim != 2 or reliability.shape != hard.shape:
        raise ValueError("E13 hard predictions and reliability scores do not align")
    if len(families) != hard.shape[0]:
        raise ValueError("E13 family list does not align with the roster")
    family_names = sorted(set(families))
    scores = np.full((hard.shape[1], len(LABELS)), -np.inf, dtype=np.float64)
    for row in range(hard.shape[1]):
        for label in np.unique(hard[:, row]):
            vals = []
            for family in family_names:
                voters = np.flatnonzero((hard[:, row] == label) &
                                        (np.asarray(families, dtype=object) == family))
                if len(voters):
                    vals.append(float(reliability[voters, row].max()))
            scores[row, label] = float(np.mean(vals))
    return scores


def select_candidates(hard: np.ndarray, reliability: np.ndarray,
                      ensemble: np.ndarray, families: list[str]):
    """Return E13-B's selected label and its strongest supported alternative.

    Ties use descending family-balanced ensemble probability, then canonical
    label order.  An unsupported label is never selected as the alternative.
    """
    hard = np.asarray(hard, dtype=np.int64)
    reliability = np.asarray(reliability, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    if hard.shape != reliability.shape or hard.ndim != 2:
        raise ValueError("Candidate hard/reliability arrays must be [checkpoints, rows]")
    if ensemble.shape != (hard.shape[1], len(LABELS)) or len(families) != hard.shape[0]:
        raise ValueError("Candidate family/probability metadata do not align")
    scores = _pair_scores(hard, reliability, ensemble, families)
    pred = np.empty(hard.shape[1], dtype=np.int64)
    alternative = np.empty_like(pred)
    for row in range(hard.shape[1]):
        supported = np.flatnonzero(np.isfinite(scores[row]))
        if not len(supported):
            raise ValueError("E13-B has no supported label for a row")
        best = np.max(scores[row, supported])
        tied = supported[np.isclose(scores[row, supported], best, rtol=0, atol=1e-12)]
        pred[row] = min(tied.tolist(), key=lambda lab: (-ensemble[row, lab], lab))
        remaining_supported = supported[supported != pred[row]]
        if not len(remaining_supported):
            # With a single supported candidate, choose the highest-probability
            # different label deterministically; its family score stays absent.
            remaining = [lab for lab in range(len(LABELS)) if lab != pred[row]]
            alternative[row] = min(remaining, key=lambda lab: (-ensemble[row, lab], lab))
        else:
            alt_best = np.max(scores[row, remaining_supported])
            alt_tied = remaining_supported[np.isclose(
                scores[row, remaining_supported], alt_best, rtol=0, atol=1e-12)]
            alternative[row] = min(alt_tied.tolist(), key=lambda lab: (-ensemble[row, lab], lab))
    return pred, alternative, scores


def _feature_names(checkpoint_ids: list[str]) -> list[str]:
    names = ([f"selected::{label}" for label in LABELS] +
             [f"alternative::{label}" for label in LABELS] +
             [f"family_probability::{label}" for label in LABELS] +
             ["family_entropy", "family_top_margin", "family_top_probability",
              "e13_selected_reliability", "e13_alternative_reliability",
              "e13_reliability_margin", "family_probability_selected",
              "family_probability_alternative", "ensemble_probability_margin",
              "supporting_checkpoint_count_selected", "supporting_checkpoint_count_alternative",
              "supporting_family_count_selected", "supporting_family_count_alternative",
              "checkpoint_disagreement_rate", "family_disagreement_rate"])
    for sid in checkpoint_ids:
        names.append(f"{sid}::present")
        names.extend(f"{sid}::prob::{label}" for label in LABELS)
        names.extend((f"{sid}::p_selected", f"{sid}::p_alternative",
                      f"{sid}::p_alternative_minus_selected", f"{sid}::reliability",
                      f"{sid}::votes_selected", f"{sid}::votes_alternative"))
    if len(names) != len(set(names)):
        raise ValueError("E15 feature names are not unique")
    return names


def build_features(probabilities: np.ndarray, checkpoint_ids: list[str],
                   families: list[str], roster: np.ndarray,
                   hard: np.ndarray, reliability: np.ndarray,
                   ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build fixed-layout E15 features for target rows in the supplied order."""
    p = np.asarray(probabilities, dtype=np.float64)
    roster = np.asarray(roster, dtype=np.int64)
    hard = np.asarray(hard, dtype=np.int64)
    reliability = np.asarray(reliability, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != len(checkpoint_ids) or p.shape[2] != len(LABELS):
        raise ValueError("E15 probabilities must align as [checkpoints, rows, 18 labels]")
    if len(families) != len(checkpoint_ids) or len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("E15 checkpoint identity layout is invalid")
    if hard.shape != reliability.shape or hard.shape != (len(roster), p.shape[1]):
        raise ValueError("E13 scores and selected roster do not align")
    if ensemble.shape != (p.shape[1], len(LABELS)) or not np.isfinite(p).all():
        raise ValueError("E15 input probabilities are invalid")
    if not np.allclose(p.sum(axis=2), 1, atol=1e-4):
        raise ValueError("E15 checkpoint probability vectors are not normalized")
    roster_ids = [checkpoint_ids[i] for i in roster]
    roster_families = [families[i] for i in roster]
    selected, alternative, agg = select_candidates(hard, reliability, ensemble, roster_families)
    n = p.shape[1]
    row = np.arange(n)
    top_order = np.argsort(-ensemble, axis=1, kind="stable")
    sorted_ens = np.sort(ensemble, axis=1)
    entropy = -(np.clip(ensemble, 1e-12, 1) * np.log(np.clip(ensemble, 1e-12, 1))).sum(1)
    support_selected = hard == selected[None, :]
    support_alt = hard == alternative[None, :]
    unique_families = sorted(set(roster_families))
    fam_counts_selected = np.zeros(n, dtype=np.float64)
    fam_counts_alt = np.zeros(n, dtype=np.float64)
    for fam in unique_families:
        ix = np.asarray(roster_families, dtype=object) == fam
        fam_counts_selected += np.any(support_selected[ix], axis=0)
        fam_counts_alt += np.any(support_alt[ix], axis=0)
    selected_rel = agg[row, selected]
    alt_rel = agg[row, alternative]
    # Alternatives selected from hard-supported labels always have finite
    # scores. A one-candidate fallback has no learned reliability evidence.
    alt_rel = np.where(np.isfinite(alt_rel), alt_rel, 0.0)
    selected_rel = np.where(np.isfinite(selected_rel), selected_rel, 0.0)
    parts = [np.eye(len(LABELS), dtype=np.float32)[selected],
             np.eye(len(LABELS), dtype=np.float32)[alternative],
             ensemble.astype(np.float32),
             np.column_stack([
                 entropy / np.log(len(LABELS)), sorted_ens[:, -1] - sorted_ens[:, -2],
                 sorted_ens[:, -1], selected_rel, alt_rel, selected_rel - alt_rel,
                 ensemble[row, selected], ensemble[row, alternative],
                 ensemble[row, alternative] - ensemble[row, selected],
                 support_selected.sum(axis=0), support_alt.sum(axis=0),
                 fam_counts_selected, fam_counts_alt,
                 (hard != hard[0:1]).mean(axis=0),
                 # Fraction of distinct families whose modal label differs from
                 # the row's plurality family label.
                 np.zeros(n, dtype=np.float64),
             ]).astype(np.float32)]
    # Fill family disagreement with a per-row family-vote statistic.
    family_disagreement = np.zeros(n, dtype=np.float64)
    family_vote_mats = []
    for fam in unique_families:
        ix = np.flatnonzero(np.asarray(roster_families, dtype=object) == fam)
        family_pred = np.empty(n, dtype=np.int64)
        for j in range(n):
            family_pred[j] = np.bincount(hard[ix, j], minlength=len(LABELS)).argmax()
        family_vote_mats.append(family_pred)
    family_vote_mats = np.stack(family_vote_mats)
    for j in range(n):
        freq = np.bincount(family_vote_mats[:, j], minlength=len(LABELS))
        family_disagreement[j] = 1.0 - freq.max() / max(1, len(unique_families))
    parts[-1][:, -1] = family_disagreement

    # The explicit global checkpoint-ID layout makes fold-specific rosters
    # directly comparable: absent checkpoints have a 0 presence flag and 0
    # in every corresponding value slot.
    for cp_index, sid in enumerate(checkpoint_ids):
        cp_part = np.zeros((n, 1 + len(LABELS) + 6), dtype=np.float32)
        if cp_index in set(roster.tolist()):
            local = int(np.flatnonzero(roster == cp_index)[0])
            q = p[cp_index]
            cp_part[:, 0] = 1.0
            cp_part[:, 1:1 + len(LABELS)] = q.astype(np.float32)
            cp_part[:, 1 + len(LABELS)] = q[row, selected]
            cp_part[:, 2 + len(LABELS)] = q[row, alternative]
            cp_part[:, 3 + len(LABELS)] = q[row, alternative] - q[row, selected]
            cp_part[:, 4 + len(LABELS)] = reliability[local]
            cp_part[:, 5 + len(LABELS)] = (hard[local] == selected)
            cp_part[:, 6 + len(LABELS)] = (hard[local] == alternative)
        parts.append(cp_part)
    x = np.column_stack(parts).astype(np.float32)
    names = _feature_names(checkpoint_ids)
    if x.shape != (n, len(names)) or not np.isfinite(x).all():
        raise ValueError("E15 feature matrix does not match its fixed feature layout")
    return x, selected, alternative, names


def _make_model(c: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    # sklearn's default penalty is L2; omitting the deprecated explicit
    # argument avoids a warning without changing the requested regularizer.
    model = LogisticRegression(C=float(c), solver="lbfgs",
                               max_iter=MAX_ITER, random_state=SEED)
    return scaler, model


def _fit_logistic(x: np.ndarray, y: np.ndarray, c: float):
    if len(np.unique(y)) < 2:
        raise ValueError("E15 fitting rows must contain both correction outcomes")
    scaler, model = _make_model(c)
    model.fit(scaler.fit_transform(x), y)
    return scaler, model


def _probability(scaler, model, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(scaler.transform(x))[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("E15 logistic output contains a non-finite probability")
    return p


def _apply_threshold(base: np.ndarray, alternative: np.ndarray,
                     probability: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(probability >= threshold, alternative, base).astype(np.int64)


def _choose_threshold(truth, base, alternative, probability) -> tuple[float, dict]:
    truth = np.asarray(truth, dtype=np.int64)
    base = np.asarray(base, dtype=np.int64)
    alternative = np.asarray(alternative, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    thresholds = np.unique(np.r_[0.0, probability, 1.0 + 1e-12])
    best = None
    for threshold in thresholds:
        pred = _apply_threshold(base, alternative, probability, float(threshold))
        ba = float(balanced_accuracy_score(truth, pred))
        acc = float(accuracy_score(truth, pred))
        switched = int(np.count_nonzero(pred != base))
        key = (ba, acc, -switched, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), pred)
    return best[1], {"balanced_accuracy": best[0][0], "accuracy": best[0][1],
                     "corrections": best[0][2] * -1}


def _fit_e13_predict(matrices, ids, families, truth, groups, fit_rows,
                     target_rows, row_ids, *, model_path=None):
    """Fit the existing E13-B router on fit_rows and score target_rows."""
    zeros = np.zeros(len(truth), dtype=np.int64)
    pred, metrics, model, aux = e13._fit_router(
        matrices, ids, families, truth, groups, np.asarray(fit_rows, dtype=np.int64),
        np.asarray(target_rows, dtype=np.int64), 15, "family_max", zeros, row_ids,
        model_path=model_path)
    reliability, hard, roster = aux
    selected_ids = [ids[i] for i in roster]
    selected_families = [families[i] for i in roster]
    ensemble = e13.family_balanced_probabilities(
        matrices[roster][:, target_rows], selected_families)
    if not np.array_equal(pred, e13.route_predictions(
            "family_max", hard, reliability, ensemble, selected_families)[0]):
        raise ValueError("E13 prediction could not be reconstructed consistently")
    x, chosen, alternative, names = build_features(
        matrices[:, target_rows], ids, families, roster, hard, reliability, ensemble)
    return {"x": x, "base": chosen, "alternative": alternative,
            "roster": roster, "roster_ids": selected_ids,
            "feature_names": names, "metrics": metrics,
            "reliability": reliability, "hard": hard, "ensemble": ensemble,
            "model": model}


def _oof_e13_features(rows, matrices, ids, families, truth, groups, row_ids,
                      *, n_splits=FEATURE_OOF_FOLDS, stage="fit"):
    rows = np.asarray(rows, dtype=np.int64)
    x_out = base_out = alt_out = None
    assigned = np.zeros(len(rows), dtype=np.int8)
    for fold, (local_fit, local_held) in enumerate(GroupKFold(n_splits=n_splits).split(
            rows, truth[rows], groups[rows]), 1):
        fit_rows, held_rows = rows[local_fit], rows[local_held]
        if set(groups[fit_rows]) & set(groups[held_rows]):
            raise ValueError("E15 feature cross-fit split shares prompt groups")
        result = _fit_e13_predict(matrices, ids, families, truth, groups,
                                  fit_rows, held_rows, row_ids)
        if x_out is None:
            x_out = np.zeros((len(rows), result["x"].shape[1]), dtype=np.float32)
            base_out = np.full(len(rows), -1, dtype=np.int64)
            alt_out = np.full(len(rows), -1, dtype=np.int64)
        local_positions = {int(row): i for i, row in enumerate(rows)}
        pos = np.asarray([local_positions[int(row)] for row in held_rows])
        x_out[pos] = result["x"]
        base_out[pos] = result["base"]
        alt_out[pos] = result["alternative"]
        assigned[pos] += 1
        print(f"  {stage} E13 OOF {fold}/{n_splits}: fit={len(fit_rows)} held={len(held_rows)}",
              flush=True)
    if x_out is None or not np.all(assigned == 1) or (base_out < 0).any() or (alt_out < 0).any():
        raise ValueError("E15 E13 OOF records do not cover each row exactly once")
    return x_out, base_out, alt_out


def _tune_outer_fold(outer_rows, matrices, ids, families, truth, groups, row_ids,
                     *, outer_no, output):
    """Nested group-disjoint selection of C and correction threshold."""
    all_prob = {c: np.full(len(outer_rows), np.nan, dtype=np.float64) for c in CS}
    all_base = np.full(len(outer_rows), -1, dtype=np.int64)
    all_alt = np.full(len(outer_rows), -1, dtype=np.int64)
    assigned = np.zeros(len(outer_rows), dtype=np.int8)
    posmap = {int(row): i for i, row in enumerate(outer_rows)}
    folds_log = []
    for split_no, (local_fit, local_valid) in enumerate(GroupKFold(n_splits=TUNING_FOLDS).split(
            outer_rows, truth[outer_rows], groups[outer_rows]), 1):
        fit_rows, valid_rows = outer_rows[local_fit], outer_rows[local_valid]
        if set(groups[fit_rows]) & set(groups[valid_rows]):
            raise ValueError("E15 parameter-selection split shares prompt groups")
        print(f" E15 outer {outer_no}: tune split {split_no}/{TUNING_FOLDS} "
              f"(train={len(fit_rows)}, validation={len(valid_rows)})", flush=True)
        x_fit, base_fit, alt_fit = _oof_e13_features(
            fit_rows, matrices, ids, families, truth, groups, row_ids,
            n_splits=FEATURE_OOF_FOLDS,
            stage=f"outer{outer_no}-tune{split_no}-train")
        y_fit = (alt_fit == truth[fit_rows]).astype(np.int8)
        if not np.any(y_fit == 0) or not np.any(y_fit == 1):
            raise ValueError("Nested E15 fitting rows do not contain both target classes")
        valid = _fit_e13_predict(matrices, ids, families, truth, groups,
                                 fit_rows, valid_rows, row_ids)
        all_base[np.asarray([posmap[int(r)] for r in valid_rows])] = valid["base"]
        all_alt[np.asarray([posmap[int(r)] for r in valid_rows])] = valid["alternative"]
        valid_pos = np.asarray([posmap[int(r)] for r in valid_rows])
        assigned[valid_pos] += 1
        fold_record = {"fold": split_no, "fit_ids": row_ids[fit_rows].tolist(),
                       "held_ids": row_ids[valid_rows].tolist(),
                       "train_correction_positive": int(y_fit.sum()),
                       "train_rows": int(len(y_fit)), "validation_rows": int(len(valid_rows))}
        for c in CS:
            scaler, model = _fit_logistic(x_fit, y_fit, c)
            p = _probability(scaler, model, valid["x"])
            all_prob[c][valid_pos] = p
            fold_record[f"C_{c:g}_model_iterations"] = int(np.max(model.n_iter_))
        folds_log.append(fold_record)
    if not np.all(assigned == 1) or any(not np.isfinite(v).all() for v in all_prob.values()):
        raise ValueError("E15 tuning folds failed to produce exactly one prediction per row")
    tune_results = []
    for c in CS:
        threshold, train_score = _choose_threshold(
            truth[outer_rows], all_base, all_alt, all_prob[c])
        pred = _apply_threshold(all_base, all_alt, all_prob[c], threshold)
        result = {"C": c, "threshold": threshold,
                  "balanced_accuracy": float(balanced_accuracy_score(truth[outer_rows], pred)),
                  "accuracy": float(accuracy_score(truth[outer_rows], pred)),
                  "threshold_fit_score": train_score,
                  "correction_probability_oof": all_prob[c].tolist()}
        tune_results.append(result)
    # BA is primary; use accuracy, fewer changes, then smaller C for stable ties.
    selected = max(tune_results, key=lambda d: (
        d["balanced_accuracy"], d["accuracy"],
        -int(np.count_nonzero(_apply_threshold(all_base, all_alt,
                  np.asarray(d["correction_probability_oof"]), d["threshold"]) != all_base)),
        -d["C"]))
    _write_json(output / f"outer{outer_no}_tuning.json", {
        "outer_fold": outer_no, "folds": folds_log,
        "oof_rows": len(outer_rows), "configurations": tune_results,
        "selected_C": selected["C"], "selected_threshold": selected["threshold"],
    })
    return selected


def _predict_saved_e13_outer(matrices, ids, families, truth, row_ids,
                             v0, fold_details):
    """Recreate E13-B on the documented outer fold with its saved CatBoost model."""
    row_map = {str(value): i for i, value in enumerate(row_ids)}
    fit_rows = np.asarray([row_map[x] for x in fold_details["fit_ids"]], dtype=np.int64)
    held_rows = np.asarray([row_map[x] for x in fold_details["held_ids"]], dtype=np.int64)
    roster_ids = list(fold_details["checkpoint_ids"])
    id_map = {sid: i for i, sid in enumerate(ids)}
    roster = np.asarray([id_map[sid] for sid in roster_ids], dtype=np.int64)
    roster_families = [families[i] for i in roster]
    x, _ = e13.build_feature_frame(
        matrices[roster], roster_ids, roster_families,
        truth[fit_rows], fit_rows, held_rows)
    # E13's saved model predicts per-checkpoint correctness, trained on exactly
    # the same outer-fit groups and serialized when its fold completed.
    model_path = Path(fold_details["model_path"])
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    reliability = e13._score_correctness(model, x).reshape(len(held_rows), len(roster)).T
    hard = matrices[roster][:, held_rows].argmax(axis=2)
    ensemble = e13.family_balanced_probabilities(matrices[roster][:, held_rows], roster_families)
    pred, _ = e13.route_predictions("family_max", hard, reliability, ensemble, roster_families)
    x15, selected, alternative, names = build_features(
        matrices[:, held_rows], ids, families, roster, hard, reliability, ensemble)
    if not np.array_equal(pred, selected):
        raise ValueError("Reconstructed E13-B held predictions disagree with saved model route")
    return {"rows": held_rows, "x": x15, "base": selected,
            "alternative": alternative, "roster": roster_ids,
            "reliability": reliability, "hard": hard, "ensemble": ensemble,
            "feature_names": names}


def _metric_pack(truth, pred, base, alt, correction_prob, threshold):
    truth = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    base = np.asarray(base, dtype=np.int64)
    alt = np.asarray(alt, dtype=np.int64)
    cm = confusion_matrix(truth, pred, labels=np.arange(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(LABELS)), zero_division=0)
    corrected = (base != truth) & (pred == truth)
    introduced = (base == truth) & (pred != truth)
    changed = pred != base
    adjusted = np.zeros((len(truth), len(LABELS)), dtype=np.float64)
    adjusted[np.arange(len(truth)), base] = 1.0 - np.asarray(correction_prob)
    adjusted[np.arange(len(truth)), alt] += np.asarray(correction_prob)
    adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-15)
    logloss = float(log_loss(truth, adjusted, labels=np.arange(len(LABELS))))
    return {
        "rows": len(truth), "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "e13_b_accuracy": float(accuracy_score(truth, base)),
        "e13_b_balanced_accuracy": float(balanced_accuracy_score(truth, base)),
        "delta_balanced_accuracy": float(balanced_accuracy_score(truth, pred) -
                                          balanced_accuracy_score(truth, base)),
        "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
        "wrong_to_wrong_switches": int((changed & (base != truth) & (pred != truth)).sum()),
        "net_correct": int(corrected.sum() - introduced.sum()),
        "corrections_made": int(changed.sum()),
        "correction_precision": float(corrected.sum() / max(1, corrected.sum() + introduced.sum())),
        "log_loss_adjusted_top2": logloss,
        "threshold": float(threshold),
        "correction_probability_mean": float(np.mean(correction_prob)),
        "confusion_matrix": cm.tolist(),
        "per_label": {LABELS[i]: {
            "support": int(support[i]), "precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]),
            "tp": int(cm[i, i]), "fp": int(cm[:, i].sum() - cm[i, i]),
            "fn": int(cm[i].sum() - cm[i, i]),
        } for i in range(len(LABELS))},
    }


def _load_inputs(root: Path):
    v0 = catalog.cascades._v0(root)
    truth = _label_codes(v0.model)
    row_ids = v0.id.astype(str).to_numpy()
    groups = v0.prompt_id.astype(str).to_numpy()
    if len(v0) != 2699 or len(set(row_ids)) != len(v0):
        raise ValueError("E15 requires the complete, ID-unique 2,699-row V0 set")
    config = json.loads(BASELINE_CONFIG.read_text())
    if config["trust_list"]["version"] != "trust-list-2-v037":
        raise ValueError("The saved E13-B baseline is not Trust List 2 v037")
    current = json.loads((root / "docs/trust-list-2-latest.json").read_text())
    if current["version"] != "trust-list-2-v037":
        raise ValueError("Current Trust List pointer differs from required E13-B v037 snapshot")
    trust, members, probability_map, excluded, snapshot = catalog._load_inputs(root, v0)
    ids = list(config["ids"])
    missing = sorted(set(ids) - set(probability_map))
    if missing:
        raise ValueError(f"E13 v037 checkpoints have missing cached probabilities: {missing}")
    if snapshot["version"] != config["trust_list"]["version"] or snapshot["sha256"] != config["trust_list"]["sha256"]:
        raise ValueError("E15 input trust-list snapshot does not match saved E13-B")
    ids = sorted(ids, key=lambda sid: int(sid.lstrip("N")))
    members = {sid: members[sid] for sid in ids}
    matrices = np.stack([probability_map[sid] for sid in ids]).astype(np.float32)
    families = [e13._model_family(members[sid]["model"]) for sid in ids]
    if matrices.shape != (len(ids), len(v0), len(LABELS)):
        raise ValueError("Cached checkpoint probability tensor has the wrong alignment")
    if not np.isfinite(matrices).all() or not np.allclose(matrices.sum(2), 1, atol=1e-4):
        raise ValueError("Cached checkpoint probabilities are invalid")
    cache_hashes = {sid: members[sid]["cache_sha256"] for sid in ids}
    return v0, truth, row_ids, groups, ids, families, matrices, snapshot, excluded, cache_hashes


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output) if output else root / "artifacts/analysis/e15"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"E15 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
    e13.CATBOOST_PARAMS["thread_count"] = THREADS
    started = time.monotonic()
    v0, truth, row_ids, groups, ids, families, matrices, snapshot, excluded, cache_hashes = _load_inputs(root)
    baseline_pred_frame = pd.read_parquet(BASELINE_PREDICTIONS)
    if baseline_pred_frame.id.astype(str).tolist() != row_ids.tolist() or \
            baseline_pred_frame.y_true.astype(str).tolist() != v0.model.astype(str).tolist():
        raise ValueError("Saved E13-B rows or labels are not aligned with full V0")
    baseline_stored = _label_codes(baseline_pred_frame["E13-B"])
    results = json.loads(BASELINE_RESULTS.read_text())
    baseline_result = next(x for x in results["results"] if x["ensemble"] == "E13-B")
    if not np.isclose(balanced_accuracy_score(truth, baseline_stored), 0.6464653243847874,
                      atol=1e-12):
        raise ValueError("Saved E13-B baseline does not reproduce the required 64.6465% BA")
    names = _feature_names(ids)
    pred = np.full(len(v0), -1, dtype=np.int64)
    alt_pred = np.full(len(v0), -1, dtype=np.int64)
    correction_prob = np.full(len(v0), np.nan, dtype=np.float64)
    selected_c = []
    selected_thresholds = []
    c_by_row = np.full(len(v0), np.nan, dtype=np.float64)
    threshold_by_row = np.full(len(v0), np.nan, dtype=np.float64)
    fold_metrics = []
    reproduced_folds = []
    coefficient_rows = []
    row_map = {str(value): i for i, value in enumerate(row_ids)}
    outer_folds = baseline_result["folds"]
    if len(outer_folds) != OUTER_FOLDS:
        raise ValueError("Saved E13-B does not contain five outer prompt-group folds")

    with threadpool_limits(limits=THREADS):
        for fold_details in outer_folds:
            fold_no = int(fold_details["fold"])
            held = np.asarray([row_map[x] for x in fold_details["held_ids"]], dtype=np.int64)
            fit = np.asarray([row_map[x] for x in fold_details["fit_ids"]], dtype=np.int64)
            if set(groups[fit]) & set(groups[held]):
                raise ValueError(f"E15 outer fold {fold_no} leaks prompt groups")
            print(f"E15 outer fold {fold_no}/{OUTER_FOLDS}: nested tuning begins", flush=True)
            tuned = _tune_outer_fold(fit, matrices, ids, families, truth, groups,
                                     row_ids, outer_no=fold_no, output=output)
            x_fit, base_fit, alt_fit = _oof_e13_features(
                fit, matrices, ids, families, truth, groups, row_ids,
                n_splits=FEATURE_OOF_FOLDS, stage=f"outer{fold_no}-final-train")
            y_fit = (alt_fit == truth[fit]).astype(np.int8)
            if len(np.unique(y_fit)) != 2:
                raise ValueError(f"E15 outer fold {fold_no} has a single correction target class")
            scaler, model = _fit_logistic(x_fit, y_fit, tuned["C"])
            # The E13 held-out state is the original E13-B model trained on the
            # identical outer fitting rows. Reuse it to reproduce the baseline.
            held_e13 = _predict_saved_e13_outer(
                matrices, ids, families, truth, row_ids, v0, fold_details)
            if not np.array_equal(held_e13["base"], baseline_stored[held]):
                n_bad = int(np.count_nonzero(held_e13["base"] != baseline_stored[held]))
                raise ValueError(f"Reconstructed E13-B differs on {n_bad} outer-fold rows")
            reproduced_folds.append(fold_no)
            if not np.array_equal(held_e13["rows"], held):
                raise ValueError("Saved E13-B outer fold row order changed")
            p_corr = _probability(scaler, model, held_e13["x"])
            fold_pred = _apply_threshold(held_e13["base"], held_e13["alternative"],
                                         p_corr, tuned["threshold"])
            pred[held] = fold_pred
            alt_pred[held] = held_e13["alternative"]
            correction_prob[held] = p_corr
            c_by_row[held] = float(tuned["C"])
            threshold_by_row[held] = float(tuned["threshold"])
            selected_c.append(float(tuned["C"]))
            selected_thresholds.append(float(tuned["threshold"]))
            fm = _metric_pack(truth[held], fold_pred, held_e13["base"],
                              held_e13["alternative"], p_corr, tuned["threshold"])
            fm.update({"fold": fold_no, "held_row_count": len(held), "fit_row_count": len(fit),
                       "selected_C": tuned["C"], "selected_threshold": tuned["threshold"],
                       "tuning_balanced_accuracy": tuned["balanced_accuracy"],
                       "tuning_accuracy": tuned["accuracy"],
                       "training_rows": int(len(y_fit)),
                       "training_correction_positives": int(y_fit.sum()),
                       "e13_roster": held_e13["roster"],
                       "model_iterations": int(np.max(model.n_iter_))})
            fold_metrics.append(fm)
            joblib.dump({"scaler": scaler, "model": model, "C": tuned["C"],
                         "threshold": tuned["threshold"], "feature_names": names,
                         "checkpoint_ids": ids, "roster": held_e13["roster"],
                         "labels": list(LABELS), "seed": SEED},
                        output / f"E15_fold{fold_no}.joblib", compress=3)
            model_outer_path = output / f"E13_B_fold{fold_no}.cbm"
            CatBoostClassifier().load_model(fold_details["model_path"]).save_model(str(model_outer_path))
            coef = model.coef_[0]
            coefficient_rows.append(coef)
            print(f"E15 outer fold {fold_no}: C={tuned['C']:g}, threshold={tuned['threshold']:.6f}, "
                  f"BA={fm['balanced_accuracy']:.4%}, corrected/introduced="
                  f"{fm['corrected']}/{fm['introduced']}", flush=True)

    if (pred < 0).any() or not np.isfinite(correction_prob).all():
        raise ValueError("E15 outer folds did not cover every V0 row")
    baseline_exact = reproduced_folds == list(range(1, OUTER_FOLDS + 1))
    overall = _metric_pack(truth, pred, baseline_stored, alt_pred,
                           correction_prob, float(np.median(selected_thresholds)))
    # Correction precision is based on changes relative to E13-B; expected
    # alternative correctness is also retained for the explicit task target.
    changed = pred != baseline_stored
    corrected = changed & (pred == truth)
    introduced = (baseline_stored == truth) & (pred != truth)
    wrong_to_wrong = changed & (baseline_stored != truth) & (pred != truth)
    overall["corrected"] = int(corrected.sum())
    overall["introduced"] = int(introduced.sum())
    overall["wrong_to_wrong_switches"] = int(wrong_to_wrong.sum())
    overall["net_correct"] = int(corrected.sum() - introduced.sum())
    overall["corrections_made"] = int(changed.sum())
    overall["correction_precision"] = float(
        corrected.sum() / max(1, corrected.sum() + introduced.sum()))
    overall["delta_accuracy"] = float(overall["accuracy"] - accuracy_score(truth, baseline_stored))
    overall["delta_balanced_accuracy"] = float(
        overall["balanced_accuracy"] - balanced_accuracy_score(truth, baseline_stored))
    overall["e13_baseline_reproduced_exactly"] = baseline_exact
    overall["mean_threshold"] = float(np.mean(selected_thresholds))
    overall["median_threshold"] = float(np.median(selected_thresholds))
    overall["selected_C_by_fold"] = selected_c
    overall["selected_threshold_by_fold"] = selected_thresholds
    overall["folds"] = fold_metrics
    overall["training_runtime_seconds"] = float(time.monotonic() - started)
    overall["peak_process_memory_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    predictions = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    predictions["fold"] = -1
    for f in outer_folds:
        idx = np.asarray([row_map[x] for x in f["held_ids"]], dtype=np.int64)
        predictions.loc[idx, "fold"] = int(f["fold"])
    predictions["E13_B"] = LABELS_ARRAY[baseline_stored]
    predictions["alternative"] = LABELS_ARRAY[alt_pred]
    predictions["alternative_correct"] = alt_pred == truth
    predictions["correction_probability"] = correction_prob
    predictions["selected_C"] = c_by_row
    predictions["threshold"] = threshold_by_row
    predictions["corrected"] = pred != baseline_stored
    predictions["E15"] = LABELS_ARRAY[pred]
    predictions["y_true_code"] = truth
    row_decisions = predictions.loc[predictions.corrected,
        ["id", "prompt_id", "y_true", "E13_B", "alternative", "alternative_correct",
         "correction_probability", "fold", "E15"]].copy()
    row_decisions["outcome"] = np.where(
        row_decisions.E15 == row_decisions.y_true, "corrected",
        np.where(row_decisions.E13_B == row_decisions.y_true,
                 "introduced", "wrong_to_wrong"))
    predictions.to_parquet(output / "predictions.parquet", index=False)
    row_decisions.to_csv(output / "corrections.csv", index=False)
    np.savez_compressed(output / "predictions.npz", y_true=truth, e13_b=baseline_stored,
                        alternative=alt_pred, correction_probability=correction_prob,
                        e15=pred, row_ids=row_ids)
    np.save(output / "coefficient_mean.npy", np.mean(coefficient_rows, axis=0))
    coefficient_mean = np.mean(coefficient_rows, axis=0)
    importance = sorted(zip(names, coefficient_mean.tolist()), key=lambda z: (abs(z[1]), z[0]), reverse=True)
    overall["top_positive_coefficients"] = [
        {"feature": name, "coefficient": coef} for name, coef in importance if coef > 0][:50]
    overall["top_negative_coefficients"] = [
        {"feature": name, "coefficient": coef} for name, coef in importance if coef < 0][:50]
    snapshot_file = json.loads(BASELINE_CONFIG.read_text())
    configuration = {
        "experiment": "E15", "objective": "binary logistic correction to E13-B selected label",
        "baseline": "E13-B Trust List 2 v037", "baseline_oof_accuracy": float(accuracy_score(truth, baseline_stored)),
        "baseline_oof_balanced_accuracy": float(balanced_accuracy_score(truth, baseline_stored)),
        "baseline_reproduced_folds": reproduced_folds,
        "baseline_artifact": str(E13_RUN), "baseline_prediction_sha256": _sha(BASELINE_PREDICTIONS),
        "trust_list_snapshot": snapshot, "trust_list_sha256": snapshot_file["trust_list"]["sha256"],
        "checkpoint_ids": ids, "checkpoint_families": dict(zip(ids, families)),
        "cached_prediction_sha256": cache_hashes,
        "labels": list(LABELS), "v0_row_count": len(v0), "v0_row_ids_sha256": hashlib.sha256(
            "\n".join(row_ids).encode()).hexdigest(),
        "outer_folds": OUTER_FOLDS, "tuning_folds": TUNING_FOLDS,
        "e15_training_feature_oof_folds": FEATURE_OOF_FOLDS,
        "C_values": list(CS), "threshold_selection": "nested group OOF; maximize balanced accuracy, then accuracy, then fewer switches, then higher threshold",
        "model": {"class": "sklearn.linear_model.LogisticRegression", "penalty": "l2",
                  "solver": "lbfgs", "max_iter": MAX_ITER, "random_state": SEED},
        "feature_layout": names, "feature_dimension": len(names),
        "alternative_rule": "Highest E13 family-max aggregated reliability among labels other than selected label; ties by family-balanced probability then canonical label order.",
        "target": "1 iff alternative equals true label; 0 otherwise, including neither-candidate-correct rows",
        "threads": THREADS, "base_classifier_training": False, "new_inference": False,
        "excluded_checkpoint_ids": excluded,
    }
    _write_json(output / "config.json", configuration)
    _write_json(output / "metrics.json", overall)
    _write_json(output / "fold_metrics.json", {"folds": fold_metrics})
    _write_report(root, output, overall, predictions, row_decisions)
    print(f"E15: accuracy={overall['accuracy']:.4%}, BA={overall['balanced_accuracy']:.4%}, "
          f"delta BA={overall['delta_balanced_accuracy']:+.4%}, corrected/introduced="
          f"{overall['corrected']}/{overall['introduced']}; saved to {output}", flush=True)
    return overall


LABELS_ARRAY = np.asarray(LABELS, dtype=object)


def _write_report(root, output, metrics, predictions, corrections):
    rows = ["# E15 — Logistic Correction of E13-B", "",
            "E15 is a regularized binary logistic gate that may replace E13-B's selected label with its strongest alternative. It uses only cached Trust List 2 v037 checkpoint probabilities and CPU fitting (24 threads). The five outer folds are prompt-group held out; V0 remains a development set because the underlying checkpoints and trust list were selected against it. E13-B's historical Kaggle public score is 0.606; E15 was not submitted.", "",
            "## Result", "",
            "| Experiment | Accuracy | Balanced accuracy | Δ accuracy | Δ balanced accuracy | Corrected | Introduced | Wrong→wrong | Corrections | Correction precision | Adjusted top-2 log loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows.append(f"| E13-B | {metrics['e13_b_accuracy']:.4%} | {metrics['e13_b_balanced_accuracy']:.4%} | — | — | — | — | — |")
    rows.append(f"| E15 | {metrics['accuracy']:.4%} | {metrics['balanced_accuracy']:.4%} | {metrics['delta_accuracy']:+.4%} | {metrics['delta_balanced_accuracy']:+.4%} | {metrics['corrected']} | {metrics['introduced']} | {metrics['wrong_to_wrong_switches']} | {metrics['corrections_made']} | {metrics['correction_precision']:.4%} | {metrics['log_loss_adjusted_top2']:.4f} |")
    rows += ["", f"E13-B reproduced exactly from its saved fold models: **{metrics['e13_baseline_reproduced_exactly']}**. The E13-B reference is 64.6465% balanced accuracy; E15's fold-specific C and threshold are selected using only nested prompt-group splits within the corresponding outer training partition. `Corrected` counts baseline-wrong rows fixed by E15; `Introduced` counts previously correct E13-B rows made wrong; `Wrong→wrong` counts switches that remain wrong. Correction precision is corrected / (corrected + introduced). Adjusted top-2 log loss assigns probability mass only to the E13-B label and its alternative, so it is not directly comparable with a full 18-class calibrated classifier.", "",
             f"E15 nets **{metrics['net_correct']}** additional correct rows across the five held-out folds. This is a small V0 development gain, not evidence to replace E13-B: E13-B remains unchanged as champion, and V0-based checkpoint selection means these folds do not eliminate all development-set contamination.", "",
             "## Outer folds", "", "| Fold | Rows | Accuracy | Balanced accuracy | Δ BA | Corrected | Introduced | C | Threshold | Corrections |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in metrics["folds"]:
        rows.append(f"| {f['fold']} | {f['rows']} | {f['accuracy']:.4%} | {f['balanced_accuracy']:.4%} | {f['delta_balanced_accuracy']:+.4%} | {f['corrected']} | {f['introduced']} | {f['selected_C']:g} | {f['selected_threshold']:.6f} | {f['corrections_made']} |")
    rows += ["", "## Per-label precision and recall", "", "| Label | Support | Precision | Recall | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        m = metrics["per_label"][label]
        rows.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} | {m['f1']:.4%} | {m['tp']} | {m['fp']} | {m['fn']} |")
    rows += ["", "## Confusion matrix", "", "True labels are rows; predictions are columns.", "", "```text", "\t" + "\t".join(LABELS)]
    for label, row in zip(LABELS, metrics["confusion_matrix"]):
        rows.append(label + "\t" + "\t".join(map(str, row)))
    rows += ["```", "", "## Logistic coefficients", "", "Fold-mean coefficients are calculated in the fold-local standardized feature space.", "", "### Positive", "", "| Feature | Mean coefficient |", "|---|---:|"]
    for x in metrics["top_positive_coefficients"]:
        rows.append(f"| `{x['feature']}` | {x['coefficient']:.6g} |")
    rows += ["", "### Negative", "", "| Feature | Mean coefficient |", "|---|---:|"]
    for x in metrics["top_negative_coefficients"]:
        rows.append(f"| `{x['feature']}` | {x['coefficient']:.6g} |")
    rows += ["", "## Corrected and introduced rows", "", f"The row-level decisions are in `{output / 'corrections.csv'}`. Corrected: **{metrics['corrected']}**; introduced: **{metrics['introduced']}**.", "", "## Runtime and memory", "", f"- Nested fitting and evaluation runtime: {metrics['training_runtime_seconds']:.1f} seconds.", f"- Peak process RSS: {metrics['peak_process_memory_mb']:.0f} MB.", "- Base-classifier training and new checkpoint inference: none.", "- Thread limit: 24.", "", "## Artifacts", "", f"- Analysis directory: `{output}`", "- `predictions.parquet` / `predictions.npz`: aligned full-V0 OOF predictions.", "- `corrections.csv`: row IDs and corrected/introduced decisions.", "- `E15_fold*.joblib`: scaler and logistic classifier for each outer fold.", "- `E13_B_fold*.cbm`: exact E13-B fold model copies.", "- `outer*_tuning.json`: nested tuning folds and per-C selection results.", "- `metrics.json`, `fold_metrics.json`, `config.json`: metrics, fold results, and reproducibility metadata.", "", "E13-B remains the production/champion method; this is a comparison experiment only.", ""]
    report = "\n".join(rows)
    (output / "e15.md").write_text(report, encoding="utf-8")
    (root / "docs/e15.md").write_text(report, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
