"""Leakage-controlled learned label arbitration on top of E13-B reliability."""
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
import catboost
from catboost import CatBoostClassifier
import lightgbm
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

from . import ensemble_catalog as catalog
from . import e13_reliability_router as e13
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
THREADS = 24
CHECKPOINTS = 15
INNER_FOLDS = 4
SMOOTHING = 1.0
CAT_FEATURES = ("candidate_label",)
BASE_FEATURES = (
    "candidate_label",
    "family_reliability_max", "family_reliability_mean",
    "family_reliability_min", "family_reliability_std",
    "supporting_checkpoint_count", "supporting_family_count",
    "family_balanced_probability", "probability_margin_vs_competitor",
    "competing_reliability_max", "reliability_margin_vs_competitor",
    "supporting_checkpoint_probability_max", "supporting_checkpoint_probability_mean",
)
PAIR_FEATURES = BASE_FEATURES + (
    "directional_candidate_to_competitor_rate",
    "directional_competitor_to_candidate_rate",
    "historical_candidate_win_probability_when_both_supported",
    "competing_family_count", "shared_conflicting_family_count",
    "candidate_only_family_count", "competitor_only_family_count",
    "family_support_disagreement", "top_two_reliability_margin",
)
EXPERIMENTS = {
    "E14-A": {"features": BASE_FEATURES, "kind": "catboost"},
    "E14-B": {"features": PAIR_FEATURES, "kind": "catboost"},
    "E14-C": {"features": PAIR_FEATURES, "kind": "lambdarank"},
}
CATBOOST_PARAMS = {
    **e13.CATBOOST_PARAMS,
    "task_type": "CPU", "thread_count": THREADS,
    "random_seed": SEED, "allow_writing_files": False,
}
LIGHTGBM_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "n_estimators": 1500, "learning_rate": 0.04, "num_leaves": 31,
    "max_depth": 7, "reg_lambda": 5.0, "random_state": SEED,
    "n_jobs": THREADS, "verbosity": -1, "deterministic": True,
    "force_col_wise": True,
}
LABELS = tuple(LABELS)
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def _catboost_params(iterations: int | None = None) -> dict:
    params = dict(CATBOOST_PARAMS)
    if iterations is not None:
        params["iterations"] = max(1, int(iterations))
        params.pop("od_type", None)
        params.pop("od_wait", None)
    return params


def _groups_from_records(frame: pd.DataFrame) -> list[int]:
    """Get contiguous query sizes and reject rows split across candidate groups."""
    row_ids = frame["_row_index"].to_numpy()
    if not len(row_ids):
        return []
    boundaries = np.r_[0, np.flatnonzero(row_ids[1:] != row_ids[:-1]) + 1, len(row_ids)]
    groups = np.diff(boundaries).astype(int).tolist()
    if len(set(row_ids.tolist())) != len(groups):
        raise ValueError("Candidate records for a row must be contiguous")
    if any(size < 1 for size in groups):
        raise ValueError("Empty ranking query")
    return groups


def _validated_group_splits(rows, truth, groups, n_splits: int):
    rows = np.asarray(rows, dtype=np.int64)
    splits = list(GroupKFold(n_splits=n_splits).split(rows, truth[rows], groups[rows]))
    coverage = np.zeros(len(rows), dtype=np.int8)
    for fit_local, held_local in splits:
        fit_rows, held_rows = rows[fit_local], rows[held_local]
        if set(groups[fit_rows]) & set(groups[held_rows]):
            raise ValueError("Prompt group crosses a cross-fitting split")
        coverage[held_local] += 1
    if not np.all(coverage == 1):
        raise ValueError("Cross-fitting folds must cover rows exactly once")
    return splits


def _pair_statistics(hard: np.ndarray, families: list[str], truth_fit: np.ndarray,
                     fit_rows: np.ndarray, alpha: float = SMOOTHING) -> dict:
    """Smoothed pair statistics from fitting rows, with one vote per family."""
    if alpha <= 0 or hard.ndim != 2 or hard.shape[1] != len(fit_rows):
        raise ValueError("Invalid pair-statistic inputs")
    unique_families = sorted(set(families))
    family_preds = []
    for family in unique_families:
        member_rows = np.flatnonzero(np.asarray(families, dtype=object) == family)
        family_pred = np.empty(hard.shape[1], dtype=np.int64)
        for col in range(hard.shape[1]):
            family_pred[col] = np.bincount(hard[member_rows, col], minlength=len(LABELS)).argmax()
        family_preds.append(family_pred)
    family_preds = np.stack(family_preds)
    k = len(LABELS)
    direction = np.full((k, k), alpha, dtype=np.float64)
    denominators = np.full(k, 2 * alpha, dtype=np.float64)
    support_matrix = np.zeros((k, k), dtype=np.float64)
    both_counts = np.zeros((k, k), dtype=np.float64)
    pair_outcomes = np.zeros((k, k), dtype=np.float64)
    per_row_support = [set(hard[:, col].tolist()) for col in range(hard.shape[1])]
    for fam_pred in family_preds:
        for row, actual in enumerate(truth_fit):
            direction[actual, fam_pred[row]] += 1.0
            denominators[actual] += 1.0
    rates = direction / denominators[:, None]
    for row, supported in enumerate(per_row_support):
        for a in supported:
            for b in supported:
                if a == b:
                    continue
                support_matrix[a, b] += 1.0
                if truth_fit[row] in (a, b):
                    both_counts[a, b] += 1.0
                    if truth_fit[row] == a:
                        pair_outcomes[a, b] += 1.0
    pair_win = (pair_outcomes + alpha) / (both_counts + 2 * alpha)
    return {"directional": rates, "support_matrix": support_matrix,
            "pair_win": pair_win, "families": unique_families}


def build_candidate_frame(probabilities: np.ndarray, reliability: np.ndarray,
                          hard: np.ndarray, families: list[str],
                          truth: np.ndarray, fit_rows: np.ndarray,
                          target_rows: np.ndarray, *, include_pairwise: bool,
                          alpha: float = SMOOTHING) -> tuple[pd.DataFrame, np.ndarray]:
    """Make one ordered row per distinct supported label and target example."""
    p = np.asarray(probabilities, dtype=np.float64)
    rel = np.asarray(reliability, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.int64)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if p.ndim != 3 or p.shape[0] != len(families) or p.shape[2] != len(LABELS):
        raise ValueError("E14 probabilities must be [checkpoints, rows, canonical labels]")
    if rel.shape != hard.shape or hard.shape != (p.shape[0], len(target_rows)):
        raise ValueError("Reliability, hard predictions and probabilities do not align")
    if not np.isfinite(p).all() or not np.isfinite(rel).all() or (rel < 0).any() or (rel > 1).any():
        raise ValueError("E14 inputs contain invalid values")
    if not np.allclose(p.sum(axis=2), 1, atol=1e-4):
        raise ValueError("E14 probability rows are not normalized")
    if np.intersect1d(fit_rows, target_rows).size:
        raise ValueError("E14 fitting and target rows must be disjoint")
    if len(fit_rows) == 0 or len(target_rows) == 0 or truth.shape != (p.shape[1],):
        raise ValueError("Invalid E14 row indices or labels")
    if not np.array_equal(p[:, target_rows].argmax(axis=2), hard):
        raise ValueError("Hard predictions must match canonical probability argmax")
    pair_stats = (_pair_statistics(p[:, fit_rows].argmax(axis=2), families,
                                   truth[fit_rows], fit_rows, alpha)
                  if include_pairwise else None)
    rows: list[dict] = []
    target_prob = p[:, target_rows, :]
    family_probs = e13.family_balanced_probabilities(target_prob, families)
    family_names = sorted(set(families))
    family_cp = {f: np.flatnonzero(np.asarray(families) == f) for f in family_names}
    target_position = {int(row): i for i, row in enumerate(target_rows)}
    for local_row, row_index in enumerate(target_rows):
        support_labels = sorted(set(hard[:, local_row].tolist()))
        if not support_labels:
            raise ValueError("Every example must have at least one candidate label")
        candidates = []
        for label_index in support_labels:
            supporting = np.flatnonzero(hard[:, local_row] == label_index)
            per_family = {}
            family_probability = {}
            for family in family_names:
                members = supporting[np.asarray(families, dtype=object)[supporting] == family]
                if len(members):
                    per_family[family] = float(np.max(rel[members, target_position[int(row_index)]]))
                    family_probability[family] = float(p[family_cp[family], row_index, label_index].mean())
            candidate_reliability = float(np.mean(list(per_family.values())))
            candidates.append({
                "label_index": label_index,
                "supporting": supporting,
                "per_family": per_family,
                "family_probability": family_probability,
                "reliability": candidate_reliability,
                "ensemble_probability": float(family_probs[local_row, label_index]),
            })
        ordered = sorted(candidates, key=lambda x: (-x["reliability"],
                                                     -x["ensemble_probability"],
                                                     x["label_index"]))
        competitor_lookup = {}
        for c in candidates:
            competitors = [x for x in ordered if x["label_index"] != c["label_index"]]
            competitor_lookup[c["label_index"]] = competitors[0] if competitors else None
        top_margin = (ordered[0]["reliability"] - ordered[1]["reliability"]
                      if len(ordered) > 1 else 0.0)
        for c in sorted(candidates, key=lambda x: x["label_index"]):
            comp = competitor_lookup[c["label_index"]]
            ci = c["label_index"]
            comp_i = comp["label_index"] if comp else ci
            fam_rel = np.asarray(list(c["per_family"].values()), dtype=np.float64)
            cp_probs = p[c["supporting"], row_index, ci]
            record = {
                "_row_index": int(row_index), "candidate_label": LABELS[ci],
                "_label_index": int(ci), "_is_correct": int(ci == truth[row_index]),
                "_candidate_count": len(candidates),
                "_candidate_covered": int(truth[row_index] in support_labels),
                "family_reliability_max": float(fam_rel.max()),
                "family_reliability_mean": float(fam_rel.mean()),
                "family_reliability_min": float(fam_rel.min()),
                "family_reliability_std": float(fam_rel.std(ddof=0)),
                "supporting_checkpoint_count": int(len(c["supporting"])),
                "supporting_family_count": int(len(c["per_family"])),
                "family_balanced_probability": c["ensemble_probability"],
                "probability_margin_vs_competitor": c["ensemble_probability"] - (
                    comp["ensemble_probability"] if comp else c["ensemble_probability"]),
                "competing_reliability_max": comp["reliability"] if comp else c["reliability"],
                "reliability_margin_vs_competitor": c["reliability"] - (
                    comp["reliability"] if comp else c["reliability"]),
                "supporting_checkpoint_probability_max": float(cp_probs.max()),
                "supporting_checkpoint_probability_mean": float(cp_probs.mean()),
            }
            if include_pairwise:
                shared = set(c["per_family"]) & (set(comp["per_family"]) if comp else set())
                candidate_only = set(c["per_family"]) - shared
                competitor_only = set(comp["per_family"]) - shared if comp else set()
                record.update({
                    "directional_candidate_to_competitor_rate": float(pair_stats["directional"][ci, comp_i]) if comp else 0.0,
                    "directional_competitor_to_candidate_rate": float(pair_stats["directional"][comp_i, ci]) if comp else 0.0,
                    "historical_candidate_win_probability_when_both_supported": float(pair_stats["pair_win"][ci, comp_i]) if comp else 0.5,
                    "competing_family_count": len(comp["per_family"]) if comp else 0,
                    "shared_conflicting_family_count": len(shared),
                    "candidate_only_family_count": len(candidate_only),
                    "competitor_only_family_count": len(competitor_only),
                    "family_support_disagreement": float(len(candidate_only) - len(competitor_only)),
                    "top_two_reliability_margin": top_margin,
                })
            rows.append(record)
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame if c not in {"_row_index", "candidate_label", "_label_index"}]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("E14 feature table contains non-finite values")
    return frame, family_probs


def select_candidate_scores(frame: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select one label per example, with deterministic reliability/probability ties."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("E14 selector scores do not align with candidate records")
    predictions, rows = [], []
    for row_index, positions in frame.groupby("_row_index", sort=True).indices.items():
        best_score = np.max(scores[positions])
        tied = [int(i) for i in positions if np.isclose(scores[i], best_score, rtol=0, atol=1e-12)]
        chosen = min(tied, key=lambda i: (
            -float(frame.iloc[i]["family_reliability_mean"]),
            -float(frame.iloc[i]["family_balanced_probability"]),
            int(frame.iloc[i]["_label_index"])))
        predictions.append(int(frame.iloc[chosen]["_label_index"]))
        rows.append(int(row_index))
    return np.asarray(rows, dtype=np.int64), np.asarray(predictions, dtype=np.int64)


def _candidate_groups(frame: pd.DataFrame):
    groups = _groups_from_records(frame)
    y = frame["_is_correct"].to_numpy(dtype=np.int8)
    return groups, y


def _features(frame: pd.DataFrame, names: tuple[str, ...], kind: str = "catboost") -> pd.DataFrame:
    x = frame.loc[:, names].copy()
    if kind == "lambdarank":
        x["candidate_label"] = pd.Categorical(x["candidate_label"].astype(str), categories=LABELS)
    else:
        x["candidate_label"] = x["candidate_label"].astype(str)
    return x


def _fit_selector(kind: str, features: tuple[str, ...], train_frame: pd.DataFrame,
                  stop_frame: pd.DataFrame):
    x_train, x_stop = (_features(train_frame, features, kind),
                       _features(stop_frame, features, kind))
    y_train, y_stop = train_frame["_is_correct"].to_numpy(), stop_frame["_is_correct"].to_numpy()
    if kind == "catboost":
        early = CatBoostClassifier(**_catboost_params())
        early.fit(x_train, y_train, cat_features=["candidate_label"],
                  eval_set=(x_stop, y_stop), early_stopping_rounds=120,
                  verbose=False, use_best_model=True)
        iterations = max(1, int(early.get_best_iteration()) + 1)
        final = CatBoostClassifier(**_catboost_params(iterations))
        final.fit(x_train, y_train, cat_features=["candidate_label"], verbose=False)
        importance = dict(zip(features, map(float, final.feature_importances_)))
        return final, iterations, importance, early
    train_group, _ = _candidate_groups(train_frame)
    stop_group, _ = _candidate_groups(stop_frame)
    early = LGBMRanker(**LIGHTGBM_PARAMS)
    early.fit(x_train, y_train, group=train_group, eval_X=x_stop, eval_y=y_stop,
              eval_group=[stop_group], eval_metric="ndcg", eval_at=[1],
              callbacks=[early_stopping(120, verbose=False), log_evaluation(0)])
    iterations = int(early.best_iteration_ or LIGHTGBM_PARAMS["n_estimators"])
    params = dict(LIGHTGBM_PARAMS, n_estimators=iterations)
    final = LGBMRanker(**params)
    final.fit(x_train, y_train, group=train_group)
    importance = dict(zip(features, map(float, final.feature_importances_)))
    return final, iterations, importance, early


def _predict_scores(model, kind, frame, features):
    x = _features(frame, features, kind)
    if kind == "catboost":
        scores = model.predict_proba(x)[:, 1]
    else:
        scores = model.predict(x, num_threads=THREADS)
    return np.asarray(scores, dtype=np.float64)


def _confusion_features_for_set(probabilities, reliability, hard, roster,
                                 all_families, truth, fit_rows, target_rows,
                                 *, include_pairwise):
    selected_prob = probabilities[roster]
    families = [all_families[i] for i in roster]
    return build_candidate_frame(selected_prob, reliability, hard, families, truth,
                                 fit_rows, target_rows, include_pairwise=include_pairwise)


def _e13_aux(probabilities, ids, families, truth, groups, fit_rows, target_rows,
             h09, row_ids):
    _, _, _, aux = e13._fit_router(
        probabilities, ids, families, truth, groups, np.asarray(fit_rows),
        np.asarray(target_rows), CHECKPOINTS, "family_max", h09, row_ids,
        model_path=None, exclude_ids=())
    return aux


def _make_oof_records(rows, matrices, ids, families, truth, groups, h09, row_ids,
                      *, include_pairwise):
    frames, logs, seen_rows = [], [], set()
    for fold_no, (fit_local, held_local) in enumerate(
            _validated_group_splits(rows, truth, groups, INNER_FOLDS), 1):
        fit_rows, held_rows = rows[fit_local], rows[held_local]
        if set(groups[fit_rows]) & set(groups[held_rows]):
            raise ValueError("E14 inner cross-fit prompt groups overlap")
        overlap = seen_rows.intersection(map(int, held_rows))
        if overlap:
            raise ValueError("An E14 OOF training example appeared in more than one inner fold")
        seen_rows.update(map(int, held_rows))
        reliability, hard, roster = _e13_aux(
            matrices, ids, families, truth, groups, fit_rows, held_rows, h09, row_ids)
        frame, _ = _confusion_features_for_set(
            matrices, reliability, hard, roster, families, truth, fit_rows, held_rows,
            include_pairwise=include_pairwise)
        frames.append(frame)
        logs.append({"fold": fold_no, "fit_rows": len(fit_rows), "target_rows": len(held_rows),
                     "fit_prompt_groups": len(set(groups[fit_rows])),
                     "target_prompt_groups": len(set(groups[held_rows])),
                     "checkpoint_ids": [ids[i] for i in roster],
                     "target_row_ids": [str(row_ids[i]) for i in held_rows],
                     "candidate_records": int(len(frame)),
                     "uncovered_examples": int(frame.groupby("_row_index")["_candidate_covered"].first().eq(0).sum())})
    result = pd.concat(frames, ignore_index=True)
    if seen_rows != set(map(int, rows)):
        raise ValueError("E14 inner folds did not cover every training example exactly once")
    return result, logs


def _metrics(truth, pred, baseline, candidate_covered):
    cm = confusion_matrix(truth, pred, labels=np.arange(len(LABELS)))
    precision, recall, _, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "corrected_vs_E13B": int(((baseline != truth) & (pred == truth)).sum()),
        "introduced_vs_E13B": int(((baseline == truth) & (pred != truth)).sum()),
        "candidate_coverage": float(np.mean(candidate_covered)),
        "uncovered_count": int((~candidate_covered).sum()),
        "confusion_matrix": cm.tolist(),
        "per_label": {LABELS[i]: {"support": int(support[i]), "precision": float(precision[i]),
                                   "recall": float(recall[i]), "tp": int(cm[i, i]),
                                   "fp": int(cm[:, i].sum() - cm[i, i]),
                                   "fn": int(cm[i].sum() - cm[i, i])}
                      for i in range(len(LABELS))},
        "selected_candidate_distribution": {
            LABELS[i]: int(np.count_nonzero(pred == i)) for i in range(len(LABELS))},
    }


def _find_e13_baseline(root: Path, v0: pd.DataFrame):
    expected_ba = 0.6464653243847874
    candidates = []
    for results_path in (root / "artifacts/analysis/e13").glob("*/results.json"):
        try:
            result = json.loads(results_path.read_text())
            if result["trust_list"]["version"] != "trust-list-2-v037":
                continue
            item = next(x for x in result["results"] if x["ensemble"] == "E13-B")
            if abs(float(item["balanced_accuracy"]) - expected_ba) < 1e-12:
                candidates.append((results_path.parent, result, item))
        except (KeyError, ValueError, StopIteration, json.JSONDecodeError):
            continue
    if len(candidates) != 1:
        raise ValueError(f"Expected one saved v037 E13-B OOF result, found {len(candidates)}")
    path, result, details = candidates[0]
    frame = pd.read_parquet(path / "predictions.parquet")
    if frame.id.astype(str).tolist() != v0.id.astype(str).tolist():
        raise ValueError("Saved E13-B OOF predictions do not align to the current V0 row order")
    if frame.y_true.astype(str).tolist() != v0.model.astype(str).tolist():
        raise ValueError("Saved E13-B truth labels differ from current V0")
    pred = pd.Categorical(frame["E13-B"].astype(str), categories=LABELS).codes
    if (pred < 0).any():
        raise ValueError("Saved E13-B contains noncanonical labels")
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes
    ba = balanced_accuracy_score(truth, pred)
    if abs(ba - expected_ba) > 1e-12:
        raise ValueError(f"Saved E13-B local BA changed: {ba}")
    expected_rows = []
    for fold in details["folds"]:
        expected_rows.extend(fold["held_ids"])
    if len(expected_rows) != len(v0) or len(set(expected_rows)) != len(v0):
        raise ValueError("Saved E13-B folds do not cover V0 exactly once")
    return pred.astype(np.int64), path, result, details


def _prepare_fold(fold_no, outer_fit, outer_held, matrices, ids, families,
                  truth, groups, h09, row_ids, output):
    # The E14 early-stop rows are excluded from all E13 reliability fitting for
    # both the E14 training records and their validation records.
    early_split = _validated_group_splits(outer_fit, truth, groups, 4)[-1]
    local_router_fit, local_stop = early_split
    router_fit_rows, stop_rows = outer_fit[local_router_fit], outer_fit[local_stop]
    if set(groups[router_fit_rows]) & set(groups[stop_rows]):
        raise ValueError("E14 early-stopping prompt groups overlap")

    # Isolated early-stopping training data: E13 OOF reliability over router_fit.
    early_train, early_logs = _make_oof_records(
        router_fit_rows, matrices, ids, families, truth, groups, h09, row_ids,
        include_pairwise=True)
    stop_rel, stop_hard, stop_roster = _e13_aux(
        matrices, ids, families, truth, groups, router_fit_rows, stop_rows, h09, row_ids)
    stop_frames = {}
    for code, config in EXPERIMENTS.items():
        stop_frames[code], _ = _confusion_features_for_set(
            matrices, stop_rel, stop_hard, stop_roster, families, truth,
            router_fit_rows, stop_rows,
            include_pairwise=(code != "E14-A"))

    # Final E14 fit rows: honest E13 reliability predictions via inner cross-fit.
    final_train, final_logs = _make_oof_records(
        outer_fit, matrices, ids, families, truth, groups, h09, row_ids,
        include_pairwise=True)
    # Full outer-fit E13-B router predicts reliability for untouched outer rows.
    held_rel, held_hard, held_roster = _e13_aux(
        matrices, ids, families, truth, groups, outer_fit, outer_held, h09, row_ids)
    held_frames = {}
    for code, config in EXPERIMENTS.items():
        held_frames[code], family_probs = _confusion_features_for_set(
            matrices, held_rel, held_hard, held_roster, families, truth,
            outer_fit, outer_held, include_pairwise=(code != "E14-A"))
    # Save shared features once; E14-A/B/C reuse this materialization.
    out = output / "features"
    out.mkdir(exist_ok=True)
    early_train.to_parquet(out / f"fold{fold_no}_early_train.parquet", index=False)
    final_train.to_parquet(out / f"fold{fold_no}_train.parquet", index=False)
    for code in EXPERIMENTS:
        stop_frames[code].to_parquet(out / f"fold{fold_no}_{code}_stop.parquet", index=False)
        held_frames[code].to_parquet(out / f"fold{fold_no}_{code}_held.parquet", index=False)
    metadata = {"fold": fold_no, "fit_row_ids": [str(row_ids[i]) for i in outer_fit],
                "held_row_ids": [str(row_ids[i]) for i in outer_held],
                "early_router_fit_row_ids": [str(row_ids[i]) for i in router_fit_rows],
                "early_stop_row_ids": [str(row_ids[i]) for i in stop_rows],
                "early_inner_folds": early_logs, "final_inner_folds": final_logs,
                "held_checkpoint_ids": [ids[i] for i in held_roster],
                "candidate_coverage": float(held_frames["E14-A"].groupby("_row_index")["_candidate_covered"].first().mean()),
                "uncovered_examples": int(held_frames["E14-A"].groupby("_row_index")["_candidate_covered"].first().eq(0).sum()),
                "candidate_record_count": int(len(held_frames["E14-A"]))}
    return {"early_train": early_train, "final_train": final_train,
            "stop_frames": stop_frames, "held_frames": held_frames,
            "held_roster": [ids[i] for i in held_roster],
            "metadata": metadata}


def _fold_metrics(truth, pred, baseline, frame, fold, details):
    ordered = frame.sort_values("_row_index", kind="stable")
    coverage = ordered.groupby("_row_index")["_candidate_covered"].first().to_numpy(dtype=bool)
    m = _metrics(truth, pred, baseline, coverage)
    m.update({"fold": fold, "rows": len(truth), **details})
    return m


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output) if output else root / "artifacts/analysis/e14" / stamp
    if output.exists():
        raise FileExistsError(output)
    v0 = catalog.cascades._v0(root)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    groups = v0.prompt_id.astype(str).to_numpy()
    row_ids = v0.id.astype(str).to_numpy()
    if len(v0) != 2699 or (truth < 0).any():
        raise ValueError("Expected 2,699 V0 rows with canonical labels")
    baseline, baseline_dir, baseline_result, baseline_details = _find_e13_baseline(root, v0)
    eligible, members, probability_dict, excluded, snapshot = catalog._load_inputs(root, v0)
    ids = e13._sorted_ids(probability_dict)
    if snapshot["version"] != "trust-list-2-v037":
        raise ValueError(f"E14 instructions pin v037 E13-B comparison; latest is {snapshot['version']}")
    if len(ids) < CHECKPOINTS:
        raise ValueError("Less than 15 eligible probability caches are available")
    matrices = np.stack([probability_dict[sid] for sid in ids])
    families = [e13._model_family(members[sid]["model"]) for sid in ids]
    if matrices.shape != (len(ids), len(v0), len(LABELS)):
        raise ValueError("Checkpoint probability shape is not [checkpoint, 2699, 18]")
    # E13 is reused unchanged. Configure the helper's CatBoost instances to 24 CPU threads.
    e13.CATBOOST_PARAMS["thread_count"] = THREADS
    output.mkdir(parents=True)
    (output / "models").mkdir()
    all_predictions = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for code in EXPERIMENTS:
        all_predictions[code] = ""
    all_results = {code: [] for code in EXPERIMENTS}
    all_importance = {code: {f: [] for f in EXPERIMENTS[code]["features"]}
                      for code in EXPERIMENTS}
    coverage_bits = np.zeros(len(v0), dtype=bool)
    assignment = np.zeros(len(v0), dtype=np.int8)
    elapsed_start = time.monotonic()
    fold_rosters, cache_meta = [], []
    outer_splits = _validated_group_splits(np.arange(len(v0)), truth, groups, 5)

    # Match the saved E13-B fold partition exactly before evaluating E14.
    saved_folds = baseline_details["folds"]
    for (fit, held), saved in zip(outer_splits, saved_folds):
        if set(row_ids[held]) != set(saved["held_ids"]):
            raise ValueError(f"Current outer fold assignments do not match saved E13-B fold {saved['fold']}")
    for fold, (fit, held) in enumerate(outer_splits, 1):
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Outer prompt groups overlap")
        prepared = _prepare_fold(fold, fit, held, matrices, ids, families, truth, groups,
                                 baseline, row_ids, output)
        fold_rosters.append(prepared["metadata"])
        held_frame = prepared["held_frames"]["E14-A"]
        held_coverage = (held_frame.groupby("_row_index")["_candidate_covered"].first().astype(bool))
        coverage_bits[held_coverage.index.to_numpy(dtype=int)] = held_coverage.to_numpy()
        assignment[held] += 1
        for code, config in EXPERIMENTS.items():
            features = config["features"]
            train_frame = prepared["final_train"]
            stop_frame = prepared["stop_frames"][code]
            test_frame = prepared["held_frames"][code]
            fold_start = time.monotonic()
            selector, iterations, importance, early_model = _fit_selector(
                config["kind"], features,
                train_frame.loc[train_frame["_candidate_count"].notna()].copy(), stop_frame)
            scores = _predict_scores(selector, config["kind"], test_frame, features)
            predicted_rows, pred = select_candidate_scores(test_frame, scores)
            if not np.array_equal(predicted_rows, held):
                raise ValueError("E14 selected predictions do not preserve outer row alignment")
            all_predictions.loc[held, code] = np.asarray(LABELS, dtype=object)[pred]
            fold_metric = _fold_metrics(truth[held], pred, baseline[held], test_frame, fold, {
                "training_seconds": float(time.monotonic() - fold_start),
                "best_iterations": iterations,
                "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                "candidate_records_train": int(len(train_frame)),
                "candidate_records_stop": int(len(stop_frame)),
                "roster": prepared["held_roster"],
                "early_stopping_prompt_groups": len(set(groups[held])),
            })
            all_results[code].append(fold_metric)
            for name, value in importance.items():
                all_importance[code][name].append(value)
            model_dir = output / "models" / code
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / (f"fold{fold}." +
                                      ("cbm" if config["kind"] == "catboost" else "txt"))
            if config["kind"] == "catboost":
                selector.save_model(str(model_path))
            else:
                selector.booster_.save_model(str(model_path))
            print(f"{code} fold {fold}/5: acc={fold_metric['accuracy']:.4%} "
                  f"ba={fold_metric['balanced_accuracy']:.4%} coverage={fold_metric['candidate_coverage']:.4%} "
                  f"trees={iterations} seconds={fold_metric['training_seconds']:.1f}", flush=True)
    if not np.all(assignment == 1) or (all_predictions[list(EXPERIMENTS)] == "").any().any():
        raise ValueError("Each of 2,699 V0 examples must be evaluated exactly once per experiment")
    prediction_path = output / "predictions.parquet"
    all_predictions.to_parquet(prediction_path, index=False)
    result_entries = []
    truth_np = truth
    for code in EXPERIMENTS:
        pred = pd.Categorical(all_predictions[code].astype(str), categories=LABELS).codes
        overall = _metrics(truth_np, pred, baseline, coverage_bits)
        importance = {name: float(np.mean(vals)) if vals else 0.0
                      for name, vals in all_importance[code].items()}
        result_entries.append({
            "experiment": code, **overall,
            "delta_vs_E13B_accuracy": float(overall["accuracy"] - accuracy_score(truth_np, baseline)),
            "delta_vs_E13B_balanced_accuracy": float(overall["balanced_accuracy"] -
                                                         balanced_accuracy_score(truth_np, baseline)),
            "folds": all_results[code], "feature_importance_mean": importance,
            "training_duration_seconds": float(sum(x["training_seconds"] for x in all_results[code])),
            "peak_rss_mb": float(max(x["peak_rss_mb"] for x in all_results[code])),
        })
    baseline_truth_acc = float(accuracy_score(truth_np, baseline))
    baseline_ba = float(balanced_accuracy_score(truth_np, baseline))
    total_elapsed = time.monotonic() - elapsed_start
    selector_seconds = sum(item["training_duration_seconds"] for item in result_entries)
    (output / "config.json").write_text(json.dumps({
        "trust_list": snapshot, "seed": SEED, "cpu_only": True, "threads": THREADS,
        "folds": "5-fold GroupKFold(prompt_id)", "e13": {"implementation": "src/e13_reliability_router.py",
            "roster_method": "E13-B family_max", "checkpoint_count": CHECKPOINTS,
            "reliability_catboost": CATBOOST_PARAMS, "smoothing": e13.SMOOTHING},
        "e14_early_stopping": "prompt-group-disjoint split; E13 reliability rows fitted without stop groups",
        "inner_reliability_crossfit_folds": INNER_FOLDS,
        "pairwise_smoothing_alpha": SMOOTHING,
        "catboost": CATBOOST_PARAMS, "lightgbm": LIGHTGBM_PARAMS,
        "experiments": {k: {"kind": v["kind"], "features": list(v["features"])}
                        for k, v in EXPERIMENTS.items()},
        "e13_b_oof_source": str(baseline_dir),
        "e13_b_local_accuracy": baseline_truth_acc,
        "e13_b_local_balanced_accuracy": baseline_ba,
        "e13_b_kaggle_public_score": 0.606,
        "package_versions": {"catboost": catboost.__version__, "lightgbm": lightgbm.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__},
        "source_cache_count": len(ids), "eligible_checkpoint_ids": ids,
        "checkpoint_families": dict(zip(ids, families)),
        "excluded_from_trust_list": excluded,
    }, indent=2) + "\n")
    result = {"trust_list": snapshot, "validation_rows": len(v0), "labels": list(LABELS),
              "row_ids": row_ids.tolist(), "baseline": {
                  "name": "E13-B", "accuracy": baseline_truth_acc,
                  "balanced_accuracy": baseline_ba, "kaggle_public_score": 0.606,
                  "oof_source": str(baseline_dir),
                  "folds": baseline_details["folds"]},
              "training_threads": THREADS, "total_elapsed_seconds": total_elapsed,
              "shared_crossfit_feature_pipeline_and_io_seconds": max(0.0, total_elapsed - selector_seconds),
              "fold_metadata": fold_rosters, "results": result_entries}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _write_report(root, output, result, all_predictions)
    print("| Experiment | Accuracy | Balanced accuracy | Δ vs E13-B | Corrected | Introduced | Candidate coverage |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
    for item in result_entries:
        print(f"| {item['experiment']} | {item['accuracy']:.4%} | {item['balanced_accuracy']:.4%} | "
              f"{item['delta_vs_E13B_balanced_accuracy']:+.4%} | {item['corrected_vs_E13B']} | "
              f"{item['introduced_vs_E13B']} | {item['candidate_coverage']:.4%} |", flush=True)
    print(f"Saved E14 artifacts to {output}", flush=True)
    return result


def _write_report(root: Path, output: Path, result: dict, predictions: pd.DataFrame):
    truth = pd.Categorical(predictions.y_true.astype(str), categories=LABELS).codes
    baseline_dir = Path(result["baseline"]["oof_source"])
    baseline_frame = pd.read_parquet(baseline_dir / "predictions.parquet")
    baseline = pd.Categorical(baseline_frame["E13-B"].astype(str), categories=LABELS).codes
    summary = ["# E14 — Learned Label-Level Arbitration", "",
               f"Trust List: **{result['trust_list']['version']}** · V0 rows: **{result['validation_rows']}** · "
               "five prompt-group outer folds · CPU only · 24 threads.", "",
               "E13-B's 15-checkpoint roster selection and CatBoost correctness router are reused. E14 learns "
               "which supported label to choose from the E13 reliability evidence. All checkpoint selection, "
               "reliability and pairwise statistics are fit on training-fold labels only. This is a V0 "
               "development evaluation, not a Kaggle estimate.", "",
               f"Saved E13-B local OOF balanced accuracy: **{result['baseline']['balanced_accuracy']:.4%}**. "
               f"Its Kaggle public score is **{result['baseline']['kaggle_public_score']:.3f}**; that external "
               "score is separate from this local five-fold comparison.", "",
               "## Results", "",
               "| Experiment | Accuracy | Balanced accuracy | Δ vs E13-B | Corrected | Introduced | Candidate coverage |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for item in result["results"]:
        summary.append(f"| {item['experiment']} | {item['accuracy']:.4%} | {item['balanced_accuracy']:.4%} | "
                       f"{item['delta_vs_E13B_balanced_accuracy']:+.4%} | {item['corrected_vs_E13B']} | "
                       f"{item['introduced_vs_E13B']} | {item['candidate_coverage']:.4%} |")
    selector_seconds = sum(item["training_duration_seconds"] for item in result["results"])
    summary += ["", f"Selector fitting and held-out scoring took **{selector_seconds:.1f} seconds** "
                "across the three experiments. Shared E13 cross-fitting, feature preparation and pipeline "
                f"work took **{result['shared_crossfit_feature_pipeline_and_io_seconds']:.1f} seconds**; "
                f"end-to-end runtime was **{result['total_elapsed_seconds']:.1f} seconds**. Peak process "
                f"RSS was **{max(item['peak_rss_mb'] for item in result['results']):.0f} MB**. "
                f"All variants had **{result['results'][0]['uncovered_count']} uncovered examples** "
                "because none of the selected checkpoints predicted the correct label.", ""]
    summary += ["", "## Fold results", "",
                "| Experiment | Fold | Rows | Accuracy | Balanced accuracy | Coverage | Uncovered | Trees | Seconds | Peak RSS MB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in result["results"]:
        for fold in item["folds"]:
            summary.append(f"| {item['experiment']} | {fold['fold']} | {fold['rows']} | {fold['accuracy']:.4%} | "
                           f"{fold['balanced_accuracy']:.4%} | {fold['candidate_coverage']:.4%} | "
                           f"{fold['uncovered_count']} | {fold['best_iterations']} | "
                           f"{fold['training_seconds']:.1f} | {fold['peak_rss_mb']:.0f} |")
    for item in result["results"]:
        summary += ["", f"## {item['experiment']} details", "", "### Per-label precision and recall", "",
                    "| Label | Support | Precision | Recall | TP | FP | FN |",
                    "|---|---:|---:|---:|---:|---:|---:|"]
        for label in LABELS:
            m = item["per_label"][label]
            summary.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} | "
                           f"{m['tp']} | {m['fp']} | {m['fn']} |")
        summary += ["", "### Confusion matrix", "", "```text", "\t" + "\t".join(LABELS)]
        for label, row in zip(LABELS, item["confusion_matrix"]):
            summary.append(label + "\t" + "\t".join(map(str, row)))
        summary += ["```", "", "### Selected candidate distribution", "",
                    "| Label | Predictions |", "|---|---:|"]
        for label in LABELS:
            summary.append(f"| {label} | {item['selected_candidate_distribution'][label]} |")
        summary += ["", "### Mean feature importance", "",
                    "| Feature | Importance |", "|---|---:|"]
        for name, value in sorted(item["feature_importance_mean"].items(),
                                  key=lambda x: (-x[1], x[0])):
            summary.append(f"| {name} | {value:.6g} |")
    summary += ["", "## Error changes versus E13-B", ""]
    ids = predictions.id.astype(str).to_numpy()
    for code in EXPERIMENTS:
        pred_code = pd.Categorical(predictions[code].astype(str), categories=LABELS).codes
        corrected = np.flatnonzero((baseline != truth) & (pred_code == truth))
        introduced = np.flatnonzero((baseline == truth) & (pred_code != truth))
        summary += [f"### {code}: corrected examples (up to 20, V0 row order)", "",
                    "| Row ID | True label | E13-B prediction | E14 prediction |",
                    "|---|---|---|---|"]
        for i in corrected[:20]:
            summary.append(f"| {ids[i]} | {LABELS[truth[i]]} | {LABELS[baseline[i]]} | {LABELS[pred_code[i]]} |")
        summary += ["", f"### {code}: errors introduced (up to 20, V0 row order)", "",
                    "| Row ID | True label | E13-B prediction | E14 prediction |",
                    "|---|---|---|---|"]
        for i in introduced[:20]:
            summary.append(f"| {ids[i]} | {LABELS[truth[i]]} | {LABELS[baseline[i]]} | {LABELS[pred_code[i]]} |")
    summary += ["", "## Saved artifacts", "",
                f"- Run directory: `{output}`",
                "- `predictions.parquet`: all three held-out predictions for exactly 2,699 V0 rows.",
                "- `features/`: shared E13-derived candidate-level features for the three selectors.",
                "- `models/`: fold-specific CatBoost and LightGBM models.",
                "- `config.json` and `results.json`: pinned Trust List and reproducible settings/results.", ""]
    text = "\n".join(summary)
    (output / "e14.md").write_text(text)
    (root / "docs/e14.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
