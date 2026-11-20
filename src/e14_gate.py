"""A cross-fitted CatBoost gate for switching E13-B to its strongest rival label."""
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

from . import ensemble_catalog as catalog
from . import e13_reliability_router as e13
from .v2_checkpoint_trust import LABELS
from .e14 import _validated_group_splits

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
E13_CHECKPOINTS = 15
INNER_FOLDS = 4
SMOOTHING = 1.0
CATEGORICAL = ("baseline_label", "competitor_label")
FEATURES = (
    "baseline_label", "competitor_label",
    "baseline_e13_score", "competitor_e13_score", "e13_reliability_margin",
    "baseline_probability", "competitor_probability", "probability_margin",
    "baseline_checkpoint_probability_max", "baseline_checkpoint_probability_mean",
    "competitor_checkpoint_probability_max", "competitor_checkpoint_probability_mean",
    "baseline_checkpoint_support", "competitor_checkpoint_support",
    "baseline_family_support", "competitor_family_support",
    "baseline_to_competitor_confusion", "competitor_to_baseline_confusion",
    "historical_competitor_win_given_pair_support",
    "shared_conflicting_families", "baseline_only_families", "competitor_only_families",
    "family_support_margin", "checkpoint_probability_margin",
)
GATE_PARAMS = {
    "loss_function": "Logloss", "eval_metric": "Logloss", "task_type": "CPU",
    "depth": 7, "learning_rate": 0.04, "iterations": 1500, "l2_leaf_reg": 5.0,
    "random_seed": SEED, "thread_count": 24, "od_type": "Iter", "od_wait": 120,
    "verbose": False, "allow_writing_files": False, "random_strength": 0.5,
}
LABELS = tuple(LABELS)


def _sorted_ids(probabilities):
    return sorted(probabilities, key=lambda sid: int(str(sid).lstrip("N")))


def _load_baseline(root, v0):
    target_ba = 0.6464653243847874
    found = []
    for path in (root / "artifacts/analysis/e13").glob("*/results.json"):
        try:
            result = json.loads(path.read_text())
            if result["trust_list"]["version"] != "trust-list-2-v037":
                continue
            item = next(x for x in result["results"] if x["ensemble"] == "E13-B")
            if abs(item["balanced_accuracy"] - target_ba) <= 1e-12:
                found.append((path.parent, result, item))
        except (KeyError, StopIteration, json.JSONDecodeError):
            continue
    if len(found) != 1:
        raise ValueError(f"Expected one v037 E13-B result at {target_ba}, found {len(found)}")
    directory, result, item = found[0]
    frame = pd.read_parquet(directory / "predictions.parquet")
    if frame.id.astype(str).tolist() != v0.id.astype(str).tolist():
        raise ValueError("E13-B baseline row IDs do not match V0")
    if frame.y_true.astype(str).tolist() != v0.model.astype(str).tolist():
        raise ValueError("E13-B baseline truth labels do not match V0")
    pred = pd.Categorical(frame["E13-B"].astype(str), categories=LABELS).codes
    if (pred < 0).any():
        raise ValueError("E13-B baseline contains non-canonical labels")
    return pred.astype(np.int64), directory, result, item


def _family_pair_statistics(hard_fit, truth_fit, families, alpha=SMOOTHING):
    """Family-vote directional confusion and pairwise conditional win rates."""
    k = len(LABELS)
    names = sorted(set(families))
    family_predictions = []
    for name in names:
        members = np.flatnonzero(np.asarray(families, dtype=object) == name)
        row_pred = np.empty(hard_fit.shape[1], dtype=np.int64)
        for col in range(hard_fit.shape[1]):
            row_pred[col] = np.bincount(hard_fit[members, col], minlength=k).argmax()
        family_predictions.append(row_pred)
    family_predictions = np.stack(family_predictions)
    direction = np.full((k, k), alpha, dtype=np.float64)
    denominator = np.full(k, 2 * alpha, dtype=np.float64)
    for pred in family_predictions:
        for i, actual in enumerate(truth_fit):
            direction[actual, pred[i]] += 1
            denominator[actual] += 1
    direction /= denominator[:, None]

    pair_total = np.zeros((k, k), dtype=np.float64)
    pair_right = np.zeros((k, k), dtype=np.float64)
    support_sets = [set(hard_fit[:, i].tolist()) for i in range(hard_fit.shape[1])]
    for row, supported in enumerate(support_sets):
        for a in supported:
            for b in supported:
                if a != b and truth_fit[row] in (a, b):
                    pair_total[a, b] += 1
                    pair_right[a, b] += truth_fit[row] == b
    # Oriented P(true=b | true in {a,b}, both labels supported).
    competitor_win = (pair_right + alpha) / (pair_total + 2 * alpha)
    return direction, competitor_win


def _label_scores(hard, reliability, families):
    n_models, n_rows = hard.shape
    scores = np.full((n_rows, len(LABELS)), -np.inf, dtype=np.float64)
    family_sets = [[set() for _ in range(n_rows)] for _ in range(len(LABELS))]
    support_counts = np.zeros((n_rows, len(LABELS)), dtype=np.int16)
    family_reliability = [[{} for _ in range(n_rows)] for _ in range(len(LABELS))]
    for row in range(n_rows):
        for label in np.unique(hard[:, row]):
            votes = np.flatnonzero(hard[:, row] == label)
            per_family = {}
            for idx in votes:
                per_family[families[idx]] = max(per_family.get(families[idx], -np.inf),
                                                 float(reliability[idx, row]))
            scores[row, label] = float(np.mean(list(per_family.values())))
            family_sets[label][row] = set(per_family)
            support_counts[row, label] = len(votes)
            family_reliability[label][row] = per_family
    return scores, family_sets, support_counts, family_reliability


def _make_pair_frame(probabilities, reliability, hard, roster, all_families,
                     truth, fit_rows, target_rows):
    roster = np.asarray(roster, dtype=np.int64)
    p = probabilities[roster]
    families = [all_families[i] for i in roster]
    reliability = np.asarray(reliability, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.int64)
    fit_rows, target_rows = np.asarray(fit_rows, dtype=np.int64), np.asarray(target_rows, dtype=np.int64)
    if hard.shape != reliability.shape or hard.shape != (len(roster), len(target_rows)):
        raise ValueError("E13-B held reliability and hard predictions are not aligned")
    if not np.array_equal(p[:, target_rows].argmax(axis=2), hard):
        raise ValueError("E13-B hard predictions do not align with canonical probability scores")
    fit_hard = p[:, fit_rows].argmax(axis=2)
    direction, pair_win = _family_pair_statistics(fit_hard, truth[fit_rows], families)
    ensemble = e13.family_balanced_probabilities(p[:, target_rows], families)
    label_scores, family_sets, support_counts, family_rel = _label_scores(hard, reliability, families)
    base_pred, _ = e13.route_predictions("family_max", hard, reliability, ensemble, families)
    rows = []
    family_array = np.asarray(families, dtype=object)
    for local, row_index in enumerate(target_rows):
        supported = np.flatnonzero(np.isfinite(label_scores[local]))
        competitors = [int(label) for label in supported if int(label) != int(base_pred[local])]
        if not competitors:
            # No gate record is possible here; the outer inference path keeps E13-B.
            continue
        baseline = int(base_pred[local])
        competitor = min(competitors, key=lambda label: (
            -label_scores[local, label], -ensemble[local, label], label))
        base_families = family_sets[baseline][local]
        comp_families = family_sets[competitor][local]
        base_rel = family_rel[baseline][local]
        comp_rel = family_rel[competitor][local]
        shared = base_families & comp_families
        base_only, comp_only = base_families - shared, comp_families - shared
        base_cp = np.flatnonzero(hard[:, local] == baseline)
        comp_cp = np.flatnonzero(hard[:, local] == competitor)
        base_probs = p[base_cp, row_index, baseline]
        comp_probs = p[comp_cp, row_index, competitor]
        base_score = label_scores[local, baseline]
        comp_score = label_scores[local, competitor]
        record = {
            "_row_index": int(row_index), "_baseline_index": baseline,
            "_competitor_index": competitor,
            "_baseline_prediction": LABELS[baseline],
            "_competitor_prediction": LABELS[competitor],
            "_target_switch_corrects": int(truth[row_index] == competitor),
            "_candidate_count": len(supported),
            "baseline_label": LABELS[baseline], "competitor_label": LABELS[competitor],
            "baseline_e13_score": float(base_score), "competitor_e13_score": float(comp_score),
            "e13_reliability_margin": float(base_score - comp_score),
            "baseline_probability": float(ensemble[local, baseline]),
            "competitor_probability": float(ensemble[local, competitor]),
            "probability_margin": float(ensemble[local, baseline] - ensemble[local, competitor]),
            "baseline_checkpoint_probability_max": float(base_probs.max()),
            "baseline_checkpoint_probability_mean": float(base_probs.mean()),
            "competitor_checkpoint_probability_max": float(comp_probs.max()),
            "competitor_checkpoint_probability_mean": float(comp_probs.mean()),
            "baseline_checkpoint_support": int(support_counts[local, baseline]),
            "competitor_checkpoint_support": int(support_counts[local, competitor]),
            "baseline_family_support": len(base_families),
            "competitor_family_support": len(comp_families),
            "baseline_to_competitor_confusion": float(direction[baseline, competitor]),
            "competitor_to_baseline_confusion": float(direction[competitor, baseline]),
            "historical_competitor_win_given_pair_support": float(pair_win[baseline, competitor]),
            "shared_conflicting_families": len(shared),
            "baseline_only_families": len(base_only),
            "competitor_only_families": len(comp_only),
            "family_support_margin": len(base_families) - len(comp_families),
            "checkpoint_probability_margin": float(base_probs.mean() - comp_probs.mean()),
        }
        rows.append(record)
    frame = pd.DataFrame(rows)
    if (frame._baseline_index == frame._competitor_index).any():
        raise ValueError("Gate training requires different baseline and competitor labels")
    values = frame.loc[:, [x for x in FEATURES if x not in CATEGORICAL]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Gate feature matrix has non-finite values")
    return frame


def _fit_e13(probabilities, ids, families, truth, groups, fit_rows, held_rows,
             baseline, row_ids):
    # E13's configured CatBoost reliability model is kept unchanged, including its
    # original 20-thread setting and roster/feature/selection behavior.
    return e13._fit_router(probabilities, ids, families, truth, groups,
                           np.asarray(fit_rows), np.asarray(held_rows),
                           E13_CHECKPOINTS, "family_max", baseline, row_ids)


def _crossfit_features(rows, matrices, ids, families, truth, groups, baseline, row_ids):
    frames, logs, seen = [], [], set()
    for fold_no, (fit_local, held_local) in enumerate(
            _validated_group_splits(rows, truth, groups, INNER_FOLDS), 1):
        fit_rows, held_rows = rows[fit_local], rows[held_local]
        if seen.intersection(map(int, held_rows)):
            raise ValueError("Gate OOF example was duplicated across folds")
        seen.update(map(int, held_rows))
        _, _, _, (reliability, hard, roster) = _fit_e13(
            matrices, ids, families, truth, groups, fit_rows, held_rows, baseline, row_ids)
        frame = _make_pair_frame(matrices, reliability, hard, roster, families,
                                 truth, fit_rows, held_rows)
        frames.append(frame)
        logs.append({"fold": fold_no, "fit_rows": len(fit_rows), "held_rows": len(held_rows),
                     "fit_prompt_groups": int(len(set(groups[fit_rows]))),
                     "held_prompt_groups": int(len(set(groups[held_rows]))),
                     "held_ids": row_ids[held_rows].astype(str).tolist(),
                     "roster": [ids[i] for i in roster],
                     "positive_switches": int(frame["_target_switch_corrects"].sum()),
                     "record_count": len(frame)})
    if seen != set(map(int, rows)):
        raise ValueError("Gate OOF folds did not cover fitting rows exactly once")
    return pd.concat(frames, ignore_index=True), logs


def _cat_frame(frame):
    x = frame.loc[:, FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = x[col].astype(str)
    return x


def _fit_gate(train_frame, stop_frame):
    x_train, x_stop = _cat_frame(train_frame), _cat_frame(stop_frame)
    y_train = train_frame["_target_switch_corrects"].to_numpy(dtype=np.int8)
    y_stop = stop_frame["_target_switch_corrects"].to_numpy(dtype=np.int8)
    early = CatBoostClassifier(**GATE_PARAMS)
    early.fit(x_train, y_train, cat_features=list(CATEGORICAL), eval_set=(x_stop, y_stop),
              early_stopping_rounds=GATE_PARAMS["od_wait"], verbose=False, use_best_model=True)
    iterations = max(1, int(early.get_best_iteration()) + 1)
    params = dict(GATE_PARAMS, iterations=iterations)
    params.pop("od_type", None)
    params.pop("od_wait", None)
    final = CatBoostClassifier(**params)
    final.fit(x_train, y_train, cat_features=list(CATEGORICAL), verbose=False)
    return final, iterations, early


def _predictions(model, frame):
    probability = model.predict_proba(_cat_frame(frame))[:, 1]
    base = frame["_baseline_index"].to_numpy(dtype=np.int64)
    competitor = frame["_competitor_index"].to_numpy(dtype=np.int64)
    switched = probability >= 0.5
    return np.where(switched, competitor, base), probability, switched


def _metrics(truth, pred, baseline, switch_probability, switched):
    cm = confusion_matrix(truth, pred, labels=np.arange(len(LABELS)))
    precision, recall, _, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "delta_vs_E13B_balanced_accuracy": float(balanced_accuracy_score(truth, pred) -
                                                    balanced_accuracy_score(truth, baseline)),
        "corrected_vs_E13B": int(((baseline != truth) & (pred == truth)).sum()),
        "introduced_vs_E13B": int(((baseline == truth) & (pred != truth)).sum()),
        "switch_count": int(np.count_nonzero(switched)),
        "mean_switch_probability": float(np.mean(switch_probability)),
        "confusion_matrix": cm.tolist(),
        "per_label": {LABELS[i]: {"support": int(support[i]), "precision": float(precision[i]),
                                   "recall": float(recall[i]), "tp": int(cm[i, i]),
                                   "fp": int(cm[:, i].sum() - cm[i, i]),
                                   "fn": int(cm[i].sum() - cm[i, i])}
                      for i in range(len(LABELS))},
    }


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    os.environ["OMP_NUM_THREADS"] = "24"
    os.environ["MKL_NUM_THREADS"] = "24"
    os.environ["OPENBLAS_NUM_THREADS"] = "24"
    os.environ["NUMEXPR_NUM_THREADS"] = "24"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output) if output else root / "artifacts/analysis/e14_gate" / stamp
    if output.exists():
        raise FileExistsError(output)
    v0 = catalog.cascades._v0(root)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    groups = v0.prompt_id.astype(str).to_numpy()
    row_ids = v0.id.astype(str).to_numpy()
    if len(v0) != 2699 or (truth < 0).any():
        raise ValueError("Expected 2,699 V0 rows with canonical labels")
    baseline, baseline_dir, baseline_result, baseline_details = _load_baseline(root, v0)
    _, members, prob_dict, excluded, trust = catalog._load_inputs(root, v0)
    if trust["version"] != "trust-list-2-v037":
        raise ValueError(f"This comparison requires the saved v037 E13-B champion; found {trust['version']}")
    ids = _sorted_ids(prob_dict)
    matrices = np.stack([prob_dict[sid] for sid in ids])
    families = [e13._model_family(members[sid]["model"]) for sid in ids]
    if matrices.shape != (len(ids), len(v0), len(LABELS)) or len(ids) < E13_CHECKPOINTS:
        raise ValueError("Eligible checkpoint probability shapes/count are invalid")
    output.mkdir(parents=True)
    (output / "models").mkdir()
    (output / "features").mkdir()
    e13.CATBOOST_PARAMS["thread_count"] = 20
    predictions = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    predictions["E13-B"] = np.asarray(LABELS, dtype=object)[baseline]
    predictions["E14-B-GATE"] = ""
    predictions["switch_probability"] = np.nan
    assign = np.zeros(len(v0), dtype=np.int8)
    fold_results = []
    fold_splits = _validated_group_splits(np.arange(len(v0)), truth, groups, 5)
    for fold, (fit, held) in enumerate(fold_splits, 1):
        if set(row_ids[held]) != set(baseline_details["folds"][fold - 1]["held_ids"]):
            raise ValueError("Gate folds do not match saved E13-B folds")
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Outer prompt groups overlap")
        early_split = _validated_group_splits(fit, truth, groups, 4)[-1]
        router_fit = fit[early_split[0]]
        stop_rows = fit[early_split[1]]
        early_train, early_logs = _crossfit_features(
            router_fit, matrices, ids, families, truth, groups, baseline, row_ids)
        stop_e13_pred, _, _, (stop_rel, stop_hard, stop_roster) = _fit_e13(
            matrices, ids, families, truth, groups, router_fit, stop_rows, baseline, row_ids)
        stop_frame = _make_pair_frame(matrices, stop_rel, stop_hard, stop_roster,
                                      families, truth, router_fit, stop_rows)
        stop_pred = pd.Categorical(stop_frame._baseline_prediction, categories=LABELS).codes
        stop_e13_codes = pd.Categorical(np.asarray(LABELS, dtype=object)[stop_e13_pred],
                                        categories=LABELS).codes
        stop_local = {int(row): i for i, row in enumerate(stop_rows)}
        stop_positions = np.asarray([stop_local[int(row)] for row in stop_frame["_row_index"]])
        if not np.array_equal(stop_pred, stop_e13_codes[stop_positions]):
            raise ValueError("E13-B early-stop pair labels differ from the fitted router")

        train_frame, train_logs = _crossfit_features(
            fit, matrices, ids, families, truth, groups, baseline, row_ids)
        held_e13_pred, _, _, (held_rel, held_hard, held_roster) = _fit_e13(
            matrices, ids, families, truth, groups, fit, held, baseline, row_ids)
        held_frame = _make_pair_frame(matrices, held_rel, held_hard, held_roster,
                                      families, truth, fit, held)
        held_pred_e13 = pd.Categorical(held_frame._baseline_prediction, categories=LABELS).codes
        held_record_rows = held_frame["_row_index"].to_numpy(dtype=np.int64)
        held_e13_codes = pd.Categorical(np.asarray(LABELS, dtype=object)[held_e13_pred],
                                        categories=LABELS).codes
        held_local = {int(row): i for i, row in enumerate(held)}
        held_positions = np.asarray([held_local[int(row)] for row in held_record_rows])
        if not np.array_equal(held_pred_e13, held_e13_codes[held_positions]):
            raise ValueError("E13-B outer pair labels differ from the fitted router")
        if not np.array_equal(held_e13_codes, baseline[held]):
            raise ValueError("E13-B outer predictions differ from the saved champion")

        train_frame.to_parquet(output / "features" / f"fold{fold}_train.parquet", index=False)
        stop_frame.to_parquet(output / "features" / f"fold{fold}_stop.parquet", index=False)
        held_frame.to_parquet(output / "features" / f"fold{fold}_held.parquet", index=False)
        gate, best_iterations, early = _fit_gate(train_frame, stop_frame)
        model_path = output / "models" / f"fold{fold}.cbm"
        gate.save_model(str(model_path))
        p_switch = np.zeros(len(held), dtype=np.float64)
        switched = np.zeros(len(held), dtype=bool)
        pred = baseline[held].copy()
        active_local = np.searchsorted(held, held_record_rows)
        p_active = gate.predict_proba(_cat_frame(held_frame))[:, 1]
        active_switched = p_active >= 0.5
        p_switch[active_local] = p_active
        switched[active_local] = active_switched
        challenger = held_frame["_competitor_index"].to_numpy(dtype=np.int64)
        pred[active_local] = np.where(active_switched, challenger,
                                      held_frame["_baseline_index"].to_numpy(dtype=np.int64))
        predictions.loc[held, "E14-B-GATE"] = np.asarray(LABELS, dtype=object)[pred]
        predictions.loc[held, "switch_probability"] = p_switch
        assign[held] += 1
        m = _metrics(truth[held], pred, baseline[held], p_switch, switched)
        m.update({"fold": fold, "rows": len(held), "best_iterations": best_iterations,
                  "gate_eligible_rows": len(held_frame),
                  "no_distinct_competitor_rows": len(held) - len(held_frame),
                  "early_stopping_best_iteration": int(early.get_best_iteration()),
                  "training_records": len(train_frame), "stop_records": len(stop_frame),
                  "positive_training_targets": int(train_frame._target_switch_corrects.sum()),
                  "positive_stop_targets": int(stop_frame._target_switch_corrects.sum()),
                  "switched_to_correct_competitor": int(np.count_nonzero(switched & (pred == truth[held]))),
                  "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                  "fit_ids": row_ids[fit].tolist(), "held_ids": row_ids[held].tolist(),
                  "outer_roster": [ids[i] for i in held_roster],
                  "gate_training_crossfit": train_logs, "early_fit_crossfit": early_logs})
        fold_results.append(m)
        print(f"E14-B-GATE fold {fold}/5: acc={m['accuracy']:.4%} ba={m['balanced_accuracy']:.4%} "
              f"switches={m['switch_count']} iterations={best_iterations}", flush=True)
    if not np.all(assign == 1) or (predictions["E14-B-GATE"] == "").any():
        raise ValueError("Gate predictions do not cover all V0 rows exactly once")
    pred_all = pd.Categorical(predictions["E14-B-GATE"], categories=LABELS).codes
    if (pred_all < 0).any():
        raise ValueError("Gate produced a noncanonical label")
    overall = _metrics(truth, pred_all, baseline,
                       predictions.switch_probability.to_numpy(dtype=float),
                       predictions.E14_B_SWITCHED.to_numpy() if "E14_B_SWITCHED" in predictions else
                       pred_all != baseline)
    overall["folds"] = fold_results
    overall["experiment"] = "E14-B-GATE"
    overall["baseline_accuracy"] = float(accuracy_score(truth, baseline))
    overall["baseline_balanced_accuracy"] = float(balanced_accuracy_score(truth, baseline))
    overall["baseline_kaggle_public_score"] = 0.606
    overall["trust_list"] = trust
    overall["baseline_e13_oof_dir"] = str(baseline_dir)
    overall["rows"] = len(v0)
    overall["labels"] = list(LABELS)
    overall["row_ids"] = row_ids.tolist()
    overall["training_threads"] = 24
    overall["catboost"] = GATE_PARAMS
    overall["feature_columns"] = list(FEATURES)
    overall["switch_threshold"] = 0.5
    overall["smoothing"] = SMOOTHING
    overall["e13_checkpoints_per_fold"] = E13_CHECKPOINTS
    overall["gate_eligible_rows"] = int(sum(f["gate_eligible_rows"] for f in fold_results))
    overall["no_distinct_competitor_rows"] = int(sum(f["no_distinct_competitor_rows"] for f in fold_results))
    overall["excluded_checkpoints_from_catalog"] = excluded
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "results.json").write_text(json.dumps(overall, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps({
        "method": "fold-local CatBoost intervention gate; switch to E13-B strongest supported rival iff P(switch correct) >= 0.5",
        "trust_list": trust, "baseline_e13_oof_dir": str(baseline_dir),
        "outer_folds": 5, "inner_crossfit_folds": INNER_FOLDS,
        "seed": SEED, "threads": 24, "e13_router_catboost_threads": 20,
        "e13_router": "src/e13_reliability_router.py, E13-B family_max, 15 checkpoints",
        "catboost_gate": GATE_PARAMS, "features": list(FEATURES),
        "target": "1 iff switching from E13-B to strongest E13-B-scored supported competitor yields true label",
        "comparison": "saved E13-B v037 OOF predictions on the identical five outer prompt-group folds",
    }, indent=2) + "\n")
    _write_report(root, output, overall, predictions)
    return overall


def _write_report(root, output, result, predictions):
    lines = ["# E14-B Focused Intervention Gate", "",
             f"Trust List: **{result['trust_list']['version']}** · V0 rows: **{result['rows']}** · "
             "five prompt-group outer folds.", "",
             "E13-B remains the champion and is not modified. For each example, the gate evaluates switching "
             "from E13-B's family-max reliability label to its strongest differently labelled supported "
             "competitor. A CatBoost gate switches at a fixed probability threshold of 0.5.", "",
             f"E13-B baseline: **{result['baseline_accuracy']:.4%} accuracy**, "
             f"**{result['baseline_balanced_accuracy']:.4%} balanced accuracy**; Kaggle public score **0.606** "
             "is reported separately.", "",
             "## Result", "",
             "| Method | Accuracy | Balanced accuracy | Δ balanced accuracy (pp) | Corrected | Introduced | Switched |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| E13-B | {result['baseline_accuracy']:.4%} | {result['baseline_balanced_accuracy']:.4%} | — | — | — | 0 |")
    lines.append(f"| E14-B-GATE | {result['accuracy']:.4%} | {result['balanced_accuracy']:.4%} | "
                  f"{result['delta_vs_E13B_balanced_accuracy'] * 100:+.4f} pp | {result['corrected_vs_E13B']} | "
                  f"{result['introduced_vs_E13B']} | {result['switch_count']} |")
    lines.append(f"\nThe gate had a distinct supported competitor on {result['gate_eligible_rows']} rows; "
                 f"on {result['no_distinct_competitor_rows']} rows it kept E13-B unchanged.")
    lines += ["", "## Fold results", "",
              "| Fold | Rows | Gate eligible | No competitor | Accuracy | Balanced accuracy | Corrected | Introduced | Switches | Trees |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in result["folds"]:
        lines.append(f"| {f['fold']} | {f['rows']} | {f['gate_eligible_rows']} | "
                     f"{f['no_distinct_competitor_rows']} | {f['accuracy']:.4%} | {f['balanced_accuracy']:.4%} | "
                     f"{f['corrected_vs_E13B']} | {f['introduced_vs_E13B']} | {f['switch_count']} | {f['best_iterations']} |")
    lines += ["", "## Per-label precision and recall", "",
              "| Label | Support | Precision | Recall |", "|---|---:|---:|---:|"]
    for label in LABELS:
        m = result["per_label"][label]
        lines.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} |")
    lines += ["", "## Confusion matrix", "", "```text", "\t" + "\t".join(LABELS)]
    for label, row in zip(LABELS, result["confusion_matrix"]):
        lines.append(label + "\t" + "\t".join(map(str, row)))
    lines += ["```", "", "Features include E13-B label scores and reliability margins, family-balanced "
              "probability margins, checkpoint probability margins, directional family-vote confusion, "
              "smoothed pairwise historical wins, and family overlap/support differences.", "",
              f"Artifacts: `{output}` (fold models, cross-fitted features, predictions, configuration and results).", ""]
    report = "\n".join(lines)
    (output / "e14_gate.md").write_text(report)
    (root / "docs/e14_gate.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
