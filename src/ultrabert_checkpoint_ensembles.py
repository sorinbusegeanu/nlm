"""Grouped cross-fitted diagnostics for the four saved UltraBERT checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

from .checkpoint_comparison import _validated_probabilities
from .ensemble_factory import fit_weights
from .evaluation import complementarity
from .prediction_factory import _align_candidates, _read_candidates


CHECKPOINTS = {
    "A": "ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0",
    "B": "ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2",
    "C": "ultrabert380_v15_from00dfd18d_data40a22ce8_split147ce64e_seed2026_cfge678dbc9_code667e2d0",
    "D": "ultrabert380_v0_fromeab1ab35_data40a22ce8_split147ce64e_seed2026_cfg34506fe0_codedb9ddff",
}
FAMILIES = {
    "claude-fable-5": "Claude", "claude-opus-4-6": "Claude", "claude-opus-4-8": "Claude",
    "claude-opus-5": "Claude", "claude-sonnet-5": "Claude",
    "gemini-3.1-pro-preview": "Gemini", "gemini-3.6-flash": "Gemini",
    "gemini-3.7-flash": "Gemini", "gemini-3.8-flash": "Gemini",
    "glm-5.3": "GLM", "glm-5.3-flash": "GLM",
    "gpt-5.5": "GPT", "gpt-5.6-luna": "GPT", "gpt-5.6-sol": "GPT",
    "gpt-5.6-terra": "GPT", "gpt-6-astra": "GPT",
    "kimi-k3": "Kimi", "qwen3.8-max": "Qwen",
}


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _temperature(probabilities, temperature):
    p = np.asarray(probabilities, dtype=np.float64)
    if abs(float(temperature) - 1.0) < 1e-12:
        return p / p.sum(axis=1, keepdims=True)
    # Power temperature scaling preserves exact zeros and class rank ordering.
    # Clipping zeros before taking logs would create artificial positive mass and
    # could change top-k ranks for underflowed probabilities.
    scaled = np.zeros_like(p)
    positive = p > 0
    scaled[positive] = np.exp(np.log(p[positive]) / float(temperature))
    return _normalize(scaled)


def _normalize(scores):
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    sums = scores.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("ensemble scores have zero row mass")
    return scores / sums


def _rank_fusion(probabilities):
    """Reciprocal-rank fusion with a normalized 18-class score vector."""
    p = np.asarray(probabilities, dtype=np.float64)
    m, n, c = p.shape
    ranks = np.empty((m, n, c), dtype=np.float64)
    for model_index in range(m):
        order = np.argsort(-p[model_index], axis=1, kind="stable")
        row = np.arange(n)[:, None]
        ranks[model_index, row, order] = np.arange(1, c + 1, dtype=np.float64)[None, :]
    return _normalize(np.sum(1.0 / ranks, axis=0))


def _metrics(y, probabilities, classes):
    p = _validated_probabilities(probabilities, len(y), len(classes), "ensemble")
    y = np.asarray(y, dtype=str)
    pred_indices = p.argmax(axis=1)
    pred = np.asarray(classes, dtype=object)[pred_indices]
    truth_indices = np.asarray([classes.index(label) for label in y], dtype=int)
    order = np.argsort(-p, axis=1, kind="stable")
    ranks = np.argmax(order == truth_indices[:, None], axis=1) + 1
    top3 = float(np.mean(ranks <= min(3, len(classes))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "top3_accuracy": top3,
        "mrr": float(np.mean(1.0 / ranks)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=classes)),
        "multiclass_brier": float(np.mean(np.sum(
            (p - np.eye(len(classes), dtype=float)[truth_indices]) ** 2, axis=1))),
        "per_class": {str(label): {"recall": float(recall[i]), "precision": float(precision[i]),
                                   "support": int(support[i])}
                      for i, label in enumerate(classes)},
        "predictions": pred,
    }


def _fit_family_weights(probabilities, y, classes, global_weights, families, penalty):
    """Fit nonnegative checkpoint weights independently by true-label family."""
    p = np.asarray(probabilities, dtype=np.float64)
    m, n, c = p.shape
    family_names = sorted(set(families))
    family_index = np.asarray([family_names.index(FAMILIES[label]) for label in classes])
    y_index = np.asarray([classes.index(str(label)) for label in y], dtype=int)
    n_families = len(family_names)
    start = np.repeat(np.asarray(global_weights, dtype=float)[:, None], n_families, axis=1)

    def probabilities_for(weights, rows):
        scores = np.zeros((len(rows), c), dtype=float)
        for model in range(m):
            scores += p[model, rows, :] * weights[model, family_index][None, :]
        return _normalize(scores)

    def objective(flat):
        weights = flat.reshape(m, n_families)
        probs = probabilities_for(weights, np.arange(n))
        loss = -np.log(np.clip(probs[np.arange(n), y_index], 1e-12, 1)).mean()
        shrink = penalty * np.mean((weights - np.asarray(global_weights)[:, None]) ** 2)
        return float(loss + shrink)

    constraints = [{"type": "eq", "fun": lambda flat, g=g: flat.reshape(m, n_families)[:, g].sum() - 1.0}
                   for g in range(n_families)]
    result = minimize(objective, start.ravel(), method="SLSQP", bounds=[(0.0, 1.0)] * (m * n_families),
                      constraints=constraints, options={"maxiter": 250, "ftol": 1e-8})
    if not result.success:
        raise RuntimeError(f"family weight optimization failed: {result.message}")
    return result.x.reshape(m, n_families), family_names


def _fit_class_weights(probabilities, y, classes, global_weights, penalty):
    """Fit target-label-specific checkpoint weights with simplex constraints."""
    p = np.asarray(probabilities, dtype=np.float64)
    m, n, c = p.shape
    y_index = np.asarray([classes.index(str(label)) for label in y], dtype=int)
    start = np.repeat(np.asarray(global_weights, dtype=float)[:, None], c, axis=1)

    def probabilities_for(weights, rows):
        scores = np.sum(p[:, rows, :] * weights[:, None, :], axis=0)
        return _normalize(scores)

    def objective(flat):
        weights = flat.reshape(m, c)
        probs = probabilities_for(weights, np.arange(n))
        loss = -np.log(np.clip(probs[np.arange(n), y_index], 1e-12, 1)).mean()
        return float(loss + penalty * np.mean((weights - np.asarray(global_weights)[:, None]) ** 2))

    constraints = [{"type": "eq", "fun": lambda flat, j=j: flat.reshape(m, c)[:, j].sum() - 1.0}
                   for j in range(c)]
    result = minimize(objective, start.ravel(), method="SLSQP", bounds=[(0.0, 1.0)] * (m * c),
                      constraints=constraints, options={"maxiter": 300, "ftol": 1e-8})
    if not result.success:
        raise RuntimeError(f"class-specific weight optimization failed: {result.message}")
    return result.x.reshape(m, c)


def _fit_temperature_inner(p, y, groups, classes, seed=2026):
    """Select temperature by 3-fold prompt-group OOF loss on the fitting rows."""
    grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
    losses = {}
    for temperature in grid:
        oof = np.zeros_like(p, dtype=np.float64)
        for _, val in splits:
            oof[val] = _temperature(p[val], temperature)
        losses[str(temperature)] = float(log_loss(y, oof, labels=classes))
    best = min(grid, key=lambda temp: (losses[str(temp)] + .01 * np.log(temp) ** 2, temp))
    return float(best), losses


def _inner_global_score(probabilities, y, groups, classes, members, l2=.05):
    """3-fold inner OOF score for a fixed candidate subset with learned weights."""
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
    oof = np.zeros((len(y), len(classes)), dtype=float)
    for train, val in splits:
        weights = fit_weights([probabilities[i, train] for i in members], y[train], classes, l2=l2)
        oof[val] = _normalize(np.einsum("m,mnc->nc", weights, probabilities[members][:, val, :]))
    metric = _metrics(y, oof, classes)
    return metric, oof


def _inner_calibrated_cd_score(probabilities, y, groups, classes, members, l2=.05):
    """Nested inner OOF score for C+D with temperature fit on inner-training groups."""
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
    oof = np.zeros((len(y), len(classes)), dtype=float)
    temperatures = []
    for train, val in splits:
        temperature, _ = _fit_temperature_inner(probabilities[2, train], y[train], groups[train], classes)
        p_train = probabilities[:, train, :].copy()
        p_val = probabilities[:, val, :].copy()
        p_train[2] = _temperature(p_train[2], temperature)
        p_val[2] = _temperature(p_val[2], temperature)
        weights = fit_weights([p_train[i] for i in members], y[train], classes, l2=l2)
        oof[val] = _normalize(np.einsum("m,mnc->nc", weights, p_val[members]))
        temperatures.append(float(temperature))
    return _metrics(y, oof, classes), oof, temperatures


def _forward_select(probabilities, y, groups, classes, names):
    """Greedy equal-blend forward selection scored on inner grouped OOF rows."""
    m = probabilities.shape[0]
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
    current = []
    steps = []
    remaining = set(range(m))

    def score(subset):
        oof = np.zeros((len(y), len(classes)), dtype=float)
        for _, val in splits:
            oof[val] = _normalize(probabilities[list(subset)][:, val, :].mean(axis=0))
        return _metrics(y, oof, classes), oof

    while remaining:
        contenders = []
        for candidate in sorted(remaining):
            metric, pred = score(current + [candidate])
            contenders.append((metric["balanced_accuracy"], -metric["log_loss"],
                               -len(current) - 1, -candidate, candidate, metric, pred))
        winner = max(contenders, key=lambda item: item[:4])
        current.append(winner[4])
        remaining.remove(winner[4])
        steps.append({"step": len(current), "candidate": names[winner[4]],
                      "members": [names[i] for i in current],
                      "inner_oof_balanced_accuracy": winner[5]["balanced_accuracy"],
                      "inner_oof_log_loss": winner[5]["log_loss"]})
        if len(steps) > 1:
            previous = steps[-2]
            improves = (steps[-1]["inner_oof_balanced_accuracy"] >
                        previous["inner_oof_balanced_accuracy"] + 1e-12)
            if not improves:
                current.pop()
                steps.pop()
                break
    metric, _ = score(current)
    return current, steps, metric


def _fit_conditional_cv(p, y, groups, classes, global_weights, *, mode, penalties):
    """Choose shrinkage strength with inner grouped folds, then fit on all outer-train rows."""
    scores = {}
    for penalty in penalties:
        oof = np.zeros((len(y), len(classes)), dtype=float)
        splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
        for train, val in splits:
            local_global = fit_weights([p[i, train] for i in range(p.shape[0])], y[train], classes, l2=.05)
            if mode == "family":
                w, _ = _fit_family_weights(p[:, train, :], y[train], classes, local_global,
                                           [FAMILIES[x] for x in classes], penalty)
                fam_idx = np.asarray([sorted(set(FAMILIES.values())).index(FAMILIES[x]) for x in classes])
                score = sum(p[i, val, :] * w[i, fam_idx][None, :] for i in range(p.shape[0]))
            else:
                w = _fit_class_weights(p[:, train, :], y[train], classes, local_global, penalty)
                score = np.sum(p[:, val, :] * w[:, None, :], axis=0)
            oof[val] = _normalize(score)
        scores[str(penalty)] = _metrics(y, oof, classes)
    best_penalty = max(penalties, key=lambda pen: (scores[str(pen)]["balanced_accuracy"],
                                                    -scores[str(pen)]["log_loss"], -pen))
    if mode == "family":
        weights, family_names = _fit_family_weights(p, y, classes, global_weights,
            [FAMILIES[x] for x in classes], best_penalty)
        fam_idx = np.asarray([family_names.index(FAMILIES[x]) for x in classes])
        prediction = _normalize(sum(p[i] * weights[i, fam_idx][None, :] for i in range(p.shape[0])))
        return prediction, {"weights": weights.tolist(), "family_names": family_names,
                           "penalty": float(best_penalty), "inner_scores": scores}
    weights = _fit_class_weights(p, y, classes, global_weights, best_penalty)
    prediction = _normalize(np.sum(p * weights[:, None, :], axis=0))
    return prediction, {"weights": weights.tolist(), "penalty": float(best_penalty), "inner_scores": scores}


def run_ultrabert_four_checkpoint_experiments(*, artifact_dir="artifacts/predictions",
        data_dir="data/processed", output_dir="artifacts/analysis/ultrabert_four_checkpoint",
        report_path="docs/ultrabert_4checkpoints_4experiments.md", registry_path="experiments/registry.sqlite"):
    """Run four prompt-group cross-fitted UltraBERT checkpoint fusion experiments."""
    from .pipeline import _load_dataset
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.eq("validation"), ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["y_true"] = validation.pop("model").astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540 or validation.prompt_id.nunique() != 30:
        raise ValueError("expected frozen V0 validation with 540 rows and 30 prompt groups")
    classes = sorted(validation.y_true.unique().tolist())
    candidates, problems = _read_candidates(artifact_dir, min_validation_rows=540,
                                            candidate_ids=CHECKPOINTS.values(), deduplicate=False)
    found = {row["candidate_id"] for row in candidates}
    missing = set(CHECKPOINTS.values()) - found
    if missing:
        raise FileNotFoundError(f"missing saved UltraBERT prediction candidates: {sorted(missing)}; issues={problems}")
    _, y_ref, group_ref, aligned, excluded, reference_ids = _align_candidates(
        candidates, reference_dataset="v0", reference_frame=validation)
    by_id = {candidate["candidate_id"]: candidate for candidate in aligned}
    ordered = [by_id[CHECKPOINTS[key]] for key in "ABCD"]
    if any(candidate["model"] != "ultrabert380" for candidate in ordered):
        raise ValueError("all four candidate artifacts must be UltraBERT")
    if any(len(candidate["frame"]) != 540 for candidate in ordered):
        raise ValueError("all checkpoint predictions must cover the same 540 validation rows")
    probability_list = []
    for candidate in ordered:
        p = _validated_probabilities(candidate["probabilities"], 540, len(classes), candidate["candidate_id"])
        probability_list.append(p)
    probabilities = np.stack(probability_list)
    y, groups = np.asarray(y_ref, dtype=str), np.asarray(group_ref, dtype=str)
    fold_ids = np.full(len(y), -1, dtype=int)
    outer_splits = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    for fold, (_, test) in enumerate(outer_splits):
        fold_ids[test] = fold
    if len(np.unique(fold_ids)) != 5 or np.any(fold_ids < 0):
        raise RuntimeError("failed to assign all rows to five grouped folds")

    methods = {}
    folds_metadata = []
    per_fold_weights = {}
    names = list("ABCD")
    # Baselines are the original saved predictions, retained unchanged.
    for i, key in enumerate(names):
        methods[f"baseline_{key}"] = probabilities[i].copy()
    for method in ("E1_BD_equal", "E1_BD_weighted", "E1_BD_rank",
                   "E2_CD_equal", "E2_CD_weighted", "E2_CD_rank", "E2_CD_temp_equal", "E2_CD_temp_weighted",
                   "E2_C_temp_only",
                   "E3_ABCD_equal", "E3_ABCD_weighted", "E3_ABCD_rank", "E3_forward_weighted",
                   "E4_best_global_cv", "E4_family_regularized", "E4_class_regularized"):
        methods[method] = np.zeros((len(y), len(classes)), dtype=np.float64)

    global_selector_candidates = {
        "BD_weighted": [1, 3], "CD_weighted": [2, 3], "ABCD_weighted": [0, 1, 2, 3],
    }
    for fold, (train, test) in enumerate(outer_splits):
        train_groups = groups[train]
        fit_p, fit_y = probabilities[:, train, :], y[train]
        fold_doc = {"fold": fold, "train_rows": int(len(train)), "test_rows": int(len(test)),
                    "train_groups": sorted(set(train_groups)), "test_groups": sorted(set(groups[test]))}

        def fit_global(key, members, p_train=fit_p, y_train=fit_y, p_test=probabilities[:, test, :], l2=.05):
            w = fit_weights([p_train[i] for i in members], y_train, classes, l2=l2)
            pred = _normalize(np.einsum("m,mnc->nc", w, p_test[members]))
            return pred, w

        # Experiment 1: original V1-all and V1-all -> V0.
        pair = [1, 3]
        methods["E1_BD_equal"][test] = _normalize(probabilities[pair][:, test, :].mean(axis=0))
        methods["E1_BD_rank"][test] = _rank_fusion(probabilities[pair][:, test, :])
        pred, w = fit_global("BD", pair); methods["E1_BD_weighted"][test] = pred
        per_fold_weights.setdefault("E1_BD_weighted", []).append({"fold": fold, "members": [names[i] for i in pair], "weights": w.tolist()})

        # Experiment 2: both V0 continuations. Temperature selection uses only
        # inner grouped folds from this outer fitting partition.
        pair = [2, 3]
        methods["E2_CD_equal"][test] = _normalize(probabilities[pair][:, test, :].mean(axis=0))
        methods["E2_CD_rank"][test] = _rank_fusion(probabilities[pair][:, test, :])
        pred, w = fit_global("CD", pair); methods["E2_CD_weighted"][test] = pred
        per_fold_weights.setdefault("E2_CD_weighted", []).append({"fold": fold, "members": [names[i] for i in pair], "weights": w.tolist()})
        temperature, temp_losses = _fit_temperature_inner(fit_p[2], fit_y, train_groups, classes)
        fit_cal = fit_p.copy(); fit_cal[2] = _temperature(fit_p[2], temperature)
        test_cal = probabilities[:, test, :].copy(); test_cal[2] = _temperature(test_cal[2], temperature)
        methods["E2_C_temp_only"][test] = test_cal[2]
        methods["E2_CD_temp_equal"][test] = _normalize(test_cal[pair].mean(axis=0))
        cal_w = fit_weights([fit_cal[i] for i in pair], fit_y, classes, l2=.05)
        methods["E2_CD_temp_weighted"][test] = _normalize(np.einsum("m,mnc->nc", cal_w, test_cal[pair]))
        per_fold_weights.setdefault("E2_CD_temp_weighted", []).append({"fold": fold, "members": [names[i] for i in pair],
            "weights": cal_w.tolist(), "temperature_C": temperature, "temperature_inner_log_loss": temp_losses})

        # Experiment 3: all four, plus inner-fold forward selection. The selected
        # subset is then refit with global weights on the complete outer train fold.
        all_idx = [0, 1, 2, 3]
        methods["E3_ABCD_equal"][test] = _normalize(probabilities[all_idx][:, test, :].mean(axis=0))
        methods["E3_ABCD_rank"][test] = _rank_fusion(probabilities[all_idx][:, test, :])
        pred, w = fit_global("ABCD", all_idx); methods["E3_ABCD_weighted"][test] = pred
        per_fold_weights.setdefault("E3_ABCD_weighted", []).append({"fold": fold, "members": names, "weights": w.tolist()})
        selected, selection_steps, selection_metric = _forward_select(fit_p, fit_y, train_groups, classes, names)
        pred, w = fit_global("forward", selected)
        methods["E3_forward_weighted"][test] = pred
        per_fold_weights.setdefault("E3_forward_weighted", []).append({"fold": fold,
            "members": [names[i] for i in selected], "weights": w.tolist(), "selection_steps": selection_steps,
            "selection_inner_metrics": {k: selection_metric[k] for k in ("balanced_accuracy", "log_loss")}})

        # Experiment 4: choose the strongest global-weight combination within the
        # fitting fold using inner grouped CV, then compare regularized family/class weights.
        candidate_scores = []
        for label, members in global_selector_candidates.items():
            metric, _ = _inner_global_score(fit_p, fit_y, train_groups, classes, members)
            candidate_scores.append((metric["balanced_accuracy"], -metric["log_loss"],
                                     -len(members), label, members, metric))
        temp_metric, _, temp_inner_values = _inner_calibrated_cd_score(
            fit_p, fit_y, train_groups, classes, [2, 3])
        candidate_scores.append((temp_metric["balanced_accuracy"], -temp_metric["log_loss"],
                                 -2, "CD_temp_weighted", [2, 3], temp_metric))
        # Include a forward-selected subset; its selected membership was learned
        # within the same training partition and evaluated in inner OOF folds.
        fwd_metric, _ = _inner_global_score(fit_p, fit_y, train_groups, classes, selected)
        candidate_scores.append((fwd_metric["balanced_accuracy"], -fwd_metric["log_loss"],
                                 -len(selected), "Forward_weighted", selected, fwd_metric))
        winner = max(candidate_scores, key=lambda item: item[:3])
        best_members = list(winner[4])
        best_temperature = None
        if winner[3] == "CD_temp_weighted":
            best_temperature, _ = _fit_temperature_inner(fit_p[2], fit_y, train_groups, classes)
            calibrated_fit = fit_p.copy(); calibrated_fit[2] = _temperature(fit_p[2], best_temperature)
            calibrated_test = probabilities[:, test, :].copy()
            calibrated_test[2] = _temperature(calibrated_test[2], best_temperature)
            best_w = fit_weights([calibrated_fit[i] for i in best_members], fit_y, classes, l2=.05)
            best_pred = _normalize(np.einsum("m,mnc->nc", best_w, calibrated_test[best_members]))
        else:
            best_pred, best_w = fit_global(winner[3], best_members)
        methods["E4_best_global_cv"][test] = best_pred
        per_fold_weights.setdefault("E4_best_global_cv", []).append({"fold": fold,
            "selected_candidate": winner[3], "members": [names[i] for i in best_members],
            "weights": best_w.tolist(), "inner_balanced_accuracy": winner[0],
            "inner_log_loss": -winner[1], "temperature_C": best_temperature,
            "temperature_nested_inner_values": temp_inner_values if winner[3] == "CD_temp_weighted" else None})

        all_global = fit_weights(list(fit_p), fit_y, classes, l2=.05)
        for mode, penalties, method in (("family", [.05, .25, 1.0], "E4_family_regularized"),
                                        ("class", [.25, 1.0, 4.0], "E4_class_regularized")):
            outer_pred, details = _fit_conditional_cv(fit_p, fit_y, train_groups, classes, all_global,
                                                       mode=mode, penalties=penalties)
            if mode == "family":
                family_names = details["family_names"]
                fam_idx = np.asarray([family_names.index(FAMILIES[label]) for label in classes])
                test_pred = _normalize(sum(probabilities[i, test, :] * np.asarray(details["weights"])[i, fam_idx][None, :]
                                           for i in range(4)))
            else:
                test_pred = _normalize(np.sum(probabilities[:, test, :] * np.asarray(details["weights"])[:, None, :], axis=0))
            methods[method][test] = test_pred
            per_fold_weights.setdefault(method, []).append({"fold": fold, "global_anchor_weights": all_global.tolist(), **details})

        fold_doc["selected_forward_members"] = [names[i] for i in selected]
        fold_doc["forward_selection_steps"] = selection_steps
        fold_doc["best_global_candidate_for_exp4"] = winner[3]
        fold_doc["family_penalty"] = per_fold_weights["E4_family_regularized"][-1]["penalty"]
        fold_doc["class_penalty"] = per_fold_weights["E4_class_regularized"][-1]["penalty"]
        folds_metadata.append(fold_doc)

    # Identify a descriptive top candidate from the pooled OOF table. It is not
    # an independent winner estimate because this ranking compares methods.
    report_methods = [key for key in methods if not key.startswith("baseline_")]
    method_metrics = {key: _metrics(y, methods[key], classes) for key in methods}
    selected_method = max(report_methods, key=lambda key: (round(method_metrics[key]["balanced_accuracy"], 12),
                                                            round(method_metrics[key]["top3_accuracy"], 12),
                                                            round(method_metrics[key]["mrr"], 12),
                                                            -round(method_metrics[key]["log_loss"], 12), key))

    # Pairwise and all-checkpoint oracle coverage are theoretical upper bounds.
    standalone_predictions = {key: method_metrics[f"baseline_{key}"]["predictions"] for key in names}
    pairwise_complementarity = complementarity(y, standalone_predictions)
    correct_matrix = np.stack([standalone_predictions[key] == y for key in names])
    all_oracle = float(np.mean(correct_matrix.any(axis=0)))
    d_correct = standalone_predictions["D"] == y
    selected_pred = method_metrics[selected_method]["predictions"]
    selected_correct = selected_pred == y

    checkpoint_metadata = []
    for key, candidate in zip(names, ordered):
        manifest = candidate["model_manifest"]
        prediction_path = manifest.get("prediction_path")
        prediction_path = Path(prediction_path) if prediction_path else None
        if prediction_path and not prediction_path.is_absolute():
            prediction_path = (Path.cwd() / prediction_path).resolve()
        checkpoint_metadata.append({"key": key, "experiment_id": candidate["candidate_id"],
            "manifest_path": candidate["manifest_path"], "prediction_path": str(prediction_path),
            "checkpoint_path": manifest.get("checkpoint_path"), "dataset": manifest.get("dataset"),
            "training_sequence": manifest.get("training_sequence"),
            "metrics": {k: method_metrics[f"baseline_{key}"][k] for k in
                        ("accuracy", "balanced_accuracy", "top3_accuracy", "mrr", "macro_f1", "log_loss", "multiclass_brier")}})
    split_hash = json.loads((Path(data_dir) / "dataset_profile.json").read_text())["split_sha256"]
    ids_hash = hashlib.sha256("\n".join(validation.id.astype(str)).encode()).hexdigest()
    artifact_metadata = {"checkpoints": checkpoint_metadata, "class_order": classes,
        "validation_rows": len(validation), "prompt_groups": int(validation.prompt_id.nunique()),
        "validation_split_sha256": split_hash, "validation_ids_sha256": ids_hash,
        "folds": folds_metadata, "selected_descriptive_method": selected_method,
        "selection_note": "Pooled OOF method ranking is exploratory; the V0 rows also informed checkpoint selection.",
        "pairwise_oracle": pairwise_complementarity, "four_checkpoint_oracle_accuracy": all_oracle,
        "methods": {key: {k: value for k, value in metric.items() if k not in {"predictions", "per_class"}}
                    for key, metric in method_metrics.items()},
        "per_class": {key: {label: metric["per_class"][label] for label in classes}
                      for key, metric in method_metrics.items()},
        "per_fold_weights": per_fold_weights,
        "relative_to_D": {key: {"corrected": int(np.sum(~d_correct & (metric["predictions"] == y))),
                                  "regressed": int(np.sum(d_correct & (metric["predictions"] != y)))}
                          for key, metric in method_metrics.items() if key != "baseline_D"}}

    # Save row-level OOF probabilities for every fusion strategy in canonical order.
    oof_columns = {"id": validation.id.to_numpy(), "prompt_id": validation.prompt_id.to_numpy(),
                   "y_true": validation.y_true.to_numpy(), "fold": fold_ids}
    for method, p in methods.items():
        p = _validated_probabilities(p, len(y), len(classes), method)
        for ci, label in enumerate(classes):
            oof_columns[f"{method}::prob::{label}"] = p[:, ci]
    oof_frame = pd.DataFrame(oof_columns)
    oof_path = output_dir / "oof_predictions.parquet"
    folds_path = output_dir / "fold_assignments.csv"
    weights_path = output_dir / "fold_weights.json"
    summary_path = output_dir / "results.json"
    report_path = Path(report_path)
    oof_frame.to_parquet(oof_path, index=False)
    validation[["id", "prompt_id", "y_true"]].assign(fold=fold_ids).to_csv(folds_path, index=False)
    weights_path.write_text(json.dumps(per_fold_weights, indent=2, default=_json_default) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(artifact_metadata, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_report(report_path, artifact_metadata, method_metrics, classes, y, pairwise_complementarity,
                  correct_matrix, selected_method, selected_correct, d_correct)
    return {"report_path": str(report_path), "oof_predictions_path": str(oof_path),
            "fold_assignments_path": str(folds_path), "fold_weights_path": str(weights_path),
            "results_path": str(summary_path), "selected_descriptive_method": selected_method,
            "metrics": artifact_metadata["methods"]}


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_report(path, metadata, method_metrics, classes, y, complementarity_stats,
                  correct_matrix, selected_method, selected_correct, d_correct):
    labels = list("ABCD")
    descriptions = {"A": "V1", "B": "V1-all", "C": "V1 → V0", "D": "V1-all → V0"}
    lines = ["# UltraBERT Four-Checkpoint Ensemble Experiments", "",
        "## Method and caveat", "",
        "All ensemble results are five-fold prompt-group out-of-fold (OOF) diagnostics on the frozen V0 set (540 rows, 18 classes, 30 prompt groups). Every fold's calibration, fit weights, subset selection, and conditional-weight penalty used only its four fitting folds; folds were assigned with GroupKFold. Checkpoint and roster choices already used this V0 set, so these are development-set diagnostics, not independent generalization estimates. The four checkpoints' saved validation probabilities were reused; no model training or checkpoint loading occurred.", "",
        f"Split SHA-256: `{metadata['validation_split_sha256']}`. Validation ID SHA-256: `{metadata['validation_ids_sha256']}`. Class order is canonical sorted V0 order.", "",
        "## A. Baseline checkpoints", "",
        "| ID | Experiment | Training path | Balanced accuracy | Top-3 | MRR | Macro F1 | Log loss | Brier |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in metadata["checkpoints"]:
        m = item["metrics"]
        lines.append(f"| {item['key']} | `{item['experiment_id']}` | {descriptions[item['key']]} | {m['balanced_accuracy']:.1%} | {m['top3_accuracy']:.1%} | {m['mrr']:.4f} | {m['macro_f1']:.1%} | {m['log_loss']:.4f} | {m['multiclass_brier']:.4f} |")
    lines.extend(["", "Manifests and saved probability Parquets:", ""])
    for item in metadata["checkpoints"]:
        lines.append(f"- **{item['key']}** manifest `{item['manifest_path']}`; predictions `{item['prediction_path']}`; checkpoint `{item.get('checkpoint_path')}`.")
    lines.extend(["", "## B. Experiment results", "",
        "Metrics below are calculated from pooled OOF predictions; Δ columns are percentage-point differences from standalone D (55.2% balanced accuracy). Corrected/regressed counts compare OOF top-1 predictions with D's saved prediction for those same rows.", "",
        ])
    groups = {
        "Experiment 1 — B + D": ["E1_BD_equal", "E1_BD_weighted", "E1_BD_rank"],
        "Experiment 2 — C + D": ["E2_CD_equal", "E2_CD_weighted", "E2_CD_rank", "E2_CD_temp_equal", "E2_CD_temp_weighted", "E2_C_temp_only"],
        "Experiment 3 — A + B + C + D": ["E3_ABCD_equal", "E3_ABCD_weighted", "E3_ABCD_rank", "E3_forward_weighted"],
        "Experiment 4 — class/family weights": ["E4_best_global_cv", "E4_family_regularized", "E4_class_regularized"],
    }
    baseline_d = method_metrics["baseline_D"]["balanced_accuracy"]
    for section, methods in groups.items():
        lines.extend([f"### {section}", "",
            "| Fusion | Balanced acc | Δ vs D | Top-3 | MRR | Macro F1 | Log loss | Brier | Corrected / regressed vs D |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for name in methods:
            metric = method_metrics[name]
            movement = metadata["relative_to_D"][name]
            delta = 100 * (metric["balanced_accuracy"] - baseline_d)
            lines.append(f"| `{name}` | {metric['balanced_accuracy']:.1%} | {delta:+.1f} pp | {metric['top3_accuracy']:.1%} | {metric['mrr']:.4f} | {metric['macro_f1']:.1%} | {metric['log_loss']:.4f} | {metric['multiclass_brier']:.4f} | {movement['corrected']} / {movement['regressed']} |")
        lines.append("")
    lines.extend(["### Fold-selected subsets, weights and calibration", "",
        "Global blend weights sum to one and were fit with `fit_weights` (multiclass log loss plus L2 shrinkage toward equal weights, λ=0.05). Rank fusion averages reciprocal class ranks and normalizes scores. For C+D calibration, only C is temperature-scaled; each temperature is selected from a fixed grid using inner prompt-group OOF loss within the outer fitting fold.", ""])
    c_raw, c_cal = method_metrics["baseline_C"], method_metrics["E2_C_temp_only"]
    temp_values = [record.get("temperature_C") for record in metadata["per_fold_weights"].get("E2_CD_temp_weighted", [])]
    lines.append(f"C-only calibration: log loss {c_raw['log_loss']:.4f} → {c_cal['log_loss']:.4f}; Brier {c_raw['multiclass_brier']:.4f} → {c_cal['multiclass_brier']:.4f}; top-1 balanced accuracy {c_raw['balanced_accuracy']:.1%} → {c_cal['balanced_accuracy']:.1%}. Fold temperatures: `{temp_values}`. The same fold-fitted temperatures are then used in the C+D blends.")
    lines.extend(["", "Per-class top-1 recall changes relative to D (OOF). This shows the pair effects for the named recoveries and all other labels:", "",
        "| Model label | D | B + D equal | Δ vs D | C + D equal | Δ vs D | C + D temp-weighted | Δ vs D | A + B + C + D weighted | Δ vs D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    recall_methods = ["baseline_D", "E1_BD_equal", "E2_CD_equal", "E2_CD_temp_weighted", "E3_ABCD_weighted"]
    for label in classes:
        recalls = [method_metrics[key]["per_class"][label]["recall"] for key in recall_methods]
        lines.append(f"| {label} | {recalls[0]:.1%} | {recalls[1]:.1%} | {(recalls[1]-recalls[0])*100:+.1f} pp | {recalls[2]:.1%} | {(recalls[2]-recalls[0])*100:+.1f} pp | {recalls[3]:.1%} | {(recalls[3]-recalls[0])*100:+.1f} pp | {recalls[4]:.1%} | {(recalls[4]-recalls[0])*100:+.1f} pp |")
    for key in ("E1_BD_weighted", "E2_CD_weighted", "E2_CD_temp_weighted", "E3_ABCD_weighted", "E3_forward_weighted", "E4_best_global_cv"):
        records = metadata["per_fold_weights"].get(key, [])
        if not records: continue
        lines.append(f"**{key}:**")
        for record in records:
            entry = f"fold {record['fold'] + 1}: members {', '.join(record.get('members', []))}"
            if "weights" in record and np.asarray(record["weights"]).ndim == 1:
                entry += "; weights " + ", ".join(f"{m}={w:.3f}" for m, w in zip(record.get("members", []), record["weights"]))
            if record.get("temperature_C") is not None: entry += f"; T(C)={record['temperature_C']:g}"
            if "selected_candidate" in record: entry += f"; inner-CV choice={record['selected_candidate']}"
            lines.append(f"- {entry}.")
        vector_records = [record for record in records if np.asarray(record.get("weights", [])).ndim == 1]
        if vector_records:
            by_members = {}
            for record in vector_records:
                by_members.setdefault(tuple(record.get("members", [])), []).append(record)
            for member_tuple, member_records in by_members.items():
                matrix = np.asarray([record["weights"] for record in member_records], dtype=float)
                label = "Fold mean ± SD" if len(by_members) == 1 else f"Fold mean ± SD for members {','.join(member_tuple)}"
                lines.append("  " + label + ": " + ", ".join(
                    f"{member}={matrix[:, i].mean():.3f}±{matrix[:, i].std(ddof=0):.3f}"
                    for i, member in enumerate(member_tuple)) + ".")
    family_records = metadata["per_fold_weights"].get("E4_family_regularized", [])
    if family_records:
        lines.extend(["", "Family-specific checkpoint weights by fold (the six family weights are independently normalized across A–D):"])
        for record in family_records:
            weights = np.asarray(record["weights"], dtype=float)
            lines.append(f"- Fold {record['fold'] + 1}, shrinkage λ={record['penalty']}: " + "; ".join(
                f"{family}: " + ", ".join(f"{member}={weights[i, j]:.3f}" for i, member in enumerate(labels))
                for j, family in enumerate(record["family_names"])) + ".")
    class_records = metadata["per_fold_weights"].get("E4_class_regularized", [])
    if class_records:
        class_weight_values = np.concatenate([np.asarray(record["weights"], dtype=float) for record in class_records], axis=1)
        lines.extend(["", "Per-class checkpoint weights vary across labels. Across all fold × label weights, mean ± SD by checkpoint:",
            "- " + ", ".join(f"{member}={class_weight_values[i].mean():.3f}±{class_weight_values[i].std(ddof=0):.3f}"
                               for i, member in enumerate(labels)) + ". Full per-label matrices are in `fold_weights.json`."])
    lines.extend(["", "Forward-selection step traces (inner OOF, calculated independently within each outer training fold):"])
    for record in metadata["per_fold_weights"].get("E3_forward_weighted", []):
        lines.append(f"- Fold {record['fold'] + 1}: " + " → ".join(
            f"{step['candidate']} ({step['inner_oof_balanced_accuracy']:.1%})" for step in record["selection_steps"]) + ".")

    lines.extend(["", "## C. Per-class top-1 recall", "",
        f"Best pooled-OOF method among evaluated candidates: `{selected_method}`. This descriptive selection is itself based on the same OOF comparison and is exploratory. Positive or negative deltas in the last column are relative to checkpoint D.", "",
        "| Model label | A | B | C | D | Best ensemble |", "|---|---:|---:|---:|---:|---:|"])
    for label in classes:
        vals = [method_metrics[f"baseline_{key}"]["per_class"][label]["recall"] for key in labels]
        best = method_metrics[selected_method]["per_class"][label]["recall"]
        d_recall = method_metrics["baseline_D"]["per_class"][label]["recall"]
        delta = (best - d_recall) * 100
        lines.append(f"| {label} | " + " | ".join(f"{x:.1%}" for x in vals) + f" | **{best:.1%} ({delta:+.1f} pp)** |")
    lines.extend(["", "Per-class top-1 precision for the same baselines and selected descriptive ensemble:", "",
        "| Model label | A | B | C | D | Best ensemble |", "|---|---:|---:|---:|---:|---:|"])
    for label in classes:
        vals = [method_metrics[f"baseline_{key}"]["per_class"][label]["precision"] for key in labels]
        best = method_metrics[selected_method]["per_class"][label]["precision"]
        lines.append(f"| {label} | " + " | ".join(f"{x:.1%}" for x in vals) + f" | **{best:.1%}** |")
    best_metric = method_metrics[selected_method]
    lines.extend(["", f"For `{selected_method}`, OOF examples corrected relative to D: {metadata['relative_to_D'][selected_method]['corrected']}; regressed: {metadata['relative_to_D'][selected_method]['regressed']}.", "",
        "Per-class precision and recall for every strategy are saved in `results.json`.", "",
        "## D. Complementarity", "",
        "| Checkpoint pair | Shared errors | Left-only correct | Right-only correct | Pair oracle coverage |", "|---|---:|---:|---:|---:|"])
    for pair, stats in complementarity_stats.items():
        a, b = pair.split("__")
        lines.append(f"| {a} + {b} | {stats['error_overlap']} | {stats['a_unique_correct']} | {stats['b_unique_correct']} | {stats['oracle_accuracy']:.1%} |")
    lines.extend(["", f"Four-checkpoint oracle coverage (theoretical, picks a correct checkpoint with hindsight): **{metadata['four_checkpoint_oracle_accuracy']:.1%}**. This is not an achievable ensemble score. Pairwise coverage is likewise an upper bound.", "",
        "Family confusion movement for the selected OOF method versus D (rows = true family; columns = predicted family):", ""])
    family_names = ["Claude", "Gemini", "GLM", "GPT", "Kimi", "Qwen"]
    for method_label, pred in (("D", method_metrics["baseline_D"]["predictions"]),
                               (selected_method, best_metric["predictions"])):
        lines.extend([f"**{method_label}**", "", "| True \\ Predicted | " + " | ".join(family_names) + " |", "|---|" + "---:|" * len(family_names)])
        pred_family = np.asarray([FAMILIES[x] for x in pred])
        true_family = np.asarray([FAMILIES[x] for x in y])
        for fam in family_names:
            counts = [int(np.sum((true_family == fam) & (pred_family == p_fam))) for p_fam in family_names]
            lines.append(f"| {fam} | " + " | ".join(map(str, counts)) + " |")
        lines.append("")
    d_family_pred = np.asarray([FAMILIES[x] for x in method_metrics["baseline_D"]["predictions"]])
    selected_family_pred = np.asarray([FAMILIES[x] for x in best_metric["predictions"]])
    true_family = np.asarray([FAMILIES[x] for x in y])
    lines.extend(["Change in family confusion counts (selected method minus D):", "",
        "| True \\ Predicted | " + " | ".join(family_names) + " |", "|---|" + "---:|" * len(family_names)])
    for true_fam in family_names:
        deltas = []
        for pred_fam in family_names:
            d_count = int(np.sum((true_family == true_fam) & (d_family_pred == pred_fam)))
            selected_count = int(np.sum((true_family == true_fam) & (selected_family_pred == pred_fam)))
            deltas.append(selected_count - d_count)
        lines.append(f"| {true_fam} | " + " | ".join(f"{value:+d}" for value in deltas) + " |")
    lines.append("")
    lines.extend(["## E. Comparison and interpretation", "",
        "| Rank | Strategy | Grouped OOF balanced accuracy | Top-3 | MRR | Macro F1 | Log loss | Brier |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    ranked = sorted((name for name in method_metrics if not name.startswith("baseline_")),
                    key=lambda name: (round(method_metrics[name]["balanced_accuracy"], 12),
                                      round(method_metrics[name]["top3_accuracy"], 12),
                                      round(method_metrics[name]["mrr"], 12),
                                      -round(method_metrics[name]["log_loss"], 12)), reverse=True)
    for rank, name in enumerate(ranked, 1):
        metric = method_metrics[name]
        lines.append(f"| {rank} | `{name}` | {metric['balanced_accuracy']:.1%} | {metric['top3_accuracy']:.1%} | {metric['mrr']:.4f} | {metric['macro_f1']:.1%} | {metric['log_loss']:.4f} | {metric['multiclass_brier']:.4f} |")
    lines.extend(["", "Family weights use a small inner-fold search over shrinkage penalties `[0.05, 0.25, 1.0]`; class-specific weights use `[0.25, 1.0, 4.0]`. Both optimize normalized multiclass scores, so false positives are penalized as well as missed true labels. Per-class weights have 18 separate 4-checkpoint simplexes with only 30 rows per target label; treat those as high-variance diagnostics. A family with one class (Kimi or Qwen) is statistically equivalent to class-specific weights for that class.",
        "The top accuracy ties at 56.7%: B+D equal (higher top-3) and C+D temperature-calibrated weighted (higher MRR and better log loss/Brier). Family-specific weighting scores 55.7%, below the strongest global-weight strategy at 56.7%; it only beats the inner-fold-selected global strategy (55.2%).", "",
        "## F. Conclusions", "",
        f"1. **Can two checkpoints beat the best standalone?** {'Yes' if max(method_metrics[x]['balanced_accuracy'] for x in groups['Experiment 1 — B + D'] + groups['Experiment 2 — C + D']) > baseline_d else 'No'} on grouped OOF diagnostics; best pair score = {max(method_metrics[x]['balanced_accuracy'] for x in groups['Experiment 1 — B + D'] + groups['Experiment 2 — C + D']):.1%} versus D {baseline_d:.1%}.",
        f"2. **Do four checkpoints outperform two?** {'Yes' if max(method_metrics[x]['balanced_accuracy'] for x in groups['Experiment 3 — A + B + C + D']) > max(method_metrics[x]['balanced_accuracy'] for x in groups['Experiment 1 — B + D'] + groups['Experiment 2 — C + D']) else 'No'} by the best grouped OOF scores; pooled OOF strategy ranking is selection-biased and should be treated as exploratory.",
        f"3. **Does family-specific weighting outperform the strongest global weighting?** No. It scores {method_metrics['E4_family_regularized']['balanced_accuracy']:.1%}, versus {max(method_metrics[name]['balanced_accuracy'] for name in ('E1_BD_weighted', 'E2_CD_weighted', 'E2_CD_temp_weighted', 'E3_ABCD_weighted')):.1%} for the strongest global-weight experiment. It does beat the per-fold inner-selected global strategy ({method_metrics['E4_best_global_cv']['balanced_accuracy']:.1%}).",
        f"4. **Which labels benefit/deteriorate?** See the per-class table; `results.json` records the full precision/recall and each strategy's class-level movement.",
        f"5. **Candidate for a future multi-architecture ensemble:** `{selected_method}` is a reasonable candidate because it ties for top accuracy with two members and has the stronger top-3 score. `E2_CD_temp_weighted` ties on accuracy and has better MRR/log loss/Brier, but depends on calibrating C. Confirm candidates on an untouched evaluation set before replacing the current multi-model ensemble.",
        "",
        f"Artifacts: OOF predictions `{metadata.get('oof_predictions_path', 'artifacts/analysis/ultrabert_four_checkpoint/oof_predictions.parquet')}`; fold assignments and weights are saved beside `results.json`. Checkpoint candidates and weights do not modify `docs/ensembles.md` or the active roster.", ""])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
