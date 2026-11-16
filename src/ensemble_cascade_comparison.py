"""Cross-fitted evaluation of Trust List 2 checkpoint-rule cascades.

Uses only the aligned probability cache created by the prior
``ensemble_precision_recall_overlap`` analysis. No checkpoints are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS
from .trust_list2_weighted_vote import _model_family

ROOT = Path(__file__).resolve().parents[1]
TRUST_PATH = Path("docs/trust-list-2-v025.csv")
CACHE_DIR = Path("artifacts/analysis/ensemble_precision_recall_overlap")
OUT_DIR = Path("artifacts/analysis/ensemble_cascade_comparison")
REPORT_PATH = Path("docs/ensemble_cascade_comparison.md")
EQUAL9_IDS = ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
ORDERINGS = ("precision", "f1", "precision_recall")
MIN_TRAIN_SUPPORT = 10


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _v0(root: Path) -> pd.DataFrame:
    frame = _dataset_rows("v0", root / "data/processed").copy()
    frame["id"] = frame.id.astype(str)
    frame["prompt_id"] = frame.prompt_id.astype(str)
    frame["model"] = frame.model.astype(str)
    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    if len(frame) != 2699 or frame.id.duplicated().any() or set(frame.model) != set(LABELS):
        raise ValueError("Expected the 2,699 unique official V0 rows and canonical labels")
    return frame


def _load_inputs(root: Path):
    trust_path = root / TRUST_PATH
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    cache = root / CACHE_DIR
    cache_config = json.loads((cache / "experiment_config.json").read_text())
    prediction_manifest = json.loads((cache / "prediction_cache_manifest.json").read_text())
    expected_trust_hash = _sha256(trust_path)
    if cache_config.get("trust_list_sha256") != expected_trust_hash:
        raise ValueError("Cached rule predictions were built from a different Trust List 2 file")
    v0 = _v0(root)
    ids = v0.id.tolist()
    labels = v0.model.to_numpy(dtype=object)
    prompts = v0.prompt_id.to_numpy(dtype=object)
    state_rows = pd.read_parquet(cache / "row_predictions.parquet")[["id", "prompt_id", "y_true"]]
    state_rows = state_rows.copy()
    for col in ("id", "prompt_id", "y_true"):
        state_rows[col] = state_rows[col].astype(str)
    if (state_rows.id.tolist() != ids or state_rows.prompt_id.tolist() != prompts.tolist() or
            state_rows.y_true.tolist() != labels.tolist()):
        raise ValueError("Reusable ensemble cache is not aligned to current sorted V0")
    checkpoint_ids = sorted(trust.sequence_id.unique(), key=lambda value: int(value[1:]))
    stored_ids = prediction_manifest.get("checkpoint_ids", [])
    if set(stored_ids) != set(checkpoint_ids):
        raise ValueError("Cached checkpoint roster does not equal Trust List 2 v025")
    archive = np.load(cache / "checkpoint_probabilities.npz")
    probabilities = {}
    predictions = {}
    for checkpoint_id in checkpoint_ids:
        key = f"p_{checkpoint_id}"
        if key not in archive.files:
            raise FileNotFoundError(f"Cached probabilities missing {key}")
        matrix = np.asarray(archive[key], dtype=float)
        if matrix.shape != (len(v0), len(LABELS)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid probability matrix for {checkpoint_id}: {matrix.shape}")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid normalized probabilities for {checkpoint_id}")
        probabilities[checkpoint_id] = matrix
        predictions[checkpoint_id] = np.asarray(LABELS, dtype=object)[matrix.argmax(axis=1)]
    # Reuse the existing five GroupKFold assignments so results are directly
    # comparable with the earlier V0 overlap/weight analysis.
    held_sets = []
    for fold in range(1, 6):
        record = json.loads((cache / "folds" / f"fold_{fold:02d}.json").read_text())
        held_sets.append(np.asarray(record["held_indexes"], dtype=int))
    held_flat = np.concatenate(held_sets)
    if len(held_flat) != len(v0) or set(held_flat.tolist()) != set(range(len(v0))):
        raise ValueError("Existing folds do not form a full partition of V0")
    fold_id = np.empty(len(v0), dtype=int)
    for fold, held in enumerate(held_sets):
        fold_id[held] = fold
        held_groups = set(prompts[held])
        fit_groups = set(prompts[np.setdiff1d(np.arange(len(v0)), held)])
        if held_groups & fit_groups:
            raise ValueError(f"Prompt group leakage in fold {fold + 1}")
    member_rows = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    baseline_prob = np.mean(np.stack([probabilities[x] for x in EQUAL9_IDS]), axis=0)
    baseline_prob /= baseline_prob.sum(axis=1, keepdims=True)
    baseline_pred = np.asarray(LABELS, dtype=object)[baseline_prob.argmax(axis=1)]
    if not np.isclose(balanced_accuracy_score(labels, baseline_pred), 0.653898, atol=2e-5):
        raise ValueError("EQUAL9 did not reproduce 65.3898% balanced accuracy")
    return v0, trust, probabilities, predictions, baseline_prob, baseline_pred, fold_id, member_rows


def _rule_rows(trust: pd.DataFrame, checkpoint_ids: list[str]):
    row_by_id = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    records = []
    for checkpoint_id in checkpoint_ids:
        for label in LABELS:
            records.append({"checkpoint_id": checkpoint_id, "label": label,
                            "model": str(row_by_id.loc[checkpoint_id, "model"]),
                            "experiment_id": str(row_by_id.loc[checkpoint_id, "experiment_id"])})
    return records


def _stat(y: np.ndarray, firing: np.ndarray, label: str) -> dict:
    n = int(firing.sum())
    tp = int(np.sum(firing & (y == label)))
    fp = n - tp
    support = int(np.sum(y == label))
    precision = tp / n if n else 0.0
    recall = tp / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"support": support, "prediction_count": n, "tp": tp, "fp": fp,
            "fn": support - tp, "precision": precision, "recall": recall, "f1": f1}


def _ordering_metric(stat: dict, method: str) -> float:
    if method == "precision":
        return float(stat["precision"])
    if method == "f1":
        return float(stat["f1"])
    return float(stat["precision"] * stat["recall"])


def _sort_rules(rules: list[tuple[str, str]], stats: dict, method: str):
    label_order = {label: i for i, label in enumerate(LABELS)}
    return sorted(rules, key=lambda rule: (-_ordering_metric(stats[rule], method),
                                           rule[0], label_order[rule[1]]))


def _fit_stats(rule: tuple[str, str], y: np.ndarray, preds: dict[str, np.ndarray],
               rows: np.ndarray, available: np.ndarray | None = None) -> dict:
    cp, label = rule
    fire = preds[cp][rows] == label
    row_y = y[rows]
    if available is not None:
        fire = fire[available]
        row_y = row_y[available]
    return _stat(row_y, fire, label)


def _best_for_label(candidates: list[tuple[str, str]], label: str,
                    fit_y: np.ndarray, preds: dict[str, np.ndarray], rows: np.ndarray,
                    objective: str, available: np.ndarray | None = None,
                    exclude: set[str] | None = None, precision_floor: float | None = None):
    scored = []
    for rule in candidates:
        if rule[1] != label or (exclude and rule[0] in exclude):
            continue
        stat = _fit_stats(rule, fit_y, preds, rows, available)
        if stat["prediction_count"] < MIN_TRAIN_SUPPORT:
            continue
        if precision_floor is not None and stat["precision"] + 1e-12 < precision_floor:
            continue
        scored.append((rule, stat))
    if not scored:
        return None, None
    columns = {
        "precision": ("precision", "recall", "f1", "tp", "prediction_count"),
        "f1": ("f1", "precision", "recall", "tp", "prediction_count"),
        "recall": ("recall", "precision", "f1", "tp", "prediction_count"),
        "incremental_f1": ("f1", "precision", "tp", "prediction_count"),
        "incremental_tp_fp": ("tp_fp_ratio", "tp", "precision", "f1"),
    }[objective]
    if objective == "incremental_tp_fp":
        for _, stat in scored:
            stat["tp_fp_ratio"] = stat["tp"] / max(stat["fp"], 1)
    scored.sort(key=lambda item: tuple(-float(item[1].get(k, 0)) for k in columns) + (item[0][0],))
    return scored[0]


def _apply(rules: list[tuple[str, str]], rows: np.ndarray, y: np.ndarray,
           preds: dict[str, np.ndarray], fallback: np.ndarray,
           rule_round: dict[tuple[str, str], int] | None = None):
    output = fallback[rows].copy()
    unclaimed = np.ones(len(rows), dtype=bool)
    contributions = []
    for pos, rule in enumerate(rules, 1):
        checkpoint_id, label = rule
        fire = unclaimed & (preds[checkpoint_id][rows] == label)
        count = int(fire.sum())
        tp = int(np.sum(fire & (y[rows] == label)))
        fp = count - tp
        remaining_true = int(np.sum(unclaimed & (y[rows] == label)))
        contributions.append({"rule_position": pos, "round": int((rule_round or {}).get(rule, 1)),
                              "checkpoint_id": checkpoint_id, "label": label,
                              "handled_rows": count, "unique_tp": tp,
                              "additional_fp": fp,
                              "incremental_precision": tp / count if count else 0.0,
                              "incremental_recall": tp / remaining_true if remaining_true else 0.0,
                              "incremental_f1": (2 * tp / (2 * tp + fp + (remaining_true - tp))
                                                 if 2 * tp + fp + (remaining_true - tp) else 0.0),
                              "remaining_true_label_rows": remaining_true})
        output[fire] = label
        unclaimed[fire] = False
    return output, contributions, unclaimed


def _select_single(strategy: str, order: str, rows: np.ndarray, y: np.ndarray,
                   preds: dict[str, np.ndarray], candidates: list[tuple[str, str]]):
    by_label, stats = [], {}
    objective = {"C1": "precision", "C2": "f1", "C3": "recall"}[strategy]
    for label in LABELS:
        rule, stat = _best_for_label(candidates, label, y, preds, rows, objective,
            precision_floor=0.90 if strategy == "C3" else None)
        if rule is not None:
            by_label.append(rule)
            stats[rule] = stat
    return _sort_rules(by_label, stats, order), stats


def _select_two(strategy: str, order: str, rows: np.ndarray, y: np.ndarray,
                preds: dict[str, np.ndarray], candidates: list[tuple[str, str]],
                fallback: np.ndarray):
    first_objective = "precision" if strategy == "C4" else "f1"
    first, first_stats = [], {}
    for label in LABELS:
        rule, stat = _best_for_label(candidates, label, y, preds, rows, first_objective)
        if rule is not None:
            first.append(rule)
            first_stats[rule] = stat
    first_ordered = _sort_rules(first, first_stats, order)
    _, _, residual = _apply(first_ordered, rows, y, preds, fallback)

    # Rank labels by the best complementary second rule on rows still available
    # after the complete first-rule round. At selection time below, recompute a
    # label's candidate after earlier second rules have consumed rows.
    provisional = {}
    second_objective = "incremental_f1" if strategy == "C4" else "incremental_tp_fp"
    for label in LABELS:
        first_rule = next((r for r in first if r[1] == label), None)
        if first_rule is None:
            continue
        second, stat = _best_for_label(candidates, label, y, preds, rows,
            second_objective, available=residual, exclude={first_rule[0]})
        if second is not None:
            provisional[label] = (second, stat)
    second_labels = sorted(provisional,
        key=lambda label: (-_ordering_metric(provisional[label][1], order),
                           provisional[label][0][0], LABELS.index(label)))

    second_ordered, second_stats, chosen_rounds = [], {}, {}
    still_available = residual.copy()
    for label in second_labels:
        first_rule = next((r for r in first if r[1] == label), None)
        second, stat = _best_for_label(candidates, label, y, preds, rows,
            second_objective, available=still_available, exclude={first_rule[0]})
        if second is None:
            continue
        second_ordered.append(second)
        second_stats[second] = stat
        chosen_rounds[second] = 2
        fire = preds[second[0]][rows] == label
        still_available &= ~fire
    ordered = first_ordered + second_ordered
    round_by_rule = {r: 1 for r in first_ordered} | chosen_rounds
    return ordered, {**first_stats, **second_stats}, round_by_rule


def _balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, p))


def _cascade_c6(rows: np.ndarray, y: np.ndarray, preds: dict[str, np.ndarray],
                candidates: list[tuple[str, str]], fallback: np.ndarray):
    """Greedy forward rule addition, followed by same-label rule replacements."""
    selected: list[tuple[str, str]] = []
    per_label = {label: 0 for label in LABELS}
    support = np.asarray([np.sum(y[rows] == label) for label in LABELS], dtype=float)
    label_index = {label: i for i, label in enumerate(LABELS)}
    current = fallback[rows].copy()
    claimed = np.zeros(len(rows), dtype=bool)
    tol = 1e-12
    while len(selected) < 2 * len(LABELS):
        best = None
        for cp, label in candidates:
            rule = (cp, label)
            if rule in selected or per_label[label] >= 2:
                continue
            fire = (~claimed) & (preds[cp][rows] == label)
            if int(fire.sum()) < MIN_TRAIN_SUPPORT:
                continue
            y_idx = np.fromiter((label_index[v] for v in y[rows][fire]), dtype=int)
            old_ok = current[fire] == y[rows][fire]
            new_ok = y[rows][fire] == label
            delta = float(np.sum((new_ok.astype(float) - old_ok.astype(float)) / support[y_idx]) / len(LABELS))
            if (best is None or delta > best[0] + tol or
                    (abs(delta - best[0]) <= tol and (cp, label_index[label]) <
                     (best[1][0], label_index[best[1][1]]))):
                best = (delta, rule, fire)
        if best is None or best[0] <= tol:
            break
        delta, rule, fire = best
        selected.append(rule)
        per_label[rule[1]] += 1
        current[fire] = rule[1]
        claimed[fire] = True

    def score(rule_list):
        predicted, _, _ = _apply(rule_list, rows, y, preds, fallback)
        return _balanced_accuracy(y[rows], predicted), predicted

    # One-for-one swaps among checkpoints of the same label preserve the
    # two-rules-per-label constraint while improving the training-fold score.
    improved = True
    passes = 0
    while improved and passes < 4:
        improved = False
        passes += 1
        base_score, _ = score(selected)
        for pos, old in enumerate(list(selected)):
            alternatives = [r for r in candidates if r[1] == old[1] and r not in selected]
            best_rule, best_score = old, base_score
            for rule in alternatives:
                trial = selected.copy(); trial[pos] = rule
                trial_score, _ = score(trial)
                if trial_score > best_score + tol or (abs(trial_score-best_score) <= tol and rule < best_rule):
                    best_rule, best_score = rule, trial_score
            if best_rule != old and best_score > base_score + tol:
                selected[pos] = best_rule
                improved = True
                base_score = best_score
        # Adjacent order swaps are tested until no local ordering improvement.
        order_improved = True
        while order_improved:
            order_improved = False
            base_score, _ = score(selected)
            for pos in range(len(selected)-1):
                trial = selected.copy(); trial[pos], trial[pos+1] = trial[pos+1], trial[pos]
                trial_score, _ = score(trial)
                if trial_score > base_score + tol:
                    selected = trial; base_score = trial_score
                    improved = order_improved = True
    return selected


def _vote_predict(fit_rows: np.ndarray, eval_rows: np.ndarray, y: np.ndarray,
                 preds: dict[str, np.ndarray], trust: pd.DataFrame,
                 fallback: np.ndarray, *, weighted: bool):
    """Fold-fitted version of the prior family-normalized checkpoint voting."""
    model_by_id = trust.drop_duplicates("sequence_id").set_index("sequence_id").model.astype(str).to_dict()
    family_by_id = {cp: _model_family(model_by_id[cp]) for cp in preds}
    active_labels = {}
    fit_y = y[fit_rows]
    for cp, cp_pred in preds.items():
        active_labels[cp] = {label for label in LABELS
            if int(np.sum(cp_pred[fit_rows] == label)) >= MIN_TRAIN_SUPPORT}
    family_counts = {}
    for cp, labels in active_labels.items():
        if labels:
            family_counts[family_by_id[cp]] = family_counts.get(family_by_id[cp], 0) + 1
    scores = np.zeros((len(eval_rows), len(LABELS)), dtype=float)
    label_idx = {label: i for i, label in enumerate(LABELS)}
    for cp, cp_pred in preds.items():
        eligible = active_labels[cp]
        if not eligible:
            continue
        family_denominator = family_counts[family_by_id[cp]]
        fit_prediction = cp_pred[fit_rows]
        for label in eligible:
            if weighted:
                fire = fit_prediction == label
                support = int(fire.sum())
                tp = int(np.sum(fire & (fit_y == label)))
                raw = (tp / support) * support / (support + 30)
            else:
                raw = 1.0
            eval_mask = cp_pred[eval_rows] == label
            scores[eval_mask, label_idx[label]] += raw / family_denominator
    prediction = np.asarray(LABELS, dtype=object)[scores.argmax(axis=1)]
    no_votes = scores.sum(axis=1) == 0
    prediction[no_votes] = fallback[eval_rows][no_votes]
    return prediction, {"active_checkpoints": int(sum(bool(v) for v in active_labels.values())),
                         "no_vote_rows": int(no_votes.sum())}


def _round_numbered(rules: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    seen = {}
    rounds = {}
    for rule in rules:
        seen[rule[1]] = seen.get(rule[1], 0) + 1
        rounds[rule] = seen[rule[1]]
    return rounds


def _evaluate(y: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=LABELS, zero_division=0)
    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {"precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "support": int(support[i]),
                            "correct": int(np.sum((y == label) & (prediction == label)))}
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "correct": int(np.sum(y == prediction)), "per_label": per_label}


def _metrics_row(scope: str, name: str, y: np.ndarray, prediction: np.ndarray,
                 baseline: np.ndarray, rule_count: int = 0, contributions=None):
    m = _evaluate(y, prediction)
    old_ok = baseline == y; new_ok = prediction == y
    m.update({"scope": scope, "configuration": name,
        "delta_balanced_accuracy_pp_vs_equal9": (m["balanced_accuracy"] - _evaluate(y, baseline)["balanced_accuracy"]) * 100,
        "delta_accuracy_pp_vs_equal9": (m["accuracy"] - _evaluate(y, baseline)["accuracy"]) * 100,
        "corrected_vs_equal9": int(np.sum(~old_ok & new_ok)),
        "introduced_vs_equal9": int(np.sum(old_ok & ~new_ok)),
        "disagreements_vs_equal9": int(np.sum(prediction != baseline)),
        "rule_count": (float(rule_count) if rule_count is not None and np.isfinite(rule_count) else np.nan)})
    if contributions is not None:
        m["fallback_rows"] = int(sum(x["handled_rows"] for x in contributions) * 0)  # assigned below from trace
    return m


def run(*, root: str | Path = ROOT, output_dir: str | Path = OUT_DIR,
        report_path: str | Path = REPORT_PATH) -> dict:
    root = Path(root).resolve()
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    report = Path(report_path); report = report if report.is_absolute() else root / report
    out.mkdir(parents=True, exist_ok=True)
    v0, trust, probs, preds, baseline_prob, baseline_pred, folds, _ = _load_inputs(root)
    y = v0.model.to_numpy(dtype=object)
    cp_ids = sorted(probs, key=lambda x: int(x[1:]))
    rules = [(cp, label) for cp in cp_ids for label in LABELS]
    indices = np.arange(len(v0))

    # Full V0 descriptive metrics and exact same-label pair overlap.
    rule_metric_rows = []
    firing = np.zeros((len(rules), len(v0)), dtype=bool)
    rule_pos = {rule: i for i, rule in enumerate(rules)}
    for rule_i, (cp, label) in enumerate(rules):
        mask = preds[cp] == label
        firing[rule_i] = mask
        stat = _stat(y, mask, label)
        trust_row = trust.loc[trust.sequence_id.astype(str).eq(cp) & trust.target_label.astype(str).eq(label)].iloc[0]
        rule_metric_rows.append({"checkpoint_id": cp, "experiment_id": trust_row.experiment_id,
            "model": trust_row.model, "label": label, **stat,
            "trust_list_precision": float(trust_row.precision),
            "trust_list_recall": float(trust_row.recall),
            "trust_list_prediction_count": int(trust_row.prediction_count)})
    pd.DataFrame(rule_metric_rows).to_csv(out / "checkpoint_rule_metrics.csv", index=False)
    overlap_rows = []
    for label in LABELS:
        label_rules = [r for r in rules if r[1] == label]
        true_mask = y == label
        for ai, a in enumerate(label_rules):
            ma = firing[rule_pos[a]]
            for b in label_rules[ai+1:]:
                mb = firing[rule_pos[b]]
                overlap_rows.append({"label": label, "checkpoint_a": a[0], "checkpoint_b": b[0],
                    "shared_tp": int(np.sum(ma & mb & true_mask)),
                    "shared_fp": int(np.sum(ma & mb & ~true_mask)),
                    "a_unique_tp": int(np.sum(ma & ~mb & true_mask)),
                    "a_unique_fp": int(np.sum(ma & ~mb & ~true_mask)),
                    "b_unique_tp": int(np.sum(mb & ~ma & true_mask)),
                    "b_unique_fp": int(np.sum(mb & ~ma & ~true_mask))})
    pd.DataFrame(overlap_rows).to_csv(out / "same_label_pairwise_overlap.csv", index=False)
    np.savez_compressed(out / "rule_row_membership.npz", firing=firing,
        rule_keys=np.asarray([f"{cp}::{label}" for cp, label in rules]),
        ids=np.asarray(v0.id.astype(str).tolist()),
        true_labels=np.asarray(v0.model.astype(str).tolist()))

    # Reproducible row-to-fold assignments.
    pd.DataFrame({"id": v0.id, "prompt_id": v0.prompt_id, "fold": folds + 1}).to_csv(
        out / "fold_assignments.csv", index=False)

    strategy_names = [*(f"{c}_{o}" for c in ("C1", "C2", "C3", "C4", "C5") for o in ORDERINGS),
                      "C6_greedy_swaps", "majority_vote", "precision_weighted_vote"]
    oof = {name: np.full(len(v0), "", dtype=object) for name in strategy_names}
    per_fold_configs = {}
    held_contribution_rows = []
    vote_fold_details = {}
    fold_dir = out / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        fit_rows = indices[folds != fold]
        held_rows = indices[folds == fold]
        fit_y = y
        per_fold_configs[str(fold + 1)] = {}
        fold_json = fold_dir / f"fold_{fold + 1:02d}.json"
        fold_npz = fold_dir / f"fold_{fold + 1:02d}.npz"
        fold_contrib = fold_dir / f"fold_{fold + 1:02d}_contributions.json"
        if fold_json.is_file() and fold_npz.is_file() and fold_contrib.is_file():
            saved = json.loads(fold_json.read_text())
            if (saved.get("algorithm_version") == 2 and
                    saved.get("trust_sha256") == _sha256(root / TRUST_PATH) and
                    saved.get("held_indexes") == held_rows.tolist() and
                    saved.get("configurations") == strategy_names):
                per_fold_configs[str(fold + 1)] = saved["selected_rules"]
                with np.load(fold_npz, allow_pickle=True) as archive:
                    for name in strategy_names:
                        oof[name][held_rows] = archive[name]
                np.savez_compressed(fold_npz, **{key: values[held_rows].astype(str)
                                                 for key, values in oof.items()})
                held_contribution_rows.extend(json.loads(fold_contrib.read_text()))
                continue
        candidates = []
        for cp, label in rules:
            stat = _fit_stats((cp, label), fit_y, preds, fit_rows)
            if stat["prediction_count"] >= MIN_TRAIN_SUPPORT:
                candidates.append((cp, label))
        for strategy in ("C1", "C2", "C3"):
            for order in ORDERINGS:
                name = f"{strategy}_{order}"
                selected, _ = _select_single(strategy, order, fit_rows, fit_y, preds, candidates)
                predicted, _, fallback_mask = _apply(selected, held_rows, y, preds, baseline_pred)
                oof[name][held_rows] = predicted
                per_fold_configs[str(fold + 1)][name] = [{"checkpoint_id": cp, "label": label, "round": 1}
                                                         for cp, label in selected]
                _, held_contrib, _ = _apply(selected, held_rows, y, preds, baseline_pred)
                for row in held_contrib:
                    held_contribution_rows.append({"fold": fold + 1, "configuration": name, **row})
                held_contribution_rows.append({"fold": fold + 1, "configuration": name,
                    "rule_position": len(selected)+1, "round": 0, "checkpoint_id": "EQUAL9",
                    "label": "fallback", "handled_rows": int(fallback_mask.sum()),
                    "unique_tp": int(np.sum(fallback_mask & (y[held_rows] == baseline_pred[held_rows])),),
                    "additional_fp": int(np.sum(fallback_mask & (y[held_rows] != baseline_pred[held_rows]))),
                    "incremental_precision": np.nan, "incremental_recall": np.nan,
                    "incremental_f1": np.nan, "remaining_true_label_rows": np.nan})
        for strategy in ("C4", "C5"):
            for order in ORDERINGS:
                name = f"{strategy}_{order}"
                selected, _, rounds = _select_two(strategy, order, fit_rows, fit_y, preds, candidates, baseline_pred)
                predicted, _, fallback_mask = _apply(selected, held_rows, y, preds, baseline_pred, rounds)
                oof[name][held_rows] = predicted
                per_fold_configs[str(fold + 1)][name] = [{"checkpoint_id": cp, "label": label,
                                                          "round": rounds[(cp, label)]} for cp, label in selected]
                _, held_contrib, _ = _apply(selected, held_rows, y, preds, baseline_pred, rounds)
                for row in held_contrib:
                    held_contribution_rows.append({"fold": fold + 1, "configuration": name, **row})
                held_contribution_rows.append({"fold": fold + 1, "configuration": name,
                    "rule_position": len(selected)+1, "round": 0, "checkpoint_id": "EQUAL9",
                    "label": "fallback", "handled_rows": int(fallback_mask.sum()),
                    "unique_tp": int(np.sum(fallback_mask & (y[held_rows] == baseline_pred[held_rows]))),
                    "additional_fp": int(np.sum(fallback_mask & (y[held_rows] != baseline_pred[held_rows]))),
                    "incremental_precision": np.nan, "incremental_recall": np.nan,
                    "incremental_f1": np.nan, "remaining_true_label_rows": np.nan})
        c6 = _cascade_c6(fit_rows, y, preds, candidates, baseline_pred)
        name = "C6_greedy_swaps"
        predicted, _, fallback_mask = _apply(c6, held_rows, y, preds, baseline_pred)
        oof[name][held_rows] = predicted
        c6_rounds = _round_numbered(c6)
        per_fold_configs[str(fold + 1)][name] = [{"checkpoint_id": cp, "label": label,
            "round": c6_rounds[(cp, label)]} for cp, label in c6]
        _, held_contrib, _ = _apply(c6, held_rows, y, preds, baseline_pred, c6_rounds)
        for row in held_contrib:
            held_contribution_rows.append({"fold": fold + 1, "configuration": name, **row})
        held_contribution_rows.append({"fold": fold + 1, "configuration": name,
            "rule_position": len(c6)+1, "round": 0, "checkpoint_id": "EQUAL9",
            "label": "fallback", "handled_rows": int(fallback_mask.sum()),
            "unique_tp": int(np.sum(fallback_mask & (y[held_rows] == baseline_pred[held_rows]))),
            "additional_fp": int(np.sum(fallback_mask & (y[held_rows] != baseline_pred[held_rows]))),
            "incremental_precision": np.nan, "incremental_recall": np.nan,
            "incremental_f1": np.nan, "remaining_true_label_rows": np.nan})
        for vote_name, weighted in (("majority_vote", False), ("precision_weighted_vote", True)):
            vote_pred, vote_info = _vote_predict(fit_rows, held_rows, y, preds, trust,
                                                baseline_pred, weighted=weighted)
            oof[vote_name][held_rows] = vote_pred
            per_fold_configs[str(fold + 1)][vote_name] = []
            vote_fold_details.setdefault(str(fold + 1), {})[vote_name] = vote_info
        np.savez_compressed(fold_npz, **{key: values[held_rows].astype(str) for key, values in oof.items()})
        fold_json.write_text(json.dumps({"algorithm_version": 2,
            "trust_sha256": _sha256(root / TRUST_PATH),
            "held_indexes": held_rows.tolist(), "configurations": strategy_names,
            "selected_rules": per_fold_configs[str(fold + 1)]}, indent=2) + "\n")
        fold_contrib.write_text(json.dumps([x for x in held_contribution_rows
            if x.get("fold") == fold + 1], indent=2) + "\n")
        (out / "analysis_state.json").write_text(json.dumps({
            "status": "running", "trust_sha256": _sha256(root / TRUST_PATH),
            "folds_completed": fold + 1, "completed_fold_ids": list(range(1, fold + 2)),
            "rows": len(v0)}, indent=2) + "\n")

    # Full V0 selection is retained strictly as a diagnostic, not the ranking basis.
    full_configs = {}
    full_predictions = {}
    full_contrib_rows = []
    candidates_full = [r for r in rules if _fit_stats(r, y, preds, indices)["prediction_count"] >= MIN_TRAIN_SUPPORT]
    for strategy in ("C1", "C2", "C3"):
        for order in ORDERINGS:
            name = f"{strategy}_{order}"
            selected, _ = _select_single(strategy, order, indices, y, preds, candidates_full)
            pred, contrib, fallback_mask = _apply(selected, indices, y, preds, baseline_pred)
            full_predictions[name] = pred
            full_configs[name] = [{"checkpoint_id": cp, "label": label, "round": 1} for cp, label in selected]
            full_contrib_rows.extend({"configuration": name, **row} for row in contrib)
            full_contrib_rows.append({"configuration": name, "rule_position": len(selected)+1,
                "round": 0, "checkpoint_id": "EQUAL9", "label": "fallback",
                "handled_rows": int(fallback_mask.sum()),
                "unique_tp": int(np.sum(fallback_mask & (y == baseline_pred))),
                "additional_fp": int(np.sum(fallback_mask & (y != baseline_pred)))})
    for strategy in ("C4", "C5"):
        for order in ORDERINGS:
            name = f"{strategy}_{order}"
            selected, _, rounds = _select_two(strategy, order, indices, y, preds, candidates_full, baseline_pred)
            pred, contrib, fallback_mask = _apply(selected, indices, y, preds, baseline_pred, rounds)
            full_predictions[name] = pred
            full_configs[name] = [{"checkpoint_id": cp, "label": label, "round": rounds[(cp, label)]}
                                  for cp, label in selected]
            full_contrib_rows.extend({"configuration": name, **row} for row in contrib)
            full_contrib_rows.append({"configuration": name, "rule_position": len(selected)+1,
                "round": 0, "checkpoint_id": "EQUAL9", "label": "fallback",
                "handled_rows": int(fallback_mask.sum()),
                "unique_tp": int(np.sum(fallback_mask & (y == baseline_pred))),
                "additional_fp": int(np.sum(fallback_mask & (y != baseline_pred)))})
    c6_full = _cascade_c6(indices, y, preds, candidates_full, baseline_pred)
    c6_full_rounds = _round_numbered(c6_full)
    full_predictions["C6_greedy_swaps"] = _apply(c6_full, indices, y, preds, baseline_pred, c6_full_rounds)[0]
    full_configs["C6_greedy_swaps"] = [{"checkpoint_id": cp, "label": label,
        "round": c6_full_rounds[(cp, label)]} for cp, label in c6_full]
    _, contrib, fallback_mask = _apply(c6_full, indices, y, preds, baseline_pred, c6_full_rounds)
    full_contrib_rows.extend({"configuration": "C6_greedy_swaps", **row} for row in contrib)
    full_contrib_rows.append({"configuration": "C6_greedy_swaps", "rule_position": len(c6_full)+1,
        "round": 0, "checkpoint_id": "EQUAL9", "label": "fallback",
        "handled_rows": int(fallback_mask.sum()), "unique_tp": int(np.sum(fallback_mask & (y == baseline_pred))),
        "additional_fp": int(np.sum(fallback_mask & (y != baseline_pred)))})

    vote_full_details = {}
    for vote_name, weighted in (("majority_vote", False), ("precision_weighted_vote", True)):
        vote_prediction, vote_info = _vote_predict(indices, indices, y, preds, trust,
                                                   baseline_pred, weighted=weighted)
        full_predictions[vote_name] = vote_prediction
        vote_full_details[vote_name] = vote_info

    # Reuse the already completed same-v025 weighted probability study.
    prior_metrics = json.loads((root / "artifacts/analysis/ensemble_precision_recall_overlap/metrics.json").read_text())
    prior_rows = pd.read_parquet(root / "artifacts/analysis/ensemble_precision_recall_overlap/row_predictions.parquet")
    metrics_rows = []
    oof_fallback_counts = {}
    for contribution in held_contribution_rows:
        if contribution.get("label") == "fallback":
            key = contribution["configuration"]
            oof_fallback_counts[key] = oof_fallback_counts.get(key, 0) + int(contribution["handled_rows"])
    baseline_oof = baseline_pred
    for scope, pred_map in (("prompt_group_oof", oof), ("full_v0_diagnostic", full_predictions)):
        baseline = baseline_oof
        metrics_rows.append(_metrics_row(scope, "EQUAL9", y, baseline, baseline))
        for name, pred in pred_map.items():
            if name in full_configs and scope == "prompt_group_oof":
                count_rules = float(np.mean([len(per_fold_configs[str(fold + 1)].get(name, [])) for fold in range(5)]))
            elif name in full_configs:
                count_rules = len(full_configs[name])
            else:
                count_rules = np.nan
            row = _metrics_row(scope, name, y, pred, baseline, count_rules)
            if scope == "prompt_group_oof" and name in oof_fallback_counts:
                row["fallback_rows"] = oof_fallback_counts[name]
            elif scope == "full_v0_diagnostic" and name in full_configs:
                rule_pairs = [(x["checkpoint_id"], x["label"]) for x in full_configs[name]]
                _, _, fallback_mask = _apply(rule_pairs, indices, y, preds, baseline_pred,
                    {(x["checkpoint_id"], x["label"]): x.get("round", 1) for x in full_configs[name]})
                row["fallback_rows"] = int(fallback_mask.sum())
            metrics_rows.append(row)
    for method in ("equal_selected", "per_label_f1", "complementarity"):
        for scope, scope_key in (("prompt_group_oof", "cross_fitted"), ("full_v0_diagnostic", "full_v0_diagnostic")):
            m = prior_metrics[scope_key][method]
            metrics_rows.append({"scope": scope, "configuration": f"weighted_{method}",
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "correct": m["correct"], "delta_balanced_accuracy_pp_vs_equal9":
                    (m["balanced_accuracy"] - prior_metrics[scope_key]["equal9"]["balanced_accuracy"])*100,
                "delta_accuracy_pp_vs_equal9": (m["accuracy"]-prior_metrics[scope_key]["equal9"]["accuracy"])*100,
                "corrected_vs_equal9": m.get("corrected"),
                "introduced_vs_equal9": m.get("introduced"),
                "disagreements_vs_equal9": m.get("disagreements"), "rule_count": np.nan})

    # Metrics and contributions, with OOF folds kept on held-out rows only.
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(held_contribution_rows).to_csv(out / "oof_rule_contributions.csv", index=False)
    pd.DataFrame(full_contrib_rows).to_csv(out / "full_v0_rule_contributions.csv", index=False)
    per_label_rows = []
    for scope, pred_map in (("prompt_group_oof", oof), ("full_v0_diagnostic", full_predictions)):
        scoped_predictions = ({"EQUAL9": baseline_pred, **pred_map}
                             if scope == "prompt_group_oof" else pred_map)
        for name, prediction in scoped_predictions.items():
            m = _evaluate(y, prediction)
            for label, stats in m["per_label"].items():
                per_label_rows.append({"scope": scope, "configuration": name, "label": label, **stats})
    for method in ("equal_selected", "per_label_f1", "complementarity"):
        for scope, scope_key in (("prompt_group_oof", "cross_fitted"),
                                 ("full_v0_diagnostic", "full_v0_diagnostic")):
            for label, stats in prior_metrics[scope_key][method]["per_label"].items():
                per_label_rows.append({"scope": scope, "configuration": f"weighted_{method}",
                                       "label": label, **stats})
    pd.DataFrame(per_label_rows).to_csv(out / "per_label_metrics.csv", index=False)

    prediction_frame = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    prediction_frame["pred_equal9"] = baseline_pred
    for name, prediction in oof.items(): prediction_frame[f"pred_oof::{name}"] = prediction
    for name, prediction in full_predictions.items(): prediction_frame[f"pred_full::{name}"] = prediction
    # Bring over cached weighted ensemble predictions for exact row-level comparison.
    weighted_cols = [c for c in prior_rows.columns if c in {"pred_oof::equal_selected", "pred_full::equal_selected",
        "pred_oof::per_label_f1", "pred_full::per_label_f1", "pred_oof::complementarity", "pred_full::complementarity"}]
    if prior_rows.id.astype(str).tolist() != v0.id.astype(str).tolist():
        prior_rows = prior_rows.set_index(prior_rows.id.astype(str)).loc[v0.id.astype(str)].reset_index(drop=True)
    for col in weighted_cols: prediction_frame[col] = prior_rows[col].to_numpy()
    prediction_frame.to_parquet(out / "cascade_predictions.parquet", index=False)
    json_payload = {"cross_fitted_folds": per_fold_configs, "full_v0_diagnostic": full_configs}
    (out / "selected_rules.json").write_text(json.dumps(json_payload, indent=2) + "\n")
    (out / "vote_fit_details.json").write_text(json.dumps({"cross_fitted": vote_fold_details,
        "full_v0_diagnostic": vote_full_details}, indent=2) + "\n")

    # Fold-level rule stability and C3 no-rule diagnostics.
    stability = []
    for name in strategy_names:
        positions = {}
        for fold_text, configs in per_fold_configs.items():
            for position, record in enumerate(configs[name], 1):
                key = (record["checkpoint_id"], record["label"], record["round"])
                positions.setdefault(key, []).append(position)
        for (cp, label, round_num), pvals in positions.items():
            stability.append({"configuration": name, "checkpoint_id": cp, "label": label,
                              "round": round_num, "folds_selected": len(pvals),
                              "mean_position": float(np.mean(pvals)), "positions": ",".join(map(str,pvals))})
    pd.DataFrame(stability).to_csv(out / "rule_stability.csv", index=False)
    no_c3 = {}
    for fold in range(5):
        rows = indices[folds != fold]
        selected, _ = _select_single("C3", "precision", rows, y, preds,
            [r for r in rules if _fit_stats(r,y,preds,rows)["prediction_count"] >= MIN_TRAIN_SUPPORT])
        selected_labels = {label for _,label in selected}
        no_c3[str(fold+1)] = [label for label in LABELS if label not in selected_labels]
    full_c3_labels = {item["label"] for item in full_configs.get("C3_precision", [])}
    no_c3["full_v0_diagnostic"] = [label for label in LABELS if label not in full_c3_labels]
    (out / "c3_fallback_labels.json").write_text(json.dumps(no_c3, indent=2) + "\n")

    config = {"trust_list_version": "trust-list-2-v025", "trust_list_path": str(TRUST_PATH),
        "trust_sha256": _sha256(root / TRUST_PATH), "v0_rows": len(v0),
        "v0_ordered_ids_sha256": hashlib.sha256("\n".join(v0.id.tolist()).encode()).hexdigest(),
        "prompt_groups": int(v0.prompt_id.nunique()), "checkpoint_count": len(cp_ids),
        "class_order": LABELS, "baseline": "EQUAL9", "equal9_checkpoint_ids": EQUAL9_IDS,
        "fold_source": "existing ensemble_precision_recall_overlap five-fold GroupKFold artifacts",
        "min_training_fold_rule_support": MIN_TRAIN_SUPPORT,
        "second_rule_recall_denominator": "true-label rows that remain unclaimed when the rule is reached",
        "C5_complementarity_objective": "incremental TP / max(incremental FP, 1), then TP, precision, F1",
        "C6": "greedy forward selection by training-fold balanced-accuracy gain, max two rules per label; same-label rule replacements and adjacent order swaps",
        "prediction_cache": str(CACHE_DIR / "checkpoint_probabilities.npz"),
        "prediction_cache_sha256": _sha256(root / CACHE_DIR / "checkpoint_probabilities.npz")}
    (out / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"status": "completed", "trust_sha256": config["trust_sha256"],
        "prediction_cache_sha256": config["prediction_cache_sha256"], "rows": len(v0),
        "folds_completed": 5, "artifact_dir": str(out.relative_to(root)),
        "report_path": str(report.relative_to(root))}
    (out / "analysis_state.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(report, metrics_df, per_label_rows, full_configs, per_fold_configs,
                  no_c3, config, out)
    return manifest


def _write_report(path: Path, metrics: pd.DataFrame, per_label_rows: list[dict],
                  full_configs: dict, fold_configs: dict, no_c3: dict,
                  config: dict, out: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cross = metrics[metrics.scope.eq("prompt_group_oof")].copy()
    casc = cross[cross.configuration.str.match(r"C[1-6]")].sort_values(
        ["balanced_accuracy", "accuracy"], ascending=False)
    baseline = float(cross.loc[cross.configuration.eq("EQUAL9"), "balanced_accuracy"].iloc[0])
    cached_predictions = pd.read_parquet(out / "cascade_predictions.parquet")
    top3 = []
    for row in casc.itertuples(index=False):
        column = f"pred_oof::{row.configuration}"
        prediction = cached_predictions[column].to_numpy()
        if any(np.array_equal(prediction, cached_predictions[f"pred_oof::{name}"].to_numpy())
               for name in [x.configuration for x in top3]):
            continue
        top3.append(row)
        if len(top3) == 3:
            break
    lines = ["# Trust List 2 v025 cascade comparison", "",
        "Cache-only, prompt-group cross-fitted comparison on the 2,699 official V0 rows. No model training, model loading, new inference, production-ensemble changes, or Kaggle submissions were performed.", "",
        f"Trust List 2: `{config['trust_list_version']}`; `{config['trust_list_path']}`; SHA-256 `{config['trust_sha256']}`. The 78-checkpoint probability cache is verified against this list and the ordered V0 IDs. The existing five GroupKFold prompt partitions were reused.", "",
        "The cross-fitted scores are the primary comparison. Checkpoints were previously selected using V0, so even held-out prompt folds do not remove checkpoint-selection exposure. Full-V0-selected cascades are optimistic diagnostics only.", "",
        f"EQUAL9 baseline: **{baseline:.4%} balanced accuracy**. Rule selection uses only each fold's other four partitions. A rule requires at least {MIN_TRAIN_SUPPORT} matching rows in its fitting partition. Rules fire in listed order; unmatched rows use EQUAL9.", "",
        "## Overall results", "",
        "| Scope | Configuration | Rules | Accuracy | Balanced accuracy | Δ BA vs EQUAL9 | Corrected | Introduced | Rows differing | Fallback rows |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.sort_values(["scope", "balanced_accuracy"], ascending=[True, False]).itertuples(index=False):
        if row.configuration == "EQUAL9":
            continue
        fallback = getattr(row, "fallback_rows", "")
        rule_text = (f"{row.rule_count:.1f} avg" if row.scope == "prompt_group_oof" and pd.notna(row.rule_count)
                     else str(int(row.rule_count)) if pd.notna(row.rule_count) else "—")
        lines.append(f"| {row.scope} | {row.configuration} | {rule_text} | {row.accuracy:.4%} | {row.balanced_accuracy:.4%} | {row.delta_balanced_accuracy_pp_vs_equal9:+.3f} pp | {int(row.corrected_vs_equal9) if pd.notna(row.corrected_vs_equal9) else '—'} | {int(row.introduced_vs_equal9) if pd.notna(row.introduced_vs_equal9) else '—'} | {int(row.disagreements_vs_equal9) if pd.notna(row.disagreements_vs_equal9) else '—'} | {int(fallback) if pd.notna(fallback) and fallback != '' else '—'} |")
    lines += ["", "`weighted_equal_selected`, `weighted_per_label_f1`, and `weighted_complementarity` are the previously reported probability ensembles, reused from the same v025 cache and fold assignments. Family-normalized majority and precision-weighted votes were also refit using the v025 predictions, with fit-fold-only per-label support and precision, then applied to held-out rows.", "",
        "## Best three distinct cross-fitted cascades", "",
        "Ranked by prompt-group OOF balanced accuracy. Ordering variants with identical OOF predictions are collapsed so the list shows distinct cascades.", ""]
    for rank, row in enumerate(top3, 1):
        lines.append(f"{rank}. **{row.configuration}** — {row.balanced_accuracy:.4%} balanced accuracy ({row.delta_balanced_accuracy_pp_vs_equal9:+.3f} pp vs EQUAL9).")
        for fold_name in sorted(fold_configs, key=int):
            rules = fold_configs[fold_name].get(row.configuration, [])
            order = " → ".join(f"{x['checkpoint_id']}:{x['label']}" for x in rules) if rules else "EQUAL9 only"
            lines.append(f"   - Fold {fold_name}: {order}.")
    label_lookup = {(item["scope"], item["configuration"], item["label"]): item
                    for item in per_label_rows}
    lines += ["", "## Per-label OOF metrics for EQUAL9 and the three leading distinct cascades", "",
        "Cells show precision / recall / F1. Per-label values for every ordering and weighted comparator are in `per_label_metrics.csv`.", "",
        "| Label | EQUAL9 P/R/F1 | " + " | ".join(x.configuration for x in top3) + " |",
        "|---|---:|" + "---:|" * len(top3)]
    for label in LABELS:
        base_stats = label_lookup[("prompt_group_oof", "EQUAL9", label)]
        cells = [f"{base_stats['precision']:.3f} / {base_stats['recall']:.3f} / {base_stats['f1']:.3f}"]
        for row in top3:
            stats = label_lookup[("prompt_group_oof", row.configuration, label)]
            cells.append(f"{stats['precision']:.3f} / {stats['recall']:.3f} / {stats['f1']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Full-V0 diagnostics", "",
        "These configurations select rules and score them on the same V0 rows; do not treat them as held-out performance. Exact selected sequences are in `selected_rules.json`.", "",
        "## Incremental rule contributions", "",
        "Held-out rule handling, unique true positives, additional false positives and incremental precision/recall/F1 are in `oof_rule_contributions.csv`; full-V0 diagnostics are in `full_v0_rule_contributions.csv`. Incremental recall is measured against still-unclaimed true rows of the rule's label. The final EQUAL9 fallback row reports remaining examples.", "",
        "## Rule metrics and overlap", "",
        f"`checkpoint_rule_metrics.csv` contains full-V0 precision, recall, F1, TP, FP and support for every checkpoint-label pair ({len(LABELS)} labels × {config['checkpoint_count']} checkpoints). `same_label_pairwise_overlap.csv` gives exact shared and unique TP/FP counts for every same-label rule pair. `rule_row_membership.npz` stores the row-level firing matrix, V0 IDs and true labels so overlaps can be reconstructed exactly.", "",
        "## Rule stability", "",
        "`rule_stability.csv` reports how often each exact checkpoint-label rule is selected across the five fit folds and its mean cascade position. C3 labels without a ≥90% precision rule fall back to EQUAL9; fold-wise and full-V0 diagnostic label lists are in `c3_fallback_labels.json`.", "",
        "## Per-label metrics", "",
        "Full precision, recall, F1, support and correct counts for all cascade, voting and previously reported probability-ensemble configurations are in `per_label_metrics.csv`.", "",
        "## Reproducibility", "",
        "- `experiment_config.json`: trust/cache hashes, fold source, selection thresholds and strategy definitions.",
        "- `fold_assignments.csv`: prompt-group fold for every V0 row.",
        "- `selected_rules.json`: fold-specific and full-V0 rule orders.",
        "- `cascade_predictions.parquet`: row-aligned OOF/full predictions plus prior weighted ensemble predictions.",
        "- `metrics.csv`, `per_label_metrics.csv`, and contribution/overlap files: machine-readable outputs.",
        "- `analysis_state.json`: completion and input hashes.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()
    print(json.dumps(run(root=args.root, output_dir=args.output_dir, report_path=args.report), indent=2))


if __name__ == "__main__":
    main()
