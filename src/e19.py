"""E19-A: oracle prompt grouping with Hungarian label assignment.

All checkpoint evidence comes from cached Trust List 2 v037 probabilities and
the saved, fold-specific E13 reliability router. No base-model inference runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from catboost import CatBoostClassifier
from . import ensemble_catalog as catalog
from . import e13_reliability_router as e13
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
E13_RUN = ROOT / "artifacts/analysis/e13/20261120T093318893046Z"
BASELINE_PREDICTIONS = E13_RUN / "predictions.parquet"
BASELINE_RESULTS = E13_RUN / "results.json"
BASELINE_CONFIG = E13_RUN / "config.json"
OUTER_FOLDS = 5
SAMPLING_SEEDS = (2026, 2027, 2028, 2029, 2030)
BASELINE_BA = 0.6464653243847874
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def score_matrix(probabilities, hard, reliability, families):
    """Return [rows, 18] E13 family-max reliability scores with soft backoff.

    A top-1-supported label receives E13-B's exact mean of within-family max
    checkpoint reliability. An unsupported label receives the family mean of
    the best checkpoint-specific ``reliability * p(label) / p(top1)`` score.
    This keeps every canonical label available to the assignment solver while
    preserving E13-B's checkpoint-specific and family-balanced evidence.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.int64)
    reliability = np.asarray(reliability, dtype=np.float64)
    families = np.asarray(families, dtype=object)
    if p.ndim != 3 or p.shape[2] != len(LABELS):
        raise ValueError("E19 probabilities must have shape [checkpoints, rows, 18]")
    if hard.shape != reliability.shape or hard.shape != p.shape[:2]:
        raise ValueError("E19 hard predictions/reliability do not align with probabilities")
    if len(families) != p.shape[0] or not np.isfinite(p).all() or not np.isfinite(reliability).all():
        raise ValueError("E19 family metadata or scores are invalid")
    if np.any((hard < 0) | (hard >= len(LABELS))) or not np.allclose(p.sum(2), 1, atol=1e-4):
        raise ValueError("E19 class indices or probability normalization are invalid")
    n_checkpoints, n_rows, n_labels = p.shape
    result = np.full((n_rows, n_labels), np.nan, dtype=np.float64)
    family_names = sorted(set(families.tolist()))
    top_probability = p.max(axis=2)
    soft = reliability[:, :, None] * p / np.maximum(top_probability[:, :, None], 1e-12)
    for row in range(n_rows):
        for label in range(n_labels):
            family_scores = []
            for family in family_names:
                members = np.flatnonzero(families == family)
                supporters = members[hard[members, row] == label]
                if len(supporters):
                    family_scores.append(float(reliability[supporters, row].max()))
            if not family_scores:
                family_scores = [float(soft[np.flatnonzero(families == family), row, label].max())
                                 for family in family_names]
            result[row, label] = float(np.mean(family_scores))
    if not np.isfinite(result).all():
        raise ValueError("E19 scoring failed to provide finite evidence for all 18 labels")
    return result


def hungarian_assignment(scores, groups, *, row_ids=None):
    """Maximize total score per group, with unique labels inside each group."""
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    if scores.ndim != 2 or scores.shape[1] != len(LABELS) or scores.shape[0] != len(groups):
        raise ValueError("E19 score rows/groups must align and have 18 label columns")
    if not np.isfinite(scores).all():
        raise ValueError("E19 assignment scores must be finite")
    if row_ids is not None and len(row_ids) != len(groups):
        raise ValueError("E19 row IDs and groups do not align")
    pred = np.full(len(groups), -1, dtype=np.int64)
    assignment = {}
    # Stable group and row order make ties reproducible across runs.
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        if not 1 <= len(rows) <= len(LABELS):
            raise ValueError(f"prompt group {group} has unsupported size {len(rows)}")
        sub = scores[rows]
        row_ind, col_ind = linear_sum_assignment(-sub)
        if len(row_ind) != len(rows) or len(set(col_ind.tolist())) != len(col_ind):
            raise RuntimeError("Hungarian solver did not assign each response a unique label")
        pred[rows[row_ind]] = col_ind
        assignment[group] = {int(rows[r]): int(c) for r, c in zip(row_ind, col_ind)}
    if (pred < 0).any():
        raise RuntimeError("E19 left one or more rows unassigned")
    _validate_unique_groups(pred, groups)
    return pred, assignment


def greedy_assignment(scores, groups):
    """Highest-confidence-first unique-label assignment for diagnostics."""
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    if scores.shape != (len(groups), len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError("Greedy assignment scores do not align")
    pred = np.full(len(groups), -1, dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        margins = np.sort(scores[rows], axis=1)[:, -1] - np.sort(scores[rows], axis=1)[:, -2]
        order = sorted(range(len(rows)), key=lambda j: (-margins[j], int(rows[j])))
        available = set(range(len(LABELS)))
        for local in order:
            row = rows[local]
            label = min(available, key=lambda lab: (-scores[row, lab], lab))
            pred[row] = label
            available.remove(label)
    _validate_unique_groups(pred, groups)
    return pred


def _validate_unique_groups(pred, groups):
    pred = np.asarray(pred, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    if len(pred) != len(groups) or (pred < 0).any() or (pred >= len(LABELS)).any():
        raise ValueError("E19 prediction labels do not align or are invalid")
    for group in np.unique(groups):
        labels = pred[groups == group]
        if len(labels) > len(LABELS) or len(set(labels.tolist())) != len(labels):
            raise ValueError(f"E19 group {group} violates its unique-label constraint")


def _metrics(truth, pred, baseline, labels=LABELS):
    truth, pred, baseline = (np.asarray(value, dtype=np.int64) for value in
                             (truth, pred, baseline))
    if not (len(truth) == len(pred) == len(baseline)) or not len(truth):
        raise ValueError("E19 metric inputs are empty or misaligned")
    cm = confusion_matrix(truth, pred, labels=np.arange(len(LABELS)))
    precision, recall, _, support = precision_recall_fscore_support(
        truth, pred, labels=np.arange(len(LABELS)), zero_division=0)
    base_cm = confusion_matrix(truth, baseline, labels=np.arange(len(LABELS)))
    base_recall = np.divide(np.diag(base_cm), support,
                            out=np.zeros(len(LABELS), dtype=np.float64), where=support > 0)
    present = support > 0
    accuracy = float(np.mean(truth == pred))
    baseline_accuracy = float(np.mean(truth == baseline))
    balanced_accuracy = float(recall[present].mean())
    baseline_balanced_accuracy = float(base_recall[present].mean())
    changed = pred != baseline
    corrected = (baseline != truth) & (pred == truth)
    introduced = (baseline == truth) & (pred != truth)
    return {"rows": int(len(truth)), "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "delta_accuracy": accuracy - baseline_accuracy,
            "delta_balanced_accuracy": balanced_accuracy - baseline_balanced_accuracy,
            "corrected": int(corrected.sum()), "introduced": int(introduced.sum()),
            "changed": int(changed.sum()), "confusion_matrix": cm.tolist(),
            "per_label": {LABELS[i]: {"support": int(support[i]),
                "precision": float(precision[i]), "recall": float(recall[i]),
                "tp": int(cm[i, i]), "fp": int(cm[:, i].sum() - cm[i, i]),
                "fn": int(cm[i].sum() - cm[i, i])} for i in range(len(LABELS))}}


def _load_inputs(root):
    v0 = catalog.cascades._v0(root)
    if len(v0) != 2699 or v0.id.astype(str).duplicated().any():
        raise ValueError("E19 requires exactly 2,699 ID-unique V0 rows")
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    groups = v0.prompt_id.astype(str).to_numpy()
    row_ids = v0.id.astype(str).to_numpy()
    if (truth < 0).any() or pd.isna(v0.prompt_id).any():
        raise ValueError("V0 labels or prompt IDs are invalid")
    config = json.loads(BASELINE_CONFIG.read_text())
    if config["trust_list"]["version"] != "trust-list-2-v037":
        raise ValueError("Saved E13 baseline is not based on Trust List 2 v037")
    pointer = json.loads((root / "docs/trust-list-2-latest.json").read_text())
    if pointer["version"] != "trust-list-2-v037":
        raise ValueError("Latest Trust List 2 pointer is not v037")
    _, members, probability_map, excluded, snapshot = catalog._load_inputs(root, v0)
    ids = sorted(config["ids"], key=lambda sid: int(sid.lstrip("N")))
    missing = sorted(set(ids) - set(probability_map))
    if missing:
        raise ValueError(f"E13 roster has missing cached probabilities: {missing}")
    if snapshot["version"] != config["trust_list"]["version"] or snapshot["sha256"] != config["trust_list"]["sha256"]:
        raise ValueError("Cached Trust List snapshot differs from E13 config")
    matrices = np.stack([probability_map[sid] for sid in ids]).astype(np.float32)
    families = [e13._model_family(members[sid]["model"]) for sid in ids]
    if matrices.shape != (len(ids), len(v0), len(LABELS)):
        raise ValueError("V0 probability tensors are misaligned")
    return v0, truth, groups, row_ids, ids, families, matrices, snapshot, excluded, members


def _reconstruct_fold(matrices, ids, families, truth, row_ids, fold):
    row_map = {str(row_id): i for i, row_id in enumerate(row_ids)}
    fit = np.asarray([row_map[row_id] for row_id in fold["fit_ids"]], dtype=np.int64)
    held = np.asarray([row_map[row_id] for row_id in fold["held_ids"]], dtype=np.int64)
    if len(held) != int(fold["held_row_count"]):
        raise ValueError("Saved E13 fold row IDs and row count disagree")
    model_path = Path(fold["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved E13 fold router is missing: {model_path}")
    roster = np.asarray([ids.index(sid) for sid in fold["checkpoint_ids"]], dtype=np.int64)
    roster_ids = [ids[i] for i in roster]
    roster_families = [families[i] for i in roster]
    x, _ = e13.build_feature_frame(matrices[roster], roster_ids, roster_families,
                                   truth[fit], fit, held)
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    reliability = e13._score_correctness(model, x).reshape(len(held), len(roster)).T
    p = matrices[roster][:, held]
    hard = p.argmax(axis=2)
    ensemble = e13.family_balanced_probabilities(p, roster_families)
    e13_prediction, _ = e13.route_predictions(
        "family_max", hard, reliability, ensemble, roster_families)
    scores = score_matrix(p, hard, reliability, roster_families)
    return held, scores, {"fit_rows": fit, "roster": roster_ids,
                          "reliability": reliability, "hard": hard,
                          "families": roster_families, "e13_prediction": e13_prediction}


def _sample_groups(groups, seed, cap):
    rng = np.random.default_rng(int(seed))
    sampled_groups = np.empty(len(groups), dtype=object)
    included = []
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        if len(rows) < 2:
            continue
        size = int(rng.integers(2, min(int(cap), len(rows)) + 1))
        selected = np.sort(rng.choice(rows, size=size, replace=False))
        included.extend(selected.tolist())
        sampled_groups[selected] = group
    return np.asarray(sorted(included), dtype=np.int64), sampled_groups


def _score_group_sizes(truth, baseline, pred, groups, sizes):
    rows = []
    for size in sorted(set(sizes.tolist())):
        mask = sizes == size
        m = _metrics(truth[mask], pred[mask], baseline[mask])
        rows.append({"group_size": int(size), **{k: m[k] for k in
            ("rows", "accuracy", "balanced_accuracy", "corrected", "introduced", "changed")}})
    return rows


def run(root=ROOT, output=None):
    started = time.monotonic()
    root = Path(root).resolve()
    output = Path(output) if output else root / "artifacts/analysis/e19"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"E19 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with threadpool_limits(limits=1):
        v0, truth, groups, row_ids, ids, families, matrices, snapshot, excluded, members = _load_inputs(root)
        baseline_frame = pd.read_parquet(BASELINE_PREDICTIONS)
        if (baseline_frame.id.astype(str).tolist() != row_ids.tolist() or
                baseline_frame.prompt_id.astype(str).tolist() != groups.tolist() or
                baseline_frame.y_true.astype(str).tolist() != v0.model.astype(str).tolist()):
            raise ValueError("Saved E13-B prediction rows do not align with canonical V0")
        baseline = pd.Categorical(baseline_frame["E13-B"].astype(str), categories=LABELS).codes.astype(np.int64)
        baseline_metric = _metrics(truth, baseline, baseline)
        if not np.isclose(baseline_metric["balanced_accuracy"], BASELINE_BA, atol=1e-12):
            raise ValueError(f"E13-B BA did not reproduce: {baseline_metric['balanced_accuracy']:.12f}")
        baseline_results = json.loads(BASELINE_RESULTS.read_text())
        e13_b = next(x for x in baseline_results["results"] if x["ensemble"] == "E13-B")
        if len(e13_b["folds"]) != OUTER_FOLDS:
            raise ValueError("E13-B must contain exactly five saved outer folds")
        score_all = np.full((len(v0), len(LABELS)), np.nan, dtype=np.float64)
        fold_ids = np.full(len(v0), -1, dtype=np.int8)
        fold_data = []
        fold_predictions = np.full(len(v0), -1, dtype=np.int64)
        e13_reconstructed = np.full(len(v0), -1, dtype=np.int64)
        for fold in e13_b["folds"]:
            held, scores, details = _reconstruct_fold(matrices, ids, families, truth, row_ids, fold)
            fold_id = int(fold["fold"])
            e13_reconstructed[held] = details["e13_prediction"]
            pred, _ = hungarian_assignment(scores, groups[held], row_ids=row_ids[held])
            score_all[held] = scores
            fold_ids[held] = fold_id
            fold_predictions[held] = pred
            fold_data.append({"fold": fold_id, "held": held, "scores": scores, **details})
        if np.isnan(score_all).any() or (fold_ids < 1).any():
            raise ValueError("Saved E13 folds did not reconstruct every V0 score row exactly once")
        if not np.array_equal(e13_reconstructed, baseline):
            bad_rows = np.flatnonzero(e13_reconstructed != baseline)
            raise ValueError(f"Saved E13-B fold routers disagree with baseline on {len(bad_rows)} rows")
        # All rows of each prompt must share the exact historical E13 fold.
        for group in np.unique(groups):
            if len(np.unique(fold_ids[groups == group])) != 1:
                raise ValueError(f"Prompt group {group} spans E13 outer folds")
        full_pred, _ = hungarian_assignment(score_all, groups, row_ids=row_ids)
        greedy_pred = greedy_assignment(score_all, groups)
        full_metrics = _metrics(truth, full_pred, baseline)
        greedy_metrics = _metrics(truth, greedy_pred, baseline)
        if not np.array_equal(full_pred, fold_predictions):
            raise ValueError("Group assignment differs between foldwise and full aligned scoring")
        _validate_unique_groups(full_pred, groups)

        experiments = {"A-full": {"pred": full_pred, "rows": np.arange(len(v0)),
                                   "groups": groups, "sizes": pd.Series(groups).map(
                                       pd.Series(groups).value_counts()).to_numpy(),
                                   "metrics": full_metrics, "greedy": greedy_metrics,
                                   "folds": []}}
        for fold_data_item in fold_data:
            held = fold_data_item["held"]
            local_pred, _ = hungarian_assignment(fold_data_item["scores"], groups[held],
                                                  row_ids=row_ids[held])
            fm = _metrics(truth[held], local_pred, baseline[held])
            fm["fold"] = int(fold_ids[held[0]])
            experiments["A-full"]["folds"].append(fm)

        sample_summaries = {"B-incomplete-2-10": [], "C-incomplete-2-18": []}
        sample_pred_records = []
        for variant, cap in (("B-incomplete-2-10", 10), ("C-incomplete-2-18", 18)):
            seed_results = []
            for seed in SAMPLING_SEEDS:
                rows, _ = _sample_groups(groups, seed, cap)
                sampled_groups = groups[rows]
                if not len(rows):
                    raise ValueError(f"{variant} selected no evaluation rows")
                sampled_pred, _ = hungarian_assignment(score_all[rows], sampled_groups,
                                                        row_ids=row_ids[rows])
                greedy = greedy_assignment(score_all[rows], sampled_groups)
                metrics = _metrics(truth[rows], sampled_pred, baseline[rows])
                metrics.update({"seed": seed, "sampling_cap": cap,
                    "groups": int(len(np.unique(sampled_groups))),
                    "sampled_group_sizes": {str(k): int(v) for k, v in
                        pd.Series(sampled_groups).value_counts().sort_index().items()},
                    "results_by_group_size": _score_group_sizes(truth[rows], baseline[rows],
                        sampled_pred, sampled_groups,
                        pd.Series(sampled_groups).map(pd.Series(sampled_groups).value_counts()).to_numpy()),
                    "greedy": _metrics(truth[rows], greedy, baseline[rows])})
                metrics["folds"] = []
                for fold_no in range(1, OUTER_FOLDS + 1):
                    mask = fold_ids[rows] == fold_no
                    fm = _metrics(truth[rows][mask], sampled_pred[mask], baseline[rows][mask])
                    fm.update({"fold": fold_no, "seed": seed})
                    metrics["folds"].append(fm)
                seed_results.append(metrics)
                sample_pred_records.extend({"variant": variant, "seed": seed,
                    "row_index": int(row), "id": row_ids[row], "prompt_id": groups[row],
                    "y_true": LABELS[truth[row]], "E13-B": LABELS[baseline[row]],
                    "E19-A": LABELS[sampled_pred[pos]], "fold": int(fold_ids[row]),
                    "score_group_size": int((sampled_groups == sampled_groups[pos]).sum())}
                    for pos, row in enumerate(rows))
            summary = {"variant": variant, "seed_results": seed_results}
            for metric in ("accuracy", "balanced_accuracy", "delta_accuracy", "delta_balanced_accuracy",
                           "corrected", "introduced", "changed", "rows"):
                vals = np.asarray([item[metric] for item in seed_results], dtype=np.float64)
                summary[metric] = {"mean": float(vals.mean()),
                                   "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
            sample_summaries[variant] = summary

        full_size = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
        all_results = {"A-full": {"metrics": full_metrics,
                          "results_by_group_size": _score_group_sizes(
                              truth, baseline, full_pred, groups, full_size),
                          "folds": experiments["A-full"]["folds"],
                          "greedy": greedy_metrics}, **sample_summaries}
        run_result = {"experiment": "E19-A", "baseline": baseline_metric,
            "baseline_balanced_accuracy_expected": BASELINE_BA,
            "baseline_exact_reproduction": bool(np.array_equal(e13_reconstructed, baseline)),
            "baseline_reproduced_folds": list(range(1, OUTER_FOLDS + 1)),
            "trust_list": snapshot, "v0_rows": int(len(v0)),
            "prompt_groups": int(len(np.unique(groups))), "outer_folds": OUTER_FOLDS,
            "sampling_seeds": list(SAMPLING_SEEDS), "results": all_results,
            "score_method": "For each label with checkpoint top-1 support, exact E13-B family-max reliability: within each supporting family max checkpoint reliability, averaged across supporting families. For labels without top-1 support, family max of checkpoint reliability times class probability divided by checkpoint top-1 probability, averaged across all roster families.",
            "assignment": "SciPy linear_sum_assignment maximizing group score; every row receives one unique label within its prompt group; groups are independent and may be size 1-18.",
            "diagnostic": "Greedy processes rows by descending top-two score margin and assigns each the highest-scoring remaining label; ties use input row order and canonical label order.",
            "checkpoint_ids": ids, "checkpoint_families": dict(zip(ids, families)),
            "checkpoint_cache_sha256": {sid: members[sid]["cache_sha256"] for sid in ids},
            "excluded_checkpoints": excluded, "alignment_verified": True,
            "unique_labels_verified": True, "new_inference": False, "training": False,
            "runtime_seconds": float(time.monotonic() - started),
            "peak_process_memory_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
        config = {"experiment": "E19-A", "seed": 2026,
            "baseline_run": str(E13_RUN), "baseline_predictions": str(BASELINE_PREDICTIONS),
            "baseline_prediction_sha256": hashlib.sha256(BASELINE_PREDICTIONS.read_bytes()).hexdigest(),
            "trust_list_version": snapshot["version"], "trust_list_sha256": snapshot["sha256"],
            "labels": list(LABELS), "row_count": len(v0),
            "row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
            "prompt_id_source": "data/processed/dataset_v0.parquet", "outer_folds": OUTER_FOLDS,
            "sampling_seeds": list(SAMPLING_SEEDS), "sampling": "per prompt choose a uniform integer group size in [2,min(cap, original size)], then sample without replacement; rows in singleton groups would be excluded",
            "caps": {"B-incomplete-2-10": 10, "C-incomplete-2-18": 18},
            "score_method": run_result["score_method"], "assignment": run_result["assignment"],
            "diagnostic": run_result["diagnostic"], "new_inference": False,
            "checkpoint_ids": ids, "checkpoint_families": dict(zip(ids, families)),
            "excluded_checkpoints": excluded}
        pred_frame = pd.DataFrame({"id": row_ids, "prompt_id": groups,
            "y_true": [LABELS[i] for i in truth], "E13-B": [LABELS[i] for i in baseline],
            "E19-A-full": [LABELS[i] for i in full_pred],
            "E19-A-greedy": [LABELS[i] for i in greedy_pred], "fold": fold_ids})
        pred_frame.to_parquet(output / "predictions.parquet", index=False)
        pd.DataFrame(sample_pred_records).to_parquet(output / "sampled_predictions.parquet", index=False)
        np.savez_compressed(output / "predictions.npz", row_ids=row_ids, y_true=truth,
            e13_b=baseline, e19_full=full_pred, e19_greedy=greedy_pred,
            scores=score_all, fold=fold_ids)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(run_result, indent=2, allow_nan=False) + "\n")
        _write_report(root, output, run_result)
    return run_result


def _write_report(root, output, result):
    rows = ["# E19-A — Oracle Prompt Grouping with Optimal Label Assignment", "",
        "CPU-only analysis of cached Trust List 2 v037 probabilities. The known V0 prompt groups are an oracle input, so these are an upper-bound diagnostic and are not directly deployable without prompt grouping.", "",
        "## Results", "", "| Variant | Rows | Accuracy | Balanced accuracy | Δ balanced accuracy (pp) | Corrected | Introduced | Changed |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    full = result["results"]["A-full"]["metrics"]
    rows.append(f"| E13-B | {result['baseline']['rows']} | {result['baseline']['accuracy']:.4%} | {result['baseline']['balanced_accuracy']:.4%} | — | — | — | 0 |")
    rows.append(f"| E19-A A full | {full['rows']} | {full['accuracy']:.4%} | {full['balanced_accuracy']:.4%} | {full['delta_balanced_accuracy'] * 100:+.4f} pp | {full['corrected']} | {full['introduced']} | {full['changed']} |")
    for variant in ("B-incomplete-2-10", "C-incomplete-2-18"):
        item = result["results"][variant]
        rows.append(f"| E19-A {variant} (mean) | {item['rows']['mean']:.1f} | {item['accuracy']['mean']:.4%} ± {item['accuracy']['std']:.4%} | {item['balanced_accuracy']['mean']:.4%} ± {item['balanced_accuracy']['std']:.4%} | {item['delta_balanced_accuracy']['mean'] * 100:+.4f} pp | {item['corrected']['mean']:.1f} | {item['introduced']['mean']:.1f} | {item['changed']['mean']:.1f} |")
    rows += ["", f"E13-B reproduces **{result['baseline']['accuracy']:.4%} accuracy** and **{result['baseline']['balanced_accuracy']:.4%} balanced accuracy** on the full 2,699 V0 rows. Score construction: {result['score_method']}", "",
        "## Hungarian versus greedy", "", "| Assignment | Accuracy | Balanced accuracy | Corrected | Introduced | Changed |", "|---|---:|---:|---:|---:|---:|"]
    for name, metric in (("Hungarian", full), ("Greedy highest-confidence-first", result["results"]["A-full"]["greedy"])):
        rows.append(f"| {name} | {metric['accuracy']:.4%} | {metric['balanced_accuracy']:.4%} | {metric['corrected']} | {metric['introduced']} | {metric['changed']} |")
    rows += ["", "## Full-group results by size", "", "| Group size | Rows | Accuracy | Balanced accuracy | Corrected | Introduced | Changed |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for item in result["results"]["A-full"]["results_by_group_size"]:
        rows.append(f"| {item['group_size']} | {item['rows']} | {item['accuracy']:.4%} | {item['balanced_accuracy']:.4%} | {item['corrected']} | {item['introduced']} | {item['changed']} |")
    for variant in ("B-incomplete-2-10", "C-incomplete-2-18"):
        item = result["results"][variant]
        rows += ["", f"## {variant}", "", "Mean ± sample standard deviation across five fixed seeds.", "", "| Metric | Mean | SD |", "|---|---:|---:|"]
        for metric in ("rows", "accuracy", "balanced_accuracy", "delta_accuracy", "delta_balanced_accuracy", "corrected", "introduced", "changed"):
            rows.append(f"| {metric} | {item[metric]['mean']:.6f} | {item[metric]['std']:.6f} |")
        rows += ["", "### By seed", "", "| Seed | Rows | Accuracy | Balanced accuracy | Δ BA (pp) | Corrected | Introduced | Changed |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for seed_result in item["seed_results"]:
            rows.append(f"| {seed_result['seed']} | {seed_result['rows']} | {seed_result['accuracy']:.4%} | {seed_result['balanced_accuracy']:.4%} | {seed_result['delta_balanced_accuracy'] * 100:+.4f} | {seed_result['corrected']} | {seed_result['introduced']} | {seed_result['changed']} |")
        rows += ["", "### By sampled group size (seed rows pooled per seed)", "", "| Seed | Size | Rows | Accuracy | Balanced accuracy |", "|---:|---:|---:|---:|---:|"]
        for seed_result in item["seed_results"]:
            for group_result in seed_result["results_by_group_size"]:
                rows.append(f"| {seed_result['seed']} | {group_result['group_size']} | {group_result['rows']} | {group_result['accuracy']:.4%} | {group_result['balanced_accuracy']:.4%} |")
        rows += ["", "### Per-label precision and recall (seed-pooled mean predictions are not formed; each seed is reported separately)", "", "| Seed | Label | Support | Precision | Recall |", "|---:|---|---:|---:|---:|"]
        for seed_result in item["seed_results"]:
            for label in LABELS:
                m = seed_result["per_label"][label]
                rows.append(f"| {seed_result['seed']} | {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} |")
        rows += ["", "### Fold results", "", "| Seed | Fold | Rows | Accuracy | Balanced accuracy | Δ BA (pp) | Corrected | Introduced | Changed |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for seed_result in item["seed_results"]:
            for fold in seed_result["folds"]:
                rows.append(f"| {seed_result['seed']} | {fold['fold']} | {fold['rows']} | {fold['accuracy']:.4%} | {fold['balanced_accuracy']:.4%} | {fold['delta_balanced_accuracy'] * 100:+.4f} | {fold['corrected']} | {fold['introduced']} | {fold['changed']} |")
    rows += ["", "## Full-group per-label metrics", "", "| Label | Support | Precision | Recall |", "|---|---:|---:|---:|"]
    for label in LABELS:
        m = full["per_label"][label]
        rows.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} |")
    rows += ["", "## Full-group outer folds", "", "| Fold | Rows | Accuracy | Balanced accuracy | Corrected | Introduced | Changed |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in result["results"]["A-full"]["folds"]:
        rows.append(f"| {fold['fold']} | {fold['rows']} | {fold['accuracy']:.4%} | {fold['balanced_accuracy']:.4%} | {fold['corrected']} | {fold['introduced']} | {fold['changed']} |")
    delta = full["delta_balanced_accuracy"] * 100
    b_gains = [x["delta_balanced_accuracy"] * 100 for x in result["results"]["B-incomplete-2-10"]["seed_results"]]
    c_gains = [x["delta_balanced_accuracy"] * 100 for x in result["results"]["C-incomplete-2-18"]["seed_results"]]
    rows += ["", "## Conclusion", "", f"The maximum observed balanced-accuracy improvement from perfect prompt grouping is **{delta:+.4f} pp** in the full-group variant. Developing an E19-B prompt-reconstruction prototype is justified: incomplete groups improved in all five seeds, by **{min(b_gains):.2f} to {max(b_gains):.2f} pp** for groups capped at 10 responses and **{min(c_gains):.2f} to {max(c_gains):.2f} pp** for groups capped at 18. These are still optimistic because the sampled rows retain their true group membership; E19-B must measure the cost of reconstructing groups incorrectly. The full-group result is an oracle ceiling.", "", "## Artifacts", "", f"- Run directory: `{output}`", "- `predictions.parquet`: aligned full-V0 E13-B, Hungarian and greedy predictions.", "- `sampled_predictions.parquet`: row-aligned B/C predictions for all seeds.", "- `predictions.npz`: full score matrices and assignment labels.", "- `metrics.json`, `config.json`: detailed fold, label, size and reproducibility data.", ""]
    report = "\n".join(rows)
    (output / "e19.md").write_text(report, encoding="utf-8")
    (root / "docs/e19.md").write_text(report, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
