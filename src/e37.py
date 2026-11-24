"""E37: length-aware and full-candidate E13-B reliability routers.

Uses only cached Trust List 2 v044 checkpoint probabilities. All correctness
statistics and CatBoost training records are cross-fitted within each outer
prompt-disjoint training partition; each response is classified independently.
"""
from __future__ import annotations

import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from . import e13_reliability_router as e13, ensemble_catalog as catalog
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e37"
V044 = "trust-list-2-v044"
E35_ROSTER = ROOT / "artifacts/analysis/e35/selection_results.json"
E13_RESULTS = ROOT / "artifacts/analysis/e13/20261120T093318893046Z/results.json"
METHODS = ("E37-A", "E37-B", "E37-C", "E37-D")
SEED = 2026
ALPHA = e13.SMOOTHING

TOP_FEATURES = tuple(e13.FEATURE_COLUMNS)
LENGTH_FEATURES = TOP_FEATURES + ("passage_word_count",)
FULL_CANDIDATE_FEATURES = (
    "checkpoint_id", "model_family", "candidate_label", "candidate_probability",
    "candidate_rank", "rank_historical_precision", "candidate_probability_margin",
    "ensemble_candidate_probability", "cross_family_agreement",
)
FULL_CANDIDATE_LENGTH_FEATURES = FULL_CANDIDATE_FEATURES + ("passage_word_count",)
TOP_CATEGORICAL = tuple(e13.CATEGORICAL)
FULL_CANDIDATE_CATEGORICAL = ("checkpoint_id", "model_family", "candidate_label")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _catalog_update(*, status: str, stage: str, results=None, error=None):
    path = ROOT / "configs/ensemble_catalog_v2.json"
    document = json.loads(path.read_text())
    entry = next(x for x in document["experiments"] if x["id"] == "E37")
    entry["status"], entry["current_stage"] = status, stage
    entry["progress_source"] = "artifacts/analysis/e37/evaluation_metrics.json"
    if error:
        entry["error"] = str(error)
    if results is not None:
        entry.pop("error", None)
        entry["status"] = "Complete"
        entry["current_stage"] = "Complete"
        entry["result_source"] = "artifacts/analysis/e37/evaluation_metrics.json"
        entry["balanced_accuracy"] = {
            variant: value["balanced_accuracy"]
            for variant, value in results["summary"].items()}
        entry["accuracy"] = {variant: value["accuracy"]
                              for variant, value in results["summary"].items()}
        entry["artifact_directory"] = "artifacts/analysis/e37"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n")
    tmp.replace(path)


def _load_inputs():
    v0 = catalog.cascades._v0(ROOT).reset_index(drop=True)
    _, members, probabilities, excluded, snapshot = catalog._load_inputs(
        ROOT, v0, version=V044)
    roster_doc = json.loads(E35_ROSTER.read_text())
    roster = list(map(str, roster_doc["final_roster"]))
    if len(roster) != 40 or len(set(roster)) != 40:
        raise ValueError(f"E35 fixed roster must contain 40 unique IDs; found {len(roster)}")
    missing = sorted(set(roster) - set(probabilities))
    if missing:
        raise ValueError(f"E35 roster is missing v044 cached probabilities: {missing}")
    if len(v0) != 2699 or v0.id.astype(str).duplicated().any():
        raise ValueError("E37 expects all 2,699 uniquely identified V0 responses")
    y = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if (y < 0).any():
        raise ValueError("V0 includes a label outside the canonical 18-label order")
    ids = v0.id.astype(str).to_numpy()
    prompts = v0.prompt_id.astype(str).to_numpy()
    texts = v0.text.fillna("").astype(str).to_numpy()
    tensor = np.stack([probabilities[sid] for sid in roster]).astype(np.float32)
    if tensor.shape != (40, 2699, len(LABELS)) or not np.isfinite(tensor).all():
        raise ValueError(f"Unexpected v044 probability tensor shape or values: {tensor.shape}")
    if not np.allclose(tensor.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError("A v044 probability row is not normalized")
    families = [_model_family(members[sid]["model"]) for sid in roster]
    e13_doc = json.loads(E13_RESULTS.read_text())
    e13_baseline = next(x for x in e13_doc["results"] if x["ensemble"] == "E13-B")
    index = {rid: i for i, rid in enumerate(ids)}
    folds = []
    held_all = []
    for number, fold in enumerate(e13_baseline["folds"], start=1):
        fit = np.asarray([index[str(x)] for x in fold["fit_ids"]], dtype=np.int64)
        held = np.asarray([index[str(x)] for x in fold["held_ids"]], dtype=np.int64)
        if set(prompts[fit]) & set(prompts[held]):
            raise ValueError(f"Prompt leakage in saved E13-B outer fold {number}")
        folds.append((fit, held))
        held_all.extend(held.tolist())
    if sorted(held_all) != list(range(len(v0))) or len(folds) != 5:
        raise ValueError("Saved E13-B folds must cover V0 exactly once across five folds")
    return v0, ids, prompts, texts, y, roster, families, tensor, members, snapshot, folds, excluded


def _word_counts(texts, rows):
    return np.asarray([len(str(texts[int(row)]).split()) for row in rows], dtype=np.float32)


def _top_features(tensor, roster, families, y, texts, fit_rows, target_rows,
                  *, include_length):
    frame, hard = e13.build_feature_frame(tensor, roster, families, y[fit_rows],
        fit_rows, target_rows, alpha=ALPHA)
    if include_length:
        counts = _word_counts(texts, target_rows)
        frame["passage_word_count"] = np.repeat(counts, len(roster))
    columns = LENGTH_FEATURES if include_length else TOP_FEATURES
    return frame.loc[:, columns], hard


def _rank_precision(tensor_fit, y_fit):
    """Smoothed P(correct candidate | checkpoint, candidate, candidate rank)."""
    cp, n_rows, n_labels = tensor_fit.shape
    ranks = np.argsort(np.argsort(-tensor_fit, axis=2, kind="stable"), axis=2, kind="stable")
    result = np.empty((cp, n_labels, n_labels), dtype=np.float32)
    denominators = np.zeros((cp, n_labels, n_labels), dtype=np.int32)
    correct = np.zeros_like(denominators)
    row_truth = np.asarray(y_fit, dtype=np.int64)
    for checkpoint in range(cp):
        for candidate in range(n_labels):
            candidate_ranks = ranks[checkpoint, :, candidate]
            for rank in range(n_labels):
                mask = candidate_ranks == rank
                den = int(mask.sum())
                tp = int(np.count_nonzero(row_truth[mask] == candidate))
                denominators[checkpoint, candidate, rank] = den
                correct[checkpoint, candidate, rank] = tp
                result[checkpoint, candidate, rank] = (tp + ALPHA) / (den + 2 * ALPHA)
    return {"rank_precision": result, "rank_support": denominators,
            "rank_correct": correct}


def _candidate_features(tensor, roster, families, y, texts, fit_rows, target_rows,
                        *, include_length):
    fit_p = np.asarray(tensor[:, fit_rows], dtype=np.float32)
    stats = _rank_precision(fit_p, y[fit_rows])
    p = np.asarray(tensor[:, target_rows], dtype=np.float32)
    n_checkpoints, n_rows, n_labels = p.shape
    ranks = np.argsort(np.argsort(-p, axis=2, kind="stable"), axis=2, kind="stable")
    cp_index = np.arange(n_checkpoints, dtype=np.int64)[:, None, None]
    candidate_index = np.arange(n_labels, dtype=np.int64)[None, None, :]
    rank_history = stats["rank_precision"][cp_index, candidate_index, ranks]
    ensemble = e13.family_balanced_probabilities(p, families)
    top = p.argmax(axis=2)
    family_agreement = np.zeros((n_checkpoints, n_rows, n_labels), dtype=np.float32)
    family_array = np.asarray(families, dtype=object)
    for checkpoint, own_family in enumerate(families):
        for family in sorted(set(families)):
            if family == own_family:
                continue
            for label in range(n_labels):
                family_agreement[checkpoint, :, label] += np.any(
                    top[family_array == family] == label, axis=0).astype(np.float32)
    max_other = np.max(np.where(np.eye(n_labels, dtype=bool)[None, None, :, :],
        -np.inf, p[:, :, None, :]), axis=3)
    margin = p - max_other
    shape = (n_rows, n_checkpoints, n_labels)

    def row_cp_label(array):
        return np.asarray(array).transpose(1, 0, 2).reshape(-1)

    data = {
        "checkpoint_id": np.tile(np.repeat(np.asarray(roster, dtype=str), n_labels), n_rows),
        "model_family": np.tile(np.repeat(np.asarray(families, dtype=str), n_labels), n_rows),
        "candidate_label": np.tile(np.asarray(LABELS, dtype=str), n_rows * n_checkpoints),
        "candidate_probability": row_cp_label(p),
        "candidate_rank": row_cp_label(ranks).astype(np.float32) + 1.0,
        "rank_historical_precision": row_cp_label(rank_history),
        "candidate_probability_margin": row_cp_label(margin),
        "ensemble_candidate_probability": np.repeat(ensemble[:, None, :], n_checkpoints, axis=1).reshape(-1),
        "cross_family_agreement": row_cp_label(family_agreement),
    }
    if include_length:
        data["passage_word_count"] = np.repeat(_word_counts(texts, target_rows),
                                                n_checkpoints * n_labels)
    frame = pd.DataFrame(data)
    for col in FULL_CANDIDATE_CATEGORICAL:
        frame[col] = frame[col].astype(str)
    columns = FULL_CANDIDATE_LENGTH_FEATURES if include_length else FULL_CANDIDATE_FEATURES
    numeric = [c for c in columns if c not in FULL_CANDIDATE_CATEGORICAL]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("E37 full-candidate features contain non-finite values")
    target = np.broadcast_to(np.asarray(y[target_rows], dtype=np.int64)[:, None, None] ==
        np.arange(n_labels, dtype=np.int64)[None, None, :], shape).reshape(-1).astype(np.int8)
    return frame.loc[:, columns], target, stats, ensemble


def _records(kind, tensor, roster, families, y, texts, prompts, rows,
             *, include_length):
    xs, ys, stats_meta = [], [], []
    for inner_fit_local, inner_valid_local in GroupKFold(n_splits=4).split(
            rows, y[rows], groups=prompts[rows]):
        inner_fit, inner_valid = rows[inner_fit_local], rows[inner_valid_local]
        if set(prompts[inner_fit]) & set(prompts[inner_valid]):
            raise AssertionError("E37 nested training features share prompt groups")
        if kind == "top":
            frame, hard = _top_features(tensor, roster, families, y, texts,
                inner_fit, inner_valid, include_length=include_length)
            target = e13._targets(hard, y[inner_valid])
            stats_meta.append({"fit_rows": int(len(inner_fit)),
                "feature_rows": int(len(frame)), "fit_row_ids": inner_fit.tolist()})
        else:
            frame, target, stats, _ = _candidate_features(tensor, roster, families,
                y, texts, inner_fit, inner_valid, include_length=include_length)
            stats_meta.append({"fit_rows": int(len(inner_fit)),
                "feature_rows": int(len(frame)), "rank_statistics": stats})
        xs.append(frame)
        ys.append(target)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), stats_meta


def _early_stop_split(rows, y, prompts):
    splits = list(GroupKFold(n_splits=4).split(rows, y[rows], groups=prompts[rows]))
    train_local, stop_local = splits[-1]
    train_rows, stop_rows = rows[train_local], rows[stop_local]
    if set(prompts[train_rows]) & set(prompts[stop_rows]):
        raise AssertionError("E37 early-stop groups overlap")
    return train_rows, stop_rows


def _save_nested_reliability_statistics(fold_number, outer_fit_rows, tensor, y,
                                        prompts, ids, roster, output_dir):
    """Persist each fitting-only reliability matrix used by cross-fit records."""
    early_train_rows, _ = _early_stop_split(outer_fit_rows, y, prompts)
    arrays = {"checkpoint_ids": np.asarray(roster),
              "labels": np.asarray(LABELS)}
    manifest = {"fold": int(fold_number), "training_inner_folds": {},
                "early_stopping_inner_folds": {}}
    for split_name, rows in (("early_stopping_inner_folds", early_train_rows),
                             ("training_inner_folds", outer_fit_rows)):
        for inner, (inner_fit_local, inner_valid_local) in enumerate(GroupKFold(n_splits=4).split(
                rows, y[rows], groups=prompts[rows]), start=1):
            fit_rows, valid_rows = rows[inner_fit_local], rows[inner_valid_local]
            if set(prompts[fit_rows]) & set(prompts[valid_rows]):
                raise AssertionError("Nested E37 reliability statistics leak prompt groups")
            top = e13.reliability_statistics(tensor[:, fit_rows].argmax(axis=2), y[fit_rows], ALPHA)
            candidate = _rank_precision(tensor[:, fit_rows], y[fit_rows])
            prefix = ("early" if split_name.startswith("early") else "train") + f"_inner{inner}_"
            for stat_name, value in top.items():
                arrays[prefix + "top_" + stat_name] = value
            for stat_name, value in candidate.items():
                arrays[prefix + "candidate_" + stat_name] = value
            manifest[split_name][str(inner)] = {
                "fit_rows": int(len(fit_rows)), "validation_rows": int(len(valid_rows)),
                "fit_row_ids_sha256": hashlib.sha256("\n".join(ids[fit_rows]).encode()).hexdigest(),
                "fit_prompt_group_count": int(len(np.unique(prompts[fit_rows]))),
                "validation_prompt_group_count": int(len(np.unique(prompts[valid_rows])))}
    path = Path(output_dir).resolve() / f"fold{fold_number}_nested_reliability_statistics.npz"
    np.savez_compressed(path, **arrays)
    manifest["statistics_path"] = str(path.relative_to(ROOT))
    manifest["statistics_sha256"] = _sha256(path)
    return manifest


def _new_router(iterations=None):
    params = dict(e13.CATBOOST_PARAMS)
    if iterations is not None:
        params["iterations"] = max(1, int(iterations))
        params.pop("od_type", None)
        params.pop("od_wait", None)
    return CatBoostClassifier(**params)


def _cat_frame(frame, categoricals):
    result = frame.copy()
    for column in categoricals:
        result[column] = result[column].fillna("").astype(str)
    return result


def _fit_model(early_X, early_y, stop_X, stop_y, train_X, train_y, columns, categoricals, path):
    early = _new_router()
    early.fit(_cat_frame(early_X.loc[:, columns], categoricals), early_y,
        cat_features=list(categoricals), eval_set=(_cat_frame(stop_X.loc[:, columns], categoricals), stop_y),
        early_stopping_rounds=e13.CATBOOST_PARAMS["od_wait"], verbose=False, use_best_model=True)
    best_iterations = max(1, int(early.get_best_iteration()) + 1)
    del early
    model = _new_router(best_iterations)
    model.fit(_cat_frame(train_X.loc[:, columns], categoricals), train_y,
              cat_features=list(categoricals), verbose=False)
    model.save_model(str(path))
    return model, best_iterations


def _top_reliability(model, frame, n_checkpoints, n_rows):
    return np.asarray(model.predict_proba(_cat_frame(frame, TOP_CATEGORICAL))[:, 1],
                      dtype=np.float64).reshape(n_rows, n_checkpoints).T


def _top_label_scores(hard, reliability, families, ensemble):
    n_checkpoints, n_rows = hard.shape
    scores = np.full((n_rows, len(LABELS)), -np.inf, dtype=np.float64)
    for row in range(n_rows):
        for label in range(len(LABELS)):
            voters = np.flatnonzero(hard[:, row] == label)
            if not len(voters):
                continue
            by_family = {}
            for cp in voters:
                by_family[families[cp]] = max(by_family.get(families[cp], -np.inf),
                                               reliability[cp, row])
            scores[row, label] = np.mean(list(by_family.values()))
    pred = np.empty(n_rows, dtype=np.int64)
    for row in range(n_rows):
        best = scores[row].max()
        tied = np.flatnonzero(np.isclose(scores[row], best, rtol=0, atol=1e-12))
        pred[row] = min(tied, key=lambda label: (-ensemble[row, label], int(label)))
    probability = np.zeros_like(scores)
    for row in range(n_rows):
        finite = np.isfinite(scores[row])
        z = scores[row, finite]
        exp = np.exp(z - z.max())
        probability[row, finite] = exp / exp.sum()
    return pred, scores, probability


def _candidate_reliability(model, frame, n_checkpoints, n_rows):
    return np.asarray(model.predict_proba(_cat_frame(frame, FULL_CANDIDATE_CATEGORICAL))[:, 1],
                      dtype=np.float64).reshape(n_rows, n_checkpoints, len(LABELS)).transpose(1, 0, 2)


def _candidate_family_scores(reliability, families, ensemble):
    unique = sorted(set(families))
    family_scores = [np.max(reliability[np.asarray(families) == family], axis=0)
                       for family in unique]
    scores = np.mean(family_scores, axis=0)
    pred = np.empty(scores.shape[0], dtype=np.int64)
    for row in range(scores.shape[0]):
        best = scores[row].max()
        tied = np.flatnonzero(np.isclose(scores[row], best, rtol=0, atol=1e-12))
        pred[row] = min(tied, key=lambda label: (-ensemble[row, label], int(label)))
    shifted = scores - scores.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    return pred, scores, probability


def _classification_metrics(y_true, prediction):
    cm = confusion_matrix(y_true, prediction, labels=np.arange(len(LABELS)))
    per_label = {}
    for label, name in enumerate(LABELS):
        tp = int(cm[label, label])
        support = int(cm[label].sum())
        predicted = int(cm[:, label].sum())
        per_label[name] = {"support": support,
            "precision": float(tp / predicted) if predicted else 0.0,
            "recall": float(tp / support) if support else 0.0,
            "true_positive": tp, "false_positive": int(predicted - tp),
            "false_negative": int(support - tp)}
    return {"rows": int(len(y_true)), "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "confusion_matrix": cm.tolist(), "per_label": per_label}


def _run_fold(number, fit_rows, held_rows, tensor, roster, families, y, prompts,
              texts, ids, outdir):
    fold_started = time.monotonic()
    train_rows, stop_rows = _early_stop_split(fit_rows, y, prompts)
    fitted, model_metadata, reliability_metadata = {}, {}, {}
    variant_specs = {
        "E37-A": ("top", False, TOP_FEATURES, TOP_CATEGORICAL),
        "E37-B": ("top", True, LENGTH_FEATURES, TOP_CATEGORICAL),
        "E37-C": ("candidate", False, FULL_CANDIDATE_FEATURES, FULL_CANDIDATE_CATEGORICAL),
        "E37-D": ("candidate", True, FULL_CANDIDATE_LENGTH_FEATURES, FULL_CANDIDATE_CATEGORICAL),
    }
    cache = {}
    for kind, include_length in (("top", False), ("top", True),
                                 ("candidate", False), ("candidate", True)):
        key = (kind, include_length)
        early_X, early_y, early_meta = _records(kind, tensor, roster, families,
            y, texts, prompts, train_rows, include_length=include_length)
        if kind == "top":
            stop_X, _ = _top_features(tensor, roster, families, y, texts,
                train_rows, stop_rows, include_length=include_length)
            stop_hard = tensor[:, stop_rows].argmax(axis=2)
            stop_y = e13._targets(stop_hard, y[stop_rows])
        else:
            stop_X, stop_y, stop_stats, _ = _candidate_features(tensor, roster,
                families, y, texts, train_rows, stop_rows,
                include_length=include_length)
            del stop_stats
        train_X, train_y, train_meta = _records(kind, tensor, roster, families,
            y, texts, prompts, fit_rows, include_length=include_length)
        cache[key] = {"early_X": early_X, "early_y": early_y,
            "stop_X": stop_X, "stop_y": stop_y,
            "train_X": train_X, "train_y": train_y,
            "records": {"early": early_meta, "train": train_meta}}

    outdir.mkdir(parents=True, exist_ok=True)
    fold_reliability = {}
    top_stats = e13.reliability_statistics(tensor[:, fit_rows].argmax(axis=2), y[fit_rows], ALPHA)
    fold_reliability["top"] = top_stats
    full_stats = _rank_precision(tensor[:, fit_rows], y[fit_rows])
    fold_reliability["candidate"] = full_stats
    for name, (kind, include_length, columns, categoricals) in variant_specs.items():
        data = cache[(kind, include_length)]
        model_path = outdir / f"{name.lower()}_fold{number}.cbm"
        model_started = time.monotonic()
        model, best_iterations = _fit_model(data["early_X"], data["early_y"],
            data["stop_X"], data["stop_y"], data["train_X"], data["train_y"],
            columns, categoricals, model_path)
        if kind == "top":
            held_X, held_hard = _top_features(tensor, roster, families, y,
                texts, fit_rows, held_rows, include_length=include_length)
            reliability = _top_reliability(model, held_X, len(roster), len(held_rows))
            ensemble = e13.family_balanced_probabilities(tensor[:, held_rows], families)
            pred, scores, probs = _top_label_scores(held_hard, reliability, families, ensemble)
            feature_schema = list(columns)
        else:
            held_X, _, held_stats, ensemble = _candidate_features(tensor, roster,
                families, y, texts, fit_rows, held_rows, include_length=include_length)
            reliability = _candidate_reliability(model, held_X, len(roster), len(held_rows))
            pred, scores, probs = _candidate_family_scores(reliability, families, ensemble)
            feature_schema = list(columns)
            del held_stats
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{name} fold {number} emitted invalid 18-label probabilities")
        fitted[name] = {"prediction": pred, "scores": scores, "probabilities": probs,
            "metrics": _classification_metrics(y[held_rows], pred)}
        model_metadata[name] = {"fold": number, "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": _sha256(model_path), "best_iterations": best_iterations,
            "training_rows": int(len(data["train_y"])),
            "early_stopping_rows": int(len(data["early_y"])),
            "held_rows": int(len(held_rows)), "fit_row_ids": ids[fit_rows].tolist(),
            "held_row_ids": ids[held_rows].tolist(),
            "fit_prompt_group_count": int(len(np.unique(prompts[fit_rows]))),
            "feature_schema": feature_schema,
            "training_seconds": float(time.monotonic() - model_started),
            "process_peak_memory_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
        del model, held_X, reliability
        print(f"E37 {name} fold {number}/5: BA={fitted[name]['metrics']['balanced_accuracy']:.4%}; "
              f"best_iter={best_iterations}", flush=True)
    np.savez_compressed(outdir / f"fold{number}_reliability_statistics.npz",
        top_counts=top_stats["counts"], top_support=top_stats["support"],
        top_precision=top_stats["precision"], top_recall=top_stats["recall"],
        top_directional=top_stats["directional"],
        candidate_rank_precision=full_stats["rank_precision"],
        candidate_rank_support=full_stats["rank_support"],
        candidate_rank_correct=full_stats["rank_correct"],
        fit_rows=fit_rows, checkpoint_ids=np.asarray(roster), labels=np.asarray(LABELS))
    for key, data in cache.items():
        del data["early_X"], data["stop_X"], data["train_X"]
    del cache
    gc.collect()
    return fitted, model_metadata, {"fit_row_count": len(fit_rows),
        "fit_rows_sha256": hashlib.sha256(fit_rows.tobytes()).hexdigest(),
        "top_stat_shape": list(top_stats["counts"].shape),
        "candidate_rank_precision_shape": list(full_stats["rank_precision"].shape),
        "fold_seconds": float(time.monotonic() - fold_started)}


def _write_report(result, path):
    lines = ["# E37 — Length-Aware Reliability and Full Candidate Coverage", "",
        "All four variants use the fixed 40-checkpoint E35 roster from Trust List 2 v044, cached probabilities, and identical saved E13-B five-fold prompt-disjoint splits. CatBoost settings are unchanged from E13-B. Each response is classified independently; no prompt-level decoding is used.", "",
        "| Experiment | Checkpoints | Balanced accuracy | Delta vs E37-A |", "|---|---:|---:|---:|"]
    baseline = result["summary"]["E37-A"]["balanced_accuracy"]
    for name in METHODS:
        metric = result["summary"][name]
        delta = "—" if name == "E37-A" else f"{100*(metric['balanced_accuracy']-baseline):+.4f} pp"
        lines.append(f"| {name} | 40 | {metric['balanced_accuracy']:.4%} | {delta} |")
    lines.extend(["", "## OOF metrics", "",
        "| Experiment | Accuracy | Balanced accuracy | Correct |", "|---|---:|---:|---:|"])
    for name in METHODS:
        metric = result["summary"][name]
        lines.append(f"| {name} | {metric['accuracy']:.4%} | {metric['balanced_accuracy']:.4%} | {metric['correct']}/2699 |")
    lines.extend(["", "## Outer folds", "",
        "| Experiment | Fold | Accuracy | Balanced accuracy | Trees | Seconds |", "|---|---:|---:|---:|---:|---:|"])
    for name in METHODS:
        for fold in result["folds"][name]:
            lines.append(f"| {name} | {fold['fold']} | {fold['accuracy']:.4%} | {fold['balanced_accuracy']:.4%} | {fold['best_iterations']} | {fold['training_seconds']:.1f} |")
    lines.extend(["", "## Feature definitions", "",
        "E37-A is E13-B's checkpoint correctness feature frame and family-max routing, using the fixed E35 roster. E37-B adds whitespace word count from original V0 response text. E37-C expands to all 18 labels for each checkpoint/response and predicts candidate correctness using checkpoint, family, candidate probability/rank, fold-fit rank-specific historical precision, candidate-vs-best-other probability margin, family-balanced ensemble candidate probability, and cross-family top-rank agreement. E37-D adds word count to E37-C. For C/D, candidate reliability is aggregated as the maximum checkpoint score within each family, averaged over families; the top label wins, with ensemble probabilities as the deterministic tie-breaker.", "",
        "Historical precision and CatBoost training features are computed from prompt-group-isolated fitting partitions only. Training records are four-fold cross-fitted within each outer fit partition; outer held labels are used only for evaluation. The E35 roster itself was developed using V0, so these are development estimates, not independent Kaggle estimates.", "",
        "## Per-label precision and recall", ""])
    for name in METHODS:
        lines.extend([f"### {name}", "", "| Label | Support | Precision | Recall | TP | FP |", "|---|---:|---:|---:|---:|---:|"])
        for label in LABELS:
            m = result["summary"][name]["per_label"][label]
            lines.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} | {m['true_positive']} | {m['false_positive']} |")
        lines.extend(["", "Confusion matrix, true labels by row and predictions by column:", "", "```text", "\t" + "\t".join(LABELS)])
        for label, row in zip(LABELS, result["summary"][name]["confusion_matrix"]):
            lines.append(label + "\t" + "\t".join(map(str, row)))
        lines.extend(["```", ""])
    lines.extend(["## Artifacts", "",
        "`artifacts/analysis/e37/oof_predictions.parquet` has one row per V0 response, the four predicted labels, and four aligned 18-label probability vectors. `evaluation_metrics.json` contains aggregate and fold results, including confusion matrices and per-label precision/recall. Each outer-fold CatBoost model is saved as a `.cbm`; `foldN_reliability_statistics.npz` saves outer-fit statistics and `foldN_nested_reliability_statistics.npz` saves the fitting-only statistics used to construct nested router training records. `feature_schemas.json`, `manifest.json`, and `configuration.json` record feature order, checkpoint revisions/cache hashes, labels, folds, seed, and the CatBoost configuration.", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run(*, root=ROOT, output_dir=OUT):
    root = Path(root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _catalog_update(status="Running", stage="Fitting four fold-local routers")
    started = time.monotonic()
    v0, ids, prompts, texts, y, roster, families, tensor, members, snapshot, folds, excluded = _load_inputs()
    assigned = {name: np.zeros(len(v0), dtype=np.int8) for name in METHODS}
    predictions = {name: np.full(len(v0), -1, dtype=np.int64) for name in METHODS}
    probabilities = {name: np.zeros((len(v0), len(LABELS)), dtype=np.float32) for name in METHODS}
    fold_results = {name: [] for name in METHODS}
    router_metadata = {name: [] for name in METHODS}
    fold_manifests = []
    _catalog_update(status="Running", stage="Inputs validated; outer fold 1 of 5")
    for fold_no, (fit_rows, held_rows) in enumerate(folds, start=1):
        fold_output = output_dir / f"fold{fold_no}"
        nested_stats = _save_nested_reliability_statistics(fold_no, fit_rows,
            tensor, y, prompts, ids, roster, fold_output)
        fitted, models, reliability = _run_fold(fold_no, fit_rows, held_rows,
            tensor, roster, families, y, prompts, texts, ids, fold_output)
        for name in METHODS:
            block = fitted[name]
            predictions[name][held_rows] = block["prediction"]
            probabilities[name][held_rows] = block["probabilities"]
            assigned[name][held_rows] += 1
            fold_results[name].append({"fold": fold_no, "rows": len(held_rows),
                **block["metrics"], **{k: v for k,v in models[name].items()
                    if k not in ("fit_row_ids", "held_row_ids", "feature_schema", "model_sha256")}})
            router_metadata[name].append(models[name])
        fold_manifests.append({"fold": fold_no, **reliability,
            "nested_reliability_statistics": nested_stats,
            "fit_prompt_groups": len(np.unique(prompts[fit_rows])),
            "held_prompt_groups": len(np.unique(prompts[held_rows])),
            "held_rows_sha256": hashlib.sha256(held_rows.tobytes()).hexdigest()})
        _catalog_update(status="Running", stage=f"Completed outer fold {fold_no} of 5")
    summary = {}
    for name in METHODS:
        if not np.all(assigned[name] == 1) or (predictions[name] < 0).any():
            raise ValueError(f"{name} OOF predictions do not cover every row once")
        summary[name] = _classification_metrics(y, predictions[name])
        summary[name]["correct"] = int(np.count_nonzero(y == predictions[name]))
        summary[name]["per_fold"] = [{k: f[k] for k in
            ("fold", "accuracy", "balanced_accuracy", "correct", "false_positive") if k in f}
            for f in fold_results[name]]
    frame = v0[["id", "prompt_id", "model"]].rename(columns={"model":"y_true"}).copy()
    for name in METHODS:
        frame[f"pred::{name}"] = np.asarray(LABELS, dtype=object)[predictions[name]]
        for label_no, label in enumerate(LABELS):
            frame[f"prob::{name}::{label}"] = probabilities[name][:, label_no]
    frame.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    np.savez_compressed(output_dir / "oof_probability_matrices.npz",
        row_ids=ids, labels=np.asarray(LABELS), **{
            name.replace("-", "_"): probabilities[name] for name in METHODS})
    (output_dir / "reliability_statistics_manifest.json").write_text(
        json.dumps(fold_manifests, indent=2) + "\n")
    schemas = {"E37-A": {"features": list(TOP_FEATURES), "categoricals": list(TOP_CATEGORICAL)},
        "E37-B": {"features": list(LENGTH_FEATURES), "categoricals": list(TOP_CATEGORICAL)},
        "E37-C": {"features": list(FULL_CANDIDATE_FEATURES), "categoricals": list(FULL_CANDIDATE_CATEGORICAL)},
        "E37-D": {"features": list(FULL_CANDIDATE_LENGTH_FEATURES), "categoricals": list(FULL_CANDIDATE_CATEGORICAL)}}
    (output_dir / "feature_schemas.json").write_text(json.dumps(schemas, indent=2) + "\n")
    metrics = {"experiment":"E37", "trust_list_version":V044,
        "checkpoint_count":40, "checkpoint_ids":roster, "families":families,
        "rows":len(v0), "labels":list(LABELS), "folds":fold_results,
        "summary":summary, "total_runtime_seconds":time.monotonic()-started,
        "process_peak_memory_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "excluded_from_v044_pool":excluded}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    config = {"trust_list_snapshot":snapshot, "trust_list_csv_sha256":_sha256(root/snapshot["csv_path"]),
        "v0_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "v0_text_sha256":hashlib.sha256("\0".join(texts).encode()).hexdigest(),
        "v0_label_sha256":hashlib.sha256(y.tobytes()).hexdigest(),
        "outer_folds_source":str(E13_RESULTS.relative_to(root)), "outer_folds_sha256":_sha256(E13_RESULTS),
        "roster_source":str(E35_ROSTER.relative_to(root)), "roster_file_sha256":_sha256(E35_ROSTER),
        "seed":SEED,"catboost":e13.CATBOOST_PARAMS,"smoothing_alpha":ALPHA,
        "inner_feature_folds":4,"outer_folds":5,
        "family_probability_aggregation":"mean checkpoints within family, then equal mean across families",
        "candidate_reliability_aggregation":"maximum per-family candidate correctness probability, then mean across families",
        "checkpoint_cache_sha256":{sid:members[sid]["cache_sha256"] for sid in roster},
        "checkpoint_model_ids":{sid:members[sid]["model"] for sid in roster},
        "checkpoint_families":dict(zip(roster,families)),"excluded":excluded}
    (output_dir / "configuration.json").write_text(json.dumps(config,indent=2)+"\n")
    manifest = {"experiment":"E37", "trust_list_version":V044,
        "roster":roster,"labels":list(LABELS),"fold_routers":router_metadata,
        "fold_reliability_statistics":"reliability_statistics_manifest.json",
        "nested_reliability_statistics":"foldN/foldN_nested_reliability_statistics.npz",
        "feature_schemas":"feature_schemas.json","oof_predictions":"oof_predictions.parquet",
        "oof_probability_matrices":"oof_probability_matrices.npz",
        "catboost":e13.CATBOOST_PARAMS,"seed":SEED,
        "validation_note":"V0 prompt-disjoint OOF; roster development used V0, so treat as development metrics."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    _write_report({"summary":summary,"folds":fold_results}, root/"docs/e37.md")
    _catalog_update(status="Complete", stage="Complete", results=metrics)
    return metrics


if __name__ == "__main__":
    try:
        final = run()
        print(json.dumps({name: {"accuracy": final["summary"][name]["accuracy"],
            "balanced_accuracy": final["summary"][name]["balanced_accuracy"]}
            for name in METHODS}, indent=2))
    except Exception as exc:
        _catalog_update(status="Failed", stage="Evaluation failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
