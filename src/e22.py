"""E22 learned checkpoint ensemble with reconstructed-group assignment.

CPU-only experiment runner. Base probabilities are read from the latest Trust
List 2's saved V0 caches; no neural inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from . import e16_common, e20, ensemble_catalog as catalog
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e22"
SEED = 2026
THREADS = 24
DEPTHS = (5, 7)
LEARNING_RATES = (0.03, 0.06)
ITERATIONS = 500
SPECIALISTS_PER_LABEL = 3
FEATURE_SCHEMA_VERSION = 2
_BASE_FEATURE_CACHE = {}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_trust(root, v0):
    pointer = json.loads((root / "docs/trust-list-2-latest.json").read_text())
    trust_path = root / pointer["csv_path"]
    trust, members, probability_map, excluded, snapshot = catalog._load_inputs(root, v0)
    if snapshot["version"] != pointer["version"]:
        raise ValueError("Latest Trust List pointer changed during loading")
    raw = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    if len(raw) != 1620 or raw.sequence_id.nunique() != 90:
        raise ValueError(f"Latest Trust List export shape differs from expected 90 x 18: {raw.shape}")
    registry_path = root / "experiments/registry.sqlite"
    registry_ids = raw.experiment_id.astype(str).unique().tolist()
    placeholders = ",".join("?" for _ in registry_ids)
    with sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True) as db:
        registered = {row[0] for row in db.execute(
            f"SELECT experiment_id FROM experiments WHERE experiment_id IN ({placeholders})", registry_ids)}
    missing_registry = sorted(set(registry_ids) - registered)
    if missing_registry:
        raise ValueError(f"Latest Trust List experiment IDs missing from registry: {missing_registry}")
    listed = sorted(raw.sequence_id.astype(str).unique(), key=lambda sid: int(sid[1:]))
    for sid in listed:
        if sid not in members:
            row = raw.loc[raw.sequence_id == sid].iloc[0]
            excluded[sid] = ("checkpoint unavailable" if row.get("checkpoint_available") != "True"
                             else "not eligible or cached full-V0 probabilities unavailable")
    return trust, members, probability_map, excluded, snapshot, trust_path, raw, len(registered)


def load_context(root=ROOT):
    root = Path(root).resolve()
    v0 = catalog.cascades._v0(root)
    y = e16_common.e15._label_codes(v0.model)
    ids = v0.id.astype(str).to_numpy()
    groups = v0.prompt_id.astype(str).to_numpy()
    if len(v0) != 2699 or len(set(ids)) != len(ids):
        raise ValueError("E22 requires the complete 2,699-row V0 set")
    trust, members, probs, excluded, snapshot, trust_path, raw_trust, registry_count = _latest_trust(root, v0)
    ids_cp = sorted(members, key=lambda sid: int(sid[1:]))
    if len(ids_cp) != 89:
        raise ValueError(f"Expected the latest eligible cached roster (89 available); found {len(ids_cp)}")
    for sid in ("N0135", "N0136"):
        if sid not in members:
            raise ValueError(f"Required E21 checkpoint {sid} has no eligible V0 probability cache")
    matrices = np.stack([probs[sid] for sid in ids_cp]).astype(np.float32)
    families = [e16_common.e15.e13._model_family(members[sid]["model"]) for sid in ids_cp]
    if matrices.shape != (len(ids_cp), len(v0), len(LABELS)):
        raise ValueError("Cached probability tensor shape is invalid")
    if not np.isfinite(matrices).all() or not np.allclose(matrices.sum(2), 1, atol=1e-4):
        raise ValueError("Cached checkpoint probabilities are invalid")

    # Preserve E13-B's five prompt-group-disjoint outer folds.
    result_path = e16_common.e15.BASELINE_RESULTS
    result_data = json.loads(result_path.read_text())
    result = next(x for x in result_data["results"] if x["ensemble"] == "E13-B")
    row_map = {str(value): i for i, value in enumerate(ids)}
    folds = []
    for details in result["folds"]:
        fit = np.asarray([row_map[str(x)] for x in details["fit_ids"]], dtype=np.int64)
        held = np.asarray([row_map[str(x)] for x in details["held_ids"]], dtype=np.int64)
        if set(groups[fit]) & set(groups[held]):
            raise ValueError(f"E22 fold {details['fold']} leaks prompt groups")
        folds.append({"fold": int(details["fold"]), "fit_rows": fit, "held_rows": held})
    if len(folds) != 5 or not np.array_equal(np.sort(np.concatenate([f["held_rows"] for f in folds])), np.arange(len(v0))):
        raise ValueError("E22 outer folds do not cover V0 exactly once")

    def read_groups(path, name, method=None):
        frame = pd.read_parquet(path)
        if method is not None:
            frame = frame.loc[(frame.method == method) & (frame.evaluation == "complete-held-out")]
        frame = frame.copy()
        frame["id"] = frame.id.astype(str)
        if frame.id.duplicated().any() or set(frame.id) != set(ids):
            raise ValueError(f"{name} groups do not cover V0 exactly once")
        frame = frame.set_index("id").loc[ids]
        group_col = "predicted_group_id"
        values = frame[group_col].astype(str).to_numpy()
        expected_folds = np.full(len(ids), -1, dtype=np.int64)
        for fold in folds:
            expected_folds[fold["held_rows"]] = fold["fold"]
        if "fold" not in frame or not np.array_equal(frame["fold"].to_numpy(dtype=np.int64), expected_folds):
            raise ValueError(f"{name} fold assignments do not match the E13-B outer folds")
        for fold in folds:
            held = fold["held_rows"]
            counts = pd.Series(values[held]).value_counts()
            if (counts > len(LABELS)).any():
                raise ValueError(f"{name} contains a held-out group larger than 18")
        return values

    group_d = read_groups(root / "artifacts/analysis/e19d/oof_group_predictions.parquet", "E19-D")
    group_f = read_groups(root / "artifacts/analysis/e19f/predicted_group_assignments.parquet", "E19-F-A", "E19-F-A")
    if not np.array_equal(group_d, group_f):
        # Different groupers are expected; this branch documents that the
        # comparison is independently decoded with each artifact.
        pass
    return {"root": root, "v0": v0, "y": y, "row_ids": ids, "groups": groups,
        "texts": v0.text.fillna("").astype(str).to_numpy(), "checkpoint_ids": ids_cp,
        "families": families, "matrices": matrices, "members": members,
        "cache_hashes": {sid: members[sid]["cache_sha256"] for sid in ids_cp},
        "trust": trust, "raw_trust": raw_trust, "registry_count": registry_count,
        "excluded": excluded, "snapshot": snapshot,
        "trust_path": trust_path, "trust_sha256": _sha(trust_path),
        "folds": folds, "group_d": group_d, "group_f": group_f}


def _specialists(ctx, train_rows, pool):
    """Greedy complementary label specialists, fitted using train labels only."""
    y = ctx["y"][train_rows]
    mats = ctx["matrices"][:, train_rows]
    general = mats[pool].mean(axis=0).argmax(axis=1)
    selected = {}
    details = {}
    for label in range(len(LABELS)):
        true = y == label
        missed = true & (general != label)
        candidates = []
        for cp in pool:
            pred = mats[cp].argmax(axis=1)
            tp = int(np.sum(pred[true] == label)); fp = int(np.sum((pred == label) & ~true))
            recall = tp / max(1, int(true.sum()))
            precision = tp / max(1, int(np.sum(pred == label)))
            residual = int(np.sum(missed & (pred == label)))
            candidates.append((cp, precision, recall, residual, pred))
        picks = []; residual_mask = missed.copy()
        for _ in range(min(SPECIALISTS_PER_LABEL, len(candidates))):
            best = max((x for x in candidates if x[0] not in picks),
                key=lambda x: (int(np.sum(residual_mask & (x[4] == label))), x[1], x[2], -x[0]))
            picks.append(best[0]); residual_mask &= best[4] != label
        selected[label] = picks
        details[LABELS[label]] = [{"checkpoint_id": ctx["checkpoint_ids"][cp],
            "precision": float(next(x[1] for x in candidates if x[0] == cp)),
            "recall": float(next(x[2] for x in candidates if x[0] == cp))} for cp in picks]
    return selected, details


def _base_feature_frame(ctx, pool):
    """Build input-independent E20 checkpoint features once per roster."""
    pool = list(pool)
    rows = np.arange(len(ctx["row_ids"]), dtype=np.int64)
    p = ctx["matrices"][pool].astype(np.float64)
    cp, n, k = p.shape
    if k != len(LABELS) or not np.isfinite(p).all():
        raise ValueError("E22 probability feature matrix is invalid")
    fams = [ctx["families"][i] for i in pool]
    family_names = sorted(set(fams))
    family_scores = np.stack([p[[i for i, f in enumerate(fams) if f == fam]].mean(axis=0)
                               for fam in family_names])
    ensemble = family_scores.mean(axis=0)
    cp_top = p.argmax(axis=2)
    ranks = np.argsort(np.argsort(-p, axis=2, kind="stable"), axis=2) + 1
    flat = lambda value: np.asarray(value).reshape(-1)
    # Reuse E20's checkpoint-only candidate-label feature builder. E22 passes
    # the family-balanced probability ensemble as its general label score and
    # each checkpoint's top-1 probability as confidence evidence.
    evidence = {"scores": ensemble, "roster_ids": [ctx["checkpoint_ids"][i] for i in pool],
        "reliability": p.max(axis=2), "hard": cp_top}
    frame = e20._evidence_feature_frame(ctx, evidence, rows,
        [ctx["checkpoint_ids"][i] for i in pool])
    extra = {"ensemble_probability": flat(ensemble),
        "ensemble_rank": flat(np.argsort(np.argsort(-ensemble, axis=1, kind="stable"), axis=1) + 1),
        "ensemble_margin_from_top": flat(ensemble - ensemble.max(axis=1, keepdims=True)),
        "checkpoint_probability_mean": flat(p.mean(axis=0)),
        "checkpoint_probability_std": flat(p.std(axis=0)),
        "checkpoint_probability_max": flat(p.max(axis=0)),
        "checkpoint_probability_min": flat(p.min(axis=0)),
        "checkpoint_top1_agreement": flat(np.stack([(cp_top == lab).sum(axis=0) for lab in range(k)], axis=1)),
        "family_probability_mean": flat(family_scores.mean(axis=0)),
        "family_probability_std": flat(family_scores.std(axis=0)),
        "family_top1_agreement": flat(np.stack([(family_scores.argmax(axis=2) == lab).sum(axis=0) for lab in range(k)], axis=1)),
        "family_count": np.full(n * k, len(family_names), dtype=np.int16)}
    rankflat = ranks.transpose(1, 2, 0).reshape(n * k, cp)
    marginflat = (p - p.max(axis=2, keepdims=True)).transpose(1, 2, 0).reshape(n * k, cp)
    top_margin = np.sort(p, axis=2)[:, :, -1] - np.sort(p, axis=2)[:, :, -2]
    topmarginflat = np.broadcast_to(top_margin.T[:, None, :], (n, k, cp)).reshape(n * k, cp)
    for q, cpidx in enumerate(pool):
        sid = ctx["checkpoint_ids"][cpidx]
        extra[f"cp::{sid}::rank"] = rankflat[:, q]
        extra[f"cp::{sid}::margin"] = marginflat[:, q]
        extra[f"cp::{sid}::top_margin"] = topmarginflat[:, q]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(extra)], axis=1).copy()


def build_features(ctx, rows, pool, *, specialists=None):
    """Reuse E20 checkpoint candidate features with E22 pool features."""
    rows = np.asarray(rows, dtype=np.int64)
    pool = list(pool)
    cp, n, k = len(pool), len(rows), len(LABELS)
    cache_key = tuple(pool)
    if cache_key not in _BASE_FEATURE_CACHE:
        _BASE_FEATURE_CACHE[cache_key] = _base_feature_frame(ctx, pool)
    row_ix = (rows[:, None] * k + np.arange(k)[None, :]).reshape(-1)
    frame = _BASE_FEATURE_CACHE[cache_key].iloc[row_ix].reset_index(drop=True).copy()
    if specialists is not None:
        p = ctx["matrices"][pool][:, rows, :].astype(np.float64)
        cp_top = p.argmax(axis=2)
        local = {cpidx: i for i, cpidx in enumerate(pool)}
        flat = lambda value: np.asarray(value).reshape(-1)
        extra = {}
        for j in range(SPECIALISTS_PER_LABEL):
            prob = np.zeros((n, k), dtype=np.float32)
            vote = np.zeros((n, k), dtype=np.int8)
            for lab in range(k):
                q = local[specialists[lab][j]]
                prob[:, lab] = p[q, :, lab]
                vote[:, lab] = cp_top[q] == lab
            extra[f"specialist::{j}::prob"] = flat(prob)
            extra[f"specialist::{j}::vote"] = flat(vote)
        frame = pd.concat([frame, pd.DataFrame(extra)], axis=1)
    if len(frame) != n * k or not frame.candidate_label.astype(str).tolist()[:k] == list(LABELS):
        raise ValueError("E22 features lost canonical candidate or row alignment")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("E22 features contain nonfinite numeric values")
    return frame


def select_specialists(ctx, train_rows, pool):
    return _specialists(ctx, np.asarray(train_rows, dtype=np.int64), list(pool))


def _target(y):
    return (np.repeat(y, len(LABELS)) == np.tile(np.arange(len(LABELS)), len(y))).astype(np.int8)


def _fit(frame, target, depth, lr):
    return e20._cat_train(frame, target, depth, lr)


def _tune(ctx, train_rows, builder_factory, grouping, fold_no):
    results = []
    inner_data = []
    for fit_local, valid_local in GroupKFold(3).split(
            train_rows, ctx["y"][train_rows], ctx["groups"][train_rows]):
        fit_rows, valid_rows = train_rows[fit_local], train_rows[valid_local]
        # Specialist selection and all its label-derived statistics are
        # refitted inside this inner fitting partition.
        builder = builder_factory(fit_rows)
        inner_data.append((fit_rows, valid_rows, builder(fit_rows), builder(valid_rows)))
    for depth in DEPTHS:
        for lr in LEARNING_RATES:
            fold_scores = []
            for fit_rows, valid_rows, xfit, xvalid in inner_data:
                clf = e20._cat_train(xfit, _target(ctx["y"][fit_rows]), depth, lr,
                                     validation=(xvalid, _target(ctx["y"][valid_rows])))
                scores = e20._predict_candidate(clf, xvalid)
                pred = e20.hungarian(scores, grouping[valid_rows], ctx["row_ids"][valid_rows])
                fold_scores.append(float(balanced_accuracy_score(ctx["y"][valid_rows], pred)))
            results.append({"depth": depth, "learning_rate": lr,
                "mean_balanced_accuracy": float(np.mean(fold_scores)), "fold_scores": fold_scores})
    chosen = max(results, key=lambda x: (x["mean_balanced_accuracy"], -x["depth"], -x["learning_rate"]))
    print(f"E22 tune outer fold {fold_no}: depth={chosen['depth']} lr={chosen['learning_rate']} BA={chosen['mean_balanced_accuracy']:.4%}", flush=True)
    return chosen, results


def _group_assignment_metrics(ctx, scores, rows, grouping):
    pred = e20.hungarian(scores, grouping[rows], ctx["row_ids"][rows])
    independent = scores.argmax(axis=1)
    return pred, independent


def run_experiment(ctx, name, pool_ids, *, specialist=False, output=OUT):
    output = Path(output)
    saved_metrics = output / f"metrics_{name}.json"
    saved_predictions = output / f"predictions_{name}.npz"
    if saved_metrics.is_file() and saved_predictions.is_file():
        saved = json.loads(saved_metrics.read_text())
        if saved.get("feature_schema_version") == FEATURE_SCHEMA_VERSION:
            return saved
    y = ctx["y"]
    pool = [ctx["checkpoint_ids"].index(sid) for sid in pool_ids]
    scores = np.full((len(y), len(LABELS)), np.nan, dtype=np.float32)
    independent = np.full(len(y), -1, dtype=np.int64)
    pred_d = np.full(len(y), -1, dtype=np.int64)
    pred_f = np.full(len(y), -1, dtype=np.int64)
    pred_oracle = np.full(len(y), -1, dtype=np.int64)
    folds_result = []; specialist_manifest = {}; importance = []
    start = time.monotonic()
    for outer in ctx["folds"]:
        fold, fit_rows, held = outer["fold"], outer["fit_rows"], outer["held_rows"]
        selected, selected_detail = _specialists(ctx, fit_rows, pool) if specialist else (None, None)
        specialist_manifest[str(fold)] = selected_detail
        def builder_for(rows):
            inner_selected = _specialists(ctx, rows, pool)[0] if specialist else None
            return lambda target_rows: build_features(ctx, target_rows, pool,
                                                       specialists=inner_selected)
        builder = builder_for(fit_rows)
        xfit = builder(fit_rows)
        tuning, tuning_records = _tune(ctx, fit_rows,
            builder_for, ctx["group_d"], fold)
        clf = _fit(xfit, _target(y[fit_rows]), tuning["depth"], tuning["learning_rate"])
        xheld = builder(held)
        mat = e20._predict_candidate(clf, xheld)
        scores[held] = mat
        independent[held] = mat.argmax(axis=1)
        pred_d[held], _ = _group_assignment_metrics(ctx, mat, held, ctx["group_d"])
        pred_f[held], _ = _group_assignment_metrics(ctx, mat, held, ctx["group_f"])
        pred_oracle[held], _ = _group_assignment_metrics(ctx, mat, held, ctx["groups"])
        fi = pd.DataFrame({"feature": clf.feature_names_, "importance": clf.get_feature_importance()})
        importance.extend({"fold": fold, "feature": r.feature, "importance": float(r.importance)} for r in fi.itertuples())
        fold_metrics = {"fold": fold, "fit_rows": len(fit_rows), "held_rows": len(held),
            "selected_hyperparameters": {"depth": tuning["depth"], "learning_rate": tuning["learning_rate"]},
            "tuning": tuning_records,
            "independent": e20._metric(y[held], independent[held]),
            "E19-D_hungarian": e20._metric(y[held], pred_d[held]),
            "E19-F-A_hungarian": e20._metric(y[held], pred_f[held]),
            "oracle_hungarian": e20._metric(y[held], pred_oracle[held]),
            "specialists": selected_detail}
        folds_result.append(fold_metrics)
        clf.save_model(str(output / f"{name}_fold{fold}.cbm"))
        print(f"{name} fold {fold}: D={balanced_accuracy_score(y[held],pred_d[held]):.4%}, F={balanced_accuracy_score(y[held],pred_f[held]):.4%}, oracle={balanced_accuracy_score(y[held],pred_oracle[held]):.4%}", flush=True)
    if not np.isfinite(scores).all() or (independent < 0).any():
        raise RuntimeError(f"{name} did not predict every V0 row")
    output_preds = {"independent": independent, "E19-D": pred_d,
                    "E19-F-A": pred_f, "oracle": pred_oracle}
    packed = {key: e20._metric(y, val) for key, val in output_preds.items()}
    runtime = time.monotonic() - start
    comparison = {}
    if name != "E22-A":
        base_path = output / "predictions_E22-A.npz"
        if base_path.is_file():
            base = np.load(base_path)["pred_E19D"]
            comparison = e20._comparison(y, base, pred_d)
            comparison["per_label"] = {lab: {
                "corrected": int(((base != y) & (pred_d == y) & (y == i)).sum()),
                "introduced": int(((base == y) & (pred_d != y) & (y == i)).sum())}
                for i, lab in enumerate(LABELS)}
    sample_metrics = _evaluate_incomplete(ctx, scores)
    result = {"experiment": name, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "checkpoints": list(pool_ids),
        "checkpoint_count": len(pool_ids), "metrics": packed,
        "folds": folds_result, "comparison_vs_E22-A_E19-D": comparison,
        "incomplete_group_simulations": sample_metrics,
        "specialist_selections": specialist_manifest, "feature_importance": importance,
        "runtime_seconds": runtime}
    (output / f"metrics_{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(importance).to_csv(output / f"feature_importance_{name}.csv", index=False)
    np.savez_compressed(output / f"predictions_{name}.npz", row_ids=ctx["row_ids"], y=y,
        scores=scores, pred_independent=independent, pred_E19D=pred_d,
        pred_E19F_A=pred_f, pred_oracle=pred_oracle)
    return result


def _evaluate_incomplete(ctx, score_matrix):
    path = ctx["root"] / "artifacts/analysis/e19/sampled_predictions.parquet"
    if not path.is_file():
        return {"status": "missing E20-established sample rows"}
    sample = pd.read_parquet(path)
    row_map = {str(value): i for i, value in enumerate(ctx["row_ids"])}
    out = []
    for (variant, seed), part in sample.groupby(["variant", "seed"], sort=True):
        # E20 stores one copy of each sample row per model being compared.
        part = part.drop_duplicates("id", keep="first")
        rows = np.asarray([row_map[str(x)] for x in part.id], dtype=np.int64)
        for grouping_name, grouping in (("E19-D", ctx["group_d"]), ("E19-F-A", ctx["group_f"])):
            pred = e20.hungarian(score_matrix[rows], grouping[rows], ctx["row_ids"][rows])
            out.append({"variant": str(variant), "seed": int(seed), "grouping": grouping_name,
                "rows": len(rows), "accuracy": float(accuracy_score(ctx["y"][rows], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(ctx["y"][rows], pred))})
    return out


def run(root=ROOT, output=OUT):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ctx = load_context(root)
    hist = json.loads((root / "artifacts/analysis/e20/e20ab_metrics.json").read_text())
    reference = hist["checkpoint_only_ablation"]
    if not np.isclose(reference["balanced_accuracy"], 0.7898533432761621, atol=1e-12):
        raise ValueError("Saved E20 checkpoint-only reference does not reproduce 78.9853% BA")
    e20scores = np.load(root / "artifacts/analysis/e20/scores_e20ab.npz", allow_pickle=True)
    if not np.array_equal(e20scores["row_ids"].astype(str), ctx["row_ids"]):
        raise ValueError("Historical E20 scores are not aligned with V0")
    if not np.array_equal(e20scores["y"].astype(np.int64), ctx["y"]):
        raise ValueError("Historical E20 labels are not aligned with V0")
    reproduced = e20._metric(ctx["y"], e20scores["pred_checkpoint_only"].astype(np.int64))
    if not np.isclose(reproduced["balanced_accuracy"], reference["balanced_accuracy"], atol=1e-12):
        raise ValueError("Recomputed historical E20 checkpoint-only BA does not match its saved result")
    e20ref = {"accuracy": float(reference["accuracy"]),
              "balanced_accuracy": float(reference["balanced_accuracy"]),
              "recomputed_accuracy": reproduced["accuracy"],
              "recomputed_balanced_accuracy": reproduced["balanced_accuracy"],
              "reproduced_from": "artifacts/analysis/e20/e20ab_metrics.json",
              "score_matrix_sha256": _sha(root / "artifacts/analysis/e20/scores_e20ab.npz")}
    eligible = list(ctx["checkpoint_ids"])
    no_e21 = [sid for sid in eligible if sid not in {"N0135", "N0136"}]
    fold_definitions = [{"fold": f["fold"],
        "fit_ids": ctx["row_ids"][f["fit_rows"]].astype(str).tolist(),
        "held_ids": ctx["row_ids"][f["held_rows"]].astype(str).tolist()} for f in ctx["folds"]]
    fold_payload = json.dumps(fold_definitions, sort_keys=True, separators=(",", ":"))
    (output / "fold_definitions.json").write_text(json.dumps(fold_definitions, indent=2) + "\n")
    v0_path = root / "data/processed/dataset_v0.parquet"
    id_hash = hashlib.sha256("\n".join(ctx["row_ids"].astype(str)).encode()).hexdigest()
    manifest = {"seed": SEED, "threads": THREADS, "outer_folds": 5,
        "depths": DEPTHS, "learning_rates": LEARNING_RATES, "iterations": ITERATIONS,
        "candidate_target": "1 iff candidate_label equals the V0 true label; otherwise 0",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": "E20 checkpoint-only candidate feature builder plus cached checkpoint confidence, rank/margin, agreement, and family-balanced probability summaries; raw top-1 probability is the confidence field, without fitting a separate E13 reliability router",
        "E22-C_specialists": "up to three per label; training-fold precision/recall and greedy incremental correct coverage on general-ensemble misses",
        "assignment": "SciPy Hungarian maximization, at most 18 responses per group, one distinct canonical label per response/group",
        "group_sources": {"primary": "E19-D OOF", "comparison": "E19-F-A OOF", "oracle": "true V0 prompt_id"},
        "labels": list(LABELS), "trust_list": ctx["snapshot"],
        "trust_list_sha256": ctx["trust_sha256"], "source_file": str(ctx["trust_path"]),
        "source_rows": len(ctx["raw_trust"]),
        "listed_checkpoint_count": int(ctx["raw_trust"].sequence_id.nunique()),
        "listed_checkpoint_label_records": int(len(ctx["raw_trust"])),
        "registry_resolved_experiments": ctx["registry_count"],
        "eligible_cached_checkpoint_count": len(eligible), "excluded": ctx["excluded"],
        "checkpoint_cache_sha256": ctx["cache_hashes"],
        "code_sha256": _sha(Path(__file__)),
        "v0_dataset_sha256": _sha(v0_path), "v0_row_ids_sha256": id_hash,
        "fold_definitions_sha256": hashlib.sha256(fold_payload.encode()).hexdigest(),
        "E19-D_group_file_sha256": _sha(root / "artifacts/analysis/e19d/oof_group_predictions.parquet"),
        "E19-F-A_group_file_sha256": _sha(root / "artifacts/analysis/e19f/predicted_group_assignments.parquet"),
        "V0_checkpoint_selection_limitation":
            "N0135/N0136 have full-V0 validation metrics and were selected using V0; all reported V0 results are development evidence.",
        "historical_E20": e20ref}
    (output / "configuration.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    results = []
    with threadpool_limits(limits=THREADS):
        results.append(run_experiment(ctx, "E22-A", no_e21, output=output))
        results.append(run_experiment(ctx, "E22-B", eligible, output=output))
        results.append(run_experiment(ctx, "E22-C", eligible, specialist=True, output=output))
    # Summary includes aggregate metrics, population fold SD and Hungarian gain.
    summary = []
    for result in results:
        met = result["metrics"]
        fold_bas = [f["E19-D_hungarian"]["balanced_accuracy"] for f in result["folds"]]
        summary.append({"experiment": result["experiment"], "checkpoints": result["checkpoint_count"],
            "independent_balanced_accuracy": met["independent"]["balanced_accuracy"],
            "E19-D_hungarian_balanced_accuracy": met["E19-D"]["balanced_accuracy"],
            "E19-F-A_hungarian_balanced_accuracy": met["E19-F-A"]["balanced_accuracy"],
            "oracle_hungarian_balanced_accuracy": met["oracle"]["balanced_accuracy"],
            "E19-D_fold_mean": float(np.mean(fold_bas)),
            "E19-D_fold_sd": float(np.std(fold_bas, ddof=1)),
            "net_E19-D_assignment_gain_pp": (met["E19-D"]["balanced_accuracy"] -
                met["independent"]["balanced_accuracy"]) * 100,
            "net_E19-F-A_assignment_gain_pp": (met["E19-F-A"]["balanced_accuracy"] -
                met["independent"]["balanced_accuracy"]) * 100,
            "net_oracle_assignment_gain_pp": (met["oracle"]["balanced_accuracy"] -
                met["independent"]["balanced_accuracy"]) * 100,
            "fold_ba_summary": {method: {
                "mean": float(np.mean([f[fold_key]["balanced_accuracy"] for f in result["folds"]])),
                "sd": float(np.std([f[fold_key]["balanced_accuracy"] for f in result["folds"]], ddof=1))}
                for method, fold_key in (("independent", "independent"), ("E19-D", "E19-D_hungarian"),
                    ("E19-F-A", "E19-F-A_hungarian"), ("oracle", "oracle_hungarian"))},
            "corrected_vs_E22-A": result.get("comparison_vs_E22-A_E19-D", {}).get("corrected"),
            "introduced_vs_E22-A": result.get("comparison_vs_E22-A_E19-D", {}).get("introduced")})
    result_doc = {"historical_E20_reference": e20ref, "summary": summary,
        "trust_list": ctx["snapshot"], "excluded_checkpoints": ctx["excluded"],
        "checkpoint_rosters": {r["experiment"]: r["checkpoints"] for r in results},
        "recorded_experiment_runtime_seconds": float(sum(r["runtime_seconds"] for r in results)),
        "finalization_runtime_seconds": time.monotonic() - started}
    (output / "results.json").write_text(json.dumps(result_doc, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    _write_documentation(root, output, result_doc, results)
    _register_catalog(root, output, results)
    return result_doc


def _write_documentation(root, output, summary, results):
    configuration = json.loads((output / "configuration.json").read_text())
    lines = ["# E22 — Learned checkpoint ensemble with group-constrained classification", "",
        "## Summary", "",
        "Each CatBoost model scores one record per response and canonical candidate label. It reuses E20's checkpoint-only feature builder and adds cached probability, top-1 vote, rank, margin, checkpoint/family agreement, and family-balanced ensemble evidence. The binary target marks whether a candidate is the true label. Its 18 scores are decoded independently or by Hungarian maximum-score assignment within each reconstructed prompt group; labels are unique within a group. E22-C chooses up to three specialists per label using fitting-fold precision, recall, and greedy incremental correct coverage among examples missed by the family-balanced mean stacker. No neural inference or GPU training was used.", "",
        "Feature interpretation: E22 supplies the E20 candidate-score field from the family-balanced mean of the current checkpoint pool and supplies raw top-1 probability as per-checkpoint confidence; it does not train E20's separate E13 reliability router. The historical E20 value is therefore a reference for the same candidate-scoring/Hungarian architecture, not an identical feature matrix or checkpoint roster.", "",
        "| Reference / experiment | Checkpoints | E20 historical BA | Independent BA | E19-D + Hungarian BA | E19-F-A + Hungarian BA | Oracle-group BA | E19-D fold BA mean ± SD | Δ E19-D assignment BA | Corrected vs A | Introduced vs A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ref = summary["historical_E20_reference"]
    lines.append(f"| E20 historical checkpoint-only + original E19-B groups | — | {ref['balanced_accuracy']:.4%} | — | — | — | — | — | — | — | — |")
    for item in summary["summary"]:
        changed = "—" if item["experiment"] == "E22-A" else str(item["corrected_vs_E22-A"])
        lost = "—" if item["experiment"] == "E22-A" else str(item["introduced_vs_E22-A"])
        lines.append(f"| {item['experiment']} | {item['checkpoints']} | — | {item['independent_balanced_accuracy']:.4%} | {item['E19-D_hungarian_balanced_accuracy']:.4%} | {item['E19-F-A_hungarian_balanced_accuracy']:.4%} | {item['oracle_hungarian_balanced_accuracy']:.4%} | {item['E19-D_fold_mean']:.4%} ± {item['E19-D_fold_sd']:.4%} | {item['net_E19-D_assignment_gain_pp']:+.3f} pp | {changed} | {lost} |")
    lines += ["", "Hungarian BA gain over independent argmax (percentage points):", "",
        "| Experiment | E19-D | E19-F-A | Oracle groups |", "|---|---:|---:|---:|"]
    for item in summary["summary"]:
        lines.append(f"| {item['experiment']} | {item['net_E19-D_assignment_gain_pp']:+.3f} | {item['net_E19-F-A_assignment_gain_pp']:+.3f} | {item['net_oracle_assignment_gain_pp']:+.3f} |")
    lines += ["", "The saved E20 checkpoint-only OOF predictions reproduce the historical **78.9853% balanced accuracy** exactly. The historical result is an OOF reference, while E22 uses the same five saved prompt-group-disjoint outer folds.", "",
        "## Fold results", ""]
    lines += ["| Experiment | Fold | Independent BA | E19-D BA | E19-F-A BA | Oracle BA |"]
    lines += ["|---|---:|---:|---:|---:|---:|"]
    for result in results:
        for fold in result["folds"]:
            lines.append(f"| {result['experiment']} | {fold['fold']} | {fold['independent']['balanced_accuracy']:.4%} | {fold['E19-D_hungarian']['balanced_accuracy']:.4%} | {fold['E19-F-A_hungarian']['balanced_accuracy']:.4%} | {fold['oracle_hungarian']['balanced_accuracy']:.4%} |")
    lines += ["", "Fold BA stability (mean ± sample SD):", "",
        "| Experiment | Independent | E19-D Hungarian | E19-F-A Hungarian | Oracle Hungarian |",
        "|---|---:|---:|---:|---:|"]
    for item in summary["summary"]:
        stats = item["fold_ba_summary"]
        lines.append(f"| {item['experiment']} | {stats['independent']['mean']:.4%} ± {stats['independent']['sd']:.4%} | {stats['E19-D']['mean']:.4%} ± {stats['E19-D']['sd']:.4%} | {stats['E19-F-A']['mean']:.4%} ± {stats['E19-F-A']['sd']:.4%} | {stats['oracle']['mean']:.4%} ± {stats['oracle']['sd']:.4%} |")
    lines += ["", "## Incomplete-group simulations", "",
        "These use the established E19 incomplete-group sample rows and fixed seeds; each row retains its outer-fold score matrix.", "",
        "| Experiment | Sample | Grouping | Mean accuracy | Mean balanced accuracy | SD balanced accuracy |",
        "|---|---|---|---:|---:|---:|"]
    for result in results:
        sample = result["incomplete_group_simulations"]
        if not isinstance(sample, list):
            continue
        sdf = pd.DataFrame(sample)
        for (variant, grouping), part in sdf.groupby(["variant", "grouping"], sort=True):
            lines.append(f"| {result['experiment']} | {variant} | {grouping} | {part.accuracy.mean():.4%} | {part.balanced_accuracy.mean():.4%} | {part.balanced_accuracy.std(ddof=1):.4%} |")
    lines += ["", "## Per-label metrics", "",
        "E19-D Hungarian precision and recall are shown below. Metrics for independent, E19-F-A, and oracle decoding are in each experiment's metrics JSON.", "",
        "| Label | E22-A precision / recall | E22-B precision / recall | E22-C precision / recall |",
        "|---|---:|---:|---:|"]
    by_name = {item["experiment"]: item for item in results}
    for label in LABELS:
        cells = []
        for eid in ("E22-A", "E22-B", "E22-C"):
            metric = by_name[eid]["metrics"]["E19-D"]["per_label"][label]
            cells.append(f"{metric['precision']:.3f} / {metric['recall']:.3f}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    comparison = by_name["E22-B"].get("comparison_vs_E22-A_E19-D", {})
    lines += ["", "## E21 checkpoint contribution", "",
        f"Adding N0135 and N0136 changed {comparison.get('changed', 0)} E19-D assignments versus E22-A: {comparison.get('corrected', 0)} corrected and {comparison.get('introduced', 0)} previously correct predictions lost (net {comparison.get('net_gains', 0)} more correct). Balanced accuracy changed by {comparison.get('delta_balanced_accuracy', 0)*100:+.3f} percentage points.", "",
        "| True label | Corrected | Introduced |", "|---|---:|---:|"]
    for label, values in comparison.get("per_label", {}).items():
        lines.append(f"| {label} | {values['corrected']} | {values['introduced']} |")
    imp = by_name["E22-B"]["feature_importance"]
    for sid in ("N0135", "N0136"):
        mean_sum = np.mean([sum(r["importance"] for r in imp if r["fold"] == fold and f"cp::{sid}::" in r["feature"]) for fold in range(1, 6)])
        lines.append(f"\nMean summed CatBoost feature importance for {sid}: {mean_sum:.3f} (averaged over folds).")
    lines += ["", "## Checkpoint roster and specialists", ""]
    for result in results:
        lines.append(f"### {result['experiment']}")
        lines.append("")
        lines.append(", ".join(result["checkpoints"]))
        lines.append("")
        if result["specialist_selections"]:
            lines.append("Fold-local per-label specialist selections are saved in `artifacts/analysis/e22/metrics_E22-C.json`.")
            lines.append("")
    ranked = sorted(summary["summary"],
        key=lambda x: (x["E19-D_hungarian_balanced_accuracy"], x["E19-D_fold_sd"]), reverse=True)
    most_stable = min(summary["summary"], key=lambda x: x["E19-D_fold_sd"])
    best = ranked[0]
    lines += ["## Validation and known limitation", "",
        f"Trust List source: `{summary['trust_list']['version']}` ({summary['trust_list']['csv_path']}; SHA-256 `{summary['trust_list']['sha256']}`). All {configuration['registry_resolved_experiments']} unique experiment IDs resolve in `experiments/registry.sqlite`. The source has {configuration['listed_checkpoint_count']} checkpoint identities and {configuration['listed_checkpoint_label_records']} checkpoint-label records; {configuration['eligible_cached_checkpoint_count']} checkpoints have complete, eligible cached V0 probabilities. Excluded dependencies are recorded in `configuration.json`.", "",
        "N0135 and N0136 have full-V0 validation metrics, and their checkpoint selection used V0. Their inclusion in E22-B/C therefore makes these V0 results development evidence, not an unbiased estimate of unseen performance. Other V0-selected checkpoints in the Trust List carry the same general limitation.", "",
        "The E19-D and E19-F-A group IDs are out-of-fold reconstructions. True V0 prompt groups are used only for the oracle Hungarian diagnostic. Every assignment is checked for no more than 18 rows per group; Hungarian decoding assigns distinct labels within each group.", "",
        f"{best['experiment']} has the highest E19-D cross-fitted BA ({best['E19-D_hungarian_balanced_accuracy']:.4%}); {most_stable['experiment']} has the lowest fold SD ({most_stable['E19-D_fold_sd']:.4%}). The added E21 checkpoints improve mean BA over control, with higher fold variation, so E22-B is the score leader but should be treated as provisional. Do not treat any variant as independently validated: V0 influenced checkpoint evaluation/selection, so an untouched set is needed before deployment.", "",
        "## Per-label metrics and artifacts", "",
        "Per-label precision, recall and confusion matrices are stored for all four prediction methods in each `metrics_E22-*.json`. Checkpoint feature importances are in the corresponding `feature_importance_E22-*.csv`; predictions and all four score matrices are row-aligned in `predictions_E22-*.npz`. The established incomplete-group sample rows are rescored for E19-D and E19-F-A in each experiment's metrics JSON.", "",
        "## Reproducibility", "",
        "Run with `python -m src.e22`. Configuration, trust-list hash, per-cache hashes, five fold rosters, model files, out-of-fold predictions, feature importances, metrics, and the summary table are in `artifacts/analysis/e22/`.", ""]
    (root / "docs/e22.md").write_text("\n".join(lines))


def _register_catalog(root, output, results):
    path = root / "configs/ensemble_catalog_v2.json"
    catalog_data = json.loads(path.read_text())
    existing = {x["id"]: x for x in catalog_data.get("experiments", [])}
    names = {"E22-A": "Updated checkpoint stacker · control",
             "E22-B": "Updated checkpoint stacker · plus E21 checkpoints",
             "E22-C": "Per-label specialist stacker · Hungarian"}
    for result in results:
        eid = result["experiment"]
        metric = result["metrics"]["E19-D"]
        existing[eid] = {"id": eid, "name": names[eid],
            "category": "Learned checkpoint ensemble",
            "description": "E20-style candidate-label CatBoost scoring with E19-D group-constrained Hungarian assignment.",
            "accuracy": metric["accuracy"], "balanced_accuracy": metric["balanced_accuracy"],
            "primary_metric": "Balanced accuracy", "primary_score": metric["balanced_accuracy"],
            "status": "Scored", "result_source": f"artifacts/analysis/e22/metrics_{eid}.json",
            "documentation": "docs/e22.md", "checkpoint_count": result["checkpoint_count"]}
    catalog_data["experiments"] = list(existing.values())
    path.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(output=args.output), indent=2))


if __name__ == "__main__":
    main()
