"""E31: leakage-controlled diagnosis of three confused NLM labels.

All base predictions are loaded from Trust List 2 v037 caches.  The fixed
E13-B deployment roster is used in every outer fold; router reliability
statistics and router fitting are confined to each outer training partition.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import e13_reliability_router as e13, e20, e28, ensemble_catalog
from .data_factory import _normal_text
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
E13_RUN = ROOT / "artifacts/analysis/e13/20261120T093318893046Z"
MANIFEST = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
V0_GROUP = ROOT / "data/processed/dataset_v0_group.parquet"
OUT = ROOT / "artifacts/analysis/e31_confusion"
TARGETS = ("glm-5.3", "glm-5.3-flash", "claude-fable-5")
SEED = 2026
N_THREADS = 24


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(y, pred, labels, names=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0)
    return {"rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_label": {(names[j] if names is not None else LABELS[int(label)]): {
            "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for j, (label,p,r,f,s) in enumerate(zip(labels, precision, recall, f1, support))}}


def _compare(y, base, pred):
    y, base, pred = np.asarray(y), np.asarray(base), np.asarray(pred)
    return {"corrected": int(((base != y) & (pred == y)).sum()),
        "introduced": int(((base == y) & (pred != y)).sum()),
        "net_correct": int(((pred == y).astype(int) - (base == y).astype(int)).sum()),
        "changed": int((base != pred).sum())}


def _fixed_roster_inputs(root=ROOT):
    root = Path(root)
    v0 = ensemble_catalog.cascades._v0(root)
    _, members, probabilities, excluded, snapshot = ensemble_catalog._load_inputs(
        root, v0, version="trust-list-2-v037")
    deployment = json.loads(MANIFEST.read_text())
    roster = list(deployment["deployment"]["checkpoint_ids"])
    if len(roster) != 15 or len(set(roster)) != 15:
        raise ValueError("E13-B v037 deployment roster must contain 15 unique checkpoints")
    missing = sorted(set(roster) - set(probabilities))
    if missing:
        raise FileNotFoundError(f"Missing cached probabilities for fixed roster: {missing}")
    labels = list(LABELS)
    if deployment["deployment"].get("classes") and deployment["deployment"]["classes"] != labels:
        raise ValueError("E13-B manifest label ordering differs from canonical ordering")
    matrices = [np.asarray(probabilities[sid], dtype=np.float64) for sid in roster]
    if any(p.shape != (len(v0), len(labels)) for p in matrices):
        raise ValueError("Cached checkpoint probabilities are not full-V0 aligned")
    tensor = np.stack(matrices)
    if not np.isfinite(tensor).all() or not np.allclose(tensor.sum(axis=2), 1, atol=1e-4):
        raise ValueError("Cached checkpoint probabilities are invalid")
    families = [e13._model_family(members[sid]["model"]) for sid in roster]
    return v0, roster, families, tensor, members, excluded, snapshot


def _folds_from_saved(v0):
    result = json.loads((E13_RUN / "results.json").read_text())
    run = next(x for x in result["results"] if x["ensemble"] == "E13-B")
    id_to_row = {str(x): i for i, x in enumerate(v0.id.astype(str))}
    groups = v0.prompt_id.astype(str).to_numpy()
    folds, assignment = [], np.full(len(v0), -1, dtype=np.int8)
    for item in run["folds"]:
        fit = np.asarray([id_to_row[str(x)] for x in item["fit_ids"]], dtype=np.int64)
        held = np.asarray([id_to_row[str(x)] for x in item["held_ids"]], dtype=np.int64)
        if np.intersect1d(fit, held).size or (assignment[held] >= 0).any():
            raise ValueError("Saved E13-B outer folds overlap")
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Saved E13-B outer fold splits prompt groups")
        assignment[held] = int(item["fold"])
        folds.append((int(item["fold"]), fit, held))
    if (assignment < 0).any():
        raise ValueError("Saved E13-B folds do not cover V0 exactly once")
    return folds, assignment


def _records(tensor, roster, families, truth, prompts, rows, n_splits=4):
    xs, ys = [], []
    rows = np.asarray(rows, dtype=np.int64)
    for fit_loc, valid_loc in GroupKFold(n_splits=n_splits).split(
            rows, truth[rows], prompts[rows]):
        stats_rows, valid_rows = rows[fit_loc], rows[valid_loc]
        x, _ = e13.build_feature_frame(tensor, roster, families,
            truth[stats_rows], stats_rows, valid_rows)
        hard = tensor[:, valid_rows].argmax(axis=2)
        xs.append(x)
        ys.append(e13._targets(hard, truth[valid_rows]))
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def _fit_fixed_router(tensor, roster, families, truth, prompts, fit_rows, held_rows,
                      path: Path):
    """Fit E13's correctness router using one fixed 15-member roster."""
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    held_rows = np.asarray(held_rows, dtype=np.int64)
    # Match E13's inner early-stopping split, but never select a different roster.
    splits = list(GroupKFold(n_splits=4).split(fit_rows, truth[fit_rows], prompts[fit_rows]))
    train_local, stop_local = splits[-1]
    router_fit, stop_rows = fit_rows[train_local], fit_rows[stop_local]
    train_x, train_y = _records(tensor, roster, families, truth, prompts, router_fit)
    stop_x, _ = e13.build_feature_frame(tensor, roster, families,
        truth[router_fit], router_fit, stop_rows)
    stop_y = e13._targets(tensor[:, stop_rows].argmax(axis=2), truth[stop_rows])
    early = e13._new_catboost()
    early.fit(e13._cat_frame(train_x), train_y, cat_features=list(e13.CATEGORICAL),
              eval_set=(e13._cat_frame(stop_x), stop_y),
              early_stopping_rounds=e13.CATBOOST_PARAMS["od_wait"],
              verbose=False, use_best_model=True)
    iterations = max(1, int(early.get_best_iteration()) + 1)
    full_x, full_y = _records(tensor, roster, families, truth, prompts, fit_rows)
    model = e13._new_catboost(iterations=iterations)
    model.fit(e13._cat_frame(full_x), full_y, cat_features=list(e13.CATEGORICAL), verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    held_x, _ = e13.build_feature_frame(tensor, roster, families,
        truth[fit_rows], fit_rows, held_rows)
    reliability = e13._score_correctness(model, held_x).reshape(len(held_rows), len(roster)).T
    hard = tensor[:, held_rows].argmax(axis=2)
    family_prob = e13.family_balanced_probabilities(tensor[:, held_rows], families)
    pred, _, _, scores = e28.route_family_max(hard, reliability, family_prob, families)
    return pred, scores, {"fold": int(path.stem.split("fold")[-1]),
        "fit_rows": int(len(fit_rows)), "held_rows": int(len(held_rows)),
        "iterations": int(iterations), "training_records": int(len(full_x)),
        "model_path": str(path)}


def _three_class_features(tensor, rows, target_indices):
    # Concatenated raw class probabilities from all fixed-roster checkpoints.
    return tensor[:, np.asarray(rows, dtype=int), :][:, :, target_indices].transpose(1, 0, 2).reshape(len(rows), -1)


def _fit_three_class(x_train, y_train, x_test):
    scaler = StandardScaler()
    a = scaler.fit_transform(x_train)
    b = scaler.transform(x_test)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED,
                               class_weight="balanced", solver="lbfgs")
    model.fit(a, y_train)
    return model, model.predict_proba(b), scaler


def _pair_auc(y, prob, target_local_labels):
    result = {}
    for a in range(3):
        for b in range(a + 1, 3):
            keep = (y == a) | (y == b)
            result[f"{TARGETS[a]} vs {TARGETS[b]}"] = float(roc_auc_score(
                (y[keep] == b).astype(int), prob[keep, b] - prob[keep, a]))
    return result


def _suspicion_for_training(tensor, text, truth, target_rows, target_indices, prompts):
    """Training-only flags: inner-CV classifier disagreement plus text collisions."""
    target_rows = np.asarray(target_rows, dtype=int)
    target_mask = np.isin(truth[target_rows], target_indices)
    rows = target_rows[target_mask]
    if not len(rows):
        return {}, {"flagged": 0, "reason_counts": {}}
    x = _three_class_features(tensor, rows, target_indices)
    local_y = np.asarray([list(target_indices).index(int(v)) for v in truth[rows]])
    disagree = np.zeros(len(rows), dtype=bool)
    local_prompts = prompts[rows]
    split_count = min(4, len(np.unique(local_prompts)))
    if split_count >= 2:
        for fit, valid in GroupKFold(n_splits=split_count).split(x, local_y, local_prompts):
            _, pr, _ = _fit_three_class(x[fit], local_y[fit], x[valid])
            disagree[valid] = pr.argmax(axis=1) != local_y[valid]
    hard = tensor[:, rows, :].argmax(axis=2)
    vote_disagree = (hard != truth[rows][None, :]).sum(axis=0) >= int(np.ceil(0.60 * len(hard)))

    # Exact and near duplicate evidence is scoped to the current outer training rows.
    norm = [_normal_text(text[i]) for i in rows]
    duplicate_reason = [set() for _ in rows]
    buckets = {}
    for pos, key in enumerate(norm):
        if key:
            buckets.setdefault(key, []).append(pos)
    for members in buckets.values():
        labels = {int(truth[rows[p]]) for p in members}
        if len(labels) > 1:
            for p in members:
                duplicate_reason[p].add("exact_cross_label_duplicate")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
        max_features=100000, sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform([str(text[i]) for i in rows])
    sim = (matrix @ matrix.T).tocsr()
    for left in range(sim.shape[0]):
        for ptr in range(sim.indptr[left], sim.indptr[left + 1]):
            right, value = int(sim.indices[ptr]), float(sim.data[ptr])
            if right <= left or value < 0.95 or truth[rows[left]] == truth[rows[right]]:
                continue
            duplicate_reason[left].add("near_cross_label_duplicate")
            duplicate_reason[right].add("near_cross_label_duplicate")
    flags = {}
    for pos, row in enumerate(rows):
        reasons = set()
        if disagree[pos]: reasons.add("cross_fitted_three_class_disagreement")
        if vote_disagree[pos]: reasons.add("checkpoint_vote_disagreement_9_of_15")
        reasons |= duplicate_reason[pos]
        if reasons:
            flags[int(row)] = sorted(reasons)
    return flags, {"flagged": len(flags),
        "reason_counts": {name: int(sum(name in reasons for reasons in flags.values()))
                          for name in sorted({r for rs in flags.values() for r in rs})},
        "per_label": {label: int(sum(int(truth[row]) == LABELS.index(label) for row in flags))
                      for label in TARGETS}}


def _random_remove(rows, truth, flags, seed):
    rng = np.random.default_rng(seed)
    flagged = set(flags)
    removed = set()
    for label in TARGETS:
        label_rows = [int(r) for r in rows if truth[int(r)] == LABELS.index(label)]
        n = sum(r in flagged for r in label_rows)
        if n:
            removed.update(rng.choice(label_rows, size=min(n, max(0, len(label_rows)-1)),
                                     replace=False).tolist())
    return removed


def _group_metrics(y, pred, independent, targets):
    all_idx = np.arange(len(LABELS))
    target = np.isin(y, targets)
    target_local = np.asarray([list(targets).index(int(value)) for value in y[target]])
    pred_local = np.asarray([list(targets).index(int(value)) if int(value) in targets else 3
                             for value in pred[target]])
    return {"whole": {**_metrics(y, pred, all_idx), "vs_independent": _compare(y, independent, pred)},
        "target_three": {**_metrics(y[target], pred[target], targets),
                         "vs_independent": _compare(y[target], independent[target], pred[target]),
                         "confusion_with_other_predictions": confusion_matrix(
                             target_local, pred_local, labels=np.arange(4)).tolist()[:3]}}


def _validate_scenarios(frame, v0):
    if frame.scenario_id.nunique() != 50:
        raise ValueError("Expected exactly 50 v0_group scenarios")
    original = v0.set_index(v0.id.astype(str))
    mapping = {}
    for scenario, part in frame.groupby("scenario_id", sort=True):
        ids = part.original_id.astype(str)
        if ids.duplicated().any() or not ids.isin(original.index).all():
            raise ValueError(f"Invalid or duplicate original IDs in {scenario}")
        if not np.array_equal(part.text.astype(str).to_numpy(), original.loc[ids, "text"].astype(str).to_numpy()):
            raise ValueError(f"Response text mismatch in {scenario}")
        if not np.array_equal(part.model.astype(str).to_numpy(), original.loc[ids, "model"].astype(str).to_numpy()):
            raise ValueError(f"Labels mismatch in {scenario}")
        for _, group in part.groupby("group_id", sort=False):
            if group.original_prompt_id.nunique() != 1 or group.model.nunique() != len(group):
                raise ValueError(f"Invalid intended group composition in {scenario}")
        mapping[str(scenario)] = part.reset_index(drop=True)
    return mapping


def _reconstructed_groups(method, rows, ids, texts, embeddings, fc, d_bundle,
                          recon, semantic):
    if method.startswith("E19-F"):
        bundle = fc if method == "E19-F-C" else {"threshold": 0.85,
            "neighbors": 40, "vectorizers": fc["vectorizers"]}
        return e28._predict_f_groups(method, ids, texts, embeddings, bundle)
    subset = recon.iloc[rows].reset_index(drop=True)
    return e28._predict_d_groups(ids, texts, subset, semantic[rows], d_bundle)


def run(root=ROOT, output=OUT):
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    v0, roster, families, tensor, members, excluded, snapshot = _fixed_roster_inputs(root)
    labels = list(LABELS)
    truth = pd.Categorical(v0.model.astype(str), categories=labels).codes.astype(int)
    if (truth < 0).any(): raise ValueError("Unknown V0 labels")
    prompts = v0.prompt_id.astype(str).to_numpy()
    ids = v0.id.astype(str).to_numpy()
    texts = v0.text.fillna("").astype(str).to_numpy()
    target_idx = np.asarray([labels.index(x) for x in TARGETS], dtype=int)
    folds, fold_assignment = _folds_from_saved(v0)
    scores = np.full((len(v0), len(labels)), np.nan, dtype=np.float64)
    routed = np.full(len(v0), -1, dtype=int)
    fold_log = []
    for number, fit, held in folds:
        pred, score, meta = _fit_fixed_router(tensor, roster, families, truth,
            prompts, fit, held, output / f"router_fold{number}.cbm")
        routed[held], scores[held] = pred, score
        meta["fold"] = number
        fold_log.append(meta)
        print(f"E31 router fold {number}/5: held {len(held)} rows; {meta['iterations']} iterations", flush=True)
    if not np.isfinite(scores).all() or (routed < 0).any():
        raise ValueError("E31 OOF router scores are incomplete")
    if not np.array_equal(scores.argmax(axis=1), routed):
        raise AssertionError("Fixed-roster scores and E13 family-max predictions diverge")

    # Three-class, five-fold prompt-group evaluation and fold-local noise ablations.
    target_rows = np.flatnonzero(np.isin(truth, target_idx))
    target_local = np.asarray([list(target_idx).index(int(truth[r])) for r in target_rows])
    feature_all = _three_class_features(tensor, np.arange(len(v0)), target_idx)
    three_prob = np.full((len(v0), 3), np.nan, dtype=np.float64)
    variant_prob = {key: np.full_like(three_prob, np.nan) for key in ("all", "flagged_removed", "random_removed")}
    flag_records, quality_folds, fitted_models = [], [], []
    for fold_no, fit, held in folds:
        fit_target = fit[np.isin(truth[fit], target_idx)]
        held_target = held[np.isin(truth[held], target_idx)]
        fit_y = np.asarray([list(target_idx).index(int(truth[r])) for r in fit_target])
        held_y = np.asarray([list(target_idx).index(int(truth[r])) for r in held_target])
        flags, audit = _suspicion_for_training(tensor, texts, truth, fit, target_idx, prompts)
        removed_random = _random_remove(fit_target, truth, flags, SEED + fold_no)
        removed_flagged = set(flags)
        fold_variant_pred = {}
        for name, removed in (("all", set()), ("flagged_removed", removed_flagged),
                              ("random_removed", removed_random)):
            keep = np.asarray([i for j, i in enumerate(fit_target) if int(i) not in removed], dtype=int)
            keep_y = np.asarray([list(target_idx).index(int(truth[r])) for r in keep])
            model, pr, scaler = _fit_three_class(feature_all[keep], keep_y, feature_all[held_target])
            variant_prob[name][held_target] = pr
            fold_variant_pred[name] = pr.argmax(axis=1)
            model_record = {"fold": fold_no, "variant": name, "rows": int(len(keep)),
                "feature_count": int(feature_all.shape[1]), "classifier": model,
                "scaler": scaler}
            joblib.dump(model_record, output / f"three_class_{name}_fold{fold_no}.joblib", compress=3)
        three_prob[held_target] = variant_prob["all"][held_target]
        for row, reasons in flags.items():
            flag_records.append({"fold": fold_no, "id": ids[row], "true_label": labels[truth[row]],
                                 "reasons": ";".join(reasons)})
        variant_metrics = {}
        for name, pr in fold_variant_pred.items():
            variant_metrics[name] = {"accuracy": float(accuracy_score(held_y, pr)),
                "balanced_accuracy": float(balanced_accuracy_score(held_y, pr)),
                "rows": int(len(held_target)),
                "correct": int((pr == held_y).sum())}
        quality_folds.append({"fold": fold_no, "flag_audit": audit,
            "flagged_per_label": {label: int(sum(labels[truth[row]] == label for row in flags)) for label in TARGETS},
            "random_removed_per_label": {label: int(sum(truth[row] == labels.index(label) for row in removed_random)) for label in TARGETS},
            "held_metrics": variant_metrics})
        print(f"E31 label-quality fold {fold_no}/5: flagged {len(flags)}/{len(fit_target)}", flush=True)
    if not np.isfinite(variant_prob["all"][target_rows]).all():
        raise ValueError("Three-class OOF predictions are incomplete")

    # Full-V0 classification separability metrics.
    three_pred_local = variant_prob["all"][target_rows].argmax(axis=1)
    three_metrics = _metrics(target_local, three_pred_local, np.arange(3), names=TARGETS)
    three_metrics["pairwise_auc"] = _pair_auc(target_local, variant_prob["all"][target_rows], target_local)
    ranks = np.argsort(-variant_prob["all"][target_rows], axis=1, kind="stable")
    three_metrics["correct_label_rank"] = {"mean": float(np.mean([np.where(row == y)[0][0] + 1
        for row, y in zip(ranks, target_local)])),
        "top1": int(np.sum(three_pred_local == target_local)),
        "top2_or_better": int(sum(y in row[:2] for row, y in zip(ranks, target_local))),
        "top3": int(len(target_rows))}
    full_ranks = np.argsort(-scores, axis=1, kind="stable")
    true_rank = np.asarray([int(np.flatnonzero(full_ranks[row] == truth[row])[0]) + 1
                            for row in target_rows])
    three_metrics["e13b_score_correct_label_rank_full18"] = {
        "mean": float(true_rank.mean()), "top1": int((true_rank == 1).sum()),
        "top3": int((true_rank <= 3).sum()), "top5": int((true_rank <= 5).sum())}
    original_e13 = pd.read_parquet(E13_RUN / "predictions.parquet")
    e13_pred = pd.Categorical(original_e13["E13-B"].astype(str), categories=labels).codes
    e13_target_pred = np.asarray([list(target_idx).index(int(x)) if int(x) in target_idx else -1
                                  for x in e13_pred[target_rows]])
    e13_correct = (e13_target_pred == target_local)
    e13_cm = confusion_matrix(target_local,
        np.where(e13_target_pred < 0, 3, e13_target_pred), labels=np.arange(4))
    e13_three = {"rows": int(len(target_rows)), "accuracy": float(e13_correct.mean()),
        "balanced_accuracy": float(np.mean(np.diag(e13_cm)[:3] / np.maximum(e13_cm.sum(axis=1)[:3], 1))),
        "predicted_outside_target_count": int((e13_target_pred < 0).sum()),
        "confusion_matrix_with_outside_as_miss": e13_cm.tolist()}
    quality_summary = {}
    for name in ("all", "flagged_removed", "random_removed"):
        pr = variant_prob[name][target_rows]
        pred = pr.argmax(axis=1)
        quality_summary[name] = {**_metrics(target_local, pred, np.arange(3), names=TARGETS),
            "vs_all_removed": _compare(target_local, three_pred_local, pred),
        "per_label_flagged_fold_occurrences": {label: int(sum(x["true_label"] == label for x in flag_records))
                                  for label in TARGETS}}

    # Reuse the identical OOF 18-class score matrix for every decoder.
    independent = scores.argmax(axis=1)
    # score tie policy is stable canonical order; E13's independent route is separately retained above.
    embeddings = np.load(root / "artifacts/analysis/e19f/response_embeddings.npy", mmap_mode="r")
    if len(embeddings) != len(v0): raise ValueError("Cached E28 embeddings do not align with V0")
    fc = joblib.load(root / "artifacts/analysis/e28/e19_f_c_external_grouper.joblib")
    d_bundle = joblib.load(root / "artifacts/analysis/e28/e19_d_external_grouper.joblib")
    recon = pd.read_parquet(root / "artifacts/analysis/e19d/reconstructed_prompts.parquet")
    semantic = np.load(root / "artifacts/analysis/e19d/semantic_embeddings.npy", mmap_mode="r")
    if recon.id.astype(str).tolist() != ids.tolist() or len(semantic) != len(v0):
        raise ValueError("E28 E19-D rowwise caches do not align with V0")
    methods = ("E19-F-C", "E19-F-A", "E19-D")
    full_metrics = {"Independent": _group_metrics(truth, independent, independent, target_idx)}
    full_predictions = {"Independent": independent.copy()}
    full_groups = {}
    for method in methods:
        group_ids = _reconstructed_groups(method, np.arange(len(v0)), ids, texts,
            embeddings, fc, d_bundle, recon, semantic)
        pred = e20.hungarian(scores, group_ids, ids)
        full_groups[method] = group_ids
        full_predictions[method] = pred
        full_metrics[method] = _group_metrics(truth, pred, independent, target_idx)
    true_groups = v0.prompt_id.astype(str).to_numpy()
    oracle_pred = e20.hungarian(scores, true_groups, ids)
    full_predictions["True prompt groups"] = oracle_pred
    full_metrics["True prompt groups"] = _group_metrics(truth, oracle_pred, independent, target_idx)
    pd.DataFrame({"id": ids, "true_label": np.asarray(labels, dtype=object)[truth],
        "independent": np.asarray(labels, dtype=object)[independent],
        **{name.replace(" ", "_"): np.asarray(labels, dtype=object)[pred]
           for name, pred in full_predictions.items() if name != "Independent"}}).to_parquet(
               output / "full_v0_group_predictions.parquet", index=False)

    scenarios_frame = pd.read_parquet(V0_GROUP)
    scenarios = _validate_scenarios(scenarios_frame, v0)
    id_to_row = {value: i for i, value in enumerate(ids)}
    scenario_records = []
    scenario_predictions = []
    for sidx, (scenario_id, part) in enumerate(sorted(scenarios.items())):
        rows = np.asarray([id_to_row[str(x)] for x in part.original_id], dtype=int)
        sy = truth[rows]
        base = independent[rows]
        scenario_result = {"scenario_id": scenario_id, "rows": len(rows)}
        all_preds = {"Independent": base}
        for method in methods:
            group_ids = _reconstructed_groups(method, rows, ids[rows], texts[rows],
                embeddings[rows], fc, d_bundle, recon, semantic)
            all_preds[method] = e20.hungarian(scores[rows], group_ids, ids[rows])
        true_group_ids = part.group_id.astype(str).to_numpy()
        all_preds["True prompt groups"] = e20.hungarian(scores[rows], true_group_ids, ids[rows])
        target = np.isin(sy, target_idx)
        for name, pred in all_preds.items():
            scenario_result[name] = {"whole_accuracy": float(accuracy_score(sy, pred)),
                "whole_balanced_accuracy": float(balanced_accuracy_score(sy, pred)),
                "target_accuracy": float(accuracy_score(sy[target], pred[target])),
                "target_balanced_accuracy": float(balanced_accuracy_score(sy[target], pred[target])),
                "whole_vs_independent": _compare(sy, base, pred),
                "target_vs_independent": _compare(sy[target], base[target], pred[target]),
                "target_recall": {label: float(np.mean(pred[sy == i] == i))
                                  for label, i in zip(TARGETS, target_idx)}}
            scenario_predictions.append(pd.DataFrame({"scenario_id": scenario_id,
                "original_id": ids[rows], "true_label": np.asarray(labels, dtype=object)[sy],
                "method": name, "prediction": np.asarray(labels, dtype=object)[pred]}))
        scenario_records.append(scenario_result)
        if (sidx + 1) % 10 == 0: print(f"E31 v0_group scenarios {sidx+1}/50", flush=True)
    scenario_summary = {}
    for name in ["Independent", *methods, "True prompt groups"]:
        scenario_summary[name] = {}
        for metric in ("whole_accuracy", "whole_balanced_accuracy",
                       "target_accuracy", "target_balanced_accuracy"):
            values = np.asarray([row[name][metric] for row in scenario_records], dtype=float)
            scenario_summary[name][metric] = {"mean": float(values.mean()),
                "std": float(values.std(ddof=0)), "min": float(values.min()),
                "p10": float(np.quantile(values, .10)), "max": float(values.max())}
        scenario_summary[name]["scenarios_improved_vs_independent"] = int(sum(
            row[name]["whole_balanced_accuracy"] > row["Independent"]["whole_balanced_accuracy"]
            for row in scenario_records))
        scenario_summary[name]["scenarios_regressed_vs_independent"] = int(sum(
            row[name]["whole_balanced_accuracy"] < row["Independent"]["whole_balanced_accuracy"]
            for row in scenario_records))
        scenario_summary[name]["aggregate_whole_changes"] = {
            key: int(sum(row[name]["whole_vs_independent"][key] for row in scenario_records))
            for key in ("corrected", "introduced", "net_correct", "changed")}
        scenario_summary[name]["aggregate_target_changes"] = {
            key: int(sum(row[name]["target_vs_independent"][key] for row in scenario_records))
            for key in ("corrected", "introduced", "net_correct", "changed")}
        scenario_summary[name]["mean_target_recall"] = {
            label: float(np.mean([row[name]["target_recall"][label] for row in scenario_records]))
            for label in TARGETS}
    # Persist row-level results and artifact provenance.
    pd.DataFrame({"id": ids, "prompt_id": prompts, "true_label": np.asarray(labels, dtype=object)[truth],
        "fold": fold_assignment, "e13b_saved": np.asarray(labels, dtype=object)[e13_pred],
        "e31_independent": np.asarray(labels, dtype=object)[routed],
        "scores": [x.tolist() for x in scores]}).drop(columns="scores").to_parquet(
            output / "oof_predictions.parquet", index=False)
    np.savez_compressed(output / "oof_score_matrices.npz", row_ids=ids, labels=np.asarray(labels),
        y=truth, scores=scores, predictions=routed, fold=fold_assignment,
        checkpoint_ids=np.asarray(roster), checkpoint_probability_tensor=tensor)
    np.savez_compressed(output / "three_class_oof_probabilities.npz",
        row_ids=ids[target_rows], true_local=target_local,
        all=variant_prob["all"][target_rows], flagged_removed=variant_prob["flagged_removed"][target_rows],
        random_removed=variant_prob["random_removed"][target_rows], labels=np.asarray(TARGETS))
    pd.DataFrame(flag_records).to_csv(output / "suspicious_training_rows.csv", index=False)
    pd.concat(scenario_predictions, ignore_index=True).to_parquet(output / "scenario_predictions.parquet", index=False)
    pd.DataFrame(scenario_records).to_json(output / "scenario_metrics.json", orient="records", indent=2)
    pd.DataFrame([{"method": name, "metric": metric, **value}
        for name, obj in scenario_summary.items() for metric, value in obj.items()
        if isinstance(value, dict)]).to_csv(output / "scenario_summary.csv", index=False)
    result = {"experiment": "E31 Three-Model Confusion Diagnosis", "seed": SEED,
        "cpu_threads": N_THREADS, "target_labels": list(TARGETS), "checkpoint_roster": roster,
        "roster_source": "frozen E13-B v037 deployment manifest; 15 checkpoints fixed in all folds",
        "folds": fold_log, "metrics": {"fixed_roster_router_oof": _metrics(truth, routed, np.arange(18)),
            "original_e13b_saved_oof": _metrics(truth, e13_pred, np.arange(18)),
            "router_vs_saved_e13b": _compare(truth, e13_pred, routed),
            "class_separability": {"three_class_oof": three_metrics,
                "original_e13b_three_label_restricted": e13_three},
            "label_quality_ablation": {"variants": quality_summary, "folds": quality_folds},
            "group_constraints_full_v0": full_metrics,
            "group_constraints_v0_group": scenario_summary},
        "scenario_count": len(scenarios), "scenario_metrics": scenario_records,
        "source_hashes": {"trust_list": snapshot["sha256"],
            "e13_results": _sha(E13_RUN / "results.json"), "e13_predictions": _sha(E13_RUN / "predictions.parquet"),
            "manifest": _sha(MANIFEST), "v0": _sha(root / "data/processed/dataset_v0.parquet"),
            "v0_group": _sha(V0_GROUP)},
        "excluded_checkpoints": excluded, "runtime_seconds": float(time.perf_counter()-started)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


if __name__ == "__main__":
    run()
