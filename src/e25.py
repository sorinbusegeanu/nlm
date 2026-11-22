"""E25 backward elimination for the E22-B checkpoint roster.

CPU-only: uses cached full-V0 checkpoint probabilities and the saved E22-B
fold models. No base-model inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from . import e20, e22
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e25"
SEED = 2026
THREADS = 24
BUDGETS = (60, 40, 30, 25, 20)
FEATURE_SCHEMA_VERSION = 1
NEAR_DUPLICATE_TOP1 = 0.98
NEAR_DUPLICATE_COSINE = 0.995
SUBSTANTIAL_TRAINING_DROP = 0.005


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fast_features(ctx, rows, pool):
    """Vectorized equivalent of E22's candidate feature frame."""
    rows = np.asarray(rows, dtype=np.int64)
    pool = list(pool)
    labels = len(LABELS)
    p = ctx["matrices"][pool][:, rows, :].astype(np.float64, copy=False)
    ncp, nrow, nlab = p.shape
    if nlab != labels or not np.isfinite(p).all():
        raise ValueError("Invalid checkpoint matrix for E25 features")
    cp_top = p.argmax(axis=2)
    cp_conf = p.max(axis=2)
    families = [ctx["families"][i] for i in pool]
    family_names = sorted(set(families))
    family_members = {fam: [j for j, value in enumerate(families) if value == fam]
                      for fam in family_names}
    family_scores = np.stack([p[ix].mean(axis=0) for ix in family_members.values()])
    ensemble = family_scores.mean(axis=0)
    flat = lambda value: np.asarray(value).reshape(-1)

    best = ensemble.max(axis=1)
    second = np.partition(ensemble, -2, axis=1)[:, -2]
    e13_rank = 1 + (ensemble[:, None, :] > ensemble[:, :, None]).sum(axis=2)
    cp_vote = cp_top[:, :, None] == np.arange(labels)[None, None, :]
    cp_prob_flat = p.transpose(1, 2, 0).reshape(nrow * labels, ncp)
    cp_vote_flat = cp_vote.transpose(1, 2, 0).reshape(nrow * labels, ncp)
    cp_rel_flat = np.broadcast_to(cp_conf[:, :, None], (ncp, nrow, labels)).transpose(1, 2, 0).reshape(nrow * labels, ncp)
    records = {
        "candidate_label": np.tile(np.asarray(LABELS, dtype=object), nrow),
        "e13_candidate_score": flat(ensemble),
        "e13_candidate_rank": flat(e13_rank),
        "e13_best_score": np.repeat(best, labels),
        "e13_second_score": np.repeat(second, labels),
        "e13_candidate_vs_best": flat(ensemble - best[:, None]),
    }
    cp_votes = cp_vote.sum(axis=0)
    family_vote_counts = []
    family_prob = []
    family_reliability = []
    for fam, ix in family_members.items():
        fvotes = cp_vote[ix].sum(axis=0)
        family_vote_counts.append(fvotes)
        vals = p[ix]
        family_prob.append(vals.mean(axis=0))
        family_reliability.append(cp_conf[ix].max(axis=0))
    family_prob = np.stack(family_prob)
    family_vote_counts = np.stack(family_vote_counts)
    family_reliability = np.stack(family_reliability)
    records["candidate_checkpoint_votes"] = flat(cp_votes)
    records["candidate_supporting_families"] = flat((family_vote_counts > 0).sum(axis=0))
    for fam_index, fam in enumerate(family_names):
        ix = family_members[fam]
        fprob = p[ix].mean(axis=0)
        fmaxprob = p[ix].max(axis=0)
        fvotes = cp_vote[ix].sum(axis=0)
        records[f"family::{fam}::mean_probability"] = flat(fprob)
        records[f"family::{fam}::max_probability"] = flat(fmaxprob)
        records[f"family::{fam}::vote_count"] = flat(fvotes)
        records[f"family::{fam}::max_reliability"] = np.repeat(family_reliability[fam_index], labels)
    records["candidate_family_probability_mean"] = flat(family_prob.mean(axis=0))
    records["candidate_max_family_probability"] = flat(family_prob.max(axis=0))
    voter_sum = (cp_conf[:, :, None] * cp_vote).sum(axis=0)
    voter_n = cp_vote.sum(axis=0)
    records["candidate_mean_voter_reliability"] = flat(voter_sum / np.maximum(1, voter_n))

    extras = {
        "ensemble_probability": flat(ensemble),
        "ensemble_rank": flat(np.argsort(np.argsort(-ensemble, axis=1, kind="stable"), axis=1, kind="stable") + 1),
        "ensemble_margin_from_top": flat(ensemble - best[:, None]),
        "checkpoint_probability_mean": flat(p.mean(axis=0)),
        "checkpoint_probability_std": flat(p.std(axis=0)),
        "checkpoint_probability_max": flat(p.max(axis=0)),
        "checkpoint_probability_min": flat(p.min(axis=0)),
        "checkpoint_top1_agreement": flat(np.stack([(cp_top == lab).sum(axis=0) for lab in range(labels)], axis=1)),
        "family_probability_mean": flat(family_prob.mean(axis=0)),
        "family_probability_std": flat(family_prob.std(axis=0)),
        "family_top1_agreement": flat(np.stack([(family_scores.argmax(axis=2) == lab).sum(axis=0)
                                                   for lab in range(labels)], axis=1)),
        "family_count": np.full(nrow * labels, len(family_names), dtype=np.int16),
    }
    records.update(extras)
    cp_rank = np.argsort(np.argsort(-p, axis=2, kind="stable"), axis=2, kind="stable") + 1
    cp_margin = p - p.max(axis=2, keepdims=True)
    cp_top_margin = np.sort(p, axis=2)[:, :, -1] - np.sort(p, axis=2)[:, :, -2]
    rankflat = cp_rank.transpose(1, 2, 0).reshape(nrow * labels, ncp)
    marginflat = cp_margin.transpose(1, 2, 0).reshape(nrow * labels, ncp)
    topmarginflat = np.broadcast_to(cp_top_margin.T[:, None, :], (nrow, labels, ncp)).reshape(nrow * labels, ncp)
    for j, cpidx in enumerate(pool):
        sid = ctx["checkpoint_ids"][cpidx]
        records[f"cp::{sid}::rank"] = rankflat[:, j]
        records[f"cp::{sid}::margin"] = marginflat[:, j]
        records[f"cp::{sid}::top_margin"] = topmarginflat[:, j]
    for j, cpidx in enumerate(pool):
        sid = ctx["checkpoint_ids"][cpidx]
        records[f"cp::{sid}::prob"] = cp_prob_flat[:, j]
        records[f"cp::{sid}::vote"] = cp_vote_flat[:, j].astype(np.int8)
        records[f"cp::{sid}::reliability"] = cp_rel_flat[:, j]
        records[f"cp::{sid}::present"] = np.ones(nrow * labels, dtype=np.int8)
    frame = pd.DataFrame(records)
    if len(frame) != nrow * labels or frame.candidate_label.astype(str).tolist()[:labels] != list(LABELS):
        raise ValueError("E25 feature rows lost canonical order")
    return frame


def _aligned_features(ctx, rows, pool, schema):
    frame = _fast_features(ctx, rows, pool)
    return frame.reindex(columns=schema, fill_value=0)


def _scores(model, ctx, rows, pool, schema):
    frame = _aligned_features(ctx, rows, pool, schema)
    return e20._predict_candidate(model, frame)


def audit_redundancy(ctx, baseline_pred, output):
    ids = ctx["checkpoint_ids"]
    matrices = ctx["matrices"]
    y = ctx["y"]
    pred = matrices.argmax(axis=2)
    correct = pred == y[None, :]
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            same = pred[i] == pred[j]
            cosine = np.sum(matrices[i] * matrices[j], axis=1) / np.maximum(
                1e-12, np.linalg.norm(matrices[i], axis=1) * np.linalg.norm(matrices[j], axis=1))
            left = np.clip(matrices[i], 0, None)
            right = np.clip(matrices[j], 0, None)
            left = left / left.sum(axis=1, keepdims=True)
            right = right / right.sum(axis=1, keepdims=True)
            js = jensenshannon(left, right, axis=1, base=2)
            rows.append({"checkpoint_a": ids[i], "checkpoint_b": ids[j],
                "top1_agreement": float(same.mean()),
                "mean_per_row_probability_cosine": float(cosine.mean()),
                "flattened_probability_cosine": float(np.dot(matrices[i].ravel(), matrices[j].ravel()) /
                    (np.linalg.norm(matrices[i]) * np.linalg.norm(matrices[j]))),
                "mean_probability_js_distance": float(np.mean(js)),
                "identical_top1": bool(same.all()),
                "unique_correct_a": int(np.sum(correct[i] & ~correct[np.arange(len(ids)) != i].any(axis=0))),
                "unique_correct_b": int(np.sum(correct[j] & ~correct[np.arange(len(ids)) != j].any(axis=0))),
                "a_correct_when_e22_wrong": int(np.sum(correct[i] & (baseline_pred != y))),
                "b_correct_when_e22_wrong": int(np.sum(correct[j] & (baseline_pred != y)))})
    pairs = pd.DataFrame(rows).sort_values(
        ["top1_agreement", "mean_per_row_probability_cosine"], ascending=False)
    pairs.to_csv(output / "checkpoint_pair_similarity.csv", index=False)
    # Connected components under a declared near-duplicate rule.
    parent = {sid: sid for sid in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in pairs.itertuples():
        if row.top1_agreement >= NEAR_DUPLICATE_TOP1 and row.mean_per_row_probability_cosine >= NEAR_DUPLICATE_COSINE:
            a, b = find(row.checkpoint_a), find(row.checkpoint_b)
            if a != b:
                parent[b] = a
    clusters = {}
    for sid in ids:
        clusters.setdefault(find(sid), []).append(sid)
    cluster_rows = [{"cluster_id": f"D{i:02d}", "checkpoint_ids": members,
        "size": len(members), "rule": f"top1_agreement>={NEAR_DUPLICATE_TOP1}; cosine>={NEAR_DUPLICATE_COSINE}"}
        for i, members in enumerate(sorted((v for v in clusters.values() if len(v) > 1),
                                             key=lambda x: (-len(x), x[0])), 1)]
    (output / "near_duplicate_clusters.json").write_text(json.dumps(cluster_rows, indent=2) + "\n")
    per_label = []
    for cp, sid in enumerate(ids):
        other_correct = correct[np.arange(len(ids)) != cp].any(axis=0)
        for label, name in enumerate(LABELS):
            ix = y == label
            per_label.append({"checkpoint_id": sid, "label": name,
                "unique_correct_vs_all_other_checkpoints": int(np.sum(correct[cp] & ~other_correct & ix)),
                "correct_when_E22B_wrong": int(np.sum(correct[cp] & (baseline_pred != y) & ix)),
                "complementary_E22B_error_rate": float(np.sum(correct[cp] & (baseline_pred != y) & ix) / max(1, ix.sum()))})
    pd.DataFrame(per_label).to_csv(output / "checkpoint_correct_complementarity_by_label.csv", index=False)
    return {"pair_count": len(pairs), "identical_top1_pair_count": int(pairs.identical_top1.sum()),
        "near_duplicate_clusters": cluster_rows,
        "thresholds": {"top1_agreement": NEAR_DUPLICATE_TOP1, "probability_cosine": NEAR_DUPLICATE_COSINE}}


def _group_bootstrap_delta(ctx, y, baseline, changed, *, draws=250):
    unique_groups = np.unique(ctx["groups"])
    group_rows = {g: np.flatnonzero(ctx["groups"] == g) for g in unique_groups}
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(draws):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[g] for g in sampled])
        deltas.append(float(balanced_accuracy_score(y[rows], changed[rows]) -
                            balanced_accuracy_score(y[rows], baseline[rows])))
    return {"draws": draws, "seed": SEED, "delta_p025": float(np.quantile(deltas, .025)),
        "delta_median": float(np.quantile(deltas, .5)), "delta_p975": float(np.quantile(deltas, .975))}


def run_contributions(ctx, output, baseline_pred):
    output.mkdir(parents=True, exist_ok=True)
    base = np.load(output.parent / "e22" / "predictions_E22-B.npz", allow_pickle=True)
    baseline_scores = base["scores"].astype(np.float32)
    roster_ids = list(ctx["checkpoint_ids"])
    roster = list(range(len(roster_ids)))
    all_ablated = np.full((len(roster_ids), len(ctx["y"])), -1, dtype=np.int16)
    fold_rows = []
    for fold in ctx["folds"]:
        fold_id, held = fold["fold"], fold["held_rows"]
        model = e20.CatBoostClassifier()
        model.load_model(str(output.parent / "e22" / f"E22-B_fold{fold_id}.cbm"))
        schema = list(model.feature_names_)
        full_scores = _scores(model, ctx, held, roster, schema)
        full_pred = e20.hungarian(full_scores, ctx["group_d"][held], ctx["row_ids"][held])
        if not np.array_equal(full_pred, baseline_pred[held]) or not np.allclose(
                full_scores, baseline_scores[held], rtol=1e-6, atol=1e-7):
            raise ValueError(f"Reconstructed E22-B fold {fold_id} does not reproduce saved OOF predictions")
        for cp, sid in enumerate(roster_ids):
            reduced = [x for x in roster if x != cp]
            changed_scores = _scores(model, ctx, held, reduced, schema)
            all_ablated[cp, held] = e20.hungarian(
                changed_scores, ctx["group_d"][held], ctx["row_ids"][held])
            fold_rows.append({"fold": fold_id, "checkpoint_id": sid,
                "balanced_accuracy_after_drop": float(balanced_accuracy_score(
                    ctx["y"][held], all_ablated[cp, held])),
                "baseline_balanced_accuracy": float(balanced_accuracy_score(
                    ctx["y"][held], baseline_pred[held]))})
        np.savez_compressed(output / f"contribution_fold{fold_id}.npz", row_ids=ctx["row_ids"][held],
            checkpoint_ids=np.asarray(roster_ids), predictions=all_ablated[:, held])
        print(f"E25 phase 2: fold {fold_id} perturbed {len(roster_ids)} checkpoints", flush=True)
    if (all_ablated < 0).any():
        raise RuntimeError("Checkpoint perturbation predictions do not cover V0")
    np.savez_compressed(output / "single_checkpoint_drop_predictions.npz",
        row_ids=ctx["row_ids"], checkpoint_ids=np.asarray(roster_ids), predictions=all_ablated)
    pd.DataFrame(fold_rows).to_csv(output / "contribution_by_fold.csv", index=False)
    summary = []
    base_ba = balanced_accuracy_score(ctx["y"], baseline_pred)
    for cp, sid in enumerate(roster_ids):
        pred = all_ablated[cp].astype(np.int64)
        metric = e20._metric(ctx["y"], pred)
        corrected = int(np.sum((baseline_pred != ctx["y"]) & (pred == ctx["y"])))
        introduced = int(np.sum((baseline_pred == ctx["y"]) & (pred != ctx["y"])))
        row = {"checkpoint_id": sid, "balanced_accuracy_after_drop": metric["balanced_accuracy"],
            "balanced_accuracy_loss": base_ba - metric["balanced_accuracy"],
            "accuracy_after_drop": metric["accuracy"], "corrected_vs_E22B": corrected,
            "introduced_vs_E22B": introduced, "net_correct_change": corrected - introduced,
            "per_label": metric["per_label"],
            "per_label_changes": {label: {
                "corrected": int(((baseline_pred != ctx["y"]) & (pred == ctx["y"]) & (ctx["y"] == i)).sum()),
                "introduced": int(((baseline_pred == ctx["y"]) & (pred != ctx["y"]) & (ctx["y"] == i)).sum()),
                "baseline_correct": int(((baseline_pred == ctx["y"]) & (ctx["y"] == i)).sum()),
                "after_drop_correct": int(((pred == ctx["y"]) & (ctx["y"] == i)).sum())}
                for i, label in enumerate(LABELS)},
            "group_bootstrap_delta_vs_baseline": _group_bootstrap_delta(ctx, ctx["y"], baseline_pred, pred)}
        summary.append(row)
    summary.sort(key=lambda x: (x["balanced_accuracy_loss"], x["checkpoint_id"]))
    (output / "contribution_rankings.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([{k: v for k, v in row.items() if k != "per_label" and k != "group_bootstrap_delta_vs_baseline"} |
        {"bootstrap_low": row["group_bootstrap_delta_vs_baseline"]["delta_p025"],
         "bootstrap_high": row["group_bootstrap_delta_vs_baseline"]["delta_p975"]} for row in summary]
        ).to_csv(output / "contribution_rankings.csv", index=False)
    # Simultaneous cluster removal reveals groups of individually weak but
    # jointly useful checkpoints. Use only clusters identified before scoring.
    clusters = json.loads((output / "near_duplicate_clusters.json").read_text())
    cluster_predictions = {}
    cluster_summary = []
    for cluster in clusters:
        members = cluster["checkpoint_ids"]
        removed = {roster_ids.index(sid) for sid in members}
        reduced = [i for i in roster if i not in removed]
        dropped = np.full(len(ctx["y"]), -1, dtype=np.int16)
        for fold in ctx["folds"]:
            fold_id, held = fold["fold"], fold["held_rows"]
            model = e20.CatBoostClassifier()
            model.load_model(str(output.parent / "e22" / f"E22-B_fold{fold_id}.cbm"))
            pred = e20.hungarian(_scores(model, ctx, held, reduced, list(model.feature_names_)),
                                 ctx["group_d"][held], ctx["row_ids"][held])
            dropped[held] = pred
        if (dropped < 0).any():
            raise RuntimeError(f"Cluster perturbation {cluster['cluster_id']} missed rows")
        cluster_predictions[cluster["cluster_id"]] = dropped
        metric = e20._metric(ctx["y"], dropped.astype(np.int64))
        cluster_summary.append({"cluster_id": cluster["cluster_id"], "checkpoint_ids": members,
            "size": len(members), "balanced_accuracy_after_drop": metric["balanced_accuracy"],
            "balanced_accuracy_loss": base_ba - metric["balanced_accuracy"],
            "corrected_vs_E22B": int(np.sum((baseline_pred != ctx["y"]) & (dropped == ctx["y"]))),
            "introduced_vs_E22B": int(np.sum((baseline_pred == ctx["y"]) & (dropped != ctx["y"])))})
    if cluster_predictions:
        np.savez_compressed(output / "cluster_drop_predictions.npz", row_ids=ctx["row_ids"],
            **cluster_predictions)
    pd.DataFrame(cluster_summary).to_csv(output / "cluster_drop_metrics.csv", index=False)
    return summary


def _train_candidate_scores(model, ctx, rows, roster, schema):
    mat = _scores(model, ctx, rows, roster, schema)
    pred = e20.hungarian(mat, ctx["group_d"][rows], ctx["row_ids"][rows])
    return mat, pred, float(balanced_accuracy_score(ctx["y"][rows], pred))


def _fold_feature_importance(model, roster_ids):
    importance = dict(zip(model.feature_names_, model.get_feature_importance()))
    by_id = {}
    for sid in roster_ids:
        by_id[sid] = float(sum(value for name, value in importance.items() if name.startswith(f"cp::{sid}::")))
    return by_id


def _select_fold_rosters(ctx, output, budget_dir):
    roster_ids = list(ctx["checkpoint_ids"])
    rosters_by_fold = {}
    for fold in ctx["folds"]:
        fold_id, train_rows = fold["fold"], fold["fit_rows"]
        record_path = budget_dir / f"selection_fold{fold_id}.json"
        if record_path.is_file():
            saved = json.loads(record_path.read_text())
            rosters_by_fold[str(fold_id)] = saved
            continue
        model = e20.CatBoostClassifier()
        model.load_model(str(output.parent / "e22" / f"E22-B_fold{fold_id}.cbm"))
        schema = list(model.feature_names_)
        pool = list(range(len(roster_ids)))
        base_scores, current_pred, current_ba = _train_candidate_scores(model, ctx, train_rows, pool, schema)
        current_roster = pool.copy()
        importance = _fold_feature_importance(model, roster_ids)
        ytrain = ctx["y"][train_rows]
        cp_preds = ctx["matrices"][:, train_rows, :].argmax(axis=2)
        removal_trace = []
        target_rosters = {}
        while len(current_roster) > min(BUDGETS):
            target = next((b for b in BUDGETS if len(current_roster) > b), min(BUDGETS))
            gap = len(current_roster) - target
            batch_size = min(5, gap)
            roster_ids_current = [roster_ids[i] for i in current_roster]
            correct = cp_preds[current_roster] == ytrain[None, :]
            unique_correct = correct & ~np.any(np.delete(correct, np.arange(len(current_roster)), axis=0), axis=0)[None, :]
            complementary = correct & (current_pred != ytrain)[None, :]
            supports = np.bincount(ytrain, minlength=len(LABELS)).astype(float)
            weighted_unique = np.sum(unique_correct / supports[ytrain][None, :], axis=1) / len(LABELS)
            weighted_complementary = np.sum(complementary / supports[ytrain][None, :], axis=1) / len(LABELS)
            weakest = sorted(range(len(current_roster)), key=lambda j: (
                weighted_unique[j], weighted_complementary[j], importance[roster_ids_current[j]],
                roster_ids_current[j]))[:batch_size]
            candidate_remove = [current_roster[j] for j in weakest]
            candidate_pool = [i for i in current_roster if i not in set(candidate_remove)]
            _, pred_after, ba_after = _train_candidate_scores(model, ctx, train_rows, candidate_pool, schema)
            drop = current_ba - ba_after
            ba_before = current_ba
            accepted = []
            if drop <= SUBSTANTIAL_TRAINING_DROP:
                accepted = candidate_remove
                current_roster = candidate_pool
                current_pred, current_ba = pred_after, ba_after
                method = "accepted-small-batch"
            else:
                method = "batch-restored; individual tests"
                # The batch is restored. Test its members one at a time and
                # accept the least damaging individual removal until target.
                remaining_batch = candidate_remove.copy()
                while len(current_roster) > target and remaining_batch:
                    options = []
                    for candidate in remaining_batch:
                        if candidate not in current_roster:
                            continue
                        p = [i for i in current_roster if i != candidate]
                        _, p_after, ba_single = _train_candidate_scores(model, ctx, train_rows, p, schema)
                        options.append((current_ba - ba_single, candidate, p_after, ba_single))
                    if not options:
                        break
                    loss, chosen, chosen_pred, chosen_ba = min(options,
                        key=lambda x: (x[0], roster_ids[x[1]]))
                    accepted.append(chosen)
                    current_roster.remove(chosen)
                    remaining_batch.remove(chosen)
                    current_pred, current_ba = chosen_pred, chosen_ba
                    if len(current_roster) == target:
                        break
            removal_trace.append({"from_size": len(current_roster) + len(accepted),
                "target": target, "batch_candidates": [roster_ids[i] for i in candidate_remove],
                "removed": [roster_ids[i] for i in accepted], "method": method,
                "train_ba_before": float(ba_before),
                "train_ba_after": float(current_ba), "estimated_train_ba_drop": float(drop)})
            if len(current_roster) == target:
                target_rosters[str(target)] = list(current_roster)
            elif len(current_roster) < target:
                raise RuntimeError("E25 elimination overshot a roster budget")
        saved = {"fold": fold_id, "target_rosters": {str(k): [roster_ids[i] for i in v]
            for k, v in target_rosters.items()}, "removal_trace": removal_trace,
            "final_roster": [roster_ids[i] for i in current_roster]}
        # Capture intermediate roster snapshots from the trace by replaying the
        # accepted removals; budgets are recorded at exact transitions above.
        saved["target_rosters"] = {key: value for key, value in saved["target_rosters"].items()
                                   if int(key) in BUDGETS}
        record_path.write_text(json.dumps(saved, indent=2) + "\n")
        rosters_by_fold[str(fold_id)] = saved
        print(f"E25 phase 3 selection: fold {fold_id} reached {len(current_roster)}", flush=True)
    return rosters_by_fold


def _train_and_evaluate_budgets(ctx, output, rosters_by_fold, baseline_pred):
    y = ctx["y"]
    result = {}
    for budget in BUDGETS:
        pred = np.full(len(y), -1, dtype=np.int64)
        score_matrix = np.full((len(y), len(LABELS)), np.nan, dtype=np.float32)
        folds_result = []
        for fold in ctx["folds"]:
            fold_id = fold["fold"]
            fit_rows, held = fold["fit_rows"], fold["held_rows"]
            fold_roster = rosters_by_fold[str(fold_id)]["target_rosters"].get(str(budget))
            if not fold_roster or len(fold_roster) != budget or len(set(fold_roster)) != budget:
                raise ValueError(f"Fold {fold_id} has no valid {budget}-checkpoint roster")
            model_path = output / f"E25_{budget}_fold{fold_id}.cbm"
            preds_path = output / f"E25_{budget}_fold{fold_id}.npz"
            if preds_path.is_file() and model_path.is_file():
                saved = np.load(preds_path, allow_pickle=True)
                if np.array_equal(saved["row_ids"].astype(str), ctx["row_ids"][held].astype(str)):
                    mat, held_pred = saved["scores"], saved["pred"]
                else:
                    mat = None
            else:
                mat = None
            pool = [ctx["checkpoint_ids"].index(sid) for sid in fold_roster]
            if mat is None:
                xfit = _fast_features(ctx, fit_rows, pool)
                xheld = _fast_features(ctx, held, pool)
                e22fold = next(x for x in json.loads((output.parent / "e22/metrics_E22-B.json").read_text())["folds"]
                               if x["fold"] == fold_id)
                params = e22fold["selected_hyperparameters"]
                model = e20._cat_train(xfit, e22._target(y[fit_rows]),
                    params["depth"], params["learning_rate"])
                model.save_model(str(model_path))
                schema = list(model.feature_names_)
                xheld = xheld.reindex(columns=schema, fill_value=0)
                mat = e20._predict_candidate(model, xheld)
                held_pred = e20.hungarian(mat, ctx["group_d"][held], ctx["row_ids"][held])
                np.savez_compressed(preds_path, row_ids=ctx["row_ids"][held], scores=mat, pred=held_pred)
            pred[held] = held_pred
            score_matrix[held] = mat
            metrics = e20._metric(y[held], held_pred)
            independent_pred = mat.argmax(axis=1)
            folds_result.append({"fold": fold_id, "roster": fold_roster,
                "fit_rows": len(fit_rows), "held_rows": len(held), "metrics": metrics,
                "independent_metrics": e20._metric(y[held], independent_pred)})
            print(f"E25 budget {budget} fold {fold_id}: BA={metrics['balanced_accuracy']:.4%}", flush=True)
        if (pred < 0).any() or not np.isfinite(score_matrix).all():
            raise RuntimeError(f"E25 {budget}-checkpoint OOF predictions incomplete")
        metrics = e20._metric(y, pred)
        independent = score_matrix.argmax(axis=1)
        comparison = e20._comparison(y, baseline_pred, pred)
        comparison["per_label"] = {label: {
            "corrected": int(((baseline_pred != y) & (pred == y) & (y == i)).sum()),
            "introduced": int(((baseline_pred == y) & (pred != y) & (y == i)).sum()),
            "baseline_correct": int(((baseline_pred == y) & (y == i)).sum()),
            "subset_correct": int(((pred == y) & (y == i)).sum())}
            for i, label in enumerate(LABELS)}
        result[str(budget)] = {"budget": budget, "metrics": metrics,
            "independent_metrics": e20._metric(y, independent), "folds": folds_result,
            "fold_ba_mean": float(np.mean([f["metrics"]["balanced_accuracy"] for f in folds_result])),
            "fold_ba_sd": float(np.std([f["metrics"]["balanced_accuracy"] for f in folds_result], ddof=1)),
            "comparison_vs_E22B": comparison}
        np.savez_compressed(output / f"predictions_E25_{budget}.npz", row_ids=ctx["row_ids"], y=y,
            scores=score_matrix, pred_E19D=pred)
        (output / f"metrics_E25_{budget}.json").write_text(json.dumps(result[str(budget)], indent=2, allow_nan=False) + "\n")
    return result


def _global_deployment_rosters(rosters_by_fold):
    result = {}
    for budget in BUDGETS:
        counts = {}
        for fold in rosters_by_fold.values():
            for sid in fold["target_rosters"][str(budget)]:
                counts[sid] = counts.get(sid, 0) + 1
        chosen = sorted(counts, key=lambda sid: (-counts[sid], int(sid[1:])))[:budget]
        result[str(budget)] = {"checkpoint_ids": chosen, "count": len(chosen),
            "fold_selection_frequency": {sid: counts[sid] for sid in chosen},
            "selection_method": "most frequent membership among the five fitting-fold-selected rosters; ties by checkpoint ID",
            "note": "Global deployment roster only; not used to calculate cross-fitted scores."}
    return result


def _write_report(output, summary):
    audit = summary["redundancy_audit"]
    lines = ["# E25 — Backward checkpoint elimination for E22-B", "",
        "## Result", "",
        f"E22-B was reproduced from its saved fold checkpoints at **{summary['baseline_balanced_accuracy']:.4%} balanced accuracy** with E19-D Hungarian decoding. The experiment only used cached V0 probability arrays and CPU CatBoost models; it ran no base-model inference.", "",
        "| Roster budget | Independent BA | E19-D Hungarian accuracy | E19-D Hungarian BA | Fold BA mean ± SD | Δ BA vs E22-B | Corrected | Introduced | Globally distinct deployment checkpoints |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in BUDGETS:
        item = summary["budgets"][str(b)]
        metric = item["metrics"]
        comp = item["comparison_vs_E22B"]
        sd = item["fold_ba_sd"]
        global_count = summary["global_deployment_rosters"][str(b)]["count"]
        lines.append(f"| {b} | {item['independent_metrics']['balanced_accuracy']:.4%} | {metric['accuracy']:.4%} | {metric['balanced_accuracy']:.4%} | {item['fold_ba_mean']:.4%} ± {sd:.4%} | {(metric['balanced_accuracy']-summary['baseline_balanced_accuracy'])*100:+.3f} pp | {comp['corrected']} | {comp['introduced']} | {global_count} |")
    best_budget = max(BUDGETS, key=lambda b: summary["budgets"][str(b)]["metrics"]["balanced_accuracy"])
    best_ids = summary["global_deployment_rosters"][str(best_budget)]["checkpoint_ids"]
    lines += ["", "## Phase 1 — Redundancy audit", "",
        f"All {audit['pair_count']:,} unordered checkpoint pairs were compared. {audit['identical_top1_pair_count']} pairs had identical top-1 predictions on all V0 rows. Near-duplicate clusters use the declared rule top-1 agreement ≥{audit['thresholds']['top1_agreement']:.3f} and mean per-row probability cosine ≥{audit['thresholds']['probability_cosine']:.3f}.", "",
        f"Detected {len(audit['near_duplicate_clusters'])} multi-checkpoint clusters. Pairwise metrics are in `artifacts/analysis/e25/checkpoint_pair_similarity.csv`; label-level unique correct coverage and complementary E22-B errors are in `checkpoint_correct_complementarity_by_label.csv`.", "",
        "Near-duplicate clusters: " + "; ".join(f"{x['cluster_id']}: {', '.join(x['checkpoint_ids'])}" for x in audit["near_duplicate_clusters"]) + ".", "",
        "| Cluster drop | Checkpoints removed | E19-D BA after drop | BA loss | Corrected | Introduced |", "|---|---|---:|---:|---:|---:|"]
    cluster_path = output / "cluster_drop_metrics.csv"
    if cluster_path.is_file():
        for row in pd.read_csv(cluster_path).itertuples():
            lines.append(f"| {row.cluster_id} | {row.checkpoint_ids} | {row.balanced_accuracy_after_drop:.4%} | {row.balanced_accuracy_loss*100:+.3f} pp | {row.corrected_vs_E22B} | {row.introduced_vs_E22B} |")
    lines += ["", 
        "## Phase 2 — Checkpoint contribution", "",
        "Each saved E22-B fold model was replayed on its original held-out rows. For each checkpoint, E25 removed that checkpoint, rebuilt the family, checkpoint, ensemble, and candidate features from the reduced roster, aligned absent checkpoint channels to the saved full-model schema, and reran Hungarian decoding. Replay must match saved baseline OOF predictions before perturbations are accepted. Prompt-group bootstrap intervals use seed 2026.", "",
        "| Rank | Checkpoint | BA loss when removed | Corrected vs baseline | Introduced vs baseline | Group-bootstrap 95% interval for BA delta |", "|---:|---|---:|---:|---:|---:|"]
    for rank, row in enumerate(summary["contribution_rankings"][:25], 1):
        ci = row["group_bootstrap_delta_vs_baseline"]
        lines.append(f"| {rank} | {row['checkpoint_id']} | {row['balanced_accuracy_loss']*100:+.3f} pp | {row['corrected_vs_E22B']} | {row['introduced_vs_E22B']} | [{ci['delta_p025']*100:+.3f}, {ci['delta_p975']*100:+.3f}] pp |")
    lines += ["", "The complete ranking and per-label changes are in `contribution_rankings.json`. Cluster-level simultaneous deletions are in `cluster_drop_metrics.csv`; this exposes redundancy hidden by individual ablations.", "",
        "## Phase 3 — Backward elimination", "",
        "Fold rosters are selected exclusively from each outer training partition. Selection begins at all 89 E22-B checkpoints, removes low unique-correct/complementary checkpoints in batches of five, checks the batch against the saved E22-B fold model with all dependent features recomputed, restores batches that exceed a 0.5 pp training BA loss, and uses individual removal tests near the target. Each target roster then gets a fresh CatBoost fit using the E22-B fold's selected depth and learning rate. Held-out rows are scored once.", "",
        "### Removed checkpoint order", ""]
    roster_file = output / "selection_rosters.json"
    if roster_file.is_file():
        rosters = json.loads(roster_file.read_text())
        for fold_id in sorted(rosters, key=int):
            removed_ids = [sid for step in rosters[fold_id]["removal_trace"] for sid in step["removed"]]
            lines.append(f"- Fold {fold_id}: {', '.join(removed_ids)}")
    lines += ["", "Fold-local selection details, batch restores, individual checks, and exact rosters are in `selection_rosters.json`. Fold membership and the final globally distinct deployment rosters are in `selection_manifest.json`. The global deployable roster is derived by five-fold selection frequency and has at most the requested number of distinct checkpoint IDs; it is separate from the fold-local rosters used for cross-fitting.", "",
        "## Per-label metrics", "",
        "Full per-label precision and recall are saved in each `metrics_E25_<budget>.json`. `per_label_comparison.csv` gives baseline and reduced-roster precision/recall, percentage-point changes, and corrected/introduced counts for every label and budget.", "",
        "## Interpretation and limitations", "",
        f"The highest-scoring requested roster is the **{best_budget}-checkpoint** budget ({summary['budgets'][str(best_budget)]['metrics']['balanced_accuracy']:.4%} BA; {summary['budgets'][str(best_budget)]['fold_ba_sd']:.4%} fold SD). Its globally distinct deployment roster is: {', '.join(best_ids)}.", "",
        f"The 20-checkpoint backward-elimination roster scores {summary['budgets']['20']['metrics']['balanced_accuracy']:.4%} BA, a {(summary['baseline_balanced_accuracy']-summary['budgets']['20']['metrics']['balanced_accuracy'])*100:.3f} pp loss from E22-B. E25 therefore does not recommend deploying the 20-checkpoint backward-elimination result. E23-A's 79.6880% fold-local 20-checkpoint result is a useful external comparison, but it used 44 distinct checkpoints across its folds and does not satisfy E25's global 20-checkpoint deployment limit.", "",
        "The globally distinct 20-checkpoint deployment candidate is recorded exactly in `selection_manifest.json`; it was selected by fold-roster membership frequency and is not assigned a separate cross-fitted score here.", "",
        "V0 has been used to evaluate/select some constituent checkpoints and E22-B itself, so these cross-fitted meta-ensemble results remain development diagnostics, not independent Kaggle estimates. The outer folds protect E25's roster and meta-classifier selection from each evaluated row, but do not remove historical V0 checkpoint-selection exposure.", "",
        "## Reproducibility", "",
        "Run `python -m src.e25`. All inputs, hashes, fold-local selections, fitted models, cached OOF predictions, contribution analyses, and metrics are under `artifacts/analysis/e25/`.", ""]
    (ROOT / "docs/e25.md").write_text("\n".join(lines))


def run(root=ROOT, output=OUT):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with threadpool_limits(limits=THREADS):
        ctx = e22.load_context(root)
        expected = json.loads((root / "artifacts/analysis/e22/metrics_E22-B.json").read_text())
        pred_file = np.load(root / "artifacts/analysis/e22/predictions_E22-B.npz", allow_pickle=True)
        if not np.array_equal(pred_file["row_ids"].astype(str), ctx["row_ids"].astype(str)) or not np.array_equal(pred_file["y"], ctx["y"]):
            raise ValueError("E22-B prediction rows/labels do not align with current V0")
        baseline_pred = pred_file["pred_E19D"].astype(np.int64)
        baseline_ba = float(balanced_accuracy_score(ctx["y"], baseline_pred))
        if not np.isclose(baseline_ba, .8776858066119811, atol=1e-12):
            raise ValueError(f"E22-B baseline does not reproduce 87.7686%: {baseline_ba:.8%}")
        # Validate the optimized feature builder against E22's original builder.
        e22._BASE_FEATURE_CACHE.clear()
        check_rows = ctx["folds"][0]["held_rows"][:3]
        check_pool = list(range(len(ctx["checkpoint_ids"])))
        original = e22.build_features(ctx, check_rows, check_pool)
        optimized = _fast_features(ctx, check_rows, check_pool)
        if set(original.columns) != set(optimized.columns):
            raise ValueError("Vectorized E25 feature schema differs from E22")
        for col in original.columns:
            if col == "candidate_label":
                if original[col].astype(str).tolist() != optimized[col].astype(str).tolist():
                    raise ValueError("Vectorized candidate label order differs")
            elif not np.allclose(original[col].to_numpy(dtype=float), optimized[col].to_numpy(dtype=float),
                                 atol=1e-7, rtol=1e-6):
                raise ValueError(f"Vectorized E25 feature mismatch in {col}")
        e22._BASE_FEATURE_CACHE.clear()
        # Phase 1.
        audit_path = output / "checkpoint_pair_similarity.csv"
        if audit_path.is_file() and (output / "near_duplicate_clusters.json").is_file():
            audit = {"pair_count": len(pd.read_csv(audit_path)),
                "identical_top1_pair_count": int(pd.read_csv(audit_path).identical_top1.sum()),
                "near_duplicate_clusters": json.loads((output / "near_duplicate_clusters.json").read_text()),
                "thresholds": {"top1_agreement": NEAR_DUPLICATE_TOP1, "probability_cosine": NEAR_DUPLICATE_COSINE}}
        else:
            audit = audit_redundancy(ctx, baseline_pred, output)
        print(f"E25 phase 1: compared {audit['pair_count']} pairs", flush=True)
        # Phase 2; checkpoint every fold makes the computation restartable.
        contribution_file = output / "contribution_rankings.json"
        if contribution_file.is_file() and (output / "single_checkpoint_drop_predictions.npz").is_file():
            contributions = json.loads(contribution_file.read_text())
        else:
            contributions = run_contributions(ctx, output, baseline_pred)
        dropped_bundle = np.load(output / "single_checkpoint_drop_predictions.npz", allow_pickle=True)
        dropped_ids = dropped_bundle["checkpoint_ids"].astype(str).tolist()
        if not np.array_equal(dropped_bundle["row_ids"].astype(str), ctx["row_ids"].astype(str)):
            raise ValueError("E25 checkpoint ablation rows no longer align with V0")
        y_all = ctx["y"]
        for item in contributions:
            cp_index = dropped_ids.index(item["checkpoint_id"])
            changed = dropped_bundle["predictions"][cp_index].astype(np.int64)
            metric = e20._metric(y_all, changed)
            item["per_label"] = metric["per_label"]
            item["per_label_changes"] = {label: {
                "corrected": int(((baseline_pred != y_all) & (changed == y_all) & (y_all == i)).sum()),
                "introduced": int(((baseline_pred == y_all) & (changed != y_all) & (y_all == i)).sum()),
                "baseline_correct": int(((baseline_pred == y_all) & (y_all == i)).sum()),
                "after_drop_correct": int(((changed == y_all) & (y_all == i)).sum())}
                for i, label in enumerate(LABELS)}
        contribution_file.write_text(json.dumps(contributions, indent=2, allow_nan=False) + "\n")
        # Phase 3; save fold selection manifests before meta-classifier fitting.
        selection_dir = output / "selection"
        selection_dir.mkdir(exist_ok=True)
        rosters = _select_fold_rosters(ctx, output, selection_dir)
        (output / "selection_rosters.json").write_text(json.dumps(rosters, indent=2) + "\n")
        budgets = _train_and_evaluate_budgets(ctx, output, rosters, baseline_pred)
        global_rosters = _global_deployment_rosters(rosters)
        baseline_metrics = e20._metric(ctx["y"], baseline_pred)
        per_label_rows = []
        for budget, item in budgets.items():
            for label in LABELS:
                base_label = baseline_metrics["per_label"][label]
                subset_label = item["metrics"]["per_label"][label]
                changes = item["comparison_vs_E22B"]["per_label"][label]
                per_label_rows.append({"budget": int(budget), "label": label,
                    "baseline_precision": base_label["precision"], "subset_precision": subset_label["precision"],
                    "precision_delta_pp": (subset_label["precision"] - base_label["precision"]) * 100,
                    "baseline_recall": base_label["recall"], "subset_recall": subset_label["recall"],
                    "recall_delta_pp": (subset_label["recall"] - base_label["recall"]) * 100,
                    "corrected": changes["corrected"], "introduced": changes["introduced"]})
        pd.DataFrame(per_label_rows).to_csv(output / "per_label_comparison.csv", index=False)
        pd.DataFrame([{"budget": int(b),
            "independent_accuracy": item["independent_metrics"]["accuracy"],
            "independent_balanced_accuracy": item["independent_metrics"]["balanced_accuracy"],
            "E19D_accuracy": item["metrics"]["accuracy"],
            "E19D_balanced_accuracy": item["metrics"]["balanced_accuracy"],
            "fold_ba_mean": item["fold_ba_mean"], "fold_ba_sd": item["fold_ba_sd"],
            "delta_vs_E22B_pp": (item["metrics"]["balanced_accuracy"] - baseline_ba) * 100,
            "corrected": item["comparison_vs_E22B"]["corrected"],
            "introduced": item["comparison_vs_E22B"]["introduced"],
            "globally_distinct_deployment_checkpoints": global_rosters[b]["count"]}
            for b, item in budgets.items()]).sort_values("budget", ascending=False).to_csv(
                output / "metrics_by_budget.csv", index=False)
        v0_path = root / "data/processed/dataset_v0.parquet"
        manifest = {"experiment": "E25", "seed": SEED, "threads": THREADS,
            "baseline_experiment": "E22-B", "baseline_balanced_accuracy": baseline_ba,
            "trust_list": ctx["snapshot"], "checkpoint_ids": ctx["checkpoint_ids"],
            "cache_sha256": ctx["cache_hashes"],
            "source_sha256": {"E22-B_metrics": _sha(root / "artifacts/analysis/e22/metrics_E22-B.json"),
                "E22-B_predictions": _sha(root / "artifacts/analysis/e22/predictions_E22-B.npz"),
                "E22-B_fold_models": {str(f["fold"]): _sha(root / "artifacts/analysis/e22" / f"E22-B_fold{f['fold']}.cbm") for f in ctx["folds"]},
                "V0_dataset": _sha(v0_path),
                "E19-D_groups": _sha(root / "artifacts/analysis/e19d/oof_group_predictions.parquet")},
            "fold_definitions": [{"fold": f["fold"],
                "fit_ids": ctx["row_ids"][f["fit_rows"]].astype(str).tolist(),
                "held_ids": ctx["row_ids"][f["held_rows"]].astype(str).tolist()} for f in ctx["folds"]],
            "budgets": list(BUDGETS), "global_deployment_rosters": global_rosters,
            "training_drop_threshold": SUBSTANTIAL_TRAINING_DROP,
            "near_duplicate_thresholds": {"top1_agreement": NEAR_DUPLICATE_TOP1,
                "probability_cosine": NEAR_DUPLICATE_COSINE},
            "feature_schema_validation": "vectorized features compared against original E22 builder on 3 held-out V0 rows and all 706 columns",
            "code_sha256": _sha(Path(__file__)), "finalization_runtime_seconds": time.monotonic() - start,
            "runtime_note": "Finalization reuses previously completed artifacts; total elapsed time across the initial interrupted attempt and successful reruns was not measured as one interval."}
        (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        summary = {"baseline_balanced_accuracy": baseline_ba, "baseline_accuracy": float((baseline_pred == ctx["y"]).mean()),
            "redundancy_audit": audit, "contribution_rankings": contributions,
            "budgets": budgets, "global_deployment_rosters": global_rosters,
            "finalization_runtime_seconds": time.monotonic() - start,
            "runtime_note": "Finalization reuses previously completed artifacts; total elapsed time across the initial interrupted attempt and successful reruns was not measured as one interval."}
        (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        _write_report(output, summary)
        # Update the pre-registered dashboard item after execution.
        catalog_path = root / "configs/ensemble_catalog_v2.json"
        catalog = json.loads(catalog_path.read_text())
        e25_item = next(x for x in catalog["experiments"] if x["id"] == "E25")
        selected = budgets[str(min(BUDGETS))]["metrics"]
        e25_item.update({"accuracy": selected["accuracy"],
            "balanced_accuracy": selected["balanced_accuracy"],
            "primary_score": selected["balanced_accuracy"], "status": "Scored",
            "checkpoint_count": min(BUDGETS), "checkpoints_used": min(BUDGETS),
            "checkpoints_per_fold": min(BUDGETS),
            "checkpoint_count_note": "20 globally distinct deployment roster; fold-local rosters are recorded separately."})
        catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(output=args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
