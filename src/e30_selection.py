"""Leakage-controlled greedy checkpoint selection for the E30 E13-B roster."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import e13_reliability_router as e13
from .e13_b_inference import family_max_scores, make_features, route_family_max
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS


def _digest_ids(roster):
    return hashlib.sha256("\n".join(map(str, roster)).encode()).hexdigest()[:16]


def _metric(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


class E30GreedySelector:
    """Select additions inside outer training folds using nested group CV.

    Candidate caches store only inner OOF outputs. All reliability statistics and
    router training rows are constructed from prompt-disjoint fitting groups.
    """

    def __init__(self, v0, members, probabilities, baseline_roster, output_dir,
                 *, outer_folds=5, inner_folds=4, steps=5, iterations=119):
        self.v0 = v0.reset_index(drop=True)
        self.members = members
        self.probabilities = probabilities
        self.baseline_roster = list(map(str, baseline_roster))
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "candidate_oof_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.outer_folds = int(outer_folds)
        self.inner_folds = int(inner_folds)
        self.steps = int(steps)
        self.iterations = int(iterations)
        self.row_ids = self.v0.id.astype(str).to_numpy()
        self.truth = pd.Categorical(self.v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
        self.groups = self.v0.prompt_id.astype(str).to_numpy()
        if (self.truth < 0).any() or not set(self.baseline_roster) <= set(probabilities):
            raise ValueError("E30 selector baseline roster or labels are invalid")
        self.candidates = sorted(set(map(str, probabilities)) - set(self.baseline_roster),
                                 key=lambda sid: int(sid[1:]))
        if not np.allclose(np.stack([probabilities[s] for s in probabilities]).sum(2), 1.0,
                           atol=1e-4):
            raise ValueError("E30 checkpoint probabilities are not normalized")

    def _roster_arrays(self, roster):
        ids = list(roster)
        p = np.stack([self.probabilities[sid] for sid in ids]).astype(np.float64)
        families = [_model_family(self.members[sid]["model"]) for sid in ids]
        return ids, p, families

    def _fit_router(self, roster, fit_rows):
        """Fit a deployment-equivalent router on fit_rows using four-fold OOF features."""
        ids, p, families = self._roster_arrays(roster)
        xs, ys = [], []
        splitter = GroupKFold(n_splits=self.inner_folds)
        for local_fit, local_held in splitter.split(fit_rows, self.truth[fit_rows], self.groups[fit_rows]):
            stat_rows, held_rows = fit_rows[local_fit], fit_rows[local_held]
            frame, _ = e13.build_feature_frame(p, ids, families, self.truth[stat_rows],
                                               stat_rows, held_rows)
            hard = p[:, held_rows].argmax(axis=2)
            xs.append(frame)
            ys.append(e13._targets(hard, self.truth[held_rows]))
        x = pd.concat(xs, ignore_index=True)
        y = np.concatenate(ys)
        router = e13._new_catboost(iterations=self.iterations)
        router.fit(e13._cat_frame(x), y, cat_features=list(e13.CATEGORICAL), verbose=False)
        return router, ids, p, families

    def _predict_fitted(self, router, ids, p, families, fit_rows, target_rows):
        target_x, _ = e13.build_feature_frame(p, ids, families, self.truth[fit_rows],
                                              fit_rows, target_rows)
        reliability = np.asarray(router.predict_proba(e13._cat_frame(target_x))[:, 1],
                                 dtype=np.float64).reshape(len(target_rows), len(ids)).T
        target_p = p[:, target_rows]
        hard = target_p.argmax(axis=2)
        family_p = e13.family_balanced_probabilities(target_p, families)
        pred, _ = e13.route_predictions("family_max", hard, reliability, family_p, families)
        raw_scores, safe_scores = family_max_scores(hard, reliability, family_p, families)
        shared_pred = route_family_max(hard, reliability, family_p, families)
        if not np.array_equal(pred, shared_pred):
            raise AssertionError("E30 shared and trainer family-max predictions differ")
        return pred, safe_scores

    def _cache_path(self, fold_number, roster):
        return self.cache_dir / f"outer_{fold_number:02d}_{_digest_ids(roster)}.npz"

    def evaluate_roster(self, fold_number, outer_train, roster):
        roster = list(roster)
        path = self._cache_path(fold_number, roster)
        if path.exists():
            with np.load(path, allow_pickle=True) as saved:
                if saved["row_ids"].astype(str).tolist() != self.row_ids[outer_train].tolist():
                    raise ValueError(f"Stale E30 candidate cache: {path}")
                pred = saved["predictions"].astype(np.int64)
            return _metric(self.truth[outer_train], pred), pred
        pred = np.full(len(outer_train), -1, dtype=np.int64)
        global_to_local = {int(row): i for i, row in enumerate(outer_train)}
        for local_fit, local_valid in GroupKFold(n_splits=self.inner_folds).split(
                outer_train, self.truth[outer_train], self.groups[outer_train]):
            inner_fit, inner_valid = outer_train[local_fit], outer_train[local_valid]
            router, ids, p, families = self._fit_router(roster, inner_fit)
            inner_pred, _ = self._predict_fitted(router, ids, p, families, inner_fit, inner_valid)
            pred[[global_to_local[int(row)] for row in inner_valid]] = inner_pred
            del router
        if (pred < 0).any():
            raise AssertionError("Inner prompt-group folds did not predict all outer training rows")
        metric = _metric(self.truth[outer_train], pred)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, row_ids=self.row_ids[outer_train].astype(str), predictions=pred)
        tmp.replace(path)
        return metric, pred

    def _outer_predict(self, roster, fit_rows, held_rows):
        router, ids, p, families = self._fit_router(roster, fit_rows)
        pred, scores = self._predict_fitted(router, ids, p, families, fit_rows, held_rows)
        return pred, scores

    def run(self, fold_definitions):
        fold_outputs = []
        event_records = []
        candidate_eval_records = []
        outer_step_preds = {}
        for fold_number, (fit_rows, held_rows) in enumerate(fold_definitions, 1):
            print(f"E30 outer fold {fold_number}/{self.outer_folds}: selection start", flush=True)
            selected = []
            current = self.baseline_roster.copy()
            current_metric, _ = self.evaluate_roster(fold_number, fit_rows, current)
            fold_steps = []
            available = self.candidates.copy()
            for step in range(1, self.steps + 1):
                candidate_metrics = []
                for ci, candidate in enumerate(available, 1):
                    roster = current + [candidate]
                    metric, _ = self.evaluate_roster(fold_number, fit_rows, roster)
                    candidate_metrics.append((candidate, metric))
                    candidate_eval_records.append({"outer_fold": fold_number, "step": step,
                        "checkpoint_id": candidate,
                        "previous_inner_oof_balanced_accuracy": current_metric["balanced_accuracy"],
                        "candidate_inner_oof_balanced_accuracy": metric["balanced_accuracy"],
                        "marginal_balanced_accuracy": metric["balanced_accuracy"] -
                            current_metric["balanced_accuracy"]})
                    if ci % 10 == 0 or ci == len(available):
                        print(f"E30 fold {fold_number} step {step}: scored {ci}/{len(available)}",
                              flush=True)
                # Maximize inner held-out balanced accuracy; checkpoint ID is the stable tie break.
                chosen, chosen_metric = sorted(candidate_metrics,
                    key=lambda item: (-item[1]["balanced_accuracy"], item[0]))[0]
                delta = chosen_metric["balanced_accuracy"] - current_metric["balanced_accuracy"]
                selected.append(chosen)
                event_records.append({"outer_fold": fold_number, "step": step,
                    "checkpoint_id": chosen, "inner_oof_balanced_accuracy": chosen_metric["balanced_accuracy"],
                    "previous_inner_oof_balanced_accuracy": current_metric["balanced_accuracy"],
                    "marginal_balanced_accuracy": delta})
                fold_steps.append({"step": step, "selected_checkpoint": chosen,
                    "inner_oof": chosen_metric, "previous_inner_oof": current_metric,
                    "marginal_balanced_accuracy": delta,
                    "candidate_scores": [{"checkpoint_id": sid, **m} for sid, m in candidate_metrics]})
                current.append(chosen)
                current_metric = chosen_metric
                available.remove(chosen)
                print(f"E30 outer fold {fold_number} step {step}: selected {chosen}, "
                      f"inner OOF BA={chosen_metric['balanced_accuracy']:.6f}, delta={delta:+.6f}",
                      flush=True)
            fold_result = {"fold": fold_number, "fit_ids": self.row_ids[fit_rows].tolist(),
                "held_ids": self.row_ids[held_rows].tolist(), "fold_additions": selected,
                "steps": fold_steps, "outer_metrics_by_step": {}}
            fold_roster = self.baseline_roster.copy()
            for step in range(0, len(selected) + 1):
                if step:
                    fold_roster.append(selected[step - 1])
                pred, _ = self._outer_predict(fold_roster, fit_rows, held_rows)
                outer_step_preds.setdefault(str(step), {})[str(fold_number)] = {
                    "truth": self.truth[held_rows].tolist(), "predictions": pred.tolist(),
                    "row_ids": self.row_ids[held_rows].tolist()}
                fold_result["outer_metrics_by_step"][str(step)] = _metric(self.truth[held_rows], pred)
            fold_outputs.append(fold_result)
            state = {"folds": fold_outputs, "selection_events": event_records,
                     "candidate_evaluations": candidate_eval_records,
                     "outer_step_predictions": outer_step_preds}
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "selection_progress.json").write_text(json.dumps(state, indent=2) + "\n")
        return self.aggregate(fold_outputs, event_records, candidate_eval_records,
                              outer_step_preds)

    def aggregate(self, fold_outputs, events, candidate_evaluations, outer_step_preds):
        grouped = {}; gains = {}
        for event in events:
            grouped.setdefault(event["checkpoint_id"], []).append(event)
        for record in candidate_evaluations:
            gains.setdefault(record["checkpoint_id"], []).append(
                record["marginal_balanced_accuracy"])
        ranking = []
        for candidate in self.candidates:
            records = grouped.get(candidate, [])
            ranking.append({"checkpoint_id": candidate, "selection_frequency": len(records),
                "mean_selection_rank": (float(np.mean([r["step"] for r in records]))
                                         if records else None),
                "mean_marginal_balanced_accuracy": (float(np.mean(gains[candidate]))
                                                     if gains.get(candidate) else None),
                "marginal_evaluation_count": len(gains.get(candidate, [])),
                "marginal_gains_by_fold": {str(r["outer_fold"]): r["marginal_balanced_accuracy"]
                                           for r in records}})
        ranking.sort(key=lambda x: (-x["selection_frequency"],
            x["mean_selection_rank"] if x["mean_selection_rank"] is not None else float("inf"),
            -(x["mean_marginal_balanced_accuracy"] or 0.0), x["checkpoint_id"]))
        qualified = [r for r in ranking if r["selection_frequency"] > 0 and
                     r["mean_marginal_balanced_accuracy"] is not None and
                     r["mean_marginal_balanced_accuracy"] > 0]
        additions = [r["checkpoint_id"] for r in qualified[:self.steps]]
        roster = self.baseline_roster + additions
        outer_summary = {}
        for step, per_fold in outer_step_preds.items():
            y = np.concatenate([np.asarray(per_fold[str(f)]["truth"], dtype=np.int64)
                                for f in sorted(map(int, per_fold))])
            pred = np.concatenate([np.asarray(per_fold[str(f)]["predictions"], dtype=np.int64)
                                   for f in sorted(map(int, per_fold))])
            outer_summary[step] = _metric(y, pred)
        result = {"baseline_roster": self.baseline_roster, "candidate_pool": self.candidates,
            "outer_folds": fold_outputs, "selection_events": events,
            "candidate_evaluations": candidate_evaluations,
            "candidate_ranking": ranking, "qualified_additions": additions,
            "final_roster": roster, "outer_oof_metrics_by_step": outer_summary,
            "outer_step_predictions": outer_step_preds,
            "selection_rule": "descending selection frequency, ascending mean selection rank, descending mean marginal BA, then checkpoint ID; only positive mean marginal BA candidates are deployable"}
        (self.output_dir / "selection_results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
