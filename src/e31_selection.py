"""Prompt-group cross-fitted selection of E31's 23 roster additions."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .e30_selection import E30GreedySelector, _metric


def _paired(y, before, after):
    y = np.asarray(y, dtype=np.int64)
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    if not (len(y) == len(before) == len(after)):
        raise ValueError("E31 paired predictions are misaligned")
    return {"corrected": int(((before != y) & (after == y)).sum()),
            "introduced": int(((before == y) & (after != y)).sum()),
            "changed": int((before != after).sum()),
            "net_correct": int((after == y).sum() - (before == y).sum())}


class E31GreedySelector(E30GreedySelector):
    """Reuse E30's nested router fits/caches but fill the requested 40 slots.

    Every fold chooses 23 candidates sequentially by inner out-of-fold balanced
    accuracy. A fixed deployment roster is then formed by fold consensus. Unlike
    E30's stop rule, E31 does not discard negative-gain additions: exactly 23
    additions are required to reach 40, and their measured gains remain in the
    report.
    """

    def __init__(self, *args, candidate_workers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_workers = max(1, int(candidate_workers))

    def run(self, fold_definitions):
        fold_outputs = []
        event_records = []
        candidate_eval_records = []
        outer_step_preds = {}
        progress_path = self.output_dir / "selection_progress.json"

        for fold_number, (fit_rows, held_rows) in enumerate(fold_definitions, 1):
            print(f"E31 outer fold {fold_number}/{self.outer_folds}: selection start",
                  flush=True)
            current = self.baseline_roster.copy()
            current_metric, current_pred = self.evaluate_roster(
                fold_number, fit_rows, current)
            available = self.candidates.copy()
            selected = []
            fold_steps = []

            for step in range(1, self.steps + 1):
                candidate_metrics = []
                y_fit = self.truth[fit_rows]
                def evaluate_candidate(candidate):
                    metric, pred = self.evaluate_roster(
                        fold_number, fit_rows, current + [candidate])
                    paired = _paired(y_fit, current_pred, pred)
                    record = {"outer_fold": fold_number, "step": step,
                        "checkpoint_id": candidate,
                        "previous_inner_oof_balanced_accuracy": current_metric["balanced_accuracy"],
                        "candidate_inner_oof_balanced_accuracy": metric["balanced_accuracy"],
                        "marginal_balanced_accuracy": metric["balanced_accuracy"] -
                            current_metric["balanced_accuracy"], **paired}
                    return candidate, metric, pred, paired, record

                if self.candidate_workers == 1:
                    completed = [evaluate_candidate(candidate) for candidate in available]
                else:
                    # Evaluations are independent for a fixed roster step. The
                    # results are reassembled in checkpoint-ID order, so tie
                    # breaking and output ordering remain deterministic.
                    with ThreadPoolExecutor(max_workers=self.candidate_workers) as pool:
                        completed = list(pool.map(evaluate_candidate, available))
                for candidate_index, (candidate, metric, pred, paired, record) in enumerate(completed, 1):
                    candidate_eval_records.append(record)
                    candidate_metrics.append((candidate, metric, pred, paired))
                    if candidate_index % 10 == 0 or candidate_index == len(available):
                        print(f"E31 fold {fold_number} step {step}: scored "
                              f"{candidate_index}/{len(available)}", flush=True)

                # BA is primary. Paired net-correct and introduced-error counts
                # resolve exact BA ties; checkpoint ID makes the result stable.
                chosen, chosen_metric, chosen_pred, chosen_pair = min(
                    candidate_metrics,
                    key=lambda item: (-item[1]["balanced_accuracy"],
                        -item[3]["net_correct"], item[3]["introduced"], item[0]))
                delta = chosen_metric["balanced_accuracy"] - current_metric["balanced_accuracy"]
                selected.append(chosen)
                event = {"outer_fold": fold_number, "step": step,
                    "checkpoint_id": chosen,
                    "inner_oof_balanced_accuracy": chosen_metric["balanced_accuracy"],
                    "previous_inner_oof_balanced_accuracy": current_metric["balanced_accuracy"],
                    "marginal_balanced_accuracy": delta, **chosen_pair}
                event_records.append(event)
                fold_steps.append({"step": step, "selected_checkpoint": chosen,
                    "inner_oof": chosen_metric, "previous_inner_oof": current_metric,
                    "marginal_balanced_accuracy": delta,
                    "comparison_vs_previous_roster": chosen_pair,
                    "candidate_scores": [{"checkpoint_id": sid, **m, **pair}
                        for sid, m, _, pair in candidate_metrics]})
                current.append(chosen)
                current_metric, current_pred = chosen_metric, chosen_pred
                available.remove(chosen)
                print(f"E31 fold {fold_number} step {step}: selected {chosen}; "
                      f"inner OOF BA={chosen_metric['balanced_accuracy']:.6f}, "
                      f"delta={delta:+.6f}", flush=True)
                # Candidate-level caches make an interrupted fold restartable;
                # this progress snapshot also records each completed greedy step.
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(json.dumps({"completed_outer_folds": fold_outputs,
                    "active_fold": fold_number, "active_fold_completed_steps": fold_steps,
                    "selection_events": event_records,
                    "candidate_evaluations": candidate_eval_records}, indent=2) + "\n")

            fold_result = {"fold": fold_number,
                "fit_ids": self.row_ids[fit_rows].tolist(),
                "held_ids": self.row_ids[held_rows].tolist(),
                "fold_additions": selected, "steps": fold_steps,
                "outer_metrics_by_step": {}}
            fold_roster = self.baseline_roster.copy()
            prior_held = None
            for step in range(0, self.steps + 1):
                if step:
                    fold_roster.append(selected[step - 1])
                pred, _ = self._outer_predict(fold_roster, fit_rows, held_rows)
                outer_step_preds.setdefault(str(step), {})[str(fold_number)] = {
                    "truth": self.truth[held_rows].tolist(), "predictions": pred.tolist(),
                    "row_ids": self.row_ids[held_rows].tolist()}
                outer_metric = _metric(self.truth[held_rows], pred)
                fold_result["outer_metrics_by_step"][str(step)] = outer_metric
                if prior_held is not None:
                    fold_result["outer_metrics_by_step"][str(step)][
                        "comparison_vs_previous_roster"] = _paired(
                            self.truth[held_rows], prior_held, pred)
                prior_held = pred
            fold_outputs.append(fold_result)
            progress_path.write_text(json.dumps({"completed_outer_folds": fold_outputs,
                "selection_events": event_records,
                "candidate_evaluations": candidate_eval_records}, indent=2) + "\n")

        return self.aggregate(fold_outputs, event_records, candidate_eval_records,
                              outer_step_preds)

    def aggregate(self, fold_outputs, events, candidate_evaluations, outer_step_preds):
        selected_by_candidate = {}
        all_gains = {}
        all_comparisons = {}
        for event in events:
            selected_by_candidate.setdefault(event["checkpoint_id"], []).append(event)
        for record in candidate_evaluations:
            sid = record["checkpoint_id"]
            all_gains.setdefault(sid, []).append(record["marginal_balanced_accuracy"])
            all_comparisons.setdefault(sid, []).append(record)

        ranking = []
        for candidate in self.candidates:
            selected = selected_by_candidate.get(candidate, [])
            gains = [float(x["marginal_balanced_accuracy"]) for x in selected]
            evaluated = [float(x) for x in all_gains.get(candidate, [])]
            ranking.append({"checkpoint_id": candidate,
                "selection_frequency": len(selected),
                "mean_selection_rank": float(np.mean([x["step"] for x in selected]))
                    if selected else None,
                "mean_selected_marginal_balanced_accuracy": float(np.mean(gains))
                    if gains else None,
                "mean_evaluated_marginal_balanced_accuracy": float(np.mean(evaluated))
                    if evaluated else None,
                "selected_marginal_gains": gains,
                "candidate_evaluation_count": len(evaluated),
                "selected_corrections": int(sum(x["corrected"] for x in selected)),
                "selected_introduced": int(sum(x["introduced"] for x in selected)),
                "selected_net_correct": int(sum(x["net_correct"] for x in selected))})

        # Consensus order: frequency, selected rank, selected marginal BA,
        # then mean evaluated marginal BA for candidates never selected, ID.
        ranking.sort(key=lambda x: (-x["selection_frequency"],
            x["mean_selection_rank"] if x["mean_selection_rank"] is not None else float("inf"),
            -(x["mean_selected_marginal_balanced_accuracy"]
              if x["mean_selected_marginal_balanced_accuracy"] is not None
              else (x["mean_evaluated_marginal_balanced_accuracy"] or 0.0)),
            x["checkpoint_id"]))
        additions = [x["checkpoint_id"] for x in ranking[:self.steps]]
        if len(additions) != self.steps or len(set(additions)) != self.steps:
            raise AssertionError("E31 consensus did not produce exactly 23 unique additions")

        outer_summary = {}
        for step, per_fold in outer_step_preds.items():
            y = np.concatenate([np.asarray(per_fold[str(f)]["truth"], dtype=np.int64)
                                for f in sorted(map(int, per_fold))])
            pred = np.concatenate([np.asarray(per_fold[str(f)]["predictions"], dtype=np.int64)
                                   for f in sorted(map(int, per_fold))])
            outer_summary[step] = {**_metric(y, pred)}
            if int(step) > 0:
                prior = np.concatenate([np.asarray(outer_step_preds[str(int(step)-1)][str(f)][
                    "predictions"], dtype=np.int64) for f in sorted(map(int, per_fold))])
                outer_summary[step]["comparison_vs_previous_roster"] = _paired(y, prior, pred)

        size_metrics = {str(len(self.baseline_roster) + int(step)): metrics
                        for step, metrics in outer_summary.items()}
        eligible_sizes = {int(size): metrics for size, metrics in size_metrics.items()
                          if 20 <= int(size) <= 40}
        best_size = min(eligible_sizes, key=lambda size: (
            -eligible_sizes[size]["balanced_accuracy"], size))
        result = {"baseline_roster": self.baseline_roster,
            "candidate_pool": self.candidates,
            "outer_folds": fold_outputs, "selection_events": events,
            "candidate_evaluations": candidate_evaluations,
            "candidate_ranking": ranking,
            "consensus_additions": additions,
            "final_roster": self.baseline_roster + additions,
            "outer_oof_metrics_by_addition": outer_summary,
            "outer_oof_metrics_by_roster_size": size_metrics,
            "outer_step_predictions": outer_step_preds,
            "best_oof_roster_size_20_to_40": best_size,
            "best_oof_balanced_accuracy_20_to_40": eligible_sizes[best_size]["balanced_accuracy"],
            "selection_rule": "23 fixed additions: descending fold selection frequency, ascending mean selected rank, descending mean selected marginal BA; never-selected candidates use mean evaluated marginal BA; checkpoint ID is deterministic final tie-break.",
            "metric_tie_break": "Greedy candidate BA first, then inner-OOF net correct, fewer introduced errors, checkpoint ID.",
            "incomplete_progress_path": str(self.output_dir / "selection_progress.json")}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "selection_results.json").write_text(
            json.dumps(result, indent=2) + "\n")
        return result
