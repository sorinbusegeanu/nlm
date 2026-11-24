"""E32 greedy, prompt-disjoint additions on top of the frozen E31 roster."""
from __future__ import annotations

import json

from .e31_selection import E31GreedySelector


class E32GreedySelector(E31GreedySelector):
    """Reuse E31 nested selection while extending its 40-member roster by 20."""

    def aggregate(self, fold_outputs, events, candidate_evaluations, outer_step_preds):
        result = super().aggregate(fold_outputs, events, candidate_evaluations,
                                   outer_step_preds)
        metrics = result["outer_oof_metrics_by_roster_size"]
        eligible = {int(size): value for size, value in metrics.items()
                    if 40 <= int(size) <= 60}
        best_size = min(eligible, key=lambda size: (
            -eligible[size]["balanced_accuracy"], size))
        result.pop("best_oof_roster_size_20_to_40", None)
        result.pop("best_oof_balanced_accuracy_20_to_40", None)
        result["best_oof_roster_size_40_to_60"] = best_size
        result["best_oof_balanced_accuracy_40_to_60"] = eligible[best_size]["balanced_accuracy"]
        result["target_roster_size"] = 60
        result["baseline_experiment"] = "E31 fixed 40-checkpoint deployment"
        result["selection_rule"] = (
            "20 fixed additions selected fold-locally by inner OOF balanced accuracy; "
            "consensus order is descending fold selection frequency, ascending mean "
            "selected rank, descending selected marginal BA, then checkpoint ID. "
            "Never-selected candidates use mean evaluated marginal BA.")
        target = self.output_dir / "selection_results.json"
        target.write_text(json.dumps(result, indent=2) + "\n")
        return result
