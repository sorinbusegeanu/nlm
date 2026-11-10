"""Run ZOD03 from the versioned, complete checkpoint-label trust list."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import _load_dataset
from .zod01_fixed_cascade import _load_prediction, _resolve_fallback


def apply_zod03(*, rules: pd.DataFrame, y_true, predictions: dict[str, np.ndarray],
                fallback_id: str, rule_limit: int | None = None,
                support_growth: float = 1.2):
    """Apply ranked rules with a two-rule cap and 20% support growth.

    Support comparisons use the recorded full-validation ``prediction_count``
    and compare each rule to the previous applied rule for its label. Skipped
    rules do not become the comparison baseline for later rules.
    """
    if support_growth <= 1.0:
        raise ValueError("support_growth must be greater than 1.0")
    ranked = rules.copy()
    ranked["rank"] = pd.to_numeric(ranked["rank"], errors="coerce")
    ranked = ranked.loc[ranked["rank"].notna()].sort_values("rank", kind="stable")
    if rule_limit is not None:
        ranked = ranked.head(rule_limit)
    ranked["rank"] = ranked["rank"].astype(int)
    ranked["prediction_count"] = pd.to_numeric(ranked["prediction_count"], errors="raise").astype(int)

    y = np.asarray(y_true, dtype=object)
    pending = np.ones(len(y), dtype=bool)
    final = np.empty(len(y), dtype=object)
    previous_support: dict[str, int] = {}
    applied_rules_by_label: dict[str, int] = {}
    trace = []
    for row in ranked.to_dict("records"):
        rank = int(row["rank"])
        label = str(row["label"])
        support = int(row["prediction_count"])
        candidate_id = str(row["candidate_id"])
        if candidate_id not in predictions:
            raise KeyError(f"missing cached predictions for {candidate_id}")
        prior_support = previous_support.get(label)
        eligible = str(row.get("rule_eligible", True)).lower() in {"true", "1"}
        rule_count_for_label = applied_rules_by_label.get(label, 0)
        if not eligible:
            skip_reason = "minimum-support eligibility"
        elif rule_count_for_label >= 2:
            skip_reason = "two rules already applied for this label"
        elif prior_support is not None and support <= prior_support * support_growth:
            skip_reason = ("support did not exceed previous applied same-label rule by "
                           f"{(support_growth - 1) * 100:g}%")
        else:
            skip_reason = ""
        skipped = bool(skip_reason)
        if skipped:
            assigned = np.zeros(len(y), dtype=bool)
        else:
            assigned = pending & (np.asarray(predictions[candidate_id], dtype=object) == label)
            final[assigned] = label
            pending[assigned] = False
            previous_support[label] = support
            applied_rules_by_label[label] = rule_count_for_label + 1
        n_assigned = int(assigned.sum())
        correct = int(np.sum(y[assigned] == label))
        wrong = n_assigned - correct
        trace.append({
            "rank": rank,
            "skipped": bool(skipped),
            "skip_reason": skip_reason,
            "prediction_count": support,
            "previous_same_label_prediction_count": prior_support,
            "candidate_id": candidate_id,
            "label": label,
            "assigned": n_assigned,
            "correct": correct,
            "wrong": wrong,
            "accuracy": correct / n_assigned if n_assigned else None,
            "rows_left": int(pending.sum()),
        })

    remaining = np.flatnonzero(pending)
    if fallback_id not in predictions:
        raise KeyError(f"missing cached fallback predictions for {fallback_id}")
    fallback_pred = np.asarray(predictions[fallback_id], dtype=object)
    final[remaining] = fallback_pred[remaining]
    fallback_correct = int(np.sum(final[remaining] == y[remaining]))
    trace.append({
        "rank": "Fallback", "skipped": False, "prediction_count": None,
        "previous_same_label_prediction_count": None, "candidate_id": fallback_id,
        "label": "fallback", "assigned": int(len(remaining)),
        "correct": fallback_correct, "wrong": int(len(remaining) - fallback_correct),
        "accuracy": fallback_correct / len(remaining) if len(remaining) else None,
        "rows_left": 0,
    })
    return final, pd.DataFrame(trace)


def run_zod03(*, data_dir="data/processed",
              trust_list_path="docs/checkpoint_label_trust_list.csv",
              predictions_dir="artifacts/analysis/zod01_specialist_discovery/predictions",
              registry_path="experiments/registry.sqlite", rule_limit=None,
              support_growth=1.2,
              report_path="docs/zod_03.md",
              trace_path="artifacts/analysis/zod03/rule_trace.csv"):
    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.eq("validation"), ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    validation["y_true"] = validation.pop("model").astype(str)
    if len(validation) != 540:
        raise ValueError(f"expected frozen V0 validation with 540 rows; got {len(validation)}")

    trust = pd.read_csv(trust_list_path, dtype=str, keep_default_na=False)
    required = {"SeqID", "trust_list_version", "trust_list_generated_at_utc", "experiment_id",
                "target_label", "prediction_count", "rule_eligible"}
    if not required.issubset(trust.columns):
        raise ValueError(f"trust list missing fields: {sorted(required - set(trust.columns))}")
    version = str(trust.trust_list_version.iloc[0])
    generated_at = str(trust.trust_list_generated_at_utc.iloc[0])
    rules = trust.rename(columns={"SeqID": "rank", "experiment_id": "candidate_id",
                                  "target_label": "label"}).copy()
    rules["rank"] = pd.to_numeric(rules["rank"], errors="raise")
    rules = rules.sort_values("rank", kind="stable")
    if rule_limit is not None:
        rules = rules.head(rule_limit)
    classes = sorted(validation.y_true.unique().tolist())
    fallback = _resolve_fallback(registry_path)
    fallback_id = str(fallback["experiment_id"])
    candidate_ids = list(dict.fromkeys([*rules.candidate_id.astype(str).tolist(), fallback_id]))
    predictions = {}
    prediction_root = Path(predictions_dir)
    for candidate_id in candidate_ids:
        filename = f"{candidate_id}_validation.parquet"
        path = next((root / filename for root in (prediction_root, Path("artifacts/predictions"))
                     if (root / filename).is_file()), None)
        if path is None:
            path = next(iter(Path("artifacts/runs").rglob(f"{candidate_id}*validation*.parquet")), None)
        if path is None:
            raise FileNotFoundError(f"no cached validation predictions for {candidate_id}")
        predictions[candidate_id] = _load_prediction(path, validation, classes, candidate_id)

    final, trace = apply_zod03(rules=rules, y_true=validation.y_true.to_numpy(),
                               predictions=predictions, fallback_id=fallback_id,
                               rule_limit=rule_limit, support_growth=support_growth)
    correct_total = int(np.sum(final == validation.y_true.to_numpy()))
    accuracy_total = correct_total / len(validation)
    report = [
        f"# ZOD03 — {version}, {support_growth:.0%} support threshold ({generated_at})",
        "",
        f"**{correct_total}/{len(validation)} correct ({accuracy_total:.2%})**",
        "",
        "| Rule | Model label | Correct | Wrong | Rule accuracy | Rows left to identify |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in trace.to_dict("records"):
        if row["rank"] == "Fallback":
            rule_name = "Fallback"
        else:
            rule_name = f"{row['rank']} (skipped)" if row["skipped"] else str(row["rank"])
        label = "Fallback" if row["rank"] == "Fallback" else row["label"]
        accuracy = "—" if pd.isna(row["accuracy"]) else f"{row['accuracy']:.2%}"
        report.append(f"| {rule_name} | {label} | {row['correct']} | {row['wrong']} | {accuracy} | {row['rows_left']} |")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(trace_path, index=False)
    return {
        "validation_rows": int(len(validation)),
        "rule_count": int(len(rules)),
        "skipped_rules": int(trace.loc[trace["rank"] != "Fallback", "skipped"].sum()),
        "correct": correct_total,
        "accuracy": float(accuracy_total),
        "fallback_id": fallback_id,
        "support_growth": float(support_growth),
        "trust_list_version": version,
        "trust_list_generated_at_utc": generated_at,
        "report_path": str(report_path),
        "trace_path": str(trace_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ZOD03 from the versioned checkpoint-label trust list")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--trust-list", default="docs/checkpoint_label_trust_list.csv")
    parser.add_argument("--predictions-dir", default="artifacts/analysis/zod01_specialist_discovery/predictions")
    parser.add_argument("--registry", default="experiments/registry.sqlite")
    parser.add_argument("--rule-limit", type=int, default=None,
                        help="optional prefix limit; default uses the complete trust list")
    parser.add_argument("--support-growth", type=float, default=1.2,
                        help="minimum required prediction-support multiplier for repeated labels")
    parser.add_argument("--report", default="docs/zod_03.md")
    parser.add_argument("--trace", default="artifacts/analysis/zod03/rule_trace.csv")
    args = parser.parse_args()
    result = run_zod03(data_dir=args.data_dir, trust_list_path=args.trust_list,
        predictions_dir=args.predictions_dir,
        registry_path=args.registry, rule_limit=args.rule_limit,
        support_growth=args.support_growth,
        report_path=args.report, trace_path=args.trace)
    print(result)


if __name__ == "__main__":
    main()
