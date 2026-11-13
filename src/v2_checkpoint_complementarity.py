"""Read-only complementarity analysis for the frozen V2 checkpoint trust list.

This analysis consumes existing full-V0 prediction artifacts. It never loads a
checkpoint and does not modify the trust-list source files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRUST_LIST = Path("docs/trust-list-2-v001.csv")
DEFAULT_TRUST_METADATA = Path("docs/trust-list-2-v001.json")
DEFAULT_CACHE_DIR = Path("artifacts/analysis/trust_list_2_v0")
DEFAULT_ARTIFACT_DIR = Path("artifacts/predictions")
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/trust_list_2_v0")
DEFAULT_REPORT = Path("docs/trust_list_v2_complementarity.md")


def _sha256_v0(frame: pd.DataFrame) -> str:
    columns = ["id", "prompt_id", "text", "model"]
    canonical = frame[columns].astype(str).sort_values("id", kind="stable").reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _resolve_prediction_path(experiment_id: str, cache_dir: Path,
                             artifact_dir: Path) -> Path:
    cached = cache_dir / f"{experiment_id}.parquet"
    if cached.is_file():
        return cached
    trained = artifact_dir / f"{experiment_id}_validation.parquet"
    if trained.is_file():
        return trained
    raise FileNotFoundError(
        f"No saved full-V0 prediction parquet found for {experiment_id}; checked {cached} and {trained}"
    )


def _load_predictions(trust: pd.DataFrame, *, v0: pd.DataFrame,
                      cache_dir: Path, artifact_dir: Path,
                      labels: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    predictions: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    for experiment_id in trust.experiment_id.astype(str).drop_duplicates():
        predictions[experiment_id], sources[experiment_id] = _load_single_prediction(
            experiment_id, v0=v0, cache_dir=cache_dir, artifact_dir=artifact_dir,
            labels=labels)
        rows = trust.loc[trust.experiment_id.astype(str).eq(experiment_id)]
        expected_y = v0.model.astype(str).to_numpy(dtype=object)
        prediction = predictions[experiment_id]
        for rule in rows.itertuples(index=False):
            mask = prediction == str(rule.target_label)
            support = int(mask.sum())
            tp = int(np.sum(mask & (expected_y == str(rule.target_label))))
            measured_precision = tp / support if support else np.nan
            if support != int(rule.prediction_count):
                raise ValueError(f"prediction support disagrees with trust list for {experiment_id}/{rule.target_label}")
            if support and not np.isclose(measured_precision, float(rule.precision), atol=1e-10, rtol=0):
                raise ValueError(f"precision disagrees with trust list for {experiment_id}/{rule.target_label}")
    return predictions, sources


def _load_single_prediction(experiment_id: str, *, v0: pd.DataFrame,
                            cache_dir: Path, artifact_dir: Path,
                            labels: list[str]) -> tuple[np.ndarray, str]:
    """Load and align one saved top-1/probability prediction file to full V0."""
    expected_ids = v0.id.astype(str).tolist()
    expected_y = v0.model.astype(str).to_numpy(dtype=object)
    expected_prompts = v0.prompt_id.astype(str).to_numpy(dtype=object)
    path = _resolve_prediction_path(experiment_id, cache_dir, artifact_dir)
    frame = pd.read_parquet(path)
    required = {"id", "y_true"}
    if missing := required - set(frame.columns):
        raise ValueError(f"prediction file {path} is missing {sorted(missing)}")
    frame = frame.copy()
    frame["id"] = frame.id.astype(str)
    frame["y_true"] = frame.y_true.astype(str)
    if frame.id.duplicated().any() or len(frame) != len(v0) or set(frame.id) != set(expected_ids):
        raise ValueError(f"prediction IDs do not exactly match full V0 for {experiment_id}: {path}")
    aligned = frame.set_index("id").loc[expected_ids]
    if not np.array_equal(aligned.y_true.to_numpy(dtype=object), expected_y):
        raise ValueError(f"true labels do not align with full V0 for {experiment_id}: {path}")
    if "prompt_id" in aligned and not np.array_equal(
            aligned.prompt_id.astype(str).to_numpy(dtype=object), expected_prompts):
        raise ValueError(f"prompt IDs do not align with full V0 for {experiment_id}: {path}")
    if "y_pred" in aligned:
        prediction = aligned.y_pred.astype(str).to_numpy(dtype=object)
    else:
        probability_columns = [f"prob::{label}" for label in labels]
        if missing := set(probability_columns) - set(aligned.columns):
            raise ValueError(f"prediction file {path} lacks probability columns {sorted(missing)}")
        probabilities = aligned[probability_columns].to_numpy(dtype=float)
        if probabilities.shape != (len(v0), len(labels)) or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid probabilities for {experiment_id}: {path}")
        prediction = np.asarray(labels, dtype=object)[probabilities.argmax(axis=1)]
    if not set(prediction).issubset(set(labels)):
        raise ValueError(f"unknown predicted label in {path}")
    return prediction, str(path)


def _is_eligible(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _skip_reason(row) -> str:
    if not _is_eligible(row.evaluation_eligible):
        return "not evaluation eligible"
    overlap = pd.to_numeric(row.validation_training_overlap, errors="coerce")
    if pd.notna(overlap) and float(overlap) != 0:
        return "validation training overlap"
    if int(row.prediction_count) < 10:
        return "minimum support below 10"
    return ""


def _is_eligible_fallback(row) -> bool:
    return _is_eligible(row.rule_eligible) and not _skip_reason(row)


def _metrics(y_true: np.ndarray, prediction: np.ndarray, labels: list[str]) -> dict:
    recalls = {
        label: float(np.mean(prediction[y_true == label] == label))
        for label in labels
    }
    correct = int(np.sum(prediction == y_true))
    return {
        "rows": int(len(y_true)),
        "correct": correct,
        "incorrect": int(len(y_true) - correct),
        "accuracy": float(correct / len(y_true)) if len(y_true) else 0.0,
        "balanced_accuracy": float(np.mean(list(recalls.values()))) if recalls else 0.0,
        "per_label_recall": recalls,
    }


def _replay_fixed_order(trust: pd.DataFrame, predictions: dict[str, np.ndarray],
                        y_true: np.ndarray) -> tuple[pd.DataFrame, dict, np.ndarray]:
    unclaimed = np.ones(len(y_true), dtype=bool)
    output = np.full(len(y_true), None, dtype=object)
    audit = []
    for row in trust.itertuples(index=False):
        before = int(unclaimed.sum())
        eligible = _is_eligible_fallback(row)
        assigned = tp = fp = 0
        if eligible:
            prediction = predictions[str(row.experiment_id)]
            indexes = np.flatnonzero(unclaimed & (prediction == str(row.target_label)))
            assigned = int(len(indexes))
            tp = int(np.sum(y_true[indexes] == str(row.target_label)))
            fp = assigned - tp
            output[indexes] = str(row.target_label)
            unclaimed[indexes] = False
        support = int(row.prediction_count)
        global_precision = float(row.true_positive) / support if support else np.nan
        incremental_precision = tp / assigned if assigned else np.nan
        audit.append({
            "SeqID": int(row.SeqID), "sequence_id": str(row.sequence_id),
            "experiment_id": str(row.experiment_id), "model": str(row.model),
            "training_dataset": str(row.training_dataset), "label": str(row.target_label),
            "eligible": eligible, "skip_reason": "" if eligible else _skip_reason(row),
            "global_prediction_count": support, "global_true_positive": int(row.true_positive),
            "global_false_positive": int(row.false_positive),
            "global_precision": global_precision,
            "global_smoothed_precision": float(row.smoothed_precision),
            "unclaimed_before": before, "incremental_assigned": assigned,
            "incremental_true_positive": tp, "incremental_false_positive": fp,
            "incremental_precision": incremental_precision,
            "incremental_precision_delta": (incremental_precision - global_precision
                                              if assigned and support else np.nan),
            "cumulative_claimed": len(y_true) - int(unclaimed.sum()),
            "cumulative_correct": int(sum(item["incremental_true_positive"] for item in audit)),
            "cumulative_errors": int(sum(item["incremental_false_positive"] for item in audit)),
            "unclaimed_after": int(unclaimed.sum()),
        })
    frame = pd.DataFrame(audit)
    claimed = frame.incremental_assigned.sum()
    correct = frame.incremental_true_positive.sum()
    summary = {
        "rules": int(len(frame)),
        "eligible_rules": int(frame.eligible.sum()),
        "rules_with_new_assignments": int(frame.incremental_assigned.gt(0).sum()),
        "eligible_rules_with_zero_assignments": int((frame.eligible & frame.incremental_assigned.eq(0)).sum()),
        "claimed_rows": int(claimed),
        "unclaimed_rows": int(unclaimed.sum()),
        "correct_assignments": int(correct),
        "incorrect_assignments": int(claimed - correct),
        "accuracy_on_claimed_rows": float(correct / claimed) if claimed else 0.0,
        "last_seqid_to_claim_any_row": int(frame.loc[frame.incremental_assigned.gt(0), "SeqID"].max()) if claimed else None,
    }
    cutoffs = {}
    for cutoff in (10, 20, 50, summary["last_seqid_to_claim_any_row"] or 0):
        sub = frame.loc[frame.SeqID.le(cutoff)]
        assigned = int(sub.incremental_assigned.sum())
        correct = int(sub.incremental_true_positive.sum())
        cutoffs[str(cutoff)] = {
            "claimed": assigned, "correct": correct, "incorrect": assigned - correct,
            "remaining": int(len(y_true) - assigned),
            "accuracy_on_claimed": float(correct / assigned) if assigned else 0.0,
        }
    summary["cutoffs"] = cutoffs
    return frame, summary, output


def _compare_checkpoints_to_cascade(trust: pd.DataFrame, predictions: dict[str, np.ndarray],
                                    y_true: np.ndarray, cascade_prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    cascade_correct = cascade_prediction == y_true
    for row in trust.sort_values("SeqID", kind="stable").drop_duplicates("experiment_id").itertuples(index=False):
        prediction_correct = predictions[str(row.experiment_id)] == y_true
        corrected = int(np.sum(prediction_correct & ~cascade_correct))
        regressed = int(np.sum(~prediction_correct & cascade_correct))
        rows.append({
            "sequence_id": str(row.sequence_id), "experiment_id": str(row.experiment_id),
            "model": str(row.model), "training_dataset": str(row.training_dataset),
            "checkpoint_correct": int(np.sum(prediction_correct)),
            "checkpoint_accuracy": float(np.mean(prediction_correct)),
            "cascade_errors_corrected_if_substituted": corrected,
            "cascade_correct_rows_lost_if_substituted": regressed,
            "net_correct_count_change_if_substituted": corrected - regressed,
        })
    return pd.DataFrame(rows)


def _greedy_replay(trust: pd.DataFrame, predictions: dict[str, np.ndarray],
                   y_true: np.ndarray, *, objective: str, fallback_prediction: np.ndarray,
                   minimum_incremental_support: int = 10) -> tuple[pd.DataFrame, dict]:
    if objective not in {"precision", "net_gain"}:
        raise ValueError(f"unknown greedy objective: {objective}")
    candidates = [row for row in trust.itertuples(index=False) if _is_eligible_fallback(row)]
    unclaimed = np.ones(len(y_true), dtype=bool)
    used: set[tuple[str, str]] = set()
    trace = []
    while unclaimed.any():
        choices = []
        for row in candidates:
            key = (str(row.experiment_id), str(row.target_label))
            if key in used:
                continue
            indexes = np.flatnonzero(unclaimed & (predictions[key[0]] == key[1]))
            support = int(len(indexes))
            if support < minimum_incremental_support:
                continue
            tp = int(np.sum(y_true[indexes] == key[1]))
            fp = support - tp
            precision = tp / support
            net_gain = tp - fp
            if objective == "net_gain" and net_gain <= 0:
                continue
            common = (int(row.SeqID), key[1])
            if objective == "precision":
                sort_key = (-precision, -tp, -support, *common)
            else:
                sort_key = (-net_gain, -precision, -tp, *common)
            choices.append((sort_key, row, indexes, tp, fp, precision, net_gain))
        if not choices:
            break
        choices.sort(key=lambda choice: choice[0])
        _, row, indexes, tp, fp, precision, net_gain = choices[0]
        key = (str(row.experiment_id), str(row.target_label))
        unclaimed[indexes] = False
        used.add(key)
        trace.append({
            "selection_order": len(trace) + 1, "original_SeqID": int(row.SeqID),
            "sequence_id": str(row.sequence_id), "experiment_id": str(row.experiment_id),
            "model": str(row.model), "training_dataset": str(row.training_dataset),
            "label": str(row.target_label), "incremental_support": int(len(indexes)),
            "incremental_true_positive": tp, "incremental_false_positive": fp,
            "incremental_precision": precision, "net_TP_minus_FP": net_gain,
            "remaining_after": int(unclaimed.sum()),
        })

    prediction = np.asarray(fallback_prediction, dtype=object).copy()
    remaining = np.ones(len(y_true), dtype=bool)
    for row in trace:
        indexes = np.flatnonzero(remaining & (predictions[row["experiment_id"]] == row["label"]))
        prediction[indexes] = row["label"]
        remaining[indexes] = False
    fallback_rows = int(remaining.sum())
    fallback_correct = int(np.sum(prediction[remaining] == y_true[remaining]))
    # The loop above applies selected rules in their actual selection order.
    metrics = _metrics(y_true, prediction, sorted(set(y_true.tolist())))
    summary = {
        "objective": objective,
        "minimum_incremental_support": int(minimum_incremental_support),
        "selected_rules": int(len(trace)),
        "claimed_by_rules": int(len(y_true) - fallback_rows),
        "fallback_rows": fallback_rows,
        "fallback_correct": fallback_correct,
        "metrics_with_fallback": metrics,
        "selected_new_v2_rules": [row for row in trace if row["sequence_id"] in {"N0103", "N0104"}],
    }
    return pd.DataFrame(trace), summary


def _write_report(*, version: str, metadata: dict, fixed: pd.DataFrame,
                  fixed_summary: dict, greedy_summaries: dict,
                  checkpoint_comparison: pd.DataFrame,
                  fallback_sequence_id: str, fallback_accuracy: float,
                  best_sequence_id: str, best_accuracy: float, best_correct: int,
                  report_path: Path, validation_hash: str) -> None:
    difference = fixed_summary["correct_assignments"] - best_correct
    example_word = "example" if abs(difference) == 1 else "examples"
    examples = [
        (1, "N0086", "gpt-6-astra"),
        (3, "N0072", "gpt-6-astra"),
        (4, "N0097", "gemini-3.1-pro-preview"),
        (11, "N0096", "gemini-3.1-pro-preview"),
        (15, "N0104", "gemini-3.1-pro-preview"),
        (19, "N0081", "gemini-3.1-pro-preview"),
    ]
    example_rows = []
    for rank, seq, label in examples:
        matches = fixed.loc[fixed.SeqID.eq(rank)]
        if matches.empty:
            continue
        row = matches.iloc[0]
        if str(row.sequence_id) == seq and str(row.label) == label:
            example_rows.append(row)
    lines = [
        "# Trust-list v2 checkpoint complementarity",
        "",
        f"Snapshot: **{version}**, generated `{metadata.get('generated_at_utc', 'unknown')}`. "
        f"Validation: full V0, {metadata.get('validation_rows', len(fixed)):,} rows; SHA-256 `{validation_hash}`.",
        "",
        "## Fixed trust-list order",
        "",
        "Rules are replayed in the saved SeqID order. Each eligible checkpoint-label rule claims only still-unclaimed rows whose checkpoint top-1 prediction equals the rule label. Unsupported/ineligible rules are skipped. This is a same-V0 development diagnostic: V0 supplied the ranking metrics and was used for checkpoint selection.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Checkpoints / rules | {int(metadata.get('checkpoints', fixed.experiment_id.nunique()))} / {fixed_summary['rules']:,} |",
        f"| Eligible rules | {fixed_summary['eligible_rules']:,} |",
        f"| Rules claiming at least one row | {fixed_summary['rules_with_new_assignments']:,} |",
        f"| Eligible rules claiming zero rows | {fixed_summary['eligible_rules_with_zero_assignments']:,} |",
        f"| Rows claimed / left unclaimed | {fixed_summary['claimed_rows']:,} / {fixed_summary['unclaimed_rows']:,} |",
        f"| Correct / incorrect assignments | {fixed_summary['correct_assignments']:,} / {fixed_summary['incorrect_assignments']:,} |",
        f"| Accuracy | {fixed_summary['accuracy_on_claimed_rows']:.2%} |",
        f"| Last SeqID assigning any row | {fixed_summary['last_seqid_to_claim_any_row']} |",
        "",
        "| SeqID cutoff | Claimed | Correct | Incorrect | Remaining | Accuracy on claimed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cutoff in sorted({10, 20, 50, fixed_summary['last_seqid_to_claim_any_row'] or 0}):
        metrics = fixed_summary["cutoffs"][str(cutoff)]
        lines.append(f"| {cutoff} | {metrics['claimed']:,} | {metrics['correct']:,} | {metrics['incorrect']:,} | {metrics['remaining']:,} | {metrics['accuracy_on_claimed']:.2%} |")
    lines.extend([
        "",
        "### Global versus incremental precision examples",
        "",
        "| SeqID | Checkpoint | Label | Global precision / support | Newly claimed | Incremental TP / FP | Incremental precision |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for row in example_rows:
        lines.append(f"| {int(row.SeqID)} | {row.sequence_id} | {row.label} | {row.global_precision:.1%} / {int(row.global_prediction_count)} | {int(row.incremental_assigned)} | {int(row.incremental_true_positive)} / {int(row.incremental_false_positive)} | {row.incremental_precision:.1%} |")

    v2_specialists = checkpoint_comparison.loc[
        checkpoint_comparison.sequence_id.isin(["N0103", "N0104"])]
    lines.extend([
        "",
        "### Candidate checkpoints versus fixed-cascade errors",
        "",
        "This substitution diagnostic asks how many current cascade mistakes each checkpoint gets right, and how many current cascade-correct rows it would lose if used alone.",
        "",
        "| Checkpoint | Standalone accuracy | Current cascade errors it predicts correctly | Current correct rows it would lose | Net change if substituted |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in v2_specialists.itertuples(index=False):
        lines.append(f"| {row.sequence_id} | {row.checkpoint_accuracy:.2%} | {row.cascade_errors_corrected_if_substituted} | {row.cascade_correct_rows_lost_if_substituted} | {row.net_correct_count_change_if_substituted:+d} |")

    lines.extend([
        "",
        "## Greedy rule-selection diagnostics",
        "",
        f"Both methods use the full V0 labels to choose rules and evaluate them, require at least 10 newly claimed rows per selected rule, and use `{fallback_sequence_id}` as the fallback for rows left unclaimed. Residual-precision selection stops when no rule meets residual support; net-gain selection also requires positive TP − FP. Scores are exploratory and in-sample.",
        "",
        "| Greedy objective | Rules | Rule-claimed | Fallback rows correct | Correct total | Accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, title in (("precision", "Residual precision"), ("net_gain", "Residual TP − FP")):
        summary = greedy_summaries[key]
        metrics = summary["metrics_with_fallback"]
        lines.append(f"| {title} | {summary['selected_rules']} | {summary['claimed_by_rules']:,} | {summary['fallback_correct']:,} | {metrics['correct']:,}/{metrics['rows']:,} | {metrics['accuracy']:.2%} | {metrics['balanced_accuracy']:.2%} |")
    lines.extend([
        "",
        f"Best standalone checkpoint in this snapshot: `{best_sequence_id}`, accuracy {best_accuracy:.2%} ({best_correct}/{fixed_summary['claimed_rows']}). The fixed trust-list replay is {abs(difference)} {example_word} {'above' if difference >= 0 else 'below'} it on these same rows; this is not an independent estimate. The greedy results also compare with this checkpoint only because `{fallback_sequence_id}` was selected as their fallback.",
        "",
        "### New V2 checkpoints",
        "",
        "N0103 and N0104 are included using saved full-V0 prediction Parquets. Under the fixed order, N0104's highest-ranked Gemini 3.1 rule claims only two residual rows and both are wrong; N0103's first rule is after full coverage. In greedy diagnostics, N0104 contributes one 11-row rule per objective (7/11 Terra under precision selection; 6/11 Sol under net-gain selection); N0103 is not selected. These are ordering-specific observations, not an exhaustive estimate of their routing potential.",
        "",
        "## Interpretation and limits",
        "",
        "- The global precision is measured over all predictions, but the cascade sees only the subset left after earlier rules. These can have sharply different precision.",
        "- In the fixed order, 1,108 eligible rules add no rows. Most cascade coverage and correctness comes from a small number of early checkpoints; later checkpoints mostly overlap already-claimed predictions.",
        "- Greedy results are selected and scored on the same 2,699 V0 rows. The apparent gains over the fixed order are subject to substantial selection optimism and must not be treated as expected Kaggle accuracy.",
        "- No checkpoint was loaded, no inference was run, and the active trust list was not modified.",
        "",
        "Rule-by-rule audit and greedy selection traces are saved under `artifacts/analysis/trust_list_2_v0/`.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_v2_checkpoint_complementarity(*, trust_list: str | Path = DEFAULT_TRUST_LIST,
                                      trust_metadata: str | Path = DEFAULT_TRUST_METADATA,
                                      cache_dir: str | Path = DEFAULT_CACHE_DIR,
                                      artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
                                      data_dir: str | Path = DEFAULT_DATA_DIR,
                                      output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                                      report_path: str | Path = DEFAULT_REPORT,
                                      minimum_incremental_support: int = 10,
                                      fallback_sequence_id: str = "N0081") -> dict:
    """Analyze saved V0 prediction artifacts without changing the trust-list source."""
    trust_list, trust_metadata = Path(trust_list), Path(trust_metadata)
    cache_dir, artifact_dir, data_dir = Path(cache_dir), Path(artifact_dir), Path(data_dir)
    output_dir, report_path = Path(output_dir), Path(report_path)
    if minimum_incremental_support < 1:
        raise ValueError("minimum incremental support must be positive")
    metadata = json.loads(trust_metadata.read_text(encoding="utf-8"))
    trust = pd.read_csv(trust_list).sort_values("SeqID", kind="stable").reset_index(drop=True)
    if trust.empty or trust.SeqID.duplicated().any():
        raise ValueError("trust list must have unique, ordered SeqID rows")
    if str(trust.trust_list_version.iloc[0]) != str(metadata.get("version")):
        raise ValueError("trust CSV and metadata versions do not match")
    labels = sorted(trust.target_label.astype(str).unique().tolist())
    required_labels = {"id", "prompt_id", "text", "model"}
    v0_path = data_dir / "dataset_v0.parquet"
    v0 = pd.read_parquet(v0_path)
    if missing := required_labels - set(v0.columns):
        raise ValueError(f"V0 dataset missing {sorted(missing)}")
    v0 = v0[list(required_labels)].copy()
    v0["id"] = v0.id.astype(str)
    v0 = v0.sort_values("id", kind="stable").reset_index(drop=True)
    expected_hash = str(metadata.get("validation_sha256", ""))
    observed_hash = _sha256_v0(v0)
    if len(v0) != int(metadata.get("validation_rows", 2699)) or (expected_hash and observed_hash != expected_hash):
        raise ValueError("local V0 rows do not match the trust-list validation snapshot")
    if set(v0.model.astype(str).unique()) != set(labels):
        raise ValueError("V0 labels do not match trust-list labels")
    predictions, prediction_sources = _load_predictions(trust, v0=v0,
        cache_dir=cache_dir, artifact_dir=artifact_dir, labels=labels)
    y_true = v0.model.astype(str).to_numpy(dtype=object)
    fixed, fixed_summary, fixed_prediction = _replay_fixed_order(trust, predictions, y_true)
    if fixed_summary["unclaimed_rows"]:
        raise ValueError("fixed trust-list order left rows unclaimed; cannot compare the complete cascade")
    checkpoint_comparison = _compare_checkpoints_to_cascade(trust, predictions,
        y_true, fixed_prediction)
    best_checkpoint = checkpoint_comparison.sort_values(
        ["checkpoint_accuracy", "sequence_id"], ascending=[False, True], kind="stable").iloc[0]
    fallback_rows = trust.loc[trust.sequence_id.astype(str).eq(fallback_sequence_id)]
    if fallback_rows.empty:
        raise ValueError(f"fallback sequence ID not present in trust list: {fallback_sequence_id}")
    fallback_experiment = str(fallback_rows.experiment_id.iloc[0])
    fallback_prediction = predictions[fallback_experiment]
    fallback_accuracy = float(np.mean(fallback_prediction == y_true))
    greedy_traces, greedy_summaries = {}, {}
    for objective in ("precision", "net_gain"):
        trace, summary = _greedy_replay(trust, predictions, y_true, objective=objective,
            fallback_prediction=fallback_prediction,
            minimum_incremental_support=minimum_incremental_support)
        greedy_traces[objective], greedy_summaries[objective] = trace, summary

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_dir / "incremental_rule_audit.csv", index=False)
    checkpoint_comparison.to_csv(output_dir / "checkpoint_vs_fixed_cascade.csv", index=False)
    for objective, trace in greedy_traces.items():
        trace.to_csv(output_dir / f"greedy_{objective}_selection.csv", index=False)
    metrics = {
        "trust_list_version": str(metadata["version"]),
        "trust_list_generated_at_utc": metadata.get("generated_at_utc"),
        "validation_rows": int(len(v0)), "validation_sha256": observed_hash,
        "checkpoint_count": int(trust.experiment_id.nunique()),
        "rule_count": int(len(trust)),
        "minimum_incremental_support": int(minimum_incremental_support),
        "fallback_sequence_id": fallback_sequence_id,
        "fallback_experiment_id": fallback_experiment,
        "fallback_accuracy": fallback_accuracy,
        "best_standalone_sequence_id": str(best_checkpoint.sequence_id),
        "best_standalone_accuracy": float(best_checkpoint.checkpoint_accuracy),
        "checkpoint_vs_fixed_cascade": checkpoint_comparison.to_dict("records"),
        "fixed_order": fixed_summary,
        "greedy": greedy_summaries,
        "prediction_sources": prediction_sources,
        "selection_and_evaluation_scope": "same full V0; exploratory in-sample diagnostics only",
    }
    (output_dir / "complementarity_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_report(version=str(metadata["version"]), metadata=metadata,
        fixed=fixed, fixed_summary=fixed_summary, greedy_summaries=greedy_summaries,
        checkpoint_comparison=checkpoint_comparison,
        fallback_sequence_id=fallback_sequence_id, fallback_accuracy=fallback_accuracy,
        best_sequence_id=str(best_checkpoint.sequence_id),
        best_accuracy=float(best_checkpoint.checkpoint_accuracy),
        best_correct=int(best_checkpoint.checkpoint_correct),
        report_path=report_path, validation_hash=observed_hash)
    return {
        "status": "completed", "report": str(report_path), "output_dir": str(output_dir),
        "trust_list_version": str(metadata["version"]),
        "checkpoints": int(trust.experiment_id.nunique()), "rules": int(len(trust)),
        "fixed_order": fixed_summary,
        "greedy": {
            key: {"selected_rules": value["selected_rules"],
                  "claimed_by_rules": value["claimed_by_rules"],
                  "fallback_rows": value["fallback_rows"],
                  "correct": value["metrics_with_fallback"]["correct"],
                  "accuracy": value["metrics_with_fallback"]["accuracy"]}
            for key, value in greedy_summaries.items()
        },
    }

