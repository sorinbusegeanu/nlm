"""Per-label reliability-weighted voting over the current Trust List 2.

This is a separate, deterministic analysis path. It consumes only saved
full-V0 predictions and does not change any cascade or production ensemble.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .evaluation import evaluate
from .v2_checkpoint_complementarity import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    _is_eligible_fallback,
    _load_predictions,
    _load_single_prediction,
    _sha256_v0,
)
from .v2_checkpoint_trust import LABELS, latest_trust_paths


DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/trust_list_2_weighted_vote")
DEFAULT_REPORT = Path("docs/trust_list_2_weighted_vote.md")
DEFAULT_COMPARISON_REPORT = Path("docs/trust_list_2_voting_comparison.md")
DEFAULT_FAMILY_ABLATION_DIR = Path("artifacts/analysis/trust_list_2_family_ablation")
DEFAULT_FAMILY_ABLATION_REPORT = Path("docs/trust_list_2_family_ablation.md")
TRUST_LIST2_FAMILIES = (
    "catboost", "gemma", "granite", "llama", "mistral", "qwen", "smollm", "svc", "ultrabert",
)
N0122_EXPERIMENT_ID = (
    "ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_"
    "cfgb9e735ff_code81bce4a"
)
FAMILY_COUNTS = 30


def _model_family(model: str) -> str:
    """Map checkpoint variants to the architecture family used for normalization."""
    name = str(model).strip().lower().replace("-", "_")
    if name in {"embeddinggemma2", "gemma4_e2b", "gemma4_e4b"} or name.startswith("gemma"):
        return "gemma"
    if name.startswith("granite"):
        return "granite"
    if name.startswith("qwen"):
        return "qwen"
    if name.startswith("llama"):
        return "llama"
    if name.startswith("ministral") or name.startswith("mistral"):
        return "mistral"
    if name.startswith("ultrabert"):
        return "ultrabert"
    if name.startswith("catboost"):
        return "catboost"
    if name.startswith("svc"):
        return "svc"
    if name.startswith("smollm"):
        return "smollm"
    return name


def _load_v0(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(data_dir / "dataset_v0.parquet")
    needed = {"id", "prompt_id", "text", "model"}
    if missing := needed - set(frame.columns):
        raise ValueError(f"V0 dataset is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in needed:
        frame[column] = frame[column].astype(str)
    if frame.id.duplicated().any() or len(frame) != 2699:
        raise ValueError(f"expected 2,699 unique full-V0 rows, got {len(frame)}")
    if set(frame.model) != set(LABELS):
        raise ValueError("V0 labels do not match the canonical Trust List 2 classes")
    return frame.sort_values("id", kind="stable").reset_index(drop=True)


def _serialize_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items()}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".4f")


def _normalized_vote_scores(eligible: pd.DataFrame, checkpoints: pd.DataFrame,
                            family_by_exp: dict[str, str], family_checkpoint_counts: dict,
                            eligible_by_key: pd.DataFrame,
                            predictions: dict[str, np.ndarray], *, weighted: bool) -> np.ndarray:
    """Apply equal or reliability weights with the existing family normalization."""
    scores = np.zeros((len(next(iter(predictions.values()))), len(LABELS)), dtype=float)
    label_index = {label: index for index, label in enumerate(LABELS)}
    for experiment_id in checkpoints.experiment_id.astype(str):
        family = family_by_exp[experiment_id]
        family_count = int(family_checkpoint_counts[family])
        eligible_labels = set(eligible.loc[eligible.experiment_id.eq(experiment_id), "target_label"])
        for row_index, label in enumerate(predictions[experiment_id]):
            label = str(label)
            if label not in eligible_labels:
                continue
            if weighted:
                rule = eligible_by_key.loc[(experiment_id, label)]
                support = int(rule.prediction_count)
                raw_weight = float(rule.precision) * support / (support + FAMILY_COUNTS)
            else:
                raw_weight = 1.0
            scores[row_index, label_index[label]] += raw_weight / family_count
    return scores


def _label_prediction(scores: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS, dtype=object)[scores.argmax(axis=1)]


def _comparison_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    label_index = {label: index for index, label in enumerate(LABELS)}
    one_hot_scores = np.eye(len(LABELS), dtype=float)[
        np.asarray([label_index[label] for label in prediction], dtype=int)]
    metrics = evaluate(y_true, one_hot_scores, list(LABELS))
    metrics["accuracy"] = float(np.mean(prediction == y_true))
    metrics["correct"] = int(np.sum(prediction == y_true))
    return metrics


def run_trust_list2_weighted_vote(*, root: str | Path = ".",
                                  data_dir: str | Path = DEFAULT_DATA_DIR,
                                  cache_dir: str | Path = DEFAULT_CACHE_DIR,
                                  artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
                                  output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                                  report_path: str | Path = DEFAULT_REPORT) -> dict:
    root = Path(root).resolve()
    data_dir, cache_dir, artifact_dir = map(Path, (data_dir, cache_dir, artifact_dir))
    output_dir, report_path = Path(output_dir), Path(report_path)
    latest = latest_trust_paths(root)
    if latest is None:
        raise FileNotFoundError("docs/trust-list-2-latest.json is missing")
    trust_path, metadata_path, metadata = latest
    trust_version = str(metadata.get("version", ""))
    trust = pd.read_csv(trust_path)
    v0 = _load_v0(data_dir)
    validation_hash = _sha256_v0(v0)
    if int(metadata.get("validation_rows", -1)) != len(v0):
        raise ValueError("Trust List 2 validation row count does not match local V0")
    if str(metadata.get("validation_sha256", "")) != validation_hash:
        raise ValueError("Trust List 2 validation hash does not match local V0")
    if set(trust.target_label.astype(str)) - set(LABELS):
        raise ValueError("Trust List 2 contains labels outside canonical class order")

    eligible = trust.loc[trust.apply(_is_eligible_fallback, axis=1)].copy()
    eligible["experiment_id"] = eligible.experiment_id.astype(str)
    eligible["target_label"] = eligible.target_label.astype(str)
    eligible["prediction_count"] = pd.to_numeric(eligible.prediction_count, errors="raise").astype(int)
    eligible["precision"] = pd.to_numeric(eligible.precision, errors="raise").astype(float)
    if eligible.empty:
        raise ValueError(f"{trust_version} has no eligible checkpoint-label entries")
    if eligible.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError("Trust List 2 has duplicate checkpoint-label entries")

    # A candidate checkpoint is active if it owns at least one eligible rule.
    checkpoints = (eligible.sort_values("SeqID", kind="stable")
                   .drop_duplicates("experiment_id").copy())
    model_by_exp = checkpoints.set_index("experiment_id").model.astype(str).to_dict()
    family_by_exp = {exp: _model_family(model) for exp, model in model_by_exp.items()}
    family_checkpoint_counts = pd.Series(list(family_by_exp.values())).value_counts().to_dict()
    eligible_by_key = eligible.set_index(["experiment_id", "target_label"])

    predictions, prediction_sources = _load_predictions(
        eligible, v0=v0, cache_dir=cache_dir, artifact_dir=artifact_dir, labels=list(LABELS))
    y_true = v0.model.to_numpy(dtype=object)
    scores = np.zeros((len(v0), len(LABELS)), dtype=float)
    voter_count = np.zeros(len(v0), dtype=np.int64)
    label_index = {label: index for index, label in enumerate(LABELS)}
    contribution_rows = []
    per_checkpoint = []

    for experiment_id in checkpoints.experiment_id.astype(str):
        prediction = predictions[experiment_id]
        model = model_by_exp[experiment_id]
        family = family_by_exp[experiment_id]
        family_count = int(family_checkpoint_counts[family])
        eligible_labels = set(eligible.loc[eligible.experiment_id.eq(experiment_id), "target_label"])
        assigned_rule = np.zeros(len(v0), dtype=bool)
        for row_index, label in enumerate(prediction):
            rule_key = (experiment_id, str(label))
            if str(label) not in eligible_labels:
                contribution_rows.append({
                    "id": v0.id.iloc[row_index], "experiment_id": experiment_id,
                    "sequence_id": str(checkpoints.loc[checkpoints.experiment_id.eq(experiment_id), "sequence_id"].iloc[0]),
                    "model": model, "model_family": family, "predicted_label": str(label),
                    "eligible_rule": False, "precision": np.nan, "prediction_count": np.nan,
                    "raw_weight": 0.0, "family_normalized_weight": 0.0,
                })
                continue
            rule = eligible_by_key.loc[rule_key]
            precision = float(rule.precision)
            support = int(rule.prediction_count)
            raw_weight = precision * support / (support + FAMILY_COUNTS)
            normalized_weight = raw_weight / family_count
            scores[row_index, label_index[str(label)]] += normalized_weight
            assigned_rule[row_index] = True
            contribution_rows.append({
                "id": v0.id.iloc[row_index], "experiment_id": experiment_id,
                "sequence_id": str(rule.sequence_id), "model": model,
                "model_family": family, "predicted_label": str(label),
                "eligible_rule": True, "precision": precision,
                "prediction_count": support, "raw_weight": raw_weight,
                "family_normalized_weight": normalized_weight,
            })
        per_checkpoint.append({
            "experiment_id": experiment_id,
            "sequence_id": str(checkpoints.loc[checkpoints.experiment_id.eq(experiment_id), "sequence_id"].iloc[0]),
            "model": model, "model_family": family,
            "eligible_label_rules": int(len(eligible_labels)),
            "rows_with_eligible_vote": int(assigned_rule.sum()),
            "rows_without_eligible_vote": int((~assigned_rule).sum()),
            "prediction_file": prediction_sources[experiment_id],
        })
        voter_count += assigned_rule.astype(np.int64)

    row_totals = scores.sum(axis=1)
    zero_vote = row_totals <= 0
    prediction = np.asarray(LABELS, dtype=object)[scores.argmax(axis=1)]
    vote_shares = np.zeros_like(scores)
    nonzero = ~zero_vote
    vote_shares[nonzero] = scores[nonzero] / row_totals[nonzero, None]
    vote_shares[zero_vote] = 1.0 / len(LABELS)
    metrics = evaluate(y_true, scores, list(LABELS))
    metrics["accuracy"] = float(accuracy_score(y_true, prediction))
    metrics["correct"] = int(np.sum(prediction == y_true))
    metrics["rows_with_zero_eligible_votes"] = int(zero_vote.sum())

    # N0122 is the registry sequence 122; load its saved prediction directly.
    baseline_prediction, baseline_source = _load_single_prediction(
        N0122_EXPERIMENT_ID, v0=v0, cache_dir=cache_dir,
        artifact_dir=artifact_dir, labels=list(LABELS))
    baseline_scores = np.eye(len(LABELS))[np.asarray([label_index[x] for x in baseline_prediction])]
    baseline_metrics = evaluate(y_true, baseline_scores, list(LABELS))
    baseline_metrics["accuracy"] = float(accuracy_score(y_true, baseline_prediction))
    baseline_metrics["correct"] = int(np.sum(baseline_prediction == y_true))

    prediction_frame = pd.DataFrame({
        "id": v0.id, "prompt_id": v0.prompt_id, "y_true": y_true,
        "y_pred": prediction, "eligible_checkpoint_voters": voter_count,
        "zero_eligible_votes": zero_vote,
    })
    for index, label in enumerate(LABELS):
        prediction_frame[f"score::{label}"] = scores[:, index]
        prediction_frame[f"vote_share::{label}"] = vote_shares[:, index]
    baseline_frame = pd.DataFrame({
        "id": v0.id, "prompt_id": v0.prompt_id, "y_true": y_true,
        "y_pred": baseline_prediction,
    })
    per_label = pd.DataFrame({
        "label": list(LABELS),
        "precision": [metrics["per_class_precision"][label] for label in LABELS],
        "recall": [metrics["per_class_recall"][label] for label in LABELS],
        "correct": [int(np.sum((prediction == label) & (y_true == label))) for label in LABELS],
        "predicted": [int(np.sum(prediction == label)) for label in LABELS],
        "support": [int(np.sum(y_true == label)) for label in LABELS],
        "N0122_precision": [baseline_metrics["per_class_precision"][label] for label in LABELS],
        "N0122_recall": [baseline_metrics["per_class_recall"][label] for label in LABELS],
    })
    family_table = pd.DataFrame([
        {"model_family": family, "eligible_checkpoints": int(count)}
        for family, count in sorted(family_checkpoint_counts.items())
    ])
    vote_counts = pd.Series(voter_count).value_counts().sort_index()
    vote_count_table = pd.DataFrame({
        "eligible_checkpoint_votes": vote_counts.index.astype(int),
        "validation_rows": vote_counts.values.astype(int),
        "row_percent": vote_counts.values / len(v0),
    })
    contributions = pd.DataFrame(contribution_rows)
    checkpoint_table = pd.DataFrame(per_checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.parquet"
    baseline_path = output_dir / "n0122_predictions.parquet"
    contribution_path = output_dir / "checkpoint_contributions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    baseline_frame.to_parquet(baseline_path, index=False)
    contributions.to_parquet(contribution_path, index=False)
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    checkpoint_table.to_csv(output_dir / "checkpoint_contributions_summary.csv", index=False)
    family_table.to_csv(output_dir / "family_normalization.csv", index=False)
    vote_count_table.to_csv(output_dir / "vote_count_distribution.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_label")

    summary = {
        "algorithm": "Trust List 2 per-label reliability-weighted voting",
        "trust_list_version": trust_version,
        "trust_list_csv": str(trust_path),
        "trust_list_metadata": str(metadata_path),
        "trust_list_generated_at_utc": metadata.get("generated_at_utc"),
        "validation_scope": "full_v0", "validation_rows": int(len(v0)),
        "validation_sha256": validation_hash, "canonical_classes": list(LABELS),
        "eligible_checkpoint_label_rules": int(len(eligible)),
        "eligible_checkpoints": int(len(checkpoints)),
        "per_row_voter_count": {
            "minimum": int(voter_count.min()), "median": float(np.median(voter_count)),
            "mean": float(voter_count.mean()), "maximum": int(voter_count.max()),
            "distribution": {str(int(key)): int(value) for key, value in vote_counts.items()},
        },
        "family_checkpoint_counts": family_checkpoint_counts,
        "weight_formula": "precision * prediction_count / (prediction_count + 30)",
        "family_normalization": "divide each checkpoint vote by the number of active eligible checkpoints in its model family",
        "tie_break": "canonical Trust List 2 class order",
        "weighted_vote_metrics": _serialize_metrics(metrics),
        "n0122_experiment_id": N0122_EXPERIMENT_ID,
        "n0122_saved_prediction": baseline_source,
        "n0122_metrics": _serialize_metrics(baseline_metrics),
        "prediction_artifact": str(prediction_path),
        "n0122_prediction_artifact": str(baseline_path),
        "checkpoint_contribution_artifact": str(contribution_path),
        "development_set_caveat": "Trust List 2 precision values and the checkpoints were evaluated/selected using this same full V0 set; these are in-sample development diagnostics, not independent generalization estimates.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    confusion = pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS)
    lines = [
        "# Trust List 2 Per-Label Weighted Voting",
        "",
        f"Trust snapshot: `{trust_version}` (generated {metadata.get('generated_at_utc')}).",
        f"Validation: full V0, {len(v0):,} rows, hash `{validation_hash}`.",
        "",
        "## Overall comparison",
        "",
        "| Method | Accuracy | Balanced accuracy | Macro F1 | Correct / rows |",
        "|---|---:|---:|---:|---:|",
        f"| Weighted vote | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['correct']:,} / {len(v0):,} |",
        f"| N0122 UltraBERT standalone | {baseline_metrics['accuracy']:.4f} | {baseline_metrics['balanced_accuracy']:.4f} | {baseline_metrics['macro_f1']:.4f} | {baseline_metrics['correct']:,} / {len(v0):,} |",
        "",
        "## Vote roster and normalization",
        "",
        f"{len(checkpoints)} checkpoints supplied at least one eligible checkpoint-label vote ({len(eligible)} eligible label rules). A checkpoint votes once per response only when its top-1 predicted label has an eligible row in the trust list. The raw vote is `precision × prediction_count / (prediction_count + 30)`. Each raw vote is divided by the number of included checkpoints in its model family; this gives each family a mean contribution instead of letting a family gain weight by checkpoint count. Scores are summed by label; the normalized `vote_share::` columns in the prediction artifact are score shares, not calibrated probabilities. Zero-vote rows: {int(zero_vote.sum())}.",
        "",
        _markdown_table(family_table, ["model_family", "eligible_checkpoints"]),
        "",
        f"Per response, eligible checkpoint voter count ranged from {voter_count.min()} to {voter_count.max()} (median {np.median(voter_count):.0f}, mean {voter_count.mean():.2f}).",
        "",
        _markdown_table(vote_count_table, ["eligible_checkpoint_votes", "validation_rows", "row_percent"]),
        "",
        "## Per-label precision and recall",
        "",
        _markdown_table(per_label, ["label", "precision", "recall", "N0122_precision", "N0122_recall", "predicted", "support"]),
        "",
        "## Confusion matrix",
        "",
        "Rows are true labels; columns are predicted labels.",
        "",
        confusion.to_markdown(),
        "",
        "## Artifacts",
        "",
        f"- Weighted-vote full-V0 predictions and per-label scores: `{prediction_path}`",
        f"- N0122 baseline predictions: `{baseline_path}`",
        f"- Per-checkpoint, per-response contribution audit: `{contribution_path}`",
        f"- Per-response voter-count distribution: `{output_dir / 'vote_count_distribution.csv'}`",
        f"- Machine-readable metrics: `{output_dir / 'metrics.json'}`",
        f"- Confusion CSV: `{output_dir / 'confusion_matrix.csv'}`",
        "",
        "## Limitation",
        "",
        summary["development_set_caveat"],
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return summary


def run_trust_list2_voting_comparison(*, root: str | Path = ".",
                                      data_dir: str | Path = DEFAULT_DATA_DIR,
                                      cache_dir: str | Path = DEFAULT_CACHE_DIR,
                                      artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
                                      report_path: str | Path = DEFAULT_COMPARISON_REPORT) -> dict:
    """Compare equal-weight and existing precision-weighted voting on V0."""
    from sklearn.metrics import confusion_matrix

    root = Path(root).resolve()
    data_dir, cache_dir, artifact_dir = map(Path, (data_dir, cache_dir, artifact_dir))
    report_path = Path(report_path)
    latest = latest_trust_paths(root)
    if latest is None:
        raise FileNotFoundError("docs/trust-list-2-latest.json is missing")
    trust_path, metadata_path, metadata = latest
    trust_version = str(metadata.get("version", ""))
    if trust_version != "trust-list-2-v012":
        raise ValueError(f"expected Trust List 2 v012, found {trust_version}")
    trust = pd.read_csv(trust_path)
    v0 = _load_v0(data_dir)
    validation_hash = _sha256_v0(v0)
    if int(metadata.get("validation_rows", -1)) != len(v0):
        raise ValueError("Trust List 2 validation row count does not match local V0")
    if str(metadata.get("validation_sha256", "")) != validation_hash:
        raise ValueError("Trust List 2 validation hash does not match local V0")
    if set(trust.target_label.astype(str)) - set(LABELS):
        raise ValueError("Trust List 2 contains labels outside canonical class order")

    eligible = trust.loc[trust.apply(_is_eligible_fallback, axis=1)].copy()
    eligible["experiment_id"] = eligible.experiment_id.astype(str)
    eligible["target_label"] = eligible.target_label.astype(str)
    eligible["prediction_count"] = pd.to_numeric(eligible.prediction_count, errors="raise").astype(int)
    eligible["precision"] = pd.to_numeric(eligible.precision, errors="raise").astype(float)
    if eligible.empty:
        raise ValueError(f"{trust_version} has no eligible checkpoint-label entries")
    if eligible.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError("Trust List 2 has duplicate checkpoint-label entries")

    checkpoints = (eligible.sort_values("SeqID", kind="stable")
                   .drop_duplicates("experiment_id").copy())
    if len(checkpoints) != 73:
        raise ValueError(f"Trust List 2 v012 should contribute 73 eligible checkpoints, found {len(checkpoints)}")
    model_by_exp = checkpoints.set_index("experiment_id").model.astype(str).to_dict()
    family_by_exp = {experiment_id: _model_family(model)
                     for experiment_id, model in model_by_exp.items()}
    family_checkpoint_counts = pd.Series(list(family_by_exp.values())).value_counts().to_dict()
    eligible_by_key = eligible.set_index(["experiment_id", "target_label"])
    predictions, prediction_sources = _load_predictions(
        eligible, v0=v0, cache_dir=cache_dir, artifact_dir=artifact_dir, labels=list(LABELS))
    if len(predictions) != 73:
        raise ValueError(f"expected cached predictions for 73 eligible checkpoints, found {len(predictions)}")

    majority_scores = _normalized_vote_scores(
        eligible, checkpoints, family_by_exp, family_checkpoint_counts,
        eligible_by_key, predictions, weighted=False)
    weighted_scores = _normalized_vote_scores(
        eligible, checkpoints, family_by_exp, family_checkpoint_counts,
        eligible_by_key, predictions, weighted=True)
    y_true = v0.model.to_numpy(dtype=object)
    majority_prediction = _label_prediction(majority_scores)
    weighted_prediction = _label_prediction(weighted_scores)
    majority_metrics = _comparison_metrics(y_true, majority_prediction)
    weighted_metrics = _comparison_metrics(y_true, weighted_prediction)

    baseline_prediction, baseline_source = _load_single_prediction(
        N0122_EXPERIMENT_ID, v0=v0, cache_dir=cache_dir,
        artifact_dir=artifact_dir, labels=list(LABELS))
    baseline_metrics = _comparison_metrics(y_true, baseline_prediction)

    majority_correct = majority_prediction == y_true
    weighted_correct = weighted_prediction == y_true
    oracle_correct = majority_correct | weighted_correct
    disagreement = majority_prediction != weighted_prediction
    corrected = ~majority_correct & weighted_correct
    lost = majority_correct & ~weighted_correct
    pairwise = pd.crosstab(
        pd.Categorical(majority_prediction, categories=LABELS, ordered=True),
        pd.Categorical(weighted_prediction, categories=LABELS, ordered=True),
        rownames=["equal_weight_prediction"], colnames=["precision_weighted_prediction"],
        dropna=False).reindex(index=LABELS, columns=LABELS, fill_value=0)
    confusion = {
        "Equal-weight majority": confusion_matrix(y_true, majority_prediction, labels=LABELS),
        "Precision-weighted": confusion_matrix(y_true, weighted_prediction, labels=LABELS),
        "N0122 UltraBERT": confusion_matrix(y_true, baseline_prediction, labels=LABELS),
    }

    per_label = pd.DataFrame({
        "label": list(LABELS),
        "equal_precision": [majority_metrics["per_class_precision"][label] for label in LABELS],
        "equal_recall": [majority_metrics["per_class_recall"][label] for label in LABELS],
        "equal_correct": [int(np.sum(majority_correct & (y_true == label))) for label in LABELS],
        "weighted_precision": [weighted_metrics["per_class_precision"][label] for label in LABELS],
        "weighted_recall": [weighted_metrics["per_class_recall"][label] for label in LABELS],
        "weighted_correct": [int(np.sum(weighted_correct & (y_true == label))) for label in LABELS],
        "correct_delta_weighted_minus_equal": [
            int(np.sum(weighted_correct & (y_true == label)) - np.sum(majority_correct & (y_true == label)))
            for label in LABELS],
        "n0122_precision": [baseline_metrics["per_class_precision"][label] for label in LABELS],
        "n0122_recall": [baseline_metrics["per_class_recall"][label] for label in LABELS],
        "support": [int(np.sum(y_true == label)) for label in LABELS],
    })

    family_table = pd.DataFrame([
        {"model_family": family, "eligible_checkpoints": int(count)}
        for family, count in sorted(family_checkpoint_counts.items())])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    method_metrics = pd.DataFrame([
        {"method": "Equal-weight majority", "accuracy": majority_metrics["accuracy"],
         "balanced_accuracy": majority_metrics["balanced_accuracy"],
         "correct": majority_metrics["correct"]},
        {"method": "Precision-weighted", "accuracy": weighted_metrics["accuracy"],
         "balanced_accuracy": weighted_metrics["balanced_accuracy"],
         "correct": weighted_metrics["correct"]},
        {"method": "N0122 UltraBERT", "accuracy": baseline_metrics["accuracy"],
         "balanced_accuracy": baseline_metrics["balanced_accuracy"],
         "correct": baseline_metrics["correct"]},
    ])
    matrix_sections = []
    for name, matrix in confusion.items():
        matrix_sections.extend([f"### {name}", "", "Rows are true labels; columns are predicted labels.", "",
                                pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_markdown(), ""])
    summary = {
        "trust_list_version": trust_version,
        "trust_list_csv": str(trust_path),
        "validation_rows": len(v0), "validation_sha256": validation_hash,
        "eligible_checkpoints": len(checkpoints), "eligible_checkpoint_label_rules": len(eligible),
        "family_checkpoint_counts": family_checkpoint_counts,
        "prediction_artifacts": prediction_sources,
        "equal_weight_metrics": majority_metrics,
        "precision_weighted_metrics": weighted_metrics,
        "n0122_metrics": baseline_metrics,
        "pairwise_disagreements": int(disagreement.sum()),
        "pairwise_disagreement_rate": float(disagreement.mean()),
        "errors_corrected_by_weighting": int(corrected.sum()),
        "correct_predictions_lost_by_weighting": int(lost.sum()),
        "oracle_correct": int(oracle_correct.sum()),
        "oracle_accuracy": float(oracle_correct.mean()),
        "n0122_prediction_source": baseline_source,
        "weight_formula": "precision * prediction_count / (prediction_count + 30)",
        "family_normalization": "divide each checkpoint vote by active eligible checkpoints in its model family",
    }
    lines = [
        "# Trust List 2 Voting Comparison", "",
        f"Trust List 2 snapshot: `{trust_version}` ({len(checkpoints)} checkpoints, {len(eligible)} eligible checkpoint-label rules).",
        f"Evaluation: the same full V0 set for every method ({len(v0):,} rows; SHA256 `{validation_hash}`).",
        "Equal-weight majority gives each eligible checkpoint-label vote weight 1 before dividing by the active checkpoint count for its family. Precision-weighted voting uses the existing `precision × prediction_count / (prediction_count + 30)` weight and the same family normalization. Both use canonical label order to break ties.",
        "",
        "## Overall metrics", "",
        _markdown_table(method_metrics, ["method", "accuracy", "balanced_accuracy", "correct"]), "",
        "## Pairwise comparison", "",
        "| Measure | Count | Rate / accuracy |", "|---|---:|---:|",
        f"| Rows where methods disagree | {int(disagreement.sum()):,} | {disagreement.mean():.2%} |",
        f"| Errors corrected by switching to weighted | {int(corrected.sum()):,} | — |",
        f"| Majority correct predictions lost to weighted | {int(lost.sum()):,} | — |",
        f"| Correct under at least one voting method (oracle) | {int(oracle_correct.sum()):,} | {oracle_correct.mean():.4f} |",
        "",
        "## Per-label precision, recall, and correct-count changes", "",
        _markdown_table(per_label, ["label", "equal_precision", "equal_recall", "equal_correct",
                                    "weighted_precision", "weighted_recall", "weighted_correct",
                                    "correct_delta_weighted_minus_equal", "n0122_precision", "n0122_recall", "support"]),
        "",
        "## Pairwise disagreement matrix", "",
        "Rows are equal-weight predictions; columns are precision-weighted predictions.", "",
        pairwise.to_markdown(), "",
        "## Confusion matrices", "", *matrix_sections,
        "## Model-family normalization", "", _markdown_table(family_table, ["model_family", "eligible_checkpoints"]), "",
        "## Scope note", "",
        "This is an in-sample diagnostic: Trust List 2 precision values and checkpoint selection use the same full V0 evaluation set. It does not estimate independent generalization performance. No checkpoint was loaded and no inference or training was run.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def _weighted_roster_prediction(eligible: pd.DataFrame,
                                predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Weight the remaining active checkpoints with freshly computed family divisors."""
    checkpoints = (eligible.sort_values("SeqID", kind="stable")
                   .drop_duplicates("experiment_id").copy())
    model_by_exp = checkpoints.set_index("experiment_id").model.astype(str).to_dict()
    family_by_exp = {experiment_id: _model_family(model)
                     for experiment_id, model in model_by_exp.items()}
    family_checkpoint_counts = pd.Series(list(family_by_exp.values())).value_counts().to_dict()
    eligible_by_key = eligible.set_index(["experiment_id", "target_label"])
    scores = _normalized_vote_scores(
        eligible, checkpoints, family_by_exp, family_checkpoint_counts,
        eligible_by_key, predictions, weighted=True)
    return _label_prediction(scores), {
        "scores": scores,
        "checkpoints": checkpoints,
        "family_by_experiment": family_by_exp,
        "family_checkpoint_counts": family_checkpoint_counts,
    }


def run_trust_list2_family_ablation(*, root: str | Path = ".",
                                    data_dir: str | Path = DEFAULT_DATA_DIR,
                                    cache_dir: str | Path = DEFAULT_CACHE_DIR,
                                    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
                                    output_dir: str | Path = DEFAULT_FAMILY_ABLATION_DIR,
                                    report_path: str | Path = DEFAULT_FAMILY_ABLATION_REPORT) -> dict:
    """Run leave-one-family-out weighted voting using saved Trust List 2 V0 predictions."""
    root = Path(root).resolve()
    data_dir, cache_dir, artifact_dir, output_dir, report_path = [
        Path(value) if Path(value).is_absolute() else root / Path(value)
        for value in (data_dir, cache_dir, artifact_dir, output_dir, report_path)
    ]
    latest = latest_trust_paths(root)
    if latest is None:
        raise FileNotFoundError("docs/trust-list-2-latest.json is missing")
    trust_path, metadata_path, metadata = latest
    trust_version = str(metadata.get("version", ""))
    if trust_version != "trust-list-2-v012":
        raise ValueError(f"expected Trust List 2 v012, found {trust_version}")
    trust = pd.read_csv(trust_path)
    v0 = _load_v0(data_dir)
    validation_hash = _sha256_v0(v0)
    if int(metadata.get("validation_rows", -1)) != len(v0):
        raise ValueError("Trust List 2 validation row count does not match local V0")
    if str(metadata.get("validation_sha256", "")) != validation_hash:
        raise ValueError("Trust List 2 validation hash does not match local V0")
    if set(trust.target_label.astype(str)) - set(LABELS):
        raise ValueError("Trust List 2 contains labels outside canonical class order")

    eligible = trust.loc[trust.apply(_is_eligible_fallback, axis=1)].copy()
    eligible["experiment_id"] = eligible.experiment_id.astype(str)
    eligible["target_label"] = eligible.target_label.astype(str)
    eligible["prediction_count"] = pd.to_numeric(eligible.prediction_count, errors="raise").astype(int)
    eligible["precision"] = pd.to_numeric(eligible.precision, errors="raise").astype(float)
    if eligible.empty or eligible.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError("Trust List 2 has no eligible rules or duplicate checkpoint-label entries")
    full_checkpoints = eligible.sort_values("SeqID", kind="stable").drop_duplicates("experiment_id")
    if len(full_checkpoints) != 73:
        raise ValueError(f"Trust List 2 v012 should contain 73 active checkpoints, found {len(full_checkpoints)}")
    family_by_experiment = {
        str(row.experiment_id): _model_family(str(row.model))
        for row in full_checkpoints.itertuples(index=False)
    }
    actual_families = set(family_by_experiment.values())
    if actual_families != set(TRUST_LIST2_FAMILIES):
        raise ValueError(f"Trust List 2 families differ from expected roster: {sorted(actual_families)}")

    predictions, prediction_sources = _load_predictions(
        eligible, v0=v0, cache_dir=cache_dir, artifact_dir=artifact_dir, labels=list(LABELS))
    if len(predictions) != 73:
        raise ValueError(f"expected saved predictions for 73 checkpoints, found {len(predictions)}")
    y_true = v0.model.to_numpy(dtype=object)
    baseline_prediction, baseline_roster = _weighted_roster_prediction(eligible, predictions)
    baseline_metrics = _comparison_metrics(y_true, baseline_prediction)

    # Confirm this baseline reproduces the existing weighted-vote prediction artifact.
    existing_weighted_path = root / DEFAULT_OUTPUT_DIR / "predictions.parquet"
    if existing_weighted_path.is_file():
        existing = pd.read_parquet(existing_weighted_path)
        if not {"id", "y_pred"}.issubset(existing.columns):
            raise ValueError(f"existing weighted-vote artifact is incomplete: {existing_weighted_path}")
        aligned_existing = existing.set_index(existing.id.astype(str)).loc[v0.id.astype(str).tolist()]
        if not np.array_equal(aligned_existing.y_pred.astype(str).to_numpy(dtype=object), baseline_prediction):
            raise ValueError("recomputed baseline does not match the existing weighted-vote predictions")

    prediction_dir = output_dir / "predictions"
    confusion_dir = output_dir / "confusion_matrices"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_run = {}
    label_deltas = []

    def save_run(name: str, prediction: np.ndarray, metrics: dict, scores: np.ndarray) -> None:
        frame = pd.DataFrame({
            "id": v0.id.astype(str), "prompt_id": v0.prompt_id.astype(str),
            "y_true": y_true, "y_pred": prediction, "correct": prediction == y_true,
        })
        for index, label in enumerate(LABELS):
            frame[f"score::{label}"] = scores[:, index]
        frame.to_parquet(prediction_dir / f"{name}.parquet", index=False)
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS)
        matrix.to_csv(confusion_dir / f"{name}.csv", index_label="true_label")

    save_run("baseline_all_families", baseline_prediction, baseline_metrics,
             baseline_roster["scores"])
    metrics_by_run["all_families"] = {
        "accuracy": baseline_metrics["accuracy"],
        "balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "correct": baseline_metrics["correct"], "excluded_family": None,
        "active_checkpoints": len(baseline_roster["checkpoints"]),
        "family_checkpoint_counts": baseline_roster["family_checkpoint_counts"],
        "prediction_path": str((prediction_dir / "baseline_all_families.parquet").relative_to(root)),
        "confusion_matrix_path": str((confusion_dir / "baseline_all_families.csv").relative_to(root)),
    }

    ablation_rows = []
    for family in TRUST_LIST2_FAMILIES:
        remove_ids = {experiment_id for experiment_id, model_family in family_by_experiment.items()
                      if model_family == family}
        reduced = eligible.loc[~eligible.experiment_id.astype(str).isin(remove_ids)].copy()
        if not remove_ids or reduced.empty:
            raise ValueError(f"cannot exclude {family}: no remaining eligible checkpoints")
        prediction, roster_info = _weighted_roster_prediction(reduced, predictions)
        metrics = _comparison_metrics(y_true, prediction)
        errors_corrected = int(np.sum((baseline_prediction != y_true) & (prediction == y_true)))
        correct_lost = int(np.sum((baseline_prediction == y_true) & (prediction != y_true)))
        accuracy_delta = metrics["accuracy"] - baseline_metrics["accuracy"]
        balanced_delta = metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
        name = f"without_{family}"
        save_run(name, prediction, metrics, roster_info["scores"])
        row = {
            "family": family,
            "removed_checkpoints": int(len(remove_ids)),
            "remaining_checkpoints": int(len(roster_info["checkpoints"])),
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy_delta_pp": 100 * accuracy_delta,
            "balanced_accuracy_delta_pp": 100 * balanced_delta,
            "errors_corrected": errors_corrected, "correct_predictions_lost": correct_lost,
            "net_correct_delta": int(metrics["correct"] - baseline_metrics["correct"]),
            "correct": metrics["correct"],
            "prediction_path": str((prediction_dir / f"{name}.parquet").relative_to(root)),
            "confusion_matrix_path": str((confusion_dir / f"{name}.csv").relative_to(root)),
            "family_checkpoint_counts": roster_info["family_checkpoint_counts"],
        }
        metrics_by_run[family] = row
        ablation_rows.append(row)
        for label in LABELS:
            baseline_label_correct = int(np.sum((baseline_prediction == label) & (y_true == label)))
            ablated_label_correct = int(np.sum((prediction == label) & (y_true == label)))
            label_deltas.append({
                "family_removed": family, "label": label,
                "baseline_correct": baseline_label_correct,
                "without_family_correct": ablated_label_correct,
                "correct_delta": ablated_label_correct - baseline_label_correct,
            })

    ablation_metrics = pd.DataFrame(ablation_rows)
    ablation_metrics["contribution_rank"] = ablation_metrics["balanced_accuracy_delta_pp"].rank(
        ascending=True, method="min").astype(int)
    # Positive removal delta means the full ensemble was worse without that family.
    # Rank largest positive full-ensemble contribution first.
    ablation_metrics["measured_contribution_pp"] = -ablation_metrics["balanced_accuracy_delta_pp"]
    ablation_metrics = ablation_metrics.sort_values(
        ["measured_contribution_pp", "accuracy_delta_pp"], ascending=[False, False], kind="stable")
    ablation_metrics["contribution_rank"] = range(1, len(ablation_metrics) + 1)
    label_delta_frame = pd.DataFrame(label_deltas)
    label_delta_wide = label_delta_frame.pivot(index="label", columns="family_removed", values="correct_delta")
    label_delta_wide = label_delta_wide.reindex(index=LABELS, columns=TRUST_LIST2_FAMILIES).reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_metrics.to_csv(output_dir / "family_ablation_metrics.csv", index=False)
    label_delta_frame.to_csv(output_dir / "per_label_correct_changes.csv", index=False)
    label_delta_wide.to_csv(output_dir / "per_label_correct_changes_wide.csv", index=False)
    summary = {
        "trust_list_version": trust_version, "trust_list_csv": str(trust_path),
        "trust_list_metadata": str(metadata_path), "validation_rows": len(v0),
        "validation_sha256": validation_hash, "eligible_checkpoints": len(full_checkpoints),
        "eligible_checkpoint_label_rules": int(len(eligible)),
        "full_family_checkpoint_counts": baseline_roster["family_checkpoint_counts"],
        "prediction_sources": prediction_sources,
        "weight_formula": "precision * prediction_count / (prediction_count + 30)",
        "family_normalization": "divide by active eligible checkpoints in each remaining model family; recomputed after each exclusion",
        "baseline": metrics_by_run["all_families"], "ablations": metrics_by_run,
        "family_contribution_ranking": ablation_metrics[[
            "contribution_rank", "family", "measured_contribution_pp",
            "accuracy_delta_pp", "balanced_accuracy_delta_pp", "net_correct_delta"]].to_dict("records"),
        "artifacts_dir": str(output_dir.relative_to(root)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    lines = [
        "# Trust List 2 Leave-One-Family-Out Analysis", "",
        f"Trust snapshot `{trust_version}`: {len(full_checkpoints)} active checkpoints, {len(eligible)} eligible checkpoint-label rules, and {len(v0):,} full-V0 rows (SHA256 `{validation_hash}`).",
        "The baseline uses all nine families. Each ablation removes the named family, then recomputes the divisor from the eligible checkpoints remaining in each family. Per-vote weighting stays `precision × prediction_count / (prediction_count + 30)`, and ties follow canonical label order.",
        "",
        "## Baseline and family ablations", "",
        "Difference columns are ablation minus baseline in percentage points. `Measured contribution` is baseline balanced accuracy minus ablated balanced accuracy; positive values mean the full ensemble benefits from that family.",
        "",
        "| Rank | Removed family | Removed checkpoints | Accuracy | Balanced accuracy | Δ accuracy (pp) | Δ balanced accuracy (pp) | Errors corrected | Correct lost | Net correct Δ | Measured contribution (pp) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| — | **All families baseline** | 0 | {baseline_metrics['accuracy']:.4f} | {baseline_metrics['balanced_accuracy']:.4f} | 0.000 | 0.000 | — | — | 0 | — |",
    ]
    for row in ablation_metrics.to_dict("records"):
        lines.append(
            f"| {row['contribution_rank']} | {row['family']} | {row['removed_checkpoints']} | "
            f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['accuracy_delta_pp']:+.3f} | "
            f"{row['balanced_accuracy_delta_pp']:+.3f} | {row['errors_corrected']} | "
            f"{row['correct_predictions_lost']} | {row['net_correct_delta']:+d} | "
            f"{row['measured_contribution_pp']:+.3f} |"
        )
    lines += ["", "## Per-label changes in correct predictions", "",
              "Each value is the number correct after removing a family minus the baseline count.", "",
              _markdown_table(label_delta_wide, ["label", *TRUST_LIST2_FAMILIES]),
              "", "## Artifacts", "",
              f"- Summary and machine-readable metrics: `{output_dir / 'summary.json'}`",
              f"- Family metrics and contribution ranking: `{output_dir / 'family_ablation_metrics.csv'}`",
              f"- Per-label changes: `{output_dir / 'per_label_correct_changes.csv'}` and `{output_dir / 'per_label_correct_changes_wide.csv'}`",
              f"- Baseline and nine ablation predictions: `{prediction_dir}`",
              f"- Confusion matrices: `{confusion_dir}`", "",
              "## Scope", "",
              "This is a diagnostic on the same V0 set used to select Trust List 2 checkpoint rules, so it is not an independent estimate of generalization. The analysis reads saved full-V0 predictions only; it does not load models, train, or run inference.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


DEFAULT_SUBSET_OUTPUT_DIR = Path("artifacts/analysis/trust_list_2_checkpoint_subsets")
DEFAULT_SUBSET_REPORT = Path("docs/trust_list_2_checkpoint_subset_selection.md")
SUBSET_SIZES = (5, 6, 7, 8, 9, 10)
MIN_FOLD_RULE_SUPPORT = 10


def _sequence_sort_key(value: str) -> tuple[int, str]:
    import re
    match = re.fullmatch(r"N(\d+)", str(value))
    return (int(match.group(1)) if match else 10**12, str(value))


def _checkpoint_payload_size(root: Path, checkpoint_path: str) -> int | None:
    path = Path(str(checkpoint_path))
    if not path.is_absolute():
        path = root / path
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return None


def _audit_v0_training_overlap(root: Path, data_dir: Path,
                               checkpoints: pd.DataFrame, v0: pd.DataFrame) -> pd.DataFrame:
    """Cross-check training lineage against V0 IDs and normalized response text.

    Some older model runs used a training subset whose dataset file is no longer
    present. In that case, scan the complete source dataset when its manifest
    dataset hash identifies a locally available parent dataset. A zero overlap
    in the complete source is sufficient to establish that its subset is clean.
    """
    v0_ids = set(v0.id.astype(str))
    v0_texts = set(v0.text.astype(str).str.replace("\u00a0", " ", regex=False)
                    .str.replace(r"\s+", " ", regex=True).str.strip().str.casefold())
    manifest_cache: dict[str, dict] = {}
    dataset_sources: dict[str, str] = {}
    for experiment_id in checkpoints.experiment_id.astype(str):
        manifest_path = root / "artifacts" / "predictions" / f"{experiment_id}_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            manifest_cache[experiment_id] = manifest
            dataset_name = str(manifest.get("dataset") or "")
            dataset_path = data_dir / f"dataset_{dataset_name}.parquet"
            dataset_sha = str(manifest.get("dataset_sha256") or "")
            if dataset_sha and dataset_path.is_file():
                dataset_sources.setdefault(dataset_sha, dataset_name)

    dataset_cache: dict[str, dict] = {}
    rows = []
    for row in checkpoints.itertuples(index=False):
        experiment_id = str(row.experiment_id)
        manifest = manifest_cache.get(experiment_id, {})
        dataset = str(manifest.get("dataset") or row.training_dataset or "")
        dataset_path = data_dir / f"dataset_{dataset}.parquet"
        dataset_sha = str(manifest.get("dataset_sha256") or "")
        scanned_dataset = dataset
        if not dataset_path.is_file() and dataset_sha in dataset_sources:
            scanned_dataset = dataset_sources[dataset_sha]
            dataset_path = data_dir / f"dataset_{scanned_dataset}.parquet"
        overlap_ids = overlap_texts = None
        hash_match = None
        audit_status = "manifest or source dataset unavailable"
        if dataset_path.is_file():
            if scanned_dataset not in dataset_cache:
                try:
                    data = pd.read_parquet(dataset_path, columns=["id", "text", "split"])
                except (ValueError, KeyError):
                    data = pd.read_parquet(dataset_path, columns=["id", "text"])
                train = data.loc[data.split.astype(str).eq("train")].copy() if "split" in data else data.copy()
                ordered_ids = train.id.astype(str).tolist() if "id" in train else []
                train_hash = hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest() if ordered_ids else ""
                sorted_hash = hashlib.sha256("\n".join(sorted(ordered_ids)).encode()).hexdigest() if ordered_ids else ""
                text = (train.text.astype(str).str.replace("\u00a0", " ", regex=False)
                        .str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()) if "text" in train else None
                dataset_cache[scanned_dataset] = {
                    "train": train,
                    "ordered_hash": train_hash,
                    "sorted_hash": sorted_hash,
                    "overlap_ids": int(train.id.astype(str).isin(v0_ids).sum()) if "id" in train else None,
                    "overlap_texts": int(text.isin(v0_texts).sum()) if text is not None else None,
                }
            scanned = dataset_cache[scanned_dataset]
            declared_hash = str(manifest.get("train_id_sha256") or "")
            hash_match = bool(declared_hash and declared_hash in {scanned["ordered_hash"], scanned["sorted_hash"]})
            overlap_ids, overlap_texts = scanned["overlap_ids"], scanned["overlap_texts"]
            is_full_source_scan = scanned_dataset != dataset
            if overlap_ids == 0 and overlap_texts == 0:
                audit_status = (
                    f"source dataset {scanned_dataset} scanned; zero V0 ID/text overlap"
                    if is_full_source_scan else "training partition scanned; zero V0 ID/text overlap"
                )
            elif overlap_ids and overlap_texts:
                audit_status = "training partition contains exact V0 ID and normalized-text overlap"
            else:
                audit_status = "source scan found possible overlap; inspect split lineage"
        rows.append({
            "sequence_id": str(row.sequence_id), "experiment_id": experiment_id,
            "model": str(row.model), "training_dataset": dataset,
            "manifest_training_rows": manifest.get("train_rows"),
            "manifest_validation_rows": manifest.get("validation_rows"),
            "trust_list_reported_overlap": int(getattr(row, "validation_training_overlap", 0) or 0),
            "audited_source_dataset": scanned_dataset if dataset_path.is_file() else None,
            "local_training_rows": int(len(dataset_cache[scanned_dataset]["train"])) if dataset_path.is_file() else None,
            "v0_training_id_overlap": overlap_ids, "v0_training_normalized_text_overlap": overlap_texts,
            "manifest_train_id_hash_matches_local_partition": hash_match,
            "audit_status": audit_status,
        })
    return pd.DataFrame(rows)


def _fit_fold_weights(structural_rules: pd.DataFrame,
                       predictions: dict[str, np.ndarray], y_true: np.ndarray,
                       fit_indices: np.ndarray) -> dict[tuple[str, str], float]:
    """Estimate precision/support on training rows only for fold evaluation."""
    weights: dict[tuple[str, str], float] = {}
    y_fit = y_true[fit_indices]
    for rule in structural_rules.itertuples(index=False):
        experiment_id, label = str(rule.experiment_id), str(rule.target_label)
        predicted = predictions[experiment_id][fit_indices]
        mask = predicted == label
        support = int(mask.sum())
        if support < MIN_FOLD_RULE_SUPPORT:
            continue
        precision = float(np.mean(y_fit[mask] == label))
        weights[(experiment_id, label)] = precision * support / (support + FAMILY_COUNTS)
    return weights


def _checkpoint_score_matrices(checkpoint_ids: list[str], predictions: dict[str, np.ndarray],
                               weights: dict[tuple[str, str], float],
                               labels: tuple[str, ...] = LABELS) -> dict[str, np.ndarray]:
    label_index = {label: index for index, label in enumerate(labels)}
    rows = len(next(iter(predictions.values())))
    matrices = {experiment_id: np.zeros((rows, len(labels)), dtype=np.float64)
                for experiment_id in checkpoint_ids}
    for (experiment_id, label), weight in weights.items():
        if experiment_id not in matrices or weight <= 0:
            continue
        mask = predictions[experiment_id] == label
        matrices[experiment_id][mask, label_index[label]] = float(weight)
    return matrices


def _subset_scores(members: list[str], matrices: dict[str, np.ndarray],
                   family_by_exp: dict[str, str]) -> np.ndarray:
    if not members:
        raise ValueError("a checkpoint subset cannot be empty")
    family_counts: dict[str, int] = {}
    for experiment_id in members:
        family = family_by_exp[experiment_id]
        family_counts[family] = family_counts.get(family, 0) + 1
    result = np.zeros_like(matrices[members[0]])
    for experiment_id in members:
        result += matrices[experiment_id] / family_counts[family_by_exp[experiment_id]]
    return result


def _greedy_checkpoint_selection(candidates: list[str], matrices: dict[str, np.ndarray],
                                 family_by_exp: dict[str, str], y_true: np.ndarray,
                                 sequence_by_exp: dict[str, str], *, max_size: int = 10,
                                 evaluation_indices: np.ndarray | None = None
                                 ) -> tuple[list[str], pd.DataFrame]:
    """Greedily add the checkpoint maximizing balanced accuracy at each step."""
    ordered = sorted(candidates, key=lambda exp: (_sequence_sort_key(sequence_by_exp[exp]), exp))
    selected: list[str] = []
    rows = []
    previous_bacc = None
    while len(selected) < min(max_size, len(ordered)):
        choices = []
        for experiment_id in ordered:
            if experiment_id in selected:
                continue
            trial = [*selected, experiment_id]
            scores = _subset_scores(trial, matrices, family_by_exp)
            eval_scores = scores if evaluation_indices is None else scores[evaluation_indices]
            eval_truth = y_true if evaluation_indices is None else y_true[evaluation_indices]
            metrics = evaluate(eval_truth, eval_scores, list(LABELS))
            eval_prediction = np.asarray(LABELS, dtype=object)[eval_scores.argmax(axis=1)]
            choices.append((float(metrics["balanced_accuracy"]),
                            float(np.mean(eval_prediction == eval_truth)),
                            experiment_id, metrics))
        if not choices:
            break
        best_bacc = max(choice[0] for choice in choices)
        tied = [choice for choice in choices if np.isclose(choice[0], best_bacc, atol=1e-12, rtol=0)]
        chosen = min(tied, key=lambda choice: (_sequence_sort_key(sequence_by_exp[choice[2]]), choice[2]))
        bacc, accuracy, experiment_id, metrics = chosen
        selected.append(experiment_id)
        rows.append({
            "selection_order": len(selected), "sequence_id": sequence_by_exp[experiment_id],
            "experiment_id": experiment_id, "model_family": family_by_exp[experiment_id],
            "training_accuracy": accuracy, "training_balanced_accuracy": bacc,
            "marginal_balanced_accuracy": bacc - previous_bacc if previous_bacc is not None else None,
            "training_correct": int(metrics["confusion_matrix"] and
                                     np.trace(np.asarray(metrics["confusion_matrix"], dtype=np.int64))),
        })
        previous_bacc = bacc
    return selected, pd.DataFrame(rows)


def _selection_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    pred = _label_prediction(scores)
    metrics = evaluate(y_true, scores, list(LABELS))
    metrics["accuracy"] = float(accuracy_score(y_true, pred))
    metrics["correct"] = int(np.sum(pred == y_true))
    return metrics


def _selection_prediction_frame(v0: pd.DataFrame, y_true: np.ndarray,
                                scores: np.ndarray, *, fold_id: int | None = None) -> pd.DataFrame:
    prediction = _label_prediction(scores)
    row_sums = scores.sum(axis=1)
    share = np.zeros_like(scores)
    nonzero = row_sums > 0
    share[nonzero] = scores[nonzero] / row_sums[nonzero, None]
    share[~nonzero] = 1 / len(LABELS)
    frame = pd.DataFrame({
        "id": v0.id.astype(str).to_numpy(), "prompt_id": v0.prompt_id.astype(str).to_numpy(),
        "y_true": y_true, "y_pred": prediction, "correct": prediction == y_true,
    })
    if fold_id is not None:
        frame["fold_id"] = int(fold_id)
    for index, label in enumerate(LABELS):
        frame[f"score::{label}"] = scores[:, index]
        frame[f"vote_share::{label}"] = share[:, index]
    return frame


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Convert tabular selection traces to strict-JSON-compatible records."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def run_trust_list2_checkpoint_subsets(*, root: str | Path = ".",
                                       data_dir: str | Path = DEFAULT_DATA_DIR,
                                       cache_dir: str | Path = DEFAULT_CACHE_DIR,
                                       artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
                                       output_dir: str | Path = DEFAULT_SUBSET_OUTPUT_DIR,
                                       report_path: str | Path = DEFAULT_SUBSET_REPORT,
                                       n_splits: int = 5) -> dict:
    """Run full-V0 greedy subset search and prompt-group out-of-fold evaluation."""
    from sklearn.model_selection import GroupKFold

    root = Path(root).resolve()
    def rooted(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    data_dir, cache_dir, artifact_dir = map(rooted, (data_dir, cache_dir, artifact_dir))
    output_dir, report_path = rooted(output_dir), rooted(report_path)
    latest = latest_trust_paths(root)
    if latest is None:
        raise FileNotFoundError("docs/trust-list-2-latest.json is missing")
    trust_path, metadata_path, metadata = latest
    trust_version = str(metadata.get("version", ""))
    trust = pd.read_csv(trust_path)
    v0 = _load_v0(data_dir)
    validation_hash = _sha256_v0(v0)
    if int(metadata.get("validation_rows", -1)) != len(v0) or metadata.get("validation_sha256") != validation_hash:
        raise ValueError(f"Trust List 2 {trust_version} validation metadata does not match local full V0")

    # The full-V0 selection pool contains checkpoints with at least one eligible
    # rule. For CV, eligibility is re-established from fitting-fold support.
    full_eligible = trust.loc[trust.apply(_is_eligible_fallback, axis=1)].copy()
    full_eligible["experiment_id"] = full_eligible.experiment_id.astype(str)
    full_eligible["target_label"] = full_eligible.target_label.astype(str)
    full_eligible["precision"] = pd.to_numeric(full_eligible.precision, errors="raise").astype(float)
    full_eligible["prediction_count"] = pd.to_numeric(full_eligible.prediction_count, errors="raise").astype(int)
    if full_eligible.empty or full_eligible.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError(f"Trust List 2 {trust_version} eligible rules are empty or duplicated")
    checkpoints = (full_eligible.sort_values("SeqID", kind="stable")
                   .drop_duplicates("experiment_id").copy())
    checkpoint_count = int(len(checkpoints))
    candidates = checkpoints.experiment_id.astype(str).tolist()
    model_by_exp = checkpoints.set_index("experiment_id").model.astype(str).to_dict()
    family_by_exp = {experiment_id: _model_family(model)
                     for experiment_id, model in model_by_exp.items()}
    sequence_by_exp = checkpoints.set_index("experiment_id").sequence_id.astype(str).to_dict()
    payload_bytes = {str(row.experiment_id): _checkpoint_payload_size(root, row.checkpoint_path)
                     for row in checkpoints.itertuples(index=False)}
    if any(value is None for value in payload_bytes.values()):
        missing = [sequence_by_exp[exp] for exp, value in payload_bytes.items() if value is None]
        raise FileNotFoundError(f"checkpoint payload size could not be measured for {missing}")
    training_overlap_audit = _audit_v0_training_overlap(root, data_dir, checkpoints, v0)
    contaminated_mask = (
        pd.to_numeric(training_overlap_audit.v0_training_id_overlap, errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(training_overlap_audit.v0_training_normalized_text_overlap,
                        errors="coerce").fillna(0).gt(0)
    )
    contaminated = training_overlap_audit.loc[contaminated_mask].copy()
    verified_training_hashes = int(training_overlap_audit.manifest_train_id_hash_matches_local_partition.fillna(False).sum())
    unverified_training_rows = int(len(training_overlap_audit) - verified_training_hashes)

    predictions, prediction_sources = _load_predictions(
        full_eligible, v0=v0, cache_dir=cache_dir, artifact_dir=artifact_dir, labels=list(LABELS))
    if set(predictions) != set(candidates):
        raise ValueError(f"cached prediction inventory does not match the {checkpoint_count} Trust List 2 checkpoints")
    y_true = v0.model.to_numpy(dtype=object)
    full_weights = {
        (str(row.experiment_id), str(row.target_label)):
        float(row.precision) * int(row.prediction_count) /
        (int(row.prediction_count) + FAMILY_COUNTS)
        for row in full_eligible.itertuples(index=False)
    }
    full_matrices = _checkpoint_score_matrices(candidates, predictions, full_weights)
    sequence_order = {exp: sequence_by_exp[exp] for exp in candidates}
    full_selected, full_steps = _greedy_checkpoint_selection(
        candidates, full_matrices, family_by_exp, y_true, sequence_order, max_size=10)
    full_all_scores = _subset_scores(candidates, full_matrices, family_by_exp)
    full_all_metrics = _selection_metrics(y_true, full_all_scores)
    n0122_prediction, n0122_source = _load_single_prediction(
        N0122_EXPERIMENT_ID, v0=v0, cache_dir=cache_dir, artifact_dir=artifact_dir,
        labels=list(LABELS))
    n0122_scores = np.eye(len(LABELS), dtype=float)[
        np.asarray([{label: i for i, label in enumerate(LABELS)}[label]
                    for label in n0122_prediction], dtype=int)]
    n0122_metrics = _selection_metrics(y_true, n0122_scores)
    existing_path = root / DEFAULT_OUTPUT_DIR / "predictions.parquet"
    if trust_version == "trust-list-2-v012" and existing_path.is_file():
        previous = pd.read_parquet(existing_path)
        aligned = previous.set_index(previous.id.astype(str)).loc[v0.id.astype(str).tolist()]
        if not np.array_equal(aligned.y_pred.astype(str).to_numpy(dtype=object),
                              _label_prediction(full_all_scores)):
            raise ValueError("recomputed v012 baseline differs from saved weighted-vote predictions")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_pred_dir = output_dir / "full_v0_predictions"
    oof_pred_dir = output_dir / "oof_predictions"
    full_pred_dir.mkdir(parents=True, exist_ok=True)
    oof_pred_dir.mkdir(parents=True, exist_ok=True)
    full_rows = []
    full_selection = []
    selected_members_by_size = {}
    full_metrics_by_size = {}
    for size in SUBSET_SIZES:
        members = full_selected[:size]
        scores = _subset_scores(members, full_matrices, family_by_exp)
        metrics = _selection_metrics(y_true, scores)
        selected_members_by_size[str(size)] = [
            {"sequence_id": sequence_by_exp[exp], "experiment_id": exp,
             "model": model_by_exp[exp], "model_family": family_by_exp[exp],
             "checkpoint_path": str(checkpoints.loc[checkpoints.experiment_id.astype(str).eq(exp), "checkpoint_path"].iloc[0]),
             "checkpoint_size_bytes": int(payload_bytes[exp]),
             "checkpoint_size_mib": float(payload_bytes[exp] / 1024**2)}
            for exp in members
        ]
        selected_families = pd.Series([family_by_exp[exp] for exp in members]).value_counts().to_dict()
        model_payload = int(sum(int(payload_bytes[exp]) for exp in members))
        full_metrics_by_size[str(size)] = {
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "correct": metrics["correct"], "rows": int(len(v0)),
            "delta_accuracy_vs_full_roster_pp": (metrics["accuracy"] - full_all_metrics["accuracy"]) * 100,
            "delta_balanced_accuracy_vs_full_roster_pp": (metrics["balanced_accuracy"] - full_all_metrics["balanced_accuracy"]) * 100,
            "delta_accuracy_vs_n0122_pp": (metrics["accuracy"] - n0122_metrics["accuracy"]) * 100,
            "delta_balanced_accuracy_vs_n0122_pp": (metrics["balanced_accuracy"] - n0122_metrics["balanced_accuracy"]) * 100,
            "checkpoint_payload_bytes": model_payload,
            "checkpoint_payload_mib": model_payload / 1024**2,
            "selected_families": selected_families,
            "prediction_path": str((full_pred_dir / f"k{size:02d}.parquet").relative_to(root)),
        }
        frame = _selection_prediction_frame(v0, y_true, scores)
        frame.to_parquet(full_pred_dir / f"k{size:02d}.parquet", index=False)
        full_rows.append({"checkpoints": size, **full_metrics_by_size[str(size)]})
        for position, member in enumerate(selected_members_by_size[str(size)], start=1):
            full_selection.append({"subset_size": size, "order": position, **member})

    full_step_metrics = []
    for step in _json_records(full_steps):
        exp = step["experiment_id"]
        step["checkpoint_size_mib"] = float(payload_bytes[exp] / 1024**2)
        full_step_metrics.append(step)
    pd.DataFrame(full_step_metrics).to_csv(output_dir / "full_v0_greedy_steps.csv", index=False)
    pd.DataFrame(full_selection).drop_duplicates(["subset_size", "experiment_id"]).to_csv(
        output_dir / "full_v0_selected_checkpoints.csv", index=False)

    # Five-fold GroupKFold. The checkpoint roster is fixed from the selected
    # immutable list version, but each
    # fold fits rule support, precision, weights, and the greedy checkpoint
    # order using only its training prompts.
    structural_rules = trust.loc[
        trust.evaluation_eligible.astype(str).str.lower().isin({"true", "1", "yes"})
        & pd.to_numeric(trust.validation_training_overlap, errors="coerce").fillna(0).eq(0)
    ].copy()
    structural_rules["experiment_id"] = structural_rules.experiment_id.astype(str)
    structural_rules["target_label"] = structural_rules.target_label.astype(str)
    structural_rules = structural_rules.loc[structural_rules.experiment_id.isin(candidates)]
    if structural_rules.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError("structurally eligible checkpoint-label rows are duplicated")
    groups = v0.prompt_id.astype(str).to_numpy(dtype=object)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"need at least {n_splits} prompt groups for grouped CV")
    splitter = GroupKFold(n_splits=n_splits)
    oof_scores = {size: np.zeros((len(v0), len(LABELS)), dtype=np.float64) for size in SUBSET_SIZES}
    oof_predictions = {size: np.full(len(v0), None, dtype=object) for size in SUBSET_SIZES}
    oof_selected_by_fold = {}
    fold_summaries = []
    fold_id_by_row = np.zeros(len(v0), dtype=np.int64)
    full_roster_oof_scores = np.zeros((len(v0), len(LABELS)), dtype=np.float64)
    for fold_id, (fit_indices, test_indices) in enumerate(splitter.split(v0, y_true, groups), start=1):
        fold_id_by_row[test_indices] = fold_id
        if set(groups[fit_indices]) & set(groups[test_indices]):
            raise AssertionError("prompt group crossed a CV fold boundary")
        fold_weights = _fit_fold_weights(structural_rules, predictions, y_true, fit_indices)
        fold_matrices = _checkpoint_score_matrices(candidates, predictions, fold_weights)
        fold_selected, fold_steps = _greedy_checkpoint_selection(
            candidates, fold_matrices, family_by_exp, y_true,
            sequence_order, max_size=10, evaluation_indices=fit_indices)
        fold_selected_by_size = {}
        for size in SUBSET_SIZES:
            members = fold_selected[:size]
            scores_full = _subset_scores(members, fold_matrices, family_by_exp)
            scores_test = scores_full[test_indices]
            pred_test = _label_prediction(scores_test)
            oof_scores[size][test_indices] = scores_test
            oof_predictions[size][test_indices] = pred_test
            fold_selected_by_size[str(size)] = [
                {"sequence_id": sequence_by_exp[exp], "experiment_id": exp,
                 "model": model_by_exp[exp], "model_family": family_by_exp[exp],
                 "checkpoint_size_mib": float(payload_bytes[exp] / 1024**2)}
                for exp in members
            ]
        # Same fold-specific precision/support fit, using the full clean roster.
        scores_all_test = _subset_scores(candidates, fold_matrices, family_by_exp)[test_indices]
        full_roster_oof_scores[test_indices] = scores_all_test
        oof_selected_by_fold[str(fold_id)] = {
            "heldout_rows": int(len(test_indices)),
            "heldout_prompt_groups": sorted(set(groups[test_indices].tolist())),
            "fit_eligible_checkpoint_label_rules": int(len(fold_weights)),
            "greedy_selection_steps": _json_records(fold_steps),
            "selected_by_size": fold_selected_by_size,
        }
        fold_summaries.append({
            "fold_id": fold_id, "fit_rows": int(len(fit_indices)),
            "heldout_rows": int(len(test_indices)),
            "fit_prompt_groups": int(len(set(groups[fit_indices]))),
            "heldout_prompt_groups": int(len(set(groups[test_indices]))),
            "fit_eligible_checkpoint_label_rules": int(len(fold_weights)),
            "full_roster_checkpoint_ids": [sequence_by_exp[exp] for exp in candidates],
        })

    if any(np.any(pd.isna(oof_predictions[size])) for size in SUBSET_SIZES):
        raise ValueError("some examples lack a held-out subset prediction")
    full_roster_oof_metrics = _selection_metrics(y_true, full_roster_oof_scores)
    n0122_oof_metrics = n0122_metrics
    cv_metrics_by_size = {}
    cv_rows = []
    for size in SUBSET_SIZES:
        metrics = _selection_metrics(y_true, oof_scores[size])
        selected_occurrences = [
            exp for fold in oof_selected_by_fold.values()
            for exp in [item["experiment_id"] for item in fold["selected_by_size"][str(size)]]
        ]
        mean_payload = float(np.mean([sum(payload_bytes[exp] for exp in
                                      [item["experiment_id"] for item in fold["selected_by_size"][str(size)]])
                                      for fold in oof_selected_by_fold.values()]))
        cv_metrics_by_size[str(size)] = {
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "correct": metrics["correct"], "rows": int(len(v0)),
            "delta_accuracy_vs_foldfit_full_roster_pp": (metrics["accuracy"] - full_roster_oof_metrics["accuracy"]) * 100,
            "delta_balanced_accuracy_vs_foldfit_full_roster_pp": (metrics["balanced_accuracy"] - full_roster_oof_metrics["balanced_accuracy"]) * 100,
            "delta_accuracy_vs_n0122_pp": (metrics["accuracy"] - n0122_oof_metrics["accuracy"]) * 100,
            "delta_balanced_accuracy_vs_n0122_pp": (metrics["balanced_accuracy"] - n0122_oof_metrics["balanced_accuracy"]) * 100,
            "mean_selected_checkpoint_payload_mib": mean_payload / 1024**2,
            "unique_checkpoints_selected_across_folds": sorted(set(selected_occurrences),
                key=lambda exp: (_sequence_sort_key(sequence_by_exp[exp]), exp)),
            "prediction_path": str((oof_pred_dir / f"k{size:02d}_oof.parquet").relative_to(root)),
        }
        oof_frame = _selection_prediction_frame(v0, y_true, oof_scores[size])
        oof_frame["fold_id"] = fold_id_by_row
        oof_frame.to_parquet(oof_pred_dir / f"k{size:02d}_oof.parquet", index=False)
        cv_rows.append({"checkpoints": size, **cv_metrics_by_size[str(size)]})
    _selection_prediction_frame(v0, y_true, full_roster_oof_scores).assign(
        fold_id=fold_id_by_row).to_parquet(oof_pred_dir / "full_roster_oof.parquet", index=False)
    _selection_prediction_frame(v0, y_true, n0122_scores).assign(
        fold_id=fold_id_by_row).to_parquet(oof_pred_dir / "n0122_oof.parquet", index=False)
    (output_dir / "fold_selected_checkpoints.json").write_text(
        json.dumps(oof_selected_by_fold, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(full_rows).to_csv(output_dir / "full_v0_subset_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "prompt_group_oof_subset_metrics.csv", index=False)
    training_overlap_audit.to_csv(output_dir / "checkpoint_training_overlap_audit.csv", index=False)

    caveat = (
        "The full-V0 greedy scores are selection-set scores. OOF selection fits reliability statistics and subset order only on fold training prompts, but checkpoint roster membership and prior checkpoint/trust-list selection used full V0; OOF is therefore a diagnostic, not a fully independent estimate."
    )
    if not contaminated.empty:
        contaminated_ids = ", ".join(contaminated.sequence_id.astype(str).tolist())
        overlap_counts = sorted(set(pd.to_numeric(contaminated.v0_training_id_overlap, errors="coerce").dropna().astype(int)))
        v15_rows = contaminated.loc[contaminated.training_dataset.eq("v1.5")]
        v15_train_rows = sorted(set(pd.to_numeric(v15_rows.local_training_rows, errors="coerce").dropna().astype(int)))
        v15_validation_rows = sorted(set(pd.to_numeric(v15_rows.manifest_validation_rows, errors="coerce").dropna().astype(int)))
        caveat += (
            f" A local ID-and-text audit found {len(contaminated)} remaining checkpoints with V0 training overlap ({contaminated_ids}); "
            f"their rows are excluded by this run's trusted list only if the list itself has already been filtered."
        )
    else:
        caveat += (
            f" Training-data audit found no V0 ID or normalized-response overlap among the {checkpoint_count} active checkpoints. "
            f"It hash-verified {verified_training_hashes}/{len(training_overlap_audit)} recorded partitions; "
            f"{unverified_training_rows} runs lacked a directly verifiable local training partition, but their complete source datasets were scanned where locally available."
        )

    # Persist a complete selected prefix registry plus baseline references.
    selected_path = output_dir / "selected_checkpoints.json"
    result = {
        "algorithm": "greedy forward selection of reliability-weighted Trust List 2 checkpoints",
        "trust_list_version": trust_version, "trust_list_csv": str(trust_path.relative_to(root)),
        "trust_list_metadata": str(metadata_path.relative_to(root)),
        "validation_scope": "full_v0", "validation_rows": int(len(v0)),
        "validation_sha256": validation_hash, "prompt_groups": int(len(np.unique(groups))),
        "family_normalization": "divide each selected checkpoint's per-label weight by the number of selected checkpoints in its model family; recomputed for every subset",
        "weight_formula": "precision * prediction_count / (prediction_count + 30)",
        "full_v0_selection": {
            "selection_objective": "greedy maximum balanced accuracy; deterministic tie-break by ascending Trust List sequence ID",
            "greedy_steps": full_step_metrics,
            "selected_by_size": selected_members_by_size,
            "metrics_by_size": full_metrics_by_size,
            "full_roster_metrics": full_all_metrics,
            "n0122_metrics": n0122_metrics,
            "full_roster_prediction_sources": prediction_sources,
        },
        "prompt_group_oof": {
            "method": f"{n_splits}-fold sklearn GroupKFold, grouped by prompt_id",
            "selection_fit_scope": "each fold's training rows; per-rule precision/support and minimum-support eligibility recomputed on fit rows only",
            "folds": fold_summaries,
            "selected_by_fold": oof_selected_by_fold,
            "metrics_by_size": cv_metrics_by_size,
            "full_roster_foldfit_metrics": full_roster_oof_metrics,
            "n0122_metrics": n0122_oof_metrics,
        },
        "checkpoint_size_method": "sum of actual on-disk checkpoint_path file bytes; size is a load/storage proxy, not measured inference latency",
        "checkpoint_payload_bytes_by_experiment": payload_bytes,
        "checkpoint_training_overlap_audit": {
            "path": str((output_dir / "checkpoint_training_overlap_audit.csv").relative_to(root)),
            "checkpoints_with_v0_training_ids": int(len(contaminated)),
            "contaminated_sequence_ids": contaminated.sequence_id.astype(str).tolist(),
            "manifest_train_hashes_verified": verified_training_hashes,
            "normalized_text_audited_for_all_local_sources": True,
            "training_partitions_not_hash_verified": unverified_training_rows,
            "rows": _json_records(training_overlap_audit),
        },
        "caveat": caveat,
        "output_dir": str(output_dir.relative_to(root)), "report_path": str(report_path.relative_to(root)),
        "selected_checkpoints_path": str(selected_path.relative_to(root)),
    }
    selected_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    full_table = pd.DataFrame(full_rows)
    cv_table = pd.DataFrame(cv_rows)
    def id_list(size: int) -> str:
        return ", ".join(f"{item['sequence_id']} ({item['model_family']})"
                         for item in selected_members_by_size[str(size)])
    lines = [
        "# Trust List 2 Greedy Checkpoint Subset Selection",
        "",
        f"Trust List 2 {trust_version}, full V0 (2,699 rows), {checkpoint_count} checkpoints, {len(full_eligible):,} eligible rules; "
        f"validation hash `{validation_hash}`.",
        "",
        "Greedy selection starts from the best single checkpoint by balanced accuracy, then adds the candidate with the largest balanced-accuracy gain. Family divisors are recomputed for every tested subset. Ties go to the lower Trust List sequence ID. This procedure is greedy and does not guarantee a globally optimal subset.",
        "",
    ]
    removed_ids = metadata.get("removed_contaminated_checkpoint_ids", [])
    if removed_ids:
        lines += [
            "## Trust-list cleanup",
            "",
            f"This version was filtered from {metadata.get('filtered_from_version', 'an earlier version')} by removing {', '.join(map(str, removed_ids))}. The affected V1.5 training partition contained 2,159 V0 rows, matching both response IDs and normalized response text. The other checkpoints were checked against their complete local source datasets; no additional V0 ID or normalized-text overlap was found. The full source-version audit and removed rows are preserved with the archived prior results.",
            "",
        ]
    lines += [
        "## Full-V0 greedy selection",
        "",
        f"Reference full-roster ensemble: {full_all_metrics['correct']:,}/{len(v0):,} correct, accuracy {full_all_metrics['accuracy']:.4%}, balanced accuracy {full_all_metrics['balanced_accuracy']:.4%}. N0122 UltraBERT: {n0122_metrics['correct']:,}/{len(v0):,}, accuracy {n0122_metrics['accuracy']:.4%}, balanced accuracy {n0122_metrics['balanced_accuracy']:.4%}.",
        "",
        "| Checkpoints | Selected IDs (family) | Accuracy | Balanced accuracy | Correct | Δ Acc / BAcc vs full roster (pp) | Δ Acc / BAcc vs N0122 (pp) | Checkpoint payload (MiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full_rows:
        size = int(row["checkpoints"])
        met = full_metrics_by_size[str(size)]
        lines.append(
            f"| {size} | {id_list(size)} | {row['accuracy']:.4%} | {row['balanced_accuracy']:.4%} | {row['correct']:,} | "
            f"{met['delta_accuracy_vs_full_roster_pp']:+.3f} / {met['delta_balanced_accuracy_vs_full_roster_pp']:+.3f} | "
            f"{met['delta_accuracy_vs_n0122_pp']:+.3f} / {met['delta_balanced_accuracy_vs_n0122_pp']:+.3f} | {met['checkpoint_payload_mib']:,.1f} |"
        )
    lines += [
        "",
        "Accuracy deltas are also available in `full_v0_subset_metrics.csv`. The selected IDs and individual checkpoint payloads are in `full_v0_selected_checkpoints.csv`.",
        "",
        "## Training-overlap audit",
        "",
        f"{caveat}",
        "",
        f"Checkpoint-level audit: `{output_dir / 'checkpoint_training_overlap_audit.csv'}`.",
        "",
        "## Five-fold prompt-group OOF results",
        "",
        f"Each fold recomputed per-label precision/support and the minimum support rule using only its training prompts. Greedy selection used those fitting rows; its held-out predictions were pooled across the five folds. Fold-fit full-roster reference: accuracy {full_roster_oof_metrics['accuracy']:.4%}, balanced accuracy {full_roster_oof_metrics['balanced_accuracy']:.4%}. N0122 reference: accuracy {n0122_metrics['accuracy']:.4%}, balanced accuracy {n0122_metrics['balanced_accuracy']:.4%}.",
        "",
        "| Checkpoints | OOF accuracy | OOF balanced accuracy | Correct | Δ Acc / BAcc vs fold-fit full roster (pp) | Δ Acc / BAcc vs N0122 (pp) | Mean selected payload per fold (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cv_rows:
        size = int(row["checkpoints"])
        met = cv_metrics_by_size[str(size)]
        lines.append(
            f"| {size} | {row['accuracy']:.4%} | {row['balanced_accuracy']:.4%} | {row['correct']:,} | "
            f"{met['delta_accuracy_vs_foldfit_full_roster_pp']:+.3f} / {met['delta_balanced_accuracy_vs_foldfit_full_roster_pp']:+.3f} | "
            f"{met['delta_accuracy_vs_n0122_pp']:+.3f} / {met['delta_balanced_accuracy_vs_n0122_pp']:+.3f} | "
            f"{met['mean_selected_checkpoint_payload_mib']:,.1f} |"
        )
    lines += [
        "",
        "Fold-specific selected checkpoint IDs are saved in `fold_selected_checkpoints.json`; OOF predictions are in `oof_predictions/`.",
        "",
        "## Inference-cost proxy and limitations",
        "",
        "Checkpoint payload MiB is the sum of actual files under each selected Trust List checkpoint path. It estimates model payload and loading/storage burden; it is not a measured latency or throughput estimate. Repeated checkpoints from one architecture are counted separately because they are separate saved checkpoints.",
        "",
        result["caveat"],
        "",
        "## Artifacts",
        "",
        f"- Selected lists and complete metrics: `{selected_path}`",
        f"- Full-V0 predictions: `{full_pred_dir}`",
        f"- OOF predictions: `{oof_pred_dir}`",
        f"- Full-V0 metrics: `{output_dir / 'full_v0_subset_metrics.csv'}`",
        f"- OOF metrics: `{output_dir / 'prompt_group_oof_subset_metrics.csv'}`",
        f"- Greedy selection trace: `{output_dir / 'full_v0_greedy_steps.csv'}`",
        f"- Manifest/data training-overlap audit: `{output_dir / 'checkpoint_training_overlap_audit.csv'}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    result["report_path"] = str(report_path.relative_to(root))
    return result
