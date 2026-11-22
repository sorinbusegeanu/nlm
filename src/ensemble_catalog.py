"""Resolve and evaluate catalogued ensembles using only saved V0 probabilities."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from . import ensemble_cascade_comparison as cascades
from .v2_checkpoint_trust import LABELS, _read_saved_probabilities

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/ensemble_catalog_v2.json"
LABEL_ARRAY = np.asarray(LABELS, dtype=object)


def load_catalog():
    return json.loads(CATALOG.read_text())


def _enabled(series):
    return series.str.lower().isin({"true", "1"})


def _load_inputs(root, v0, *, version=None):
    if version is None:
        pointer = json.loads((root / "docs/trust-list-2-latest.json").read_text())
    else:
        # Reproducible analyses may pin a historical trust-list snapshot. The
        # mutable latest pointer must not invalidate an explicitly pinned run.
        pointer = {"version": str(version),
                   "csv_path": f"docs/{version}.csv",
                   "metadata_path": f"docs/{version}.json"}
    path = root / pointer["csv_path"]
    trust_hash = cascades._sha256(path)
    metadata = json.loads((root / pointer["metadata_path"]).read_text())
    trust = pd.read_csv(path, dtype=str, keep_default_na=False)
    if (metadata["version"] != pointer["version"] or metadata["validation_rows"] != len(v0)
            or set(trust.trust_list_version) != {pointer["version"]}
            or cascades._sha256(path) != trust_hash):
        raise ValueError("Trust List 2 snapshot and pointer disagree")
    for name in ("validation_rows", "overall_accuracy", "balanced_accuracy", "precision",
                 "recall", "prediction_count", "SeqID", "validation_training_overlap"):
        trust[name] = pd.to_numeric(trust[name], errors="coerce")
    trust = trust.loc[_enabled(trust.evaluation_eligible) & _enabled(trust.checkpoint_available)
                      & trust.validation_rows.eq(len(v0)) & trust.validation_training_overlap.eq(0)]
    members, probabilities, excluded = {}, {}, {}
    for sid, group in trust.groupby("sequence_id", sort=True):
        row = group.iloc[0]
        if any(group[key].nunique() != 1 for key in
               ("experiment_id", "model", "training_dataset", "checkpoint_path")):
            raise ValueError(f"Ambiguous identity for {sid}")
        if set(group.target_label) != set(LABELS) or group.target_label.duplicated().any():
            excluded[sid] = "incomplete or duplicate label metrics"
            continue
        if not row.checkpoint_path or not (root / row.checkpoint_path).exists():
            excluded[sid] = "checkpoint unavailable on disk"
            continue
        try:
            matrix, cache = _read_saved_probabilities(root, row.experiment_id,
                                                     v0.id.tolist(), v0.model.tolist())
        except (FileNotFoundError, ValueError) as exc:
            excluded[sid] = str(exc)
            continue
        if not np.isfinite([row.overall_accuracy, row.balanced_accuracy]).all():
            excluded[sid] = "missing overall metrics"
            continue
        probabilities[sid] = matrix
        members[sid] = {
            "experiment_id": row.experiment_id,
            "model": row.model,
            "training_dataset": row.training_dataset,
            "checkpoint_path": row.checkpoint_path,
            "balanced_accuracy": float(row.balanced_accuracy),
            "accuracy": float(row.overall_accuracy),
            "cache": str(cache),
            "cache_sha256": cascades._sha256(cache),
        }
    return trust[trust.sequence_id.isin(members)], members, probabilities, excluded, {
        "version": pointer["version"], "csv_path": str(path), "sha256": trust_hash,
    }


def _fit_limited_precision_roster(probabilities, truth, *, max_checkpoints, min_support):
    """Select one shared roster using only the fitting partition."""
    label_count = len(LABELS)
    actual_support = np.bincount(truth, minlength=label_count)
    if not actual_support.all():
        raise ValueError("The fitting partition must contain all canonical labels")
    top = probabilities.argmax(axis=2)
    predicted_support = np.stack([np.bincount(row, minlength=label_count) for row in top])
    correct = top == truth
    true_positives = np.stack([
        np.bincount(row[mask], minlength=label_count) for row, mask in zip(top, correct)
    ])
    precision = np.divide(true_positives, predicted_support,
                          out=np.zeros_like(true_positives, dtype=float),
                          where=predicted_support >= min_support)
    row_weight = 1.0 / (label_count * actual_support[truth])
    standalone_ba = (true_positives / actual_support).mean(axis=1)
    accuracy = true_positives.sum(axis=1)
    ranked = sorted(range(len(top)), key=lambda i: (-standalone_ba[i], -accuracy[i], i))

    weighted = probabilities * precision[:, None, :]
    roster = ranked[:1]
    numerator = weighted[roster[0]].copy()
    denominator = precision[roster[0]].copy()

    def score(candidate_numerator, candidate_denominator):
        if not np.all(candidate_denominator > 0):
            return None
        predicted = (candidate_numerator / candidate_denominator).argmax(axis=1)
        return float((predicted == truth) @ row_weight)

    current = score(numerator, denominator)
    steps = [{"size": 1, "fit_balanced_accuracy": current}]
    while len(roster) < min(max_checkpoints, len(ranked)):
        remaining = [index for index in ranked if index not in roster]
        candidate_scores = [
            score(numerator + weighted[index], denominator + precision[index])
            for index in remaining
        ]
        if current is None and all(value is None for value in candidate_scores):
            # Establish coverage before comparing complete blends.
            chosen = max(remaining, key=lambda i: np.count_nonzero((denominator == 0) & (precision[i] > 0)))
            if not np.any((denominator == 0) & (precision[chosen] > 0)):
                break
        else:
            valid = [(index, value) for index, value in zip(remaining, candidate_scores)
                     if value is not None]
            chosen, best = max(valid, key=lambda item: item[1])
            if current is not None and best <= current + 1e-12:
                break
        roster.append(chosen)
        numerator += weighted[chosen]
        denominator += precision[chosen]
        current = score(numerator, denominator)
        steps.append({"size": len(roster), "fit_balanced_accuracy": current})

    if current is None:
        raise ValueError(f"No positive weight for every label within {max_checkpoints} checkpoints")
    return roster, precision[roster], steps


def _limited_precision_blend(strategy, v0, probabilities):
    ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    if not ids:
        raise ValueError("No eligible cached checkpoints")
    matrices = np.stack([probabilities[sid] for sid in ids])
    truth = pd.Categorical(v0.model, categories=LABELS).codes
    settings = {key: strategy[key] for key in ("max_checkpoints", "min_support")}

    def fit_roster(rows):
        roster, weights, steps = _fit_limited_precision_roster(matrices[:, rows], truth[rows], **settings)
        checkpoint_ids = [ids[index] for index in roster]
        records = {
            "checkpoint_ids": checkpoint_ids,
            "weights": {sid: dict(zip(LABELS, row.tolist())) for sid, row in zip(checkpoint_ids, weights)},
            "selection_steps": steps,
        }
        return roster, weights, records

    predictions = np.empty(len(v0), dtype=object)
    folds = []
    for fitting, held in GroupKFold(n_splits=strategy["folds"]).split(v0, truth, v0.prompt_id):
        roster, weights, record = fit_roster(fitting)
        scores = np.einsum("mnl,ml->nl", matrices[roster][:, held], weights) / weights.sum(axis=0)
        predictions[held] = LABEL_ARRAY[scores.argmax(axis=1)]
        folds.append({**record, "held_ids": v0.id.iloc[held].tolist()})

    _, _, deployment = fit_roster(np.arange(len(v0)))
    return predictions, deployment, folds


def _evaluate_strategy(strategy, catalog, v0, members, probabilities):
    if strategy["algorithm"] == "checkpoint_reliability_router":
        from .e13_reliability_router import evaluate_catalog_strategy

        pred, details = evaluate_catalog_strategy(strategy, catalog["folds"], v0,
                                                 probabilities, members)
        roster = sorted({sid for fold in details["folds"]
                         for sid in fold["checkpoint_ids"]}, key=lambda sid: int(sid[1:]))
        details["checkpoint_ids"] = roster
    elif strategy["algorithm"] == "pairwise_arbitration":
        from .e12 import evaluate_catalog_strategy

        pred, details = evaluate_catalog_strategy(strategy, catalog["folds"], v0, probabilities, members)
        roster = details["deployment"]["checkpoint_ids"]
    elif strategy["algorithm"] == "confusion_evidence":
        from .confusion_ensemble import evaluate_confusion_ensemble

        pred, details = evaluate_confusion_ensemble(strategy, catalog["folds"], v0, probabilities)
        roster = details["deployment"]["checkpoint_ids"]
    elif strategy["algorithm"] == "limited_label_precision_mean":
        pred, deployment, folds = _limited_precision_blend(strategy, v0, probabilities)
        roster = deployment["checkpoint_ids"]
        details = {
            "deployment": deployment, "folds": folds,
            "evaluation_checkpoint_ids": sorted(
                {sid for fold in folds for sid in fold["checkpoint_ids"]}, key=lambda sid: int(sid[1:])),
        }
    else:
        from .ensemble_catalog_rules import evaluate_rule

        pred, folds, deployment = evaluate_rule(strategy, catalog["folds"], v0, members, probabilities)
        roster = deployment["checkpoint_ids"]
        details = {"folds": folds, "deployment": deployment,
                   "evaluation_checkpoint_ids": sorted(
                       {sid for fold in folds for sid in fold["checkpoint_ids"]},
                       key=lambda sid: int(sid[1:]))}
    return pred, {
        "accuracy": float(accuracy_score(v0.model, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(v0.model, pred)),
        "checkpoint_ids": sorted(roster, key=lambda sid: int(sid[1:])),
        **details,
    }


def evaluate(catalog, strategies, root, output):
    v0 = cascades._v0(root)
    _, members, probabilities, excluded, snapshot = _load_inputs(root, v0)
    results = []
    predictions = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for strategy in strategies:
        code = strategy["id"]
        result = {"ensemble": code, "strategy": strategy}
        try:
            predicted, metrics = _evaluate_strategy(strategy, catalog, v0, members, probabilities)
            result.update(metrics, status="ok")
            predictions[code] = predicted
            if strategy["algorithm"] == "confusion_evidence":
                from .confusion_ensemble import predict_confusion_ensemble

                predictions[f"deployment::{code}"] = predict_confusion_ensemble(
                    result["deployment"], probabilities)
        except ValueError as exc:
            result.update(status="unavailable", reason=str(exc))
        results.append(result)
    if {"E08", "E11"}.issubset(predictions.columns):
        reference_correct = predictions.E08.eq(predictions.y_true)
        e11_correct = predictions.E11.eq(predictions.y_true)
        e11 = next(result for result in results if result["ensemble"] == "E11")
        e11["comparison_vs_E08"] = {
            "corrected": int((~reference_correct & e11_correct).sum()),
            "lost": int((reference_correct & ~e11_correct).sum()),
            "net_correct": int(e11_correct.sum() - reference_correct.sum()),
            "accuracy_delta_pp": float((e11_correct.mean() - reference_correct.mean()) * 100),
        }
    if "E08" in predictions:
        reference_correct = predictions.E08.eq(predictions.y_true)
        for result in results:
            code = result["ensemble"]
            if result.get("status") == "ok" and code.startswith("E12-"):
                current_correct = predictions[code].eq(predictions.y_true)
                result["comparison_vs_E08"] = {
                    "corrected": int((~reference_correct & current_correct).sum()),
                    "lost": int((reference_correct & ~current_correct).sum()),
                }
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "results.json").write_text(json.dumps({
        "catalog_version": catalog["version"], "catalog_sha256": cascades._sha256(CATALOG),
        "trust_list": snapshot, "validation_rows": len(v0),
        "validation_sha256": cascades._sha256(root / "data/processed/dataset_v0.parquet"),
        "labels": list(LABELS), "checkpoints": members, "excluded_checkpoints": excluded,
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print("| Code | Ensemble | Trust List version | Checkpoints used | Accuracy | Balanced accuracy |")
    print("|---|---|---|---:|---:|---:|")
    for result in results:
        code = result["ensemble"]
        name = result["strategy"]["name"]
        version = snapshot["version"]
        if result["status"] == "ok":
            print(f"| {code} | {name} | {version} | {len(result['checkpoint_ids'])} | "
                  f"{result['accuracy']:.4%} | {result['balanced_accuracy']:.4%} |")
        else:
            print(f"| {code} | {name} (unavailable) | {version} | — | — | — |")
    print(f"Saved results and resolved rosters to {output}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=str.upper, help="Evaluate one catalog ID")
    parser.add_argument("--list", action="store_true", help="List catalog IDs without evaluating")
    parser.add_argument("--output", type=Path, help="New directory for predictions and results")
    args = parser.parse_args()
    catalog = load_catalog()
    strategies = {item["id"]: item for item in catalog["strategies"]}
    if args.strategy is not None and args.strategy not in strategies:
        parser.error(f"unknown strategy {args.strategy}. Use --list to see available IDs")
    if args.list:
        for item in strategies.values():
            reason = f" ({item['reason']})" if "reason" in item else ""
            print(f"{item['id']}  {item['name']}{reason}")
        return 0
    selected = [strategies[args.strategy]] if args.strategy else list(strategies.values())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts/analysis/ensemble_catalog" / stamp
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    results = evaluate(catalog, selected, ROOT, output)
    return 0 if any(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
