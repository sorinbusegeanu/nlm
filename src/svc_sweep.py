"""Run and report the fixed V1.1 SVC C-value comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from .ensemble_registry import read_ensemble_registry
from .pipeline import train_models
from .prediction_factory import _align_candidates, _resolve_artifact_path


DEFAULT_C_VALUES = (0.1, 2.0, 5.0)


def _member_id(member):
    return str(member.get("candidate_id") if isinstance(member, dict) else member)


def _probability_metrics(probabilities, y_true, classes):
    predicted = np.asarray(classes, dtype=object)[np.asarray(probabilities).argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(classes))),
    }, predicted


def _read_selected_candidates(prediction_dir, candidate_ids):
    """Load only requested predictions instead of scanning every saved model run."""
    candidates = []
    for candidate_id in candidate_ids:
        manifest_path = Path(prediction_dir) / f"{candidate_id}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"candidate manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        prediction_path = manifest.get("prediction_path")
        if not prediction_path:
            raise ValueError(f"candidate {candidate_id} has no validation prediction path")
        prediction_path = _resolve_artifact_path(prediction_path, Path.cwd())
        if not prediction_path.is_file():
            raise FileNotFoundError(f"candidate predictions are missing: {prediction_path}")
        frame = pd.read_parquet(prediction_path)
        columns = sorted(column for column in frame if column.startswith("prob::"))
        classes = [column.removeprefix("prob::") for column in columns]
        if not classes or not {"id", "prompt_id", "y_true"}.issubset(frame.columns):
            raise ValueError(f"candidate {candidate_id} lacks validation labels, groups, or probabilities")
        probabilities = frame[columns].to_numpy(dtype=np.float64)
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
            raise ValueError(f"candidate {candidate_id} has invalid probability rows")
        ordered = frame[["id", "prompt_id", "y_true"]].copy()
        ordered["id"] = ordered.id.astype(str)
        ordered["prompt_id"] = ordered.prompt_id.astype(str)
        ordered["y_true"] = ordered.y_true.astype(str)
        if ordered.id.duplicated().any():
            raise ValueError(f"candidate {candidate_id} has duplicate validation IDs")
        order = np.argsort(ordered.id.to_numpy(), kind="stable")
        ordered = ordered.iloc[order].reset_index(drop=True)
        probabilities = probabilities[order]
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        candidates.append({
            "candidate_id": candidate_id,
            "model": str(manifest.get("model", "")),
            "dataset": str(manifest.get("dataset", "unknown")),
            "manifest_path": str(manifest_path.resolve()),
            "model_manifest": manifest,
            "validation_score": float((manifest.get("metrics") or {}).get("balanced_accuracy", 0.0) or 0.0),
            "classes": classes,
            "frame": ordered,
            "probabilities": probabilities,
        })
    return candidates


def evaluate_svc_sweep(artifact_dir="artifacts", *, registry_path="docs/ensembles.md",
                       output_dir=None, variant_ids=None, baseline_id=None):
    """Evaluate standalone SVCs and champion blends with only the SVC member replaced."""
    artifact_dir = Path(artifact_dir)
    prediction_dir = artifact_dir / "predictions"
    output_dir = Path(output_dir) if output_dir else prediction_dir / "svc_v1_1_c_sweep"
    registry = read_ensemble_registry(registry_path)
    if not registry.get("ensembles"):
        raise ValueError(f"no champion ensemble found in {registry_path}")
    champion = registry["ensembles"][-1]
    members = [_member_id(item) for item in champion.get("selected_members", [])]
    svc_members = [item for item in members if item.startswith("svc_")]
    if len(svc_members) != 1:
        raise ValueError(f"expected exactly one SVC in current champion, found {svc_members}")
    champion_svc_id = svc_members[0]
    strategy = champion.get("selected_strategy", "equal_probability_blend")
    if strategy not in {"equal_probability_blend", "optimized_probability_blend"}:
        raise ValueError(f"unsupported champion strategy for SVC replacement: {strategy}")
    weights = champion.get("selected_weights") or {}
    if strategy == "equal_probability_blend":
        member_weights = np.full(len(members), 1.0 / len(members), dtype=float)
    else:
        if set(weights) != set(members):
            raise ValueError("weighted champion must have a saved weight for every selected member")
        member_weights = np.asarray([float(weights[item]) for item in members], dtype=float)
        if (not np.isfinite(member_weights).all() or np.any(member_weights < 0)
                or not np.isclose(member_weights.sum(), 1.0, atol=1e-6)):
            raise ValueError("champion weights are invalid or do not sum to one")

    requested = set(variant_ids or [])
    if baseline_id:
        requested.add(str(baseline_id))
    requested.update(members)
    candidates = _read_selected_candidates(prediction_dir, sorted(requested))
    by_id = {item["candidate_id"]: item for item in candidates}
    missing = requested - set(by_id)
    if missing:
        raise FileNotFoundError(f"missing validation prediction artifacts: {sorted(missing)}")
    aligned_ids = [*members, *([baseline_id] if baseline_id else []), *sorted(set(variant_ids or []))]
    aligned_candidates = [by_id[item] for item in dict.fromkeys(aligned_ids)]
    classes, y_true, groups, aligned, excluded, reference_ids = _align_candidates(
        aligned_candidates, reference_frame=by_id[champion_svc_id]["frame"])
    if excluded:
        raise ValueError(f"sweep candidates do not align to the champion validation rows: {excluded}")
    aligned_by_id = {item["candidate_id"]: item for item in aligned}
    if len(reference_ids) != 540:
        raise ValueError(f"expected the fixed 540-row validation set, found {len(reference_ids)} rows")

    champion_probabilities = sum(
        weight * aligned_by_id[candidate_id]["probabilities"]
        for candidate_id, weight in zip(members, member_weights))
    champion_metrics, champion_predictions = _probability_metrics(
        champion_probabilities, y_true, classes)
    baseline_probabilities = None
    baseline_metrics = None
    baseline_predictions = None
    baseline_standalone_metrics = None
    baseline_standalone_predictions = None
    if baseline_id:
        baseline_standalone_metrics, baseline_standalone_predictions = _probability_metrics(
            aligned_by_id[baseline_id]["probabilities"], y_true, classes)
        baseline_probabilities = sum(
            weight * (aligned_by_id[baseline_id]["probabilities"] if candidate_id == champion_svc_id
                      else aligned_by_id[candidate_id]["probabilities"])
            for candidate_id, weight in zip(members, member_weights))
        baseline_metrics, baseline_predictions = _probability_metrics(
            baseline_probabilities, y_true, classes)

    rows = []
    predictions = {"id": reference_ids, "prompt_id": groups, "y_true": y_true,
                   "champion_prediction": champion_predictions}
    if baseline_predictions is not None:
        predictions["c_0_5_blend_prediction"] = baseline_predictions
        predictions["svc_0_5_prediction"] = baseline_standalone_predictions
        rows.append({
            "C": 0.5,
            "candidate_id": baseline_id,
            "is_reference_baseline": True,
            "standalone_accuracy": baseline_standalone_metrics["accuracy"],
            "standalone_balanced_accuracy": baseline_standalone_metrics["balanced_accuracy"],
            "standalone_log_loss": baseline_standalone_metrics["log_loss"],
            "champion_blend_accuracy": baseline_metrics["accuracy"],
            "champion_blend_balanced_accuracy": baseline_metrics["balanced_accuracy"],
            "champion_blend_log_loss": baseline_metrics["log_loss"],
            "delta_accuracy_vs_champion": baseline_metrics["accuracy"] - champion_metrics["accuracy"],
            "unique_corrections_vs_champion": int(np.sum(
                (champion_predictions != y_true) & (baseline_predictions == y_true))),
            "unique_regressions_vs_champion": int(np.sum(
                (champion_predictions == y_true) & (baseline_predictions != y_true))),
            "delta_accuracy_vs_c0_5_blend": 0.0,
            "unique_corrections_vs_c0_5": 0,
            "unique_regressions_vs_c0_5": 0,
        })
    for candidate_id in sorted(set(variant_ids or [])):
        variant = aligned_by_id[candidate_id]
        standalone_metrics, standalone_predictions = _probability_metrics(
            variant["probabilities"], y_true, classes)
        blend_probabilities = sum(
            weight * (variant["probabilities"] if member_id == champion_svc_id
                      else aligned_by_id[member_id]["probabilities"])
            for member_id, weight in zip(members, member_weights))
        blend_metrics, blend_predictions = _probability_metrics(blend_probabilities, y_true, classes)
        row = {
            "C": float(variant["model_manifest"].get("config", {}).get("C")),
            "candidate_id": candidate_id,
            "is_reference_baseline": False,
            "standalone_accuracy": standalone_metrics["accuracy"],
            "standalone_balanced_accuracy": standalone_metrics["balanced_accuracy"],
            "standalone_log_loss": standalone_metrics["log_loss"],
            "champion_blend_accuracy": blend_metrics["accuracy"],
            "champion_blend_balanced_accuracy": blend_metrics["balanced_accuracy"],
            "champion_blend_log_loss": blend_metrics["log_loss"],
            "delta_accuracy_vs_champion": blend_metrics["accuracy"] - champion_metrics["accuracy"],
            "unique_corrections_vs_champion": int(np.sum(
                (champion_predictions != y_true) & (blend_predictions == y_true))),
            "unique_regressions_vs_champion": int(np.sum(
                (champion_predictions == y_true) & (blend_predictions != y_true))),
        }
        if baseline_metrics is not None:
            row.update({
                "delta_accuracy_vs_c0_5_blend": blend_metrics["accuracy"] - baseline_metrics["accuracy"],
                "unique_corrections_vs_c0_5": int(np.sum(
                    (baseline_predictions != y_true) & (blend_predictions == y_true))),
                "unique_regressions_vs_c0_5": int(np.sum(
                    (baseline_predictions == y_true) & (blend_predictions != y_true))),
            })
        rows.append(row)
        predictions[f"svc_{str(row['C']).replace('.', '_')}_prediction"] = standalone_predictions
        predictions[f"blend_{str(row['C']).replace('.', '_')}_prediction"] = blend_predictions

    results = pd.DataFrame(rows).sort_values(
        ["champion_blend_accuracy", "champion_blend_log_loss", "standalone_accuracy", "C"],
        ascending=[False, True, False, True], kind="stable").reset_index(drop=True)
    best_variant = (results.loc[~results.is_reference_baseline].iloc[0].to_dict()
                    if not results.empty else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(predictions).to_parquet(output_dir / "validation_predictions.parquet", index=False)
    report = {
        "experiment": "svc_v1.1_C_sweep",
        "dataset": "v1.1",
        "validation_rows": len(y_true),
        "validation_prompt_groups": int(len(set(groups))),
        "validation_ids_sha256": by_id[champion_svc_id]["model_manifest"].get("id_sha256"),
        "seed": 2026,
        "features": {"views": ["char", "word"]},
        "class_weight": "balanced",
        "calibration": "sigmoid",
        "calibration_cv": 3,
        "champion_experiment_id": champion.get("prediction_experiment_id"),
        "champion_strategy": strategy,
        "champion_members": members,
        "champion_weights": dict(zip(members, member_weights.tolist())),
        "champion_metrics": champion_metrics,
        "v1_1_c_0_5_reference": ({
            "candidate_id": baseline_id,
            "standalone_metrics": baseline_standalone_metrics,
            "champion_replacement_metrics": baseline_metrics,
        } if baseline_id else None),
        "candidates": results.to_dict(orient="records"),
        "best_variant": best_variant,
        "selection": "highest substituted-champion accuracy, then lower blend log loss, then higher standalone accuracy",
        "scope_note": "Exploratory: these variants are compared and ranked on the same fixed 540-row validation set. Do not treat the winning score as an independent estimate.",
        "champion_replacement_note": "Each candidate replaces only the champion's SVC member; all other members, strategy, and weights are held fixed.",
        "candidate_read_problems": [],
        "results_csv": str(output_dir / "results.csv"),
        "validation_predictions": str(output_dir / "validation_predictions.parquet"),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_svc_sweep(*, dataset="v1.1", data_dir="data/processed", artifact_dir="artifacts",
                  registry_path="experiments/registry.sqlite", ensemble_registry="docs/ensembles.md",
                  output_dir=None, c_values=DEFAULT_C_VALUES, seed=2026):
    """Train requested C variants without refreshing the active prediction roster."""
    artifact_dir = Path(artifact_dir)
    manifests_dir = artifact_dir / "predictions"
    # Preserve the frozen C=0.5 reference run; the sweep never retrains it.
    baseline = []
    for manifest_path in sorted(manifests_dir.glob("*_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = manifest.get("config", {})
        if (manifest.get("model") == "svc" and manifest.get("dataset") == dataset
                and manifest.get("seed") == seed and manifest.get("split") == "validation"
                and float(config.get("C", -1)) == 0.5
                and config.get("features", {}).get("views") == ["char", "word"]
                and config.get("calibration") == "sigmoid"
                and int(config.get("calibration_cv", 0)) == 3):
            baseline.append(str(manifest.get(
                "experiment_id", manifest_path.stem.removesuffix("_manifest"))))
    if len(baseline) != 1:
        raise ValueError(f"expected one matching V1.1 C=0.5 baseline run, found {len(baseline)}")
    baseline_id = baseline[0]

    variant_ids = []
    for c_value in c_values:
        run = train_models(dataset, "svc", data_dir, artifact_dir, registry_path,
                           seed=seed, config_overrides={"C": float(c_value)},
                           refresh_prediction=False)
        variant_ids.append(run["experiment_id"])
        config = run.get("config", {})
        if (float(config.get("C", -1)) != float(c_value)
                or run.get("dataset") != dataset or run.get("seed") != seed
                or run.get("split") != "validation"):
            raise RuntimeError(f"run metadata does not match requested C={c_value}: {run['experiment_id']}")
    report = evaluate_svc_sweep(artifact_dir, registry_path=ensemble_registry,
                                output_dir=output_dir, variant_ids=variant_ids,
                                baseline_id=baseline_id)
    report["trained_candidate_ids"] = variant_ids
    report["baseline_candidate_id"] = baseline_id
    report["registry_path"] = str(ensemble_registry)
    report["experiment_registry_path"] = str(registry_path)
    target = Path(output_dir) if output_dir else artifact_dir / "predictions" / "svc_v1_1_c_sweep"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v1.1")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--registry", default="experiments/registry.sqlite")
    parser.add_argument("--ensemble-registry", default="docs/ensembles.md")
    parser.add_argument("--output-dir")
    parser.add_argument("--c-values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = run_svc_sweep(dataset=args.dataset, data_dir=args.data_dir,
                           artifact_dir=args.artifact_dir, registry_path=args.registry,
                           ensemble_registry=args.ensemble_registry, output_dir=args.output_dir,
                           c_values=args.c_values, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
