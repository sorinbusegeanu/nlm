"""Sequential CatBoost feature-recipe experiments across V1-all and V0."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .evaluation import evaluate
from .feature_factory import (BASE_STYLOMETRY_GROUPS, STYLOMETRY_GROUPS,
                              stylometry)
from .model_factory import train_catboost_stylometry
from .registry import Registry


EXPERIMENTS = {
    "A": {"name": "P+F", "recipe": {"baseline": list(BASE_STYLOMETRY_GROUPS),
                                      "groups": ["P", "F"]}},
    "B": {"name": "All", "recipe": {"baseline": list(BASE_STYLOMETRY_GROUPS),
                                       "groups": list(STYLOMETRY_GROUPS)}},
}
BASELINE_FEATURE_NAMES = {
    "length": {"char_count", "word_count", "sentence_count", "chars_per_word",
               "words_per_sentence", "mean_word_length"},
    "case": {"uppercase_char_fraction", "digit_char_fraction"},
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _normal_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def _require_same_validation(stage1: pd.DataFrame, stage2: pd.DataFrame, classes1, classes2) -> None:
    """Fail closed unless both stages score the exact same rows and labels."""
    a = stage1.sort_values("id", kind="stable").reset_index(drop=True)
    b = stage2.sort_values("id", kind="stable").reset_index(drop=True)
    if list(classes1) != list(classes2):
        raise ValueError("stage validation class order differs")
    if not a.id.astype(str).equals(b.id.astype(str)):
        raise ValueError("stage validation IDs differ")
    if not a.model.astype(str).equals(b.model.astype(str)):
        raise ValueError("stage validation labels differ")


def _feature_group(name: str) -> str:
    if name.startswith("base_punct_"):
        return "punctuation"
    if name.startswith("base_fw_"):
        return "function_words"
    for group in STYLOMETRY_GROUPS:
        if name.startswith(group + "_"):
            return group
    for group, names in BASELINE_FEATURE_NAMES.items():
        if name in names:
            return group
    raise ValueError(f"cannot assign stylometry feature to a group: {name}")


def _aggregate_feature_importance(feature_names, importances):
    importances = np.asarray(importances, dtype=float)
    if importances.shape != (len(feature_names),) or not np.isfinite(importances).all():
        raise ValueError("CatBoost returned invalid feature importances")
    rows = {}
    for name, value in zip(feature_names, importances):
        group = _feature_group(name)
        rows.setdefault(group, []).append(float(value))
    return {group: {"feature_count": len(values), "mean_importance": float(np.mean(values)),
                    "total_importance": float(np.sum(values))}
            for group, values in sorted(rows.items())}


def _pruned_recipe(group_importance):
    eligible = [group for group in STYLOMETRY_GROUPS if group not in {"P", "F"}]
    missing = set(eligible) - set(group_importance)
    if missing:
        raise ValueError(f"missing importance values for eligible groups: {sorted(missing)}")
    removed = sorted(eligible, key=lambda group: (group_importance[group]["mean_importance"], group))[:2]
    kept = [group for group in STYLOMETRY_GROUPS if group not in removed]
    if not {"P", "F"}.issubset(kept):
        raise RuntimeError("pruned recipe must retain P and F")
    return {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": kept}, removed


def _select_matrix(full_matrix, all_feature_names, recipe):
    _, wanted = stylometry(["feature schema probe"], recipe, return_feature_names=True)
    indexes = {name: index for index, name in enumerate(all_feature_names)}
    absent = [name for name in wanted if name not in indexes]
    if absent:
        raise ValueError(f"requested recipe features are absent from all-feature matrix: {absent[:5]}")
    return np.asarray(full_matrix[:, [indexes[name] for name in wanted]], dtype=np.float64), wanted


def _metrics(y_true, scores, probabilities, classes):
    result = evaluate(y_true, scores, classes, probabilities)
    prediction = np.asarray(classes)[np.asarray(scores).argmax(axis=1)]
    result["accuracy"] = float(accuracy_score(y_true, prediction))
    return result


def _write_prediction_frame(path, valid, classes, scores, probabilities):
    frame = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for index, label in enumerate(classes):
        frame[f"score::{label}"] = scores[:, index]
        frame[f"prob::{label}"] = probabilities[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _candidate_probabilities(prediction_dir, candidate_id, valid_ids, classes):
    path = prediction_dir / f"{candidate_id}_validation.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"saved champion member predictions are missing: {path}")
    frame = pd.read_parquet(path).sort_values("id", kind="stable").reset_index(drop=True)
    expected_ids = list(valid_ids)
    if frame.id.astype(str).tolist() != expected_ids:
        raise ValueError(f"saved champion candidate {candidate_id} does not align to V0 validation IDs")
    columns = [f"prob::{label}" for label in classes]
    if set(columns) - set(frame.columns):
        raise ValueError(f"saved champion candidate {candidate_id} is missing probability columns")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(1), 1, atol=1e-5):
        raise ValueError(f"saved champion candidate {candidate_id} has invalid probabilities")
    return values


def _saved_champion_probabilities(prediction_dir, registry_path, valid, classes):
    from .ensemble_registry import read_ensemble_registry

    registry = read_ensemble_registry(registry_path)
    entries = registry.get("ensembles", [])
    if not entries:
        return None
    champion = entries[-1]
    selected = champion.get("selected_members") or []
    strategy = champion.get("selected_strategy", "equal_probability_blend")
    if strategy not in {"equal_probability_blend", "optimized_weight_blend", "weighted_probability_blend"}:
        raise ValueError(f"cannot reproduce saved champion strategy: {strategy}")
    member_ids = [str(item.get("candidate_id") if isinstance(item, dict) else item)
                  for item in selected]
    if not member_ids:
        raise ValueError("saved champion has no selected members")
    member_probabilities = [_candidate_probabilities(prediction_dir, item, valid.id.astype(str), classes)
                            for item in member_ids]
    if strategy == "equal_probability_blend":
        weights = np.full(len(member_ids), 1.0 / len(member_ids))
    else:
        saved_weights = champion.get("selected_weights") or {}
        if not isinstance(saved_weights, dict) or set(saved_weights) != set(member_ids):
            raise ValueError("saved weighted champion has incomplete member weights")
        weights = np.asarray([float(saved_weights[item]) for item in member_ids], dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError("saved weighted champion weights are invalid")
    probabilities = np.einsum("m,mnc->nc", weights, np.stack(member_probabilities))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    prediction_id = champion.get("prediction_experiment_id")
    saved_output = prediction_dir / f"{prediction_id}_validation_predictions.parquet" if prediction_id else None
    if saved_output and saved_output.is_file():
        hard = pd.read_parquet(saved_output).sort_values("id", kind="stable").reset_index(drop=True)
        if hard.id.astype(str).tolist() != valid.id.astype(str).tolist():
            raise ValueError("saved champion hard predictions do not align to frozen V0 validation")
        if "champion_prediction" in hard:
            reproduced = np.asarray(classes)[probabilities.argmax(axis=1)]
            if not np.array_equal(reproduced.astype(str), hard.champion_prediction.astype(str).to_numpy()):
                raise ValueError("reconstructed champion probabilities do not reproduce saved champion predictions")
    return {"version": champion.get("version"), "prediction_experiment_id": prediction_id,
            "strategy": strategy, "member_ids": member_ids,
            "weights": {member: float(weight) for member, weight in zip(member_ids, weights)},
            "probabilities": probabilities}


def _stage_id(run_id, experiment, stage):
    return f"catboost_two_stage_{run_id}_{experiment}_{stage}"


def _stage_artifact_dir(run_root, experiment, stage):
    return Path(run_root) / experiment / stage


def _save_stage(*, experiment, experiment_name, stage, dataset_name, train, valid,
                classes, recipe, iterations, cfg, seed, checkpoint, parent_checkpoint,
                parent_sha256, matrices, report_dir, data_profile,
                registry_path, code_revision, validation_dataset):
    selected_matrix, feature_names = _select_matrix(matrices["full"], matrices["all_names"], recipe)
    if selected_matrix.shape[1] != len(feature_names):
        raise ValueError("feature matrix width differs from recipe schema")
    start, end = matrices["offsets"][dataset_name]
    x_train = selected_matrix[start:end]
    valid_start, valid_end = matrices["offsets"][validation_dataset]
    x_validation = selected_matrix[valid_start:valid_end]
    if len(x_train) != len(train) or len(x_validation) != len(valid):
        raise ValueError("stylometry matrix row count differs from selected data partition")

    checkpoint.mkdir(parents=True, exist_ok=False)
    init_path = parent_checkpoint / "model.cbm" if parent_checkpoint else None
    result = train_catboost_stylometry(
        train.text.tolist(), train.model.astype(str).tolist(), valid.text.tolist(), classes,
        output_dir=checkpoint, iterations=iterations, depth=cfg["depth"],
        learning_rate=cfg["learning_rate"], l2_leaf_reg=cfg["l2_leaf_reg"],
        thread_count=cfg["thread_count"], seed=seed,
        temperature_calibration=cfg.get("temperature_calibration", True),
        calibration_enabled=cfg.get("calibration_enabled", cfg.get("temperature_calibration", True)),
        calibration_max_samples=cfg.get("calibration_max_samples", 5000),
        calibration_seed=cfg.get("calibration_seed", seed),
        calibration_full_dataset=cfg.get("calibration_full_dataset", False),
        validation_labels=valid.model.astype(str).tolist(),
        early_stopping_rounds=cfg.get("early_stopping_rounds", 100),
        checkpoint_selection_metric="balanced_accuracy", feature_recipe=recipe,
        feature_matrices=(x_train, x_validation, feature_names), init_model=init_path,
        checkpoint_metadata={"experiment": experiment, "experiment_name": experiment_name,
                             "stage": stage, "dataset": dataset_name,
                             "validation_dataset": validation_dataset,
                             "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
                             "parent_checkpoint_sha256": parent_sha256})
    if list(result["feature_names"]) != feature_names:
        raise ValueError("trainer feature schema differs from the selected recipe")
    parent_tree_count = (int(json.loads((parent_checkpoint / "stylometry.json").read_text()).get("tree_count", 0))
                         if parent_checkpoint else 0)
    expected_trees = iterations + parent_tree_count
    if int(result["tree_count"]) < 1 or int(result["tree_count"]) > expected_trees:
        raise RuntimeError(f"CatBoost produced {result['tree_count']} trees outside 1..{expected_trees}")

    scores = np.asarray(result["scores"], dtype=float)
    # CatBoost's fitted model already holds the selected best iteration and is
    # the object serialized to model.cbm; reuse its aligned validation outputs.
    probabilities = np.asarray(result["probabilities"], dtype=float)
    metric_values = _metrics(valid.model.astype(str).to_numpy(), scores, probabilities, classes)

    run_id = _stage_id(report_dir.name, experiment, stage)
    stage_artifact_dir = checkpoint.parent
    prediction_path = stage_artifact_dir / f"{run_id}_validation.parquet"
    export_started = time.perf_counter()
    post_training_started = export_started
    prediction_frame = _write_prediction_frame(prediction_path, valid, classes, scores, probabilities)
    dataset_hash = data_profile["dataset_sha256"]
    split_hash = data_profile.get("split_sha256", "")
    validation_hash = hashlib.sha256(valid[["id", "prompt_id", "text", "model"]]
        .astype(str).to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    config = {"model": "catboost_stylometry", "experiment": experiment,
              "experiment_name": experiment_name, "stage": stage,
              "training_dataset": dataset_name, "iterations_added": int(iterations),
              "cumulative_tree_count": int(result["tree_count"]),
              "depth": int(cfg["depth"]), "learning_rate": float(cfg["learning_rate"]),
              "l2_leaf_reg": float(cfg["l2_leaf_reg"]), "class_weights": "Balanced",
              "seed": int(seed), "feature_recipe": recipe,
              "feature_names": feature_names,
              "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
              "parent_checkpoint_sha256": parent_sha256,
              "calibration_method": result["calibration_method"],
              "calibration_temperature": result["calibration_temperature"],
              "calibration_max_samples": int(cfg.get("calibration_max_samples", 5000)),
              "calibration_full_dataset": bool(cfg.get("calibration_full_dataset", False)),
              "calibration_seed": int(cfg.get("calibration_seed", seed)),
              "validation_fitting": False,
              "validation_dataset": validation_dataset,
              "early_stopping_metric": "balanced_accuracy"}
    config_hash = _sha(config)
    experiment_id = _stage_id(report_dir.name, experiment, stage)
    manifest_path = stage_artifact_dir / f"{experiment_id}_manifest.json"
    checkpoint_size = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    manifest = {
        "experiment_id": experiment_id, "experiment_name": experiment_name,
        "model": "catboost_stylometry", "dataset": dataset_name,
        "training_dataset": dataset_name, "validation_dataset": validation_dataset,
        "stage": stage, "source_dataset": dataset_name, "split": "validation",
        "classes": list(classes), "train_rows": len(train), "validation_rows": len(valid),
        "training_dataset_rows": len(train), "validation_dataset_rows": len(valid),
        "training_dataset_sha256": hashlib.sha256(train[["id", "prompt_id", "text", "model"]]
            .astype(str).to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "validation_dataset_sha256": validation_hash,
        "train_id_sha256": hashlib.sha256("\n".join(train.id.astype(str)).encode()).hexdigest(),
        "id_sha256": hashlib.sha256("\n".join(valid.id.astype(str)).encode()).hexdigest(),
        "dataset_sha256": dataset_hash, "split_sha256": split_hash,
        "code_revision": code_revision, "seed": int(seed), "config": config,
        "config_sha256": config_hash, "fit_seconds": float(result["fit_seconds"]),
        "inference_seconds": float(result["inference_seconds"]),
        "runtime_breakdown": {**(result.get("runtime_breakdown") or {}),
            "best_checkpoint_restoration_seconds": 0.0,
            "calibration_inference_seconds": result.get("calibration_inference_seconds"),
            "temperature_fit_seconds": result.get("temperature_fit_seconds"),
            "final_validation_inference_seconds": float(result["inference_seconds"]),
            "checkpoint_serialization_seconds": (result.get("runtime_breakdown") or {}).get(
                "checkpoint_serialization_seconds", (result.get("runtime_breakdown") or {}).get("checkpoint_save_seconds", 0.0)),
            "manifest_prediction_export_seconds": 0.0,
            "downstream_refresh_seconds": 0.0},
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_path": str(checkpoint), "parent_checkpoint": config["parent_checkpoint"],
        "checkpoint_scoring_method": result.get("checkpoint_scoring_method"),
        "final_validation_reused_existing_predictions": result.get("final_validation_reused_existing_predictions", True),
        "parent_checkpoint_sha256": parent_sha256,
        "iterations_added": int(iterations), "tree_count": int(result["tree_count"]),
        "feature_count": len(feature_names), "feature_names": feature_names,
        "feature_recipe": recipe, "calibration_method": result["calibration_method"],
        "calibration_enabled": result.get("calibration_enabled", cfg.get("calibration_enabled", True)),
        "calibration_temperature": float(result["calibration_temperature"]),
        "calibration_sample_size": result.get("calibration_sample_size"),
        "calibration_full_dataset": result.get("calibration_full_dataset"),
        "validation_metrics_by_epoch": [], "early_stopping": None,
        "validation_labels_used_for_checkpoint_selection": True,
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha_file(prediction_path), "metrics": metric_values,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (stage_artifact_dir / f"{experiment_id}_metrics.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    Registry(registry_path).add(
        experiment_id, "catboost_stylometry", "validation", config, metric_values,
        dataset_hash=dataset_hash, code_revision=code_revision, seed=seed,
        runtime_seconds=float(result["fit_seconds"]), artifact_path=str(prediction_path),
        config_hash=config_hash, split_hash=split_hash,
        description=(f"{experiment} {experiment_name} stage {stage}: CatBoost trained on "
                     f"{dataset_name}, {len(train)} train rows; {len(valid)} frozen V0 validation rows."))
    manifest["runtime_breakdown"]["manifest_prediction_export_seconds"] = time.perf_counter() - export_started
    manifest["runtime_breakdown"]["manifest_export_seconds"] = manifest["runtime_breakdown"]["manifest_prediction_export_seconds"]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (stage_artifact_dir / f"{experiment_id}_metrics.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    manifest["runtime_breakdown"]["total_post_training_seconds"] = time.perf_counter() - post_training_started
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (stage_artifact_dir / f"{experiment_id}_metrics.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if experiment == "B" and stage == "stage2":
        importance = result["model"].get_feature_importance(type="FeatureImportance")
        importance_by_group = _aggregate_feature_importance(feature_names, importance)
    else:
        importance_by_group = None
    tree_count = int(result["tree_count"])
    del result
    return {"experiment_id": experiment_id, "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint), "prediction_path": str(prediction_path),
            "metrics": metric_values, "feature_count": len(feature_names),
            "feature_recipe": recipe, "feature_names": feature_names,
            "iterations_added": int(iterations), "tree_count": tree_count,
            "fit_seconds": manifest["fit_seconds"],
            "importance_by_group": importance_by_group,
            "per_class_recall": metric_values["per_class_recall"],
            "prediction_frame": prediction_frame}


def _render_report(path, summary):
    lines = ["# CatBoost two-stage feature experiments", "",
             f"Run: `{summary['run_id']}`", "",
             f"Validation uses all {summary['validation_rows']} rows from `{summary['validation_dataset']}`. "
             "Balanced accuracy selects the best boosting iteration; calibration uses training rows only. "
             "The scores remain development results because this set has been used for prior model selection.", "",
             f"Exact response-text overlap between Stage 1 V1-all training and Stage 2 V0 training: "
             f"**{summary['training_overlap']['exact_rows']}**; normalized overlap: "
             f"**{summary['training_overlap']['normalized_rows']}**.", "",
             "## Stage comparison", "",
             "| Experiment | Recipe | Features | Stage 1 V1-all bAcc | Stage 2 V0 bAcc | Delta |",
             "|---|---|---:|---:|---:|---:|"]
    for code in ("A", "B", "C"):
        exp = summary["experiments"][code]
        first, second = exp["stage1"], exp["stage2"]
        delta = second["metrics"]["balanced_accuracy"] - first["metrics"]["balanced_accuracy"]
        lines.append(f"| {code} — {exp['name']} | `{exp['recipe_label']}` | {second['feature_count']} | "
                     f"{first['metrics']['balanced_accuracy']:.4f} | "
                     f"{second['metrics']['balanced_accuracy']:.4f} | {delta:+.4f} |")
    lines += ["", "## All six checkpoints", "",
              "| Experiment / stage | Training data | Trees (added / cumulative) | Accuracy | Balanced accuracy | Log loss |",
              "|---|---|---:|---:|---:|---:|"]
    for code in ("A", "B", "C"):
        exp = summary["experiments"][code]
        for key, label in (("stage1", "Stage 1"), ("stage2", "Stage 2")):
            stage = exp[key]
            metrics = stage["metrics"]
            lines.append(f"| {code} — {label} | {stage['dataset']} | {stage['iterations_added']} / "
                         f"{stage['tree_count']} | {metrics['accuracy']:.4f} | "
                         f"{metrics['balanced_accuracy']:.4f} | {metrics['log_loss']:.4f} |")
    lines += ["", "## Experiment B importance and Experiment C recipe", "",
              "Importance is CatBoost `PredictionValuesChange`; group means are the average feature importance per feature.", "",
              "| Group | Features | Mean importance | Total importance |",
              "|---|---:|---:|---:|"]
    for group, row in sorted(summary["b_feature_importance"].items()):
        lines.append(f"| {group} | {row['feature_count']} | {row['mean_importance']:.6f} | {row['total_importance']:.4f} |")
    lines += ["", f"Removed for C: **{', '.join(summary['c_removed_groups'])}**.",
              f"C recipe groups: `{summary['experiments']['C']['recipe']['groups']}`.", "",
              "Per-class recall for each of the six checkpoints is in [`per_class_recall.csv`](per_class_recall.csv).", "",
              "## Best checkpoint and champion contribution", ""]
    champion = summary.get("champion_comparison")
    if champion:
        best = summary["best_checkpoint"]
        lines.append(f"Best balanced accuracy: **{best['balanced_accuracy']:.4f}**.")
        ties = best.get("tied_checkpoints", [])
        if len(ties) > 1:
            names = ", ".join(f"`{item['experiment_id']}` ({item['feature_count']} features)" for item in ties)
            lines.append(f"Co-best checkpoints: {names}. Representative selected: "
                         f"`{best['experiment_id']}` (fewest features). Their saved probability predictions are "
                         f"{'identical' if best.get('co_best_predictions_identical') else 'not identical'}.")
        else:
            lines.append(f"Best checkpoint: `{best['experiment_id']}`.")
        lines += [
                  f"Saved champion: {champion['champion_version']} / `{champion['strategy']}`; "
                  f"accuracy {champion['champion_metrics']['accuracy']:.4f}, "
                  f"balanced accuracy {champion['champion_metrics']['balanced_accuracy']:.4f}.",
                  f"50:50 champion + best-checkpoint blend: accuracy {champion['blend_metrics']['accuracy']:.4f}, "
                  f"balanced accuracy {champion['blend_metrics']['balanced_accuracy']:.4f}, "
                  f"delta {champion['blend_metrics']['balanced_accuracy'] - champion['champion_metrics']['balanced_accuracy']:+.4f}.",
                  "The comparison is exploratory on the same validation set and does not change the active champion."]
    else:
        lines.append("No saved champion was available for a probability-level contribution comparison.")
    lines += ["", "## Artifacts", "",
              "Six stage manifests, prediction files, and checkpoint directories are recorded in the summary JSON; "
              "the active prediction roster and ensemble registry were not modified.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_catboost_two_stage(*, load_dataset, effective_profile, data_dir="data/processed",
                           artifact_dir="artifacts", registry_path="experiments/registry.sqlite",
                           ensemble_registry_path="docs/ensembles.md",
                           config=None, seed=2026, stage1_iterations=800,
                           stage2_iterations=200, run_id=None, validation_dataset="v0"):
    """Train A/B/C in order without changing the active prediction roster."""
    if stage1_iterations < 1 or stage2_iterations < 1:
        raise ValueError("stage iteration budgets must be positive")
    cfg = {"depth": 7, "learning_rate": .05, "l2_leaf_reg": 5.0,
           "thread_count": 8, "temperature_calibration": True, **(config or {})}
    if run_id is None:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, dots, underscores, and hyphens")
    artifact_dir = Path(artifact_dir)
    data_dir = Path(data_dir)
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "analysis-output" / "catboost_two_stage_v1_all_vs_v0" / run_id
    run_root = artifact_dir / "runs" / "catboost_two_stage" / run_id
    if report_dir.exists() or run_root.exists():
        raise FileExistsError(f"two-stage run already exists: {run_id}")

    v1_rows = load_dataset("v1_all", data_dir)[0].drop(columns=["split"], errors="ignore")
    v0_rows = load_dataset("v0", data_dir)[0].drop(columns=["split"], errors="ignore")
    validation_rows = load_dataset(validation_dataset, data_dir)[0].drop(columns=["split"], errors="ignore")
    v1_rows = v1_rows.sort_values("id", kind="stable").reset_index(drop=True)
    v0_rows = v0_rows.sort_values("id", kind="stable").reset_index(drop=True)
    validation_rows = validation_rows.sort_values("id", kind="stable").reset_index(drop=True)
    profile_v1 = effective_profile(data_dir, "v1_all", v1_rows)
    profile_v0 = effective_profile(data_dir, "v0", v0_rows)
    profile_validation = effective_profile(data_dir, validation_dataset, validation_rows)
    stage1_train, stage2_train, stage1_valid = v1_rows, v0_rows, validation_rows
    stage2_valid = validation_rows
    classes = sorted(validation_rows.model.astype(str).unique())
    if set(stage1_train.model.astype(str).unique()) != set(classes) or set(stage2_train.model.astype(str).unique()) != set(classes):
        raise ValueError("Stage 1 and Stage 2 must each contain all canonical classes")
    _require_same_validation(stage1_valid, stage2_valid, classes, classes)
    champion = None
    if set(stage1_train.id.astype(str)) & set(stage2_train.id.astype(str)):
        raise ValueError("training IDs overlap across Stage 1 and Stage 2")
    stage1_texts = stage1_train.text.astype(str)
    stage2_texts = stage2_train.text.astype(str)
    exact_overlap = set(stage1_texts) & set(stage2_texts)
    normalized_overlap = set(stage1_texts.map(_normal_text)) & set(stage2_texts.map(_normal_text))
    validation_texts = stage1_valid.text.astype(str)
    training_ids = set(stage1_train.id.astype(str)) | set(stage2_train.id.astype(str))
    validation_ids = set(stage1_valid.id.astype(str))
    if training_ids & validation_ids:
        raise ValueError("validation IDs overlap a training partition")
    if (set(validation_texts) & set(stage1_texts)) or (set(validation_texts) & set(stage2_texts)):
        raise ValueError("validation response text occurs in a training partition")
    if set(validation_texts.map(_normal_text)) & (set(stage1_texts.map(_normal_text)) | set(stage2_texts.map(_normal_text))):
        raise ValueError("normalized validation response text occurs in a training partition")
    training_prompt_ids = set(stage1_train.prompt_id.astype(str)) | set(stage2_train.prompt_id.astype(str))
    if training_prompt_ids & set(stage1_valid.prompt_id.astype(str)):
        raise ValueError("validation prompt groups overlap a training partition")
    for prompt_column in ("prompt_canon", "prompt", "prompt_text"):
        if prompt_column in stage1_train and prompt_column in stage1_valid:
            train_prompts = set(pd.concat([stage1_train[prompt_column], stage2_train[prompt_column]], ignore_index=True).fillna("").astype(str).map(_normal_text)) - {""}
            valid_prompts = set(stage1_valid[prompt_column].fillna("").astype(str).map(_normal_text)) - {""}
            if train_prompts & valid_prompts:
                raise ValueError("normalized validation prompt content occurs in a training partition")
    if set(stage1_valid.model.astype(str)) != set(classes):
        raise ValueError("validation label set differs from canonical classes")

    all_recipe = {"baseline": list(BASE_STYLOMETRY_GROUPS), "groups": list(STYLOMETRY_GROUPS)}
    all_texts = (stage1_texts.tolist() + stage2_texts.tolist() + validation_texts.tolist())
    feature_started = time.perf_counter()
    full_matrix, all_feature_names = stylometry(all_texts, all_recipe, return_feature_names=True)
    feature_seconds = time.perf_counter() - feature_started
    feature_data = {"full": full_matrix, "all_names": all_feature_names,
                    "offsets": {"v1_all": (0, len(stage1_train)),
                                "v0": (len(stage1_train), len(stage1_train) + len(stage2_train)),
                                validation_dataset: (len(stage1_train) + len(stage2_train), len(all_texts))}}
    report_dir.mkdir(parents=True)
    run_root.mkdir(parents=True)

    summary = {"run_id": run_id, "seed": seed,
               "dataset_hashes": {"v1_all": profile_v1["dataset_sha256"], "v0": profile_v0["dataset_sha256"],
                                  validation_dataset: profile_validation["dataset_sha256"]},
               "split_hashes": {"v1_all": profile_v1.get("split_sha256"), "v0": profile_v0.get("split_sha256"),
                                validation_dataset: profile_validation.get("split_sha256")},
               "feature_extraction_seconds": feature_seconds,
               "validation_dataset": validation_dataset,
               "validation_rows": len(stage1_valid),
               "validation_ids_sha256": hashlib.sha256("\n".join(stage1_valid.id.astype(str)).encode()).hexdigest(),
               "classes": classes,
               "training_overlap": {"exact_rows": len(exact_overlap), "normalized_rows": len(normalized_overlap),
                                    "v1_all_train_rows": len(stage1_train), "v0_train_rows": len(stage2_train)},
               "experiments": {}, "code_revision": _revision(),
        "training_parameters": {"stage1_iterations": int(stage1_iterations),
                   "stage2_iterations": int(stage2_iterations), "depth": int(cfg["depth"]),
                   "learning_rate": float(cfg["learning_rate"]), "l2_leaf_reg": float(cfg["l2_leaf_reg"]),
                   "class_weights": "Balanced", "seed": int(seed), "thread_count": int(cfg["thread_count"]),
                   "calibration": "train-only scalar temperature", "validation_fitting": False,
                   "early_stopping_rounds": int(cfg.get("early_stopping_rounds", 100)),
                   "checkpoint_selection_metric": "balanced_accuracy"}}

    recipes = {"A": EXPERIMENTS["A"]["recipe"], "B": EXPERIMENTS["B"]["recipe"]}
    removed_groups = []
    for experiment in ("A", "B"):
        definition = EXPERIMENTS[experiment]
        experiment_dir = run_root / experiment
        experiment_dir.mkdir()
        recipe = recipes[experiment]
        record = {"name": definition["name"], "recipe": recipe,
                  "recipe_label": "Baseline + P,F" if experiment == "A" else "Baseline + all groups"}
        for stage, dataset_name, train, valid, profile, budget, parent in (
            ("stage1", "v1_all", stage1_train, stage1_valid, profile_v1, stage1_iterations, None),
            ("stage2", "v0", stage2_train, stage2_valid, profile_v0, stage2_iterations,
             experiment_dir / "stage1" / "checkpoint"),
        ):
            stage_dir = _stage_artifact_dir(run_root, experiment, stage)
            stage_dir.mkdir()
            checkpoint = stage_dir / "checkpoint"
            parent_hash = _sha_file(parent / "model.cbm") if parent else None
            stage_result = _save_stage(
                experiment=experiment, experiment_name=definition["name"], stage=stage,
                dataset_name=dataset_name,
                train=train, valid=valid, classes=classes, recipe=recipe,
                iterations=budget, cfg=cfg, seed=seed, checkpoint=checkpoint,
                parent_checkpoint=parent, parent_sha256=parent_hash, matrices=feature_data,
                report_dir=report_dir, data_profile=profile,
                registry_path=registry_path, code_revision=summary["code_revision"],
                validation_dataset=validation_dataset)
            stage_result.pop("prediction_frame")
            record[stage] = {**stage_result, "dataset": dataset_name}
        summary["experiments"][experiment] = record
        if experiment == "B":
            importance = record["stage2"]["importance_by_group"]
            if importance is None:
                raise RuntimeError("B Stage 2 did not yield feature importances")
            recipes["C"], removed_groups = _pruned_recipe(importance)
            summary["b_feature_importance"] = importance

    cdir = run_root / "C"
    cdir.mkdir()
    recipe = recipes["C"]
    record = {"name": "Pruned", "recipe": recipe,
              "recipe_label": "Baseline + all groups except " + ",".join(removed_groups)}
    for stage, dataset_name, train, valid, profile, budget, parent in (
        ("stage1", "v1_all", stage1_train, stage1_valid, profile_v1, stage1_iterations, None),
        ("stage2", "v0", stage2_train, stage2_valid, profile_v0, stage2_iterations,
         cdir / "stage1" / "checkpoint"),
    ):
        stage_dir = _stage_artifact_dir(run_root, "C", stage)
        stage_dir.mkdir()
        checkpoint = stage_dir / "checkpoint"
        parent_hash = _sha_file(parent / "model.cbm") if parent else None
        stage_result = _save_stage(
            experiment="C", experiment_name="Pruned", stage=stage,
            dataset_name=dataset_name,
            train=train, valid=valid, classes=classes, recipe=recipe,
            iterations=budget, cfg=cfg, seed=seed, checkpoint=checkpoint,
            parent_checkpoint=parent, parent_sha256=parent_hash, matrices=feature_data,
            report_dir=report_dir, data_profile=profile,
            registry_path=registry_path, code_revision=summary["code_revision"],
            validation_dataset=validation_dataset)
        stage_result.pop("prediction_frame")
        record[stage] = {**stage_result, "dataset": dataset_name}
    summary["experiments"]["C"] = record
    summary["c_removed_groups"] = removed_groups

    all_stages = [(code, key, summary["experiments"][code][key])
                  for code in ("A", "B", "C") for key in ("stage1", "stage2")]
    best_score = max(item[2]["metrics"]["balanced_accuracy"] for item in all_stages)
    tied = [item for item in all_stages
            if abs(item[2]["metrics"]["balanced_accuracy"] - best_score) <= 1e-12]
    code, key, best = min(tied, key=lambda item: (item[2]["feature_count"],
                                                   item[1] != "stage2", item[0]))
    tie_records = [{"experiment": item[0], "stage": item[1],
                    "experiment_id": item[2]["experiment_id"],
                    "feature_count": item[2]["feature_count"],
                    "balanced_accuracy": item[2]["metrics"]["balanced_accuracy"]}
                   for item in tied]
    identical_tied_predictions = True
    if len(tied) > 1:
        first = pd.read_parquet(tied[0][2]["prediction_path"]).sort_values("id", kind="stable").reset_index(drop=True)
        probability_columns = [f"prob::{label}" for label in classes]
        first_probabilities = first[probability_columns].to_numpy(dtype=float)
        for item in tied[1:]:
            other = pd.read_parquet(item[2]["prediction_path"]).sort_values("id", kind="stable").reset_index(drop=True)
            if not first.id.astype(str).equals(other.id.astype(str)) or not np.array_equal(
                    first_probabilities, other[probability_columns].to_numpy(dtype=float)):
                identical_tied_predictions = False
                break
    summary["best_checkpoint"] = {"experiment_id": best["experiment_id"],
                                   "balanced_accuracy": best_score,
                                   "checkpoint_path": best["checkpoint_path"], "experiment": code,
                                   "stage": key, "tied_checkpoints": tie_records,
                                   "co_best_predictions_identical": identical_tied_predictions}

    recall_rows = []
    for exp_code in ("A", "B", "C"):
        for stage_name in ("stage1", "stage2"):
            stage_record = summary["experiments"][exp_code][stage_name]
            for label, recall in stage_record["per_class_recall"].items():
                recall_rows.append({"experiment": exp_code, "stage": stage_name,
                                    "dataset": stage_record["dataset"], "model": label,
                                    "recall": recall})
    recalls_path = report_dir / "per_class_recall.csv"
    pd.DataFrame(recall_rows).to_csv(recalls_path, index=False)
    if champion:
        y_true = stage1_valid.model.astype(str).to_numpy()
        best_prob = pd.read_parquet(best["prediction_path"]).sort_values("id", kind="stable")
        best_prob = best_prob[[f"prob::{label}" for label in classes]].to_numpy(dtype=float)
        if not np.allclose(best_prob.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("best checkpoint probabilities are not normalized")
        champion_metrics = _metrics(y_true, np.log(np.clip(champion["probabilities"], 1e-12, 1)),
                                    champion["probabilities"], classes)
        blend = (champion["probabilities"] + best_prob) / 2.0
        blend /= blend.sum(axis=1, keepdims=True)
        blend_metrics = _metrics(y_true, np.log(np.clip(blend, 1e-12, 1)), blend, classes)
        summary["champion_comparison"] = {
            "champion_version": champion["version"],
            "prediction_experiment_id": champion["prediction_experiment_id"],
            "strategy": champion["strategy"], "member_ids": champion["member_ids"],
            "champion_metrics": champion_metrics, "blend_metrics": blend_metrics,
            "blend_rule": "50:50 mean of saved champion probability vector and best checkpoint probabilities",
            "selected_checkpoint_id": best["experiment_id"],
            "validation_rows": len(y_true), "active_champion_modified": False,
        }

    summary_path = report_dir / "summary.json"
    summary_for_json = json.loads(json.dumps(summary, default=str))
    summary_path.write_text(json.dumps(summary_for_json, indent=2, allow_nan=False) + "\n")
    _render_report(report_dir / "report.md", summary_for_json)
    return {"run_id": run_id, "report_path": str(report_dir / "report.md"),
            "summary_path": str(summary_path), "registry_path": str(registry_path),
            "experiments": {code: {key: summary["experiments"][code][key]["experiment_id"]
                                   for key in ("stage1", "stage2")} for code in ("A", "B", "C")},
            "c_removed_groups": removed_groups, "best_checkpoint": summary["best_checkpoint"],
            "champion_comparison": summary.get("champion_comparison"),
            "active_champion_modified": False}
