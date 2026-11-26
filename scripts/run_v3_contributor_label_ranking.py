#!/usr/bin/env python3
"""Train the V3 baseline models and rank V3.0 contributor-label subsets."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import e13_reliability_router as e13
from src.pipeline import MODEL_CONFIGS, train_models
from src.trust_list2_weighted_vote import _model_family

RUN_ID = "V3.0-QUALITY-RANKING"
MODELS = ("svc", "catboost_stylometry", "ultrabert380")
BASELINE = ROOT / "data/processed/dataset_v3.0_baseline.parquet"
BASE_PROFILE = ROOT / "data/processed/dataset_profile_v3.0_baseline.json"
V3 = ROOT / "data/processed/dataset_v3.0.parquet"
V0 = ROOT / "data/processed/dataset_v0.parquet"
V0_EXT = ROOT / "data/processed/dataset_v0_ext.parquet"
OUT = ROOT / "artifacts/analysis/v3/contributor_label_ranking"
CSV = ROOT / "artifacts/analysis/v3/contributor_label_ranking.csv"
REPORT = ROOT / "docs/v3.0-quality-ranking.md"
CATALOG = ROOT / "configs/ensemble_catalog_v2.json"
MEASURED_INFERENCE_PROFILE = ROOT / "artifacts/analysis/e20d_runtime_profile.json"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_sha(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return file_sha(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(file_sha(child)))
    return h.hexdigest()


def update_catalog(stage: str, status: str = "Running", **fields) -> None:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next(row for row in data["experiments"] if row.get("id") == RUN_ID)
    entry.update({"status": status, "current_stage": stage,
                  "progress_source": str((OUT / "progress.json").relative_to(ROOT)), **fields})
    temporary = CATALOG.with_suffix(".v3-quality.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, CATALOG)
    progress = {"experiment_id": RUN_ID, "status": status, "current_stage": stage,
                "updated_at": pd.Timestamp.now(tz="UTC").isoformat(), **fields}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


def _canonical_frame(path: Path, name: str, classes: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"id", "prompt_id", "text", "model"}
    if required - set(frame):
        raise ValueError(f"{path} missing columns {sorted(required - set(frame))}")
    frame["id"] = frame.id.astype(str)
    frame["prompt_id"] = frame.prompt_id.astype(str)
    frame["model"] = frame.model.astype(str)
    # V0 predates the source_dataset field. Preserve it where present and use
    # the registered dataset name as the explicit source for older snapshots.
    if "source_dataset" not in frame:
        frame["source_dataset"] = name
    frame["source_dataset"] = frame.source_dataset.astype(str)
    if frame.id.duplicated().any() or frame.model.isna().any():
        raise ValueError(f"invalid row IDs or labels in {path}")
    if set(frame.model) != set(classes):
        raise ValueError(f"{name} does not contain all canonical classes")
    frame["_dataset"] = name
    return frame


def _manifest_checkpoint(manifest: dict) -> Path:
    for key in ("best_checkpoint_path", "selected_checkpoint_path", "checkpoint_path"):
        value = manifest.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                return path
    raise FileNotFoundError(f"no saved checkpoint path in {manifest.get('experiment_id')}")


def _probability_frame(path: str | Path, expected: pd.DataFrame, classes: list[str]):
    frame = pd.read_parquet(path)
    required = {"id", "prompt_id", "y_true", *(f"prob::{label}" for label in classes)}
    if required - set(frame):
        raise ValueError(f"cached prediction file lacks columns: {sorted(required - set(frame))}")
    if frame.id.astype(str).duplicated().any():
        raise ValueError("cached prediction IDs are not unique")
    aligned = expected[["id", "prompt_id", "model"]].merge(
        frame[["id", "prompt_id", "y_true", *(f"prob::{label}" for label in classes)]],
        left_on="id", right_on="id", how="left", validate="one_to_one", sort=False)
    if aligned.prompt_id_x.astype(str).tolist() != expected.prompt_id.astype(str).tolist():
        raise ValueError("cached prediction prompt IDs do not align")
    if not np.array_equal(aligned.y_true.astype(str).to_numpy(), expected.model.astype(str).to_numpy()):
        raise ValueError("cached prediction labels do not align")
    probabilities = aligned[[f"prob::{label}" for label in classes]].to_numpy(dtype=np.float64)
    if probabilities.shape != (len(expected), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError("cached probability matrix has invalid shape or non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("cached prediction probabilities are not normalized")
    return probabilities


def _infer(manifest: dict, frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
    checkpoint = _manifest_checkpoint(manifest)
    model = manifest["model"]
    config = manifest.get("config", {})
    profile = json.loads(MEASURED_INFERENCE_PROFILE.read_text(encoding="utf-8"))
    measured_batches = profile.get("measured_batch_defaults", {})
    inference_batch_size = int(measured_batches.get(
        model, config.get("inference_batch_size", 8)))
    texts = frame.text.astype(str).tolist()
    if model == "svc":
        import joblib
        bundle = joblib.load(checkpoint)
        probabilities, found_classes = predict_svc_bundle(bundle, texts)
    elif model == "catboost_stylometry":
        probabilities, found_classes = predict_catboost_stylometry(
            checkpoint, texts, classes, temperature=manifest.get("calibration_temperature", 1.0))
    else:
        probabilities, found_classes = _predict_neural(
            model, checkpoint, texts, classes,
            batch_size=inference_batch_size,
            max_length=int(config.get("max_length", 384)),
            temperature=float(manifest.get("calibration_temperature") or 1.0),
            base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision"),
            fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
            text_only_backbone=bool(config.get("text_only_backbone", False)),
            local_base_model=config.get("local_base_model"),
            attention_backend=config.get("attention_backend", manifest.get("attention_backend")),
            code_revision=config.get("code_revision"))
    if list(found_classes) != classes:
        reindex = {name: i for i, name in enumerate(found_classes)}
        probabilities = probabilities[:, [reindex[name] for name in classes]]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.shape != (len(frame), len(classes)) or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError(f"invalid predictions from {manifest['experiment_id']} on {len(frame)} rows")
    return probabilities


def _equal_dataset_weights(dataset_names: np.ndarray) -> np.ndarray:
    names = np.asarray(dataset_names, dtype=str)
    unique = sorted(set(names))
    if len(unique) != 2:
        raise ValueError("router fitting requires exactly V0 and V0-EXT")
    weight = np.zeros(len(names), dtype=np.float64)
    for name in unique:
        mask = names == name
        weight[mask] = len(names) / (len(unique) * int(mask.sum()))
    return weight


def _fit_equal_weight_router(train_rows: pd.DataFrame, probabilities: np.ndarray,
                             classes: list[str]):
    model_ids = list(MODELS)
    families = [_model_family(name) for name in model_ids]
    truth = pd.Categorical(train_rows.model.astype(str), categories=classes).codes.astype(np.int64)
    if np.any(truth < 0):
        raise ValueError("router fit labels are outside canonical class order")
    dataset_names = train_rows._dataset.astype(str).to_numpy()
    groups = train_rows.prompt_id.astype(str).to_numpy()
    global_weights = _equal_dataset_weights(dataset_names)
    feature_frames, targets, target_weights, fold_meta = [], [], [], []
    splitter = GroupKFold(n_splits=4)
    for fold, (fit, held) in enumerate(splitter.split(train_rows, truth, groups), 1):
        fit_weights = _equal_dataset_weights(dataset_names[fit])
        features, _ = e13.build_feature_frame(
            probabilities, model_ids, families, truth[fit], fit, held,
            fit_sample_weights=fit_weights)
        hard = probabilities[:, held].argmax(axis=2)
        target = e13._targets(hard, truth[held])
        feature_frames.append(features)
        targets.append(target)
        target_weights.append(np.repeat(global_weights[held], len(model_ids)))
        fold_meta.append({"fold": fold, "fit_rows": int(len(fit)), "held_rows": int(len(held)),
            "fit_prompt_groups": int(pd.Series(groups[fit]).nunique()),
            "held_prompt_groups": int(pd.Series(groups[held]).nunique()),
            "fit_v0_rows": int(np.sum(dataset_names[fit] == "v0")),
            "fit_v0_ext_rows": int(np.sum(dataset_names[fit] == "v0_ext")),
            "held_v0_rows": int(np.sum(dataset_names[held] == "v0")),
            "held_v0_ext_rows": int(np.sum(dataset_names[held] == "v0_ext"))})
    X = pd.concat(feature_frames, ignore_index=True)
    y = np.concatenate(targets)
    sample_weight = np.concatenate(target_weights)
    router = e13._new_catboost(iterations=e13.CATBOOST_PARAMS["iterations"])
    router.fit(e13._cat_frame(X), y, sample_weight=sample_weight,
               cat_features=list(e13.CATEGORICAL), verbose=False)
    metadata = {"model_ids": model_ids, "families": families,
        "fit_method": "E13-B 4-fold prompt-group out-of-fold correctness-router training",
        "router_features": list(e13.FEATURE_COLUMNS), "categorical_features": list(e13.CATEGORICAL),
        "equal_dataset_importance": {"v0": 0.5, "v0_ext": 0.5},
        "row_weight_rule": "within each V0/V0-EXT fit or target partition, each dataset receives equal total sample weight",
        "fit_rows": int(len(train_rows)), "oof_records": int(len(X)),
        "oof_positive_fraction_unweighted": float(y.mean()),
        "oof_positive_fraction_weighted": float(np.average(y, weights=sample_weight)),
        "catboost_params": dict(e13.CATBOOST_PARAMS), "folds": fold_meta}
    return router, metadata, global_weights


def _render_report(table: pd.DataFrame, summary: dict) -> None:
    lines = ["# V3.0 contributor-label quality ranking", "",
        "Ranked processing queue for the V3 training factory. Three base classifiers were trained on V3.0-BASELINE; the E13-B-style family-max reliability router was fit using V0 and V0-EXT with equal total dataset weight.", "",
        "## Run summary", "", "| Item | Value |", "|---|---|",
        f"| Baseline training rows | {summary['baseline_training_rows']:,} |",
        f"| Router rows | {summary['router_rows']:,} ({summary['router_dataset_importance']['v0']:.0%} V0 + {summary['router_dataset_importance']['v0_ext']:.0%} V0-EXT total weight) |",
        f"| V3.0 rows available after baseline ID exclusion | {summary['available_v3_rows']:,} |",
        f"| V3.0 rows evaluated after baseline prompt-group exclusion | {summary['evaluated_v3_rows']:,} |",
        f"| Contributor-label combinations | {len(table):,} |",
        f"| Baseline contributors / prompt groups | {summary['baseline_contributors']} / {summary['baseline_prompt_groups']:,} |", "",
        "Base model manifests and checkpoints are retained under `artifacts/runs/` and summarized in the analysis run manifest. The router is saved as [`router.cbm`](../artifacts/analysis/v3/contributor_label_ranking/router.cbm). V3.0 quality assessment excludes both baseline response IDs and every V3.0 row in a baseline prompt group.", "",
        "## Ranked contributor-label processing queue", "",
        "Relative accuracy is subset accuracy minus the overall ensemble accuracy for the same true label, in percentage points. Available rows count V3.0 responses after baseline-ID removal; evaluated rows additionally require that their prompt group is not in V3.0-BASELINE.", "",
        "| Rank | Contributor | Label | Available rows | Evaluated rows | Accuracy | Label average | Relative accuracy |",
        "|---:|---|---|---:|---:|---:|---:|---:|"]
    for _, row in table.iterrows():
        def pct(value):
            return "Unavailable" if pd.isna(value) else f"{float(value):.2%}"
        relative = row["Relative accuracy"]
        rel = "Unavailable" if pd.isna(relative) else f"{float(relative):+.2f} pp"
        lines.append(f"| {int(row['Rank'])} | {row['Contributor']} | {row['Label']} | {int(row['Available rows']):,} | {int(row['Evaluated rows']):,} | {pct(row['Accuracy'])} | {pct(row['Label average'])} | {rel} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not all(path.is_file() for path in (BASELINE, BASE_PROFILE, V3, V0, V0_EXT)):
        raise FileNotFoundError("V3.0 baseline, parent dataset, V0 or V0-EXT artifact is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_profile = json.loads(BASE_PROFILE.read_text(encoding="utf-8"))
    baseline_hash = file_sha(BASELINE)
    if baseline_hash != baseline_profile.get("dataset_sha256"):
        raise ValueError("V3.0-BASELINE does not match its immutable profile")
    v3 = pd.read_parquet(V3).reset_index(drop=True)
    baseline = pd.read_parquet(BASELINE).reset_index(drop=True)
    if file_sha(V3) != baseline_profile.get("parent_dataset_sha256"):
        raise ValueError("V3.0 changed after V3.0-BASELINE creation")
    if baseline.id.duplicated().any() or baseline.id.nunique() != 1000:
        raise ValueError("V3.0-BASELINE must contain exactly 1,000 unique rows")
    classes = sorted(pd.read_parquet(V0).model.astype(str).unique())
    if set(v3.model.astype(str)) != set(classes) or set(baseline.model.astype(str)) != set(classes):
        raise ValueError("canonical label roster mismatch across baseline, V3.0 and V0")

    update_catalog("Baseline classifiers training", "Running")
    model_manifests = {}
    for model_name in MODELS:
        update_catalog(f"Training {model_name}", "Running",
                      completed_models=list(model_manifests))
        print(f"TRAINING {model_name} on all {len(baseline)} v3.0-baseline rows", flush=True)
        overrides = ({"epochs": 20, "early_stopping_patience": 2}
                     if model_name == "ultrabert380" else None)
        manifest = train_models(dataset="v3.0-baseline", model=model_name,
            data_dir=str(ROOT / "data/processed"), artifact_dir=str(ROOT / "artifacts"),
            registry_path=str(ROOT / "experiments/registry.sqlite"),
            config_path=str(ROOT / MODEL_CONFIGS[model_name]), seed=2026,
            validation_dataset="v0", config_overrides=overrides)
        model_manifests[model_name] = manifest
        (OUT / "model_manifests.json").write_text(
            json.dumps({k: v.get("experiment_id") for k, v in model_manifests.items()}, indent=2) + "\n")
        update_catalog(f"Completed {model_name}; next baseline classifier",
                      "Running", model_experiments={k: v.get("experiment_id") for k, v in model_manifests.items()})

    v0 = _canonical_frame(V0, "v0", classes).sort_values("id", kind="stable").reset_index(drop=True)
    v0_ext = _canonical_frame(V0_EXT, "v0_ext", classes).reset_index(drop=True)
    router_rows = pd.concat([v0, v0_ext], ignore_index=True)
    if set(v0.prompt_id.astype(str)) & set(v0_ext.prompt_id.astype(str)):
        raise ValueError("V0 and V0-EXT contain overlapping router-fit prompt groups")
    for name, frame in (("v0", v0), ("v0_ext", v0_ext)):
        update_catalog(f"Collecting {name} classifier probabilities", "Running")
        for model_name, manifest in model_manifests.items():
            if name == "v0" and manifest.get("prediction_path"):
                try:
                    probabilities = _probability_frame(manifest["prediction_path"], frame, classes)
                except (OSError, ValueError, KeyError):
                    probabilities = _infer(manifest, frame, classes)
            else:
                probabilities = _infer(manifest, frame, classes)
            model_manifests[model_name].setdefault("evaluation_predictions", {})[name] = probabilities
    p_router = np.stack([
        np.concatenate([model_manifests[name]["evaluation_predictions"]["v0"],
                        model_manifests[name]["evaluation_predictions"]["v0_ext"]], axis=0)
        for name in MODELS])
    router, router_meta, router_weights = _fit_equal_weight_router(router_rows, p_router, classes)
    router_path = OUT / "router.cbm"
    router.save_model(str(router_path))
    (OUT / "router_manifest.json").write_text(json.dumps({**router_meta,
        "router_sha256": file_sha(router_path), "router_fit_dataset_sha256": {
            "v0": file_sha(V0), "v0_ext": file_sha(V0_EXT)},
        "model_experiments": {name: model_manifests[name].get("experiment_id") for name in MODELS}},
        indent=2, sort_keys=True) + "\n")

    baseline_ids = set(baseline.id.astype(str))
    baseline_groups = set(baseline.prompt_id.astype(str))
    available = v3.loc[~v3.id.astype(str).isin(baseline_ids)].copy().reset_index(drop=True)
    quality = available.loc[~available.prompt_id.astype(str).isin(baseline_groups)].copy().reset_index(drop=True)
    if not len(quality):
        raise ValueError("no V3.0 responses remain for quality evaluation")
    update_catalog("Inferring ensemble on V3.0 outside baseline IDs and prompt groups", "Running",
                  available_v3_rows=len(available), evaluated_v3_rows=len(quality))
    p_quality = []
    for model_name, manifest in model_manifests.items():
        print(f"INFERENCE {model_name} on {len(quality)} eligible V3.0 rows", flush=True)
        p_quality.append(_infer(manifest, quality, classes))
    p_quality = np.stack(p_quality)
    # Reliability features for deployment use all router-fit labels, equally weighted by dataset.
    combined_for_features = pd.concat([router_rows, quality], ignore_index=True)
    target_rows = np.arange(len(router_rows), len(combined_for_features), dtype=np.int64)
    fit_rows = np.arange(len(router_rows), dtype=np.int64)
    y_router = pd.Categorical(router_rows.model.astype(str), categories=classes).codes.astype(np.int64)
    p_combined = np.concatenate([p_router, p_quality], axis=1)
    deploy_weights = _equal_dataset_weights(router_rows._dataset.astype(str).to_numpy())
    X_target, _ = e13.build_feature_frame(p_combined, list(MODELS),
        [_model_family(name) for name in MODELS], y_router, fit_rows, target_rows,
        fit_sample_weights=deploy_weights)
    reliability = e13._score_correctness(router, X_target).reshape(len(quality), len(MODELS)).T
    hard = p_quality.argmax(axis=2)
    family_probabilities = e13.family_balanced_probabilities(p_quality,
        [_model_family(name) for name in MODELS])
    ensemble_pred, _ = e13.route_predictions("family_max", hard, reliability,
        family_probabilities, [_model_family(name) for name in MODELS])
    truth = pd.Categorical(quality.model.astype(str), categories=classes).codes
    quality_correct = ensemble_pred == truth
    if np.any(truth < 0):
        raise ValueError("quality-evaluation labels are outside canonical order")

    # Save row-aligned predictions so rankings can be audited and recreated without inference.
    predictions = quality[["id", "prompt_id", "source_dataset", "model"]].rename(
        columns={"model": "y_true"}).copy()
    predictions["ensemble_prediction"] = np.asarray(classes, dtype=object)[ensemble_pred]
    predictions["ensemble_correct"] = quality_correct
    for i, name in enumerate(MODELS):
        predictions[f"prediction::{name}"] = np.asarray(classes, dtype=object)[hard[i]]
        for j, label in enumerate(classes):
            predictions[f"prob::{name}::{label}"] = p_quality[i, :, j]
    predictions.to_parquet(OUT / "v3_quality_predictions.parquet", index=False, compression="zstd")
    v0pred = router_rows[["id", "prompt_id", "_dataset", "model"]].rename(columns={"model":"y_true"}).copy()
    for i, name in enumerate(MODELS):
        for j, label in enumerate(classes):
            v0pred[f"prob::{name}::{label}"] = p_router[i, :, j]
    v0pred.to_parquet(OUT / "router_fit_predictions.parquet", index=False, compression="zstd")

    available_counts = available.groupby(["source_dataset", "model"]).size()
    evaluation = quality[["source_dataset", "model"]].copy()
    evaluation["correct"] = quality_correct
    evaluation_stats = evaluation.groupby(["source_dataset", "model"]).agg(
        evaluated_rows=("correct", "size"), correct_rows=("correct", "sum"))
    label_average = evaluation.groupby("model").correct.mean().to_dict()
    records = []
    for (contributor, label), available_rows in available_counts.items():
        if (contributor, label) in evaluation_stats.index:
            stat = evaluation_stats.loc[(contributor, label)]
            evaluated_rows = int(stat.evaluated_rows)
            accuracy = float(stat.correct_rows / evaluated_rows) if evaluated_rows else np.nan
        else:
            evaluated_rows, accuracy = 0, np.nan
        average = float(label_average[label]) if label in label_average else np.nan
        relative = (accuracy - average) * 100.0 if evaluated_rows and np.isfinite(average) else np.nan
        records.append({"Contributor": str(contributor), "Label": str(label),
            "Available rows": int(available_rows), "Evaluated rows": evaluated_rows,
            "Accuracy": accuracy, "Label average": average, "Relative accuracy": relative})
    table = pd.DataFrame.from_records(records)
    table["_has_quality"] = table["Relative accuracy"].notna()
    table.sort_values(["_has_quality", "Relative accuracy", "Evaluated rows", "Contributor", "Label"],
        ascending=[False, False, False, True, True], kind="stable", inplace=True)
    table.drop(columns="_has_quality", inplace=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1, dtype=int))
    table.to_csv(CSV, index=False, na_rep="")

    summary = {"experiment_id": RUN_ID, "status": "Complete",
        "baseline_training_dataset": "v3.0-baseline", "baseline_training_rows": len(baseline),
        "baseline_dataset_sha256": baseline_hash, "parent_v3_dataset_sha256": file_sha(V3),
        "models": {name: {"experiment_id": model_manifests[name].get("experiment_id"),
            "config": model_manifests[name].get("config"),
            "checkpoint": model_manifests[name].get("best_checkpoint_path") or
                         model_manifests[name].get("selected_checkpoint_path") or
                         model_manifests[name].get("checkpoint_path"),
            "checkpoint_sha256": path_sha(_manifest_checkpoint(model_manifests[name])),
            "validation_dataset": model_manifests[name].get("validation_dataset"),
            "validation_metrics": model_manifests[name].get("validation_metrics")}
            for name in MODELS},
        "router_path": str(router_path.relative_to(ROOT)), "router_sha256": file_sha(router_path),
        "router_dataset_importance": {"v0":0.5,"v0_ext":0.5},
        "router_rows": len(router_rows), "router_weights": {
            name: float(router_weights[router_rows._dataset.astype(str).to_numpy() == name].sum())
            for name in ("v0", "v0_ext")},
        "available_v3_rows": len(available), "evaluated_v3_rows": len(quality),
        "excluded_baseline_ids": len(baseline_ids), "excluded_baseline_prompt_groups": len(baseline_groups),
        "baseline_contributors": int(baseline.source_dataset.nunique()),
        "baseline_prompt_groups": int(baseline.prompt_id.nunique()),
        "contributor_label_combinations": len(table), "predictions_sha256": file_sha(OUT / "v3_quality_predictions.parquet"),
        "ranking_csv": str(CSV.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT)),
        "ranking_rule": "relative accuracy descending, evaluated rows descending, contributor then label ascending; unavailable subsets last"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _render_report(table, summary)
    update_catalog("Complete", "Scored", result_source=str((OUT / "summary.json").relative_to(ROOT)),
        artifact_dir=str(OUT.relative_to(ROOT)), processing_queue=str(CSV.relative_to(ROOT)),
        documentation=str(REPORT.relative_to(ROOT)), evaluated_rows=len(quality),
        contributor_label_combinations=len(table), router_sha256=summary["router_sha256"])
    print(json.dumps({"complete":True,"rows_ranked":len(table),"available_v3_rows":len(available),
        "evaluated_v3_rows":len(quality),"csv":str(CSV),"report":str(REPORT),
        "summary":str(OUT/"summary.json")},indent=2),flush=True)


if __name__ == "__main__":
    main()
