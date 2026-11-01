"""CLI orchestration for immutable V0 data, model, ensemble, and export runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import balanced_accuracy_score

from .data_factory import prepare
from .evaluation import (align_predictions, complementarity, evaluate, paired_group_bootstrap)
from .ensemble_factory import cross_fitted_weights, ensemble_probabilities
from .feature_factory import load_or_fit
from .model_factory import train_encoder_fold, train_qwen_lora_fold, train_svc
from .registry import Registry

MODEL_CONFIGS = {"svc": "configs/models/svc.yaml", "ultrabert150": "configs/models/ultrabert150.yaml", "qwen06": "configs/models/qwen06.yaml"}


def _revision():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: return "unknown"


def _read_yaml(path):
    with open(path, encoding="utf-8") as stream: return yaml.safe_load(stream) or {}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def prepare_data(config):
    cfg = _read_yaml(config)
    return prepare(cfg["train_path"], cfg["output_dir"], cfg.get("test_path"), seed=cfg.get("seed", 2026),
                   holdout_fraction=cfg.get("holdout_fraction", .2), n_splits=cfg.get("n_splits", 5),
                   expected_classes=cfg.get("expected_classes", 18))


def _load_dataset(dataset, data_dir):
    data_dir = Path(data_dir)
    rows = pd.read_parquet(data_dir / f"dataset_{dataset}.parquet")
    assignments = pd.read_parquet(data_dir / f"folds_{dataset}.parquet")
    if rows.id.duplicated().any() or assignments.id.duplicated().any(): raise ValueError("dataset and fold assignment IDs must be unique")
    if set(rows.id.astype(str)) != set(assignments.id.astype(str)): raise ValueError("dataset and fold assignment ID sets differ")
    rows = rows.merge(assignments[["id", "split", "fold"]], on="id", how="left", validate="one_to_one")
    if rows["fold"].isna().any() or rows.loc[rows.split.eq("development"), "fold"].lt(0).any(): raise ValueError("development rows have missing fold assignments")
    return rows, data_dir


def _model_cfg(model, config_path=None):
    path = Path(config_path or MODEL_CONFIGS[model])
    if not path.exists(): raise FileNotFoundError(f"model config not found: {path}")
    cfg = _read_yaml(path)
    cfg.setdefault("model", model)
    if cfg["model"] != model: raise ValueError(f"model config {path} declares {cfg['model']!r}, expected {model!r}")
    cfg["_config_path"] = str(path)
    if model != "svc" and (not cfg.get("checkpoint") or not cfg.get("revision")):
        raise ValueError(f"{model} needs an explicit checkpoint and immutable revision in {path}")
    if model != "svc" and not re.fullmatch(r"[0-9a-fA-F]{40}", str(cfg["revision"])):
        raise ValueError(f"{model} revision in {path} must be a full 40-character commit hash, not a branch or tag")
    if model == "svc" and (float(cfg.get("C", .5)) <= 0 or int(cfg.get("calibration_cv", 3)) < 2):
        raise ValueError(f"SVC config {path} needs C > 0 and calibration_cv >= 2")
    if model != "svc" and (int(cfg.get("epochs", 1)) < 1 or int(cfg.get("batch_size", 1)) < 1 or int(cfg.get("max_length", 1)) < 1):
        raise ValueError(f"model config {path} needs positive epochs, batch_size, and max_length")
    return cfg


def _train_fold(model, train, valid, classes, cfg, output_dir, seed, fold, cache_dir, *, final=False, cache_fold=None):
    if model == "svc":
        result = train_svc(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
                           c=cfg.get("C", .5), calibration_cv=cfg.get("calibration_cv", 3), seed=seed,
                           feature_config=cfg.get("features", {"views": ["char", "word"]}),
                           cache_dir=cache_dir, fold_token=cache_fold or f"fold-{fold}")
        checkpoint = Path(output_dir) / "checkpoints" / f"svc_fold{fold}.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": result["model"], "raw_model": result["raw_model"], "features": result["features"], "classes": classes, "config": cfg}, checkpoint)
        result["checkpoint"] = str(checkpoint)
    elif model == "ultrabert150":
        checkpoint = Path(output_dir) / "checkpoints" / f"ultrabert150_fold{fold}"
        result = train_encoder_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 3), batch_size=cfg.get("batch_size", 16), max_length=cfg.get("max_length", 384),
            learning_rate=cfg.get("learning_rate", 2e-5), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1))
        result["checkpoint"] = str(checkpoint)
    elif model == "qwen06":
        checkpoint = Path(output_dir) / "checkpoints" / f"qwen06_fold{fold}"
        result = train_qwen_lora_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 3), batch_size=cfg.get("batch_size", 4), max_length=cfg.get("max_length", 384),
            learning_rate=cfg.get("learning_rate", 1e-4), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            lora_rank=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32), save_merged=final)
        result["checkpoint"] = str(checkpoint)
    else: raise ValueError(f"unknown model: {model}")
    return result


def build_features(dataset="v0", data_dir="data/processed", config_path="configs/features/default.yaml", cache_dir="artifacts/features"):
    rows, _ = _load_dataset(dataset, data_dir)
    cfg = _read_yaml(config_path)
    dev = rows[rows.split.eq("development")]
    report = []
    for fold in sorted(dev.fold.unique()):
        valid = dev.fold.eq(fold)
        tr_text, va_text = dev.loc[~valid, "text"], dev.loc[valid, "text"]
        x_train, x_val, features, key = load_or_fit(tr_text, va_text, config=cfg, cache_dir=cache_dir, fold_token=f"{dataset}-fold-{fold}")
        report.append({"fold": int(fold), "cache_key": key, "train_shape": list(x_train.shape), "validation_shape": list(x_val.shape)})
    output = Path(cache_dir) / f"{dataset}_features_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": dataset, "feature_config": cfg, "folds": report}, indent=2) + "\n")
    return report


def train_models(dataset="v0", model="svc", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, config_path=None, feature_config_path=None, seed=2026, folds=None):
    cfg = _model_cfg(model, config_path)
    if model == "svc" and feature_config_path:
        cfg["features"] = _read_yaml(feature_config_path)
    rows, data_dir = _load_dataset(dataset, data_dir)
    profile = json.loads((data_dir / "dataset_profile.json").read_text())
    code_revision = _revision()
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config_hash = _sha(config_snapshot)
    experiment_id = f"{model}_{dataset}_data{profile['dataset_sha256'][:8]}_split{profile['split_sha256'][:8]}_seed{seed}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    run_artifact_dir = Path(artifact_dir) / "runs" / experiment_id
    pred_dir = Path(artifact_dir) / "predictions"
    prediction_path = pred_dir / f"{experiment_id}_oof.parquet"
    manifest_path = pred_dir / f"{experiment_id}_manifest.json"
    if manifest_path.exists() and prediction_path.exists(): return json.loads(manifest_path.read_text())
    if run_artifact_dir.exists(): raise FileExistsError(f"incomplete immutable experiment exists at {run_artifact_dir}; remove that generated run directory to retry")
    classes = sorted(rows.model.astype(str).unique().tolist())
    dev = rows[rows.split.eq("development")].reset_index(drop=True)
    fold_ids = sorted(int(x) for x in dev.fold.unique())
    if folds is not None: fold_ids = [x for x in fold_ids if x in folds]
    if len(fold_ids) != dev.fold.nunique(): raise ValueError("partial fold training cannot be published as complete OOF")
    oof = np.full((len(dev), len(classes)), np.nan); probabilities = oof.copy()
    timings, runtimes, checkpoints, cache_keys, gpu_memory, artifact_sizes, temperatures = [], [], [], [], [], [], []
    feature_cache = Path(artifact_dir) / "features"
    for fold in fold_ids:
        valid = dev.fold.eq(fold).to_numpy()
        train_rows, val_rows = dev.loc[~valid], dev.loc[valid]
        result = _train_fold(model, train_rows, val_rows, classes, cfg, run_artifact_dir, seed + fold, fold, feature_cache,
                             cache_fold=f"{dataset}-fold-{fold}")
        oof[valid] = result["scores"]; probabilities[valid] = result["probabilities"]
        timings.append(float(result["fit_seconds"])); runtimes.append(result.get("inference_seconds")); checkpoints.append(result["checkpoint"])
        gpu_memory.append(result.get("peak_gpu_memory_bytes"))
        temperatures.append(result.get("calibration_temperature"))
        ckpt_path = Path(result["checkpoint"])
        artifact_sizes.append(sum(f.stat().st_size for f in ckpt_path.rglob("*") if f.is_file()) if ckpt_path.is_dir() else ckpt_path.stat().st_size)
        cache_keys.append(result.get("cache_key"))
    if not np.isfinite(oof).all() or not np.isfinite(probabilities).all(): raise RuntimeError("OOF output is incomplete or non-finite")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5): raise RuntimeError("OOF probability rows do not sum to one")
    metrics = evaluate(dev.model.astype(str), oof, classes, probabilities)
    pred = dev[["id", "prompt_id", "model", "fold"]].rename(columns={"model": "y_true"}).copy()
    for i, label in enumerate(classes):
        pred[f"score::{label}"] = oof[:, i]; pred[f"prob::{label}"] = probabilities[:, i]
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(prediction_path, index=False)
    manifest = {"experiment_id": experiment_id, "model": model, "dataset": dataset,
        "split": "development_oof", "classes": classes, "row_count": len(dev),
        "id_sha256": hashlib.sha256("\n".join(dev.id.astype(str)).encode()).hexdigest(),
        "dataset_sha256": profile["dataset_sha256"], "split_sha256": profile["split_sha256"],
        "code_revision": code_revision, "model_revision": cfg.get("revision"), "checkpoint_id": cfg.get("checkpoint"),
        "config": config_snapshot, "config_sha256": config_hash,
        "seed": seed, "folds": fold_ids, "fit_seconds_by_fold": timings,
        "fit_seconds": sum(timings), "inference_seconds_by_fold": runtimes,
        "inference_rows_per_second_by_fold": [len(dev.loc[dev.fold.eq(f)]) / seconds if seconds and seconds > 0 else None for f, seconds in zip(fold_ids, runtimes)],
        "peak_gpu_memory_bytes_by_fold": gpu_memory, "checkpoint_size_bytes_by_fold": artifact_sizes,
        "calibration_method": "sigmoid train-only CV" if model == "svc" else "train-only scalar temperature",
        "calibration_temperature_by_fold": temperatures,
        "feature_cache_keys": cache_keys,
        "checkpoint_paths": checkpoints, "prediction_path": str(prediction_path),
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "metrics": metrics, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    Registry(registry_path).add(manifest["experiment_id"], model, "development_oof", manifest["config"], metrics,
        dataset_hash=manifest["dataset_sha256"], code_revision=manifest["code_revision"], seed=seed,
        fold_id=None, runtime_seconds=manifest["fit_seconds"], artifact_path=str(prediction_path), config_hash=manifest["config_sha256"],
        split_hash=manifest["split_sha256"], model_revision=manifest["model_revision"])
    (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_oof(dataset, artifact_dir):
    files = sorted(p for p in (Path(artifact_dir) / "predictions").glob("*_oof.parquet")
                   if f"_{dataset}_" in p.name and not p.name.startswith("ensemble_"))
    if not files: raise FileNotFoundError(f"no OOF prediction files found for dataset {dataset}")
    frames = {p.name.removesuffix("_oof.parquet"): pd.read_parquet(p) for p in files}
    first = next(iter(frames.values())); classes = sorted(c.removeprefix("prob::") for c in first if c.startswith("prob::"))
    aligned = align_predictions({name: frame.rename(columns={f"prob::{c}": c for c in classes}) for name, frame in frames.items()}, classes)
    manifests = {}
    for name in frames:
        manifest_path = Path(artifact_dir)/"predictions"/f"{name}_manifest.json"
        manifests[name] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dataset_hashes = {m.get("dataset_sha256") for m in manifests.values() if m.get("dataset_sha256")}
    split_hashes = {m.get("split_sha256") for m in manifests.values() if m.get("split_sha256")}
    if len(dataset_hashes) > 1 or len(split_hashes) > 1: raise ValueError("OOF artifacts were produced from different dataset or split hashes")
    for name, manifest in manifests.items():
        if manifest.get("classes") and manifest["classes"] != classes: raise ValueError(f"OOF class order differs for {name}")
    base = first.sort_values("id", kind="stable").reset_index(drop=True)
    if not np.array_equal(base.id.astype(str).to_numpy(), aligned.id.astype(str).to_numpy()): raise ValueError("OOF ids do not align")
    for name, frame in frames.items():
        aligned_meta = frame.sort_values("id", kind="stable").reset_index(drop=True)
        for column in ("y_true", "prompt_id", "fold"):
            if column not in aligned_meta or not np.array_equal(base[column].astype(str).to_numpy(), aligned_meta[column].astype(str).to_numpy()):
                raise ValueError(f"OOF metadata mismatch for {name}.{column}")
    y = base.y_true.astype(str).to_numpy(); groups = base.prompt_id.to_numpy(); folds = base.fold.to_numpy()
    probabilities = {name: np.column_stack([aligned[f"{name}::{c}"].to_numpy() for c in classes]) for name in frames}
    return base, classes, y, groups, folds, probabilities, next(iter(dataset_hashes), None), next(iter(split_hashes), None)


def evaluate_ensembles(dataset="v0", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, bootstrap_replicates=2000, seed=2026):
    base, classes, y, groups, folds, models, dataset_hash, split_hash = _load_oof(dataset, artifact_dir)
    names = list(models)
    if len(names) < 2: raise ValueError("ensemble evaluation requires at least two model OOF files")
    reports = {}; predictions = {n: np.asarray(classes)[p.argmax(1)] for n, p in models.items()}
    for size in range(2, len(names) + 1):
        for subset in itertools.combinations(names, size):
            probabilities = ensemble_probabilities([models[n] for n in subset])
            label = "+".join(subset); scores = np.log(np.clip(probabilities, 1e-12, 1))
            metric = evaluate(y, scores, classes, probabilities)
            metric["members"] = list(subset); metric["runtime_seconds"] = None
            predictions[label] = np.asarray(classes)[probabilities.argmax(1)]
            reports[label] = {"metrics": metric, "probabilities": probabilities}
    all_names = names
    weighted, weights = cross_fitted_weights([models[n] for n in all_names], y, folds, classes)
    weighted_name = "weighted:" + "+".join(all_names)
    reports[weighted_name] = {"metrics": evaluate(y, np.log(np.clip(weighted, 1e-12, 1)), classes, weighted), "probabilities": weighted, "weights_by_meta_fold": weights}
    predictions[weighted_name] = np.asarray(classes)[weighted.argmax(1)]
    base_metrics = {name: evaluate(y, np.log(np.clip(p, 1e-12, 1)), classes, p) for name, p in models.items()}
    single_best = max(names, key=lambda n: base_metrics[n]["balanced_accuracy"])
    model_costs = {}
    for name in names:
        manifest_path = Path(artifact_dir)/"predictions"/f"{name}_manifest.json"
        m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        model_costs[name] = {"fit_seconds": m.get("fit_seconds"),
            "inference_seconds_by_fold": m.get("inference_seconds_by_fold"),
            "peak_gpu_memory_bytes_by_fold": m.get("peak_gpu_memory_bytes_by_fold"),
            "checkpoint_size_bytes_by_fold": m.get("checkpoint_size_bytes_by_fold")}
    for label, record in reports.items():
        record["complementarity"] = complementarity(y, {single_best: predictions[single_best], label: predictions[label]}).get(f"{single_best}__{label}")
        record["paired_bootstrap_vs_best_single"] = paired_group_bootstrap(y, predictions[single_best], predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)
        record["per_class_recall_delta_vs_best_single"] = {cls: record["metrics"]["per_class_recall"][cls] - base_metrics[single_best]["per_class_recall"][cls] for cls in classes}
        members = record["metrics"].get("members", names)
        if label.startswith("weighted:"): members = names
        marginal = {}
        if label.startswith("weighted:"):
            equal_p = ensemble_probabilities([models[item] for item in names])
            equal_pred = np.asarray(classes)[equal_p.argmax(1)]
            marginal["vs_equal_weight"] = {"delta_balanced_accuracy": float(record["metrics"]["balanced_accuracy"] - balanced_accuracy_score(y, equal_pred)),
                "paired_bootstrap": paired_group_bootstrap(y, equal_pred, predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)}
        elif len(members) > 1:
            for member in members:
                previous = [item for item in members if item != member]
                previous_p = ensemble_probabilities([models[item] for item in previous])
                previous_pred = np.asarray(classes)[previous_p.argmax(1)]
                marginal[member] = {"delta_balanced_accuracy": float(record["metrics"]["balanced_accuracy"] - balanced_accuracy_score(y, previous_pred)),
                    "paired_bootstrap": paired_group_bootstrap(y, previous_pred, predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)}
        record["marginal_contribution"] = marginal
        record["inference_cost_seconds"] = sum(sum(v["inference_seconds_by_fold"] or []) for n,v in model_costs.items() if n in members)
    pred_dir = Path(artifact_dir) / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    output = base[["id", "prompt_id", "y_true", "fold"]].copy()
    for name, record in reports.items():
        p = record["probabilities"]
        for i, cls in enumerate(classes): output[f"{name}::score::{cls}"] = np.log(np.clip(p[:, i], 1e-12, 1)); output[f"{name}::prob::{cls}"] = p[:, i]
    serializable = {k: {kk: vv for kk, vv in v.items() if kk != "probabilities"} for k, v in reports.items()}
    best_ensemble = max(reports, key=lambda name: reports[name]["metrics"]["balanced_accuracy"])
    best_metrics = reports[best_ensemble]["metrics"]
    code_revision = _revision()
    member_key = _sha({name: str(Path(artifact_dir)/"predictions"/f"{name}_oof.parquet") for name in names})[:10]
    report = {"experiment_id": f"ensemble_{dataset}_seed{seed}_members{member_key}_code{code_revision[:7]}", "dataset": dataset, "split": "development_oof",
        "classes": classes, "members": names, "best_single": single_best, "best_ensemble": best_ensemble,
        "balanced_accuracy": best_metrics["balanced_accuracy"], "macro_f1": best_metrics["macro_f1"],
        "log_loss": best_metrics.get("log_loss"), "per_class_recall": best_metrics["per_class_recall"],
        "confusion_matrix": best_metrics["confusion_matrix"], "model_costs": model_costs, "models": base_metrics,
        "ensembles": serializable, "complementarity_all": complementarity(y, predictions),
        "seed": seed, "code_revision": code_revision, "dataset_sha256": dataset_hash, "split_sha256": split_hash}
    path = pred_dir / f"{report['experiment_id']}_oof.parquet"
    metrics_path=pred_dir / f"{report['experiment_id']}_metrics.json"
    if metrics_path.exists() and path.exists(): return json.loads(metrics_path.read_text())
    output.to_parquet(path, index=False)
    report["prediction_path"] = str(path)
    (pred_dir / f"{report['experiment_id']}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    Registry(registry_path).add(report["experiment_id"], "ensemble", "development_oof", {"members": names, "seed": seed}, report,
        dataset_hash=dataset_hash, split_hash=split_hash, code_revision=report["code_revision"], seed=seed, artifact_path=str(path))
    return report


def evaluate_holdout(model="svc", dataset="v0", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, config_path=None, seed=2026):
    """Evaluate once on the untouched prompt-group holdout; refuse a second run."""
    if model not in MODEL_CONFIGS: raise ValueError(f"unknown model {model}")
    cfg = _model_cfg(model, config_path)
    rows, data_dir = _load_dataset(dataset, data_dir)
    out = Path(artifact_dir) / "predictions"; out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{model}_{dataset}_holdout_metrics.json"
    holdout_guard = out / f"{dataset}_holdout_evaluated.json"
    if holdout_guard.exists() or report_path.exists(): raise FileExistsError(f"untouched holdout already evaluated; refusing to reuse it: {holdout_guard}")
    classes = sorted(rows.model.astype(str).unique()); train = rows[rows.split.eq("development")]; val = rows[rows.split.eq("holdout")]
    if set(train.prompt_id) & set(val.prompt_id): raise RuntimeError("prompt leakage between holdout and development")
    # The complete development set is treated as a single fit split.
    result = _train_fold(model, train, val, classes, cfg, artifact_dir, seed, "holdout", Path(artifact_dir)/"features")
    metrics = evaluate(val.model.astype(str), result["scores"], classes, result["probabilities"])
    report = {"experiment_id": f"{model}_{dataset}_holdout_seed{seed}", "model": model, "split": "holdout",
        "dataset": dataset, "classes": classes, "rows": len(val), "metrics": metrics,
        "fit_seconds": result["fit_seconds"], "inference_seconds": result.get("inference_seconds"),
        "note": "One-time untouched holdout evaluation. Do not use for model or hyperparameter selection."}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    holdout_guard.write_text(json.dumps({"model": model, "experiment_id": report["experiment_id"], "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "report": str(report_path)}, indent=2)+"\n")
    Registry(registry_path).add(report["experiment_id"], model, "holdout", cfg, metrics,
        dataset_hash=json.loads((data_dir/"dataset_profile.json").read_text())["dataset_sha256"],
        split_hash=json.loads((data_dir/"dataset_profile.json").read_text())["split_sha256"],
        model_revision=cfg.get("revision"), code_revision=_revision(), seed=seed, runtime_seconds=result["fit_seconds"], artifact_path=str(report_path))
    return report


def fit_final(model="svc", dataset="v0", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, config_path=None, seed=2026):
    """Fit selected recipe on all labelled official rows for offline inference."""
    cfg = _model_cfg(model, config_path); rows, data_dir = _load_dataset(dataset, data_dir)
    profile = json.loads((data_dir/"dataset_profile.json").read_text())
    config_snapshot = {k:v for k,v in cfg.items() if not k.startswith("_")}
    config_hash = _sha(config_snapshot); code_revision = _revision()
    run_name = f"{model}_{dataset}_final_data{profile['dataset_sha256'][:8]}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    final_manifest_path = Path(artifact_dir,"checkpoints",f"{run_name}.json")
    if final_manifest_path.exists(): return json.loads(final_manifest_path.read_text())
    classes = sorted(rows.model.astype(str).unique()); train = rows.sort_values("id", kind="stable")
    result = _train_fold(model, train, train, classes, cfg, Path(artifact_dir)/"final_runs"/run_name, seed, "final", Path(artifact_dir)/"features", final=True)
    path = Path(result["checkpoint"]) if model != "svc" else Path(artifact_dir)/"checkpoints"/"final"/run_name/"model.joblib"
    if model == "svc":
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({"model": result["model"], "raw_model": result["raw_model"], "features": result["features"], "classes": classes, "config": cfg}, path)
    manifest = {"experiment_id": run_name, "model": model, "dataset": dataset, "classes": classes, "checkpoint": str(path),
        "checkpoint_id": cfg.get("checkpoint"), "model_revision": cfg.get("revision"), "seed": seed,
        "trained_rows": len(train), "dataset_sha256": profile["dataset_sha256"], "config_sha256": config_hash,
        "code_revision": code_revision, "fit_seconds": result["fit_seconds"],
        "calibration_temperature": result.get("calibration_temperature"),
        "calibration_method": result.get("calibration_method", "sigmoid train-only CV" if model == "svc" else "train-only scalar temperature"), "config": cfg}
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoints_dir=Path(artifact_dir,"checkpoints");checkpoints_dir.mkdir(parents=True,exist_ok=True)
    final_manifest_path.write_text(json.dumps(manifest, indent=2, default=str)+"\n")
    (checkpoints_dir/f"{model}_{dataset}_final.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n")
    Registry(registry_path).add(run_name, model, "full_train", config_snapshot,
        {"trained_rows":len(train),"fit_seconds":result["fit_seconds"]},
        dataset_hash=profile["dataset_sha256"], code_revision=code_revision,
        config_hash=config_hash, seed=seed, runtime_seconds=result["fit_seconds"],
        artifact_path=str(final_manifest_path), model_revision=cfg.get("revision"))
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m src.pipeline");sub=parser.add_subparsers(dest="command",required=True)
    d=sub.add_parser("data").add_subparsers(dest="action",required=True);p=d.add_parser("prepare");p.add_argument("--config",default="configs/data/official_v0.yaml")
    f=sub.add_parser("features").add_subparsers(dest="action",required=True);b=f.add_parser("build");b.add_argument("--dataset",default="v0");b.add_argument("--data-dir",default="data/processed");b.add_argument("--config",default="configs/features/default.yaml")
    m=sub.add_parser("models").add_subparsers(dest="action",required=True)
    t=m.add_parser("train");t.add_argument("--model",choices=MODEL_CONFIGS,default="svc");t.add_argument("--dataset",default="v0");t.add_argument("--data-dir",default="data/processed");t.add_argument("--artifact-dir",default="artifacts");t.add_argument("--config");t.add_argument("--features-config");t.add_argument("--seed",type=int,default=2026)
    h=m.add_parser("evaluate-holdout");h.add_argument("--model",choices=MODEL_CONFIGS,default="svc");h.add_argument("--dataset",default="v0");h.add_argument("--config")
    ft=m.add_parser("fit-final");ft.add_argument("--model",choices=MODEL_CONFIGS,default="svc");ft.add_argument("--dataset",default="v0");ft.add_argument("--config");ft.add_argument("--registry",default="experiments/registry.sqlite")
    e=sub.add_parser("ensemble").add_subparsers(dest="action",required=True);q=e.add_parser("evaluate");q.add_argument("--dataset",default="v0");q.add_argument("--artifact-dir",default="artifacts");q.add_argument("--bootstrap-replicates",type=int,default=2000)
    x=sub.add_parser("export");x.add_argument("--model",choices=MODEL_CONFIGS,default="svc");x.add_argument("--checkpoint",default="artifacts/checkpoints/svc_v0_final.json");x.add_argument("--test",default="data/raw/test.csv");x.add_argument("--output",default="submission.parquet")
    k=sub.add_parser("package-kaggle");k.add_argument("--manifest",default="artifacts/checkpoints/svc_v0_final.json");k.add_argument("--output",default="artifacts/kaggle/model_package.zip")
    a=parser.parse_args(argv)
    if a.command=="data": result=prepare_data(a.config)
    elif a.command=="features": result=build_features(a.dataset,a.data_dir,a.config)
    elif a.command=="models" and a.action=="train": result=train_models(a.dataset,a.model,a.data_dir,a.artifact_dir,config_path=a.config,feature_config_path=a.features_config,seed=a.seed)
    elif a.command=="models" and a.action=="evaluate-holdout": result=evaluate_holdout(a.model,a.dataset,config_path=a.config)
    elif a.command=="models": result=fit_final(a.model,a.dataset,registry_path=a.registry,config_path=a.config)
    elif a.command=="ensemble": result=evaluate_ensembles(a.dataset,a.artifact_dir,bootstrap_replicates=a.bootstrap_replicates)
    elif a.command=="export":
        from .inference import export_submission
        result=export_submission(a.model,a.checkpoint,a.test,a.output)
    elif a.command=="package-kaggle":
        from .inference import package_offline_model
        result=package_offline_model(a.manifest,a.output)
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":main()
