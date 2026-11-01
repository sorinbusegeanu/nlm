"""CLI orchestration for immutable V0 data, model, ensemble, and export runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
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
from .ensemble_factory import ensemble_probabilities
from .feature_factory import load_or_fit
from .model_factory import train_encoder_fold, train_qwen_lora_fold, train_svc
from .registry import Registry

MODEL_CONFIGS = {"svc": "configs/models/svc.yaml", "ultrabert380": "configs/models/ultrabert380.yaml",
    "qwen06": "configs/models/qwen06.yaml", "qwen17": "configs/models/qwen17.yaml", "qwen4": "configs/models/qwen4.yaml"}
DEFAULT_MVP_MODELS = ["svc", "ultrabert380", "qwen06"]


def _revision():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: return "unknown"


def _read_yaml(path):
    with open(path, encoding="utf-8") as stream: return yaml.safe_load(stream) or {}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _ensemble_member_labels(members):
    display_names = {"svc": "SVC", "ultrabert380": "UltraBERT-380M", "qwen06": "Qwen3-0.6B",
        "qwen17": "Qwen3-1.7B", "qwen4": "Qwen3-4B"}
    return [display_names.get(name.split("_", 1)[0], name.split("_", 1)[0]) for name in members]


def _ensemble_model_label(members):
    return " + ".join(_ensemble_member_labels(members)) + " ensemble"


def _ensemble_variant_label(name):
    train_weighted = name.startswith("train_weighted:")
    if train_weighted:
        name = name.removeprefix("train_weighted:")
    labels = _ensemble_member_labels(name.split("+"))
    method = "Train-fitted weighted blend" if train_weighted else "Equal-weight blend"
    return f"{method}: {' + '.join(labels)}"


def prepare_data(config):
    cfg = _read_yaml(config)
    return prepare(cfg["train_path"], cfg["output_dir"], cfg.get("test_path"), seed=cfg.get("seed", 2026),
                   validation_fraction=cfg.get("validation_fraction", .2), candidates=cfg.get("split_candidates", 1000),
                   expected_classes=cfg.get("expected_classes", 18))


def _load_dataset(dataset, data_dir):
    data_dir = Path(data_dir)
    rows = pd.read_parquet(data_dir / f"dataset_{dataset}.parquet")
    assignments = pd.read_parquet(data_dir / f"splits_{dataset}.parquet")
    if rows.id.duplicated().any() or assignments.id.duplicated().any(): raise ValueError("dataset and split assignment IDs must be unique")
    if set(rows.id.astype(str)) != set(assignments.id.astype(str)): raise ValueError("dataset and split assignment ID sets differ")
    rows = rows.merge(assignments[["id", "split"]], on="id", how="left", validate="one_to_one")
    if rows["split"].isna().any() or set(rows.split) != {"train", "validation"}: raise ValueError("rows need train and validation split assignments")
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
    elif model == "ultrabert380":
        checkpoint = Path(output_dir) / "checkpoints" / f"ultrabert380_fold{fold}"
        result = train_encoder_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 3), batch_size=cfg.get("batch_size", 16), max_length=cfg.get("max_length", 384),
            learning_rate=cfg.get("learning_rate", 2e-5), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1))
        result["checkpoint"] = str(checkpoint)
    elif model in {"qwen06", "qwen17", "qwen4"}:
        checkpoint = Path(output_dir) / "checkpoints" / f"{model}_fold{fold}"
        result = train_qwen_lora_fold(train.text.tolist(), train.model.tolist(), valid.text.tolist(), classes,
            checkpoint=cfg["checkpoint"], revision=cfg["revision"], output_dir=checkpoint,
            epochs=cfg.get("epochs", 3), batch_size=cfg.get("batch_size", 4), max_length=cfg.get("max_length", 384),
            learning_rate=cfg.get("learning_rate", 1e-4), seed=seed,
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            lora_rank=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            save_merged=final, validation_labels=None if final else valid.model.tolist())
        result["checkpoint"] = str(checkpoint)
    else: raise ValueError(f"unknown model: {model}")
    return result


def build_features(dataset="v0", data_dir="data/processed", config_path="configs/features/default.yaml", cache_dir="artifacts/features"):
    rows, _ = _load_dataset(dataset, data_dir)
    cfg = _read_yaml(config_path)
    train, valid = rows[rows.split.eq("train")], rows[rows.split.eq("validation")]
    x_train, x_val, features, key = load_or_fit(train.text, valid.text, config=cfg, cache_dir=cache_dir, fold_token=f"{dataset}-train-validation")
    report = [{"cache_key": key, "train_shape": list(x_train.shape), "validation_shape": list(x_val.shape)}]
    output = Path(cache_dir) / f"{dataset}_features_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": dataset, "feature_config": cfg, "split": "train_validation", "partitions": report}, indent=2) + "\n")
    return report


def train_models(dataset="v0", model="svc", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, config_path=None, feature_config_path=None, seed=2026, folds=None):
    """Fit one model on the fixed 80% train split and score its 20% validation split."""
    if folds is not None:
        raise ValueError("fold selection was removed; this pipeline uses one fixed train/validation split")
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
    prediction_path = pred_dir / f"{experiment_id}_validation.parquet"
    manifest_path = pred_dir / f"{experiment_id}_manifest.json"
    if manifest_path.exists() and prediction_path.exists(): return json.loads(manifest_path.read_text())
    if run_artifact_dir.exists(): raise FileExistsError(f"incomplete immutable experiment exists at {run_artifact_dir}; remove that generated run directory to retry")
    classes = sorted(rows.model.astype(str).unique().tolist())
    train = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    valid = rows[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    if set(train.prompt_id.astype(str)) & set(valid.prompt_id.astype(str)):
        raise RuntimeError("prompt leakage between train and validation")
    result = _train_fold(model, train, valid, classes, cfg, run_artifact_dir, seed, "validation", Path(artifact_dir)/"features",
                         cache_fold=f"{dataset}-train-validation")
    scores, probabilities = result["scores"], result["probabilities"]
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all(): raise RuntimeError("validation output is incomplete or non-finite")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5): raise RuntimeError("validation probability rows do not sum to one")
    metrics = evaluate(valid.model.astype(str), scores, classes, probabilities)
    pred = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for i, label in enumerate(classes):
        pred[f"score::{label}"] = scores[:, i]; pred[f"prob::{label}"] = probabilities[:, i]
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(prediction_path, index=False)
    checkpoint = Path(result["checkpoint"])
    artifact_size = sum(f.stat().st_size for f in checkpoint.rglob("*") if f.is_file()) if checkpoint.is_dir() else checkpoint.stat().st_size
    manifest = {"experiment_id": experiment_id, "model": model, "dataset": dataset,
        "split": "validation", "classes": classes, "train_rows": len(train), "validation_rows": len(valid),
        "train_id_sha256": hashlib.sha256("\n".join(train.id.astype(str)).encode()).hexdigest(),
        "id_sha256": hashlib.sha256("\n".join(valid.id.astype(str)).encode()).hexdigest(),
        "dataset_sha256": profile["dataset_sha256"], "split_sha256": profile["split_sha256"],
        "code_revision": code_revision, "model_revision": cfg.get("revision"), "checkpoint_id": cfg.get("checkpoint"),
        "config": config_snapshot, "config_sha256": config_hash, "seed": seed,
        "fit_seconds": float(result["fit_seconds"]), "inference_seconds": result.get("inference_seconds"),
        "inference_rows_per_second": len(valid) / result["inference_seconds"] if result.get("inference_seconds") else None,
        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"), "checkpoint_size_bytes": artifact_size,
        "calibration_method": "sigmoid train-only CV" if model == "svc" else "train-only scalar temperature",
        "calibration_temperature": result.get("calibration_temperature"),
        "feature_cache_key": result.get("cache_key"), "checkpoint_path": str(checkpoint),
        "prediction_path": str(prediction_path), "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "metrics": metrics, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    Registry(registry_path).add(manifest["experiment_id"], model, "validation", manifest["config"], metrics,
        dataset_hash=manifest["dataset_sha256"], code_revision=manifest["code_revision"], seed=seed,
        runtime_seconds=manifest["fit_seconds"], artifact_path=str(prediction_path), config_hash=manifest["config_sha256"],
        split_hash=manifest["split_sha256"], model_revision=manifest["model_revision"],
        description=f"{model} trained on {len(train)} rows and validated on {len(valid)} prompt-grouped rows (80/20).")
    (pred_dir / f"{experiment_id}_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def continue_qwen(dataset="v0", data_dir="data/processed", artifact_dir="artifacts",
                  registry_path="experiments/registry.sqlite", *, from_experiment=None, epochs=4, seed=2026,
                  config_path=None):
    """Continue a saved Qwen LoRA model and validate after every additional epoch."""
    from .model_factory import train_qwen_lora_fold
    cfg = _model_cfg("qwen06", config_path)
    rows, data_dir = _load_dataset(dataset, data_dir)
    profile = json.loads((data_dir/"dataset_profile.json").read_text())
    manifests_dir = Path(artifact_dir)/"predictions"
    candidates = []
    for path in manifests_dir.glob(f"qwen06_{dataset}_*_manifest.json"):
        record = json.loads(path.read_text())
        if record.get("model") == "qwen06" and record.get("split_sha256") == profile["split_sha256"]:
            candidates.append((path, record))
    if from_experiment:
        candidates = [(p, m) for p, m in candidates if m.get("experiment_id") == from_experiment]
    if not candidates: raise FileNotFoundError("no matching Qwen validation run found to continue")
    source_path, source = max(candidates, key=lambda item: item[1].get("created_at_utc", ""))
    source_checkpoint = Path(source["checkpoint_path"])
    if not source_checkpoint.is_absolute() and not source_checkpoint.exists(): source_checkpoint = Path(artifact_dir)/source_checkpoint
    if not source_checkpoint.is_dir(): raise FileNotFoundError(f"Qwen continuation checkpoint not found: {source_checkpoint}")
    if epochs < 1: raise ValueError("additional epochs must be positive")
    cfg["epochs"] = epochs
    snapshot = {k:v for k,v in cfg.items() if not k.startswith("_")}
    snapshot["continued_from_experiment"] = source["experiment_id"]
    snapshot["additional_epochs"] = epochs
    config_hash = _sha(snapshot); code_revision = _revision()
    source_key = hashlib.sha256(source["experiment_id"].encode()).hexdigest()[:8]
    experiment_id = f"qwen06_{dataset}_continue{epochs}_from{source_key}_seed{seed}_cfg{config_hash[:8]}_code{code_revision[:7]}"
    run_dir = Path(artifact_dir)/"runs"/experiment_id
    if run_dir.exists(): raise FileExistsError(f"continuation run already exists: {run_dir}; choose a new seed or clean generated run")
    train = rows[rows.split.eq("train")].sort_values("id",kind="stable").reset_index(drop=True)
    valid = rows[rows.split.eq("validation")].sort_values("id",kind="stable").reset_index(drop=True)
    if set(train.prompt_id.astype(str)) & set(valid.prompt_id.astype(str)): raise RuntimeError("prompt leakage between train and validation")
    checkpoint = run_dir/"checkpoints"/"qwen06_continued"
    result = train_qwen_lora_fold(train.text.tolist(),train.model.tolist(),valid.text.tolist(),
        sorted(rows.model.astype(str).unique()),checkpoint=cfg["checkpoint"],revision=cfg["revision"],
        output_dir=checkpoint,epochs=epochs,batch_size=cfg.get("batch_size",4),max_length=cfg.get("max_length",384),
        learning_rate=cfg.get("learning_rate",1e-4),seed=seed,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",1),
        lora_rank=cfg.get("lora_rank",16),lora_alpha=cfg.get("lora_alpha",32),
        lora_target_modules=cfg.get("lora_target_modules",["q_proj","k_proj","v_proj","o_proj"]),
        resume_checkpoint=source_checkpoint,validation_labels=valid.model.tolist())
    classes=sorted(rows.model.astype(str).unique())
    metrics=evaluate(valid.model.astype(str),result["scores"],classes,result["probabilities"])
    metrics["validation_metrics_by_epoch"]=result["validation_metrics_by_epoch"]
    pred=valid[["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy()
    for i,label in enumerate(classes): pred[f"score::{label}"]=result["scores"][:,i];pred[f"prob::{label}"]=result["probabilities"][:,i]
    pred_dir=Path(artifact_dir)/"predictions";pred_dir.mkdir(parents=True,exist_ok=True)
    prediction_path=pred_dir/f"{experiment_id}_validation.parquet";pred.to_parquet(prediction_path,index=False)
    manifest={"experiment_id":experiment_id,"model":"qwen06","dataset":dataset,"split":"validation",
        "continued_from_experiment":source["experiment_id"],"continued_from_checkpoint":str(source_checkpoint),
        "additional_epochs":epochs,"total_epochs":int(source.get("config",{}).get("epochs",0))+epochs,
        "train_rows":len(train),"validation_rows":len(valid),"classes":classes,"config":snapshot,
        "config_sha256":config_hash,"dataset_sha256":profile["dataset_sha256"],"split_sha256":profile["split_sha256"],
        "code_revision":code_revision,"model_revision":cfg["revision"],"checkpoint_id":cfg["checkpoint"],"seed":seed,
        "fit_seconds":result["fit_seconds"],"inference_seconds":result["inference_seconds"],
        "calibration_temperature":result["calibration_temperature"],"validation_metrics_by_epoch":result["validation_metrics_by_epoch"],
        "checkpoint_path":str(checkpoint),"prediction_path":str(prediction_path),"metrics":metrics,
        "created_at_utc":pd.Timestamp.now(tz="UTC").isoformat()}
    manifest_path=pred_dir/f"{experiment_id}_manifest.json";manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    (pred_dir/f"{experiment_id}_metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    Registry(registry_path).add(experiment_id,"qwen06","validation",snapshot,metrics,
        dataset_hash=profile["dataset_sha256"],split_hash=profile["split_sha256"],model_revision=cfg["revision"],
        code_revision=code_revision,seed=seed,runtime_seconds=result["fit_seconds"],artifact_path=str(prediction_path),
        config_hash=config_hash,description=f"Continued Qwen LoRA from {source['experiment_id']} for {epochs} epochs; validation metrics recorded each epoch.")
    return manifest


def _load_validation(dataset, artifact_dir):
    files = sorted(p for p in (Path(artifact_dir) / "predictions").glob("*_validation.parquet")
                   if f"_{dataset}_" in p.name and not p.name.startswith("ensemble_"))
    if not files: raise FileNotFoundError(f"no validation prediction files found for dataset {dataset}")
    frames = {p.name.removesuffix("_validation.parquet"): pd.read_parquet(p) for p in files}
    first = next(iter(frames.values())); classes = sorted(c.removeprefix("prob::") for c in first if c.startswith("prob::"))
    aligned = align_predictions({name: frame.rename(columns={f"prob::{c}": c for c in classes}) for name, frame in frames.items()}, classes)
    manifests = {name: json.loads((Path(artifact_dir)/"predictions"/f"{name}_manifest.json").read_text()) for name in frames}
    dataset_hashes = {m.get("dataset_sha256") for m in manifests.values()}
    split_hashes = {m.get("split_sha256") for m in manifests.values()}
    if len(dataset_hashes) != 1 or len(split_hashes) != 1: raise ValueError("validation artifacts have inconsistent dataset or split hashes")
    for name, manifest in manifests.items():
        if manifest.get("classes") != classes: raise ValueError(f"validation class order differs for {name}")
    base = first.sort_values("id", kind="stable").reset_index(drop=True)
    if not np.array_equal(base.id.astype(str).to_numpy(), aligned.id.astype(str).to_numpy()): raise ValueError("validation IDs do not align")
    for name, frame in frames.items():
        aligned_meta = frame.sort_values("id", kind="stable").reset_index(drop=True)
        for column in ("y_true", "prompt_id"):
            if not np.array_equal(base[column].astype(str).to_numpy(), aligned_meta[column].astype(str).to_numpy()):
                raise ValueError(f"validation metadata mismatch for {name}.{column}")
    y = base.y_true.astype(str).to_numpy(); groups = base.prompt_id.to_numpy()
    probabilities = {name: np.column_stack([aligned[f"{name}::{c}"].to_numpy() for c in classes]) for name in frames}
    return base, classes, y, groups, probabilities, manifests, next(iter(dataset_hashes)), next(iter(split_hashes))


def _load_training_predictions(dataset, artifact_dir, manifests):
    """Predict the fit rows with each trained checkpoint; never read validation labels here."""
    from .inference import _predict_neural, predict_svc_bundle
    rows, _ = _load_dataset(dataset, Path(__file__).resolve().parents[1]/"data/processed")
    train = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(rows.model.astype(str).unique())
    output = {}; y = train.model.astype(str).to_numpy()
    for name, manifest in manifests.items():
        path = Path(artifact_dir)/"predictions"/f"{name}_train.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if not np.array_equal(frame.id.astype(str).to_numpy(), train.id.astype(str).to_numpy()):
                raise ValueError(f"stored train predictions do not align for {name}")
            probabilities = np.column_stack([frame[f"prob::{cls}"].to_numpy() for cls in classes])
        elif manifest["model"] == "svc":
            bundle = joblib.load(manifest["checkpoint_path"])
            probabilities, _ = predict_svc_bundle(bundle, train.text.tolist())
        else:
            probabilities, _ = _predict_neural(manifest["model"], manifest["checkpoint_path"], train.text.tolist(), classes,
                max_length=manifest.get("config", {}).get("max_length", 384),
                temperature=manifest.get("calibration_temperature", 1.0),
                base_checkpoint=manifest.get("config", {}).get("checkpoint"), revision=manifest.get("model_revision"))
        if probabilities.shape != (len(train), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid train predictions for {name}")
        if not path.exists():
            frame = train[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
            for i, cls in enumerate(classes): frame[f"prob::{cls}"] = probabilities[:, i]
            frame.to_parquet(path, index=False)
        output[name] = probabilities
    return y, output


def _train_only_weights(models, y, classes, *, grid_steps=10, shrinkage=0.01):
    """Choose a simplex blend on train-fit predictions, with a small equal-weight prior."""
    names = list(models); n = len(names); equal = np.full(n, 1/n)
    best = None
    for cuts in itertools.product(range(grid_steps + 1), repeat=n):
        if sum(cuts) != grid_steps: continue
        weights = np.asarray(cuts, dtype=float) / grid_steps
        probabilities = np.average(np.stack([models[name] for name in names]), axis=0, weights=weights)
        score = balanced_accuracy_score(y, np.asarray(classes)[probabilities.argmax(1)])
        penalty = shrinkage * float(np.sum((weights - equal) ** 2))
        key = (score - penalty, -float(np.sum((weights-equal)**2)))
        if best is None or key > best[0]: best = (key, weights, float(score))
    if best is None: raise RuntimeError("could not find a train-only ensemble weight vector")
    return dict(zip(names, best[1].tolist())), best[2]


def evaluate_ensembles(dataset="v0", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, bootstrap_replicates=2000, seed=2026, model_names=None, force=False):
    """Compare equal-weight blends on the shared validation partition (exploratory)."""
    started = time.perf_counter()
    base, classes, y, groups, models, manifests, dataset_hash, split_hash = _load_validation(dataset, artifact_dir)
    if model_names:
        requested = set(model_names); available = {name.split("_", 1)[0]: name for name in models}
        missing = requested - set(available)
        if missing: raise ValueError(f"requested models have no validation artifacts: {sorted(missing)}")
        models = {name: p for name, p in models.items() if name.split("_", 1)[0] in requested}
    names = list(models)
    if len(names) < 2: raise ValueError("ensemble evaluation requires at least two model validation files")
    manifests = {name: manifests[name] for name in names}
    train_y, train_models = _load_training_predictions(dataset, artifact_dir, manifests)
    weights, train_weighted_accuracy = _train_only_weights(train_models, train_y, classes)
    reports = {}; predictions = {n: np.asarray(classes)[p.argmax(1)] for n, p in models.items()}
    for size in range(2, len(names) + 1):
        for subset in itertools.combinations(names, size):
            prob = ensemble_probabilities([models[n] for n in subset]); label = "+".join(subset)
            metric = evaluate(y, np.log(np.clip(prob, 1e-12, 1)), classes, prob); metric["members"] = list(subset)
            predictions[label] = np.asarray(classes)[prob.argmax(1)]
            reports[label] = {"metrics": metric, "probabilities": prob}
    weighted_name = "train_weighted:" + "+".join(names)
    weighted_prob = np.average(np.stack([models[name] for name in names]), axis=0,
                               weights=[weights[name] for name in names])
    weighted_metrics = evaluate(y, np.log(np.clip(weighted_prob, 1e-12, 1)), classes, weighted_prob)
    weighted_metrics["members"] = names
    predictions[weighted_name] = np.asarray(classes)[weighted_prob.argmax(1)]
    reports[weighted_name] = {"metrics": weighted_metrics, "probabilities": weighted_prob,
                              "weights": weights, "weight_source": "train_fit_predictions_only",
                              "train_balanced_accuracy": train_weighted_accuracy}
    base_metrics = {name: evaluate(y, np.log(np.clip(prob, 1e-12, 1)), classes, prob) for name, prob in models.items()}
    best_single = max(names, key=lambda name: base_metrics[name]["balanced_accuracy"])
    model_costs = {name: {key: manifests[name].get(key) for key in ("fit_seconds", "inference_seconds", "peak_gpu_memory_bytes", "checkpoint_size_bytes")} for name in names}
    for label, record in reports.items():
        record["paired_bootstrap_vs_best_single"] = paired_group_bootstrap(y, predictions[best_single], predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)
        record["per_class_recall_delta_vs_best_single"] = {cls: record["metrics"]["per_class_recall"][cls] - base_metrics[best_single]["per_class_recall"][cls] for cls in classes}
        members = record["metrics"]["members"]; marginal = {}
        if label == weighted_name:
            equal_prob = ensemble_probabilities([models[item] for item in names])
            equal_pred = np.asarray(classes)[equal_prob.argmax(1)]
            marginal["vs_equal_weight"] = {"delta_balanced_accuracy": float(record["metrics"]["balanced_accuracy"] - balanced_accuracy_score(y, equal_pred)),
                "paired_bootstrap": paired_group_bootstrap(y, equal_pred, predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)}
        elif len(members) > 1:
            for member in members:
                previous = [item for item in members if item != member]
                prev_prob = ensemble_probabilities([models[item] for item in previous]); prev_pred = np.asarray(classes)[prev_prob.argmax(1)]
                marginal[member] = {"delta_balanced_accuracy": float(record["metrics"]["balanced_accuracy"] - balanced_accuracy_score(y, prev_pred)),
                    "paired_bootstrap": paired_group_bootstrap(y, prev_pred, predictions[label], groups, n_bootstrap=bootstrap_replicates, seed=seed)}
        record["marginal_contribution"] = marginal
        record["inference_cost_seconds"] = sum((model_costs[n]["inference_seconds"] or 0) for n in members)
    pred_dir = Path(artifact_dir)/"predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    output = base[["id", "prompt_id", "y_true"]].copy()
    for label, record in reports.items():
        for i, cls in enumerate(classes):
            output[f"{label}::score::{cls}"] = np.log(np.clip(record["probabilities"][:, i], 1e-12, 1))
            output[f"{label}::prob::{cls}"] = record["probabilities"][:, i]
    serializable = {k: {kk: vv for kk, vv in v.items() if kk != "probabilities"} for k, v in reports.items()}
    best = max(reports, key=lambda n: reports[n]["metrics"]["balanced_accuracy"])
    best_metrics = reports[best]["metrics"]; code_revision = _revision()
    member_key = _sha({name: manifests[name].get("prediction_sha256") for name in names})[:10]
    model_label = _ensemble_model_label(names)
    report = {"experiment_id": f"ensemble_{dataset}_seed{seed}_members{member_key}_code{code_revision[:7]}",
        "model": model_label, "dataset": dataset, "split": "validation", "classes": classes, "members": names,
        "validation_rows": len(y),
        "best_single": best_single, "best_ensemble": best, "balanced_accuracy": best_metrics["balanced_accuracy"],
        "calibrated_balanced_accuracy": best_metrics.get("calibrated_balanced_accuracy"), "macro_f1": best_metrics["macro_f1"],
        "log_loss": best_metrics.get("log_loss"), "per_class_recall": best_metrics["per_class_recall"],
        "confusion_matrix": best_metrics["confusion_matrix"], "model_costs": model_costs, "models": base_metrics,
        "ensembles": serializable, "complementarity_all": complementarity(y, predictions), "seed": seed,
        "train_weighting": {"weights": weights, "train_balanced_accuracy": train_weighted_accuracy,
            "source": "in-sample predictions on train rows; validation labels were not used to fit weights",
            "grid_step": 1/10, "equal_weight_shrinkage": 0.01},
        "code_revision": code_revision, "dataset_sha256": dataset_hash, "split_sha256": split_hash,
        "validation_selection_note": "Blend comparisons use the same validation rows as the base model scores; treat them as exploratory, not an unbiased final estimate."}
    report["runtime_seconds"] = time.perf_counter() - started
    path = pred_dir/f"{report['experiment_id']}_validation.parquet"; metrics_path=pred_dir/f"{report['experiment_id']}_metrics.json"
    if metrics_path.exists() and path.exists() and not force: return json.loads(metrics_path.read_text())
    output.to_parquet(path,index=False); report["prediction_path"] = str(path)
    metrics_path.write_text(json.dumps(report,indent=2)+"\n")
    Registry(registry_path).add(report["experiment_id"], model_label, "validation", {"members":names,"seed":seed}, report,
        dataset_hash=dataset_hash, split_hash=split_hash, code_revision=code_revision, seed=seed,
        runtime_seconds=report["runtime_seconds"], artifact_path=str(path),
        description=f"Validation blend comparison of {' + '.join(_ensemble_member_labels(names))} on {dataset} (exploratory).")
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
    # Manifests are consumed by export/package code, which resolves relative
    # checkpoint paths from the manifest directory. Store that relative path
    # here so both the versioned manifest and stable alias resolve correctly.
    checkpoint_for_manifest = os.path.relpath(path, final_manifest_path.parent)
    manifest = {"experiment_id": run_name, "model": model, "dataset": dataset, "classes": classes, "checkpoint": checkpoint_for_manifest,
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
        artifact_path=str(final_manifest_path), model_revision=cfg.get("revision"),
        description=f"Final {model} fit on all labelled {dataset} rows for inference; seed {seed}.")
    return manifest


def restart_mvp(*, models=None, seed=2026, bootstrap_replicates=2000, data_config="configs/data/official_v0.yaml",
                feature_config="configs/features/default.yaml", registry_path="experiments/registry.sqlite"):
    """Clear generated experiment state and rebuild the complete MVP from raw CSVs."""
    root = Path(__file__).resolve().parents[1]
    processed = (root / "data/processed").resolve()
    artifacts = (root / "artifacts").resolve()
    protected = [(root / "data/raw").resolve(), (Path.home()/".cache/huggingface").resolve()]
    for target in (processed, artifacts):
        if target in protected or root not in target.parents:
            raise RuntimeError(f"refusing to clean unexpected path: {target}")
    for target in (processed, artifacts):
        if target.exists():
            for child in target.iterdir():
                if child.is_symlink() or not child.is_dir(): child.unlink()
                else: shutil.rmtree(child)
        else:
            target.mkdir(parents=True)
    registry = (root / registry_path).resolve()
    if root not in registry.parents or registry in protected:
        raise RuntimeError(f"refusing to remove unexpected registry path: {registry}")
    for suffix in ("", "-wal", "-shm"):
        Path(str(registry) + suffix).unlink(missing_ok=True)
    data_cfg = _read_yaml(root / data_config)
    data_cfg["seed"] = seed
    profile = prepare(root / data_cfg["train_path"], root / data_cfg["output_dir"],
        root / data_cfg["test_path"] if data_cfg.get("test_path") else None,
        seed=seed, validation_fraction=data_cfg.get("validation_fraction", .2),
        candidates=data_cfg.get("split_candidates", 1000), expected_classes=data_cfg.get("expected_classes", 18))
    feature_report = build_features("v0", root / "data/processed", root / feature_config, root / "artifacts/features")
    selected = models or list(DEFAULT_MVP_MODELS)
    model_reports = [train_models("v0", name, root/"data/processed", root/"artifacts", registry,
        config_path=root/MODEL_CONFIGS[name], feature_config_path=root/feature_config if name == "svc" else None, seed=seed)
        for name in selected]
    ensemble_report = evaluate_ensembles("v0", root/"artifacts", registry, bootstrap_replicates=bootstrap_replicates,
        seed=seed, model_names=selected)
    return {"cleaned": [str(processed), str(artifacts), str(registry)], "dataset_profile": profile,
        "feature_report": feature_report, "models": model_reports, "ensemble": ensemble_report}


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m src.pipeline");sub=parser.add_subparsers(dest="command",required=True)
    restart=sub.add_parser("restart", help="clean generated state and rebuild data, features, models, and ensembles")
    restart.add_argument("--models",nargs="+",choices=MODEL_CONFIGS,default=list(DEFAULT_MVP_MODELS))
    restart.add_argument("--seed",type=int,default=2026);restart.add_argument("--bootstrap-replicates",type=int,default=2000)
    d=sub.add_parser("data").add_subparsers(dest="action",required=True);p=d.add_parser("prepare");p.add_argument("--config",default="configs/data/official_v0.yaml")
    f=sub.add_parser("features").add_subparsers(dest="action",required=True);b=f.add_parser("build");b.add_argument("--dataset",default="v0");b.add_argument("--data-dir",default="data/processed");b.add_argument("--config",default="configs/features/default.yaml")
    m=sub.add_parser("models").add_subparsers(dest="action",required=True)
    t=m.add_parser("train");t.add_argument("--model",choices=MODEL_CONFIGS,default="svc");t.add_argument("--dataset",default="v0");t.add_argument("--data-dir",default="data/processed");t.add_argument("--artifact-dir",default="artifacts");t.add_argument("--config");t.add_argument("--features-config");t.add_argument("--seed",type=int,default=2026)
    cq=m.add_parser("continue-qwen",help="continue the latest Qwen LoRA checkpoint and validate after every epoch")
    cq.add_argument("--dataset",default="v0");cq.add_argument("--epochs",type=int,default=4);cq.add_argument("--seed",type=int,default=2026)
    cq.add_argument("--from-experiment");cq.add_argument("--config")
    ft=m.add_parser("fit-final");ft.add_argument("--model",choices=MODEL_CONFIGS,default="svc");ft.add_argument("--dataset",default="v0");ft.add_argument("--config");ft.add_argument("--registry",default="experiments/registry.sqlite")
    e=sub.add_parser("ensemble").add_subparsers(dest="action",required=True);q=e.add_parser("evaluate");q.add_argument("--dataset",default="v0");q.add_argument("--artifact-dir",default="artifacts");q.add_argument("--bootstrap-replicates",type=int,default=2000);q.add_argument("--models",nargs="+",choices=MODEL_CONFIGS);q.add_argument("--force",action="store_true")
    x=sub.add_parser("export");x.add_argument("--model",choices=MODEL_CONFIGS,default="svc");x.add_argument("--checkpoint",default="artifacts/checkpoints/svc_v0_final.json");x.add_argument("--test",default="data/raw/test.csv");x.add_argument("--output",default="submission.parquet")
    k=sub.add_parser("package-kaggle");k.add_argument("--manifest",default="artifacts/checkpoints/svc_v0_final.json");k.add_argument("--output",default="artifacts/kaggle/model_package.zip")
    a=parser.parse_args(argv)
    if a.command=="restart": result=restart_mvp(models=a.models,seed=a.seed,bootstrap_replicates=a.bootstrap_replicates)
    elif a.command=="data": result=prepare_data(a.config)
    elif a.command=="features": result=build_features(a.dataset,a.data_dir,a.config)
    elif a.command=="models" and a.action=="train": result=train_models(a.dataset,a.model,a.data_dir,a.artifact_dir,config_path=a.config,feature_config_path=a.features_config,seed=a.seed)
    elif a.command=="models" and a.action=="continue-qwen": result=continue_qwen(a.dataset,from_experiment=a.from_experiment,epochs=a.epochs,seed=a.seed,config_path=a.config)
    elif a.command=="models": result=fit_final(a.model,a.dataset,registry_path=a.registry,config_path=a.config)
    elif a.command=="ensemble": result=evaluate_ensembles(a.dataset,a.artifact_dir,bootstrap_replicates=a.bootstrap_replicates,model_names=a.models,force=a.force)
    elif a.command=="export":
        from .inference import export_submission
        result=export_submission(a.model,a.checkpoint,a.test,a.output)
    elif a.command=="package-kaggle":
        from .inference import package_offline_model
        result=package_offline_model(a.manifest,a.output)
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":main()
