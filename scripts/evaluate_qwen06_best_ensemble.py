"""Register Qwen3-0.6B's best epoch and compare all 2/3 model blends."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.evaluation import evaluate
from src.inference import _predict_neural
from src.model_factory import _fit_temperature
from src.pipeline import _load_dataset, _revision, _sha, evaluate_ensembles
from src.registry import Registry


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SOURCE_EXPERIMENT = "qwen06_v0_data40a22ce8_split147ce64e_seed2026_cfg42943789_code8422c8c"
SOURCE_MANIFEST = ARTIFACTS / "predictions" / f"{SOURCE_EXPERIMENT}_manifest.json"
MODEL_ID = "qwen06_best_epoch03"
EPOCH = 3


def _prediction_frame(rows, y_true, probabilities, classes):
    frame = rows[["id", "prompt_id"]].copy()
    frame["y_true"] = np.asarray(y_true, dtype=str)
    for index, label in enumerate(classes):
        frame[f"prob::{label}"] = probabilities[:, index]
    return frame


def main():
    source = json.loads(SOURCE_MANIFEST.read_text())
    source_checkpoint = Path(source["checkpoint_path"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    checkpoint = source_checkpoint / f"epoch-{EPOCH:02d}"
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"best Qwen checkpoint is missing: {checkpoint}")

    rows, data_dir = _load_dataset("v0", ROOT / "data/processed")
    train = rows[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    valid = rows[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(rows.model.astype(str).unique().tolist())
    if not np.array_equal(classes, source["classes"]):
        raise ValueError("Qwen checkpoint class order does not match the current dataset")

    start = time.perf_counter()
    prediction_kwargs = {"batch_size": 4, "max_length": source.get("config", {}).get("max_length", 384),
                         "temperature": 1.0, "base_checkpoint": source["config"]["checkpoint"],
                         "revision": source["model_revision"]}
    train_probs, _ = _predict_neural(MODEL_ID, checkpoint, train.text.tolist(), classes, **prediction_kwargs)
    valid_probs, _ = _predict_neural(MODEL_ID, checkpoint, valid.text.tolist(), classes, **prediction_kwargs)
    temperature = _fit_temperature(np.log(np.clip(train_probs, 1e-12, 1)), train.model.astype(str).tolist(), classes)
    train_probs = softmax(np.log(np.clip(train_probs, 1e-12, 1)) / temperature, axis=1)
    valid_probs = softmax(np.log(np.clip(valid_probs, 1e-12, 1)) / temperature, axis=1)
    inference_seconds = time.perf_counter() - start

    experiment_id = f"{MODEL_ID}_v0_from{SOURCE_EXPERIMENT[-16:]}"
    prediction_dir = ARTIFACTS / "predictions"
    validation_path = prediction_dir / f"{experiment_id}_validation.parquet"
    train_path = prediction_dir / f"{experiment_id}_train.parquet"
    metrics_path = prediction_dir / f"{experiment_id}_metrics.json"
    manifest_path = prediction_dir / f"{experiment_id}_manifest.json"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    train_frame = _prediction_frame(train, train.model.astype(str), train_probs, classes)
    valid_frame = _prediction_frame(valid, valid.model.astype(str), valid_probs, classes)
    train_frame.to_parquet(train_path, index=False)
    valid_frame.to_parquet(validation_path, index=False)
    model_metrics = evaluate(valid.model.astype(str), np.log(np.clip(valid_probs, 1e-12, 1)), classes, valid_probs)
    model_metrics["source_validation_accuracy_by_epoch"] = source.get("validation_metrics_by_epoch", [])[EPOCH - 1]
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    config = {**source.get("config", {}), "model": MODEL_ID,
              "source_experiment": SOURCE_EXPERIMENT, "selected_epoch": EPOCH}
    manifest = {
        "experiment_id": experiment_id, "model": MODEL_ID, "dataset": "v0", "split": "validation",
        "source_experiment": SOURCE_EXPERIMENT, "selected_epoch": EPOCH,
        "checkpoint_path": str(checkpoint), "checkpoint_id": source.get("checkpoint_id"),
        "model_revision": source.get("model_revision"), "classes": classes, "config": config,
        "dataset_sha256": source.get("dataset_sha256"), "split_sha256": source.get("split_sha256"),
        "train_rows": len(train), "validation_rows": len(valid), "calibration_temperature": temperature,
        "calibration_method": "scalar temperature fitted on in-sample train predictions only",
        "fit_seconds": source.get("fit_seconds"), "inference_seconds": inference_seconds,
        "checkpoint_size_bytes": checkpoint_bytes,
        "train_prediction_path": str(train_path), "prediction_path": str(validation_path),
        "prediction_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "metrics": model_metrics, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "code_revision": _revision(), "config_sha256": _sha(config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    metrics_path.write_text(json.dumps(manifest, indent=2) + "\n")
    Registry(ROOT / "experiments/registry.sqlite").add(
        experiment_id, MODEL_ID, "validation", config, model_metrics,
        dataset_hash=manifest["dataset_sha256"], split_hash=manifest["split_sha256"],
        model_revision=manifest["model_revision"], code_revision=manifest["code_revision"],
        seed=source.get("seed"), runtime_seconds=manifest["fit_seconds"],
        artifact_path=str(validation_path), config_hash=manifest["config_sha256"],
        description="Qwen3-0.6B best checkpoint from epoch 3 of the 10-epoch run; selected by validation accuracy.")

    report = evaluate_ensembles(
        "v0", ARTIFACTS, ROOT / "experiments/registry.sqlite",
        bootstrap_replicates=2000, seed=source.get("seed", 2026),
        model_names=["svc", "ultrabert380", MODEL_ID], force=True)
    print(json.dumps({"best_checkpoint_experiment": experiment_id,
                      "best_checkpoint_accuracy": model_metrics["balanced_accuracy"],
                      "ensemble_experiment": report["experiment_id"],
                      "best_ensemble": report["best_ensemble"],
                      "ensemble_balanced_accuracy": {
                          name: value["metrics"]["balanced_accuracy"]
                          for name, value in report["ensembles"].items()}}, indent=2))


if __name__ == "__main__":
    main()
