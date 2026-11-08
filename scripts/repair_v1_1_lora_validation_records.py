"""Re-score completed v1.1 LoRA runs from their selected saved checkpoints."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.ensemble_registry import refresh_prediction_factory
from src.evaluation import evaluate
from src.inference import _predict_neural
from src.model_factory import _fit_temperature
from src.pipeline import _load_dataset
from src.registry import Registry


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/predictions"
REGISTRY_PATH = ROOT / "experiments/registry.sqlite"
REPAIRED_MODELS = {
    "granite4_350m", "llama32_3b", "ministral3_3b", "qwen06", "qwen17",
    "qwen4", "smollm3_3b", "granite4_h_micro",
}
CALIBRATION_ROWS = 8192


def main() -> None:
    rows, _ = _load_dataset("v1.1", ROOT / "data/processed")
    train = rows.loc[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    validation = rows.loc[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(rows.model.astype(str).unique().tolist())
    descriptions = {row["experiment_id"]: row.get("description", "")
                    for row in Registry(REGISTRY_PATH).rows()}
    repaired = []

    for manifest_path in sorted(PREDICTIONS.glob("*_v1.1_*_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest.get("model")
        metrics = manifest.get("metrics") or {}
        recorded = metrics.get("balanced_accuracy")
        selected = manifest.get("selected_validation_accuracy")
        if (model not in REPAIRED_MODELS or selected is None or recorded is None
                or abs(float(selected) - float(recorded)) <= 0.1):
            continue

        experiment_id = manifest["experiment_id"]
        config = manifest["config"]
        checkpoint = ROOT / manifest["checkpoint_path"]
        prediction_path = ROOT / manifest["prediction_path"]
        old_predictions = pd.read_parquet(prediction_path)
        if old_predictions.id.astype(str).tolist() != validation.id.astype(str).tolist():
            raise ValueError(f"{experiment_id}: saved prediction IDs do not match frozen v1.1 validation order")

        n_calibration = min(CALIBRATION_ROWS, len(train))
        calibration = train.sample(n=n_calibration, random_state=int(manifest.get("seed", 2026)))
        texts = calibration.text.astype(str).tolist() + validation.text.astype(str).tolist()
        started = time.perf_counter()
        probabilities, returned_classes, scores = _predict_neural(
            model, checkpoint, texts, classes,
            batch_size=max(1, int(manifest.get("effective_batch_size") or config.get("batch_size", 4))),
            max_length=int(config.get("max_length", 384)), temperature=1.0,
            base_checkpoint=config["checkpoint"], revision=manifest["model_revision"],
            fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
            text_only_backbone=bool(config.get("text_only_backbone", False)),
            return_scores=True)
        if returned_classes != classes:
            raise ValueError(f"{experiment_id}: checkpoint returned a different class order")

        train_scores = scores[:n_calibration]
        validation_scores = scores[n_calibration:]
        temperature = _fit_temperature(train_scores, calibration.model.astype(str).tolist(), classes)
        validation_probabilities = softmax(validation_scores / temperature, axis=1)
        y_true = validation.model.astype(str).to_numpy()
        corrected_metrics = evaluate(y_true, validation_scores, classes, validation_probabilities)

        repaired_frame = old_predictions[["id", "prompt_id", "y_true"]].copy()
        for column, label in enumerate(classes):
            repaired_frame[f"score::{label}"] = validation_scores[:, column]
            repaired_frame[f"prob::{label}"] = validation_probabilities[:, column]
        repaired_frame.to_parquet(prediction_path, index=False)

        manifest["metrics"] = corrected_metrics
        manifest["prediction_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        manifest["calibration_temperature"] = temperature
        manifest["calibration_method"] = f"train-only scalar temperature ({n_calibration} sampled train rows)"
        manifest["posthoc_repair"] = {
            "method": "reloaded selected checkpoint; direct offline inference",
            "reason": "post-training Trainer.predict metrics disagreed with epoch validation and saved checkpoint",
            "calibration_rows": n_calibration,
            "validation_rows": len(validation),
            "repair_seconds": time.perf_counter() - started,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        metrics_path = manifest_path.with_name(manifest_path.name.removesuffix("_manifest.json") + "_metrics.json")
        metrics_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        Registry(REGISTRY_PATH).add(
            experiment_id, model, "validation", manifest["config"], corrected_metrics,
            dataset_hash=manifest.get("dataset_sha256"), code_revision=manifest.get("code_revision"),
            seed=manifest.get("seed"), runtime_seconds=manifest.get("fit_seconds"),
            artifact_path=str(prediction_path), config_hash=manifest.get("config_sha256"),
            split_hash=manifest.get("split_sha256"), model_revision=manifest.get("model_revision"),
            description=descriptions.get(experiment_id, ""))
        repaired.append((experiment_id, corrected_metrics["balanced_accuracy"], temperature))
        print(f"REPAIRED {experiment_id} acc={corrected_metrics['balanced_accuracy']:.6f} "
              f"temperature={temperature:.6g}", flush=True)

        del probabilities, returned_classes, scores
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if not repaired:
        print("No completed v1.1 LoRA records needed repair.", flush=True)
        return

    last_manifest = PREDICTIONS / f"{repaired[-1][0]}_manifest.json"
    update = refresh_prediction_factory(
        artifact_dir=ROOT / "artifacts", data_dir=ROOT / "data/processed",
        reference_dataset="v0", completed_manifest=last_manifest,
        evaluate_if_unchanged=True, registry_path=ROOT / "docs/ensembles.md",
        roster_path=PREDICTIONS / "active_model_roster.json")
    for experiment_id, _, _ in repaired:
        path = PREDICTIONS / f"{experiment_id}_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["prediction_factory_update"] = update
        path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        metrics_path = path.with_name(path.name.removesuffix("_manifest.json") + "_metrics.json")
        metrics_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"PREDICTION_FACTORY {json.dumps(update, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
