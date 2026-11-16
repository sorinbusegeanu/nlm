"""Evaluate a trained DeBERTa specialist on its matching labels in all of V0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.inference import _predict_neural

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("deberta_gemini", "deberta_glm"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()

    matches = []
    for path in (ROOT / "artifacts/predictions").glob("*_manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("experiment_name") == args.experiment_name
                and manifest.get("model") == args.model
                and manifest.get("dataset") == args.dataset):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(f"expected one matching experiment manifest, found {len(matches)}")
    manifest_path, manifest = matches[0]
    classes = list(manifest["classes"])
    checkpoint = Path(manifest.get("best_checkpoint_path") or manifest.get("selected_checkpoint_path")
                      or manifest["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"selected checkpoint is missing: {checkpoint}")

    v0 = pd.read_parquet(ROOT / "data/processed/dataset_v0.parquet")
    eval_rows = v0.loc[v0.model.astype(str).isin(classes)].copy().sort_values("id", kind="stable")
    if eval_rows.empty or set(eval_rows.model.astype(str)) != set(classes):
        raise ValueError("full V0 does not contain all specialist target labels")
    config = manifest.get("config", {})
    max_length = int(config.get("max_length", 512))
    batch_size = max(1, int(manifest.get("effective_inference_batch_size") or
                            manifest.get("requested_inference_batch_size") or 8))
    probabilities, returned_classes, scores = _predict_neural(
        args.model, checkpoint, eval_rows.text.astype(str).tolist(), classes,
        batch_size=batch_size, max_length=max_length, return_scores=True,
        attention_backend=config.get("attention_backend"))
    if returned_classes != classes:
        raise ValueError("specialist class order changed during full-V0 inference")
    metrics = evaluate(eval_rows.model.astype(str), scores, classes, probabilities)
    result = eval_rows[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    result["prediction"] = np.asarray(classes)[probabilities.argmax(axis=1)]
    for i, label in enumerate(classes):
        result[f"score::{label}"] = scores[:, i]
        result[f"prob::{label}"] = probabilities[:, i]

    output_dir = ROOT / "artifacts/analysis/deberta_specialists" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{manifest['experiment_id']}_full_v0_predictions.parquet"
    metrics_path = output_dir / f"{manifest['experiment_id']}_full_v0_metrics.json"
    result.to_parquet(predictions_path, index=False)
    summary = {
        "experiment_id": manifest["experiment_id"], "experiment_manifest": str(manifest_path),
        "model": args.model, "dataset": args.dataset, "checkpoint": str(checkpoint),
        "selection_validation": manifest.get("validation_dataset"),
        "evaluation_dataset": "v0", "evaluation_rows": int(len(eval_rows)),
        "evaluation_scope": "all V0 rows whose true label is in the specialist class set",
        "classes": classes, "metrics": metrics,
        "predictions_path": str(predictions_path),
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
