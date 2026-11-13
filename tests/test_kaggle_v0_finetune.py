import json

import numpy as np
import pandas as pd

from src import pipeline


LABELS = [f"label-{index:02d}" for index in range(18)]


def test_align_saved_validation_predictions_uses_holdout_order_and_checks_metadata(tmp_path):
    valid = pd.DataFrame({
        "id": ["b", "a"], "prompt_id": ["pb", "pa"],
        "model": [LABELS[1], LABELS[0]],
    })
    saved = pd.DataFrame({
        "id": ["a", "b"], "prompt_id": ["pa", "pb"],
        "y_true": [LABELS[0], LABELS[1]],
        **{f"prob::{label}": [float(label == LABELS[0]), float(label == LABELS[1])]
           for label in LABELS},
    })
    path = tmp_path / "source.parquet"
    saved.to_parquet(path, index=False)
    probs, resolved = pipeline._align_saved_validation_predictions(
        {"prediction_path": str(path)}, valid, LABELS, tmp_path)
    assert resolved == path
    assert probs.argmax(axis=1).tolist() == [1, 0]

    saved.loc[saved.id == "a", "prompt_id"] = "wrong"
    saved.to_parquet(path, index=False)
    try:
        pipeline._align_saved_validation_predictions(
            {"prediction_path": str(path)}, valid, LABELS, tmp_path)
    except ValueError as exc:
        assert "prompt_id" in str(exc)
    else:
        raise AssertionError("mismatched source prompt metadata should fail")


def test_kaggle_finetune_isolated_from_registry_and_trust_list(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    predictions_dir = artifact_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    source_id = "source-v214"
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")

    valid = pd.DataFrame({
        "id": [f"v-{i}" for i in range(540)],
        "prompt_id": [f"vp-{i // 18}" for i in range(540)],
        "text": [f"validation {i}" for i in range(540)],
        "model": [LABELS[i % 18] for i in range(540)],
    })
    train = pd.DataFrame({
        "id": [f"t-{i}" for i in range(2159)],
        "prompt_id": [f"tp-{i // 18}" for i in range(2159)],
        "text": [f"training {i}" for i in range(2159)],
        "model": [LABELS[i % 18] for i in range(2159)],
    })
    source_probs = np.full((540, 18), 0.001, dtype=np.float64)
    source_probs[np.arange(540), np.arange(540) % 18] = 1 - 17 * 0.001
    source_frame = valid[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    for idx, label in enumerate(LABELS):
        source_frame[f"prob::{label}"] = source_probs[:, idx]
    source_prediction = predictions_dir / f"{source_id}_validation.parquet"
    source_frame.to_parquet(source_prediction, index=False)
    source_manifest = {
        "experiment_id": source_id, "model": "ultrabert380", "dataset": "v2.14",
        "training_dataset": "v2.14", "validation_rows": 2699,
        "best_checkpoint_path": str(checkpoint), "checkpoint_path": str(checkpoint),
        "prediction_path": str(source_prediction), "classes": LABELS,
        "checkpoint_id": "dmis-lab/UltraBERT-380M", "model_revision": "revision",
        "config": {"max_length": 581, "batch_size": 8, "auto_batch_size": True,
                    "max_batch_size": 1024, "batch_probe_safety_fraction": .95,
                    "gradient_checkpointing": True, "optimizer": "adamw_torch"},
    }
    (predictions_dir / f"{source_id}_manifest.json").write_text(json.dumps(source_manifest))
    monkeypatch.setattr(pipeline, "_training_partitions", lambda *args: (train, valid))

    captured = {}
    selected_checkpoint = tmp_path / "selected-checkpoint"
    selected_checkpoint.mkdir()

    def fake_train(*args, **kwargs):
        captured.update(kwargs)
        return {
            "probabilities": source_probs.copy(), "scores": np.log(source_probs),
            "best_checkpoint_path": str(selected_checkpoint), "best_epoch": 1,
            "best_validation_accuracy": 1.0, "validation_metrics_by_epoch": [],
            "validation_predictions_by_epoch": [], "fit_seconds": 1.0,
            "batch_size_probe": {"selected_batch_size": 8},
        }

    monkeypatch.setattr(pipeline, "train_encoder_fold", fake_train)
    monkeypatch.setattr(pipeline, "_release_completed_training_model", lambda result: None)
    result = pipeline.kaggle_v0_finetune(from_experiment=source_id,
                                         data_dir=str(tmp_path), artifact_dir=str(artifact_dir))

    assert captured["resume_checkpoint"] == checkpoint
    assert captured["learning_rate"] == 2e-6
    assert captured["epochs"] == 10 and captured["early_stopping_patience"] == 2
    assert captured["copy_resume_state"] is False
    assert "trainer_resume_checkpoint" not in captured
    assert result["scope"] == "kaggle_only_not_in_trust_list_2"
    assert result["train_rows"] == 2159 and result["validation_rows"] == 540
    assert result["validation_metrics"]["balanced_accuracy"] == 1.0
    assert result["corrected_rows"] == result["regressed_rows"] == 0
    assert (artifact_dir / "kaggle_only" / "runs" / result["experiment_id"] / "manifest.json").is_file()
    assert not (artifact_dir / "predictions" / f"{result['experiment_id']}_manifest.json").exists()
