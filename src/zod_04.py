"""ZOD_04 frozen UltraBERT multi-layer representations and logistic readouts."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate
from .trusted_checkpoint_ensemble import ULTRABERT_CHECKPOINTS


N_TRANSFORMER_LAYERS = 24
N_CLASSES = 18
N_SPLITS = 5
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 1e-4,
    "class_weight": None,
    "random_state": 2026,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame[["id", "prompt_id", "text", "model"]].copy()
    for column in normalized:
        normalized[column] = normalized[column].astype(str)
    return _sha256_bytes(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _checkpoint_fingerprint(checkpoint: Path) -> str:
    files = [checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob("*") if p.is_file())
    entries = []
    for path in files:
        stat = path.stat()
        entries.append((path.relative_to(checkpoint).as_posix() if checkpoint.is_dir() else path.name,
                        stat.st_size, stat.st_mtime_ns))
    return _sha256_bytes(json.dumps(entries, separators=(",", ":")).encode("utf-8"))


def _resolve_checkpoint(experiment_id: str, artifact_dir: str | Path) -> tuple[dict, Path, Path]:
    manifest_path = Path(artifact_dir) / "predictions" / f"{experiment_id}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"UltraBERT training manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != experiment_id or manifest.get("model") != "ultrabert380":
        raise ValueError(f"manifest does not describe the requested UltraBERT run: {manifest_path}")
    checkpoint_value = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError(f"manifest has no selected checkpoint path: {manifest_path}")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"selected UltraBERT checkpoint not found: {checkpoint}")
    return manifest, checkpoint, manifest_path


def _load_v0_splits(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    from .pipeline import _load_dataset

    rows, _ = _load_dataset("v0", data_dir)
    required = {"id", "prompt_id", "text", "model", "split"}
    if not required.issubset(rows.columns):
        raise ValueError(f"V0 dataset is missing required columns: {sorted(required - set(rows.columns))}")
    rows = rows.copy()
    for column in ("id", "prompt_id", "text", "model", "split"):
        rows[column] = rows[column].astype(str)
    if rows.id.duplicated().any():
        raise ValueError("V0 dataset row IDs must be unique")
    train = rows.loc[rows.split.eq("train")].sort_values("id", kind="stable").reset_index(drop=True)
    validation = rows.loc[rows.split.eq("validation")].sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(train.model.unique().tolist())
    if len(classes) != N_CLASSES or set(validation.model.unique()) != set(classes):
        raise ValueError(f"ZOD_04 requires the same {N_CLASSES} classes in V0 train and validation")
    if train.prompt_id.nunique() < 2:
        raise ValueError("V0 training data needs at least two prompt groups")
    if set(train.id) & set(validation.id):
        raise ValueError("V0 training and validation IDs overlap")
    return train, validation, classes


def _feature_cache_key(manifest: dict, checkpoint: Path, train: pd.DataFrame,
                       validation: pd.DataFrame, max_length: int) -> tuple[str, dict]:
    details = {
        "format_version": 1,
        "experiment_id": manifest["experiment_id"],
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint),
        "train_rows_sha256": _frame_fingerprint(train),
        "validation_rows_sha256": _frame_fingerprint(validation),
        "max_length": int(max_length),
        "n_transformer_layers": N_TRANSFORMER_LAYERS,
        "pooling": "attention_mask_mean_per_layer",
    }
    key = _sha256_bytes(json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return key, details


def _extract_split(texts: list[str], *, model, tokenizer, torch, max_length: int,
                   batch_size: int, device, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    representations, head_probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(batch_texts, truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()
                       if key in {"input_ids", "attention_mask", "token_type_ids"}}
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) != n_layers + 1:
                actual = None if hidden_states is None else len(hidden_states) - 1
                raise ValueError(f"expected {n_layers} transformer layers, found {actual}")
            mask = encoded["attention_mask"].unsqueeze(-1)
            layer_vectors = []
            for hidden in hidden_states[1:]:  # omit the embedding output; retain transformer layers 1-24
                weights = mask.to(dtype=hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
                layer_vectors.append(pooled.float().cpu().numpy())
            representations.append(np.stack(layer_vectors, axis=1).astype(np.float16, copy=False))
            logits = output.logits.float()
            head_probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(representations), np.concatenate(head_probabilities)


def _extract_and_cache(*, checkpoint: Path, manifest: dict, train: pd.DataFrame,
                       validation: pd.DataFrame, classes: list[str], max_length: int,
                       batch_size: int, feature_path: Path, metadata_path: Path,
                       cache_key: str, cache_details: dict) -> dict:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("ZOD_04 feature extraction requires the project's neural dependencies") from exc

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, output_hidden_states=True)
    model.config.output_hidden_states = True
    head_map = getattr(model.config, "id2label", {}) or {}
    head_classes = [str(head_map.get(index, head_map.get(str(index), classes[index])))
                    for index in range(len(classes))]
    manifest_classes = list(map(str, manifest.get("classes", [])))
    if set(manifest_classes) != set(classes) or len(manifest_classes) != len(classes):
        raise ValueError("UltraBERT manifest labels do not match the frozen V0 class set")
    if set(head_classes) != set(classes) or len(head_classes) != len(classes):
        raise ValueError("UltraBERT checkpoint classifier labels do not match the frozen V0 class set")
    head_order = [head_classes.index(label) for label in classes]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device("cpu")
    train_features, _ = _extract_split(train.text.tolist(), model=model, tokenizer=tokenizer,
        torch=torch, max_length=max_length, batch_size=batch_size, device=device,
        n_layers=N_TRANSFORMER_LAYERS)
    validation_features, raw_head_probabilities = _extract_split(validation.text.tolist(), model=model,
        tokenizer=tokenizer, torch=torch, max_length=max_length, batch_size=batch_size,
        device=device, n_layers=N_TRANSFORMER_LAYERS)
    validation_head_probabilities = raw_head_probabilities[:, head_order]
    if train_features.shape[1] != N_TRANSFORMER_LAYERS or validation_features.shape[1] != N_TRANSFORMER_LAYERS:
        raise AssertionError("feature extraction did not return all transformer layers")
    if train_features.shape[2] != validation_features.shape[2]:
        raise ValueError("train and validation hidden dimensions differ")

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = feature_path.with_name(feature_path.name + ".tmp")
    with temp_path.open("wb") as stream:
        np.savez(stream,
            train_features=train_features,
            validation_features=validation_features,
            train_ids=train.id.to_numpy(dtype=str),
            train_prompt_ids=train.prompt_id.to_numpy(dtype=str),
            train_labels=train.model.to_numpy(dtype=str),
            validation_ids=validation.id.to_numpy(dtype=str),
            validation_prompt_ids=validation.prompt_id.to_numpy(dtype=str),
            validation_labels=validation.model.to_numpy(dtype=str),
            classes=np.asarray(classes, dtype=str),
            validation_head_probabilities=validation_head_probabilities,
            cache_key=np.asarray(cache_key))
    temp_path.replace(feature_path)
    metadata = {**cache_details, "cache_key": cache_key,
                "train_rows": len(train), "validation_rows": len(validation),
                "hidden_size": int(train_features.shape[2]),
                "feature_dtype": str(train_features.dtype),
                "feature_path": str(feature_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _load_feature_cache(feature_path, metadata_path, cache_key, train, validation, classes)


def _load_feature_cache(feature_path: Path, metadata_path: Path, cache_key: str,
                        train: pd.DataFrame, validation: pd.DataFrame,
                        classes: list[str]) -> dict:
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("ZOD_04 feature cache or its manifest is missing")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("cache_key") != cache_key:
        raise ValueError("ZOD_04 feature-cache metadata does not match the requested inputs")
    with np.load(feature_path, allow_pickle=False) as cached:
        if str(cached["cache_key"].item()) != cache_key:
            raise ValueError("ZOD_04 feature archive cache key does not match its manifest")
        result = {key: cached[key] for key in cached.files}
    expected = {
        "train_ids": train.id.to_numpy(dtype=str),
        "train_prompt_ids": train.prompt_id.to_numpy(dtype=str),
        "train_labels": train.model.to_numpy(dtype=str),
        "validation_ids": validation.id.to_numpy(dtype=str),
        "validation_prompt_ids": validation.prompt_id.to_numpy(dtype=str),
        "validation_labels": validation.model.to_numpy(dtype=str),
        "classes": np.asarray(classes, dtype=str),
    }
    for key, values in expected.items():
        if key not in result or not np.array_equal(result[key], values):
            raise ValueError(f"ZOD_04 feature cache row alignment mismatch: {key}")
    train_features = result["train_features"]
    validation_features = result["validation_features"]
    if (train_features.ndim != 3 or validation_features.ndim != 3
            or train_features.shape[1] != N_TRANSFORMER_LAYERS
            or validation_features.shape[1] != N_TRANSFORMER_LAYERS
            or train_features.shape[0] != len(train)
            or validation_features.shape[0] != len(validation)
            or train_features.shape[2] != validation_features.shape[2]):
        raise ValueError("ZOD_04 feature cache has an unexpected representation shape")
    if result["validation_head_probabilities"].shape != (len(validation), len(classes)):
        raise ValueError("ZOD_04 cached UltraBERT head probabilities have an unexpected shape")
    return result


def _classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
    )


def _ordered_probabilities(model, features: np.ndarray, classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(features)
    output = np.zeros((len(features), len(classes)), dtype=np.float64)
    for column, label in enumerate(model.classes_):
        output[:, classes.index(str(label))] = raw[:, column]
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("classifier returned invalid class probabilities")
    return output


def _rank_layers(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 ids: np.ndarray, prompt_ids: np.ndarray, classes: list[str],
                 *, n_splits: int, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = list(splitter.split(np.zeros(len(labels)), labels, groups))
    rows, prediction_data = [], {"id": ids.astype(str), "prompt_id": prompt_ids.astype(str),
                                 "y_true": labels.astype(str)}
    for layer_index in range(N_TRANSFORMER_LAYERS):
        layer_x = features[:, layer_index, :].astype(np.float32, copy=False)
        oof_probabilities = np.zeros((len(labels), len(classes)), dtype=np.float32)
        oof_seen = np.zeros(len(labels), dtype=bool)
        fold_scores = []
        for train_idx, held_idx in fold_indices:
            model = _classifier()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(layer_x[train_idx], labels[train_idx])
            probabilities = _ordered_probabilities(model, layer_x[held_idx], classes)
            predictions = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
            score = float(balanced_accuracy_score(labels[held_idx], predictions))
            fold_scores.append(score)
            oof_probabilities[held_idx] = probabilities.astype(np.float32)
            oof_seen[held_idx] = True
        if not oof_seen.all():
            raise RuntimeError(f"layer {layer_index + 1} did not receive one OOF prediction per row")
        oof_predictions = np.asarray(classes, dtype=object)[oof_probabilities.argmax(axis=1)]
        prefix = f"layer_{layer_index + 1:02d}"
        prediction_data[f"pred::{prefix}"] = oof_predictions
        for class_index, label in enumerate(classes):
            prediction_data[f"prob::{prefix}::{label}"] = oof_probabilities[:, class_index]
        rows.append({"layer": layer_index + 1,
                     "mean_cv_balanced_accuracy": float(np.mean(fold_scores)),
                     "std_cv_balanced_accuracy": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                     "oof_balanced_accuracy": float(balanced_accuracy_score(labels, oof_predictions)),
                     "fold_balanced_accuracies": json.dumps(fold_scores)})
    summary = pd.DataFrame(rows).sort_values(
        ["mean_cv_balanced_accuracy", "oof_balanced_accuracy", "layer"],
        ascending=[False, False, True], kind="stable").reset_index(drop=True)
    summary.insert(0, "cv_rank", np.arange(1, len(summary) + 1))
    cv_predictions = pd.DataFrame(prediction_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "layer_cv_ranking.csv", index=False)
    cv_predictions.to_parquet(output_dir / "cv_oof_predictions.parquet", index=False)
    return summary, cv_predictions


def _write_report(path: Path, *, experiment_id: str, checkpoint: Path, max_length: int,
                  hidden_size: int, cv_summary: pd.DataFrame, validation_summary: pd.DataFrame,
                  recall_tables: dict[str, dict], classes: list[str],
                  n_splits: int, n_train: int, n_validation: int, artifact_dir: Path) -> None:
    lines = [
        "# ZOD_04 — UltraBERT Multi-Layer Readout", "",
        "## Setup", "",
        f"- Frozen checkpoint: `{experiment_id}` (`{checkpoint}`)",
        f"- V0 rows: {n_train} train, {n_validation} frozen validation; {len(classes)} classes.",
        f"- Extracted {N_TRANSFORMER_LAYERS} transformer-layer vectors with attention-mask mean pooling; embedding output excluded.",
        f"- Checkpoint input length: {max_length}; hidden size: {hidden_size}.",
        f"- Layer ranking: {n_splits}-fold `StratifiedGroupKFold`, grouped by `prompt_id`, scoring mean fold balanced accuracy.",
        f"- Logistic regression: `StandardScaler` + `LogisticRegression({json.dumps(LOGISTIC_REGRESSION_PARAMS, sort_keys=True)})`.",
        "- Layer combinations use the CV-ranked individual layers; the 24-layer combination uses all layers in model order.", "",
        "## Validation balanced accuracy", "",
        "| Classifier | Layers | Validation balanced accuracy | Δ vs UltraBERT head |",
        "|---|---|---:|---:|",
    ]
    baseline = float(validation_summary.loc[validation_summary.classifier.eq("ultrabert_head"),
                                            "balanced_accuracy"].iloc[0])
    for row in validation_summary.to_dict("records"):
        layer_text = ", ".join(map(str, row["layers"])) if row["layers"] else "original head"
        lines.append(f"| {row['classifier']} | {layer_text} | {row['balanced_accuracy']:.2%} | {row['balanced_accuracy'] - baseline:+.2%} |")
    lines += ["", "## Individual layer ranking", "",
              "| CV rank | Layer | Mean fold balanced accuracy | SD | Pooled OOF balanced accuracy | Validation balanced accuracy |",
              "|---:|---:|---:|---:|---:|---:|"]
    validation_by_layer = {int(row["layers"][0]): float(row["balanced_accuracy"])
                           for row in validation_summary.to_dict("records")
                           if row["classifier"].startswith("layer_")}
    for row in cv_summary.to_dict("records"):
        layer = int(row["layer"])
        lines.append(f"| {int(row['cv_rank'])} | {layer} | {row['mean_cv_balanced_accuracy']:.2%} | {row['std_cv_balanced_accuracy']:.2%} | {row['oof_balanced_accuracy']:.2%} | {validation_by_layer[layer]:.2%} |")
    for model_name, recalls in recall_tables.items():
        lines += ["", f"## Per-label recall: {model_name}", "",
                  "| Label | Recall |", "|---|---:|"]
        for label in classes:
            lines.append(f"| {label} | {float(recalls[label]):.2%} |")
    lines += ["", "## Artifacts", "",
              f"- Frozen features and row alignment: `{artifact_dir / 'frozen_features_v0.npz'}`",
              f"- Grouped CV predictions: `{artifact_dir / 'cv_oof_predictions.parquet'}`",
              f"- Frozen V0 validation predictions: `{artifact_dir / 'validation_predictions.parquet'}`",
              f"- Classifier bundles: `{artifact_dir / 'classifiers/'}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_zod04(*, data_dir="data/processed", artifact_dir="artifacts",
              output_dir="artifacts/analysis/zod_04", report_path="docs/zod_04.md",
              experiment_id=None, batch_size=8, n_splits=N_SPLITS, seed=2026,
              refresh_features=False):
    """Extract frozen layer features, rank them by grouped CV, and fit final readouts."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    experiment_id = experiment_id or ULTRABERT_CHECKPOINTS["D"]
    manifest, checkpoint, _ = _resolve_checkpoint(experiment_id, artifact_dir)
    train, validation, classes = _load_v0_splits(data_dir)
    config = manifest.get("config", {})
    max_length = int(config.get("max_length") or 0)
    if max_length < 1:
        raise ValueError("selected UltraBERT manifest must record the checkpoint max_length")
    if int(train.prompt_id.nunique()) < n_splits:
        raise ValueError(f"V0 training data has fewer than {n_splits} prompt groups")
    output_dir = Path(output_dir)
    feature_path, metadata_path = output_dir / "frozen_features_v0.npz", output_dir / "feature_manifest.json"
    cache_key, cache_details = _feature_cache_key(manifest, checkpoint, train, validation, max_length)
    if not refresh_features and feature_path.is_file() and metadata_path.is_file():
        try:
            features = _load_feature_cache(feature_path, metadata_path, cache_key, train, validation, classes)
            print(f"Reusing ZOD_04 frozen features from {feature_path}", flush=True)
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            features = _extract_and_cache(checkpoint=checkpoint, manifest=manifest, train=train,
                validation=validation, classes=classes, max_length=max_length, batch_size=batch_size,
                feature_path=feature_path, metadata_path=metadata_path, cache_key=cache_key,
                cache_details=cache_details)
    else:
        features = _extract_and_cache(checkpoint=checkpoint, manifest=manifest, train=train,
            validation=validation, classes=classes, max_length=max_length, batch_size=batch_size,
            feature_path=feature_path, metadata_path=metadata_path, cache_key=cache_key,
            cache_details=cache_details)

    train_features = features["train_features"].astype(np.float32)
    validation_features = features["validation_features"].astype(np.float32)
    y_train = features["train_labels"].astype(str)
    y_validation = features["validation_labels"].astype(str)
    groups = features["train_prompt_ids"].astype(str)
    cv_summary, _ = _rank_layers(train_features, y_train, groups,
        features["train_ids"], features["train_prompt_ids"], classes,
        n_splits=n_splits, seed=seed, output_dir=output_dir)

    ranked_layers = cv_summary.layer.astype(int).tolist()
    specifications = [(f"layer_{layer:02d}", [layer - 1]) for layer in range(1, N_TRANSFORMER_LAYERS + 1)]
    specifications += [
        (f"best_{count}_layers", [layer - 1 for layer in ranked_layers[:count]])
        for count in (2, 4, 8)
    ]
    specifications.append(("all_24_layers", list(range(N_TRANSFORMER_LAYERS))))

    classifier_dir = output_dir / "classifiers"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame({
        "id": features["validation_ids"].astype(str),
        "prompt_id": features["validation_prompt_ids"].astype(str),
        "y_true": y_validation,
    })
    validation_results = []
    for name, layer_indices in specifications:
        selected_train = np.concatenate([train_features[:, index, :] for index in layer_indices], axis=1)
        selected_validation = np.concatenate([validation_features[:, index, :] for index in layer_indices], axis=1)
        model = _classifier()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(selected_train, y_train)
        probabilities = _ordered_probabilities(model, selected_validation, classes)
        predictions = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
        metrics = evaluate(y_validation, probabilities, classes, probabilities=probabilities)
        prediction_frame[f"pred::{name}"] = predictions
        for class_index, label in enumerate(classes):
            prediction_frame[f"prob::{name}::{label}"] = probabilities[:, class_index]
        validation_results.append({"classifier": name,
            "layers": [index + 1 for index in layer_indices],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"], "per_class_recall": metrics["per_class_recall"]})
        joblib.dump({"classifier": model, "layers": [index + 1 for index in layer_indices],
                     "classes": classes, "checkpoint_experiment_id": experiment_id,
                     "max_length": max_length,
                     "logistic_regression": LOGISTIC_REGRESSION_PARAMS},
                    classifier_dir / f"{name}.joblib")

    head_probabilities = np.asarray(features["validation_head_probabilities"], dtype=np.float64)
    head_metrics = evaluate(y_validation, head_probabilities, classes, probabilities=head_probabilities)
    head_predictions = np.asarray(classes, dtype=object)[head_probabilities.argmax(axis=1)]
    prediction_frame["pred::ultrabert_head"] = head_predictions
    for class_index, label in enumerate(classes):
        prediction_frame[f"prob::ultrabert_head::{label}"] = head_probabilities[:, class_index]
    validation_results.append({"classifier": "ultrabert_head", "layers": [],
        "balanced_accuracy": head_metrics["balanced_accuracy"],
        "macro_f1": head_metrics["macro_f1"], "per_class_recall": head_metrics["per_class_recall"]})

    validation_summary = pd.DataFrame(validation_results).sort_values(
        "balanced_accuracy", ascending=False, kind="stable").reset_index(drop=True)
    learned_results = validation_summary.loc[~validation_summary.classifier.eq("ultrabert_head")]
    csv_summary = validation_summary.copy()
    csv_summary["layers"] = csv_summary.layers.map(json.dumps)
    csv_summary["per_class_recall"] = csv_summary.per_class_recall.map(json.dumps)
    csv_summary.to_csv(output_dir / "validation_metrics.csv", index=False)
    temp_predictions = output_dir / "validation_predictions.tmp.parquet"
    prediction_frame.to_parquet(temp_predictions, index=False)
    temp_predictions.replace(output_dir / "validation_predictions.parquet")
    metrics_payload = {
        "experiment": "ZOD_04",
        "checkpoint_experiment_id": experiment_id,
        "checkpoint_path": str(checkpoint),
        "dataset": "v0",
        "classes": classes,
        "n_transformer_layers": N_TRANSFORMER_LAYERS,
        "n_splits": n_splits,
        "group_column": "prompt_id",
        "max_length": max_length,
        "train_rows": len(train), "validation_rows": len(validation),
        "classifier": {"preprocessing": "StandardScaler", "type": "LogisticRegression",
                       "parameters": LOGISTIC_REGRESSION_PARAMS},
        "layer_combinations": {f"best_{count}_layers": [int(x) for x in ranked_layers[:count]]
                               for count in (2, 4, 8)},
        "cv_layer_ranking": json.loads(cv_summary[["cv_rank", "layer", "mean_cv_balanced_accuracy",
                                                   "std_cv_balanced_accuracy", "oof_balanced_accuracy"]].to_json(orient="records")),
        "validation_results": json.loads(validation_summary.to_json(orient="records")),
        "best_validation_classifier_overall": str(validation_summary.iloc[0]["classifier"]),
        "best_validation_learned_classifier": str(learned_results.iloc[0]["classifier"]),
        "original_head_validation": head_metrics,
        "prediction_path": str(output_dir / "validation_predictions.parquet"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n")
    best_row = learned_results.iloc[0]
    best_name = str(best_row["classifier"])
    multi_names = {"best_2_layers", "best_4_layers", "best_8_layers", "all_24_layers"}
    multi_results = validation_summary.loc[validation_summary.classifier.isin(multi_names)]
    best_multi = multi_results.iloc[0]
    recall_tables = {
        f"Best learned classifier ({best_name})": best_row["per_class_recall"],
        f"Best multi-layer classifier ({best_multi['classifier']})": best_multi["per_class_recall"],
        "Original UltraBERT classification head": head_metrics["per_class_recall"],
    }
    _write_report(Path(report_path), experiment_id=experiment_id, checkpoint=checkpoint,
        max_length=max_length, hidden_size=int(train_features.shape[2]), cv_summary=cv_summary,
        validation_summary=validation_summary, recall_tables=recall_tables, classes=classes,
        n_splits=n_splits, n_train=len(train), n_validation=len(validation), artifact_dir=output_dir)
    return {"status": "completed", "metrics": str(output_dir / "metrics.json"),
            "feature_cache": str(feature_path), "prediction_path": str(output_dir / "validation_predictions.parquet"),
            "report": str(report_path), "best_validation_classifier": best_name,
            "best_validation_balanced_accuracy": float(best_row["balanced_accuracy"])}

