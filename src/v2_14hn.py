"""Build V2.14HN from V2.14 using only externally trained OOS predictions.

This module performs CPU inference with existing V1.1 SVC and CatBoost
checkpoints. It never fits classifiers and never uses V0 examples as mining
examples. V2.14 rows whose normalized response occurred in either checkpoint's
training data are left explicitly unscored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/v2.14hn.yaml"
DEFAULT_CLASSES = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha(frame: pd.DataFrame) -> str:
    return _sha_bytes(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def normalize_response(text: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_checkpoint(spec: dict, dataset: pd.DataFrame, v1_1: pd.DataFrame,
                      v0: pd.DataFrame) -> tuple[dict, Path, Path]:
    manifest_path = ROOT / spec["manifest"]
    manifest = _load_json(manifest_path)
    checkpoint = Path(manifest["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if manifest.get("dataset") != "v1.1" or int(manifest.get("train_rows", -1)) != len(v1_1):
        raise ValueError(f"{spec['name']} manifest does not describe complete V1.1 training")
    if manifest.get("dataset_sha256") != spec["v1_1_sha256"]:
        raise ValueError(f"{spec['name']} V1.1 dataset hash mismatch")
    if list(manifest.get("classes", [])) != DEFAULT_CLASSES:
        raise ValueError(f"{spec['name']} class order differs from canonical classes")
    trained_texts = set(v1_1.text.astype(str).map(normalize_response))
    if not trained_texts:
        raise ValueError("V1.1 training text set is empty")
    if v0.empty or v0.model.isna().any():
        raise ValueError("V0 validation reference failed integrity checks")
    # These checkpoints were trained on V1.1 only. Reject accidental V0 lineage.
    if manifest.get("validation_dataset") not in (None, "v0"):
        raise ValueError(f"unexpected validation dataset in {spec['name']} manifest")
    v0_norm = set(v0.text.astype(str).map(normalize_response))
    if trained_texts & v0_norm:
        raise ValueError(f"{spec['name']} training responses overlap V0 ({len(trained_texts & v0_norm)})")
    # Verify manifest's recorded training IDs when available.
    train_ids_hash = _sha_bytes("\n".join(v1_1.id.astype(str)).encode())
    if manifest.get("train_id_sha256") and manifest["train_id_sha256"] != train_ids_hash:
        sorted_hash = _sha_bytes("\n".join(sorted(v1_1.id.astype(str))).encode())
        if manifest["train_id_sha256"] != sorted_hash:
            raise ValueError(f"{spec['name']} training ID hash does not match V1.1")
    return manifest, checkpoint, manifest_path


def _eligible_rows(dataset: pd.DataFrame, training_texts: set[str], v0_texts: set[str]) -> np.ndarray:
    norms = dataset.text.astype(str).map(normalize_response)
    overlap_train = norms.isin(training_texts).to_numpy()
    overlap_v0 = norms.isin(v0_texts).to_numpy()
    if overlap_v0.any():
        raise ValueError(f"V2.14 overlaps {int(overlap_v0.sum())} V0 response texts")
    return ~(overlap_train | overlap_v0)


def _predict_checkpoint(name: str, checkpoint: Path, texts: list[str], batch_size: int,
                        cache_path: Path, cache_meta_path: Path, cache_key: dict,
                        classes: list[str], row_ids: np.ndarray,
                        temperature: float = 1.0) -> np.ndarray:
    """CPU-only, batched inference with a cache bound to inputs and checkpoint."""
    if cache_path.is_file() and cache_meta_path.is_file():
        meta = _load_json(cache_meta_path)
        if meta == cache_key:
            with np.load(cache_path, allow_pickle=False) as cached:
                probs, cached_ids = cached["probabilities"], cached["ids"].astype(str)
            if (probs.shape == (len(texts), len(classes)) and np.isfinite(probs).all()
                    and np.array_equal(cached_ids, row_ids.astype(str))):
                print(f"Using cached {name} OOS predictions ({len(texts):,} rows)", flush=True)
                return probs.astype(np.float32, copy=False)

    print(f"Running CPU-only {name} inference for {len(texts):,} OOS V2.14 rows", flush=True)
    import joblib
    from threadpoolctl import threadpool_limits
    from .feature_factory import stylometry
    from .inference import _readout_probabilities

    probabilities = np.empty((len(texts), len(classes)), dtype=np.float32)
    if name == "svc_v1_1":
        bundle = joblib.load(checkpoint)
        with threadpool_limits(limits=2):
            for start in range(0, len(texts), batch_size):
                end = min(len(texts), start + batch_size)
                matrix = bundle["features"].transform(texts[start:end])
                probabilities[start:end] = _readout_probabilities(
                    bundle["model"], matrix, classes).astype(np.float32)
                if end == len(texts) or end % (batch_size * 10) == 0:
                    print(f"  {name}: {end:,}/{len(texts):,}", flush=True)
    elif name == "catboost_v1_1":
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier()
        cat.load_model(str(checkpoint / "model.cbm" if checkpoint.is_dir() else checkpoint))
        model_classes = list(cat.classes_)
        if set(model_classes) != set(classes):
            raise ValueError("CatBoost checkpoint class set differs from canonical classes")
        metadata_path = checkpoint / "stylometry.json" if checkpoint.is_dir() else None
        metadata = _load_json(metadata_path) if metadata_path and metadata_path.is_file() else {}
        from scipy.special import softmax
        for start in range(0, len(texts), batch_size):
            end = min(len(texts), start + batch_size)
            raw = np.asarray(cat.predict(
                stylometry(texts[start:end], metadata.get("feature_recipe")),
                prediction_type="RawFormulaVal", thread_count=2))
            if raw.ndim == 1:
                raw = np.column_stack([-raw, raw])
            aligned = np.zeros((end - start, len(classes)), dtype=np.float64)
            for j, label in enumerate(model_classes):
                aligned[:, classes.index(label)] = raw[:, j]
            probabilities[start:end] = softmax(aligned / max(float(temperature), 1e-12), axis=1).astype(np.float32)
            if end == len(texts) or end % (batch_size * 10) == 0:
                print(f"  {name}: {end:,}/{len(texts):,}", flush=True)
    else:
        raise ValueError(f"unsupported checkpoint: {name}")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=2e-4):
        raise ValueError(f"{name} returned invalid probabilities")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=row_ids.astype(str), probabilities=probabilities)
    cache_meta_path.write_text(json.dumps(cache_key, indent=2) + "\n", encoding="utf-8")
    return probabilities


def score_hardness(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                   classes: list[str], confidence_threshold: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add transparent row-level difficulty labels and return probability audit."""
    n = len(frame)
    names = list(predictions)
    if any(values.shape != (n, len(classes)) for values in predictions.values()):
        raise ValueError("prediction arrays must be aligned to frame rows and canonical labels")
    labels = frame.model.astype(str).to_numpy()
    idx = {label: i for i, label in enumerate(classes)}
    true_idx = np.fromiter((idx[label] for label in labels), dtype=np.int32, count=n)
    top_labels, top_conf, true_probs, max_wrong_probs = {}, {}, {}, {}
    wrong_count = np.zeros(n, dtype=np.int16)
    family_errors = np.zeros(n, dtype=np.int16)
    wrong_votes = np.zeros((n, len(classes)), dtype=np.int16)
    wrong_confidence_sums = np.zeros((n, len(classes)), dtype=np.float64)
    confident_wrong_votes = np.zeros((n, len(classes)), dtype=np.int16)
    confident_wrong = np.zeros(n, dtype=bool)
    for name, probs in predictions.items():
        best = np.argmax(probs, axis=1)
        confidence = probs[np.arange(n), best]
        correct = best == true_idx
        wrong_count += ~correct
        family_errors += (~correct).astype(np.int16)  # one independent family per candidate
        confident_wrong |= (~correct) & (confidence >= confidence_threshold)
        top_labels[name] = np.asarray(classes, dtype=object)[best]
        top_conf[name] = confidence
        true_probs[name] = probs[np.arange(n), true_idx]
        masked = probs.copy()
        masked[np.arange(n), true_idx] = -np.inf
        max_wrong_probs[name] = masked.max(axis=1)
        for label_i in range(len(classes)):
            mask = (best == label_i) & ~correct
            wrong_votes[mask, label_i] += 1
            wrong_confidence_sums[mask, label_i] += confidence[mask]
            confident_mask = mask & (confidence >= confidence_threshold)
            confident_wrong_votes[confident_mask, label_i] += 1

    max_wrong = np.maximum.reduce(list(max_wrong_probs.values()))
    most_wrong = np.full(n, "", dtype=object)
    most_confident_wrong = np.full(n, "", dtype=object)
    for row in range(n):
        if wrong_votes[row].max() > 0:
            candidates = np.flatnonzero(wrong_votes[row] == wrong_votes[row].max())
            # Tie: higher summed top-1 confidence, then canonical class order.
            chosen = sorted(candidates, key=lambda j: (-wrong_confidence_sums[row, j], j))[0]
            most_wrong[row] = classes[chosen]
        if confident_wrong_votes[row].max() > 0:
            candidates = np.flatnonzero(confident_wrong_votes[row] == confident_wrong_votes[row].max())
            chosen = sorted(candidates, key=lambda j: (-wrong_confidence_sums[row, j], j))[0]
            most_confident_wrong[row] = classes[chosen]

    consensus_high = np.zeros(n, dtype=bool)
    if len(names) >= 2:
        first, second = names[:2]
        consensus_high = ((top_labels[first] == top_labels[second])
                          & (top_labels[first] != labels)
                          & (top_conf[first] >= confidence_threshold)
                          & (top_conf[second] >= confidence_threshold))
    error_families = family_errors
    has_oos = len(names) > 0
    categories = np.full(n, "unscored_no_oos_predictions", dtype=object)
    raw_weight = np.ones(n, dtype=np.float64)
    if has_oos:
        categories[:] = "regular"
        one_error = wrong_count == 1
        several_families = error_families >= 2
        categories[one_error] = "single_model_error"
        raw_weight[one_error] = 2.0
        categories[several_families] = "multi_family_error"
        raw_weight[several_families] = 3.0
        categories[consensus_high] = "high_confidence_consensus_error"
        raw_weight[consensus_high] = 4.0

    out = frame.copy()
    out["mining_oos_prediction_count"] = len(names)
    out["mining_misclassified_classifier_count"] = wrong_count
    out["mining_misclassified_family_count"] = error_families
    out["mining_most_frequent_wrong_label"] = pd.Series(most_wrong).replace("", pd.NA)
    out["mining_hard_positive"] = confident_wrong
    out["mining_hard_negative_label"] = pd.Series(most_confident_wrong).replace("", pd.NA)
    out["mining_mean_true_label_probability"] = (
        np.mean(np.column_stack(list(true_probs.values())), axis=1) if names
        else np.full(n, np.nan))
    out["mining_highest_incorrect_label_probability"] = (
        max_wrong if names else np.full(n, np.nan))
    out["mining_high_confidence_consensus_error"] = consensus_high
    out["hardness_category"] = categories
    out["sampling_weight_raw"] = raw_weight
    means = out.assign(_weight=raw_weight).groupby("model", sort=True)["_weight"].transform("mean").to_numpy()
    out["sampling_weight"] = raw_weight / means
    out["mining_oos_scored"] = len(names) > 0
    for name in names:
        out[f"mining_pred::{name}"] = top_labels[name]
        out[f"mining_confidence::{name}"] = top_conf[name]
        out[f"mining_true_probability::{name}"] = true_probs[name]
        out[f"mining_highest_wrong_probability::{name}"] = max_wrong_probs[name]

    audit_columns = ["id", "prompt_id", "model", "hardness_category", "sampling_weight",
                     "mining_oos_prediction_count", "mining_misclassified_classifier_count",
                     "mining_misclassified_family_count", "mining_most_frequent_wrong_label",
                     "mining_hard_positive", "mining_hard_negative_label",
                     "mining_mean_true_label_probability", "mining_highest_incorrect_label_probability"]
    audit = out[audit_columns].copy()
    for name, probs in predictions.items():
        for j, label in enumerate(classes):
            audit[f"prob::{name}::{label}"] = probs[:, j]
    return out, audit


def _class_counts(frame: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    counts = frame.model.astype(str).value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in classes}


def _resample(frame: pd.DataFrame, classes: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for label in classes:
        part = frame.loc[frame.model.astype(str).eq(label)].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"cannot resample missing class {label}")
        probs = part.sampling_weight.to_numpy(dtype=np.float64)
        probs = probs / probs.sum()
        chosen = rng.choice(len(part), size=len(part), replace=True, p=probs)
        sample = part.iloc[chosen].copy().reset_index(drop=True)
        occurrence = pd.Series(sample.groupby("source_id", sort=False).cumcount().to_numpy() + 1)
        sample["source_id"] = sample.id.astype(str).to_numpy()
        sample["id"] = [f"{source}::hnr::{int(number):06d}"
                        for source, number in zip(sample.source_id, occurrence)]
        sample["source_sampling_weight"] = sample.sampling_weight.astype(float)
        sample["sampling_weight"] = 1.0
        pieces.append(sample)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _write_profile(frame: pd.DataFrame, version: str, output_path: Path,
                   source_path: Path, source_profile_path: Path, metadata: dict) -> dict:
    classes = sorted(frame.model.astype(str).unique())
    profile = {
        "dataset_version": version,
        "dataset_sha256": _frame_sha(frame),
        "rows": int(len(frame)),
        "classes": classes,
        "class_counts": _class_counts(frame, classes),
        "prompt_groups": int(frame.prompt_id.astype(str).nunique()),
        "unique_sources": int(frame.source_dataset.nunique()) if "source_dataset" in frame else None,
        "training_only": True,
        "validation_dataset_version": None,
        "source_dataset_path": str(source_path.relative_to(ROOT)),
        "source_dataset_sha256": _sha_file(source_path),
        "source_dataset_profile": str(source_profile_path.relative_to(ROOT)),
        **metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return profile


def build(config_path: Path = DEFAULT_CONFIG, *, force: bool = False) -> dict:
    import yaml
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / cfg["input_dataset"]
    profile_path = ROOT / cfg["input_profile"]
    v1_1_path = ROOT / cfg["mining_training_dataset"]
    v0_path = ROOT / cfg["validation_dataset"]
    out_path = ROOT / cfg["output_dataset"]
    out_profile_path = ROOT / cfg["output_profile"]
    resampled_path = ROOT / cfg["resampled_dataset"]
    resampled_profile_path = ROOT / cfg["resampled_profile"]
    artifact_dir = ROOT / cfg["artifact_dir"]
    for path in (data_path, profile_path, v1_1_path, v0_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (out_path.exists() or out_profile_path.exists() or resampled_path.exists() or resampled_profile_path.exists()) and not force:
        raise FileExistsError("V2.14HN outputs already exist; refusing to overwrite")

    original = pd.read_parquet(data_path)
    v1_1 = pd.read_parquet(v1_1_path)
    v0 = pd.read_parquet(v0_path)
    source_profile = _load_json(profile_path)
    v1_1_profile = _load_json(ROOT / "data/processed/dataset_profile_v1.1.json")
    v0_profile = _load_json(ROOT / "data/processed/dataset_profile.json")
    required = {"id", "prompt_id", "text", "model"}
    if required - set(original.columns) or required - set(v1_1.columns) or required - set(v0.columns):
        raise ValueError("source, mining-train and V0 artifacts must contain id,prompt_id,text,model")
    if original.id.duplicated().any() or original["model"].isna().any() or original.text.isna().any():
        raise ValueError("V2.14 must have unique IDs and complete text/labels")
    classes = list(source_profile.get("classes", []))
    if classes != DEFAULT_CLASSES or set(original.model.astype(str)) != set(classes):
        raise ValueError("V2.14 class roster differs from the canonical 18 labels")
    if source_profile.get("dataset_sha256") != cfg["input_dataset_sha256"]:
        raise ValueError("V2.14 source hash differs from the configuration")
    if v1_1_profile.get("dataset_sha256") != cfg["v1_1_dataset_sha256"]:
        raise ValueError("V1.1 source hash differs from the configuration")
    if v0_profile.get("dataset_sha256") != cfg["v0_dataset_sha256"]:
        raise ValueError("V0 validation source hash differs from the configuration")

    v1_1_texts = set(v1_1.text.astype(str).map(normalize_response))
    v0_texts = set(v0.text.astype(str).map(normalize_response))
    if v1_1_texts & v0_texts:
        raise ValueError("V1.1 training dataset overlaps V0 response text")
    eligible_mask = _eligible_rows(original, v1_1_texts, v0_texts)
    eligible = original.loc[eligible_mask].copy().reset_index(drop=True)
    excluded_count = len(original) - len(eligible)

    checkpoint_specs = cfg["checkpoints"]
    if len({spec["family"] for spec in checkpoint_specs}) != len(checkpoint_specs):
        raise ValueError("this build expects one classifier per independent model family")
    checkpoints = {}
    validation_provenance = {}
    v0_split = pd.read_parquet(ROOT / "data/processed/splits_v0.parquet")
    v0_validation_ids = set(v0_split.loc[v0_split.split.astype(str).eq("validation"), "id"].astype(str))
    for spec in checkpoint_specs:
        manifest, checkpoint, manifest_path = _check_checkpoint(spec, original, v1_1, v0)
        prediction_path = Path(manifest["prediction_path"])
        if not prediction_path.is_absolute():
            prediction_path = ROOT / prediction_path
        if not prediction_path.is_file():
            raise FileNotFoundError(f"checkpoint validation predictions missing: {prediction_path}")
        validation_predictions = pd.read_parquet(prediction_path, columns=["id"])
        prediction_ids = set(validation_predictions.id.astype(str))
        if prediction_ids != v0_validation_ids:
            raise ValueError(f"{spec['name']} validation prediction IDs do not equal frozen V0 holdout")
        validation_provenance[spec["name"]] = {
            "rows": int(len(prediction_ids)),
            "validation_ids_sha256": _sha_bytes("\n".join(sorted(prediction_ids)).encode()),
            "source": str(prediction_path.relative_to(ROOT)),
        }
        checkpoints[spec["name"]] = (spec, manifest, checkpoint, manifest_path)
    model_names = [spec["name"] for spec in checkpoint_specs]
    training_text_set = v1_1_texts
    if len(eligible):
        # Both models share the same V1.1 training corpus; exclusion is based on
        # normalized response content, not a dataset-version label alone.
        if any(normalize_response(t) in training_text_set for t in eligible.text.astype(str)):
            raise RuntimeError("eligible V2.14 rows leaked into V1.1 training responses")

    classes = list(classes)
    eligible_ids = eligible.id.astype(str).to_numpy()
    eligible_hash = _sha_bytes("\n".join(eligible_ids).encode())
    predictions = {}
    for spec in checkpoint_specs:
        name, manifest, checkpoint, manifest_path = (
            spec["name"], *checkpoints[spec["name"]][1:])
        cache_base = artifact_dir / "predictions" / name
        cache_path = cache_base.with_suffix(".npz")
        cache_meta_path = cache_base.with_suffix(".json")
        cache_key = {
            "checkpoint_id": manifest["experiment_id"],
            "checkpoint_manifest_sha256": _sha_file(manifest_path),
            "eligible_ids_sha256": eligible_hash,
            "dataset_sha256": cfg["input_dataset_sha256"],
            "classes": classes,
            "prediction_type": "softmax_probabilities",
            # Do not import a historical temperature fitted on V0 into mining.
            "temperature": 1.0,
            "device": "cpu",
        }
        predictions[name] = _predict_checkpoint(
            name, checkpoint, eligible.text.astype(str).tolist(),
            int(cfg.get("batch_size", 1024)), cache_path, cache_meta_path, cache_key, classes, eligible_ids,
            1.0)

    eligible_y = eligible.model.astype(str).to_numpy()
    model_metrics, top_predictions, top_confidences = {}, {}, {}
    class_array = np.asarray(classes, dtype=object)
    true_indices = np.fromiter((classes.index(label) for label in eligible_y),
                               dtype=np.int32, count=len(eligible_y))
    for name, probs in predictions.items():
        top_i = np.argmax(probs, axis=1)
        pred_labels = class_array[top_i]
        confidence = probs[np.arange(len(probs)), top_i]
        top_predictions[name] = pred_labels
        top_confidences[name] = confidence
        model_metrics[name] = {
            "rows": int(len(probs)),
            "accuracy": float(np.mean(pred_labels == eligible_y)),
            "errors": int(np.count_nonzero(pred_labels != eligible_y)),
            "mean_true_label_probability": float(np.mean(probs[np.arange(len(probs)), true_indices])),
        }
    agreement_wrong = np.zeros(len(eligible), dtype=bool)
    agreement_counts = {}
    if len(model_names) == 2:
        first, second = model_names
        agreement_wrong = ((top_predictions[first] == top_predictions[second])
                           & (top_predictions[first] != eligible_y))
        for threshold in (0.50, 0.60, 0.70, 0.80):
            agreement_counts[f"{threshold:.2f}"] = int(np.count_nonzero(
                agreement_wrong & (top_confidences[first] >= threshold)
                & (top_confidences[second] >= threshold)))
    scored_rows = original.loc[eligible_mask].copy().reset_index(drop=True)
    scored_frame, scored_audit = score_hardness(
        scored_rows, predictions, classes,
        float(cfg.get("high_confidence_threshold", 0.80)))
    all_rows = original.copy()
    metadata_cols = [
        "mining_oos_prediction_count", "mining_misclassified_classifier_count",
        "mining_misclassified_family_count", "mining_most_frequent_wrong_label",
        "mining_hard_positive", "mining_hard_negative_label",
        "mining_mean_true_label_probability", "mining_highest_incorrect_label_probability",
        "mining_high_confidence_consensus_error", "hardness_category", "sampling_weight_raw",
        "sampling_weight", "mining_oos_scored",
    ]
    for name in model_names:
        metadata_cols.extend([f"mining_pred::{name}", f"mining_confidence::{name}",
                              f"mining_true_probability::{name}",
                              f"mining_highest_wrong_probability::{name}"])
    for column in metadata_cols:
        all_rows[column] = pd.NA
    for column in metadata_cols:
        all_rows.loc[eligible_mask, column] = scored_frame[column].to_numpy()
    all_rows["mining_oos_prediction_count"] = all_rows["mining_oos_prediction_count"].fillna(0).astype("int16")
    for column in ("mining_misclassified_classifier_count", "mining_misclassified_family_count"):
        all_rows[column] = all_rows[column].fillna(0).astype("int16")
    all_rows["mining_high_confidence_consensus_error"] = all_rows["mining_high_confidence_consensus_error"].fillna(False).astype(bool)
    all_rows["mining_hard_positive"] = all_rows["mining_hard_positive"].fillna(False).astype(bool)
    all_rows["hardness_category"] = all_rows["hardness_category"].fillna("unscored_no_oos_predictions")
    all_rows["sampling_weight_raw"] = all_rows["sampling_weight_raw"].fillna(1.0).astype(float)
    all_rows["mining_oos_scored"] = all_rows["mining_oos_scored"].fillna(False).astype(bool)
    means_by_label = all_rows.groupby("model", sort=False).sampling_weight_raw.transform("mean")
    all_rows["sampling_weight"] = all_rows.sampling_weight_raw / means_by_label
    if all_rows.id.duplicated().any() or set(all_rows.model.astype(str)) != set(classes):
        raise RuntimeError("dataset row identity or class preservation check failed")
    if len(all_rows) != len(original) or not all_rows.text.astype(str).equals(original.text.astype(str)):
        raise RuntimeError("V2.14HN must preserve all original rows and texts")
    if np.any(all_rows.sampling_weight.to_numpy(dtype=float) <= 0):
        raise RuntimeError("all sampling weights must be positive")
    per_label_weight_sums = all_rows.groupby("model").sampling_weight.sum()
    per_label_counts = all_rows.model.value_counts().sort_index()
    if not np.allclose(per_label_weight_sums.sort_index(), per_label_counts.astype(float), atol=1e-8):
        raise RuntimeError("sampling weights must sum to original count within each true label")
    if set(all_rows.id.astype(str)) & set(v0.id.astype(str)):
        raise RuntimeError("V2.14HN contains V0 response IDs")
    if set(all_rows.text.astype(str).map(normalize_response)) & v0_texts:
        raise RuntimeError("V2.14HN contains normalized V0 response text")

    category_counts = {str(k): int(v) for k, v in all_rows.hardness_category.value_counts().items()}
    scored_by_label = eligible.model.astype(str).value_counts().to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_parquet(out_path, index=False)
    profile = _write_profile(all_rows, cfg.get("dataset_version", "v2.14HN"),
        out_profile_path, data_path, profile_path, {
            "source_dataset_version": "v2.14",
            "source_dataset_sha256": cfg["input_dataset_sha256"],
            "mining_training_dataset": "v1.1",
            "mining_training_dataset_sha256": cfg["v1_1_dataset_sha256"],
            "v0_validation_dataset_sha256": cfg["v0_dataset_sha256"],
            "v0_used_as_mining_data": False,
            "v0_used_for_mining_checkpoint_selection": True,
            "checkpoint_validation_provenance": validation_provenance,
            "prediction_checkpoint_ids": {n: checkpoints[n][1]["experiment_id"] for n in model_names},
            "prediction_families": {n: checkpoints[n][0]["family"] for n in model_names},
            "prediction_checkpoint_manifests": {
                n: str(checkpoints[n][3].relative_to(ROOT)) for n in model_names},
            "prediction_checkpoint_manifest_sha256": {
                n: _sha_file(checkpoints[n][3]) for n in model_names},
            "prediction_calibration_temperatures": {
                n: 1.0 for n in model_names},
            "prediction_calibration_policy": "SVC train-internal probability calibration; CatBoost raw softmax; no V0-fitted calibration",
            "oos_normalization": "Unicode NFKC, case-fold, collapse whitespace",
            "oos_scored_rows": int(len(eligible)),
            "unscored_rows": int(excluded_count),
            "unscored_reason": "normalized response was present in V1.1 checkpoint training corpus",
            "high_confidence_threshold": float(cfg.get("high_confidence_threshold", 0.80)),
            "hardness_sampling_weights": {"regular": 1, "single_model_error": 2,
                "multi_family_error": 3, "high_confidence_consensus_error": 4,
                "unscored_no_oos_predictions": 1},
            "hardness_category_counts": category_counts,
            "oos_classifier_metrics": model_metrics,
            "same_wrong_label_family_agreement_rows": int(agreement_wrong.sum()),
            "same_wrong_label_high_confidence_counts": agreement_counts,
            "sampling_weight_normalization": "mean weight equals 1 separately within each true label",
            "sampling_weight_sums_by_class": {str(k): float(v) for k, v in per_label_weight_sums.items()},
            "oos_scored_rows_by_class": {label: int(scored_by_label.get(label, 0)) for label in classes},
            "source_rows_retained": True,
            "source_dataset_profile_path": str(profile_path.relative_to(ROOT)),
            "build_config_path": str(config_path.relative_to(ROOT)),
            "build_config_sha256": _sha_file(config_path),
            "build_script_sha256": _sha_file(Path(__file__).resolve()),
            "seed": int(cfg.get("seed", 2026)),
        })
    # Resample each true class to its original count with probabilities
    # proportional to within-class normalized weights. This is the drop-in
    # physical-row option for trainers that do not consume sample weights.
    resample_input = all_rows.copy()
    resample_input["source_id"] = resample_input.id.astype(str)
    resampled = _resample(resample_input, classes, int(cfg.get("seed", 2026)))
    if _class_counts(resampled, classes) != _class_counts(original, classes):
        raise RuntimeError("weighted resampling changed per-class row counts")
    if resampled.id.duplicated().any() or (resampled.sampling_weight != 1.0).any():
        raise RuntimeError("resampled export must have unique IDs and unit row weights")
    resampled.to_parquet(resampled_path, index=False)
    resampled_profile = _write_profile(resampled, cfg.get("resampled_version", "v2.14HN-resampled"),
        resampled_profile_path, data_path, profile_path, {
            "parent_dataset": "v2.14HN", "parent_dataset_sha256": profile["dataset_sha256"],
            "resampling_seed": int(cfg.get("seed", 2026)),
            "resampling_method": "per-label replacement sampling proportional to sampling_weight",
            "sampling_weight_column": "source_sampling_weight",
            "physical_training_weight_column": "sampling_weight",
            "v0_used_as_mining_data": False,
        })

    # Full OOS prediction audit contains probabilities only for the safe rows.
    full_audit = scored_audit.copy()
    audit_path = artifact_dir / "oos_prediction_audit.parquet"
    full_audit.to_parquet(audit_path, index=False)
    pair_rows = []
    for name, probs in predictions.items():
        pred = np.asarray(classes, dtype=object)[np.argmax(probs, axis=1)]
        for true_label, predicted_label in zip(scored_rows.model.astype(str), pred):
            if true_label != predicted_label:
                pair_rows.append({"checkpoint": name, "family": checkpoints[name][0]["family"],
                                  "true_label": true_label, "predicted_label": predicted_label})
    pairs = (pd.DataFrame(pair_rows).groupby(
        ["checkpoint", "family", "true_label", "predicted_label"], as_index=False).size()
        if pair_rows else pd.DataFrame(columns=["checkpoint", "family", "true_label", "predicted_label", "size"]))
    pairs.to_csv(artifact_dir / "directed_confusions.csv", index=False, lineterminator="\n")
    scored_audit.to_parquet(artifact_dir / "hardness_audit.parquet", index=False)

    category_counts = {str(k): int(v) for k, v in all_rows.hardness_category.value_counts().items()}
    scored_by_label = eligible.model.astype(str).value_counts().to_dict()
    weight_summary = all_rows.groupby("model").sampling_weight.agg(["min", "median", "max"])
    label_lines = []
    confusion_counts = (pairs.groupby(["true_label", "predicted_label"], as_index=False)["size"].sum()
                        if not pairs.empty else pd.DataFrame(columns=["true_label", "predicted_label", "size"]))
    for label in classes:
        ranked_pairs = confusion_counts.loc[confusion_counts.true_label.eq(label)].sort_values(
            ["size", "predicted_label"], ascending=[False, True]).head(3)
        top_text = ", ".join(f"{r.predicted_label} ({int(r['size']):,})" for _, r in ranked_pairs.iterrows()) or "—"
        ws = weight_summary.loc[label]
        label_lines.append(
            f"| {label} | {int(original.model.astype(str).eq(label).sum()):,} | "
            f"{int(scored_by_label.get(label, 0)):,} | {int(original.model.astype(str).eq(label).sum() - scored_by_label.get(label, 0)):,} | "
            f"{top_text} | {ws['min']:.3f} / {ws['median']:.3f} / {ws['max']:.3f} |")
    confusion_lines = []
    for _, row in confusion_counts.sort_values(["size", "true_label", "predicted_label"],
                                                ascending=[False, True, True]).head(25).iterrows():
        confusion_lines.append(f"| {row.true_label} → {row.predicted_label} | {int(row['size']):,} |")
    report_path = ROOT / cfg.get("report", "docs/v2.14HN.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V2.14HN — Hard-Negative Dataset", "",
        "V2.14HN preserves all V2.14 rows and source text. Existing V1.1 SVC and CatBoost checkpoints score only rows whose normalized responses are absent from their complete V1.1 training corpus. V0 examples are not mined or added to the dataset. The historical checkpoints were selected/validated on the frozen 540-row V0 holdout; that indirect selection exposure is a limitation. No classifier was trained.", "",
        "## Coverage and row counts", "",
        f"- Source: `{cfg['input_dataset']}` — {len(original):,} rows, 18 labels.",
        f"- OOS-scored rows: {len(eligible):,} ({len(eligible)/max(1,len(original)):.1%}).",
        f"- Unscored rows: {excluded_count:,} ({excluded_count/max(1,len(original)):.1%}), because their normalized responses occurred in the V1.1 training corpus.",
        f"- V0 overlap check used the full {len(v0):,}-row V0 file; the historical checkpoint validation artifacts contain the frozen 540-row V0 holdout. V0 mining examples used: 0.",
        f"- Model families: {', '.join(spec['family'] for spec in checkpoint_specs)}. Threshold for a high-confidence shared error: {cfg.get('high_confidence_threshold', 0.80):.0%} top-1 probability from both families.",
        "- Probability handling: SVC uses its training-internal calibrated probabilities; CatBoost uses raw softmax. No temperature or calibration fitted on V0 is used to score mining examples.",
        "- Both checkpoint validation prediction files were verified to use exactly the historical 540-row V0 holdout. V0 is therefore used for checkpoint validation, but not as a mining example source or as a per-row hardness target.",
        "- Main dataset: `data/processed/dataset_v2.14HN.parquet`; physical weighted-resampling option: `data/processed/dataset_v2.14HN-resampled.parquet`.",
        "- The resampled export preserves each original class count and samples with replacement proportional to within-class `sampling_weight`; repeated rows receive unique IDs and retain the source ID.", "",
        "The neural SFT trainer supports per-row replacement sampling directly. To train from the original rows, set `weighted_sampling: true` in the model YAML and select `v2.14HN`:",
        "```bash",
        "python -m src.pipeline models train --model ultrabert380 --dataset v2.14HN --validation-dataset v0 --accept-config-defaults",
        "```",
        "The sampler uses replacement, draws exactly one dataset-length epoch, and records expected and observed class sampling distributions in the run manifest. Validation remains unweighted. The physical resampled export remains available for workflows that do not use the neural SFT trainer.", "",
        "## Hardness categories", "",
        "| Category | Rows | Sampling weight before class normalization |", "|---|---:|---:|",
    ]
    raw_weights = {"regular": 1, "single_model_error": 2, "multi_family_error": 3,
                   "high_confidence_consensus_error": 4, "unscored_no_oos_predictions": 1}
    for category in ["regular", "single_model_error", "multi_family_error",
                     "high_confidence_consensus_error", "unscored_no_oos_predictions"]:
        lines.append(f"| `{category}` | {category_counts.get(category, 0):,} | {raw_weights[category]} |")
    lines.extend([
        "", "`mining_hard_positive` flags a true-label example confidently assigned to another label by at least one family. `mining_hard_negative_label` identifies the most frequent confidently predicted wrong label for that row, if any. A multi-family error means both independent families predicted the row incorrectly; the consensus category further requires both to predict the same wrong label at or above the configured confidence threshold. Category weights are divided by the mean raw weight within each true label. Thus each label's normalized weights sum to its original row count, and every per-row weight remains positive.", "",
        "## OOS classifier performance and agreement", "",
        "| Checkpoint | OOS accuracy | Incorrect rows | Mean probability on true label |", "|---|---:|---:|---:|",
        *[f"| `{name}` | {model_metrics[name]['accuracy']:.2%} | {model_metrics[name]['errors']:,} | {model_metrics[name]['mean_true_label_probability']:.3f} |" for name in model_names],
        "",
        f"The two families predicted the same wrong top-1 label on {int(agreement_wrong.sum()):,} OOS rows. Both family confidences were at least 50% on {agreement_counts.get('0.50', 0):,}, 60% on {agreement_counts.get('0.60', 0):,}, 70% on {agreement_counts.get('0.70', 0):,}, and 80% on {agreement_counts.get('0.80', 0):,}. The high-confidence consensus category uses the configured 80% threshold.",
        "",
        "## Per-label coverage and confusion", "",
        "| True label | V2.14 rows | OOS-scored | Unscored | Most frequent wrong labels (count across OOS classifiers) | Normalized weight min / median / max |",
        "|---|---:|---:|---:|---|---:|", *label_lines,
        "", "### Most frequent directed confusions", "",
        "Counts pool OOS top-1 errors from SVC and CatBoost; the complete per-checkpoint table is in `artifacts/analysis/v2.14HN/directed_confusions.csv`.", "",
        "| True label → predicted label | Error votes |", "|---|---:|", *(confusion_lines or ["| No OOS errors | 0 |"]),
        "", "## Reproducibility and lineage", "",
        f"- Seed: `{cfg.get('seed', 2026)}`. Source V2.14 SHA-256: `{cfg['input_dataset_sha256']}`.",
        f"- Mining-model training dataset: V1.1 SHA-256 `{cfg['v1_1_dataset_sha256']}`. Checkpoint IDs: " + ", ".join(f"`{name}` = `{checkpoints[name][1]['experiment_id']}`" for name in model_names) + ".",
        "- OOS eligibility uses NFKC normalization, case-folding and whitespace collapse. Any V2.14 response matching a V1.1 training response is excluded from model scoring, but retained in V2.14HN as unscored with base weight 1.",
        "- Full row-level probabilities and hardness decisions: `artifacts/analysis/v2.14HN/oos_prediction_audit.parquet` and `hardness_audit.parquet`.",
        "- Deterministic weighted resampling lineage is recorded in `data/processed/dataset_profile_v2.14HN-resampled.json`.",
        "- Validation command: `python -m src.v2_14hn validate`.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results = {
        "dataset": str(out_path.relative_to(ROOT)),
        "profile": str(out_profile_path.relative_to(ROOT)),
        "resampled_dataset": str(resampled_path.relative_to(ROOT)),
        "resampled_profile": str(resampled_profile_path.relative_to(ROOT)),
        "rows": len(all_rows), "oos_scored_rows": len(eligible), "unscored_rows": excluded_count,
        "class_counts": _class_counts(all_rows, classes),
        "hardness_counts": {str(k): int(v) for k, v in all_rows.hardness_category.value_counts().items()},
        "oos_classifier_metrics": model_metrics,
        "same_wrong_label_family_agreement_rows": int(agreement_wrong.sum()),
        "same_wrong_label_high_confidence_counts": agreement_counts,
        "checkpoint_validation_provenance": validation_provenance,
        "prediction_checkpoint_ids": {n: checkpoints[n][1]["experiment_id"] for n in model_names},
        "prediction_audit": str(audit_path.relative_to(ROOT)),
        "hardness_audit": str((artifact_dir / "hardness_audit.parquet").relative_to(ROOT)),
        "directed_confusions": str((artifact_dir / "directed_confusions.csv").relative_to(ROOT)),
        "dataset_sha256": profile["dataset_sha256"],
        "resampled_dataset_sha256": resampled_profile["dataset_sha256"],
        "build_config_sha256": _sha_file(config_path),
        "build_script_sha256": _sha_file(Path(__file__).resolve()),
        "report": str(report_path.relative_to(ROOT)),
    }
    (artifact_dir / "build_manifest.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def validate(config_path: Path = DEFAULT_CONFIG) -> dict:
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / cfg["output_dataset"]
    resampled_path = ROOT / cfg["resampled_dataset"]
    profile_path = ROOT / cfg["output_profile"]
    resampled_profile_path = ROOT / cfg["resampled_profile"]
    original = pd.read_parquet(ROOT / cfg["input_dataset"])
    v0 = pd.read_parquet(ROOT / cfg["validation_dataset"])
    frame = pd.read_parquet(dataset_path)
    resampled = pd.read_parquet(resampled_path)
    profile = _load_json(profile_path)
    resampled_profile = _load_json(resampled_profile_path)
    classes = list(profile["classes"])
    if len(frame) != len(original) or not frame.text.astype(str).equals(original.text.astype(str)):
        raise ValueError("V2.14HN did not retain every original V2.14 row/text")
    if set(frame.model.astype(str)) != set(DEFAULT_CLASSES):
        raise ValueError("V2.14HN does not contain all 18 canonical labels")
    if set(frame.id.astype(str)) & set(v0.id.astype(str)):
        raise ValueError("V0 IDs overlap V2.14HN")
    v0_texts = set(v0.text.astype(str).map(normalize_response))
    if set(frame.text.astype(str).map(normalize_response)) & v0_texts:
        raise ValueError("V0 response text overlaps V2.14HN")
    if frame.sampling_weight.isna().any() or (frame.sampling_weight <= 0).any():
        raise ValueError("sampling weights are missing or non-positive")
    sums = frame.groupby("model").sampling_weight.sum().sort_index()
    counts = frame.model.value_counts().sort_index().astype(float)
    if not np.allclose(sums, counts, atol=1e-8):
        raise ValueError("per-label sampling weights are not normalized")
    if frame.mining_oos_scored.astype(bool).sum() != int(profile["oos_scored_rows"]):
        raise ValueError("profile OOS row count differs from dataset")
    if resampled.id.duplicated().any() or not np.allclose(resampled.sampling_weight, 1.0):
        raise ValueError("weighted-resampling export must have unique IDs and unit weights")
    if _class_counts(frame, classes) != _class_counts(resampled, classes):
        raise ValueError("weighted-resampling export changed class counts")
    if profile["dataset_sha256"] != _frame_sha(frame) or resampled_profile["dataset_sha256"] != _frame_sha(resampled):
        raise ValueError("profile content hash mismatch")
    return {"status": "ok", "rows": len(frame), "oos_scored_rows": int(frame.mining_oos_scored.sum()),
            "unscored_rows": int((~frame.mining_oos_scored).sum()),
            "classes": len(set(frame.model.astype(str))), "resampled_rows": len(resampled)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "validate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true",
                        help="overwrite V2.14HN outputs after recomputing/cache-validating predictions")
    args = parser.parse_args(argv)
    result = build(args.config, force=args.force) if args.action == "build" else validate(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
