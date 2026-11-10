"""Classical and optional neural model adapters with aligned outputs."""
from __future__ import annotations

import time
import json
import math
import os
import inspect
import hashlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from .feature_factory import (DEFAULT_STYLOMETRY_RECIPE, STYLOMETRY_VERSION,
                              FoldFeatures, load_or_fit, stylometry)


def _aligned(values, fitted_classes, classes):
    result = np.zeros((len(values), len(classes)), dtype=float)
    indices = {label: i for i, label in enumerate(classes)}
    for j, label in enumerate(fitted_classes):
        result[:, indices[label]] = values[:, j]
    return result


def _calibration_indices(labels, *, max_samples=5000, seed=2026, full_dataset=False):
    """Deterministic class-stratified training-only calibration sample."""
    labels = np.asarray(labels, dtype=object)
    if full_dataset or max_samples is None or len(labels) <= int(max_samples):
        return np.arange(len(labels), dtype=int)
    quota = max(1, int(max_samples))
    rng = np.random.default_rng(int(seed))
    selected = []
    groups = {label: np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))}
    if quota < len(groups):
        raise ValueError(f"calibration_max_samples={quota} cannot represent all {len(groups)} classes")
    # Allocate proportional quotas by largest remainder, with at least one per class.
    raw = {label: quota * len(indices) / len(labels) for label, indices in groups.items()}
    counts = {label: min(len(indices), max(1, int(np.floor(raw[label]))))
              for label, indices in groups.items()}
    while sum(counts.values()) > quota:
        choices = [label for label in counts if counts[label] > 1]
        if not choices: break
        counts[max(choices, key=lambda label: (counts[label] - raw[label], counts[label]))] -= 1
    while sum(counts.values()) < quota:
        choices = [label for label in counts if counts[label] < len(groups[label])]
        if not choices: break
        counts[max(choices, key=lambda label: (raw[label] - counts[label], len(groups[label]) - counts[label]))] += 1
    for label, indices in groups.items():
        selected.extend(rng.choice(indices, size=counts[label], replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _fit_temperature(train_logits, train_labels, classes, *, max_samples=5000,
                     seed=2026, full_dataset=False, return_metrics=False):
    """Fit temperature using a deterministic stratified sample of training rows only."""
    all_logits = np.asarray(train_logits, dtype=float)
    indexes = _calibration_indices(train_labels, max_samples=max_samples, seed=seed,
                                   full_dataset=full_dataset)
    logits = all_logits[indexes]
    sample_labels = np.asarray(train_labels, dtype=object)[indexes].tolist()
    indexes = {label: i for i, label in enumerate(classes)}
    targets = np.asarray([indexes[label] for label in sample_labels], dtype=int)
    def loss(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))
    result = minimize_scalar(loss, bounds=(-3.0, 3.0), method="bounded")
    temperature = float(np.exp(result.x))
    if not return_metrics:
        return temperature
    raw_loss = loss(0.0)
    calibrated_loss = loss(float(result.x))
    return temperature, {"calibration_sample_size": int(len(sample_labels)),
                         "calibration_full_dataset": bool(full_dataset),
                         "calibration_seed": int(seed),
                         "calibration_raw_loss": float(raw_loss),
                         "calibration_calibrated_loss": float(calibrated_loss)}


def _length_order(lengths):
    """Stable short-to-long order for dynamic-padding inference batches."""
    return np.argsort(np.asarray(lengths, dtype=np.int64), kind="stable")


def _restore_prediction_order(sorted_predictions, sorted_indices):
    """Map predictions generated in a sorted batch order back to source rows."""
    predictions = np.asarray(sorted_predictions)
    order = np.asarray(sorted_indices, dtype=np.int64)
    if len(predictions) != len(order):
        raise ValueError("sorted predictions and source indices must have equal lengths")
    return predictions[np.argsort(order, kind="stable")]


def _selected_epoch_scores(epoch_predictions, selected_epoch, expected_shape):
    """Return aligned logits only when they belong to the selected checkpoint epoch."""
    if selected_epoch is None:
        return None
    for item in epoch_predictions or []:
        if int(item.get("epoch", -1)) != int(selected_epoch):
            continue
        scores = np.asarray(item.get("scores"))
        if scores.shape == tuple(expected_shape) and np.isfinite(scores).all():
            return scores
        return None
    return None


def _predict_cached_dataset(model, dataset, *, batch_size, data_collator, device,
                            amp_dtype=None, num_workers=0):
    """Infer over pre-tokenized rows in length order, restoring original alignment."""
    import torch
    from torch.utils.data import DataLoader, Subset

    if hasattr(dataset, "lengths"):
        lengths = dataset.lengths
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, "lengths"):
        lengths = [dataset.dataset.lengths[int(index)] for index in dataset.indices]
    else:
        raise ValueError("cached inference dataset must expose token lengths")
    order = _length_order(lengths)
    ordered = Subset(dataset, order.tolist())
    loader = DataLoader(ordered, batch_size=max(1, int(batch_size)), shuffle=False,
                        collate_fn=data_collator, num_workers=int(num_workers))
    chunks = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            amp = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if getattr(device, "type", None) == "cuda" and amp_dtype is not None
                   else nullcontext())
            with amp:
                output = model(**batch)
            logits = output.logits if hasattr(output, "logits") else output
            chunks.append(logits.float().cpu().numpy())
    sorted_predictions = (np.concatenate(chunks, axis=0) if chunks
                          else np.empty((0, 0), dtype=np.float32))
    return _restore_prediction_order(sorted_predictions, order)


def _validation_losses(y_true, raw_scores, calibrated_probabilities, classes):
    """Return comparable raw and calibrated multiclass validation log losses."""
    from sklearn.metrics import log_loss

    raw_probabilities = softmax(np.asarray(raw_scores, dtype=float), axis=1)
    calibrated_probabilities = np.asarray(calibrated_probabilities, dtype=float)
    return {
        "validation_loss_raw": float(log_loss(y_true, raw_probabilities, labels=list(classes))),
        "validation_loss_calibrated": float(log_loss(
            y_true, calibrated_probabilities, labels=list(classes))),
    }


def train_svc(train_text, y_train, validation_text, classes, *, c=.5, calibration_cv=3, seed=2026, feature_config=None, cache_dir=None, fold_token="", validation_labels=None):
    """Fit TF-IDF on train only, then save raw margins and train-only calibration."""
    start = time.perf_counter()
    if cache_dir:
        x_train, x_val, features, cache_key = load_or_fit(train_text, validation_text,
            config=feature_config or {"views": ["char", "word"]}, cache_dir=cache_dir, fold_token=fold_token)
    else:
        features = FoldFeatures(feature_config)
        x_train = features.fit_transform(train_text)
        x_val = features.transform(validation_text)
        cache_key = None
    base = LinearSVC(C=c, class_weight="balanced", random_state=seed)
    try:
        calibrated = CalibratedClassifierCV(estimator=LinearSVC(C=c, class_weight="balanced", random_state=seed), method="sigmoid", cv=calibration_cv)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=LinearSVC(C=c, class_weight="balanced", random_state=seed), method="sigmoid", cv=calibration_cv)
    calibration_started = time.perf_counter()
    calibrated.fit(x_train, y_train)
    calibration_fitting_seconds = time.perf_counter() - calibration_started
    base.fit(x_train, y_train)
    pred_start = time.perf_counter()
    raw = _aligned(np.asarray(base.decision_function(x_val)), base.classes_, classes)
    prob = _aligned(calibrated.predict_proba(x_val), calibrated.classes_, classes)
    validation_losses = (_validation_losses(validation_labels, raw, prob, classes)
                         if validation_labels is not None else {})
    inference_seconds = time.perf_counter() - pred_start
    elapsed = time.perf_counter() - start
    return {"scores": raw, "probabilities": prob, "model": calibrated, "raw_model": base,
            "features": features, "cache_key": cache_key, "fit_seconds": elapsed, "classes": list(classes),
            "inference_seconds": inference_seconds, "peak_gpu_memory_bytes": None,
            "checkpoint_scoring_method": "in_memory_fitted_model_validation_predictions",
            "final_validation_reused_existing_predictions": True,
            "runtime_breakdown": {"training_and_calibration_seconds": float(pred_start - start),
                "calibration_fitting_seconds": float(calibration_fitting_seconds),
                "final_validation_inference_seconds": float(inference_seconds),
                "best_checkpoint_restoration_seconds": 0.0,
                "calibration_inference_seconds": None, "temperature_fit_seconds": None},
            **validation_losses}


class _BalancedAccuracyMetric:
    """CatBoost eval metric matching sklearn's unweighted macro recall."""

    def __init__(self, classes):
        self.class_to_index = {str(label): index for index, label in enumerate(classes)}
        self.class_count = len(classes)

    def is_max_optimal(self):
        return True

    def get_final_error(self, error, weight):
        return float(error) / max(float(weight), 1.0)

    def evaluate(self, approxes, target, weight):
        predicted = np.asarray(approxes, dtype=float).argmax(axis=0)
        actual = []
        for value in target:
            label = str(value)
            if label in self.class_to_index:
                actual.append(self.class_to_index[label])
                continue
            numeric = int(float(value))
            if not 0 <= numeric < self.class_count:
                raise ValueError(f"CatBoost validation target is not a known class: {value!r}")
            actual.append(numeric)
        actual = np.asarray(actual, dtype=int)
        recalls = [float(np.mean(predicted[actual == index] == index))
                   for index in range(self.class_count) if np.any(actual == index)]
        return float(sum(recalls)), float(len(recalls))


def train_catboost_stylometry(train_text, y_train, validation_text, classes, *, output_dir,
                              iterations=800, depth=7, learning_rate=.05, l2_leaf_reg=5.0,
                              thread_count=8, seed=2026, temperature_calibration=True,
                              validation_labels=None, early_stopping_rounds=100,
                              feature_recipe=None, feature_matrices=None, init_model=None,
                              checkpoint_metadata=None, checkpoint_selection_metric="balanced_accuracy",
                              calibration_max_samples=5000, calibration_seed=2026,
                              calibration_full_dataset=False):
    """Fit CatBoost on deterministic row-local style statistics; no validation fitting."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost model requires catboost; install the project's requirements.txt") from exc
    start = time.perf_counter()
    if checkpoint_selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be accuracy or balanced_accuracy")
    feature_recipe = {"baseline": list(DEFAULT_STYLOMETRY_RECIPE["baseline"]),
                      "groups": [], **(feature_recipe or {})}
    if feature_matrices is None:
        x_train, feature_names = stylometry(train_text, feature_recipe, return_feature_names=True)
        x_validation = stylometry(validation_text, feature_recipe)
    else:
        x_train, x_validation, feature_names = feature_matrices
        x_train = np.asarray(x_train, dtype=np.float64)
        x_validation = np.asarray(x_validation, dtype=np.float64)
        feature_names = list(feature_names)
    metric = _BalancedAccuracyMetric(classes) if checkpoint_selection_metric == "balanced_accuracy" else "Accuracy"
    model = CatBoostClassifier(iterations=int(iterations), depth=int(depth),
        learning_rate=float(learning_rate), l2_leaf_reg=float(l2_leaf_reg),
        loss_function="MultiClass", eval_metric=metric, auto_class_weights="Balanced",
        random_seed=int(seed), thread_count=int(thread_count), verbose=False,
        allow_writing_files=False)
    fit_kwargs = {}
    if init_model is not None:
        fit_kwargs["init_model"] = str(init_model)
    if validation_labels is not None:
        fit_kwargs.update({"eval_set": (x_validation, list(validation_labels)),
                           "use_best_model": True,
                           "early_stopping_rounds": int(early_stopping_rounds)})
    startup_seconds = time.perf_counter() - start
    training_started = time.perf_counter()
    model.fit(x_train, list(y_train), **fit_kwargs)
    training_seconds = time.perf_counter() - training_started
    iteration_metrics = []
    selected_iteration = (int(model.get_best_iteration()) if validation_labels is not None
                          else int(model.tree_count_) if model.tree_count_ else None)
    if validation_labels is not None and selected_iteration < 0:
        raise RuntimeError("CatBoost did not identify a best validation iteration")
    selected_validation_accuracy = None
    selected_validation_balanced_accuracy = None
    if validation_labels is not None:
        history = model.get_evals_result().get("validation", {})
        metric_prefix = "Balanced" if checkpoint_selection_metric == "balanced_accuracy" else "Accuracy"
        metric_values = next((values for name, values in history.items()
                              if metric_prefix.casefold() in name.casefold()), [])
        iteration_metrics = [{"iteration": index + 1,
                              f"validation_{checkpoint_selection_metric}": float(value)}
                             for index, value in enumerate(metric_values)]
    pred_start = time.perf_counter()
    validation_raw = np.asarray(model.predict(x_validation, prediction_type="RawFormulaVal"))
    inference_seconds = time.perf_counter() - pred_start
    model_classes = list(model.classes_)
    if validation_raw.ndim == 1:
        validation_raw = np.column_stack([-validation_raw, validation_raw])
    raw = _aligned(validation_raw, model_classes, classes)
    calibration_metrics = {}
    if temperature_calibration:
        calibration_inference_started = time.perf_counter()
        calibration_indices = _calibration_indices(y_train, max_samples=calibration_max_samples,
            seed=calibration_seed, full_dataset=calibration_full_dataset)
        calibration_labels = np.asarray(y_train, dtype=object)[calibration_indices].tolist()
        train_raw = np.asarray(model.predict(x_train[calibration_indices], prediction_type="RawFormulaVal"))
        if train_raw.ndim == 1:
            train_raw = np.column_stack([-train_raw, train_raw])
        train_raw = _aligned(train_raw, model_classes, classes)
        calibration_inference_seconds = time.perf_counter() - calibration_inference_started
        temperature_fit_started = time.perf_counter()
        temperature, calibration_metrics = _fit_temperature(train_raw, calibration_labels, classes,
            max_samples=None, seed=calibration_seed, full_dataset=True, return_metrics=True)
        # The supplied logits already correspond to the deterministic training subset.
        calibration_metrics["calibration_sample_size"] = int(len(calibration_indices))
        calibration_metrics["calibration_full_dataset"] = bool(len(calibration_indices) == len(y_train))
        calibration_metrics["calibration_seed"] = int(calibration_seed)
        calibration_metrics["calibration_inference_seconds"] = calibration_inference_seconds
        calibration_metrics["temperature_fit_seconds"] = time.perf_counter() - temperature_fit_started
    else:
        temperature = 1.0
    probabilities = softmax(raw / temperature, axis=1)
    validation_losses = (_validation_losses(validation_labels, raw, probabilities, classes)
                         if validation_labels is not None else {})
    from sklearn.metrics import accuracy_score
    validation_accuracy = (float(accuracy_score(validation_labels,
        np.asarray(classes)[raw.argmax(axis=1)])) if validation_labels is not None else None)
    if validation_labels is not None:
        selected_validation_accuracy = validation_accuracy
        from sklearn.metrics import balanced_accuracy_score
        selected_validation_balanced_accuracy = float(balanced_accuracy_score(
            validation_labels, np.asarray(classes)[raw.argmax(axis=1)]))
    best_iteration = selected_iteration if validation_labels is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_started = time.perf_counter()
    model.save_model(str(output_dir / "model.cbm"))
    metadata = {
        "feature_version": STYLOMETRY_VERSION,
        "feature_names": feature_names,
        "feature_recipe": feature_recipe,
        "feature_count": int(x_train.shape[1]), "temperature": temperature,
        **calibration_metrics,
        **validation_losses,
        "selected_iteration": best_iteration,
        "selected_validation_accuracy": selected_validation_accuracy,
        "selected_validation_balanced_accuracy": selected_validation_balanced_accuracy,
        "selection_metric": f"validation_{checkpoint_selection_metric}" if validation_labels is not None else None,
        "classes": list(classes), "recipe": {"iterations": int(iterations), "depth": int(depth),
            "learning_rate": float(learning_rate), "l2_leaf_reg": float(l2_leaf_reg),
            "thread_count": int(thread_count), "seed": int(seed),
            "feature_recipe": feature_recipe}
    }
    metadata.update({"tree_count": int(model.tree_count_),
                     "added_iterations": int(iterations),
                     "parent_checkpoint": str(init_model) if init_model is not None else None})
    if checkpoint_metadata:
        metadata["experiment"] = dict(checkpoint_metadata)
    (output_dir / "stylometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checkpoint_save_seconds = time.perf_counter() - checkpoint_started
    return {"scores": raw, "probabilities": probabilities, "model": model,
        "fit_seconds": time.perf_counter() - start, "inference_seconds": inference_seconds,
        "runtime_breakdown": {"startup_seconds": startup_seconds,
            "training_including_validation_seconds": training_seconds,
            "validation_seconds": None,
            "final_validation_seconds": inference_seconds,
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "calibration_inference_seconds": float(calibration_metrics.get("calibration_inference_seconds", 0.0)),
            "temperature_fit_seconds": float(calibration_metrics.get("temperature_fit_seconds", 0.0))},
        "classes": list(classes), "calibration_temperature": temperature,
        **calibration_metrics,
        "calibration_method": "train-only scalar temperature" if temperature_calibration else "none",
        "selected_iteration": best_iteration, "selected_validation_accuracy": selected_validation_accuracy,
        "selected_validation_balanced_accuracy": selected_validation_balanced_accuracy,
        "validation_metrics_by_iteration": iteration_metrics,
        **validation_losses,
        "feature_count": int(x_train.shape[1]), "feature_names": feature_names,
        "feature_recipe": feature_recipe,
        "checkpoint_selection_metric": f"validation_{checkpoint_selection_metric}" if validation_labels is not None else None,
        "checkpoint": str(output_dir), "tree_count": int(model.tree_count_),
        "checkpoint_scoring_method": "in_memory_selected_iteration_validation_predictions",
        "final_validation_reused_existing_predictions": validation_labels is not None,
        "added_iterations": int(iterations), "parent_checkpoint": str(init_model) if init_model is not None else None,
        "peak_gpu_memory_bytes": None}


FROZEN_BACKBONE_MODELS = {"embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"}
EMBEDDINGGEMMA2_TASK_PREFIX = "task: classification | query: "


def calculate_max_token_length(model_name, texts, *, checkpoint, revision=None,
                               task_prefix="", use_processor=False,
                               trust_remote_code=False, fix_mistral_regex=False,
                               local_files_only=True, batch_size=256):
    """Return the uncapped maximum encoded length and reject context overflow."""
    _force_huggingface_offline()
    _, tr = _require_neural()
    load_kwargs = {"local_files_only": local_files_only}
    if revision is not None:
        load_kwargs["revision"] = revision
    if not use_processor and trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if use_processor and model_name == "ministral3_3b" and fix_mistral_regex:
        load_kwargs["fix_mistral_regex"] = True
    tokenizer = (tr.AutoProcessor.from_pretrained(checkpoint, **load_kwargs) if use_processor
                 else tr.AutoTokenizer.from_pretrained(checkpoint, **load_kwargs))
    config = tr.AutoConfig.from_pretrained(checkpoint, **{
        key: value for key, value in load_kwargs.items() if key != "fix_mistral_regex"})

    context_fields = ("max_position_embeddings", "n_positions", "max_sequence_length",
                      "max_seq_len", "max_seq_length", "seq_length")
    context_limits = []
    configs = [config]
    for field in ("text_config", "language_config"):
        nested = getattr(config, field, None)
        if nested is not None:
            configs.append(nested)
    for item in configs:
        for field in context_fields:
            value = getattr(item, field, None)
            if isinstance(value, int) and 0 < value < 10_000_000:
                context_limits.append(value)
    tokenizer_obj = getattr(tokenizer, "tokenizer", tokenizer)
    tokenizer_limit = getattr(tokenizer_obj, "model_max_length", None)
    if not context_limits and isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        context_limits.append(tokenizer_limit)
    if not context_limits:
        raise ValueError(f"could not determine supported context window for {model_name} ({checkpoint})")
    context_window = min(context_limits)

    prefix = (EMBEDDINGGEMMA2_TASK_PREFIX if use_processor and model_name == "embeddinggemma2"
              else "" if use_processor else str(task_prefix or ""))
    all_texts = [prefix + str(text) for text in texts]
    cache_payload = json.dumps({"model": model_name, "checkpoint": str(checkpoint),
        "revision": revision, "prefix": prefix, "processor": bool(use_processor),
        "trust_remote_code": bool(trust_remote_code), "fix_mistral_regex": bool(fix_mistral_regex),
        "texts_sha256": hashlib.sha256("\0".join(all_texts).encode("utf-8")).hexdigest(),
        "rows": len(all_texts)}, sort_keys=True)
    cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()
    cache_path = Path(os.environ.get("NLM_TOKEN_LENGTH_CACHE", "artifacts/cache/token_lengths")) / f"{cache_key}.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            maximum = int(cached["maximum_tokens"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # Ignore malformed/stale cache files and recompute from the tokenizer.
            cached, maximum = None, None
        if (cached is not None and cached.get("cache_key") == cache_key
                and int(cached.get("rows", -1)) == len(all_texts)):
            if maximum > context_window:
                raise ValueError(f"{model_name} dataset requires {maximum} tokens (including special tokens and task prefix), "
                                 f"exceeding the model context window of {context_window}")
            return maximum
    maximum = 0
    for start in range(0, len(all_texts), max(1, int(batch_size))):
        batch_texts = all_texts[start:start + max(1, int(batch_size))]
        encoded = (tokenizer(text=batch_texts, truncation=False, padding=False,
                             add_special_tokens=True)
                   if use_processor else tokenizer(batch_texts, truncation=False, padding=False,
                                                   add_special_tokens=True))
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if lengths:
            maximum = max(maximum, max(lengths))
    if maximum > context_window:
        raise ValueError(f"{model_name} dataset requires {maximum} tokens (including special tokens and task prefix), "
                         f"exceeding the model context window of {context_window}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"cache_key": cache_key, "maximum_tokens": maximum,
        "rows": len(all_texts), "context_window": context_window}, indent=2) + "\n", encoding="utf-8")
    return maximum


def _load_frozen_backbone(model_name, checkpoint, revision, *, fix_mistral_regex=False):
    """Load the text feature path for one of the additional multimodal backbones."""
    _force_huggingface_offline()
    torch, tr = _require_neural()
    processor_kwargs = {"revision": revision, "local_files_only": True}
    if model_name == "ministral3_3b" and fix_mistral_regex:
        processor_kwargs["fix_mistral_regex"] = True
    processor = tr.AutoProcessor.from_pretrained(checkpoint, **processor_kwargs)
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16 if torch.cuda.is_available() else torch.float32)
    if model_name == "embeddinggemma2":
        model = tr.AutoModel.from_pretrained(checkpoint, revision=revision, local_files_only=True,
                                             torch_dtype=dtype)
        text_model = model
    elif model_name in {"gemma4_e2b", "gemma4_e4b"}:
        if not hasattr(tr, "AutoModelForMultimodalLM"):
            raise RuntimeError("Gemma 4 requires a Transformers release with AutoModelForMultimodalLM")
        model = tr.AutoModelForMultimodalLM.from_pretrained(checkpoint, revision=revision,
                                                            local_files_only=True, torch_dtype=dtype)
        text_model = model.model
    elif model_name == "ministral3_3b":
        model = tr.Mistral3ForConditionalGeneration.from_pretrained(
            checkpoint, revision=revision, local_files_only=True, torch_dtype=dtype)
        text_model = model.model
    else:
        raise ValueError(f"unsupported frozen backbone: {model_name}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, text_model, processor


def _read_embedding_cache(cache_path, metadata_path, cache_key, row_ids):
    """Load cached features only when their exact ordered row alignment matches."""
    cache_path, metadata_path = Path(cache_path), Path(metadata_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") != cache_key or metadata.get("row_ids") != list(row_ids):
            return None
        with np.load(cache_path) as archive:
            features = archive["features"]
        if features.shape[0] != len(row_ids) or not np.isfinite(features).all():
            return None
        return features
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _restore_qwen_selected_state(backbone, classifier, adapter_state, classifier_state, *, setter=None):
    """Restore the selected LoRA adapter and its matching classifier readout."""
    if setter is None:
        from peft.utils.save_and_load import set_peft_model_state_dict
        setter = set_peft_model_state_dict
    setter(backbone, adapter_state, adapter_name="default")
    classifier.load_state_dict(classifier_state)


def extract_frozen_backbone_features(model_name, checkpoint, revision, texts, *, max_length=384,
                                     batch_size=8, model_dir=None, fix_mistral_regex=False,
                                     row_ids=None, cache_dir="artifacts/cache/frozen_embeddings"):
    """Extract deterministic, text-only mean-pooled features without gradient updates."""
    import torch
    texts = [str(text) for text in texts]
    aligned_ids = ([str(value) for value in row_ids] if row_ids is not None else
                   [f"{index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
                    for index, text in enumerate(texts)])
    if len(aligned_ids) != len(texts) or len(set(aligned_ids)) != len(aligned_ids):
        raise ValueError("frozen embedding row IDs must be unique and aligned to texts")
    cache_config = {"model": model_name, "checkpoint": str(checkpoint), "revision": revision,
        "tokenizer_config": {"processor": True, "fix_mistral_regex": bool(fix_mistral_regex)},
        "max_length": int(max_length), "pooling": "attention-mask mean",
        "texts_sha256": hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest(),
        "row_ids_sha256": hashlib.sha256("\0".join(aligned_ids).encode("utf-8")).hexdigest()}
    cache_key = hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.npz"
    metadata_path = Path(cache_dir) / f"{cache_key}.json"
    cached_features = _read_embedding_cache(cache_path, metadata_path, cache_key, aligned_ids)
    if cached_features is not None:
        return cached_features, 0.0
    model, text_model, processor = _load_frozen_backbone(
        model_name, model_dir or checkpoint, revision,
        fix_mistral_regex=fix_mistral_regex)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prefix = EMBEDDINGGEMMA2_TASK_PREFIX if model_name == "embeddinggemma2" else ""
    prepared = [prefix + str(text) for text in texts]
    outputs = []
    started = time.perf_counter()
    inference_batch_size = max(1, int(batch_size))
    tokenization_chunk_size = inference_batch_size * 8
    with torch.inference_mode():
        for start in range(0, len(prepared), tokenization_chunk_size):
            stop = min(len(prepared), start + tokenization_chunk_size)
            encoded = processor(text=prepared[start:stop], truncation=True,
                                max_length=max_length, padding=False)
            # These three text models need token IDs and masks; Gemma 4 also
            # consumes its text modality markers. Ignore processor-only extras.
            keys = {"input_ids", "attention_mask"}
            if model_name == "gemma4_e4b":
                keys.add("mm_token_type_ids")
            row_count = stop - start
            lengths = [len(encoded["input_ids"][index]) for index in range(row_count)]
            order = _length_order(lengths)
            tokenizer_obj = getattr(processor, "tokenizer", processor)
            pad_id = getattr(tokenizer_obj, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer_obj, "eos_token_id", 0)
            chunk_outputs = []
            for offset in range(0, row_count, inference_batch_size):
                indexes = order[offset:offset + inference_batch_size].tolist()
                padded = {}
                for key in keys:
                    if key not in encoded:
                        if key == "attention_mask":
                            seqs = [torch.ones(len(encoded["input_ids"][index]), dtype=torch.long)
                                    for index in indexes]
                        else:
                            continue
                    else:
                        seqs = [torch.as_tensor(encoded[key][index], dtype=torch.long)
                                for index in indexes]
                    padding_value = int(pad_id) if key == "input_ids" else 0
                    padded[key] = torch.nn.utils.rnn.pad_sequence(
                        seqs, batch_first=True, padding_value=padding_value).to(device)
                result = text_model(**padded, use_cache=False, return_dict=True)
                hidden = result.last_hidden_state.float()
                mask = padded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                if model_name == "embeddinggemma2":
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunk_outputs.append(pooled.cpu().numpy())
            outputs.append(_restore_prediction_order(np.concatenate(chunk_outputs, axis=0), order))
    elapsed = time.perf_counter() - started
    features = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
    del model, text_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not np.isfinite(features).all():
        raise ValueError(f"{model_name} produced non-finite text features")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    metadata_path.write_text(json.dumps({"cache_key": cache_key, "config": cache_config,
        "row_ids": aligned_ids, "rows": len(texts), "shape": list(features.shape)},
        indent=2) + "\n", encoding="utf-8")
    return features, elapsed


def train_frozen_backbone_fold(model_name, train_text, y_train, validation_text, classes, *,
                               checkpoint, revision, output_dir, max_length=384, batch_size=8,
                               c=.5, calibration_cv=3, seed=2026, fix_mistral_regex=False,
                               validation_labels=None, train_row_ids=None, validation_row_ids=None,
                               embedding_cache_dir="artifacts/cache/frozen_embeddings"):
    """Fit a train-only LinearSVC readout over frozen model-specific text features."""
    if model_name not in FROZEN_BACKBONE_MODELS:
        raise ValueError(f"unsupported frozen backbone: {model_name}")
    start = time.perf_counter()
    same_rows = list(train_text) == list(validation_text)
    all_text = list(train_text) if same_rows else list(train_text) + list(validation_text)
    all_ids = (list(train_row_ids) if train_row_ids is not None else
               [f"train:{i}:{hashlib.sha256(str(text).encode()).hexdigest()}" for i, text in enumerate(train_text)])
    if not same_rows:
        all_ids += (list(validation_row_ids) if validation_row_ids is not None else
                    [f"validation:{i}:{hashlib.sha256(str(text).encode()).hexdigest()}" for i, text in enumerate(validation_text)])
    all_features, extraction_seconds = extract_frozen_backbone_features(
        model_name, checkpoint, revision, all_text, max_length=max_length,
        batch_size=batch_size, fix_mistral_regex=fix_mistral_regex, row_ids=all_ids,
        cache_dir=embedding_cache_dir)
    train_features = all_features[:len(train_text)]
    validation_features = all_features if same_rows else all_features[len(train_text):]
    base = LinearSVC(C=float(c), class_weight="balanced", random_state=seed)
    readout_fit_started = time.perf_counter()
    base.fit(train_features, y_train)
    readout_fit_seconds = time.perf_counter() - readout_fit_started
    calibration_started = time.perf_counter()
    try:
        calibrated = CalibratedClassifierCV(
            estimator=LinearSVC(C=float(c), class_weight="balanced", random_state=seed),
            method="sigmoid", cv=int(calibration_cv))
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=LinearSVC(C=float(c), class_weight="balanced", random_state=seed),
            method="sigmoid", cv=int(calibration_cv))
    calibrated.fit(train_features, y_train)
    calibration_fit_seconds = time.perf_counter() - calibration_started
    inference_started = time.perf_counter()
    raw = _aligned(np.asarray(base.decision_function(validation_features)), base.classes_, classes)
    probabilities = _aligned(calibrated.predict_proba(validation_features), calibrated.classes_, classes)
    validation_losses = (_validation_losses(validation_labels, raw, probabilities, classes)
                         if validation_labels is not None else {})
    validation_fraction = 1.0 if same_rows else len(validation_text) / max(1, len(train_text) + len(validation_text))
    validation_inference_seconds = time.perf_counter() - inference_started
    inference_seconds = extraction_seconds * validation_fraction + validation_inference_seconds
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = output_dir / "readout.joblib"
    import joblib
    serialization_started = time.perf_counter()
    joblib.dump({"model": calibrated, "raw_model": base, "classes": list(classes),
                 "model_name": model_name, "checkpoint": checkpoint, "revision": revision,
                 "max_length": int(max_length), "feature_pooling": "attention-mask mean",
                 "task_prefix": "task: classification | query: " if model_name == "embeddinggemma2" else "",
                 "fix_mistral_regex": bool(fix_mistral_regex)}, joblib_path)
    checkpoint_serialization_seconds = time.perf_counter() - serialization_started
    return {"scores": raw, "probabilities": probabilities, "model": calibrated,
            "raw_model": base, "fit_seconds": time.perf_counter() - start,
            "inference_seconds": inference_seconds, "classes": list(classes),
            "checkpoint_scoring_method": "in_memory_fitted_readout_validation_predictions",
            "final_validation_reused_existing_predictions": True,
            "calibration_method": "sigmoid train-only CV", "checkpoint": str(output_dir),
            "readout_path": str(joblib_path), "peak_gpu_memory_bytes": None,
            "runtime_breakdown": {"frozen_feature_extraction_seconds": float(extraction_seconds),
                "readout_fit_seconds": float(readout_fit_seconds),
                "calibration_fit_seconds": float(calibration_fit_seconds),
                "final_validation_inference_seconds": float(validation_inference_seconds),
                "checkpoint_serialization_seconds": float(checkpoint_serialization_seconds),
                "best_checkpoint_restoration_seconds": 0.0,
                "calibration_inference_seconds": 0.0, "temperature_fit_seconds": 0.0},
            **validation_losses}


def _require_neural():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Neural training needs the optional neural dependencies. Install this project's [neural] extra on the training computer.") from exc
    return torch, transformers


def _length_grouping_args(transformers_module):
    """Enable length grouping across supported Transformers APIs."""
    parameters = inspect.signature(transformers_module.TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in parameters:
        return {"train_sampling_strategy": "group_by_length"}
    if "group_by_length" in parameters:
        return {"group_by_length": True}
    return {}


def _force_huggingface_offline():
    """Prevent model training processes from making any Hugging Face Hub requests."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _live_training_progress_callback(TrainerCallback, output_dir):
    """Write per-epoch trainer metrics atomically for the live dashboard."""
    progress_path = Path(output_dir) / "training_progress.json"
    class LiveTrainingProgressCallback(TrainerCallback):
        def __init__(self):
            self.epochs = {}
            self.baseline = None

        def _write_progress(self):
            payload = {"pid": os.getpid(), "updated_at": time.time(),
                       "epochs": [self.epochs[key] for key in sorted(self.epochs)]}
            if self.baseline is not None:
                payload["baseline"] = self.baseline
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, progress_path)

        def set_baseline(self, metrics):
            self.baseline = dict(metrics)
            self._write_progress()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            raw_epoch = logs.get("epoch", state.epoch)
            if raw_epoch is None:
                return control
            epoch = int(round(float(raw_epoch)))
            if epoch < 1:
                return control
            row = self.epochs.setdefault(epoch, {"epoch": epoch})
            for source, target in (("learning_rate", "learning_rate"), ("loss", "train_loss"),
                                   ("eval_loss", "validation_loss_raw"), ("eval_accuracy", "accuracy"),
                                   ("eval_balanced_accuracy", "balanced_accuracy")):
                if logs.get(source) is not None:
                    row[target] = float(logs[source])
            if logs.get("eval_loss") is not None:
                row["validation_loss"] = float(logs["eval_loss"])
            self._write_progress()
            return control
    return LiveTrainingProgressCallback()


def _accuracy_early_stopping_callback(TrainerCallback, patience, metric="accuracy"):
    """Stop after `patience` consecutive validations without a better metric."""
    patience = int(patience)
    if patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")

    class AccuracyEarlyStoppingCallback(TrainerCallback):
        def __init__(self):
            self.patience = patience
            self.best_metric_value = float("-inf")
            self.best_accuracy = None
            self.best_balanced_accuracy = None
            self.best_epoch = None
            self.bad_epochs = 0
            self.stopped_epoch = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            value = (metrics or {}).get(f"eval_{metric}")
            if value is None:
                return control
            epoch = int(round(float(state.epoch or 0)))
            value = float(value)
            if value > self.best_metric_value:
                self.best_metric_value = value
                self.best_accuracy = (metrics or {}).get("eval_accuracy")
                self.best_balanced_accuracy = (metrics or {}).get("eval_balanced_accuracy")
                self.best_epoch = epoch
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= patience:
                    self.stopped_epoch = epoch
                    control.should_training_stop = True
            return control

    return AccuracyEarlyStoppingCallback()


def _early_stopping_summary(callback, patience, max_epochs, metric="accuracy"):
    return {
        "metric": f"validation_{metric}",
        "patience_epochs": int(patience),
        "best_epoch": callback.best_epoch,
        "best_validation_accuracy": callback.best_accuracy,
        "best_validation_balanced_accuracy": callback.best_balanced_accuracy,
        "best_validation_metric_value": (callback.best_metric_value
                                         if callback.best_epoch is not None else None),
        "stopped_early": callback.stopped_epoch is not None and callback.stopped_epoch < max_epochs,
        "stopped_epoch": callback.stopped_epoch,
    }


def _epoch_prediction_capture_callback(TrainerCallback, latest_prediction):
    """Capture the probability-model inputs emitted by each validation epoch."""
    class EpochPredictionCaptureCallback(TrainerCallback):
        def __init__(self):
            self.predictions = []

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            logits = latest_prediction.pop("logits", None)
            if logits is not None:
                self.predictions.append({
                    "epoch": int(round(float(state.epoch or len(self.predictions) + 1))),
                    "scores": logits,
                })
            return control

    return EpochPredictionCaptureCallback()


def _epoch_decay_lambda(*, decay, steps_per_epoch, warmup_steps=0):
    """Combine linear warmup with a multiplicative decay at each epoch boundary."""
    steps_per_epoch = max(1, int(steps_per_epoch))
    warmup_steps = max(0, int(warmup_steps))
    def scale(step):
        warmup = min(1.0, max(0.0, float(step) / warmup_steps)) if warmup_steps else 1.0
        epoch = max(0, int(step) // steps_per_epoch)
        return warmup * float(decay) ** epoch
    return scale


def train_encoder_fold(train_text, y_train, validation_text, classes, *, checkpoint, revision, output_dir,
                       epochs=3, batch_size=16, max_length=384, learning_rate=2e-5, seed=2026,
                       gradient_accumulation_steps=1, local_files_only=True, validation_labels=None,
                       resume_checkpoint=None, learning_rate_decay=None,
                       select_best_checkpoint=False,
                       optimizer="adamw_torch", weight_decay=0.0, lr_scheduler_type=None,
                       warmup_ratio=0.0, warmup_epochs=None, early_stopping_patience=2,
                       auto_batch_size=False, max_batch_size=1024,
                       batch_probe_safety_fraction=0.95,
                       checkpoint_selection_metric="balanced_accuracy", calibration_max_samples=5000,
                       calibration_seed=2026, calibration_full_dataset=False,
                       save_epoch_checkpoints=False):
    """Fine-tune or continue an encoder after recording its validation baseline."""
    _force_huggingface_offline()
    local_files_only = True
    torch, tr = _require_neural()
    start = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    label_map = {label: i for i, label in enumerate(classes)}
    y_indices = np.asarray([label_map[y] for y in y_train], dtype=np.int64)
    model_source = str(resume_checkpoint or checkpoint)
    model_revision = None if resume_checkpoint else revision
    tokenizer = tr.AutoTokenizer.from_pretrained(model_source, revision=model_revision, trust_remote_code=True,
                                                  local_files_only=local_files_only)
    model = tr.AutoModelForSequenceClassification.from_pretrained(
        model_source, revision=model_revision, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id=label_map, trust_remote_code=True,
        local_files_only=local_files_only)
    train = _TextDataset(train_text, y_indices, tokenizer, max_length)
    val_indices = ([label_map[label] for label in validation_labels]
                   if validation_labels is not None else None)
    val = _TextDataset(validation_text, val_indices, tokenizer, max_length)
    data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train.collate_fn = data_collator
    val.collate_fn = data_collator
    requested_batch_size = int(batch_size)
    effective_batch_size = requested_batch_size
    batch_probe = {"enabled": False, "selected_batch_size": effective_batch_size}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    if auto_batch_size:
        effective_batch_size, batch_probe = _probe_max_batch_size(
            model, train, requested_batch_size=requested_batch_size,
            max_batch_size=max_batch_size, safety_fraction=batch_probe_safety_fraction,
            use_bf16=use_bf16, use_fp16=use_fp16, optimizer_name=optimizer,
            weight_decay=weight_decay)
    batch_probe.update({"requested_batch_size": requested_batch_size,
                        "effective_batch_size": effective_batch_size,
                        "gradient_accumulation_steps": int(gradient_accumulation_steps),
                        "padding_strategy": "dynamic", "pad_to_multiple_of": 8})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "batch_size_probe.json").write_text(json.dumps(batch_probe, indent=2) + "\n")
    print(f"Encoder batch-size selection: {json.dumps(batch_probe, sort_keys=True)}", flush=True)
    captured_validation_prediction = {}
    validation_order = _length_order(val.lengths)
    def compute_metrics(eval_prediction):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        logits = _restore_prediction_order(logits, validation_order)
        labels = _restore_prediction_order(eval_prediction.label_ids, validation_order)
        captured_validation_prediction["logits"] = np.asarray(logits, dtype=np.float32).copy()
        predictions = np.asarray(logits).argmax(axis=1)
        return {"accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions))}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import TrainerCallback
    from torch.utils.data import Sampler

    class LengthOrderedSampler(Sampler):
        def __init__(self, dataset):
            self.order = _length_order(dataset.lengths).tolist()
        def __iter__(self): return iter(self.order)
        def __len__(self): return len(self.order)

    class OrderedValidationTrainer(tr.Trainer):
        validation_seconds = 0.0
        def _get_eval_sampler(self, eval_dataset):
            # Keep each epoch's saved validation outputs aligned to input rows.
            # Training still uses its configured length-grouped sampler.
            if hasattr(eval_dataset, "lengths"):
                return LengthOrderedSampler(eval_dataset)
            return LengthOrderedSampler(val)
        def evaluate(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                self.validation_seconds += time.perf_counter() - started

    live_callback = _live_training_progress_callback(TrainerCallback, out)
    epoch_prediction_callback = _epoch_prediction_capture_callback(
        TrainerCallback, captured_validation_prediction)
    if learning_rate_decay is not None and not 0 < float(learning_rate_decay) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1]")
    if validation_labels is None:
        raise ValueError("encoder training requires validation_labels for baseline and epoch validation")
    if checkpoint_selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be accuracy or balanced_accuracy")
    best_state = {"metric": -1.0, "accuracy": None, "balanced_accuracy": None,
                  "epoch": None, "model_state": None, "snapshot_seconds": 0.0}

    class BestCheckpointCallback(TrainerCallback):
        save_seconds = 0.0
        epoch_checkpoints = []
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metric_value = (metrics or {}).get(f"eval_{checkpoint_selection_metric}")
            if select_best_checkpoint and metric_value is not None and float(metric_value) > best_state["metric"]:
                best_state["metric"] = float(metric_value)
                best_state["accuracy"] = (metrics or {}).get("eval_accuracy")
                best_state["balanced_accuracy"] = (metrics or {}).get("eval_balanced_accuracy")
                best_state["epoch"] = int(round(float(state.epoch or 0)))
                snapshot_started = time.perf_counter()
                best_state["model_state"] = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_state["snapshot_seconds"] += time.perf_counter() - snapshot_started
            if save_epoch_checkpoints:
                epoch = int(round(float(state.epoch or 0)))
                epoch_dir = out / f"epoch-{epoch:02d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                save_started = time.perf_counter()
                model.save_pretrained(epoch_dir, safe_serialization=True)
                tokenizer.save_pretrained(epoch_dir)
                (epoch_dir / "validation_metrics.json").write_text(
                    json.dumps(metrics or {}, indent=2) + "\n")
                self.epoch_checkpoints.append(str(epoch_dir))
                self.save_seconds += time.perf_counter() - save_started
            return control

    encoder_steps_per_epoch = max(1, math.ceil(len(train) / (effective_batch_size * gradient_accumulation_steps)))
    warmup_basis_epochs = min(int(epochs), int(warmup_epochs)) if warmup_epochs is not None else int(epochs)
    encoder_warmup_steps = math.ceil(float(warmup_ratio) * warmup_basis_epochs * encoder_steps_per_epoch)
    args = tr.TrainingArguments(output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch_size, per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate,
        optim=optimizer, weight_decay=float(weight_decay), warmup_steps=encoder_warmup_steps,
        lr_scheduler_type=("constant" if learning_rate_decay is not None else (lr_scheduler_type or "linear")),
        seed=seed, data_seed=seed, save_strategy="no", eval_strategy="epoch" if validation_labels is not None else "no", report_to=[],
        logging_strategy="epoch", fp16=use_fp16, bf16=use_bf16,
        dataloader_num_workers=0, **_length_grouping_args(tr))

    trainer_type = OrderedValidationTrainer
    if learning_rate_decay is not None:
        decay = float(learning_rate_decay)

        class EpochDecayTrainer(OrderedValidationTrainer):
            def create_scheduler(self, num_training_steps, optimizer=None):
                if self.lr_scheduler is None:
                    optimizer = optimizer or self.optimizer
                    steps_per_epoch = encoder_steps_per_epoch
                    warmup_steps = self.args.get_warmup_steps(num_training_steps)
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=_epoch_decay_lambda(
                            decay=decay, steps_per_epoch=steps_per_epoch, warmup_steps=warmup_steps))
                return self.lr_scheduler

        trainer_type = EpochDecayTrainer

    callbacks = [live_callback, epoch_prediction_callback]
    early_stopping_callback = None
    best_checkpoint_callback = BestCheckpointCallback()
    if validation_labels is not None:
        early_stopping_callback = _accuracy_early_stopping_callback(
            TrainerCallback, early_stopping_patience, metric=checkpoint_selection_metric)
        callbacks.append(early_stopping_callback)
        callbacks.append(best_checkpoint_callback)
    trainer = trainer_type(model=model, args=args, train_dataset=train, data_collator=data_collator,
        eval_dataset=val if validation_labels is not None else None,
        compute_metrics=compute_metrics if validation_labels is not None else None,
        processing_class=tokenizer, callbacks=callbacks)
    # Trainer's length-grouped prediction sampler can return predictions in a
    # different row order from the supplied text list. Use a sequential loader
    # for baseline, calibration and final validation scores so outputs remain
    # aligned with their original ids and labels.
    prediction_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(prediction_device)
    prediction_dtype = (torch.bfloat16 if use_bf16 else
                       torch.float16 if use_fp16 else None)

    def predict_cached(dataset):
        return _predict_cached_dataset(model, dataset, batch_size=effective_batch_size,
            data_collator=data_collator, device=prediction_device, amp_dtype=prediction_dtype)

    baseline_validation_started = time.perf_counter()
    baseline_logits = predict_cached(val)
    baseline_validation_seconds = time.perf_counter() - baseline_validation_started
    baseline_predictions = np.asarray(classes)[baseline_logits.argmax(axis=1)]
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
    baseline_metrics = {
        "accuracy": float(accuracy_score(validation_labels, baseline_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(validation_labels, baseline_predictions)),
        "validation_loss": float(log_loss(validation_labels, softmax(baseline_logits, axis=1), labels=classes)),
        "checkpoint": str(resume_checkpoint or checkpoint),
    }
    (out / "baseline_validation.json").write_text(json.dumps(baseline_metrics, indent=2) + "\n")
    live_callback.set_baseline(baseline_metrics)
    startup_seconds = max(0.0, time.perf_counter() - start - baseline_validation_seconds)
    training_started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_started
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir)/"training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    best_checkpoint_restoration_seconds = 0.0
    if select_best_checkpoint:
        if best_state["epoch"] is None:
            raise RuntimeError("best checkpoint selection requested but no epoch validation metric was recorded")
        restore_started = time.perf_counter()
        model.load_state_dict(best_state["model_state"])
        best_checkpoint_restoration_seconds = time.perf_counter() - restore_started
        (out / "checkpoint_selection.json").write_text(json.dumps({
            "metric": f"validation_{checkpoint_selection_metric}", "selected_epoch": best_state["epoch"],
            "selected_validation_accuracy": best_state["accuracy"],
            "selected_validation_balanced_accuracy": best_state["balanced_accuracy"]}, indent=2) + "\n")
    calibration_indices = _calibration_indices(y_train, max_samples=calibration_max_samples,
        seed=calibration_seed, full_dataset=calibration_full_dataset)
    from torch.utils.data import Subset
    calibration_dataset = Subset(train, calibration_indices.tolist())
    calibration_labels = np.asarray(y_train, dtype=object)[calibration_indices].tolist()
    calibration_inference_started = time.perf_counter()
    train_logits = predict_cached(calibration_dataset)
    calibration_inference_seconds = time.perf_counter() - calibration_inference_started
    temperature_fit_started = time.perf_counter()
    temperature, calibration_metrics = _fit_temperature(train_logits, calibration_labels, classes,
        max_samples=None, seed=calibration_seed, full_dataset=True, return_metrics=True)
    temperature_fit_seconds = time.perf_counter() - temperature_fit_started
    calibration_metrics["calibration_sample_size"] = int(len(calibration_indices))
    calibration_metrics["calibration_full_dataset"] = bool(len(calibration_indices) == len(y_train))
    calibration_metrics["calibration_seed"] = int(calibration_seed)
    calibration_metrics["calibration_inference_seconds"] = calibration_inference_seconds
    calibration_metrics["temperature_fit_seconds"] = temperature_fit_seconds
    selected_prediction_epoch = (best_state["epoch"] if select_best_checkpoint else
        max((int(item["epoch"]) for item in epoch_prediction_callback.predictions), default=None))
    selected_prediction = _selected_epoch_scores(epoch_prediction_callback.predictions,
        selected_prediction_epoch, (len(val), len(classes)))
    final_validation_reused_epoch_predictions = selected_prediction is not None
    pred_start = time.perf_counter()
    if final_validation_reused_epoch_predictions:
        logits = selected_prediction
    else:
        logits = predict_cached(val)
    inference_seconds = (0.0 if final_validation_reused_epoch_predictions
                         else time.perf_counter() - pred_start)
    checkpoint_started = time.perf_counter()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    checkpoint_save_seconds = time.perf_counter() - checkpoint_started
    if not np.isfinite(logits).all(): raise ValueError("encoder produced non-finite scores")
    probs = softmax(logits / temperature, axis=1)
    epoch_history = [dict(entry) for entry in trainer.state.log_history if "eval_accuracy" in entry]
    for entry in epoch_history:
        if entry.get("eval_loss") is not None:
            entry["validation_loss_raw"] = float(entry["eval_loss"])
    result_epoch = (best_state["epoch"] if select_best_checkpoint else
                    max((int(round(float(item.get("epoch", 0)))) for item in epoch_history), default=None))
    selected_epoch_metrics = next((item for item in epoch_history
        if int(round(float(item.get("epoch", 0)))) == result_epoch), {})
    validation_loss_raw = selected_epoch_metrics.get("eval_loss")
    if validation_loss_raw is None:
        validation_loss_raw = log_loss(validation_labels, softmax(logits, axis=1), labels=classes)
    validation_loss_calibrated = log_loss(validation_labels, probs, labels=classes)
    if selected_epoch_metrics:
        selected_epoch_metrics["validation_loss_calibrated"] = float(validation_loss_calibrated)
        selected_epoch_metrics["calibration_temperature"] = float(temperature)
    if learning_rate_decay is not None:
        for entry in epoch_history:
            epoch = int(round(float(entry.get("epoch", 1))))
            entry["learning_rate_used"] = float(learning_rate) * float(learning_rate_decay) ** (epoch - 1)
    return {"scores": logits, "probabilities": probs, "model": model, "tokenizer": tokenizer,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "batch_size_probe": batch_probe,
            "fit_seconds": time.perf_counter() - start, "inference_seconds": inference_seconds,
            "classes": list(classes), "calibration_temperature": temperature, "calibration_method": "train-only scalar temperature",
            **calibration_metrics,
            "baseline_validation_metrics": baseline_metrics,
            "validation_metrics_by_epoch": epoch_history,
            "validation_predictions_by_epoch": epoch_prediction_callback.predictions,
            "validation_loss_raw": float(validation_loss_raw),
            "validation_loss_calibrated": float(validation_loss_calibrated),
            "calibration_temperature": float(temperature),
            "best_epoch": best_state["epoch"],
            "best_validation_accuracy": best_state["accuracy"] if best_state["epoch"] is not None else None,
            "best_validation_balanced_accuracy": best_state["balanced_accuracy"] if best_state["epoch"] is not None else None,
            "checkpoint_scoring_method": "best_epoch_validation_predictions" if final_validation_reused_epoch_predictions else "length_grouped_restored_model_forward",
            "final_validation_reused_epoch_predictions": bool(final_validation_reused_epoch_predictions),
            "runtime_breakdown": {"startup_seconds": startup_seconds,
                "baseline_validation_seconds": baseline_validation_seconds,
                "training_including_validation_seconds": training_seconds,
                "validation_seconds": float(getattr(trainer, "validation_seconds", 0.0)),
                "best_checkpoint_restoration_seconds": float(best_checkpoint_restoration_seconds),
                "calibration_inference_seconds": float(calibration_inference_seconds),
                "temperature_fit_seconds": float(temperature_fit_seconds),
                "final_validation_inference_seconds": float(inference_seconds),
                "checkpoint_serialization_seconds": float(checkpoint_save_seconds + best_checkpoint_callback.save_seconds),
                "best_checkpoint_state_snapshot_seconds": float(best_state["snapshot_seconds"]),
                "training_excluding_validation_seconds": max(0.0, training_seconds -
                    float(getattr(trainer, "validation_seconds", 0.0)))},
            "early_stopping": (_early_stopping_summary(early_stopping_callback, early_stopping_patience, epochs,
                                                        metric=checkpoint_selection_metric)
                               if early_stopping_callback is not None else None),
            "checkpoint_selection_metric": f"validation_{checkpoint_selection_metric}" if select_best_checkpoint else None,
            "best_checkpoint_path": str(out) if best_state["epoch"] is not None else None,
            "epoch_checkpoints": list(best_checkpoint_callback.epoch_checkpoints),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}


class _TextDataset:
    def __new__(cls, texts, labels, tokenizer, max_length):
        torch, _ = _require_neural()
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):
                self.enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
                self.lengths = [len(ids) for ids in self.enc["input_ids"]]
                self.labels = labels
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.enc.items()}
                if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        return Dataset()


def _probe_max_batch_size(model, dataset, *, requested_batch_size, max_batch_size,
                         safety_fraction=0.95, use_bf16=False, use_fp16=False,
                         optimizer_name="adamw_torch", weight_decay=0.0,
                         runtime_overhead_reserve_bytes=None):
    """Find the largest batch that survives a forward/backward and optimizer step."""
    torch, _ = _require_neural()
    requested = int(requested_batch_size)
    maximum = max(1, min(int(max_batch_size or requested), len(dataset)))
    report = {"enabled": True, "max_batch_size": maximum, "attempts": []}
    if not torch.cuda.is_available():
        report.update({"status": "gpu_unavailable", "selected_batch_size": requested})
        return requested, report
    if not 0 < float(safety_fraction) <= 1:
        raise ValueError("batch_probe_safety_fraction must be in (0, 1]")

    device = torch.device("cuda", torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    base_allocated = torch.cuda.memory_allocated(device)
    if runtime_overhead_reserve_bytes is None:
        runtime_overhead_reserve_bytes = min(2 * 1024**3, max(256 * 1024**2, int(free_bytes * 0.12)))
    runtime_overhead_reserve_bytes = max(0, int(runtime_overhead_reserve_bytes))
    max_allowed = max(base_allocated, base_allocated + int(free_bytes * float(safety_fraction))
                      - runtime_overhead_reserve_bytes)
    report.update({"status": "probing", "device": str(device),
                   "free_memory_bytes_before_probe": int(free_bytes),
                   "total_memory_bytes": int(total_bytes),
                   "safety_fraction": float(safety_fraction),
                   "runtime_overhead_reserve_bytes": runtime_overhead_reserve_bytes,
                   "memory_limit_bytes": max_allowed})
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    dataset_lengths = getattr(dataset, "lengths", None)
    probe_order = (sorted(range(len(dataset)), key=lambda idx: dataset_lengths[idx], reverse=True)
                   if dataset_lengths is not None else list(range(len(dataset))))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    saved_trainable = [parameter.detach().clone() for parameter in trainable]
    # Trainer initializes AdamW moments lazily on its first update. Simulate a
    # full-precision AdamW update during probing so the selected batch includes
    # that persistent memory. This is conservative for lower-memory optimizers.
    optimizer_probe = str(optimizer_name)

    def make_probe_optimizer():
        name = str(optimizer_name).lower()
        if name in {"paged_adamw_8bit", "adamw_bnb_8bit"}:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise RuntimeError(f"configured optimizer {name} requires bitsandbytes") from exc
            return bnb.optim.PagedAdamW8bit(trainable, lr=1e-4, weight_decay=float(weight_decay))
        if name in {"adamw_torch", "adamw", "adamw_torch_fused"}:
            cls = torch.optim.AdamW
            extra = {"fused": True} if name == "adamw_torch_fused" else {}
            return cls(trainable, lr=1e-4, weight_decay=float(weight_decay), **extra)
        if name == "adam":
            return torch.optim.Adam(trainable, lr=1e-4, weight_decay=float(weight_decay))
        if name == "sgd":
            return torch.optim.SGD(trainable, lr=1e-4, weight_decay=float(weight_decay))
        raise ValueError(f"unsupported optimizer for batch-size probe: {optimizer_name}")

    def attempt(size):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        batch = None
        probe_optimizer = None
        try:
            # Probe the longest examples first so the chosen batch also fits
            # length-grouped batches at the truncation limit.
            examples = [dataset[i] for i in probe_order[:size]]
            collate_fn = getattr(dataset, "collate_fn", None)
            if collate_fn is not None:
                batch = {key: value.to(device) for key, value in collate_fn(examples).items()}
            else:
                batch = {key: torch.stack([example[key] for example in examples]).to(device)
                         for key in examples[0]}
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16 or use_fp16):
                loss = model(**batch).loss
            if loss is None:
                raise RuntimeError("batch size probe needs labeled training examples")
            loss.backward()
            probe_optimizer = make_probe_optimizer()
            probe_optimizer.step()
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
            fits_headroom = peak <= max_allowed
            report["attempts"].append({"batch_size": size, "fits": fits_headroom,
                                       "peak_allocated_bytes": peak,
                                       "reason": None if fits_headroom else "safety_margin"})
            return fits_headroom
        except RuntimeError as exc:
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            is_oom = (oom_type is not None and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            report["attempts"].append({"batch_size": size, "fits": False, "reason": "cuda_oom"})
            return False
        finally:
            model.zero_grad(set_to_none=True)
            if probe_optimizer is not None:
                del probe_optimizer
            with torch.no_grad():
                for parameter, saved in zip(trainable, saved_trainable):
                    parameter.copy_(saved)
            del batch
            torch.cuda.empty_cache()

    try:
        # BitsAndBytes models are dispatched at load time and cannot be moved
        # later with nn.Module.to().
        if not getattr(model, "_qlora_quantized", False):
            model.to(device)
        model.train()
        config = getattr(model, "config", None)
        identity = (getattr(model, "_batch_probe_model_id", None) or
                    getattr(config, "_name_or_path", None) or model.__class__.__qualname__)
        cache_payload = {"model": identity, "model_class": model.__class__.__qualname__,
            "gpu": torch.cuda.get_device_name(device), "device": str(device),
            "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
            "sequence_length": max(dataset_lengths) if dataset_lengths else None,
            "optimizer": str(optimizer_name), "gradient_checkpointing": bool(
                getattr(getattr(model, "backbone", model), "is_gradient_checkpointing", False)),
            "weight_decay": float(weight_decay),
            "safety_fraction": float(safety_fraction),
            "max_batch_size": maximum}
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
        cache_path = Path(os.environ.get("NLM_BATCH_SIZE_CACHE", "artifacts/cache/batch_sizes")) / f"{cache_key}.json"
        cached_size = None
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                candidate_size = int(cached.get("selected_batch_size", 0))
                if cached.get("cache_key") == cache_key and 0 < candidate_size <= maximum:
                    cached_size = candidate_size
            except (OSError, ValueError, json.JSONDecodeError):
                cached_size = None
        if cached_size is not None and attempt(cached_size):
            report.update({"status": "cached_validated", "selected_batch_size": cached_size,
                           "cache_key": cache_key, "cache_validation_passes": 1,
                           "configured_optimizer": str(optimizer_name),
                           "probe_optimizer": optimizer_probe,
                           "optimizer_state_included": True})
            return cached_size, report
        if cached_size is not None:
            cache_path.unlink(missing_ok=True)
        successful = 0
        failed = None
        candidate = 1
        while candidate <= maximum:
            if not attempt(candidate):
                failed = candidate
                break
            successful = candidate
            if candidate == maximum:
                break
            candidate = min(candidate * 2, maximum)
        if successful == 0:
            raise MemoryError("GPU batch-size probe could not fit batch size 1; reduce max_length or use a smaller model")
        if failed is not None:
            low, high = successful + 1, failed - 1
            while low <= high:
                mid = (low + high) // 2
                if attempt(mid):
                    successful = mid
                    low = mid + 1
                else:
                    high = mid - 1
        report.update({"status": "selected", "selected_batch_size": successful,
                       "cache_key": cache_key,
                       "configured_optimizer": str(optimizer_name),
                       "probe_optimizer": optimizer_probe,
                       "optimizer_state_included": True,
                       "peak_probe_allocated_bytes": max(
                           (x.get("peak_allocated_bytes", 0) for x in report["attempts"]), default=0)})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"cache_key": cache_key,
            "selected_batch_size": successful, "config": cache_payload}, indent=2) + "\n")
        return successful, report
    finally:
        model.zero_grad(set_to_none=True)
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.empty_cache()


def load_text_only_backbone(transformers_module, checkpoint, revision, *, torch_dtype,
                            local_files_only=True, trust_remote_code=False, **model_kwargs):
    """Load a multimodal checkpoint and retain its text tower only."""
    composite = transformers_module.AutoModel.from_pretrained(
        checkpoint, revision=revision, torch_dtype=torch_dtype, local_files_only=local_files_only,
        trust_remote_code=trust_remote_code, **model_kwargs)
    text_backbone = getattr(composite, "language_model", None)
    if text_backbone is None:
        raise ValueError(f"{checkpoint} has no language_model text backbone")
    del composite
    return text_backbone


def _require_flash_attention_2_runtime(torch):
    """Fail closed unless the installed FlashAttention-2 kernel runs on this GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("flash_attention_2 requires CUDA")
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError("flash_attention_2 was requested but flash-attn is not installed") from exc
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"flash_attention_2 needs SM80 or newer; this GPU is SM{major}{minor}")
    try:
        q = torch.zeros((1, 8, 1, 64), device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q, causal=True)
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"installed flash-attn has no working kernel for this GPU/CUDA stack: {exc}"
        ) from exc


def train_qwen_lora_fold(train_text, y_train, validation_text, classes, *, checkpoint="Qwen/Qwen3-0.6B",
                         revision, output_dir="artifacts/checkpoints/qwen", epochs=3, batch_size=4,
                         max_length=384, learning_rate=1e-4, seed=2026, gradient_accumulation_steps=1,
                         lora_rank=16, lora_alpha=32,
                         lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                         resume_checkpoint=None, validation_labels=None, gradient_checkpointing=False,
                         local_files_only=False, learning_rate_decay=1.0, auto_batch_size=False,
                         max_batch_size=1024, batch_probe_safety_fraction=0.95,
                         readout_hidden_size=None, task_prefix="", fix_mistral_regex=False,
                         lora_exclude_modules=None, text_only_backbone=False,
                         lr_scheduler_type=None, warmup_ratio=0.0,
                         warmup_epochs=None, optimizer="adamw_torch", weight_decay=0.0,
                         early_stopping_patience=2, trust_remote_code=False,
                         quantization="none", attention_backend=None,
                         checkpoint_selection_metric="balanced_accuracy", save_epoch_checkpoints=False,
                         calibration_max_samples=5000, calibration_seed=2026,
                         calibration_full_dataset=False):
    """LoRA fine-tune causal hidden states with masked-mean classifier readout."""
    _force_huggingface_offline()
    torch, tr = _require_neural()
    try:
        from peft import LoraConfig, TaskType, PeftModel, get_peft_model
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError("Qwen LoRA training needs the project's optional neural dependencies") from exc
    # Training is deliberately cache-only, even if a caller supplies a false config value.
    local_files_only = True
    from torch import nn
    from torch.utils.data import Dataset, Sampler
    from transformers.modeling_outputs import SequenceClassifierOutput
    start = time.perf_counter(); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    tokenizer_kwargs = {"revision": revision, "local_files_only": local_files_only,
                        "trust_remote_code": trust_remote_code}
    if fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = tr.AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    quantization = str(quantization or "none").lower()
    if quantization not in {"none", "qlora4"}:
        raise ValueError("quantization must be 'none' or 'qlora4'")
    if attention_backend not in {None, "sdpa", "flash_attention_2"}:
        raise ValueError("attention_backend must be 'sdpa', 'flash_attention_2', or None")
    checkpointing_limit = (int(max_batch_size or batch_size) if auto_batch_size else int(batch_size))
    checkpointing_disabled_for_batch = bool(gradient_checkpointing and checkpointing_limit > 8)
    if checkpointing_disabled_for_batch:
        gradient_checkpointing = False
    load_kwargs = {"attn_implementation": attention_backend} if attention_backend else {}
    quantization_config = None
    if attention_backend == "flash_attention_2":
        _require_flash_attention_2_runtime(torch)
    if quantization == "qlora4":
        if not torch.cuda.is_available():
            raise RuntimeError("qlora4 requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("qlora4 requires BF16-capable CUDA hardware")
        from transformers import BitsAndBytesConfig
        quantization_config = {"load_in_4bit": True,
                               "bnb_4bit_quant_type": "nf4",
                               "bnb_4bit_use_double_quant": True,
                               "bnb_4bit_compute_dtype": "bfloat16"}
        load_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": {"": torch.cuda.current_device()},
        })
    if text_only_backbone:
        backbone = load_text_only_backbone(tr, checkpoint, revision, torch_dtype=model_dtype,
                                           local_files_only=local_files_only,
                                           trust_remote_code=trust_remote_code, **load_kwargs)
    else:
        backbone = tr.AutoModel.from_pretrained(checkpoint, revision=revision, torch_dtype=model_dtype,
                                                 local_files_only=local_files_only,
                                                 trust_remote_code=trust_remote_code, **load_kwargs)
    if quantization == "qlora4":
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("QLoRA requires PEFT prepare_model_for_kbit_training") from exc
        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=bool(gradient_checkpointing))
    if resume_checkpoint:
        backbone = PeftModel.from_pretrained(backbone, resume_checkpoint, is_trainable=True, local_files_only=True)
    else:
        lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_rank, lora_alpha=lora_alpha,
                          lora_dropout=.05, target_modules=list(lora_target_modules),
                          exclude_modules=lora_exclude_modules)
        backbone = get_peft_model(backbone, lora)
    backbone.config.use_cache = False
    if gradient_checkpointing and quantization != "qlora4":
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    label_map = {label: i for i, label in enumerate(classes)}
    config = backbone.config
    text_config = getattr(config, "text_config", None)
    hidden_size = (int(readout_hidden_size) if readout_hidden_size is not None else
                   getattr(config, "hidden_size", None) or
                   getattr(text_config, "hidden_size", None))
    if hidden_size is None:
        raise ValueError(f"{checkpoint} needs a configured readout_hidden_size")
    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone
            self.classifier=nn.Linear(hidden_size,len(classes))
            if quantization == "qlora4":
                backbone_device = next(backbone.parameters()).device
                self.classifier.to(backbone_device)
                self.hf_device_map = getattr(backbone, "hf_device_map", {"": backbone_device.index or 0})
                self._qlora_quantized = True
        def forward(self,input_ids,attention_mask,labels=None):
            hidden=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            logits=self.classifier(pooled.float())
            loss=nn.functional.cross_entropy(logits,labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss,logits=logits)
    class Rows(Dataset):
        def __init__(self,texts,ys=None):
            prepared=[str(task_prefix)+str(text) for text in texts]
            encoded=tokenizer(prepared,truncation=True,max_length=max_length,padding=False)
            self.enc={k:v for k,v in encoded.items() if k in {"input_ids","attention_mask"}}
            self.lengths=[len(ids) for ids in self.enc["input_ids"]]
            self.ys=ys
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self,i):
            item={k:torch.tensor(v[i],dtype=torch.long) for k,v in self.enc.items()}
            if self.ys is not None: item["labels"]=torch.tensor(label_map[self.ys[i]],dtype=torch.long)
            return item
    model=Head()
    model._batch_probe_model_id = f"{checkpoint}@{revision}:{quantization}"
    if resume_checkpoint:
        classifier_path = Path(resume_checkpoint)/"classifier.pt"
        if not classifier_path.is_file(): raise FileNotFoundError(f"continuation checkpoint has no classifier head: {classifier_path}")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=True))
    captured_validation_prediction = {}
    def compute_metrics(eval_prediction):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        logits = _restore_prediction_order(logits, validation_order)
        captured_validation_prediction["logits"] = np.asarray(logits, dtype=np.float32).copy()
        labels = _restore_prediction_order(eval_prediction.label_ids, validation_order)
        predicted = np.asarray(logits).argmax(axis=1)
        return {"accuracy": float(accuracy_score(labels, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted))}
    out=Path(output_dir)
    checkpoint_metadata = {
        "checkpoint": checkpoint, "revision": revision,
        "max_length": int(max_length), "task_prefix": str(task_prefix),
        "classes": list(classes), "pooling": "attention-mask mean",
        "quantization": quantization, "quantization_config": quantization_config,
        "attention_backend": attention_backend,
        "lora_rank": int(lora_rank), "lora_alpha": int(lora_alpha),
        "lora_target_modules": list(lora_target_modules),
        "gradient_checkpointing": bool(gradient_checkpointing),
    }
    class EpochSnapshotCallback(TrainerCallback):
        def __init__(self):
            self.checkpoints=[]; self.validation_predictions=[]; self.best_state=None
            self.save_seconds=0.0; self.snapshot_seconds=0.0
        def on_evaluate(self,args,state,control,metrics=None,**kwargs):
            epoch=int(round(state.epoch or len(self.checkpoints)+1))
            logits=captured_validation_prediction.pop("logits",None)
            if logits is not None:
                self.validation_predictions.append({"epoch":epoch,"scores":logits})
            metric_value = (metrics or {}).get(f"eval_{checkpoint_selection_metric}")
            if metric_value is not None and (self.best_state is None or
                    float(metric_value) > float(self.best_state["metrics"].get(
                        f"eval_{checkpoint_selection_metric}", -float("inf")))):
                snapshot_started = time.perf_counter()
                from peft import get_peft_model_state_dict
                self.best_state = {"adapter": {k:v.detach().cpu().clone() for k,v in
                    get_peft_model_state_dict(backbone).items()},
                    "classifier": {k:v.detach().cpu().clone() for k,v in model.classifier.state_dict().items()},
                    "metrics": dict(metrics or {}), "epoch": epoch}
                self.snapshot_seconds += time.perf_counter() - snapshot_started
            if save_epoch_checkpoints:
                save_started = time.perf_counter()
                epoch_dir=out/f"epoch-{epoch:02d}"
                epoch_dir.mkdir(parents=True,exist_ok=True)
                model.backbone.save_pretrained(epoch_dir,safe_serialization=True)
                tokenizer.save_pretrained(epoch_dir)
                torch.save({k:v.detach().cpu() for k,v in model.classifier.state_dict().items()},
                           epoch_dir/"classifier.pt")
                (epoch_dir/"checkpoint_manifest.json").write_text(json.dumps(checkpoint_metadata,indent=2)+"\n")
                (epoch_dir/"validation_metrics.json").write_text(json.dumps(metrics or {},indent=2)+"\n")
                self.checkpoints.append(str(epoch_dir))
                self.save_seconds += time.perf_counter() - save_started
            return control
    snapshot_callback=EpochSnapshotCallback()
    progress_callback=_live_training_progress_callback(TrainerCallback, out)
    if checkpoint_selection_metric not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("checkpoint_selection_metric must be accuracy or balanced_accuracy")
    eval_each_epoch = validation_labels is not None
    early_stopping_callback = (_accuracy_early_stopping_callback(
        TrainerCallback, early_stopping_patience, metric=checkpoint_selection_metric)
                               if eval_each_epoch else None)
    if not 0 < float(learning_rate_decay) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1]")
    if not 0 <= float(warmup_ratio) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    # Preserve the historical epoch-wise multiplicative schedule when callers
    # explicitly provide its decay. New recipes can select a standard HF schedule.
    scheduler = lr_scheduler_type or "epoch_decay"
    if scheduler not in {"linear", "cosine", "constant", "epoch_decay"}:
        raise ValueError("lr_scheduler_type must be linear, cosine, constant, or epoch_decay")
    train_dataset=Rows(train_text,y_train)
    validation_dataset_rows=Rows(validation_text,validation_labels)
    validation_order = _length_order(validation_dataset_rows.lengths)

    class LengthOrderedSampler(Sampler):
        def __init__(self, dataset):
            self.order = _length_order(dataset.lengths).tolist()
        def __iter__(self): return iter(self.order)
        def __len__(self): return len(self.order)
    data_collator=tr.DataCollatorWithPadding(tokenizer=tokenizer,pad_to_multiple_of=8)
    train_dataset.collate_fn=data_collator
    use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16=torch.cuda.is_available() and not use_bf16
    effective_batch_size=int(batch_size)
    batch_probe={"enabled": False, "selected_batch_size": effective_batch_size}
    if auto_batch_size:
        effective_batch_size, batch_probe = _probe_max_batch_size(
            model, train_dataset, requested_batch_size=batch_size,
            max_batch_size=max_batch_size or batch_size,
            safety_fraction=batch_probe_safety_fraction, use_bf16=use_bf16, use_fp16=use_fp16,
            optimizer_name=optimizer, weight_decay=weight_decay)
    checkpointing_probe_batch_size = int(effective_batch_size)
    if checkpointing_disabled_for_batch:
        disable_checkpointing = getattr(backbone, "gradient_checkpointing_disable", None)
        if disable_checkpointing is None and quantization != "qlora4":
            raise RuntimeError("batch size threshold requires the model to disable gradient checkpointing")
        if disable_checkpointing is not None:
            disable_checkpointing()
    out.mkdir(parents=True,exist_ok=True)
    batch_probe.update({"requested_batch_size":int(batch_size),
                        "effective_batch_size":effective_batch_size,
                        "gradient_accumulation_steps":int(gradient_accumulation_steps),
                        "gradient_checkpointing_enabled":bool(gradient_checkpointing),
                        "gradient_checkpointing_disabled_for_batch_threshold":checkpointing_disabled_for_batch,
                        "checkpointing_probe_batch_size":checkpointing_probe_batch_size,
                        "padding_strategy":"dynamic",
                        "pad_to_multiple_of":8})
    (out/"batch_size_probe.json").write_text(json.dumps(batch_probe,indent=2)+"\n")
    print(f"LoRA batch-size selection: {json.dumps(batch_probe,sort_keys=True)}",flush=True)
    steps_per_epoch=max(1,math.ceil(len(train_dataset)/(effective_batch_size*gradient_accumulation_steps)))
    warmup_basis_epochs=min(int(epochs),int(warmup_epochs)) if warmup_epochs is not None else int(epochs)
    warmup_steps=math.ceil(float(warmup_ratio)*steps_per_epoch*warmup_basis_epochs)
    args=tr.TrainingArguments(output_dir=str(output_dir),num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch_size,per_device_eval_batch_size=effective_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,learning_rate=learning_rate,
        lr_scheduler_type="constant" if scheduler == "epoch_decay" else scheduler,
        warmup_steps=warmup_steps, optim=optimizer, weight_decay=float(weight_decay),
        seed=seed,data_seed=seed,save_strategy="no",eval_strategy="epoch" if eval_each_epoch else "no",report_to=[],
        logging_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,dataloader_num_workers=0,
        **_length_grouping_args(tr))
    class OrderedValidationTrainer(tr.Trainer):
        validation_seconds = 0.0
        def _get_eval_sampler(self, eval_dataset):
            if hasattr(eval_dataset, "lengths"):
                return LengthOrderedSampler(eval_dataset)
            return LengthOrderedSampler(validation_dataset_rows)
        def evaluate(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                self.validation_seconds += time.perf_counter() - started

    trainer_type=OrderedValidationTrainer
    if scheduler == "epoch_decay":
        class EpochDecayTrainer(OrderedValidationTrainer):
            def create_scheduler(self,num_training_steps,optimizer=None):
                if self.lr_scheduler is None:
                    optimizer=optimizer or self.optimizer
                    decay=float(learning_rate_decay)
                    warmup_steps=self.args.get_warmup_steps(num_training_steps)
                    self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                        optimizer,lr_lambda=_epoch_decay_lambda(
                            decay=decay,steps_per_epoch=steps_per_epoch,warmup_steps=warmup_steps))
                return self.lr_scheduler
        trainer_type=EpochDecayTrainer
    trainer=trainer_type(model=model,args=args,train_dataset=train_dataset,data_collator=data_collator,
        eval_dataset=validation_dataset_rows if eval_each_epoch else None,
        compute_metrics=compute_metrics if eval_each_epoch else None,
        callbacks=([snapshot_callback,progress_callback,early_stopping_callback]
                   if eval_each_epoch else [progress_callback]))
    startup_seconds = time.perf_counter() - start
    training_started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_started
    out.mkdir(parents=True,exist_ok=True)
    (out/"training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    eval_history = [entry for entry in trainer.state.log_history if "eval_balanced_accuracy" in entry]
    train_history = {int(round(float(entry["epoch"]))): entry for entry in trainer.state.log_history
                     if "loss" in entry and "epoch" in entry}
    validation_metrics_by_epoch = []
    for epoch, entry in enumerate(eval_history, start=1):
        epoch_number=int(round(float(entry.get("epoch", epoch))))
        train_entry=train_history.get(epoch_number,{})
        validation_metrics_by_epoch.append({**entry,
            "train_loss": train_entry.get("loss"),
            "learning_rate_used": train_entry.get("learning_rate",
                float(learning_rate) * float(learning_rate_decay) ** (epoch_number-1)
                if scheduler == "epoch_decay" else None)})
    best_epoch = None
    best_validation_accuracy = None
    best_validation_balanced_accuracy = None
    best_checkpoint_restoration_seconds = 0.0
    if eval_each_epoch:
        metric_key = f"eval_{checkpoint_selection_metric}"
        candidates = [item for item in validation_metrics_by_epoch if item.get(metric_key) is not None]
        if not candidates:
            raise RuntimeError(f"epoch validation ran without recording {checkpoint_selection_metric}")
        selected = max(candidates, key=lambda item: (float(item[metric_key]),
                                                     -int(round(float(item.get("epoch", 0))))))
        best_epoch = int(round(float(selected["epoch"])))
        best_validation_accuracy = float(selected["eval_accuracy"])
        best_validation_balanced_accuracy = float(selected["eval_balanced_accuracy"])
        selected_state = snapshot_callback.best_state
        if selected_state is None or selected_state["epoch"] != best_epoch:
            raise RuntimeError("selected Qwen epoch has no captured adapter and classifier state")
        restore_started = time.perf_counter()
        _restore_qwen_selected_state(backbone, model.classifier,
            selected_state["adapter"], selected_state["classifier"])
        best_checkpoint_restoration_seconds = time.perf_counter() - restore_started
    selection = {"metric": f"validation_{checkpoint_selection_metric}" if eval_each_epoch else None,
                 "best_epoch": best_epoch,
                 "best_validation_accuracy": best_validation_accuracy,
                 "best_validation_balanced_accuracy": best_validation_balanced_accuracy,
                 "tie_break": "earliest epoch" if eval_each_epoch else None}
    (out / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Score the exact restored model object. Trainer.predict uses its cached
    # accelerator wrapper, which can retain pre-restore weights after the
    # selected epoch adapter/head have been loaded above.
    prediction_device = next(model.classifier.parameters()).device
    amp_dtype = (torch.bfloat16 if use_bf16 else
                 torch.float16 if use_fp16 else None)

    def predict_cached(dataset):
        return _predict_cached_dataset(model, dataset, batch_size=effective_batch_size,
            data_collator=data_collator, device=prediction_device, amp_dtype=amp_dtype)

    calibration_indices = _calibration_indices(y_train, max_samples=calibration_max_samples,
        seed=calibration_seed, full_dataset=calibration_full_dataset)
    calibration_labels = np.asarray(y_train, dtype=object)[calibration_indices].tolist()
    calibration_dataset = torch.utils.data.Subset(train_dataset, calibration_indices.tolist())
    calibration_inference_started = time.perf_counter()
    train_logits = predict_cached(calibration_dataset)
    calibration_inference_seconds = time.perf_counter() - calibration_inference_started
    temperature_fit_started = time.perf_counter()
    temperature, calibration_metrics = _fit_temperature(train_logits, calibration_labels, classes,
        max_samples=None, seed=calibration_seed, full_dataset=True, return_metrics=True)
    temperature_fit_seconds = time.perf_counter() - temperature_fit_started
    calibration_metrics["calibration_sample_size"] = int(len(calibration_indices))
    calibration_metrics["calibration_full_dataset"] = bool(len(calibration_indices) == len(y_train))
    calibration_metrics["calibration_seed"] = int(calibration_seed)
    calibration_metrics["calibration_inference_seconds"] = calibration_inference_seconds
    calibration_metrics["temperature_fit_seconds"] = temperature_fit_seconds
    selected_prediction = _selected_epoch_scores(snapshot_callback.validation_predictions,
        best_epoch, (len(validation_dataset_rows), len(classes)))
    final_validation_reused_epoch_predictions = selected_prediction is not None
    pred_start = time.perf_counter()
    logits = (selected_prediction if final_validation_reused_epoch_predictions
              else predict_cached(validation_dataset_rows))
    inference_seconds = (0.0 if final_validation_reused_epoch_predictions
                         else time.perf_counter() - pred_start)
    checkpoint_started = time.perf_counter()
    backbone.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
    torch.save(model.classifier.state_dict(),out/"classifier.pt")
    (out/"checkpoint_manifest.json").write_text(json.dumps(checkpoint_metadata,indent=2)+"\n")
    checkpoint_save_seconds = time.perf_counter() - checkpoint_started
    if not np.isfinite(logits).all(): raise ValueError("Qwen adapter produced non-finite scores")
    probs=softmax(logits/temperature,axis=1)
    validation_loss_raw = (selected.get("eval_loss") if eval_each_epoch else None)
    if validation_loss_raw is None and validation_labels is not None:
        from sklearn.metrics import log_loss
        validation_loss_raw = float(log_loss(
            validation_labels, softmax(logits, axis=1), labels=list(classes)))
    validation_loss_calibrated = None
    if validation_labels is not None:
        from sklearn.metrics import log_loss
        validation_loss_calibrated = float(log_loss(
            validation_labels, probs, labels=list(classes)))
        if eval_each_epoch:
            selected["validation_loss_raw"] = float(validation_loss_raw)
            selected["validation_loss_calibrated"] = validation_loss_calibrated
            selected["calibration_temperature"] = float(temperature)
        for entry in validation_metrics_by_epoch:
            if entry.get("eval_loss") is not None:
                entry["validation_loss_raw"] = float(entry["eval_loss"])
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"scores":logits,"probabilities":probs,"model":model,"tokenizer":tokenizer,
        "fit_seconds":time.perf_counter()-start,"inference_seconds":inference_seconds,
        "classes":list(classes),"trainable_parameters":trainable,"calibration_temperature":temperature,
        "calibration_method":"train-only scalar temperature",
        **calibration_metrics,
        "validation_metrics_by_epoch":validation_metrics_by_epoch,
        "validation_loss_raw": (float(validation_loss_raw)
                                if validation_loss_raw is not None else None),
        "validation_loss_calibrated": validation_loss_calibrated,
        "validation_predictions_by_epoch":snapshot_callback.validation_predictions,
        "early_stopping": (_early_stopping_summary(early_stopping_callback, early_stopping_patience, epochs,
                                                    metric=checkpoint_selection_metric)
                           if early_stopping_callback is not None else None),
        "epoch_checkpoints":snapshot_callback.checkpoints,
        "best_epoch":best_epoch,"best_validation_accuracy":best_validation_accuracy,
        "best_validation_balanced_accuracy":best_validation_balanced_accuracy,
        "checkpoint_selection_metric":f"validation_{checkpoint_selection_metric}" if eval_each_epoch else None,
        "checkpoint_scoring_method":"best_epoch_validation_predictions" if final_validation_reused_epoch_predictions else "length_grouped_restored_model_forward",
        "final_validation_reused_epoch_predictions":bool(final_validation_reused_epoch_predictions),
        "runtime_breakdown":{"startup_seconds":startup_seconds,
            "training_including_validation_seconds":training_seconds,
            "validation_seconds":float(getattr(trainer,"validation_seconds",0.0)),
            "best_checkpoint_restoration_seconds":float(best_checkpoint_restoration_seconds),
            "calibration_inference_seconds":float(calibration_inference_seconds),
            "temperature_fit_seconds":float(temperature_fit_seconds),
            "final_validation_inference_seconds":float(inference_seconds),
            "checkpoint_serialization_seconds":float(checkpoint_save_seconds+snapshot_callback.save_seconds),
            "checkpoint_snapshot_seconds":snapshot_callback.snapshot_seconds,
            "training_excluding_validation_seconds":max(0.0,training_seconds-
                float(getattr(trainer,"validation_seconds",0.0)))},
        "requested_batch_size":int(batch_size),"effective_batch_size":effective_batch_size,
        "lr_scheduler_type":scheduler,"warmup_ratio":float(warmup_ratio),
        "batch_size_probe":batch_probe,
        "quantization":quantization,"quantization_config":quantization_config,
        "attention_backend":attention_backend,
        "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
