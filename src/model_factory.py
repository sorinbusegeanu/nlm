"""Classical and optional neural model adapters with aligned outputs."""
from __future__ import annotations

import time
import json
import math
import os
import inspect
import shutil
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


def _fit_temperature(train_logits, train_labels, classes):
    """Fit scalar temperature using training rows only; validation logits are never used."""
    logits = np.asarray(train_logits, dtype=float)
    indexes = {label: i for i, label in enumerate(classes)}
    targets = np.asarray([indexes[label] for label in train_labels], dtype=int)
    def loss(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))
    result = minimize_scalar(loss, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def train_svc(train_text, y_train, validation_text, classes, *, c=.5, calibration_cv=3, seed=2026, feature_config=None, cache_dir=None, fold_token=""):
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
    calibrated.fit(x_train, y_train)
    base.fit(x_train, y_train)
    pred_start = time.perf_counter()
    raw = _aligned(np.asarray(base.decision_function(x_val)), base.classes_, classes)
    prob = _aligned(calibrated.predict_proba(x_val), calibrated.classes_, classes)
    inference_seconds = time.perf_counter() - pred_start
    elapsed = time.perf_counter() - start
    return {"scores": raw, "probabilities": prob, "model": calibrated, "raw_model": base,
            "features": features, "cache_key": cache_key, "fit_seconds": elapsed, "classes": list(classes),
            "inference_seconds": inference_seconds, "peak_gpu_memory_bytes": None}


def train_catboost_stylometry(train_text, y_train, validation_text, classes, *, output_dir,
                              iterations=800, depth=7, learning_rate=.05, l2_leaf_reg=5.0,
                              thread_count=8, seed=2026, temperature_calibration=True,
                              validation_labels=None, early_stopping_rounds=100,
                              feature_recipe=None, feature_matrices=None, init_model=None,
                              checkpoint_metadata=None):
    """Fit CatBoost on deterministic row-local style statistics; no validation fitting."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost model requires catboost; install the project's requirements.txt") from exc
    start = time.perf_counter()
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
    model = CatBoostClassifier(iterations=int(iterations), depth=int(depth),
        learning_rate=float(learning_rate), l2_leaf_reg=float(l2_leaf_reg),
        loss_function="MultiClass", eval_metric="Accuracy", auto_class_weights="Balanced",
        random_seed=int(seed), thread_count=int(thread_count), verbose=False,
        allow_writing_files=False)
    fit_kwargs = {}
    if init_model is not None:
        fit_kwargs["init_model"] = str(init_model)
    if validation_labels is not None:
        fit_kwargs.update({"eval_set": (x_validation, list(validation_labels)),
                           "use_best_model": True,
                           "early_stopping_rounds": int(early_stopping_rounds)})
    model.fit(x_train, list(y_train), **fit_kwargs)
    train_raw = np.asarray(model.predict(x_train, prediction_type="RawFormulaVal"))
    pred_start = time.perf_counter()
    validation_raw = np.asarray(model.predict(x_validation, prediction_type="RawFormulaVal"))
    inference_seconds = time.perf_counter() - pred_start
    model_classes = list(model.classes_)
    if train_raw.ndim == 1:
        train_raw = np.column_stack([-train_raw, train_raw])
    if validation_raw.ndim == 1:
        validation_raw = np.column_stack([-validation_raw, validation_raw])
    train_raw = _aligned(train_raw, model_classes, classes)
    raw = _aligned(validation_raw, model_classes, classes)
    temperature = _fit_temperature(train_raw, list(y_train), classes) if temperature_calibration else 1.0
    probabilities = softmax(raw / temperature, axis=1)
    from sklearn.metrics import accuracy_score
    validation_accuracy = (float(accuracy_score(validation_labels,
        np.asarray(classes)[raw.argmax(axis=1)])) if validation_labels is not None else None)
    best_iteration = int(model.get_best_iteration()) if validation_labels is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "model.cbm"))
    metadata = {
        "feature_version": STYLOMETRY_VERSION,
        "feature_names": feature_names,
        "feature_recipe": feature_recipe,
        "feature_count": int(x_train.shape[1]), "temperature": temperature,
        "selected_iteration": best_iteration,
        "selected_validation_accuracy": validation_accuracy,
        "selection_metric": "validation_accuracy" if validation_labels is not None else None,
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
    return {"scores": raw, "probabilities": probabilities, "model": model,
        "fit_seconds": time.perf_counter() - start, "inference_seconds": inference_seconds,
        "classes": list(classes), "calibration_temperature": temperature,
        "calibration_method": "train-only scalar temperature" if temperature_calibration else "none",
        "selected_iteration": best_iteration, "selected_validation_accuracy": validation_accuracy,
        "feature_count": int(x_train.shape[1]), "feature_names": feature_names,
        "feature_recipe": feature_recipe,
        "checkpoint_selection_metric": "validation_accuracy" if validation_labels is not None else None,
        "checkpoint": str(output_dir), "tree_count": int(model.tree_count_),
        "added_iterations": int(iterations), "parent_checkpoint": str(init_model) if init_model is not None else None,
        "peak_gpu_memory_bytes": None}


FROZEN_BACKBONE_MODELS = {"embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"}


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


def extract_frozen_backbone_features(model_name, checkpoint, revision, texts, *, max_length=384,
                                     batch_size=8, model_dir=None, fix_mistral_regex=False):
    """Extract deterministic, text-only mean-pooled features without gradient updates."""
    import torch
    model, text_model, processor = _load_frozen_backbone(
        model_name, model_dir or checkpoint, revision,
        fix_mistral_regex=fix_mistral_regex)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prefix = "task: classification | query: " if model_name == "embeddinggemma2" else ""
    prepared = [prefix + str(text) for text in texts]
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(prepared), max(1, int(batch_size))):
            batch = processor(text=prepared[start:start + max(1, int(batch_size))],
                              truncation=True, max_length=max_length, padding=True,
                              return_tensors="pt")
            # These three text models need token IDs and masks; Gemma 4 also
            # consumes its text modality markers. Ignore processor-only extras.
            keys = {"input_ids", "attention_mask"}
            if model_name == "gemma4_e4b":
                keys.add("mm_token_type_ids")
            batch = {key: value.to(device) for key, value in batch.items() if key in keys}
            result = text_model(**batch, use_cache=False, return_dict=True)
            hidden = result.last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            if model_name == "embeddinggemma2":
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy())
    elapsed = time.perf_counter() - started
    features = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
    del model, text_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not np.isfinite(features).all():
        raise ValueError(f"{model_name} produced non-finite text features")
    return features, elapsed


def train_frozen_backbone_fold(model_name, train_text, y_train, validation_text, classes, *,
                               checkpoint, revision, output_dir, max_length=384, batch_size=8,
                               c=.5, calibration_cv=3, seed=2026, fix_mistral_regex=False):
    """Fit a train-only LinearSVC readout over frozen model-specific text features."""
    if model_name not in FROZEN_BACKBONE_MODELS:
        raise ValueError(f"unsupported frozen backbone: {model_name}")
    start = time.perf_counter()
    same_rows = list(train_text) == list(validation_text)
    all_text = list(train_text) if same_rows else list(train_text) + list(validation_text)
    all_features, extraction_seconds = extract_frozen_backbone_features(
        model_name, checkpoint, revision, all_text, max_length=max_length,
        batch_size=batch_size, fix_mistral_regex=fix_mistral_regex)
    train_features = all_features[:len(train_text)]
    validation_features = all_features if same_rows else all_features[len(train_text):]
    inference_started = time.perf_counter()
    base = LinearSVC(C=float(c), class_weight="balanced", random_state=seed)
    base.fit(train_features, y_train)
    try:
        calibrated = CalibratedClassifierCV(
            estimator=LinearSVC(C=float(c), class_weight="balanced", random_state=seed),
            method="sigmoid", cv=int(calibration_cv))
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=LinearSVC(C=float(c), class_weight="balanced", random_state=seed),
            method="sigmoid", cv=int(calibration_cv))
    calibrated.fit(train_features, y_train)
    raw = _aligned(np.asarray(base.decision_function(validation_features)), base.classes_, classes)
    probabilities = _aligned(calibrated.predict_proba(validation_features), calibrated.classes_, classes)
    validation_fraction = 1.0 if same_rows else len(validation_text) / max(1, len(train_text) + len(validation_text))
    inference_seconds = extraction_seconds * validation_fraction + time.perf_counter() - inference_started
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = output_dir / "readout.joblib"
    import joblib
    joblib.dump({"model": calibrated, "raw_model": base, "classes": list(classes),
                 "model_name": model_name, "checkpoint": checkpoint, "revision": revision,
                 "max_length": int(max_length), "feature_pooling": "attention-mask mean",
                 "task_prefix": "task: classification | query: " if model_name == "embeddinggemma2" else "",
                 "fix_mistral_regex": bool(fix_mistral_regex)}, joblib_path)
    return {"scores": raw, "probabilities": probabilities, "model": calibrated,
            "raw_model": base, "fit_seconds": time.perf_counter() - start,
            "inference_seconds": inference_seconds, "classes": list(classes),
            "calibration_method": "sigmoid train-only CV", "checkpoint": str(output_dir),
            "readout_path": str(joblib_path), "peak_gpu_memory_bytes": None}


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
                                   ("eval_loss", "validation_loss"), ("eval_accuracy", "accuracy"),
                                   ("eval_balanced_accuracy", "balanced_accuracy")):
                if logs.get(source) is not None:
                    row[target] = float(logs[source])
            self._write_progress()
            return control
    return LiveTrainingProgressCallback()


def _accuracy_early_stopping_callback(TrainerCallback, patience):
    """Stop after `patience` consecutive epoch validations without better accuracy."""
    patience = int(patience)
    if patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")

    class AccuracyEarlyStoppingCallback(TrainerCallback):
        def __init__(self):
            self.patience = patience
            self.best_accuracy = float("-inf")
            self.best_epoch = None
            self.bad_epochs = 0
            self.stopped_epoch = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            accuracy = (metrics or {}).get("eval_accuracy")
            if accuracy is None:
                return control
            epoch = int(round(float(state.epoch or 0)))
            accuracy = float(accuracy)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= patience:
                    self.stopped_epoch = epoch
                    control.should_training_stop = True
            return control

    return AccuracyEarlyStoppingCallback()


def _early_stopping_summary(callback, patience, max_epochs):
    return {
        "metric": "validation_accuracy",
        "patience_epochs": int(patience),
        "best_epoch": callback.best_epoch,
        "best_validation_accuracy": (callback.best_accuracy
                                     if callback.best_epoch is not None else None),
        "stopped_early": callback.stopped_epoch is not None and callback.stopped_epoch < max_epochs,
        "stopped_epoch": callback.stopped_epoch,
    }


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
                       validate_before_training=False, select_best_checkpoint=False,
                       optimizer="adamw_torch", weight_decay=0.0, lr_scheduler_type=None,
                       warmup_ratio=0.0, warmup_epochs=None, early_stopping_patience=3,
                       auto_batch_size=False, max_batch_size=1024,
                       batch_probe_safety_fraction=0.95):
    """Fine-tune an encoder, optionally resuming and recording a pre-train baseline."""
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
    def compute_metrics(eval_prediction):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        predictions = np.asarray(logits).argmax(axis=1)
        labels = np.asarray(eval_prediction.label_ids)
        return {"accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions))}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import TrainerCallback
    live_callback = _live_training_progress_callback(TrainerCallback, out)
    if learning_rate_decay is not None and not 0 < float(learning_rate_decay) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1]")
    if validate_before_training and validation_labels is None:
        raise ValueError("validate_before_training requires validation_labels")
    best_dir = out / "best_checkpoint"
    best_state = {"accuracy": -1.0, "epoch": None}

    class BestCheckpointCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            accuracy = (metrics or {}).get("eval_accuracy")
            if select_best_checkpoint and accuracy is not None and float(accuracy) > best_state["accuracy"]:
                best_state["accuracy"] = float(accuracy)
                best_state["epoch"] = int(round(float(state.epoch or 0)))
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir, safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
                (best_dir / "selection.json").write_text(json.dumps({
                    "metric": "validation_accuracy", "epoch": best_state["epoch"],
                    "value": best_state["accuracy"]}, indent=2) + "\n")
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

    trainer_type = tr.Trainer
    if learning_rate_decay is not None:
        decay = float(learning_rate_decay)

        class EpochDecayTrainer(tr.Trainer):
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

    callbacks = [live_callback]
    early_stopping_callback = None
    if validation_labels is not None:
        early_stopping_callback = _accuracy_early_stopping_callback(TrainerCallback, early_stopping_patience)
        callbacks.append(early_stopping_callback)
        callbacks.append(BestCheckpointCallback)
    trainer = trainer_type(model=model, args=args, train_dataset=train, data_collator=data_collator,
        eval_dataset=val if validation_labels is not None else None,
        compute_metrics=compute_metrics if validation_labels is not None else None,
        processing_class=tokenizer, callbacks=callbacks)
    baseline_logits = None
    baseline_metrics = None
    if validate_before_training:
        baseline_logits = np.asarray(trainer.predict(val).predictions)
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
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir)/"training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    if select_best_checkpoint:
        if best_state["epoch"] is None:
            raise RuntimeError("best checkpoint selection requested but no epoch validation metric was recorded")
        best_model = tr.AutoModelForSequenceClassification.from_pretrained(
            str(best_dir), trust_remote_code=True, local_files_only=True)
        model.load_state_dict(best_model.state_dict())
        model.save_pretrained(output_dir, safe_serialization=True)
        (Path(output_dir) / "checkpoint_selection.json").write_text(json.dumps({
            "metric": "validation_accuracy", "selected_epoch": best_state["epoch"],
            "selected_validation_accuracy": best_state["accuracy"],
            "best_checkpoint": str(best_dir)}, indent=2) + "\n")
    train_logits = np.asarray(trainer.predict(train).predictions)
    train_labels = list(y_train)
    temperature = _fit_temperature(train_logits, train_labels, classes)
    pred_start = time.perf_counter()
    logits = np.asarray(trainer.predict(val).predictions)
    inference_seconds = time.perf_counter() - pred_start
    model.save_pretrained(output_dir, safe_serialization=True); tokenizer.save_pretrained(output_dir)
    if not np.isfinite(logits).all(): raise ValueError("encoder produced non-finite scores")
    probs = softmax(logits / temperature, axis=1)
    epoch_history = [dict(entry) for entry in trainer.state.log_history if "eval_accuracy" in entry]
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
            "baseline_validation_metrics": baseline_metrics,
            "validation_metrics_by_epoch": epoch_history,
            "best_epoch": best_state["epoch"],
            "best_validation_accuracy": best_state["accuracy"] if best_state["epoch"] is not None else None,
            "early_stopping": (_early_stopping_summary(early_stopping_callback, early_stopping_patience, epochs)
                               if early_stopping_callback is not None else None),
            "checkpoint_selection_metric": "validation_accuracy" if select_best_checkpoint else None,
            "best_checkpoint_path": str(best_dir) if best_state["epoch"] is not None else None,
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
    saved_trainable = [parameter.detach().cpu().clone() for parameter in trainable]
    # Trainer initializes AdamW moments lazily on its first update. Simulate a
    # full-precision AdamW update during probing so the selected batch includes
    # that persistent memory. This is conservative for lower-memory optimizers.
    optimizer_probe = "adamw_torch"

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
            probe_optimizer = torch.optim.AdamW(trainable, lr=1e-4,
                                                weight_decay=float(weight_decay))
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
                    parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
            del batch
            torch.cuda.empty_cache()

    try:
        model.to(device)
        model.train()
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
                       "configured_optimizer": str(optimizer_name),
                       "probe_optimizer": optimizer_probe,
                       "optimizer_state_included": True,
                       "peak_probe_allocated_bytes": max(
                           (x.get("peak_allocated_bytes", 0) for x in report["attempts"]), default=0)})
        return successful, report
    finally:
        model.zero_grad(set_to_none=True)
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.empty_cache()


def load_text_only_backbone(transformers_module, checkpoint, revision, *, torch_dtype, local_files_only=True):
    """Load a multimodal checkpoint on CPU and retain its text tower only."""
    composite = transformers_module.AutoModel.from_pretrained(
        checkpoint, revision=revision, torch_dtype=torch_dtype, local_files_only=local_files_only)
    text_backbone = getattr(composite, "language_model", None)
    if text_backbone is None:
        raise ValueError(f"{checkpoint} has no language_model text backbone")
    del composite
    return text_backbone


def train_qwen_lora_fold(train_text, y_train, validation_text, classes, *, checkpoint="Qwen/Qwen3-0.6B",
                         revision, output_dir="artifacts/checkpoints/qwen", epochs=3, batch_size=4,
                         max_length=384, learning_rate=1e-4, seed=2026, gradient_accumulation_steps=1,
                         lora_rank=16, lora_alpha=32,
                         lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), save_merged=False,
                         resume_checkpoint=None, validation_labels=None, gradient_checkpointing=False,
                         local_files_only=False, learning_rate_decay=1.0, auto_batch_size=False,
                         max_batch_size=1024, batch_probe_safety_fraction=0.95,
                         readout_hidden_size=None, task_prefix="", fix_mistral_regex=False,
                         lora_exclude_modules=None, text_only_backbone=False,
                         lr_scheduler_type=None, warmup_ratio=0.0,
                         warmup_epochs=None, optimizer="adamw_torch", weight_decay=0.0,
                         early_stopping_patience=3, trust_remote_code=False):
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
    from torch.utils.data import Dataset
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
    if text_only_backbone:
        backbone = load_text_only_backbone(tr, checkpoint, revision, torch_dtype=model_dtype,
                                           local_files_only=local_files_only)
    else:
        backbone = tr.AutoModel.from_pretrained(checkpoint, revision=revision, torch_dtype=model_dtype,
                                                 local_files_only=local_files_only,
                                                 trust_remote_code=trust_remote_code)
    if resume_checkpoint:
        backbone = PeftModel.from_pretrained(backbone, resume_checkpoint, is_trainable=True, local_files_only=True)
    else:
        lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_rank, lora_alpha=lora_alpha,
                          lora_dropout=.05, target_modules=list(lora_target_modules),
                          exclude_modules=lora_exclude_modules)
        backbone = get_peft_model(backbone, lora)
    backbone.config.use_cache = False
    if gradient_checkpointing:
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
            super().__init__(); self.backbone=backbone; self.classifier=nn.Linear(hidden_size,len(classes))
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
            self.ys=ys
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self,i):
            item={k:torch.tensor(v[i],dtype=torch.long) for k,v in self.enc.items()}
            if self.ys is not None: item["labels"]=torch.tensor(label_map[self.ys[i]],dtype=torch.long)
            return item
    model=Head()
    if resume_checkpoint:
        classifier_path = Path(resume_checkpoint)/"classifier.pt"
        if not classifier_path.is_file(): raise FileNotFoundError(f"continuation checkpoint has no classifier head: {classifier_path}")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=True))
    def compute_metrics(eval_prediction):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        labels = np.asarray(eval_prediction.label_ids)
        predicted = np.asarray(logits).argmax(axis=1)
        return {"accuracy": float(accuracy_score(labels, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted))}
    out=Path(output_dir)
    class EpochSnapshotCallback(TrainerCallback):
        def __init__(self): self.checkpoints=[]
        def on_evaluate(self,args,state,control,metrics=None,**kwargs):
            epoch=int(round(state.epoch or len(self.checkpoints)+1))
            epoch_dir=out/f"epoch-{epoch:02d}"
            epoch_dir.mkdir(parents=True,exist_ok=True)
            model.backbone.save_pretrained(epoch_dir,safe_serialization=True)
            tokenizer.save_pretrained(epoch_dir)
            torch.save({k:v.detach().cpu() for k,v in model.classifier.state_dict().items()},epoch_dir/"classifier.pt")
            (epoch_dir/"validation_metrics.json").write_text(json.dumps(metrics or {},indent=2)+"\n")
            self.checkpoints.append(str(epoch_dir))
            return control
    snapshot_callback=EpochSnapshotCallback()
    progress_callback=_live_training_progress_callback(TrainerCallback, out)
    eval_each_epoch = validation_labels is not None
    early_stopping_callback = (_accuracy_early_stopping_callback(TrainerCallback, early_stopping_patience)
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
    train_dataset.lengths=[len(ids) for ids in train_dataset.enc["input_ids"]]
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
    checkpointing_disabled_for_batch = False
    checkpointing_probe_batch_size = int(effective_batch_size)
    if gradient_checkpointing and effective_batch_size > 8:
        disable_checkpointing = getattr(backbone, "gradient_checkpointing_disable", None)
        if disable_checkpointing is None:
            raise RuntimeError("batch size exceeded 8 but the model cannot disable gradient checkpointing")
        disable_checkpointing()
        gradient_checkpointing = False
        checkpointing_disabled_for_batch = True
        # The memory probe must reflect the mode that will actually train.
        if auto_batch_size:
            effective_batch_size, batch_probe = _probe_max_batch_size(
                model, train_dataset, requested_batch_size=batch_size,
                max_batch_size=max_batch_size or batch_size,
                safety_fraction=batch_probe_safety_fraction, use_bf16=use_bf16, use_fp16=use_fp16,
                optimizer_name=optimizer, weight_decay=weight_decay)
        batch_probe["gradient_checkpointing_reprobed_without_checkpointing"] = True
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
    trainer_type=tr.Trainer
    if scheduler == "epoch_decay":
        class EpochDecayTrainer(tr.Trainer):
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
        eval_dataset=Rows(validation_text,validation_labels) if eval_each_epoch else None,
        compute_metrics=compute_metrics if eval_each_epoch else None,
        callbacks=([snapshot_callback,progress_callback,early_stopping_callback]
                   if eval_each_epoch else [progress_callback]))
    trainer.train()
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
    if eval_each_epoch:
        candidates = [item for item in validation_metrics_by_epoch if item.get("eval_accuracy") is not None]
        if not candidates:
            raise RuntimeError("epoch validation ran without recording ordinary accuracy")
        selected = max(candidates, key=lambda item: (float(item["eval_accuracy"]),
                                                     -int(round(float(item.get("epoch", 0))))))
        best_epoch = int(round(float(selected["epoch"])))
        best_validation_accuracy = float(selected["eval_accuracy"])
        selected_dir = Path(snapshot_callback.checkpoints[best_epoch - 1])
        adapter_path = selected_dir / "adapter_model.safetensors"
        if adapter_path.is_file():
            from safetensors.torch import load_file
            adapter_state = load_file(str(adapter_path), device="cpu")
        else:
            adapter_path = selected_dir / "adapter_model.bin"
            if not adapter_path.is_file():
                raise FileNotFoundError(f"epoch {best_epoch} has no saved LoRA adapter: {selected_dir}")
            adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        from peft.utils.save_and_load import set_peft_model_state_dict
        set_peft_model_state_dict(backbone, adapter_state, adapter_name="default")
        model.classifier.load_state_dict(torch.load(selected_dir / "classifier.pt", map_location="cpu", weights_only=True))
    selection = {"metric": "validation_accuracy" if eval_each_epoch else None,
                 "best_epoch": best_epoch,
                 "best_validation_accuracy": best_validation_accuracy,
                 "tie_break": "earliest epoch" if eval_each_epoch else None}
    (out / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Score the exact restored model object. Trainer.predict uses its cached
    # accelerator wrapper, which can retain pre-restore weights after the
    # selected epoch adapter/head have been loaded above.
    from torch.utils.data import DataLoader
    prediction_device = next(model.parameters()).device
    amp_dtype = (torch.bfloat16 if use_bf16 else
                 torch.float16 if use_fp16 else None)

    def predict_restored_model(texts):
        model.eval()
        loader = DataLoader(Rows(texts), batch_size=effective_batch_size,
                            shuffle=False, collate_fn=data_collator,
                            num_workers=0)
        chunks = []
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(prediction_device) for key, value in batch.items()}
                amp = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                       if prediction_device.type == "cuda" and amp_dtype is not None
                       else nullcontext())
                with amp:
                    output = model(**batch)
                chunks.append(output.logits.float().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(classes)))

    train_logits = predict_restored_model(train_text)
    temperature = _fit_temperature(train_logits, list(y_train), classes)
    pred_start = time.perf_counter()
    logits = predict_restored_model(validation_text)
    inference_seconds = time.perf_counter() - pred_start
    backbone.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
    torch.save(model.classifier.state_dict(),out/"classifier.pt")
    if save_merged:
        merged = backbone.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(out/"merged_backbone", safe_serialization=True)
    if not np.isfinite(logits).all(): raise ValueError("Qwen adapter produced non-finite scores")
    probs=softmax(logits/temperature,axis=1)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"scores":logits,"probabilities":probs,"model":model,"tokenizer":tokenizer,
        "fit_seconds":time.perf_counter()-start,"inference_seconds":inference_seconds,
        "classes":list(classes),"trainable_parameters":trainable,"calibration_temperature":temperature,
        "calibration_method":"train-only scalar temperature",
        "validation_metrics_by_epoch":validation_metrics_by_epoch,
        "early_stopping": (_early_stopping_summary(early_stopping_callback, early_stopping_patience, epochs)
                           if early_stopping_callback is not None else None),
        "epoch_checkpoints":snapshot_callback.checkpoints,
        "best_epoch":best_epoch,"best_validation_accuracy":best_validation_accuracy,
        "checkpoint_selection_metric":"validation_accuracy" if eval_each_epoch else None,
        "checkpoint_scoring_method":"restored_model_direct_forward",
        "requested_batch_size":int(batch_size),"effective_batch_size":effective_batch_size,
        "lr_scheduler_type":scheduler,"warmup_ratio":float(warmup_ratio),
        "batch_size_probe":batch_probe,
        "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
