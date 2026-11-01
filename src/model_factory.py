"""Classical and optional neural model adapters with aligned outputs."""
from __future__ import annotations

import time
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from .feature_factory import FoldFeatures, load_or_fit


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


def _require_neural():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Neural training needs the optional neural dependencies. Install this project's [neural] extra on the training computer.") from exc
    return torch, transformers


def train_encoder_fold(train_text, y_train, validation_text, classes, *, checkpoint, revision, output_dir,
                       epochs=3, batch_size=16, max_length=384, learning_rate=2e-5, seed=2026,
                       gradient_accumulation_steps=1):
    """Fine-tune a pinned AutoModelForSequenceClassification checkpoint."""
    torch, tr = _require_neural()
    start = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    label_map = {label: i for i, label in enumerate(classes)}
    y_indices = np.asarray([label_map[y] for y in y_train], dtype=np.int64)
    tokenizer = tr.AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    model = tr.AutoModelForSequenceClassification.from_pretrained(
        checkpoint, revision=revision, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id=label_map, trust_remote_code=True)
    train = _TextDataset(train_text, y_indices, tokenizer, max_length)
    val = _TextDataset(validation_text, None, tokenizer, max_length)
    args = tr.TrainingArguments(output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate,
        seed=seed, data_seed=seed, save_strategy="no", eval_strategy="no", report_to=[],
        logging_strategy="epoch", fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0)
    trainer = tr.Trainer(model=model, args=args, train_dataset=train, processing_class=tokenizer)
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir)/"training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    train_logits = np.asarray(trainer.predict(train).predictions)
    train_labels = list(y_train)
    temperature = _fit_temperature(train_logits, train_labels, classes)
    pred_start = time.perf_counter()
    logits = np.asarray(trainer.predict(val).predictions)
    inference_seconds = time.perf_counter() - pred_start
    model.save_pretrained(output_dir, safe_serialization=True); tokenizer.save_pretrained(output_dir)
    if not np.isfinite(logits).all(): raise ValueError("encoder produced non-finite scores")
    probs = softmax(logits / temperature, axis=1)
    return {"scores": logits, "probabilities": probs, "model": model, "tokenizer": tokenizer,
            "fit_seconds": time.perf_counter() - start, "inference_seconds": inference_seconds,
            "classes": list(classes), "calibration_temperature": temperature, "calibration_method": "train-only scalar temperature",
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}


class _TextDataset:
    def __new__(cls, texts, labels, tokenizer, max_length):
        torch, _ = _require_neural()
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):
                self.enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding="max_length")
                self.labels = labels
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.enc.items()}
                if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        return Dataset()


def train_qwen_lora_fold(train_text, y_train, validation_text, classes, *, checkpoint="Qwen/Qwen3-0.6B",
                         revision, output_dir="artifacts/checkpoints/qwen", epochs=3, batch_size=4,
                         max_length=384, learning_rate=1e-4, seed=2026, gradient_accumulation_steps=1,
                         lora_rank=16, lora_alpha=32,
                         lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), save_merged=False,
                         resume_checkpoint=None, validation_labels=None, gradient_checkpointing=False,
                         local_files_only=False, learning_rate_decay=1.0):
    """LoRA fine-tune causal hidden states with masked-mean classifier readout."""
    torch, tr = _require_neural()
    try:
        from peft import LoraConfig, TaskType, PeftModel, get_peft_model
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError("Qwen LoRA training needs the project's optional neural dependencies") from exc
    from torch import nn
    from torch.utils.data import Dataset
    from transformers.modeling_outputs import SequenceClassifierOutput
    start = time.perf_counter(); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    tokenizer = tr.AutoTokenizer.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    backbone = tr.AutoModel.from_pretrained(checkpoint, revision=revision, torch_dtype=model_dtype,
                                             local_files_only=local_files_only)
    if resume_checkpoint:
        backbone = PeftModel.from_pretrained(backbone, resume_checkpoint, is_trainable=True, local_files_only=True)
    else:
        lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_rank, lora_alpha=lora_alpha,
                          lora_dropout=.05, target_modules=list(lora_target_modules))
        backbone = get_peft_model(backbone, lora)
    backbone.config.use_cache = False
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    label_map = {label: i for i, label in enumerate(classes)}
    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone; self.classifier=nn.Linear(backbone.config.hidden_size,len(classes))
        def forward(self,input_ids,attention_mask,labels=None):
            hidden=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            logits=self.classifier(pooled.float())
            loss=nn.functional.cross_entropy(logits,labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss,logits=logits)
    class Rows(Dataset):
        def __init__(self,texts,ys=None):
            encoded=tokenizer(list(texts),truncation=True,max_length=max_length,padding="max_length")
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
    eval_each_epoch = validation_labels is not None
    if not 0 < float(learning_rate_decay) <= 1:
        raise ValueError("learning_rate_decay must be in (0, 1]")
    args=tr.TrainingArguments(output_dir=str(output_dir),num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,learning_rate=learning_rate,
        seed=seed,data_seed=seed,save_strategy="no",eval_strategy="epoch" if eval_each_epoch else "no",report_to=[],
        logging_strategy="epoch",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,dataloader_num_workers=0)
    train_dataset=Rows(train_text,y_train)
    steps_per_epoch=max(1,math.ceil(len(train_dataset)/(batch_size*gradient_accumulation_steps)))
    class EpochDecayTrainer(tr.Trainer):
        def create_scheduler(self,num_training_steps,optimizer=None):
            if self.lr_scheduler is None:
                optimizer=optimizer or self.optimizer
                decay=float(learning_rate_decay)
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                    optimizer,lr_lambda=lambda step: decay ** min(epochs, step // steps_per_epoch))
            return self.lr_scheduler
    trainer=EpochDecayTrainer(model=model,args=args,train_dataset=train_dataset,
        eval_dataset=Rows(validation_text,validation_labels) if eval_each_epoch else None,
        compute_metrics=compute_metrics if eval_each_epoch else None,
        callbacks=[snapshot_callback] if eval_each_epoch else None)
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
            "learning_rate_used": float(learning_rate) * float(learning_rate_decay) ** (epoch_number-1)})
    train_logits=np.asarray(trainer.predict(Rows(train_text,y_train)).predictions)
    temperature=_fit_temperature(train_logits,list(y_train),classes)
    pred_start=time.perf_counter(); logits=np.asarray(trainer.predict(Rows(validation_text)).predictions)
    inference_seconds=time.perf_counter()-pred_start
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
        "epoch_checkpoints":snapshot_callback.checkpoints,
        "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
