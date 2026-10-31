"""Classical baseline and optional transformer training adapters."""
from __future__ import annotations
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from .feature_factory import build_features


def train_svc(train_text, y_train, validation_text, classes, *, c=.5, calibration_cv=3, seed=2026):
    """Fold-local char+word TF-IDF, SVC and train-only probability calibration."""
    start = time.perf_counter()
    x_train, x_val, vectorizers = build_features(train_text, validation_text)
    base = LinearSVC(C=c, class_weight="balanced", random_state=seed)
    try:
        model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=calibration_cv)
    except TypeError:  # scikit-learn < 1.2
        model = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=calibration_cv)
    model.fit(x_train, y_train)
    raw = np.column_stack([m.decision_function(x_val) if len(classes) == 2 else None for m in []]) if False else None
    probabilities = model.predict_proba(x_val)
    aligned = np.zeros((len(validation_text), len(classes)), dtype=float)
    for col, label in enumerate(model.classes_):
        aligned[:, list(classes).index(label)] = probabilities[:, col]
    scores = np.log(np.clip(aligned, 1e-12, 1))
    return {"scores": scores, "probabilities": aligned, "model": model, "vectorizers": vectorizers,
            "fit_seconds": time.perf_counter() - start, "classes": list(classes)}


def _require_neural():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Neural training requires the optional 'neural' dependencies; install with pip install -e '.[neural]'") from exc
    return torch, transformers


def train_encoder_fold(train_text, y_train, validation_text, classes, *, checkpoint, output_dir, epochs=3, batch_size=16, max_length=384, learning_rate=2e-5, seed=2026):
    """Fine-tune a sequence-classification checkpoint on one grouped fold."""
    torch, tr = _require_neural()
    from sklearn.preprocessing import LabelEncoder
    labels = LabelEncoder().fit(list(classes))
    tokenizer = tr.AutoTokenizer.from_pretrained(checkpoint)
    model = tr.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id={c:i for i,c in enumerate(classes)})
    train = _TextDataset(train_text, labels.transform(y_train), tokenizer, max_length)
    val = _TextDataset(validation_text, None, tokenizer, max_length)
    args = tr.TrainingArguments(output_dir=str(output_dir), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, learning_rate=learning_rate, seed=seed, data_seed=seed,
        save_strategy="no", eval_strategy="no", report_to=[], logging_strategy="epoch", fp16=torch.cuda.is_available())
    trainer = tr.Trainer(model=model, args=args, train_dataset=train, tokenizer=tokenizer)
    start = time.perf_counter(); trainer.train(); elapsed = time.perf_counter() - start
    logits = trainer.predict(val).predictions
    model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
    probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()
    return {"scores": logits, "probabilities": probs, "model": model, "tokenizer": tokenizer, "fit_seconds": elapsed, "classes": list(classes)}


class _TextDataset:
    def __new__(cls, texts, labels, tokenizer, max_length):
        torch, _ = _require_neural()
        class Dataset(torch.utils.data.Dataset):
            def __init__(self):
                self.enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True)
                self.labels = labels
            def __len__(self): return len(self.enc["input_ids"])
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
                if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        return Dataset()


def train_qwen_lora_fold(train_text, y_train, validation_text, classes, *, checkpoint="Qwen/Qwen3-0.6B-Base", output_dir="artifacts/checkpoints/qwen", epochs=3, batch_size=4, max_length=384, learning_rate=1e-4, seed=2026):
    """LoRA tune a causal LM with deterministic masked-mean hidden-state readout."""
    torch, tr = _require_neural()
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Qwen LoRA training requires the optional 'neural' dependencies") from exc
    from torch import nn
    from torch.utils.data import Dataset
    torch.manual_seed(seed)
    tokenizer = tr.AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    backbone = tr.AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=.05, target_modules=["q_proj","k_proj","v_proj","o_proj"])
    backbone = get_peft_model(backbone, lora)
    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone; self.classifier=nn.Linear(backbone.config.hidden_size,len(classes))
        def forward(self,input_ids,attention_mask,labels=None):
            hidden=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype); pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            logits=self.classifier(pooled.float()); loss=nn.functional.cross_entropy(logits,labels) if labels is not None else None
            return tr.modeling_outputs.SequenceClassifierOutput(loss=loss,logits=logits)
    label_map={c:i for i,c in enumerate(classes)}
    class Rows(Dataset):
        def __init__(self,texts,ys=None): self.enc=tokenizer(list(texts),truncation=True,max_length=max_length,padding="max_length"); self.ys=ys
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self,i):
            z={k:torch.tensor(v[i]) for k,v in self.enc.items()}
            if self.ys is not None:z["labels"]=torch.tensor([label_map[y] for y in self.ys][i])
            return z
    model=Head()
    args=tr.TrainingArguments(output_dir=str(output_dir),num_train_epochs=epochs,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,learning_rate=learning_rate,seed=seed,data_seed=seed,save_strategy="no",eval_strategy="no",report_to=[],fp16=torch.cuda.is_available(),remove_unused_columns=False)
    trainer=tr.Trainer(model=model,args=args,train_dataset=Rows(train_text,y_train))
    start=time.perf_counter();trainer.train();elapsed=time.perf_counter()-start
    logits=trainer.predict(Rows(validation_text)).predictions
    import os
    os.makedirs(output_dir,exist_ok=True); backbone.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir); torch.save(model.classifier.state_dict(),str(output_dir)+"/classifier.pt")
    probs=torch.softmax(torch.tensor(logits),dim=1).cpu().numpy()
    return {"scores":logits,"probabilities":probs,"model":model,"tokenizer":tokenizer,"fit_seconds":elapsed,"classes":list(classes)}
