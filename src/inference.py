"""Network-free prediction and submission export helpers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import joblib
import numpy as np
import pandas as pd

from .data_factory import REQUIRED_TEST


def predict_svc_bundle(bundle, texts):
    features, model, classes = bundle["features"], bundle["model"], list(bundle["classes"])
    matrix = features.transform([str(t) for t in texts])
    predicted = model.predict_proba(matrix)
    aligned = np.zeros((len(texts), len(classes)), dtype=float)
    indexes = {label: i for i, label in enumerate(classes)}
    for col, label in enumerate(model.classes_): aligned[:, indexes[label]] = predicted[:, col]
    if not np.isfinite(aligned).all() or not np.allclose(aligned.sum(1), 1, atol=1e-5):
        raise ValueError("model returned invalid probabilities")
    return aligned, classes


def _predict_neural(model_name, checkpoint, texts, classes, *, batch_size=8, max_length=384, temperature=1.0,
                    base_checkpoint=None, revision=None):
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Offline neural inference requires the project's [neural] dependencies") from exc
    checkpoint = Path(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if model_name == "ultrabert380":
        network = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    elif model_name in {"qwen06", "qwen17", "qwen4", "llama32_3b"}:
        backbone_path = checkpoint / "merged_backbone"
        if backbone_path.exists():
            network = AutoModel.from_pretrained(backbone_path, local_files_only=True, torch_dtype=model_dtype)
        else:
            if not base_checkpoint:
                raise ValueError("Qwen adapter inference requires the configured base checkpoint")
            network = AutoModel.from_pretrained(base_checkpoint, revision=revision, local_files_only=True,
                                                torch_dtype=model_dtype)
            network = PeftModel.from_pretrained(network, checkpoint, local_files_only=True)
        classifier = torch.nn.Linear(network.config.hidden_size, len(classes))
        state = torch.load(checkpoint / "classifier.pt", map_location="cpu", weights_only=True)
        classifier.load_state_dict(state)
        class Readout(torch.nn.Module):
            def __init__(self): super().__init__(); self.backbone=network; self.classifier=classifier
            def forward(self, input_ids, attention_mask):
                hidden=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
                mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                return self.classifier(pooled.float())
        network = Readout()
    else: raise ValueError(f"unsupported neural model: {model_name}")
    network.to(device).eval(); batches=[]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            enc=tokenizer(list(texts[start:start+batch_size]),truncation=True,max_length=max_length,padding=True,return_tensors="pt")
            if model_name in {"qwen06", "qwen17", "qwen4", "llama32_3b"}: enc={k:v for k,v in enc.items() if k in {"input_ids","attention_mask"}}
            enc={k:v.to(device) for k,v in enc.items()}; logits=network(**enc)
            if hasattr(logits,"logits"): logits=logits.logits
            batches.append(torch.softmax(logits.float()/float(temperature),dim=-1).cpu().numpy())
    matrix=np.concatenate(batches,axis=0) if batches else np.empty((0,len(classes)))
    if matrix.shape != (len(texts),len(classes)) or not np.isfinite(matrix).all(): raise ValueError("neural model returned invalid probabilities")
    return matrix, list(classes)


def export_submission(model, checkpoint, test_path, output_path):
    """Write generic offline id,model predictions for observed test.csv schema."""
    checkpoint, test_path, output_path = Path(checkpoint), Path(test_path), Path(output_path)
    manifest = {}
    if checkpoint.suffix.lower() == ".json":
        manifest = json.loads(checkpoint.read_text())
        manifest_dir = checkpoint.parent
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_absolute(): checkpoint = manifest_dir / checkpoint
    frame = pd.read_csv(test_path, dtype={"id": "string"})
    missing = REQUIRED_TEST - set(frame.columns)
    if missing: raise ValueError(f"test input is missing columns: {sorted(missing)}")
    if frame.id.isna().any() or frame.id.duplicated().any(): raise ValueError("test IDs must be present and unique")
    if frame.text.isna().any() or frame.text.astype(str).str.strip().eq("").any(): raise ValueError("test passages must be nonempty")
    if model == "svc":
        bundle=joblib.load(checkpoint); probabilities, classes=predict_svc_bundle(bundle,frame.text.tolist())
    else:
        classes=manifest.get("classes")
        if not classes: raise ValueError("neural final manifest must include canonical classes")
        probabilities, classes=_predict_neural(model,checkpoint,frame.text.tolist(),classes,
            max_length=manifest.get("config",{}).get("max_length",384),temperature=manifest.get("calibration_temperature",1.0),
            base_checkpoint=manifest.get("config",{}).get("checkpoint"),revision=manifest.get("model_revision"))
    output = pd.DataFrame({"id": frame.id, "model": np.asarray(classes)[probabilities.argmax(1)]})
    if output.id.astype(str).tolist() != frame.id.astype(str).tolist(): raise RuntimeError("output row order does not match input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet": output.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv": output.to_csv(output_path, index=False)
    else: raise ValueError("output must use .parquet or .csv")
    report = {"model": model, "checkpoint": str(checkpoint), "test_path": str(test_path), "output_path": str(output_path),
        "rows": len(output), "classes": list(classes), "schema": ["id", "model"],
        "offline_local_files_only": True, "network_required": False,
        "note": "Generic offline export. Validate against competition rules before submission."}
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def export_svc_submission(checkpoint, test_path, output_path):
    return export_submission("svc", checkpoint, test_path, output_path)


def package_offline_model(manifest_path, output_zip):
    """Package a final model and required project inference code for offline use."""
    manifest_path, output_zip = Path(manifest_path), Path(output_zip)
    source = json.loads(manifest_path.read_text())
    model_name = source["model"]
    checkpoint = Path(source["checkpoint"])
    if not checkpoint.is_absolute(): checkpoint = manifest_path.parent / checkpoint
    if model_name == "svc" and (not checkpoint.is_file() or checkpoint.suffix != ".joblib"):
        raise FileNotFoundError(f"SVC model checkpoint not found: {checkpoint}")
    if model_name != "svc" and not checkpoint.is_dir():
        raise FileNotFoundError(f"Neural model checkpoint directory not found: {checkpoint}")
    source_root = Path(__file__).resolve().parent
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlm-offline-") as temp:
        root = Path(temp)
        (root / "src").mkdir()
        for module in ("__init__.py", "data_factory.py", "feature_factory.py", "inference.py"):
            shutil.copy2(source_root / module, root / "src" / module)
        if model_name == "svc":
            shutil.copy2(checkpoint, root / "model.joblib")
            packaged_checkpoint = "model.joblib"
        else:
            shutil.copytree(checkpoint, root / "model")
            packaged_checkpoint = "model"
        portable = {"experiment_id": source.get("experiment_id"), "model": model_name,
            "checkpoint": packaged_checkpoint, "classes": source["classes"],
            "calibration_temperature": source.get("calibration_temperature", 1.0),
            "config": source.get("config", {}), "dataset_sha256": source.get("dataset_sha256"),
            "model_revision": source.get("model_revision"),
            "submission_schema": ["id", "model"],
            "requirements": ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib"] +
                (["torch", "transformers", "peft", "safetensors"] if model_name != "svc" else []),
            "note": "Use with the repository Kaggle notebook and competition internet disabled; verify competition-specific rules separately."}
        (root / "manifest.json").write_text(json.dumps(portable, indent=2) + "\n")
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            for file in sorted(root.rglob("*")):
                if file.is_file(): archive.write(file, file.relative_to(root))
    return {"package": str(output_zip), "model": model_name, "checkpoint": packaged_checkpoint,
            "experiment_id": source.get("experiment_id"), "bytes": output_zip.stat().st_size,
            "network_required": False, "competition_rules_verified": False}
