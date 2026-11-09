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
from .feature_factory import stylometry


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


def predict_catboost_stylometry(checkpoint, texts, classes, *, temperature=1.0):
    """Predict with the CatBoost stylometry classifier saved by the model factory."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost inference requires catboost; install the project requirements.txt") from exc
    checkpoint = Path(checkpoint)
    model = CatBoostClassifier()
    if checkpoint.is_dir():
        model.load_model(str(checkpoint / "model.cbm"))
        metadata_path = checkpoint / "stylometry.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    else:
        model.load_model(str(checkpoint))
        metadata = {}
    raw = np.asarray(model.predict(stylometry(texts, metadata.get("feature_recipe")), prediction_type="RawFormulaVal"))
    model_classes = list(model.classes_)
    if raw.ndim == 1:
        raw = np.column_stack([-raw, raw])
    aligned = np.zeros((len(texts), len(classes)), dtype=float)
    class_indexes = {label: i for i, label in enumerate(classes)}
    for column, label in enumerate(model_classes):
        aligned[:, class_indexes[label]] = raw[:, column]
    from scipy.special import softmax
    probabilities = softmax(aligned / max(float(temperature), 1e-12), axis=1)
    if not np.isfinite(probabilities).all():
        raise ValueError("CatBoost stylometry model returned invalid probabilities")
    return probabilities, list(classes)


def _predict_neural(model_name, checkpoint, texts, classes, *, batch_size=8, max_length=384, temperature=1.0,
                    base_checkpoint=None, revision=None, fix_mistral_regex=False,
                    text_only_backbone=False, local_base_model=None, return_scores=False,
                    quantization=None, attention_backend=None):
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Offline neural inference requires the project's [neural] dependencies") from exc
    checkpoint = Path(checkpoint)
    checkpoint_metadata_path = checkpoint / "checkpoint_manifest.json"
    checkpoint_metadata = (json.loads(checkpoint_metadata_path.read_text())
                           if checkpoint_metadata_path.is_file() else {})
    quantization = str(quantization or checkpoint_metadata.get("quantization") or "none").lower()
    attention_backend = attention_backend or checkpoint_metadata.get("attention_backend")
    if quantization not in {"none", "qlora4"}:
        raise ValueError(f"unsupported checkpoint quantization: {quantization}")
    if attention_backend not in {None, "sdpa", "flash_attention_2"}:
        raise ValueError(f"unsupported attention backend: {attention_backend}")
    if attention_backend == "flash_attention_2":
        from .model_factory import _require_flash_attention_2_runtime
        _require_flash_attention_2_runtime(torch)
    if model_name in {"embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"} and (checkpoint / "readout.joblib").is_file():
        from .model_factory import extract_frozen_backbone_features
        bundle = joblib.load(checkpoint / "readout.joblib")
        local_backbone = Path(local_base_model) if local_base_model else checkpoint / "base_model"
        features, _ = extract_frozen_backbone_features(
            model_name, base_checkpoint, revision, texts, max_length=max_length,
            batch_size=batch_size, model_dir=local_backbone if local_backbone.is_dir() else None,
            fix_mistral_regex=bundle.get("fix_mistral_regex", False))
        class_order = list(bundle["classes"])
        probabilities = bundle["model"].predict_proba(features)
        aligned = np.zeros((len(texts), len(classes)), dtype=float)
        class_indexes = {label: i for i, label in enumerate(classes)}
        for column, label in enumerate(class_order):
            aligned[:, class_indexes[label]] = probabilities[:, column]
        if not np.isfinite(aligned).all() or not np.allclose(aligned.sum(1), 1, atol=1e-5):
            raise ValueError(f"{model_name} returned invalid probabilities")
        return aligned, list(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # EmbeddingGemma's activations can overflow fp16. Use bf16 where available
    # and fp32 on GPUs such as T4 that do not support bf16.
    if device.type == "cuda":
        model_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else
                       torch.float32 if model_name == "embeddinggemma2" else torch.float16)
    else:
        model_dtype = torch.float32
    trust_remote_code = model_name == "phi35_mini"
    tokenizer_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if model_name == "ultrabert380":
        network = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    elif model_name in {"qwen06", "qwen06_best_epoch03", "qwen17", "qwen4", "qwen8", "llama32_3b", "smollm3_3b", "granite4_h_micro", "granite4_350m", "embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b", "phi35_mini"}:
        backbone_path = checkpoint / "merged_backbone"
        if backbone_path.exists() and quantization != "qlora4":
            network = AutoModel.from_pretrained(backbone_path, local_files_only=True, torch_dtype=model_dtype,
                                                trust_remote_code=trust_remote_code)
        else:
            if not base_checkpoint:
                raise ValueError("Qwen adapter inference requires the configured base checkpoint")
            if local_base_model:
                base_checkpoint = str(local_base_model)
                revision = None
            load_kwargs = {"attn_implementation": attention_backend} if attention_backend else {}
            if quantization == "qlora4":
                if device.type != "cuda":
                    raise RuntimeError("QLoRA checkpoint inference requires a CUDA GPU")
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("QLoRA checkpoint inference requires BF16-capable CUDA hardware")
                from transformers import BitsAndBytesConfig
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
                if quantization == "qlora4":
                    raise ValueError("qlora4 inference does not support text-only composite loading")
                from .model_factory import load_text_only_backbone
                import transformers
                network = load_text_only_backbone(transformers, base_checkpoint, revision,
                                                  torch_dtype=model_dtype, local_files_only=True)
            else:
                network = AutoModel.from_pretrained(base_checkpoint, revision=revision, local_files_only=True,
                                                    torch_dtype=model_dtype, trust_remote_code=trust_remote_code,
                                                    **load_kwargs)
            network = PeftModel.from_pretrained(network, checkpoint, local_files_only=True)
        state = torch.load(checkpoint / "classifier.pt", map_location="cpu", weights_only=True)
        classifier = torch.nn.Linear(int(state["weight"].shape[1]), len(classes))
        classifier.load_state_dict(state)
        if quantization == "qlora4":
            classifier.to(device)
        class Readout(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.backbone=network; self.classifier=classifier
                if quantization == "qlora4":
                    self.hf_device_map = getattr(network, "hf_device_map", {"": device.index or 0})
            def forward(self, input_ids, attention_mask):
                hidden=self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
                mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                return self.classifier(pooled.float())
        network = Readout()
    else: raise ValueError(f"unsupported neural model: {model_name}")
    if quantization == "qlora4":
        network.eval()
    else:
        network.to(device).eval()
    batches=[]; score_batches=[]
    batch_size = max(1, int(batch_size))
    next_batch_size = batch_size
    with torch.inference_mode():
        start = 0
        while start < len(texts):
            current_size = min(next_batch_size, len(texts) - start)
            text_batch = list(texts[start:start + current_size])
            if model_name == "embeddinggemma2":
                text_batch = ["task: classification | query: " + str(text) for text in text_batch]
            try:
                enc=tokenizer(text_batch,truncation=True,max_length=max_length,padding=True,return_tensors="pt")
                if model_name in {"qwen06", "qwen06_best_epoch03", "qwen17", "qwen4", "qwen8", "llama32_3b", "smollm3_3b", "granite4_h_micro", "granite4_350m", "embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b", "phi35_mini"}: enc={k:v for k,v in enc.items() if k in {"input_ids","attention_mask"}}
                enc={k:v.to(device) for k,v in enc.items()}; logits=network(**enc)
                if hasattr(logits,"logits"): logits=logits.logits
                raw_logits = logits.float()
                score_batches.append(raw_logits.cpu().numpy())
                start += current_size
            except RuntimeError as exc:
                is_cuda_oom = device.type == "cuda" and (
                    isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())) or
                    "out of memory" in str(exc).lower())
                if not is_cuda_oom or current_size <= 1:
                    raise
                if "enc" in locals():
                    del enc
                if "logits" in locals():
                    del logits
                torch.cuda.empty_cache()
                next_batch_size = max(1, current_size // 2)
                continue
            batches.append(torch.softmax(raw_logits/float(temperature),dim=-1).cpu().numpy())
    matrix=np.concatenate(batches,axis=0) if batches else np.empty((0,len(classes)))
    if matrix.shape != (len(texts),len(classes)) or not np.isfinite(matrix).all(): raise ValueError("neural model returned invalid probabilities")
    if return_scores:
        raw_scores=np.concatenate(score_batches,axis=0) if score_batches else np.empty((0,len(classes)))
        return matrix, list(classes), raw_scores
    return matrix, list(classes)


def export_submission(model, checkpoint, test_path, output_path):
    """Write generic offline id,model predictions for observed test.csv schema."""
    checkpoint, test_path, output_path = Path(checkpoint), Path(test_path), Path(output_path)
    manifest = {}
    if checkpoint.suffix.lower() == ".json":
        manifest = json.loads(checkpoint.read_text())
        manifest_dir = checkpoint.parent
        checkpoint_key = "checkpoint" if manifest.get("checkpoint") else "checkpoint_path"
        checkpoint_value = manifest.get(checkpoint_key)
        if not checkpoint_value:
            raise ValueError("checkpoint manifest needs a checkpoint or checkpoint_path")
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            checkpoint = (manifest_dir / checkpoint if checkpoint_key == "checkpoint"
                          else project_root / checkpoint)
    frame = pd.read_csv(test_path, dtype={"id": "string"})
    missing = REQUIRED_TEST - set(frame.columns)
    if missing: raise ValueError(f"test input is missing columns: {sorted(missing)}")
    if frame.id.isna().any() or frame.id.duplicated().any(): raise ValueError("test IDs must be present and unique")
    if frame.text.isna().any() or frame.text.astype(str).str.strip().eq("").any(): raise ValueError("test passages must be nonempty")
    if model == "svc":
        bundle=joblib.load(checkpoint); probabilities, classes=predict_svc_bundle(bundle,frame.text.tolist())
    elif model == "catboost_stylometry":
        classes=manifest.get("classes")
        if not classes: raise ValueError("CatBoost final manifest must include canonical classes")
        probabilities, classes=predict_catboost_stylometry(checkpoint,frame.text.tolist(),classes,
            temperature=manifest.get("calibration_temperature",1.0))
    else:
        classes=manifest.get("classes")
        if not classes: raise ValueError("neural final manifest must include canonical classes")
        probabilities, classes=_predict_neural(model,checkpoint,frame.text.tolist(),classes,
            max_length=manifest.get("config",{}).get("max_length",384),temperature=manifest.get("calibration_temperature",1.0),
            base_checkpoint=manifest.get("config",{}).get("checkpoint"),revision=manifest.get("model_revision"),
            fix_mistral_regex=manifest.get("config",{}).get("fix_mistral_regex",False),
            text_only_backbone=manifest.get("config",{}).get("text_only_backbone",False),
            quantization=manifest.get("config",{}).get("quantization",manifest.get("quantization")),
            attention_backend=manifest.get("config",{}).get("attention_backend",manifest.get("attention_backend")))
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


def predict_from_trust_table(probability_by_candidate, classes, trust_rows, *, minimum_support=10,
                             fallback_candidate=None):
    """Select a model's class proposal using validation precision by model and proposed class."""
    classes = list(classes)
    if not probability_by_candidate:
        raise ValueError("at least one candidate probability matrix is required")
    rows = list(trust_rows)
    lookup = {(str(row["candidate_id"]), str(row["predicted_nlm"])): row for row in rows}
    matrices = {name: np.asarray(probabilities, dtype=float)
                for name, probabilities in probability_by_candidate.items()}
    first = next(iter(matrices.values()))
    if first.ndim != 2 or first.shape[1] != len(classes):
        raise ValueError("candidate probabilities must have shape [rows, classes]")
    if any(value.shape != first.shape for value in matrices.values()):
        raise ValueError("candidate probability matrices must have matching shapes")
    output = []
    for row_index in range(first.shape[0]):
        options = []
        for candidate_id, probabilities in matrices.items():
            vector = probabilities[row_index]
            class_index = int(np.argmax(vector))
            label = classes[class_index]
            evidence = lookup.get((str(candidate_id), label))
            if evidence is None or int(evidence.get("predicted_count", 0)) < minimum_support:
                continue
            ordered = np.partition(vector, -2)
            margin = float(ordered[-1] - ordered[-2])
            options.append((float(evidence["smoothed_trust"]), float(vector[class_index]),
                            margin, str(candidate_id), label))
        if options:
            options.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
            output.append(options[0][4])
        else:
            if fallback_candidate not in matrices:
                raise ValueError("no supported trust-table predictions and no valid fallback candidate")
            output.append(classes[int(np.argmax(matrices[fallback_candidate][row_index]))])
    return np.asarray(output)


def predict_candidate(model_spec, texts, classes, bundle_root):
    """Run one candidate stored in a prediction bundle and align its class probabilities."""
    bundle_root = Path(bundle_root)
    model = str(model_spec["model"])
    checkpoint = bundle_root / model_spec["checkpoint"]
    config = model_spec.get("config", {})
    classes = list(classes)
    if model == "svc":
        probabilities, returned_classes = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probabilities, returned_classes = predict_catboost_stylometry(checkpoint, texts, classes,
            temperature=model_spec.get("calibration_temperature", 1.0))
    else:
        probabilities, returned_classes = _predict_neural(model, checkpoint, texts, classes,
            max_length=config.get("max_length", 384),
            temperature=model_spec.get("calibration_temperature", 1.0),
            base_checkpoint=config.get("checkpoint"), revision=model_spec.get("model_revision"),
            fix_mistral_regex=config.get("fix_mistral_regex", False),
            text_only_backbone=config.get("text_only_backbone", False),
            quantization=config.get("quantization", model_spec.get("quantization")),
            attention_backend=config.get("attention_backend", model_spec.get("attention_backend")),
            local_base_model=(bundle_root / model_spec["base_model"] if model_spec.get("base_model") else None))
    if returned_classes != classes:
        indexes = {label: index for index, label in enumerate(returned_classes)}
        probabilities = probabilities[:, [indexes[label] for label in classes]]
    if probabilities.shape != (len(texts), len(classes)):
        raise ValueError(f"{model_spec['candidate_id']} returned an unexpected probability shape")
    return probabilities


def predict_prediction_bundle(bundle_dir, texts, *, base_model_overrides=None):
    """Apply the selected deterministic blend or reliability rule from a portable bundle."""
    bundle_dir = Path(bundle_dir)
    selection = json.loads((bundle_dir / "prediction_bundle.json").read_text())
    classes = list(selection["classes"])
    base_model_overrides = base_model_overrides or {}
    specs = {item["candidate_id"]: item for item in selection["candidate_models"]}
    probabilities = {}
    for candidate_id in selection["required_candidates"]:
        spec = dict(specs[candidate_id])
        if spec["model"] in base_model_overrides:
            spec["base_model"] = str(base_model_overrides[spec["model"]])
        probabilities[candidate_id] = predict_candidate(spec, texts, classes, bundle_dir)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if selection["selected_strategy"] == "equal_probability_blend":
        members = selection["selected_members"]
        combined = np.mean(np.stack([probabilities[name] for name in members]), axis=0)
        predictions = np.asarray(classes)[combined.argmax(axis=1)]
    elif selection["selected_strategy"] == "optimized_probability_blend":
        members = selection["selected_members"]
        weight_map = selection.get("selected_weights", {})
        weights = np.asarray([weight_map.get(name) for name in members], dtype=float)
        if (weights.shape != (len(members),) or not np.isfinite(weights).all() or
                np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6)):
            raise ValueError("weighted prediction bundle has invalid or unnormalized member weights")
        combined = np.einsum("m,mnc->nc", weights,
                             np.stack([probabilities[name] for name in members]))
        predictions = np.asarray(classes)[combined.argmax(axis=1)]
    elif selection["selected_strategy"] == "per_response_reliability":
        predictions = predict_from_trust_table(probabilities, classes, selection["trust_table"],
            minimum_support=selection["minimum_support"],
            fallback_candidate=selection["fallback_candidate"])
    else:
        raise ValueError(f"unsupported prediction bundle strategy: {selection['selected_strategy']}")
    return predictions, classes


def package_prediction_bundle(selection_manifest, output_zip, *, include_base_models=True):
    """Package the selected prediction rule and only the checkpoints it needs."""
    import hashlib
    import os
    from huggingface_hub import snapshot_download

    selection_manifest = Path(selection_manifest)
    report = json.loads(selection_manifest.read_text())
    if report.get("module") != "prediction_factory":
        raise ValueError("selection manifest was not created by prediction_factory")
    candidate_by_id = {item["candidate_id"]: item for item in report["candidates"]}
    strategy = report["selected_strategy"]
    if strategy == "equal_probability_blend":
        required = list(report["selected_members"])
    elif strategy == "optimized_probability_blend":
        required = list(report["selected_members"])
    elif strategy == "per_response_reliability":
        required = sorted({row["candidate_id"] for row in pd.read_csv(report["trust_table_path"]).to_dict("records")
                           if row["eligible"]})
    else:
        raise ValueError(f"unsupported selected strategy: {strategy}")
    fallback_candidate = None
    trust_rows = []
    if strategy == "per_response_reliability":
        fallback_candidate = max(report["candidates"],
            key=lambda item: (item.get("training_balanced_accuracy") or 0.0,
                              item["candidate_id"]))["candidate_id"]
        if fallback_candidate not in required:
            required.append(fallback_candidate)
        for row in pd.read_csv(report["trust_table_path"]).to_dict("records"):
            trust_rows.append({key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
                               for key, value in row.items()})
    root_module = Path(__file__).resolve().parent
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    def copy_inference_tree(source, destination):
        source, destination = Path(source), Path(destination)
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            # Training runs keep epoch snapshots beside the selected root
            # checkpoint. Inference loads only the root files; avoid shipping
            # every duplicate adapter and tokenizer in the Kaggle bundle.
            if any(part.startswith("epoch-") and part[6:].isdigit() for part in relative.parts):
                continue
            if any(part in {"onnx", ".git", "__pycache__"} for part in relative.parts):
                continue
            if path.name in {"README.md", "training_log.json", "training_progress.json",
                             "batch_size_probe.json", "checkpoint_selection.json",
                             "notebook.ipynb", "consolidated.safetensors"}:
                continue
            if not path.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    external_sources = {
        "HuggingFaceTB/SmolLM3-3B-Base": {
            "kind": "dataset", "ref": "johnnyhyland/smollm3-3b-base-36l-text"},
        "Qwen/Qwen3-0.6B": {
            "kind": "dataset", "ref": "pingjiale/qwen3-0-6b", "subdirectory": "Qwen3-0.6B"},
        "mistralai/Ministral-3-3B-Base-2512": {
            "kind": "model", "ref": "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512"},
    }
    with tempfile.TemporaryDirectory(prefix="nlm-prediction-bundle-", dir=output_zip.parent) as temp:
        root = Path(temp)
        (root / "src").mkdir()
        for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
            shutil.copy2(root_module / module, root / "src" / module)
        candidates = []
        base_model_dirs = {}
        for candidate_id in required:
            item = candidate_by_id[candidate_id]
            source_manifest = json.loads(Path(item["source_manifest"]).read_text())
            checkpoint = Path(item["checkpoint_path"]) if item.get("checkpoint_path") else None
            if checkpoint is None or not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint unavailable for prediction candidate {candidate_id}: {checkpoint}")
            safe_name = hashlib.sha256(candidate_id.encode()).hexdigest()[:16]
            candidate_root = root / "models" / safe_name
            candidate_root.mkdir(parents=True)
            model = item["model"]
            if model == "svc":
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"SVC checkpoint must be a file: {checkpoint}")
                shutil.copy2(checkpoint, candidate_root / "model.joblib")
                packed_checkpoint = str(Path("models") / safe_name / "model.joblib")
            else:
                if checkpoint.is_file():
                    shutil.copy2(checkpoint, candidate_root / checkpoint.name)
                    packed_checkpoint = str(Path("models") / safe_name / checkpoint.name)
                else:
                    copy_inference_tree(checkpoint, candidate_root / "model")
                    packed_checkpoint = str(Path("models") / safe_name / "model")
            spec = {"candidate_id": candidate_id, "model": model,
                "checkpoint": packed_checkpoint, "config": item.get("config", {}),
                "model_revision": item.get("model_revision"),
                "calibration_temperature": item.get("calibration_temperature", 1.0)}
            if model not in {"svc", "catboost_stylometry", "ultrabert380"}:
                model_dir = root / packed_checkpoint
                if model_dir.is_file():
                    model_dir = model_dir.parent
                local_base = model_dir / "base_model"
                if (model_dir / "merged_backbone").exists():
                    local_base = None
                elif not local_base.exists():
                    config = source_manifest.get("config", {})
                    repo_id = config.get("checkpoint")
                    revision = source_manifest.get("model_revision")
                    if not repo_id or not revision:
                        raise ValueError(f"{candidate_id} needs a pinned base-model checkpoint and revision")
                    base_key = (repo_id, revision)
                    if not include_base_models:
                        if repo_id not in external_sources:
                            raise ValueError(f"no Kaggle base-model input is configured for {repo_id}")
                        spec["base_model_source"] = external_sources[repo_id]
                    elif base_key not in base_model_dirs:
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        downloaded = Path(snapshot_download(repo_id=repo_id, revision=revision,
                                                            local_files_only=True))
                        shared_base = root / "base_models" / hashlib.sha256(
                            f"{repo_id}@{revision}".encode()).hexdigest()[:16]
                        shared_base.parent.mkdir(parents=True, exist_ok=True)
                        copy_inference_tree(downloaded, shared_base)
                        base_model_dirs[base_key] = shared_base
                    if include_base_models:
                        spec["base_model"] = str(base_model_dirs[base_key].relative_to(root))
                elif local_base.exists():
                    spec["base_model"] = str(local_base.relative_to(root))
            candidates.append(spec)
        portable = {"experiment_id": report["experiment_id"], "classes": report["classes"],
            "selected_strategy": strategy, "selected_members": report["selected_members"],
            "selected_weights": report.get("selected_weights", {}),
            "required_candidates": required, "candidate_models": candidates,
            "trust_table": trust_rows, "minimum_support": (report["reliability_rule"]["minimum_support"]
                                                       if strategy == "per_response_reliability" else None),
            "fallback_candidate": fallback_candidate,
            "validation_accuracy": report["selected_validation_accuracy"],
            "validation_balanced_accuracy": report["selected_validation_balanced_accuracy"],
            "validation_ids_sha256": report["validation_ids_sha256"],
            "external_base_models": external_sources if not include_base_models else {},
            "submission_schema": ["id", "model"]}
        (root / "prediction_bundle.json").write_text(json.dumps(portable, indent=2) + "\n")
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(root))
    return {"package": str(output_zip), "bytes": output_zip.stat().st_size,
            "strategy": strategy, "candidate_count": len(required), "experiment_id": report["experiment_id"]}


def export_prediction_bundle(bundle_path, test_path, output_path):
    """Run an extracted or zipped prediction bundle and export competition-format predictions."""
    bundle_path, test_path, output_path = Path(bundle_path), Path(test_path), Path(output_path)
    frame = pd.read_csv(test_path, dtype={"id": "string"})
    missing = REQUIRED_TEST - set(frame.columns)
    if missing:
        raise ValueError(f"test input is missing columns: {sorted(missing)}")
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("test IDs must be present and unique")
    if frame.text.isna().any() or frame.text.astype(str).str.strip().eq("").any():
        raise ValueError("test passages must be nonempty")
    if bundle_path.is_file():
        with tempfile.TemporaryDirectory(prefix="nlm-prediction-export-") as temp:
            with zipfile.ZipFile(bundle_path) as archive:
                archive.extractall(temp)
            prediction, classes = predict_prediction_bundle(temp, frame.text.astype(str).tolist())
            return _write_prediction_bundle_output(frame, prediction, classes, test_path, output_path)
    prediction, classes = predict_prediction_bundle(bundle_path, frame.text.astype(str).tolist())
    return _write_prediction_bundle_output(frame, prediction, classes, test_path, output_path)


def _write_prediction_bundle_output(frame, prediction, classes, test_path, output_path):
    output_path = Path(output_path)
    output = pd.DataFrame({"id": frame.id, "model": prediction})
    if output.id.astype(str).tolist() != frame.id.astype(str).tolist():
        raise RuntimeError("output row order does not match test input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        output.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        output.to_csv(output_path, index=False)
    else:
        raise ValueError("output must use .parquet or .csv")
    report = {"strategy": "prediction_bundle", "test_path": str(test_path),
        "output_path": str(output_path), "rows": len(output), "classes": list(classes),
        "schema": ["id", "model"], "offline_local_files_only": True}
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_prediction_submission(probability_by_candidate, classes, trust_rows, test_path, output_path,
                                 *, minimum_support=10, fallback_candidate=None,
                                 report_metadata=None):
    """Export predictions already computed by a packaged candidate set using a trust table."""
    test_path, output_path = Path(test_path), Path(output_path)
    frame = pd.read_csv(test_path, dtype={"id": "string"})
    missing = REQUIRED_TEST - set(frame.columns)
    if missing:
        raise ValueError(f"test input is missing columns: {sorted(missing)}")
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("test IDs must be present and unique")
    if frame.text.isna().any() or frame.text.astype(str).str.strip().eq("").any():
        raise ValueError("test passages must be nonempty")
    prediction = predict_from_trust_table(probability_by_candidate, classes, trust_rows,
        minimum_support=minimum_support, fallback_candidate=fallback_candidate)
    output = pd.DataFrame({"id": frame.id, "model": prediction})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        output.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        output.to_csv(output_path, index=False)
    else:
        raise ValueError("output must use .parquet or .csv")
    report = {"strategy": "per_response_reliability", "test_path": str(test_path),
        "output_path": str(output_path), "rows": len(output), "classes": list(classes),
        "schema": ["id", "model"], "minimum_support": minimum_support,
        "fallback_candidate": fallback_candidate, **(report_metadata or {})}
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def package_offline_model(manifest_path, output_zip):
    """Package a saved training or continuation checkpoint for offline inference."""
    manifest_path, output_zip = Path(manifest_path), Path(output_zip)
    source = json.loads(manifest_path.read_text())
    model_name = source["model"]
    source_root = Path(__file__).resolve().parent
    checkpoint_value = source.get("checkpoint")
    checkpoint_is_experiment_path = checkpoint_value is None
    if checkpoint_is_experiment_path:
        checkpoint_value = source.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError("model manifest needs a checkpoint or checkpoint_path")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = (source_root.parent / checkpoint if checkpoint_is_experiment_path
                      else manifest_path.resolve().parent / checkpoint)
    if model_name == "svc" and (not checkpoint.is_file() or checkpoint.suffix != ".joblib"):
        raise FileNotFoundError(f"SVC model checkpoint not found: {checkpoint}")
    if model_name != "svc" and not checkpoint.is_dir():
        raise FileNotFoundError(f"Neural model checkpoint directory not found: {checkpoint}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nlm-offline-") as temp:
        root = Path(temp)
        (root / "src").mkdir()
        for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
            shutil.copy2(source_root / module, root / "src" / module)
        if model_name == "svc":
            shutil.copy2(checkpoint, root / "model.joblib")
            packaged_checkpoint = "model.joblib"
        else:
            shutil.copytree(checkpoint, root / "model")
            if model_name in {"embeddinggemma2", "gemma4_e2b", "gemma4_e4b", "ministral3_3b"}:
                config = source.get("config", {})
                repo_id, revision = config.get("checkpoint"), source.get("model_revision")
                if not repo_id or not revision:
                    raise ValueError(f"{model_name} package needs a pinned base checkpoint and revision")
                import os
                os.environ["HF_HUB_OFFLINE"] = "1"
                from huggingface_hub import snapshot_download
                local_base = snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
                shutil.copytree(local_base, root / "model" / "base_model", symlinks=False)
            packaged_checkpoint = "model"
        portable = {"experiment_id": source.get("experiment_id"), "model": model_name,
            "checkpoint": packaged_checkpoint, "classes": source["classes"],
            "selected_epoch": source.get("selected_epoch"),
            "selected_iteration": source.get("selected_iteration"),
            "selected_validation_accuracy": source.get("selected_validation_accuracy"),
            "continued_from_experiment": source.get("continued_from_experiment"),
            "training_dataset": source.get("training_dataset"),
            "calibration_temperature": source.get("calibration_temperature", 1.0),
            "config": source.get("config", {}), "dataset_sha256": source.get("dataset_sha256"),
            "model_revision": source.get("model_revision"),
            "submission_schema": ["id", "model"],
            "requirements": ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib"] +
                (["catboost"] if model_name == "catboost_stylometry" else
                 ["torch", "transformers", "peft", "safetensors"] if model_name != "svc" else []),
            "note": "Use with the repository Kaggle notebook and competition internet disabled; verify competition-specific rules separately."}
        (root / "manifest.json").write_text(json.dumps(portable, indent=2) + "\n")
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            for file in sorted(root.rglob("*")):
                if file.is_file(): archive.write(file, file.relative_to(root))
    return {"package": str(output_zip), "model": model_name, "checkpoint": packaged_checkpoint,
            "experiment_id": source.get("experiment_id"), "bytes": output_zip.stat().st_size,
            "network_required": False, "competition_rules_verified": False}
