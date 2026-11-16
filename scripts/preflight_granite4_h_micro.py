"""Verify Granite H-Micro LoRA module coverage and one 512-token backward pass."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
import pyarrow.parquet as pq
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["revision"] == "372ede0bc484284e08ce574e773c96cd2b89b367"
    assert int(config["max_length"]) == 512
    targets = list(config["lora_target_modules"])
    if len(targets) != 8 or len(set(targets)) != 8:
        raise ValueError(f"expected eight distinct LoRA target types, got {targets}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Granite preflight requires a BF16-capable CUDA GPU")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], revision=config["revision"],
        local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(config["checkpoint"], revision=config["revision"],
        torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False).cuda()
    model.config.use_cache = False
    available_names = [name for name, _ in model.named_modules()]
    matched = {target: [name for name in available_names
                        if name == target or name.endswith("." + target)] for target in targets}
    missing = [target for target, names in matched.items() if not names]
    if missing:
        raise ValueError(f"configured LoRA target types not found: {missing}")
    backbone = get_peft_model(model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=int(config["lora_rank"]), lora_alpha=int(config["lora_alpha"]), lora_dropout=0.05,
        target_modules=targets))
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    trainable_by_type = {target: 0 for target in targets}
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad:
            for target in targets:
                if re.search(rf"(?:^|\.){re.escape(target)}\.lora_[AB]\.default\.weight$", name):
                    trainable_by_type[target] += parameter.numel()
                    break
    empty = [target for target, count in trainable_by_type.items() if count == 0]
    if empty:
        raise ValueError(f"PEFT did not create trainable adapters for targets: {empty}")
    hidden_size = int(backbone.config.hidden_size)
    classifier = torch.nn.Linear(hidden_size, 18, dtype=torch.float32, device="cuda")
    head = torch.nn.Module()
    head.backbone = backbone
    head.classifier = classifier
    head.train()
    # Use real, varied training text for the gradient check. Repeating one
    # synthetic sentence can leave a specialized hybrid-block projection
    # inactive for this single batch.
    dataset_path = ROOT / "data/processed/dataset_v2.14.parquet"
    parquet = pq.ParquetFile(dataset_path)
    sample_texts = []
    for batch in parquet.iter_batches(batch_size=32, columns=["text"]):
        sample_texts = [str(value) for value in batch.column(0).to_pylist()
                        if value is not None and str(value).strip()]
        if sample_texts:
            break
    if not sample_texts:
        raise RuntimeError(f"no non-empty text available in {dataset_path}")
    sample = "\n".join(sample_texts)
    encoded = tokenizer(sample, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    if encoded["input_ids"].shape != (1, 512):
        raise RuntimeError(f"expected one 512-token input, got {tuple(encoded['input_ids'].shape)}")
    ids = encoded["input_ids"].cuda()
    mask = encoded["attention_mask"].cuda()
    head.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = head.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
    maskf = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * maskf).sum(1) / maskf.sum(1).clamp_min(1)
    logits = head.classifier(pooled.float())
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0], device="cuda"))
    loss.backward()
    target_gradients = {target: {"finite": True, "nonzero_elements": 0} for target in targets}
    for name, parameter in backbone.named_parameters():
        if parameter.grad is None or not parameter.requires_grad:
            continue
        for target in targets:
            if re.search(rf"(?:^|\.){re.escape(target)}\.lora_[AB]\.default\.weight$", name):
                grad = parameter.grad
                target_gradients[target]["finite"] &= bool(torch.isfinite(grad).all().item())
                target_gradients[target]["nonzero_elements"] += int(torch.count_nonzero(grad).item())
                break
    classifier_grad = classifier.weight.grad
    if classifier_grad is None or not torch.isfinite(classifier_grad).all() or not torch.count_nonzero(classifier_grad):
        raise RuntimeError("classifier head did not receive finite, nonzero gradients")
    bad = [target for target, report in target_gradients.items()
           if not report["finite"]]
    total_lora_gradient_elements = sum(report["nonzero_elements"]
                                       for report in target_gradients.values())
    if bad or total_lora_gradient_elements == 0:
        raise RuntimeError(f"LoRA gradients invalid: non-finite targets={bad}; "
                           f"total nonzero elements={total_lora_gradient_elements}")
    report = {
        "model": config["model"], "checkpoint": config["checkpoint"], "revision": config["revision"],
        "precision": "bfloat16", "gradient_checkpointing": True, "max_length": 512,
        "input_shape": list(encoded["input_ids"].shape), "loss": float(loss.detach().cpu()),
        "matched_module_counts": {target: len(names) for target, names in matched.items()},
        "trainable_lora_parameters_by_target": trainable_by_type,
        "total_trainable_lora_parameters": int(sum(trainable_by_type.values())),
        "classifier_parameters": int(sum(p.numel() for p in classifier.parameters())),
        "classifier_gradient_finite": bool(torch.isfinite(classifier_grad).all().item()),
        "classifier_gradient_nonzero_elements": int(torch.count_nonzero(classifier_grad).item()),
        "lora_gradient_checks": target_gradients,
        "lora_targets_with_zero_sample_gradient": [
            target for target, values in target_gradients.items()
            if values["nonzero_elements"] == 0],
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    del head, classifier, backbone, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
