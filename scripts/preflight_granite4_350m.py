"""Check Granite 350M's six LoRA targets and one 512-token backward pass."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["revision"] != "a50b46cef21c8a86b15f0496cb794487a78a910b":
        raise ValueError("Granite 350M checkpoint revision is not the requested pinned revision")
    if int(config["max_length"]) != 512:
        raise ValueError("preflight must use exactly 512 tokens")
    targets = list(config["lora_target_modules"])
    if len(targets) != 6 or len(set(targets)) != 6:
        raise ValueError(f"expected six distinct configured LoRA target types, got {targets}")
    if not torch.cuda.is_available():
        raise RuntimeError("Granite preflight requires CUDA")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], revision=config["revision"],
        local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(config["checkpoint"], revision=config["revision"],
        torch_dtype=dtype, local_files_only=True, trust_remote_code=False).cuda()
    model.config.use_cache = False
    names = [name for name, _ in model.named_modules()]
    matched = {target: [name for name in names if name == target or name.endswith("." + target)]
               for target in targets}
    missing = [target for target, found in matched.items() if not found]
    if missing:
        raise ValueError(f"configured LoRA target types are absent from the loaded model: {missing}")
    backbone = get_peft_model(model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=int(config["lora_rank"]), lora_alpha=int(config["lora_alpha"]), lora_dropout=0.05,
        target_modules=targets))
    backbone.config.use_cache = False
    trainable_by_target = {target: 0 for target in targets}
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad:
            for target in targets:
                if re.search(rf"(?:^|\.){re.escape(target)}\.lora_[AB]\.default\.weight$", name):
                    trainable_by_target[target] += parameter.numel()
                    break
    if any(count == 0 for count in trainable_by_target.values()):
        raise RuntimeError(f"PEFT adapter missing for target types: {trainable_by_target}")
    hidden_size = int(config["readout_hidden_size"])
    if hidden_size != int(backbone.config.hidden_size):
        raise ValueError(f"mean-pooled head size {hidden_size} differs from backbone hidden size {backbone.config.hidden_size}")
    classifier = torch.nn.Linear(hidden_size, 18, dtype=torch.float32, device="cuda")
    sample = "Summarize the important details in this response. " * 100
    encoded = tokenizer(sample, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    if tuple(encoded["input_ids"].shape) != (1, 512):
        raise RuntimeError(f"expected input shape (1, 512), got {tuple(encoded['input_ids'].shape)}")
    backbone.train()
    classifier.train()
    with torch.autocast(device_type="cuda", dtype=dtype):
        hidden = backbone(input_ids=encoded["input_ids"].cuda(),
                          attention_mask=encoded["attention_mask"].cuda()).last_hidden_state
    mask = encoded["attention_mask"].cuda().unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    logits = classifier(pooled.float())
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0], device="cuda"))
    loss.backward()
    classifier_grad = classifier.weight.grad
    if classifier_grad is None or not torch.isfinite(classifier_grad).all() or int(torch.count_nonzero(classifier_grad)) == 0:
        raise RuntimeError("18-class classifier did not receive finite, nonzero gradients")
    grads = {target: {"finite": True, "nonzero_elements": 0} for target in targets}
    for name, parameter in backbone.named_parameters():
        if parameter.grad is None or not parameter.requires_grad:
            continue
        for target in targets:
            if re.search(rf"(?:^|\.){re.escape(target)}\.lora_[AB]\.default\.weight$", name):
                grads[target]["finite"] &= bool(torch.isfinite(parameter.grad).all().item())
                grads[target]["nonzero_elements"] += int(torch.count_nonzero(parameter.grad).item())
                break
    if any(not info["finite"] or info["nonzero_elements"] == 0 for info in grads.values()):
        raise RuntimeError(f"LoRA gradients missing or invalid: {grads}")
    report = {
        "model": config["model"], "checkpoint": config["checkpoint"], "revision": config["revision"],
        "precision": "bfloat16" if dtype == torch.bfloat16 else "float16",
        "max_length": 512, "input_shape": list(encoded["input_ids"].shape), "loss": float(loss.detach().cpu()),
        "matched_module_counts": {key: len(value) for key, value in matched.items()},
        "trainable_lora_parameters_by_target": trainable_by_target,
        "total_trainable_lora_parameters": int(sum(trainable_by_target.values())),
        "classifier_parameters": int(sum(p.numel() for p in classifier.parameters())),
        "classifier_gradient_finite": bool(torch.isfinite(classifier_grad).all().item()),
        "classifier_gradient_nonzero_elements": int(torch.count_nonzero(classifier_grad).item()),
        "lora_gradient_checks": grads,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    del classifier, backbone, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
