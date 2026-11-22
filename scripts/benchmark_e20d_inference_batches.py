"""Measure safe inference batches on longest V0 responses for E20-D models.

This benchmark loads one representative checkpoint per architecture, starts at
batch 32, and uses the production inference OOM backoff. Results are intended
for the E20-D Kaggle inference configuration; no training or Kaggle calls occur.
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from scripts.publish_kaggle_checkpoint_store import ROOT, build_checkpoint, resolve_checkpoint
from src.inference import predict_candidate
from src.v2_checkpoint_trust import LABELS


MANIFEST = ROOT / "artifacts/kaggle/e20d-checkpoints-v001/e20d_manifest.json"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
OUTPUT = ROOT / "artifacts/analysis/e20d_inference_batch_benchmark.json"
START_BATCH = 64
SAMPLE_ROWS = 64
QWEN_GENERATION_BATCH = 64
EMBEDDING_BATCH = 64
NEURAL_EXCLUDED = {"svc", "catboost_stylometry"}


def _prepared_texts(tokenizer, texts, model, config):
    prepared = list(map(str, texts))
    if model == "embeddinggemma2":
        prepared = ["task: classification | query: " + text for text in prepared]
    if config.get("use_chat_template", False):
        user_prompt = config.get("chat_template_user_prompt",
                                 "Identify the source model for the assistant response.")
        prepared = [tokenizer.apply_chat_template(
            [{"role": "user", "content": str(user_prompt)},
             {"role": "assistant", "content": text}],
            tokenize=False, add_generation_prompt=False) for text in prepared]
    return prepared


def _representatives(manifest):
    representatives = OrderedDict()
    for sid in manifest["checkpoint_ids"]:
        item = manifest["checkpoint_members"][sid]
        model = item["model"]
        if model in NEURAL_EXCLUDED:
            continue
        config = item.get("config", {})
        previous = representatives.get(model)
        score = int(config.get("max_length") or 384)
        previous_score = int(previous[1].get("config", {}).get("max_length") or 384) if previous else -1
        if previous is None or score > previous_score:
            representatives[model] = (sid, item)
    return representatives


def _benchmark(model, sid, item, v0):
    config = dict(item.get("config", {}))
    experiment_id = item["experiment_id"]
    checkpoint = resolve_checkpoint(experiment_id)[1]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                               trust_remote_code=True)
    raw_texts = v0.text.fillna("").astype(str).tolist()
    prepared = _prepared_texts(tokenizer, raw_texts, model, config)
    encoded = tokenizer(prepared, add_special_tokens=True, truncation=False,
                        padding=False, return_attention_mask=False)["input_ids"]
    lengths = np.asarray([len(row) for row in encoded], dtype=np.int64)
    order = np.argsort(-lengths, kind="stable")[:min(SAMPLE_ROWS, len(v0))]
    sample = [raw_texts[int(i)] for i in order]
    sample_lengths = lengths[order]

    base_checkpoint = config.get("checkpoint")
    revision = item.get("model_revision") or config.get("revision")
    base_model = None
    if base_checkpoint and str(config.get("training_mode", "")).lower() not in {"full", "encoder"}:
        base_model = snapshot_download(repo_id=base_checkpoint, revision=revision,
                                       local_files_only=True,
                                       allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                                                       "pytorch_model*.bin", "tokenizer*", "vocab*", "merges.txt",
                                                       "special_tokens_map.json", "added_tokens.json", "*.py",
                                                       "generation_config.json"])
    spec = {
        "candidate_id": sid,
        "model": model,
        "checkpoint": str(checkpoint),
        "base_model": base_model,
        "config": {**config, "inference_batch_size": START_BATCH},
        "model_revision": revision,
        "calibration_temperature": float(item.get("calibration_temperature") or 1.0),
    }


def _benchmark_qwen_reconstruction(v0):
    from transformers import AutoModelForMultimodalLM
    from src.e19b_reconstruct import _generation_inputs, MODEL_REVISION

    repo = "Qwen/Qwen3.5-2B"
    model_dir = snapshot_download(repo_id=repo, revision=MODEL_REVISION,
                                  local_files_only=True,
                                  allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                                                  "tokenizer*", "vocab*", "merges.txt",
                                                  "special_tokens_map.json", "added_tokens.json", "*.py"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    class TextOnlyProcessor:
        def __init__(self, tok): self.tokenizer = tok
        def apply_chat_template(self, conversations, **kwargs):
            processor_kwargs = kwargs.pop("processor_kwargs", {}) or {}
            for key, value in processor_kwargs.items(): kwargs.setdefault(key, value)
            return self.tokenizer.apply_chat_template(conversations, **kwargs)
        def batch_decode(self, *args, **kwargs):
            return self.tokenizer.batch_decode(*args, **kwargs)

    processor = TextOnlyProcessor(tokenizer)
    raw_texts = v0.text.fillna("").astype(str).tolist()
    lens = tokenizer(raw_texts, add_special_tokens=True, truncation=False,
                     padding=False, return_attention_mask=False)["input_ids"]
    lengths = np.asarray([len(row) for row in lens], dtype=np.int64)
    order = np.argsort(-lengths, kind="stable")[:SAMPLE_ROWS]
    texts = [raw_texts[int(i)] for i in order]
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, device_map="auto", dtype="auto", local_files_only=True).eval()
    target = next(model.parameters()).device
    size = QWEN_GENERATION_BATCH
    start = 0; first_success = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    timer = time.perf_counter()
    while start < len(texts):
        current = min(size, len(texts)-start)
        try:
            inputs = _generation_inputs(processor, texts[start:start+current])
            inputs = {k: (v.to(target) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=512, use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id)
            prefix = inputs["input_ids"].shape[1]
            processor.batch_decode(outputs[:, prefix:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            if "inputs" in locals(): del inputs
            if "outputs" in locals(): del outputs
            __import__("gc").collect(); torch.cuda.empty_cache()
            if current <= 1: raise
            size = max(1, current // 2)
            continue
        if start == 0: first_success = current
        start += current
        del inputs, outputs
    elapsed = time.perf_counter()-timer
    result = {"model":"qwen_reconstruction", "status":"ok",
        "configured_probe_batch_size":QWEN_GENERATION_BATCH,
        "effective_batch_size":int(first_success or size),
        "sample_rows":len(texts), "longest_token_length":int(lengths[order[0]]),
        "seconds":round(elapsed,3), "responses_per_second":round(len(texts)/elapsed,3),
        "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/(1024**2),1)}
    del model,processor,tokenizer; __import__("gc").collect(); torch.cuda.empty_cache()
    return result


def _benchmark_embedding_prompt_encoder(v0):
    from transformers import AutoModel

    rec = pd.read_parquet(ROOT / "artifacts/analysis/e19b/reconstructed_prompts.parquet")
    rec = rec.set_index("id").loc[v0.id.astype(str)].reset_index()
    repo = "google/embeddinggemma-2"
    revision = "f9972ac22e696a105c482202646979e33e4b5a1e"
    model_dir = snapshot_download(repo_id=repo, revision=revision, local_files_only=True,
                                  allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                                                  "tokenizer*", "vocab*", "merges.txt",
                                                  "special_tokens_map.json", "added_tokens.json"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="right")
    texts = ["task: clustering | query: " + str(x or "")
             for x in rec.reconstructed_prompt.fillna("").astype(str)]
    lengths = np.asarray([len(x) for x in tokenizer(texts, truncation=False, padding=False)["input_ids"]])
    order = np.argsort(-lengths, kind="stable")[:128]
    sample = [texts[int(i)] for i in order]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=dtype).to("cuda").eval()
    size = EMBEDDING_BATCH; start=0; first_success=None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); timer=time.perf_counter()
    while start < len(sample):
        current=min(size,len(sample)-start)
        try:
            inputs=tokenizer(sample[start:start+current],padding=True,truncation=True,
                             max_length=512,return_tensors="pt").to("cuda")
            with torch.inference_mode():
                hidden=model(**inputs).last_hidden_state.float()
                mask=inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                torch.nn.functional.normalize(pooled,p=2,dim=1)
        except torch.cuda.OutOfMemoryError:
            if "inputs" in locals(): del inputs
            if "hidden" in locals(): del hidden
            if "mask" in locals(): del mask
            if "pooled" in locals(): del pooled
            __import__("gc").collect(); torch.cuda.empty_cache()
            if current <= 1: raise
            size=max(1,current//2); continue
        if start==0: first_success=current
        start+=current
        del inputs,hidden,mask,pooled
    elapsed=time.perf_counter()-timer
    result={"model":"embeddinggemma_prompt_encoder","status":"ok",
        "configured_probe_batch_size":EMBEDDING_BATCH,
        "effective_batch_size":int(first_success or size),"sample_rows":len(sample),
        "longest_token_length":int(lengths[order[0]]),"seconds":round(elapsed,3),
        "responses_per_second":round(len(sample)/elapsed,3),
        "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/(1024**2),1)}
    del model,tokenizer; __import__("gc").collect(); torch.cuda.empty_cache()
    return result
    del tokenizer, encoded, prepared
    gc_collect = __import__("gc").collect
    gc_collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    probabilities, metadata = predict_candidate(spec, sample, LABELS, Path("/"),
                                                return_batch_metadata=True)
    elapsed = time.perf_counter() - started
    assert probabilities.shape == (len(sample), len(LABELS))
    assert np.isfinite(probabilities).all()
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "sequence_id": sid,
        "experiment_id": experiment_id,
        "model": model,
        "max_length_config": int(config.get("max_length") or 384),
        "sample_rows": len(sample),
        "longest_token_lengths": sample_lengths[:10].astype(int).tolist(),
        "longest_token_length": int(sample_lengths[0]),
        "configured_probe_batch_size": START_BATCH,
        "effective_batch_size": int(metadata["effective_batch_size"]),
        "seconds": round(elapsed, 3),
        "responses_per_second": round(len(sample) / max(elapsed, 1e-9), 3),
        "peak_allocated_mib": round(peak_mib, 1),
        "status": "ok",
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Batch-size benchmark requires the local CUDA GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(MANIFEST.read_text())
    v0 = pd.read_parquet(V0_PATH)
    if len(v0) != 2699 or not {"id", "text", "model"}.issubset(v0.columns):
        raise ValueError("Expected the complete 2,699-row V0 dataset")
    cached = {}
    if OUTPUT.is_file():
        try:
            old = json.loads(OUTPUT.read_text())
            cached = {item["model"]: item for item in old.get("results", [])}
        except (OSError, json.JSONDecodeError, KeyError):
            cached = {}
    results = []
    for model, (sid, item) in _representatives(manifest).items():
        prior = cached.get(model)
        if (prior and prior.get("status") == "ok" and
                prior.get("configured_probe_batch_size") == START_BATCH):
            results.append(prior)
            print(f"Reusing verified local probe for {model}: batch {prior['effective_batch_size']}",flush=True)
            continue
        print(f"Benchmarking {model} ({sid})", flush=True)
        try:
            result = _benchmark(model, sid, item, v0)
        except Exception as exc:
            result = {"sequence_id": sid, "experiment_id": item["experiment_id"],
                      "model": model, "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
    for benchmark in (_benchmark_qwen_reconstruction, _benchmark_embedding_prompt_encoder):
        expected = QWEN_GENERATION_BATCH if benchmark is _benchmark_qwen_reconstruction else EMBEDDING_BATCH
        model_key = "qwen_reconstruction" if benchmark is _benchmark_qwen_reconstruction else "embeddinggemma_prompt_encoder"
        prior = cached.get(model_key)
        if (prior and prior.get("status") == "ok" and
                prior.get("configured_probe_batch_size") == expected):
            results.append(prior)
            print(f"Reusing verified local probe for {model_key}: batch {prior['effective_batch_size']}",flush=True)
            continue
        try:
            result = benchmark(v0)
        except Exception as exc:
            result = {"model": benchmark.__name__.removeprefix("_benchmark_"),
                      "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            torch.cuda.empty_cache()
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
        partial = {
            "experiment": "E20-D inference batch benchmark",
            "device": torch.cuda.get_device_name(0),
            "device_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1),
            "v0_rows": len(v0),
            "selection": f"Top {SAMPLE_ROWS} V0 responses by each model tokenizer length, longest first",
            "probe_policy": f"Classifier batches start at {START_BATCH}; Qwen generation starts at {QWEN_GENERATION_BATCH}; embedding prompt encoder starts at {EMBEDDING_BATCH}; all retry with production CUDA OOM halving",
            "cpu_threads": 1,
            "results": results,
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(partial, indent=2) + "\n")
    document = {
        "experiment": "E20-D inference batch benchmark",
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1),
        "v0_rows": len(v0),
        "selection": f"Top {SAMPLE_ROWS} V0 responses by each model tokenizer length, longest first",
        "probe_policy": f"Classifier batches start at {START_BATCH}; Qwen generation starts at {QWEN_GENERATION_BATCH}; embedding prompt encoder starts at {EMBEDDING_BATCH}; all retry with production CUDA OOM halving",
        "cpu_threads": 1,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
