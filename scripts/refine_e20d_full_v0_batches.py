"""Refine E20-D batch limits and retime full-V0 inference at those limits."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from scripts.publish_kaggle_checkpoint_store import ROOT, resolve_checkpoint
from src.inference import predict_candidate
from src.v2_checkpoint_trust import LABELS

MANIFEST = ROOT / "artifacts/kaggle/e20d-checkpoints-v001/e20d_manifest.json"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
JSON_PATH = ROOT / "artifacts/analysis/e20d_full_v0_batch_calibration.json"
CSV_PATH = ROOT / "artifacts/analysis/e20d_full_v0_batch_calibration.csv"
MAX_BATCH = 2699
CPU_MODELS = {"svc", "catboost_stylometry"}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prep(tokenizer, texts, model, config):
    values = list(map(str, texts))
    if model == "embeddinggemma2":
        values = ["task: classification | query: " + x for x in values]
    if config.get("use_chat_template", False):
        prompt = config.get("chat_template_user_prompt", "Identify the source model for the assistant response.")
        values = [tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": text}],
            tokenize=False, add_generation_prompt=False) for text in values]
    return values


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    doc = json.loads(JSON_PATH.read_text())
    source = json.loads(MANIFEST.read_text())
    v0 = pd.read_parquet(V0_PATH)
    if len(v0) != MAX_BATCH:
        raise ValueError(f"Expected {MAX_BATCH} V0 rows")
    raw = v0.text.fillna("").astype(str).tolist()
    data_hash = file_sha(V0_PATH)
    if data_hash != doc.get("v0_sha256"):
        raise ValueError("V0 data hash changed since the initial calibration")
    results = {r["checkpoint_id"]: r for r in doc["results"]}
    for checkpoint_id in source["checkpoint_ids"]:
        result = results[checkpoint_id]
        if result["status"] != "ok" or result["model"] in CPU_MODELS:
            continue
        if result.get("refined_max_batch_size"):
            print(f"Reuse refined {checkpoint_id}: {result['refined_max_batch_size']}", flush=True)
            continue
        item = source["checkpoint_members"][checkpoint_id]
        config = dict(item["config"])
        _, checkpoint = resolve_checkpoint(item["experiment_id"])
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
        prepared = prep(tokenizer, raw, item["model"], config)
        encoded = tokenizer(prepared, add_special_tokens=True, truncation=False,
                            padding=False, return_attention_mask=False)["input_ids"]
        lengths = np.asarray([len(row) for row in encoded], dtype=np.int32)
        order = np.argsort(-lengths, kind="stable")
        sorted_texts = [raw[int(i)] for i in order]
        longest = int(lengths[order[0]])
        del tokenizer, prepared, encoded, lengths
        gc.collect()

        safe = int(result["max_batch_size"])
        # The halving search failed immediately above this range. Verify that
        # upper is indeed an OOM, then binary-search for the largest passing N.
        upper = min(MAX_BATCH, safe * 2 + 1)
        spec = {"candidate_id": checkpoint_id, "model": item["model"],
                "checkpoint": str(checkpoint), "base_model": None,
                "config": {**config, "inference_batch_size": MAX_BATCH},
                "model_revision": item.get("model_revision"),
                "calibration_temperature": float(item.get("calibration_temperature") or 1.0)}
        probes = []

        def probe(size):
            torch.cuda.empty_cache()
            sample = sorted_texts[:size]
            spec["config"]["inference_batch_size"] = size
            started = time.perf_counter()
            probs, meta = predict_candidate(spec, sample, LABELS, Path("/"), return_batch_metadata=True)
            elapsed = time.perf_counter() - started
            if np.asarray(probs).shape != (size, len(LABELS)):
                raise ValueError(f"{checkpoint_id} probe shape mismatch at batch {size}")
            passed = int(meta["effective_batch_size"]) == size
            probes.append({"batch_size": size, "passed": passed,
                           "effective_batch_size": int(meta["effective_batch_size"]),
                           "seconds": round(elapsed, 3)})
            del probs
            gc.collect()
            torch.cuda.empty_cache()
            return passed

        if upper > safe and probe(upper):
            safe = upper
            upper = min(MAX_BATCH, upper * 2)
            while upper > safe and probe(upper):
                safe = upper
                upper = min(MAX_BATCH, upper * 2)
                if safe == MAX_BATCH:
                    break
        if safe < upper:
            # In the existing halving path upper may be a known failing value.
            if not any(p["batch_size"] == upper and not p["passed"] for p in probes):
                probe(upper)
            if not probes[-1]["passed"] and probes[-1]["batch_size"] == upper:
                lo, hi = safe, upper
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if probe(mid):
                        lo = mid
                    else:
                        hi = mid
                safe = lo
        result["refined_max_batch_size"] = int(safe)
        result["batch_limit_probes"] = probes
        result["longest_token_length"] = longest

        # Time one complete, row-aligned V0 inference at the refined batch.
        spec["config"]["inference_batch_size"] = safe
        sorted_texts = [raw[int(i)] for i in order]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        probs, meta = predict_candidate(spec, sorted_texts, LABELS, Path("/"), return_batch_metadata=True)
        elapsed = time.perf_counter() - started
        probs = np.asarray(probs)
        if probs.shape != (MAX_BATCH, len(LABELS)) or int(meta["effective_batch_size"]) != safe:
            raise ValueError(f"{checkpoint_id} final batch verification failed: {probs.shape}, {meta}")
        restored = np.empty_like(probs)
        restored[order] = probs
        if not np.isfinite(restored).all():
            raise ValueError(f"{checkpoint_id} returned non-finite probabilities")
        result["inference_seconds"] = round(elapsed, 3)
        result["responses_per_second"] = round(MAX_BATCH / max(elapsed, 1e-9), 3)
        result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        result["probability_sha256_float32"] = hashlib.sha256(
            np.asarray(restored, dtype=np.float32).tobytes()).hexdigest()
        result["batch_note"] = "Exact maximum safe batch by binary search on longest-token V0 rows; full V0 retimed at this batch."
        print(json.dumps({"checkpoint_id": checkpoint_id,
                          "refined_max_batch_size": safe,
                          "inference_seconds": result["inference_seconds"],
                          "peak_allocated_mib": result["peak_allocated_mib"],
                          "probes": probes}, sort_keys=True), flush=True)
        del restored, probs, spec
        gc.collect()
        torch.cuda.empty_cache()
        doc["probe_method"] = (
            "For each GPU checkpoint, binary-search the largest batch whose first forward pass "
            "fits on the longest V0 examples; then run and time all 2,699 V0 rows at that exact "
            "batch. Outputs are restored to original V0 order. CPU baselines are evaluated whole-array.")
        doc["results"] = [results[x] for x in source["checkpoint_ids"]]
        JSON_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        pd.DataFrame(doc["results"]).to_csv(CSV_PATH, index=False)
    doc["results"] = [results[x] for x in source["checkpoint_ids"]]
    JSON_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(doc["results"]).to_csv(CSV_PATH, index=False)
    print(f"Wrote refined results to {JSON_PATH} and {CSV_PATH}", flush=True)


if __name__ == "__main__":
    run()
