#!/usr/bin/env python3
"""Find the largest stable UltraBERT inference batch on V0 and V0-EXT."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ[name] = "1" if name != "TOKENIZERS_PARALLELISM" else "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.inference import _predict_neural

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "ultrabert380_v3.0-baseline_valv0_data376d14f2_split0576ffdd_seed2026_cfgc18b26e3_code2105fce"
MODEL_MANIFEST = ROOT / "artifacts/predictions" / f"{EXPERIMENT_ID}_manifest.json"
DATASETS = {
    "v0": ROOT / "data/processed/dataset_v0.parquet",
    "v0_ext": ROOT / "data/processed/dataset_v0_ext.parquet",
}
OUT = ROOT / "artifacts/analysis/v3/validation_inference_batch_sizes.json"
PROGRESS = ROOT / "artifacts/analysis/v3/validation_inference_batch_sizes_progress.json"
CATALOG = ROOT / "configs/ensemble_catalog_v2.json"
RUN_ID = "V3.0-VALIDATION-INFERENCE-BATCH"
INITIAL_BATCH = 64


def _catalog(stage: str, status: str = "Running", **fields) -> None:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = next(row for row in data["experiments"] if row.get("id") == RUN_ID)
    entry.update({"status": status, "current_stage": stage, **fields})
    fd, tmp = tempfile.mkstemp(dir=CATALOG.parent, prefix=".v3-inference-batch.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    os.replace(tmp, CATALOG)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"experiment_id": RUN_ID, "status": status,
        "current_stage": stage, "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **fields}, indent=2) + "\n", encoding="utf-8")


def _probe_candidate(model: str, checkpoint: Path, classes: list[str], texts: list[str],
                     max_length: int, size: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        probabilities, returned_classes, metadata = _predict_neural(
            model, checkpoint, texts[:size], classes,
            batch_size=size, max_length=max_length,
            return_batch_metadata=True)
        elapsed = time.perf_counter() - started
        if returned_classes != classes or probabilities.shape != (size, len(classes)):
            raise ValueError("inference output has an unexpected shape or label order")
        if not np.isfinite(probabilities).all():
            raise ValueError("inference returned non-finite probabilities")
        effective = int(metadata["effective_batch_size"])
        return {"candidate_batch_size": size, "fits_as_requested": effective == size,
            "effective_batch_size": effective, "sample_rows": size,
            "call_seconds": round(elapsed, 3),
            "forward_seconds": round(float(metadata.get("pure_inference_seconds", 0.0)), 3),
            "examples_per_second": round(size / max(elapsed, 1e-9), 3),
            "forward_examples_per_second": round(size / max(float(metadata.get("pure_inference_seconds", 0.0)), 1e-9), 3),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            "oom_backoff_used": effective != size}
    finally:
        torch.cuda.empty_cache()


def _largest_safe(model: str, checkpoint: Path, classes: list[str], texts: list[str],
                  max_length: int, cap: int) -> tuple[int, list[dict]]:
    attempts: list[dict] = []
    tested: dict[int, dict] = {}

    def attempt(size: int) -> dict:
        size = min(int(size), cap)
        if size not in tested:
            record = _probe_candidate(model, checkpoint, classes, texts, max_length, size)
            attempts.append(record)
            tested[size] = record
            _catalog(f"Probing batch {size} on validation data", attempted_batches=[x["candidate_batch_size"] for x in attempts])
            print(json.dumps(record), flush=True)
        return tested[size]

    first = min(INITIAL_BATCH, cap)
    if attempt(first)["fits_as_requested"]:
        low = first
        high = None
        while low < cap:
            candidate = min(cap, low * 2)
            if attempt(candidate)["fits_as_requested"]:
                low = candidate
            else:
                high = candidate
                break
        if high is None:
            return low, attempts
    else:
        low, high = 0, first

    while high - low > 1:
        candidate = (low + high) // 2
        if attempt(candidate)["fits_as_requested"]:
            low = candidate
        else:
            high = candidate
    return low, attempts


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validation inference batch calibration requires CUDA")
    torch.set_num_threads(1)
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["best_checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    model = "ultrabert380"
    config = manifest["config"]
    max_length = int(config.get("max_length", 384))
    classes = list(manifest["classes"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                               trust_remote_code=True)
    result = {"experiment_id": RUN_ID, "status": "Running", "model": model,
        "checkpoint_experiment": EXPERIMENT_ID, "checkpoint_path": str(checkpoint),
        "checkpoint_max_length": max_length, "device": torch.cuda.get_device_name(),
        "device_total_vram_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        "initial_batch": INITIAL_BATCH, "search": "exponential then integer binary search",
        "sample_policy": "longest tokenized validation responses first; all rows at or above tested batch size",
        "datasets": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, path in DATASETS.items():
        _catalog(f"Preparing longest-sequence probe for {name}", dataset=name)
        frame = pd.read_parquet(path)
        if not {"id", "text", "model"}.issubset(frame.columns) or frame.id.astype(str).duplicated().any():
            raise ValueError(f"{path} is missing required columns or has duplicate IDs")
        texts = frame.text.fillna("").astype(str).tolist()
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False,
                            padding=False, return_attention_mask=False)["input_ids"]
        lengths = np.asarray([min(len(row), max_length) for row in encoded], dtype=np.int64)
        order = np.argsort(-lengths, kind="stable")
        ordered_texts = [texts[int(i)] for i in order]
        cap = len(ordered_texts)
        safe, attempts = _largest_safe(model, checkpoint, classes, ordered_texts,
                                        max_length, cap)
        result["datasets"][name] = {"rows": len(frame), "distinct_labels": int(frame.model.nunique()),
            "max_truncated_token_length": int(lengths.max()),
            "tested_longest_token_lengths": lengths[order[:min(cap, max(attempt["candidate_batch_size"] for attempt in attempts))]].astype(int).tolist(),
            "maximum_safe_batch_size": safe, "attempts": attempts}
        OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"RESULT {name}: maximum safe batch={safe}", flush=True)
        del encoded, lengths, ordered_texts
        torch.cuda.empty_cache()
    result["conservative_batch_for_both_datasets"] = min(
        item["maximum_safe_batch_size"] for item in result["datasets"].values())
    result["status"] = "Complete"
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _catalog("Complete", "Complete", result_path=str(OUT.relative_to(ROOT)),
        maximum_safe_batch_by_dataset={name: item["maximum_safe_batch_size"]
            for name, item in result["datasets"].items()},
        conservative_batch_for_both_datasets=result["conservative_batch_for_both_datasets"])
    print(json.dumps({"complete": True, "result": str(OUT),
        "maximum_safe_batch_by_dataset": {name: item["maximum_safe_batch_size"]
            for name, item in result["datasets"].items()},
        "conservative_batch_for_both_datasets": result["conservative_batch_for_both_datasets"]}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        try:
            _catalog("Failed", "Failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            raise
