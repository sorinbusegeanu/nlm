"""Benchmark each frozen E30-B member on one 4,000-row local GPU workload."""
from __future__ import annotations

import csv, json, os, time, gc
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from scripts.publish_kaggle_checkpoint_store import resolve_checkpoint
from src.inference import predict_candidate
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/analysis/e30/e30_b_manifest.json"
DATA = ROOT / "data/processed/dataset_v0_group.parquet"
TRUST = ROOT / "docs/trust-list-2-v039.csv"
OUT = ROOT / "artifacts/analysis/e30/inference_benchmark.json"
CAP = 1024
ROWS = 4000


def _sample(frame):
    """Select 4k rows balanced across scenarios, labels and text length."""
    rng = np.random.default_rng(20261123)
    picked = []
    by_scenario = frame.groupby("scenario_id", sort=True)
    per_scenario = ROWS // len(by_scenario)
    remainder = ROWS % len(by_scenario)
    for n, (_, part) in enumerate(by_scenario):
        need = per_scenario + int(n < remainder)
        part = part.copy()
        words = part.text.astype(str).str.split().str.len()
        part["length_bucket"] = pd.cut(words, [-1, 99, 300, np.inf], labels=["short", "medium", "long"])
        strata = part.groupby(["model", "length_bucket"], observed=False, sort=True)
        chosen = []
        # Proportional allocation with deterministic random order in each stratum.
        for _, group in strata:
            if len(group):
                order = rng.permutation(group.index.to_numpy())
                chosen.extend(order.tolist())
        if len(chosen) > need:
            chosen = rng.permutation(np.asarray(chosen))[:need].tolist()
        picked.extend(chosen)
    sample = frame.loc[picked].reset_index(drop=True)
    if len(sample) != ROWS:
        raise ValueError(f"Expected {ROWS} benchmark rows, selected {len(sample)}")
    return sample


def _run_one(sid, member, spec_base, texts, batch_cap=None):
    model = member["model"]
    effective_cap = int(batch_cap or member["config"].get("inference_batch_size") or CAP)
    spec = dict(spec_base)
    spec.update({"candidate_id": sid, "model": model, "checkpoint": str(spec_base["checkpoint"]),
                 "config": {**member["config"], "inference_batch_size": effective_cap},
                 "model_revision": member.get("model_revision"),
                 "calibration_temperature": member.get("calibration_temperature", 1.0)})
    if model not in {"svc", "catboost_stylometry"}:
        checkpoint_conf = member["config"].get("checkpoint")
        revision = member.get("model_revision") or member["config"].get("revision")
        if checkpoint_conf and str(member["config"].get("training_mode", "")).lower() not in {"full", "encoder"}:
            spec["base_model"] = snapshot_download(repo_id=checkpoint_conf, revision=revision,
                local_files_only=True, allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                    "pytorch_model*.bin", "tokenizer*", "vocab*", "merges.txt", "special_tokens_map.json",
                    "added_tokens.json", "*.py", "generation_config.json"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    probabilities, metadata = predict_candidate(spec, texts, LABELS, ROOT,
                                                return_batch_metadata=True)
    total = time.perf_counter() - started
    if probabilities.shape != (len(texts), len(LABELS)) or not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid {sid} benchmark probabilities")
    if not np.allclose(probabilities.sum(1), 1, atol=1e-4):
        raise ValueError(f"Unnormalized {sid} benchmark probabilities")
    device = "CPU" if model in {"svc", "catboost_stylometry"} else torch.cuda.get_device_name(0)
    is_neural = model not in {"svc", "catboost_stylometry"}
    peak_allocated = torch.cuda.max_memory_allocated() if is_neural else None
    peak_reserved = torch.cuda.max_memory_reserved() if is_neural else None
    record = {"checkpoint_id": sid, "model": model, "device": device,
        "rows": len(texts), "configured_batch_cap": effective_cap,
        "maximum_stable_batch_size": metadata.get("effective_batch_size"),
        "peak_allocated_mib": round(peak_allocated / (1024**2), 1) if peak_allocated is not None else None,
        "peak_reserved_mib": round(peak_reserved / (1024**2), 1) if peak_reserved is not None else None,
        "model_loading_seconds": metadata.get("model_loading_seconds"),
        "pure_inference_seconds": metadata.get("pure_inference_seconds"),
        "cleanup_seconds": metadata.get("cleanup_seconds"),
        "total_checkpoint_seconds": total,
        "throughput_rows_per_second": len(texts) / max(float(metadata.get("pure_inference_seconds") or total), 1e-9),
        "timing_note": metadata.get("timing_note"), "batch_size_note": metadata.get("batch_size_note"),
        "status": "ok"}
    del probabilities, spec
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return record


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E30 local benchmark requires CUDA")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    deploy = json.loads(MANIFEST.read_text())
    rows = pd.read_parquet(DATA)
    sample = _sample(rows)
    # Resolve names against the same pinned trust snapshot used for E30.
    table = pd.read_csv(TRUST, dtype=str).drop_duplicates("sequence_id").set_index("sequence_id")
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    completed = {x["checkpoint_id"]: x for x in old.get("checkpoints", []) if x.get("status") == "ok"}
    result = {"experiment": "E30 corrected-roster inference benchmark", "device": torch.cuda.get_device_name(0),
        "device_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory/(1024**2),1),
        "sample_rows": len(sample), "sample_seed": 20261123,
        "sample_policy": "80 rows per v0_group scenario, deterministic stratification by true label and word-length bucket",
        "batch_cap": CAP, "checkpoints": list(completed.values())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for sid in deploy["roster"]:
        if sid in completed:
            print(f"reuse {sid}", flush=True); continue
        row = table.loc[sid]
        experiment_id = row.experiment_id
        _, checkpoint = resolve_checkpoint(experiment_id)
        manifest_path = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
        run_manifest = json.loads(manifest_path.read_text())
        member = {"model": row.model, "experiment_id": experiment_id,
            "config": run_manifest.get("config", {}),
            "model_revision": run_manifest.get("model_revision"),
            "calibration_temperature": run_manifest.get("calibration_temperature", 1.0)}
        try:
            record = _run_one(sid, member, {"checkpoint": checkpoint}, sample.text.astype(str).tolist())
        except Exception as exc:
            record = {"checkpoint_id": sid, "model": row.model, "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
        completed[sid] = record
        result["checkpoints"] = [completed[key] for key in deploy["roster"] if key in completed]
        result["completed_count"] = len(result["checkpoints"])
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
    records = result["checkpoints"]
    # Measure the actual sequential end-to-end pass for the exact current roster.
    # Per-checkpoint probe timings above remain reusable; this pass measures the
    # wall clock including Python/model setup and inter-checkpoint cleanup.
    started = time.perf_counter()
    e2e_records = []
    for sid in deploy["roster"]:
        row = table.loc[sid]
        experiment_id = row.experiment_id
        _, checkpoint = resolve_checkpoint(experiment_id)
        run_manifest = json.loads((ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json").read_text())
        member = {"model": row.model, "experiment_id": experiment_id,
            "config": run_manifest.get("config", {}),
            "model_revision": run_manifest.get("model_revision"),
            "calibration_temperature": run_manifest.get("calibration_temperature", 1.0)}
        cp_started = time.perf_counter()
        measured = completed.get(sid, {}).get("maximum_stable_batch_size")
        record = _run_one(sid, member, {"checkpoint": checkpoint},
                          sample.text.astype(str).tolist(), batch_cap=measured)
        record["e2e_elapsed_before_checkpoint_seconds"] = cp_started - started
        e2e_records.append(record)
        print(f"e2e {sid}: {record['total_checkpoint_seconds']:.1f}s", flush=True)
    end_to_end = time.perf_counter() - started
    sum_pure = float(sum(x.get("pure_inference_seconds") or 0 for x in e2e_records))
    sum_checkpoint = float(sum(x.get("total_checkpoint_seconds") or 0 for x in e2e_records))
    result["end_to_end_checkpoint_records"] = e2e_records
    e2e_by_id = {x["checkpoint_id"]: x for x in e2e_records}
    for item in result["checkpoints"]:
        observed = e2e_by_id[item["checkpoint_id"]]
        effective = observed.get("maximum_stable_batch_size")
        if effective and item.get("maximum_stable_batch_size"):
            item["maximum_stable_batch_size"] = min(
                int(item["maximum_stable_batch_size"]), int(effective))
        elif effective:
            item["maximum_stable_batch_size"] = int(effective)
        item["e2e_observed_batch_size"] = effective
    result["sum_pure_inference_seconds"] = sum_pure
    result["sum_checkpoint_total_seconds"] = sum_checkpoint
    result["end_to_end_seconds"] = float(end_to_end)
    result["wall_vs_pure_inference_seconds"] = float(end_to_end - sum_pure)
    result["wall_vs_checkpoint_sum_seconds"] = float(end_to_end - sum_checkpoint)
    from src.e28_deploy import distribute_checkpoints
    schedule_rows = [{"candidate_id": x["checkpoint_id"], "model": x["model"], "files": []}
                     for x in e2e_records]
    runtime_by_id = {x["checkpoint_id"]: x["total_checkpoint_seconds"] for x in e2e_records}
    runtime_by_id["__sample_rows__"] = len(sample)
    schedule = distribute_checkpoints(schedule_rows, runtime_by_id, {}, len(sample))
    result["kaggle_estimate"] = {
        "scheduler": "E28 greedy longest-estimated-runtime-first two-GPU distribution",
        "rows": len(sample),
        "estimated_worker_seconds": [float(x["estimated_seconds"]) for x in schedule],
        "estimated_wall_seconds": float(max(x["estimated_seconds"] for x in schedule)),
        "worker_checkpoint_ids": [x["checkpoint_ids"] for x in schedule],
        "note": "Extrapolates local RTX 5070 Ti per-checkpoint totals to the same 4,000-row workload and applies the E28 two-worker scheduler. Kaggle T4 throughput and OOM-driven batch changes are not hardware-calibrated; the visible Kaggle test file is only a sample."}
    (OUT.parent / "inference_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
