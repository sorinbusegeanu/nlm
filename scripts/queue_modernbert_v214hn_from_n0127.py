"""Queue one weighted ModernBERT continuation after the active DeBERTa V2.14 job."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
SOURCE_EXPERIMENT = (
    "modernbert_large_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_"
    "cfgbb2e9041_code3458579"
)
SOURCE_MANIFEST = ROOT / "artifacts/predictions" / f"{SOURCE_EXPERIMENT}_manifest.json"
DATASET = "v2.14HN"
VALIDATION_DATASET = "v0"
DEPENDENCY_QUEUE = ROOT / "artifacts/training_queues" / (
    "deberta_v3_large_v214_generic_20261118T092824Z/queue.json"
)
QUEUE_ROOT = ROOT / "artifacts/training_queues"
QUEUE_TAG = "modernbert_v214hn_weighted_continue_n0127"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _active_resources() -> tuple[list[dict], list[int]]:
    processes = []
    script_name = Path(__file__).name
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models" in parts[1] and script_name not in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                continue
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    job = state["results"][0]
    dependency = Path(state["depends_on_queue_json"])
    if not dependency.is_absolute():
        dependency = ROOT / dependency
    while True:
        if not dependency.is_file():
            state.update(status="blocked", error=f"dependency queue missing: {dependency}", updated_at_utc=now())
            save(queue_path, state)
            return 1
        previous = json.loads(dependency.read_text(encoding="utf-8"))
        state.update(dependency_status=previous.get("status"), last_dependency_check_at_utc=now())
        if previous.get("status") in TERMINAL and previous.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = previous["finished_at_utc"]
            break
        state.update(status="waiting_for_previous_queue", updated_at_utc=now())
        save(queue_path, state)
        time.sleep(20)

    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        try:
            processes, gpu = _active_resources()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(status="blocked", error=f"resource check failed: {exc}", updated_at_utc=now())
            save(queue_path, state)
            return 1
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_path, state)
        if not processes and not gpu:
            break
        time.sleep(20)

    job.update(status="running", started_at_utc=now())
    state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
    save(queue_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting offline: {' '.join(job['command'])}\n")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = process.pid
            save(queue_path, state)
            code = process.wait()
    except Exception as exc:
        code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"
    job.update(status="completed" if code == 0 else "failed", exit_code=code,
                elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
    state.update(status=job["status"], active_job=None, exit_code=code,
                 finished_at_utc=now(), updated_at_utc=now())
    save(queue_path, state)
    return code


def create_queue() -> Path:
    if not DEPENDENCY_QUEUE.is_file():
        raise FileNotFoundError(f"active training queue not found: {DEPENDENCY_QUEUE}")
    dependency = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError("the DeBERTa queue is already terminal; refusing to create a stale dependent job")

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(f"N0127 manifest missing: {SOURCE_MANIFEST}")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_config = source.get("config") or {}
    if source.get("model") != "modernbert_large" or source.get("dataset") != "v2.14":
        raise ValueError("N0127 does not resolve to the expected ModernBERT V2.14 experiment")
    if float(source_config.get("learning_rate", 0)) <= 0:
        raise ValueError("N0127 manifest has no valid original learning rate")
    source_learning_rate = float(source_config["learning_rate"])
    learning_rate = source_learning_rate * 0.25

    checkpoint = Path(source.get("best_checkpoint_path") or source["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"N0127 selected model checkpoint is unavailable: {checkpoint}")
    trainer_state = checkpoint / "continuation_checkpoint"
    required_state = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    if not trainer_state.is_dir() or any(not (trainer_state / name).is_file() for name in required_state):
        raise FileNotFoundError(f"N0127 exact continuation state is incomplete: {trainer_state}")

    from src.pipeline import _training_partitions, _continuation_source_matches
    train, valid = _training_partitions(DATASET, VALIDATION_DATASET, ROOT / "data/processed")
    if len(valid) != 2699 or train.model.nunique() != 18 or set(train.model.astype(str)) != set(source["classes"]):
        raise ValueError("V2.14HN/full-V0 rows or class labels differ from expectations")
    weights = np.asarray(train["sampling_weight"], dtype=np.float64)
    if len(weights) != len(train) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("V2.14HN sampling_weight must be complete, finite and strictly positive")
    if not _continuation_source_matches(source, DATASET, VALIDATION_DATASET, train, valid,
                                        None, allow_cross_dataset_source=True):
        raise ValueError("N0127 full-V0 validation rows differ from V2.14HN continuation validation")

    # Start from N0127's exact fine-tuning recipe and change only the requested
    # epoch count, learning rate, sampling switch, and experiment identity.
    config = dict(source_config)
    config.update({"model": "modernbert_large", "epochs": 1,
        "learning_rate": learning_rate, "weighted_sampling": True,
        "experiment_name": "modernbert_large_v2.14HN_weighted_continue_N0127"})
    required_batch_settings = ("batch_size", "auto_batch_size", "max_batch_size",
                               "batch_probe_safety_fraction", "gradient_accumulation_steps")
    if any(key not in config for key in required_batch_settings):
        raise ValueError("N0127 config is missing its recorded batch/memory optimization settings")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"modernbert_v214hn_weighted_continue_n0127_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config_path = config_dir / "modernbert_large_v214hn_continue_n0127.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "continue-encoder",
        "--model", "modernbert_large", "--dataset", DATASET,
        "--validation-dataset", VALIDATION_DATASET, "--from-experiment", SOURCE_EXPERIMENT,
        "--epochs", "1", "--learning-rate", f"{learning_rate:.12g}", "--seed", "2026",
        "--config", str(config_path), "--allow-cross-dataset-source",
        "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    profile = json.loads((ROOT / "data/processed/dataset_profile_v2.14HN.json").read_text(encoding="utf-8"))
    job = {"index": 1, "job_id": config["experiment_name"] + "_" + stamp,
        "model": "modernbert_large", "display_name": "ModernBERT-large",
        "source_checkpoint_id": "N0127", "source_experiment_id": SOURCE_EXPERIMENT,
        "source_checkpoint_path": str(checkpoint), "source_trainer_state_path": str(trainer_state),
        "source_learning_rate": source_learning_rate, "learning_rate": learning_rate,
        "dataset": DATASET, "dataset_path": "data/processed/dataset_v2.14HN.parquet",
        "dataset_rows": len(train), "dataset_sha256": profile.get("dataset_sha256"),
        "validation_dataset": VALIDATION_DATASET, "validation_rows": len(valid),
        "epochs": 1, "seed": 2026, "training_mode": "continued_full_fine_tuning",
        "weighted_sampling": True, "sampling_weight_column": "sampling_weight",
        "sampling_replacement": True, "samples_per_epoch": len(train),
        "expected_sampling_distribution_by_label": {
            str(label): float(total / weights.sum())
            for label, total in train.assign(_w=weights).groupby("model", sort=True)._w.sum().items()},
        "preserved_settings": {key: config[key] for key in (
            "batch_size", "inference_batch_size", "auto_batch_size", "max_batch_size",
            "batch_probe_safety_fraction", "gradient_accumulation_steps", "gradient_checkpointing",
            "max_length", "attention_backend", "lr_scheduler_type", "warmup_ratio", "optimizer",
            "weight_decay", "early_stopping_patience", "checkpoint_selection_metric") if key in config},
        "optimizer_state_policy": "reinitialized_for_learning_rate_override_from_N0127_selected_weights",
        "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "log": "01_modernbert_v214hn_weighted_continue_n0127.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": QUEUE_TAG,
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Continue N0127 ModernBERT-large on V2.14HN for one epoch at one quarter of its original LR, using sampling_weight; validate on full V0.",
        "execution_policy": "wait for the active DeBERTa V2.14 queue to finish, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"), "dataset": DATASET,
        "dataset_path": "data/processed/dataset_v2.14HN.parquet", "dataset_rows": len(train),
        "dataset_sha256": profile.get("dataset_sha256"), "validation_dataset": VALIDATION_DATASET,
        "validation_rows": len(valid), "source_experiment_id": SOURCE_EXPERIMENT,
        "source_checkpoint_id": "N0127", "model": "modernbert_large", "revision": source.get("model_revision"),
        "results": [job], "active_job": None}
    save(queue_dir / "queue.json", state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["worker_pid"] = process.pid
    save(queue_dir / "queue.json", state)
    return queue_dir


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_previous_queue"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
