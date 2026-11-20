#!/usr/bin/env python3
"""Create or explicitly run the pending Granite V0-exposed training queue entry.

`--create` only writes queue metadata. Training can run only when a caller
explicitly invokes `--worker QUEUE_DIR` after the dependency queue completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DEPENDENCY = QUEUE_ROOT / "modernbert_v214cw_suite_20261119T162227Z/queue.json"
DATASET_ID = "granite4_350m_v214_openpool16_v0w10"
DATASET_PATH = ROOT / "data/processed" / f"dataset_{DATASET_ID}.parquet"
PROFILE_PATH = ROOT / "data/processed" / f"dataset_profile_{DATASET_ID}.json"
CONFIG = "configs/models/granite4_350m_v214_openpool16_v0w10.yaml"
QUEUE_TAG = "granite4_350m_v214_openpool16_v0w10"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_queue() -> Path:
    if not DEPENDENCY.is_file():
        raise FileNotFoundError(f"training dependency queue is missing: {DEPENDENCY}")
    if not DATASET_PATH.is_file() or not PROFILE_PATH.is_file():
        raise FileNotFoundError("weighted training dataset artifacts must be built before queue registration")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    dependency = json.loads(DEPENDENCY.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"{QUEUE_TAG}_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    command = [PYTHON, "-m", "src.pipeline", "models", "train",
        "--model", "granite4_350m", "--dataset", DATASET_ID,
        "--validation-dataset", "none", "--no-validation", "--config", CONFIG,
        "--seed", "2026", "--accept-config-defaults"]
    job = {"index": 1, "job_id": queue_id, "model": "granite4_350m",
        "dataset": DATASET_ID, "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "dataset_rows": profile["rows"], "dataset_sha256": profile["dataset_sha256"],
        "components": profile["component_hashes"], "sampling_components": profile["sampling_components"],
        "sampling_weight_sums": profile["sampling_weight_sums"],
        "dataset_config": "configs/data/granite4_350m_v214_openpool16_v0w10.yaml",
        "dataset_config_sha256": profile.get("configuration_sha256"),
        "training_mode": "LoRA with per-row WeightedRandomSampler",
        "validation_dataset": None, "validation_enabled": False,
        "v0_exposed": True, "trust_list_eligible": False,
        "seed": 2026, "epochs": 4, "early_stopping": False,
        "save_every_epoch": True, "retain_epoch_checkpoints": 4,
        "expected_epoch_checkpoints": ["epoch_1", "epoch_2", "epoch_3", "epoch_4"],
        "config": CONFIG, "command": command,
        "model_config_sha256": sha256(ROOT / CONFIG),
        "max_length": 512,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                         "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"},
        "log": "01_granite4_350m_v214_openpool16_v0w10.log", "status": "queued",
        "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": QUEUE_TAG,
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Train Granite 4 350M for four epochs on V2.14, Open Pool v16 increment, and V0 weighted 1:1:10; no validation.",
        "execution_policy": "Run sequentially after the existing ModernBERT V2.14CW suite completes and GPU resources are idle.",
        "worker_policy": "Not started at user request; a queue runner must explicitly invoke this queue's worker.",
        "depends_on_queue_json": str(DEPENDENCY.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"),
        "dataset": DATASET_ID, "validation_dataset": None, "results": [job],
        "active_job": None, "worker_pid": None, "worker_started": False}
    save(queue_dir / "queue.json", state)
    return queue_dir


def _training_resources() -> tuple[list[dict], list[int]]:
    processes = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models train" in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:350]})
            except ValueError:
                pass
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    dependency = ROOT / state["depends_on_queue_json"]
    while True:
        previous = json.loads(dependency.read_text(encoding="utf-8"))
        state.update(dependency_status=previous.get("status"), last_dependency_check_at_utc=now())
        if previous.get("status") in {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"} and previous.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = previous["finished_at_utc"]
            break
        state.update(status="waiting_for_previous_queue", updated_at_utc=now())
        save(queue_path, state)
        time.sleep(20)
    env = {**os.environ, **state["results"][0]["environment"]}
    for job in state["results"]:
        while True:
            processes, gpu = _training_resources()
            state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                         gpu_compute_pids=gpu, updated_at_utc=now())
            save(queue_path, state)
            if not processes and not gpu:
                break
            time.sleep(20)
        log_path = queue_dir / job["log"]
        job.update(status="running", started_at_utc=now(), pid=os.getpid())
        state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
        save(queue_path, state)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            proc = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, check=False)
        job.update(status="completed" if proc.returncode == 0 else "failed",
                   return_code=proc.returncode, finished_at_utc=now())
        state.update(status=job["status"], active_job=None, updated_at_utc=now())
        if proc.returncode:
            break
    state.update(finished_at_utc=now(), worker_pid=os.getpid(), worker_started=True)
    save(queue_path, state)
    return 0 if all(job.get("status") == "completed" for job in state["results"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="write queue JSON only; does not start a worker")
    parser.add_argument("--worker", type=Path, help="explicitly run the queue after dependencies finish")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.create:
        parser.error("use --create to register without starting, or --worker QUEUE_DIR to run")
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_previous_queue", "worker_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
