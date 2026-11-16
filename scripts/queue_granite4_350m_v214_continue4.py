"""Queue four additional Granite 350M LoRA epochs from experiment 0129."""
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

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
SOURCE_EXPERIMENT = "granite4_350m_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfg1f564168_code9c2e334"
ACTIVE_QUEUE = ROOT / "artifacts/training_queues/granite4_h_micro_v214_20261114T070347Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DATASET = "v2.14"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resource_state() -> tuple[list[dict], list[int]]:
    processes = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models" in parts[1] and Path(__file__).name not in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                pass
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    gpu_pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu_pids


def worker(queue_dir: Path) -> int:
    state_path = queue_dir / "queue.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = state["results"][0]
    dependency = Path(state["depends_on_queue_json"])
    if not dependency.is_absolute():
        dependency = ROOT / dependency
    while True:
        if not dependency.is_file():
            state.update(status="blocked", error=f"dependency queue missing: {dependency}", updated_at_utc=now())
            save(state_path, state)
            return 1
        previous = json.loads(dependency.read_text(encoding="utf-8"))
        state.update(dependency_status=previous.get("status"), last_dependency_check_at_utc=now())
        if previous.get("status") in TERMINAL and previous.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = previous["finished_at_utc"]
            break
        state["status"] = "waiting_for_dependency"
        state["updated_at_utc"] = now()
        save(state_path, state)
        time.sleep(20)

    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        try:
            processes, gpu_pids = resource_state()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(status="blocked", error=f"resource check failed: {exc}", updated_at_utc=now())
            save(state_path, state)
            return 1
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu_pids, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(state_path, state)
        if not processes and not gpu_pids:
            break
        time.sleep(20)

    job.update(status="running", started_at_utc=now())
    state.update(status="running", active_job=job["job_id"], updated_at_utc=now())
    save(state_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = process.pid
            save(state_path, state)
            code = process.wait()
    except Exception as exc:
        code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"
    job.update(status="completed" if code == 0 else "failed", exit_code=code,
                elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
    state.update(status=job["status"], active_job=None, exit_code=code,
                 finished_at_utc=now(), updated_at_utc=now())
    save(state_path, state)
    return code


def create_queue() -> Path:
    if not ACTIVE_QUEUE.is_file():
        raise FileNotFoundError(f"active Granite queue not found: {ACTIVE_QUEUE}")
    manifest_path = ROOT / f"artifacts/predictions/{SOURCE_EXPERIMENT}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("model") != "granite4_350m" or source.get("training_dataset") != DATASET:
        raise ValueError("experiment 0129 is not a Granite 350M V2.14 run")
    if source.get("validation_dataset") != "v0" or int(source.get("validation_rows", 0)) != 2699:
        raise ValueError("source run did not use all 2,699 V0 validation rows")
    if int(source.get("selected_epoch", 0)) != 4:
        raise ValueError(f"expected selected best epoch 4, found {source.get('selected_epoch')}")
    source_checkpoint_value = source.get("best_checkpoint_path") or source.get("selected_checkpoint_path")
    if not source_checkpoint_value:
        raise ValueError("source manifest has no selected checkpoint")
    source_checkpoint = Path(source_checkpoint_value)
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(source_checkpoint)
    for required in ("adapter_model.safetensors", "classifier.pt", "checkpoint_manifest.json"):
        if not (source_checkpoint / required).is_file():
            raise FileNotFoundError(f"selected Granite checkpoint missing {required}")

    from src.pipeline import _matching_continuation_source
    _matching_continuation_source("granite4_350m", DATASET, "v0", "data/processed", "artifacts",
                                  SOURCE_EXPERIMENT)
    dataset_path = ROOT / "data/processed/dataset_v2.14.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.14.json"
    v0_path = ROOT / "data/processed/dataset_v0.parquet"
    for path in (dataset_path, profile_path, v0_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    train = pd.read_parquet(dataset_path, columns=["model"])
    v0 = pd.read_parquet(v0_path, columns=["model"])
    if len(train) != 231336 or len(v0) != 2699 or set(train.model.astype(str)) != set(v0.model.astype(str)):
        raise ValueError("V2.14/V0 row counts or label rosters differ from source experiment 0129")
    if profile.get("dataset_sha256") != source.get("dataset_sha256"):
        raise ValueError("V2.14 content hash differs from the source checkpoint training data")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"granite4_350m_v214_continue4_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = dict(source["config"])
    config["epochs"] = 4
    # The request is four additional complete epochs; do not let the source
    # run's two-epoch early-stop patience shorten this continuation.
    config["early_stopping_patience"] = 4
    config["model"] = "granite4_350m"
    config_path = config_dir / "granite4_350m_v214_continue4.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "continue-lora",
               "--model", "granite4_350m", "--dataset", DATASET,
               "--validation-dataset", "v0", "--from-experiment", SOURCE_EXPERIMENT,
               "--epochs", "4", "--config", str(config_path), "--seed", "2026",
               "--accept-config-defaults"]
    dep = json.loads(ACTIVE_QUEUE.read_text(encoding="utf-8"))
    job = {
        "index": 1, "job_id": f"granite4_350m_v2.14_continue4_from_{SOURCE_EXPERIMENT}_{stamp}",
        "model": "granite4_350m", "dataset": DATASET, "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_rows": len(train), "dataset_sha256": profile["dataset_sha256"],
        "validation_dataset": "v0", "validation_rows": len(v0),
        "training_mode": "continue_from_selected_lora_checkpoint", "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)), "source_selected_epoch": 4,
        "source_validation_balanced_accuracy": source["metrics"]["balanced_accuracy"],
        "epochs": 4, "early_stopping_patience": 4,
        "learning_rate": config["learning_rate"], "max_length": config["max_length"],
        "checkpoint_selection_metric": "balanced_accuracy", "status": "queued",
        "config": str(config_path.relative_to(ROOT)), "command": command,
        "log": "01_granite4_350m_v214_continue4.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "granite4_350m_v214_continue4",
        "status": "queued_waiting_for_dependency", "created_at_utc": now(),
        "description": "Continue experiment 0129's selected Granite 350M V2.14 LoRA checkpoint for four additional epochs.",
        "execution_policy": "wait for the active Granite H-Micro V2.14 queue, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(ACTIVE_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dep.get("queue_id"), "dataset": DATASET,
        "dataset_path": str(dataset_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": "v0",
        "validation_rows": len(v0), "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_selected_epoch": 4,
        "source_validation_balanced_accuracy": source["metrics"]["balanced_accuracy"],
        "results": [job], "active_job": None,
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    state["worker_pid"] = process.pid
    save(queue_path, state)
    return queue_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_dependency"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
