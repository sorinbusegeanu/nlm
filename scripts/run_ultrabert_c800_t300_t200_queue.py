"""Queue the combined-view UltraBERT run behind the active local training job."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
SOURCE_CONFIG = ROOT / "configs/models/ultrabert380_v2_14_scratch.yaml"
BLOCKER_PID = 3119030
BLOCKER_MATCH = "--dataset v2.14 --validation-dataset v0 --config configs/models/ultrabert380_v2_14_scratch.yaml"
DATASET = "v2_0_c800_t300_t200"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_is_blocking(pid: int, expected: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return expected in command


def gpu_compute_pids() -> list[int] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pids = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def worker(queue_dir: Path) -> int:
    state_path = queue_dir / "queue.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = state["results"][0]
    state["status"] = "waiting_for_current_training"
    save(state_path, state)
    while process_is_blocking(int(state["blocker_pid"]), state["blocker_command_match"]):
        time.sleep(30)
        state["last_blocker_check_at_utc"] = now()
        save(state_path, state)
    state["status"] = "waiting_for_gpu_release"
    save(state_path, state)
    while True:
        compute_pids = gpu_compute_pids()
        if compute_pids == []:
            break
        state["last_gpu_check_at_utc"] = now()
        state["gpu_compute_pids"] = compute_pids
        save(state_path, state)
        time.sleep(30)

    job["status"] = "running"
    job["started_at_utc"] = now()
    state["status"] = "running"
    state["active_job"] = job["job_id"]
    save(state_path, state)
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    log_path = queue_dir / job["log"]
    started = time.monotonic()
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            save(state_path, state)
            exit_code = process.wait()
    except Exception as exc:
        exit_code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"
    job.update({"status": "completed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "finished_at_utc": now()})
    state.update({"status": job["status"], "active_job": None,
                  "finished_at_utc": now(), "exit_code": exit_code})
    save(state_path, state)
    return exit_code


def create_queue(blocker_pid: int) -> Path:
    data_profile = ROOT / "data/processed/dataset_profile_v2_0_c800_t300_t200.json"
    profile = json.loads(data_profile.read_text(encoding="utf-8"))
    if not process_is_blocking(blocker_pid, BLOCKER_MATCH):
        raise RuntimeError(f"expected active UltraBERT training process {blocker_pid} was not found")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_{DATASET}_{stamp}"
    queue_dir = ROOT / "artifacts/training_queues" / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config["experiment_name"] = f"ultrabert380_{DATASET}_from_scratch_{stamp}"
    config_path = config_dir / f"ultrabert380_{DATASET}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "ultrabert380", "--dataset", DATASET,
               "--validation-dataset", "v0", "--config", str(config_path),
               "--accept-config-defaults"]
    job = {
        "index": 1, "job_id": config["experiment_name"], "model": "ultrabert380",
        "dataset": DATASET, "validation_dataset": "v0", "status": "queued",
        "training_mode": "from_scratch", "initial_checkpoint": config["checkpoint"],
        "epochs": config["epochs"], "early_stopping_patience": config["early_stopping_patience"],
        "learning_rate": config["learning_rate"], "learning_rate_decay": config["learning_rate_decay"],
        "lr_scheduler_type": config["lr_scheduler_type"], "warmup_ratio": config["warmup_ratio"],
        "optimizer": config["optimizer"], "weight_decay": config["weight_decay"],
        "requested_batch_size": config["batch_size"],
        "auto_batch_size": config["auto_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "target_effective_batch_size": config.get("target_effective_batch_size"),
        "max_batch_size": config["max_batch_size"],
        "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "config": str(config_path.relative_to(ROOT)), "command": command,
        "log": "01_ultrabert380_combined.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "status": "queued_waiting_for_current_training",
        "created_at_utc": now(), "description": "One from-scratch UltraBERT experiment on the concatenated c800, t300, and t200 V2.0 views.",
        "execution_policy": "wait for active training PID and all GPU compute processes to exit, then run one job",
        "blocker_pid": blocker_pid, "blocker_command_match": BLOCKER_MATCH,
        "dataset": DATASET, "dataset_rows": profile["rows"],
        "dataset_sha256": profile["dataset_sha256"], "source_datasets": profile["source_datasets"],
        "validation_dataset": "v0", "results": [job], "active_job": None,
    }
    save(queue_dir / "queue.json", state)
    worker_log = (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=worker_log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    worker_log.close()
    state["worker_pid"] = process.pid
    save(queue_dir / "queue.json", state)
    return queue_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--blocker-pid", type=int, default=BLOCKER_PID)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue(args.blocker_pid)
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_current_training"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
