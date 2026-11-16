"""Queue Granite 350M V2.14 preflight and training after H-Micro."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
CONFIG = ROOT / "configs/models/granite4_350m_v214.yaml"
DEPENDENCY = ROOT / "artifacts/training_queues/granite4_h_micro_v214_20261113T203729Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def active_resources() -> tuple[list[dict], list[int]]:
    processes = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models" in parts[1] and Path(__file__).name not in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                pass
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=20)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def execute(command: list[str], log, state_path: Path, state: dict, phase: str) -> int:
    state.update(status=phase, updated_at_utc=now())
    save(state_path, state)
    log.write(f"\n[{now()}] Starting {phase}: {' '.join(command)}\n")
    log.flush()
    process = subprocess.Popen(command, cwd=ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state[f"{phase}_pid"] = process.pid
    save(state_path, state)
    return process.wait()


def worker(queue_dir: Path) -> int:
    state_path = queue_dir / "queue.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = state["results"][0]
    while True:
        dep = json.loads(DEPENDENCY.read_text(encoding="utf-8"))
        state.update(dependency_status=dep.get("status"), last_dependency_check_at_utc=now())
        if dep.get("status") in TERMINAL and dep.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = dep["finished_at_utc"]
            break
        state["status"] = "waiting_for_granite4_h_micro"
        state["updated_at_utc"] = now()
        save(state_path, state)
        time.sleep(20)
    while True:
        processes, gpu = active_resources()
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(state_path, state)
        if not processes and not gpu:
            break
        time.sleep(20)
    started = time.monotonic()
    with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
        preflight = execute(job["preflight_command"], log, state_path, state, "preflight_running")
        state["preflight_exit_code"] = preflight
        if preflight:
            job.update(status="failed", exit_code=preflight, failed_phase="preflight", finished_at_utc=now())
            state.update(status="failed", active_job=None, exit_code=preflight,
                         finished_at_utc=now(), updated_at_utc=now())
            save(state_path, state)
            return preflight
        state["preflight_report"] = job["preflight_report"]
        job.update(status="running", started_at_utc=now())
        state.update(status="running", active_job=job["job_id"], updated_at_utc=now())
        save(state_path, state)
        code = execute(job["command"], log, state_path, state, "training_running")
    job.update(status="completed" if code == 0 else "failed", exit_code=code,
                elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
    state.update(status=job["status"], active_job=None, exit_code=code,
                 finished_at_utc=now(), updated_at_utc=now())
    save(state_path, state)
    return code


def create_queue() -> Path:
    if not CONFIG.is_file() or not DEPENDENCY.is_file():
        raise FileNotFoundError("Granite 350M config or H-Micro queue is missing")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config.get("model") != "granite4_350m" or int(config.get("epochs", 0)) != 4:
        raise ValueError("Granite 350M config model/epochs mismatch")
    if int(config.get("max_length", 0)) != 512 or len(config.get("lora_target_modules", [])) != 6:
        raise ValueError("Granite 350M config must specify 512 tokens and six LoRA target types")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"granite4_350m_v214_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    preflight_report = queue_dir / "granite_preflight.json"
    preflight_command = [PYTHON, "-u", "scripts/preflight_granite4_350m.py",
                         "--config", str(CONFIG), "--output", str(preflight_report)]
    train_command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
        "--model", "granite4_350m", "--dataset", "v2.14", "--validation-dataset", "v0",
        "--config", str(CONFIG), "--max-length", "512", "--seed", "2026",
        "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    job = {"index": 1, "job_id": f"granite4_350m_v2.14_maxlen512_{stamp}",
        "model": "granite4_350m", "dataset": "v2.14", "dataset_rows": 231336,
        "validation_dataset": "v0", "validation_rows": 2699, "seed": 2026,
        "training_mode": "lora", "epochs": 4, "early_stopping_patience": 2,
        "max_length": 512, "checkpoint_selection_metric": "balanced_accuracy",
        "config": str(CONFIG.relative_to(ROOT)), "preflight_command": preflight_command,
        "preflight_report": str(preflight_report.relative_to(ROOT)), "command": train_command,
        "status": "queued", "log": "01_granite4_350m_v214.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": "granite4_350m_v214", "status": "queued_waiting_for_granite4_h_micro",
        "created_at_utc": now(), "description": "Granite 4.0 350M Base V2.14 four-epoch LoRA run with 512-token input and full V0 balanced-accuracy selection.",
        "execution_policy": "wait for Granite H-Micro queue, then wait for training processes and GPU compute to be idle; preflight before training",
        "depends_on_queue_json": str(DEPENDENCY.relative_to(ROOT)), "dataset": "v2.14",
        "dataset_path": "data/processed/dataset_v2.14.parquet", "dataset_rows": 231336,
        "validation_dataset": "v0", "validation_rows": 2699, "max_length": 512,
        "results": [job], "active_job": None}
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["worker_pid"] = proc.pid
    save(queue_path, state)
    return queue_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker.resolve()))
    print(json.dumps({"queue_dir": str(create_queue())}, indent=2))
