"""Run the V2.14 Qwen3-0.6B LoRA job after the current V2.14 queue."""
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
PREVIOUS_QUEUE = ROOT / "artifacts/training_queues/v2_14_classical_pf_20261112T145950Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
CONFIG_PATH = ROOT / "configs/models/qwen06_v2_14_lora.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    pids = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def active_training_processes() -> list[dict]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True,
                            capture_output=True, text=True, timeout=20)
    active = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1]
        if "src.pipeline models train" in command and "run_qwen06_v2_14_lora_queue.py" not in command:
            active.append({"pid": pid, "command": command[:400]})
    return active


def wait_for_previous(state_path: Path, state: dict) -> bool:
    if not PREVIOUS_QUEUE.is_file():
        state.update({"status": "blocked", "error": f"queue dependency missing: {PREVIOUS_QUEUE}",
                      "updated_at_utc": now()})
        save(state_path, state)
        return False
    terminal = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}
    while True:
        try:
            previous = json.loads(PREVIOUS_QUEUE.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            state.update({"status": "blocked", "error": f"could not read queue dependency: {exc}",
                          "updated_at_utc": now()})
            save(state_path, state)
            return False
        previous_status = previous.get("status")
        state.update({"previous_queue_status": previous_status,
                      "last_dependency_check_at_utc": now()})
        if previous_status in terminal:
            state["previous_queue_finished_at_utc"] = previous.get("finished_at_utc")
            state["previous_queue_id"] = previous.get("queue_id")
            save(state_path, state)
            break
        state["status"] = "waiting_for_previous_queue"
        save(state_path, state)
        time.sleep(30)

    while True:
        try:
            processes = active_training_processes()
            gpu_pids = gpu_compute_pids()
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            state.update({"status": "blocked", "error": f"resource check failed: {exc}",
                          "updated_at_utc": now()})
            save(state_path, state)
            return False
        state.update({"active_training_processes": processes,
                      "gpu_compute_pids": gpu_pids,
                      "last_resource_check_at_utc": now()})
        if not processes and not gpu_pids:
            return True
        state["status"] = "waiting_for_gpu_release"
        save(state_path, state)
        time.sleep(20)


def worker(queue_dir: Path) -> int:
    state_path = queue_dir / "queue.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not wait_for_previous(state_path, state):
        return 1
    job = state["jobs"][0]
    job.update({"status": "running", "started_at_utc": now()})
    state.update({"status": "running", "active_job": job["job_id"], "updated_at_utc": now()})
    save(state_path, state)

    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            job["pid"] = process.pid
            save(state_path, state)
            exit_code = process.wait()
    except Exception as exc:
        exit_code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"

    job.update({"status": "completed" if exit_code == 0 else "failed",
                "exit_code": int(exit_code), "elapsed_seconds": round(time.monotonic() - started, 2),
                "finished_at_utc": now()})
    state.update({"status": job["status"], "active_job": None,
                  "exit_code": int(exit_code), "finished_at_utc": now(), "updated_at_utc": now()})
    save(state_path, state)
    return int(exit_code)


def create_queue() -> Path:
    if not PREVIOUS_QUEUE.is_file() or not CONFIG_PATH.is_file():
        raise FileNotFoundError("the current V2.14 queue or Qwen config is missing")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if config.get("model") != "qwen06" or config.get("quantization") != "none":
        raise ValueError("Qwen06 V2.14 queue config does not match the requested unquantized LoRA setup")
    for manifest_path in (ROOT / "artifacts/predictions").glob("*_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (manifest.get("model") == "qwen06"
                and (manifest.get("training_dataset") or manifest.get("dataset")) == "v2.14"
                and manifest.get("config", {}).get("experiment_name") == config.get("experiment_name")):
            raise RuntimeError(f"matching completed run already exists: {manifest_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"qwen06_v2_14_lora_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "qwen06", "--dataset", "v2.14", "--validation-dataset", "v0",
               "--config", str(CONFIG_PATH), "--seed", "2026", "--accept-config-defaults"]
    job = {"index": 1, "job_id": config["experiment_name"], "model": "qwen06",
           "dataset": "v2.14", "validation_dataset": "v0", "status": "queued",
           "training_mode": "lora", "quantization": "none",
           "config": str(CONFIG_PATH.relative_to(ROOT)), "config_values": config,
           "command": command, "log": "01_qwen06_v2_14_lora.log", "seed": 2026,
           "queued_at_utc": now()}
    state = {"queue_id": queue_id, "status": "queued_waiting_for_previous_queue",
             "created_at_utc": now(), "dataset": "v2.14", "validation_dataset": "v0",
             "validation_rows": 2699, "execution_policy": "wait for current V2.14 queue to finish, then wait for training/GPU release",
             "depends_on_queue_json": str(PREVIOUS_QUEUE.relative_to(ROOT)),
             "jobs": [job], "active_job": None}
    state_path = queue_dir / "queue.json"
    save(state_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()),
                                    "--worker", str(queue_dir)], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    state["worker_pid"] = process.pid
    save(state_path, state)
    return queue_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_previous_queue", "job": "qwen06 v2.14 LoRA"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
