"""Queue SVC then CatBoost P+F on full V2.14, preserving any active run."""
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

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
SVC_CONFIG = ROOT / "configs/models/svc.yaml"
CATBOOST_CONFIG = ROOT / "configs/models/catboost_stylometry_v2_14_pf.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
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
    worker_pid = os.getpid()
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        command = fields[1]
        if pid == worker_pid:
            continue
        if "src.pipeline models train" in command and "queue_v2_14_classical_models.py" not in command:
            active.append({"pid": pid, "command": command[:400]})
    return active


def wait_for_training_release(state_path: Path, state: dict) -> bool:
    while True:
        processes = active_training_processes()
        try:
            gpu_pids = gpu_compute_pids()
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            state.update({"status": "blocked", "error": f"GPU check failed: {exc}",
                          "updated_at_utc": now()})
            save(state_path, state)
            return False
        state.update({"active_training_processes": processes,
                      "gpu_compute_pids": gpu_pids,
                      "last_resource_check_at_utc": now()})
        if not processes and not gpu_pids:
            return True
        state["status"] = "waiting_for_current_training"
        save(state_path, state)
        time.sleep(20)


def worker(queue_dir: Path, start_index: int = 0) -> int:
    state_path = queue_dir / "queue.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous_failure_count = int(state.get("failure_count", 0))
    if not wait_for_training_release(state_path, state):
        return 1

    environment = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8"}
    failures = 0
    for index, job in enumerate(state["jobs"][start_index:], start=start_index):
        if index and not wait_for_training_release(state_path, state):
            job.update({"status": "blocked", "error": state.get("error"), "finished_at_utc": now()})
            for later in state["jobs"][index + 1:]:
                later.update({"status": "blocked", "reason": "resource check failed", "finished_at_utc": now()})
            break

        job.update({"status": "running", "started_at_utc": now()})
        state.update({"status": "running", "active_job": job["job_id"], "updated_at_utc": now()})
        save(state_path, state)
        log_path = queue_dir / job["log"]
        started = time.monotonic()
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
                process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                job["pid"] = process.pid
                save(state_path, state)
                code = process.wait()
        except Exception as exc:
            code = 1
            job["error"] = f"{type(exc).__name__}: {exc}"

        job.update({"status": "completed" if code == 0 else "failed",
                    "exit_code": int(code), "elapsed_seconds": round(time.monotonic() - started, 2),
                    "finished_at_utc": now()})
        failures += int(code != 0)
        state["active_job"] = None
        state["updated_at_utc"] = now()
        save(state_path, state)

    failures += previous_failure_count
    state.update({"status": "completed_with_failures" if failures else "completed",
                  "failure_count": failures, "active_job": None,
                  "finished_at_utc": now(), "updated_at_utc": now()})
    save(state_path, state)
    return 0 if failures == 0 else 1


def create_queue() -> Path:
    if not SVC_CONFIG.is_file() or not CATBOOST_CONFIG.is_file():
        raise FileNotFoundError("one or more requested model configurations are missing")

    requested_configs = {
        "svc": yaml.safe_load(SVC_CONFIG.read_text(encoding="utf-8")),
        "catboost_stylometry": yaml.safe_load(CATBOOST_CONFIG.read_text(encoding="utf-8")),
    }
    requested_hashes = {
        model: hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        for model, config in requested_configs.items()
    }
    # Refuse an identical pending queue entry or an exact completed rerun.
    for queue_path in QUEUE_ROOT.rglob("queue.json"):
        try:
            old = json.loads(queue_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for job in old.get("jobs", old.get("results", [])):
            model = job.get("model")
            same_job = (job.get("dataset") == "v2.14" and model in requested_configs
                        and job.get("config_sha256") == requested_hashes[model])
            if same_job and job.get("status") in {"queued", "running", "completed"}:
                raise RuntimeError(f"matching queue job already exists: {queue_path}")

    for manifest_path in (ROOT / "artifacts/predictions").glob("*_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model = manifest.get("model")
        dataset = manifest.get("training_dataset") or manifest.get("dataset")
        config_hash = manifest.get("config_sha256")
        if (dataset == "v2.14" and model in requested_hashes
                and config_hash == requested_hashes[model]):
            raise RuntimeError(f"matching completed experiment already exists: {manifest_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"v2_14_classical_pf_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    commands = [
        ("svc", SVC_CONFIG, "01_svc_v2_14.log"),
        ("catboost_stylometry", CATBOOST_CONFIG, "02_catboost_stylometry_v2_14_pf.log"),
    ]
    jobs = []
    for index, (model, config, log) in enumerate(commands, start=1):
        command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                   "--model", model, "--dataset", "v2.14", "--validation-dataset", "v0",
                   "--config", str(config), "--seed", "2026", "--accept-config-defaults"]
        jobs.append({"index": index, "job_id": f"{queue_id}_{model}", "queue_tag": "v2_14_classical_pf",
                     "model": model, "dataset": "v2.14", "validation_dataset": "v0",
                     "status": "queued", "config": str(config.relative_to(ROOT)),
                     "config_sha256": requested_hashes[model],
                     "command": command, "log": log, "queued_at_utc": now()})

    state = {"queue_id": queue_id, "queue_tag": "v2_14_classical_pf", "status": "queued",
             "created_at_utc": now(), "dataset": "v2.14", "dataset_rows": 231336,
             "validation_dataset": "v0", "validation_rows": 2699,
             "execution_policy": "sequential; wait for active training/GPU use; continue to job 2 if job 1 fails",
             "jobs": jobs, "active_job": None}
    state_path = queue_dir / "queue.json"
    save(state_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    state["worker_pid"] = process.pid
    save(state_path, state)
    return queue_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve(), start_index=args.start_index)
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued", "jobs": ["svc", "catboost_stylometry"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
