"""Queue three more Qwen06 training epochs after the active UltraBERT run."""
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
SOURCE = "qwen06_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgba46767d_code81bce4a"
DEPENDENCY = ROOT / "artifacts/training_queues/ultrabert_v214_v215_scratch_20261113T201859Z/queue.json"
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


def worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    job = state["results"][0]
    while True:
        dep = json.loads(DEPENDENCY.read_text(encoding="utf-8"))
        state.update(dependency_status=dep.get("status"), last_dependency_check_at_utc=now())
        if dep.get("status") in TERMINAL and dep.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = dep["finished_at_utc"]
            break
        state["status"] = "waiting_for_current_training"
        state["updated_at_utc"] = now()
        save(queue_path, state)
        time.sleep(20)
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        processes, gpu = active_resources()
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_path, state)
        if not processes and not gpu:
            break
        time.sleep(20)
    job.update(status="running", started_at_utc=now())
    state.update(status="running", active_job=job["job_id"], updated_at_utc=now())
    save(queue_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            proc = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = proc.pid
            save(queue_path, state)
            code = proc.wait()
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
    manifest_path = ROOT / f"artifacts/predictions/{SOURCE}_manifest.json"
    if not manifest_path.is_file() or not DEPENDENCY.is_file():
        raise FileNotFoundError("Qwen source manifest or current training queue is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != "qwen06" or manifest.get("training_dataset") != "v2.14":
        raise ValueError("latest Qwen06 source manifest does not match V2.14")
    checkpoint = Path(manifest.get("selected_checkpoint_path") or manifest.get("best_checkpoint_path"))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"selected Qwen06 checkpoint missing: {checkpoint}")
    config = {"model": "qwen06", "epochs": 3,
              "experiment_name": f"qwen06_v2.14_continue3_from_{SOURCE[-12:]}",
              "local_files_only": True}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"qwen06_v214_continue3_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config_path = config_dir / "qwen06_continue3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "continue-qwen", "--model", "qwen06",
               "--dataset", "v2.14", "--validation-dataset", "v0", "--from-experiment", SOURCE,
               "--epochs", "3", "--seed", "2026", "--config", str(config_path),
               "--refresh-prediction", "--refresh-checkpoint-catalog", "--accept-config-defaults"]
    job = {"index": 1, "job_id": config["experiment_name"], "model": "qwen06", "dataset": "v2.14",
           "validation_dataset": "v0", "training_mode": "continue_lora", "additional_epochs": 3,
           "source_experiment": SOURCE, "source_checkpoint": str(checkpoint.relative_to(ROOT)),
           "source_selected_epoch": manifest.get("selected_epoch"), "seed": 2026,
           "config": str(config_path.relative_to(ROOT)), "command": command, "status": "queued",
           "log": "01_qwen06_continue3.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": "qwen06_v214_continue3", "status": "queued_waiting_for_current_training",
             "created_at_utc": now(), "description": "Continue the latest V2.14 Qwen06 selected checkpoint for three more epochs after current training completes.",
             "execution_policy": "wait for the active UltraBERT training queue, then wait for training processes and GPU compute to be idle",
             "depends_on_queue_json": str(DEPENDENCY.relative_to(ROOT)), "source_experiment": SOURCE,
             "source_checkpoint": str(checkpoint.relative_to(ROOT)), "source_selected_epoch": manifest.get("selected_epoch"),
             "dataset": "v2.14", "validation_dataset": "v0", "validation_rows": manifest.get("validation_rows"),
             "results": [job], "active_job": None}
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
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
