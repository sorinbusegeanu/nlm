"""Durably queue a resume of the interrupted UltraBERT V2.14 training run."""
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
SOURCE_RUN = ROOT / (
    "artifacts/runs/ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_"
    "seed2026_cfg19c94ea0_code81bce4a"
)
SOURCE_OUTPUT = SOURCE_RUN / "checkpoints/ultrabert380_foldvalidation"
RESUME_CHECKPOINT = SOURCE_OUTPUT / "checkpoint-5142"
QUEUE_ROOT = ROOT / "artifacts/training_queues"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
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
    state["status"] = "waiting_for_gpu_release"
    save(state_path, state)
    while True:
        try:
            pids = gpu_compute_pids()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update({"status": "blocked", "error": f"GPU check failed: {exc}",
                          "updated_at_utc": now()})
            save(state_path, state)
            return 1
        state.update({"gpu_compute_pids": pids, "last_gpu_check_at_utc": now()})
        if not pids:
            break
        save(state_path, state)
        time.sleep(20)

    job.update({"status": "running", "started_at_utc": now()})
    state.update({"status": "running", "active_job": job["job_id"]})
    save(state_path, state)
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
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
            exit_code = process.wait()
    except Exception as exc:
        exit_code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"
    job.update({"status": "completed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "finished_at_utc": now()})
    state.update({"status": job["status"], "active_job": None,
                  "exit_code": exit_code, "finished_at_utc": now()})
    save(state_path, state)
    return exit_code


def create_queue() -> Path:
    if not RESUME_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"consistent resume checkpoint missing: {RESUME_CHECKPOINT}")
    saved_state = json.loads((RESUME_CHECKPOINT / "trainer_state.json").read_text())
    if int(saved_state.get("global_step", -1)) != 5142 or float(saved_state.get("epoch", -1)) != 2.0:
        raise ValueError("resume checkpoint is not the saved end-of-epoch-2 state")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_v2_14_resume_ep2_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model": "ultrabert380",
        "experiment_name": f"ultrabert380_v2.14_resume_from_epoch2_{stamp}",
        "checkpoint": "dmis-lab/UltraBERT-380M",
        "revision": "671a38c087233c0cc72c5bfceea6fffa2822f1db",
        "trainer_resume_checkpoint": str(RESUME_CHECKPOINT),
        "epochs": 40,
        "early_stopping_patience": 4,
        "learning_rate": 3.0e-5,
        "learning_rate_decay": 0.9,
        "lr_scheduler_type": "epoch_decay",
        "warmup_ratio": 0.0,
        "optimizer": "adamw_torch",
        "weight_decay": 0.0,
        # Preserve the effective batch and optimizer schedule saved in checkpoint-5142.
        "batch_size": 90,
        "gradient_accumulation_steps": 1,
        "target_effective_batch_size": None,
        "inference_batch_size": 8,
        "auto_batch_size": False,
        "max_batch_size": 1024,
        "batch_probe_safety_fraction": 0.95,
        "gradient_checkpointing": True,
        "local_files_only": True,
        "checkpoint_selection_metric": "balanced_accuracy",
    }
    config_path = config_dir / "ultrabert380_v2_14_resume.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "ultrabert380", "--dataset", "v2.14",
               "--validation-dataset", "v0", "--config", str(config_path),
               "--accept-config-defaults"]
    job = {
        "index": 1,
        "job_id": config["experiment_name"],
        "model": "ultrabert380",
        "dataset": "v2.14",
        "validation_dataset": "v0",
        "status": "queued",
        "training_mode": "resume_from_checkpoint",
        "source_run": str(SOURCE_RUN.relative_to(ROOT)),
        "resume_checkpoint": str(RESUME_CHECKPOINT.relative_to(ROOT)),
        "resume_epoch": 2,
        "resume_global_step": 5142,
        "target_total_epochs": 40,
        "learning_rate": config["learning_rate"],
        "learning_rate_decay": config["learning_rate_decay"],
        "early_stopping_patience": config["early_stopping_patience"],
        "physical_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "config": str(config_path.relative_to(ROOT)),
        "command": command,
        "log": "01_ultrabert380_v2_14_resume.log",
        "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id,
        "status": "queued_waiting_for_gpu_release",
        "created_at_utc": now(),
        "description": "Resume the interrupted V2.14 UltraBERT run from its last complete checkpoint (epoch 2, step 5142).",
        "execution_policy": "detached worker waits until all GPU compute processes exit, then runs the checkpoint resume",
        "dataset": "v2.14",
        "validation_dataset": "v0",
        "validation_rows": 2699,
        "results": [job],
        "active_job": None,
    }
    state_path = queue_dir / "queue.json"
    save(state_path, state)
    worker_log = (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        [PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=worker_log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    worker_log.close()
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
    print(json.dumps({"queue_dir": str(queue_dir),
                      "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_gpu_release"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
