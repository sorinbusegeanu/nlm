#!/usr/bin/env python3
"""Wait for V3.1 completion, then run the three queued E41 LoRA experiments."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/ensemble_catalog_v2.json"
V31_STATE = ROOT / "artifacts/analysis/v3.1/contributor_factory/factory_state.json"
V31_RANKING = ROOT / "artifacts/analysis/v3.1/contributor_factory/contributor_ranking.csv"
PYTHON = Path("/home/zodrak/cuda13.x/bin/python")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def v31_complete():
    try:
        catalog = json.loads(CATALOG.read_text())
        entry = next(x for x in catalog["experiments"] if x.get("id") == "V3.1-CONTRIBUTOR-FACTORY")
        state = json.loads(V31_STATE.read_text())
        import pandas as pd
        total = len(pd.read_csv(V31_RANKING))
        return (entry.get("status") == "Complete" and not state.get("in_progress")
                and int(state.get("next_rank", 0)) > total), {
                    "catalog_status": entry.get("status"),
                    "next_rank": state.get("next_rank"), "queue_ranks": total,
                    "in_progress": state.get("in_progress")}
    except Exception as exc:
        return False, {"reason": str(exc)}


def gpu_idle():
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, "nvidia-smi unavailable; holding queue"
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return not rows, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = ROOT / queue_path
    queue = json.loads(queue_path.read_text())
    queue["worker_pid"] = os.getpid()
    queue["worker_started_at"] = now()
    queue["status"] = "queued_waiting_for_v3.1"
    save(queue_path, queue)
    while True:
        ready, detail = v31_complete()
        queue["dependency_check"] = detail
        queue["last_dependency_check_at"] = now()
        if ready:
            idle, gpu_detail = gpu_idle()
            queue["gpu_check"] = gpu_detail
            if idle:
                break
        queue["status"] = "queued_waiting_for_v3.1" if not ready else "queued_waiting_for_gpu"
        save(queue_path, queue)
        time.sleep(max(10, args.poll_seconds))

    queue["status"] = "running"
    queue["started_at"] = now()
    save(queue_path, queue)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if not PYTHON.is_file():
        queue["status"] = "failed"
        queue["failure"] = f"required project Python environment missing: {PYTHON}"
        save(queue_path, queue)
        raise SystemExit(queue["failure"])
    for job in queue["jobs"]:
        job["status"] = "running"
        job["started_at"] = now()
        save(queue_path, queue)
        log_path = ROOT / job["log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", "-m", "src.pipeline", "models", "train",
                   "--model", "qwen35_08b", "--dataset", "v2.14", "--validation-dataset", "v0",
                   "--config", job["config"], "--seed", "2026", "--max-length", "562",
                   "--accept-config-defaults", "--refresh-checkpoint-catalog"]
        job["command"] = command
        with log_path.open("w") as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        job["finished_at"] = now()
        job["return_code"] = proc.returncode
        job["status"] = "complete" if proc.returncode == 0 else "failed"
        save(queue_path, queue)
        if proc.returncode:
            queue["status"] = "failed"
            queue["failure"] = f"{job['name']} exited with status {proc.returncode}"
            queue["finished_at"] = now()
            save(queue_path, queue)
            raise SystemExit(queue["failure"])
    queue["status"] = "evaluating"
    queue["evaluation_started_at"] = now()
    save(queue_path, queue)
    evaluation_log = ROOT / queue["evaluation_log"]
    with evaluation_log.open("w") as log:
        proc = subprocess.run([str(PYTHON), "-u", "scripts/evaluate_e41_qwen35_layer_removal.py",
                               "--queue", str(queue_path)], cwd=ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    queue["evaluation_return_code"] = proc.returncode
    queue["finished_at"] = now()
    queue["status"] = "complete" if proc.returncode == 0 else "evaluation_failed"
    save(queue_path, queue)
    if proc.returncode:
        raise SystemExit(f"E41 evaluation exited with status {proc.returncode}; see {evaluation_log}")

if __name__ == "__main__":
    main()
