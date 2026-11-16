"""Queue a from-scratch UltraBERT run on V2.14 prediction-filtered 80%."""
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
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DATASET = "v2.14-prediction-filtered-80pct"
DATA_PATH = ROOT / "data/processed/dataset_v2.14-prediction-filtered-80pct.parquet"
PROFILE_PATH = ROOT / "data/processed/dataset_profile_v2.14-prediction-filtered-80pct.json"
DEPENDENCIES = [
    ROOT / "artifacts/training_queues/v2_14_frozen_style_20261113T113511Z/queue.json",
    ROOT / "artifacts/training_queues/ultrabert_v2.15_extrato_14_continue_20261113T123914Z/queue.json",
    ROOT / "artifacts/training_queues/ultrabert_v2.15_extrato_14_continue_20261113T123942Z/queue.json",
]
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def active_resources() -> tuple[list[dict], list[int]]:
    processes: list[dict] = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        line = line.strip()
        if "src.pipeline models" in line and "queue_ultrabert_v214pf80_scratch.py" not in line:
            try:
                pid, command = line.split(None, 1)
                processes.append({"pid": int(pid), "command": command[:400]})
            except ValueError:
                continue
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
    dependencies = [Path(p) if Path(p).is_absolute() else ROOT / p
                    for p in state["depends_on_queue_jsons"]]

    while True:
        pending = []
        dependency_states = {}
        for path in dependencies:
            if not path.is_file():
                state.update(status="blocked", error=f"dependency queue missing: {path}", updated_at_utc=now())
                save(state_path, state)
                return 1
            dep = json.loads(path.read_text(encoding="utf-8"))
            dependency_states[str(path.relative_to(ROOT))] = dep.get("status")
            if dep.get("status") not in TERMINAL or not dep.get("finished_at_utc"):
                pending.append(str(path.relative_to(ROOT)))
        state.update(dependency_statuses=dependency_states, pending_dependencies=pending,
                     last_dependency_check_at_utc=now(), updated_at_utc=now())
        if not pending:
            break
        state["status"] = "waiting_for_previous_queues"
        save(state_path, state)
        time.sleep(20)

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        try:
            processes, gpu_pids = active_resources()
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
            proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = proc.pid
            save(state_path, state)
            code = proc.wait()
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
    for path in [DATA_PATH, PROFILE_PATH, ROOT / "data/processed/dataset_v0.parquet", *DEPENDENCIES]:
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    data = pd.read_parquet(DATA_PATH, columns=["id", "model"])
    v0 = pd.read_parquet(ROOT / "data/processed/dataset_v0.parquet", columns=["id", "model"])
    if len(data) != 186029 or len(data) != int(profile.get("rows", -1)):
        raise ValueError(f"unexpected {DATASET} row count: {len(data)}")
    if not profile.get("training_only") or not profile.get("external_source_rows_only"):
        raise ValueError("filtered dataset profile must designate external-only training data")
    if set(data.model.astype(str)) != set(v0.model.astype(str)):
        raise ValueError("training and V0 class rosters differ")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_v214pf80_scratch_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    # Match the successful V2.14 prediction-filtered from-scratch recipe.
    # max_length is intentionally left to the training factory's full-data
    # token count for this dataset and the complete V0 validation set.
    config = {
        "model": "ultrabert380",
        "experiment_name": f"ultrabert380_{DATASET}_from_scratch_{stamp}",
        "checkpoint": "dmis-lab/UltraBERT-380M",
        "revision": "671a38c087233c0cc72c5bfceea6fffa2822f1db",
        "epochs": 40,
        "early_stopping_patience": 2,
        "learning_rate": 8.0e-6,
        "learning_rate_decay": None,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.0,
        "optimizer": "adamw_torch",
        "weight_decay": 0.0,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "target_effective_batch_size": None,
        "inference_batch_size": 8,
        "auto_batch_size": True,
        "max_batch_size": 1024,
        "batch_probe_safety_fraction": 0.95,
        "gradient_checkpointing": True,
        "local_files_only": True,
        "checkpoint_selection_metric": "balanced_accuracy",
        "calibration_enabled": True,
    }
    config_path = config_dir / "ultrabert380_v214pf80_scratch.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "ultrabert380", "--dataset", DATASET,
               "--validation-dataset", "v0", "--config", str(config_path),
               "--seed", "2026", "--accept-config-defaults"]
    dataset_sha = sha256(DATA_PATH)
    job = {
        "index": 1, "job_id": config["experiment_name"], "model": "ultrabert380",
        "dataset": DATASET, "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "dataset_rows": len(data), "dataset_sha256": dataset_sha,
        "validation_dataset": "v0", "validation_rows": len(v0),
        "training_mode": "from_scratch", "initial_checkpoint": config["checkpoint"],
        "epochs": config["epochs"], "early_stopping_patience": config["early_stopping_patience"],
        "learning_rate": config["learning_rate"], "scheduler": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"], "optimizer": config["optimizer"],
        "weight_decay": config["weight_decay"], "requested_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "auto_batch_size": config["auto_batch_size"], "max_batch_size": config["max_batch_size"],
        "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
        "checkpoint_selection_metric": config["checkpoint_selection_metric"],
        "config": str(config_path.relative_to(ROOT)), "command": command,
        "status": "queued", "log": "01_ultrabert380_v214pf80_scratch.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "ultrabert_v214pf80_scratch",
        "status": "queued_waiting_for_previous_queues", "created_at_utc": now(),
        "description": "From-scratch UltraBERT comparison on v2.14-prediction-filtered-80pct using the V2.14 prediction-filtered run recipe.",
        "execution_policy": "wait for the active V2.14 queue and both queued V2.15 continuation requests to finish, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_jsons": [str(p.relative_to(ROOT)) for p in DEPENDENCIES],
        "dataset": DATASET, "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "dataset_rows": len(data), "dataset_sha256": dataset_sha,
        "validation_dataset": "v0", "validation_rows": len(v0),
        "source_comparison_run": "ultrabert380_v2.14-prediction-filtered_valv0_data6166df4b_split572d8c56_seed2026_cfg0afcd063_code9c2e334",
        "source_comparison_config": "artifacts/predictions/ultrabert380_v2.14-prediction-filtered_valv0_data6166df4b_split572d8c56_seed2026_cfg0afcd063_code9c2e334_manifest.json",
        "results": [job], "active_job": None,
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    state["worker_pid"] = proc.pid
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
                      "status": "queued_waiting_for_previous_queues"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
