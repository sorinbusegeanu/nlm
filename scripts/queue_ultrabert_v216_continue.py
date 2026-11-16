"""Queue a three-epoch UltraBERT continuation on V2.16 after current training work."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
DATASET = "v2.16_14plus15"
SOURCE_EXPERIMENT = (
    "ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_"
    "cfgb9e735ff_code81bce4a"
)
DEPENDENCY_QUEUE = ROOT / (
    "artifacts/training_queues/granite4_350m_v214_continue4_20261114T092723Z/queue.json"
)
QUEUE_ROOT = ROOT / "artifacts/training_queues"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_record() -> tuple[dict, Path]:
    manifest_path = ROOT / f"artifacts/predictions/{SOURCE_EXPERIMENT}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("model") != "ultrabert380" or source.get("training_dataset") != "v2.14":
        raise ValueError("source must be the selected UltraBERT checkpoint trained on V2.14")
    if source.get("validation_dataset") != "v0" or int(source.get("validation_rows", 0)) != 2699:
        raise ValueError("source checkpoint must have been selected using all 2,699 V0 rows")
    if float(source.get("metrics", {}).get("balanced_accuracy", -1)) < 0.5623:
        raise ValueError("the selected V2.14 checkpoint's recorded balanced accuracy is unexpected")
    raw = source.get("selected_checkpoint_path") or source.get("best_checkpoint_path")
    if not raw:
        raise ValueError("source manifest has no selected checkpoint")
    checkpoint = Path(raw)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"selected source checkpoint is missing: {checkpoint}")
    nested = checkpoint / "best_checkpoint"
    if nested.is_dir():
        checkpoint = nested
    return source, checkpoint


def resource_state() -> tuple[list[dict], list[int]]:
    ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    processes = []
    this_script = Path(__file__).name
    for line in ps.splitlines():
        line = line.strip()
        if "src.pipeline models" in line and this_script not in line:
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
    path = queue_dir / "queue.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    job = state["results"][0]
    dependency = Path(state["depends_on_queue_json"])
    if not dependency.is_absolute():
        dependency = ROOT / dependency
    terminal = {"completed", "completed_with_failures", "failed", "stopped"}
    while True:
        if not dependency.is_file():
            state.update(status="blocked", error=f"dependency queue missing: {dependency}", updated_at_utc=now())
            save(path, state)
            return 1
        dep = json.loads(dependency.read_text(encoding="utf-8"))
        state.update(dependency_status=dep.get("status"), last_dependency_check_at_utc=now())
        if dep.get("status") in terminal and dep.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = dep["finished_at_utc"]
            break
        state["status"] = "waiting_for_previous_queue"
        state["updated_at_utc"] = now()
        save(path, state)
        time.sleep(20)

    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
    while True:
        processes, gpu_pids = resource_state()
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu_pids, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(path, state)
        if not processes and not gpu_pids:
            break
        time.sleep(20)

    job.update(status="running", started_at_utc=now())
    state.update(status="running", active_job=job["job_id"], updated_at_utc=now())
    save(path, state)
    log_path = queue_dir / job["log"]
    started = time.monotonic()
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = proc.pid
            save(path, state)
            code = proc.wait()
    except Exception as exc:
        code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"
    job.update(status="completed" if code == 0 else "failed", exit_code=code,
                elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
    state.update(status=job["status"], active_job=None, exit_code=code,
                 finished_at_utc=now(), updated_at_utc=now())
    save(path, state)
    return code


def create_queue() -> Path:
    if not DEPENDENCY_QUEUE.is_file():
        raise FileNotFoundError(f"training queue dependency not found: {DEPENDENCY_QUEUE}")
    dep = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if dep.get("status") in {"completed", "completed_with_failures", "failed", "stopped"}:
        raise ValueError("the expected training queue dependency is already terminal; inspect and update queue ordering")
    source, checkpoint = source_record()
    dataset_path = ROOT / "data/processed/dataset_v2.16_14plus15.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.16_14plus15.json"
    v0_path = ROOT / "data/processed/dataset_v0.parquet"
    for required in (dataset_path, profile_path, v0_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("dataset_version") != DATASET or int(profile.get("rows", 0)) != 277683:
        raise ValueError("V2.16 dataset profile does not match the expected combined dataset")
    if profile.get("component_datasets", {}).keys() != {"v2.14", "v2.15_extrato_14"}:
        raise ValueError("V2.16 profile does not identify V2.14 and V2.15 as its components")

    from src.pipeline import _matching_continuation_source, _training_partitions
    train, valid = _training_partitions(DATASET, "v0", "data/processed")
    if len(train) != 277683 or len(valid) != 2699:
        raise ValueError(f"unexpected V2.16/V0 sizes: {len(train)}/{len(valid)}")
    _matching_continuation_source(
        "ultrabert380", DATASET, "v0", "data/processed", "artifacts", SOURCE_EXPERIMENT,
        allow_cross_dataset_source=True,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_{DATASET}_continue_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = dict(source.get("config", {}))
    config.update({
        "model": "ultrabert380",
        "experiment_name": f"ultrabert380_{DATASET}_continue_v214best_{stamp}",
        "epochs": 3,
        "early_stopping_patience": 3,
        "learning_rate": 1e-6,
        "learning_rate_decay": None,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.02,
        "checkpoint_selection_metric": "balanced_accuracy",
    })
    config_path = config_dir / f"ultrabert380_{DATASET}_continue.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [
        PYTHON, "-u", "-m", "src.pipeline", "models", "continue-encoder",
        "--model", "ultrabert380", "--dataset", DATASET, "--validation-dataset", "v0",
        "--from-experiment", SOURCE_EXPERIMENT, "--epochs", "3", "--learning-rate", "1e-6",
        "--config", str(config_path), "--allow-cross-dataset-source", "--accept-config-defaults",
    ]
    job = {
        "index": 1,
        "job_id": config["experiment_name"],
        "model": "ultrabert380",
        "dataset": DATASET,
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"],
        "validation_dataset": "v0",
        "validation_rows": len(valid),
        "training_mode": "continue_from_selected_checkpoint",
        "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_selected_epoch": int(source["selected_epoch"]),
        "source_balanced_accuracy": source["metrics"]["balanced_accuracy"],
        "epochs": 3,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.02,
        "checkpoint_selection_metric": "balanced_accuracy",
        "status": "queued",
        "config": str(config_path.relative_to(ROOT)),
        "command": command,
        "log": "01_ultrabert380_v2.16_14plus15_continue.log",
        "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id,
        "queue_tag": "ultrabert_v2.16_14plus15_continue",
        "status": "queued_waiting_for_previous_queue",
        "created_at_utc": now(),
        "description": "Continue the best full-V2.14 UltraBERT checkpoint on V2.16_14plus15 for three epochs using LR 1e-6, linear decay, and 2% warmup.",
        "execution_policy": "wait for the currently queued Granite continuation chain to finish, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dep.get("queue_id"),
        "dataset": DATASET,
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"],
        "validation_dataset": "v0",
        "validation_rows": len(valid),
        "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_selected_epoch": int(source["selected_epoch"]),
        "source_balanced_accuracy": source["metrics"]["balanced_accuracy"],
        "results": [job],
        "active_job": None,
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            [PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
    print(json.dumps({
        "queue_dir": str(queue_dir),
        "queue_json": str(queue_dir / "queue.json"),
        "status": "queued_waiting_for_previous_queue",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
