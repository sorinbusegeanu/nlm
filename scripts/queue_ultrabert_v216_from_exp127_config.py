"""Queue an UltraBERT V2.16 from-scratch run with experiment 0127 settings."""
from __future__ import annotations

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
SOURCE_EXPERIMENT = "ultrabert380_v2.14-prediction-filtered-80pct_valv0_dataee63cf16_splitab1d2edc_seed2026_cfgf9f6cf6c_code9c2e334"
SOURCE_MANIFEST = ROOT / f"artifacts/predictions/{SOURCE_EXPERIMENT}_manifest.json"
DEPENDENCY_QUEUE = ROOT / "artifacts/training_queues/granite4_350m_v214_continue4_from_exp129_after131_20261115T061924Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DATASET = "v2.16"
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
        line = line.strip()
        if "src.pipeline models" in line and Path(__file__).name not in line:
            try:
                pid, command = line.split(None, 1)
                processes.append({"pid": int(pid), "command": command[:400]})
            except ValueError:
                continue
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


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
        state.update(status="waiting_for_previous_queue", updated_at_utc=now())
        save(state_path, state)
        time.sleep(20)

    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        processes, gpu_pids = active_resources()
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
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("model") != "ultrabert380" or source.get("training_dataset") != "v2.14-prediction-filtered-80pct":
        raise ValueError("experiment 0127 manifest does not match the expected UltraBERT source")
    config = dict(source["config"])
    dependency = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError("queue tail is already terminal; inspect order before adding this run")

    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, "v0", "data/processed")
    if len(valid) != 2699 or len(train) != 277683:
        raise ValueError(f"unexpected V2.16/full-V0 partitions: {len(train)}/{len(valid)}")
    if set(train.model.astype(str)) != set(valid.model.astype(str)):
        raise ValueError("V2.16 and V0 label rosters differ")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_v216_from_exp127_config_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config.update({
        "model": "ultrabert380",
        "experiment_name": f"ultrabert380_v2.16_from_scratch_exp127_config_{stamp}",
    })
    config_path = config_dir / "ultrabert380_v2.16.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "ultrabert380", "--dataset", DATASET,
               "--validation-dataset", "v0", "--config", str(config_path),
               "--max-length", str(config["max_length"]), "--seed", "2026",
               "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    dataset_path = ROOT / "data/processed/dataset_v2.16.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.16.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    job = {
        "index": 1, "job_id": config["experiment_name"], "model": "ultrabert380",
        "dataset": DATASET, "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_rows": len(train), "dataset_sha256": profile["dataset_sha256"],
        "validation_dataset": "v0", "validation_rows": len(valid),
        "training_mode": "from_scratch", "config_source_experiment": SOURCE_EXPERIMENT,
        "epochs": config["epochs"], "early_stopping_patience": config["early_stopping_patience"],
        "learning_rate": config["learning_rate"], "scheduler": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"], "max_length": config["max_length"],
        "checkpoint_selection_metric": config["checkpoint_selection_metric"],
        "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
        "log": "01_ultrabert380_v2.16.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "ultrabert_v216_from_exp127_config",
        "status": "queued_waiting_for_dependency", "created_at_utc": now(),
        "description": "Train UltraBERT from scratch on V2.16 using experiment 0127 training settings.",
        "execution_policy": "wait for the queued Granite 350M experiment 0129 continuation, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"), "config_source_experiment": SOURCE_EXPERIMENT,
        "dataset": DATASET, "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_rows": len(train), "dataset_sha256": profile["dataset_sha256"],
        "validation_dataset": "v0", "validation_rows": len(valid), "results": [job],
        "active_job": None,
    }
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
    import argparse
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
