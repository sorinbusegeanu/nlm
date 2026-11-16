"""Queue a fresh Qwen3-0.6B LoRA run on V2.16 using experiment 0125 settings."""
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
SOURCE_EXPERIMENT = "qwen06_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgba46767d_code81bce4a"
SOURCE_MANIFEST = ROOT / f"artifacts/predictions/{SOURCE_EXPERIMENT}_manifest.json"
DEPENDENCY_QUEUE = ROOT / "artifacts/training_queues/ultrabert_v216_from_exp127_config_20261115T062126Z/queue.json"
DATASET = "v2.16"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
HF_HOME = Path.home() / ".cache/huggingface"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}


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
                continue
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    job = state["results"][0]
    dependency = Path(state["depends_on_queue_json"])
    if not dependency.is_absolute():
        dependency = ROOT / dependency
    while True:
        if not dependency.is_file():
            state.update(status="blocked", error=f"dependency queue missing: {dependency}", updated_at_utc=now())
            save(queue_path, state)
            return 1
        previous = json.loads(dependency.read_text(encoding="utf-8"))
        state.update(dependency_status=previous.get("status"), last_dependency_check_at_utc=now())
        if previous.get("status") in TERMINAL and previous.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = previous["finished_at_utc"]
            break
        state.update(status="waiting_for_previous_queue", updated_at_utc=now())
        save(queue_path, state)
        time.sleep(20)

    environment = {
        **os.environ,
        "HF_HOME": str(HF_HOME),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
    while True:
        try:
            processes, gpu = active_resources()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(status="blocked", error=f"resource check failed: {exc}", updated_at_utc=now())
            save(queue_path, state)
            return 1
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_path, state)
        if not processes and not gpu:
            break
        time.sleep(20)

    job.update(status="running", started_at_utc=now())
    state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
    save(queue_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting offline: {' '.join(job['command'])}\n")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = process.pid
            save(queue_path, state)
            code = process.wait()
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
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_config = source.get("config") or {}
    if source.get("model") != "qwen06" or source.get("training_dataset") != "v2.14":
        raise ValueError("experiment 0125 is not the expected Qwen3-0.6B V2.14 run")
    if source_config.get("checkpoint") != MODEL_ID or source_config.get("revision") != MODEL_REVISION:
        raise ValueError("experiment 0125 model identity or pinned revision differs")
    dependency = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError("current queue tail is already terminal; inspect ordering before adding this job")

    snapshot = HF_HOME / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / MODEL_REVISION
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / filename).resolve().is_file():
            raise FileNotFoundError(f"pinned local model file missing: {snapshot / filename}")
    if not any((snapshot / name).resolve().is_file()
               for name in ("model.safetensors", "model.safetensors.index.json")):
        raise FileNotFoundError(f"pinned Qwen weight files missing from {snapshot}")

    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, "v0", "data/processed")
    if len(train) != 277683 or len(valid) != 2699 or train.model.nunique() != 18:
        raise ValueError(f"unexpected V2.16/full-V0 data sizes/classes: {len(train)}/{len(valid)}/{train.model.nunique()}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"qwen06_v216_scratch_exp125_params_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = dict(source_config)
    config.update({
        "model": "qwen06",
        "experiment_name": f"qwen06_v2.16_from_scratch_exp125_params_{stamp}",
        "epochs": 10,
    })
    # The old run's 538-token maximum was measured on V2.14. Leave it unset
    # so the factory recalculates the dynamic tokenizer maximum for V2.16/V0.
    config.pop("max_length", None)
    config_path = config_dir / "qwen06_v2.16_scratch.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "qwen06", "--dataset", DATASET, "--validation-dataset", "v0",
               "--config", str(config_path), "--seed", "2026", "--accept-config-defaults",
               "--refresh-checkpoint-catalog"]
    profile = json.loads((ROOT / "data/processed/dataset_profile_v2.16.json").read_text(encoding="utf-8"))
    job = {
        "index": 1, "job_id": config["experiment_name"], "model": "qwen06",
        "display_name": "Qwen3-0.6B", "dataset": DATASET,
        "dataset_path": "data/processed/dataset_v2.16.parquet", "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": "v0",
        "validation_rows": len(valid), "training_mode": "lora_from_scratch",
        "source_parameter_experiment": SOURCE_EXPERIMENT,
        "checkpoint": MODEL_ID, "revision": MODEL_REVISION,
        "epochs": 10, "early_stopping_patience": config["early_stopping_patience"],
        "learning_rate": config["learning_rate"], "optimizer": config["optimizer"],
        "batch_size": config["batch_size"], "auto_batch_size": config["auto_batch_size"],
        "max_batch_size": config["max_batch_size"],
        "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "dynamic_max_length": True, "checkpoint_selection_metric": "balanced_accuracy",
        "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "log": "01_qwen06_v216_scratch.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "qwen06_v216_scratch_exp125_params",
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Train Qwen3-0.6B from scratch on V2.16 for up to 10 epochs using experiment 0125 parameters.",
        "execution_policy": "wait for the current serialized training queue to complete, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"), "source_parameter_experiment": SOURCE_EXPERIMENT,
        "model": MODEL_ID, "revision": MODEL_REVISION, "dataset": DATASET,
        "dataset_path": "data/processed/dataset_v2.16.parquet", "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": "v0",
        "validation_rows": len(valid), "dynamic_max_length": True,
        "results": [job], "active_job": None,
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    state["worker_pid"] = process.pid
    save(queue_path, state)
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
                      "status": "queued_waiting_for_previous_queue"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
