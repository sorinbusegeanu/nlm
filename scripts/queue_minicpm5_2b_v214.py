"""Queue MiniCPM5-2B QLoRA training after the DeBERTa V2.14 job."""
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
DATASET = "v2.14"
VALIDATION_DATASET = "v0"
MODEL_ID = "openbmb/MiniCPM5-2B-Base"
MODEL_REVISION = "96a57cd572a02506b4500f54427dca24970c1bac"
HF_HOME = Path.home() / ".cache/huggingface"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DEPENDENCY_QUEUE = QUEUE_ROOT / "deberta_v3_large_v214_scratch_20261115T184936Z/queue.json"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def active_resources() -> tuple[list[dict], list[int]]:
    processes = []
    script_name = Path(__file__).name
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models" in parts[1] and script_name not in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                continue
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
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

    environment = {**os.environ, "HF_HOME": str(HF_HOME), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "OMP_NUM_THREADS": "1"}
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
    if not DEPENDENCY_QUEUE.is_file():
        raise FileNotFoundError(f"DeBERTa dependency queue not found: {DEPENDENCY_QUEUE}")
    dependency = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError("DeBERTa queue is already terminal; inspect queue order before adding MiniCPM")

    snapshot = HF_HOME / "hub" / "models--openbmb--MiniCPM5-2B-Base" / "snapshots" / MODEL_REVISION
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (snapshot / name).resolve().is_file()]
    if missing:
        raise FileNotFoundError(f"pinned local MiniCPM files missing under {snapshot}: {missing}")

    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, VALIDATION_DATASET, ROOT / "data/processed")
    if len(train) != 231336 or len(valid) != 2699 or train.model.nunique() != 18:
        raise ValueError(f"unexpected V2.14/full-V0 rows/classes: {len(train)}/{len(valid)}/{train.model.nunique()}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"minicpm5_2b_v214_qlora_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "configs/models/minicpm5_2b.yaml").read_text(encoding="utf-8"))
    if config.get("checkpoint") != MODEL_ID or config.get("revision") != MODEL_REVISION:
        raise ValueError("MiniCPM model ID or pinned revision differs from the approved config")
    expected = {"epochs": 5, "learning_rate": 5e-5, "lora_rank": 16, "lora_alpha": 32,
                "lr_scheduler_type": "linear", "warmup_ratio": 0.02, "weight_decay": 0.0,
                "quantization": "qlora4", "optimizer": "paged_adamw_8bit"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("MiniCPM config differs from the requested QLoRA training parameters")
    if config.get("checkpoint_selection_metric") != "balanced_accuracy":
        raise ValueError("MiniCPM must select checkpoints using full-V0 balanced accuracy")
    config["experiment_name"] = f"minicpm5_2b_v2.14_qlora_{stamp}"
    config_path = config_dir / "minicpm5_2b_v2.14_qlora.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "minicpm5_2b", "--dataset", DATASET,
               "--validation-dataset", VALIDATION_DATASET,
               "--config", str(config_path), "--seed", "2026",
               "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    profile = json.loads((ROOT / "data/processed/dataset_profile_v2.14.json").read_text(encoding="utf-8"))
    job = {"index": 1, "job_id": config["experiment_name"], "model": "minicpm5_2b",
        "display_name": "MiniCPM5-2B-Base", "dataset": DATASET,
        "dataset_path": "data/processed/dataset_v2.14.parquet", "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": VALIDATION_DATASET,
        "validation_rows": len(valid), "training_mode": "qlora4_nf4",
        "compute_dtype": "bfloat16", "checkpoint": MODEL_ID, "revision": MODEL_REVISION,
        "epochs": config["epochs"], "early_stopping_patience": config["early_stopping_patience"],
        "learning_rate": config["learning_rate"], "optimizer": config["optimizer"],
        "lr_scheduler_type": config["lr_scheduler_type"], "warmup_ratio": config["warmup_ratio"],
        "weight_decay": config["weight_decay"], "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"], "lora_target_modules": config["lora_target_modules"],
        "batch_size": config["batch_size"], "auto_batch_size": config["auto_batch_size"],
        "max_batch_size": config["max_batch_size"],
        "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "sequence_length": "automatic_from_training_data", "checkpoint_selection_metric": "balanced_accuracy",
        "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "log": "01_minicpm5_2b_v214.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": "minicpm5_2b_v214_qlora",
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Train MiniCPM5-2B-Base with QLoRA 4-bit NF4 on all V2.14 rows for up to 5 epochs; validate on full V0.",
        "execution_policy": "wait for the DeBERTa V2.14 queue to finish, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"), "dataset": DATASET,
        "dataset_path": "data/processed/dataset_v2.14.parquet", "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": VALIDATION_DATASET,
        "validation_rows": len(valid), "model": MODEL_ID, "revision": MODEL_REVISION,
        "results": [job], "active_job": None}
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
