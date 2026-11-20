"""Queue five additional MiniCPM5 QLoRA epochs from experiment 136's selected state."""
from __future__ import annotations

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
SOURCE_EXPERIMENT = "minicpm5_2b_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfg7b636036_code3458579"
ACTIVE_QUEUE = ROOT / "artifacts/training_queues/deberta_glm_v214-glm_20261116T054124Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DATASET = "v2.14"
VALIDATION_DATASET = "v0"
MODEL_ID = "openbmb/MiniCPM5-2B-Base"
MODEL_REVISION = "96a57cd572a02506b4500f54427dca24970c1bac"
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
                pass
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=20)
    return processes, [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]


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

    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        try:
            processes, gpu_pids = active_resources()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(status="blocked", error=f"resource check failed: {exc}", updated_at_utc=now())
            save(queue_path, state)
            return 1
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu_pids, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_path, state)
        if not processes and not gpu_pids:
            break
        time.sleep(20)

    job.update(status="running", started_at_utc=now())
    state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
    save(queue_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting continuation: {' '.join(job['command'])}\n")
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
    if not ACTIVE_QUEUE.is_file():
        raise FileNotFoundError(f"queue-chain tail not found: {ACTIVE_QUEUE}")
    dependency = json.loads(ACTIVE_QUEUE.read_text(encoding="utf-8"))
    manifest_path = ROOT / f"artifacts/predictions/{SOURCE_EXPERIMENT}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("experiment_id") != SOURCE_EXPERIMENT or source.get("model") != "minicpm5_2b":
        raise ValueError("source manifest is not the requested MiniCPM experiment 136")
    if source.get("training_dataset") != DATASET or source.get("validation_dataset") != VALIDATION_DATASET:
        raise ValueError("source dataset lineage differs from V2.14/full V0")
    if int(source.get("selected_epoch", 0)) != 5:
        raise ValueError(f"source selected epoch is {source.get('selected_epoch')}, expected final epoch 5")
    if source.get("checkpoint_id") != MODEL_ID or source.get("model_revision") != MODEL_REVISION:
        raise ValueError("source base model revision differs from the pinned MiniCPM revision")
    source_checkpoint_value = source.get("best_checkpoint_path") or source.get("selected_checkpoint_path")
    if not source_checkpoint_value:
        raise ValueError("source manifest has no best checkpoint")
    source_checkpoint = Path(source_checkpoint_value)
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(source_checkpoint)
    trainer_state_value = source.get("continuation_checkpoint_path")
    trainer_state = Path(trainer_state_value) if trainer_state_value else source_checkpoint / "continuation_checkpoint"
    if not trainer_state.is_absolute():
        trainer_state = ROOT / trainer_state
    required = ["adapter_model.safetensors", "classifier.pt", "checkpoint_manifest.json"]
    missing = [name for name in required if not (source_checkpoint / name).is_file()]
    state_required = ["adapter_model.safetensors", "classifier.pt", "optimizer.pt", "scheduler.pt",
                      "rng_state.pth", "trainer_state.json", "training_args.bin"]
    missing += [f"continuation_checkpoint/{name}" for name in state_required if not (trainer_state / name).is_file()]
    if missing:
        raise FileNotFoundError(f"best checkpoint or resumable Trainer state is incomplete: {missing}")
    trainer_state_json = json.loads((trainer_state / "trainer_state.json").read_text(encoding="utf-8"))
    if int(round(float(trainer_state_json.get("epoch", 0)))) != 5:
        raise ValueError(f"Trainer state is not at completed epoch 5: {trainer_state_json.get('epoch')}")

    from src.pipeline import _matching_continuation_source
    _matching_continuation_source("minicpm5_2b", DATASET, VALIDATION_DATASET,
                                  "data/processed", "artifacts", SOURCE_EXPERIMENT)
    dataset_path = ROOT / "data/processed/dataset_v2.14.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.14.json"
    v0_path = ROOT / "data/processed/dataset_v0.parquet"
    for path in (dataset_path, profile_path, v0_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    train = pd.read_parquet(dataset_path, columns=["model"])
    v0 = pd.read_parquet(v0_path, columns=["model"])
    if len(train) != 231336 or len(v0) != 2699 or train.model.nunique() != 18:
        raise ValueError("V2.14/full-V0 data dimensions differ from experiment 136")
    if profile.get("dataset_sha256") != source.get("dataset_sha256"):
        raise ValueError("V2.14 dataset hash differs from experiment 136")

    # Refuse accidental duplicate continuations of this source and duration.
    for path in QUEUE_ROOT.glob("*/queue.json"):
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("source_experiment") == SOURCE_EXPERIMENT and prior.get("additional_epochs") == 5:
            raise FileExistsError(f"matching continuation already queued: {path.parent}")
        if any(job.get("source_experiment") == SOURCE_EXPERIMENT and job.get("epochs") == 5
               for job in prior.get("results", [])):
            raise FileExistsError(f"matching continuation already queued: {path.parent}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"minicpm5_2b_v214_continue5_from_exp136_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = {k: v for k, v in source.get("config", {}).items() if not str(k).startswith("_")}
    config.update(model="minicpm5_2b", epochs=5)
    config_path = config_dir / "minicpm5_2b_v214_continue5.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "continue-lora",
               "--model", "minicpm5_2b", "--dataset", DATASET,
               "--validation-dataset", VALIDATION_DATASET,
               "--from-experiment", SOURCE_EXPERIMENT, "--epochs", "5",
               "--config", str(config_path), "--seed", str(source.get("seed", 2026)),
               "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    job = {
        "index": 1, "job_id": f"minicpm5_2b_v2.14_continue5_from_exp136_{stamp}",
        "model": "minicpm5_2b", "display_name": "MiniCPM5-2B-Base", "dataset": DATASET,
        "dataset_path": str(dataset_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": VALIDATION_DATASET,
        "validation_rows": len(v0), "training_mode": "resume_from_best_checkpoint_with_trainer_state",
        "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_trainer_state": str(trainer_state.relative_to(ROOT)),
        "source_selected_epoch": 5, "source_global_step": trainer_state_json.get("global_step"),
        "source_validation_accuracy": source.get("metrics", {}).get("accuracy"),
        "source_validation_balanced_accuracy": source.get("metrics", {}).get("balanced_accuracy"),
        "additional_epochs": 5, "target_total_epochs": 10,
        "learning_rate": config.get("learning_rate"), "optimizer": config.get("optimizer"),
        "lr_scheduler_type": config.get("lr_scheduler_type"), "warmup_ratio": config.get("warmup_ratio"),
        "weight_decay": config.get("weight_decay"), "quantization": config.get("quantization"),
        "lora_rank": config.get("lora_rank"), "lora_alpha": config.get("lora_alpha"),
        "batch_size": source.get("selected_physical_batch_size"),
        "gradient_accumulation_steps": source.get("gradient_accumulation_steps"),
        "sequence_length": source.get("config", {}).get("max_length"),
        "checkpoint_selection_metric": source.get("checkpoint_selection_metric"),
        "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "log": "01_minicpm5_2b_v214_continue5.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "minicpm5_2b_v214_continue5_from_exp136",
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Resume MiniCPM experiment 136 from its selected final-epoch checkpoint and Trainer state for five additional epochs.",
        "execution_policy": "wait until the existing DeBERTa specialist queue chain finishes, then wait for GPU training processes and compute to be idle",
        "depends_on_queue_json": str(ACTIVE_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"), "dataset": DATASET,
        "dataset_path": str(dataset_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": VALIDATION_DATASET,
        "validation_rows": len(v0), "model": MODEL_ID, "revision": MODEL_REVISION,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_trainer_state": str(trainer_state.relative_to(ROOT)),
        "source_selected_epoch": 5, "source_global_step": trainer_state_json.get("global_step"),
        "additional_epochs": 5, "results": [job], "active_job": None,
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
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
