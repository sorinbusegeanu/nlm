"""Queue UltraBERT continuation from the best full-V2.14 checkpoint onto V2.15."""
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
SOURCE_EXPERIMENT = (
    "ultrabert380_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_"
    "cfgb9e735ff_code81bce4a"
)
SOURCE_MANIFEST = ROOT / f"artifacts/predictions/{SOURCE_EXPERIMENT}_manifest.json"
ACTIVE_QUEUE = ROOT / "artifacts/training_queues/v2_14_frozen_style_20261113T113511Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DATASET = "v2.15_extrato_14"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_and_lr() -> tuple[dict, Path, float]:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("model") != "ultrabert380" or manifest.get("training_dataset") != "v2.14":
        raise ValueError("selected source manifest is not a full-V2.14 UltraBERT run")
    if manifest.get("validation_dataset") != "v0" or int(manifest.get("validation_rows", 0)) != 2699:
        raise ValueError("source checkpoint was not selected on all 2,699 V0 rows")
    checkpoint_value = manifest.get("selected_checkpoint_path") or manifest.get("best_checkpoint_path")
    if not checkpoint_value:
        raise ValueError("source manifest has no selected checkpoint path")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"selected checkpoint is missing: {checkpoint}")
    selected_epoch = int(manifest.get("selected_epoch", 0))
    log_path = checkpoint / "training_log.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"checkpoint training log is missing: {log_path}")
    events = json.loads(log_path.read_text(encoding="utf-8"))
    matching = [e for e in events if int(float(e.get("epoch", -1))) == selected_epoch
                and e.get("learning_rate") is not None and e.get("step") is not None]
    if not matching:
        raise ValueError(f"no recorded learning rate for selected epoch {selected_epoch}")
    last_event = max(matching, key=lambda e: int(e["step"]))
    learning_rate = float(last_event["learning_rate"])
    return manifest, checkpoint, learning_rate


def resource_state() -> tuple[list[dict], list[int]]:
    ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    processes = []
    for line in ps.splitlines():
        line = line.strip()
        if "src.pipeline models" in line and "queue_ultrabert_v2_15_extrato_14_continue.py" not in line:
            try:
                pid, cmd = line.split(None, 1)
                processes.append({"pid": int(pid), "command": cmd[:400]})
            except ValueError:
                continue
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=20)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


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

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        processes, gpu = resource_state()
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(path, state)
        if not processes and not gpu:
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
    if not ACTIVE_QUEUE.is_file():
        raise FileNotFoundError(f"active training queue not found: {ACTIVE_QUEUE}")
    source, checkpoint, learning_rate = source_and_lr()
    dataset_path = ROOT / "data/processed/dataset_v2.15_extrato_14.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.15_extrato_14.json"
    v0_path = ROOT / "data/processed/dataset_v0.parquet"
    for path in (dataset_path, profile_path, v0_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    data = pd.read_parquet(dataset_path, columns=["model"])
    v0 = pd.read_parquet(v0_path, columns=["model"])
    if set(data.model.astype(str)) != set(v0.model.astype(str)):
        raise ValueError("V2.15 and V0 class labels differ")
    # Use the same full-V0 scope as source selection and validate cross-dataset
    # continuation eligibility before publishing the queue record.
    from src.pipeline import _matching_continuation_source
    _matching_continuation_source("ultrabert380", DATASET, "v0", "data/processed", "artifacts",
        SOURCE_EXPERIMENT, allow_cross_dataset_source=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_{DATASET}_continue_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    parent_config = source.get("config", {})
    config = {
        "model": "ultrabert380",
        "experiment_name": f"ultrabert380_{DATASET}_continue_v214best_{stamp}",
        "checkpoint": parent_config.get("checkpoint", source.get("checkpoint_id", "dmis-lab/UltraBERT-380M")),
        "revision": parent_config.get("revision", source.get("model_revision")),
        "epochs": 20,
        "early_stopping_patience": int(parent_config.get("early_stopping_patience", 2)),
        "learning_rate": learning_rate,
        "learning_rate_decay": None,
        "lr_scheduler_type": parent_config.get("lr_scheduler_type", "linear"),
        "warmup_ratio": float(parent_config.get("warmup_ratio", 0.0)),
        "optimizer": parent_config.get("optimizer", "adamw_torch"),
        "weight_decay": float(parent_config.get("weight_decay", 0.0)),
        "batch_size": int(parent_config.get("batch_size", 8)),
        "gradient_accumulation_steps": int(parent_config.get("gradient_accumulation_steps", 1)),
        "target_effective_batch_size": parent_config.get("target_effective_batch_size"),
        "inference_batch_size": int(parent_config.get("inference_batch_size", 8)),
        "auto_batch_size": bool(parent_config.get("auto_batch_size", True)),
        "max_batch_size": int(parent_config.get("max_batch_size", 1024)),
        "batch_probe_safety_fraction": float(parent_config.get("batch_probe_safety_fraction", 0.95)),
        "gradient_checkpointing": bool(parent_config.get("gradient_checkpointing", True)),
        "local_files_only": True,
        "checkpoint_selection_metric": "balanced_accuracy",
    }
    config_path = config_dir / f"ultrabert380_{DATASET}_continue.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "continue-encoder",
        "--model", "ultrabert380", "--dataset", DATASET, "--validation-dataset", "v0",
        "--from-experiment", SOURCE_EXPERIMENT, "--epochs", str(config["epochs"]),
        "--learning-rate", repr(learning_rate), "--config", str(config_path),
        "--allow-cross-dataset-source", "--accept-config-defaults"]
    dependency_state = json.loads(ACTIVE_QUEUE.read_text(encoding="utf-8"))
    job = {"index": 1, "job_id": config["experiment_name"], "model": "ultrabert380",
        "dataset": DATASET, "dataset_path": "data/processed/dataset_v2.15_extrato_14.parquet",
        "dataset_rows": int(len(data)), "dataset_sha256": profile.get("dataset_sha256"),
        "validation_dataset": "v0", "validation_rows": int(len(v0)),
        "training_mode": "continue_from_selected_checkpoint", "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": str(checkpoint.relative_to(ROOT)), "source_balanced_accuracy": source["metrics"]["balanced_accuracy"],
        "source_selected_epoch": int(source["selected_epoch"]), "source_learning_rate_last_used": learning_rate,
        "epochs": config["epochs"], "learning_rate": learning_rate,
        "learning_rate_scheduler": config["lr_scheduler_type"], "status": "queued",
        "config": str(config_path.relative_to(ROOT)), "command": command,
        "log": "01_ultrabert380_v2.15_extrato_14_continue.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": "ultrabert_v2_15_extrato_14_continue",
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Continue the best full-V2.14 UltraBERT checkpoint on V2.15_extrato_14 using the source checkpoint's last recorded learning rate.",
        "execution_policy": "wait for the existing V2.14 frozen-style queue to finish, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(ACTIVE_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": dependency_state.get("queue_id"), "dataset": DATASET,
        "dataset_path": str(dataset_path.relative_to(ROOT)), "dataset_rows": int(len(data)),
        "dataset_sha256": profile.get("dataset_sha256"), "validation_dataset": "v0",
        "validation_rows": int(len(v0)), "source_experiment": SOURCE_EXPERIMENT,
        "source_balanced_accuracy": source["metrics"]["balanced_accuracy"],
        "source_selected_epoch": int(source["selected_epoch"]),
        "source_learning_rate_last_used": learning_rate,
        "results": [job], "active_job": None}
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    worker_log = (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=worker_log, stderr=subprocess.STDOUT,
        start_new_session=True)
    worker_log.close()
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
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_previous_queue"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
