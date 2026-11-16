"""Queue Granite 4.1 3B LoRA training on V2.16 after the current training chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
MODEL = "granite41_3b"
DATASET = "v2.16_14plus15"
CONFIG = ROOT / "configs/models/granite41_3b.yaml"
HF_HOME = Path("/var/tmp/nlm_hf_cache")
DEPENDENCY_QUEUE = ROOT / (
    "artifacts/training_queues/ultrabert_v2.16_14plus15_continue_20261114T094355Z/queue.json"
)
QUEUE_ROOT = ROOT / "artifacts/training_queues"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resource_state() -> tuple[list[dict], list[int]]:
    processes = []
    script_name = Path(__file__).name
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models" in parts[1] and script_name not in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                pass
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

    env = {
        **os.environ,
        "HF_HOME": str(HF_HOME),
        "HF_HUB_CACHE": str(HF_HOME / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
    while True:
        try:
            processes, gpu_pids = resource_state()
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
    state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
    save(state_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env,
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
    if not CONFIG.is_file() or not DEPENDENCY_QUEUE.is_file():
        raise FileNotFoundError("Granite 4.1 config or preceding V2.16 queue is missing")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if config.get("model") != MODEL or config.get("checkpoint") != "ibm-granite/granite-4.1-3b-base":
        raise ValueError("Granite 4.1 config has an unexpected model identity")
    if config.get("revision") != "dacb9cb9157bec98e99b09f285c92a4d58405c96":
        raise ValueError("Granite 4.1 config is not pinned to the verified revision")
    if config.get("checkpoint_selection_metric") != "balanced_accuracy":
        raise ValueError("Granite 4.1 checkpoint selection must use V0 balanced accuracy")
    if int(config.get("max_length", 0)) != 512 or int(config.get("gradient_accumulation_steps", 0)) != 1:
        raise ValueError("Granite 4.1 config must preserve max_length=512 and accumulation=1")
    previous = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if previous.get("status") in TERMINAL:
        raise ValueError("the preceding V2.16 queue is already terminal; inspect queue ordering before appending")

    data_path = ROOT / "data/processed/dataset_v2.16_14plus15.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.16_14plus15.json"
    if not data_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("V2.16 dataset artifact or profile is missing")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("dataset_version") != DATASET or int(profile.get("rows", 0)) != 277683:
        raise ValueError("V2.16 dataset profile does not match the expected full dataset")

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=config["checkpoint"], revision=config["revision"], local_files_only=True)

    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, "v0", "data/processed")
    if len(train) != 277683 or len(valid) != 2699:
        raise ValueError(f"unexpected V2.16/V0 sizes: {len(train)}/{len(valid)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"granite41_3b_v216_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    config_copy = queue_dir / "granite41_3b_v216.yaml"
    config_copy.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
        "--model", MODEL, "--dataset", DATASET, "--validation-dataset", "v0",
        "--config", str(config_copy), "--max-length", "512", "--seed", "2026",
        "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    job = {
        "index": 1, "job_id": f"granite41_3b_{DATASET}_{stamp}", "model": MODEL,
        "display_name": "Granite 4.1 3B Base", "dataset": DATASET,
        "dataset_path": str(data_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": "v0",
        "validation_rows": len(valid), "training_mode": "lora", "quantization": "none",
        "epochs": int(config["epochs"]), "early_stopping_patience": int(config["early_stopping_patience"]),
        "learning_rate": float(config["learning_rate"]), "batch_size": int(config["batch_size"]),
        "max_batch_size": int(config["max_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "max_length": int(config["max_length"]),
        "checkpoint_selection_metric": "balanced_accuracy", "revision": config["revision"],
        "config": str(config_copy.relative_to(ROOT)), "command": command,
        "status": "queued", "log": "01_granite41_3b_v216.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "granite41_3b_v216",
        "status": "queued_waiting_for_ultrabert_v216", "created_at_utc": now(),
        "description": "Train Granite 4.1 3B Base with LoRA on V2.16 and select by full-V0 balanced accuracy.",
        "execution_policy": "wait for the existing V2.16 UltraBERT queue, then wait for all training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": previous.get("queue_id"), "dataset": DATASET,
        "dataset_path": str(data_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": "v0",
        "validation_rows": len(valid), "checkpoint": config["checkpoint"],
        "revision": config["revision"], "model_cache": str(HF_HOME),
        "checkpoint_selection_metric": "balanced_accuracy", "results": [job], "active_job": None,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_ultrabert_v216"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
