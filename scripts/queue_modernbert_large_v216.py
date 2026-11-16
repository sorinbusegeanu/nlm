"""Queue ModernBERT-large on V2.16 after the current serialized GPU queue."""
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
MODEL = "modernbert_large"
DATASET = "v2.16"
CONFIG = ROOT / "configs/models/modernbert_large.yaml"
HF_HOME = Path("/var/tmp/nlm_hf_cache")
DEPENDENCY_QUEUE = ROOT / "artifacts/training_queues/granite41_3b_v216_20261114T193633Z/queue.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}
REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            log.write(f"[{now()}] Starting: HF_HUB_OFFLINE=1 {' '.join(job['command'])}\n")
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
        raise FileNotFoundError("ModernBERT config or preceding Granite 4.1 queue is missing")
    for existing in QUEUE_ROOT.glob("*/queue.json"):
        try:
            entry = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (entry.get("queue_tag") == "modernbert_large_v216"
                and entry.get("status") not in TERMINAL):
            raise ValueError(f"an active ModernBERT V2.16 queue already exists: {existing}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if config.get("model") != MODEL or config.get("checkpoint") != "answerdotai/ModernBERT-large":
        raise ValueError("ModernBERT config has an unexpected model identity")
    if config.get("revision") != REVISION or config.get("training_mode") != "full":
        raise ValueError("ModernBERT config must use the pinned full-finetuning model revision")
    if config.get("checkpoint_selection_metric") != "balanced_accuracy":
        raise ValueError("ModernBERT checkpoint selection must use V0 balanced accuracy")
    previous = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if previous.get("status") in TERMINAL:
        raise ValueError("the preceding Granite 4.1 queue is already terminal; inspect queue ordering before appending")

    data_path = ROOT / "data/processed/dataset_v2.16.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.16.json"
    if not data_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("V2.16 alias artifact or profile is missing")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    data_sha = sha256(data_path)
    if profile.get("dataset_version") != DATASET or int(profile.get("rows", 0)) != 277683:
        raise ValueError("V2.16 profile does not match the verified full dataset")
    if profile.get("dataset_sha256") != data_sha:
        raise ValueError("V2.16 dataset hash differs from its profile")

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=config["checkpoint"], revision=config["revision"], local_files_only=True,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json"])

    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, "v0", "data/processed")
    if len(train) != 277683 or len(valid) != 2699 or train.model.nunique() != 18:
        raise ValueError(f"unexpected V2.16/full-V0 sizes or classes: {len(train)}/{len(valid)}/{train.model.nunique()}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"modernbert_large_v216_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    queue_dir.mkdir(parents=True, exist_ok=False)
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
        "--model", MODEL, "--dataset", DATASET, "--validation-dataset", "v0",
        "--config", "configs/models/modernbert_large.yaml", "--max-length", "512",
        "--accept-config-defaults"]
    job = {
        "index": 1, "job_id": f"modernbert_large_{DATASET}_{stamp}", "model": MODEL,
        "display_name": "ModernBERT-large", "dataset": DATASET,
        "dataset_path": str(data_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": data_sha, "validation_dataset": "v0", "validation_rows": len(valid),
        "training_mode": "full", "epochs": int(config["epochs"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "learning_rate": float(config["learning_rate"]), "batch_size": int(config["batch_size"]),
        "max_batch_size": int(config["max_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "max_length": int(config["max_length"]),
        "checkpoint_selection_metric": "balanced_accuracy", "revision": config["revision"],
        "attention_backend": config["attention_backend"], "config": "configs/models/modernbert_large.yaml",
        "command": command, "environment": {"HF_HUB_OFFLINE": "1"},
        "status": "queued", "log": "01_modernbert_large_v216.log", "queued_at_utc": now(),
    }
    state = {
        "queue_id": queue_id, "queue_tag": "modernbert_large_v216",
        "status": "queued_waiting_for_granite41_3b_v216", "created_at_utc": now(),
        "description": "Full-finetune ModernBERT-large on V2.16 and select by full-V0 balanced accuracy.",
        "execution_policy": "run after Granite 4.1 queue completion, then wait for all training processes and GPU compute to be idle",
        "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
        "dependency_queue_id": previous.get("queue_id"), "dataset": DATASET,
        "dataset_path": str(data_path.relative_to(ROOT)), "dataset_rows": len(train),
        "dataset_sha256": data_sha, "validation_dataset": "v0", "validation_rows": len(valid),
        "checkpoint": config["checkpoint"], "revision": config["revision"],
        "model_cache": str(HF_HOME), "attention_backend": config["attention_backend"],
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
                      "status": "queued_waiting_for_granite41_3b_v216"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
