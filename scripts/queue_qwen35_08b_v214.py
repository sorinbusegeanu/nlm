"""Queue Qwen3.5-0.8B-Base LoRA training after ModernBERT's V2.14HN run."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DATASET = "v2.14"
VALIDATION_DATASET = "v0"
HF_HOME = Path.home() / ".cache/huggingface"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DEPENDENCY_QUEUE = QUEUE_ROOT / (
    "modernbert_v214hn_weighted_continue_n0127_20261118T184913Z/queue.json"
)
FAILED_QWEN_QUEUE = QUEUE_ROOT / (
    "qwen35_08b_v214_lora_20261118T192920Z/queue.json"
)
QUEUE_TAG = "qwen35_08b_v214_lora"
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


def wait_for_gpu(queue_path: Path, state: dict) -> None:
    while True:
        try:
            processes, gpu = active_resources()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(status="blocked", error=f"resource check failed: {exc}", updated_at_utc=now())
            save(queue_path, state)
            raise
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_path, state)
        if not processes and not gpu:
            return
        time.sleep(20)


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

    wait_for_gpu(queue_path, state)
    state.update(status="downloading_pinned_model", download_started_at_utc=now(), updated_at_utc=now())
    save(queue_path, state)
    try:
        # Training remains strictly offline. This is the only stage allowed to
        # contact the Hub, and it runs after the preceding queue releases GPU.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from huggingface_hub import snapshot_download
        try:
            snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                              cache_dir=HF_HOME, local_files_only=True))
        except Exception:
            snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                              cache_dir=HF_HOME, local_files_only=False))
        index_path = snapshot / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"pinned snapshot has no safetensors index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_files = sorted(set(index.get("weight_map", {}).values()))
        required = (snapshot / "config.json", snapshot / "tokenizer_config.json", snapshot / "tokenizer.json")
        if any(not path.is_file() for path in required) or not weight_files or any(
                not (snapshot / filename).is_file() for filename in weight_files):
            raise FileNotFoundError(f"pinned snapshot is incomplete: {snapshot}")
        state.update(model_snapshot=str(snapshot), model_download_finished_at_utc=now(),
                     status="model_ready", updated_at_utc=now())
        job["model_snapshot"] = str(snapshot)
        save(queue_path, state)
    except Exception as exc:
        job.update(status="failed_download", error=f"{type(exc).__name__}: {exc}", finished_at_utc=now())
        state.update(status="failed_download", error=job["error"], finished_at_utc=now(), updated_at_utc=now())
        save(queue_path, state)
        return 1

    # Another process may have claimed the GPU while the model downloaded.
    wait_for_gpu(queue_path, state)
    environment = {**os.environ, "HF_HOME": str(HF_HOME), "HF_HUB_CACHE": str(HF_HOME),
                   "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "OMP_NUM_THREADS": "1"}
    job.update(status="running", started_at_utc=now())
    state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
    save(queue_path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting offline training: {' '.join(job['command'])}\n")
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


def create_queue(*, retry_failed=False) -> Path:
    dependency_path = FAILED_QWEN_QUEUE if retry_failed else DEPENDENCY_QUEUE
    if not dependency_path.is_file():
        raise FileNotFoundError(f"dependency queue not found: {dependency_path}")
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    if retry_failed:
        if dependency.get("status") != "failed" or not dependency.get("finished_at_utc"):
            raise ValueError("cache-fix retry requires the original Qwen3.5 queue to be failed and finalized")
    elif dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError("ModernBERT queue is already terminal; refusing to append a stale first-attempt job")
    queue_tag = f"{QUEUE_TAG}_cachefix_retry" if retry_failed else QUEUE_TAG
    for existing in QUEUE_ROOT.glob("*/queue.json"):
        try:
            item = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("queue_tag") == queue_tag:
            raise FileExistsError(f"Qwen3.5 V2.14 queue already exists: {existing}")

    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, VALIDATION_DATASET, ROOT / "data/processed")
    if len(train) != 231336 or len(valid) != 2699 or train.model.nunique() != 18:
        raise ValueError(f"unexpected V2.14/full-V0 rows/classes: {len(train)}/{len(valid)}/{train.model.nunique()}")
    profile_path = ROOT / "data/processed/dataset_profile_v2.14.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    config = yaml.safe_load((ROOT / "configs/models/qwen35_08b.yaml").read_text(encoding="utf-8"))
    expected = {"model": "qwen35_08b", "checkpoint": MODEL_ID, "revision": MODEL_REVISION,
                "epochs": 10, "learning_rate": 5e-5, "training_mode": "lora",
                "quantization": "none", "text_only_backbone": True, "use_chat_template": True}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Qwen3.5 configuration differs from the queued base-model recipe")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"qwen35_08b_v214_lora{'_cachefix_retry' if retry_failed else ''}_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config["experiment_name"] = f"qwen35_08b_v2.14_lora{'_cachefix_retry' if retry_failed else ''}_{stamp}"
    config_path = config_dir / "qwen35_08b_v2.14.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "qwen35_08b", "--dataset", DATASET,
               "--validation-dataset", VALIDATION_DATASET,
               "--config", str(config_path), "--seed", "2026",
               "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    job = {"index": 1, "job_id": config["experiment_name"], "model": "qwen35_08b",
        "display_name": "Qwen3.5-0.8B-Base", "checkpoint": MODEL_ID,
        "revision": MODEL_REVISION, "training_mode": "bf16_lora_text_backbone",
        "dataset": DATASET, "dataset_path": "data/processed/dataset_v2.14.parquet",
        "dataset_rows": len(train), "dataset_sha256": profile.get("dataset_sha256"),
        "validation_dataset": VALIDATION_DATASET, "validation_rows": len(valid),
        "epochs": 10, "early_stopping_patience": config["early_stopping_patience"],
        "checkpoint_selection_metric": "balanced_accuracy", "learning_rate": 5e-5,
        "optimizer": config["optimizer"], "lr_scheduler_type": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"], "weight_decay": config["weight_decay"],
        "compute_dtype": "bfloat16_when_supported", "quantization": "none",
        "lora_rank": config["lora_rank"], "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"], "lora_target_modules": config["lora_target_modules"],
        "lora_exclude_modules": config.get("lora_exclude_modules", []),
        "text_only_backbone": True, "use_chat_template": True,
        "sequence_length": "automatic_from_training_data_and_full_v0",
        "batch_size": config["batch_size"], "auto_batch_size": config["auto_batch_size"],
        "max_batch_size": config["max_batch_size"],
        "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "status": "queued", "config": str(config_path.relative_to(ROOT)),
        "command": command, "environment": {"HF_HUB_CACHE": str(HF_HOME),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "log": "01_qwen35_08b_v214.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": QUEUE_TAG,
        "status": "queued_waiting_for_previous_queue", "created_at_utc": now(),
        "description": "Train Qwen3.5-0.8B-Base with BF16 LoRA on all V2.14 rows for up to 10 epochs, using Qwen's official chat template; validate on full V0.",
        "execution_policy": "wait for the preceding queue to finish, wait for GPU release, verify/download the pinned base checkpoint, then train offline",
        "depends_on_queue_json": str(dependency_path.relative_to(ROOT)),
        "dependency_queue_id": dependency.get("queue_id"), "dataset": DATASET,
        "dataset_path": "data/processed/dataset_v2.14.parquet", "dataset_rows": len(train),
        "dataset_sha256": profile.get("dataset_sha256"), "validation_dataset": VALIDATION_DATASET,
        "validation_rows": len(valid), "model": MODEL_ID, "revision": MODEL_REVISION,
        "model_author_recipe": "LoRA-style PEFT with official chat template; numeric values follow established NLM Qwen3 LoRA settings because the model card does not specify them",
        "results": [job], "active_job": None,
        "retry_of_queue_id": dependency.get("queue_id") if retry_failed else None}
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
    parser.add_argument("--retry-failed", action="store_true",
                        help="queue a new cache-fix retry after the recorded failed Qwen3.5 attempt")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue(retry_failed=args.retry_failed)
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_previous_queue"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
