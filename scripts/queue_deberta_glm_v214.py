"""Queue a from-scratch DeBERTa-GLM specialist run with a chosen learning rate."""
from __future__ import annotations

import argparse
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
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DEPENDENCY = QUEUE_ROOT / "deberta_glm_v214-glm_20261118T082203Z/queue.json"
MODEL_ID = "microsoft/deberta-v3-base"
REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DATASET = "v2.14-glm"
VALIDATION = "v0_glm"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def active_resources() -> tuple[list[dict], list[int]]:
    training = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "src.pipeline models train" in parts[1] and Path(__file__).name not in parts[1]:
            try:
                training.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                pass
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi resource check failed: {result.stderr.strip()}")
    gpu_pids = [int(s.strip()) for s in result.stdout.splitlines() if s.strip().isdigit()]
    return training, gpu_pids


def worker(queue_dir: Path) -> int:
    queue_file = queue_dir / "queue.json"
    state = json.loads(queue_file.read_text(encoding="utf-8"))
    job = state["results"][0]
    dep_path = Path(state["depends_on_queue_json"])
    if not dep_path.is_absolute():
        dep_path = ROOT / dep_path
    while True:
        if not dep_path.is_file():
            state.update(status="blocked", error=f"Missing predecessor queue: {dep_path}", updated_at_utc=now())
            save(queue_file, state)
            return 1
        dep = json.loads(dep_path.read_text(encoding="utf-8"))
        state.update(dependency_status=dep.get("status"), last_dependency_check_at_utc=now())
        if dep.get("status") in TERMINAL and dep.get("finished_at_utc"):
            state["dependency_finished_at_utc"] = dep["finished_at_utc"]
            break
        state.update(status="waiting_for_previous_queue", updated_at_utc=now())
        save(queue_file, state)
        time.sleep(20)

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        try:
            train_processes, gpu_pids = active_resources()
        except Exception as exc:
            state.update(status="blocked", error=f"GPU/process check failed: {exc}", updated_at_utc=now())
            save(queue_file, state)
            return 1
        state.update(status="waiting_for_gpu_release", active_training_processes=train_processes,
                     gpu_compute_pids=gpu_pids, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_file, state)
        if not train_processes and not gpu_pids:
            break
        time.sleep(20)

    state.update(status="training_running", active_job=job["job_id"], updated_at_utc=now())
    job.update(status="running", started_at_utc=now())
    save(queue_file, state)
    started = time.monotonic()
    code = 1
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now()}] Starting: {' '.join(job['command'])}\n")
            proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = proc.pid
            save(queue_file, state)
            code = proc.wait()
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {exc}"
    job.update(status="completed" if code == 0 else "failed", exit_code=code,
               elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
    state.update(status=job["status"], active_job=None, exit_code=code, finished_at_utc=now(), updated_at_utc=now())
    save(queue_file, state)
    return code


def create_queue(learning_rate: float) -> Path:
    learning_rate = float(learning_rate)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not DEPENDENCY.is_file():
        raise FileNotFoundError(f"Expected previous GLM queue: {DEPENDENCY}")
    previous = json.loads(DEPENDENCY.read_text(encoding="utf-8"))
    if previous.get("status") not in TERMINAL or not previous.get("finished_at_utc"):
        raise RuntimeError(f"Previous GLM queue is not finished: {previous.get('status')}")

    # Prevent accidental duplication of this exact training request.
    for existing in QUEUE_ROOT.rglob("queue.json"):
        try:
            other = json.loads(existing.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(r.get("model") == "deberta_glm" and r.get("dataset") == DATASET
               and abs(float(r.get("learning_rate") or 0) - learning_rate) < 1e-15
               for r in other.get("results", [])):
            raise RuntimeError(f"Matching learning-rate job already exists: {existing}")

    from src.pipeline import _training_partitions
    train, val = _training_partitions(DATASET, VALIDATION, ROOT / "data/processed")
    if len(train) != 24584 or len(val) != 300:
        raise ValueError(f"Unexpected train/validation sizes: {len(train)}/{len(val)}")
    if train.model.value_counts().to_dict() != {"glm-5.3": 12292, "glm-5.3-flash": 12292}:
        raise ValueError("Unexpected GLM training class counts")
    if val.model.value_counts().to_dict() != {"glm-5.3": 150, "glm-5.3-flash": 150}:
        raise ValueError("Unexpected GLM V0 validation class counts")

    cache = Path.home() / ".cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots" / REVISION
    if not all((cache / f).is_file() for f in ("config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json")):
        raise FileNotFoundError(f"Pinned DeBERTa checkpoint is incomplete: {cache}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lr_slug = f"{learning_rate:.0e}".replace("-0", "").replace("-", "")
    queue_id = f"deberta_glm_v214_lr{lr_slug}_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "configs/models/deberta_glm.yaml").read_text(encoding="utf-8"))
    if config.get("model") != "deberta_glm" or config.get("checkpoint") != MODEL_ID or config.get("revision") != REVISION:
        raise ValueError("Current DeBERTa GLM config has unexpected model identity")
    config["learning_rate"] = learning_rate
    lr_label = f"{learning_rate:.0e}"
    config["experiment_name"] = f"deberta_glm_v2.14-glm_lr{lr_label}_from_scratch_{stamp}"
    config_path = config_dir / f"deberta_glm_v2.14-glm_lr{lr_label}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train", "--model", "deberta_glm",
               "--dataset", DATASET, "--validation-dataset", VALIDATION, "--config", str(config_path),
               "--seed", "2026", "--accept-config-defaults", "--refresh-checkpoint-catalog"]
    profile = json.loads((ROOT / "data/processed/dataset_profile_v2.14-glm.json").read_text(encoding="utf-8"))
    job = {"index": 1, "job_id": config["experiment_name"], "model": "deberta_glm",
           "display_name": f"DeBERTa-v3-base GLM specialist (LR {learning_rate:g})", "dataset": DATASET,
           "dataset_path": "data/processed/dataset_v2.14-glm.parquet", "dataset_rows": len(train),
           "dataset_sha256": profile["dataset_sha256"], "validation_dataset": VALIDATION,
           "validation_rows": len(val), "training_mode": "full_from_scratch", "checkpoint": MODEL_ID,
           "revision": REVISION, "epochs": config["epochs"],
           "early_stopping_patience": config["early_stopping_patience"], "learning_rate": learning_rate,
           "optimizer": config["optimizer"], "weight_decay": config["weight_decay"],
           "lr_scheduler_type": config["lr_scheduler_type"], "warmup_ratio": config["warmup_ratio"],
           "batch_size": config["batch_size"], "max_batch_size": config["max_batch_size"],
           "auto_batch_size": config["auto_batch_size"],
           "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
           "gradient_accumulation_steps": config["gradient_accumulation_steps"],
           "sequence_length": "automatic_from_training_data", "checkpoint_selection_metric": "balanced_accuracy",
           "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
           "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
           "log": f"01_deberta_glm_v214_lr{lr_slug}.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": f"deberta_glm_v214_lr{lr_slug}", "status": "queued_waiting_for_previous_queue",
             "created_at_utc": now(), "description": f"From-scratch DeBERTa GLM specialist on full v2.14-glm with LR {learning_rate:g}; selects using the full v0_glm subset.",
             "execution_policy": "wait for the previous GLM run to finish, then wait for GPU compute to be idle",
             "depends_on_queue_json": str(DEPENDENCY.relative_to(ROOT)), "dependency_queue_id": previous.get("queue_id"),
             "dataset": DATASET, "dataset_path": "data/processed/dataset_v2.14-glm.parquet", "dataset_rows": len(train),
             "dataset_sha256": profile["dataset_sha256"], "validation_dataset": VALIDATION,
             "validation_rows": len(val), "model": MODEL_ID, "revision": REVISION, "results": [job], "active_job": None}
    queue_file = queue_dir / "queue.json"
    save(queue_file, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    state["worker_pid"] = proc.pid
    save(queue_file, state)
    return queue_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--learning-rate", type=float, default=4e-5,
                        help="Learning rate for the new from-scratch specialist run (default: 4e-5)")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue(args.learning_rate)
    print(json.dumps({"queue_id": queue_dir.name, "queue_json": str(queue_dir / "queue.json"),
                      "status": "queued_waiting_for_previous_queue"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
