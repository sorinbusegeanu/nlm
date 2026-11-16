"""Queue full-BF16 V2.14 encoder jobs behind an existing queue."""
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
HF_HOME = Path.home() / ".cache/huggingface"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DATASET = "v2.14"
VALIDATION_DATASET = "v0"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}
SPECS = {
    "electra_large": {
        "model_id": "google/electra-large-discriminator",
        "revision": "c13c3df7efadc2162f42588bd28eb4e187d602a5",
        "display_name": "ELECTRA-large-discriminator",
        "config_path": "configs/models/electra_large.yaml",
        "required_files": ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"),
    },
    "gte_large": {
        "model_id": "Alibaba-NLP/gte-large-en-v1.5",
        "revision": "104333d6af6f97649377c2afbde10a7704870c7b",
        "code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4",
        "display_name": "GTE-large-en-v1.5",
        "config_path": "configs/models/gte_large.yaml",
        "required_files": ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"),
        "code_files": ("configuration.py", "modeling.py"),
    },
    "mpnet_base": {
        "model_id": "microsoft/mpnet-base",
        "revision": "6996ce1e91bd2a9c7d7f61daec37463394f73f09",
        "display_name": "MPNet-base",
        "config_path": "configs/models/mpnet_base.yaml",
        "required_files": ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"),
    },
    "deberta_gemini": {
        "model_id": "microsoft/deberta-v3-base",
        "revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "display_name": "DeBERTa-v3-base Gemini specialist",
        "config_path": "configs/models/deberta_gemini.yaml",
        "dataset": "v2.14-gemini",
        "validation_dataset": "prompt_grouped_80_20",
        "labels": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
        "required_files": ("config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json"),
    },
    "deberta_glm": {
        "model_id": "microsoft/deberta-v3-base",
        "revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "display_name": "DeBERTa-v3-base GLM specialist",
        "config_path": "configs/models/deberta_glm.yaml",
        "dataset": "v2.14-glm",
        "validation_dataset": "prompt_grouped_80_20",
        "labels": ["glm-5.3", "glm-5.3-flash"],
        "required_files": ("config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json"),
    },
}


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
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
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

    env = {**os.environ, "HF_HOME": str(HF_HOME), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        try:
            processes, gpu = active_resources()
        except (OSError, subprocess.SubprocessError) as exc:
            state.update(status="blocked", error=f"resource check failed: {exc}", updated_at_utc=now())
            save(queue_path, state)
            return 1
        state.update(status="waiting_for_gpu_release", active_training_processes=processes, gpu_compute_pids=gpu,
                     last_resource_check_at_utc=now(), updated_at_utc=now())
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
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            job["pid"] = process.pid
            save(queue_path, state)
            code = process.wait()
    except Exception as exc:
        code = 1
        job["error"] = f"{type(exc).__name__}: {exc}"
    if code == 0 and job.get("post_command"):
        try:
            with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n[{now()}] Starting post-training evaluation: {' '.join(job['post_command'])}\n")
                process = subprocess.Popen(job["post_command"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                job["post_pid"] = process.pid
                save(queue_path, state)
                post_code = process.wait()
            job["postprocess_exit_code"] = post_code
            if post_code != 0:
                code = post_code
                job["error"] = f"post-training V0 evaluation exited {post_code}"
        except Exception as exc:
            code = 1
            job["error"] = f"post-training evaluation failed: {type(exc).__name__}: {exc}"
    job.update(status="completed" if code == 0 else "failed", exit_code=code,
                elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
    state.update(status=job["status"], active_job=None, exit_code=code, finished_at_utc=now(), updated_at_utc=now())
    save(queue_path, state)
    return code


def create_queue(model: str, dependency_path: Path) -> Path:
    spec = SPECS[model]
    dataset = spec.get("dataset", DATASET)
    validation_dataset = spec.get("validation_dataset", VALIDATION_DATASET)
    if not dependency_path.is_absolute():
        dependency_path = ROOT / dependency_path
    if not dependency_path.is_file():
        raise FileNotFoundError(f"previous queue not found: {dependency_path}")
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    if dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError(f"dependency queue is already terminal: {dependency_path}")

    snapshot = HF_HOME / "hub" / f"models--{spec['model_id'].replace('/', '--')}" / "snapshots" / spec["revision"]
    missing = [name for name in spec["required_files"] if not (snapshot / name).resolve().is_file()]
    if missing:
        raise FileNotFoundError(f"pinned local model files missing under {snapshot}: {missing}")
    if spec.get("code_revision"):
        code_snapshot = HF_HOME / "hub" / "models--Alibaba-NLP--new-impl" / "snapshots" / spec["code_revision"]
        missing_code = [name for name in spec["code_files"] if not (code_snapshot / name).resolve().is_file()]
        if missing_code:
            raise FileNotFoundError(f"pinned local GTE custom code missing under {code_snapshot}: {missing_code}")
    from src.pipeline import _training_partitions
    train, valid = _training_partitions(dataset, validation_dataset, ROOT / "data/processed")
    if model in {"deberta_gemini", "deberta_glm"}:
        expected_labels = set(spec["labels"])
        total_rows = len(train) + len(valid)
        if set(train.model.astype(str)) != expected_labels or set(valid.model.astype(str)) != expected_labels:
            raise ValueError(f"{dataset} prompt split does not contain all specialist classes on both sides")
        if not 0.18 <= len(valid) / total_rows <= 0.22:
            raise ValueError(f"{dataset} validation rows do not form the requested approximate 20 percent")
        if len(valid) != int(json.loads((ROOT / f"data/processed/dataset_profile_{dataset}.json").read_text())["split_validation_rows"]):
            raise ValueError(f"{dataset} split rows differ from its dataset profile")
    elif len(train) != 231336 or len(valid) != 2699 or train.model.nunique() != 18:
        raise ValueError(f"unexpected V2.14/full-V0 rows/classes: {len(train)}/{len(valid)}/{train.model.nunique()}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"{model}_{dataset.replace('.', '')}_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / spec["config_path"]).read_text(encoding="utf-8"))
    if config.get("checkpoint") != spec["model_id"] or config.get("revision") != spec["revision"]:
        raise ValueError(f"{model} model ID or pinned revision differs from its config")
    if config.get("code_revision") != spec.get("code_revision"):
        raise ValueError(f"{model} custom code revision differs from the pinned revision")
    expected = {"epochs": 3 if model in {"deberta_gemini", "deberta_glm"} else 5,
                "learning_rate": 2e-5 if model in {"deberta_gemini", "deberta_glm"} else 1e-5,
                "optimizer": "adamw_torch", "weight_decay": 0.01,
                "lr_scheduler_type": "linear", "warmup_ratio": 0.02, "gradient_accumulation_steps": 1,
                "checkpoint_selection_metric": "balanced_accuracy"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{model} config differs from the requested training settings")
    if model == "gte_large" and config.get("trust_remote_code") is not True:
        raise ValueError("GTE must enable trust_remote_code for its AutoModelForSequenceClassification implementation")
    if not config.get("auto_batch_size") or "max_length" in config:
        raise ValueError(f"{model} must use automatic batch and sequence-length selection")
    if model == "mpnet_base" and config.get("max_sequence_length_cap") != 512:
        raise ValueError("MPNet automatic sequence selection must cap inputs at its supported 512-token limit")
    if model in {"deberta_gemini", "deberta_glm"} and config.get("cap_sequence_length_at_context_window") is not True:
        raise ValueError("DeBERTa specialist sequence selection must respect its model context window")
    config["experiment_name"] = f"{model}_{dataset}_from_scratch_{stamp}"
    config_path = config_dir / f"{model}_{dataset}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train", "--model", model,
               "--dataset", dataset, "--validation-dataset", validation_dataset,
               "--config", str(config_path), "--seed", "2026", "--accept-config-defaults",
               "--refresh-checkpoint-catalog"]
    profile = json.loads((ROOT / f"data/processed/dataset_profile_{dataset}.json").read_text(encoding="utf-8"))
    post_command = None
    if model in {"deberta_gemini", "deberta_glm"}:
        post_command = [PYTHON, "-u", "scripts/evaluate_deberta_specialist_v0.py",
                        "--model", model, "--dataset", dataset,
                        "--experiment-name", config["experiment_name"]]
    all_dataset_rows = len(train) + len(valid)
    is_specialist = model in {"deberta_gemini", "deberta_glm"}
    job = {"index": 1, "job_id": config["experiment_name"], "model": model,
        "display_name": spec["display_name"], "dataset": dataset,
        "dataset_path": f"data/processed/dataset_{dataset}.parquet", "dataset_rows": all_dataset_rows,
        "selection_train_rows": len(train), "dataset_sha256": profile["dataset_sha256"], "validation_dataset": validation_dataset,
        "validation_rows": len(valid), "training_mode": "full_finetune_from_scratch" if is_specialist else "full_from_scratch",
        "compute_dtype": "bfloat16", "checkpoint": spec["model_id"], "revision": spec["revision"],
        "epochs": config["epochs"], "learning_rate": config["learning_rate"], "optimizer": "adamw_torch", "weight_decay": 0.01,
        "lr_scheduler_type": "linear", "warmup_ratio": 0.02, "gradient_accumulation_steps": 1,
        "batch_size": config["batch_size"], "auto_batch_size": True,
        "max_batch_size": config["max_batch_size"], "batch_probe_safety_fraction": config["batch_probe_safety_fraction"],
        "sequence_length": "automatic_from_training_data", "checkpoint_selection_metric": "balanced_accuracy",
        "max_sequence_length_cap": config.get("max_sequence_length_cap"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "code_revision": config.get("code_revision"),
        "status": "queued", "config": str(config_path.relative_to(ROOT)), "command": command,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "log": f"01_{model}_{dataset}.log", "queued_at_utc": now()}
    if post_command:
        job["post_command"] = post_command
    state = {"queue_id": queue_id, "queue_tag": f"{model}_{dataset}", "status": "queued_waiting_for_previous_queue",
        "created_at_utc": now(), "description": (f"Train {spec['display_name']} from scratch on {dataset}; "
            f"select checkpoints using {validation_dataset} and then evaluate on the corresponding full-V0 labels." if is_specialist else
            f"Train {spec['display_name']} from scratch on all V2.14 rows for 5 epochs; validate on full V0."),
        "execution_policy": "wait for the prior serialized queue to finish, then wait for training processes and GPU compute to be idle",
        "depends_on_queue_json": str(dependency_path.relative_to(ROOT)), "dependency_queue_id": dependency.get("queue_id"),
        "dataset": dataset, "dataset_path": f"data/processed/dataset_{dataset}.parquet", "dataset_rows": all_dataset_rows,
        "dataset_sha256": profile["dataset_sha256"], "validation_dataset": validation_dataset,
        "validation_rows": len(valid), "model": spec["model_id"], "revision": spec["revision"],
        "code_revision": config.get("code_revision"),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS))
    parser.add_argument("--depends-on", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    if not args.model or not args.depends_on:
        parser.error("--model and --depends-on are required when creating a queue entry")
    path = create_queue(args.model, args.depends_on)
    print(json.dumps({"queue_dir": str(path), "queue_json": str(path / "queue.json"),
                      "status": "queued_waiting_for_previous_queue"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
