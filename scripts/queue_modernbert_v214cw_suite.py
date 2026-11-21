"""Queue a ModernBERT continuation plus from-scratch HN and CW comparisons."""
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
SOURCE_EXPERIMENT = (
    "modernbert_large_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_"
    "cfgbb2e9041_code3458579"
)
SOURCE_MANIFEST = ROOT / "artifacts/predictions" / f"{SOURCE_EXPERIMENT}_manifest.json"
DEPENDENCY_QUEUE = ROOT / "artifacts/training_queues" / (
    "granite4_h_micro_v214hn_from_n0110_20261119T055835Z/queue.json"
)
QUEUE_ROOT = ROOT / "artifacts/training_queues"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}
LR = 1e-5 / 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
                continue
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=20)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def _validate_inputs() -> tuple[dict, dict[str, dict]]:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(f"source experiment manifest missing: {SOURCE_MANIFEST}")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("experiment_id") != SOURCE_EXPERIMENT or source.get("training_dataset") != "v2.14":
        raise ValueError("pinned source experiment does not identify the expected V2.14 ModernBERT run")
    source_config = source.get("config") or {}
    starting_lr = float(source_config.get("learning_rate", 0))
    if starting_lr != 1e-5 or not abs(LR - starting_lr / 3) < 1e-15:
        raise ValueError("source starting learning rate is unexpected")
    checkpoint = Path(source.get("best_checkpoint_path") or source.get("checkpoint_path", ""))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    trainer_state = Path(source.get("continuation_checkpoint_path", ""))
    if not trainer_state.is_absolute():
        trainer_state = ROOT / trainer_state
    required = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")
    for path in (checkpoint, trainer_state):
        if not path.is_dir():
            raise FileNotFoundError(f"source checkpoint state is missing: {path}")
    for filename in required:
        if not (checkpoint / filename).is_file() and not (trainer_state / filename).is_file():
            raise FileNotFoundError(f"source checkpoint is missing {filename}")

    from src.pipeline import _continuation_source_matches, _partition_sha256, _training_partitions
    train_cw, valid = _training_partitions("v2.14CW", "v0", ROOT / "data/processed")
    train_hn, valid_hn = _training_partitions("v2.14HN", "v0", ROOT / "data/processed")
    if len(valid) != 2699 or len(valid_hn) != 2699 or train_cw.model.nunique() != 18 or train_hn.model.nunique() != 18:
        raise ValueError("the queued variants must have all 18 labels and full 2,699-row V0 validation")
    if _partition_sha256(train_cw) != source.get("training_dataset_sha256"):
        raise ValueError("V2.14CW training rows do not exactly match the pinned source run")
    if not _continuation_source_matches(source, "v2.14CW", "v0", train_cw, valid,
                                        source.get("training_dataset_sha256"),
                                        allow_cross_dataset_source=True):
        raise ValueError("source checkpoint does not match the V2.14CW/full-V0 continuation partitions")
    config = yaml.safe_load((ROOT / "configs/models/modernbert_large.yaml").read_text(encoding="utf-8"))
    if config.get("checkpoint") != source.get("checkpoint_id") and config.get("checkpoint") != source_config.get("checkpoint"):
        raise ValueError("ModernBERT model config differs from the source checkpoint architecture")
    if config.get("revision") != source.get("model_revision"):
        raise ValueError("ModernBERT model revision differs from the source run")
    dependency = json.loads(DEPENDENCY_QUEUE.read_text(encoding="utf-8"))
    if dependency.get("status") in TERMINAL and dependency.get("finished_at_utc"):
        raise ValueError("dependency queue is already terminal; refusing to add a stale dependent queue")
    datasets = {}
    for name in ("v2.14CW", "v2.14HN"):
        profile_id = "v2.14" if name == "v2.14CW" else name
        profile = json.loads((ROOT / "data/processed" / f"dataset_profile_{profile_id}.json").read_text())
        datasets[name] = {"rows": int(profile["rows"]),
                          "sha256": profile["dataset_sha256"],
                          "class_counts": profile.get("class_counts", {})}
    return source, datasets


def create_queue() -> Path:
    source, datasets = _validate_inputs()
    queue_id = "modernbert_v214cw_suite_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    base_config = yaml.safe_load((ROOT / "configs/models/modernbert_large.yaml").read_text())

    continuation_config = {"model": "modernbert_large", "epochs": 5,
                           "early_stopping_patience": 2,
                           "learning_rate": LR,
                           "resume_optimizer_state_with_lr_override": True}
    continuation_path = config_dir / "modernbert_large_v214cw_continue.yaml"
    continuation_path.write_text(yaml.safe_dump(continuation_config, sort_keys=False))

    hn_config = dict(base_config)
    hn_config["weighted_sampling"] = True
    hn_path = config_dir / "modernbert_large_v214hn_scratch.yaml"
    hn_path.write_text(yaml.safe_dump(hn_config, sort_keys=False))
    cw_config = dict(base_config)
    cw_path = config_dir / "modernbert_large_v214cw_scratch.yaml"
    cw_path.write_text(yaml.safe_dump(cw_config, sort_keys=False))

    common = ["--validation-dataset", "v0", "--seed", "2026", "--accept-config-defaults"]
    commands = [
        [PYTHON, "-u", "-m", "src.pipeline", "models", "continue-encoder",
         "--model", "modernbert_large", "--dataset", "v2.14CW",
         "--from-experiment", SOURCE_EXPERIMENT, "--epochs", "5",
         "--learning-rate", repr(LR), "--config", str(continuation_path),
         "--allow-cross-dataset-source", "--accept-config-defaults"],
        [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
         "--model", "modernbert_large", "--dataset", "v2.14HN",
         "--config", str(hn_path), *common],
        [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
         "--model", "modernbert_large", "--dataset", "v2.14CW",
         "--config", str(cw_path), *common],
    ]
    jobs = []
    definitions = [
        ("modernbert_v214cw_continue_from_n0127", "v2.14CW", "continued", continuation_path),
        ("modernbert_v214hn_scratch", "v2.14HN", "scratch_weighted_sampling", hn_path),
        ("modernbert_v214cw_scratch", "v2.14CW", "scratch_class_weighted_loss", cw_path),
    ]
    for index, ((job_id, dataset, mode, config_path), command) in enumerate(zip(definitions, commands), 1):
        job_id = f"{job_id}_{queue_id}"
        jobs.append({"index": index, "job_id": job_id, "model": "modernbert_large",
                     "dataset": dataset, "dataset_path": f"data/processed/dataset_{dataset}.parquet" if dataset != "v2.14CW" else "data/processed/dataset_v2.14.parquet",
                     "dataset_rows": datasets[dataset]["rows"], "dataset_sha256": datasets[dataset]["sha256"],
                     "validation_dataset": "v0", "validation_rows": 2699,
                     "training_mode": mode, "epochs": 5, "early_stopping_patience": 2,
                     "source_experiment_id": SOURCE_EXPERIMENT if index == 1 else None,
                     "source_checkpoint_path": source.get("best_checkpoint_path") if index == 1 else None,
                     "source_trainer_state_path": source.get("continuation_checkpoint_path") if index == 1 else None,
                     "source_learning_rate": source["config"]["learning_rate"] if index == 1 else None,
                     "learning_rate": LR if index == 1 else base_config["learning_rate"],
                     "optimizer_scheduler_policy": "restore optimizer moments and scheduler position; rebase next LR to requested value" if index == 1 else "fresh optimizer and scheduler",
                     "config": str(config_path.relative_to(ROOT)), "command": command,
                     "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                     "log": f"{index:02d}_{dataset}_modernbert_large.log", "status": "queued",
                     "queued_at_utc": now()})
    state = {"queue_id": queue_id, "queue_tag": "modernbert_v214cw_suite", "status": "queued_waiting_for_previous_queue",
             "created_at_utc": now(), "description": "Continue the pinned V2.14 ModernBERT state on V2.14CW, then run ModernBERT from scratch on V2.14HN and V2.14CW.",
             "execution_policy": "wait for the existing Granite V2.14HN queue to finish, wait for training and GPU compute to be idle, then execute these three jobs sequentially",
             "depends_on_queue_json": str(DEPENDENCY_QUEUE.relative_to(ROOT)),
             "dependency_queue_id": json.loads(DEPENDENCY_QUEUE.read_text())["queue_id"],
             "dataset": "multiple", "validation_dataset": "v0", "source_experiment_id": SOURCE_EXPERIMENT,
             "results": jobs, "active_job": None, "worker_pid": None}
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    return queue_dir


def worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
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
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    for job in state["results"]:
        # Resuming a stopped suite must leave completed/stopped jobs untouched
        # and run only the jobs that are explicitly queued again.
        if job.get("status") != "queued":
            continue
        while True:
            try:
                processes, gpu = resource_state()
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
                log.write(f"[{now()}] Starting offline job: {' '.join(job['command'])}\n")
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job["pid"] = process.pid
                save(queue_path, state)
                code = process.wait()
        except Exception as exc:
            code = 1
            job["error"] = f"{type(exc).__name__}: {exc}"
        job.update(status="completed" if code == 0 else "failed", exit_code=code,
                    elapsed_seconds=round(time.monotonic() - started, 2), finished_at_utc=now())
        if code != 0:
            for pending in state["results"]:
                if pending["status"] == "queued":
                    pending.update(status="blocked", reason=f"prior job {job['job_id']} failed")
            state.update(status="completed_with_failures", active_job=None, exit_code=code,
                         finished_at_utc=now(), updated_at_utc=now())
            save(queue_path, state)
            return code
        save(queue_path, state)
    state.update(status="completed", active_job=None, exit_code=0, finished_at_utc=now(), updated_at_utc=now())
    save(queue_path, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.create:
        parser.error("use --create or --worker QUEUE_DIR")
    queue_dir = create_queue()
    log = (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    state = json.loads((queue_dir / "queue.json").read_text())
    state["worker_pid"] = process.pid
    save(queue_dir / "queue.json", state)
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "worker_pid": process.pid, "status": "queued_waiting_for_previous_queue",
                      "jobs": [job["job_id"] for job in state["results"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
