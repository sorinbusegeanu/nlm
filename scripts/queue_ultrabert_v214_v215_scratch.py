"""Queue a fresh UltraBERT fit on the complete V2.14 + V2.15 corpus."""
from __future__ import annotations

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
DATA_DIR = ROOT / "data/processed"
DATASET = "v2.14_plus_v2.15_extrato_14"
DATA_PATH = DATA_DIR / f"dataset_{DATASET}.parquet"
PROFILE_PATH = DATA_DIR / f"dataset_profile_{DATASET}.json"
QUEUE_ROOT = ROOT / "artifacts/training_queues"
DEPENDENCIES = [
    ROOT / "artifacts/training_queues/v2_14_frozen_style_20261113T113511Z/queue.json",
    ROOT / "artifacts/training_queues/ultrabert_v2.15_extrato_14_continue_20261113T134428Z/queue.json",
    ROOT / "artifacts/training_queues/ultrabert_v214pf80_scratch_20261113T130631Z/queue.json",
]
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_dataset() -> dict:
    v214_path = DATA_DIR / "dataset_v2.14.parquet"
    v215_path = DATA_DIR / "dataset_v2.15_extrato_14.parquet"
    for path in (v214_path, v215_path, DATA_DIR / "dataset_v0.parquet"):
        if not path.is_file():
            raise FileNotFoundError(path)
    v214 = pd.read_parquet(v214_path)
    v215 = pd.read_parquet(v215_path)
    required = {"id", "prompt_id", "text", "model"}
    for name, rows in (("v2.14", v214), ("v2.15_extrato_14", v215)):
        if required - set(rows.columns):
            raise ValueError(f"{name} missing required columns: {sorted(required - set(rows.columns))}")
        if rows[list(required)].isna().any().any() or rows.id.duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs or missing training fields")
    for col in ("id", "prompt_id"):
        overlap = set(v214[col].astype(str)) & set(v215[col].astype(str))
        if overlap:
            raise ValueError(f"V2.14/V2.15 overlap on {col}: {len(overlap)}")
    norm = lambda s: s.astype(str).str.normalize("NFKC").str.casefold().str.split().str.join(" ")
    if set(norm(v214.text)) & set(norm(v215.text)):
        raise ValueError("V2.14/V2.15 contain duplicate normalized response text")
    if set(v214.model.astype(str)) != set(v215.model.astype(str)):
        raise ValueError("V2.14 and V2.15 class rosters differ")
    combined = pd.concat([v214, v215], ignore_index=True, sort=False)
    combined = combined.sort_values("id", kind="stable").reset_index(drop=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(DATA_PATH, index=False)
    labels = sorted(combined.model.astype(str).unique().tolist())
    profile = {
        "dataset_version": DATASET,
        "dataset_sha256": sha256_file(DATA_PATH),
        "rows": int(len(combined)),
        "classes": labels,
        "class_counts": {str(k): int(v) for k, v in combined.model.astype(str).value_counts().sort_index().items()},
        "training_only": True,
        "external_source_rows_only": True,
        "validation_dataset_version": "v0",
        "validation_rows": 2699,
        "component_datasets": {
            "v2.14": {"path": str(v214_path.relative_to(ROOT)), "rows": len(v214), "sha256": sha256_file(v214_path)},
            "v2.15_extrato_14": {"path": str(v215_path.relative_to(ROOT),), "rows": len(v215), "sha256": sha256_file(v215_path)},
        },
        "creation_method": "concatenate complete v2.14 with incremental v2.15_extrato_14; stable sort by id",
        "seed": 2026,
    }
    PROFILE_PATH.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    from src.pipeline import _training_partitions
    train, valid = _training_partitions(DATASET, "v0", DATA_DIR)
    if len(train) != 277683 or len(valid) != 2699 or labels != sorted(valid.model.astype(str).unique()):
        raise ValueError(f"unexpected partition dimensions/classes: train={len(train)}, valid={len(valid)}")
    return profile


def active_resources() -> tuple[list[dict], list[int]]:
    processes = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        line = line.strip()
        if "src.pipeline models" in line and "queue_ultrabert_v214_v215_scratch.py" not in line:
            try:
                pid, command = line.split(None, 1)
                processes.append({"pid": int(pid), "command": command[:400]})
            except ValueError:
                pass
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=20)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def worker(queue_dir: Path) -> int:
    path = queue_dir / "queue.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    job = state["results"][0]
    dependencies = [ROOT / p for p in state["depends_on_queue_jsons"]]
    while True:
        pending = []
        dep_statuses = {}
        for dep_path in dependencies:
            dep = json.loads(dep_path.read_text(encoding="utf-8"))
            dep_statuses[str(dep_path.relative_to(ROOT))] = dep.get("status")
            if dep.get("status") not in TERMINAL or not dep.get("finished_at_utc"):
                pending.append(str(dep_path.relative_to(ROOT)))
        state.update(dependency_statuses=dep_statuses, pending_dependencies=pending, updated_at_utc=now())
        if not pending:
            break
        state["status"] = "waiting_for_previous_queues"
        save(path, state)
        time.sleep(20)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    while True:
        processes, gpu = active_resources()
        state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(path, state)
        if not processes and not gpu:
            break
        time.sleep(20)
    job.update(status="running", started_at_utc=now())
    state.update(status="running", active_job=job["job_id"], updated_at_utc=now())
    save(path, state)
    started = time.monotonic()
    try:
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
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
    profile = make_dataset()
    v0 = pd.read_parquet(DATA_DIR / "dataset_v0.parquet", columns=["id", "model"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"ultrabert_v214_v215_scratch_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "configs/models/ultrabert380_v2_14_scratch.yaml").read_text())
    config["experiment_name"] = f"ultrabert380_{DATASET}_from_scratch_{stamp}"
    config_path = config_dir / f"ultrabert380_{DATASET}_scratch.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train", "--model", "ultrabert380",
               "--dataset", DATASET, "--validation-dataset", "v0", "--config", str(config_path),
               "--seed", "2026", "--refresh-prediction", "--refresh-checkpoint-catalog", "--accept-config-defaults"]
    deps = [str(p.relative_to(ROOT)) for p in DEPENDENCIES]
    for p in DEPENDENCIES:
        if not p.is_file():
            raise FileNotFoundError(p)
    job = {"index": 1, "job_id": config["experiment_name"], "model": "ultrabert380", "dataset": DATASET,
           "dataset_path": str(DATA_PATH.relative_to(ROOT)), "dataset_rows": profile["rows"],
           "dataset_sha256": profile["dataset_sha256"], "validation_dataset": "v0", "validation_rows": len(v0),
           "training_mode": "from_scratch", "initial_checkpoint": config["checkpoint"],
           "seed": 2026, "config": str(config_path.relative_to(ROOT)), "command": command,
           "status": "queued", "log": "01_ultrabert380_v214_v215_scratch.log", "queued_at_utc": now()}
    state = {"queue_id": queue_id, "queue_tag": "ultrabert_v214_v215_scratch", "status": "queued_waiting_for_previous_queues",
             "created_at_utc": now(), "description": "Fresh UltraBERT training on the complete V2.14 corpus plus incremental V2.15_extrato_14 rows; validate on full V0.",
             "execution_policy": "wait for prerequisite queues to finish, then wait for training processes and GPU compute to be idle",
             "depends_on_queue_jsons": deps, "dataset": DATASET, "dataset_path": str(DATA_PATH.relative_to(ROOT)),
             "dataset_rows": profile["rows"], "dataset_sha256": profile["dataset_sha256"],
             "validation_dataset": "v0", "validation_rows": len(v0), "results": [job], "active_job": None}
    path = queue_dir / "queue.json"
    save(path, state)
    with (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    state["worker_pid"] = proc.pid
    save(path, state)
    return queue_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker.resolve()))
    print(json.dumps({"queue_dir": str(create_queue())}, indent=2))
