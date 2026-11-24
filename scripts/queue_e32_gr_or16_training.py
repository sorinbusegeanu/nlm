"""Run the five V2.GR.OR16 model jobs sequentially after successful E32 completion."""
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
CATALOG = ROOT / "configs/ensemble_catalog_v2.json"
CATALOG_ID = "E32-GR-OR16"
DATASET = "v2.GR.OR16"
DATASET_ROWS = 231036
DATASET_SHA256 = "9373bbd85f03de1f384cb8ae3b6f9b2b27d576629b8fcf0ab344d966fa266a0a"
VALIDATION_ROWS = 2699
E32_MANIFEST = ROOT / "artifacts/analysis/e32/e32_manifest.json"
E32_ROUTER = ROOT / "artifacts/analysis/e32/e32_router.cbm"
E32_BENCHMARK = ROOT / "artifacts/analysis/e32/inference_benchmark.json"
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled", "blocked"}
MODEL_CONFIGS = {
    "svc": "configs/models/svc.yaml",
    "catboost_stylometry": "configs/models/catboost_stylometry.yaml",
    "mpnet_base": "configs/models/mpnet_base.yaml",
    "embeddinggemma2": "configs/models/embeddinggemma2.yaml",
    "granite4_350m": "configs/models/granite4_350m.yaml",
}
DISPLAY = {
    "svc": "SVC",
    "catboost_stylometry": "CatBoost + Stylometry",
    "mpnet_base": "MPNet-base",
    "embeddinggemma2": "EmbeddingGemma 2",
    "granite4_350m": "Granite 4.0 350M Base",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def update_catalog(queue: dict) -> None:
    catalog = read_catalog()
    matches = [x for x in catalog.get("experiments", []) if x.get("id") == CATALOG_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one {CATALOG_ID} catalogue entry, found {len(matches)}")
    entry = matches[0]
    jobs = queue.get("results", [])
    entry.update({
        "status": "Scored" if queue.get("status") == "completed" else (
            "Failed" if queue.get("status") in {"failed", "completed_with_failures", "blocked"} else "Running"),
        "current_stage": queue.get("status"),
        "progress_source": str((Path("artifacts/training_queues") / queue["queue_id"] / "queue.json")),
        "documentation": "docs/e32_gr_or16_training.md",
        "training_dataset": DATASET,
        "training_dataset_rows": DATASET_ROWS,
        "training_dataset_sha256": DATASET_SHA256,
        "validation_dataset": "v0 (all 2,699 rows)",
        "primary_metric": "V0 balanced accuracy; best checkpoint per model",
        "queue_id": queue["queue_id"],
        "job_order": [job["model"] for job in jobs],
        "job_results": [{k: job.get(k) for k in (
            "index", "model", "status", "experiment_id", "training_duration_seconds",
            "queue_wall_seconds", "validation_balanced_accuracy", "selected_checkpoint")}
            for job in jobs],
        "e32_dependency": "artifacts/analysis/e32/e32_manifest.json",
        "updated_at_utc": now(),
    })
    if queue.get("status") == "completed":
        entry["completed_at_utc"] = queue.get("finished_at_utc")
    if queue.get("status") in {"failed", "completed_with_failures", "blocked"}:
        entry["failure"] = queue.get("error") or queue.get("failed_job")
    tmp = CATALOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CATALOG)


def write_report(queue: dict, queue_dir: Path) -> None:
    queue_dir = queue_dir.resolve()
    lines = [
        "# V2.GR.OR16 Sequential Training Queue",
        "",
        f"Queue: `{queue['queue_id']}`  ",
        f"Status: **{queue.get('status')}**  ",
        f"Dependency: E32 status **{queue.get('dependency_status', 'pending')}**  ",
        f"Dataset: `{DATASET}` ({DATASET_ROWS:,} rows; SHA-256 `{DATASET_SHA256}`)  ",
        f"Validation: full V0 ({VALIDATION_ROWS:,} rows)  ",
        "Checkpoint selection: validation balanced accuracy.",
        "",
        "Jobs run sequentially after E32's scored artifacts are verified and the GPU/training process list is idle.",
        "SVC and CatBoost use their existing single-fit procedures. MPNet uses its existing five-epoch full fine-tune recipe.",
        "EmbeddingGemma 2's registered recipe is LoRA with 10 configured epochs (the existing config is epoch-based); it is preserved as-is.",
        "Granite 4.0 350M uses its existing LoRA recipe with epochs changed to the requested five.",
        "",
        "| Order | Model | Epochs/configuration | Status | Duration (s) | V0 balanced accuracy | Selected checkpoint |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for job in queue["results"]:
        epochs = (str(job["epochs"]) if job.get("epochs") is not None
                  else "existing single-fit procedure")
        duration = job.get("training_duration_seconds")
        ba = job.get("validation_balanced_accuracy")
        checkpoint = job.get("selected_checkpoint")
        lines.append(
            f"| {job['index']} | {job['model']} | {epochs} | {job.get('status')} | "
            f"{f'{duration:.1f}' if isinstance(duration, (int, float)) else '—'} | "
            f"{f'{ba:.4%}' if isinstance(ba, (int, float)) else '—'} | "
            f"{f'`{checkpoint}`' if checkpoint else '—'} |"
        )
    lines.extend(["", f"Queue state: `{queue_dir.relative_to(ROOT) / 'queue.json'}`", ""])
    (ROOT / "docs/e32_gr_or16_training.md").write_text("\n".join(lines), encoding="utf-8")


def verify_e32_success() -> None:
    catalog = read_catalog()
    entries = [x for x in catalog.get("experiments", []) if x.get("id") == "E32"]
    if len(entries) != 1 or entries[0].get("status") != "Scored":
        raise RuntimeError("E32 is not marked Scored in the ensemble catalogue")
    for path in (E32_MANIFEST, E32_ROUTER, E32_BENCHMARK):
        if not path.is_file():
            raise FileNotFoundError(f"E32 success artifact missing: {path}")
    manifest = json.loads(E32_MANIFEST.read_text(encoding="utf-8"))
    if int(manifest.get("checkpoint_count", -1)) != 60:
        raise RuntimeError("E32 manifest does not record the completed 60-checkpoint result")
    benchmark = json.loads(E32_BENCHMARK.read_text(encoding="utf-8"))
    if not benchmark:
        raise RuntimeError("E32 benchmark artifact is empty")


def active_resources() -> tuple[list[dict], list[int]]:
    processes = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "-m src.pipeline models train" in parts[1]:
            try:
                processes.append({"pid": int(parts[0]), "command": parts[1][:400]})
            except ValueError:
                pass
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
    gpu = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu


def find_manifest(job: dict, started_epoch: float) -> dict:
    candidates = []
    for path in (ROOT / "artifacts/predictions").glob("*_manifest.json"):
        try:
            if path.stat().st_mtime < started_epoch - 5:
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("model") != job["model"] or value.get("training_dataset") != DATASET:
                continue
            if value.get("validation_dataset") != "v0":
                continue
            candidates.append((path.stat().st_mtime, value, path))
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(f"no published training manifest found for {job['model']} after job start")
    _, manifest, path = max(candidates, key=lambda x: x[0])
    manifest["_manifest_path"] = str(path.relative_to(ROOT))
    return manifest


def metrics_from_manifest(manifest: dict) -> dict:
    metrics = manifest.get("validation_metrics") or manifest.get("metrics") or {}
    ba = metrics.get("balanced_accuracy", manifest.get("balanced_accuracy"))
    if ba is None:
        raise ValueError("training manifest has no validation balanced accuracy")
    return {"balanced_accuracy": float(ba), "accuracy": metrics.get("accuracy")}


def wait_for_resources(queue: dict, queue_path: Path) -> None:
    while True:
        processes, gpu = active_resources()
        queue.update(status="waiting_for_gpu_release", active_training_processes=processes,
                     gpu_compute_pids=gpu, last_resource_check_at_utc=now(), updated_at_utc=now())
        save(queue_path, queue)
        write_report(queue, queue_path.parent)
        if not processes and not gpu:
            return
        time.sleep(20)


def worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    try:
        while True:
            catalog = read_catalog()
            e32 = next((x for x in catalog.get("experiments", []) if x.get("id") == "E32"), None)
            queue["dependency_status"] = e32.get("status") if e32 else None
            queue["updated_at_utc"] = now()
            save(queue_path, queue)
            if e32 and e32.get("status") == "Scored":
                verify_e32_success()
                break
            if not e32 or e32.get("status") in {"Failed", "Cancelled"}:
                raise RuntimeError(f"E32 did not complete successfully; catalogue status={queue['dependency_status']}")
            queue["status"] = "waiting_for_e32_success"
            write_report(queue, queue_dir)
            time.sleep(30)

        queue["e32_verified_at_utc"] = now()
        queue["status"] = "queued"
        update_catalog(queue)
        save(queue_path, queue)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
        for job in queue["results"]:
            if job["status"] != "queued":
                continue
            wait_for_resources(queue, queue_path)
            job["status"] = "running"
            job["started_at_utc"] = now()
            queue.update(status="running", active_job=job["job_id"], updated_at_utc=now())
            update_catalog(queue)
            save(queue_path, queue)
            write_report(queue, queue_dir)
            started = time.monotonic()
            started_epoch = time.time()
            log_path = queue_dir / job["log"]
            try:
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    log.write(f"[{now()}] START {' '.join(job['command'])}\n")
                    process = subprocess.Popen(job["command"], cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    job["pid"] = process.pid
                    save(queue_path, queue)
                    code = process.wait()
                if code != 0:
                    raise subprocess.CalledProcessError(code, job["command"])
                manifest = find_manifest(job, started_epoch)
                metrics = metrics_from_manifest(manifest)
                selected = (manifest.get("selected_checkpoint_path") or manifest.get("best_checkpoint_path")
                            or manifest.get("checkpoint_path"))
                if not selected:
                    raise ValueError("training manifest does not identify its selected checkpoint")
                job.update(status="completed", exit_code=0, finished_at_utc=now(),
                           queue_wall_seconds=round(time.monotonic() - started, 2),
                           training_duration_seconds=float(manifest.get("fit_seconds", 0.0)),
                           validation_balanced_accuracy=metrics["balanced_accuracy"],
                           validation_accuracy=metrics.get("accuracy"),
                           selected_checkpoint=selected,
                           experiment_id=manifest.get("experiment_id"),
                           manifest=manifest["_manifest_path"])
                queue["last_completed_job"] = job["job_id"]
            except Exception as exc:
                job.update(status="failed", exit_code=getattr(exc, "returncode", 1),
                           error=f"{type(exc).__name__}: {exc}", finished_at_utc=now(),
                           queue_wall_seconds=round(time.monotonic() - started, 2))
                queue["failed_job"] = job["job_id"]
                queue["error"] = job["error"]
                for pending in queue["results"]:
                    if pending["status"] == "queued":
                        pending.update(status="blocked", reason=f"prior job {job['job_id']} failed")
                queue.update(status="completed_with_failures", active_job=None, exit_code=1,
                             finished_at_utc=now(), updated_at_utc=now())
                update_catalog(queue)
                save(queue_path, queue)
                write_report(queue, queue_dir)
                return 1
            queue.update(active_job=None, updated_at_utc=now())
            update_catalog(queue)
            save(queue_path, queue)
            write_report(queue, queue_dir)

        queue.update(status="completed", active_job=None, exit_code=0,
                     finished_at_utc=now(), updated_at_utc=now())
        update_catalog(queue)
        save(queue_path, queue)
        write_report(queue, queue_dir)
        return 0
    except Exception as exc:
        queue.update(status="blocked", error=f"{type(exc).__name__}: {exc}",
                     updated_at_utc=now())
        update_catalog(queue)
        save(queue_path, queue)
        write_report(queue, queue_dir)
        return 1


def create_queue() -> Path:
    from src.pipeline import _training_partitions

    dataset_path = ROOT / "data/processed/dataset_v2.GR.OR16.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.GR.OR16.json"
    if not dataset_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("registered v2.GR.OR16 parquet/profile is missing")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if int(profile.get("rows", -1)) != DATASET_ROWS or profile.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("v2.GR.OR16 profile differs from the verified dataset snapshot")
    train, valid = _training_partitions(DATASET, "v0", ROOT / "data/processed")
    if len(train) != DATASET_ROWS or len(valid) != VALIDATION_ROWS or train.model.nunique() != 18:
        raise ValueError(f"unexpected training/validation partition: {len(train)}/{len(valid)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"e32_gr_or16_models_{stamp}"
    queue_dir = QUEUE_ROOT / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index, (model, relative) in enumerate(MODEL_CONFIGS.items(), start=1):
        config = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        if config.get("model") != model:
            raise ValueError(f"config model mismatch for {model}")
        if model in {"mpnet_base", "granite4_350m", "embeddinggemma2"} and not config.get("auto_batch_size"):
            raise ValueError(f"automatic batch sizing is not enabled for {model}")
        if model == "granite4_350m":
            config["epochs"] = 5
        if model in {"catboost_stylometry", "mpnet_base", "embeddinggemma2", "granite4_350m"}:
            config["checkpoint_selection"] = "highest_validation_balanced_accuracy"
            config["checkpoint_selection_metric"] = "balanced_accuracy"
        config_path = config_dir / f"{model}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                   "--model", model, "--dataset", DATASET, "--validation-dataset", "v0",
                   "--config", str(config_path), "--seed", "2026",
                   "--accept-config-defaults", "--refresh-checkpoint-catalog"]
        results.append({
            "index": index, "job_id": f"{model}_{DATASET}_seed2026_{stamp}",
            "model": model, "display_name": DISPLAY[model], "dataset": DATASET,
            "dataset_rows": DATASET_ROWS, "dataset_sha256": DATASET_SHA256,
            "validation_dataset": "v0", "validation_rows": VALIDATION_ROWS,
            "epochs": config.get("epochs"), "training_mode": config.get("training_mode", "single_fit"),
            "checkpoint_selection_metric": "balanced_accuracy", "auto_batch_size": config.get("auto_batch_size"),
            "config": str(config_path.relative_to(ROOT)), "command": command,
            "status": "queued", "log": f"{index:02d}_{model}.log", "queued_at_utc": now(),
        })
    queue = {
        "queue_id": queue_id, "queue_tag": CATALOG_ID, "status": "queued_waiting_for_e32_success",
        "created_at_utc": now(), "description": "Sequential SVC, CatBoost + Stylometry, MPNet-base, EmbeddingGemma 2 and Granite 4.0 350M training on v2.GR.OR16 with full-V0 balanced-accuracy selection.",
        "depends_on_experiment": "E32", "depends_on_artifacts": [
            str(E32_MANIFEST.relative_to(ROOT)), str(E32_ROUTER.relative_to(ROOT)),
            str(E32_BENCHMARK.relative_to(ROOT))],
        "dataset": DATASET, "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_rows": DATASET_ROWS, "dataset_sha256": DATASET_SHA256,
        "validation_dataset": "v0", "validation_rows": VALIDATION_ROWS,
        "execution_policy": "wait for successful E32 catalogue/artifacts; then run jobs sequentially, waiting for GPU and training-process idle between jobs; stop the suite after the first failure",
        "results": results, "active_job": None,
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, queue)
    write_report(queue, queue_dir)

    catalog = read_catalog()
    if any(x.get("id") == CATALOG_ID for x in catalog.get("experiments", [])):
        raise ValueError(f"catalogue already contains {CATALOG_ID}")
    catalog.setdefault("experiments", []).append({
        "id": CATALOG_ID, "name": "V2.GR.OR16 five-model training suite",
        "category": "Supervised classifiers",
        "description": "Sequentially train SVC, CatBoost + Stylometry, MPNet-base, EmbeddingGemma 2 and Granite 4.0 350M on the external-only V2.GR.OR16 dataset after successful E32 completion; select checkpoints using full-V0 balanced accuracy.",
        "status": "Queued", "baseline_experiment": "E32", "training_dataset": DATASET,
        "training_dataset_rows": DATASET_ROWS, "training_dataset_sha256": DATASET_SHA256,
        "validation_dataset": "v0 (all 2,699 rows)",
        "primary_metric": "Validation balanced accuracy and selected checkpoint per job",
        "current_stage": "Queued after E32", "progress_source": str(queue_path.relative_to(ROOT)),
        "documentation": "docs/e32_gr_or16_training.md", "queue_id": queue_id,
        "job_order": [j["model"] for j in results],
        "checkpoint_selection": "highest validation balanced accuracy",
        "e32_dependency": str(E32_MANIFEST.relative_to(ROOT)),
    })
    tmp = CATALOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CATALOG)
    return queue_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    queue_dir = create_queue()
    worker_log = (queue_dir / "queue_worker.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen([PYTHON, "-u", str(Path(__file__).resolve()), "--worker", str(queue_dir)],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=worker_log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    queue_path = queue_dir / "queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["worker_pid"] = process.pid
    save(queue_path, queue)
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_path),
                      "worker_pid": process.pid, "status": queue["status"],
                      "jobs": [x["model"] for x in queue["results"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
