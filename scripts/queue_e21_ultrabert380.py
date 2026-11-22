#!/usr/bin/env python3
"""Register E21 UltraBERT runs; use --worker explicitly to execute the queue."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
DATASET = "v2.GR.OR16"
SOURCE_CONFIGS = {
    "E21-A": ROOT / "configs/models/ultrabert380_e21_a.yaml",
    "E21-B": ROOT / "configs/models/ultrabert380_e21_b.yaml",
    "E21-C": ROOT / "configs/models/ultrabert380_e21_c.yaml",
}
DEPENDENCIES = [
    ROOT / "artifacts/training_queues/granite4_350m_v214_openpool16_v0w10_20261120T101917Z/queue.json",
    ROOT / "artifacts/training_queues/granite4_350m_v214_openpool16_v0w10_20261120T180943Z/queue.json",
]
TERMINAL = {"completed", "completed_with_failures", "failed", "stopped", "cancelled"}
MODEL_CACHE = (Path.home() / ".cache/huggingface/hub/models--dmis-lab--UltraBERT-380M/snapshots"
               / "671a38c087233c0cc72c5bfceea6fffa2822f1db")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_report(queue_dir: Path, *, results_available: bool) -> None:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    lines = [
        "# E21 — Group-aware UltraBERT classification", "",
        "## Configuration", "",
        "All three runs use UltraBERT-380M revision `671a38c087233c0cc72c5bfceea6fffa2822f1db`, initialized from the pretrained backbone, trained on all 231,036 V2.GR.OR16 rows, and validated on all 2,699 V0 rows. They share seed 2026, five epochs, learning rate 8e-6, linear schedule, AdamW, automatic batch sizing, batch size 8 starting point, accumulation 2, and the deterministic group-aware sampler. Every original training row receives classification loss once per epoch; rank candidates are selected once per represented label and same-label outputs rotate by epoch.", "",
        "The ranking loss uses the same 18 classification logits. For each group and represented label, it applies a softmax over the selected responses and targets the response whose ground-truth label is that class. Group losses are averaged over groups. E21-A sets lambda to 0, E21-B to 0.1, and E21-C to 0.3. Checkpoint selection uses independent V0 balanced accuracy; all epoch checkpoints and logits are retained.", "",
        f"Queue: `{state['queue_id']}`. Status: `{state['status']}`. Worker started: `{state.get('worker_started', False)}`.", "",
    ]
    if state.get("retry_reason"):
        lines += [f"Retry note: {state['retry_reason']}", ""]
    if not results_available:
        active = state.get("active_job")
        if active:
            progress = f"`{active}` is running; results will be populated as each run and grouped evaluation complete."
        elif state.get("worker_started"):
            progress = "The queue worker is running and waiting to launch the next job; results are pending."
        else:
            progress = "Training has not started; results will be populated by the queue worker after all three jobs and the grouped evaluation complete."
        lines += ["## Results", "", progress, "",
                  "The evaluator reports per-epoch independent argmax, E19-F-A grouped Hungarian, E19-D grouped Hungarian, and true-prompt-group Hungarian metrics. Hungarian assignment uses raw classification logits. Per-epoch outputs and per-label metrics are saved under `artifacts/analysis/e21/`.", ""]
    else:
        summaries = []
        for job in state["results"]:
            path = job.get("evaluation_results")
            if not path or not Path(path).is_file():
                continue
            summaries.append((job, json.loads(Path(path).read_text(encoding="utf-8"))))
        lines += ["## Selected-checkpoint results", "",
                  "| Experiment | Selected epoch | Method | Accuracy | Balanced accuracy | Top-3 accuracy | Corrected | Degraded |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        for job, summary in summaries:
            epoch = summary.get("selected_epoch")
            selected = [item for item in summary.get("metrics_by_epoch", []) if item.get("epoch") == epoch]
            for metric in selected:
                lines.append(f"| {job['experiment']} | {epoch} | {metric['method']} | {metric['accuracy']:.4f} | {metric['balanced_accuracy']:.4f} | {metric.get('top3_accuracy', '—') if metric.get('top3_accuracy') is not None else '—'} | {metric.get('corrected', '—')} | {metric.get('degraded', '—')} |")
            lines.append("")
            lines += [f"### {job['experiment']}", "",
                      f"Fit runtime: {summary.get('fit_seconds')} seconds; selected by independent V0 balanced accuracy at epoch {epoch}.",
                      f"Training loss configuration: `{json.dumps(summary.get('loss_config', {}), sort_keys=True)}`.",
                      f"Full metrics: [`{Path(job['evaluation_results']).name}`]({job['evaluation_results']}); per-label metrics: [`per_label_metrics.csv`]({job['evaluation_dir']}/per_label_metrics.csv); per-epoch predictions: [`predictions_by_epoch.parquet`]({job['evaluation_dir']}/predictions_by_epoch.parquet).", ""]
        if len(summaries) != 3:
            lines += ["Evaluation artifacts for all three runs were not found; see the E21 queue JSON and per-run logs for status.", ""]
    (ROOT / "docs/e21.md").write_text("\n".join(lines), encoding="utf-8")


def validate_preflight() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dataset_path = ROOT / "data/processed/dataset_v2.GR.OR16.parquet"
    profile_path = ROOT / "data/processed/dataset_profile_v2.GR.OR16.json"
    if not dataset_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("V2.GR.OR16 parquet/profile have not both been built")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if digest(dataset_path) != profile.get("dataset_sha256"):
        raise ValueError("V2.GR.OR16 parquet does not match its registered dataset hash")
    if profile.get("rows") != 231036 or profile.get("ranking", {}).get("eligible_groups") != 26347:
        raise ValueError("V2.GR.OR16 profile differs from the audited expected population")
    if not MODEL_CACHE.joinpath("config.json").is_file() or not MODEL_CACHE.joinpath("model.safetensors").is_file():
        raise FileNotFoundError(f"pinned pretrained UltraBERT weights are not cached at {MODEL_CACHE}")
    from src.pipeline import _training_partitions, _validate_training_overrides
    train, valid = _training_partitions(DATASET, "v0", ROOT / "data/processed")
    if len(train) != 231036 or len(valid) != 2699:
        raise ValueError(f"expected 231036 train / 2699 full V0 validation rows; got {len(train)}/{len(valid)}")
    if set(train.model.astype(str)) != set(valid.model.astype(str)) or train.model.nunique() != 18:
        raise ValueError("training and complete V0 validation must use the same 18 labels")
    loaded_configs = {}
    for experiment_id, source_path in SOURCE_CONFIGS.items():
        config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        loaded_configs[experiment_id] = config
        _validate_training_overrides("ultrabert380", config)
        if (config.get("epochs") != 5 or float(config.get("learning_rate")) != 8e-6
                or config.get("lr_scheduler_type") != "linear"
                or not config.get("auto_batch_size")
                or not config.get("group_aware_sampling")
                or config.get("weighted_sampling")):
            raise ValueError(f"{experiment_id} differs from the common E21 configuration")
        if config.get("checkpoint") != "dmis-lab/UltraBERT-380M" or config.get("revision") != "671a38c087233c0cc72c5bfceea6fffa2822f1db":
            raise ValueError(f"{experiment_id} must use the pinned pretrained UltraBERT base")
    common_fields = set(loaded_configs["E21-A"]) - {
        "experiment_name", "group_ranking_lambda", "early_stopping_patience"}
    common = {field: loaded_configs["E21-A"].get(field) for field in common_fields}
    for experiment_id, config in loaded_configs.items():
        actual = {field: config.get(field) for field in common_fields}
        if actual != common:
            raise ValueError(f"{experiment_id} differs from E21-A in common training settings")
    return train, valid, profile


def create_queue() -> Path:
    train, valid, profile = validate_preflight()
    for path in DEPENDENCIES:
        if not path.is_file():
            raise FileNotFoundError(f"training queue prerequisite is missing: {path}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_id = f"e21_group_aware_ultrabert380_{stamp}"
    queue_dir = ROOT / "artifacts/training_queues" / queue_id
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    dependency_states = [json.loads(path.read_text(encoding="utf-8")) for path in DEPENDENCIES]
    jobs = []
    for index, (experiment_id, source_path) in enumerate(SOURCE_CONFIGS.items(), start=1):
        config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        config["experiment_name"] = f"{config['experiment_name']}_{stamp}"
        config_path = config_dir / source_path.name
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                   "--model", "ultrabert380", "--dataset", DATASET,
                   "--validation-dataset", "v0", "--config", str(config_path),
                   "--seed", "2026", "--accept-config-defaults"]
        jobs.append({
            "index": index, "experiment": experiment_id, "job_id": config["experiment_name"],
            "model": "ultrabert380", "dataset": DATASET,
            "dataset_path": "data/processed/dataset_v2.GR.OR16.parquet",
            "dataset_rows": len(train), "dataset_sha256": profile["dataset_sha256"],
            "validation_dataset": "v0", "validation_rows": len(valid),
            "validation_scope": "complete V0, all 2699 rows",
            "initialization": "original pretrained backbone",
            "base_checkpoint": config["checkpoint"], "base_revision": config["revision"],
            "classes": int(train.model.nunique()), "epochs": config["epochs"],
            "learning_rate": config["learning_rate"], "scheduler": config["lr_scheduler_type"],
            "group_ranking_lambda": config["group_ranking_lambda"],
            "loss": {"classification": "cross_entropy", "group_ranking": "within_group_softmax",
                     "ranking_lambda": config["group_ranking_lambda"]},
            "auto_batch_size": config["auto_batch_size"],
            "checkpoint_selection_metric": "validation_balanced_accuracy",
            "save_epoch_checkpoints": True, "seed": 2026,
            "config": str(config_path.relative_to(ROOT)), "command": command,
            "experiment_name": config["experiment_name"],
            "config_sha256": digest(config_path),
            "source_config_sha256": digest(source_path),
            "status": "queued", "log": f"{index:02d}_{experiment_id.lower().replace('-', '_')}.log",
            "queued_at_utc": now(),
        })
    state = {
        "queue_id": queue_id, "queue_tag": "e21_group_aware_ultrabert380",
        "status": "queued_waiting_for_existing_training", "created_at_utc": now(),
        "description": "E21-A/B/C UltraBERT group-aware runs on V2.GR.OR16 with full-V0 validation.",
        "execution_policy": "Wait for both already queued Granite jobs and for training/GPU processes to be idle; execute E21-A, E21-B, E21-C sequentially.",
        "worker_started": False,
        "depends_on_queue_jsons": [str(path.relative_to(ROOT)) for path in DEPENDENCIES],
        "dependency_queue_ids": [item.get("queue_id") for item in dependency_states],
        "dataset": DATASET, "dataset_path": "data/processed/dataset_v2.GR.OR16.parquet",
        "dataset_rows": len(train), "dataset_sha256": profile["dataset_sha256"],
        "validation_dataset": "v0", "validation_rows": len(valid),
        "base_checkpoint": "dmis-lab/UltraBERT-380M",
        "base_revision": "671a38c087233c0cc72c5bfceea6fffa2822f1db",
        "same_seed_and_sampler": True, "results": jobs, "active_job": None,
        "manual_worker_command": f"{PYTHON} scripts/queue_e21_ultrabert380.py --worker {queue_dir}",
    }
    queue_path = queue_dir / "queue.json"
    save(queue_path, state)
    analysis_root = ROOT / "artifacts/analysis/e21"
    analysis_root.mkdir(parents=True, exist_ok=True)
    plan = {"queue_id": queue_id, "queue_json": str(queue_path.relative_to(ROOT)),
            "status": state["status"], "dataset_id": DATASET,
            "dataset_sha256": profile["dataset_sha256"], "training_rows": len(train),
            "validation_dataset": "v0", "validation_rows": len(valid),
            "ranking_eligible_groups": profile["ranking"]["eligible_groups"],
            "ranking_eligible_responses": profile["ranking"]["eligible_responses"],
            "group_label_combinations": profile["ranking"]["unique_group_label_combinations"],
            "pairwise_comparisons": profile["ranking"]["potential_unordered_label_pair_comparisons_per_full_group_pass"],
            "base_model": {"checkpoint": state["base_checkpoint"], "revision": state["base_revision"],
                           "cached_weights_sha256": digest(MODEL_CACHE / "model.safetensors"),
                           "cached_weights_bytes": (MODEL_CACHE / "model.safetensors").stat().st_size},
            "sampler": "GroupAwareBatchSampler; all rows once/epoch; one row per label per eligible group; epoch rotation",
            "evaluation_methods": ["independent_argmax", "E19-F-A Hungarian", "E19-D Hungarian",
                                   "true prompt_id Hungarian"],
            "experiments": jobs}
    (analysis_root / f"{queue_id}_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    write_report(queue_dir, results_available=False)
    return queue_dir


def active_resources() -> tuple[list[dict], list[int]]:
    processes = []
    for line in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
        line = line.strip()
        if "src.pipeline models" in line and Path(__file__).name not in line:
            try:
                pid, command = line.split(None, 1)
                processes.append({"pid": int(pid), "command": command[:400]})
            except ValueError:
                continue
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=20)
    gpu_pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return processes, gpu_pids


def run_worker(queue_dir: Path) -> int:
    queue_path = queue_dir / "queue.json"
    state = json.loads(queue_path.read_text(encoding="utf-8"))
    if state.get("queue_tag") != "e21_group_aware_ultrabert380":
        raise ValueError(f"not an E21 queue: {queue_path}")
    state.update(worker_started=True, worker_pid=os.getpid(), worker_host=socket.gethostname(),
                 worker_started_at_utc=now())
    save(queue_path, state)
    for dep_path in state.get("depends_on_queue_jsons", []):
        path = ROOT / dep_path
        while True:
            dep = json.loads(path.read_text(encoding="utf-8"))
            if dep.get("status") in TERMINAL and dep.get("finished_at_utc"):
                break
            state.update(status="queued_waiting_for_existing_training", last_dependency_check_at_utc=now())
            save(queue_path, state); time.sleep(20)
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    for job in state["results"]:
        saved_evaluation = job.get("evaluation_results")
        if (job.get("status") in {"completed", "completed_with_warning"}
                and saved_evaluation and (ROOT / saved_evaluation).is_file()):
            continue
        while True:
            processes, gpu_pids = active_resources()
            state.update(status="waiting_for_gpu_release", active_training_processes=processes,
                         gpu_compute_pids=gpu_pids, last_resource_check_at_utc=now())
            save(queue_path, state)
            if not processes and not gpu_pids:
                break
            time.sleep(20)
        # Re-read the queued config immediately before launch so edits to jobs
        # that have not started yet are reflected in the queue's audit record.
        config_path = ROOT / job["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        job["config_sha256"] = digest(config_path)
        job["source_config_sha256"] = digest(SOURCE_CONFIGS[job["experiment"]])
        job["early_stopping_patience"] = int(config["early_stopping_patience"])
        job.update(status="running", started_at_utc=now())
        state.update(status="running", active_job=job["job_id"], updated_at_utc=now())
        save(queue_path, state)
        start = time.monotonic()
        with (queue_dir / job["log"]).open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job["pid"] = process.pid; save(queue_path, state)
            code = process.wait()
        job.update(status="completed" if code == 0 else "failed", exit_code=code,
                   elapsed_seconds=round(time.monotonic() - start, 2), finished_at_utc=now())
        state.update(active_job=None, updated_at_utc=now())
        if code == 0:
            candidates = []
            for path in (ROOT / "artifacts/predictions").glob("*_manifest.json"):
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if manifest.get("experiment_name") == job["experiment_name"]:
                    candidates.append((path, manifest))
            if len(candidates) != 1:
                job.update(status="failed", error=f"expected one completed manifest, found {len(candidates)}")
                state.update(status="completed_with_failures", finished_at_utc=now())
                save(queue_path, state); return 1
            manifest_path = candidates[0][0]
            evaluation_dir = ROOT / "artifacts/analysis/e21" / queue_dir.name / job["experiment"]
            eval_command = [PYTHON, "-u", "-m", "src.e21_evaluate", "--manifest", str(manifest_path),
                            "--output-dir", str(evaluation_dir)]
            eval_log = queue_dir / f"{job['index']:02d}_{job['experiment'].lower().replace('-', '_')}_evaluation.log"
            with eval_log.open("a", encoding="utf-8", buffering=1) as log:
                evaluation = subprocess.run(eval_command, cwd=ROOT, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
            if evaluation.returncode:
                job.update(status="failed", error=f"E21 evaluation exited {evaluation.returncode}",
                           manifest=str(manifest_path), evaluation_exit_code=evaluation.returncode)
                state.update(status="completed_with_failures", finished_at_utc=now())
                save(queue_path, state); return evaluation.returncode
            job.update(manifest=str(manifest_path), evaluation_dir=str(evaluation_dir),
                       evaluation_results=str(evaluation_dir / "results.json"))
        if code:
            state["status"] = "completed_with_failures"
            state["finished_at_utc"] = now(); save(queue_path, state)
            return code
        save(queue_path, state)
    state.update(status="completed", finished_at_utc=now(), active_job=None)
    save(queue_path, state)
    write_report(queue_dir, results_available=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="validate and register the three runs without starting training")
    parser.add_argument("--worker", type=Path, help="explicitly execute an existing E21 queue")
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker.resolve())
    queue_dir = create_queue()
    print(json.dumps({"queue_dir": str(queue_dir), "queue_json": str(queue_dir / "queue.json"),
                      "worker_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
