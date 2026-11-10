"""Append the two requested 8B QLoRA jobs after the existing V1-all queue.

This worker never contacts Hugging Face. It waits for pinned model snapshots to
exist in the local Hub cache before launching each offline training command.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "artifacts/training_queues/v1_all_20261108T185511Z"
STATE_PATH = QUEUE_DIR / "queue.json"
REQUEST_PATH = QUEUE_DIR / "deferred_8b_qlora_jobs.json"
UPSTREAM_MODELS = ("qwen8", "minicpm5_2b", "hunyuan_1_8b", "gemma4_e2b")
JOBS = (
    ("llama31_8b", "configs/models/llama31_8b_qlora.yaml",
     "meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    ("ministral3_8b", "configs/models/ministral3_8b_qlora.yaml",
     "mistralai/Ministral-3-8B-Base-2512", "d4883f9b36aa2e5d775730d3fdba3d30de51a8ef"),
)
TERMINAL = {"completed", "failed", "skipped"}


def persist(state: dict) -> None:
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE_PATH)


def read_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def cache_snapshot(repo_id: str, revision: str) -> Path:
    cache_root = Path(os.environ.get("HF_HUB_CACHE", "")) if os.environ.get("HF_HUB_CACHE") else None
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache_root = hf_home if hf_home.name == "hub" else hf_home / "hub"
    return cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision


def wait_for_upstream() -> dict:
    while True:
        state = read_state()
        items = {str(item.get("model")): item for item in state.get("results", [])}
        if all(name in items and items[name].get("status") in TERMINAL for name in UPSTREAM_MODELS):
            return state
        time.sleep(30)


def main() -> int:
    os.chdir(ROOT)
    lock = QUEUE_DIR / "8b_qlora_append.lock"
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(f"8B QLoRA queue worker already claimed: {lock}", flush=True)
        return 0
    os.close(lock_fd)

    REQUEST_PATH.write_text(json.dumps({
        "status": "waiting_for_existing_queue_tail",
        "models": [{"model": model, "config": config, "checkpoint": repo_id,
                    "revision": revision, "dataset": "v1_all", "quantization": "qlora4"}
                   for model, config, repo_id, revision in JOBS],
        "appended_after": list(UPSTREAM_MODELS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")

    state = wait_for_upstream()
    results = state.setdefault("results", [])
    existing = {str(item.get("model")) for item in results}
    added = []
    for model, config, repo_id, revision in JOBS:
        if model in existing:
            continue
        index = len(results) + 1
        command = ["/home/zodrak/cuda13.x/bin/python", "-m", "src.pipeline", "models", "train",
                   "--model", model, "--dataset", "v1_all", "--config", config,
                   "--accept-config-defaults"]
        item = {"index": index, "model": model, "config": config, "status": "waiting_for_local_weights",
                "checkpoint": repo_id, "model_revision": revision, "dataset": "v1_all",
                "command": command, "log": f"artifacts/training_queues/v1_all_20261108T185511Z/{index:02d}_{model}.log",
                "queued_at_utc": datetime.now(timezone.utc).isoformat(),
                "note": "QLoRA; offline-only; waits for the pinned base model snapshot in the local Hugging Face cache."}
        results.append(item)
        added.append(item)
    if added:
        state["status"] = "waiting_for_local_model_weights"
        state.pop("finished_at_utc", None)
        persist(state)
    elif not all(any(item.get("model") == model for item in results) for model, *_ in JOBS):
        raise RuntimeError("Queue append failed to register both 8B jobs")
    REQUEST_PATH.write_text(json.dumps({
        "status": "queued_waiting_for_local_weights",
        "queue_id": state.get("queue_id"),
        "models": [{"model": model, "config": config, "checkpoint": repo_id,
                    "revision": revision, "dataset": "v1_all", "quantization": "qlora4"}
                   for model, config, repo_id, revision in JOBS],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    items_by_model = {str(entry.get("model")): entry for entry in results}
    for model, _, repo_id, revision in JOBS:
        item = items_by_model[model]
        if item.get("status") in TERMINAL:
            continue
        snapshot = cache_snapshot(repo_id, revision)
        log_path = ROOT / item["log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            while not (snapshot / "config.json").is_file():
                item["status"] = "waiting_for_local_weights"
                item["local_snapshot_expected"] = str(snapshot)
                state["status"] = "waiting_for_local_model_weights"
                persist(state)
                message = f"Waiting for local-only model files: {snapshot}\n"
                if not log_path.exists() or log_path.stat().st_size == 0:
                    log.write(message)
                log.flush()
                time.sleep(60)
            item["status"] = "running"
            item["started_at_utc"] = datetime.now(timezone.utc).isoformat()
            item["local_snapshot"] = str(snapshot)
            state["status"] = "running"
            persist(state)
            started = time.monotonic()
            command = list(item["command"])
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
            item.update({"status": "completed" if code == 0 else "failed", "exit_code": code,
                         "elapsed_seconds": round(time.monotonic() - started, 2),
                         "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            persist(state)

    state["status"] = ("completed" if all(item.get("status") == "completed" for item in results)
                       else "completed_with_failures")
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist(state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
