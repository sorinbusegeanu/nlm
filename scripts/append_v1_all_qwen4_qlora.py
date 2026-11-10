"""Queue a QLoRA retry of Qwen3-4B after the existing V1-all model queue.

The worker waits for the existing queue and its already-scheduled appenders to
finish, then runs Qwen4 serially with the QLoRA config. Training's automatic
batch probe is enabled in that config and will rerun on the quantized model.
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
REQUEST_PATH = QUEUE_DIR / "qwen4_qlora_retry.json"
LOG_NAME = "qwen4_qlora_retry.log"
UPSTREAM_MODELS = (
    "catboost_stylometry", "embeddinggemma2", "granite4_350m", "ultrabert380",
    "gemma4_e2b", "granite4_h_micro", "llama32_3b", "ministral3_3b",
    "qwen06", "qwen17", "qwen4", "smollm3_3b", "svc", "qwen8",
    "minicpm5_2b", "hunyuan_1_8b", "llama31_8b", "ministral3_8b",
)
TERMINAL = {"completed", "completed_with_failures", "failed", "skipped"}


def persist(state: dict) -> None:
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def read_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def latest_by_model(state: dict) -> dict[str, dict]:
    latest = {}
    for item in state.get("results", []):
        latest[str(item.get("model"))] = item
    return latest


def wait_for_upstream() -> dict:
    while True:
        state = read_state()
        latest = latest_by_model(state)
        if all(name in latest and latest[name].get("status") in TERMINAL
               for name in UPSTREAM_MODELS):
            return state
        time.sleep(30)


def main() -> int:
    os.chdir(ROOT)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    lock = QUEUE_DIR / "qwen4_qlora_retry.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(f"Qwen4 QLoRA retry worker already claimed: {lock}", flush=True)
        return 0
    os.close(fd)

    REQUEST_PATH.write_text(json.dumps({
        "status": "waiting_for_existing_queue_tail",
        "model": "qwen4",
        "dataset": "v1_all",
        "config": "configs/models/qwen4.yaml",
        "quantization": "qlora4",
        "batch_probe": "auto_batch_size enabled; reruns against quantized model",
        "waits_for": list(UPSTREAM_MODELS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")

    state = wait_for_upstream()
    results = state.setdefault("results", [])
    command = ["/home/zodrak/cuda13.x/bin/python", "-m", "src.pipeline", "models", "train",
               "--model", "qwen4", "--dataset", "v1_all", "--config",
               "configs/models/qwen4.yaml", "--accept-config-defaults"]
    index = len(results) + 1
    item = {
        "index": index, "model": "qwen4", "attempt": 2,
        "config": "configs/models/qwen4.yaml", "dataset": "v1_all",
        "status": "queued", "command": command,
        "log": f"artifacts/training_queues/v1_all_20261108T185511Z/{LOG_NAME}",
        "queued_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "QLoRA 4-bit retry after interrupted full-precision LoRA run; automatic batch-size probe reruns with 0.95 safety fraction.",
    }
    results.append(item)
    state["status"] = "running"
    state.pop("finished_at_utc", None)
    persist(state)
    REQUEST_PATH.write_text(json.dumps({
        "status": "queued", "queue_id": state.get("queue_id"), "index": index,
        "model": "qwen4", "dataset": "v1_all", "config": item["config"],
        "quantization": "qlora4", "batch_probe_safety_fraction": 0.95,
        "queued_at_utc": item["queued_at_utc"],
    }, indent=2) + "\n", encoding="utf-8")

    item["status"] = "running"
    item["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist(state)
    started = time.monotonic()
    log_path = QUEUE_DIR / LOG_NAME
    with log_path.open("w", encoding="utf-8") as log:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
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
    state["status"] = ("completed" if all(entry.get("status") == "completed" for entry in results)
                       else "completed_with_failures")
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist(state)
    REQUEST_PATH.write_text(json.dumps({
        "status": item["status"], "queue_id": state.get("queue_id"), "index": index,
        "model": "qwen4", "dataset": "v1_all", "quantization": "qlora4",
        "exit_code": code, "finished_at_utc": item["finished_at_utc"],
    }, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
