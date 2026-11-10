"""Append the newly downloaded MiniCPM5 and Hunyuan candidates after Qwen8."""
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
JOBS = [
    ("minicpm5_2b", "configs/models/minicpm5_2b.yaml"),
    ("hunyuan_1_8b", "configs/models/hunyuan_1_8b.yaml"),
    ("gemma4_e2b", "configs/models/gemma4_e2b_v1_all_qlora.yaml"),
]


def persist(state: dict) -> None:
    temp = STATE_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE_PATH)


def wait_for_qwen8() -> dict:
    while True:
        state = json.loads(STATE_PATH.read_text())
        qwen8 = next((item for item in state.get("results", []) if item.get("model") == "qwen8"), None)
        if qwen8 and qwen8.get("status") in {"completed", "failed"}:
            return state
        if all(any(item.get("model") == model for item in state.get("results", [])) for model, _ in JOBS):
            raise RuntimeError("MiniCPM5 and Hunyuan are already present in this queue")
        time.sleep(30)


def main() -> int:
    os.chdir(ROOT)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    lock = QUEUE_DIR / "downloaded_models_append.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        print(f"Append worker already claimed: {lock}", flush=True)
        return 0
    os.close(fd)
    state = wait_for_qwen8()
    for model, config in JOBS:
        if any(item.get("model") == model for item in state.get("results", [])):
            raise RuntimeError(f"Refusing to duplicate queue item {model}")
        index = len(state["results"]) + 1
        log_path = QUEUE_DIR / f"{index:02d}_{model}.log"
        command = ["/home/zodrak/cuda13.x/bin/python", "-m", "src.pipeline", "models", "train",
                   "--model", model, "--dataset", "v1_all", "--config", config,
                   "--accept-config-defaults"]
        item = {"index": index, "model": model, "config": config, "status": "queued",
                "command": command, "log": str(log_path.relative_to(ROOT)),
                "queued_at_utc": datetime.now(timezone.utc).isoformat()}
        state["results"].append(item)
    state["status"] = "running"
    state.pop("finished_at_utc", None)
    persist(state)

    for item in state["results"][-len(JOBS):]:
        item["status"] = "running"
        item["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist(state)
        started = time.monotonic()
        with (ROOT / item["log"]).open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(item["command"], cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = proc.wait()
        item.update({"status": "completed" if code == 0 else "failed", "exit_code": code,
                     "elapsed_seconds": round(time.monotonic() - started, 2),
                     "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        persist(state)
        print(f"{item['model']}: {item['status']} (exit {code})", flush=True)
    state["status"] = "completed" if all(item.get("status") == "completed" for item in state["results"]) else "completed_with_failures"
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist(state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
