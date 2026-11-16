"""Append Llama 3.2 3B QLoRA after the existing V2.14 classical queue."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zodrak/cuda13.x/bin/python"
QUEUE_DIR = ROOT / "artifacts/training_queues/v2_14_classical_pf_20261112T145950Z"
QUEUE_PATH = QUEUE_DIR / "queue.json"
CONFIG_PATH = ROOT / "configs/models/llama32_3b_v2_14_qlora.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if not QUEUE_PATH.is_file() or not CONFIG_PATH.is_file():
        raise FileNotFoundError("the completed V2.14 queue or Llama QLoRA config is missing")
    state = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    for job in state.get("jobs", []):
        if job.get("model") == "llama32_3b" and job.get("dataset") == "v2.14":
            print(json.dumps({"status": "already_queued", "job": job}, indent=2))
            return
    if state.get("status") not in {"completed", "completed_with_failures"}:
        raise RuntimeError(f"cannot append behind nonterminal queue status {state.get('status')!r}")
    if any(job.get("status") not in {"completed", "failed", "cancelled", "blocked"}
           for job in state.get("jobs", [])):
        raise RuntimeError("cannot append while an existing queue job is nonterminal")

    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    index = len(state["jobs"]) + 1
    job_id = "llama32_3b_v2_14_qlora_20261112"
    command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
               "--model", "llama32_3b", "--dataset", "v2.14",
               "--validation-dataset", "v0", "--config", str(CONFIG_PATH),
               "--seed", "2026", "--accept-config-defaults"]
    job = {"index": index, "job_id": job_id, "queue_tag": "llama32_3b_v2_14_qlora",
           "model": "llama32_3b", "dataset": "v2.14", "validation_dataset": "v0",
           "status": "queued", "training_mode": "qlora4_from_base",
           "config": str(CONFIG_PATH.relative_to(ROOT)), "config_sha256": config_hash,
           "command": command, "log": f"03_llama32_3b_v2_14_qlora.log",
           "queued_at_utc": now(), "epochs": 5, "early_stopping_patience": 2,
           "learning_rate": 5.0e-5, "optimizer": "paged_adamw_8bit",
           "scheduler": "linear", "warmup_ratio": 0.03,
           "quantization": "qlora4", "compute_dtype": "bfloat16",
           "quantization_type": "nf4", "double_quantization": True,
           "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
           "gradient_accumulation_steps": 1, "checkpoint_selection": "balanced_accuracy"}
    state["jobs"].append(job)
    state.update({"status": "queued", "active_job": None,
                  "appended_job_at_utc": now(),
                  "appended_job_dependency": "jobs 1 and 2 terminal; wait for GPU release"})
    state.pop("finished_at_utc", None)
    save(QUEUE_PATH, state)

    with (QUEUE_DIR / "queue_worker.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            [PYTHON, "-u", str(ROOT / "scripts/queue_v2_14_classical_models.py"),
             "--worker", str(QUEUE_DIR), "--start-index", str(index - 1)],
            cwd=ROOT, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    state["continuation_worker_pid"] = process.pid
    save(QUEUE_PATH, state)
    print(json.dumps({"queue_id": state["queue_id"], "queue_json": str(QUEUE_PATH),
                      "job": job, "worker_pid": process.pid,
                      "status": "queued_after_classical_jobs"}, indent=2))


if __name__ == "__main__":
    main()
