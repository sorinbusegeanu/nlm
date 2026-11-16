"""Run four from-scratch UltraBERT jobs, then Qwen3-8B QLoRA on v2.0.

The Qwen3-8B base snapshot downloads in parallel with the UltraBERT jobs. All
training commands run sequentially and offline through the existing pipeline.
"""
from __future__ import annotations

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
DATASETS = ("v2.0", "v2.1", "v2.14.balanced", "v2.14")
QWEN_MODEL = "qwen8"
QWEN_REPO = "Qwen/Qwen3-8B"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_config(dataset: str, config_dir: Path) -> Path:
    config = yaml.safe_load((ROOT / "configs/models/ultrabert380.yaml").read_text())
    config.update({
        "epochs": 20,
        "experiment_name": f"ultrabert380_{dataset}_from_scratch_20261110",
        "checkpoint": "dmis-lab/UltraBERT-380M",
        "revision": "671a38c087233c0cc72c5bfceea6fffa2822f1db",
        "learning_rate": 8e-6,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.0,
        "early_stopping_patience": 2,
    })
    path = config_dir / f"ultrabert380_{dataset.replace('.', '_')}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_training(item: dict, state: dict, state_path: Path, queue_dir: Path) -> None:
    item["status"] = "running"
    item["started_at_utc"] = now()
    state["status"] = "running"
    state["active_job"] = item["job_id"]
    persist(state_path, state)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    log_path = queue_dir / item["log"]
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{now()}] Starting command: {' '.join(item['command'])}\n")
        started = time.monotonic()
        process = subprocess.Popen(item["command"], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        item["pid"] = process.pid
        persist(state_path, state)
        code = process.wait()
    item.update({
        "status": "completed" if code == 0 else "failed",
        "exit_code": code,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "finished_at_utc": now(),
    })
    state["active_job"] = None
    persist(state_path, state)


def main() -> int:
    os.chdir(ROOT)
    queue_dir = Path(sys.argv[1]).resolve()
    state_path = queue_dir / "queue.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for item in state["results"][:4]:
        config_path = make_config(item["dataset"], config_dir)
        item["config"] = str(config_path.relative_to(ROOT))
        item["command"] = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                           "--model", "ultrabert380", "--dataset", item["dataset"],
                           "--validation-dataset", "v0", "--config", str(config_path),
                           "--accept-config-defaults"]
    qwen = state["results"][4]
    qwen["command"] = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                       "--model", QWEN_MODEL, "--dataset", "v2.0",
                       "--validation-dataset", "v0", "--config",
                       "configs/models/qwen8.yaml", "--accept-config-defaults"]
    persist(state_path, state)

    downloader_code = r'''
from pathlib import Path
from huggingface_hub import snapshot_download
token_path = Path("/home/zodrak/hf.token")
token = token_path.read_text().strip() if token_path.is_file() else None
snapshot = snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    revision="b968826d9c46dd6066d109eabc6255188de91218",
    token=token,
)
print(f"Downloaded Qwen3-8B pinned snapshot: {snapshot}", flush=True)
'''
    download_env = dict(os.environ)
    download_env.pop("HF_HUB_OFFLINE", None)
    download_env.pop("TRANSFORMERS_OFFLINE", None)
    qwen["status"] = "downloading_weights"
    qwen["download_started_at_utc"] = now()
    state["status"] = "running_ultrabert_and_downloading_qwen8"
    persist(state_path, state)
    with (queue_dir / "qwen8_download.log").open("a", encoding="utf-8", buffering=1) as download_log:
        downloader = subprocess.Popen([PYTHON, "-u", "-c", downloader_code], cwd=ROOT,
                                      env=download_env, stdout=download_log,
                                      stderr=subprocess.STDOUT)
        qwen["download_pid"] = downloader.pid
        persist(state_path, state)

    failures = []
    for item in state["results"][:4]:
        run_training(item, state, state_path, queue_dir)
        if item["status"] == "failed":
            failures.append(item["job_id"])

    download_code = downloader.wait()
    qwen["download_exit_code"] = download_code
    qwen["download_finished_at_utc"] = now()
    if download_code != 0:
        qwen.update({"status": "failed", "exit_code": download_code,
                     "error": "Pinned Qwen3-8B base snapshot download failed; training was not started."})
        failures.append(qwen["job_id"])
    else:
        qwen["status"] = "queued"
        persist(state_path, state)
        run_training(qwen, state, state_path, queue_dir)
        if qwen["status"] == "failed":
            failures.append(qwen["job_id"])

    state.update({
        "status": "completed_with_failures" if failures else "completed",
        "active_job": None,
        "failed_jobs": failures,
        "finished_at_utc": now(),
    })
    persist(state_path, state)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
