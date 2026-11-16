"""Retry Qwen3-8B with the repaired probe, then Qwen3-4B only on failure."""
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
QUEUE_ID = "ultrabert_v2x_qwen8_20261110T215001Z"
UPSTREAM_JOB = "ultrabert380_v2.14.balanced_from_scratch_retry_probe"
QWEN4_REPO = "Qwen/Qwen3-4B-Base"
QWEN4_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
QWEN8_JOB_ID = "qwen8_v2.0_qlora4_retry_repaired_probe"
QWEN4_JOB_ID = "qwen4_v2.0_qlora4_fallback"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def cache_snapshot(repo_id: str, revision: str) -> Path:
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    return cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision


def make_config(source: str, output: Path, experiment_name: str) -> Path:
    config = yaml.safe_load((ROOT / source).read_text(encoding="utf-8"))
    config["experiment_name"] = experiment_name
    path = output / Path(source).name.replace(".yaml", "_queue_retry.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_job(item: dict, state: dict, state_path: Path, queue_dir: Path) -> int:
    item["status"] = "running"
    item["started_at_utc"] = now()
    state["status"] = "running"
    state["active_job"] = item["job_id"]
    save(state_path, state)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    started = time.monotonic()
    with (queue_dir / item["log"]).open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{now()}] Starting offline training command: {' '.join(item['command'])}\n")
        process = subprocess.Popen(item["command"], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        item["pid"] = process.pid
        save(state_path, state)
        code = process.wait()
    item.update({"status": "completed" if code == 0 else "failed",
                 "exit_code": code,
                 "elapsed_seconds": round(time.monotonic() - started, 2),
                 "finished_at_utc": now()})
    state["active_job"] = None
    save(state_path, state)
    return code


def main() -> int:
    os.chdir(ROOT)
    queue_dir = ROOT / "artifacts/training_queues" / QUEUE_ID
    state_path = queue_dir / "queue.json"
    chain_path = queue_dir / "qwen_fallback_chain.json"
    config_dir = queue_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    qwen8_config = make_config("configs/models/qwen8.yaml", config_dir,
                               "qwen8_v2.0_qlora_retry_repaired_probe_20261111")
    qwen4_config = make_config("configs/models/qwen4.yaml", config_dir,
                               "qwen4_v2.0_qlora4_fallback_20261111")
    qwen8_command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                     "--model", "qwen8", "--dataset", "v2.0", "--validation-dataset", "v0",
                     "--config", str(qwen8_config), "--accept-config-defaults"]
    qwen4_command = [PYTHON, "-u", "-m", "src.pipeline", "models", "train",
                     "--model", "qwen4", "--dataset", "v2.0", "--validation-dataset", "v0",
                     "--config", str(qwen4_config), "--accept-config-defaults"]

    request = {"status": "waiting_for_active_ultrabert_retry",
               "qwen8_first": True, "qwen4_conditional_on_qwen8_failure": True,
               "dataset": "v2.0", "validation_dataset": "v0",
               "created_at_utc": now(), "model_order": ["qwen8", "qwen4"]}
    save(chain_path, request)

    # Prepare the conditional fallback weights while the active GPU job runs.
    download_code = f'''
from pathlib import Path
from huggingface_hub import snapshot_download
token_path = Path("/home/zodrak/hf.token")
token = token_path.read_text().strip() if token_path.is_file() else None
snapshot = snapshot_download(repo_id={QWEN4_REPO!r}, revision={QWEN4_REVISION!r}, token=token)
print(f"Qwen3-4B pinned snapshot ready: {{snapshot}}", flush=True)
'''
    download_env = dict(os.environ)
    download_env.pop("HF_HUB_OFFLINE", None)
    download_env.pop("TRANSFORMERS_OFFLINE", None)
    with (queue_dir / "qwen4_download.log").open("a", encoding="utf-8", buffering=1) as download_log:
        downloader = subprocess.Popen([PYTHON, "-u", "-c", download_code], cwd=ROOT,
                                      env=download_env, stdout=download_log,
                                      stderr=subprocess.STDOUT)
        request["qwen4_download_pid"] = downloader.pid
        request["qwen4_expected_snapshot"] = str(cache_snapshot(QWEN4_REPO, QWEN4_REVISION))
        request["status"] = "downloading_conditional_qwen4_and_waiting_for_ultrabert"
        save(chain_path, request)

    while True:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        upstream = next((x for x in state["results"] if x.get("job_id") == UPSTREAM_JOB), None)
        if upstream and upstream.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(20)

    results = state.setdefault("results", [])
    known = {x.get("job_id") for x in results}
    if QWEN8_JOB_ID not in known:
        results.append({"index": len(results) + 1, "job_id": QWEN8_JOB_ID,
            "model": "qwen8", "dataset": "v2.0", "validation_dataset": "v0",
            "status": "queued", "training_mode": "qlora4_from_base_retry",
            "checkpoint": "Qwen/Qwen3-8B", "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "config": str(qwen8_config.relative_to(ROOT)), "epochs": 20,
            "early_stopping_patience": 2, "learning_rate": 5e-5,
            "lr_scheduler_type": "linear", "warmup_ratio": 0.02,
            "quantization": "qlora4", "batch_probe_safety_fraction": 0.95,
            "log": "08_qwen8_v2.0_qlora_retry.log", "command": qwen8_command,
            "note": "Retry after corrected gradient-checkpointing-aware batch probe; start from base."})
    if QWEN4_JOB_ID not in known:
        results.append({"index": len(results) + 1, "job_id": QWEN4_JOB_ID,
            "model": "qwen4", "dataset": "v2.0", "validation_dataset": "v0",
            "status": "waiting_for_qwen8_result", "training_mode": "conditional_qlora4_from_base",
            "checkpoint": QWEN4_REPO, "model_revision": QWEN4_REVISION,
            "config": str(qwen4_config.relative_to(ROOT)), "epochs": 20,
            "early_stopping_patience": 2, "learning_rate": 5e-5,
            "lr_scheduler_type": "linear", "warmup_ratio": 0.02,
            "quantization": "qlora4", "batch_probe_safety_fraction": 0.95,
            "condition": f"run only if {QWEN8_JOB_ID} fails",
            "log": "09_qwen4_v2.0_qlora_fallback.log", "command": qwen4_command,
            "note": "Conditional fallback; skips training if Qwen3-8B QLoRA completes."})
    state["status"] = "queued_qwen8_then_conditional_qwen4"
    state["active_job"] = None
    save(state_path, state)
    request["status"] = "queued_after_ultrabert_retry"
    request["qwen8_job_id"] = QWEN8_JOB_ID
    request["qwen4_job_id"] = QWEN4_JOB_ID
    save(chain_path, request)

    qwen8 = next(x for x in results if x.get("job_id") == QWEN8_JOB_ID)
    code8 = run_job(qwen8, state, state_path, queue_dir)
    qwen4 = next(x for x in results if x.get("job_id") == QWEN4_JOB_ID)
    if code8 == 0:
        qwen4.update({"status": "skipped", "skip_reason": "Qwen3-8B QLoRA completed successfully.",
                      "finished_at_utc": now()})
        request["status"] = "completed_qwen8_succeeded_qwen4_skipped"
    else:
        qwen4["status"] = "waiting_for_local_weights"
        state["status"] = "waiting_for_qwen4_weights"
        save(state_path, state)
        download_code_status = downloader.wait()
        qwen4["download_exit_code"] = download_code_status
        if download_code_status != 0 or not (cache_snapshot(QWEN4_REPO, QWEN4_REVISION) / "config.json").is_file():
            qwen4.update({"status": "failed", "exit_code": download_code_status or 1,
                          "error": "Conditional Qwen3-4B base download did not produce the pinned local snapshot."})
            request["status"] = "qwen8_failed_qwen4_download_failed"
        else:
            qwen4["local_snapshot"] = str(cache_snapshot(QWEN4_REPO, QWEN4_REVISION))
            state["status"] = "running_conditional_qwen4"
            save(state_path, state)
            code4 = run_job(qwen4, state, state_path, queue_dir)
            request["status"] = "completed_qwen4_fallback_" + ("succeeded" if code4 == 0 else "failed")

    state["active_job"] = None
    state["fallback_chain_status"] = request["status"]
    state["failed_jobs"] = [x["job_id"] for x in results if x.get("status") == "failed"]
    state["status"] = "completed_with_failures" if state["failed_jobs"] else "completed"
    state["finished_at_utc"] = now()
    save(state_path, state)
    request["finished_at_utc"] = now()
    save(chain_path, request)
    return 0 if request["status"] in {
        "completed_qwen8_succeeded_qwen4_skipped",
        "completed_qwen4_fallback_succeeded",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
