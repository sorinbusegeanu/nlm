"""Publish dynamic ORACLE04 notebooks and submit their code outputs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = "/home/zodrak/cuda13.x/bin/kaggle"
STRATEGIES = [
    ("combined_logistic_core", "oracle04-combined-logistic-core"),
    ("global_logistic_core", "oracle04-global-logistic-core"),
    ("global_catboost_core", "oracle04-global-catboost-core"),
]
OUT = ROOT / "artifacts/kaggle/oracle04_dynamic_submissions.json"
LOG = ROOT / "artifacts/kaggle/oracle04_dynamic_submissions.log"


def run(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    with LOG.open("a") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        log.write(f"[{datetime.now(timezone.utc).isoformat()}] exit={result.returncode}\n")
        log.flush()
    return result


def main() -> int:
    state = {"asset_dataset": "zodrak/oracle04-dynamic-assets", "strategies": {},
             "upload": "starting"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, indent=2) + "\n")
    package = ROOT / "artifacts/kaggle/oracle04-dynamic-assets"
    uploaded = run([KAGGLE, "datasets", "create", "-p", str(package)])
    if uploaded.returncode:
        state["upload"] = "failed"
        state["upload_exit_code"] = uploaded.returncode
        OUT.write_text(json.dumps(state, indent=2) + "\n")
        return uploaded.returncode
    state["upload"] = "created"
    OUT.write_text(json.dumps(state, indent=2) + "\n")

    files = None
    for attempt in range(30):
        files = run([KAGGLE, "datasets", "files", state["asset_dataset"], "--csv"])
        if files.returncode == 0:
            break
        time.sleep(60)
    if files is None or files.returncode:
        state["upload_error"] = f"dataset file listing failed with exit {files.returncode}"
        OUT.write_text(json.dumps(state, indent=2) + "\n")
        return files.returncode

    for strategy, slug in STRATEGIES:
        kernel = f"zodrak/{slug}"
        folder = ROOT / "artifacts/kaggle/oracle04-submission-kernels" / strategy
        item = {"kernel": kernel, "status": "publishing"}
        state["strategies"][strategy] = item
        OUT.write_text(json.dumps(state, indent=2) + "\n")
        pushed = run([KAGGLE, "kernels", "push", "--no-run", "-p", str(folder)])
        if pushed.returncode:
            item.update(status="publish_failed", exit_code=pushed.returncode)
            continue
        item["status"] = "submitted"
        OUT.write_text(json.dumps(state, indent=2) + "\n")
        message = f"ORACLE04 {strategy} runtime inference, Trust List 2 v025"
        submitted = run([KAGGLE, "competitions", "submit", "name-that-llm",
                         "--kernel", kernel, "--file", "submission.parquet",
                         "--message", message, "--wait", "0", "--poll-interval", "60"])
        item.update(status="scored" if submitted.returncode == 0 else "scoring_or_submission_failed",
                    exit_code=submitted.returncode)
        OUT.write_text(json.dumps(state, indent=2) + "\n")

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(state, indent=2) + "\n")
    return 0 if all(v.get("status") == "scored" for v in state["strategies"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
