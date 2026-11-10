"""Stop the V1-all training queue immediately after the active SmolLM3 job.

Waiting append workers are cancelled up front by the launcher. This watcher
waits for SmolLM3 to finish, interrupts any next queued training command, and
marks remaining queue entries cancelled while preserving completed history.
"""
from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "artifacts/training_queues/v1_all_20261108T185511Z"
STATE_PATH = QUEUE_DIR / "queue.json"
PENDING = {"queued", "running", "waiting", "waiting_for_local_weights",
           "waiting_for_upstream", "downloading_checkpoint"}
QUEUE_MODELS = {"svc", "qwen8", "minicpm5_2b", "hunyuan_1_8b", "gemma4_e2b",
                "llama31_8b", "ministral3_8b", "qwen4"}


def read_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def pipeline_processes() -> list[tuple[int, int, str]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            status = (entry / "status").read_text(errors="replace")
        except (OSError, ProcessLookupError):
            continue
        if "-m src.pipeline models train" not in args:
            continue
        parent = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        found.append((int(entry.name), parent, args))
    return found


def wait_for_smollm() -> tuple[int | None, int | None]:
    controller = None
    shell = None
    while True:
        for pid, parent, args in pipeline_processes():
            if "--model smollm3_3b" in args and "--dataset v1_all" in args:
                controller = parent
                try:
                    shell_status = Path(f"/proc/{controller}/status").read_text()
                    shell = next(int(line.split()[1]) for line in shell_status.splitlines()
                                 if line.startswith("PPid:"))
                except (OSError, StopIteration):
                    shell = None
                break
        state = read_state()
        smollm = next((item for item in reversed(state.get("results", []))
                       if item.get("model") == "smollm3_3b"), None)
        if smollm and smollm.get("status") in {"completed", "failed", "cancelled", "skipped"}:
            return controller, shell
        time.sleep(0.5)


def interrupt(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pass


def main() -> int:
    os.chdir(ROOT)
    controller, shell = wait_for_smollm()
    # If the original queue already launched its last item, interrupt that
    # child first so the queue manager can record its exit cleanly.
    for pid, parent, args in pipeline_processes():
        if parent == controller and "--model smollm3_3b" not in args:
            interrupt(pid)
    interrupt(controller)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        live = {pid for pid, _, _ in pipeline_processes()}
        if controller not in live and not any(
            parent == controller for _, parent, _ in pipeline_processes()
        ):
            break
        time.sleep(0.25)

    # Avoid leaving a just-started queued child behind if it raced the SIGINT.
    for pid, _, args in pipeline_processes():
        if ("--model smollm3_3b" not in args and
                any(f"--model {model}" in args for model in QUEUE_MODELS)):
            interrupt(pid)
    if controller is not None and Path(f"/proc/{controller}").exists():
        try:
            os.kill(controller, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if shell is not None and Path(f"/proc/{shell}").exists():
        try:
            os.kill(shell, signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Let the queue driver's last write settle before applying the final state.
    time.sleep(1)
    state = read_state()
    stopped_at = datetime.now(timezone.utc).isoformat()
    for item in state.get("results", []):
        if item.get("status") in PENDING:
            item["status"] = "cancelled"
            item["cancelled_at_utc"] = stopped_at
            item["cancellation_reason"] = "User requested queue stop after smollm3_3b completed."
    state["status"] = "cancelled"
    state["finished_at_utc"] = stopped_at
    state["stop_reason"] = "Stopped after smollm3_3b; all remaining training jobs cancelled by user request."
    write_json(STATE_PATH, state)

    for name in ("qwen4_qlora_retry.json", "deferred_8b_qlora_jobs.json"):
        path = QUEUE_DIR / name
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            data["status"] = "cancelled"
            data["cancelled_at_utc"] = stopped_at
            data["cancellation_reason"] = "User requested the V1-all training queue be emptied after smollm3_3b."
            write_json(path, data)
    for name in ("qwen4_qlora_retry.lock", "8b_qlora_append.lock", "downloaded_models_append.lock"):
        (QUEUE_DIR / name).unlink(missing_ok=True)
    write_json(QUEUE_DIR / "queue_stop_record.json", {
        "status": "cancelled", "after_model": "smollm3_3b",
        "cancelled_at_utc": stopped_at,
        "remaining_queue_items": [item.get("model") for item in state.get("results", [])
                                   if item.get("status") == "cancelled"],
    })
    print(f"Queue cancelled after SmolLM3 at {stopped_at}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
