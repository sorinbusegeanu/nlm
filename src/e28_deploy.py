"""Deterministic two-GPU checkpoint scheduling helpers for E28."""
from __future__ import annotations

from collections import defaultdict


def estimated_seconds(checkpoint, runtime_by_id, batch_by_model, rows):
    """Scale measured 64-row inference runtime; fall back to checkpoint bytes."""
    sid = str(checkpoint["candidate_id"])
    if sid in runtime_by_id:
        measured = float(runtime_by_id[sid])
        sample_rows = float(runtime_by_id.get("__sample_rows__", 64))
        return max(0.001, measured * int(rows) / max(sample_rows, 1.0))
    model = str(checkpoint["model"])
    batch = max(1, int(batch_by_model.get(model, 1)))
    byte_count = sum(int(item.get("size", 0)) for item in checkpoint.get("files", []))
    return max(1.0, byte_count / (128 * 1024**2)) * int(rows) / batch


def distribute_checkpoints(checkpoints, runtime_by_id, batch_by_model, rows):
    """Greedy longest-estimated-runtime-first split across two GPU workers."""
    loads = [0.0, 0.0]
    workers = [[], []]
    costs = {str(cp["candidate_id"]): estimated_seconds(
        cp, runtime_by_id, batch_by_model, rows) for cp in checkpoints}
    ordered = sorted(checkpoints, key=lambda cp: (-costs[str(cp["candidate_id"])],
                                                   str(cp["candidate_id"])))
    for cp in ordered:
        worker = min(range(2), key=lambda ix: (loads[ix], ix))
        workers[worker].append(cp)
        loads[worker] += costs[str(cp["candidate_id"])]
    return [{"gpu_id": ix, "estimated_seconds": loads[ix],
             "checkpoint_ids": [str(cp["candidate_id"]) for cp in workers[ix]],
             "checkpoints": workers[ix]} for ix in range(2)]


def intervals_overlap(records):
    """Return whether checkpoint work actually overlapped across both workers."""
    by_worker = defaultdict(list)
    for item in records:
        by_worker[int(item["gpu_id"])].append((float(item["start_epoch"]),
                                                float(item["end_epoch"])))
    if set(by_worker) != {0, 1}:
        return False
    return any(max(a0, b0) < min(a1, b1)
               for a0, a1 in by_worker[0] for b0, b1 in by_worker[1])
