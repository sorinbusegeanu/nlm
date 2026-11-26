"""Persistent state and deterministic row-level evaluation for the V3 factory.

This module deliberately does not use prompt/group identifiers. Model fitting is
provided by the repository pipeline adapter in the launch script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .v2_checkpoint_trust import LABELS as LABELS

MODELS = ("svc", "catboost_stylometry", "ultrabert380")
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fixed_row_folds(rows: pd.DataFrame, n_splits: int = 5, seed: int = 2026) -> dict[str, int]:
    """Create deterministic folds stratified by validation dataset and label."""
    required = {"id", "model", "_dataset"}
    if required - set(rows):
        raise ValueError(f"validation rows missing {sorted(required - set(rows))}")
    if rows.id.astype(str).duplicated().any():
        raise ValueError("validation response IDs must be unique across datasets")
    strata = rows._dataset.astype(str) + "\x1f" + rows.model.astype(str)
    fold = np.full(len(rows), -1, dtype=np.int8)
    for _, indexes in rows.groupby(strata, sort=True).indices.items():
        indexes = np.asarray(indexes, dtype=np.int64)
        if len(indexes) < n_splits:
            raise ValueError("every dataset-label stratum must have at least five rows")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # A constant stratum is represented as a single class, and stratification
        # is done within each stratum to obtain deterministic, balanced row folds.
        for fold_id, (_, held) in enumerate(splitter.split(indexes, np.zeros(len(indexes)))):
            fold[indexes[held]] = fold_id
    if (fold < 0).any():
        raise AssertionError("fold assignment left rows unassigned")
    return {str(row_id): int(fold_id)
            for row_id, fold_id in zip(rows.id.astype(str), fold)}


def equal_dataset_label_weights(rows: pd.DataFrame) -> np.ndarray:
    """Each present dataset-label cell receives equal total weight."""
    keys = rows._dataset.astype(str) + "\x1f" + rows.model.astype(str)
    counts = keys.value_counts()
    weight = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    weight /= weight.sum()
    return weight


def load_queue(path: Path) -> pd.DataFrame:
    queue = pd.read_csv(path)
    required = {"Rank", "Contributor", "Label"}
    if required - set(queue):
        raise ValueError(f"ranking queue missing {sorted(required - set(queue))}")
    queue = queue.sort_values("Rank", kind="stable").reset_index(drop=True)
    if queue.Rank.astype(int).tolist() != list(range(1, len(queue) + 1)):
        raise ValueError("ranking queue must have contiguous ranks starting at one")
    if queue.duplicated(["Contributor", "Label"]).any():
        raise ValueError("ranking queue contains duplicate contributor-label pairs")
    return queue


class FactoryState:
    """Atomic commit layer; CSV outputs are always regenerated from committed state."""
    def __init__(self, root: Path, queue: pd.DataFrame, hashes: dict[str, str], folds: dict[str, int]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "factory_state.json"
        self.queue = queue
        self.hashes = hashes
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("input_sha256") != hashes:
                raise ValueError("factory inputs changed since initialization; refusing to resume")
            if self.state.get("fold_assignments") != folds:
                raise ValueError("fixed validation folds changed; refusing to resume")
        else:
            self.state = {"version": 1, "input_sha256": hashes,
                          "fold_assignments": folds, "next_rank": 1,
                          "accepted": [], "baseline": None, "episodes": [],
                          "in_progress": None}
            self.commit()
        self.render()

    def commit(self):
        atomic_json(self.state_path, self.state)
        self.render()

    def render(self):
        accepted_path = self.root / "accepted_subsets.csv"
        with accepted_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_dataset", "label"])
            writer.writerows(self.state.get("accepted", []))
        pd.DataFrame(self.state.get("episodes", [])).to_csv(
            self.root / "episode_results.csv", index=False)
        labels = [{"episode": episode["episode"], "contributor": episode["contributor"],
                   **row} for episode in self.state.get("episodes", [])
                  for row in episode.get("label_results", [])]
        pd.DataFrame(labels).to_csv(self.root / "episode_label_results.csv", index=False)
        completed = {int(row["rank"]): row for row in self.state.get("episodes", [])}
        lines = []
        if self.state.get("baseline_started_at"):
            lines.append(f"{self.state['baseline_started_at']} | BASELINE | STARTED | V3.0-BASELINE")
        if self.state.get("baseline_completed_at"):
            lines.append(f"{self.state['baseline_completed_at']} | BASELINE | COMPLETE")
        for rank, row in sorted(completed.items()):
            lines.append(f"{row['started_at']} | {rank:03d} | STARTED | {row['contributor']} | {row['label']}")
            decision = row["decision"]
            if decision == "ACCEPTED":
                lines.append(f"{row['finished_at']} | {rank:03d} | ACCEPTED | Overall BA: {row['baseline_ba']:.2%} -> {row['mean_ba']:.2%} ({row['improvement_pp']:+.2f} pp) | {row['label']}: {row['baseline_label_recall']:.2%} -> {row['mean_label_recall']:.2%} ({row['label_delta_pp']:+.2f} pp)")
            else:
                lines.append(f"{row['finished_at']} | {rank:03d} | REJECTED")
        pending = self.state.get("in_progress")
        if pending and int(pending.get("rank", 0)) > 0:
            lines.append(f"{pending['started_at']} | {int(pending['rank']):03d} | STARTED | {pending['contributor']} | {pending['label']}")
        log = self.root / "factory.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        # Append only events absent from the journal. Reconciliation from
        # committed state recovers a result line after a crash without truncating
        # earlier episodes or duplicating a STARTED line on retry.
        import fcntl
        with log.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            present = set(stream.read().splitlines())
            for line in lines:
                identity = line.split(" | ", 2)
                if " | STARTED | " in line:
                    # A failed/retried episode keeps its first start timestamp,
                    # ensuring one STARTED and one result for the committed rank.
                    identity_text = (identity[1] + " | STARTED") if len(identity) > 2 else line
                    exists = any((entry.split(" | ", 2)[1] + " | STARTED") == identity_text
                                 for entry in present if " | STARTED | " in entry
                                 and len(entry.split(" | ", 2)) > 2)
                    if exists:
                        continue
                if line not in present:
                    stream.seek(0, os.SEEK_END)
                    stream.write(line + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    present.add(line)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def run(self, episode_runner, tmp_dir: Path):
        tmp_dir = Path(tmp_dir)
        # Establish the reference model before considering queue rank 1.
        if self.state["baseline"] is None:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            self.state["baseline_started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.state["in_progress"] = {"rank": 0, "phase": "baseline",
                                          "started_at": self.state["baseline_started_at"]}
            self.commit()
            try:
                self.state["baseline"] = episode_runner(
                    0, "V3.0-BASELINE", "", [], tmp_dir, None)
            except Exception as exc:
                self.state["last_failure"] = {"rank": 0, "error": repr(exc)}
                self.commit()
                raise
            self.state["in_progress"] = None
            self.state["baseline_completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.commit()
        while self.state["next_rank"] <= len(self.queue):
            rank = int(self.state["next_rank"])
            item = self.queue.iloc[rank - 1]
            contributor, label = str(item.Contributor), str(item.Label)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # This unmatched STARTED record is intentionally omitted from committed
            # outputs after interruption; resume retries from this rank.
            self.state["in_progress"] = {"rank": rank, "started_at": started,
                                          "contributor": contributor, "label": label}
            self.state.pop("last_failure", None)
            self.commit()
            try:
                result = episode_runner(rank, contributor, label,
                    list(self.state["accepted"]), tmp_dir, self.state["baseline"])
            except Exception as exc:
                self.state["last_failure"] = {"rank": rank, "error": repr(exc)}
                self.commit()
                raise
            baseline = self.state["baseline"]
            base_ba = float(baseline["overall_ba"])
            candidate_ba = float(result["overall_ba"])
            accepted = candidate_ba > base_ba
            base_recall = float(baseline["label_recall"][label])
            candidate_recall = float(result["label_recall"][label])
            finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {"episode": rank, "rank": rank, "contributor": contributor,
                "label": label, "v0_ba": result["v0_ba"], "v0_ext_ba": result["v0_ext_ba"],
                "mean_ba": candidate_ba, "baseline_ba": base_ba,
                "improvement_pp": (candidate_ba - base_ba) * 100,
                "decision": "ACCEPTED" if accepted else "REJECTED",
                "started_at": started, "finished_at": finished,
                "baseline_label_recall": base_recall,
                "mean_label_recall": candidate_recall,
                "label_delta_pp": (candidate_recall - base_recall) * 100,
                "label_results": result["per_label"]}
            if accepted:
                pair = [contributor, label]
                if pair not in self.state["accepted"]:
                    self.state["accepted"].append(pair)
                self.state["baseline"] = result
            self.state["episodes"].append(record)
            self.state["next_rank"] = rank + 1
            self.state["in_progress"] = None
            self.state.pop("last_failure", None)
            self.commit()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def pooled_metrics(predictions: np.ndarray, truth: np.ndarray, datasets: np.ndarray,
                   labels: list[str]) -> dict:
    from sklearn.metrics import balanced_accuracy_score, recall_score
    out = {}
    per_label = {}
    for name in ("v0", "v0_ext"):
        mask = datasets == name
        if not mask.any():
            raise ValueError(f"missing validation dataset {name}")
        out[f"{name}_ba"] = float(balanced_accuracy_score(truth[mask], predictions[mask]))
        per_label[name] = recall_score(truth[mask], predictions[mask], labels=labels,
                                      average=None, zero_division=0)
    out["overall_ba"] = (out["v0_ba"] + out["v0_ext_ba"]) / 2
    out["label_recall"] = {label: float((per_label["v0"][i] + per_label["v0_ext"][i]) / 2)
                            for i, label in enumerate(labels)}
    out["per_label"] = [{"label": label, "v0_recall": float(per_label["v0"][i]),
        "v0_ext_recall": float(per_label["v0_ext"][i]),
        "mean_recall": out["label_recall"][label]} for i, label in enumerate(labels)]
    return out
