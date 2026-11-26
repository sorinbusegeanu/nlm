"""Persistent state and ranking for the V3 contributor-level training factory."""
from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from .v3_contributor_label_factory import atomic_json, fixed_row_folds


def make_contributor_ranking(label_ranking: pd.DataFrame) -> pd.DataFrame:
    required = {"Contributor", "Label", "Available rows", "Relative accuracy"}
    if required - set(label_ranking):
        raise ValueError(f"label ranking missing columns {sorted(required - set(label_ranking))}")
    frame = label_ranking.copy()
    frame["Available rows"] = pd.to_numeric(frame["Available rows"], errors="raise")
    frame["Relative accuracy"] = pd.to_numeric(frame["Relative accuracy"], errors="raise")
    if frame.duplicated(["Contributor", "Label"]).any():
        raise ValueError("label ranking contains duplicate contributor-label rows")
    grouped = frame.groupby("Contributor", sort=False).agg(
        available_rows=("Available rows", "sum"),
        labels=("Label", "nunique"),
        quality_score=("Relative accuracy", "mean"),
    ).reset_index()
    grouped = grouped.rename(columns={"Contributor": "Source dataset"})
    grouped = grouped.sort_values(
        ["quality_score", "available_rows", "Source dataset"],
        ascending=[False, False, True], kind="stable").reset_index(drop=True)
    grouped.insert(0, "Rank", range(1, len(grouped) + 1))
    return grouped.rename(columns={"available_rows": "Available rows",
                                   "labels": "Labels",
                                   "quality_score": "Quality score"})[
        ["Rank", "Source dataset", "Available rows", "Labels", "Quality score"]]


def make_size_contributor_ranking(rows: pd.DataFrame, baseline_ids) -> pd.DataFrame:
    """Rank every source by eligible rows after excluding the fixed baseline IDs."""
    required = {"id", "model", "source_dataset"}
    if required - set(rows):
        raise ValueError(f"V3.1 rows lack columns {sorted(required - set(rows))}")
    frame = rows.copy()
    frame["id"] = frame.id.astype(str)
    frame["model"] = frame.model.astype(str)
    frame["source_dataset"] = frame.source_dataset.astype(str)
    baseline_ids = set(map(str, baseline_ids))
    eligible = frame.loc[~frame.id.isin(baseline_ids)]
    grouped = eligible.groupby("source_dataset", sort=False).agg(
        available_rows=("id", "size"), labels=("model", "nunique")).reset_index()
    # Include contributors entirely represented in the baseline with zero eligible rows.
    all_sources = sorted(frame.source_dataset.unique())
    grouped = grouped.set_index("source_dataset").reindex(all_sources, fill_value=0).reset_index()
    grouped = grouped.sort_values(
        ["available_rows", "source_dataset"], ascending=[True, True], kind="stable")
    grouped.insert(0, "Rank", range(1, len(grouped) + 1))
    return grouped.rename(columns={"source_dataset": "Source dataset",
                                  "available_rows": "Available rows",
                                  "labels": "Labels"})[
        ["Rank", "Source dataset", "Available rows", "Labels"]].reset_index(drop=True)


def ranking_queue(path: Path) -> pd.DataFrame:
    queue = pd.read_csv(path)
    required = {"Rank", "Source dataset", "Available rows", "Labels"}
    if required - set(queue):
        raise ValueError(f"contributor ranking missing {sorted(required - set(queue))}")
    queue = queue.sort_values("Rank", kind="stable").reset_index(drop=True)
    if queue["Source dataset"].duplicated().any():
        raise ValueError("contributor ranking contains duplicate sources")
    if queue.Rank.astype(int).tolist() != list(range(1, len(queue) + 1)):
        raise ValueError("contributor ranking must have contiguous ranks starting at one")
    return queue


class ContributorFactoryState:
    """Commit episode state atomically and append log events without truncation."""
    def __init__(self, root: Path, queue: pd.DataFrame, hashes: dict[str, str],
                 folds: dict[str, int], initial_baseline: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "factory_state.json"
        self.queue = queue
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("input_sha256") != hashes:
                raise ValueError("contributor-factory inputs changed; refusing to resume")
            if self.state.get("fold_assignments") != folds:
                raise ValueError("fixed validation folds changed; refusing to resume")
        else:
            self.state = {"version": 1, "input_sha256": hashes,
                          "fold_assignments": folds, "next_rank": 1,
                          "accepted": [], "baseline": initial_baseline,
                          "episodes": [], "in_progress": None}
            self.commit()
        self._render_csvs()
        self._reconcile_log()

    def commit(self):
        atomic_json(self.state_path, self.state)
        self._render_csvs()
        self._reconcile_log()

    def _render_csvs(self):
        with (self.root / "accepted_contributors.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_dataset"])
            writer.writerows([[source] for source in self.state.get("accepted", [])])
        pd.DataFrame(self.state.get("episodes", [])).drop(
            columns=["label_results"], errors="ignore").to_csv(
                self.root / "episode_results.csv", index=False)
        labels = [{"episode": episode["episode"], "source_dataset": episode["source_dataset"],
                   **row} for episode in self.state.get("episodes", [])
                  for row in episode.get("label_results", [])]
        columns = ["episode", "source_dataset", "label", "v0_recall", "v0_ext_recall", "mean_recall"]
        pd.DataFrame(labels, columns=columns).to_csv(
            self.root / "episode_label_results.csv", index=False)

    def _expected_log(self):
        events = []
        for episode in self.state.get("episodes", []):
            rank = int(episode["rank"])
            events.append((rank, 0,
                f"{episode['started_at']} | {rank:03d} | STARTED | {episode['source_dataset']}"))
            if episode["decision"] == "ACCEPTED":
                result = (f"{episode['finished_at']} | {rank:03d} | ACCEPTED | Overall BA: "
                          f"{episode['baseline_ba']:.2%} -> {episode['mean_ba']:.2%} "
                          f"({episode['improvement_pp']:+.2f} pp)")
            else:
                result = f"{episode['finished_at']} | {rank:03d} | REJECTED"
            events.append((rank, 1, result))
        pending = self.state.get("in_progress")
        if pending:
            rank = int(pending["rank"])
            events.append((rank, 0,
                f"{pending['started_at']} | {rank:03d} | STARTED | {pending['source_dataset']}"))
        return [line for _, _, line in sorted(events)]

    def _reconcile_log(self):
        log = self.root / "factory.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            present = set(stream.read().splitlines())
            for line in self._expected_log():
                if line not in present:
                    stream.seek(0, os.SEEK_END)
                    stream.write(line + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    present.add(line)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def run(self, episode_runner, tmp_dir: Path):
        tmp_dir = Path(tmp_dir)
        while self.state["next_rank"] <= len(self.queue):
            rank = int(self.state["next_rank"])
            item = self.queue.iloc[rank - 1]
            source = str(item["Source dataset"])
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            self.state["in_progress"] = {"rank": rank, "source_dataset": source,
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            self.state.pop("last_failure", None)
            self.commit()
            try:
                result = episode_runner(rank, source, list(self.state["accepted"]),
                                        tmp_dir, self.state["baseline"])
            except Exception as exc:
                self.state["last_failure"] = {"rank": rank, "error": repr(exc)}
                self.commit()
                raise
            baseline = self.state["baseline"]
            baseline_ba = float(baseline["overall_ba"])
            candidate_ba = float(result["overall_ba"])
            accepted = candidate_ba > baseline_ba
            finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {"episode": rank, "rank": rank, "source_dataset": source,
                "v0_ba": result["v0_ba"], "v0_ext_ba": result["v0_ext_ba"],
                "mean_ba": candidate_ba, "baseline_ba": baseline_ba,
                "improvement_pp": (candidate_ba - baseline_ba) * 100,
                "decision": "ACCEPTED" if accepted else "REJECTED",
                "started_at": self.state["in_progress"]["started_at"],
                "finished_at": finished, "label_results": result["per_label"]}
            if accepted:
                if source not in self.state["accepted"]:
                    self.state["accepted"].append(source)
                self.state["baseline"] = result
            self.state["episodes"].append(record)
            self.state["next_rank"] = rank + 1
            self.state["in_progress"] = None
            self.state.pop("last_failure", None)
            self.commit()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

