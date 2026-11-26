import json

import numpy as np
import pandas as pd

from src.v3_contributor_label_factory import (FactoryState,
    equal_dataset_label_weights, fixed_row_folds, load_queue)


def _queue():
    return pd.DataFrame({"Rank": [1, 2, 3], "Contributor": ["c", "a", "b"],
                         "Label": ["z", "y", "x"]})


def _scores(value):
    labels = ["x", "y", "z"]
    return {"overall_ba": value, "v0_ba": value, "v0_ext_ba": value,
            "label_recall": {label: value for label in labels},
            "per_label": [{"label": label, "v0_recall": value,
                "v0_ext_recall": value, "mean_recall": value} for label in labels]}


def test_row_folds_and_equal_dataset_label_weights_ignore_prompt_ids():
    rows = pd.DataFrame({"id": [f"r{i}" for i in range(60)],
        "model": ["x", "y", "z"] * 20,
        "_dataset": ["v0"] * 30 + ["v0_ext"] * 30,
        "prompt_id": ["irrelevant"] * 60})
    folds_a = fixed_row_folds(rows)
    rows.prompt_id = [f"changed{i}" for i in range(len(rows))]
    folds_b = fixed_row_folds(rows)
    assert folds_a == folds_b
    assert set(folds_a.values()) == set(range(5))
    weights = equal_dataset_label_weights(rows)
    keys = rows._dataset + ":" + rows.model
    totals = pd.Series(weights).groupby(keys.to_numpy()).sum()
    assert np.allclose(totals.to_numpy(), np.repeat(1 / 6, 6))


def test_queue_order_validation(tmp_path):
    path = tmp_path / "queue.csv"
    _queue().iloc[::-1].to_csv(path, index=False)
    loaded = load_queue(path)
    assert loaded.Rank.tolist() == [1, 2, 3]
    loaded.loc[1, "Rank"] = 4
    loaded.to_csv(path, index=False)
    try:
        load_queue(path)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous queue ranks must be rejected")


def test_accept_reject_commit_log_resume_and_temp_cleanup(tmp_path):
    root = tmp_path / "factory"
    temp = tmp_path / "tmp"
    queue = _queue()
    folds = {"v0-1": 0}
    state = FactoryState(root, queue, {"queue": "h"}, folds)
    calls = []

    def runner(rank, contributor, label, accepted, work, baseline):
        calls.append((rank, contributor, label, list(accepted)))
        if rank == 0:
            return _scores(.5)
        return _scores({1: .6, 2: .6, 3: .4}[rank])

    state.run(runner, temp)
    assert [call[:3] for call in calls] == [(0, "V3.0-BASELINE", ""),
        (1, "c", "z"), (2, "a", "y"), (3, "b", "x")]
    assert state.state["accepted"] == [["c", "z"]]
    assert state.state["next_rank"] == 4
    assert not temp.exists()
    lines = (root / "factory.log").read_text().splitlines()
    assert len(lines) == 8
    assert sum("STARTED" in line for line in lines) == 4
    assert sum("ACCEPTED" in line for line in lines) == 1
    assert sum("REJECTED" in line for line in lines) == 2
    assert (root / "accepted_subsets.csv").read_text().splitlines() == [
        "source_dataset,label", "c,z"]
    label_rows = pd.read_csv(root / "episode_label_results.csv")
    assert len(label_rows) == 9
    assert {"episode", "contributor", "label", "v0_recall", "v0_ext_recall"} <= set(label_rows.columns)
    resumed = FactoryState(root, queue, {"queue": "h"}, folds)
    assert resumed.state["next_rank"] == 4


def test_interrupted_episode_retries_without_duplicate_commit(tmp_path):
    root, temp = tmp_path / "factory", tmp_path / "tmp"
    state = FactoryState(root, _queue().iloc[:1], {"queue": "h"}, {"v0-1": 0})
    attempts = []

    def fail_once(rank, contributor, label, accepted, work, baseline):
        attempts.append(rank)
        if rank == 0:
            return _scores(.5)
        if len(attempts) == 2:
            raise RuntimeError("interrupted")
        return _scores(.5)

    try:
        state.run(fail_once, temp)
    except RuntimeError:
        pass
    assert state.state["next_rank"] == 1
    assert state.state["in_progress"]["rank"] == 1
    resumed = FactoryState(root, _queue().iloc[:1], {"queue": "h"}, {"v0-1": 0})
    resumed.run(fail_once, temp)
    assert resumed.state["next_rank"] == 2
    assert len(resumed.state["episodes"]) == 1
    lines = (root / "factory.log").read_text().splitlines()
    assert len(lines) == 4  # baseline lifecycle plus one committed candidate pair
    assert sum("001 | STARTED" in line for line in lines) == 1


def test_log_render_appends_without_truncating_existing_events(tmp_path):
    root = tmp_path / "factory"
    state = FactoryState(root, _queue().iloc[:1], {"queue": "h"}, {"v0-1": 0})
    log = root / "factory.log"
    log.write_text("2026-01-01 00:00:00 | BASELINE | STARTED | V3.0-BASELINE\n")
    state.state["in_progress"] = {"rank": 1, "started_at": "2026-01-01 00:01:00",
        "contributor": "c", "label": "z"}
    state.commit()
    lines = log.read_text().splitlines()
    assert lines[0].endswith("BASELINE | STARTED | V3.0-BASELINE")
    assert lines[1].endswith("001 | STARTED | c | z")
