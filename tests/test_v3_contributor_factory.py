import json

import pandas as pd

from src.v3_contributor_factory import (ContributorFactoryState,
    make_contributor_ranking, make_size_contributor_ranking)
from scripts.run_v3_contributor_training_factory import combine_first_four_contributors


def test_contributor_ranking_uses_equal_label_mean_and_deterministic_ties():
    frame = pd.DataFrame([
        {"Contributor": "z", "Label": "a", "Available rows": 20, "Relative accuracy": 10.0},
        {"Contributor": "z", "Label": "b", "Available rows": 30, "Relative accuracy": 0.0},
        {"Contributor": "a", "Label": "a", "Available rows": 25, "Relative accuracy": 5.0},
        {"Contributor": "b", "Label": "a", "Available rows": 60, "Relative accuracy": -1.0},
    ])
    ranked = make_contributor_ranking(frame)
    assert ranked["Source dataset"].tolist() == ["z", "a", "b"]
    assert ranked["Available rows"].tolist() == [50, 25, 60]
    assert ranked["Labels"].tolist() == [2, 1, 1]
    assert ranked["Quality score"].tolist() == [5.0, 5.0, -1.0]
    # z precedes a on equal quality because it has more available rows.


def test_v31_size_ranking_excludes_baseline_and_breaks_ties_alphabetically():
    rows = pd.DataFrame([
        {"id": "a0", "source_dataset": "alpha", "model": "x"},
        {"id": "a1", "source_dataset": "alpha", "model": "y"},
        {"id": "b0", "source_dataset": "beta", "model": "x"},
        {"id": "b1", "source_dataset": "beta", "model": "x"},
        {"id": "c0", "source_dataset": "charlie", "model": "x"},
    ])
    ranked = make_size_contributor_ranking(rows, {"a0", "b0", "c0"})
    assert ranked["Source dataset"].tolist() == ["charlie", "alpha", "beta"]
    assert ranked["Available rows"].tolist() == [0, 1, 1]
    assert ranked["Labels"].tolist() == [0, 1, 1]
    assert ranked["Rank"].tolist() == [1, 2, 3]


def test_v31_initial_four_contributors_are_one_episode():
    ranking = pd.DataFrame({
        "Rank": [1, 2, 3, 4, 5],
        "Source dataset": ["a", "b", "c", "d", "e"],
        "Available rows": [2, 6, 10, 94, 100],
        "Labels": [1, 2, 3, 4, 5],
    })
    combined, episode_sources, source_members = combine_first_four_contributors(ranking)
    assert combined["Rank"].tolist() == [1, 2]
    assert combined["Source dataset"].tolist() == [
        "COMBINED first 4 size-ranked contributors", "e"]
    assert combined["Available rows"].tolist() == [112, 100]
    assert episode_sources[1] == ["a", "b", "c", "d"]
    assert source_members[combined.loc[0, "Source dataset"]] == ["a", "b", "c", "d"]


def test_contributor_factory_acceptance_rejection_and_append_log(tmp_path):
    ranking = pd.DataFrame({"Rank": [1, 2], "Source dataset": ["good", "bad"],
        "Available rows": [10, 8], "Labels": [2, 1], "Quality score": [1.0, 0.0]})
    folds = {"v0": 0}
    state = ContributorFactoryState(tmp_path, ranking, {"input": "hash"}, folds,
                                   {"overall_ba": .5})
    def run(rank, source, accepted, work, baseline):
        assert work.is_dir()
        assert baseline["overall_ba"] == (.6 if rank == 2 else .5)
        return {"overall_ba": .6 if rank == 1 else .55,
            "v0_ba": .6 if rank == 1 else .55,
            "v0_ext_ba": .6 if rank == 1 else .55,
            "per_label": [{"label": f"l{i}", "v0_recall": .5,
                "v0_ext_recall": .5, "mean_recall": .5} for i in range(18)]}
    temp = tmp_path / "tmp"
    state.run(run, temp)
    assert state.state["accepted"] == ["good"]
    assert state.state["next_rank"] == 3
    assert not temp.exists()
    log = (tmp_path / "factory.log").read_text().splitlines()
    assert len(log) == 4
    assert "001 | STARTED | good" in log[0]
    assert "001 | ACCEPTED" in log[1]
    assert "002 | STARTED | bad" in log[2]
    assert "002 | REJECTED" in log[3]
    assert (tmp_path / "accepted_contributors.csv").read_text().splitlines() == [
        "source_dataset", "good"]
    labels = pd.read_csv(tmp_path / "episode_label_results.csv")
    assert len(labels) == 36
    assert labels.columns.tolist() == [
        "episode", "source_dataset", "label", "v0_recall", "v0_ext_recall", "mean_recall"]


def test_contributor_factory_resume_reconciles_missing_result_event(tmp_path):
    ranking = pd.DataFrame({"Rank": [1], "Source dataset": ["c"],
        "Available rows": [5], "Labels": [1], "Quality score": [0.0]})
    state = ContributorFactoryState(tmp_path, ranking, {"i": "x"}, {"r": 0},
                                    {"overall_ba": .4})
    state.state["in_progress"] = {"rank": 1, "source_dataset": "c",
                                   "started_at": "2026-01-01 00:00:00"}
    state.commit()
    # Simulate an episode decision committed before its result log line was flushed.
    state.state["episodes"] = [{"rank": 1, "episode": 1, "source_dataset": "c",
        "started_at": "2026-01-01 00:00:00", "finished_at": "2026-01-01 00:01:00",
        "decision": "REJECTED", "baseline_ba": .4, "mean_ba": .3,
        "improvement_pp": -10.0, "label_results": []}]
    state.state["in_progress"] = None
    state.state["next_rank"] = 2
    state.commit()
    lines = (tmp_path / "factory.log").read_text().splitlines()
    assert len(lines) == 2
    assert "001 | REJECTED" in lines[1]
    state.commit()
    assert len((tmp_path / "factory.log").read_text().splitlines()) == 2
