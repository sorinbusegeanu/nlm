#!/usr/bin/env python3
"""Run the resumable V3 contributor-level training factory."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v3_contributor_label_training_factory import (
    DATA, LABELS, _frame, _route_cv, _train_models,
)
from src.v3_contributor_label_factory import fixed_row_folds, sha256_file
from src.v3_contributor_factory import (
    ContributorFactoryState, make_contributor_ranking,
    make_size_contributor_ranking, ranking_queue,
)

OUT = ROOT / "artifacts/analysis/v3/contributor_factory"
TMP = OUT / "tmp"
LABEL_RANKING = ROOT / "artifacts/analysis/v3/contributor_label_ranking.csv"
BASELINE_BA = 0.275096810690573
BASELINE_DATASET = "v3.0-baseline"


def combine_first_four_contributors(ranking: pd.DataFrame):
    """Collapse the four smallest V3.1 contributors into the first episode."""
    if len(ranking) < 4:
        raise ValueError("V3.1 size ranking must contain at least four contributors")
    first_four = ranking.iloc[:4]
    combined_name = "COMBINED first 4 size-ranked contributors"
    members = first_four["Source dataset"].astype(str).tolist()
    combined = {"Rank": 1, "Source dataset": combined_name,
                "Available rows": int(first_four["Available rows"].sum()),
                "Labels": int(first_four["Labels"].max())}
    remainder = ranking.iloc[4:].copy().reset_index(drop=True)
    remainder["Rank"] = range(2, len(remainder) + 2)
    result = pd.concat([pd.DataFrame([combined]), remainder], ignore_index=True)
    return result, {1: members}, {combined_name: members}


def _runner(v0: pd.DataFrame, v0ext: pd.DataFrame, baseline: pd.DataFrame,
            v3: pd.DataFrame, folds: dict[str, int],
            episode_sources: dict[int, list[str]] | None = None,
            source_members: dict[str, list[str]] | None = None):
    valid = pd.concat([v0, v0ext], ignore_index=True).reset_index(drop=True)
    base_ids = set(baseline.id.astype(str))
    episode_sources = episode_sources or {}
    source_members = source_members or {}

    def run(rank: int, source: str, accepted: list[str], work: Path, prior_baseline):
        sources = {member for accepted_source in accepted
                   for member in source_members.get(accepted_source, [accepted_source])}
        if rank > 0:
            sources.update(episode_sources.get(rank, [source]))
        if sources:
            candidate = v3.loc[
                ~v3.id.astype(str).isin(base_ids)
                & v3.source_dataset.astype(str).isin(sources)
            ]
            train = pd.concat([baseline, candidate], ignore_index=True).drop_duplicates("id")
        else:
            train = baseline.copy()
        if train.id.astype(str).duplicated().any():
            raise ValueError("candidate training rows have duplicate response IDs")
        probabilities = _train_models(train, valid, work / "models")
        return _route_cv(probabilities, valid, folds)
    return run


def _source_digest(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", choices=("v3.0", "v3.1"), default="v3.0",
                        help="V3 source version; v3.1 uses ascending contributor size")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="run at most this many episodes in this invocation")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")

    version = args.dataset_version
    out = (ROOT / "artifacts/analysis/v3.1/contributor_factory"
           if version == "v3.1" else ROOT / "artifacts/analysis/v3/contributor_factory")
    tmp = out / "tmp"
    out.mkdir(parents=True, exist_ok=True)
    ranking_path = out / "contributor_ranking.csv"
    baseline_path = DATA / "dataset_v3.0_baseline.parquet"
    dataset_path = DATA / f"dataset_{version}.parquet"
    baseline = _frame(baseline_path, BASELINE_DATASET)
    v3 = _frame(dataset_path, version)
    if "source_dataset" not in v3:
        raise ValueError(f"{version} rows must preserve source_dataset")
    if version == "v3.1":
        ranking = make_size_contributor_ranking(v3, baseline.id.astype(str))
        ranking, episode_sources, source_members = combine_first_four_contributors(ranking)
        combined_sources = episode_sources[1]
        combined_labels = v3.loc[v3.source_dataset.astype(str).isin(combined_sources), "model"].nunique()
        ranking.loc[0, "Labels"] = int(combined_labels)
        ranking.to_csv(ranking_path, index=False)
    else:
        source_ranking = pd.read_csv(LABEL_RANKING)
        make_contributor_ranking(source_ranking).to_csv(ranking_path, index=False)
    queue = ranking_queue(ranking_path)
    if version == "v3.0" and len(queue) != 53:
        raise ValueError(f"expected 53 contributors, found {len(queue)}")
    v0 = _frame(DATA / "dataset_v0.parquet", "v0").sort_values(
        "id", kind="stable").reset_index(drop=True)
    v0ext = _frame(DATA / "dataset_v0_ext.parquet", "v0_ext").reset_index(drop=True)
    if set(v0.id.astype(str)) & set(v0ext.id.astype(str)):
        raise ValueError("V0 and V0-EXT response IDs must be disjoint")
    validation = pd.concat([v0, v0ext], ignore_index=True).reset_index(drop=True)
    folds = fixed_row_folds(validation)

    config_paths = {
        "svc_config": ROOT / "configs/models/svc.yaml",
        "catboost_config": ROOT / "configs/models/catboost_stylometry.yaml",
        "ultrabert_config": ROOT / "configs/models/ultrabert380.yaml",
    }
    input_paths = {"contributor_ranking": ranking_path,
        "baseline": baseline_path, "v3_dataset": dataset_path,
        "v0": DATA / "dataset_v0.parquet", "v0_ext": DATA / "dataset_v0_ext.parquet",
        **config_paths}
    if version == "v3.0":
        input_paths["label_ranking"] = LABEL_RANKING
    hashes = _source_digest(input_paths)
    initial_baseline = {"overall_ba": BASELINE_BA,
                        "baseline_source": "original V3.0-BASELINE result; identical training/evaluation configuration",
                        "baseline_dataset": str(baseline_path.relative_to(ROOT))}
    state = ContributorFactoryState(out, queue, hashes, folds, initial_baseline)
    runner = _runner(v0, v0ext, baseline, v3, folds,
                     episode_sources=locals().get("episode_sources"),
                     source_members=locals().get("source_members"))

    catalog_path = ROOT / "configs/ensemble_catalog_v2.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    experiment_id = "V3.1-CONTRIBUTOR-FACTORY" if version == "v3.1" else "V3-CONTRIBUTOR-FACTORY"
    entry = next(row for row in catalog["experiments"]
                 if row.get("id") == experiment_id)
    entry.update({"status": "Running", "current_stage": f"episode rank {state.state['next_rank']}",
                  "factory_state": str((out / "factory_state.json").relative_to(ROOT)),
                  "processing_queue": str(ranking_path.relative_to(ROOT)),
                  "training_dataset": f"v3.0-baseline + accepted {version} contributors + current contributor"})
    tmp_catalog = catalog_path.with_suffix(".tmp")
    tmp_catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_catalog, catalog_path)
    queue_to_run = state.queue
    if args.max_episodes is not None:
        end = min(state.state["next_rank"] - 1 + args.max_episodes, len(queue))
        state.queue = state.queue.iloc[:end].reset_index(drop=True)
    try:
        state.run(runner, tmp)
    except Exception as exc:
        entry["status"] = "Failed episode; resumable"
        entry["current_stage"] = str(exc)
        tmp_catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_catalog, catalog_path)
        raise
    finally:
        state.queue = queue_to_run
    entry["status"] = "Complete" if state.state["next_rank"] > len(queue) else "Paused"
    entry["current_stage"] = ("Complete" if entry["status"] == "Complete" else
                               f"Paused at queue rank {state.state['next_rank']}")
    tmp_catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_catalog, catalog_path)


if __name__ == "__main__":
    main()
