"""Create balanced Gemini and GLM specialist datasets from V2.14."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.dataset_catalog import render_docs
from src.pipeline import _normal_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"
SOURCE = DATA_DIR / "dataset_v2.14.parquet"
SOURCE_PROFILE = DATA_DIR / "dataset_profile_v2.14.json"
SEED = 2026
GROUPS = {
    "v2.14-gemini": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "v2.14-glm": ["glm-5.3", "glm-5.3-flash"],
}


def prompt_components(rows: pd.DataFrame) -> pd.Series:
    """Join prompt IDs and duplicate normalized prompt strings into leakage-safe groups."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[str, int] = {}
    for index, row in enumerate(rows.itertuples(index=False)):
        keys = [f"id:{str(getattr(row, 'prompt_id'))}"]
        for column in ("prompt_canon", "prompt", "prompt_text"):
            if column in rows.columns:
                value = getattr(row, column, "")
                if pd.isna(value):
                    continue
                normalized = _normal_text(value)
                if normalized:
                    keys.append(f"text:{normalized}")
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    return pd.Series([f"group-{find(i):08d}" for i in range(len(rows))], index=rows.index)


def grouped_balanced_split(rows: pd.DataFrame, *, seed: int = SEED,
                           validation_fraction: float = .2,
                           candidates: int = 250) -> pd.DataFrame:
    """Find a deterministic row- and class-balanced split without dividing prompt components."""
    labels = sorted(rows.model.astype(str).unique())
    group_ids = prompt_components(rows).astype(str)
    counts = pd.crosstab(group_ids, rows.model.astype(str)).reindex(columns=labels, fill_value=0)
    matrix = counts.to_numpy(dtype=np.int64)
    totals = matrix.sum(axis=0)
    def score(values: np.ndarray) -> float:
        row_error = abs(float(values.sum()) / max(1, len(rows)) - validation_fraction)
        class_error = float(np.mean(np.abs(values / np.maximum(totals, 1) - validation_fraction)))
        return 2.0 * row_error + class_error

    best_ids = None
    best_score = float("inf")
    for attempt in range(candidates):
        order = np.random.default_rng(seed + attempt).permutation(len(counts))
        selected = np.zeros(len(labels), dtype=np.int64)
        selected_groups: list[int] = []
        current_score = score(selected)
        for index in order:
            proposed = selected + matrix[index]
            proposed_score = score(proposed)
            if proposed_score + 1e-12 < current_score:
                selected = proposed
                selected_groups.append(int(index))
                current_score = proposed_score
        if (np.any(selected == 0) or np.any(selected == totals)
                or not .18 <= float(selected.sum()) / len(rows) <= .22):
            continue
        if current_score < best_score:
            best_score = current_score
            best_ids = set(counts.index[np.asarray(selected_groups, dtype=int)].astype(str))
    if best_ids is None:
        raise ValueError("could not construct an isolated prompt-group 80/20 split with every class represented")
    output = rows[["id", "prompt_id", "model"]].copy()
    output["split"] = np.where(group_ids.isin(best_ids), "validation", "train")
    return output


def digest_csv(rows: pd.DataFrame) -> str:
    return hashlib.sha256(rows.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def create_dataset(version: str, labels: list[str], source: pd.DataFrame,
                   source_hash: str, v0: pd.DataFrame, *, force: bool = False) -> dict:
    target = source.loc[source.model.astype(str).isin(labels)].copy()
    before = target.model.astype(str).value_counts().reindex(labels, fill_value=0)
    if (before == 0).any():
        raise ValueError(f"{version} missing source labels: {before[before == 0].to_dict()}")
    quota = int(before.min())
    pieces = [target.loc[target.model.astype(str).eq(label)].sample(
        n=quota, replace=False, random_state=SEED) for label in labels]
    selected_ids = set(pd.concat(pieces).id.astype(str))
    result = target.loc[target.id.astype(str).isin(selected_ids)].copy()
    result = result.sort_values("id", kind="stable").reset_index(drop=True)
    if result.model.astype(str).value_counts().to_dict() != {label: quota for label in labels}:
        raise RuntimeError(f"{version}: deterministic class balancing failed")
    if result.id.duplicated().any():
        raise ValueError(f"{version}: selected IDs are not unique")

    # Enforce V0 separation before either internal selection or final evaluation.
    v0_ids = set(v0.id.astype(str))
    ids_overlap = set(result.id.astype(str)) & v0_ids
    text_overlap = set(result.text.astype(str).map(_normal_text)) & set(v0.text.astype(str).map(_normal_text))
    prompt_overlap = set(result.prompt_id.astype(str)) & set(v0.prompt_id.astype(str))
    if ids_overlap or text_overlap or prompt_overlap:
        raise ValueError(f"{version} overlaps V0: IDs={len(ids_overlap)}, normalized text={len(text_overlap)}, prompts={len(prompt_overlap)}")

    assignments = grouped_balanced_split(result, seed=SEED, validation_fraction=.2, candidates=250)
    assignments = assignments.sort_values("id", kind="stable").reset_index(drop=True)
    train_rows = int(assignments.split.eq("train").sum())
    valid_rows = int(assignments.split.eq("validation").sum())
    if set(assignments.loc[assignments.split.eq("train"), "prompt_id"]) & set(
            assignments.loc[assignments.split.eq("validation"), "prompt_id"]):
        raise RuntimeError(f"{version}: prompt groups crossed the split")
    train_classes = set(assignments.loc[assignments.split.eq("train"), "model"])
    valid_classes = set(assignments.loc[assignments.split.eq("validation"), "model"])
    if train_classes != set(labels) or valid_classes != set(labels):
        raise RuntimeError(f"{version}: not all target labels occur in both split partitions")

    artifact = DATA_DIR / f"dataset_{version}.parquet"
    split_path = DATA_DIR / f"splits_{version}.parquet"
    profile_path = DATA_DIR / f"dataset_profile_{version}.json"
    if any(path.exists() for path in (artifact, split_path, profile_path)) and not force:
        raise FileExistsError(f"{version} output already exists; inspect it before replacing: {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_tmp = artifact.with_suffix(".parquet.tmp")
    split_tmp = split_path.with_suffix(".parquet.tmp")
    result.to_parquet(artifact_tmp, index=False)
    assignments.to_parquet(split_tmp, index=False)
    artifact_tmp.replace(artifact)
    split_tmp.replace(split_path)
    length = result.assign(_chars=result.text.astype(str).str.len()).groupby("model")._chars.agg(
        minimum="min", median="median", maximum="max")
    profile = {
        "dataset_version": version,
        "dataset_sha256": digest_csv(result),
        "rows": int(len(result)),
        "classes": sorted(labels),
        "class_counts": {label: quota for label in sorted(labels)},
        "prompt_groups": int(result.prompt_id.nunique()),
        "parent_dataset": "v2.14",
        "parent_dataset_sha256": source_hash,
        "selection_method": "deterministic_per_class_random_sample",
        "seed": SEED,
        "balanced_target_per_class": quota,
        "source_class_counts": {str(k): int(v) for k, v in before.items()},
        "split_method": "GroupShuffleSplit; deterministic candidate search minimizing row and class proportion error",
        "split_seed": SEED,
        "split_validation_fraction_requested": 0.2,
        "split_train_rows": train_rows,
        "split_validation_rows": valid_rows,
        "split_validation_fraction_actual": valid_rows / len(result),
        "split_prompt_groups": int(assignments.prompt_id.nunique()),
        "split_assignment_path": str(split_path.relative_to(ROOT)),
        "split_assignment_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "validation_dataset_version": "prompt_grouped_80_20",
        "validation_isolation_from_v0": {
            "id_overlap": 0, "normalized_response_overlap": 0, "prompt_group_overlap": 0,
            "v0_rows_for_final_evaluation": int(v0.model.astype(str).isin(labels).sum()),
        },
        "response_length_chars_by_class": {str(label): {
            "min": int(row["minimum"]), "median": float(row["median"]), "max": int(row["maximum"])}
            for label, row in length.iterrows()},
        "original_schema": list(result.columns),
        "preserved_source_row_schema": True,
    }
    temporary = profile_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary.replace(profile_path)
    return {"version": version, "labels": labels, "quota_per_class": quota,
            "rows": len(result), "train_rows": train_rows, "validation_rows": valid_rows,
            "validation_fraction": valid_rows / len(result), "artifact": str(artifact.relative_to(ROOT)),
            "split_path": str(split_path.relative_to(ROOT)), "profile": str(profile_path.relative_to(ROOT)),
            "dataset_sha256": profile["dataset_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="rebuild deterministic artifacts from V2.14")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("the specialist dataset recipe is fixed to seed 2026")
    source = pd.read_parquet(SOURCE)
    source_profile = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    source_hash = source_profile["dataset_sha256"]
    v0 = pd.read_parquet(DATA_DIR / "dataset_v0.parquet")
    catalog_path = ROOT / "configs/datasets.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    existing = {item["id"] for item in catalog["datasets"]}
    entries = []
    for version, labels in GROUPS.items():
        if version in existing and not args.force:
            raise ValueError(f"dataset catalog already contains {version}")
        summary = create_dataset(version, labels, source, source_hash, v0, force=args.force)
        entries.append({
            "id": version,
            "name": "V2.14 Gemini specialist" if version.endswith("gemini") else "V2.14 GLM specialist",
            "section": "Specialist classifier datasets",
            "description": (f"Deterministically balanced V2.14 subset for labels {', '.join(labels)}; "
                            f"{summary['quota_per_class']:,} rows per label, seed {SEED}."),
            "artifact": summary["artifact"], "profile": summary["profile"],
            "validation": (f"Prompt-grouped 80/20 split for checkpoint selection; full V0 rows for these labels "
                           "are predicted and evaluated after training."),
        })
    if args.force:
        catalog_text = catalog_path.read_text(encoding="utf-8")
        # Remove only the two generated catalog stanzas before appending their refreshed versions.
        import re
        for version in GROUPS:
            catalog_text = re.sub(
                rf"(?ms)^  - id: {re.escape(version)}\n.*?(?=^  - id: |\Z)", "", catalog_text)
        catalog_path.write_text(catalog_text.rstrip() + "\n", encoding="utf-8")
    with catalog_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
        for entry in entries:
            lines = yaml.safe_dump(entry, sort_keys=False, allow_unicode=True).rstrip().splitlines()
            stream.write("  - " + lines[0] + "\n")
            for line in lines[1:]:
                stream.write("    " + line + "\n")
    render_docs(catalog_path, ROOT / "docs/datasets.md")
    print(json.dumps(entries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
