"""Validate official data and create immutable grouped dataset artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

REQUIRED_TRAIN = {"id", "prompt_id", "text", "model"}
REQUIRED_TEST = {"id", "text"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data(train: pd.DataFrame, test: pd.DataFrame | None = None) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TRAIN - set(train.columns)
    if missing:
        errors.append(f"train is missing columns: {sorted(missing)}")
        return errors
    if train.empty:
        errors.append("train is empty")
    for col in ("id", "prompt_id", "model"):
        if train[col].isna().any():
            errors.append(f"train.{col} contains missing values")
    if train["id"].duplicated().any():
        errors.append("train.id must be unique")
    if train["text"].isna().any() or train["text"].astype(str).str.strip().eq("").any():
        errors.append("train.text contains missing or empty passages")
    if train["model"].nunique() < 2:
        errors.append("train must contain at least two labels")
    if test is not None:
        missing = REQUIRED_TEST - set(test.columns)
        if missing:
            errors.append(f"test is missing columns: {sorted(missing)}")
        elif test["id"].isna().any() or test["id"].duplicated().any():
            errors.append("test.id must be present and unique")
        elif test["text"].isna().any() or test["text"].astype(str).str.strip().eq("").any():
            errors.append("test.text contains missing or empty passages")
    return errors


def make_splits(train: pd.DataFrame, seed: int = 2026, holdout_fraction: float = .2, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row assignments and fold ids; all splits are isolated by prompt."""
    if not 0 < holdout_fraction < 1 or n_splits < 2:
        raise ValueError("holdout_fraction must be in (0,1) and n_splits >= 2")
    y, groups = train["model"].astype(str), train["prompt_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
    dev_idx, holdout_idx = next(splitter.split(train, y, groups))
    dev = train.iloc[dev_idx]
    holdout = train.iloc[holdout_idx]
    if set(dev.prompt_id) & set(holdout.prompt_id):
        raise RuntimeError("prompt leakage in holdout split")
    counts = dev.groupby("model").prompt_id.nunique()
    if counts.min() < n_splits:
        raise ValueError(f"development has fewer than {n_splits} groups for a class")
    folds = np.full(len(train), -1, dtype=np.int8)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (fit_rel, val_rel) in enumerate(cv.split(dev, y.iloc[dev_idx], groups.iloc[dev_idx])):
        fit_groups = set(dev.iloc[fit_rel].prompt_id)
        val_groups = set(dev.iloc[val_rel].prompt_id)
        if fit_groups & val_groups:
            raise RuntimeError("prompt leakage in CV split")
        folds[dev_idx[val_rel]] = fold_id
    assignments = train[["id", "prompt_id", "model"]].copy()
    assignments["split"] = "development"
    assignments.loc[holdout_idx, "split"] = "holdout"
    assignments["fold"] = folds
    fold_rows = assignments.loc[assignments.split.eq("development"), ["id", "prompt_id", "fold"]].copy()
    return assignments, fold_rows


def prepare(train_path: str | Path, output_dir: str | Path, test_path: str | Path | None = None, *, seed: int = 2026, holdout_fraction: float = .2, n_splits: int = 5) -> dict:
    train_path, output_dir = Path(train_path), Path(output_dir)
    train = pd.read_csv(train_path, dtype={"id": "string", "prompt_id": "string", "model": "string"})
    test = pd.read_csv(test_path, dtype={"id": "string"}) if test_path else None
    errors = validate_data(train, test)
    if errors:
        raise ValueError("Invalid dataset:\n- " + "\n- ".join(errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    train = train.sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(train.model.unique().tolist())
    assignments, folds = make_splits(train, seed, holdout_fraction, n_splits)
    dataset_hash = _hash_file(train_path)
    train.to_parquet(output_dir / "dataset_v0.parquet", index=False)
    assignments.to_parquet(output_dir / "folds_v0.parquet", index=False)
    if test is not None:
        test.to_parquet(output_dir / "test_v0.parquet", index=False)
    lengths = train.text.astype(str).str.len()
    profile = {
        "dataset_version": "v0", "dataset_sha256": dataset_hash, "rows": len(train),
        "prompt_groups": int(train.prompt_id.nunique()), "classes": classes,
        "class_counts": {str(k): int(v) for k, v in train.model.value_counts().sort_index().items()},
        "class_prompt_groups": {str(k): int(v) for k, v in train.groupby("model").prompt_id.nunique().items()},
        "duplicate_ids": int(train.id.duplicated().sum()), "duplicate_prompt_text_rows": int(train.duplicated(["prompt_id", "text"]).sum()),
        "text_length_chars": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
        "seed": seed, "holdout_fraction": holdout_fraction, "n_splits": n_splits,
        "development_rows": int(assignments.split.eq("development").sum()), "holdout_rows": int(assignments.split.eq("holdout").sum()),
    }
    (output_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile
