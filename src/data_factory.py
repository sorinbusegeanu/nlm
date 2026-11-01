"""Validate official data and create immutable grouped dataset artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REQUIRED_TRAIN = {"id", "prompt_id", "text", "model"}
REQUIRED_TEST = {"id", "text"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data(train: pd.DataFrame, test: pd.DataFrame | None = None, expected_classes: int | None = 18) -> list[str]:
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
    n_classes = train["model"].nunique()
    if n_classes < 2:
        errors.append("train must contain at least two labels")
    if expected_classes is not None and n_classes != expected_classes:
        errors.append(f"train must contain exactly {expected_classes} labels; found {n_classes}")
    if test is not None:
        missing = REQUIRED_TEST - set(test.columns)
        if missing:
            errors.append(f"test is missing columns: {sorted(missing)}")
        elif test["id"].isna().any() or test["id"].duplicated().any():
            errors.append("test.id must be present and unique")
        elif test["text"].isna().any() or test["text"].astype(str).str.strip().eq("").any():
            errors.append("test.text contains missing or empty passages")
        elif set(train.id.astype(str)) & set(test.id.astype(str)):
            errors.append("train and test ids overlap")
    return errors


def make_splits(train: pd.DataFrame, seed: int = 2026, validation_fraction: float = .2,
                candidates: int = 1000) -> pd.DataFrame:
    """Choose a deterministic, label-balanced row split without separating prompts."""
    if not 0 < validation_fraction < 1 or candidates < 1:
        raise ValueError("validation_fraction must be in (0,1) and candidates >= 1")
    y, groups = train["model"].astype(str), train["prompt_id"].astype(str)
    target = y.value_counts(normalize=True).sort_index()
    best = None
    # GroupShuffleSplit preserves prompt integrity; score candidates using only
    # labels and split sizes, never model predictions.
    for candidate in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction,
                                     random_state=seed + candidate)
        fit_idx, val_idx = next(splitter.split(train, y, groups))
        val_y = y.iloc[val_idx]
        if set(val_y.unique()) != set(target.index):
            continue
        fit_y = y.iloc[fit_idx]
        if set(fit_y.unique()) != set(target.index):
            continue
        row_fraction_error = abs(len(val_idx) / len(train) - validation_fraction)
        class_error = float((val_y.value_counts(normalize=True).reindex(target.index) - target).abs().mean())
        objective = row_fraction_error + class_error
        if best is None or objective < best[0]:
            best = (objective, val_idx)
    if best is None:
        raise ValueError("could not create a prompt-grouped split containing every class on both sides")
    assignments = train[["id", "prompt_id", "model"]].copy()
    assignments["split"] = "train"
    assignments.iloc[best[1], assignments.columns.get_loc("split")] = "validation"
    train_groups = set(assignments.loc[assignments.split.eq("train"), "prompt_id"])
    val_groups = set(assignments.loc[assignments.split.eq("validation"), "prompt_id"])
    if train_groups & val_groups:
        raise RuntimeError("prompt leakage in train/validation split")
    return assignments


def prepare(train_path: str | Path, output_dir: str | Path, test_path: str | Path | None = None, *, seed: int = 2026, validation_fraction: float = .2, candidates: int = 1000, expected_classes: int | None = 18) -> dict:
    train_path, output_dir = Path(train_path), Path(output_dir)
    train = pd.read_csv(train_path, dtype={"id": "string", "prompt_id": "string", "model": "string"})
    test = pd.read_csv(test_path, dtype={"id": "string"}) if test_path else None
    errors = validate_data(train, test, expected_classes=expected_classes)
    if errors:
        raise ValueError("Invalid dataset:\n- " + "\n- ".join(errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    train = train.sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(train.model.unique().tolist())
    assignments = make_splits(train, seed, validation_fraction, candidates)
    dataset_hash = _hash_file(train_path)
    test_hash = _hash_file(Path(test_path)) if test_path else None
    lengths = train.text.astype(str).str.len()
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = seed
        def detect_safely(text):
            try: return detect(text) if str(text).strip() else "unknown"
            except Exception: return "unknown"
        languages = train.text.astype(str).map(detect_safely)
        language_status = "langdetect"
    except ImportError:
        languages = pd.Series(["unavailable"] * len(train), index=train.index)
        language_status = "install declared langdetect dependency to enable language counts"
    except Exception:
        languages = pd.Series(["unknown"] * len(train), index=train.index)
        language_status = "language detection returned errors; affected rows labeled unknown"
    language_lengths = train.assign(_language=languages.to_numpy(), _length=lengths.to_numpy()).groupby("_language")["_length"].agg(["count", "median", "min", "max"])
    group_sizes = train.groupby("prompt_id", dropna=False).size()
    split_payload = assignments[["id", "split"]].to_csv(index=False, lineterminator="\n").encode()
    train_rows = assignments.split.eq("train").sum()
    validation_rows = assignments.split.eq("validation").sum()
    profile = {
        "dataset_version": "v0", "dataset_sha256": dataset_hash, "test_sha256": test_hash, "rows": len(train),
        "prompt_groups": int(train.prompt_id.nunique()), "classes": classes,
        "class_counts": {str(k): int(v) for k, v in train.model.value_counts().sort_index().items()},
        "class_prompt_groups": {str(k): int(v) for k, v in train.groupby("model").prompt_id.nunique().items()},
        "prompt_group_size": {"min": int(group_sizes.min()), "median": float(group_sizes.median()), "max": int(group_sizes.max())},
        "duplicate_prompt_text_rows": int(train.duplicated(["prompt_id", "text"]).sum()),
        "duplicate_ids": int(train.id.duplicated().sum()),
        "text_length_chars": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
        "language_detection": language_status,
        "language_counts": {str(k): int(v) for k, v in languages.value_counts().sort_index().items()},
        "text_length_chars_by_language": {str(k): {"rows": int(v["count"]), "median": float(v["median"]), "min": int(v["min"]), "max": int(v["max"])} for k, v in language_lengths.iterrows()},
        "quality_exceptions": {"short_text_rows_under_50_chars": int(lengths.lt(50).sum()),
                               "duplicate_prompt_text_rows": int(train.duplicated(["prompt_id", "text"]).sum())},
        "seed": seed, "validation_fraction": validation_fraction, "split_candidates": candidates,
        "train_rows": int(train_rows), "validation_rows": int(validation_rows),
        "train_class_counts": {str(k): int(v) for k, v in train.loc[assignments.split.eq("train"), "model"].value_counts().sort_index().items()},
        "validation_class_counts": {str(k): int(v) for k, v in train.loc[assignments.split.eq("validation"), "model"].value_counts().sort_index().items()},
        "split_sha256": hashlib.sha256(split_payload).hexdigest(),
    }
    existing_profile = output_dir / "dataset_profile.json"
    if existing_profile.exists():
        prior = json.loads(existing_profile.read_text())
        if prior.get("dataset_sha256") != profile["dataset_sha256"]:
            raise FileExistsError("V0 artifacts already exist for different training data; choose a new output directory/version")
        if prior.get("test_sha256") != profile["test_sha256"]:
            raise FileExistsError("V0 artifacts already exist for different test data; choose a new output directory/version")
        if prior.get("split_sha256") != profile["split_sha256"]:
            raise FileExistsError("V0 fold assignments already exist and differ; refusing to replace frozen splits")
    train.to_parquet(output_dir / "dataset_v0.parquet", index=False)
    assignments.to_parquet(output_dir / "splits_v0.parquet", index=False)
    if test is not None:
        test.to_parquet(output_dir / "test_v0.parquet", index=False)
    (output_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile
