"""Validate official data and create immutable grouped dataset artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REQUIRED_TRAIN = {"id", "prompt_id", "text", "model"}
REQUIRED_TEST = {"id", "text"}
OPENPOOL_VERTEX_G37_SOURCES = {
    "alperen5252525/ntl-vertex-gemini-proxy-gpt-generations",
    "davidpiln/ntl-vertex-gemini-gen",
}


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


def _dataset_artifact_names(dataset_version: str) -> tuple[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_version):
        raise ValueError("dataset_version may contain only letters, digits, dots, underscores, and hyphens")
    profile_name = "dataset_profile.json" if dataset_version == "v0" else f"dataset_profile_{dataset_version}.json"
    return f"dataset_{dataset_version}.parquet", profile_name


def _prepare_frame(train: pd.DataFrame, output_dir: str | Path, *, dataset_version: str = "v0",
                   test: pd.DataFrame | None = None, seed: int = 2026, validation_fraction: float = .2,
                   candidates: int = 1000, expected_classes: int | None = 18,
                   dataset_hash: str | None = None, source_hashes: dict[str, str] | None = None,
                   profile_extra: dict | None = None, known_languages: pd.Series | None = None,
                   test_hash: str | None = None) -> dict:
    output_dir = Path(output_dir)
    errors = validate_data(train, test, expected_classes=expected_classes)
    if errors:
        raise ValueError("Invalid dataset:\n- " + "\n- ".join(errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    train = train.sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(train.model.unique().tolist())
    assignments = make_splits(train, seed, validation_fraction, candidates)
    if dataset_hash is None:
        dataset_hash = hashlib.sha256(train.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    lengths = train.text.astype(str).str.len()
    if known_languages is not None:
        languages = train.id.astype(str).map(known_languages).fillna("unknown").astype(str)
        language_status = "langdetect deterministic English-only curation" if dataset_version != "v0" else "provided"
    else:
        try:
            from langdetect import DetectorFactory, detect
        except ImportError as exc:
            raise RuntimeError("langdetect is required to profile this dataset") from exc
        DetectorFactory.seed = seed
        def detect_safely(text):
            try: return detect(text) if str(text).strip() else "unknown"
            except Exception: return "unknown"
        languages = train.text.astype(str).map(detect_safely)
        language_status = "langdetect"
    language_lengths = train.assign(_language=languages.to_numpy(), _length=lengths.to_numpy()).groupby("_language")["_length"].agg(["count", "median", "min", "max"])
    group_sizes = train.groupby("prompt_id", dropna=False).size()
    split_payload = assignments[["id", "split"]].to_csv(index=False, lineterminator="\n").encode()
    train_rows = assignments.split.eq("train").sum()
    validation_rows = assignments.split.eq("validation").sum()
    profile = {
        "dataset_version": dataset_version, "dataset_sha256": dataset_hash, "test_sha256": test_hash, "rows": len(train),
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
    if source_hashes:
        profile["source_hashes"] = source_hashes
    if profile_extra:
        profile.update(profile_extra)
    dataset_filename, profile_filename = _dataset_artifact_names(dataset_version)
    existing_profile = output_dir / profile_filename
    if existing_profile.exists():
        prior = json.loads(existing_profile.read_text())
        if prior.get("dataset_sha256") != profile["dataset_sha256"]:
            raise FileExistsError(f"{dataset_version} artifacts already exist for different training data; choose a new output directory/version")
        if prior.get("test_sha256") != profile["test_sha256"]:
            raise FileExistsError(f"{dataset_version} artifacts already exist for different test data; choose a new output directory/version")
        if prior.get("split_sha256") != profile["split_sha256"]:
            raise FileExistsError(f"{dataset_version} fold assignments already exist and differ; refusing to replace frozen splits")
    train.to_parquet(output_dir / dataset_filename, index=False)
    assignments.to_parquet(output_dir / f"splits_{dataset_version}.parquet", index=False)
    if test is not None:
        test.to_parquet(output_dir / f"test_{dataset_version}.parquet", index=False)
    (output_dir / profile_filename).write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def prepare(train_path: str | Path, output_dir: str | Path, test_path: str | Path | None = None, *, seed: int = 2026,
            validation_fraction: float = .2, candidates: int = 1000, expected_classes: int | None = 18,
            dataset_version: str = "v0") -> dict:
    train_path = Path(train_path)
    train = pd.read_csv(train_path, dtype={"id": "string", "prompt_id": "string", "model": "string"})
    test = pd.read_csv(test_path, dtype={"id": "string"}) if test_path else None
    return _prepare_frame(train, output_dir, dataset_version=dataset_version, test=test, seed=seed,
                          validation_fraction=validation_fraction, candidates=candidates,
                          expected_classes=expected_classes, dataset_hash=_hash_file(train_path),
                          test_hash=_hash_file(Path(test_path)) if test_path else None)


def _normal_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _language(text: str) -> str:
    from langdetect import detect
    try:
        return detect(text) if str(text).strip() else "unknown"
    except Exception:
        return "unknown"


def _proportional_source_sample(frame: pd.DataFrame, quota: int, *, seed: int) -> pd.DataFrame:
    """Allocate a fixed class quota proportionally across source datasets."""
    if quota >= len(frame):
        return frame.copy()
    counts = frame.source_dataset.value_counts().sort_index()
    exact = counts / counts.sum() * quota
    allocation = exact.apply(math.floor).astype(int)
    remaining = quota - int(allocation.sum())
    remainders = (exact - allocation).sort_values(ascending=False, kind="stable")
    for source in remainders.index:
        if remaining == 0:
            break
        if allocation[source] < counts[source]:
            allocation[source] += 1
            remaining -= 1
    selected = []
    for source, n in allocation.items():
        subset = frame[frame.source_dataset.eq(source)].sort_values("id", kind="stable")
        if n:
            source_seed = (seed + zlib.crc32(str(source).encode("utf-8"))) % (2**32 - 1)
            selected.append(subset.sample(n=int(n), random_state=source_seed))
    return pd.concat(selected, ignore_index=True) if selected else frame.iloc[:0].copy()


def curate_openpool(source_path: str | Path, output_dir: str | Path, *, dataset_version: str,
                    official_reference_path: str | Path | None = None, seed: int = 2026,
                    expected_classes: int = 18, rows_per_class: int | None = 2711,
                    validation_dataset: str = "v0", selection_method: str = "source_stratified",
                    base_dataset_path: str | Path | None = None,
                    additional_rows_by_class: dict[str, int] | None = None) -> dict:
    """Curate Open Pool V11; rows_per_class=None keeps every eligible row."""
    if selection_method not in {"source_stratified", "shortest_response"}:
        raise ValueError("selection_method must be source_stratified or shortest_response")
    source_path = Path(source_path)
    source = pd.read_parquet(source_path)
    output_dir = Path(output_dir)
    dataset_filename, profile_filename = _dataset_artifact_names(dataset_version)
    source_hashes = {str(source_path): _hash_file(source_path)}
    base_hash = _hash_file(Path(base_dataset_path)) if base_dataset_path else None
    if base_dataset_path:
        source_hashes[str(base_dataset_path)] = base_hash
    official_hash = _hash_file(Path(official_reference_path)) if official_reference_path else None
    if official_reference_path:
        source_hashes[str(official_reference_path)] = official_hash
    existing_data_path = output_dir / dataset_filename
    existing_profile_path = output_dir / profile_filename
    audit_filename = f"openpool_{dataset_version}_removals.csv"
    existing_audit_path = output_dir / audit_filename
    if existing_data_path.exists() and existing_profile_path.exists() and existing_audit_path.exists():
        prior = json.loads(existing_profile_path.read_text())
        v0_profile_path = output_dir / "dataset_profile.json"
        if v0_profile_path.exists():
            v0_profile = json.loads(v0_profile_path.read_text())
            profile_is_training_only = (
                prior.get("curation_method") in {
                    "openpool_v11_hard_defect_source_stratified_sample",
                    "openpool_v11_hard_defect_all_eligible_rows",
                    "openpool_v11_hard_defect_baseline_plus_class_additions",
                }
                and "split_sha256" not in prior
                and prior.get("validation_dataset_version") == "v0"
                and prior.get("validation_split_sha256") == v0_profile.get("split_sha256")
                and prior.get("source_hashes") == source_hashes
                and prior.get("official_reference_sha256") == official_hash
                and prior.get("target_rows_per_class") == (None if base_dataset_path else rows_per_class)
                and prior.get("selection_method", "source_stratified") == selection_method
                and prior.get("class_caps") == {}
                and prior.get("base_dataset_sha256") == base_hash
                and prior.get("additional_rows_by_class", {}) == (additional_rows_by_class or {})
                and prior.get("seed") == seed
            )
            if profile_is_training_only:
                cached = pd.read_parquet(existing_data_path)
                cached_hash = hashlib.sha256(cached.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
                if cached_hash != prior.get("dataset_sha256"):
                    raise ValueError("existing Open Pool training artifact does not match its profile hash")
                if _hash_file(existing_audit_path) != prior.get("removal_manifest_sha256"):
                    raise ValueError("existing Open Pool removal audit does not match its profile hash")
                return prior
    required = {"id", "text", "model", "prompt_canon", "source_dataset", "source_file", "source_license"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Open Pool source is missing required columns: {sorted(missing)}")
    initial_rows = len(source)
    source = source.reset_index(drop=True)
    source["_source_row_number"] = np.arange(len(source), dtype=np.int64)
    filter_counts = {"input_rows": initial_rows}
    removed_chunks = []

    def reject(frame: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            removed = frame.loc[mask].copy()
            removed["removal_reason"] = reason
            removed_chunks.append(removed)
        return frame.loc[~mask].copy()

    valid = source.copy()
    valid["id"] = valid.id.astype("string")
    valid["model"] = valid.model.astype("string")
    valid["text"] = valid.text.astype("string")
    filter_counts["missing_id"] = int(valid.id.isna().sum())
    filter_counts["missing_or_blank_text"] = int(valid.text.isna().sum() +
        (valid.text.notna() & valid.text.str.strip().eq("")).sum())
    filter_counts["missing_model"] = int(valid.model.isna().sum())
    filter_counts["missing_source_metadata"] = int(valid[["source_dataset", "source_file", "source_license"]].isna().any(axis=1).sum())
    required_valid = (valid.id.notna() & valid.model.notna() & valid.text.notna()
                      & valid.text.str.strip().ne("")
                      & valid[["source_dataset", "source_file", "source_license"]].notna().all(axis=1))
    valid = reject(valid, ~required_valid, "missing_required_text_label_or_provenance")
    filter_counts["missing_prompt_group"] = int(valid.prompt_canon.fillna("").str.strip().eq("").sum())
    valid = reject(valid, valid.prompt_canon.fillna("").str.strip().eq(""), "missing_canonical_prompt")
    filter_counts["rows_without_prompt_group"] = int(filter_counts["missing_prompt_group"])

    allowed_classes = set(source.model.dropna().astype(str).unique())
    if len(allowed_classes) != expected_classes:
        raise ValueError(f"expected {expected_classes} source labels before curation; found {len(allowed_classes)}")
    unknown_class = ~valid.model.astype(str).isin(allowed_classes)
    filter_counts["unknown_class"] = int(unknown_class.sum())

    g37 = valid.model.astype(str).eq("gemini-3.7-flash")
    allowed_g37 = valid.source_dataset.astype(str).isin(OPENPOOL_VERTEX_G37_SOURCES)
    filter_counts["gemini37_non_vertex_source"] = int((g37 & ~allowed_g37).sum())
    valid = reject(valid, g37 & ~allowed_g37, "gemini37_outside_allowed_vertex_sources")

    valid["_normalized_text"] = valid.text.map(_normal_text)
    filter_counts["empty_normalized_text"] = int(valid._normalized_text.eq("").sum())
    valid = reject(valid, valid._normalized_text.eq(""), "empty_normalized_text")
    if official_reference_path:
        official_path = Path(official_reference_path)
        official = pd.read_csv(official_path, usecols=["text"])
        official_texts = set(official.text.dropna().astype(str).map(_normal_text))
        exact_official = valid._normalized_text.isin(official_texts)
        filter_counts["exact_official_train_text"] = int(exact_official.sum())
        valid = reject(valid, exact_official, "exact_match_to_official_training_text")
        official_hash = _hash_file(official_path)
    else:
        official_hash = None
        filter_counts["exact_official_train_text"] = 0

    conflicting = valid.groupby("_normalized_text").model.nunique()
    conflicting_texts = set(conflicting[conflicting.gt(1)].index)
    filter_counts["conflicting_label_text_groups"] = len(conflicting_texts)
    valid = reject(valid, valid._normalized_text.isin(conflicting_texts), "same_text_has_conflicting_labels")
    duplicate_mask = valid.duplicated("_normalized_text", keep="first")
    filter_counts["duplicate_text_rows"] = int(duplicate_mask.sum())
    valid = reject(valid, duplicate_mask, "duplicate_normalized_response")

    try:
        from langdetect import DetectorFactory
    except ImportError as exc:
        raise RuntimeError("langdetect is required for English-only Open Pool curation") from exc
    DetectorFactory.seed = seed
    valid["language"] = valid.text.map(_language)
    filter_counts["non_english_or_undetermined"] = int(valid.language.ne("en").sum())
    valid = reject(valid, valid.language.ne("en"), "non_english_or_language_undetermined")

    if rows_per_class is not None and rows_per_class < 1:
        raise ValueError("rows_per_class must be a positive integer or None for all eligible rows")
    additional_rows_by_class = {str(k): int(v) for k, v in (additional_rows_by_class or {}).items()}
    if any(v < 1 for v in additional_rows_by_class.values()):
        raise ValueError("additional_rows_by_class values must be positive integers")
    if not set(additional_rows_by_class).issubset(allowed_classes):
        raise ValueError(f"additional rows requested for unknown classes: {sorted(set(additional_rows_by_class)-allowed_classes)}")
    quota_shortfalls = {}
    selected_frames = []
    base_ids: set[str] = set()
    if base_dataset_path:
        base = pd.read_parquet(base_dataset_path)
        if not {"id", "model"}.issubset(base.columns):
            raise ValueError("base dataset must contain id and model columns")
        if base.id.astype(str).duplicated().any():
            raise ValueError("base dataset contains duplicate IDs")
        if not set(base.model.astype(str)).issubset(allowed_classes):
            raise ValueError("base dataset contains labels outside the Open Pool class set")
        base_ids = set(base.id.astype(str))
        eligible_ids = set(valid.id.astype(str))
        missing_base = base_ids - eligible_ids
        if missing_base:
            raise ValueError(f"base dataset has {len(missing_base)} rows that no longer pass current hard quality filters")
        baseline = valid[valid.id.astype(str).isin(base_ids)].copy()
        if len(baseline) != len(base):
            raise ValueError("base dataset rows do not map one-to-one to eligible Open Pool rows")
        selected_frames.append(baseline)
    elif additional_rows_by_class:
        raise ValueError("additional_rows_by_class requires base_dataset_path")

    for label in sorted(allowed_classes):
        candidates_for_label = valid[valid.model.astype(str).eq(label)].copy()
        if base_dataset_path:
            candidates_for_label = candidates_for_label[~candidates_for_label.id.astype(str).isin(base_ids)]
            quota = additional_rows_by_class.get(label, 0)
            if quota == 0:
                reject(candidates_for_label, pd.Series(True, index=candidates_for_label.index),
                       "not_selected_outside_requested_class_additions")
                continue
        elif rows_per_class is None:
            selected_frames.append(candidates_for_label.copy())
            continue
        else:
            quota = rows_per_class
        if len(candidates_for_label) < quota:
            quota_shortfalls[label] = {"requested": quota, "available": int(len(candidates_for_label))}
            continue
        if selection_method == "shortest_response":
            # Prefer the shortest text by character count; word count and ID make
            # ties deterministic. Length is computed from text, not source metadata.
            candidates_for_label["_response_length_chars"] = candidates_for_label.text.str.len()
            candidates_for_label["_response_length_words"] = candidates_for_label.text.str.split().str.len()
            sampled = candidates_for_label.sort_values(
                ["_response_length_chars", "_response_length_words", "id"], kind="stable"
            ).head(quota)
        else:
            sampled = _proportional_source_sample(candidates_for_label, quota, seed=seed)
        selected_frames.append(sampled)
        selected_keys = set(sampled._source_row_number.astype(int))
        reject(candidates_for_label, ~candidates_for_label._source_row_number.isin(selected_keys),
               "quota_shortest_response_downsample" if selection_method == "shortest_response"
               else "quota_source_stratified_downsample")
    if quota_shortfalls:
        raise ValueError(f"Open Pool curation cannot meet requested class quotas after cleaning: {quota_shortfalls}")
    selected = pd.concat(selected_frames, ignore_index=True).sort_values("id", kind="stable").reset_index(drop=True)
    selected["prompt_id"] = selected.prompt_canon.astype("string")
    classes_after = set(selected.model.astype(str).unique())
    if classes_after != allowed_classes:
        raise ValueError(f"curated dataset label mismatch; missing={sorted(allowed_classes-classes_after)}")
    if selected.id.duplicated().any():
        raise ValueError("curated Open Pool IDs are not unique")
    if selected._normalized_text.duplicated().any():
        raise ValueError("normalized duplicate responses remain after curation")
    counts = selected.model.astype(str).value_counts().sort_index()
    expected_counts = ({str(k): int(v) for k, v in base.model.astype(str).value_counts().items()}
                       if base_dataset_path else {})
    if base_dataset_path:
        for label, additional in additional_rows_by_class.items():
            expected_counts[label] = expected_counts.get(label, 0) + additional
        if any(int(counts.get(label, 0)) != expected_counts.get(label, 0) for label in expected_counts):
            raise RuntimeError("curated class counts do not match baseline plus requested additions")
    elif rows_per_class is not None and any(count != rows_per_class for count in counts.values):
        raise RuntimeError("curated class quotas do not match configured target")

    public_columns = [c for c in source.columns if c in selected.columns]
    if "prompt_id" not in public_columns:
        public_columns.append("prompt_id")
    selected = selected[public_columns + (["language"] if "language" not in public_columns else [])].copy()
    # Content hash covers selected records and metadata in a stable row/column order.
    content_hash = hashlib.sha256(selected.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    if validation_dataset != "v0":
        raise ValueError("Open Pool validation currently reuses the frozen V0 validation partition")
    validation_profile_path = output_dir / "dataset_profile.json"
    validation_split_path = output_dir / "splits_v0.parquet"
    if not validation_profile_path.exists() or not validation_split_path.exists():
        raise FileNotFoundError("prepare V0 first; Open Pool experiments reuse its frozen validation split")
    validation_profile = json.loads(validation_profile_path.read_text())
    validation_assignments = pd.read_parquet(validation_split_path)
    validation_ids = validation_assignments.loc[validation_assignments.split.eq("validation"), "id"].astype(str)
    if len(validation_ids) != validation_profile.get("validation_rows"):
        raise ValueError("V0 validation assignments do not match the frozen V0 profile")
    length_stats = (selected.assign(_length_chars=selected.text.str.len())
                    .groupby("model")["_length_chars"].agg(["min", "median", "max"]))
    profile = {
        "dataset_version": dataset_version, "dataset_sha256": content_hash,
        "rows": len(selected), "classes": sorted(selected.model.astype(str).unique()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "prompt_groups": int(selected.prompt_id.nunique()),
        "seed": seed, "target_rows_per_class": None if base_dataset_path else rows_per_class,
        "target_rows_by_class": {str(k): int(v) for k, v in counts.items()},
        "base_dataset": str(base_dataset_path) if base_dataset_path else None,
        "base_dataset_sha256": base_hash,
        "additional_rows_by_class": additional_rows_by_class,
        "selection_method": selection_method,
        "class_caps": {},
        "curation_method": ("openpool_v11_hard_defect_baseline_plus_class_additions" if base_dataset_path else
                            ("openpool_v11_hard_defect_all_eligible_rows" if rows_per_class is None
                            else f"openpool_v11_hard_defect_{selection_method}_sample")),
        "sampling_method": (f"preserve every baseline row; add {selection_method} rows only for configured classes" if base_dataset_path else
            ("none; retain all eligible rows" if rows_per_class is None else
            ("fixed per-class quota; shortest response by character count, then word count and ID"
             if selection_method == "shortest_response" else
             "fixed per-class quota sampled proportionally by source"))),
        "selected_response_length_chars_by_class": {
            str(label): {"min": int(values.min()), "median": float(values.median()), "max": int(values.max())}
            for label, values in length_stats.iterrows()
        },
        "source_dataset": str(source_path), "source_hashes": source_hashes,
        "filter_counts": filter_counts,
        "selected_rows_by_class": {str(k): int(v) for k, v in counts.items()},
        "selected_rows_by_source": {str(k): int(v) for k, v in selected.source_dataset.value_counts().sort_index().items()},
        "selected_rows_by_language": {str(k): int(v) for k, v in selected.language.value_counts().sort_index().items()},
        "gemini37_allowed_sources": sorted(OPENPOOL_VERTEX_G37_SOURCES),
        "gemini37_origin_hint_rows": int(selected.loc[selected.model.astype(str).eq("gemini-3.7-flash"), "origin_hint"].fillna("").ne("").sum()) if "origin_hint" in selected else 0,
        "official_reference_sha256": official_hash,
        "validation_dataset_version": validation_dataset,
        "validation_split_sha256": validation_profile["split_sha256"],
        "validation_rows": int(len(validation_ids)),
        "validation_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
    }
    audit = pd.concat(removed_chunks, ignore_index=True) if removed_chunks else source.iloc[:0].copy()
    audit_columns = [c for c in ("_source_row_number", "id", "model", "source_dataset", "source_file", "source_license", "removal_reason") if c in audit]
    audit = audit[audit_columns].rename(columns={"_source_row_number": "source_row_number"})
    audit_summary = audit.removal_reason.value_counts().sort_index().astype(int).to_dict() if len(audit) else {}
    profile["removals_by_reason"] = audit_summary
    profile["removed_rows"] = int(len(audit))
    profile["removals_by_class"] = ({str(k): int(v) for k, v in audit.groupby("model").size().sort_index().items()}
                                    if len(audit) else {})
    profile["removals_by_source"] = ({str(k): int(v) for k, v in audit.groupby("source_dataset").size().sort_index().items()}
                                     if len(audit) else {})
    audit_bytes = audit.to_csv(index=False, lineterminator="\n").encode("utf-8")
    profile["removal_manifest_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    if existing_profile_path.exists():
        prior = json.loads(existing_profile_path.read_text())
        compatible_source = prior.get("source_hashes") == source_hashes
        compatible_validation = prior.get("validation_split_sha256") == profile["validation_split_sha256"]
        known_previous_curation = prior.get("curation_method") in {
            "openpool_v11_strict_provenance_response_sample",
            "openpool_v11_hard_defect_source_stratified_sample",
            "openpool_v11_hard_defect_all_eligible_rows",
            "openpool_v11_hard_defect_shortest_response_sample",
            "openpool_v11_hard_defect_baseline_plus_class_additions",
        }
        if not (compatible_source and compatible_validation and known_previous_curation):
            raise FileExistsError(f"{dataset_version} artifacts exist for a different source or validation split")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected.drop(columns="_normalized_text", errors="ignore")
    selected.to_parquet(output_dir / dataset_filename, index=False)
    audit.to_csv(output_dir / audit_filename, index=False, lineterminator="\n")
    (output_dir / profile_filename).write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def assemble_openpool_mix(input_dir: str | Path, output_dir: str | Path, *,
                          dataset_version: str, openpool_version: str = "v1",
                          expected_classes: int = 18) -> dict:
    """Persist V0 train + a curated Open Pool version with frozen V0 validation.

    This creates one combined parquet dataset and a split-assignment sidecar;
    it does not generate a new validation split. Every Open Pool row is train-only.
    """
    if dataset_version in {"v0", openpool_version}:
        raise ValueError("mixed dataset version must be distinct from its source versions")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    required = {
        "v0_data": input_dir / "dataset_v0.parquet",
        "v0_splits": input_dir / "splits_v0.parquet",
        "v0_profile": input_dir / "dataset_profile.json",
        f"{openpool_version}_data": input_dir / f"dataset_{openpool_version}.parquet",
        f"{openpool_version}_profile": input_dir / f"dataset_profile_{openpool_version}.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepare V0 and curate {openpool_version} before assembling {dataset_version}: " + ", ".join(missing))

    v0 = pd.read_parquet(required["v0_data"])
    v0_splits = pd.read_parquet(required["v0_splits"])
    openpool = pd.read_parquet(required[f"{openpool_version}_data"])
    v0_profile = json.loads(required["v0_profile"].read_text())
    openpool_profile = json.loads(required[f"{openpool_version}_profile"].read_text())
    for label, frame in (("V0", v0), (openpool_version, openpool)):
        errors = validate_data(frame, expected_classes=expected_classes)
        if errors:
            raise ValueError(f"Invalid {label} input dataset:\n- " + "\n- ".join(errors))

    if not {"id", "split"}.issubset(v0_splits.columns):
        raise ValueError("V0 split assignments must contain id and split columns")
    if v0_splits.id.isna().any() or v0_splits.id.duplicated().any():
        raise ValueError("V0 split assignment IDs must be present and unique")
    if set(v0.id.astype(str)) != set(v0_splits.id.astype(str)):
        raise ValueError("V0 split assignment IDs do not match V0 dataset IDs")
    if set(v0_splits.split.astype(str)) != {"train", "validation"}:
        raise ValueError("V0 split assignments must contain train and validation partitions")
    if "model" in v0_splits:
        expected_labels = v0.set_index("id").model.astype(str).sort_index()
        assigned_labels = v0_splits.set_index("id").model.astype(str).sort_index()
        if not expected_labels.equals(assigned_labels):
            raise ValueError("V0 split assignment labels do not match V0 dataset labels")
    if "prompt_id" in v0_splits:
        expected_prompts = v0.set_index("id").prompt_id.astype(str).sort_index()
        assigned_prompts = v0_splits.set_index("id").prompt_id.astype(str).sort_index()
        if not expected_prompts.equals(assigned_prompts):
            raise ValueError("V0 split assignment prompts do not match V0 dataset prompts")

    v0 = v0.merge(v0_splits[["id", "split"]], on="id", how="left", validate="one_to_one")
    v0["training_data_version"] = "v0"
    openpool = openpool.copy()
    openpool["split"] = "train"
    openpool["training_data_version"] = openpool_version
    if set(v0.model.astype(str)) != set(openpool.model.astype(str)):
        raise ValueError(f"V0 and {openpool_version} label sets differ")
    if set(v0.id.astype(str)) & set(openpool.id.astype(str)):
        raise ValueError(f"V0 and {openpool_version} row IDs overlap")

    rows = pd.concat([v0, openpool], ignore_index=True, sort=False)
    rows = rows.sort_values("id", kind="stable").reset_index(drop=True)
    for column in ("id", "prompt_id", "text", "model", "split", "training_data_version"):
        rows[column] = rows[column].astype("string")
    errors = validate_data(rows, expected_classes=expected_classes)
    if errors:
        raise ValueError(f"Invalid combined {dataset_version} dataset:\n- " + "\n- ".join(errors))

    assignments = rows[["id", "prompt_id", "model", "split"]].copy()
    train = rows.loc[rows.split.eq("train")]
    validation = rows.loc[rows.split.eq("validation")]
    train_groups = set(train.prompt_id.astype(str))
    validation_groups = set(validation.prompt_id.astype(str))
    if train_groups & validation_groups:
        raise ValueError("V1.5 train rows overlap the frozen V0 validation prompt groups")
    if not openpool.id.astype(str).isin(set(train.id.astype(str))).all():
        raise RuntimeError(f"some {openpool_version} rows were not assigned to the {dataset_version} training partition")
    if set(validation.id.astype(str)) != set(
            v0_splits.loc[v0_splits.split.eq("validation"), "id"].astype(str)):
        raise RuntimeError(f"{dataset_version} validation IDs differ from the frozen V0 validation IDs")

    split_payload = assignments[["id", "split"]].to_csv(index=False, lineterminator="\n").encode()
    dataset_payload = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    dataset_hash = hashlib.sha256(dataset_payload).hexdigest()
    split_hash = hashlib.sha256(split_payload).hexdigest()
    source_hashes = {name: _hash_file(path) for name, path in required.items()}
    train_counts = {str(k): int(v) for k, v in train.model.value_counts().sort_index().items()}
    validation_counts = {str(k): int(v) for k, v in validation.model.value_counts().sort_index().items()}
    profile = {
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_hash,
        "split_sha256": split_hash,
        "rows": int(len(rows)),
        "classes": sorted(rows.model.astype(str).unique().tolist()),
        "class_counts": {str(k): int(v) for k, v in rows.model.value_counts().sort_index().items()},
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "prompt_groups": int(rows.prompt_id.nunique()),
        "train_prompt_groups": int(train.prompt_id.nunique()),
        "validation_prompt_groups": int(validation.prompt_id.nunique()),
        "prompt_group_leakage": 0,
        "training_data_version_rows": {str(k): int(v) for k, v in rows.training_data_version.value_counts().sort_index().items()},
        "training_component_rows": {
            "v0_train": int(v0.split.eq("train").sum()),
            "v0_validation": int(v0.split.eq("validation").sum()),
            f"{openpool_version}_train": int(len(openpool)),
        },
        "validation_dataset_version": "v0",
        "validation_split_sha256": v0_profile["split_sha256"],
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(sorted(validation.id.astype(str))).encode()).hexdigest(),
        "split_method": f"preserved V0 assignments; all {openpool_version} rows assigned to train",
        "source_hashes": source_hashes,
    }

    dataset_path = output_dir / f"dataset_{dataset_version}.parquet"
    split_path = output_dir / f"splits_{dataset_version}.parquet"
    profile_path = output_dir / f"dataset_profile_{dataset_version}.json"
    existing = [path.exists() for path in (dataset_path, split_path, profile_path)]
    if any(existing):
        if not all(existing):
            raise FileExistsError(f"partial {dataset_version} artifacts exist; refusing to overwrite them")
        previous = json.loads(profile_path.read_text())
        if (previous.get("dataset_sha256") != dataset_hash
                or previous.get("split_sha256") != split_hash
                or previous.get("source_hashes") != source_hashes):
            raise FileExistsError(f"{dataset_version} artifacts exist for different inputs; choose a new version")
        profile["artifacts"] = {"dataset": str(dataset_path), "splits": str(split_path), "profile": str(profile_path)}
        return profile

    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(dataset_path, index=False)
    assignments.to_parquet(split_path, index=False)
    profile["artifacts"] = {"dataset": str(dataset_path), "splits": str(split_path), "profile": str(profile_path)}
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def assemble_v1_5(input_dir: str | Path, output_dir: str | Path, *,
                  dataset_version: str = "v1.5", expected_classes: int = 18) -> dict:
    """Backward-compatible V0-train + V1 assembly entry point."""
    return assemble_openpool_mix(input_dir, output_dir, dataset_version=dataset_version,
                                 openpool_version="v1", expected_classes=expected_classes)
