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


def build_full_v0_external_scope(external: pd.DataFrame, v0: pd.DataFrame, *,
                                 expected_classes: int = 18,
                                 expected_v0_rows: int = 2699) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use every external row for training and all official V0 rows for validation."""
    external = external.copy().reset_index(drop=True)
    v0 = v0.copy().reset_index(drop=True)
    for label, frame in (("external", external), ("V0", v0)):
        errors = validate_data(frame, expected_classes=expected_classes)
        if errors:
            raise ValueError(f"Invalid {label} dataset:\n- " + "\n- ".join(errors))
    if expected_v0_rows is not None and len(v0) != int(expected_v0_rows):
        raise ValueError(f"full V0 scope requires {expected_v0_rows} official rows; found {len(v0)}")
    if set(external.model.astype(str)) != set(v0.model.astype(str)):
        raise ValueError("external training labels differ from the official V0 labels")

    external_ids = set(external.id.astype(str))
    v0_ids = set(v0.id.astype(str))
    id_overlap = external_ids & v0_ids
    external_text = external.text.astype(str).map(_normal_text)
    v0_text = v0.text.astype(str).map(_normal_text)
    response_overlap = set(external_text) & set(v0_text)
    external_prompt_ids = set(external.prompt_id.astype(str))
    v0_prompt_ids = set(v0.prompt_id.astype(str))
    prompt_id_overlap = external_prompt_ids & v0_prompt_ids
    external_prompt_text = (set(external.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}
                            if "prompt_canon" in external else set())
    v0_prompt_text = (set(v0.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}
                      if "prompt_canon" in v0 else set())
    prompt_text_overlap = external_prompt_text & v0_prompt_text
    overlap_counts = {
        "id": len(id_overlap), "normalized_response_text": len(response_overlap),
        "prompt_id": len(prompt_id_overlap), "normalized_prompt_text": len(prompt_text_overlap),
    }
    if any(overlap_counts.values()):
        raise ValueError("external data overlaps full V0; refusing leakage: " +
                         json.dumps(overlap_counts, sort_keys=True))

    external = external.sort_values("id", kind="stable").reset_index(drop=True)
    external_scoped = external.drop(columns=["split"], errors="ignore").copy()
    external_scoped["split"] = "train"
    external_scoped["validation_role"] = "training"
    v0 = v0.sort_values("id", kind="stable").reset_index(drop=True)
    v0_scoped = v0.drop(columns=["split"], errors="ignore").copy()
    v0_scoped["split"] = "validation"
    v0_scoped["validation_role"] = "validation_and_checkpoint_selection"

    rows = pd.concat([external_scoped, v0_scoped], ignore_index=True, sort=False)
    if rows.id.astype(str).duplicated().any():
        raise ValueError("scoped external and V0 row IDs are not unique")
    train_rows = external_scoped
    validation_rows = v0_scoped
    assignments = pd.concat([
        external_scoped[["id", "prompt_id", "model", "split", "validation_role"]],
        v0_scoped[["id", "prompt_id", "model", "split", "validation_role"]],
    ], ignore_index=True, sort=False).sort_values("id", kind="stable").reset_index(drop=True)
    split_payload = assignments.to_csv(index=False, lineterminator="\n").encode("utf-8")

    def ordered_hash(values):
        return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()

    def normalized_content_hash(frame):
        pairs = sorted(f"{row.id}\t{_normal_text(row.text)}"
                       for row in frame.sort_values("id", kind="stable").itertuples())
        return hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()

    def prompt_content_hash(frame):
        values = (frame.prompt_canon.fillna("").astype(str).map(_normal_text).tolist()
                  if "prompt_canon" in frame else frame.prompt_id.astype(str).tolist())
        return ordered_hash(value for value in values if value)

    v0_by_id = v0.assign(_id=v0.id.astype(str)).sort_values("_id", kind="stable")
    normalized_v0_text = [f"{row.id}\t{_normal_text(row.text)}" for row in v0_by_id.itertuples()]
    profile = {
        "validation_scope": "full_v0",
        "selection_validation_scope": "full_v0",
        "validation_dataset_version": "v0", "validation_rows": int(len(v0)),
        "validation_prompt_groups": int(v0.prompt_id.nunique()),
        "validation_ids_sha256": ordered_hash(v0.id.astype(str)),
        "validation_normalized_text_sha256": hashlib.sha256("\n".join(normalized_v0_text).encode("utf-8")).hexdigest(),
        "validation_prompt_ids_sha256": ordered_hash(v0.prompt_id.astype(str).unique()),
        "validation_prompt_content_sha256": prompt_content_hash(v0),
        "training_dataset_rows": int(len(train_rows)),
        "training_dataset_ids_sha256": ordered_hash(train_rows.id.astype(str)),
        "training_dataset_prompt_groups": int(train_rows.prompt_id.nunique()),
        "training_dataset_normalized_text_sha256": normalized_content_hash(train_rows),
        "training_dataset_prompt_content_sha256": prompt_content_hash(train_rows),
        "training_rows": int(len(train_rows)),
        "training_ids_sha256": ordered_hash(train_rows.id.astype(str)),
        "training_prompt_groups": int(train_rows.prompt_id.nunique()),
        "training_prompt_ids_sha256": ordered_hash(train_rows.prompt_id.astype(str).unique()),
        "split_sha256": hashlib.sha256(split_payload).hexdigest(),
        "split_method": "all external rows for training; all official V0 rows for validation",
        "isolation_overlap_counts": overlap_counts,
        "training_validation_overlap_counts": overlap_counts,
        "external_v0_overlap": overlap_counts,
        "v0_original_split_sha256": None,
    }
    return rows.sort_values("id", kind="stable").reset_index(drop=True), assignments, profile


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


def _external_content_text(value) -> str:
    """Flatten the text-bearing shapes used by the downloaded conversation rows."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _stable_prompt_id(prompt: str, fallback: str) -> str:
    normalized = _normal_text(prompt)
    key = normalized if normalized else fallback
    return "external_prompt_" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_v1_3_external_dataset(glm_files: list[str | Path],
                                glm_provenance_path: str | Path, gpt_traces_path: str | Path,
                                output_dir: str | Path, *, dataset_version: str = "v1.3",
                                validation_dataset: str = "v0",
                                max_response_chars: int | None = None) -> dict:
    """Build V1.3 from only the downloaded GLM-5.3-Flash and GPT-5.6 Sol/Luna rows.

    The source files are normalized into the training row schema, with row-level
    provenance retained. GLM's label is dataset-attributed; GPT labels are read
    from each trace. Empty assistant responses remain in the assembled input so
    the clean pass can record and remove them in its audit.
    """
    if validation_dataset != "v0":
        raise ValueError("V1.3 currently reuses the frozen V0 validation set")
    if max_response_chars is not None and max_response_chars < 1:
        raise ValueError("max_response_chars must be positive when set")
    output_dir = Path(output_dir)
    glm_files = [Path(path) for path in glm_files]
    glm_provenance_path, gpt_traces_path = Path(glm_provenance_path), Path(gpt_traces_path)
    required_files = [*glm_files, glm_provenance_path, gpt_traces_path]
    if not all(path.is_file() for path in required_files):
        missing = [str(path) for path in required_files if not path.is_file()]
        raise FileNotFoundError("missing V1.3 source files: " + ", ".join(missing))

    provenance = pd.read_parquet(glm_provenance_path) if glm_provenance_path.is_file() else pd.DataFrame()
    if not provenance.empty:
        if "id" not in provenance:
            raise ValueError("GLM provenance table has no id column")
        provenance = provenance.drop_duplicates("id", keep="first").set_index("id")
    glm_rows = []
    glm_input_rows = 0
    for source_path in glm_files:
        frame = pd.read_parquet(source_path)
        needed = {"id", "prompt", "response"}
        if needed - set(frame.columns):
            raise ValueError(f"{source_path} is missing columns: {sorted(needed - set(frame.columns))}")
        glm_input_rows += len(frame)
        if not provenance.empty:
            joined = provenance.reindex(frame.id.astype(str))
            for column in ("finish", "truncated", "cap_hit", "round", "source", "mode", "tokens"):
                if column in joined:
                    frame[f"_prov_{column}"] = joined[column].to_numpy()
        frame = frame.reset_index(drop=True)
        source_name = source_path.name
        canonical_prompts = frame.prompt.fillna("").astype(str)
        raw_ids = frame.id.fillna("").astype(str)
        records = pd.DataFrame({
            "id": ["v13_" + hashlib.sha256(
                f"glm53_flash\0{source_name}\0{row_index}\0{raw_id}".encode("utf-8")).hexdigest()[:32]
                for row_index, raw_id in enumerate(raw_ids)],
            "text": frame.response.fillna("").astype(str),
            "model": "glm-5.3-flash",
            "source_dataset": "Zek-Takai/glm53-flash-harvest",
            "source_file": source_name,
            "source_license": "mit",
            "source_row_id": raw_ids,
            "source_corpus": "Zek-Takai/glm53-flash-harvest",
            "source_model_attribution": "dataset_level",
            "prompt_canon": canonical_prompts,
            "prompt_id": [_stable_prompt_id(prompt, f"glm53_flash:{source_name}:{raw_id}")
                          for prompt, raw_id in zip(canonical_prompts, raw_ids)],
            "language": frame.get("lang", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str),
            "provider": "zai",
            "model_version": "glm-5.3-flash",
            "register": frame.get("register"),
            "thinking": frame.get("thinking"),
            "source_finish_reason": frame.get("_prov_finish"),
            "source_truncated": frame.get("_prov_truncated"),
            "source_cap_hit": frame.get("_prov_cap_hit"),
            "source_round": frame.get("_prov_round"),
            "source_origin": frame.get("_prov_source"),
            "source_mode": frame.get("_prov_mode"),
            "source_tokens": frame.get("_prov_tokens", frame.get("tokens")),
        })
        glm_rows.append(records)
    glm = pd.concat(glm_rows, ignore_index=True, sort=False) if glm_rows else pd.DataFrame()

    gpt = pd.read_parquet(gpt_traces_path).reset_index(drop=True)
    if "row_json" not in gpt:
        raise ValueError("GPT Sol/Luna trace table is missing row_json")
    gpt_records, parse_errors = [], 0
    for row_index, row in gpt.iterrows():
        try:
            raw = json.loads(row.row_json) if isinstance(row.row_json, str) else row.row_json
            if not isinstance(raw, dict):
                raise ValueError("row_json is not an object")
            messages = raw.get("messages") or []
            users = [message.get("content") for message in messages
                     if isinstance(message, dict) and message.get("role") == "user"]
            assistants = [message.get("content") for message in messages
                          if isinstance(message, dict) and message.get("role") == "assistant"]
            raw_model = raw.get("teacher_model") or raw.get("model")
            model = str(raw_model or "").removeprefix("openai/")
            if model not in {"gpt-5.6-sol", "gpt-5.6-luna"}:
                raise ValueError(f"unrecognized GPT model label {model!r}")
            prompt = _external_content_text(users[-1]) if users else ""
            response = _external_content_text(assistants[-1]) if assistants else ""
            raw_id = str(row.get("row_hash") or row_index)
            gpt_records.append({
                "id": "v13_" + hashlib.sha256(
                    f"gpt56_sol_luna\0{row_index}\0{raw_id}".encode("utf-8")).hexdigest()[:32],
                "text": response, "model": model,
                "source_dataset": str(row.get("first_source_dataset") or
                                      "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces"),
                "source_file": gpt_traces_path.name, "source_license": "cc-by-4.0",
                "source_row_id": raw_id,
                "source_corpus": "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces",
                "source_model_attribution": "row_level_teacher_model_or_model",
                "prompt_canon": prompt,
                "prompt_id": _stable_prompt_id(prompt, f"gpt56_sol_luna:{raw_id}"),
                "language": "en", "provider": "openai", "model_version": model,
                "source_original_dataset": row.get("first_source_dataset"),
                "source_original_config": row.get("first_source_config"),
                "source_original_split": row.get("first_source_split"),
                "source_trace_id": raw.get("source_trajectory_id"),
                "source_category": raw.get("category"), "source_domain": raw.get("domain"),
                "source_task": raw.get("task"), "source_reasoning_effort": raw.get("reasoning_effort"),
            })
        except (ValueError, TypeError, json.JSONDecodeError):
            parse_errors += 1
            # Keep a structurally invalid row in the assembled artifact so that
            # its absence is represented by the subsequent clean audit.
            raw_id = str(row.get("row_hash") or row_index)
            gpt_records.append({
                "id": "v13_" + hashlib.sha256(
                    f"gpt56_sol_luna\0{row_index}\0{raw_id}".encode("utf-8")).hexdigest()[:32],
                "text": "", "model": None,
                "source_dataset": str(row.get("first_source_dataset") or
                                      "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces"),
                "source_file": gpt_traces_path.name, "source_license": "cc-by-4.0",
                "source_row_id": raw_id,
                "source_corpus": "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces",
                "source_model_attribution": "parse_error", "prompt_canon": "",
                "prompt_id": _stable_prompt_id("", f"gpt56_sol_luna:{raw_id}"),
            })
    gpt_normalized = pd.DataFrame(gpt_records)

    external = pd.concat([glm, gpt_normalized], ignore_index=True, sort=False)
    if set(external.model.dropna().astype(str).unique()) != {
            "glm-5.3-flash", "gpt-5.6-sol", "gpt-5.6-luna"}:
        raise ValueError("external source labels did not include the expected GLM Flash, GPT Sol, and GPT Luna classes")
    labels = set(external.model.dropna().astype(str).unique())
    rows_before_length_filter = len(external)
    length_filter_audit = pd.DataFrame()
    if max_response_chars is not None:
        response_lengths = external.text.fillna("").astype(str).str.len()
        too_long = response_lengths.gt(max_response_chars)
        audit_columns = [column for column in (
            "id", "model", "source_dataset", "source_file", "source_row_id", "prompt_id")
            if column in external]
        length_filter_audit = external.loc[too_long, audit_columns].copy()
        length_filter_audit["response_chars"] = response_lengths.loc[too_long].to_numpy()
        length_filter_audit["max_response_chars"] = max_response_chars
        external = external.loc[~too_long].reset_index(drop=True)

    validation_profile_path = output_dir / "dataset_profile.json"
    validation_split_path = output_dir / "splits_v0.parquet"
    validation_data_path = output_dir / "dataset_v0.parquet"
    if not all(path.is_file() for path in (validation_profile_path, validation_split_path, validation_data_path)):
        raise FileNotFoundError("prepare V0 first; V1.3 must use its frozen validation split")
    validation_profile = json.loads(validation_profile_path.read_text(encoding="utf-8"))
    validation_assignments = pd.read_parquet(validation_split_path)
    validation_ids = set(validation_assignments.loc[
        validation_assignments.split.eq("validation"), "id"].astype(str))
    v0 = pd.read_parquet(validation_data_path)
    v0_validation = v0.loc[v0.id.astype(str).isin(validation_ids)]
    if len(v0_validation) != int(validation_profile.get("validation_rows", -1)):
        raise ValueError("V0 validation assignments do not match the frozen profile")
    if "prompt_canon" in v0_validation:
        validation_prompts = set(v0_validation.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}
    else:
        validation_prompts = set()
    validation_prompt_ids = set(v0_validation.prompt_id.astype(str))
    external_prompt_norm = external.prompt_canon.fillna("").astype(str).map(_normal_text)
    overlap = (external.prompt_id.astype(str).isin(validation_prompt_ids)
               | external_prompt_norm.isin(validation_prompts))
    overlap_count = int(overlap.sum())
    external = external.loc[~overlap].reset_index(drop=True)

    external_ids = set(external.id.astype(str))
    if external.id.astype(str).duplicated().any():
        raise ValueError("generated V1.3 dataset has duplicate row IDs")
    combined = external

    output_path = output_dir / f"dataset_{dataset_version}.parquet"
    profile_path = output_dir / f"dataset_profile_{dataset_version}.json"
    length_audit_path = (output_dir / f"{dataset_version}_over_{max_response_chars}_chars_removals.csv"
                         if max_response_chars is not None else None)
    if output_path.exists() or profile_path.exists():
        if not (output_path.is_file() and profile_path.is_file()):
            raise FileExistsError(f"partial {dataset_version} artifacts exist; refusing to overwrite")
        old_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        old_data = pd.read_parquet(output_path)
        if (old_profile.get("dataset_sha256") == hashlib.sha256(
                old_data.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
                and old_profile.get("source_hashes") == {str(path): _hash_file(path) for path in required_files}
                and old_profile.get("max_response_chars") == max_response_chars
                and old_profile.get("validation_split_sha256") == validation_profile.get("split_sha256")):
            return old_profile
        raise FileExistsError(f"{dataset_version} artifacts exist for different source data; choose a new version")

    combined = combined.sort_values("id", kind="stable").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    if length_audit_path is not None:
        length_filter_audit.to_csv(length_audit_path, index=False, lineterminator="\n")
    source_hashes = {str(path): _hash_file(path) for path in required_files}
    dataset_hash = hashlib.sha256(combined.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    profile = {
        "dataset_version": dataset_version, "dataset_sha256": dataset_hash,
        "rows": int(len(combined)), "training_only": True,
        "classes": sorted(labels),
        "class_counts": {str(k): int(v) for k, v in combined.model.value_counts().sort_index().items()},
        "prompt_groups": int(combined.prompt_id.nunique()),
        "source_counts": {str(k): int(v) for k, v in combined.source_dataset.value_counts().sort_index().items()},
        "external_added_rows": int(len(external)),
        "external_added_rows_by_model": {str(k): int(v) for k, v in external.model.value_counts().sort_index().items()},
        "external_input_rows": {"glm53_flash": int(glm_input_rows), "gpt56_sol_luna": int(len(gpt))},
        "rows_before_response_length_filter": int(rows_before_length_filter),
        "max_response_chars": max_response_chars,
        "response_length_filter_removed_rows": int(len(length_filter_audit)),
        "response_length_filter_removed_by_model": ({str(k): int(v) for k, v in
            length_filter_audit.model.value_counts().sort_index().items()} if len(length_filter_audit) else {}),
        "response_length_filter_audit_path": str(length_audit_path) if length_audit_path else None,
        "gpt_trace_parse_errors": int(parse_errors),
        "validation_prompt_overlap_rows_removed": overlap_count,
        "source_hashes": source_hashes,
        "curation_method": "normalized_glm53_flash_and_gpt56_sol_luna_only",
        "validation_dataset_version": validation_dataset,
        "validation_split_sha256": validation_profile["split_sha256"],
        "validation_rows": int(len(v0_validation)),
        "validation_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
    }
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


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


def curate_v1_all_balanced_deduped(source_path: str | Path, output_dir: str | Path, *,
                                    dataset_version: str = "v1-all_balanced_deduped",
                                    validation_dataset: str = "v0", seed: int = 2026,
                                    near_duplicate_threshold: float = 0.90,
                                    char_ngram_size: int = 5, minhash_permutations: int = 64,
                                    minimum_template_chars: int = 100,
                                    template_min_repeats: int = 5,
                                    source_median_multiplier: float = 3.0) -> dict:
    """Curate V1-all by conservative quality, duplicate, template and diversity rules.

    No class quotas are used. Selection depends only on the supplied response,
    prompt, label and provenance columns. The artifact reuses frozen V0 validation.
    """
    from datasketch import MinHash, MinHashLSH
    import zlib

    if not 0 < near_duplicate_threshold < 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1)")
    if char_ngram_size < 2 or minhash_permutations < 16:
        raise ValueError("char_ngram_size must be >= 2 and minhash_permutations >= 16")
    if source_median_multiplier < 1:
        raise ValueError("source_median_multiplier must be >= 1")
    if validation_dataset != "v0":
        raise ValueError("this curation reuses the frozen V0 validation set")
    source_path, output_dir = Path(source_path), Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    data = pd.read_parquet(source_path).reset_index(drop=True)
    required = {"id", "text", "model", "source_dataset", "prompt_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"source dataset missing required columns: {sorted(missing)}")
    if data.model.nunique(dropna=True) != 18:
        raise ValueError(f"expected 18 model labels; found {data.model.nunique(dropna=True)}")
    original_classes = set(data.model.dropna().astype(str))
    data["_original_index"] = np.arange(len(data), dtype=np.int64)
    data["_text_norm"] = data.text.fillna("").astype(str).map(_normal_text)
    data["_prompt_key"] = data.get("prompt_canon", pd.Series(index=data.index, dtype=object)).fillna("").astype(str).str.strip()
    data.loc[data._prompt_key.eq(""), "_prompt_key"] = data.loc[data._prompt_key.eq(""), "prompt_id"].astype(str)
    removal_parts: list[pd.DataFrame] = []

    def remove(mask, reason):
        mask = pd.Series(mask, index=data.index).fillna(False).astype(bool) & data._keep
        if mask.any():
            chunk = data.loc[mask].copy()
            chunk["removal_reason"] = reason
            removal_parts.append(chunk)
            data.loc[mask, "_keep"] = False
        return int(mask.sum())

    data["_keep"] = True
    reasons: dict[str, int] = {}
    required_valid = (data.id.notna() & data.id.astype(str).str.strip().ne("") & data.model.notna()
                      & data.text.notna() & data._text_norm.str.len().gt(0)
                      & data.source_dataset.notna() & data.source_dataset.astype(str).str.strip().ne("")
                      & data.prompt_id.notna() & data.prompt_id.astype(str).str.strip().ne(""))
    reasons["missing_or_structurally_empty_required_fields"] = remove(~required_valid, "missing_or_structurally_empty_required_fields")
    duplicate_ids = data.id.astype(str).duplicated(keep="first")
    reasons["duplicate_row_id"] = remove(duplicate_ids, "duplicate_row_id")
    reasons["malformed_placeholder_or_extraction_artifact"] = remove(
        data._text_norm.str.contains(r"\[object object\]|<\|(?:im_start|im_end|assistant|user)\|>|\x00", regex=True),
        "malformed_placeholder_or_extraction_artifact")
    # Truncation is flagged only for unmistakable export markers; sentence fragments
    # and short but complete answers remain eligible.
    reasons["explicit_truncation_marker"] = remove(
        data._text_norm.str.contains(r"(?:\[truncated\]|\(truncated\)|\.\.\.\s*\[end\])\s*$", regex=True),
        "explicit_truncation_marker")
    # Prompt echoes require a long prompt repeated at the beginning of the answer
    # with little answer content remaining. A shared phrase alone is insufficient.
    prompt_norm = data.get("prompt_canon", pd.Series("", index=data.index)).fillna("").astype(str).map(_normal_text)
    echo = pd.Series([len(p) >= 80 and t.startswith(p) and len(t) - len(p) <= 180
                      for p, t in zip(prompt_norm, data._text_norm)], index=data.index)
    reasons["prompt_echo_or_extraction"] = remove(echo, "prompt_echo_or_extraction")

    kept = data.loc[data._keep].copy()
    # A model_version that exactly names another canonical class is a direct
    # provenance contradiction. General provider/route prose is too ambiguous.
    canonical_label = {re.sub(r"[^a-z0-9]", "", str(label).casefold()): str(label)
                       for label in original_classes}
    version = data.get("model_version", pd.Series("", index=data.index)).fillna("").astype(str)
    version_key = version.map(lambda value: re.sub(r"[^a-z0-9]", "", value.casefold()))
    recognized_version = version_key.map(canonical_label)
    provenance_conflict = recognized_version.notna() & recognized_version.ne(data.model.astype(str))
    reasons["provenance_label_conflict"] = remove(provenance_conflict, "provenance_label_conflict")
    kept = data.loc[data._keep].copy()
    # Any identical normalized response assigned to multiple labels is ambiguous;
    # remove the entire conflicting group instead of arbitrarily choosing a label.
    label_counts = kept.groupby("_text_norm").model.nunique()
    conflict_texts = set(label_counts[label_counts.gt(1)].index)
    reasons["exact_response_label_conflict"] = remove(data._text_norm.isin(conflict_texts), "exact_response_label_conflict")
    suspicious_conflict_rows = int(kept._text_norm.isin(conflict_texts).sum()) + reasons["provenance_label_conflict"]

    def quality_key(i):
        row = data.loc[i]
        text = str(row.text)
        provenance_fields = [c for c in ("source_dataset", "source_file", "source_license", "origin_hint", "provider", "model_version", "route") if c in data]
        provenance = sum(pd.notna(row[c]) and str(row[c]).strip().casefold() not in {"", "n/s", "none", "nan"} for c in provenance_fields)
        endings = text.rstrip()[-1:] in ".!?`)'\"”’)]}"
        integrity = int(bool(re.search(r"[A-Za-z0-9]", text))) + int(endings)
        return (provenance, integrity, len(text), hashlib.sha256(text.encode("utf-8")).hexdigest())

    exact_duplicate_rows = set()
    same_label_groups = data.loc[data._keep].groupby("_text_norm", sort=True)
    for _, group in same_label_groups:
        if len(group) > 1:
            winner = max(group.index.tolist(), key=quality_key)
            exact_duplicate_rows.update(i for i in group.index if i != winner)
    reasons["exact_duplicate_response"] = remove(data.index.isin(exact_duplicate_rows), "exact_duplicate_response")

    # Word-trigram MinHash narrows candidate pairs, then exact character n-gram
    # Jaccard verifies candidates before near-duplicate removal.
    eligible_idx = data.index[data._keep].tolist()
    # Fixed 4-row bands keep candidate recall high even for conservative custom
    # thresholds; exact Jaccard applies the configured threshold afterward.
    lsh = MinHashLSH(threshold=near_duplicate_threshold, num_perm=minhash_permutations,
                     params=(minhash_permutations // 4, 4))
    signatures = {}
    for i in eligible_idx:
        text = data.at[i, "_text_norm"]
        words = text.split()
        # Word-trigram MinHash is a compact candidate index. Exact character
        # n-gram Jaccard below verifies every candidate before removal.
        word_grams = {" ".join(words[j:j + 3]) for j in range(max(1, len(words) - 2))}
        signature = MinHash(num_perm=minhash_permutations, seed=seed,
                           hashfunc=lambda value: zlib.crc32(value))
        signature.update_batch([gram.encode("utf-8") for gram in word_grams])
        key = str(i)
        signatures[i] = signature
        lsh.insert(key, signature)
    parent = {i: i for i in eligible_idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    near_pairs = set()
    cross_label_near_pairs = set()
    for i in eligible_idx:
        for match in lsh.query(signatures[i]):
            j = int(match)
            if j <= i:
                continue
            ta, tb = data.at[i, "_text_norm"], data.at[j, "_text_norm"]
            ga = {ta[k:k + char_ngram_size] for k in range(max(1, len(ta) - char_ngram_size + 1))}
            gb = {tb[k:k + char_ngram_size] for k in range(max(1, len(tb) - char_ngram_size + 1))}
            similarity = len(ga & gb) / max(1, len(ga | gb))
            if similarity >= near_duplicate_threshold:
                near_pairs.add((i, j))
                if str(data.at[i, "model"]) == str(data.at[j, "model"]):
                    union(i, j)
                else:
                    cross_label_near_pairs.add((i, j))
    clusters: dict[int, list[int]] = {}
    for i in eligible_idx:
        clusters.setdefault(find(i), []).append(i)
    duplicate_cluster_count = sum(len(items) > 1 for items in clusters.values())

    def diverse_order(frame):
        """Order rows round-robin across length, formatting and lexical-richness bins."""
        sub = frame.copy()
        lengths = sub.text.astype(str).str.len()
        sub["_length_bin"] = pd.qcut(lengths.rank(method="first"), q=min(8, len(sub)),
                                     labels=False, duplicates="drop")
        sub["_format_bin"] = sub.text.astype(str).str.contains(
            r"(?m)^\s*(?:[-*#]|\d+[.)])\s|```|\|", regex=True).astype(int)
        lexical = sub.text.astype(str).map(lambda value: len(set(re.findall(r"[a-z]+", value.casefold()))) /
                                           max(1, len(re.findall(r"[a-z]+", value.casefold()))))
        sub["_lexical_bin"] = pd.qcut(lexical.rank(method="first"), q=min(3, len(sub)),
                                      labels=False, duplicates="drop")
        queues = {}
        for key, bucket in sub.groupby(["_length_bin", "_format_bin", "_lexical_bin"], sort=True, dropna=False):
            queues[key if isinstance(key, tuple) else (key,)] = sorted(bucket.index, key=quality_key, reverse=True)
        ordered = []
        keys = sorted(queues, key=lambda value: tuple(str(x) for x in value))
        while any(queues.values()):
            for key in keys:
                if queues[key]:
                    ordered.append(queues[key].pop(0))
        return ordered

    near_removed = set()
    for members in clusters.values():
        if len(members) < 2:
            continue
        winner = max(members, key=quality_key)
        near_removed.update(i for i in members if i != winner)
    reasons["near_duplicate_response"] = remove(data.index.isin(near_removed), "near_duplicate_response")

    # Repeated templated boilerplate is detected after replacing numbers and URLs.
    # It is capped only when the normalized template is long and occurs repeatedly.
    def template_signature(text):
        value = _normal_text(text)
        value = re.sub(r"https?://\S+", " <url> ", value)
        value = re.sub(r"\b\d+(?:[.,]\d+)*\b", " <num> ", value)
        return re.sub(r"\s+", " ", value).strip()

    data["_template"] = data.text.fillna("").astype(str).map(template_signature)
    template_rows = data.loc[data._keep]
    template_removed = set()
    for (_, template), group in template_rows.groupby(["model", "_template"], sort=True):
        if len(group) < template_min_repeats or len(template) < minimum_template_chars:
            continue
        # Keep a template representative from each source, not each prompt. A
        # long signature repeated across prompts inside one source is the target.
        ordered = sorted(group.index, key=quality_key, reverse=True)
        seen = set()
        for i in ordered:
            key = str(data.at[i, "source_dataset"])
            if key in seen:
                template_removed.add(i)
            else:
                seen.add(key)
    reasons["repeated_template_boilerplate"] = remove(data.index.isin(template_removed), "repeated_template_boilerplate")

    # Adaptive caps use each label's observed source and source-prompt redundancy
    # distributions. No model receives an equal row quota.
    source_removed, prompt_removed = set(), set()
    active = data.loc[data._keep].copy()
    active["_source_count"] = active.groupby(["model", "source_dataset"]).id.transform("size")
    for label, group in active.groupby("model", sort=True):
        counts = group.groupby("source_dataset").size()
        if len(counts) >= 4:
            q1, q3 = counts.quantile([.25, .75])
            tukey_cap = int(math.ceil(q3 + 1.5 * (q3 - q1)))
            median_cap = int(math.ceil(counts.median() * source_median_multiplier))
            cap = max(int(math.ceil(q3)), min(tukey_cap, median_cap))
            overloaded = set(counts[counts > cap].index.astype(str))
            for source in overloaded:
                sub = group[group.source_dataset.astype(str).eq(source)].copy()
                # Round-robin length, formatting, and vocabulary-richness bins,
                # using provenance completeness and integrity within each bin.
                priority = diverse_order(sub)
                source_removed.update(priority[cap:])
    reasons["adaptive_source_redundancy_cap"] = remove(data.index.isin(source_removed), "adaptive_source_redundancy_cap")

    active = data.loc[data._keep].copy()
    for _, source_group in active.groupby(["model", "source_dataset"], sort=True):
        all_counts = source_group.groupby("_prompt_key").size()
        if len(all_counts) < 4:
            continue
        q1, q3 = all_counts.quantile([.25, .75])
        cap = max(2, int(math.ceil(q3 + 1.5 * (q3 - q1))))
        for prompt, count in all_counts[all_counts.gt(cap)].items():
            members = source_group.index[source_group._prompt_key.eq(prompt)].tolist()
            ordered = diverse_order(data.loc[members])
            prompt_removed.update(ordered[cap:])
    reasons["adaptive_prompt_redundancy_cap"] = remove(data.index.isin(prompt_removed), "adaptive_prompt_redundancy_cap")

    selected = data.loc[data._keep].copy()
    if set(selected.model.astype(str)) != original_classes:
        raise ValueError("curation removed an entire model class")
    # Freeze V0 validation identity and ensure neither response IDs nor prompt IDs
    # overlap. Open Pool prompt groups may be canonical text or source prompt IDs.
    v0_profile_path = output_dir / "dataset_profile.json"
    v0_split_path = output_dir / "splits_v0.parquet"
    v0_data_path = output_dir / "dataset_v0.parquet"
    if not (v0_profile_path.is_file() and v0_split_path.is_file() and v0_data_path.is_file()):
        raise FileNotFoundError("prepare V0 first; its frozen validation rows are required")
    v0_profile = json.loads(v0_profile_path.read_text())
    v0_assignments = pd.read_parquet(v0_split_path)
    v0 = pd.read_parquet(v0_data_path)
    v0_val_ids = set(v0_assignments.loc[v0_assignments.split.eq("validation"), "id"].astype(str))
    v0_val = v0[v0.id.astype(str).isin(v0_val_ids)]
    v0_ids_hash = hashlib.sha256("\n".join(sorted(v0_val_ids)).encode()).hexdigest()
    if len(v0_val) != int(v0_profile.get("validation_rows", -1)):
        raise ValueError("V0 validation assignments do not match the frozen profile")
    id_overlap = set(selected.id.astype(str)) & v0_val_ids
    v0_prompt_ids = set(v0_val.prompt_id.astype(str))
    prompt_id_overlap = set(selected.prompt_id.astype(str)) & v0_prompt_ids
    selected_prompt_keys = set(selected._prompt_key.astype(str))
    if "prompt_canon" in v0_val:
        v0_prompt_keys = set(v0_val.prompt_canon.fillna("").astype(str).map(_normal_text))
        v0_prompt_keys.discard("")
        canonical_overlap = selected_prompt_keys & v0_prompt_keys
    else:
        canonical_overlap = set()
    prompt_overlap = prompt_id_overlap | canonical_overlap
    if id_overlap or prompt_overlap:
        raise ValueError(f"curated training data overlaps V0 validation: {len(id_overlap)} IDs, {len(prompt_overlap)} prompt groups")
    if set(selected.model.astype(str)) != set(v0_val.model.astype(str)):
        raise ValueError("curated training dataset must preserve all V0 model labels")
    retained_prompt_keys = selected.groupby("model")._prompt_key.nunique()
    retained_unique_prompts = int(selected._prompt_key.nunique())
    selected = selected.drop(columns=["_keep", "_text_norm", "_prompt_key", "_template", "_original_index", "_source_count"], errors="ignore")
    selected = selected.sort_values("id", kind="stable").reset_index(drop=True)
    removals = pd.concat(removal_parts, ignore_index=True) if removal_parts else pd.DataFrame()
    audit_cols = [c for c in ("_original_index", "id", "model", "source_dataset", "source_file", "source_license", "prompt_id", "removal_reason") if c in removals]
    audit = removals[audit_cols].rename(columns={"_original_index": "original_row_index"})
    if not audit.empty:
        audit = audit.sort_values("original_row_index", kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file, profile_file = _dataset_artifact_names(dataset_version)
    output_path, profile_path = output_dir / dataset_file, output_dir / profile_file
    if output_path.exists() or profile_path.exists():
        raise FileExistsError(f"refusing to overwrite existing curation artifacts for {dataset_version}")
    selected.to_parquet(output_path, index=False)
    audit_path = output_dir / f"{dataset_version}_removals.csv"
    audit.to_csv(audit_path, index=False, lineterminator="\n")
    source_hash = _hash_file(source_path)
    before_class = data.groupby("model").size()
    after_class = selected.groupby("model").size()
    prompts_after = retained_prompt_keys
    profile = {
        "dataset_version": dataset_version, "rows": int(len(selected)),
        "dataset_sha256": hashlib.sha256(selected.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
        "source_dataset_path": str(source_path), "source_dataset_sha256": source_hash,
        "classes": sorted(selected.model.astype(str).unique()),
        "class_counts": {str(k): int(v) for k, v in after_class.items()},
        "prompt_groups": int(selected.prompt_id.nunique()),
        "unique_sources": int(selected.source_dataset.nunique()),
        "training_only": True, "curation_method": "v1_all_quality_minhash_adaptive_diversity",
        "seed": seed, "near_duplicate_threshold": near_duplicate_threshold,
        "char_ngram_size": char_ngram_size, "minhash_permutations": minhash_permutations,
        "source_median_multiplier": source_median_multiplier,
        "minhash_candidate_shingles": "word trigrams",
        "near_duplicate_verification": "character n-gram Jaccard",
        "near_duplicate_clusters": int(duplicate_cluster_count),
        "near_duplicate_pairs_verified": int(len(near_pairs)),
        "suspicious_near_duplicate_cross_label_clusters": int(len(cross_label_near_pairs)),
        "suspicious_exact_label_conflict_rows": suspicious_conflict_rows,
        "removals_by_reason": reasons, "removed_rows": int(len(audit)),
        "removal_audit_path": str(audit_path),
        "removal_manifest_sha256": _hash_file(audit_path),
        "unique_prompts_before": int(data._prompt_key.nunique()),
        "unique_prompts_after": retained_unique_prompts,
        "unique_sources_before": int(data.source_dataset.nunique()),
        "source_counts_before": {str(k): int(v) for k, v in data.source_dataset.value_counts().sort_index().items()},
        "source_counts_after": {str(k): int(v) for k, v in selected.source_dataset.value_counts().sort_index().items()},
        "class_summary": {str(label): {"original_count": int(before_class.get(label, 0)),
            "retained_count": int(after_class.get(label, 0)),
            "retained_fraction": float(after_class.get(label, 0) / max(1, before_class.get(label, 0))),
            "unique_prompts_retained": int(prompts_after.get(label, 0)),
            "sources_retained": int(selected.loc[selected.model.eq(label), "source_dataset"].nunique())}
            for label in sorted(before_class.index.astype(str))},
        "length_chars_before": {str(k): float(v) for k, v in data.text.fillna("").astype(str).str.len().quantile([0, .1, .5, .9, 1]).items()},
        "length_chars_after": {str(k): float(v) for k, v in selected.text.fillna("").astype(str).str.len().quantile([0, .1, .5, .9, 1]).items()},
        "source_concentration_before": _source_concentration(data),
        "source_concentration_after": _source_concentration(selected),
        "validation_dataset_version": "v0", "validation_split_sha256": v0_profile["split_sha256"],
        "validation_rows": int(len(v0_val)), "validation_ids_sha256": v0_ids_hash,
        "validation_id_overlap": 0, "validation_prompt_overlap": 0,
    }
    # Use the pipeline's training-only profile contract; the loader attaches V0
    # validation and will independently verify ID and prompt isolation.
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def curate_v1_all_clean_compact(source_path: str | Path, output_dir: str | Path, *,
                                dataset_version: str = "v1-all",
                                validation_dataset: str = "v0", seed: int = 2026,
                                expected_classes: int | None = 18,
                                minhash_permutations: int = 64,
                                compact_char_similarity: float = 0.92,
                                compact_word_similarity: float = 0.85,
                                tfidf_diagnostic_sample: int = 3000) -> dict:
    """Create defect-only clean and conservative redundancy-reduced compact sets.

    Clean removes only explicit, auditable defects. Compact starts from clean and
    removes only same-label responses verified as highly similar in both character
    and word n-gram Jaccard space. ``dataset_version`` names the paired ``_clean``
    and ``_compact`` outputs. No source quotas or model-performance signals affect
    selection.
    """
    from datasketch import MinHash, MinHashLSH
    import zlib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    if validation_dataset != "v0":
        raise ValueError("these datasets reuse the frozen V0 validation set")
    if not 0 < compact_char_similarity < 1 or not 0 < compact_word_similarity < 1:
        raise ValueError("compact similarity thresholds must be in (0, 1)")
    if minhash_permutations < 16 or tfidf_diagnostic_sample < 0:
        raise ValueError("invalid MinHash or diagnostic sample setting")
    source_path, output_dir = Path(source_path), Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    data = pd.read_parquet(source_path).reset_index(drop=True)
    required = {"id", "text", "model", "source_dataset", "prompt_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"source dataset missing required columns: {sorted(missing)}")
    if expected_classes is not None and data.model.nunique(dropna=True) != expected_classes:
        raise ValueError(f"expected {expected_classes} model labels; found {data.model.nunique(dropna=True)}")
    original_classes = set(data.model.dropna().astype(str))
    data["_original_row_index"] = np.arange(len(data), dtype=np.int64)
    data["_normalized_text"] = data.text.fillna("").astype(str).map(_normal_text)

    # Dataset-level counts and baseline distributions are collected before curation.
    def descriptive_stats(frame):
        texts = frame.text.fillna("").astype(str)
        lengths = texts.str.len()
        words = texts.str.findall(r"\b[\w'-]+\b").str.len()
        markdown = texts.str.contains(r"(?m)^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)|```|\|", regex=True)
        code = texts.str.contains("```")
        urls = texts.str.contains(r"https?://", case=False, regex=True)
        def boundary_signature(value, from_end=False):
            tokens = re.findall(r"[a-z0-9]+|[^\w\s]", value.casefold())
            tokens = tokens[-8:] if from_end else tokens[:8]
            return " ".join(re.sub(r"\d+", "<n>", token) for token in tokens)
        openings = texts.map(boundary_signature).value_counts()
        endings = texts.map(lambda value: boundary_signature(value, True)).value_counts()
        paragraphs = texts.str.count(r"\n\s*\n") + 1
        return {
            "rows": int(len(frame)),
            "prompt_groups": int(frame.prompt_id.nunique(dropna=True)),
            "unique_sources": int(frame.source_dataset.nunique(dropna=True)),
            "length_chars": {str(q): float(lengths.quantile(q)) for q in (0, .1, .5, .9, 1)},
            "length_words": {str(q): float(words.quantile(q)) for q in (0, .1, .5, .9, 1)},
            "mean_lexical_diversity": float(texts.map(lambda t: len(set(re.findall(r"[a-z]+", t.casefold()))) /
                                                       max(1, len(re.findall(r"[a-z]+", t.casefold())))).mean()),
            "markdown_fraction": float(markdown.mean()), "code_block_fraction": float(code.mean()),
            "url_fraction": float(urls.mean()),
            "paragraph_count_median": float(paragraphs.median()),
            "repeated_openings_10plus": int((openings >= 10).sum()),
            "repeated_opening_rows_10plus": int(openings[openings >= 10].sum()),
            "repeated_endings_10plus": int((endings >= 10).sum()),
            "repeated_ending_rows_10plus": int(endings[endings >= 10].sum()),
        }

    before_stats = descriptive_stats(data)
    clean = data.copy()
    clean["_keep"] = True
    clean_removals = []

    def remove_clean(mask, reason, evidence):
        mask = pd.Series(mask, index=clean.index).fillna(False).astype(bool) & clean._keep
        for idx in clean.index[mask]:
            row = clean.loc[idx].copy()
            row["removal_reason"] = reason
            row["evidence"] = str(evidence(row)) if callable(evidence) else str(evidence)
            row["representative_id"] = ""
            clean_removals.append(row)
        clean.loc[mask, "_keep"] = False
        return int(mask.sum())

    clean_reasons = {}
    valid = (clean.id.notna() & clean.id.astype(str).str.strip().ne("") & clean.model.notna()
             & clean.text.notna() & clean._normalized_text.str.len().gt(0)
             & clean.source_dataset.notna() & clean.source_dataset.astype(str).str.strip().ne("")
             & clean.prompt_id.notna() & clean.prompt_id.astype(str).str.strip().ne(""))
    clean_reasons["missing_or_empty_required_field"] = remove_clean(
        ~valid, "missing_or_empty_required_field", "one or more required fields are null/empty")
    clean_reasons["duplicate_row_id"] = remove_clean(
        clean.id.astype(str).duplicated(keep="first"), "duplicate_row_id", "duplicate id; first source row retained")

    # Only unmistakable serialization/log artifacts are removed. Content that merely
    # discusses tools, errors, truncation, or code remains untouched.
    artifact_re = re.compile(
        r"(?is)(?:\[object object\]|\x00)"
        r"|(?:^\s*<\|im_start\|>(?:system|user|assistant|tool)\b)"
        r"|(?:<\|im_start\|>.*?<\|im_end\|>)"
        r"|(?:^\s*(?:tool call|tool response|function call|execution log)\s*:\s*\{.{0,100}\}\s*$)"
    )
    clean_reasons["serialized_envelope_or_extraction_artifact"] = remove_clean(
        clean._normalized_text.map(lambda value: bool(artifact_re.search(value))),
        "serialized_envelope_or_extraction_artifact", "matched explicit serialized role/object/tool-log marker")
    trunc_re = re.compile(r"(?i)(?:\[(?:truncated|incomplete response|generation failed)\]|\((?:truncated|generation failed)\)|\.\.\.\s*\[end\])\s*$")
    clean_reasons["explicit_truncation_or_failure_marker"] = remove_clean(
        clean._normalized_text.map(lambda value: bool(trunc_re.search(value))),
        "explicit_truncation_or_failure_marker", "response ends with an explicit truncation/failure marker")
    def explicit_true(column):
        if column not in clean:
            return pd.Series(False, index=clean.index)
        return clean[column].map(lambda value: (str(value).strip().casefold() in {"true", "1", "yes"})
                                 if pd.notna(value) else False)
    finish = clean.get("source_finish_reason", pd.Series("", index=clean.index)).fillna("").astype(str).str.strip().str.casefold()
    truncated_finish = finish.isin({"length", "max_tokens", "token_limit", "length_capped"})
    clean_reasons["source_explicit_truncation_metadata"] = remove_clean(
        explicit_true("source_truncated") | truncated_finish,
        "source_explicit_truncation_metadata", "source metadata explicitly marks a truncated or length-capped response")
    clean_reasons["source_explicit_cap_hit"] = remove_clean(
        explicit_true("source_cap_hit"),
        "source_explicit_cap_hit", "source metadata explicitly marks a generation cap hit")

    # Cross-label identical normalized text is a direct ambiguity; remove all members
    # and retain the row IDs as evidence. Same-label duplicates keep the best-backed row.
    active = clean.loc[clean._keep]
    label_n = active.groupby("_normalized_text", sort=True).model.nunique()
    conflict_texts = set(label_n[label_n.gt(1)].index)
    conflict_ids = active.loc[active._normalized_text.isin(conflict_texts)].groupby("_normalized_text").id.apply(
        lambda values: ",".join(sorted(values.astype(str))))
    clean_reasons["conflicting_labels_for_identical_response"] = remove_clean(
        clean._normalized_text.isin(conflict_texts), "conflicting_labels_for_identical_response",
        lambda row: f"normalized text assigned to IDs {conflict_ids.get(row._normalized_text, '')}")

    def provenance_score(row):
        fields = ("source_dataset", "source_file", "source_license", "origin_hint", "provider", "model_version", "route")
        return sum(c in clean and pd.notna(row.get(c)) and str(row.get(c)).strip().casefold() not in {"", "nan", "none", "n/s"} for c in fields)

    active = clean.loc[clean._keep]
    exact_groups = active.groupby("_normalized_text", sort=True)
    exact_duplicate_indices = set()
    exact_representative = {}
    for text_key, group in exact_groups:
        if len(group) < 2:
            continue
        ordered = sorted(group.index, key=lambda i: (provenance_score(clean.loc[i]),
                          len(clean.at[i, "_normalized_text"]), str(clean.at[i, "id"])), reverse=True)
        exact_representative.update({i: str(clean.at[ordered[0], "id"]) for i in ordered[1:]})
        exact_duplicate_indices.update(ordered[1:])
    n = remove_clean(clean.index.isin(exact_duplicate_indices), "normalized_duplicate_response",
                     lambda row: f"normalized text equals retained representative {exact_representative.get(row.name, '')}")
    clean_reasons["normalized_duplicate_response"] = n

    # Direct contradictions only: model_version must normalize exactly to a known
    # canonical class name. Other provenance strings are retained as ambiguous.
    if "model_version" in clean:
        canonical = {re.sub(r"[^a-z0-9]", "", str(label).casefold()): str(label) for label in original_classes}
        version = clean.model_version.fillna("").astype(str).map(
            lambda value: re.sub(r"[^a-z0-9]", "", value.casefold()))
        identified = version.map(canonical)
        contradictions = identified.notna() & identified.ne(clean.model.astype(str))
    else:
        contradictions = pd.Series(False, index=clean.index)
    clean_reasons["contradictory_model_version"] = remove_clean(
        contradictions, "contradictory_model_version", "model_version exactly names a different canonical class")

    # Prompt-copy rule requires a long canonical prompt reproduced at the start and
    # at least 80% of response characters to be copied; no short/common prefix hits.
    prompt = clean.get("prompt_canon", pd.Series("", index=clean.index)).fillna("").astype(str).map(_normal_text)
    copied = pd.Series([len(p) >= 100 and text.startswith(p) and len(text) > 0 and len(p) / len(text) >= .8
                        for p, text in zip(prompt, clean._normalized_text)], index=clean.index)
    clean_reasons["response_is_copied_prompt"] = remove_clean(
        copied, "response_is_copied_prompt", "canonical prompt is >=80% of response and repeated at its start")

    clean_rows = clean.loc[clean._keep].drop(columns=["_keep", "_normalized_text", "_original_row_index"], errors="ignore")
    if set(clean_rows.model.astype(str)) != original_classes:
        raise ValueError("clean curation must preserve every source model label")
    if clean_rows.id.astype(str).duplicated().any():
        raise ValueError("clean dataset still contains duplicate IDs")
    if clean_rows.text.isna().any() or clean_rows.text.astype(str).str.strip().eq("").any():
        raise ValueError("clean dataset contains empty responses")

    # Freeze V0 validation identity and enforce ID, prompt ID and canonical prompt isolation.
    v0_profile_path = output_dir / "dataset_profile.json"
    v0_split_path = output_dir / "splits_v0.parquet"
    v0_data_path = output_dir / "dataset_v0.parquet"
    if not all(path.is_file() for path in (v0_profile_path, v0_split_path, v0_data_path)):
        raise FileNotFoundError("prepare V0 first; its frozen validation rows are required")
    v0_profile = json.loads(v0_profile_path.read_text(encoding="utf-8"))
    v0_assignments = pd.read_parquet(v0_split_path)
    v0 = pd.read_parquet(v0_data_path)
    v0_val_ids = set(v0_assignments.loc[v0_assignments.split.eq("validation"), "id"].astype(str))
    v0_val = v0[v0.id.astype(str).isin(v0_val_ids)]
    if len(v0_val) != int(v0_profile.get("validation_rows", -1)):
        raise ValueError("V0 validation assignments do not match the frozen profile")
    v0_prompt_ids = set(v0_val.prompt_id.astype(str))
    canonical_overlap = set()
    if "prompt_canon" in v0_val and "prompt_canon" in clean_rows:
        canonical_overlap = (set(clean_rows.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}) & \
                            (set(v0_val.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""})
    id_overlap = set(clean_rows.id.astype(str)) & v0_val_ids
    prompt_overlap = (set(clean_rows.prompt_id.astype(str)) & v0_prompt_ids) | canonical_overlap
    if id_overlap or prompt_overlap:
        raise ValueError(f"source dataset overlaps frozen V0 validation: {len(id_overlap)} IDs, {len(prompt_overlap)} prompt groups")
    if not set(clean_rows.model.astype(str)).issubset(set(v0_val.model.astype(str))):
        raise ValueError("curated dataset contains model labels absent from V0 validation")

    # Compact: union character and word-trigram MinHash candidates, then verify
    # exact character 5-gram and word-trigram Jaccard. Only within-label pairs qualify.
    active = clean_rows.reset_index(drop=True).copy()
    active["_normalized_text"] = active.text.astype(str).map(_normal_text)
    n_perm = int(minhash_permutations)
    word_lsh = MinHashLSH(threshold=compact_word_similarity, num_perm=n_perm)
    char_lsh = MinHashLSH(threshold=compact_char_similarity, num_perm=n_perm)
    signatures = {}
    for i, text in enumerate(active._normalized_text):
        chars = {text[k:k+5] for k in range(max(1, len(text)-4))}
        words = text.split()
        word_grams = {" ".join(words[j:j+3]) for j in range(max(1, len(words)-2))}
        word_sig = MinHash(num_perm=n_perm, seed=seed, hashfunc=lambda value: zlib.crc32(value))
        word_sig.update_batch([gram.encode("utf-8") for gram in word_grams])
        char_sig = MinHash(num_perm=n_perm, seed=seed, hashfunc=lambda value: zlib.crc32(value))
        char_sig.update_batch([gram.encode("utf-8") for gram in chars])
        signatures[i] = (char_sig, word_sig)
        word_lsh.insert(str(i), word_sig)
        char_lsh.insert(str(i), char_sig)

    parent = list(range(len(active)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    verified_pairs = []
    candidate_similarity = []
    for i in range(len(active)):
        matches = set(char_lsh.query(signatures[i][0])) | set(word_lsh.query(signatures[i][1]))
        text_i = active.at[i, "_normalized_text"]
        chars_i = {text_i[k:k+5] for k in range(max(1, len(text_i)-4))} if matches else set()
        words_i = text_i.split()
        word_grams_i = {" ".join(words_i[k:k+3]) for k in range(max(1, len(words_i)-2))} if matches else set()
        for match in matches:
            j = int(match)
            if j <= i or str(active.at[i, "model"]) != str(active.at[j, "model"]):
                continue
            text_j = active.at[j, "_normalized_text"]
            chars_j = {text_j[k:k+5] for k in range(max(1, len(text_j)-4))}
            words_j = text_j.split()
            word_grams_j = {" ".join(words_j[k:k+3]) for k in range(max(1, len(words_j)-2))}
            char_sim = len(chars_i & chars_j) / max(1, len(chars_i | chars_j))
            word_sim = len(word_grams_i & word_grams_j) / max(1, len(word_grams_i | word_grams_j))
            candidate_similarity.append((float(char_sim), float(word_sim)))
            if char_sim >= compact_char_similarity and word_sim >= compact_word_similarity:
                same_prompt = str(active.at[i, "prompt_id"]) == str(active.at[j, "prompt_id"])
                verified_pairs.append((i, j, float(char_sim), float(word_sim), same_prompt))
                union(i, j)
    clusters = {}
    for i in range(len(active)):
        clusters.setdefault(find(i), []).append(i)
    redundant_indices = {}
    cluster_records = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        def representative_key(i):
            row = active.iloc[i]
            prov = provenance_score(row)
            complete_text = int(bool(re.search(r"[A-Za-z0-9]", str(row.text))) and
                                str(row.text).rstrip()[-1:] in ".!?`)'\"”’)]}")
            tokens = re.findall(r"[a-z]+", str(row.text).casefold())
            lexical = len(set(tokens)) / max(1, len(tokens))
            return (prov, complete_text, lexical, len(str(row.text)), hashlib.sha256(str(row.id).encode()).hexdigest())
        rep_idx = max(members, key=representative_key)
        rep_id = str(active.at[rep_idx, "id"])
        for idx in members:
            if idx != rep_idx:
                redundant_indices[idx] = rep_id
        pairs = [item for item in verified_pairs if item[0] in members and item[1] in members]
        cluster_records.append({"representative_id": rep_id, "members": [str(active.at[i, "id"]) for i in members],
                                "size": len(members),
                                "max_char_jaccard": max((item[2] for item in pairs), default=1.0),
                                "max_word_jaccard": max((item[3] for item in pairs), default=1.0),
                                "same_prompt_pair_count": sum(bool(item[4]) for item in pairs)})

    compact_removed = active.iloc[sorted(redundant_indices)].copy() if redundant_indices else active.iloc[:0].copy()
    if len(compact_removed):
        compact_removed["removal_reason"] = "verified_same_label_near_duplicate"
        compact_removed["evidence"] = compact_removed.index.map(lambda i: next(
            (f"char5_jaccard={c:.4f}; word3_jaccard={w:.4f}" for a,b,c,w,_same_prompt in verified_pairs
             if i in (a,b) and str(active.at[a if b == i else b, "id"]) == redundant_indices[i]), "verified cluster member"))
        compact_removed["representative_id"] = compact_removed.index.map(redundant_indices)
    compact = active.loc[~active.index.isin(redundant_indices)].drop(columns="_normalized_text")
    if set(compact.id.astype(str)) - set(clean_rows.id.astype(str)):
        raise RuntimeError("compact contains rows absent from clean")
    if set(compact.model.astype(str)) != original_classes:
        raise ValueError("compact curation must preserve every source model label")

    # Diagnostic template and prompt redundancy counts; these are not removal rules.
    def template_key(text):
        value = _normal_text(text)
        value = re.sub(r"https?://\S+", " <url> ", value)
        value = re.sub(r"\b\d+(?:[.,]\d+)*\b", " <num> ", value)
        return re.sub(r"\s+", " ", value).strip()
    templates = active.assign(_template=active.text.map(template_key))
    template_sizes = templates.groupby(["model", "_template"], sort=True).size()
    repeated_templates = template_sizes[template_sizes.ge(3)]
    same_prompt_sizes = active.groupby(["model", "prompt_id"], dropna=False).size()
    prompt_repeated = same_prompt_sizes[same_prompt_sizes.gt(1)]

    # TF-IDF nearest-neighbor diagnostic is deterministic and sampled, never a selector.
    tfidf_report = {"method": "word unigram/bigram TF-IDF cosine nearest-neighbor on deterministic sample",
                    "sample_size": 0, "nearest_cosine_p50": None, "nearest_cosine_p90": None,
                    "nearest_cosine_max": None, "note": "diagnostic only; not used for selection"}
    if tfidf_diagnostic_sample and len(active) > 1:
        sample = active.sort_values("id", kind="stable").sample(
            min(int(tfidf_diagnostic_sample), len(active)), random_state=seed)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2,
                                     sublinear_tf=True, dtype=np.float32)
        matrix = vectorizer.fit_transform(sample.text.astype(str))
        if matrix.shape[1] and matrix.shape[0] > 1:
            nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=1).fit(matrix)
            distances, _ = nn.kneighbors(matrix)
            nearest = np.clip(1.0 - distances[:, 1], 0, 1)
            tfidf_report = {"method": tfidf_report["method"], "sample_size": int(len(sample)),
                            "nearest_cosine_p50": float(np.quantile(nearest, .5)),
                            "nearest_cosine_p90": float(np.quantile(nearest, .9)),
                            "nearest_cosine_max": float(nearest.max()),
                            "note": "diagnostic only; not used for selection"}

    def similarity_summary(values, column):
        scores = np.asarray([item[column] for item in values], dtype=float)
        return {"count": int(len(scores)), "p50": float(np.quantile(scores, .5)) if len(scores) else None,
                "p90": float(np.quantile(scores, .9)) if len(scores) else None,
                "max": float(scores.max()) if len(scores) else None}

    clean_removals = (pd.DataFrame(clean_removals) if clean_removals else
                      pd.DataFrame(columns=[*data.columns, "removal_reason", "evidence", "representative_id"]))
    compact_removed = compact_removed.reset_index(drop=False).rename(columns={"index": "_compact_row_index"})
    clean_rows = clean_rows.sort_values("id", kind="stable").reset_index(drop=True)
    compact = compact.sort_values("id", kind="stable").reset_index(drop=True)
    clean_stats, compact_stats = descriptive_stats(clean_rows), descriptive_stats(compact)
    source_hash = _hash_file(source_path)
    split_hash = str(v0_profile["split_sha256"])
    val_ids_hash = hashlib.sha256("\n".join(sorted(v0_val_ids)).encode()).hexdigest()

    def write_dataset(frame, version, removals, summary, parent_version=None):
        if frame.id.astype(str).duplicated().any():
            raise ValueError(f"{version} contains duplicate IDs")
        dataset_file, profile_file = _dataset_artifact_names(version)
        out_path, prof_path = output_dir / dataset_file, output_dir / profile_file
        audit_path = output_dir / f"{version}_removals.csv"
        if any(path.exists() for path in (out_path, prof_path, audit_path)):
            raise FileExistsError(f"refusing to overwrite existing artifacts for {version}")
        public = frame.copy()
        public.to_parquet(out_path, index=False)
        audit_columns = [c for c in ("_original_row_index", "_compact_row_index", "id", "model",
            "source_dataset", "source_file", "source_row_id", "source_finish_reason",
            "source_truncated", "source_cap_hit", "prompt_id", "removal_reason", "evidence",
            "representative_id") if c in removals]
        audit = removals[audit_columns].copy() if len(removals) else pd.DataFrame(columns=audit_columns or
            ["original_row_index", "id", "model", "source_dataset", "source_file", "source_row_id", "prompt_id",
             "removal_reason", "evidence", "representative_id"])
        audit.to_csv(audit_path, index=False, lineterminator="\n")
        profile = {
            "dataset_version": version, "rows": int(len(public)),
            "dataset_sha256": hashlib.sha256(public.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
            "source_dataset_path": str(source_path), "source_dataset_sha256": source_hash,
            "parent_dataset_version": parent_version, "parent_dataset_sha256": (
                hashlib.sha256(clean_rows.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
                if parent_version else None),
            "classes": sorted(public.model.astype(str).unique()),
            "class_counts": {str(k): int(v) for k,v in public.model.value_counts().sort_index().items()},
            "prompt_groups": int(public.prompt_id.nunique()), "unique_sources": int(public.source_dataset.nunique()),
            "training_only": True, "curation_method": f"{dataset_version}_{'defect_clean' if version.endswith('_clean') else 'verified_redundancy_compact'}",
            "seed": seed, "removals_by_reason": summary,
            "removed_rows": int(len(audit)), "removal_audit_path": str(audit_path),
            "removed_rows_with_representative": int(audit.representative_id.fillna("").astype(str).str.strip().ne("").sum())
                if "representative_id" in audit else 0,
            "removal_manifest_sha256": _hash_file(audit_path),
            "validation_dataset_version": "v0", "validation_split_sha256": split_hash,
            "validation_rows": int(len(v0_val)), "validation_ids_sha256": val_ids_hash,
            "validation_id_overlap": 0, "validation_prompt_overlap": 0,
        }
        prof_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        return profile

    clean_removed = pd.concat([clean_removals], ignore_index=True) if len(clean_removals) else pd.DataFrame()
    clean_version = f"{dataset_version}_clean"
    compact_version = f"{dataset_version}_compact"
    clean_profile = write_dataset(clean_rows, clean_version, clean_removed, clean_reasons)
    compact_profile = write_dataset(compact, compact_version, compact_removed,
                                    {"verified_same_label_near_duplicate": int(len(compact_removed))},
                                    parent_version=clean_version)
    return {
        "source_dataset_path": str(source_path), "source_dataset_sha256": source_hash,
        "clean": clean_profile, "compact": compact_profile,
        "before_stats": before_stats, "clean_stats": clean_stats, "compact_stats": compact_stats,
        "verified_near_duplicate_pairs": int(len(verified_pairs)),
        "minhash_candidate_pair_count": int(len(candidate_similarity)),
        "candidate_similarity": {"character_5gram_jaccard": similarity_summary(candidate_similarity, 0),
                                 "word_3gram_jaccard": similarity_summary(candidate_similarity, 1)},
        "verified_similarity": {"character_5gram_jaccard": similarity_summary(verified_pairs, 2),
                                "word_3gram_jaccard": similarity_summary(verified_pairs, 3)},
        "verified_pairs_same_prompt": int(sum(bool(item[4]) for item in verified_pairs)),
        "near_duplicate_clusters": int(len(cluster_records)),
        "near_duplicate_cluster_sizes": sorted([row["size"] for row in cluster_records]),
        "near_duplicate_clusters_detail": cluster_records,
        "similarity_thresholds": {"character_5gram_jaccard": compact_char_similarity,
                                  "word_3gram_jaccard": compact_word_similarity,
                                  "minhash_permutations": n_perm,
                                  "minhash_candidate_thresholds": {"character": compact_char_similarity,
                                                                   "word": compact_word_similarity},
                                  "seed": seed},
        "template_diagnostic": {"templates_repeated_3plus": int(len(repeated_templates)),
                                 "rows_in_repeated_templates": int(template_sizes[template_sizes.ge(3)].sum()),
                                 "removed_by_template_alone": 0},
        "prompt_redundancy_diagnostic": {"model_prompt_groups_with_multiple_rows": int(len(prompt_repeated)),
                                         "rows_in_repeated_model_prompt_groups": int(prompt_repeated.sum()),
                                         "removed_by_prompt_alone": 0},
        "tfidf_diagnostic": tfidf_report,
        "class_counts_before": {str(k): int(v) for k,v in data.model.value_counts().sort_index().items()},
        "class_counts_clean": clean_profile["class_counts"], "class_counts_compact": compact_profile["class_counts"],
        "before_stats": before_stats,
    }


def _source_concentration(frame: pd.DataFrame) -> dict:
    out = {}
    for label, group in frame.groupby("model", sort=True):
        counts = group.source_dataset.value_counts(normalize=True)
        out[str(label)] = {"largest_source_fraction": float(counts.max()),
                           "top3_source_fraction": float(counts.head(3).sum()),
                           "source_count": int(group.source_dataset.nunique())}
    return out


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
