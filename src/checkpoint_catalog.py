"""Persistent, checkpoint-level per-label metrics for the dashboard."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CATALOG_COLUMNS = [
    "sequence_id", "legacy_candidate_key", "experiment_id", "model", "model_title",
    "target_label", "training_dataset", "dataset_lineage", "checkpoint", "checkpoint_path",
    "checkpoint_created_at_utc", "checkpoint_date", "precision", "recall",
    "prediction_count", "true_positive", "false_positive", "false_negative",
    "evaluation_eligible", "validation_training_overlap", "evaluation_note",
    "checkpoint_available", "validation_split_sha256", "catalog_source",
]
LOCAL_TIMEZONE = ZoneInfo("Europe/Zurich")
SNAPSHOT_INVENTORY = Path("artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv")
SNAPSHOT_RULES = Path("artifacts/analysis/zod01_specialist_discovery/ranked_checkpoint_label_rules.csv")
DEFAULT_CATALOG = Path("docs/checkpoint_label_catalog.csv")
DEFAULT_TRUST_LIST = Path("docs/checkpoint_label_trust_list.csv")
DEFAULT_TRUST_METADATA = Path("docs/checkpoint_label_trust_list.json")
DEFAULT_ARTIFACT_ROOT = Path("artifacts")
NEW_RUNS_START_UTC = "2026-11-08T00:00:00+00:00"
TRUST_LIST_ORDERING = ["precision descending", "recall descending",
                       "checkpoint sequence ascending", "label ascending"]


def _resolve(value, root: Path) -> Path:
    path = Path(value) if value else Path()
    return path if path.is_absolute() else root / path


def _timestamp(manifest: dict, manifest_path: Path) -> datetime:
    value = manifest.get("created_at_utc") or manifest.get("created_at")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc)


def _display_date(value: datetime) -> str:
    return value.astimezone(LOCAL_TIMEZONE).strftime("%m.%d")


def _dataset_lineage(manifest: dict, manifest_path: Path) -> str:
    try:
        from dashboard.training_lineage import training_lineage
        value = training_lineage(manifest, manifest_path.parent)
    except (ImportError, OSError, ValueError, TypeError):
        value = None
    if value:
        return value
    return str(manifest.get("training_dataset") or manifest.get("dataset") or "")


def _checkpoint_basename(value) -> str:
    return Path(str(value or "")).name


def _portable_path(value, root: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _sequence_id(number: int) -> str:
    if number < 1 or number > 999_999_999:
        raise ValueError("checkpoint sequence must be in 1..999999999")
    return f"N{number:04d}"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.reindex(columns=CATALOG_COLUMNS).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_catalog(path=DEFAULT_CATALOG) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    frame = pd.read_csv(path)
    for column in CATALOG_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[CATALOG_COLUMNS]
    if frame.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError(f"duplicate checkpoint-label rows in {path}")
    return frame


def refresh_trust_list(*, catalog_path=DEFAULT_CATALOG, trust_list_path=None,
                       metadata_path=None, minimum_support=10) -> dict:
    """Write a versioned, complete checkpoint-label ranking from the catalog.

    Every catalog row receives a stable numeric SeqID. Rules meeting the
    frozen validation eligibility and support requirements sort first; other
    rows remain in the reference list after them and are marked ineligible.
    """
    import hashlib

    catalog_path = Path(catalog_path)
    if trust_list_path is None:
        trust_list_path = (DEFAULT_TRUST_LIST if catalog_path.resolve() == DEFAULT_CATALOG.resolve()
                           else catalog_path.with_name(f"{catalog_path.stem}_trust_list.csv"))
    if metadata_path is None:
        metadata_path = (DEFAULT_TRUST_METADATA if catalog_path.resolve() == DEFAULT_CATALOG.resolve()
                         else Path(trust_list_path).with_suffix(".json"))
    trust_list_path, metadata_path = Path(trust_list_path), Path(metadata_path)

    source = load_catalog(catalog_path)
    if source.empty:
        raise ValueError(f"cannot create a trust list from an empty checkpoint catalog: {catalog_path}")
    source_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    prior = {}
    if metadata_path.is_file():
        try:
            prior = json.loads(metadata_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            prior = {}
    if (prior.get("source_catalog_sha256") == source_hash and trust_list_path.is_file()
            and int(prior.get("rows", -1)) == len(source)
            and prior.get("ordering") == TRUST_LIST_ORDERING
            and int(prior.get("minimum_support", -1)) == int(minimum_support)):
        return {"status": "unchanged", "version": prior.get("version"),
                "generated_at_utc": prior.get("generated_at_utc"), "rows": len(source),
                "trust_list_path": str(trust_list_path)}

    numeric = ["precision", "recall", "prediction_count", "true_positive", "false_positive",
               "false_negative", "validation_training_overlap"]
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["smoothed_precision"] = (source.true_positive.fillna(0) + 1.0) / (source.prediction_count.fillna(0) + 2.0)
    evaluation_ok = source.evaluation_eligible.astype(str).str.lower().isin({"true", "1"})
    no_overlap = source.validation_training_overlap.fillna(0).eq(0)
    support_ok = source.prediction_count.fillna(0).ge(int(minimum_support))
    source["rule_eligible"] = evaluation_ok & no_overlap & support_ok
    source["SeqID_checkpoint_order"] = source.sequence_id.astype(str).str.extract(r"N(\d+)", expand=False).astype(float)

    source = source.sort_values(
        ["precision", "recall", "SeqID_checkpoint_order", "target_label"],
        ascending=[False, False, True, True],
        na_position="last", kind="stable").reset_index(drop=True)
    source.insert(0, "SeqID", range(1, len(source) + 1))
    source = source.drop(columns="SeqID_checkpoint_order")

    old_number = 0
    old_version = str(prior.get("version", ""))
    match = re.fullmatch(r"trust-v(\d+)", old_version)
    if match:
        old_number = int(match.group(1))
    version = f"trust-v{old_number + 1:03d}"
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source.insert(1, "trust_list_version", version)
    source.insert(2, "trust_list_generated_at_utc", generated_at)
    trust_list_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(trust_list_path, index=False)
    metadata = {
        "version": version,
        "generated_at_utc": generated_at,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": source_hash,
        "rows": int(len(source)),
        "checkpoints": int(source.experiment_id.nunique()),
        "eligible_rules": int(source.rule_eligible.sum()),
        "minimum_support": int(minimum_support),
        "smoothing": "Beta(1,1): (true_positive + 1) / (prediction_count + 2)",
        "ordering": TRUST_LIST_ORDERING,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"status": "generated", "version": version, "generated_at_utc": generated_at,
            "rows": len(source), "eligible_rules": int(source.rule_eligible.sum()),
            "trust_list_path": str(trust_list_path), "metadata_path": str(metadata_path)}


def sort_catalog_for_dashboard(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic precision-first view without changing catalog data."""
    required = {"precision", "recall", "sequence_id", "target_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"catalog missing display fields: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["precision"] = pd.to_numeric(result.precision, errors="coerce")
    result["recall"] = pd.to_numeric(result.recall, errors="coerce")
    return result.sort_values(["precision", "recall", "sequence_id", "target_label"],
                              ascending=[False, False, True, True], na_position="last",
                              kind="stable").reset_index(drop=True)


def _next_sequence(frame: pd.DataFrame) -> int:
    numbers = [int(match.group(1)) for value in frame.sequence_id.dropna().astype(str)
               if (match := re.fullmatch(r"N(\d+)", value))]
    return max(numbers, default=0) + 1


def _manifest_fields(manifest: dict, manifest_path: Path, *, sequence_id: str,
                     target_label: str, precision, recall, prediction_count,
                     true_positive, false_positive, false_negative,
                     evaluation_eligible=True, validation_training_overlap=0,
                     evaluation_note="", catalog_source="v1_all_training_refresh",
                     validation_split_sha256="") -> dict:
    created = _timestamp(manifest, manifest_path)
    raw_checkpoint_path = str(manifest.get("checkpoint_path") or manifest.get("checkpoint") or "")
    checkpoint_path = _portable_path(raw_checkpoint_path, Path.cwd())
    model = str(manifest.get("model") or "unknown")
    dataset = str(manifest.get("training_dataset") or manifest.get("dataset") or "unknown")
    experiment_id = str(manifest.get("experiment_id") or manifest_path.stem.removesuffix("_manifest"))
    available = bool(checkpoint_path and _resolve(checkpoint_path, Path.cwd()).exists())
    return {
        "sequence_id": sequence_id,
        "legacy_candidate_key": "",
        "experiment_id": experiment_id,
        "model": model,
        "model_title": manifest.get("model_title") or model,
        "target_label": str(target_label),
        "training_dataset": dataset,
        "dataset_lineage": _dataset_lineage(manifest, manifest_path),
        "checkpoint": manifest.get("checkpoint_filename") or _checkpoint_basename(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_created_at_utc": created.isoformat(),
        "checkpoint_date": _display_date(created),
        "precision": precision,
        "recall": recall,
        "prediction_count": int(prediction_count),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "evaluation_eligible": bool(evaluation_eligible),
        "validation_training_overlap": int(validation_training_overlap or 0),
        "evaluation_note": evaluation_note or "",
        "checkpoint_available": available,
        "validation_split_sha256": validation_split_sha256,
        "catalog_source": catalog_source,
    }


def seed_from_specialist_snapshot(*, catalog_path=DEFAULT_CATALOG,
                                  inventory_path=SNAPSHOT_INVENTORY,
                                  rules_path=SNAPSHOT_RULES) -> pd.DataFrame:
    """Seed an empty catalog from the saved 89-checkpoint analysis snapshot."""
    catalog_path, inventory_path, rules_path = map(Path, (catalog_path, inventory_path, rules_path))
    current = load_catalog(catalog_path)
    if not current.empty:
        return current
    if not inventory_path.is_file() or not rules_path.is_file():
        raise FileNotFoundError("the existing specialist-discovery inventory and ranked rules are required")

    inventory = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    rules = pd.read_csv(rules_path, dtype=str, keep_default_na=False)
    inventory_by_key = {str(row.candidate_key): row for row in inventory.itertuples(index=False)}
    missing = set(rules.candidate_key.astype(str)) - set(inventory_by_key)
    if missing:
        raise ValueError(f"snapshot rules lack checkpoint inventory rows: {sorted(missing)}")

    existing_numbers = [int(match.group(1)) for value in inventory.candidate_key.astype(str)
                        if (match := re.fullmatch(r"N(\d+)", value))]
    next_number = max(existing_numbers, default=0) + 1
    legacy_ids = {}
    for legacy in ("A", "B", "C", "D"):
        if legacy in inventory_by_key:
            legacy_ids[legacy] = _sequence_id(next_number)
            next_number += 1

    rows = []
    for rule in rules.to_dict("records"):
        key = str(rule["candidate_key"])
        inv = inventory_by_key[key]._asdict()
        manifest_path = Path(str(inv.get("manifest_path") or ""))
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        manifest = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                manifest = {}
        experiment_id = str(inv.get("experiment_id") or rule.get("candidate_id") or key)
        manifest.setdefault("experiment_id", experiment_id)
        manifest.setdefault("model", inv.get("model") or rule.get("model"))
        manifest.setdefault("dataset", inv.get("training_dataset") or rule.get("dataset"))
        manifest.setdefault("checkpoint_path", inv.get("checkpoint_path", ""))
        created = _timestamp(manifest, manifest_path if manifest_path.exists() else inventory_path)
        manifest["created_at_utc"] = created.isoformat()
        overlap = int(float(inv.get("validation_training_overlap") or 0))
        available = str(inv.get("checkpoint_available", "")).lower() in {"true", "1"}
        entry = _manifest_fields(
            manifest, manifest_path if manifest_path.exists() else inventory_path,
            sequence_id=(key if re.fullmatch(r"N\d+", key) else legacy_ids.get(key, _sequence_id(next_number))),
            target_label=rule["label"], precision=rule.get("precision", ""),
            recall=rule.get("recall", ""), prediction_count=rule.get("prediction_count", 0),
            true_positive=rule.get("true_positive", 0), false_positive=rule.get("false_positive", 0),
            false_negative=rule.get("false_negative", 0),
            evaluation_eligible=str(rule.get("evaluation_eligible", "")).lower() in {"true", "1"},
            validation_training_overlap=overlap,
            evaluation_note=rule.get("evaluation_exclusion_reason", ""),
            catalog_source="zod01_specialist_snapshot",
            validation_split_sha256=str(manifest.get("split_sha256", "")))
        entry["legacy_candidate_key"] = key if key in {"A", "B", "C", "D"} else ""
        entry["model_title"] = str(inv.get("model_title") or rule.get("model_title") or entry["model"])
        entry["training_dataset"] = str(inv.get("training_dataset") or rule.get("dataset") or entry["training_dataset"])
        entry["checkpoint"] = str(inv.get("checkpoint_basename") or entry["checkpoint"])
        entry["checkpoint_path"] = _portable_path(
            inv.get("checkpoint_path") or entry["checkpoint_path"], Path.cwd())
        entry["checkpoint_available"] = available
        rows.append(entry)
    seeded = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    if seeded.empty or seeded.experiment_id.nunique() != len(inventory):
        raise ValueError("snapshot seeding did not produce one sequence ID per inventoried checkpoint")
    _atomic_csv(seeded, catalog_path)
    return seeded


def _v0_reference(data_dir: Path):
    from .pipeline import _load_dataset
    rows, _ = _load_dataset("v0", str(data_dir))
    validation = rows.loc[rows.split.astype(str).eq("validation"), ["id", "prompt_id", "model"]].copy()
    validation["id"] = validation.id.astype(str)
    validation["prompt_id"] = validation.prompt_id.astype(str)
    validation["model"] = validation.model.astype(str)
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    if len(validation) != 540 or validation.prompt_id.nunique() != 30:
        raise ValueError("checkpoint catalog requires the frozen V0 validation set (540 rows, 30 prompts)")
    profile_path = Path(data_dir) / "dataset_profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
    return validation, sorted(validation.model.unique().tolist()), str(profile.get("split_sha256", ""))


def _align_v0_predictions(frame: pd.DataFrame, validation: pd.DataFrame,
                          experiment_id: str) -> pd.DataFrame:
    """Return prediction rows in frozen V0 order after strict identity checks."""
    required = {"id", "prompt_id", "y_true"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{experiment_id}: prediction cache is missing {sorted(required - set(frame.columns))}")
    aligned = frame.copy()
    aligned["id"] = aligned.id.astype(str)
    aligned["prompt_id"] = aligned.prompt_id.astype(str)
    aligned["y_true"] = aligned.y_true.astype(str)
    if aligned.id.duplicated().any() or set(aligned.id) != set(validation.id):
        raise ValueError(f"{experiment_id}: prediction IDs do not exactly match frozen V0 validation")
    aligned = aligned.set_index("id").loc[validation.id].reset_index()
    if not np.array_equal(aligned.prompt_id.to_numpy(), validation.prompt_id.to_numpy()):
        raise ValueError(f"{experiment_id}: prompt IDs do not match frozen V0 validation")
    if not np.array_equal(aligned.y_true.to_numpy(), validation.model.to_numpy()):
        raise ValueError(f"{experiment_id}: labels do not match frozen V0 validation")
    return aligned


def append_manifest(*, manifest: dict, manifest_path, catalog_path=DEFAULT_CATALOG,
                    data_dir="data/processed") -> dict:
    """Append 18 per-label records for a completed V1-all run with V0 predictions."""
    dataset = str(manifest.get("training_dataset") or manifest.get("dataset") or "").lower().replace("-", "_")
    if dataset != "v1_all":
        return {"status": "skipped", "reason": "training dataset is not v1_all"}
    catalog_path, manifest_path = Path(catalog_path), Path(manifest_path)
    current = load_catalog(catalog_path)
    if current.empty:
        current = seed_from_specialist_snapshot(catalog_path=catalog_path)
    experiment_id = str(manifest.get("experiment_id") or manifest_path.stem.removesuffix("_manifest"))
    if experiment_id in set(current.experiment_id.astype(str)):
        return {"status": "already_recorded", "sequence_id": current.loc[current.experiment_id.eq(experiment_id), "sequence_id"].iloc[0]}

    root = Path.cwd()
    checkpoint_path = _resolve(manifest.get("checkpoint_path") or manifest.get("checkpoint"), root)
    prediction_path = _resolve(manifest.get("prediction_path"), root)
    if not prediction_path.is_file():
        return {"status": "skipped", "reason": f"prediction file unavailable: {prediction_path}"}
    frame = pd.read_parquet(prediction_path)
    validation, classes, split_hash = _v0_reference(Path(data_dir))
    ordered = _align_v0_predictions(frame, validation, experiment_id)
    probability_columns = [f"prob::{label}" for label in classes]
    if not set(probability_columns).issubset(ordered.columns):
        raise ValueError(f"{experiment_id}: probability columns do not match the canonical 18 labels")
    probabilities = ordered[probability_columns].to_numpy(dtype=np.float64)
    if (probabilities.shape != (540, 18) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError(f"{experiment_id}: saved probability matrix is invalid")
    predicted = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
    y_true = validation.model.to_numpy(dtype=object)
    sequence_id = _sequence_id(_next_sequence(current))
    new_rows = []
    for label in classes:
        pred_mask = predicted == label
        actual_mask = y_true == label
        tp = int(np.sum(pred_mask & actual_mask))
        fp = int(np.sum(pred_mask & ~actual_mask))
        fn = int(np.sum(~pred_mask & actual_mask))
        pred_count = int(pred_mask.sum())
        actual_count = int(actual_mask.sum())
        row = _manifest_fields(manifest, manifest_path, sequence_id=sequence_id,
            target_label=label, precision=(tp / pred_count if pred_count else np.nan),
            recall=(tp / actual_count if actual_count else np.nan), prediction_count=pred_count,
            true_positive=tp, false_positive=fp, false_negative=fn,
            validation_training_overlap=0, evaluation_note="",
            validation_split_sha256=split_hash)
        row["experiment_id"] = experiment_id
        row["checkpoint_path"] = _portable_path(checkpoint_path, root)
        row["checkpoint_available"] = checkpoint_path.exists()
        new_rows.append(row)
    merged = pd.concat([current, pd.DataFrame(new_rows, columns=CATALOG_COLUMNS)], ignore_index=True)
    _atomic_csv(merged, catalog_path)
    return {"status": "added", "sequence_id": sequence_id, "labels": len(new_rows)}


def refresh_catalog(*, artifact_root=DEFAULT_ARTIFACT_ROOT, data_dir="data/processed",
                    catalog_path=DEFAULT_CATALOG, since_utc=NEW_RUNS_START_UTC,
                    trust_list_path=None, trust_metadata_path=None) -> dict:
    """Seed the historical snapshot, then append eligible newer V1-all checkpoints."""
    catalog_path = Path(catalog_path)
    current = load_catalog(catalog_path)
    if current.empty:
        current = seed_from_specialist_snapshot(catalog_path=catalog_path)
    since = datetime.fromisoformat(str(since_utc).replace("Z", "+00:00")).astimezone(timezone.utc)
    root = Path(artifact_root)
    manifests = []
    for path in root.rglob("*_manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        dataset = str(manifest.get("training_dataset") or manifest.get("dataset") or "").lower().replace("-", "_")
        if dataset != "v1_all":
            continue
        if _timestamp(manifest, path) < since:
            continue
        if not manifest.get("prediction_path") or not (manifest.get("checkpoint_path") or manifest.get("checkpoint")):
            continue
        manifests.append((path, manifest))
    additions = []
    for path, manifest in sorted(manifests, key=lambda pair: (_timestamp(pair[1], pair[0]), str(pair[1].get("experiment_id", "")))):
        result = append_manifest(manifest=manifest, manifest_path=path,
                                 catalog_path=catalog_path, data_dir=data_dir)
        if result.get("status") == "added":
            additions.append({"experiment_id": manifest.get("experiment_id"), **result})
    final = load_catalog(catalog_path)
    trust_update = refresh_trust_list(catalog_path=catalog_path,
        trust_list_path=trust_list_path, metadata_path=trust_metadata_path)
    return {"status": "refreshed", "checkpoint_count": int(final.experiment_id.nunique()),
            "label_rows": len(final), "added": additions, "catalog_path": str(catalog_path),
            "trust_list": trust_update}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh the persistent per-label checkpoint catalog")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--since-utc", default=NEW_RUNS_START_UTC)
    parser.add_argument("--trust-list", default=None)
    parser.add_argument("--trust-metadata", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(refresh_catalog(artifact_root=args.artifact_root, data_dir=args.data_dir,
        catalog_path=args.catalog, since_utc=args.since_utc,
        trust_list_path=args.trust_list, trust_metadata_path=args.trust_metadata), indent=2))


if __name__ == "__main__":
    main()
