"""Training-fed trust list for checkpoints trained without V0.

The dashboard reads the materialized CSV and never runs inference or computes
metrics. New rows are appended from the completed training manifest's saved
full-V0 metrics. The source CSV defines the eligible model roster only.
"""
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


ROSTER_PATH = Path("docs/v2-ckpt-nov0.csv")
TRUST_PATH = Path("docs/v2-ckpt-nov0_active.csv")
METADATA_PATH = Path("docs/v2-ckpt-nov0_active.json")
INITIAL_EXPERIMENTS = (
    "ultrabert380_v2.0_valv0_datad12bb8be_split870e47f9_seed2026_cfgfe74ea98_codeab9f6d2",
    "ultrabert380_v2.1_valv0_datac59bc6ca_split6330072d_seed2026_cfgaf9f028b_codeab9f6d2",
)
LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
LOCAL_TIMEZONE = ZoneInfo("Europe/Zurich")
V0_ROWS = 2699
MINIMUM_SUPPORT = 10
V0_TOKEN = re.compile(r"(?<![A-Za-z0-9])v0(?![A-Za-z0-9])", re.IGNORECASE)

TRUST_COLUMNS = [
    "SeqID", "trust_list_version", "trust_list_generated_at_utc", "sequence_id",
    "experiment_id", "model", "model_title", "target_label", "training_dataset",
    "dataset_lineage", "checkpoint", "checkpoint_path", "checkpoint_created_at_utc",
    "checkpoint_date", "precision", "recall", "prediction_count", "true_positive",
    "false_positive", "false_negative", "evaluation_eligible",
    "validation_training_overlap", "evaluation_note", "checkpoint_available",
    "validation_split_sha256", "catalog_source", "smoothed_precision", "rule_eligible",
    "overall_accuracy", "balanced_accuracy", "validation_rows",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _relative_path(value: str | Path, root: Path) -> str:
    path = Path(str(value or ""))
    if not path:
        return ""
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _lineage(manifest: dict, manifest_path: Path) -> str:
    try:
        from dashboard.training_lineage import training_lineage
        value = training_lineage(manifest, manifest_path.parent)
    except (ImportError, OSError, ValueError, TypeError):
        value = None
    return str(value or manifest.get("training_dataset") or manifest.get("dataset") or "")


def _eligible_models(roster_path: Path) -> set[str]:
    if not roster_path.is_file():
        raise FileNotFoundError(f"V2 checkpoint roster is missing: {roster_path}")
    roster = pd.read_csv(roster_path, dtype=str, keep_default_na=False)
    if "model" not in roster:
        raise ValueError(f"V2 checkpoint roster has no model column: {roster_path}")
    return set(roster.model.astype(str))


def _next_checkpoint_id(frame: pd.DataFrame) -> str:
    numbers = [int(m.group(1)) for value in frame.get("sequence_id", pd.Series(dtype=str)).dropna().astype(str)
               if (m := re.fullmatch(r"N(\d+)", value))]
    return f"N{max(numbers, default=0) + 1:04d}"


def _rows_from_training_manifest(manifest: dict, manifest_path: Path, *, sequence_id: str,
                                 root: Path) -> list[dict]:
    """Convert saved full-V0 training metrics into label rows; never infer here."""
    model = str(manifest.get("model") or "")
    training_dataset = str(manifest.get("training_dataset") or manifest.get("dataset") or "")
    lineage = _lineage(manifest, manifest_path)
    if V0_TOKEN.search(training_dataset) or V0_TOKEN.search(lineage):
        return []
    if str(manifest.get("validation_dataset", "")).lower() != "v0":
        return []
    validation_rows = int(manifest.get("validation_rows") or manifest.get("validation_dataset_rows") or 0)
    if validation_rows != V0_ROWS:
        return []

    metrics = manifest.get("validation_metrics") or manifest.get("metrics") or {}
    classes = list(metrics.get("classes") or manifest.get("classes") or [])
    if set(classes) != set(LABELS) or len(classes) != len(LABELS):
        return []
    matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or (matrix < 0).any():
        return []
    class_index = {label: i for i, label in enumerate(classes)}
    precision_values = metrics.get("per_class_precision") or {}
    recall_values = metrics.get("per_class_recall") or {}
    true_total = int(matrix.sum())
    if true_total != V0_ROWS:
        return []

    checkpoint_path = Path(str(manifest.get("checkpoint_path") or manifest.get("checkpoint") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_available = bool(str(checkpoint_path) and checkpoint_path.exists())
    created = (manifest.get("checkpoint_created_at_utc") or manifest.get("created_at_utc")
               or manifest.get("created_at"))
    if not created:
        created = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc).isoformat()
    try:
        created_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        created_dt = created_dt.astimezone(timezone.utc)
    except ValueError:
        created_dt = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc)
    display_date = created_dt.astimezone(LOCAL_TIMEZONE).strftime("%m.%d")
    split_hash = str(manifest.get("validation_ids_sha256") or
                     manifest.get("validation_dataset_sha256") or "")
    overall_accuracy = float(np.trace(matrix) / true_total)
    balanced_accuracy = metrics.get("balanced_accuracy")
    rows = []
    for label in LABELS:
        i = class_index[label]
        tp = int(matrix[i, i])
        predicted = int(matrix[:, i].sum())
        actual = int(matrix[i, :].sum())
        fp, fn = predicted - tp, actual - tp
        precision = precision_values.get(label)
        recall = recall_values.get(label)
        # Existing V2.0/V2.1 manifests predate per-class precision export.
        # Their full-V0 confusion matrices are saved training metrics, so this
        # only reads and serializes those already-computed results.
        if precision is None:
            precision = tp / predicted if predicted else np.nan
        if recall is None:
            recall = tp / actual if actual else np.nan
        rows.append({
            "sequence_id": sequence_id,
            "experiment_id": str(manifest.get("experiment_id") or manifest_path.stem.removesuffix("_manifest")),
            "model": model,
            "model_title": manifest.get("model_title") or model,
            "target_label": label,
            "training_dataset": training_dataset,
            "dataset_lineage": lineage,
            "checkpoint": Path(str(checkpoint_path)).name,
            "checkpoint_path": _relative_path(checkpoint_path, root),
            "checkpoint_created_at_utc": created_dt.isoformat(),
            "checkpoint_date": display_date,
            "precision": float(precision) if pd.notna(precision) else np.nan,
            "recall": float(recall) if pd.notna(recall) else np.nan,
            "prediction_count": predicted,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "evaluation_eligible": True,
            "validation_training_overlap": 0,
            "evaluation_note": "Full V0 (2,699 rows); V0 was used for checkpoint selection.",
            "checkpoint_available": checkpoint_available,
            "validation_split_sha256": split_hash,
            "catalog_source": "full_v0_training_metrics",
            "smoothed_precision": (tp + 1.0) / (predicted + 2.0),
            "rule_eligible": predicted >= MINIMUM_SUPPORT,
            "overall_accuracy": overall_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "validation_rows": validation_rows,
        })
    return rows


def _sort_and_rank(frame: pd.DataFrame, *, generated_at: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=TRUST_COLUMNS)
    frame = frame.copy()
    for column in ("precision", "recall", "prediction_count", "true_positive", "false_positive",
                   "false_negative", "validation_training_overlap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["rule_eligible"] = (frame["evaluation_eligible"].astype(str).str.lower().isin({"true", "1"})
                               & frame["validation_training_overlap"].fillna(0).eq(0)
                               & frame["prediction_count"].fillna(0).ge(MINIMUM_SUPPORT))
    frame["_checkpoint_number"] = frame.sequence_id.astype(str).str.extract(r"N(\d+)", expand=False).astype(int)
    frame = frame.sort_values(["precision", "recall", "_checkpoint_number", "target_label"],
                              ascending=[False, False, True, True], na_position="last", kind="stable")
    frame = frame.drop(columns="_checkpoint_number").reset_index(drop=True)
    frame.insert(0, "SeqID", range(1, len(frame) + 1))
    frame.insert(1, "trust_list_version", "trust-v2")
    frame.insert(2, "trust_list_generated_at_utc", generated_at)
    for column in TRUST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[TRUST_COLUMNS]


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initialize_v2_trust_list(*, root: str | Path = ".",
                             roster_path: str | Path = ROSTER_PATH,
                             output_path: str | Path = TRUST_PATH,
                             metadata_path: str | Path = METADATA_PATH) -> dict:
    """Initialize the separate list from the two explicitly selected V2 runs."""
    root = Path(root).resolve()
    roster_path = Path(roster_path)
    if not roster_path.is_absolute():
        roster_path = root / roster_path
    output_path, metadata_path = Path(output_path), Path(metadata_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    allowed_models = _eligible_models(roster_path)
    rows = []
    for experiment_id in INITIAL_EXPERIMENTS:
        manifest_path = root / "artifacts" / "predictions" / f"{experiment_id}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"required initial V2 run manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if str(manifest.get("model")) not in allowed_models:
            raise ValueError(f"initial V2 checkpoint model is outside the v2-ckpt-nov0 roster: {manifest.get('model')}")
        sequence_id = "N0103" if experiment_id == INITIAL_EXPERIMENTS[0] else "N0104"
        new_rows = _rows_from_training_manifest(manifest, manifest_path, sequence_id=sequence_id, root=root)
        if len(new_rows) != len(LABELS):
            raise ValueError(f"initial run {experiment_id} lacks complete full-V0 training metrics")
        rows.extend(new_rows)
    generated_at = _utc_now()
    frame = _sort_and_rank(pd.DataFrame(rows), generated_at=generated_at)
    _atomic_write(frame, output_path)
    metadata = {
        "version": "trust-v2",
        "generated_at_utc": generated_at,
        "source_roster": str(roster_path.relative_to(root)),
        "initial_runs": list(INITIAL_EXPERIMENTS),
        "rows": int(len(frame)),
        "checkpoints": int(frame.experiment_id.nunique()),
        "validation_scope": "full_v0",
        "validation_rows": V0_ROWS,
        "minimum_support": MINIMUM_SUPPORT,
        "smoothing": "Beta(1,1): (true_positive + 1) / (prediction_count + 2)",
        "ordering": ["precision descending", "recall descending", "checkpoint sequence ascending", "label ascending"],
        "future_updates": "training pipeline appends saved full-V0 metrics; dashboard does not discover runs or score checkpoints",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def append_training_run(manifest: dict, manifest_path: str | Path, *,
                        root: str | Path = ".",
                        roster_path: str | Path = ROSTER_PATH,
                        output_path: str | Path = TRUST_PATH,
                        metadata_path: str | Path = METADATA_PATH) -> dict:
    """Append one completed, eligible run using metrics already saved by training."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    roster_path = Path(roster_path)
    if not roster_path.is_absolute():
        roster_path = root / roster_path
    output_path, metadata_path = Path(output_path), Path(metadata_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    allowed_models = _eligible_models(roster_path)
    if str(manifest.get("model") or "") not in allowed_models:
        return {"status": "skipped", "reason": "model is outside the v2-ckpt-nov0 roster"}
    training_dataset = str(manifest.get("training_dataset") or manifest.get("dataset") or "")
    lineage = _lineage(manifest, manifest_path)
    if V0_TOKEN.search(training_dataset) or V0_TOKEN.search(lineage):
        return {"status": "skipped", "reason": "training lineage includes V0"}
    if not output_path.is_file():
        initialize_v2_trust_list(root=root, roster_path=roster_path,
                                 output_path=output_path, metadata_path=metadata_path)
    current = pd.read_csv(output_path)
    experiment_id = str(manifest.get("experiment_id") or manifest_path.stem.removesuffix("_manifest"))
    if experiment_id in set(current.experiment_id.astype(str)):
        return {"status": "already_recorded", "experiment_id": experiment_id}
    sequence_id = _next_checkpoint_id(current)
    rows = _rows_from_training_manifest(manifest, manifest_path, sequence_id=sequence_id, root=root)
    if len(rows) != len(LABELS):
        return {"status": "skipped", "reason": "complete full-V0 training metrics are unavailable"}
    merged = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    generated_at = _utc_now()
    ranked = _sort_and_rank(merged, generated_at=generated_at)
    _atomic_write(ranked, output_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {"version": "trust-v2"}
    metadata.update({"generated_at_utc": generated_at, "rows": int(len(ranked)),
                     "checkpoints": int(ranked.experiment_id.nunique()),
                     "last_added_experiment_id": experiment_id})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"status": "added", "sequence_id": sequence_id, "labels": len(rows)}


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(initialize_v2_trust_list(root=args.root), indent=2))


if __name__ == "__main__":
    _cli()
