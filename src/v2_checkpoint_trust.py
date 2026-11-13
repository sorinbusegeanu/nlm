"""Versioned V0 trust lists for the eligible external-only checkpoint roster.

The initial version scores the frozen roster once. Later versions are immutable
snapshots and are updated from completed training manifests by the pipeline.
"""
from __future__ import annotations

import argparse
import gc
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
VERSION_PREFIX = "trust-list-2-v"
LATEST_PATH = Path("docs/trust-list-2-latest.json")
PREDICTION_DIR = Path("artifacts/analysis/trust_list_2_v0")
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
    frame = frame.drop(columns=[column for column in
                                ("SeqID", "trust_list_version", "trust_list_generated_at_utc")
                                if column in frame], errors="ignore")
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
    frame.insert(1, "trust_list_version", str(frame.attrs.get("trust_list_version", "trust-v2")))
    frame.insert(2, "trust_list_generated_at_utc", generated_at)
    for column in TRUST_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[TRUST_COLUMNS]


def _version_number(version: str) -> int:
    match = re.fullmatch(r"trust-list-2-v(\d{3,})", str(version))
    if not match:
        raise ValueError(f"invalid trust list version: {version}")
    return int(match.group(1))


def _latest_version(root: Path) -> tuple[str, Path, Path] | None:
    pointer = root / LATEST_PATH
    if not pointer.is_file():
        return None
    value = json.loads(pointer.read_text())
    version = str(value["version"])
    _version_number(version)
    return version, root / value["csv_path"], root / value["metadata_path"]


def latest_trust_paths(root: str | Path = ".") -> tuple[Path, Path, dict] | None:
    """Resolve the current immutable CSV/metadata pair for dashboard use."""
    root = Path(root).resolve()
    latest = _latest_version(root)
    if latest is None:
        return None
    version, csv_path, metadata_path = latest
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"latest trust list {version} is incomplete")
    return csv_path, metadata_path, json.loads(metadata_path.read_text())


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish_version(root: Path, frame: pd.DataFrame, metadata: dict, version: str) -> dict:
    csv_path = root / "docs" / f"{version}.csv"
    metadata_path = root / "docs" / f"{version}.json"
    frame = frame.copy()
    frame.attrs["trust_list_version"] = version
    frame = _sort_and_rank(frame, generated_at=metadata["generated_at_utc"])
    # Publish data first and move the latest pointer last. Readers never see a
    # pointer to a partially written version.
    _atomic_write(frame, csv_path)
    metadata.update({"version": version, "csv_path": str(csv_path.relative_to(root)),
                     "metadata_path": str(metadata_path.relative_to(root)),
                     "rows": int(len(frame)),
                     "checkpoints": int(frame.experiment_id.nunique())})
    _atomic_json(metadata, metadata_path)
    _atomic_json({"version": version, "csv_path": metadata["csv_path"],
                  "metadata_path": metadata["metadata_path"],
                  "generated_at_utc": metadata["generated_at_utc"]}, root / LATEST_PATH)
    return metadata


def _file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        raise FileNotFoundError(path)
    return digest.hexdigest()


def _find_manifest(root: Path, experiment_id: str) -> Path:
    direct = root / "artifacts" / "predictions" / f"{experiment_id}_manifest.json"
    if direct.is_file():
        return direct
    matches = []
    for path in (root / "artifacts" / "runs").glob("**/*manifest.json"):
        try:
            if json.loads(path.read_text()).get("experiment_id") == experiment_id:
                matches.append(path)
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one manifest for {experiment_id}; found {len(matches)}")
    return matches[0]


def _predict_checkpoint(manifest: dict, checkpoint: Path, texts: list[str], root: Path,
                        batch_size: int = 4):
    from .inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
    model = str(manifest.get("model") or "")
    config = manifest.get("config") or {}
    classes = list(manifest.get("classes") or [])
    temperature = float(manifest.get("calibration_temperature") or 1.0)
    if model == "svc":
        import joblib
        probabilities, predicted_classes = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model in {"catboost_stylometry", "catboost"}:
        probabilities, predicted_classes = predict_catboost_stylometry(
            checkpoint, texts, classes, temperature=temperature)
    else:
        probabilities, predicted_classes = _predict_neural(
            model, checkpoint, texts, classes,
            batch_size=max(1, int(config.get("inference_batch_size", batch_size))),
            max_length=int(config.get("max_length", 384)), temperature=temperature,
            base_checkpoint=config.get("checkpoint"), revision=manifest.get("model_revision") or config.get("revision"),
            fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
            text_only_backbone=bool(config.get("text_only_backbone", False)),
            local_base_model=config.get("local_base_model"), quantization=manifest.get("quantization"),
            attention_backend=manifest.get("attention_backend"))
    if set(predicted_classes) != set(LABELS) or len(predicted_classes) != len(LABELS):
        raise ValueError(f"{model} checkpoint does not expose the 18 expected labels")
    order = [list(predicted_classes).index(label) for label in LABELS]
    probabilities = np.asarray(probabilities, dtype=float)[:, order]
    if probabilities.shape != (len(texts), len(LABELS)) or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid probability output from {model}")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError(f"probabilities do not sum to one for {model}")
    return probabilities


def _score_rows(ids, y_true, probabilities, *, manifest, manifest_path: Path,
                checkpoint: Path, checkpoint_sha: str, sequence_id: str, root: Path,
                source: str) -> list[dict]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    matrix = confusion_matrix(y_true, predictions, labels=LABELS)
    recalls = {label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0
               for i, label in enumerate(LABELS)}
    precisions = {label: float(matrix[i, i] / matrix[:, i].sum()) if matrix[:, i].sum() else 0.0
                  for i, label in enumerate(LABELS)}
    balanced_accuracy = float(balanced_accuracy_score(y_true, predictions))
    created = manifest.get("checkpoint_created_at_utc") or manifest.get("created_at_utc") or _utc_now()
    created_dt = pd.Timestamp(created)
    created_dt = created_dt.tz_localize("UTC") if created_dt.tzinfo is None else created_dt.tz_convert("UTC")
    rows = []
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index]); predicted = int(matrix[:, index].sum())
        actual = int(matrix[index, :].sum())
        rows.append({
            "sequence_id": sequence_id,
            "experiment_id": str(manifest.get("experiment_id") or manifest_path.stem.removesuffix("_manifest")),
            "model": str(manifest.get("model") or ""), "model_title": manifest.get("model_title") or manifest.get("model") or "",
            "target_label": label,
            "training_dataset": str(manifest.get("training_dataset") or manifest.get("dataset") or ""),
            "dataset_lineage": _lineage(manifest, manifest_path), "checkpoint": checkpoint.name,
            "checkpoint_path": _relative_path(checkpoint, root),
            "checkpoint_created_at_utc": created,
            "checkpoint_date": created_dt.tz_convert(LOCAL_TIMEZONE).strftime("%m.%d"),
            "precision": precisions[label],
            "recall": recalls[label], "prediction_count": predicted,
            "true_positive": tp, "false_positive": predicted-tp, "false_negative": actual-tp,
            "evaluation_eligible": True, "validation_training_overlap": 0,
            "evaluation_note": "Full V0 (2,699 rows); inference on frozen official V0 rows.",
            "checkpoint_available": checkpoint.exists(), "validation_split_sha256": "",
            "catalog_source": source, "smoothed_precision": (tp+1)/(predicted+2),
            "rule_eligible": predicted >= MINIMUM_SUPPORT,
            "overall_accuracy": float(np.mean(predictions == np.asarray(y_true))),
            "balanced_accuracy": balanced_accuracy, "validation_rows": len(ids),
        })
    return rows


def build_roster_trust_list(*, root: str | Path = ".", batch_size: int = 4) -> dict:
    """Run and cache predictions for every unique checkpoint in the roster."""
    root = Path(root).resolve()
    roster_path = root / ROSTER_PATH
    roster = pd.read_csv(roster_path, dtype=str, keep_default_na=False).drop_duplicates("experiment_id")
    if len(roster) != 63:
        raise ValueError(f"expected 63 unique roster checkpoints, found {len(roster)}")
    from .pipeline import _dataset_rows, _partition_sha256
    v0 = _dataset_rows("v0", root / "data" / "processed")
    if len(v0) != V0_ROWS or set(v0.model.astype(str)) != set(LABELS):
        raise ValueError("V0 reference is not the expected complete 2,699-row, 18-label dataset")
    v0_sha = _partition_sha256(v0)
    ids, texts, y_true = v0.id.astype(str).tolist(), v0.text.astype(str).tolist(), v0.model.astype(str).tolist()
    output_dir = root / PREDICTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows, run_info = [], []
    for position, record in enumerate(roster.to_dict("records"), start=1):
        experiment_id = record["experiment_id"]
        manifest_path = _find_manifest(root, experiment_id)
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(manifest.get("checkpoint_path") or record.get("checkpoint_path") or "")
        if not checkpoint.is_absolute(): checkpoint = root / checkpoint
        if not checkpoint.exists(): raise FileNotFoundError(f"checkpoint missing for {experiment_id}: {checkpoint}")
        checkpoint_sha = _file_sha256(checkpoint)
        cache_path = output_dir / f"{experiment_id}.parquet"
        cache_manifest = output_dir / f"{experiment_id}.json"
        cache_key = {"experiment_id": experiment_id, "checkpoint_sha256": checkpoint_sha,
                     "v0_sha256": v0_sha, "rows": len(v0), "labels": list(LABELS),
                     "model": manifest.get("model"), "config": manifest.get("config"),
                     "calibration_temperature": manifest.get("calibration_temperature", 1.0)}
        if cache_path.is_file() and cache_manifest.is_file():
            cached_meta = json.loads(cache_manifest.read_text())
        else: cached_meta = {}
        if cached_meta.get("cache_key") == cache_key:
            pred_frame = pd.read_parquet(cache_path)
            cache_aligned = (pred_frame.id.astype(str).tolist() == ids and
                             all(f"prob::{label}" in pred_frame for label in LABELS))
        else:
            pred_frame, cache_aligned = None, False
        if cache_aligned:
            probabilities = pred_frame[[f"prob::{label}" for label in LABELS]].to_numpy(dtype=float)
            source = "roster_cached_full_v0_predictions"
        else:
            try:
                probabilities = _predict_checkpoint(manifest, checkpoint, texts, root, batch_size=batch_size)
            except Exception as exc:
                progress = {"completed": position - 1, "total": len(roster), "failed": experiment_id,
                            "error": f"{type(exc).__name__}: {exc}", "updated_at_utc": _utc_now(),
                            "runs": run_info}
                _atomic_json(progress, output_dir / "progress.json")
                raise RuntimeError(f"full-V0 prediction failed for {experiment_id} ({manifest.get('model')}): {exc}") from exc
            columns = {"id": ids, "y_true": y_true,
                       "y_pred": np.asarray(LABELS)[probabilities.argmax(axis=1)]}
            columns.update({f"prob::{label}": probabilities[:, i] for i, label in enumerate(LABELS)})
            pd.DataFrame(columns).to_parquet(cache_path, index=False)
            cache_manifest.write_text(json.dumps({"cache_key": cache_key, "prediction_path": str(cache_path.relative_to(root)),
                                                  "checkpoint_sha256": checkpoint_sha, "v0_sha256": v0_sha,
                                                  "rows": len(v0)}, indent=2)+"\n")
            source = "roster_inference_full_v0"
        sequence_id = str(record.get("sequence_id") or "")
        prediction_rows.extend(_score_rows(ids, y_true, probabilities, manifest=manifest,
                                           manifest_path=manifest_path, checkpoint=checkpoint,
                                           checkpoint_sha=checkpoint_sha, sequence_id=sequence_id,
                                           root=root, source=source))
        # HF/PEFT modules can form reference cycles. Collect them and release
        # unused allocator blocks before the next checkpoint load.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        from sklearn.metrics import balanced_accuracy_score
        run_info.append({"experiment_id": experiment_id, "sequence_id": sequence_id,
                         "checkpoint_sha256": checkpoint_sha, "prediction_path": str(cache_path.relative_to(root)),
                         "balanced_accuracy": float(balanced_accuracy_score(
                             y_true, np.asarray(LABELS)[probabilities.argmax(1)]))})
        progress = {"completed": position, "total": len(roster), "current": experiment_id,
                    "updated_at_utc": _utc_now(), "runs": run_info}
        _atomic_json(progress, output_dir / "progress.json")

    # Preserve the two already evaluated V2 dataset checkpoints in the initial list.
    legacy_path = root / TRUST_PATH
    if not legacy_path.is_file():
        raise FileNotFoundError(f"existing V2.0/V2.1 trust rows missing: {legacy_path}")
    legacy = pd.read_csv(legacy_path)
    initial_rows = pd.DataFrame(prediction_rows)
    initial_rows = initial_rows[~initial_rows.experiment_id.isin(INITIAL_EXPERIMENTS)]
    existing_rows = legacy[legacy.experiment_id.isin(INITIAL_EXPERIMENTS)].copy()
    combined = pd.concat([initial_rows, existing_rows], ignore_index=True, sort=False)
    if combined.experiment_id.nunique() != 65 or len(combined) != 65 * len(LABELS):
        raise ValueError(f"initial trust list must contain 65 complete checkpoints; got {combined.experiment_id.nunique()} checkpoints / {len(combined)} rows")
    generated_at = _utc_now()
    combined.attrs["trust_list_version"] = "trust-list-2-v001"
    ranked = _sort_and_rank(combined, generated_at=generated_at)
    metadata = {"generated_at_utc": generated_at, "source_roster": str(ROSTER_PATH),
                "initial_runs": list(INITIAL_EXPERIMENTS), "roster_checkpoints": len(roster),
                "validation_scope": "full_v0", "validation_rows": len(v0), "validation_sha256": v0_sha,
                "inference_runs": run_info, "method": "one-time inference of unique roster checkpoints; V2.0/V2.1 rows retained from saved full-V0 metrics"}
    _publish_version(root, ranked, metadata, "trust-list-2-v001")
    return {"status": "published", "version": "trust-list-2-v001", "checkpoints": 65,
            "rows": len(ranked), "roster_predictions": len(roster), "prediction_dir": str(output_dir)}


def _read_saved_probabilities(root: Path, experiment_id: str, expected_ids: list[str],
                              expected_y: list[str]) -> tuple[np.ndarray, Path]:
    """Read aligned probabilities from a cache or the run's saved V0 prediction file."""
    cache_path = root / PREDICTION_DIR / f"{experiment_id}.parquet"
    candidates = [cache_path]
    try:
        manifest_path = _find_manifest(root, experiment_id)
        manifest = json.loads(manifest_path.read_text())
        saved_path = Path(str(manifest.get("prediction_path") or ""))
        if str(saved_path):
            candidates.append(saved_path if saved_path.is_absolute() else root / saved_path)
    except FileNotFoundError:
        manifest = {}
    for path in candidates:
        if not path.is_file():
            continue
        predictions = pd.read_parquet(path)
        required = {"id", *(f"prob::{label}" for label in LABELS)}
        if required - set(predictions.columns):
            continue
        if predictions.id.astype(str).tolist() != expected_ids:
            raise ValueError(f"saved probability rows are not aligned to complete V0 for {experiment_id}: {path}")
        if "y_true" in predictions and predictions.y_true.astype(str).tolist() != expected_y:
            raise ValueError(f"saved probability labels disagree with complete V0 for {experiment_id}: {path}")
        probabilities = predictions[[f"prob::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
        if probabilities.shape != (V0_ROWS, len(LABELS)) or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid saved probability matrix for {experiment_id}: {path}")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"saved probabilities are not normalized for {experiment_id}: {path}")
        return probabilities, path
    raise FileNotFoundError(f"no saved full-V0 probability artifact found for {experiment_id}")


def _rank_k_metrics(y_true: list[str], probabilities: np.ndarray, rank: int) -> dict[str, dict]:
    from sklearn.metrics import balanced_accuracy_score
    order = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = np.asarray(LABELS, dtype=object)[order[:, rank - 1]]
    y_array = np.asarray(y_true, dtype=object)
    metrics = {}
    for label in LABELS:
        mask = predicted == label
        correct = mask & (y_array == label)
        prediction_count = int(mask.sum())
        true_positive = int(correct.sum())
        actual_count = int((y_array == label).sum())
        metrics[label] = {
            "precision": true_positive / prediction_count if prediction_count else 0.0,
            "recall": true_positive / actual_count if actual_count else 0.0,
            "prediction_count": prediction_count,
            "true_positive": true_positive,
            "false_positive": prediction_count - true_positive,
            "false_negative": actual_count - true_positive,
        }
    metrics["_overall"] = {
        "overall_accuracy": float(np.mean(predicted == y_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, predicted)),
        "correct_predictions": int(np.sum(predicted == y_array)),
    }
    return metrics


def generate_topk_trust_lists(*, root: str | Path = ".") -> dict:
    """Generate rank-2/rank-3 trust CSVs from the current list's saved V0 probabilities."""
    from .pipeline import _dataset_rows, _partition_sha256

    root = Path(root).resolve()
    latest = _latest_version(root)
    if latest is None:
        raise FileNotFoundError("Trust List 2 latest-version pointer is missing")
    source_version, source_path, source_metadata_path = latest
    source = pd.read_csv(source_path)
    source_metadata = json.loads(source_metadata_path.read_text())
    v0 = _dataset_rows("v0", root / "data" / "processed")
    if len(v0) != V0_ROWS or set(v0.model.astype(str)) != set(LABELS):
        raise ValueError("V0 reference is not the expected complete 2,699-row, 18-label dataset")
    v0_hash = _partition_sha256(v0)
    if source_metadata.get("validation_sha256") and source_metadata["validation_sha256"] != v0_hash:
        raise ValueError("latest Trust List 2 V0 hash differs from the current V0 dataset")
    ids = v0.id.astype(str).tolist()
    y_true = v0.model.astype(str).tolist()
    if source.duplicated(["experiment_id", "target_label"]).any():
        raise ValueError("latest Trust List 2 has duplicate checkpoint-label entries")

    rows_by_rank = {2: [], 3: []}
    rank_summary = {rank: {"correct_predictions": 0, "eligible_combinations": 0,
                           "high_precision_eligible_combinations": 0}
                    for rank in rows_by_rank}
    checkpoints = source.drop_duplicates("experiment_id").to_dict("records")
    for checkpoint_row in checkpoints:
        experiment_id = str(checkpoint_row["experiment_id"])
        checkpoint_rows = source[source.experiment_id.astype(str).eq(experiment_id)]
        if len(checkpoint_rows) != len(LABELS) or set(checkpoint_rows.target_label.astype(str)) != set(LABELS):
            raise ValueError(f"Trust List 2 entry {experiment_id} does not have all 18 labels")
        probabilities, prediction_path = _read_saved_probabilities(root, experiment_id, ids, y_true)
        base_by_label = checkpoint_rows.set_index("target_label", drop=False)
        for rank in rows_by_rank:
            metrics = _rank_k_metrics(y_true, probabilities, rank)
            summary = rank_summary[rank]
            summary["correct_predictions"] += metrics["_overall"]["correct_predictions"]
            summary["checkpoints_with_probabilities"] = summary.get("checkpoints_with_probabilities", 0) + 1
            summary.setdefault("probability_sources", {})
            source_key = "roster_cache" if prediction_path.parent == root / PREDICTION_DIR else "saved_validation_predictions"
            summary["probability_sources"][source_key] = summary["probability_sources"].get(source_key, 0) + 1
            for label in LABELS:
                base = base_by_label.loc[label].to_dict()
                calculated = metrics[label]
                base.update(calculated)
                base.update({
                    "smoothed_precision": (calculated["true_positive"] + 1.0) /
                                          (calculated["prediction_count"] + 2.0),
                    "overall_accuracy": metrics["_overall"]["overall_accuracy"],
                    "balanced_accuracy": metrics["_overall"]["balanced_accuracy"],
                    "validation_rows": V0_ROWS,
                })
                rows_by_rank[rank].append(base)

    generated_at = _utc_now()
    outputs = {}
    for rank, rows in rows_by_rank.items():
        frame = pd.DataFrame(rows)
        frame.attrs["trust_list_version"] = source_version
        ranked = _sort_and_rank(frame, generated_at=generated_at)
        rank_summary[rank]["eligible_combinations"] = int(ranked.rule_eligible.fillna(False).astype(bool).sum())
        rank_summary[rank]["high_precision_eligible_combinations"] = int(
            (ranked.rule_eligible.fillna(False).astype(bool) & ranked.precision.ge(0.90)).sum())
        output_path = root / "docs" / f"trust-list-2-top{rank}.csv"
        _atomic_write(ranked, output_path)
        outputs[rank] = {"path": str(output_path.relative_to(root)), "rows": int(len(ranked)),
                         "eligible_combinations": rank_summary[rank]["eligible_combinations"],
                         "correct_predictions": rank_summary[rank]["correct_predictions"]}

    # Verify that rank 1 in the same saved artifacts reproduces Trust List 2's
    # recorded per-label counts and metrics; rank 2/3 are otherwise independent.
    rank1_mismatches = []
    for checkpoint_row in checkpoints:
        experiment_id = str(checkpoint_row["experiment_id"])
        checkpoint_rows = source[source.experiment_id.astype(str).eq(experiment_id)]
        probabilities, _ = _read_saved_probabilities(root, experiment_id, ids, y_true)
        calculated = _rank_k_metrics(y_true, probabilities, 1)
        for row in checkpoint_rows.to_dict("records"):
            label = str(row["target_label"])
            metric = calculated[label]
            if (int(row.get("true_positive", 0)) != metric["true_positive"] or
                    int(row.get("prediction_count", 0)) != metric["prediction_count"]):
                rank1_mismatches.append({"experiment_id": experiment_id, "label": label})
    report = {
        "source_trust_list_version": source_version,
        "source_csv": str(source_path.relative_to(root)),
        "generated_at_utc": generated_at,
        "validation_rows": V0_ROWS,
        "validation_sha256": v0_hash,
        "checkpoints": len(checkpoints),
        "rank_summaries": rank_summary,
        "rank1_artifact_mismatches": rank1_mismatches,
        "outputs": outputs,
    }
    return report


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
    latest = _latest_version(root)
    if latest is None:
        return {"status": "deferred", "reason": "trust-list-2-v001 roster evaluation is not published yet"}
    current_version, current_path, current_metadata_path = latest
    current = pd.read_csv(current_path)
    experiment_id = str(manifest.get("experiment_id") or manifest_path.stem.removesuffix("_manifest"))
    if experiment_id in set(current.experiment_id.astype(str)):
        return {"status": "already_recorded", "experiment_id": experiment_id}
    sequence_id = _next_checkpoint_id(current)
    rows = _rows_from_training_manifest(manifest, manifest_path, sequence_id=sequence_id, root=root)
    if len(rows) != len(LABELS):
        return {"status": "skipped", "reason": "complete full-V0 training metrics are unavailable"}
    merged = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    generated_at = _utc_now()
    next_version = f"{VERSION_PREFIX}{_version_number(current_version) + 1:03d}"
    merged.attrs["trust_list_version"] = next_version
    metadata = json.loads(current_metadata_path.read_text())
    metadata.update({"generated_at_utc": generated_at,
                     "last_added_experiment_id": experiment_id,
                     "parent_version": current_version})
    published = _publish_version(root, merged, metadata, next_version)
    return {"status": "added", "version": next_version, "sequence_id": sequence_id,
            "labels": len(rows), "checkpoints": published["checkpoints"]}


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("action", nargs="?", choices=("initialize", "build-roster", "generate-topk"), default="initialize")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.action == "build-roster":
        result = build_roster_trust_list(root=args.root, batch_size=args.batch_size)
    elif args.action == "generate-topk":
        result = generate_topk_trust_lists(root=args.root)
    else:
        result = initialize_v2_trust_list(root=args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
