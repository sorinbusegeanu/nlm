"""Prepare a content-addressed checkpoint delta for Kaggle.

The command is dry-run only. Kaggle dataset versions replace the latest file
set, so this must not be used to append files to the shared ZOD006 dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATASET_REF = "zodrak/zod006-models"
STORE_DIR = ROOT / "artifacts/kaggle/zod006-models"
PREDICTIONS_DIR = ROOT / "artifacts/predictions"
CATALOG_NAME = "checkpoint_catalog.json"
SCHEMA_VERSION = 1

TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt",
    "merges.txt", "added_tokens.json", "chat_template.jinja",
    "generation_config.json",
}
MODEL_FILES = {
    "config.json", "model.safetensors", "pytorch_model.bin",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
    "adapter_config.json", "adapter_model.safetensors", "classifier.pt",
    "checkpoint_manifest.json", "model.cbm", "stylometry.json", "readout.joblib",
}
SKIP_NAMES = {
    "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth",
    "trainer_state.json", "training_log.json", "training_progress.json",
    "checkpoint_selection.json", "batch_size_probe.json",
    "baseline_validation.json", "validation_metrics.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle_binary() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    fallback = Path.home() / "cuda13.x/bin/kaggle"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("Kaggle CLI not found; install/authenticate it before publishing")


def _run_json(command: list[str]) -> Any:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    # Kaggle CLI may print a paging token before the JSON payload.
    payload = result.stdout[result.stdout.find("["):] if "[" in result.stdout else result.stdout
    return json.loads(payload)


def kaggle_status(dataset_ref: str = DATASET_REF) -> dict[str, Any]:
    return _run_json([kaggle_binary(), "datasets", "status", dataset_ref, "--format", "json"])


def kaggle_file_inventory(dataset_ref: str = DATASET_REF) -> dict[str, int]:
    """Read every page; Kaggle defaults to only 20 files per page."""
    command_base = [kaggle_binary(), "datasets", "files", dataset_ref, "--format", "json", "--page-size", "200"]
    page_token: str | None = None
    inventory: dict[str, int] = {}
    while True:
        command = command_base + (["--page-token", page_token] if page_token else [])
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        token = None
        for line in result.stdout.splitlines():
            if line.startswith("Next Page Token = "):
                token = line.partition("=")[2].strip()
        start = result.stdout.find("[")
        if start < 0:
            raise ValueError("Kaggle file listing did not contain JSON")
        files = json.loads(result.stdout[start:])
        for item in files:
            inventory[str(item["name"])] = int(item.get("size", item.get("totalBytes", 0)))
        if not token or token == page_token:
            break
        page_token = token
    return inventory


def resolve_checkpoint(experiment_id: str) -> tuple[dict[str, Any], Path]:
    manifest_path = PREDICTIONS_DIR / f"{experiment_id}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    raw_path = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path")
    if not raw_path:
        raise ValueError(f"checkpoint path missing in {manifest_path}")
    source = Path(raw_path)
    source = source if source.is_absolute() else ROOT / source
    if source.is_dir() and (source / "best_checkpoint").is_dir():
        source = source / "best_checkpoint"
    elif source.is_dir() and manifest.get("selected_epoch"):
        selected = source / f"epoch-{int(manifest['selected_epoch']):02d}"
        if selected.is_dir():
            source = selected
    if not source.exists():
        raise FileNotFoundError(f"checkpoint is missing for {experiment_id}: {source}")
    return manifest, source


def inference_files(source: Path, model: str) -> list[tuple[Path, str]]:
    if source.is_file():
        return [(source, source.name)]
    found = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.startswith("epoch-") or part in {"continuation_checkpoint", "optimizer_state"}
               for part in relative.parts):
            continue
        if path.name in SKIP_NAMES or path.name.endswith((".log", ".lock", ".tmp", ".incomplete")):
            continue
        if (path.name in MODEL_FILES or path.name in TOKENIZER_FILES
                or (path.name.startswith(("configuration_", "modeling_")) and path.suffix == ".py")):
            found.append((path, relative.as_posix()))
    if not found:
        raise FileNotFoundError(f"no inference assets found in {source} for {model}")
    return found


def _legacy_catalog(store_dir: Path, remote: dict[str, int]) -> dict[str, Any]:
    """Import already-published candidates without treating local-only files as remote."""
    catalog: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_ref": DATASET_REF,
        "dataset_version": None,
        "assets": {},
        "checkpoints": {},
    }
    manifests = [
        (store_dir / "zod006_manifest.json", "candidates"),
        (store_dir / "lc3_lc4_manifest.json", "candidates"),
    ]
    for manifest_path, key in manifests:
        if not manifest_path.is_file():
            continue
        bundle = json.loads(manifest_path.read_text())
        for experiment_id, candidate in bundle.get(key, {}).items():
            records = candidate.get("files", [])
            resolved = []
            complete = bool(records)
            for record in records:
                filename = record.get("flat_filename")
                local = store_dir / str(filename or "")
                if not filename or filename not in remote or not local.is_file():
                    complete = False
                    break
                size = local.stat().st_size
                if size != remote[filename]:
                    complete = False
                    break
                digest = sha256_file(local)
                expected = record.get("sha256")
                if expected and digest != expected:
                    raise ValueError(f"local legacy asset hash mismatch: {local}")
                resolved.append({
                    "path": record.get("relative_path", local.name),
                    "sha256": digest,
                    "size": size,
                    "kaggle_filename": filename,
                })
            if not complete:
                continue
            for asset in resolved:
                catalog["assets"].setdefault(asset["sha256"], {
                    "size": asset["size"], "kaggle_filename": asset["kaggle_filename"]
                })
            catalog["checkpoints"][str(experiment_id)] = {
                "experiment_id": str(experiment_id),
                "model": candidate.get("model"),
                "training_dataset": candidate.get("training_dataset"),
                "base_model": candidate.get("base_model"),
                "base_model_revision": candidate.get("base_model_revision") or candidate.get("model_revision"),
                "classes": candidate.get("classes"),
                "config": candidate.get("config", {}),
                "files": resolved,
                "introduced_in_version": None,
                "source_manifest": candidate.get("source_manifest"),
            }
    return catalog


def seed_catalog(store_dir: Path, remote: dict[str, int], current_version: int) -> dict[str, Any]:
    path = store_dir / CATALOG_NAME
    if path.is_file():
        catalog = json.loads(path.read_text())
    else:
        catalog = _legacy_catalog(store_dir, remote)
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint catalog schema: {catalog.get('schema_version')}")
    catalog["dataset_ref"] = DATASET_REF
    catalog["dataset_version"] = current_version
    catalog.setdefault("assets", {})
    catalog.setdefault("checkpoints", {})
    return catalog


def resolve_catalog_checkpoint(catalog: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    """Resolve an ID and fail if its catalog entry is incomplete."""
    try:
        checkpoint = catalog["checkpoints"][experiment_id]
    except KeyError as exc:
        raise KeyError(f"checkpoint is not in the Kaggle catalog: {experiment_id}") from exc
    if not checkpoint.get("files"):
        raise ValueError(f"checkpoint has no published files: {experiment_id}")
    for item in checkpoint["files"]:
        if not item.get("path") or not item.get("kaggle_filename") or not item.get("sha256"):
            raise ValueError(f"checkpoint catalog entry is incomplete: {experiment_id}")
    return checkpoint


def prepare_delta(
    *,
    store_dir: Path,
    stage_dir: Path,
    remote: dict[str, int],
    catalog: dict[str, Any],
    checkpoints: Iterable[dict[str, Any]],
    next_version: int,
) -> dict[str, Any]:
    """Stage only absent content-addressed assets plus a refreshed small catalog."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    if any(stage_dir.iterdir()):
        raise ValueError(f"delta staging directory must be empty: {stage_dir}")
    staged_assets: set[str] = set()
    changed_catalog = False
    assets = catalog.setdefault("assets", {})
    checkpoint_index = catalog.setdefault("checkpoints", {})

    for item in checkpoints:
        experiment_id = str(item["experiment_id"])
        model = str(item["model"])
        files = []
        for source, relative in item["files"]:
            source = Path(source)
            digest = sha256_file(source)
            size = source.stat().st_size
            prior_asset = assets.get(digest)
            filename = (prior_asset or {}).get("kaggle_filename", f"asset__{digest}")
            if filename in remote and remote[filename] != size:
                raise ValueError(f"remote asset size conflict for {filename}")
            if filename not in remote and filename not in staged_assets:
                destination = stage_dir / filename
                shutil.copy2(source, destination)
                staged_assets.add(filename)
            assets[digest] = {"size": size, "kaggle_filename": filename}
            files.append({"path": str(relative), "sha256": digest, "size": size,
                          "kaggle_filename": filename})

        entry = {
            "experiment_id": experiment_id,
            "model": model,
            "training_dataset": item.get("training_dataset"),
            "base_model": item.get("base_model"),
            "base_model_revision": item.get("base_model_revision"),
            "classes": item.get("classes"),
            "config": item.get("config", {}),
            "files": files,
            "introduced_in_version": next_version,
            "source_manifest": item.get("source_manifest"),
        }
        old = checkpoint_index.get(experiment_id)
        if old is not None:
            if [(f["path"], f["sha256"]) for f in old.get("files", [])] != [
                    (f["path"], f["sha256"]) for f in files]:
                raise ValueError(f"checkpoint ID already exists with different payload: {experiment_id}")
            continue
        checkpoint_index[experiment_id] = entry
        changed_catalog = True

    if not changed_catalog and not staged_assets:
        return {"staged_files": [], "staged_bytes": 0, "checkpoint_count": len(checkpoint_index),
                "catalog_changed": False}

    catalog["dataset_version"] = next_version
    catalog_path = stage_dir / CATALOG_NAME
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    metadata_path = store_dir / "dataset-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Kaggle dataset metadata missing: {metadata_path}")
    shutil.copy2(metadata_path, stage_dir / "dataset-metadata.json")
    staged = sorted(path for path in stage_dir.iterdir() if path.is_file())
    # Kaggle's CLI excludes dataset-metadata.json from uploads.
    upload_files = [path for path in staged if path.name != "dataset-metadata.json"]
    return {
        "staged_files": [path.name for path in upload_files],
        "staged_bytes": sum(path.stat().st_size for path in upload_files),
        "checkpoint_count": len(checkpoint_index),
        "catalog_changed": changed_catalog,
    }


def build_checkpoint(experiment_id: str) -> dict[str, Any]:
    manifest, source = resolve_checkpoint(experiment_id)
    model = str(manifest.get("model", "unknown"))
    return {
        "experiment_id": experiment_id,
        "model": model,
        "training_dataset": manifest.get("training_dataset") or manifest.get("dataset"),
        "base_model": manifest.get("config", {}).get("checkpoint"),
        "base_model_revision": manifest.get("model_revision") or manifest.get("config", {}).get("revision"),
        "classes": list(map(str, manifest.get("classes", []))),
        "config": manifest.get("config", {}),
        "files": inference_files(source, model),
        "source_manifest": str((PREDICTIONS_DIR / f"{experiment_id}_manifest.json").relative_to(ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", action="append", required=True,
                        help="Local experiment ID; may be specified more than once")
    parser.add_argument("--store-dir", type=Path, default=STORE_DIR)
    parser.add_argument("--stage-dir", type=Path, help="Optional explicit delta staging directory")
    parser.add_argument("--message", default="Append NLM inference checkpoints")
    parser.add_argument("--publish", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    status = kaggle_status()
    current_version = int(status["current_version_number"])
    remote = kaggle_file_inventory()
    catalog = seed_catalog(args.store_dir, remote, current_version)
    checkpoints = [build_checkpoint(experiment_id) for experiment_id in args.experiment_id]
    if args.stage_dir:
        stage_dir = args.stage_dir
    else:
        stage_dir = Path(tempfile.mkdtemp(prefix="kaggle-checkpoint-delta-", dir=args.store_dir.parent))
    result = prepare_delta(store_dir=args.store_dir, stage_dir=stage_dir, remote=remote,
                           catalog=catalog, checkpoints=checkpoints,
                           next_version=current_version + 1)
    print(json.dumps({"dataset": DATASET_REF, "current_version": current_version,
                      "proposed_version": current_version + 1, "stage_dir": str(stage_dir),
                      **result}, indent=2))
    if args.publish:
        raise RuntimeError(
            "Kaggle dataset versions replace the latest file set; do not publish this delta to zod006-models. "
            "Prepare a new immutable checkpoint shard with scripts/prepare_lc3_lc4_checkpoint_shard.py."
        )
    print("Dry run only; no upload performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
