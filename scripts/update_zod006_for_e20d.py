"""Merge E20-D checkpoint assets into a complete ZOD006 dataset version.

Kaggle dataset versions replace the visible file set, so this script requires a
full local copy of the current ZOD006 dataset and preserves every existing
file before adding the missing E20-D checkpoint payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Complete extracted current zodrak/zod006-models file set")
    parser.add_argument("--e20d-shard", type=Path,
                        default=Path("artifacts/kaggle/e20d-checkpoints-v001"))
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    shard_dir = args.e20d_shard.resolve()
    current_catalog_path = dataset_dir / "checkpoint_catalog.json"
    current_manifest_path = dataset_dir / "zod006_manifest.json"
    incoming_catalog_path = shard_dir / "checkpoint_catalog.json"
    if not all(path.is_file() for path in
               (current_catalog_path, current_manifest_path, incoming_catalog_path)):
        raise FileNotFoundError("dataset or E20-D shard is missing its catalog/manifest")

    catalog = json.loads(current_catalog_path.read_text())
    incoming = json.loads(incoming_catalog_path.read_text())
    manifest = json.loads(current_manifest_path.read_text())
    if not isinstance(catalog.get("assets"), dict) or not isinstance(catalog.get("checkpoints"), dict):
        raise ValueError("unsupported existing ZOD006 checkpoint catalog schema")

    added_assets: list[str] = []
    for digest, item in incoming["assets"].items():
        if digest in catalog["assets"]:
            continue
        source = shard_dir / item["kaggle_filename"]
        if not source.is_file() or source.stat().st_size != int(item["size"]):
            raise ValueError(f"missing/size-mismatched E20-D asset {source}")
        if sha256(source) != digest:
            raise ValueError(f"sha256 mismatch for E20-D asset {source}")
        filename = f"asset__{digest}"
        target = dataset_dir / filename
        if target.exists() and sha256(target) != digest:
            raise ValueError(f"refusing to overwrite conflicting asset {target}")
        if not target.exists():
            shutil.copy2(source, target)
        catalog["assets"][digest] = {"kaggle_filename": filename,
                                     "size": int(item["size"])}
        added_assets.append(digest)

    added_checkpoints: list[str] = []
    for experiment_id, entry in incoming["checkpoints"].items():
        if experiment_id in catalog["checkpoints"]:
            continue
        files = []
        for source_file in entry["files"]:
            digest = source_file["sha256"]
            if digest in catalog["assets"]:
                filename = catalog["assets"][digest]["kaggle_filename"]
                dataset_ref = "zodrak/zod006-models"
            else:
                filename = source_file["flat_filename"]
                dataset_ref = source_file["dataset_ref"]
            files.append({
                "path": source_file["relative_path"],
                "kaggle_filename": filename,
                "dataset_ref": dataset_ref,
                "sha256": digest,
                "size": int(source_file["size"]),
            })
        catalog["checkpoints"][experiment_id] = {
            "experiment_id": experiment_id,
            "sequence_id": entry.get("sequence_id"),
            "model": entry["model"],
            "training_dataset": entry.get("config", {}).get("dataset"),
            "config": entry.get("config", {}),
            "family": entry.get("family"),
            "classes": incoming.get("classes"),
            "files": files,
            "base_model": entry.get("config", {}).get("checkpoint"),
            "base_model_revision": entry.get("config", {}).get("revision"),
            "introduced_in_version": None,
        }
        added_checkpoints.append(experiment_id)

    catalog["dataset_version"] = int(catalog.get("dataset_version", 0)) + 1
    catalog["e20d_release"] = {
        "experiment": "E20-D",
        "source_dataset": "zodrak/e20d-checkpoints-v001",
        "checkpoint_ids": incoming.get("checkpoints", {}).keys(),
        "added_checkpoint_ids": added_checkpoints,
        "added_asset_sha256": added_assets,
    }
    # JSON cannot serialize dict_keys in all Python versions.
    catalog["e20d_release"]["checkpoint_ids"] = list(catalog["e20d_release"]["checkpoint_ids"])
    manifest["e20d_checkpoint_release"] = {
        "source": "zodrak/e20d-checkpoints-v001",
        "checkpoint_ids": list(incoming["checkpoints"]),
        "added_checkpoint_ids": added_checkpoints,
        "added_asset_sha256": added_assets,
    }
    current_catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    current_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Every old file remains present; assets are added once by hash.
    summary = {
        "preserved_existing_file_count": sum(
            1 for path in dataset_dir.rglob("*") if path.is_file()
            and path.name not in {"checkpoint_catalog.json", "zod006_manifest.json"}
            and not path.name.startswith("asset__")
        ),
        "added_checkpoint_count": len(added_checkpoints),
        "added_asset_count": len(added_assets),
        "added_asset_bytes": sum(int(incoming["assets"][digest]["size"])
                                  for digest in added_assets),
        "total_file_count": sum(path.is_file() for path in dataset_dir.rglob("*")),
        "catalog_checkpoints": len(catalog["checkpoints"]),
        "added_checkpoint_ids": added_checkpoints,
        "added_asset_sha256": added_assets,
    }
    (dataset_dir / "e20d_update_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
