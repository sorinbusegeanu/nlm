"""Prepare and optionally publish an immutable LC3/LC4 Kaggle asset shard.

Unlike a Kaggle dataset version, this creates one new dataset containing only
the assets required by this LC3/LC4 release. Existing ZOD006 assets are
referenced from their existing dataset and are not uploaded again.
"""
from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.publish_kaggle_checkpoint_store import (
    CATALOG_NAME,
    DATASET_REF,
    ROOT,
    STORE_DIR,
    build_checkpoint,
    kaggle_binary,
    kaggle_file_inventory,
    kaggle_status,
    seed_catalog,
    sha256_file,
)

LC_MANIFEST = STORE_DIR / "lc3_lc4_manifest.json"
TRUST_CSV = ROOT / "docs/trust-list-2-v035.csv"
KERNEL_DIR = ROOT / "artifacts/kaggle/lc3-lc4-kernel"
SHARD_REF = "zodrak/lc3-lc4-checkpoints-v035"
ASSET_STAGING_DIR = ".asset_payload"
MAX_ARCHIVE_BYTES = 5 * 1024**3
REUSE_BUNDLES = {
    "zodrak/three-strategy-checkpoints": "three-strategy-checkpoints",
    "zodrak/trust2-v013-top5-models": "trust2-v013-top5-models",
    "zodrak/ensemble-hypotheses-nov15-extra-checkpoints": "ensemble-hypotheses-nov15-extra-checkpoints",
    "zodrak/zod04-models": "zod04-models",
}


def _content_asset(source: Path, stage: Path, remote: dict[str, dict[str, int]], catalog: dict[str, Any],
                   shard_ref: str, staged: set[str]) -> dict[str, Any]:
    digest = sha256_file(source)
    size = source.stat().st_size
    old = catalog.get("assets", {}).get(digest)
    if old:
        filename = old["kaggle_filename"]
        source_dataset = old.get("dataset_ref", DATASET_REF)
        if filename in staged and source_dataset == shard_ref:
            return {"sha256": digest, "size": size, "flat_filename": filename,
                    "dataset_ref": source_dataset}
        source_remote = remote.get(source_dataset, {})
        if filename not in source_remote or int(source_remote[filename]) != size:
            raise ValueError(f"catalog references an unavailable or size-mismatched asset: {filename}")
        return {"sha256": digest, "size": size, "flat_filename": filename,
                "dataset_ref": source_dataset}
    filename = f"assets/asset__{digest}"
    shard_remote = remote.get(shard_ref, {})
    if filename in shard_remote and int(shard_remote[filename]) != size:
        raise ValueError(f"content-addressed filename collision: {filename}")
    if filename not in shard_remote and filename not in staged:
        destination = stage / ASSET_STAGING_DIR / Path(filename).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        staged.add(filename)
    catalog.setdefault("assets", {})[digest] = {
        "size": size,
        "kaggle_filename": filename,
        "dataset_ref": shard_ref,
    }
    return {"sha256": digest, "size": size, "flat_filename": filename,
            "dataset_ref": shard_ref}


def _external_asset_index(remote: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """Index exact checkpoint files already published in reusable bundles."""
    bundles = {
        "zodrak/three-strategy-checkpoints": (
            "three-strategy-checkpoints", "three_strategy_supplement_manifest.json"),
        "zodrak/trust2-v013-top5-models": (
            "trust2-v013-top5-models", "trust2_v013_top5_manifest.json"),
        "zodrak/ensemble-hypotheses-nov15-extra-checkpoints": (
            "ensemble-hypotheses-nov15-extra-checkpoints", "ensemble_hypotheses_nov15_extras_manifest.json"),
    }
    lc = json.loads(LC_MANIFEST.read_text())
    lc_experiments = {str(c["experiment_id"]) for c in lc["candidates"].values()}
    index: dict[str, dict[str, Any]] = {}
    for dataset_ref, (directory, manifest_name) in bundles.items():
        if dataset_ref not in remote:
            continue
        root = ROOT / "artifacts/kaggle" / directory
        manifest_path = root / manifest_name
        if not manifest_path.is_file():
            continue
        bundle = json.loads(manifest_path.read_text())
        for candidate in bundle.get("candidates", {}).values():
            if str(candidate.get("experiment_id")) not in lc_experiments:
                continue
            for item in candidate.get("files", []):
                filename = str(item.get("flat_filename", ""))
                expected_size = int(item.get("size_bytes", item.get("size", 0)))
                path = root / filename
                if (not filename or filename not in remote[dataset_ref]
                        or remote[dataset_ref][filename] != expected_size
                        or not path.is_file() or path.stat().st_size != expected_size):
                    continue
                digest = sha256_file(path)
                expected_hash = item.get("sha256")
                if expected_hash and digest != expected_hash:
                    raise ValueError(f"published candidate hash mismatch: {dataset_ref}/{filename}")
                old = index.get(digest)
                if old and old["size"] != expected_size:
                    raise ValueError(f"same asset hash has inconsistent sizes: {digest}")
                index.setdefault(digest, {
                    "size": expected_size, "kaggle_filename": filename,
                    "dataset_ref": dataset_ref,
                })
    return index


def _stage_metadata_file(source: Path, dest_name: str, stage: Path) -> None:
    shutil.copy2(source, stage / dest_name)


def _pack_assets(stage: Path, staged_asset_names: set[str]) -> int:
    payload_root = stage / ASSET_STAGING_DIR
    payloads = sorted((p for p in payload_root.iterdir() if p.is_file()),
                      key=lambda p: (-p.stat().st_size, p.name)) if payload_root.exists() else []
    groups: list[list[Path]] = []
    sizes: list[int] = []
    for payload in payloads:
        size = payload.stat().st_size
        suitable = [i for i, current in enumerate(sizes) if current + size <= MAX_ARCHIVE_BYTES]
        index = suitable[0] if suitable else len(groups)
        if index == len(groups):
            groups.append([])
            sizes.append(0)
        groups[index].append(payload)
        sizes[index] += size
    for index, group in enumerate(groups):
        target_root = stage / f"part_{index:02d}" / "assets"
        target_root.mkdir(parents=True)
        for source in group:
            os.link(source, target_root / source.name)
    if payload_root.exists():
        shutil.rmtree(payload_root)
    expected = {Path(name).name for name in staged_asset_names}
    actual = {p.name for folder in stage.glob("part_*")
              for p in (folder / "assets").iterdir() if p.is_file()}
    if actual != expected:
        raise AssertionError(f"asset packing mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return len(groups)


def prepare_shard(stage_dir: Path, shard_ref: str = SHARD_REF) -> dict[str, Any]:
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise ValueError(f"shard staging directory must be empty: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    if not LC_MANIFEST.is_file():
        raise FileNotFoundError(LC_MANIFEST)

    original = json.loads(LC_MANIFEST.read_text())
    status = kaggle_status(DATASET_REF)
    zod006_version = int(status["current_version_number"])
    remote: dict[str, dict[str, int]] = {DATASET_REF: kaggle_file_inventory(DATASET_REF)}
    wanted_experiments = {str(c["experiment_id"]) for c in original["candidates"].values()}
    bundle_manifests = {
        "zodrak/three-strategy-checkpoints": "three-strategy-checkpoints/three_strategy_supplement_manifest.json",
        "zodrak/trust2-v013-top5-models": "trust2-v013-top5-models/trust2_v013_top5_manifest.json",
        "zodrak/ensemble-hypotheses-nov15-extra-checkpoints": "ensemble-hypotheses-nov15-extra-checkpoints/ensemble_hypotheses_nov15_extras_manifest.json",
    }
    for dataset_ref, relpath in bundle_manifests.items():
        manifest_path = ROOT / "artifacts/kaggle" / relpath
        if manifest_path.is_file():
            bundle = json.loads(manifest_path.read_text())
            if any(str(c.get("experiment_id")) in wanted_experiments
                   for c in bundle.get("candidates", {}).values()):
                remote[dataset_ref] = kaggle_file_inventory(dataset_ref)
    catalog = seed_catalog(STORE_DIR, remote[DATASET_REF], zod006_version)
    for digest, asset in _external_asset_index(remote).items():
        catalog.setdefault("assets", {}).setdefault(digest, asset)
    ids = list(original["candidate_ids"])
    checkpoints = [build_checkpoint(str(experiment_id)) for experiment_id in ids]
    staged: set[str] = set()
    candidates: dict[str, dict[str, Any]] = {}

    for checkpoint in checkpoints:
        cid = checkpoint["experiment_id"]
        files = []
        for source, relative in checkpoint["files"]:
            asset = _content_asset(Path(source), stage_dir, remote, catalog, shard_ref, staged)
            files.append({"path": relative, **asset})
        candidates[cid] = {
            "experiment_id": cid,
            "model": checkpoint["model"],
            "training_dataset": checkpoint.get("training_dataset"),
            "base_model": checkpoint.get("base_model"),
            "base_model_revision": checkpoint.get("base_model_revision"),
            "classes": checkpoint.get("classes"),
            "config": checkpoint.get("config", {}),
            "files": files,
            "source_manifest": checkpoint.get("source_manifest"),
        }

    shard_catalog = {
        "schema_version": 1,
        "dataset_ref": shard_ref,
        "zod006_base_dataset": DATASET_REF,
        "zod006_base_version": zod006_version,
        "assets": catalog.get("assets", {}),
        "checkpoints": candidates,
        "base_models": {},
    }

    # Publish the exact pinned MiniCPM5 base snapshot used by this notebook.
    mini_manifest_path = STORE_DIR / "lc3_lc4_minicpm5_base_manifest.json"
    mini_manifest = json.loads(mini_manifest_path.read_text())
    for item in mini_manifest["files"]:
        asset = _content_asset(STORE_DIR / item["flat_filename"], stage_dir, remote,
                               catalog, shard_ref, staged)
        item.update(asset)
        shard_catalog["base_models"].setdefault("openbmb/MiniCPM5-2B-Base", []).append(asset)
    _stage_metadata_file(mini_manifest_path, mini_manifest_path.name, stage_dir)
    (stage_dir / mini_manifest_path.name).write_text(json.dumps(mini_manifest, indent=2) + "\n")

    granite_manifest_path = STORE_DIR / "lc3_lc4_granite_base_manifest.json"
    granite_manifest = json.loads(granite_manifest_path.read_text())
    for model, model_files in granite_manifest["models"].items():
        for item in model_files:
            asset = _content_asset(STORE_DIR / item["flat_filename"], stage_dir, remote,
                                   catalog, shard_ref, staged)
            item.update(asset)
            shard_catalog["base_models"].setdefault(model, []).append(asset)
    (stage_dir / granite_manifest_path.name).write_text(json.dumps(granite_manifest, indent=2) + "\n")

    runtime_manifest = {}
    for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        source = STORE_DIR / f"source__{name}"
        if not source.is_file():
            raise FileNotFoundError(source)
        runtime_manifest[name] = _content_asset(source, stage_dir, remote, catalog, shard_ref, staged)
    runtime_path = stage_dir / "lc3_lc4_runtime_manifest.json"
    runtime_path.write_text(json.dumps(runtime_manifest, indent=2) + "\n")

    mounted_sources = sorted({
        item["dataset_ref"]
        for item in [
            *(f for candidate in candidates.values() for f in candidate["files"]),
            *(f for model_files in shard_catalog["base_models"].values() for f in model_files),
            *runtime_manifest.values(),
        ]
        if item["dataset_ref"] != shard_ref
    })
    shard_catalog["mounted_dataset_sources"] = mounted_sources
    referenced_assets = [
        *(f for candidate in candidates.values() for f in candidate["files"]),
        *(f for model_files in shard_catalog["base_models"].values() for f in model_files),
        *runtime_manifest.values(),
    ]
    reuse_summary: dict[str, dict[str, Any]] = {}
    for ref in sorted({f["dataset_ref"] for f in referenced_assets if f["dataset_ref"] != shard_ref}):
        unique = {f["sha256"]: f for f in referenced_assets if f["dataset_ref"] == ref}
        reuse_summary[ref] = {"unique_files": len(unique),
                              "bytes": sum(int(f["size"]) for f in unique.values())}
    shard_catalog["reused_asset_summary"] = reuse_summary

    # Rewrite the LC manifest's checkpoint references to content-addressed files.
    updated = original
    for cid, candidate in updated["candidates"].items():
        if cid not in candidates:
            raise ValueError(f"candidate missing from prepared catalog: {cid}")
        candidate["files"] = [
            {"relative_path": f["path"], "flat_filename": f["flat_filename"],
             "sha256": f["sha256"], "size": f["size"], "dataset_ref": f["dataset_ref"]}
            for f in candidates[cid]["files"]
        ]
    updated["asset_catalog"] = CATALOG_NAME
    (stage_dir / "lc3_lc4_manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
    (stage_dir / CATALOG_NAME).write_text(json.dumps(shard_catalog, indent=2, sort_keys=True) + "\n")

    if not TRUST_CSV.is_file():
        raise FileNotFoundError(TRUST_CSV)
    shutil.copy2(TRUST_CSV, stage_dir / TRUST_CSV.name)
    metadata = {
        "title": "NLM LC3 LC4 Trust List 2 Checkpoints",
        "id": shard_ref,
        "licenses": [{"name": "other"}],
        "subtitle": "Inference assets for LC3 and LC4 cascades",
        "description": "Immutable LC3/LC4 inference checkpoint shard. Existing shared assets are referenced from their original Kaggle dataset.",
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    archive_groups = _pack_assets(stage_dir, staged)
    files = [p for p in stage_dir.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"]
    return {"dataset": shard_ref, "base_dataset_version": zod006_version,
            "candidates": len(candidates), "logical_file_count": len(files),
            "archive_groups": archive_groups,
            "mounted_dataset_sources": mounted_sources,
            "reused_asset_summary": reuse_summary,
            "upload_bytes": sum(p.stat().st_size for p in files),
            "upload_gib": round(sum(p.stat().st_size for p in files) / 1024**3, 3),
            "shared_assets_not_reuploaded": sum(f["dataset_ref"] == DATASET_REF
                for c in candidates.values() for f in c["files"])}


def update_kernel_metadata(shard_ref: str = SHARD_REF, mounted_sources: list[str] | None = None) -> None:
    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    sources = list(metadata.get("dataset_sources", []))
    metadata["dataset_sources"] = list(dict.fromkeys([
        *sources, "zodrak/zod006-models", shard_ref, *(mounted_sources or [])
    ]))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def stage_summary(stage_dir: Path, shard_ref: str) -> dict[str, Any]:
    metadata_path = stage_dir / "dataset-metadata.json"
    catalog_path = stage_dir / CATALOG_NAME
    manifest_path = stage_dir / "lc3_lc4_manifest.json"
    if not (metadata_path.is_file() and catalog_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError(f"prepared shard is incomplete: {stage_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("id") != shard_ref:
        raise ValueError(f"staged dataset ID differs: {metadata.get('id')} != {shard_ref}")
    catalog = json.loads(catalog_path.read_text())
    files = [p for p in stage_dir.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"]
    return {"dataset": shard_ref, "candidates": len(catalog.get("checkpoints", {})),
            "logical_file_count": len(files), "archive_groups": len(list(stage_dir.glob("part_*"))),
            "upload_bytes": sum(p.stat().st_size for p in files),
            "upload_gib": round(sum(p.stat().st_size for p in files) / 1024**3, 3)}


def wait_ready(dataset_ref: str, timeout_seconds: int = 3600) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = kaggle_status(dataset_ref)
        if str(status.get("status", "")).lower() == "ready":
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Kaggle dataset did not become ready: {dataset_ref}; status={status}")
        print(f"Waiting for Kaggle dataset readiness: {dataset_ref} status={status.get('status')}", flush=True)
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=ROOT / "artifacts/kaggle/lc3-lc4-shard")
    parser.add_argument("--shard-ref", default=SHARD_REF)
    parser.add_argument("--publish", action="store_true", help="Create a new immutable Kaggle dataset")
    parser.add_argument("--push-kernel", action="store_true", help="Upload the prepared LC3/LC4 notebook")
    args = parser.parse_args()
    from scripts.package_lc3_lc4_kaggle import build_kernel

    build_kernel()
    if (args.stage_dir / CATALOG_NAME).is_file():
        result = stage_summary(args.stage_dir, args.shard_ref)
    else:
        result = prepare_shard(args.stage_dir, args.shard_ref)
    print(json.dumps(result, indent=2))
    if args.publish:
        existing = {item["ref"] for item in json.loads(subprocess.run(
            [kaggle_binary(), "datasets", "list", "--mine", "--page-size", "200", "--format", "json"],
            check=True, text=True, capture_output=True).stdout)}
        if args.shard_ref in existing:
            raise ValueError(f"immutable shard already exists; choose a new cohort slug: {args.shard_ref}")
        subprocess.run([kaggle_binary(), "datasets", "create", "-p", str(args.stage_dir),
                        "--keep-tabular", "--dir-mode", "zip"], check=True)
        wait_ready(args.shard_ref)
        remote = kaggle_file_inventory(args.shard_ref)
        direct = {p.name for p in args.stage_dir.iterdir()
                  if p.is_file() and p.name != "dataset-metadata.json"}
        shard_catalog = json.loads((args.stage_dir / CATALOG_NAME).read_text())
        remote_by_basename = {Path(name).name: int(size) for name, size in remote.items()}
        missing_direct = direct - set(remote)
        expected_assets = {
            Path(item["kaggle_filename"]).name: int(item["size"])
            for item in shard_catalog["assets"].values()
            if item.get("dataset_ref") == args.shard_ref
        }
        mismatched_assets = {
            name: {"expected": size, "remote": remote_by_basename.get(name)}
            for name, size in expected_assets.items()
            if remote_by_basename.get(name) != size
        }
        if missing_direct or mismatched_assets:
            raise RuntimeError(
                "published shard inventory mismatch; "
                f"missing direct files={sorted(missing_direct)}, asset mismatches={mismatched_assets}"
            )
        if args.push_kernel:
            update_kernel_metadata(args.shard_ref, shard_catalog.get("mounted_dataset_sources", []))
            # Saving the notebook version must not execute LC3/LC4 inference.
            subprocess.run([kaggle_binary(), "kernels", "push", "-p", str(KERNEL_DIR), "--no-run"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
