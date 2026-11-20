"""Build runtime-inference Kaggle assets/notebooks for all ORACLE04 strategies."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRUST = ROOT / "docs/trust-list-2-v025.csv"
DEPLOY = ROOT / "artifacts/deployment/oracle04"
ASSETS = ROOT / "artifacts/kaggle/oracle04-dynamic-assets"
KERNELS = ROOT / "artifacts/kaggle/oracle04-submission-kernels"
STRATEGIES = {
    "combined_logistic_core": ("oracle04-combined-logistic-core", "ORACLE04 Combined Logistic Core"),
    "global_logistic_core": ("oracle04-global-logistic-core", "ORACLE04 Global Logistic Core"),
    "global_catboost_core": ("oracle04-global-catboost-core", "ORACLE04 Global CatBoost Core"),
}
MODEL_SOURCES = [
    "google/gemma-4/transformers/gemma-4-e2b/1",
    "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
    "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
    "qwen-lm/qwen-3/transformers/1.7b-base/1",
]
SKIP_DIRS = {"continuation_checkpoint", "training_state", "optimizer", "optimizer_state"}
SKIP_FILES = {"training_progress.json", "training_log.json", "trainer_state.json",
              "optimizer.pt", "optimizer.bin", "scheduler.pt", "rng_state.pth",
              "scaler.pt", "checkpoint_selection.json", "baseline_validation.json"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source = source.resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _checkpoint_path(row: pd.Series, manifest: dict) -> Path:
    value = (manifest.get("best_checkpoint_path") or manifest.get("selected_checkpoint_path")
             or manifest.get("checkpoint_path") or row["checkpoint_path"])
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir() and (path / "best_checkpoint").is_dir():
        path = path / "best_checkpoint"
    return path


def _copy_checkpoint(path: Path, sid: str) -> tuple[list[dict], str]:
    files = []
    checkpoint_type = "file" if path.is_file() else "directory"
    candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in candidates:
        rel = Path(file.name) if path.is_file() else file.relative_to(path)
        if any(part in SKIP_DIRS or part.startswith("epoch-") for part in rel.parts):
            continue
        if file.name in SKIP_FILES or file.name.endswith((".tmp", ".lock")):
            continue
        name = f"ckpt__{sid}__{str(rel).replace('/', '__')}"
        _link(file, ASSETS / name)
        files.append({"relative_path": rel.as_posix(), "flat_filename": name})
    if not files:
        raise FileNotFoundError(f"no inference checkpoint files found in {path}")
    return files, checkpoint_type


def _copy_hf_base(model_id: str, revision: str, key: str) -> list[dict]:
    hub = Path.home() / ".cache/huggingface/hub"
    snapshot = hub / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"base model snapshot not cached: {model_id}@{revision}")
    allowed = {".json", ".model", ".txt", ".py", ".tiktoken", ".safetensors"}
    files = []
    for file in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        rel = file.relative_to(snapshot)
        if rel.parts[0] == "onnx" or file.suffix not in allowed:
            continue
        name = f"base__{key}__{str(rel).replace('/', '__')}"
        _link(file, ASSETS / name)
        files.append({"relative_path": rel.as_posix(), "flat_filename": name})
    if not files:
        raise FileNotFoundError(f"no tokenizer/weights found for {model_id}@{revision}")
    return files


def build_assets() -> dict:
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    trust = pd.read_csv(TRUST, dtype=str, keep_default_na=False)
    trust = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    deploy = json.loads((DEPLOY / "deployment_manifest.json").read_text())
    ids = list(deploy["checkpoint_ids"])
    classes = list(map(str, deploy["class_order"]))
    if len(ids) != 78 or not set(ids).issubset(trust.index):
        raise ValueError(f"ORACLE04 expects 78 v025 checkpoints; found {len(ids)}")
    candidates = {}
    bundled_bases = {}
    for sid in ids:
        row = trust.loc[sid]
        source_manifest = ROOT / "artifacts/predictions" / f"{row.experiment_id}_manifest.json"
        if not source_manifest.is_file():
            matches = list((ROOT / "artifacts/runs").rglob(f"{row.experiment_id}_manifest.json"))
            if len(matches) == 1:
                source_manifest = matches[0]
        if not source_manifest.is_file():
            raise FileNotFoundError(f"source manifest missing for {sid}: {source_manifest}")
        manifest = json.loads(source_manifest.read_text())
        path = _checkpoint_path(row, manifest)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {sid} missing: {path}")
        files, kind = _copy_checkpoint(path, sid)
        config = manifest.get("config") or {}
        model = str(manifest["model"])
        base_id = config.get("checkpoint")
        revision = manifest.get("model_revision") or config.get("revision")
        candidates[sid] = {
            "sequence_id": sid, "experiment_id": str(row.experiment_id),
            "model": model, "training_dataset": str(row.training_dataset),
            "checkpoint_type": kind, "files": files,
            "classes": list(map(str, manifest.get("classes") or classes)),
            "config": {k: v for k, v in config.items() if not str(k).startswith("_")},
            "base_model": str(base_id) if base_id else None,
            "base_revision": str(revision) if revision else None,
            "temperature": float(manifest.get("calibration_temperature", 1.0) or 1.0),
        }
    if any(c["classes"] != classes for c in candidates.values()):
        raise ValueError("checkpoint class order differs from the ORACLE04 canonical order")

    # Bundle the model bases without an exact, previously validated Kaggle
    # Model source. The remaining large model bases are attached as read-only
    # model or dataset inputs.
    for model in ("granite4_h_micro", "granite4_350m", "smollm3_3b"):
        rows = [c for c in candidates.values() if c["model"] == model]
        if not rows:
            continue
        model_id, revision = rows[0]["base_model"], rows[0]["base_revision"]
        if not model_id or not revision:
            raise ValueError(f"missing base ID/revision for {model}")
        if any((c["base_model"], c["base_revision"]) != (model_id, revision) for c in rows):
            raise ValueError(f"inconsistent base revisions for {model}")
        bundled_bases[model] = {"model_id": model_id, "revision": revision,
                                "files": _copy_hf_base(model_id, revision, model)}

    for name in ("oracle02_catboost.cbm", "global_catboost_core.cbm",
                 "global_logistic_core.joblib", "combined_logistic_core.joblib",
                 "group_A_logistic_core.joblib", "group_B_logistic_core.joblib",
                 "deployment_manifest.json"):
        source = DEPLOY / name
        if source.is_file():
            _link(source, ASSETS / f"deployment__{name}")
    for module in sorted((ROOT / "src").glob("*.py")):
        _link(module, ASSETS / f"source__{module.name}")
    _link(ROOT / "scripts/oracle04_kaggle_runtime.py", ASSETS / "source__oracle04_kaggle_runtime.py")
    wheel = next((ROOT / "artifacts/kaggle/zod01-sklearn").glob("scikit_learn-1.9.1-*.whl"), None)
    if wheel is None:
        wheel = next((ROOT / "artifacts/kaggle/zod02-models").glob("scikit_learn-1.9.1-*.whl"), None)
    if wheel is not None:
        _link(wheel, ASSETS / wheel.name)
    bundle = {
        "name": "ORACLE04 dynamic runtime assets", "trust_list_version": "trust-list-2-v025",
        "algorithm": "oracle04 deployed 78-checkpoint meta ensemble",
        "checkpoint_ids": ids, "checkpoint_count": len(ids), "classes": classes,
        "candidates": candidates, "bundled_bases": bundled_bases,
        "deployment_prefix": "deployment__",
        "source_prefix": "source__",
        "archive": "oracle04-assets.tar",
    }
    (ASSETS / "oracle04_assets.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (ASSETS / "dataset-metadata.json").write_text(json.dumps({
        "title": "ORACLE04 Dynamic Inference Assets", "id": "zodrak/oracle04-dynamic-assets",
        "licenses": [{"name": "other"}],
        "subtitle": "Full runtime inference assets for ORACLE04 strategies",
        "description": "78 v025 checkpoints, their metadata, deployment meta-models and inference code. Kaggle notebooks infer over the runtime competition test set.",
    }, indent=2) + "\n")
    archive = ASSETS / "oracle04-assets.tar"
    with tarfile.open(archive, "w") as tf:
        for file in sorted(ASSETS.iterdir()):
            if file.is_file() and file.name not in {
                "oracle04-assets.tar", "oracle04_assets.json", "dataset-metadata.json"
            }:
                # Kaggle expands uploaded archives into the dataset. Keep all
                # extracted files under one directory so the expanded dataset
                # still respects its 50 top-level-file limit.
                tf.add(file, arcname=f"oracle04_assets/{file.name}", recursive=False)
    # Keep the dataset root below Kaggle's 50 top-level file limit. The runtime
    # extracts individual members from this uncompressed tar as they are used.
    for file in list(ASSETS.iterdir()):
        if file.is_file() and file.name not in {
            "oracle04-assets.tar", "oracle04_assets.json", "dataset-metadata.json"
        }:
            file.unlink()
    return bundle


def build_notebooks() -> dict:
    KERNELS.mkdir(parents=True, exist_ok=True)
    result = {}
    for strategy, (slug, title) in STRATEGIES.items():
        folder = KERNELS / strategy
        folder.mkdir(parents=True, exist_ok=True)
        notebook = nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(
                f"# {title}\n\nThis notebook runs ORACLE04 against every row in Kaggle's runtime test set. It generates probabilities from all 78 frozen Trust List 2 v025 checkpoints; it does not reuse public-test predictions or assume fixed test IDs."
            ),
            nbf.v4.new_code_cell(f'''from pathlib import Path
import json, os, subprocess, sys, tarfile
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
assert torch.cuda.is_available(), "Kaggle GPU is required"
assets_manifests = list(Path("/kaggle/input").rglob("oracle04_assets.json"))
assert len(assets_manifests) == 1, f"expected one ORACLE04 asset manifest, found {{assets_manifests}}"
ASSETS = assets_manifests[0].parent
from importlib.metadata import version
if version("scikit-learn") != "1.9.1":
    wheels = list(Path("/kaggle/input").rglob("scikit_learn-1.9.1-*.whl"))
    assert wheels, "scikit-learn 1.9.1 wheel is missing"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheels[0])], check=True)
runtime_module = Path("/kaggle/working/oracle04_kaggle_runtime.py")
expanded_runtime = ASSETS / "oracle04_assets/source__oracle04_kaggle_runtime.py"
if expanded_runtime.is_file():
    runtime_module.write_bytes(expanded_runtime.read_bytes())
else:
    with tarfile.open(ASSETS / "oracle04-assets.tar", "r:") as archive:
        runtime_module.write_bytes(archive.extractfile("oracle04_assets/source__oracle04_kaggle_runtime.py").read())
sys.path.insert(0, str(runtime_module.parent))
from oracle04_kaggle_runtime import run
result = run({strategy!r}, ASSETS, Path("/kaggle/working/submission.parquet"))
print(json.dumps(result, indent=2))
''')
        ])
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        nbf.write(notebook, folder / "submission.ipynb")
        metadata = {
            "id": f"zodrak/{slug}", "title": title, "code_file": "submission.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": ["oracle04", "dynamic-inference", "trust-list-2-v025"],
            "dataset_sources": ["zodrak/oracle04-dynamic-assets",
                                "zodrak/zod03-llama-base", "pingjiale/qwen3-0-6b"],
            "kernel_sources": [], "competition_sources": ["name-that-llm"],
            "model_sources": MODEL_SOURCES, "machine_shape": "NvidiaTeslaT4",
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        result[strategy] = {"kernel": metadata["id"], "folder": str(folder.relative_to(ROOT))}
    return result


def build() -> dict:
    bundle = build_assets()
    notebooks = build_notebooks()
    size = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file())
    report = {
        "status": "ready_to_publish", "root_cause": "The former kernels exported frozen local predictions for 3 public test IDs. Kaggle's hidden rerun uses a different test set.",
        "checkpoint_count": len(bundle["checkpoint_ids"]), "asset_dir": str(ASSETS.relative_to(ROOT)),
        "asset_size_gib": round(size / 1024**3, 2), "kernels": notebooks,
        "dynamic_test_inference": True,
    }
    (ROOT / "artifacts/kaggle/oracle04_dynamic_package_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
