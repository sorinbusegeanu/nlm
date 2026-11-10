"""Package the V005 one-rule-per-label cascade as a separate Kaggle notebook."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRUST_PATH = ROOT / "docs/checkpoint_label_trust_list.csv"
CATALOG_PATH = ROOT / "docs/checkpoint_label_catalog.csv"
REFERENCE_PACKAGE = ROOT / "artifacts/kaggle/zod03-models"
REFERENCE_MANIFEST = REFERENCE_PACKAGE / "zod03_manifest.json"
OUTPUT_DATA = ROOT / "artifacts/kaggle/zod04-models"
OUTPUT_KERNEL = ROOT / "artifacts/kaggle/zod04-kernel"
OUTPUT_NOTEBOOK = ROOT / "notebooks/zod04.ipynb"
FALLBACK_SEQUENCE = "N0089"
SKIP_NAMES = {
    "README.md", "checkpoint_selection.json", "baseline_validation.json",
    "training_log.json", "training_progress.json", "validation_metrics.json",
}


def _copy_checkpoint(source: Path, sequence_id: str) -> list[dict]:
    destination = OUTPUT_DATA / f"ckpt__{sequence_id}"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    if source.is_file():
        flat_name = f"{destination.name}__{source.name}"
        shutil.copy2(source, OUTPUT_DATA / flat_name)
        return [{"relative_path": source.name, "flat_filename": flat_name}]
    for file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = file.relative_to(source)
        if any(part.startswith("epoch-") for part in relative.parts) or file.name in SKIP_NAMES:
            continue
        flat_name = f"{destination.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, OUTPUT_DATA / flat_name)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": flat_name})
    if not copied:
        raise FileNotFoundError(f"no inference files copied from {source}")
    return copied


def _checkpoint_path(candidate_id: str, manifest: dict, catalog_row: pd.Series) -> Path:
    raw = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path") or catalog_row["checkpoint_path"]
    source = Path(raw)
    source = source if source.is_absolute() else ROOT / source
    if source.is_dir() and (source / "best_checkpoint").is_dir():
        source = source / "best_checkpoint"
    elif source.is_dir() and manifest.get("selected_epoch"):
        selected = source / f"epoch-{int(manifest['selected_epoch']):02d}"
        if selected.is_dir():
            source = selected
    if not source.exists():
        raise FileNotFoundError(f"checkpoint missing for {candidate_id}: {source}")
    return source


def _selected_rules(trust: pd.DataFrame) -> tuple[list[dict], str, list[str]]:
    if trust.trust_list_version.nunique() != 1 or trust.trust_list_version.iloc[0] != "trust-v005":
        raise ValueError("ZOD04 must be built from exactly one trust-v005 snapshot")
    trust["SeqID"] = pd.to_numeric(trust.SeqID, errors="raise")
    trust = trust.sort_values("SeqID", kind="stable")
    chosen_labels: set[str] = set()
    records = []
    for row in trust.to_dict("records"):
        label = str(row["target_label"])
        if label in chosen_labels or str(row["rule_eligible"]).lower() not in {"true", "1"}:
            continue
        chosen_labels.add(label)
        records.append({
            "order": len(records) + 1,
            "trust_sequence": int(row["SeqID"]),
            "sequence_id": str(row["sequence_id"]),
            "candidate_id": str(row["experiment_id"]),
            "label": label,
            "precision_v0": float(row["precision"]),
            "prediction_count_v0": int(row["prediction_count"]),
        })
    if len(records) != 18:
        raise ValueError(f"expected one eligible rule for each of 18 labels; got {len(records)}")

    from src.zod01_fixed_cascade import _resolve_fallback
    fallback_id = str(_resolve_fallback(str(ROOT / "experiments/registry.sqlite"))["experiment_id"])
    catalog = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)
    catalog = catalog.drop_duplicates("experiment_id").set_index("experiment_id")
    fallback_sequence = str(catalog.loc[fallback_id, "sequence_id"])
    if fallback_sequence != FALLBACK_SEQUENCE:
        raise ValueError(f"expected the established fallback {FALLBACK_SEQUENCE}, got {fallback_sequence}")
    candidate_ids = list(dict.fromkeys([record["candidate_id"] for record in records] + [fallback_id]))
    return records, fallback_id, candidate_ids


def build_dataset_package() -> Path:
    if OUTPUT_DATA.exists():
        shutil.rmtree(OUTPUT_DATA)
    OUTPUT_DATA.mkdir(parents=True)

    trust = pd.read_csv(TRUST_PATH, dtype=str, keep_default_na=False)
    rules, fallback_id, candidate_ids = _selected_rules(trust)
    catalog = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)
    catalog = catalog.drop_duplicates("experiment_id").set_index("experiment_id")
    reference = json.loads(REFERENCE_MANIFEST.read_text())
    reference_candidates = reference["candidates"]
    reference_bases: dict[str, dict] = {}
    for candidate in reference_candidates.values():
        base_name = candidate.get("base_model")
        if base_name and candidate.get("base_files"):
            key = (base_name, candidate.get("base_model_revision"))
            reference_bases[str(key)] = candidate

    candidates = {}
    reference_by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in reference_candidates.values()
    }
    for candidate_id in candidate_ids:
        prediction_manifest = ROOT / "artifacts/predictions" / f"{candidate_id}_manifest.json"
        manifest = json.loads(prediction_manifest.read_text())
        catalog_row = catalog.loc[candidate_id]
        sequence_id = str(catalog_row["sequence_id"])
        source = _checkpoint_path(candidate_id, manifest, catalog_row)
        prior_asset = reference_by_id.get(candidate_id)
        if prior_asset:
            # Reuse the immutable ZOD03 asset dataset instead of uploading the
            # same large checkpoints again. Only newly selected checkpoints
            # belong in the ZOD04 dataset.
            files = prior_asset["files"]
            asset_source = "zod03-models"
        else:
            files = _copy_checkpoint(source, sequence_id)
            asset_source = "zod04-models"
        model = str(manifest["model"])
        config = manifest.get("config", {})
        base_model = config.get("checkpoint")
        revision = manifest.get("model_revision")
        base_source = None
        if model in {"granite4_350m", "granite4_h_micro"}:
            key = str((base_model, revision))
            if key not in reference_bases:
                raise ValueError(f"the existing ZOD03 package lacks exact base assets for {key}")
            base_source = "zod03-models"
        elif model == "llama32_3b":
            base_source = "zod03-llama-base"
        elif model == "qwen06":
            base_source = "qwen3-0-6b"
        elif model in {"gemma4_e2b", "embeddinggemma2", "ministral3_3b"}:
            base_source = "versioned_kaggle_model"
        elif model != "ultrabert380" and base_model:
            raise ValueError(f"no Kaggle base source configured for {model}: {base_model}")

        candidates[sequence_id] = {
            "candidate_id": candidate_id,
            "sequence_id": sequence_id,
            "model": model,
            "checkpoint_type": "file" if source.is_file() else "directory",
            "files": files,
            "asset_source": asset_source,
            "classes": list(map(str, manifest["classes"])),
            "config": config,
            "base_model": base_model,
            "base_model_revision": revision,
            "base_source": base_source,
            "training_dataset": str(catalog_row["training_dataset"]),
            "checkpoint_path": str(source.relative_to(ROOT)),
        }

    classes = next(iter(candidates.values()))["classes"]
    if any(candidate["classes"] != classes for candidate in candidates.values()):
        raise ValueError("selected candidate checkpoints disagree on canonical class ordering")
    if set(record["label"] for record in rules) != set(classes):
        raise ValueError("selected rules do not cover the canonical 18 labels")

    # Save the exact trust snapshot used so later trust-list refreshes cannot alter ZOD04.
    shutil.copy2(TRUST_PATH, OUTPUT_DATA / "trust-v005.csv")
    (OUTPUT_DATA / "zod04_manifest.json").write_text(json.dumps({
        "name": "zod04",
        "algorithm": "first_eligible_precision_ranked_rule_per_label_then_fallback",
        "trust_list_version": str(trust.trust_list_version.iloc[0]),
        "trust_list_generated_at_utc": str(trust.trust_list_generated_at_utc.iloc[0]),
        "rules": rules,
        "fallback_sequence_id": str(catalog.loc[fallback_id, "sequence_id"]),
        "fallback_candidate_id": fallback_id,
        "candidates": candidates,
        "classes": classes,
        "submission_schema": ["id", "model"],
        "development_metric": "342/540 (63.33%) on frozen V0 validation; rules were ranked on this same set",
        "rule_source": "docs/checkpoint_label_trust_list.csv",
        "fallback_source": "experiments/registry.sqlite best V1-all to V0 UltraBERT checkpoint",
        "validation_selection_note": "V0 is a development diagnostic; the trust list was ranked using these validation predictions.",
    }, indent=2) + "\n")

    for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / module, OUTPUT_DATA / f"source__{module}")
    wheel = next(REFERENCE_PACKAGE.glob("scikit_learn-1.9.1-*.whl"), None)
    if wheel is None:
        raise FileNotFoundError("scikit-learn 1.9.1 wheel missing from ZOD03 inference assets")
    shutil.copy2(wheel, OUTPUT_DATA / wheel.name)
    (OUTPUT_DATA / "dataset-metadata.json").write_text(json.dumps({
        "title": "ZOD04 one-rule-per-label cascade assets",
        "id": "zodrak/zod04-models",
        "licenses": [{"name": "other"}],
        "subtitle": "Frozen V005 first-rule specialists and fallback for Name That LLM",
        "description": "Inference-only checkpoints selected from trust-v005. Exact Granite and Llama bases are mounted from the ZOD03 assets; remaining bases are attached as pinned Kaggle model or dataset sources.",
    }, indent=2) + "\n")
    return OUTPUT_DATA


def build_notebook() -> Path:
    import nbformat as nbf

    OUTPUT_KERNEL.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "zodrak/zod04", "title": "zod04", "code_file": "zod04.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["offline-inference", "zod04", "one-rule-per-label", "checkpoint-cascade"],
        "dataset_sources": ["zodrak/zod04-models", "zodrak/zod03-models", "zodrak/zod03-llama-base", "pingjiale/qwen3-0-6b"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": [
            "google/gemma-4/transformers/gemma-4-e2b/1",
            "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
        ],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUTPUT_KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    intro = """# ZOD04 — One ranked rule per label plus fallback

This notebook uses the first eligible trust-v005 rule for each of the 18 labels, then predicts remaining rows with the V1-all → V0 UltraBERT fallback (N0089). On the frozen V0 development set this produced 342/540 (63.33%); this is not an independent generalization estimate. It caches each selected checkpoint's top-1 predictions once and writes `submission.parquet` with exactly `id,model`. Internet is disabled; model bases are attached as Kaggle sources.
"""

    setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
INPUT_ROOT = Path("/kaggle/input")
asset_manifests = list(INPUT_ROOT.rglob("zod04_manifest.json"))
assert len(asset_manifests) == 1, f"expected one ZOD04 asset manifest; found {asset_manifests}"
DATA = asset_manifests[0].parent
BUNDLE = json.loads((DATA / "zod04_manifest.json").read_text())
assert BUNDLE["trust_list_version"] == "trust-v005"
assert BUNDLE["algorithm"] == "first_eligible_precision_ranked_rule_per_label_then_fallback"
assert len(BUNDLE["rules"]) == 18 and len(BUNDLE["candidates"]) == 16
from importlib.metadata import version
SKLEARN_VERSION = "1.9.1"
if version("scikit-learn") != SKLEARN_VERSION:
    wheel = next(INPUT_ROOT.rglob("scikit_learn-1.9.1-*.whl"), None)
    assert wheel is not None, "pinned scikit-learn wheel missing"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], check=True)
import joblib, numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "Kaggle GPU unavailable"
WORK = Path("/kaggle/working/zod04_runtime")
SRC = WORK / "src"
SRC.mkdir(parents=True, exist_ok=True)
for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
    (SRC / name).symlink_to(DATA / f"source__{name}")
sys.path.insert(0, str(WORK))
CHECKPOINTS = WORK / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
zod03_manifests = list(INPUT_ROOT.rglob("zod03_manifest.json"))
assert len(zod03_manifests) == 1, f"expected one ZOD03 base manifest; found {zod03_manifests}"
ZOD03_DATA = zod03_manifests[0].parent
ZOD03 = json.loads(zod03_manifests[0].read_text())
for sequence_id, candidate in BUNDLE["candidates"].items():
    dest = CHECKPOINTS / sequence_id
    dest.mkdir(parents=True, exist_ok=True)
    source_root = ZOD03_DATA if candidate["asset_source"] == "zod03-models" else DATA
    for item in candidate["files"]:
        source = source_root / item["flat_filename"]
        assert source.is_file(), f"checkpoint asset missing: {sequence_id}/{item['flat_filename']}"
        target = dest / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    candidate["runtime_checkpoint"] = str(dest if candidate["checkpoint_type"] == "directory" else dest / candidate["files"][0]["relative_path"])

BASES = {}
def resolve_model(marker):
    matches = [p.parent for p in INPUT_ROOT.rglob("config.json") if marker.lower() in str(p.parent).lower()]
    assert matches, f"no attached model found for {marker}"
    # Some sentence-transformer sources contain nested pooling/normalization
    # configs. The model root is the shallowest matching config directory.
    depth = min(len(path.parts) for path in matches)
    roots = sorted({path for path in matches if len(path.parts) == depth})
    assert len(roots) == 1, f"expected one root model for {marker}; found {roots}"
    return roots[0]
for model, marker in (("gemma4_e2b", "gemma-4-e2b"), ("embeddinggemma2", "embeddinggemma-2"),
                      ("ministral3_3b", "ministral-3-3b-base-2512"), ("qwen06", "qwen3-0-6b")):
    if model in {c["model"] for c in BUNDLE["candidates"].values()}:
        BASES[model] = resolve_model(marker)
for model in ("granite4_350m", "granite4_h_micro"):
    candidate = next(c for c in BUNDLE["candidates"].values() if c["model"] == model)
    ref = next((c for c in ZOD03["candidates"].values()
                if c.get("base_model") == candidate["base_model"]
                and c.get("base_model_revision") == candidate["base_model_revision"]
                and c.get("base_files")), None)
    assert ref is not None, f"exact Granite base not found in ZOD03 assets: {model}"
    dest = WORK / "bases" / model
    dest.mkdir(parents=True, exist_ok=True)
    for item in ref["base_files"]:
        source = ZOD03_DATA / item["flat_filename"]
        target = dest / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    BASES[model] = dest
llama_manifests = list(INPUT_ROOT.rglob("llama_base_manifest.json"))
assert len(llama_manifests) == 1, f"expected one Llama base manifest; found {llama_manifests}"
if "llama32_3b" in {c["model"] for c in BUNDLE["candidates"].values()}:
    info = json.loads(llama_manifests[0].read_text())
    candidate = next(c for c in BUNDLE["candidates"].values() if c["model"] == "llama32_3b")
    assert info["model"] == candidate["base_model"] and info["revision"] == candidate["base_model_revision"]
    dest = WORK / "bases" / "llama32_3b"
    dest.mkdir(parents=True, exist_ok=True)
    for item in info["files"]:
        source = llama_manifests[0].parent / item["flat_filename"]
        target = dest / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    BASES["llama32_3b"] = dest
try:
    from importlib.metadata import PackageNotFoundError
    from packaging.version import Version
    if Version(version("torchao")) <= Version("0.16.0"):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True, capture_output=True)
except (PackageNotFoundError, ImportError):
    pass
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
TEST = next((p for p in (INPUT_ROOT / "competitions/name-that-llm/test.csv", INPUT_ROOT / "name-that-llm/test.csv") if p.is_file()), None)
assert TEST is not None, "competition test.csv missing"
print({"algorithm": BUNDLE["algorithm"], "trust_version": BUNDLE["trust_list_version"],
       "rules": len(BUNDLE["rules"]), "checkpoints": len(BUNDLE["candidates"]),
       "fallback": BUNDLE["fallback_sequence_id"]})'''

    inference = r'''test = pd.read_csv(TEST, dtype={"id": "string"})
assert {"id", "text"}.issubset(test.columns)
assert test.id.notna().all() and not test.id.duplicated().any()
test.text = test.text.fillna("").astype(str)
classes = list(map(str, BUNDLE["classes"]))
texts, n = test.text.tolist(), len(test)
def get_top1(sequence_id):
    candidate = BUNDLE["candidates"][sequence_id]
    model, checkpoint = candidate["model"], Path(candidate["runtime_checkpoint"])
    if model == "svc":
        probabilities, labels = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probabilities, labels = predict_catboost_stylometry(checkpoint, texts, classes)
    else:
        base = BASES.get(model)
        if model != "ultrabert380" and candidate.get("base_model"):
            assert base is not None, f"base model was not resolved for {model}"
        batch_size = min(4, max(1, int(candidate["config"].get("batch_size", 4) or 4)))
        while True:
            try:
                probabilities, labels = _predict_neural(
                    model, checkpoint, texts, classes, batch_size=batch_size,
                    max_length=int(candidate["config"].get("max_length", 384)),
                    base_checkpoint=str(base) if base else None, revision=None,
                    text_only_backbone=bool(candidate["config"].get("text_only_backbone", False)))
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                gc.collect(); torch.cuda.empty_cache()
    labels = list(map(str, labels))
    class_order = [labels.index(label) for label in classes]
    prediction = np.asarray(classes, dtype=object)[np.asarray(probabilities)[:, class_order].argmax(axis=1)]
    del probabilities; gc.collect(); torch.cuda.empty_cache()
    return prediction

started = time.time()
checkpoint_order = list(dict.fromkeys([r["sequence_id"] for r in BUNDLE["rules"]] + [BUNDLE["fallback_sequence_id"]]))
assert set(checkpoint_order) == set(BUNDLE["candidates"])
top1_cache = {sequence_id: get_top1(sequence_id) for sequence_id in checkpoint_order}
pending = np.ones(n, dtype=bool)
output = np.empty(n, dtype=object)
source = np.empty(n, dtype=object)
trace = []
for rule in BUNDLE["rules"]:
    sequence_id, label = rule["sequence_id"], rule["label"]
    selected = pending & (top1_cache[sequence_id] == label)
    output[selected], source[selected], pending[selected] = label, sequence_id, False
    trace.append({"order": rule["order"], "trust_sequence": rule["trust_sequence"],
        "sequence_id": sequence_id, "label": label,
        "precision_v0": rule["precision_v0"], "assigned": int(selected.sum()),
        "remaining": int(pending.sum())})
remaining = np.flatnonzero(pending)
fallback = BUNDLE["fallback_sequence_id"]
output[remaining], source[remaining] = top1_cache[fallback][remaining], fallback
submission = pd.DataFrame({"id": test.id, "model": output})
assert submission.columns.tolist() == ["id", "model"] and len(submission) == len(test)
assert submission.id.astype(str).tolist() == test.id.astype(str).tolist()
assert not submission.id.duplicated().any() and submission.id.notna().all()
assert submission.model.notna().all() and submission.model.isin(classes).all()
submission.to_parquet("/kaggle/working/submission.parquet", index=False)
pd.DataFrame(trace).to_csv("/kaggle/working/zod04_rule_counts.csv", index=False)
pd.DataFrame({"id": test.id, "model": submission.model,
              "selected_sequence_id": source}).to_parquet("/kaggle/working/zod04_decisions.parquet", index=False)
print({"approach": "zod04 one eligible rule per label plus fallback", "rows": len(submission),
       "runtime_seconds": round(time.time() - started, 1), "schema": submission.columns.tolist(),
       "cached_checkpoints": len(top1_cache), "output": "/kaggle/working/submission.parquet"})
display(pd.DataFrame(trace)); display(submission.head())'''

    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell(inference),
    ])
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    notebook.metadata["kaggle"] = {"accelerator": "GPU", "dataSources": [], "isInternetEnabled": False}
    nbf.write(notebook, OUTPUT_NOTEBOOK)
    shutil.copy2(OUTPUT_NOTEBOOK, OUTPUT_KERNEL / "zod04.ipynb")
    return OUTPUT_NOTEBOOK


def main() -> None:
    data = build_dataset_package()
    notebook = build_notebook()
    size = sum(path.stat().st_size for path in data.rglob("*") if path.is_file())
    print(json.dumps({"dataset_dir": str(data), "dataset_bytes": size,
                      "notebook": str(notebook),
                      "kernel_metadata": str(OUTPUT_KERNEL / "kernel-metadata.json")}, indent=2))


if __name__ == "__main__":
    main()
