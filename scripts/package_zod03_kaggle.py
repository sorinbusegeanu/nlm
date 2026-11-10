"""Package and build the full 110% ZOD03 cascade for Kaggle."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "artifacts/analysis/zod03_110/rule_trace.csv"
TRUST = ROOT / "docs/checkpoint_label_trust_list.csv"
OUTPUT_DATA = ROOT / "artifacts/kaggle/zod03-models"
OUTPUT_LLAMA = ROOT / "artifacts/kaggle/zod03-llama-base"
OUTPUT_KERNEL = ROOT / "artifacts/kaggle/zod03-kernel"
FALLBACK_LABEL = "fallback"
SKIP_NAMES = {"README.md", "checkpoint_selection.json", "baseline_validation.json",
              "training_log.json", "training_progress.json", "validation_metrics.json"}


def applied_rules() -> tuple[pd.DataFrame, str]:
    trace = pd.read_csv(TRACE)
    trust = pd.read_csv(TRUST, dtype=str, keep_default_na=False)
    trust["SeqID"] = pd.to_numeric(trust["SeqID"], errors="raise").astype(int)
    applied = trace[(trace["rank"].astype(str) != "Fallback") & ~trace["skipped"].astype(bool)].copy()
    applied["rank"] = pd.to_numeric(applied["rank"], errors="raise").astype(int)
    fallback = str(trace.loc[trace["rank"].astype(str) == "Fallback", "candidate_id"].iloc[0])
    by_key = trust.set_index(["experiment_id", "target_label"])
    records = []
    for row in applied.sort_values("rank", kind="stable").to_dict("records"):
        key = (str(row["candidate_id"]), str(row["label"]))
        if key not in by_key.index:
            raise ValueError(f"applied ZOD03 rule not found in trust list: {key}")
        trusted = by_key.loc[key]
        if isinstance(trusted, pd.DataFrame):
            trusted = trusted.iloc[0]
        if int(trusted["SeqID"]) != int(row["rank"]):
            raise ValueError(f"ZOD03 trace/trust order mismatch for {key}: {row['rank']} != {trusted['SeqID']}")
        records.append({
            "order": int(row["rank"]), "sequence_id": str(trusted["sequence_id"]),
            "candidate_id": key[0], "label": key[1],
            "precision": float(trusted["precision"]),
            "smoothed_precision": float(trusted["smoothed_precision"]),
            "prediction_count": int(trusted["prediction_count"]),
            "support_growth_threshold": 1.10, "max_rules_per_label": 2,
        })
    # The frozen rule set must be exactly the applied rows from the saved 110% run.
    if len(records) != 36:
        raise ValueError(f"expected 36 applied ZOD03-110 rules, found {len(records)}")
    return pd.DataFrame(records), fallback


def _copy_checkpoint(source: Path, sequence_id: str) -> list[dict]:
    destination = OUTPUT_DATA / f"ckpt__{sequence_id}"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    if source.is_file():
        name = f"{destination.name}__{source.name}"
        shutil.copy2(source, OUTPUT_DATA / name)
        return [{"relative_path": source.name, "flat_filename": name}]
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = file.relative_to(source)
        if any(part.startswith("epoch-") for part in relative.parts) or file.name in SKIP_NAMES:
            continue
        name = f"{destination.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, OUTPUT_DATA / name)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": name})
    if not copied:
        raise FileNotFoundError(f"no inference files copied from {source}")
    return copied


def _copy_hf_base(model_id: str, revision: str, prefix: str, output_dir: Path = OUTPUT_DATA) -> list[dict]:
    cache = Path.home() / ".cache/huggingface/hub"
    snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing cached base revision {model_id}@{revision}")
    allowed = {".json", ".model", ".txt", ".py", ".tiktoken", ".safetensors"}
    copied = []
    for file in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        relative = file.relative_to(snapshot)
        if relative.parts[0] == "onnx" or file.suffix not in allowed:
            continue
        name = f"base__{prefix}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, output_dir / name)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": name})
    if not copied:
        raise FileNotFoundError(f"no usable base files for {model_id}@{revision}")
    return copied


def build_dataset_package() -> Path:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DATA.iterdir():
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    rules, fallback_id = applied_rules()
    ordered_ids = list(dict.fromkeys([*rules.candidate_id.astype(str), fallback_id]))
    catalog = pd.read_csv(ROOT / "docs/checkpoint_label_catalog.csv", dtype=str, keep_default_na=False)
    catalog = catalog.drop_duplicates("experiment_id").set_index("experiment_id")
    candidates: dict[str, dict] = {}
    base_revisions: dict[str, str] = {}
    for candidate_id in ordered_ids:
        manifest_path = ROOT / "artifacts/predictions" / f"{candidate_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        catalog_row = catalog.loc[candidate_id]
        source_value = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path") or catalog_row["checkpoint_path"]
        source = Path(source_value)
        source = source if source.is_absolute() else ROOT / source
        if source.is_dir() and (source / "best_checkpoint").is_dir():
            source = source / "best_checkpoint"
        elif source.is_dir() and manifest.get("selected_epoch"):
            chosen = source / f"epoch-{int(manifest['selected_epoch']):02d}"
            if chosen.is_dir():
                source = chosen
        if not source.exists():
            raise FileNotFoundError(f"checkpoint missing: {candidate_id}: {source}")
        sequence_id = str(catalog_row["sequence_id"])
        model = str(manifest["model"])
        files = _copy_checkpoint(source, sequence_id)
        config = manifest.get("config", {})
        base_model = config.get("checkpoint")
        base_files = []
        if base_model in {"ibm-granite/granite-4.0-350m-base", "ibm-granite/granite-4.0-h-micro-base"}:
            revision = str(manifest.get("model_revision") or "")
            if not revision:
                raise ValueError(f"missing immutable base revision in manifest: {candidate_id}")
            if base_model not in base_revisions:
                base_revisions[base_model] = revision
                base_files = _copy_hf_base(base_model, revision, model)
            elif base_revisions[base_model] != revision:
                raise ValueError(f"different saved revisions for {base_model}")
        candidates[sequence_id] = {
            "candidate_id": candidate_id, "sequence_id": sequence_id, "model": model,
            "checkpoint_type": "file" if source.is_file() else "directory", "files": files,
            "classes": list(map(str, manifest["classes"])), "config": config,
            "base_model": base_model, "base_files": base_files,
            "base_model_revision": manifest.get("model_revision"),
            "training_dataset": str(catalog_row["training_dataset"]),
            "validation_accuracy": manifest.get("metrics", {}).get("accuracy"),
            "selected_epoch": manifest.get("selected_epoch"),
        }

    fallback_seq = str(catalog.loc[fallback_id, "sequence_id"])
    rules_json = rules.to_dict("records")
    classes = next(iter(candidates.values()))["classes"]
    if any(candidate["classes"] != classes for candidate in candidates.values()):
        raise ValueError("candidate checkpoints do not share a canonical class ordering")
    bundle = {
        "name": "zod03", "algorithm": "zod03_110_full_cascade",
        "rules": rules_json, "fallback_sequence_id": fallback_seq,
        "fallback_candidate_id": fallback_id, "candidates": candidates, "classes": classes,
        "base_model_revisions": base_revisions,
        "submission_schema": ["id", "model"],
        "rule_source": str(TRACE.relative_to(ROOT)), "trust_source": str(TRUST.relative_to(ROOT)),
        "support_growth_threshold": 1.10, "max_rules_per_label": 2,
        "development_metric": "345/540 (63.89%) on frozen V0 validation; rule selection is in-sample",
        "validation_selection_note": "Rules were selected using V0 validation predictions; this is a development-set diagnostic, not an independent generalization estimate.",
    }
    (OUTPUT_DATA / "zod03_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / module, OUTPUT_DATA / f"source__{module}")
    wheel = next((ROOT / "artifacts/kaggle/zod01-sklearn").glob("scikit_learn-1.9.1-*.whl"), None)
    if wheel is None:
        wheel = next((ROOT / "artifacts/kaggle/zod02-models").glob("scikit_learn-1.9.1-*.whl"), None)
    if wheel is None:
        raise FileNotFoundError("scikit-learn 1.9.1 wheel needed for the saved SVC checkpoints")
    shutil.copy2(wheel, OUTPUT_DATA / wheel.name)
    (OUTPUT_DATA / "dataset-metadata.json").write_text(json.dumps({
        "title": "ZOD03 full 110% cascade assets", "id": "zodrak/zod03-models",
        "licenses": [{"name": "other"}],
        "subtitle": "Frozen inference checkpoints for the complete ZOD03 110% cascade",
        "description": "Only the 22 required checkpoints and inference source are included, plus the exact cached Granite bases. Other bases are attached as versioned Kaggle models or the Qwen 0.6B dataset.",
    }, indent=2) + "\n")
    return OUTPUT_DATA


def build_llama_base_package() -> Path:
    """Package the exact gated Llama base separately from checkpoint assets."""
    for path in OUTPUT_LLAMA.iterdir() if OUTPUT_LLAMA.exists() else []:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    OUTPUT_LLAMA.mkdir(parents=True, exist_ok=True)
    model_id, revision = "meta-llama/Llama-3.2-3B", "13afe5124825b4f3751f836b40dafda64c1ed062"
    files = _copy_hf_base(model_id, revision, "llama32_3b", OUTPUT_LLAMA)
    (OUTPUT_LLAMA / "llama_base_manifest.json").write_text(json.dumps({
        "model": model_id, "revision": revision, "files": files,
    }, indent=2) + "\n")
    (OUTPUT_LLAMA / "dataset-metadata.json").write_text(json.dumps({
        "title": "ZOD03 exact Llama 3.2 3B base", "id": "zodrak/zod03-llama-base",
        "licenses": [{"name": "other"}],
        "subtitle": "Exact base revision required by ZOD03 LoRA adapter N0063",
        "description": "Inference-only exact Meta Llama 3.2 3B base weights and tokenizer files. No training data included.",
    }, indent=2) + "\n")
    return OUTPUT_LLAMA


def build_kernel() -> Path:
    import nbformat as nbf

    OUTPUT_KERNEL.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "zodrak/zod03", "title": "zod03", "code_file": "zod03.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["offline-inference", "zod03", "checkpoint-cascade"],
        "dataset_sources": ["zodrak/zod03-models", "zodrak/zod03-llama-base", "pingjiale/qwen3-0-6b"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": [
            "google/gemma-4/transformers/gemma-4-e2b/1",
            "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
            "qwen-lm/qwen-3/transformers/1.7b-base/1",
        ],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUTPUT_KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
INPUT_ROOT = Path("/kaggle/input")
bundle_manifests = list(INPUT_ROOT.rglob("zod03_manifest.json"))
assert len(bundle_manifests) == 1, f"expected one ZOD03 asset manifest; found {bundle_manifests}"
DATA = bundle_manifests[0].parent
BUNDLE = json.loads((DATA / "zod03_manifest.json").read_text())
from importlib.metadata import version
SKLEARN_VERSION = "1.9.1"
if version("scikit-learn") != SKLEARN_VERSION:
    wheel = next(Path("/kaggle/input").rglob("scikit_learn-1.9.1-*.whl"), None)
    assert wheel is not None, "pinned scikit-learn wheel missing"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], check=True)
import joblib, numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "Kaggle GPU unavailable"
WORK = Path("/kaggle/working/zod03_runtime")
SRC = WORK / "src"
SRC.mkdir(parents=True, exist_ok=True)
for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
    (SRC / name).symlink_to(DATA / f"source__{name}")
CHECKPOINTS = WORK / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
BASES = {}
for seq, candidate in BUNDLE["candidates"].items():
    dest = CHECKPOINTS / seq
    dest.mkdir(parents=True, exist_ok=True)
    for item in candidate["files"]:
        source = DATA / item["flat_filename"]
        assert source.is_file(), f"checkpoint asset missing: {seq}/{item['flat_filename']}"
        target = dest / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    candidate["runtime_checkpoint"] = str(dest if candidate["checkpoint_type"] == "directory" else dest / candidate["files"][0]["relative_path"])
    if candidate.get("base_files") and candidate["model"] not in BASES:
        bdir = WORK / "bases" / candidate["model"]
        bdir.mkdir(parents=True, exist_ok=True)
        for item in candidate["base_files"]:
            source = DATA / item["flat_filename"]
            assert source.is_file(), f"base asset missing: {item['flat_filename']}"
            target = bdir / item["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        BASES[candidate["model"]] = bdir
def resolve_model(marker):
    for p in Path("/kaggle/input").rglob("config.json"):
        if marker.lower() in str(p.parent).lower():
            return p.parent
    raise FileNotFoundError(f"model base not found under /kaggle/input: {marker}")
for model, marker in (("qwen06", "qwen3-0.6b"), ("gemma4_e2b", "gemma-4-e2b"),
                      ("embeddinggemma2", "embeddinggemma-2"),
                      ("ministral3_3b", "ministral-3-3b-base-2512"), ("qwen17", "1.7b-base")):
    if model in {c["model"] for c in BUNDLE["candidates"].values()} and model not in BASES:
        BASES[model] = resolve_model(marker)
llama_manifests = list(INPUT_ROOT.rglob("llama_base_manifest.json"))
assert len(llama_manifests) == 1, f"expected one exact Llama base manifest; found {llama_manifests}"
LLAMA_DATA = llama_manifests[0].parent
if "llama32_3b" in {c["model"] for c in BUNDLE["candidates"].values()}:
    llama_info = json.loads((LLAMA_DATA / "llama_base_manifest.json").read_text())
    assert llama_info["model"] == "meta-llama/Llama-3.2-3B"
    assert llama_info["revision"] == BUNDLE["candidates"][next(s for s,c in BUNDLE["candidates"].items() if c["model"] == "llama32_3b")]["base_model_revision"]
    llama_dir = WORK / "bases" / "llama32_3b"
    llama_dir.mkdir(parents=True, exist_ok=True)
    for item in llama_info["files"]:
        source = LLAMA_DATA / item["flat_filename"]
        target = llama_dir / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    BASES["llama32_3b"] = llama_dir
try:
    from importlib.metadata import PackageNotFoundError
    from packaging.version import Version
    if Version(version("torchao")) <= Version("0.16.0"):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True, capture_output=True)
except (PackageNotFoundError, ImportError):
    pass
sys.path.insert(0, str(WORK))
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
TEST = next((p for p in (Path("/kaggle/input/competitions/name-that-llm/test.csv"), Path("/kaggle/input/name-that-llm/test.csv")) if p.is_file()), None)
assert TEST is not None, "competition test.csv missing"
assert BUNDLE["algorithm"] == "zod03_110_full_cascade"
assert len(BUNDLE["rules"]) == 36 and len(BUNDLE["candidates"]) == 22
print({"approach": BUNDLE["algorithm"], "applied_rules": len(BUNDLE["rules"]), "checkpoints": len(BUNDLE["candidates"]), "fallback": BUNDLE["fallback_sequence_id"]})'''
    predict = r'''test = pd.read_csv(TEST, dtype={"id": "string"})
assert {"id", "text"}.issubset(test.columns)
assert test.id.notna().all() and not test.id.duplicated().any()
test.text = test.text.fillna("").astype(str)
classes = list(map(str, BUNDLE["classes"]))
texts, n = test.text.tolist(), len(test)
def get_top1(seq):
    c = BUNDLE["candidates"][seq]
    model, cp = c["model"], Path(c["runtime_checkpoint"])
    if model == "svc":
        probs, labels = predict_svc_bundle(joblib.load(cp), texts)
    elif model == "catboost_stylometry":
        probs, labels = predict_catboost_stylometry(cp, texts, classes)
    else:
        base = BASES.get(model)
        if model != "ultrabert380":
            assert base is not None, f"base model not resolved for {model}"
        bs = min(4, max(1, int(c["config"].get("batch_size", 4) or 4)))
        while True:
            try:
                probs, labels = _predict_neural(model, cp, texts, classes, batch_size=bs,
                    max_length=int(c["config"].get("max_length", 384)),
                    base_checkpoint=str(base) if base else None, revision=None,
                    text_only_backbone=bool(c["config"].get("text_only_backbone", False)))
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or bs <= 1: raise
                bs = max(1, bs // 2); gc.collect(); torch.cuda.empty_cache()
    labels = list(map(str, labels))
    order = [labels.index(label) for label in classes]
    top = np.asarray(classes, dtype=object)[np.asarray(probs)[:, order].argmax(axis=1)]
    del probs; gc.collect(); torch.cuda.empty_cache()
    return top
started = time.time()
# Run each checkpoint once on the entire test set, then cache top-1 labels.
checkpoint_order = list(dict.fromkeys([r["sequence_id"] for r in BUNDLE["rules"]] + [BUNDLE["fallback_sequence_id"]]))
assert set(checkpoint_order) == set(BUNDLE["candidates"])
top1_cache = {seq: get_top1(seq) for seq in checkpoint_order}
assigned, output = np.zeros(n, dtype=bool), np.empty(n, dtype=object)
source = np.empty(n, dtype=object)
trace = []
for rule in BUNDLE["rules"]:
    seq, label = rule["sequence_id"], rule["label"]
    chosen = np.flatnonzero((~assigned) & (top1_cache[seq] == label))
    assigned[chosen], output[chosen], source[chosen] = True, label, seq
    trace.append({"order": rule["order"], "sequence_id": seq, "label": label,
        "precision_v0": rule["precision"], "smoothed_precision_v0": rule["smoothed_precision"],
        "assigned": len(chosen), "remaining": int((~assigned).sum())})
remaining = np.flatnonzero(~assigned)
fallback = BUNDLE["fallback_sequence_id"]
output[remaining], source[remaining] = top1_cache[fallback][remaining], fallback
submission = pd.DataFrame({"id": test.id, "model": output})
assert submission.columns.tolist() == ["id", "model"] and len(submission) == len(test)
assert submission.id.astype(str).tolist() == test.id.astype(str).tolist()
assert not submission.id.duplicated().any() and submission.id.notna().all()
assert submission.model.notna().all() and submission.model.isin(classes).all()
out = Path("/kaggle/working/submission.parquet")
submission.to_parquet(out, index=False)
pd.DataFrame(trace).to_csv("/kaggle/working/zod03_rule_counts.csv", index=False)
pd.DataFrame({"id": test.id, "model": submission.model, "selected_sequence_id": source}).to_parquet("/kaggle/working/zod03_decisions.parquet", index=False)
print({"approach": "zod03 full 110% cascade", "rows": len(submission), "output": str(out),
       "runtime_seconds": round(time.time()-started, 1), "schema": submission.columns.tolist(),
       "cached_checkpoints": len(top1_cache)})
display(pd.DataFrame(trace)); display(submission.head())'''
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# ZOD03 — Full 110% checkpoint cascade\n\nRuns the complete 36-rule cascade selected by the 110% support-growth rule, with the recorded fallback. It caches top-1 predictions from each of the 22 distinct checkpoints over the complete competition test set before routing. The rules and fallback were selected using frozen V0 validation results (345/540, 63.89%), so this is a development-set score, not an independent generalization estimate. Internet is disabled; base models are attached as Kaggle model/data sources."),
        nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(predict)],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "cascade": "zod03_110_full"})
    nbf.write(nb, OUTPUT_KERNEL / "zod03.ipynb")
    local_notebook = ROOT / "notebooks/zod03.ipynb"
    local_notebook.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, local_notebook)
    return OUTPUT_KERNEL


if __name__ == "__main__":
    data = build_dataset_package()
    llama = build_llama_base_package()
    kernel = build_kernel()
    size = sum(p.stat().st_size for p in data.rglob("*") if p.is_file())
    llama_size = sum(p.stat().st_size for p in llama.rglob("*") if p.is_file())
    print(json.dumps({"dataset": str(data), "dataset_bytes": size,
                      "llama_base_dataset": str(llama), "llama_base_bytes": llama_size,
                      "kernel": str(kernel), "rule_count": 36, "candidate_count": 22}, indent=2))
