"""Package the frozen TRUST5 ensemble as a Name That LLM Kaggle kernel."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "artifacts/kaggle/trust2-v013-top5-models"
KERNEL_DIR = ROOT / "artifacts/kaggle/trust2-v013-top5-kernel"
NOTEBOOK = ROOT / "notebooks/trust2_v013_top5.ipynb"
TRUST = ROOT / "docs/trust-list-2-v013.csv"
SELECTION = ROOT / "artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/selected_checkpoints.json"
AUDIT = ROOT / "artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/checkpoint_training_overlap_audit.csv"
EXPECTED = ["N0071", "N0056", "N0005", "N0058", "N0072"]
SKIP_FILES = {"README.md", "checkpoint_selection.json", "baseline_validation.json",
              "training_log.json", "training_progress.json", "validation_metrics.json"}
MODEL_SOURCE = "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1"


def _copy_checkpoint(source: Path, sequence_id: str) -> list[dict]:
    destination = ASSET_DIR / f"ckpt__{sequence_id}"
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        flat = f"{destination.name}__{source.name}"
        shutil.copy2(source, ASSET_DIR / flat)
        return [{"relative_path": source.name, "flat_filename": flat}]
    copied = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        if any(part.startswith("epoch-") for part in relative.parts) or path.name in SKIP_FILES:
            continue
        flat = f"{destination.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(path, ASSET_DIR / flat)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": flat})
    if not copied:
        raise FileNotFoundError(f"no inference files found in {source}")
    return copied


def build_assets() -> Path:
    if ASSET_DIR.exists():
        raise FileExistsError(f"refusing to replace existing Kaggle assets directory: {ASSET_DIR}")
    ASSET_DIR.mkdir(parents=True)
    selection = json.loads(SELECTION.read_text())
    selected = selection["full_v0_selection"]["selected_by_size"]["5"]
    sequence_order = [str(row["sequence_id"]) for row in selected]
    if sequence_order != EXPECTED:
        raise ValueError(f"V013 frozen top five changed: {sequence_order}")
    trust = pd.read_csv(TRUST, dtype={"sequence_id": str})
    audit = pd.read_csv(AUDIT, dtype={"sequence_id": str}).set_index("sequence_id")
    zod03 = json.loads((ROOT / "artifacts/kaggle/zod03-models/zod03_manifest.json").read_text())
    if set(trust.trust_list_version.astype(str)) != {"trust-list-2-v013"}:
        raise ValueError("only the v013 Trust List is allowed")
    members = (trust.loc[trust.sequence_id.isin(EXPECTED)]
               .drop_duplicates("sequence_id").set_index("sequence_id").loc[EXPECTED])
    checkpoint_candidates = {}
    for sequence_id in EXPECTED:
        row = members.loc[sequence_id]
        manifest_path = ROOT / "artifacts/predictions" / f"{row.experiment_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(str(row.checkpoint_path))
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"selected checkpoint missing: {sequence_id} {checkpoint}")
        audit_row = audit.loc[sequence_id]
        if int(audit_row.v0_training_id_overlap) or int(audit_row.v0_training_normalized_text_overlap):
            raise ValueError(f"V0 overlap audit failed for {sequence_id}")
        model = str(manifest["model"])
        config = manifest.get("config", {})
        base_files = []
        base_source = None
        if model == "granite4_h_micro":
            base_source = "zod03-models"
            base_candidate = next((item for item in zod03["candidates"].values()
                                   if item.get("base_model") == config.get("checkpoint")
                                   and item.get("base_model_revision") == manifest.get("model_revision")
                                   and item.get("base_files")), None)
            if base_candidate is None:
                raise ValueError(f"exact Granite base not present in existing ZOD03 assets for {sequence_id}")
            base_files = base_candidate["base_files"]
        elif model == "ministral3_3b":
            base_source = "mistral_model_source"
        checkpoint_candidates[sequence_id] = {
            "sequence_id": sequence_id,
            "experiment_id": str(row.experiment_id),
            "model": model,
            "checkpoint_type": "file" if checkpoint.is_file() else "directory",
            "files": _copy_checkpoint(checkpoint, sequence_id),
            "base_model": config.get("checkpoint"),
            "base_model_revision": manifest.get("model_revision"),
            "base_source": base_source,
            "base_files": base_files,
            "config": config,
            "model_revision": manifest.get("model_revision"),
            "classes": list(map(str, manifest["classes"])),
            "calibration_temperature": manifest.get("calibration_temperature", 1.0),
            "training_dataset": str(row.training_dataset),
            "source_manifest": str(manifest_path.relative_to(ROOT)),
            "source_manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
            "audit_status": str(audit_row.audit_status),
            "v0_training_id_overlap": int(audit_row.v0_training_id_overlap),
            "v0_training_normalized_text_overlap": int(audit_row.v0_training_normalized_text_overlap),
        }
    classes = checkpoint_candidates[EXPECTED[0]]["classes"]
    if len(classes) != 18 or any(item["classes"] != classes for item in checkpoint_candidates.values()):
        raise ValueError("selected checkpoints must share the canonical 18-class ordering")

    rules_df = trust.loc[
        trust.sequence_id.isin(EXPECTED)
        & trust.rule_eligible.astype(str).str.lower().isin({"true", "1"})
        & trust.evaluation_eligible.astype(str).str.lower().isin({"true", "1"})
        & pd.to_numeric(trust.validation_training_overlap, errors="coerce").fillna(-1).eq(0)
        & pd.to_numeric(trust.prediction_count, errors="coerce").ge(10)
    ].copy()
    rules = []
    for row in rules_df.to_dict("records"):
        rules.append({
            "sequence_id": str(row["sequence_id"]),
            "experiment_id": str(row["experiment_id"]),
            "label": str(row["target_label"]),
            "precision": float(row["precision"]),
            "prediction_count": int(row["prediction_count"]),
        })
    if len(rules) != 89:
        raise ValueError(f"expected the 89 eligible v013 rules for the frozen five; got {len(rules)}")
    if {rule["label"] for rule in rules} != set(classes):
        raise ValueError("eligible v013 rules do not cover all 18 classes")
    fallback = max(EXPECTED, key=lambda sequence_id: (
        float(members.loc[sequence_id, "overall_accuracy"]), -EXPECTED.index(sequence_id)))
    bundle = {
        "name": "trust2_v013_top5",
        "trust_list_version": "trust-list-2-v013",
        "algorithm": "family_normalized_precision_support_weighted_class_score",
        "sequence_ids": EXPECTED,
        "candidates": checkpoint_candidates,
        "rules": rules,
        "classes": classes,
        "weight_formula": "precision * prediction_count / (prediction_count + 30)",
        "family_normalization": "divide by selected checkpoint count in the candidate model family",
        "tie_break": "canonical class order",
        "fallback_sequence_id": fallback,
        "fallback_rule": "highest recorded full-V0 overall accuracy among selected five; selection order breaks ties; used only if no eligible rule votes",
        "submission_schema": ["id", "model"],
        "full_v0_development_accuracy": 1717 / 2699,
    }
    (ASSET_DIR / "trust2_v013_top5_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / module, ASSET_DIR / f"source__{module}")
    sklearn_wheel = next((ROOT / "artifacts/kaggle/zod03-models").glob("scikit_learn-1.9.1-*.whl"), None)
    if sklearn_wheel is None:
        raise FileNotFoundError("pinned scikit-learn wheel from the established ZOD assets is missing")
    shutil.copy2(sklearn_wheel, ASSET_DIR / sklearn_wheel.name)
    (ASSET_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "Trust List 2 v013 top-five checkpoint assets",
        "id": "zodrak/trust2-v013-top5-models",
        "licenses": [{"name": "other"}],
        "subtitle": "Five frozen audited checkpoints for TRUST5",
        "description": "Inference-only package for the five-checkpoint Trust List 2 v013 ensemble. The exact Granite base is included; Ministral uses the versioned Kaggle model source.",
    }, indent=2) + "\n")
    return ASSET_DIR


def build_notebook() -> tuple[Path, Path]:
    import nbformat as nbf

    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "zodrak/trust2-v013-top5",
        "title": "Trust2 V013 Top5 TRUST5",
        "code_file": "trust2_v013_top5.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["trust-list-2", "v013", "TRUST5", "offline-inference"],
        "dataset_sources": ["zodrak/trust2-v013-top5-models", "zodrak/zod03-models"],
        "kernel_sources": [],
        "competition_sources": ["name-that-llm"],
        "model_sources": [MODEL_SOURCE],
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2) + "\n")

    intro = """# Trust List 2 v013 — TRUST5\n\nThis kernel applies the frozen five-checkpoint TRUST5 score rule. It loads the checkpoints sequentially, writes `submission.parquet` with exactly `id,model`, and does not retrain or reselect models.\n"""
    setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
INPUT_ROOT = Path("/kaggle/input")
matches = list(INPUT_ROOT.rglob("trust2_v013_top5_manifest.json"))
assert len(matches) == 1, f"expected one frozen TRUST5 manifest, got {matches}"
DATA = matches[0].parent
BUNDLE = json.loads(matches[0].read_text())
assert BUNDLE["trust_list_version"] == "trust-list-2-v013"
assert BUNDLE["sequence_ids"] == ["N0071", "N0056", "N0005", "N0058", "N0072"]
assert len(BUNDLE["candidates"]) == 5 and len(BUNDLE["rules"]) == 89
from importlib.metadata import version
SKLEARN_VERSION = "1.9.1"
if version("scikit-learn") != SKLEARN_VERSION:
    wheel = next(INPUT_ROOT.rglob("scikit_learn-1.9.1-*.whl"), None)
    assert wheel is not None, "pinned sklearn wheel missing"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], check=True)
import joblib, numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "Kaggle GPU is unavailable"
WORK = Path("/kaggle/working/trust2_v013_top5_runtime")
SRC = WORK / "src"
SRC.mkdir(parents=True, exist_ok=True)
for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
    (SRC / name).symlink_to(DATA / f"source__{name}")
sys.path.insert(0, str(WORK))
CHECKPOINTS = WORK / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
for sequence_id, candidate in BUNDLE["candidates"].items():
    dest = CHECKPOINTS / sequence_id
    dest.mkdir(parents=True, exist_ok=True)
    for item in candidate["files"]:
        source = DATA / item["flat_filename"]
        assert source.is_file(), f"checkpoint asset missing: {sequence_id}/{item['flat_filename']}"
        target = dest / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    candidate["runtime_checkpoint"] = str(dest if candidate["checkpoint_type"] == "directory" else dest / candidate["files"][0]["relative_path"])

granite_manifest = next(INPUT_ROOT.rglob("zod03_manifest.json"), None)
assert granite_manifest is not None, "the established ZOD03 base assets are not attached"
ZOD03_DATA = granite_manifest.parent
ZOD03 = json.loads(granite_manifest.read_text())
granite_candidate = next(c for c in BUNDLE["candidates"].values() if c["model"] == "granite4_h_micro")
granite_base = next((c for c in ZOD03["candidates"].values()
                     if c.get("base_model") == granite_candidate["base_model"]
                     and c.get("base_model_revision") == granite_candidate["base_model_revision"]
                     and c.get("base_files")), None)
assert granite_base is not None, "exact Granite base revision not found in established ZOD03 assets"
BASES = {}
granite_dir = WORK / "bases" / "granite4_h_micro"
granite_dir.mkdir(parents=True, exist_ok=True)
for item in granite_base["base_files"]:
    source = ZOD03_DATA / item["flat_filename"]
    target = granite_dir / item["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
BASES["granite4_h_micro"] = granite_dir

def resolve_model(marker):
    found = [path.parent for path in INPUT_ROOT.rglob("config.json") if marker.lower() in str(path.parent).lower()]
    assert found, f"attached Kaggle model not found: {marker}"
    depth = min(len(path.parts) for path in found)
    roots = sorted({path for path in found if len(path.parts) == depth})
    assert len(roots) == 1, f"expected one model root for {marker}, found {roots}"
    return roots[0]
BASES["ministral3_3b"] = resolve_model("ministral-3-3b-base-2512")
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
TEST = next((path for path in (INPUT_ROOT / "competitions/name-that-llm/test.csv", INPUT_ROOT / "name-that-llm/test.csv") if path.is_file()), None)
assert TEST is not None, "competition test.csv not found"
'''
    inference = r'''test = pd.read_csv(TEST, dtype={"id": "string"})
assert {"id", "text"}.issubset(test.columns)
assert test.id.notna().all() and not test.id.duplicated().any()
test.text = test.text.fillna("").astype(str)
classes = list(map(str, BUNDLE["classes"]))
texts = test.text.tolist()
def predict_checkpoint(sequence_id):
    candidate = BUNDLE["candidates"][sequence_id]
    model, checkpoint = candidate["model"], Path(candidate["runtime_checkpoint"])
    if model == "svc":
        probabilities, labels = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probabilities, labels = predict_catboost_stylometry(checkpoint, texts, classes,
            temperature=float(candidate.get("calibration_temperature", 1.0)))
    else:
        base = BASES.get(model)
        assert model == "ultrabert380" or base is not None, f"model base not resolved for {model}"
        batch_size = min(4, max(1, int(candidate["config"].get("inference_batch_size", 4) or 4)))
        while True:
            try:
                probabilities, labels = _predict_neural(
                    model, checkpoint, texts, classes, batch_size=batch_size,
                    max_length=int(candidate["config"].get("max_length", 384)),
                    temperature=float(candidate.get("calibration_temperature", 1.0)),
                    base_checkpoint=str(base) if base else candidate.get("base_model"),
                    revision=None if base else candidate.get("model_revision"),
                    fix_mistral_regex=bool(candidate["config"].get("fix_mistral_regex", False)),
                    text_only_backbone=bool(candidate["config"].get("text_only_backbone", False)),
                    quantization=candidate["config"].get("quantization"),
                    attention_backend=candidate["config"].get("attention_backend"))
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                gc.collect(); torch.cuda.empty_cache()
    labels = list(map(str, labels))
    order = [labels.index(label) for label in classes]
    probabilities = np.asarray(probabilities, dtype=np.float64)[:, order]
    assert probabilities.shape == (len(test), 18) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    top1 = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
    cache_dir = Path("/kaggle/working/checkpoint_probabilities")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{sequence_id}.npz", id=test.id.astype(str).to_numpy(), probabilities=probabilities, classes=np.asarray(classes))
    del probabilities
    gc.collect(); torch.cuda.empty_cache()
    return top1

started = time.time()
top1 = {sequence_id: predict_checkpoint(sequence_id) for sequence_id in BUNDLE["sequence_ids"]}
family_by_sequence = {key: value["model"].split("_")[0] for key, value in BUNDLE["candidates"].items()}
# Use the same model-family aliases and family divisors as the local subset implementation.
family_by_sequence.update({"N0005": "catboost", "N0056": "granite", "N0058": "mistral", "N0071": "ultrabert", "N0072": "svc"})
family_counts = {family: sum(value == family for value in family_by_sequence.values()) for family in set(family_by_sequence.values())}
rule_by_key = {(rule["sequence_id"], rule["label"]): rule for rule in BUNDLE["rules"]}
scores = np.zeros((len(test), len(classes)), dtype=np.float64)
label_index = {label: index for index, label in enumerate(classes)}
for sequence_id in BUNDLE["sequence_ids"]:
    family_divisor = family_counts[family_by_sequence[sequence_id]]
    for row_index, label in enumerate(top1[sequence_id]):
        rule = rule_by_key.get((sequence_id, str(label)))
        if rule is None:
            continue
        support = int(rule["prediction_count"])
        contribution = float(rule["precision"]) * support / (support + 30)
        scores[row_index, label_index[str(label)]] += contribution / family_divisor
prediction = np.asarray(classes, dtype=object)[scores.argmax(axis=1)]
no_votes = scores.sum(axis=1) == 0
fallback = BUNDLE["fallback_sequence_id"]
prediction[no_votes] = top1[fallback][no_votes]
submission = pd.DataFrame({"id": test.id.astype(str), "model": prediction.astype(str)})
assert submission.columns.tolist() == ["id", "model"]
assert submission.id.tolist() == test.id.astype(str).tolist()
assert submission.id.nunique() == len(test) and submission.model.notna().all()
assert submission.model.isin(classes).all()
submission.to_parquet("/kaggle/working/submission.parquet", index=False)
print({"algorithm": BUNDLE["algorithm"], "trust_list_version": BUNDLE["trust_list_version"],
       "checkpoint_ids": BUNDLE["sequence_ids"], "rows": len(submission),
       "fallback_rows": int(no_votes.sum()), "runtime_seconds": round(time.time()-started, 1),
       "schema": submission.columns.tolist()})
display(submission.head())
'''
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell(inference),
    ])
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    shutil.copy2(NOTEBOOK, KERNEL_DIR / NOTEBOOK.name)
    return NOTEBOOK, KERNEL_DIR / "kernel-metadata.json"


if __name__ == "__main__":
    assets = build_assets()
    notebook, metadata = build_notebook()
    print(json.dumps({"assets": str(assets), "notebook": str(notebook),
                      "kernel_metadata": str(metadata)}, indent=2))
