"""Package EQUAL9 and EQUAL10-Qwen for separate offline Kaggle submissions."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts/kaggle/trust2-v013-equal9-models"
SELECTION = ROOT / "artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013/selected_checkpoints.json"
W6W7 = ROOT / "artifacts/analysis/trust2_w6_w7/evaluation_metrics.json"
SKIP_NAMES = {"README.md", "checkpoint_selection.json", "baseline_validation.json", "training_log.json",
              "training_progress.json", "validation_metrics.json", "optimizer.pt", "scheduler.pt",
              "rng_state.pth", "trainer_state.json", "training_args.bin"}
BASE_IDS = ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
ALL_IDS = BASE_IDS + ["N0074"]
MODEL_SOURCES = [
    "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
    "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
]
DATASET_SOURCES = [
    "zodrak/trust2-v013-equal9-models",
    "johnnyhyland/smollm3-3b-base-36l-text",
    "pingjiale/qwen3-0-6b",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hash(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and
                       not any(part.startswith("epoch-") for part in p.relative_to(path).parts) and
                       p.name not in SKIP_NAMES and "best_checkpoint" not in p.relative_to(path).parts):
        h.update(item.relative_to(path).as_posix().encode())
        h.update(bytes.fromhex(sha256(item)))
    return h.hexdigest()


def copy_checkpoint(source: Path, sequence_id: str) -> list[dict]:
    dest = ASSETS / f"ckpt__{sequence_id}"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    if source.is_file():
        files = [(source, Path(source.name))]
    else:
        files = [(item, item.relative_to(source)) for item in sorted(source.rglob("*")) if item.is_file()
                 and not any(part.startswith("epoch-") for part in item.relative_to(source).parts)
                 and item.name not in SKIP_NAMES
                 and "best_checkpoint" not in item.relative_to(source).parts]
    for item, relative in files:
        flat = f"{dest.name}__{relative.as_posix().replace('/', '__')}"
        shutil.copy2(item, ASSETS / flat)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": flat,
                       "sha256": sha256(item), "size_bytes": item.stat().st_size})
    if not copied:
        raise FileNotFoundError(f"no inference checkpoint files copied for {sequence_id}: {source}")
    return copied


def build_assets() -> None:
    if ASSETS.exists():
        raise FileExistsError(f"refusing to overwrite existing Kaggle model dataset: {ASSETS}")
    ASSETS.mkdir(parents=True)
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selected.get("trust_list_version") != "trust-list-2-v013":
        raise ValueError("checkpoint selection artifact is not Trust List 2 v013")
    ordered = selected["full_v0_selection"]["selected_by_size"]["10"]
    ordered_ids = [str(row["sequence_id"]) for row in ordered]
    if ordered_ids != ["N0071", "N0056", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]:
        raise ValueError(f"v013 checkpoint order changed: {ordered_ids}")
    rows = {str(row["sequence_id"]): row for row in ordered}
    w6w7 = json.loads(W6W7.read_text(encoding="utf-8"))
    qwen_id = w6w7["inputs"]["probability_cache_sources"]["N0074"]["experiment_id"]
    qwen_manifest_path = ROOT / "artifacts/predictions" / f"{qwen_id}_manifest.json"
    qwen_manifest = json.loads(qwen_manifest_path.read_text(encoding="utf-8"))
    if qwen_manifest.get("experiment_id") != qwen_id or qwen_manifest.get("model") != "qwen06":
        raise ValueError("N0074 manifest does not match the W6/W7 checkpoint identity")
    classes = list(map(str, qwen_manifest["classes"]))
    candidates = {}
    for sid in ALL_IDS:
        if sid == "N0074":
            experiment_id = qwen_id
            manifest = qwen_manifest
            source_manifest_path = qwen_manifest_path
            source_path = Path(manifest.get("selected_checkpoint_path") or manifest["checkpoint_path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
        else:
            row = rows[sid]
            experiment_id = str(row["experiment_id"])
            source_manifest_path = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
            manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_path = Path(row["checkpoint_path"])
            source_path = source_path if source_path.is_absolute() else ROOT / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"{sid}: selected checkpoint missing: {source_path}")
        if manifest.get("experiment_id") != experiment_id:
            raise ValueError(f"{sid}: source manifest experiment ID mismatch")
        if list(map(str, manifest.get("classes", []))) != classes or len(classes) != 18:
            raise ValueError(f"{sid}: noncanonical class ordering or mismatch across checkpoints")
        model = str(manifest["model"])
        base_source = {"embeddinggemma2": "embeddinggemma_model_source",
                       "ministral3_3b": "ministral_model_source",
                       "smollm3_3b": "smollm3_dataset",
                       "qwen06": "qwen3_dataset"}.get(model)
        candidates[sid] = {
            "sequence_id": sid, "experiment_id": experiment_id, "model": model,
            "checkpoint_type": "file" if source_path.is_file() else "directory",
            "checkpoint_files": copy_checkpoint(source_path, sid),
            "checkpoint_sha256": checkpoint_hash(source_path),
            "config": manifest.get("config", {}),
            "base_model": manifest.get("config", {}).get("checkpoint"),
            "base_model_revision": manifest.get("model_revision"),
            "base_source": base_source,
            "calibration_temperature": manifest.get("calibration_temperature", 1.0),
            "classes": classes,
            "source_manifest": str(source_manifest_path.relative_to(ROOT)),
            "source_manifest_sha256": sha256(source_manifest_path),
            "selected_checkpoint_source": str(source_path.relative_to(ROOT)),
        }
    # Runtime inference imports only these project modules.
    for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / module, ASSETS / f"source__{module}")
    sklearn_wheel = next((ROOT / "artifacts/kaggle/zod03-models").glob("scikit_learn-1.9.1-*.whl"), None)
    if sklearn_wheel is None:
        raise FileNotFoundError("pinned scikit-learn 1.9.1 wheel not found")
    shutil.copy2(sklearn_wheel, ASSETS / sklearn_wheel.name)
    bundle = {
        "name": "trust2_v013_equal9_and_equal10_qwen", "trust_list_version": "trust-list-2-v013",
        "strategy_ids": {"EQUAL9": BASE_IDS, "EQUAL10-Qwen": BASE_IDS + ["N0074"]},
        "sequence_ids": ALL_IDS, "candidates": candidates, "classes": classes,
        "weights": {"EQUAL9": {sid: 1 / 9 for sid in BASE_IDS},
                    "EQUAL10-Qwen": {sid: 1 / 10 for sid in BASE_IDS + ["N0074"]}},
        "prediction_schema": ["id", "model"],
        "test_inference": "per-checkpoint cached once; equal arithmetic mean of probability vectors",
    }
    (ASSETS / "trust2_v013_equal9_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (ASSETS / "dataset-metadata.json").write_text(json.dumps({
        "title": "Trust2 v013 EQUAL9 and EQUAL10-Qwen inference assets",
        "id": "zodrak/trust2-v013-equal9-models",
        "licenses": [{"name": "other"}],
        "subtitle": "Frozen inference checkpoints for two equal-weight submissions",
        "description": "Inference-only frozen v013 checkpoints N0071, N0005, N0058, N0072, N0030, N0061, N0044, N0009, N0048 plus N0074 for EQUAL10-Qwen.",
    }, indent=2) + "\n")


NOTEBOOK_SOURCE = r'''from pathlib import Path
import gc, json, os, shutil, subprocess, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
INPUT_ROOT = Path("/kaggle/input")
asset_manifests = list(INPUT_ROOT.rglob("trust2_v013_equal9_manifest.json"))
assert len(asset_manifests) == 1, f"expected one EQUAL9 manifest, got {asset_manifests}"
DATA = asset_manifests[0].parent
BUNDLE = json.loads(asset_manifests[0].read_text())
assert BUNDLE["trust_list_version"] == "trust-list-2-v013"
from importlib.metadata import version
if version("scikit-learn") != "1.9.1":
    wheel = next(INPUT_ROOT.rglob("scikit_learn-1.9.1-*.whl"), None)
    assert wheel is not None, "pinned scikit-learn wheel missing"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], check=True)
import joblib, numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "Kaggle GPU unavailable"
WORK = Path("/kaggle/working/trust2_equal9_runtime")
SRC = WORK / "src"; SRC.mkdir(parents=True, exist_ok=True)
for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
    (SRC / name).symlink_to(DATA / f"source__{name}")
sys.path.insert(0, str(WORK))
CHECKPOINTS = WORK / "checkpoints"; CHECKPOINTS.mkdir(parents=True, exist_ok=True)
for sid, candidate in BUNDLE["candidates"].items():
    dest = CHECKPOINTS / sid; dest.mkdir(parents=True, exist_ok=True)
    for item in candidate["checkpoint_files"]:
        src = DATA / item["flat_filename"]
        assert src.is_file(), f"missing checkpoint asset: {sid}/{item['flat_filename']}"
        target = dest / item["relative_path"]; target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(src)
    candidate["runtime_checkpoint"] = str(dest if candidate["checkpoint_type"] == "directory" else dest / candidate["checkpoint_files"][0]["relative_path"])

def resolve_model(marker):
    matches = [p.parent for p in INPUT_ROOT.rglob("config.json") if marker.lower() in str(p.parent).lower()]
    assert matches, f"attached offline model not found: {marker}"
    depth = min(len(p.parts) for p in matches)
    roots = sorted({p for p in matches if len(p.parts) == depth})
    assert len(roots) == 1, f"expected one model root for {marker}: {roots}"
    return roots[0]
BASES = {
    "embeddinggemma2": resolve_model("embeddinggemma-2"),
    "ministral3_3b": resolve_model("ministral-3-3b-base-2512"),
    "qwen06": resolve_model("Qwen3-0.6B"),
}
assert (BASES["qwen06"] / "config.json").is_file(), f"Qwen base missing: {BASES['qwen06']}"
# SmolLM3 dataset shards are mounted read-only. Copy its config and create a local index if the source omits one.
SMOL_SOURCE = resolve_model("smollm3-3b-base-36l-text")
SMOL_BASE = WORK / "bases/smollm3_3b"; SMOL_BASE.mkdir(parents=True, exist_ok=True)
for item in SMOL_SOURCE.iterdir():
    target = SMOL_BASE / item.name
    if target.exists(): continue
    if item.name == "config.json": shutil.copy2(item, target)
    else: target.symlink_to(item)
smol_config_path = SMOL_BASE / "config.json"
smol_config = json.loads(smol_config_path.read_text())
if smol_config.get("model_type") == "smollm3_text_base":
    smol_config["model_type"] = "smollm3"
    smol_config_path.write_text(json.dumps(smol_config))
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
assert "smollm3" in CONFIG_MAPPING, "Transformers does not support SmolLM3"
smol_shards = sorted(SMOL_BASE.glob("*.safetensors"))
assert smol_shards, f"SmolLM3 shards missing under {SMOL_SOURCE}"
smol_index = SMOL_BASE / "model.safetensors.index.json"
if not smol_index.exists():
    from safetensors import safe_open
    weight_map = {}
    for shard in smol_shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            weight_map.update({key: shard.name for key in sf.keys()})
    smol_index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
BASES["smollm3_3b"] = SMOL_BASE
# N0074's adapter folder does not contain tokenizer files; link the tokenizer from its pinned Qwen base.
qwen_checkpoint = Path(BUNDLE["candidates"]["N0074"]["runtime_checkpoint"])
for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
    src = BASES["qwen06"] / name; dest = qwen_checkpoint / name
    if src.exists() and not dest.exists(): dest.symlink_to(src)
# TorchAO is optional; PEFT's Kaggle image dispatcher may be incompatible with its installed version.
try:
    import peft.tuners.lora.torchao as peft_torchao
    peft_torchao.is_torchao_available = lambda: False
except ImportError:
    pass
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
TEST = next((p for p in (INPUT_ROOT / "competitions/name-that-llm/test.csv", INPUT_ROOT / "name-that-llm/test.csv") if p.is_file()), None)
assert TEST is not None, "competition test.csv missing"
'''

PREDICTION_SOURCE = r'''test = pd.read_csv(TEST, dtype={"id": "string"})
assert {"id", "text"}.issubset(test.columns) and test.id.notna().all() and not test.id.duplicated().any()
test.text = test.text.fillna("").astype(str)
classes = list(map(str, BUNDLE["classes"]))
texts = test.text.tolist()
def predict_checkpoint(sid):
    candidate = BUNDLE["candidates"][sid]
    model, checkpoint = candidate["model"], Path(candidate["runtime_checkpoint"])
    if model == "svc":
        probs, labels = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probs, labels = predict_catboost_stylometry(checkpoint, texts, classes,
            temperature=float(candidate.get("calibration_temperature", 1.0)))
    else:
        base = BASES.get(model)
        batch_size = min(4, max(1, int(candidate["config"].get("inference_batch_size", 4) or 4)))
        while True:
            try:
                probs, labels = _predict_neural(model, checkpoint, texts, classes,
                    batch_size=batch_size, max_length=int(candidate["config"].get("max_length", 384)),
                    temperature=float(candidate.get("calibration_temperature", 1.0)),
                    base_checkpoint=str(base) if base else candidate.get("base_model"),
                    revision=None if base else candidate.get("base_model_revision"),
                    fix_mistral_regex=bool(candidate["config"].get("fix_mistral_regex", False)),
                    text_only_backbone=bool(candidate["config"].get("text_only_backbone", False)),
                    quantization=candidate["config"].get("quantization"),
                    attention_backend=candidate["config"].get("attention_backend"))
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1: raise
                batch_size = max(1, batch_size // 2)
                gc.collect(); torch.cuda.empty_cache()
    labels = list(map(str, labels)); probs = np.asarray(probs, dtype=np.float64)
    probs = probs[:, [labels.index(label) for label in classes]]
    assert probs.shape == (len(test), 18) and np.isfinite(probs).all() and np.allclose(probs.sum(1), 1, atol=1e-5)
    del labels
    gc.collect(); torch.cuda.empty_cache()
    return probs

started = time.time()
# Infer each member once. These per-checkpoint predictions are shared by both blends.
test_probabilities = {sid: predict_checkpoint(sid) for sid in BUNDLE["sequence_ids"]}
def make_submission(strategy, members, filename):
    matrix = np.mean(np.stack([test_probabilities[sid] for sid in members]), axis=0)
    prediction = np.asarray(classes, dtype=object)[matrix.argmax(axis=1)]
    result = pd.DataFrame({"id": test.id.astype(str).to_numpy(), "model": prediction.astype(str)})
    assert result.columns.tolist() == ["id", "model"] and len(result) == len(test)
    assert result.id.tolist() == test.id.astype(str).tolist() and not result.id.duplicated().any()
    assert result.model.notna().all() and result.model.isin(classes).all()
    result.to_parquet(f"/kaggle/working/{filename}", index=False)
    check = pd.read_parquet(f"/kaggle/working/{filename}")
    assert check.columns.tolist() == ["id", "model"] and check.id.astype(str).tolist() == test.id.astype(str).tolist()
    return result

equal9 = make_submission("EQUAL9", BUNDLE["strategy_ids"]["EQUAL9"], "submission.parquet")
equal10_qwen = make_submission("EQUAL10-Qwen", BUNDLE["strategy_ids"]["EQUAL10-Qwen"], "submission_equal10_qwen.parquet")
print({"rows": len(test), "test_ids": test.id.astype(str).tolist(),
       "equal9_counts": equal9.model.value_counts().sort_index().to_dict(),
       "equal10_qwen_counts": equal10_qwen.model.value_counts().sort_index().to_dict(),
       "checkpoint_ids_inferred": BUNDLE["sequence_ids"], "runtime_seconds": round(time.time()-started, 1),
       "files": ["submission.parquet", "submission_equal10_qwen.parquet"]})
display(equal9); display(equal10_qwen)
'''


def build_kernel(folder_name: str, kernel_id: str, title: str, notebook_name: str) -> None:
    folder = ROOT / "artifacts/kaggle" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(
        "# Trust List 2 v013 — EQUAL9 and EQUAL10-Qwen\n\n"
        "Offline frozen-checkpoint inference. This notebook computes both requested equal-weight blends from one pass over the ten required checkpoints. "
        "It writes EQUAL9 to `submission.parquet` and EQUAL10-Qwen to `submission_equal10_qwen.parquet`."),
        nbf.v4.new_code_cell(NOTEBOOK_SOURCE), nbf.v4.new_code_cell(PREDICTION_SOURCE)]
    nbf.write(notebook, folder / notebook_name)
    metadata = {"id": kernel_id, "title": title, "code_file": notebook_name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["trust-list-2", "v013", "equal-weight", "offline-inference"],
        "dataset_sources": DATASET_SOURCES, "kernel_sources": [],
        "competition_sources": ["name-that-llm"], "model_sources": MODEL_SOURCES,
        "machine_shape": "NvidiaTeslaT4"}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    build_assets()
    build_kernel("trust2-equal9-kernel", "zodrak/trust2-equal9", "Trust2 v013 EQUAL9 and EQUAL10-Qwen", "trust2_equal9.ipynb")
    print(json.dumps({"assets": str(ASSETS), "kernel": "artifacts/kaggle/trust2-equal9-kernel",
                      "submission_files": ["submission.parquet", "submission_equal10_qwen.parquet"]}, indent=2))


if __name__ == "__main__":
    main()
