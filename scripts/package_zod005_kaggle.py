"""Package the evaluated ZOD005 cascade as an offline Kaggle notebook."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/analysis/zod005/v001"
DATA_DIR = ROOT / "artifacts/kaggle/zod005-models"
KERNEL_DIR = ROOT / "artifacts/kaggle/zod005-kernel"
SKIP = {"README.md", "checkpoint_selection.json", "baseline_validation.json",
        "training_log.json", "training_progress.json", "validation_metrics.json"}
MODEL_SOURCE_BY_NAME = {
    "embeddinggemma2": "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
    "gemma4_e2b": "google/gemma-4/transformers/gemma-4-e2b/1",
    "gemma4_e4b": "google/gemma-4/transformers/gemma-4-e4b/1",
    "llama32_3b": "meta/llama-3-2/transformers/llama-3-2-3b/1",
    "ministral3_3b": "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
    "qwen17": "qwen-lm/qwen-3/transformers/1.7b-base/1",
}


def _copy_checkpoint(source: Path, key: str) -> list[dict]:
    dest = DATA_DIR / f"ckpt__{key}"
    dest.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        flat = f"{dest.name}__{source.name}"
        shutil.copy2(source, DATA_DIR / flat)
        return [{"relative_path": source.name, "flat_filename": flat}]
    records = []
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = file.relative_to(source)
        if any(part.startswith("epoch-") for part in relative.parts) or file.name in SKIP:
            continue
        flat = f"{dest.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, DATA_DIR / flat)
        records.append({"relative_path": relative.as_posix(), "flat_filename": flat})
    if not records:
        raise FileNotFoundError(f"no inference files copied from {source}")
    return records


def build_dataset_package() -> Path:
    config = json.loads((ARTIFACTS / "cascade.json").read_text())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for item in DATA_DIR.iterdir():
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    candidates = {}
    for record in config["selected_checkpoints"] + [config["fallback"]]:
        candidate_id = str(record["experiment_id"])
        if candidate_id in candidates:
            continue
        manifest_path = ROOT / "artifacts/predictions" / f"{candidate_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        source = Path(str(record["checkpoint_path"]))
        source = source if source.is_absolute() else ROOT / source
        if not source.exists():
            raise FileNotFoundError(f"ZOD005 checkpoint missing: {candidate_id}: {source}")
        if source.is_dir() and (source / "best_checkpoint").is_dir():
            source = source / "best_checkpoint"
        elif source.is_dir() and manifest.get("selected_epoch"):
            epoch = source / f"epoch-{int(manifest['selected_epoch']):02d}"
            if epoch.is_dir():
                source = epoch
        candidates[candidate_id] = {
            **record,
            "checkpoint_type": "file" if source.is_file() else "directory",
            "files": _copy_checkpoint(source, candidate_id),
            "config": manifest.get("config", {}),
            "model_revision": manifest.get("model_revision"),
            "classes": manifest.get("classes", []),
            "calibration_temperature": manifest.get("calibration_temperature", 1.0),
        }
    bundle = {
        **config,
        "candidates": candidates,
        "submission_schema": ["id", "model"],
        "source_validation_predictions": "artifacts/analysis/zod005/v001/validation_predictions.parquet",
    }
    (DATA_DIR / "zod005_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / name, DATA_DIR / f"source__{name}")
    sklearn_wheel = next((ROOT / "artifacts/kaggle/zod01-sklearn").glob("scikit_learn-1.9.1-*.whl"), None)
    if any(c["model"] == "svc" for c in candidates.values()):
        if sklearn_wheel is None:
            raise FileNotFoundError("pinned scikit-learn wheel is required for the ZOD005 SVC")
        shutil.copy2(sklearn_wheel, DATA_DIR / sklearn_wheel.name)
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "ZOD005 inference checkpoints", "id": "zodrak/zod005-models",
        "licenses": [{"name": "other"}],
        "subtitle": "Frozen checkpoints and inference source for ZOD005",
        "description": "Inference-only ZOD005 checkpoint files. No training data is included.",
    }, indent=2) + "\n")
    return DATA_DIR


def build_kernel() -> Path:
    import nbformat as nbf

    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted({MODEL_SOURCE_BY_NAME[name] for name in
        [c["model"] for c in json.loads((DATA_DIR / "zod005_manifest.json").read_text())["candidates"].values()]
        if name in MODEL_SOURCE_BY_NAME})
    bundle = json.loads((DATA_DIR / "zod005_manifest.json").read_text())
    models = {str(c["model"]) for c in bundle["candidates"].values()}
    setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path('/kaggle/input/zod005-models')
BUNDLE = json.loads((ROOT / 'zod005_manifest.json').read_text())
WORK = Path('/kaggle/working/zod005_runtime')
SRC = WORK / 'src'; SRC.mkdir(parents=True, exist_ok=True)
for name in ('__init__.py', 'data_factory.py', 'feature_factory.py', 'model_factory.py', 'inference.py'):
    (SRC / name).symlink_to(ROOT / f'source__{name}')
sys.path.insert(0, str(WORK))
CHECKPOINTS = WORK / 'checkpoints'; CHECKPOINTS.mkdir(parents=True, exist_ok=True)
for key, item in BUNDLE['candidates'].items():
    dest = CHECKPOINTS / key; dest.mkdir(parents=True, exist_ok=True)
    for file in item['files']:
        target = dest / file['relative_path']; target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / file['flat_filename'])
    item['runtime_checkpoint'] = str(dest if item['checkpoint_type'] == 'directory' else dest / item['files'][0]['relative_path'])
'''
    if "svc" in models:
        setup += r'''from importlib.metadata import version as package_version
SKLEARN_VERSION = '1.9.1'
if package_version('scikit-learn') != SKLEARN_VERSION:
    wheel = next(ROOT.glob('scikit_learn-1.9.1-*.whl'), None)
    assert wheel is not None, 'pinned scikit-learn wheel missing'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall', str(wheel)], check=True)
'''
    setup += r'''# Remove the image's incompatible optional TorchAO before PEFT is imported.
try:
    from importlib.metadata import PackageNotFoundError, version
    from packaging.version import Version
    try:
        if Version(version('torchao')) <= Version('0.16.0'):
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
                           check=True, capture_output=True)
    except PackageNotFoundError:
        pass
except ImportError:
    pass
import torch, numpy as np, pandas as pd, joblib
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
TEST = next((p for p in (Path('/kaggle/input/competitions/name-that-llm/test.csv'), Path('/kaggle/input/name-that-llm/test.csv')) if p.is_file()), None)
assert TEST is not None, 'competition test.csv missing'
def resolve_base(marker):
    root = Path('/kaggle/input')
    for p in root.rglob('config.json'):
        if marker.lower() in str(p.parent).lower(): return p.parent
    return None
BASES = {
    'qwen06': resolve_base('Qwen3-0.6B'),
    'qwen17': resolve_base('1.7b-base'),
    'gemma4_e2b': resolve_base('gemma-4-e2b'),
    'gemma4_e4b': resolve_base('gemma-4-e4b'),
    'embeddinggemma2': resolve_base('embeddinggemma-2'),
    'ministral3_3b': resolve_base('ministral-3-3b-base-2512'),
    'llama32_3b': resolve_base('llama-3-2-3b'),
}
test = pd.read_csv(TEST, dtype={'id': 'string'})
assert {'id','text'}.issubset(test.columns) and test.id.notna().all() and not test.id.duplicated().any()
texts = test.text.fillna('').astype(str).tolist()
classes = sorted({r['target_label'] for r in BUNDLE['ordered_rules']})
# Include all competition labels even if no rule targets a label.
classes = list(BUNDLE['candidates'][BUNDLE['fallback']['experiment_id']]['classes'] or classes)
def predict_one(key):
    c = BUNDLE['candidates'][key]; model = c['model']; cp = Path(c['runtime_checkpoint'])
    if model == 'svc': probs, labels = predict_svc_bundle(joblib.load(cp), texts)
    elif model == 'catboost_stylometry': probs, labels = predict_catboost_stylometry(cp, texts, classes)
    else:
        base = BASES.get(model)
        if model not in {'ultrabert380'}: assert base is not None, f'base model missing: {model}'
        probs, labels = _predict_neural(model, cp, texts, classes,
            batch_size=min(4, int(c.get('config',{}).get('batch_size',4) or 4)),
            max_length=int(c.get('config',{}).get('max_length',384) or 384),
            base_checkpoint=str(base) if base else None,
            fix_mistral_regex=bool(c.get('config',{}).get('fix_mistral_regex',False)),
            text_only_backbone=bool(c.get('config',{}).get('text_only_backbone',False)))
    labels = list(map(str, labels)); order = [labels.index(x) for x in classes]
    out = np.asarray(classes,dtype=object)[np.asarray(probs)[:,order].argmax(1)]
    del probs; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out.astype(str)
started = time.time()
predictions = {key: predict_one(key) for key in BUNDLE['candidates']}
assigned = np.zeros(len(test),dtype=bool); output = np.empty(len(test),dtype=object); source = np.empty(len(test),dtype=object)
for rule in BUNDLE['ordered_rules']:
    key,label = rule['experiment_id'],rule['target_label']
    take = (~assigned) & (predictions[key] == label)
    output[take],source[take],assigned[take] = label,key,True
fallback = BUNDLE['fallback']['experiment_id']; output[~assigned] = predictions[fallback][~assigned]; source[~assigned] = fallback
submission = pd.DataFrame({'id':test.id.astype(str),'model':output.astype(str)})
assert submission.columns.tolist() == ['id','model'] and len(submission) == len(test)
assert submission.id.tolist() == test.id.astype(str).tolist() and not submission.id.duplicated().any()
assert submission.model.notna().all() and submission.model.isin(classes).all()
submission.to_csv('/kaggle/working/submission.csv',index=False)
submission.to_parquet('/kaggle/working/submission.parquet',index=False)
pd.DataFrame({'id':test.id.astype(str),'model':output.astype(str),'selected_experiment_id':source.astype(str)}).to_parquet('/kaggle/working/zod005_decisions.parquet',index=False)
print({'rows':len(submission),'rule_covered':int(assigned.sum()),'fallback_rows':int((~assigned).sum()),'runtime_seconds':round(time.time()-started,1),'schema':submission.columns.tolist()})
display(submission.head())'''
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# ZOD005 offline checkpoint cascade\n\nRuns the frozen V2 trust-list cascade. Checkpoint and rule selection were measured on full V0 and are development-set results."),
        nbf.v4.new_code_cell(setup),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "cascade": "zod005_v001"})
    nbf.write(notebook, KERNEL_DIR / "zod005.ipynb")
    metadata = {
        "id": "zodrak/zod005", "title": "ZOD005", "code_file": "zod005.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["zodrak/zod005-models", "pingjiale/qwen3-0-6b"] if "qwen06" in models else ["zodrak/zod005-models"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": sources, "machine_shape": "NvidiaTeslaT4",
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return KERNEL_DIR


if __name__ == "__main__":
    data = build_dataset_package()
    kernel = build_kernel()
    print(json.dumps({"dataset": str(data), "dataset_bytes": sum(p.stat().st_size for p in data.rglob("*") if p.is_file()),
                      "kernel": str(kernel)}, indent=2))
