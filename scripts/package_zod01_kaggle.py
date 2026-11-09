"""Stage the fixed ZOD01 cascade as a flat Kaggle dataset and kernel."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["N0070", "N0072", "N0058", "N0082", "D", "C", "N0065", "N0075",
        "N0083", "N0031", "N0071", "N0057", "A", "N0018"]
SKIP_FILES = {"training_log.json", "training_progress.json", "checkpoint_selection.json",
              "batch_size_probe.json", "baseline_validation.json", "README.md"}


def _copy_flat(source: Path, output: Path, prefix: str) -> list[dict]:
    records = []
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = file.relative_to(source)
        if any(part.startswith("epoch-") for part in rel.parts):
            continue
        if file.name in SKIP_FILES:
            continue
        flat_name = f"{prefix}__{str(rel).replace('/', '__')}"
        shutil.copy2(file, output / flat_name)
        records.append({"relative_path": rel.as_posix(), "flat_filename": flat_name})
    if not records:
        raise ValueError(f"no inference files copied from {source}")
    return records


def build_dataset_package():
    package_dir = ROOT / "artifacts/kaggle/zod01-models"
    package_dir.mkdir(parents=True, exist_ok=True)
    for path in package_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    inventory = pd.read_csv(ROOT / "artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv")
    inventory = inventory.set_index("candidate_key")
    candidates = {}
    for key in KEYS:
        row = inventory.loc[key]
        experiment_id = str(row.experiment_id)
        source_manifest = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
        manifest = json.loads(source_manifest.read_text())
        source = Path(manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path") or "")
        if not source.is_absolute():
            source = ROOT / source
        if not source.exists():
            raise FileNotFoundError(f"selected inference checkpoint missing for {key}: {source}")
        if source.is_file():
            flat = f"ckpt__{key}__{source.name}"
            shutil.copy2(source, package_dir / flat)
            records = [{"relative_path": source.name, "flat_filename": flat}]
            checkpoint_type = "file"
        else:
            records = _copy_flat(source, package_dir, f"ckpt__{key}")
            checkpoint_type = "directory"
        candidates[key] = {
            "candidate_key": key,
            "experiment_id": experiment_id,
            "model": str(manifest["model"]),
            "checkpoint_type": checkpoint_type,
            "files": records,
            "classes": list(map(str, manifest["classes"])),
            "config": manifest.get("config", {}),
            "model_revision": manifest.get("model_revision"),
            "selected_epoch": manifest.get("selected_epoch"),
            "validation_accuracy": float(row.validation_accuracy),
        }

    # Keep the large Gemma base weights on Kaggle's model storage rather than
    # duplicating them in the private checkpoint dataset.
    gemma_revision = candidates["N0058"]["model_revision"]
    gemma_files = []

    # Package only the Python modules needed by the existing inference helpers.
    for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / module, package_dir / f"source__{module}")

    cascade_rules = [
        ["gemini-3.1-pro-preview", "N0070"], ["gpt-6-astra", "N0072"],
        ["claude-opus-4-6", "N0058"], ["claude-opus-5", "N0082"],
        ["claude-opus-4-8", "D"], ["claude-sonnet-5", "N0058"],
        ["gemini-3.8-flash", "C"], ["qwen3.8-max", "C"],
        ["gpt-5.5", "N0065"], ["gpt-5.6-luna", "N0075"],
        ["gemini-3.6-flash", "N0083"], ["glm-5.3-flash", "N0031"],
        ["gpt-5.6-sol", "N0071"], ["gemini-3.7-flash", "N0082"],
        ["kimi-k3", "D"], ["gpt-5.6-terra", "N0057"],
        ["claude-fable-5", "A"], ["glm-5.3", "N0018"],
    ]
    metadata = {"name": "zod01", "classes": candidates["D"]["classes"],
        "rules": cascade_rules, "fallback_candidate_key": "D",
        "fallback_experiment_id": candidates["D"]["experiment_id"],
        "fallback_parent_experiment_id": "ultrabert380_v1_all_data3c4b61a6_split0b0fd08e_seed2026_cfgebf0c70f_code1b9bfc2",
        "candidates": candidates, "gemma4_base_files": gemma_files,
        "gemma4_base_revision": gemma_revision,
        "submission_schema": ["id", "model"]}
    (package_dir / "zod01_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (package_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "zod01 assets", "id": "zodrak/zod01-models",
        "licenses": [{"name": "other"}],
        "subtitle": "Offline checkpoints for the ZOD01 cascade",
        "description": "Private inference-only package for the fixed 18-label ZOD01 cascade. No training data is included. Other base models are attached using versioned Kaggle model sources."
    }, indent=2) + "\n")
    (package_dir / "README.md").write_text(
        "# ZOD01 Kaggle model package\n\n"+
        "Private inference-only checkpoint package. Build with `python scripts/package_zod01_kaggle.py`.\n"+
        "Checkpoint and Gemma base files are flattened for `kaggle datasets create --dir-mode skip`; "
        "the notebook recreates their directory trees with symlinks.\n")
    return package_dir


def build_kernel():
    import nbformat as nbf
    kernel_dir = ROOT / "artifacts/kaggle/zod01-kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": "zodrak/zod01", "title": "zod01", "code_file": "zod01.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu", "offline-inference"],
        "dataset_sources": ["zodrak/zod01-models", "zodrak/zod01-sklearn", "pingjiale/qwen3-0-6b"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": [
            "google/gemma-4/transformers/gemma-4-e2b/1",
            "qwen-lm/qwen-3/transformers/1.7b-base/1",
            "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
        ], "machine_shape": "NvidiaTeslaT4"
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cell_setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DATA = Path('/kaggle/input/zod01-models')
BUNDLE = json.loads((DATA / 'zod01_manifest.json').read_text())
# SVC joblib files were serialized with scikit-learn 1.9.1. Kaggle's image has
# shipped older versions, so install the matching wheel from the attached data
# before importing NLM modules or unpickling either SVC bundle. Internet is off.
from importlib.metadata import version as package_version
SKLEARN_VERSION = '1.9.1'
if package_version('scikit-learn') != SKLEARN_VERSION:
    wheel_matches = list(Path('/kaggle/input').rglob('scikit_learn-1.9.1-cp313-cp313-*.whl'))
    wheel = wheel_matches[0] if wheel_matches else None
    assert wheel is not None, 'pinned scikit-learn wheel missing from zod01 assets'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--force-reinstall', str(wheel)], check=True)
import sklearn
assert sklearn.__version__ == SKLEARN_VERSION, sklearn.__version__
print({'scikit_learn_runtime': sklearn.__version__})
WORK = Path('/kaggle/working/zod01_runtime')
SRC = WORK / 'src'
SRC.mkdir(parents=True, exist_ok=True)
for name in ('__init__.py', 'data_factory.py', 'feature_factory.py', 'model_factory.py', 'inference.py'):
    (SRC / name).symlink_to(DATA / f'source__{name}')
CHECKPOINTS = WORK / 'checkpoints'
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
for key, candidate in BUNDLE['candidates'].items():
    dest = CHECKPOINTS / key
    dest.mkdir(parents=True, exist_ok=True)
    for item in candidate['files']:
        target = dest / item['relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(DATA / item['flat_filename'])
    candidate['runtime_checkpoint'] = str(dest if candidate['checkpoint_type'] == 'directory' else dest / candidate['files'][0]['relative_path'])
GEMMA = Path('/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1')
TEST_CANDIDATES = [Path('/kaggle/input/competitions/name-that-llm/test.csv'), Path('/kaggle/input/name-that-llm/test.csv')]
TEST_CSV = next((p for p in TEST_CANDIDATES if p.is_file()), None)
assert TEST_CSV is not None, f'test.csv not found: {TEST_CANDIDATES}'
import torch
assert torch.cuda.is_available(), 'Kaggle GPU unavailable'
try:
    from importlib.metadata import PackageNotFoundError, version
    from packaging.version import Version
    try:
        if Version(version('torchao')) <= Version('0.16.0'):
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True, capture_output=True)
    except PackageNotFoundError:
        pass
except ImportError:
    pass
sys.path.insert(0, str(WORK))
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
import joblib, numpy as np, pandas as pd
def resolve_base(path, marker):
    path = Path(path)
    if (path / 'config.json').is_file():
        return path
    input_root = Path('/kaggle/input')
    for config in input_root.rglob('config.json'):
        if marker.lower() in str(config.parent).lower():
            return config.parent
    top = {str(p): [x.name for x in list(p.iterdir())[:10]] for p in input_root.iterdir()}
    raise FileNotFoundError(f'model base config missing for {marker}; expected {path}; input roots={top}')

QWEN06_BASE = resolve_base('/kaggle/input/qwen3-0-6b/Qwen3-0.6B', 'Qwen3-0.6B')
QWEN17_BASE = resolve_base('/kaggle/input/models/qwen-lm/qwen-3/transformers/1.7b-base/1', '1.7b-base')
EMBED_BASE = resolve_base('/kaggle/input/models/google/embeddinggemma-2/transformers/embeddinggemma-2/1', 'embeddinggemma-2')
MINISTRAL_BASE = resolve_base('/kaggle/input/models/mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1', 'ministral-3-3b-base-2512')
GEMMA = resolve_base(GEMMA, 'gemma-4-e2b')
BASES = {'qwen06': QWEN06_BASE, 'qwen17': QWEN17_BASE, 'gemma4_e2b': GEMMA, 'embeddinggemma2': EMBED_BASE, 'ministral3_3b': MINISTRAL_BASE}
print({'cascade': 'zod01', 'candidates': len(BUNDLE['candidates']), 'fallback': BUNDLE['fallback_experiment_id'], 'test': str(TEST_CSV)})'''
    cell_predict = r'''test = pd.read_csv(TEST_CSV, dtype={'id': 'string'})
assert {'id', 'text'}.issubset(test.columns), test.columns.tolist()
assert test.id.notna().all() and not test.id.duplicated().any()
test['text'] = test.text.fillna('').astype(str)
classes = BUNDLE['classes']
class_index = {label: i for i, label in enumerate(classes)}
n = len(test)
assigned = np.zeros(n, dtype=bool)
predicted = np.empty(n, dtype=object)
selected = np.empty(n, dtype=object)
prediction_cache = {}

def infer_top1(key, texts):
    candidate = BUNDLE['candidates'][key]
    model = candidate['model']
    checkpoint = Path(candidate['runtime_checkpoint'])
    if model == 'svc':
        probs, model_classes = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == 'catboost_stylometry':
        probs, model_classes = predict_catboost_stylometry(checkpoint, texts, classes)
    else:
        base = BASES.get(model)
        if model != 'ultrabert380':
            assert base is not None and (base / 'config.json').is_file(), f'base weights missing for {model}: {base}'
        batch = min(4, int(candidate['config'].get('batch_size', 4) or 4))
        while True:
            try:
                probs, model_classes = _predict_neural(model, checkpoint, texts, classes,
                    batch_size=batch, max_length=int(candidate['config'].get('max_length', 384)),
                    base_checkpoint=str(base) if base else None, revision=None,
                    text_only_backbone=bool(candidate['config'].get('text_only_backbone', False)))
                break
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower() or batch <= 1:
                    raise
                batch = max(1, batch // 2)
                gc.collect()
                torch.cuda.empty_cache()
                print({'oom_retry': key, 'batch_size': batch})
    if list(map(str, model_classes)) != classes:
        order = {label: i for i, label in enumerate(map(str, model_classes))}
        probs = probs[:, [order[label] for label in classes]]
    top = np.asarray(classes, dtype=object)[np.asarray(probs).argmax(axis=1)]
    del probs
    gc.collect()
    torch.cuda.empty_cache()
    return top

started = time.time()
rule_rows = []
for order, (label, key) in enumerate(BUNDLE['rules'], start=1):
    remaining = np.flatnonzero(~assigned)
    if key not in prediction_cache:
        prediction_cache[key] = np.full(n, -1, dtype=np.int16)
    cache = prediction_cache[key]
    missing = remaining[cache[remaining] < 0]
    if len(missing):
        top = infer_top1(key, test.text.iloc[missing].tolist())
        cache[missing] = np.asarray([class_index[x] for x in top], dtype=np.int16)
    assert (cache[remaining] >= 0).all(), f'incomplete cached predictions for {key}'
    chosen = remaining[cache[remaining] == class_index[label]]
    assigned[chosen] = True
    predicted[chosen] = label
    selected[chosen] = key
    print({'rule': order, 'label': label, 'checkpoint': key, 'assigned': len(chosen), 'remaining': int((~assigned).sum())})
    rule_rows.append({'order': order, 'label': label, 'checkpoint': key, 'assigned': int(len(chosen)), 'remaining': int((~assigned).sum())})

remaining = np.flatnonzero(~assigned)
if len(remaining):
    key = BUNDLE['fallback_candidate_key']
    if key not in prediction_cache:
        prediction_cache[key] = np.full(n, -1, dtype=np.int16)
    cache = prediction_cache[key]
    missing = remaining[cache[remaining] < 0]
    if len(missing):
        top = infer_top1(key, test.text.iloc[missing].tolist())
        cache[missing] = np.asarray([class_index[x] for x in top], dtype=np.int16)
    predicted[remaining] = np.asarray(classes, dtype=object)[cache[remaining]]
    selected[remaining] = key
    rule_rows.append({'order': 'fallback', 'label': 'UltraBERT V1-all → V0', 'checkpoint': key, 'assigned': int(len(remaining)), 'remaining': 0})

submission = pd.DataFrame({'id': test.id, 'model': predicted})
assert submission.columns.tolist() == ['id', 'model'] and len(submission) == len(test)
assert submission.id.astype(str).tolist() == test.id.astype(str).tolist()
assert submission.id.notna().all() and not submission.id.duplicated().any()
assert submission.model.notna().all() and submission.model.isin(classes).all()
OUTPUT = Path('/kaggle/working/submission.parquet')
submission.to_parquet(OUTPUT, index=False)
pd.DataFrame(rule_rows).to_csv('/kaggle/working/zod01_rule_counts.csv', index=False)
print({'approach': 'zod01', 'rows': len(submission), 'output': str(OUTPUT), 'runtime_seconds': round(time.time()-started, 1), 'schema': submission.columns.tolist()})
display(submission.head())'''
    cells = [
        nbf.v4.new_markdown_cell("# zod01 — sequential specialist cascade\n\n"
            "Runs the fixed 18-label cascade from saved NLM checkpoints. Rule order is fixed; remaining rows use the registry-resolved UltraBERT V1-all → V0 checkpoint. All model access is local to attached Kaggle inputs; internet is disabled."),
        nbf.v4.new_code_cell(cell_setup),
        nbf.v4.new_code_cell(cell_predict),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
        "cascade": "zod01"})
    nbf.write(notebook, kernel_dir / "zod01.ipynb")
    return kernel_dir


if __name__ == "__main__":
    package = build_dataset_package()
    kernel = build_kernel()
    print(json.dumps({"dataset_dir": str(package), "kernel_dir": str(kernel)}, indent=2))
