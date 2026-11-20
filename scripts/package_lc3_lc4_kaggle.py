"""Append inference assets for the latest Trust List 2 LC3/LC4 run.

This intentionally preserves the existing zod006 Kaggle dataset. It adds a
separate LC3/LC4 manifest and only files needed by those checkpoints at
inference time; optimizer, scheduler and epoch-training state are excluded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRUST_POINTER = ROOT / "docs/trust-list-2-latest.json"
DATA_DIR = ROOT / "artifacts/kaggle/zod006-models"
KERNEL_DIR = ROOT / "artifacts/kaggle/lc3-lc4-kernel"
OUT_DIR = ROOT / "artifacts/kaggle/lc3_lc4"
OLD_BUNDLE = DATA_DIR / "zod006_manifest.json"
PRED_DIR = ROOT / "artifacts/analysis/trust_list_2_v0"

TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt",
    "merges.txt", "added_tokens.json", "chat_template.jinja",
    "generation_config.json",
}
MODEL_FILES = {
    "config.json", "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
    "pytorch_model.bin.index.json", "adapter_config.json",
    "adapter_model.safetensors", "classifier.pt", "checkpoint_manifest.json",
    "model.cbm", "stylometry.json", "readout.joblib",
}
SKIP_NAMES = {
    "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth",
    "trainer_state.json", "training_log.json", "training_progress.json",
    "checkpoint_selection.json", "batch_size_probe.json",
    "baseline_validation.json", "validation_metrics.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size != source.stat().st_size or sha256_file(dest) != sha256_file(source):
            raise FileExistsError(f"refusing to overwrite nonmatching existing Kaggle asset: {dest}")
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _selected_rules() -> tuple[str, Path, dict, list[str], dict[str, list[dict]]]:
    pointer = json.loads(TRUST_POINTER.read_text())
    trust_path = ROOT / pointer["csv_path"]
    metadata_path = ROOT / pointer["metadata_path"]
    meta = json.loads(metadata_path.read_text())
    if meta.get("version") != pointer.get("version"):
        raise ValueError("Trust List 2 pointer and metadata version disagree")
    if int(meta.get("validation_rows", 0)) != 2699:
        raise ValueError("expected full-V0 Trust List 2 metrics")
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    for col in ("rule_eligible", "evaluation_eligible", "checkpoint_available"):
        trust = trust[trust[col].str.lower().isin({"true", "1"})]
    overlap = pd.to_numeric(trust.validation_training_overlap, errors="coerce")
    trust = trust[overlap.eq(0)].copy()
    trust["_precision"] = pd.to_numeric(trust.precision, errors="raise")
    trust["_seq"] = pd.to_numeric(trust.SeqID, errors="raise")
    # The canonical label order comes from the experiment prediction manifest.
    first_id = trust.experiment_id.iloc[0]
    first_manifest = json.loads((ROOT / "artifacts/predictions" / f"{first_id}_manifest.json").read_text())
    classes = list(map(str, first_manifest["classes"]))
    if len(classes) != 18 or len(set(classes)) != 18:
        raise ValueError("canonical class order must contain 18 unique labels")
    per_label: dict[str, list[dict]] = {}
    for label in classes:
        subset = trust[trust.target_label.eq(label)].sort_values(
            ["_precision", "_seq"], ascending=[False, True], kind="stable"
        ).head(18)
        if len(subset) != 18:
            raise ValueError(f"expected 18 eligible rules for {label}; got {len(subset)}")
        per_label[label] = []
        for row in subset.to_dict("records"):
            per_label[label].append({
                "sequence_id": str(row["sequence_id"]),
                "seq_id": int(row["SeqID"]),
                "candidate_id": str(row["experiment_id"]),
                "model": str(row["model"]),
                "label": label,
                "precision": float(row["_precision"]),
                "prediction_count": int(row["prediction_count"]),
            })
    return str(pointer["version"]), trust_path, meta, classes, per_label


def _checkpoint_path(experiment_id: str, manifest: dict) -> Path:
    value = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path")
    if not value:
        raise FileNotFoundError(f"checkpoint path missing in manifest for {experiment_id}")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir() and (path / "best_checkpoint").is_dir():
        path = path / "best_checkpoint"
    elif path.is_dir() and manifest.get("selected_epoch"):
        ep = path / f"epoch-{int(manifest['selected_epoch']):02d}"
        if ep.is_dir():
            path = ep
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist for {experiment_id}: {path}")
    return path


def _inference_files(source: Path, model: str) -> list[tuple[Path, str]]:
    if model == "svc":
        if not source.is_file():
            raise FileNotFoundError(f"SVC checkpoint must be a file: {source}")
        return [(source, source.name)]
    if source.is_file():
        return [(source, source.name)]
    result = []
    for f in sorted(source.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(source)
        if any(part.startswith("epoch-") or part in {"continuation_checkpoint", "optimizer_state"} for part in rel.parts):
            continue
        if f.name in SKIP_NAMES or f.name.endswith((".log", ".lock", ".tmp")):
            continue
        if f.name in MODEL_FILES or f.name in TOKENIZER_FILES or f.name.startswith(("configuration_", "modeling_")) and f.suffix == ".py":
            result.append((f, rel.as_posix()))
    if not result:
        raise FileNotFoundError(f"no inference files found for {model}: {source}")
    return result


def _local_v0_replay(classes: list[str], per_label: dict[str, list[dict]]) -> dict:
    probabilities: dict[str, np.ndarray] = {}
    ids = None
    y_true = None
    for rules in per_label.values():
        for rule in rules:
            experiment_id = rule["candidate_id"]
            if experiment_id in probabilities:
                continue
            cache = PRED_DIR / f"{experiment_id}.parquet"
            if not cache.is_file():
                cache = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.parquet"
            if not cache.is_file():
                raise FileNotFoundError(f"full-V0 probability cache missing: {experiment_id}")
            frame = pd.read_parquet(cache)
            if len(frame) != 2699 or not {"id", "y_true"}.issubset(frame.columns):
                raise ValueError(f"invalid validation prediction cache for {experiment_id}")
            if ids is None:
                ids = frame.id.astype(str).tolist()
                y_true = frame.y_true.astype(str).to_numpy()
            elif frame.id.astype(str).tolist() != ids or not np.array_equal(frame.y_true.astype(str).to_numpy(), y_true):
                raise ValueError(f"V0 row alignment mismatch for {experiment_id}")
            cols = [f"prob::{label}" for label in classes]
            if not set(cols).issubset(frame.columns):
                raise ValueError(f"class probability columns missing for {experiment_id}")
            probabilities[experiment_id] = frame[cols].to_numpy(dtype=np.float64)
    if len(probabilities) != len({r["candidate_id"] for rs in per_label.values() for r in rs}):
        raise ValueError("not all selected checkpoints have aligned V0 predictions")
    result = {"rows": len(ids), "checkpoints": len(probabilities), "strategies": {}}
    prediction_frame = pd.DataFrame({"id": ids, "y_true": y_true})
    for strategy in ("LC3", "LC4"):
        # LC3 round 18 and LC4 round 6 both consume rules 1..18 per class.
        score = np.zeros((len(ids), len(classes)), dtype=np.float64)
        denominator = np.zeros(len(classes), dtype=np.float64)
        for j, label in enumerate(classes):
            for rule in per_label[label][:18]:
                weight = float(rule["precision"])
                score[:, j] += weight * probabilities[rule["candidate_id"]][:, j]
                denominator[j] += weight
        if np.any(denominator <= 0):
            raise ValueError("a target label has no positive total rule precision")
        predicted = np.asarray(classes, dtype=object)[(score / denominator[None, :]).argmax(axis=1)].astype(str)
        balanced = float(np.mean([np.mean(predicted[y_true == label] == label) for label in classes]))
        accuracy = float(np.mean(predicted == y_true))
        result["strategies"][strategy] = {
            "best_round": 18 if strategy == "LC3" else 6,
            "rules": 324,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "correct": int(np.sum(predicted == y_true)),
        }
        prediction_frame[f"{strategy.lower()}_prediction"] = predicted
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(OUT_DIR / "v0_predictions.parquet", index=False)
    return result


def build_assets() -> dict:
    version, trust_path, trust_meta, classes, by_label = _selected_rules()
    old = json.loads(OLD_BUNDLE.read_text())
    old_candidates = old["candidates"]
    old_ids = set(old_candidates)
    all_rules = [r for label in classes for r in by_label[label]]
    # Both selected best rounds use the same 18 rules per label: LC3 round 18,
    # LC4 round 6. Keep an explicit stable global priority for auditability.
    all_rules.sort(key=lambda r: (-r["precision"], r["seq_id"], r["label"], r["candidate_id"]))
    candidate_ids = list(dict.fromkeys(r["candidate_id"] for r in all_rules))
    if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_ids) < 1:
        raise AssertionError("candidate selection failed")
    candidates = {}
    for experiment_id in candidate_ids:
        if experiment_id in old_candidates:
            # These three checkpoint payloads already exist in zodrak/zod006-models.
            original = old_candidates[experiment_id]
            candidates[experiment_id] = {
                **original,
                "experiment_id": experiment_id,
                "source_manifest": f"artifacts/predictions/{experiment_id}_manifest.json",
                "already_in_zod006_dataset": True,
            }
            continue
        manifest_path = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        model = str(manifest["model"])
        source = _checkpoint_path(experiment_id, manifest)
        files = []
        prefix = f"lc_ckpt__{experiment_id}"
        for src, relative in _inference_files(source, model):
            flat = f"{prefix}__{relative.replace('/', '__')}"
            _link_or_copy(src, DATA_DIR / flat)
            files.append({"relative_path": relative, "flat_filename": flat, "sha256": sha256_file(DATA_DIR / flat), "size": src.stat().st_size})
        candidates[experiment_id] = {
            "candidate_id": experiment_id,
            "experiment_id": experiment_id,
            "model": model,
            "checkpoint_type": "file" if model == "svc" else "directory",
            "checkpoint_path": manifest.get("checkpoint_path"),
            "checkpoint_source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "source_manifest": f"artifacts/predictions/{experiment_id}_manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "training_dataset": manifest.get("training_dataset") or manifest.get("dataset"),
            "model_revision": manifest.get("model_revision"),
            "base_model": manifest.get("config", {}).get("checkpoint"),
            "config": manifest.get("config", {}),
            "classes": list(map(str, manifest["classes"])),
            "files": files,
            "already_in_zod006_dataset": False,
        }
    if any(c.get("classes") and list(c["classes"]) != classes for c in candidates.values()):
        raise ValueError("checkpoint class order mismatch")
    # MiniCPM5 has no Kaggle Transformers base-model variation, so include its
    # pinned public base snapshot as flat files in the existing dataset.
    mini_source = DATA_DIR / "base_minicpm5_2b"
    if not (mini_source / "model.safetensors").is_file():
        raise FileNotFoundError("pinned MiniCPM5 base weights are not fully downloaded")
    mini_files = []
    for src in sorted(mini_source.rglob("*")):
        if not src.is_file() or src.name in SKIP_NAMES or src.name.endswith(".incomplete"):
            continue
        if ".cache" in src.parts:
            continue
        if src.name not in MODEL_FILES and src.name not in TOKENIZER_FILES:
            continue
        rel = src.relative_to(mini_source).as_posix()
        flat = f"base__minicpm5_2b__{rel.replace('/', '__')}"
        _link_or_copy(src, DATA_DIR / flat)
        mini_files.append({"relative_path": rel, "flat_filename": flat,
                           "size": src.stat().st_size, "sha256": sha256_file(DATA_DIR / flat)})
    if not any(item["relative_path"] == "model.safetensors" for item in mini_files):
        raise FileNotFoundError("flattened MiniCPM5 base weights missing")
    mini_manifest_path = DATA_DIR / "lc3_lc4_minicpm5_base_manifest.json"
    mini_manifest_path.write_text(json.dumps({
        "model": "openbmb/MiniCPM5-2B-Base",
        "revision": "e09861095569642c3cf632d80e3200ee1d30b547",
        "files": mini_files,
    }, indent=2) + "\n")
    # Reuse only the two pinned Granite base backbones needed by LC3/LC4,
    # avoiding the unrelated 16 GB ZOD03 checkpoint dataset at inference.
    z3_manifest = json.loads((ROOT / "artifacts/kaggle/zod03-models/zod03_manifest.json").read_text())
    granite_bases = {}
    for model in ("granite4_h_micro", "granite4_350m"):
        candidate = next((c for c in z3_manifest["candidates"].values()
                          if c.get("model") == model and c.get("base_files")), None)
        if candidate is None:
            raise FileNotFoundError(f"pinned {model} base assets missing from ZOD03 bundle")
        granite_bases[model] = []
        for item in candidate["base_files"]:
            src = ROOT / "artifacts/kaggle/zod03-models" / item["flat_filename"]
            if not src.is_file():
                raise FileNotFoundError(src)
            _link_or_copy(src, DATA_DIR / item["flat_filename"])
            granite_bases[model].append({**item, "size": src.stat().st_size,
                                         "sha256": sha256_file(DATA_DIR / item["flat_filename"])})
    base_manifest_path = DATA_DIR / "lc3_lc4_granite_base_manifest.json"
    base_manifest_path.write_text(json.dumps({"models": granite_bases}, indent=2) + "\n")
    # LC3 and LC4 best-round selections are algebraically identical: each label
    # has its top 18 precision-ranked rules, so their final probability scores
    # must match exactly.
    lc3 = {label: by_label[label][:18] for label in classes}
    lc4 = {label: by_label[label][:18] for label in classes}
    diagnostics = _local_v0_replay(classes, by_label)
    if not np.array_equal(
        diagnostics["strategies"]["LC3"]["correct"], diagnostics["strategies"]["LC4"]["correct"]
    ):
        raise AssertionError("LC3/LC4 selected-round V0 predictions should be identical")
    manifest = {
        "name": "lc3_lc4",
        "trust_list_version": version,
        "trust_list_path": str(trust_path.relative_to(ROOT)),
        "trust_list_sha256": sha256_file(trust_path),
        "validation_sha256": trust_meta["validation_sha256"],
        "validation_rows": 2699,
        "classes": classes,
        "algorithm": "precision_weighted_per_label_probability_mean",
        "ranking": "descending recorded raw precision; tie by ascending Trust List SeqID",
        "rule_eligibility": "rule_eligible and evaluation_eligible and checkpoint_available and zero V0 training overlap",
        "rounds": {
            "LC3": {"best_round": 18, "rules_per_label": 18, "selected_rules": lc3},
            "LC4": {"best_round": 6, "rules_per_label": 18, "selected_rules": lc4},
        },
        "candidate_ids": candidate_ids,
        "candidates": candidates,
        "local_v0_diagnostic": diagnostics,
        "submission_schema": ["id", "model"],
        "selection_note": "LC3 round 18 and LC4 round 6 both aggregate the top 18 eligible rules for every label and therefore are prediction-identical under the requested weighted-mean formula.",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve zod006 assets and copy only current runtime source modules.
    for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        source = ROOT / "src" / name
        target = DATA_DIR / f"source__{name}"
        shutil.copy2(source, target)
    shutil.copy2(trust_path, DATA_DIR / trust_path.name)
    manifest_path = DATA_DIR / "lc3_lc4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    metadata_path = DATA_DIR / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["subtitle"] = "ZOD006 plus LC3/LC4 Trust List 2 checkpoint inference assets"
    metadata["description"] = "Existing ZOD006 assets preserved; LC3/LC4 inference-only checkpoints and manifest appended."
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


def build_kernel() -> Path:
    import nbformat as nbf

    setup = r'''from pathlib import Path
import gc, json, os, sys, time
import hashlib
import importlib.metadata as importlib_metadata
import subprocess
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
INPUT=Path('/kaggle/input')
matches=list(INPUT.rglob('lc3_lc4_manifest.json'))
assert len(matches)==1, f'expected exactly one LC3/LC4 manifest, got {matches}'
MANIFEST_PATH=matches[0]; DATA=MANIFEST_PATH.parent; B=json.loads(MANIFEST_PATH.read_text())
assert B['validation_rows']==2699 and len(B['classes'])==18
assert B['rounds']['LC3']['best_round']==18 and B['rounds']['LC4']['best_round']==6
assert B['rounds']['LC3']['selected_rules']==B['rounds']['LC4']['selected_rules']
assert torch.cuda.is_available(), 'Kaggle GPU unavailable'
WORK=Path('/kaggle/working/lc3_lc4_runtime'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)
def unique_path(paths, desc):
    paths=list(dict.fromkeys(Path(x) for x in paths))
    if len(paths)!=1: raise AssertionError(f'{desc}: expected one path, got {paths[:10]}')
    return paths[0]
def resolve_asset(asset):
    filename=Path(asset['flat_filename'] if isinstance(asset,dict) else str(asset)).name
    paths=[p for p in INPUT.rglob(filename) if p.is_file()]
    return unique_path(paths,filename)
runtime_path=unique_path(list(DATA.glob('lc3_lc4_runtime_manifest.json')),'runtime manifest')
runtime_manifest=json.loads(runtime_path.read_text())
for name,asset in runtime_manifest.items():
    (SRC/name).symlink_to(resolve_asset(asset))
sys.path.insert(0,str(WORK))
# Kaggle's base image can include TorchAO 0.10, which PEFT's optional
# dispatcher rejects. It is not needed for these checkpoints; remove only
# versions older than the PEFT-supported minimum before importing inference.
try:
    from packaging.version import Version
    torchao_version=Version(importlib_metadata.version('torchao'))
except importlib_metadata.PackageNotFoundError:
    torchao_version=None
if torchao_version is not None and torchao_version < Version('0.16.0'):
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
from src.inference import predict_candidate
def resolve_model(marker):
    hits=[]
    for p in INPUT.rglob('config.json'):
        if marker.casefold() in str(p.parent).casefold(): hits.append(p.parent)
    if not hits: raise FileNotFoundError(f'base model not mounted: {marker}')
    best=min(len(p.parts) for p in hits); hits=sorted({p for p in hits if len(p.parts)==best})
    return unique_path(hits,marker)
BASES={}
BASE_MARKERS={
  'qwen06':'Qwen3-0.6B','qwen17':'1.7b-base',
  'ministral3_3b':'ministral-3-3b-base-2512','gemma4_e2b':'gemma-4-e2b',
  'embeddinggemma2':'embeddinggemma-2','smollm3_3b':'smollm3-3b-base',
  'hunyuan_1_8b':'hunyuan-1-8b',
}
for model,marker in BASE_MARKERS.items(): BASES[model]=resolve_model(marker)
def rebuild_flat_base(directory, manifest_file, manifest_key, base_name):
    manifest_path=unique_path(list(directory.rglob(manifest_file)),manifest_file)
    info=json.loads(manifest_path.read_text()); base=WORK/'bases'/base_name; base.mkdir(parents=True,exist_ok=True)
    for item in info[manifest_key]:
        src=resolve_asset(item); assert src.is_file(), f'missing base asset {src}'
        out=base/item['relative_path']; out.parent.mkdir(parents=True,exist_ok=True); out.symlink_to(src)
    return base
BASES['llama32_3b']=rebuild_flat_base(INPUT,'llama_base_manifest.json','files','llama32_3b')
granite_manifest=json.loads(unique_path(list(DATA.glob('lc3_lc4_granite_base_manifest.json')),'Granite base manifest').read_text())
for model in ('granite4_h_micro','granite4_350m'):
    files=granite_manifest['models'].get(model,[])
    assert files, f'{model} pinned base files missing'
    base=WORK/'bases'/model; base.mkdir(parents=True,exist_ok=True)
    for item in files:
        src=resolve_asset(item); assert src.is_file(), f'missing base file {src}'
        out=base/item['relative_path']; out.parent.mkdir(parents=True,exist_ok=True); out.symlink_to(src)
    BASES[model]=base
mini_manifest=json.loads(unique_path(list(DATA.glob('lc3_lc4_minicpm5_base_manifest.json')),'MiniCPM5 manifest').read_text())
mini_base=WORK/'bases/minicpm5_2b'; mini_base.mkdir(parents=True,exist_ok=True)
for item in mini_manifest['files']:
    src=resolve_asset(item); assert src.is_file(), f'MiniCPM5 base file missing: {src}'
    out=mini_base/item['relative_path']; out.parent.mkdir(parents=True,exist_ok=True); out.symlink_to(src)
BASES['minicpm5_2b']=mini_base
smollm_source=BASES['smollm3_3b']; idx=unique_path(list(INPUT.rglob('smollm3_base_model.safetensors.index.json')),'SmolLM3 index')
smollm=WORK/'bases/smollm3_3b_original'; smollm.mkdir(parents=True,exist_ok=True)
for f in smollm_source.rglob('*'):
    if f.is_file():
        dest=smollm/f.relative_to(smollm_source); dest.parent.mkdir(parents=True,exist_ok=True); dest.symlink_to(f)
(smollm/'model.safetensors.index.json').symlink_to(idx)
cfg=json.loads((smollm/'config.json').read_text())
smollm_runtime=WORK/'bases/smollm3_3b_runtime'; smollm_runtime.mkdir(parents=True,exist_ok=True)
for f in smollm.rglob('*'):
    if f.is_file():
        o=smollm_runtime/f.relative_to(smollm); o.parent.mkdir(parents=True,exist_ok=True)
        if f.name=='config.json' and cfg.get('model_type')=='smollm3_text_base':
            c=json.loads(f.read_text()); c['model_type']='smollm3'; o.write_text(json.dumps(c))
        elif not o.exists() and not o.is_symlink():
            o.symlink_to(f)
runtime_index=smollm_runtime/'model.safetensors.index.json'
if not runtime_index.exists() and not runtime_index.is_symlink():
    runtime_index.symlink_to(idx)
BASES['smollm3_3b']=smollm_runtime
TEST=unique_path([p for p in (INPUT/'competitions/name-that-llm/test.csv',INPUT/'name-that-llm/test.csv') if p.is_file()],'competition test.csv')
test=pd.read_csv(TEST,dtype={'id':'string'}); assert {'id','text'} <= set(test.columns)
assert test.id.notna().all() and test.id.is_unique
texts=test.text.fillna('').astype(str).tolist(); classes=B['classes']; n=len(test)
scores=np.zeros((2,n,len(classes)),dtype=np.float64); den=np.zeros((2,len(classes)),dtype=np.float64)
rules={strategy:{label:items for label,items in spec['selected_rules'].items()} for strategy,spec in B['rounds'].items()}
rules_by_id={}
for strategy,per_label in rules.items():
    for label,items in per_label.items():
        for r in items: rules_by_id.setdefault(r['candidate_id'],[]).append((strategy,label,float(r['precision'])))
CHECKPOINTS=WORK/'checkpoints'; CHECKPOINTS.mkdir(parents=True,exist_ok=True)
specs={}
for cid in B['candidate_ids']:
    c=B['candidates'][cid]; dest=CHECKPOINTS/cid; dest.mkdir(parents=True,exist_ok=True)
    for f in c['files']:
        source=resolve_asset(f); assert source.is_file(), f'missing checkpoint file: {source}'
        target=dest/f['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(source)
    if c.get('checkpoint_type')=='file': ckpt=dest/Path(c['files'][0]['relative_path']).name
    else: ckpt=dest
    specs[cid]={**c,'checkpoint':str(ckpt)}
assert len(specs)==len(B['candidate_ids'])
'''
    inference = r'''started=time.time(); cache=Path('/kaggle/working/lc3_lc4_probability_cache'); cache.mkdir(exist_ok=True)
for i,cid in enumerate(B['candidate_ids'],1):
    spec=specs[cid]; model=spec['model']; local_base=BASES.get(model)
    model_spec={**spec,'checkpoint':str(spec['checkpoint'])}
    if model in {'hunyuan_1_8b','minicpm5_2b'} and model_spec.get('config',{}).get('quantization')=='qlora4':
        # The saved adapters are QLoRA, but inference can load these <=2B
        # backbones in FP16 on Kaggle T4 without the incompatible bundled bnb.
        model_spec['config']={**model_spec['config'],'quantization':'none'}
    if local_base:
        model_spec['base_model']=str(local_base)
    else:
        model_spec.pop('base_model',None)
    probs=predict_candidate(model_spec,texts,classes,WORK)
    assert probs.shape==(n,18) and np.isfinite(probs).all()
    np.savez_compressed(cache/f'{i:03d}_{cid}.npz',probabilities=probs.astype(np.float32),ids=test.id.astype(str).to_numpy())
    for strategy_idx,strategy in enumerate(('LC3','LC4')):
        for label,weight in sorted(set((label,w) for s,label,w in rules_by_id[cid] if s==strategy)):
            j=classes.index(label); scores[strategy_idx,:,j]+=weight*probs[:,j]; den[strategy_idx,j]+=weight
    del probs; gc.collect(); torch.cuda.empty_cache()
    if i%5==0 or i==len(B['candidate_ids']):
        print(f'loaded {i}/{len(B["candidate_ids"])} checkpoints; elapsed {time.time()-started:.1f}s')
assert (den>0).all(), 'some class has no selected rules'
preds=[]
for si,strategy in enumerate(('LC3','LC4')):
    final=scores[si]/den[si][None,:]
    labels=np.asarray(classes,dtype=object)[final.argmax(axis=1)].astype(str)
    frame=pd.DataFrame({'id':test.id.astype(str),'model':labels})
    assert frame.columns.tolist()==['id','model'] and len(frame)==len(test)
    assert frame.id.tolist()==test.id.astype(str).tolist() and frame.id.is_unique
    assert frame.model.notna().all() and frame.model.isin(classes).all()
    dest=Path('/kaggle/working')/f'submission_{strategy.lower()}.parquet'; frame.to_parquet(dest,index=False)
    preds.append(labels)
    np.savez_compressed(Path('/kaggle/working')/f'{strategy.lower()}_scores.npz',scores=final.astype(np.float32),ids=test.id.astype(str).to_numpy(),classes=np.asarray(classes))
assert np.array_equal(preds[0],preds[1]), 'LC3 and LC4 should be identical at their best rounds'
pd.DataFrame({'id':test.id.astype(str),'model':preds[0]}).to_parquet('/kaggle/working/submission.parquet',index=False)
report={'trust_list_version':B['trust_list_version'],'rules_per_label':18,'unique_checkpoints':len(B['candidate_ids']),
        'rows':len(test),'same_predictions':True,'accuracy':None,'runtime_seconds':round(time.time()-started,1),
        'inference_quantization_overrides':{'hunyuan_1_8b':'none','minicpm5_2b':'none'},
        'test_csv_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),
        'submissions':['submission_lc3.parquet','submission_lc4.parquet']}
for name in report['submissions']+['submission.parquet']:
    p=Path('/kaggle/working')/name
    report.setdefault('submission_sha256',{})[name]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('/kaggle/working/lc3_lc4_run_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2)); display(pd.read_parquet('/kaggle/working/submission.parquet').head())'''
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# LC3 and LC4 — Trust List 2 probability cascades\n\nRuns the frozen best-round configurations from the latest Trust List 2. At their selected best rounds, LC3 round 18 and LC4 round 6 each use 18 rules for every label; under the precision-weighted mean formula these therefore have identical predictions. Checkpoints are loaded and released sequentially."),
        nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(inference)],
        metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    KERNEL_DIR.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,KERNEL_DIR/'lc3_lc4.ipynb')
    sources=["zodrak/zod006-models","zodrak/lc3-lc4-checkpoints-v035",
             "zodrak/three-strategy-checkpoints",
             "zodrak/ensemble-hypotheses-nov15-extra-checkpoints",
             "zodrak/zod03-llama-base","johnnyhyland/smollm3-3b-base-36l-text",
             "pingjiale/qwen3-0-6b"]
    model_sources=["google/gemma-4/transformers/gemma-4-e2b/1",
                   "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
                   "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
                   "qwen-lm/qwen-3/transformers/1.7b-base/1",
                   "huangxin14/hunyuan-1-8b/pytorch/default/1"]
    (KERNEL_DIR/'kernel-metadata.json').write_text(json.dumps({
        "id":"zodrak/lc3-and-lc4-trust-list-2-v035","title":"LC3 and LC4 Trust List 2 v035",
        "code_file":"lc3_lc4.ipynb","language":"python","kernel_type":"notebook",
        "is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "dataset_sources":sources,"kernel_sources":[],"competition_sources":["name-that-llm"],
        "model_sources":model_sources,"machine_shape":"NvidiaTeslaT4"},indent=2)+"\n")
    return KERNEL_DIR


def main() -> None:
    manifest=build_assets()
    kernel=build_kernel()
    files=[p for p in DATA_DIR.iterdir() if p.is_file()]
    size=sum(p.stat().st_size for p in files)
    print(json.dumps({"trust_list_version":manifest["trust_list_version"],
                      "selected_rules_per_strategy":324,"unique_checkpoints":len(manifest["candidate_ids"]),
                      "already_present":sum(c.get("already_in_zod006_dataset",False) for c in manifest["candidates"].values()),
                      "added_checkpoint_count":sum(not c.get("already_in_zod006_dataset",False) for c in manifest["candidates"].values()),
                      "dataset_files":len(files),"dataset_logical_bytes":size,
                      "dataset_logical_gib":round(size/1024**3,3),"kernel_dir":str(kernel)},indent=2))


if __name__ == '__main__':
    main()
