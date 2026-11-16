"""Package the corrected Trust List 2 one-rule cascade with the V2.14 fallback."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from scripts.package_zod04_kaggle import _selected_rules


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts/kaggle/zod006-models"
KERNEL_DIR = ROOT / "artifacts/kaggle/zod006-kernel"
LATEST_TRUST = ROOT / "docs/trust-list-2-latest.json"
FALLBACK_ID = (
    "ultrabert380_v2.14.balanced_valv0_datacd67b24a_split727ebb7e_"
    "seed2026_cfg5044b89f_codeab9f6d2"
)
REMOVED_EXPERIMENTS = {
    "svc_v1_datada64f007_split147ce64e_seed2026_cfgbd998651_code667e2d0",
    "ultrabert380_v1_datada64f007_split147ce64e_seed2026_cfgb5a91dee_code667e2d0",
}
SKIP = {"README.md", "training_log.json", "training_progress.json", "batch_size_probe.json",
        "checkpoint_selection.json", "baseline_validation.json", "validation_metrics.json"}


def _copy_checkpoint(source: Path, key: str) -> list[dict]:
    dest = DATA_DIR / f"ckpt__{key}"
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = file.relative_to(source)
        if any(part.startswith("epoch-") for part in relative.parts) or file.name in SKIP:
            continue
        flat = f"{dest.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, DATA_DIR / flat)
        records.append({"relative_path": relative.as_posix(), "flat_filename": flat})
    if not records:
        raise FileNotFoundError(f"no inference files found under {source}")
    return records


def _checkpoint_source(experiment_id: str, manifest: dict) -> Path:
    source = Path(manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path") or "")
    source = source if source.is_absolute() else ROOT / source
    if source.is_dir() and (source / "best_checkpoint").is_dir():
        source = source / "best_checkpoint"
    elif source.is_dir() and manifest.get("selected_epoch"):
        epoch_path = source / f"epoch-{int(manifest['selected_epoch']):02d}"
        if epoch_path.is_dir():
            source = epoch_path
    if not source.exists():
        raise FileNotFoundError(f"checkpoint missing for {experiment_id}: {source}")
    return source


def build_dataset() -> Path:
    latest = json.loads(LATEST_TRUST.read_text())
    trust_path = ROOT / latest["csv_path"]
    metadata_path = ROOT / latest["metadata_path"]
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    trust_meta = json.loads(metadata_path.read_text())
    if trust_meta.get("version") != latest.get("version"):
        raise ValueError("latest Trust List 2 pointer and metadata disagree")
    if "validation_sha256" not in trust_meta or int(trust_meta.get("validation_rows", 0)) != 2699:
        raise ValueError("Trust List 2 is not aligned to the complete 2,699-row V0 dataset")
    trust = trust.loc[
        ~trust.experiment_id.isin(REMOVED_EXPERIMENTS)
        & trust.evaluation_eligible.str.lower().isin({"true", "1"})
        & trust.rule_eligible.str.lower().isin({"true", "1"})
        & trust.checkpoint_available.str.lower().isin({"true", "1"})
    ].copy()
    trust["validation_training_overlap"] = pd.to_numeric(trust.validation_training_overlap, errors="raise")
    trust = trust.loc[trust.validation_training_overlap.eq(0)].copy()
    trust["trust_list_version"] = trust_meta["version"]
    rules, _, _ = _selected_rules(trust, expected_trust_version=trust_meta["version"])
    if len(rules) != 18 or len({rule["label"] for rule in rules}) != 18:
        raise ValueError("expected the first eligible ZOD04 rule for each of the 18 labels")
    if {rule["candidate_id"] for rule in rules} & REMOVED_EXPERIMENTS:
        raise ValueError("a removed checkpoint was selected")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in DATA_DIR.iterdir():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    rule_candidate_ids = list(dict.fromkeys(rule["candidate_id"] for rule in rules))
    candidate_ids = list(dict.fromkeys(rule_candidate_ids + [FALLBACK_ID]))
    candidates = {}
    for experiment_id in candidate_ids:
        manifest_path = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"training manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        source = _checkpoint_source(experiment_id, manifest)
        candidates[experiment_id] = {
            "experiment_id": experiment_id,
            "sequence_id": ("V2.14" if experiment_id == FALLBACK_ID else
                            next(rule["sequence_id"] for rule in rules if rule["candidate_id"] == experiment_id)),
            "model": str(manifest["model"]),
            "files": _copy_checkpoint(source, experiment_id),
            "checkpoint_type": "directory",
            "checkpoint_source": str(source.relative_to(ROOT)),
            "classes": list(map(str, manifest["classes"])),
            "config": manifest.get("config", {}),
            "model_revision": manifest.get("model_revision"),
            "base_model": manifest.get("config", {}).get("checkpoint"),
            "base_model_revision": manifest.get("model_revision"),
            "training_dataset": manifest.get("training_dataset") or manifest.get("dataset"),
        }
    classes = candidates[FALLBACK_ID]["classes"]
    if len(classes) != 18 or any(item["classes"] != classes for item in candidates.values()):
        raise ValueError("candidate checkpoints have inconsistent canonical label ordering")
    if set(rule["label"] for rule in rules) != set(classes):
        raise ValueError("selected rules do not cover all fallback labels")

    rules_path = DATA_DIR / "trust-list-2-selected-rules.json"
    rules_path.write_text(json.dumps(rules, indent=2) + "\n")
    shutil.copy2(trust_path, DATA_DIR / trust_path.name)
    smollm_index = ROOT / "artifacts/kaggle/prediction-run-v15-output/base_models/smollm3_3b/model.safetensors.index.json"
    if not smollm_index.is_file():
        raise FileNotFoundError(f"pinned SmolLM3 weight index is missing: {smollm_index}")
    shutil.copy2(smollm_index, DATA_DIR / "smollm3_base_model.safetensors.index.json")
    bundle = {
        "name": "zod006", "algorithm": "first_eligible_rule_per_label_in_trust_seqid_order_then_v214_fallback",
        "trust_list_version": trust_meta["version"],
        "trust_list_sha256": _sha256_file(trust_path),
        "v0_sha256": trust_meta["validation_sha256"], "v0_rows": 2699,
        "removed_experiment_ids": sorted(REMOVED_EXPERIMENTS),
        "rules": rules, "fallback_experiment_id": FALLBACK_ID,
        "fallback": {"experiment_id": FALLBACK_ID, "sequence_id": "V2.14",
                      "training_dataset": "v2.14.balanced", "training_rows": 63702,
                      "validation_dataset": "v0", "validation_rows": 2699,
                      "checkpoint_selection_metric": "validation_balanced_accuracy",
                      "checkpoint_selection_scope": "full_v0"},
        "candidates": candidates, "classes": classes,
        "submission_schema": ["id", "model"],
        "selection_note": "Exactly one first eligible rule per label; no second rules are applied.",
        "validation_note": "V0 was used for Trust List metrics and for V2.14 fallback checkpoint selection.",
    }
    (DATA_DIR / "zod006_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    for name in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / name, DATA_DIR / f"source__{name}")
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "ZOD006 Trust List 2 cascade assets", "id": "zodrak/zod006-models",
        "licenses": [{"name": "other"}],
        "subtitle": "First eligible rule per label with V2.14 UltraBERT fallback",
        "description": "Inference-only checkpoint files and runtime source for ZOD006.",
    }, indent=2) + "\n")
    return DATA_DIR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_kernel() -> Path:
    import nbformat as nbf

    bundle = json.loads((DATA_DIR / "zod006_manifest.json").read_text())
    models = {candidate["model"] for candidate in bundle["candidates"].values()}
    if models != {"qwen06", "granite4_h_micro", "smollm3_3b", "gemma4_e2b", "ultrabert380"}:
        raise ValueError(f"unexpected ZOD006 model set: {models}")
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
INPUT=Path('/kaggle/input')
manifests=list(INPUT.rglob('zod006_manifest.json'))
assert len(manifests)==1, f'expected one ZOD006 manifest, got {manifests}'
DATA=manifests[0].parent; BUNDLE=json.loads(manifests[0].read_text())
assert BUNDLE['algorithm']=='first_eligible_rule_per_label_in_trust_seqid_order_then_v214_fallback'
assert len(BUNDLE['rules'])==18 and len(BUNDLE['classes'])==18
assert len({r['label'] for r in BUNDLE['rules']})==18
assert BUNDLE['fallback']['training_rows']==63702 and BUNDLE['fallback']['validation_rows']==2699
WORK=Path('/kaggle/working/zod006_runtime'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)
for name in ('__init__.py','data_factory.py','feature_factory.py','model_factory.py','inference.py'):
    (SRC/name).symlink_to(DATA/f'source__{name}')
sys.path.insert(0,str(WORK))
CHECKPOINTS=WORK/'checkpoints'; CHECKPOINTS.mkdir(parents=True,exist_ok=True)
for key,c in BUNDLE['candidates'].items():
    dest=CHECKPOINTS/key; dest.mkdir(parents=True,exist_ok=True)
    for f in c['files']:
        src=DATA/f['flat_filename']; assert src.is_file(), f'missing {src}'
        target=dest/f['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(src)
    c['runtime_checkpoint']=str(dest)
assert len(list(INPUT.rglob('zod03_manifest.json')))==1, 'zod03-models base assets missing'
ZOD03_PATH=next(INPUT.rglob('zod03_manifest.json')); ZOD03_DATA=ZOD03_PATH.parent
ZOD03=json.loads(ZOD03_PATH.read_text()); BASES={}
def resolve(marker):
    matches=[p.parent for p in INPUT.rglob('config.json') if marker.lower() in str(p.parent).lower()]
    assert matches, f'base model source not mounted: {marker}'
    depth=min(len(p.parts) for p in matches); roots=sorted({p for p in matches if len(p.parts)==depth})
    assert len(roots)==1, f'ambiguous model source for {marker}: {roots}'
    return roots[0]
BASES['qwen06']=resolve('Qwen3-0.6B')
BASES['smollm3_3b']=resolve('smollm3-3b-base')
BASES['gemma4_e2b']=resolve('gemma-4-e2b')
# The attached SmolLM3 dataset has the nonstandard model_type
# `smollm3_text_base`; normalize it in a small runtime overlay to the native
# Transformers architecture name while symlinking its large weights.
smollm_source=BASES['smollm3_3b']; smollm_runtime=WORK/'bases/smollm3_3b'
smollm_runtime.mkdir(parents=True,exist_ok=True)
for source_file in smollm_source.rglob('*'):
    if not source_file.is_file(): continue
    target=smollm_runtime/source_file.relative_to(smollm_source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if source_file.name=='config.json':
        config=json.loads(source_file.read_text())
        if config.get('model_type')=='smollm3_text_base': config['model_type']='smollm3'
        target.write_text(json.dumps(config,indent=2)+'\n')
    else:
        target.symlink_to(source_file)
smollm_indexes=list(INPUT.rglob('smollm3_base_model.safetensors.index.json'))
assert len(smollm_indexes)==1, f'SmolLM3 safetensors shard index missing or ambiguous: {smollm_indexes}'
smollm_index=smollm_indexes[0]
(smollm_runtime/'model.safetensors.index.json').symlink_to(smollm_index)
BASES['smollm3_3b']=smollm_runtime
granite=next((c for c in ZOD03['candidates'].values()
              if c.get('base_model')=='ibm-granite/granite-4.0-h-micro-base'
              and c.get('base_model_revision')=='372ede0bc484284e08ce574e773c96cd2b89b367'
              and c.get('base_files')),None)
assert granite is not None, 'exact pinned Granite H Micro base missing from ZOD03 assets'
BASES['granite4_h_micro']=WORK/'bases/granite4_h_micro'; BASES['granite4_h_micro'].mkdir(parents=True,exist_ok=True)
for f in granite['base_files']:
    src=ZOD03_DATA/f['flat_filename']; assert src.is_file(), f'missing Granite base file {src}'
    target=BASES['granite4_h_micro']/f['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(src)
# PEFT's optional TorchAO dispatcher raises on Kaggle's older preinstalled TorchAO.
try:
    from importlib.metadata import PackageNotFoundError, version as package_version
    from packaging.version import Version
    try:
        if Version(package_version('torchao')) <= Version('0.16.0'):
            subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,capture_output=True)
    except PackageNotFoundError:
        pass
except ImportError:
    pass
import torch, numpy as np, pandas as pd
assert torch.cuda.is_available(), 'Kaggle GPU unavailable'
from src.inference import _predict_neural
TEST=next((p for p in (INPUT/'competitions/name-that-llm/test.csv',INPUT/'name-that-llm/test.csv') if p.is_file()),None)
assert TEST is not None, 'competition test.csv missing'
test=pd.read_csv(TEST,dtype={'id':'string'}); assert {'id','text'}.issubset(test.columns)
assert test.id.notna().all() and not test.id.duplicated().any()
texts=test.text.fillna('').astype(str).tolist(); classes=list(map(str,BUNDLE['classes']))
'''
    inference = r'''def predict_top1(key):
    c=BUNDLE['candidates'][key]; model=c['model']; cp=Path(c['runtime_checkpoint'])
    batch=max(1,min(4,int(c.get('config',{}).get('inference_batch_size',c.get('config',{}).get('batch_size',4)) or 4)))
    while True:
        try:
            probs,labels=_predict_neural(model,cp,texts,classes,batch_size=batch,
                max_length=int(c.get('config',{}).get('max_length',384) or 384),
                base_checkpoint=str(BASES[model]) if model in BASES else None,
                revision=None,text_only_backbone=bool(c.get('config',{}).get('text_only_backbone',False)))
            break
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower() or batch<=1: raise
            gc.collect(); torch.cuda.empty_cache(); batch=max(1,batch//2)
    labels=list(map(str,labels)); order=[labels.index(label) for label in classes]
    prediction=np.asarray(classes,dtype=object)[np.asarray(probs)[:,order].argmax(axis=1)].astype(str)
    del probs; gc.collect(); torch.cuda.empty_cache()
    return prediction
started=time.time()
top1={key:predict_top1(key) for key in BUNDLE['candidates']}
pending=np.ones(len(test),dtype=bool); output=np.empty(len(test),dtype=object); source=np.empty(len(test),dtype=object)
for rule in BUNDLE['rules']:
    key,label=rule['candidate_id'],rule['label']
    claim=pending & (top1[key]==label)
    output[claim],source[claim],pending[claim]=label,key,False
fallback=BUNDLE['fallback_experiment_id']
output[pending]=top1[fallback][pending]; source[pending]=fallback
submission=pd.DataFrame({'id':test.id.astype(str),'model':output.astype(str)})
assert submission.columns.tolist()==['id','model'] and len(submission)==len(test)
assert submission.id.tolist()==test.id.astype(str).tolist() and submission.id.is_unique
assert submission.model.notna().all() and submission.model.isin(classes).all()
out=Path('/kaggle/working/submission.parquet'); submission.to_parquet(out,index=False)
submission.to_csv('/kaggle/working/submission.csv',index=False)
pd.DataFrame({'id':test.id.astype(str),'model':output.astype(str),'selected_experiment_id':source.astype(str)}).to_parquet('/kaggle/working/zod006_decisions.parquet',index=False)
pd.DataFrame([{'order':r['order'],'label':r['label'],'sequence_id':r['sequence_id'],
    'experiment_id':r['candidate_id'],'trust_sequence':r['trust_sequence']} for r in BUNDLE['rules']]).to_csv('/kaggle/working/zod006_rules.csv',index=False)
print({'kernel':'zod006','trust_list':BUNDLE['trust_list_version'],'rows':len(submission),
    'rules':len(BUNDLE['rules']),'rule_covered':int((~pending).sum()),'fallback_rows':int(pending.sum()),
    'fallback':fallback,'runtime_seconds':round(time.time()-started,1),'submission':str(out)})
display(submission.head())'''
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(
            "# ZOD006 — corrected Trust List 2 cascade\n\n"
            "One first-eligible Trust List 2 rule per label, in SeqID order, followed by the "
            "UltraBERT V2.14 fallback selected by validation balanced accuracy. The fallback "
            "was selected using all V0 rows; local V0 metrics are development diagnostics, not "
            "an independent test estimate. This notebook runs offline inference only and writes "
            "the competition submission as `submission.parquet`."),
        nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(inference)],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "cascade": "zod006_trust_list_2_v004_v214_fallback"})
    nbf.write(notebook, KERNEL_DIR / "zod006.ipynb")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "zodrak/zod006", "title": "ZOD006", "code_file": "zod006.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["zodrak/zod006-models", "zodrak/zod03-models",
                            "johnnyhyland/smollm3-3b-base-36l-text", "pingjiale/qwen3-0-6b"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": ["google/gemma-4/transformers/gemma-4-e2b/1"],
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2) + "\n")
    return KERNEL_DIR


if __name__ == "__main__":
    data = build_dataset()
    kernel = build_kernel()
    size = sum(path.stat().st_size for path in data.rglob("*") if path.is_file())
    print(json.dumps({"dataset": str(data), "bytes": size, "kernel": str(kernel)}, indent=2))
