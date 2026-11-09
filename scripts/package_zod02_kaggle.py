"""Build the independent ZOD02 inference dataset and Kaggle kernel."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts/analysis/zod02"
CATALOG = ROOT / "docs/checkpoint_label_catalog.csv"
OUTPUT_DATA = ROOT / "artifacts/kaggle/zod02-models"
OUTPUT_KERNEL = ROOT / "artifacts/kaggle/zod02-kernel"
SKIP_NAMES = {"README.md", "checkpoint_selection.json", "baseline_validation.json",
              "training_log.json", "training_progress.json", "validation_metrics.json"}


def _copy_checkpoint(source: Path, destination: Path) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    if source.is_file():
        target = destination / source.name
        shutil.copy2(source, target)
        return [{"relative_path": source.name, "flat_filename": target.name}]
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = file.relative_to(source)
        if any(part.startswith("epoch-") for part in relative.parts) or file.name in SKIP_NAMES:
            continue
        flat_name = f"{destination.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, OUTPUT_DATA / flat_name)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": flat_name})
    if not copied:
        raise FileNotFoundError(f"no inference files copied from {source}")
    return copied


def _copy_hf_base(model_id: str, revision: str, prefix: str) -> list[dict]:
    cache_root = Path.home() / ".cache/huggingface/hub"
    repo_folder = "models--" + model_id.replace("/", "--")
    snapshot = cache_root / repo_folder / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing cached base model revision {model_id}@{revision}")
    allowed_suffixes = {".json", ".model", ".txt", ".py", ".tiktoken", ".safetensors"}
    copied = []
    for file in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        relative = file.relative_to(snapshot)
        if relative.parts[0] == "onnx":
            continue
        if file.suffix not in allowed_suffixes:
            continue
        flat_name = f"base__{prefix}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, OUTPUT_DATA / flat_name)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": flat_name})
    if not copied:
        raise FileNotFoundError(f"no usable base files for {model_id}@{revision}")
    return copied


def build_dataset_package() -> Path:
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DATA.iterdir():
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    metrics = json.loads((ANALYSIS / "metrics.json").read_text())
    rules = metrics["final_rules_exploratory"]
    fallback_id = metrics["fallback_candidate_id"]
    ordered_ids = list(dict.fromkeys([rule["candidate_id"] for rule in rules] + [fallback_id]))
    inventory = pd.read_csv(ROOT / "artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv",
                            dtype=str, keep_default_na=False).set_index("experiment_id")
    catalog = pd.read_csv(CATALOG, dtype=str, keep_default_na=False).drop_duplicates("experiment_id").set_index("experiment_id")
    candidates, base_revisions = {}, {}
    for candidate_id in ordered_ids:
        row = inventory.loc[candidate_id]
        manifest_path = ROOT / "artifacts/predictions" / f"{candidate_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source_value = manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path") or row.checkpoint_path
        source = Path(source_value)
        source = source if source.is_absolute() else ROOT / source
        if source.is_dir() and (source / "best_checkpoint").is_dir():
            source = source / "best_checkpoint"
        elif source.is_dir() and manifest.get("selected_epoch"):
            selected_epoch = source / f"epoch-{int(manifest['selected_epoch']):02d}"
            if selected_epoch.is_dir():
                source = selected_epoch
        if not source.exists():
            raise FileNotFoundError(f"checkpoint missing: {candidate_id}: {source}")
        sequence_id = str(catalog.loc[candidate_id, "sequence_id"])
        checkpoint_type = "file" if source.is_file() else "directory"
        files = _copy_checkpoint(source, OUTPUT_DATA / f"ckpt__{sequence_id}")
        model = str(manifest["model"])
        config = manifest.get("config", {})
        base_model = None
        base_files = []
        if model not in {"ultrabert380", "svc", "catboost_stylometry"}:
            # Adapter config is authoritative about the exact base model.
            adapter_config = next((OUTPUT_DATA / item["flat_filename"] for item in files
                                   if item["relative_path"].endswith("adapter_config.json")), None)
            if adapter_config is None:
                adapter_config = source / "adapter_config.json"
            if adapter_config.exists():
                base_model = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
            # Some multimodal model adapters deliberately store null here; the
            # immutable run manifest's checkpoint field is the authoritative id.
            base_model = base_model or config.get("checkpoint")
            if not base_model:
                raise ValueError(f"adapter has no base_model_name_or_path: {candidate_id}")
            base_revisions.setdefault(base_model, None)
            # These three models lack a guaranteed exact Kaggle model source, so bundle their base weights.
            if base_model in {"ibm-granite/granite-4.0-350m-base",
                              "ibm-granite/granite-4.0-h-micro-base",
                              "HuggingFaceTB/SmolLM3-3B-Base"}:
                repo = Path.home() / ".cache/huggingface/hub" / ("models--" + base_model.replace("/", "--"))
                ref = repo / "refs/main"
                revision = str(manifest.get("model_revision") or (ref.read_text().strip() if ref.is_file() else ""))
                if not revision or not (repo / "snapshots" / revision).is_dir():
                    snapshots = sorted((repo / "snapshots").iterdir())
                    if len(snapshots) != 1:
                        raise FileNotFoundError(f"cannot resolve cached revision for {base_model}: {revision}")
                    revision = snapshots[0].name
                if base_model not in base_revisions or base_revisions[base_model] is None:
                    base_revisions[base_model] = revision
                    base_files = _copy_hf_base(base_model, revision, model)
                else:
                    base_files = []
        candidates[sequence_id] = {"candidate_id": candidate_id, "sequence_id": sequence_id,
            "model": model, "checkpoint_type": checkpoint_type, "files": files,
            "classes": list(map(str, manifest["classes"])), "config": config,
            "base_model": base_model, "base_files": base_files,
            "training_dataset": str(row.training_dataset),
            "validation_accuracy": float(row.validation_accuracy),
            "selected_epoch": manifest.get("selected_epoch")}

    # If several candidate adapters share one bundled base, the first candidate owns the file list.
    base_index = {}
    for candidate in candidates.values():
        if candidate["base_model"] in base_revisions and candidate["base_files"]:
            base_index[candidate["base_model"]] = candidate["base_files"]
    for candidate in candidates.values():
        candidate["base_files"] = base_index.get(candidate["base_model"], [])

    for module in ("__init__.py", "data_factory.py", "feature_factory.py", "model_factory.py", "inference.py"):
        shutil.copy2(ROOT / "src" / module, OUTPUT_DATA / f"source__{module}")
    for module in ("trusted_checkpoint_ensemble.py", "zod01.py", "zod01_fixed_cascade.py"):
        # These module files satisfy inference helper imports but no training path is run.
        shutil.copy2(ROOT / "src" / module, OUTPUT_DATA / f"source__{module}")
    ordered_rules = []
    for order, rule in enumerate(rules, start=1):
        sequence_id = str(catalog.loc[rule["candidate_id"], "sequence_id"])
        ordered_rules.append({"order": order, "label": rule["label"], "sequence_id": sequence_id,
            "candidate_id": rule["candidate_id"], "smoothed_precision": rule["smoothed_precision"],
            "prediction_count": rule["prediction_count"]})
    fallback_sequence = str(catalog.loc[fallback_id, "sequence_id"])
    bundle = {"name": "zod02", "classes": next(iter(candidates.values()))["classes"],
        "rules": ordered_rules, "fallback_sequence_id": fallback_sequence,
        "fallback_candidate_id": fallback_id, "candidates": candidates,
        "base_model_revisions": base_revisions, "submission_schema": ["id", "model"],
        "fit_score_exploratory": metrics["final_fit_metrics_exploratory"],
        "oof_zod02_balanced_accuracy": metrics["zod02_oof"]["balanced_accuracy"],
        "oof_zod01_balanced_accuracy": metrics["zod01_oof"]["balanced_accuracy"]}
    (OUTPUT_DATA / "zod02_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (OUTPUT_DATA / "dataset-metadata.json").write_text(json.dumps({"title": "zod02 inference assets",
        "id": "zodrak/zod02-models", "licenses": [{"name": "other"}],
        "subtitle": "Frozen inference assets for the ZOD02 multi-specialist cascade",
        "description": "Inference-only checkpoints and required exact model bases for ZOD02. No training data included."}, indent=2) + "\n")
    return OUTPUT_DATA


def build_kernel() -> Path:
    import nbformat as nbf
    OUTPUT_KERNEL.mkdir(parents=True, exist_ok=True)
    metadata = {"id": "zodrak/zod02", "title": "zod02", "code_file": "zod02.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["offline-inference", "multi-specialist-cascade"],
        "dataset_sources": ["zodrak/zod02-models", "zodrak/zod01-sklearn", "pingjiale/qwen3-0-6b"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": ["google/gemma-4/transformers/gemma-4-e2b/1",
            "qwen-lm/qwen-3/transformers/1.7b-base/1",
            "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1"],
        "machine_shape": "NvidiaTeslaT4"}
    (OUTPUT_KERNEL / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    setup = r'''from pathlib import Path
import gc, json, os, subprocess, sys, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
DATA=Path('/kaggle/input/zod02-models'); BUNDLE=json.loads((DATA/'zod02_manifest.json').read_text())
from importlib.metadata import version, PackageNotFoundError
SKLEARN_VERSION='1.9.1'
if version('scikit-learn') != SKLEARN_VERSION:
    wheel=next(Path('/kaggle/input').rglob('scikit_learn-1.9.1-cp313-cp313-*.whl'),None)
    assert wheel is not None, 'pinned sklearn wheel missing'
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--force-reinstall',str(wheel)],check=True)
import torch, numpy as np, pandas as pd, joblib
assert torch.cuda.is_available(), 'Kaggle GPU unavailable'
WORK=Path('/kaggle/working/zod02_runtime'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)
for name in ('__init__.py','data_factory.py','feature_factory.py','model_factory.py','inference.py','trusted_checkpoint_ensemble.py','zod01.py','zod01_fixed_cascade.py'):
    (SRC/name).symlink_to(DATA/f'source__{name}')
CHECKPOINTS=WORK/'checkpoints'; CHECKPOINTS.mkdir(parents=True,exist_ok=True)
BASES={}
    for seq,candidate in BUNDLE['candidates'].items():
    dest=CHECKPOINTS/seq; dest.mkdir(parents=True,exist_ok=True)
    for item in candidate['files']:
        target=dest/item['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(DATA/item['flat_filename'])
    candidate['runtime_checkpoint']=str(dest if candidate['checkpoint_type']=='directory' else dest/candidate['files'][0]['relative_path'])
    if candidate['base_model'] and candidate['model'] not in BASES:
        bdir=WORK/'bases'/candidate['model']; bdir.mkdir(parents=True,exist_ok=True)
        for item in candidate['base_files']:
            target=bdir/item['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(DATA/item['flat_filename'])
        if (bdir/'config.json').is_file(): BASES[candidate['model']]=bdir
def resolve_model(path,marker):
    path=Path(path)
    if (path/'config.json').is_file(): return path
    for p in Path('/kaggle/input/models').rglob('config.json'):
        if marker.lower() in str(p.parent).lower(): return p.parent
    raise FileNotFoundError(f'model base not found: {marker}')
BASES.setdefault('qwen06',resolve_model('/kaggle/input/qwen3-0-6b/Qwen3-0.6B','Qwen3-0.6B'))
BASES.setdefault('qwen17',resolve_model('/kaggle/input/models/qwen-lm/qwen-3/transformers/1.7b-base/1','1.7b-base'))
BASES.setdefault('gemma4_e2b',resolve_model('/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1','gemma-4-e2b'))
BASES.setdefault('embeddinggemma2',resolve_model('/kaggle/input/models/google/embeddinggemma-2/transformers/embeddinggemma-2/1','embeddinggemma-2'))
BASES.setdefault('ministral3_3b',resolve_model('/kaggle/input/models/mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1','ministral-3-3b-base-2512'))
try:
    from importlib.metadata import version as v
    from packaging.version import Version
    if Version(v('torchao')) <= Version('0.16.0'):
        subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,capture_output=True)
except (PackageNotFoundError, ImportError): pass
sys.path.insert(0,str(WORK))
from src.inference import _predict_neural, predict_catboost_stylometry, predict_svc_bundle
TEST=next((p for p in [Path('/kaggle/input/competitions/name-that-llm/test.csv'),Path('/kaggle/input/name-that-llm/test.csv')] if p.is_file()),None)
assert TEST is not None, 'competition test.csv missing'
print({'rules':len(BUNDLE['rules']),'checkpoints':len(BUNDLE['candidates']),'fallback':BUNDLE['fallback_sequence_id']})'''
    predict = r'''test=pd.read_csv(TEST,dtype={'id':'string'}); assert {'id','text'}.issubset(test.columns)
assert test.id.notna().all() and not test.id.duplicated().any(); test.text=test.text.fillna('').astype(str)
classes=BUNDLE['classes']; ci={x:i for i,x in enumerate(classes)}; n=len(test)
assigned=np.zeros(n,dtype=bool); output=np.empty(n,dtype=object); source=np.empty(n,dtype=object); cache={}; trace=[]
def get_top1(seq, indices):
    c=BUNDLE['candidates'][seq]; model=c['model']; cp=Path(c['runtime_checkpoint']); texts=test.text.iloc[indices].tolist()
    if model=='svc': probs,labels=predict_svc_bundle(joblib.load(cp),texts)
    elif model=='catboost_stylometry': probs,labels=predict_catboost_stylometry(cp,texts,classes)
    else:
        base=BASES.get(model)
        if model not in {'ultrabert380'}: assert base is not None
        bs=min(4,int(c['config'].get('batch_size',4) or 4))
        while True:
            try:
                probs,labels=_predict_neural(model,cp,texts,classes,batch_size=bs,max_length=int(c['config'].get('max_length',384)),base_checkpoint=str(base) if base else None,revision=None,text_only_backbone=bool(c['config'].get('text_only_backbone',False))); break
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower() or bs<=1: raise
                bs=max(1,bs//2); gc.collect(); torch.cuda.empty_cache()
    labels=list(map(str,labels)); order=[labels.index(x) for x in classes]; top=np.asarray(classes,dtype=object)[np.asarray(probs)[:,order].argmax(1)]
    del probs; gc.collect(); torch.cuda.empty_cache(); return top
start=time.time()
for rule in BUNDLE['rules']:
    seq,label=rule['sequence_id'],rule['label']; remaining=np.flatnonzero(~assigned)
    if seq not in cache: cache[seq]=np.full(n,'',dtype=object)
    missing=remaining[cache[seq][remaining]=='']
    if len(missing): cache[seq][missing]=get_top1(seq,missing)
    chosen=remaining[cache[seq][remaining]==label]; assigned[chosen]=True; output[chosen]=label; source[chosen]=seq
    trace.append({'order':rule['order'],'sequence_id':seq,'label':label,'assigned':len(chosen),'remaining':int((~assigned).sum())})
remaining=np.flatnonzero(~assigned); seq=BUNDLE['fallback_sequence_id']
if seq not in cache: cache[seq]=np.full(n,'',dtype=object)
missing=remaining[cache[seq][remaining]=='']
if len(missing): cache[seq][missing]=get_top1(seq,missing)
output[remaining]=cache[seq][remaining]; source[remaining]=seq
submission=pd.DataFrame({'id':test.id,'model':output})
assert submission.columns.tolist()==['id','model'] and len(submission)==len(test)
assert submission.id.astype(str).tolist()==test.id.astype(str).tolist() and not submission.id.duplicated().any()
assert submission.model.notna().all() and submission.model.isin(classes).all()
out=Path('/kaggle/working/submission.parquet'); submission.to_parquet(out,index=False)
pd.DataFrame(trace).to_csv('/kaggle/working/zod02_rule_counts.csv',index=False)
pd.DataFrame({'id':test.id,'model':submission.model,'selected_sequence_id':source}).to_parquet('/kaggle/working/zod02_decisions.parquet',index=False)
print({'approach':'zod02','rows':len(submission),'output':str(out),'runtime_seconds':round(time.time()-start,1),'schema':submission.columns.tolist()})
display(pd.DataFrame(trace)); display(submission.head())'''
    notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell("# ZOD02 — multi-specialist cascade\n\nFrozen rules selected on all 540 V0 validation rows; the measured five-fold grouped out-of-fold balanced accuracy was 52.4%, compared with 61.3% for ZOD01. The full-V0 fit score is exploratory and not an independent estimate. This notebook applies the fixed ZOD02 rules and V1-all → V0 UltraBERT fallback to the competition test set. Internet is disabled."), nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(predict)], metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"cascade":"zod02"})
    nbf.write(notebook,OUTPUT_KERNEL/'zod02.ipynb')
    return OUTPUT_KERNEL


if __name__ == "__main__":
    print(json.dumps({"dataset": str(build_dataset_package()), "kernel": str(build_kernel())}, indent=2))
