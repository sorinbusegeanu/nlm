"""Package the E12-2 deployment and prepare its offline Kaggle notebook.

Checkpoint payloads already published in earlier immutable Kaggle shards are
referenced by dataset and filename. Only missing payloads are staged into the
new E12-2 shard.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import nbformat as nbf

from scripts.publish_kaggle_checkpoint_store import ROOT, build_checkpoint, kaggle_file_inventory, sha256_file

RESULTS = ROOT / "artifacts/analysis/ensemble_catalog/20261120T090715676658Z/results.json"
TRUST = ROOT / "docs/trust-list-2-v036.csv"
KAGGLE_ROOT = ROOT / "artifacts/kaggle/e12-2-kernel"
SHARD_DIR = ROOT / "artifacts/kaggle/e12-2-checkpoint-shard-v001"
SHARD_REF = "zodrak/e12-2-checkpoints-v001"
KERNEL_REF = "zodrak/e12-2-top-3-pairwise-arbitration"

DATASETS = {
    "zodrak/zod006-models": "zod006-models",
    "zodrak/lc3-lc4-checkpoints-v035": "lc3-lc4-checkpoint-shard-v035",
    "zodrak/three-strategy-checkpoints": "three-strategy-checkpoints",
    "zodrak/ensemble-hypotheses-nov15-extra-checkpoints": "ensemble-hypotheses-nov15-extra-checkpoints",
    "zodrak/trust2-v013-top5-models": "trust2-v013-top5-models",
}

NOTEBOOK_CODE = r'''from pathlib import Path
import gc, hashlib, importlib.metadata as importlib_metadata, json, os, shutil, subprocess, sys, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch

INPUT=Path('/kaggle/input')
print('Kaggle input entries:', sorted(p.name for p in INPUT.iterdir()))
DATASET_MARKERS={
    'zodrak/e12-2-checkpoints-v001':'e12_2_manifest.json',
    'zodrak/lc3-lc4-checkpoints-v035':'lc3_lc4_manifest.json',
    'zodrak/zod006-models':'zod006_manifest.json',
    'zodrak/three-strategy-checkpoints':'three_strategy_supplement_manifest.json',
    'zodrak/ensemble-hypotheses-nov15-extra-checkpoints':'ensemble_hypotheses_nov15_extras_manifest.json',
}
ROOTS={}
for ref,marker in DATASET_MARKERS.items():
    hits=[p.parent for p in INPUT.rglob(marker) if p.is_file()]
    if len(hits)==1: ROOTS[ref]=hits[0]
def only(paths, description):
    paths=list(dict.fromkeys(Path(p) for p in paths))
    assert len(paths)==1, f'{description}: expected exactly one path; got {paths[:8]}'
    return paths[0]
def dataset_root(dataset_ref):
    if dataset_ref not in ROOTS:
        marker=DATASET_MARKERS.get(dataset_ref)
        hits=[p.parent for p in INPUT.rglob(marker)] if marker else []
        if len(hits)!=1:
            raise AssertionError(f'missing or ambiguous Kaggle dataset input {dataset_ref}: {hits[:8]}')
        ROOTS[dataset_ref]=hits[0]
    return ROOTS[dataset_ref]
def resolve_asset(asset):
    root=dataset_root(asset['dataset_ref'])
    filename=Path(asset['flat_filename']).name
    path=only([p for p in root.rglob(filename) if p.is_file()], f"{asset['dataset_ref']}/{filename}")
    assert path.stat().st_size==int(asset['size']), f'asset size mismatch: {path}'
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''): hasher.update(chunk)
    digest=hasher.hexdigest()
    assert digest==asset['sha256'], f'asset SHA-256 mismatch: {path}'
    return path

bundle_path=only([dataset_root('zodrak/e12-2-checkpoints-v001')/'e12_2_manifest.json'],'E12-2 manifest')
B=json.loads(bundle_path.read_text())
assert B['ensemble']=='E12-2' and B['candidate_count']==3 and len(B['members'])==15
assert B['classes']==B['deployment']['classes']
assert torch.cuda.is_available(), 'Kaggle GPU unavailable'

# Reuse the same pinned NLM inference runtime already published with LC3/LC4.
runtime_root=dataset_root('zodrak/lc3-lc4-checkpoints-v035')
runtime_manifest=json.loads((runtime_root/'lc3_lc4_runtime_manifest.json').read_text())
WORK=Path('/kaggle/working/e12_2_runtime'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)
for name,asset in runtime_manifest.items():
    (SRC/name).symlink_to(resolve_asset(asset))
sys.path.insert(0,str(WORK))
try:
    from packaging.version import Version
    torchao_version=Version(importlib_metadata.version('torchao'))
except importlib_metadata.PackageNotFoundError:
    torchao_version=None
if torchao_version is not None and torchao_version < Version('0.16.0'):
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
from src.inference import predict_candidate

# Reconstruct only the local base models required by E12-2's LoRA checkpoints.
def resolve_model(marker):
    hits=[p.parent for p in INPUT.rglob('config.json') if marker.casefold() in str(p.parent).casefold()]
    if not hits: raise FileNotFoundError(f'base model not mounted: {marker}')
    depth=min(len(p.parts) for p in hits)
    return only(sorted(set(p for p in hits if len(p.parts)==depth)),marker)
BASES={
    'embeddinggemma2':resolve_model('embeddinggemma-2'),
    'ministral3_3b':resolve_model('ministral-3-3b-base-2512'),
    'qwen06':resolve_model('Qwen3-0.6B'),
}
granite_manifest=json.loads((runtime_root/'lc3_lc4_granite_base_manifest.json').read_text())
for model in ('granite4_h_micro','granite4_350m'):
    files=granite_manifest['models'].get(model,[])
    assert files, f'missing pinned Granite base assets for {model}'
    base=WORK/'bases'/model; base.mkdir(parents=True,exist_ok=True)
    for item in files:
        src=resolve_asset(item); dest=base/item['relative_path']; dest.parent.mkdir(parents=True,exist_ok=True)
        dest.symlink_to(src)
    BASES[model]=base

checkpoint_root=WORK/'checkpoints'; checkpoint_root.mkdir(parents=True,exist_ok=True)
specs={}
for member in B['members']:
    cid=member['candidate_id']; target=checkpoint_root/cid; target.mkdir(parents=True,exist_ok=True)
    for item in member['files']:
        src=resolve_asset(item); dest=target/item['path']; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.symlink_to(src)
    checkpoint=target/member['checkpoint_relative_path']
    assert checkpoint.exists(), f'checkpoint path missing for {cid}: {checkpoint}'
    specs[cid]={**member,'checkpoint':str(checkpoint)}

test_paths=[p for p in (INPUT/'competitions/name-that-llm/test.csv',INPUT/'name-that-llm/test.csv') if p.is_file()]
TEST=only(test_paths,'competition test.csv')
test=pd.read_csv(TEST,dtype={'id':'string'}); assert {'id','text'} <= set(test.columns)
assert test.id.notna().all() and test.id.is_unique and test.text.notna().all()
texts=test.text.astype(str).tolist(); classes=B['classes']; n=len(texts)

probabilities=[]; families=[]; ids=[]
started=time.time()
for index,member in enumerate(B['members'],1):
    cid=member['candidate_id']; spec=specs[cid]
    if spec['model'] in BASES: spec['base_model']=str(BASES[spec['model']])
    probs=predict_candidate(spec,texts,classes,WORK)
    assert probs.shape==(n,len(classes)) and np.isfinite(probs).all(), f'invalid probabilities from {cid}'
    probabilities.append(probs.astype(np.float64)); families.append(member['family']); ids.append(cid)
    del probs, spec; gc.collect(); torch.cuda.empty_cache()
    if index%3==0 or index==len(B['members']):
        print(f'inferred {index}/{len(B["members"])} checkpoints; elapsed={time.time()-started:.1f}s')
probs=np.stack(probabilities)

# This mirrors src.e12.pair_features, fit_classifier(StandardScaler with_mean=False),
# and arbitrate using the full-V0 deployment coefficients and reliability tables.
dep=B['deployment']; k=len(classes); m,rows,_=probs.shape
precision=np.asarray(dep['precision'],dtype=float)
recall=np.asarray(dep['recall'],dtype=float)
directional=np.asarray(dep['directional_confusion'],dtype=float)
scale=np.asarray(dep['feature_scale'],dtype=float); coef=np.asarray(dep['coefficients'],dtype=float)
assert precision.shape==(m,k) and recall.shape==(m,k) and directional.shape==(m,k,k)
assert scale.shape==coef.shape and len(scale)==m*7+3
base=probs.mean(axis=0)
candidate_indices=np.argsort(-base,axis=1,kind='stable')[:,:3]
ranks=np.argsort(np.argsort(-probs,axis=2,kind='stable'),axis=2,kind='stable')
base_ranks=np.argsort(np.argsort(-base,axis=1,kind='stable'),axis=1,kind='stable')
tops=probs.argmax(axis=2); row=np.arange(rows); pair_scores=np.zeros((rows,3),dtype=np.float64)
def pair_features(a,b):
    parts=[]
    for j in range(m):
        pa,pb=probs[j,row,a],probs[j,row,b]; delta=pa-pb
        parts.extend([delta,delta*(pa+pb),(ranks[j,row,b]-ranks[j,row,a])/k,
                      (tops[j]==a).astype(float)-(tops[j]==b).astype(float),
                      precision[j,a]-precision[j,b],recall[j,a]-recall[j,b],
                      directional[j,b,a]-directional[j,a,b]])
    fa,fb=np.zeros(rows),np.zeros(rows)
    for family in sorted(set(families)):
        group=np.flatnonzero(np.asarray(families)==family)
        fa+=np.any(tops[group]==a,axis=0); fb+=np.any(tops[group]==b,axis=0)
    parts.extend([fa-fb,base[row,a]-base[row,b],(base_ranks[row,b]-base_ranks[row,a])/k])
    return np.column_stack(parts)
for left,right in ((0,1),(0,2),(1,2)):
    a,b=candidate_indices[:,left],candidate_indices[:,right]
    z=(pair_features(a,b)/scale)@coef
    p=1/(1+np.exp(-np.clip(z,-60,60)))
    pair_scores[:,left]+=p; pair_scores[:,right]+=1-p
pred_indices=candidate_indices[np.arange(rows),pair_scores.argmax(axis=1)]
predictions=np.asarray(classes,dtype=object)[pred_indices].astype(str)
submission=pd.DataFrame({'id':test.id.astype(str),'model':predictions})
assert submission.columns.tolist()==['id','model'] and len(submission)==len(test)
assert submission.id.tolist()==test.id.astype(str).tolist() and submission.id.is_unique
assert submission.model.isin(classes).all()
output=Path('/kaggle/working/submission.parquet'); submission.to_parquet(output,index=False)
check=pd.read_parquet(output); assert check.equals(submission)
run={
    'ensemble':'E12-2','algorithm':'top-3 pairwise arbitration','rows':len(submission),
    'checkpoint_ids':ids,'checkpoint_experiments':[x['experiment_id'] for x in B['members']],
    'v0_cross_fitted_accuracy':B['cross_fitted_accuracy'],
    'v0_cross_fitted_balanced_accuracy':B['cross_fitted_balanced_accuracy'],
    'v0_deployment_fit_accuracy':dep['fit_accuracy'],
    'v0_deployment_fit_balanced_accuracy':dep['fit_balanced_accuracy'],
    'submission_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
    'test_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),
    'runtime_seconds':round(time.time()-started,1),
    'dataset_shard':B['new_checkpoint_shard'],
}
Path('/kaggle/working/e12_2_run_manifest.json').write_text(json.dumps(run,indent=2)+'\n')
print(json.dumps(run,indent=2)); display(submission.head())
'''


def _load_results() -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(RESULTS.read_text())
    result = next(row for row in data["results"] if row["ensemble"] == "E12-2")
    trust = {row["sequence_id"]: row for row in csv.DictReader(TRUST.open())}
    return data, {"result": result, "trust": trust}


def _remote_assets() -> tuple[dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    inventory = {ref: kaggle_file_inventory(ref) for ref in DATASETS}
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for ref, folder in DATASETS.items():
        root = ROOT / "artifacts/kaggle" / folder
        for catalog_path in root.rglob("checkpoint_catalog.json"):
            catalog = json.loads(catalog_path.read_text())
            default_ref = catalog.get("dataset_ref", ref)
            for digest, asset in catalog.get("assets", {}).items():
                by_hash.setdefault(digest, []).append({
                    "dataset_ref": asset.get("dataset_ref", default_ref),
                    "flat_filename": asset["kaggle_filename"], "sha256": digest,
                    "size": int(asset["size"]),
                })
            for checkpoint in catalog.get("checkpoints", {}).values():
                for item in checkpoint.get("files", []):
                    if item.get("sha256") and item.get("flat_filename"):
                        by_hash.setdefault(item["sha256"], []).append({
                            "dataset_ref": item.get("dataset_ref", default_ref),
                            "flat_filename": item["flat_filename"], "sha256": item["sha256"],
                            "size": int(item.get("size", item.get("size_bytes", 0))),
                        })
        for manifest_path in root.glob("*manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for candidate in manifest.get("candidates", {}).values():
                for item in candidate.get("files", []):
                    digest, filename = item.get("sha256"), item.get("flat_filename")
                    if digest and filename:
                        by_hash.setdefault(digest, []).append({
                            "dataset_ref": ref, "flat_filename": filename, "sha256": digest,
                            "size": int(item.get("size", item.get("size_bytes", 0))),
                        })
    return inventory, by_hash


def _asset_ref(digest: str, size: int, by_hash: dict[str, list[dict[str, Any]]],
               inventory: dict[str, dict[str, int]]) -> dict[str, Any] | None:
    candidates = []
    for item in by_hash.get(digest, []):
        ref, name = item["dataset_ref"], item["flat_filename"]
        if ref not in inventory or not any(Path(remote_name).name == Path(name).name
                                           and remote_size == size
                                           for remote_name, remote_size in inventory[ref].items()):
            continue
        if int(item["size"]) not in (0, size):
            continue
        candidates.append(item)
    if not candidates:
        return None
    priority = {ref: i for i, ref in enumerate(DATASETS)}
    chosen = min(candidates, key=lambda item: (priority.get(item["dataset_ref"], 99), item["flat_filename"]))
    return {**chosen, "size": size}


def prepare() -> dict[str, Any]:
    data, loaded = _load_results()
    result, trust = loaded["result"], loaded["trust"]
    assert result["strategy"]["algorithm"] == "pairwise_arbitration"
    assert result["strategy"]["candidates"] == 3 and result["strategy"]["max_checkpoints"] == 15
    roster = result["deployment"]["checkpoint_ids"]
    assert len(roster) == 15 and len(set(roster)) == 15
    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)
    (SHARD_DIR / "part_00" / "assets").mkdir(parents=True)
    inventory, by_hash = _remote_assets()
    members, new_assets = [], {}
    all_files = 0
    for sid in roster:
        row = trust[sid]
        experiment_id = row["experiment_id"]
        model = build_checkpoint(experiment_id)
        pred_manifest = json.loads((ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json").read_text())
        checkpoint_path = Path(row["checkpoint_path"])
        member = {
            "candidate_id": sid, "experiment_id": experiment_id, "model": row["model"],
            "family": __import__("src.trust_list2_weighted_vote", fromlist=["_model_family"])._model_family(row["model"]),
            "config": pred_manifest.get("config", {}), "model_revision": pred_manifest.get("model_revision"),
            "calibration_temperature": float(pred_manifest.get("calibration_temperature") or 1.0),
            "checkpoint_relative_path": (Path(sid) / checkpoint_path.name).as_posix(),
            "checkpoint_type": "file" if checkpoint_path.is_file() else "directory",
            "files": [],
        }
        if checkpoint_path.is_file():
            member["checkpoint_relative_path"] = (Path(sid) / checkpoint_path.name).as_posix()
        else:
            member["checkpoint_relative_path"] = Path(sid).as_posix()
        for source, relative in model["files"]:
            digest, size = sha256_file(source), source.stat().st_size
            ref = _asset_ref(digest, size, by_hash, inventory)
            if ref is None:
                filename = f"assets/asset__{digest}"
                destination = SHARD_DIR / "part_00" / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                ref = {"dataset_ref": SHARD_REF, "flat_filename": filename,
                       "sha256": digest, "size": size}
                new_assets[digest] = {"size": size, "kaggle_filename": filename,
                                      "dataset_ref": SHARD_REF}
            member["files"].append({"path": (Path(sid) / relative).as_posix(), **ref})
            all_files += 1
        members.append(member)

    deployment = {key: value for key, value in result["deployment"].items()
                  if key not in {"fit_ids", "inner_folds"}}
    deployment["classes"] = data["labels"]
    bundle = {
        "schema_version": 1, "ensemble": "E12-2", "algorithm": "pairwise_arbitration",
        "candidate_count": 3, "max_checkpoints": 15,
        "source_analysis": str(RESULTS.relative_to(ROOT)),
        "source_analysis_sha256": sha256_file(RESULTS),
        "trust_list": "trust-list-2-v036", "trust_list_sha256": sha256_file(TRUST),
        "classes": data["labels"], "members": members, "deployment": deployment,
        "cross_fitted_accuracy": result["accuracy"],
        "cross_fitted_balanced_accuracy": result["balanced_accuracy"],
        "deployment_fit_accuracy": result["deployment"]["fit_accuracy"],
        "deployment_fit_balanced_accuracy": result["deployment"]["fit_balanced_accuracy"],
        "new_checkpoint_shard": SHARD_REF,
        "mounted_checkpoint_datasets": sorted({f["dataset_ref"] for m in members for f in m["files"]}),
    }
    (SHARD_DIR / "e12_2_manifest.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    catalog = {
        "schema_version": 1, "dataset_ref": SHARD_REF, "dataset_version": 1,
        "assets": new_assets,
        "checkpoints": {m["experiment_id"]: {
            "experiment_id": m["experiment_id"], "sequence_id": m["candidate_id"],
            "model": m["model"], "family": m["family"], "config": m["config"],
            "files": m["files"],
        } for m in members},
    }
    (SHARD_DIR / "checkpoint_catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    (SHARD_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "NLM E12-2 Pairwise Arbitration Checkpoints",
        "id": SHARD_REF,
        "licenses": [{"name": "other"}],
        "subtitle": "Only inference assets missing from existing immutable checkpoint shards",
        "description": "E12-2 top-3 pairwise arbitration deployment manifest. Existing checkpoint assets are referenced from their original Kaggle datasets; only missing content-addressed files are included.",
    }, indent=2) + "\n")

    KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# E12-2 · Top-3 Pairwise Arbitration\n\nRuns the frozen full-V0 deployment classifier from the E12-2 ensemble catalogue. The notebook loads 15 exact checkpoints from existing Kaggle shards plus the small E12-2 delta shard, then applies the stored pairwise arbitration coefficients to the competition test rows."),
        nbf.v4.new_code_cell(NOTEBOOK_CODE),
    ])
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    notebook_path = KAGGLE_ROOT / "e12_2.ipynb"
    nbf.write(notebook, notebook_path)
    metadata = {
        "id": KERNEL_REF, "title": "E12-2 Top-3 Pairwise Arbitration",
        "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["name-that-llm", "E12-2", "pairwise-arbitration", "offline-inference"],
        "dataset_sources": sorted({SHARD_REF, "zodrak/lc3-lc4-checkpoints-v035",
                                    "pingjiale/qwen3-0-6b",
                                    *[f["dataset_ref"] for m in members for f in m["files"]]}),
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": [
            "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
        ],
        "machine_shape": "NvidiaTeslaT4",
    }
    (KAGGLE_ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {
        "shard_ref": SHARD_REF, "kernel_ref": KERNEL_REF,
        "roster": [{"sequence_id": m["candidate_id"], "experiment_id": m["experiment_id"],
                    "model": m["model"]} for m in members],
        "new_file_count": len(new_assets), "new_bytes": sum(v["size"] for v in new_assets.values()),
        "all_checkpoint_files": all_files, "mounted_existing_datasets": bundle["mounted_checkpoint_datasets"],
        "cross_fitted_balanced_accuracy": result["balanced_accuracy"],
        "deployment_fit_balanced_accuracy": result["deployment"]["fit_balanced_accuracy"],
        "bundle": str(SHARD_DIR / "e12_2_manifest.json"),
        "kernel_dir": str(KAGGLE_ROOT),
    }
    (KAGGLE_ROOT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
