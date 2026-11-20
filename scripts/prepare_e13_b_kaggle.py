"""Prepare a deployable E13-B Kaggle kernel and its missing-checkpoint shard."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from scripts.prepare_e12_2_kaggle import _remote_assets, _asset_ref
from scripts.publish_kaggle_checkpoint_store import build_checkpoint, sha256_file
from src import ensemble_catalog as catalog
from src.e13_reliability_router import (
    CATBOOST_PARAMS, CATEGORICAL, CONFIGS, FEATURE_COLUMNS, _cat_frame, _new_catboost,
    _sorted_ids, _targets, build_feature_frame, select_roster,
)
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts/analysis/e13/20261120T093318893046Z"
SUPPLEMENT = ROOT / "artifacts/kaggle/e13-b-checkpoint-shard-v001"
KERNEL_DIR = ROOT / "artifacts/kaggle/e13-b-kernel"
SUPPLEMENT_REF = "zodrak/e13-b-checkpoints-v001"
KERNEL_REF = "zodrak/e13-b-trust-list-2-v037"
LC3_REF = "zodrak/lc3-lc4-checkpoints-v035"

NOTEBOOK_CODE = r'''from pathlib import Path
import gc, hashlib, importlib.metadata as im, json, os, shutil, subprocess, sys, time, zipfile
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, pandas as pd, torch
from catboost import CatBoostClassifier

INPUT=Path('/kaggle/input')
def only(paths,desc):
    paths=list(dict.fromkeys(Path(p) for p in paths)); assert len(paths)==1, f'{desc}: {paths[:8]}'; return paths[0]
def root_for(marker): return only([p.parent for p in INPUT.rglob(marker) if p.is_file()],marker)
LC3=root_for('lc3_lc4_manifest.json'); BROOT=root_for('e13_b_manifest.json')
DATASET_MARKERS={'zodrak/lc3-lc4-checkpoints-v035':'lc3_lc4_manifest.json','zodrak/zod006-models':'zod006_manifest.json','zodrak/three-strategy-checkpoints':'three_strategy_supplement_manifest.json','zodrak/ensemble-hypotheses-nov15-extra-checkpoints':'ensemble_hypotheses_nov15_extras_manifest.json','zodrak/trust2-v013-top5-models':'trust2_v013_top5_manifest.json','zodrak/e13-b-checkpoints-v001':'e13_b_manifest.json'}
def dataset_root(ref): return root_for(DATASET_MARKERS[ref])
B=json.loads((BROOT/'e13_b_manifest.json').read_text())
assert B['ensemble']=='E13-B' and B['routing_method']=='family_max' and len(B['members'])==15
assert B['classes']==B['deployment']['classes']
assert torch.cuda.is_available(), 'Kaggle GPU is required for checkpoint inference'
def resolve_asset(item):
    filename=Path(item['flat_filename']).name
    root=dataset_root(item['dataset_ref'])
    hits=[p for p in root.rglob(filename) if p.is_file()]
    if not hits:
        extracted=Path('/kaggle/working/e13_b_unpacked')/item['dataset_ref'].replace('/','_')/filename
        extracted.parent.mkdir(parents=True,exist_ok=True)
        for archive in root.rglob('*.zip'):
            with zipfile.ZipFile(archive) as zf:
                names=[name for name in zf.namelist() if Path(name).name==filename]
                if names:
                    assert len(names)==1, f'ambiguous {filename} in {archive}'
                    with zf.open(names[0]) as src, extracted.open('wb') as dst: shutil.copyfileobj(src,dst)
                    hits=[extracted]; break
    path=only(hits,f"{item['dataset_ref']}/{filename}")
    assert path.stat().st_size==int(item['size']), f'size mismatch {path}'
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    assert digest.hexdigest()==item['sha256'], f'hash mismatch {path}'
    return path

# Mount the exact project inference runtime from the tested LC3/LC4 shard.
runtime=json.loads((LC3/'lc3_lc4_runtime_manifest.json').read_text())
WORK=Path('/kaggle/working/e13_b_runtime'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)
for name,item in runtime.items(): (SRC/name).symlink_to(resolve_asset(item))
sys.path.insert(0,str(WORK))
sys.path.insert(0,str(BROOT))
try:
    from packaging.version import Version
    tv=Version(im.version('torchao'))
    if tv<Version('0.16.0'): subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
except im.PackageNotFoundError: pass
from src.inference import predict_candidate

# Reconstruct the two Granite base models from the immutable LC3 base payloads.
granite=json.loads((LC3/'lc3_lc4_granite_base_manifest.json').read_text())
BASES={}
for model in ('granite4_h_micro','granite4_350m'):
    files=granite['models'].get(model,[]); assert files, f'missing base payload for {model}'
    dest=WORK/'bases'/model; dest.mkdir(parents=True,exist_ok=True)
    for item in files:
        src=resolve_asset(item); target=dest/item['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(src)
    BASES[model]=dest
for model,marker in [('embeddinggemma2','embeddinggemma-2'),('ministral3_3b','ministral-3-3b-base-2512'),('qwen06','qwen-3/transformers/0.6b')]:
    paths=[p.parent for p in INPUT.rglob('config.json') if marker.casefold() in str(p.parent).casefold()]
    assert paths, f'missing base model {marker}'; depth=min(len(p.parts) for p in paths)
    BASES[model]=only([p for p in paths if len(p.parts)==depth],marker)

members=B['members']; classes=B['classes']
test_paths=[INPUT/'competitions/name-that-llm/test.csv',INPUT/'name-that-llm/test.csv']
present=[p for p in test_paths if p.is_file()]; assert present, 'competition test.csv is missing'
texts_file=present[0]
test=pd.read_csv(texts_file,dtype={'id':'string'}); assert {'id','text'}<=set(test.columns)
assert test.id.notna().all() and test.id.is_unique and test.text.notna().all()
texts=test.text.astype(str).tolist(); probs=[]; started=time.time()
for num,m in enumerate(members,1):
    checkpoint_root=WORK/'checkpoints'/m['candidate_id']; checkpoint_root.mkdir(parents=True,exist_ok=True)
    for item in m['files']:
        src=resolve_asset(item); dest=checkpoint_root/item['relative_path']; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.symlink_to(src)
    spec={k:v for k,v in m.items() if k not in ('files','deployment')}
    spec['checkpoint']=m['checkpoint_relative_path']
    if m['model'] in BASES: spec['base_model']=str(BASES[m['model']])
    p=predict_candidate(spec,texts,classes,WORK)
    assert p.shape==(len(texts),len(classes)) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-4)
    probs.append(p.astype(np.float64)); del p; gc.collect(); torch.cuda.empty_cache()
    if num%3==0 or num==len(members): print(f'inferred {num}/{len(members)} checkpoints')
probs=np.stack(probs)

# Feature builder is packaged with the exact E13 implementation used for fitting.
from e13_b_inference import make_features
def route_family_max(hard,reliability,probs,families):
    hard=np.asarray(hard); reliability=np.asarray(reliability); probs=np.asarray(probs)
    m,n=hard.shape; k=probs.shape[1]; scores=np.full((n,k),-np.inf)
    for row in range(n):
        for label in range(k):
            voters=np.flatnonzero(hard[:,row]==label)
            if len(voters):
                by_family={}
                for index in voters:
                    family=families[index]
                    by_family[family]=max(by_family.get(family,-np.inf),reliability[index,row])
                scores[row,label]=np.mean(list(by_family.values()))
    pred=np.empty(n,dtype=np.int64)
    for row in range(n):
        best=scores[row].max(); tied=np.flatnonzero(np.isclose(scores[row],best,rtol=0,atol=1e-12))
        pred[row]=min(tied,key=lambda label:(-probs[row,label],label))
    return pred
dep=B['deployment']; features=make_features(probs,members,classes,dep['reliability_statistics'])
router=CatBoostClassifier(); router.load_model(str(BROOT/'e13_b_router.cbm'))
scores=router.predict_proba(features)[:,1].reshape(len(texts),len(members)).T
families=np.asarray([m['family'] for m in members]); family_names=sorted(set(families))
ensemble_probs=np.mean([probs[families==f].mean(0) for f in family_names],axis=0)
ensemble_probs/=ensemble_probs.sum(1,keepdims=True)
pred=route_family_max(probs.argmax(2),scores,ensemble_probs,families)
submission=pd.DataFrame({'id':test.id.astype(str),'model':np.asarray(classes)[pred].astype(str)})
assert submission.id.tolist()==test.id.astype(str).tolist() and submission.id.is_unique
assert submission.model.notna().all() and submission.model.isin(classes).all()
out=Path('/kaggle/working/submission.parquet'); submission.to_parquet(out,index=False)
assert pd.read_parquet(out).equals(submission)
run={'ensemble':'E13-B','trust_list':B['trust_list'],'checkpoint_ids':[m['candidate_id'] for m in members],
     'rows':len(submission),'test_sha256':hashlib.sha256(texts_file.read_bytes()).hexdigest(),
     'submission_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'runtime_seconds':round(time.time()-started,1)}
Path('/kaggle/working/e13_b_run_manifest.json').write_text(json.dumps(run,indent=2)+'\n')
print(json.dumps(run,indent=2)); display(submission)
'''


def _deployment(v0, ids, members, probabilities, roster_indices):
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    groups = v0.prompt_id.astype(str).to_numpy()
    ids15 = [ids[i] for i in roster_indices]
    families = [_model_family(members[sid]["model"]) for sid in ids15]
    p = np.stack([probabilities[sid] for sid in ids15]).astype(np.float64)
    Xs, ys, fit_folds = [], [], []
    for fit, held in GroupKFold(n_splits=4).split(v0, truth, groups):
        frame, _ = build_feature_frame(p, ids15, families, truth[fit], fit, held)
        hard = p[:, held].argmax(axis=2)
        Xs.append(frame)
        ys.append(_targets(hard, truth[held]))
        fit_folds.append({"fit_rows": len(fit), "held_rows": len(held),
                          "fit_groups": len(set(groups[fit])), "held_groups": len(set(groups[held]))})
    X, y = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    previous = json.loads((RUN_DIR / "results.json").read_text())
    e13a = next(x for x in previous["results"] if x["ensemble"] == "E13-A")
    iterations = max(1, int(np.median([f["best_iterations"] for f in e13a["folds"]])))
    router = _new_catboost(iterations=iterations)
    router.fit(_cat_frame(X), y, cat_features=list(CATEGORICAL), verbose=False)
    return router, {"classes": list(LABELS), "checkpoint_ids": ids15, "families": families,
                    "iterations": iterations, "fit_method": "4-fold prompt-group OOF features; router fit on all OOF records",
                    "feature_columns": list(FEATURE_COLUMNS), "categorical_features": list(CATEGORICAL),
                    "reliability_statistics": "computed in each OOF fitting partition; Laplace alpha 1.0",
                    "oof_record_count": int(len(X)), "oof_positive_fraction": float(y.mean()),
                    "oof_folds": fit_folds}


def prepare():
    if KERNEL_DIR.exists(): shutil.rmtree(KERNEL_DIR)
    if SUPPLEMENT.exists(): shutil.rmtree(SUPPLEMENT)
    SUPPLEMENT.mkdir(parents=True)
    KERNEL_DIR.mkdir(parents=True)
    v0 = catalog.cascades._v0(ROOT)
    _, members, probabilities, excluded, snapshot = catalog._load_inputs(ROOT, v0)
    ids = _sorted_ids(probabilities)
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes
    hard = np.stack([probabilities[sid].argmax(1) for sid in ids])
    families_all = [_model_family(members[sid]["model"]) for sid in ids]
    roster, _ = select_roster(hard, truth, families_all, ids, 15)
    selected_ids = [ids[i] for i in roster]
    router, deployment = _deployment(v0, ids, members, probabilities, roster)
    router_path = SUPPLEMENT / "e13_b_router.cbm"
    router.save_model(str(router_path))

    # Match existing payloads by content hash and verify against Kaggle's live inventory.
    inventory, by_hash = _remote_assets()
    mounted = {ref: items for ref, items in inventory.items()}
    lc3_manifest = json.loads((ROOT / "artifacts/kaggle/lc3-lc4-checkpoint-shard-v035/lc3_lc4_manifest.json").read_text())
    lc3_candidates = lc3_manifest["candidates"]
    lc3_assets = {}
    for item in lc3_candidates.values():
        for asset in item.get("files", []):
            digest = asset.get("sha256")
            if digest:
                lc3_assets.setdefault(digest, asset)

    missing = []
    trust = {row["sequence_id"]: row for row in pd.read_csv(ROOT / "docs/trust-list-2-v037.csv").to_dict("records")}
    for sid in selected_ids:
        exp = members[sid]["experiment_id"]
        model = build_checkpoint(exp)
        manifest = json.loads((ROOT / "artifacts/predictions" / f"{exp}_manifest.json").read_text())
        source_path = Path(manifest.get("best_checkpoint_path") or manifest.get("checkpoint_path"))
        source_path = source_path if source_path.is_absolute() else ROOT / source_path
        if source_path.is_dir() and (source_path / "best_checkpoint").is_dir(): source_path /= "best_checkpoint"
        elif source_path.is_dir() and manifest.get("selected_epoch"):
            ep=source_path / f"epoch-{int(manifest['selected_epoch']):02d}"
            if ep.is_dir(): source_path=ep
        source_name = source_path.name
        candidate = {"candidate_id": sid, "experiment_id": exp, "model": members[sid]["model"],
                     "family": _model_family(members[sid]["model"]),
                     "config": manifest.get("config", {}),
                     "model_revision": manifest.get("model_revision"),
                     "calibration_temperature": float(manifest.get("calibration_temperature") or 1.0),
                     "checkpoint_relative_path": ((Path("checkpoints") / sid / source_name)
                         if source_path.is_file() else (Path("checkpoints") / sid)).as_posix(),
                     "checkpoint_type": "file" if source_path.is_file() else "directory", "files": []}
        for src, relative in model["files"]:
            digest, size = sha256_file(src), src.stat().st_size
            existing = lc3_assets.get(digest)
            ref = None
            if existing and existing.get("flat_filename"):
                remote_name = existing["flat_filename"]
                if any(Path(rn).name == Path(remote_name).name and rs == size for rn, rs in mounted[LC3_REF].items()):
                    ref = {"dataset_ref": LC3_REF, "flat_filename": remote_name, "sha256": digest, "size": size}
            if ref is None:
                # Check other published shards before packaging a new copy.
                ref = _asset_ref(digest, size, by_hash, mounted)
            if ref is None:
                flat = f"assets/asset__{digest}"
                target = SUPPLEMENT / flat
                target.parent.mkdir(parents=True, exist_ok=True)
                try: os.link(src, target)
                except OSError: shutil.copy2(src, target)
                ref = {"dataset_ref": SUPPLEMENT_REF, "flat_filename": flat, "sha256": digest, "size": size}
                missing.append({"path": str(src), **ref})
            candidate["files"].append({"relative_path": Path(relative).as_posix(), **ref})
        deployment.setdefault("_member_specs", []).append(candidate)

    # Ensure the three expected files absent from the current 4-dataset inventory are now staged.
    staged_hashes={item["sha256"] for item in missing}
    bundle_members=deployment.pop("_member_specs")
    deployment["classes"] = list(LABELS)
    deployment["reliability_statistics"] = "Run-time target-dependent features are computed from all V0 labels with Laplace alpha 1.0."
    # Full V0 smoothed statistics for test feature construction.
    from src.e13_reliability_router import reliability_statistics
    p15=np.stack([probabilities[sid] for sid in selected_ids])
    stats=reliability_statistics(p15.argmax(2),truth)
    deployment["reliability_statistics"] = {k: v.tolist() for k,v in stats.items()}
    bundle={"schema_version":1,"ensemble":"E13-B","routing_method":"family_max",
            "max_checkpoints":15,"trust_list":snapshot,"classes":list(LABELS),
            "members":bundle_members,"deployment":deployment,
            "mounted_checkpoint_datasets":{"lc3":LC3_REF,"supplement":SUPPLEMENT_REF},
            "source_analysis":str(RUN_DIR.relative_to(ROOT)),
            "source_analysis_sha256":sha256_file(RUN_DIR/"results.json"),
            "missing_checkpoint_count":len({x['sha256'] for x in missing}),
            "missing_checkpoints":[m['candidate_id'] for m in bundle_members if any(f['dataset_ref']==SUPPLEMENT_REF for f in m['files'])]}
    (SUPPLEMENT/"e13_b_manifest.json").write_text(json.dumps(bundle,indent=2,sort_keys=True)+"\n")
    (SUPPLEMENT/"e13_b_inference.py").write_text(_inference_helper())
    (SUPPLEMENT/"dataset-metadata.json").write_text(json.dumps({"title":"NLM E13-B Reliability Router Checkpoints",
        "id":SUPPLEMENT_REF,"licenses":[{"name":"other"}],"subtitle":"Missing checkpoint assets and E13-B router",
        "description":"E13-B family-max reliability router. Supplement shard contains only checkpoint payloads absent from the existing LC3/LC4 and published Kaggle checkpoint datasets, plus the deployment router and manifest."},indent=2)+"\n")
    notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        "# E13-B · Learned Checkpoint Reliability Routing\n\nRuns the frozen E13-B family-max deployment router over the competition test set. Twelve checkpoint payloads are reused from LC3/LC4 and three missing payloads are supplied by the compact E13-B shard."),
        nbf.v4.new_code_cell(NOTEBOOK_CODE)])
    notebook.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
    nbf.write(notebook,KERNEL_DIR/"e13_b.ipynb")
    sources={f["dataset_ref"] for m in bundle_members for f in m["files"]}
    metadata={"id":KERNEL_REF,"title":"E13-B Trust List 2 v037",
        "code_file":"e13_b.ipynb","language":"python","kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "keywords":["name-that-llm","E13-B","reliability-routing","offline-inference"],
        "dataset_sources":sorted(sources),"kernel_sources":[],"competition_sources":["name-that-llm"],
        "model_sources":["google/embeddinggemma-2/transformers/embeddinggemma-2/1",
                         "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1",
                         "qwen-lm/qwen-3/transformers/0.6b/1"],"machine_shape":"NvidiaTeslaT4"}
    (KERNEL_DIR/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    summary={"kernel_ref":KERNEL_REF,"supplement_ref":SUPPLEMENT_REF,"roster":selected_ids,
        "roster_experiments":[members[sid]["experiment_id"] for sid in selected_ids],
        "missing_checkpoints":bundle["missing_checkpoints"],"new_payload_files":len(staged_hashes),
        "new_payload_bytes":sum(x.stat().st_size for x in SUPPLEMENT.rglob("asset__*")),
        "router_iterations":deployment["iterations"],"oof_record_count":deployment["oof_record_count"],
        "dataset_sources":metadata["dataset_sources"],"kernel_dir":str(KERNEL_DIR),"supplement_dir":str(SUPPLEMENT)}
    (KERNEL_DIR/"prepare_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def _inference_helper():
    # Standalone mirror of src.e13_reliability_router.build_feature_frame and routing.
    return '''import numpy as np, pandas as pd\nFEATURES=("checkpoint_id","model_family","predicted_label","competing_label","top1_probability","top2_probability","margin","normalized_entropy","label_precision","label_recall","pred_to_competitor_rate","competitor_to_pred_rate","agreement_count","family_agreement_count","ensemble_pred_probability","ensemble_competitor_probability","checkpoint_ensemble_pred_delta","checkpoint_ensemble_competitor_delta")\nCATS=("checkpoint_id","model_family","predicted_label","competing_label")\ndef make_features(probs,members,classes,stats):\n    p=np.asarray(probs,dtype=np.float64); ids=[m["candidate_id"] for m in members]; fam=[m["family"] for m in members]; m,n,k=p.shape\n    assert p.shape[0]==len(ids)==len(fam) and k==len(classes) and np.isfinite(p).all() and np.allclose(p.sum(2),1,atol=1e-4)\n    top=p.argmax(2); ordered=np.sort(p,axis=2); top1=ordered[:,:,-1]; top2=ordered[:,:,-2]; margin=top1-top2\n    entropy=-(np.clip(p,1e-12,1)*np.log(np.clip(p,1e-12,1))).sum(2)/np.log(k)\n    names=sorted(set(fam)); family_probs=np.mean([p[np.asarray(fam)==f].mean(0) for f in names],axis=0); family_probs/=family_probs.sum(1,keepdims=True)\n    rank=np.argsort(-family_probs,axis=1,kind="stable"); ep=family_probs.argmax(1)\n    competing=np.where(top==ep[None,:],rank[:,1][None,:],ep[None,:])\n    for j in range(m):\n        same=top[j]==ep\n        competing[j,same]=rank[same,1]\n    precision=np.asarray(stats["precision"],float); recall=np.asarray(stats["recall"],float); directional=np.asarray(stats["directional"],float)\n    row=np.arange(n); ownp=family_probs[row[None,:],top]; altp=family_probs[row[None,:],competing]\n    owncp=p[np.arange(m)[:,None],row[None,:],top]; altcp=p[np.arange(m)[:,None],row[None,:],competing]\n    agreements=(top[:,None,:]==top[None,:,:]).sum(1)-1; famagree=np.zeros((m,n),float)\n    for j in range(m):\n        famagree[j]=sum(np.any(top[np.asarray(fam)==f]==top[j],axis=0) for f in names if f!=fam[j])\n    frame=pd.DataFrame({"checkpoint_id":np.tile(np.asarray(ids,dtype=str),n),"model_family":np.tile(np.asarray(fam,dtype=str),n),"predicted_label":np.asarray(classes,dtype=object)[top].T.reshape(-1),"competing_label":np.asarray(classes,dtype=object)[competing].T.reshape(-1),"top1_probability":top1.T.reshape(-1),"top2_probability":top2.T.reshape(-1),"margin":margin.T.reshape(-1),"normalized_entropy":entropy.T.reshape(-1),"label_precision":precision[np.arange(m)[:,None],top].T.reshape(-1),"label_recall":recall[np.arange(m)[:,None],top].T.reshape(-1),"pred_to_competitor_rate":directional[np.arange(m)[:,None],top,competing].T.reshape(-1),"competitor_to_pred_rate":directional[np.arange(m)[:,None],competing,top].T.reshape(-1),"agreement_count":agreements.T.reshape(-1),"family_agreement_count":famagree.T.reshape(-1),"ensemble_pred_probability":ownp.T.reshape(-1),"ensemble_competitor_probability":altp.T.reshape(-1),"checkpoint_ensemble_pred_delta":(owncp-ownp).T.reshape(-1),"checkpoint_ensemble_competitor_delta":(altcp-altp).T.reshape(-1)})\n    for c in CATS: frame[c]=frame[c].astype(str)\n    return frame.loc[:,FEATURES]\ndef route_family_max(hard,reliability,probs,families):\n    hard=np.asarray(hard); reliability=np.asarray(reliability); probs=np.asarray(probs); m,n=hard.shape; k=probs.shape[1]; scores=np.full((n,k),-np.inf)\n    for r in range(n):\n        for label in range(k):\n            voters=np.flatnonzero(hard[:,r]==label)\n            if len(voters):\n                by={}\n                for i in voters: by[families[i]]=max(by.get(families[i],-np.inf),reliability[i,r])\n                scores[r,label]=np.mean(list(by.values()))\n    return np.asarray([min(np.flatnonzero(np.isclose(s,np.max(s),rtol=0,atol=1e-12)),key=lambda l:(-probs[r,l],l)) for r,s in enumerate(scores)],dtype=np.int64)\n'''


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
