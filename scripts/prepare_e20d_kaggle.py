"""Stage the E20-D Kaggle inference bundle using shared checkpoint assets.

This builder points at assets already published in the shared checkpoint
dataset and other immutable shards. The original pretrained Qwen3.5-2B is
mounted from its existing Kaggle Transformers model source.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import joblib
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.publish_kaggle_checkpoint_store import (
    ROOT, build_checkpoint, kaggle_file_inventory, sha256_file,
)
from src import e16_common, e20, ensemble_catalog
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

SHARD_REF = "zodrak/e20d-checkpoints-v001"
KERNEL_REF = "zodrak/e20-d-group-aware-classifier"
SHARD_DIR = ROOT / "artifacts/kaggle/e20d-checkpoints-v001"
KERNEL_DIR = ROOT / "artifacts/kaggle/e20-d-kernel"
REMOTE_META = ROOT / "artifacts/kaggle/e20d-remote-catalogs"
QWEN_MODEL_SOURCE = "nguyenminhtric/qwen3-5-2b/Transformers/default/1"

DATASETS = {
    "zodrak/zod006-models": ("zod006-models", "zod006_manifest.json"),
    "zodrak/lc3-lc4-checkpoints-v035": ("lc3-lc4-checkpoints-v035", "lc3_lc4_manifest.json"),
    "zodrak/three-strategy-checkpoints": ("three-strategy-checkpoints", "three_strategy_supplement_manifest.json"),
    "zodrak/ensemble-hypotheses-nov15-extra-checkpoints": ("ensemble-hypotheses-nov15-extra-checkpoints", "ensemble_hypotheses_nov15_extras_manifest.json"),
    "zodrak/trust2-v013-top5-models": ("trust2-v013-top5-models", "trust2_v013_top5_manifest.json"),
    "zodrak/e13-b-checkpoints-v001": ("e13-b-checkpoints-v001", "e13_b_manifest.json"),
}
BASE_DATASETS = [
    "pingjiale/qwen3-0-6b",
    "johnnyhyland/smollm3-3b-base-36l-text",
    "zodrak/zod03-llama-base",
]
MODEL_SOURCES = [
    QWEN_MODEL_SOURCE,
    "google/embeddinggemma-2/Transformers/embeddinggemma-2/1",
    "google/gemma-4/Transformers/gemma-4-e2b/1",
    "mistral-ai/ministral-3/Transformers/ministral-3-3b-base-2512/1",
]


def _download_metadata():
    REMOTE_META.mkdir(parents=True, exist_ok=True)
    for ref, (folder, metadata_file) in DATASETS.items():
        destination = REMOTE_META / folder
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / metadata_file
        if not target.is_file():
            subprocess.run(["kaggle", "datasets", "download", ref, "-f", metadata_file,
                            "-p", str(destination), "--unzip"], check=True)
        catalog = destination / "checkpoint_catalog.json"
        # The two content-addressed stores expose the exact file-hash catalog.
        if ref in {"zodrak/zod006-models", "zodrak/lc3-lc4-checkpoints-v035"} and not catalog.exists():
            subprocess.run(["kaggle", "datasets", "download", ref, "-f", catalog.name,
                            "-p", str(destination), "--unzip"], check=True)


def _remote_assets():
    _download_metadata()
    inventory = {ref: kaggle_file_inventory(ref) for ref in DATASETS}
    by_hash: dict[str, list[dict]] = {}
    for ref, (folder, _) in DATASETS.items():
        for path in (REMOTE_META / folder).glob("*.json"):
            try:
                document = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue

            def walk(value, default_ref=ref):
                if isinstance(value, dict):
                    asset_ref = value.get("dataset_ref", default_ref)
                    filename = (value.get("flat_filename") or value.get("kaggle_filename")
                                or value.get("filename"))
                    digest = value.get("sha256")
                    size = value.get("size", value.get("size_bytes"))
                    if filename and digest and size:
                        by_hash.setdefault(str(digest), []).append({
                            "dataset_ref": asset_ref, "flat_filename": str(filename),
                            "sha256": str(digest), "size": int(size)})
                    for child in value.values():
                        walk(child, asset_ref)
                elif isinstance(value, list):
                    for child in value:
                        walk(child, default_ref)
            walk(document)
    # Immutable shards use asset__<sha256> filenames. Verify their live size.
    for ref, files in inventory.items():
        for filename, size in files.items():
            base = Path(filename).name
            if base.startswith("asset__") and len(base) == len("asset__") + 64:
                digest = base[len("asset__"):]
                by_hash.setdefault(digest, []).append({"dataset_ref": ref,
                    "flat_filename": filename, "sha256": digest, "size": size})
    return inventory, by_hash


def _asset_reference(digest, size, inventory, by_hash):
    candidates = []
    for item in by_hash.get(digest, []):
        ref, filename = item["dataset_ref"], item["flat_filename"]
        if ref in inventory and int(item["size"]) == size and any(
                Path(name).name == Path(filename).name and int(remote_size) == size
                for name, remote_size in inventory[ref].items()):
            candidates.append(item)
    if not candidates:
        return None
    order = {ref: i for i, ref in enumerate(DATASETS)}
    return min(candidates, key=lambda x: (order.get(x["dataset_ref"], 100), x["flat_filename"]))


def _copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare():
    for path in (SHARD_DIR, KERNEL_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    (SHARD_DIR / "assets").mkdir()
    (SHARD_DIR / "deployment").mkdir()
    (KERNEL_DIR / "submission.ipynb").parent.mkdir(parents=True, exist_ok=True)

    ctx = e16_common.load_context(ROOT)
    outer_rosters = {int(x["fold"]): list(x["roster"]) for x in ctx["outer_folds"]}
    union = sorted(set().union(*map(set, outer_rosters.values())), key=lambda x: int(x[1:]))
    _, model_members, _, _, _ = ensemble_catalog._load_inputs(ROOT, ctx["v0"])
    inventory, by_hash = _remote_assets()
    checkpoint_entries = {}
    new_assets = {}
    reused_count = 0
    missing_count = 0

    for sequence_id in union:
        entry = model_members[sequence_id]
        experiment_id = entry["experiment_id"]
        checkpoint = build_checkpoint(experiment_id)
        prediction_manifest = json.loads((ROOT / "artifacts/predictions" /
                                          f"{experiment_id}_manifest.json").read_text())
        spec = {
            "candidate_id": sequence_id,
            "experiment_id": experiment_id,
            "model": entry["model"],
            "family": _model_family(entry["model"]),
            "config": prediction_manifest.get("config", {}),
            "model_revision": prediction_manifest.get("model_revision"),
            "calibration_temperature": float(prediction_manifest.get("calibration_temperature") or 1.0),
            "checkpoint": f"checkpoints/{sequence_id}",
            "files": [],
        }
        for source, relative in checkpoint["files"]:
            source = Path(source)
            digest, size = sha256_file(source), source.stat().st_size
            reference = _asset_reference(digest, size, inventory, by_hash)
            if reference is None:
                flat = f"assets/asset__{digest}"
                _copy_file(source, SHARD_DIR / flat)
                reference = {"dataset_ref": SHARD_REF, "flat_filename": flat,
                             "sha256": digest, "size": size}
                new_assets[digest] = {"kaggle_filename": flat, "size": size,
                                      "dataset_ref": SHARD_REF}
                missing_count += 1
            else:
                reused_count += 1
            spec["files"].append({"relative_path": str(relative), **reference})
        checkpoint_entries[sequence_id] = spec

    # Save the V0 probability cache required to apply the exact fold-trained
    # E13 reliability routers to test rows. The labels are used only to compute
    # per-checkpoint reliability statistics on each fold's training rows.
    union_indexes = [ctx["checkpoint_ids"].index(sequence_id) for sequence_id in union]
    v0_probability_cache = SHARD_DIR / "deployment" / "e20d_v0_probabilities.npz"
    np.savez_compressed(v0_probability_cache,
        row_ids=ctx["row_ids"].astype(str),
        texts=ctx["texts"].astype(str),
        y=np.asarray(ctx["y"], dtype=np.int64),
        checkpoint_ids=np.asarray(union, dtype=str),
        families=np.asarray([_model_family(model_members[x]["model"]) for x in union], dtype=str),
        probabilities=np.asarray(ctx["matrices"][union_indexes], dtype=np.float32))

    fold_deployment = []
    for outer in ctx["outer_folds"]:
        fold = int(outer["fold"])
        fit = np.asarray(outer["fit_rows"], dtype=np.int64)
        roster = list(outer_rosters[fold])
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1,
            sublinear_tf=True, dtype=np.float32, max_features=150000)
        vectorizer.fit(ctx["texts"][fit])
        vectorizer_path = SHARD_DIR / "deployment" / f"e20_group_vectorizer_fold{fold}.joblib"
        joblib.dump(vectorizer, vectorizer_path, compress=3)
        router_path = ROOT / "artifacts/kaggle/e20d-prep-v001" / f"e13_router_fold{fold}.cbm"
        catboost_path = ROOT / "artifacts/analysis/e20" / f"E20C_fold{fold}.cbm"
        if not router_path.is_file() or not catboost_path.is_file():
            raise FileNotFoundError(f"E20-D fold {fold} needs both saved E13 and E20-C models")
        for label, source in (("e13_router", router_path), ("e20c_classifier", catboost_path)):
            target = SHARD_DIR / "deployment" / f"{label}_fold{fold}.cbm"
            _copy_file(source, target)
        fold_deployment.append({"fold": fold, "fit_rows": fit.tolist(),
            "checkpoint_ids": roster,
            "families": [_model_family(model_members[x]["model"]) for x in roster],
            "router_file": f"deployment/e13_router_fold{fold}.cbm",
            "classifier_file": f"deployment/e20c_classifier_fold{fold}.cbm",
            "vectorizer_file": f"deployment/e20_group_vectorizer_fold{fold}.joblib"})

    e19b_root = ROOT / "artifacts/analysis/e19b"
    grouper_path = SHARD_DIR / "deployment" / "e19b_grouper.joblib"
    _copy_file(e19b_root / "e19b_grouper.joblib", grouper_path)
    shutil.copytree(ROOT / "src", SHARD_DIR / "runtime" / "src")
    shutil.copytree(ROOT / "configs", SHARD_DIR / "runtime" / "configs")

    profile = json.loads((ROOT / "artifacts/analysis/e20/e20d_config.json").read_text())
    manifest = {
        "schema_version": 1,
        "experiment": "E20-D",
        "classes": list(LABELS),
        "checkpoint_ids": union,
        "checkpoint_members": checkpoint_entries,
        "folds": fold_deployment,
        "v0_cache": "deployment/e20d_v0_probabilities.npz",
        "e19b_grouper": "deployment/e19b_grouper.joblib",
        "qwen_reconstruction_base": QWEN_MODEL_SOURCE,
        "qwen_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "embedding_revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "e20d_config": profile,
        "asset_reuse": {"checkpoint_files_reused": reused_count,
                         "checkpoint_files_added": missing_count,
                         "new_checkpoint_asset_bytes": sum(x["size"] for x in new_assets.values())},
    }
    (SHARD_DIR / "e20d_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (SHARD_DIR / "checkpoint_catalog.json").write_text(json.dumps({
        "schema_version": 1, "dataset_ref": SHARD_REF, "dataset_version": 6,
        "assets": new_assets,
        "checkpoints": {spec["experiment_id"]: {
            "experiment_id": spec["experiment_id"], "sequence_id": sid,
            "model": spec["model"], "family": spec["family"], "config": spec["config"],
            "files": spec["files"]} for sid, spec in checkpoint_entries.items()},
    }, indent=2, sort_keys=True) + "\n")
    (SHARD_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "NLM E20-D Inference Assets",
        "id": SHARD_REF,
        "licenses": [{"name": "other"}],
        "subtitle": "E20-D fold classifiers and compact inference runtime",
        "description": "E20-D deployment artifacts and runtime. Checkpoint payloads are mounted from the shared ZOD006 dataset or earlier immutable shards; original Qwen3.5-2B and other base models are attached as Kaggle model sources.",
    }, indent=2) + "\n")

    sources = sorted({SHARD_REF} | {item["dataset_ref"] for spec in checkpoint_entries.values()
        for item in spec["files"]} | set(BASE_DATASETS))
    kernel_metadata = {
        "id": KERNEL_REF, "title": "E20-D Group-Aware Classifier",
        "code_file": "e20-d-group-aware-classifier.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [],
        "dataset_sources": sources, "kernel_sources": [],
        "competition_sources": ["name-that-llm"], "model_sources": MODEL_SOURCES,
        "machine_shape": "NvidiaTeslaT4",
    }
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# E20-D — Oracle-Trained Classifier with Reconstructed Groups\n\nApplies the five saved E20-C fold classifiers to test-time E19-B reconstructed groups. It reuses the exact fold checkpoint rosters and E13 router models, averages fold score matrices, and applies Hungarian assignment within predicted groups. No classifier is fitted in this kernel."),
        nbf.v4.new_code_cell(NOTEBOOK_CODE),
    ])
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nbf.write(notebook, KERNEL_DIR / "e20-d-group-aware-classifier.ipynb")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n")
    summary = {"kernel_ref": KERNEL_REF, "shard_ref": SHARD_REF, "checkpoint_count": len(union),
        "checkpoint_files_reused": reused_count, "checkpoint_files_added": missing_count,
        "new_checkpoint_asset_bytes": sum(x["size"] for x in new_assets.values()),
        "qwen_base_bytes": 0,
        "qwen_model_source": QWEN_MODEL_SOURCE,
        "checkpoint_ids": union, "dataset_sources": sources, "model_sources": MODEL_SOURCES,
        "shard_dir": str(SHARD_DIR), "kernel_dir": str(KERNEL_DIR)}
    (KERNEL_DIR / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


NOTEBOOK_CODE = r'''from pathlib import Path
import gc, hashlib, json, os, shutil, subprocess, sys, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
import joblib, numpy as np, pandas as pd, torch
from catboost import CatBoostClassifier

INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working/e20d'); WORK.mkdir(parents=True,exist_ok=True)
def only(values,desc):
    values=list(dict.fromkeys(Path(x) for x in values)); assert len(values)==1, f'{desc}: {values[:6]}'; return values[0]
def root_for(marker): return only([p.parent for p in INPUT.rglob(marker) if p.is_file()],marker)
BROOT=root_for('e20d_manifest.json'); B=json.loads((BROOT/'e20d_manifest.json').read_text())
assert B['experiment']=='E20-D' and len(B['classes'])==18
LC3=root_for('lc3_lc4_manifest.json')
DATASET_MARKERS={
 'zodrak/zod006-models':'zod006_manifest.json',
 'zodrak/lc3-lc4-checkpoints-v035':'lc3_lc4_manifest.json',
 'zodrak/three-strategy-checkpoints':'three_strategy_supplement_manifest.json',
 'zodrak/ensemble-hypotheses-nov15-extra-checkpoints':'ensemble_hypotheses_nov15_extras_manifest.json',
 'zodrak/trust2-v013-top5-models':'trust2_v013_top5_manifest.json',
 'zodrak/e13-b-checkpoints-v001':'e13_b_manifest.json',
 'zodrak/e20d-checkpoints-v001':'e20d_manifest.json',
 'pingjiale/qwen3-0-6b':'config.json',
 'johnnyhyland/smollm3-3b-base-36l-text':'config.json',
 'zodrak/zod03-llama-base':'config.json'}
ROOTS={}
def dataset_root(ref):
    if ref not in ROOTS:
        marker=DATASET_MARKERS[ref]; ROOTS[ref]=root_for(marker)
    return ROOTS[ref]
def resolve_asset(item):
    # The E20-D-only checkpoint delta was merged into the shared ZOD006
    # dataset before inference was prepared. Prefer that updated model store;
    # retain the original E20-D shard as a hash-checked fallback.
    candidates=[]; name=Path(item['flat_filename']).name
    if item['dataset_ref']=='zodrak/e20d-checkpoints-v001':
        shared=dataset_root('zodrak/zod006-models')/f"asset__{item['sha256']}"
        if shared.is_file(): candidates.append(shared)
    if not candidates:
        root=dataset_root(item['dataset_ref']); name=Path(item['flat_filename']).name
        candidates.extend(p for p in root.rglob(name) if p.is_file())
    path=only(candidates,f"{item['dataset_ref']}/{name}")
    assert path.stat().st_size==int(item['size']), f'size mismatch: {path}'
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    assert h.hexdigest()==item['sha256'], f'hash mismatch: {path}'
    return path

RUNTIME=BROOT/'runtime'; sys.path.insert(0,str(RUNTIME))
# Kaggle's preinstalled TorchAO 0.10 is incompatible with the PEFT version
# used by the inference runtime. Remove it before importing PEFT; LoRA does
# not use TorchAO kernels here.
from importlib import metadata as importlib_metadata
from packaging.version import Version
try:
    torchao_version=Version(importlib_metadata.version('torchao'))
except importlib_metadata.PackageNotFoundError:
    torchao_version=None
if torchao_version is not None and torchao_version < Version('0.16.0'):
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
from src.inference import predict_candidate
from src import e19b_group, e20, e15
from src.e13_reliability_router import build_feature_frame, _score_correctness, _cat_frame
from src.e16_common import finite_scores
from src.e19b_reconstruct import _generation_inputs, parse_reconstruction, MODEL_REVISION, MAX_NEW_TOKENS

def model_root(marker):
    hits=[p.parent for p in INPUT.rglob('config.json') if marker.casefold() in str(p.parent).casefold()]
    if not hits: raise FileNotFoundError(f'base model not mounted: {marker}')
    depth=min(len(p.parts) for p in hits); return only([p for p in hits if len(p.parts)==depth],marker)

torch.backends.cuda.matmul.allow_tf32=True
assert torch.cuda.is_available(), 'E20-D requires a Kaggle GPU'
device='cuda'; classes=B['classes']; qwen_dir=model_root('nguyenminhtric/qwen3-5-2b')
assert (qwen_dir/'config.json').is_file()
assert any(p.is_file() and p.stat().st_size==4548221488 for p in qwen_dir.glob('model.safetensors-*'))
try:
    from transformers import AutoModelForMultimodalLM, AutoModel, AutoTokenizer
except ImportError as exc:
    raise RuntimeError(f"Kaggle Transformers runtime lacks Qwen3.5 multimodal support: {exc}")
class TextOnlyQwenProcessor:
    """Use Qwen's tokenizer template for text-only reconstruction prompts."""
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def apply_chat_template(self, conversations, **kwargs):
        processor_kwargs=kwargs.pop('processor_kwargs',{}) or {}
        for key,value in processor_kwargs.items(): kwargs.setdefault(key,value)
        return self.tokenizer.apply_chat_template(conversations,**kwargs)
    def batch_decode(self,*args,**kwargs): return self.tokenizer.batch_decode(*args,**kwargs)
processor=TextOnlyQwenProcessor(AutoTokenizer.from_pretrained(qwen_dir,local_files_only=True))
qwen=AutoModelForMultimodalLM.from_pretrained(qwen_dir,device_map='auto',dtype='auto',local_files_only=True).eval()
test_paths=[p for p in (INPUT/'competitions/name-that-llm/test.csv',INPUT/'name-that-llm/test.csv') if p.is_file()]
TEST=only(test_paths,'competition test.csv')
test=pd.read_csv(TEST,dtype={'id':'string'}); assert {'id','text'}<=set(test.columns)
assert test.id.notna().all() and test.id.is_unique and test.text.notna().all()
ids=test.id.astype(str).to_numpy(); texts=test.text.fillna('').astype(str).to_numpy(); n=len(ids)
def reconstruct_one(text):
    inputs=_generation_inputs(processor,[str(text)])
    target=next(qwen.parameters()).device
    inputs={k:(v.to(target) if hasattr(v,'to') else v) for k,v in inputs.items()}
    with torch.inference_mode():
        generated=qwen.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=512,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
    prefix=inputs['input_ids'].shape[1]
    raw=processor.batch_decode(generated[:,prefix:],skip_special_tokens=True)[0]
    return parse_reconstruction(raw)
recon=[]
for i,text in enumerate(texts):
    recon.append(reconstruct_one(text)); print(f'reconstructed {i+1}/{n}',flush=True)
del qwen,processor; gc.collect(); torch.cuda.empty_cache()

# Encode reconstructed prompts with the same pinned EmbeddingGemma revision.
embedding_root=model_root('embeddinggemma-2')
tokenizer=AutoTokenizer.from_pretrained(embedding_root,local_files_only=True,padding_side='right')
encoder=AutoModel.from_pretrained(embedding_root,local_files_only=True,dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32).to(device).eval()
emb=[]
for start in range(0,n,8):
    batch=['task: clustering | query: '+str(x.get('reconstructed_prompt','')) for x in recon[start:start+8]]
    inputs=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
    with torch.inference_mode():
        hidden=encoder(**inputs).last_hidden_state.float(); mask=inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1); pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
    emb.append(pooled.cpu().numpy().astype(np.float32))
emb=np.concatenate(emb,axis=0)
del encoder,tokenizer; gc.collect(); torch.cuda.empty_cache()

grouper=joblib.load(BROOT/B['e19b_grouper'])
reconstructed=np.asarray([x.get('reconstructed_prompt','') for x in recon],dtype=str)
topics=np.asarray([x.get('topic','') for x in recon],dtype=str)
details=[x.get('distinctive_details',[]) for x in recon]; constraints=[x.get('constraints',[]) for x in recon]
spaces=e19b_group.transform_similarity_spaces(reconstructed,texts,grouper['vectorizers'])
row_ix=np.arange(n); pairs=e19b_group.candidate_pairs(row_ix,spaces,emb,neighbors=grouper.get('neighbors',40))
pair_features=e19b_group.pair_features(row_ix,pairs,spaces,reconstructed,texts,emb,topics,details,constraints)
pair_prob=e19b_group.score_pairs(grouper['pair_scorer'],pair_features) if len(pairs) else np.empty(0,dtype=np.float32)
groups,_=e19b_group.cluster_complete_link(ids,pairs,pair_prob,threshold=float(grouper['threshold']),max_cluster_size=int(grouper.get('max_cluster_size',18)))
pair_scores={tuple(sorted((ids[int(a)],ids[int(b)]))):float(p) for (a,b),p in zip(pairs,pair_prob)}
print('E19-B reconstructed groups:',len(set(groups.tolist())),'of',n,'test rows')

# Resolve and run each unique checkpoint once, then reuse its probabilities in
# the five fold-specific E13 evidence builders and E20-C classifiers.
base_models={}
base_models['embeddinggemma2']=embedding_root
base_models['gemma4_e2b']=model_root('gemma-4-e2b')
base_models['ministral3_3b']=model_root('ministral-3-3b-base-2512')
granite=json.loads((LC3/'lc3_lc4_granite_base_manifest.json').read_text())
for model in ('granite4_h_micro','granite4_350m'):
    files=granite['models'].get(model,[]); assert files,f'missing LC3 base {model}'
    dest=WORK/'bases'/model; dest.mkdir(parents=True,exist_ok=True)
    for item in files:
        src=resolve_asset(item); target=dest/item['relative_path']; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.symlink_to(src)
    base_models[model]=dest
base_models['qwen06']=model_root('qwen3-0-6b')
# ZOD03 publishes the pinned Llama base as flat files. Rebuild its normal HF
# directory layout from the source manifest instead of searching for a
# config.json that is not present at the dataset root.
llama_manifest_path=root_for('llama_base_manifest.json')/'llama_base_manifest.json'
llama_manifest=json.loads(llama_manifest_path.read_text())
llama_dir=WORK/'bases/llama32_3b'; llama_dir.mkdir(parents=True,exist_ok=True)
for item in llama_manifest['files']:
    src=llama_manifest_path.parent/item['flat_filename']
    if not src.is_file(): raise FileNotFoundError(f"Llama base asset missing: {src}")
    dst=llama_dir/item['relative_path']; dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists(): dst.symlink_to(src)
base_models['llama32_3b']=llama_dir
base_models['smollm3_3b']=model_root('smollm3-3b-base')
checkpoint_root=WORK/'checkpoints'; checkpoint_root.mkdir(parents=True,exist_ok=True)
probabilities={}; families={}
for num,sid in enumerate(B['checkpoint_ids'],1):
    spec=dict(B['checkpoint_members'][sid]); target=checkpoint_root/sid; target.mkdir(parents=True,exist_ok=True)
    for item in spec['files']:
        src=resolve_asset(item); dst=target/item['relative_path']; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): dst.symlink_to(src)
    spec['checkpoint']=str(checkpoint_root/sid)
    if spec['model']=='svc':
        checkpoint_file=only([p for p in (checkpoint_root/sid).rglob('*.joblib') if p.is_file()],f'{sid} SVC checkpoint')
        spec['checkpoint']=str(checkpoint_file)
    if spec['model'] in base_models: spec['base_model']=str(base_models[spec['model']])
    probabilities[sid]=predict_candidate(spec,texts,classes,WORK)
    families[sid]=spec['family']
    gc.collect(); torch.cuda.empty_cache()
    if num%4==0 or num==len(B['checkpoint_ids']): print(f'inferred checkpoint {num}/{len(B["checkpoint_ids"])}',flush=True)

cache=np.load(BROOT/B['v0_cache'],allow_pickle=False)
v0_ids=np.asarray(cache['row_ids'],dtype=str); v0_texts=np.asarray(cache['texts'],dtype=str); y=np.asarray(cache['y'],dtype=np.int64)
cache_ids=np.asarray(cache['checkpoint_ids'],dtype=str); v0_prob=np.asarray(cache['probabilities'],dtype=np.float32)
assert v0_prob.shape==(len(cache_ids),len(v0_ids),len(classes))
all_ids=np.concatenate([v0_ids,ids]); all_texts=np.concatenate([v0_texts,texts])
cp_order=list(B['checkpoint_ids']); cp_pos={sid:i for i,sid in enumerate(cp_order)}
all_probs=np.zeros((len(cp_order),len(all_ids),len(classes)),dtype=np.float32)
v0_pos={sid:i for i,sid in enumerate(cache_ids)}
for sid in cp_order:
    all_probs[cp_pos[sid],:len(v0_ids)]=v0_prob[v0_pos[sid]]
    all_probs[cp_pos[sid],len(v0_ids):]=probabilities[sid]
ctx={'matrices':all_probs,'checkpoint_ids':cp_order,'families':[families.get(sid, next(x['family'] for x in B['checkpoint_members'].values() if x['candidate_id']==sid)) for sid in cp_order],
     'row_ids':all_ids,'texts':all_texts}
fold_scores=[]
for fold in B['folds']:
    f=int(fold['fold']); fit=np.asarray(fold['fit_rows'],dtype=np.int64); roster=list(fold['checkpoint_ids'])
    p=all_probs[[cp_pos[sid] for sid in roster]]
    router=CatBoostClassifier(); router.load_model(str(BROOT/fold['router_file']))
    frame,_=build_feature_frame(p,roster,fold['families'],y[fit],fit,np.arange(len(v0_ids),len(all_ids)))
    reliability=_score_correctness(router,_cat_frame(frame)).reshape(n,len(roster)).T
    hard=p[:,len(v0_ids):].argmax(2)
    ensemble=e15.e13.family_balanced_probabilities(p[:,len(v0_ids):],fold['families'])
    e13_scores=finite_scores(e15._pair_scores(hard,reliability,ensemble,fold['families']))
    evidence={'scores':e13_scores,'roster_ids':roster,'reliability':reliability,'hard':hard}
    vectorizer=joblib.load(BROOT/fold['vectorizer_file'])
    feature=e20._combine_features(ctx,evidence,np.arange(len(v0_ids),len(all_ids)),roster,groups,
        vectorizer,pair_scores=pair_scores,oracle=False,fit_texts=v0_texts[fit])
    classifier=CatBoostClassifier(); classifier.load_model(str(BROOT/fold['classifier_file']))
    matrix=e20._score_with_reused_classifier(classifier,feature); fold_scores.append(matrix)
    print(f'E20-D fold {f}: scored {n} responses',flush=True)
scores=np.mean(np.stack(fold_scores),axis=0)
pred=e20.hungarian(scores,groups,ids)
submission=pd.DataFrame({'id':ids,'model':np.asarray(classes,dtype=object)[pred].astype(str)})
assert submission.id.tolist()==test.id.astype(str).tolist() and submission.id.is_unique
assert submission.model.notna().all() and submission.model.isin(classes).all()
output=Path('/kaggle/working/submission.parquet'); submission.to_parquet(output,index=False)
assert pd.read_parquet(output).equals(submission)
run={'experiment':'E20-D','rows':n,'checkpoint_ids':B['checkpoint_ids'],'folds':[x['fold'] for x in B['folds']],
     'groups':{str(i):str(g) for i,g in zip(ids,groups)},'test_sha256':hashlib.sha256(TEST.read_bytes()).hexdigest(),
     'submission_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'predictions':submission.to_dict('records')}
Path('/kaggle/working/e20d_run_manifest.json').write_text(json.dumps(run,indent=2)+'\n')
print(json.dumps(run,indent=2)); display(submission)
'''


def main():
    print(json.dumps(prepare(), indent=2))


if __name__ == "__main__":
    main()
