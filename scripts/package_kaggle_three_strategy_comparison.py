"""Build offline Kaggle assets and one-pass inference notebook for three Trust v025 strategies."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts/kaggle/three-strategy-checkpoints"
KERNEL = ROOT / "artifacts/kaggle/three-strategy-kernel"
TRUST_PATH = ROOT / "docs/trust-list-2-v025.csv"
SUPPLEMENT_IDS = ["N0003", "N0006", "N0022", "N0034", "N0054", "N0055", "N0057", "N0059", "N0064", "N0076", "N0080"]
SKIP_DIRS = {"best_checkpoint"}
SKIP_NAMES = {"README.md", "checkpoint_selection.json", "baseline_validation.json", "training_log.json", "training_progress.json", "validation_metrics.json"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_files(source: Path, sid: str) -> list[dict]:
    dest_dir = ASSETS / f"ckpt__{sid}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    items = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    copied = []
    for file in items:
        relative = Path(file.name) if source.is_file() else file.relative_to(source)
        if any(part in SKIP_DIRS or part.startswith("epoch-") for part in relative.parts) or file.name in SKIP_NAMES:
            continue
        flat = f"{dest_dir.name}__{str(relative).replace('/', '__')}"
        shutil.copy2(file, ASSETS / flat)
        copied.append({"relative_path": relative.as_posix(), "flat_filename": flat, "size_bytes": file.stat().st_size, "sha256": sha(file)})
    if not copied:
        raise FileNotFoundError(f"No inference files found for {sid}: {source}")
    return copied


def build_assets() -> Path:
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    trust = pd.read_csv(TRUST_PATH, dtype={"sequence_id": str})
    trust = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    candidates = {}
    for sid in SUPPLEMENT_IDS:
        row = trust.loc[sid]
        manifest_path = ROOT / "artifacts/predictions" / f"{row.experiment_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(str(row.checkpoint_path))
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Selected Trust v025 checkpoint is missing: {sid} {checkpoint}")
        files = copy_files(checkpoint, sid)
        config = manifest.get("config", {})
        candidates[sid] = {
            "sequence_id": sid, "experiment_id": str(row.experiment_id), "model": str(row.model),
            "checkpoint_type": "file" if checkpoint.is_file() else "directory", "files": files,
            "base_model": config.get("checkpoint"), "base_model_revision": manifest.get("model_revision"),
            "config": config, "classes": list(map(str, manifest["classes"])),
            "calibration_temperature": float(manifest.get("calibration_temperature") or 1.0),
            "training_dataset": str(row.training_dataset),
            "source_manifest_sha256": sha(manifest_path),
        }
    classes = next(iter(candidates.values()))["classes"]
    if any(c["classes"] != classes for c in candidates.values()):
        raise ValueError("Supplemental checkpoints do not use one canonical class order")
    bundle = {"name": "trust_list_2_v025_three_strategy_supplement", "trust_list_version": "trust-list-2-v025",
              "sequence_ids": SUPPLEMENT_IDS, "classes": classes, "candidates": candidates}
    (ASSETS / "three_strategy_supplement_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    source_dir = ROOT / "artifacts/kaggle/trust2-v013-top5-models"
    for name in ["source____init__.py", "source__data_factory.py", "source__feature_factory.py", "source__model_factory.py", "source__inference.py"]:
        shutil.copy2(source_dir / name, ASSETS / name)
    wheel = next(source_dir.glob("scikit_learn-1.9.1-*.whl"))
    shutil.copy2(wheel, ASSETS / wheel.name)
    (ASSETS / "dataset-metadata.json").write_text(json.dumps({
        "title": "Trust v025 Strategy Checkpoints",
        "id": "zodrak/three-strategy-checkpoints", "licenses": [{"name": "other"}],
        "subtitle": "Exact additional selected checkpoints for equal-selected and C1",
        "description": "Inference-only checkpoint files resolved from Trust List 2 v025 experiment IDs; no training data included. N0074 is supplied by the existing Trust v013 Equal9 package, whose manifest confirms the exact matching v025 experiment ID."}, indent=2) + "\n")
    return ASSETS


def build_kernel() -> Path:
    import nbformat as nbf
    KERNEL.mkdir(parents=True, exist_ok=True)
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": "zodrak/trust-v025-c3-equal-selected-c1", "title": "Trust v025 C3 Equal Selected C1",
        "code_file": "trust_v025_three_strategies.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["trust-list-2-v025", "offline-inference", "ensemble"],
        "dataset_sources": ["zodrak/trust2-v013-top5-models", "zodrak/ensemble-hypotheses-nov15-extra-checkpoints",
                            "zodrak/equal9-c3-n0105-checkpoint", "zodrak/three-strategy-checkpoints",
                            "zodrak/trust2-v013-equal9-models", "zodrak/zod03-models", "zodrak/zod03-llama-base",
                            "pingjiale/qwen3-0-6b", "johnnyhyland/smollm3-3b-base-36l-text"],
        "kernel_sources": [], "competition_sources": ["name-that-llm"],
        "model_sources": ["google/gemma-4/transformers/gemma-4-e2b/1",
                          "google/embeddinggemma-2/transformers/embeddinggemma-2/1",
                          "mistral-ai/ministral-3/transformers/ministral-3-3b-base-2512/1"],
        "machine_shape": "NvidiaTeslaT4"}, indent=2) + "\n")
    intro = "# Trust List 2 v025: C3, equal-selected, and C1\n\nOne offline pass scores the exact required checkpoint union, exports three independently verifiable parquet files, and uses C3 as the kernel's primary submission."
    setup = r'''from pathlib import Path
import gc, json, os, shutil, subprocess, sys, time
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
INPUT=Path('/kaggle/input')
def one(pattern):
    found=list(INPUT.rglob(pattern)); assert len(found)==1,(pattern,found); return found[0]
TOP=one('trust2_v013_top5_manifest.json').parent; EXTRA=one('ensemble_hypotheses_nov15_extras_manifest.json').parent
N0105=one('n0105_checkpoint_manifest.json').parent; SUP=one('three_strategy_supplement_manifest.json').parent
OLD=one('trust2_v013_equal9_manifest.json').parent
top=json.loads((TOP/'trust2_v013_top5_manifest.json').read_text()); extra=json.loads((EXTRA/'ensemble_hypotheses_nov15_extras_manifest.json').read_text())
n0105=json.loads((N0105/'n0105_checkpoint_manifest.json').read_text()); sup=json.loads((SUP/'three_strategy_supplement_manifest.json').read_text())
old=json.loads((OLD/'trust2_v013_equal9_manifest.json').read_text())
classes=top['classes']; assert len(classes)==18
C={**top['candidates'],**extra['candidates'],**sup['candidates']}; ROOTS={sid: (TOP if sid in top['candidates'] else EXTRA if sid in extra['candidates'] else SUP) for sid in C}
C['N0105']=n0105; ROOTS['N0105']=N0105
q=old['candidates']['N0074']; C['N0074']={**q,'files':q['checkpoint_files'],'calibration_temperature':q.get('calibration_temperature',1.0)}; ROOTS['N0074']=OLD
assert C['N0074']['experiment_id']=='qwen06_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgba46767d_code81bce4a'
EQUAL9=['N0071','N0005','N0058','N0072','N0030','N0061','N0044','N0009','N0048']
EQUAL_SELECTED=['N0003','N0005','N0006','N0009','N0022','N0030','N0034','N0044','N0048','N0055','N0056','N0057','N0058','N0061','N0071','N0072','N0076','N0080','N0081','N0105']
C3=[('N0071','gpt-6-astra'),('N0105','gemini-3.1-pro-preview'),('N0048','claude-opus-4-6')]
C1=[('N0056','gemini-3.1-pro-preview'),('N0105','gpt-6-astra'),('N0048','claude-opus-4-6'),('N0076','claude-opus-5'),('N0105','qwen3.8-max'),('N0034','claude-opus-4-8'),('N0081','gpt-5.6-luna'),('N0064','gemini-3.6-flash'),('N0061','claude-sonnet-5'),('N0022','glm-5.3-flash'),('N0055','gpt-5.5'),('N0057','gpt-5.6-sol'),('N0054','kimi-k3'),('N0074','gemini-3.7-flash'),('N0059','gpt-5.6-terra'),('N0030','gemini-3.8-flash'),('N0009','glm-5.3'),('N0003','claude-fable-5')]
needed=sorted(set(EQUAL9+EQUAL_SELECTED+[x for x,_ in C3]+[x for x,_ in C1])); assert len(needed)==24 and all(x in C for x in needed)
from importlib.metadata import version
if version('scikit-learn')!='1.9.1':
    wheel=next(INPUT.rglob('scikit_learn-1.9.1-*.whl'),None); assert wheel; subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--force-reinstall',str(wheel)],check=True)
import joblib,numpy as np,pandas as pd,torch
assert torch.cuda.is_available()
WORK=Path('/kaggle/working/three_strategy'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)
for name in ('__init__.py','data_factory.py','feature_factory.py','model_factory.py','inference.py'):
    SRC.joinpath(name).symlink_to(TOP/f'source__{name}')
sys.path.insert(0,str(WORK)); CP=WORK/'checkpoints'; CP.mkdir(parents=True,exist_ok=True)
for sid in needed:
    c=C[sid]; dest=CP/sid; dest.mkdir(parents=True,exist_ok=True); files=c.get('files',c.get('checkpoint_files',[])); assert files,(sid,'empty files')
    for f in files:
        src=ROOTS[sid]/f['flat_filename']; assert src.is_file(),src; to=dest/f['relative_path']; to.parent.mkdir(parents=True,exist_ok=True); to.symlink_to(src)
    c['runtime_checkpoint']=str(dest if c['checkpoint_type']=='directory' else dest/files[0]['relative_path'])
def model_root(marker):
    found=[p.parent for p in INPUT.rglob('config.json') if marker.lower() in str(p.parent).lower()]; assert found,marker
    depth=min(map(lambda p:len(p.parts),found)); roots=sorted({p for p in found if len(p.parts)==depth}); assert len(roots)==1,(marker,roots); return roots[0]
BASES={'embeddinggemma2':model_root('embeddinggemma-2'),'ministral3_3b':model_root('ministral-3-3b-base-2512'),
       'gemma4_e2b':model_root('gemma-4-e2b'),'qwen06':model_root('Qwen3-0.6B')}
# N0074's historical package contains the exact adapter but no tokenizer files.
qcp=Path(C['N0074']['runtime_checkpoint'])
for name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'):
    src=BASES['qwen06']/name; dst=qcp/name
    if src.exists() and not dst.exists(): dst.symlink_to(src)
# Recreate the SmolLM3 config/index fixes used by the existing Trust List inference notebook.
smol_source=model_root('smollm3-3b-base-36l-text'); smol=WORK/'bases/smollm3_3b'; smol.mkdir(parents=True,exist_ok=True)
for item in smol_source.iterdir():
    target=smol/item.name
    if target.exists(): continue
    if item.name=='config.json': shutil.copy2(item,target)
    else: target.symlink_to(item)
sc=smol/'config.json'; scj=json.loads(sc.read_text())
if scj.get('model_type')=='smollm3_text_base': scj['model_type']='smollm3'; sc.write_text(json.dumps(scj))
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
assert 'smollm3' in CONFIG_MAPPING
if not (smol/'model.safetensors.index.json').exists():
    from safetensors import safe_open
    wm={}
    for shard in sorted(smol.glob('*.safetensors')):
        with safe_open(str(shard),framework='pt',device='cpu') as sf: wm.update({key:shard.name for key in sf.keys()})
    assert wm; (smol/'model.safetensors.index.json').write_text(json.dumps({'metadata':{},'weight_map':wm}))
BASES['smollm3_3b']=smol
# Reconstitute exact, immutable Granite revisions from the established ZOD03 inference package.
zpath=one('zod03_manifest.json'); zd=zpath.parent; z=json.loads(zpath.read_text())
for model in ('granite4_350m','granite4_h_micro'):
    cand=next(c for c in C.values() if c['model']==model); base=next((v for v in z['candidates'].values() if v.get('base_model')==cand.get('base_model') and v.get('base_model_revision')==cand.get('base_model_revision') and v.get('base_files')),None)
    assert base,(model,cand.get('base_model_revision')); bd=WORK/'bases'/model; bd.mkdir(parents=True,exist_ok=True)
    for f in base['base_files']:
        src=zd/f['flat_filename']; assert src.is_file(),src; dst=bd/f['relative_path']; dst.parent.mkdir(parents=True,exist_ok=True); dst.symlink_to(src)
    BASES[model]=bd
# Exact Llama base package for N0057.
lm=one('llama_base_manifest.json'); ld=lm.parent; l=json.loads(lm.read_text()); bd=WORK/'bases/llama32_3b'; bd.mkdir(parents=True,exist_ok=True)
for f in l['files']:
    src=ld/f['flat_filename']; dst=bd/f['relative_path']; dst.parent.mkdir(parents=True,exist_ok=True); dst.symlink_to(src)
BASES['llama32_3b']=bd
try:
    import peft.tuners.lora.torchao as pt; pt.is_torchao_available=lambda:False
except ImportError: pass
from src.inference import _predict_neural,predict_catboost_stylometry,predict_svc_bundle
TEST=next((p for p in (INPUT/'competitions/name-that-llm/test.csv',INPUT/'name-that-llm/test.csv') if p.is_file()),None); assert TEST
test=pd.read_csv(TEST,dtype={'id':'string'}); assert {'id','text'}<=set(test.columns) and test.id.notna().all() and test.id.is_unique
texts=test.text.fillna('').astype(str).tolist()
def infer(sid):
    c=C[sid]; model=c['model']; cp=Path(c['runtime_checkpoint'])
    if model=='svc': probs,labs=predict_svc_bundle(joblib.load(cp),texts)
    elif model=='catboost_stylometry': probs,labs=predict_catboost_stylometry(cp,texts,classes,temperature=float(c.get('calibration_temperature') or 1.0))
    else:
        bs=min(4,max(1,int(c.get('config',{}).get('inference_batch_size') or 4)))
        while True:
            try:
                probs,labs=_predict_neural(model,cp,texts,classes,batch_size=bs,max_length=int(c.get('config',{}).get('max_length') or 384),
                  temperature=float(c.get('calibration_temperature') or 1.0),base_checkpoint=str(BASES[model]) if model in BASES else c.get('base_model'),
                  revision=None if model in BASES else c.get('base_model_revision'),fix_mistral_regex=bool(c.get('config',{}).get('fix_mistral_regex',False)),
                  text_only_backbone=bool(c.get('config',{}).get('text_only_backbone',False)),quantization=c.get('config',{}).get('quantization'),attention_backend=c.get('config',{}).get('attention_backend')); break
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower() or bs<=1: raise
                bs=max(1,bs//2); gc.collect(); torch.cuda.empty_cache()
    probs=np.asarray(probs,dtype=np.float64); labs=list(map(str,labs)); probs=probs[:,[labs.index(x) for x in classes]]
    assert probs.shape==(len(test),18) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
    gc.collect(); torch.cuda.empty_cache(); return probs
start=time.time(); P={sid:infer(sid) for sid in needed}
idx={label:i for i,label in enumerate(classes)}
equal9=np.mean(np.stack([P[x] for x in EQUAL9]),axis=0); pred9=np.asarray(classes,dtype=object)[equal9.argmax(1)]
eqsel=np.mean(np.stack([P[x] for x in EQUAL_SELECTED]),axis=0); pred_eqsel=np.asarray(classes,dtype=object)[eqsel.argmax(1)]
def cascade(rules):
    out=pred9.copy(); used=np.zeros(len(test),dtype=bool); hits={}
    for sid,label in rules:
        top=np.asarray(classes,dtype=object)[P[sid].argmax(1)]; fire=(~used)&(top==label); out[fire]=label; used|=fire; hits[sid+':'+label]=int(fire.sum())
    return out,hits,int(used.sum())
pred_c3,hits3,handled3=cascade(C3); pred_c1,hits1,handled1=cascade(C1)
def save(pred,name):
    f=pd.DataFrame({'id':test.id.astype(str).to_numpy(),'model':np.asarray(pred,dtype=str)}); assert f.id.tolist()==test.id.astype(str).tolist() and f.model.isin(classes).all()
    p=Path('/kaggle/working')/name; f.to_parquet(p,index=False); assert pd.read_parquet(p).equals(f); return f
o3=save(pred_c3,'submission_c3.parquet'); oe=save(pred_eqsel,'submission_equal_selected.parquet'); o1=save(pred_c1,'submission_c1.parquet'); o3.to_parquet('/kaggle/working/submission.parquet',index=False)
print(json.dumps({'rows':len(test),'test_ids':test.id.astype(str).tolist(),'predictions':{'C3':o3.model.tolist(),'equal_selected':oe.model.tolist(),'C1':o1.model.tolist()},'C3_rule_hits':hits3,'C3_handled':handled3,'C1_rule_hits':hits1,'C1_handled':handled1,'checkpoint_ids':needed,'runtime_seconds':round(time.time()-start,2)},indent=2))
'''
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(setup)])
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    path = KERNEL / "trust_v025_three_strategies.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    print(build_assets())
    print(build_kernel())
