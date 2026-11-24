"""Package E33-F over the frozen E31 roster and validate its Kaggle kernel."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nbformat

from scripts.kaggle_preflight import embed_runtime_preflight, validate_kernel_package

ROOT = Path(__file__).resolve().parents[1]
E31_DATA = ROOT / "artifacts/kaggle/e31-checkpoints-v001"
E31_KERNEL = ROOT / "artifacts/kaggle/e31-kernel/e31.ipynb"
DEPLOY = ROOT / "artifacts/analysis/e33"
E19D = ROOT / "artifacts/analysis/e28/e19_d_external_grouper.joblib"
DATA = ROOT / "artifacts/kaggle/e33f-deployment-v001"
KERNEL = ROOT / "artifacts/kaggle/e33f-kernel"
DATA_REF = "zodrak/e33f-deployment-v001"
KERNEL_REF = "zodrak/e33-f-e31-40-checkpoint-candidate-router"
QWEN_MODEL = "limamateus/qwen3-5-4b/PyTorch/default/1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _main_cell_source() -> str:
    nb = nbformat.read(E31_KERNEL, as_version=4)
    code = nb.cells[2].source
    if "len(B['members'])==40" not in code or "write_submission(\"E19-F-C\"" not in code:
        raise ValueError("E31 source is not the expected fixed 40-checkpoint deployment")
    # Insert E33-F's frozen candidate correctness adjustment immediately after
    # the independent E31 family-max scores have been constructed.
    needle = "independent=route_family_max(hard,scores,ensemble_probs,families)\nassert np.isfinite(finite_scores).all()"
    replacement = needle + '''

# E33-F candidate correctness adjustment. The deployment model was trained
# with 5-fold prompt-group OOF reliability features; no test labels are used.
from src import e33
from catboost import CatBoostClassifier
E33ROOT=root_for("e33f_manifest.json")
E33=json.loads((E33ROOT/"e33f_manifest.json").read_text())
def e33asset(name):
    hits=[p for p in E33ROOT.rglob(name) if p.is_file()]
    return only(hits,name)
reliability_file=e33asset("e33_final_reliability.npz")
with np.load(reliability_file,allow_pickle=False) as z: reliability={k:z[k] for k in z.files}
schema=json.loads(e33asset("e33_final_feature_schema.json").read_text())
if schema["checkpoint_order"]!=[m["candidate_id"] for m in members] or schema["class_order"]!=classes:
    raise ValueError("E33 feature schema does not match frozen checkpoint/class ordering")
e33._BUILD_CP_IDS=[m["candidate_id"] for m in members]
rank_position=e33._rank_positions(probs)
candidate_features=e33._build_candidate_features(np.arange(n,dtype=np.int64),reliability,
    probabilities=probs,rank_position=rank_position,base_scores=finite_scores,
    families=[m["family"] for m in members])
candidate_features=candidate_features.loc[:,schema["feature_order"]]
if list(candidate_features.columns)!=schema["feature_order"]:
    raise ValueError("E33 inference feature ordering mismatch")
e33_model=CatBoostClassifier(); e33_model.load_model(str(e33asset("e33_final.cbm")))
wrong_class=schema["model_class_order"].index(1)
p_wrong=e33_model.predict_proba(candidate_features)[:,wrong_class]
p_correct=(1.0-p_wrong).reshape(n,len(classes))
e33_scores=finite_scores*p_correct
independent=np.argmax(e33_scores,axis=1)
if not np.isfinite(e33_scores).all(): raise ValueError("E33 adjusted score matrix is nonfinite")
'''
    if needle not in code:
        raise ValueError("E31 E13-B score construction anchor not found")
    code = code.replace(needle, replacement)
    # The original response embeddings still serve E19-F-A/F-C. E19-D uses
    # separately reconstructed prompts, Gemma semantic vectors, and its own
    # frozen complete-link pair scorer.
    needle2 = '''groups_a=cluster_f_candidates("E19-F-A",ids,ordered,pairs,features,{"threshold":0.85})
groups_c=cluster_f_candidates("E19-F-C",ids,ordered,pairs,features,grouper)'''
    replacement2 = needle2 + '''

# Reconstruct each test response's original prompt independently with the
# attached Qwen3.5-4B model, then run the frozen external E19-D grouper.
from transformers import AutoModelForMultimodalLM, AutoProcessor
from src import e19b_reconstruct, e19b_group, e27
qwen_markers=[p.parent for p in INPUT.rglob("config.json") if "qwen3-5-4b" in str(p.parent).casefold()]
if not qwen_markers: raise FileNotFoundError("attached Qwen3.5-4B model is missing")
qwen_root=only([p for p in qwen_markers if len(p.parts)==min(len(x.parts) for x in qwen_markers)],"Qwen3.5-4B model")
processor=AutoProcessor.from_pretrained(qwen_root,local_files_only=True)
qwen=AutoModelForMultimodalLM.from_pretrained(qwen_root,device_map="auto",dtype="auto",local_files_only=True).eval()
recon_path=Path("/kaggle/working/e33_d_reconstructed.json")
if recon_path.is_file():
    recon_records=json.loads(recon_path.read_text())
else: recon_records=[]
if len(recon_records)>n or any(str(x.get("id"))!=str(ids[i]) for i,x in enumerate(recon_records)):
    raise ValueError("resumed E19-D reconstruction cache is not a prefix of test IDs")
reconstructed=[]; topics=[]; details=[]; constraints=[]
for start in range(0,len(recon_records),1):
    rec=recon_records[start]; reconstructed.append(rec["reconstructed_prompt"]); topics.append(rec["topic"]); details.append(rec["distinctive_details"]); constraints.append(rec["constraints"])
batch_size=2; qwen_started=time.time()
while len(recon_records)<n:
    start=len(recon_records); take=min(batch_size,n-start)
    batch_texts=texts[start:start+take]
    try:
        inputs=e19b_reconstruct._generation_inputs(processor,batch_texts)
        inputs={k:(v.to(qwen.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
        prefix=inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated=qwen.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=1024,
                use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
        raw=processor.batch_decode(generated[:,prefix:],skip_special_tokens=True)
        parsed=[]
        for j,value in enumerate(raw):
            try: item=e19b_reconstruct.parse_reconstruction(value)
            except Exception:
                # Match the established repair fallback for irrecoverable JSON.
                item={"reconstructed_prompt":str(batch_texts[j])[:4000],"topic":"",
                      "distinctive_details":[],"constraints":[],"status":"fallback"}
            item["id"]=str(ids[start+j]); parsed.append(item)
        recon_records.extend(parsed)
        temp=recon_path.with_suffix(".tmp.json"); temp.write_text(json.dumps(recon_records,ensure_ascii=False)); temp.replace(recon_path)
        reconstructed.extend(x["reconstructed_prompt"] for x in parsed); topics.extend(x["topic"] for x in parsed)
        details.extend(x["distinctive_details"] for x in parsed); constraints.extend(x["constraints"] for x in parsed)
        if start%100==0: print("E19-D reconstructed",len(recon_records),"/",n,flush=True)
    except torch.cuda.OutOfMemoryError:
        if take<=1: raise
        batch_size=max(1,take//2); torch.cuda.empty_cache()
del qwen,processor; gc.collect(); torch.cuda.empty_cache()
qwen_seconds=time.time()-qwen_started
if len(reconstructed)!=n or [x["id"] for x in recon_records]!=ids.tolist():
    raise ValueError("E19-D reconstruction output is not aligned to test order")

# E19-D's semantic similarity space is Gemma over reconstructed prompts.
tokenizer=AutoTokenizer.from_pretrained(embed_base,local_files_only=True,padding_side="right")
model=AutoModel.from_pretrained(embed_base,local_files_only=True,torch_dtype=torch.float16).to("cuda:0").eval()
semantic=[]; pos=0; batch_size=int(e28["grouping_batch_size"])
while pos<n:
    take=min(batch_size,n-pos)
    batch=["task: clustering | query: "+str(x) for x in reconstructed[pos:pos+take]]
    try:
        inputs=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            hidden=model(**inputs).last_hidden_state.float(); mask=inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1); pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
        semantic.append(pooled.cpu().numpy().astype(np.float32)); pos+=take
    except torch.cuda.OutOfMemoryError:
        if take<=1: raise
        batch_size=max(1,take//2); torch.cuda.empty_cache()
semantic=np.concatenate(semantic); semantic/=np.linalg.norm(semantic,axis=1,keepdims=True).clip(1e-12)
del model,tokenizer; gc.collect(); torch.cuda.empty_cache()
d_path=e33asset("e19_d_external_grouper.joblib"); d_grouper=joblib.load(d_path)
ensure_binary_logistic_predict_proba_compat(d_grouper["pair_scorer"]["model"])
d_vectors={name:d_grouper["vectorizers"][name] for name in ("reconstruction_char","reconstruction_word","response_char","response_word")}
d_spaces=e19b_group.transform_similarity_spaces(reconstructed,texts,d_vectors)
drows,dpairs,dx=e27._reconstruction_features(np.arange(n),d_spaces,reconstructed,texts,semantic,
    topics,details,constraints)
dg,_,_=e27._pair_cluster(ids[drows],dpairs,dx,d_grouper["pair_scorer"],float(d_grouper["threshold"]))
groups_d=np.empty(n,dtype=object); groups_d[drows]=dg
if len(groups_d)!=n or any(len(np.flatnonzero(groups_d==g))>18 for g in set(map(str,groups_d))):
    raise ValueError("E19-D groups are misaligned or exceed 18 rows")'''
    if needle2 not in code:
        raise ValueError("E31 grouping insertion anchor not found")
    code = code.replace(needle2, replacement2)
    # Use adjusted E33 scores for every assignment and emit E19-D as a fourth
    # artifact. Keep E31's canonical output filename for competition submit.
    code = code.replace('pred_a=hungarian(finite_scores,groups_a,ids)',
                        'pred_a=hungarian(e33_scores,groups_a,ids)')
    code = code.replace('pred_c=hungarian(finite_scores,groups_c,ids)',
                        'pred_c=hungarian(e33_scores,groups_c,ids)')
    code = code.replace('write_submission("E19-F-C","submission_e19_f_c.parquet",pred_c)',
        'write_submission("E19-F-C","submission_e19_f_c.parquet",pred_c)\npred_d=hungarian(e33_scores,groups_d,ids)\nwrite_submission("E19-D","submission_e19_d.parquet",pred_d)')
    code = code.replace('"grouping_candidate_edges":len(pairs)',
        '"e33_model":"e33_final.cbm","e33_adjustment":"E31_score * p_correct",\n    "e19d_reconstruction_seconds":qwen_seconds,"e19d_group_count":len(set(map(str,groups_d))),\n    "grouping_candidate_edges":len(pairs)')
    # E33 adjustment must also be the independent output; the earlier
    # independent variable was replaced after computing the E31 audit baseline.
    code = code.replace('write_submission("independent_E13-B","submission.parquet",independent)',
                        'write_submission("E33-F Independent","submission.parquet",independent)')
    code = code.replace('write_submission("E19-F-A","submission_e19_f_a.parquet",pred_a)',
                        'write_submission("E33-F E19-F-A","submission_e19_f_a.parquet",pred_a)')
    code = code.replace('write_submission("E19-F-C","submission_e19_f_c.parquet",pred_c)',
                        'write_submission("E33-F E19-F-C","submission_e19_f_c.parquet",pred_c)')
    code = code.replace('write_submission("E19-D","submission_e19_d.parquet",pred_d)',
                        'write_submission("E33-F E19-D","submission_e19_d.parquet",pred_d)')
    # Ensure the serialized E19-D subgrouper's compatibility repair is applied
    # only to the explicitly declared artifact in the early runtime gate.
    return code


def prepare():
    if not E31_DATA.is_dir() or not E31_KERNEL.is_file():
        raise FileNotFoundError("E31 packaged checkpoint assets/kernel are missing")
    if DATA.exists(): shutil.rmtree(DATA)
    if KERNEL.exists(): shutil.rmtree(KERNEL)
    DATA.mkdir(parents=True); KERNEL.mkdir(parents=True)
    required = [DEPLOY / "e33_final.cbm", DEPLOY / "e33_final_reliability.npz",
                DEPLOY / "e33_final_feature_schema.json", E19D]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError(f"E33-F deployment assets missing: {[str(x) for x in required if not x.is_file()]}")
    assets = []
    for source in required:
        target = DATA / source.name
        shutil.copy2(source,target)
        assets.append({"filename":target.name,"size":target.stat().st_size,"sha256":sha(target)})
    e31 = json.loads((E31_DATA / "e31_manifest.json").read_text())
    e33_schema = json.loads((DEPLOY / "e33_final_feature_schema.json").read_text())
    if len(e31["roster"]) != 40 or e33_schema["checkpoint_order"] != e31["roster"]:
        raise ValueError("E31 roster and E33-F checkpoint order differ")
    manifest = {"schema_version":1,"experiment":"E33-F","trust_list_version":"trust-list-2-v037",
        "roster":e31["roster"],"class_order":e31["classes"],"assets":assets,
        "model":"e33_final.cbm","reliability":"e33_final_reliability.npz",
        "feature_schema":"e33_final_feature_schema.json","e19d_grouper":"e19_d_external_grouper.joblib",
        "e19d_threshold":0.35,"e19d_max_cluster_size":18,
        "qwen_model_revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "output_files":["submission.parquet","submission_e19_f_a.parquet","submission_e19_f_c.parquet","submission_e19_d.parquet"]}
    (DATA/"e33f_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (DATA/"dataset-metadata.json").write_text(json.dumps({"title":"E33-F E31-40 Deployment Artifacts",
        "id":DATA_REF,"licenses":[{"name":"other"}],"subtitle":"E33-F final scorer and external E19-D grouper",
        "description":"E33-F classifier, OOF reliability statistics, fixed schema and E19-D grouper. Checkpoint assets remain in the existing E31 dataset sources."},indent=2)+"\n")
    nb=nbformat.read(E31_KERNEL,as_version=4)
    nb.cells[0].source="# E33-F — Frozen E31-40 family-max router with learned candidate correctness adjustment\n\nProduces four validated submission variants from one 40-checkpoint inference pass."
    nb.cells[2].source=_main_cell_source()
    nbformat.write(nb,KERNEL/"e33f.ipynb")
    old_meta=json.loads((ROOT/"artifacts/kaggle/e31-kernel/kernel-metadata.json").read_text())
    old_meta.update({"id":KERNEL_REF,"title":"E33 F E31 40 Checkpoint Candidate Router",
        "code_file":"e33f.ipynb","competition_sources":["name-that-llm"],
        "dataset_sources":sorted(set(old_meta.get("dataset_sources",[]))|{DATA_REF}),
        "model_sources":sorted(set(old_meta.get("model_sources",[]))|{QWEN_MODEL}),
        "enable_gpu":True,"enable_internet":False,"machine_shape":"NvidiaTeslaT4x2"})
    (KERNEL/"kernel-metadata.json").write_text(json.dumps(old_meta,indent=2)+"\n")
    # Keep only E19-F-C and E19-D in the early sklearn compatibility allowlist;
    # unrelated serialized checkpoint joblibs remain unopened until runtime.
    compat=[{"dataset_ref":"zodrak/e31-checkpoints-v001","filename":"e19_f_c_external_grouper.joblib"},
            {"dataset_ref":DATA_REF,"filename":"e19_d_external_grouper.joblib"}]
    embed_runtime_preflight(KERNEL,manifest_paths=[E31_DATA/"e31_manifest.json",E31_DATA/"e28_manifest.json",DATA/"e33f_manifest.json"],
        processor_sources=[QWEN_MODEL],
        sklearn_compatibility_assets=compat)
    return {"dataset":str(DATA),"kernel":str(KERNEL),"kernel_ref":KERNEL_REF,
        "qwen_model_source":QWEN_MODEL,"assets":assets,"preflight":validate_kernel_package(KERNEL)}


if __name__=="__main__":
    print(json.dumps(prepare(),indent=2))
