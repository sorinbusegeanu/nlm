"""Package the frozen E13-B v037 route with E28 external groupers."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nbformat as nbf
import pandas as pd

from scripts.prepare_e13_b_kaggle import NOTEBOOK_CODE as E13_NOTEBOOK
from scripts.prepare_e13_b_kaggle import ROOT
from scripts.kaggle_preflight import embed_runtime_preflight, validate_kernel_package
from src.e28_deploy import checkpoint_bundle_relative_path, distribute_checkpoints

ANALYSIS = ROOT / "artifacts/analysis/e28"
MANIFEST_SOURCE = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
GROUPER_DIR = ROOT / "artifacts/kaggle/e28-groupers-v001"
KERNEL_DIR = ROOT / "artifacts/kaggle/e28-e13b-kernel"
GROUPER_REF = "zodrak/e28-external-groupers-v001"
KERNEL_REF = "zodrak/e28-e13-b-external-grouping-hungarian"
BATCH_BENCHMARK = ROOT / "artifacts/analysis/e20d_inference_batch_benchmark.json"
RUNTIME_ESTIMATES = ROOT / "artifacts/analysis/e20d_checkpoint_runtime_estimates.json"


WORKER_CODE = r'''import gc,json,os,sys,time
from pathlib import Path
import inspect
import numpy as np
import torch

# The checkpoint shard also has a package named ``src``. Import the pinned
# inference runtime explicitly before that working directory can shadow it.
runtime_root=Path(os.environ["NLM_INFERENCE_RUNTIME"]).resolve()
sys.path.insert(0,str(runtime_root))
from src.inference import predict_candidate
expected_inference=(runtime_root/"src"/"inference.py").resolve()
actual_inference=Path(inspect.getfile(predict_candidate)).resolve()
if actual_inference!=expected_inference:
    raise RuntimeError(f"wrong inference module loaded: {actual_inference}; expected {expected_inference}")
if "return_batch_metadata" not in inspect.signature(predict_candidate).parameters:
    raise RuntimeError(f"packaged inference API lacks return_batch_metadata: {actual_inference}")

tasks_path,out_dir,text_path,classes_path,go_path,ready_path=sys.argv[1:]
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise RuntimeError("E28 worker must see exactly its assigned GPU")
worker=int(os.environ["WORKER_GPU_ID"])
tasks=json.loads(Path(tasks_path).read_text())
texts=json.loads(Path(text_path).read_text())
classes=json.loads(Path(classes_path).read_text())
Path(ready_path).touch()
deadline=time.time()+600
while not Path(go_path).exists():
    if time.time()>deadline: raise TimeoutError("E28 parent did not release worker barrier")
    time.sleep(.05)
records=[]; total_start=time.time()
for spec in tasks:
    sid=spec["candidate_id"]; started=time.time(); torch.cuda.reset_peak_memory_stats()
    probs,batch=predict_candidate(spec,texts,classes,Path("/kaggle/working/e28_runtime_root"),
                                  return_batch_metadata=True)
    probs=np.asarray(probs,dtype=np.float32)
    if probs.shape!=(len(texts),len(classes)) or not np.isfinite(probs).all():
        raise ValueError(f"invalid prediction cache for {sid}: {probs.shape}")
    if not np.allclose(probs.sum(1),1,atol=1e-4): raise ValueError(f"unnormalized probabilities: {sid}")
    out=Path(out_dir)/f"{sid}.npz"; temp=out.with_name(out.stem+".tmp.npz")
    np.savez_compressed(temp,probabilities=probs); os.replace(temp,out)
    ended=time.time()
    records.append({"candidate_id":sid,"gpu_id":worker,"start_epoch":started,"end_epoch":ended,
        "seconds":ended-started,"effective_batch_size":batch.get("effective_batch_size"),
        "peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes":int(torch.cuda.max_memory_reserved()),"cache":str(out)})
    del probs; gc.collect(); torch.cuda.empty_cache()
Path(out_dir,f"worker{worker}_records.json").write_text(json.dumps({
    "gpu_id":worker,"worker_start_epoch":total_start,"worker_end_epoch":time.time(),
    "records":records},indent=2)+"\n")
print(json.dumps({"gpu_id":worker,"records":records},indent=2),flush=True)
'''


WORKER_REPLACEMENT = r'''WORKER_CODE = __WORKER_CODE_LITERAL__
members=B["members"]; classes=B["classes"]
test_paths=[INPUT/"competitions/name-that-llm/test.csv",INPUT/"name-that-llm/test.csv"]
present=[p for p in test_paths if p.is_file()]; assert present, "competition test.csv is missing"
test=pd.read_csv(present[0],dtype={"id":"string"})
assert {"id","text"}<=set(test.columns) and test.id.notna().all() and test.id.is_unique and test.text.notna().all()
texts=test.text.astype(str).tolist(); n=len(texts); started=time.time()
if torch.cuda.device_count()<2:
    raise RuntimeError(f"E28 requires Kaggle T4 x2; found {torch.cuda.device_count()} visible GPU(s)")
E28ROOT=root_for("e28_manifest.json")
e28=json.loads((E28ROOT/"e28_manifest.json").read_text())
NLM_RUNTIME_ROOT=E28ROOT/"runtime"
if not (NLM_RUNTIME_ROOT/"src"/"inference.py").is_file():
    runtime_archives=[p for p in E28ROOT.glob("runtime*.zip") if p.is_file()]
    if not runtime_archives: raise FileNotFoundError(f"Pinned inference runtime missing under {E28ROOT}")
    extracted=Path("/kaggle/working/e28_pinned_runtime"); extracted.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(runtime_archives[0]) as archive: archive.extractall(extracted)
    NLM_RUNTIME_ROOT=extracted/"runtime" if (extracted/"runtime"/"src"/"inference.py").is_file() else extracted
if not (NLM_RUNTIME_ROOT/"src"/"inference.py").is_file():
    raise FileNotFoundError(f"Could not unpack pinned inference runtime from {E28ROOT}")
runtime_by_id=dict(e28["runtime_by_id"]); runtime_by_id["__sample_rows__"]=64
schedule=distribute_checkpoints(members,runtime_by_id,e28["inference_batch_defaults"],n)
specs={}
for m in members:
    checkpoint_root=WORK/"checkpoints"/m["candidate_id"]; checkpoint_root.mkdir(parents=True,exist_ok=True)
    for item in m["files"]:
        src=resolve_asset(item); dest=checkpoint_root/item["relative_path"]; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.symlink_to(src)
    spec={k:v for k,v in m.items() if k not in ("files","deployment")}
    spec["checkpoint"]=checkpoint_bundle_relative_path(m["checkpoint_relative_path"])
    spec.setdefault("config",{})["inference_batch_size"]=int(e28["inference_batch_defaults"].get(m["model"],64))
    if m["model"] in BASES: spec["base_model"]=str(BASES[m["model"]])
    specs[m["candidate_id"]]=spec
runtime=Path("/kaggle/working/e28_runtime"); runtime.mkdir(parents=True,exist_ok=True)
pred_dir=Path("/kaggle/working/e28_probabilities"); pred_dir.mkdir(exist_ok=True)
text_path=WORK/"texts.json"; text_path.write_text(json.dumps(texts))
classes_path=WORK/"classes.json"; classes_path.write_text(json.dumps(classes))
go_path=WORK/"workers_go"; worker_path=WORK/"e28_gpu_worker.py"; worker_path.write_text(WORKER_CODE)
processes=[]; logs=[]; ready=[]
for gpu in (0,1):
    task_path=WORK/f"worker{gpu}_tasks.json"
    assigned=[specs[sid] for sid in schedule[gpu]["checkpoint_ids"]]
    if not assigned: raise RuntimeError(f"E28 GPU {gpu} received no checkpoint tasks")
    task_path.write_text(json.dumps(assigned))
    log=(WORK/f"worker{gpu}.log").open("w"); logs.append(log)
    ready_path=WORK/f"worker{gpu}.ready"; ready.append(ready_path)
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env["WORKER_GPU_ID"]=str(gpu)
    env["OMP_NUM_THREADS"]="1"; env["MKL_NUM_THREADS"]="1"; env["OPENBLAS_NUM_THREADS"]="1"
    env["TOKENIZERS_PARALLELISM"]="false"; env["PYTHONPATH"]=str(E28ROOT/"runtime")+os.pathsep+str(RUNTIME)+os.pathsep+env.get("PYTHONPATH","")
    env["NLM_INFERENCE_RUNTIME"]=str(NLM_RUNTIME_ROOT)
    processes.append(subprocess.Popen([sys.executable,str(worker_path),str(task_path),str(pred_dir),
        str(text_path),str(classes_path),str(go_path),str(ready_path)],env=env,stdout=log,stderr=subprocess.STDOUT))
ready_deadline=time.time()+600
while not all(p.exists() for p in ready):
    if any(proc.poll() not in (None,0) for proc in processes):
        raise RuntimeError("An E28 worker exited before reaching the start barrier")
    if time.time()>ready_deadline: raise TimeoutError("E28 GPU workers did not become ready")
    time.sleep(.1)
go_path.touch()
codes=[proc.wait() for proc in processes]
for log in logs: log.close()
for gpu,code in enumerate(codes):
    print((WORK/f"worker{gpu}.log").read_text(),flush=True)
    if code: raise RuntimeError(f"E28 GPU worker {gpu} failed with exit code {code}")
worker_records=[json.loads((pred_dir/f"worker{gpu}_records.json").read_text()) for gpu in (0,1)]
records=[record for worker in worker_records for record in worker["records"]]
overlap=any(max(a["start_epoch"],b["start_epoch"])<min(a["end_epoch"],b["end_epoch"])
    for a in records if a["gpu_id"]==0 for b in records if b["gpu_id"]==1)
if not overlap: raise RuntimeError("E28 concurrency check failed: the GPU workers did not overlap")
if {r["candidate_id"] for r in records}!={m["candidate_id"] for m in members}:
    raise RuntimeError("E28 worker output roster differs from frozen v037 roster")
by_id={sid:np.load(pred_dir/f"{sid}.npz",allow_pickle=False)["probabilities"] for sid in
       [m["candidate_id"] for m in members]}
probs=np.stack([by_id[m["candidate_id"]] for m in members]).astype(np.float64)
effective_batches={r["candidate_id"]:r["effective_batch_size"] for r in records}
worker_report={"schedule":[{k:v for k,v in w.items() if k!="checkpoints"} for w in schedule],
    "records":records,"parallel_overlap_verified":overlap,
    "parallel_inference_seconds":time.time()-started,
    "peak_allocated_bytes_by_gpu":{str(g):max((r["peak_allocated_bytes"] for r in records if r["gpu_id"]==g),default=0) for g in (0,1)}}
Path("/kaggle/working/e28_worker_report.json").write_text(json.dumps(worker_report,indent=2)+"\n")
print(json.dumps(worker_report,indent=2),flush=True)
'''


GROUPING_REPLACEMENT = r'''# Family-max score outputs retain the frozen v037 routing decision.
def family_max_score_matrix(hard,reliability,families):
    hard=np.asarray(hard,dtype=np.int64); reliability=np.asarray(reliability,dtype=np.float64)
    m,n=hard.shape; k=len(classes); scores=np.full((n,k),-np.inf,dtype=np.float64)
    for row in range(n):
        for label in range(k):
            voters=np.flatnonzero(hard[:,row]==label)
            if len(voters):
                by={}
                for i in voters: by[families[i]]=max(by.get(families[i],-np.inf),reliability[i,row])
                scores[row,label]=np.mean(list(by.values()))
    return scores
families=np.asarray([m["family"] for m in members]); family_names=sorted(set(families))
ensemble_probs=np.mean([probs[families==f].mean(0) for f in family_names],axis=0)
ensemble_probs/=ensemble_probs.sum(1,keepdims=True)
hard=probs.argmax(2)
independent=route_family_max(hard,scores,ensemble_probs,families)
score_matrix=family_max_score_matrix(hard,scores,families)
for row in range(n):
    best=score_matrix[row].max()
    tied=np.flatnonzero(np.isclose(score_matrix[row],best,rtol=0,atol=1e-12))
    expected=min(tied,key=lambda label:(-ensemble_probs[row,label],label))
    if independent[row]!=expected: raise AssertionError("E13-B v037 independent prediction parity failed")
finite_scores=score_matrix.copy()
for row in range(n):
    supported=np.isfinite(score_matrix[row])
    if not supported.any(): raise ValueError(f"no supported labels for row {row}")
    fallback=score_matrix[row,supported].min()-1.0
    p=ensemble_probs[row]/max(float(ensemble_probs[row].max()),1e-12)
    finite_scores[row,supported]+=p[supported]*1e-12
    finite_scores[row,~supported]=fallback+p[~supported]*1e-9
assert np.isfinite(finite_scores).all()

# No checkpoint inference competes with grouping inference. Reuse the freed
# GPUs for exact dense top-40 semantic-neighbor retrieval after both workers end.
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel,AutoTokenizer
from src import e19f
from src.e28_deploy import ensure_binary_logistic_predict_proba_compat
grouper_path=E28ROOT/"deployment"/"e19_f_c_external_grouper.joblib"
if not grouper_path.is_file(): grouper_path=E28ROOT/"e19_f_c_external_grouper.joblib"
if not grouper_path.is_file(): raise FileNotFoundError(f"E19-F-C grouper artifact missing under {E28ROOT}")
grouper=joblib.load(grouper_path)
# E19-F-C's binary LogisticRegression was serialized by sklearn 1.9.1.
# Kaggle's offline image currently has sklearn 1.6.1, whose predict_proba
# still reads ``multi_class`` although the newer estimator no longer stores
# that deprecated attribute.  The fitted binary logistic model's equivalent
# legacy setting is ``auto``; restoring it keeps its probability semantics.
pair_estimator=grouper["pair_model"]["model"]
ensure_binary_logistic_predict_proba_compat(pair_estimator)
embed_base=BASES["embeddinggemma2"]
tokenizer=AutoTokenizer.from_pretrained(embed_base,local_files_only=True,padding_side="right")
dtype=torch.float16
model=AutoModel.from_pretrained(embed_base,local_files_only=True,torch_dtype=dtype).to("cuda:0").eval()
embedding_started=time.time(); emb=[]; pos=0; batch_size=int(e28["grouping_batch_size"])
while pos<n:
    take=min(batch_size,n-pos)
    batch=["task: clustering | query: "+str(x) for x in texts[pos:pos+take]]
    try:
        inputs=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            hidden=model(**inputs).last_hidden_state.float()
            mask=inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
        emb.append(pooled.cpu().numpy().astype(np.float32)); pos+=take
    except torch.cuda.OutOfMemoryError:
        if take<=1: raise
        batch_size=max(1,take//2); torch.cuda.empty_cache()
emb=np.concatenate(emb); emb/=np.linalg.norm(emb,axis=1,keepdims=True).clip(1e-12)
del model,tokenizer; gc.collect(); torch.cuda.empty_cache()
embedding_seconds=time.time()-embedding_started
lexical=e19f._transform_lexical(np.asarray(texts,dtype=str),grouper["vectorizers"])
ordered=np.argsort(np.asarray(test.id.astype(str)),kind="stable")
ids=test.id.astype(str).to_numpy()
vectors=torch.as_tensor(emb[ordered],device="cuda:0")
candidate_edges=set(); k=min(41,n)
for start in range(0,n,512):
    sim=vectors[start:start+512]@vectors.T
    nearest=torch.topk(sim,k=k,dim=1,largest=True,sorted=True).indices.cpu().numpy()
    for offset,neighbors in enumerate(nearest):
        left=start+offset
        for right in neighbors:
            right=int(right)
            if right!=left: candidate_edges.add((min(left,right),max(left,right)))
pairs=np.asarray(sorted(candidate_edges),dtype=np.int64).reshape(-1,2)
features=e19f._pair_features(ordered,pairs,lexical,emb)
score_a=features[:,0]
score_c=e19f.score_similarity_model(grouper["pair_model"],features)
pred_a,_=e19f._cluster_fc(ids[ordered],pairs,score_a,0.85)
pred_c,_=e19f._cluster_fc(ids[ordered],pairs,score_c,float(grouper["threshold"]))
groups_a=np.empty(n,dtype=object); groups_a[ordered]=pred_a
groups_c=np.empty(n,dtype=object); groups_c[ordered]=pred_c

def hungarian(scores,groups,row_ids):
    out=np.full(len(groups),-1,dtype=np.int64)
    for group in sorted(set(map(str,groups))):
        rows=np.flatnonzero(np.asarray(groups,dtype=str)==group)
        if len(rows)>len(classes): raise ValueError(f"group exceeds 18 rows: {group}")
        ri,ci=linear_sum_assignment(-scores[rows]); out[rows[ri]]=ci
    if (out<0).any(): raise ValueError("Hungarian assignment missed rows")
    return out
submissions={}
for code,groups in (("E28-B",groups_c),("E28-C",groups_a)):
    pred=hungarian(finite_scores,groups,ids)
    submission=pd.DataFrame({"id":ids,"model":np.asarray(classes,dtype=object)[pred].astype(str)})
    assert submission.id.tolist()==test.id.astype(str).tolist() and submission.id.is_unique
    assert submission.model.notna().all() and submission.model.isin(classes).all()
    out=Path("/kaggle/working")/f"submission_{code.lower().replace('-','_')}.parquet"
    submission.to_parquet(out,index=False)
    assert pd.read_parquet(out).equals(submission)
    if code=="E28-B":
        canonical=Path("/kaggle/working/submission.parquet")
        submission.to_parquet(canonical,index=False)
        assert pd.read_parquet(canonical).equals(submission)
    submissions[code]={"path":str(out),"rows":len(submission),
        "submission_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "changed_from_independent":int(np.count_nonzero(pred!=independent))}
run={"experiment":"E28","trust_list":B["trust_list"],"variants":submissions,
    "checkpoint_ids":[m["candidate_id"] for m in members],"classes":classes,
    "rows":n,"test_sha256":hashlib.sha256(present[0].read_bytes()).hexdigest(),
    "checkpoint_runtime":worker_report,"grouping_method":"E27 external E19-F-A threshold 0.85; E19-F-C frozen external scorer threshold 0.45",
    "grouping_candidate_edges":len(pairs),"grouping_embedding_seconds":embedding_seconds,
    "grouping_seconds":time.time()-embedding_started,"total_seconds":time.time()-started,
    "inference_batch_sizes":effective_batches}
Path("/kaggle/working/e28_run_manifest.json").write_text(json.dumps(run,indent=2)+"\n")
print(json.dumps(run,indent=2),flush=True)
'''


def _notebook_code():
    code = E13_NOTEBOOK
    # The original single-worker inference import is replaced in the parent
    # notebook before embedding WORKER_CODE, so this replacement cannot alter
    # the worker's quoted Python source.
    code = code.replace("from src.inference import predict_candidate",
                        "from src.e28_deploy import checkpoint_bundle_relative_path, distribute_checkpoints")
    code = code.replace("__WORKER_CODE_LITERAL__", repr(WORKER_CODE))
    code = code.replace(
        "assert torch.cuda.is_available(), 'Kaggle GPU is required for checkpoint inference'",
        "assert torch.cuda.is_available(), 'Kaggle GPU is required'; assert torch.cuda.device_count() >= 2, f'E28 requires two GPUs, found {torch.cuda.device_count()}'")
    code = code.replace("LC3=root_for('lc3_lc4_manifest.json'); BROOT=root_for('e13_b_manifest.json')",
        "LC3=root_for('lc3_lc4_manifest.json'); BROOT=root_for('e13_b_manifest.json')")
    begin = code.index("members=B['members']; classes=B['classes']")
    end = code.index("probs=np.stack(probs)", begin) + len("probs=np.stack(probs)")
    replacement = WORKER_REPLACEMENT.replace("__WORKER_CODE_LITERAL__", repr(WORKER_CODE))
    code = code[:begin] + replacement + code[end:]
    start = code.index("families=np.asarray([m['family'] for m in members]); family_names=sorted(set(families))", begin)
    # Replace the original final scoring and submission block, keeping the
    # packaged feature builder and CatBoost inference above this point.
    code = code[:start] + GROUPING_REPLACEMENT
    code = code.replace("BROOT=root_for('e13_b_manifest.json')",
        "BROOT=root_for('e13_b_manifest.json'); E28ROOT=root_for('e28_manifest.json')")
    code = code.replace("RUNTIME=WORK/'src'", "RUNTIME=WORK/'src'")
    code = code.replace("sys.path.insert(0,str(WORK))",
        "sys.path.insert(0,str(E28ROOT/'runtime')); sys.path.insert(0,str(WORK))")
    code = code.replace("sys.path.insert(0,str(BROOT))",
        "sys.path.insert(0,str(BROOT)); sys.path.insert(0,str(E28ROOT/'runtime'))")
    code = code.replace('str(RUNTIME)+os.pathsep+env.get("PYTHONPATH","")',
        'str(WORK)+os.pathsep+env.get("PYTHONPATH","")')
    code = code.replace("WORK=Path('/kaggle/working/e13_b_runtime');", "WORK=Path('/kaggle/working/e28_runtime_root');")
    code = code.replace("BASES= {}", "BASES= {}")
    code = code.replace("WORK=Path('/kaggle/working/e28_runtime_root'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)",
        "WORK=Path('/kaggle/working/e28_runtime_root'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)")
    code = code.replace("Path('/kaggle/working/submission.parquet')", "Path('/kaggle/working/submission.parquet')")
    return code


def _manifest():
    if not MANIFEST_SOURCE.is_file():
        raise FileNotFoundError(f"frozen E13-B v037 package manifest missing: {MANIFEST_SOURCE}")
    return json.loads(MANIFEST_SOURCE.read_text())


def prepare():
    if not (ANALYSIS / "e19_f_c_external_grouper.joblib").is_file():
        raise FileNotFoundError("Run E28 external-only grouper fitting/evaluation first")
    GROUPER_DIR.mkdir(parents=True, exist_ok=True)
    (GROUPER_DIR / "deployment").mkdir(exist_ok=True)
    shutil.copy2(ANALYSIS / "e19_f_c_external_grouper.joblib",
                 GROUPER_DIR / "deployment/e19_f_c_external_grouper.joblib")
    shutil.copytree(ROOT / "src", GROUPER_DIR / "runtime/src", dirs_exist_ok=True)
    batch = json.loads(BATCH_BENCHMARK.read_text())
    batch_by_model = {item["model"]: int(item["effective_batch_size"])
                      for item in batch["results"] if item.get("status") == "ok"}
    if not batch_by_model:
        raise ValueError("No successful calibrated inference batch sizes are available")
    runtime_by_id = {}
    runtime_by_id.update({str(item["sequence_id"]): float(item["seconds"])
        for item in batch["results"] if item.get("status") == "ok" and
        item.get("sequence_id") and item.get("seconds") is not None})
    if RUNTIME_ESTIMATES.is_file():
        runtime_doc = json.loads(RUNTIME_ESTIMATES.read_text())
        runtime_by_id.update({str(k): float(v) for k, v in runtime_doc.get("seconds_by_checkpoint", {}).items()})
    e13 = _manifest()
    roster = list(e13["deployment"]["checkpoint_ids"])
    members = {m["candidate_id"]: m for m in e13["members"]}
    if len(roster) != 15 or set(roster) != set(members):
        raise ValueError("E28 packaging requires the frozen E13-B v037 15-member deployment roster")
    planned = distribute_checkpoints(e13["members"], runtime_by_id, batch_by_model, 4000)
    package_manifest = {"schema_version": 1, "experiment": "E28",
        "trust_list_version": e13["trust_list"]["version"], "checkpoint_ids": roster,
        "e13_b_manifest_source": "zodrak/e13-b-checkpoints-v001",
        "frozen_groupers": {
            "E19-F-A": {"threshold": 0.85, "method": "external E27 E19-F-A", "max_cluster_size": 18},
            "E19-F-C": {"file": "deployment/e19_f_c_external_grouper.joblib",
                "threshold": 0.45, "method": "external E27 E19-F-C", "max_cluster_size": 18}},
        "grouping_artifact": {"dataset_ref": GROUPER_REF,
            "flat_filename": "deployment/e19_f_c_external_grouper.joblib",
            "size": (GROUPER_DIR / "deployment/e19_f_c_external_grouper.joblib").stat().st_size},
        "grouping_artifact_sha256": hashlib.sha256(
            (ANALYSIS / "e19_f_c_external_grouper.joblib").read_bytes()).hexdigest(),
        "inference_batch_defaults": batch_by_model, "runtime_by_id": runtime_by_id,
        "runtime_estimate_source": "local 64-row per-checkpoint batch calibration" +
            (f" plus {RUNTIME_ESTIMATES}" if RUNTIME_ESTIMATES.is_file() else "; size-based fallback for unmeasured IDs"),
        "schedule_estimate": [{"gpu_id": x["gpu_id"], "estimated_seconds": x["estimated_seconds"],
                               "checkpoint_ids": x["checkpoint_ids"]} for x in planned],
        "grouping_batch_size": 64}
    (GROUPER_DIR / "e28_manifest.json").write_text(json.dumps(package_manifest, indent=2) + "\n")
    (GROUPER_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "E28 External Grouping Assets",
        "id": GROUPER_REF, "licenses": [{"name": "other"}],
        "subtitle": "E27 externally trained E19-F-C grouping scorer and E28 runtime",
        "description": "External-only grouping model and inference runtime for E28; no V0 labels used for training or threshold selection."}, indent=2) + "\n")
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# E28 — E13-B v037 + External Grouping + Hungarian\n\nRuns the frozen 15-checkpoint E13-B router with two concurrent T4 workers, then applies E27 external E19-F-C and E19-F-A grouping followed by Hungarian assignment. The kernel fails unless both GPUs are visible and checkpoint work overlaps."),
        nbf.v4.new_code_cell(_notebook_code())])
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nbf.write(notebook, KERNEL_DIR / "e28.ipynb")
    dataset_sources = sorted({GROUPER_REF} |
        {item["dataset_ref"] for member in e13["members"] for item in member["files"]})
    kernel_metadata = {"id": KERNEL_REF, "title": "E28 E13-B External Grouping Hungarian",
        "code_file": "e28.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": ["name-that-llm", "E28", "Hungarian"],
        "dataset_sources": dataset_sources, "kernel_sources": [],
        "competition_sources": ["name-that-llm"],
        "model_sources": ["google/embeddinggemma-2/Transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/Transformers/ministral-3-3b-base-2512/1",
            "qwen-lm/qwen-3/Transformers/0.6b/1"],
        "machine_shape": "NvidiaTeslaT4"}
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n")
    embed_runtime_preflight(
        KERNEL_DIR,
        manifest_paths=[MANIFEST_SOURCE, GROUPER_DIR / "e28_manifest.json"],
    )
    package_preflight = validate_kernel_package(KERNEL_DIR)
    return {"kernel_ref": KERNEL_REF, "groupers_ref": GROUPER_REF,
        "checkpoint_ids": roster, "checkpoint_count": len(roster),
        "dataset_sources": dataset_sources, "kernel_dir": str(KERNEL_DIR),
        "preflight": package_preflight,
        "groupers_dir": str(GROUPER_DIR)}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
