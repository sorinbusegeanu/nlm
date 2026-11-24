"""Build a local E31 40-checkpoint two-GPU Kaggle dataset delta and notebook; never submits."""
from __future__ import annotations

import csv, hashlib, json, os, shutil
from pathlib import Path

import nbformat as nbf
from huggingface_hub import snapshot_download

from scripts.publish_kaggle_checkpoint_store import ROOT, build_checkpoint, inference_files, sha256_file
from scripts.prepare_e12_2_kaggle import _asset_ref, _remote_assets
from scripts.prepare_e28_kaggle import _notebook_code, WORKER_CODE as E28_WORKER_CODE
from scripts.kaggle_preflight import embed_runtime_preflight, validate_kernel_package

LOCAL = ROOT / "artifacts/kaggle/e31-checkpoints-v001"
KERNEL = ROOT / "artifacts/kaggle/e31-kernel"
ANALYSIS = ROOT / "artifacts/analysis/e31"
TRUST = ROOT / "docs/trust-list-2-v037.csv"
E28_MANIFEST = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
DATASET_REF = "zodrak/e31-checkpoints-v001"
KERNEL_REF = "zodrak/e31-40-checkpoint-e13-b-with-external-hungarian"


def prepare():
    deployment = json.loads((ANALYSIS / "e31_manifest.json").read_text())
    bench = json.loads((ANALYSIS / "inference_benchmark.json").read_text())
    bench_by_id = {r["checkpoint_id"]: r for r in bench["checkpoints"]}
    for observed in bench.get("end_to_end_checkpoint_records", []):
        sid = observed["checkpoint_id"]
        effective = observed.get("maximum_stable_batch_size")
        if effective:
            previous = bench_by_id[sid].get("maximum_stable_batch_size")
            bench_by_id[sid]["maximum_stable_batch_size"] = (
                min(int(previous), int(effective)) if previous else int(effective))
            bench_by_id[sid]["e2e_observed_batch_size"] = int(effective)
    if len(deployment["roster"]) != 40:
        raise ValueError("E31 deployment must contain exactly 40 checkpoints")
    if LOCAL.exists(): shutil.rmtree(LOCAL)
    if KERNEL.exists(): shutil.rmtree(KERNEL)
    (LOCAL / "assets").mkdir(parents=True)
    KERNEL.mkdir(parents=True)

    # Search the published NLM checkpoint datasets, deduplicating by hash.
    data_sources = {
        "zodrak/zod006-models": "e20d-remote-catalogs/zod006-models",
        "zodrak/lc3-lc4-checkpoints-v035": "e20d-remote-catalogs/lc3-lc4-checkpoints-v035",
        "zodrak/three-strategy-checkpoints": "e20d-remote-catalogs/three-strategy-checkpoints",
        "zodrak/ensemble-hypotheses-nov15-extra-checkpoints": "e20d-remote-catalogs/ensemble-hypotheses-nov15-extra-checkpoints",
        "zodrak/trust2-v013-top5-models": "e20d-remote-catalogs/trust2-v013-top5-models",
        "zodrak/e13-b-checkpoints-v001": "e20d-remote-catalogs/e13-b-checkpoints-v001",
        "zodrak/e28-external-groupers-v001": "e28-groupers-v001",
    }
    from scripts import prepare_e12_2_kaggle as e12
    previous = e12.DATASETS
    e12.DATASETS = data_sources
    try:
        inventory, by_hash = _remote_assets()
    finally:
        e12.DATASETS = previous

    old = json.loads(E28_MANIFEST.read_text())
    old_members = {m["candidate_id"]: dict(m) for m in old["members"]}
    with TRUST.open() as stream:
        trust = {row["sequence_id"]: row for row in csv.DictReader(stream)}
    new_assets = {}
    members = []
    for sid in deployment["roster"]:
        if sid in old_members:
            member = old_members[sid]
        else:
            row = trust[sid]
            exp = row["experiment_id"]
            cp_path = Path(row["checkpoint_path"])
            if not cp_path.is_absolute(): cp_path = ROOT / cp_path
            prediction_manifest = ROOT / "artifacts/predictions" / f"{exp}_manifest.json"
            if prediction_manifest.is_file():
                pred = json.loads(prediction_manifest.read_text())
                local = build_checkpoint(exp)
            else:
                # Trust List 2 can include a checkpoint whose run manifest is
                # stored beside its model payload (for example two-stage
                # CatBoost), with no artifacts/predictions manifest.
                run_manifest = cp_path.parent / f"{exp}_manifest.json"
                if not run_manifest.is_file() or not cp_path.exists():
                    raise FileNotFoundError(
                        f"Cannot package {sid}: missing prediction and checkpoint-side manifests "
                        f"for {exp} ({run_manifest})")
                pred = json.loads(run_manifest.read_text())
                local = {"files": inference_files(cp_path, row["model"])}
            member = {"candidate_id": sid, "experiment_id": exp, "model": row["model"],
                "family": deployment["families"][deployment["roster"].index(sid)],
                "config": pred.get("config", {}), "model_revision": pred.get("model_revision"),
                "calibration_temperature": float(pred.get("calibration_temperature") or 1.0),
                "checkpoint_relative_path": (Path(sid) if cp_path.is_dir() else Path(sid)/cp_path.name).as_posix(),
                "checkpoint_type": "directory" if cp_path.is_dir() else "file", "files": []}
            for source, relative in local["files"]:
                digest, size = sha256_file(source), source.stat().st_size
                ref = _asset_ref(digest, size, by_hash, inventory)
                if ref is None:
                    filename = f"assets/asset__{digest}"
                    target = LOCAL / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    ref = {"dataset_ref": DATASET_REF, "flat_filename": filename,
                           "sha256": digest, "size": size}
                    new_assets[digest] = {"size": size, "flat_filename": filename,
                                          "dataset_ref": DATASET_REF}
                member["files"].append({"relative_path": relative, "path": (Path(sid)/relative).as_posix(), **ref})
        # Enforce per-checkpoint batch values; dynamic OOM reduction stays active in src.inference.
        record = bench_by_id.get(sid, {})
        batch = record.get("maximum_stable_batch_size")
        if member["model"] not in {"svc", "catboost_stylometry"} and not batch:
            raise ValueError(f"No measured local batch size for {sid}")
        member["config"] = dict(member.get("config", {}))
        if batch: member["config"]["inference_batch_size"] = int(batch)
        members.append(member)

    # The selected E31 roster has no additional base model requirements beyond
    # the pinned base inputs inherited from E13-B. Keep this mapping generic so
    # a future selected checkpoint can explicitly package its own base payload.
    base_models = deployment.get("base_models", {})

    # Assets shared with E28 are copied here so the Kaggle input has one unique
    # E31 marker and one self-contained runtime/grouping bundle.
    group_source = ROOT / "artifacts/kaggle/e28-groupers-v001/deployment"
    (LOCAL / "deployment").mkdir(exist_ok=True)
    shutil.copy2(group_source / "e19_f_c_external_grouper.joblib", LOCAL / "e19_f_c_external_grouper.joblib")
    shutil.copytree(ROOT / "src", LOCAL / "runtime/src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "src/e13_b_inference.py", LOCAL / "e13_b_inference.py")
    shutil.copy2(ROOT / "src/e31_deploy.py", LOCAL / "e31_deploy.py")
    e28_package = json.loads((ROOT / "artifacts/kaggle/e28-groupers-v001/e28_manifest.json").read_text())
    batch_by_cp = {r["checkpoint_id"]: r["maximum_stable_batch_size"]
                   for r in bench["checkpoints"] if r.get("maximum_stable_batch_size")}
    member_manifest = {**old, "ensemble": "E13-B", "members": members,
        "deployment": {**deployment["router_fit"], "checkpoint_ids": deployment["roster"],
            "families": deployment["families"], "classes": deployment["classes"],
            "reliability_statistics": deployment["reliability_statistics"]}}
    member_manifest["members"] = members
    member_manifest["classes"] = deployment["classes"]
    member_manifest["routing_method"] = "family_max"
    member_manifest["deployment"]["reliability_statistics"] = deployment["reliability_statistics"]
    member_manifest["deployment"]["checkpoint_ids"] = deployment["roster"]
    member_manifest["deployment"]["families"] = deployment["families"]
    member_manifest["e31_variant"] = "E31-40"
    member_manifest["base_models"] = base_models
    member_manifest["trust_list"] = {"version": deployment["trust_list_version"]}
    member_manifest["roster"] = deployment["roster"]
    member_manifest["router"] = "e31_router.cbm"
    member_manifest["router_sha256"] = deployment["router_sha256"]
    member_manifest["selection"] = deployment.get("selection", {
        "consensus_additions": deployment.get("selection_consensus", []),
        "selection_rule": deployment.get("selection_rule")})
    member_manifest["selection_rule"] = deployment["selection_rule"]
    member_manifest["baseline_roster"] = deployment.get("baseline", {}).get("roster", [])
    member_manifest["inference_batch_parameters"] = {
        sid: {"measured_effective_batch_size": bench_by_id.get(sid, {}).get("maximum_stable_batch_size"),
              "device": bench_by_id.get(sid, {}).get("device"),
              "peak_allocated_mib": bench_by_id.get(sid, {}).get("peak_allocated_mib"),
              "peak_reserved_mib": bench_by_id.get(sid, {}).get("peak_reserved_mib"),
              "oom_fallback": "enabled"}
        for sid in deployment["roster"]}
    (LOCAL / "e31_manifest.json").write_text(json.dumps(member_manifest, indent=2)+"\n")
    shutil.copy2(ANALYSIS / "e31_router.cbm", LOCAL / "e31_router.cbm")
    e28_package["inference_batch_by_checkpoint"] = batch_by_cp
    e28_package["inference_batch_defaults"] = {m["model"]: min(
        batch_by_cp.get(m["candidate_id"], 1), e28_package.get("inference_batch_defaults", {}).get(m["model"], batch_by_cp.get(m["candidate_id"], 1))) for m in members}
    e28_package["runtime_by_id"] = {r["checkpoint_id"]: r.get("total_checkpoint_seconds")
                                    for r in bench["checkpoints"]}
    e28_package["runtime_by_id"]["__sample_rows__"] = int(bench["sample_rows"])
    e28_package["checkpoint_runtime_by_id"] = e28_package["runtime_by_id"]
    e28_package["grouping_artifact"] = {"dataset_ref": DATASET_REF,
        "flat_filename": "e19_f_c_external_grouper.joblib",
        "size": (LOCAL / "e19_f_c_external_grouper.joblib").stat().st_size}
    from src.e28_deploy import distribute_checkpoints
    balanced_schedule = distribute_checkpoints(members, e28_package["runtime_by_id"],
        e28_package["inference_batch_by_checkpoint"], int(bench["sample_rows"]))
    e28_package["schedule_estimate"] = [{k:v for k,v in worker.items() if k!="checkpoints"}
                                        for worker in balanced_schedule]
    e28_package["experiment"] = "E31"
    e28_package["checkpoint_ids"] = deployment["roster"]
    (LOCAL / "e28_manifest.json").write_text(json.dumps(e28_package, indent=2)+"\n")
    (LOCAL / "dataset-metadata.json").write_text(json.dumps({"title":f"NLM E31-40 {len(members)} Checkpoint Deployment",
        "id":DATASET_REF,"licenses":[{"name":"other"}],
        "subtitle":"E31-40 router and only checkpoint assets not already published",
        "description":f"E31-40 frozen {len(members)}-checkpoint E13-B family-max deployment, with independent and E19-F-A/F-C Hungarian outputs."},indent=2)+"\n")

    notebook_code = _notebook_code()
    notebook_code = notebook_code.replace("__WORKER_CODE_LITERAL__", repr(E28_WORKER_CODE))
    notebook_code = notebook_code.replace("'johnnyhyland/smollm3-3b-base-36l-text':'config.json',", "")
    notebook_code = notebook_code.replace(
        "('qwen06','qwen-3/transformers/0.6b')]",
        "('qwen06','qwen-3/transformers/0.6b'),('qwen17','qwen-3/transformers/1.7b-base')]")
    notebook_code = notebook_code.replace("schedule=distribute_checkpoints(members,runtime_by_id,e28[\"inference_batch_defaults\"],n)",
        "schedule=distribute_checkpoints(members,runtime_by_id,e28[\"inference_batch_defaults\"],max(4000,n))")
    notebook_code = notebook_code.replace("spec.setdefault(\"config\",{})[\"inference_batch_size\"]=int(e28[\"inference_batch_defaults\"].get(m[\"model\"],64))",
        "spec.setdefault(\"config\",{})[\"inference_batch_size\"]=int(e28.get(\"inference_batch_by_checkpoint\",{}).get(m[\"candidate_id\"],e28[\"inference_batch_defaults\"].get(m[\"model\"],32)))")
    notebook_code = notebook_code.replace("BROOT=root_for('e13_b_manifest.json')",
                                          "BROOT=root_for('e31_manifest.json'); E28ROOT=BROOT")
    notebook_code = notebook_code.replace("E28ROOT=root_for('e28_manifest.json')", "E28ROOT=BROOT")
    notebook_code = notebook_code.replace('E28ROOT=root_for("e28_manifest.json")', 'E28ROOT=BROOT')
    notebook_code = notebook_code.replace("B=json.loads((BROOT/'e13_b_manifest.json').read_text())",
                                          "B=json.loads((BROOT/'e31_manifest.json').read_text())")
    notebook_code = notebook_code.replace("'zodrak/e13-b-checkpoints-v001':'e13_b_manifest.json'",
        "'zodrak/e13-b-checkpoints-v001':'e13_b_manifest.json','zodrak/e31-checkpoints-v001':'e31_manifest.json'")
    notebook_code = notebook_code.replace("e13_b_router.cbm", "e31_router.cbm")
    notebook_code = notebook_code.replace("B['ensemble']=='E13-B' and B['routing_method']=='family_max' and len(B['members'])==15",
                                          "B['ensemble']=='E13-B' and B['routing_method']=='family_max' and len(B['members'])==40")
    notebook_code = notebook_code.replace("from e13_b_inference import make_features", "from e13_b_inference import make_features, family_max_scores, route_family_max")
    base_code = '''# Materialize any additional pinned base assets in the E31 manifest.
for model,entry in B.get("base_models",{}).items():
    dest=WORK/"bases"/model; dest.mkdir(parents=True,exist_ok=True)
    for item in entry["files"]:
        source=resolve_asset(item); target=dest/item["relative_path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.symlink_to(source)
    BASES[model]=dest
'''
    notebook_code = notebook_code.replace('members=B["members"]; classes=B["classes"]',
        base_code + '\nmembers=B["members"]; classes=B["classes"]')
    notebook_code = notebook_code.replace('runtime_by_id=dict(e28["runtime_by_id"]); runtime_by_id["__sample_rows__"]=64',
        'runtime_by_id=dict(e28["runtime_by_id"])')
    old_grouping = '''score_a=features[:,0]
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
'''
    new_grouping = '''from src.e31_deploy import cluster_f_candidates, hungarian
groups_a=cluster_f_candidates("E19-F-A",ids,ordered,pairs,features,{"threshold":0.85})
groups_c=cluster_f_candidates("E19-F-C",ids,ordered,pairs,features,grouper)
'''
    if old_grouping not in notebook_code:
        raise ValueError("Could not replace E28's embedded grouping/assignment with shared E31 helpers")
    notebook_code = notebook_code.replace(old_grouping, new_grouping)
    runtime_bootstrap = '''# Prefer the exact E31 runtime. Kaggle may expose the uploaded directory as a ZIP.
E31_RUNTIME_ROOT=E28ROOT/"runtime"
if not (E31_RUNTIME_ROOT/"src"/"e31_deploy.py").is_file():
    archives=[]
    for candidate in E28ROOT.rglob("*.zip"):
        with zipfile.ZipFile(candidate) as zf:
            if any(name.endswith("src/e31_deploy.py") for name in zf.namelist()): archives.append(candidate)
    if not archives: raise FileNotFoundError("E31 runtime directory/ZIP is missing")
    unpack=WORK/"e31_runtime_unpacked"; unpack.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archives[0]) as zf: zf.extractall(unpack)
    matches=list(unpack.rglob("src/e31_deploy.py"))
    if len(matches)!=1: raise FileNotFoundError(f"expected one packaged e31_deploy.py, found {matches[:5]}")
    E31_RUNTIME_ROOT=matches[0].parent.parent
sys.path.insert(0,str(E31_RUNTIME_ROOT))
'''
    import_statement = "\nfrom src.e31_deploy import cluster_f_candidates, hungarian\n"
    if notebook_code.count(import_statement) != 1:
        raise ValueError("Expected one outer E31 grouping import line")
    notebook_code = notebook_code.replace(import_statement,
        "\n" + runtime_bootstrap + "from src.e31_deploy import cluster_f_candidates, hungarian\n", 1)
    start = notebook_code.index("# Family-max score outputs retain the frozen v037 routing decision.")
    end = notebook_code.index("# No checkpoint inference competes with grouping inference.", start)
    notebook_code = notebook_code[:start] + """# Shared E13-B routing implementation used locally and in this notebook.
families=np.asarray([m['family'] for m in members]); family_names=sorted(set(families))
ensemble_probs=np.mean([probs[families==f].mean(0) for f in family_names],axis=0)
ensemble_probs/=ensemble_probs.sum(1,keepdims=True)
hard=probs.argmax(2)
raw_scores,finite_scores=family_max_scores(hard,scores,ensemble_probs,families)
independent=route_family_max(hard,scores,ensemble_probs,families)
assert np.isfinite(finite_scores).all()

""" + notebook_code[end:]
    notebook_code = notebook_code.replace("e28_manifest.json", "e28_manifest.json")
    notebook_code = notebook_code.replace("e28_runtime_root", "e31_runtime_root")
    notebook_code = notebook_code.replace(
        "WORK=Path('/kaggle/working/e31_runtime_root'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True)",
        "WORK=Path('/kaggle/working/e31_runtime_root'); SRC=WORK/'src'; SRC.mkdir(parents=True,exist_ok=True); RUNTIME=WORK")
    notebook_code = notebook_code.replace("e28_probabilities", "e31_probabilities")
    notebook_code = notebook_code.replace("e28_worker_report", "e31_worker_report")
    notebook_code = notebook_code.replace("E28-B", "E31-E19-F-C").replace("E28-C", "E31-E19-F-A")
    notebook_code = notebook_code.replace("'experiment':'E28'", "'experiment':'E31-40'")
    notebook_code = notebook_code.replace("\"experiment\":\"E28\"", "\"experiment\":\"E31-40\"")
    notebook_code = notebook_code.replace("e28_run_manifest.json", "e31_run_manifest.json")
    new_outputs = '''submissions={}
def write_submission(key,filename,pred):
    submission=pd.DataFrame({"id":ids,"model":np.asarray(classes,dtype=object)[pred].astype(str)})
    assert submission.id.tolist()==test.id.astype(str).tolist() and submission.id.is_unique
    assert submission.model.notna().all() and submission.model.isin(classes).all()
    out=Path("/kaggle/working")/filename
    submission.to_parquet(out,index=False)
    reread=pd.read_parquet(out)
    assert reread.equals(submission) and list(reread.columns)==["id","model"]
    submissions[key]={"path":str(out),"rows":len(submission),
        "submission_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "changed_from_independent":int(np.count_nonzero(pred!=independent))}
write_submission("independent_E13-B","submission.parquet",independent)
pred_a=hungarian(finite_scores,groups_a,ids)
write_submission("E19-F-A","submission_e19_f_a.parquet",pred_a)
pred_c=hungarian(finite_scores,groups_c,ids)
write_submission("E19-F-C","submission_e19_f_c.parquet",pred_c)'''
    output_start = notebook_code.index("submissions={}")
    output_end = notebook_code.index("run={", output_start)
    notebook_code = notebook_code[:output_start] + new_outputs + "\n" + notebook_code[output_end:]
    notebook_code = notebook_code.replace("spec.setdefault('config',{})['inference_batch_size']=int(e28['inference_batch_defaults'].get(m['model'],64))",
        "spec.setdefault('config',{})['inference_batch_size']=int(e28['inference_batch_by_checkpoint'].get(m['candidate_id'],e28['inference_batch_defaults'].get(m['model'],32)))")
    model_sources = []
    model_map = {
        "embeddinggemma2": "google/embeddinggemma-2/Transformers/embeddinggemma-2/1",
        "ministral3_3b": "mistral-ai/ministral-3/Transformers/ministral-3-3b-base-2512/1",
        "qwen06": "qwen-lm/qwen-3/Transformers/0.6b/1",
        "qwen17": "qwen-lm/qwen-3/Transformers/1.7b-base/1",
    }
    for model in sorted({m["model"] for m in members}):
        if model in model_map:
            model_sources.append(model_map[model])
    nbf.write(nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(f"# E31-40 — {len(members)}-checkpoint E13-B family-max + E19-F grouping\n\nTwo GPU inference workers, frozen deployment artifacts and shared local/Kaggle E13-B scoring functions. Produces the primary independent E13-B submission plus E19-F-A and E19-F-C Hungarian variants."),
        nbf.v4.new_code_cell(notebook_code)],
        metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}), KERNEL / "e31.ipynb")
    source_refs = sorted({DATASET_REF, *[f["dataset_ref"] for m in members for f in m["files"]],
                          "zodrak/e13-b-checkpoints-v001", "zodrak/lc3-lc4-checkpoints-v035"})
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({"id":KERNEL_REF,"title":f"E31-40 Checkpoint E13-B with External Hungarian",
        "code_file":"e31.ipynb","language":"python","kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "keywords":["name-that-llm","E31","Hungarian"],"dataset_sources":source_refs,
        "kernel_sources":[],"competition_sources":["name-that-llm"],
        "model_sources":model_sources,"machine_shape":"NvidiaTeslaT4"},indent=2)+"\n")
    safe_json_calls = embed_runtime_preflight(
        KERNEL, manifest_paths=[LOCAL / "e31_manifest.json", LOCAL / "e28_manifest.json"],
        sklearn_compatibility_assets=[{"dataset_ref": DATASET_REF,
                                      "filename": "e19_f_c_external_grouper.joblib"}])
    package_preflight = validate_kernel_package(KERNEL)
    return {"dataset_dir":str(LOCAL),"notebook_dir":str(KERNEL),"dataset_ref":DATASET_REF,
        "kernel_ref":KERNEL_REF,"roster":deployment["roster"],"new_files":len(new_assets),
        "new_bytes":sum(x["size"] for x in new_assets.values()),"data_sources":source_refs,
        "preflight":package_preflight,"runtime_json_calls_wrapped":safe_json_calls}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
