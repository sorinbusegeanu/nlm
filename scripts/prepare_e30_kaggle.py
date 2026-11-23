"""Build a local E30-B two-GPU Kaggle dataset delta and notebook; never submits."""
from __future__ import annotations

import csv, hashlib, json, os, shutil
from pathlib import Path

import nbformat as nbf
from huggingface_hub import snapshot_download

from scripts.publish_kaggle_checkpoint_store import ROOT, build_checkpoint, sha256_file
from scripts.prepare_e12_2_kaggle import _asset_ref, _remote_assets
from scripts.prepare_e28_kaggle import _notebook_code

LOCAL = ROOT / "artifacts/kaggle/e30-checkpoints-v001"
KERNEL = ROOT / "artifacts/kaggle/e30-kernel"
ANALYSIS = ROOT / "artifacts/analysis/e30"
TRUST = ROOT / "docs/trust-list-2-v037.csv"
E28_MANIFEST = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
DATASET_REF = "zodrak/e30-checkpoints-v001"
KERNEL_REF = "zodrak/e30-e13-b-family-max-hungarian"


def prepare():
    deployment = json.loads((ANALYSIS / "e30_b_manifest.json").read_text())
    bench = json.loads((ANALYSIS / "inference_benchmark.json").read_text())
    bench_by_id = {r["checkpoint_id"]: r for r in bench["checkpoints"]}
    if len(deployment["roster"]) != 20:
        raise ValueError("E30 deployment must contain exactly 20 checkpoints")
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
            local = build_checkpoint(exp)
            pred = json.loads((ROOT / "artifacts/predictions" / f"{exp}_manifest.json").read_text())
            cp_path = Path(row["checkpoint_path"])
            if not cp_path.is_absolute(): cp_path = ROOT / cp_path
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

    # MiniCPM and Hunyuan bases have no Kaggle model source in the existing
    # deployment inventory. Package these pinned base weights in the E30 delta.
    base_models = {}
    allowed = {"config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
        "spiece.model", "vocab.json", "vocab.txt", "merges.txt", "added_tokens.json",
        "generation_config.json", "chat_template.jinja"}
    for model, sid in (("minicpm5_2b", "N0130"), ("hunyuan_1_8b", "N0129")):
        member = next(x for x in members if x["candidate_id"] == sid)
        repo = member["config"].get("checkpoint")
        revision = member.get("model_revision") or member["config"].get("revision")
        base_dir = Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=True))
        files = []
        for source in sorted(base_dir.rglob("*")):
            if not source.is_file():
                continue
            if source.name not in allowed and not source.name.endswith((".safetensors", ".bin", ".py")):
                continue
            relative = source.relative_to(base_dir).as_posix()
            digest, size = sha256_file(source), source.stat().st_size
            ref = _asset_ref(digest, size, by_hash, inventory)
            if ref is None:
                filename = f"assets/{model}__{digest}"
                target = LOCAL / filename; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                ref = {"dataset_ref": DATASET_REF, "flat_filename": filename,
                       "sha256": digest, "size": size}
                new_assets[digest] = {"size": size, "flat_filename": filename,
                                      "dataset_ref": DATASET_REF}
            files.append({"relative_path": relative, **ref})
        if not any(Path(item["relative_path"]).name == "config.json" for item in files):
            raise FileNotFoundError(f"Pinned {model} base has no config.json")
        base_models[model] = {"repo_id": repo, "revision": revision, "files": files}

    # Assets shared with E28 are copied here so the Kaggle input has one unique
    # E30 marker and one self-contained runtime/grouping bundle.
    group_source = ROOT / "artifacts/kaggle/e28-groupers-v001/deployment"
    (LOCAL / "deployment").mkdir(exist_ok=True)
    shutil.copy2(group_source / "e19_f_c_external_grouper.joblib", LOCAL / "deployment/e19_f_c_external_grouper.joblib")
    shutil.copy2(group_source / "e19_f_c_external_grouper.joblib", LOCAL / "e19_f_c_external_grouper.joblib")
    shutil.copytree(ROOT / "src", LOCAL / "runtime/src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "src/e13_b_inference.py", LOCAL / "e13_b_inference.py")
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
    member_manifest["e30_variant"] = "E30-B"
    member_manifest["base_models"] = base_models
    (LOCAL / "e30_b_manifest.json").write_text(json.dumps(member_manifest, indent=2)+"\n")
    shutil.copy2(ANALYSIS / "e30_b_router.cbm", LOCAL / "e30_b_router.cbm")
    e28_package["inference_batch_by_checkpoint"] = batch_by_cp
    e28_package["inference_batch_defaults"] = {m["model"]: min(
        batch_by_cp.get(m["candidate_id"], 1), e28_package.get("inference_batch_defaults", {}).get(m["model"], batch_by_cp.get(m["candidate_id"], 1))) for m in members}
    e28_package["runtime_by_id"] = {r["checkpoint_id"]: r.get("total_checkpoint_seconds")
                                    for r in bench["checkpoints"]}
    e28_package["runtime_by_id"]["__sample_rows__"] = int(bench["sample_rows"])
    e28_package["checkpoint_runtime_by_id"] = e28_package["runtime_by_id"]
    from src.e28_deploy import distribute_checkpoints
    balanced_schedule = distribute_checkpoints(members, e28_package["runtime_by_id"],
        e28_package["inference_batch_by_checkpoint"], int(bench["sample_rows"]))
    e28_package["schedule_estimate"] = [{k:v for k,v in worker.items() if k!="checkpoints"}
                                        for worker in balanced_schedule]
    e28_package["experiment"] = "E30"
    e28_package["checkpoint_ids"] = deployment["roster"]
    (LOCAL / "e28_manifest.json").write_text(json.dumps(e28_package, indent=2)+"\n")
    (LOCAL / "dataset-metadata.json").write_text(json.dumps({"title":"NLM E30-B 20 Checkpoint Deployment",
        "id":DATASET_REF,"licenses":[{"name":"other"}],
        "subtitle":"E30-B router and only checkpoint assets not already published",
        "description":"E30-B frozen 20-checkpoint E13-B family-max deployment, grouped with E19-F-A/F-C and Hungarian assignment."},indent=2)+"\n")

    notebook_code = _notebook_code()
    notebook_code = notebook_code.replace("'johnnyhyland/smollm3-3b-base-36l-text':'config.json',", "")
    notebook_code = notebook_code.replace("schedule=distribute_checkpoints(members,runtime_by_id,e28[\"inference_batch_defaults\"],n)",
        "schedule=distribute_checkpoints(members,runtime_by_id,e28[\"inference_batch_defaults\"],max(4000,n))")
    notebook_code = notebook_code.replace("spec.setdefault(\"config\",{})[\"inference_batch_size\"]=int(e28[\"inference_batch_defaults\"].get(m[\"model\"],64))",
        "spec.setdefault(\"config\",{})[\"inference_batch_size\"]=int(e28.get(\"inference_batch_by_checkpoint\",{}).get(m[\"candidate_id\"],e28[\"inference_batch_defaults\"].get(m[\"model\"],32)))")
    notebook_code = notebook_code.replace("BROOT=root_for('e13_b_manifest.json')",
                                          "BROOT=root_for('e30_b_manifest.json')")
    notebook_code = notebook_code.replace("E28ROOT=root_for('e28_manifest.json')", "E28ROOT=BROOT")
    notebook_code = notebook_code.replace('E28ROOT=root_for("e28_manifest.json")', 'E28ROOT=BROOT')
    notebook_code = notebook_code.replace("B=json.loads((BROOT/'e13_b_manifest.json').read_text())",
                                          "B=json.loads((BROOT/'e30_b_manifest.json').read_text())")
    notebook_code = notebook_code.replace("'zodrak/e13-b-checkpoints-v001':'e13_b_manifest.json'",
        "'zodrak/e13-b-checkpoints-v001':'e13_b_manifest.json','zodrak/e30-checkpoints-v001':'e30_b_manifest.json','zodrak/zod03-llama-base':'llama_base_manifest.json'")
    notebook_code = notebook_code.replace("e13_b_router.cbm", "e30_b_router.cbm")
    notebook_code = notebook_code.replace("B['ensemble']=='E13-B' and B['routing_method']=='family_max' and len(B['members'])==15",
                                          "B['ensemble']=='E13-B' and B['routing_method']=='family_max' and len(B['members'])==20")
    notebook_code = notebook_code.replace("from e13_b_inference import make_features", "from e13_b_inference import make_features, family_max_scores, route_family_max")
    base_code = '''# Resolve all additional E30 base models from pinned packaged assets.
for model,entry in B.get("base_models",{}).items():
    dest=WORK/"bases"/model; dest.mkdir(parents=True,exist_ok=True)
    for item in entry["files"]:
        source=resolve_asset(item); target=dest/item["relative_path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.symlink_to(source)
    BASES[model]=dest
llama_manifest_path=root_for("llama_base_manifest.json")/"llama_base_manifest.json"
llama_manifest=json.loads(llama_manifest_path.read_text())
llama_dir=WORK/"bases"/"llama32_3b"; llama_dir.mkdir(parents=True,exist_ok=True)
for item in llama_manifest["files"]:
    source=llama_manifest_path.parent/item["flat_filename"]
    target=llama_dir/item["relative_path"]; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.symlink_to(source)
BASES["llama32_3b"]=llama_dir
smollm_paths=[p.parent for p in INPUT.rglob("config.json") if "smollm3-3b-base" in str(p.parent).casefold()]
assert smollm_paths, "SmolLM3 base dataset is missing"
smollm_source=only(smollm_paths,"SmolLM3 base")
smollm_dir=WORK/"bases"/"smollm3_3b"; smollm_dir.mkdir(parents=True,exist_ok=True)
for source in smollm_source.rglob("*"):
    if not source.is_file(): continue
    target=smollm_dir/source.relative_to(smollm_source); target.parent.mkdir(parents=True,exist_ok=True)
    if source.name=="config.json":
        cfg=json.loads(source.read_text())
        if cfg.get("model_type")=="smollm3_text_base": cfg["model_type"]="smollm3"
        target.write_text(json.dumps(cfg)+"\\n")
    elif not target.exists(): target.symlink_to(source)
index_files=list(INPUT.rglob("smollm3_base_model.safetensors.index.json"))
if index_files:
    target=smollm_dir/"model.safetensors.index.json"
    if not target.exists(): target.symlink_to(only(index_files,"SmolLM3 index"))
BASES["smollm3_3b"]=smollm_dir
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
    new_grouping = '''from src.e30_deploy import cluster_f_candidates, hungarian
groups_a=cluster_f_candidates("E19-F-A",ids,ordered,pairs,features,{"threshold":0.85})
groups_c=cluster_f_candidates("E19-F-C",ids,ordered,pairs,features,grouper)
'''
    if old_grouping not in notebook_code:
        raise ValueError("Could not replace E28's embedded grouping/assignment with shared E30 helpers")
    notebook_code = notebook_code.replace(old_grouping, new_grouping)
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
    notebook_code = notebook_code.replace("e28_runtime_root", "e30_runtime_root")
    notebook_code = notebook_code.replace("e28_probabilities", "e30_probabilities")
    notebook_code = notebook_code.replace("e28_worker_report", "e30_worker_report")
    notebook_code = notebook_code.replace("E28-B", "E30-E19-F-C").replace("E28-C", "E30-E19-F-A")
    notebook_code = notebook_code.replace("'experiment':'E28'", "'experiment':'E30-B'")
    notebook_code = notebook_code.replace("\"experiment\":\"E28\"", "\"experiment\":\"E30-B\"")
    notebook_code = notebook_code.replace("e28_run_manifest.json", "e30_run_manifest.json")
    notebook_code = notebook_code.replace('f"submission_{code.lower().replace(\'-\',\'_\')}.parquet"',
        'f"submission_{code.lower().replace(\'-\',\'_\')}.csv"')
    notebook_code = notebook_code.replace("submission.to_parquet(out,index=False)", "submission.to_csv(out,index=False)")
    notebook_code = notebook_code.replace("assert pd.read_parquet(out).equals(submission)",
        "check=pd.read_csv(out,dtype={\"id\":\"string\"}); assert check.equals(submission)")
    notebook_code = notebook_code.replace("spec.setdefault('config',{})['inference_batch_size']=int(e28['inference_batch_defaults'].get(m['model'],64))",
        "spec.setdefault('config',{})['inference_batch_size']=int(e28['inference_batch_by_checkpoint'].get(m['candidate_id'],e28['inference_batch_defaults'].get(m['model'],32)))")
    nbf.write(nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# E30-B — 20-checkpoint E13-B family-max + E19-F grouping\n\nTwo GPU inference workers, frozen deployment artifacts and the shared E13-B score/routing functions. Writes E19-F-C and E19-F-A Hungarian submission files; this notebook is prepared only and is not submitted or run."),
        nbf.v4.new_code_cell(notebook_code)],
        metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}), KERNEL / "e30.ipynb")
    source_refs = sorted({DATASET_REF, *[f["dataset_ref"] for m in members for f in m["files"]],
                          "zodrak/lc3-lc4-checkpoints-v035", "zodrak/zod03-llama-base",
                          "johnnyhyland/smollm3-3b-base-36l-text"})
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({"id":KERNEL_REF,"title":"E30-B 20 Checkpoint E13-B with External Hungarian",
        "code_file":"e30.ipynb","language":"python","kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "keywords":["name-that-llm","E30","Hungarian"],"dataset_sources":source_refs,
        "kernel_sources":[],"competition_sources":["name-that-llm"],
        "model_sources":["google/embeddinggemma-2/Transformers/embeddinggemma-2/1",
            "mistral-ai/ministral-3/Transformers/ministral-3-3b-base-2512/1",
            "qwen-lm/qwen-3/Transformers/0.6b/1"],"machine_shape":"NvidiaTeslaT4"},indent=2)+"\n")
    return {"dataset_dir":str(LOCAL),"notebook_dir":str(KERNEL),"dataset_ref":DATASET_REF,
        "kernel_ref":KERNEL_REF,"roster":deployment["roster"],"new_files":len(new_assets),
        "new_bytes":sum(x["size"] for x in new_assets.values()),"data_sources":source_refs}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
