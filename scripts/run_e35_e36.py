"""Select v044 E35/E36 rosters, fit frozen routers, and run E33/E34 variants."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_e13_b_kaggle import _deployment
from scripts.run_e30_corrected_selection import route
from scripts.run_e31 import _fold_definitions
from scripts.run_e32 import _e13_iterations
from src import e33, e34, e13_reliability_router as e13, ensemble_catalog as catalog
from src.e31_selection import E31GreedySelector
from src.e32_selection import E32GreedySelector
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
E31 = ROOT / "artifacts/analysis/e31"
E32 = ROOT / "artifacts/analysis/e32"
E33 = ROOT / "artifacts/analysis/e33"
E34 = ROOT / "artifacts/analysis/e34"
METHODS = e33.METHODS
V044 = "trust-list-2-v044"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def update_catalogue(eid, status, *, result=None, error=None, stage=None):
    path = ROOT / "configs/ensemble_catalog_v2.json"
    doc = json.loads(path.read_text())
    entry = next(x for x in doc["experiments"] if x["id"] == eid)
    entry["status"] = status
    entry["current_stage"] = stage or status
    entry["progress_source"] = f"artifacts/analysis/{eid.lower()}/metrics.json"
    if error:
        entry["error"] = error
    if result is not None:
        entry["status"] = "Scored"
        entry["current_stage"] = "Complete"
        entry["scenario_results"] = {
            experiment: {method: value["mean_balanced_accuracy"]
                         for method, value in methods.items()}
            for experiment, methods in result["summary"].items()}
        entry["result_source"] = f"artifacts/analysis/{eid.lower()}/metrics.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def _seed_cache_links(source: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    added = 0
    for path in source.glob("*.npz"):
        target = dest / path.name
        if not target.exists():
            target.symlink_to(path.resolve())
            added += 1
    return added


def _fit_router(eid, roster, v0, y, ids, members, probabilities, outdir):
    router, fit = _deployment(v0, ids, members, probabilities,
                              [ids.index(sid) for sid in roster])
    router_path = outdir / f"{eid.lower()}_router.cbm"
    router.save_model(str(router_path))
    tensor = np.stack([probabilities[sid] for sid in roster]).astype(np.float64)
    stats = e13.reliability_statistics(tensor.argmax(axis=2), y)
    stats_json = {key: val.tolist() for key, val in stats.items()}
    pred, scores = route(v0, roster, members, probabilities, router_path, stats_json)
    if scores.shape != (len(v0), len(LABELS)) or not np.array_equal(scores.argmax(1), pred):
        raise ValueError(f"{eid} router predictions and score matrix are inconsistent")
    return router_path, fit, stats_json, pred, scores


def _select_rosters(v0, members, probabilities, snapshot):
    out35 = ROOT / "artifacts/analysis/e35"
    out36 = ROOT / "artifacts/analysis/e36"
    out35.mkdir(parents=True, exist_ok=True); out36.mkdir(parents=True, exist_ok=True)
    # Existing E31/E32 OOF caches are valid where a candidate roster matches:
    # overlapping checkpoint cache hashes were verified equal between v037/v044.
    n31 = _seed_cache_links(E31 / "selection/candidate_oof_cache",
                            out35 / "selection/candidate_oof_cache")
    n32 = _seed_cache_links(E32 / "selection/candidate_oof_cache",
                            out36 / "selection/candidate_oof_cache")
    y = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    ids = sorted(probabilities, key=lambda sid: int(sid[1:]))
    definitions, _ = _fold_definitions(v0)
    iterations = _e13_iterations()
    start40 = list(map(str, json.loads((E32 / "e32_manifest.json").read_text())["roster"][:17]))
    # The E31 recipe starts from the exact 17-member E30-B roster.
    e30 = json.loads((ROOT / "artifacts/analysis/e30/e30_b_manifest.json").read_text())
    start40 = list(map(str, e30["roster"]))
    if len(start40) != 17 or set(start40) - set(probabilities):
        raise ValueError("The original E31 17-checkpoint starting roster is not present in v044")
    selector35 = E31GreedySelector(v0, members, probabilities, start40,
        out35 / "selection", outer_folds=5, inner_folds=4, steps=23,
        iterations=iterations, candidate_workers=3)
    selection35 = selector35.run(definitions)
    selection35.update({"trust_list_version": V044,
        "checkpoint_pool_count": len(probabilities),
        "outer_fold_definitions": [{"fold": i+1, "fit_rows": len(fit), "held_rows": len(held),
            "fit_prompt_groups": len(set(v0.prompt_id.iloc[fit].astype(str))),
            "held_prompt_groups": len(set(v0.prompt_id.iloc[held].astype(str)))}
            for i,(fit,held) in enumerate(definitions)],
        "checkpoint_cache_sha256": {sid: members[sid]["cache_sha256"] for sid in ids},
        "selection_reused_candidate_cache_files": n31,
        "selection_configuration": {"starting_roster": 17,"target_roster": 40,
            "steps": 23,"outer_folds":5,"inner_folds":4,"router_iterations":iterations,
            "selection_method":"E31GreedySelector: fold-local greedy forward selection by inner OOF BA; consensus ranking across five outer folds"}})
    (out35 / "selection_results.json").write_text(json.dumps(selection35,indent=2)+"\n")
    roster40 = list(map(str, selection35["final_roster"]))
    if len(roster40)!=40 or len(set(roster40))!=40 or set(roster40)-set(probabilities):
        raise ValueError("E35 selected roster is not a valid 40-checkpoint set")
    update_catalogue("E35","Running",stage="E35 roster selected; fitting router and scoring variants")
    router40, fit40, stats40, pred40, scores40 = _fit_router(
        "e35",roster40,v0,y,ids,members,probabilities,out35)
    # E32 extends its supplied 40-member baseline by 20 more fold-selected IDs.
    selector36 = E32GreedySelector(v0,members,probabilities,roster40,
        out36/"selection",outer_folds=5,inner_folds=4,steps=20,iterations=iterations,
        candidate_workers=3)
    selection36 = selector36.run(definitions)
    roster60 = list(map(str,selection36["final_roster"]))
    if len(roster60)!=60 or roster60[:40]!=roster40 or len(set(roster60))!=60:
        raise ValueError("E36 roster does not preserve E35's 40 and append 20 unique checkpoints")
    selection36.update({"trust_list_version":V044,"checkpoint_pool_count":len(probabilities),
        "checkpoint_cache_sha256":{sid:members[sid]["cache_sha256"] for sid in roster60},
        "selection_reused_candidate_cache_files":n32,
        "selection_configuration":{"starting_roster":40,"target_roster":60,"steps":20,
            "outer_folds":5,"inner_folds":4,"router_iterations":iterations,
            "selection_method":"E32GreedySelector: E31 nested prompt-group selection with 20 fixed additions and fold-consensus deployment roster"}})
    (out36/"selection_results.json").write_text(json.dumps(selection36,indent=2)+"\n")
    update_catalogue("E36","Running",stage="E36 roster selected; fitting router and scoring variants")
    router60,fit60,stats60,pred60,scores60=_fit_router(
        "e36",roster60,v0,y,ids,members,probabilities,out36)
    deployment35={"experiment":"E35","trust_list_version":V044,"roster":roster40,
        "families":[_model_family(members[s]["model"]) for s in roster40],
        "classes":list(LABELS),"router":"e35_router.cbm","router_sha256":sha(router40),
        "router_fit":fit40,"reliability_statistics":stats40,
        "checkpoint_cache_sha256":{s:members[s]["cache_sha256"] for s in roster40},
        "selection":"selection_results.json","validation":"full V0 (2,699 rows)"}
    deployment36={"experiment":"E36","trust_list_version":V044,"roster":roster60,
        "families":[_model_family(members[s]["model"]) for s in roster60],
        "classes":list(LABELS),"router":"e36_router.cbm","router_sha256":sha(router60),
        "router_fit":fit60,"reliability_statistics":stats60,
        "checkpoint_cache_sha256":{s:members[s]["cache_sha256"] for s in roster60},
        "selection":"selection_results.json","validation":"full V0 (2,699 rows)"}
    (out35/"deployment_manifest.json").write_text(json.dumps(deployment35,indent=2)+"\n")
    (out36/"deployment_manifest.json").write_text(json.dumps(deployment36,indent=2)+"\n")
    return (roster40,router40,stats40,pred40,scores40,selection35,
            roster60,router60,stats60,pred60,scores60,selection36)


def _scenario_inputs(ids):
    scenarios=pd.read_parquet(ROOT/"data/processed/dataset_v0_group.parquet")
    saved=pd.read_parquet(E31/"roster_scenario_evaluation/scenario_predictions.parquet")
    saved=saved.loc[saved.checkpoint_count.eq(40)]
    group_map={}
    for method in METHODS[1:]:
        block=saved.loc[saved.configuration.eq(method)]
        if len(block)!=25000 or block.scenario_row_id.astype(str).duplicated().any():
            raise ValueError(f"E33/{method} cached groups are not complete")
        group_map[method]=block.set_index(block.scenario_row_id.astype(str)).predicted_group.astype(str).to_dict()
    frames=[]
    for sid,frame in scenarios.groupby("scenario_id",sort=True):
        frame=frame.reset_index(drop=True).copy()
        if len(frame)!=500 or frame.original_id.astype(str).duplicated().any():
            raise ValueError(f"Invalid v0_group scenario {sid}")
        for method in METHODS[1:]: frame[f"group::{method}"]=[group_map[method][str(x)] for x in frame.scenario_row_id]
        frames.append((str(sid),frame))
    if len(frames)!=50: raise ValueError("Expected 50 fixed v0_group scenarios")
    return frames,saved


def _run_variants(eid,roster,base_scores,base_pred,members,probabilities,ids,prompts,y,trust_snapshot):
    out=ROOT/f"artifacts/analysis/{eid.lower()}";out.mkdir(parents=True,exist_ok=True)
    families=[_model_family(members[s]["model"]) for s in roster]
    tensor=np.stack([probabilities[s] for s in roster]).astype(np.float32)
    rank=e33._rank_positions(tensor);e33._BUILD_CP_IDS=list(roster);e33._SCENARIO_TRUTH=y
    folds=e33._folds_from_e31(ids,prompts)
    scores={eid+" baseline":base_scores.copy(),**{f"{eid}-{letter}":np.zeros_like(base_scores) for letter in "ABCDEF"}}
    exclusions={f"{eid}-{letter}":np.zeros_like(base_scores,dtype=bool) for letter in "ABC"}
    audits=[];counts=[];f_meta=[]
    for fn,(fit,held) in enumerate(folds,1):
        st=e33._fit_rank_stats(rank,y,fit,baseline_pred=base_pred);rules=e33._negative_rules(st,families)
        counts.append({"fold":fn,"A_rules":int(rules["single"].sum()),"B_consensus":int(rules["consensus"].sum()),"C_conditional":int(rules["conditional"].sum())})
        for name,src in zip((f"{eid}-A",f"{eid}-B",f"{eid}-C"),("E33-A","E33-B","E33-C")):
            exclusions[name][held]=e33._candidate_exclusions(rank,held,st,rules,families,base_pred,src)
            scores[name][held]=base_scores[held]
        scores[f"{eid}-D"][held]=base_scores[held]*e33._weighted_factors(rank,held,st,families,combined=False)
        scores[f"{eid}-E"][held]=base_scores[held]*e33._weighted_factors(rank,held,st,families,combined=True)
        for cp,sid in enumerate(roster):
            for label in range(len(LABELS)):
                for r in range(1,4):
                    sup=int(st["support"][cp,r-1,label])
                    if sup:
                        audits.append({"fold":fn,"checkpoint_id":sid,"family":families[cp],"rank":r,
                            "label":LABELS[label],"support":sup,"correct":int(st["correct"][cp,r-1,label]),
                            "precision":float(st["precision"][cp,r-1,label]),"prior":float(st["prior"][label]),
                            "negative_rule":bool(rules["single"][cp,r-1,label]),
                            "conditional_support":int(st["conditional_support"][cp,r-1,label]),
                            "conditional_precision":float(st["conditional_precision"][cp,r-1,label]),
                            "conditional_rule":bool(rules["conditional"][cp,r-1,label])})
        fitted,meta=e33._fit_e33_f(fit,held,ids=ids,prompts=prompts,y=y,
            probabilities=tensor,rank_position=rank,base_scores=base_scores,families=families,
            baseline_pred=base_pred,fold_number=fn,output_dir=out)
        old=out/f"e33_f_fold{fn}.cbm"; target=out/f"{eid.lower()}_f_fold{fn}.cbm"
        if old.exists():old.replace(target)
        scores[f"{eid}-F"][held]=fitted;meta["model_path"]=str(target.relative_to(ROOT));f_meta.append(meta)
        print(f"{eid} outer fold {fn}/5 complete",flush=True)
    if any(not np.isfinite(x).all() for x in scores.values()):raise ValueError(f"{eid} scores incomplete")
    frames,_=_scenario_inputs(ids);lookup={x:i for i,x in enumerate(ids)}
    summary={};records=[]
    for sid,frame in frames:
        rows=np.asarray([lookup[str(x)] for x in frame.original_id],dtype=np.int64)
        true=pd.Categorical(frame.model.astype(str),categories=LABELS).codes.astype(np.int64)
        base_res,base_records=e33._evaluate_scenario_scores(frame,lookup,base_scores,None,base_pred)
        base_pred_by_method={method:[] for method in METHODS}
        for rec in base_records:
            base_pred_by_method[rec["method"]].append(rec["prediction"])
        for method in METHODS:
            val=base_res[method]; summary.setdefault(eid+" baseline",{}).setdefault(method,[]).append(val["balanced_accuracy"])
            records.extend({"scenario_id":sid,"scenario_row_id":str(frame.scenario_row_id.iloc[i]),"original_id":str(frame.original_id.iloc[i]),
                "truth":LABELS[true[i]],"prediction":base_pred_by_method[method][i],"experiment":eid+" baseline","method":method} for i in range(len(frame)))
        for letter in "ABCDEF":
            name=f"{eid}-{letter}";res,block=e33._evaluate_scenario_scores(frame,lookup,scores[name],exclusions.get(name),base_pred)
            for method in METHODS:summary.setdefault(name,{}).setdefault(method,[]).append(res[method]["balanced_accuracy"])
            for rec in block:rec["experiment"]=name;records.append(rec)
    summary_out={}
    for name,by_method in summary.items():
        summary_out[name]={}
        for method,vals in by_method.items():
            a=np.asarray(vals,dtype=float);summary_out[name][method]={"mean_balanced_accuracy":float(a.mean()),
                "std_balanced_accuracy":float(a.std()),"min_balanced_accuracy":float(a.min()),
                "p10_balanced_accuracy":float(np.quantile(a,.1)),"scenario_count":len(a),"scenario_values":a.tolist()}
    np.savez_compressed(out/"oof_adjusted_scores.npz",**{k.replace(" ","_").replace("-","_"):v for k,v in scores.items()},
        **{f"exclusion_{k.replace('-','_')}":v for k,v in exclusions.items()},row_ids=ids,classes=np.asarray(LABELS))
    pd.DataFrame(audits).to_parquet(out/"fold_rule_statistics.parquet",index=False)
    pd.DataFrame(counts).to_csv(out/"fold_rule_counts.csv",index=False)
    pd.DataFrame(records).to_parquet(out/"scenario_predictions.parquet",index=False)
    final=e34._fit_final_deployment(tensor,rank,base_scores,y,prompts,families,roster,out,tag=eid.lower())
    dep_path=out/"deployment_manifest.json";dep=json.loads(dep_path.read_text())
    dep.update({"experiment":eid,"trust_list_version":trust_snapshot["version"],"roster":roster,
        "families":families,"classes":list(LABELS),"baseline_router":f"artifacts/analysis/{eid.lower()}/{eid.lower()}_router.cbm",
        "candidate_classifier":final["model_path"],"candidate_classifier_sha256":final["model_sha256"],
        "reliability_statistics":final["reliability_path"],
        "feature_definition":{"checkpoint_count":len(roster),"per_checkpoint_features":4,
            "shared_features":6,"total_features":4*len(roster)+6},
        "crossfit":{"outer_folds":5,"inner_folds":4,"iterations":400,"depth":6,"learning_rate":.05,"thread_count":8},
        "baseline_router_sha256":sha(out/f"{eid.lower()}_router.cbm"),
        "deployment_note":"Router and meta-classifier were fit using V0 development data; scenario metrics reuse V0."})
    dep_path.write_text(json.dumps(dep,indent=2)+"\n")
    result={"experiment":eid,"status":"Scored","trust_list_version":V044,"roster":roster,"families":families,
        "checkpoint_count":len(roster),"scenario_count":50,"rows_per_scenario":500,"methods":list(METHODS),
        "summary":summary_out,"fold_rule_counts":counts,"learned_classifier_folds":f_meta,
        "final_deployment_classifier":final,"rule_thresholds":e33_thresholds(),
        "baseline_router_sha256":sha(out/f"{eid.lower()}_router.cbm"),
        "inputs":{"trust_list_sha256":trust_snapshot["sha256"],"v044_manifest_sha256":sha(ROOT/"docs/trust-list-2-v044.json"),
            "v0_group_sha256":sha(ROOT/"data/processed/dataset_v0_group.parquet"),
            "group_cache_sha256":sha(E31/"roster_scenario_evaluation/scenario_predictions.parquet")},
        "notes":["The fixed roster was selected using the E31/E32 prompt-group greedy procedure from the full v044 eligible pool.",
            "Reliability statistics and learned candidate-error models use the same E33/E34 prompt-disjoint cross-fitting, thresholds and CatBoost settings.",
            "All rows use the same 50 v0_group scenario memberships and cached E19-F-A/E19-F-C/E19-D group assignments as E33/E34.",
            "The scenarios reuse V0 responses and are correlated robustness diagnostics, not independent validation."],
        "artifacts":["selection_results.json","deployment_manifest.json","oof_adjusted_scores.npz","scenario_predictions.parquet",
            "fold_rule_statistics.parquet","fold_rule_counts.csv",f"{eid.lower()}_f_fold1.cbm through {eid.lower()}_f_fold5.cbm",
            f"{eid.lower()}_final.cbm",f"{eid.lower()}_final_reliability.npz"]}
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    _write_local_report(eid,result)
    return result


def e33_thresholds():
    return {"A_B":{"support":30,"wrong_rate":.97,"precision_below_class_prior":True},
        "C":{"conditional_support":20,"conditional_wrong_rate":.95},"D":{"support":30,"precision_below_class_prior":True},
        "E":{"support":30,"prior_smoothing_strength":30},"F":{"inner_group_folds":4,"iterations":400,"depth":6,"learning_rate":.05}}


def _write_local_report(eid,result):
    names=[eid+" baseline",*(f"{eid}-{x}" for x in "ABCDEF")]
    data=result["summary"]
    values=np.asarray([[100*data[x][m]["mean_balanced_accuracy"] for m in METHODS] for x in names])
    maxima=values.max(axis=0);lines=[f"# {eid} — Trust List 2 v044 negative evidence", "",
        f"Fixed roster selected from the 94 eligible checkpoints in Trust List 2 v044. The baseline uses a new deployment router fitted to this roster; variants apply E33's thresholds, family aggregation and prompt-group cross-fitting.","",
        "| Experiment | Checkpoints | Independent | E19-F-A | E19-F-C | E19-D |","|---|---:|---:|---:|---:|---:|"]
    for i,name in enumerate(names):
        cells=[f"**{values[i,j]:.4f}%**" if np.isclose(values[i,j],maxima[j],rtol=0,atol=1e-12) else f"{values[i,j]:.4f}%" for j in range(4)]
        lines.append(f"| {name} | {result['checkpoint_count']} | "+" | ".join(cells)+" |")
    lines += ["", "All values are mean balanced accuracy over 50 fixed 500-row scenarios. Each scenario reuses V0 responses; treat these as correlated development diagnostics.",
        "", "A/B use support ≥30, wrong rate ≥97%, and precision below the training-fold class prior. C uses conditional support ≥20 and conditional wrong rate ≥95%. D/E use the E33 family-level reliability factors, with prior shrinkage 30 for E. F uses CatBoost with 400 iterations, depth 6, learning rate 0.05, balanced class weights and 8 threads. Reliability is refit in each E31 prompt-group fold; F training features use four inner prompt-group folds.",
        "", "Selection and deployment router metadata are in `artifacts/analysis/"+eid.lower()+"/selection_results.json` and `deployment_manifest.json`. Full row-level scores and predictions are saved in the analysis directory.",""]
    (ROOT/f"docs/{eid.lower()}.md").write_text("\n".join(lines))


def run():
    v0=catalog.cascades._v0(ROOT)
    _,members,probabilities,excluded,snapshot=catalog._load_inputs(ROOT,v0,version=V044)
    if excluded:raise ValueError(f"v044 eligible prediction cache exclusions: {excluded}")
    if len(probabilities)!=94:raise ValueError(f"Expected complete 94-checkpoint v044 pool, got {len(probabilities)}")
    ids=v0.id.astype(str).to_numpy();prompts=v0.prompt_id.astype(str).to_numpy()
    y=pd.Categorical(v0.model.astype(str),categories=LABELS).codes.astype(np.int64)
    if len(v0)!=2699 or (y<0).any():raise ValueError("Invalid full V0 dataset/order")
    for eid in ("E35","E36"):update_catalogue(eid,"Running",stage="Roster selection")
    (r40,router40,stats40,pred40,score40,sel40,
     r60,router60,stats60,pred60,score60,sel60)=_select_rosters(v0,members,probabilities,snapshot)
    out35=ROOT/"artifacts/analysis/e35";out36=ROOT/"artifacts/analysis/e36"
    # Store full V0 aligned score arrays and exact raw probability-cache provenance.
    for eid,roster,stats,pred,scores in (("e35",r40,stats40,pred40,score40),("e36",r60,stats60,pred60,score60)):
        np.savez_compressed(ROOT/f"artifacts/analysis/{eid}/baseline_v0_scores.npz",
            row_ids=ids,classes=np.asarray(LABELS),baseline_scores=scores,baseline_predictions=pred)
    update_catalogue("E35","Running",stage="E35 six variants")
    result35=_run_variants("E35",r40,score40,pred40,members,probabilities,ids,prompts,y,snapshot)
    update_catalogue("E35","Scored",result=result35)
    update_catalogue("E36","Running",stage="E36 six variants")
    result36=_run_variants("E36",r60,score60,pred60,members,probabilities,ids,prompts,y,snapshot)
    update_catalogue("E36","Scored",result=result36)
    _write_consolidated(result35,result36)
    return result35,result36


def _write_consolidated(r35,r36):
    e33m=json.loads((E33/"metrics.json").read_text());e34m=json.loads((E34/"metrics.json").read_text())
    rows=[("E33-F","v037",40,e33m["summary"]["E33-F"]),("E34-F","v037",60,e34m["summary"]["E34-F"])]
    for eid,result in (("E35",r35),("E36",r36)):
        for name in (eid+" baseline",*(f"{eid}-{x}" for x in "ABCDEF")):
            rows.append((name,"v044",result["checkpoint_count"],result["summary"][name]))
    vals=np.asarray([[100*x[3][m]["mean_balanced_accuracy"] for m in METHODS] for x in rows])
    maxima=vals.max(axis=0)
    lines=["# E36 — v044 roster negative evidence comparison", "",
        "All results use 50 identical `v0_group` scenarios and the cached E33 group assignments. Baseline scores are produced by the new frozen router for each selected roster. E35 repeats E33 on 40 selected checkpoints; E36 repeats E34 on the 60-checkpoint extension.","",
        "| Experiment | Trust List | Checkpoints | Independent | E19-F-A | E19-F-C | E19-D |","|---|---|---:|---:|---:|---:|---:|"]
    for i,(name,version,count,_) in enumerate(rows):
        cells=[f"**{vals[i,j]:.4f}%**" if np.isclose(vals[i,j],maxima[j],rtol=0,atol=1e-12) else f"{vals[i,j]:.4f}%" for j in range(4)]
        lines.append(f"| {name} | {version} | {count} | "+" | ".join(cells)+" |")
    lines += ["", "## Interpretation", "",
        "E35 and E36 selections use the complete v044 pool (94 eligible checkpoints), E31/E32 prompt-group greedy procedures, and new deployment routers. Cross-fitted rule and CatBoost metrics are evaluated on the same fixed V0-derived scenarios as E33/E34. The E33/E34 rows are retained from their saved results for direct comparison.",
        "", "The v0_group scenarios reuse the official V0 responses, and the trust-list/checkpoint development process has used V0. These scores are robustness diagnostics, not independent Kaggle estimates.",
        "", "Artifacts: `artifacts/analysis/e35/` and `artifacts/analysis/e36/`; inference duration comparison is in [docs/e36_inference_benchmark.md](e36_inference_benchmark.md).",""]
    (ROOT/"docs/e36.md").write_text("\n".join(lines))


if __name__=="__main__":
    try:
        r35,r36=run()
        print(json.dumps({"E35":{k:{m:v["mean_balanced_accuracy"] for m,v in x.items()} for k,x in r35["summary"].items()},
            "E36":{k:{m:v["mean_balanced_accuracy"] for m,v in x.items()} for k,x in r36["summary"].items()}},indent=2))
    except Exception as exc:
        doc=json.loads((ROOT/"configs/ensemble_catalog_v2.json").read_text())
        states={x["id"]:x.get("status") for x in doc.get("experiments",[])}
        for eid in ("E35","E36"):
            if states.get(eid)!="Scored":
                update_catalogue(eid,"Failed",error=f"{type(exc).__name__}: {exc}")
        raise
