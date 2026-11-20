"""E18: matched-prompt contrastive specialists for sibling model labels."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from . import e16_common as common
from .feature_factory import FoldFeatures
from .v2_checkpoint_trust import LABELS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/analysis/e18"
SEED=2026
THREADS=24
INNER_FOLDS=3
CS=(0.01,0.1,1.0,10.0)
SWITCH_THRESHOLDS=(0.0,0.25,0.5,1.0,2.0,4.0)
GROUPS={
 "gemini_flash_pair":("gemini-3.7-flash","gemini-3.8-flash"),
 "glm_pair":("glm-5.3","glm-5.3-flash"),
 "gpt56_triplet":("gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra"),
}


def _group_indices(labels, group):
    return np.asarray([LABELS.index(x) for x in group],dtype=np.int64)


def _make_pairs(labels,groups,group_labels):
    labels=np.asarray(labels,dtype=str);groups=np.asarray(groups,dtype=str)
    found={label:np.flatnonzero(labels==label) for label in group_labels}
    if any(not len(ix) for ix in found.values()):raise ValueError("Each contrastive specialist needs examples for every target label")
    pairs={};
    for a,b in combinations(group_labels,2):
        ai=defaultdict(list);bi=defaultdict(list)
        for ix in found[a]:ai[groups[ix]].append(int(ix))
        for ix in found[b]:bi[groups[ix]].append(int(ix))
        common_groups=sorted(set(ai)&set(bi))
        left=[];right=[]
        for prompt in common_groups:
            for i in ai[prompt]:
                for j in bi[prompt]:left.append(i);right.append(j)
        if not left:raise ValueError(f"No matched prompt pairs for {a} vs {b}")
        pairs[(a,b)]=(np.asarray(left,dtype=np.int64),np.asarray(right,dtype=np.int64))
    return pairs


def fit_specialist(texts,labels,groups,group_labels,*,c=1.0):
    """Fit a character TF-IDF contrastive specialist and ordinary comparator."""
    texts=np.asarray(texts,dtype=str);labels=np.asarray(labels,dtype=str);groups=np.asarray(groups,dtype=str)
    eligible=np.flatnonzero(np.isin(labels,group_labels))
    if len(eligible)<20:raise ValueError("Too few sibling-family training rows")
    features=FoldFeatures({"views":["char"]})
    x=features.fit_transform(texts[eligible])
    pair_rows=_make_pairs(labels[eligible],groups[eligible],group_labels)
    contrastive={}
    for (a,b),(left_global,right_global) in pair_rows.items():
        left=np.asarray(left_global,dtype=np.int64);right=np.asarray(right_global,dtype=np.int64)
        delta=x[left]-x[right]
        dx=vstack([delta,-delta],format="csr")
        y=np.r_[np.ones(delta.shape[0],dtype=np.int8),np.zeros(delta.shape[0],dtype=np.int8)]
        clf=LogisticRegression(C=float(c),solver="liblinear",max_iter=2000,random_state=SEED)
        clf.fit(dx,y)
        contrastive[(a,b)]={"classifier":clf,"matched_pairs":int(delta.shape[0])}
    # Conventional row-level character TF-IDF baseline on the same vocabulary.
    # The GPT specialist is three-class; lbfgs supports both that case and the
    # binary sibling pairs while keeping one consistent ordinary baseline.
    ordinary=LogisticRegression(C=float(c),solver="lbfgs",max_iter=2000,random_state=SEED)
    label_map={label:i for i,label in enumerate(group_labels)}
    ordinary.fit(x,np.asarray([label_map[z] for z in labels[eligible]],dtype=np.int64))
    return {"features":features,"contrastive":contrastive,"ordinary":ordinary,
            "group_labels":tuple(group_labels),"ordinary_classes":ordinary.classes_,
            "C":float(c),"training_rows":int(len(eligible)),
            "matched_pairs":{f"{a}::{b}":v["matched_pairs"] for (a,b),v in contrastive.items()},
            "feature_count":int(x.shape[1])}


def specialist_scores(bundle,texts):
    """Score individual texts without using or accepting prompt relationships."""
    texts=np.asarray(texts,dtype=str);x=bundle["features"].transform(texts)
    labs=bundle["group_labels"];scores=np.zeros((len(texts),len(labs)),dtype=np.float64)
    for (a,b),item in bundle["contrastive"].items():
        clf=item["classifier"]
        # Pairwise classifier learns which side's response style is more
        # characteristic from paired feature differences; its linear margin
        # is applied to one response at inference.
        margin=0.5*np.asarray(x @ clf.coef_.T).reshape(-1)+float(clf.intercept_[0])/2.0
        ia,ib=labs.index(a),labs.index(b)
        scores[:,ia]+=margin;scores[:,ib]-=margin
    scores/=max(1,len(labs)-1)
    return scores


def ordinary_scores(bundle,texts):
    x=bundle["features"].transform(np.asarray(texts,dtype=str))
    p=bundle["ordinary"].predict_proba(x)
    out=np.zeros((len(texts),len(bundle["group_labels"])),dtype=float)
    for j,c in enumerate(bundle["ordinary"].classes_):out[:,int(c)]=p[:,j]
    return out


def _decide_group(y,base,alt,groups,e13_scores,specialist_score,group_labels,threshold):
    y=np.asarray(y,dtype=np.int64);base=np.asarray(base,dtype=np.int64);alt=np.asarray(alt,dtype=np.int64)
    group_idx=_group_indices(LABELS,group_labels);eligible=np.isin(base,group_idx)&np.isin(alt,group_idx)
    local={global_i:i for i,global_i in enumerate(group_idx)}
    sp=np.zeros(len(y),dtype=float)
    for i in range(len(y)):
        if eligible[i]:sp[i]=specialist_score[i,local[int(alt[i])]]-specialist_score[i,local[int(base[i])]]
    switch=eligible&(sp>=float(threshold));pred=np.where(switch,alt,base)
    return pred,switch,eligible,sp


def _tune_group(ctx,fit_rows,group_labels,group_name):
    y=ctx["y"];g=ctx["groups"]
    records=[]
    for split,(fit_local,val_local) in enumerate(GroupKFold(INNER_FOLDS).split(fit_rows,y[fit_rows],g[fit_rows]),1):
        tr,va=fit_rows[fit_local],fit_rows[val_local]
        state=common.fit_e13_predict(ctx,tr,va)
        record={"valid":va,"e13":state,"contrastive":{},"ordinary":{}}
        for c in CS:
            bundle=fit_specialist(ctx["texts"][tr],ctx["v0"].model.to_numpy()[tr],g[tr],group_labels,c=c)
            record["contrastive"][c]=specialist_scores(bundle,ctx["texts"][va])
            record["ordinary"][c]=ordinary_scores(bundle,ctx["texts"][va])
        records.append(record)
        print(f"E18 {group_name} nested fold {split}/{INNER_FOLDS}",flush=True)
    # C for matched-prompt specialist + correction threshold; optimize full 18-way BA.
    best=None
    for c in CS:
        for threshold in SWITCH_THRESHOLDS:
            yall=[];baseall=[];predall=[];corrected_count=0;eligible_count=0
            for r in records:
                va=r["valid"];state=r["e13"]
                p,sw,elig,_=_decide_group(y[va],state["baseline"],state["alternative"],g[va],state["label_scores"],r["contrastive"][c],group_labels,threshold)
                yall.extend(y[va].tolist());baseall.extend(state["baseline"].tolist());predall.extend(p.tolist());corrected_count+=int(sw.sum());eligible_count+=int(elig.sum())
            ba=float(balanced_accuracy_score(yall,predall));acc=float(accuracy_score(yall,predall));key=(ba,acc,-corrected_count,-threshold,-c)
            if best is None or key>best[0]:best=(key,{"C":c,"threshold":threshold,"tuning_balanced_accuracy":ba,"tuning_accuracy":acc,"eligible_rows":eligible_count,"switches":corrected_count})
    # Conventional classifier gets its own group OOF C chosen by specialist-set accuracy.
    ordinary_best=None
    local_map={z:i for i,z in enumerate(group_labels)}
    for c in CS:
        yy=[];pp=[]
        for r in records:
            va=r["valid"];mask=np.isin(ctx["v0"].model.to_numpy()[va],group_labels)
            if mask.any():
                labs=ctx["v0"].model.to_numpy()[va][mask]
                yy.extend([local_map[z] for z in labs]);pp.extend(r["ordinary"][c][mask].argmax(1).tolist())
        ba=float(balanced_accuracy_score(yy,pp));acc=float(accuracy_score(yy,pp));key=(ba,acc,-c)
        if ordinary_best is None or key>ordinary_best[0]:ordinary_best=(key,{"C":c,"balanced_accuracy":ba,"accuracy":acc})
    return best[1],ordinary_best[1]


def run(root=ROOT,output=OUT):
    os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]=str(THREADS)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);ctx=common.load_context(root);start=time.monotonic()
    y=ctx["y"];truth_labels=ctx["v0"].model.astype(str).to_numpy();groups=ctx["groups"];base=ctx["baseline"]
    correction=np.full(len(y),-1,dtype=np.int64);ordinary_oof={name:np.full(len(y),-1,dtype=np.int64) for name in GROUPS};contrastive_oof={name:np.full(len(y),-1,dtype=np.int64) for name in GROUPS}
    group_results={};fold_results=[]
    with threadpool_limits(limits=THREADS):
        for outer in ctx["outer_folds"]:
            fold=outer["fold"];fit=outer["fit_rows"];held=outer["held_rows"]
            print(f"E18 outer fold {fold}/5",flush=True)
            fold_info={"fold":fold,"rows":len(held),"groups":{}}
            final_pred=outer["baseline"].copy()
            for name,labels in GROUPS.items():
                params,ordinary_params=_tune_group(ctx,fit,labels,name)
                bundle=fit_specialist(ctx["texts"][fit],truth_labels[fit],groups[fit],labels,c=params["C"])
                cscore=specialist_scores(bundle,ctx["texts"][held]);oscore=ordinary_scores(bundle,ctx["texts"][held])
                mask=np.isin(truth_labels[held],labels)
                local={z:i for i,z in enumerate(labels)}
                contrastive_pred=np.asarray(labels,dtype=object)[cscore.argmax(1)]
                ordinary_pred=np.asarray(labels,dtype=object)[oscore.argmax(1)]
                contrastive_oof[name][held]=np.asarray([LABELS.index(z) if z in labels else -1 for z in contrastive_pred],dtype=np.int64)
                ordinary_oof[name][held]=np.asarray([LABELS.index(z) if z in labels else -1 for z in ordinary_pred],dtype=np.int64)
                specialist_global=np.zeros((len(held),len(labels)),dtype=float)
                # Map the held-out E13 candidate labels to specialist evidence.
                chosen,switch,eligible,gap=_decide_group(y[held],outer["baseline"],outer["alternative"],groups[held],outer["label_scores"],cscore,labels,params["threshold"])
                final_pred=np.where(eligible,chosen,final_pred)
                correction[held]=final_pred
                y_group=np.asarray([local[x] for x in truth_labels[held][mask]],dtype=np.int64)
                ordinary_metric={"rows":int(mask.sum()),"accuracy":float(accuracy_score(y_group,oscore[mask].argmax(1))),
                                 "balanced_accuracy":float(balanced_accuracy_score(y_group,oscore[mask].argmax(1)))}
                contrastive_metric={"rows":int(mask.sum()),"accuracy":float(accuracy_score(y_group,cscore[mask].argmax(1))),
                                    "balanced_accuracy":float(balanced_accuracy_score(y_group,cscore[mask].argmax(1)))}
                # For these subset rows, compare E13 only where its assigned label is in the group.
                e13_eligible=np.isin(outer["baseline"],_group_indices(LABELS,labels))
                eligible_count=int(eligible.sum())
                fold_info["groups"][name]={"labels":labels,"contrastive_C":params["C"],"threshold":params["threshold"],
                    "ordinary_C":ordinary_params["C"],"eligible_rows":eligible_count,"switches":int(switch.sum()),
                    "corrected":int(((outer["baseline"]!=y[held])&(chosen==y[held])&eligible).sum()),
                    "introduced":int(((outer["baseline"]==y[held])&(chosen!=y[held])&eligible).sum()),
                    "ordinary":ordinary_metric,"contrastive":contrastive_metric,
                    "matched_pairs":bundle["matched_pairs"],"training_rows":bundle["training_rows"],
                    "fold_group_baseline_rows":int((mask&e13_eligible).sum())}
                joblib.dump(bundle,output/f"E18_{name}_fold{fold}.joblib",compress=3)
            correction[held]=final_pred
            fold_info["combined"]=common.comparison(y[held],outer["baseline"],final_pred)
            fold_results.append(fold_info)
            print(f"E18 fold {fold}: BA {fold_info['combined']['balanced_accuracy']:.4%} vs E13 {fold_info['combined']['baseline_balanced_accuracy']:.4%}",flush=True)
    if (correction<0).any():raise ValueError("E18 folds failed to assign predictions to all V0 rows")
    metrics=common.comparison(y,base,correction);standalone={}
    for name,labels in GROUPS.items():
        ix=np.flatnonzero(np.isin(truth_labels,labels));local={z:i for i,z in enumerate(labels)};yt=np.asarray([local[truth_labels[i]] for i in ix])
        contrastive_codes=np.asarray([local[LABELS[int(contrastive_oof[name][i])]] for i in ix]);ordinary_codes=np.asarray([local[LABELS[int(ordinary_oof[name][i])]] for i in ix])
        standalone[name]={"labels":labels,"rows":int(len(ix)),"contrastive":{"accuracy":float(accuracy_score(yt,contrastive_codes)),"balanced_accuracy":float(balanced_accuracy_score(yt,contrastive_codes))},
            "ordinary_character_tfidf":{"accuracy":float(accuracy_score(yt,ordinary_codes)),"balanced_accuracy":float(balanced_accuracy_score(yt,ordinary_codes))}}
    predictions=ctx["v0"][["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy();predictions["E13_B"]=np.asarray(LABELS,dtype=object)[base];predictions["E18"]=np.asarray(LABELS,dtype=object)[correction]
    predictions["changed"]=correction!=base
    for name in GROUPS:
        col=np.where(contrastive_oof[name]>=0,np.asarray(LABELS,dtype=object)[np.maximum(0,contrastive_oof[name])],"")
        predictions[f"{name}_contrastive"]=col
        col2=np.where(ordinary_oof[name]>=0,np.asarray(LABELS,dtype=object)[np.maximum(0,ordinary_oof[name])],"")
        predictions[f"{name}_ordinary"]=col2
    predictions.to_parquet(output/"predictions.parquet",index=False)
    result={"experiment":"E18","baseline":"E13-B Trust List 2 v037","validation_rows":len(y),"baseline_reproduced_exactly":True,
      "combined":{**metrics,"per_label":common.metrics(y,correction)["per_label"],"confusion_matrix":common.metrics(y,correction)["confusion_matrix"]},
      "standalone_specialists":standalone,"folds":fold_results,"groups":GROUPS,
      "runtime_seconds":time.monotonic()-start,"threads":THREADS,
      "inference_features":"individual response char-TFIDF vector; no prompt IDs or prompt pairing required"}
    config={"seed":SEED,"threads":THREADS,"C_values":CS,"switch_thresholds":SWITCH_THRESHOLDS,"inner_folds":INNER_FOLDS,
      "feature_view":"src.feature_factory.FoldFeatures char TF-IDF (3-5 grams)","groups":GROUPS,"trust_list_snapshot":ctx["snapshot"],"cache_hashes":ctx["cache_hashes"],
      "matched_pair_training":"for each same-prompt sibling response pair, fit both oriented sparse vector differences with opposite binary targets"}
    (output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");(output/"config.json").write_text(json.dumps(config,indent=2,allow_nan=False)+"\n");_write_report(output,result)
    print(f"E18 complete: accuracy={metrics['accuracy']:.4%} BA={metrics['balanced_accuracy']:.4%} ΔBA={metrics['delta_balanced_accuracy']:+.4%}",flush=True);return result


def _write_report(output,result):
    m=result["combined"]
    lines=["# E18 — Sibling-Model Contrastive Specialists","",
      "Three prompt-matched character n-gram specialists rerank E13-B only when its selected label and strongest alternative are in the same sibling group. Matched prompt IDs are used for training pairs and fold isolation only; inference uses a response's own character-TF-IDF vector.","",
      "## Overall E13-B correction","","| Accuracy | Balanced accuracy | Δ accuracy | Δ balanced accuracy | Corrected | Introduced | Net gains | Switches |","|---:|---:|---:|---:|---:|---:|---:|---:|",
      f"| {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['delta_accuracy']:+.4%} | {m['delta_balanced_accuracy']:+.4%} | {m['corrected']} | {m['introduced']} | {m['net_gains']} | {m['switches']} |","",
      "## Standalone specialist comparison","","| Confusion group | Rows | Ordinary char TF-IDF accuracy | Ordinary BA | Matched-prompt contrastive accuracy | Contrastive BA |","|---|---:|---:|---:|---:|---:|"]
    for name,v in result["standalone_specialists"].items():
        a=v["ordinary_character_tfidf"];b=v["contrastive"]
        lines.append(f"| {name} ({', '.join(v['labels'])}) | {v['rows']} | {a['accuracy']:.4%} | {a['balanced_accuracy']:.4%} | {b['accuracy']:.4%} | {b['balanced_accuracy']:.4%} |")
    lines += ["","## Five-fold correction results","","| Fold | Rows | Accuracy | Balanced accuracy | Δ BA | Corrected | Introduced |","|---:|---:|---:|---:|---:|---:|---:|"]
    for f in result["folds"]:
        q=f["combined"];lines.append(f"| {f['fold']} | {f['rows']} | {q['accuracy']:.4%} | {q['balanced_accuracy']:.4%} | {q['delta_balanced_accuracy']:+.4%} | {q['corrected']} | {q['introduced']} |")
    lines += ["","## Specialist activation and class metrics","","| Fold | Specialist | Eligible | Corrected | Introduced | C | Threshold | Ordinary BA | Contrastive BA |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in result["folds"]:
        for name,g in f["groups"].items():lines.append(f"| {f['fold']} | {name} | {g['eligible_rows']} | {g['corrected']} | {g['introduced']} | {g['contrastive_C']:g} | {g['threshold']:g} | {g['ordinary']['balanced_accuracy']:.4%} | {g['contrastive']['balanced_accuracy']:.4%} |")
    lines += ["","## Per-label performance","","| Label | Support | Precision | Recall | F1 | TP | FP | FN |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        q=m["per_label"][label];lines.append(f"| {label} | {q['support']} | {q['precision']:.4%} | {q['recall']:.4%} | {q['f1']:.4%} | {q['tp']} | {q['fp']} | {q['fn']} |")
    lines += ["","## Method and limits","","- TF-IDF uses the existing `FoldFeatures` character view (3–5 character n-grams), fit inside each training fold.","- For each sibling pair, matched same-prompt training responses create a sparse difference vector; both orientations are used with opposite binary labels.","- GPT-5.6 Luna/Sol/Terra uses three one-vs-one contrastive classifiers. Their individual-response margins are averaged into per-model specialist scores.","- Specialist C and switch threshold are selected using nested prompt-group folds. Ordinary TF-IDF C is tuned separately on the same folds.","- E13-B is reproduced from its saved fold models and remains unchanged as the baseline.","",
      "## Artifacts","",f"- Analysis folder: `{output}`","- `predictions.parquet`: full-V0 OOF baseline, specialist, and corrected predictions.","- `E18_*_fold*.joblib`: fold-local pairwise TF-IDF specialist models.","- `config.json`, `results.json`: settings, pair counts, fold metrics and standalone comparisons.",""]
    text="\n".join(lines);(output/"e18.md").write_text(text);(ROOT/"docs/e18.md").write_text(text)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--output",type=Path,default=OUT);a=p.parse_args(argv);run(a.root,a.output);return 0


if __name__=="__main__":raise SystemExit(main())
