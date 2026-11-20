"""ORACLE03: candidate coverage, text-augmented stacking and confusion specialists.

This analysis consumes the saved Trust List 2 v025 full-V0 probability matrices.
All supervised fitting/selection uses the existing outer prompt-group folds and
inner GroupKFold splits. Text representations are deterministic, row-local or
frozen; no base checkpoint is loaded and no Kaggle submission is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import oracle02
from .ensemble_precision_recall_overlap import EQUAL9
from .feature_factory import stylometry
from .model_factory import extract_frozen_backbone_features
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/oracle03"
REPORT = ROOT / "docs/oracle03.md"
ORACLE02_PRED = ROOT / "artifacts/analysis/oracle02/row_predictions.parquet"
ORACLE02_METRICS = ROOT / "artifacts/analysis/oracle02/metrics.json"
HYP_META = ROOT / "artifacts/analysis/ensemble_hypotheses_nov15/metadata.json"
STYLO_RECIPE = {"baseline": ["length", "case", "punctuation", "function_words"],
                "groups": ["L", "V", "C", "P", "W", "F", "R", "X", "M", "U"]}
TEXT_VARIANTS = ("stack", "stack_stylometry", "stack_embedding", "stack_stylometry_embedding")
CAT_GRID = [{"depth": 3, "iterations": 100, "l2_leaf_reg": 12.0, "rsm": 0.3},
            {"depth": 4, "iterations": 100, "l2_leaf_reg": 20.0, "rsm": 0.3}]
BLEND_GRID = [0.0, .25, .5, .75, 1.0]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _metrics(y, p):
    pred = np.asarray(p).argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    tp = cm.diagonal(); support = cm.sum(1); fp = cm.sum(0) - tp
    precision = tp / np.maximum(tp + fp, 1); recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {"rows": int(len(y)), "correct": int((pred == y).sum()),
            "accuracy": float((pred == y).mean()), "balanced_accuracy": float(recall.mean()),
            "log_loss": float(log_loss(y, p, labels=np.arange(len(LABELS)))),
            "per_label": {l: {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(fp[i]),
                               "fn": int(support[i]-tp[i]), "support": int(support[i])}
                          for i, l in enumerate(LABELS)}, "pred": pred}


def _candidate_coverage(matrices, roster, y, equal9, h09, all78, oracle02_probs=None):
    rosters = {"EQUAL9": list(equal9), "H09": list(h09), "ALL78": list(all78)}
    ensembles = {"EQUAL9": np.mean([matrices[s] for s in equal9], 0),
                 "H09": np.mean([matrices[s] for s in h09], 0),
                 "ALL78": np.mean([matrices[s] for s in all78], 0)}
    rows=[]; detail={}; confusion={}
    if oracle02_probs is not None:
        ensembles["ORACLE02_all78"] = np.asarray(oracle02_probs)
    eye=np.eye(len(LABELS))
    for name, ids in rosters.items():
        probs=np.stack([matrices[s] for s in ids], axis=0)
        top=np.argmax(probs, axis=2)
        vote=np.stack([(top == j).sum(0) for j in range(len(LABELS))], axis=1)
        summed=probs.sum(0)
        methods={"top1_vote_count":vote.astype(float), "summed_checkpoint_probability":summed,
                 "ensemble_probability":ensembles[name]}
        if name == "ALL78" and oracle02_probs is not None:
            methods["ORACLE02_oof_probability"] = np.asarray(oracle02_probs)
        detail[name]={}
        distinct_count=(vote>0).sum(1)
        detail[name]["oracle_coverage"]={"covered_rows":int((vote[np.arange(len(y)),y]>0).sum()),
            "coverage":float((vote[np.arange(len(y)),y]>0).mean()),
            "per_label_supported_rows":{label:int(((y==ci)&(vote[:,ci]>0)).sum()) for ci,label in enumerate(LABELS)},
            "candidate_set_size_distribution":{str(k):int((distinct_count==k).sum()) for k in range(1,len(ids)+1)}}
        for method, scores in methods.items():
            rank_scores=np.asarray(scores).copy()
            # A vote-count shortlist cannot include a label that no member proposed.
            # Without this mask, zero-vote ties would create artificial oracle coverage.
            if method == "top1_vote_count":
                rank_scores[vote == 0] = -1
            ranked=np.argsort(-rank_scores, axis=1, kind="stable")
            hit=(ranked == y[:,None])
            per={}
            for k in (1,2,3,5):
                row_hit=(ranked[:,:k] == y[:,None]).any(1)
                if method == "top1_vote_count": row_hit &= vote[np.arange(len(y)),y] > 0
                count=int(row_hit.sum())
                oracle_supported=vote[np.arange(len(y)),y] > 0
                supported_hit=(ranked[:,:k] == y[:,None]).any(1)&oracle_supported
                supported_count=int(supported_hit.sum())
                excluded=int((oracle_supported&~((ranked[:,:k] == y[:,None]).any(1))).sum())
                per[str(k)]={"covered":count,"coverage":count/len(y),
                    "oracle_supported_in_shortlist":supported_count,
                    "supported_but_excluded":excluded,
                    "fraction_of_oracle_covered":supported_count/max(1,int(oracle_supported.sum()))}
                rows.append({"roster":name,"method":method,"top_k":k,"covered":count,"coverage":count/len(y),
                    "oracle_supported_in_shortlist":supported_count,"supported_but_excluded":excluded,
                    "fraction_of_oracle_covered":supported_count/max(1,int(oracle_supported.sum()))})
            per_label={}
            for ci,label in enumerate(LABELS):
                mask=y==ci
                per_label[label]={}
                for k in (1,2,3,5):
                    candidate_hit=(ranked[:,:k]==ci).any(1)&mask
                    supported_hit=candidate_hit&(vote[:,ci]>0)
                    per_label[label][str(k)]={"covered_rows":int(candidate_hit.sum()),
                        "oracle_supported_rows_in_shortlist":int(supported_hit.sum()),
                        "oracle_supported_rows_omitted":int(((mask&(vote[:,ci]>0))&~(ranked[:,:k]==ci).any(1)).sum())}
            size_hist={str(k):int((np.minimum(distinct_count,k)==k).sum()) for k in (1,2,3,5)}
            support=vote[np.arange(len(y)),y]
            for n in range(len(ids)+1):
                detail[name].setdefault("correct_label_checkpoint_top1_support",{})[str(n)]=int((support==n).sum())
            detail[name][method]={"top_k":per,"per_label_covered_rows":per_label,
                "effective_candidate_count_at_k":size_hist,
                "correct_label_support_mean":float(support.mean()),
                "correct_label_minority_support_rows":int((support * 2 < len(ids)).sum()),
                "correct_label_zero_support_rows":int((support==0).sum())}
        # Count directed errors for each member and ensemble, preserving labels.
        member_votes=top
        pred=np.argmax(ensembles[name],axis=1)
        cm=np.bincount(y*len(LABELS)+pred,minlength=len(LABELS)**2).reshape(len(LABELS),len(LABELS))
        confusion[name]=[{"true":LABELS[i],"predicted":LABELS[j],"count":int(cm[i,j])}
                         for i in range(len(LABELS)) for j in range(len(LABELS)) if i!=j and cm[i,j]]
        detail[name]["ensemble_confusions"] = sorted(confusion[name],key=lambda x:(-x["count"],x["true"],x["predicted"]))[:30]
        detail[name]["member_top1_confusion_pairs"] = _pair_confusions(member_votes,y)
    return pd.DataFrame(rows),detail,ensembles


def refresh_candidate_coverage(output_dir=OUT):
    output_dir=Path(output_dir)
    ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3,equal9,hashes=oracle02._source_inputs(ROOT,output_dir,True)
    pred02=pd.read_parquet(ORACLE02_PRED)
    p_a=np.column_stack([pred02[f"prob_oof::all78::catboost_equal9_blend::{l}"].to_numpy() for l in LABELS])
    h09=json.loads(HYP_META.read_text())["scenario_definitions"]["H09"]
    frame,detail,ensembles=_candidate_coverage(matrices,ids,y,EQUAL9,h09,ids,p_a)
    frame.to_csv(output_dir/"candidate_coverage.csv",index=False)
    _json(output_dir/"candidate_coverage.json",detail)
    metrics_path=output_dir/"metrics.json"
    if metrics_path.exists():
        metrics=json.loads(metrics_path.read_text());metrics["candidate_coverage"]=detail
        _json(metrics_path,metrics)
        _write_report(REPORT,metrics,frame,detail["ALL78"]["member_top1_confusion_pairs"])
    return frame,detail


def _pair_confusions(member_top, y):
    count={}
    for preds in member_top:
        for t,p in zip(y,preds):
            if t!=p:
                key=(LABELS[int(t)],LABELS[int(p)])
                count[key]=count.get(key,0)+1
    return [{"true":a,"predicted":b,"count":v} for (a,b),v in
            sorted(count.items(),key=lambda kv:(-kv[1],kv[0]))[:40]]


def _fit_cat(X,y,cfg):
    m=CatBoostClassifier(loss_function="MultiClass",eval_metric="MultiClass",depth=cfg["depth"],
        iterations=cfg["iterations"],learning_rate=.05,l2_leaf_reg=cfg["l2_leaf_reg"],
        random_seed=2026,thread_count=1,task_type="CPU",verbose=False,allow_writing_files=False,
        random_strength=.5,bootstrap_type="Bernoulli",subsample=.85,border_count=64,rsm=cfg["rsm"])
    m.fit(X,y); return m


def _cat_proba(model,X):
    raw=model.predict_proba(X); out=np.zeros((len(X),len(LABELS)))
    for j,c in enumerate(model.classes_): out[:,int(c)]=raw[:,j]
    return out / np.maximum(out.sum(1,keepdims=True),1e-15)


def _inner(train,groups):
    return [(train[a],train[b]) for a,b in GroupKFold(3).split(
        np.zeros(len(train)), groups=groups[train])]


def _nested_cat_variant(X,y,groups,outer_train,eq):
    splits=_inner(outer_train,groups); pos={int(i):j for j,i in enumerate(outer_train)}
    cfg_oof={}; cfg_scores=[]
    for cfg in CAT_GRID:
        pred=np.zeros((len(outer_train),len(LABELS)))
        for tr,va in splits:
            model=_fit_cat(X[tr],y[tr],cfg)
            pred[[pos[int(i)] for i in va]]=_cat_proba(model,X[va])
        met=_metrics(y[outer_train],pred)
        cfg_oof[tuple(cfg[k] for k in ("depth","iterations","l2_leaf_reg","rsm"))]=pred
        cfg_scores.append({**cfg,"balanced_accuracy":met["balanced_accuracy"],"log_loss":met["log_loss"]})
    selected=min(cfg_scores,key=lambda z:(-z["balanced_accuracy"],z["log_loss"],z["depth"]))
    cfg={k:selected[k] for k in ("depth","iterations","l2_leaf_reg","rsm")}
    cat_oof=cfg_oof[tuple(cfg[k] for k in ("depth","iterations","l2_leaf_reg","rsm"))]
    blends=[]
    for alpha in BLEND_GRID:
        p=alpha*cat_oof+(1-alpha)*eq[outer_train]
        m=_metrics(y[outer_train],p)
        blends.append({"alpha_catboost":alpha,"balanced_accuracy":m["balanced_accuracy"],"log_loss":m["log_loss"]})
    blend=min(blends,key=lambda z:(-z["balanced_accuracy"],z["log_loss"],z["alpha_catboost"]))
    return cfg,float(blend["alpha_catboost"]),cat_oof,{"grid":cfg_scores,"selected":selected,"blend_grid":blends,"selected_blend":blend}


def _pair_groups(confusion_pairs):
    fixed=[["glm-5.3","glm-5.3-flash"],
           ["gemini-3.7-flash","gemini-3.8-flash"],
           ["gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra"]]
    # Add up to three independent frequent pairs, not duplicates of the requested groups.
    requested={frozenset(g) for g in fixed}; added=[]
    for row in confusion_pairs:
        pair=frozenset((row["true"],row["predicted"]))
        if len(pair)==2 and pair not in requested and not any(pair.issubset(set(g)) for g in fixed):
            added.append(sorted(pair)); requested.add(pair)
            if len(added)==3: break
    return fixed+added


def _specialist_predict(train_idx,test_idx,group,X,y,groups,base_p,threshold_base):
    members=np.asarray([LABELS.index(l) for l in group]); in_train=np.isin(y[train_idx],members)
    tr= train_idx[in_train]
    if len(np.unique(y[tr]))<2 or len(tr)<20:
        return base_p[test_idx].copy(),{"status":"insufficient training examples","threshold":None},None,np.zeros((len(test_idx),len(LABELS)))
    # Nested OOF estimates for a conservative override threshold.
    oof=np.zeros((len(train_idx),len(members))); local_pos={int(i):j for j,i in enumerate(train_idx)}
    group_rows=np.flatnonzero(np.isin(y[train_idx],members))
    group_idx=train_idx[group_rows]
    n_groups=len(np.unique(groups[group_idx])); n_inner=min(3,n_groups)
    for aa,bb in GroupKFold(n_inner).split(np.zeros(len(group_idx)),groups=groups[group_idx]):
        a=group_idx[aa];b=group_idx[bb]
        scaler=StandardScaler().fit(X[a])
        model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,random_state=2026)
        model.fit(scaler.transform(X[a]),y[a])
        pp=model.predict_proba(scaler.transform(X[b]));
        for col,cls in enumerate(model.classes_): oof[[local_pos[int(i)] for i in b],np.where(members==int(cls))[0][0]]=pp[:,col]
    # Select only an override threshold from inner-fold OOF data. Compare candidates on
    # overall BA, with log-loss/trigger rate not used to leak held-out labels.
    specialist_train_pred=members[oof.argmax(1)]
    # Threshold learning uses the fixed EQUAL9 probabilities, whose prediction rule
    # does not fit labels. This avoids using globally cross-fitted ORACLE02 training
    # rows whose fitted meta-model may have seen the current outer holdout labels.
    base_train=threshold_base[train_idx].argmax(1)
    margins=np.max(oof,axis=1)
    thresholds=[0.0,.35,.45,.55,.65,.75,.85]
    scored=[]
    for th in thresholds:
        out=base_train.copy(); trigger=(np.isin(base_train,members)&(margins>=th))
        out[trigger]=specialist_train_pred[trigger]
        scored.append((oracle02._canonical_metrics(y[train_idx],np.eye(len(LABELS))[out]*.999999999999+1e-12)["balanced_accuracy"],-int(trigger.sum()),th))
    threshold=max(scored)[2]
    scaler=StandardScaler().fit(X[tr])
    model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,random_state=2026)
    model.fit(scaler.transform(X[tr]),y[tr]);raw=model.predict_proba(scaler.transform(X[test_idx]))
    spec=np.zeros((len(test_idx),len(members)))
    for col,cls in enumerate(model.classes_): spec[:,np.where(members==int(cls))[0][0]]=raw[:,col]
    raw_full=np.zeros((len(test_idx),len(LABELS)))
    raw_full[:,members]=spec
    out=base_p[test_idx].copy(); pred=members[spec.argmax(1)]; confidence=spec.max(1)
    trigger=np.isin(out.argmax(1),members)&(confidence>=threshold)
    out[trigger]=0
    out[np.flatnonzero(trigger)[:,None],members[None,:]]=spec[trigger]
    out/=np.maximum(out.sum(1,keepdims=True),1e-15)
    return out,{"status":"ok","threshold":float(threshold),"training_support":{LABELS[i]:int((y[tr]==i).sum()) for i in members},"trigger_count":int(trigger.sum()),"group":group},{"scaler":scaler,"model":model},raw_full


def _candidate_matrix(scores, y, k):
    order=np.argsort(-scores,axis=1,kind="stable")[:,:k]
    return order


def refresh_specialists(output_dir=OUT, report_path=REPORT):
    """Recompute only the specialist cascade from saved features and cached outputs."""
    output_dir=Path(output_dir)
    ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3,equal9,hashes=oracle02._source_inputs(ROOT,output_dir,True)
    frame=pd.read_parquet(output_dir/"row_predictions.parquet")
    if frame.id.astype(str).tolist()!=row_ids: raise ValueError("ORACLE03 result rows changed")
    p_a=np.column_stack([frame[f"prob::ORACLE02_A::{l}"].to_numpy() for l in LABELS])
    p_eq=np.column_stack([frame[f"prob::EQUAL9::{l}"].to_numpy() for l in LABELS])
    z=np.load(output_dir/"text_features.npz")
    if z["ids"].astype(str).tolist()!=row_ids: raise ValueError("Text feature rows changed")
    Xbase,_=oracle02._build_features(ids,matrices,equal9)
    Xspec=np.column_stack((Xbase,z["stylometry"],z["embedding"]))
    member_top=np.stack([matrices[s].argmax(1) for s in ids])
    outputs={"rerank_top2":np.zeros_like(p_a),"rerank_top3":np.zeros_like(p_a),"selective_override":np.zeros_like(p_a)}
    group_info={};fold_details=[]
    for fold in range(1,6):
        tr=np.flatnonzero(fold_ids!=fold);va=np.flatnonzero(fold_ids==fold)
        for key in outputs: outputs[key][va]=p_a[va]
        fold_groups=_pair_groups(_pair_confusions(member_top[:,tr],y[tr]))
        fold_info=[]
        for group in fold_groups:
            selective,info,model,raw_spec=_specialist_predict(tr,va,group,Xspec,y,prompts,p_a,p_eq)
            if info["status"]!="ok": continue
            model_path=output_dir/"models"/"specialists"/f"outer_{fold}"/("_".join(group)+".joblib")
            model_path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,model_path,compress=3)
            rerank_stats={}
            for k,key in ((2,"rerank_top2"),(3,"rerank_top3")):
                result=outputs[key][va].copy(); cand=np.argsort(-result,axis=1,kind="stable")[:,:k]
                member_ids=[LABELS.index(v) for v in group]
                original_cand=np.argsort(-p_a[va],axis=1,kind="stable")[:,:k]
                eligible=np.asarray([sum(j in member_ids for j in row)>1 for row in original_cand])
                individual_pred=np.asarray([max([j for j in row if j in member_ids],
                    key=lambda j:(raw_spec[r,j],-j)) if eligible[r] else int(p_a[va[r]].argmax())
                    for r,row in enumerate(original_cand)])
                base_pred=p_a[va].argmax(1); truth=y[va]
                rerank_stats[str(k)]={"eligible_rows":int(eligible.sum()),
                    "selected_correct":int(((individual_pred==truth)&eligible).sum()),
                    "selected_wrong":int(((individual_pred!=truth)&eligible).sum()),
                    "corrected_vs_oracle02":int(((individual_pred==truth)&(base_pred!=truth)&eligible).sum()),
                    "introduced_vs_oracle02":int(((individual_pred!=truth)&(base_pred==truth)&eligible).sum()),
                    "conditional_accuracy":float((individual_pred[eligible]==truth[eligible]).mean()) if eligible.any() else None}
                for ri,row in enumerate(cand):
                    competing=[j for j in row if j in member_ids]
                    if len(competing)>1:
                        chosen=max(competing,key=lambda j:(raw_spec[ri,j],-j))
                        result[ri,chosen]=result[ri].max()+1e-9
                result/=np.maximum(result.sum(1,keepdims=True),1e-15)
                outputs[key][va]=result
            changed=selective.argmax(1)!=p_a[va].argmax(1)
            merged=outputs["selective_override"][va].copy();merged[changed]=selective[changed]
            outputs["selective_override"][va]=merged
            member_ids=np.asarray([LABELS.index(v) for v in group])
            base_pred=p_a[va].argmax(1);true=y[va]
            trigger=np.isin(base_pred,member_ids)&(raw_spec.max(1)>=info["threshold"])
            routed_pred=member_ids[raw_spec[:,member_ids].argmax(1)]
            info["selective_heldout"]={"triggered_rows":int(trigger.sum()),
                "correct":int(((routed_pred==true)&trigger).sum()),"incorrect":int(((routed_pred!=true)&trigger).sum()),
                "corrected_vs_oracle02":int(((routed_pred==true)&(base_pred!=true)&trigger).sum()),
                "introduced_vs_oracle02":int(((routed_pred!=true)&(base_pred==true)&trigger).sum()),
                "conditional_accuracy":float((routed_pred[trigger]==true[trigger]).mean()) if trigger.any() else None}
            info["candidate_reranking_heldout"]=rerank_stats
            fold_info.append({**info,"fold":fold})
        for key in outputs:
            old=p_a[va].argmax(1);new=outputs[key][va].argmax(1);yy=y[va];changed=new!=old
            base_ba=_metrics(yy,p_a[va])["balanced_accuracy"]
            new_ba=_metrics(yy,outputs[key][va])["balanced_accuracy"]
            fold_details.append({"fold":fold,"strategy":key,"trigger_count":int(changed.sum()),
                "conditional_accuracy":float((new[changed]==yy[changed]).mean()) if changed.any() else None,
                "corrected":int(((new==yy)&(old!=yy)).sum()),"introduced":int(((new!=yy)&(old==yy)).sum()),
                "baseline_balanced_accuracy":base_ba,"strategy_balanced_accuracy":new_ba,
                "delta_balanced_accuracy_pp":(new_ba-base_ba)*100,
                "heldout_rows":int(len(va))})
        group_info[str(fold)]={"selected_groups":fold_groups,"rules":fold_info}
    m=json.loads((output_dir/"metrics.json").read_text())
    coverage_df,coverage_detail=refresh_candidate_coverage(output_dir)
    m["candidate_coverage"]=coverage_detail
    _json(output_dir/"metrics.json",m)
    for key,p in outputs.items():
        metric=_metrics(y,p);pred=metric.pop("pred");m["metrics"][key]=metric
        for ref,ref_p in (("EQUAL9",p_eq),("H09",np.column_stack([frame[f"prob::H09::{l}"].to_numpy() for l in LABELS])),("ORACLE02",p_a)):
            m["corrections"].setdefault(key,{})[ref]={"corrected":int(((pred==y)&(ref_p.argmax(1)!=y)).sum()),
                "introduced":int(((pred!=y)&(ref_p.argmax(1)==y)).sum()),"changed":int((pred!=ref_p.argmax(1)).sum())}
        frame[f"pred::{key}"]=np.asarray(LABELS,dtype=object)[pred]
        for j,label in enumerate(LABELS): frame[f"prob::{key}::{label}"]=p[:,j]
    m["specialist_groups_by_fold"]={k:{"selected_groups":v["selected_groups"]} for k,v in group_info.items()}
    m["specialist_groups"]=sorted({tuple(g) for v in group_info.values() for g in v["selected_groups"]})
    m["specialist_folds"]=[x for v in group_info.values() for x in v["rules"]]
    m["specialist_oof_by_fold"]=fold_details
    frame.to_parquet(output_dir/"row_predictions.parquet",index=False)
    _json(output_dir/"metrics.json",m)
    per=[]
    for name in m["metrics"]:
        probs=np.column_stack([frame[f"prob::{name}::{l}"].to_numpy() for l in LABELS])
        mm=_metrics(y,probs);per.extend({"strategy":name,"label":l,**v} for l,v in mm["per_label"].items())
    pd.DataFrame(per).to_csv(output_dir/"per_label_metrics.csv",index=False)
    coverage=pd.read_csv(output_dir/"candidate_coverage.csv")
    confusions=m["candidate_coverage"]["ALL78"]["member_top1_confusion_pairs"]
    _write_report(Path(report_path),m,coverage,confusions)
    manifest_path=output_dir/"run_manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"status":"completed","analysis":"ORACLE03","trust_list":"v025",
        "validation_rows":len(y),"prompt_groups":int(pd.Series(prompts).nunique()),
        "class_order":LABELS,"outer_fold_source":str(oracle02.V0_FOLDS),
        "script_sha256":_sha(Path(__file__).resolve()),
        "input_hashes":hashes,"oracle02_oof_sha256":_sha(ORACLE02_PRED),
        "stylometry_feature_sha256":_sha(output_dir/"text_features.npz"),
        "frozen_embedding_artifact_sha256":_sha(output_dir/"text_embeddings.npz"),
        "script_sha256":_sha(Path(__file__).resolve()),
        "test_inference_or_base_retraining":False,"kaggle_submission":False,
        "selected_checkpoint_rosters":{"EQUAL9":EQUAL9,"H09":json.loads(HYP_META.read_text())["scenario_definitions"]["H09"],"ALL78":ids},
        "class_order_alignment":"canonical LABELS; probability arrays verified by ID and prompt ID"})
    _json(manifest_path,manifest)
    return m


def _refresh_text_only_models(output_dir):
    """Nested standalone text classifiers and blends against ORACLE02."""
    output_dir=Path(output_dir)
    ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3,equal9,hashes=oracle02._source_inputs(ROOT,output_dir,True)
    frame=pd.read_parquet(output_dir/"row_predictions.parquet")
    z=np.load(output_dir/"text_features.npz")
    if z["ids"].astype(str).tolist()!=row_ids: raise ValueError("Text feature rows changed")
    p_a=np.column_stack([frame[f"prob::ORACLE02_A::{l}"].to_numpy() for l in LABELS])
    p_eq=np.column_stack([frame[f"prob::EQUAL9::{l}"].to_numpy() for l in LABELS])
    sets={"stylometry":z["stylometry"],"embedding":z["embedding"],
          "stylometry_embedding":np.column_stack((z["stylometry"],z["embedding"]))}
    result=json.loads((output_dir/"metrics.json").read_text())
    for feature_name,X in sets.items():
        held_lr=np.zeros_like(p_a);held_blend=np.zeros_like(p_a);fold_records=[]
        for fold in range(1,6):
            train=np.flatnonzero(fold_ids!=fold);held=np.flatnonzero(fold_ids==fold)
            splits=_inner(train,np.asarray(prompts));position={int(i):j for j,i in enumerate(train)}
            scored=[];predictions={}
            for C in (0.1,1.0,10.0):
                inner=np.zeros((len(train),len(LABELS)))
                for tr,va in splits:
                    bundle=oracle02._fit_logistic(X[tr],y[tr],C)
                    inner[[position[int(i)] for i in va]]=oracle02._predict_logistic(bundle,X[va])
                mm=_metrics(y[train],inner)
                scored.append({"C":C,"balanced_accuracy":mm["balanced_accuracy"],"log_loss":mm["log_loss"]})
                predictions[C]=inner
            selected=min(scored,key=lambda a:(-a["balanced_accuracy"],a["log_loss"],a["C"]))
            C=selected["C"];inner=predictions[C]
            blends=[]
            for alpha in BLEND_GRID:
                # EQUAL9 is a fixed probability baseline with no supervised meta-fit.
                # Using globally cross-fitted ORACLE02 probabilities here could let
                # an outer held-out prompt group influence a training-fold blend fit.
                pp=alpha*inner+(1-alpha)*p_eq[train]
                mm=_metrics(y[train],pp)
                blends.append({"alpha_text":alpha,"balanced_accuracy":mm["balanced_accuracy"],"log_loss":mm["log_loss"]})
            blend=min(blends,key=lambda a:(-a["balanced_accuracy"],a["log_loss"],a["alpha_text"]))
            model=oracle02._fit_logistic(X[train],y[train],C)
            held_lr[held]=oracle02._predict_logistic(model,X[held])
            held_blend[held]=blend["alpha_text"]*held_lr[held]+(1-blend["alpha_text"])*p_a[held]
            path=output_dir/"models"/"text_only"/feature_name/f"outer_{fold}.joblib"
            path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path,compress=3)
            fold_records.append({"fold":fold,"selected_C":C,"inner_C_grid":scored,
                "blend_selection":blend,"inner_blend_grid":blends,
                "training_inner_oof":{k:v for k,v in _metrics(y[train],blend["alpha_text"]*inner+(1-blend["alpha_text"])*p_a[train]).items() if k!="pred"},
                "heldout":{k:v for k,v in _metrics(y[held],held_blend[held]).items() if k!="pred"}})
        for suffix,p in (("logistic",held_lr),("blend",held_blend)):
            name=f"text_only_{feature_name}_{suffix}"
            mm=_metrics(y,p);pred=mm.pop("pred");result["metrics"][name]=mm
            result.setdefault("corrections",{})[name]={}
            for ref,ref_p in (("EQUAL9",p_eq),("H09",np.column_stack([frame[f"prob::H09::{l}"].to_numpy() for l in LABELS])),("ORACLE02",p_a)):
                result["corrections"][name][ref]={"corrected":int(((pred==y)&(ref_p.argmax(1)!=y)).sum()),
                    "introduced":int(((pred!=y)&(ref_p.argmax(1)==y)).sum()),"changed":int((pred!=ref_p.argmax(1)).sum())}
            frame[f"pred::{name}"]=np.asarray(LABELS,dtype=object)[pred]
            for j,label in enumerate(LABELS):frame[f"prob::{name}::{label}"]=p[:,j]
            result.setdefault("text_only_fold_selection",{})[name]=fold_records
        result["text_only_blend_selection_reference"]="EQUAL9 probabilities inside fitting folds; held-out blend uses the ORACLE02 OOF baseline to avoid outer-fold label leakage from its globally cross-fitted train rows. This is a conservative nested proxy, not a fully refit ORACLE02 inner stacker."
    frame.to_parquet(output_dir/"row_predictions.parquet",index=False)
    _json(output_dir/"metrics.json",result)
    manifest_path=output_dir/"run_manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());manifest["script_sha256"]=_sha(Path(__file__).resolve())
        manifest["text_only_blend_selection_reference"]=result["text_only_blend_selection_reference"]
        _json(manifest_path,manifest)
    per=[]
    for name in result["metrics"]:
        pp=np.column_stack([frame[f"prob::{name}::{l}"].to_numpy() for l in LABELS])
        mm=_metrics(y,pp);per.extend({"strategy":name,"label":l,**v} for l,v in mm["per_label"].items())
    pd.DataFrame(per).to_csv(output_dir/"per_label_metrics.csv",index=False)


def run(output_dir=OUT,report_path=REPORT,*,with_embeddings=True):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3,equal9,hashes=oracle02._source_inputs(ROOT,output_dir,True)
    hyp=json.loads(HYP_META.read_text()); h09=hyp["scenario_definitions"]["H09"]
    if not set(h09).issubset(matrices): raise ValueError("H09 contains checkpoint IDs absent from the v025 cache")
    all78=list(ids); Xbase,names=oracle02._build_features(all78,matrices,equal9)
    pred02=pd.read_parquet(ORACLE02_PRED)
    if pred02.id.astype(str).tolist()!=row_ids or pred02.prompt_id.astype(str).tolist()!=prompts.tolist():
        raise ValueError("ORACLE02 rows are not aligned to the current frozen V0 rows")
    p_a=np.column_stack([pred02[f"prob_oof::all78::catboost_equal9_blend::{l}"].to_numpy() for l in LABELS])
    # Reuse full-V0 V0 responses for row-local features; no label statistics enter extraction.
    stylo,stylo_names=stylometry(v0.text.astype(str).tolist(),STYLO_RECIPE,return_feature_names=True)
    stylo=np.asarray(stylo,dtype=np.float32)
    embed_path=output_dir/"text_embeddings.npz"
    if with_embeddings:
        if embed_path.exists():
            ez=np.load(embed_path); embedding=ez["embedding"]
            if ez["ids"].astype(str).tolist()!=row_ids: raise ValueError("Saved embedding IDs do not align to V0")
        else:
            os.environ.setdefault("HF_HUB_OFFLINE","1")
            embedding,_=extract_frozen_backbone_features("embeddinggemma2","google/embeddinggemma-2",
                "f9972ac22e696a105c482202646979e33e4b5a1e",v0.text.astype(str).tolist(),max_length=384,
                batch_size=4,row_ids=row_ids,cache_dir=str(ROOT/"artifacts/cache/frozen_embeddings"))
            np.savez_compressed(embed_path,embedding=np.asarray(embedding,dtype=np.float32),ids=np.asarray(row_ids))
    else: embedding=np.empty((len(y),0),dtype=np.float32)
    np.savez_compressed(output_dir/"text_features.npz",stylometry=stylo,embedding=embedding,ids=np.asarray(row_ids))
    eq_probs=equal9
    feature_sets={"stack":Xbase,"stack_stylometry":np.column_stack((Xbase,stylo)),
        "stack_embedding":np.column_stack((Xbase,embedding)),
        "stack_stylometry_embedding":np.column_stack((Xbase,stylo,embedding))}
    # Candidate ranking/oracle coverage.
    coverage_df,cov_detail,ensemble_probs=_candidate_coverage(matrices,ids,y,EQUAL9,h09,all78,p_a)
    coverage_df.to_csv(output_dir/"candidate_coverage.csv",index=False)
    _json(output_dir/"candidate_coverage.json",cov_detail)
    # Existing ORACLE02 OOF provides exact A results and baseline row-wise scores.
    p_eq=np.column_stack([pred02[f"prob_equal9::{l}"].to_numpy() for l in LABELS])
    # H09 and all78 mean-probability candidate ranks are included among deployable rankers.
    # Run nested text-feature models for B/C/D; A uses the original ORACLE02 OOF.
    oof={"ORACLE02_A":p_a}; fold_info={}; fold_group=np.asarray(prompts)
    for variant in TEXT_VARIANTS[1:]:
        X=feature_sets[variant]; held_cat=np.zeros_like(p_a); held_blend=np.zeros_like(p_a); details=[]
        for fold in range(1,6):
            tr=np.flatnonzero(fold_ids!=fold);va=np.flatnonzero(fold_ids==fold)
            cfg,alpha,inner_oof,sel=_nested_cat_variant(X,y,fold_group,tr,p_eq)
            model=_fit_cat(X[tr],y[tr],cfg);held_cat[va]=_cat_proba(model,X[va])
            model_path=output_dir/"models"/variant/f"outer_{fold}.cbm"
            model_path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(model_path))
            held_blend[va]=alpha*held_cat[va]+(1-alpha)*p_eq[va]
            details.append({"fold":fold,"config":cfg,"blend_alpha":alpha,"inner_selection":sel,
                            "train_metrics":_metrics(y[tr],alpha*inner_oof+(1-alpha)*p_eq[tr])|{"pred":None},
                            "heldout_metrics":_metrics(y[va],held_blend[va])|{"pred":None}})
        oof[variant+"_catboost"]=held_cat;oof[variant+"_blend"]=held_blend;fold_info[variant]=details
    # Pairwise/family specialist evaluation. The common base is the already cross-fitted
    # ORACLE02 blend; only specialist training/threshold decisions vary per outer fold.
    member_top=np.stack([matrices[s].argmax(1) for s in ids])
    # Full-V0 confusion pairs are descriptive only. Selection of extra specialist
    # groups is repeated from each outer fitting partition below.
    confusion_pairs=_pair_confusions(member_top,y)
    groups_by_fold={}
    Xspec=feature_sets["stack_stylometry_embedding"]
    specialist_oof={"rerank_top2":np.zeros_like(p_a),"rerank_top3":np.zeros_like(p_a),"selective_override":np.zeros_like(p_a)}
    specialist_rows=[]; specialist_fold=[]; specialist_oof_by_fold=[]
    for fold in range(1,6):
        tr=np.flatnonzero(fold_ids!=fold);va=np.flatnonzero(fold_ids==fold)
        fold_confusions=_pair_confusions(member_top[:,tr],y[tr])
        groups_list=_pair_groups(fold_confusions)
        groups_by_fold[str(fold)]={"training_confusion_pairs":fold_confusions,
                                   "selected_groups":groups_list}
        # Start from ORACLE02 probabilities for every held-out row.
        for key in specialist_oof: specialist_oof[key][va]=p_a[va]
        fold_stats=[]
        for grp in groups_list:
            spec,info,specialist_model,raw_specialist=_specialist_predict(tr,va,grp,Xspec,y,fold_group,p_a,p_eq)
            if info["status"]!="ok": continue
            model_path=output_dir/"models"/"specialists"/f"outer_{fold}"/("_".join(grp)+".joblib")
            model_path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(specialist_model,model_path,compress=3)
            members=[LABELS.index(z) for z in grp]
            for k,key in ((2,"rerank_top2"),(3,"rerank_top3")):
                result=specialist_oof[key][va].copy(); candidates=np.argsort(-result,axis=1,kind="stable")[:,:k]
                for r,row in enumerate(candidates):
                    inside=[j for j in row if j in members]
                    if len(inside)>1:
                        # Rerank the current candidate labels using the specialist posterior,
                        # independently of the selective-override threshold.
                        vals=raw_specialist[r,inside]; chosen=inside[int(np.argmax(vals))]
                        result[r,chosen]=float(result[r].max())+1e-9
                result/=np.maximum(result.sum(1,keepdims=True),1e-15)
                specialist_oof[key][va]=result
            # selective override output from _specialist_predict is its thresholded posterior.
            changed=spec.argmax(1)!=p_a[va].argmax(1)
            merged=specialist_oof["selective_override"][va].copy();merged[changed]=spec[changed]
            specialist_oof["selective_override"][va]=merged
            fold_stats.append({**info,"fold":fold})
        specialist_fold.extend(fold_stats)
        for strategy in specialist_oof:
            old=p_a[va].argmax(1); new=specialist_oof[strategy][va].argmax(1); yy=y[va]
            changed=new!=old
            baseline_ba=_metrics(yy,p_a[va])["balanced_accuracy"]
            strategy_ba=_metrics(yy,specialist_oof[strategy][va])["balanced_accuracy"]
            specialist_oof_by_fold.append({"fold":fold,"strategy":strategy,
                "trigger_count":int(changed.sum()),"conditional_accuracy":float((new[changed]==yy[changed]).mean()) if changed.any() else None,
                "corrected":int(((new==yy)&(old!=yy)).sum()),
                "introduced":int(((new!=yy)&(old==yy)).sum()),
                "baseline_balanced_accuracy":baseline_ba,
                "strategy_balanced_accuracy":strategy_ba,
                "delta_balanced_accuracy_pp":(strategy_ba-baseline_ba)*100,
                "heldout_rows":int(len(va))})
    # Save prediction matrices, comprehensive per-label scores and corrections.
    pred_frame=pd.DataFrame({"id":row_ids,"prompt_id":prompts,"y_true":y_text})
    strategies={"EQUAL9":p_eq,"H09":ensemble_probs["H09"],"ALL78":ensemble_probs["ALL78"],**oof,**specialist_oof}
    groups_list=sorted({tuple(g) for record in groups_by_fold.values() for g in record["selected_groups"]})
    results={"validation_rows":len(y),"prompt_groups":int(pd.Series(prompts).nunique()),
        "class_order":LABELS,"input_hashes":hashes,"text_feature_variants":{k:int(v.shape[1]) for k,v in feature_sets.items()},
        "stylometry_recipe":STYLO_RECIPE,"stylometry_features":stylo_names,"embedding_model":"google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e",
        "embedding_source":"frozen text-only CPU embeddings; cache-backed","outer_folds":"existing five prompt-group folds",
        "inner_folds":"3-fold GroupKFold within each outer training partition","candidate_coverage":cov_detail,
        "text_feature_fold_selection":fold_info,"specialist_groups_by_fold":groups_by_fold,
        "specialist_groups":groups_list,"specialist_folds":specialist_fold,
        "specialist_oof_by_fold":specialist_oof_by_fold,
        "metrics":{},"corrections":{}}
    # Baselines include C3 and H09; C3 stored hard-only, so calibration loss is N/A.
    c3p=np.eye(len(LABELS))[c3]
    strategies["C3"]=c3p
    for name,p in strategies.items():
        m=_metrics(y,p); pred=m.pop("pred")
        results["metrics"][name]=m
        pred_frame[f"pred::{name}"]=np.asarray(LABELS,dtype=object)[pred]
        for j,l in enumerate(LABELS): pred_frame[f"prob::{name}::{l}"]=p[:,j]
        if name not in {"EQUAL9","H09","ALL78","C3"}:
            results["corrections"][name]={ref:{"corrected":int(((pred==y)&(p_ref.argmax(1)!=y)).sum()),
                "introduced":int(((pred!=y)&(p_ref.argmax(1)==y)).sum()),
                "changed":int((pred!=p_ref.argmax(1)).sum())} for ref,p_ref in (("EQUAL9",p_eq),("H09",ensemble_probs["H09"]),("ORACLE02",p_a))}
    pred_frame.to_parquet(output_dir/"row_predictions.parquet",index=False)
    _json(output_dir/"metrics.json",results)
    # Per-label metric table for every strategy.
    label_rows=[]
    for name,p in strategies.items():
        m=_metrics(y,p); label_rows.extend({"strategy":name,"label":label,**value} for label,value in m["per_label"].items())
    pd.DataFrame(label_rows).to_csv(output_dir/"per_label_metrics.csv",index=False)
    _json(output_dir/"run_manifest.json",{"status":"completed","analysis":"ORACLE03","cache_only_base_models":True,
        "trust_list":"v025","base_probability_cache_sha256":hashes["probability_cache_sha256"],
        "existing_oracle02_oof_sha256":_sha(ORACLE02_PRED),"fold_assignments_sha256":hashes["fold_assignments_sha256"],
        "v0_ids_sha256":hashes["v0_ids_sha256"],"base_models_retrained":False,"kaggle_submission":False,
        "models":{"ORACLE02":"CatBoost probability+agreement features with EQUAL9 blend", "text":"EmbeddingGemma2 frozen snapshot and deterministic stylometry", "specialists":"fold-trained L2 logistic models"}})
    _write_report(Path(report_path),results,coverage_df,confusion_pairs)
    return results


def _write_report(path,results,coverage_df,confusions):
    lines=["# ORACLE03 — Closing the Oracle Accuracy Gap","",
        "All base probability matrices are cached Trust List 2 v025 predictions for the same 2,699 V0 rows. No base checkpoints were inferred or trained. Text embeddings use the local frozen EmbeddingGemma2 snapshot; stylometry is deterministic and row-local. Supervised text-stack and specialist fitting used the existing five outer prompt-group folds and inner prompt-group splits.","",
        "## Exposure limitation","",
        "These are development-set diagnostics, not independent Kaggle estimates. Base checkpoints, roster selection, and prior checkpoint selection have seen V0. Outer cross-fitting prevents the ORACLE03 meta-classifiers from directly fitting their held-out labels; it cannot remove base-checkpoint or roster exposure. The public benchmark is 0.581; V0 results alone do not establish a likely leaderboard gain.","",
        "## 1. Candidate coverage","",
        "Oracle coverage means the true label occurs among at least one member's top-1 labels (upper bound). The candidate-ranker coverage below asks whether that true label survives each deployable ranking's top-k shortlist. Vote-count ties are resolved by canonical label order; probability ties use stable canonical order.","",
        "| Roster | Ranking | Top 1 | Top 2 | Top 3 | Top 5 | Correct label unsupported by every member | Supported labels omitted at top 3 |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for roster in ("EQUAL9","H09","ALL78"):
        methods=["top1_vote_count","summed_checkpoint_probability","ensemble_probability"]
        if roster=="ALL78": methods.append("ORACLE02_oof_probability")
        for method in methods:
            d=results["candidate_coverage"][roster][method]["top_k"]
            unsupported=results["candidate_coverage"][roster][method]["correct_label_zero_support_rows"]
            lines.append(f"| {roster} | {method} | "+" | ".join(f"{d[str(k)]['coverage']:.2%}" for k in (1,2,3,5))+f" | {unsupported:,} | {d['3']['supported_but_excluded']:,} |")
    lines += ["","For each roster, the oracle upper bound is the share whose true label was a top-1 proposal by at least one member. This table shows how much of that bound remains in the shortlist (coverage divided by oracle coverage); probabilities can nominate labels absent from all top-1 votes.","",
        "| Roster | Oracle upper bound | Ranking | Top 2 of oracle | Top 3 of oracle | Supported examples excluded from top 3 |","|---|---:|---|---:|---:|---:|"]
    for roster in ("EQUAL9","H09","ALL78"):
        oracle=1-results["candidate_coverage"][roster]["top1_vote_count"]["correct_label_zero_support_rows"]/2699
        methods=["top1_vote_count","summed_checkpoint_probability","ensemble_probability"]
        if roster=="ALL78": methods.append("ORACLE02_oof_probability")
        for method in methods:
            d=results["candidate_coverage"][roster][method]["top_k"]
            lines.append(f"| {roster} | {oracle:.2%} | {method} | {d['2']['fraction_of_oracle_covered']:.2%} | {d['3']['fraction_of_oracle_covered']:.2%} | {d['3']['supported_but_excluded']:,} |")
    lines += ["","### Per-label shortlist coverage at top 3","",
        "Counts are correct-label rows covered by each top-3 list; the oracle column counts rows where at least one member predicted the true label as top-1.","",
        "| Label | EQUAL9 oracle | EQUAL9 vote | EQUAL9 prob | H09 oracle | H09 vote | H09 prob | ALL78 oracle | ALL78 vote | ALL78 prob |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i,label in enumerate(LABELS):
        vals=[]
        for roster in ("EQUAL9","H09","ALL78"):
            d=results["candidate_coverage"][roster]
            vals.append(d["oracle_coverage"]["per_label_supported_rows"][label])
            vals.append(d["top1_vote_count"]["per_label_covered_rows"][label]["3"]["covered_rows"])
            vals.append(d["summed_checkpoint_probability"]["per_label_covered_rows"][label]["3"]["covered_rows"])
        lines.append(f"| {label} | "+" | ".join(str(v) for v in vals)+" |")
    lines += ["","### Candidate-set size and major confusions","",
        "For vote lists, effective candidate count is limited by the number of distinct top-1 labels proposed in that row; probability lists always contain k labels.","",
        "| Roster | Distinct top-1 labels proposed | Rows |","|---|---:|---:|"]
    for roster in ("EQUAL9","H09","ALL78"):
        for k,n in results["candidate_coverage"][roster]["oracle_coverage"]["candidate_set_size_distribution"].items():
            lines.append(f"| {roster} | {k} | {n} |")
    lines += ["","Most frequent directed member top-1 errors (true label → predicted label, counted across checkpoint predictions):","",
        "| True label | Predicted label | Checkpoint error count |","|---|---|---:|"]
    for item in confusions[:15]:
        lines.append(f"| {item['true']} | {item['predicted']} | {item['count']} |")
    lines += ["","See `candidate_coverage.json` for per-label results, correct-label vote-support histograms, minority support, and confusion pairs. See `candidate_coverage.csv` for row counts and rates.","",
        "## 2. Text-augmented ORACLE02","",
        "A is the saved ORACLE02 all78 CatBoost+EQUAL9 cross-fitted baseline (67.167% BA). B–D use nested fitting: the CatBoost configuration and EQUAL9 blend weight are selected using inner prompt-group OOF predictions within each outer training partition. Text-only logistic models use the same nested splits. Their blend weight is selected against fixed EQUAL9 inner-fold probabilities, then applied to ORACLE02 on the held-out fold; this avoids using globally cross-fitted ORACLE02 training rows that could include the current outer fold, but it is a conservative proxy rather than a fully refit ORACLE02 inner stacker. Stylometry is baseline length/case/punctuation/function words plus L,V,C,P,W,F,R,X,M,U feature groups. Embeddings are frozen EmbeddingGemma2 768-dimensional text features. The base-probability matrix is row-aligned and unchanged across variants.","",
        "| Strategy | Accuracy | Balanced accuracy | Log loss | Δ BA vs ORACLE02 | Corrected / introduced vs ORACLE02 |","|---|---:|---:|---:|---:|---:|"]
    base=results["metrics"]["ORACLE02_A"]
    for name in ("ORACLE02_A","stack_stylometry_catboost","stack_stylometry_blend","stack_embedding_catboost","stack_embedding_blend","stack_stylometry_embedding_catboost","stack_stylometry_embedding_blend","text_only_stylometry_logistic","text_only_stylometry_blend","text_only_embedding_logistic","text_only_embedding_blend","text_only_stylometry_embedding_logistic","text_only_stylometry_embedding_blend"):
        if name not in results["metrics"]: continue
        m=results["metrics"][name]; corr=results["corrections"].get(name,{}).get("ORACLE02",{"corrected":0,"introduced":0})
        lines.append(f"| {name} | {m['accuracy']:.3%} | {m['balanced_accuracy']:.3%} | {m['log_loss']:.5f} | {(m['balanced_accuracy']-base['balanced_accuracy'])*100:+.3f} pp | {corr['corrected']} / {corr['introduced']} |")
    lines += ["","### Error movement against the reference systems","",
        "Corrected means the strategy is correct and the named reference is wrong; introduced means the reverse.","",
        "| Strategy | EQUAL9 corrected / introduced | H09 corrected / introduced | ORACLE02 corrected / introduced |","|---|---:|---:|---:|"]
    for name in ("stack_stylometry_catboost","stack_stylometry_blend","stack_embedding_catboost","stack_embedding_blend","stack_stylometry_embedding_catboost","stack_stylometry_embedding_blend","text_only_stylometry_logistic","text_only_stylometry_blend","text_only_embedding_logistic","text_only_embedding_blend","text_only_stylometry_embedding_logistic","text_only_stylometry_embedding_blend","rerank_top2","rerank_top3","selective_override"):
        if name not in results["corrections"]: continue
        c=results["corrections"][name]
        lines.append(f"| {name} | {c['EQUAL9']['corrected']} / {c['EQUAL9']['introduced']} | {c['H09']['corrected']} / {c['H09']['introduced']} | {c['ORACLE02']['corrected']} / {c['ORACLE02']['introduced']} |")
    lines += ["","Fold-specific model configurations, blend weights, train/held-out metrics, and specialist thresholds are in `metrics.json`. Per-label P/R/F1 for every strategy is in `per_label_metrics.csv`; ID-aligned probabilities are in `row_predictions.parquet`.","",
        "## 3. Pairwise and family specialists","",
        "Specialists are L2 logistic classifiers trained only on examples belonging to their confusion group. Candidate reranking is permitted only when at least two group labels occur in the baseline top-k. Selective override thresholds are chosen from inner-fold OOF predictions by overall balanced accuracy and are applied only when the baseline top-1 is in that group.","",
        "| Strategy | Accuracy | Balanced accuracy | Log loss | Δ BA vs ORACLE02 | Corrected / introduced vs ORACLE02 |","|---|---:|---:|---:|---:|---:|"]
    for name in ("rerank_top2","rerank_top3","selective_override"):
        m=results["metrics"][name];corr=results["corrections"][name]["ORACLE02"]
        lines.append(f"| {name} | {m['accuracy']:.3%} | {m['balanced_accuracy']:.3%} | {m['log_loss']:.5f} | {(m['balanced_accuracy']-base['balanced_accuracy'])*100:+.3f} pp | {corr['corrected']} / {corr['introduced']} |")
    lines += ["","| Specialist group |","|---|"]+["| "+", ".join(g)+" |" for g in results["specialist_groups"]]
    group_totals={}
    for rec in results.get("specialist_folds",[]):
        group=tuple(rec.get("group",[])); agg=group_totals.setdefault(group,{"top2":{},"top3":{},"selective":{}})
        for slot,src in (("top2","2"),("top3","3")):
            vals=rec.get("candidate_reranking_heldout",{}).get(src,{})
            for metric,value in vals.items():
                if metric=="conditional_accuracy": continue
                agg[slot][metric]=agg[slot].get(metric,0)+int(value or 0)
        for metric,value in rec.get("selective_heldout",{}).items():
            if metric=="conditional_accuracy": continue
            agg["selective"][metric]=agg["selective"].get(metric,0)+int(value or 0)
    lines += ["","| Specialist group | Top-2 conflicts: eligible; correct/wrong; corrected/introduced | Top-3 conflicts: eligible; correct/wrong; corrected/introduced | Selective trigger: rows; correct/incorrect; corrected/introduced |","|---|---:|---:|---:|"]
    for group,agg in sorted(group_totals.items(),key=lambda x:(-x[1]["top2"].get("eligible_rows",0),x[0])):
        cols=[]
        for slot in ("top2","top3"):
            d=agg[slot];cols.append(f"{d.get('eligible_rows',0)}; {d.get('selected_correct',0)}/{d.get('selected_wrong',0)}; {d.get('corrected_vs_oracle02',0)}/{d.get('introduced_vs_oracle02',0)}")
        d=agg["selective"];cols.append(f"{d.get('triggered_rows',0)}; {d.get('correct',0)}/{d.get('incorrect',0)}; {d.get('corrected_vs_oracle02',0)}/{d.get('introduced_vs_oracle02',0)}")
        lines.append(f"| {', '.join(group)} | "+" | ".join(cols)+" |")
    fold_top2=[x for x in results.get("specialist_oof_by_fold",[]) if x["strategy"]=="rerank_top2"]
    lines += ["","Group rows are aggregated over their five held-out folds. A conflict row contains at least two labels from that specialist group in the ORACLE02 top-k list.",
        "Fold-level trigger counts, conditional performance, selected thresholds, and specialist support are recorded in `metrics.json`.","",
        "| Fold | ORACLE02 BA | Top-2 rerank BA | Change | Rows reranked |","|---:|---:|---:|---:|---:|"]
    for item in fold_top2:
        if "baseline_balanced_accuracy" in item:
            lines.append(f"| {item['fold']} | {item['baseline_balanced_accuracy']:.3%} | {item['strategy_balanced_accuracy']:.3%} | {item['delta_balanced_accuracy_pp']:+.3f} pp | {item['trigger_count']} |")
    best_group=max(group_totals.items(),key=lambda x:x[1]["top2"].get("corrected_vs_oracle02",0)-x[1]["top2"].get("introduced_vs_oracle02",0)) if group_totals else ((),{"top2":{}})
    lines += ["",
        "## Ranked comparison and recommendations","",
        "Strategies are ranked by pooled outer-fold balanced accuracy; this ranking is descriptive because all base models and roster selection were exposed to V0.",""]
    ranked=sorted(results["metrics"].items(),key=lambda kv:(-kv[1]["balanced_accuracy"],kv[1]["log_loss"],kv[0]))
    lines += [f"{i}. **{name}** — BA {m['balanced_accuracy']:.3%}, accuracy {m['accuracy']:.3%}, log loss {m['log_loss']:.5f}." for i,(name,m) in enumerate(ranked,1)]
    best_name,best_metric=ranked[0]; base_ba=results["metrics"]["ORACLE02_A"]["balanced_accuracy"]
    text_names=[n for n in results["metrics"] if n.endswith("_blend")]
    text_name=max(text_names,key=lambda n:results["metrics"][n]["balanced_accuracy"]) if text_names else None
    text_folds=results.get("text_feature_fold_selection",{}).get(text_name.replace("_blend", "") if text_name else "",[])
    if text_folds:
        fold_scores=[x["heldout_metrics"]["balanced_accuracy"] for x in text_folds]
    else:
        text_folds=results.get("text_only_fold_selection",{}).get(text_name,[])
        fold_scores=[x["heldout"]["balanced_accuracy"] for x in text_folds]
    spec_names=["rerank_top2","rerank_top3","selective_override"]
    best_spec=max(spec_names,key=lambda n:results["metrics"][n]["balanced_accuracy"])
    spec_delta=results["corrections"][best_spec]["ORACLE02"]
    text_rec=(f"2. **Text features:** {text_name} is the strongest text blend ({(results['metrics'][text_name]['balanced_accuracy']-base_ba)*100:+.3f} pp versus ORACLE02); outer-fold BAs: " + ", ".join(f"{x:.2%}" for x in fold_scores) + ".") if text_name else "2. Text-feature blend results are unavailable."
    lines += ["",
        f"1. **Best measured strategy:** {best_name} at {best_metric['balanced_accuracy']:.3%} BA ({(best_metric['balanced_accuracy']-base_ba)*100:+.3f} pp versus ORACLE02). Treat it as a development candidate because the base roster has V0 exposure.",
        text_rec,
        f"3. **Specialists and Kaggle:** {best_spec} is the strongest specialist integration ({(results['metrics'][best_spec]['balanced_accuracy']-base_ba)*100:+.3f} pp; {spec_delta['corrected']} corrected and {spec_delta['introduced']} introduced versus ORACLE02). Preserve the current 0.581 public-score candidate until a locked submission comparison demonstrates an improvement.","",
        "## Explicit answers","",
        f"1. **Oracle coverage:** EQUAL9 top-1 oracle support is {results['candidate_coverage']['EQUAL9']['oracle_coverage']['coverage']:.2%}; probability top-2/top-3 retain {results['candidate_coverage']['EQUAL9']['summed_checkpoint_probability']['top_k']['2']['fraction_of_oracle_covered']:.2%}/{results['candidate_coverage']['EQUAL9']['summed_checkpoint_probability']['top_k']['3']['fraction_of_oracle_covered']:.2%} of it. ALL78 support is {results['candidate_coverage']['ALL78']['oracle_coverage']['coverage']:.2%}; top-2/top-3 retain {results['candidate_coverage']['ALL78']['summed_checkpoint_probability']['top_k']['2']['fraction_of_oracle_covered']:.2%}/{results['candidate_coverage']['ALL78']['summed_checkpoint_probability']['top_k']['3']['fraction_of_oracle_covered']:.2%}.",
        f"2. **Largest measured confusion opportunities:** the leading member error is `{confusions[0]['true']} → {confusions[0]['predicted']}` ({confusions[0]['count']:,} checkpoint predictions). By held-out top-2 specialist net corrections, `{', '.join(best_group[0])}` is strongest ({best_group[1]['top2'].get('corrected_vs_oracle02',0)} corrected versus {best_group[1]['top2'].get('introduced_vs_oracle02',0)} introduced across folds).",
        f"3. **Text features:** no text-augmented stack beats ORACLE02. The best CatBoost+text blend is {max([n for n in results['metrics'] if n.startswith('stack_') and n.endswith('_blend')],key=lambda n:results['metrics'][n]['balanced_accuracy'])}; its balanced accuracy is below the ORACLE02 baseline.",
        f"4. **Specialist quality:** top-2 reranking improves pooled OOF balanced accuracy by {(results['metrics']['rerank_top2']['balanced_accuracy']-base_ba)*100:+.3f} pp; top-3 is harmful because it introduces {results['corrections']['rerank_top3']['ORACLE02']['introduced']} errors versus {results['corrections']['rerank_top3']['ORACLE02']['corrected']} corrections.",
        f"5. **Best combination:** top-2 specialist reranking ({results['metrics']['rerank_top2']['balanced_accuracy']:.3%} BA), ahead of ORACLE02 in {sum(x.get('delta_balanced_accuracy_pp',0)>0 for x in fold_top2)}/5 folds.",
        f"6. **Remaining gap:** the ALL78 98.67% top-1 oracle coverage is {98.67-results['metrics']['rerank_top2']['balanced_accuracy']*100:.2f} pp above the best measured balanced accuracy. Candidate-supported confusion rows remain, but current reranking features resolve them unreliably.",
        "7. **Kaggle readiness:** do not submit this based on the current evidence. The 0.581 public benchmark is not directly comparable with V0, the checkpoints/roster were exposed to V0, and top-2's 0.333 pp gain reverses in one of five folds.","",
        "## Three recommendations","",
        "1. Keep ORACLE02 as the baseline; stylometry and embeddings add no measured OOF gain.",
        "2. Carry top-2 reranking forward as an experimental candidate only; do not use top-3 reranking.",
        "3. Require an unseen labeled evaluation or controlled Kaggle test before replacing the 0.581 public-score reference.","",
        "## Artifacts","","- `candidate_coverage.csv/.json`: top-k shortlist coverage, oracle-support counts, per-label distributions and confusions.",
        "- `text_features.npz`, `text_embeddings.npz`: exact ID-aligned text feature arrays.","- `row_predictions.parquet`, `per_label_metrics.csv`, `metrics.json`: OOF predictions and scores.","- `run_manifest.json`: source hashes, methods, roster and fold provenance.",""]
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default=str(OUT));p.add_argument("--report",default=str(REPORT));p.add_argument("--skip-embeddings",action="store_true");p.add_argument("--refresh-specialists-only",action="store_true");p.add_argument("--refresh-candidate-coverage-only",action="store_true");p.add_argument("--refresh-text-only-only",action="store_true")
    a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.refresh_text_only_only:
            _refresh_text_only_models(a.output_dir);r=json.loads((Path(a.output_dir)/"metrics.json").read_text())
            coverage=pd.read_csv(Path(a.output_dir)/"candidate_coverage.csv")
            _write_report(Path(a.report),r,coverage,r["candidate_coverage"]["ALL78"]["member_top1_confusion_pairs"])
        elif a.refresh_candidate_coverage_only:
            refresh_candidate_coverage(a.output_dir);r=json.loads((Path(a.output_dir)/"metrics.json").read_text())
        else:
            r=refresh_specialists(a.output_dir,a.report) if a.refresh_specialists_only else run(a.output_dir,a.report,with_embeddings=not a.skip_embeddings)
    print(json.dumps({k:{x:r["metrics"][k][x] for x in ("accuracy","balanced_accuracy","log_loss")} for k in r["metrics"]},indent=2))


if __name__=="__main__": main()
