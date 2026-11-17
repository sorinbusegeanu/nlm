"""ORACLE02 supervised stacking over cached Trust List 2 predictions.

All outer and inner splits are grouped by prompt. Base checkpoints are never
loaded; only the saved full-V0 probability cache is consumed.
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

from .ensemble_precision_recall_overlap import EQUAL9
from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("artifacts/analysis/oracle02")
REPORT = Path("docs/oracle02.md")
V0_FOLDS = Path("artifacts/analysis/ensemble_cascade_comparison/fold_assignments.csv")
BASE_CACHE = Path("artifacts/analysis/ensemble_precision_recall_overlap/checkpoint_probabilities.npz")
BASE_CACHE_META = Path("artifacts/analysis/ensemble_precision_recall_overlap/prediction_cache_manifest.json")
C3_CACHE = Path("artifacts/analysis/ensemble_cascade_comparison/cascade_predictions.parquet")
LOGISTIC_C = [0.1, 1.0, 10.0]
CATBOOST_GRID = [
    {"depth": 3, "iterations": 120, "l2_leaf_reg": 10.0, "rsm": 0.25},
    {"depth": 4, "iterations": 120, "l2_leaf_reg": 20.0, "rsm": 0.25},
]
BLEND_GRID = [float(round(float(x), 1)) for x in np.linspace(0.0, 1.0, 11)]
ROSTERS = {"equal9": EQUAL9, "all78": None}
STRATEGIES = ["logistic", "catboost", "catboost_equal9_blend"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _canonical_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    tp = np.diag(cm); fp = cm.sum(axis=0) - tp; support = cm.sum(axis=1)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {"rows": int(len(y)), "correct": int(np.sum(pred == y)),
        "accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(recall.mean()),
        "log_loss": float(log_loss(y, probabilities, labels=np.arange(len(LABELS)))),
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(fp[i]),
            "fn": int(support[i] - tp[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)}, "y_pred": pred}


def _hard_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    onehot = np.eye(len(LABELS), dtype=float)[pred]
    return _canonical_metrics(y, onehot * (1 - 1e-12) + (1 - onehot) * (1e-12 / (len(LABELS)-1)))


def _delta(pred: np.ndarray, reference: np.ndarray, y: np.ndarray) -> dict:
    a = pred == y; b = reference == y
    return {"corrected": int(np.sum(~b & a)), "introduced": int(np.sum(b & ~a)),
        "disagreements": int(np.sum(pred != reference)),
        "disagreement_rate": float(np.mean(pred != reference))}


def _build_features(checkpoint_ids: list[str], matrices: dict[str, np.ndarray],
                    equal9_probabilities: np.ndarray) -> tuple[np.ndarray, list[str]]:
    parts=[]; names=[]; top1=[]
    eye=np.eye(len(LABELS),dtype=np.float32)
    for sid in checkpoint_ids:
        p=np.asarray(matrices[sid],dtype=np.float32)
        order=np.argsort(-p,axis=1,kind="stable")
        top=order[:,0]; top1.append(top)
        parts.append(p); names.extend([f"{sid}::prob::{label}" for label in LABELS])
        parts.append(eye[top]); names.extend([f"{sid}::top1::{label}" for label in LABELS])
        p1=p[np.arange(len(p)),order[:,0]]; p2=p[np.arange(len(p)),order[:,1]]
        margin=(p1-p2)[:,None]
        entropy=(-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1))[:,None]
        scalar=np.column_stack((p1,margin[:,0],entropy[:,0])).astype(np.float32)
        parts.append(scalar); names.extend([f"{sid}::max_probability",f"{sid}::top1_top2_margin",f"{sid}::entropy"])
    votes=np.stack(top1,axis=1)
    agreement=np.stack([(votes==ci).sum(axis=1) for ci in range(len(LABELS))],axis=1).astype(np.float32)
    parts.append(agreement); names.extend([f"agreement_count::{label}" for label in LABELS])
    parts.append(np.asarray(equal9_probabilities,dtype=np.float32)); names.extend([f"equal9_prob::{label}" for label in LABELS])
    X=np.concatenate(parts,axis=1).astype(np.float32,copy=False)
    if X.shape[1]!=len(names) or not np.isfinite(X).all(): raise ValueError("Invalid stacking feature matrix")
    return X,names


def _align_proba(model, X: np.ndarray) -> np.ndarray:
    raw=np.asarray(model.predict_proba(X),dtype=np.float64)
    out=np.zeros((len(X),len(LABELS)),dtype=np.float64)
    for j,cls in enumerate(model.classes_): out[:,int(cls)]=raw[:,j]
    return out/np.maximum(out.sum(axis=1,keepdims=True),1e-15)


def _fit_logistic(X: np.ndarray,y: np.ndarray,C: float):
    scaler=StandardScaler()
    transformed=scaler.fit_transform(X)
    clf=LogisticRegression(C=C,solver="lbfgs",max_iter=500,random_state=2026)
    clf.fit(transformed,y)
    return (scaler,clf)


def _predict_logistic(bundle, X: np.ndarray) -> np.ndarray:
    scaler,clf=bundle
    return _align_proba(clf,scaler.transform(X))


def _fit_catboost(X: np.ndarray,y: np.ndarray,config: dict) -> CatBoostClassifier:
    model=CatBoostClassifier(loss_function="MultiClass",eval_metric="MultiClass",
        depth=int(config["depth"]),iterations=int(config["iterations"]),
        learning_rate=0.05,l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_seed=2026,thread_count=1,task_type="CPU",verbose=False,
        allow_writing_files=False,random_strength=0.5,bootstrap_type="Bernoulli",
        subsample=0.85,border_count=64,rsm=float(config.get("rsm",0.25)))
    model.fit(X,y)
    return model


def _inner_splits(train_idx: np.ndarray, groups: np.ndarray, n_splits: int=3):
    local_groups=groups[train_idx]
    return [(train_idx[a],train_idx[b]) for a,b in GroupKFold(n_splits=n_splits).split(
        np.zeros(len(train_idx)),groups=local_groups)]


def _score_probs(y: np.ndarray, p: np.ndarray) -> tuple[float,float]:
    m=_canonical_metrics(y,p)
    return m["balanced_accuracy"],m["log_loss"]


def _tune_logistic(X: np.ndarray,y: np.ndarray,groups: np.ndarray,
                   outer_train: np.ndarray) -> tuple[float,dict]:
    splits=_inner_splits(outer_train,groups)
    scored=[]
    for C in LOGISTIC_C:
        oof=np.zeros((len(outer_train),len(LABELS)),dtype=np.float64)
        pos={int(g):i for i,g in enumerate(outer_train)}
        for inner_train,inner_val in splits:
            bundle=_fit_logistic(X[inner_train],y[inner_train],C)
            oof[[pos[int(x)] for x in inner_val]]=_predict_logistic(bundle,X[inner_val])
        ba,ll=_score_probs(y[outer_train],oof)
        scored.append({"C":C,"balanced_accuracy":ba,"log_loss":ll})
    best=min(scored,key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["C"]))
    return float(best["C"]),{"grid":scored,"selected":best}


def _tune_catboost(X: np.ndarray,y: np.ndarray,groups: np.ndarray,
                   outer_train: np.ndarray,equal9_probs: np.ndarray) -> tuple[dict,float,dict]:
    splits=_inner_splits(outer_train,groups)
    pos={int(g):i for i,g in enumerate(outer_train)}
    scored=[]; pred_by_config={}
    for config in CATBOOST_GRID:
        oof=np.zeros((len(outer_train),len(LABELS)),dtype=np.float64)
        for inner_train,inner_val in splits:
            model=_fit_catboost(X[inner_train],y[inner_train],config)
            oof[[pos[int(x)] for x in inner_val]]=_align_proba(model,X[inner_val])
        ba,ll=_score_probs(y[outer_train],oof)
        rec={**config,"balanced_accuracy":ba,"log_loss":ll}
        scored.append(rec);pred_by_config[tuple(config[k] for k in ("depth","iterations","l2_leaf_reg","rsm"))]=oof
    best=min(scored,key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["depth"],r["l2_leaf_reg"]))
    config={k:best[k] for k in ("depth","iterations","l2_leaf_reg","rsm")}
    selected_oof=pred_by_config[tuple(config[k] for k in ("depth","iterations","l2_leaf_reg","rsm"))]
    eq=equal9_probs[outer_train]
    blend_scores=[]
    for alpha in BLEND_GRID:
        blend=(1-alpha)*eq+alpha*selected_oof
        blend/=blend.sum(axis=1,keepdims=True)
        ba,ll=_score_probs(y[outer_train],blend)
        blend_scores.append({"catboost_weight":alpha,"equal9_weight":1-alpha,
                             "balanced_accuracy":ba,"log_loss":ll})
    best_blend=min(blend_scores,key=lambda r:(-r["balanced_accuracy"],r["log_loss"],r["catboost_weight"]))
    return config,float(best_blend["catboost_weight"]),{"grid":scored,"selected":best,
        "blend_grid":blend_scores,"selected_blend":best_blend}


def _source_inputs(root:Path, output:Path, resume:bool):
    trust_meta=json.loads((root/"artifacts/analysis/ensemble_precision_recall_overlap/prediction_cache_manifest.json").read_text())
    trust_path=root/"docs/trust-list-2-latest.json"
    pointer=json.loads(trust_path.read_text())
    if trust_meta.get("trust_list_version")!=pointer.get("version"):
        raise ValueError("ORACLE01 probability cache does not match the latest Trust List 2")
    ids=trust_meta["checkpoint_ids"]
    if len(ids)!=78 or len(set(ids))!=78: raise ValueError(f"Expected 78 unique eligible checkpoint matrices, got {len(ids)}")
    trust_csv=root/pointer["csv_path"]
    if _sha(trust_csv)!=trust_meta.get("trust_sha256"):
        raise ValueError("ORACLE01 cache hash does not match the latest Trust List 2 CSV")
    trust=pd.read_csv(trust_csv,dtype=str,keep_default_na=False)
    truth=lambda s:s.astype(str).str.strip().str.lower().isin({"true","1","yes","y"})
    false=lambda s:s.astype(str).str.strip().str.lower().isin({"false","0","no","n",""})
    eligible=trust.loc[truth(trust.evaluation_eligible)&truth(trust.rule_eligible)&
        truth(trust.checkpoint_available)&false(trust.validation_training_overlap)&
        (pd.to_numeric(trust.validation_rows,errors="coerce")==2699),"sequence_id"].astype(str).unique()
    if set(ids)!=set(eligible): raise ValueError("Cached all78 roster differs from current eligible Trust List 2")
    npz_path=root/BASE_CACHE; npz=np.load(npz_path)
    matrices={sid:np.asarray(npz[f"p_{sid}"],dtype=np.float32) for sid in ids}
    v0=_dataset_rows("v0",root/"data/processed").sort_values("id",kind="stable").reset_index(drop=True)
    if len(v0)!=2699: raise ValueError(f"Expected 2699 V0 rows, got {len(v0)}")
    row_ids=v0.id.astype(str).tolist(); prompts=v0.prompt_id.astype(str).to_numpy()
    y_text=v0.model.astype(str).to_numpy(); y=np.asarray([LABELS.index(x) for x in y_text],dtype=np.int64)
    digest=hashlib.sha256("\n".join(row_ids).encode()).hexdigest()
    if digest!=trust_meta.get("v0_ids_sha256"): raise ValueError("V0 response ID hash differs from ORACLE01 cache")
    for sid,p in matrices.items():
        if p.shape!=(2699,len(LABELS)) or not np.isfinite(p).all(): raise ValueError(f"Bad probability matrix for {sid}")
        if not np.allclose(p.sum(axis=1),1,atol=1e-5): raise ValueError(f"Unnormalized probabilities: {sid}")
    folds=pd.read_csv(root/V0_FOLDS,dtype={"id":str,"prompt_id":str})
    if len(folds)!=2699 or folds.id.astype(str).tolist()!=row_ids or folds.prompt_id.astype(str).tolist()!=prompts.tolist():
        raise ValueError("Existing V0 prompt fold assignments do not exactly align by ID and prompt")
    fold_ids=folds.fold.to_numpy(dtype=int)
    if set(np.unique(fold_ids))!={1,2,3,4,5}: raise ValueError("Expected five existing prompt-group folds")
    c3=pd.read_parquet(root/C3_CACHE)
    if c3.id.astype(str).tolist()!=row_ids or c3.prompt_id.astype(str).tolist()!=prompts.tolist() or c3.y_true.astype(str).tolist()!=y_text.tolist():
        raise ValueError("C3 predictions do not align with the V0 rows")
    c3_pred=np.asarray([LABELS.index(x) for x in c3["pred_oof::C3_precision"].astype(str)],dtype=np.int64)
    equal9_p=np.mean([matrices[sid] for sid in EQUAL9],axis=0)
    equal9_p/=equal9_p.sum(axis=1,keepdims=True)
    if not np.array_equal(equal9_p.argmax(axis=1),np.asarray([LABELS.index(x) for x in c3.pred_equal9.astype(str)])):
        raise ValueError("EQUAL9 cached probabilities disagree with prior C3 comparison predictions")
    input_hashes={"trust_list_sha256":trust_meta.get("trust_sha256"),"probability_cache_sha256":_sha(npz_path),
        "fold_assignments_sha256":_sha(root/V0_FOLDS),"c3_predictions_sha256":_sha(root/C3_CACHE),
        "v0_ids_sha256":digest}
    return ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3_pred,equal9_p,input_hashes


def _fit_roster(roster_name:str, ids:list[str], matrices:dict[str,np.ndarray],
                X:np.ndarray,y:np.ndarray,groups:np.ndarray,fold_ids:np.ndarray,
                equal9_p:np.ndarray,output:Path,input_hashes:dict,resume:bool,code_hash:str):
    n=len(y);oof={name:np.zeros((n,len(LABELS)),dtype=np.float64) for name in STRATEGIES}
    train_metrics={name:[] for name in STRATEGIES};fold_details=[]
    fold_dir=output/"folds"/roster_name;fold_dir.mkdir(parents=True,exist_ok=True)
    for fold in range(1,6):
        train=np.flatnonzero(fold_ids!=fold);held=np.flatnonzero(fold_ids==fold)
        pred_path=fold_dir/f"outer_{fold}.npz";meta_path=fold_dir/f"outer_{fold}.json"
        if resume and pred_path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta.get("code_sha256")==code_hash and meta.get("input_hashes")==input_hashes:
                z=np.load(pred_path)
                if z["held_indexes"].tolist()==held.tolist() and z["train_indexes"].tolist()==train.tolist():
                    for strategy in STRATEGIES:
                        oof[strategy][held]=z[f"held_{strategy}"]
                        train_metrics[strategy].append(meta["training_metrics"][strategy])
                    fold_details.append(meta["details"]);continue
        C,tune_lr=_tune_logistic(X,y,groups,train)
        lr_model=_fit_logistic(X[train],y[train],C)
        lr_train=_predict_logistic(lr_model,X[train]);lr_held=_predict_logistic(lr_model,X[held])
        cb_config,alpha,tune_cb=_tune_catboost(X,y,groups,train,equal9_p)
        cb_model=_fit_catboost(X[train],y[train],cb_config)
        cb_train=_align_proba(cb_model,X[train]);cb_held=_align_proba(cb_model,X[held])
        blend_train=(1-alpha)*equal9_p[train]+alpha*cb_train
        blend_held=(1-alpha)*equal9_p[held]+alpha*cb_held
        blend_train/=blend_train.sum(axis=1,keepdims=True);blend_held/=blend_held.sum(axis=1,keepdims=True)
        train_probs={"logistic":lr_train,"catboost":cb_train,"catboost_equal9_blend":blend_train}
        held_probs={"logistic":lr_held,"catboost":cb_held,"catboost_equal9_blend":blend_held}
        for strategy in STRATEGIES:
            oof[strategy][held]=held_probs[strategy]
            train_metrics[strategy].append(_canonical_metrics(y[train],train_probs[strategy]) | {"y_pred":None})
        model_dir=fold_dir/f"models_outer_{fold}";model_dir.mkdir(exist_ok=True)
        joblib.dump(lr_model,model_dir/"logistic_pipeline.joblib",compress=3)
        cb_model.save_model(str(model_dir/"catboost.cbm"))
        details={"fold":fold,"train_rows":len(train),"heldout_rows":len(held),
            "train_prompt_groups":sorted(set(groups[train])),"heldout_prompt_groups":sorted(set(groups[held])),
            "selected_logistic_C":C,"logistic_inner_selection":tune_lr,
            "selected_catboost_config":cb_config,"catboost_inner_selection":tune_cb,
            "selected_blend_weights":{"catboost":alpha,"equal9":1-alpha},
            "model_artifacts":{"logistic_pipeline":str((model_dir/"logistic_pipeline.joblib").relative_to(output)),
                "catboost":str((model_dir/"catboost.cbm").relative_to(output))}}
        training_fold_metrics={s:{k:v for k,v in train_metrics[s][-1].items() if k!="per_label"} for s in STRATEGIES}
        # Keep full training metrics (including per-class values) in fold metadata.
        training_fold_metrics={s:train_metrics[s][-1] for s in STRATEGIES}
        np.savez_compressed(pred_path,train_indexes=train,held_indexes=held,
            **{f"held_{s}":held_probs[s] for s in STRATEGIES},
            **{f"train_{s}":train_probs[s] for s in STRATEGIES})
        _atomic_json(meta_path,{"code_sha256":code_hash,"input_hashes":input_hashes,
            "training_metrics":training_fold_metrics,"details":details})
        fold_details.append(details)
        _atomic_json(output/"analysis_state.json",{"status":"running","roster":roster_name,
            "completed_outer_folds":len(fold_details),"input_hashes":input_hashes})
    return oof,train_metrics,fold_details


def _summarize_train(fold_metrics:list[dict]) -> dict:
    keys=("accuracy","balanced_accuracy","log_loss")
    return {key:{"mean":float(np.mean([m[key] for m in fold_metrics])),
                 "std":float(np.std([m[key] for m in fold_metrics],ddof=0))} for key in keys}


def run(*,root:str|Path=ROOT,output_dir:str|Path=OUT,report_path:str|Path=REPORT,
        resume:bool=True) -> dict:
    root=Path(root).resolve();output=Path(output_dir)
    if not output.is_absolute():output=root/output
    output.mkdir(parents=True,exist_ok=True)
    report=Path(report_path)
    if not report.is_absolute():report=root/report
    code_hash=_sha(Path(__file__).resolve())
    (ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3_pred,equal9_p,input_hashes)=_source_inputs(root,output,resume)
    rosters={"equal9":EQUAL9,"all78":ids}
    feature_mats={};feature_names={}
    feature_path=output/"features.npz";feature_meta_path=output/"feature_manifest.json"
    feature_cache_ok=False
    if resume and feature_path.exists() and feature_meta_path.exists():
        fm=json.loads(feature_meta_path.read_text())
        feature_cache_ok=fm.get("input_hashes")==input_hashes and fm.get("code_sha256")==code_hash
        if feature_cache_ok:
            z=np.load(feature_path)
            for roster in rosters:feature_mats[roster]=z[f"X_{roster}"]
            feature_names=fm["feature_names"]
    if not feature_cache_ok:
        for roster,cpids in rosters.items():feature_mats[roster],feature_names[roster]=_build_features(cpids,matrices,equal9_p)
        np.savez_compressed(feature_path,**{f"X_{k}":v for k,v in feature_mats.items()})
        _atomic_json(feature_meta_path,{"input_hashes":input_hashes,"code_sha256":code_hash,
            "feature_count":{k:int(v.shape[1]) for k,v in feature_mats.items()},
            "feature_names":feature_names,
            "feature_definition":{"per_checkpoint":"18 probabilities; 18 one-hot top-1 labels; max probability, top1-top2 margin, entropy",
                "agreement":"per-class count of selected checkpoints whose top-1 equals that class",
                "equal9":"EQUAL9 mean probability vector included in both rosters"}})
    # On a resumed feature cache, names are already keyed per roster.
    if isinstance(feature_names,list): raise ValueError("Invalid feature manifest")
    results={"trust_list_version":"trust-list-2-v025","input_hashes":input_hashes,
        "validation_rows":len(y),"prompt_groups":int(pd.Series(prompts).nunique()),
        "class_order":LABELS,"roster_sizes":{"equal9":len(EQUAL9),"all78":len(ids)},
        "results":{},"nested_selection":{"outer":"fixed existing 5-fold prompt-group split",
            "inner":"3-fold GroupKFold(prompt_id) on outer training rows only",
            "logistic_C_grid":LOGISTIC_C,"catboost_grid":CATBOOST_GRID,
            "blend_catboost_weight_grid":BLEND_GRID,"selection_metric":"inner pooled balanced accuracy; log loss tie-break"}}
    pred_frame=pd.DataFrame({"id":row_ids,"prompt_id":prompts,"y_true":y_text,
        "pred_equal9":np.asarray(LABELS,dtype=object)[equal9_p.argmax(axis=1)],
        "pred_c3_oof":np.asarray(LABELS,dtype=object)[c3_pred]})
    for i,label in enumerate(LABELS):pred_frame[f"prob_equal9::{label}"]=equal9_p[:,i]
    baseline_equal=_canonical_metrics(y,equal9_p);equal_pred=baseline_equal.pop("y_pred")
    baseline_c3=_hard_metrics(y,c3_pred);baseline_c3.pop("y_pred",None);baseline_c3["log_loss"]=None
    results["baselines"]={"equal9_oof":baseline_equal,"c3_oof":baseline_c3,
        "c3_log_loss":"N/A; cached C3 artifact contains only hard predictions"}
    summary_by_strategy={}
    for roster,cpids in rosters.items():
        X=feature_mats[roster]
        oof,train_folds,fold_details=_fit_roster(roster,ids,matrices,X,y,prompts,fold_ids,
            equal9_p,output,input_hashes,resume,code_hash)
        results["results"][roster]={}
        for strategy in STRATEGIES:
            metric=_canonical_metrics(y,oof[strategy]);pred=metric.pop("y_pred")
            metric["delta_vs_equal9"]=_delta(pred,equal_pred,y)
            metric["delta_vs_c3"]=_delta(pred,c3_pred,y)
            train_summary=_summarize_train(train_folds[strategy])
            metric["training_performance_outer_fold_mean"]=train_summary
            results["results"][roster][strategy]={"heldout_oof":metric,
                "training_folds":[{k:v for k,v in m.items() if k!="y_pred"} for m in train_folds[strategy]],
                "outer_fold_selection":[{k:v for k,v in d.items() if k not in {"logistic_inner_selection","catboost_inner_selection"}} for d in fold_details]}
            summary_by_strategy[(roster,strategy)]=metric
            pred_frame[f"pred_oof::{roster}::{strategy}"]=np.asarray(LABELS,dtype=object)[pred]
            for ci,label in enumerate(LABELS):pred_frame[f"prob_oof::{roster}::{strategy}::{label}"]=oof[strategy][:,ci]
        results["results"][roster]["fold_tuning_details"]=fold_details
    # Baseline row-level labels and all cross-fitted meta probabilities.
    base_pred_c3=np.asarray(LABELS,dtype=object)[c3_pred]
    pred_frame.to_parquet(output/"row_predictions.parquet",index=False)
    per_label=[]
    for name,p in [("EQUAL9",equal9_p),("C3",np.eye(len(LABELS))[c3_pred])]:
        hard=p.argmax(axis=1);cm=confusion_matrix(y,hard,labels=np.arange(len(LABELS)))
        tp=np.diag(cm);fp=cm.sum(0)-tp;support=cm.sum(1)
        for i,label in enumerate(LABELS):
            pr=tp[i]/max(1,tp[i]+fp[i]);rc=tp[i]/max(1,support[i]);f=2*pr*rc/max(1e-15,pr+rc)
            per_label.append({"roster":"baseline","strategy":name,"label":label,"precision":pr,"recall":rc,"f1":f,"tp":int(tp[i]),"fp":int(fp[i]),"fn":int(support[i]-tp[i])})
    for roster in rosters:
        for strategy in STRATEGIES:
            m=summary_by_strategy[(roster,strategy)]
            for label,values in m["per_label"].items():per_label.append({"roster":roster,"strategy":strategy,"label":label,**values})
    pd.DataFrame(per_label).to_csv(output/"per_label_metrics.csv",index=False)
    _atomic_json(output/"metrics.json",results)
    _atomic_json(output/"run_manifest.json",{"status":"completed","version":"oracle02",
        "input_hashes":input_hashes,"trust_list_version":"trust-list-2-v025",
        "outer_fold_source":str(V0_FOLDS),"base_prediction_cache":str(BASE_CACHE),
        "c3_reference_cache":str(C3_CACHE),"baseline_checkpoint_ids":EQUAL9,
        "all_checkpoint_ids":ids,"class_order":LABELS,"feature_counts":{k:int(v.shape[1]) for k,v in feature_mats.items()},
        "hyperparameter_grids":results["nested_selection"],"models_saved":True})
    _write_report(report,results,summary_by_strategy,per_label,fold_details_by_roster={})
    _atomic_json(output/"analysis_state.json",{"status":"completed","input_hashes":input_hashes,
        "rosters_completed":list(rosters),"outer_folds":5})
    return results


def _write_report(path:Path,results:dict,summary:dict,per_label:list,fold_details_by_roster:dict)->None:
    lines=["# ORACLE02 — Learned stacking ensemble","",
        "A supervised meta-classifier over the saved V0 probability vectors from Trust List 2 v025. Base checkpoints were not loaded or retrained.","",
        "## Validation and exposure limitation","",
        "The 2,699 V0 rows are scored with the existing five prompt-group folds. Hyperparameter and blend weight selection use three inner prompt-group folds restricted to each outer training partition. Feature scalers are fitted inside each inner/final training fit. The 78-checkpoint and EQUAL9 rosters are fixed. The base checkpoints and Trust List roster were selected/developed using V0, so outer-fold stacking does not remove that exposure; results are diagnostics, not independent Kaggle estimates.","",
        "C3 is the existing EQUAL9 + three-rule cascade (`N0071:gpt-6-astra`, `N0105:gemini-3.1-pro-preview`, `N0048:claude-opus-4-6`) with EQUAL9 fallback. Its stored artifact provides OOF hard labels only, so its log loss is reported as N/A.","",
        "## Baselines","",
        "| Baseline | Accuracy | Balanced accuracy | Log loss | Correct / 2,699 |","|---|---:|---:|---:|---:|"]
    e=results["baselines"]["equal9_oof"];c=results["baselines"]["c3_oof"]
    lines.append(f"| EQUAL9 | {e['accuracy']:.4%} | {e['balanced_accuracy']:.4%} | {e['log_loss']:.6f} | {e['correct']:,} |")
    lines.append(f"| C3 | {c['accuracy']:.4%} | {c['balanced_accuracy']:.4%} | N/A | {c['correct']:,} |")
    lines += ["","## Cross-fitted held-out comparison","",
        "| Checkpoints | Meta-classifier | Accuracy | Balanced accuracy | Log loss | Δ BA vs EQUAL9 | Δ BA vs C3 | Corrected / introduced vs EQUAL9 | vs C3 | Disagreements vs EQUAL9 / C3 |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for roster in ("equal9","all78"):
        for strategy in STRATEGIES:
            m=summary[(roster,strategy)];eq=m["delta_vs_equal9"];c3=m["delta_vs_c3"]
            d_eq=(m['balanced_accuracy']-results['baselines']['equal9_oof']['balanced_accuracy'])*100
            # percentage-point deltas are easier to interpret than relative change.
            d_c3=(m['balanced_accuracy']-results['baselines']['c3_oof']['balanced_accuracy'])*100
            lines.append(f"| {9 if roster=='equal9' else 78} | {strategy} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['log_loss']:.6f} | {d_eq:+.3f} pp | {d_c3:+.3f} pp | {eq['corrected']} / {eq['introduced']} | {c3['corrected']} / {c3['introduced']} | {eq['disagreements']} / {c3['disagreements']} |")
    lines += ["","## Outer-fold training versus held-out results","",
        "Training columns are the mean apparent scores on each outer fold's training rows after inner-fold selection. Held-out columns pool each row's single outer-fold prediction.","",
        "| Checkpoints | Strategy | Train accuracy | Train balanced accuracy | Train log loss | Held-out accuracy | Held-out balanced accuracy | Held-out log loss |","|---:|---|---:|---:|---:|---:|---:|---:|"]
    for roster in ("equal9","all78"):
        for strategy in STRATEGIES:
            m=summary[(roster,strategy)];t=m["training_performance_outer_fold_mean"]
            lines.append(f"| {9 if roster=='equal9' else 78} | {strategy} | {t['accuracy']['mean']:.4%} ± {t['accuracy']['std']:.3%} | {t['balanced_accuracy']['mean']:.4%} ± {t['balanced_accuracy']['std']:.3%} | {t['log_loss']['mean']:.6f} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['log_loss']:.6f} |")
    lines += ["","## Per-label recall","",
        "P/R/F1, TP, FP and FN for every label and all six held-out experiments are in `artifacts/analysis/oracle02/per_label_metrics.csv`. Training-fold per-label metrics are recorded in `metrics.json`.","",
        "## Nested selection settings","",
        f"- Logistic regression: L2 multinomial, C grid `{LOGISTIC_C}`, max_iter 500, inner pooled balanced accuracy with log-loss tie-break.",
        f"- CatBoost: CPU, one thread, depth 3–4, 120 iterations, L2 leaf regularization 10 or 20, learning rate 0.05, feature subsampling rsm=0.25, Bernoulli subsample 0.85; selected on inner prompt-group OOF balanced accuracy with log-loss tie-break.",
        f"- Blend: `alpha * CatBoost + (1-alpha) * EQUAL9`, alpha selected from `{BLEND_GRID}` using inner-fold held-out predictions only.",
        "- The feature matrix includes per-checkpoint 18-class probabilities, top-1 one-hot, max probability, top-1/top-2 margin, entropy, class agreement counts, and EQUAL9 probabilities. Logistic scaling is fit within training folds.","",
        "## Artifacts","",
        "- `run_manifest.json`, `feature_manifest.json`: input hashes, exact rosters, feature definitions and fold/configuration provenance.",
        "- `folds/<roster>/outer_<fold>.json/.npz`: inner selection, fold predictions, training predictions, and resumable state.",
        "- `folds/<roster>/models_outer_<fold>/`: fitted scaler/logistic and CatBoost models for each outer fold.",
        "- `row_predictions.parquet`: ID-aligned full-OOF probabilities and labels for all six experiments plus EQUAL9 and C3.",
        "- `metrics.json`, `per_label_metrics.csv`: overall, per-fold, training, held-out and per-label scores.","",
        "All artifacts are derived from cached V0 predictions. No base-model inference, base-model training or Kaggle submission was performed.",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default=str(OUT));p.add_argument("--report",default=str(REPORT));p.add_argument("--no-resume",action="store_true")
    a=p.parse_args()
    with threadpool_limits(limits=1):
        result=run(output_dir=a.output_dir,report_path=a.report,resume=not a.no_resume)
    print(json.dumps({"report":a.report,"output_dir":a.output_dir,"results":{
        roster:{s:{k:result["results"][roster][s]["heldout_oof"][k] for k in ("accuracy","balanced_accuracy","log_loss","delta_vs_equal9","delta_vs_c3")} for s in STRATEGIES}
        for roster in ("equal9","all78")}},indent=2,default=str))


if __name__=="__main__":main()
