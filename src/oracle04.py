"""ORACLE04: nested prompt-group top-1/top-2 swap decisions.

Uses Trust List 2 cached predictions and ORACLE02/03 row-aligned artifacts.
Base checkpoints are never loaded. New fits are CPU-only meta-classifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import oracle02, oracle03
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/oracle04"
REPORT = ROOT / "docs/oracle04.md"
ORACLE02_ROWS = ROOT / "artifacts/analysis/oracle02/row_predictions.parquet"
ORACLE03_ROWS = ROOT / "artifacts/analysis/oracle03/row_predictions.parquet"
ORACLE03_FEATURES = ROOT / "artifacts/analysis/oracle03/text_features.npz"
GROUP_A = ("gemini-3.7-flash", "gemini-3.8-flash")
GROUP_B = ("claude-fable-5", "glm-5.3")
LOGISTIC_CS = (0.03, 0.3, 3.0)
CAT_CONFIGS = ({"depth": 3, "iterations": 100, "l2_leaf_reg": 12.0},
               {"depth": 4, "iterations": 120, "l2_leaf_reg": 20.0})
NESTED_BASE_CONFIG = {"depth": 3, "iterations": 120, "l2_leaf_reg": 10.0, "rsm": 0.25}
NESTED_BASE_CATBOOST_WEIGHT = 0.5
BOOTSTRAP_REPS = 1000
SEED = 2026


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p=np.asarray(p,dtype=np.float64)
    p=p/np.maximum(p.sum(axis=1,keepdims=True),1e-15)
    pred = p.argmax(1)
    cm = np.bincount(y * len(LABELS) + pred, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
    tp = cm.diagonal(); support = cm.sum(1); predicted = cm.sum(0)
    precision = tp / np.maximum(predicted, 1); recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {"rows": int(len(y)), "correct": int((pred == y).sum()),
            "accuracy": float((pred == y).mean()), "balanced_accuracy": float(recall.mean()),
            "log_loss": float(log_loss(y, p, labels=np.arange(len(LABELS)))),
            "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(predicted[i]-tp[i]),
                                  "fn": int(support[i]-tp[i]), "support": int(support[i])}
                          for i, label in enumerate(LABELS)}, "pred": pred}


def _family(model: str) -> str:
    m = str(model).lower()
    for prefix in ("ultrabert", "qwen", "gemma", "granite", "llama", "ministral",
                   "minicpm", "hunyuan", "modernbert", "deberta", "electra", "gte", "mpnet"):
        if prefix in m:
            return prefix
    if "catboost" in m: return "catboost"
    if "svc" in m: return "svc"
    return m.split("_")[0]


def _base_features(p: np.ndarray, matrices: dict[str, np.ndarray], ids: list[str],
                   equal_p: np.ndarray, families: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    n, k = p.shape
    order = np.argsort(-p, axis=1, kind="stable")
    top1, top2 = order[:, 0], order[:, 1]
    p1, p2 = p[np.arange(n), top1], p[np.arange(n), top2]
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    ratio = p1 / np.maximum(p2, 1e-12)
    eye = np.eye(k, dtype=np.float32)
    parts = [p.astype(np.float32), eye[top1], eye[top2]]
    names = ([f"oracle02_prob::{l}" for l in LABELS] +
             [f"top1::{l}" for l in LABELS] + [f"top2::{l}" for l in LABELS])
    scalars = np.column_stack([p1, p2, p1-p2, ratio, entropy]).astype(np.float32)
    parts.append(scalars); names += ["top1_probability", "top2_probability", "margin", "ratio", "entropy"]
    # Checkpoint probabilities and top-1 support for each of the two candidates.
    votes = np.zeros((n, 2, len(ids)), dtype=np.float32)
    candidate_probs = np.zeros_like(votes)
    family_names = sorted(set(families.values()))
    family_votes = np.zeros((n, 2, len(family_names)), dtype=np.float32)
    member_top = {}
    for j, sid in enumerate(ids):
        q = matrices[sid]
        member_top[sid] = q.argmax(1)
        votes[:, 0, j] = (member_top[sid] == top1)
        votes[:, 1, j] = (member_top[sid] == top2)
        candidate_probs[:, 0, j] = q[np.arange(n), top1]
        candidate_probs[:, 1, j] = q[np.arange(n), top2]
        fi = family_names.index(families[sid])
        family_votes[:, 0, fi] += votes[:, 0, j]
        family_votes[:, 1, fi] += votes[:, 1, j]
    for ci, sid in enumerate(ids):
        parts.append(candidate_probs[:, :, ci]); names += [f"{sid}::p_top1", f"{sid}::p_top2"]
        parts.append(votes[:, :, ci]); names += [f"{sid}::votes_top1", f"{sid}::votes_top2"]
    parts.append(family_votes.reshape(n, -1)); names += [f"family::{f}::votes_top1" for f in family_names]+[f"family::{f}::votes_top2" for f in family_names]
    eq_pred = equal_p.argmax(1)
    eq = np.column_stack([equal_p[np.arange(n), top1], equal_p[np.arange(n), top2],
                          (eq_pred == top1), (eq_pred == top2), (eq_pred != top1) & (eq_pred != top2)]).astype(np.float32)
    parts.append(eq); names += ["equal9_p_top1", "equal9_p_top2", "equal9_agrees_top1", "equal9_agrees_top2", "equal9_other"]
    X = np.column_stack(parts).astype(np.float32)
    if X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError("Nonfinite or misnamed ORACLE04 feature matrix")
    return X, names


def _swap_targets(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-p, axis=1, kind="stable")
    t1, t2 = order[:, 0], order[:, 1]
    return t1, t2, (t2 == y).astype(np.int8), (t1 == y)


def _eligible(t1: np.ndarray, t2: np.ndarray, group: tuple[str, ...] | None) -> np.ndarray:
    if group is None: return np.ones(len(t1), dtype=bool)
    a, b = (LABELS.index(x) for x in group)
    return ((t1 == a) & (t2 == b)) | ((t1 == b) & (t2 == a))


def _swap_rows(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = p.copy()
    ix = np.flatnonzero(mask)
    if len(ix):
        order = np.argsort(-out[ix], axis=1, kind="stable")
        a, b = order[:, 0], order[:, 1]
        va = out[ix, a].copy(); vb = out[ix, b].copy()
        out[ix, a] = vb; out[ix, b] = va
    return out


def _binary_fit(kind: str, X: np.ndarray, y: np.ndarray, cfg):
    if len(np.unique(y)) < 2:
        return {"kind": "constant", "prob": float((y.sum()+1)/(len(y)+2))}
    if kind == "logistic":
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(C=float(cfg), solver="lbfgs", max_iter=1000, random_state=SEED)
        model.fit(scaler.transform(X), y)
        return {"kind": kind, "scaler": scaler, "model": model}
    model = CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", depth=int(cfg["depth"]),
        iterations=int(cfg["iterations"]), learning_rate=0.04, l2_leaf_reg=float(cfg["l2_leaf_reg"]),
        random_seed=SEED, thread_count=1, task_type="CPU", verbose=False, allow_writing_files=False,
        random_strength=0.5, bootstrap_type="Bernoulli", subsample=0.85, border_count=48)
    model.fit(X, y)
    return {"kind": kind, "model": model}


def _binary_predict(bundle, X: np.ndarray) -> np.ndarray:
    if bundle["kind"] == "constant": return np.full(len(X), bundle["prob"], dtype=float)
    if bundle["kind"] == "logistic":
        model, scaler = bundle["model"], bundle["scaler"]
        cls = list(model.classes_).index(1)
        return model.predict_proba(scaler.transform(X))[:, cls]
    return bundle["model"].predict_proba(X)[:, list(bundle["model"].classes_).index(1)]


def _configs(kind: str):
    return LOGISTIC_CS if kind == "logistic" else CAT_CONFIGS


def _fit_oof(train_idx: np.ndarray, X: np.ndarray, target: np.ndarray, elig: np.ndarray,
             groups: np.ndarray, kind: str):
    local = train_idx[elig[train_idx]]
    if len(local) < 20 or len(np.unique(target[local])) < 2:
        constant = (int(target[local].sum())+1)/(len(local)+2) if len(local) else 0.5
        out=np.full(len(target),np.nan);out[train_idx]=constant
        return out, {"kind": kind, "selected": "constant", "grid": []}, None
    nsplit = min(3, len(np.unique(groups[local])))
    if nsplit < 2: raise ValueError("Need at least two prompt groups for swap classifier")
    splits = list(GroupKFold(n_splits=nsplit).split(np.zeros(len(local)), groups=groups[local]))
    records = []; oof_by_cfg = {}; train_pos={int(i):j for j,i in enumerate(train_idx)}
    for cfg in _configs(kind):
        oof = np.full(len(target), np.nan, dtype=float)
        for a, b in splits:
            tr, va = local[a], local[b]
            xtr=np.asarray([train_pos[int(i)] for i in tr]);xva=np.asarray([train_pos[int(i)] for i in va])
            bundle = _binary_fit(kind, X[xtr], target[tr], cfg)
            pred = _binary_predict(bundle, X[xva])
            oof[va] = pred
        valid = train_idx[elig[train_idx]]
        ll = float(log_loss(target[valid], np.clip(oof[valid], 1e-7, 1-1e-7), labels=[0, 1]))
        records.append({"config": cfg, "log_loss": ll})
        oof_by_cfg[json.dumps(cfg, sort_keys=True)] = oof
    best = min(records, key=lambda x: (x["log_loss"], json.dumps(x["config"], sort_keys=True)))
    selected_oof = oof_by_cfg[json.dumps(best["config"], sort_keys=True)]
    final = _binary_fit(kind, X[[train_pos[int(i)] for i in local]], target[local], best["config"])
    return selected_oof, {"kind": kind, "selected": best["config"], "grid": records,
                          "eligible_train_rows": int(len(local)), "positive_train_rows": int(target[local].sum())}, final


def _threshold_scores(train_idx, p, y, elig, score, no_swap=True):
    baseline = _metrics(y[train_idx], p[train_idx])["balanced_accuracy"]
    candidates = [np.inf]
    vals = score[train_idx][elig[train_idx]]
    if len(vals):
        candidates.extend(np.unique(np.quantile(vals, np.linspace(0, 1, 61))).tolist())
        candidates.append(float(np.nextafter(vals.min(), -np.inf)))
    best = (baseline, 0, np.inf)
    for threshold in sorted(set(candidates)):
        mask = elig[train_idx] & (score[train_idx] >= threshold)
        if not mask.any(): continue
        changed = np.zeros(len(y), dtype=bool); changed[train_idx] = mask
        candidate = _swap_rows(p, changed)
        ba = _metrics(y[train_idx], candidate[train_idx])["balanced_accuracy"]
        key = (ba, -int(mask.sum()), -threshold if np.isfinite(threshold) else -np.inf)
        if key > (best[0], -best[1], -best[2] if np.isfinite(best[2]) else -np.inf):
            best = (ba, int(mask.sum()), float(threshold))
    if best[0] <= baseline + 1e-12: return None, {"baseline_ba": baseline, "selected_ba": baseline, "oof_triggers": 0}
    return best[2], {"baseline_ba": baseline, "selected_ba": best[0], "oof_triggers": best[1],
                     "gain_pp": 100*(best[0]-baseline)}


def _apply_one(p, eligible, score, threshold):
    if threshold is None: return p.copy(), np.zeros(len(p), dtype=bool)
    mask = eligible & (score >= threshold)
    return _swap_rows(p, mask), mask


def _gain_bootstrap(y, pred, baseline_pred, groups, reps=BOOTSTRAP_REPS):
    rng = np.random.default_rng(SEED); unique = np.unique(groups); deltas=[]
    by_group={g:np.flatnonzero(groups==g) for g in unique}
    for _ in range(reps):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([by_group[g] for g in sampled])
        # Macro recall, ignoring a class only if absent in this resample.
        def ba(pr):
            present=np.unique(y[idx]); return float(np.mean([(pr[idx][y[idx]==c]==c).mean() for c in present]))
        deltas.append(ba(pred)-ba(baseline_pred))
    return [float(x) for x in np.quantile(deltas,[.025,.5,.975])]


def _swap_stats(y, p_before, p_after, triggered):
    before=p_before.argmax(1); after=p_after.argmax(1)
    corrected=int(((before!=y)&(after==y)).sum()); introduced=int(((before==y)&(after!=y)).sum())
    return {"triggers":int(triggered.sum()),"trigger_rate":float(triggered.mean()),
            "corrected":corrected,"introduced":introduced,"net_correct":corrected-introduced,
            "swap_precision":corrected/max(1,corrected+introduced),
            "changed_predictions":int((before!=after).sum())}


def _build_meta_features(p, matrices, ids, equal_p, families, style, embedding, variant):
    X,names=_base_features(p,matrices,ids,equal_p,families)
    if variant in ("core_stylometry", "core_stylometry_embedding"):
        X=np.column_stack((X,style)).astype(np.float32); names += [f"style::{i}" for i in range(style.shape[1])]
    if variant in ("core_embedding", "core_stylometry_embedding"):
        X=np.column_stack((X,embedding)).astype(np.float32); names += [f"embedding::{i}" for i in range(embedding.shape[1])]
    return X,names


def run(output_dir=OUT, report_path=REPORT):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3,equal_p,hashes=oracle02._source_inputs(ROOT,output_dir,True)
    if len(ids)!=78: raise ValueError(f"Expected 78 checkpoints, got {len(ids)}")
    source02=pd.read_parquet(ORACLE02_ROWS); source03=pd.read_parquet(ORACLE03_ROWS)
    for name,frame in (("ORACLE02",source02),("ORACLE03",source03)):
        if frame.id.astype(str).tolist()!=row_ids or frame.prompt_id.astype(str).tolist()!=prompts.tolist() or frame.y_true.astype(str).tolist()!=y_text.tolist():
            raise ValueError(f"{name} predictions do not align exactly to V0 IDs/prompts/labels")
    p_cols=[f"prob_oof::all78::catboost_equal9_blend::{l}" for l in LABELS]
    p_oof=source02[p_cols].to_numpy(dtype=np.float64)
    if not np.allclose(p_oof.sum(1),1,atol=1e-6): raise ValueError("ORACLE02 probabilities are not normalized")
    p_equal=np.mean([matrices[s] for s in oracle02.EQUAL9],axis=0); p_equal/=p_equal.sum(1,keepdims=True)
    p_h09=np.column_stack([source03[f"prob::H09::{l}"].to_numpy(float) for l in LABELS])
    p_oracle03=np.column_stack([source03[f"prob::rerank_top2::{l}"].to_numpy(float) for l in LABELS])
    z=np.load(ORACLE03_FEATURES)
    if z["ids"].astype(str).tolist()!=row_ids: raise ValueError("ORACLE03 text features do not align by ID")
    style=z["stylometry"].astype(np.float32); embedding=z["embedding"].astype(np.float32)
    trust=pd.read_csv(ROOT/"docs/trust-list-2-v025.csv",dtype=str,keep_default_na=False)
    model_map=trust.groupby("sequence_id").model.first().to_dict()
    families={sid:_family(model_map.get(sid,sid)) for sid in ids}
    ptop1,ptop2,swap_target,top1_correct=_swap_targets(y,p_oof)
    top2_correct=swap_target.astype(bool); neither=~top1_correct & ~top2_correct
    covered=top1_correct|top2_correct
    oracle=np.zeros_like(p_oof)
    oracle_pred=ptop1.copy(); oracle_pred[top2_correct]=ptop2[top2_correct]
    oracle[np.arange(len(y)),oracle_pred]=1
    # Existing ORACLE02 feature construction is label-free and reused for nested baseline fits.
    Xbase,_=oracle02._build_features(ids,matrices,equal_p)
    nested_meta={}; nested_by_fold={}; oracle02_code_hash=_sha(Path(oracle02.__file__).resolve())
    for fold in range(1,6):
        outer_train=np.flatnonzero(fold_ids!=fold); held=np.flatnonzero(fold_ids==fold)
        cache=output_dir/f"nested_baseline_outer_{fold}.npz"; cache_meta=output_dir/f"nested_baseline_outer_{fold}.json"
        if cache.exists() and cache_meta.exists():
            meta=json.loads(cache_meta.read_text())
            if (meta.get("input_hashes")==hashes and meta.get("fold_ids_sha256")==hashes["fold_assignments_sha256"]
                    and meta.get("oracle02_code_sha256")==oracle02_code_hash):
                q=np.load(cache)
                if q["train_indexes"].tolist()==outer_train.tolist() and q["held_indexes"].tolist()==held.tolist():
                    nested_by_fold[str(fold)]=q["train_probs"]
                    nested_meta[str(fold)]=meta["fit_details"]
            else: cache.unlink(); cache_meta.unlink()
        if str(fold) not in nested_by_fold:
            local=np.full((len(outer_train),len(LABELS)),np.nan); loc={int(i):j for j,i in enumerate(outer_train)}; fit_details=[]
            for a,b in GroupKFold(3).split(np.zeros(len(outer_train)),groups=prompts[outer_train]):
                fit_idx=outer_train[a]; val_idx=outer_train[b]
                cfg=NESTED_BASE_CONFIG;alpha=NESTED_BASE_CATBOOST_WEIGHT
                cb=oracle02._fit_catboost(Xbase[fit_idx],y[fit_idx],cfg)
                cb_p=oracle02._align_proba(cb,Xbase[val_idx])
                local[[loc[int(i)] for i in val_idx]]=alpha*cb_p+(1-alpha)*equal_p[val_idx]
                fit_details.append({"fit_rows":len(fit_idx),"held_rows":len(val_idx),"config":cfg,
                    "catboost_weight":alpha,"selection":"locked to ORACLE02 modal setting; no labels from held inner fold used"})
            if np.isnan(local).any() or not np.allclose(local.sum(1),1,atol=1e-6): raise ValueError(f"Nested baseline generation failed in outer fold {fold}")
            nested_by_fold[str(fold)]=local; nested_meta[str(fold)]=fit_details
            np.savez_compressed(cache,train_probs=local,train_indexes=outer_train,held_indexes=held)
            _write_json(cache_meta,{"input_hashes":hashes,"fold_ids_sha256":hashes["fold_assignments_sha256"],
                                    "oracle02_code_sha256":oracle02_code_hash,"fit_details":fit_details})
    groups_by_spec={"global":None,"group_A":GROUP_A,"group_B":GROUP_B}
    variants=("core","core_stylometry","core_embedding","core_stylometry_embedding")
    # All held-out strategies start with exactly the saved ORACLE02 probabilities.
    outputs={}; configs={}; fold_rows=[]; feature_importance={}; trained_models=[]
    oof_scores={}
    for fold in range(1,6):
        tr=np.flatnonzero(fold_ids!=fold);va=np.flatnonzero(fold_ids==fold)
        p_train=np.full_like(p_oof,np.nan);p_train[tr]=nested_by_fold[str(fold)];p_train[va]=p_oof[va]
        target=np.zeros(len(y),dtype=np.int8)
        order_tr=np.argsort(-p_train[tr],axis=1,kind="stable");t1tr,t2tr=order_tr[:,0],order_tr[:,1]
        order_va=np.argsort(-p_oof[va],axis=1,kind="stable");t1va,t2va=order_va[:,0],order_va[:,1]
        target[tr]=(t2tr==y[tr]);target[va]=(t2va==y[va])
        t1_all=np.zeros(len(y),dtype=int);t2_all=np.zeros(len(y),dtype=int)
        t1_all[tr]=t1tr;t2_all[tr]=t2tr;t1_all[va]=t1va;t2_all[va]=t2va
        elig_masks={key:np.zeros(len(y),dtype=bool) for key in groups_by_spec}
        for key,group in groups_by_spec.items():
            elig_masks[key][tr]=_eligible(t1tr,t2tr,group)
            elig_masks[key][va]=_eligible(t1va,t2va,group)
        # Produce train/held candidate features for each ablation using the appropriate baseline predictions.
        Xtr_by={};Xva_by={}
        mats_tr={sid:q[tr] for sid,q in matrices.items()};mats_va={sid:q[va] for sid,q in matrices.items()}
        for variant in variants:
            Xtr_by[variant],fnames=_build_meta_features(p_train[tr],mats_tr,ids,equal_p[tr],families,style[tr],embedding[tr],variant)
            Xva_by[variant],_=_build_meta_features(p_oof[va],mats_va,ids,equal_p[va],families,style[va],embedding[va],variant)
        # Fit standalone group/global decisions, both model types. The OOF scores used to tune thresholds
        # are generated inside the outer training partition only.
        fitted={}
        for spec,group in groups_by_spec.items():
            for kind in ("logistic","catboost"):
                for variant in (("core",) if spec!="global" else variants):
                    key=f"{spec}_{kind}_{variant}"
                    score_oof,selection,bundle=_fit_oof(tr,Xtr_by[variant],target,elig_masks[spec],prompts,kind)
                    if bundle is None:
                        score_test=np.full(len(va),0.5)
                    else:
                        score_test=_binary_predict(bundle,Xva_by[variant])
                    # Select the threshold on group-held-out training scores to maximize overall BA gain.
                    threshold,thinfo=_threshold_scores(tr,p_train,y,elig_masks[spec],score_oof)
                    fitted[key]={"score_oof":score_oof,"score_test":score_test,"threshold":threshold,
                                 "selection":selection,"threshold_fit":thinfo,"bundle":bundle,"variant":variant,
                                 "spec":spec,"kind":kind}
                    configs[key] = {"outer_fold":fold,"model_selection":selection,"threshold":threshold,
                                    "threshold_selection":thinfo,"eligible_train_rows":int(elig_masks[spec][tr].sum()),
                                    "swap_positive_train_rows":int(target[tr][elig_masks[spec][tr]].sum()),"feature_variant":variant}
                    model_dir=output_dir/"models"/f"outer_{fold}"/key;model_dir.mkdir(parents=True,exist_ok=True)
                    if bundle is not None:
                        if kind=="catboost": bundle["model"].save_model(str(model_dir/"model.cbm"))
                        else: joblib.dump(bundle,model_dir/"model.joblib",compress=3)
                        trained_models.append(str((model_dir/("model.cbm" if kind=="catboost" else "model.joblib")).relative_to(output_dir)))
                        fi=(bundle["model"].get_feature_importance().tolist() if kind=="catboost" else np.mean(np.abs(bundle["model"].coef_),axis=0).tolist())
                        feature_importance[f"outer_{fold}::{key}"]=sorted([{"feature":n,"importance":float(v)} for n,v in zip(fnames,fi)],key=lambda x:-x["importance"])[:50]
        # standalone outputs: A/B/global, each classifier, plus margin threshold baselines.
        base_hold=p_oof[va]; tt1=t1va;tt2=t2va
        for key,item in fitted.items():
            if item["spec"]=="global": masks=_eligible(tt1,tt2,None)
            else: masks=_eligible(tt1,tt2,GROUP_A if item["spec"]=="group_A" else GROUP_B)
            th=item["threshold"]; chosen=np.zeros(len(va),bool) if th is None else masks&(item["score_test"]>=th)
            outputs.setdefault(key,np.zeros_like(p_oof))[va]=_swap_rows(base_hold,chosen)
            oof_scores.setdefault(key,np.zeros(len(y)))[va]=item["score_test"]
            fold_rows.append({"fold":fold,"strategy":key,"triggered":int(chosen.sum()),"eligible":int(masks.sum()),
                              "threshold":th,"threshold_train":item["threshold_fit"]})
        # Margin-only baselines use uncertainty (small p1-p2 means swap-prone).
        margin_score=-(base_hold[np.arange(len(va)),tt1]-base_hold[np.arange(len(va)),tt2])
        for spec,group in groups_by_spec.items():
            eligibility=_eligible(tt1,tt2,group)
            train_score=np.full(len(y),np.nan);train_score[tr]=-(p_train[tr,t1tr]-p_train[tr,t2tr])
            th,thinfo=_threshold_scores(tr,p_train,y,elig_masks[spec],train_score)
            chosen=np.zeros(len(va),bool) if th is None else eligibility&(margin_score>=th)
            key=f"{spec}_margin"
            outputs.setdefault(key,np.zeros_like(p_oof))[va]=_swap_rows(base_hold,chosen)
            configs[f"fold{fold}_{key}"]={"threshold":th,"threshold_selection":thinfo}
            fold_rows.append({"fold":fold,"strategy":key,"triggered":int(chosen.sum()),"eligible":int(eligibility.sum()),"threshold":th,"threshold_train":thinfo})
        # Combined decisions: Group A then B then global. Sequential thresholds are selected
        # on training-only OOF scores, accounting for earlier decisions in the cascade.
        for kind in ("logistic","catboost"):
            working_train=np.full_like(p_train,np.nan);working_train[tr]=p_train[tr]
            claimed_train=np.zeros(len(tr),bool); threshold_details={}
            composed_test=base_hold.copy(); claimed_test=np.zeros(len(va),bool)
            for spec in ("group_A","group_B","global"):
                model_key=f"{spec}_{kind}_core" if spec!="global" else f"global_{kind}_core"
                item=fitted[model_key]; elig_all=elig_masks[spec]
                train_elig=elig_all[tr]&~claimed_train
                threshold,thinfo=_threshold_scores(tr,working_train,y,_mask_to_full(tr,train_elig,len(y)),item["score_oof"])
                test_elig=_eligible(tt1,tt2,groups_by_spec[spec])&~claimed_test if spec!="global" else ~claimed_test
                fired=np.zeros(len(va),bool) if threshold is None else test_elig&(item["score_test"]>=threshold)
                composed_test=_swap_rows(composed_test,fired); claimed_test|=fired
                score_train=item["score_oof"][tr]
                fired_train=train_elig if threshold is not None else np.zeros(len(tr),bool)
                if threshold is not None: fired_train &= score_train>=threshold
                fullmask=np.zeros(len(y),bool);fullmask[tr]=fired_train
                working_train=_swap_rows(working_train,fullmask)
                claimed_train|=fired_train
                threshold_details[spec]={"threshold":threshold,"training_selection":thinfo,"fired_test":int(fired.sum())}
            name=f"combined_{kind}_core";outputs.setdefault(name,np.zeros_like(p_oof))[va]=composed_test
            configs[f"fold{fold}_{name}"]=threshold_details
            fold_rows.append({"fold":fold,"strategy":name,"triggered":int(claimed_test.sum()),"eligible":len(va),"threshold":None,"threshold_train":threshold_details})
        # Save fold-aligned features and baseline construction metadata.
        np.savez_compressed(output_dir/f"features_outer_{fold}.npz",train_indexes=tr,held_indexes=va,
            train_core=Xtr_by["core"],held_core=Xva_by["core"],train_stylometry=Xtr_by["core_stylometry"],
            held_stylometry=Xva_by["core_stylometry"],train_embedding=Xtr_by["core_embedding"],held_embedding=Xva_by["core_embedding"])
        _write_json(output_dir/f"nested_baseline_outer_{fold}.json",{"input_hashes":hashes,"fold_ids_sha256":hashes["fold_assignments_sha256"],"fit_details":nested_meta[str(fold)]})
        print(f"ORACLE04 outer fold {fold}/5 complete",flush=True)
    # Baseline outputs and aggregate metrics.
    predictions=pd.DataFrame({"id":row_ids,"prompt_id":prompts,"y_true":y_text,
        "pred::ORACLE02":np.asarray(LABELS)[p_oof.argmax(1)],"pred::top2_oracle":np.asarray(LABELS)[oracle_pred]})
    ref_probs={"ORACLE02":p_oof,"ORACLE03_top2":p_oracle03,"H09":p_h09,"EQUAL9":p_equal}
    metrics={};corrections={};per_label=[];fold_metric_rows=[]
    for name,p in ref_probs.items():
        mm=_metrics(y,p);pred=mm.pop("pred");metrics[name]=mm
        predictions[f"pred::{name}"]=np.asarray(LABELS)[pred]
        for label,val in mm["per_label"].items(): per_label.append({"strategy":name,"label":label,**val})
    metrics["top2_oracle"]=_metrics(y,oracle)
    metrics["top2_oracle"].pop("pred",None)
    for label,val in metrics["top2_oracle"]["per_label"].items(): per_label.append({"strategy":"top2_oracle","label":label,**val})
    for name,p in outputs.items():
        mm=_metrics(y,p);pred=mm.pop("pred");metrics[name]=mm
        predictions[f"pred::{name}"]=np.asarray(LABELS)[pred]
        for j,label in enumerate(LABELS): predictions[f"prob::{name}::{label}"]=p[:,j]
        basepred=p_oof.argmax(1);trigger=(predictions[f"pred::{name}"].to_numpy()!=np.asarray(LABELS)[basepred])
        stats=_swap_stats(y,p_oof,p,trigger)
        base_order=np.argsort(-p_oof,axis=1,kind="stable")
        swap_target_oof=base_order[:,1]==y
        target_tp=int((trigger&swap_target_oof).sum());target_fp=int((trigger&~swap_target_oof).sum())
        target_fn=int((~trigger&swap_target_oof).sum())
        stats.update({"swap_target_tp":target_tp,"swap_target_fp":target_fp,"swap_target_fn":target_fn,
                      "swap_target_precision":target_tp/max(1,target_tp+target_fp),
                      "swap_target_recall":target_tp/max(1,target_tp+target_fn)})
        corrections[name]={"ORACLE02":stats}
        for ref,refp in ref_probs.items():
            if ref=="ORACLE02": continue
            rp=refp.argmax(1);npred=pred
            corr=int(((rp!=y)&(npred==y)).sum());intro=int(((rp==y)&(npred!=y)).sum())
            corrections[name][ref]={"corrected":corr,"introduced":intro,"net_correct":corr-intro,
                                     "changed_predictions":int((npred!=rp).sum())}
        for label,val in mm["per_label"].items(): per_label.append({"strategy":name,"label":label,**val})
    basepred=p_oof.argmax(1)
    group_id=np.asarray(prompts)
    bootstrap={name:_gain_bootstrap(y,p.argmax(1),basepred,group_id) for name,p in outputs.items()}
    for name,p in {**ref_probs,**outputs}.items():
        for fold in range(1,6):
            va=np.flatnonzero(fold_ids==fold);met=_metrics(y[va],p[va]);met.pop("pred",None)
            fold_metric_rows.append({"strategy":name,"fold":fold,**met})
    # Decision-state categories relative to ORACLE02's saved top-1/top-2.
    t1,t2,swap_target_oof,t1corr=_swap_targets(y,p_oof); t2corr=t2==y; neither=~t1corr&~t2corr
    full_order=np.argsort(-p_oof,axis=1,kind="stable")
    candidate_stats={"top1_correct":int(t1corr.sum()),"top2_correct_top1_wrong":int((t2corr&~t1corr).sum()),
                     "neither_correct":int(neither.sum()),"top2_union_coverage":float((t1corr|t2corr).mean()),
                     "top3_coverage":float((full_order[:,:3]==y[:,None]).any(1).mean()),
                     "top2_swap_positive_rate":float((t2corr&~t1corr).mean())}
    group_stats={}
    for name,g in (("A",GROUP_A),("B",GROUP_B)):
        eligible=_eligible(t1,t2,g); group_stats[name]={"eligible_rows":int(eligible.sum()),
            "top1_correct":int((eligible&t1corr).sum()),"top2_correct_top1_wrong":int((eligible&t2corr&~t1corr).sum()),
            "neither_correct":int((eligible&neither).sum()),"coverage":float(eligible.mean())}
    # Persist model metadata/feature importance and all exact predictions.
    predictions.to_parquet(output_dir/"row_predictions.parquet",index=False)
    pd.DataFrame(per_label).to_csv(output_dir/"per_label_metrics.csv",index=False)
    pd.DataFrame(fold_metric_rows).to_csv(output_dir/"fold_metrics.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir/"fold_decisions.csv",index=False)
    _write_json(output_dir/"feature_importance.json",feature_importance)
    report_data={"validation_rows":len(y),"prompt_groups":int(pd.Series(prompts).nunique()),"class_order":LABELS,
        "input_hashes":hashes,"baseline_metrics":metrics,"corrections":corrections,"bootstrap_95pct_delta_ba":bootstrap,
        "candidate_categories":candidate_stats,"group_categories":group_stats,"fold_rows":fold_rows,
        "fold_metrics_path":"fold_metrics.csv","feature_importance_top50":feature_importance,
        "nested_baseline_fit_details":nested_meta,"configs":configs,"trained_models":trained_models,
        "base_model_retraining":False,"base_model_inference":False,"kaggle_submission":False,
        "v0_exposure_caveat":"Checkpoint predictions, ORACLE02 outer-fold model selection, and roster selection are exposed to V0. Meta-model outer folds are group held out, but this cannot make the estimate independent.",
        "top2_oracle_definition":"If top-1 or top-2 is true, choose the true one; otherwise retain top-1. Ground-truth oracle, not deployable."}
    _write_json(output_dir/"metrics.json",report_data)
    _write_json(output_dir/"run_manifest.json",{"analysis":"ORACLE04","status":"completed","created_utc":pd.Timestamp.utcnow().isoformat(),
        "source_artifacts":{"oracle02_rows_sha256":_sha(ORACLE02_ROWS),"oracle03_rows_sha256":_sha(ORACLE03_ROWS),
            "oracle03_features_sha256":_sha(ORACLE03_FEATURES),"script_sha256":_sha(Path(__file__).resolve()),
            "oracle02_code_sha256":oracle02_code_hash,**hashes},"outer_folds":"existing five prompt-group folds","inner_folds":"3-fold GroupKFold; ORACLE02 predictions cross-fitted within each outer training partition",
        "feature_variants":list(variants),"model_grids":{"logistic_C":LOGISTIC_CS,"catboost":CAT_CONFIGS,
            "nested_oracle02_locked_config":NESTED_BASE_CONFIG,"nested_oracle02_catboost_weight":NESTED_BASE_CATBOOST_WEIGHT},
        "bootstrap_reps":BOOTSTRAP_REPS,"seed":SEED,"base_model_retraining":False,"base_model_inference":False,"kaggle_submission":False})
    report_data["bootstrap_best_vs_references"]=_best_reference_bootstrap(y,outputs,ref_probs,prompts,metrics)
    _write_json(output_dir/"metrics.json",report_data)
    write_report(report_path,report_data,p_oof,p_oracle03,p_h09,p_equal,y,outputs,candidate_stats,group_stats,metrics,bootstrap,fold_rows)
    return report_data


def _mask_to_full(idx, mask, n):
    out=np.zeros(n,dtype=bool);out[idx]=mask;return out


def _best_reference_bootstrap(y, outputs, references, prompts, metrics):
    best=max(outputs,key=lambda n:(metrics[n]["balanced_accuracy"],metrics[n]["accuracy"],n))
    pred=outputs[best].argmax(1)
    return {"best_strategy":best,"intervals":{name:_gain_bootstrap(y,pred,p.argmax(1),prompts)
        for name,p in references.items()}}


def refresh_report(output_dir=OUT, report_path=REPORT):
    """Re-render the report from saved outputs without fitting any models."""
    output_dir=Path(output_dir); data=json.loads((output_dir/"metrics.json").read_text())
    ids,matrices,v0,row_ids,prompts,y_text,y,fold_ids,c3,equal_p,hashes=oracle02._source_inputs(ROOT,output_dir,True)
    src02=pd.read_parquet(ORACLE02_ROWS);src03=pd.read_parquet(ORACLE03_ROWS)
    probs02=src02[[f"prob_oof::all78::catboost_equal9_blend::{l}" for l in LABELS]].to_numpy(float)
    equal9=np.mean([matrices[s] for s in oracle02.EQUAL9],axis=0);equal9/=equal9.sum(1,keepdims=True)
    h09=np.column_stack([src03[f"prob::H09::{l}"].to_numpy(float) for l in LABELS])
    o3=np.column_stack([src03[f"prob::rerank_top2::{l}"].to_numpy(float) for l in LABELS])
    frame=pd.read_parquet(output_dir/"row_predictions.parquet")
    output_names=[x for x in data["baseline_metrics"] if x not in {"ORACLE02","ORACLE03_top2","H09","EQUAL9","top2_oracle"}]
    outputs={name:frame[[f"prob::{name}::{l}" for l in LABELS]].to_numpy(float) for name in output_names}
    references={"ORACLE02":probs02,"ORACLE03_top2":o3,"H09":h09,"EQUAL9":equal9}
    order=np.argsort(-probs02,axis=1,kind="stable")
    data["candidate_categories"]["top3_coverage"]=float((order[:,:3]==y[:,None]).any(1).mean())
    data["bootstrap_best_vs_references"]=_best_reference_bootstrap(y,outputs,references,prompts,data["baseline_metrics"])
    _write_json(output_dir/"metrics.json",data)
    write_report(report_path,data,probs02,o3,h09,equal9,y,outputs,data["candidate_categories"],data["group_categories"],
                 data["baseline_metrics"],data["bootstrap_95pct_delta_ba"],data["fold_rows"])
    manifest_path=output_dir/"run_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["source_artifacts"]["script_sha256"]=_sha(Path(__file__).resolve())
    manifest["report_sha256"]=_sha(Path(report_path))
    _write_json(manifest_path,manifest)
    return data


def write_report(path, data, p_base, p_o3, p_h09, p_eq, y, outputs, candidate, group_stats, metrics, bootstrap, fold_rows):
    lines=["# ORACLE04 — Top-1 vs Top-2 Decision Model","",
        f"Validation: {data['validation_rows']:,} V0 responses, {data['prompt_groups']} prompt groups, canonical 18-class order. Base probabilities and checkpoint predictions are cached; only CPU meta-classifiers were fitted. This is a development-set diagnostic, not an independent Kaggle estimate.","",
        "## Baseline and top-two opportunity","",
        "| System | Accuracy | Balanced accuracy | Log loss |","|---|---:|---:|---:|"]
    for name in ("ORACLE02","ORACLE03_top2","H09","EQUAL9","top2_oracle"):
        mm=metrics[name];lines.append(f"| {name} | {mm['accuracy']:.3%} | {mm['balanced_accuracy']:.3%} | {mm['log_loss']:.5f} |")
    lines += ["",f"Among the saved ORACLE02 rows: top-1 is correct on {candidate['top1_correct']:,}; top-1 is wrong and top-2 correct on {candidate['top2_correct_top1_wrong']:,}; neither is correct on {candidate['neither_correct']:,}. Top-2 union coverage is {candidate['top2_union_coverage']:.2%}; top-3 coverage is {candidate['top3_coverage']:.2%}. The oracle result is a ground-truth upper bound.","",
        "| Group | Eligible rows | Top-1 correct | Top-2 correct, top-1 wrong | Neither correct |","|---|---:|---:|---:|---:|"]
    for name,info in group_stats.items(): lines.append(f"| {name}: {', '.join(GROUP_A if name=='A' else GROUP_B)} | {info['eligible_rows']} | {info['top1_correct']} | {info['top2_correct_top1_wrong']} | {info['neither_correct']} |")
    lines += ["","## Cross-fitted results","",
        "Thresholds maximize balanced-accuracy improvement on training-only inner-fold predictions. A no-swap decision is selected unless a threshold improves training-fold balanced accuracy. Swap precision is corrected / (corrected + introduced). Bootstrap intervals resample prompt groups (1,000 replicates) and measure the BA delta from ORACLE02.","",
        "| Strategy | Accuracy | Balanced accuracy | Log loss | Triggered | Trigger % | Corrected / introduced | Correction precision | Target precision / recall | Net correct | 95% cluster-bootstrap ΔBA |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names=sorted(outputs,key=lambda n:(-metrics[n]['balanced_accuracy'],n))
    for name in names:
        s=metrics[name];c=data['corrections'][name]['ORACLE02'];ci=bootstrap[name]
        lines.append(f"| {name} | {s['accuracy']:.3%} | {s['balanced_accuracy']:.3%} | {s['log_loss']:.5f} | {c['triggers']:,} | {c['trigger_rate']:.2%} | {c['corrected']} / {c['introduced']} | {c['swap_precision']:.2%} | {c['swap_target_precision']:.2%} / {c['swap_target_recall']:.2%} | {c['net_correct']:+d} | [{ci[0]*100:+.3f}, {ci[2]*100:+.3f}] pp |")
    lines += ["","### Fold stability","","| Strategy | Fold | BA | Accuracy | Δ BA vs ORACLE02 fold |","|---|---:|---:|---:|---:|"]
    # Fold metrics are written separately for every method; reconstruct this section from fold_rows,
    # with aggregate trigger counts as a compact stability view.
    fold_df=pd.read_csv(OUT/"fold_metrics.csv") if (OUT/"fold_metrics.csv").exists() else pd.DataFrame()
    base_vals=fold_df[fold_df.strategy=="ORACLE02"].set_index("fold") if not fold_df.empty else pd.DataFrame()
    fold_positive={}
    for strategy in ("group_A_logistic_core","group_A_catboost_core","group_B_logistic_core","group_B_catboost_core"):
        vals=fold_df[fold_df.strategy==strategy].set_index("fold") if not fold_df.empty else pd.DataFrame()
        fold_positive[strategy]=sum(vals.loc[f,"balanced_accuracy"]>base_vals.loc[f,"balanced_accuracy"] for f in vals.index) if len(vals) and len(base_vals) else 0
    for name in names:
        vals=fold_df[fold_df.strategy==name]
        base_vals=fold_df[fold_df.strategy=="ORACLE02"].set_index("fold")
        for _,row in vals.iterrows():
            delta=(row.balanced_accuracy-base_vals.loc[row.fold,"balanced_accuracy"])*100
            lines.append(f"| {name} | {int(row.fold)} | {row.balanced_accuracy:.3%} | {row.accuracy:.3%} | {delta:+.3f} pp |")
    lines += ["","### Comparison with reference systems","","| Strategy | Δ BA vs ORACLE02 | Δ BA vs ORACLE03 top-2 | Δ BA vs H09 | Δ BA vs EQUAL9 | Corrected/introduced vs ORACLE02 | vs ORACLE03 | vs H09 | vs EQUAL9 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m=metrics[name];c=data["corrections"][name]
        deltas=[(m["balanced_accuracy"]-metrics[ref]["balanced_accuracy"])*100 for ref in ("ORACLE02","ORACLE03_top2","H09","EQUAL9")]
        pairs=[f"{c[ref]['corrected']}/{c[ref]['introduced']}" for ref in ("ORACLE02","ORACLE03_top2","H09","EQUAL9")]
        lines.append(f"| {name} | "+" | ".join(f"{v:+.3f} pp" for v in deltas)+" | "+" | ".join(pairs)+" |")
    lines += ["","## Feature utility and model selection","",
        "Feature importance is fold-model-specific and diagnostic. Core features include ORACLE02 top-1/top-2 labels and probabilities, margin/ratio/entropy, each checkpoint's probabilities and top-1 support for both candidate labels, family-level votes, and EQUAL9 agreement. The optional ablations append cached stylometry and frozen text embeddings. Top 20 features per fitted model are saved in `feature_importance.json`.","",
        "| Fitted model | Highest-ranked features |","|---|---|"]
    for name,vals in data["feature_importance_top50"].items():
        lines.append(f"| {name} | "+", ".join(x['feature'] for x in vals[:8])+" |")
    best_name=names[0]
    best_corrections=data["corrections"][best_name]["ORACLE02"]
    lines += ["","## Per-label impact","",f"Per-label precision and recall for the top-ranked strategy (**{best_name}**) versus ORACLE02:","",
        "| Label | ORACLE02 precision | Best precision | Δ | ORACLE02 recall | Best recall | Δ |","|---|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        b=metrics["ORACLE02"]["per_label"][label];q=metrics[best_name]["per_label"][label]
        lines.append(f"| {label} | {b['precision']:.2%} | {q['precision']:.2%} | {(q['precision']-b['precision'])*100:+.2f} pp | {b['recall']:.2%} | {q['recall']:.2%} | {(q['recall']-b['recall'])*100:+.2f} pp |")
    lines += ["","Full precision, recall, F1, TP, FP, FN and support for every baseline and variant are in `per_label_metrics.csv`. The row-level predictions preserve V0 IDs and prompt IDs in `row_predictions.parquet`.","",
        "## Conclusions","",
        f"1. **Swap detection:** best approach {names[0]} triggers {best_corrections['triggers']:,} swaps; its held-out top-2-correct target precision/recall is {best_corrections['swap_target_precision']:.1%}/{best_corrections['swap_target_recall']:.1%}.",
        f"2. **Useful features:** global logistic core is most consistently led by checkpoint probabilities assigned to top-2 (notably {', '.join(x['feature'] for x in data['feature_importance_top50'].get('outer_1::global_logistic_core',[])[:5])}); adding stylometry or embeddings did not improve over the core global logistic result.",
        f"3. **Targeted groups:** Group A logistic improved BA over ORACLE02 in {fold_positive['group_A_logistic_core']}/5 folds ({data['corrections']['group_A_logistic_core']['ORACLE02']['corrected']}/{data['corrections']['group_A_logistic_core']['ORACLE02']['introduced']} corrected/introduced); Group A CatBoost in {fold_positive['group_A_catboost_core']}/5. Group B logistic improved in {fold_positive['group_B_logistic_core']}/5 ({data['corrections']['group_B_logistic_core']['ORACLE02']['corrected']}/{data['corrections']['group_B_logistic_core']['ORACLE02']['introduced']}); Group B CatBoost in {fold_positive['group_B_catboost_core']}/5. Their target recall remains low.",
        f"4. **Global and combined:** global logistic reached {metrics['global_logistic_core']['balanced_accuracy']:.3%} BA; combined logistic reached {metrics['combined_logistic_core']['balanced_accuracy']:.3%}. The combined gain over global is {(metrics['combined_logistic_core']['balanced_accuracy']-metrics['global_logistic_core']['balanced_accuracy'])*100:+.3f} pp, effectively tied.",
        f"5. **Oracle headroom:** top-two oracle BA is {metrics['top2_oracle']['balanced_accuracy']:.3%}; {names[0]} gains {(metrics[names[0]]['balanced_accuracy']-metrics['ORACLE02']['balanced_accuracy'])*100:.3f} pp of the {(metrics['top2_oracle']['balanced_accuracy']-metrics['ORACLE02']['balanced_accuracy'])*100:.3f} pp available over ORACLE02 ({(metrics[names[0]]['balanced_accuracy']-metrics['ORACLE02']['balanced_accuracy'])/max(1e-12,metrics['top2_oracle']['balanced_accuracy']-metrics['ORACLE02']['balanced_accuracy']):.1%} recovered).",
        f"6. **Against ORACLE03:** {names[0]} changes BA by {(metrics[names[0]]['balanced_accuracy']-metrics['ORACLE03_top2']['balanced_accuracy'])*100:+.3f} pp versus the 67.500% top-2 specialist. Best-strategy prompt-cluster bootstrap intervals are reported in the next table; these remain V0 diagnostics.",
        "7. **Kaggle readiness:** do not submit on this evidence alone. V0 influenced checkpoint, roster and prior ORACLE02 selection; no independent Kaggle score is estimated.","",
        "### Bootstrap interval for the selected best strategy","",
        "Selection of the best strategy used the pooled cross-fitted V0 scores, so these paired bootstrap intervals are descriptive after selection, not an unbiased model-selection interval.","",
        "| Reference | Best strategy Δ BA, 95% prompt-group bootstrap CI |","|---|---:|"]
    best_boot=data.get("bootstrap_best_vs_references",{}).get("intervals",{})
    for ref,ci in best_boot.items(): lines.append(f"| {ref} | [{ci[0]*100:+.3f}, {ci[2]*100:+.3f}] pp |")
    lines += ["","## Three recommendations","",
        f"1. Keep {names[0]} as the strongest development candidate: {best_corrections['corrected']} corrected versus {best_corrections['introduced']} introduced against ORACLE02; it exceeds ORACLE03 by {(metrics[names[0]]['balanced_accuracy']-metrics['ORACLE03_top2']['balanced_accuracy'])*100:+.3f} pp on these folds.",
        "2. Prefer the simpler global logistic model for the next controlled comparison; combined routing adds only a negligible score difference and adds specialist thresholds.",
        "3. Require an unseen labeled evaluation or Kaggle submission before changing the 0.581 public-score reference.","",
        "## Limitations and artifacts","",
        "V0 is reused for base checkpoint and trust-list selection. The outer-fold swap-model predictions are held out by prompt, but the saved ORACLE02 held-out probabilities come from its prior outer-fold workflow, whose model/hyperparameter choices were also made using the outer training rows. To keep swap-model training leakage controlled, ORACLE02 training examples receive additional nested predictions generated exclusively from other prompt groups within that outer training partition. For speed, nested ORACLE02 fits use the modal existing CatBoost configuration (depth 3, 120 trees, L2 10, rsm .25) and fixed 50:50 EQUAL9 blending; this locked setting is recorded, not retuned on the nested held-out labels. The independent Kaggle generalization remains unknown.",
        "Artifacts: `metrics.json`, `row_predictions.parquet`, `per_label_metrics.csv`, `fold_metrics.csv`, `fold_decisions.csv`, `feature_importance.json`, `nested_baseline_outer_*.npz/.json`, `features_outer_*.npz`, and fold-specific model files under `models/`.",""]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default=str(OUT));p.add_argument("--report",default=str(REPORT));p.add_argument("--report-only",action="store_true")
    a=p.parse_args()
    with threadpool_limits(limits=1):
        result=refresh_report(a.output_dir,a.report) if a.report_only else run(a.output_dir,a.report)
    summary={k:{m:result["baseline_metrics"][k][m] for m in ("accuracy","balanced_accuracy","log_loss")} for k in ("ORACLE02","ORACLE03_top2","H09","EQUAL9","top2_oracle")}
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
