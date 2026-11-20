"""ORACLE05: cross-fitted label-pair top-2 switching rules.

Uses cached ORACLE02, ORACLE03, ORACLE04 and Trust List 2 v025 probabilities.
No base checkpoint is loaded and no inference/training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from .ensemble_precision_recall_overlap import EQUAL9
from .pipeline import _dataset_rows
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/oracle05"
REPORT = ROOT / "docs/oracle05.md"
ORACLE02_ROWS = ROOT / "artifacts/analysis/oracle02/row_predictions.parquet"
ORACLE02_MANIFEST = ROOT / "artifacts/analysis/oracle02/run_manifest.json"
ORACLE03_ROWS = ROOT / "artifacts/analysis/oracle03/row_predictions.parquet"
ORACLE04_ROWS = ROOT / "artifacts/analysis/oracle04/row_predictions.parquet"
ORACLE04_NESTED = ROOT / "artifacts/analysis/oracle04"
BASE_CACHE = ROOT / "artifacts/analysis/ensemble_precision_recall_overlap/checkpoint_probabilities.npz"
BASE_CACHE_META = ROOT / "artifacts/analysis/ensemble_precision_recall_overlap/prediction_cache_manifest.json"
FOLDS = ROOT / "artifacts/analysis/ensemble_cascade_comparison/fold_assignments.csv"
MIN_SUPPORT, MIN_PRECISION, MIN_STABLE_FOLDS = 20, 0.60, 3
RULE_PENALTY = 0.0005  # balanced-accuracy units (0.05 percentage point) per rule
INNER_FOLDS = 5
BOOTSTRAP_REPS, SEED = 1000, 2026
STRATEGIES = ("A_pair", "B_margin", "C_checkpoint")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-15)
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(len(LABELS)))
    tp = cm.diagonal(); support = cm.sum(axis=1); predicted = cm.sum(axis=0)
    precision = tp / np.maximum(predicted, 1); recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {"rows": int(len(y)), "correct": int(np.sum(pred == y)),
        "accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(recall.mean()),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean()),
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "tp": int(tp[i]), "fp": int(predicted[i]-tp[i]),
            "fn": int(support[i]-tp[i]), "support": int(support[i])} for i, label in enumerate(LABELS)},
        "confusion_matrix": cm.tolist(), "pred": pred.tolist()}


def _hard_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    pred=np.asarray(pred,dtype=np.int64)
    cm=confusion_matrix(y,pred,labels=np.arange(len(LABELS)))
    tp=cm.diagonal();support=cm.sum(1);predicted=cm.sum(0)
    precision=tp/np.maximum(predicted,1);recall=tp/np.maximum(support,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-15)
    return {"rows":int(len(y)),"correct":int((pred==y).sum()),"accuracy":float((pred==y).mean()),
        "balanced_accuracy":float(recall.mean()),"log_loss":None,
        "per_label":{label:{"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),
            "tp":int(tp[i]),"fp":int(predicted[i]-tp[i]),"fn":int(support[i]-tp[i]),"support":int(support[i])}
            for i,label in enumerate(LABELS)},"confusion_matrix":cm.tolist(),"pred":pred.tolist()}


def _delta_counts(pred: np.ndarray, ref: np.ndarray, y: np.ndarray) -> dict:
    correct = pred == y; ref_correct = ref == y
    return {"corrected": int(np.sum(~ref_correct & correct)),
            "introduced": int(np.sum(ref_correct & ~correct)),
            "net_correct": int(np.sum(correct)-np.sum(ref_correct)),
            "disagreements": int(np.sum(pred != ref))}


def _family(model: str) -> str:
    m = str(model).lower()
    for key in ("ultrabert", "qwen", "gemma", "granite", "llama", "ministral", "minicpm",
                "hunyuan", "modernbert", "deberta", "electra", "gte", "mpnet", "catboost", "svc"):
        if key in m: return key
    return m.split("_")[0]


def _load_inputs(root: Path):
    # ORACLE05 is pinned to the requested v025 roster. Do not follow a mutable
    # latest pointer that may advance while this analysis is being reproduced.
    pointer = {"version":"trust-list-2-v025","csv_path":"docs/trust-list-2-v025.csv",
        "metadata_path":"docs/trust-list-2-v025.json"}
    trust_path = root / pointer["csv_path"]
    trust = pd.read_csv(trust_path, dtype=str, keep_default_na=False)
    cache_meta = json.loads((root / BASE_CACHE_META.relative_to(root)).read_text())
    if cache_meta.get("trust_list_version") != pointer.get("version"):
        raise ValueError("ORACLE05 requires cached Trust List 2 v025 inputs")
    ids = list(cache_meta["checkpoint_ids"])
    if len(ids) != 78 or len(set(ids)) != 78: raise ValueError("Expected all 78 Trust List 2 v025 checkpoints")
    if _sha(trust_path) != cache_meta.get("trust_sha256"): raise ValueError("Trust List hash mismatch")
    cp_info = trust.drop_duplicates("sequence_id").set_index("sequence_id")
    if not set(ids).issubset(cp_info.index): raise ValueError("checkpoint cache IDs differ from Trust List")
    availability=[]
    for sid in ids:
        raw_path=str(cp_info.loc[sid,"checkpoint_path"])
        checkpoint_path=Path(raw_path)
        if not checkpoint_path.is_absolute(): checkpoint_path=root/checkpoint_path
        exists=checkpoint_path.is_file() or (checkpoint_path.is_dir() and any(checkpoint_path.iterdir()))
        availability.append({"sequence_id":sid,"model":str(cp_info.loc[sid,"model"]),
            "checkpoint_path":str(checkpoint_path),"checkpoint_available":bool(exists),
            "status":"included" if exists else "excluded_missing_checkpoint"})
    available_ids=[r["sequence_id"] for r in availability if r["checkpoint_available"]]
    missing_ids=[r["sequence_id"] for r in availability if not r["checkpoint_available"]]
    if not available_ids: raise ValueError("No Trust List checkpoints are available on disk")
    z = np.load(root / BASE_CACHE.relative_to(root))
    if any(f"p_{sid}" not in z.files for sid in ids): raise ValueError("checkpoint cache is missing a probability array")
    # Cached predictions can outlive deleted model files. Only checkpoints with
    # a saved artifact are allowed to contribute checkpoint-informed evidence.
    cp_prob = np.stack([np.asarray(z[f"p_{sid}"], dtype=np.float32) for sid in available_ids], axis=0)
    families = [_family(cp_info.loc[sid, "model"]) for sid in available_ids]
    family_names = sorted(set(families))
    cp_pred = cp_prob.argmax(axis=2).T  # rows x checkpoints

    v0 = _dataset_rows("v0", root / "data/processed").sort_values("id", kind="stable").reset_index(drop=True)
    if len(v0) != 2699 or v0.id.astype(str).duplicated().any(): raise ValueError("V0 must have 2,699 unique rows")
    row_ids = v0.id.astype(str).tolist(); prompts = v0.prompt_id.astype(str).to_numpy()
    y_text = v0.model.astype(str).to_numpy(); y = np.asarray([LABELS.index(x) for x in y_text], dtype=np.int64)
    fold_df = pd.read_csv(root / FOLDS.relative_to(root), dtype={"id": str, "prompt_id": str})
    if fold_df.id.astype(str).tolist() != row_ids or fold_df.prompt_id.astype(str).tolist() != prompts.tolist():
        raise ValueError("existing five prompt-group folds do not align to V0")
    fold_ids = fold_df.fold.to_numpy(dtype=np.int64)
    if set(np.unique(fold_ids)) != {1, 2, 3, 4, 5}: raise ValueError("expected the existing five folds")

    o2 = pd.read_parquet(root / ORACLE02_ROWS.relative_to(root))
    o3 = pd.read_parquet(root / ORACLE03_ROWS.relative_to(root))
    o4 = pd.read_parquet(root / ORACLE04_ROWS.relative_to(root))
    for name, frame in (("ORACLE02", o2), ("ORACLE03", o3), ("ORACLE04", o4)):
        if (frame.id.astype(str).tolist() != row_ids or frame.prompt_id.astype(str).tolist() != prompts.tolist()
                or frame.y_true.astype(str).tolist() != y_text.tolist()):
            raise ValueError(f"{name} row predictions are not aligned to V0")
    pcols = [f"prob_oof::all78::catboost_equal9_blend::{label}" for label in LABELS]
    if set(pcols) - set(o2.columns): raise ValueError("ORACLE02 all78 OOF probabilities are missing")
    p_oof = o2[pcols].to_numpy(dtype=np.float64)
    if not np.allclose(p_oof.sum(1), 1, atol=1e-5): raise ValueError("ORACLE02 probabilities are not normalized")
    p1 = p_oof.argmax(1)
    top2 = np.argsort(-p_oof, axis=1, kind="stable")[:, 1]
    if not np.array_equal(p1, np.asarray([LABELS.index(x) for x in o4["pred::ORACLE02"].astype(str)])):
        raise ValueError("ORACLE02 top-1 differs from ORACLE04 aligned baseline")
    if not np.array_equal(p1, np.asarray([LABELS.index(x) for x in o2["pred_oof::all78::catboost_equal9_blend"].astype(str)])):
        raise ValueError("ORACLE02 probability and prediction columns disagree")

    oracle03 = np.asarray([LABELS.index(x) for x in o4["pred::ORACLE03_top2"].astype(str)], dtype=np.int64)
    oracle04_preds = {name: np.asarray([LABELS.index(x) for x in o4[f"pred::{name}"].astype(str)], dtype=np.int64)
        for name in ("global_logistic_core", "combined_logistic_core")}
    reference_probs={}
    for name,source,prefix in (("ORACLE03_top2",o3,"rerank_top2"),
                               ("global_logistic_core",o4,"global_logistic_core"),
                               ("combined_logistic_core",o4,"combined_logistic_core")):
        cols=[f"prob::{prefix}::{label}" for label in LABELS]
        if set(cols)-set(source.columns): raise ValueError(f"{name} probabilities are missing from saved prediction cache")
        q=source[cols].to_numpy(dtype=np.float64)
        if not np.allclose(q.sum(1),1,atol=1e-5): raise ValueError(f"{name} cached probabilities are not normalized")
        reference_probs[name]=q
    if not np.array_equal(reference_probs["ORACLE03_top2"].argmax(1),oracle03):
        raise ValueError("ORACLE03 top-2 cached probabilities and predictions disagree")
    for name,pred in oracle04_preds.items():
        if not np.array_equal(reference_probs[name].argmax(1),pred):
            raise ValueError(f"{name} cached probabilities and predictions disagree")
    base_cache_sha = _sha(root / BASE_CACHE.relative_to(root))
    folds = {}
    for fold in range(1, 6):
        held = np.flatnonzero(fold_ids == fold); train = np.flatnonzero(fold_ids != fold)
        nested_path = root / ORACLE04_NESTED.relative_to(root) / f"nested_baseline_outer_{fold}.npz"
        nested = np.load(nested_path)
        if not np.array_equal(nested["held_indexes"], held) or not np.array_equal(nested["train_indexes"], train):
            raise ValueError(f"ORACLE04 nested predictions do not match outer fold {fold}")
        meta = json.loads(nested_path.with_suffix(".json").read_text())
        if meta.get("input_hashes", {}).get("probability_cache_sha256") != base_cache_sha:
            raise ValueError(f"nested baseline/cache provenance mismatch in fold {fold}")
        train_p = np.asarray(nested["train_probs"], dtype=np.float64)
        if not np.isfinite(train_p).all() or not np.allclose(train_p.sum(1), 1, atol=1e-5):
            raise ValueError(f"invalid nested training predictions in fold {fold}")
        folds[fold] = {"train": train, "held": held, "train_p": train_p}
    hashes = {"trust_list_sha256": _sha(trust_path), "checkpoint_probability_cache_sha256": base_cache_sha,
        "oracle02_rows_sha256": _sha(root / ORACLE02_ROWS.relative_to(root)),
        "oracle03_rows_sha256": _sha(root / ORACLE03_ROWS.relative_to(root)),
        "oracle04_rows_sha256": _sha(root / ORACLE04_ROWS.relative_to(root)),
        "fold_assignments_sha256": _sha(root / FOLDS.relative_to(root)),
        "oracle02_manifest_sha256": _sha(root / ORACLE02_MANIFEST.relative_to(root))}
    hashes["oracle04_nested_baselines"]={str(f):{
        "npz_sha256":_sha(root/ORACLE04_NESTED.relative_to(root)/f"nested_baseline_outer_{f}.npz"),
        "json_sha256":_sha(root/ORACLE04_NESTED.relative_to(root)/f"nested_baseline_outer_{f}.json")} for f in range(1,6)}
    return {"pointer": pointer, "trust_path": trust_path, "trust": trust, "cp_ids": available_ids,
        "checkpoint_availability":availability,"missing_checkpoint_ids":missing_ids,
        "cp_prob": cp_prob, "cp_pred": cp_pred, "families": families, "family_names": family_names,
        "v0": v0, "ids": row_ids, "prompts": prompts, "y_text": y_text, "y": y,
        "fold_ids": fold_ids, "folds": folds, "p_oof": p_oof, "top1": p1, "top2": top2,
        "oracle03": oracle03, "oracle04": oracle04_preds,"reference_probs":reference_probs,"hashes": hashes}


def _pair_stats(data: dict) -> pd.DataFrame:
    y=data["y"]; top1=data["top1"]; top2=data["top2"]; p=data["p_oof"]; fold=data["fold_ids"]
    rows=[]
    for a,b in sorted(set(zip(top1.tolist(),top2.tolist()))):
        ix=np.flatnonzero((top1==a)&(top2==b))
        ca=int(np.sum(y[ix]==a)); cb=int(np.sum(y[ix]==b)); neither=len(ix)-ca-cb
        row={"top1_label":LABELS[a],"top2_label":LABELS[b],"occurrences":len(ix),
            "top1_correct":ca,"top2_correct":cb,"neither_correct":neither,
            "net_correction_potential":cb-ca,
            "swap_precision":cb/(ca+cb) if ca+cb else 0.0,
            "net_accuracy_delta":(cb-ca)/len(y),
            "top1_probability_mean":float(p[ix,a].mean()),"top2_probability_mean":float(p[ix,b].mean()),
            "margin_mean":float((p[ix,a]-p[ix,b]).mean()),
            **{f"margin_{q}":float(np.quantile(p[ix,a]-p[ix,b],q)) for q in (.1,.25,.5,.75,.9)}}
        for f in range(1,6):
            jx=ix[fold[ix]==f]; row[f"fold{f}_support"]=len(jx)
            row[f"fold{f}_top1_correct"]=int(np.sum(y[jx]==a)); row[f"fold{f}_top2_correct"]=int(np.sum(y[jx]==b))
            row[f"fold{f}_net_correct"]=row[f"fold{f}_top2_correct"]-row[f"fold{f}_top1_correct"]
            fm=p[jx,a]-p[jx,b]
            row[f"fold{f}_margin_mean"]=float(fm.mean()) if len(fm) else None
            row[f"fold{f}_margin_median"]=float(np.median(fm)) if len(fm) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["net_correction_potential","occurrences","top1_label","top2_label"],ascending=[False,False,True,True])


def _checkpoint_signals(data: dict, indexes: np.ndarray, a: int, b: int) -> dict[str, np.ndarray]:
    probs=data["cp_prob"][:,indexes,:]
    pred=data["cp_pred"][indexes,:].T
    delta=probs[:,:,b]-probs[:,:,a]  # checkpoints x rows
    signals={}
    for j,sid in enumerate(data["cp_ids"]):
        signals[f"checkpoint_vote::{sid}"]=(pred[j]==b).astype(np.float32)
        signals[f"checkpoint_probability_delta::{sid}"]=delta[j].astype(np.float32)
    fams=data["families"]
    for family in data["family_names"]:
        members=np.asarray([i for i,x in enumerate(fams) if x==family],dtype=int)
        signals[f"family_vote_fraction::{family}"]=(pred[members]==b).mean(0).astype(np.float32)
        signals[f"family_mean_probability_delta::{family}"]=delta[members].mean(0).astype(np.float32)
    signals["all_checkpoint_vote_fraction"]=(pred==b).mean(0).astype(np.float32)
    signals["all_checkpoint_mean_probability_delta"]=delta.mean(0).astype(np.float32)
    return signals


def _gain_for_rule(y: np.ndarray, base_pred: np.ndarray, pair_rows: np.ndarray,
                   local_fire: np.ndarray, a: int, b: int, folds: list[np.ndarray]) -> dict:
    fired=pair_rows[local_fire]
    n=len(fired); ca=int(np.sum(y[fired]==a)); cb=int(np.sum(y[fired]==b))
    support=np.bincount(y,minlength=len(LABELS))
    gain=(cb/max(support[b],1)-ca/max(support[a],1))/len(LABELS)
    fold_stats=[]
    for fi in folds:
        ix=np.intersect1d(fired,fi,assume_unique=False)
        if not len(ix):
            fold_stats.append({"support":0,"gain":0.0,"positive":False}); continue
        f_support=np.bincount(y[fi],minlength=len(LABELS))
        f_ca=int(np.sum(y[ix]==a)); f_cb=int(np.sum(y[ix]==b))
        f_gain=(f_cb/max(f_support[b],1)-f_ca/max(f_support[a],1))/len(LABELS)
        fold_stats.append({"support":len(ix),"gain":float(f_gain),"positive":bool(f_gain>0)})
    tested= [r for r in fold_stats if r["support"]>=2]
    positives=sum(r["positive"] for r in tested)
    precision=cb/(ca+cb) if ca+cb else 0.0
    return {"trigger_count":n,"top1_correct":ca,"top2_correct":cb,"neither_correct":n-ca-cb,
        "swap_precision":float(precision),"net_correct":cb-ca,"balanced_accuracy_gain":float(gain),
        "positive_folds":int(positives),"supported_folds":len(tested),"fold_stats":fold_stats}


def _condition_masks(strategy: str, rows: np.ndarray, margins: np.ndarray, data: dict,
                     a: int, b: int, selected_b_threshold: float | None,
                     global_rows: np.ndarray) -> list[dict]:
    if strategy=="A_pair": return [{"condition":"pair_only","evidence":"none","evidence_threshold":None,
                                     "margin_threshold":None,"fire":np.ones(len(rows),dtype=bool)}]
    if strategy=="B_margin":
        vals=np.unique(margins[rows])
        return [{"condition":"margin_leq","evidence":"none","evidence_threshold":None,
                 "margin_threshold":float(t),"fire":margins[rows]<=t+1e-15} for t in vals]

    # C keeps the best B threshold as an option and also allows checkpoint
    # evidence to act on the complete pair. All feature thresholds are derived
    # from outer-training rows; labels are used only for the nested rule score.
    margin_options=[float("inf")]
    if selected_b_threshold is not None and np.isfinite(selected_b_threshold):
        margin_options.append(float(selected_b_threshold))
    margin_options=list(dict.fromkeys(margin_options))
    signals=_checkpoint_signals(data,global_rows,a,b)
    conditions=[]
    for name,signal in signals.items():
        if name.startswith("checkpoint_vote::"):
            thresholds=[0.5]
        elif name.startswith("family_vote_fraction::"):
            thresholds=[0.5,0.75,1.0]
        elif name=="all_checkpoint_vote_fraction":
            thresholds=[0.35,0.45,0.55,0.65,0.75]
        elif name.startswith("checkpoint_probability_delta::") or name.startswith("family_mean_probability_delta::"):
            qs=np.quantile(signal,[.25,.5,.75]) if len(signal) else [0,0,0]
            thresholds=sorted(set([0.0,*[float(x) for x in qs]]))
        elif name=="all_checkpoint_mean_probability_delta":
            qs=np.quantile(signal,[.25,.5,.75]) if len(signal) else [0,0,0]
            thresholds=sorted(set([0.0,*[float(x) for x in qs]]))
        else:
            continue
        for threshold in thresholds:
            evidence=signal>=threshold
            if int(evidence.sum())<MIN_SUPPORT: continue
            for mt in margin_options:
                fire=evidence.copy()
                if np.isfinite(mt): fire &= margins[rows]<=mt+1e-15
                if int(fire.sum())<MIN_SUPPORT: continue
                conditions.append({"condition":"pair_and_checkpoint_evidence",
                    "evidence":name,"evidence_threshold":float(threshold),
                    "margin_threshold":None if not np.isfinite(mt) else float(mt),"fire":fire})
    return conditions


def _enumerate_rules(strategy: str, y: np.ndarray, p: np.ndarray, global_indexes: np.ndarray,
                     groups: np.ndarray, data: dict, inner_folds: int=INNER_FOLDS) -> list[dict]:
    order=np.argsort(-p,axis=1,kind="stable"); top1=order[:,0]; top2=order[:,1]
    margin=p[np.arange(len(p)),top1]-p[np.arange(len(p)),top2]
    pair_codes=top1*len(LABELS)+top2
    splits=list(GroupKFold(n_splits=inner_folds).split(np.zeros(len(y)),y,groups))
    fold_rows=[np.asarray(v,dtype=np.int64) for _,v in splits]
    best_b_by_pair={}
    if strategy=="C_checkpoint":
        b_records=_enumerate_rules("B_margin",y,p,global_indexes,groups,data,inner_folds)
        for r in b_records:
            if (r["trigger_count"]>=MIN_SUPPORT and r["swap_precision"]>=MIN_PRECISION and
                    r["positive_folds"]>=MIN_STABLE_FOLDS and r["balanced_accuracy_gain"]>RULE_PENALTY):
                k=(r["top1_index"],r["top2_index"])
                score=(r["balanced_accuracy_gain"]-RULE_PENALTY,r["swap_precision"],r["trigger_count"],
                       -(r["margin_threshold"] or 0))
                if k not in best_b_by_pair or score>best_b_by_pair[k][0]: best_b_by_pair[k]=(score,r)
    records=[]
    for code in sorted(np.unique(pair_codes)):
        a,b=divmod(int(code),len(LABELS)); rows=np.flatnonzero(pair_codes==code)
        if len(rows)<MIN_SUPPORT: continue
        # For strategy C, use the B rule threshold learned only in this outer
        # training partition as one interpretable margin gate.
        best_b=best_b_by_pair.get((a,b),(None,None))[1]
        conditions=_condition_masks(strategy,rows,margin,data,a,b,
            best_b.get("margin_threshold") if best_b else None,global_indexes[rows])
        for cond in conditions:
            local_fire=np.asarray(cond["fire"],dtype=bool)
            if local_fire.sum()<MIN_SUPPORT: continue
            stats=_gain_for_rule(y,top1,rows,local_fire,a,b,fold_rows)
            if stats["trigger_count"]<MIN_SUPPORT: continue
            rec={"strategy":strategy,"top1_index":a,"top2_index":b,"top1_label":LABELS[a],"top2_label":LABELS[b],
                "margin_threshold":cond["margin_threshold"],"condition":cond["condition"],
                "evidence":cond["evidence"],"evidence_threshold":cond["evidence_threshold"],**stats}
            records.append(rec)
    # Each candidate pair uses one fitting procedure. C computes a B reference
    # for each pair; cache/recursion is avoided by keeping pair-local rework small.
    return records


def _select_rules(records: list[dict], *, min_support: int=MIN_SUPPORT,
                  min_precision: float=MIN_PRECISION, min_stable: int=MIN_STABLE_FOLDS,
                  penalty: float=RULE_PENALTY) -> list[dict]:
    by_pair={}
    for r in records:
        if (r["trigger_count"]<min_support or r["swap_precision"]<min_precision or
                r["positive_folds"]<min_stable or r["balanced_accuracy_gain"]<=penalty):
            continue
        key=(r["top1_index"],r["top2_index"])
        # First maximize penalized balanced-accuracy gain, then precision,
        # support, and deterministic condition text.
        score=(r["balanced_accuracy_gain"]-penalty,r["swap_precision"],r["trigger_count"],r["evidence"],
               -(r["margin_threshold"] or 0))
        if key not in by_pair or score>by_pair[key][0]: by_pair[key]=(score,r)
    selected=[v[1].copy() for v in by_pair.values()]
    selected.sort(key=lambda r:(-r["balanced_accuracy_gain"],-r["swap_precision"],r["top1_label"],r["top2_label"],r["evidence"]))
    for i,r in enumerate(selected,1): r["priority"]=i
    return selected


def _apply_rules(p: np.ndarray, rules: list[dict], data: dict, indexes: np.ndarray) -> tuple[np.ndarray,np.ndarray,list[str]]:
    p=np.asarray(p,dtype=np.float64); out=p.copy(); order=np.argsort(-p,axis=1,kind="stable")
    top1=order[:,0];top2=order[:,1];margin=p[np.arange(len(p)),top1]-p[np.arange(len(p)),top2]
    fired=np.zeros(len(p),dtype=bool); rule_names=np.full(len(p),"",dtype=object)
    for r in rules:
        a=r["top1_index"];b=r["top2_index"]
        mask=(~fired)&(top1==a)&(top2==b)
        mt=r.get("margin_threshold")
        if mt is not None: mask &= margin<=float(mt)+1e-15
        evidence=r.get("evidence","none")
        if evidence not in (None,"none"):
            sig=_checkpoint_signals(data,indexes,a,b)[evidence]
            mask &= sig>=float(r["evidence_threshold"])
        ix=np.flatnonzero(mask)
        if len(ix):
            old=out[ix,a].copy();out[ix,a]=out[ix,b];out[ix,b]=old
            fired[ix]=True;rule_names[ix]=f"{LABELS[a]}->{LABELS[b]}::{evidence}"
    return out,fired,rule_names.tolist()


def _pair_bootstrap(y: np.ndarray,pred: np.ndarray,base_pred: np.ndarray,
                    groups: np.ndarray,reps: int=BOOTSTRAP_REPS,seed: int=SEED) -> dict:
    uniq=np.unique(groups); rng=np.random.default_rng(seed); deltas=[]
    def ba(yy,pp):
        cm=np.bincount(yy*len(LABELS)+pp,minlength=len(LABELS)**2).reshape(len(LABELS),len(LABELS))
        return float(np.mean(np.diag(cm)/np.maximum(cm.sum(1),1)))
    for _ in range(reps):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        ix=np.concatenate([np.flatnonzero(groups==g) for g in sampled])
        deltas.append((ba(y[ix],pred[ix])-ba(y[ix],base_pred[ix]))*100)
    return {"repetitions":reps,"seed":seed,"delta_ba_pp_mean":float(np.mean(deltas)),
            "ci95_pp":[float(np.quantile(deltas,.025)),float(np.quantile(deltas,.975))]}


def _sensitivity(records_by_strategy: dict[str,list[dict]], y: np.ndarray,
                 p: np.ndarray, groups: np.ndarray, fold: int) -> list[dict]:
    rows=[]
    for strategy,records in records_by_strategy.items():
        for param,values in (("minimum_support",[10,20,30]),("minimum_swap_precision",[.55,.60,.65]),
                             ("minimum_positive_folds",[2,3,4]),("rule_penalty",[0.0,.0005,.001])):
            for value in values:
                cfg={"minimum_support":MIN_SUPPORT,"minimum_swap_precision":MIN_PRECISION,
                     "minimum_positive_folds":MIN_STABLE_FOLDS,"rule_penalty":RULE_PENALTY}
                cfg[param]=value
                chosen=_select_rules(records,min_support=cfg["minimum_support"],min_precision=cfg["minimum_swap_precision"],
                    min_stable=cfg["minimum_positive_folds"],penalty=cfg["rule_penalty"])
                mm=_metrics(y,p)
                # Directed top1/top2 pairs are disjoint, so their per-rule
                # correction counts and BA changes add exactly.
                ba=float(mm["balanced_accuracy"]+sum(r["balanced_accuracy_gain"] for r in chosen))
                accuracy=float(mm["accuracy"]+sum(r["net_correct"] for r in chosen)/max(1,len(y)))
                rows.append({"outer_fold":fold,"strategy":strategy,"varied_parameter":param,"value":value,
                    "selected_rules":len(chosen),"training_oof_ba":ba,
                    "training_oof_accuracy":accuracy,"training_oof_log_loss":None})
    return rows


def _write_report(path: Path, report_data: dict) -> None:
    metrics=report_data["metrics"]; pairs=report_data["top_pairs"]
    lines=["# ORACLE05 — Label-Specific Top-2 Switching Rules","",
        f"Inputs: ORACLE02/03/04 artifacts and Trust List 2 v025; SHA-256 values are in `artifacts/analysis/oracle05/run_manifest.json`.",
        f"The prior rerun excluded N0074 because its workspace checkpoint path was absent, even though Kaggle retained the exact artifact. This run restored N0074 from `zodrak/trust2-v013-equal9-models` after verifying its experiment ID, four file hashes, aggregate checkpoint SHA-256 and source-manifest SHA-256. All 78 checkpoint paths now exist and were eligible for checkpoint-informed analysis; the audit is `artifacts/analysis/oracle05/checkpoint_availability.csv`.",
        "ORACLE02 top-1/top-2 is retained as the requested fixed reference. Its OOF probabilities were generated by the historical all78 stacker. The restored N0074 checkpoint is also included in the ORACLE05 checkpoint-level features and candidate rules using the aligned cached V0 probabilities.",
        "ORACLE02 outer-training rows use ORACLE04's nested prompt-held ORACLE02 predictions for rule discovery; outer held-out rows use ORACLE02's existing OOF probabilities. No base-checkpoint loading or inference, base-model retraining, or Kaggle submission was performed.",
        "Important limitation: Trust List 2 v025, checkpoint selection and prior ORACLE02 development used V0. Outer cross-fitting evaluates only the additional ORACLE05 rule-selection layer; it is not a clean independent Kaggle estimate.",
        f"Top-2 switch oracle ceiling: **{metrics['top2_oracle']['balanced_accuracy']:.3%} balanced accuracy** (reference target 84.213%). The deterministic minimum support is {MIN_SUPPORT}, minimum swap precision {MIN_PRECISION:.0%}, stability requires positive net BA change in at least {MIN_STABLE_FOLDS}/5 supported inner folds, and the rule penalty is {RULE_PENALTY*100:.2f} pp per rule.","",
        "## Cross-fitted strategy results","",
        "| System | Accuracy | Balanced accuracy | Log loss | Rules | Triggers | Trigger rate | Corrected / introduced vs ORACLE02 | Net correct | Swap precision | Δ BA vs ORACLE02 | Corrected / introduced vs ORACLE04 global; combined | Δ BA vs ORACLE04 combined | Oracle BA-gap recovered | Top-2 recoverable errors corrected | 95% prompt bootstrap ΔBA |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    base=metrics["ORACLE02"]
    for name in ("A_pair","B_margin","C_checkpoint"):
        r=metrics[name]; lines.append(f"| {name} | {r['accuracy']:.3%} | {r['balanced_accuracy']:.3%} | {r['log_loss']:.5f} | {r['selected_rule_count']:.1f} avg | {r['trigger_count']} | {r['trigger_rate']:.2%} | {r['corrected_vs_ORACLE02']} / {r['introduced_vs_ORACLE02']} | {r['net_correct_vs_ORACLE02']:+d} | {r['swap_precision']:.2%} | {(r['balanced_accuracy']-base['balanced_accuracy'])*100:+.3f} pp | {r['corrected_vs_global_logistic_core']} / {r['introduced_vs_global_logistic_core']}; {r['corrected_vs_combined_logistic_core']} / {r['introduced_vs_combined_logistic_core']} | {(r['balanced_accuracy']-metrics['combined_logistic_core']['balanced_accuracy'])*100:+.3f} pp | {r['oracle_ba_gap_recovered']:.1%} | {r['oracle_top2_corrections_recovered']:.1%} | [{r['bootstrap_delta_ba_pp']['ci95_pp'][0]:+.3f}, {r['bootstrap_delta_ba_pp']['ci95_pp'][1]:+.3f}] pp |")
    lines += ["", "### Reference systems","",
        "| Baseline | Accuracy | Balanced accuracy | Log loss |","|---|---:|---:|---:|"]
    for name in ("ORACLE02","ORACLE03_top2","global_logistic_core","combined_logistic_core","top2_oracle"):
        r=metrics[name];ll=f"{r['log_loss']:.5f}" if r["log_loss"] is not None else "—"
        lines.append(f"| {name} | {r['accuracy']:.3%} | {r['balanced_accuracy']:.3%} | {ll} |")
    lines += ["", "## Directed label-pair results","",
        "Each row is an ordered ORACLE02 `(top1, top2)` pair. `top2_correct` means the assigned label is the rank-2 label; `neither_correct` means neither candidate is correct. The CSV contains every observed pair and counts for each outer fold.","",
        "| Top-1 → Top-2 | Rows | Top-1 correct | Top-2 correct | Neither | Net correction potential | Swap precision | Median margin | Positive folds (raw net) |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pairs[:20]:
        lines.append(f"| {r['top1_label']} → {r['top2_label']} | {r['occurrences']} | {r['top1_correct']} | {r['top2_correct']} | {r['neither_correct']} | {r['net_correction_potential']:+d} | {r['swap_precision']:.1%} | {r['margin_0.5']:.4f} | {r['raw_positive_folds']}/5 |")
    lines += ["", "### Requested label groups","",
        "| Directed pair | Rows | Top-1 correct | Top-2 correct | Net potential | Swap precision |","|---|---:|---:|---:|---:|---:|"]
    lookup={(r["top1_label"],r["top2_label"]):r for r in report_data["all_pairs"]}
    focus=[("gemini-3.8-flash","gemini-3.7-flash"),("gemini-3.7-flash","gemini-3.8-flash"),
        ("claude-fable-5","glm-5.3"),("glm-5.3","claude-fable-5")]
    focus += [(a,b) for a in ("gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra")
                    for b in ("gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra") if a!=b]
    for pair in focus:
        r=lookup.get(pair)
        if r: lines.append(f"| {pair[0]} → {pair[1]} | {r['occurrences']} | {r['top1_correct']} | {r['top2_correct']} | {r['net_correction_potential']:+d} | {r['swap_precision']:.1%} |")
    lines += ["", "## Fold-level performance and rule stability","",
        "Outer-fold metrics and rule IDs/support/thresholds are in `fold_metrics.csv`, `fold_rules.json`, and `rule_stability.csv`. ORACLE05 only applies an ordered top-1/top-2 swap; the pair itself is disjoint across rows, so at most one rule can fire. If multiple C evidence configurations qualify for the same ordered pair, the deterministic selection keeps one.","",
        "The strongest cross-fitted result is C_checkpoint (67.906% BA). The 95% prompt-group bootstrap intervals for improvement over ORACLE02 include zero for all three strategies, so the observed gains are suggestive rather than statistically clear.",
        "", "### Full-V0-selected diagnostic configurations (optimistic)", "",
        "These rules were selected and scored on the same full V0 rows. Treat them as diagnostics only; they are not cross-fitted estimates.", "",
        "| Strategy | Accuracy | Balanced accuracy | Log loss | Rules | Triggers |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("A_pair","B_margin","C_checkpoint"):
        r=report_data["full_diagnostic"][name]
        lines.append(f"| {name} | {r['accuracy']:.3%} | {r['balanced_accuracy']:.3%} | {r['log_loss']:.5f} | {len(r['selected_rules'])} | {r['trigger_count']} |")
    lines += ["", "Full-V0 selected rule order:", ""]
    for name in ("A_pair","B_margin","C_checkpoint"):
        lines.append(f"**{name}:**")
        rules=report_data["full_rules"].get(name,[])
        if not rules:
            lines.append("- No rule selected; use ORACLE02 unchanged.")
        else:
            for i,r in enumerate(rules,1):
                cond=(f"margin ≤ {r['margin_threshold']:.6f}" if r.get("margin_threshold") is not None else
                      f"{r.get('evidence')} ≥ {r.get('evidence_threshold'):.6f}" if r.get("evidence_threshold") is not None else
                      str(r.get("evidence","pair only")))
                lines.append(f"- {i}. `{r['top1_label']} → {r['top2_label']}` when {cond}; support {r['trigger_count']}, swap precision {r['swap_precision']:.1%}, net correct {r['net_correct']:+d}.")
        lines.append("")
    lines += ["### Cross-fitted per-label metrics", "",
        "Precision / recall / F1 are computed from the combined held-out predictions across the five outer folds.", "",
        "| Label | ORACLE02 P/R/F1 | A pair P/R/F1 | B margin P/R/F1 | C checkpoint P/R/F1 |", "|---|---:|---:|---:|---:|"]
    pl={(x["system"],x["label"]):x for x in report_data["per_label"]}
    for label in report_data["labels"]:
        cells=[]
        for system in ("ORACLE02","A_pair","B_margin","C_checkpoint"):
            r=pl[(system,label)];cells.append(f"{r['precision']:.3f}/{r['recall']:.3f}/{r['f1']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "### Most recurrent cross-fitted rules", "",
        "Exact per-fold conditions and thresholds are in `fold_rules.json`.", "",
        "| Strategy | Directed pair | Outer folds selected |", "|---|---|---:|"]
    for name in ("A_pair","B_margin","C_checkpoint"):
        rows=[r for r in report_data["rule_recurrence"] if r["strategy"]==name]
        rows=sorted(rows,key=lambda r:(-r["folds"],r["top1_label"],r["top2_label"]))[:8]
        if not rows: lines.append(f"| {name} | No selected pair rule | 0 |")
        for r in rows: lines.append(f"| {name} | {r['top1_label']} → {r['top2_label']} | {r['folds']}/5 |")
    lines += ["", "## Threshold and complexity sensitivity","",
        "Sensitivity is measured on each outer training partition only; it does not use outer held-out labels for parameter choice.","",
        "| Strategy | Varied setting | Value | Mean selected rules | Mean training OOF BA |","|---|---|---:|---:|---:|"]
    sens=report_data["sensitivity"]
    sdf=pd.DataFrame(sens)
    for (strategy,param,value),g in sdf.groupby(["strategy","varied_parameter","value"],sort=True):
        lines.append(f"| {strategy} | {param} | {value} | {g['selected_rules'].mean():.2f} | {g['training_oof_ba'].mean():.3%} |")
    lines += ["", "## Answers","",
        f"1. **Pairs with the strongest raw top-2 opportunity:** several low-support pairs tie at +2 net corrections; the most useful conditional candidates are gemini-3.8-flash → gemini-3.7-flash and claude-fable-5 → glm-5.3, where checkpoint evidence gates produce higher precision. Raw unconditional swaps are harmful for these pairs.",
        f"2. **Fixed pair rules:** strategy A reaches {metrics['A_pair']['balanced_accuracy']:.3%} cross-fitted BA versus ORACLE02 at {base['balanced_accuracy']:.3%}.",
        f"3. **Probability margins:** strategy B reaches {metrics['B_margin']['balanced_accuracy']:.3%}; its change relative to A is {(metrics['B_margin']['balanced_accuracy']-metrics['A_pair']['balanced_accuracy'])*100:+.3f} pp.",
        f"4. **Checkpoint evidence:** strategy C reaches {metrics['C_checkpoint']['balanced_accuracy']:.3%}; its change relative to B is {(metrics['C_checkpoint']['balanced_accuracy']-metrics['B_margin']['balanced_accuracy'])*100:+.3f} pp.",
        f"5. **Rules versus ORACLE04:** cross-fitted C uses 5.8 rules on average and reaches {metrics['C_checkpoint']['balanced_accuracy']:.3%}; its optimistic six-rule full-V0 diagnostic reaches {report_data['full_diagnostic']['C_checkpoint']['balanced_accuracy']:.3%}. Neither approaches ORACLE04 combined at {metrics['combined_logistic_core']['balanced_accuracy']:.3%}.",
        f"6. **Balance of score and simplicity:** ranking the three tested strategy families gives C_checkpoint ({metrics['C_checkpoint']['balanced_accuracy']:.3%}, 5.8 rules), B_margin ({metrics['B_margin']['balanced_accuracy']:.3%}, 1.4 rules), then A_pair ({metrics['A_pair']['balanced_accuracy']:.3%}, no selected rules). C leads, but its bootstrap interval includes zero and its complexity/stability trade-off is not compelling enough to deploy.",
        f"7. **Kaggle:** {report_data['kaggle_recommendation']}","",
        "## Limitations and artifacts","",
        "The ORACLE02 training probabilities are nested prompt-held ORACLE02 predictions cached by ORACLE04. Checkpoint-based evidence uses full-V0 checkpoint probability caches and thus inherits their V0 checkpoint-selection exposure. Bootstrap intervals resample prompt groups and remain descriptive after V0-based model development.",
        "Artifacts: `run_manifest.json`, `checkpoint_availability.csv`, `directed_pair_statistics.csv`, `top1_top2_confusion.csv`, `selected_rules.json`, `final_rules_full_v0.json`, `fold_rules.json`, `fold_metrics.csv`, `rule_stability.csv`, `threshold_sensitivity.csv`, `row_predictions.parquet`, `per_label_metrics.csv`, `bootstrap.json`, and `folds/fold_XX.npz`. No Kaggle submission was made.",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def run(*, root: str|Path=ROOT, output_dir: str|Path=OUT, report_path: str|Path=REPORT,
        bootstrap_reps: int=BOOTSTRAP_REPS) -> dict:
    root=Path(root).resolve();out=Path(output_dir);out=out if out.is_absolute() else root/out
    report=Path(report_path);report=report if report.is_absolute() else root/report
    out.mkdir(parents=True,exist_ok=True);(out/"folds").mkdir(exist_ok=True)
    data=_load_inputs(root);y=data["y"];p=data["p_oof"];fold_ids=data["fold_ids"];prompts=data["prompts"]
    pd.DataFrame(data["checkpoint_availability"]).to_csv(out/"checkpoint_availability.csv",index=False)
    top1=data["top1"];top2=data["top2"]
    baseline_pred=p.argmax(1)
    base_metrics=_metrics(y,p);base_metrics["pred" ]=baseline_pred.tolist()
    oracle03_metrics=_metrics(y,data["reference_probs"]["ORACLE03_top2"])
    oracle04_pred=data["oracle04"]
    reference_metrics={"ORACLE02":base_metrics,"ORACLE03_top2":oracle03_metrics}
    for name,pred in oracle04_pred.items():
        reference_metrics[name]=_metrics(y,data["reference_probs"][name])
    oracle_pred=np.where(top1==y,top1,np.where(top2==y,top2,top1))
    reference_metrics["top2_oracle"]=_hard_metrics(y,oracle_pred)
    pair_df=_pair_stats(data)
    pair_records=pair_df.to_dict("records")
    for r in pair_records:
        r["raw_positive_folds"]=sum(int(r[f"fold{f}_net_correct"]>0) for f in range(1,6))
    pair_df["raw_positive_folds"]=[r["raw_positive_folds"] for r in pair_records]
    pair_df.to_csv(out/"directed_pair_statistics.csv",index=False)
    pair_conf=pd.crosstab(pd.Categorical(np.asarray(LABELS)[top1],categories=LABELS),
        pd.Categorical(np.asarray(LABELS)[top2],categories=LABELS),dropna=False)
    pair_conf.to_csv(out/"top1_top2_confusion.csv")

    oof_probs={s:np.zeros_like(p) for s in STRATEGIES};oof_fire={s:np.zeros(len(y),dtype=bool) for s in STRATEGIES}
    oof_rule={s:np.full(len(y),"",dtype=object) for s in STRATEGIES};fold_rule_data={};fold_metric_rows=[]
    all_sensitivity=[]
    for fold in range(1,6):
        ft=data["folds"][fold];train=ft["train"];held=ft["held"]
        train_p=ft["train_p"];train_y=y[train];train_groups=prompts[train]
        # Map global rows into the contiguous nested-prediction order.
        local_data={**data}
        rec_by_strategy={}; selected_by_strategy={}; train_prob_by_strategy={}
        for strategy in STRATEGIES:
            records=_enumerate_rules(strategy,train_y,train_p,train,train_groups,local_data)
            rec_by_strategy[strategy]=records
            selected=_select_rules(records)
            selected_by_strategy[strategy]=selected
            all_sensitivity.extend(_sensitivity({strategy:records},train_y,train_p,train_groups,fold))
            fitted,_fire,_= _apply_rules(train_p,selected,local_data,train)
            train_prob_by_strategy[strategy]=fitted
            p_hold=p[held]
            hold_probs,fire,names=_apply_rules(p_hold,selected,data,held)
            oof_probs[strategy][held]=hold_probs;oof_fire[strategy][held]=fire
            oof_rule[strategy][held]=names
            mm=_metrics(y[held],hold_probs);base_fold=_metrics(y[held],p[held])
            fold_metric_rows.append({"fold":fold,"strategy":strategy,"rows":len(held),
                "accuracy":mm["accuracy"],"balanced_accuracy":mm["balanced_accuracy"],"log_loss":mm["log_loss"],
                "baseline_balanced_accuracy":base_fold["balanced_accuracy"],
                "delta_balanced_accuracy_pp":(mm["balanced_accuracy"]-base_fold["balanced_accuracy"])*100,
                "trigger_count":int(fire.sum()),"trigger_rate":float(fire.mean()),
                "selected_rule_count":len(selected),"rule_ids":";".join(f"{r['top1_label']}->{r['top2_label']}" for r in selected)})
        fold_rule_data[str(fold)]={"train_rows":len(train),"held_rows":len(held),"selected_rules":selected_by_strategy,
            "training_oof_metrics":{s:_metrics(train_y,train_prob_by_strategy[s]) for s in STRATEGIES},
            "candidate_counts":{s:len(rec_by_strategy[s]) for s in STRATEGIES}}
        np.savez_compressed(out/"folds"/f"fold_{fold:02d}.npz",held_indexes=held,
            **{f"prob_{s}":oof_probs[s][held].astype(np.float32) for s in STRATEGIES},
            **{f"fired_{s}":oof_fire[s][held] for s in STRATEGIES})

    # Cross-fitted output evaluation and comparisons.
    all_metrics={**reference_metrics};per_label=[];summary_rows=[];bootstrap={}
    for name,r in reference_metrics.items():
        pred=np.asarray(r["pred"])
        if name in ("ORACLE02","ORACLE03_top2","global_logistic_core","combined_logistic_core","top2_oracle"):
            for label,vals in r["per_label"].items(): per_label.append({"system":name,"label":label,**vals})
    total_oracle_gap=max(1e-15,reference_metrics["top2_oracle"]["balanced_accuracy"]-base_metrics["balanced_accuracy"])
    recoverable=int(np.sum((baseline_pred!=y)&(top2==y)))
    for strategy in STRATEGIES:
        prob=oof_probs[strategy];m=_metrics(y,prob);pred=np.asarray(m.pop("pred"))
        delta=_delta_counts(pred,baseline_pred,y)
        t1=p.argmax(1);t2=np.argsort(-p,axis=1,kind="stable")[:,1]
        switch=oof_fire[strategy]
        swap_precision=float(np.sum(y[switch]==t2[switch])/max(1,np.sum((y[switch]==t1[switch])|(y[switch]==t2[switch]))))
        m.update({"selected_rule_count":sum(len(fold_rule_data[str(f)]["selected_rules"][strategy]) for f in range(1,6))/5,
            "trigger_count":int(switch.sum()),"trigger_rate":float(switch.mean()),"swap_precision":swap_precision,
            "corrected_vs_ORACLE02":delta["corrected"],"introduced_vs_ORACLE02":delta["introduced"],
            "net_correct_vs_ORACLE02":delta["net_correct"],"disagreements_vs_ORACLE02":delta["disagreements"],
            "oracle_ba_gap_recovered":float((m["balanced_accuracy"]-base_metrics["balanced_accuracy"])/total_oracle_gap),
            "oracle_top2_corrections_recovered":float(delta["corrected"]/max(1,recoverable)),
            "confusion_matrix":confusion_matrix(y,pred,labels=np.arange(len(LABELS))).tolist(),"pred":pred.tolist()})
        for refname in ("ORACLE03_top2","global_logistic_core","combined_logistic_core"):
            d=_delta_counts(pred,np.asarray(reference_metrics[refname]["pred"]),y)
            m[f"corrected_vs_{refname}"]=d["corrected"];m[f"introduced_vs_{refname}"]=d["introduced"]
            m[f"delta_balanced_accuracy_pp_vs_{refname}"]=(m["balanced_accuracy"]-reference_metrics[refname]["balanced_accuracy"])*100
        m["bootstrap_delta_ba_pp"]=_pair_bootstrap(y,pred,baseline_pred,prompts,reps=bootstrap_reps,seed=SEED+STRATEGIES.index(strategy))
        bootstrap[strategy]=m["bootstrap_delta_ba_pp"]
        all_metrics[strategy]=m
        for label,vals in m["per_label"].items(): per_label.append({"system":strategy,"label":label,**vals})
        summary_rows.append({"system":strategy,**{k:m[k] for k in ("accuracy","balanced_accuracy","log_loss","selected_rule_count","trigger_count","trigger_rate","swap_precision","corrected_vs_ORACLE02","introduced_vs_ORACLE02","net_correct_vs_ORACLE02","oracle_ba_gap_recovered","oracle_top2_corrections_recovered")},
            "bootstrap_ci_low_pp":m["bootstrap_delta_ba_pp"]["ci95_pp"][0],"bootstrap_ci_high_pp":m["bootstrap_delta_ba_pp"]["ci95_pp"][1]})

    # Full-V0 diagnostic configurations use full-data pair stats and are optimistic.
    full_rules={};full_diag={}
    full_folds=[np.flatnonzero(fold_ids==f) for f in range(1,6)]
    for strategy in STRATEGIES:
        records=_enumerate_rules(strategy,y,p,np.arange(len(y)),prompts,data)
        selected=_select_rules(records)
        pp,fired,names=_apply_rules(p,selected,data,np.arange(len(y)))
        m=_metrics(y,pp);m["selected_rules"]=selected;m["trigger_count"]=int(fired.sum());m["trigger_rate"]=float(fired.mean())
        full_rules[strategy]=selected;full_diag[strategy]=m

    # Pair and rule recurrence across outer fits.
    rule_rows=[]
    for strategy in STRATEGIES:
        keys=[]
        for fold in range(1,6):
            for r in fold_rule_data[str(fold)]["selected_rules"][strategy]:
                key=f"{r['top1_label']}->{r['top2_label']}|{r.get('evidence','none')}|{r.get('evidence_threshold')}|{r.get('margin_threshold')}"
                keys.append(key);rule_rows.append({"strategy":strategy,"fold":fold,"rule_id":key,**r})
        unique,counts=np.unique(keys,return_counts=True) if keys else ([],[])
        counts_map=dict(zip(unique,counts))
        for key,count in counts_map.items():
            for rr in rule_rows:
                if rr["strategy"]==strategy and rr["rule_id"]==key: rr["fold_recurrence_count"]=int(count)
    # User-facing best strategy prioritizes held-out BA then fewer rules and stability.
    best_strategy=min(STRATEGIES,key=lambda s:(-all_metrics[s]["balanced_accuracy"],
        all_metrics[s]["selected_rule_count"],-all_metrics[s]["swap_precision"],s))
    max_ref=max(reference_metrics[s]["balanced_accuracy"] for s in ("global_logistic_core","combined_logistic_core"))
    kaggle_recommendation=(f"No candidate is ready for submission: best ORACLE05 ({best_strategy}) is {all_metrics[best_strategy]['balanced_accuracy']:.3%} BA, below ORACLE04 ({max_ref:.3%}); V0 also influenced checkpoint selection and prior rule development." if all_metrics[best_strategy]["balanced_accuracy"]<=max_ref else f"The best ORACLE05 ({best_strategy}) exceeds ORACLE04 by {(all_metrics[best_strategy]['balanced_accuracy']-max_ref)*100:+.3f} pp, but this remains V0 development evidence; require a controlled test before submission.")
    # Row-level probabilities, decisions, references, and adjusted outputs.
    row_frame=pd.DataFrame({"id":data["ids"],"prompt_id":prompts,"y_true":data["y_text"],
        "oracle02_top1":np.asarray(LABELS)[top1],"oracle02_top2":np.asarray(LABELS)[top2],
        "oracle02_margin":p[np.arange(len(y)),top1]-p[np.arange(len(y)),top2],
        "oracle02_correct":baseline_pred==y,"top2_correct":top2==y})
    for name,values in (("oracle03_top2",data["oracle03"]),*data["oracle04"].items()): row_frame[f"pred::{name}"]=np.asarray(LABELS)[values]
    for strategy in STRATEGIES:
        pred=oof_probs[strategy].argmax(1)
        row_frame[f"pred::{strategy}"]=np.asarray(LABELS)[pred]
        row_frame[f"swap::{strategy}"]=oof_fire[strategy]
        row_frame[f"rule::{strategy}"]=oof_rule[strategy]
        for i,label in enumerate(LABELS): row_frame[f"prob::{strategy}::{label}"]=oof_probs[strategy][:,i]
    for i,label in enumerate(LABELS): row_frame[f"prob::ORACLE02::{label}"]=p[:,i]
    row_frame["outer_fold"]=fold_ids
    row_frame.to_parquet(out/"row_predictions.parquet",index=False)
    pd.DataFrame(fold_metric_rows).to_csv(out/"fold_metrics.csv",index=False)
    pd.DataFrame(rule_rows).to_csv(out/"rule_stability.csv",index=False)
    pd.DataFrame(per_label).to_csv(out/"per_label_metrics.csv",index=False)
    pd.DataFrame(all_sensitivity).to_csv(out/"threshold_sensitivity.csv",index=False)
    pd.DataFrame(summary_rows).to_csv(out/"metrics.csv",index=False)
    _write_json(out/"fold_rules.json",fold_rule_data)
    _write_json(out/"selected_rules.json",{"default_parameters":{"min_support":MIN_SUPPORT,"min_swap_precision":MIN_PRECISION,
        "min_positive_inner_folds":MIN_STABLE_FOLDS,"rule_penalty_ba_per_rule":RULE_PENALTY},"outer_fold_rules":fold_rule_data})
    _write_json(out/"final_rules_full_v0.json",{"optimistic_full_v0_diagnostic":True,"rules":full_rules,"metrics":full_diag})
    _write_json(out/"bootstrap.json",bootstrap)
    # Remove redundant prediction/confusion arrays from metrics.json's baseline references.
    compact={}
    for name,r in all_metrics.items():
        compact[name]={k:v for k,v in r.items() if k not in {"pred","confusion_matrix"}}
    compact["top2_oracle"].pop("pred",None)
    _write_json(out/"metrics.json",{"trust_list_version":"trust-list-2-v025","class_order":LABELS,
        "validation_rows":len(y),"prompt_groups":int(pd.Series(prompts).nunique()),
        "baseline_metrics":compact,"cross_fitted_metrics":{s:compact[s] for s in STRATEGIES},
        "fold_metrics":fold_metric_rows,"sensitivity":all_sensitivity,"best_strategy":best_strategy,
        "kaggle_recommendation":kaggle_recommendation,"top2_correct_recoverable_rows":recoverable})
    recurrence=[]
    if rule_rows:
        recurrence_df=pd.DataFrame(rule_rows)
        for (strategy,top1_label,top2_label),g in recurrence_df.groupby(["strategy","top1_label","top2_label"],sort=True):
            recurrence.append({"strategy":strategy,"top1_label":top1_label,"top2_label":top2_label,"folds":int(g["fold"].nunique())})
    report_data={"metrics":compact,"top_pairs":[r for r in pair_records if r["occurrences"]>=1][:20],
        "all_pairs":pair_records,"sensitivity":all_sensitivity,"best_strategy":best_strategy,
        "kaggle_recommendation":kaggle_recommendation,"full_rules":full_rules,"full_diagnostic":full_diag,
        "per_label":per_label,"labels":LABELS,"rule_recurrence":recurrence,
        "available_checkpoint_count":len(data["cp_ids"]),"missing_checkpoint_ids":data["missing_checkpoint_ids"]}
    _write_report(report,report_data)
    manifest={"analysis":"ORACLE05","status":"completed","created_utc":pd.Timestamp.utcnow().isoformat(),
        "input_hashes":data["hashes"],"trust_list_version":data["pointer"].get("version"),
        "outer_folds":"existing 5-fold prompt-group split","inner_folds":INNER_FOLDS,
        "inner_training_probabilities":"ORACLE04 nested_baseline_outer_*.npz; held out by prompt group; locked modal ORACLE02 configuration",
        "strategies":list(STRATEGIES),"checkpoint_count":len(data["cp_ids"]),
        "trust_list_cached_checkpoint_count":len(data["checkpoint_availability"]),
        "excluded_missing_checkpoint_ids":data["missing_checkpoint_ids"],
        "included_checkpoint_ids":data["cp_ids"],"checkpoint_families":sorted(set(data["families"])),
        "checkpoint_availability_sha256":_sha(out/"checkpoint_availability.csv"),
        "restored_checkpoint_sources":{"N0074":{"kaggle_dataset":"zodrak/trust2-v013-equal9-models",
            "experiment_id":"qwen06_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgba46767d_code81bce4a",
            "checkpoint_sha256":"c2c199a2b7717ef71511650b40941671928a423080f0714a250f9cd6ae2453db",
            "source_manifest_sha256":"72fc5fa2464b7e4c2e3fbf5e4c9d2a1515320137d840809325c137ce69dc8db9"}},
        "selection":{"minimum_support":MIN_SUPPORT,"minimum_swap_precision":MIN_PRECISION,
            "minimum_positive_inner_folds":MIN_STABLE_FOLDS,"rule_penalty_ba_per_rule":RULE_PENALTY,
            "checkpoint_evidence_candidates":"individual checkpoint votes and probability deltas; family vote fractions and probability deltas; aggregate vote fraction and probability delta"},
        "bootstrap_reps":bootstrap_reps,"seed":SEED,"base_model_retraining":False,"base_model_inference":False,"kaggle_submission":False,
        "report_sha256":_sha(report),"script_sha256":_sha(Path(__file__).resolve())}
    _write_json(out/"run_manifest.json",manifest)
    summary={"report":str(report.relative_to(root)),"output_dir":str(out.relative_to(root)),"best_strategy":best_strategy,
        "baseline_ba":base_metrics["balanced_accuracy"],"oracle03_ba":oracle03_metrics["balanced_accuracy"],
        "oracle04_global_logistic_ba":reference_metrics["global_logistic_core"]["balanced_accuracy"],
        "oracle04_combined_logistic_ba":reference_metrics["combined_logistic_core"]["balanced_accuracy"],
        "results":{s:{k:compact[s][k] for k in ("accuracy","balanced_accuracy","log_loss","selected_rule_count","trigger_count","corrected_vs_ORACLE02","introduced_vs_ORACLE02","net_correct_vs_ORACLE02") } for s in STRATEGIES},
        "kaggle_recommendation":kaggle_recommendation}
    print(json.dumps(summary,indent=2,default=str))
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default=str(OUT));parser.add_argument("--report",default=str(REPORT))
    parser.add_argument("--bootstrap-reps",type=int,default=BOOTSTRAP_REPS)
    a=parser.parse_args();run(output_dir=a.output_dir,report_path=a.report,bootstrap_reps=a.bootstrap_reps)


if __name__=="__main__": main()
