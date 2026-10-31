"""Command line pipeline for dataset, grouped SVC CV, and ensemble reports."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .data_factory import prepare
from .evaluation import evaluate
from .model_factory import train_svc
from .registry import Registry
from .ensemble_factory import cross_fitted_weights


def _revision():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return "unknown"


def _config(path):
    with open(path,encoding="utf-8") as f:return yaml.safe_load(f)


def prepare_data(config):
    cfg=_config(config)
    return prepare(cfg["train_path"],cfg["output_dir"],cfg.get("test_path"),seed=cfg.get("seed",2026),holdout_fraction=cfg.get("holdout_fraction",.2),n_splits=cfg.get("n_splits",5))


def train_models(dataset="v0", model="svc", data_dir="data/processed", artifact_dir="artifacts", registry_path="experiments/registry.sqlite", *, c=.5, calibration_cv=3, seed=2026):
    if model!="svc":
        raise RuntimeError("Neural CLI fold orchestration is not enabled yet; use the train_encoder_fold / train_qwen_lora_fold adapters with the frozen fold rows.")
    data_dir=Path(data_dir); artifact_dir=Path(artifact_dir)
    rows=pd.read_parquet(data_dir/f"dataset_{dataset}.parquet"); assignments=pd.read_parquet(data_dir/f"folds_{dataset}.parquet")
    rows=rows.merge(assignments[["id","split","fold"]],on="id",validate="one_to_one")
    classes=sorted(rows.model.unique().tolist());dev=rows[rows.split.eq("development")].reset_index(drop=True)
    oof=np.full((len(dev),len(classes)),np.nan);fold_ids=dev.fold.to_numpy();durations=[]
    for fold in sorted(dev.fold.unique()):
        val=dev.fold.eq(fold).to_numpy(); trn=~val
        start=time.perf_counter()
        result=train_svc(dev.loc[trn,"text"],dev.loc[trn,"model"],dev.loc[val,"text"],classes,c=c,calibration_cv=calibration_cv,seed=seed+int(fold))
        oof[val]=result["probabilities"]; durations.append(time.perf_counter()-start)
        out=artifact_dir/"checkpoints"/f"svc_{dataset}_fold{fold}.joblib";out.parent.mkdir(parents=True,exist_ok=True)
        import joblib;joblib.dump({"model":result["model"],"vectorizers":result["vectorizers"],"classes":classes},out)
    if not np.isfinite(oof).all():raise RuntimeError("OOF predictions are incomplete")
    metrics=evaluate(dev.model,oof,classes,oof)
    pred=dev[["id","prompt_id","model","fold"]].copy()
    for i,label in enumerate(classes):pred[f"score::{label}"]=np.log(np.clip(oof[:,i],1e-12,1));pred[f"prob::{label}"]=oof[:,i]
    pred_path=artifact_dir/"predictions"/f"svc_{dataset}_oof.parquet";pred_path.parent.mkdir(parents=True,exist_ok=True);pred.to_parquet(pred_path,index=False)
    report={"experiment_id":f"svc_{dataset}_seed{seed}","model":"svc","split":"development_oof","dataset":dataset,"classes":classes,"metrics":metrics,"fold_seconds":durations,"fit_seconds":sum(durations),"prediction_path":str(pred_path),"config":{"C":c,"calibration":"sigmoid train-only CV","calibration_cv":calibration_cv,"seed":seed,"vectorizer":"char 3-5 + word 1-2"}}
    Registry(registry_path).add(report["experiment_id"],"svc","development_oof",report["config"],metrics,dataset_hash=json.loads((data_dir/f"dataset_profile.json").read_text())["dataset_sha256"],code_revision=_revision())
    (artifact_dir/"predictions"/f"svc_{dataset}_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def evaluate_holdout(model="svc",dataset="v0",data_dir="data/processed",artifact_dir="artifacts",seed=2026):
    """Fit on development and evaluate the untouched prompt-group holdout once."""
    if model!="svc":raise RuntimeError("holdout evaluation currently supports svc")
    data_dir=Path(data_dir); rows=pd.read_parquet(data_dir/f"dataset_{dataset}.parquet").merge(pd.read_parquet(data_dir/f"folds_{dataset}.parquet")[["id","split","fold"]],on="id")
    train=rows[rows.split.eq("development")];val=rows[rows.split.eq("holdout")];classes=sorted(rows.model.unique())
    result=train_svc(train.text,train.model,val.text,classes,seed=seed)
    metrics=evaluate(val.model,result["scores"],classes,result["probabilities"])
    report={"experiment_id":f"svc_{dataset}_holdout_seed{seed}","model":"svc","split":"holdout","metrics":metrics,"rows":len(val),"note":"One-time untouched holdout evaluation; do not use for model or hyperparameter selection."}
    out=Path(artifact_dir)/"predictions";out.mkdir(parents=True,exist_ok=True);(out/f"svc_{dataset}_holdout_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    Registry().add(report["experiment_id"],"svc","holdout",{"seed":seed},metrics,dataset_hash=json.loads((data_dir/"dataset_profile.json").read_text())["dataset_sha256"],code_revision=_revision())
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m src.pipeline");sub=parser.add_subparsers(dest="command",required=True)
    d=sub.add_parser("data").add_subparsers(dest="action",required=True);p=d.add_parser("prepare");p.add_argument("--config",default="configs/data/official_v0.yaml")
    f=sub.add_parser("features").add_subparsers(dest="action",required=True);f.add_parser("build").add_argument("--dataset",default="v0")
    m=sub.add_parser("models").add_subparsers(dest="action",required=True);t=m.add_parser("train");t.add_argument("--model",default="svc");t.add_argument("--dataset",default="v0");t.add_argument("--data-dir",default="data/processed");t.add_argument("--calibration-cv",type=int,default=3)
    h=m.add_parser("evaluate-holdout");h.add_argument("--model",default="svc");h.add_argument("--dataset",default="v0")
    e=sub.add_parser("ensemble").add_subparsers(dest="action",required=True);q=e.add_parser("evaluate");q.add_argument("--dataset",default="v0")
    a=parser.parse_args(argv)
    if a.command=="data": print(json.dumps(prepare_data(a.config),indent=2))
    elif a.command=="features": print("Feature matrices are fitted inside each training fold to prevent leakage.")
    elif a.command=="models" and a.action=="train": print(json.dumps(train_models(dataset=a.dataset,model=a.model,data_dir=a.data_dir,calibration_cv=a.calibration_cv),indent=2))
    elif a.command=="models": print(json.dumps(evaluate_holdout(model=a.model,dataset=a.dataset),indent=2))
    elif a.command=="ensemble": raise RuntimeError("Ensemble evaluation requires at least two OOF prediction artifacts.")

if __name__=="__main__":main()
