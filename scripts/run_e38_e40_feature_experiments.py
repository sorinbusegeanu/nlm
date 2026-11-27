#!/usr/bin/env python3
"""Run the E39 -> E40 -> E38 V2.14 CPU feature comparison, resumably by stage."""
from __future__ import annotations
import argparse, json, os, shutil, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from src.feature_factory import (EmpiricalSubwordStats, stylometry, FoldFeatures,
                                 load_or_fit, tokenizer_fingerprint)
from src.model_factory import train_catboost_stylometry, train_svc

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed'; OUT=ROOT/'artifacts/analysis'
LABELS=sorted(pd.read_parquet(DATA/'dataset_v2.14.parquet',columns=['model']).model.astype(str).unique())

def read_data():
    train=pd.read_parquet(DATA/'dataset_v2.14.parquet',columns=['id','text','model'])
    v0=pd.read_parquet(DATA/'dataset_v0.parquet',columns=['id','text','model'])
    vx=pd.read_parquet(DATA/'dataset_v0_ext.parquet',columns=['id','text','model'])
    for name,df in [('train',train),('v0',v0),('v0_ext',vx)]:
        df['text']=df.text.fillna('').astype(str); df['model']=df.model.astype(str)
        if set(df.model)-set(LABELS): raise ValueError(f'{name} contains unknown labels')
    return train,v0,vx

def metrics(y,p):
    pred=np.asarray(LABELS)[np.asarray(p).argmax(1)]
    precision,recall,_,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),
      'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'per_label':{lab:{'accuracy':float(recall[i]),'recall':float(recall[i]),
                        'precision':float(precision[i]),'support':int(support[i])} for i,lab in enumerate(LABELS)}}

def predict_cat(model, x):
    raw=np.asarray(model.predict(x,prediction_type='RawFormulaVal'))
    if raw.ndim==1: raw=np.c_[-raw,raw]
    from scipy.special import softmax
    cls=list(model.classes_); aligned=np.zeros((len(x),len(LABELS)))
    for j,c in enumerate(cls): aligned[:,LABELS.index(str(c))]=raw[:,j]
    return softmax(aligned,axis=1)

def save_result(exp, name, y0, p0, yx, px, elapsed, extra=None):
    result={'experiment':exp,'model':name,'classes':LABELS,'training_dataset':'v2.14',
      'validation':{'v0':metrics(y0,p0),'v0_ext':metrics(yx,px)},'elapsed_seconds':elapsed,
      'mean_balanced_accuracy':(metrics(y0,p0)['balanced_accuracy']+metrics(yx,px)['balanced_accuracy'])/2,
      'per_label_accuracy':{'v0':metrics(y0,p0)['per_label'],'v0_ext':metrics(yx,px)['per_label']}}
    if extra: result.update(extra)
    dest=OUT/exp; dest.mkdir(parents=True,exist_ok=True)
    (dest/f'{name}_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(dest/f'{name}_predictions.npz',v0=p0,v0_ext=px)
    return result

def run_catboost(train,v0,vx,*,exp,recipe,feature_state=None,tokenizer_paths=None,artifact=None,train_fit=False):
    artifact=Path(artifact or OUT/exp/'model'); artifact.mkdir(parents=True,exist_ok=True)
    all_text=train.text.tolist()+v0.text.tolist()+vx.text.tolist()
    rec={'baseline':['length','case','punctuation','function_words'],'groups':[],**recipe}
    if tokenizer_paths: rec['tokenizer_paths']=tokenizer_paths
    t=time.time(); print(f'{exp}: extracting {len(all_text)} rows',flush=True)
    x,feature_names=stylometry(all_text,rec,feature_state=feature_state,return_feature_names=True)
    a,b,c=np.split(x,[len(train),len(train)+len(v0)])
    del x
    if train_fit:
        cfg=yaml.safe_load((ROOT/'configs/models/catboost_stylometry.yaml').read_text())
        result=train_catboost_stylometry(train.text.tolist(),train.model.tolist(),v0.text.tolist(),LABELS,
          output_dir=artifact,iterations=cfg['iterations'],depth=cfg['depth'],learning_rate=cfg['learning_rate'],
          l2_leaf_reg=cfg['l2_leaf_reg'],thread_count=4,seed=2026,
          validation_labels=v0.model.tolist(),early_stopping_rounds=cfg['early_stopping_rounds'],
          feature_recipe=rec,feature_matrices=(a,b, feature_names),
          feature_state=feature_state,checkpoint_selection_metric='accuracy')
        model=result['model']; model.save_model(str(artifact/'model.cbm'))
    else:
        from catboost import CatBoostClassifier
        model=CatBoostClassifier();model.load_model(str(artifact/'model.cbm'))
    p0=predict_cat(model,b); px=predict_cat(model,c)
    if tokenizer_paths:
        # Keep inference self-contained: resolved model assets live beside model.cbm.
        asset=artifact/'tokenizers'; asset.mkdir(exist_ok=True)
        for key,src in tokenizer_paths.items():
            dest=asset/key; shutil.copytree(src,dest,dirs_exist_ok=True)
        rec['tokenizer_paths']={k:f'tokenizers/{k}' for k in tokenizer_paths}
        meta=artifact/'stylometry.json'
        if meta.exists():
            m=json.loads(meta.read_text());m['feature_recipe']=rec;meta.write_text(json.dumps(m,indent=2)+'\n')
    elapsed=time.time()-t
    feature_count=int(a.shape[1])
    del a,b,c
    return save_result(exp,'enhanced' if exp!='e39' else 'E39',v0.model.tolist(),p0,vx.model.tolist(),px,elapsed,
      {'feature_recipe':rec,'feature_state':('feature_state.json' if feature_state is not None else None),
       'feature_count':feature_count,'feature_names':feature_names})

def train_cat_baseline(train,v0,vx):
    out=OUT/'e39'/'baseline_catboost';
    res=run_catboost(train,v0,vx,exp='baseline_catboost',recipe={},artifact=out,
                     train_fit=not (out/'model.cbm').is_file())
    # Canonical shared baseline pointer for E39/E40.
    (OUT/'e39').mkdir(parents=True,exist_ok=True)
    return res,out

def download_tokenizers():
    from huggingface_hub import snapshot_download
    dest=OUT/'e39'/'tokenizer_sources'; dest.mkdir(parents=True,exist_ok=True)
    sources={'qwen':'Qwen/Qwen3-Embedding-0.6B','gpt2':'gpt2'}
    for key,repo in sources.items():
        path=dest/key
        if not (path/'tokenizer_config.json').exists():
            path.mkdir(parents=True,exist_ok=True)
            snap=snapshot_download(repo_id=repo,allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                'special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'],local_dir=str(path),
                local_dir_use_symlinks=False)
        print(f'tokenizer {key}: {path}',flush=True)
    return {k:str(dest/k) for k in sources}

def run_svc(train,v0,vx,enhanced):
    exp='e38'; root=OUT/exp; root.mkdir(parents=True,exist_ok=True)
    if enhanced:
        cfg=yaml.safe_load((ROOT/'configs/features/e38_extended_svc.yaml').read_text())
        cfg['view_options']={k:{**v,**({'ngram_range':tuple(v['ngram_range'])} if 'ngram_range' in v else {}),
                                **({'dtype':np.float32} if str(v.get('dtype','')).lower()=='float32' else {})}
                             for k,v in cfg['view_options'].items()}
    else: cfg={'views':['char','word']}
    model_cfg=yaml.safe_load((ROOT/'configs/models/svc.yaml').read_text())
    cache=root/('features_enhanced' if enhanced else 'features_baseline')
    t=time.time(); print(f'E38 {"enhanced" if enhanced else "baseline"}: fitting; four CPU threads',flush=True)
    res=train_svc(train.text.tolist(),train.model.tolist(),v0.text.tolist(),LABELS,c=model_cfg['C'],seed=2026,
      feature_config=cfg,cache_dir=cache,fold_token='v2.14-full-v0-v0ext',validation_labels=v0.model.tolist(),
      calibration=model_cfg.get('calibration'),calibration_cv=model_cfg.get('calibration_cv',3))
    p0=res['probabilities']; pext=res['features'].transform(vx.text.tolist())
    from src.inference import _readout_probabilities
    px=_readout_probabilities(res['model'],pext,LABELS)
    name='enhanced' if enhanced else 'baseline'
    checkpoint=root/name; checkpoint.mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump({'model':res['model'],'features':res['features'],'classes':LABELS,'config':cfg},checkpoint/'svc_bundle.joblib',compress=3)
    serial_cfg=json.loads(json.dumps(cfg,default=lambda value: value.__name__ if isinstance(value,type) else str(value)))
    report=save_result(exp,name,v0.model.tolist(),p0,vx.model.tolist(),px,time.time()-t,{'feature_config':serial_cfg,'fit_seconds':res['fit_seconds']})
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['e39','e40','e38','e38-enhanced','all'],default='all');args=ap.parse_args()
    train,v0,vx=read_data();print(f'rows train={len(train)} V0={len(v0)} V0-EXT={len(vx)} labels={len(LABELS)}',flush=True)
    # Stage E39 includes the one shared baseline fit. E40 consumes that unchanged model.
    if args.stage in {'e39','all'}:
        tok=download_tokenizers()
        baseline,_=train_cat_baseline(train,v0,vx)
        feature_cfg=yaml.safe_load((ROOT/'configs/features/catboost_stylometry_e39.yaml').read_text())
        recipe={**feature_cfg['stylometry_recipe'],'tokenizer_paths':tok}
        run_catboost(train,v0,vx,exp='e39',recipe=recipe,tokenizer_paths=tok,train_fit=True)
        # Copy the shared baseline result for convenient paired comparison.
        (OUT/'e39'/'baseline_metrics.json').write_text(json.dumps(baseline,indent=2)+'\n')
    if args.stage in {'e40','all'}:
        base=OUT/'e39'/'baseline_catboost'
        if not (base/'model.cbm').exists(): raise FileNotFoundError('run E39 first to create the shared CatBoost baseline')
        # Fit empirical statistics solely on V2.14 training text.
        feature_cfg=yaml.safe_load((ROOT/'configs/features/catboost_stylometry_e40.yaml').read_text())
        recipe=feature_cfg['stylometry_recipe']
        stat=EmpiricalSubwordStats(alpha=recipe['unigram_smoothing_alpha'],
              interpolation=recipe['bigram_interpolation_lambda']).fit(train.text.tolist())
        (OUT/'e40').mkdir(parents=True,exist_ok=True)
        state=stat.state;(OUT/'e40'/'empirical_subword_stats.json').write_text(json.dumps(state,separators=(',',':'))+'\n')
        enhanced=run_catboost(train,v0,vx,exp='e40',recipe=recipe,feature_state=state,train_fit=True)
        from src.feature_factory import stylometry
        validation_text=v0.text.tolist()+vx.text.tolist()
        x=stylometry(validation_text,{'baseline':['length','case','punctuation','function_words'],'groups':[]})
        bv=len(v0)
        from catboost import CatBoostClassifier
        m=CatBoostClassifier();m.load_model(str(base/'model.cbm'))
        p0=predict_cat(m,x[:bv]);px=predict_cat(m,x[bv:])
        save_result('e40','baseline',v0.model.tolist(),p0,vx.model.tolist(),px,0,{'reused_checkpoint':str(base/'model.cbm')})
    if args.stage in {'e38','all'}:
        run_svc(train,v0,vx,False);run_svc(train,v0,vx,True)
    elif args.stage=='e38-enhanced':
        run_svc(train,v0,vx,True)

if __name__=='__main__': main()
