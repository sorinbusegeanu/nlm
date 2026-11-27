#!/usr/bin/env python3
"""Rebuild E39 validation predictions from an already trained CatBoost model."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from src.feature_factory import stylometry
from scripts.run_e38_e40_feature_experiments import LABELS, metrics

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'data/processed'; out=ROOT/'artifacts/analysis/e39'; model_dir=out/'model'
v0=pd.read_parquet(data/'dataset_v0.parquet',columns=['id','text','model'])
vx=pd.read_parquet(data/'dataset_v0_ext.parquet',columns=['id','text','model'])
recipe={'baseline':['length','case','punctuation','function_words'],'groups':[],
        'features':['tokenizer_fingerprints'],
        'tokenizer_repositories':{'qwen':'Qwen/Qwen3-Embedding-0.6B','gpt2':'gpt2'},
        'tokenizer_paths':{'qwen':str((model_dir/'tokenizers/qwen').resolve()),
                           'gpt2':str((model_dir/'tokenizers/gpt2').resolve())}}
t=time.time()
x0,names=stylometry(v0.text.fillna('').astype(str).tolist(),recipe,return_feature_names=True)
xx=stylometry(vx.text.fillna('').astype(str).tolist(),recipe)
cb=CatBoostClassifier();cb.load_model(str(model_dir/'model.cbm'))
def predict(x):
    raw=np.asarray(cb.predict(x,prediction_type='RawFormulaVal'))
    if raw.ndim==1:raw=np.c_[-raw,raw]
    from scipy.special import softmax
    aligned=np.zeros((len(x),len(LABELS)))
    for i,label in enumerate(cb.classes_):aligned[:,LABELS.index(str(label))]=raw[:,i]
    return softmax(aligned,axis=1)
p0,px=predict(x0),predict(xx)
v0y=v0.model.astype(str).tolist();vxy=vx.model.astype(str).tolist()
res={'experiment':'E39','model':'CatBoost tokenizer fingerprints','training_dataset':'v2.14',
 'validation':{'v0':metrics(v0y,p0),'v0_ext':metrics(vxy,px)},
 'mean_balanced_accuracy':(metrics(v0y,p0)['balanced_accuracy']+metrics(vxy,px)['balanced_accuracy'])/2,
 'elapsed_seconds':time.time()-t,'feature_count':int(x0.shape[1]),'feature_names':names,
 'feature_recipe':{**recipe,'tokenizer_paths':{'qwen':'tokenizers/qwen','gpt2':'tokenizers/gpt2'}},
 'prediction_row_ids':{'v0':v0.id.astype(str).tolist(),'v0_ext':vx.id.astype(str).tolist()}}
out.mkdir(parents=True,exist_ok=True)
(out/'enhanced_metrics.json').write_text(json.dumps(res,indent=2)+'\n')
np.savez_compressed(out/'enhanced_predictions.npz',v0=p0,v0_ext=px)
meta=json.loads((model_dir/'stylometry.json').read_text());meta.update({'feature_names':names,'feature_count':len(names),
    'feature_recipe':res['feature_recipe'],'feature_state_file':None})
(model_dir/'stylometry.json').write_text(json.dumps(meta,indent=2)+'\n')
hashes={}
for p in sorted((model_dir/'tokenizers').rglob('*')):
 if p.is_file():hashes[str(p.relative_to(model_dir))]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'tokenizer_asset_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'e39_mean_ba':res['mean_balanced_accuracy'],'v0_ba':res['validation']['v0']['balanced_accuracy'],
 'v0_ext_ba':res['validation']['v0_ext']['balanced_accuracy'],'features':len(names),'rows':[len(v0),len(vx)]},indent=2))
