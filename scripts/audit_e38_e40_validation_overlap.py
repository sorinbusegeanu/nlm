#!/usr/bin/env python3
"""Audit V2.14 response overlap and score the non-overlapping V0-EXT subset."""
import json, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score

ROOT=Path(__file__).resolve().parents[1]; data=ROOT/'data/processed'; art=ROOT/'artifacts/analysis'
train=pd.read_parquet(data/'dataset_v2.14.parquet',columns=['id','text'])
v0=pd.read_parquet(data/'dataset_v0.parquet',columns=['id','text','model'])
vx=pd.read_parquet(data/'dataset_v0_ext.parquet',columns=['id','text','model'])
def norm(x):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(x)).casefold()).strip()
train_ids=set(train.id.astype(str)); train_text=set(train.text.fillna('').map(norm))
v0_ids=np.asarray([str(i) in train_ids for i in v0.id]); v0_txt=np.asarray([norm(t) in train_text for t in v0.text])
vx_ids=np.asarray([str(i) in train_ids for i in vx.id]); vx_txt=np.asarray([norm(t) in train_text for t in vx.text])
clean=~(vx_ids|vx_txt)
labels=sorted(pd.read_parquet(data/'dataset_v2.14.parquet',columns=['model']).model.astype(str).unique())
output={'training_rows':len(train),'v0_rows':len(v0),'v0_overlap_id':int(v0_ids.sum()),'v0_overlap_text':int(v0_txt.sum()),
 'v0_ext_rows':len(vx),'v0_ext_overlap_id':int(vx_ids.sum()),'v0_ext_overlap_text':int(vx_txt.sum()),
 'v0_ext_overlap_union':int((vx_ids|vx_txt).sum()),'v0_ext_nonoverlap_rows':int(clean.sum()),
 'v0_ext_nonoverlap_fraction':float(clean.mean()),'models':{}}
def add(name,path):
 p=art/path
 if not p.exists():return
 z=np.load(p); report={}
 for split,df in [('v0',v0),('v0_ext',vx)]:
  probs=z[split];pred=np.asarray(labels)[probs.argmax(1)];truth=df.model.astype(str).to_numpy()
  if probs.shape!=(len(df),len(labels)) or not np.isfinite(probs).all():
   raise ValueError(f'{name}/{split}: prediction shape or values do not align')
  mask=np.ones(len(df),dtype=bool) if split=='v0' else clean
  report[split]={'rows':int(mask.sum()),'accuracy':float(accuracy_score(truth[mask],pred[mask])),
     'balanced_accuracy':float(balanced_accuracy_score(truth[mask],pred[mask])),
     'labels_present':int(np.unique(truth[mask]).size)}
 output['models'][name]=report
for name,path in [
 ('E38 SVC baseline','e38/baseline_predictions.npz'),('E38 SVC enhanced','e38/enhanced_predictions.npz'),
 ('E39 CatBoost baseline','e39/baseline_predictions.npz'),('E39 CatBoost enhanced','e39/enhanced_predictions.npz'),
 ('E40 CatBoost baseline','e40/baseline_predictions.npz'),('E40 CatBoost enhanced','e40/enhanced_predictions.npz')]:add(name,path)
(art/'e38_e40'/'validation_overlap_audit.json').parent.mkdir(parents=True,exist_ok=True)
(art/'e38_e40'/'validation_overlap_audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
