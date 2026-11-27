#!/usr/bin/env python3
"""Full-V0 comparison and inference benchmark for E42 Qwen3.5-4B QLoRA."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.inference import predict_candidate
OUT=ROOT/'artifacts/analysis/e42_qwen35_4b_qlora'; DOC=ROOT/'docs/e42_qwen35_4b_qlora.md'
EXPERIMENT='E42-QWEN35-4B-QLORA'
def read(p): return json.loads(Path(p).read_text())
def find_manifest(variant):
    matches=[]
    for p in (ROOT/'artifacts/predictions').glob('*_manifest.json'):
        try:m=read(p)
        except Exception:continue
        cfg=m.get('config',{})
        if cfg.get('e42_experiment_id')!=EXPERIMENT: continue
        got=cfg.get('layer_pruning',{}).get('variant','BASE')
        if got==variant and m.get('selected_checkpoint_path'): matches.append((p.stat().st_mtime,p,m))
    if not matches: raise FileNotFoundError(f'completed manifest not found for {variant}')
    _,path,manifest=max(matches,key=lambda x:x[0]); return path,manifest
def evaluate(name,manifest,frame,classes):
    checkpoint=Path(manifest.get('selected_checkpoint_path') or manifest['checkpoint_path'])
    if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
    cpmanifest=read(checkpoint/'checkpoint_manifest.json')
    run_cfg=manifest.get('config',{})
    spec={'candidate_id':name,'model':'qwen35_4b','checkpoint':str(checkpoint.relative_to(ROOT)),
          'config':run_cfg,'model_revision':cpmanifest.get('revision') or manifest.get('model_revision') or run_cfg.get('revision'),
          'quantization':cpmanifest.get('quantization','qlora4'),'attention_backend':cpmanifest.get('attention_backend')}
    t=time.perf_counter(); probs,batch=predict_candidate(spec,frame.text.astype(str).tolist(),classes,ROOT,return_batch_metadata=True); wall=time.perf_counter()-t
    y=frame.model.astype(str).to_numpy(); pred=np.asarray(classes)[probs.argmax(1)]
    OUT.mkdir(parents=True,exist_ok=True)
    out=pd.DataFrame({'id':frame.id.astype(str),'y_true':y,'prediction':pred})
    for i,c in enumerate(classes):out[f'prob::{c}']=probs[:,i]
    out.to_parquet(OUT/f'{name.lower().replace("-","_")}_v0_predictions.parquet',index=False)
    return {'name':name,'experiment_id':manifest.get('experiment_id'),'checkpoint':str(checkpoint),
      'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'correct':int((y==pred).sum()),'rows':len(frame),'inference_wall_seconds':wall,'inference':batch,
      'runtime_breakdown':manifest.get('runtime_breakdown',{}),'effective_training_batch_size':manifest.get('effective_batch_size'),
      'training_peak_gpu_memory_bytes':manifest.get('peak_gpu_memory_bytes'),
      'layer_pruning':cpmanifest.get('layer_pruning'),'inference_probe':cpmanifest.get('inference_batch_size_probe')}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',required=True);ap.parse_args()
    v0=pd.read_parquet(ROOT/'data/processed/dataset_v0.parquet')
    if len(v0)!=2699:raise ValueError(f'full V0 expected 2699 rows, got {len(v0)}')
    classes=sorted(v0.model.astype(str).unique().tolist()); results=[]
    for variant,name in [('BASE','E42 baseline'),('E42-A','E42-A'),('E42-B','E42-B'),('E42-C','E42-C')]:
        _,m=find_manifest(variant);results.append(evaluate(name,m,v0,classes))
    ba=results[0]['balanced_accuracy']; sec=results[0]['inference_wall_seconds']
    for x in results:
        x['delta_vs_baseline_pp']=(x['balanced_accuracy']-ba)*100
        x['inference_speedup_vs_baseline']=sec/x['inference_wall_seconds'] if x['inference_wall_seconds'] else None
    summary={'experiment':EXPERIMENT,'base_model':'Qwen/Qwen3.5-4B-Base','revision':'1001bb4d826a52d1f399e183466143f4da7b741b','training_dataset':'v2.14','validation_dataset':'v0','validation_rows':2699,'classes':classes,'results':results,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# E42 — Qwen3.5-4B-Base QLoRA layer removal','',
      'Four independent QLoRA runs start from the pinned pretrained-only `Qwen/Qwen3.5-4B-Base` revision. The baseline retains all 32 layers; variants A/B/C each remove 11 layers and retain 21 original blocks in order. All use V2.14 training and all 2,699 V0 rows for checkpoint selection and final comparison. V0 is therefore a development benchmark, not independent validation.','',
      '## Results','',
      '| Variant | Retained layers | V0 balanced accuracy | Δ vs baseline (pp) | Training duration (s) | Safe train batch | Calibrated inference batch | Full-V0 inference (s) | Speedup vs baseline | Peak training VRAM (GB) | Peak inference VRAM (GB) |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in results:
        p=x.get('layer_pruning') or {}; count=len(p.get('retained_layer_indices',[])) or 32
        rt=x.get('runtime_breakdown') or {}; inf=x.get('inference') or {}
        fmt=lambda v:'n/a' if v is None else f'{v/1e9:.2f}'
        lines.append(f"| {x['name']} | {count} | {x['balanced_accuracy']:.4%} | {x['delta_vs_baseline_pp']:+.4f} | {rt.get('training_including_validation_seconds','n/a')} | {x.get('effective_training_batch_size') or 'n/a'} | {inf.get('effective_batch_size','n/a')} | {x['inference_wall_seconds']:.2f} | {x['inference_speedup_vs_baseline']:.2f}× | {fmt(x.get('training_peak_gpu_memory_bytes'))} | {fmt(inf.get('peak_gpu_memory_bytes'))} |")
    lines += ['', '## Retained layer indices','']
    for x in results:
        p=x.get('layer_pruning') or {}
        lines.append(f"- **{x['name']}** retains `{p.get('retained_layer_indices', list(range(32)))}` and removes `{p.get('removed_layer_indices', [])}`.")
    lines += ['', 'Batch calibration uses the longest V0 inputs, the configured 95% VRAM safety fraction and the existing OOM-safe probe. Later inference retains smaller-batch retry. Per-run probe attempts, architecture metadata, probability exports and timings are saved under `artifacts/analysis/e42_qwen35_4b_qlora/`.','']
    DOC.write_text('\n'.join(lines))
    catp=ROOT/'configs/ensemble_catalog_v2.json';catalog=read(catp)
    entry=next(x for x in catalog['experiments'] if x.get('id')==EXPERIMENT)
    entry['status']='Complete';entry['current_stage']='Four QLoRA layer-removal variants evaluated on full V0'
    entry['results']={x['name']:{'balanced_accuracy':x['balanced_accuracy'],'accuracy':x['accuracy'],'inference_seconds':x['inference_wall_seconds']} for x in results}
    entry['artifact_directory']='artifacts/analysis/e42_qwen35_4b_qlora';entry['documentation']='docs/e42_qwen35_4b_qlora.md'
    catp.write_text(json.dumps(catalog,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
