#!/usr/bin/env python3
"""Post-queue full-V0 evaluation and inference benchmark for E41 layer masks."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import predict_candidate

BASELINE_ID = "qwen35_08b_v2.14_valv0_data3f873d57_splitba4ac459_seed2026_cfgb17689db_code2a8f778"
OUT = ROOT / "artifacts/analysis/e41_qwen35_layer_removal"
DOC = ROOT / "docs/e41_qwen35_layer_removal.md"


def read_json(path):
    return json.loads(Path(path).read_text())


def find_variant_manifest(variant):
    matches=[]
    for path in (ROOT / "artifacts/predictions").glob("*_manifest.json"):
        try: m=read_json(path)
        except Exception: continue
        cfg=m.get("config", {})
        if cfg.get("layer_pruning", {}).get("variant") == variant and m.get("selected_checkpoint_path"):
            matches.append((path.stat().st_mtime, path, m))
    if not matches:
        raise FileNotFoundError(f"completed manifest missing for {variant}")
    return max(matches, key=lambda x:x[0])[1:]


def eval_one(label, run_manifest, checkpoint_manifest, frame, classes):
    checkpoint_path = Path(run_manifest.get("selected_checkpoint_path") or run_manifest["checkpoint_path"])
    if not checkpoint_path.is_absolute(): checkpoint_path = ROOT / checkpoint_path
    cfg = dict(run_manifest.get("config", {}))
    spec = {"candidate_id": label, "model": "qwen35_08b",
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "config": cfg, "model_revision": checkpoint_manifest.get("revision") or cfg.get("revision"),
            "quantization": checkpoint_manifest.get("quantization", "none"),
            "attention_backend": checkpoint_manifest.get("attention_backend")}
    started=time.perf_counter()
    probs, batch = predict_candidate(spec, frame.text.astype(str).tolist(), classes, ROOT,
                                    return_batch_metadata=True)
    wall=time.perf_counter()-started
    pred=np.asarray(classes)[probs.argmax(axis=1)]
    y=frame.model.astype(str).to_numpy()
    result={"name":label,"experiment_id":run_manifest.get("experiment_id"),
            "checkpoint":str(checkpoint_path),
            "accuracy":float(accuracy_score(y,pred)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "correct":int((pred==y).sum()),"rows":len(frame),
            "inference_wall_seconds":wall,"batch_metadata":batch,
            "training_runtime_breakdown":run_manifest.get("runtime_breakdown",{}),
            "selected_validation_balanced_accuracy":run_manifest.get("selected_validation_balanced_accuracy"),
            "effective_training_batch_size":run_manifest.get("effective_batch_size") or checkpoint_manifest.get("effective_batch_size"),
            "training_peak_gpu_memory_bytes":run_manifest.get("peak_gpu_memory_bytes") or checkpoint_manifest.get("peak_gpu_memory_bytes"),
            "layer_pruning":checkpoint_manifest.get("layer_pruning")}
    OUT.mkdir(parents=True,exist_ok=True)
    pred_frame=pd.DataFrame({"id":frame.id.astype(str),"y_true":y,"prediction":pred})
    for i,c in enumerate(classes): pred_frame[f"prob::{c}"]=probs[:,i]
    pred_frame.to_parquet(OUT/f"{label.lower().replace(' ','_')}_v0_predictions.parquet",index=False)
    return result


def update_catalog(result):
    path=ROOT/"configs/ensemble_catalog_v2.json"
    catalog=read_json(path)
    entry=next((x for x in catalog["experiments"] if x.get("id")=="E41-QWEN35-LAYER-REMOVAL"),None)
    if entry is None: return
    entry["status"]="Complete"
    entry["current_stage"]="Completed three Qwen3.5 layer-removal LoRA runs and full-V0 comparison"
    entry["results"]={x["name"]:{"balanced_accuracy":x["balanced_accuracy"],"accuracy":x["accuracy"],"inference_wall_seconds":x["inference_wall_seconds"]} for x in result["models"]}
    entry["artifact_directory"]="artifacts/analysis/e41_qwen35_layer_removal"
    entry["documentation"]="docs/e41_qwen35_layer_removal.md"
    path.write_text(json.dumps(catalog,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--queue",required=True); args=parser.parse_args()
    queue_path=Path(args.queue); queue=read_json(queue_path)
    v0=pd.read_parquet(ROOT/"data/processed/dataset_v0.parquet")
    if len(v0)!=2699: raise ValueError(f"expected full V0 2699 rows, found {len(v0)}")
    classes=sorted(v0.model.astype(str).unique().tolist())
    results=[]
    baseline_manifest=read_json(ROOT/f"artifacts/predictions/{BASELINE_ID}_manifest.json")
    checkpoint=Path(baseline_manifest.get("selected_checkpoint_path") or baseline_manifest["checkpoint_path"])
    if not checkpoint.is_absolute(): checkpoint=ROOT/checkpoint
    results.append(eval_one("N0137 baseline",baseline_manifest,read_json(checkpoint/"checkpoint_manifest.json"),v0,classes))
    for variant in ("E41-A","E41-B","E41-C"):
        path, manifest=find_variant_manifest(variant)
        cp=Path(manifest.get("selected_checkpoint_path") or manifest["checkpoint_path"])
        if not cp.is_absolute(): cp=ROOT/cp
        results.append(eval_one(variant,manifest,read_json(cp/"checkpoint_manifest.json"),v0,classes))
    baseline=results[0]["balanced_accuracy"]
    baseline_inference=results[0]["inference_wall_seconds"]
    for item in results:
        item["delta_vs_n0137_pp"]=(item["balanced_accuracy"]-baseline)*100
        item["inference_speedup_vs_n0137"]=(baseline_inference/item["inference_wall_seconds"]
                                             if item["inference_wall_seconds"] else None)
    summary={"baseline_experiment_id":BASELINE_ID,"validation_dataset":"v0","validation_rows":2699,
             "classes":classes,"models":results,"created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# E41 — Qwen3.5-0.8B layer removal", "",
      "Three independently initialized LoRA runs use the pinned Qwen3.5-0.8B-Base backbone, V2.14 training rows and all 2,699 V0 rows for validation. The runs preserve N0137's recipe; only transformer layer masks differ. The dataset is V0-exposed and these scores are development results, not an independent test.","",
      "## Full-V0 comparison","",
      "| Model | Retained layers | Balanced accuracy | Δ vs N0137 (pp) | Training duration (s) | Effective train batch | Calibrated inference batch | Full-V0 inference (s) | Speedup vs N0137 | Peak train VRAM (GB) | Peak inference VRAM (GB) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in results:
        bp=x["batch_metadata"]
        pruning=x.get("layer_pruning") or {}
        train_dur=x["training_runtime_breakdown"].get("training_including_validation_seconds")
        tmem=x.get("training_peak_gpu_memory_bytes")
        imem=bp.get("peak_gpu_memory_bytes")
        fmt=lambda v: "n/a" if v is None else f"{v/1e9:.2f}"
        speedup=x.get('inference_speedup_vs_n0137')
        lines.append(f"| {x['name']} | {len(pruning.get('retained_layer_indices',[])) or '24'} | {x['balanced_accuracy']:.4%} | {x['delta_vs_n0137_pp']:+.4f} | {train_dur if train_dur is not None else 'n/a'} | {x['effective_training_batch_size'] or 'n/a'} | {bp.get('effective_batch_size','n/a')} | {x['inference_wall_seconds']:.2f} | {f'{speedup:.2f}×' if speedup is not None else 'n/a'} | {fmt(tmem)} | {fmt(imem)} |")
    lines += ["", "## Layer masks", ""]
    for x in results[1:]:
        pruning=x.get("layer_pruning") or {}
        lines.append(f"- **{x['name']}** retained indices: `{pruning.get('retained_layer_indices', [])}`; removed indices: `{pruning.get('removed_layer_indices', [])}`.")
    lines += ["", "Inference timings are fresh full-V0 measurements from the common local inference implementation. Batch calibration and peak-memory data are saved in each checkpoint and in `results.json`.", "", "Artifacts: `artifacts/analysis/e41_qwen35_layer_removal/results.json` and the per-model V0 probability Parquet files.",""]
    DOC.write_text("\n".join(lines))
    update_catalog(summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
