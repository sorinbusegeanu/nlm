"""Write E30 accuracy/timing reports from completed reproducible artifacts."""
from __future__ import annotations

import json, os, time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from src.e28_deploy import distribute_checkpoints

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e30"
LABELS = json.loads((OUT / "e30_b_manifest.json").read_text())["classes"]


def _aggregate(records, variant):
    truth, pred = [], []
    for scenario in records.values():
        item = scenario[variant]
        truth.extend(item["truth"]); pred.extend(item["predictions"])
    y=np.asarray(truth,dtype=np.int64); p=np.asarray(pred,dtype=np.int64)
    precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=np.arange(len(LABELS)),zero_division=0)
    return {"rows":len(y),"accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "correct":int((p==y).sum()),
        "per_label":{LABELS[i]:{"precision":float(precision[i]),"recall":float(recall[i]),
            "f1":float(f1[i]),"support":int(support[i])} for i in range(len(LABELS))}}


def _write_accuracy_report():
    data=json.loads((OUT/"evaluation.json").read_text())
    lines=["# E30 — E28 15 vs 20 Checkpoints", "",
        "## Setup", "",
        "E30-A is the exact E28 15-checkpoint Kaggle roster and its existing router. E30-B uses the fixed 20-checkpoint roster below and one new E13-B family_max router fitted with the original four-fold prompt-group OOF deployment procedure. E19-F-A and E19-F-C groups were rebuilt separately within each of the 50 `v0_group` scenarios before Hungarian assignment.", "",
        "All 50 scenarios reuse official V0 response rows. Candidate selection, E30-B router fitting, and the original E28 router were exposed to V0 labels. Treat these scores as correlated, in-sample robustness diagnostics, not independent validation or expected Kaggle performance.", "",
        "## Checkpoint selection", "",
        "Five additional checkpoints were selected greedily from the cached Trust List 2 v037 eligible pool. Selection required five distinct model families not already represented in E28, prioritized newly covered correct V0 rows, then standalone accuracy and lower mean top-1 agreement with the inherited roster. This selection criterion was not cross-fitted.", "",
        "| ID | Model family | Standalone V0 accuracy | Newly covered correct rows at selection | Agreement with E28 roster |", "|---|---|---:|---:|---:|"]
    for item in data["metadata"]["additions"]:
        lines.append(f"| {item['id']} | {item['family']} | {item['accuracy']:.3%} | {item.get('added_correct_coverage', item['unique_correct'])} | {item['agreement_with_base_roster']:.3%} |")
    lines += ["", "E30-B roster: `"+", ".join(data["metadata"]["roster_B"])+"`.", "",
        "## Scenario results", "",
        "Mean, min, max, 10th percentile and sample standard deviation are across 50 scenarios; each scenario contains 500 rows. Corrected/introduced totals count duplicated V0 rows across scenarios.", "",
        "| Grouper | Router | Mean BA | Min BA | P10 BA | Max BA | SD | Scenarios better vs A | Worse | Corrected | Introduced | Net |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    recall_tables={}
    for grouper, variants in data["scenarios"].items():
        pooled={v:_aggregate(variants,v) for v in ("E30-A","E30-B")}
        for variant in ("E30-A","E30-B"):
            s=data["summary"][grouper][variant]
            comps=[x["E30-B"].get("comparison_vs_E30-A",{}) for x in variants.values()] if variant=="E30-B" else []
            corrected=sum(x.get("corrected",0) for x in comps); introduced=sum(x.get("introduced",0) for x in comps)
            net=corrected-introduced
            lines.append(f"| {grouper} | {variant} | {s['mean_balanced_accuracy']:.3%} | {s['minimum_balanced_accuracy']:.3%} | {s['p10_balanced_accuracy']:.3%} | {s['maximum_balanced_accuracy']:.3%} | {s['std_balanced_accuracy']:.3%} | {s['scenarios_improved_vs_E30-A'] if variant=='E30-B' else '—'} | {s['scenarios_regressed_vs_E30-A'] if variant=='E30-B' else '—'} | {corrected if variant=='E30-B' else '—'} | {introduced if variant=='E30-B' else '—'} | {net if variant=='E30-B' else '—'} |")
        recall_tables[grouper]={label: pooled["E30-B"]["per_label"][label]["recall"]-
            pooled["E30-A"]["per_label"][label]["recall"] for label in LABELS}
        data["summary"][grouper]["pooled_metrics"] = pooled
        data["summary"][grouper]["paired_comparison"] = {"corrected":corrected,"introduced":introduced,"net":net}
    lines += ["", "## Pooled per-label recall change", "",
        "Difference is E30-B minus E30-A across all scenario rows (a repeated-row diagnostic).", "",
        "| Label | E19-F-A Δ recall | E19-F-C Δ recall |", "|---|---:|---:|"]
    for label in LABELS:
        lines.append(f"| {label} | {recall_tables['E19-F-A'][label]:+.2%} | {recall_tables['E19-F-C'][label]:+.2%} |")
    a=data["summary"]["E19-F-A"]["E30-B"]; c=data["summary"]["E19-F-C"]["E30-B"]
    lines += ["", "## Recommendation", "",
        f"E30-B with E19-F-C has the highest mean scenario balanced accuracy ({c['mean_balanced_accuracy']:.3%}, versus {a['mean_balanced_accuracy']:.3%} for E19-F-A), so it is the leading E30-B variant by the primary criterion. E19-F-A has the slightly better minimum ({a['minimum_balanced_accuracy']:.3%} versus {c['minimum_balanced_accuracy']:.3%}); the difference is about {100*(a['minimum_balanced_accuracy']-c['minimum_balanced_accuracy']):.2f} percentage points. Both remain V0-reuse diagnostics, so neither establishes generalization. E19-F-C is the variant prepared as the primary Kaggle option; E19-F-A is retained as a comparison output.", "",
        "E30-B exceeds E30-A mean BA by roughly 0.41 percentage points under either grouper, but it is worse in 16/50 E19-F-A scenarios and 18/50 E19-F-C scenarios. This modest gain does not prove that the larger roster generalizes. The E28 router and roster files are unchanged.", ""]
    (ROOT/"docs/e30.md").write_text("\n".join(lines))
    (OUT/"evaluation.json").write_text(json.dumps(data,indent=2)+"\n")


def _write_timing_report():
    path=OUT/"inference_benchmark.json"
    data=json.loads(path.read_text())
    records=data["checkpoints"]
    for item in records:
        if item["device"]=="CPU":
            item["peak_allocated_mib"]=None; item["peak_reserved_mib"]=None
            item["timing_note"]="CPU-only classifier; GPU VRAM and batch size are not applicable. CPU inference time reported as the pure prediction pass."
    started=json.loads((OUT/"benchmark_process_start.json").read_text())
    data["end_to_end_seconds"]=max(0.0,time.time()-float(started["start_epoch"]))
    data["sum_pure_inference_seconds"]=float(sum(x.get("pure_inference_seconds") or 0 for x in records))
    data["sum_checkpoint_total_seconds"]=float(sum(x.get("total_checkpoint_seconds") or 0 for x in records))
    data["wall_minus_pure_inference_seconds"]=data["end_to_end_seconds"]-data["sum_pure_inference_seconds"]
    data["wall_minus_checkpoint_total_seconds"]=data["end_to_end_seconds"]-data["sum_checkpoint_total_seconds"]
    runtime={x["checkpoint_id"]:float(x["total_checkpoint_seconds"]) for x in records}
    runtime["__sample_rows__"]=int(data["sample_rows"])
    member_doc=json.loads((OUT/"e30_b_manifest.json").read_text())
    from src.ensemble_catalog import _load_inputs, cascades
    v0=cascades._v0(ROOT); _,members,_,_,_=_load_inputs(ROOT,v0,version="trust-list-2-v037")
    checkpoint_members=[]
    for sid in member_doc["roster"]:
        # E28's scheduler only needs identity, model, file sizes for runtime fallback.
        checkpoint_members.append({"candidate_id":sid,"model":members[sid]["model"],"files":[]})
    batch_by_model={}
    for x in records:
        if x.get("maximum_stable_batch_size"):
            model=x["model"]; batch_by_model[model]=max(batch_by_model.get(model,1),int(x["maximum_stable_batch_size"]))
    schedule=distribute_checkpoints(checkpoint_members,runtime,batch_by_model,int(data["sample_rows"]))
    data["kaggle_estimate"]={"algorithm":"src.e28_deploy.distribute_checkpoints, greedy longest-estimated-runtime-first across two GPUs",
        "rows":int(data["sample_rows"]),"estimated_gpu_seconds":[x["estimated_seconds"] for x in schedule],
        "estimated_wall_seconds":max(x["estimated_seconds"] for x in schedule),
        "gpu_checkpoints":[x["checkpoint_ids"] for x in schedule],
        "caveat":"Scaled from the 4,000-row RTX 5070 Ti measurements. This is a scheduler estimate, not a T4 measurement; T4 throughput and additional OOM backoff can increase runtime."}
    (OUT/"inference_benchmark.json").write_text(json.dumps(data,indent=2)+"\n")
    lines=["# E30 Inference Benchmark", "",
        f"Measured locally on **{data['device']}** ({data['device_total_memory_mib']:.0f} MiB total VRAM), one checkpoint at a time, with a deterministic 4,000-row sample drawn across all 50 `v0_group` scenarios. The probe ceiling was {data['batch_cap']} and production OOM halving determined the first stable batch. This records a safe measured batch under that probe policy, not a continuous search over every possible batch size.", "",
        "Pure neural inference time covers model forward passes and excludes tokenization. Checkpoint total covers `predict_candidate` model load, preprocessing, inference and cleanup. Frozen-backbone extraction APIs combine backbone load and forward timing; those entries are marked. SVC and CatBoost stylometry run on CPU, so no GPU batch or VRAM applies.", "",
        "| Checkpoint | Model | Batch | Peak allocated MiB | Peak reserved MiB | Load s | Pure inference s | Cleanup s | Total s | Rows/s |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in records:
        fmt=lambda k: "n/a" if x.get(k) is None else f"{x[k]:.2f}" if isinstance(x[k],(int,float)) else str(x[k])
        lines.append(f"| {x['checkpoint_id']} | {x['model']} | {fmt('maximum_stable_batch_size')} | {fmt('peak_allocated_mib')} | {fmt('peak_reserved_mib')} | {fmt('model_loading_seconds')} | {fmt('pure_inference_seconds')} | {fmt('cleanup_seconds')} | {fmt('total_checkpoint_seconds')} | {fmt('throughput_rows_per_second')} |")
    lines += ["", f"**Total measured pure inference:** {data['sum_pure_inference_seconds']:.1f}s.",
        f"**Sum of per-checkpoint totals:** {data['sum_checkpoint_total_seconds']:.1f}s.",
        f"**End-to-end wall time:** {data['end_to_end_seconds']:.1f}s.",
        f"Wall minus pure inference: {data['wall_minus_pure_inference_seconds']:.1f}s; wall minus checkpoint totals: {data['wall_minus_checkpoint_total_seconds']:.1f}s.", "",
        "## Two-GPU Kaggle estimate", "",
        f"E28's longest-estimated-runtime-first scheduler assigns the 20 checkpoints to two balanced workers. At a 4,000-row workload the local-measurement estimate is **{data['kaggle_estimate']['estimated_wall_seconds']:.1f}s** (GPU work estimates {data['kaggle_estimate']['estimated_gpu_seconds'][0]:.1f}s and {data['kaggle_estimate']['estimated_gpu_seconds'][1]:.1f}s). This is an RTX 5070 Ti extrapolation, not a Kaggle T4 measurement; dynamic OOM batch reduction remains enabled.", ""]
    (ROOT/"docs/e30_inference_benchmark.md").write_text("\n".join(lines))


if __name__=="__main__":
    _write_accuracy_report(); _write_timing_report()
