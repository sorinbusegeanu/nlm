"""Time E33-F/E34-F candidate scoring on E32's exact cached 4k workload.

Base checkpoint and grouping durations are reused from the completed E31/E32
RTX 5070 Ti passes. The new classifier stages and Hungarian passes are timed
locally against the exact cached probabilities and group assignments.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src import e33, e34, ensemble_catalog as catalog
from src.e13_b_inference import family_max_scores, make_features
from src.e20 import hungarian
from src.e13_reliability_router import family_balanced_probabilities
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
E31 = ROOT / "artifacts/analysis/e31"
E32 = ROOT / "artifacts/analysis/e32"
E33 = ROOT / "artifacts/analysis/e33"
E34 = ROOT / "artifacts/analysis/e34"
OUT = E34 / "inference_benchmark.json"


def _sha_values(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _score_base(count, workload, roster, members, probs):
    manifest_path = E31 / "e31_manifest.json" if count == 40 else E32 / "e32_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    router_name = manifest["router"]
    router_path = manifest_path.parent / router_name
    model = CatBoostClassifier()
    model.load_model(str(router_path))
    families = [_model_family(members[sid]["model"]) for sid in roster]
    tensor = np.stack([probs[sid] for sid in roster]).astype(np.float32)
    if tensor.shape != (count, len(workload), len(LABELS)):
        raise ValueError(f"{count}-checkpoint workload tensor is misaligned: {tensor.shape}")
    member_rows = [{"candidate_id": sid, "family": family}
                   for sid, family in zip(roster, families)]
    feature_frame = make_features(tensor, member_rows, LABELS,
                                  manifest["reliability_statistics"])
    pred = model.predict_proba(feature_frame)[:, list(model.classes_).index(1)]
    reliability = pred.reshape(len(workload), count).T
    family_probs = family_balanced_probabilities(tensor, families)
    score, safe = family_max_scores(tensor.argmax(axis=2), reliability,
                                    family_probs, families)
    score = np.where(np.isfinite(score), score, safe)
    return tensor, families, score, model


def _time_rescorer(tag, count, roster, families, probabilities, base_scores,
                   final_model_path, reliability_path, groups_frame):
    e33._BUILD_CP_IDS = list(roster)
    started = time.perf_counter()
    rank = e33._rank_positions(probabilities)
    reliability_feature_seconds = time.perf_counter() - started
    stats_npz = np.load(reliability_path, allow_pickle=True)
    stats = {key: stats_npz[key] for key in stats_npz.files}
    started = time.perf_counter()
    features = e33._build_candidate_features(np.arange(base_scores.shape[0]), stats,
        probabilities=probabilities, rank_position=rank, base_scores=base_scores,
        families=families)
    candidate_feature_seconds = time.perf_counter() - started
    model = CatBoostClassifier()
    model.load_model(str(final_model_path))
    started = time.perf_counter()
    p_wrong = model.predict_proba(features)[:, list(model.classes_).index(1)]
    catboost_seconds = time.perf_counter() - started
    p_correct = (1.0 - p_wrong).reshape(base_scores.shape)
    started = time.perf_counter()
    adjusted = base_scores * p_correct
    score_adjustment_seconds = time.perf_counter() - started

    row_ids = groups_frame.scenario_row_id.astype(str).to_numpy()
    scenario_ids = groups_frame.scenario_id.astype(str).to_numpy()
    group_times = {}
    for method in ("E19-F-A", "E19-F-C", "E19-D"):
        values = groups_frame[f"group_{method}"].astype(str).to_numpy()
        groups = np.asarray([f"{scenario}::{group}" for scenario, group in zip(scenario_ids, values)])
        started = time.perf_counter()
        pred = hungarian(adjusted, groups, row_ids)
        elapsed = time.perf_counter() - started
        group_times[method] = {"hungarian_seconds": elapsed,
            "predicted_rows": int(len(pred)), "unique_groups": int(len(set(groups)))}
    return adjusted, {"checkpoint_count": count,
        "reliability_feature_calculation_seconds": reliability_feature_seconds,
        "candidate_feature_construction_seconds": candidate_feature_seconds,
        "catboost_inference_seconds": catboost_seconds,
        "score_adjustment_seconds": score_adjustment_seconds,
        "candidate_rescoring_seconds": reliability_feature_seconds + candidate_feature_seconds
            + catboost_seconds + score_adjustment_seconds,
        "hungarian_by_grouping_method": group_times}


def run():
    e32_manifest = json.loads((E32 / "e32_manifest.json").read_text())
    e31_manifest = json.loads((E31 / "e31_manifest.json").read_text())
    roster40, roster60 = list(e31_manifest["roster"]), list(e32_manifest["roster"])
    if roster60[:40] != roster40 or len(roster60) != 60:
        raise ValueError("E32 roster is not the exact E31 roster plus 20 checkpoints")
    prob40 = np.load(E32 / "workload_probabilities_40.npz", allow_pickle=True)
    prob60 = np.load(E32 / "workload_probabilities_60.npz", allow_pickle=True)
    row_ids = prob60["row_ids"].astype(str)
    if prob40["row_ids"].astype(str).tolist() != row_ids.tolist() or len(row_ids) != 4000:
        raise ValueError("E31/E32 inference workload row order differs")
    workload = pd.read_parquet(ROOT / "data/processed/dataset_v0_group.parquet")
    workload = workload.set_index("scenario_row_id").loc[row_ids].reset_index()
    bench32 = json.loads((E32 / "inference_benchmark.json").read_text())
    if _sha_values(workload.scenario_row_id.astype(str)) != bench32["sample_row_ids_sha256"]:
        raise ValueError("Inference benchmark rows do not match the E32 workload hash")
    v0 = catalog.cascades._v0(ROOT)
    _, members, _, _, _ = catalog._load_inputs(ROOT, v0, version="trust-list-2-v037")
    probs_by_count = {
        40: {sid: prob40[sid].astype(np.float32) for sid in roster40},
        60: {sid: prob60[sid].astype(np.float32) for sid in roster60}}
    records = {}
    scores = {}
    tensors = {}
    model_objs = {}
    for count, roster in ((40, roster40), (60, roster60)):
        tensor, families, base, router = _score_base(count, workload, roster, members,
            probs_by_count[count])
        tensors[count], scores[count], model_objs[count] = tensor, base, router
        # Sanity-check the deployment route against the previously measured
        # benchmark's independent predictions before timing the new layer.
        saved = pd.read_parquet(E32 / f"workload_predictions_{count}.parquet")
        p = np.asarray(LABELS, dtype=object)[base.argmax(axis=1)]
        if (saved.scenario_row_id.astype(str).tolist() != row_ids.tolist()
                or not np.array_equal(p, saved.Independent.astype(str).to_numpy())):
            raise AssertionError(f"Reconstructed frozen {count}-checkpoint route differs from saved run")
        tag = "e33" if count == 40 else "e34"
        final_model = E33 / "e33_final.cbm" if count == 40 else E34 / "e34_final.cbm"
        rel_path = E33 / "e33_final_reliability.npz" if count == 40 else E34 / "e34_final_reliability.npz"
        adjusted, timing = _time_rescorer(tag, count, roster, families, tensor,
            base, final_model, rel_path, saved)
        records[tag] = {**timing, "roster": roster,
            "final_classifier": str(final_model.relative_to(ROOT)),
            "final_classifier_sha256": e34._sha(final_model),
            "local_probability_cache": f"artifacts/analysis/e32/workload_probabilities_{count}.npz"}

    # Use direct E31 benchmark values for the historical row. For the 40-row
    # complete all-three-grouper comparison, E32 also measured a replay of the
    # exact E31 roster on this identical workload.
    b31 = json.loads((E31 / "inference_benchmark.json").read_text())
    full = bench32["full_pipeline_by_roster"]
    sched = bench32["e28_two_gpu_scheduler_estimates"]
    base_rows = {}
    for key, count in (("E31", 40), ("E32", 60)):
        f = full[str(count)]
        grouping = f["grouping_and_assignment"]
        embedding = f["embedding_stage"]
        grouping_only = (embedding["loading_seconds"] + embedding["inference_seconds"]
            + embedding["cleanup_seconds"] + sum(x["reconstruction_seconds"]
                for x in grouping.values()))
        old_hungarian = sum(x["hungarian_seconds"] for x in grouping.values())
        base_rows[key] = {"checkpoint_wall_seconds": f["checkpoint_wall_seconds"],
            "group_reconstruction_and_embedding_seconds": grouping_only,
            "old_hungarian_seconds": old_hungarian,
            "end_to_end_seconds": f["end_to_end_seconds_including_grouping_and_hungarian"],
            "measured_full_pipeline_overhead_seconds":
                f["end_to_end_seconds_including_grouping_and_hungarian"]
                - f["checkpoint_wall_seconds"],
            "scheduler": sched[str(count)]}
    # E31's own report is retained as a provenance cross-check.
    base_rows["E31"]["e31_reported_end_to_end_seconds"] = b31["end_to_end_seconds"]
    base_rows["E31"]["e31_reported_grouping_seconds"] = b31["grouping_runtime"][
        "total_local_grouping_and_assignment_seconds"]

    table = []
    e31_cand = 0.0
    for experiment, key in (("E31", "E31"), ("E33-F", "E31"),
                            ("E32", "E32"), ("E34-F", "E32")):
        cand = records["e33"]["candidate_rescoring_seconds"] if experiment == "E33-F" else (
            records["e34"]["candidate_rescoring_seconds"] if experiment == "E34-F" else 0.0)
        base = base_rows[key]
        measured_hungarian = sum(records["e33" if key == "E31" else "e34"][
            "hungarian_by_grouping_method"][m]["hungarian_seconds"]
            for m in ("E19-F-A", "E19-F-C", "E19-D"))
        grouping_adjusted = base["measured_full_pipeline_overhead_seconds"]
        if experiment in ("E33-F", "E34-F"):
            grouping_adjusted += measured_hungarian - base["old_hungarian_seconds"]
        total = base["checkpoint_wall_seconds"] + cand + grouping_adjusted
        scheduler_seconds = base["scheduler"]["estimated_wall_seconds_including_grouping"]
        if experiment in ("E33-F", "E34-F"):
            scheduler_seconds += cand + measured_hungarian - base["old_hungarian_seconds"]
        table.append({"experiment": experiment,
            "checkpoints": 40 if key == "E31" else 60,
            "checkpoint_inference_wall_seconds": base["checkpoint_wall_seconds"],
            "candidate_rescoring_seconds": cand,
            "grouping_and_remaining_overhead_seconds": grouping_adjusted,
            "total_local_wall_seconds": total,
            "estimated_dual_t4_kaggle_seconds": scheduler_seconds,
            "estimated_dual_t4_kaggle_minutes": scheduler_seconds / 60,
            "timing_basis": "Existing measured checkpoint and grouping stages plus measured new CPU rescoring; same 4,000-row RTX 5070 Ti workload."})
    output = {"experiment": "E34/E33-F inference benchmark",
        "device": bench32["device"], "device_total_memory_mib": bench32["device_total_memory_mib"],
        "sample_rows": 4000, "sample_seed": bench32["sample_seed"],
        "sample_row_ids_sha256": bench32["sample_row_ids_sha256"],
        "sample_text_sha256": bench32["sample_text_sha256"],
        "new_candidate_rescoring_measurements": records,
        "reused_base_benchmark_measurements": base_rows,
        "comparison": table,
        "methodology_note": "E33-F uses E31's exact 40-member roster and E31 router; E34-F uses the exact E32 60-member roster and frozen E32 router. The base checkpoint inference/loading/cleanup and grouping reconstruction stages are reused from measured E31/E32 runs on the same 4,000 rows. New classifier feature/rank/scoring/adjustment and Hungarian passes were measured locally against their cached probabilities; totals add these measured stages to the reused base stages. Kaggle estimates use E28's longest-runtime-first two-GPU schedule plus measured classifier overhead, without assuming T4 parity beyond the prior scheduler estimate."}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    _write_report(output)
    return output


def _write_report(result):
    lines = ["# E34 inference benchmark", "",
        "All rows refer to the same deterministic 4,000-response workload (seed 20261123) used by E31/E32 and the local RTX 5070 Ti. Existing checkpoint and group reconstruction timings are reused; the new candidate rank/features, CatBoost scoring, score adjustment and per-method Hungarian passes are measured in this run from the exact cached workload probabilities.", "",
        "| Experiment | Checkpoints | Checkpoint inference wall time | Candidate rescoring time | Grouping and remaining overhead | Total local wall time | Estimated dual-T4 Kaggle time |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["comparison"]:
        lines.append(f"| {row['experiment']} | {row['checkpoints']} | {row['checkpoint_inference_wall_seconds']:.2f} s | {row['candidate_rescoring_seconds']:.2f} s | {row['grouping_and_remaining_overhead_seconds']:.2f} s | {row['total_local_wall_seconds']:.2f} s | {row['estimated_dual_t4_kaggle_seconds']:.2f} s ({row['estimated_dual_t4_kaggle_minutes']:.2f} min) |")
    lines += ["", "## Candidate classifier timings", "",
        "| Classifier | Roster | Reliability/rank features | Candidate feature construction | CatBoost inference | Score adjustment | Total rescoring |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, data in result["new_candidate_rescoring_measurements"].items():
        lines.append(f"| {name.upper()}-F | {data['checkpoint_count']} | {data['reliability_feature_calculation_seconds']:.3f} s | {data['candidate_feature_construction_seconds']:.3f} s | {data['catboost_inference_seconds']:.3f} s | {data['score_adjustment_seconds']:.4f} s | {data['candidate_rescoring_seconds']:.3f} s |")
    lines += ["", "## Grouping and assignment timings", "",
        "The existing E32 grouping measurements include EmbeddingGemma loading/inference/cleanup and E19-F-A, E19-F-C, E19-D reconstruction. The old assignment durations are removed from that stage total and the newly timed assignment durations for the rescored outputs are added.", "",
        "| Classifier | E19-F-A Hungarian | E19-F-C Hungarian | E19-D Hungarian |",
        "|---|---:|---:|---:|"]
    for name, data in result["new_candidate_rescoring_measurements"].items():
        g = data["hungarian_by_grouping_method"]
        lines.append(f"| {name.upper()}-F | {g['E19-F-A']['hungarian_seconds']:.3f} s | {g['E19-F-C']['hungarian_seconds']:.3f} s | {g['E19-D']['hungarian_seconds']:.3f} s |")
    lines += ["", "The dual-T4 figures are E28 scheduler estimates, not direct Kaggle measurements. Candidate rescoring is added after the checkpoint workers complete. The measured GPU inference stages themselves were not repeated for this report; their saved per-checkpoint loading, inference and cleanup records remain in the E31/E32 benchmark artifacts. This avoids redundant base-model inference while preserving the exact workload and checkpoint timing measurements.", "",
        "Detailed machine-readable timings: `artifacts/analysis/e34/inference_benchmark.json`.", ""]
    (ROOT / "docs/e34_inference_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["comparison"], indent=2))
