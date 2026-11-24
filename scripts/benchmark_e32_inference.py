"""Benchmark E32's added checkpoints and 40/60 full inference pipelines."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from scripts.benchmark_e30_inference import _run_one, _sample
from scripts.benchmark_e31_inference import _member
from scripts.run_e30_corrected_selection import route
from src import e28
from src.e20 import hungarian
from src.e30_deploy import predict_f_groups
from src.ensemble_catalog import _load_inputs, cascades
from src.inference import predict_candidate
from src.trust_list2_weighted_vote import _model_family
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
E32 = ROOT / "artifacts/analysis/e32"
E31 = ROOT / "artifacts/analysis/e31"
DATA = ROOT / "data/processed/dataset_v0_group.parquet"
TRUST = ROOT / "docs/trust-list-2-v037.csv"
GROUPER_C = ROOT / "artifacts/analysis/e28/e19_f_c_external_grouper.joblib"
GROUPER_D = ROOT / "artifacts/analysis/e28/e19_d_external_grouper.joblib"
E19D_RECON = ROOT / "artifacts/analysis/e19d/reconstructed_prompts.parquet"
E19D_SEMANTIC = ROOT / "artifacts/analysis/e19d/semantic_embeddings.npy"
OUT = E32 / "inference_benchmark.json"


def _digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _prepare_spec(sid, table, batch_cap):
    member, checkpoint = _member(sid, table)
    spec = {"checkpoint": str(checkpoint), "candidate_id": sid,
        "model": member["model"], "config": {**member["config"],
        "inference_batch_size": int(batch_cap or 1)},
        "model_revision": member.get("model_revision"),
        "calibration_temperature": member.get("calibration_temperature", 1.0)}
    if member["model"] not in {"svc", "catboost_stylometry"}:
        checkpoint_conf = member["config"].get("checkpoint")
        revision = member.get("model_revision") or member["config"].get("revision")
        if checkpoint_conf and str(member["config"].get("training_mode", "")).lower() not in {"full", "encoder"}:
            spec["base_model"] = snapshot_download(repo_id=checkpoint_conf,
                revision=revision, local_files_only=True,
                allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                    "pytorch_model*.bin", "tokenizer*", "vocab*", "merges.txt",
                    "special_tokens_map.json", "added_tokens.json", "*.py",
                    "generation_config.json"])
    return member, checkpoint, spec


def _infer_one(sid, table, texts, batch_cap):
    member, checkpoint, spec = _prepare_spec(sid, table, batch_cap)
    if torch.cuda.is_available() and member["model"] not in {"svc", "catboost_stylometry"}:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    probs, metadata = predict_candidate(spec, texts, LABELS, ROOT,
        return_batch_metadata=True)
    elapsed = time.perf_counter() - started
    probs = np.asarray(probs, dtype=np.float64)
    if (probs.shape != (len(texts), len(LABELS)) or not np.isfinite(probs).all()
            or not np.allclose(probs.sum(1), 1, atol=1e-4)):
        raise ValueError(f"Invalid probability output for {sid}")
    neural = member["model"] not in {"svc", "catboost_stylometry"}
    record = {"checkpoint_id": sid, "model": member["model"], "rows": len(texts),
        "configured_batch_cap": int(batch_cap or 1),
        "effective_batch_size": metadata.get("effective_batch_size"),
        "peak_allocated_mib": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if neural and torch.cuda.is_available() else None),
        "peak_reserved_mib": (round(torch.cuda.max_memory_reserved() / 2**20, 1)
            if neural and torch.cuda.is_available() else None),
        "model_loading_seconds": metadata.get("model_loading_seconds"),
        "pure_inference_seconds": metadata.get("pure_inference_seconds"),
        "cleanup_seconds": metadata.get("cleanup_seconds"),
        "total_checkpoint_seconds": float(elapsed),
        "throughput_rows_per_second": len(texts) / max(float(
            metadata.get("pure_inference_seconds") or elapsed), 1e-9),
        "status": "ok"}
    del spec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, record


def _encode_for_grouping(sample, table):
    started = time.perf_counter()
    member, _ = _member("N0054", table)
    model_id = member["config"]["checkpoint"]
    revision = member["config"].get("revision")
    model_path = snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               padding_side="right")
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.float16).to(device).eval()
    model_loading_seconds = time.perf_counter() - started
    texts = sample.text.astype(str).tolist()
    embeddings = []
    position = 0
    batch_size = 64
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    infer_start = time.perf_counter()
    while position < len(texts):
        take = min(batch_size, len(texts) - position)
        try:
            tokens = tokenizer(["task: clustering | query: " + x
                for x in texts[position:position + take]], padding=True,
                truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state.float()
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
            position += take
        except torch.cuda.OutOfMemoryError:
            if take <= 1:
                raise
            batch_size = max(1, take // 2)
            torch.cuda.empty_cache()
    inference_seconds = time.perf_counter() - infer_start
    peak_allocated = float(torch.cuda.max_memory_allocated() / 2**20)
    peak_reserved = float(torch.cuda.max_memory_reserved() / 2**20)
    cleanup_start = time.perf_counter()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_seconds = time.perf_counter() - cleanup_start
    return np.concatenate(embeddings), {"model": model_id, "revision": revision,
        "loading_seconds": model_loading_seconds, "inference_seconds": inference_seconds,
        "cleanup_seconds": cleanup_seconds, "batch_size_final": batch_size,
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved}


def _score_and_group(count, sample, probabilities, members, deployment, router_path,
                     truth_v0, v0, groupers, embeddings):
    roster = deployment["roster"]
    pred, scores = route(sample, roster, members, probabilities, router_path,
                         deployment["reliability_statistics"])
    row_index = {str(x): i for i, x in enumerate(v0.id.astype(str))}
    grouped = {name: [] for name in ("E19-F-A", "E19-F-C", "E19-D")}
    predictions = {"Independent": pred}
    timers = {name: {"reconstruction_seconds": 0.0, "hungarian_seconds": 0.0}
              for name in grouped}
    d_recon = groupers["d_recon"]
    d_semantic = groupers["d_semantic"]
    for scenario_id, frame in sample.groupby("scenario_id", sort=True):
        frame = frame.reset_index(drop=True)
        positions = frame.index.to_numpy()
        # Recover these rows' offsets in the unsplit sample frame.
        sample_positions = sample.index[sample.scenario_id.astype(str) == str(scenario_id)].to_numpy()
        scenario_ids = frame.scenario_row_id.astype(str).to_numpy()
        text = frame.text.astype(str).to_numpy()
        v0_rows = np.asarray([row_index[str(x)] for x in frame.original_id], dtype=np.int64)
        emb = embeddings[sample_positions]
        for method in ("E19-F-A", "E19-F-C"):
            start = time.perf_counter()
            groups = predict_f_groups(method, scenario_ids, text, emb,
                                      groupers[method]).astype(str)
            timers[method]["reconstruction_seconds"] += time.perf_counter() - start
            assign_start = time.perf_counter()
            assigned = hungarian(scores[sample_positions], groups, scenario_ids)
            timers[method]["hungarian_seconds"] += time.perf_counter() - assign_start
            predictions.setdefault(method, np.full(len(sample), -1, dtype=np.int64))
            predictions[method][sample_positions] = assigned
            grouped[method].append({"scenario_id": str(scenario_id),
                                    "scenario_row_id": scenario_ids, "groups": groups})
        recon = d_recon.iloc[v0_rows].reset_index(drop=True).copy()
        recon["id"] = scenario_ids
        start = time.perf_counter()
        groups = e28._predict_d_groups(scenario_ids, text, recon,
            np.asarray(d_semantic)[v0_rows], groupers["D"])
        timers["E19-D"]["reconstruction_seconds"] += time.perf_counter() - start
        assign_start = time.perf_counter()
        assigned = hungarian(scores[sample_positions], groups, scenario_ids)
        timers["E19-D"]["hungarian_seconds"] += time.perf_counter() - assign_start
        predictions.setdefault("E19-D", np.full(len(sample), -1, dtype=np.int64))
        predictions["E19-D"][sample_positions] = assigned
        grouped["E19-D"].append({"scenario_id": str(scenario_id),
                                 "scenario_row_id": scenario_ids, "groups": groups})
    if any((np.asarray(p) < 0).any() for p in predictions.values()):
        raise AssertionError("E32 grouping/assignment did not produce every row")
    return scores, predictions, grouped, timers


def _pipeline(count, sample, roster, members, deployment, router_path, batch_by_id,
              table, v0, groupers):
    started = time.perf_counter()
    text = sample.text.astype(str).tolist()
    probabilities = {}
    records = []
    infer_started = time.perf_counter()
    for sid in roster:
        probs, record = _infer_one(sid, table, text, batch_by_id[sid])
        probabilities[sid] = probs
        records.append(record)
        print(f"E32 {count}-checkpoint pipeline: {sid} {record['total_checkpoint_seconds']:.1f}s",
              flush=True)
    checkpoint_inference_seconds = time.perf_counter() - infer_started

    _, members_all, _, excluded, _ = _load_inputs(ROOT, v0, version="trust-list-2-v037")
    if excluded:
        raise ValueError(f"E32 cached input metadata missing: {excluded}")
    embeddings, embedding_meta = _encode_for_grouping(sample, table)
    _, predictions, grouped, grouping_timers = _score_and_group(count, sample,
        probabilities, members_all, deployment, router_path,
        pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64),
        v0, groupers, embeddings)
    wall = time.perf_counter() - started
    probs_path = E32 / f"workload_probabilities_{count}.npz"
    np.savez_compressed(probs_path, row_ids=sample.scenario_row_id.astype(str).to_numpy(),
        **{sid: probabilities[sid].astype(np.float32) for sid in roster})
    predictions_frame = pd.DataFrame({"scenario_id": sample.scenario_id.astype(str),
        "scenario_row_id": sample.scenario_row_id.astype(str),
        "original_id": sample.original_id.astype(str), "checkpoint_count": count})
    for method, pred in predictions.items():
        predictions_frame[method] = np.asarray(LABELS, dtype=object)[pred]
    for method, scenario_records in grouped.items():
        group_ids = np.full(len(sample), "", dtype=object)
        scenario_positions = {str(sid): sample.index[
            sample.scenario_id.astype(str) == str(sid)].to_numpy()
            for sid in sample.scenario_id.astype(str).unique()}
        for item in scenario_records:
            group_ids[scenario_positions[str(item["scenario_id"])]] = item["groups"]
        if any(not value for value in group_ids):
            raise AssertionError(f"Missing group assignments in {method}/{count}")
        predictions_frame[f"group_{method}"] = group_ids.astype(str)
    predictions_frame["group_Independent"] = sample.scenario_id.astype(str).to_numpy()
    predictions_frame.to_parquet(E32 / f"workload_predictions_{count}.parquet", index=False)
    return {"checkpoint_count": count, "rows": len(sample),
        "checkpoint_wall_seconds": checkpoint_inference_seconds,
        "end_to_end_seconds_including_grouping_and_hungarian": float(wall),
        "checkpoint_sum_total_seconds": float(sum(x["total_checkpoint_seconds"] for x in records)),
        "checkpoint_sum_pure_inference_seconds": float(sum(x["pure_inference_seconds"] or 0 for x in records)),
        "checkpoint_records": records, "embedding_stage": embedding_meta,
        "grouping_and_assignment": grouping_timers,
        "probabilities_file": probs_path.name,
        "predictions_file": f"workload_predictions_{count}.parquet"}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E32 inference benchmark requires CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    E32.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((E32 / "e32_manifest.json").read_text())
    e31 = json.loads((E31 / "e31_manifest.json").read_text())
    e31_bench = json.loads((E31 / "inference_benchmark.json").read_text())
    if len(e31_bench.get("checkpoints", [])) != 40 or not e31_bench.get("end_to_end_seconds"):
        raise RuntimeError("Wait for the current E31 40-checkpoint benchmark to finish")
    roster40, roster60 = list(e31["roster"]), list(manifest["roster"])
    if roster60[:40] != roster40 or len(roster60) != 60:
        raise ValueError("E31/E32 fixed rosters do not align")
    data = pd.read_parquet(DATA)
    sample = _sample(data)
    sample_id_hash = _digest(sample.scenario_row_id.astype(str))
    sample_text_hash = _digest(sample.text.astype(str))
    if (sample_id_hash != e31_bench.get("sample_row_ids_sha256") or
            sample_text_hash != e31_bench.get("sample_text_sha256")):
        raise AssertionError("E32 workload differs from E31's exact 4,000-row sample")
    table = pd.read_csv(TRUST, dtype=str).drop_duplicates("sequence_id").set_index("sequence_id")
    old_by_id = {x["checkpoint_id"]: x for x in e31_bench["checkpoints"]}
    # CPU-only models in the E31 roster do not have a GPU batch-size measurement;
    # their prediction functions consume the workload as a single full batch.
    existing_batch = {sid: int(old_by_id[sid].get("maximum_stable_batch_size") or 1)
                      for sid in roster40}
    new_ids = roster60[40:]
    added_records = []
    added_batch = {}
    if OUT.is_file():
        try:
            previous = json.loads(OUT.read_text())
            if (previous.get("phase") == "individual_added_checkpoint_benchmarks"
                    and previous.get("sample_row_ids_sha256") == sample_id_hash
                    and previous.get("sample_text_sha256") == sample_text_hash):
                added_records = list(previous.get("completed_added", []))
                added_records = [x for x in added_records
                                 if x.get("checkpoint_id") in new_ids and x.get("status") == "ok"]
                added_batch = {x["checkpoint_id"]:
                    int(x.get("maximum_stable_batch_size") or 1) for x in added_records}
        except (OSError, ValueError, KeyError, TypeError):
            added_records, added_batch = [], {}
    for sid in new_ids:
        if sid in added_batch:
            continue
        member, checkpoint = _member(sid, table)
        rec = _run_one(sid, member, {"checkpoint": checkpoint},
                       sample.text.astype(str).tolist())
        if rec.get("status") != "ok":
            raise RuntimeError(f"Batch benchmark failed for {sid}: {rec}")
        added_records.append(rec)
        # SVC/CatBoost checkpoints execute on CPU, so there is no GPU OOM-tested
        # batch size or VRAM measurement. They still get timed on the full workload.
        added_batch[sid] = int(rec.get("maximum_stable_batch_size") or 1)
        OUT.write_text(json.dumps({"phase": "individual_added_checkpoint_benchmarks",
            "sample_rows": len(sample), "sample_row_ids_sha256": sample_id_hash,
            "sample_text_sha256": sample_text_hash, "completed_added": added_records}, indent=2) + "\n")
        print(json.dumps(rec, indent=2), flush=True)

    _, members, cached, excluded, _ = _load_inputs(ROOT, cascades._v0(ROOT),
                                                     version="trust-list-2-v037")
    if excluded:
        raise ValueError(f"E32 cached prediction metadata missing: {excluded}")
    full_v0 = cascades._v0(ROOT)
    groupers = {"E19-F-C": joblib.load(GROUPER_C),
        "E19-F-A": {"threshold": 0.85, "neighbors": 40,
                    "vectorizers": joblib.load(GROUPER_C)["vectorizers"]},
        "D": joblib.load(GROUPER_D),
        "d_recon": pd.read_parquet(E19D_RECON),
        "d_semantic": np.load(E19D_SEMANTIC, mmap_mode="r")}
    if groupers["d_recon"].id.astype(str).tolist() != full_v0.id.astype(str).tolist():
        raise ValueError("E19-D cached reconstructions do not align to V0")
    runtime_by_count = {}
    full_pipeline = {}
    for count, roster in ((40, roster40), (60, roster60)):
        roster_deployment = ({"roster": roster40,
            "reliability_statistics": e31["reliability_statistics"]} if count == 40 else manifest)
        router_path = (E31 / "e31_router.cbm") if count == 40 else E32 / "e32_router.cbm"
        batch_by_id = {**existing_batch, **added_batch}
        pipeline = _pipeline(count, sample, roster, members, roster_deployment,
            router_path, batch_by_id, table, full_v0, groupers)
        full_pipeline[str(count)] = pipeline
        runtime_by_count[count] = {x["checkpoint_id"]: x["total_checkpoint_seconds"]
                                   for x in pipeline["checkpoint_records"]}
        print(f"E32 full {count}-checkpoint pipeline wall: "
              f"{pipeline['end_to_end_seconds_including_grouping_and_hungarian']:.1f}s",
              flush=True)

    frame40 = pd.read_parquet(E32 / "workload_predictions_40.parquet")
    frame60 = pd.read_parquet(E32 / "workload_predictions_60.parquet")
    if (frame40.scenario_row_id.astype(str).tolist() !=
            frame60.scenario_row_id.astype(str).tolist()):
        raise AssertionError("40/60 full-pipeline workload row ordering differs")
    for method in ("E19-F-A", "E19-F-C", "E19-D", "Independent"):
        if frame40[f"group_{method}"].astype(str).tolist() != frame60[
                f"group_{method}"].astype(str).tolist():
            raise AssertionError(f"Full-pipeline grouping differs across rosters: {method}")

    # E28 scheduling applies longest-runtime-first balancing on two workers.
    from src.e28_deploy import distribute_checkpoints
    schedule_results = {}
    for count, roster in ((40, roster40), (60, roster60)):
        rows = [{"candidate_id": sid, "model": members[sid]["model"], "files": []}
                for sid in roster]
        times = {**runtime_by_count[count], "__sample_rows__": len(sample)}
        schedule = distribute_checkpoints(rows, times, {}, len(sample))
        group_runtime = full_pipeline[str(count)]["grouping_and_assignment"]
        grouping_seconds = (full_pipeline[str(count)]["embedding_stage"]["loading_seconds"] +
            full_pipeline[str(count)]["embedding_stage"]["inference_seconds"] +
            full_pipeline[str(count)]["embedding_stage"]["cleanup_seconds"] +
            sum(x["reconstruction_seconds"] + x["hungarian_seconds"]
                for x in group_runtime.values()))
        checkpoint_worker_seconds = float(max(x["estimated_seconds"] for x in schedule))
        schedule_results[str(count)] = {"scheduler": "E28 longest-runtime-first two-GPU scheduler",
            "estimated_worker_seconds": [float(x["estimated_seconds"]) for x in schedule],
            "estimated_checkpoint_wall_seconds": checkpoint_worker_seconds,
            "grouping_and_hungarian_seconds_added": grouping_seconds,
            "estimated_wall_seconds_including_grouping": checkpoint_worker_seconds + grouping_seconds,
            "worker_checkpoint_ids": [x["checkpoint_ids"] for x in schedule]}

    final = {"experiment": "E32 inference benchmark", "device": torch.cuda.get_device_name(0),
        "device_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
        "sample_rows": len(sample), "sample_seed": 20261123,
        "sample_row_ids_sha256": sample_id_hash, "sample_text_sha256": sample_text_hash,
        "added_checkpoints": added_records, "full_pipeline_by_roster": full_pipeline,
        "e28_two_gpu_scheduler_estimates": schedule_results,
        "note": "Both full-roster passes used the same exact 4,000-row E31 workload and one local RTX 5070 Ti. Kaggle estimates apply E28's two-GPU scheduler to measured per-checkpoint local times; T4 throughput is extrapolated rather than directly measured."}
    OUT.write_text(json.dumps(final, indent=2) + "\n")
    # Persist measured batch sizes in the frozen deployment manifest.
    manifest["inference_batch_parameters"] = {sid: {
        "maximum_stable_batch_size": (existing_batch.get(sid) or added_batch.get(sid)),
        "benchmark_rows": len(sample), "benchmark_sample_id_sha256": sample_id_hash}
        for sid in roster60}
    manifest["inference_benchmark"] = "inference_benchmark.json"
    (E32 / "e32_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"full_pipeline_by_roster": {k: {
        "end_to_end_seconds_including_grouping_and_hungarian": v["end_to_end_seconds_including_grouping_and_hungarian"]}
        for k, v in full_pipeline.items()}, "e28_estimates": schedule_results}, indent=2))


if __name__ == "__main__":
    main()
