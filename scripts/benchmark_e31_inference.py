"""Benchmark E31's fixed 40-member roster on the E30 4k-row workload."""
from __future__ import annotations

import gc, json, os, time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from scripts.benchmark_e30_inference import _run_one, _sample
from scripts.publish_kaggle_checkpoint_store import resolve_checkpoint
from src.e28_deploy import distribute_checkpoints

ROOT = Path(__file__).resolve().parents[1]
E31 = ROOT / "artifacts/analysis/e31"
MANIFEST = E31 / "e31_manifest.json"
DATA = ROOT / "data/processed/dataset_v0_group.parquet"
TRUST = ROOT / "docs/trust-list-2-v037.csv"
E30_BENCH = ROOT / "artifacts/analysis/e30/inference_benchmark.json"
OUT = E31 / "inference_benchmark.json"


def _member(sid, table):
    row = table.loc[sid]
    exp = row.experiment_id
    try:
        _, checkpoint = resolve_checkpoint(exp)
    except FileNotFoundError:
        # Some Trust List 2 entries, notably CatBoost two-stage checkpoints,
        # keep their run manifest beside the checkpoint rather than under
        # artifacts/predictions. The exported checkpoint_path is authoritative.
        checkpoint = Path(row.checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Trust List checkpoint path missing for {sid}: {checkpoint}")
    prediction_manifest = ROOT / "artifacts/predictions" / f"{exp}_manifest.json"
    if prediction_manifest.is_file():
        run = json.loads(prediction_manifest.read_text())
    else:
        run_manifest = checkpoint.parent / f"{exp}_manifest.json"
        if not run_manifest.is_file():
            run_manifest = checkpoint / "checkpoint_manifest.json"
        run = json.loads(run_manifest.read_text()) if run_manifest.is_file() else {}
    return {"model": row.model, "experiment_id": exp, "config": run.get("config", {}),
            "model_revision": run.get("model_revision"),
            "calibration_temperature": run.get("calibration_temperature", 1.0)}, checkpoint


def _benchmark_grouping(sample, roster, table):
    """Measure E28-compatible semantic embedding, grouping and assignment stages."""
    import joblib
    from transformers import AutoModel, AutoTokenizer
    from src.e30_deploy import predict_f_groups, hungarian
    from scripts.run_e30_corrected_selection import route
    from src.ensemble_catalog import _load_inputs, cascades

    v0 = cascades._v0(ROOT)
    _, members, probabilities, excluded, snapshot = _load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    if excluded:
        raise ValueError(f"Missing E31 probability caches for grouping benchmark: {excluded}")
    ids_v0 = v0.id.astype(str).to_numpy()
    row_map = {rid: i for i, rid in enumerate(ids_v0)}
    rowix = np.asarray([row_map[str(x)] for x in sample.original_id], dtype=np.int64)
    grouped_probs = {sid: probabilities[sid][rowix] for sid in roster}
    subframe = sample.reset_index(drop=True)
    stats = json.loads(MANIFEST.read_text())["reliability_statistics"]
    _, score_matrix = route(subframe, roster, members, grouped_probs,
                            E31 / "e31_router.cbm", stats)
    ids = subframe.scenario_row_id.astype(str).to_numpy()
    texts = subframe.text.astype(str).to_numpy()

    # Match the grouping encoder, pooling and truncation used in E28's notebook.
    member, _ = _member("N0054", table)
    repo, revision = member["config"]["checkpoint"], member["config"].get("revision")
    from huggingface_hub import snapshot_download
    base = snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="right")
    model = AutoModel.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.float16).to(device).eval()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    embed_start = time.perf_counter(); embeddings = []; pos = 0; batch_size = 64
    while pos < len(texts):
        take = min(batch_size, len(texts) - pos)
        try:
            toks = tokenizer(["task: clustering | query: " + str(x)
                for x in texts[pos:pos + take]], padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**toks).last_hidden_state.float()
                mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32)); pos += take
        except torch.cuda.OutOfMemoryError:
            if take <= 1:
                raise
            batch_size = max(1, take // 2); torch.cuda.empty_cache()
    embedding_seconds = time.perf_counter() - embed_start
    emb = np.concatenate(embeddings)
    peak_allocated = float(torch.cuda.max_memory_allocated() / 2**20)
    peak_reserved = float(torch.cuda.max_memory_reserved() / 2**20)
    del model, tokenizer, embeddings; gc.collect(); torch.cuda.empty_cache()

    c_bundle = joblib.load(ROOT / "artifacts/analysis/e28/e19_f_c_external_grouper.joblib")
    a_bundle = {"threshold": 0.85, "neighbors": 40, "vectorizers": c_bundle["vectorizers"]}
    grouping = {}; group_ids = {}
    for method, bundle in (("E19-F-A", a_bundle), ("E19-F-C", c_bundle)):
        start = time.perf_counter()
        groups = predict_f_groups(method, ids, texts, emb, bundle).astype(str)
        duration = time.perf_counter() - start
        assignment_start = time.perf_counter()
        assigned = hungarian(score_matrix, groups, ids)
        assignment_seconds = time.perf_counter() - assignment_start
        grouping[method] = {"feature_retrieval_pair_scoring_clustering_seconds": float(duration),
            "hungarian_assignment_seconds": float(assignment_seconds),
            "predicted_groups": len(set(groups.tolist())),
            "maximum_group_size": max(np.unique(groups, return_counts=True)[1].tolist(), default=0),
            "predicted_label_counts": np.bincount(assigned, minlength=18).tolist()}
        group_ids[method] = groups
    return {"embedding_model": repo, "embedding_model_revision": revision,
        "embedding_rows": len(texts), "embedding_max_length": 512,
        "embedding_batch_size_final": batch_size,
        "grouping_embedding_seconds": float(embedding_seconds),
        "grouping_embedding_peak_allocated_mib": peak_allocated,
        "grouping_embedding_peak_reserved_mib": peak_reserved,
        "grouping_methods": grouping,
        "total_local_grouping_and_assignment_seconds": float(embedding_seconds + sum(
            x["feature_retrieval_pair_scoring_clustering_seconds"] +
            x["hungarian_assignment_seconds"] for x in grouping.values()))}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E31 inference benchmark requires CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(MANIFEST.read_text())
    roster = list(manifest["roster"])
    if len(roster) != 40 or len(set(roster)) != 40:
        raise ValueError("Expected exactly 40 unique E31 checkpoints")
    table = pd.read_csv(TRUST, dtype=str).drop_duplicates("sequence_id").set_index("sequence_id")
    sample = _sample(pd.read_parquet(DATA))
    texts = sample.text.astype(str).tolist()
    old = json.loads(E30_BENCH.read_text())
    reusable = {x["checkpoint_id"]: x for x in old.get("checkpoints", []) if x.get("status") == "ok"}
    measured = dict(reusable)
    result = {"experiment": "E31 fixed 40 roster inference benchmark",
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
        "sample_rows": len(sample), "sample_seed": 20261123,
        "sample_row_ids_sha256": __import__("hashlib").sha256(
            "\n".join(sample.scenario_row_id.astype(str)).encode()).hexdigest(),
        "sample_text_sha256": __import__("hashlib").sha256(
            "\n".join(sample.text.astype(str)).encode()).hexdigest(),
        "sample_policy": "Same deterministic 4,000-row sample used for E30: stratified across 50 v0_group scenarios by label and length bucket.",
        "roster": roster, "reused_E30_individual_measurements": sorted(set(roster) & set(reusable)),
        "checkpoints": []}
    if OUT.is_file():
        previous = json.loads(OUT.read_text())
        if (previous.get("roster") == roster
                and previous.get("sample_row_ids_sha256") == result["sample_row_ids_sha256"]
                and previous.get("sample_text_sha256") == result["sample_text_sha256"]):
            for record in previous.get("checkpoints", []):
                if record.get("status") == "ok":
                    measured[record["checkpoint_id"]] = record
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for sid in roster:
        if sid in measured:
            print(f"reuse E30 individual timing {sid}", flush=True)
            continue
        member, checkpoint = _member(sid, table)
        try:
            rec = _run_one(sid, member, {"checkpoint": checkpoint}, texts)
        except Exception as exc:
            rec = {"checkpoint_id": sid, "model": member["model"], "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
        measured[sid] = rec
        result["checkpoints"] = [measured[x] for x in roster if x in measured]
        result["completed_individual_count"] = len(result["checkpoints"])
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(rec, indent=2), flush=True)
        if rec.get("status") != "ok":
            raise RuntimeError(f"Checkpoint benchmark failed for {sid}; see {OUT}")

    # Actual single-GPU sequential pass of the complete roster, using each
    # member's proven batch size and the normal inference OOM fallback.
    e2e = []
    start = time.perf_counter()
    for sid in roster:
        member, checkpoint = _member(sid, table)
        cap = measured[sid].get("maximum_stable_batch_size")
        cp_start = time.perf_counter()
        rec = _run_one(sid, member, {"checkpoint": checkpoint}, texts, batch_cap=cap)
        rec["e2e_elapsed_before_checkpoint_seconds"] = cp_start - start
        e2e.append(rec)
        print(f"E31 end-to-end {sid}: {rec['total_checkpoint_seconds']:.1f}s", flush=True)
    wall = time.perf_counter() - start
    if any(x.get("status") != "ok" for x in e2e):
        raise RuntimeError("One or more E31 end-to-end checkpoint runs failed")
    schedule_rows = [{"candidate_id": x["checkpoint_id"], "model": x["model"], "files": []}
                     for x in e2e]
    runtime = {x["checkpoint_id"]: x["total_checkpoint_seconds"] for x in e2e}
    runtime["__sample_rows__"] = len(sample)
    schedule = distribute_checkpoints(schedule_rows, runtime, {}, len(sample))
    grouping_runtime = _benchmark_grouping(sample, roster, table)
    checkpoint_worker = float(max(x["estimated_seconds"] for x in schedule))
    estimated_pipeline = checkpoint_worker + grouping_runtime["total_local_grouping_and_assignment_seconds"]
    # Deployment overhead outside neural checkpoint inference is measured by
    # the package's full preprocess/group/assign timing when available.
    overhead = {"seconds": None, "source": "not separately benchmarked yet"}
    final = {**result, "checkpoints": [measured[sid] for sid in roster],
        "end_to_end_checkpoint_records": e2e,
        "sum_pure_inference_seconds": float(sum(x.get("pure_inference_seconds") or 0 for x in e2e)),
        "sum_checkpoint_total_seconds": float(sum(x["total_checkpoint_seconds"] for x in e2e)),
        "end_to_end_seconds": float(wall), "non_checkpoint_runtime_overhead": overhead,
        "two_T4_scheduler_estimate": {"scheduler": "E28 longest-runtime-first two-GPU scheduler",
            "estimated_worker_seconds": [float(x["estimated_seconds"]) for x in schedule],
            "estimated_wall_seconds": float(max(x["estimated_seconds"] for x in schedule)),
            "worker_checkpoint_ids": [x["checkpoint_ids"] for x in schedule],
            "within_7_2h_ceiling_before_grouping_overhead": bool(max(x["estimated_seconds"] for x in schedule) <= 7.2 * 3600),
            "grouping_overhead_added_seconds": grouping_runtime["total_local_grouping_and_assignment_seconds"],
            "estimated_pipeline_seconds_before_file_io_and_submission_writes": float(estimated_pipeline),
            "within_7_2h_ceiling_including_local_grouping_estimate": bool(estimated_pipeline <= 7.2 * 3600),
            "caveat": "Local RTX 5070 Ti checkpoint timings are extrapolated to T4; grouping uses measured local embedding/clustering/assignment time. Hidden Kaggle row count and T4 throughput/OOM behavior are not observed."},
        "grouping_runtime": grouping_runtime}
    OUT.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({k: v for k, v in final.items() if k not in {"checkpoints", "end_to_end_checkpoint_records"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
