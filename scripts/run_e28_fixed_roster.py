"""Replay E28 with the exact fixed 15-checkpoint E13-B deployment router."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src import e13_reliability_router as e13
from src import e20
from src.e28 import (_group_and_score, _grouping_metrics, _metric, _comparison,
                     route_family_max)
from src.ensemble_catalog import _load_inputs, cascades
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e28_fixed_roster"
E28_OUT = ROOT / "artifacts/analysis/e28"
MANIFEST = ROOT / "artifacts/kaggle/e20d-remote-catalogs/e13-b-checkpoints-v001/e13_b_manifest.json"
ROUTER = OUT / "e13_b_router.cbm"
SCENARIOS = E28_OUT / "scenario_rows.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    started = time.perf_counter()
    if not ROUTER.is_file():
        raise FileNotFoundError(f"Download the packaged E13-B router to {ROUTER}")
    v0 = cascades._v0(ROOT)
    ids = v0.id.astype(str).to_numpy()
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("V0 row IDs are not unique")
    truth = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if (truth < 0).any():
        raise ValueError("V0 contains a label outside canonical ordering")
    _, members, probabilities, _, snapshot = _load_inputs(ROOT, v0, version="trust-list-2-v037")
    manifest = json.loads(MANIFEST.read_text())
    deployment = manifest["deployment"]
    cp_ids = list(deployment["checkpoint_ids"])
    families = list(deployment["families"])
    if len(cp_ids) != 15 or cp_ids != [str(x["candidate_id"]) for x in manifest["members"]]:
        raise ValueError("Manifest roster and member order do not match the fixed 15-checkpoint deployment")
    if list(deployment["classes"]) != list(LABELS) or list(manifest["classes"]) != list(LABELS):
        raise ValueError("Packaged deployment labels do not match canonical order")
    missing = set(cp_ids) - set(probabilities)
    if missing:
        raise FileNotFoundError(f"V0 probability caches missing for deployment checkpoints: {sorted(missing)}")
    p = np.stack([probabilities[sid] for sid in cp_ids]).astype(np.float64)
    if p.shape != (15, len(v0), len(LABELS)):
        raise ValueError(f"Unexpected deployment probability shape: {p.shape}")

    # The shipped router has frozen all-V0 reliability statistics. Rebuild them
    # independently and assert they match before reproducing its inference path.
    stats = e13.reliability_statistics(p.argmax(axis=2), truth)
    for key, value in stats.items():
        if not np.allclose(value, np.asarray(deployment["reliability_statistics"][key]),
                           rtol=0, atol=1e-14):
            raise AssertionError(f"Packaged reliability statistics mismatch: {key}")
    n = len(v0)
    # build_feature_frame requires disjoint fit and target indices. Duplicate the
    # probabilities in memory so the frozen fit statistics and deployment rows
    # retain identical probability values without changing its feature semantics.
    augmented = np.concatenate([p, p], axis=1)
    features, _ = e13.build_feature_frame(
        augmented, cp_ids, families, truth, np.arange(n), np.arange(n, 2 * n))
    router = CatBoostClassifier()
    router.load_model(str(ROUTER))
    reliability = router.predict_proba(e13._cat_frame(features))[:, 1].reshape(n, 15).T
    family_probs = e13.family_balanced_probabilities(p, families)
    baseline, _, label_scores, safe_scores = route_family_max(
        p.argmax(axis=2), reliability, family_probs, families)
    if not np.isfinite(safe_scores).all() or safe_scores.shape != (n, len(LABELS)):
        raise AssertionError("Fixed-roster E28 scores are invalid")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fixed_roster_scores.npz", row_ids=ids,
        labels=np.asarray(LABELS), truth=truth, base_predictions=baseline,
        reliability=reliability, label_scores=label_scores, assignment_scores=safe_scores,
        family_probabilities=family_probs)
    context = {"v0": v0, "row_ids": ids, "truth": truth,
               "prompt_ids": v0.prompt_id.astype(str).to_numpy()}
    embeddings = np.load(ROOT / "artifacts/analysis/e19f/response_embeddings.npy", mmap_mode="r")
    f_c = joblib.load(E28_OUT / "e19_f_c_external_grouper.joblib")
    d_model = joblib.load(E28_OUT / "e19_d_external_grouper.joblib")
    methods = ["E19-F-C", "E19-F-A", "E19-D"]
    cache_hashes = {sid: members[sid]["cache_sha256"] for sid in cp_ids}
    result = {"experiment": "E28-fixed-deployment-roster", "trust_list_version": snapshot["version"],
        "checkpoint_count": 15, "checkpoint_roster": cp_ids, "families": families,
        "labels": list(LABELS), "rows": n, "baseline": _metric(truth, baseline),
        "router_model_sha256": sha256(ROUTER), "manifest_sha256": sha256(MANIFEST),
        "probability_cache_sha256": {sid: cache_hashes[sid] for sid in cp_ids},
        "validation_caveat": "Frozen packaged router and reliability statistics were fit using V0 development data; this is a deployment replay, not an untouched holdout estimate.",
        "groupers": {}, "scenarios": {}, "runtime_seconds": {}}
    result["scenario_rows_sha256"] = sha256(SCENARIOS)

    for method in methods:
        tic = time.perf_counter()
        groups = _group_and_score(method, np.arange(n), context, embeddings, f_c, d_model)
        if len(groups) != n or pd.Series(groups).value_counts().max() > len(LABELS):
            raise ValueError(f"{method} group output is misaligned or exceeds 18 rows")
        pred = e20.hungarian(safe_scores, groups, ids)
        result["groupers"][method] = {
            "metrics": _metric(truth, pred), "comparison_vs_fixed_E28_A": _comparison(truth, baseline, pred),
            "grouping_metrics": _grouping_metrics(context["prompt_ids"], groups),
            "group_size_distribution": {str(k): int(v) for k, v in
                pd.Series(groups).value_counts().value_counts().sort_index().items()},
            "runtime_seconds": time.perf_counter() - tic}
        pd.DataFrame({"id": ids, "prediction": np.asarray(LABELS)[pred],
                      "baseline_prediction": np.asarray(LABELS)[baseline],
                      "predicted_group": groups}).to_parquet(
            OUT / f"predictions_{method.replace('-', '_')}_full.parquet", index=False)
        print(f"fixed-roster {method} full V0 done: "
              f"BA={result['groupers'][method]['metrics']['balanced_accuracy']:.4%}", flush=True)

    scenario_doc = json.loads(SCENARIOS.read_text())
    id_to_row = {rid: ix for ix, rid in enumerate(ids)}
    sampled_scenarios = scenario_doc["scenario_rows"]
    if len(sampled_scenarios) != 50:
        raise ValueError(f"Expected the established 50 incomplete scenarios, found {len(sampled_scenarios)}")
    for method in methods:
        values, prediction_records = {}, []
        method_started = time.perf_counter()
        for scenario_id, spec in sampled_scenarios.items():
            scenario_ids = list(map(str, spec["row_ids"]))
            if len(set(scenario_ids)) != len(scenario_ids) or not set(scenario_ids) <= set(id_to_row):
                raise ValueError(f"Invalid row IDs in scenario {scenario_id}")
            rows = np.asarray([id_to_row[rid] for rid in scenario_ids], dtype=np.int64)
            groups = _group_and_score(method, rows, context, embeddings, f_c, d_model)
            pred = e20.hungarian(safe_scores[rows], groups, ids[rows])
            values[scenario_id] = {"rows": int(len(rows)),
                "metrics": _metric(truth[rows], pred),
                "baseline_metrics": _metric(truth[rows], baseline[rows]),
                "comparison_vs_fixed_E28_A": _comparison(truth[rows], baseline[rows], pred),
                "grouping_metrics": _grouping_metrics(context["prompt_ids"][rows], groups),
                "group_size_distribution": {str(k): int(v) for k, v in
                    pd.Series(groups).value_counts().value_counts().sort_index().items()}}
            prediction_records.append(pd.DataFrame({"scenario": int(scenario_id),
                "id": ids[rows], "prediction": np.asarray(LABELS)[pred],
                "baseline_prediction": np.asarray(LABELS)[baseline[rows]],
                "predicted_group": groups}))
            if int(scenario_id) % 5 == 0:
                print(f"fixed-roster {method} scenario {scenario_id}/50 done", flush=True)
        result["scenarios"][method] = values
        result["runtime_seconds"][method + "_scenarios"] = time.perf_counter() - method_started
        pd.concat(prediction_records, ignore_index=True).to_parquet(
            OUT / f"predictions_{method.replace('-', '_')}_scenarios.parquet", index=False)

    for method, values in result["scenarios"].items():
        b = [x["metrics"]["balanced_accuracy"] for x in values.values()]
        base = [x["baseline_metrics"]["balanced_accuracy"] for x in values.values()]
        result.setdefault("scenario_summary", {})[method] = {
            "mean_balanced_accuracy": float(np.mean(b)), "std_balanced_accuracy": float(np.std(b, ddof=1)),
            "min_balanced_accuracy": float(np.min(b)), "max_balanced_accuracy": float(np.max(b)),
            "mean_baseline_balanced_accuracy": float(np.mean(base)),
            "aggregate_corrected": int(sum(x["comparison_vs_fixed_E28_A"]["corrected"] for x in values.values())),
            "aggregate_introduced": int(sum(x["comparison_vs_fixed_E28_A"]["introduced"] for x in values.values()))}
    result["runtime_seconds"]["total"] = time.perf_counter() - started
    result["v0_row_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    summary = run()
    print(json.dumps({"experiment": summary["experiment"], "baseline": summary["baseline"],
        "groupers": {k: {"metrics": v["metrics"],
            "comparison_vs_fixed_E28_A": v["comparison_vs_fixed_E28_A"]}
            for k, v in summary["groupers"].items()},
        "scenario_summary": summary["scenario_summary"],
        "runtime_seconds": summary["runtime_seconds"]}, indent=2))
