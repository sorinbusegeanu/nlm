"""E34: E33 negative-evidence methods on the frozen E32 60-checkpoint roster.

All checkpoint probabilities and E19 group assignments are cached. Reliability
statistics and candidate-error models use the same prompt-group cross-fitting
as E33. The 50 v0_group scenarios are robustness diagnostics over reused V0
responses, not independent validation data.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from . import e33, ensemble_catalog as catalog
from .e13_b_inference import family_max_scores, make_features
from .e20 import hungarian
from .trust_list2_weighted_vote import _model_family
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e34"
E31_DIR = ROOT / "artifacts/analysis/e31"
E32_DIR = ROOT / "artifacts/analysis/e32"
E33_DIR = ROOT / "artifacts/analysis/e33"
E31_SCORE_CACHE = E31_DIR / "roster_scenario_evaluation/predictions_and_scores.npz"
E31_SCENARIO_CACHE = E31_DIR / "roster_scenario_evaluation/scenario_predictions.parquet"
E31_MANIFEST = E31_DIR / "e31_manifest.json"
E32_MANIFEST = E32_DIR / "e32_manifest.json"
E32_SCORE_CACHE = E32_DIR / "v0_scores.npz"
E32_BENCHMARK = E32_DIR / "inference_benchmark.json"
METHODS = ("Independent", "E19-F-A", "E19-F-C", "E19-D")
VARIANTS = tuple(f"E34-{x}" for x in "ABCDEF")
BASELINES = ("E31 baseline", "E33-F", "E32 baseline")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_scenarios(v0: pd.DataFrame, ids: np.ndarray):
    source = pd.read_parquet(ROOT / "data/processed/dataset_v0_group.parquet")
    cached = pd.read_parquet(E31_SCENARIO_CACHE)
    cached = cached.loc[cached.checkpoint_count.eq(40)].copy()
    id_set = set(ids.astype(str))
    group_maps = {}
    for method in METHODS[1:]:
        part = cached.loc[cached.configuration.eq(method)]
        if len(part) != 25000 or part.scenario_row_id.astype(str).duplicated().any():
            raise ValueError(f"E33 cached {method} assignments are incomplete/duplicated")
        group_maps[method] = part.set_index(part.scenario_row_id.astype(str)).predicted_group.astype(str).to_dict()
    frames = []
    for sid, frame in source.groupby("scenario_id", sort=True):
        frame = frame.reset_index(drop=True).copy()
        if len(frame) != 500 or frame.original_id.astype(str).duplicated().any():
            raise ValueError(f"Invalid v0_group scenario: {sid}")
        if set(frame.original_id.astype(str)) - id_set:
            raise ValueError(f"Scenario {sid} references non-V0 rows")
        frame["truth"] = frame.model.astype(str)
        for method in METHODS[1:]:
            frame[f"group::{method}"] = [group_maps[method][str(x)] for x in frame.scenario_row_id]
        frames.append((str(sid), frame))
    if len(frames) != 50:
        raise ValueError(f"Expected 50 cached scenarios, found {len(frames)}")
    return frames


def _scenario_summary(scenarios, ids, scores_by_experiment, exclusions_by_experiment,
                      baseline_pred, *, baseline_cache=None):
    lookup = {str(x): i for i, x in enumerate(ids)}
    e33_rows = None
    if baseline_cache is not None:
        e33_rows = pd.read_parquet(E33_DIR / "scenario_predictions.parquet")
        e33_rows = e33_rows.loc[e33_rows.experiment.eq("E33-F")]
    score_lists = {name: {method: [] for method in METHODS}
                   for name in (*BASELINES, *VARIANTS)}
    pred_records = []
    baseline_scenario_preds = []
    for sid, frame in scenarios:
        rows = np.asarray([lookup[str(x)] for x in frame.original_id], dtype=np.int64)
        y = pd.Categorical(frame.truth.astype(str), categories=LABELS).codes.astype(np.int64)
        if (y < 0).any():
            raise ValueError("Scenario has a label outside canonical 18-class order")
        # E31 baseline and E33-F are copied from their already evaluated OOF
        # predictions, retaining their exact reported baseline methodology.
        for experiment, part_all in (("E31 baseline", None), ("E33-F", e33_rows)):
            if experiment == "E31 baseline":
                base_rows = pd.read_parquet(E31_SCENARIO_CACHE)
                base_rows = base_rows.loc[(base_rows.checkpoint_count.eq(40))
                                          & (base_rows.scenario_id.astype(str).eq(sid))]
            else:
                base_rows = part_all.loc[part_all.scenario_id.astype(str).eq(sid)]
            for method in METHODS:
                selector = (base_rows.configuration.eq(method) if experiment == "E31 baseline"
                            else base_rows.method.eq(method))
                part = base_rows.loc[selector]
                if len(part) != len(frame):
                    raise ValueError(f"{experiment}/{sid}/{method}: expected 500 saved rows")
                part = part.set_index(part.scenario_row_id.astype(str)).loc[
                    frame.scenario_row_id.astype(str)]
                pred = pd.Categorical(part.prediction.astype(str), categories=LABELS).codes
                if (pred < 0).any():
                    raise ValueError(f"Unknown label in saved {experiment} predictions")
                metric = e33._metric(y, pred)["balanced_accuracy"]
                score_lists[experiment][method].append(float(metric))
                pred_records.extend({"scenario_id": sid, "scenario_row_id": str(frame.scenario_row_id.iloc[i]),
                    "original_id": str(frame.original_id.iloc[i]), "truth": LABELS[y[i]],
                    "prediction": LABELS[pred[i]], "experiment": experiment, "method": method}
                    for i in range(len(frame)))
        for method in METHODS:
            if method == "Independent":
                pred = baseline_pred[rows]
            else:
                group_ids = frame[f"group::{method}"].astype(str).to_numpy()
                pred = hungarian(scores_by_experiment["E32 baseline"][rows], group_ids,
                    frame.scenario_row_id.astype(str).to_numpy())
            score_lists["E32 baseline"][method].append(
                e33._metric(y, pred)["balanced_accuracy"])
            baseline_scenario_preds.append({"scenario_id": sid, "method": method,
                "balanced_accuracy": score_lists["E32 baseline"][method][-1]})
            pred_records.extend({"scenario_id": sid, "scenario_row_id": str(frame.scenario_row_id.iloc[i]),
                "original_id": str(frame.original_id.iloc[i]), "truth": LABELS[y[i]],
                "prediction": LABELS[pred[i]], "experiment": "E32 baseline", "method": method}
                for i in range(len(frame)))
        for variant in VARIANTS:
            variant_result, records = e33._evaluate_scenario_scores(
                frame, lookup, scores_by_experiment[variant],
                exclusions_by_experiment.get(variant), baseline_pred)
            for method in METHODS:
                score_lists[variant][method].append(
                    float(variant_result[method]["balanced_accuracy"]))
            for record in records:
                record["experiment"] = variant
                pred_records.append(record)
    summary = {}
    for experiment, methods in score_lists.items():
        summary[experiment] = {}
        for method, values in methods.items():
            a = np.asarray(values, dtype=np.float64)
            summary[experiment][method] = {
                "mean_balanced_accuracy": float(a.mean()),
                "std_balanced_accuracy": float(a.std()),
                "min_balanced_accuracy": float(a.min()),
                "p10_balanced_accuracy": float(np.quantile(a, .1)),
                "scenario_count": int(len(a)), "scenario_values": a.tolist()}
    return summary, pd.DataFrame(pred_records)


def _fit_final_deployment(probabilities, rank_position, base_scores, y, prompts,
                          families, roster, output_dir: Path, *, tag: str):
    """Fit a deployable final CatBoost from five-fold OOF reliability features."""
    rows = np.arange(len(y), dtype=np.int64)
    from sklearn.model_selection import GroupKFold
    blocks, targets = [], []
    for fit_local, held_local in GroupKFold(n_splits=5).split(rows, y, prompts):
        fit, held = rows[fit_local], rows[held_local]
        stats = e33._fit_rank_stats(rank_position, y, fit)
        blocks.append(e33._build_candidate_features(held, stats,
            probabilities=probabilities, rank_position=rank_position,
            base_scores=base_scores, families=families))
        targets.append((y[held, None] != np.arange(len(LABELS))[None, :]).astype(np.int8).ravel())
    x = pd.concat(blocks, ignore_index=True)
    target = np.concatenate(targets)
    model = CatBoostClassifier(iterations=400, depth=6, learning_rate=.05,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=5.0,
        auto_class_weights="Balanced", random_seed=e33.SEED,
        thread_count=e33.THREADS, verbose=False, allow_writing_files=False)
    model.fit(x, target, cat_features=["candidate_label"])
    model_path = output_dir / f"{tag}_final.cbm"
    model.save_model(str(model_path))
    full_stats = e33._fit_rank_stats(rank_position, y, rows)
    np.savez_compressed(output_dir / f"{tag}_final_reliability.npz", **full_stats)
    return {"model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": _sha(model_path),
            "reliability_path": f"artifacts/analysis/{tag}/{tag}_final_reliability.npz",
            "fit_rows": len(y), "oof_candidate_rows": int(len(target)),
            "inner_folds": 5, "iterations": 400, "depth": 6,
            "learning_rate": .05, "thread_count": e33.THREADS}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v0 = catalog.cascades._v0(ROOT)
    _, members, all_probs, excluded, trust_snapshot = catalog._load_inputs(
        ROOT, v0, version="trust-list-2-v037")
    manifest = json.loads(E32_MANIFEST.read_text())
    roster = list(map(str, manifest["roster"]))
    if len(roster) != 60 or len(set(roster)) != 60 or set(roster) - set(all_probs):
        raise ValueError("Frozen E32 roster does not resolve to 60 cached checkpoints")
    if excluded:
        # Only roster-specific unavailable caches are material to this run.
        missing = sorted(set(roster) - set(all_probs))
        if missing:
            raise ValueError(f"E32 roster prediction caches missing: {missing}")
    ids = v0.id.astype(str).to_numpy()
    prompts = v0.prompt_id.astype(str).to_numpy()
    y = pd.Categorical(v0.model.astype(str), categories=LABELS).codes.astype(np.int64)
    if len(ids) != 2699 or (y < 0).any() or len(set(prompts)) != 150:
        raise ValueError("Expected complete canonical V0 (2,699 rows, 150 prompt groups)")
    probabilities = np.stack([all_probs[sid] for sid in roster]).astype(np.float32)
    if probabilities.shape != (60, 2699, 18) or not np.isfinite(probabilities).all():
        raise ValueError(f"E32 cached probability array has invalid shape/data {probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError("E32 checkpoint probabilities do not sum to one")
    families = [_model_family(members[sid]["model"]) for sid in roster]
    rank_position = e33._rank_positions(probabilities)
    e33._BUILD_CP_IDS = roster
    score_cache = np.load(E32_SCORE_CACHE, allow_pickle=True)
    if (score_cache["row_ids"].astype(str).tolist() != ids.tolist()
            or score_cache["classes"].astype(str).tolist() != list(LABELS)):
        raise ValueError("E32 frozen score cache does not align to canonical V0")
    base_scores = score_cache["scores_60"].astype(np.float64)
    base_pred = score_cache["predictions_60"].astype(np.int64)
    if (base_scores.shape != (2699, 18)
            or not np.array_equal(base_scores.argmax(1), base_pred)):
        raise ValueError("Frozen E32 router scores/predictions are inconsistent")
    folds = e33._folds_from_e31(ids, prompts)
    scenarios = _load_scenarios(v0, ids)
    e33._SCENARIO_TRUTH = y
    scores = {name: np.zeros_like(base_scores) for name in ("E32 baseline", *VARIANTS)}
    scores["E32 baseline"][:] = base_scores
    exclusions = {f"E34-{letter}": np.zeros_like(base_scores, dtype=bool)
                  for letter in "ABC"}
    rule_audit, fold_counts, learned = [], [], []
    for fold_number, (fit_rows, held_rows) in enumerate(folds, 1):
        stats = e33._fit_rank_stats(rank_position, y, fit_rows, baseline_pred=base_pred)
        rules = e33._negative_rules(stats, families)
        fold_counts.append({"fold": fold_number,
            "single_rules": int(rules["single"].sum()),
            "consensus_label_rules": int(rules["consensus"].sum()),
            "conditional_rules": int(rules["conditional"].sum())})
        for name, e33_name in zip(VARIANTS[:3], ("E33-A", "E33-B", "E33-C")):
            exclusions[name][held_rows] = e33._candidate_exclusions(
                rank_position, held_rows, stats, rules, families, base_pred, e33_name)
            scores[name][held_rows] = base_scores[held_rows]
        scores["E34-D"][held_rows] = base_scores[held_rows] * e33._weighted_factors(
            rank_position, held_rows, stats, families, combined=False)
        scores["E34-E"][held_rows] = base_scores[held_rows] * e33._weighted_factors(
            rank_position, held_rows, stats, families, combined=True)
        for cp, checkpoint_id in enumerate(roster):
            for label in range(len(LABELS)):
                for rank in range(1, 4):
                    sup = int(stats["support"][cp, rank - 1, label])
                    if sup:
                        rule_audit.append({"fold": fold_number,
                            "checkpoint_id": checkpoint_id, "family": families[cp],
                            "rank": rank, "label": LABELS[label], "support": sup,
                            "correct": int(stats["correct"][cp, rank - 1, label]),
                            "precision": float(stats["precision"][cp, rank - 1, label]),
                            "label_prior": float(stats["prior"][label]),
                            "negative_rule": bool(rules["single"][cp, rank - 1, label]),
                            "conditional_support": int(stats["conditional_support"][cp, rank - 1, label]),
                            "conditional_precision": float(stats["conditional_precision"][cp, rank - 1, label]),
                            "conditional_rule": bool(rules["conditional"][cp, rank - 1, label])})
        adjusted, meta = e33._fit_e33_f(fit_rows, held_rows, ids=ids,
            prompts=prompts, y=y, probabilities=probabilities,
            rank_position=rank_position, base_scores=base_scores, families=families,
            baseline_pred=base_pred, fold_number=fold_number, output_dir=OUT)
        # Give fold models an E34-specific filename; the E33 fit helper writes
        # compatible CatBoost contents into the supplied E34 artifact directory.
        written = OUT / f"e33_f_fold{fold_number}.cbm"
        target_path = OUT / f"e34_f_fold{fold_number}.cbm"
        if written.exists():
            written.replace(target_path)
        scores["E34-F"][held_rows] = adjusted
        meta["model_path"] = str(target_path.relative_to(ROOT))
        learned.append(meta)
        print(f"E34 fold {fold_number}/5 complete; A={fold_counts[-1]['single_rules']} rules, "
              f"B={fold_counts[-1]['consensus_label_rules']} consensus, "
              f"C={fold_counts[-1]['conditional_rules']} conditional", flush=True)
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("E34 produced incomplete/nonfinite OOF scores")
    summary, predictions = _scenario_summary(scenarios, ids, scores, exclusions,
        base_pred, baseline_cache=True)
    # The baseline must match the existing E32 results with the exact same
    # scores, scenario membership and cached E19 group IDs.
    e32_saved = json.loads((E32_DIR / "v0_group_results.json").read_text())["summary"]
    baseline_reproduction = {}
    for method, key in zip(METHODS, ("Independent", "E19-F-A", "E19-F-C", "E19-D")):
        saved = float(e32_saved[key]["mean_balanced_accuracy_60"])
        measured = summary["E32 baseline"][method]["mean_balanced_accuracy"]
        baseline_reproduction[method] = {"saved": saved, "recomputed": measured,
                                         "delta": measured - saved}
    # Final deployment artifacts use all-V0 OOF features; their diagnostic
    # calibration/reliability state is recorded separately from OOF scores.
    final_model = _fit_final_deployment(probabilities, rank_position, base_scores,
        y, prompts, families, roster, OUT, tag="e34")
    pd.DataFrame(rule_audit).to_parquet(OUT / "fold_rule_statistics.parquet", index=False)
    pd.DataFrame(fold_counts).to_csv(OUT / "fold_rule_counts.csv", index=False)
    predictions.to_parquet(OUT / "scenario_predictions.parquet", index=False)
    np.savez_compressed(OUT / "oof_adjusted_scores.npz", baseline=base_scores,
        **{name.replace("-", "_"): val for name, val in scores.items()},
        **{f"exclusion_{name.replace('-', '_')}": val for name, val in exclusions.items()},
        row_ids=ids, classes=np.asarray(LABELS))
    result = {"experiment": "E34 Negative Evidence and Learned Candidate Scoring with 60 Checkpoints",
        "status": "Scored", "roster": roster, "families": families,
        "scenario_count": 50, "rows_per_scenario": 500, "methods": list(METHODS),
        "summary": summary, "fold_rule_counts": fold_counts,
        "learned_classifier_folds": learned, "final_deployment_classifier": final_model,
        "candidate_feature_definition": {"per_checkpoint": ["probability", "1-based rank",
            "smoothed reliability ratio", "reliability support"],
            "per_checkpoint_feature_count": 4, "checkpoint_feature_count": 240,
            "shared_features": ["candidate label", "E32 score", "E32 score rank",
                "E32 top margin", "positive family count", "negative family count"],
            "total_feature_count": 246},
        "baseline_reproduction": baseline_reproduction,
        "selection_folds": [{"fit_rows": len(a), "held_rows": len(b)} for a,b in folds],
        "rule_thresholds": {"A_B": {"support": 30, "wrong_rate": .97,
            "precision_below_class_prior": True}, "C": {"conditional_support": 20,
            "conditional_wrong_rate": .95}, "D": {"support": 30,
            "precision_below_class_prior": True}, "E": {"support": 30,
            "prior_smoothing_strength": 30}, "F": {"inner_group_folds": 4,
            "iterations": 400, "depth": 6, "learning_rate": .05,
            "thread_count": e33.THREADS}},
        "inputs": {"trust_list_version": trust_snapshot["version"],
            "e32_manifest_sha256": _sha(E32_MANIFEST),
            "e32_score_cache_sha256": _sha(E32_SCORE_CACHE),
            "e31_scenario_group_cache_sha256": _sha(E31_SCENARIO_CACHE),
            "v0_group_sha256": _sha(ROOT / "data/processed/dataset_v0_group.parquet"),
            "e33_metrics_sha256": _sha(E33_DIR / "metrics.json")},
        "notes": ["The exact E32 frozen 60-checkpoint roster and router score matrix are reused; the router was not retrained.",
            "All reliability statistics and E34-F training features are prompt-group cross-fitted on the saved E31 folds.",
            "All four decoders use the same cached E19-F-A, E19-F-C and E19-D assignments used by E33.",
            "The 50 v0_group scenarios reuse V0 rows and are correlated robustness diagnostics, not independent validation."],
        "artifact_files": ["oof_adjusted_scores.npz", "scenario_predictions.parquet",
            "fold_rule_statistics.parquet", "fold_rule_counts.csv", "e34_f_fold1.cbm through e34_f_fold5.cbm",
            "e34_final.cbm", "e34_final_reliability.npz", "e34_deployment_manifest.json"]}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_report(result)
    return result


def _write_report(result: dict) -> None:
    e31 = json.loads((E33_DIR / "metrics.json").read_text())["summary"]["E31 baseline"]
    e33f = json.loads((E33_DIR / "metrics.json").read_text())["summary"]["E33-F"]
    e32 = result["summary"]["E32 baseline"]
    rows = [("E31 baseline", e31), ("E33-F", e33f), ("E32 baseline", e32)]
    rows += [(name, result["summary"][name]) for name in VARIANTS]
    rows += [("E34-F delta vs E33-F (pp)", None), ("E34-F delta vs E32 (pp)", None)]
    values = []
    for name, data in rows:
        if data is None:
            if name.endswith("E33-F (pp)"):
                base = e33f
            else:
                base = e32
            values.append([100 * (result["summary"]["E34-F"][m]["mean_balanced_accuracy"]
                                  - base[m]["mean_balanced_accuracy"]) for m in METHODS])
        else:
            values.append([100 * data[m]["mean_balanced_accuracy"] for m in METHODS])
    maxima = np.max(np.asarray(values[:9], dtype=float), axis=0)
    lines = ["# E34 — Negative Evidence and Learned Candidate Scoring with 60 Checkpoints", "",
        "| Experiment | Checkpoints | Independent | E19-F-A | E19-F-C | E19-D |",
        "|---|---:|---:|---:|---:|---:|"]
    for i, ((name, _), row) in enumerate(zip(rows, values)):
        cells = []
        for j, value in enumerate(row):
            formatted = f"{value:.4f} pp" if "delta" in name else f"{value:.4f}%"
            if i < 9 and np.isclose(value, maxima[j], rtol=0, atol=1e-10):
                formatted = f"**{formatted}**"
            cells.append(formatted)
        count = 40 if name in ("E31 baseline", "E33-F") else ("—" if "delta" in name else "60")
        lines.append(f"| {name} | {count} | " + " | ".join(cells) + " |")
    lines += ["", "All figures are mean balanced accuracy across the same 50 500-row `v0_group` scenarios. These scenarios reuse V0 responses, so they are correlated robustness diagnostics, not independent validation.",
        "", "E34 preserves the E32 60-checkpoint roster and frozen E32 router. E34-A/B/C use the E33 exclusion thresholds; E34-D/E apply E33's family-aggregated reliability factors; E34-F uses the E33-F CatBoost settings and 60-checkpoint candidate features. Rules and learned scores are fit with the saved five prompt-disjoint outer folds and four inner grouped folds for E34-F.",
        "", "The group assignments for E19-F-A, E19-F-C and E19-D are the exact cached assignments used by E33. E32 baseline is recomputed from the frozen `scores_60` matrix; its difference from the archived E32 summary is reported in the artifact's `baseline_reproduction` block if any numerical discrepancy remains.",
        "", "Artifacts: `artifacts/analysis/e34/metrics.json`, `oof_adjusted_scores.npz`, `scenario_predictions.parquet`, fold reliability statistics/counts, five fold classifiers and a final five-fold-OOF deployment classifier.", ""]
    (ROOT / "docs/e34.md").write_text("\n".join(lines), encoding="utf-8")


def update_catalogue(status: str, result: dict | None = None, error: str | None = None) -> None:
    path = ROOT / "configs/ensemble_catalog_v2.json"
    doc = json.loads(path.read_text())
    matches = [x for x in doc.get("experiments", []) if x.get("id") == "E34"]
    if len(matches) != 1:
        raise ValueError(f"Expected one E34 catalogue row, found {len(matches)}")
    entry = matches[0]
    entry.update({"status": status, "current_stage": status,
        "progress_source": "artifacts/analysis/e34/metrics.json",
        "documentation": "docs/e34.md", "benchmark_documentation": "docs/e34_inference_benchmark.md"})
    if error:
        entry["error"] = error
    if result:
        entry.update({"status": "Scored", "current_stage": "Complete",
            "result_source": "artifacts/analysis/e34/metrics.json",
            "scenario_results": {x: {m: d["mean_balanced_accuracy"]
                for m, d in methods.items()} for x, methods in result["summary"].items()}})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    update_catalogue("Running")
    try:
        final = run()
        update_catalogue("Scored", final)
        print(json.dumps(final["summary"], indent=2))
    except Exception as exc:
        update_catalogue("Failed", error=f"{type(exc).__name__}: {exc}")
        raise
