"""Cache-only comparison of the November 15 Trust List 2 ensemble hypotheses."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ensemble_factory import cross_fitted_weights
from src.pipeline import _dataset_rows
from src.v2_checkpoint_trust import LABELS, _read_saved_probabilities
from src.trust_list2_weighted_vote import _model_family

OUT = ROOT / "artifacts/analysis/ensemble_hypotheses_nov15"
REPORT = ROOT / "docs/ensemble_hypotheses_nov15.md"
TRUST_POINTER = ROOT / "docs/trust-list-2-latest.json"
INVENTORY_NAME = "2026-11-15T06-10_export.csv"
WEIGHTED5_WEIGHTS = {
    "N0071": 0.2811918866145629,
    "N0056": 0.23247150955573173,
    "N0005": 0.18335267365939653,
    "N0058": 0.23454557290930692,
    "N0072": 0.0684383572610019,
}
HISTORICAL_EQUAL10 = ["N0071", "N0056", "N0005", "N0058", "N0072",
                      "N0030", "N0061", "N0044", "N0009", "N0048"]
HISTORICAL_EQUAL9 = ["N0071", "N0005", "N0058", "N0072", "N0030",
                     "N0061", "N0044", "N0009", "N0048"]
HISTORICAL_EQUAL10_QWEN = HISTORICAL_EQUAL9 + ["N0074"]

SCENARIOS = {
    "H01": ["N0081", "N0071", "N0058", "N0074", "N0005"],
    "H02": ["N0081", "N0071", "N0058", "N0074", "N0005", "N0030"],
    "H03": ["N0081", "N0071", "N0058", "N0074", "N0005", "N0030", "N0056"],
    "H04": ["N0081", "N0071", "N0058", "N0074", "N0005", "N0030", "N0056", "N0061"],
    "H05": ["N0081", "N0071", "N0058", "N0074", "N0005", "N0030", "N0056", "N0061", "N0057"],
    "H06": ["N0081", "N0071", "N0058", "N0074", "N0005", "N0030", "N0056", "N0061", "N0057", "N0060"],
    "H07": ["N0081", "N0071", "N0074", "N0080", "N0076", "N0058", "N0030", "N0005"],
    "H08": ["N0081", "N0071", "N0058", "N0005", "N0030", "N0056", "N0061", "N0065"],
    "H09": ["N0071", "N0081", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"],
    "H10": ["N0071", "N0056", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0081"],
}
FOCUS_LABELS = ["claude-fable-5", "gemini-3.7-flash", "glm-5.3",
                "glm-5.3-flash", "gpt-5.6-sol"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def blend(ids: list[str], matrices: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    w = np.asarray([weights[x] for x in ids], dtype=np.float64)
    if not np.isclose(w.sum(), 1.0, atol=1e-10) or (w < 0).any():
        raise ValueError(f"invalid weights for {ids}: {w}")
    result = np.einsum("m,mnc->nc", w, np.stack([matrices[x] for x in ids]))
    result /= result.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-5):
        raise ValueError("ensemble produced invalid probabilities")
    return result


def score(name: str, scenario: str, strategy: str, ids: list[str], weights: dict[str, float],
          p: np.ndarray, y: np.ndarray, baseline_pred: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    p /= p.sum(axis=1, keepdims=True)
    pred = np.asarray(LABELS, dtype=object)[p.argmax(axis=1)]
    accuracy = float(accuracy_score(y, pred))
    balanced = float(balanced_accuracy_score(y, pred))
    precision, recall, _, support = precision_recall_fscore_support(
        y, pred, labels=list(LABELS), zero_division=0)
    base_precision, base_recall, _, _ = precision_recall_fscore_support(
        y, baseline_pred, labels=list(LABELS), zero_division=0)
    corrected = int(((baseline_pred != y) & (pred == y)).sum())
    introduced = int(((baseline_pred == y) & (pred != y)).sum())
    row = {
        "name": name, "scenario": scenario, "strategy": strategy,
        "checkpoint_count": len(ids), "checkpoint_ids": ";".join(ids),
        "weights_json": json.dumps(weights, sort_keys=True),
        "accuracy": accuracy, "balanced_accuracy": balanced,
        "accuracy_delta_vs_historical_equal10_pp": (accuracy - float(accuracy_score(y, baseline_pred))) * 100,
        "balanced_accuracy_delta_vs_historical_equal10_pp": (balanced - float(balanced_accuracy_score(y, baseline_pred))) * 100,
        "correct": int((pred == y).sum()), "corrected_vs_historical_equal10": corrected,
        "correct_predictions_lost_vs_historical_equal10": introduced,
        "log_loss": float(log_loss(y, p, labels=list(LABELS))),
    }
    label_rows = []
    for i, label in enumerate(LABELS):
        mask = y == label
        tp = int(((pred == label) & mask).sum())
        predicted_count = int((pred == label).sum())
        label_rows.append({
            "name": name, "scenario": scenario, "strategy": strategy, "label": label,
            "support": int(support[i]), "predicted_count": predicted_count,
            "correct": tp, "precision": float(precision[i]), "recall": float(recall[i]),
            "correct_delta_vs_historical_equal10": tp - int(((baseline_pred == label) & mask).sum()),
            "precision_delta_vs_historical_equal10_pp": float((precision[i] - base_precision[i]) * 100),
            "recall_delta_vs_historical_equal10_pp": float((recall[i] - base_recall[i]) * 100),
        })
    return row, label_rows, pred


def render_report(result: dict) -> str:
    rows = result["ranked_fixed"]
    hist = result["historical"]
    best = result["optimized"]
    lines = [
        "# November 15 ensemble hypotheses", "",
        "Cache-only comparison on all 2,699 official V0 rows. No model was loaded; no inference, model training, or Kaggle submission was run.", "",
        f"ID resolution used [{result['trust_list_version']}](../{result['trust_list_csv']}) because `{INVENTORY_NAME}` was not found in the local workspace. Prediction files were aligned by row ID, true label, and the canonical 18-label order. Cache files and hashes are recorded in `artifacts/analysis/ensemble_hypotheses_nov15/metadata.json`.", "",
        "Full-V0 results are exploratory: V0 was used for checkpoint selection and previous trust-list/ensemble development. Cross-fitted optimized scores use five GroupKFold folds with prompt IDs kept intact; report them separately from the full-V0 fixed-weight scores.", "",
        "## Historical baselines", "",
        "| Ensemble | Members | V0 accuracy | V0 balanced accuracy | Kaggle public |", "|---|---:|---:|---:|---:|",
    ]
    for x in hist:
        lines.append(f"| {x['name']} | {x['checkpoint_count']} | {x['accuracy']:.4%} | {x['balanced_accuracy']:.4%} | {x['kaggle_public']} |")
    lines += ["", "Historical EQUAL10 is the reference for the scenario deltas (64.2409% balanced accuracy). EQUAL9 and EQUAL10-Qwen had public scores of 0.567; WEIGHTED5 had 0.564. Those public scores are treated as practically equivalent.", "",
              "## Fixed-weight scenarios", "",
              "Ranked by full-V0 balanced accuracy. Corrected/errors-lost compare each row’s predictions against historical EQUAL10.", "",
              "| Rank | Scenario | Method | Members | Accuracy | Balanced acc. | Δ bal. acc. | Corrected | Correct lost | Log loss |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, x in enumerate(rows, 1):
        lines.append(f"| {rank} | {x['scenario']} | {x['strategy']} | {x['checkpoint_count']} | {x['accuracy']:.4%} | {x['balanced_accuracy']:.4%} | {x['balanced_accuracy_delta_vs_historical_equal10_pp']:+.3f} pp | {x['corrected_vs_historical_equal10']} | {x['correct_predictions_lost_vs_historical_equal10']} | {x['log_loss']:.5f} |")
    lines += ["", "### Progressive additions H01–H06", "", "Accuracy change from each preceding scenario (H02−H01 through H06−H05), separately by fixed weighting method:", "",
              "| Step | Checkpoint added | Equal accuracy Δ | Family-equal accuracy Δ | Equal balanced-accuracy Δ | Family-equal balanced-accuracy Δ |", "|---|---|---:|---:|---:|---:|"]
    by_key = {(x["scenario"], x["strategy"]): x for x in rows}
    for n in range(2, 7):
        child, parent = f"H{n:02d}", f"H{n-1:02d}"
        added = list(set(SCENARIOS[child]) - set(SCENARIOS[parent]))
        added_id = added[0] if len(added) == 1 else ",".join(added)
        e0, e1 = by_key[(parent, "EQUAL")], by_key[(child, "EQUAL")]
        f0, f1 = by_key[(parent, "FAMILY-EQUAL")], by_key[(child, "FAMILY-EQUAL")]
        lines.append(f"| {parent} → {child} | {added_id} | {(e1['accuracy']-e0['accuracy'])*100:+.3f} pp | {(f1['accuracy']-f0['accuracy'])*100:+.3f} pp | {(e1['balanced_accuracy']-e0['balanced_accuracy'])*100:+.3f} pp | {(f1['balanced_accuracy']-f0['balanced_accuracy'])*100:+.3f} pp |")
    lines += ["", "### H09–H10 versus historical EQUAL10", "", "| Scenario | Method | Accuracy Δ | Balanced accuracy Δ | Corrected | Correct lost |", "|---|---|---:|---:|---:|---:|"]
    for scenario in ("H09", "H10"):
        for method in ("EQUAL", "FAMILY-EQUAL"):
            x = by_key[(scenario, method)]
            lines.append(f"| {scenario} | {method} | {x['accuracy_delta_vs_historical_equal10_pp']:+.3f} pp | {x['balanced_accuracy_delta_vs_historical_equal10_pp']:+.3f} pp | {x['corrected_vs_historical_equal10']} | {x['correct_predictions_lost_vs_historical_equal10']} |")
    lines += ["", "## Cross-fitted optimized weights", "", "The three highest-ranked distinct checkpoint sets among the 20 fixed-weight scenario/method results were selected. Nonnegative weights were refit with the existing `fit_weights` optimizer (`l2=0.05`) on the other four GroupKFold prompt folds, then applied to each held-out fold. These are out-of-fold predictions, not full-fit resubstitution scores.", "",
              "| Scenario | Best fixed method | Members | OOF accuracy | OOF balanced acc. | OOF log loss | Weights by fold |", "|---|---|---:|---:|---:|---:|---|"]
    for x in best:
        lines.append(f"| {x['scenario']} | {x['selected_fixed_method']} | {x['checkpoint_count']} | {x['accuracy']:.4%} | {x['balanced_accuracy']:.4%} | {x['log_loss']:.5f} | `artifacts/analysis/ensemble_hypotheses_nov15/optimized_weights.json` |")
    lines += ["", "## Focus labels", "", "Precision and recall are from the fixed-weight configurations and the three cross-fitted optimized configurations. Deltas compare with historical EQUAL10. Full per-label metrics for every method, including all H01–H10 variants, are in `artifacts/analysis/ensemble_hypotheses_nov15/per_label_metrics.csv`.", ""]
    focus = result["focus_metrics"]
    lines += ["| Configuration | Label | Precision | Recall | Δ recall | Δ correct | Support |", "|---|---|---:|---:|---:|---:|---:|"]
    for x in focus:
        lines.append(f"| {x['name']} | {x['label']} | {x['precision']:.1%} | {x['recall']:.1%} | {x['recall_delta_vs_historical_equal10_pp']:+.1f} pp | {x['correct_delta_vs_historical_equal10']:+d} | {x['support']} |")
    specialist = result["specialist_recall"]
    lines += ["", "## Specialist check: N0065", "",
              f"N0065 alone recalls `gemini-3.7-flash` at {specialist['N0065_gemini37']:.1%}; N0081 alone recalls it at {specialist['N0081_gemini37']:.1%}. H08's equal and family-equal blends recall it at {specialist['H08_EQUAL_gemini37']:.1%} and {specialist['H08_FAMILY_EQUAL_gemini37']:.1%}. The ensemble dilutes most of N0065's specialist recall.", "",
              "## Recommendation", ""]
    lines.extend(f"{i}. **{x['recommendation']}** — {x['reason']}" for i, x in enumerate(result["recommendations"], 1))
    lines += ["", "Recommendations prioritize balanced accuracy, then cross-fitted behavior and label-specific regressions. Validate on a genuinely unseen set or Kaggle before treating small V0 differences as gains.", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trust_pointer = json.loads(TRUST_POINTER.read_text())
    trust_csv = ROOT / trust_pointer["csv_path"]
    trust = pd.read_csv(trust_csv, dtype=str, keep_default_na=False)
    v0 = _dataset_rows("v0", ROOT / "data/processed")
    if len(v0) != 2699 or v0.id.astype(str).duplicated().any():
        raise ValueError("V0 must contain exactly 2,699 unique response IDs")
    ids = v0.id.astype(str).tolist()
    y = v0.model.astype(str).to_numpy(dtype=object)
    prompts = v0.prompt_id.astype(str).to_numpy()
    label_order = list(LABELS)
    if set(y) != set(label_order) or len(label_order) != 18:
        raise ValueError("V0 labels differ from the canonical 18-label vocabulary")
    if trust.target_label.nunique() != 18:
        raise ValueError("Trust List 2 does not contain the canonical 18 labels")

    all_ids = sorted({sid for members in SCENARIOS.values() for sid in members} |
                     set(HISTORICAL_EQUAL10) | set(HISTORICAL_EQUAL9) |
                     set(HISTORICAL_EQUAL10_QWEN) | set(WEIGHTED5_WEIGHTS))
    metadata_by_id, matrices, sources = {}, {}, {}
    for sid in all_ids:
        matches = trust.loc[trust.sequence_id.eq(sid)]
        if matches.empty or matches.experiment_id.nunique() != 1 or matches.model.nunique() != 1:
            raise ValueError(f"{sid} does not resolve uniquely in {trust_csv}")
        experiment_id = str(matches.experiment_id.iloc[0])
        probabilities, cache_path = _read_saved_probabilities(ROOT, experiment_id, ids, y.tolist())
        if probabilities.shape != (len(v0), len(label_order)):
            raise ValueError(f"{sid}: probability shape {probabilities.shape} is invalid")
        matrices[sid] = probabilities
        metadata_by_id[sid] = {"sequence_id": sid, "experiment_id": experiment_id,
                               "model": str(matches.model.iloc[0]),
                               "training_dataset": str(matches.training_dataset.iloc[0]),
                               "checkpoint_path": str(matches.checkpoint_path.iloc[0])}
        sources[sid] = {"path": str(cache_path.relative_to(ROOT)), "sha256": sha256(cache_path)}

    # Historical equal-10 is both a required reproduction and the error-comparison baseline.
    base_p = blend(HISTORICAL_EQUAL10, matrices,
                   {sid: 1 / len(HISTORICAL_EQUAL10) for sid in HISTORICAL_EQUAL10})
    baseline_pred = np.asarray(label_order, dtype=object)[base_p.argmax(axis=1)]
    base_acc = float(accuracy_score(y, baseline_pred))
    base_bal = float(balanced_accuracy_score(y, baseline_pred))
    if int((baseline_pred == y).sum()) != 1734 or not np.isclose(base_bal, 0.642409, atol=2e-6):
        raise ValueError(f"historical EQUAL10 reproduction failed: acc={base_acc}, bal_acc={base_bal}")

    metric_rows, label_rows, pred_columns, confusion = [], [], {}, {}
    def add_result(name, scenario, method, members, weights, p):
        row, per_label, pred = score(name, scenario, method, members, weights, p, y, baseline_pred)
        metric_rows.append(row)
        label_rows.extend(per_label)
        pred_columns[f"pred::{name}"] = pred
        for i, label in enumerate(label_order):
            pred_columns[f"prob::{name}::{label}"] = p[:, i]
        confusion[name] = confusion_matrix(y, pred, labels=label_order).tolist()
        return row, per_label, pred

    fixed_results = {}
    for scenario, members in SCENARIOS.items():
        equal = {sid: 1 / len(members) for sid in members}
        family_counts: dict[str, int] = {}
        for sid in members:
            family = _model_family(metadata_by_id[sid]["model"])
            family_counts[family] = family_counts.get(family, 0) + 1
        fam_equal = {sid: 1 / (len(family_counts) * family_counts[_model_family(metadata_by_id[sid]["model"])])
                     for sid in members}
        for method, weights in (("EQUAL", equal), ("FAMILY-EQUAL", fam_equal)):
            name = f"{scenario}-{method}"
            p = blend(members, matrices, weights)
            row, labels, pred = add_result(name, scenario, method, members, weights, p)
            row["family_count"] = len(family_counts)
            row["family_member_counts_json"] = json.dumps(family_counts, sort_keys=True)
            fixed_results[(scenario, method)] = {"row": row, "labels": labels, "pred": pred,
                                                 "weights": weights}

    # Reproduce recorded historical mixtures and weights.
    historical_specs = [
        ("HIST-EQUAL9-v013", HISTORICAL_EQUAL9,
         {sid: 1 / len(HISTORICAL_EQUAL9) for sid in HISTORICAL_EQUAL9}, "EQUAL"),
        ("HIST-EQUAL10-Qwen-v013", HISTORICAL_EQUAL10_QWEN,
         {sid: 1 / len(HISTORICAL_EQUAL10_QWEN) for sid in HISTORICAL_EQUAL10_QWEN}, "EQUAL"),
        ("HIST-WEIGHTED5-v013", list(WEIGHTED5_WEIGHTS), WEIGHTED5_WEIGHTS, "OPTIMIZED-FULL-V0"),
        ("HIST-EQUAL10", HISTORICAL_EQUAL10,
         {sid: 1 / len(HISTORICAL_EQUAL10) for sid in HISTORICAL_EQUAL10}, "EQUAL"),
    ]
    historical_rows = []
    historical_label_metrics = {}
    for name, members, weights, method in historical_specs:
        p = blend(members, matrices, weights)
        row, labels, _ = add_result(name, "HISTORICAL", method, members, weights, p)
        historical_rows.append(row)
        historical_label_metrics[name] = labels

    # Use deterministic prompt-group folds shared by all optimized scenarios.
    fold_ids = np.full(len(v0), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    fold_indices = []
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(v0)), y, groups=prompts)):
        fold_ids[held] = fold
        fold_indices.append(held)
    if (fold_ids < 0).any() or any(
            set(prompts[fold_ids == fold]) & set(prompts[fold_ids != fold]) for fold in range(5)):
        raise ValueError("prompt-group cross-fitting left rows unassigned or split prompt groups")

    ranked_fixed = sorted(metric_rows[:20], key=lambda r: (-r["balanced_accuracy"], -r["accuracy"], r["name"]))
    selected_scenarios, seen = [], set()
    for row in ranked_fixed:
        if row["scenario"] not in seen:
            selected_scenarios.append((row["scenario"], row["strategy"]))
            seen.add(row["scenario"])
        if len(selected_scenarios) == 3:
            break
    optimized_rows, optimized_fold_weights = [], {}
    for scenario, selected_method in selected_scenarios:
        members = SCENARIOS[scenario]
        oof_p, weights_by_fold = cross_fitted_weights(
            [matrices[sid] for sid in members], y, fold_ids, label_order, l2=0.05)
        name = f"{scenario}-OPTIMIZED-CV"
        row, labels, pred = add_result(name, scenario, "OPTIMIZED-CV", members,
                                       {sid: float(np.mean([w[i] for w in weights_by_fold.values()]))
                                        for i, sid in enumerate(members)}, oof_p)
        row["selected_fixed_method"] = selected_method
        row["family_count"] = len({_model_family(metadata_by_id[sid]["model"]) for sid in members})
        row["family_member_counts_json"] = ""
        row["weights_json"] = "cross_fitted; see optimized_weights.json"
        optimized_rows.append(row)
        optimized_fold_weights[name] = {str(fold + 1): dict(zip(members, weights_by_fold[fold]))
                                        for fold in range(5)}

    all_metrics = pd.DataFrame(metric_rows)
    # Add a unified metric row for every evaluated method; keep historical full-set baseline first in report.
    all_metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(label_rows).to_csv(OUT / "per_label_metrics.csv", index=False)
    pred_frame = pd.DataFrame({"id": ids, "prompt_id": prompts, "y_true": y, **pred_columns})
    pred_frame.to_parquet(OUT / "predictions.parquet", index=False)
    (OUT / "confusion_matrices.json").write_text(json.dumps({"labels": label_order, "matrices": confusion}, indent=2) + "\n")
    (OUT / "optimized_weights.json").write_text(json.dumps({
        "optimizer": "src.ensemble_factory.cross_fitted_weights / fit_weights",
        "l2": 0.05, "folding": "GroupKFold(n_splits=5), groups=V0 prompt_id",
        "selected_configurations": [{"scenario": s, "best_fixed_method": m} for s, m in selected_scenarios],
        "weights_by_held_out_fold": optimized_fold_weights,
    }, indent=2) + "\n")

    # Full per-label precision/recall for named focus labels in fixed and optimized configurations.
    focus_names = {r["name"] for r in ranked_fixed[:3]} | {r["name"] for r in optimized_rows} | {"HIST-EQUAL10"}
    focus_metrics = [r for r in label_rows if r["name"] in focus_names and r["label"] in FOCUS_LABELS]
    top3_fixed_rows = ranked_fixed[:3]
    recommendations = []
    hist_by_name = {x["name"]: x for x in historical_rows}
    for row in top3_fixed_rows:
        opt = next((x for x in optimized_rows if x["scenario"] == row["scenario"]), None)
        vs_equal9 = (row["balanced_accuracy"] - hist_by_name["HIST-EQUAL9-v013"]["balanced_accuracy"]) * 100
        vs_equal10_qwen = (row["balanced_accuracy"] - hist_by_name["HIST-EQUAL10-Qwen-v013"]["balanced_accuracy"]) * 100
        vs_weighted5 = (row["balanced_accuracy"] - hist_by_name["HIST-WEIGHTED5-v013"]["balanced_accuracy"]) * 100
        if opt:
            delta_cv = opt["balanced_accuracy"] - row["balanced_accuracy"]
            reason = (f"{row['checkpoint_count']} checkpoints, {row['balanced_accuracy']:.2%} full-V0 balanced accuracy; "
                      f"{vs_equal9:+.2f} pp vs EQUAL9, {vs_equal10_qwen:+.2f} pp vs EQUAL10-Qwen, "
                      f"{vs_weighted5:+.2f} pp vs WEIGHTED5; cross-fitted optimized result "
                      f"{opt['balanced_accuracy']:.2%} ({delta_cv*100:+.2f} pp vs fixed).")
        else:
            reason = f"{row['checkpoint_count']} checkpoints and {row['balanced_accuracy']:.2%} full-V0 balanced accuracy."
        recommendations.append({"recommendation": row["name"], "reason": reason})

    # Capture the focused diagnostics for all evaluated candidate methods, keeping the main report compact.
    focus_metrics = [r for r in label_rows if r["name"] in {x["name"] for x in ranked_fixed[:3]} |
                     {x["name"] for x in optimized_rows} | {"HIST-EQUAL10"} and r["label"] in FOCUS_LABELS]
    hist_public = {"HIST-EQUAL9-v013": "0.567", "HIST-EQUAL10-Qwen-v013": "0.567",
                   "HIST-WEIGHTED5-v013": "0.564", "HIST-EQUAL10": "not submitted"}
    for row in historical_rows:
        row["kaggle_public"] = hist_public[row["name"]]
    n0065_pred = np.asarray(label_order, dtype=object)[matrices["N0065"].argmax(axis=1)]
    n0081_pred = np.asarray(label_order, dtype=object)[matrices["N0081"].argmax(axis=1)]
    gemini37_mask = y == "gemini-3.7-flash"
    result = {
        "trust_list_version": trust_pointer["version"],
        "trust_list_csv": str(trust_csv.relative_to(ROOT)),
        "ranked_fixed": ranked_fixed,
        "historical": historical_rows,
        "optimized": optimized_rows,
        "focus_metrics": focus_metrics,
        "recommendations": recommendations,
        "specialist_recall": {
            "N0065_gemini37": float((n0065_pred[gemini37_mask] == "gemini-3.7-flash").mean()),
            "N0081_gemini37": float((n0081_pred[gemini37_mask] == "gemini-3.7-flash").mean()),
            "H08_EQUAL_gemini37": fixed_results[("H08", "EQUAL")]["labels"][LABELS.index("gemini-3.7-flash")]["recall"],
            "H08_FAMILY_EQUAL_gemini37": fixed_results[("H08", "FAMILY-EQUAL")]["labels"][LABELS.index("gemini-3.7-flash")]["recall"],
        },
    }
    metadata = {
        "inventory_requested": INVENTORY_NAME,
        "inventory_found": False,
        "id_resolution_source": str(trust_csv.relative_to(ROOT)),
        "trust_list_version": trust_pointer["version"],
        "validation_dataset": "v0", "validation_rows": len(v0),
        "validation_labels": label_order,
        "validation_id_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "validation_prompt_group_count": int(pd.Series(prompts).nunique()),
        "cache_only": True, "no_inference_or_training": True,
        "cache_sources_by_sequence_id": sources,
        "checkpoint_metadata_by_sequence_id": metadata_by_id,
        "scenario_definitions": SCENARIOS,
        "historical_ensembles": {"EQUAL9-v013": HISTORICAL_EQUAL9,
                                 "EQUAL10-Qwen-v013": HISTORICAL_EQUAL10_QWEN,
                                 "EQUAL10": HISTORICAL_EQUAL10,
                                 "WEIGHTED5-v013": WEIGHTED5_WEIGHTS},
        "optimized_selection": [{"scenario": s, "best_fixed_method": m} for s, m in selected_scenarios],
        "optimized_protocol": "GroupKFold(5) by prompt_id; fit weights on four folds and score the held-out fold; src.ensemble_factory.cross_fitted_weights l2=0.05",
        "baseline_reproduction": {"accuracy": base_acc, "balanced_accuracy": base_bal,
                                  "correct": int((baseline_pred == y).sum())},
        "public_scores_from_prior_kaggle_records": {"EQUAL9-v013": 0.567,
                                                     "EQUAL10-Qwen-v013": 0.567,
                                                     "WEIGHTED5-v013": 0.564},
        "note": "Historical full-V0 metrics are development scores because checkpoints and prior ensemble decisions used V0.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    REPORT.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "metrics_csv": str(OUT / "metrics.csv"),
                      "fixed_top10": [{"name": r['name'], "balanced_accuracy": r['balanced_accuracy']}
                                      for r in ranked_fixed[:10]],
                      "optimized": [{"name": r['name'], "accuracy": r['accuracy'],
                                     "balanced_accuracy": r['balanced_accuracy']} for r in optimized_rows],
                      "baseline": metadata['baseline_reproduction'],
                      "selected_optimized": metadata['optimized_selection']}, indent=2))


if __name__ == "__main__":
    main()
