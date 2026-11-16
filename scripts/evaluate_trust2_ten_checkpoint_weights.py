"""Compare B10, equal and cross-fitted learned weights from saved V0 caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.ensemble_factory import cross_fitted_weights, fit_weights
from src.evaluation import evaluate
from src.v2_checkpoint_trust import LABELS
from evaluate_trust2_extended_weighted import load_probs, load_v0

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "artifacts/analysis/trust2_extended_weighted"
OUT_DIR = ROOT / "artifacts/analysis/trust2_ten_checkpoint_weights"
REPORT = ROOT / "docs/trust_list_2_ten_checkpoint_weight_comparison.md"
IDS = ["N0071", "N0056", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
SEED = 2026
L2 = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Match fit_weights' unclipped-to-one multiclass log-loss objective."""
    class_index = {label: i for i, label in enumerate(LABELS)}
    target = np.fromiter((class_index[label] for label in y), dtype=np.int64, count=len(y))
    return float(-np.log(np.clip(p[np.arange(len(y)), target], 1e-12, 1.0)).mean())


def metric_block(y: np.ndarray, p: np.ndarray, reference_pred: np.ndarray | None = None) -> dict:
    pred = np.asarray(LABELS, dtype=object)[p.argmax(axis=1)]
    ev = evaluate(y, p, LABELS)
    correct = pred == y
    out = {
        "accuracy": ev["accuracy"],
        "balanced_accuracy": ev["balanced_accuracy"],
        "correct": int(correct.sum()),
        "log_loss": log_loss(y, p),
        "per_label_accuracy": {},
    }
    if reference_pred is not None:
        ref_correct = reference_pred == y
        out["delta_balanced_accuracy_pp_vs_B10"] = (ev["balanced_accuracy"] - evaluate(y, np.eye(len(LABELS))[np.array([LABELS.index(x) for x in reference_pred])], LABELS)["balanced_accuracy"]) * 100
        out["corrected_B10_errors"] = int((~ref_correct & correct).sum())
        out["introduced_errors_vs_B10"] = int((ref_correct & ~correct).sum())
    for label in LABELS:
        mask = y == label
        out["per_label_accuracy"][label] = {
            "support": int(mask.sum()), "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
        }
    return out


def report_text(result: dict) -> str:
    metrics = result["metrics"]
    order = ["B10", "EQUAL10", "OPTIMIZED10 OOF", "OPTIMIZED10 full-V0 fit"]
    lines = [
        "# Trust List 2 ten-checkpoint weight comparison", "",
        "Cache-only evaluation over all 2,699 V0 rows. No inference, classifier training, Kaggle work, or production ensemble changes were performed. The V0-optimized weights use V0 labels; only the cross-fitted OOF result is treated as the primary generalization comparison.", "",
        "## Comparison", "",
        "| Strategy | V0 accuracy | V0 balanced accuracy | Correct | Δ vs B10 | Corrected | Introduced | Log loss |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        m = metrics[name]
        if name == "B10":
            delta, fixed, introduced = "—", "—", "—"
        else:
            delta = f"{m['delta_balanced_accuracy_pp_vs_B10']:+.3f} pp"
            fixed, introduced = str(m["corrected_B10_errors"]), str(m["introduced_errors_vs_B10"])
        lines.append(f"| {name} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['correct']:,}/2,699 | {delta} | {fixed} | {introduced} | {m['log_loss']:.6f} |")
    lines.extend(["", "## Weights", "", "B10 preserves the original WEIGHTED5 relative weights: each original weight is halved, and each of the five added checkpoints receives 0.1. EQUAL10 assigns 0.1 to every checkpoint.", "", "| Checkpoint | B10 | EQUAL10 | OPT OOF mean | OPT OOF SD | OPT OOF min–max | Full-V0 fit |", "|---|---:|---:|---:|---:|---:|---:|"])
    for sid in IDS:
        s = result["weight_stability"][sid]
        lines.append(f"| {sid} | {result['weights']['B10'][sid]:.6f} | 0.100000 | {s['mean']:.6f} | {s['std']:.6f} | {s['min']:.6f}–{s['max']:.6f} | {result['weights']['OPTIMIZED10 full-V0 fit'][sid]:.6f} |")
    lines.extend(["", "### Fold-specific OPTIMIZED10 OOF weights", "", "| Checkpoint | " + " | ".join(f"Fold {f}" for f in result["fold_weights"]) + " |", "|---|" + "---:|" * len(result["fold_weights"])])
    for i, sid in enumerate(IDS):
        lines.append("| " + sid + " | " + " | ".join(f"{result['fold_weights'][fold][i]:.6f}" for fold in result["fold_weights"]) + " |")
    lines.extend(["", "## Per-label accuracy", "", "| Label | B10 | EQUAL10 | OPTIMIZED10 OOF | Full-V0 fit |", "|---|---:|---:|---:|---:|"])
    for label in LABELS:
        vals = [metrics[name]["per_label_accuracy"][label]["accuracy"] for name in order]
        lines.append(f"| {label} | " + " | ".join(f"{v:.2%}" for v in vals) + " |")
    lines.extend(["", "## Error overlap", "", "Pairwise and three-way intersections count rows misclassified by each listed strategy.", "", "| Error set intersection | Rows |", "|---|---:|"])
    for key, count in result["error_overlap"].items():
        lines.append(f"| {key} | {count} |")
    c = result["conclusions"]
    top_oof_summary = ", ".join(
        f"{sid} ({result['weight_stability'][sid]['mean']:.3f})"
        for sid in c["top_oof_checkpoints"]
    )
    lines.extend(["", "## Conclusions", "",
        f"1. EQUAL10 {'improves' if c['equal10_improves_b10'] else 'does not improve'} on B10 by {metrics['EQUAL10']['delta_balanced_accuracy_pp_vs_B10']:+.3f} pp balanced accuracy.",
        f"2. Cross-fitted OPTIMIZED10 {'improves' if c['optimized_oof_improves_b10'] else 'does not improve'} on B10 by {metrics['OPTIMIZED10 OOF']['delta_balanced_accuracy_pp_vs_B10']:+.3f} pp. This is the primary out-of-fold comparison.",
        f"3. Fold-weight stability: mean per-checkpoint standard deviation {c['mean_fold_weight_sd']:.4f}; the largest fold range is {c['largest_fold_weight_range']:.4f} ({c['largest_fold_weight_range_checkpoint']}).",
        f"4. Highest mean OOF weights: {top_oof_summary}.",
        f"5. Best accuracy is {c['best_accuracy_strategy']} ({metrics[c['best_accuracy_strategy']]['balanced_accuracy']:.4%}); best generalization evidence is OPTIMIZED10 OOF because its fold weights are learned without the held-out prompt groups.",
        "", "OPTIMIZED10 full-V0 fit metrics are in-sample: weights were fit using all V0 labels and are not a generalization estimate.",
        "", "## Inputs and reproducibility", "",
        f"- Checkpoint order: `{', '.join(IDS)}`.",
        f"- Prompt-group folds: `GroupKFold(n_splits=5, shuffle=True, random_state={SEED})`; each prompt group is assigned wholly to one held-out fold.",
        f"- Weight fitting uses `fit_weights` / `cross_fitted_weights` from `src/ensemble_factory.py`, log loss with `l2={L2}`, simplex constraints, and five fold-specific training complements.",
        f"- Extended evaluation input artifact SHA256: `{result['inputs']['extended_metrics_sha256']}`.",
        "- Per-checkpoint cache paths and SHA256 hashes, exact row order, canonical label order, fold assignments, weight vectors, and metrics are recorded in `artifacts/analysis/trust2_ten_checkpoint_weights/evaluation_metrics.json`.",
        "- Row-level probabilities and predictions for all four strategies are in `artifacts/analysis/trust2_ten_checkpoint_weights/v0_ensemble_predictions.parquet`.", ""])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v0 = load_v0()
    y = v0.model.to_numpy(dtype=object)
    ext_metrics_path = EXT_DIR / "evaluation_metrics.json"
    ext_rows_path = EXT_DIR / "v0_ensemble_predictions.parquet"
    ext = json.loads(ext_metrics_path.read_text(encoding="utf-8"))
    ext_rows = pd.read_parquet(ext_rows_path)
    if ext_rows.id.astype(str).tolist() != v0.id.tolist():
        raise ValueError("extended ensemble artifact does not align to canonical V0 ID order")
    if ext["inputs"]["checkpoint_sequence_order"] != IDS:
        raise ValueError("extended evaluation checkpoint order differs from requested v013 order")

    matrices: list[np.ndarray] = []
    cache_records = {}
    for sid in IDS:
        rec = ext["inputs"]["checkpoint_records"][sid]
        path = ROOT / rec["v0_probability_cache"]
        if sha256(path) != rec["v0_probability_cache_sha256"]:
            raise ValueError(f"{sid}: cached probability artifact hash changed: {path}")
        matrices.append(load_probs(path, v0))
        cache_records[sid] = {"experiment_id": rec["experiment_id"], "path": str(path.relative_to(ROOT)),
                              "sha256": sha256(path), "model": rec["model"]}

    # B10 comes from the prior evaluation's exact saved matrix, rather than a
    # recalculation, and is independently checked against the specified formula.
    b10 = ext_rows[[f"prob::B10-Baseline::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    w5 = sum(matrices[i] * ext["inputs"]["weighted5_weights_from_previous_report"][IDS[i]] for i in range(5))
    expected_b10 = (5 * w5 + sum(matrices[5:])) / 10
    if not np.allclose(b10, expected_b10, rtol=0, atol=1e-12):
        raise ValueError("saved B10 probability matrix differs from the exact specified weighted formula")

    equal10 = np.mean(np.stack(matrices), axis=0)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.full(len(v0), -1, dtype=np.int64)
    groups = v0.prompt_id.to_numpy(dtype=object)
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(v0)), y, groups)):
        folds[val_idx] = fold
    if np.any(folds < 0) or len(np.unique(folds)) != 5:
        raise ValueError("failed to assign every V0 row to one of five prompt-group folds")
    # Confirm groups never straddle folds.
    if pd.DataFrame({"prompt_id": groups, "fold": folds}).groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("a prompt group crossed cross-fitting folds")
    opt_oof, raw_fold_weights = cross_fitted_weights(matrices, y, folds, LABELS, l2=L2)
    fold_weights = {str(f): raw_fold_weights[f] for f in sorted(raw_fold_weights)}
    opt_full_weights = fit_weights(matrices, y, LABELS, l2=L2)
    opt_full = np.einsum("m,nmc->nc", opt_full_weights, np.stack(matrices, axis=1))

    predictions = {name: np.asarray(LABELS, dtype=object)[p.argmax(axis=1)] for name, p in {
        "B10": b10, "EQUAL10": equal10, "OPTIMIZED10 OOF": opt_oof,
        "OPTIMIZED10 full-V0 fit": opt_full,
    }.items()}
    probs = {"B10": b10, "EQUAL10": equal10, "OPTIMIZED10 OOF": opt_oof,
             "OPTIMIZED10 full-V0 fit": opt_full}
    b10_pred = predictions["B10"]
    metric_names = ["B10", "EQUAL10", "OPTIMIZED10 OOF", "OPTIMIZED10 full-V0 fit"]
    metrics = {name: metric_block(y, probs[name], None if name == "B10" else b10_pred) for name in metric_names}
    b10_metrics = ext["ranked_experiments"]
    b10_row = next(row for row in b10_metrics if row["name"] == "B10-Baseline")
    if not np.array_equal(predictions["B10"], ext_rows["pred::B10-Baseline"].astype(str).to_numpy()):
        raise ValueError("B10 row predictions differ from the saved reference cache")
    if abs(metrics["B10"]["balanced_accuracy"] - b10_row["balanced_accuracy"]) > 1e-12:
        raise ValueError("B10 balanced accuracy does not reproduce 63.9073%")

    fold_matrix = np.asarray([fold_weights[str(f)] for f in sorted(int(x) for x in fold_weights)])
    stability = {}
    for i, sid in enumerate(IDS):
        vals = fold_matrix[:, i]
        stability[sid] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                          "min": float(vals.min()), "max": float(vals.max()), "range": float(vals.max() - vals.min())}
    error = {name: predictions[name] != y for name in metric_names[:3]}
    overlap = {}
    for a, b in [("B10", "EQUAL10"), ("B10", "OPTIMIZED10 OOF"), ("EQUAL10", "OPTIMIZED10 OOF")]:
        overlap[f"{a} ∩ {b}"] = int(np.sum(error[a] & error[b]))
    overlap["B10 ∩ EQUAL10 ∩ OPTIMIZED10 OOF"] = int(np.sum(error["B10"] & error["EQUAL10"] & error["OPTIMIZED10 OOF"]))
    weights = {
        "B10": {sid: (0.5 * ext["inputs"]["weighted5_weights_from_previous_report"][sid] if i < 5 else 0.1)
                for i, sid in enumerate(IDS)},
        "EQUAL10": {sid: 0.1 for sid in IDS},
        "OPTIMIZED10 full-V0 fit": {sid: float(opt_full_weights[i]) for i, sid in enumerate(IDS)},
    }
    top_oof = sorted(IDS, key=lambda sid: (-stability[sid]["mean"], sid))[:3]
    best_name = max(metric_names, key=lambda name: metrics[name]["balanced_accuracy"])
    result = {
        "inputs": {
            "dataset": "v0", "rows": 2699, "canonical_label_order": list(LABELS),
            "checkpoint_order": IDS, "checkpoint_caches": cache_records,
            "extended_metrics_artifact": str(ext_metrics_path.relative_to(ROOT)),
            "extended_metrics_sha256": sha256(ext_metrics_path),
            "extended_rows_artifact": str(ext_rows_path.relative_to(ROOT)),
            "extended_rows_sha256": sha256(ext_rows_path),
            "validation_id_sha256_sorted_ids": hashlib.sha256("\n".join(v0.id).encode()).hexdigest(),
            "prompt_group_count": int(pd.Series(groups).nunique()), "fold_seed": SEED,
            "fold_method": "GroupKFold(n_splits=5, shuffle=True, random_state=2026)",
            "l2": L2, "evaluation_mode": "cached probabilities only",
        },
        "metrics": metrics, "weights": weights, "fold_weights": fold_weights,
        "weight_stability": stability, "error_overlap": overlap,
        "fold_assignments": {str(v0.id[i]): int(folds[i]) for i in range(len(v0))},
        "conclusions": {
            "equal10_improves_b10": metrics["EQUAL10"]["balanced_accuracy"] > metrics["B10"]["balanced_accuracy"],
            "optimized_oof_improves_b10": metrics["OPTIMIZED10 OOF"]["balanced_accuracy"] > metrics["B10"]["balanced_accuracy"],
            "mean_fold_weight_sd": float(fold_matrix.std(axis=0).mean()),
            "largest_fold_weight_range": max(x["range"] for x in stability.values()),
            "largest_fold_weight_range_checkpoint": max(stability, key=lambda sid: stability[sid]["range"]),
            "top_oof_checkpoints": top_oof, "best_accuracy_strategy": best_name,
        },
    }
    output_columns = {"id": v0.id.to_numpy(), "prompt_id": groups, "y_true": y, "fold": folds}
    for name in metric_names:
        output_columns[f"pred::{name}"] = predictions[name]
        output_columns[f"correct::{name}"] = predictions[name] == y
        for i, label in enumerate(LABELS):
            output_columns[f"prob::{name}::{label}"] = probs[name][:, i]
    row_path = OUT_DIR / "v0_ensemble_predictions.parquet"
    pd.DataFrame(output_columns).to_parquet(row_path, index=False)
    metric_path = OUT_DIR / "evaluation_metrics.json"
    metric_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(report_text(result), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "artifact_dir": str(OUT_DIR),
                      "metrics": {name: {"balanced_accuracy": metrics[name]["balanced_accuracy"],
                                          "accuracy": metrics[name]["accuracy"], "log_loss": metrics[name]["log_loss"]}
                                  for name in metric_names},
                      "weights": weights, "fold_weights": fold_weights,
                      "conclusions": result["conclusions"]}, indent=2))


if __name__ == "__main__":
    main()
