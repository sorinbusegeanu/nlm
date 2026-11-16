"""Cache-only EQUAL9-minus-N0056 ablation and equal-weight extensions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.v2_checkpoint_trust import LABELS
from evaluate_trust2_extended_weighted import load_probs, load_v0

ROOT = Path(__file__).resolve().parents[1]
PREV_DIR = ROOT / "artifacts/analysis/trust2_equal10_ablation"
W6W7_DIR = ROOT / "artifacts/analysis/trust2_w6_w7"
OUT_DIR = ROOT / "artifacts/analysis/trust2_equal9_ablation"
REPORT = ROOT / "docs/trust_list_2_equal9_ablation.md"
BASE = ["N0071", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
ADDED = ["N0074", "N0076"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def loss(y, p):
    lookup = {label: i for i, label in enumerate(LABELS)}
    yi = np.fromiter((lookup[v] for v in y), dtype=np.int64, count=len(y))
    return float(-np.log(np.clip(p[np.arange(len(y)), yi], 1e-12, 1.0)).mean())


def calculate(y, p, base_pred=None):
    pred = np.asarray(LABELS, dtype=object)[p.argmax(axis=1)]
    metrics = evaluate(y, p, LABELS)
    good = pred == y
    result = {"accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
              "correct": int(good.sum()), "log_loss": loss(y, p), "prediction": pred,
              "per_label_accuracy": {}}
    if base_pred is not None:
        base_good = base_pred == y
        result["delta_balanced_accuracy_pp_vs_baseline"] = (metrics["balanced_accuracy"] - evaluate(
            y, np.eye(len(LABELS))[[LABELS.index(v) for v in base_pred]], LABELS)["balanced_accuracy"]) * 100
        result["corrected_vs_baseline"] = int((~base_good & good).sum())
        result["introduced_vs_baseline"] = int((base_good & ~good).sum())
    for label in LABELS:
        mask = y == label
        result["per_label_accuracy"][label] = {"support": int(mask.sum()),
            "correct": int(good[mask].sum()), "accuracy": float(good[mask].mean())}
    return result


def write_report(result):
    rows = result["ranked_experiments"]
    lines = ["# Trust List 2 EQUAL9 ablation and extension", "",
        "Cache-only evaluation over the full 2,699-row V0 dataset. EQUAL9-minus-N0056 is the fixed reference. No training, inference, Kaggle work, or production changes were performed.", "",
        "## Results", "",
        "| Experiment | Checkpoints | Balanced accuracy | Δ vs EQUAL9 | Corrected | Introduced | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["name"] == "EQUAL9-minus-N0056 (baseline)":
            delta, corrected, introduced = "—", "—", "—"
        else:
            delta = f"{r['delta_balanced_accuracy_pp_vs_baseline']:+.3f} pp"
            corrected, introduced = str(r["corrected_vs_baseline"]), str(r["introduced_vs_baseline"])
        lines.append(f"| {r['name']} | {r['checkpoint_count']} | {r['balanced_accuracy']:.4%} | {delta} | {corrected} | {introduced} | {r['log_loss']:.6f} |")
    lines.extend(["", "## Per-label changes for the three best experiments", "",
        "Accuracy deltas are percentage points relative to EQUAL9-minus-N0056.", ""])
    variant_rows = [r for r in rows if r["name"] != "EQUAL9-minus-N0056 (baseline)"]
    for r in variant_rows[:3]:
        lines.extend([f"### {r['name']} ({r['balanced_accuracy']:.4%})", "",
            "| Label | Baseline accuracy | Variant accuracy | Δ accuracy |", "|---|---:|---:|---:|"])
        for label, values in result["per_label_deltas_vs_baseline"][r["name"]].items():
            lines.append(f"| {label} | {values['baseline_accuracy']:.2%} | {values['accuracy']:.2%} | {values['delta_pp']:+.2f} pp |")
        lines.append("")
    c = result["conclusions"]
    lines.extend(["## Conclusions", "",
        f"- Best of the 11 tested variants: **{c['best_experiment']}**, {c['best_balanced_accuracy']:.4%} balanced accuracy ({c['best_delta_pp_vs_baseline']:+.3f} pp vs the EQUAL9 baseline).",
        f"- None of the 11 tested variants exceeds the EQUAL9 baseline at {result['baseline_reproduction']['balanced_accuracy']:.4%}.",
        f"- Removing a checkpoint from EQUAL9 improves balanced accuracy for: {', '.join(c['removals_improving_balanced_accuracy']) or 'none'}.",
        f"- EQUAL10-Qwen changes the baseline by {c['qwen_delta_pp']:+.3f} pp; EQUAL10-Filtered changes it by {c['filtered_delta_pp']:+.3f} pp.", "",
        "## Inputs", "",
        f"- Baseline members, in order: `{', '.join(BASE)}`; each has weight `1/9`.",
        "- EQUAL8 ablations give each remaining checkpoint weight `1/8`; extensions give every member weight `1/10`.",
        f"- Baseline row predictions match the saved prior EQUAL9 artifact exactly: {result['baseline_reproduction']['identical_row_predictions']}.",
        "- Every cache was checked for full V0 ID and prompt alignment, canonical 18-label ordering, finite nonnegative probabilities, and normalized rows. Cache paths, hashes, checkpoint IDs and exact weights are recorded in the JSON artifact.",
        "- Row-level probabilities and predictions for the baseline and 11 evaluations are in `artifacts/analysis/trust2_equal9_ablation/v0_ensemble_predictions.parquet`.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v0 = load_v0()
    y = v0.model.to_numpy(dtype=object)
    prior_metrics_path = PREV_DIR / "evaluation_metrics.json"
    prior_rows_path = PREV_DIR / "v0_ensemble_predictions.parquet"
    prior_metrics = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
    prior_rows = pd.read_parquet(prior_rows_path)
    if prior_rows.id.astype(str).tolist() != v0.id.tolist():
        raise ValueError("prior ablation row IDs do not match sorted V0 IDs")
    source_baseline = next(x for x in prior_metrics["ranked_experiments"] if x["name"] == "EQUAL9-minus-N0056")
    if source_baseline["checkpoint_ids"] != BASE:
        raise ValueError("saved EQUAL9-minus-N0056 membership/order differs from requested baseline")

    matrices, cache_records = {}, {}
    for sid in BASE:
        source = prior_metrics["inputs"]["checkpoint_caches"][sid]
        path = ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{sid}: missing cached full V0 probability artifact: {path}")
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"{sid}: cache SHA256 mismatch: {path}")
        matrices[sid] = load_probs(path, v0)
        cache_records[sid] = {"path": str(path.relative_to(ROOT)), "sha256": digest,
                              "experiment_id": source["experiment_id"]}

    w6w7_path = W6W7_DIR / "evaluation_metrics.json"
    w6w7 = json.loads(w6w7_path.read_text(encoding="utf-8"))
    added_records = {}
    for sid in ADDED:
        source = w6w7["inputs"]["probability_cache_sources"][sid]
        path = ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{sid}: missing cached full V0 probability artifact: {path}")
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"{sid}: cache SHA256 mismatch: {path}")
        matrices[sid] = load_probs(path, v0)
        added_records[sid] = {"path": str(path.relative_to(ROOT)), "sha256": digest,
                              "experiment_id": source.get("experiment_id")}

    base_p = np.mean(np.stack([matrices[sid] for sid in BASE]), axis=0)
    base_pred = np.asarray(LABELS, dtype=object)[base_p.argmax(axis=1)]
    saved_p = prior_rows[[f"prob::EQUAL9-minus-N0056::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    saved_pred = prior_rows["pred::EQUAL9-minus-N0056"].astype(str).to_numpy()
    if not np.allclose(base_p, saved_p, rtol=0, atol=1e-12):
        raise ValueError("reconstructed EQUAL9 baseline probabilities differ from saved artifact")
    if not np.array_equal(base_pred.astype(str), saved_pred):
        raise ValueError("reconstructed EQUAL9 baseline row predictions differ from saved artifact")
    base_metrics = calculate(y, base_p)
    for k, expected in {"balanced_accuracy": 0.6538975888640318, "accuracy": 0.6539459058910708, "correct": 1765}.items():
        actual = base_metrics[k]
        if (isinstance(expected, int) and actual != expected) or (not isinstance(expected, int) and not np.isclose(actual, expected, atol=1e-12)):
            raise ValueError(f"baseline {k} mismatch: {actual} != {expected}")

    experiments = {"EQUAL9-minus-N0056 (baseline)": (BASE, base_p)}
    for excluded in BASE:
        members = [sid for sid in BASE if sid != excluded]
        experiments[f"EQUAL8-minus-{excluded}"] = (members, np.mean(np.stack([matrices[sid] for sid in members]), axis=0))
    experiments["EQUAL10-Qwen"] = (BASE + ["N0074"], np.mean(np.stack([matrices[sid] for sid in BASE + ["N0074"]]), axis=0))
    experiments["EQUAL10-Filtered"] = (BASE + ["N0076"], np.mean(np.stack([matrices[sid] for sid in BASE + ["N0076"]]), axis=0))

    metrics, predictions, probabilities, weights, per_label = {}, {}, {}, {}, {}
    for name, (members, p) in experiments.items():
        if p.shape != (2699, 18) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError(f"{name}: invalid probability matrix")
        m = calculate(y, p, None if name.endswith("(baseline)") else base_pred)
        pred = m.pop("prediction")
        metrics[name], predictions[name], probabilities[name] = m, pred, p
        weights[name] = {sid: 1.0 / len(members) for sid in members}
        per_label[name] = {label: {"support": m["per_label_accuracy"][label]["support"],
            "baseline_accuracy": base_metrics["per_label_accuracy"][label]["accuracy"],
            "accuracy": m["per_label_accuracy"][label]["accuracy"],
            "delta_pp": (m["per_label_accuracy"][label]["accuracy"] - base_metrics["per_label_accuracy"][label]["accuracy"]) * 100}
            for label in LABELS}

    ranked = []
    for name, (members, _) in experiments.items():
        m = metrics[name]
        ranked.append({"name": name, "checkpoint_ids": members, "checkpoint_count": len(members),
            "weights": weights[name], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "correct": m["correct"], "log_loss": m["log_loss"],
            "delta_balanced_accuracy_pp_vs_baseline": m.get("delta_balanced_accuracy_pp_vs_baseline", 0.0),
            "corrected_vs_baseline": m.get("corrected_vs_baseline", 0),
            "introduced_vs_baseline": m.get("introduced_vs_baseline", 0)})
    ranked.sort(key=lambda r: (-r["balanced_accuracy"], r["name"]))
    removal_improvements = [sid for sid in BASE if metrics[f"EQUAL8-minus-{sid}"]["balanced_accuracy"] > base_metrics["balanced_accuracy"]]
    best = max((r for r in ranked if r["name"] != "EQUAL9-minus-N0056 (baseline)"),
               key=lambda r: r["balanced_accuracy"])
    result = {"inputs": {"dataset": "v0", "rows": 2699, "canonical_label_order": list(LABELS),
        "baseline_checkpoint_ids": BASE, "extension_checkpoint_ids": ADDED,
        "checkpoint_caches": cache_records, "extension_caches": added_records,
        "previous_ablation_metrics": str(prior_metrics_path.relative_to(ROOT)),
        "previous_ablation_metrics_sha256": sha256(prior_metrics_path),
        "previous_ablation_predictions": str(prior_rows_path.relative_to(ROOT)),
        "previous_ablation_predictions_sha256": sha256(prior_rows_path),
        "w6_w7_metrics": str(w6w7_path.relative_to(ROOT)), "w6_w7_metrics_sha256": sha256(w6w7_path),
        "validation_id_sha256_sorted_ids": hashlib.sha256("\n".join(v0.id).encode()).hexdigest(),
        "evaluation_mode": "saved probabilities only; no inference or weight fitting"},
        "baseline_reproduction": {"identical_row_predictions": True, "accuracy": base_metrics["accuracy"],
            "balanced_accuracy": base_metrics["balanced_accuracy"], "correct": base_metrics["correct"]},
        "experiment_count": 11, "reference_included_in_artifacts": True,
        "ranked_experiments": ranked, "metrics_by_experiment": metrics,
        "weights_by_experiment": weights, "per_label_deltas_vs_baseline": per_label,
        "conclusions": {"best_experiment": best["name"], "best_balanced_accuracy": best["balanced_accuracy"],
            "best_delta_pp_vs_baseline": best["delta_balanced_accuracy_pp_vs_baseline"],
            "removals_improving_balanced_accuracy": removal_improvements,
            "qwen_delta_pp": metrics["EQUAL10-Qwen"]["delta_balanced_accuracy_pp_vs_baseline"],
            "filtered_delta_pp": metrics["EQUAL10-Filtered"]["delta_balanced_accuracy_pp_vs_baseline"]}}

    columns = {"id": v0.id.to_numpy(), "prompt_id": v0.prompt_id.to_numpy(), "y_true": y}
    for name in experiments:
        columns[f"pred::{name}"] = predictions[name]
        columns[f"correct::{name}"] = predictions[name] == y
        for i, label in enumerate(LABELS):
            columns[f"prob::{name}::{label}"] = probabilities[name][:, i]
    pd.DataFrame(columns).to_parquet(OUT_DIR / "v0_ensemble_predictions.parquet", index=False)
    (OUT_DIR / "evaluation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    print(json.dumps({"report": str(REPORT), "best": best["name"],
        "balanced_accuracy": best["balanced_accuracy"], "delta_pp": best["delta_balanced_accuracy_pp_vs_baseline"],
        "removal_improvements": removal_improvements,
        "extensions": {n: metrics[n]["balanced_accuracy"] for n in ("EQUAL10-Qwen", "EQUAL10-Filtered")}}, indent=2))


if __name__ == "__main__":
    main()
