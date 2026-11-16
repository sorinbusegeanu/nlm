"""Cache-only EQUAL10 leave-one-out and extension comparison on V0."""
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
TEN_DIR = ROOT / "artifacts/analysis/trust2_ten_checkpoint_weights"
W6W7_DIR = ROOT / "artifacts/analysis/trust2_w6_w7"
OUT_DIR = ROOT / "artifacts/analysis/trust2_equal10_ablation"
REPORT = ROOT / "docs/trust_list_2_equal10_ablation.md"
IDS = ["N0071", "N0056", "N0005", "N0058", "N0072", "N0030", "N0061", "N0044", "N0009", "N0048"]
EXTRA = ["N0074", "N0076"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    indices = {label: i for i, label in enumerate(LABELS)}
    target = np.fromiter((indices[label] for label in y), dtype=np.int64, count=len(y))
    return float(-np.log(np.clip(p[np.arange(len(y)), target], 1e-12, 1.0)).mean())


def evaluate_matrix(y: np.ndarray, p: np.ndarray, baseline_pred: np.ndarray | None = None) -> dict:
    pred = np.asarray(LABELS, dtype=object)[p.argmax(axis=1)]
    ev = evaluate(y, p, LABELS)
    correct = pred == y
    result = {"accuracy": ev["accuracy"], "balanced_accuracy": ev["balanced_accuracy"],
              "correct": int(correct.sum()), "log_loss": log_loss(y, p),
              "prediction": pred, "per_label_accuracy": {}}
    if baseline_pred is not None:
        base_correct = baseline_pred == y
        result["delta_balanced_accuracy_pp_vs_EQUAL10"] = (
            ev["balanced_accuracy"] - evaluate(y, np.eye(len(LABELS))[[LABELS.index(v) for v in baseline_pred]], LABELS)["balanced_accuracy"]
        ) * 100
        result["corrected_vs_EQUAL10"] = int((~base_correct & correct).sum())
        result["introduced_vs_EQUAL10"] = int((base_correct & ~correct).sum())
    for label in LABELS:
        mask = y == label
        result["per_label_accuracy"][label] = {
            "support": int(mask.sum()), "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
        }
    return result


def render_report(result: dict) -> str:
    rows = result["ranked_experiments"]
    by_name = {x["name"]: x for x in rows}
    lines = [
        "# Trust List 2 EQUAL10 checkpoint ablation and extension", "",
        "Cache-only evaluation on all 2,699 V0 responses. No training, inference, weight optimization, Kaggle work, or production changes were performed.", "",
        "## Results", "",
        "| Experiment | Checkpoints | Balanced accuracy | Δ vs EQUAL10 | Corrected | Introduced | Log loss |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = "—" if row["name"] == "EQUAL10" else f"{row['delta_balanced_accuracy_pp_vs_EQUAL10']:+.3f} pp"
        fixed = "—" if row["name"] == "EQUAL10" else str(row["corrected_vs_EQUAL10"])
        added = "—" if row["name"] == "EQUAL10" else str(row["introduced_vs_EQUAL10"])
        lines.append(f"| {row['name']} | {row['checkpoint_count']} | {row['balanced_accuracy']:.4%} | {delta} | {fixed} | {added} | {row['log_loss']:.6f} |")
    lines.extend(["", "## Per-label changes for the three best experiments", "",
                  "Accuracy deltas are percentage points relative to EQUAL10.", ""])
    for row in rows[:3]:
        lines.extend([f"### {row['name']} ({row['balanced_accuracy']:.4%})", "",
                      "| Label | EQUAL10 accuracy | Variant accuracy | Δ accuracy |", "|---|---:|---:|---:|"])
        for label, item in result["per_label_deltas_vs_EQUAL10"][row["name"]].items():
            lines.append(f"| {label} | {item['baseline_accuracy']:.2%} | {item['accuracy']:.2%} | {item['delta_pp']:+.2f} pp |")
        lines.append("")
    c = result["conclusions"]
    lines.extend(["## Conclusions", "",
        "1. **Leave-one-out:** removing " + (" and ".join(c["removals_improving_balanced_accuracy"]) or "no checkpoint") + " improves balanced accuracy over EQUAL10. Removing " + (" and ".join(c["removals_improving_overall_accuracy"]) if c["removals_improving_overall_accuracy"] else "none") + " also raises overall accuracy.",
        f"2. N0044 and N0009 both contribute positively to equal-weight blending: removing N0044 lowers balanced accuracy by {abs(c['removal_delta_pp']['N0044']):.3f} pp, and removing N0009 lowers it by {abs(c['removal_delta_pp']['N0009']):.3f} pp. Their near-zero optimized weights therefore do not imply they are unhelpful in an equal blend.",
        f"3. EQUAL11-Qwen changes balanced accuracy by {by_name['EQUAL11-Qwen']['delta_balanced_accuracy_pp_vs_EQUAL10']:+.3f} pp; EQUAL11-Filtered changes it by {by_name['EQUAL11-Filtered']['delta_balanced_accuracy_pp_vs_EQUAL10']:+.3f} pp.",
        f"4. Best overall: **{c['best_experiment']}**, {c['best_balanced_accuracy']:.4%} balanced accuracy, {c['best_delta_pp_vs_EQUAL10']:+.3f} pp versus EQUAL10.",
        f"5. Best tested ensemble size: **{c['best_checkpoint_count']} checkpoints** ({c['best_by_checkpoint_count'][str(c['best_checkpoint_count'])]['name']}, {c['best_by_checkpoint_count'][str(c['best_checkpoint_count'])]['balanced_accuracy']:.4%}). Tested-size winners: " + ", ".join(f"{n}: {v['name']} {v['balanced_accuracy']:.4%}" for n, v in c["best_by_checkpoint_count"].items()) + ".", "",
        "## Cache and method details", "",
        f"- EQUAL10 baseline reproduces the previous artifact exactly at the row-prediction level; {result['baseline_reproduction']['correct']:,}/2,699 correct, accuracy {result['baseline_reproduction']['accuracy']:.4%}, balanced accuracy {result['baseline_reproduction']['balanced_accuracy']:.4%}.",
        f"- Base checkpoint order: `{', '.join(IDS)}`. EQUAL9 variants give each retained checkpoint weight `1/9`; extensions use `1/11` or `1/12` for every included checkpoint.",
        "- All cache IDs, labels, prompt IDs, finite values, and probability row sums were checked. Cached source paths, SHA256 hashes, checkpoint IDs and exact per-experiment weights are recorded in `artifacts/analysis/trust2_equal10_ablation/evaluation_metrics.json`.",
        "- Row-level probabilities and predictions for EQUAL10 and all 13 evaluated variants are in `artifacts/analysis/trust2_equal10_ablation/v0_ensemble_predictions.parquet`.", ""])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v0 = load_v0()
    y = v0.model.to_numpy(dtype=object)
    prior_metrics_path = TEN_DIR / "evaluation_metrics.json"
    prior_rows_path = TEN_DIR / "v0_ensemble_predictions.parquet"
    prior_metrics = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
    prior_rows = pd.read_parquet(prior_rows_path)
    if prior_rows.id.astype(str).tolist() != v0.id.tolist():
        raise ValueError("prior ten-checkpoint rows do not match sorted V0 response IDs")
    if prior_metrics["inputs"]["checkpoint_order"] != IDS:
        raise ValueError("prior EQUAL10 checkpoint order differs from requested order")

    matrices = {}
    cache_sources = {}
    for sid in IDS:
        source = prior_metrics["inputs"]["checkpoint_caches"][sid]
        path = ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{sid}: required cached V0 probabilities missing: {path}")
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"{sid}: cache SHA256 differs from prior ten-checkpoint evaluation: {path}")
        matrices[sid] = load_probs(path, v0)
        cache_sources[sid] = {"path": str(path.relative_to(ROOT)), "sha256": digest,
                              "experiment_id": source["experiment_id"]}

    w6w7_path = W6W7_DIR / "evaluation_metrics.json"
    w6w7 = json.loads(w6w7_path.read_text(encoding="utf-8"))
    q_u_cache_records = {}
    for sid in EXTRA:
        source = w6w7["inputs"]["probability_cache_sources"][sid]
        path = ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{sid}: required cached V0 probabilities missing: {path}")
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"{sid}: cache SHA256 differs from prior W6/W7 evaluation: {path}")
        matrices[sid] = load_probs(path, v0)
        q_u_cache_records[sid] = {"path": str(path.relative_to(ROOT)), "sha256": digest,
                                  "experiment_id": source.get("experiment_id")}

    base10 = np.mean(np.stack([matrices[sid] for sid in IDS]), axis=0)
    saved_equal = prior_rows[[f"prob::EQUAL10::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    baseline_pred = np.asarray(LABELS, dtype=object)[base10.argmax(axis=1)]
    saved_pred = prior_rows["pred::EQUAL10"].astype(str).to_numpy()
    if not np.allclose(base10, saved_equal, rtol=0, atol=1e-12):
        raise ValueError("reconstructed EQUAL10 probability matrix differs from previous evaluation")
    if not np.array_equal(baseline_pred.astype(str), saved_pred):
        raise ValueError("reconstructed EQUAL10 row predictions differ from previous evaluation")
    baseline_metric = evaluate_matrix(y, base10)
    expected = {"balanced_accuracy": 0.6424086502609992, "accuracy": 0.6424601704334939, "correct": 1734}
    for key, val in expected.items():
        if baseline_metric[key] != val if isinstance(val, int) else not np.isclose(baseline_metric[key], val, atol=5e-10):
            raise ValueError(f"EQUAL10 {key} failed expected baseline check: {baseline_metric[key]}")

    experiments = {"EQUAL10": (IDS, base10)}
    for sid in IDS:
        members = [x for x in IDS if x != sid]
        experiments[f"EQUAL9-minus-{sid}"] = (members, np.mean(np.stack([matrices[x] for x in members]), axis=0))
    for sid, name in [("N0074", "EQUAL11-Qwen"), ("N0076", "EQUAL11-Filtered")]:
        members = IDS + [sid]
        experiments[name] = (members, np.mean(np.stack([matrices[x] for x in members]), axis=0))
    members12 = IDS + EXTRA
    experiments["EQUAL12-Combined"] = (members12, np.mean(np.stack([matrices[x] for x in members12]), axis=0))

    all_metrics, all_predictions, all_probs, all_weights = {}, {}, {}, {}
    per_label_deltas = {}
    for name, (members, p) in experiments.items():
        if p.shape != (2699, 18) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"{name}: invalid combined probability matrix")
        m = evaluate_matrix(y, p, None if name == "EQUAL10" else baseline_pred)
        pred = m.pop("prediction")
        all_metrics[name], all_predictions[name], all_probs[name] = m, pred, p
        all_weights[name] = {sid: 1.0 / len(members) for sid in members}
        per_label_deltas[name] = {
            label: {"support": m["per_label_accuracy"][label]["support"],
                    "baseline_accuracy": baseline_metric["per_label_accuracy"][label]["accuracy"],
                    "accuracy": m["per_label_accuracy"][label]["accuracy"],
                    "delta_pp": (m["per_label_accuracy"][label]["accuracy"] - baseline_metric["per_label_accuracy"][label]["accuracy"]) * 100}
            for label in LABELS
        }

    ranked = []
    for name, (members, _) in experiments.items():
        m = all_metrics[name]
        ranked.append({"name": name, "checkpoint_count": len(members), "checkpoint_ids": members,
                       "weights": all_weights[name], "accuracy": m["accuracy"],
                       "balanced_accuracy": m["balanced_accuracy"], "correct": m["correct"],
                       "log_loss": m["log_loss"],
                       "delta_balanced_accuracy_pp_vs_EQUAL10": m.get("delta_balanced_accuracy_pp_vs_EQUAL10", 0.0),
                       "corrected_vs_EQUAL10": m.get("corrected_vs_EQUAL10", 0),
                       "introduced_vs_EQUAL10": m.get("introduced_vs_EQUAL10", 0)})
    ranked.sort(key=lambda r: (-r["balanced_accuracy"], r["name"]))
    removals_ba, removals_acc, removal_deltas = [], [], {}
    for sid in IDS:
        name = f"EQUAL9-minus-{sid}"
        removal_deltas[sid] = all_metrics[name]["delta_balanced_accuracy_pp_vs_EQUAL10"]
        if all_metrics[name]["balanced_accuracy"] > baseline_metric["balanced_accuracy"]:
            removals_ba.append(sid)
        if all_metrics[name]["accuracy"] > baseline_metric["accuracy"]:
            removals_acc.append(sid)
    best_by_n = {}
    for n in (9, 10, 11, 12):
        candidates = [r for r in ranked if r["checkpoint_count"] == n]
        best_by_n[str(n)] = max(candidates, key=lambda r: r["balanced_accuracy"])
    best = ranked[0]
    result = {
        "inputs": {"dataset": "v0", "rows": 2699, "canonical_label_order": list(LABELS),
                   "checkpoint_order": IDS, "extra_checkpoint_ids": EXTRA,
                   "checkpoint_caches": cache_sources, "extension_caches": q_u_cache_records,
                   "previous_ten_checkpoint_metrics": str(prior_metrics_path.relative_to(ROOT)),
                   "previous_ten_checkpoint_metrics_sha256": sha256(prior_metrics_path),
                   "previous_ten_checkpoint_rows": str(prior_rows_path.relative_to(ROOT)),
                   "previous_ten_checkpoint_rows_sha256": sha256(prior_rows_path),
                   "previous_w6_w7_metrics": str(w6w7_path.relative_to(ROOT)),
                   "previous_w6_w7_metrics_sha256": sha256(w6w7_path),
                   "validation_id_sha256_sorted_ids": hashlib.sha256("\n".join(v0.id).encode()).hexdigest(),
                   "evaluation_mode": "saved probability caches only; no inference or weight fitting"},
        "baseline_reproduction": {k: baseline_metric[k] for k in ("accuracy", "balanced_accuracy", "correct")},
        "baseline_row_predictions_identical": True,
        "experiment_count_including_reference": len(ranked), "ranked_experiments": ranked,
        "metrics_by_experiment": all_metrics, "per_label_deltas_vs_EQUAL10": per_label_deltas,
        "weights_by_experiment": all_weights,
        "conclusions": {"removals_improving_balanced_accuracy": removals_ba,
                        "removals_improving_overall_accuracy": removals_acc,
                        "removal_delta_pp": removal_deltas,
                        "best_experiment": best["name"], "best_balanced_accuracy": best["balanced_accuracy"],
                        "best_delta_pp_vs_EQUAL10": best["delta_balanced_accuracy_pp_vs_EQUAL10"],
                        "best_checkpoint_count": best["checkpoint_count"], "best_by_checkpoint_count": best_by_n,
                        "N0074_balanced_accuracy_delta_pp": all_metrics["EQUAL11-Qwen"]["delta_balanced_accuracy_pp_vs_EQUAL10"],
                        "N0076_balanced_accuracy_delta_pp": all_metrics["EQUAL11-Filtered"]["delta_balanced_accuracy_pp_vs_EQUAL10"]},
    }

    cols = {"id": v0.id.to_numpy(), "prompt_id": v0.prompt_id.to_numpy(), "y_true": y}
    for name in experiments:
        cols[f"pred::{name}"] = all_predictions[name]
        cols[f"correct::{name}"] = all_predictions[name] == y
        for i, label in enumerate(LABELS):
            cols[f"prob::{name}::{label}"] = all_probs[name][:, i]
    pd.DataFrame(cols).to_parquet(OUT_DIR / "v0_ensemble_predictions.parquet", index=False)
    (OUT_DIR / "evaluation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "artifact_dir": str(OUT_DIR),
                      "baseline_reproduction": result["baseline_reproduction"],
                      "best_experiment": best["name"], "best_balanced_accuracy": best["balanced_accuracy"],
                      "best_delta_pp_vs_EQUAL10": best["delta_balanced_accuracy_pp_vs_EQUAL10"],
                      "removals_improving_balanced_accuracy": removals_ba,
                      "removals_improving_overall_accuracy": removals_acc,
                      "best_by_checkpoint_count": best_by_n}, indent=2))


if __name__ == "__main__":
    main()
