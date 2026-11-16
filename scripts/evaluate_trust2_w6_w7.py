"""Evaluate fixed W6/W7 probability blends from saved V0 caches only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
V013_DIR = ROOT / "artifacts/kaggle/trust2_v013_top5"
OUT_DIR = ROOT / "artifacts/analysis/trust2_w6_w7"
TRUST_CURRENT = ROOT / "docs/trust-list-2-v017.csv"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
BASELINE_IDS = ["N0071", "N0056", "N0005", "N0058", "N0072"]
EXPERIMENTS = {
    "W6-Qwen": {"WEIGHTED5": 0.85, "N0074": 0.15},
    "W6-Filtered": {"WEIGHTED5": 0.90, "N0076": 0.10},
    "W7-Combined": {"WEIGHTED5": 0.75, "N0074": 0.15, "N0076": 0.10},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_v0() -> pd.DataFrame:
    frame = pd.read_parquet(V0_PATH)
    for col in ("id", "prompt_id", "text", "model"):
        if col not in frame:
            raise ValueError(f"V0 data is missing {col}")
        frame[col] = frame[col].astype(str)
    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    if len(frame) != 2699 or frame.id.duplicated().any() or set(frame.model) != set(LABELS):
        raise ValueError("expected all 2,699 unique V0 responses and the canonical 18 labels")
    return frame


def load_cache(path: Path, v0: pd.DataFrame) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required saved V0 probability cache is missing: {path}")
    frame = pd.read_parquet(path)
    required = {"id", "y_true", *(f"prob::{label}" for label in LABELS)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cache {path} lacks required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any() or len(frame) != len(v0) or set(frame.id) != set(v0.id):
        raise ValueError(f"V0 IDs in probability cache do not match: {path}")
    ordered = frame.set_index("id").loc[v0.id.tolist()]
    if not np.array_equal(ordered.y_true.astype(str).to_numpy(), v0.model.to_numpy()):
        raise ValueError(f"V0 labels do not align in cache: {path}")
    if "prompt_id" in ordered and not np.array_equal(ordered.prompt_id.astype(str).to_numpy(), v0.prompt_id.to_numpy()):
        raise ValueError(f"V0 prompt IDs do not align in cache: {path}")
    values = ordered[[f"prob::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    if (values.shape != (len(v0), len(LABELS)) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError(f"invalid or non-normalized probability cache: {path}")
    return values


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v0 = load_v0()
    y_true = v0.model.to_numpy(dtype=object)

    # Frozen W5 values and V0 cache paths are taken from the successful v013 package.
    w5_manifest_path = V013_DIR / "experiment_manifest.json"
    w5_manifest = json.loads(w5_manifest_path.read_text(encoding="utf-8"))
    if w5_manifest.get("trust_list_version") != "trust-list-2-v013":
        raise ValueError("the baseline package is not Trust List 2 v013")
    checkpoint_meta = {str(item["sequence_id"]): item for item in w5_manifest["checkpoints"]}
    kaggle_bundle_path = ROOT / "artifacts/kaggle/trust2-v013-top5-models/trust2_v013_top5_manifest.json"
    kaggle_bundle = json.loads(kaggle_bundle_path.read_text(encoding="utf-8"))
    if kaggle_bundle.get("trust_list_version") != "trust-list-2-v013":
        raise ValueError("the packaged Kaggle inference bundle is not Trust List 2 v013")
    if list(item["sequence_id"] for item in w5_manifest["checkpoints"]) != BASELINE_IDS:
        raise ValueError("the frozen v013 baseline checkpoint order changed")
    weights = w5_manifest["strategies"]["WEIGHTED5"]["final_weights"]
    if list(weights) != BASELINE_IDS or not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ValueError("frozen WEIGHTED5 weights or order do not match v013")
    base_probabilities = {}
    cache_sources = {}
    for sid in BASELINE_IDS:
        path = Path(checkpoint_meta[sid]["validation_prediction_cache"])
        if not path.is_absolute():
            path = ROOT / path
        base_probabilities[sid] = load_cache(path, v0)
        cache_sources[sid] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    w5 = sum(base_probabilities[sid] * float(weights[sid]) for sid in BASELINE_IDS)
    if not np.allclose(w5.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("WEIGHTED5 probability blend is not normalized")

    # Resolve additions from the current catalog version and reuse their stored V0 predictions.
    trust = pd.read_csv(TRUST_CURRENT, dtype={"sequence_id": str})
    additions = {}
    for sid, expected_model, expected_dataset in (
        ("N0074", "qwen06", "v2.14"),
        ("N0076", "ultrabert380", "v2.14-prediction-filtered-80pct"),
    ):
        rows = trust.loc[trust.sequence_id.eq(sid)]
        if rows.empty:
            raise ValueError(f"{sid} is missing from current checkpoint catalog {TRUST_CURRENT}")
        row = rows.iloc[0]
        if str(row.model) != expected_model or str(row.training_dataset) != expected_dataset:
            raise ValueError(f"{sid} does not identify the requested checkpoint: {row.to_dict()}")
        experiment_id = str(row.experiment_id)
        manifest_path = ROOT / "artifacts/predictions" / f"{experiment_id}_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("model") != expected_model or manifest.get("training_dataset") != expected_dataset
                or manifest.get("validation_dataset") != "v0" or int(manifest.get("validation_rows", -1)) != 2699):
            raise ValueError(f"{sid} manifest does not match the current catalog record")
        cache_path = Path(manifest["prediction_path"])
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        additions[sid] = load_cache(cache_path, v0)
        cache_sources[sid] = {"path": str(cache_path.relative_to(ROOT)), "sha256": sha256(cache_path),
                              "experiment_id": experiment_id}

    probability_matrices = {"WEIGHTED5": w5}
    predictions = {"WEIGHTED5": np.asarray(LABELS, dtype=object)[w5.argmax(axis=1)]}
    for name, formula in EXPERIMENTS.items():
        matrix = float(formula["WEIGHTED5"]) * w5
        for sid, coefficient in formula.items():
            if sid == "WEIGHTED5":
                continue
            matrix = matrix + float(coefficient) * additions[sid]
        if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{name} blend has invalid probabilities")
        probability_matrices[name] = matrix
        predictions[name] = np.asarray(LABELS, dtype=object)[matrix.argmax(axis=1)]

    summary = {}
    per_label = []
    row_output = v0[["id", "prompt_id", "model"]].rename(columns={"model": "y_true"}).copy()
    baseline_correct = predictions["WEIGHTED5"] == y_true
    for name, matrix in probability_matrices.items():
        pred = predictions[name]
        correct = pred == y_true
        summary[name] = {
            "rows": int(len(y_true)),
            "correct": int(correct.sum()),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "corrected_vs_weighted5": 0 if name == "WEIGHTED5" else int((~baseline_correct & correct).sum()),
            "introduced_errors_vs_weighted5": 0 if name == "WEIGHTED5" else int((baseline_correct & ~correct).sum()),
            "accuracy_delta_pp_vs_weighted5": 0.0 if name == "WEIGHTED5" else float((correct.mean() - baseline_correct.mean()) * 100),
            "balanced_accuracy_delta_pp_vs_weighted5": 0.0 if name == "WEIGHTED5" else float((balanced_accuracy_score(y_true, pred) - balanced_accuracy_score(y_true, predictions["WEIGHTED5"])) * 100),
        }
        row_output[f"pred::{name}"] = pred
        row_output[f"correct::{name}"] = correct
        for col, label in enumerate(LABELS):
            row_output[f"prob::{name}::{label}"] = matrix[:, col]
    for label in LABELS:
        mask = y_true == label
        baseline_hits = int((predictions["WEIGHTED5"][mask] == y_true[mask]).sum())
        base_acc = baseline_hits / int(mask.sum())
        item = {"label": label, "support": int(mask.sum()), "WEIGHTED5_correct": baseline_hits,
                "WEIGHTED5_accuracy": base_acc}
        for name in EXPERIMENTS:
            hits = int((predictions[name][mask] == y_true[mask]).sum())
            acc = hits / int(mask.sum())
            item[f"{name}_correct"] = hits
            item[f"{name}_accuracy"] = acc
            item[f"{name}_delta_pp"] = (acc - base_acc) * 100
        per_label.append(item)
    per_label_frame = pd.DataFrame(per_label)

    row_output.to_parquet(OUT_DIR / "v0_ensemble_predictions.parquet", index=False)
    per_label_frame.to_csv(OUT_DIR / "per_label_accuracy.csv", index=False)
    inputs = {"trust_list_baseline": "trust-list-2-v013", "addition_catalog": "docs/trust-list-2-v017.csv",
              "validation_dataset": "v0", "validation_rows": len(v0), "validation_id_sha256": hashlib.sha256("\n".join(v0.id).encode()).hexdigest(),
              "labels": list(LABELS), "weighted5_sequence_ids": BASELINE_IDS, "weighted5_weights": weights,
              "experiment_formulas": EXPERIMENTS, "probability_cache_sources": cache_sources,
              "baseline_manifest": "artifacts/kaggle/trust2_v013_top5/experiment_manifest.json",
              "baseline_manifest_sha256": sha256(w5_manifest_path),
              "evaluation_caveat": "Exploratory full-V0 results; V0 was used for checkpoint selection and earlier ensemble optimization."}
    result = {"inputs": inputs, "metrics": summary,
              "best_by_balanced_accuracy": max(summary, key=lambda key: summary[key]["balanced_accuracy"]),
              "per_label_accuracy": per_label}
    (OUT_DIR / "evaluation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Include full per-label V0 changes and only the positive/negative changes in the compact report.
    lines = ["# Trust List 2 W6/W7 fixed-weight V0 evaluation", "",
        "These are exploratory full-V0 results. The checkpoints were selected on V0 and earlier ensemble optimization also used V0.",
        "No inference was run: all seven probability matrices were loaded from saved V0 caches, checked for exact response-ID alignment, and aligned to the canonical 18-label order.", "",
        "## Comparison", "",
        "| Ensemble | V0 accuracy | V0 balanced accuracy | Correct | Δ accuracy vs WEIGHTED5 | Δ balanced accuracy vs WEIGHTED5 | Corrected | Introduced errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = ["WEIGHTED5", "W6-Qwen", "W6-Filtered", "W7-Combined"]
    for name in order:
        m = summary[name]
        delta_accuracy = "—" if name == "WEIGHTED5" else f"{m['accuracy_delta_pp_vs_weighted5']:+.3f} pp"
        delta_balanced = "—" if name == "WEIGHTED5" else f"{m['balanced_accuracy_delta_pp_vs_weighted5']:+.3f} pp"
        corrected = "—" if name == "WEIGHTED5" else str(m["corrected_vs_weighted5"])
        introduced = "—" if name == "WEIGHTED5" else str(m["introduced_errors_vs_weighted5"])
        lines.append(f"| {name} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['correct']:,}/2,699 | "
            f"{delta_accuracy} | {delta_balanced} | {corrected} | {introduced} |")
    best = result["best_by_balanced_accuracy"]
    lines.extend(["", f"**Best by V0 balanced accuracy:** {best} ({summary[best]['balanced_accuracy']:.4%}).", "",
        "## Exact baseline and fixed formulas", "",
        "Baseline: successful Trust List 2 v013 Kaggle WEIGHTED5, using the same five checkpoint probability caches and saved WEIGHTED5 weights:", "",
        "| Sequence | Model | Weight | Cached V0 prediction source | Max length / processing note |", "|---|---|---:|---|---|"])
    for sid in BASELINE_IDS:
        item = checkpoint_meta[sid]
        candidate = kaggle_bundle["candidates"][sid]
        inf = candidate.get("config", {})
        temp = candidate.get("calibration_temperature", 1.0)
        process_note = (f"max_length={inf.get('max_length')}; inference_batch_size="
                        f"{inf.get('inference_batch_size', 8)}; calibration_temperature={temp}")
        if sid == "N0005":
            process_note += "; CatBoost temperature calibration"
        if sid == "N0072":
            process_note += "; SVC char+word feature views"
        lines.append(f"| {sid} | {item['model']} | {weights[sid]:.16f} | `{Path(item['validation_prediction_cache']).relative_to(ROOT)}` | {process_note} |")
    lines.extend(["", "Additional cached checkpoints:", "",
        f"- N0074 (`qwen06`, V2.14): `{cache_sources['N0074']['path']}`.",
        f"- N0076 (`ultrabert380`, V2.14-prediction-filtered-80pct): `{cache_sources['N0076']['path']}`.", "",
        "Fixed probability formulas:", "",
        "- W6-Qwen = `0.85 × WEIGHTED5 + 0.15 × N0074`.",
        "- W6-Filtered = `0.90 × WEIGHTED5 + 0.10 × N0076`.",
        "- W7-Combined = `0.75 × WEIGHTED5 + 0.15 × N0074 + 0.10 × N0076`.", "",
        "## Per-label accuracy changes", "",
        "Accuracy is correct predictions divided by V0 support for that true label. Deltas are percentage points versus WEIGHTED5.", "",
        "| Label | V0 support | WEIGHTED5 | W6-Qwen Δ | W6-Filtered Δ | W7-Combined Δ |", "|---|---:|---:|---:|---:|---:|"])
    for item in per_label:
        lines.append(f"| {item['label']} | {item['support']} | {item['WEIGHTED5_accuracy']:.1%} | "
                     f"{item['W6-Qwen_delta_pp']:+.1f} pp | {item['W6-Filtered_delta_pp']:+.1f} pp | {item['W7-Combined_delta_pp']:+.1f} pp |")
    lines.extend(["", "### Per-experiment improvements and regressions", ""])
    for name in EXPERIMENTS:
        improved = [x["label"] for x in per_label if x[f"{name}_delta_pp"] > 1e-10]
        regressed = [x["label"] for x in per_label if x[f"{name}_delta_pp"] < -1e-10]
        unchanged = [x["label"] for x in per_label if abs(x[f"{name}_delta_pp"]) <= 1e-10]
        lines.append(f"- **{name}:** improved {', '.join(improved) if improved else 'none'}; regressed {', '.join(regressed) if regressed else 'none'}; unchanged {', '.join(unchanged) if unchanged else 'none'}.")
    lines.extend(["", "## Artifacts and interpretation", "",
        "- Row-level predictions and all 18 probability columns: `artifacts/analysis/trust2_w6_w7/v0_ensemble_predictions.parquet`.",
        "- Per-label accuracy table: `artifacts/analysis/trust2_w6_w7/per_label_accuracy.csv`.",
        "- Machine-readable metrics, formulas, cache paths and hashes: `artifacts/analysis/trust2_w6_w7/evaluation_metrics.json`.",
        "- Kaggle public score: WEIGHTED5 `0.564` as previously recorded; new blends are not scored here.",
        "- Baseline artifacts and production ensemble selection were left unchanged.", ""])
    report_path = ROOT / "docs/trust_list_2_w6_w7_evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "artifacts": str(OUT_DIR),
                      "metrics": summary, "best_by_balanced_accuracy": best}, indent=2))


if __name__ == "__main__":
    main()
