"""Cache-only evaluation of fixed-weight Trust List 2 B6-B10 ensembles."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
SUBSET_DIR = ROOT / "artifacts/analysis/trust_list_2_checkpoint_subsets/trust-list-2-v013"
V013_PACKAGE = ROOT / "artifacts/kaggle/trust2_v013_top5/experiment_manifest.json"
V013_BUNDLE = ROOT / "artifacts/kaggle/trust2-v013-top5-models/trust2_v013_top5_manifest.json"
PREVIOUS = ROOT / "artifacts/analysis/trust2_w6_w7"
DOC_W6W7 = ROOT / "docs/trust_list_2_w6_w7_evaluation.md"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
OUT_DIR = ROOT / "artifacts/analysis/trust2_extended_weighted"
BASE5 = ["N0071", "N0056", "N0005", "N0058", "N0072"]
ADDED = ["N0030", "N0061", "N0044", "N0009", "N0048"]
Q_ID, U_ID = "N0074", "N0076"
VARIANTS = ("Baseline", "Qwen", "Filtered", "Combined")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rooted(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def load_v0() -> pd.DataFrame:
    rows = pd.read_parquet(V0_PATH)
    required = {"id", "prompt_id", "text", "model"}
    if missing := required - set(rows.columns):
        raise ValueError(f"V0 data missing columns: {sorted(missing)}")
    for col in required:
        rows[col] = rows[col].astype(str)
    rows = rows.sort_values("id", kind="stable").reset_index(drop=True)
    if len(rows) != 2699 or rows.id.duplicated().any() or set(rows.model) != set(LABELS):
        raise ValueError("expected complete, unique 2,699-row V0 with the canonical 18 labels")
    return rows


def load_probs(path: Path, v0: pd.DataFrame) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required cached V0 probabilities are missing: {path}")
    frame = pd.read_parquet(path)
    need = {"id", "y_true", *(f"prob::{label}" for label in LABELS)}
    if missing := need - set(frame.columns):
        raise ValueError(f"probability cache {path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any() or len(frame) != len(v0) or set(frame.id) != set(v0.id):
        raise ValueError(f"probability cache IDs do not match V0 exactly: {path}")
    ordered = frame.set_index("id").loc[v0.id.tolist()]
    if not np.array_equal(ordered.y_true.astype(str).to_numpy(), v0.model.to_numpy()):
        raise ValueError(f"ground-truth labels in probability cache are misaligned: {path}")
    if "prompt_id" in ordered and not np.array_equal(ordered.prompt_id.astype(str).to_numpy(), v0.prompt_id.to_numpy()):
        raise ValueError(f"prompt IDs in probability cache are misaligned: {path}")
    matrix = ordered[[f"prob::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    if (matrix.shape != (len(v0), len(LABELS)) or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or not np.allclose(matrix.sum(1), 1.0, atol=1e-5)):
        raise ValueError(f"invalid, nonfinite, or non-normalized cache: {path}")
    return matrix


def weights_from_report() -> dict[str, float]:
    text = DOC_W6W7.read_text(encoding="utf-8")
    inside = False
    weights = {}
    for line in text.splitlines():
        if line.startswith("| Sequence | Model | Weight |"):
            inside = True
            continue
        if inside and not line.startswith("|"):
            break
        if inside:
            match = re.match(r"\|\s*(N\d{4})\s*\|[^|]*\|\s*([0-9.]+)\s*\|", line)
            if match:
                weights[match.group(1)] = float(match.group(2))
    if list(weights) != BASE5 or not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ValueError(f"could not recover exact ordered W5 weights from {DOC_W6W7}: {weights}")
    return weights


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v0 = load_v0()
    y = v0.model.to_numpy(dtype=object)
    selected_path = SUBSET_DIR / "selected_checkpoints.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("trust_list_version") != "trust-list-2-v013":
        raise ValueError("selected checkpoint list is not from Trust List 2 v013")
    selected_by_size = selected["full_v0_selection"]["selected_by_size"]
    expected_order = BASE5 + ADDED
    for k in range(5, 11):
        observed = [str(item["sequence_id"]) for item in selected_by_size[str(k)]]
        if observed != expected_order[:k]:
            raise ValueError(f"v013 greedy selection order for B{k} changed: {observed}")
    selection_rows = {str(row["sequence_id"]): row for row in selected_by_size["10"]}

    weights_w5 = weights_from_report()
    previous_metrics = json.loads((PREVIOUS / "evaluation_metrics.json").read_text(encoding="utf-8"))
    prior_weights = previous_metrics["inputs"]["weighted5_weights"]
    if any(not np.isclose(weights_w5[sid], prior_weights[sid], rtol=0, atol=1e-15) for sid in BASE5):
        raise ValueError("WEIGHTED5 weights differ from the prior W6/W7 evaluation")
    package = json.loads(V013_PACKAGE.read_text(encoding="utf-8"))
    bundle = json.loads(V013_BUNDLE.read_text(encoding="utf-8"))
    if package.get("trust_list_version") != "trust-list-2-v013" or bundle.get("sequence_ids") != BASE5:
        raise ValueError("the frozen v013 WEIGHTED5 package or bundle has unexpected membership")
    package_weights = package["strategies"]["WEIGHTED5"]["final_weights"]
    if any(not np.isclose(weights_w5[sid], package_weights[sid], rtol=0, atol=1e-15) for sid in BASE5):
        raise ValueError("report weights differ from the original successful v013 WEIGHTED5 weights")

    probability_by_id = {}
    checkpoint_records = {}
    prior_cache_sources = previous_metrics["inputs"]["probability_cache_sources"]
    for sid in expected_order:
        record = selection_rows[sid]
        experiment = str(record["experiment_id"])
        manifest_path = ROOT / "artifacts/predictions" / f"{experiment}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{sid}: manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_id") != experiment:
            raise ValueError(f"{sid}: experiment ID disagrees with v013 selection")
        # The v013 subset analysis stores complete V0 caches separately for
        # full-data checkpoints; their normal experiment prediction files may
        # cover only the original validation split. W5 sources are frozen to
        # the exact caches used by the preceding W6/W7 evaluation.
        if sid in BASE5:
            cache_path = rooted(prior_cache_sources[sid]["path"])
        else:
            cache_path = ROOT / "artifacts/analysis/trust_list_2_v0" / f"{experiment}.parquet"
            if not cache_path.is_file():
                raise FileNotFoundError(f"{sid}: required full-V0 probability cache missing: {cache_path}")
        probability_by_id[sid] = load_probs(cache_path, v0)
        checkpoint_records[sid] = {
            "experiment_id": experiment,
            "model": str(record["model"]),
            "training_dataset": str(record.get("training_dataset", manifest.get("training_dataset"))),
            "checkpoint_path": str(record.get("checkpoint_path", manifest.get("selected_checkpoint_path", manifest.get("checkpoint_path")))),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "v0_probability_cache": str(cache_path.relative_to(ROOT)),
            "v0_probability_cache_sha256": sha256(cache_path),
            "model_revision": manifest.get("model_revision"),
            "calibration_temperature": manifest.get("calibration_temperature", 1.0),
        }

    # Confirm B5 from these inputs reproduces both saved W5 probability and row predictions.
    w5 = sum(probability_by_id[sid] * weights_w5[sid] for sid in BASE5)
    previous_rows_path = PREVIOUS / "v0_ensemble_predictions.parquet"
    previous_rows = pd.read_parquet(previous_rows_path)
    if previous_rows.id.astype(str).tolist() != v0.id.tolist():
        raise ValueError("previous W6/W7 row-level predictions are not in canonical V0 order")
    previous_w5 = previous_rows[[f"prob::WEIGHTED5::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    if not np.allclose(w5, previous_w5, rtol=0, atol=1e-12):
        raise ValueError("reconstructed W5 probability matrix does not match previous W6/W7 artifact")
    w5_pred = np.asarray(LABELS, dtype=object)[w5.argmax(axis=1)]
    previous_w5_pred = previous_rows["pred::WEIGHTED5"].astype(str).to_numpy()
    if not np.array_equal(w5_pred.astype(str), previous_w5_pred):
        raise ValueError("reconstructed W5 row predictions do not match the previous evaluation")
    w5_metrics = evaluate(y, w5, LABELS)
    if (int(np.sum(w5_pred == y)) != 1630
            or not np.isclose(w5_metrics["balanced_accuracy"], 0.6038876460352971, atol=5e-9)):
        raise ValueError("W5 did not reproduce the expected 60.3888% balanced accuracy")

    q = load_probs(rooted(previous_metrics["inputs"]["probability_cache_sources"]["N0074"]["path"]), v0)
    u = load_probs(rooted(previous_metrics["inputs"]["probability_cache_sources"]["N0076"]["path"]), v0)
    cache_records = {sid: dict(record) for sid, record in checkpoint_records.items()}
    for sid in (Q_ID, U_ID):
        meta = previous_metrics["inputs"]["probability_cache_sources"][sid]
        path = rooted(meta["path"])
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"{sid} cache SHA256 changed since the preceding W6/W7 evaluation")
        cache_records[sid] = {"v0_probability_cache": str(path.relative_to(ROOT)),
                              "v0_probability_cache_sha256": sha256(path),
                              "experiment_id": meta.get("experiment_id")}

    matrices: dict[str, np.ndarray] = {"WEIGHTED5": w5}
    predictions: dict[str, np.ndarray] = {"WEIGHTED5": w5_pred}
    metrics_by_name = {"WEIGHTED5": w5_metrics}
    spec_by_name = {}
    for k in range(6, 11):
        extras = expected_order[5:k]
        base = (5.0 * w5 + sum(probability_by_id[sid] for sid in extras)) / k
        base_name = f"B{k}-Baseline"
        variant_matrices = {
            base_name: base,
            f"B{k}-Qwen": 0.85 * base + 0.15 * q,
            f"B{k}-Filtered": 0.90 * base + 0.10 * u,
            f"B{k}-Combined": 0.75 * base + 0.15 * q + 0.10 * u,
        }
        for name, matrix in variant_matrices.items():
            if (matrix.shape != (2699, 18) or not np.isfinite(matrix).all() or np.any(matrix < 0)
                    or not np.allclose(matrix.sum(1), 1.0, atol=1e-5)):
                raise ValueError(f"{name} generated invalid probability rows")
            pred = np.asarray(LABELS, dtype=object)[matrix.argmax(axis=1)]
            matrices[name] = matrix
            predictions[name] = pred
            metrics_by_name[name] = evaluate(y, matrix, LABELS)
            spec_by_name[name] = {
                "base_size": k,
                "variant": name.removeprefix(f"B{k}-"),
                "formula": f"B{k}=(5*WEIGHTED5 + sum({extras}))/ {k}; variant={name.removeprefix(f'B{k}-')}",
                "base_member_ids": BASE5 + extras,
                "base_checkpoint_weights": {**{sid: 5.0 * weights_w5[sid] / k for sid in BASE5},
                                            **{sid: 1.0 / k for sid in extras}},
                "additional_coefficients": ({"N0074": 0.15} if name.endswith("-Qwen") else
                    {"N0076": 0.10} if name.endswith("-Filtered") else
                    {"N0074": 0.15, "N0076": 0.10} if name.endswith("-Combined") else {}),
            }

    # The prior W7 result is retained as an independent reference, not included among the 20 ranked runs.
    reference_name = "Previous-W7-Combined"
    previous_w7_matrix = previous_rows[[f"prob::W7-Combined::{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    previous_w7_pred = np.asarray(LABELS, dtype=object)[previous_w7_matrix.argmax(axis=1)]
    metrics_by_name[reference_name] = evaluate(y, previous_w7_matrix, LABELS)
    matrices[reference_name] = previous_w7_matrix
    predictions[reference_name] = previous_w7_pred
    expected_prev_w7 = previous_metrics["metrics"]["W7-Combined"]["balanced_accuracy"]
    if not np.isclose(metrics_by_name[reference_name]["balanced_accuracy"], expected_prev_w7, atol=1e-12):
        raise ValueError("previous W7-Combined reference does not reproduce its saved metric")

    metric_rows = []
    per_label = {}
    row_prediction_columns = {
        "id": v0.id.to_numpy(), "prompt_id": v0.prompt_id.to_numpy(),
        "y_true": v0.model.to_numpy(),
    }
    for name, matrix in matrices.items():
        row_prediction_columns[f"pred::{name}"] = predictions[name]
        row_prediction_columns[f"correct::{name}"] = predictions[name] == y
        for j, label in enumerate(LABELS):
            row_prediction_columns[f"prob::{name}::{label}"] = matrix[:, j]
    row_predictions = pd.DataFrame(row_prediction_columns)
    for k in range(6, 11):
        base_name = f"B{k}-Baseline"
        base_pred = predictions[base_name]
        base_correct = base_pred == y
        for variant in VARIANTS:
            name = f"B{k}-{variant}"
            pred = predictions[name]
            corr = pred == y
            m = metrics_by_name[name]
            base_ba = metrics_by_name[base_name]["balanced_accuracy"]
            row = {
                "name": name, "base_size": k, "variant": variant,
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "correct": int(corr.sum()),
                "balanced_accuracy_delta_pp_vs_Bk": (m["balanced_accuracy"] - base_ba) * 100,
                "balanced_accuracy_delta_pp_vs_WEIGHTED5": (m["balanced_accuracy"] - w5_metrics["balanced_accuracy"]) * 100,
                "corrected_vs_Bk": int((~base_correct & corr).sum()),
                "introduced_errors_vs_Bk": int((base_correct & ~corr).sum()),
                "accuracy_delta_pp_vs_Bk": (m["accuracy"] - metrics_by_name[base_name]["accuracy"]) * 100,
                "per_label_accuracy": {}, "per_label_delta_pp_vs_Bk": {},
                **spec_by_name[name],
            }
            label_metrics = {}
            for label in LABELS:
                mask = y == label
                acc = float(np.mean(pred[mask] == y[mask]))
                base_acc = float(np.mean(base_pred[mask] == y[mask]))
                label_metrics[label] = {"support": int(mask.sum()), "accuracy": acc,
                                        "base_accuracy": base_acc, "delta_pp_vs_Bk": (acc - base_acc) * 100}
                row["per_label_accuracy"][label] = acc
                row["per_label_delta_pp_vs_Bk"][label] = (acc - base_acc) * 100
            per_label[name] = label_metrics
            metric_rows.append(row)

    ranked = sorted(metric_rows, key=lambda r: (-r["balanced_accuracy"], r["name"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    b_metrics = {}
    for k in range(6, 11):
        name = f"B{k}-Baseline"
        m = metrics_by_name[name]
        b_metrics[str(k)] = {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                             "correct": int(np.sum(predictions[name] == y))}
    by_size_best = {str(k): max((r for r in ranked if r["base_size"] == k),
                                key=lambda r: r["balanced_accuracy"])["name"] for k in range(6, 11)}

    inputs = {
        "validation_dataset": "v0", "validation_rows": 2699,
        "validation_id_sha256_sorted_ids": hashlib.sha256("\n".join(v0.id).encode()).hexdigest(),
        "labels_canonical_order": list(LABELS),
        "selection_artifact": str(selected_path.relative_to(ROOT)),
        "selection_artifact_sha256": sha256(selected_path),
        "checkpoint_sequence_order": expected_order,
        "weighted5_weights_from_previous_report": weights_w5,
        "weighted5_probability_cache_sources": {sid: {"path": checkpoint_records[sid]["v0_probability_cache"],
            "sha256": checkpoint_records[sid]["v0_probability_cache_sha256"]} for sid in BASE5},
        "addition_and_reference_cache_sources": cache_records,
        "checkpoint_records": checkpoint_records,
        "previous_w6_w7_prediction_artifact": str(previous_rows_path.relative_to(ROOT)),
        "previous_w6_w7_prediction_artifact_sha256": sha256(previous_rows_path),
        "previous_w6_w7_metrics_artifact": str((PREVIOUS / "evaluation_metrics.json").relative_to(ROOT)),
        "previous_w6_w7_metrics_artifact_sha256": sha256(PREVIOUS / "evaluation_metrics.json"),
        "weighting_rule": "Bk=(5*WEIGHTED5+sum(P_i for i=6..k))/k; the five original W5 internal weights are preserved proportionally; each added checkpoint contributes 1/k.",
        "variant_formulas": {"Baseline": "Bk", "Qwen": "0.85*Bk+0.15*N0074",
            "Filtered": "0.90*Bk+0.10*N0076", "Combined": "0.75*Bk+0.15*N0074+0.10*N0076"},
        "evaluation_mode": "saved probability caches only; no inference or weight fitting",
        "w5_reproduction": {"balanced_accuracy": w5_metrics["balanced_accuracy"],
            "accuracy": w5_metrics["accuracy"], "correct": int(np.sum(w5_pred == y)),
            "identical_to_previous_row_predictions": True},
    }
    best = ranked[0]
    best_base = max((b_metrics[str(k)]["balanced_accuracy"], k) for k in range(6, 11))
    w7_ba = metrics_by_name[reference_name]["balanced_accuracy"]
    base_ba_seq = [b_metrics[str(k)]["balanced_accuracy"] for k in range(6, 11)]
    full_result = {
        "inputs": inputs,
        "experiment_count": len(ranked),
        "ranked_experiments": ranked,
        "base_ensemble_metrics": b_metrics,
        "best_variant_by_base_size": by_size_best,
        "per_label_accuracy_deltas_vs_base": per_label,
        "references": {
            "WEIGHTED5": {"accuracy": w5_metrics["accuracy"], "balanced_accuracy": w5_metrics["balanced_accuracy"],
                          "correct": int(np.sum(w5_pred == y))},
            "Previous-W7-Combined": {"accuracy": metrics_by_name[reference_name]["accuracy"],
                "balanced_accuracy": w7_ba, "correct": int(np.sum(previous_w7_pred == y)),
                "source_artifact": str(previous_rows_path.relative_to(ROOT))},
        },
        "conclusion": {
            "best_base_ensemble_size_by_Bk_balanced_accuracy": int(best_base[1]),
            "best_overall_experiment": best["name"], "best_overall_balanced_accuracy": best["balanced_accuracy"],
            "best_overall_delta_pp_vs_previous_W7": (best["balanced_accuracy"] - w7_ba) * 100,
            "any_experiment_beats_previous_W7": bool(any(r["balanced_accuracy"] > w7_ba for r in ranked)),
            "base_balanced_accuracy_by_size": b_metrics,
            "base_Bk_is_monotonic": bool(all(a <= b for a, b in zip(base_ba_seq, base_ba_seq[1:]))),
            "best_variant_balanced_accuracy_by_size": {str(k): next(r["balanced_accuracy"] for r in ranked if r["name"] == by_size_best[str(k)]) for k in range(6, 11)},
            "best_variant_is_monotonic": bool(all(
                next(r["balanced_accuracy"] for r in ranked if r["name"] == by_size_best[str(k)]) <=
                next(r["balanced_accuracy"] for r in ranked if r["name"] == by_size_best[str(k + 1)])
                for k in range(6, 10))),
        },
    }
    row_predictions.to_parquet(OUT_DIR / "v0_ensemble_predictions.parquet", index=False)
    (OUT_DIR / "evaluation_metrics.json").write_text(json.dumps(full_result, indent=2) + "\n", encoding="utf-8")
    write_report(full_result)
    print(json.dumps({"report": str(ROOT / "docs/trust_list_2_extended_weighted_ensembles.md"),
                      "artifact_dir": str(OUT_DIR), "best": best["name"],
                      "best_balanced_accuracy": best["balanced_accuracy"],
                      "best_delta_vs_previous_w7_pp": (best["balanced_accuracy"] - w7_ba) * 100,
                      "best_base_size": int(best_base[1]),
                      "base_Bk_balanced_accuracy": b_metrics}, indent=2))


def write_report(result: dict) -> None:
    ranked = result["ranked_experiments"]
    lines = ["# Trust List 2 extended weighted ensembles: B6–B10", "",
        "Cache-only evaluation on all 2,699 V0 responses. These are exploratory results because V0 was used for checkpoint selection and prior ensemble optimization. No inference, weight fitting, checkpoint reselection, Kaggle operation, or production change was performed.", "",
        "## Results", "",
        "The exact WEIGHTED5 weights were read from `docs/trust_list_2_w6_w7_evaluation.md`. Reconstructed W5 balanced accuracy is 60.3888%, and all 2,699 W5 row predictions exactly match the saved W6/W7 evaluation artifact. The previous W7-Combined reference is 61.9456%.", "",
        "| Rank | Experiment | Base size | Variant | V0 accuracy | V0 balanced accuracy | Correct | Δ BA vs Bk | Δ BA vs W5 | Corrected vs Bk | Regressed vs Bk |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ranked:
        lines.append(f"| {r['rank']} | {r['name']} | {r['base_size']} | {r['variant']} | {r['accuracy']:.4%} | "
                     f"{r['balanced_accuracy']:.4%} | {r['correct']:,}/2,699 | {r['balanced_accuracy_delta_pp_vs_Bk']:+.3f} pp | "
                     f"{r['balanced_accuracy_delta_pp_vs_WEIGHTED5']:+.3f} pp | {r['corrected_vs_Bk']} | {r['introduced_errors_vs_Bk']} |")
    conc = result["conclusion"]
    best_base_size = conc["best_base_ensemble_size_by_Bk_balanced_accuracy"]
    best_base_ba = result["base_ensemble_metrics"][str(best_base_size)]["balanced_accuracy"]
    base_sequence = ", ".join(
        "B{} {:.4%}".format(k, result["base_ensemble_metrics"][str(k)]["balanced_accuracy"])
        for k in range(6, 11))
    best_sequence = ", ".join(
        "B{} {} {:.4%}".format(k, result["best_variant_by_base_size"][str(k)],
                                conc["best_variant_balanced_accuracy_by_size"][str(k)])
        for k in range(6, 11))
    lines.extend(["", "## Conclusions", "",
        f"- Best Bk probability mean by balanced accuracy: **B{best_base_size}** at {best_base_ba:.4%}.",
        f"- Best of the 20 combinations: **{conc['best_overall_experiment']}** at {conc['best_overall_balanced_accuracy']:.4%}, {conc['best_overall_delta_pp_vs_previous_W7']:+.3f} percentage points versus previous W7-Combined.",
        f"- Beats previous W7-Combined (61.9456%): **{'yes' if conc['any_experiment_beats_previous_W7'] else 'no'}**.",
        f"- Base Bk balanced accuracy across B6–B10: {base_sequence}; monotonic: {'yes' if conc['base_Bk_is_monotonic'] else 'no'}.",
        f"- Best variant at each size: {best_sequence}; monotonic: {'yes' if conc['best_variant_is_monotonic'] else 'no'}.", "",
        "## Per-label changes for top three", "",
        "Deltas below are accuracy percentage points versus the corresponding Bk baseline, using the true label for each V0 row.", ""])
    for r in ranked[:3]:
        lines.extend([f"### {r['name']} (rank {r['rank']})", "",
                      "| Label | Support | Bk accuracy | Variant accuracy | Δ accuracy |", "|---|---:|---:|---:|---:|"])
        for label, item in result["per_label_accuracy_deltas_vs_base"][r["name"]].items():
            lines.append(f"| {label} | {item['support']} | {item['base_accuracy']:.1%} | {item['accuracy']:.1%} | {item['delta_pp_vs_Bk']:+.1f} pp |")
        gains = [label for label, item in result["per_label_accuracy_deltas_vs_base"][r["name"]].items() if item["delta_pp_vs_Bk"] > 1e-10]
        losses = [label for label, item in result["per_label_accuracy_deltas_vs_base"][r["name"]].items() if item["delta_pp_vs_Bk"] < -1e-10]
        lines.extend(["", f"Gains: {', '.join(gains) if gains else 'none'}. Regressions: {', '.join(losses) if losses else 'none'}.", ""])
    lines.extend(["## Fixed membership, weights, and inputs", "",
        f"- Greedy order from the v013 subset artifact: `{', '.join(result['inputs']['checkpoint_sequence_order'])}`.",
        f"- Original WEIGHTED5 weights: `{json.dumps(result['inputs']['weighted5_weights_from_previous_report'], separators=(',', ':'))}`.",
        "- Each Bk is `(5 × W5 + sum(P_i for i=6..k)) / k`; this retains relative W5 weights and assigns each added checkpoint weight `1/k`.",
        "- Per size, variants use fixed coefficients: Baseline `Bk`; Qwen `0.85 Bk + 0.15 N0074`; Filtered `0.90 Bk + 0.10 N0076`; Combined `0.75 Bk + 0.15 N0074 + 0.10 N0076`.",
        "- All cached probability matrices were checked for exactly 2,699 unique V0 IDs, canonical 18-class ordering, matching labels/prompt IDs, finite nonnegative values, and unit row sums.",
        "- Cache paths, hashes, manifests, IDs, class order and formulas are recorded in `artifacts/analysis/trust2_extended_weighted/evaluation_metrics.json`.", "",
        "## Artifacts", "",
        "- `artifacts/analysis/trust2_extended_weighted/evaluation_metrics.json`.",
        "- `artifacts/analysis/trust2_extended_weighted/v0_ensemble_predictions.parquet` (probabilities, predictions and correctness flags for the 20 runs plus W5/W7 references).", ""])
    (ROOT / "docs/trust_list_2_extended_weighted_ensembles.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
