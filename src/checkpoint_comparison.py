"""Compare two same-model checkpoints on the frozen V0 validation rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate


WEAK_CLASS_ORDER = (
    "claude-fable-5", "glm-5.3", "gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-4-8",
)
WEAK_CLASSES = frozenset(WEAK_CLASS_ORDER)
DISPLAY_NAMES = {
    "claude-fable-5": "Claude Fable 5",
    "glm-5.3": "GLM 5.3",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "claude-opus-4-8": "Claude Opus 4.8",
}


def _validated_probabilities(probabilities, n_rows, n_classes, name):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (n_rows, n_classes):
        raise ValueError(f"{name}: expected probability shape {(n_rows, n_classes)}, got {values.shape}")
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError(f"{name}: probabilities must be finite, nonnegative, and sum to one")
    return values


def align_checkpoint_predictions(previous, new, classes, validation_rows):
    """Align prediction tables by ID and verify truth/prompt metadata against frozen V0."""
    classes = list(classes)
    reference = validation_rows[["id", "prompt_id", "model"]].copy()
    reference = reference.rename(columns={"model": "y_true"})
    reference["id"] = reference.id.astype(str)
    reference["prompt_id"] = reference.prompt_id.astype(str)
    reference["y_true"] = reference.y_true.astype(str)
    if reference.id.duplicated().any():
        raise ValueError("frozen validation IDs must be unique")
    reference = reference.sort_values("id", kind="stable").reset_index(drop=True)
    aligned = {}
    for name, frame in (("previous", previous), ("new", new)):
        if not {"id", "prompt_id", "y_true"}.issubset(frame.columns):
            raise ValueError(f"{name}: prediction table needs id, prompt_id, and y_true")
        frame = frame.copy()
        frame["id"] = frame.id.astype(str)
        frame["prompt_id"] = frame.prompt_id.astype(str)
        frame["y_true"] = frame.y_true.astype(str)
        if frame.id.duplicated().any():
            raise ValueError(f"{name}: prediction IDs must be unique")
        if set(frame.id) != set(reference.id):
            raise ValueError(f"{name}: prediction IDs differ from frozen V0 validation IDs")
        frame = frame.set_index("id").loc[reference.id].reset_index()
        for column in ("y_true", "prompt_id"):
            if not np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"{name}: {column} differs from frozen V0 validation rows")
        missing = [f"prob::{label}" for label in classes if f"prob::{label}" not in frame]
        if missing:
            raise ValueError(f"{name}: missing probability columns {missing[:3]}")
        aligned[name] = _validated_probabilities(
            frame[[f"prob::{label}" for label in classes]].to_numpy(),
            len(frame), len(classes), name)
    return reference, aligned["previous"], aligned["new"]


def checkpoint_metrics(y_true, probabilities, classes):
    """Calculate top-1, top-3, reciprocal-rank, and per-class metrics."""
    y = np.asarray(y_true, dtype=str)
    classes = list(map(str, classes))
    probs = _validated_probabilities(probabilities, len(y), len(classes), "checkpoint")
    class_index = {label: index for index, label in enumerate(classes)}
    if set(y) - set(class_index):
        raise ValueError("validation labels contain classes absent from checkpoint class order")
    truth_indices = np.asarray([class_index[label] for label in y], dtype=int)
    # Stable sorting makes tied scores deterministic in canonical class order.
    order = np.argsort(-probs, axis=1, kind="stable")
    true_rank = np.argmax(order == truth_indices[:, None], axis=1) + 1
    top1_prediction = np.asarray(classes, dtype=object)[order[:, 0]]
    top3_correct = (true_rank <= min(3, len(classes)))
    basic = evaluate(y, probs, classes, probabilities=probs)
    support = np.bincount(truth_indices, minlength=len(classes))
    per_class = {}
    for index, label in enumerate(classes):
        mask = truth_indices == index
        per_class[label] = {
            "support": int(support[index]),
            "top1_recall": float(np.mean(top1_prediction[mask] == label)) if mask.any() else 0.0,
            "top3_accuracy": float(np.mean(top3_correct[mask])) if mask.any() else 0.0,
            "mrr": float(np.mean(1.0 / true_rank[mask])) if mask.any() else 0.0,
        }
    true_probability = probs[np.arange(len(y)), truth_indices]
    return {
        "balanced_accuracy": basic["balanced_accuracy"],
        "top3_accuracy": float(np.mean(top3_correct)),
        "mrr": float(np.mean(1.0 / true_rank)),
        "log_loss": basic["log_loss"],
        "per_class": per_class,
        "predictions": top1_prediction,
        "true_rank": true_rank,
        "true_probability": true_probability,
    }


def compare_probability_matrices(y_true, previous_probabilities, new_probabilities, classes):
    """Return overall metric deltas and prediction/rank movements."""
    previous = checkpoint_metrics(y_true, previous_probabilities, classes)
    new = checkpoint_metrics(y_true, new_probabilities, classes)
    y = np.asarray(y_true, dtype=str)
    corrected = (previous["predictions"] != y) & (new["predictions"] == y)
    regressed = (previous["predictions"] == y) & (new["predictions"] != y)
    rank_delta = previous["true_rank"] - new["true_rank"]
    probability_delta = new["true_probability"] - previous["true_probability"]
    return {
        "previous": previous,
        "new": new,
        "corrected_top1": int(corrected.sum()),
        "regressed_top1": int(regressed.sum()),
        "rank_up": int(np.sum(rank_delta > 0)),
        "rank_down": int(np.sum(rank_delta < 0)),
        "rank_same": int(np.sum(rank_delta == 0)),
        "mean_true_probability_change": float(probability_delta.mean()),
        "mean_true_probability_previous": float(previous["true_probability"].mean()),
        "mean_true_probability_new": float(new["true_probability"].mean()),
        "weak_classes_improved": [label for label in WEAK_CLASS_ORDER
                                   if new["per_class"].get(label, {}).get("top1_recall", 0)
                                   > previous["per_class"].get(label, {}).get("top1_recall", 0)],
    }


def _resolve_path(value, *, manifest_path, artifact_dir):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path,
                  Path(artifact_dir) / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _load_manifest(reference, *, artifact_dir, registry_path):
    supplied = Path(reference)
    if supplied.is_file():
        manifest_path = supplied
    else:
        experiment_id = str(reference)
        candidates = [Path(artifact_dir) / "predictions" / f"{experiment_id}_manifest.json"]
        try:
            from .registry import Registry
            row = next((item for item in Registry(registry_path).rows()
                        if item["experiment_id"] == experiment_id), None)
        except Exception:
            row = None
        if row and row.get("artifact_path"):
            artifact_path = Path(row["artifact_path"])
            candidates.insert(0, artifact_path.with_name(f"{experiment_id}_manifest.json"))
        manifest_path = next((path for path in candidates if path.is_file()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"could not find manifest for experiment {experiment_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("experiment_id") or not manifest.get("model"):
        raise ValueError(f"{manifest_path} is not a model experiment manifest")
    return manifest, manifest_path.resolve()


def _generate_predictions(manifest, manifest_path, validation_rows, classes, artifact_dir):
    from .inference import (predict_catboost_stylometry, predict_svc_bundle,
                            _predict_neural)
    model = str(manifest["model"])
    checkpoint_value = manifest.get("checkpoint_path") or manifest.get("checkpoint")
    if not checkpoint_value:
        raise ValueError(f"{manifest['experiment_id']} has no checkpoint path")
    checkpoint = _resolve_path(checkpoint_value, manifest_path=manifest_path,
                               artifact_dir=artifact_dir)
    config = manifest.get("config") or {}
    texts = validation_rows.text.astype(str).tolist()
    temperature = float(manifest.get("calibration_temperature") or 1.0)
    if model == "svc":
        import joblib
        probabilities, returned_classes = predict_svc_bundle(joblib.load(checkpoint), texts)
    elif model == "catboost_stylometry":
        probabilities, returned_classes = predict_catboost_stylometry(
            checkpoint, texts, classes, temperature=temperature)
    else:
        probabilities, returned_classes = _predict_neural(
            model, checkpoint, texts, classes, max_length=int(config.get("max_length", 384)),
            temperature=temperature, base_checkpoint=config.get("checkpoint"),
            revision=manifest.get("model_revision") or config.get("revision"),
            fix_mistral_regex=bool(config.get("fix_mistral_regex", False)),
            text_only_backbone=bool(config.get("text_only_backbone", False)))
    returned_classes = list(map(str, returned_classes))
    if set(returned_classes) != set(classes):
        raise ValueError(f"{manifest['experiment_id']} prediction classes differ from frozen V0")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if returned_classes != list(classes):
        probabilities = probabilities[:, [returned_classes.index(label) for label in classes]]
    probabilities = _validated_probabilities(probabilities, len(validation_rows), len(classes), model)
    frame = validation_rows[["id", "prompt_id", "model"]].copy()
    frame = frame.rename(columns={"model": "y_true"})
    for index, label in enumerate(classes):
        frame[f"prob::{label}"] = probabilities[:, index]
    destination = Path(artifact_dir) / "predictions" / f"{manifest['experiment_id']}_validation.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame, destination


def _prediction_frame(manifest, manifest_path, validation_rows, classes, artifact_dir):
    prediction_value = manifest.get("prediction_path")
    if prediction_value:
        path = _resolve_path(prediction_value, manifest_path=manifest_path,
                             artifact_dir=artifact_dir)
        if path.is_file():
            return pd.read_parquet(path), path, False
    # Also reuse a previously generated artifact before loading a checkpoint again.
    generated_path = Path(artifact_dir) / "predictions" / f"{manifest['experiment_id']}_validation.parquet"
    if generated_path.is_file():
        return pd.read_parquet(generated_path), generated_path, True
    frame, path = _generate_predictions(manifest, manifest_path, validation_rows,
                                        classes, artifact_dir)
    return frame, path, True


def _display(label):
    if label in DISPLAY_NAMES:
        return DISPLAY_NAMES[label]
    return str(label).replace("-", " ").title()


def _pct(value):
    return f"{100 * float(value):.1f}%"


def _pp_change(value):
    points = 100 * float(value)
    return f"{points:+.1f} pp"


def _dataset_description(manifest):
    dataset = manifest.get("dataset", "unknown")
    training = manifest.get("training_dataset")
    if training and training != dataset:
        return f"{dataset} (trained on {training})"
    return str(dataset)


def render_report(previous_manifest, new_manifest, classes, comparison, *,
                  validation_split_hash, validation_ids_hash, validation_rows,
                  previous_prediction_path, new_prediction_path,
                  previous_generated=False, new_generated=False):
    """Render the concise four-section Markdown report."""
    old, new = comparison["previous"], comparison["new"]
    metrics = [
        ("Balanced accuracy", old["balanced_accuracy"], new["balanced_accuracy"], True),
        ("Top-3 accuracy", old["top3_accuracy"], new["top3_accuracy"], True),
        ("MRR", old["mrr"], new["mrr"], False),
        ("Log loss", old["log_loss"], new["log_loss"], False),
    ]
    lines = [
        "# Checkpoint comparison",
        "",
        "## A. Overall comparison",
        "",
        f"- Model: `{previous_manifest['model']}`",
        f"- Previous: `{previous_manifest['experiment_id']}`; dataset `{_dataset_description(previous_manifest)}`; checkpoint `{Path(previous_manifest.get('checkpoint_path', 'N/A')).name}`.",
        f"- New: `{new_manifest['experiment_id']}`; dataset `{_dataset_description(new_manifest)}`; checkpoint `{Path(new_manifest.get('checkpoint_path', 'N/A')).name}`.",
        f"- Frozen V0 validation rows: {validation_rows}; classes: {len(classes)}; split SHA-256: `{validation_split_hash}`; validation IDs SHA-256: `{validation_ids_hash}`.",
        f"- Prediction source: previous `{Path(previous_prediction_path).name}`{' (generated)' if previous_generated else ''}; new `{Path(new_prediction_path).name}`{' (generated)' if new_generated else ''}.",
        "",
        "| Metric | Previous | New | Change |",
        "|---|---:|---:|---:|",
    ]
    for label, previous_value, new_value, percentage in metrics:
        if percentage:
            before, after, change = _pct(previous_value), _pct(new_value), _pp_change(new_value - previous_value)
        else:
            before, after = f"{previous_value:.4f}", f"{new_value:.4f}"
            change = f"{new_value - previous_value:+.4f}"
        lines.append(f"| {label} | {before} | {after} | {change} |")

    lines.extend(["", "## B. Per-model comparison", "",
                  "| Target model | Top-1 recall: previous → new | Top-3 accuracy: previous → new | Δ MRR (new − previous) |",
                  "|---|---:|---:|---:|"])
    for label in classes:
        p = old["per_class"][label]
        n = new["per_class"][label]
        name = _display(label)
        if label in WEAK_CLASSES:
            name = f"**{name}**"
        lines.append(f"| {name} | {_pct(p['top1_recall'])} → {_pct(n['top1_recall'])} | "
                     f"{_pct(p['top3_accuracy'])} → {_pct(n['top3_accuracy'])} | {n['mrr'] - p['mrr']:+.4f} |")

    lines.extend([
        "", "## C. Prediction movement", "",
        f"- Corrected top-1 predictions: **{comparison['corrected_top1']}**; regressed: **{comparison['regressed_top1']}**.",
        f"- True-label rank moved up: **{comparison['rank_up']}**; down: **{comparison['rank_down']}**; unchanged: **{comparison['rank_same']}**.",
        f"- Mean probability of the true label: {_pct(comparison['mean_true_probability_previous'])} → "
        f"{_pct(comparison['mean_true_probability_new'])} ({_pp_change(comparison['mean_true_probability_change'])}).",
        "", "## D. Conclusions", "",
    ])
    bacc_delta = new["balanced_accuracy"] - old["balanced_accuracy"]
    if bacc_delta > 0:
        lines.append(f"- Overall top-1 balanced accuracy improved by {_pp_change(bacc_delta)}.")
    elif bacc_delta < 0:
        lines.append(f"- Overall top-1 balanced accuracy declined by {_pp_change(bacc_delta)}.")
    else:
        lines.append("- Overall top-1 balanced accuracy was unchanged.")
    weak_improved = comparison["weak_classes_improved"]
    lines.append(f"- Of the five previously weak classes, {len(weak_improved)}/5 improved top-1 recall"
                 + (f" ({', '.join(_display(x) for x in weak_improved)})." if weak_improved else "."))
    if new["top3_accuracy"] > old["top3_accuracy"] and new["mrr"] > old["mrr"]:
        lines.append("- Ranking improved independently of top-1 through both top-3 accuracy and MRR.")
    elif new["top3_accuracy"] < old["top3_accuracy"] and new["mrr"] < old["mrr"]:
        lines.append("- Ranking weakened: both top-3 accuracy and MRR declined.")
    else:
        lines.append("- Ranking evidence is mixed: top-3 accuracy and MRR did not move together.")
    confidence = (new["log_loss"] < old["log_loss"]
                  and comparison["mean_true_probability_change"] > 0)
    lines.append("- Confidence quality improved." if confidence else
                 "- Confidence quality did not improve on both log loss and mean true-label probability.")
    preferable = bacc_delta > 0 or (bacc_delta == 0 and
                  (new["top3_accuracy"] > old["top3_accuracy"] or
                   (new["top3_accuracy"] == old["top3_accuracy"] and new["log_loss"] < old["log_loss"])))
    lines.append("- Prefer the new checkpoint on this frozen validation comparison." if preferable else
                 "- The new checkpoint is not preferable by the declared metric order (balanced accuracy, top-3, then log loss).")
    lines.extend(["", "These are development comparisons: the frozen V0 validation set has also been used for checkpoint selection.", ""])
    return "\n".join(lines)


def compare_checkpoints(previous, new, *, data_dir="data/processed", artifact_dir="artifacts",
                        registry_path="experiments/registry.sqlite",
                        output_path="docs/checkpoint_comparison.md", dataset="v0"):
    """Compare two manifests/experiment IDs and write the Markdown report."""
    from .pipeline import _load_dataset

    previous_manifest, previous_path = _load_manifest(
        previous, artifact_dir=artifact_dir, registry_path=registry_path)
    new_manifest, new_path = _load_manifest(new, artifact_dir=artifact_dir,
                                            registry_path=registry_path)
    if str(previous_manifest["model"]).lower() != str(new_manifest["model"]).lower():
        raise ValueError("checkpoint comparison requires two checkpoints of the same model")
    if dataset != "v0":
        raise ValueError("checkpoint comparison currently uses the frozen V0 validation split only")
    rows, _ = _load_dataset("v0", data_dir)
    validation = rows.loc[rows.split.eq("validation")].copy()
    validation = validation.sort_values("id", kind="stable").reset_index(drop=True)
    classes = sorted(rows.model.astype(str).unique().tolist())
    profile_path = Path(data_dir) / "dataset_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected_split_hash = str(profile["split_sha256"])
    expected_ids_hash = hashlib.sha256("\n".join(validation.id.astype(str)).encode()).hexdigest()
    if len(validation) != 540:
        raise ValueError(f"frozen V0 validation must contain 540 rows; found {len(validation)}")
    for label, manifest in (("previous", previous_manifest), ("new", new_manifest)):
        if manifest.get("classes") != classes:
            raise ValueError(f"{label} manifest class ordering differs from canonical V0 class order")
        if manifest.get("validation_split_sha256") not in (None, expected_split_hash):
            raise ValueError(f"{label} manifest references a different frozen V0 split")
        if manifest.get("id_sha256") not in (None, expected_ids_hash):
            raise ValueError(f"{label} manifest validation IDs differ from frozen V0")
    previous_frame, previous_prediction_path, previous_generated = _prediction_frame(
        previous_manifest, previous_path, validation, classes, artifact_dir)
    new_frame, new_prediction_path, new_generated = _prediction_frame(
        new_manifest, new_path, validation, classes, artifact_dir)
    aligned, previous_probs, new_probs = align_checkpoint_predictions(
        previous_frame, new_frame, classes, validation)
    comparison = compare_probability_matrices(
        aligned.y_true.to_numpy(dtype=str), previous_probs, new_probs, classes)
    report = render_report(previous_manifest, new_manifest, classes, comparison,
        validation_split_hash=expected_split_hash, validation_ids_hash=expected_ids_hash,
        validation_rows=len(validation), previous_prediction_path=previous_prediction_path,
        new_prediction_path=new_prediction_path, previous_generated=previous_generated,
        new_generated=new_generated)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {
        "report_path": str(output_path), "previous_experiment_id": previous_manifest["experiment_id"],
        "new_experiment_id": new_manifest["experiment_id"], "model": previous_manifest["model"],
        "previous_balanced_accuracy": comparison["previous"]["balanced_accuracy"],
        "new_balanced_accuracy": comparison["new"]["balanced_accuracy"],
        "previous_top3_accuracy": comparison["previous"]["top3_accuracy"],
        "new_top3_accuracy": comparison["new"]["top3_accuracy"],
        "previous_mrr": comparison["previous"]["mrr"], "new_mrr": comparison["new"]["mrr"],
        "previous_log_loss": comparison["previous"]["log_loss"],
        "new_log_loss": comparison["new"]["log_loss"],
        "prediction_generated": {"previous": previous_generated, "new": new_generated},
    }
