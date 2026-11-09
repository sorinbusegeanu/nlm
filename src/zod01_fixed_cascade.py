"""Run the user-specified fixed 18-label ZOD01 checkpoint cascade."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import Registry
from .trusted_checkpoint_ensemble import ULTRABERT_CHECKPOINTS


RULES = [
    ("gemini-3.1-pro-preview", "N0070"),
    ("gpt-6-astra", "N0072"),
    ("claude-opus-4-6", "N0058"),
    ("claude-opus-5", "N0082"),
    ("claude-opus-4-8", "D"),
    ("claude-sonnet-5", "N0058"),
    ("gemini-3.8-flash", "C"),
    ("qwen3.8-max", "C"),
    ("gpt-5.5", "N0065"),
    ("gpt-5.6-luna", "N0075"),
    ("gemini-3.6-flash", "N0083"),
    ("glm-5.3-flash", "N0031"),
    ("gpt-5.6-sol", "N0071"),
    ("gemini-3.7-flash", "N0082"),
    ("kimi-k3", "D"),
    ("gpt-5.6-terra", "N0057"),
    ("claude-fable-5", "A"),
    ("glm-5.3", "N0018"),
]


def _resolve_fallback(registry_path: str, requested_parent: str | None = None) -> dict:
    """Resolve the V1-all -> V0 UltraBERT run from recorded training lineage."""
    registry = Registry(registry_path)
    rows = registry.rows()
    parent_id = requested_parent or ULTRABERT_CHECKPOINTS["B"]
    matches = []
    for row in rows:
        if row["model"] != "ultrabert380":
            continue
        try:
            config = json.loads(row["config_json"])
            metrics = json.loads(row["metrics_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if config.get("continued_from_experiment") == parent_id and str(config.get("dataset")) == "v0":
            matches.append({**row, "config": config, "metrics": metrics})
    if len(matches) != 1:
        raise ValueError(f"Expected one V1-all -> V0 UltraBERT continuation for {parent_id}; found {len(matches)}")
    return matches[0]


def _load_prediction(path: Path, validation: pd.DataFrame, classes: list[str], candidate_id: str):
    frame = pd.read_parquet(path)
    if not {"id", "y_true", "prompt_id"}.issubset(frame.columns):
        raise ValueError(f"prediction cache lacks row identity columns: {path}")
    frame["id"] = frame["id"].astype(str)
    if frame.id.duplicated().any() or set(frame.id) != set(validation.id):
        raise ValueError(f"validation IDs do not match frozen V0 for {candidate_id}: {path}")
    frame = frame.set_index("id").loc[validation.id].reset_index()
    if not np.array_equal(frame.y_true.astype(str).to_numpy(), validation.y_true.to_numpy()):
        raise ValueError(f"true-label alignment mismatch for {candidate_id}")
    if not np.array_equal(frame.prompt_id.astype(str).to_numpy(), validation.prompt_id.to_numpy()):
        raise ValueError(f"prompt alignment mismatch for {candidate_id}")
    probability_columns = [f"prob::{label}" for label in classes]
    if not set(probability_columns).issubset(frame.columns):
        raise ValueError(f"class probability columns missing or out of schema for {candidate_id}")
    probs = frame[probability_columns].to_numpy(dtype=float)
    if probs.shape != (len(validation), len(classes)) or not np.isfinite(probs).all():
        raise ValueError(f"invalid probability matrix for {candidate_id}")
    if not np.allclose(probs.sum(axis=1), 1, atol=2e-4):
        raise ValueError(f"probabilities do not sum to one for {candidate_id}")
    return np.asarray(classes, dtype=object)[probs.argmax(axis=1)]


def run_fixed_zod01_cascade(*, data_dir="data/processed",
        inventory_path="artifacts/analysis/zod01_specialist_discovery/checkpoint_inventory.csv",
        predictions_dir="artifacts/analysis/zod01_specialist_discovery/predictions",
        registry_path="experiments/registry.sqlite",
        output_dir="artifacts/analysis/zod01_18label_cascade",
        report_path="docs/zod01_18label_cascade.md"):
    from .pipeline import _load_dataset

    rows, _ = _load_dataset("v0", data_dir)
    valid = rows.loc[rows.split.eq("validation"), ["id", "prompt_id", "model"]].copy()
    valid["id"] = valid.id.astype(str)
    valid["prompt_id"] = valid.prompt_id.astype(str)
    valid["y_true"] = valid.pop("model").astype(str)
    valid = valid.sort_values("id", kind="stable").reset_index(drop=True)
    if len(valid) != 540 or valid.prompt_id.nunique() != 30:
        raise ValueError(f"expected frozen V0 (540 rows, 30 groups), got {len(valid)} / {valid.prompt_id.nunique()}")
    classes = sorted(valid.y_true.unique().tolist())

    inventory = pd.read_csv(inventory_path).fillna("")
    key_rows = {str(row.candidate_key): row for row in inventory.itertuples(index=False)}
    for key in {key for _, key in RULES} - {"A", "C", "D"}:
        if key not in key_rows:
            raise ValueError(f"checkpoint key missing from inventory: {key}")
    fallback = _resolve_fallback(registry_path)
    fallback_id = fallback["experiment_id"]
    if fallback_id != ULTRABERT_CHECKPOINTS["D"]:
        raise ValueError(f"resolved fallback differs from selected checkpoint D: {fallback_id}")
    inventory_by_id = {str(row.experiment_id): row for row in inventory.itertuples(index=False)}

    candidate_ids = {}
    for _, key in RULES:
        candidate_ids[key] = ULTRABERT_CHECKPOINTS[key] if key in {"A", "C", "D"} else str(key_rows[key].experiment_id)
    candidate_ids["D"] = fallback_id
    prediction_by_id = {}
    for key, candidate_id in candidate_ids.items():
        row = inventory_by_id.get(candidate_id)
        if row is None:
            raise ValueError(f"no prediction inventory entry for {candidate_id} ({key})")
        prediction_path = Path(predictions_dir) / f"{candidate_id}_validation.parquet"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        prediction_by_id[candidate_id] = _load_prediction(prediction_path, valid, classes, candidate_id)

    assigned = np.zeros(len(valid), dtype=bool)
    final = np.empty(len(valid), dtype=object)
    selected = np.empty(len(valid), dtype=object)
    trace = []
    for order, (label, key) in enumerate(RULES, start=1):
        candidate_id = candidate_ids[key]
        pred = prediction_by_id[candidate_id]
        take = (~assigned) & (pred == label)
        idx = np.flatnonzero(take)
        assigned[idx] = True
        final[idx] = label
        selected[idx] = candidate_id
        correct = int(np.sum(valid.y_true.to_numpy()[idx] == label))
        trace.append({"order": order, "label": label, "checkpoint_key": key,
            "experiment_id": candidate_id, "assigned": int(len(idx)), "correct": correct,
            "incorrect": int(len(idx) - correct), "cumulative_assigned": int(assigned.sum()),
            "cumulative_coverage": float(assigned.mean()), "remaining": int((~assigned).sum())})

    remaining_idx = np.flatnonzero(~assigned)
    fallback_pred = prediction_by_id[fallback_id][remaining_idx]
    final[remaining_idx] = fallback_pred
    selected[remaining_idx] = fallback_id
    fallback_correct = int(np.sum(fallback_pred == valid.y_true.to_numpy()[remaining_idx]))
    trace.append({"order": "fallback", "label": "UltraBERT V1-all -> V0", "checkpoint_key": "D",
        "experiment_id": fallback_id, "assigned": int(len(remaining_idx)), "correct": fallback_correct,
        "incorrect": int(len(remaining_idx) - fallback_correct), "cumulative_assigned": len(valid),
        "cumulative_coverage": 1.0, "remaining": 0})

    y = valid.y_true.to_numpy()
    correct = int(np.sum(final == y))
    fallback_alone = prediction_by_id[fallback_id]
    fallback_correct_all = int(np.sum(fallback_alone == y))
    result = {"rows": len(valid), "correct": correct, "accuracy": correct / len(valid),
        "fallback_candidate_id": fallback_id, "fallback_lineage_parent": fallback["config"].get("continued_from_experiment"),
        "fallback_alone_correct": fallback_correct_all, "fallback_alone_accuracy": fallback_correct_all / len(valid),
        "delta_correct_vs_fallback": correct - fallback_correct_all,
        "validation_split_sha256": fallback.get("split_hash"), "prompt_groups": int(valid.prompt_id.nunique()),
        "rules": trace}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_frame = pd.DataFrame(trace)
    trace_frame.to_csv(output / "rule_trace.csv", index=False)
    decisions = valid[["id", "prompt_id", "y_true"]].copy()
    decisions["prediction"] = final
    decisions["correct"] = final == y
    decisions["selected_experiment_id"] = selected
    decisions.to_parquet(output / "predictions.parquet", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# ZOD01 Fixed 18-Label Cascade", "",
        "Fixed-rule cascade evaluated on the frozen V0 validation set (540 responses, 30 prompt groups). Checkpoint order and label assignments were used exactly as specified; no model training, threshold tuning, or checkpoint reselection was performed.", "",
        f"Fallback resolved through the experiment registry: `{fallback_id}`; parent checkpoint `{result['fallback_lineage_parent']}`. Validation split SHA-256: `{result['validation_split_sha256']}`.", "",
        "## Rule results", "",
        "| Order | Label | Checkpoint | Assigned | Correct | Incorrect | Cumulative coverage | Remaining |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in trace:
        lines.append(f"| {row['order']} | {row['label']} | `{row['checkpoint_key']}` | {row['assigned']} | {row['correct']} | {row['incorrect']} | {row['cumulative_coverage']:.1%} | {row['remaining']} |")
    lines.extend(["", "## Overall comparison", "",
        "| Method | Correct | Accuracy |", "|---|---:|---:|",
        f"| 18-label cascade | {correct}/540 | {correct/540:.2%} |",
        f"| UltraBERT V1-all → V0 fallback alone | {fallback_correct_all}/540 | {fallback_correct_all/540:.2%} |",
        f"| Difference | {correct-fallback_correct_all:+d} rows | {(correct-fallback_correct_all)/540:+.2%} |", "",
        f"{len(remaining_idx)} examples reached fallback after all 18 rules; fallback correctly classified {fallback_correct} of those.", "",
        "This is a development-set result: the saved checkpoint selection and specialist choices were informed by V0 validation performance.", "",
        f"Detailed per-row decisions and rule trace: `{output}`.", ""])
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines))
    return result

