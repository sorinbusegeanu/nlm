"""Helpers for showing model training provenance in the dashboard."""
from __future__ import annotations

import json
from pathlib import Path


def _display_dataset(value) -> str:
    label = str(value).strip().replace("_", "-")
    if label[:1].lower() == "v":
        return "V" + label[1:]
    return label


def training_lineage(manifest: dict, manifest_dir: str | Path, _visited=None) -> str | None:
    """Render the dataset lineage, including parent checkpoints for continuations.

    Explicit staged sequences take precedence. Otherwise, continuations resolve
    their parent experiment manifest and append the current training dataset.
    """
    visited = set() if _visited is None else set(_visited)
    experiment_id = manifest.get("experiment_id")
    if experiment_id:
        if experiment_id in visited:
            return None
        visited.add(experiment_id)

    sequence = manifest.get("training_sequence")
    if sequence:
        labels = [_display_dataset(item) for item in sequence if item]
        return " → ".join(labels) if labels else None

    target = manifest.get("training_dataset") or manifest.get("dataset")
    parent_id = manifest.get("continued_from_experiment")
    if not parent_id:
        return _display_dataset(target) if target else None

    source_path = Path(manifest_dir) / f"{parent_id}_manifest.json"
    parent_lineage = None
    if source_path.is_file() and parent_id not in visited:
        try:
            parent = json.loads(source_path.read_text(encoding="utf-8"))
            parent_lineage = training_lineage(parent, manifest_dir, visited)
        except (OSError, ValueError, TypeError):
            parent_lineage = None

    parts = parent_lineage.split(" → ") if parent_lineage else ["continued"]
    if target:
        parts.append(_display_dataset(target))
    return " → ".join(parts) if parts else None
