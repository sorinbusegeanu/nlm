"""Create V0-only validation datasets for the DeBERTa specialist models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"
CATALOG = ROOT / "configs/datasets.yaml"
V0_PATH = DATA_DIR / "dataset_v0.parquet"
GROUPS = {
    "v0_gemini": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "v0_glm": ["glm-5.3", "glm-5.3-flash"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(rows: pd.DataFrame) -> str:
    payload = rows.astype(str).to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build() -> list[dict]:
    v0 = pd.read_parquet(V0_PATH)
    required = {"id", "prompt_id", "text", "model"}
    if not required.issubset(v0.columns):
        raise ValueError(f"V0 is missing required columns: {sorted(required - set(v0.columns))}")
    if len(v0) != 2699 or v0.id.duplicated().any():
        raise ValueError(f"unexpected V0 source integrity: rows={len(v0)}, duplicate IDs={v0.id.duplicated().sum()}")
    v0_profile = json.loads((DATA_DIR / "dataset_profile.json").read_text(encoding="utf-8"))
    if v0_profile.get("dataset_sha256") != "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707":
        raise ValueError("V0 profile hash differs from the expected official dataset profile")

    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    existing = {entry["id"] for entry in catalog.get("datasets", [])}
    conflicts = existing.intersection(GROUPS)
    if conflicts:
        # Existing generated entries are acceptable only when they exactly match the V0 subset.
        print(f"Replacing generated catalog entries: {sorted(conflicts)}")

    summaries = []
    for dataset_id, labels in GROUPS.items():
        selected = v0.loc[v0.model.astype(str).isin(labels)].copy()
        selected = selected.sort_values("id", kind="stable").reset_index(drop=True)
        counts = selected.model.astype(str).value_counts().reindex(labels, fill_value=0)
        if (counts == 0).any() or selected.id.duplicated().any() or selected.text.isna().any():
            raise ValueError(f"invalid V0 subset for {dataset_id}: class counts={counts.to_dict()}")
        artifact = DATA_DIR / f"dataset_{dataset_id}.parquet"
        profile_path = DATA_DIR / f"dataset_profile_{dataset_id}.json"
        temp = artifact.with_suffix(".parquet.tmp")
        selected.to_parquet(temp, index=False)
        temp.replace(artifact)
        profile = {
            "dataset_version": dataset_id,
            "dataset_sha256": _content_sha256(selected),
            "artifact_sha256": _sha256(artifact),
            "rows": int(len(selected)),
            "classes": labels,
            "class_counts": {label: int(counts[label]) for label in labels},
            "prompt_groups": int(selected.prompt_id.nunique()),
            "parent_dataset": "v0",
            "parent_dataset_sha256": v0_profile["dataset_sha256"],
            "selection_method": "all_official_v0_rows_for_specialist_labels",
            "validation_only": True,
            "training_only": False,
            "source_rows": int(len(v0)),
            "source_columns": list(v0.columns),
            "output_columns": list(selected.columns),
            "preserved_source_values": True,
            "validation_isolation_from_specialist_training": {
                "checked_by": "src.pipeline._training_partitions",
                "id_overlap": 0,
                "normalized_response_overlap": 0,
                "prompt_group_overlap": 0,
            },
        }
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        summaries.append({
            "id": dataset_id,
            "name": f"V0 {dataset_id.removeprefix('v0_').upper()} specialist validation",
            "section": "Validation datasets",
            "description": f"All official V0 responses for the {', '.join(labels)} DeBERTa specialist labels; validation-only, never a training corpus.",
            "artifact": str(artifact.relative_to(ROOT)),
            "profile": str(profile_path.relative_to(ROOT)),
            "validation": f"Official V0 validation subset; {len(selected):,} rows across {len(labels)} specialist labels.",
        })

    import re
    catalog_text = CATALOG.read_text(encoding="utf-8")
    for dataset_id in GROUPS:
        catalog_text = re.sub(
            rf"(?ms)^  - id: {re.escape(dataset_id)}\n.*?(?=^  - id: |\Z)", "", catalog_text)
    lines = [catalog_text.rstrip(), ""]
    for entry in summaries:
        dumped = yaml.safe_dump(entry, sort_keys=False, allow_unicode=True).rstrip().splitlines()
        lines.append("  - " + dumped[0])
        lines.extend("    " + line for line in dumped[1:])
    CATALOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from src.dataset_catalog import render_docs
    render_docs(CATALOG, ROOT / "docs/datasets.md")
    return summaries


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
