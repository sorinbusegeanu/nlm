"""Build and read the V2.x-only Trust List 2.2 snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .v2_checkpoint_trust import _atomic_json, _atomic_write, _sort_and_rank, latest_trust_paths


LATEST_PATH = Path("docs/trust-list-2.2-latest.json")
VERSION_RE = re.compile(r"trust-list-2\.2-v(\d{4,})$")
V2_DATASET_RE = re.compile(r"^v2\.\d", re.IGNORECASE)


def latest_trust_list_2_2_paths(root: str | Path = "."):
    root = Path(root).resolve()
    pointer = root / LATEST_PATH
    if not pointer.is_file():
        return None
    metadata = json.loads(pointer.read_text(encoding="utf-8"))
    csv_path = root / metadata["csv_path"]
    metadata_path = root / metadata["metadata_path"]
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Trust List 2.2 latest pointer refers to missing files")
    return csv_path, metadata_path, json.loads(metadata_path.read_text(encoding="utf-8"))


def create_trust_list_2_2(*, root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    source = latest_trust_paths(root)
    if source is None:
        raise FileNotFoundError("Trust List 2 latest pointer is missing")
    source_csv, source_json, source_metadata = source
    frame = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    required = {"experiment_id", "training_dataset", "sequence_id", "target_label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"source trust list is missing columns: {sorted(missing)}")
    if int(source_metadata.get("validation_rows", 0)) != 2699:
        raise ValueError("source trust list does not use all 2,699 V0 rows")

    training_dataset = frame.training_dataset.astype(str).str.strip()
    selected = frame.loc[training_dataset.map(lambda value: bool(V2_DATASET_RE.match(value)))].copy()
    if selected.empty:
        raise ValueError("source Trust List 2 has no checkpoints trained on v2.x datasets")
    if selected.groupby("experiment_id").training_dataset.nunique().gt(1).any():
        raise ValueError("a checkpoint is assigned multiple training datasets in the source list")
    incomplete = selected.groupby("experiment_id").target_label.nunique().ne(18)
    if incomplete.any():
        raise ValueError("each included checkpoint must have a row for all 18 target labels")

    latest_22 = latest_trust_list_2_2_paths(root)
    source_hash = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    if latest_22 is not None and latest_22[2].get("parent_csv_sha256") == source_hash:
        # Training finalization may run after every experiment. Avoid creating
        # a new immutable 2.2 version when Trust List 2 itself did not change.
        unchanged = dict(latest_22[2])
        unchanged["status"] = "unchanged"
        return unchanged
    if latest_22 is None:
        next_number = 1
    else:
        current_version = str(latest_22[2].get("version", ""))
        match = VERSION_RE.fullmatch(current_version)
        if not match:
            raise ValueError(f"invalid Trust List 2.2 version in latest pointer: {current_version}")
        next_number = int(match.group(1)) + 1
    version = f"trust-list-2.2-v{next_number:04d}"
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    selected.attrs["trust_list_version"] = version
    ranked = _sort_and_rank(selected, generated_at=generated)

    datasets = sorted(ranked.training_dataset.astype(str).unique())
    per_dataset = {}
    for dataset in datasets:
        rows = ranked.loc[ranked.training_dataset.eq(dataset)]
        per_dataset[dataset] = {
            "checkpoints": int(rows.experiment_id.nunique()),
            "checkpoint_label_rows": int(len(rows)),
            "rule_eligible_rows": int(rows.rule_eligible.astype(str).str.lower().isin({"true", "1"}).sum()),
        }
    metadata = {
        "version": version,
        "trust_list_release": "2.2",
        "generated_at_utc": generated,
        "parent_version": source_metadata.get("version"),
        "parent_csv_path": str(source_csv.relative_to(root)),
        "parent_metadata_path": str(source_json.relative_to(root)),
        "parent_csv_sha256": source_hash,
        "filter": "Keep only rows whose recorded training_dataset matches ^v2\\.\\d; eligibility, precision, recall, support, and sorting use existing Trust List 2 rules.",
        "training_dataset_prefix": "v2.x",
        "validation_scope": source_metadata.get("validation_scope", "full_v0"),
        "validation_rows": int(source_metadata["validation_rows"]),
        "validation_sha256": source_metadata.get("validation_sha256"),
        "source_rows": int(len(frame)),
        "source_checkpoints": int(frame.experiment_id.nunique()),
        "filtered_rows": int(len(ranked)),
        "filtered_checkpoints": int(ranked.experiment_id.nunique()),
        "included_training_datasets": per_dataset,
        "rows": int(len(ranked)),
        "checkpoints": int(ranked.experiment_id.nunique()),
        "eligibility_method": "Recomputed with the same evaluation eligibility, zero validation-training overlap, and minimum support of 10 used by Trust List 2.",
    }
    csv_path = root / "docs" / f"{version}.csv"
    metadata_path = root / "docs" / f"{version}.json"
    _atomic_write(ranked, csv_path)
    metadata["csv_path"] = str(csv_path.relative_to(root))
    metadata["metadata_path"] = str(metadata_path.relative_to(root))
    _atomic_json(metadata, metadata_path)
    _atomic_json({"version": version, "csv_path": metadata["csv_path"],
                  "metadata_path": metadata["metadata_path"], "generated_at_utc": generated},
                 root / LATEST_PATH)
    _write_report(root, metadata, ranked)
    return metadata


def _write_report(root: Path, metadata: dict, frame: pd.DataFrame) -> None:
    lines = [
        "# Trust List 2.2",
        "",
        f"Version: `{metadata['version']}`. This list contains checkpoint-label scores only for checkpoints whose recorded training dataset is a V2.x dataset.",
        "",
        f"Source: [{metadata['parent_version']}](../{metadata['parent_csv_path']}) ({metadata['source_checkpoints']} checkpoints, {metadata['source_rows']:,} rows).",
        f"Validation: {metadata['validation_rows']:,} full V0 rows; SHA-256 `{metadata.get('validation_sha256')}`.",
        "",
        "The source Trust List 2 files remain unchanged. The dashboard reads the Trust List 2.2 latest pointer and does not display Trust List 1 or the general Trust List 2.",
        "",
        "## Included training datasets",
        "",
        "| Dataset | Checkpoints | Checkpoint-label rows | Rule-eligible rows |",
        "|---|---:|---:|---:|",
    ]
    for dataset, counts in metadata["included_training_datasets"].items():
        lines.append(f"| `{dataset}` | {counts['checkpoints']} | {counts['checkpoint_label_rows']:,} | {counts['rule_eligible_rows']:,} |")
    eligible = frame.rule_eligible.astype(str).str.lower().isin({"true", "1"})
    lines += [
        "",
        "## Totals",
        "",
        f"- Included checkpoints: {metadata['checkpoints']}",
        f"- Checkpoint-label rows: {metadata['rows']:,}",
        f"- Rule-eligible rows: {int(eligible.sum()):,}",
        f"- Excluded checkpoints: {metadata['source_checkpoints'] - metadata['checkpoints']}",
        "- Precision, recall, and support values are copied from Trust List 2; the existing eligibility threshold and sort order are reapplied.",
        "",
    ]
    (root / "docs" / "trust-list-2.2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(create_trust_list_2_2(root=args.root), indent=2))


if __name__ == "__main__":
    main()
