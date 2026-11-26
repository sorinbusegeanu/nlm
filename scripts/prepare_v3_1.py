#!/usr/bin/env python3
"""Build V3.1 from Open Pool v16 plus selected verified v17 contributors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_factory import _normal_text
from src.v2_gr_or16 import LABELS, prompt_key

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/open_pool/v16/ntl_open_pool.parquet"
APPEND_SOURCE = ROOT / "data/raw/external_candidates/v3_external_audit_2026-11-25/crtproduction_v17/ntl_open_pool.parquet"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
V0_EXT_PATH = ROOT / "data/processed/dataset_v0_ext.parquet"
V3_PATH = ROOT / "data/processed/dataset_v3.0.parquet"
OUTPUT = ROOT / "data/processed/dataset_v3.1.parquet"
PROFILE = ROOT / "data/processed/dataset_profile_v3.1.json"
REPORT = ROOT / "docs/v3.1.md"
CONTRIBUTOR_LABEL_CSV = ROOT / "data/processed/v3.1_contributor_label_counts.csv"
EXPECTED_SOURCE_ROWS = 231_036
APPEND_CONTRIBUTORS = {
    "tadeuhayashida/ntl-gen003-claude-panel": 5_927,
    "ankiteshtiwari/ntl-blocks-v1": 1_027,
    "beroli/ntl-benchmark-c2-generated-panels": 432,
    "beroli/ntl-benchmark-c1-generated-panels": 328,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_v31(raw: pd.DataFrame, v0: pd.DataFrame, v0_ext: pd.DataFrame,
              additions: pd.DataFrame | None = None):
    """Exclude exact validation response matches; prompt groups never filter rows."""
    required_source = {"id", "text", "model", "prompt_canon", "prompt_group",
                       "source_dataset", "source_file"}
    required_reference = {"id", "text"}
    if required_source - set(raw):
        raise ValueError(f"Open Pool source lacks columns: {sorted(required_source - set(raw))}")
    for name, frame in (("V0", v0), ("V0-EXT", v0_ext)):
        if required_reference - set(frame):
            raise ValueError(f"{name} lacks columns: {sorted(required_reference - set(frame))}")
        if frame.id.isna().any() or frame.id.astype(str).duplicated().any():
            raise ValueError(f"{name} response IDs must be present and unique")
        if frame.text.isna().any():
            raise ValueError(f"{name} response text must be present")
    base = raw.copy().reset_index(drop=True)
    base["_v31_source_batch"] = "open_pool_v16"
    data = base if additions is None else pd.concat([
        base,
        additions.copy().reset_index(drop=True).assign(_v31_source_batch="selected_open_pool_v17_additions"),
    ], ignore_index=True, sort=False)
    data["id"] = data.id.astype("string")
    data["model"] = data.model.astype("string")
    if data.id.isna().any() or data.id.astype(str).duplicated().any():
        raise ValueError("Open Pool and appended response IDs must be present and unique")
    if data.text.isna().any() or data.text.astype(str).str.strip().eq("").any():
        raise ValueError("Open Pool response text must be present and nonempty")
    if set(data.model.dropna().astype(str)) != LABELS:
        raise ValueError("V3.1 source labels differ from the 18 canonical labels")

    id_values = data.id.astype(str)
    text_values = data.text.map(_normal_text)
    if text_values.eq("").any():
        raise ValueError("Open Pool contains empty normalized response text")
    if text_values.duplicated().any():
        duplicate_rows = int(text_values.duplicated(keep=False).sum())
        raise ValueError(f"normalized response duplicates across V3.1 source components: {duplicate_rows} rows")
    reasons: dict[str, pd.Series] = {}
    for name, frame in (("v0", v0), ("v0_ext", v0_ext)):
        ids = set(frame.id.astype(str))
        texts = set(frame.text.map(_normal_text)) - {""}
        reasons[f"{name}_response_id"] = id_values.isin(ids)
        reasons[f"{name}_normalized_text"] = text_values.isin(texts)
    excluded = pd.DataFrame(reasons).any(axis=1)

    # Keep the same canonical/fallback prompt ID convention required by the
    # training factory, but do not use it to filter rows.
    data["prompt_id"] = [prompt_key(row) for row in data.itertuples(index=False)]
    retained_additions = data.loc[
        ~excluded & data._v31_source_batch.ne("open_pool_v16")].copy()
    retained = data.loc[~excluded].reset_index(drop=True)
    retained = retained.drop(columns="_v31_source_batch")
    overlap = {
        name: int(retained_mask.sum())
        for name, retained_mask in {
            "v0_response_id": retained.id.astype(str).isin(set(v0.id.astype(str))),
            "v0_normalized_text": retained.text.map(_normal_text).isin(
                set(v0.text.map(_normal_text)) - {""}),
            "v0_ext_response_id": retained.id.astype(str).isin(set(v0_ext.id.astype(str))),
            "v0_ext_normalized_text": retained.text.map(_normal_text).isin(
                set(v0_ext.text.map(_normal_text)) - {""}),
        }.items()
    }
    if any(overlap.values()):
        raise RuntimeError(f"response-level overlap remains after filtering: {overlap}")
    if retained.model.isna().any() or set(retained.model.astype(str)) != LABELS:
        raise ValueError("V3.1 must retain all 18 canonical labels")

    stats = {
        "source_rows": int(len(raw)),
        "added_rows": int(0 if additions is None else len(additions)),
        "candidate_rows": int(len(data)),
        "rows": int(len(retained)),
        "contributors": int(retained.source_dataset.nunique()),
        "source_contributors": int(raw.source_dataset.nunique()),
        "candidate_source_contributors": int(data.source_dataset.nunique()),
        "fully_excluded_contributors": sorted(
            set(data.source_dataset.astype(str)) - set(retained.source_dataset.astype(str))),
        "fully_excluded_contributor_rows": {
            str(source): int(data.source_dataset.astype(str).eq(str(source)).sum())
            for source in sorted(set(data.source_dataset.astype(str)) - set(retained.source_dataset.astype(str)))},
        "excluded_rows": int(excluded.sum()),
        "exclusion_matches_by_reason": {key: int(mask.sum()) for key, mask in reasons.items()},
        "exclusion_intersections": {},
        "overlap_checks": overlap,
        "prompt_group_diagnostics": {
            "v0_overlapping_retained_rows": int(retained.prompt_id.astype(str).isin(
                set(v0.prompt_id.astype(str))).sum()) if "prompt_id" in v0 else 0,
            "v0_ext_overlapping_retained_rows": int(retained.prompt_id.astype(str).isin(
                set(v0_ext.prompt_id.astype(str))).sum()) if "prompt_id" in v0_ext else 0,
            "v0_ext_overlapping_prompt_groups": int(len(set(retained.prompt_id.astype(str)) &
                set(v0_ext.prompt_id.astype(str)))) if "prompt_id" in v0_ext else 0,
        },
        "label_counts": {key: int(value) for key, value in
                         retained.model.astype(str).value_counts().reindex(sorted(LABELS), fill_value=0).items()},
        "contributor_counts": {str(key): int(value) for key, value in
                               retained.source_dataset.astype(str).value_counts().sort_index().items()},
        "prompt_groups": int(retained.prompt_id.nunique()),
        "added_contributor_counts": ({str(key): int(value) for key, value in
            retained_additions.source_dataset.astype(str).value_counts().sort_index().items()}
            if additions is not None else {}),
    }
    return retained, stats


def _format_report(retained: pd.DataFrame, stats: dict[str, Any], *, hashes: dict[str, str],
                   v3: pd.DataFrame | None):
    labels = sorted(LABELS)
    lines = [
        "# V3.1 — Open Pool v16 plus selected v17 contributors", "",
        "Built from the original Open Pool v16 main parquet, then augmented with four specifically requested contributor datasets from the verified Open Pool v17 export. V0 and V0-EXT were read-only inputs and were not modified.", "",
        "## Dataset summary", "",
        "| Measure | V3.1 |", "|---|---:|",
        f"| Original Open Pool v16 responses | {stats['source_rows']:,} |",
        f"| Added Open Pool v17 contributor responses | {stats['added_rows']:,} |",
        f"| Total source responses before exclusions | {stats['candidate_rows']:,} |",
        f"| Retained responses | {stats['rows']:,} |",
        f"| Contributor datasets | {stats['contributors']:,} |",
        f"| Prompt groups (metadata only) | {stats['prompt_groups']:,} |",
        f"| Excluded responses (union of response ID/text matches) | {stats['excluded_rows']:,} |", "",
        "Prompt-group membership does not remove a response. The stable `prompt_id` is retained as metadata for later group-aware sampling.", "",
        "## Added contributor datasets", "",
        "| Contributor dataset | Added rows |", "|---|---:|",
    ]
    lines += [f"| {name} | {count:,} |" for name, count in stats["added_contributor_counts"].items()]
    lines += [
        "",
        "## Prompt-group overlap diagnostic (not an exclusion rule)", "",
        "| Validation dataset | Retained rows sharing a prompt ID | Shared prompt groups |",
        "|---|---:|---:|",
        f"| V0 | {stats['prompt_group_diagnostics']['v0_overlapping_retained_rows']:,} | 0 |",
        f"| V0-EXT | {stats['prompt_group_diagnostics']['v0_ext_overlapping_retained_rows']:,} | {stats['prompt_group_diagnostics']['v0_ext_overlapping_prompt_groups']:,} |",
        "",
        "V0-EXT prompt siblings are intentionally retained. The current pipeline's generic prompt-disjoint partition check will reject V3.1 paired with V0-EXT validation; response-level ID/text separation is clean, but V0-EXT evaluation must account for this prompt-group overlap. V0 has no prompt-ID overlap and remains the configured validation dataset.", "",
        "## Exclusions", "",
        "Reason counts can overlap; the union count above is the number of unique source rows excluded.", "",
        "| Match reason | Source rows matched |", "|---|---:|",
    ]
    lines += [f"| {name} | {count:,} |" for name, count in stats["exclusion_matches_by_reason"].items()]
    lines += ["", "## Contributor-label distribution", "",
              "The complete contributor-by-label counts are included below and in [`v3.1_contributor_label_counts.csv`](../data/processed/v3.1_contributor_label_counts.csv).", "",
              "| Contributor dataset | " + " | ".join(labels) + " | Total |",
              "|" + "---|" * (len(labels) + 2)]
    matrix = pd.crosstab(retained.source_dataset.astype(str), retained.model.astype(str)).reindex(
        columns=labels, fill_value=0)
    for source, row in matrix.iterrows():
        lines.append("| " + str(source) + " | " + " | ".join(f"{int(row[label]):,}" for label in labels) +
                     f" | {int(row.sum()):,} |")
    lines += ["", f"The original V16 source has {stats['source_contributors']} contributors; after adding four new sources, the combined candidate pool has {stats['candidate_source_contributors']}. "
              f"{stats['contributors']} contribute retained V3.1 rows; fully excluded contributor datasets and source rows: "
              f"{', '.join(f'{name} ({count} rows)' for name, count in stats['fully_excluded_contributor_rows'].items()) or 'none'}."]
    lines += ["", "## Label distribution", "", "| Label | V3.1 rows |", "|---|---:|"]
    lines += [f"| {label} | {stats['label_counts'][label]:,} |" for label in labels]
    lines += ["", "## Comparison with V3.0", ""]
    if v3 is not None:
        v3_ids = set(v3.id.astype(str))
        v31_ids = set(retained.id.astype(str))
        lines += [
            "| Measure | V3.0 | V3.1 | Change |", "|---|---:|---:|---:|",
            f"| Responses | {len(v3):,} | {len(retained):,} | {len(retained)-len(v3):+,} |",
            f"| Contributor datasets | {v3.source_dataset.nunique():,} | {stats['contributors']:,} | {stats['contributors']-v3.source_dataset.nunique():+,} |",
            f"| V3.0 response IDs present in V3.1 | — | {len(v3_ids & v31_ids):,} | — |", "",
            f"V3.1 contains {len(v31_ids - v3_ids):,} response IDs absent from V3.0. These include responses retained because V3.1 does not exclude siblings solely for sharing a V0-EXT prompt group.",
        ]
        v3_counts = v3.model.astype(str).value_counts().reindex(labels, fill_value=0)
        lines += ["", "| Label | V3.0 rows | V3.1 rows | Change |", "|---|---:|---:|---:|"]
        lines += [f"| {label} | {int(v3_counts[label]):,} | {stats['label_counts'][label]:,} | {stats['label_counts'][label]-int(v3_counts[label]):+,} |" for label in labels]
    else:
        lines += ["The existing V3.0 parquet was unavailable; row-level comparison could not be calculated."]
    lines += ["", "## Verification", "", "| Check | Overlap rows |", "|---|---:|"]
    lines += [f"| {key} | {value:,} |" for key, value in stats["overlap_checks"].items()]
    lines += ["", "All four response-level overlap checks are zero. V0 and V0-EXT were not written by this preparation script.", "",
              "## SHA-256", ""]
    lines += [f"- `{name}`: `{value}`" for name, value in hashes.items()]
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing V3.1 outputs")
    args = parser.parse_args()
    required_inputs = [SOURCE, APPEND_SOURCE, V0_PATH, V0_EXT_PATH]
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    if not args.force and any(path.exists() for path in (OUTPUT, PROFILE, REPORT, CONTRIBUTOR_LABEL_CSV)):
        raise FileExistsError("V3.1 output exists; inspect it or pass --force")
    source_hash, append_source_hash, v0_hash, ext_hash = map(
        file_sha256, (SOURCE, APPEND_SOURCE, V0_PATH, V0_EXT_PATH))
    raw, append_source, v0, v0_ext = (pd.read_parquet(path).reset_index(drop=True)
                                      for path in (SOURCE, APPEND_SOURCE, V0_PATH, V0_EXT_PATH))
    if len(raw) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS:,} Open Pool responses; found {len(raw):,}")
    if len(v0) != 2699 or len(v0_ext) != 4000:
        raise ValueError(f"unexpected frozen evaluation sizes: V0={len(v0)}, V0-EXT={len(v0_ext)}")
    append_source = append_source.loc[append_source.source_dataset.astype(str).isin(
        APPEND_CONTRIBUTORS)].copy().reset_index(drop=True)
    observed_additions = {str(key): int(value) for key, value in
                          append_source.source_dataset.astype(str).value_counts().sort_index().items()}
    if observed_additions != dict(sorted(APPEND_CONTRIBUTORS.items())):
        raise ValueError(f"verified v17 contributor counts changed: {observed_additions}")
    retained, stats = build_v31(raw, v0, v0_ext, append_source)
    if stats["added_contributor_counts"] != dict(sorted(APPEND_CONTRIBUTORS.items())):
        raise ValueError("one or more requested v17 rows were excluded or lost during V3.1 deduplication: "
                         + json.dumps(stats["added_contributor_counts"], sort_keys=True))
    stats["exclusion_intersections"] = {
        "V0_and_V0_EXT_ID_or_text": int(((pd.concat([raw, append_source], ignore_index=True).id.astype(str).isin(set(v0.id.astype(str))) |
            pd.concat([raw, append_source], ignore_index=True).text.map(_normal_text).isin(set(v0.text.map(_normal_text)))) &
            (pd.concat([raw, append_source], ignore_index=True).id.astype(str).isin(set(v0_ext.id.astype(str))) |
             pd.concat([raw, append_source], ignore_index=True).text.map(_normal_text).isin(set(v0_ext.text.map(_normal_text))))).sum())
    }
    v3 = pd.read_parquet(V3_PATH).reset_index(drop=True) if V3_PATH.is_file() else None
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".parquet.tmp")
    retained.to_parquet(temp, index=False, compression="zstd")
    temp.replace(OUTPUT)
    contributor_label_counts = pd.crosstab(
        retained.source_dataset.astype(str), retained.model.astype(str)).reindex(
            columns=sorted(LABELS), fill_value=0)
    contributor_label_counts.to_csv(CONTRIBUTOR_LABEL_CSV, index_label="source_dataset")
    hashes = {"open_pool_v16_source": source_hash,
              "open_pool_v17_additions_source": append_source_hash,
              "v0_frozen_input": v0_hash,
              "v0_ext_frozen_input": ext_hash,
              "v3.1_artifact": file_sha256(OUTPUT),
              "v3.1_contributor_label_counts": file_sha256(CONTRIBUTOR_LABEL_CSV)}
    profile = {
        "dataset_id": "v3.1", "name": "V3.1 — Open Pool v16 plus selected v17 contributors",
        "rows": stats["rows"], "contributors": stats["contributors"],
        "source_contributors": stats["source_contributors"],
        "candidate_source_contributors": stats["candidate_source_contributors"],
        "fully_excluded_contributors": stats["fully_excluded_contributors"],
        "fully_excluded_contributor_rows": stats["fully_excluded_contributor_rows"],
        "source": "Original Open Pool v16 main parquet plus four selected Open Pool v17 contributors",
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": source_hash,
        "append_source_path": str(APPEND_SOURCE.relative_to(ROOT)),
        "append_source_sha256": append_source_hash,
        "added_rows": stats["added_rows"],
        "added_contributor_counts": stats["added_contributor_counts"],
        "artifact_sha256": hashes["v3.1_artifact"],
        "dataset_sha256": hashes["v3.1_artifact"],
        "contributor_label_counts_path": str(CONTRIBUTOR_LABEL_CSV.relative_to(ROOT)),
        "contributor_label_counts_sha256": hashes["v3.1_contributor_label_counts"],
        "v0_sha256": v0_hash, "v0_ext_sha256": ext_hash,
        "row_id_sha256": ordered_digest(retained.id),
        "normalized_text_sha256": ordered_digest(retained.text.map(_normal_text)),
        "label_counts": stats["label_counts"], "contributor_counts": stats["contributor_counts"],
        "prompt_groups": stats["prompt_groups"],
        "selection": "all original Open Pool v16 responses plus all four requested Open Pool v17 contributors, except individual response ID or normalized-text matches against V0/V0-EXT",
        "normalization": "NFKC, casefold, collapsed whitespace (src.data_factory._normal_text)",
        "exclusions": {"unique_rows": stats["excluded_rows"],
                       "matching_rows_by_reason": stats["exclusion_matches_by_reason"],
                       "intersection_rows": stats["exclusion_intersections"]},
        "overlap_checks": stats["overlap_checks"],
        "prompt_group_diagnostics": stats["prompt_group_diagnostics"],
        "prompt_group_policy": "prompt_id retained as stable metadata; prompt-group membership is not an exclusion criterion",
        "external_only": True, "validation_dataset": "v0",
        "source_columns": list(raw.columns), "append_source_columns": list(append_source.columns),
        "output_columns": list(retained.columns),
        "comparison_v3.0": ({"rows": int(len(v3)),
                             "contributors": int(v3.source_dataset.nunique()),
                             "v3.0_ids_retained_in_v3.1": int(len(set(v3.id.astype(str)) & set(retained.id.astype(str)))),
                             "v3.1_ids_absent_from_v3.0": int(len(set(retained.id.astype(str)) - set(v3.id.astype(str))))}
                            if v3 is not None else None),
    }
    PROFILE.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(_format_report(retained, stats, hashes=hashes, v3=v3), encoding="utf-8")
    # Prove the immutable reference artifacts still have their input hashes.
    if file_sha256(V0_PATH) != v0_hash or file_sha256(V0_EXT_PATH) != ext_hash:
        raise RuntimeError("frozen V0 or V0-EXT changed during V3.1 preparation")
    print(json.dumps({"profile": str(PROFILE.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT)),
                      **stats, "hashes": hashes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
