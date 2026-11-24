#!/usr/bin/env python3
"""Create a frozen balanced V3.0 baseline and its ID-excluded candidate pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "data/processed/dataset_v3.0.parquet"
V3_PROFILE = ROOT / "data/processed/dataset_profile_v3.0.json"
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
V0_EXT_PATH = ROOT / "data/processed/dataset_v0_ext.parquet"
BASELINE_PATH = ROOT / "data/processed/dataset_v3.0_baseline.parquet"
PROFILE_PATH = ROOT / "data/processed/dataset_profile_v3.0_baseline.json"
CANDIDATE_PATH = ROOT / "data/processed/dataset_v3.0_baseline_candidates.parquet"
CANDIDATE_PROFILE_PATH = ROOT / "data/processed/dataset_profile_v3.0_baseline_candidates.json"
REPORT_PATH = ROOT / "docs/v3.0-baseline.md"
SEED = 2026


def norm(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def series_sha(values) -> str:
    h = hashlib.sha256()
    for value in pd.Series(values).astype(str):
        h.update(value.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


def prompt_values(frame: pd.DataFrame) -> set[str]:
    values: set[str] = set()
    for column in ("prompt_id", "prompt_canon", "prompt_group"):
        if column in frame:
            values.update(v for v in frame[column].map(norm) if v)
    return values


def select_baseline(v3: pd.DataFrame, labels: list[str], seed: int):
    """Select one response per prompt group, round-robin across each label's sources."""
    rng = np.random.default_rng(seed)
    high_quota_labels = set(labels[:10])
    quotas = {label: 56 if label in high_quota_labels else 55 for label in labels}
    selected_ids: list[str] = []
    selected_groups: set[str] = set()
    contributor_counts: dict[str, dict[str, int]] = {label: {} for label in labels}
    selected_by_label: dict[str, list[str]] = {label: [] for label in labels}
    indexed = v3.reset_index(drop=False).rename(columns={"index": "_source_row"})
    grouped_by_label: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}

    # Store each contributor's available prompt groups for each label, with source-row
    # positions for any repeated same-label responses from the same prompt/contributor.
    for label in labels:
        label_rows = indexed.loc[indexed.model.astype(str).eq(label)]
        by_source: dict[str, list[tuple[str, np.ndarray]]] = {}
        for (source, prompt), rows in label_rows.groupby(["source_dataset", "prompt_id"], sort=False):
            by_source.setdefault(str(source), []).append(
                (str(prompt), rows["_source_row"].to_numpy(dtype=np.int64)))
        for groups in by_source.values():
            rng.shuffle(groups)
        grouped_by_label[label] = by_source

    positions = {label: {source: 0 for source in by_source}
                 for label, by_source in grouped_by_label.items()}
    remaining = dict(quotas)
    # Interleave labels so one class cannot consume a prompt group needed by another.
    while any(count > 0 for count in remaining.values()):
        advanced = False
        for label in labels:
            if remaining[label] <= 0:
                continue
            by_source = grouped_by_label[label]
            available_sources: list[str] = []
            for source, groups in by_source.items():
                pos = positions[label][source]
                while pos < len(groups) and groups[pos][0] in selected_groups:
                    pos += 1
                positions[label][source] = pos
                if pos < len(groups):
                    available_sources.append(source)
            if not available_sources:
                raise ValueError(f"cannot meet quota for label {label}; selected {quotas[label] - remaining[label]}")
            minimum = min(contributor_counts[label].get(source, 0) for source in available_sources)
            tied_sources = [source for source in available_sources
                            if contributor_counts[label].get(source, 0) == minimum]
            source = str(rng.choice(tied_sources))
            prompt, row_indices = by_source[source][positions[label][source]]
            positions[label][source] += 1
            row_index = int(rng.choice(row_indices))
            response_id = str(v3.iloc[row_index].id)
            if prompt in selected_groups or response_id in selected_ids:
                raise AssertionError("selection attempted a repeated prompt group or response ID")
            selected_groups.add(prompt)
            selected_ids.append(response_id)
            selected_by_label[label].append(response_id)
            contributor_counts[label][source] = contributor_counts[label].get(source, 0) + 1
            remaining[label] -= 1
            advanced = True
        if not advanced:
            raise RuntimeError("baseline selection made no progress")
    return selected_ids, selected_groups, quotas, contributor_counts, selected_by_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing baseline artifacts")
    args = parser.parse_args()
    outputs = (BASELINE_PATH, PROFILE_PATH, CANDIDATE_PATH, CANDIDATE_PROFILE_PATH, REPORT_PATH)
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("V3.0 baseline artifacts already exist; pass --force only to intentionally refreeze")
    for path in (V3_PATH, V3_PROFILE, V0_PATH, V0_EXT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent_profile = json.loads(V3_PROFILE.read_text(encoding="utf-8"))
    if file_sha(V3_PATH) != parent_profile.get("dataset_sha256"):
        raise ValueError("V3.0 parquet does not match its recorded profile hash")
    v3 = pd.read_parquet(V3_PATH).reset_index(drop=True)
    v0 = pd.read_parquet(V0_PATH)
    v0_ext = pd.read_parquet(V0_EXT_PATH)
    labels = sorted(v0.model.astype(str).unique())
    if len(labels) != 18 or set(v3.model.astype(str).unique()) != set(labels):
        raise ValueError("V3.0 labels must exactly match the 18 canonical V0 labels")
    if len(v3) != int(parent_profile["rows"]) or v3.id.duplicated().any():
        raise ValueError("V3.0 row count or response-ID uniqueness check failed")

    ids, groups, quotas, contributor_counts, selected_by_label = select_baseline(v3, labels, SEED)
    baseline_ids = set(ids)
    baseline = v3.loc[v3.id.astype(str).isin(baseline_ids)].copy().reset_index(drop=True)
    # The separately materialized pool ensures subsequent contributor-label selection
    # cannot accidentally reintroduce baseline responses.
    candidates = v3.loc[~v3.id.astype(str).isin(baseline_ids)].copy().reset_index(drop=True)

    if len(baseline) != 1000 or baseline.id.nunique() != 1000:
        raise ValueError("baseline must contain exactly 1,000 unique responses")
    observed = baseline.model.astype(str).value_counts().reindex(labels, fill_value=0).to_dict()
    if observed != quotas:
        raise ValueError(f"label quotas differ from specification: {observed}")
    if baseline.prompt_id.astype(str).nunique() != 1000:
        raise ValueError("baseline must maximize prompt diversity by using 1,000 distinct prompt groups")

    v0_ids, ext_ids = set(v0.id.astype(str)), set(v0_ext.id.astype(str))
    v0_texts, ext_texts = set(v0.text.map(norm)), set(v0_ext.text.map(norm))
    v0_prompts, ext_prompts = prompt_values(v0), prompt_values(v0_ext)
    base_prompts = set(baseline.prompt_id.map(norm))
    checks = {
        "baseline_v0_response_id_overlap": len(baseline_ids & v0_ids),
        "baseline_v0_normalized_text_overlap": len(set(baseline.text.map(norm)) & v0_texts),
        "baseline_v0_known_prompt_group_overlap": len(base_prompts & v0_prompts),
        "baseline_v0_ext_response_id_overlap": len(baseline_ids & ext_ids),
        "baseline_v0_ext_normalized_text_overlap": len(set(baseline.text.map(norm)) & ext_texts),
        "baseline_v0_ext_known_prompt_group_overlap": len(base_prompts & ext_prompts),
        "candidate_pool_baseline_id_overlap": len(set(candidates.id.astype(str)) & baseline_ids),
    }
    if any(checks.values()):
        raise ValueError("integrity/overlap check failed: " + json.dumps(checks, sort_keys=True))

    original_columns = list(v3.columns)
    if not baseline[original_columns].equals(v3.set_index("id", drop=False).loc[baseline.id, original_columns].reset_index(drop=True)):
        raise ValueError("baseline does not preserve original V3.0 source columns and values")
    if len(candidates) + len(baseline) != len(v3):
        raise ValueError("candidate pool and baseline row accounting failed")

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(BASELINE_PATH, index=False, compression="zstd")
    candidates.to_parquet(CANDIDATE_PATH, index=False, compression="zstd")
    matrix = pd.crosstab(baseline.source_dataset.astype(str), baseline.model.astype(str))
    matrix = matrix.reindex(index=sorted(matrix.index), columns=labels, fill_value=0)
    matrix["Row total"] = matrix.sum(axis=1)
    matrix.loc["Column total"] = matrix.sum(axis=0)
    baseline_profile = {
        "dataset_id": "v3.0-baseline", "name": "V3.0-BASELINE", "rows": len(baseline),
        "seed": SEED, "parent_dataset_id": "v3.0", "parent_dataset_sha256": parent_profile["dataset_sha256"],
        "dataset_sha256": file_sha(BASELINE_PATH), "row_id_sha256": series_sha(baseline.id),
        "normalized_text_sha256": series_sha(baseline.text.map(norm)),
        "labels": labels, "label_counts": {label: int(observed[label]) for label in labels},
        "label_quotas": quotas, "contributor_counts_by_label": {
            label: dict(sorted(counts.items())) for label, counts in contributor_counts.items()},
        "distinct_contributors": int(baseline.source_dataset.nunique()),
        "distinct_prompt_groups": int(baseline.prompt_id.nunique()),
        "candidate_pool": {"artifact": str(CANDIDATE_PATH.relative_to(ROOT)), "rows": len(candidates),
            "dataset_sha256": file_sha(CANDIDATE_PATH), "baseline_ids_excluded": len(baseline_ids),
            "exclusion_rule": "remove V3.0 rows whose response ID is in V3.0-BASELINE; retain prompt-group siblings"},
        "overlap_checks": checks, "preserved_columns": original_columns,
        "selection": {"per_label_contributor_policy": "round-robin among contributors with unused prompt groups, independently within each label",
            "global_prompt_group_policy": "at most one selected response per prompt_id across all labels",
            "56_response_labels": labels[:10], "55_response_labels": labels[10:]},
        "frozen": True,
    }
    candidate_profile = {"dataset_id": "v3.0-baseline-candidates", "rows": len(candidates),
        "parent_dataset_id": "v3.0", "parent_dataset_sha256": parent_profile["dataset_sha256"],
        "dataset_sha256": file_sha(CANDIDATE_PATH), "excluded_baseline_ids": len(baseline_ids),
        "baseline_ids_sha256": series_sha(sorted(baseline_ids)), "overlap_with_baseline_ids": 0,
        "prompt_group_siblings_retained": True}
    PROFILE_PATH.write_text(json.dumps(baseline_profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CANDIDATE_PROFILE_PATH.write_text(json.dumps(candidate_profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    contributor_counts_table = baseline.groupby("source_dataset").size()
    lines = ["# V3.0-BASELINE", "", "A frozen, deterministic 1,000-response sample from V3.0. No model training or GPU inference was run.", "",
        "## Summary", "", "| Measure | Value |", "|---|---:|", f"| Total responses | {len(baseline):,} |",
        f"| Labels | {baseline.model.nunique()} |", f"| Distinct contributors | {baseline.source_dataset.nunique()} |",
        f"| Distinct prompt groups | {baseline.prompt_id.nunique():,} |", f"| Seed | {SEED} |",
        f"| Remaining V3.0 candidate responses | {len(candidates):,} |", "",
        "The first ten labels in sorted canonical label order have 56 responses each; the remaining eight have 55 each. One response per `prompt_id` was selected globally, giving the maximum possible 1,000 distinct prompt groups. Within each label, source contributors were selected round-robin before reuse where available.", "",
        "## Contributor–label distribution matrix", "", matrix.to_markdown(), "",
        "## Per-label contributor and prompt coverage", "", "| Label | Responses | Contributors | Prompt groups |", "|---|---:|---:|---:|"]
    for label in labels:
        sub = baseline.loc[baseline.model.astype(str).eq(label)]
        lines.append(f"| {label} | {len(sub)} | {sub.source_dataset.nunique()} | {sub.prompt_id.nunique()} |")
    lines += ["", "## Integrity and overlap verification", "", "| Check | Overlap rows |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in checks.items()]
    lines += ["", f"- V3.0 parent SHA-256: `{parent_profile['dataset_sha256']}`.",
        f"- Baseline SHA-256: `{baseline_profile['dataset_sha256']}`.",
        f"- Candidate-pool SHA-256: `{candidate_profile['dataset_sha256']}`.",
        f"- Candidate construction removes all {len(baseline_ids):,} baseline response IDs; the remaining {len(candidates):,} V3.0 rows are saved at [`{CANDIDATE_PATH.relative_to(ROOT)}`](../{CANDIDATE_PATH.relative_to(ROOT)}).",
        "- Candidate-pool construction excludes the baseline response IDs; other responses sharing those prompt groups remain available for contributor-label selection.",
        "- IDs, text, labels, prompt IDs, and source metadata are copied unchanged from V3.0.", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(baseline), "label_counts": observed,
        "distinct_contributors": int(baseline.source_dataset.nunique()),
        "distinct_prompt_groups": int(baseline.prompt_id.nunique()),
        "candidate_rows": len(candidates), "baseline_sha256": baseline_profile["dataset_sha256"],
        "candidate_sha256": candidate_profile["dataset_sha256"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
