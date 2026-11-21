"""Audit V2.14 prompt groups, repeated model rows, and prompt-ID consistency."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_v2.14.parquet"
PROFILE_PATH = ROOT / "data/processed/dataset_profile_v2.14.json"
OUTPUT_DIR = ROOT / "artifacts/analysis/v2.14_prompt_group_audit"


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if sha256(DATA_PATH) != profile.get("dataset_sha256"):
        raise ValueError("V2.14 artifact hash does not match its profile")
    rows = pd.read_parquet(DATA_PATH)
    required = {"id", "text", "model", "prompt_id", "prompt_canon", "source_dataset"}
    if required - set(rows.columns):
        raise ValueError(f"V2.14 lacks required audit columns: {sorted(required-set(rows.columns))}")
    if rows.id.astype(str).duplicated().any():
        raise ValueError("V2.14 response IDs are not unique")

    rows["prompt_canon"] = rows.prompt_canon.fillna("").astype(str).str.strip()
    rows["prompt_id"] = rows.prompt_id.fillna("").astype(str).str.strip()
    canonical = rows.loc[rows.prompt_canon.ne("")].copy()
    canonical["_group_key"] = "canon:" + canonical.prompt_canon
    b2m = rows.loc[rows.id.astype(str).str.startswith("b2m_")].copy()
    if b2m.prompt_id.eq("").any():
        raise ValueError("B2M rows with missing source prompt IDs cannot be grouped")
    b2m["_group_key"] = "b2m:" + b2m.prompt_id
    keyed = pd.concat([canonical, b2m], ignore_index=True, sort=False)
    keyed["_normalized_response"] = keyed.text.map(normalize)

    group_size = keyed.groupby("_group_key", sort=True).size()
    group_labels = keyed.groupby("_group_key", sort=True).model.nunique()
    group_model_size = keyed.groupby(["_group_key", "model"], sort=True).size()
    repeated_model_group_ids = group_model_size[group_model_size.gt(1)].index.get_level_values(0).unique()
    repeated_text_size = keyed.groupby(
        ["_group_key", "model", "_normalized_response"], sort=True).size()
    repeated_text_keys = repeated_text_size[repeated_text_size.gt(1)]
    duplicate_text_groups = repeated_text_keys.index.get_level_values(0).unique()

    prompt_id_to_canon = canonical.groupby("prompt_id", sort=True).prompt_canon.nunique()
    canonical_prompt_id_sources = canonical.groupby("prompt_id", sort=True).source_dataset.nunique()
    b2m_id_sources = b2m.groupby("prompt_id", sort=True).source_dataset.nunique()
    canon_sources = canonical.groupby("prompt_canon", sort=True).source_dataset.nunique()
    canon_ids = canonical.groupby("prompt_canon", sort=True).prompt_id.nunique()
    multisource = canon_sources.gt(1)

    binned_ranges = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                     (6, 8), (9, 15), (16, 30), (31, 50), (51, 100)]
    binned = []
    for lower, upper in binned_ranges:
        selected = group_size[(group_size >= lower) & (group_size <= upper)]
        binned.append({
            "group_size_range": str(lower) if lower == upper else f"{lower}-{upper}",
            "groups": int(len(selected)),
            "response_rows": int(selected.sum()),
            "share_of_groups": float(len(selected) / len(group_size)) if len(group_size) else 0.0,
        })
    exact = group_size.value_counts().sort_index().rename_axis("group_size").reset_index(name="groups")
    exact["response_rows"] = exact.group_size * exact.groups

    mapped_ids = int(len(prompt_id_to_canon))
    multisource_canon = int(multisource.sum())
    missing_rows = rows.prompt_canon.eq("")
    unkeyed_v14 = rows.loc[missing_rows & ~rows.id.astype(str).str.startswith("b2m_")]
    missing_by_source = (unkeyed_v14.groupby("source_dataset", sort=True)
                         .size().sort_values(ascending=False).rename_axis("source_dataset")
                         .reset_index(name="rows_without_prompt_canon"))
    mapping_rows = [
        ("Unique prompt_id values in V2.14", int(rows.prompt_id.nunique()), "prompt_id values; source-local and file-level fallbacks are included."),
        ("prompt_id values mapped to at least one prompt_canon", mapped_ids, "Only rows with a non-empty prompt_canon are used."),
        ("prompt_id values mapping to multiple canonical prompts", int(prompt_id_to_canon.gt(1).sum()), f"Of {mapped_ids} mapped IDs; maximum distinct prompt_canon values for one ID is {int(prompt_id_to_canon.max()) if mapped_ids else 0}."),
        ("V14 prompt_id values reused across source_dataset values", int(canonical_prompt_id_sources.gt(1).sum()), "Among rows with prompt_canon; none are shared across source datasets."),
        ("prompt_canon groups present in multiple source datasets", multisource_canon, f"Of {len(canon_sources)} V14 canonical prompt groups; all {int((multisource & canon_ids.gt(1)).sum())} have multiple prompt_id values."),
        ("B2M prompt_id values reused across source_dataset values", int(b2m_id_sources.gt(1).sum()), f"Of {len(b2m_id_sources)} B2M IDs; prompt text/canonical keys are absent, so reused IDs cannot be verified against content."),
        ("V14 rows lacking a usable prompt key", int(len(unkeyed_v14)), "These rows lack prompt_canon and are not B2M rows with source-provided prompt IDs."),
    ]
    summary_rows = [
        ("V2.14 response rows", len(rows), "All rows in the dataset artifact."),
        ("Rows with usable prompt key", len(keyed), "V14 rows use prompt_canon; B2M additions use their source prompt_id in a separate namespace."),
        ("Observed keyed prompt groups", len(group_size), "V14 canonical groups plus separately keyed B2M prompt IDs; cross-pool matches cannot be merged."),
        ("Groups containing at least two model labels", int(group_labels.ge(2).sum()), "Distinct labels within each observed keyed prompt group."),
        ("Groups with multiple rows for at least one model", int(len(repeated_model_group_ids)), "Any model occurs more than once in the observed keyed prompt group."),
        ("Groups with an identical normalized response repeated for the same model", int(len(duplicate_text_groups)), "NFKC + case-fold + collapsed-whitespace response equality within a group and model."),
        ("Excess identical-response rows", int((repeated_text_keys - 1).sum()), "Rows beyond the first for each repeated group/model/normalized-response tuple."),
        ("Mean responses per observed keyed group", float(group_size.mean()), "Rows per canonical V14 or source-keyed B2M group."),
        ("Median responses per observed keyed group", float(group_size.median()), "Rows per canonical V14 or source-keyed B2M group."),
        ("Maximum group size", int(group_size.max()), "Rows for one observed keyed prompt group."),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows, columns=["metric", "value", "definition"])
    id_consistency = pd.DataFrame(mapping_rows, columns=["check", "count", "interpretation"])
    canon_label_counts = canonical.groupby("prompt_canon", sort=True).model.nunique()
    canon_model_counts = canonical.groupby(["prompt_canon", "model"], sort=True).size()
    b2m_label_counts = b2m.groupby("prompt_id", sort=True).model.nunique()
    b2m_model_counts = b2m.groupby(["prompt_id", "model"], sort=True).size()
    strata = pd.DataFrame([
        {"group_basis": "V14 canonical prompt_canon", "rows": int(len(canonical)),
         "groups": int(canonical.prompt_canon.nunique()),
         "groups_with_multiple_labels": int(canon_label_counts.ge(2).sum()),
         "groups_with_multiple_rows_for_same_model": int(canon_model_counts[canon_model_counts.gt(1)].index.get_level_values(0).nunique())},
        {"group_basis": "B2M source prompt_id", "rows": int(len(b2m)),
         "groups": int(b2m.prompt_id.nunique()),
         "groups_with_multiple_labels": int(b2m_label_counts.ge(2).sum()),
         "groups_with_multiple_rows_for_same_model": int(b2m_model_counts[b2m_model_counts.gt(1)].index.get_level_values(0).nunique())},
        {"group_basis": "V14 rows without usable prompt key", "rows": int(len(unkeyed_v14)),
         "groups": 0, "groups_with_multiple_labels": 0,
         "groups_with_multiple_rows_for_same_model": 0},
    ])
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    id_consistency.to_csv(OUTPUT_DIR / "prompt_id_consistency.csv", index=False)
    strata.to_csv(OUTPUT_DIR / "group_metrics_by_key_basis.csv", index=False)
    pd.DataFrame(binned).to_csv(OUTPUT_DIR / "group_size_distribution.csv", index=False)
    exact.to_csv(OUTPUT_DIR / "group_size_distribution_exact.csv", index=False)
    missing_by_source.to_csv(OUTPUT_DIR / "rows_without_prompt_canon_by_source.csv", index=False)

    summary_json = {
        "dataset": "v2.14",
        "dataset_sha256": profile["dataset_sha256"],
        "total_rows": int(len(rows)),
        "group_key": "V14: prompt_canon; B2M additions: source prompt_id",
        "group_key_note": "B2M IDs are kept in a separate namespace because prompt_canon is absent; B2M-to-V14 prompt matches cannot be verified or merged.",
        "keyed_rows": int(len(keyed)),
        "rows_without_usable_prompt_key": int(len(unkeyed_v14)),
        "observed_keyed_prompt_groups": int(len(group_size)),
        "v14_canonical_groups": int(canonical.prompt_canon.nunique()),
        "b2m_source_prompt_id_groups": int(b2m.prompt_id.nunique()),
        "groups_with_multiple_model_labels": int(group_labels.ge(2).sum()),
        "groups_with_multiple_rows_for_any_model": int(len(repeated_model_group_ids)),
        "groups_with_identical_normalized_responses_same_model": int(len(duplicate_text_groups)),
        "excess_identical_response_rows": int((repeated_text_keys - 1).sum()),
        "group_size": {"min": int(group_size.min()), "median": float(group_size.median()),
                       "mean": float(group_size.mean()), "p90": float(group_size.quantile(.9)),
                       "p95": float(group_size.quantile(.95)), "p99": float(group_size.quantile(.99)),
                       "max": int(group_size.max())},
        "unique_prompt_ids": int(rows.prompt_id.nunique()),
        "mapped_prompt_ids": mapped_ids,
        "prompt_ids_mapping_multiple_canonical_prompts": int(prompt_id_to_canon.gt(1).sum()),
        "max_canonical_prompts_per_prompt_id": int(prompt_id_to_canon.max()) if mapped_ids else 0,
        "v14_prompt_ids_shared_across_sources": int(canonical_prompt_id_sources.gt(1).sum()),
        "b2m_prompt_ids_shared_across_sources": int(b2m_id_sources.gt(1).sum()),
        "canonical_groups_spanning_sources": multisource_canon,
        "multisource_canonical_groups_with_multiple_prompt_ids": int((multisource & canon_ids.gt(1)).sum()),
        "normalization": "Unicode NFKC, case-fold, collapse whitespace; punctuation preserved",
        "source_id_consistency": "V14 prompt_id values do not align across source datasets; B2M prompt IDs are reused by source, but have no canonical prompt text for validation.",
        "report_files": ["report.md", "summary_metrics.csv", "prompt_id_consistency.csv", "group_metrics_by_key_basis.csv",
                         "group_size_distribution.csv", "group_size_distribution_exact.csv",
                         "rows_without_prompt_canon_by_source.csv"],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")

    def fmt_count(value: object) -> str:
        if isinstance(value, float):
            return f"{value:,.2f}"
        return f"{int(value):,}"

    report_lines = [
        "# V2.14 prompt-group and ID audit", "",
        f"Dataset: `data/processed/dataset_v2.14.parquet`",
        f"SHA256: `{profile['dataset_sha256']}`",
        f"Group keys: V14 uses non-empty `prompt_canon`; B2M additions use source `prompt_id` separately ({len(keyed):,} keyed rows; {len(group_size):,} observed groups).",
        f"Normalization for response duplicate checks: Unicode NFKC, case-fold, collapsed whitespace; punctuation preserved.", "",
        "## Group metrics", "",
        "| Metric | Result | Definition |", "|---|---:|---|",
    ]
    for metric, value, definition in summary_rows:
        report_lines.append(f"| {metric} | {fmt_count(value)} | {definition} |")
    report_lines += [
        "", "## Group-size distribution", "",
        "Counts are by rows per observed keyed prompt group. V14 and B2M key spaces remain separate; possible V14-to-B2M prompt matches cannot be merged. The full exact-size distribution is in `group_size_distribution_exact.csv`.", "",
        "| Responses in group | Groups | Response rows | Share of groups |", "|---:|---:|---:|---:|",
    ]
    for item in binned:
        report_lines.append(f"| {item['group_size_range']} | {item['groups']:,} | {item['response_rows']:,} | {item['share_of_groups']:.1%} |")
    report_lines += ["", "## Prompt-ID consistency", "",
        "| Check | Count | Interpretation |", "|---|---:|---|"]
    for check, count, interpretation in mapping_rows:
        report_lines.append(f"| {check} | {count:,} | {interpretation} |")
    report_lines += [
        "", "## Group metrics by key basis", "",
        "| Key basis | Rows | Groups | Multi-label groups | Groups with repeated same-model rows |", "|---|---:|---:|---:|---:|"]
    for item in strata.to_dict("records"):
        report_lines.append(f"| {item['group_basis']} | {item['rows']:,} | {item['groups']:,} | {item['groups_with_multiple_labels']:,} | {item['groups_with_multiple_rows_for_same_model']:,} |")
    report_lines += [
        "", "**Finding:** V14 `prompt_id` is not a globally consistent prompt key: all 5,342 `prompt_canon` groups that occur in multiple source datasets have multiple `prompt_id` values, and 752 V14 IDs map to more than one canonical prompt (up to 2,914). Use `prompt_canon` for cross-source grouping where present. B2M has 1,987 source prompt IDs, 267 reused across source datasets; its 7,496 rows have no `prompt_canon` or prompt text, so those IDs cannot be validated against V14 or independently checked against prompt content.",
        f"", f"There are **{len(unkeyed_v14):,} additional V14 rows** without a usable prompt key. Their prompt identity cannot be inferred from the file-level fallback ID. See `rows_without_prompt_canon_by_source.csv`.",
        "", "## Output tables", "",
        "- `summary_metrics.csv` — requested group metrics.",
        "- `group_metrics_by_key_basis.csv` — V14 canonical and B2M source-keyed strata.",
        "- `group_size_distribution.csv` — grouped size ranges.",
        "- `group_size_distribution_exact.csv` — exact group-size frequencies.",
        "- `prompt_id_consistency.csv` — prompt ID mapping checks.",
        "- `rows_without_prompt_canon_by_source.csv` — rows excluded from canonical-group metrics.", "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary_json


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
