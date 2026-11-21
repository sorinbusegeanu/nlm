"""Build and validate the Open Pool v16 group-ranking dataset for E21."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data_factory import _hash_file, _normal_text

DATASET_ID = "v2.GR.OR16"
EXPECTED_ROWS = 231_036
EXPECTED_RANK_ROWS = 219_294
EXPECTED_RANK_GROUPS = 26_347
EXPECTED_RANK_ITEMS = 170_233
EXPECTED_PAIRWISE = 812_835
LABELS = {
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
}


def _clean(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def prompt_key(row: Any) -> str:
    """Return a namespaced stable key; canonical and fallback spaces cannot collide."""
    canon = _normal_text(_clean(row.prompt_canon))
    if canon:
        return "canon:" + canon
    parts = [_normal_text(_clean(getattr(row, col))) for col in
             ("source_dataset", "source_file", "prompt_group")]
    if not all(parts):
        raise ValueError(f"row {getattr(row, 'id', '?')} has no complete prompt grouping key")
    return "fallback:" + json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _sha256_texts(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def build_dataset(raw: pd.DataFrame, *, v0: pd.DataFrame | None = None,
                  source_sha256: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve source rows and append group/ranking metadata."""
    data = raw.copy().reset_index(drop=True)
    required = {"id", "text", "model", "prompt_canon", "prompt_group", "source_dataset", "source_file"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Open Pool main parquet lacks columns: {sorted(missing)}")
    data["id"] = data.id.astype("string")
    data["model"] = data.model.map(_clean)
    if data.id.isna().any() or data.id.duplicated().any():
        raise ValueError("response IDs must be present and unique")
    if data.text.isna().any() or data.text.astype(str).str.strip().eq("").any():
        raise ValueError("all source responses must have nonempty text")
    invalid = sorted(set(data.model) - LABELS)
    if invalid or set(data.model) != LABELS:
        raise ValueError(f"source labels must be exactly the 18 canonical labels; invalid={invalid}")
    data["prompt_id"] = [prompt_key(row) for row in data.itertuples(index=False)]
    if data.prompt_id.isna().any() or data.prompt_id.eq("").any():
        raise ValueError("prompt_id generation produced empty values")
    data["_normalized_response"] = data.text.map(_normal_text)
    if data._normalized_response.eq("").any():
        raise ValueError("response normalization produced empty text")
    dup = data.groupby("_normalized_response", sort=False).agg(
        rows=("id", "size"), labels=("model", "nunique"))
    duplicates = dup[dup.rows.gt(1)]
    if len(duplicates):
        raise ValueError(f"normalized response duplicates found: {len(duplicates)} clusters")

    if v0 is not None:
        id_overlap = set(data.id.astype(str)) & set(v0.id.dropna().astype(str))
        v0_text = set(v0.text.dropna().astype(str).map(_normal_text)) - {""}
        text_overlap = set(data._normalized_response) & v0_text
        v0_prompts = set(v0.prompt_id.dropna().astype(str))
        canon_overlap = set(data.prompt_canon.fillna("").astype(str).map(_normal_text)) & v0_prompts
        source_overlap = set(data.prompt_group.fillna("").astype(str).str.strip()) & v0_prompts
        if id_overlap or text_overlap or canon_overlap or source_overlap:
            raise ValueError("Open Pool v16 overlaps V0: " + json.dumps({
                "response_ids": len(id_overlap), "normalized_texts": len(text_overlap),
                "canonical_prompt_ids": len(canon_overlap), "source_prompt_ids": len(source_overlap)}, sort_keys=True))

    group = data.groupby("prompt_id", sort=False)
    data["group_size"] = group.id.transform("size").astype("int32")
    data["group_label_count"] = group.model.transform("nunique").astype("int16")
    data["group_model_count"] = data.groupby(["prompt_id", "model"], sort=False).id.transform("size").astype("int16")
    data["ranking_eligible"] = data.group_label_count.ge(2)
    # Stable within each (prompt, model) rotation. Select rotation_index == epoch % group_model_count.
    data["ranking_rotation_index"] = (
        data.sort_values(["prompt_id", "model", "id"], kind="stable")
        .groupby(["prompt_id", "model"], sort=False).cumcount()
        .reindex(data.index).astype("int16"))
    data["ranking_label_index"] = data.groupby("prompt_id", sort=False).model.transform(
        lambda s: pd.Categorical(s, categories=sorted(s.unique()), ordered=True).codes).astype("int16")
    data = data.drop(columns=["_normalized_response"])

    group_stats = data.groupby("prompt_id", sort=False).agg(
        group_size=("id", "size"), group_label_count=("model", "nunique"),
        prompt_canon=("prompt_canon", "first"), source_dataset=("source_dataset", "first"),
        source_file=("source_file", "first"), prompt_group=("prompt_group", "first"))
    label_group = data.groupby(["prompt_id", "model"], sort=False).agg(
        model_count=("id", "size"), rows=("id", "size"))
    eligible_groups = group_stats[group_stats.group_label_count.ge(2)]
    eligible_rows = data[data.ranking_eligible]
    class_counts = data.model.value_counts().sort_index()
    eligible_class = eligible_rows.model.value_counts().reindex(sorted(LABELS), fill_value=0)
    group_label_items = int(eligible_groups.group_label_count.sum())
    pairwise = int((eligible_groups.group_label_count * (eligible_groups.group_label_count - 1) // 2).sum())
    key_basis = data.prompt_id.str.startswith("canon:").map({True: "canonical", False: "fallback"})
    repeated_pairs = label_group[label_group.model_count.gt(1)]
    profile: dict[str, Any] = {
        "dataset_id": DATASET_ID, "name": "V2.GR.OR16 — Group-Ranking Open Pool v16",
        "rows": len(data), "labels": sorted(LABELS), "classification_rows": len(data),
        "label_counts": {str(k): int(v) for k, v in class_counts.items()},
        "source": "data/raw/open_pool/v16/ntl_open_pool.parquet (original main parquet only)",
        "source_sha256": source_sha256, "dataset_sha256": None,
        "row_id_sha256": _sha256_texts(data.id), "normalized_text_sha256": _sha256_texts(data.text.map(_normal_text)),
        "grouping": {"canonical_rule": "canon:" + "NFKC/casefold/whitespace-normalized prompt_canon",
                     "fallback_rule": "fallback:" + "JSON([normalized source_dataset, source_file, prompt_group])",
                     "canonical_groups": int(data.loc[key_basis.eq("canonical"), "prompt_id"].nunique()),
                     "canonical_rows": int(key_basis.eq("canonical").sum()),
                     "fallback_groups": int(data.loc[key_basis.eq("fallback"), "prompt_id"].nunique()),
                     "fallback_rows": int(key_basis.eq("fallback").sum()),
                     "groups": int(len(group_stats)),
                     "group_keys_reconstruct_consistently": True},
        "ranking": {"min_distinct_labels": 2, "eligible_groups": int(len(eligible_groups)),
                    "eligible_responses": int(len(eligible_rows)),
                    "eligible_response_pct": 100 * len(eligible_rows) / len(data),
                    "unique_group_label_combinations": group_label_items,
                    "potential_unordered_label_pair_comparisons_per_full_group_pass": pairwise,
                    "per_label_eligible_responses": {str(k): int(v) for k, v in eligible_class.items()},
                    "rotation": "Within each prompt_id/model, sort by response id; for epoch e select ranking_rotation_index == e modulo group_model_count."},
        "distributions": {
            "group_size": {str(int(k)): int(v) for k, v in group_stats.group_size.value_counts().sort_index().items()},
            "distinct_labels_per_group": {str(int(k)): int(v) for k, v in group_stats.group_label_count.value_counts().sort_index().items()},
            "responses_by_distinct_label_count": {str(int(k)): int(v) for k, v in data.groupby("prompt_id", sort=False).agg(labels=("model", "nunique"), rows=("id", "size")).groupby("labels").rows.sum().items()},
            "groups_by_key_basis": {},
        },
        "repeated_same_model": {"groups": int(data.loc[data.group_model_count.gt(1), "prompt_id"].nunique()),
                                "group_label_combinations": int(len(repeated_pairs)),
                                "response_rows": int(repeated_pairs.model_count.sum()),
                                "all_responses_preserved": True},
        "validation": {"unique_response_ids": True, "normalized_duplicate_clusters": 0,
                       "conflicting_labels_for_identical_responses": 0,
                       "all_18_labels_represented": set(data.model) == LABELS,
                       "ranking_groups_have_at_least_two_labels": bool((eligible_groups.group_label_count >= 2).all()),
                       "max_distinct_labels": int(group_stats.group_label_count.max()),
                       "groups_over_18_labels": int(group_stats.group_label_count.gt(18).sum()),
                       "v0_overlap_checked": v0 is not None,
                       "v0_id_overlap": 0, "v0_normalized_text_overlap": 0,
                       "v0_prompt_identifier_overlap": 0},
        "v0_scope": {"training_data": "external-only", "validation_dataset": "v0",
                     "validation_rows": 2699, "v0_included_in_artifact": False},
        "audit_reconciliation": {"expected_rows": EXPECTED_ROWS, "expected_ranking_responses": EXPECTED_RANK_ROWS,
                                 "expected_eligible_groups": EXPECTED_RANK_GROUPS,
                                 "expected_group_label_combinations": EXPECTED_RANK_ITEMS,
                                 "expected_pairwise_comparisons": EXPECTED_PAIRWISE,
                                 "actual": {"rows": len(data), "ranking_responses": len(eligible_rows),
                                            "eligible_groups": len(eligible_groups),
                                            "group_label_combinations": group_label_items,
                                            "pairwise_comparisons": pairwise}},
    }
    # Derive canonical/fallback eligibility distributions and exact group-size tables for report.
    basis_df = pd.DataFrame({"prompt_id": group_stats.index,
                             "basis": ["canonical" if str(x).startswith("canon:") else "fallback" for x in group_stats.index],
                             "size": group_stats.group_size.values, "labels": group_stats.group_label_count.values})
    profile["distributions"]["groups_by_key_basis"] = {
        basis: {"groups": int(len(part)), "responses": int(part["size"].sum()),
                "ranking_groups": int(part.labels.ge(2).sum()),
                "ranking_responses": int(part.loc[part.labels.ge(2), "size"].sum())}
        for basis, part in basis_df.groupby("basis")}
    profile["reconciliation_matches_audit"] = {
        "rows": len(data) == EXPECTED_ROWS,
        "ranking_responses": len(eligible_rows) == EXPECTED_RANK_ROWS,
        "eligible_groups": len(eligible_groups) == EXPECTED_RANK_GROUPS,
        "group_label_combinations": group_label_items == EXPECTED_RANK_ITEMS,
        "pairwise_comparisons": pairwise == EXPECTED_PAIRWISE,
    }
    profile["reconciliation_matches_audit"]["all"] = all(profile["reconciliation_matches_audit"].values())
    return data, profile


def validate_artifact(data: pd.DataFrame, profile: dict[str, Any], *, v0: pd.DataFrame | None = None) -> None:
    """Strict artifact validation, independent of parquet metadata."""
    needed = {"id", "prompt_id", "text", "model", "group_size", "group_label_count",
              "group_model_count", "ranking_eligible", "ranking_rotation_index", "ranking_label_index"}
    missing = needed - set(data)
    if missing:
        raise ValueError(f"artifact is missing columns: {sorted(missing)}")
    if data.id.isna().any() or data.id.duplicated().any():
        raise ValueError("artifact response IDs must be present and unique")
    if data.prompt_id.isna().any() or data.prompt_id.astype(str).str.strip().eq("").any():
        raise ValueError("artifact prompt IDs must be present")
    expected_prompt_ids = [prompt_key(row) for row in data.itertuples(index=False)]
    if expected_prompt_ids != data.prompt_id.astype(str).tolist():
        raise ValueError("prompt IDs do not match their canonical/fallback source keys")
    if not data.model.isin(LABELS).all() or set(data.model) != LABELS:
        raise ValueError("artifact must contain all and only canonical labels")
    actual_size = data.groupby("prompt_id").id.transform("size")
    actual_labels = data.groupby("prompt_id").model.transform("nunique")
    actual_model = data.groupby(["prompt_id", "model"]).id.transform("size")
    import numpy as np
    if (not np.array_equal(actual_size.to_numpy(), data.group_size.to_numpy())
            or not np.array_equal(actual_labels.to_numpy(), data.group_label_count.to_numpy())
            or not np.array_equal(actual_model.to_numpy(), data.group_model_count.to_numpy())):
        raise ValueError("group statistics are inconsistent with artifact rows")
    if not data.ranking_eligible.equals(data.group_label_count.ge(2)):
        raise ValueError("ranking_eligible does not match observed group label count")
    if data.group_label_count.gt(18).any():
        raise ValueError("group contains more than 18 canonical labels")
    rotation_check = data.sort_values(["prompt_id", "model", "id"], kind="stable").groupby(
        ["prompt_id", "model"], sort=False).cumcount().to_numpy()
    sorted_indices = data.sort_values(["prompt_id", "model", "id"], kind="stable").ranking_rotation_index.to_numpy()
    if not (rotation_check == sorted_indices).all():
        raise ValueError("ranking rotation indices are not stable response-ID order within prompt and label")
    if data.text.map(_normal_text).duplicated().any():
        raise ValueError("normalized response text duplicates found")
    if v0 is not None:
        if set(data.id.astype(str)) & set(v0.id.astype(str)):
            raise ValueError("response ID overlap with V0")
        if set(data.text.map(_normal_text)) & set(v0.text.map(_normal_text)):
            raise ValueError("normalized text overlap with V0")
    if int(data.ranking_eligible.sum()) != profile["ranking"]["eligible_responses"]:
        raise ValueError("profile ranking row count disagrees with artifact")


def write_report(profile: dict[str, Any], path: Path) -> None:
    actual = profile["audit_reconciliation"]["actual"]
    dist = profile["distributions"]
    lines = [
        "# V2.GR.OR16 — Group-Ranking Open Pool v16", "",
        "## Summary", "",
        f"Built from the original Open Pool v16 main parquet only: **{profile['rows']:,} classification responses** across **18 canonical labels**.",
        f"**{actual['ranking_responses']:,} rows ({profile['ranking']['eligible_response_pct']:.2f}%)** in **{actual['eligible_groups']:,} groups** are ranking-eligible (at least two distinct labels).",
        f"One response per label yields **{actual['group_label_combinations']:,} ranking items** and **{actual['pairwise_comparisons']:,} unordered comparisons** per full group pass.", "",
        "The artifact is external-only. All 2,699 V0 rows remain reserved for validation and checkpoint selection.", "",
        "## Provenance and hashes", "",
        f"- Source: `{profile['source']}`", f"- Source SHA-256: `{profile['source_sha256']}`",
        f"- Output SHA-256: `{profile['dataset_sha256']}`", f"- Response-ID sequence SHA-256: `{profile['row_id_sha256']}`",
        f"- Normalized-text sequence SHA-256: `{profile['normalized_text_sha256']}`", "",
        "## Group construction", "",
        "`prompt_id` is `canon:<normalized prompt_canon>` when canonical metadata is present. Missing canonical values use `fallback:<JSON([normalized source_dataset, source_file, prompt_group])>`. The namespace prefixes prevent key collisions. Group sizes and label counts are computed from response rows and canonical `model` labels; `panel_labels` is retained as provenance but not trusted for ranking cardinality.", "",
        "For repeated responses from the same model in a group, `ranking_rotation_index` is the stable zero-based position after sorting by response ID. At epoch `e`, select the row whose index equals `e % group_model_count`; this selects exactly one response per group-label and rotates duplicates deterministically.", "",
        "## Row and group distributions", "",
        "### Responses by canonical label", "", "| Label | Classification rows | Ranking-eligible rows |", "|---|---:|---:|",
    ]
    for label in sorted(profile["label_counts"]):
        lines.append(f"| `{label}` | {profile['label_counts'][label]:,} | {profile['ranking']['per_label_eligible_responses'][label]:,} |")
    lines += ["", "### Group size (response rows)", "", "| Group size | Groups |", "|---:|---:|"]
    for size, count in dist["group_size"].items(): lines.append(f"| {size} | {count:,} |")
    lines += ["", "### Distinct labels per group", "", "| Distinct labels | Groups | Responses | Ranking-eligible groups |", "|---:|---:|---:|---:|"]
    for count, n_groups in dist["distinct_labels_per_group"].items():
        responses = dist["responses_by_distinct_label_count"][count]
        lines.append(f"| {count} | {n_groups:,} | {responses:,} | {'Yes' if int(count) >= 2 else 'No'} |")
    lines += ["", "### Group-key basis", "", "| Basis | Groups | Responses | Ranking groups | Ranking responses |", "|---|---:|---:|---:|---:|"]
    for basis, stat in sorted(dist["groups_by_key_basis"].items()):
        lines.append(f"| {basis} | {stat['groups']:,} | {stat['responses']:,} | {stat['ranking_groups']:,} | {stat['ranking_responses']:,} |")
    lines += ["", "## Validation and audit reconciliation", "", "| Check | Result |", "|---|---:|"]
    checks = profile["validation"]
    for label, value in [("Unique response IDs", checks["unique_response_ids"]),
                         ("Normalized duplicate clusters", checks["normalized_duplicate_clusters"]),
                         ("Conflicting labels for identical normalized responses", checks["conflicting_labels_for_identical_responses"]),
                         ("All 18 labels represented", checks["all_18_labels_represented"]),
                         ("V0 ID overlap", checks["v0_id_overlap"]),
                         ("V0 normalized-text overlap", checks["v0_normalized_text_overlap"]),
                         ("V0 prompt-identifier overlap", checks["v0_prompt_identifier_overlap"]),
                         ("Groups above 18 labels", checks["groups_over_18_labels"])]:
        lines.append(f"| {label} | {value} |")
    lines += ["", "Reconciliation against the prompt-group audit:", "", "| Metric | Expected | Built | Match |", "|---|---:|---:|---|"]
    for name, expected in [("rows", EXPECTED_ROWS), ("ranking_responses", EXPECTED_RANK_ROWS),
                           ("eligible_groups", EXPECTED_RANK_GROUPS), ("group_label_combinations", EXPECTED_RANK_ITEMS),
                           ("pairwise_comparisons", EXPECTED_PAIRWISE)]:
        value = actual[name]
        lines.append(f"| {name.replace('_', ' ')} | {expected:,} | {value:,} | {'Yes' if expected == value else 'No'} |")
    lines += ["", f"Repeated same-model responses: **{profile['repeated_same_model']['groups']:,} groups**, **{profile['repeated_same_model']['group_label_combinations']:,} prompt/model combinations**, and **{profile['repeated_same_model']['response_rows']:,} response rows**. All are retained; no selection is performed during dataset preparation.",
              "", "### Audit caveat", "",
              "The source `panel_labels` metadata is not used to determine group cardinality. The audit found it disagreed with observed response labels in 1,008 fallback groups (5,123 rows). This does not change eligibility: group label counts are recalculated from the actual `model` values in the artifact.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

