#!/usr/bin/env python3
"""Build a deterministic, quality-filtered Open Pool v16 increment.

The source snapshot is preserved verbatim under data/raw/open_pool/v16. This
builder compares it with V2.16 and frozen V0, then writes only new eligible
rows in the existing training parquet schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_factory import OPENPOOL_VERTEX_G37_SOURCES, _normal_text, _hash_file, _language

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/open_pool/v16"
BASE = ROOT / "data/processed/dataset_v2.16_14plus15.parquet"
V0 = ROOT / "data/processed/dataset_v0.parquet"
OUT = ROOT / "data/processed/dataset_v2.16_openpool16_increment.parquet"
PROFILE = ROOT / "data/processed/dataset_profile_v2.16_openpool16_increment.json"
EXCLUSIONS = ROOT / "data/processed/v2.16_openpool16_increment_exclusions.csv"
REPORT = ROOT / "docs/openpool_v16_increment.md"
MAIN = RAW / "ntl_open_pool.parquet"
SIDECAR = RAW / "gemini37_unverified.parquet"
META = RAW / "dataset-metadata.json"
DOWNLOAD_META = RAW / "download_metadata.json"

LICENSE_TOKENS = ("CC BY 4.0", "CC-BY-SA-4.0", "CC BY-SA 4.0", "CC0-1.0", "CC0 1.0")
CANONICAL = {
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
}
ARTIFACT_RE = re.compile(
    r"(?is)(?:\[object object\]|\x00|<\|im_start\|>|<\|im_end\|>|<\|assistant\|>|<\|user\|>)"
    r"|(?:^\s*(?:tool call|tool response|function call|execution log)\s*:\s*\{.{0,100}\}\s*$)"
)
TRUNC_RE = re.compile(
    r"(?i)(?:\[(?:truncated|incomplete response|generation failed)\]"
    r"|\((?:truncated|generation failed)\)|\.\.\.\s*\[end\])\s*$"
)
REFUSAL_RE = re.compile(
    r"(?i)^\s*(?:i(?:'m| am) sorry,?\s+(?:but\s+)?i (?:cannot|can't|won't)|"
    r"i cannot assist with that request|i can't help with that request)\b"
)
CONTENT_AUDIT_PATTERNS = {
    "thinking_tags": r"(?is)<\s*(?:think|analysis|reasoning|scratchpad)\s*>|\[/?(?:think|analysis|reasoning|scratchpad)\]",
    "explicit_chain_of_thought_markers": r"(?i)\b(?:chain of thought|hidden reasoning|internal reasoning|private reasoning|scratchpad)\b",
    "reasoning_self_talk": r"(?i)\b(?:let me think (?:through|about)|let(?:'s| us) reason (?:through|step by step)|my (?:internal )?reasoning is|in my (?:private )?analysis|i should not reveal (?:my )?(?:chain of thought|reasoning))\b",
    "assistant_preamble": r"(?i)^\s*(?:sure[,! ]|certainly[,! ]|as an ai language model|i can help with that|here(?:'s| is) (?:the|a) (?:answer|response))",
    "serialized_chat_or_api": r"(?is)<\|(?:im_start|im_end|assistant|user|system|tool)\|>|\[object object\]|\b(?:messages|choices|finish_reason|tool_calls)\s*[:=]\s*[\[{]",
    "fenced_code": r"```|~~~",
    "code_like_standalone_line": r"(?m)^\s{0,4}(?:def\s+\w+\s*\(|class\s+\w+\s*[:(]|(?:const|let|var)\s+\w+\s*=|import\s+[\w.*{} ,]+\s+from\s+['\"]|#include\s*<|SELECT\s+.+\s+FROM\s+|\$\s+[\w./-]+)",
    "stack_trace": r"(?m)^\s*(?:Traceback \(most recent call last\):|at [\w.$]+\([^)]*:\d+\)|[\w.]+Error:)",
    "markdown_heading_or_list": r"(?m)^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)",
}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"} if pd.notna(value) else False


def clean_paragraph(value: str) -> str:
    # Same whitespace collapsing used by the established normalizer, while
    # retaining source punctuation and Unicode text.
    return " ".join(str(value).split())


def canonical_label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def source_version(metadata: dict, version_number: Any = None) -> str:
    return f"Open Pool v16 (Kaggle dataset version {version_number or 'unknown'}; released 2026-11-18)"


def build(root: Path = RAW, *, output: Path = OUT, profile_path: Path = PROFILE,
          exclusions_path: Path = EXCLUSIONS, report_path: Path = REPORT) -> dict:
    main_path = root / "ntl_open_pool.parquet"
    sidecar_path = root / "gemini37_unverified.parquet"
    metadata_path = root / "dataset-metadata.json"
    for path in (main_path, sidecar_path, metadata_path, BASE, V0):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    info = metadata.get("info", {})
    download_meta = json.loads(DOWNLOAD_META.read_text(encoding="utf-8")) if DOWNLOAD_META.is_file() else {}
    current_version = download_meta.get("current_version_number")
    if current_version is None:
        raise ValueError("Kaggle version number is missing from download_metadata.json")
    raw = pd.read_parquet(main_path).reset_index(drop=True)
    base = pd.read_parquet(BASE)
    v0 = pd.read_parquet(V0)
    if len(raw) != 231_036 or len(base) != 277_683 or len(v0) != 2_699:
        raise ValueError(f"unexpected row count: v16={len(raw)}, v2.16={len(base)}, v0={len(v0)}")
    if not CANONICAL.issubset(set(base.model.astype(str))):
        raise ValueError("V2.16 does not contain the expected canonical 18 labels")

    base_ids = set(base.id.dropna().astype(str))
    base_norm = {}
    for row in base[["text", "model", "id"]].itertuples(index=False):
        norm = _normal_text(row.text)
        if norm:
            base_norm.setdefault(norm, []).append((str(row.id), str(row.model)))
    v0_ids = set(v0.id.dropna().astype(str))
    v0_norm = set(v0.text.dropna().astype(str).map(_normal_text)) - {""}
    v0_prompt_groups = set(v0.prompt_id.dropna().astype(str))

    exclusions: list[dict[str, Any]] = []

    def exclude(row: pd.Series, reason: str, detail: str = "") -> None:
        exclusions.append({
            "source_row_number": int(row.name), "id": row.get("id", ""),
            "model": row.get("model", ""), "source_dataset": row.get("source_dataset", ""),
            "source_file": row.get("source_file", ""), "reason": reason,
            "detail": detail, "normalized_text_sha256": sha_text(_normal_text(row.get("text", ""))),
        })

    prequality: list[tuple[int, pd.Series, str]] = []
    existing_count = 0
    existing_label_conflict = 0
    id_conflict = 0
    for idx, row in raw.iterrows():
        rid = str(row.get("id", ""))
        norm = _normal_text(row.get("text", ""))
        matches = base_norm.get(norm, []) if norm else []
        id_match = rid in base_ids
        if id_match or matches:
            existing_count += 1
            labels_for_match = {label for _, label in matches}
            if (labels_for_match and str(row.get("model", "")) not in labels_for_match):
                existing_label_conflict += 1
                exclude(row, "v2.16_same_text_conflicting_label", ",".join(sorted(labels_for_match)))
            elif id_match and matches and not any(i == rid for i, _ in matches):
                id_conflict += 1
                exclude(row, "v2.16_id_reused_for_different_text", "")
            else:
                exclude(row, "already_present_in_v2.16", "normalized text or response ID exists in V2.16")
            continue
        prequality.append((idx, row, norm))

    # Resolve duplicates in the downloaded main parquet before row-level quality
    # checks. Conflicting labels are dropped as a whole normalized-text group.
    by_norm: dict[str, list[tuple[int, pd.Series, str]]] = {}
    for item in prequality:
        by_norm.setdefault(item[2], []).append(item)
    unique_candidates: list[tuple[int, pd.Series, str]] = []
    intra_duplicates = 0
    intra_conflict_rows = 0
    for norm, group in sorted(by_norm.items()):
        labels = {str(item[1].get("model", "")) for item in group}
        if len(labels) > 1:
            intra_conflict_rows += len(group)
            for _, row, _ in group:
                exclude(row, "conflicting_labels_within_v16", ",".join(sorted(labels)))
            continue
        group.sort(key=lambda item: (str(item[1].get("id", "")), item[0]))
        unique_candidates.append(group[0])
        for _, row, _ in group[1:]:
            intra_duplicates += 1
            exclude(row, "duplicate_normalized_response_within_v16", str(group[0][1].get("id", "")))

    new_before_quality = len(unique_candidates)
    accepted: list[pd.Series] = []
    quality_count: Counter[str] = Counter()
    source_version_value = source_version(info, current_version)
    allowed_license = lambda value: any(token.casefold() in str(value).casefold() for token in LICENSE_TOKENS)
    canonical_keys = {canonical_label_key(label): label for label in CANONICAL}

    for idx, row, norm in unique_candidates:
        reason = ""
        detail = ""
        rid = str(row.get("id", "")).strip()
        text = row.get("text")
        label = str(row.get("model", ""))
        license_value = str(row.get("source_license", ""))
        required = (rid and pd.notna(text) and str(text).strip() and label in CANONICAL
                    and str(row.get("source_dataset", "")).strip()
                    and str(row.get("source_file", "")).strip()
                    and str(row.get("source_license", "")).strip()
                    and str(row.get("prompt_group", "")).strip()
                    and (str(row.get("prompt_canon", "") or "").strip()
                         or str(row.get("prompt_group", "")).strip()))
        if not required:
            reason = "missing_or_invalid_text_label_or_provenance"
        elif not allowed_license(license_value):
            reason = "ineligible_or_unknown_source_license"
        elif len(str(text)) > 2_000:
            reason = "response_over_2000_characters"
        elif rid in v0_ids:
            reason = "v0_response_id_overlap"
        elif norm in v0_norm:
            reason = "v0_normalized_response_overlap"
        elif str(row.get("prompt_canon", "")) in v0_prompt_groups or str(row.get("prompt_group", "")) in v0_prompt_groups:
            reason = "v0_prompt_group_overlap"
        elif label == "gemini-3.7-flash" and str(row.get("source_dataset", "")) not in OPENPOOL_VERTEX_G37_SOURCES:
            reason = "gemini37_outside_existing_vertex_allowlist"
        elif any(truthy(row.get(flag)) for flag in ("flag_gemini37_routing_risk", "flag_near_train_prompt", "flag_train_prompt_regen")):
            reason = "source_quality_or_routing_flag"
        elif ARTIFACT_RE.search(str(text)):
            reason = "serialized_api_or_extraction_artifact"
        elif TRUNC_RE.search(str(text)):
            reason = "explicit_incomplete_or_truncation_marker"
        elif REFUSAL_RE.search(str(text)):
            reason = "refusal_response"
        else:
            served = str(row.get("gen_served_model", "") or "").strip()
            served_tail = served.rsplit("/", 1)[-1]
            served_label = canonical_keys.get(canonical_label_key(served_tail)) if served_tail else None
            if served_label and served_label != label:
                reason = "served_model_label_contradiction"
                detail = f"served={served}; label={label}"
        if reason:
            quality_count[reason] += 1
            exclude(row, reason, detail)
            continue
        out = row.copy()
        out["text"] = clean_paragraph(str(text))
        canonical_prompt = str(row.get("prompt_canon", "") or "").strip()
        prompt_group = str(row.get("prompt_group", "") or "").strip()
        out["prompt_id"] = canonical_prompt or prompt_group
        out["source_version"] = source_version_value
        out["source_row_id"] = rid
        out["source_original_id"] = rid
        out["source_prompt_id"] = prompt_group
        out["upstream_source"] = str(row["source_dataset"])
        out["contributing_sources_json"] = json.dumps(
            [{"source_dataset": str(row["source_dataset"]),
              "also_in": str(row.get("also_in", "")),
              "source_file": str(row["source_file"]),
              "source_license": str(row["source_license"])}],
            ensure_ascii=False, sort_keys=True,
        )
        out["provenance_json"] = json.dumps({
            "snapshot": "crtproduction/name-that-llm-open-pool",
            "snapshot_version": current_version,
            "snapshot_release": "2026-11-18",
            "source_dataset": str(row["source_dataset"]),
            "source_file": str(row["source_file"]),
            "source_prompt_id": str(row["prompt_group"]),
            "also_in": str(row.get("also_in", "")),
            "match_weight": row.get("match_weight"),
            "gen_reasoning": row.get("gen_reasoning"),
            "gen_route": row.get("gen_route"),
            "gen_served_model": row.get("gen_served_model"),
        }, ensure_ascii=False, sort_keys=True, default=str)
        accepted.append(out)

    # Match the parent dataset's complete schema and column order. New v16 fields
    # are already in that schema; unavailable legacy metadata stays null.
    result = pd.DataFrame(accepted)
    for column in base.columns:
        if column not in result:
            result[column] = pd.Series([None] * len(result), dtype=base[column].dtype)
    result = result[list(base.columns)].copy()
    if len(result):
        result = result.sort_values(["model", "prompt_id", "id"], kind="mergesort").reset_index(drop=True)

    # Record the established Open Pool language detector and visible formatting
    # scans. These are strict checks for this prose-passage increment.
    if len(result):
        from langdetect import DetectorFactory
        DetectorFactory.seed = 2026
        language = result.text.astype(str).map(_language)
        language_counts = language.value_counts().sort_index().astype(int).to_dict()
        content_audit = {
            name: int(result.text.astype(str).str.contains(pattern, regex=True, na=False).sum())
            for name, pattern in CONTENT_AUDIT_PATTERNS.items()
        }
    else:
        language_counts = {}
        content_audit = {name: 0 for name in CONTENT_AUDIT_PATTERNS}

    norm_result = result.text.astype(str).map(_normal_text) if len(result) else pd.Series(dtype=str)
    checks = {
        "accepted_ids_unique": bool(result.id.astype(str).is_unique),
        "accepted_normalized_text_unique": bool(norm_result.is_unique),
        "accepted_labels_valid": bool(set(result.model.astype(str)).issubset(CANONICAL)),
        "accepted_ids_disjoint_v216": not bool(set(result.id.astype(str)) & base_ids),
        "accepted_text_disjoint_v216": not bool(set(norm_result) & set(base_norm)),
        "accepted_ids_disjoint_v0": not bool(set(result.id.astype(str)) & v0_ids),
        "accepted_text_disjoint_v0": not bool(set(norm_result) & v0_norm),
        "accepted_prompt_groups_disjoint_v0": not bool(set(result.prompt_id.astype(str)) & v0_prompt_groups),
        "accepted_length_at_most_2000": bool(result.text.astype(str).str.len().le(2_000).all()),
        "accepted_source_license_known": bool(result.source_license.astype(str).map(allowed_license).all()),
        "all_accepted_detected_english": set(language_counts) <= {"en"},
        "no_code_or_visible_reasoning_artifacts": not any(content_audit.values()),
    }
    if not all(checks.values()):
        raise ValueError(f"increment validation failed: {checks}")

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    exclusions_df = pd.DataFrame(exclusions)
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_df.to_csv(exclusions_path, index=False, lineterminator="\n")
    ordered = result.sort_values("id", kind="mergesort") if len(result) else result
    canonical_hash = sha_text("\n".join(
        f"{r.id}\t{r.model}\t{_normal_text(r.text)}\t{r.prompt_id}" for r in ordered.itertuples()
    ))

    excluded_df = exclusions_df
    accepted_by_label = result.model.value_counts().sort_index().astype(int).to_dict() if len(result) else {}
    accepted_by_source = result.source_dataset.value_counts().sort_index().astype(int).to_dict() if len(result) else {}
    excluded_by_label = excluded_df.model.value_counts().sort_index().astype(int).to_dict() if len(excluded_df) else {}
    excluded_by_source = excluded_df.source_dataset.value_counts().sort_index().astype(int).to_dict() if len(excluded_df) else {}
    quality_only = excluded_df[~excluded_df.reason.eq("already_present_in_v2.16")] if len(excluded_df) else excluded_df
    quality_excluded_by_label = quality_only.model.value_counts().sort_index().astype(int).to_dict() if len(quality_only) else {}
    quality_excluded_by_source = quality_only.source_dataset.value_counts().sort_index().astype(int).to_dict() if len(quality_only) else {}
    v16_log = pd.read_csv(root / "v16_append_log.csv") if (root / "v16_append_log.csv").is_file() else pd.DataFrame()
    v16_source_log = pd.read_csv(root / "v16_append_by_source.csv") if (root / "v16_append_by_source.csv").is_file() else pd.DataFrame()
    counts = {
        "downloaded_main_rows": len(raw), "sidecar_rows_excluded_unverified": len(pd.read_parquet(sidecar_path)),
        "already_present_in_v2.16": existing_count,
        "existing_text_label_conflicts": existing_label_conflict,
        "existing_id_reuse_conflicts": id_conflict,
        "new_before_quality_filtering": new_before_quality,
        "duplicate_rows_within_download": intra_duplicates,
        "conflicting_label_rows_within_download": intra_conflict_rows,
        "removed_by_quality_criteria": int(sum(quality_count.values())),
        "final_accepted_additions": len(result),
    }
    if len(raw) != (existing_count + intra_duplicates + intra_conflict_rows + new_before_quality):
        raise ValueError("download row accounting does not reconcile")

    file_hashes = {p.name: _hash_file(p) for p in sorted(root.iterdir()) if p.is_file()}
    profile = {
        "dataset_version": "v2.16_openpool16_increment", "source": "crtproduction/name-that-llm-open-pool",
        "source_pool_release": "v16", "kaggle_dataset_version": current_version,
        "source_published_at": "2026-11-18T14:49:47.833Z",
        "source_dataset_sha256": _hash_file(main_path), "source_file_hashes": file_hashes,
        "source_snapshot_metadata_sha256": _hash_file(metadata_path),
        "kaggle_download_metadata_sha256": _hash_file(DOWNLOAD_META) if DOWNLOAD_META.is_file() else None,
        "parent_dataset": str(BASE.relative_to(ROOT)), "parent_dataset_sha256": _hash_file(BASE),
        "v0_dataset_sha256": _hash_file(V0), "row_counts": counts,
        "quality_exclusions_by_reason": dict(sorted(quality_count.items())),
        "excluded_by_label_and_reason": excluded_by_label,
        "excluded_by_source_and_reason": excluded_by_source,
        "quality_excluded_rows_by_label": quality_excluded_by_label,
        "quality_excluded_rows_by_source": quality_excluded_by_source,
        "accepted_rows_by_label": accepted_by_label, "accepted_rows_by_source": accepted_by_source,
        "language_detector": "langdetect with DetectorFactory.seed=2026 (same helper as src.data_factory._language)",
        "accepted_rows_by_detected_language": language_counts,
        "visible_content_audit_pattern_counts": content_audit,
        "within_download_duplicate_rows": intra_duplicates,
        "within_download_conflicting_label_rows": intra_conflict_rows,
        "all_rows_quality_screened_in_source_main": True,
        "source_quality_screen_notes": "The v16 main parquet is the publisher-cleaned main pool; API dumps, refusals, off-genre, incomplete, and known mislabeled rows are publisher-screened. Local rules additionally enforce v2.16/V0 exact isolation, 2,000 characters, exact Gemini 3.7 allowlist, explicit artifact/truncation/refusal markers, license/provenance, and served-model contradictions.",
        "gemini37_allowed_sources": sorted(OPENPOOL_VERTEX_G37_SOURCES),
        "gemini37_unverified_sidecar_included": False,
        "v16_append_log": v16_log.to_dict("records"),
        "v16_append_by_source": v16_source_log.to_dict("records"),
        "accepted_canonical_content_sha256": canonical_hash,
        "output_parquet_sha256": _hash_file(output),
        "exclusion_ledger_sha256": _hash_file(exclusions_path),
        "checks": checks,
        "deterministic_sort": ["model", "prompt_id", "id"],
        "normalization": "Unicode NFKC, casefold, collapsed whitespace; punctuation preserved",
        "text_formatting": "line breaks collapsed to single paragraph; original punctuation preserved",
        "quality_exclusions_by_reason": dict(sorted(quality_count.items())),
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if report_path:
        write_report(profile, result, exclusions_df, report_path)
    return profile


def write_report(profile: dict, result: pd.DataFrame, exclusions: pd.DataFrame, path: Path) -> None:
    counts = profile["row_counts"]
    lines = [
        "# Open Pool v16 incremental dataset",
        "",
        "Downloaded and curated the latest `crtproduction/name-that-llm-open-pool` snapshot against the immutable V2.16 parent. V2.16 was not modified; no training or Kaggle submission was run.",
        "",
        "## Source snapshot",
        "",
        f"- Kaggle dataset version: **{profile['kaggle_dataset_version']}** (current version when downloaded).",
        f"- Publisher release: **Open Pool v16, 2026-11-18**; Kaggle last updated: `{profile['source_published_at']}`.",
        f"- Main parquet SHA-256: `{profile['source_dataset_sha256']}`.",
        f"- Metadata SHA-256: `{profile['source_snapshot_metadata_sha256']}`.",
        f"- V2.16 parent SHA-256: `{profile['parent_dataset_sha256']}`.",
        f"- V0 SHA-256: `{profile['v0_dataset_sha256']}`.",
        f"- Raw files remain under [`data/raw/open_pool/v16`](../data/raw/open_pool/v16). Per-file SHA-256 values are in [`dataset_profile_v2.16_openpool16_increment.json`](../data/processed/dataset_profile_v2.16_openpool16_increment.json).",
        "- The archive main file carries the v16 source pool; the separate `gemini37_unverified.parquet` sidecar was excluded in full.",
        "",
        "## Row accounting",
        "",
        "| Metric | Rows |",
        "|---|---:|",
        f"| Downloaded Open Pool v16 main parquet | {counts['downloaded_main_rows']:,} |",
        f"| Already present in V2.16 by response ID or normalized text | {counts['already_present_in_v2.16']:,} |",
        f"| New before quality filtering (after within-file exact deduplication) | {counts['new_before_quality_filtering']:,} |",
        f"| Removed by quality criteria | {counts['removed_by_quality_criteria']:,} |",
        f"| Final accepted additions | {counts['final_accepted_additions']:,} |",
        f"| Within-download duplicate rows (also reported separately) | {counts['duplicate_rows_within_download']:,} |",
        f"| Within-download conflicting-label rows (also reported separately) | {counts['conflicting_label_rows_within_download']:,} |",
        f"| Separate unverified Gemini 3.7 sidecar rows excluded | {counts['sidecar_rows_excluded_unverified']:,} |",
        "",
        "Within-file duplicate/conflict rows are accounted for before the ‘new before quality filtering’ count; row accounting reconciles to the downloaded main parquet.",
        "",
        "## Quality criteria and exclusions",
        "",
        "The v16 main parquet already applies the publisher’s documented source-quality review, including API dump/refusal/off-genre/incomplete-text and known-label checks. Local filtering additionally requires valid text, canonical label, source/file/license/prompt provenance, a known CC license, at most 2,000 characters, no exact V2.16/V0 ID or normalized-response overlap, no known V0 prompt-group overlap, no explicit serialization/refusal/truncation marker, and no direct served-model/canonical-label contradiction. Existing Open Pool rules restrict `gemini-3.7-flash` to the two approved Vertex sources in `src/data_factory.py`; all other such rows were excluded. Publisher flags for routing, near-train-prompt, and train-prompt regeneration are preserved and checked.",
        "",
        "| Exclusion reason | Rows |",
        "|---|---:|",
    ]
    for reason, n in sorted(profile["quality_exclusions_by_reason"].items()):
        lines.append(f"| `{reason}` | {n:,} |")
    lines += [
        "",
        "The source v15/v16 append logs also report their own pre-publication quality exclusions. These have already been applied upstream and are not added again to the local 231,036-row download accounting:",
        "",
        "| Publisher append-log filter | V15 rows | V16 rows | Combined |",
        "|---|---:|---:|---:|",
        "| Accepted-source input rows | 3,104 | 6,937 | 10,041 |",
        "| Already in previous pool/train | 0 | 1,490 | 1,490 |",
        "| Unknown model label | 0 | 0 | 0 |",
        "| Same text with conflicting labels | 0 | 0 | 0 |",
        "| Duplicate within new source rows | 0 | 0 | 0 |",
        "| Served-model label mismatch removed | 0 | 0 | 0 |",
        "| Outside 36–275 word range | 11 | 76 | 87 |",
        "| Newline/Markdown formatting | 527 | 0 | 527 |",
        "| Raw API response | 0 | 0 | 0 |",
        "| Refusal | 0 | 0 | 0 |",
        "| Chat preamble/CJK | 7 | 170 | 177 |",
        "| Cut off mid-sentence | 2 | 180 | 182 |",
        "| Disallowed em-dash typography | 124 | 121 | 245 |",
        "| Too similar to train passage | 81 | 50 | 131 |",
        "| Near train prompt | 1 | 5 | 6 |",
        "| Added to main pool | 2,351 | 4,845 | 7,196 |",
        "",
    ]
    lines += ["", "### Accepted and excluded rows by label", "", "| Label | Accepted | Excluded from snapshot |", "|---|---:|---:|"]
    for label in sorted(CANONICAL):
        lines.append(f"| `{label}` | {profile['accepted_rows_by_label'].get(label, 0):,} | {profile['excluded_by_label_and_reason'].get(label, 0):,} |")
    lines += ["", "### Accepted and excluded rows by source", "", "| Source dataset | Accepted | Excluded from snapshot | Quality exclusions |", "|---|---:|---:|---:|"]
    sources = sorted(set(profile["accepted_rows_by_source"]) | set(profile["excluded_by_source_and_reason"]))
    for source in sources:
        lines.append(f"| `{source}` | {profile['accepted_rows_by_source'].get(source, 0):,} | {profile['excluded_by_source_and_reason'].get(source, 0):,} | {profile['quality_excluded_rows_by_source'].get(source, 0):,} |")
    lines += ["", "### Exclusion reasons", "", "| Reason | Rows |", "|---|---:|"]
    for reason, n in sorted(exclusions.reason.value_counts().items()):
        lines.append(f"| `{reason}` | {int(n):,} |")
    lines += ["", "Every excluded response is recorded with its label, source, original row ID and reason in the exclusion ledger [`v2.16_openpool16_increment_exclusions.csv`](../data/processed/v2.16_openpool16_increment_exclusions.csv).", ""]

    lines += [
        "## V15/V16 changelog and priority labels",
        "",
        "The v16 changelog reports **4,845** new rows over v15: 4,393 from `elliotthankinson/ntl-tells-v4` and 452 new panel-gap passages from `dhananjayk2609/ntl-panel-gap-passages-kaggle-benchmarks`. Because V2.16 has the earlier pool but not all Open Pool v15 appends, its unseen rows total 7,196: 2,351 from the v15 append log plus 4,845 from v16. Six rows exceed the 2,000-character cap, leaving 7,190 accepted additions. The source table above includes rows from both append cohorts. The exact cohort assignment of the six length exclusions is not present in the published row schema, so it is not inferred. V16 reports 785 `kimi-k3`, 559 `glm-5.3`, 531 `glm-5.3-flash`, plus GPT 5.5/5.6 coverage. The Tells hard labels use paid/local routes and a discriminator gate; route, served-model, reasoning, match-weight, source and flags are retained.",
        "",
        "| Label | Accepted additions in this V2.16 increment | Accepted from Tells v4 |",
        "|---|---:|---:|",
    ]
    tells = result[result.source_dataset.eq("elliotthankinson/ntl-tells-v4")] if len(result) else result
    for label in ("kimi-k3", "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"):
        n = int(result.model.eq(label).sum()) if len(result) else 0
        nt = int(tells.model.eq(label).sum()) if len(tells) else 0
        lines.append(f"| `{label}` | {n:,} | {nt:,} |")
    lines += [
        "",
        "## Output and verification",
        "",
        f"- Increment: [`data/processed/dataset_v2.16_openpool16_increment.parquet`](../data/processed/dataset_v2.16_openpool16_increment.parquet) (**{counts['final_accepted_additions']:,} rows**).",
        f"- Output Parquet SHA-256: `{profile['output_parquet_sha256']}`.",
        f"- Canonical content SHA-256: `{profile['accepted_canonical_content_sha256']}`.",
        f"- Exclusion ledger SHA-256: `{profile['exclusion_ledger_sha256']}`.",
        "- All structural, class, ID, normalized-response, V0 prompt-group, source-license and length checks passed.",
        f"- Existing Open Pool language detection (`langdetect`, seed 2026): `{profile['accepted_rows_by_detected_language']}`; non-English/undetermined rows: **{sum(v for k, v in profile['accepted_rows_by_detected_language'].items() if k != 'en')}**.",
        f"- Visible response scan counts: `{profile['visible_content_audit_pattern_counts']}`. The checks cover thinking tags/markers, self-referential reasoning text, assistant preambles, serialized chat/API output, fenced or standalone code, stack traces, and Markdown headings/lists.",
        "- `gen_reasoning` is retained source metadata describing the generation setting, not a chain-of-thought text field; accepted rows have setting counts recorded in the profile.",
        "- Normalization: Unicode NFKC, case-folding, collapsed whitespace; punctuation preserved.",
        "- Text line breaks are collapsed to single spaces, preserving original punctuation.",
        "- `prompt_id` uses `prompt_canon` where supplied and falls back to source `prompt_group` for v16 rows whose source has no canonical prompt hash; original `prompt_group` remains intact.",
        "- The official V0 parquet does not contain prompt text. Therefore prompt-group ID isolation is verified, but semantic prompt-text overlap cannot be independently recomputed from V0.",
        "- Rebuild with `python scripts/build_openpool_v16_increment.py`. A second build to a temporary path reproduces the same canonical content hash.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--exclusions", type=Path, default=EXCLUSIONS)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    profile = build(args.raw_dir, output=args.output, profile_path=args.profile,
                    exclusions_path=args.exclusions, report_path=args.report)
    print(json.dumps(profile["row_counts"], indent=2))
    print(json.dumps(profile["checks"], indent=2))


if __name__ == "__main__":
    main()
