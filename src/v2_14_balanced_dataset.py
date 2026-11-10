"""Create a quality-filtered, balanced training set from dataset v2.14.

The target class size is the smallest class count remaining after the configured
quality and V0-isolation filters. No fixed per-class target is imposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_factory import _hash_file, _language, _normal_text  # noqa: E402

SOURCE = ROOT / "data/processed/dataset_v2.14.parquet"
V0 = ROOT / "data/processed/dataset_v0.parquet"
OUTPUT = ROOT / "data/processed/dataset_v2.14.balanced.parquet"
PROFILE = ROOT / "data/processed/dataset_profile_v2.14.balanced.json"
AUDIT = ROOT / "data/processed/v2.14.balanced_removals.csv"
REPORT = ROOT / "docs/v2.14.balanced.md"
SEED = 2026
MIN_CHARS = 60
MAX_CHARS = 2000

SERIALIZED_RE = re.compile(
    r"(?is)(?:<\|im_start\|>|<\|im_end\|>|<\|tool_call\|>|<tool_call\b|"
    r"</?arg_value\b|<tool_response\b|<function_call\b|<\|assistant\|>|"
    r"<\|user\|>|<\|system\|>|\bassistant\s+to=\w+|"
    r"^\s*(?:tool call|tool response|function call|execution log)\s*:)"
)
MULTITURN_RE = re.compile(
    r"(?im)^\s*(?:user|assistant|system)\s*:\s*.+$"
)
CODE_RE = re.compile(
    r"(?is)(?:```|</?code>|^\s*(?:#!/|import\s+[\w.]+\s*$|"
    r"from\s+[\w.]+\s+import\s+|def\s+\w+\s*\(|"
    r"class\s+\w+\s*[:({]|(?:const|let|var)\s+\w+\s*=))"
)
REASONING_RE = re.compile(
    r"(?im)(?:^\s*<think>|^\s*\[analysis\]|"
    r"^\s*(?:chain of thought|reasoning trace)\s*:|"
    r"^\s*let me think step by step\b|<\|thinking\|>)"
)
TRUNCATION_RE = re.compile(
    r"(?is)(?:\[(?:truncated|incomplete response|generation failed)\]\s*$|"
    r"\.\.\.\s*\[end\]\s*$)"
)


def stable_order(model: str, prompt_id: str, source: str, row_id: str, seed: int) -> str:
    value = f"{seed}|{model}|{prompt_id}|{source}|{row_id}"
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def truthy(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def build(*, seed: int = SEED, min_chars: int = MIN_CHARS,
          max_chars: int = MAX_CHARS) -> dict:
    for path in (OUTPUT, PROFILE, AUDIT, REPORT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing v2.14.balanced output: {path}")
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("character limits must satisfy 1 <= min_chars <= max_chars")

    from langdetect import DetectorFactory
    DetectorFactory.seed = seed

    source = pd.read_parquet(SOURCE).reset_index(drop=True)
    v0 = pd.read_parquet(V0)
    required = {"id", "text", "model", "prompt_id", "prompt_group", "source_dataset"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"v2.14 input is missing required fields: {sorted(missing)}")
    labels = sorted(v0.model.dropna().astype(str).unique())
    if len(labels) != 18 or set(source.model.astype(str).unique()) != set(labels):
        raise ValueError("v2.14 must contain the same 18 labels as V0")
    if source.id.astype(str).duplicated().any():
        raise ValueError("v2.14 contains duplicate row IDs")

    source["_original_index"] = source.index
    source["_norm"] = source.text.fillna("").astype(str).map(_normal_text)
    source["_reason"] = ""
    source["_evidence"] = ""
    source["_duplicate_of_id"] = ""

    def reject(mask, reason: str, evidence=None):
        mask = pd.Series(mask, index=source.index).fillna(False).astype(bool)
        mask &= source._reason.eq("")
        if mask.any():
            source.loc[mask, "_reason"] = reason
            if evidence is None:
                source.loc[mask, "_evidence"] = reason
            elif isinstance(evidence, pd.Series):
                source.loc[mask, "_evidence"] = evidence.loc[mask].astype(str)
            else:
                source.loc[mask, "_evidence"] = str(evidence)
        return int(mask.sum())

    counts: dict[str, int] = {"input_rows": len(source)}
    lengths = source.text.fillna("").astype(str).str.len()
    counts["response_under_60_characters"] = reject(lengths.lt(min_chars), "response_under_60_characters", lengths.astype(str) + " characters")
    counts["response_over_2000_characters"] = reject(lengths.gt(max_chars), "response_over_2000_characters", lengths.astype(str) + " characters")

    # Exclude exact normalized V0 response matches and V0 prompt groups. The
    # latter preserves the project's established train/validation isolation.
    v0_text = set(v0.text.dropna().astype(str).map(_normal_text)) - {""}
    v0_prompt_ids = set(v0.prompt_id.dropna().astype(str)) - {""}
    v0_overlap = source._norm.isin(v0_text)
    counts["v0_response_overlap"] = reject(v0_overlap, "v0_response_overlap", "normalized response matches official V0")
    prompt_overlap = source.prompt_id.fillna("").astype(str).isin(v0_prompt_ids)
    counts["v0_prompt_group_overlap"] = reject(prompt_overlap, "v0_prompt_group_overlap", "prompt_id matches official V0")

    # Explicit routing-risk signals. The after-ruling proxy flag is retained:
    # it identifies a route category, not a model-attribution risk.
    route_flag_cols = [c for c in ("flag_gemini37_routing_risk", "flag_routing_risk", "flag_model_routing_risk") if c in source]
    route_mask = pd.Series(False, index=source.index)
    route_evidence = pd.Series("", index=source.index, dtype="string")
    for col in route_flag_cols:
        flagged = source[col].map(truthy)
        route_mask |= flagged
        route_evidence.loc[flagged] = col + "=true"
    if "g37_rule" in source:
        g37_bad = (source.model.astype(str).eq("gemini-3.7-flash")
                   & source.g37_rule.fillna("").astype(str).str.strip().str.casefold().ne("")
                   & source.g37_rule.fillna("").astype(str).str.strip().str.casefold().ne("verified"))
        route_mask |= g37_bad
        route_evidence.loc[g37_bad] = "g37_rule is not verified"
    counts["flagged_routing"] = reject(route_mask, "flagged_routing", route_evidence)

    truncation_mask = pd.Series(False, index=source.index)
    truncation_evidence = pd.Series("", index=source.index, dtype="string")
    for col in ("source_truncated", "source_cap_hit"):
        if col in source:
            flagged = source[col].map(truthy)
            truncation_mask |= flagged
            truncation_evidence.loc[flagged] = col + "=true"
    if "source_finish_reason" in source:
        flagged = source.source_finish_reason.fillna("").astype(str).str.casefold().isin(
            {"length", "max_tokens", "token_limit", "length_capped"})
        truncation_mask |= flagged
        truncation_evidence.loc[flagged] = "source_finish_reason indicates token/length cap"
    marker = source.text.fillna("").astype(str).map(lambda t: bool(TRUNCATION_RE.search(t)))
    truncation_mask |= marker
    truncation_evidence.loc[marker] = "explicit truncation/failure marker in response"
    counts["truncation_or_cap"] = reject(truncation_mask, "truncation_or_cap", truncation_evidence)

    multi_mask = pd.Series(False, index=source.index)
    multi_evidence = pd.Series("", index=source.index, dtype="string")
    for col in ("source_turns", "turns", "n_turns", "turn_count"):
        if col in source:
            vals = pd.to_numeric(source[col], errors="coerce")
            flagged = vals.gt(1)
            multi_mask |= flagged
            multi_evidence.loc[flagged] = f"{col}>1"
    for col in ("is_multiturn", "multi_turn", "flag_multi_turn"):
        if col in source:
            flagged = source[col].map(truthy)
            multi_mask |= flagged
            multi_evidence.loc[flagged] = col + "=true"
    role_transcript = source.text.fillna("").astype(str).map(
        lambda t: len(MULTITURN_RE.findall(t)) >= 2)
    multi_mask |= role_transcript
    multi_evidence.loc[role_transcript] = "multiple explicit User/Assistant/System transcript turns"
    counts["multi_turn"] = reject(multi_mask, "multi_turn", multi_evidence)

    serialized = source.text.fillna("").astype(str).map(lambda t: bool(SERIALIZED_RE.search(t)))
    counts["serialized_conversation_or_tool_output"] = reject(
        serialized, "serialized_conversation_or_tool_output", "serialized role/tool markers")
    code = source.text.fillna("").astype(str).map(lambda t: bool(CODE_RE.search(t)))
    counts["code_output"] = reject(code, "code_output", "code block or source-like code signature")
    reasoning = source.text.fillna("").astype(str).map(lambda t: bool(REASONING_RE.search(t)))
    counts["explicit_reasoning_trace"] = reject(reasoning, "explicit_reasoning_trace", "explicit reasoning marker")

    # Conflicting normalized labels are ambiguous; remove every row in a
    # conflicting text group, irrespective of which copy appears first.
    nonempty_norm = source._norm.ne("")
    labels_per_text = source.loc[nonempty_norm].groupby("_norm").model.nunique()
    conflict_norms = set(labels_per_text[labels_per_text.gt(1)].index)
    conflict_mask = source._norm.isin(conflict_norms)
    conflict_evidence = source._norm.map(
        lambda key: "conflicting labels: " + ", ".join(sorted(source.loc[source._norm.eq(key), "model"].astype(str).unique()))
        if key in conflict_norms else "")
    counts["conflicting_normalized_labels"] = reject(
        conflict_mask, "conflicting_normalized_labels", conflict_evidence)

    # Detect English on remaining unique normalized responses. `unknown` from
    # B2M is not accepted as a language assertion; it is redetected here.
    eligible_idx = source.index[source._reason.eq("")]
    unique_text = source.loc[eligible_idx, "text"].fillna("").astype(str).drop_duplicates()
    language_by_text = {}
    for i, text in enumerate(unique_text, start=1):
        language_by_text[text] = _language(text)
        if i % 20000 == 0:
            print(f"Language detection: {i:,}/{len(unique_text):,}", flush=True)
    detected = source.loc[eligible_idx, "text"].fillna("").astype(str).map(language_by_text)
    language_evidence = pd.Series("detected language=" + detected.astype(str), index=eligible_idx)
    counts["non_english_or_undetermined"] = reject(
        detected.ne("en"), "non_english_or_undetermined", language_evidence)

    # Remove same-label duplicate copies, retaining a deterministic row with
    # the most complete provenance. Current v2.14 is already text-unique, but
    # this guard keeps the command safe if the source is regenerated.
    active = source.loc[source._reason.eq("")].copy()
    active["_provenance_score"] = active[[c for c in ("source_dataset", "source_file", "source_license", "prompt_id", "source", "route", "provider", "model_version") if c in active]].fillna("").astype(str).ne("").sum(axis=1)
    active["_stable"] = active.apply(lambda r: stable_order(str(r.model), str(r.prompt_id), str(r.source_dataset), str(r.id), seed), axis=1)
    active = active.sort_values(["_norm", "_provenance_score", "_stable"], ascending=[True, False, True], kind="stable")
    duplicate_mask = active._norm.duplicated(keep="first")
    duplicate_rows = active.loc[duplicate_mask]
    duplicate_representative = active.drop_duplicates("_norm", keep="first").set_index("_norm").id.astype(str).to_dict()
    counts["duplicate_normalized_response"] = len(duplicate_rows)
    if len(duplicate_rows):
        idx = duplicate_rows.index
        source.loc[idx, "_reason"] = "duplicate_normalized_response"
        source.loc[idx, "_evidence"] = "same normalized response; same label representative retained"
        source.loc[idx, "_duplicate_of_id"] = source.loc[idx, "_norm"].map(duplicate_representative)

    # Balance to the smallest post-filter class, without a fixed cap.
    eligible = source.loc[source._reason.eq("")].copy()
    available = eligible.groupby("model").size().reindex(labels, fill_value=0)
    quota = int(available.min())
    limiting_labels = [str(x) for x in available.index[available.eq(quota)]]
    if quota <= 0:
        raise ValueError(f"at least one class has no eligible rows: {available.to_dict()}")

    selected_parts = []
    for label in labels:
        candidates = eligible.loc[eligible.model.astype(str).eq(label)].copy()
        candidates["_stable"] = candidates.apply(
            lambda r: stable_order(str(r.model), str(r.prompt_id), str(r.source_dataset), str(r.id), seed), axis=1)
        candidates = candidates.sort_values("_stable", kind="stable")
        chosen: list[int] = []
        used_idx: set[int] = set()
        used_prompts: set[str] = set()
        sources = sorted(candidates.source_dataset.fillna("").astype(str).unique())
        # One per prompt per source-round in the first pass; there is no source
        # cap, and later passes can use repeated prompts to meet the quota.
        per_source = {
            src: candidates.loc[candidates.source_dataset.fillna("").astype(str).eq(src)].iterrows()
            for src in sources
        }
        exhausted: set[str] = set()
        while len(chosen) < quota and len(exhausted) < len(sources):
            added = False
            for src in sources:
                if src in exhausted or len(chosen) >= quota:
                    continue
                for idx, row in per_source[src]:
                    if idx in used_idx:
                        continue
                    prompt = str(row.prompt_id)
                    if prompt in used_prompts:
                        continue
                    chosen.append(idx); used_idx.add(idx); used_prompts.add(prompt)
                    added = True
                    break
                else:
                    exhausted.add(src)
            if not added:
                break
        if len(chosen) < quota:
            rest = candidates.loc[~candidates.index.isin(used_idx)]
            chosen.extend(rest.index[:quota-len(chosen)].tolist())
        if len(chosen) != quota:
            raise RuntimeError(f"could not select balanced quota {quota} for {label}")
        selected_parts.append(source.loc[chosen].copy())

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop(columns=[c for c in selected.columns if c.startswith("_")], errors="ignore")
    selected = selected.sort_values(["model", "source_dataset", "id"], kind="stable").reset_index(drop=True)
    if selected.groupby("model").size().nunique() != 1 or len(selected) != quota * len(labels):
        raise RuntimeError("final dataset is not balanced")
    if selected.id.astype(str).duplicated().any() or selected.text.map(_normal_text).duplicated().any():
        raise RuntimeError("selected dataset contains duplicate IDs or normalized response texts")

    selected.to_parquet(OUTPUT, index=False)
    audit = source.loc[source._reason.ne(""), ["id", "model", "source_dataset", "source_file", "prompt_id", "_reason", "_evidence", "_duplicate_of_id"]].copy()
    audit = audit.rename(columns={"id": "row_id", "_reason": "removal_reason", "_evidence": "evidence", "_duplicate_of_id": "retained_representative_id"})
    selected_ids = set(selected.id.astype(str))
    balancing_mask = source._reason.eq("") & ~source.id.astype(str).isin(selected_ids)
    balancing = source.loc[balancing_mask, ["id", "model", "source_dataset", "source_file", "prompt_id"]].copy()
    balancing["removal_reason"] = "eligible_not_selected_to_balance_classes"
    balancing["evidence"] = f"class count reduced to {quota} to match the smallest eligible class"
    balancing["retained_representative_id"] = ""
    balancing = balancing.rename(columns={"id": "row_id"})
    audit = pd.concat([audit, balancing[audit.columns]], ignore_index=True)
    audit.to_csv(AUDIT, index=False)

    output_hash = _hash_file(OUTPUT)
    class_counts = {str(k): int(v) for k, v in selected.groupby("model").size().sort_index().items()}
    profile = {
        "dataset_version": "v2.14.balanced",
        "dataset_sha256": output_hash,
        "rows": int(len(selected)),
        "classes": labels,
        "class_counts": class_counts,
        "smallest_eligible_class_count": quota,
        "limiting_labels": limiting_labels,
        "eligible_rows_by_class_before_balancing": {str(k): int(v) for k, v in available.items()},
        "balanced_selection_seed": int(seed),
        "target_rows_per_class": quota,
        "balance_rule": "use all eligible rows in the smallest class; select the same count for every other class",
        "selection_method": "deterministic hash ordering; source round-robin and one per prompt group in first pass; no fixed source caps",
        "training_only": True,
        "external_source_rows_only": True,
        "curation_method": "v2_14_quality_filtered_min_class_balanced",
        "input_dataset": str(SOURCE.relative_to(ROOT)),
        "input_dataset_sha256": _hash_file(SOURCE),
        "v0_reference": str(V0.relative_to(ROOT)),
        "v0_reference_rows": int(len(v0)),
        "v0_reference_sha256": _hash_file(V0),
        "normalization": "Unicode NFKC, case-fold, collapse whitespace; punctuation preserved",
        "character_limits_inclusive": {"minimum": int(min_chars), "maximum": int(max_chars)},
        "filter_counts": counts,
        "quality_removed_rows": int((source._reason.ne("")).sum()),
        "balanced_quota_removed_rows": int(balancing_mask.sum()),
        "audit_path": str(AUDIT.relative_to(ROOT)),
        "source_hashes": {
            str(SOURCE.relative_to(ROOT)): _hash_file(SOURCE),
            str(V0.relative_to(ROOT)): _hash_file(V0),
        },
        "flag_policy": {
            "routing": "exclude true explicit routing-risk flags and non-verified B2M Gemini 3.7 g37_rule values",
            "flag_kb_proxy_after_ruling": "retained because it identifies a source route, not an attribution-risk flag",
            "truncation": "source cap/truncation metadata and explicit terminal truncation/failure markers",
            "multi_turn": "source turn-count/boolean metadata and explicit role-prefixed transcript rows",
            "serialized": "explicit serialized chat/tool/function markers",
            "code": "fenced/code-tagged blocks or source-like code signatures",
            "reasoning": "explicit reasoning-trace markers only; generation reasoning settings are retained",
        },
        "v0_isolation": {
            "response_overlap_before_filter": int(v0_overlap.sum()),
            "prompt_id_overlap_before_filter": int(prompt_overlap.sum()),
            "response_overlap_after_selection": 0,
            "prompt_id_overlap_after_selection": 0,
        },
        "notes": [
            "No training or model evaluation was run.",
            "The selected quota is data-dependent and equals the smallest eligible class count; it is not capped at 3,000.",
            "For multi-turn and truncation, only explicit metadata or conservative textual markers are detectable in this source schema; absence of such a marker is not proof about original generation history.",
        ],
    }
    PROFILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# V2.14 Balanced dataset",
        "",
        f"Built from `{SOURCE.relative_to(ROOT)}`. It retains **{quota:,} rows per class** across 18 classes (**{len(selected):,} rows total**), where the quota is the smallest class count remaining after the specified quality filters. There is no 3,000-row cap.",
        "",
        "## Per-class counts",
        "",
        "| Model | Input rows | Eligible after filters | Selected |",
        "|---|---:|---:|---:|",
    ]
    original_counts = source.groupby("model").size().to_dict()
    for label in labels:
        lines.append(f"| {label} | {int(original_counts.get(label, 0)):,} | {int(available[label]):,} | {quota:,} |")
    lines += [f"| **Total** | **{len(source):,}** | **{int(available.sum()):,}** | **{len(selected):,}** |", "", "## Exclusions", "",
              "The factory applied the requested rules: character length 60–2,000 inclusive; exact normalized response overlap with all V0 rows; exact V0 prompt-group overlap for training isolation; flagged routing; source-marked or explicitly marked truncation; source-marked or explicit multi-turn transcripts; serialized chat/tool output; code blocks/source-like code; explicit reasoning traces; non-English or undetermined language; and duplicate/conflicting normalized responses. Text normalization is NFKC + case-fold + collapsed whitespace, retaining punctuation.",
              "", "Generation settings such as `gen_reasoning=high` are not treated as reasoning traces; only explicit trace markers are excluded. `flag_kb_proxy_after_ruling` is retained because it describes route provenance rather than a routing-risk flag. Rows with no turn/truncation metadata are only screened by explicit text markers; the source cannot prove their original generation history.",
              "", "| Removal reason | Rows |", "|---|---:|"]
    reason_counts = audit.removal_reason.value_counts().sort_values(ascending=False)
    for reason, count in reason_counts.items():
        lines.append(f"| {reason} | {int(count):,} |")
    lines += [f"| **Total removed** | **{len(audit):,}** |", "", "## Reproducibility and validation scope", "",
              f"Input SHA-256: `{_hash_file(SOURCE)}`. Dataset SHA-256: `{output_hash}`. Seed: {seed}. Source selection uses deterministic response-row hashes, round-robin source ordering, and one response per prompt group in the first selection pass; no source cap is imposed.",
              "", "The artifact is training-only. The training loader uses full V0 validation; the V0 profile and split are checked when loaded. Row-level exclusions are in [`v2.14.balanced_removals.csv`](../data/processed/v2.14.balanced_removals.csv), and full counts/hashes are in [`dataset_profile_v2.14.balanced.json`](../data/processed/dataset_profile_v2.14.balanced.json).", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = parser.parse_args(argv)
    profile = build(seed=args.seed, min_chars=args.min_chars, max_chars=args.max_chars)
    print(json.dumps({"rows": profile["rows"], "rows_per_class": profile["target_rows_per_class"],
                      "limiting_labels": profile["limiting_labels"], "class_counts": profile["class_counts"],
                      "quality_removed": profile["quality_removed_rows"],
                      "balanced_removed": profile["balanced_quota_removed_rows"]}, indent=2))


if __name__ == "__main__":
    main()
