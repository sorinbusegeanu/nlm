"""Build the balanced external-only V2.0 training corpus.

This is a data-only workflow. It reuses the normalization/language helpers from
the data factory and records row-level exclusions in data/processed.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_factory import _hash_file, _language, _normal_text, build_v1_3_external_dataset  # noqa: E402
from src.data_factory import OPENPOOL_VERTEX_G37_SOURCES  # noqa: E402

SEED = 2026
MAX_CHARS = 2000
MIN_CHARS = 50
TARGET = 3000
V0_PATH = ROOT / "data/processed/dataset_v0.parquet"
V1_PATH = ROOT / "data/processed/dataset_v1.parquet"
OPENPOOL = ROOT / "data/raw/open_pool/ntl_open_pool.parquet"
BM_POOL = ROOT / "data/raw/kaggle_ntl_pool_b2m/pool_b2m_public.parquet"
BM_PROV = ROOT / "data/raw/kaggle_ntl_pool_b2m/pool_b2m_provenance.parquet"
V13 = ROOT / "data/processed/dataset_v1.3.parquet"
OUT = ROOT / "data/processed/dataset_v2.0.parquet"
PROFILE = ROOT / "data/processed/dataset_profile_v2.0.json"
REMOVALS = ROOT / "data/processed/v2.0_removals.csv"
REPORT = ROOT / "docs/v2.0.md"
VERSION = "v2.0"
INCLUDE_UNVERIFIED = False


def configure_outputs(version: str, include_unverified: bool) -> None:
    if version not in {"v2.0", "v2.1"}:
        raise ValueError("supported dataset versions are v2.0 and v2.1")
    global VERSION, INCLUDE_UNVERIFIED, OUT, PROFILE, REMOVALS, REPORT
    VERSION, INCLUDE_UNVERIFIED = version, include_unverified
    OUT = ROOT / f"data/processed/dataset_{version}.parquet"
    PROFILE = ROOT / f"data/processed/dataset_profile_{version}.json"
    REMOVALS = ROOT / f"data/processed/{version}_removals.csv"
    REPORT = ROOT / f"docs/{version}.md"


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _frame_defaults(frame: pd.DataFrame, source_type: str) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("id", "text", "model", "prompt_id", "prompt_canon", "source_dataset",
                "source_file", "source_license"):
        if col not in frame:
            frame[col] = ""
    frame["id"] = frame.id.fillna("").astype(str)
    frame["text"] = frame.text.fillna("").astype(str)
    for c in ("model", "prompt_id", "prompt_canon", "source_dataset", "source_file", "source_license"):
        frame[c] = frame[c].fillna("").astype(str)
    frame["_source_type"] = source_type
    frame["_source_row_id"] = frame.get("source_row_id", frame.id).fillna("").astype(str)
    frame["_norm"] = frame.text.map(_normal_text)
    frame["_prompt_norm"] = frame.prompt_canon.map(_normal_text)
    return frame


def load_candidates(labels: set[str]) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    frames = []
    originals: dict[str, int] = {}
    input_hashes: dict[str, str] = {}

    op = pd.read_parquet(OPENPOOL)
    originals["open_pool"] = len(op)
    input_hashes[str(OPENPOOL.relative_to(ROOT))] = _hash_file(OPENPOOL)
    op = op.rename(columns={"prompt_group": "prompt_id"})
    op["prompt_canon"] = op.prompt_canon.fillna("")
    frames.append(_frame_defaults(op, "open_pool"))

    bm = pd.read_parquet(BM_POOL)
    prov = pd.read_parquet(BM_PROV)
    originals["b2m"] = len(bm)
    input_hashes[str(BM_POOL.relative_to(ROOT))] = _hash_file(BM_POOL)
    input_hashes[str(BM_PROV.relative_to(ROOT))] = _hash_file(BM_PROV)
    bm = bm.reset_index(drop=True).reset_index(names="row_index")
    bm["id"] = "b2m_" + bm.row_index.astype(str)
    bm["source_row_id"] = "b2m_row_" + bm.row_index.astype("Int64").astype(str)
    prov = prov.drop_duplicates("row_index", keep="first")
    bm = bm.merge(prov, on="row_index", how="left", suffixes=("", "_prov"))
    source_map = {}
    sources_path = ROOT / "data/raw/kaggle_ntl_pool_b2m/sources.json"
    if sources_path.exists():
        obj = json.loads(sources_path.read_text())
        source_map = {str(x.get("source_key")): x for x in obj.get("sources", [])}
    licenses = {key: str(x.get("licence", "")) for key, x in source_map.items()}
    bm["source_dataset"] = bm.source.map(lambda x: source_map.get(str(x), {}).get("kaggle_slug", f"B2M/{x}"))
    bm["source_file"] = "pool_b2m_public.parquet"
    bm["source_license"] = bm.source.map(lambda x: licenses.get(str(x), ""))
    bm["origin_hint"] = bm.get("route", "")
    bm["prompt_canon"] = ""
    bm["prompt_group"] = "kaggle:mrralphie/ntl-pool-b2m:" + bm.prompt_id.fillna("").astype(str)
    bm["prompt_id"] = bm["prompt_group"]
    frames.append(_frame_defaults(bm, "b2m"))

    # Rebuild a temporary, uncapped normalization from every primary raw V1.3
    # source file. This makes the 1,901-2,000 character band eligible here and
    # lets the V2.0 rules account for >2,000-character removals themselves.
    glm_dir = ROOT / "data/raw/external_candidates/glm53_flash/data"
    glm_files = [glm_dir / f"glm53_harvest{suffix}.parquet" for suffix in ("", "_r2", "_r3", "_r4")]
    glm_prov = glm_dir / "provenance_r34.parquet"
    gpt_file = ROOT / "data/raw/external_candidates/gpt56_sol_luna/data/train.parquet"
    with tempfile.TemporaryDirectory(prefix="v2_0_v13_candidates_") as tmp:
        tmpdir = Path(tmp)
        for name in ("dataset_v0.parquet", "dataset_profile.json", "splits_v0.parquet"):
            shutil.copy2(ROOT / "data/processed" / name, tmpdir / name)
        build_v1_3_external_dataset(glm_files, glm_prov, gpt_file, tmpdir,
                                    dataset_version="v13_raw_candidates", max_response_chars=None)
        v13 = pd.read_parquet(tmpdir / "dataset_v13_raw_candidates.parquet")
    # The normalizer stores prompt/trace metadata but not the raw GLM turn
    # count, so attach it from each primary table for the conversation filter.
    turn_frames = []
    for source_path in glm_files:
        turns = pd.read_parquet(source_path, columns=["id", "response", "turns"])
        turns["source_file"] = source_path.name
        turn_frames.append(turns.rename(columns={"id": "source_row_id", "response": "text", "turns": "source_turns"}))
    turn_lookup = pd.concat(turn_frames, ignore_index=True)
    # Raw IDs can repeat within a source file. Include exact response text and
    # conservatively use the largest turn count for duplicate identical rows.
    turn_lookup = turn_lookup.groupby(["source_file", "source_row_id", "text"], dropna=False, as_index=False).source_turns.max()
    v13 = v13.merge(turn_lookup, on=["source_file", "source_row_id", "text"], how="left", validate="many_to_one")
    originals["v1.3_raw_normalized_rows"] = len(v13)
    for source_path in [*glm_files, glm_prov, gpt_file]:
        input_hashes[str(source_path.relative_to(ROOT))] = _hash_file(source_path)
    frames.append(_frame_defaults(v13, "v1.3_sources"))

    # This separate Gemini collection is explicitly unverified; retain it in
    # the candidate audit but it will fail the valid-attribution rule.
    g37_unverified_path = ROOT / "data/raw/open_pool/gemini37_unverified.parquet"
    g37_unverified = pd.read_parquet(g37_unverified_path)
    originals["gemini37_unverified"] = len(g37_unverified)
    input_hashes[str(g37_unverified_path.relative_to(ROOT))] = _hash_file(g37_unverified_path)
    g37_unverified = g37_unverified.rename(columns={"prompt_group": "prompt_id"})
    g37_unverified["source_model_attribution"] = "unverified_collection"
    frames.append(_frame_defaults(g37_unverified, "gemini37_unverified"))

    # V1 is an explicit fallback pool only. It is ranked behind every candidate
    # not already present in V1, as requested.
    v1 = pd.read_parquet(V1_PATH)
    originals["v1_fallback"] = len(v1)
    input_hashes[str(V1_PATH.relative_to(ROOT))] = _hash_file(V1_PATH)
    frames.append(_frame_defaults(v1, "v1_fallback"))

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["_in_v1"] = False
    v1_norm = set(v1.text.fillna("").astype(str).map(_normal_text)) - {""}
    data["_in_v1"] = data._norm.isin(v1_norm)
    data["_source_type"] = data._source_type.astype(str)
    data["_eligible_label"] = data.model.isin(labels)
    return data, originals, input_hashes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a balanced NLM external-only dataset")
    parser.add_argument("--version", choices=("v2.0", "v2.1"), default="v2.0")
    parser.add_argument("--include-unverified", action="store_true",
                        help="include unflagged rows from the unverified Gemini 3.7 collection")
    args = parser.parse_args(argv)
    configure_outputs(args.version, args.include_unverified)
    from langdetect import DetectorFactory
    DetectorFactory.seed = SEED
    for target in (OUT, PROFILE, REMOVALS, REPORT):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing {VERSION} output: {target}")
    v0 = pd.read_parquet(V0_PATH)
    v0_text = set(v0.text.fillna("").astype(str).map(_normal_text)) - {""}
    v0_prompt_ids = set(v0.prompt_id.fillna("").astype(str)) - {""}
    labels = set(v0.model.dropna().astype(str))
    if len(v0) != 2699 or len(labels) != 18:
        raise ValueError(f"expected official V0 reference to be 2,699 rows/18 labels; got {len(v0)}/{len(labels)}")
    data, originals, input_hashes = load_candidates(labels)
    original_rows_by_source_label = data.groupby(["_source_type", "model"], dropna=False).size().reset_index(name="original_rows")
    data["_reason"] = ""

    def reject(mask: pd.Series, reason: str) -> None:
        m = pd.Series(mask, index=data.index).fillna(False).astype(bool) & data._reason.eq("")
        data.loc[m, "_reason"] = reason

    reject(~data._eligible_label, "invalid_or_noncompetition_label")
    reject(data.model.fillna("").str.strip().eq(""), "missing_model_attribution")
    if not INCLUDE_UNVERIFIED:
        reject(data._source_type.eq("gemini37_unverified"), "model_attribution_unverified_collection")
    reject(data.text.str.strip().eq(""), "empty_response")
    lengths = data.text.str.len()
    reject(lengths.lt(MIN_CHARS), "response_under_50_characters")
    reject(lengths.gt(MAX_CHARS), "response_over_2000_characters")
    # A valid external provenance chain requires a dataset/corpus, file and
    # license; source row IDs are recorded separately where supplied.
    prov_ok = data.source_dataset.str.strip().ne("") & data.source_file.str.strip().ne("") & data.source_license.str.strip().ne("")
    reject(~prov_ok, "missing_source_provenance_or_license")
    reject(data.prompt_id.str.strip().eq(""), "missing_prompt_provenance")
    reject(data.prompt_id.isin(v0_prompt_ids), "official_v0_prompt_id_overlap")
    reject(data._norm.isin(v0_text), "official_v0_response_overlap")
    if "flag_gemini37_routing_risk" in data:
        reject(data.flag_gemini37_routing_risk.map(lambda x: str(x).strip().casefold() in {"true", "1", "yes"}),
               "model_attribution_routing_risk")
    g37 = data.model.eq("gemini-3.7-flash")
    openpool_g37 = data._source_type.isin({"open_pool", "v1_fallback"}) & g37
    reject(openpool_g37 & ~data.source_dataset.isin(OPENPOOL_VERTEX_G37_SOURCES),
           "gemini_3_7_source_not_in_verified_vertex_allowlist")
    if "g37_rule" in data:
        reject((data._source_type.eq("b2m") & g37) & ~data.g37_rule.fillna("").astype(str).str.casefold().eq("verified"),
               "gemini_3_7_b2m_attribution_quarantined")

    # Existing source metadata gives stronger evidence than text heuristics.
    trace_cols = [c for c in ("thinking", "source_category", "source_domain", "source_task", "source_mode", "register") if c in data]
    trace_meta = pd.Series(False, index=data.index)
    for c in trace_cols:
        trace_meta |= data[c].fillna("").astype(str).str.contains(
            r"(?i)(?:reasoning|chain.?of.?thought|thinking|multi.?turn|tool|agentic|coding|code)", regex=True)
    reject(trace_meta, "reasoning_conversation_tool_or_code_source_metadata")
    if "source_turns" in data:
        n_turns = pd.to_numeric(data.source_turns, errors="coerce")
    elif "turns" in data:
        n_turns = pd.to_numeric(data.turns, errors="coerce")
    else:
        n_turns = pd.Series(np.nan, index=data.index)
    if n_turns.notna().any():
        reject(n_turns.gt(1), "multi_turn_source_conversation")
    truncated = pd.Series(False, index=data.index)
    for c in ("source_truncated", "source_cap_hit"):
        if c in data:
            truncated |= data[c].map(lambda x: str(x).strip().casefold() in {"true", "1", "yes"})
    if "source_finish_reason" in data:
        truncated |= data.source_finish_reason.fillna("").astype(str).str.casefold().isin({"length", "max_tokens", "token_limit", "length_capped"})
    reject(truncated, "source_marked_truncated_or_cap_hit")
    reject(data.text.str.contains(r"(?is)\[(?:truncated|incomplete response|generation failed)\]\s*$|\.\.\.\s*\[end\]\s*$", regex=True), "explicit_truncation_marker")

    # Explicit serialization and code-output signatures, deliberately avoiding
    # rejecting ordinary prose that merely discusses software.
    serialized = data.text.str.contains(
        r"(?is)(?:<\|im_start\|>|<\|im_end\|>|<\|tool_call\|>|<tool_call\b|</?arg_value\b|<tool_response\b|<function_call\b|\bassistant\s+to=\w+|^\s*(?:tool call|tool response|function call|execution log)\s*:)", regex=True)
    reject(serialized, "conversation_or_tool_log_serialization")
    code_pattern = data.text.str.contains(
        r"(?is)(?:```|<\/?code>|^\s*(?:#!/|import\s+[\w.]+\s*$|from\s+[\w.]+\s+import\s+|def\s+\w+\s*\(|class\s+\w+\s*[:({]|(?:const|let|var)\s+\w+\s*=))",
        regex=True)
    reject(code_pattern, "code_block_or_programming_output")
    reasoning_pattern = data.text.str.contains(
        r"(?im)(?:^\s*<think>|^\s*\[analysis\]|^\s*(?:chain of thought|reasoning trace)\s*:|^\s*let me think step by step\b)", regex=True)
    reject(reasoning_pattern, "explicit_reasoning_trace")

    # Language detection for rows with no trustworthy English source marker;
    # source language values are trusted only when explicit and recognized.
    explicit_lang = data.get("language", pd.Series("", index=data.index)).fillna("").astype(str).str.casefold()
    lang = explicit_lang.where(explicit_lang.isin({"en", "eng", "english"}), "")
    unknown_idx = data.index[lang.eq("") & data._reason.eq("")]
    if len(unknown_idx):
        unknown_text = data.loc[unknown_idx, "text"]
        # Reuse one deterministic language decision for repeated normalized
        # responses instead of running the detector for duplicate source copies.
        distinct = unknown_text.drop_duplicates(keep="first")
        language_by_text = {text: _language(text) for text in distinct}
        lang.loc[unknown_idx] = unknown_text.map(language_by_text)
    reject(~lang.isin({"en", "eng", "english"}), "non_english_or_language_undetermined")
    data["language"] = lang

    # Exclude every member of an exact normalized-response duplicate group,
    # across all source snapshots, so there is only one owner/source row.
    active = data.loc[data._reason.eq("")]
    dupe = active._norm.duplicated(keep=False)
    dup_norms = set(active.loc[dupe, "_norm"])
    # Exact cross-label conflicts are ambiguous and all copies are removed.
    active_labels = active.groupby("_norm").model.nunique()
    conflicts = set(active_labels[active_labels.gt(1)].index)
    reject(data._norm.isin(conflicts), "duplicate_response_conflicting_model_labels")
    # For same-label source duplicates keep the candidate with best provenance,
    # preferring sources outside V1, then Open Pool/B2M/V1.3, then stable ID.
    active = data.loc[data._reason.eq("")].copy()
    active["_source_rank"] = active._source_type.map({"open_pool": 0, "b2m": 1, "v1.3_sources": 2, "gemini37_unverified": 3, "v1_fallback": 4}).fillna(9)
    active = active.sort_values(["_in_v1", "_source_rank", "id"], kind="stable")
    duplicate_idx = active.index[active._norm.duplicated(keep="first")]
    reject(data.index.isin(duplicate_idx), "duplicate_normalized_response")

    # Deterministic class-wise sampling: first unused responses, one per prompt
    # where feasible; then source-diverse round-robin across prompt groups.
    eligible = data.loc[data._reason.eq("")].copy()
    eligible_counts = eligible.model.value_counts().reindex(sorted(labels), fill_value=0)
    eligible["_source_rank"] = eligible._source_type.map({"open_pool": 0, "b2m": 1, "v1.3_sources": 2, "gemini37_unverified": 3, "v1_fallback": 4}).fillna(9)
    eligible["_seed_order"] = eligible.apply(lambda r: stable_hash(f"{SEED}|{r.model}|{r.prompt_id}|{r.id}"), axis=1)
    quota = min(TARGET, int(eligible.model.value_counts().reindex(sorted(labels), fill_value=0).min()))
    if quota <= 0:
        raise ValueError("one or more labels have no eligible rows after quality filters")
    selected_parts = []
    for label in sorted(labels):
        c = eligible.loc[eligible.model.eq(label)].copy()
        c = c.sort_values(["_in_v1", "_source_rank", "_seed_order"], kind="stable")
        # Greedy round robin: each pass adds at most one row per source and prompt.
        chosen = []
        used_ids, used_prompts = set(), set()
        # Round-robin source rows so a large pool cannot consume the quota
        # before other independent collections have contributed.
        source_rows = {
            source: c.loc[c._source_type.eq(source)].iterrows()
            for source in c._source_type.drop_duplicates().tolist()
        }
        exhausted = set()
        while len(chosen) < quota and len(exhausted) < len(source_rows):
            added = False
            for source in source_rows:
                if source in exhausted or len(chosen) >= quota:
                    continue
                for idx, row in source_rows[source]:
                    if idx in used_ids:
                        continue
                    if row.prompt_id in used_prompts:
                        continue
                    chosen.append(idx); used_prompts.add(row.prompt_id); used_ids.add(idx)
                    added = True
                    break
                else:
                    exhausted.add(source)
            if not added and len(chosen) < quota:
                break
        if len(chosen) < quota:
            rest = c.loc[~c.index.isin(used_ids)].sort_values(["_in_v1", "_seed_order"], kind="stable")
            chosen.extend(rest.index[:quota-len(chosen)].tolist())
        if len(chosen) != quota:
            raise RuntimeError(f"could not reach common quota {quota} for {label}")
        selected_parts.append(data.loc[chosen].copy())
    selected = pd.concat(selected_parts, ignore_index=True)
    version_tag = VERSION.replace(".", "").casefold()
    selected["id"] = [version_tag + "_" + stable_hash(f"{row['_source_type']}|{row['id']}|{row['_norm']}")[:32]
                       for _, row in selected.iterrows()]
    selected["source_row_id"] = selected["_source_row_id"]
    selected["source_type"] = selected["_source_type"]
    selected["was_in_v1"] = selected["_in_v1"].astype(bool)
    # Drop transient curation fields only; retain source-specific metadata.
    selected = selected.drop(columns=[c for c in selected if c.startswith("_") or c == "_reason"], errors="ignore")
    selected = selected.sort_values(["model", "source_type", "id"], kind="stable").reset_index(drop=True)

    # Mark otherwise eligible rows not chosen due to the fixed balanced quota.
    selected_orig_ids = set(selected.source_type.astype(str) + "\0" + selected.source_row_id.astype(str) + "\0" + selected.text.map(_normal_text))
    # Keep rejected and not-selected counts in one row-level audit.
    audit = data.loc[data._reason.ne("")].copy()
    audit["removal_reason"] = audit._reason
    audit["source_type"] = audit._source_type
    chosen_keys = selected_orig_ids
    unselected_mask = data._reason.eq("") & ~((data._source_type.astype(str) + "\0" + data._source_row_id.astype(str) + "\0" + data._norm).isin(chosen_keys))
    unselected = data.loc[unselected_mask].copy()
    unselected["removal_reason"] = "eligible_not_selected_balanced_quota"
    unselected["source_type"] = unselected._source_type
    audit = pd.concat([audit, unselected], ignore_index=True, sort=False)
    audit_cols = [c for c in ["id", "model", "source_type", "source_dataset", "source_file", "source_row_id", "prompt_id", "removal_reason"] if c in audit]
    audit[audit_cols].to_csv(REMOVALS, index=False)

    class_counts = selected.model.value_counts().sort_index().astype(int).to_dict()
    if len(class_counts) != 18 or len(set(class_counts.values())) != 1 or len(selected) > 54000:
        raise RuntimeError("final class balance validation failed")
    if selected.text.map(_normal_text).isin(v0_text).any():
        raise RuntimeError("V0 response overlap found in final dataset")
    if selected.prompt_id.astype(str).isin(v0_prompt_ids).any():
        raise RuntimeError("V0 prompt ID overlap found in final dataset")
    if selected.id.duplicated().any() or selected.text.map(_normal_text).duplicated().any():
        raise RuntimeError("final dataset has duplicate IDs or normalized responses")

    def count_table(frame, by):
        if frame.empty: return []
        counts = frame.groupby(by, dropna=False).size().reset_index(name="rows")
        return counts.to_dict("records")

    raw_v13_profile = json.loads((ROOT / "data/processed/dataset_profile_v1.3.json").read_text())
    removal_counts = audit.groupby(["source_type", "model", "removal_reason"], dropna=False).size().reset_index(name="rows")
    source_diversity = selected.groupby("model").agg(rows=("id", "size"), source_datasets=("source_dataset", "nunique"), prompt_groups=("prompt_id", "nunique"), rows_reused_from_v1=("was_in_v1", "sum")).reset_index()
    selected_in_v1 = int(selected.was_in_v1.sum())
    selected_glm = selected.loc[selected.source_type.eq("v1.3_sources") & selected.source_file.astype(str).str.startswith("glm53_harvest")]
    selected_multi_turn = int(pd.to_numeric(selected_glm.source_turns, errors="coerce").gt(1).sum())
    # Preserve every provenance column and keep an explicit canonical V0-style schema.
    selected.to_parquet(OUT, index=False)
    output_hash = _hash_file(OUT)
    profile = {
        "dataset_version": VERSION, "dataset_sha256": output_hash, "rows": len(selected),
        "seed": SEED, "target_rows_per_class": TARGET, "final_rows_per_class": quota,
        "classes": sorted(labels), "class_counts": class_counts,
        "eligible_rows_by_label_before_balancing": {str(k): int(v) for k, v in eligible_counts.items()},
        "quota_limiting_labels": [str(k) for k, v in eligible_counts.items() if int(v) == quota],
        "training_only": True, "external_source_rows_only": True,
        "unverified_gemini37_included_by_user_request": INCLUDE_UNVERIFIED,
        "attribution_caveat": ("The gemini37_unverified source is explicitly labeled unverified; rows without its routing-risk flag are included at the user's request. Model attribution is not independently confirmed." if INCLUDE_UNVERIFIED else None),
        "v1_fallback_rows_used": int(selected.source_type.eq("v1_fallback").sum()),
        "selected_rows_already_in_v1": selected_in_v1,
        "v1_overlap_definition": "normalized exact response match against data/processed/dataset_v1.parquet",
        "source_original_counts": originals,
        "source_hashes": input_hashes,
        "v0_reference": {"path": str(V0_PATH.relative_to(ROOT)), "rows": len(v0), "sha256": _hash_file(V0_PATH),
                          "unique_prompt_ids": int(v0.prompt_id.nunique()),
                          "prompt_overlap_check": "Exact official prompt IDs; prompt text is not present in the official V0 files, so semantic prompt overlap cannot be verified.",
                          "response_overlap_removed": int((data._norm.isin(v0_text)).sum()),
                          "prompt_id_overlap_removed": int((data.prompt_id.isin(v0_prompt_ids)).sum())},
        "candidate_rows_by_source_type": count_table(data, ["_source_type"]),
        "original_rows_by_source_and_label": original_rows_by_source_label.to_dict("records"),
        "retained_rows_by_source_type": count_table(selected, ["source_type"]),
        "retained_rows_by_source_and_label": count_table(selected, ["source_type", "model"]),
        "retained_source_diversity_by_label": source_diversity.to_dict("records"),
        "selected_source_trace_validation": {"selected_glm_rows_with_turn_metadata": int(selected_glm.source_turns.notna().sum()), "selected_multi_turn_rows": selected_multi_turn},
        "removal_counts_by_source_label_reason": removal_counts.to_dict("records"),
        "removal_audit_path": str(REMOVALS.relative_to(ROOT)),
        "removal_audit_sha256": _hash_file(REMOVALS),
        "deduplication": {"normalized_text": "NFKC, casefold, collapsed whitespace", "same-label_duplicate_retained": "best ranked source (Open Pool, B2M, V1.3, V1 fallback)", "cross-label_duplicate": "all conflicting copies removed"},
        "selection": {"priority": "not in V1 first; then Open Pool, B2M, V1.3, unverified Gemini 3.7 when enabled, V1 fallback; deterministic seed order", "prompt_diversity": "at most one per prompt group during first selection pass", "quota_reduced_below_3000": quota < TARGET},
        "v1_3_raw_sources": {"raw_normalized_candidate_rows": originals["v1.3_raw_normalized_rows"], "source_inputs": {k: v for k, v in input_hashes.items() if "external_candidates/" in k}},
        "quality_policy": {"min_response_chars": MIN_CHARS, "max_response_chars": MAX_CHARS, "language": "English only; source English metadata accepted, otherwise langdetect; unknown excluded", "excluded": ["invalid label", "missing source/file/license/prompt provenance", "exact V0 response or prompt-ID overlap", "duplicate responses", "code/programming signature", "reasoning/conversation/tool traces", "explicit truncation metadata or markers"]},
    }
    PROFILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")

    report = [f"# {VERSION} dataset card", "", "## Summary", "",
              f"{VERSION} contains **{len(selected):,} external training examples**, balanced at **{quota:,} per class** across 18 competition labels. Sampling seed: **{SEED}**. It uses no training or inference.", "",
              "The available V1.3 source rows are included through the normalized V1.3 build output. Open Pool and B2M are ingested from their raw public tables. Rows not already present in V1 were prioritized. If a class did not have enough unique eligible external rows, rows already present in V1 could fill the quota; these are counted below and retain their best external provenance.", "",
              "## Class balance", "", "| Label | Rows | % |", "|---|---:|---:|"]
    for label, n in class_counts.items(): report.append(f"| {label} | {n:,} | {n / len(selected):.2%} |")
    report += ["", "## Source and V1 reuse", "", "| Source type | Retained rows |", "|---|---:|"]
    for row in profile["retained_rows_by_source_type"]: report.append(f"| {row['source_type']} | {row['rows']:,} |")
    report += ["", f"Exact normalized-response overlap with V1 (`dataset_v1.parquet`): **{selected_in_v1:,} selected rows**; rows sourced exclusively from the V1 fallback table: **{int(selected.source_type.eq('v1_fallback').sum()):,}**. Full per-source/per-label original and retained counts are in [`dataset_profile_{VERSION}.json`](../data/processed/dataset_profile_{VERSION}.json); row-level exclusions are in [`{VERSION}_removals.csv`](../data/processed/{VERSION}_removals.csv).", "",
               "## Original and retained counts by source and label", "", "| Source family | Label | Original candidates | Retained |", "|---|---|---:|---:|"]
    original_counts = pd.DataFrame(profile["original_rows_by_source_and_label"]).rename(columns={"_source_type": "source_type"})
    retained_counts = pd.DataFrame(profile["retained_rows_by_source_and_label"])
    source_label_counts = original_counts.merge(retained_counts, on=["source_type", "model"], how="left").fillna({"rows": 0})
    for row in source_label_counts.sort_values(["source_type", "model"]).to_dict("records"):
        report.append(f"| {row['source_type']} | {row['model'] or 'invalid/missing'} | {int(row['original_rows']):,} | {int(row['rows']):,} |")
    report += ["", "## Removal reasons", "", "Counts include both quality exclusions and eligible rows left out after the balanced quota was reached.", "", "| Reason | Rows |", "|---|---:|"]
    for reason, n in audit.removal_reason.value_counts().sort_values(ascending=False).items():
        report.append(f"| {reason} | {int(n):,} |")
    report += ["", "## Cleaning and leakage controls", "",
               f"Rows outside **{MIN_CHARS}–{MAX_CHARS} characters** were removed. Only competition labels with nonempty source dataset, source file, license and prompt-group provenance are retained. English is taken from trusted English metadata when available and otherwise detected; unknown/non-English rows are removed. Explicit code blocks/programming outputs, reasoning or serialized conversation/tool traces, truncation markers/metadata, exact duplicate responses, cross-label response conflicts, exact V0 response matches and exact V0 prompt-ID matches are excluded.", "",
               f"The full V0 reference was checked: **{len(v0):,} rows**, spanning {v0.prompt_id.nunique()} prompt IDs. The official V0 files contain prompt IDs but no prompt text. Therefore exact ID overlap is checked, but semantic prompt-text overlap cannot be independently certified; this limitation is recorded in the profile. The response-overlap check covers all V0 rows, not only its validation split.", "",
              "## Source diversity by label", "", "| Label | Distinct source datasets | Prompt groups |", "|---|---:|---:|"]
    for row in source_diversity.sort_values("model").to_dict("records"):
        report.append(f"| {row['model']} | {int(row['source_datasets'])} | {int(row['prompt_groups']):,} |")
    report += ["", "## Audit and source availability", "",
               f"The profile records original candidate row counts, retained counts per source and label, each removal reason, source hashes, class balance, source diversity and V1 reuse. Exact duplicates are normalized with NFKC, case folding and whitespace collapse. Deterministic selection favors examples not present in V1 and spreads selections across sources and prompt groups. Selected GLM source IDs were checked against raw turn metadata; {selected_multi_turn} selected multi-turn rows were found.", "",
              ("All primary GLM and GPT V1.3 raw files were normalized temporarily for this build without the older 1,900-character cap. `gemini37_unverified.parquet` was included at the user's request; rows flagged with routing risk remain excluded, and model attribution is explicitly unverified. The indexed Claude Opus 4.8 collection was unavailable locally (download manifest records a 404), so it contributed no rows." if INCLUDE_UNVERIFIED else "All primary GLM and GPT V1.3 raw files were normalized temporarily for this build without the older 1,900-character cap, so V2.0 applies its own 2,000-character rule. The separate `gemini37_unverified.parquet` collection was audited but excluded because its model attribution is explicitly unverified. The indexed Claude Opus 4.8 collection was unavailable locally (download manifest records a 404), so it contributed no rows."), "",
              f"Rebuild command: `python -m src.v2_0_dataset --version {VERSION}{' --include-unverified' if INCLUDE_UNVERIFIED else ''}`.", "",
               "No model training, checkpoint modification or inference was performed.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report))
    print(json.dumps({"rows": len(selected), "per_class": quota, "v1_rows": selected_in_v1,
                      "retained_by_source": profile["retained_rows_by_source_type"],
                      "profile": str(PROFILE), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
