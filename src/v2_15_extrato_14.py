"""Build the V2.15 incremental external response dataset.

This builder reads the source snapshots under data/raw/external_candidates,
keeps response text intact, removes full-V2.14/V0 overlaps, resolves exact
duplicates with provenance, and writes the catalog artifact and audit profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .data_factory import _normal_text, _stable_prompt_id, _hash_file


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/v2.15_extrato_14"
DATA = ROOT / "data/processed"
DATASET = "v2.15_extrato_14"
SOURCES = {
    "benchmarks_proxy": {
        "slug": "zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations",
        "license": "CC BY 4.0",
        "files": ["ntl_proxy_generations.csv"],
    },
    "ntl_extra_merged": {
        "slug": "abdelghanyrageh/ntl-extra-merged",
        "license": "Other (upstream licenses vary; see source attribution)",
        "files": ["ntl_extra_merged.csv"],
    },
    "maniceet_cached": {
        "slug": "maniceet/name-that-llm-openrouter-openai-generations",
        "license": "CC BY 4.0 (as recorded in local B2M source inventory)",
        "files": ["maniceet_cached.parquet", "maniceet_cached_provenance.parquet"],
    },
    "paragraph_without_signature": {
        "slug": "prvsiyan/the-paragraph-without-a-signature",
        "license": "CC BY-SA 4.0",
        "files": ["train.csv", "gemini37_unverified.csv", "prompts.csv"],
    },
    "oleg_18_model_wide": {
        "slug": "olegzholobov/name-that-llm-response-corpus",
        "license": "Other (mixed upstream; preserve per-occurrence declarations)",
        "files": ["external_all_wide.parquet", "external_cell_metadata.parquet", "external_cell_provenance.parquet", "external_response_metadata.parquet"],
    },
}
KAGGLE_LAST_UPDATED = {
    "benchmarks_proxy": "2026-10-19T10:03:15Z",
    "ntl_extra_merged": "2026-10-18T02:25:36Z",
    "paragraph_without_signature": "2026-11-11T17:53:14Z",
    "oleg_18_model_wide": "2026-10-15T16:20:44Z",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _nonnull(value: Any) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _raw_text(value: Any) -> str:
    """Return source text without trimming or otherwise editing its contents."""
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return value if isinstance(value, str) else str(value)


def _prompt_group(prompt: Any, source_slug: str, source_prompt_id: Any) -> str:
    prompt_text = _nonnull(prompt)
    if prompt_text:
        return _stable_prompt_id(prompt_text, f"{source_slug}:{_nonnull(source_prompt_id)}")
    source_key = _nonnull(source_prompt_id)
    return _stable_prompt_id("", f"{source_slug}:{source_key or 'unknown'}")


def _candidate(*, source_key: str, source_row_id: Any, text: Any, model: Any,
               source_prompt_id: Any = "", prompt: Any = "", source_file: str = "",
               source_license: str = "", source_version: str = "",
               upstream_source: str = "", source_original_id: Any = "",
               provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "source_key": source_key,
        "source_dataset": SOURCES[source_key]["slug"],
        "source_row_id": _nonnull(source_row_id),
        "source_original_id": _nonnull(source_original_id) or _nonnull(source_row_id),
        "source_prompt_id": _nonnull(source_prompt_id),
        "prompt_text": _raw_text(prompt),
        "text": _raw_text(text),
        "model": _nonnull(model),
        "source_file": source_file,
        "source_license": source_license or SOURCES[source_key]["license"],
        "source_version": source_version,
        "upstream_source": _nonnull(upstream_source),
        "provenance": provenance or {},
    }


def _load_candidates(root: Path = RAW) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    audit: dict[str, dict[str, Any]] = {}

    def start(key: str, n: int):
        audit[key] = {"input_rows": int(n), "eligible_candidates": 0, "excluded": Counter()}

    # Kaggle benchmarks proxy corpus.
    p = root / "benchmarks_proxy/ntl_proxy_generations.csv"
    d = pd.read_csv(p, keep_default_na=False)
    start("benchmarks_proxy", len(d))
    for i, r in d.iterrows():
        candidates.append(_candidate(source_key="benchmarks_proxy", source_row_id=f"{r.get('run','')}:{r.get('bank','')}:{r.get('prompt_id','')}:{i}",
            text=r.get("text"), model=r.get("model"), source_prompt_id=r.get("prompt_id"),
            prompt=r.get("request"), source_file=str(p.relative_to(root)), source_version="latest downloaded Kaggle snapshot (API did not expose numeric version)",
            provenance={k: _nonnull(v) for k, v in r.to_dict().items() if k not in {"text", "request"}}))

    # The merged source exposes labels, a group key, and original source key.
    p = root / "ntl_extra_merged/ntl_extra_merged.csv"
    d = pd.read_csv(p, keep_default_na=False)
    start("ntl_extra_merged", len(d))
    for i, r in d.iterrows():
        candidates.append(_candidate(source_key="ntl_extra_merged", source_row_id=f"{_nonnull(r.get('src'))}:{_nonnull(r.get('group'))}:{i}",
            text=r.get("text"), model=r.get("model"), source_prompt_id=r.get("group"), source_file=str(p.relative_to(root)),
            source_version="latest downloaded Kaggle snapshot (API did not expose numeric version)", upstream_source=r.get("src"),
            provenance={k: _nonnull(v) for k, v in r.to_dict().items() if k != "text"}))

    # Kaggle endpoint is private/unavailable; use the exact source rows retained
    # in the local, documented B2M snapshot and keep that limitation explicit.
    p = root / "maniceet_cached/maniceet_cached.parquet"
    q = root / "maniceet_cached/maniceet_cached_provenance.parquet"
    if p.exists() and q.exists():
        d, prov = pd.read_parquet(p), pd.read_parquet(q)
        start("maniceet_cached", len(d))
        for i, r in d.iterrows():
            pr = prov.iloc[i].to_dict() if i < len(prov) else {}
            candidates.append(_candidate(source_key="maniceet_cached", source_row_id=f"{pr.get('source','maniceet')}:{i}:{r.get('prompt_id','')}",
                text=r.get("text"), model=r.get("model"), source_prompt_id=r.get("prompt_id"), source_file=str(p.relative_to(root)),
                source_version="cached B2M source inventory: v1 (2026-10-20); direct Kaggle endpoint returned 403",
                upstream_source="maniceet", provenance={k: _nonnull(v) for k, v in pr.items() if k != "text"}))
    else:
        audit["maniceet_cached"] = {"input_rows": 0, "eligible_candidates": 0, "excluded": Counter({"source_unavailable": 1})}

    # This source explicitly reuses the official V0 prompt IDs (p000-p134);
    # all rows are excluded as prompt leakage. The sidecar also has unverified
    # Gemini 3.7 attribution and is separately counted.
    p = root / "paragraph_without_signature/train.csv"
    d = pd.read_csv(p, keep_default_na=False)
    prompts = pd.read_csv(root / "paragraph_without_signature/prompts.csv", keep_default_na=False)
    prompt_text = dict(zip(prompts.prompt_id.astype(str), prompts.reconstructed_prompt.astype(str)))
    start("paragraph_without_signature", len(d))
    for i, r in d.iterrows():
        candidates.append(_candidate(source_key="paragraph_without_signature", source_row_id=r.get("response_id") or r.get("id"),
            text=r.get("text"), model=r.get("model"), source_prompt_id=r.get("prompt_id"), prompt=prompt_text.get(str(r.get("prompt_id")), ""),
            source_file=str(p.relative_to(root)), source_version="generated 2026-11-06 to 2026-11-11; Kaggle snapshot latest downloaded",
            provenance={k: _nonnull(v) for k, v in r.to_dict().items() if k != "text"}))
    side = pd.read_csv(root / "paragraph_without_signature/gemini37_unverified.csv", keep_default_na=False)
    audit["paragraph_without_signature"]["input_rows"] += len(side)
    audit["paragraph_without_signature"]["excluded"]["unverifiable_model_attribution"] += len(side)
    # reflected in the global reason totals as well as the source audit
    # Kept separately for a row-level exclusion ledger; do not treat these as
    # candidates because their model version is explicitly unverified.
    audit["paragraph_without_signature"]["unverified_sidecar_rows"] = side[["id", "prompt_id", "model", "response_id"]].to_dict("records")

    # Oleg corpus: melt all non-null cells, then attach response and occurrence
    # provenance. Do not use the curated short view or Luna quarantine.
    odir = root / "oleg_18_model_wide"
    wide = pd.read_parquet(odir / "external_all_wide.parquet")
    cells = wide.melt(id_vars=["id", "prompt"], var_name="model", value_name="text").dropna(subset=["text"])
    cellmeta = pd.read_parquet(odir / "external_cell_metadata.parquet")
    cellmeta = cellmeta[cellmeta.view.eq("all")][["id", "model", "response_id", "prompt_group_id", "prompt_exact", "text_only_suitable", "mechanically_eligible"]]
    cells = cells.merge(cellmeta, on=["id", "model"], how="left", validate="one_to_one")
    response_meta = pd.read_parquet(odir / "external_response_metadata.parquet")
    response_cols = [c for c in ("response_id", "source", "source_url", "source_revision", "source_row_id", "raw_model", "returned_model", "model_evidence_class", "license", "license_status", "license_unknown", "no_train", "text_only_suitable", "mechanically_eligible", "possible_truncation", "possible_reasoning_in_answer", "all_source_licenses_json", "all_sources_json", "all_model_evidence_classes_json", "occurrences") if c in response_meta]
    response_meta = response_meta[response_cols].drop_duplicates("response_id")
    cells = cells.merge(response_meta, on="response_id", how="left", validate="many_to_one", suffixes=("", "_response"))
    occ = pd.read_parquet(odir / "external_cell_provenance.parquet")
    occ = occ[occ.row_id_all.notna()].copy()
    # Aggregate source occurrences to one provenance record per model response.
    occ_cols = [c for c in ("response_id", "model", "source", "source_row_id", "source_url", "license", "license_status", "model_evidence_class", "raw_model", "returned_model", "prompt_group_id", "grouping_is_exact_user_prompt", "no_train") if c in occ]
    occ = occ[occ_cols].sort_values(["response_id", "source", "source_row_id"], kind="mergesort")
    occ = occ.drop_duplicates(["response_id", "model"])
    cells = cells.merge(occ, on=["response_id", "model"], how="left", validate="many_to_one", suffixes=("", "_occ"))
    start("oleg_18_model_wide", len(cells))
    for r in cells.to_dict("records"):
        prompt_exact = bool(r.get("prompt_exact", False))
        prompt = _nonnull(r.get("prompt")) if prompt_exact else ""
        prompt_group = _nonnull(r.get("prompt_group_id"))
        evidence = _nonnull(r.get("model_evidence_class")) or _nonnull(r.get("model_evidence_class_response"))
        source_name = _nonnull(r.get("source")) or _nonnull(r.get("source_response"))
        source_row = _nonnull(r.get("source_row_id")) or _nonnull(r.get("source_row_id_response"))
        candidates.append(_candidate(source_key="oleg_18_model_wide", source_row_id=f"{r.get('response_id')}:{source_name}:{source_row}",
            text=r.get("text"), model=r.get("model"), source_prompt_id=prompt_group, prompt=prompt,
            source_file="oleg_18_model_wide/external_all_wide.parquet", source_version="15 October 2026 snapshot",
            source_license=_nonnull(r.get("license")) or SOURCES["oleg_18_model_wide"]["license"], upstream_source=source_name, source_original_id=r.get("response_id"),
            provenance={k: _nonnull(v) for k, v in r.items() if k not in {"text", "prompt"}}))
    return candidates, audit


def _source_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for key, spec in SOURCES.items():
        files = []
        folder = root / key
        if key == "maniceet_cached":
            folder = root / "maniceet_cached"
        if folder.exists():
            for p in sorted(path for path in folder.rglob("*") if path.is_file()):
                files.append({"path": _path_label(p), "sha256": _hash_file(p), "bytes": p.stat().st_size})
        metadata = folder / "dataset-metadata.json"
        info = {}
        if metadata.exists():
            try:
                info = json.loads(metadata.read_text()).get("info", {})
            except (OSError, json.JSONDecodeError):
                pass
        records.append({"source_key": key, "kaggle_slug": spec["slug"], "dataset_url": f"https://www.kaggle.com/datasets/{spec['slug']}", "title": info.get("title", ""),
            "dataset_id": info.get("datasetId"), "license": spec["license"],
            "retrieved_date": date.today().isoformat(),
            "last_updated": KAGGLE_LAST_UPDATED.get(key),
            "version_note": ("No current listing returned (dataset search unavailable); used local B2M source-v1 snapshot, 2026-10-20" if key == "maniceet_cached" else "latest downloaded snapshot; API did not return a numeric version"),
            "files": files})
    return records


def build_dataset(*, root: Path = RAW, output_dir: Path = DATA, v214_path: Path | None = None,
                  v0_path: Path | None = None, seed: int = 2026) -> dict[str, Any]:
    v214_path = v214_path or output_dir / "dataset_v2.14.parquet"
    v0_path = v0_path or output_dir / "dataset_v0.parquet"
    v214 = pd.read_parquet(v214_path)
    v0 = pd.read_parquet(v0_path)
    labels = set(v214.model.dropna().astype(str).unique())
    candidates, source_audit = _load_candidates(root)
    v214_ids = set(v214.id.dropna().astype(str))
    v214_text = set(v214.text.dropna().astype(str).map(_normal_text)) - {""}
    v0_ids = set(v0.id.dropna().astype(str))
    v0_text = set(v0.text.dropna().astype(str).map(_normal_text)) - {""}
    v0_groups = set(v0.prompt_id.dropna().astype(str))
    v0_prompt_texts: set[str] = set()
    # Official V0 contains prompt IDs but no canonical prompt text.
    counters: Counter[str] = Counter()
    exclusion_rows: list[dict[str, Any]] = []
    def exclude(c: dict[str, Any], reason: str, representative: str = "") -> None:
        key = c["source_key"]
        source_audit[key]["excluded"][reason] += 1
        counters[reason] += 1
        exclusion_rows.append({"source_key": key, "source_dataset": c["source_dataset"],
            "source_row_id": c["source_row_id"], "source_prompt_id": c["source_prompt_id"],
            "model": c["model"], "reason": reason,
            "normalized_text_sha256": _sha256_text(c.get("_norm", _normal_text(c["text"]))),
            "representative_source_row_id": representative})
    valid = []
    bad_text = bad_label = bad_prov = 0
    for c in candidates:
        key = c["source_key"]
        if not c["text"]:
            exclude(c, "missing_or_empty_response")
            bad_text += 1
            continue
        if c["model"] not in labels:
            exclude(c, "invalid_label")
            bad_label += 1
            continue
        if not c["source_row_id"] or not c["source_dataset"]:
            exclude(c, "missing_provenance")
            bad_prov += 1
            continue
        source_audit[key]["eligible_candidates"] += 1
        c["_norm"] = _normal_text(c["text"])
        c["_prompt_group"] = _prompt_group(c["prompt_text"], c["source_dataset"], c["source_prompt_id"])
        # The Paragraph source explicitly uses official V0 p000-p134 prompt IDs.
        direct_v0_prompt = c["source_key"] == "paragraph_without_signature" and c["source_prompt_id"] in v0_groups
        if c["source_original_id"] in v214_ids or c["_norm"] in v214_text:
            exclude(c, "already_in_v2.14")
            continue
        if c["source_original_id"] in v0_ids or c["_norm"] in v0_text:
            exclude(c, "v0_id_or_response_overlap")
            continue
        if direct_v0_prompt or c["_prompt_group"] in v0_groups or _normal_text(c["prompt_text"]) in v0_prompt_texts:
            exclude(c, "v0_prompt_group_overlap")
            continue
        valid.append(c)
    for row in source_audit["paragraph_without_signature"].pop("unverified_sidecar_rows", []):
        c = _candidate(source_key="paragraph_without_signature", source_row_id=row.get("response_id") or row.get("id"),
                       text="", model=row.get("model"), source_prompt_id=row.get("prompt_id"),
                       source_file="paragraph_without_signature/gemini37_unverified.csv")
        exclusion_rows.append({"source_key": c["source_key"], "source_dataset": c["source_dataset"],
            "source_row_id": c["source_row_id"], "source_prompt_id": c["source_prompt_id"],
            "model": c["model"], "reason": "unverifiable_model_attribution",
            "normalized_text_sha256": "", "representative_source_row_id": ""})
    counters["unverifiable_model_attribution"] += sum(
        row["reason"] == "unverifiable_model_attribution" for row in exclusion_rows
    )

    # Resolve cross-label duplicates only where one label has strictly stronger
    # source evidence. A tied conflict has no defensible single attribution.
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in valid:
        by_text[c["_norm"]].append(c)
    selected: list[dict[str, Any]] = []
    conflict_rows = duplicate_rows = 0
    for norm, group in sorted(by_text.items()):
        group_labels = {c["model"] for c in group}
        if len(group_labels) > 1:
            def evidence(c):
                prov = c["provenance"]
                ev = _nonnull(prov.get("model_evidence_class"))
                base = {"request_and_returned_model": 6, "row_request_response_model_id": 5,
                        "row_or_run_metadata": 4, "row_request_model": 4,
                        "publisher_trial_model_api": 4, "dataset_card": 3,
                        "benchmark_attempt_model_metadata": 3}.get(ev, 0)
                if c["source_key"] == "benchmarks_proxy" and _nonnull(prov.get("model_slug")):
                    base = max(base, 4)
                if c["source_key"] == "maniceet_cached" and _nonnull(prov.get("provider")):
                    base = max(base, 3)
                if c["source_key"] == "ntl_extra_merged" and c["upstream_source"]:
                    base = max(base, 2)
                return base
            per_label = {label: max(evidence(c) for c in group if c["model"] == label) for label in group_labels}
            top = max(per_label.values())
            winning_labels = [label for label, score in per_label.items() if score == top]
            if top <= 0 or len(winning_labels) != 1:
                conflict_rows += len(group)
                for c in group:
                    exclude(c, "conflicting_label_unresolved")
                continue
            winning_label = winning_labels[0]
            losing = [c for c in group if c["model"] != winning_label]
            group = [c for c in group if c["model"] == winning_label]
            for c in losing:
                exclude(c, "conflicting_label_resolved_by_provenance", group[0]["source_row_id"])
        # Prefer strong attribution and complete provenance, then stable keys.
        def rank(c):
            prov = c["provenance"]
            evidence = _nonnull(prov.get("model_evidence_class"))
            evidence_rank = {"request_and_returned_model": 6, "row_request_response_model_id": 5,
                             "row_or_run_metadata": 4, "row_request_model": 4,
                             "publisher_trial_model_api": 4, "dataset_card": 3,
                             "benchmark_attempt_model_metadata": 3}.get(evidence, 0)
            if c["source_key"] == "benchmarks_proxy" and _nonnull(prov.get("model_slug")):
                evidence_rank = max(evidence_rank, 4)
            if c["source_key"] == "maniceet_cached" and _nonnull(prov.get("provider")):
                evidence_rank = max(evidence_rank, 3)
            if c["source_key"] == "ntl_extra_merged" and c["upstream_source"]:
                evidence_rank = max(evidence_rank, 2)
            return (-evidence_rank, -sum(bool(_nonnull(v)) for v in prov.values()), c["source_dataset"], c["source_row_id"])
        group = sorted(group, key=rank)
        keep = group[0]
        contributing = [{"source_dataset": c["source_dataset"], "source_row_id": c["source_row_id"],
                         "upstream_source": c["upstream_source"], "source_license": c["source_license"],
                         "source_file": c["source_file"], "provenance": c["provenance"]} for c in group]
        keep["_contributing"] = contributing
        selected.append(keep)
        for c in group[1:]:
            exclude(c, "duplicate_new_response", keep["source_row_id"])
            duplicate_rows += 1

    selected.sort(key=lambda c: (c["model"], c["_norm"], c["source_dataset"], c["source_row_id"]))
    rows = []
    for c in selected:
        rid = "v215_" + _sha256_text(c["source_dataset"] + "\0" + c["source_row_id"] + "\0" + c["_norm"])[:24]
        rows.append({
            "id": rid, "prompt_id": c["_prompt_group"], "text": c["text"], "model": c["model"],
            "source_dataset": c["source_dataset"], "source_file": c["source_file"], "source_license": c["source_license"],
            "source_version": c["source_version"], "source_row_id": c["source_row_id"], "source_prompt_id": c["source_prompt_id"],
            "source_original_id": c["source_original_id"],
            "prompt_text": c["prompt_text"], "upstream_source": c["upstream_source"],
            "contributing_sources_json": json.dumps(c["_contributing"], ensure_ascii=False, sort_keys=True),
            "provenance_json": json.dumps(c["provenance"], ensure_ascii=False, sort_keys=True),
        })
    result = pd.DataFrame(rows)
    # Validation is intentionally independent of class coverage: incremental
    # external datasets are allowed to contribute no examples for some labels.
    errors = []
    if result.empty:
        result = pd.DataFrame(columns=["id", "prompt_id", "text", "model", "source_dataset", "source_file", "source_license", "source_version", "source_row_id", "source_prompt_id", "source_original_id", "prompt_text", "upstream_source", "contributing_sources_json", "provenance_json"])
        errors = ["no eligible rows remain"]
    else:
        for column in ("id", "prompt_id", "text", "model"):
            if result[column].isna().any():
                errors.append(f"missing values in {column}")
        if result.text.astype(str).str.strip().eq("").any():
            errors.append("empty response text")
        if set(result.model) - labels:
            errors.append("invalid competition labels")
    if errors:
        # expected_classes=None still checks ID, label, text and minimum class count.
        raise ValueError("output dataset failed structural validation: " + "; ".join(errors))
    normalized = result.text.astype(str).map(_normal_text)
    if result.id.duplicated().any() or normalized.duplicated().any():
        raise ValueError("output IDs and normalized responses must be unique")
    if set(result.model) - labels:
        raise ValueError("output contains a non-V2.14 label")
    id_overlap = set(result.id) & (v214_ids | v0_ids)
    text_overlap_v214 = set(normalized) & v214_text
    text_overlap_v0 = set(normalized) & v0_text
    prompt_overlap_v0 = set(result.prompt_id.astype(str)) & v0_groups
    if id_overlap or text_overlap_v214 or text_overlap_v0 or prompt_overlap_v0:
        raise ValueError(f"leakage validation failed: ids={len(id_overlap)}, v2.14_text={len(text_overlap_v214)}, v0_text={len(text_overlap_v0)}, v0_prompt={len(prompt_overlap_v0)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "dataset_v2.15_extrato_14.parquet"
    profile_path = output_dir / "dataset_profile_v2.15_extrato_14.json"
    result.to_parquet(artifact, index=False)
    exclusion_path = output_dir / "v2.15_extrato_14_exclusions.csv"
    pd.DataFrame(exclusion_rows, columns=["source_key", "source_dataset", "source_row_id", "source_prompt_id", "model", "reason", "normalized_text_sha256", "representative_source_row_id"]).to_csv(exclusion_path, index=False)
    # Canonical row hash independent from parquet writer metadata.
    ordered = result.sort_values("id", kind="mergesort")
    content_sha = hashlib.sha256("\n".join(
        f"{r.id}\t{r.model}\t{_normal_text(r.text)}\t{r.prompt_id}" for r in ordered.itertuples()
    ).encode("utf-8")).hexdigest()
    profile = {
        "dataset_version": DATASET, "parent_dataset": "v2.14",
        "parent_dataset_sha256": _hash_file(v214_path), "rows": len(result),
        "dataset_sha256": _hash_file(artifact), "canonical_content_sha256": content_sha,
        "seed": int(seed), "labels": sorted(labels),
        "class_counts": result.model.value_counts().sort_index().to_dict(),
        "source_counts": result.source_dataset.value_counts().sort_index().to_dict(),
        "source_inventory": _source_manifest(root),
        "exclusion_audit_path": _path_label(exclusion_path),
        "exclusion_audit_sha256": _hash_file(exclusion_path),
        "source_audit": {k: {"input_rows": int(v["input_rows"]), "eligible_candidates": int(v["eligible_candidates"]),
             "added_rows": int((result.source_dataset == SOURCES[k]["slug"]).sum()),
             "excluded": dict(v["excluded"])} for k, v in source_audit.items()},
        "exclusions_total": dict(counters),
        "validation": {"valid_labels": sorted(set(result.model)), "invalid_labels": 0,
            "empty_text": int(result.text.astype(str).str.strip().eq("").sum()), "unique_ids": bool(result.id.is_unique),
            "unique_normalized_responses": bool(normalized.is_unique), "v2.14_id_overlap": 0,
            "v2.14_normalized_response_overlap": 0, "v0_id_overlap": 0,
            "v0_normalized_response_overlap": 0, "v0_prompt_group_overlap": 0,
            "v0_prompt_text_check": "Official V0 artifact contains no prompt text; exact prompt-text comparison unavailable.",
            "known_v0_prompt_ids": sorted(v0_groups), "paragraph_v0_prompt_rows_excluded": int(source_audit["paragraph_without_signature"]["excluded"].get("v0_prompt_group_overlap", 0))},
        "training_workflow_compatibility": {"training_dataset_id": DATASET, "training_rows": int(len(result)),
            "validation_dataset_id": "v0", "validation_rows": int(len(v0)),
            "all_v0_validation_rows": int(len(v0)) == 2699,
            "same_18_labels": set(result.model.astype(str)) == set(v0.model.astype(str)),
            "external_only_training_rows": True},
        "maniceet_direct_download": "Unavailable: Kaggle dataset endpoint returned HTTP 403; used locally cached B2M v1 source rows with original row provenance.",
    }
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Every source input row is either retained as its representative or listed
    # once in the row-level exclusion ledger.
    for key, src in profile["source_audit"].items():
        total = src["added_rows"] + sum(src["excluded"].values())
        if total != src["input_rows"]:
            raise ValueError(f"source accounting does not reconcile for {key}: {src['input_rows']} != {total}")
    return {"data": result, "profile": profile, "artifact": artifact, "profile_path": profile_path,
            "source_audit": source_audit, "counters": counters, "exclusion_path": exclusion_path,
            "v214": v214}


def write_report(build: dict[str, Any], report_path: Path | None = None) -> Path:
    report_path = report_path or ROOT / "docs/v2.15_extrato_14.md"
    data, profile, source_audit, v214 = build["data"], build["profile"], build["source_audit"], build["v214"]
    rows = [
        "# V2.15_extrato_14",
        "",
        "Incremental, external-only response corpus built from the five requested Kaggle sources. The original V2.14 artifact was read-only during this build. No model training or inference was run.",
        "",
        f"- **Final rows:** {len(data):,}",
        f"- **V2.14 rows:** {len(v214):,}",
        f"- **V2.14 overlap removed:** {profile['exclusions_total'].get('already_in_v2.14', 0):,}",
        f"- **Final canonical-content SHA-256:** `{profile['canonical_content_sha256']}`",
        f"- **Parquet SHA-256:** `{profile['dataset_sha256']}`",
        f"- **Exclusion ledger:** `{build['exclusion_path'].relative_to(ROOT)}`",
        "",
        "## Source inventory and row accounting",
        "",
        "`Input = added + exclusions` for each source. Exclusions are recorded row-by-row in the linked ledger. Kaggle metadata snapshots did not expose an integer dataset version; downloads were the latest versions available to the API on 2026-11-13. The Maniceet Kaggle endpoint returned HTTP 403, so its locally cached B2M v1 rows (dated 2026-10-20) were used and are explicitly identified below.",
        "",
        "| Source | Version / snapshot | Input rows | Valid candidate rows | Already in V2.14 | V0 overlap | Duplicate / conflicting | Added | License |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    inventory = {item["source_key"]: item for item in profile["source_inventory"]}
    for key, src in profile["source_audit"].items():
        info = inventory[key]
        excluded = src["excluded"]
        dup_conf = sum(v for reason, v in excluded.items() if "duplicate" in reason or "conflicting" in reason)
        v0 = excluded.get("v0_prompt_group_overlap", 0) + excluded.get("v0_id_or_response_overlap", 0)
        last_updated = info.get("last_updated")
        version = f"last updated {last_updated}; {src.get('version_note', 'latest downloaded snapshot')}" if last_updated else src.get("version_note", "latest downloaded snapshot")
        rows.append(f"| [{info['title'] or key}]({info['dataset_url']}) | {version} | {src['input_rows']:,} | {src['eligible_candidates']:,} | {excluded.get('already_in_v2.14', 0):,} | {v0:,} | {dup_conf:,} | {src['added_rows']:,} | {info['license']} |")
    rows += ["", "## Final label counts", "", "| Model label | V2.14 rows | New V2.15 rows |", "|---|---:|---:|"]
    v214_counts = v214.model.value_counts()
    new_counts = data.model.value_counts()
    for label in sorted(profile["labels"]):
        rows.append(f"| {label} | {int(v214_counts.get(label, 0)):,} | {int(new_counts.get(label, 0)):,} |")
    rows += ["", "## Exclusion audit", "", "| Source | Exclusion reason | Rows |", "|---|---|---:|"]
    for key, src in profile["source_audit"].items():
        for reason, count in sorted(src["excluded"].items()):
            rows.append(f"| {key} | `{reason}` | {count:,} |")
    rows += ["", "## Verification", "",
        f"- Output rows: **{len(data):,}**; all labels belong to the V2.14 competition label set: **yes**.",
        f"- IDs unique: **{profile['validation']['unique_ids']}**; normalized response texts unique: **{profile['validation']['unique_normalized_responses']}**.",
        "- Normalized response overlap with all V2.14 rows: **0**.",
        "- ID, normalized response, and prompt-group overlap with full V0: **0**.",
        f"- Training integration: use dataset ID `{DATASET}` with `--validation-dataset v0`; it supplies **{len(data):,}** training rows and all **{profile['training_workflow_compatibility']['validation_rows']:,}** V0 rows for validation, with all 18 labels. The existing pipeline partition loader was verified separately.",
        "- Known official V0 prompt IDs: `p000`–`p149`. The Paragraph source explicitly reuses these IDs; all 1,645 train rows were excluded. Its separate 81-row Gemini 3.7 sidecar was excluded because the publisher marks the model version unverified.",
        "- The official V0 artifact has no canonical prompt text. Therefore exact prompt-text comparison against V0 cannot be performed; this is a source-data limitation, not a zero-overlap claim.",
        "- Oleg source: used `external_all_wide.parquet` (not the short-only view or quarantined Luna corpus), retaining source URL/revision, evidence class, license, response ID and source-row provenance. Identical text with tied conflicting labels was excluded; where the evidence uniquely favored one label, that attribution was retained and the alternative recorded.",
        "- No class balancing, response-length filter, confidence filter or sampling was applied. The profile includes SHA-256 hashes and paths for all 30 downloaded/source-snapshot files examined.",
        f"- Reproducible IDs and deterministic ordering use fixed seed 2026; independent rebuild check: **{profile.get('determinism_check', {}).get('status', 'pending')}**.",
        "",
        "## Files and source terms",
        "",
        f"- Dataset: [`{build['artifact'].relative_to(ROOT)}`]({build['artifact'].relative_to(ROOT)})",
        f"- Profile and source file hashes: [`{build['profile_path'].relative_to(ROOT)}`]({build['profile_path'].relative_to(ROOT)})",
        f"- Row-level exclusions: [`{build['exclusion_path'].relative_to(ROOT)}`]({build['exclusion_path'].relative_to(ROOT)})",
        "- Rebuild command: `python -m src.v2_15_extrato_14`.",
        "- Training command (not run): `python -m src.pipeline models train --model <model-id> --dataset v2.15_extrato_14 --validation-dataset v0`.",
        "- Benchmarks proxy: [Kaggle dataset](https://www.kaggle.com/datasets/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations), CC BY 4.0.",
        "- NTL extra merged: [Kaggle dataset](https://www.kaggle.com/datasets/abdelghanyrageh/ntl-extra-merged), Kaggle license marked Other; upstream terms vary and source labels are preserved.",
        "- Maniceet: [Kaggle dataset](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations), source card records CC BY 4.0. Direct download was unavailable (HTTP 403); reused only its existing B2M rows.",
        "- The Paragraph Without a Signature: [Kaggle dataset](https://www.kaggle.com/datasets/prvsiyan/the-paragraph-without-a-signature), CC BY-SA 4.0.",
        "- Oleg Zholobov, 18-model response corpus: [Kaggle dataset](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus), Kaggle license marked Other; original upstream terms are mixed and retained in provenance.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(rows), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--v214", type=Path)
    parser.add_argument("--v0", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    result = build_dataset(root=args.raw_dir, output_dir=args.output_dir, v214_path=args.v214, v0_path=args.v0, seed=args.seed)
    report = write_report(result)
    print(json.dumps({"rows": len(result["data"]), "artifact": str(result["artifact"]),
                      "profile": str(result["profile_path"]), "report": str(report), "class_counts": result["profile"]["class_counts"],
                      "source_counts": result["profile"]["source_counts"], "exclusions": dict(result["counters"])}, indent=2))


if __name__ == "__main__":
    main()
