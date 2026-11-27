"""Reusable, source-agnostic quality filters for external NLM datasets.

The rules operate on the canonical NLM columns and preserve every input field.
Source adapters are responsible for mapping source language/provenance fields to
the canonical columns before calling :func:`filter_dataset`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from .data_factory import _normal_text


DEFAULT_MIN_CHARS = 50
DEFAULT_MAX_CHARS = 2000
_ARTIFACT = re.compile(
    r"(?is)(?:\[object object\]|\x00)"
    r"|(?:^\s*<\|im_start\|>(?:system|user|assistant|tool)\b)"
    r"|(?:<\|im_start\|>.*?<\|im_end\|>)"
    r"|(?:^\s*(?:tool call|tool response|function call|execution log)\s*:\s*\{.{0,100}\}\s*$)"
)
_TRUNCATION = re.compile(
    r"(?i)(?:\[(?:truncated|incomplete response|generation failed)\]"
    r"|\((?:truncated|generation failed)\)|\.\.\.\s*\[end\])\s*$"
)


def filter_dataset(
    frame: pd.DataFrame,
    *,
    language_column: str = "language",
    english_values: tuple[str, ...] = ("en", "eng", "english"),
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    valid_labels: set[str] | None = None,
    validation: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply deterministic eligibility rules; return retained rows, audit, stats.

    Exclusions are assigned a single first-match reason. Normalized exact
    duplicates with conflicting labels remove all conflicting rows; same-label
    duplicates retain the lexically smallest ID. Near-duplicate screening is
    intentionally left as a separately auditable optional compaction step.
    """
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("invalid response length bounds")
    required = {"id", "prompt_id", "text", "model", "source_dataset", "source_row_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"quality input lacks canonical columns: {sorted(missing)}")
    if language_column not in frame:
        raise ValueError(f"English-only filtering requires language column {language_column!r}")

    data = frame.copy().reset_index(drop=True)
    data["_quality_row"] = range(len(data))
    data["_quality_norm"] = data.text.fillna("").astype(str).map(_normal_text)
    active = pd.Series(True, index=data.index)
    reasons: list[dict] = []

    def reject(mask: pd.Series, reason: str, evidence):
        nonlocal active
        mask = pd.Series(mask, index=data.index).fillna(False).astype(bool) & active
        for i in data.index[mask]:
            row = data.loc[i]
            reasons.append({
                "input_row": int(i), "id": str(row.get("id", "")),
                "source_row_id": str(row.get("source_row_id", "")),
                "model": str(row.get("model", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "source_file": str(row.get("source_file", "")),
                "prompt_id": str(row.get("prompt_id", "")),
                "response_chars": len(str(row.get("text", "") or "")),
                "removal_reason": reason,
                "evidence": str(evidence(row) if callable(evidence) else evidence),
            })
        active.loc[mask] = False
        return mask

    valid = (data.id.notna() & data.id.astype(str).str.strip().ne("")
             & data.prompt_id.notna() & data.prompt_id.astype(str).str.strip().ne("")
             & data.source_row_id.notna() & data.source_row_id.astype(str).str.strip().ne("")
             & data.source_dataset.notna() & data.source_dataset.astype(str).str.strip().ne("")
             & data.model.notna() & data.text.notna() & data._quality_norm.ne(""))
    reject(~valid, "missing_required_field", "ID, prompt, text, label, or source provenance is absent")
    reject(data.id.astype(str).duplicated(keep="first"), "duplicate_response_id",
           "duplicate canonical response ID; first row retained")
    if valid_labels is not None:
        reject(~data.model.astype(str).isin(set(map(str, valid_labels))), "invalid_model_label",
               "label is not in the configured canonical label set")

    normalized_language = data[language_column].fillna("").astype(str).str.strip().str.casefold()
    allowed = {str(value).strip().casefold() for value in english_values}
    reject(~normalized_language.isin(allowed), "not_explicitly_english",
           lambda row: f"{language_column}={row.get(language_column)!r}")

    lengths = data.text.fillna("").astype(str).str.len()
    reject(lengths.lt(min_chars), "response_too_short", f"response length < {min_chars} characters")
    reject(lengths.gt(max_chars), "response_too_long", f"response length > {max_chars} characters")

    for column in ("source_truncated", "source_cap_hit"):
        if column in data:
            explicit = data[column].map(lambda x: str(x).strip().casefold() in {"true", "1", "yes"}
                                        if pd.notna(x) else False)
            reject(explicit, f"source_{column.removeprefix('source_')}",
                   f"source metadata explicitly marks {column}")
    if "source_finish_reason" in data:
        finish = data.source_finish_reason.fillna("").astype(str).str.strip().str.casefold()
        reject(finish.isin({"length", "max_tokens", "token_limit", "length_capped"}),
               "source_length_finish", "source finish reason indicates a length cap")
    reject(data._quality_norm.map(lambda value: bool(_ARTIFACT.search(value))),
           "serialized_or_tool_log_artifact", "matched explicit serialized envelope or tool-log marker")
    reject(data._quality_norm.map(lambda value: bool(_TRUNCATION.search(value))),
           "explicit_truncation_or_failure_marker", "response ends with an explicit truncation/failure marker")

    remaining = data.loc[active]
    duplicate_texts = remaining.groupby("_quality_norm", sort=True).model.nunique()
    conflicts = set(duplicate_texts[duplicate_texts.gt(1)].index)
    reject(data._quality_norm.isin(conflicts), "conflicting_label_duplicate",
           "normalized-identical response has conflicting model labels")

    remaining = data.loc[active]
    duplicate_mask = remaining._quality_norm.duplicated(keep=False)
    duplicate_indices = []
    if duplicate_mask.any():
        # Deterministic representative independent of source file iteration.
        ordered = remaining.loc[duplicate_mask].sort_values(["_quality_norm", "id"], kind="stable")
        duplicate_indices = ordered.index[ordered.duplicated("_quality_norm", keep="first")].tolist()
    reject(data.index.isin(duplicate_indices), "normalized_duplicate_response",
           "normalized-identical response; deterministic first ID retained")

    validation_overlap = {"canonical_id": 0, "source_row_id": 0, "normalized_response": 0,
                          "prompt_id": 0, "canonical_prompt": 0}
    prompt_text_available = False
    if validation is not None:
        v_required = {"id", "prompt_id", "text"}
        if v_required - set(validation.columns):
            raise ValueError(f"validation input lacks columns: {sorted(v_required - set(validation.columns))}")
        v_ids = set(validation.id.dropna().astype(str))
        v_norm = set(validation.text.fillna("").astype(str).map(_normal_text)) - {""}
        v_prompts = set(validation.prompt_id.dropna().astype(str))
        validation_overlap["canonical_id"] = int(data.id.astype(str).isin(v_ids).sum())
        validation_overlap["source_row_id"] = int(data.source_row_id.astype(str).isin(v_ids).sum())
        validation_overlap["normalized_response"] = int(data._quality_norm.isin(v_norm).sum())
        validation_overlap["prompt_id"] = int(data.prompt_id.astype(str).isin(v_prompts).sum())
        reject(data.id.astype(str).isin(v_ids), "validation_id_overlap", "ID appears in validation")
        reject(data.source_row_id.astype(str).isin(v_ids), "validation_source_id_overlap",
               "source row ID appears in validation")
        reject(data._quality_norm.isin(v_norm), "validation_response_overlap", "normalized response appears in validation")
        reject(data.prompt_id.astype(str).isin(v_prompts), "validation_prompt_id_overlap",
               "prompt ID appears in validation")
        if "prompt_canon" in data and "prompt_canon" in validation:
            prompt_text_available = True
            v_prompt_text = set(validation.prompt_canon.fillna("").astype(str).map(_normal_text)) - {""}
            p_norm = data.prompt_canon.fillna("").astype(str).map(_normal_text)
            validation_overlap["canonical_prompt"] = int(p_norm.isin(v_prompt_text).sum())
            reject(p_norm.isin(v_prompt_text), "validation_prompt_text_overlap",
                   "normalized canonical prompt appears in validation")

    kept = data.loc[active].drop(columns=["_quality_row", "_quality_norm"]).copy()
    audit = pd.DataFrame(reasons, columns=["input_row", "id", "source_row_id", "model", "source_dataset",
        "source_file", "prompt_id", "response_chars", "removal_reason", "evidence"])
    stats = {
        "input_rows": int(len(data)), "retained_rows": int(len(kept)), "removed_rows": int(len(audit)),
        "retained_by_label": {str(k): int(v) for k, v in kept.model.value_counts().sort_index().items()},
        "removed_by_reason": {str(k): int(v) for k, v in audit.removal_reason.value_counts().sort_index().items()},
        "language_column": language_column, "english_values": sorted(allowed),
        "min_response_chars": min_chars, "max_response_chars": max_chars,
        "unique_normalized_responses": int(kept.text.fillna("").astype(str).map(_normal_text).nunique()),
        "validation_overlap_counts_before_filter": validation_overlap,
        "validation_prompt_text_available": prompt_text_available,
    }
    return kept.reset_index(drop=True), audit, stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported source format {suffix!r}: {path}")


def run_config(config_path: str | Path) -> dict:
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    input_paths = [Path(p) for p in cfg["input_paths"]]
    frames = []
    for path in input_paths:
        raw = _read_frame(path)
        text_col, source_id_col = cfg.get("text_column", "text"), cfg.get("source_id_column", "id")
        prompt_col = cfg.get("prompt_column", "prompt")
        lang_col = cfg.get("language_column", "language")
        model = cfg.get("model_value")
        source_name = cfg.get("source_dataset", path.stem)
        source_file = path.name
        row_ids = raw[source_id_col].fillna("").astype(str)
        prompts = raw[prompt_col].fillna("").astype(str) if prompt_col in raw else pd.Series("", index=raw.index)
        # Keep the complete source row, then overlay canonical NLM fields and
        # stable source provenance. In particular, preserve raw source ID/text
        # columns when they have different names (as in this GLM export).
        records = raw.copy()
        records["source_original_id"] = row_ids.to_numpy()
        records["source_row_id"] = row_ids.to_numpy()
        records["id"] = ["quality_" + hashlib.sha256(f"{path.name}\0{i}\0{rid}".encode()).hexdigest()[:32]
                          for i, rid in enumerate(row_ids)]
        records["source_file"] = source_file
        source_dataset_column = cfg.get("source_dataset_column")
        records["source_dataset"] = (raw[source_dataset_column].fillna(source_name).astype(str).to_numpy()
                                      if source_dataset_column and source_dataset_column in raw else source_name)
        records["text"] = raw[text_col].to_numpy()
        records["model"] = (raw[cfg["model_column"]].to_numpy() if cfg.get("model_column") else model)
        records["prompt_canon"] = prompts.to_numpy()
        prompt_id_column = cfg.get("prompt_id_column")
        if prompt_id_column and prompt_id_column in raw:
            records["prompt_id"] = raw[prompt_id_column].fillna("").astype(str).to_numpy()
        else:
            records["prompt_id"] = ["external_prompt_" + hashlib.sha256(_normal_text(p).encode()).hexdigest()
                                    if _normal_text(p) else f"external_prompt_{path.name}_{rid}"
                                    for p, rid in zip(prompts, row_ids)]
        records["language"] = (raw[lang_col].to_numpy() if lang_col in raw
                                else cfg.get("language_value", ""))
        provenance_path = cfg.get("provenance_path")
        if provenance_path:
            provenance = pd.read_parquet(provenance_path)
            key = cfg.get("provenance_key", "id")
            if key not in provenance:
                raise ValueError(f"provenance file {provenance_path} lacks join key {key!r}")
            provenance = provenance.drop_duplicates(key, keep="first").set_index(key)
            joined = provenance.reindex(row_ids)
            for output_col, source_col in cfg.get("provenance_columns", {}).items():
                if source_col in joined:
                    records[output_col] = joined[source_col].to_numpy()
        frames.append(records)
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    validation = _read_frame(Path(cfg["validation_path"])) if cfg.get("validation_path") else None
    kept, audit, stats = filter_dataset(candidates,
        language_column="language", english_values=tuple(cfg.get("english_values", ["en", "eng", "english"])),
        min_chars=int(cfg.get("min_response_chars", DEFAULT_MIN_CHARS)),
        max_chars=int(cfg.get("max_response_chars", DEFAULT_MAX_CHARS)),
        valid_labels=set(cfg["valid_labels"]) if cfg.get("valid_labels") else None,
        validation=validation)
    output = Path(cfg["output_path"])
    audit_path = Path(cfg["audit_path"])
    profile_path = Path(cfg["profile_path"])
    for path in (output, audit_path, profile_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(output, index=False)
    audit.to_csv(audit_path, index=False, lineterminator="\n")
    profile = {
        "criteria_version": "nlm_external_quality_v1", "criteria_path": str(cfg_path),
        **stats, "dataset_path": str(output), "dataset_sha256": _sha256(output),
        "audit_path": str(audit_path), "audit_sha256": _sha256(audit_path),
        "source_hashes": {str(p): _sha256(p) for p in input_paths},
        "provenance_sha256": ({str(cfg["provenance_path"]): _sha256(Path(cfg["provenance_path"]))}
                              if cfg.get("provenance_path") else {}),
        "validation_path": cfg.get("validation_path"), "validation_rows": len(validation) if validation is not None else None,
        "limitations": ["Language eligibility relies on the configured source language field.",
                        "Near-duplicate detection is not part of this exact, deterministic filter."],
    }
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply reusable NLM external dataset quality criteria")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_config(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
