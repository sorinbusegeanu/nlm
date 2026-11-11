"""Create the v2.0.chunks800 multi-view training artifact from V2.0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_factory import _hash_file


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
CHUNK_CHARS = 800
MAX_CHUNKS = 3
BASE_DATASET = "v2.0"
DATASET = "v2.0.chunks800"


def _seed_for(original_id: str, seed: int) -> int:
    value = hashlib.sha256(f"{seed}|{original_id}".encode()).digest()
    return int.from_bytes(value[:8], "big")


def _snap_start(text: str, target: int, max_start: int, radius: int = 48) -> int:
    low, high = max(0, target-radius), min(max_start, target+radius)
    candidates = {target}
    for pos in range(low, high + 1):
        if pos == 0 or text[pos-1].isspace():
            candidates.add(pos)
    return min(candidates, key=lambda pos: (abs(pos-target), pos))


def _chunk_window(text: str, start: int, width: int = CHUNK_CHARS) -> tuple[int, int]:
    end = min(len(text), start + width)
    if end < len(text) and not text[end-1].isspace():
        # Trim at most 48 characters to keep the chunk's right edge at a word
        # boundary; very long words keep the hard character limit instead.
        floor = max(start + width - 48, start + 1)
        boundary = max((pos for pos in range(floor, end) if text[pos].isspace()), default=None)
        if boundary is not None:
            end = boundary
    return start, end


def chunk_starts(text: str, *, seed: int = SEED, original_id: str = "",
                 max_chunks: int = MAX_CHUNKS, width: int = CHUNK_CHARS) -> list[tuple[int, int]]:
    """Choose up to three reproducible windows from separate response regions."""
    if len(text) <= width:
        return []
    max_start = len(text) - width
    bins = np.linspace(0, max_start + 1, max_chunks + 1, dtype=int)
    rng = np.random.default_rng(_seed_for(original_id, seed))
    starts: list[int] = []
    for region in range(max_chunks):
        lo, hi = int(bins[region]), max(int(bins[region]), int(bins[region+1])-1)
        target = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        snapped = _snap_start(text, target, max_start)
        if snapped not in starts:
            starts.append(snapped)
    # Short responses may have fewer than three distinct starts after word
    # boundary snapping. Fill remaining slots with spread-out valid offsets.
    if len(starts) < max_chunks:
        candidates = np.linspace(0, max_start, min(max_chunks * 8, max_start + 1), dtype=int)
        for target in candidates:
            snapped = _snap_start(text, int(target), max_start)
            if snapped not in starts:
                starts.append(snapped)
            if len(starts) >= max_chunks:
                break
    starts = sorted(starts[:max_chunks])
    windows = [_chunk_window(text, start, width) for start in starts]
    # Multiple starts can converge to the same word boundary for very short
    # responses; retain unique offsets only.
    return list(dict.fromkeys(windows))


def build_dataset(*, root: str | Path = ROOT, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    base_path = root / "data/processed/dataset_v2.0.parquet"
    profile_path = root / "data/processed/dataset_profile_v2.0.json"
    base_profile = json.loads(profile_path.read_text())
    full_v0_profile_path = root / f"data/processed/dataset_profile_{BASE_DATASET}_full_v0.json"
    v0_profile_path = (full_v0_profile_path if full_v0_profile_path.is_file()
                       else root / "data/processed/dataset_profile.json")
    v0_profile = (json.loads(v0_profile_path.read_text())
                  if v0_profile_path.is_file() else {})
    original = pd.read_parquet(base_path)
    required = {"id", "text", "model", "prompt_id"}
    if required - set(original.columns):
        raise ValueError(f"V2.0 is missing required fields: {sorted(required-set(original.columns))}")
    if original.id.duplicated().any() or original.text.isna().any():
        raise ValueError("V2.0 must have unique IDs and complete response text")

    records = []
    views_per_original = {}
    for source in original.to_dict("records"):
        original_id = str(source["id"])
        text = str(source["text"])
        original_length = len(text)
        views = [("full_response", 0, original_length, text)]
        if original_length > CHUNK_CHARS:
            for number, (start, end) in enumerate(chunk_starts(
                    text, seed=seed, original_id=original_id), start=1):
                views.append(("chunk", start, end, text[start:end]))
        views_per_original[original_id] = len(views)
        for chunk_id, (view_type, start, end, view_text) in enumerate(views):
            row = dict(source)
            suffix = "full" if view_type == "full_response" else f"chunk{chunk_id:02d}"
            row.update({
                "id": f"v20c800_{hashlib.sha256(original_id.encode()).hexdigest()[:24]}_{suffix}",
                "text": view_text,
                "label": str(source["model"]),
                "original_response_id": original_id,
                "chunk_id": "full" if view_type == "full_response" else f"chunk-{chunk_id:02d}",
                "char_start": int(start), "char_end": int(end),
                "original_length": int(original_length),
                "view_type": view_type,
                "dataset_view_seed": int(seed),
            })
            records.append(row)
    result = pd.DataFrame.from_records(records)
    if result.id.duplicated().any():
        raise RuntimeError("generated view IDs are not unique")
    if (result.loc[result.view_type.eq("chunk"), "text"].str.len() > CHUNK_CHARS).any():
        raise RuntimeError("a generated chunk exceeds the character limit")
    for original_id, group in result.groupby("original_response_id", sort=False):
        full = group[group.view_type.eq("full_response")]
        if len(full) != 1 or full.iloc[0].text != original.loc[original.id.astype(str).eq(original_id), "text"].iloc[0]:
            raise RuntimeError(f"full-response training view is not preserved: {original_id}")
        chunks = group[group.view_type.eq("chunk")]
        if len(chunks) > MAX_CHUNKS or (chunks.model != full.iloc[0].model).any():
            raise RuntimeError(f"invalid chunk count or label for {original_id}")

    counts = result.groupby(["model", "view_type"]).size().unstack(fill_value=0)
    profile = {
        "dataset_version": DATASET,
        "dataset_sha256": "",
        "rows": int(len(result)),
        "original_response_count": int(len(original)),
        "seed": int(seed),
        "base_dataset_version": BASE_DATASET,
        "base_dataset_path": str(base_path.relative_to(root)),
        "base_dataset_sha256": _hash_file(base_path),
        "base_profile_sha256": _hash_file(profile_path),
        "training_only": True,
        "validation_dataset_version": "v0",
        "validation_rows": int(v0_profile.get("validation_rows", 0)),
        "validation_split_sha256": v0_profile.get("validation_split_sha256",
                                                   v0_profile.get("split_sha256")),
        "validation_ids_sha256": v0_profile.get("validation_ids_sha256"),
        "validation_prompt_groups": v0_profile.get("validation_prompt_groups"),
        "classes": list(base_profile["classes"]),
        "class_counts": {str(k): int(v) for k, v in original.model.value_counts().sort_index().items()},
        "final_rows_per_class": int(original.model.value_counts().min()),
        "original_response_class_counts": {str(k): int(v) for k, v in original.model.value_counts().sort_index().items()},
        "view_row_class_counts": {str(k): int(v) for k, v in result.model.value_counts().sort_index().items()},
        "view_counts_by_label": {str(label): {str(kind): int(value) for kind, value in row.items()}
                                 for label, row in counts.iterrows()},
        "view_type_counts": {str(k): int(v) for k, v in result.view_type.value_counts().items()},
        "sampling": {"policy": "uniformly select one view per original response at each epoch",
                     "choices": ["complete response", "up to three chunk views"],
                     "epoch_seed": "training seed plus epoch; deterministic per original response",
                     "character_window": CHUNK_CHARS, "max_chunks_per_response": MAX_CHUNKS,
                     "word_boundaries": "snap starts and trim ends near whitespace when practical"},
        "source_provenance": "All original V2.0 fields are preserved on each derived view.",
        "deduplication_and_filtering": "Inherited exactly from V2.0; chunk views are derived only after selection.",
    }
    return result, profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = ROOT
    output = root / "data/processed/dataset_v2.0.chunks800.parquet"
    profile_path = root / "data/processed/dataset_profile_v2.0.chunks800.json"
    report_path = root / "docs/v2.0.chunks800.md"
    if not args.force and any(path.exists() for path in (output, profile_path, report_path)):
        raise FileExistsError("refusing to overwrite existing v2.0.chunks800 outputs; use --force")
    frame, profile = build_dataset(root=root, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    profile["dataset_sha256"] = _hash_file(output)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    per_label_chunks = frame[frame.view_type.eq("chunk")].groupby("model").size()
    report = [
        "# v2.0.chunks800 data card", "",
        f"Derived from V2.0 ({profile['original_response_count']:,} original responses). It preserves each complete response and adds up to {MAX_CHUNKS} deterministic 800-character chunk views for responses longer than 800 characters.", "",
        f"Generated rows: **{len(frame):,}** ({profile['view_type_counts'].get('full_response', 0):,} full responses and {profile['view_type_counts'].get('chunk', 0):,} chunks). Seed: **{args.seed}**.", "",
        "The source V2.0 filtering, labels, deduplication, and provenance are inherited unchanged. Chunks store `original_response_id`, `chunk_id`, half-open character offsets, original response length, label, and view type. Training samples exactly one view per original response each epoch; validation and inference use complete responses.", "",
        "## Views by label", "", "| Label | Original responses | Chunk views |", "|---|---:|---:|",
    ]
    originals_by_label = frame[frame.view_type.eq("full_response")].groupby("model").size()
    for label in profile["classes"]:
        report.append(f"| {label} | {int(originals_by_label.get(label, 0)):,} | {int(per_label_chunks.get(label, 0)):,} |")
    report += ["", f"Base V2.0 SHA256: `{profile['base_dataset_sha256']}`.",
               f"Chunk artifact SHA256: `{profile['dataset_sha256']}`.", "",
               "Training example count per epoch remains equal to the V2.0 original response count. Per-epoch sampled view counts and sampling provenance are recorded in each training run.", "",
               "## Commands", "",
               "Build or regenerate the artifact with `python -m src.pipeline data build-v2-0-chunks800` (add `--force` to overwrite). Train any supported model with `python -m src.pipeline models train --model ultrabert380 --dataset v2.0.chunks800 --validation-dataset v0`; replace the model name to use another configured classifier.", ""]
    report_path.write_text("\n".join(report))
    print(json.dumps({"dataset": DATASET, "rows": len(frame), **profile["view_type_counts"],
                      "profile": str(profile_path.relative_to(root)), "report": str(report_path.relative_to(root))}, indent=2))


if __name__ == "__main__":
    main()
