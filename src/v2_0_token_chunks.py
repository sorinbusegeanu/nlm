"""Build V2.0 training datasets from deterministic UltraBERT token windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data_factory import _hash_file


ROOT = Path(__file__).resolve().parents[1]
BASE_DATASET = "v2.0"
TOKENIZER_ID = "dmis-lab/UltraBERT-380M"
TOKENIZER_REVISION = "671a38c087233c0cc72c5bfceea6fffa2822f1db"
DATASET_SPECS = {
    "v2.0.chunks.300TokBalanced": {"max_encoded_length": 300, "stride": 150},
    "v2.0.chunks.200TokBalanced": {"max_encoded_length": 200, "stride": 100},
}
CHUNKING_VERSION = "ultrabert_token_windows_v1"


def load_tokenizer(*, local_files_only: bool = True):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION,
        trust_remote_code=True, local_files_only=local_files_only)


def _encoded_length(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False,
                        padding=False, return_attention_mask=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                        padding=False, return_attention_mask=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(map(int, ids))


def _window_starts(token_count: int, capacity: int, stride: int) -> list[tuple[int, int]]:
    if capacity < 1 or stride < 1 or stride > capacity:
        raise ValueError("token window capacity and stride must be positive, with stride <= capacity")
    if token_count <= capacity:
        return [(0, token_count)]
    windows = []
    start = 0
    while True:
        end = min(start + capacity, token_count)
        windows.append((start, end))
        if end == token_count:
            break
        start += stride
    if len(set(windows)) != len(windows):
        raise RuntimeError("token window generator produced duplicate ranges")
    return windows


def _decode_window(tokenizer, ids: list[int]) -> str:
    try:
        text = tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    except TypeError:
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return str(text)


def _coverages(token_count: int, windows: list[tuple[int, int]]) -> int:
    intervals = sorted(windows)
    covered = 0
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            raise RuntimeError(f"token coverage gap at token offset {cursor}")
        if end > cursor:
            covered += end - max(start, cursor)
            cursor = end
    if cursor < token_count:
        raise RuntimeError(f"last token was not covered: {cursor}/{token_count}")
    return covered


def chunk_response(tokenizer, text: str, *, max_encoded_length: int,
                   stride: int) -> tuple[list[dict], int, int]:
    """Return decoded windows with half-open offsets into content token IDs."""
    ids = _token_ids(tokenizer, text)
    chunks, capacity = _chunks_from_ids(tokenizer, ids,
        max_encoded_length=max_encoded_length, stride=stride)
    return chunks, len(ids), capacity


def _chunks_from_ids(tokenizer, ids: list[int], *, max_encoded_length: int,
                     stride: int) -> tuple[list[dict], int]:
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = int(max_encoded_length) - special_count
    if capacity <= 0:
        raise ValueError("maximum encoded length leaves no room for content tokens")
    windows = _window_starts(len(ids), capacity, int(stride))
    _coverages(len(ids), windows)
    chunks = []
    seen_ranges = set()
    for index, (start, end) in enumerate(windows):
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        window_ids = ids[start:end]
        decoded = _decode_window(tokenizer, window_ids)
        if not decoded:
            raise ValueError(f"tokenizer decoded an empty chunk at offsets {start}:{end}")
        chunks.append({"chunk_index": index, "token_start": start,
                       "token_end": end, "token_count": end-start,
                       "text": decoded})
    return chunks, capacity


def _tokenize_batches(tokenizer, texts, batch_size=256, *, add_special_tokens=False):
    for start in range(0, len(texts), batch_size):
        batch = [str(text) for text in texts[start:start+batch_size]]
        encoded = tokenizer(batch, add_special_tokens=add_special_tokens, truncation=False,
                            padding=False, return_attention_mask=False)
        yield encoded["input_ids"]


def _verify_encoded_lengths(tokenizer, texts, max_encoded_length, batch_size=256):
    lengths = []
    for encoded_batch in _tokenize_batches(tokenizer, texts, batch_size=batch_size,
                                           add_special_tokens=True):
        for ids in encoded_batch:
            length = len(ids)
            if length > max_encoded_length:
                raise RuntimeError(f"generated chunk length {length} exceeds maximum {max_encoded_length}")
            lengths.append(length)
    return lengths


def build_dataset(dataset: str, *, root: str | Path = ROOT, tokenizer=None,
                  batch_size: int = 256) -> tuple[pd.DataFrame, dict]:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported token chunk dataset: {dataset}")
    root = Path(root)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    spec = DATASET_SPECS[dataset]
    max_encoded_length, stride = int(spec["max_encoded_length"]), int(spec["stride"])
    tokenizer = tokenizer or load_tokenizer()
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = max_encoded_length - special_count
    base_path = root / "data/processed/dataset_v2.0.parquet"
    base_profile_path = root / "data/processed/dataset_profile_v2.0.json"
    original = pd.read_parquet(base_path)
    base_profile = json.loads(base_profile_path.read_text())
    required = {"id", "text", "model", "prompt_id"}
    if required - set(original.columns):
        raise ValueError(f"V2.0 is missing required fields: {sorted(required-set(original.columns))}")
    if original.id.duplicated().any() or original.text.isna().any():
        raise ValueError("V2.0 must have unique IDs and complete response text")
    if int(base_profile.get("rows", len(original))) != len(original):
        raise ValueError("V2.0 source row count no longer matches its profile")

    sources = original.to_dict("records")
    records, source_token_counts = [], {}
    texts = [str(source["text"]) for source in sources]
    for source_batch, ids_batch in zip(
            (sources[i:i+batch_size] for i in range(0, len(sources), batch_size)),
            _tokenize_batches(tokenizer, texts, batch_size=batch_size)):
        for source, original_ids in zip(source_batch, ids_batch):
            original_id = str(source["id"])
            original_ids = list(map(int, original_ids))
            chunks, _ = _chunks_from_ids(tokenizer, original_ids,
                max_encoded_length=max_encoded_length, stride=stride)
            original_token_count = len(original_ids)
            source_token_counts[original_id] = original_token_count
            chunk_count = len(chunks)
            for chunk in chunks:
                index = int(chunk["chunk_index"])
                digest = hashlib.sha256(
                    f"{dataset}|{original_id}|{chunk['token_start']}|{chunk['token_end']}".encode()
                ).hexdigest()[:24]
                chunk_id = f"{dataset.rsplit('.', 1)[-1]}_{digest}_c{index:05d}"
                row = dict(source)
                row.update({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "model": str(source["model"]),
                    "label": str(source["model"]),
                    "original_response_id": original_id,
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "token_start": int(chunk["token_start"]),
                    "token_end": int(chunk["token_end"]),
                    "token_count": int(chunk["token_count"]),
                    "original_token_count": int(original_token_count),
                    "chunking_strategy": CHUNKING_VERSION,
                    "tokenizer_id": TOKENIZER_ID,
                    "tokenizer_revision": TOKENIZER_REVISION,
                })
                records.append(row)
    result = pd.DataFrame.from_records(records)
    if result.id.duplicated().any():
        raise RuntimeError("generated token chunk IDs are not unique")
    if result.empty or set(result.model.astype(str)) != set(original.model.astype(str)):
        raise RuntimeError("generated chunks must retain all source labels")
    if result.original_response_id.nunique() != len(original):
        raise RuntimeError("every original response must produce at least one chunk")
    for original_id, count in source_token_counts.items():
        group = result.loc[result.original_response_id.astype(str).eq(original_id)]
        ranges = list(zip(group.token_start.astype(int), group.token_end.astype(int)))
        if _coverages(count, ranges) != count:
            raise RuntimeError(f"original token coverage failed for {original_id}")

    verified_lengths = _verify_encoded_lengths(
        tokenizer, result.text.astype(str).tolist(), max_encoded_length,
        batch_size=batch_size)
    result["encoded_token_count"] = verified_lengths

    v0_path = root / "data/processed/dataset_v0.parquet"
    v0 = pd.read_parquet(v0_path)
    from .data_factory import _normal_text
    v0_texts = set(v0.text.astype(str).map(_normal_text))
    v0_prompts = set(v0.prompt_id.astype(str))
    chunk_texts = result.text.astype(str).map(_normal_text)
    source_texts = original.text.astype(str).map(_normal_text)
    chunk_text_overlap = int(chunk_texts.isin(v0_texts).sum())
    source_text_overlap = int(source_texts.isin(v0_texts).sum())
    prompt_overlap = int(result.prompt_id.astype(str).isin(v0_prompts).sum())
    if chunk_text_overlap or source_text_overlap or prompt_overlap:
        raise ValueError(
            "V2.0 token chunks overlap official V0 by normalized response or prompt ID: "
            f"chunk_text={chunk_text_overlap}, source_text={source_text_overlap}, "
            f"prompt_rows={prompt_overlap}")

    chunk_counts = result.groupby("model").size().sort_index()
    original_counts = original.groupby("model").size().sort_index()
    per_response_counts = result.groupby("original_response_id").size()
    length_counts = Counter(map(int, verified_lengths))
    token_array = np.asarray(verified_lengths, dtype=np.int64)
    token_percentiles = {str(p): float(np.percentile(token_array, p))
                         for p in (0, 25, 50, 75, 90, 95, 99, 100)}
    token_frequency = {str(k): int(v) for k, v in sorted(length_counts.items())}
    batch_cfg = yaml.safe_load((root / "configs/models/ultrabert380.yaml").read_text())
    configured_batch = int(batch_cfg.get("batch_size", 8))
    grad_accum = int(batch_cfg.get("gradient_accumulation_steps", 1))
    profile = {
        "dataset_version": dataset,
        "dataset_sha256": "",
        "rows": int(len(result)),
        "original_response_count": int(len(original)),
        "base_dataset_version": BASE_DATASET,
        "base_dataset_path": str(base_path.relative_to(root)),
        "base_dataset_sha256": _hash_file(base_path),
        "base_profile_sha256": _hash_file(base_profile_path),
        "training_only": True,
        "classes": list(base_profile["classes"]),
        "original_response_class_counts": {str(k): int(v) for k, v in original_counts.items()},
        "chunk_row_class_counts": {str(k): int(v) for k, v in chunk_counts.items()},
        "max_encoded_length": max_encoded_length,
        "stride": stride,
        "special_token_count": special_count,
        "content_token_capacity": capacity,
        "chunking_strategy": CHUNKING_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "chunk_count": int(len(result)),
        "average_chunks_per_response": float(len(result) / len(original)),
        "minimum_chunks_per_response": int(per_response_counts.min()),
        "maximum_chunks_per_response": int(per_response_counts.max()),
        "encoded_token_length_distribution": token_frequency,
        "encoded_token_length_percentiles": token_percentiles,
        "maximum_observed_encoded_length": int(token_array.max()),
        "original_response_coverage": {"covered": int(result.original_response_id.nunique()),
                                       "total": int(len(original)), "fraction": 1.0},
        "original_token_coverage": {"covered": int(sum(source_token_counts.values())),
                                    "total": int(sum(source_token_counts.values())),
                                    "fraction": 1.0},
        "v0_normalized_text_overlap": {"source_responses": source_text_overlap,
                                       "generated_chunks": chunk_text_overlap},
        "v0_prompt_id_overlap_rows": prompt_overlap,
        "v0_rows_used_for_reference": int(len(v0)),
        "estimated_training_steps_per_epoch": {
            "requested_microbatch": configured_batch,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_assumption": configured_batch * grad_accum,
            "optimizer_steps": int(math.ceil(len(result) / (configured_batch * grad_accum))),
            "note": "Estimate uses configured batch size; automatic probing may change effective microbatch."},
        "source_provenance": "Every V2.0 source metadata field is copied to each chunk row.",
        "v0_validation_policy": "The full, unchunked V0 artifact remains a separate validation dataset.",
    }
    return result, profile


def _write_outputs(dataset: str, frame: pd.DataFrame, profile: dict, root: Path) -> dict:
    output_path = root / "data/processed" / f"dataset_{dataset}.parquet"
    profile_path = root / "data/processed" / f"dataset_profile_{dataset}.json"
    report_path = root / "docs" / f"{dataset}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    profile["dataset_sha256"] = _hash_file(output_path)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    lines = [
        f"# {dataset} data card", "",
        f"Derived from V2.0 with the pinned UltraBERT tokenizer `{TOKENIZER_ID}` at revision `{TOKENIZER_REVISION}`. The source has {profile['original_response_count']:,} responses; the generated artifact has {profile['chunk_count']:,} independent chunk rows.", "",
        "Every chunk is an ordinary equally weighted training row. No class balancing, per-response sampling, or loss weighting is applied. Original V2.0 metadata and source attribution are copied to each chunk.", "",
        "## Verification summary", "",
        "| Metric | Result |", "|---|---:|",
        f"| Original V2.0 responses | {profile['original_response_count']:,} |",
        f"| Generated chunks | {profile['chunk_count']:,} |",
        f"| Average chunks per response | {profile['average_chunks_per_response']:.3f} |",
        f"| Min / max chunks per response | {profile['minimum_chunks_per_response']} / {profile['maximum_chunks_per_response']} |",
        f"| Maximum encoded token length | {profile['maximum_observed_encoded_length']} / {profile['max_encoded_length']} |",
        f"| Original response coverage | {profile['original_response_coverage']['covered']:,} / {profile['original_response_coverage']['total']:,} (100%) |",
        f"| Original token coverage | {profile['original_token_coverage']['covered']:,} / {profile['original_token_coverage']['total']:,} (100%) |",
        f"| V0 normalized response overlap, source / chunks | {profile['v0_normalized_text_overlap']['source_responses']} / {profile['v0_normalized_text_overlap']['generated_chunks']} |",
        f"| V0 prompt-ID overlap rows | {profile['v0_prompt_id_overlap_rows']} |",
        f"| Estimated optimizer steps / epoch | {profile['estimated_training_steps_per_epoch']['optimizer_steps']:,} |",
        "", "## Encoded token lengths", "", "| Tokens incl. special tokens | Rows |", "|---:|---:|",
    ]
    for length, count in sorted(profile["encoded_token_length_distribution"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {length} | {count:,} |")
    lines += ["", "## Chunk rows by label", "", "| Label | Original responses | Chunks |", "|---|---:|---:|"]
    for label in profile["classes"]:
        lines.append(f"| {label} | {profile['original_response_class_counts'].get(label, 0):,} | {profile['chunk_row_class_counts'].get(label, 0):,} |")
    lines += ["", "## Training and evaluation", "",
        f"Train UltraBERT with `python -m src.pipeline models train --model ultrabert380 --dataset {dataset} --validation-dataset v0 --accept-config-defaults`. The pipeline uses a maximum encoded length of {profile['max_encoded_length']} for this dataset. Validation, checkpoint selection, and inference use complete unchunked V0 responses.", "",
        f"Dataset artifact: `data/processed/{output_path.name}`. Profile: `data/processed/{profile_path.name}`. Dataset SHA256: `{profile['dataset_sha256']}`.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"dataset": dataset, "rows": int(len(frame)), "artifact": str(output_path),
            "profile": str(profile_path), "report": str(report_path),
            "dataset_sha256": profile["dataset_sha256"]}


def build_all(*, root: str | Path = ROOT, tokenizer=None, force: bool = False,
              batch_size: int = 256) -> list[dict]:
    root = Path(root)
    tokenizer = tokenizer or load_tokenizer()
    results = []
    for dataset in DATASET_SPECS:
        output_path = root / "data/processed" / f"dataset_{dataset}.parquet"
        profile_path = root / "data/processed" / f"dataset_profile_{dataset}.json"
        report_path = root / "docs" / f"{dataset}.md"
        if not force and any(path.exists() for path in (output_path, profile_path, report_path)):
            raise FileExistsError(f"refusing to overwrite {dataset}; use --force")
        frame, profile = build_dataset(dataset, root=root, tokenizer=tokenizer,
                                       batch_size=batch_size)
        results.append(_write_outputs(dataset, frame, profile, root))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(json.dumps(build_all(force=args.force, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
