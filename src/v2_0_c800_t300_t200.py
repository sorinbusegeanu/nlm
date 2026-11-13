"""Combine the V2.0 character, 300-token, and 200-token training views."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = "v2_0_c800_t300_t200"
SOURCES = (
    "v2.0.chunks800",
    "v2.0.chunks.300TokBalanced",
    "v2.0.chunks.200TokBalanced",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def build_dataset(root: str | Path = ROOT) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    data_dir = root / "data/processed"
    parts: list[pd.DataFrame] = []
    source_info: dict[str, dict] = {}
    classes = None
    for source in SOURCES:
        artifact = data_dir / f"dataset_{source}.parquet"
        profile_path = data_dir / f"dataset_profile_{source}.json"
        if not artifact.is_file() or not profile_path.is_file():
            raise FileNotFoundError(f"missing source artifact/profile for {source}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        artifact_hash = sha256(artifact)
        if profile.get("dataset_sha256") != artifact_hash:
            raise ValueError(f"source artifact hash differs from profile for {source}")
        frame = pd.read_parquet(artifact)
        required = {"id", "text", "model", "prompt_id"}
        if required - set(frame.columns):
            raise ValueError(f"{source} lacks required columns {sorted(required-set(frame.columns))}")
        if frame.id.isna().any() or frame.id.astype(str).duplicated().any():
            raise ValueError(f"{source} has missing or duplicate row IDs")
        if frame[["text", "model", "prompt_id"]].isna().any().any():
            raise ValueError(f"{source} has missing text, labels, or prompt IDs")
        source_classes = sorted(frame.model.astype(str).unique())
        if classes is None:
            classes = source_classes
        elif source_classes != classes:
            raise ValueError(f"label sets differ in {source}")

        # Keep the origin of each row and its view type without allowing the
        # legacy multi-view loader to collapse the concatenated rows back to
        # one sampled view per source response.
        if "view_type" in frame.columns:
            frame["source_view_type"] = frame["view_type"].fillna("unknown").astype(str)
            frame = frame.drop(columns=["view_type"])
        else:
            frame["source_view_type"] = "token_chunk"
        frame["source_dataset_id"] = source
        frame["source_dataset_row_id"] = frame.id.astype(str)
        source_info[source] = {
            "rows": int(len(frame)),
            "artifact": str(artifact.relative_to(root)),
            "dataset_sha256": artifact_hash,
            "profile": str(profile_path.relative_to(root)),
            "profile_sha256": sha256(profile_path),
            "chunking_strategy": profile.get("chunking_strategy", "800_character_multiview"),
        }
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined["id"] = combined.id.astype(str)
    if combined.id.duplicated().any():
        duplicates = combined.loc[combined.id.duplicated(keep=False), "id"].head(10).tolist()
        raise ValueError(f"source datasets have colliding row IDs: {duplicates}")
    if set(combined.model.astype(str).unique()) != set(classes or []):
        raise ValueError("combined data lost one or more labels")
    combined = combined.sort_values("id", kind="stable").reset_index(drop=True)

    v0_profile_path = data_dir / "dataset_profile.json"
    v0_profile = json.loads(v0_profile_path.read_text(encoding="utf-8"))
    v0 = pd.read_parquet(data_dir / "dataset_v0.parquet")
    if combined.id.astype(str).isin(v0.id.astype(str)).any():
        raise ValueError("combined training rows overlap V0 response IDs")
    prompt_overlap = set(combined.prompt_id.astype(str)) & set(v0.prompt_id.astype(str))
    if prompt_overlap:
        raise ValueError(f"combined training rows overlap {len(prompt_overlap)} V0 prompt IDs")
    training_texts = set(combined.text.astype(str).map(_normal_text))
    validation_texts = set(v0.text.astype(str).map(_normal_text))
    text_overlap = training_texts & validation_texts
    if text_overlap:
        raise ValueError(f"combined training rows overlap {len(text_overlap)} normalized V0 responses")

    counts = combined.groupby(["source_dataset_id", "model"], sort=True).size()
    profile = {
        "dataset_version": DATASET,
        "dataset_sha256": "",
        "rows": int(len(combined)),
        "classes": list(classes or []),
        "class_counts": {str(k): int(v) for k, v in combined.model.value_counts().sort_index().items()},
        "training_only": True,
        "validation_dataset_version": "v0",
        "validation_rows": int(v0_profile.get("validation_rows", 0)),
        "validation_split_sha256": v0_profile.get("split_sha256"),
        "validation_ids_sha256": v0_profile.get("validation_ids_sha256"),
        "source_datasets": source_info,
        "source_row_counts": {name: info["rows"] for name, info in source_info.items()},
        "source_class_counts": {
            source: {str(label): int(count) for (src, label), count in counts.items() if src == source}
            for source in SOURCES
        },
        "original_response_count": int(combined.original_response_id.nunique()),
        "combination_policy": "concatenate all rows from the three source datasets; every row is an independent equally weighted training example",
        "multi_view_policy": "source view_type is retained as source_view_type; view_type is omitted so the training loader does not collapse rows into one view per original response",
        "validation_overlap": {"id": 0, "prompt_id": 0, "normalized_response_text": 0},
    }
    return combined, profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT, type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    data_dir = root / "data/processed"
    artifact = data_dir / f"dataset_{DATASET}.parquet"
    profile_path = data_dir / f"dataset_profile_{DATASET}.json"
    report_path = root / f"docs/{DATASET}.md"
    if any(path.exists() for path in (artifact, profile_path, report_path)):
        raise FileExistsError(f"refusing to overwrite existing {DATASET} outputs")
    frame, profile = build_dataset(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact, index=False)
    profile["dataset_sha256"] = sha256(artifact)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# {DATASET}", "",
        "Concatenation of all rows from V2.0 chunks800, V2.0 300-token chunks, and V2.0 200-token chunks.",
        "Each concatenated row is an independent, equally weighted training example. The chunks800 `view_type` field is renamed to `source_view_type` so the training loader does not collapse these rows into one sampled view per original response.", "",
        f"Rows: **{profile['rows']:,}**; labels: **{len(profile['classes'])}**; underlying source responses: **{profile['original_response_count']:,}**.", "",
        "| Source dataset | Rows | Dataset SHA256 |", "|---|---:|---|",
    ]
    for source, info in profile["source_datasets"].items():
        lines.append(f"| {source} | {info['rows']:,} | `{info['dataset_sha256']}` |")
    lines += [
        "", "Validation: uses the full frozen V0 validation artifact; this training artifact contains no V0 IDs, prompts, or normalized responses.",
        f"", f"Combined dataset SHA256: `{profile['dataset_sha256']}`.", "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"dataset": DATASET, "rows": len(frame), "profile": str(profile_path),
                      "dataset_sha256": profile["dataset_sha256"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
