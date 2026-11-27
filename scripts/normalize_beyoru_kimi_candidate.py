"""Normalize Beyoru's Kimi-K3 canonical splits for NLM quality filtering."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/beyoru_kimi_k3_distillation/data/canonical"
OUTPUT = ROOT / "data/processed/external_candidates/beyoru_kimi_quality_input.parquet"
SPLITS = ("train", "validation", "test")


def main() -> None:
    DetectorFactory.seed = 2026
    frames = []
    for split in SPLITS:
        frame = pd.read_parquet(RAW / f"{split}-00000-of-00001.parquet")
        frame["source_file"] = f"canonical/{split}"
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)

    def language(text: object) -> str:
        try:
            return detect(str(text))
        except Exception:
            return ""

    records = []
    for row in source.to_dict("records"):
        messages = json.loads(row["messages_json"])
        response = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "assistant")
        prompt = "\n".join(f"[{m.get('role', '')}]\n{m.get('content') or ''}"
                            for m in messages if m.get("role") != "assistant")
        record = dict(row)
        record.update({
            "id": str(row["id"]),
            "source_row_id": str(row.get("source_item_id") or row["id"]),
            "source_original_id": str(row.get("source_item_id") or row["id"]),
            "text": response,
            "model": "kimi-k3",
            "source_dataset": "beyoru/kimi-k3-distillation",
            "prompt_id": str(row.get("prompt_cluster_id") or row.get("parent_id") or row["id"]),
            "prompt_canon": prompt,
            "language": language(response),
        })
        records.append(record)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(records):,} normalized rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
