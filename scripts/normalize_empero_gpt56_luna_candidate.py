"""Normalize Empero's GPT-5.6 Luna SFT snapshot for NLM quality filtering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/empero_gpt56_luna_sft_900x"
OUTPUT = ROOT / "data/processed/external_candidates/empero_gpt56_luna_quality_input.parquet"


def main() -> None:
    DetectorFactory.seed = 2026
    records = []
    with (RAW / "sft.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            messages = row.get("messages", [])
            response = "\n".join(m.get("content", "") for m in messages if m.get("role") == "assistant")
            prompt = "\n".join(f"[{m.get('role', '')}]\n{m.get('content', '')}"
                                 for m in messages if m.get("role") != "assistant")
            try:
                language = detect(response)
            except Exception:
                language = ""
            prompt_id = "empero_prompt_" + hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()
            records.append({
                "id": str(row.get("id", "")), "source_row_id": str(row.get("id", "")),
                "source_original_id": str(row.get("id", "")), "text": response,
                "model": "gpt-5.6-luna", "source_dataset": "empero-ai/gpt-5.6-luna-sft-900x",
                "source_file": "sft.jsonl", "prompt_id": prompt_id, "prompt_canon": prompt,
                "language": language, "topic": row.get("topic"),
                "teacher_model": row.get("model"),
                "source_license": "Empero License: modified MIT, attribution and non-commercial condition",
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(records):,} normalized rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
