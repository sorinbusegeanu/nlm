"""Normalize the Enaamain Kaggle corpus for the reusable NLM quality filter."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/enaimaan_ntl_openai"
OUTPUT = ROOT / "data/processed/external_candidates/enaimaan_openai_quality_input.parquet"


def main() -> None:
    DetectorFactory.seed = 2026
    source = pd.read_parquet(RAW / "train.parquet")
    prompts = {}
    with (RAW / "prompts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompts[row["response_id"]] = row
    aliases = json.loads((RAW / "lineage-aliases.json").read_text(encoding="utf-8"))

    def language(text: object) -> str:
        try:
            return detect(str(text))
        except Exception:
            return ""

    result = source.copy()
    result["source_dataset"] = "enaimaan/ntl-openai-generated-responses"
    result["source_file"] = "train.parquet"
    result["source_row_id"] = result["id"].astype(str)
    result["source_original_id"] = result["id"].astype(str)
    result["prompt_id"] = result["cv_group"].astype(str).map(lambda value: aliases.get(value, value))
    result["prompt_canon"] = [
        json.dumps(prompts.get(str(response_id), {}).get("messages", []), ensure_ascii=False, sort_keys=True)
        for response_id in result["id"]
    ]
    result["language"] = result["text"].map(language)
    result["source_license"] = result["response_license"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(result):,} normalized rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
