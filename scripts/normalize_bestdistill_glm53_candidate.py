"""Normalize the Best Distill GLM-5.3-Flash corpus for NLM quality filtering."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/best_distill_glm53_flash_distillation_chat"
INPUT = RAW / "train-00000-of-00001.parquet"
OUTPUT = ROOT / "data/processed/external_candidates/best_distill_glm53_quality_input.parquet"


def main() -> None:
    DetectorFactory.seed = 2026
    source = pd.read_parquet(INPUT)

    def language(text: object) -> str:
        try:
            return detect(str(text))
        except Exception:
            return ""

    result = source.copy()
    result["id"] = [f"bestdistill_{i:06d}_{int(index)}" for i, index in enumerate(result.source_index)]
    result["source_row_id"] = result["source_index"].astype(str)
    result["source_original_id"] = result["source_index"].astype(str)
    result["text"] = result["response"].fillna("").astype(str)
    result["model"] = "glm-5.3-flash"
    result["source_dataset"] = "best-distill/glm-5.3-flash-distillation-chat"
    result["source_file"] = "data/train-00000-of-00001.parquet"
    result["prompt_canon"] = result["instruction"].fillna("").astype(str)
    result["prompt_id"] = result["prompt_canon"].map(
        lambda value: "bestdistill_prompt_" + hashlib.sha256(" ".join(value.casefold().split()).encode()).hexdigest()
    )
    result["language"] = result["text"].map(language)
    result["source_finish_reason"] = result["finish"].fillna("").astype(str)
    result["source_truncated"] = result["finish"].astype(str).str.casefold().eq("length")
    result["source_license"] = "other; synthetic distillation corpus; see source LICENSE"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(result):,} normalized rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
