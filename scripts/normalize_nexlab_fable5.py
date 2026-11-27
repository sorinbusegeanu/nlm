"""Normalize Nexlab's FABLE.5 agentic conversations for NLM quality filtering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data_factory import _normal_text


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/nexlab_fable5_agentic_coding_sft/sft_curated_full.jsonl"
OUTPUT = ROOT / "data/processed/external_candidates/nexlab_fable5_quality_input.parquet"
SOURCE = "Nexlab/fable5-agentic-coding-sft"


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    records = []
    with RAW.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            item = json.loads(line)
            messages = item.get("messages") or []
            answers = [m.get("content") for m in messages
                       if m.get("role") == "assistant" and isinstance(m.get("content"), str)
                       and m["content"].strip()]
            response = answers[-1].strip() if answers else ""
            prompts = "\n".join(str(m.get("content") or "").strip() for m in messages
                                 if m.get("role") == "user" and isinstance(m.get("content"), str)
                                 and m.get("content", "").strip())
            row_id = f"{row_index:06d}"
            prompt_id = ("nexlab_fable5_prompt_" + hashlib.sha256(_normal_text(prompts).encode()).hexdigest()[:24]
                         if _normal_text(prompts) else f"nexlab_fable5_prompt_missing_{row_id}")
            records.append({
                "source_original_id": row_id,
                "source_row_id": row_id,
                "source_file": RAW.name,
                "source_dataset": SOURCE,
                "source_license": "MIT",
                "source_revision": "eeb020da9e0b5c41c60f6bb7b6c1ba5ff5dba379",
                "source_label_basis": "corpus-level FABLE.5 attribution; no per-row generator field",
                "prompt_id": prompt_id,
                "prompt_canon": prompts,
                "text": response,
                "model": "claude-fable-5",
                "language": "en",  # Source card declares the collection English/monolingual.
            })
    output = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(output):,} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
