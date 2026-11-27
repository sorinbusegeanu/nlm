"""Normalize the Kimi-K3 Premium agent_traces view for NLM filtering."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/saidutta_kimi_k3_premium/agent_traces"
OUTPUT = ROOT / "data/processed/external_candidates/saidutta_kimi_premium_quality_input.parquet"


def main() -> None:
    frames = [pd.read_parquet(RAW / f"{split}.parquet") for split in ("train", "validation", "test")]
    source = pd.concat(frames, ignore_index=True)
    records = []
    for row in source.to_dict("records"):
        messages = json.loads(row["messages"])
        response = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "assistant")
        prompt = "\n".join(f"[{m.get('role', '')}]\n{m.get('content') or ''}"
                            for m in messages if m.get("role") == "user")
        result = dict(row)
        result.update({
            "id": str(row["session_id"]), "source_row_id": str(row["session_id"]),
            "source_original_id": str(row["session_id"]), "text": response,
            "model": "kimi-k3", "source_dataset": "saidutta69/Kimi-K3-Premium",
            "source_file": "agent_traces", "prompt_id": str(row["session_id"]),
            "prompt_canon": prompt, "language": "en",
        })
        records.append(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(records):,} normalized rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
