"""Normalize the Qwen-labelled rows from the Saidutta clean corpus."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external_candidates/saidutta_qwen_glm_kimi_distillation_clean/data/canonical_clean"
OUTPUT = ROOT / "data/processed/external_candidates/saidutta_qwen_glm_kimi_qwen_quality_input.parquet"


def main() -> None:
    DetectorFactory.seed = 2026
    frames = [pd.read_parquet(RAW / f"{split}.parquet") for split in ("train", "validation", "test")]
    source = pd.concat(frames, ignore_index=True)
    label_map = {"qwen3.8-max-preview": "qwen3.8-max", "kimi-code/k3": "kimi-k3", "glm-5.2": "glm-5.2"}
    records = []
    for row in source.to_dict("records"):
        messages = json.loads(row["messages_json"])
        response = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "assistant")
        prompt = "\n".join(f"[{m.get('role', '')}]\n{m.get('content') or ''}"
                            for m in messages if m.get("role") in {"system", "user"})
        model = label_map.get(str(row.get("teacher_model")), str(row.get("teacher_model") or ""))
        try:
            language = detect(response) if model == "qwen3.8-max" and 50 <= len(response) <= 2000 else ""
        except Exception:
            language = ""
        record = dict(row)
        record.update({
            "id": str(row.get("id") or ""),
            "source_row_id": str(row.get("source_item_id") or row.get("id") or ""),
            "source_original_id": str(row.get("source_item_id") or row.get("id") or ""),
            "text": response, "model": model,
            "source_dataset": "saidutta69/qwen-glm-kimi-distillation-clean",
            "source_file": f"canonical_clean/{row.get('split', '')}",
            "prompt_id": str(row.get("prompt_cluster_id") or row.get("parent_id") or row.get("id") or ""),
            "prompt_canon": prompt, "language": language,
        })
        records.append(record)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(records):,} normalized rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
