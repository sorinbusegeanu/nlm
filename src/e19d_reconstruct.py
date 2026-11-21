"""Resumable Qwen3.5-4B prompt reconstruction for the E19-D comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import e19b_reconstruct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e19d"
MODEL_ID = "Qwen/Qwen3.5-4B"
# Pin the Hub snapshot observed when this experiment was prepared.
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROMPT_VERSION = "e19d_prompt_v1"


def reconstruct_v0(root=ROOT, *, batch_size=40, max_new_tokens=1024,
                    limit=None, retry_truncated=False, seed=2026):
    """Run the E19-B reconstruction prompt/model interface with the 4B model."""
    OUT.mkdir(parents=True, exist_ok=True)
    return e19b_reconstruct.reconstruct_v0(
        root, model_id=MODEL_ID, revision=MODEL_REVISION,
        cache_path=OUT / "reconstruction_cache.sqlite3",
        output_path=OUT / "reconstructed_prompts.parquet",
        batch_size=batch_size, max_new_tokens=max_new_tokens, limit=limit,
        retry_truncated=retry_truncated, seed=seed,
        prompt_version=PROMPT_VERSION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-truncated", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    result = reconstruct_v0(args.root, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, limit=args.limit,
        retry_truncated=args.retry_truncated, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0 if result["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
