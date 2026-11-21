#!/usr/bin/env python3
"""Build V2.GR.OR16 from the original Open Pool v16 main parquet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.v2_gr_or16 import build_dataset, validate_artifact, write_report
from src.data_factory import _hash_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/open_pool/v16/ntl_open_pool.parquet")
    parser.add_argument("--v0", type=Path, default=ROOT / "data/processed/dataset_v0.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/dataset_v2.GR.OR16.parquet")
    parser.add_argument("--profile", type=Path, default=ROOT / "data/processed/dataset_profile_v2.GR.OR16.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/v2.GR.OR16.md")
    args = parser.parse_args()
    for path in (args.source, args.v0):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = pd.read_parquet(args.source).reset_index(drop=True)
    v0 = pd.read_parquet(args.v0)
    data, profile = build_dataset(source, v0=v0, source_sha256=_hash_file(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False, compression="zstd")
    profile["dataset_sha256"] = _hash_file(args.output)
    validate_artifact(pd.read_parquet(args.output), profile, v0=v0)
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(profile, args.report)
    print(json.dumps({"rows": profile["rows"], "ranking": profile["ranking"],
                      "reconciliation": profile["reconciliation_matches_audit"],
                      "artifact": str(args.output), "sha256": profile["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
