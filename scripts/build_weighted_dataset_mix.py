#!/usr/bin/env python3
"""Build a reproducible weighted mix of datasets registered in configs/datasets.yaml."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data_factory import build_weighted_dataset_mix


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    catalog = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))
    registered = {entry["id"] for entry in catalog.get("datasets", [])}
    missing = {item["dataset_id"] for item in config["components"]} - registered
    if missing:
        raise SystemExit(f"unregistered input datasets: {sorted(missing)}")
    profile = build_weighted_dataset_mix(config["components"], dataset_id=config["dataset_id"],
        data_dir=ROOT / args.data_dir,
        preferred_component=config.get("preferred_component_for_duplicates", "v0"),
        configuration_path=ROOT / args.config)
    print(yaml.safe_dump(profile, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
