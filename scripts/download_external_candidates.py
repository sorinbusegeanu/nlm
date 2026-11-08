#!/usr/bin/env python3
"""Download and inventory the requested external Name That LLM datasets.

All repository files are preserved verbatim under data/raw/external_candidates.
Hugging Face's local_dir metadata makes repeated runs resumable and skips files
whose revision has already been downloaded successfully.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hf_auth import hf_token  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/raw/external_candidates"
SOURCES = {
    "claude_opus_4_8": {
        "repo_id": "Met4physics/claude-opus-4-8-xhigh-reasoning-8.7k",
        "models": ["claude-opus-4-8"],
        "primary_files": ["full_train.jsonl"],
        # Last publicly indexed card details, retained only when the live API
        # refuses metadata access. These are not presented as local measurements.
        "catalog_metadata": {
            "license": "apache-2.0 (indexed Hugging Face card)",
            "viewer_rows": 38504,
            "card_described_examples": 8706,
            "reported_size": "344 MB",
            "columns": ["category", "messages", "model"],
            "note": "Indexed Hub metadata only; current API/file access returned 404, so these values are unverified locally.",
        },
    },
    "glm53_flash": {
        "repo_id": "Zek-Takai/glm53-flash-harvest",
        "models": ["glm-5.3-flash"],
        "primary_files": ["data/glm53_harvest.parquet", "data/glm53_harvest_r2.parquet",
                          "data/glm53_harvest_r3.parquet", "data/glm53_harvest_r4.parquet"],
    },
    "gpt56_sol_luna": {
        "repo_id": "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces",
        "models": ["gpt-5.6-sol", "gpt-5.6-luna"],
        "primary_files": ["data/train.parquet"],
    },
}
URL = lambda repo: f"https://huggingface.co/datasets/{repo}"


def _read_card_data(info):
    card = getattr(info, "card_data", None)
    if card is None:
        return {}
    if hasattr(card, "to_dict"):
        return card.to_dict()
    if isinstance(card, dict):
        return card
    return {}


def _normalise_model(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.removeprefix("openai/").removeprefix("anthropic/")
    aliases = {
        "claude-opus-4.8": "claude-opus-4-8",
        "gpt-5.6-sol": "gpt-5.6-sol",
        "gpt-5.6-luna": "gpt-5.6-luna",
        "glm-5.3-flash": "glm-5.3-flash",
        "glm53-flash": "glm-5.3-flash",
    }
    for key, model in aliases.items():
        if key in text:
            return model
    return text if re.fullmatch(r"[a-z0-9][a-z0-9._/-]*", text) else None


def _model_from_row(row, targets):
    for key in ("model", "teacher_model", "model_name", "target_model", "first_source_dataset"):
        value = row.get(key)
        model = _normalise_model(value)
        if model:
            match = next((target for target in targets if target in model), None)
            if match:
                return match
    encoded = row.get("row_json")
    if encoded:
        try:
            value = json.loads(encoded) if isinstance(encoded, str) else encoded
        except (TypeError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict):
            for key in ("teacher_model", "model", "model_name", "target_model"):
                model = _normalise_model(value.get(key))
                match = next((target for target in targets if model and target in model), None)
                if match:
                    return match
    return None


def _count_table(path, targets):
    """Return exact physical row count, columns, and any row-level model counts."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        columns = list(parquet.schema_arrow.names)
        counts = Counter()
        for batch in parquet.iter_batches(batch_size=4096):
            data = batch.to_pydict()
            for index in range(batch.num_rows):
                row = {key: values[index] for key, values in data.items()}
                model = _model_from_row(row, targets)
                if model:
                    counts[model] += 1
        return parquet.metadata.num_rows, columns, counts
    if suffix in {".jsonl", ".ndjson"}:
        count = 0
        columns = set()
        model_counts = Counter()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                if isinstance(row, dict):
                    columns.update(row)
                    model = _model_from_row(row, targets)
                    if model:
                        model_counts[model] += 1
        return count, sorted(columns), model_counts
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("data", []) if isinstance(value, dict) else []
        if not isinstance(rows, list):
            rows = []
        columns = sorted({key for row in rows if isinstance(row, dict) for key in row})
        counts = Counter()
        for row in rows:
            if isinstance(row, dict):
                model = _model_from_row(row, targets)
                if model:
                    counts[model] += 1
        return len(rows), columns, counts
    if suffix == ".csv":
        import pyarrow.csv as pacsv
        table = pacsv.read_csv(path)
        data = table.to_pydict()
        counts = Counter()
        for index in range(table.num_rows):
            row = {key: values[index] for key, values in data.items()}
            model = _model_from_row(row, targets)
            if model:
                counts[model] += 1
        return table.num_rows, table.schema.names, counts
    return None, [], Counter()


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def _inspect_source(key, config, info, destination):
    card = _read_card_data(info)
    targets = config["models"]
    inventory = []
    for sibling in info.siblings or []:
        relative = sibling.rfilename
        local = destination / relative
        if not local.is_file():
            continue
        suffix = local.suffix.lower()
        if suffix not in {".parquet", ".jsonl", ".ndjson", ".json", ".csv"}:
            rows, columns, model_counts = None, [], Counter()
        else:
            rows, columns, model_counts = _count_table(local, targets)
        inventory.append({
            "path": relative, "rows": rows, "columns": columns,
            "model_rows": dict(model_counts), "size_bytes": local.stat().st_size,
            "primary": relative in config["primary_files"],
        })
    primary = [item for item in inventory if item["primary"]]
    logical_rows = sum(int(item["rows"] or 0) for item in primary)
    per_model = Counter()
    has_row_model_field = False
    for item in primary:
        per_model.update(item["model_rows"])
        has_row_model_field |= bool(item["model_rows"])
    if not has_row_model_field and logical_rows:
        per_model.update({model: logical_rows for model in targets})
    total_bytes = sum(item["size_bytes"] for item in inventory)
    return {
        "key": key, "repo_id": config["repo_id"], "url": URL(config["repo_id"]),
        "target_models": targets, "revision": info.sha,
        "license": card.get("license") or "Not specified in Hub card metadata",
        "language": card.get("language", []), "tags": card.get("tags", []),
        "logical_rows": logical_rows, "rows_per_model": dict(per_model),
        "model_counts_are_row_level": has_row_model_field,
        "primary_files": config["primary_files"], "total_downloaded_bytes": total_bytes,
        "files": inventory,
    }


def _save_inventory(output_root, records):
    lines = [
        "# External candidate datasets", "",
        "Raw snapshots are kept in separate directories without schema conversion or NLM filtering. "
        "The downloader pins each snapshot to the Hub commit it inspected and can resume an interrupted transfer.",
        "", "Run or resume all sources with:", "", "```bash",
        "/home/zodrak/cuda13.x/bin/python scripts/download_external_candidates.py", "```", "",
        "Physical table/file row counts are listed per downloaded file. The source-level total uses only the "
        "listed primary files, avoiding sums across overlapping task-specific/config exports.", "",
    ]
    for key, config in SOURCES.items():
        record = records.get(key)
        lines.extend([f"## {config['repo_id']}", "", f"- URL: {URL(config['repo_id'])}",
                      f"- Target model(s): {', '.join(config['models'])}"])
        if not record:
            lines.extend(["- Status: not downloaded/inspected yet.", ""])
            continue
        if record.get("status") != "downloaded":
            lines.extend([f"- Status: **{record.get('status', 'failed')}** — {record.get('error', 'unknown error')}",
                          f"- Target directory: `{output_root / key}`", ""])
            indexed = record.get("catalog_metadata")
            if indexed:
                lines.extend([
                    f"- Indexed license: `{indexed['license']}`; reported size: {indexed['reported_size']}; "
                    f"viewer rows: {indexed['viewer_rows']:,}; card-described examples: {indexed['card_described_examples']:,}.",
                    f"- Indexed columns: `{', '.join(indexed['columns'])}`.",
                    f"- Note: {indexed['note']}", "",
                ])
            continue
        lines.extend([
            f"- Status: downloaded at revision `{record['revision']}`.",
            *( [f"- Last repeat-download attempt: {record['last_attempt_error']}. Available local files are retained."]
               if record.get("last_attempt_error") else [] ),
            f"- Target directory: `{output_root / key}`",
            f"- License: `{record['license']}`",
            f"- Language metadata: {', '.join(record['language']) if record['language'] else 'N/A'}",
            f"- Primary rows: **{record['logical_rows']:,}**",
            f"- Rows per target model: " + ", ".join(f"`{model}` {count:,}" for model, count in record["rows_per_model"].items()),
            ("- Model attribution: counted from row-level provenance fields."
             if record["model_counts_are_row_level"] else
             "- Model attribution: the source table has no row-level model column; rows are attributed from the repository's target model."),
            f"- Downloaded repository files: {len(record['files'])}; total bytes: {record['total_downloaded_bytes']:,} "
            f"({record['total_downloaded_bytes'] / (1024**2):.2f} MiB).",
            "", "### Files", "",
            _markdown_table(["File", "Rows", "Columns", "Size (bytes)", "Role"], [
                (item["path"], f"{item['rows']:,}" if item["rows"] is not None else "—",
                 ", ".join(item["columns"]) or "—", f"{item['size_bytes']:,}",
                 "primary" if item["primary"] else "additional/metadata")
                for item in record["files"]
            ]), "",
        ])
    lines.extend(["## Provenance notes", "",
                  "The raw files retain their original model/provenance fields. Where a source does not carry a "
                  "row-level model column, the target-model count is attributed from the dataset repository itself; "
                  "this is marked in each source's `download_manifest.json`.", ""])
    target = output_root / "README.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def download_sources(output_root=DEFAULT_OUTPUT, *, selected=None, token=None):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = {}
    # Keep prior inventory entries for sources excluded by --repo.
    for key in SOURCES:
        manifest = output_root / key / "download_manifest.json"
        if manifest.is_file():
            try:
                records[key] = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    for key, config in SOURCES.items():
        if selected and key not in selected:
            continue
        destination = output_root / key
        destination.mkdir(parents=True, exist_ok=True)
        try:
            api = HfApi(token=token)
            info = api.dataset_info(config["repo_id"], files_metadata=True)
            # Pin this local snapshot to the inspected revision; local_dir uses
            # its metadata cache to resume/skip already completed files.
            snapshot_download(repo_id=config["repo_id"], repo_type="dataset",
                              revision=info.sha, local_dir=str(destination), token=token)
            record = _inspect_source(key, config, info, destination)
            record["status"] = "downloaded"
            record["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
            record["download_root"] = str(destination.resolve())
            record["model_counts_are_row_level"] = record["model_counts_are_row_level"]
            manifest_path = destination / "download_manifest.json"
            manifest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records[key] = record
            print(f"Downloaded {config['repo_id']} ({record['logical_rows']:,} primary rows; "
                  f"{record['total_downloaded_bytes'] / (1024**2):.2f} MiB)", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            previous = records.get(key, {})
            if previous.get("status") == "downloaded":
                record = {**previous, "last_attempt_error": error}
            else:
                record = {"status": "failed", "error": error,
                          "repo_id": config["repo_id"], "url": URL(config["repo_id"]),
                          "target_models": config["models"],
                          "catalog_metadata": config.get("catalog_metadata")}
            (destination / "download_manifest.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records[key] = record
            print(f"Could not download {config['repo_id']}: "
                  f"{record.get('last_attempt_error', record.get('error'))}",
                  file=sys.stderr, flush=True)
        _save_inventory(output_root, records)
    _save_inventory(output_root, records)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", choices=sorted(SOURCES), action="append",
                        help="limit to one or more source keys; default downloads all")
    args = parser.parse_args(argv)
    records = download_sources(args.output, selected=set(args.repo) if args.repo else None,
                               token=hf_token())
    failures = [key for key, record in records.items() if record.get("status") != "downloaded"]
    print(json.dumps({key: {"status": value.get("status"),
                            "primary_rows": value.get("logical_rows"),
                            "downloaded_bytes": value.get("total_downloaded_bytes"),
                            "target_directory": str(args.output / key),
                            "error": value.get("error")}
                      for key, value in records.items()}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
