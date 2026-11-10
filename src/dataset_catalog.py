"""Load, validate, and render the canonical processed-dataset catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "configs/datasets.yaml"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs/datasets.md"
REQUIRED_ENTRY_FIELDS = {
    "id", "name", "section", "description", "artifact", "profile", "validation",
}


def _validate_relative_path(value: Any, field: str, entry_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset {entry_id!r}: {field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"dataset {entry_id!r}: {field} must stay within the project root")
    return value


def validate_catalog(catalog: Any) -> list[dict[str, Any]]:
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("dataset catalog must be a mapping with schema_version: 1")
    entries = catalog.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("dataset catalog must contain a datasets list")

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{index}] must be a mapping")
        missing = REQUIRED_ENTRY_FIELDS - item.keys()
        if missing:
            raise ValueError(f"datasets[{index}] is missing fields: {', '.join(sorted(missing))}")
        entry = dict(item)
        for field in ("id", "name", "section", "description", "validation"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"dataset entry {index}: {field} must be a non-empty string")
            entry[field] = entry[field].strip()
        if entry["id"] in seen_ids:
            raise ValueError(f"duplicate dataset id: {entry['id']}")
        if entry["name"] in seen_names:
            raise ValueError(f"duplicate dataset display name: {entry['name']}")
        seen_ids.add(entry["id"])
        seen_names.add(entry["name"])
        entry["artifact"] = _validate_relative_path(entry["artifact"], "artifact", entry["id"])
        entry["profile"] = _validate_relative_path(entry["profile"], "profile", entry["id"])
        reports = entry.get("reports", [])
        if not isinstance(reports, list):
            raise ValueError(f"dataset {entry['id']!r}: reports must be a list")
        entry["reports"] = [
            _validate_relative_path(path, "reports[]", entry["id"]) for path in reports
        ]
        normalized.append(entry)
    return normalized


def load_dataset_catalog(path: str | Path = DEFAULT_CATALOG) -> dict[str, Any]:
    catalog_path = Path(path)
    try:
        with catalog_path.open("r", encoding="utf-8") as stream:
            catalog = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {catalog_path}: {exc}") from exc
    entries = validate_catalog(catalog)
    catalog["datasets"] = entries
    return catalog


def _profile_rows(root: Path, profile_path: str) -> str:
    path = root / profile_path
    if not path.is_file():
        return "—"
    with path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    rows = profile.get("rows")
    return f"{int(rows):,}" if isinstance(rows, (int, float)) else "—"


def render_markdown(catalog: dict[str, Any], root: str | Path = PROJECT_ROOT) -> str:
    root = Path(root)
    entries = validate_catalog(catalog)
    lines = [
        "# Dataset Catalog",
        "",
        "This page is generated from [`configs/datasets.yaml`](../configs/datasets.yaml). "
        "Add or update dataset entries in that file, then run `python -m src.dataset_catalog --render-docs`.",
    ]
    if catalog.get("description"):
        lines += ["", str(catalog["description"]).strip()]

    sections: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        sections.setdefault(entry["section"], []).append(entry)
    for section, section_entries in sections.items():
        lines += ["", f"## {section}", "", "| Name | Artifact | Rows | Description | Validation | Profile | Related reports |", "|---|---|---:|---|---|---|---|"]
        for entry in section_entries:
            artifact_label = Path(entry["artifact"]).name
            artifact_link = f"[`{artifact_label}`](../{entry['artifact']})"
            profile_label = Path(entry["profile"]).name
            profile_link = f"[`{profile_label}`](../{entry['profile']})"
            desc = entry["description"].replace("|", "\\|").replace("\n", " ")
            validation = entry["validation"].replace("|", "\\|").replace("\n", " ")
            reports = " · ".join(f"[{Path(report).name}](../{report})" for report in entry["reports"])
            lines.append(
                f"| **{entry['name']}** | {artifact_link} | {_profile_rows(root, entry['profile'])} "
                f"| {desc} | {validation} | {profile_link} | {reports or '—'} |"
            )

    notes = catalog.get("notes", [])
    if notes:
        lines += ["", "## Notes", ""]
        lines.extend(f"- {str(note).strip()}" for note in notes)
    resources = catalog.get("resources", [])
    if resources:
        lines += ["", "## Resources", ""]
        lines.extend(
            f"- [{item['label']}](../{item['path']}): {item.get('description', '')}"
            for item in resources
        )
    lines.append("")
    return "\n".join(lines)


def render_docs(catalog_path: str | Path = DEFAULT_CATALOG,
                output_path: str | Path = DEFAULT_MARKDOWN) -> Path:
    catalog_path = Path(catalog_path)
    root = catalog_path.resolve().parents[1]
    catalog = load_dataset_catalog(catalog_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(catalog, root), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--render-docs", action="store_true", help="render docs/datasets.md from the catalog")
    actions.add_argument("--check", action="store_true", help="validate the catalog without writing files")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    catalog = load_dataset_catalog(args.catalog)
    if args.render_docs:
        output = render_docs(args.catalog, args.output)
        print(f"Rendered {output} from {len(catalog['datasets'])} dataset entries.")
    else:
        print(f"Valid catalog: {len(catalog['datasets'])} dataset entries.")


if __name__ == "__main__":
    main()
