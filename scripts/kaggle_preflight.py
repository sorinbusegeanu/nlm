"""Preflight Kaggle notebook packages and completed submission artifacts.

This module is intentionally small and dependency-light. It checks local
notebook/package integrity before upload, can verify attached source inventories
through the Kaggle CLI, and injects a runtime gate that runs before inference.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import symtable
import tempfile
from pathlib import Path
from typing import Any

import nbformat


class PreflightError(RuntimeError):
    pass


def to_json_safe(value: Any) -> Any:
    """Convert common scientific Python values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite floats are not valid deployment metadata")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    # NumPy scalars and arrays expose item()/tolist() without requiring NumPy
    # to be imported by the packaging utility.
    if hasattr(value, "tolist"):
        return to_json_safe(value.tolist())
    if hasattr(value, "item"):
        return to_json_safe(value.item())
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(to_json_safe(value), indent=2, allow_nan=False) + "\n")


def _code_cells(notebook: nbformat.NotebookNode) -> list[str]:
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def _strip_magics(source: str) -> tuple[str, list[str]]:
    """Return compilable source and notebook-only lines that were skipped."""
    output, skipped = [], []
    for line in source.splitlines():
        if line.lstrip().startswith(("%", "!")):
            skipped.append(line.strip())
            output.append("pass  # Kaggle preflight: skipped IPython magic")
        else:
            output.append(line)
    return "\n".join(output), skipped


def notebook_code_issues(notebook: nbformat.NotebookNode) -> list[str]:
    issues: list[str] = []
    cells = _code_cells(notebook)
    cleaned: list[str] = []
    skipped_magics: list[str] = []
    for i, source in enumerate(cells):
        clean, skipped = _strip_magics(source)
        skipped_magics.extend(f"cell {i}: {line}" for line in skipped)
        try:
            compile(clean, f"<notebook-cell-{i}>", "exec")
        except (SyntaxError, ValueError) as exc:
            issues.append(f"code cell {i} does not compile: {exc}")
        cleaned.append(clean)
    if skipped_magics:
        issues.extend(f"unverified IPython magic: {line}" for line in skipped_magics)
    if not cleaned:
        issues.append("notebook has no code cells")
        return issues

    # Analyze the notebook as one module so variables defined in an earlier
    # cell are visible to later cells. This catches misspelled bootstrap names
    # such as RUNTIME before Kaggle spends time loading models.
    source = "\n\n".join(cleaned)
    try:
        table = symtable.symtable(source, "<kaggle-notebook>", "exec")
    except SyntaxError as exc:
        issues.append(f"notebook global analysis failed: {exc}")
        return issues
    module_names = {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace()
    }
    global_assignments: set[str] = set()
    refs: set[tuple[str, str]] = set()

    def visit(scope: symtable.SymbolTable) -> None:
        for sym in scope.get_symbols():
            if sym.is_global() and sym.is_assigned():
                global_assignments.add(sym.get_name())
            if sym.is_global() and sym.is_referenced():
                refs.add((scope.get_name(), sym.get_name()))
        for child in scope.get_children():
            visit(child)

    visit(table)
    injected = {"display", "get_ipython", "In", "Out", "_ih", "_oh", "__file__", "__name__"}
    defined = module_names | global_assignments | set(dir(builtins)) | injected
    for scope, name in sorted(refs):
        if name not in defined:
            issues.append(f"possibly undefined global {name!r} referenced in {scope!r}")
    return issues


def validate_kernel_package(kernel_dir: str | Path) -> dict[str, Any]:
    root = Path(kernel_dir)
    metadata_path = root / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise PreflightError(f"missing {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except Exception as exc:
        raise PreflightError(f"invalid kernel metadata JSON: {exc}") from exc
    required = {"id", "code_file", "kernel_type", "language", "enable_gpu", "dataset_sources", "model_sources"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise PreflightError(f"kernel metadata missing fields: {missing}")
    if metadata["kernel_type"] != "notebook":
        raise PreflightError("only notebook kernels are supported by this preflight")
    code_path = root / metadata["code_file"]
    if not code_path.is_file():
        raise PreflightError(f"metadata code_file does not exist: {metadata['code_file']}")
    try:
        notebook = nbformat.read(code_path, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:
        raise PreflightError(f"invalid notebook: {exc}") from exc

    problems = notebook_code_issues(notebook)
    if problems:
        raise PreflightError("notebook code checks failed:\n- " + "\n- ".join(problems))

    datasets = metadata.get("dataset_sources", [])
    models = metadata.get("model_sources", [])
    if not isinstance(datasets, list) or not isinstance(models, list):
        raise PreflightError("dataset_sources and model_sources must be lists")
    for ref in datasets:
        if not isinstance(ref, str) or len(ref.strip("/").split("/")) != 2:
            raise PreflightError(f"invalid Kaggle dataset reference: {ref!r}")
    for ref in models:
        parts = ref.strip("/").split("/") if isinstance(ref, str) else []
        if len(parts) != 5 or not parts[-1].isdigit():
            raise PreflightError(f"model source must be owner/model/framework/instance/version: {ref!r}")

    json_files = []
    for path in root.rglob("*.json"):
        try:
            json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except Exception as exc:
            raise PreflightError(f"invalid strict JSON file {path.relative_to(root)}: {exc}") from exc
        json_files.append(path.relative_to(root).as_posix())
    return {
        "kernel_ref": metadata["id"],
        "code_file": metadata["code_file"],
        "code_cells": len(_code_cells(notebook)),
        "dataset_sources": len(datasets),
        "model_sources": len(models),
        "json_files_checked": len(json_files),
        "status": "PASS",
    }


def _run_json_command(command: list[str]) -> Any:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise PreflightError(
            f"Kaggle inventory command failed ({' '.join(command)}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"Kaggle inventory was not JSON for {' '.join(command)}") from exc


def _collect_dataset_assets(
    kernel_dir: Path,
    dataset_sources: set[str],
    manifest_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> list[dict[str, Any]]:
    assets: dict[tuple[str, str], dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("dataset_ref")
            filename = value.get("flat_filename") or value.get("kaggle_filename")
            if isinstance(ref, str) and ref in dataset_sources and isinstance(filename, str):
                item = {"dataset_ref": ref, "filename": filename, "size": value.get("size")}
                assets[(ref, Path(filename).name)] = item
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    candidates = set(kernel_dir.rglob("*.json"))
    candidates.update(Path(p) for p in manifest_paths)
    for path in sorted(candidates):
        try:
            visit(json.loads(path.read_text()))
        except Exception:
            continue
    return sorted(assets.values(), key=lambda x: (x["dataset_ref"], x["filename"]))


def validate_remote_sources(kernel_dir: str | Path) -> dict[str, Any]:
    root = Path(kernel_dir)
    metadata = json.loads((root / "kernel-metadata.json").read_text())
    preflight_path = root / "kaggle-preflight.json"
    extra = json.loads(preflight_path.read_text()) if preflight_path.is_file() else {}
    processor_sources = set(extra.get("processor_sources", []))
    datasets, models = {}, {}
    remote_dataset_files: dict[str, dict[str, int | None]] = {}
    for ref in metadata.get("dataset_sources", []):
        records = _run_json_command(["kaggle", "datasets", "files", ref, "--format", "json", "--page-size", "200"])
        if not isinstance(records, list) or not records:
            raise PreflightError(f"Kaggle dataset source has no visible files: {ref}")
        remote_dataset_files[ref] = {
            str(entry.get("name")): entry.get("size")
            for entry in records if isinstance(entry, dict) and entry.get("name")
        }
        datasets[ref] = {"file_count": len(remote_dataset_files[ref]),
                         "sample_files": sorted(remote_dataset_files[ref])[:3]}
    for ref in metadata.get("model_sources", []):
        records = _run_json_command(["kaggle", "models", "instances", "versions", "files", ref,
                                     "--format", "json", "--page-size", "200"])
        names = {entry.get("name") for entry in records if isinstance(entry, dict)}
        if "config.json" not in names:
            raise PreflightError(f"Kaggle model source lacks root config.json: {ref}")
        if not names.intersection({"tokenizer.json", "tokenizer.model", "vocab.txt"}):
            raise PreflightError(f"Kaggle model source lacks a recognizable tokenizer: {ref}")
        if ref in processor_sources and not names.intersection({"preprocessor_config.json", "processor_config.json"}):
            raise PreflightError(f"Kaggle model source is marked as processor-backed but lacks processor config: {ref}")
        models[ref] = {"file_count": len(names),
                       "has_config": "config.json" in names,
                       "has_tokenizer": bool(names.intersection({"tokenizer.json", "tokenizer.model", "vocab.txt"})),
                       "has_processor_config": bool(names.intersection({"preprocessor_config.json", "processor_config.json"}))}
    expected_assets = _collect_dataset_assets(root, set(metadata.get("dataset_sources", [])))
    expected_assets.extend(extra.get("required_dataset_files", []))
    # Deduplicate exact source/basename pairs while retaining an expected size.
    unique_assets = {}
    for item in expected_assets:
        key = (item["dataset_ref"], Path(item["filename"]).name)
        previous = unique_assets.get(key)
        if previous and previous.get("size") not in (None, item.get("size")):
            raise PreflightError(f"conflicting expected sizes for {key}: {previous.get('size')} vs {item.get('size')}")
        unique_assets[key] = item if not previous or previous.get("size") is None else previous
    expected_assets = list(unique_assets.values())
    for asset in expected_assets:
        ref, filename = asset["dataset_ref"], Path(asset["filename"]).name
        matches = [size for name, size in remote_dataset_files.get(ref, {}).items()
                   if Path(name).name == filename]
        if not matches:
            raise PreflightError(f"required asset {filename} is missing from Kaggle dataset {ref}")
        if asset.get("size") is not None and int(asset["size"]) not in {int(x) for x in matches if x is not None}:
            raise PreflightError(
                f"asset size mismatch for {ref}/{filename}: expected {asset['size']}, found {matches}"
            )
    return {"datasets": datasets, "models": models,
            "required_manifest_assets_checked": len(expected_assets), "status": "PASS"}


def runtime_preflight_code(spec: dict[str, Any]) -> str:
    literal = json.dumps(to_json_safe(spec), ensure_ascii=True, separators=(",", ":"))
    return f'''# Generated by scripts/kaggle_preflight.py. Runs before any model inference.
import importlib.metadata as _kpm, json as _kpjson, os as _kpos, sys as _kpsys
from pathlib import Path as _KP
_KAGGLE_PREFLIGHT_SPEC = _kpjson.loads({literal!r})
def _kp_to_json_safe(_obj):
    if _obj is None or isinstance(_obj,(str,bool,int)): return _obj
    if isinstance(_obj,float):
        if _obj != _obj or _obj in (float("inf"),float("-inf")): raise ValueError("non-finite float in deployment JSON")
        return _obj
    if isinstance(_obj,_KP): return str(_obj)
    if isinstance(_obj,dict): return {{str(_k):_kp_to_json_safe(_v) for _k,_v in _obj.items()}}
    if isinstance(_obj,(list,tuple,set)): return [_kp_to_json_safe(_v) for _v in _obj]
    if hasattr(_obj,"tolist"): return _kp_to_json_safe(_obj.tolist())
    if hasattr(_obj,"item"): return _kp_to_json_safe(_obj.item())
    raise TypeError(f"Object of type {{type(_obj).__name__}} is not JSON serializable")
def kaggle_json_dumps(_obj,*_args,**_kwargs): return _kpjson.dumps(_kp_to_json_safe(_obj),*_args,**_kwargs)
def kaggle_json_dump(_obj,_file,*_args,**_kwargs): return _kpjson.dump(_kp_to_json_safe(_obj),_file,*_args,**_kwargs)
_kp_errors=[]; _kp_report={{"python":_kpsys.version,"packages":{{}},"dataset_mounts":{{}},"model_mounts":{{}}}}
_kp_input=_KP("/kaggle/input")
if not _kp_input.is_dir(): _kp_errors.append("/kaggle/input is missing")
for _pkg,_expected in _KAGGLE_PREFLIGHT_SPEC.get("strict_versions",{{}}).items():
    try: _actual=_kpm.version(_pkg)
    except _kpm.PackageNotFoundError: _actual=None
    _kp_report["packages"][_pkg]={{"expected":_expected,"actual":_actual}}
    if _actual != _expected: _kp_errors.append(f"package version mismatch for {{_pkg}}: expected {{_expected}}, found {{_actual}}")
for _dist in _KAGGLE_PREFLIGHT_SPEC.get("required_distributions",[]):
    try: _kp_report["packages"][_dist]=_kpm.version(_dist)
    except _kpm.PackageNotFoundError: _kp_errors.append(f"required package missing: {{_dist}}")
for _ref in _KAGGLE_PREFLIGHT_SPEC.get("dataset_sources",[]):
    _owner,_slug=_ref.split("/",1)
    _candidates=[_kp_input/"datasets"/_owner/_slug,_kp_input/_owner/_slug,_kp_input/_slug]
    _found=next((p for p in _candidates if p.exists()),None)
    if _found is None and _kp_input.is_dir():
        for _meta in _kp_input.rglob("dataset-metadata.json"):
            try:
                if _kpjson.loads(_meta.read_text()).get("id")==_ref: _found=_meta.parent; break
            except Exception: pass
    if _found is None: _kp_errors.append(f"dataset source is not mounted: {{_ref}}")
    else: _kp_report["dataset_mounts"][_ref]=str(_found)
_kp_asset_index={{}}
for _ref,_root in _kp_report["dataset_mounts"].items():
    _files={{}}
    for _dir,_subdirs,_names in _kpos.walk(_root):
        for _name in _names:
            _p=_KP(_dir)/_name
            try: _files.setdefault(_name,[]).append(_p.stat().st_size)
            except OSError: pass
    _kp_asset_index[_ref]=_files
for _asset in _KAGGLE_PREFLIGHT_SPEC.get("required_dataset_files",[]):
    _ref=_asset["dataset_ref"]; _name=_KP(_asset["filename"]).name
    _sizes=_kp_asset_index.get(_ref,{{}}).get(_name,[])
    if not _sizes: _kp_errors.append(f"required checkpoint/input asset is missing: {{_ref}}/{{_name}}")
    elif _asset.get("size") is not None and int(_asset["size"]) not in _sizes:
        _kp_errors.append(f"asset size mismatch: {{_ref}}/{{_name}} expected {{_asset['size']}}, found {{_sizes}}")
for _ref in _KAGGLE_PREFLIGHT_SPEC.get("model_sources",[]):
    _parts=_ref.split("/"); _owner,_model,_framework,_instance,_version=_parts
    _path=_kp_input/"models"/_owner/_model/_framework.lower()/_instance/_version
    if not _path.is_dir():
        _kp_errors.append(f"model source is not mounted at expected path: {{_ref}} -> {{_path}}")
        continue
    _files={{p.name for p in _path.iterdir() if p.is_file()}}
    if "config.json" not in _files: _kp_errors.append(f"model source lacks config.json: {{_path}}")
    if not _files.intersection({{"tokenizer.json","tokenizer.model","vocab.txt"}}): _kp_errors.append(f"model source lacks tokenizer files: {{_path}}")
    _kp_report["model_mounts"][_ref]=str(_path)
if not _kp_errors and _KAGGLE_PREFLIGHT_SPEC.get("model_sources"):
    try:
        from transformers import AutoConfig as _KPAutoConfig, AutoTokenizer as _KPAutoTokenizer, AutoProcessor as _KPAutoProcessor
        for _ref in _KAGGLE_PREFLIGHT_SPEC["model_sources"]:
            _parts=_ref.split("/"); _path=_kp_input/"models"/_parts[0]/_parts[1]/_parts[2].lower()/_parts[3]/_parts[4]
            _cfg=_KPAutoConfig.from_pretrained(str(_path),local_files_only=True,trust_remote_code=True)
            _KPAutoTokenizer.from_pretrained(str(_path),local_files_only=True,trust_remote_code=True)
            _kp_report.setdefault("model_configs",{{}})[_ref]={{"model_type":getattr(_cfg,"model_type",None),"tokenizer":"ok"}}
            if _ref in _KAGGLE_PREFLIGHT_SPEC.get("processor_sources",[]):
                _KPAutoProcessor.from_pretrained(str(_path),local_files_only=True,trust_remote_code=True)
                _kp_report["model_configs"][_ref]["processor"]="ok"
    except Exception as _exc: _kp_errors.append(f"model config/tokenizer/processor compatibility check failed: {{type(_exc).__name__}}: {{_exc}}")
_kp_report["status"]="FAIL" if _kp_errors else "PASS"; _kp_report["errors"]=_kp_errors
_kp_out=_KP("/kaggle/working/kaggle_preflight.json")
_kp_out.parent.mkdir(parents=True,exist_ok=True); _kp_out.write_text(_kpjson.dumps(_kp_report,indent=2)+"\\n")
print("KAGGLE_PREFLIGHT",_kpjson.dumps(_kp_report,sort_keys=True))
if _kp_errors: raise RuntimeError("Kaggle preflight failed before inference: "+"; ".join(_kp_errors))
'''


def embed_runtime_preflight(
    kernel_dir: str | Path,
    *,
    strict_versions: dict[str, str] | None = None,
    required_distributions: list[str] | None = None,
    processor_sources: list[str] | None = None,
    manifest_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> int:
    root = Path(kernel_dir)
    metadata_path = root / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    code_path = root / metadata["code_file"]
    notebook = nbformat.read(code_path, as_version=4)
    notebook.cells = [c for c in notebook.cells if "kaggle-preflight" not in c.metadata.get("tags", [])]
    insert_at = 1 if notebook.cells and notebook.cells[0].cell_type == "markdown" else 0
    spec = {
        "dataset_sources": metadata.get("dataset_sources", []),
        "model_sources": metadata.get("model_sources", []),
        "strict_versions": strict_versions or {},
        "required_distributions": required_distributions or [],
        "processor_sources": processor_sources or [],
        "required_dataset_files": _collect_dataset_assets(
            root, set(metadata.get("dataset_sources", [])), manifest_paths=manifest_paths),
    }
    write_json(root / "kaggle-preflight.json", spec)
    cell = nbformat.v4.new_code_cell(runtime_preflight_code(spec))
    cell.metadata["tags"] = ["kaggle-preflight"]
    notebook.cells.insert(insert_at, cell)
    # Route notebook JSON writes through a converter that safely handles
    # NumPy scalars/arrays and Paths. This prevents a common late-run failure
    # while leaving normal JSON values unchanged.
    rewritten = 0
    json_aliases = {"json"}
    for code_cell in notebook.cells:
        if code_cell.cell_type != "code" or "kaggle-preflight" in code_cell.metadata.get("tags", []):
            continue
        try:
            tree = ast.parse(code_cell.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                json_aliases.update(alias.asname or "json" for alias in node.names if alias.name == "json")
    for code_cell in notebook.cells:
        if code_cell.cell_type != "code" or "kaggle-preflight" in code_cell.metadata.get("tags", []):
            continue
        source = code_cell.source
        for alias in sorted(json_aliases, key=len, reverse=True):
            source, n_dumps = re.subn(rf"\b{re.escape(alias)}\.dumps\s*\(", "kaggle_json_dumps(", source)
            source, n_dump = re.subn(rf"\b{re.escape(alias)}\.dump\s*\(", "kaggle_json_dump(", source)
            rewritten += n_dumps + n_dump
        code_cell.source = source
    nbformat.validate(notebook)
    nbformat.write(notebook, code_path)
    return rewritten


def validate_submission_file(
    path: str | Path,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    expected_ids: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    import pandas as pd

    source = Path(path)
    if not source.is_file():
        raise PreflightError(f"submission file does not exist: {source}")
    frame = pd.read_parquet(source) if source.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(source)
    if list(frame.columns) != ["id", "model"]:
        raise PreflightError(f"submission must have exactly id,model columns; got {list(frame.columns)}")
    ids = frame["id"].astype(str).tolist()
    if frame.empty or len(ids) != len(set(ids)):
        raise PreflightError("submission is empty or has duplicate IDs")
    if frame["model"].isna().any():
        raise PreflightError("submission contains missing model labels")
    models = frame["model"].astype(str).tolist()
    if labels is not None and not set(models).issubset(set(labels)):
        raise PreflightError(f"submission contains unknown labels: {sorted(set(models)-set(labels))}")
    if expected_ids is not None and ids != list(map(str, expected_ids)):
        raise PreflightError("submission IDs/order do not match the expected test IDs")
    return ids


def _read_expected_ids(path: Path) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    if "id" not in frame:
        raise PreflightError(f"expected-ID file has no id column: {path}")
    return frame["id"].astype(str).tolist()


def _cmd_submit(args: argparse.Namespace) -> int:
    labels = None
    labels = json.loads(Path(args.labels_json).read_text())
    if isinstance(labels, dict):
        labels = labels.get("classes") or labels.get("labels")
    if not isinstance(labels, list):
        raise PreflightError("labels JSON must be a list or contain classes/labels")
    expected = _read_expected_ids(Path(args.expected_ids))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    versioned_ref = f"{args.kernel.rstrip('/')}/{args.version}"
    fetched_names = [*args.file, "kaggle_preflight.json"]
    file_pattern = "^(?:" + "|".join(re.escape(name) for name in fetched_names) + ")$"
    fetch = subprocess.run(["kaggle", "kernels", "output", versioned_ref,
                            "-p", str(output_dir), "--force", "--file-pattern", file_pattern],
                           text=True, capture_output=True)
    if fetch.returncode:
        raise PreflightError(f"could not fetch completed output for {versioned_ref}: {fetch.stderr.strip()}")
    runtime_report_path = output_dir / "kaggle_preflight.json"
    if not runtime_report_path.is_file():
        raise PreflightError(f"kernel output is missing {runtime_report_path.name}; runtime preflight was not run")
    runtime_report = json.loads(runtime_report_path.read_text())
    if runtime_report.get("status") != "PASS":
        raise PreflightError(f"Kaggle runtime preflight did not pass: {runtime_report.get('errors', [])}")
    ids_by_file = []
    for name in args.file:
        path = output_dir / name
        ids_by_file.append((name, validate_submission_file(path, labels=labels, expected_ids=expected)))
    if ids_by_file and any(ids != ids_by_file[0][1] for _, ids in ids_by_file[1:]):
        raise PreflightError("submission variants do not have identical ordered IDs")
    for name, _ in ids_by_file:
        result = subprocess.run(
            ["kaggle", "competitions", "submit", args.competition, "-k", args.kernel,
             "-v", str(args.version), "-f", name, "-m", args.message.format(file=name)],
            cwd=output_dir,
        )
        if result.returncode:
            raise PreflightError(f"Kaggle submission command failed for {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate local package; optionally query Kaggle inventories")
    check.add_argument("--kernel-dir", required=True)
    check.add_argument("--remote", action="store_true", help="verify attached datasets and model assets exist on Kaggle")
    check.add_argument("--embed-runtime", action="store_true", help="insert the Kaggle startup gate before inference")
    push = sub.add_parser("push", help="preflight and push a notebook kernel")
    push.add_argument("--kernel-dir", required=True)
    push.add_argument("--run", action="store_true", help="execute the pushed kernel; default only saves a version")
    push.add_argument("--skip-remote", action="store_true", help="skip read-only Kaggle input inventory checks")
    push.add_argument("--strict-version", action="append", default=[], metavar="DISTRIBUTION=VERSION",
                      help="require an exact package version at Kaggle runtime; repeatable")
    push.add_argument("--processor-source", action="append", default=[],
                      help="model-source ref that must load through AutoProcessor; repeatable")
    push.add_argument("--manifest", action="append", default=[],
                      help="local deployment manifest outside the kernel directory; repeatable")
    submit = sub.add_parser("submit", help="validate completed kernel outputs and submit them")
    submit.add_argument("--kernel", required=True)
    submit.add_argument("--version", required=True, type=int)
    submit.add_argument("--competition", required=True)
    submit.add_argument("--output-dir", required=True)
    submit.add_argument("--file", action="append", required=True, help="output file name; repeat for variants")
    submit.add_argument("--message", required=True, help="submission message; may include {file}")
    submit.add_argument("--expected-ids", required=True, help="CSV/Parquet with the expected test id order")
    submit.add_argument("--labels-json", required=True, help="JSON label list or object with classes/labels")
    args = parser.parse_args(argv)
    try:
        if args.command == "submit":
            return _cmd_submit(args)
        root = Path(args.kernel_dir)
        if args.command == "push":
            strict_versions = {}
            for item in args.strict_version:
                if "=" not in item:
                    raise PreflightError(f"invalid --strict-version {item!r}; expected DISTRIBUTION=VERSION")
                package, version = item.split("=", 1)
                strict_versions[package] = version
            embed_runtime_preflight(root, strict_versions=strict_versions,
                                    processor_sources=args.processor_source,
                                    manifest_paths=args.manifest)
        local = validate_kernel_package(root)
        report: dict[str, Any] = {"local": local}
        if getattr(args, "remote", False) or (args.command == "push" and not args.skip_remote):
            report["remote"] = validate_remote_sources(root)
        print(json.dumps(report, indent=2))
        if args.command == "push":
            command = ["kaggle", "kernels", "push", "-p", str(root)]
            if not args.run:
                command.append("--no-run")
            return subprocess.run(command).returncode
        return 0
    except PreflightError as exc:
        print(f"KAGGLE PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
