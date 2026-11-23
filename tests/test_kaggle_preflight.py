import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest

from scripts.kaggle_preflight import (
    PreflightError,
    _cmd_submit,
    embed_runtime_preflight,
    runtime_preflight_code,
    to_json_safe,
    validate_kernel_package,
    validate_remote_sources,
    validate_submission_file,
    write_json,
)


def _kernel(tmp_path: Path, code: str = "x = 1\n") -> Path:
    root = tmp_path / "kernel"
    root.mkdir()
    (root / "kernel-metadata.json").write_text(json.dumps({
        "id": "zodrak/test-kernel",
        "code_file": "main.ipynb",
        "kernel_type": "notebook",
        "language": "python",
        "enable_gpu": True,
        "dataset_sources": ["zodrak/test-data"],
        "model_sources": ["owner/model/Transformers/instance/1"],
    }))
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("Test"),
        nbformat.v4.new_code_cell(code),
    ])
    nbformat.write(notebook, root / "main.ipynb")
    (root / "manifest.json").write_text('{"classes":["a","b"]}')
    return root


def test_local_package_preflight_passes_and_catches_undefined_bootstrap(tmp_path):
    root = _kernel(tmp_path)
    assert validate_kernel_package(root)["status"] == "PASS"
    notebook_path = root / "main.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells[1].source = "print(RUNTIME)\n"
    nbformat.write(notebook, notebook_path)
    with pytest.raises(PreflightError, match="possibly undefined global 'RUNTIME'"):
        validate_kernel_package(root)


def test_runtime_preflight_is_inserted_before_inference_and_is_valid_python(tmp_path):
    root = _kernel(tmp_path, "import json as js\npayload = {'x': 1}\njson.dumps(payload)\njs.dump(payload, open('/tmp/x.json', 'w'))\n")
    embed_runtime_preflight(
        root,
        strict_versions={"scikit-learn": "1.9.1"},
        required_distributions=["torch"],
        processor_sources=["owner/model/Transformers/instance/1"],
    )
    notebook = nbformat.read(root / "main.ipynb", as_version=4)
    assert notebook.cells[1].metadata.tags == ["kaggle-preflight"]
    compile(notebook.cells[1].source, "<preflight>", "exec")
    assert "kaggle_json_dumps(payload)" in notebook.cells[2].source
    assert "kaggle_json_dump(payload," in notebook.cells[2].source
    spec = json.loads((root / "kaggle-preflight.json").read_text())
    assert spec["strict_versions"] == {"scikit-learn": "1.9.1"}
    assert "AutoProcessor" in notebook.cells[1].source
    assert validate_kernel_package(root)["status"] == "PASS"


def test_runtime_preflight_idempotent(tmp_path):
    root = _kernel(tmp_path)
    embed_runtime_preflight(root)
    embed_runtime_preflight(root)
    notebook = nbformat.read(root / "main.ipynb", as_version=4)
    assert sum("kaggle-preflight" in c.metadata.get("tags", []) for c in notebook.cells) == 1


def test_json_writer_converts_numpy_values_and_rejects_nan(tmp_path):
    converted = to_json_safe({"scalar": np.int64(7), "matrix": np.array([[1, 2]])})
    assert converted == {"scalar": 7, "matrix": [[1, 2]]}
    path = tmp_path / "out.json"
    write_json(path, converted)
    assert json.loads(path.read_text()) == converted
    with pytest.raises(ValueError, match="non-finite"):
        to_json_safe(float("nan"))


def test_submission_preflight_checks_schema_labels_and_row_order(tmp_path):
    path = tmp_path / "submission.parquet"
    pd.DataFrame({"id": ["a", "b"], "model": ["x", "y"]}).to_parquet(path, index=False)
    assert validate_submission_file(path, labels=["x", "y"], expected_ids=["a", "b"]) == ["a", "b"]
    with pytest.raises(PreflightError, match="unknown labels"):
        validate_submission_file(path, labels=["x"], expected_ids=["a", "b"])
    with pytest.raises(PreflightError, match="IDs/order"):
        validate_submission_file(path, labels=["x", "y"], expected_ids=["b", "a"])
    pd.DataFrame({"id": ["a", "a"], "model": ["x", "y"]}).to_parquet(path, index=False)
    with pytest.raises(PreflightError, match="duplicate IDs"):
        validate_submission_file(path)


def test_preflight_rejects_unpinned_or_malformed_model_references(tmp_path):
    root = _kernel(tmp_path)
    metadata_path = root / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_sources"] = ["owner/model/instance"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(PreflightError, match="owner/model/framework/instance/version"):
        validate_kernel_package(root)


def test_runtime_code_records_preflight_before_model_load():
    code = runtime_preflight_code({"dataset_sources": [], "model_sources": []})
    compile(code, "<runtime-preflight>", "exec")
    assert code.index("KAGGLE_PREFLIGHT") < code.index("if _kp_errors: raise RuntimeError")


def test_remote_preflight_checks_manifest_asset_presence_and_size(tmp_path, monkeypatch):
    root = _kernel(tmp_path)
    write_json(root / "kaggle-preflight.json", {
        "processor_sources": ["owner/model/Transformers/instance/1"]
    })
    write_json(root / "deployment_manifest.json", {
        "members": [{"files": [{"dataset_ref": "zodrak/test-data",
                                "flat_filename": "assets/ckpt.bin", "size": 5}]}]
    })

    def mock_inventory(command):
        if command[1] == "datasets":
            return [{"name": "part_00/assets/ckpt.bin", "size": 5}]
        return [{"name": name, "size": 1} for name in
                ("config.json", "tokenizer.json", "preprocessor_config.json")]

    monkeypatch.setattr("scripts.kaggle_preflight._run_json_command", mock_inventory)
    result = validate_remote_sources(root)
    assert result["required_manifest_assets_checked"] == 1
    assert result["models"]["owner/model/Transformers/instance/1"]["has_processor_config"]

    def wrong_size(command):
        records = mock_inventory(command)
        if command[1] == "datasets":
            records[0]["size"] = 6
        return records

    monkeypatch.setattr("scripts.kaggle_preflight._run_json_command", wrong_size)
    with pytest.raises(PreflightError, match="asset size mismatch"):
        validate_remote_sources(root)


def test_runtime_spec_includes_manifest_outside_kernel_directory(tmp_path):
    root = _kernel(tmp_path)
    manifest = tmp_path / "published_bundle_manifest.json"
    write_json(manifest, {"files": [{"dataset_ref": "zodrak/test-data",
                                     "flat_filename": "subdir/model.bin", "size": 9}]})
    embed_runtime_preflight(root, manifest_paths=[manifest])
    spec = json.loads((root / "kaggle-preflight.json").read_text())
    assert spec["required_dataset_files"] == [
        {"dataset_ref": "zodrak/test-data", "filename": "subdir/model.bin", "size": 9}
    ]
    assert "required checkpoint/input asset is missing" in nbformat.read(root / "main.ipynb", 4).cells[1].source


def test_submission_wrapper_fetches_exact_version_validates_and_submits(tmp_path, monkeypatch):
    expected_ids = tmp_path / "test.csv"
    pd.DataFrame({"id": ["r1", "r2"]}).to_csv(expected_ids, index=False)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"classes": ["a", "b"]}))
    output_dir = tmp_path / "output"
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append((command, kwargs.get("cwd")))
        if command[1:3] == ["kernels", "output"]:
            destination = Path(command[command.index("-p") + 1])
            pd.DataFrame({"id": ["r1", "r2"], "model": ["a", "b"]}).to_parquet(
                destination / "submission.parquet", index=False
            )
            (destination / "kaggle_preflight.json").write_text(json.dumps({"status": "PASS", "errors": []}))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.kaggle_preflight.subprocess.run", fake_run)
    args = SimpleNamespace(
        kernel="owner/kernel", version=7, competition="name-that-llm",
        output_dir=str(output_dir), file=["submission.parquet"],
        labels_json=str(labels), expected_ids=str(expected_ids), message="test {file}",
    )
    assert _cmd_submit(args) == 0
    assert calls[0][0][3] == "owner/kernel/7"
    assert calls[1][0][:4] == ["kaggle", "competitions", "submit", "name-that-llm"]
    assert calls[1][1] == output_dir.resolve()


def test_submission_wrapper_refuses_failed_runtime_preflight(tmp_path, monkeypatch):
    expected_ids = tmp_path / "test.csv"
    pd.DataFrame({"id": ["r1"]}).to_csv(expected_ids, index=False)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["a"]))
    output_dir = tmp_path / "output"
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        if command[1:3] == ["kernels", "output"]:
            destination = Path(command[command.index("-p") + 1])
            (destination / "kaggle_preflight.json").write_text(json.dumps({"status": "FAIL", "errors": ["missing model"]}))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.kaggle_preflight.subprocess.run", fake_run)
    args = SimpleNamespace(
        kernel="owner/kernel", version=3, competition="name-that-llm",
        output_dir=str(output_dir), file=["submission.parquet"],
        labels_json=str(labels), expected_ids=str(expected_ids), message="test {file}",
    )
    with pytest.raises(PreflightError, match="runtime preflight did not pass"):
        _cmd_submit(args)
    assert all("competitions" not in command for command in calls)
