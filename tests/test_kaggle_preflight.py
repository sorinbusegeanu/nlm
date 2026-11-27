import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scripts.kaggle_preflight import (
    PreflightError,
    _cmd_submit,
    _audit_local_sklearn_artifacts,
    embed_runtime_preflight,
    runtime_preflight_code,
    to_json_safe,
    validate_kernel_package,
    validate_remote_sources,
    validate_submission_file,
    write_json,
)
from src.sklearn_compat import patch_serialized_sklearn_estimators


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


def test_local_preflight_rejects_title_and_kernel_slug_mismatch(tmp_path):
    root = _kernel(tmp_path)
    metadata_path = root / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["title"] = "E42 E13-B Plus E41"
    metadata["id"] = "zodrak/e42-e13b-plus-e41"
    write_json(metadata_path, metadata)
    with pytest.raises(PreflightError, match="kernel title resolves"):
        validate_kernel_package(root)
    metadata["id"] = "zodrak/e42-e13-b-plus-e41"
    write_json(metadata_path, metadata)
    assert validate_kernel_package(root)["status"] == "PASS"


def test_worker_preflight_pins_and_smoke_tests_packaged_inference_import(tmp_path):
    worker = '''import inspect,os,sys
from pathlib import Path
runtime_root=Path(os.environ["NLM_INFERENCE_RUNTIME"]).resolve()
sys.path.insert(0,str(runtime_root))
from src.inference import predict_candidate
expected_inference=(runtime_root/"src"/"inference.py").resolve()
actual_inference=Path(inspect.getfile(predict_candidate)).resolve()
if actual_inference!=expected_inference: raise RuntimeError("wrong inference module")
if "return_batch_metadata" not in inspect.signature(predict_candidate).parameters: raise RuntimeError("wrong API")
tasks_path,out_dir,text_path,classes_path,go_path,ready_path=sys.argv[1:]
'''
    root = _kernel(tmp_path, "WORKER_CODE = " + repr(worker) + "\n")
    runtime = root.parent / "test-data" / "runtime" / "src"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("")
    inference = runtime / "inference.py"
    inference.write_text("def predict_candidate(*args, return_batch_metadata=False): return None\n")
    assert validate_kernel_package(root)["worker_import"]["shadow_package_test"] == "PASS"

    inference.write_text("def predict_candidate(*args): return None\n")
    with pytest.raises(PreflightError, match="worker import smoke test failed"):
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


def test_sklearn_compatibility_repairs_legacy_logistic_and_checks_probabilities():
    rng = np.random.RandomState(8)
    features = rng.normal(size=(64, 4)).astype(np.float32)
    labels = (features[:, 0] - features[:, 2] > 0).astype(np.int64)
    model = LogisticRegression(random_state=8).fit(features, labels)
    expected = model.predict_proba(features[:5])
    if hasattr(model, "multi_class"):
        del model.multi_class

    audit = patch_serialized_sklearn_estimators({"pair_model": model})
    assert model.multi_class == "auto"
    assert audit["patched_attributes"][0]["attribute"] == "multi_class"
    assert audit["smoke_tests"][0]["predict_proba"] == "PASS"
    np.testing.assert_array_equal(model.predict_proba(features[:5]), expected)


def test_local_sklearn_artifact_preflight_loads_joblib_and_records_repairs(tmp_path):
    root = _kernel(tmp_path)
    artifact = root.parent / "test-data" / "pair_model.joblib"
    artifact.parent.mkdir(parents=True)
    rng = np.random.RandomState(3)
    features = rng.normal(size=(40, 3)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64)
    estimator = LogisticRegression(random_state=3).fit(features, labels)
    if hasattr(estimator, "multi_class"):
        del estimator.multi_class
    joblib.dump({"model": estimator}, artifact)

    audit = _audit_local_sklearn_artifacts(root, [{
        "dataset_ref": "zodrak/test-data",
        "filename": "deployment/pair_model.joblib",
        "size": artifact.stat().st_size,
    }])
    assert audit[0]["filename"] == "pair_model.joblib"
    assert audit[0]["patched_attributes"][0]["attribute"] == "multi_class"
    assert audit[0]["smoke_tests"][0]["predict_proba"] == "PASS"


def test_runtime_preflight_audits_joblib_assets_before_inference():
    code = runtime_preflight_code({
        "dataset_sources": ["zodrak/test-data"],
        "required_dataset_files": [{
            "dataset_ref": "zodrak/test-data", "filename": "pair_model.joblib", "size": 12
        }],
    })
    compile(code, "<runtime-preflight>", "exec")
    assert "_kp_compatible_joblib_load" in code
    assert "_kpjoblib.load(_candidates[0])" in code
    assert "patch_serialized_sklearn_estimators(_obj)" in code


def test_runtime_preflight_repairs_mounted_artifact_before_inference(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    mount = input_root / "datasets" / "zodrak" / "test-data"
    mount.mkdir(parents=True)
    work = tmp_path / "working"
    work.mkdir()
    artifact = mount / "pair_model.joblib"
    rng = np.random.RandomState(31)
    features = rng.normal(size=(50, 3)).astype(np.float32)
    estimator = LogisticRegression(random_state=31).fit(features, (features[:, 0] > 0).astype(int))
    if hasattr(estimator, "multi_class"):
        del estimator.multi_class
    joblib.dump({"model": estimator}, artifact)

    code = runtime_preflight_code({
        "dataset_sources": ["zodrak/test-data"],
        "model_sources": [],
        "required_dataset_files": [{
            "dataset_ref": "zodrak/test-data", "filename": "pair_model.joblib",
            "size": artifact.stat().st_size,
        }],
    })
    original_load = joblib.load
    monkeypatch.setenv("NLM_KAGGLE_INPUT_DIR", str(input_root))
    monkeypatch.setenv("NLM_KAGGLE_WORKING_DIR", str(work))
    try:
        exec(compile(code, "<runtime-preflight>", "exec"), {})
        report = json.loads((work / "kaggle_preflight.json").read_text())
        assert report["status"] == "PASS"
        assert report["sklearn_artifacts"][0]["patched_attributes"][0]["attribute"] == "multi_class"
        assert report["sklearn_artifacts"][0]["smoke_tests"][0]["predict_proba"] == "PASS"
    finally:
        joblib.load = original_load


def test_runtime_json_wrapper_does_not_rewrite_embedded_worker_strings(tmp_path):
    root = _kernel(tmp_path, 'import json\nWORKER_CODE = "json.dumps(payload)"\njson.dumps(payload)\n')
    rewritten = embed_runtime_preflight(root)
    notebook = nbformat.read(root / "main.ipynb", as_version=4)
    source = notebook.cells[2].source
    assert rewritten == 1
    assert 'WORKER_CODE = "json.dumps(payload)"' in source
    assert "kaggle_json_dumps(payload)" in source
    compile(source, "<notebook>", "exec")


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


def test_compatibility_audit_only_loads_explicit_runtime_joblibs(tmp_path):
    root = _kernel(tmp_path)
    metadata_path = root / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sklearn_compatibility_assets"] = [{
        "dataset_ref": "zodrak/test-data", "filename": "deployment/grouper.joblib"}]
    write_json(metadata_path, metadata)
    manifest = tmp_path / "published_bundle_manifest.json"
    write_json(manifest, {"files": [
        {"dataset_ref": "zodrak/test-data", "flat_filename": "deployment/grouper.joblib", "size": 9},
        {"dataset_ref": "zodrak/test-data", "flat_filename": "unused/checkpoint.joblib", "size": 12},
    ]})
    embed_runtime_preflight(root, manifest_paths=[manifest])
    spec = json.loads((root / "kaggle-preflight.json").read_text())
    assert spec["sklearn_compatibility_assets"] == [
        {"dataset_ref": "zodrak/test-data", "filename": "deployment/grouper.joblib", "size": 9}
    ]
    assert "_KAGGLE_PREFLIGHT_SPEC.get(\"sklearn_compatibility_assets\")" in runtime_preflight_code(spec)


def test_explicit_compatibility_allowlist_survives_push_reembedding(tmp_path):
    root = _kernel(tmp_path)
    manifest = tmp_path / "published_bundle_manifest.json"
    write_json(manifest, {"files": [
        {"dataset_ref": "zodrak/test-data", "flat_filename": "grouper.joblib", "size": 10},
        {"dataset_ref": "zodrak/test-data", "flat_filename": "checkpoint.joblib", "size": 12},
    ]})
    selected = [{"dataset_ref": "zodrak/test-data", "filename": "grouper.joblib"}]
    embed_runtime_preflight(root, manifest_paths=[manifest], sklearn_compatibility_assets=selected)
    embed_runtime_preflight(root, manifest_paths=[manifest])
    spec = json.loads((root / "kaggle-preflight.json").read_text())
    assert [x["filename"] for x in spec["sklearn_compatibility_assets"]] == ["grouper.joblib"]


def test_runtime_dependency_requirements_survive_push_reembedding(tmp_path):
    root = _kernel(tmp_path)
    embed_runtime_preflight(
        root, strict_versions={"scikit-learn": "1.9.1"},
        required_distributions=["tiktoken", "catboost"],
        processor_sources=["owner/model/Transformers/default/1"])
    embed_runtime_preflight(root)
    spec = json.loads((root / "kaggle-preflight.json").read_text())
    assert spec["strict_versions"] == {"scikit-learn": "1.9.1"}
    assert spec["required_distributions"] == ["tiktoken", "catboost"]
    assert spec["processor_sources"] == ["owner/model/Transformers/default/1"]


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
    assert calls[1][0][calls[1][0].index("-f") + 1] == "submission.parquet"
    assert calls[1][1] == output_dir.resolve() / "submit-1"
    assert (output_dir / "submit-1/submission.parquet").is_file()


def test_submission_wrapper_canonicalizes_each_variant_before_submit(tmp_path, monkeypatch):
    expected_ids = tmp_path / "test.csv"
    pd.DataFrame({"id": ["r1", "r2"]}).to_csv(expected_ids, index=False)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["a", "b"]))
    output_dir = tmp_path / "output"
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append((command, kwargs.get("cwd")))
        if command[1:3] == ["kernels", "output"]:
            destination = Path(command[command.index("-p") + 1])
            for filename, models in (("submission.parquet", ["a", "b"]),
                                     ("submission_e19_f_a.parquet", ["b", "a"])):
                pd.DataFrame({"id": ["r1", "r2"], "model": models}).to_parquet(
                    destination / filename, index=False)
            (destination / "kaggle_preflight.json").write_text(json.dumps({"status": "PASS", "errors": []}))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.kaggle_preflight.subprocess.run", fake_run)
    args = SimpleNamespace(kernel="owner/kernel", version=9, competition="name-that-llm",
        output_dir=str(output_dir), file=["submission.parquet", "submission.parquet"],
        variant_source_file=["submission.parquet", "submission_e19_f_a.parquet"],
        labels_json=str(labels), expected_ids=str(expected_ids), message="E33 {file}")
    assert _cmd_submit(args) == 0
    submit_calls = [item for item in calls if item[0][1:3] == ["competitions", "submit"]]
    assert len(submit_calls) == 2
    assert all(item[0][item[0].index("-f") + 1] == "submission.parquet" for item in submit_calls)
    assert (output_dir / "submit-1/submission.parquet").is_file()
    assert (output_dir / "submit-2/submission.parquet").is_file()


def test_remote_model_preflight_accepts_unique_nested_config_and_processor(tmp_path, monkeypatch):
    kernel = _kernel(tmp_path)
    ref = "owner/nested-qwen/PyTorch/default/1"
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": "owner/test", "title": "test", "code_file": "test.ipynb",
        "kernel_type": "notebook", "language": "python", "enable_gpu": True,
        "dataset_sources": [], "model_sources": [ref]}))
    (kernel / "kaggle-preflight.json").write_text(json.dumps({"processor_sources": [ref]}))
    names = ["Qwen3.5-4B/config.json", "Qwen3.5-4B/tokenizer.json",
             "Qwen3.5-4B/processor_config.json"]
    monkeypatch.setattr("scripts.kaggle_preflight._run_json_command",
        lambda command: [{"name": name, "size": 10} for name in names])
    result = validate_remote_sources(kernel)
    assert result["models"][ref]["model_subdirectory"] == "Qwen3.5-4B"
    assert result["models"][ref]["has_processor_config"] is True


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


def test_submission_wrapper_requires_canonical_code_output_filename(tmp_path, monkeypatch):
    expected_ids = tmp_path / "test.csv"
    pd.DataFrame({"id": ["r1"]}).to_csv(expected_ids, index=False)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["a", "b"]))
    args = SimpleNamespace(
        kernel="owner/kernel", version=3, competition="name-that-llm",
        output_dir=str(tmp_path / "output"), file=["submission_e28_b.parquet"],
        labels_json=str(labels), expected_ids=str(expected_ids), message="test {file}",
    )
    calls = []
    monkeypatch.setattr("scripts.kaggle_preflight.subprocess.run", lambda *a, **k: calls.append(a))
    with pytest.raises(PreflightError, match="canonical output filename submission.parquet"):
        _cmd_submit(args)
    assert calls == []
