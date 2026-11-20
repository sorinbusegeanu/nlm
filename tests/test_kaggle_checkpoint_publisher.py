import hashlib
import json
from pathlib import Path

import pytest

from scripts.publish_kaggle_checkpoint_store import prepare_delta, resolve_catalog_checkpoint


def _catalog():
    return {
        "schema_version": 1,
        "dataset_ref": "zodrak/zod006-models",
        "dataset_version": 2,
        "assets": {},
        "checkpoints": {},
    }


def _checkpoint(experiment_id, file_path):
    return {
        "experiment_id": experiment_id,
        "model": "ultrabert380",
        "training_dataset": "v2.16",
        "base_model": "answerdotai/ultrabert-380m",
        "base_model_revision": "a" * 40,
        "classes": ["label-a", "label-b"],
        "config": {"max_length": 512},
        "files": [(file_path, "model.safetensors")],
        "source_manifest": f"artifacts/predictions/{experiment_id}_manifest.json",
    }


def test_delta_stages_new_content_and_updated_catalog(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{"id":"zodrak/zod006-models"}')
    payload = tmp_path / "model.safetensors"
    payload.write_bytes(b"new checkpoint payload")
    stage = tmp_path / "stage"

    result = prepare_delta(
        store_dir=store,
        stage_dir=stage,
        remote={"old-file": 4},
        catalog=_catalog(),
        checkpoints=[_checkpoint("run-1", payload)],
        next_version=3,
    )

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert result["staged_files"] == [f"asset__{digest}", "checkpoint_catalog.json"]
    assert (stage / f"asset__{digest}").read_bytes() == payload.read_bytes()
    catalog = json.loads((stage / "checkpoint_catalog.json").read_text())
    assert catalog["checkpoints"]["run-1"]["files"][0]["kaggle_filename"] == f"asset__{digest}"
    assert catalog["checkpoints"]["run-1"]["introduced_in_version"] == 3
    assert resolve_catalog_checkpoint(catalog, "run-1")["model"] == "ultrabert380"


def test_identical_assets_are_uploaded_once_for_distinct_checkpoints(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{}')
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same weights")
    b.write_bytes(b"same weights")

    result = prepare_delta(
        store_dir=store,
        stage_dir=tmp_path / "stage",
        remote={},
        catalog=_catalog(),
        checkpoints=[_checkpoint("run-a", a), _checkpoint("run-b", b)],
        next_version=3,
    )

    digest = hashlib.sha256(a.read_bytes()).hexdigest()
    assert result["staged_files"].count(f"asset__{digest}") == 1
    catalog = json.loads((tmp_path / "stage/checkpoint_catalog.json").read_text())
    assert catalog["checkpoints"]["run-a"]["files"][0]["kaggle_filename"] == f"asset__{digest}"
    assert catalog["checkpoints"]["run-b"]["files"][0]["kaggle_filename"] == f"asset__{digest}"


def test_remote_content_addressed_asset_is_not_restaged(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{}')
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"already uploaded")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    catalog = _catalog()
    catalog["assets"][digest] = {"size": payload.stat().st_size, "kaggle_filename": f"asset__{digest}"}

    result = prepare_delta(
        store_dir=store,
        stage_dir=tmp_path / "stage",
        remote={f"asset__{digest}": payload.stat().st_size},
        catalog=catalog,
        checkpoints=[_checkpoint("run-1", payload)],
        next_version=3,
    )

    assert result["staged_files"] == ["checkpoint_catalog.json"]
    assert not (tmp_path / "stage" / f"asset__{digest}").exists()


def test_repeat_publication_with_same_catalog_is_empty_delta(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{}')
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"already published")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    catalog = _catalog()
    first = prepare_delta(
        store_dir=store,
        stage_dir=tmp_path / "first",
        remote={},
        catalog=catalog,
        checkpoints=[_checkpoint("run-1", payload)],
        next_version=3,
    )
    catalog = json.loads((tmp_path / "first/checkpoint_catalog.json").read_text())
    remote = {f"asset__{digest}": payload.stat().st_size, "checkpoint_catalog.json": 10}

    second = prepare_delta(
        store_dir=store,
        stage_dir=tmp_path / "second",
        remote=remote,
        catalog=catalog,
        checkpoints=[_checkpoint("run-1", payload)],
        next_version=4,
    )

    assert first["staged_files"]
    assert second["staged_files"] == []
    assert second["catalog_changed"] is False


def test_checkpoint_id_cannot_be_reused_for_different_payload(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{}')
    old = tmp_path / "old.bin"
    new = tmp_path / "new.bin"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    catalog = _catalog()
    catalog["checkpoints"]["run-1"] = {
        "files": [{"path": "model.safetensors", "sha256": hashlib.sha256(b"old").hexdigest()}]
    }

    with pytest.raises(ValueError, match="already exists with different payload"):
        prepare_delta(
            store_dir=store,
            stage_dir=tmp_path / "stage",
            remote={},
            catalog=catalog,
            checkpoints=[_checkpoint("run-1", new)],
            next_version=3,
        )


def test_remote_size_conflict_is_rejected(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{}')
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"checkpoint")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    catalog = _catalog()
    catalog["assets"][digest] = {"size": payload.stat().st_size, "kaggle_filename": f"asset__{digest}"}

    with pytest.raises(ValueError, match="remote asset size conflict"):
        prepare_delta(
            store_dir=store,
            stage_dir=tmp_path / "stage",
            remote={f"asset__{digest}": payload.stat().st_size + 1},
            catalog=catalog,
            checkpoints=[_checkpoint("run-1", payload)],
            next_version=3,
        )


def test_staging_directory_must_be_empty(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "dataset-metadata.json").write_text('{}')
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "stale-large-file").write_bytes(b"do not reupload")

    with pytest.raises(ValueError, match="must be empty"):
        prepare_delta(
            store_dir=store,
            stage_dir=stage,
            remote={},
            catalog=_catalog(),
            checkpoints=[],
            next_version=3,
        )


def test_catalog_lookup_reports_missing_id():
    with pytest.raises(KeyError, match="not in the Kaggle catalog"):
        resolve_catalog_checkpoint(_catalog(), "not-published")
