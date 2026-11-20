import json

import numpy as np
import pandas as pd

from src.pipeline import (_register_no_validation_epoch_checkpoints,
                         _training_partitions, _train_fold)
from src.registry import Registry
from src.data_factory import _normal_text


def test_no_validation_partition_uses_every_combined_row(tmp_path):
    rows = pd.DataFrame({"id": ["a", "b", "c"], "prompt_id": ["p1", "p2", "p3"],
                         "text": ["external", "v0", "increment"],
                         "model": ["label-a", "label-b", "label-c"],
                         "sampling_component": ["v2.14", "v0", "increment"],
                         "sampling_weight": [1.0, 10.0, 1.0]})
    rows.to_parquet(tmp_path / "dataset_mix.parquet", index=False)

    train, valid = _training_partitions("mix", "none", tmp_path)

    assert train.id.tolist() == ["a", "b", "c"]
    assert train.loc[train.sampling_component.eq("v0"), "sampling_weight"].tolist() == [10.0]
    assert valid.empty


def test_registered_granite_mix_has_all_rows_labels_and_source_weights():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    rows = pd.read_parquet(root / "data/processed/dataset_granite4_350m_v214_openpool16_v0w10.parquet")
    profile = json.loads((root / "data/processed/dataset_profile_granite4_350m_v214_openpool16_v0w10.json").read_text())

    assert len(rows) == 241225
    assert rows.model.astype(str).nunique() == 18
    assert not rows.id.astype(str).duplicated().any()
    assert not rows.text.astype(str).map(_normal_text).duplicated().any()
    assert rows.groupby("sampling_component").sampling_weight.first().to_dict() == {
        "v0": 10.0, "v2.14": 1.0, "v2.16_openpool16_increment": 1.0}
    assert rows.groupby("sampling_component").sampling_weight.nunique().eq(1).all()
    assert profile["component_hashes"]
    assert profile["v0_exposed"] is True


def test_lora_no_validation_forwards_no_labels_and_component_sampling(monkeypatch, tmp_path):
    import src.pipeline as pipeline

    captured = {}

    def fake_train(train_text, y_train, validation_text, classes, **kwargs):
        captured.update(kwargs)
        captured["validation_text"] = validation_text
        return {"checkpoint": str(tmp_path / "checkpoint")}

    monkeypatch.setattr(pipeline, "train_qwen_lora_fold", fake_train)
    train = pd.DataFrame({"text": ["external", "v0"], "model": ["a", "b"],
                          "sampling_weight": [1.0, 10.0],
                          "sampling_component": ["v2.14", "v0"]})
    valid = train.iloc[0:0].copy()
    config = {"checkpoint": "granite", "revision": "pinned", "max_length": 512,
              "weighted_sampling": True, "epochs": 4, "save_epoch_checkpoints": True,
              "epoch_checkpoint_style": "underscore", "training_lineage": "V0-exposed"}

    _train_fold("granite4_350m", train, valid, ["a", "b"], config, tmp_path,
                2026, "no_validation", tmp_path / "features", no_validation=True)

    assert captured["validation_labels"] is None
    assert captured["validation_text"] == []
    assert np.asarray(captured["sampling_weights"]).tolist() == [1.0, 10.0]
    assert captured["sampling_components"] == ["v2.14", "v0"]
    assert captured["epoch_checkpoint_style"] == "underscore"
    assert captured["training_lineage"] == "V0-exposed"


def test_epoch_checkpoint_registration_is_separate_and_v0_exposed(tmp_path):
    checkpoint_paths = []
    for epoch in range(1, 5):
        checkpoint = tmp_path / f"epoch_{epoch}"
        checkpoint.mkdir()
        for name in ("adapter_config.json", "classifier.pt", "tokenizer_config.json", "adapter_model.safetensors"):
            (checkpoint / name).write_text("{}")
        labels = [f"label-{index}" for index in range(18)]
        (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
            "classes": labels, "label_to_id": {label: index for index, label in enumerate(labels)},
            "validation_enabled": False, "training_lineage": "V0-exposed"}))
        checkpoint_paths.append(str(checkpoint))
    manifest = {"experiment_id": "granite_mix_test", "model": "granite4_350m",
        "config": {"epochs": 4}, "epoch_checkpoints": checkpoint_paths,
        "dataset_sha256": "abc", "code_revision": "def", "seed": 2026,
        "config_sha256": "cfg", "split_sha256": "split", "model_revision": "rev",
        "training_lineage": "V0-exposed"}

    records = _register_no_validation_epoch_checkpoints(manifest, tmp_path / "registry.sqlite")
    rows = Registry(tmp_path / "registry.sqlite").rows()

    assert [record["epoch"] for record in records] == [1, 2, 3, 4]
    assert len(rows) == 4
    assert all(row["split"] == "no_validation_epoch_checkpoint" for row in rows)
    assert all(json.loads(row["config_json"])["checkpoint_exposure"] == "V0-exposed" for row in rows)
    assert all("not eligible" in row["description"] for row in rows)


def test_experiment_is_queued_without_starting_worker():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    queue_files = sorted((root / "artifacts/training_queues").glob(
        "granite4_350m_v214_openpool16_v0w10_*/queue.json"))
    assert queue_files
    state = json.loads(queue_files[-1].read_text())

    assert state["status"] == "queued_waiting_for_previous_queue"
    assert state["worker_pid"] is None
    assert state["worker_started"] is False
    assert state["results"][0]["dataset_rows"] == 241225
    assert state["results"][0]["validation_enabled"] is False
    assert state["results"][0]["v0_exposed"] is True
