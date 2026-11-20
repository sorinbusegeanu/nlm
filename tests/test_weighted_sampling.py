import numpy as np
import pandas as pd
import pytest

from src.model_factory import _make_epoch_weighted_sampler
from src.pipeline import _partition_sha256, _train_fold, _weighted_sampling_dataset_summary


def _draw(sampler):
    return list(iter(sampler))


def test_weighted_sampler_draws_more_high_weight_rows_and_exact_epoch_size():
    weights = [1.0] * 500 + [9.0] * 500
    labels = ["regular"] * 500 + ["hard"] * 500
    sampler = _make_epoch_weighted_sampler(
        weights, seed=2026, labels=labels)

    sampled = _draw(sampler)

    assert len(sampled) == len(weights)
    assert sum(index >= 500 for index in sampled) > 800
    assert sampler.epoch_records == [{
        "epoch": 1, "seed": 2026, "sampled_examples": len(weights),
        "sampled_rows_by_class": {"hard": sum(index >= 500 for index in sampled),
                                  "regular": sum(index < 500 for index in sampled)},
    }]


def test_weighted_sampler_seed_changes_each_epoch_but_is_reproducible():
    first = _make_epoch_weighted_sampler([1.0] * 32, seed=17)
    second = _make_epoch_weighted_sampler([1.0] * 32, seed=17)

    first_epoch_1 = _draw(first)
    first_epoch_2 = _draw(first)
    assert first_epoch_1 == _draw(second)
    assert first_epoch_2 == _draw(second)
    assert first_epoch_1 != first_epoch_2
    assert [row["seed"] for row in first.epoch_records] == [17, 18]


def test_weighted_sampler_records_component_draws_and_preserves_ten_to_one_weights():
    components = ["external"] * 500 + ["v0"] * 500
    sampler = _make_epoch_weighted_sampler(
        [1.0] * 500 + [10.0] * 500, seed=2026, components=components)

    sampled = _draw(sampler)

    assert len(sampled) == 1000
    counts = sampler.epoch_records[0]["sampled_rows_by_component"]
    assert counts["v0"] > 850
    assert counts["external"] < 150
    assert sum(counts.values()) == len(sampled)


@pytest.mark.parametrize("weights", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_weighted_sampler_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        _make_epoch_weighted_sampler(weights, seed=2026)


def test_weighted_sampling_manifest_reports_expected_class_draw_distribution():
    rows = pd.DataFrame({"model": ["a", "a", "b"],
                         "sampling_weight": [1.0, 3.0, 4.0]})

    summary = _weighted_sampling_dataset_summary(rows, enabled=True, seed=9)

    assert summary["replacement"] is True
    assert summary["num_samples_per_epoch"] == 3
    assert summary["class_distribution"]["a"]["expected_sampling_probability"] == pytest.approx(.5)
    assert summary["class_distribution"]["b"]["expected_draws_per_epoch"] == pytest.approx(1.5)


def test_sampling_weights_are_included_in_training_data_hash():
    first = pd.DataFrame({"id": ["r1"], "prompt_id": ["p1"], "text": ["text"],
                          "model": ["label"], "sampling_weight": [1.0]})
    changed = first.assign(sampling_weight=[2.0])

    assert _partition_sha256(first) != _partition_sha256(changed)


def test_disabled_weighted_sampling_keeps_existing_encoder_dispatch(monkeypatch, tmp_path):
    import src.pipeline as pipeline

    called = {}

    def fake_train_encoder(*args, **kwargs):
        called.update(kwargs)
        return {"checkpoint": str(tmp_path / "checkpoint")}

    monkeypatch.setattr(pipeline, "train_encoder_fold", fake_train_encoder)
    train = pd.DataFrame({"text": ["train"], "model": ["a"]})
    valid = pd.DataFrame({"text": ["valid"], "model": ["a"]})
    config = {"checkpoint": "local/model", "revision": "revision", "max_length": 32}

    result = _train_fold("ultrabert380", train, valid, ["a"], config, tmp_path,
                         2026, "validation", tmp_path / "features")

    assert result["checkpoint"] == str(tmp_path / "checkpoints" / "ultrabert380_foldvalidation")
    assert called["weighted_sampling"] is False
    assert called["sampling_weights"] is None


@pytest.mark.parametrize("model,trainer_name", [
    ("ultrabert380", "train_encoder_fold"),
    ("qwen06", "train_qwen_lora_fold"),
])
def test_enabled_weighted_sampling_forwards_training_weights_only(monkeypatch, tmp_path, model, trainer_name):
    import src.pipeline as pipeline

    called = {}

    def fake_train(*args, **kwargs):
        called.update(kwargs)
        return {"checkpoint": str(tmp_path / "checkpoint")}

    monkeypatch.setattr(pipeline, trainer_name, fake_train)
    train = pd.DataFrame({"text": ["train-a", "train-b"], "model": ["a", "b"],
                          "sampling_weight": [1.0, 4.0]})
    valid = pd.DataFrame({"text": ["valid-a", "valid-b"], "model": ["a", "b"]})
    config = {"checkpoint": "local/model", "revision": "revision", "max_length": 32,
              "weighted_sampling": True, "class_weights": {"a": 2.0}}

    _train_fold(model, train, valid, ["a", "b"], config, tmp_path,
                2026, "validation", tmp_path / "features")

    assert called["class_weights"] == {"a": 2.0}
    assert called["weighted_sampling"] is True
    np.testing.assert_array_equal(called["sampling_weights"], np.array([1.0, 4.0]))
    assert called["validation_labels"] == ["a", "b"]


def test_enabled_weighted_sampling_requires_dataset_weight_column(monkeypatch, tmp_path):
    import src.pipeline as pipeline

    monkeypatch.setattr(pipeline, "train_encoder_fold", lambda *args, **kwargs: pytest.fail(
        "model loader must not run when sampling weights are missing"))
    train = pd.DataFrame({"text": ["train"], "model": ["a"]})
    valid = pd.DataFrame({"text": ["valid"], "model": ["a"]})
    with pytest.raises(ValueError, match="requires a sampling_weight column"):
        _train_fold("ultrabert380", train, valid, ["a"],
                    {"checkpoint": "local/model", "revision": "revision", "max_length": 32,
                     "weighted_sampling": True}, tmp_path, 2026, "validation", tmp_path / "features")
