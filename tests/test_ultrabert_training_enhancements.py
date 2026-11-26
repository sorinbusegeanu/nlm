from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from src.model_factory import (
    LabelAwareBatchSampler,
    build_ultrabert_enhanced_classifier,
    evenly_spaced_layer_indices,
    normalize_multi_layer_aggregation_config,
    normalize_supervised_contrastive_config,
    supervised_contrastive_loss,
    supervised_contrastive_objective,
)
from src.pipeline import _validate_training_overrides


class TinyBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden):
        return (torch.tanh(self.proj(hidden) + hidden),)


class TinyBackbone(nn.Module):
    def __init__(self, hidden=8, layer_count=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layer_count)
        self.embeddings = nn.Embedding(32, hidden)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([TinyBlock(hidden) for _ in range(layer_count)])

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, path, safe_serialization=True):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": "tiny"}))
        torch.save(self.state_dict(), path / "tiny_backbone.pt")


def _inputs(rows=6):
    torch.manual_seed(17)
    return torch.randint(0, 32, (rows, 5)), torch.tensor(
        [[1, 1, 1, 0, 0]] * rows, dtype=torch.long)


def test_supervised_contrastive_matches_hand_calculation_and_ignores_self():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                              requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    actual = supervised_contrastive_loss(embeddings, labels, temperature=1.0)
    expected = torch.log1p(2.0 * torch.exp(torch.tensor(-1.0)))
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def test_supervised_contrastive_is_finite_in_float32_and_zero_without_positive_pairs():
    embeddings = torch.tensor([[1e20, 0.0], [0.0, 1e20], [-1e20, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    loss = supervised_contrastive_loss(embeddings, labels, temperature=0.01)
    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
    loss.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def test_label_aware_sampler_pairs_labels_diversifies_prompts_and_is_reproducible():
    labels = ["a", "a", "a", "a", "b", "b", "b", "b"]
    prompts = ["p1", "p2", "p1", "p2", "p1", "p2", "p1", "p2"]
    sampler = LabelAwareBatchSampler(labels, prompts, batch_size=4, seed=91)
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(len(labels)))
    assert sampler.epoch_records[0]["each_row_once"]
    for batch in batches:
        for label in {labels[i] for i in batch}:
            members = [i for i in batch if labels[i] == label]
            if len(members) > 1:
                assert len({prompts[i] for i in members}) > 1
    repeat = LabelAwareBatchSampler(labels, prompts, batch_size=4, seed=91)
    assert list(repeat) == batches
    # At least one cross-label same-prompt pair is arranged as a hard negative.
    assert sum(item["hard_negative_pairs"] for item in sampler.epoch_records) > 0


def test_label_aware_sampler_handles_singletons_and_batch_size_one():
    sampler = LabelAwareBatchSampler(["a", "b", "c"], ["p", "p", "q"],
                                     batch_size=1, seed=7)
    batches = list(sampler)
    assert len(batches) == 3
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]
    assert sampler.epoch_records[0]["positive_anchor_rows"] == 0


def test_dynamic_layer_selection_and_attention_mask_mean_aggregation_have_gradients():
    assert evenly_spaced_layer_indices(12, 4) == [0, 4, 7, 11]
    assert evenly_spaced_layer_indices(6, 1) == [5]
    model = build_ultrabert_enhanced_classifier(
        TinyBackbone(), ["a", "b"],
        multi_layer_aggregation={"enabled": True, "num_layers": 4})
    assert model.selected_layer_indices == [0, 1, 2, 3]
    assert torch.allclose(torch.softmax(model.aggregation_logits, dim=0), torch.full((4,), .25))
    ids, mask = _inputs()
    logits = model(ids, mask).logits
    F.cross_entropy(logits, torch.tensor([0, 1, 0, 1, 0, 1])).backward()
    assert model.aggregation_logits.grad is not None
    assert torch.isfinite(model.aggregation_logits.grad).all()
    for index in model.selected_layer_indices:
        assert model.backbone.encoder.layer[index].proj.weight.grad is not None


def test_contrastive_and_multilayer_combined_objective_backpropagates_all_heads():
    model = build_ultrabert_enhanced_classifier(
        TinyBackbone(), ["a", "b"],
        supervised_contrastive={"enabled": True, "weight": .2, "projection_dim": 5},
        multi_layer_aggregation={"enabled": True, "num_layers": 4})
    ids, mask = _inputs()
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    output = model(ids, mask, labels=labels)
    total, ce, contrastive = supervised_contrastive_objective(
        output.logits, output.contrastive_embeddings, labels,
        contrastive_weight=.2, temperature=.07)
    assert torch.allclose(total, ce + .2 * contrastive)
    total.backward()
    assert model.projection_head[-1].weight.grad is not None
    assert model.classifier.weight.grad is not None
    assert model.aggregation_logits.grad is not None
    assert all(layer.proj.weight.grad is not None for layer in model.backbone.encoder.layer)


def test_enhanced_checkpoint_roundtrip_preserves_readout_and_layer_weights(tmp_path):
    config = {"enabled": True, "num_layers": 4, "pooling": "mean",
              "aggregation": "learned_weighted_sum"}
    contrastive = {"enabled": True, "weight": .1, "temperature": .07, "projection_dim": 6}
    model = build_ultrabert_enhanced_classifier(TinyBackbone(), ["a", "b"],
        supervised_contrastive=contrastive, multi_layer_aggregation=config)
    with torch.no_grad():
        model.classifier.weight.fill_(.31)
        model.aggregation_logits.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
        for parameter in model.projection_head.parameters():
            parameter.fill_(.12)
    model.save_pretrained(tmp_path)
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert manifest["ultrabert_enhancements"]["selected_layer_indices"] == [0, 1, 2, 3]
    restored = build_ultrabert_enhanced_classifier(TinyBackbone(), ["a", "b"],
        supervised_contrastive=contrastive, multi_layer_aggregation=config,
        state_dir=tmp_path, selected_layer_indices=[0, 1, 2, 3])
    assert torch.equal(restored.classifier.weight, model.classifier.weight)
    assert torch.equal(restored.aggregation_logits, model.aggregation_logits)
    for left, right in zip(restored.projection_head.parameters(), model.projection_head.parameters()):
        assert torch.equal(left, right)
    restored.backbone.load_state_dict(torch.load(tmp_path / "tiny_backbone.pt", weights_only=True))
    ids, mask = _inputs(rows=3)
    model.eval(); restored.eval()
    assert torch.equal(model(ids, mask).logits, restored(ids, mask).logits)


def test_baseline_defaults_and_existing_ultrabert_overrides_remain_compatible():
    assert normalize_supervised_contrastive_config()["enabled"] is False
    assert normalize_multi_layer_aggregation_config()["enabled"] is False
    _validate_training_overrides("ultrabert380", {})
    _validate_training_overrides("ultrabert380", {
        "group_aware_sampling": True, "group_ranking_lambda": 0.1,
        "supervised_contrastive": {"enabled": False},
        "multi_layer_aggregation": {"enabled": False}})
    with pytest.raises(ValueError, match="cannot be combined"):
        _validate_training_overrides("ultrabert380", {
            "weighted_sampling": True,
            "supervised_contrastive": {"enabled": True}})


def test_invalid_enhancement_settings_are_rejected():
    with pytest.raises(ValueError, match="temperature"):
        normalize_supervised_contrastive_config({"enabled": True, "temperature": 0})
    with pytest.raises(ValueError, match="num_layers"):
        normalize_multi_layer_aggregation_config({"enabled": True, "num_layers": 0})
