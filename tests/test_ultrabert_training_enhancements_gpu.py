"""GPU-only smoke tests; run on the training host when its GPU is available."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model_factory import _probe_max_batch_size, build_ultrabert_enhanced_classifier


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a free CUDA device")


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return (torch.relu(self.linear(x) + x),)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=4)
        self.embeddings = nn.Embedding(64, 8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([Block(8) for _ in range(4)])

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            x = layer(x)[0]
        return SimpleNamespace(last_hidden_state=x)


class ProbeDataset:
    lengths = [16, 15, 14, 13]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor([i + 1] * 16),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.tensor(i % 2)}

    def collate_fn(self, rows):
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def test_cuda_training_backward_and_inference_forward():
    model = build_ultrabert_enhanced_classifier(Backbone(), ["a", "b"],
        supervised_contrastive={"enabled": True, "weight": .1, "projection_dim": 4},
        multi_layer_aggregation={"enabled": True, "num_layers": 4}).cuda()
    ids = torch.randint(0, 64, (4, 16), device="cuda")
    mask = torch.ones_like(ids)
    labels = torch.tensor([0, 0, 1, 1], device="cuda")
    model.train()
    output = model(ids, mask, labels=labels)
    output.loss.backward()
    assert model.aggregation_logits.grad is not None
    assert model.projection_head[-1].weight.grad is not None
    model.eval()
    with torch.inference_mode():
        scores = model(ids, mask).logits
    assert scores.shape == (4, 2) and torch.isfinite(scores).all()


def test_cuda_automatic_batch_probe_includes_enhancement_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("NLM_BATCH_SIZE_CACHE", str(tmp_path / "batch_cache"))
    model = build_ultrabert_enhanced_classifier(Backbone(), ["a", "b"],
        supervised_contrastive={"enabled": True, "weight": .1, "projection_dim": 4},
        multi_layer_aggregation={"enabled": True, "num_layers": 4})
    size, record = _probe_max_batch_size(
        model, ProbeDataset(), requested_batch_size=2, max_batch_size=4,
        safety_fraction=.95, optimizer_name="adamw_torch")
    assert size >= 1 and record["status"] in {"selected", "cached_validated", "safe_maximum_fallback"}
