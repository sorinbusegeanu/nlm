from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.model_factory import _e41_layer_indices, apply_transformer_layer_pruning
from src.pipeline import _model_cfg


class DummyAttention(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.layer_idx = index


class DummyLayer(torch.nn.Module):
    def __init__(self, index, layer_type):
        super().__init__()
        self.block_type = layer_type
        self.self_attn = DummyAttention(index) if layer_type == "full_attention" else None
        self.linear_attn = DummyAttention(index) if layer_type == "linear_attention" else None
        if self.linear_attn is not None:
            self.linear_attn.layer_type = layer_type


class DummyBackbone(torch.nn.Module):
    def __init__(self, layer_count=24):
        super().__init__()
        types = ["linear_attention" if i % 4 != 3 else "full_attention" for i in range(layer_count)]
        self.config = SimpleNamespace(num_hidden_layers=layer_count, layer_types=types,
                                      model_type="qwen3_5", hidden_size=256, use_cache=True)
        self.layers = torch.nn.ModuleList([DummyLayer(i, types[i]) for i in range(layer_count)])


def test_e41_masks_retain_same_ordered_layer_count():
    masks = [_e41_layer_indices(24, name) for name in ("E41-A", "E41-B", "E41-C")]
    assert [len(mask) for mask in masks] == [16, 16, 16]
    assert masks[0] == tuple(range(16))
    assert masks[1] == tuple(range(8)) + tuple(range(16, 24))
    assert tuple(i for i in range(24) if i not in masks[2]) == (2, 5, 8, 11, 14, 17, 20, 23)
    assert all(tuple(sorted(mask)) == mask for mask in masks)


@pytest.mark.parametrize("variant", ["E41-A", "E41-B", "E41-C"])
def test_pruning_updates_hybrid_metadata_and_reindexes_retained_layers(variant):
    backbone = DummyBackbone()
    result = apply_transformer_layer_pruning(backbone, {
        "variant": variant, "expected_num_layers": 24, "expected_retained_layers": 16,
    })
    retained = _e41_layer_indices(24, variant)
    assert len(backbone.layers) == backbone.config.num_hidden_layers == 16
    assert backbone.config.layer_types == [
        "linear_attention" if index % 4 != 3 else "full_attention" for index in retained
    ]
    assert result["retained_layer_indices"] == list(retained)
    for new_index, layer in enumerate(backbone.layers):
        module = layer.self_attn or layer.linear_attn
        assert module.layer_idx == new_index
        assert layer.block_type == backbone.config.layer_types[new_index]


def test_saved_mask_mismatch_fails_closed():
    backbone = DummyBackbone()
    with pytest.raises(ValueError, match="do not match"):
        apply_transformer_layer_pruning(backbone, {
            "variant": "E41-A", "expected_num_layers": 24,
            "expected_retained_layers": 16, "retained_layer_indices": list(range(8, 24)),
        })


def test_e42_four_billion_masks_remove_eleven_of_32_layers():
    masks = [_e41_layer_indices(32, f"E42-{suffix}") for suffix in "ABC"]
    assert [len(mask) for mask in masks] == [21, 21, 21]
    assert tuple(i for i in range(32) if i not in masks[0]) == tuple(range(21, 32))
    assert tuple(i for i in range(32) if i not in masks[1]) == tuple(range(10, 21))
    assert tuple(i for i in range(32) if i not in masks[2]) == (2, 5, 8, 11, 14, 16, 19, 22, 25, 28, 31)
    for suffix in "ABC":
        backbone = DummyBackbone(32)
        result = apply_transformer_layer_pruning(backbone, {
            "variant": f"E42-{suffix}", "expected_num_layers": 32,
            "expected_retained_layers": 21,
        })
        assert len(backbone.layers) == backbone.config.num_hidden_layers == 21
        assert result["retained_layer_indices"] == list(masks[ord(suffix)-ord("A")])


@pytest.mark.parametrize("path,variant", [
    ("configs/experiments/e42_qwen35_4b_qlora/e42_base.yaml", None),
    ("configs/experiments/e42_qwen35_4b_qlora/e42_a.yaml", "E42-A"),
    ("configs/experiments/e42_qwen35_4b_qlora/e42_b.yaml", "E42-B"),
    ("configs/experiments/e42_qwen35_4b_qlora/e42_c.yaml", "E42-C"),
])
def test_e42_qwen35_4b_configs_load(path, variant):
    cfg = _model_cfg("qwen35_4b", path)
    assert cfg["checkpoint"] == "Qwen/Qwen3.5-4B-Base"
    assert cfg["quantization"] == "qlora4"
    assert cfg["revision"] == "1001bb4d826a52d1f399e183466143f4da7b741b"
    assert cfg["auto_inference_batch_size"] is True
    if variant:
        assert cfg["layer_pruning"]["variant"] == variant
