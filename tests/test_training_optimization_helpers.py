"""Focused tests for deterministic training cache/calibration helpers (no model training)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import model_factory
from src import pipeline


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TrainingOptimizationHelperTests(unittest.TestCase):
    def test_calibration_subset_is_deterministic_stratified_and_train_only(self):
        labels = ["a"] * 80 + ["b"] * 15 + ["c"] * 5
        logits = np.random.default_rng(9).normal(size=(len(labels), 3))
        first, report = model_factory._fit_temperature(logits, labels, ["a", "b", "c"],
            max_samples=30, seed=17, return_metrics=True)
        second, _ = model_factory._fit_temperature(logits, labels, ["a", "b", "c"],
            max_samples=30, seed=17, return_metrics=True)
        self.assertEqual(first, second)
        self.assertEqual(report["calibration_sample_size"], 30)
        self.assertTrue(all(np.count_nonzero(np.asarray(labels)[
            model_factory._calibration_indices(labels, max_samples=30, seed=17)] == label) > 0
            for label in ("a", "b", "c")))
        self.assertTrue(np.isfinite([report["calibration_raw_loss"],
                                     report["calibration_calibrated_loss"]]).all())

    def test_token_length_cache_reuses_and_invalidates_on_content_change(self):
        class FakeTokenizer:
            calls = 0
            def __call__(self, batch, **kwargs):
                type(self).calls += 1
                return {"input_ids": [[1] * (len(text.split()) + 2) for text in batch]}
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs): return FakeTokenizer()
        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs): return SimpleNamespace(max_position_embeddings=64)
        fake_transformers = SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoConfig=AutoConfig)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"NLM_TOKEN_LENGTH_CACHE": temp}), \
                patch.object(model_factory, "_force_huggingface_offline"), \
                patch.object(model_factory, "_require_neural", return_value=(None, fake_transformers)):
            first = model_factory.calculate_max_token_length("fake", ["one two", "three"],
                checkpoint="model", revision="rev", task_prefix="task: ")
            calls_after_first = FakeTokenizer.calls
            second = model_factory.calculate_max_token_length("fake", ["one two", "three"],
                checkpoint="model", revision="rev", task_prefix="task: ")
            self.assertEqual(first, second)
            self.assertEqual(FakeTokenizer.calls, calls_after_first)
            changed = model_factory.calculate_max_token_length("fake", ["one two three four"],
                checkpoint="model", revision="rev", task_prefix="task: ")
            self.assertGreater(changed, first)
            self.assertGreater(FakeTokenizer.calls, calls_after_first)

    def test_embedding_cache_rejects_reordered_or_different_source_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = np.arange(6, dtype=np.float32).reshape(3, 2)
            cache = root / "features.npz"
            metadata = root / "features.json"
            np.savez_compressed(cache, features=features)
            metadata.write_text(json.dumps({"cache_key": "key", "row_ids": ["r1", "r2", "r3"]}))
            loaded = model_factory._read_embedding_cache(cache, metadata, "key", ["r1", "r2", "r3"])
            np.testing.assert_array_equal(loaded, features)
            self.assertIsNone(model_factory._read_embedding_cache(cache, metadata, "key", ["r2", "r1", "r3"]))
            self.assertIsNone(model_factory._read_embedding_cache(cache, metadata, "other", ["r1", "r2", "r3"]))

    def test_selected_qwen_checkpoint_restores_adapter_and_classifier_together(self):
        import torch
        backbone = SimpleNamespace(adapter_value=0)
        classifier = torch.nn.Linear(2, 2)
        expected = {name: torch.full_like(value, 3.0) for name, value in classifier.state_dict().items()}
        adapter_state = {"adapter": "selected"}
        def setter(target, state, adapter_name):
            target.adapter_value = (state["adapter"], adapter_name)
        model_factory._restore_qwen_selected_state(backbone, classifier, adapter_state, expected,
                                                    setter=setter)
        self.assertEqual(backbone.adapter_value, ("selected", "default"))
        for name, value in classifier.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))

    def test_continuation_uses_all_external_rows_and_full_v0_validation(self):
        rows = pd.DataFrame({"id": ["ext-2", "ext-1"] + [f"v0-{i}" for i in range(2699)],
            "split": ["train", "train"] + ["validation"] * 2699,
            "prompt_id": ["p2", "p1"] + [f"vp-{i}" for i in range(2699)],
            "text": ["second", "first"] + ["v0 response"] * 2699,
            "model": ["a", "a"] + ["a"] * 2699})
        train, valid = pipeline._full_v0_continuation_rows(rows)
        self.assertEqual(train.id.tolist(), ["ext-1", "ext-2"])
        self.assertEqual(len(train), 2)
        self.assertEqual(len(valid), 2699)
        with self.assertRaisesRegex(ValueError, "2,699"):
            pipeline._full_v0_continuation_rows(rows.iloc[:-1])

    def test_continuation_checkpoint_must_be_full_v0(self):
        with self.assertRaisesRegex(ValueError, "all 2,699"):
            pipeline._require_full_v0_checkpoint({"validation_rows": 540, "validation_scope": "frozen_v0_540"})
        pipeline._require_full_v0_checkpoint({"validation_rows": 2699, "validation_scope": "full_v0"})

    @unittest.skipUnless(_cuda_available(), "CUDA batch probe integrity check")
    def test_batch_probe_restores_model_parameters(self):
        import torch
        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
            def forward(self, input_ids, labels):
                logits = self.linear(input_ids.float())
                return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels))
        model = TinyClassifier().cuda()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        dataset = [{"input_ids": torch.ones(4), "labels": torch.tensor(0)},
                   {"input_ids": torch.zeros(4), "labels": torch.tensor(1)}]
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"NLM_BATCH_SIZE_CACHE": temp}):
            model._batch_probe_model_id = "tiny-test-model"
            model_factory._probe_max_batch_size(model, dataset, requested_batch_size=1,
                max_batch_size=1, safety_fraction=1.0, runtime_overhead_reserve_bytes=0)
            _, cached_report = model_factory._probe_max_batch_size(model, dataset, requested_batch_size=1,
                max_batch_size=1, safety_fraction=1.0, runtime_overhead_reserve_bytes=0)
            self.assertEqual(cached_report["status"], "cached_validated")
            self.assertEqual(cached_report["cache_validation_passes"], 1)
        for original, current in zip(before, model.parameters()):
            self.assertTrue(torch.equal(original, current.detach()))


if __name__ == "__main__":
    unittest.main()
