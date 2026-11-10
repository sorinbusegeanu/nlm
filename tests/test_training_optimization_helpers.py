"""Focused tests for deterministic training cache/calibration helpers (no model training)."""
import json
import inspect
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


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
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

    def test_default_calibration_sample_is_capped_at_five_thousand(self):
        labels = np.repeat(np.asarray(["a", "b", "c", "d"]), 1500)
        first = model_factory._calibration_indices(labels)
        second = model_factory._calibration_indices(labels)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 5000)
        self.assertEqual(set(labels[first]), {"a", "b", "c", "d"})

    @unittest.skipUnless(_torch_available(), "PyTorch grouped inference ordering check")
    def test_length_grouped_inference_restores_original_row_order(self):
        import torch
        class Dataset(torch.utils.data.Dataset):
            lengths = [4, 1, 3]
            def __len__(self): return 3
            def __getitem__(self, index):
                return {"input_ids": torch.full((self.lengths[index],), index + 1),
                        "attention_mask": torch.ones(self.lengths[index], dtype=torch.long)}
        class Model(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                first = input_ids[:, 0].float()
                return SimpleNamespace(logits=torch.stack([first, first + 10], dim=1))
        def collate(rows):
            width = max(len(row["input_ids"]) for row in rows)
            ids = torch.zeros((len(rows), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, row in enumerate(rows):
                size = len(row["input_ids"])
                ids[index, :size] = row["input_ids"]
                mask[index, :size] = row["attention_mask"]
            return {"input_ids": ids, "attention_mask": mask}
        predictions = model_factory._predict_cached_dataset(Model(), Dataset(), batch_size=2,
            data_collator=collate, device=torch.device("cpu"))
        np.testing.assert_array_equal(predictions[:, 0], [1, 2, 3])

    @unittest.skipUnless(_torch_available(), "PyTorch inference OOM fallback check")
    def test_length_grouped_inference_retries_smaller_batches_and_restores_order(self):
        import torch
        class Dataset(torch.utils.data.Dataset):
            lengths = [3, 1, 2]
            def __len__(self): return 3
            def __getitem__(self, index):
                return {"input_ids": torch.full((self.lengths[index],), index + 1),
                        "attention_mask": torch.ones(self.lengths[index], dtype=torch.long)}
        class Model(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                if input_ids.shape[0] > 1:
                    raise RuntimeError("CUDA out of memory in fake model")
                value = input_ids[:, 0].float()
                return SimpleNamespace(logits=torch.stack((value, -value), dim=1))
        def collate(rows):
            width = max(len(row["input_ids"]) for row in rows)
            ids = torch.zeros((len(rows), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, row in enumerate(rows):
                ids[index, :len(row["input_ids"])] = row["input_ids"]
                mask[index, :len(row["attention_mask"])] = row["attention_mask"]
            return {"input_ids": ids, "attention_mask": mask}
        predictions, info = model_factory._predict_cached_dataset(Model(), Dataset(), batch_size=4,
            data_collator=collate, device=torch.device("cpu"), return_info=True)
        np.testing.assert_array_equal(predictions[:, 0], [1, 2, 3])
        self.assertEqual(info["effective_inference_batch_size"], 1)
        self.assertEqual(info["inference_oom_fallbacks"], 2)
        self.assertGreater(info["inference_rows_per_second"], 0)

    def test_temperature_calibration_can_be_disabled_without_fitting(self):
        with patch.object(model_factory, "_fit_temperature") as fit:
            temperature, record = model_factory._temperature_if_enabled(
                np.ones((3, 2)), ["a", "b", "a"], ["a", "b"], enabled=False)
        fit.assert_not_called()
        self.assertEqual(temperature, 1.0)
        self.assertFalse(record["calibration_enabled"])
        self.assertIsNone(record["calibration_sample_size"])

    @unittest.skipUnless(_torch_available(), "PyTorch selected checkpoint restore check")
    def test_ultrabert_cpu_snapshot_restores_exact_weights(self):
        import torch
        model = torch.nn.Linear(3, 2)
        snapshot = model_factory._snapshot_state_dict_cpu(model)
        self.assertTrue(all(value.device.type == "cpu" for value in snapshot.values()))
        expected = {key: value.clone() for key, value in snapshot.items()}
        with torch.no_grad():
            for parameter in model.parameters(): parameter.add_(5)
        model_factory._restore_model_state(model, snapshot)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key])

    def test_catboost_call_signature_has_no_unsupported_epoch_checkpoint_argument(self):
        self.assertNotIn("save_epoch_checkpoints",
                         inspect.signature(model_factory.train_catboost_stylometry).parameters)

    def test_catboost_disabled_temperature_skips_calibration_prediction(self):
        import sys
        class FakeCatBoost:
            calls = []
            def __init__(self, **kwargs): self.kwargs = kwargs
            def fit(self, x, y, **kwargs):
                self.classes_ = np.asarray(["a", "b"])
                self.tree_count_ = 2
                self.fit_kwargs = kwargs
                FakeCatBoost.calls.append(("fit", kwargs))
            def get_best_iteration(self): return 1
            def get_evals_result(self): return {"validation": {"BalancedAccuracy": [0.5, 0.6]}}
            def predict(self, x, prediction_type=None):
                FakeCatBoost.calls.append(("predict", len(x)))
                return np.tile(np.asarray([[1.0, -1.0]]), (len(x), 1))
            def save_model(self, path): Path(path).write_bytes(b"fake-model")
        fake_module = SimpleNamespace(CatBoostClassifier=FakeCatBoost)
        FakeCatBoost.calls = []
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"catboost": fake_module}):
            result = model_factory.train_catboost_stylometry(
                ["train a", "train b"], ["a", "b"], ["valid a", "valid b"], ["a", "b"],
                output_dir=Path(temp), iterations=2, validation_labels=["a", "b"],
                calibration_enabled=False,
                feature_matrices=(np.zeros((2, 1)), np.zeros((2, 1)), ["f"]))
        self.assertEqual([entry for entry in FakeCatBoost.calls if entry[0] == "predict"],
                         [("predict", 2)])
        self.assertEqual(result["calibration_temperature"], 1.0)
        self.assertFalse(result["calibration_enabled"])
        self.assertIsNone(result.get("calibration_sample_size"))
        self.assertTrue(np.allclose(result["probabilities"], result["probabilities"][0]))
        fit_kwargs = next(entry[1] for entry in FakeCatBoost.calls if entry[0] == "fit")
        self.assertIsInstance(result["model"].kwargs["eval_metric"], model_factory._BalancedAccuracyMetric)
        self.assertEqual(fit_kwargs["early_stopping_rounds"], 100)

    def test_post_training_runtime_has_stable_stage_keys(self):
        runtime = pipeline._post_training_runtime({"inference_seconds": 1.25,
            "runtime_breakdown": {"temperature_fit_seconds": 0.04}})
        self.assertEqual(runtime["temperature_fit_seconds"], 0.04)
        self.assertEqual(runtime["final_validation_inference_seconds"], 1.25)
        self.assertEqual(runtime["best_checkpoint_restoration_seconds"], 0.0)
        self.assertEqual(runtime["manifest_prediction_export_seconds"], 0.0)
        self.assertEqual(runtime["total_post_training_seconds"], 0.0)
        self.assertEqual(runtime["inference_oom_fallbacks"], 0)

    def test_lora_qlora_and_continuation_paths_expose_inference_controls(self):
        parameters = inspect.signature(model_factory.train_qwen_lora_fold).parameters
        self.assertIn("inference_batch_size", parameters)
        self.assertIn("calibration_enabled", parameters)
        self.assertIn("quantization", parameters)
        qwen_cfg = pipeline._model_cfg("qwen06")
        self.assertEqual(qwen_cfg["inference_batch_size"], 8)
        self.assertTrue(qwen_cfg["calibration_enabled"])
        catboost_cfg = pipeline._model_cfg("catboost_stylometry")
        self.assertTrue(catboost_cfg["calibration_enabled"])
        encoder_parameters = inspect.signature(model_factory.train_encoder_fold).parameters
        self.assertIn("inference_batch_size", encoder_parameters)
        self.assertIn("calibration_enabled", encoder_parameters)

    def test_epoch_predictions_are_reused_only_for_matching_selected_checkpoint(self):
        epoch_predictions = [{"epoch": 1, "scores": np.ones((2, 3))},
                             {"epoch": 2, "scores": np.full((2, 3), 2.0)}]
        selected = model_factory._selected_epoch_scores(epoch_predictions, 2, (2, 3))
        np.testing.assert_array_equal(selected, np.full((2, 3), 2.0))
        self.assertIsNone(model_factory._selected_epoch_scores(epoch_predictions, 3, (2, 3)))
        self.assertIsNone(model_factory._selected_epoch_scores(epoch_predictions, 2, (3, 3)))

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

    @unittest.skipUnless(_torch_available(), "PyTorch checkpoint restore check")
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

    def test_training_partitions_use_all_rows_and_validate_isolation(self):
        rows_train = pd.DataFrame({"id": ["t1", "t2", "t3", "t4"],
            "prompt_id": ["pt1", "pt2", "pt3", "pt4"],
            "text": ["train one", "train two", "train three", "train four"],
            "model": ["a", "b", "a", "b"], "split": ["train", "validation", "train", "validation"]})
        rows_valid = pd.DataFrame({"id": ["v1", "v2", "v3", "v4"],
            "prompt_id": ["pv1", "pv2", "pv3", "pv4"],
            "text": ["valid one", "valid two", "valid three", "valid four"],
            "model": ["a", "b", "a", "b"]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows_train.to_parquet(root / "dataset_train.parquet", index=False)
            rows_valid.to_parquet(root / "dataset_valid.parquet", index=False)
            train, valid = pipeline._training_partitions("train", "valid", root)
            self.assertEqual(train.id.tolist(), ["t1", "t2", "t3", "t4"])
            self.assertEqual(valid.id.tolist(), ["v1", "v2", "v3", "v4"])
            rows_valid.loc[0, "prompt_id"] = "pt1"
            rows_valid.to_parquet(root / "dataset_valid.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "prompt group"):
                pipeline._training_partitions("train", "valid", root)

    def test_continuation_source_match_accepts_legacy_full_dataset_manifest(self):
        train = pd.DataFrame({"id": ["t1"], "prompt_id": ["pt1"],
                              "text": ["train response"], "model": ["a"]})
        valid = pd.DataFrame({"id": ["v1"], "prompt_id": ["pv1"],
                              "text": ["validation response"], "model": ["a"]})
        ids_hash = __import__("hashlib").sha256(b"v1").hexdigest()
        text_hash = __import__("hashlib").sha256(
            b"v1\tvalidation response").hexdigest()
        legacy = {"dataset": "train", "dataset_sha256": "legacy-source-hash",
                  "validation_dataset_version": "valid", "validation_rows": 1,
                  "full_v0_validation": {"ids_sha256": ids_hash,
                      "normalized_text_sha256": text_hash}}
        self.assertTrue(pipeline._continuation_source_matches(
            legacy, "train", "valid", train, valid, "legacy-source-hash"))
        self.assertFalse(pipeline._continuation_source_matches(
            legacy, "train", "valid", train, valid.assign(text="changed"), "legacy-source-hash"))

    def test_standard_training_passes_full_rows_and_balanced_selection(self):
        train_rows = pd.DataFrame({"id": ["t1", "t2", "t3", "t4"],
            "prompt_id": ["pt1", "pt2", "pt3", "pt4"],
            "text": ["train one", "train two", "train three", "train four"],
            "model": ["a", "b", "a", "b"]})
        valid_rows = pd.DataFrame({"id": ["v1", "v2", "v3", "v4"],
            "prompt_id": ["pv1", "pv2", "pv3", "pv4"],
            "text": ["valid one", "valid two", "valid three", "valid four"],
            "model": ["a", "b", "a", "b"]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"; artifacts = root / "artifacts"
            data.mkdir()
            train_rows.to_parquet(data / "dataset_train.parquet", index=False)
            valid_rows.to_parquet(data / "dataset_valid.parquet", index=False)
            for name in ("train", "valid"):
                (data / f"dataset_profile_{name}.json").write_text(json.dumps({"dataset_sha256": name}))
            checkpoint = artifacts / "runs" / "fake" / "checkpoints" / "model"
            checkpoint.mkdir(parents=True)
            calls = {}
            def fake_train(model, train, valid, classes, cfg, out, seed, fold, cache_dir, **kwargs):
                calls.update(train_rows=len(train), validation_rows=len(valid), metric=kwargs["checkpoint_selection_metric"])
                return {"scores": np.array([[2., 0.], [0., 2.], [2., 0.], [0., 2.]]),
                    "probabilities": np.array([[.9, .1], [.1, .9], [.9, .1], [.1, .9]]),
                    "checkpoint": str(checkpoint), "fit_seconds": 1.0}
            with patch.object(pipeline, "_model_cfg", return_value={"features": {"views": ["char"]}}), \
                    patch.object(pipeline, "_train_fold", side_effect=fake_train):
                manifest = pipeline.train_models("train", "svc", data, artifacts,
                    root / "registry.sqlite", validation_dataset="valid", refresh_prediction=False)
            self.assertEqual(calls, {"train_rows": 4, "validation_rows": 4,
                                     "metric": "balanced_accuracy"})
            self.assertEqual(manifest["training_dataset_rows"], 4)
            self.assertEqual(manifest["validation_dataset_rows"], 4)
            self.assertEqual(manifest["training_dataset"], "train")
            self.assertEqual(manifest["validation_dataset"], "valid")

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
