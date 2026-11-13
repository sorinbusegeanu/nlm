"""Focused tests for training optimization helpers (no model training)."""
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
    def test_batch_search_selects_arbitrary_integer_maximum(self):
        attempts = []
        selected = model_factory._largest_safe_batch_size(
            37, lambda size: attempts.append(size) or size <= 11)
        self.assertEqual(selected, 11)
        self.assertIn(11, attempts)
        self.assertIn(12, attempts)
        self.assertEqual(attempts[-2:], [12, 11])

    def test_batch_probe_retries_one_below_oom_boundary(self):
        attempts = []
        selected = model_factory._largest_safe_batch_size(
            1024, lambda size: attempts.append(size) or size <= 8)
        self.assertEqual(selected, 8)
        self.assertEqual(attempts[-2:], [9, 8])

    def test_configurable_vram_reserve_uses_five_to_ten_percent(self):
        limit_5, reserve_5 = model_factory._batch_probe_memory_limit(100, 1000, .95)
        limit_8, reserve_8 = model_factory._batch_probe_memory_limit(100, 1000, .92)
        self.assertEqual((limit_5, reserve_5), (1050, 50))
        self.assertEqual((limit_8, reserve_8), (1020, 80))
        limit_explicit, reserve_explicit = model_factory._batch_probe_memory_limit(
            100, 1000, .95, explicit_reserve_bytes=120)
        self.assertEqual((limit_explicit, reserve_explicit), (980, 120))

    def test_effective_batch_target_uses_largest_exact_fit_physical_batch(self):
        self.assertEqual(model_factory._resolve_batch_accumulation(16, 1), (16, 1))
        self.assertEqual(model_factory._resolve_batch_accumulation(16, 8, 16), (16, 1))
        self.assertEqual(model_factory._resolve_batch_accumulation(11, 1, 32), (8, 4))

    def test_encoder_gradient_checkpointing_can_be_enabled_and_disabled(self):
        class Model:
            is_gradient_checkpointing = False
            config = SimpleNamespace(use_cache=True)
            def gradient_checkpointing_enable(self):
                self.is_gradient_checkpointing = True
            def gradient_checkpointing_disable(self):
                self.is_gradient_checkpointing = False
        model = Model()
        model_factory._set_gradient_checkpointing(model, True)
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.config.use_cache)
        model_factory._set_gradient_checkpointing(model, False)
        self.assertFalse(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_policy_does_not_disable_for_large_batches(self):
        self.assertTrue(model_factory._gradient_checkpointing_for_batch(True, 16))
        self.assertTrue(model_factory._gradient_checkpointing_for_batch(True, 1024))
        self.assertFalse(model_factory._gradient_checkpointing_for_batch(False, 16))
        backbone = SimpleNamespace(is_gradient_checkpointing=True)
        self.assertIs(model_factory._gradient_checkpointing_target(
            SimpleNamespace(backbone=backbone)), backbone)

    def test_probe_selects_largest_safe_batch_even_when_smaller_is_faster(self):
        selected, metrics, fallback = model_factory._select_probed_batch_size(
            8, [{"batch_size": 4, "fits": True, "examples_per_second": 7.73},
                {"batch_size": 8, "fits": True, "examples_per_second": 7.62}],
            [{"batch_size": 8, "fits": True, "examples_per_second": 7.62}])
        self.assertEqual(selected, 8)
        self.assertEqual(metrics["examples_per_second"], 7.62)
        self.assertFalse(fallback)

    def test_v2_stage2_config_uses_max_physical_batch_without_accumulation_cap(self):
        config = pipeline._read_yaml("configs/models/ultrabert380_v2_stage2.yaml")
        self.assertEqual(config["max_batch_size"], 1024)
        self.assertEqual(config["batch_probe_safety_fraction"], 0.95)
        self.assertEqual(config["gradient_accumulation_steps"], 1)
        self.assertIsNone(config["target_effective_batch_size"])

    def test_optimizer_step_count_uses_physical_batch_and_accumulation(self):
        self.assertEqual(model_factory._optimizer_steps_per_epoch(101, 16, 1), 7)
        self.assertEqual(model_factory._optimizer_steps_per_epoch(101, 8, 2), 7)

    def test_ragged_tokenization_cache_preserves_row_order_and_invalidates_mismatches(self):
        rows = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
                {"input_ids": [4], "attention_mask": [1]}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tokens.npz"
            model_factory._write_ragged_token_cache(path, "content-key", rows)
            loaded = model_factory._read_ragged_token_cache(path, "content-key", 2)
            self.assertEqual([value.tolist() for value in loaded["input_ids"]], [[1, 2, 3], [4]])
            self.assertEqual([value.tolist() for value in loaded["attention_mask"]], [[1, 1, 1], [1]])
            self.assertIsNone(model_factory._read_ragged_token_cache(path, "different-key", 2))
            self.assertIsNone(model_factory._read_ragged_token_cache(path, "content-key", 3))

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

    def test_ultrabert_restores_only_when_selected_epoch_precedes_last_validation(self):
        self.assertFalse(model_factory._should_restore_selected_epoch(5, 5))
        self.assertTrue(model_factory._should_restore_selected_epoch(3, 5))

    def test_catboost_call_signature_has_no_unsupported_epoch_checkpoint_argument(self):
        self.assertNotIn("save_epoch_checkpoints",
                         inspect.signature(model_factory.train_catboost_stylometry).parameters)

    def test_pipeline_forwards_epoch_checkpoints_and_frozen_inference_batch(self):
        rows = pd.DataFrame({"id": ["1", "2"], "prompt_id": ["p1", "p2"],
            "text": ["text one", "text two"], "model": ["a", "b"]})
        encoder_calls, frozen_calls = {}, {}
        def fake_encoder(*args, **kwargs):
            encoder_calls.update(kwargs)
            return {"scores": np.zeros((2, 2)), "probabilities": np.full((2, 2), .5)}
        def fake_frozen(*args, **kwargs):
            frozen_calls.update(kwargs)
            return {"scores": np.zeros((2, 2)), "probabilities": np.full((2, 2), .5)}
        with patch.object(pipeline, "train_encoder_fold", side_effect=fake_encoder), \
             patch.object(pipeline, "train_frozen_backbone_fold", side_effect=fake_frozen):
            pipeline._train_fold("ultrabert380", rows, rows, ["a", "b"],
                {"checkpoint": "base", "revision": "a" * 40, "max_length": 8,
                 "save_epoch_checkpoints": True, "gradient_checkpointing": True},
                Path("/tmp/encoder"), 1, "validation", Path("/tmp/cache"))
            pipeline._train_fold("embeddinggemma2", rows, rows, ["a", "b"],
                {"training_mode": "frozen", "checkpoint": "base", "revision": "b" * 40,
                 "max_length": 8, "batch_size": 2, "inference_batch_size": 11},
                Path("/tmp/frozen"), 1, "validation", Path("/tmp/cache"))
        self.assertTrue(encoder_calls["save_epoch_checkpoints"])
        self.assertTrue(encoder_calls["gradient_checkpointing"])
        self.assertEqual(frozen_calls["inference_batch_size"], 11)

    def test_catboost_uses_standard_softmax_without_extra_training_inference(self):
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
                feature_matrices=(np.zeros((2, 1)), np.zeros((2, 1)), ["f"]))
            checkpoint_metadata = json.loads((Path(temp) / "stylometry.json").read_text())
        self.assertEqual([entry for entry in FakeCatBoost.calls if entry[0] == "predict"],
                         [("predict", 2)])
        expected = __import__("scipy.special", fromlist=["softmax"]).softmax(result["scores"], axis=1)
        np.testing.assert_allclose(result["probabilities"], expected)
        self.assertFalse(any("calibrat" in key for key in result))
        self.assertFalse(any("calibrat" in key for key in checkpoint_metadata))
        fit_kwargs = next(entry[1] for entry in FakeCatBoost.calls if entry[0] == "fit")
        self.assertIsInstance(result["model"].kwargs["eval_metric"], model_factory._BalancedAccuracyMetric)
        self.assertEqual(fit_kwargs["early_stopping_rounds"], 100)

    def test_standard_catboost_pipeline_uses_balanced_early_stopping_and_valid_kwargs(self):
        class Matrix:
            def __init__(self, value): self.value = np.asarray(value)
            def toarray(self): return self.value
        class Transform:
            stylometry_feature_names = ["f"]
        train = pd.DataFrame({"id": ["t1", "t2"], "prompt_id": ["p1", "p2"],
                              "text": ["train a", "train b"], "model": ["a", "b"]})
        valid = pd.DataFrame({"id": ["v1", "v2"], "prompt_id": ["q1", "q2"],
                              "text": ["valid a", "valid b"], "model": ["a", "b"]})
        calls = {}
        def fake_train(*args, **kwargs):
            calls.update(kwargs)
            return {"scores": np.asarray([[1., 0.], [0., 1.]]),
                    "probabilities": np.asarray([[.8, .2], [.2, .8]]),
                    "checkpoint": "fake", "fit_seconds": 0.1}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(pipeline, "load_or_fit", return_value=(Matrix([[1], [0]]),
                    Matrix([[1], [0]]), Transform(), "cache")), \
                patch.object(pipeline, "train_catboost_stylometry", side_effect=fake_train):
            pipeline._train_fold("catboost_stylometry", train, valid, ["a", "b"],
                {"feature_recipe": {"baseline": [], "groups": []}, "iterations": 3,
                 "early_stopping_rounds": 7},
                Path(temp), 2026, "validation", Path(temp) / "features")
        self.assertNotIn("save_epoch_checkpoints", calls)
        self.assertEqual(calls["checkpoint_selection_metric"], "balanced_accuracy")
        self.assertEqual(calls["early_stopping_rounds"], 7)

    def test_post_training_runtime_has_stable_noncalibration_stage_keys(self):
        runtime = pipeline._post_training_runtime({"inference_seconds": 1.25,
            "runtime_breakdown": {"final_validation_inference_seconds": 1.25}})
        self.assertNotIn("temperature_fit_seconds", runtime)
        self.assertNotIn("calibration_inference_seconds", runtime)
        self.assertEqual(runtime["final_validation_inference_seconds"], 1.25)
        self.assertEqual(runtime["best_checkpoint_restoration_seconds"], 0.0)
        self.assertEqual(runtime["manifest_prediction_export_seconds"], 0.0)
        self.assertEqual(runtime["total_post_training_seconds"], 0.0)
        self.assertEqual(runtime["inference_oom_fallbacks"], 0)

    def test_lora_qlora_and_continuation_paths_expose_inference_controls_without_calibration(self):
        parameters = inspect.signature(model_factory.train_qwen_lora_fold).parameters
        self.assertIn("inference_batch_size", parameters)
        self.assertFalse(any("calibrat" in name for name in parameters))
        self.assertIn("quantization", parameters)
        qwen_cfg = pipeline._model_cfg("qwen06")
        self.assertEqual(qwen_cfg["inference_batch_size"], 8)
        self.assertFalse(any("calibrat" in key for key in qwen_cfg))
        catboost_cfg = pipeline._model_cfg("catboost_stylometry")
        self.assertFalse(any("calibrat" in key for key in catboost_cfg))
        encoder_parameters = inspect.signature(model_factory.train_encoder_fold).parameters
        self.assertIn("inference_batch_size", encoder_parameters)
        self.assertFalse(any("calibrat" in name for name in encoder_parameters))
        self.assertIn("gradient_checkpointing", encoder_parameters)

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

    def test_cross_dataset_continuation_requires_identical_v0_validation(self):
        train = pd.DataFrame({"id": ["t1"], "prompt_id": ["pt1"],
                              "text": ["stage two row"], "model": ["a"]})
        valid = pd.DataFrame({"id": ["v1"], "prompt_id": ["pv1"],
                              "text": ["validation response"], "model": ["a"]})
        import hashlib
        source = {"training_dataset": "v2.0.chunks800", "validation_dataset": "v0",
            "training_dataset_sha256": "stage-one-data-hash",
            "validation_dataset_sha256": pipeline._partition_sha256(valid),
            "validation_rows": 1,
            "validation_ids_sha256": hashlib.sha256(b"v1").hexdigest(),
            "validation_prompt_ids_sha256": hashlib.sha256(b"pv1").hexdigest(),
            "validation_normalized_text_sha256": hashlib.sha256(
                b"validation response").hexdigest()}
        self.assertFalse(pipeline._continuation_source_matches(
            source, "v2.0", "v0", train, valid, "stage-two-data-hash"))
        self.assertTrue(pipeline._continuation_source_matches(
            source, "v2.0", "v0", train, valid, "stage-two-data-hash",
            allow_cross_dataset_source=True))
        self.assertFalse(pipeline._continuation_source_matches(
            source, "v2.0", "v0", train, valid.assign(prompt_id="different-prompt"),
            "stage-two-data-hash", allow_cross_dataset_source=True))
        self.assertFalse(pipeline._continuation_source_matches(
            source, "v2.0", "v0", train, valid.assign(text="changed"),
            "stage-two-data-hash", allow_cross_dataset_source=True))

    def test_continuation_paths_forward_inference_and_calibration_settings(self):
        train = pd.DataFrame({"id": ["t1", "t2"], "prompt_id": ["pt1", "pt2"],
            "text": ["train a", "train b"], "model": ["a", "b"]})
        valid = pd.DataFrame({"id": ["v1", "v2"], "prompt_id": ["pv1", "pv2"],
            "text": ["valid a", "valid b"], "model": ["a", "b"]})
        scores = np.asarray([[2., 0.], [0., 2.]])
        probabilities = np.asarray([[.9, .1], [.1, .9]])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); artifacts = root / "artifacts"; predictions = artifacts / "predictions"
            predictions.mkdir(parents=True)
            data = root / "data"; data.mkdir()
            profile = data / "dataset_profile_v1.json"
            profile.write_text(json.dumps({"dataset_sha256": "train-hash"}))
            calls = {}
            selected_checkpoint = root / "selected"
            selected_checkpoint.mkdir()
            def fake_result(**extra):
                return {"scores": scores, "probabilities": probabilities,
                    "fit_seconds": .1, "inference_seconds": 0.,
                    "validation_metrics_by_epoch": [],
                    "validation_predictions_by_epoch": [], "best_epoch": 1,
                    "best_validation_accuracy": 1., "best_validation_balanced_accuracy": 1.,
                    "checkpoint_selection_metric": "validation_balanced_accuracy",
                    "checkpoint_scoring_method": "test", "early_stopping": None,
                    "best_checkpoint_path": str(selected_checkpoint),
                    "baseline_validation_metrics": {"accuracy": 1.},
                    "requested_batch_size": 1, "effective_batch_size": 1,
                    "requested_inference_batch_size": 7, "effective_inference_batch_size": 7,
                    "runtime_breakdown": {"effective_inference_batch_size": 7}, **extra}
            qwen_checkpoint = artifacts / "runs" / "qwen-source" / "checkpoint"
            qwen_checkpoint.mkdir(parents=True)
            qwen_source = {"experiment_id": "qwen-source", "model": "qwen06", "dataset": "v1",
                "checkpoint_path": str(qwen_checkpoint), "best_checkpoint_path": str(qwen_checkpoint),
                "checkpoint_id": "Qwen/Qwen3-0.6B", "model_revision": "rev",
                "classes": ["a", "b"], "config": {"batch_size": 1, "epochs": 1,
                    "quantization": "none", "calibration_enabled": False, "inference_batch_size": 7,
                    "auto_batch_size": True, "max_batch_size": 32,
                    "gradient_checkpointing": True},
                "created_at_utc": "2026-01-01"}
            (predictions / "qwen06_v1_qwen-source_manifest.json").write_text(json.dumps(qwen_source))
            encoder_checkpoint = artifacts / "runs" / "encoder-source" / "checkpoint"
            encoder_checkpoint.mkdir(parents=True)
            encoder_source = {"experiment_id": "encoder-source", "model": "ultrabert380", "dataset": "v1",
                "checkpoint_path": str(encoder_checkpoint), "best_checkpoint_path": str(encoder_checkpoint),
                "checkpoint_id": "ultrabert", "model_revision": "rev", "classes": ["a", "b"],
                "config": {"batch_size": 2, "inference_batch_size": 6,
                    "calibration_enabled": False, "epochs": 1,
                    "auto_batch_size": True, "max_batch_size": 64,
                    "batch_probe_safety_fraction": 0.8}, "created_at_utc": "2026-01-01"}
            (predictions / "ultrabert380_v1_encoder-source_manifest.json").write_text(json.dumps(encoder_source))
            (predictions / "encoder-source_manifest.json").write_text(json.dumps(encoder_source))

            def fake_qwen(*args, **kwargs):
                calls["qwen"] = kwargs
                return fake_result()
            def fake_encoder(*args, **kwargs):
                calls["encoder"] = kwargs
                Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
                return fake_result()
            with patch.object(pipeline, "_training_partitions", return_value=(train, valid)), \
                    patch.object(pipeline, "_profile_path", return_value=profile), \
                    patch.object(pipeline, "_continuation_source_matches", return_value=True), \
                    patch.object(pipeline, "_dataset_max_length", return_value=16), \
                    patch("src.model_factory.train_qwen_lora_fold", side_effect=fake_qwen), \
                    patch.object(pipeline, "train_encoder_fold", side_effect=fake_encoder), \
                    patch.object(pipeline, "Registry"):
                pipeline.continue_qwen("v1", data_dir=str(data), artifact_dir=str(artifacts),
                    registry_path=str(root / "registry.sqlite"), from_experiment="qwen-source",
                    model="qwen06", epochs=1,
                    config_overrides={"batch_size": 3, "inference_batch_size": 5,
                                      "calibration_enabled": True})
                pipeline.continue_encoder("v1", data_dir=str(data), artifact_dir=str(artifacts),
                    registry_path=str(root / "registry.sqlite"), from_experiment="encoder-source",
                    epochs=1, config_overrides={"batch_size": 4, "inference_batch_size": 8,
                                                "calibration_enabled": True},
                    allow_cross_dataset_source=True)
        self.assertEqual(calls["qwen"]["inference_batch_size"], 5)
        self.assertEqual(calls["qwen"]["batch_size"], 3)
        self.assertNotIn("calibration_enabled", calls["qwen"])
        self.assertTrue(calls["qwen"]["auto_batch_size"])
        self.assertTrue(calls["qwen"]["gradient_checkpointing"])
        self.assertEqual(calls["encoder"]["inference_batch_size"], 8)
        self.assertEqual(calls["encoder"]["batch_size"], 4)
        self.assertNotIn("calibration_enabled", calls["encoder"])
        self.assertTrue(calls["encoder"]["auto_batch_size"])
        self.assertEqual(calls["encoder"]["max_batch_size"], 64)
        self.assertEqual(calls["encoder"]["batch_probe_safety_fraction"], 0.8)
        self.assertTrue(calls["encoder"]["gradient_checkpointing"])
        self.assertEqual(calls["encoder"]["lr_scheduler_type"], "linear")
        self.assertEqual(calls["encoder"]["warmup_ratio"], 0.0)

    def test_continuation_config_precedence_parent_then_yaml_then_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "custom.yaml"
            config.write_text("model: qwen06\nlearning_rate: 0.0002\nbatch_size: 3\n")
            cfg = pipeline._continuation_config("qwen06",
                {"learning_rate": 0.0001, "batch_size": 2, "inference_batch_size": 7,
                 "auto_batch_size": True}, config, {"learning_rate": 0.0003})
        self.assertEqual(cfg["learning_rate"], 0.0003)
        self.assertEqual(cfg["batch_size"], 3)
        self.assertEqual(cfg["inference_batch_size"], 7)
        self.assertTrue(cfg["auto_batch_size"])

    @unittest.skipUnless(_torch_available(), "PyTorch simulated evaluation OOM")
    def test_epoch_evaluation_oom_retries_only_eval_batch_size(self):
        import torch
        current = {"eval": 8, "train": 6, "accumulation": 3}
        attempts = []
        def evaluate():
            attempts.append(current["eval"])
            if current["eval"] > 2:
                raise RuntimeError("CUDA out of memory")
            return {"eval_balanced_accuracy": .5}
        metrics, effective, retries = model_factory._evaluate_with_oom_batch_fallback(
            evaluate, get_batch_size=lambda: current["eval"],
            set_batch_size=lambda value: current.__setitem__("eval", value),
            empty_cache=lambda: None)
        self.assertEqual(attempts, [8, 4, 2])
        self.assertEqual(metrics["eval_balanced_accuracy"], .5)
        self.assertEqual((effective, retries), (2, 2))
        self.assertEqual((current["train"], current["accumulation"]), (6, 3))

    @unittest.skipUnless(_torch_available(), "PyTorch late-pass inference OOM")
    def test_cached_inference_keeps_completed_chunks_when_late_batch_ooms(self):
        import torch
        class Dataset(torch.utils.data.Dataset):
            lengths = [5, 1, 2, 3, 4, 6]
            def __len__(self): return len(self.lengths)
            def __getitem__(self, index):
                return {"input_ids": torch.tensor([index + 1] * self.lengths[index]),
                        "attention_mask": torch.ones(self.lengths[index], dtype=torch.long)}
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.scored = []
            def forward(self, input_ids, attention_mask):
                values = input_ids[:, 0].tolist()
                if len(values) > 1 and max(values) >= 4:
                    raise RuntimeError("CUDA out of memory in late batch")
                self.scored.extend(values)
                value = input_ids[:, 0].float()
                return SimpleNamespace(logits=torch.stack((value, -value), dim=1))
        def collate(rows):
            ids = torch.nn.utils.rnn.pad_sequence([row["input_ids"] for row in rows], batch_first=True)
            mask = torch.nn.utils.rnn.pad_sequence([row["attention_mask"] for row in rows], batch_first=True)
            return {"input_ids": ids, "attention_mask": mask}
        model = Model()
        scores, _ = model_factory._predict_cached_dataset(model, Dataset(), batch_size=2,
            data_collator=collate, device=torch.device("cpu"), return_info=True)
        np.testing.assert_array_equal(scores[:, 0], np.arange(1, 7))
        self.assertEqual(sorted(model.scored), list(range(1, 7)))
        self.assertEqual(len(model.scored), 6)

    def test_selected_epoch_prediction_links_to_existing_selected_checkpoint(self):
        valid = pd.DataFrame({"id": ["v1", "v2"], "prompt_id": ["p1", "p2"],
                              "model": ["a", "b"]})
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "selected"
            checkpoint.mkdir()
            result = {"validation_predictions_by_epoch": [{"epoch": 1,
                        "scores": np.asarray([[2., 0.], [0., 2.]])}],
                      "best_epoch": 1, "selected_checkpoint_path": str(checkpoint)}
            records = pipeline._save_epoch_validation_predictions(result, valid, ["a", "b"],
                Path(temp), "experiment")
            self.assertEqual(records[0]["checkpoint_path"], str(checkpoint))
            self.assertTrue(Path(records[0]["checkpoint_path"]).is_dir())

    def test_registry_failure_retry_repairs_once_from_validated_artifacts(self):
        from src.registry import Registry
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prediction = root / "predictions.parquet"
            prediction.write_bytes(b"validated predictions")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            manifest = {"experiment_id": "run-1", "model": "svc", "split": "validation",
                "prediction_path": str(prediction),
                "prediction_sha256": __import__("hashlib").sha256(prediction.read_bytes()).hexdigest(),
                "checkpoint_path": str(checkpoint), "config": {"C": 1.},
                "validation_metrics": {"balanced_accuracy": .5}, "dataset_sha256": "data",
                "split_sha256": "split", "code_revision": "abc", "config_sha256": "cfg",
                "fit_seconds": .1, "seed": 1}
            registry_path = root / "registry.sqlite"
            with patch.object(pipeline, "Registry", side_effect=RuntimeError("simulated insert failure")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    pipeline._register_completed_manifest(manifest, registry_path)
            pipeline._register_completed_manifest(manifest, registry_path)
            pipeline._register_completed_manifest(manifest, registry_path)
            rows = Registry(registry_path).rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], "run-1")

    def test_qwen_large_probe_ceiling_uses_one_configured_checkpointing_probe(self):
        calls = []
        def fake_probe(model, dataset, **kwargs):
            calls.append(kwargs)
            return 3, {"enabled": True, "max_batch_size": kwargs["max_batch_size"],
                       "selected_batch_size": 3}
        with patch.object(model_factory, "_probe_max_batch_size", side_effect=fake_probe):
            selected, report = model_factory._probe_lora_batch_size(
                object(), [1] * 64, requested_batch_size=16, max_batch_size=1024,
                safety_fraction=.9, use_bf16=True, use_fp16=False,
                optimizer_name="adamw_torch", weight_decay=0.0,
                gradient_checkpointing_enabled=True)
        self.assertEqual(selected, 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_batch_size"], 1024)
        self.assertTrue(report["gradient_checkpointing_probe_mode"])

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
