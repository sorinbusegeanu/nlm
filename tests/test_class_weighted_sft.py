import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import pipeline
from src.dataset_catalog import load_dataset_catalog
from src.model_factory import (
    _rebase_resumed_scheduler_lr,
    classification_loss_with_optional_weights,
    normalize_class_weights,
)


class ClassWeightedSftTests(unittest.TestCase):
    def test_v214cw_maps_exactly_three_labels_to_double_weight(self):
        entry = next(item for item in load_dataset_catalog()["datasets"]
                     if item["id"] == "v2.14CW")
        classes = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
                   "claude-opus-5", "claude-sonnet-5", "gemini-3.1-pro-preview",
                   "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
                   "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna",
                   "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3",
                   "qwen3.8-max"]
        weights = normalize_class_weights(entry["class_weights"], classes)
        self.assertEqual(len(weights), 18)
        self.assertEqual({label for label, value in weights.items() if value == 2.0},
                         {"claude-fable-5", "glm-5.3", "gemini-3.7-flash"})
        self.assertEqual({value for value in weights.values()}, {1.0, 2.0})

    def test_weighted_training_loss_and_unweighted_validation_loss(self):
        import torch
        import torch.nn.functional as functional

        logits = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        labels = torch.tensor([0, 1])
        weights = torch.tensor([2.0, 1.0])
        train_loss = classification_loss_with_optional_weights(
            logits, labels, weights, training=True)
        validation_loss = classification_loss_with_optional_weights(
            logits, labels, weights, training=False)
        self.assertTrue(torch.allclose(train_loss, functional.cross_entropy(
            logits, labels, weight=weights)))
        self.assertTrue(torch.allclose(validation_loss, functional.cross_entropy(
            logits, labels)))
        self.assertFalse(torch.allclose(train_loss, validation_loss))

    def test_disabled_weights_use_unweighted_cross_entropy(self):
        import torch
        import torch.nn.functional as functional

        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        labels = torch.tensor([0, 1])
        self.assertIsNone(normalize_class_weights(None, ["a", "b"]))
        actual = classification_loss_with_optional_weights(
            logits, labels, None, training=True)
        self.assertTrue(torch.equal(actual, functional.cross_entropy(logits, labels)))

    def test_trainer_uses_index_order_and_delegates_evaluation(self):
        from types import SimpleNamespace

        import torch
        import torch.nn.functional as functional

        from src.model_factory import _class_weighted_trainer

        class BaseTrainer:
            def compute_loss(self, model, inputs, **kwargs):
                return functional.cross_entropy(inputs["logits"], inputs["labels"])

        class Model(torch.nn.Module):
            def forward(self, logits):
                return SimpleNamespace(logits=logits)

        label_map = {"a": 1, "b": 0}
        weights = normalize_class_weights({"a": 2.0}, label_map)
        self.assertIs(_class_weighted_trainer(BaseTrainer, label_map, None), BaseTrainer)
        trainer = _class_weighted_trainer(BaseTrainer, label_map, weights)()
        logits = torch.tensor([[2.0, 0.0], [1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        inputs = {"logits": logits, "labels": labels}
        model = Model()
        loss = trainer.compute_loss(model, inputs)
        torch.testing.assert_close(loss, functional.cross_entropy(
            logits, labels, weight=torch.tensor([1.0, 2.0])))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(trainer.model_accepts_loss_kwargs)
        torch.testing.assert_close(trainer.compute_loss(model.eval(), inputs),
                                   functional.cross_entropy(logits, labels))

    def test_invalid_weights_and_explicit_disable(self):
        for invalid in ({"x": 1}, {"a": True}, {"a": "2"}, {"a": 0},
                        {"a": -1}, {"a": float("inf")}, {"a": float("nan")},
                        {"a": 1e100}, {"a": 1e-100}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_class_weights(invalid, ["a"])
        self.assertIsNone(normalize_class_weights({}, ["a"]))
        cfg = pipeline._continuation_config("qwen06", {"class_weights": {"glm-5.3": 9}},
                                            dataset="v2.14CW")
        self.assertEqual(cfg["class_weights"]["glm-5.3"], 2.0)
        cfg = pipeline._continuation_config("qwen06", explicit_overrides={"class_weights": None},
                                            dataset="v2.14CW")
        self.assertIsNone(cfg["class_weights"])

    def test_alias_reads_same_v214_artifact_without_changing_rows(self):
        sample = pd.DataFrame({"id": ["b", "a"], "prompt_id": ["p2", "p1"],
                               "text": ["second", "first"], "model": ["x", "y"]})
        calls = []

        def fake_read_parquet(path):
            calls.append(Path(path))
            return sample.copy()

        with patch.object(pipeline.pd, "read_parquet", side_effect=fake_read_parquet):
            alias_rows = pipeline._dataset_rows("v2.14CW", "/tmp/data")
            source_rows = pipeline._dataset_rows("v2.14", "/tmp/data")
        self.assertEqual(calls[0], calls[1])
        pd.testing.assert_frame_equal(alias_rows, source_rows)
        self.assertEqual(len(alias_rows), len(sample))

    def test_dataset_catalog_weighting_is_default_and_configurable(self):
        cfg = pipeline._apply_dataset_training_options({}, "v2.14CW")
        self.assertEqual(cfg["class_weights"]["glm-5.3"], 2.0)
        explicit = {"class_weights": {"glm-5.3": 3.0}}
        pipeline._apply_dataset_training_options(explicit, "v2.14CW")
        self.assertEqual(explicit["class_weights"]["glm-5.3"], 3.0)

    def test_resumed_lr_rebases_schedule_without_clearing_optimizer_state(self):
        import torch

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5)
        parameter.grad = torch.tensor([0.5])
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / 10))
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        saved_state = optimizer.state[parameter]["exp_avg"].clone()
        last_epoch = scheduler.last_epoch
        info = _rebase_resumed_scheduler_lr(optimizer, scheduler, 1e-5 / 3)
        self.assertEqual(scheduler.last_epoch, last_epoch)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5 / 3)
        self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"], saved_state))
        self.assertAlmostEqual(info["next_step_learning_rate"], 1e-5 / 3)


    def test_queue_selects_cw_without_launching_worker(self):
        import json
        import tempfile
        from types import SimpleNamespace

        import yaml

        from scripts import queue_v214_encoder_models as queue

        spec = queue.SPECS["mpnet_base"]
        config_text = (queue.ROOT / spec["config_path"]).read_text()
        classes = ["claude-fable-5", "glm-5.3", "gemini-3.7-flash"]
        classes += [f"label-{i}" for i in range(15)]
        train = pd.DataFrame({"model": (classes * 12852)[:231336]})
        valid = pd.DataFrame(index=range(2699))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dependency = root / "dependency.json"
            dependency.write_text('{"status":"running","queue_id":"dependency"}')
            config = root / spec["config_path"]
            config.parent.mkdir(parents=True)
            config.write_text(config_text)
            data = root / "data/processed"
            data.mkdir(parents=True)
            (data / "dataset_profile_v2.14.json").write_text('{"dataset_sha256":"base"}')
            snapshot = root / "hub/models--microsoft--mpnet-base/snapshots" / spec["revision"]
            snapshot.mkdir(parents=True)
            for name in spec["required_files"]:
                (snapshot / name).touch()

            with patch.multiple(queue, ROOT=root, QUEUE_ROOT=root / "queues", HF_HOME=root), \
                    patch.object(pipeline, "_training_partitions", return_value=(train, valid)) as partitions, \
                    patch.object(queue.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as worker:
                output = queue.create_queue("mpnet_base", dependency, dataset_override="v2.14CW")
            partitions.assert_called_once_with("v2.14CW", "v0", data)
            worker.assert_called_once()
            job = json.loads((output / "queue.json").read_text())["results"][0]
            command = job["command"]
            self.assertEqual(command[command.index("--dataset") + 1], "v2.14CW")
            self.assertEqual(job["dataset_path"], "data/processed/dataset_v2.14.parquet")
            effective = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            self.assertFalse(effective.get("weighted_sampling", False))
            self.assertEqual(effective["class_weights"], {label: 2.0 for label in classes[:3]})


if __name__ == "__main__":
    unittest.main()
