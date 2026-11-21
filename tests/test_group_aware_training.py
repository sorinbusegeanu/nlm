from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.model_factory import (GroupAwareBatchSampler, _make_group_training_collator,
                               group_aware_loss, group_ranking_loss)
from src.pipeline import _validate_training_overrides


class GroupAwareTrainingTests(unittest.TestCase):
    def test_group_ranking_loss_matches_per_label_response_softmax(self):
        logits = torch.tensor([[3.0, 0.0, -1.0], [0.0, 2.0, -1.0],
                               [1.0, 0.0, 2.0]], requires_grad=True)
        labels = torch.tensor([0, 1, 2])
        groups = torch.tensor([7, 7, 7])
        expected = F.cross_entropy(logits[:, [0, 1, 2]].T, torch.tensor([0, 1, 2]))
        actual = group_ranking_loss(logits, labels, groups)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_loss_lambda_and_zero_lambda_backward_compatibility(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        rank_groups = torch.tensor([1, 1, -1])
        total, ce, rank = group_aware_loss(logits, labels, rank_groups, group_ranking_lambda=0.0)
        self.assertTrue(torch.allclose(total, F.cross_entropy(logits, labels)))
        self.assertTrue(torch.allclose(ce, total))
        self.assertGreater(float(rank.detach()), 0.0)
        weighted_total, _, weighted_rank = group_aware_loss(
            logits, labels, rank_groups, group_ranking_lambda=0.3)
        self.assertTrue(torch.allclose(weighted_total, ce + .3 * weighted_rank))

    def test_group_sampler_exposes_every_response_once_and_rotates_duplicates(self):
        groups = ["g", "g", "g", "g", "singleton"]
        labels = ["a", "b", "a", "b", "c"]
        rotations = [0, 0, 1, 1, 0]
        sampler0 = GroupAwareBatchSampler(groups, labels, rotations, batch_size=2, seed=44)
        batches0 = list(iter(sampler0))
        self.assertEqual(sorted(index for batch in batches0 for index in batch), list(range(5)))
        selected0 = {index for index, group_id in enumerate(sampler0.rank_group_by_index) if group_id >= 0}
        self.assertEqual(len(selected0), 2)
        self.assertEqual(len({sampler0.rank_group_by_index[i] for i in selected0}), 1)
        list(iter(sampler0))
        selected1 = {index for index, group_id in enumerate(sampler0.rank_group_by_index) if group_id >= 0}
        self.assertEqual(selected1, {2, 3})
        self.assertEqual(sampler0.epoch_records[0]["batch_count"],
                         sampler0.epoch_records[1]["batch_count"])
        sampler1_repeat = GroupAwareBatchSampler(groups, labels, rotations, batch_size=2, seed=44,
                                                 initial_epoch=1)
        sampler1_fresh = GroupAwareBatchSampler(groups, labels, rotations, batch_size=2, seed=44,
                                                initial_epoch=1)
        self.assertEqual(list(iter(sampler1_repeat)), list(iter(sampler1_fresh)))

    def test_atomic_rank_units_can_exceed_batch_cap_without_splitting(self):
        sampler = GroupAwareBatchSampler(["g"] * 4, ["a", "b", "c", "d"],
                                         [0, 0, 0, 0], batch_size=2)
        batches = list(iter(sampler))
        rank_batch = next(batch for batch in batches if any(sampler.rank_group_by_index[i] >= 0 for i in batch))
        self.assertEqual(len(rank_batch), 4)
        self.assertEqual(len(set(sampler.rank_group_by_index[i] for i in rank_batch)), 1)

    def test_training_collator_maps_row_indices_to_current_rank_groups(self):
        sampler = GroupAwareBatchSampler(["g", "g", "h"], ["a", "b", "c"],
                                         [0, 0, 0], batch_size=2)
        list(iter(sampler))
        collator = _make_group_training_collator(
            lambda rows: {"input_ids": torch.stack([row["input_ids"] for row in rows])}, sampler)
        batch = collator([
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor(0), "row_index": torch.tensor(0)},
            {"input_ids": torch.tensor([3, 4]), "labels": torch.tensor(1), "row_index": torch.tensor(1)},
        ])
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertEqual(batch["rank_group_ids"].tolist(), [sampler.rank_group_by_index[0],
                                                             sampler.rank_group_by_index[1]])
        self.assertGreaterEqual(batch["rank_group_ids"][0].item(), 0)

    def test_training_collator_uses_plain_collation_for_validation_rows(self):
        sampler = GroupAwareBatchSampler(["g", "g"], ["a", "b"], [0, 0], batch_size=2)
        collator = _make_group_training_collator(
            lambda rows: {"labels": torch.stack([row["labels"] for row in rows]),
                          "input_ids": torch.stack([row["input_ids"] for row in rows])}, sampler)
        rows = [{"input_ids": torch.tensor([1]), "labels": torch.tensor(0)},
                {"input_ids": torch.tensor([2]), "labels": torch.tensor(1)}]
        batch = collator(rows)
        self.assertNotIn("rank_group_ids", batch)
        self.assertEqual(batch["labels"].tolist(), [0, 1])

    def test_invalid_lambda_requires_group_sampling(self):
        with self.assertRaisesRegex(ValueError, "requires group_aware_sampling"):
            _validate_training_overrides("ultrabert380", {"group_ranking_lambda": .1})
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            _validate_training_overrides("ultrabert380", {
                "group_aware_sampling": True, "group_ranking_lambda": float("nan")})


if __name__ == "__main__":
    unittest.main()
