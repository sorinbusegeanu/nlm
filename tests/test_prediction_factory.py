import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ensemble_factory import fit_weights
from src.inference import predict_prediction_bundle
from src.prediction_factory import _align_candidates, _forward_select


def candidate(name, probabilities, classes=("a", "b")):
    return {"candidate_id": name, "classes": list(classes),
            "probabilities": np.asarray(probabilities, dtype=float)}


class ForwardSelectionTests(unittest.TestCase):
    def test_complementary_candidate_added_and_redundant_candidate_rejected(self):
        y = np.asarray(["a", "a", "a", "b", "b", "b"])
        a = [[.9,.1],[.9,.1],[.1,.9],[.1,.9],[.9,.1],[.1,.9]]
        b = [[.4,.6],[.4,.6],[.99,.01],[.6,.4],[.01,.99],[.4,.6]]
        c = a
        selected, probs, steps, individual = _forward_select(
            [candidate("a", a), candidate("b", b), candidate("c", c)], y, ["a", "b"])
        self.assertEqual([item["candidate_id"] for item in selected], ["a", "b"])
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["accuracy"], 4 / 6)
        self.assertEqual(steps[1]["accuracy"], 1.0)
        self.assertEqual(len(individual), 3)
        self.assertEqual(np.asarray(["a", "b"])[probs.argmax(1)].tolist(), y.tolist())

    def test_equal_accuracy_lower_log_loss_can_extend_selection(self):
        y = np.asarray(["a"] * 10 + ["b"] * 10)
        baseline_probs = np.asarray([[.6,.4]] * 10 + [[.4,.6]] * 10)
        mixed_probs = np.asarray([[.999,.001]] + [[.564,.436]] * 9 +
                                 [[.001,.999]] + [[.436,.564]] * 9)
        baseline = candidate("baseline", baseline_probs)
        mixed = candidate("mixed", mixed_probs)
        selected, _, steps, _ = _forward_select([baseline, mixed], y, ["a", "b"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(steps[-1]["accuracy"], steps[0]["accuracy"])
        self.assertLess(steps[-1]["log_loss"], steps[0]["log_loss"])

    def test_deterministic_tie_and_scaling(self):
        y = np.asarray(["a", "b"] * 4)
        probs = [[.8,.2],[.2,.8]] * 4
        roster = [candidate(f"m{i:03d}", probs) for i in range(32)]
        for _ in range(2):
            selected, _, steps, _ = _forward_select(roster, y, ["a", "b"])
            self.assertEqual([item["candidate_id"] for item in selected], ["m000"])
            self.assertEqual(len(steps), 1)

    def test_weight_constraints_and_class_alignment(self):
        y = np.asarray(["a", "b", "a", "b"])
        probs = [np.asarray([[.9,.1],[.8,.2],[.2,.8],[.1,.9]]),
                 np.asarray([[.8,.2],[.1,.9],[.7,.3],[.2,.8]])]
        weights = fit_weights(probs, y, ["a", "b"])
        self.assertTrue(np.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        frame = pd.DataFrame({"id":["1","2"],"prompt_id":["p1","p2"],"y_true":["a","b"]})
        aligned_probs = [np.asarray([[.8,.2],[.1,.9]]), np.asarray([[.7,.3],[.2,.8]])]
        items = [dict(candidate("x", aligned_probs[0]), dataset="v0", frame=frame),
                 dict(candidate("z", aligned_probs[1][::-1]), dataset="v1", frame=frame.iloc[::-1].reset_index(drop=True))]
        # The second candidate is deliberately reordered and must align to reference IDs.
        items[1]["probabilities"] = aligned_probs[1][::-1]
        classes, labels, _, aligned, excluded, ids = _align_candidates(items, reference_dataset="v0")
        self.assertEqual(classes, ["a", "b"])
        self.assertEqual(labels.tolist(), ["a", "b"])
        self.assertEqual(ids, ["1", "2"])
        self.assertFalse(excluded)
        np.testing.assert_allclose(aligned[1]["probabilities"], aligned_probs[1])


class WeightedBundleTests(unittest.TestCase):
    def test_weighted_inference_matches_saved_probability_calculation(self):
        pa = np.asarray([[.9,.1],[.6,.4],[.2,.8]])
        pb = np.asarray([[.1,.9],[.9,.1],[.7,.3]])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "prediction_bundle.json").write_text(
                '{"classes":["a","b"],"selected_strategy":"optimized_probability_blend",'
                '"selected_members":["a","b"],"selected_weights":{"a":0.75,"b":0.25},'
                '"required_candidates":["a","b"],"candidate_models":[{"candidate_id":"a","model":"fake"},{"candidate_id":"b","model":"fake"}]}')
            with patch("src.inference.predict_candidate", side_effect=[pa, pb]):
                pred, classes = predict_prediction_bundle(root, ["x", "y", "z"])
            expected = np.einsum("m,mnc->nc", [.75,.25], np.stack([pa,pb]))
            self.assertEqual(classes, ["a", "b"])
            self.assertEqual(pred.tolist(), np.asarray(classes)[expected.argmax(1)].tolist())

    def test_equal_and_historical_reliability_bundle_compatibility(self):
        probs = np.asarray([[.9,.1],[.2,.8]])
        for strategy in ("equal_probability_blend", "per_response_reliability"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                bundle = {"classes":["a","b"],"selected_strategy":strategy,
                    "selected_members":["m"],"required_candidates":["m"],
                    "candidate_models":[{"candidate_id":"m","model":"fake"}],"trust_table":[],
                    "minimum_support":1,"fallback_candidate":"m"}
                if strategy == "per_response_reliability":
                    bundle["trust_table"] = [
                        {"candidate_id":"m","predicted_nlm":"a","predicted_count":1,"smoothed_trust":.8},
                        {"candidate_id":"m","predicted_nlm":"b","predicted_count":1,"smoothed_trust":.8}]
                (Path(td) / "prediction_bundle.json").write_text(__import__("json").dumps(bundle))
                with patch("src.inference.predict_candidate", return_value=probs):
                    pred, _ = predict_prediction_bundle(root, ["x", "y"])
                self.assertEqual(pred.tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
