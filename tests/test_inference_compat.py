import unittest

import numpy as np

from src.inference import predict_svc_bundle
from src.model_factory import train_svc


class HistoricalInferenceCompatibilityTests(unittest.TestCase):
    def test_new_svc_training_uses_raw_score_softmax(self):
        texts = []
        labels = []
        for label, token in (("a", "aardvark"), ("b", "bananas")):
            for index in range(8):
                texts.append(f"{token} shared words sample {index}")
                labels.append(label)
        result = train_svc(texts, labels, ["aardvark validation", "bananas validation"],
                           ["a", "b"], validation_labels=["a", "b"])
        from scipy.special import softmax
        np.testing.assert_allclose(result["probabilities"], softmax(result["scores"], axis=1))
        self.assertIn("validation_loss_raw", result)
        self.assertFalse(any("calibrat" in key for key in result))

    def test_historical_svc_checkpoint_with_predict_proba_remains_loadable(self):
        class Features:
            def transform(self, texts):
                return np.zeros((len(texts), 1))

        class HistoricalCalibratedReadout:
            classes_ = np.asarray(["b", "a"])

            def predict_proba(self, matrix):
                return np.tile(np.asarray([[.8, .2]]), (len(matrix), 1))

        probabilities, classes = predict_svc_bundle({
            "features": Features(), "model": HistoricalCalibratedReadout(),
            "classes": ["a", "b"],
        }, ["legacy checkpoint text"])
        self.assertEqual(classes, ["a", "b"])
        np.testing.assert_allclose(probabilities, [[.2, .8]])


if __name__ == "__main__":
    unittest.main()
