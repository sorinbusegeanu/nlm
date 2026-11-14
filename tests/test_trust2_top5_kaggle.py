import numpy as np
import pandas as pd
import pytest

from src.trust2_top5_kaggle import (
    _submission_frame,
    _trust5_scores,
    _validate_probabilities,
)
from src.v2_checkpoint_trust import LABELS


def test_trust5_family_normalization_and_unvoted_fallback():
    labels = list(LABELS)
    candidates = ["N1", "N2"]
    predictions = {
        "N1": np.array([labels[0], labels[1]], dtype=object),
        "N2": np.array([labels[1], labels[0]], dtype=object),
    }
    weights = {("N1", labels[0]): 0.6, ("N2", labels[1]): 0.8}
    scores, output = _trust5_scores(
        candidates, pd.DataFrame(), predictions, weights,
        {"N1": "same-family", "N2": "same-family"}, "N1")

    # The first row's larger family-normalized trust vote wins. The second row
    # has no eligible vote and uses the explicitly supplied checkpoint fallback.
    assert output.tolist() == [labels[1], labels[1]]
    assert np.isclose(scores[0, 0], 0.3)
    assert np.isclose(scores[0, 1], 0.4)
    assert scores[1].sum() == 0


def test_submission_has_exact_schema_and_preserves_ids():
    ids = pd.Series(["x2", "x1"], dtype="string")
    frame = _submission_frame(ids, np.array(["claude-opus-4-8", "gpt-5.5"]))
    assert frame.columns.tolist() == ["id", "model"]
    assert frame.id.tolist() == ["x2", "x1"]


def test_submission_rejects_unknown_label():
    with pytest.raises(ValueError, match="outside the canonical"):
        _submission_frame(pd.Series(["x"]), np.array(["not-a-competition-label"]))


def test_probability_validation_requires_normalized_full_matrix():
    _validate_probabilities(np.full((2, 18), 1 / 18), rows=2, classes=18)
    with pytest.raises(ValueError, match="not normalized"):
        _validate_probabilities(np.full((2, 18), 0.1), rows=2, classes=18)
