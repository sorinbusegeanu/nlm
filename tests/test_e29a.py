import numpy as np
import pytest

from src.e29a import decode_scenario
from src.v2_checkpoint_trust import LABELS


def test_hungarian_is_recomputed_within_each_scenario():
    scores = np.zeros((1, len(LABELS)), dtype=float)
    scores[0, 0] = 2.0
    one = decode_scenario(scores, ["r1"], ["same-prompt"], ["fold1::frozen"])
    two = decode_scenario(scores, ["r2"], ["same-prompt"], ["fold1::frozen"])
    # If rows from separate scenarios were incorrectly combined, one would be
    # forced away from label 0 by the unique-label assignment constraint.
    assert one["true_group_hungarian"].tolist() == [0]
    assert two["true_group_hungarian"].tolist() == [0]
    assert one["frozen_E19D_subset_hungarian"].tolist() == [0]
    assert two["frozen_E19D_subset_hungarian"].tolist() == [0]


def test_hungarian_recomputed_from_scenario_score_matrix():
    scores = np.zeros((2, len(LABELS)), dtype=float)
    scores[:, 0] = [10.0, 9.0]
    scores[:, 1] = [0.0, 1.0]
    decoded = decode_scenario(scores, ["a", "b"], ["p", "p"], ["fold2::g", "fold2::g"])
    assert decoded["independent"].tolist() == [0, 0]
    assert set(decoded["true_group_hungarian"].tolist()) == {0, 1}
    assert set(decoded["frozen_E19D_subset_hungarian"].tolist()) == {0, 1}


def test_scenario_score_alignment_is_required():
    with pytest.raises(ValueError, match="misaligned"):
        decode_scenario(np.zeros((1, len(LABELS)-1)), ["a"], ["p"], ["f::g"])
    with pytest.raises(ValueError, match="unique"):
        decode_scenario(np.zeros((2, len(LABELS))), ["a", "a"], ["p", "p"], ["f::g"]*2)
