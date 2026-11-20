import numpy as np
import pytest

from src.e19 import (LABEL_TO_INDEX, _sample_groups, _validate_unique_groups,
                     greedy_assignment, hungarian_assignment, score_matrix)
from src.v2_checkpoint_trust import LABELS


def test_hungarian_finds_global_optimum_and_enforces_unique_labels():
    scores = np.zeros((2, len(LABELS)), dtype=float)
    scores[0, 0], scores[0, 1] = 100.0, 99.0
    scores[1, 0], scores[1, 1] = 99.0, 0.0

    pred, assignments = hungarian_assignment(scores, ["prompt", "prompt"])

    assert pred.tolist() == [1, 0]
    assert scores[np.arange(2), pred].sum() == 198.0
    assert len(set(pred.tolist())) == 2
    assert assignments == {"prompt": {0: 1, 1: 0}}


@pytest.mark.parametrize("size", range(1, 19))
def test_hungarian_supports_all_group_sizes(size):
    scores = np.zeros((size, len(LABELS)), dtype=float)
    scores[np.arange(size), np.arange(size)] = 1.0

    pred, _ = hungarian_assignment(scores, ["p"] * size)

    assert pred.tolist() == list(range(size))
    _validate_unique_groups(pred, ["p"] * size)


def test_hungarian_rejects_more_than_18_rows_and_misaligned_scores():
    with pytest.raises(ValueError, match="unsupported size"):
        hungarian_assignment(np.zeros((19, 18)), ["p"] * 19)
    with pytest.raises(ValueError, match="align"):
        hungarian_assignment(np.zeros((2, 18)), ["p"])


def test_e19_scores_match_e13_family_max_for_supported_labels_and_fill_all_labels():
    p = np.full((3, 1, 18), 1 / 18, dtype=float)
    p[0, 0, LABEL_TO_INDEX[LABELS[0]]] = 0.8
    p[0, 0, :] /= p[0, 0, :].sum()
    p[1, 0, LABEL_TO_INDEX[LABELS[0]]] = 0.7
    p[1, 0, :] /= p[1, 0, :].sum()
    p[2, 0, LABEL_TO_INDEX[LABELS[1]]] = 0.6
    p[2, 0, :] /= p[2, 0, :].sum()
    hard = p.argmax(axis=2)
    reliability = np.asarray([[0.8], [0.6], [0.9]])
    families = ["family-a", "family-a", "family-b"]

    scores = score_matrix(p, hard, reliability, families)

    assert scores.shape == (1, 18)
    # Exact E13 family-max score: max within family-a and family-b, mean families.
    # The supported label has only family-a support, so its score is .8.
    assert scores[0, LABEL_TO_INDEX[LABELS[0]]] == pytest.approx(0.8)
    assert np.isfinite(scores).all()


def test_sampling_is_reproducible_and_returns_only_incomplete_groups():
    groups = np.repeat(["a", "b", "c"], [18, 18, 17])
    rows_a, _ = _sample_groups(groups, seed=2026, cap=10)
    rows_b, _ = _sample_groups(groups, seed=2026, cap=10)
    rows_c, _ = _sample_groups(groups, seed=2027, cap=10)

    assert np.array_equal(rows_a, rows_b)
    assert not np.array_equal(rows_a, rows_c)
    for group in np.unique(groups[rows_a]):
        count = int(np.sum(groups[rows_a] == group))
        assert 2 <= count <= 10
    assert len(rows_a) < len(groups)


def test_greedy_and_hungarian_use_identical_scores_and_keep_alignment():
    scores = np.zeros((3, 18), dtype=float)
    scores[:, :3] = [[9, 8, 0], [9, 0, 8], [0, 9, 8]]
    groups = np.asarray(["g", "g", "g"])
    ids = np.asarray(["r1", "r2", "r3"])

    hungarian, _ = hungarian_assignment(scores, groups, row_ids=ids)
    greedy = greedy_assignment(scores, groups)

    assert len(hungarian) == len(greedy) == len(ids)
    assert np.isfinite(scores[np.arange(3), hungarian]).all()
    assert len(set(hungarian.tolist())) == len(set(greedy.tolist())) == 3
