import numpy as np
import pytest

from src.e38_e39_inference import _align, blend_equal


def test_align_restores_canonical_label_order_and_checks_normalization():
    got = _align(np.asarray([[0.7, 0.3]]), ["b", "a"], ["a", "b", "c"])
    np.testing.assert_allclose(got, [[0.3, 0.7, 0.0]])


def test_equal_blend_row_normalizes_host_scores():
    host = np.asarray([[2.0, 1.0, 0.0]])
    aux = np.asarray([[0.0, 0.25, 0.75]])
    got = blend_equal(host, aux)
    np.testing.assert_allclose(got, [[1 / 3, 0.2916666666666667, 0.375]])
    np.testing.assert_allclose(got.sum(1), 1.0)


def test_equal_blend_handles_finite_fallback_scores_below_zero():
    got = blend_equal(np.asarray([[0.7, -0.3, 0.0]]),
                      np.asarray([[0.2, 0.3, 0.5]]))
    np.testing.assert_allclose(got, [[0.6, 0.15, 0.25]])


def test_equal_blend_rejects_misaligned_or_invalid_inputs():
    with pytest.raises(ValueError, match="aligned"):
        blend_equal(np.ones((1, 2)), np.ones((1, 3)) / 3)
    with pytest.raises(ValueError, match="invalid"):
        blend_equal(np.ones((1, 2)), np.asarray([[0.2, 0.2]]))
