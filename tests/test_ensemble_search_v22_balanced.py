import numpy as np

from src.ensemble_search_v22_balanced import _fit_balanced_weights


def test_balanced_weight_optimizer_is_deterministic_and_simplex_constrained():
    rng = np.random.default_rng(4)
    y = np.tile(np.arange(3), 24)
    prob = rng.random((4, len(y), 3))
    prob /= prob.sum(axis=2, keepdims=True)

    first, diag1 = _fit_balanced_weights(prob, y, seed=2026, population_size=8, generations=2)
    second, diag2 = _fit_balanced_weights(prob, y, seed=2026, population_size=8, generations=2)

    assert np.all(first >= 0)
    assert np.isclose(first.sum(), 1.0)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
    assert diag1["nfev"] == diag2["nfev"]
    assert diag1["starts"] == ["equal", "regularized_log_loss", "seeded_dirichlet"]


def test_balanced_weight_optimizer_handles_one_checkpoint():
    y = np.array([0, 1, 0, 1])
    prob = np.array([[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]])
    weights, diagnostics = _fit_balanced_weights(prob, y, seed=9)
    np.testing.assert_array_equal(weights, np.array([1.0]))
    assert diagnostics["nfev"] == 1
