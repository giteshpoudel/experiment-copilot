"""Sequential tests against published boundaries and SciPy integration."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import multivariate_normal, norm

from expcopilot.stats.sequential import (
    group_sequential_bounds,
    group_sequential_test,
    msprt,
    msprt_likelihood_ratio,
    naive_peeking,
    obf_spending,
)


def _crossing_probability(bounds: np.ndarray, t: np.ndarray) -> float:
    """P(|Z_k| >= c_k for some k) under H0, using SciPy's multivariate normal CDF."""
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    mvn = multivariate_normal(mean=np.zeros(t.size), cov=cov)
    inside = mvn.cdf(bounds, lower_limit=-bounds, rng=np.random.default_rng(0))
    return 1 - inside


def test_single_look_is_the_fixed_horizon_test() -> None:
    assert group_sequential_bounds([1.0])[0] == pytest.approx(norm.isf(0.025), rel=1e-12)


def test_k5_equal_spacing_matches_published_values() -> None:
    # Lan-DeMets OBF-type spending, alpha=0.05 two-sided, 5 equally spaced looks
    # (as tabulated by gsDesign sfLDOF / rpact).
    bounds = group_sequential_bounds([0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(bounds, [4.877, 3.357, 2.680, 2.290, 2.031], atol=2e-3)


@pytest.mark.parametrize(
    "t", [np.linspace(0.2, 1, 5), np.arange(1, 15) / 14, np.array([0.1, 0.35, 0.5, 0.9, 1.0])]
)
def test_total_alpha_matches_scipy_mvn(t: np.ndarray) -> None:
    bounds = group_sequential_bounds(t)
    assert _crossing_probability(bounds, t) == pytest.approx(0.05, abs=5e-4)


def test_spending_function_endpoints() -> None:
    assert obf_spending(1.0) == pytest.approx(0.05)
    assert obf_spending(1e-6) == pytest.approx(0.0, abs=1e-12)


def test_group_sequential_and_naive_stop_differently() -> None:
    t = [0.25, 0.5, 0.75, 1.0]
    z = [2.2, 2.4, 1.5, 1.2]  # naive peeking "wins" at look 1; OBF never stops
    assert naive_peeking(z, t).stopped_at == 0
    assert group_sequential_test(z, t).stopped_at is None
    assert group_sequential_test([1.0, 3.5, 0, 0], t).stopped_at == 1


def test_msprt_ratio_matches_numerical_integration() -> None:
    effect, var, tau = 0.012, 0.004**2, 0.01

    def integrand(theta: float) -> float:
        lr = norm.pdf(effect, theta, np.sqrt(var)) / norm.pdf(effect, 0, np.sqrt(var))
        return lr * norm.pdf(theta, 0, tau)

    ref, _ = integrate.quad(integrand, -10 * tau, 10 * tau, points=[effect], limit=200)
    assert msprt_likelihood_ratio(effect, var, tau) == pytest.approx(ref, rel=1e-8)


def test_msprt_p_values_are_monotone() -> None:
    res = msprt([0.0, 0.01, 0.002, 0.03], [1e-4, 5e-5, 3e-5, 2e-5], tau=0.01)
    assert res.p_values is not None
    assert all(a >= b for a, b in itertools.pairwise(res.p_values))
    assert res.stopped_at == 3
