"""Uplift engine against scikit-uplift on fixed inputs, plus recovery of a known effect."""

from __future__ import annotations

import numpy as np
import pytest
from sklift.metrics import qini_auc_score
from sklift.metrics import qini_curve as sk_qini_curve

from expcopilot.stats.uplift import by_decile, fit_and_evaluate, qini_coefficient, qini_curve


def _data(n: int = 6_000, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    t = (rng.random(n) < 0.5).astype(int)
    # Treatment helps only when x0 > 0: true uplift 0.15 there, 0 elsewhere.
    p = 0.10 + 0.05 * (x[:, 1] > 0) + t * 0.15 * (x[:, 0] > 0)
    y = (rng.random(n) < p).astype(int)
    return x, y, t


def test_qini_curve_matches_sklift() -> None:
    _, y, t = _data()
    score = np.random.default_rng(3).normal(size=y.size).round(1)  # ties on purpose
    xs, qs = qini_curve(y, score, t)
    xr, qr = sk_qini_curve(y, score, t)
    np.testing.assert_allclose(xs, xr)
    np.testing.assert_allclose(qs, qr, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qini_coefficient_matches_sklift(seed: int) -> None:
    x, y, t = _data(seed=seed)
    score = x[:, 0] + np.random.default_rng(seed).normal(scale=1.0, size=y.size)
    assert qini_coefficient(y, score, t) == pytest.approx(
        qini_auc_score(y, score, t, negative_effect=True), rel=1e-10
    )


def test_learners_find_the_responsive_segment() -> None:
    """Qini is normalized by a 'perfect' ranking built from realized outcomes, so even the
    true-effect score scores only ~0.12 here; learners should reach a good share of it."""
    x, y, t = _data(n=20_000)
    ceiling = qini_coefficient(y, (x[:, 0] > 0).astype(float), t)
    for learner in ("t", "x"):
        res = fit_and_evaluate(x, y, t, learner=learner, seed=0)
        assert res.qini_coefficient > 0.4 * ceiling, (learner, res.qini_coefficient, ceiling)
        top, bottom = res.deciles[0], res.deciles[-1]
        assert top.uplift > bottom.uplift
        assert res.n_train + res.n_test == 20_000


def test_random_scores_have_qini_near_zero() -> None:
    _, y, t = _data(n=20_000)
    q = qini_coefficient(y, np.random.default_rng(9).random(y.size), t)
    assert abs(q) < 0.03


def test_by_decile_partitions_units() -> None:
    _, y, t = _data()
    rows = by_decile(y, np.arange(y.size, dtype=float), t)
    assert len(rows) == 10 and sum(r.n_treatment + r.n_control for r in rows) == y.size


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        fit_and_evaluate(np.zeros((10, 2)), np.zeros(10), np.zeros(10))
