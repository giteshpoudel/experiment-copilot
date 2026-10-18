"""Bayesian, segment and daily-effect functions against SciPy / statsmodels."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import integrate
from scipy.stats import beta, norm
from statsmodels.stats.meta_analysis import combine_effects

from expcopilot.stats.bayesian import (
    beta_binomial,
    expected_shortfall,
    normal_posterior,
    prob_b_beats_a,
)
from expcopilot.stats.frequentist import two_proportion_ztest
from expcopilot.stats.hte import cochran_q, segment_effects
from expcopilot.stats.time_effects import weighted_trend


@pytest.mark.parametrize(
    ("aa", "ba", "ab", "bb"), [(101, 901, 121, 881), (3, 50, 7, 40), (1500, 13500, 1560, 13440)]
)
def test_prob_b_beats_a_matches_integration(aa: float, ba: float, ab: float, bb: float) -> None:
    ref, _ = integrate.quad(
        lambda x: beta.pdf(x, ab, bb) * beta.cdf(x, aa, ba),
        0,
        1,
        limit=500,
        points=[aa / (aa + ba), ab / (ab + bb)],
    )
    assert prob_b_beats_a(aa, ba, ab, bb) == pytest.approx(ref, abs=1e-7)


def test_prob_b_beats_a_non_integer_prior_path() -> None:
    exact = prob_b_beats_a(101, 901, 121, 881)
    assert prob_b_beats_a(101, 901, 121.0000001, 881) == pytest.approx(exact, abs=1e-6)


def test_expected_shortfall_matches_dblquad() -> None:
    ax, bx, ay, by = 30, 270, 36, 264
    ref, _ = integrate.dblquad(
        lambda y, x: (x - y) * beta.pdf(x, ax, bx) * beta.pdf(y, ay, by),
        0,
        0.4,
        lambda x: 0,
        lambda x: x,
    )
    assert expected_shortfall(ax, bx, ay, by) == pytest.approx(ref, rel=1e-5)


def test_beta_binomial_direction_flips_for_lower_is_better() -> None:
    up = beta_binomial(300, 1000, 360, 1000)
    down = beta_binomial(300, 1000, 360, 1000, higher_is_better=False)
    assert up.prob_treatment_better == pytest.approx(1 - down.prob_treatment_better)
    assert up.expected_loss_ship == pytest.approx(down.expected_loss_keep)
    assert up.abs_credible_interval[0] > 0


def test_normal_posterior_closed_forms() -> None:
    res = normal_posterior(10.0, 0.3, 10.5, 0.4)
    s = 0.5
    assert res.prob_treatment_better == pytest.approx(norm.cdf(0.5 / s))
    ref, _ = integrate.quad(lambda d: max(-d, 0) * norm.pdf(d, 0.5, s), -10, 10, points=[0])
    assert res.expected_loss_ship == pytest.approx(ref, rel=1e-8)
    assert res.abs_credible_interval == pytest.approx((0.5 - 1.959964 * s, 0.5 + 1.959964 * s))


def test_cochran_q_matches_statsmodels() -> None:
    eff = np.array([0.02, 0.005, -0.004, 0.03])
    se = np.array([0.006, 0.007, 0.01, 0.012])
    q, df, _ = cochran_q(eff, se)
    ref = combine_effects(eff, se**2)
    assert q == pytest.approx(ref.q, rel=1e-10)
    assert df == ref.df


def test_segment_effects_adjusts_and_skips() -> None:
    rng = np.random.default_rng(3)
    n = 20_000
    seg = rng.choice(["ios", "android", "web", "tiny"], n, p=[0.45, 0.35, 0.199, 0.001])
    treat = rng.integers(0, 2, n).astype(bool)
    p = np.where(treat & (seg == "ios"), 0.13, 0.10)
    y = rng.random(n) < p

    def est(c: np.ndarray, t: np.ndarray):
        return two_proportion_ztest(int(y[c].sum()), int(c.sum()), int(y[t].sum()), int(t.sum()))

    res = segment_effects("platform", seg, treat, est)
    assert res.skipped_levels == ("tiny",)
    assert [e.level for e in res.effects] == ["android", "ios", "web"]
    assert all(e.p_adjusted >= e.estimate.p_value for e in res.effects)
    assert res.exploratory


def test_weighted_trend_matches_statsmodels_wls() -> None:
    days = np.arange(14)
    se = np.linspace(0.05, 0.03, 14)
    y = 0.3 * 0.7**days + np.random.default_rng(1).normal(0, se)
    intercept, slope, slope_se, p = weighted_trend(days, y, se)
    fit = sm.WLS(y, sm.add_constant(days), weights=1 / se**2).fit()
    assert (intercept, slope) == pytest.approx(tuple(fit.params), rel=1e-10)
    assert slope_se == pytest.approx(fit.bse[1], rel=1e-10)
    assert p == pytest.approx(fit.pvalues[1], rel=1e-8)
