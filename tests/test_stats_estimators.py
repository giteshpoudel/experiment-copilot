"""Effect estimators against SciPy / statsmodels on fixed inputs."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.stats.proportion import test_proportions_2indep as sm_test_proportions_2indep
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from expcopilot.stats.cuped import cuped
from expcopilot.stats.frequentist import bootstrap, two_proportion_ztest, welch_ttest
from expcopilot.stats.ratio import delta_method

RNG = np.random.default_rng(20261018)
Y_C = RNG.lognormal(2.0, 1.0, 3_000)
Y_T = RNG.lognormal(2.08, 1.1, 2_600)


@pytest.mark.parametrize(("sc", "nc", "st", "nt"), [(1500, 15000, 1650, 15030), (40, 900, 61, 880)])
def test_ztest_matches_statsmodels(sc: int, nc: int, st: int, nt: int) -> None:
    res = two_proportion_ztest(sc, nc, st, nt)
    ref = sm_test_proportions_2indep(st, nt, sc, nc, method="wald", compare="diff")
    lo, hi = confint_proportions_2indep(st, nt, sc, nc, method="wald", compare="diff")
    assert res.abs_effect == pytest.approx(st / nt - sc / nc, rel=1e-12)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.abs_ci == pytest.approx((lo, hi), rel=1e-9)
    assert (res.p_value < 0.05) == (res.abs_ci[0] > 0 or res.abs_ci[1] < 0)


def test_welch_matches_scipy_and_statsmodels() -> None:
    res = welch_ttest(Y_C, Y_T)
    ref = stats.ttest_ind(Y_T, Y_C, equal_var=False)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.df == pytest.approx(ref.df, rel=1e-9)
    cm = CompareMeans(DescrStatsW(Y_T), DescrStatsW(Y_C))
    assert res.abs_ci == pytest.approx(cm.tconfint_diff(usevar="unequal"), rel=1e-9)


def test_relative_ci_delta_method_by_hand() -> None:
    res = welch_ttest(Y_C, Y_T)
    mc, mt = Y_C.mean(), Y_T.mean()
    vc, vt = Y_C.var(ddof=1) / Y_C.size, Y_T.var(ddof=1) / Y_T.size
    se = np.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    assert res.rel_effect == pytest.approx(mt / mc - 1)
    assert res.rel_se == pytest.approx(se, rel=1e-12)


def test_bootstrap_close_to_scipy() -> None:
    b = 20_000
    res = bootstrap(Y_C, Y_T, n_resamples=b, seed=1)
    ref = stats.bootstrap(
        (Y_T, Y_C),
        lambda a, b, axis: a.mean(axis) - b.mean(axis),
        vectorized=True,
        n_resamples=b,
        method="percentile",
        random_state=np.random.default_rng(2),
    )
    # Different resamples, so the CIs agree only up to Monte Carlo error. The sd of a 2.5%
    # quantile is sqrt(q(1-q)/B)/phi(z_q) se ~= 0.019 se at B=20k; the gap between two
    # independent runs has sd ~0.027 se. Tolerance = 4 sd.
    welch = welch_ttest(Y_C, Y_T)
    tol = 0.11 * welch.abs_se
    assert res.abs_ci[0] == pytest.approx(ref.confidence_interval.low, abs=tol)
    assert res.abs_ci[1] == pytest.approx(ref.confidence_interval.high, abs=tol)
    assert res.abs_se == pytest.approx(ref.standard_error, rel=0.05)


def _ratio_data(seed: int, lift: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    den = 1 + rng.poisson(3, 4_000)
    num = rng.gamma(2.0, 1.5 * (1 + lift), 4_000) * den
    return num, den


def test_delta_method_matches_wls_cluster_robust() -> None:
    num_c, den_c = _ratio_data(1, 0.0)
    num_t, den_t = _ratio_data(2, 0.05)
    res = delta_method(num_c, den_c, num_t, den_t)
    # WLS of per-unit ratio on treatment with weights = denominator; HC0 sandwich is the
    # delta-method variance with ddof=0 in each arm.
    y = np.concatenate([num_c / den_c, num_t / den_t])
    w = np.concatenate([den_c, den_t])
    treat = np.r_[np.zeros(den_c.size), np.ones(den_t.size)]
    fit = sm.WLS(y, sm.add_constant(treat), weights=w).fit(cov_type="HC0")
    assert res.abs_effect == pytest.approx(fit.params[1], rel=1e-10)
    n = den_c.size  # equal arm sizes: ddof=1 vs ddof=0 differ by n / (n - 1)
    assert res.abs_se == pytest.approx(fit.bse[1] * np.sqrt(n / (n - 1)), rel=1e-6)


def test_cuped_matches_ancova() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(10, 3, 6_000)
    treat = rng.integers(0, 2, 6_000).astype(bool)
    y = 2 + 0.8 * x + 0.3 * treat + rng.normal(0, 2, 6_000)
    res = cuped(y[~treat], x[~treat], y[treat], x[treat])
    fit = sm.OLS(y, sm.add_constant(np.column_stack([treat, x - x.mean()]))).fit(cov_type="HC2")
    assert res.estimate.abs_effect == pytest.approx(fit.params[1], rel=1e-10)
    assert res.theta == pytest.approx(fit.params[2], rel=1e-10)
    # Welch on adjusted outcomes treats theta as known; agrees with HC2 to ~1%.
    assert res.estimate.abs_se == pytest.approx(fit.bse[1], rel=0.02)
    assert res.variance_reduction == pytest.approx(res.covariate_correlation**2, abs=0.02)
    plain = welch_ttest(y[~treat], y[treat])
    assert res.estimate.abs_se < plain.abs_se
