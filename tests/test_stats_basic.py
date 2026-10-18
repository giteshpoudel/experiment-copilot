"""Power, SRM and multiple testing against statsmodels / SciPy on fixed inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower, normal_sample_size_one_tail

from expcopilot.stats.multiple import adjust_pvalues
from expcopilot.stats.power import (
    mde_binary,
    mde_continuous,
    n_binary_exact,
    n_continuous_exact,
    n_per_arm_binary,
)
from expcopilot.stats.srm import srm_test


@pytest.mark.parametrize(("p", "mde"), [(0.10, 0.10), (0.03, 0.2), (0.5, -0.05)])
def test_binary_n_matches_statsmodels(p: float, mde: float) -> None:
    p2 = p * (1 + mde)
    pbar = (p + p2) / 2
    ref = normal_sample_size_one_tail(
        p2 - p,
        power=0.8,
        alpha=0.025,
        std_null=math.sqrt(2 * pbar * (1 - pbar)),
        std_alternative=math.sqrt(p * (1 - p) + p2 * (1 - p2)),
    )
    assert n_binary_exact(p, mde) == pytest.approx(ref, rel=1e-10)


def test_binary_n_known_value() -> None:
    # 10% -> 11% at alpha 0.05 two-sided, 80% power: ~14,750 per arm.
    assert n_per_arm_binary(0.10, 0.10) == pytest.approx(14_750, abs=10)


@pytest.mark.parametrize("ratio", [1.0, 0.5, 3.0])
def test_continuous_n_matches_statsmodels(ratio: float) -> None:
    sd, delta = 30.0, 1.5
    ref = NormalIndPower().solve_power(
        effect_size=delta / sd, alpha=0.05, power=0.8, ratio=ratio, alternative="two-sided"
    )
    # statsmodels also counts the far rejection tail, which is negligible here.
    assert n_continuous_exact(sd, delta, ratio=ratio) == pytest.approx(ref, rel=1e-4)


def test_mde_inverts_n() -> None:
    n = n_continuous_exact(12.0, 0.8)
    assert mde_continuous(12.0, n) == pytest.approx(0.8, rel=1e-10)
    n_b = n_binary_exact(0.1, 0.07)
    assert mde_binary(0.1, n_b) == pytest.approx(0.07, rel=1e-8)


@pytest.mark.parametrize(
    ("observed", "shares"),
    [((15_000, 14_100), (0.5, 0.5)), ((850, 152), (0.85, 0.15)), ((10, 12, 9, 15), (0.25,) * 4)],
)
def test_srm_matches_scipy(observed: tuple[int, ...], shares: tuple[float, ...]) -> None:
    res = srm_test(observed, shares)
    expected = np.array(shares) * sum(observed)
    ref = chisquare(observed, expected)
    assert res.chi2 == pytest.approx(ref.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-10)


def test_srm_flags_injected_mismatch_and_passes_planned_imbalance() -> None:
    assert not srm_test((14_901, 14_162), (0.5, 0.5)).passed
    assert srm_test((85_120, 14_880), (0.85, 0.15)).passed


@pytest.mark.parametrize("method", ["holm", "bh"])
def test_adjust_pvalues_matches_statsmodels(method: str) -> None:
    rng = np.random.default_rng(7)
    p = np.concatenate([rng.uniform(size=17), [0.001, 0.01, 0.04, 0.04, 1.0]])
    ref = multipletests(p, method={"holm": "holm", "bh": "fdr_bh"}[method])[1]
    np.testing.assert_allclose(adjust_pvalues(p, method), ref, rtol=1e-12)


def test_adjust_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2])
