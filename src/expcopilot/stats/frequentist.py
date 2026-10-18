"""Two-arm effect estimates: two-proportion z-test, Welch t-test, percentile bootstrap.

Every estimator returns an `EffectEstimate` with the absolute effect (treatment - control) and
the relative effect (treatment / control - 1), each with a CI, plus a two-sided p-value. The
relative CI uses the delta method on the ratio of the two independent arm means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t


@dataclass(frozen=True)
class EffectEstimate:
    method: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    abs_effect: float
    abs_se: float
    abs_ci: tuple[float, float]
    rel_effect: float
    rel_se: float
    rel_ci: tuple[float, float]
    p_value: float
    alpha: float
    df: float | None = None
    variance_reduction: float | None = None

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def relative_delta(
    mean_t: float, var_mean_t: float, mean_c: float, var_mean_c: float
) -> tuple[float, float]:
    """Relative effect m_t / m_c - 1 and its delta-method se, for independent arm means."""
    if mean_c == 0:
        return math.nan, math.nan
    rel = mean_t / mean_c - 1
    var = var_mean_t / mean_c**2 + mean_t**2 * var_mean_c / mean_c**4
    return rel, math.sqrt(var)


def _two_sided_p(stat: float, df: float | None) -> float:
    if math.isnan(stat):
        return math.nan
    return float(2 * (t.sf(abs(stat), df) if df is not None else norm.sf(abs(stat))))


def from_arm_means(
    method: str,
    n_c: int,
    n_t: int,
    mean_c: float,
    mean_t: float,
    var_mean_c: float,
    var_mean_t: float,
    alpha: float = 0.05,
    df: float | None = None,
    variance_reduction: float | None = None,
) -> EffectEstimate:
    """Build an estimate from arm means and the variances of those means (normal or t)."""
    diff = mean_t - mean_c
    se = math.sqrt(var_mean_c + var_mean_t)
    crit = float(t.isf(alpha / 2, df) if df is not None else norm.isf(alpha / 2))
    p = _two_sided_p(diff / se, df) if se > 0 else 1.0 if diff == 0 else 0.0
    rel, rel_se = relative_delta(mean_t, var_mean_t, mean_c, var_mean_c)
    return EffectEstimate(
        method=method,
        n_control=int(n_c),
        n_treatment=int(n_t),
        mean_control=float(mean_c),
        mean_treatment=float(mean_t),
        abs_effect=float(diff),
        abs_se=se,
        abs_ci=(diff - crit * se, diff + crit * se),
        rel_effect=rel,
        rel_se=rel_se,
        rel_ci=(rel - crit * rel_se, rel + crit * rel_se),
        p_value=p,
        alpha=alpha,
        df=df,
        variance_reduction=variance_reduction,
    )


def two_proportion_ztest(
    successes_c: int, n_c: int, successes_t: int, n_t: int, alpha: float = 0.05
) -> EffectEstimate:
    """Wald z-test for a difference in proportions, unpooled variance.

    The p-value and the CI use the same (unpooled) se, so "CI excludes zero" and "p < alpha"
    always agree; that matters because the decision rule is written in terms of the CI.
    """
    if n_c <= 0 or n_t <= 0:
        raise ValueError("both arms need units")
    pc, pt = successes_c / n_c, successes_t / n_t
    return from_arm_means(
        "two_proportion_z", n_c, n_t, pc, pt, pc * (1 - pc) / n_c, pt * (1 - pt) / n_t, alpha
    )


def welch_from_stats(
    mean_c: float,
    var_c: float,
    n_c: int,
    mean_t: float,
    var_t: float,
    n_t: int,
    alpha: float = 0.05,
    method: str = "welch_t",
    variance_reduction: float | None = None,
) -> EffectEstimate:
    """Welch t-test from summary statistics (sample variances with ddof=1)."""
    if n_c < 2 or n_t < 2:
        raise ValueError("Welch needs at least two units per arm")
    vc, vt = var_c / n_c, var_t / n_t
    denom = vc**2 / (n_c - 1) + vt**2 / (n_t - 1)
    df = (vc + vt) ** 2 / denom if denom > 0 else float(n_c + n_t - 2)
    return from_arm_means(method, n_c, n_t, mean_c, mean_t, vc, vt, alpha, df, variance_reduction)


def welch_ttest(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05) -> EffectEstimate:
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    return welch_from_stats(
        y_c.mean(), y_c.var(ddof=1), y_c.size, y_t.mean(), y_t.var(ddof=1), y_t.size, alpha
    )


def bootstrap(
    y_c: np.ndarray,
    y_t: np.ndarray,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    seed: int = 0,
    chunk: int = 100,
) -> EffectEstimate:
    """Percentile bootstrap, resampling units within each arm.

    se is the sd of the bootstrap differences; the p-value inverts the percentile CI
    (twice the smaller tail share of bootstrap differences on either side of zero).
    """
    y_c, y_t = np.asarray(y_c, float), np.asarray(y_t, float)
    rng = np.random.default_rng(seed)
    mc, mt = np.empty(n_resamples), np.empty(n_resamples)
    for start in range(0, n_resamples, chunk):
        k = min(chunk, n_resamples - start)
        mc[start : start + k] = y_c[rng.integers(0, y_c.size, (k, y_c.size))].mean(axis=1)
        mt[start : start + k] = y_t[rng.integers(0, y_t.size, (k, y_t.size))].mean(axis=1)
    diffs = mt - mc
    with np.errstate(divide="ignore", invalid="ignore"):
        rels = mt / mc - 1
    q = [alpha / 2, 1 - alpha / 2]
    abs_ci = tuple(float(x) for x in np.quantile(diffs, q))
    rel_ci = tuple(float(x) for x in np.quantile(rels, q))
    p = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    mean_c, mean_t = float(y_c.mean()), float(y_t.mean())
    return EffectEstimate(
        method="bootstrap_percentile",
        n_control=y_c.size,
        n_treatment=y_t.size,
        mean_control=mean_c,
        mean_treatment=mean_t,
        abs_effect=mean_t - mean_c,
        abs_se=float(diffs.std(ddof=1)),
        abs_ci=abs_ci,  # type: ignore[arg-type]
        rel_effect=mean_t / mean_c - 1 if mean_c else math.nan,
        rel_se=float(np.nanstd(rels, ddof=1)),
        rel_ci=rel_ci,  # type: ignore[arg-type]
        p_value=float(p),
        alpha=alpha,
    )
