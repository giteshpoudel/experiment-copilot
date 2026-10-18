"""Sequential tests that stay valid when you look at the data more than once.

- Group sequential with an O'Brien-Fleming-type Lan-DeMets alpha-spending function: any number
  of looks at any information fractions; boundaries are computed by numerical integration of
  the score process (Armitage-McPherson-Rowe recursion).
- mSPRT (mixture sequential probability ratio test, Johari et al. 2017) with a normal mixing
  distribution: always-valid p-values that can be checked after every unit.
- Naive peeking (a fixed-horizon z-test at every look), for comparison only.

All tests are two-sided and take per-look z statistics or effect estimates computed elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

GRID_POINTS = 1201  # odd, for Simpson's rule
Z_CAP = 8.0  # grid edge when a look spends ~no alpha; P(|Z| > 8) ~ 1e-15


@dataclass(frozen=True)
class SequentialResult:
    method: str
    alpha: float
    info_fractions: tuple[float, ...]
    statistics: tuple[float, ...]  # z per look (group sequential, naive) or 1/Lambda p-values
    boundaries: tuple[float, ...]  # |z| boundary per look; for mSPRT the p-value threshold
    stopped_at: int | None  # 0-based look index of the first crossing, None if never
    p_values: tuple[float, ...] | None = None  # always-valid p-values (mSPRT)

    @property
    def rejected(self) -> bool:
        return self.stopped_at is not None


def obf_spending(t: np.ndarray | float, alpha: float = 0.05) -> np.ndarray:
    """Two-sided Lan-DeMets O'Brien-Fleming-type spending: 4 - 4 Phi(z_{alpha/4} / sqrt t).

    This is the one-sided OBF spending function at alpha/2, applied to each side. (Using the
    one-sided form 2 - 2 Phi(z_{alpha/2} / sqrt t) with the total alpha also spends exactly
    alpha, but front-loads it: first boundary 4.38 instead of 4.88 for five equal looks.)
    """
    t = np.asarray(t, float)
    return 4 - 4 * norm.cdf(norm.isf(alpha / 4) / np.sqrt(t))


def _simpson_weights(n: int, h: float) -> np.ndarray:
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * h / 3


def _check_fractions(info_fractions: Sequence[float]) -> np.ndarray:
    t = np.asarray(info_fractions, float)
    if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) <= 0) or t[0] <= 0 or t[-1] > 1 + 1e-12:
        raise ValueError("information fractions must be strictly increasing in (0, 1]")
    return t


def group_sequential_bounds(info_fractions: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Two-sided |z| boundaries per look so total type I error is alpha under H0."""
    t = _check_fractions(info_fractions)
    spend = np.diff(np.concatenate([[0.0], obf_spending(t, alpha)]))
    bounds = np.empty(t.size)
    bounds[0] = norm.isf(spend[0] / 2) if spend[0] > 1e-15 else math.inf

    # Sub-density of the score S_k = Z_k sqrt(t_k) on the continuation region (-b_k, b_k).
    b = min(bounds[0], Z_CAP) * math.sqrt(t[0])
    s = np.linspace(-b, b, GRID_POINTS)
    w = _simpson_weights(GRID_POINTS, s[1] - s[0])
    f = norm.pdf(s, scale=math.sqrt(t[0]))

    for k in range(1, t.size):
        sd = math.sqrt(t[k] - t[k - 1])
        mass = w * f

        def excess(
            c: float, k: int = k, sd: float = sd, mass: np.ndarray = mass, s: np.ndarray = s
        ) -> float:
            bk = c * math.sqrt(t[k])
            cross = norm.sf((bk - s) / sd) + norm.cdf((-bk - s) / sd)
            return float(mass @ cross) - spend[k]

        if spend[k] <= 1e-15:
            bounds[k] = math.inf
        else:
            bounds[k] = brentq(excess, 1e-6, 40.0, xtol=1e-10)
        if k == t.size - 1:
            break
        bk = min(bounds[k], Z_CAP) * math.sqrt(t[k])
        s_new = np.linspace(-bk, bk, GRID_POINTS)
        kernel = norm.pdf((s_new[:, None] - s[None, :]) / sd) / sd
        f = kernel @ mass
        s, w = s_new, _simpson_weights(GRID_POINTS, s_new[1] - s_new[0])
    return bounds


def group_sequential_test(
    z_stats: Sequence[float], info_fractions: Sequence[float], alpha: float = 0.05
) -> SequentialResult:
    """Stop at the first look whose |z| crosses its OBF-spending boundary."""
    z = np.asarray(z_stats, float)
    t = _check_fractions(info_fractions)
    if z.shape != t.shape:
        raise ValueError("need one z statistic per look")
    bounds = group_sequential_bounds(t, alpha)
    hits = np.flatnonzero(np.abs(z) >= bounds)
    return SequentialResult(
        method="group_sequential_obf",
        alpha=alpha,
        info_fractions=tuple(t.tolist()),
        statistics=tuple(z.tolist()),
        boundaries=tuple(bounds.tolist()),
        stopped_at=int(hits[0]) if hits.size else None,
    )


def naive_peeking(
    z_stats: Sequence[float], info_fractions: Sequence[float], alpha: float = 0.05
) -> SequentialResult:
    """A fixed-horizon test at every look. Not valid: shown to quantify the peeking problem."""
    z = np.asarray(z_stats, float)
    t = _check_fractions(info_fractions)
    crit = float(norm.isf(alpha / 2))
    hits = np.flatnonzero(np.abs(z) >= crit)
    return SequentialResult(
        method="naive_peeking",
        alpha=alpha,
        info_fractions=tuple(t.tolist()),
        statistics=tuple(z.tolist()),
        boundaries=(crit,) * t.size,
        stopped_at=int(hits[0]) if hits.size else None,
    )


def msprt_likelihood_ratio(effect: float, variance: float, tau: float) -> float:
    """Mixture likelihood ratio for an effect estimate ~ N(theta, variance), theta ~ N(0, tau^2).

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 * effect^2 / (2 V (V + tau^2))).
    """
    if variance <= 0 or tau <= 0:
        raise ValueError("variance and tau must be positive")
    v, t2 = variance, tau**2
    return math.sqrt(v / (v + t2)) * math.exp(t2 * effect**2 / (2 * v * (v + t2)))


def msprt(
    effects: Sequence[float],
    variances: Sequence[float],
    tau: float,
    alpha: float = 0.05,
    info_fractions: Sequence[float] | None = None,
) -> SequentialResult:
    """mSPRT on a sequence of (effect, variance-of-effect) estimates at successive looks.

    The always-valid p-value is the running minimum of min(1, 1/Lambda); reject when it falls
    below alpha. `tau` is the prior sd of the effect on the same scale as `effects`; a common
    choice is the plan's absolute MDE.
    """
    eff, var = np.asarray(effects, float), np.asarray(variances, float)
    if eff.shape != var.shape or eff.size == 0:
        raise ValueError("need one variance per effect")
    lam = np.array([msprt_likelihood_ratio(e, v, tau) for e, v in zip(eff, var, strict=True)])
    p = np.minimum.accumulate(np.minimum(1.0, 1.0 / lam))
    hits = np.flatnonzero(p < alpha)
    t = (
        _check_fractions(info_fractions)
        if info_fractions is not None
        else np.arange(1, eff.size + 1) / eff.size
    )
    return SequentialResult(
        method="msprt",
        alpha=alpha,
        info_fractions=tuple(t.tolist()),
        statistics=tuple((1.0 / lam).tolist()),
        boundaries=(alpha,) * eff.size,
        stopped_at=int(hits[0]) if hits.size else None,
        p_values=tuple(p.tolist()),
    )
