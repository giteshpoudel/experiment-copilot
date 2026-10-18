"""Bayesian A/B summaries: P(treatment better), credible intervals, expected loss.

Binary metrics: Beta-binomial with a Beta(a, b) prior (default uniform). P(B > A) is exact
(closed-form sum, Evan Miller's formula); expected loss is a deterministic 1-d integral.
Continuous and ratio metrics: normal approximation with a flat prior, so the posterior of each
arm mean is N(mean, se^2).

"Better" respects direction: pass higher_is_better=False for metrics like bounce rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.special import betaln
from scipy.stats import beta as beta_dist
from scipy.stats import norm

MC_DRAWS = 200_000


@dataclass(frozen=True)
class BayesResult:
    method: str
    prob_treatment_better: float
    expected_loss_ship: float  # E[loss] in metric units if we ship treatment and it is worse
    expected_loss_keep: float  # E[loss] if we keep control and treatment is better
    posterior_mean_control: float
    posterior_mean_treatment: float
    abs_credible_interval: tuple[float, float]
    rel_credible_interval: tuple[float, float]
    credible_level: float
    prior: tuple[float, float] | None = None


def prob_b_beats_a(a_a: float, b_a: float, a_b: float, b_b: float) -> float:
    """P(p_B > p_A) for independent Beta posteriors. Exact when a_b is an integer."""
    if float(a_b).is_integer() and a_b < 1e6:
        i = np.arange(int(a_b))
        terms = betaln(a_a + i, b_a + b_b) - np.log(b_b + i) - betaln(1 + i, b_b) - betaln(a_a, b_a)
        return float(np.exp(terms).sum())
    lo, hi = _support(a_b, b_b)
    val, _ = integrate.quad(
        lambda x: beta_dist.pdf(x, a_b, b_b) * beta_dist.cdf(x, a_a, b_a), lo, hi, limit=200
    )
    return float(val)


def _support(a: float, b: float, k: float = 14.0) -> tuple[float, float]:
    m, sd = beta_dist.mean(a, b), beta_dist.std(a, b)
    return max(0.0, m - k * sd), min(1.0, m + k * sd)


def expected_shortfall(a_x: float, b_x: float, a_y: float, b_y: float) -> float:
    """E[max(X - Y, 0)] for independent X ~ Beta(a_x, b_x), Y ~ Beta(a_y, b_y).

    Inner integral in closed form: E[(X - y)+] = E[X](1 - I_y(a_x + 1, b_x)) - y(1 - I_y(a_x, b_x)).
    """
    mean_x = a_x / (a_x + b_x)

    def inner(y: float) -> float:
        return mean_x * beta_dist.sf(y, a_x + 1, b_x) - y * beta_dist.sf(y, a_x, b_x)

    lo, hi = _support(a_y, b_y)
    val, _ = integrate.quad(lambda y: beta_dist.pdf(y, a_y, b_y) * inner(y), lo, hi, limit=200)
    return float(val)


def beta_binomial(
    successes_c: int,
    n_c: int,
    successes_t: int,
    n_t: int,
    prior: tuple[float, float] = (1.0, 1.0),
    credible: float = 0.95,
    higher_is_better: bool = True,
    seed: int = 0,
) -> BayesResult:
    a0, b0 = prior
    a_c, b_c = a0 + successes_c, b0 + n_c - successes_c
    a_t, b_t = a0 + successes_t, b0 + n_t - successes_t
    p_up = prob_b_beats_a(a_c, b_c, a_t, b_t)
    loss_if_t_lower = expected_shortfall(a_c, b_c, a_t, b_t)  # E[(pC - pT)+]
    loss_if_t_higher = expected_shortfall(a_t, b_t, a_c, b_c)  # E[(pT - pC)+]

    rng = np.random.default_rng(seed)
    pc = rng.beta(a_c, b_c, MC_DRAWS)
    pt = rng.beta(a_t, b_t, MC_DRAWS)
    q = [(1 - credible) / 2, (1 + credible) / 2]
    abs_ci = tuple(float(x) for x in np.quantile(pt - pc, q))
    rel_ci = tuple(float(x) for x in np.quantile(pt / pc - 1, q))
    better = p_up if higher_is_better else 1 - p_up
    ship_loss, keep_loss = (
        (loss_if_t_lower, loss_if_t_higher)
        if higher_is_better
        else (loss_if_t_higher, loss_if_t_lower)
    )
    return BayesResult(
        method="beta_binomial",
        prob_treatment_better=better,
        expected_loss_ship=ship_loss,
        expected_loss_keep=keep_loss,
        posterior_mean_control=a_c / (a_c + b_c),
        posterior_mean_treatment=a_t / (a_t + b_t),
        abs_credible_interval=abs_ci,  # type: ignore[arg-type]
        rel_credible_interval=rel_ci,  # type: ignore[arg-type]
        credible_level=credible,
        prior=prior,
    )


def normal_posterior(
    mean_c: float,
    se_c: float,
    mean_t: float,
    se_t: float,
    credible: float = 0.95,
    higher_is_better: bool = True,
    seed: int = 0,
) -> BayesResult:
    """Flat-prior normal posterior for each arm mean (continuous or delta-method ratio)."""
    d = mean_t - mean_c
    s = math.hypot(se_c, se_t)
    if s <= 0:
        raise ValueError("standard errors must be positive")
    p_up = float(norm.cdf(d / s))
    # E[(D)+] and E[(-D)+] for D ~ N(d, s^2)
    pos = s * norm.pdf(d / s) + d * norm.cdf(d / s)
    neg = s * norm.pdf(d / s) - d * norm.cdf(-d / s)
    z = norm.isf((1 - credible) / 2)
    rng = np.random.default_rng(seed)
    mc = rng.normal(mean_c, se_c, MC_DRAWS)
    mt = rng.normal(mean_t, se_t, MC_DRAWS)
    q = [(1 - credible) / 2, (1 + credible) / 2]
    rel_ci = tuple(float(x) for x in np.quantile(mt / mc - 1, q))
    better = p_up if higher_is_better else 1 - p_up
    ship_loss, keep_loss = (neg, pos) if higher_is_better else (pos, neg)
    return BayesResult(
        method="normal_flat_prior",
        prob_treatment_better=better,
        expected_loss_ship=float(ship_loss),
        expected_loss_keep=float(keep_loss),
        posterior_mean_control=mean_c,
        posterior_mean_treatment=mean_t,
        abs_credible_interval=(d - z * s, d + z * s),
        rel_credible_interval=rel_ci,  # type: ignore[arg-type]
        credible_level=credible,
    )
