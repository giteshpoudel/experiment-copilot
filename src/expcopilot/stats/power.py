"""Sample size and minimum detectable effect for two-arm tests (normal approximation).

`ratio` is n_treatment / n_control throughout. Sizes are for the control arm.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def _z(alpha: float, power: float, sides: int) -> tuple[float, float]:
    if not (0 < alpha < 1 and 0 < power < 1):
        raise ValueError("alpha and power must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    return float(norm.isf(alpha / sides)), float(norm.ppf(power))


def n_binary_exact(
    p_control: float,
    mde_rel: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    sides: int = 2,
) -> float:
    """Control-arm n (unrounded) to detect p_control -> p_control * (1 + mde_rel).

    Uses the pooled variance under H0 and the unpooled variance under H1.
    """
    p1, p2 = p_control, p_control * (1 + mde_rel)
    if not (0 < p1 < 1 and 0 < p2 < 1) or mde_rel == 0:
        raise ValueError("need 0 < p < 1 under both hypotheses and a non-zero MDE")
    z_a, z_b = _z(alpha, power, sides)
    pbar = (p1 + ratio * p2) / (1 + ratio)
    sd_null = math.sqrt(pbar * (1 - pbar) * (1 + 1 / ratio))
    sd_alt = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    return ((z_a * sd_null + z_b * sd_alt) / (p2 - p1)) ** 2


def n_per_arm_binary(p_control: float, mde_rel: float, **kw: float) -> int:
    return math.ceil(n_binary_exact(p_control, mde_rel, **kw))


def n_continuous_exact(
    sd: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    sides: int = 2,
) -> float:
    """Control-arm n (unrounded) to detect an absolute difference `mde_abs` with unit sd `sd`.

    Also serves ratio metrics when `sd` is the sd of the linearized metric.
    """
    if sd <= 0 or mde_abs == 0:
        raise ValueError("need sd > 0 and a non-zero MDE")
    z_a, z_b = _z(alpha, power, sides)
    return (z_a + z_b) ** 2 * sd**2 * (1 + 1 / ratio) / mde_abs**2


def n_per_arm_continuous(sd: float, mde_abs: float, **kw: float) -> int:
    return math.ceil(n_continuous_exact(sd, mde_abs, **kw))


def mde_continuous(
    sd: float,
    n_control: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    sides: int = 2,
) -> float:
    """Smallest absolute difference detectable with `n_control` units in control."""
    z_a, z_b = _z(alpha, power, sides)
    return (z_a + z_b) * sd * math.sqrt((1 + 1 / ratio) / n_control)


def mde_binary(
    p_control: float,
    n_control: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    sides: int = 2,
) -> float:
    """Smallest positive relative lift detectable with `n_control` units in control."""
    upper = (1 - 1e-9) / p_control - 1

    def gap(m: float) -> float:
        return n_binary_exact(p_control, m, alpha, power, ratio, sides) - n_control

    if gap(upper) > 0:
        raise ValueError("no detectable lift below p = 1 at this sample size")
    return float(brentq(gap, 1e-9, upper, xtol=1e-12))
