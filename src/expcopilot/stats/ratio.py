"""Delta method for ratio metrics: sum(numerator) / sum(denominator) per arm.

The randomization unit (user) is not the denominator unit (session), so per-session values are
not independent and a naive t-test on them understates the variance. Linearize instead:
Var(R) ~= Var(y - R x) / (n * mean(x)^2).
"""

from __future__ import annotations

import numpy as np

from expcopilot.stats.frequentist import EffectEstimate, from_arm_means


def ratio_and_variance(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Ratio of sums and the delta-method variance of that ratio (ddof=1)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.shape != den.shape or num.size < 2:
        raise ValueError("numerator and denominator need the same length, at least 2 units")
    xbar = den.mean()
    if xbar == 0:
        raise ValueError("denominator sums to zero")
    r = num.sum() / den.sum()
    resid = num - r * den
    return float(r), float(resid.var(ddof=1) / (num.size * xbar**2))


def delta_method(
    num_c: np.ndarray,
    den_c: np.ndarray,
    num_t: np.ndarray,
    den_t: np.ndarray,
    alpha: float = 0.05,
) -> EffectEstimate:
    r_c, v_c = ratio_and_variance(num_c, den_c)
    r_t, v_t = ratio_and_variance(num_t, den_t)
    return from_arm_means("delta_method", len(num_c), len(num_t), r_c, r_t, v_c, v_t, alpha)
