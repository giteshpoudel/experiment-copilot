"""Effect by day and a trend test, to spot novelty (or learning) effects.

Given one EffectEstimate per day, regress the daily relative effect on day with weights
1/se^2 (weighted least squares, residual scale estimated). A significant negative slope while
the overall effect is positive is the novelty pattern: early lift that fades.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist

from expcopilot.stats.frequentist import EffectEstimate


@dataclass(frozen=True)
class DailyEffects:
    method: str
    days: tuple[int, ...]
    rel_effects: tuple[float, ...]
    rel_ses: tuple[float, ...]
    slope_per_day: float
    slope_se: float
    slope_p_value: float
    intercept: float
    alpha: float

    @property
    def decaying(self) -> bool:
        """Effect shrinks over time at the given alpha."""
        return self.slope_p_value < self.alpha and self.slope_per_day < 0


def weighted_trend(
    x: np.ndarray, y: np.ndarray, se: np.ndarray
) -> tuple[float, float, float, float]:
    """WLS of y on [1, x] with weights 1/se^2: (intercept, slope, slope_se, two-sided p)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), 1 / np.asarray(se, float) ** 2
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 days for a trend test")
    sw = w.sum()
    xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    slope = (w * (x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    resid = y - intercept - slope * x
    scale = (w * resid**2).sum() / (n - 2)
    slope_se = float(np.sqrt(scale / sxx))
    p = float(2 * t_dist.sf(abs(slope / slope_se), n - 2)) if slope_se > 0 else 1.0
    return float(intercept), float(slope), slope_se, p


def daily_effects(
    days: Sequence[int], estimates: Sequence[EffectEstimate], alpha: float = 0.05
) -> DailyEffects:
    if len(days) != len(estimates):
        raise ValueError("need one estimate per day")
    rel = np.array([e.rel_effect for e in estimates])
    se = np.array([e.rel_se for e in estimates])
    ok = np.isfinite(rel) & np.isfinite(se) & (se > 0)
    d = np.asarray(days)[ok]
    intercept, slope, slope_se, p = weighted_trend(d, rel[ok], se[ok])
    return DailyEffects(
        method="wls_trend_on_daily_relative_effect",
        days=tuple(int(x) for x in d),
        rel_effects=tuple(rel[ok].tolist()),
        rel_ses=tuple(se[ok].tolist()),
        slope_per_day=slope,
        slope_se=slope_se,
        slope_p_value=p,
        intercept=intercept,
        alpha=alpha,
    )
