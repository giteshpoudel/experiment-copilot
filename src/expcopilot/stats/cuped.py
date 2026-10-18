"""CUPED: adjust outcomes by a pre-period covariate, then compare arms with Welch.

theta is the pooled within-arm regression slope of y on x, so the adjusted difference equals
the treatment coefficient of an OLS regression y ~ 1 + treat + x exactly. Centering x at its
overall mean keeps the adjusted arm means on the original scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expcopilot.stats.frequentist import EffectEstimate, welch_from_stats


@dataclass(frozen=True)
class CupedResult:
    estimate: EffectEstimate
    theta: float
    covariate_correlation: float
    variance_reduction: float


def cuped(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray, alpha: float = 0.05
) -> CupedResult:
    y_c, x_c, y_t, x_t = (np.asarray(a, float) for a in (y_c, x_c, y_t, x_t))
    if y_c.shape != x_c.shape or y_t.shape != x_t.shape:
        raise ValueError("outcome and covariate lengths differ")
    dx_c, dx_t = x_c - x_c.mean(), x_t - x_t.mean()
    sxx = (dx_c**2).sum() + (dx_t**2).sum()
    if sxx == 0:
        raise ValueError("covariate has no variance")
    sxy = (dx_c * (y_c - y_c.mean())).sum() + (dx_t * (y_t - y_t.mean())).sum()
    theta = float(sxy / sxx)

    x_bar = np.concatenate([x_c, x_t]).mean()
    adj_c, adj_t = y_c - theta * (x_c - x_bar), y_t - theta * (x_t - x_bar)
    n_c, n_t = y_c.size, y_t.size
    raw = y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t
    adj = adj_c.var(ddof=1) / n_c + adj_t.var(ddof=1) / n_t
    reduction = float(1 - adj / raw) if raw > 0 else 0.0
    est = welch_from_stats(
        adj_c.mean(),
        adj_c.var(ddof=1),
        n_c,
        adj_t.mean(),
        adj_t.var(ddof=1),
        n_t,
        alpha,
        method="cuped_welch",
        variance_reduction=reduction,
    )
    corr = float(np.corrcoef(np.concatenate([x_c, x_t]), np.concatenate([y_c, y_t]))[0, 1])
    return CupedResult(est, theta, corr, reduction)
