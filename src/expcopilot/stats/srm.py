"""Sample-ratio mismatch: chi-square goodness of fit of observed arm counts to planned shares."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    observed_shares: tuple[float, ...]
    chi2: float
    df: int
    p_value: float
    alpha: float
    passed: bool
    method: str = "chi_square_goodness_of_fit"


def srm_test(
    observed: Sequence[int], expected_shares: Sequence[float], alpha: float = 0.001
) -> SRMResult:
    obs = np.asarray(observed, dtype=float)
    shares = np.asarray(expected_shares, dtype=float)
    if obs.shape != shares.shape or obs.size < 2:
        raise ValueError("need one count and one expected share per arm (at least two arms)")
    if np.any(obs < 0) or np.any(shares <= 0):
        raise ValueError("counts must be >= 0 and shares > 0")
    total = obs.sum()
    if total == 0:
        raise ValueError("no units")
    expected = total * shares / shares.sum()
    stat = float(((obs - expected) ** 2 / expected).sum())
    df = obs.size - 1
    p = float(chi2.sf(stat, df))
    return SRMResult(
        observed=tuple(int(x) for x in obs),
        expected=tuple(float(x) for x in expected),
        observed_shares=tuple(float(x) for x in obs / total),
        chi2=stat,
        df=df,
        p_value=p,
        alpha=alpha,
        passed=p >= alpha,
    )
