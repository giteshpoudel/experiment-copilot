"""Heterogeneous treatment effects by segment (exploratory).

Per-segment effects use whichever estimator fits the metric, passed in as a callable, so this
module stays independent of metric types. p-values are adjusted across segment levels, and a
Cochran's Q test asks whether the effects differ by more than noise.

Uplift models (T-/X-learners, Qini) are added in Phase 6 with the Criteo case study.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from expcopilot.stats.frequentist import EffectEstimate
from expcopilot.stats.multiple import Method, adjust_pvalues

MAX_LEVELS = 20


@dataclass(frozen=True)
class SegmentEffect:
    level: str
    estimate: EffectEstimate
    p_adjusted: float


@dataclass(frozen=True)
class SegmentEffects:
    segment: str
    adjust_method: str
    effects: tuple[SegmentEffect, ...]
    skipped_levels: tuple[str, ...]  # too few units in an arm
    heterogeneity_q: float
    heterogeneity_df: int
    heterogeneity_p: float
    exploratory: bool = True


def cochran_q(effects: np.ndarray, ses: np.ndarray) -> tuple[float, int, float]:
    """Cochran's Q for equality of effects (inverse-variance weights)."""
    w = 1 / ses**2
    pooled = (w * effects).sum() / w.sum()
    q = float((w * (effects - pooled) ** 2).sum())
    df = effects.size - 1
    return q, df, float(chi2.sf(q, df)) if df > 0 else 1.0


def segment_effects(
    segment: str,
    labels: np.ndarray,
    is_treatment: np.ndarray,
    estimate: Callable[[np.ndarray, np.ndarray], EffectEstimate],
    adjust: Method = "holm",
    min_units_per_arm: int = 100,
) -> SegmentEffects:
    """Effect per segment level.

    `estimate(control_mask, treatment_mask)` returns an EffectEstimate for the units selected
    by the two boolean masks. Levels with fewer than `min_units_per_arm` units in either arm
    are skipped and listed.
    """
    labels = np.asarray(labels).astype(str)
    is_treatment = np.asarray(is_treatment, bool)
    levels = sorted(np.unique(labels).tolist())
    if len(levels) > MAX_LEVELS:
        raise ValueError(f"segment {segment!r} has {len(levels)} levels (max {MAX_LEVELS})")
    kept, skipped, estimates = [], [], []
    for level in levels:
        in_level = labels == level
        c, t = in_level & ~is_treatment, in_level & is_treatment
        if c.sum() < min_units_per_arm or t.sum() < min_units_per_arm:
            skipped.append(level)
            continue
        kept.append(level)
        estimates.append(estimate(c, t))
    if not estimates:
        raise ValueError(f"no level of {segment!r} has {min_units_per_arm} units per arm")
    p_adj = adjust_pvalues([e.p_value for e in estimates], adjust)
    q, df, q_p = cochran_q(
        np.array([e.abs_effect for e in estimates]), np.array([e.abs_se for e in estimates])
    )
    return SegmentEffects(
        segment=segment,
        adjust_method=adjust,
        effects=tuple(
            SegmentEffect(lvl, e, float(p))
            for lvl, e, p in zip(kept, estimates, p_adj, strict=True)
        ),
        skipped_levels=tuple(skipped),
        heterogeneity_q=q,
        heterogeneity_df=df,
        heterogeneity_p=q_p,
    )
