"""Pydantic outputs for every MCP tool, plus number formatting.

Every result carries a `result_id`, the `method` and the `inputs` it ran on (from
`models.ToolResult`). Numbers are rounded to 6 significant figures, and each result has a
preformatted `summary` the agent can quote verbatim, so it never has to turn 0.0873 into
"8.73%" itself (that would be prose doing arithmetic).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from expcopilot.models import Model, PreRegisteredPlan, ToolResult

SIG_FIGS = 6
MAX_LIST = 50


# --- formatting ------------------------------------------------------------------------------


def sig(x: float | None, digits: int = SIG_FIGS) -> float | None:
    """Round to `digits` significant figures; None, nan and inf pass through as None/inf."""
    if x is None or math.isnan(x):
        return None
    if math.isinf(x) or x == 0:
        return float(x)
    return float(f"{x:.{digits}g}")


def fmt_num(x: float, digits: int = 4) -> str:
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:,.{digits}g}" if abs(x) >= 1e-4 or x == 0 else f"{x:.{digits}g}"


def fmt_signed(x: float, digits: int = 4) -> str:
    return ("+" if x > 0 else "") + fmt_num(x, digits)


def fmt_pct(x: float, digits: int = 3, signed: bool = True) -> str:
    """0.0873 -> '+8.73%'."""
    if math.isnan(x):
        return "n/a"
    s = f"{x * 100:.{digits}g}%"
    return ("+" + s) if signed and x > 0 else s


def fmt_prob(p: float) -> str:
    """Probabilities near 0 or 1 are shown as bounds, never as a flat 0% or 100%."""
    if p > 0.9999:
        return "> 99.99%"
    if p < 0.0001:
        return "< 0.01%"
    return fmt_pct(p, 4, signed=False)


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "p = n/a"
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.2g}"


# --- shared pieces ---------------------------------------------------------------------------


class Interval(Model):
    low: float | None
    high: float | None


class ExperimentRow(Model):
    experiment_id: str
    name: str
    status: str
    n_units: int


class ListExperimentsResult(ToolResult):
    experiments: list[ExperimentRow]
    total: int
    truncated: bool
    summary: str


class ArmInfo(Model):
    name: str
    is_control: bool
    expected_share: float
    observed_n: int


class MetricInfo(Model):
    name: str
    type: str
    higher_is_better: bool
    columns: list[str]
    default_method: str
    description: str


class DescribeExperimentResult(ToolResult):
    experiment_id: str
    name: str
    description: str
    status: str
    unit: str
    start_date: str
    end_date: str
    has_daily: bool
    arms: list[ArmInfo]
    n_total: int
    metrics: list[MetricInfo]
    covariates: list[str]
    segments: list[str]
    plan: PreRegisteredPlan
    decision_rule_text: str
    summary: str


class SRMArm(Model):
    arm: str
    expected_share: float
    observed_share: float
    observed: int
    expected: float


class SRMCheckResult(ToolResult):
    experiment_id: str
    arms: list[SRMArm]
    chi2: float
    df: int
    p_value: float
    alpha: float
    passed: bool
    summary: str


class PowerResult(ToolResult):
    metric_type: str
    baseline: float
    alpha: float
    power: float
    ratio: float
    n_per_arm_control: int
    n_per_arm_treatment: int
    mde_relative: float
    mde_absolute: float
    summary: str


class GuardrailCheck(Model):
    rule: Literal["no_significant_harm"] = Field(
        description="Fails only when the CI lies entirely on the harmful side of zero"
    )
    passed: bool


class Comparison(Model):
    result_id: str = Field(description="Cite this id; adjust_pvalues accepts it")
    arm: str
    control: str
    n_control: int
    n_treatment: int
    mean_control: float | None
    mean_treatment: float | None
    abs_effect: float | None
    abs_ci: Interval
    rel_effect: float | None
    rel_ci: Interval
    p_value: float | None
    significant: bool
    direction: Literal["better", "worse", "inconclusive"] = Field(
        description="From the CI at the plan's alpha, using the metric's higher_is_better"
    )
    variance_reduction: float | None = None
    guardrail: GuardrailCheck | None = None
    summary: str


class AnalyzeMetricResult(ToolResult):
    experiment_id: str
    metric: str
    metric_type: str
    role: Literal["primary", "secondary", "guardrail", "unplanned"]
    alpha: float
    cuped_covariate: str | None
    comparisons: list[Comparison]
    notes: list[str]


class AdjustedEntry(Model):
    result_id: str
    label: str
    p_value: float
    p_adjusted: float
    significant_adjusted: bool


class AdjustPValuesResult(ToolResult):
    alpha: float
    entries: list[AdjustedEntry]
    summary: str


class Look(Model):
    day: int
    information_fraction: float
    n_control: int
    n_treatment: int
    statistic: float | None = Field(description="z (O'Brien-Fleming) or always-valid p (mSPRT)")
    boundary: float | None
    crossed: bool


class SequentialResult(ToolResult):
    experiment_id: str
    metric: str
    arm: str
    alpha: float
    looks: list[Look]
    stopped: bool
    stopped_at_day: int | None
    assumptions: list[str]
    summary: str


class BayesianResult(ToolResult):
    experiment_id: str
    metric: str
    arm: str
    prob_treatment_better: float
    expected_loss_ship: float
    expected_loss_keep: float
    abs_credible_interval: Interval
    rel_credible_interval: Interval
    credible_level: float
    prior: list[float] | None
    summary: str


class SegmentLevel(Model):
    level: str
    n_control: int
    n_treatment: int
    rel_effect: float | None
    rel_ci: Interval
    abs_effect: float | None
    p_value: float | None
    p_adjusted: float
    summary: str


class SegmentEffectsResult(ToolResult):
    experiment_id: str
    metric: str
    segment: str
    arm: str
    adjust_method: str
    levels: list[SegmentLevel]
    skipped_levels: list[str]
    heterogeneity_q: float
    heterogeneity_p: float
    exploratory: bool = True
    summary: str


class DayEffect(Model):
    day: int
    rel_effect: float | None
    rel_ci: Interval
    p_value: float | None


class DailyEffectsResult(ToolResult):
    experiment_id: str
    metric: str
    arm: str
    days: list[DayEffect]
    slope_per_day: float
    slope_p_value: float
    decaying: bool
    summary: str


class CalculateResult(ToolResult):
    operation: str
    a: float
    b: float
    value: float
    summary: str
