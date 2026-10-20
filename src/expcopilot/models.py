"""Pydantic models for experiments, pre-registered plans, simulation specs and tool results.

Everything a tool or the agent can see about an experiment is an `Experiment`. The simulator's
knobs and known true effects live in `SimulationSpec` / `SimulationTruth`, which are stored
separately (see `store.TruthStore`) so the analysis side never reads them.
"""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EXPERIMENT_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
SHARE_TOLERANCE = 1e-6


class Model(BaseModel):
    """Base for all models: unknown fields are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricType(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    RATIO = "ratio"


class Source(StrEnum):
    SIMULATED = "simulated"
    HILLSTROM = "hillstrom"
    CRITEO = "criteo"


class Status(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"


Decision = Literal["ship", "dont_ship", "extend", "invalid"]
MultipleTesting = Literal["none", "holm", "bh"]


# --- Experiment definition -------------------------------------------------------------------


class Arm(Model):
    name: str = Field(min_length=1)
    expected_share: float = Field(gt=0, lt=1)
    is_control: bool = False


class MetricDef(Model):
    """A metric and the unit-table column(s) it is computed from.

    Binary and continuous metrics read one `column`. Ratio metrics (e.g. revenue per session)
    are sum(numerator) / sum(denominator) over units and need the delta method.
    """

    name: str = Field(min_length=1)
    type: MetricType
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    higher_is_better: bool = True
    description: str = ""

    @model_validator(mode="after")
    def _columns_match_type(self) -> MetricDef:
        if self.type is MetricType.RATIO:
            if not (self.numerator and self.denominator) or self.column:
                raise ValueError(
                    f"ratio metric {self.name!r} needs numerator and denominator, not column"
                )
        elif not self.column or self.numerator or self.denominator:
            raise ValueError(f"{self.type} metric {self.name!r} needs column only")
        return self

    @property
    def columns(self) -> list[str]:
        if self.type is MetricType.RATIO:
            return [self.numerator, self.denominator]  # type: ignore[list-item]
        return [self.column]  # type: ignore[list-item]


class GuardrailSpec(Model):
    """A metric that must not get significantly worse.

    "Worse" follows the metric's `higher_is_better`. The guardrail fails only when the effect's
    CI at the plan's alpha lies entirely on the harmful side of zero (statistically significant
    harm); an inconclusive or improving guardrail passes.
    """

    metric: str


class DecisionRule(Model):
    """The pre-registered decision rule, applied in this order:

    1. `invalid`   if the SRM chi-square p-value < `srm_alpha`
    2. `ship`      if the primary metric's CI (at the plan's alpha, after any multiple-testing
                   adjustment) lies entirely on the good side of zero, and no guardrail's CI lies
                   entirely on the harmful side of zero
    3. `extend`    if `extend_if_underpowered` and the primary result is inconclusive and the
                   observed sample is below the plan's `planned_n_total`
    4. `dont_ship` otherwise

    With several treatment arms, an arm qualifies if its primary p-value, adjusted with the
    plan's method, is below alpha and the effect is on the good side; the best qualifying arm
    whose guardrails pass is shipped. With planned interim looks, the primary counts as
    significant only if the O'Brien-Fleming boundary was crossed at some look. If the plan names
    a `primary_covariate`, the primary analysis uses CUPED with it.

    The fields are structured (not prose) so the Phase 5 oracle can apply the rule mechanically.
    """

    srm_alpha: float = Field(default=0.001, gt=0, lt=1)
    ship_requires: Literal["primary_ci_excludes_zero"] = "primary_ci_excludes_zero"
    guardrail_check: Literal["no_significant_harm"] = "no_significant_harm"
    extend_if_underpowered: bool = True

    def describe(self) -> str:
        parts = [
            f"invalid if SRM p < {self.srm_alpha:g}",
            "ship if the primary CI excludes zero on the good side (after multiple-testing "
            "adjustment across arms; with interim looks, only if the O'Brien-Fleming boundary "
            "was crossed) and no guardrail shows statistically significant harm",
        ]
        if self.extend_if_underpowered:
            parts.append("extend if inconclusive and n < planned n")
        parts.append("otherwise don't ship")
        return "; ".join(parts)


class PreRegisteredPlan(Model):
    """What was decided before the experiment started. `mde` is relative (0.10 = 10% lift)."""

    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    guardrails: list[GuardrailSpec] = Field(default_factory=list)
    baseline: float | None = Field(default=None, description="Expected control mean of primary")
    mde: float = Field(gt=0, lt=10)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    power: float = Field(default=0.8, gt=0, lt=1)
    sides: Literal[1, 2] = 2
    multiple_testing: MultipleTesting = "holm"
    planned_n_total: int = Field(gt=0)
    planned_looks: list[int] = Field(
        default_factory=list, description="Days (1-based) at which interim looks are planned"
    )
    primary_covariate: str | None = Field(
        default=None, description="Pre-period covariate for CUPED on the primary metric"
    )
    decision_rule: DecisionRule = Field(default_factory=DecisionRule)

    @property
    def metric_names(self) -> list[str]:
        return [self.primary_metric, *self.secondary_metrics, *(g.metric for g in self.guardrails)]


class Experiment(Model):
    """The registry entry for one experiment: everything the analysis side is allowed to see."""

    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    name: str
    description: str = ""
    status: Status = Status.COMPLETE
    source: Source
    unit: str = "user"
    start_date: date
    end_date: date
    arms: list[Arm] = Field(min_length=2)
    metrics: list[MetricDef] = Field(min_length=1)
    covariates: list[str] = Field(default_factory=list, description="Pre-period columns")
    segments: list[str] = Field(default_factory=list, description="Segment columns")
    has_daily: bool = False
    plan: PreRegisteredPlan

    @model_validator(mode="after")
    def _consistent(self) -> Experiment:
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        if sum(a.is_control for a in self.arms) != 1:
            raise ValueError("exactly one arm must be the control")
        total = sum(a.expected_share for a in self.arms)
        if not math.isclose(total, 1.0, abs_tol=SHARE_TOLERANCE):
            raise ValueError(f"expected arm shares must sum to 1, got {total:.6f}")
        metric_names = [m.name for m in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric names must be unique")
        missing = [m for m in self.plan.metric_names if m not in metric_names]
        if missing:
            raise ValueError(f"plan refers to undefined metrics: {missing}")
        if self.end_date < self.start_date:
            raise ValueError("end_date is before start_date")
        cov = self.plan.primary_covariate
        if cov is not None and cov not in self.covariates:
            raise ValueError(f"plan's primary_covariate {cov!r} is not a covariate")
        return self

    @property
    def control(self) -> Arm:
        return next(a for a in self.arms if a.is_control)

    def metric(self, name: str) -> MetricDef:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(f"experiment {self.experiment_id!r} has no metric {name!r}")

    @property
    def unit_columns(self) -> list[str]:
        """Columns the unit table must contain."""
        cols = ["unit_id", "arm", "assigned_at", *self.segments, *self.covariates]
        for m in self.metrics:
            cols.extend(c for c in m.columns if c not in cols)
        return cols


class ExperimentSummary(Model):
    experiment_id: str
    name: str
    status: Status
    source: Source
    n_units: int


# --- Simulation ------------------------------------------------------------------------------


class EffectSpec(Model):
    """True relative effect of `arm` vs control on the population mean of `metric`."""

    metric: str
    arm: str
    relative: float = Field(gt=-1)


class SegmentSpec(Model):
    """A segment column with level shares and optional effect multipliers per level.

    Multipliers are rescaled so their share-weighted mean is 1, so heterogeneity moves the
    effect between segments without changing the population-average effect.
    """

    name: str
    levels: dict[str, float]
    effect_multipliers: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid(self) -> SegmentSpec:
        total = sum(self.levels.values())
        if not math.isclose(total, 1.0, abs_tol=SHARE_TOLERANCE):
            raise ValueError(f"segment {self.name!r} level shares must sum to 1, got {total}")
        unknown = set(self.effect_multipliers) - set(self.levels)
        if unknown:
            raise ValueError(f"segment {self.name!r} multipliers for unknown levels: {unknown}")
        if any(v < 0 for v in self.effect_multipliers.values()):
            raise ValueError("effect multipliers must be >= 0")
        return self

    def normalized_multipliers(self) -> dict[str, float]:
        raw = {lvl: self.effect_multipliers.get(lvl, 1.0) for lvl in self.levels}
        mean = sum(self.levels[lvl] * m for lvl, m in raw.items())
        if mean == 0:
            raise ValueError(f"segment {self.name!r} multipliers average to zero")
        return {lvl: m / mean for lvl, m in raw.items()}


class SRMBug(Model):
    """Sample-ratio mismatch: after randomization, silently drop a share of one arm's units."""

    arm: str
    drop_share: float = Field(gt=0, lt=1)


class NoveltyBug(Model):
    """Novelty effect on conversion that decays toward the steady-state (true) effect.

    On day d (0-based) the relative effect is steady + (initial - steady) * 0.5**(d / h).
    Additive decay, so a zero steady-state effect can still look positive early on.
    """

    initial_relative: float = Field(gt=-1)
    half_life_days: float = Field(gt=0)


class SimulationSpec(Model):
    """Every simulator knob for one experiment. `plan.baseline` is filled in by the simulator."""

    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    name: str
    description: str = ""
    knobs: list[str] = Field(default_factory=list, description="Tags for eval coverage")
    seed: int
    n_total: int = Field(gt=0)
    arms: list[Arm] = Field(min_length=2)
    days: int = Field(default=14, ge=1)
    start_date: date = date(2026, 9, 1)
    # Baseline process (control arm)
    baseline_conversion: float = Field(default=0.10, gt=0, lt=1)
    revenue_mu: float = 3.5
    revenue_sigma: float = Field(default=1.0, gt=0)
    sessions_lambda: float = Field(default=4.0, gt=1)
    baseline_bounce: float = Field(default=0.30, gt=0, lt=1)
    covariate_rho: float = Field(default=0.5, ge=0, lt=1)
    # Truth and bugs
    effects: list[EffectSpec] = Field(default_factory=list)
    segments: list[SegmentSpec] = Field(default_factory=list)
    srm: SRMBug | None = None
    novelty: NoveltyBug | None = None
    daily: bool = False
    plan: PreRegisteredPlan

    @field_validator("knobs")
    @classmethod
    def _sorted_knobs(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _consistent(self) -> SimulationSpec:
        arm_names = {a.name for a in self.arms}
        control = [a.name for a in self.arms if a.is_control]
        if len(control) != 1:
            raise ValueError("exactly one arm must be the control")
        for e in self.effects:
            if e.arm not in arm_names:
                raise ValueError(f"effect for unknown arm {e.arm!r}")
            if e.arm == control[0]:
                raise ValueError("effects are relative to control; none may target the control")
        if self.srm and self.srm.arm not in arm_names:
            raise ValueError(f"SRM bug targets unknown arm {self.srm.arm!r}")
        if self.novelty and not self.daily:
            raise ValueError("a novelty effect needs daily=True")
        seen = {(e.metric, e.arm) for e in self.effects}
        if len(seen) != len(self.effects):
            raise ValueError("duplicate (metric, arm) effect")
        return self

    def effect(self, metric: str, arm: str) -> float:
        for e in self.effects:
            if e.metric == metric and e.arm == arm:
                return e.relative
        return 0.0


class SimulationTruth(Model):
    """Ground truth for one simulated experiment. Never exposed to tools or the agent."""

    experiment_id: str
    suite: str
    knobs: list[str]
    true_effects: list[EffectSpec] = Field(
        description="Steady-state population relative effects, every metric x treatment arm"
    )
    spec: SimulationSpec


# --- Tool results (shared shape for Phases 2-3) ----------------------------------------------


class ToolResult(Model):
    """Base for every tool output: citable id, the method used, and the inputs it ran on."""

    result_id: str
    method: str
    inputs: dict[str, object]
