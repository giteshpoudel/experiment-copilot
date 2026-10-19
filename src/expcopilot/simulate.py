"""Simulated experiments with known true effects.

Every simulated experiment has the same five metrics, generated per unit from a few latent
variables so pre-period covariates correlate with outcomes (the CUPED knob):

    z_eng   engagement   -> pre_sessions, sessions, bounce (negatively)
    z_spend spend level  -> pre_revenue, conversion, order value

Current-period latents are rho * z + sqrt(1 - rho^2) * noise, so `covariate_rho` is the
latent (Gaussian-copula) correlation; the realized Pearson correlation is lower for binary
and skewed outcomes.

Treatment effects are relative changes in population means and are exact by construction:

    conversion   P(convert) = p0 * (1 + e_conv)
    revenue      converted * order_value * k, with k chosen so E[revenue] = base * (1 + e_rev)
                 (conversion and order value are correlated, so k uses a closed form)
    sessions     1 + Poisson(...), with E[sessions] = lambda * (1 + e_sess)
    revenue_per_session   ratio of the two means: (1 + e_rev) / (1 + e_sess) - 1
    bounce       P(bounce) = b0 * (1 + e_bounce)

Segment multipliers (normalized to mean 1) scale each unit's effect, so heterogeneity moves the
effect between segments without changing its population average. Probabilities are clipped to
(0, 1), which only matters for extreme specs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
from scipy.special import ndtr, ndtri

from expcopilot.models import (
    Arm,
    EffectSpec,
    Experiment,
    GuardrailSpec,
    MetricDef,
    MetricType,
    NoveltyBug,
    PreRegisteredPlan,
    SegmentSpec,
    SimulationSpec,
    SimulationTruth,
    Source,
    SRMBug,
    Status,
)
from expcopilot.store import Store, TruthStore

P_EPS = 1e-6

SIM_METRICS = [
    MetricDef(
        name="conversion",
        type=MetricType.BINARY,
        column="converted",
        description="Unit made at least one purchase during the experiment",
    ),
    MetricDef(
        name="revenue",
        type=MetricType.CONTINUOUS,
        column="revenue",
        description="Revenue per unit over the experiment (zero for non-buyers; skewed)",
    ),
    MetricDef(
        name="sessions",
        type=MetricType.CONTINUOUS,
        column="sessions",
        description="Sessions per unit over the experiment",
    ),
    MetricDef(
        name="revenue_per_session",
        type=MetricType.RATIO,
        numerator="revenue",
        denominator="sessions",
        description="Total revenue / total sessions (ratio metric; unit is the user)",
    ),
    MetricDef(
        name="bounce",
        type=MetricType.BINARY,
        column="bounced",
        higher_is_better=False,
        description="Unit's first session ended without a second page view",
    ),
]
SIM_COVARIATES = ["pre_revenue", "pre_sessions"]

DEFAULT_SEGMENTS = [
    SegmentSpec(name="platform", levels={"ios": 0.45, "android": 0.35, "web": 0.20}),
    SegmentSpec(name="region", levels={"na": 0.5, "eu": 0.3, "apac": 0.2}),
    SegmentSpec(name="new_user", levels={"yes": 0.3, "no": 0.7}),
]


@dataclass(frozen=True)
class SimulatedExperiment:
    experiment: Experiment
    units: pa.Table
    daily: pa.Table | None
    truth: SimulationTruth


# --- truth -----------------------------------------------------------------------------------


def _revenue_tilt(spec: SimulationSpec, threshold: np.ndarray | float) -> np.ndarray | float:
    """E[exp(sigma * v) * 1{u > t}] / exp(sigma^2 / 2) for standard normals u, v with corr c.

    Exponential tilting shifts u's mean by c * sigma, so this is Phi(c * sigma - t).
    Here c = rho^2 (u and v both load rho on z_spend).
    """
    c = spec.covariate_rho**2
    return ndtr(c * spec.revenue_sigma - threshold)


def baseline(spec: SimulationSpec, metric: str) -> float:
    """Expected control-arm mean of a metric."""
    revenue = math.exp(spec.revenue_mu + spec.revenue_sigma**2 / 2) * float(
        _revenue_tilt(spec, float(ndtri(1 - spec.baseline_conversion)))
    )
    return {
        "conversion": spec.baseline_conversion,
        "revenue": revenue,
        "sessions": spec.sessions_lambda,
        "revenue_per_session": revenue / spec.sessions_lambda,
        "bounce": spec.baseline_bounce,
    }[metric]


def true_effects(spec: SimulationSpec) -> list[EffectSpec]:
    """Steady-state relative effect on every metric for every treatment arm."""
    out = []
    for arm in spec.arms:
        if arm.is_control:
            continue
        e = {m: spec.effect(m, arm.name) for m in ("conversion", "revenue", "sessions", "bounce")}
        e["revenue_per_session"] = (1 + e["revenue"]) / (1 + e["sessions"]) - 1
        out.extend(EffectSpec(metric=m.name, arm=arm.name, relative=e[m.name]) for m in SIM_METRICS)
    return out


# --- generation ------------------------------------------------------------------------------


def _clip_p(p: np.ndarray) -> np.ndarray:
    return np.clip(p, P_EPS, 1 - P_EPS)


def _novelty_path(
    novelty: NoveltyBug, steady: np.ndarray, initial: np.ndarray, days: int
) -> np.ndarray:
    """Per-unit, per-day conversion effect, shape (n, days).

    effect(d) = steady + (initial - steady) * 0.5**(d / h). Both endpoints are per unit,
    already scaled by the unit's segment multiplier and zero for control units.
    """
    decay = np.array([0.5 ** (d / novelty.half_life_days) for d in range(days)])
    return steady[:, None] + (initial - steady)[:, None] * decay[None, :]


def generate(spec: SimulationSpec, suite: str = "adhoc") -> SimulatedExperiment:
    rng = np.random.default_rng(spec.seed)
    n, arms, days = spec.n_total, spec.arms, spec.days
    rho = spec.covariate_rho
    w = math.sqrt(1 - rho**2)

    shares = np.array([a.expected_share for a in arms])
    arm_idx = rng.choice(len(arms), size=n, p=shares / shares.sum())
    arm_names = np.array([a.name for a in arms])

    segment_cols: dict[str, np.ndarray] = {}
    mult = np.ones(n)
    for seg in spec.segments:
        levels = list(seg.levels)
        idx = rng.choice(len(levels), size=n, p=[seg.levels[lvl] for lvl in levels])
        segment_cols[seg.name] = np.array(levels)[idx]
        norm = seg.normalized_multipliers()
        mult *= np.array([norm[lvl] for lvl in levels])[idx]

    def unit_effect(metric: str) -> np.ndarray:
        per_arm = np.array([spec.effect(metric, a.name) for a in arms])
        return per_arm[arm_idx] * mult

    e_conv, e_rev, e_sess, e_bounce = (
        unit_effect(m) for m in ("conversion", "revenue", "sessions", "bounce")
    )
    is_treated = ~np.array([a.is_control for a in arms])[arm_idx]

    novelty_path = None
    if spec.novelty is not None:
        # Conversion effect per day; the unit-level effect is its average over the window.
        initial = np.where(is_treated, spec.novelty.initial_relative * mult, 0.0)
        novelty_path = _novelty_path(spec.novelty, e_conv, initial, days)
        e_conv_unit = novelty_path.mean(axis=1)
    else:
        e_conv_unit = e_conv

    z_eng = rng.standard_normal(n)
    z_spend = rng.standard_normal(n)

    def current(z: np.ndarray) -> np.ndarray:
        return rho * z + w * rng.standard_normal(n)

    lam, mu, sigma = spec.sessions_lambda, spec.revenue_mu, spec.revenue_sigma
    pre_revenue = np.exp(mu + sigma * z_spend)
    pre_sessions = 1 + rng.poisson((lam - 1) * np.exp(0.5 * z_eng - 0.125))

    p0 = spec.baseline_conversion
    p_conv = _clip_p(p0 * (1 + e_conv_unit))
    threshold = ndtri(1 - p_conv)
    converted = current(z_spend) > threshold
    order_value = np.exp(mu + sigma * current(z_spend))
    # Conversion and order value share z_spend (latent corr rho^2), so buyers spend more.
    # Scale revenue so E[revenue] = baseline * (1 + e_rev) exactly; see _revenue_tilt.
    k = (
        _revenue_tilt(spec, ndtri(1 - p0))
        * np.maximum(1 + e_rev, 0)
        / _revenue_tilt(spec, threshold)
    )
    revenue = np.where(converted, order_value * k, 0.0)

    rate = np.maximum(lam * (1 + e_sess) - 1, 0) * np.exp(0.5 * current(z_eng) - 0.125)
    sessions = 1 + rng.poisson(rate)

    p_bounce = _clip_p(spec.baseline_bounce * (1 + e_bounce))
    bounced = current(-z_eng) > ndtri(1 - p_bounce)

    if spec.daily:
        assign_day = np.zeros(n, dtype=np.int32)
    else:
        assign_day = rng.integers(0, days, size=n, dtype=np.int32)
    start = np.datetime64(datetime.combine(spec.start_date, datetime.min.time()), "s")
    offset = assign_day.astype(np.int64) * 86_400 + rng.integers(0, 86_400, size=n)
    assigned_at = start + offset.astype("timedelta64[s]")

    unit_id = np.arange(1, n + 1, dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    if spec.srm is not None:
        target = arm_names[arm_idx] == spec.srm.arm
        keep &= ~(target & (rng.random(n) < spec.srm.drop_share))

    daily = None
    if spec.daily:
        daily = _daily_rows(
            rng, spec, unit_id, arm_names[arm_idx], converted, revenue, novelty_path, keep
        )

    columns: dict[str, np.ndarray] = {
        "unit_id": unit_id,
        "arm": arm_names[arm_idx],
        "assigned_at": assigned_at,
        "assign_day": assign_day,
        **segment_cols,
        "pre_revenue": pre_revenue,
        "pre_sessions": pre_sessions.astype(np.int64),
        "converted": converted.astype(np.int8),
        "revenue": revenue,
        "sessions": sessions.astype(np.int64),
        "bounced": bounced.astype(np.int8),
    }
    units = pa.table({name: col[keep] for name, col in columns.items()})

    plan = spec.plan.model_copy(update={"baseline": baseline(spec, spec.plan.primary_metric)})
    experiment = Experiment(
        experiment_id=spec.experiment_id,
        name=spec.name,
        description=spec.description,
        status=Status.COMPLETE,
        source=Source.SIMULATED,
        unit="user",
        start_date=spec.start_date,
        end_date=spec.start_date + timedelta(days=days - 1),
        arms=spec.arms,
        metrics=SIM_METRICS,
        covariates=SIM_COVARIATES,
        segments=[s.name for s in spec.segments],
        has_daily=spec.daily,
        plan=plan,
    )
    truth = SimulationTruth(
        experiment_id=spec.experiment_id,
        suite=suite,
        knobs=spec.knobs,
        true_effects=true_effects(spec),
        spec=spec,
    )
    return SimulatedExperiment(experiment=experiment, units=units, daily=daily, truth=truth)


def _daily_rows(
    rng: np.random.Generator,
    spec: SimulationSpec,
    unit_id: np.ndarray,
    arm: np.ndarray,
    converted: np.ndarray,
    revenue: np.ndarray,
    novelty_path: np.ndarray | None,
    keep: np.ndarray,
) -> pa.Table:
    """One row per unit per day. Each converter's purchase lands on one day.

    The day is drawn with weights proportional to 1 + effect(d), which makes the treatment
    arm's daily conversion rate p0 * (1 + effect(d)) / days, so the daily effect is exactly
    effect(d) and the novelty decay shows up in a by-day analysis.
    """
    n, days = len(unit_id), spec.days
    weights = np.ones((n, days)) if novelty_path is None else np.maximum(1 + novelty_path, 0)
    cum = np.cumsum(weights, axis=1)
    draw = rng.random(n) * cum[:, -1]
    buy_day = (cum < draw[:, None]).sum(axis=1)

    ids, arms = unit_id[keep], arm[keep]
    conv_mask, rev, day_of = converted[keep], revenue[keep], buy_day[keep]
    m = len(ids)
    day = np.tile(np.arange(days, dtype=np.int32), m)
    hit = np.repeat(conv_mask, days) & (day == np.repeat(day_of, days))
    return pa.table(
        {
            "unit_id": np.repeat(ids, days),
            "arm": np.repeat(arms, days),
            "day": day,
            "converted": hit.astype(np.int8),
            "revenue": np.where(hit, np.repeat(rev, days), 0.0),
        }
    )


# --- suites ----------------------------------------------------------------------------------

TWO_ARMS = [
    Arm(name="control", expected_share=0.5, is_control=True),
    Arm(name="treatment", expected_share=0.5),
]
FOUR_ARMS = [
    Arm(name="control", expected_share=0.25, is_control=True),
    Arm(name="variant_b", expected_share=0.25),
    Arm(name="variant_c", expected_share=0.25),
    Arm(name="variant_d", expected_share=0.25),
]

BOUNCE_GUARDRAIL = GuardrailSpec(metric="bounce")


def _plan(primary: str = "conversion", **kw: object) -> PreRegisteredPlan:
    defaults: dict[str, object] = {
        "primary_metric": primary,
        "secondary_metrics": ["revenue"] if primary == "conversion" else ["conversion"],
        "guardrails": [BOUNCE_GUARDRAIL],
        "mde": 0.10,
        "planned_n_total": 30_000,
    }
    defaults.update(kw)
    return PreRegisteredPlan.model_validate(defaults)


def _spec(i: int, name: str, description: str, knobs: list[str], **kw: object) -> SimulationSpec:
    fields: dict[str, object] = {
        "experiment_id": f"smoke-{i:02d}",
        "name": name,
        "description": description,
        "knobs": knobs,
        "seed": 1000 + i,
        "n_total": 30_000,
        "arms": TWO_ARMS,
        "segments": DEFAULT_SEGMENTS,
        "plan": _plan(),
    }
    fields.update(kw)
    return SimulationSpec.model_validate(fields)


def _effects(arm: str = "treatment", **by_metric: float) -> list[EffectSpec]:
    return [EffectSpec(metric=m, arm=arm, relative=v) for m, v in by_metric.items()]


def smoke_suite() -> list[SimulationSpec]:
    """20 experiments that together exercise every simulator knob at least once.

    Names and descriptions are deliberately neutral: they are what the agent sees.
    Planned sizes give 80% power at the MDE (alpha 0.05 two-sided, no CUPED, two arms),
    rounded up to the next 1,000; `test_planned_n_matches_power` recomputes them from a pilot.
    """

    def revenue_plan(n: int) -> PreRegisteredPlan:
        return _plan("revenue", mde=0.15, planned_n_total=n)

    rps_plan = _plan(
        "revenue_per_session",
        mde=0.15,
        planned_n_total=34_000,
        secondary_metrics=["conversion", "sessions"],
    )
    return [
        _spec(
            1,
            "checkout-button-copy",
            "New wording on the checkout button.",
            ["metric:binary", "effect:null"],
        ),
        _spec(
            2,
            "free-shipping-banner",
            "Banner announcing free shipping over $50.",
            ["metric:binary", "effect:below_mde", "underpowered"],
            n_total=12_000,
            effects=_effects(conversion=0.05),
        ),
        _spec(
            3,
            "one-page-checkout",
            "Single-page checkout instead of three steps.",
            ["metric:binary", "effect:at_mde"],
            effects=_effects(conversion=0.10),
        ),
        _spec(
            4,
            "saved-cards",
            "Offer saved payment cards at checkout.",
            ["metric:binary", "effect:large"],
            effects=_effects(conversion=0.30),
        ),
        _spec(
            5,
            "mandatory-account",
            "Require an account before checkout.",
            ["metric:binary", "effect:negative"],
            effects=_effects(conversion=-0.10),
        ),
        _spec(
            6,
            "recommendation-carousel",
            "Product recommendations on the cart page.",
            ["metric:continuous", "effect:null"],
            n_total=34_000,
            plan=revenue_plan(34_000),
        ),
        _spec(
            7,
            "bundle-discounts",
            "Discount when buying two related items.",
            ["metric:continuous", "effect:large"],
            n_total=34_000,
            plan=revenue_plan(34_000),
            effects=_effects(revenue=0.40),
        ),
        _spec(
            8,
            "search-ranking-v2",
            "New ranking model for site search.",
            ["metric:ratio", "effect:null"],
            n_total=34_000,
            plan=rps_plan,
        ),
        _spec(
            9,
            "premium-upsell",
            "Upsell premium versions on product pages.",
            ["metric:ratio", "effect:at_mde"],
            n_total=34_000,
            plan=rps_plan,
            effects=_effects(revenue=0.15),
        ),
        _spec(
            10,
            "infinite-scroll",
            "Infinite scroll on category pages.",
            ["metric:ratio", "effect:ratio_denominator_drop", "metric_conflict"],
            n_total=34_000,
            plan=rps_plan,
            effects=_effects(conversion=-0.10, sessions=-0.12),
        ),
        _spec(
            11,
            "loyalty-points-a",
            "Show loyalty points balance in the header.",
            ["metric:continuous", "effect:at_mde", "cuped:rho_0.0"],
            n_total=36_000,
            plan=revenue_plan(36_000),
            covariate_rho=0.0,
            effects=_effects(revenue=0.15),
        ),
        _spec(
            12,
            "loyalty-points-b",
            "Show loyalty points balance on product pages.",
            ["metric:continuous", "effect:at_mde", "cuped:rho_0.5"],
            n_total=34_000,
            plan=revenue_plan(34_000),
            covariate_rho=0.5,
            effects=_effects(revenue=0.15),
        ),
        _spec(
            13,
            "loyalty-points-c",
            "Show loyalty points balance at checkout.",
            ["metric:continuous", "effect:at_mde", "cuped:rho_0.9"],
            n_total=22_000,
            plan=revenue_plan(22_000),
            covariate_rho=0.9,
            effects=_effects(revenue=0.15),
        ),
        _spec(
            14,
            "apple-pay-button",
            "Express pay button at the top of checkout.",
            ["metric:binary", "effect:at_mde", "heterogeneity"],
            effects=_effects(conversion=0.10),
            segments=[
                DEFAULT_SEGMENTS[0].model_copy(
                    update={"effect_multipliers": {"ios": 1.0, "android": 0.0, "web": 0.0}}
                ),
                *DEFAULT_SEGMENTS[1:],
            ],
        ),
        _spec(
            15,
            "address-autocomplete",
            "Autocomplete for shipping address fields.",
            ["metric:binary", "effect:at_mde", "bug:srm"],
            effects=_effects(conversion=0.10),
            srm=SRMBug(arm="treatment", drop_share=0.06),
        ),
        _spec(
            16,
            "homepage-redesign",
            "Redesigned homepage hero and navigation.",
            ["metric:binary", "effect:null", "bug:novelty", "daily"],
            daily=True,
            novelty=NoveltyBug(initial_relative=0.40, half_life_days=2.0),
        ),
        _spec(
            17,
            "aggressive-popups",
            "Discount pop-up on first visit.",
            ["metric:binary", "effect:large", "bug:guardrail_harm"],
            effects=_effects(conversion=0.20, bounce=0.15),
        ),
        _spec(
            18,
            "trust-badges",
            "Security badges next to the pay button.",
            ["metric:binary", "effect:null", "peeking"],
            plan=_plan(planned_looks=list(range(1, 15))),
        ),
        _spec(
            19,
            "pricing-page-layouts",
            "Three alternative pricing page layouts.",
            ["metric:binary", "multi_arm", "effect:at_mde"],
            arms=FOUR_ARMS,
            n_total=60_000,
            plan=_plan(planned_n_total=60_000),
            effects=_effects("variant_c", conversion=0.10),
        ),
        _spec(
            20,
            "product-video",
            "Autoplay product video on product pages.",
            ["metric:binary", "multi_metric", "effect:below_mde"],
            plan=_plan(secondary_metrics=["revenue", "sessions", "revenue_per_session"]),
            effects=_effects(conversion=0.03, sessions=0.05),
        ),
    ]


SUITES: dict[str, Callable[[], list[SimulationSpec]]] = {"smoke": smoke_suite}


@dataclass(frozen=True)
class SuiteRun:
    created: list[str]
    skipped: list[str]


def run_suite(
    name: str, store: Store, truth_store: TruthStore, overwrite: bool = False
) -> SuiteRun:
    """Generate a suite into the stores. Existing experiments are skipped unless overwrite."""
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; choose from {sorted(SUITES)}")
    created, skipped = [], []
    for spec in SUITES[name]():
        if store.exists(spec.experiment_id) and not overwrite:
            skipped.append(spec.experiment_id)
            continue
        sim = generate(spec, suite=name)
        store.save(sim.experiment, sim.units, sim.daily, overwrite=overwrite)
        truth_store.save(sim.truth)
        created.append(spec.experiment_id)
    return SuiteRun(created=created, skipped=skipped)
