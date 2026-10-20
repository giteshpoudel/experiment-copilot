"""The 80-experiment eval suite: every simulator knob, several times, with neutral names.

Ids are `e001`...`e080` and names are random-looking product-feature names, so nothing the agent
sees hints at the knob or the answer. Planned sample sizes come from `stats.power` at 80% power
for the plan's MDE, rounded up to the next 1,000.
"""

from __future__ import annotations

import math

from expcopilot.models import (
    Arm,
    EffectSpec,
    NoveltyBug,
    PreRegisteredPlan,
    SimulationSpec,
    SRMBug,
)
from expcopilot.simulate import BOUNCE_GUARDRAIL, DEFAULT_SEGMENTS, FOUR_ARMS, TWO_ARMS
from expcopilot.stats.power import n_binary_exact

SEED_BASE = 50_000
BINARY_MDE, REVENUE_MDE = 0.10, 0.15
# Power-derived totals (see test_planned_n_matches_power for the method).
BINARY_N, REVENUE_N = {0.5: 30_000}, {0.0: 36_000, 0.5: 34_000, 0.9: 22_000}
RATIO_N = 34_000

_ADJ = [
    "amber",
    "brisk",
    "cobalt",
    "dusky",
    "ember",
    "fable",
    "gilded",
    "hazel",
    "ivory",
    "jade",
    "keen",
    "lunar",
    "maple",
    "nimble",
    "onyx",
    "pearl",
]
_NOUN = [
    "arch",
    "beacon",
    "canopy",
    "delta",
    "echo",
    "fern",
    "grove",
    "harbor",
    "isle",
    "juniper",
    "kite",
    "lantern",
    "meadow",
    "nova",
    "orbit",
    "prism",
    "quill",
    "ridge",
    "summit",
    "tide",
]


def _name(i: int) -> str:
    return f"{_ADJ[(i * 7) % len(_ADJ)]}-{_NOUN[(i * 11) % len(_NOUN)]}"


def _plan(primary: str, **kw: object) -> PreRegisteredPlan:
    base: dict[str, object] = {
        "primary_metric": primary,
        "guardrails": [BOUNCE_GUARDRAIL],
        "secondary_metrics": ["revenue"] if primary == "conversion" else ["conversion"],
    }
    if primary == "conversion":
        base |= {"mde": BINARY_MDE, "planned_n_total": BINARY_N[0.5]}
    elif primary == "revenue":
        base |= {
            "mde": REVENUE_MDE,
            "planned_n_total": REVENUE_N[0.5],
            "primary_covariate": "pre_revenue",
        }
    else:
        base |= {
            "mde": REVENUE_MDE,
            "planned_n_total": RATIO_N,
            "secondary_metrics": ["conversion", "sessions"],
        }
    base.update(kw)
    return PreRegisteredPlan.model_validate(base)


def _fx(arm: str = "treatment", **by_metric: float) -> list[EffectSpec]:
    return [EffectSpec(metric=m, arm=arm, relative=v) for m, v in by_metric.items() if v != 0]


class _Builder:
    def __init__(self) -> None:
        self.specs: list[SimulationSpec] = []

    def add(self, knobs: list[str], **kw: object) -> None:
        i = len(self.specs) + 1
        plan = kw.pop("plan", None) or _plan("conversion")
        fields: dict[str, object] = {
            "experiment_id": f"e{i:03d}",
            "name": _name(i),
            "description": "Product change under test.",
            "knobs": knobs,
            "seed": SEED_BASE + i,
            "n_total": plan.planned_n_total,  # type: ignore[union-attr]
            "arms": TWO_ARMS,
            "segments": DEFAULT_SEGMENTS,
            "plan": plan,
        }
        fields.update(kw)
        self.specs.append(SimulationSpec.model_validate(fields))


EFFECT_LEVELS = {"null": 0.0, "below_mde": 0.5, "at_mde": 1.0, "large": 3.0, "negative": -1.0}


def eval_suite() -> list[SimulationSpec]:
    b = _Builder()

    # A. Effect size x metric type, two seeds each (30).
    for _seed in range(2):
        for level, mult in EFFECT_LEVELS.items():
            b.add(
                ["metric:binary", f"effect:{level}"],
                plan=_plan("conversion"),
                effects=_fx(conversion=BINARY_MDE * mult),
            )
            b.add(
                ["metric:continuous", f"effect:{level}", "cuped:rho_0.5"],
                plan=_plan("revenue"),
                effects=_fx(revenue=REVENUE_MDE * mult),
            )
            b.add(
                ["metric:ratio", f"effect:{level}"],
                plan=_plan("revenue_per_session"),
                effects=_fx(revenue=REVENUE_MDE * mult),
            )

    # B. CUPED strength, revenue at the MDE (9).
    for _seed in range(3):
        for rho in (0.0, 0.5, 0.9):
            b.add(
                ["metric:continuous", "effect:at_mde", f"cuped:rho_{rho}"],
                plan=_plan("revenue", planned_n_total=REVENUE_N[rho]),
                covariate_rho=rho,
                effects=_fx(revenue=REVENUE_MDE),
            )

    # C. Heterogeneity: the whole effect sits in one platform (3).
    for target in ("ios", "android", "web"):
        mult = {lvl: (1.0 if lvl == target else 0.0) for lvl in ("ios", "android", "web")}
        segs = [
            DEFAULT_SEGMENTS[0].model_copy(update={"effect_multipliers": mult}),
            *DEFAULT_SEGMENTS[1:],
        ]
        b.add(
            ["metric:binary", "effect:at_mde", "heterogeneity"],
            segments=segs,
            effects=_fx(conversion=BINARY_MDE),
        )

    # D. Sample-ratio mismatch: one arm silently loses units (5).
    for drop, effect in ((0.06, 1.0), (0.10, 1.0), (0.15, 1.0), (0.10, 0.0), (0.06, 3.0)):
        b.add(
            ["metric:binary", "bug:srm", f"effect:{'null' if effect == 0 else 'positive'}"],
            srm=SRMBug(arm="treatment", drop_share=drop),
            effects=_fx(conversion=BINARY_MDE * effect),
        )

    # E. Planned 20/80 split: must NOT be called an SRM (4).
    ratio = 4.0
    n_c = n_binary_exact(0.10, BINARY_MDE, ratio=ratio)
    n_unequal = math.ceil(n_c * (1 + ratio) / 1000) * 1000
    unequal = [
        Arm(name="control", expected_share=0.2, is_control=True),
        Arm(name="treatment", expected_share=0.8),
    ]
    for effect in (1.0, 0.0, 1.0, 3.0):
        b.add(
            [
                "metric:binary",
                "planned_imbalance",
                f"effect:{'null' if effect == 0 else 'positive'}",
            ],
            arms=unequal,
            plan=_plan("conversion", planned_n_total=n_unequal),
            n_total=n_unequal,
            effects=_fx(conversion=BINARY_MDE * effect),
        )

    # F. Novelty: early lift that decays; daily data available (5).
    for steady, initial, half_life in (
        (0.0, 0.40, 2.0),
        (0.0, 0.30, 3.0),
        (0.0, 0.50, 1.5),
        (BINARY_MDE, 0.40, 2.0),
        (BINARY_MDE * 3, 0.60, 2.0),
    ):
        b.add(
            [
                "metric:binary",
                "bug:novelty",
                "daily",
                "effect:null" if steady == 0 else "effect:positive",
            ],
            daily=True,
            novelty=NoveltyBug(initial_relative=initial, half_life_days=half_life),
            effects=_fx(conversion=steady),
        )

    # G. Large guardrail harm with a clear primary win (3).
    for _ in range(3):
        b.add(
            ["metric:binary", "effect:large", "bug:guardrail_harm"],
            effects=_fx(conversion=BINARY_MDE * 2, bounce=0.15),
        )

    # H. Small real guardrail harm: what "fail only on significant harm" can miss (3).
    for _ in range(3):
        b.add(
            ["metric:binary", "effect:large", "bug:guardrail_small_harm"],
            effects=_fx(conversion=BINARY_MDE * 2, bounce=0.03),
        )

    # I. Pre-registered daily interim looks (4).
    for level in ("null", "null", "at_mde", "large"):
        b.add(
            ["metric:binary", "peeking", f"effect:{level}"],
            plan=_plan("conversion", planned_looks=list(range(1, 15))),
            effects=_fx(conversion=BINARY_MDE * EFFECT_LEVELS[level]),
        )

    # J. Four arms, Holm across three comparisons (5).
    four = _plan("conversion", planned_n_total=60_000)
    for arm_fx in (
        {"variant_c": 1.0},
        {"variant_b": 1.0},
        {},
        {"variant_b": 3.0, "variant_d": -1.0},
        {"variant_c": 0.5},
    ):
        effects = [e for arm, m in arm_fx.items() for e in _fx(arm, conversion=BINARY_MDE * m)]
        b.add(
            [
                "metric:binary",
                "multi_arm",
                "effect:positive" if any(v > 0 for v in arm_fx.values()) else "effect:null",
            ],
            arms=FOUR_ARMS,
            plan=four,
            effects=effects,
        )

    # K. Several secondary metrics moving; the decision rests on the primary alone (3).
    multi = _plan("conversion", secondary_metrics=["revenue", "sessions", "revenue_per_session"])
    for conv, sess in ((0.03, 0.05), (0.0, 0.08), (BINARY_MDE * 2, -0.05)):
        b.add(
            ["metric:binary", "multi_metric", "effect:positive" if conv > 0 else "effect:null"],
            plan=multi,
            effects=_fx(conversion=conv, sessions=sess),
        )

    # L. Stopped early, well short of the planned n (3).
    for effect in (1.0, 1.0, 0.0):
        b.add(
            ["metric:binary", "underpowered", f"effect:{'null' if effect == 0 else 'at_mde'}"],
            n_total=12_000,
            effects=_fx(conversion=BINARY_MDE * effect),
        )

    # M. Revenue per session up only because sessions fell (3).
    for conv, sess in ((-0.10, -0.12), (0.0, -0.15), (-0.05, -0.10)):
        b.add(
            ["metric:ratio", "metric_conflict", "effect:ratio_denominator_drop"],
            plan=_plan("revenue_per_session"),
            effects=_fx(conversion=conv, sessions=sess),
        )

    # N. Top up with binary nulls and MDE-sized wins if the groups above total under 80.
    while len(b.specs) < 80:
        level = "null" if len(b.specs) % 2 == 0 else "at_mde"
        b.add(
            ["metric:binary", f"effect:{level}"],
            effects=_fx(conversion=BINARY_MDE * EFFECT_LEVELS[level]),
        )
    return b.specs
