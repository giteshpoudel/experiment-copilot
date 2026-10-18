from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from expcopilot.models import (
    Arm,
    Experiment,
    MetricDef,
    MetricType,
    PreRegisteredPlan,
    SegmentSpec,
    Source,
)

ARMS = [Arm(name="control", expected_share=0.5, is_control=True), Arm(name="b", expected_share=0.5)]
METRICS = [
    MetricDef(name="conversion", type=MetricType.BINARY, column="converted"),
    MetricDef(name="rps", type=MetricType.RATIO, numerator="revenue", denominator="sessions"),
]
PLAN = PreRegisteredPlan(primary_metric="conversion", mde=0.1, planned_n_total=1000)


def make(**kw: object) -> Experiment:
    fields: dict[str, object] = {
        "experiment_id": "exp-1",
        "name": "test",
        "source": Source.SIMULATED,
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 1, 14),
        "arms": ARMS,
        "metrics": METRICS,
        "plan": PLAN,
    }
    fields.update(kw)
    return Experiment.model_validate(fields)


def test_valid_experiment_round_trips_through_json() -> None:
    exp = make()
    assert Experiment.model_validate_json(exp.model_dump_json()) == exp
    assert exp.control.name == "control"
    assert exp.unit_columns == ["unit_id", "arm", "assigned_at", "converted", "revenue", "sessions"]


def test_shares_must_sum_to_one() -> None:
    arms = [
        Arm(name="control", expected_share=0.5, is_control=True),
        Arm(name="b", expected_share=0.4),
    ]
    with pytest.raises(ValidationError, match="sum to 1"):
        make(arms=arms)


def test_exactly_one_control() -> None:
    arms = [Arm(name="a", expected_share=0.5), Arm(name="b", expected_share=0.5)]
    with pytest.raises(ValidationError, match="control"):
        make(arms=arms)


def test_plan_metrics_must_exist() -> None:
    plan = PreRegisteredPlan(primary_metric="revenue", mde=0.1, planned_n_total=1000)
    with pytest.raises(ValidationError, match="undefined metrics"):
        make(plan=plan)


def test_ratio_metric_needs_numerator_and_denominator() -> None:
    with pytest.raises(ValidationError, match="numerator"):
        MetricDef(name="rps", type=MetricType.RATIO, column="revenue")
    with pytest.raises(ValidationError, match="column only"):
        MetricDef(name="c", type=MetricType.BINARY)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Arm(name="a", expected_share=0.5, colour="red")


def test_segment_multipliers_normalize_to_mean_one() -> None:
    seg = SegmentSpec(
        name="platform",
        levels={"ios": 0.45, "android": 0.35, "web": 0.20},
        effect_multipliers={"ios": 1.0, "android": 0.0, "web": 0.0},
    )
    norm = seg.normalized_multipliers()
    assert sum(seg.levels[k] * v for k, v in norm.items()) == pytest.approx(1.0)
    assert norm["ios"] == pytest.approx(1 / 0.45)


def test_decision_rule_text_names_every_outcome() -> None:
    text = PLAN.decision_rule.describe()
    for word in ("invalid", "ship", "extend", "don't ship"):
        assert word in text
