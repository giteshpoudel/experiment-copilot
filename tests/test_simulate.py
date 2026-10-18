"""Simulator tests. Tolerances are ~4 standard errors for the sample sizes used."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from expcopilot.cli import app
from expcopilot.models import SimulationSpec
from expcopilot.simulate import baseline, generate, run_suite, smoke_suite
from expcopilot.store import Store, TruthStore

SUITE = {s.experiment_id: s for s in smoke_suite()}


def spec(eid: str, **update: object) -> SimulationSpec:
    return SUITE[eid].model_copy(update=update)


def col(sim, name: str) -> np.ndarray:
    return sim.units.column(name).to_numpy()


def rel_effect(sim, column: str, arm: str = "treatment") -> float:
    arms, x = col(sim, "arm"), col(sim, column).astype(float)
    return x[arms == arm].mean() / x[arms == "control"].mean() - 1


def test_smoke_suite_has_20_unique_neutral_experiments() -> None:
    specs = smoke_suite()
    assert len(specs) == 20
    assert len({s.experiment_id for s in specs}) == 20
    assert len({s.seed for s in specs}) == 20
    for s in specs:  # the agent sees names, ids and descriptions; none may leak the answer
        text = f"{s.experiment_id} {s.name} {s.description}".lower()
        assert not any(w in text for w in ("srm", "null", "novelty", "mde", "bug", "truth"))


def test_smoke_suite_covers_every_knob() -> None:
    knobs = {k for s in smoke_suite() for k in s.knobs}
    required = {
        "effect:null",
        "effect:below_mde",
        "effect:at_mde",
        "effect:large",
        "effect:negative",
        "metric:binary",
        "metric:continuous",
        "metric:ratio",
        "cuped:rho_0.0",
        "cuped:rho_0.5",
        "cuped:rho_0.9",
        "heterogeneity",
        "bug:srm",
        "bug:novelty",
        "bug:guardrail_harm",
        "peeking",
        "multi_arm",
        "multi_metric",
        "underpowered",
    }
    assert required <= knobs


def test_generation_is_deterministic() -> None:
    a = generate(spec("smoke-03", n_total=3_000))
    b = generate(spec("smoke-03", n_total=3_000))
    assert a.units.equals(b.units)
    assert a.experiment == b.experiment


def test_control_means_match_baseline() -> None:
    s = spec("smoke-01", n_total=200_000)
    sim = generate(s)
    control = col(sim, "arm") == "control"
    assert col(sim, "converted")[control].mean() == pytest.approx(
        baseline(s, "conversion"), rel=0.04
    )
    assert col(sim, "sessions")[control].mean() == pytest.approx(baseline(s, "sessions"), rel=0.01)
    assert col(sim, "bounced")[control].mean() == pytest.approx(baseline(s, "bounce"), rel=0.02)
    assert col(sim, "revenue")[control].mean() == pytest.approx(baseline(s, "revenue"), rel=0.06)


@pytest.mark.parametrize(
    ("eid", "column", "expected", "tol", "n"),
    [
        ("smoke-04", "converted", 0.30, 0.03, 200_000),
        ("smoke-05", "converted", -0.10, 0.03, 200_000),
        # Skewed revenue: sd of the estimate is ~0.023 at 200k, ~0.010 at 1M (see NOTES.md).
        ("smoke-07", "revenue", 0.40, 0.045, 1_000_000),
        ("smoke-10", "sessions", -0.12, 0.01, 200_000),
        ("smoke-17", "bounced", 0.15, 0.02, 200_000),
    ],
)
def test_realized_effects_match_truth(
    eid: str, column: str, expected: float, tol: float, n: int
) -> None:
    sim = generate(spec(eid, n_total=n))
    assert rel_effect(sim, column) == pytest.approx(expected, abs=tol)


def test_ratio_truth_combines_numerator_and_denominator() -> None:
    sim = generate(spec("smoke-10"))
    [rps] = [e for e in sim.truth.true_effects if e.metric == "revenue_per_session"]
    assert rps.relative == pytest.approx(1.0 / 0.88 - 1)


def test_srm_bug_removes_share_of_one_arm() -> None:
    sim = generate(spec("smoke-15"))
    arms = col(sim, "arm")
    ratio = (arms == "treatment").sum() / (arms == "control").sum()
    assert ratio == pytest.approx(0.94, abs=0.03)


def test_covariate_correlation_rises_with_rho() -> None:
    corrs = []
    for eid in ("smoke-11", "smoke-12", "smoke-13"):
        sim = generate(spec(eid, n_total=40_000))
        corrs.append(np.corrcoef(col(sim, "pre_revenue"), col(sim, "revenue"))[0, 1])
    assert abs(corrs[0]) < 0.03
    assert corrs[0] < corrs[1] < corrs[2]
    assert corrs[2] > 0.6


def test_heterogeneity_puts_effect_in_one_segment() -> None:
    sim = generate(spec("smoke-14", n_total=300_000))
    platform, arms, conv = col(sim, "platform"), col(sim, "arm"), col(sim, "converted")

    def lift(level: str) -> float:
        m = platform == level
        return conv[m & (arms == "treatment")].mean() / conv[m & (arms == "control")].mean() - 1

    assert lift("ios") == pytest.approx(0.10 / 0.45, abs=0.04)
    assert lift("android") == pytest.approx(0.0, abs=0.04)


def test_novelty_effect_decays_by_day() -> None:
    sim = generate(spec("smoke-16", n_total=300_000))
    assert sim.daily is not None
    d = sim.daily
    arm, day, conv = (d.column(c).to_numpy() for c in ("arm", "day", "converted"))
    by_day = [
        conv[(arm == "treatment") & (day == k)].mean()
        / conv[(arm == "control") & (day == k)].mean()
        - 1
        for k in range(14)
    ]
    assert np.mean(by_day[:2]) > 0.2
    assert abs(np.mean(by_day[-4:])) < 0.06
    # Daily rows add up to the unit table.
    assert d.num_rows == sim.units.num_rows * 14
    assert conv.sum() == col(sim, "converted").sum()


def test_peeking_assignments_are_staggered_with_planned_looks() -> None:
    sim = generate(spec("smoke-18", n_total=5_000))
    assert sim.experiment.plan.planned_looks == list(range(1, 15))
    assert set(np.unique(col(sim, "assign_day"))) == set(range(14))


def test_truth_never_appears_in_experiment() -> None:
    sim = generate(spec("smoke-15", n_total=1_000))
    dumped = sim.experiment.model_dump_json()
    assert "drop_share" not in dumped and "knobs" not in dumped


@pytest.mark.slow
def test_simulate_and_describe_cli(tmp_path) -> None:
    """Phase 1 "done when": the smoke suite creates 20 experiments and describe prints plans."""
    runner = CliRunner()
    data = str(tmp_path / "data")
    result = runner.invoke(app, ["simulate", "--suite", "smoke", "--data-dir", data])
    assert result.exit_code == 0, result.output
    assert "created 20" in result.output
    assert len(Store(data).list()) == 20

    again = runner.invoke(app, ["simulate", "--suite", "smoke", "--data-dir", data])
    assert "created 0, skipped 20" in again.output

    described = runner.invoke(app, ["describe", "--data-dir", data])
    assert described.exit_code == 0, described.output
    assert described.output.count("decision rule:") == 20
    assert "TRUTH" not in described.output


def test_run_suite_small(store: Store, truth_store: TruthStore, monkeypatch) -> None:
    from expcopilot import simulate as sim_mod

    small = [s.model_copy(update={"n_total": 500}) for s in smoke_suite()[:3]]
    monkeypatch.setitem(sim_mod.SUITES, "tiny", lambda: small)
    result = run_suite("tiny", store, truth_store)
    assert result.created == ["smoke-01", "smoke-02", "smoke-03"]
    assert truth_store.get("smoke-02").knobs == sorted(SUITE["smoke-02"].knobs)


@pytest.mark.slow
def test_planned_n_matches_power() -> None:
    """Each plan's planned_n_total is the power-derived n rounded up to the next 1,000.

    Variances come from a 400k-unit control-only pilot with a fixed seed.
    """
    from expcopilot.stats.power import n_per_arm_binary, n_per_arm_continuous

    required: dict[tuple[str, float], int] = {}
    for s in smoke_suite():
        p = s.plan
        key = (p.primary_metric, s.covariate_rho)
        if key not in required:
            pilot = generate(
                s.model_copy(
                    update={
                        "n_total": 400_000,
                        "effects": [],
                        "srm": None,
                        "novelty": None,
                        "daily": False,
                        "seed": 7,
                    }
                )
            ).units
            r = pilot.column("revenue").to_numpy()
            if p.primary_metric == "conversion":
                n = n_per_arm_binary(s.baseline_conversion, p.mde)
            elif p.primary_metric == "revenue":
                n = n_per_arm_continuous(r.std(ddof=1), r.mean() * p.mde)
            else:
                x = pilot.column("sessions").to_numpy()
                ratio = r.sum() / x.sum()
                lin = (r - ratio * x) / x.mean()
                n = n_per_arm_continuous(lin.std(ddof=1), ratio * p.mde)
            required[key] = n
        need = required[key] * len(s.arms)
        assert need <= p.planned_n_total < need + 1_000, (s.experiment_id, need, p.planned_n_total)
