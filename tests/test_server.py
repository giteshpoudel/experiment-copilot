"""MCP server tests through FastMCP's in-memory client, on the real smoke suite.

The truth store is written to a separate directory, so the server's data directory never
contains the answers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from expcopilot import results as R
from expcopilot.server import build_server
from expcopilot.simulate import run_suite
from expcopilot.store import Store, TruthStore

TOOLS = {
    "list_experiments",
    "describe_experiment",
    "check_srm",
    "power_analysis",
    "analyze_metric",
    "adjust_pvalues",
    "sequential_test",
    "bayesian_ab",
    "segment_effects",
    "daily_effects",
    "uplift_model",
    "calculate",
}
TRUTH_WORDS = ("knob", "drop_share", "true_effect", "initial_relative", "effect:", "bug:")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("server")
    run_suite("smoke", Store(root / "data"), TruthStore(root / "truth"))
    return root / "data"


@pytest.fixture(scope="module")
def server(data_dir: Path):
    return build_server(Store(data_dir, read_only=True))


def run(server, coro_fn) -> Any:
    async def go() -> Any:
        async with Client(server) as c:
            return await coro_fn(c)

    return asyncio.run(go())


def call(server, tool: str, **args: Any) -> dict[str, Any]:
    async def go(c: Client) -> dict[str, Any]:
        res = await c.call_tool(tool, args)
        return res.structured_content

    out = run(server, go)
    text = json.dumps(out)
    assert not any(w in text for w in TRUTH_WORDS), f"{tool} leaked truth: {text[:200]}"
    return out


def call_error(server, tool: str, **args: Any) -> str:
    async def go(c: Client) -> str:
        res = await c.call_tool(tool, args, raise_on_error=False)
        assert res.is_error
        return res.content[0].text

    return run(server, go)


# --- shape -----------------------------------------------------------------------------------


def test_tool_inventory_is_read_only_and_typed(server) -> None:
    tools = run(server, lambda c: c.list_tools())
    assert {t.name for t in tools} == TOOLS
    for t in tools:
        assert t.annotations.read_only_hint is True, t.name
        assert t.annotations.destructive_hint is False, t.name
        assert t.output_schema, t.name
        assert "Use " in t.description, f"{t.name} docstring should say when to use it"
        assert "data" not in t.input_schema.get("properties", {}), "tools never take raw data"


def test_resources_and_prompt(server) -> None:
    async def go(c: Client):
        resources = {str(r.uri) for r in await c.list_resources()}
        templates = {t.uri_template for t in await c.list_resource_templates()}
        registry = json.loads((await c.read_resource("experiments://registry"))[0].text)
        plan = json.loads((await c.read_resource("experiments://smoke-17/plan"))[0].text)
        metrics = json.loads((await c.read_resource("metrics://definitions"))[0].text)
        prompt = await c.get_prompt("experiment_readout", {"experiment_id": "smoke-03"})
        return resources, templates, registry, plan, metrics, prompt

    resources, templates, registry, plan, metrics, prompt = run(server, go)
    assert resources == {"experiments://registry", "metrics://definitions"}
    assert templates == {"experiments://{experiment_id}/plan"}
    assert registry["total"] == 20
    assert plan["guardrails"] == [{"metric": "bounce"}]
    assert plan["decision_rule"]["guardrail_check"] == "no_significant_harm"
    assert metrics["ratio"]["auto_method"] == "delta_method"
    text = prompt.messages[0].content.text
    assert "smoke-03" in text and "INVALID" in text and "result_id" in text


# --- every tool, with known answers on the smoke suite ---------------------------------------


def test_list_and_describe(server) -> None:
    out = R.ListExperimentsResult.model_validate(call(server, "list_experiments"))
    assert out.total == 20 and not out.truncated and len(out.experiments) == 20
    d = R.DescribeExperimentResult.model_validate(
        call(server, "describe_experiment", experiment_id="smoke-02")
    )
    assert d.n_total == 12_000 and d.plan.planned_n_total == 30_000
    assert d.plan.primary_metric == "conversion"
    assert {m.name: m.default_method for m in d.metrics}["revenue_per_session"] == "delta_method"


def test_srm_fails_only_where_injected(server) -> None:
    bad = R.SRMCheckResult.model_validate(call(server, "check_srm", experiment_id="smoke-15"))
    good = R.SRMCheckResult.model_validate(call(server, "check_srm", experiment_id="smoke-03"))
    assert not bad.passed and bad.p_value < 0.001 and "FAILED" in bad.summary
    assert good.passed


def test_analyze_metric_picks_method_by_type(server) -> None:
    ratio = R.AnalyzeMetricResult.model_validate(
        call(server, "analyze_metric", experiment_id="smoke-10", metric="revenue_per_session")
    )
    assert ratio.method == "delta_method" and ratio.metric_type == "ratio"
    binary = call(server, "analyze_metric", experiment_id="smoke-04", metric="conversion")
    assert binary["method"] == "two_proportion_z"
    assert binary["comparisons"][0]["direction"] == "better"
    assert binary["role"] == "primary"


def test_cuped_reduces_variance(server) -> None:
    out = R.AnalyzeMetricResult.model_validate(
        call(
            server,
            "analyze_metric",
            experiment_id="smoke-13",
            metric="revenue",
            cuped_covariate="pre_revenue",
        )
    )
    assert out.method == "cuped_welch"
    assert out.comparisons[0].variance_reduction > 0.5


def test_guardrail_verdict(server) -> None:
    out = R.AnalyzeMetricResult.model_validate(
        call(server, "analyze_metric", experiment_id="smoke-17", metric="bounce")
    )
    comp = out.comparisons[0]
    assert out.role == "guardrail" and comp.guardrail is not None
    assert comp.guardrail.passed is False and comp.direction == "worse"


def test_inconclusive_guardrail_passes(server) -> None:
    """Significant-harm rule: smoke-03's bounce rose a little (true effect 0) but not
    significantly, so the guardrail passes."""
    out = R.AnalyzeMetricResult.model_validate(
        call(server, "analyze_metric", experiment_id="smoke-03", metric="bounce")
    )
    comp = out.comparisons[0]
    assert comp.direction == "inconclusive" and comp.guardrail is not None
    assert comp.guardrail.passed and "no statistically significant harm" in comp.summary


def test_multi_arm_and_adjust(server) -> None:
    out = call(server, "analyze_metric", experiment_id="smoke-19", metric="conversion")
    assert [c["arm"] for c in out["comparisons"]] == ["variant_b", "variant_c", "variant_d"]

    async def go(c: Client):
        first = await c.call_tool(
            "analyze_metric", {"experiment_id": "smoke-19", "metric": "conversion"}
        )
        rid = first.structured_content["result_id"]
        return (await c.call_tool("adjust_pvalues", {"result_ids": [rid]})).structured_content

    adj = R.AdjustPValuesResult.model_validate(run(server, go))
    assert len(adj.entries) == 3
    assert all(e.p_adjusted >= e.p_value for e in adj.entries)


def test_result_ids_are_deterministic(server) -> None:
    a = call(server, "analyze_metric", experiment_id="smoke-03", metric="conversion")
    b = call(server, "analyze_metric", experiment_id="smoke-03", metric="conversion")
    assert a["result_id"] == b["result_id"]
    assert a["comparisons"][0]["result_id"] == f"{a['result_id']}.treatment"


def test_sequential_tests(server) -> None:
    obf = R.SequentialResult.model_validate(
        call(server, "sequential_test", experiment_id="smoke-18", metric="conversion")
    )
    assert len(obf.looks) == 14 and not obf.stopped
    assert obf.looks[0].information_fraction < obf.looks[-1].information_fraction
    ms = R.SequentialResult.model_validate(
        call(
            server, "sequential_test", experiment_id="smoke-18", metric="conversion", method="msprt"
        )
    )
    assert ms.method == "msprt" and len(ms.looks) == 14
    # The novelty experiment "stops" early: the early spike is real in the data.
    nov = call(
        server, "sequential_test", experiment_id="smoke-16", metric="conversion", looks=[3, 7, 14]
    )
    assert nov["stopped"] and nov["stopped_at_day"] == 3


def test_daily_effects_flags_novelty(server) -> None:
    out = R.DailyEffectsResult.model_validate(
        call(server, "daily_effects", experiment_id="smoke-16", metric="conversion")
    )
    assert out.decaying and len(out.days) == 14 and out.slope_per_day < 0
    assert "no daily data" in call_error(
        server, "daily_effects", experiment_id="smoke-03", metric="conversion"
    )


def test_bayesian(server) -> None:
    b = R.BayesianResult.model_validate(
        call(server, "bayesian_ab", experiment_id="smoke-04", metric="conversion")
    )
    assert b.prob_treatment_better > 0.99 and b.prior == [1.0, 1.0]
    lower = call(server, "bayesian_ab", experiment_id="smoke-17", metric="bounce")
    assert lower["prob_treatment_better"] < 0.01  # bounce went up: treatment is worse
    cont = call(server, "bayesian_ab", experiment_id="smoke-10", metric="revenue_per_session")
    assert cont["method"] == "normal_flat_prior"


def test_segment_effects(server) -> None:
    out = R.SegmentEffectsResult.model_validate(
        call(
            server,
            "segment_effects",
            experiment_id="smoke-14",
            metric="conversion",
            segment="platform",
        )
    )
    assert out.exploratory and {lv.level for lv in out.levels} == {"ios", "android", "web"}
    assert "Exploratory" in out.summary


def test_power_and_calculate(server) -> None:
    pw = R.PowerResult.model_validate(
        call(server, "power_analysis", metric_type="binary", baseline=0.1, mde=0.1)
    )
    assert pw.n_per_arm_control == 14_751
    calc = R.CalculateResult.model_validate(
        call(server, "calculate", operation="pct_change", a=0.1, b=0.11)
    )
    assert calc.value == pytest.approx(0.1) and calc.summary.endswith("+10%")


# --- errors and isolation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("describe_experiment", {"experiment_id": "nope"}, "known ids: smoke-01"),
        ("analyze_metric", {"experiment_id": "smoke-01", "metric": "gmv"}, "metrics: conversion"),
        (
            "analyze_metric",
            {
                "experiment_id": "smoke-10",
                "metric": "revenue_per_session",
                "cuped_covariate": "pre_revenue",
            },
            "not supported for ratio",
        ),
        (
            "analyze_metric",
            {"experiment_id": "smoke-01", "metric": "conversion", "arms": ["control"]},
            "treatment arms: treatment",
        ),
        ("adjust_pvalues", {"result_ids": ["r-00000000"]}, "unknown result id"),
        (
            "sequential_test",
            {"experiment_id": "smoke-01", "metric": "conversion"},
            "no interim looks",
        ),
        (
            "segment_effects",
            {"experiment_id": "smoke-01", "metric": "conversion", "segment": "browser"},
            "segments: platform",
        ),
        ("power_analysis", {"metric_type": "continuous", "baseline": 5.0, "mde": 0.1}, "needs sd"),
        ("calculate", {"operation": "divide", "a": 1.0, "b": 0.0}, "division by zero"),
    ],
)
def test_errors_tell_the_model_what_to_do(server, tool: str, args: dict, expected: str) -> None:
    assert expected in call_error(server, tool, **args)


def test_server_data_dir_has_no_truth(data_dir: Path) -> None:
    assert not (data_dir / "truth.duckdb").exists()
    source = Path(__file__).parents[1] / "src" / "expcopilot"
    for name in ("server.py", "analysis.py", "results.py"):
        assert "TruthStore" not in (source / name).read_text(), name


def test_read_only_store_refuses_writes(data_dir: Path, tmp_path: Path) -> None:
    store = Store(data_dir, read_only=True)
    with pytest.raises(PermissionError):
        store.delete("smoke-01")
    with pytest.raises(FileNotFoundError):
        Store(tmp_path / "missing", read_only=True)
    assert not (tmp_path / "missing").exists()


def test_uplift_model_finds_the_segment_with_the_effect(server) -> None:
    """smoke-14: the whole simulated effect is on iOS, so ranking by platform should work."""
    out = R.UpliftModelResult.model_validate(
        call(
            server,
            "uplift_model",
            experiment_id="smoke-14",
            outcome="conversion",
            features=["platform"],
        )
    )
    assert out.exploratory and len(out.deciles) == 10
    assert out.qini_coefficient is not None and out.qini_coefficient > 0
    assert "unknown features" in call_error(
        server,
        "uplift_model",
        experiment_id="smoke-14",
        outcome="conversion",
        features=["converted"],
    )
