"""FastMCP server: read-only experiment-analysis tools, resources and a readout prompt.

Tools take experiment ids and metric names, never data. Every output is a Pydantic model with
a `result_id` to cite, the method and the inputs. Docstrings below are what the model sees.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Literal, ParamSpec, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from expcopilot.analysis import DEFAULT_METHOD, METHOD_NAMES, AnalysisError, Analyzer
from expcopilot.models import MetricType
from expcopilot.results import (
    AdjustPValuesResult,
    AnalyzeMetricResult,
    BayesianResult,
    CalculateResult,
    DailyEffectsResult,
    DescribeExperimentResult,
    ListExperimentsResult,
    PowerResult,
    SegmentEffectsResult,
    SequentialResult,
    SRMCheckResult,
)
from expcopilot.store import Store

P = ParamSpec("P")
R = TypeVar("R")

READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

INSTRUCTIONS = """\
Tools for analyzing A/B tests stored in this server. Numbers come from tools: never compute,
round or convert a number yourself. Quote the `summary` strings or fields as returned and cite
each number with the result_id it came from. Use `calculate` for any arithmetic you need.
Start with describe_experiment, then check_srm, then the primary metric, then guardrails.
"""

ExperimentId = Annotated[str, Field(description="Experiment id from list_experiments")]
MetricName = Annotated[str, Field(description="Metric name from describe_experiment")]
ArmName = Annotated[
    str | None, Field(description="Treatment arm to compare with control; default: the first")
]

READOUT_PROMPT = """\
Write a ship / don't-ship readout for experiment {experiment_id}.

Process (use the tools in this order):
1. describe_experiment: note the arms, the primary metric, guardrails, MDE, alpha, planned n
   and the pre-registered decision rule.
2. check_srm: if it fails, the decision is INVALID; say so and stop analyzing effects.
3. analyze_metric on the primary metric. Use cuped_covariate if the experiment has a
   pre-period covariate for that metric (e.g. pre_revenue for revenue).
4. analyze_metric on every guardrail metric; each comparison reports whether it passed.
5. If there are several treatment arms, adjust_pvalues over the primary comparisons using the
   plan's multiple-testing method.
6. If the plan has interim looks, run sequential_test. If the experiment has daily data, run
   daily_effects on the primary metric to check for a novelty effect.
7. Optional: segment_effects, always labelled exploratory.

Decision: apply the plan's decision rule exactly, in order:
- INVALID if the SRM check failed;
- SHIP if the primary metric's CI excludes zero on the good side (direction = "better", after
  any multiple-testing adjustment) and every guardrail passed;
- EXTEND if the primary result is inconclusive and the observed n is below the planned n;
- DON'T SHIP otherwise.
Do not override the rule with your own judgement; put concerns (e.g. a decaying daily effect)
under Caveats.

Readout format:
## Decision: <SHIP | DON'T SHIP | EXTEND | INVALID>
## Primary result
## Guardrails
## Validity checks (SRM, peeking, novelty)
## Caveats
## Exploratory findings

Citation rule: every number must be copied from a tool result and followed by its result_id in
brackets, e.g. "+8.73% (95% CI +2.10% to +15.4%) [r-3fa9c2d1.treatment]". Never compute,
round or convert a number yourself; call `calculate` if you need arithmetic.
"""


def _tool_errors(fn: Callable[P, R]) -> Callable[P, R]:
    """Turn request errors into ToolErrors whose message the model can act on."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except AnalysisError as e:
            raise ToolError(str(e)) from None

    return wrapper


def build_server(store: Store) -> FastMCP:
    an = Analyzer(store)
    mcp = FastMCP("experiment-copilot", instructions=INSTRUCTIONS)

    def tool(fn: Callable[P, R]) -> Callable[P, R]:
        mcp.tool(annotations=READ_ONLY)(_tool_errors(fn))
        return fn

    @tool
    def list_experiments() -> ListExperimentsResult:
        """List experiments in the store (id, name, status, unit count; at most 50).

        Use first when you don't know an experiment id. Not needed if you already have the id.
        """
        return an.list_experiments()

    @tool
    def describe_experiment(experiment_id: ExperimentId) -> DescribeExperimentResult:
        """Arms with planned shares and observed units, metrics with types, covariates,
        segments, daily-data availability, and the full pre-registered plan (primary metric,
        guardrails with margins, MDE, alpha, planned n, interim looks, decision rule).

        Use this first for any experiment: everything else depends on the plan. It reports
        no effects; use analyze_metric for those.
        """
        return an.describe_experiment(experiment_id)

    @tool
    def check_srm(experiment_id: ExperimentId) -> SRMCheckResult:
        """Sample-ratio mismatch check: observed vs planned traffic split (chi-square
        goodness of fit), failing at the plan's SRM alpha (default p < 0.001).

        Use before interpreting any effect. If it fails, the experiment is invalid whatever the
        effects say. A planned unequal split (e.g. 85/15) is not an SRM; this test compares
        against the planned shares.
        """
        return an.check_srm(experiment_id)

    @tool
    def power_analysis(
        metric_type: Annotated[
            Literal["binary", "continuous", "ratio"],
            Field(description="binary: a rate; continuous or ratio: needs sd"),
        ],
        baseline: Annotated[float, Field(gt=0, description="Control mean (a rate for binary)")],
        mde: Annotated[
            float | None,
            Field(
                description="Relative minimum detectable effect, e.g. 0.05 for 5%. "
                "Give this to get the sample size"
            ),
        ] = None,
        n_per_arm: Annotated[
            int | None,
            Field(gt=1, description="Control-arm units. Give this instead of mde to get the MDE"),
        ] = None,
        sd: Annotated[
            float | None,
            Field(
                gt=0, description="Per-unit sd (continuous), or of the linearized metric (ratio)"
            ),
        ] = None,
        alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05,
        power: Annotated[float, Field(gt=0, lt=1)] = 0.8,
        ratio: Annotated[float, Field(gt=0, description="n_treatment / n_control")] = 1.0,
    ) -> PowerResult:
        """Sample size for a given MDE, or the MDE for a given sample size (two-sided,
        normal approximation).

        Use to judge whether an experiment was big enough, e.g. whether an inconclusive
        result means "no effect" or "underpowered". Takes plain numbers, not an experiment id.
        """
        return an.power_analysis(metric_type, baseline, mde, alpha, power, sd, n_per_arm, ratio)

    @tool
    def analyze_metric(
        experiment_id: ExperimentId,
        metric: MetricName,
        method: Annotated[
            Literal["auto", "z_test", "welch", "delta", "bootstrap"],
            Field(
                description="auto picks z_test for binary, welch for continuous, delta for ratio "
                "metrics; bootstrap on request for binary/continuous"
            ),
        ] = "auto",
        cuped_covariate: Annotated[
            str | None,
            Field(
                description="Pre-period covariate for CUPED variance reduction (binary or "
                "continuous metrics only), e.g. pre_revenue"
            ),
        ] = None,
        arms: Annotated[
            list[str] | None,
            Field(description="Treatment arms to compare with control; default: all"),
        ] = None,
    ) -> AnalyzeMetricResult:
        """Treatment-vs-control effect on one metric: absolute and relative effect with CIs
        at the plan's alpha, p-value, units per arm, direction (better / worse / inconclusive,
        respecting whether higher is better), and the guardrail verdict if the metric is a
        guardrail. One comparison per treatment arm, each with its own citable result_id.

        Use for the primary metric, each guardrail and any secondary metric. Use the delta
        method (auto) for ratio metrics; never analyze a ratio metric per session. Not for
        interim looks (sequential_test) or per-segment effects (segment_effects).
        """
        return an.analyze_metric(experiment_id, metric, method, cuped_covariate, arms)

    @tool
    def adjust_pvalues(
        result_ids: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=50,
                description="Ids from analyze_metric: whole results or single comparisons",
            ),
        ],
        method: Annotated[
            Literal["holm", "bh"],
            Field(
                description="holm controls family-wise error; bh controls the false discovery rate"
            ),
        ] = "holm",
    ) -> AdjustPValuesResult:
        """Multiple-testing adjusted p-values across several comparisons.

        Use when a decision rests on more than one comparison: several treatment arms, or
        several primary/secondary metrics. Use the plan's multiple-testing method. Only
        accepts result ids returned by analyze_metric in this session.
        """
        return an.adjust_pvalues(result_ids, method)

    @tool
    def sequential_test(
        experiment_id: ExperimentId,
        metric: MetricName,
        looks: Annotated[
            list[int] | None,
            Field(
                description="Days (1-based) at which the data were looked at; default: the "
                "plan's interim looks"
            ),
        ] = None,
        method: Annotated[
            Literal["obrien_fleming", "msprt"],
            Field(
                description="obrien_fleming: group sequential with alpha spending; msprt: "
                "always-valid p-values"
            ),
        ] = "obrien_fleming",
        arm: ArmName = None,
    ) -> SequentialResult:
        """Re-test with boundaries that stay valid when the data were looked at more than
        once. Reports, per look, the statistic, the boundary and whether it was crossed.

        Use when the plan has interim looks or the user says they peeked (e.g. "we checked
        every day"). A fixed-horizon p-value from analyze_metric is not valid after repeated
        peeking. Not a replacement for analyze_metric's effect sizes.
        """
        return an.sequential_test(experiment_id, metric, looks, method, arm)

    @tool
    def bayesian_ab(
        experiment_id: ExperimentId,
        metric: MetricName,
        arm: ArmName = None,
        prior: Annotated[
            list[float] | None,
            Field(description="Beta prior [a, b] for binary metrics; default [1, 1]"),
        ] = None,
    ) -> BayesianResult:
        """Bayesian summary: probability the treatment is better, credible intervals and the
        expected loss of shipping vs keeping control.

        Use when the user asks for a Bayesian view or "chance B beats A". The pre-registered
        decision rule is frequentist; do not base the decision on this.
        """
        return an.bayesian_ab(experiment_id, metric, arm, prior)

    @tool
    def segment_effects(
        experiment_id: ExperimentId,
        metric: MetricName,
        segment: Annotated[str, Field(description="Segment column from describe_experiment")],
        arm: ArmName = None,
        adjust: Literal["holm", "bh"] = "holm",
    ) -> SegmentEffectsResult:
        """Effect within each level of a segment (e.g. platform), with p-values adjusted
        across levels and a Cochran's Q test of whether effects differ.

        Use for exploratory findings only: never use it to override the overall decision.
        Levels with fewer than 100 units per arm are skipped.
        """
        return an.segment_effects(experiment_id, metric, segment, arm, adjust)

    @tool
    def daily_effects(
        experiment_id: ExperimentId, metric: MetricName, arm: ArmName = None
    ) -> DailyEffectsResult:
        """Relative effect for each day and a weighted trend test; flags `decaying` when the
        effect shrinks significantly over time (novelty effect).

        Use when describe_experiment shows has_daily = true, before trusting a positive
        overall effect. Not available for experiments without daily data.
        """
        return an.daily_effects(experiment_id, metric, arm)

    @tool
    def calculate(
        operation: Literal["add", "subtract", "multiply", "divide", "pct_change"],
        a: float,
        b: float,
    ) -> CalculateResult:
        """Exact arithmetic on numbers from other tool results, returned with a citable id.
        pct_change is (b - a) / a.

        Use whenever the readout needs a number no tool returned (a difference, a ratio, a
        percentage). Never do the arithmetic in prose.
        """
        return an.calculate(operation, a, b)

    @mcp.resource("experiments://registry", mime_type="application/json")
    def registry() -> str:
        """All experiments: id, name, status and unit count."""
        return an.list_experiments().model_dump_json()

    @mcp.resource("experiments://{experiment_id}/plan", mime_type="application/json")
    def plan(experiment_id: str) -> str:
        """The pre-registered plan for one experiment."""
        try:
            return an.load(experiment_id).experiment.plan.model_dump_json()
        except AnalysisError as e:
            raise ToolError(str(e)) from None

    @mcp.resource("metrics://definitions", mime_type="application/json")
    def metric_definitions() -> str:
        """Metric types and the method analyze_metric uses for each with method='auto'."""
        return json.dumps(
            {
                "binary": {
                    "meaning": "0/1 per unit, reported as a rate",
                    "auto_method": METHOD_NAMES[DEFAULT_METHOD[MetricType.BINARY]],
                },
                "continuous": {
                    "meaning": "a value per unit, often skewed (e.g. revenue)",
                    "auto_method": METHOD_NAMES[DEFAULT_METHOD[MetricType.CONTINUOUS]],
                },
                "ratio": {
                    "meaning": "sum(numerator) / sum(denominator) over units, e.g. revenue "
                    "per session; the unit of analysis is still the user",
                    "auto_method": METHOD_NAMES[DEFAULT_METHOD[MetricType.RATIO]],
                },
            }
        )

    @mcp.prompt
    def experiment_readout(experiment_id: str) -> str:
        """Readout template: tool order, the pre-registered decision rule and citation rules."""
        return READOUT_PROMPT.format(experiment_id=experiment_id)

    return mcp


def serve(data_dir: Path | None = None) -> None:
    build_server(Store(data_dir, read_only=True)).run(transport="stdio", show_banner=False)
