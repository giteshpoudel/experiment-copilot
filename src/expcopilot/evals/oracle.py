"""Oracle decisions, truth labels and the scenario file.

The oracle is a deterministic, correct analysis that applies the pre-registered decision rule
with the same `Analyzer` the MCP tools use (validated in Phase 2). The truth label comes from
the simulator's known effects instead, so the eval can also show where following the rule
correctly still gives the wrong answer (e.g. novelty).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from expcopilot.analysis import Analyzer
from expcopilot.models import MetricType, SimulationTruth
from expcopilot.readout import Decision
from expcopilot.simulate import SIM_METRICS
from expcopilot.store import Store, TruthStore

SPLIT_SEED = 7
HIGHER_IS_BETTER = {m.name: m.higher_is_better for m in SIM_METRICS}


@dataclass
class OracleResult:
    decision: Decision
    srm_failed: bool
    shipped_arm: str | None
    qualifying_arms: list[str]
    failed_guardrails: list[str]  # "arm:metric"
    required_checks: list[str]
    reasons: list[str] = field(default_factory=list)


def required_checks(an: Analyzer, experiment_id: str) -> list[str]:
    exp = an.load(experiment_id).experiment
    plan = exp.plan
    checks = ["describe_experiment", "check_srm", "analyze_primary"]
    if plan.guardrails:
        checks.append("analyze_guardrails")
    if plan.primary_covariate:
        checks.append("cuped_primary")
    if exp.metric(plan.primary_metric).type is MetricType.RATIO:
        checks.append("delta_primary")
    if sum(not a.is_control for a in exp.arms) > 1:
        checks.append("adjust_pvalues")
    if plan.planned_looks:
        checks.append("sequential_test")
    if exp.has_daily:
        checks.append("daily_effects")
    return checks


def oracle(an: Analyzer, experiment_id: str) -> OracleResult:
    exp = an.load(experiment_id).experiment
    plan = exp.plan
    checks = required_checks(an, experiment_id)
    srm = an.check_srm(experiment_id)
    if not srm.passed:
        return OracleResult(
            "invalid", True, None, [], [], checks, [f"SRM failed (p = {srm.p_value:.3g})"]
        )

    primary = an.analyze_metric(
        experiment_id, plan.primary_metric, cuped_covariate=plan.primary_covariate
    )
    comps = primary.comparisons
    p_adj = {c.result_id: c.p_value for c in comps}
    if len(comps) > 1 and plan.multiple_testing != "none":
        adj = an.adjust_pvalues([primary.result_id], plan.multiple_testing)
        p_adj = {e.result_id: e.p_adjusted for e in adj.entries}
    reasons: list[str] = []

    if plan.planned_looks:
        if len(comps) != 1:
            raise NotImplementedError("interim looks with several treatment arms")
        seq = an.sequential_test(experiment_id, plan.primary_metric)
        better = seq.stopped and comps[0].direction == "better"
        qualifying = [comps[0]] if better else []
        reasons.append(
            f"sequential test {'crossed' if seq.stopped else 'did not cross'} its boundary"
        )
        significant_worse = seq.stopped and comps[0].direction == "worse"
    else:
        qualifying = [
            c for c in comps if c.direction == "better" and p_adj[c.result_id] < plan.alpha
        ]
        significant_worse = any(
            c.direction == "worse" and p_adj[c.result_id] < plan.alpha for c in comps
        )

    failed: list[str] = []
    for comp in sorted(qualifying, key=lambda c: c.rel_effect or 0.0, reverse=True):
        arm_failed = []
        for g in plan.guardrails:
            res = an.analyze_metric(experiment_id, g.metric, arms=[comp.arm])
            gc = res.comparisons[0].guardrail
            if gc is not None and not gc.passed:
                arm_failed.append(f"{comp.arm}:{g.metric}")
        if not arm_failed:
            reasons.append(f"{comp.arm} wins on {plan.primary_metric}; guardrails pass")
            return OracleResult(
                "ship", False, comp.arm, [c.arm for c in qualifying], failed, checks, reasons
            )
        failed.extend(arm_failed)

    if qualifying:
        reasons.append(f"primary wins but guardrails failed: {', '.join(failed)}")
        return OracleResult(
            "dont_ship", False, None, [c.arm for c in qualifying], failed, checks, reasons
        )
    n_obs = sum(a.observed_n for a in an.describe_experiment(experiment_id).arms)
    if (
        not significant_worse
        and plan.decision_rule.extend_if_underpowered
        and n_obs < plan.planned_n_total
    ):
        reasons.append(f"inconclusive with n = {n_obs} < planned {plan.planned_n_total}")
        return OracleResult("extend", False, None, [], failed, checks, reasons)
    reasons.append(
        "primary significantly worse"
        if significant_worse
        else "primary inconclusive at full sample"
    )
    return OracleResult("dont_ship", False, None, [], failed, checks, reasons)


def truth_label(truth: SimulationTruth) -> dict[str, Any]:
    """Should this ship, given the true (steady-state) effects? Independent of the data."""
    spec = truth.spec
    primary = spec.plan.primary_metric
    sign = 1 if HIGHER_IS_BETTER[primary] else -1
    effects = {(e.metric, e.arm): e.relative for e in truth.true_effects}
    arms = [a.name for a in spec.arms if not a.is_control]
    harm = {
        arm: any(
            effects.get((g.metric, arm), 0.0) * (1 if HIGHER_IS_BETTER[g.metric] else -1) < 0
            for g in spec.plan.guardrails
        )
        for arm in arms
    }
    primary_fx = {arm: effects.get((primary, arm), 0.0) for arm in arms}
    good = [arm for arm in arms if sign * primary_fx[arm] > 0 and not harm[arm]]
    return {
        "truth_ship": spec.srm is None and bool(good),
        "good_arms": good,
        "true_primary_effects": primary_fx,
        "true_guardrail_harm": harm,
        "srm_injected": spec.srm is not None,
        "novelty_injected": spec.novelty is not None,
    }


def assign_splits(scenarios: list[dict[str, Any]]) -> None:
    """Stratified 50/50 dev/test: within each knob combination, alternate after a shuffle."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for s in scenarios:
        groups[tuple(s["knobs"])].append(s)
    rng = random.Random(SPLIT_SEED)
    toggle = 0
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        for s in members:
            s["split"] = "dev" if toggle % 2 == 0 else "test"
            toggle += 1


def build_scenarios(
    store: Store, truth_store: TruthStore, suite: str = "eval"
) -> list[dict[str, Any]]:
    an = Analyzer(store)
    out = []
    for row in store.list():
        try:
            truth = truth_store.get(row.experiment_id)
        except KeyError:
            continue
        if truth.suite != suite:
            continue
        o = oracle(an, row.experiment_id)
        out.append(
            {
                "experiment_id": row.experiment_id,
                "knobs": truth.knobs,
                "oracle": {k: v for k, v in asdict(o).items() if k != "required_checks"},
                "required_checks": o.required_checks,
                "truth": truth_label(truth),
            }
        )
    assign_splits(out)
    return sorted(out, key=lambda s: s["experiment_id"])


def write_scenarios(scenarios: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in scenarios))


def load_scenarios(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if split in (None, "all") or r["split"] == split]
