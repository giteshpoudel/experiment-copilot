"""No-tools baseline: the same LLM, given only the plan and counts and means per arm.

No CIs, p-values, SRM test or any other inferential number: the point of the comparison is
what the model does without the statistics engine.
"""

from __future__ import annotations

import time

from expcopilot.agent import Readout
from expcopilot.analysis import Analyzer
from expcopilot.llm import LLM
from expcopilot.models import MetricType
from expcopilot.readout import BASELINE_SYSTEM, baseline_prompt, parse_decision
from expcopilot.store import Store


def summary_table(analyzer: Analyzer, experiment_id: str) -> str:
    """Markdown table: one row per metric and arm with units and the arm mean (or ratio)."""
    loaded = analyzer.load(experiment_id)
    exp = loaded.experiment
    rows = ["| metric | type | arm | units | mean |", "| --- | --- | --- | --- | --- |"]
    for m in exp.metrics:
        for a in exp.arms:
            mask = loaded.arm == a.name
            n = int(mask.sum())
            if m.type is MetricType.RATIO:
                num = loaded.cols[m.numerator][mask].astype(float).sum()
                den = loaded.cols[m.denominator][mask].astype(float).sum()
                value = num / den if den else float("nan")
            else:
                value = float(loaded.cols[m.column][mask].astype(float).mean())
            label = a.name + (" (control)" if a.is_control else "")
            rows.append(f"| {m.name} | {m.type.value} | {label} | {n} | {value:.6g} |")
    return "\n".join(rows)


def run_baseline(experiment_id: str, llm: LLM, store: Store) -> Readout:
    started = time.perf_counter()
    analyzer = Analyzer(store)
    exp = analyzer.load(experiment_id).experiment
    prompt = baseline_prompt(
        experiment_id, exp.plan.model_dump_json(indent=1), summary_table(analyzer, experiment_id)
    )
    resp = llm.chat(
        [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": prompt}]
    )
    text = resp.content.strip()
    decision = parse_decision(text)
    return Readout(
        experiment_id=experiment_id,
        mode="baseline",
        model=llm.model,
        models_seen=[resp.model],
        text=text,
        decision=decision,
        trace=[],
        usage=resp.usage,
        latency_s=time.perf_counter() - started,
        turns=1,
        tool_calls=0,
        truncated=False,
        finish_reason=resp.finish_reason,
        citation_issues=[] if decision else ["no '## Decision:' heading found"],
        sources=[prompt],
    )
