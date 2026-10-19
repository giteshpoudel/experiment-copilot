"""Readout template, decision rule text, decision parsing and citation checks.

One template serves two clients: the MCP prompt (for Claude Code / Desktop, citing result ids)
and the readout agent (citing the short [rN] labels it attaches to each tool result).
"""

from __future__ import annotations

import re
from typing import Literal

Decision = Literal["ship", "dont_ship", "extend", "invalid"]
CitationStyle = Literal["result_id", "label"]

PROCESS = """\
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
7. Optional: segment_effects, always labelled exploratory."""

DECISION_RULE = """\
Decision: apply the plan's decision rule exactly, in order:
- INVALID if the SRM check failed;
- SHIP if the primary metric's CI excludes zero on the good side (direction = "better", after
  any multiple-testing adjustment) and every guardrail passed;
- EXTEND if the primary result is inconclusive and the observed n is below the planned n;
- DON'T SHIP otherwise.
Do not override the rule with your own judgement; put concerns (e.g. a decaying daily effect)
under Caveats."""

FORMAT = """\
Readout format:
## Decision: <SHIP | DON'T SHIP | EXTEND | INVALID>
## Primary result
## Guardrails
## Validity checks (SRM, peeking, novelty)
## Caveats
## Exploratory findings"""

CITATION = {
    "result_id": (
        "Citation rule: every number must be copied from a tool result and followed by its "
        'result_id in brackets, e.g. "+8.73% (95% CI +2.10% to +15.4%) '
        '[r-3fa9c2d1.treatment]". Never compute, round or convert a number yourself; call '
        "`calculate` if you need arithmetic."
    ),
    "label": (
        "Citation rule: each tool result you receive starts with a label such as [r3]. Every "
        "number in the readout must be copied exactly from a tool result (prefer its `summary` "
        "text) and followed by that result's label, e.g. \"+8.73% (95% CI +2.10% to +15.4%) "
        '[r3]". Never compute, round or convert a number yourself; call `calculate` if you '
        "need arithmetic, and cite its label."
    ),
}

AGENT_SYSTEM = """\
You are an experimentation analyst. You analyze A/B tests only through the tools you are given,
which read a store of experiments and run validated statistics. Numbers come from tools: you
choose methods and explain, you never compute. Be concise and specific; the readout is read by
a product team deciding whether to ship."""

BASELINE_SYSTEM = """\
You are an experimentation analyst. You are given an experiment's pre-registered plan and a
summary table of counts and means per arm. You have no other tools or data. Decide whether to
ship, following the plan's decision rule as well as you can from the table, and explain why."""


def readout_prompt(experiment_id: str, citation: CitationStyle = "result_id") -> str:
    return "\n\n".join(
        [
            f"Write a ship / don't-ship readout for experiment {experiment_id}.",
            PROCESS,
            DECISION_RULE,
            FORMAT,
            CITATION[citation],
        ]
    )


def baseline_prompt(experiment_id: str, plan_json: str, table: str) -> str:
    return "\n\n".join(
        [
            f"Write a ship / don't-ship readout for experiment {experiment_id}.",
            f"Pre-registered plan (JSON):\n{plan_json}",
            f"Summary table (per arm):\n{table}",
            DECISION_RULE,
            FORMAT,
            "Use only the numbers in the table; do not invent statistics you cannot compute.",
        ]
    )


_DECISION_RE = re.compile(
    r"^\s*#+\s*Decision\s*:\s*\**\s*(DON[\u2019']?T\s+SHIP|DO\s+NOT\s+SHIP|SHIP|EXTEND|INVALID)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_decision(text: str) -> Decision | None:
    """The decision from the '## Decision: ...' heading, or None if it is missing."""
    m = _DECISION_RE.search(text)
    if not m:
        return None
    word = re.sub(r"\s+", " ", m.group(1).upper().replace("\u2019", "'"))
    if word in ("DON'T SHIP", "DONT SHIP", "DO NOT SHIP"):
        return "dont_ship"
    return word.lower()  # type: ignore[return-value]


_BRACKET_RE = re.compile(r"\[([^\[\]]{1,80})\]")
_LABEL_RE = re.compile(r"\br(\d+)\b")


def cited_labels(text: str) -> list[str]:
    """Labels cited as [r3], [r3, r5] or [r3][r5], in order of first appearance."""
    seen: dict[str, None] = {}
    for group in _BRACKET_RE.findall(text):
        for n in _LABEL_RE.findall(group):
            seen.setdefault(f"r{n}", None)
    return list(seen)


def citation_issues(text: str, issued: set[str]) -> list[str]:
    issues = [
        f"cites {lbl}, which no tool returned" for lbl in cited_labels(text) if lbl not in issued
    ]
    if issued and not cited_labels(text):
        issues.append("cites no tool results at all")
    return issues
