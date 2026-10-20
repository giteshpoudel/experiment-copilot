"""LLM judge for explanation quality only. Decisions, methods and grounding are scored
deterministically in scoring.py; the judge never decides correctness of the decision.

The judge must be a different model from the one that wrote the readout.
"""

from __future__ import annotations

import json
import re
from typing import Any

from expcopilot.llm import LLM

CRITERIA = ("reasoning", "caveats", "clarity")
JUDGE_SYSTEM = """\
You grade written A/B test readouts for a product team. You are given a readout, the decision a
correct analysis reaches, and the known issues in this experiment. Grade only the quality of the
explanation, on three 1-5 scales. Respond with JSON only."""

RUBRIC = """\
Scales (1 = poor, 5 = excellent):
- reasoning: does the explanation correctly support the readout's own decision with the evidence
  it cites (primary result, guardrails, validity checks), and follow the pre-registered rule?
- caveats: does it flag the issues that matter for this experiment (see known issues) and avoid
  inventing problems? Missing a listed issue caps this at 2.
- clarity: is it concise, well organised and readable by a product manager?

Return exactly:
{"reasoning": <1-5>, "caveats": <1-5>, "clarity": <1-5>, "comment": "<one sentence>"}"""

KNOWN_ISSUES = {
    "bug:srm": "sample-ratio mismatch (traffic split differs from plan)",
    "planned_imbalance": "an intentionally unequal planned split, which is NOT an SRM",
    "bug:novelty": "a novelty effect: the lift decays over the days of the test",
    "bug:guardrail_harm": "a guardrail metric is harmed",
    "bug:guardrail_small_harm": "a small, possibly non-significant guardrail harm",
    "peeking": "pre-registered interim looks: needs a sequential test",
    "multi_arm": "several treatment arms: needs multiple-testing adjustment",
    "metric_conflict": "the ratio metric rose because its denominator (sessions) fell",
    "underpowered": "the experiment stopped well short of its planned sample size",
    "heterogeneity": "the effect is concentrated in one segment",
}


def judge_model_for(answer_model: str) -> str:
    return "deepseek-v4-pro" if answer_model == "deepseek-flash" else "deepseek-flash"


def judge_prompt(readout_text: str, scenario: dict[str, Any]) -> str:
    issues = [KNOWN_ISSUES[k] for k in scenario["knobs"] if k in KNOWN_ISSUES] or ["none"]
    oracle = scenario["oracle"]
    return "\n\n".join(
        [
            f"Correct decision under the pre-registered rule: {oracle['decision'].upper()} "
            f"({'; '.join(oracle['reasons'])}).",
            "Known issues in this experiment: " + "; ".join(issues) + ".",
            RUBRIC,
            "Readout to grade:\n<<<\n" + readout_text + "\n>>>",
        ]
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_grades(text: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    grades = {}
    for c in CRITERIA:
        v = data.get(c)
        if not isinstance(v, int | float) or not 1 <= v <= 5:
            return None
        grades[c] = float(v)
    return {
        **grades,
        "mean": sum(grades.values()) / len(CRITERIA),
        "comment": str(data.get("comment", ""))[:300],
    }


def judge(llm: LLM, readout_text: str, scenario: dict[str, Any], tries: int = 2) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt(readout_text, scenario)},
    ]
    cost = 0.0
    for _ in range(tries):
        resp = llm.chat(messages)
        cost += resp.usage.cost_usd
        grades = parse_grades(resp.content)
        if grades is not None:
            return {**grades, "judge_model": resp.model, "judge_cost_usd": cost}
    return {
        "mean": None,
        "error": "unparseable judge output",
        "judge_model": llm.model,
        "judge_cost_usd": cost,
    }
