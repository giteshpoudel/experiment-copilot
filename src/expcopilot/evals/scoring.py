"""Score one readout against its scenario, and aggregate runs into the eval table.

Everything here is deterministic. The LLM judge (explanation quality) lives in judge.py.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from typing import Any

# --- number grounding ----------------------------------------------------------------------

MINUS = chr(0x2212)  # typographic minus sign, as some models write it
_NUM_RE = re.compile(r"(?<![\w.\-/])[+\-" + MINUS + r"]?\d[\d,]*(?:\.\d+)?%?(?![\w])")
_CITATION_RE = re.compile(r"\[[^\[\]]{0,80}\]")
_EXCLUDE_RE = [
    re.compile(r"\b\d{2}(?:\.\d+)?% (?:CI|credible|confidence)", re.IGNORECASE),  # CI level
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # dates
    re.compile(r"\b(?:day|days|look|looks|phase|step|arm|arms)\s+\d+\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),  # list numbering
]


def normalize(token: str) -> str:
    t = token.replace(MINUS, "-").replace(",", "").lstrip("+")
    if t.startswith("-"):
        t = "-" + t[1:].lstrip("+")
    body, pct = (t[:-1], "%") if t.endswith("%") else (t, "")
    if "." in body:
        body = body.rstrip("0").rstrip(".")
    return (body or "0") + pct


def numbers_in(text: str, skip_small_ints: bool = True) -> list[str]:
    """Number tokens a reader would take as data, normalized. Ignores citations, CI levels,
    dates, list numbering and bare integers up to 20 (counts like '3 arms')."""
    text = _CITATION_RE.sub(" ", text)
    for rx in _EXCLUDE_RE:
        text = rx.sub(" ", text)
    out = []
    for tok in _NUM_RE.findall(text):
        n = normalize(tok)
        if skip_small_ints and re.fullmatch(r"-?\d+", n) and abs(int(n)) <= 20:
            continue
        out.append(n)
    return out


def grounding(text: str, sources: Iterable[str]) -> dict[str, Any]:
    """Share of the readout's numbers that appear verbatim (after normalization) in sources."""
    pool: set[str] = set()
    for s in sources:
        pool.update(numbers_in(s, skip_small_ints=False))
        pool.update(n.lstrip("-") for n in numbers_in(s, skip_small_ints=False))
    nums = numbers_in(text)
    missing = [n for n in nums if n not in pool and n.lstrip("-") not in pool]
    total = len(nums)
    return {
        "numbers": total,
        "ungrounded": missing[:20],
        "rate": None if total == 0 else round(1 - len(missing) / total, 4),
    }


# --- required checks from the trace -----------------------------------------------------------


def checks_met(
    required: list[str], trace: list[dict[str, Any]], scenario: dict[str, Any], plan: dict[str, Any]
) -> dict[str, bool]:
    ok = [t for t in trace if not t.get("is_error")]
    primary = plan["primary_metric"]
    guardrails = [g["metric"] for g in plan.get("guardrails", [])]

    def called(tool: str, **match: Any) -> list[dict[str, Any]]:
        hits = []
        for t in ok:
            args = t["arguments"] if isinstance(t["arguments"], dict) else {}
            if t["tool"] == tool and all(args.get(k) == v for k, v in match.items()):
                hits.append(t)
        return hits

    prim_calls = called("analyze_metric", metric=primary)
    result = {
        "describe_experiment": bool(called("describe_experiment")),
        "check_srm": bool(called("check_srm")),
        "analyze_primary": bool(prim_calls),
        "analyze_guardrails": all(called("analyze_metric", metric=g) for g in guardrails),
        "cuped_primary": any(
            (t["arguments"] or {}).get("cuped_covariate") == plan.get("primary_covariate")
            for t in prim_calls
        ),
        "delta_primary": any(t.get("method") == "delta_method" for t in prim_calls),
        "adjust_pvalues": bool(called("adjust_pvalues")),
        "sequential_test": bool(called("sequential_test")),
        "daily_effects": bool(called("daily_effects")),
    }
    # After a failed SRM check, stopping is correct: effect checks are not required.
    if scenario["oracle"]["srm_failed"]:
        required = [c for c in required if c in ("describe_experiment", "check_srm")]
    return {c: result[c] for c in required}


# --- per-readout score ------------------------------------------------------------------------


def score(
    readout: dict[str, Any], scenario: dict[str, Any], plan: dict[str, Any], sources: list[str]
) -> dict[str, Any]:
    decision = readout.get("decision")
    oracle = scenario["oracle"]["decision"]
    truth_ship = scenario["truth"]["truth_ship"]
    checks = (
        checks_met(scenario["required_checks"], readout.get("trace", []), scenario, plan)
        if readout["mode"] != "baseline"
        else {}
    )
    g = grounding(readout.get("text", ""), sources)
    return {
        "decision": decision,
        "oracle": oracle,
        "decision_correct": decision == oracle,
        "truth_correct": (decision == "ship") == truth_ship,
        "truth_ship": truth_ship,
        "false_ship": decision == "ship" and not truth_ship,
        "srm_injected": scenario["truth"]["srm_injected"],
        "srm_called_invalid": decision == "invalid",
        "planned_imbalance": "planned_imbalance" in scenario["knobs"],
        "checks": checks,
        "method_choice": (sum(checks.values()) / len(checks)) if checks else None,
        "grounding": g,
        "unparsed": decision is None,
        "cost_usd": readout["usage"]["cost_usd"],
        "latency_s": readout["latency_s"],
        "tool_calls": readout.get("tool_calls", 0),
        "truncated": readout.get("truncated", False),
    }


# --- aggregation ------------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def run_metrics(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    """Metrics for one config x one run over a split."""
    srm = [s for s in scores if s["srm_injected"]]
    clean = [s for s in scores if not s["srm_injected"]]
    truth_negatives = [s for s in scores if not s["truth_ship"]]
    return {
        "decision_accuracy": _mean([float(s["decision_correct"]) for s in scores]),
        "truth_agreement": _mean([float(s["truth_correct"]) for s in scores]),
        "false_ship_rate": (
            _mean([float(s["false_ship"]) for s in truth_negatives]) if truth_negatives else None
        ),
        "method_choice": _mean([s["method_choice"] for s in scores]),
        "srm_recall": _mean([float(s["srm_called_invalid"]) for s in srm]) if srm else None,
        "srm_false_alarm": _mean([float(s["srm_called_invalid"]) for s in clean]),
        "grounding": _mean([s["grounding"]["rate"] for s in scores]),
        "unparsed_rate": _mean([float(s["unparsed"]) for s in scores]),
        "judge_score": _mean([s.get("judge", {}).get("mean") for s in scores]),
        "cost_usd": _mean([s["cost_usd"] for s in scores]),
        "latency_s": _mean([s["latency_s"] for s in scores]),
    }


def summarize(per_run: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    """Mean, min and max of each metric across runs."""
    out: dict[str, dict[str, float | None]] = {}
    for key in per_run[0]:
        vals = [r[key] for r in per_run if r[key] is not None]
        out[key] = (
            {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}
            if vals
            else {"mean": None, "min": None, "max": None}
        )
    return out
