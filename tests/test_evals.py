"""Eval pipeline tests: oracle, truth labels, scenario file, scoring, judge parsing and an
end-to-end run with a scripted LLM. No network."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from expcopilot.analysis import Analyzer
from expcopilot.evals.judge import judge, judge_model_for, parse_grades
from expcopilot.evals.oracle import (
    assign_splits,
    build_scenarios,
    load_scenarios,
    oracle,
    truth_label,
)
from expcopilot.evals.run import CONFIGS, aggregate, run_eval, table
from expcopilot.evals.scoring import checks_met, grounding, numbers_in, run_metrics, score
from expcopilot.simulate import run_suite
from expcopilot.store import Store, TruthStore
from fakes import FakeLLM, reply

SCENARIOS = Path(__file__).parents[1] / "evals" / "scenarios.jsonl"


@pytest.fixture(scope="module")
def dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("evals")
    run_suite("smoke", Store(root / "data"), TruthStore(root / "truth"))
    return root / "data", root / "truth"


@pytest.fixture(scope="module")
def analyzer(dirs) -> Analyzer:
    return Analyzer(Store(dirs[0], read_only=True))


# --- oracle and truth ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("eid", "decision"),
    [
        ("smoke-15", "invalid"),
        ("smoke-17", "dont_ship"),
        ("smoke-04", "ship"),
        ("smoke-02", "extend"),
        ("smoke-19", "ship"),
        ("smoke-18", "dont_ship"),
        ("smoke-05", "dont_ship"),
    ],
)
def test_oracle_on_smoke(analyzer: Analyzer, eid: str, decision: str) -> None:
    assert oracle(analyzer, eid).decision == decision


def test_oracle_picks_the_winning_arm_and_lists_checks(analyzer: Analyzer) -> None:
    o = oracle(analyzer, "smoke-19")
    assert o.shipped_arm == "variant_c"
    assert "adjust_pvalues" in o.required_checks
    assert "cuped_primary" in oracle(analyzer, "smoke-13").required_checks
    assert "delta_primary" in oracle(analyzer, "smoke-10").required_checks
    assert {"daily_effects"} <= set(oracle(analyzer, "smoke-16").required_checks)
    assert "sequential_test" in oracle(analyzer, "smoke-18").required_checks


def test_truth_labels(dirs) -> None:
    ts = TruthStore(dirs[1])
    assert truth_label(ts.get("smoke-04"))["truth_ship"] is True
    assert truth_label(ts.get("smoke-15"))["truth_ship"] is False  # SRM
    assert truth_label(ts.get("smoke-16"))["truth_ship"] is False  # novelty, true effect 0
    assert truth_label(ts.get("smoke-17"))["truth_ship"] is False  # guardrail harm
    assert truth_label(ts.get("smoke-19"))["good_arms"] == ["variant_c"]


def test_splits_are_stratified() -> None:
    rows = [{"knobs": ["a"]} for _ in range(4)] + [{"knobs": ["b"]} for _ in range(3)]
    assign_splits(rows)
    by = Counter((r["knobs"][0], r["split"]) for r in rows)
    assert by[("a", "dev")] == 2 and by[("a", "test")] == 2
    assert abs(by[("b", "dev")] - by[("b", "test")]) == 1


def test_committed_scenario_file() -> None:
    rows = load_scenarios(SCENARIOS)
    assert len(rows) == 80 and Counter(r["split"] for r in rows) == {"dev": 40, "test": 40}
    knobs = Counter(k for r in rows for k in r["knobs"])
    for knob, n in knobs.items():
        splits = {r["split"] for r in rows if knob in r["knobs"]}
        assert n < 2 or splits == {"dev", "test"}, f"{knob} only in {splits}"
    assert {r["oracle"]["decision"] for r in rows} == {"ship", "dont_ship", "invalid", "extend"}
    assert len(load_scenarios(SCENARIOS, "dev")) == 40


@pytest.mark.slow
def test_scenario_labels_reproduce(tmp_path: Path) -> None:
    """The committed labels follow from the seeds: regenerating gives identical labels."""
    store, truth = Store(tmp_path / "d"), TruthStore(tmp_path / "t")
    run_suite("eval", store, truth)
    fresh = build_scenarios(store, truth)
    assert fresh == load_scenarios(SCENARIOS)


# --- scoring -------------------------------------------------------------------------------


def test_numbers_in_skips_noise() -> None:
    text = (
        "## Decision: SHIP [r3]\n1. Conversion +11.6% (95% CI +4.18% to +19%), p = 0.0012 "
        "[r3, r4]; 14,901 units on 2026-09-01, day 14, 3 arms, id r-953b2be2."
    )
    assert numbers_in(text) == ["11.6%", "4.18%", "19%", "0.0012", "14901"]


def test_grounding() -> None:
    g = grounding(
        "lift +11.6%, CI 4.18% to 19%, se 0.0035", ['{"summary": "+11.6% (4.18% to +19%)"}']
    )
    assert g["numbers"] == 4 and g["ungrounded"] == ["0.0035"] and g["rate"] == 0.75
    assert grounding("no numbers here", [])["rate"] is None


def _trace(*calls: tuple[str, dict, str | None]) -> list[dict]:
    return [{"tool": t, "arguments": a, "method": m, "is_error": False} for t, a, m in calls]


def test_checks_met() -> None:
    plan = {
        "primary_metric": "revenue",
        "guardrails": [{"metric": "bounce"}],
        "primary_covariate": "pre_revenue",
    }
    sc = {"oracle": {"srm_failed": False}}
    req = ["check_srm", "analyze_primary", "analyze_guardrails", "cuped_primary"]
    trace = _trace(
        ("check_srm", {}, None),
        ("analyze_metric", {"metric": "revenue"}, "welch_t"),
        ("analyze_metric", {"metric": "bounce"}, "two_proportion_z"),
    )
    assert checks_met(req, trace, sc, plan) == {
        "check_srm": True,
        "analyze_primary": True,
        "analyze_guardrails": True,
        "cuped_primary": False,
    }
    # After a failed SRM, only describe/SRM are required.
    assert checks_met(req, trace, {"oracle": {"srm_failed": True}}, plan) == {"check_srm": True}


def test_score_and_run_metrics() -> None:
    sc = {
        "oracle": {"decision": "dont_ship", "srm_failed": False},
        "truth": {"truth_ship": False, "srm_injected": False},
        "knobs": [],
        "required_checks": ["check_srm"],
    }
    rd = {
        "mode": "agent",
        "decision": "ship",
        "text": "+5% [r1]",
        "trace": [],
        "usage": {"cost_usd": 0.01},
        "latency_s": 2.0,
    }
    s = score(rd, sc, {"primary_metric": "conversion"}, ["+5%"])
    assert s["false_ship"] and not s["decision_correct"] and s["method_choice"] == 0.0
    m = run_metrics([s])
    assert m["false_ship_rate"] == 1.0 and m["srm_recall"] is None and m["grounding"] == 1.0


# --- judge ---------------------------------------------------------------------------------


def test_judge_parsing_and_model_choice() -> None:
    g = parse_grades('Sure: {"reasoning": 4, "caveats": 2, "clarity": 5, "comment": "ok"}')
    assert g is not None and g["mean"] == pytest.approx(11 / 3)
    assert parse_grades('{"reasoning": 9, "caveats": 2, "clarity": 5}') is None
    assert judge_model_for("deepseek-v4-pro") == "deepseek-flash"
    assert judge_model_for("deepseek-flash") == "deepseek-v4-pro"
    sc = {"knobs": ["bug:novelty"], "oracle": {"decision": "ship", "reasons": ["wins"]}}
    llm = FakeLLM([reply("garbage"), reply('{"reasoning": 3, "caveats": 1, "clarity": 4}')])
    out = judge(llm, "## Decision: SHIP", sc)
    assert out["mean"] == pytest.approx(8 / 3)
    assert "novelty" in llm.calls[0]["messages"][1]["content"]


# --- end to end with a scripted model --------------------------------------------------------


def test_run_eval_end_to_end(dirs) -> None:
    an = Analyzer(Store(dirs[0], read_only=True))
    scenarios = []
    for eid in ("smoke-15", "smoke-04"):
        o = oracle(an, eid)
        scenarios.append(
            {
                "experiment_id": eid,
                "split": "dev",
                "knobs": [],
                "oracle": {
                    "decision": o.decision,
                    "srm_failed": o.srm_failed,
                    "reasons": o.reasons,
                },
                "required_checks": o.required_checks,
                "truth": {"truth_ship": eid == "smoke-04", "srm_injected": eid == "smoke-15"},
            }
        )

    def make_llm(model: str) -> FakeLLM:
        def script(messages):
            if messages[0]["content"].startswith("You grade"):
                return reply('{"reasoning": 4, "caveats": 4, "clarity": 4}')
            eid = "smoke-15" if "smoke-15" in messages[1]["content"] else "smoke-04"
            if (
                not any(m["role"] == "tool" for m in messages)
                and "Summary table" not in messages[1]["content"]
            ):
                return reply(calls=[("check_srm", {"experiment_id": eid})])
            return reply(
                "## Decision: INVALID\nSRM [r1]"
                if eid == "smoke-15"
                else "## Decision: SHIP\nwin [r1]"
            )

        llm = FakeLLM(script)
        llm.model = model
        return llm

    configs = [CONFIGS["agent"], CONFIGS["baseline"]]
    records = asyncio.run(run_eval(scenarios, configs, 2, dirs[0], make_llm, concurrency=4))
    assert len(records) == 8 and not any("error" in r for r in records)
    summary = aggregate(records, configs, 2)
    agent = summary["agent"]["metrics"]
    assert agent["decision_accuracy"]["mean"] == 1.0 and agent["srm_recall"]["mean"] == 1.0
    assert agent["judge_score"]["mean"] == 4.0
    md = table(summary)
    assert "Decision accuracy" in md and "agent (deepseek-v4-pro)" in md
    json.dumps(records, default=str)  # results must serialize
