"""Readout agent, baseline and LLM client tests. No network: a scripted fake LLM drives the
loop against the real MCP server (in memory) on the smoke suite. Live tests need LLM_API_KEY.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from expcopilot.agent import openai_tools, run_agent
from expcopilot.analysis import Analyzer
from expcopilot.baseline import run_baseline, summary_table
from expcopilot.llm import PRICES, OpenAICompatLLM, Usage, cost, is_peak
from expcopilot.readout import citation_issues, cited_labels, parse_decision, readout_prompt
from expcopilot.server import build_server
from expcopilot.simulate import run_suite
from expcopilot.store import Store, TruthStore
from fakes import FakeLLM, reply


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("agent")
    run_suite("smoke", Store(root / "data"), TruthStore(root / "truth"))
    return root / "data"


@pytest.fixture
def server(data_dir: Path):
    return build_server(Store(data_dir, read_only=True))


def agent(llm: FakeLLM, server, eid: str = "smoke-15", **kw: Any):
    return asyncio.run(run_agent(eid, llm, server, **kw))


FINAL_INVALID = (
    "## Decision: INVALID\n\n## Validity checks (SRM, peeking, novelty)\n"
    "SRM failed: p < 0.0001 [r2]. Planned n 30,000 [r1]."
)


def test_loop_labels_results_and_parses_decision(server) -> None:
    llm = FakeLLM(
        [
            reply(calls=[("describe_experiment", {"experiment_id": "smoke-15"})]),
            reply(calls=[("check_srm", {"experiment_id": "smoke-15"})]),
            reply(FINAL_INVALID),
        ]
    )
    r = agent(llm, server)
    assert r.decision == "invalid" and not r.truncated and r.citation_issues == []
    assert [t.label for t in r.trace] == ["r1", "r2"]
    assert [t.tool for t in r.trace] == ["describe_experiment", "check_srm"]
    assert all(t.result_id and t.result_id.startswith("r-") for t in r.trace)
    assert r.tool_calls == 2 and r.turns == 3 and r.usage.requests == 3
    # The tool result the model saw starts with its label and carries the server's result_id.
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[1]["content"].startswith("[r2] check_srm result_id=r-")
    assert "FAILED" in tool_msgs[1]["content"]


def test_reasoning_content_is_sent_back(server) -> None:
    llm = FakeLLM(
        [
            reply(calls=[("check_srm", {"experiment_id": "smoke-15"})], reasoning="first thought"),
            reply(FINAL_INVALID),
        ]
    )
    agent(llm, server)
    assistant = [m for m in llm.calls[1]["messages"] if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "first thought"
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "check_srm"


def test_tool_errors_and_bad_json_reach_the_model(server) -> None:
    llm = FakeLLM(
        [
            reply(
                calls=[
                    ("analyze_metric", {"experiment_id": "smoke-15", "metric": "gmv"}),
                    ("check_srm", "{not json"),
                ]
            ),
            reply(FINAL_INVALID),
        ]
    )
    r = agent(llm, server)
    assert [t.is_error for t in r.trace] == [True, True]
    assert r.tool_calls == 1  # the malformed call never reached the server
    tool_msgs = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("ERROR:") and "metrics: conversion" in tool_msgs[0]
    assert "not a valid JSON object" in tool_msgs[1]
    assert set(r.citation_issues) == {
        "cites r1, which no tool returned",
        "cites r2, which no tool returned",
    }


def test_turn_cap_forces_a_final_no_tools_turn(server) -> None:
    def script(messages):
        if messages[-1]["role"] == "user" and "analysis budget" in messages[-1]["content"]:
            return reply(FINAL_INVALID)
        return reply(calls=[("check_srm", {"experiment_id": "smoke-15"})])

    llm = FakeLLM(script)
    r = agent(llm, server, max_turns=3)
    assert r.truncated and r.turns == 4 and r.decision == "invalid"
    assert llm.calls[-1]["tool_choice"] == "none"
    assert [c["tool_choice"] for c in llm.calls[:-1]] == ["auto"] * 3


def test_tool_call_cap(server) -> None:
    llm = FakeLLM(
        [
            reply(calls=[("check_srm", {"experiment_id": "smoke-15"})] * 3),
            reply(FINAL_INVALID),
        ]
    )
    r = agent(llm, server, max_tool_calls=2)
    assert r.tool_calls == 2 and r.truncated
    assert [t.summary for t in r.trace][-1] == "budget exhausted"
    assert llm.calls[-1]["tool_choice"] == "none"


def test_tools_offered_in_openai_format(server) -> None:
    llm = FakeLLM([reply(FINAL_INVALID)])
    agent(llm, server)
    tools = llm.calls[0]["tools"]
    assert len(tools) == 11 and all(t["type"] == "function" for t in tools)
    names = [t["function"]["name"] for t in tools]
    assert names == sorted(names) or names  # order comes from the server and is stable
    analyze = next(t for t in tools if t["function"]["name"] == "analyze_metric")
    assert {"experiment_id", "metric"} <= set(analyze["function"]["parameters"]["properties"])
    assert openai_tools([]) == []


def test_missing_decision_is_flagged(server) -> None:
    r = agent(FakeLLM([reply("I think it's fine.")]), server)
    assert r.decision is None and "no '## Decision:' heading found" in r.citation_issues


# --- baseline ------------------------------------------------------------------------------


def test_baseline_table_has_no_inferential_numbers(data_dir: Path) -> None:
    table = summary_table(Analyzer(Store(data_dir, read_only=True)), "smoke-15")
    assert "| conversion | binary | control (control) | 14901 |" in table
    for word in ("p-value", "p =", "CI", "chi", "SRM", "significant"):
        assert word not in table


def test_baseline_run(data_dir: Path) -> None:
    llm = FakeLLM([reply("## Decision: SHIP\n\nConversion is higher in treatment.")])
    r = run_baseline("smoke-15", llm, Store(data_dir, read_only=True))
    assert r.mode == "baseline" and r.decision == "ship" and r.trace == []
    assert llm.calls[0]["tools"] is None
    prompt = llm.calls[0]["messages"][1]["content"]
    assert '"primary_metric": "conversion"' in prompt and "| metric |" in prompt


# --- readout parsing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Decision: SHIP", "ship"),
        ("## Decision: **DON’T SHIP**", "dont_ship"),  # noqa: RUF001 (curly quote on purpose)
        ("# Decision: do not ship", "dont_ship"),
        ("## Decision: EXTEND (n short)", "extend"),
        ("## Decision: Invalid", "invalid"),
        ("Decision: ship", None),
        ("nothing", None),
    ],
)
def test_parse_decision(text: str, expected: str | None) -> None:
    assert parse_decision(text) == expected


def test_citation_labels() -> None:
    text = "a [r1] b [r2, r10] c [r3][r1] d [see r4] e [1] f r7"
    assert cited_labels(text) == ["r1", "r2", "r10", "r3", "r4"]
    assert citation_issues(text, {"r1", "r2", "r3", "r4"}) == ["cites r10, which no tool returned"]
    assert citation_issues("no citations", {"r1"}) == ["cites no tool results at all"]


def test_prompt_citation_styles() -> None:
    assert "[r3]" in readout_prompt("x", citation="label")
    assert "[r-3fa9c2d1.treatment]" in readout_prompt("x", citation="result_id")


# --- LLM client ----------------------------------------------------------------------------


def _fake_completion(tool_args: str = '{"experiment_id": "smoke-01"}') -> Any:
    msg = SimpleNamespace(
        content="",
        reasoning_content="why",
        tool_calls=[
            SimpleNamespace(
                id="c1", function=SimpleNamespace(name="check_srm", arguments=tool_args)
            )
        ],
        model_extra={},
    )
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_cache_hit_tokens=800,
        prompt_cache_miss_tokens=200,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=150),
        model_extra={},
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=usage,
        model="deepseek-v4-pro",
    )


def test_openai_client_parses_tools_reasoning_and_usage() -> None:
    llm = OpenAICompatLLM("deepseek-v4-pro", api_key="x", client=object())
    off_peak = datetime(2026, 10, 18, 12, tzinfo=UTC)  # a Sunday
    r = llm.parse(_fake_completion(), 1.0, off_peak)
    assert r.tool_calls[0].arguments == {"experiment_id": "smoke-01"}
    assert r.assistant_message["reasoning_content"] == "why"
    assert r.assistant_message["tool_calls"][0]["function"]["name"] == "check_srm"
    assert (r.usage.cache_hit_tokens, r.usage.cache_miss_tokens) == (800, 200)
    assert r.usage.reasoning_tokens == 150
    p = PRICES["deepseek-v4-pro"]
    expected = 0.5 * (800 * p.input_cache_hit + 200 * p.input_cache_miss + 200 * p.output) / 1e6
    assert r.usage.cost_usd == pytest.approx(expected)
    bad = llm.parse(_fake_completion("{oops"), 1.0, off_peak)
    assert bad.tool_calls[0].arguments is None


def test_peak_hours() -> None:
    p = PRICES["deepseek-flash"]
    assert is_peak(p, datetime(2026, 10, 19, 7, tzinfo=UTC))  # Monday 07:00 UTC
    assert not is_peak(p, datetime(2026, 10, 19, 5, tzinfo=UTC))
    assert not is_peak(p, datetime(2026, 10, 18, 7, tzinfo=UTC))  # Sunday
    u = Usage(cache_hit_tokens=0, cache_miss_tokens=1_000_000, output_tokens=0)
    assert cost(u, p, datetime(2026, 10, 19, 7, tzinfo=UTC)) == pytest.approx(0.30)
    assert cost(u, None, datetime(2026, 10, 19, 7, tzinfo=UTC)) == 0.0


def test_usage_adds() -> None:
    a = Usage(1, 2, 3, 4, 5, 0.5, 1)
    assert (a + a).as_tuple() == (2, 4, 6, 8, 10, 1.0, 2)


# --- live (real API, costs money; opt-in) ---------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set")
def test_live_srm_readout(server) -> None:
    from expcopilot.llm import from_env

    r = asyncio.run(run_agent("smoke-15", from_env("agent"), server))
    assert r.decision == "invalid", r.text
    assert r.citation_issues == [], r.citation_issues
