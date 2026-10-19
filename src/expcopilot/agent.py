"""Readout agent: an MCP client plus a capped function-calling loop.

The agent starts (or is handed) the experiment-copilot MCP server, offers its tools to the LLM,
executes the calls the model makes, and labels every successful tool result [r1], [r2], ... in
call order. The model cites those labels; the trace maps each label to the tool, its inputs
and the server's stable result_id.

Caps: at most `max_turns` model turns and `max_tool_calls` tool calls. When either is hit the
model gets one last turn with tool_choice="none" and is told to write the readout from what
it has; the readout is marked truncated.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport

from expcopilot.llm import LLM, LLMResponse, Usage
from expcopilot.readout import (
    AGENT_SYSTEM,
    Decision,
    citation_issues,
    parse_decision,
    readout_prompt,
)

MAX_TURNS = 12
MAX_TOOL_CALLS = 30
MAX_RESULT_CHARS = 12_000  # per tool result sent to the model
FINALIZE = (
    "You have reached the analysis budget. Write the readout now from the tool results "
    "you already have, in the required format, without calling any more tools."
)


@dataclass
class TraceEntry:
    step: int
    label: str | None  # None for failed calls
    tool: str
    arguments: dict[str, Any] | str
    result_id: str | None
    summary: str
    is_error: bool
    latency_s: float


@dataclass
class Readout:
    experiment_id: str
    mode: str  # "agent" | "baseline"
    model: str
    models_seen: list[str]
    text: str
    decision: Decision | None
    trace: list[TraceEntry]
    usage: Usage
    latency_s: float
    turns: int
    tool_calls: int
    truncated: bool
    finish_reason: str
    citation_issues: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d


def server_transport(data_dir: Path | None = None) -> StdioTransport:
    """`expc serve` as a subprocess, using this interpreter so the venv is always right."""
    env = dict(os.environ)
    env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")  # keep the server quiet on stderr
    if data_dir is not None:
        env["EXPC_DATA_DIR"] = str(Path(data_dir).resolve())
    return StdioTransport(command=sys.executable, args=["-m", "expcopilot.cli", "serve"], env=env)


def openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """MCP tool definitions -> OpenAI function-calling format (order kept for caching)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in mcp_tools
    ]


def _summary_of(result: dict[str, Any]) -> str:
    if "summary" in result:
        return str(result["summary"])
    comps = result.get("comparisons") or []
    return " | ".join(str(c.get("summary", "")) for c in comps)[:500]


async def run_agent(
    experiment_id: str,
    llm: LLM,
    server: FastMCP | StdioTransport,
    max_turns: int = MAX_TURNS,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> Readout:
    started = time.perf_counter()
    trace: list[TraceEntry] = []
    usage = Usage()
    models_seen: list[str] = []
    notes: list[str] = []
    issued: dict[str, str] = {}
    n_calls, turns, truncated = 0, 0, False
    last: LLMResponse | None = None

    async with Client(server) as mcp:
        tools = openai_tools(await mcp.list_tools())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": readout_prompt(experiment_id, citation="label")},
        ]

        def record(resp: LLMResponse) -> None:
            nonlocal usage, last
            usage = usage + resp.usage
            last = resp
            if resp.model not in models_seen:
                models_seen.append(resp.model)
            messages.append(resp.assistant_message)

        while True:
            if turns >= max_turns or n_calls >= max_tool_calls:
                truncated = True
                notes.append(f"budget reached after {turns} turns and {n_calls} tool calls")
                messages.append({"role": "user", "content": FINALIZE})
                record(await asyncio.to_thread(llm.chat, messages, tools, "none"))
                turns += 1
                break
            resp = await asyncio.to_thread(llm.chat, messages, tools)
            turns += 1
            record(resp)
            if resp.finish_reason == "length":
                notes.append("a model turn hit max_tokens")
            if not resp.tool_calls:
                break
            for call in resp.tool_calls:
                t0 = time.perf_counter()
                if n_calls >= max_tool_calls:
                    content = "ERROR: tool-call budget exhausted; write the readout now."
                    entry = TraceEntry(
                        turns,
                        None,
                        call.name,
                        call.arguments or call.raw_arguments,
                        None,
                        "budget exhausted",
                        True,
                        0.0,
                    )
                elif call.arguments is None:
                    content = (
                        f"ERROR: arguments were not a valid JSON object: {call.raw_arguments[:200]}"
                    )
                    entry = TraceEntry(
                        turns,
                        None,
                        call.name,
                        call.raw_arguments,
                        None,
                        "invalid JSON arguments",
                        True,
                        0.0,
                    )
                else:
                    n_calls += 1
                    res = await mcp.call_tool(call.name, call.arguments, raise_on_error=False)
                    if res.is_error:
                        msg = res.content[0].text if res.content else "tool error"
                        content = f"ERROR: {msg}"
                        entry = TraceEntry(
                            turns,
                            None,
                            call.name,
                            call.arguments,
                            None,
                            msg,
                            True,
                            time.perf_counter() - t0,
                        )
                    else:
                        label = f"r{len(issued) + 1}"
                        data = res.structured_content or {}
                        rid = data.get("result_id")
                        issued[label] = rid or ""
                        body = json.dumps(data, separators=(",", ":"))
                        if len(body) > MAX_RESULT_CHARS:
                            body = body[:MAX_RESULT_CHARS] + " ...(truncated)"
                            notes.append(f"{label} ({call.name}) result truncated for the model")
                        content = f"[{label}] {call.name} result_id={rid}\n{body}"
                        entry = TraceEntry(
                            turns,
                            label,
                            call.name,
                            call.arguments,
                            rid,
                            _summary_of(data),
                            False,
                            time.perf_counter() - t0,
                        )
                trace.append(entry)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    assert last is not None
    text = last.content.strip()
    issues = citation_issues(text, set(issued))
    decision = parse_decision(text)
    if decision is None:
        issues.append("no '## Decision:' heading found")
    return Readout(
        experiment_id=experiment_id,
        mode="agent",
        model=llm.model,
        models_seen=models_seen,
        text=text,
        decision=decision,
        trace=trace,
        usage=usage,
        latency_s=time.perf_counter() - started,
        turns=turns,
        tool_calls=n_calls,
        truncated=truncated,
        finish_reason=last.finish_reason,
        citation_issues=issues,
        notes=notes,
    )
