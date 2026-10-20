"""Scripted stand-in for the LLM client, shared by the agent and eval tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from expcopilot.llm import LLMResponse, ToolCall, Usage, parse_arguments

Script = Callable[[list[dict[str, Any]]], LLMResponse]


def reply(
    content: str = "",
    calls: list[tuple[str, dict | str]] | None = None,
    reasoning: str | None = "thinking...",
) -> LLMResponse:
    tool_calls = [
        ToolCall(
            id=f"call_{i}",
            name=name,
            raw_arguments=args if isinstance(args, str) else json.dumps(args),
            arguments=parse_arguments(args if isinstance(args, str) else json.dumps(args)),
        )
        for i, (name, args) in enumerate(calls or [])
    ]
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.raw_arguments},
            }
            for c in tool_calls
        ]
    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        model="fake-1",
        usage=Usage(
            prompt_tokens=100,
            cache_hit_tokens=60,
            cache_miss_tokens=40,
            output_tokens=10,
            cost_usd=0.001,
            requests=1,
        ),
        latency_s=0.0,
        assistant_message=msg,
    )


class FakeLLM:
    model = "fake-1"

    def __init__(self, script: list[LLMResponse] | Script) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice}
        )
        if callable(self.script):
            return self.script(messages)
        return self.script[len(self.calls) - 1]
