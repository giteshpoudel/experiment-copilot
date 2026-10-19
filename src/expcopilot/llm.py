"""Thin LLM client: one `chat` call returning text, tool calls and usage with cost.

The agent only depends on the `LLM` protocol, so providers are swappable. The implementation
here talks to any OpenAI-compatible endpoint (DeepSeek by default) with the `openai` SDK.

DeepSeek specifics handled here:
- thinking mode is on by default; its `reasoning_content` must be sent back on every later
  request in a tool-using turn (or the API returns 400), so `assistant_message` keeps it;
- `temperature` is ignored in thinking mode, so runs are not deterministic;
- context caching is automatic; usage reports cache-hit and cache-miss prompt tokens;
- prices halve off-peak (peak: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

ToolChoice = Literal["auto", "none"]


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens at peak; off-peak is `off_peak_factor` of these."""

    input_cache_hit: float
    input_cache_miss: float
    output: float
    off_peak_factor: float = 1.0
    peak_hours_utc: tuple[tuple[int, int], ...] = ()  # [start, end) hour ranges, Mon-Fri


DEEPSEEK_PEAK = ((1, 4), (6, 10))
PRICES: dict[str, Prices] = {
    "deepseek-v4-pro": Prices(0.044, 1.32, 3.96, 0.5, DEEPSEEK_PEAK),
    "deepseek-flash": Prices(0.006, 0.30, 1.20, 0.5, DEEPSEEK_PEAK),
}


def is_peak(prices: Prices, when: datetime) -> bool:
    when = when.astimezone(UTC)
    if when.weekday() >= 5:
        return False
    return any(lo <= when.hour < hi for lo, hi in prices.peak_hours_utc)


@dataclass
class Usage:
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)))

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.prompt_tokens,
            self.cache_hit_tokens,
            self.cache_miss_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.cost_usd,
            self.requests,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "requests": self.requests,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None  # None when raw_arguments is not a JSON object


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
    model: str  # as reported by the API, so silent alias changes are visible
    usage: Usage
    latency_s: float
    assistant_message: dict[str, Any] = field(repr=False)  # append verbatim to history


class LLM(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse: ...


def parse_arguments(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def cost(usage: Usage, prices: Prices | None, when: datetime) -> float:
    if prices is None:
        return 0.0
    factor = 1.0 if is_peak(prices, when) else prices.off_peak_factor
    return (
        factor
        * (
            usage.cache_hit_tokens * prices.input_cache_hit
            + usage.cache_miss_tokens * prices.input_cache_miss
            + usage.output_tokens * prices.output
        )
        / 1e6
    )


class OpenAICompatLLM:
    """Chat completions against an OpenAI-compatible endpoint (DeepSeek by default)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        reasoning_effort: str | None = "high",
        thinking: bool = True,
        max_tokens: int = 16_000,
        timeout_s: float = 300.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=3)
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.prices = PRICES.get(model)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self.thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        started, when = time.perf_counter(), datetime.now(UTC)
        response = self.client.chat.completions.create(**kwargs)
        return self.parse(response, time.perf_counter() - started, when)

    def parse(self, response: Any, latency_s: float, when: datetime) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content")
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                raw_arguments=tc.function.arguments or "",
                arguments=parse_arguments(tc.function.arguments or ""),
            )
            for tc in (msg.tool_calls or [])
        ]
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments},
                }
                for c in calls
            ]
        u = response.usage
        u_extra = getattr(u, "model_extra", None) or {}
        details = getattr(u, "completion_tokens_details", None)
        hit = int(
            getattr(u, "prompt_cache_hit_tokens", None)
            or u_extra.get("prompt_cache_hit_tokens", 0)
            or 0
        )
        miss = getattr(u, "prompt_cache_miss_tokens", None) or u_extra.get(
            "prompt_cache_miss_tokens"
        )
        miss = int(miss) if miss is not None else int(u.prompt_tokens) - hit
        usage = Usage(
            prompt_tokens=int(u.prompt_tokens),
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            output_tokens=int(u.completion_tokens),
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
            requests=1,
        )
        usage.cost_usd = cost(usage, self.prices, when)
        return LLMResponse(
            content=msg.content or "",
            reasoning=reasoning,
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
            model=response.model or self.model,
            usage=usage,
            latency_s=latency_s,
            assistant_message=assistant,
        )


class MissingAPIKeyError(RuntimeError):
    pass


def from_env(role: Literal["agent", "judge"] = "agent", model: str | None = None) -> LLM:
    """Build the client from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL / JUDGE_MODEL."""
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError("LLM_API_KEY is not set; add it to .env (see .env.example)")
    default = "deepseek-v4-pro" if role == "agent" else "deepseek-flash"
    env_name = "LLM_MODEL" if role == "agent" else "JUDGE_MODEL"
    return OpenAICompatLLM(
        model=model or os.environ.get(env_name) or default,
        api_key=key,
        base_url=os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com",
        reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or "high",
    )
