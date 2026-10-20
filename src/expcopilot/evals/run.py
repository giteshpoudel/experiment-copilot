"""Run eval configs over a split, score every readout, aggregate runs into a table.

Agent runs use the MCP server in memory (same tools and code as `expc serve`, minus the stdio
transport) so many readouts can run concurrently.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from expcopilot.agent import run_agent
from expcopilot.analysis import Analyzer
from expcopilot.baseline import run_baseline
from expcopilot.evals.judge import judge, judge_model_for
from expcopilot.evals.scoring import run_metrics, score, summarize
from expcopilot.llm import LLM
from expcopilot.readout import AGENT_SYSTEM, BASELINE_SYSTEM, readout_prompt
from expcopilot.server import build_server
from expcopilot.store import Store


@dataclass(frozen=True)
class Config:
    name: str
    mode: str  # "agent" | "baseline"
    model: str


CONFIGS = {
    "agent": Config("agent", "agent", "deepseek-v4-pro"),
    "baseline": Config("baseline", "baseline", "deepseek-v4-pro"),
    "agent-flash": Config("agent-flash", "agent", "deepseek-flash"),
}

PERCENT_METRICS = {
    "decision_accuracy",
    "truth_agreement",
    "false_ship_rate",
    "method_choice",
    "srm_recall",
    "srm_false_alarm",
    "grounding",
    "unparsed_rate",
}
METRIC_LABELS = {
    "decision_accuracy": "Decision accuracy (vs oracle)",
    "truth_agreement": "Agreement with true effect",
    "false_ship_rate": "False-ship rate",
    "method_choice": "Method-choice accuracy",
    "srm_recall": "SRM detected",
    "srm_false_alarm": "False SRM alarms",
    "grounding": "Numbers grounded in tool output",
    "unparsed_rate": "No parseable decision",
    "judge_score": "Explanation quality (judge, 1-5)",
    "cost_usd": "Cost per readout (USD)",
    "latency_s": "Latency per readout (s)",
}


def prompt_version() -> str:
    text = AGENT_SYSTEM + readout_prompt("X", citation="label") + BASELINE_SYSTEM
    return hashlib.sha1(text.encode()).hexdigest()[:8]


LLMFactory = Callable[[str], LLM]  # model name -> client


async def run_one(
    cfg: Config,
    run: int,
    scenario: dict[str, Any],
    data_dir: Path,
    make_llm: LLMFactory,
    use_judge: bool,
) -> dict[str, Any]:
    eid = scenario["experiment_id"]
    store = Store(data_dir, read_only=True)
    llm = make_llm(cfg.model)
    started = time.perf_counter()
    try:
        if cfg.mode == "agent":
            readout = await run_agent(eid, llm, build_server(store))
        else:
            readout = await asyncio.to_thread(run_baseline, eid, llm, store)
    except Exception as e:  # one failed readout must not sink the whole run
        return {
            "config": cfg.name,
            "run": run,
            "experiment_id": eid,
            "error": f"{type(e).__name__}: {e}",
            "latency_s": time.perf_counter() - started,
        }
    plan = Analyzer(store).load(eid).experiment.plan.model_dump()
    rd = readout.to_dict()
    s = score(rd, scenario, plan, readout.sources)
    if use_judge:
        s["judge"] = await asyncio.to_thread(
            judge, make_llm(judge_model_for(cfg.model)), readout.text, scenario
        )
    rd.pop("sources", None)
    return {
        "config": cfg.name,
        "run": run,
        "experiment_id": eid,
        "split": scenario["split"],
        "score": s,
        "readout": rd,
    }


async def run_eval(
    scenarios: list[dict[str, Any]],
    configs: list[Config],
    runs: int,
    data_dir: Path,
    make_llm: LLMFactory,
    concurrency: int = 8,
    use_judge: bool = True,
    on_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(cfg: Config, run: int, sc: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            rec = await run_one(cfg, run, sc, data_dir, make_llm, use_judge)
        if on_done:
            on_done(rec)
        return rec

    tasks = [guarded(cfg, r, sc) for cfg in configs for r in range(1, runs + 1) for sc in scenarios]
    return await asyncio.gather(*tasks)


def aggregate(records: list[dict[str, Any]], configs: list[Config], runs: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cfg in configs:
        per_run = []
        for r in range(1, runs + 1):
            scores = [
                rec["score"]
                for rec in records
                if rec["config"] == cfg.name and rec["run"] == r and "score" in rec
            ]
            if scores:
                per_run.append(run_metrics(scores))
        errors = sum(1 for rec in records if rec["config"] == cfg.name and "error" in rec)
        judge_cost = sum(
            rec["score"].get("judge", {}).get("judge_cost_usd", 0.0)
            for rec in records
            if rec["config"] == cfg.name and "score" in rec
        )
        out[cfg.name] = {
            "model": cfg.model,
            "mode": cfg.mode,
            "runs": len(per_run),
            "errors": errors,
            "judge_cost_usd": round(judge_cost, 4),
            "metrics": summarize(per_run) if per_run else {},
        }
    return out


def _fmt(metric: str, stats: dict[str, float | None]) -> str:
    if stats.get("mean") is None:
        return "n/a"

    def f(x: float) -> str:
        if metric in PERCENT_METRICS:
            return f"{x * 100:.0f}%"
        if metric == "cost_usd":
            return f"${x:.4f}"
        return f"{x:.1f}" if metric in ("latency_s", "judge_score") else f"{x:.3g}"

    mean, lo, hi = stats["mean"], stats["min"], stats["max"]
    return f(mean) if lo == hi else f"{f(mean)} ({f(lo)} to {f(hi)})"


def table(summary: dict[str, Any]) -> str:
    names = list(summary)
    head = "| Metric | " + " | ".join(f"{n} ({summary[n]['model']})" for n in names) + " |"
    lines = [head, "| --- |" + " --- |" * len(names)]
    for key, label in METRIC_LABELS.items():
        row = [_fmt(key, summary[n]["metrics"].get(key, {})) for n in names]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def save(
    records: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any], out_root: Path
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"{stamp}-{meta['split']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2, default=str) + "\n"
    )
    (out / "table.md").write_text(table(summary) + "\n")
    return out
