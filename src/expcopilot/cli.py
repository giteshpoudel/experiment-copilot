"""`expc` command line: simulate, list, describe, validate, serve, readout, eval."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from expcopilot.models import Experiment
from expcopilot.simulate import SUITES, run_suite
from expcopilot.store import ExperimentNotFoundError, Store, TruthStore

app = typer.Typer(help="Experimentation Copilot", no_args_is_help=True, add_completion=False)

DataDir = Annotated[
    Path | None,
    typer.Option("--data-dir", help="Store location (default: $EXPC_DATA_DIR or ./data)"),
]


@app.callback()
def _main() -> None:
    load_dotenv()


@app.command()
def simulate(
    suite: Annotated[str, typer.Option(help=f"Suite to generate: {', '.join(SUITES)}")] = "smoke",
    overwrite: Annotated[bool, typer.Option(help="Regenerate experiments that exist")] = False,
    data_dir: DataDir = None,
) -> None:
    """Generate a suite of simulated experiments into the store."""
    store, truth = Store(data_dir), TruthStore(data_dir)
    try:
        result = run_suite(suite, store, truth, overwrite=overwrite)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0])) from e
    typer.echo(
        f"Suite {suite!r}: created {len(result.created)}, skipped {len(result.skipped)} "
        f"(already present; use --overwrite to regenerate)."
    )
    typer.echo(f"Store: {store.data_dir.resolve()}")


@app.command()
def validate(
    reps: Annotated[int, typer.Option(help="Simulated experiments per check")] = 2000,
    out: Annotated[Path, typer.Option(help="Where to write the report")] = Path(
        "results/validation.json"
    ),
) -> None:
    """Validate the statistics engine on simulated A/A and A/B tests."""
    from expcopilot.validation import run_all, write

    report = run_all(reps=reps)
    write(report, out)
    for c in report["checks"]:
        mark = "pass" if c["passed"] else "FAIL"
        lo, hi = c["band"]
        typer.echo(f"{mark}  {c['name']:<34} {c['observed']:.4f}  band [{lo:.3f}, {hi:.3f}]")
    status = "all checks passed" if report["all_passed"] else "SOME CHECKS FAILED"
    typer.echo(f"{status} in {report['runtime_seconds']} s; wrote {out}")
    if not report["all_passed"]:
        raise typer.Exit(1)


@app.command()
def serve(data_dir: DataDir = None) -> None:
    """Run the MCP server over stdio (for Claude Code, Claude Desktop or any MCP client)."""
    from expcopilot.server import serve as run

    try:
        run(data_dir)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None


@app.command()
def readout(
    experiment_id: Annotated[str, typer.Argument(help="Experiment to write a readout for")],
    no_tools: Annotated[
        bool, typer.Option("--no-tools", help="Baseline: plan + summary table, no tools")
    ] = False,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the readout and trace as JSON")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Override LLM_MODEL")] = None,
    max_turns: Annotated[int, typer.Option(help="Cap on model turns")] = 12,
    max_tool_calls: Annotated[int, typer.Option(help="Cap on tool calls")] = 30,
    data_dir: DataDir = None,
) -> None:
    """Agent readout with its tool trace (or the no-tools baseline)."""
    import asyncio
    import json

    from expcopilot.agent import MAX_TURNS, run_agent, server_transport
    from expcopilot.baseline import run_baseline
    from expcopilot.llm import MissingAPIKeyError, from_env

    try:
        llm = from_env("agent", model=model)
    except MissingAPIKeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None
    store = Store(data_dir, read_only=True)
    if not store.exists(experiment_id):
        typer.echo(f"No experiment {experiment_id!r}.", err=True)
        raise typer.Exit(1)
    if no_tools:
        result = run_baseline(experiment_id, llm, store)
    else:
        result = asyncio.run(
            run_agent(
                experiment_id,
                llm,
                server_transport(store.data_dir),
                max_turns=max_turns or MAX_TURNS,
                max_tool_calls=max_tool_calls,
            )
        )
    typer.echo(format_readout(result))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")
        typer.echo(f"Wrote {json_out}")


def format_readout(r) -> str:  # r: agent.Readout
    u = r.usage
    lines = [r.text, "", "=" * 78, f"{r.mode} readout | decision: {r.decision or 'UNPARSED'}"]
    if r.trace:
        lines += ["", "Tool trace:"]
        for t in r.trace:
            args = (
                t.arguments
                if isinstance(t.arguments, str)
                else ", ".join(f"{k}={v}" for k, v in t.arguments.items() if v is not None)
            )
            tag = f"[{t.label}]" if t.label else "[error]"
            rid = f" {t.result_id}" if t.result_id else ""
            lines.append(f"  {tag:<7} {t.tool}({args}){rid}")
            lines.append(f"          {t.summary[:160]}")
    lines += [
        "",
        f"model {r.model} (served by {', '.join(r.models_seen)}); {r.turns} turns, "
        f"{r.tool_calls} tool calls{' (TRUNCATED)' if r.truncated else ''}",
        f"tokens: {u.prompt_tokens:,} prompt ({u.cache_hit_tokens:,} cache hits), "
        f"{u.output_tokens:,} output ({u.reasoning_tokens:,} reasoning); "
        f"cost ${u.cost_usd:.4f}; {r.latency_s:.1f} s",
    ]
    for issue in r.citation_issues:
        lines.append(f"CITATION ISSUE: {issue}")
    for note in r.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


SCENARIOS = Path("evals/scenarios.jsonl")


@app.command("eval-build")
def eval_build(
    force: Annotated[
        bool, typer.Option(help="Overwrite existing labels (only before any eval has run)")
    ] = False,
    data_dir: DataDir = None,
) -> None:
    """Generate the eval suite and write oracle + truth labels to evals/scenarios.jsonl."""
    from collections import Counter

    from expcopilot.evals.oracle import build_scenarios, write_scenarios

    if SCENARIOS.exists() and not force:
        typer.echo(
            f"{SCENARIOS} exists; labels are never regenerated silently. "
            "Use --force only before any eval run.",
            err=True,
        )
        raise typer.Exit(1)
    store, truth = Store(data_dir), TruthStore(data_dir)
    run_suite("eval", store, truth)
    scenarios = build_scenarios(store, truth)
    write_scenarios(scenarios, SCENARIOS)
    splits = Counter(s["split"] for s in scenarios)
    decisions = Counter(s["oracle"]["decision"] for s in scenarios)
    typer.echo(
        f"Wrote {len(scenarios)} scenarios to {SCENARIOS}: {dict(splits)}; "
        f"oracle decisions {dict(decisions)}"
    )


@app.command("eval")
def eval_(
    split: Annotated[str, typer.Option(help="dev | test | all")] = "dev",
    configs: Annotated[
        str, typer.Option(help="Comma-separated: agent, baseline, agent-flash")
    ] = "agent,baseline",
    runs: Annotated[int, typer.Option(help="Runs per config (report mean and range)")] = 3,
    limit: Annotated[int | None, typer.Option(help="Only the first N scenarios")] = None,
    concurrency: Annotated[int, typer.Option(help="Readouts in flight at once")] = 8,
    judge: Annotated[bool, typer.Option(help="Score explanation quality with the judge")] = True,
    data_dir: DataDir = None,
) -> None:
    """Run the eval: agent vs baseline on a split, scored against oracle and truth labels."""
    import asyncio

    from expcopilot.evals.oracle import load_scenarios
    from expcopilot.evals.run import CONFIGS, aggregate, prompt_version, run_eval, save, table
    from expcopilot.llm import MissingAPIKeyError, from_env

    if not SCENARIOS.exists():
        typer.echo("No scenarios; run `expc eval-build` first.", err=True)
        raise typer.Exit(1)
    scenarios = load_scenarios(SCENARIOS, split)[:limit]
    try:
        chosen = [CONFIGS[c.strip()] for c in configs.split(",") if c.strip()]
    except KeyError as e:
        raise typer.BadParameter(
            f"unknown config {e.args[0]!r}; choose from {list(CONFIGS)}"
        ) from None
    try:
        from_env("agent")
    except MissingAPIKeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from None
    store = Store(data_dir, read_only=True)
    total = len(scenarios) * len(chosen) * runs
    done = 0

    def progress(rec: dict) -> None:
        nonlocal done
        done += 1
        if done % 10 == 0 or done == total:
            typer.echo(f"  {done}/{total} readouts", err=True)

    typer.echo(
        f"Running {total} readouts ({len(scenarios)} {split} scenarios x "
        f"{len(chosen)} configs x {runs} runs)...",
        err=True,
    )
    records = asyncio.run(
        run_eval(
            scenarios,
            chosen,
            runs,
            store.data_dir,
            lambda m: from_env("agent", model=m),
            concurrency=concurrency,
            use_judge=judge,
            on_done=progress,
        )
    )
    summary = aggregate(records, chosen, runs)
    meta = {
        "split": split,
        "runs": runs,
        "scenarios": len(scenarios),
        "configs": [c.name for c in chosen],
        "prompt_version": prompt_version(),
        "judge": judge,
    }
    out = save(records, summary, meta, Path("evals/results"))
    typer.echo(table(summary))
    spent = sum(r["score"]["cost_usd"] for r in records if "score" in r) + sum(
        s["judge_cost_usd"] for s in summary.values()
    )
    typer.echo(f"\nTotal spend ${spent:.3f}; results in {out}")


@app.command("list")
def list_experiments(data_dir: DataDir = None) -> None:
    """List experiments in the store."""
    rows = Store(data_dir).list()
    if not rows:
        typer.echo("No experiments. Run `expc simulate --suite smoke`.")
        return
    width = max(len(r.experiment_id) for r in rows)
    for r in rows:
        typer.echo(
            f"{r.experiment_id:<{width}}  {r.status:<8}  {r.source:<9}  "
            f"{r.n_units:>7,} units  {r.name}"
        )


@app.command()
def describe(
    experiment_id: Annotated[str | None, typer.Argument(help="Omit to describe all")] = None,
    truth: Annotated[
        bool, typer.Option("--truth", help="Also print simulator ground truth (humans only)")
    ] = False,
    data_dir: DataDir = None,
) -> None:
    """Print an experiment's arms, metrics and pre-registered plan."""
    store = Store(data_dir)
    ids = [experiment_id] if experiment_id else [r.experiment_id for r in store.list()]
    truth_store = TruthStore(data_dir) if truth else None
    for i, eid in enumerate(ids):
        try:
            text = format_experiment(store.get(eid), store.arm_counts(eid))
        except ExperimentNotFoundError:
            typer.echo(f"No experiment {eid!r}.", err=True)
            raise typer.Exit(1) from None
        if i:
            typer.echo("")
        typer.echo(text)
        if truth_store is not None:
            typer.echo(format_truth(truth_store, eid))


def format_experiment(exp: Experiment, counts: dict[str, int]) -> str:
    p = exp.plan
    primary = exp.metric(p.primary_metric)
    arms = ", ".join(
        f"{a.name}{' (control)' if a.is_control else ''} {a.expected_share:.0%} "
        f"n={counts.get(a.name, 0):,}"
        for a in exp.arms
    )
    metrics = "; ".join(
        f"{m.name} [{m.type}{'' if m.higher_is_better else ', lower is better'}]"
        for m in exp.metrics
    )
    guardrails = ", ".join(g.metric for g in p.guardrails) or "none"
    baseline = f"{p.baseline:.4g}" if p.baseline is not None else "n/a"
    looks = (
        ", ".join(str(d) for d in p.planned_looks) if p.planned_looks else "none (fixed horizon)"
    )
    lines = [
        f"{exp.experiment_id}  {exp.name}",
        f"  {exp.description}" if exp.description else None,
        f"  {exp.source} | {exp.status} | unit: {exp.unit} | {exp.start_date} to {exp.end_date}"
        f"{' | daily rows' if exp.has_daily else ''}",
        f"  arms:       {arms}",
        f"  metrics:    {metrics}",
        f"  covariates: {', '.join(exp.covariates) or 'none'}",
        f"  segments:   {', '.join(exp.segments) or 'none'}",
        "  plan:",
        f"    primary:          {primary.name} [{primary.type}], baseline {baseline}, "
        f"MDE {p.mde:.1%} relative",
        f"    secondary:        {', '.join(p.secondary_metrics) or 'none'}",
        f"    guardrails:       {guardrails}",
        f"    alpha / power:    {p.alpha:g} {'two' if p.sides == 2 else 'one'}-sided / {p.power:g}",
        f"    planned n:        {p.planned_n_total:,} total",
        f"    planned looks:    {looks}",
        f"    multiple testing: {p.multiple_testing}",
        f"    decision rule:    {p.decision_rule.describe()}",
    ]
    return "\n".join(line for line in lines if line is not None)


def format_truth(truth_store: TruthStore, experiment_id: str) -> str:
    try:
        t = truth_store.get(experiment_id)
    except ExperimentNotFoundError:
        return "  truth: none (not simulated)"
    effects = ", ".join(
        f"{e.metric}@{e.arm} {e.relative:+.1%}" for e in t.true_effects if e.relative != 0
    )
    return f"  TRUTH knobs: {', '.join(t.knobs)}\n  TRUTH effects: {effects or 'all zero'}"


if __name__ == "__main__":
    app()
