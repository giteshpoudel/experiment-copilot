"""`expc` command line: simulate, list, describe, validate.

Later phases add serve, readout and eval.
"""

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
    guardrails = ", ".join(f"{g.metric} (margin {g.margin:.0%})" for g in p.guardrails) or "none"
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
