# Experimentation Copilot

An MCP server that gives any AI client a toolkit for analyzing A/B tests, plus an agent that writes
cited ship / don't-ship readouts, measured against simulated experiments with known true effects.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for scope and phases. Status: Phases 1–3 done (store and
simulator, validated statistics engine, MCP server). Next: the readout agent.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
expc simulate --suite smoke      # 20 simulated experiments with known truth
expc list
expc describe smoke-15
expc validate                    # statistics validation -> results/validation.json
pytest -m "not slow"
```

## Use it from Claude Code or Claude Desktop

The server is read-only and talks MCP over stdio: `expc serve`.

**Claude Code:** this repo ships a project [`.mcp.json`](.mcp.json). Run `claude` in the repo
folder and approve the `experiment-copilot` server when asked. Then try:

> Should we ship smoke-15? Use the experiment-copilot tools and the experiment_readout prompt.

**Claude Desktop:** add this to `claude_desktop_config.json` (absolute paths required):

```json
{
  "mcpServers": {
    "experiment-copilot": {
      "command": "/absolute/path/to/experiment-copilot/.venv/bin/expc",
      "args": ["serve", "--data-dir", "/absolute/path/to/experiment-copilot/data"]
    }
  }
}
```

**MCP Inspector:** `npx @modelcontextprotocol/inspector .venv/bin/expc serve`

### Tools

| Tool | What it answers |
| --- | --- |
| `list_experiments`, `describe_experiment` | What's in the store; arms, metrics and the pre-registered plan |
| `check_srm` | Is the traffic split what was planned? |
| `analyze_metric` | Effect, CI and p-value (z-test, Welch, delta method, bootstrap, CUPED); guardrail verdicts |
| `adjust_pvalues` | Holm / Benjamini-Hochberg across arms or metrics |
| `sequential_test` | Valid results after peeking (O'Brien-Fleming spending, mSPRT) |
| `daily_effects` | Effect by day and a novelty-decay test |
| `segment_effects` | Exploratory effects by segment |
| `bayesian_ab` | P(treatment better), expected loss |
| `power_analysis` | Sample size or MDE |
| `calculate` | Exact arithmetic, so prose never does math |

Resources: `experiments://registry`, `experiments://{id}/plan`, `metrics://definitions`.
Prompt: `experiment_readout(experiment_id)`.

## Validation

Measured on thousands of simulated experiments; see `results/validation.json`.

| Check | Result |
| --- | --- |
| 95% CI coverage (z, Welch, delta, CUPED) | 94.1–94.8% |
| A/A false-positive rate | 4.5–4.9% |
| 14 daily peeks: naive / O'Brien-Fleming / mSPRT | 22.3% / 4.75% / 0.35% |
| CUPED CI width at ρ = 0.5 / 0.9 | 0.935× / 0.607× |
