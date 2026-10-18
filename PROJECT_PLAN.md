# Project Plan: Experimentation Copilot

An MCP server that gives any AI client a toolkit for analyzing A/B tests, plus an agent that uses those tools to write ship / don't-ship readouts. It is **measured**: a suite of simulated experiments with known true effects scores whether the agent reaches the right decision, uses the right method, and reports only numbers that came from tools.

Built with free, open data and open-source tools. The only paid piece is the LLM API.

---

## 1. The core rule

> **Numbers come from tools. The agent chooses methods and explains. Every number in a readout cites the tool result it came from.**

Same rule as `edgar-rag-analyst`, applied to experimentation. The skills this project shows:

| Skill | What it means here |
| --- | --- |
| **MCP server design** | Typed tools, resources and a prompt that work in Claude Desktop, Claude Code or any MCP client |
| **Function calling / agents** | An agent that plans an analysis, calls tools in order and writes a cited readout |
| **Experimentation statistics** | Power, SRM, CUPED, delta method, sequential tests, Bayesian A/B, multiple testing, uplift |
| **Validation by simulation** | CI coverage and false-positive rates measured on thousands of simulated experiments |
| **Evals** | Scenario suite with ground truth; agent vs an LLM-without-tools baseline |

Example questions it should answer:

- "Should we ship variant B of checkout-redesign? Write the readout."
- "We peeked every day for two weeks. Is this result still valid?"
- "Revenue per session went up but conversion went down. What's going on?"
- "Which customer segments respond best to the Womens email in Hillstrom?"

---

## 2. How a readout flows

```
User: "Should we ship checkout-redesign?"
        │
        ▼
Agent (any function-calling LLM, via MCP client)
        │
        ├─ describe_experiment  → arms, metrics, pre-registered decision rule
        ├─ check_srm            → is the traffic split what was planned?
        ├─ analyze_metric       → primary metric (method chosen by metric type, CUPED if covariate)
        ├─ analyze_metric       → guardrail metrics
        ├─ adjust_pvalues       → if several metrics or arms
        ├─ segment_effects      → optional, flagged as exploratory
        │
        ▼
Readout: decision + reasoning, every number tagged [r1], [r2] … → tool results
```

Tools take **experiment ids and metric names, not raw data**. The data stays in the server's store; tool results are small, typed and citable.

---

## 3. Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Same as EDGAR |
| MCP server | `fastmcp` (pin the major version; v3 has breaking changes from v2) | Decorator tools from type hints; in-memory `Client` for tests |
| Data store | DuckDB over local Parquet | Embedded, fast group-bys, no server |
| Stats | NumPy, SciPy, statsmodels | Standard, and the reference for validation tests |
| Uplift | scikit-learn (hand-built T- and X-learners) | Shows the method, avoids a heavy dependency |
| Data models | Pydantic | Typed tool inputs and outputs |
| LLM | OpenAI-compatible client (DeepSeek or Claude), model set in `.env` | Same pattern as EDGAR |
| CLI | `typer` | `expc simulate`, `expc serve`, `expc readout`, `expc eval` |
| Demo | Streamlit | Hosted demo for the website |
| CI | GitHub Actions | Lint, tests, simulation checks, smoke eval |

---

## 4. Data

### Experiment store (DuckDB + Parquet)

- `experiments` registry: `experiment_id`, name, arms, expected split, unit, start/end, **pre-registered plan** (primary metric, guardrails, MDE, alpha, decision rule).
- Per-experiment unit table: `unit_id`, `arm`, `assigned_at`, segment columns, pre-period covariates, outcome columns (binary, continuous, and numerator/denominator pairs for ratio metrics), optional daily rows for time effects.

### Simulated experiments (the ground truth)

`simulate.py` generates experiments with known parameters. Every knob becomes an eval scenario:

- true effect: zero, below MDE, at MDE, large, negative
- metric type: binary, skewed continuous (lognormal revenue), ratio (revenue per session)
- covariate correlation with the outcome (for CUPED)
- heterogeneity by segment
- **injected bugs:** sample-ratio mismatch (drop a share of one arm), novelty effect that decays, guardrail harm
- peeking schedules for sequential tests
- several metrics and arms (multiple testing)

### Real data (case studies)

- **Hillstrom MineThatData email test:** 64,000 customers, three arms (Mens email, Womens email, no email), visit / conversion / spend over two weeks. Prior-year spend (`history`) is the CUPED covariate; three arms need multiple-comparison handling.
- **Criteo Uplift v2.1:** ~14M users, 12 features, treatment / visit / conversion / exposure. Treatment share is ~85% **by design**, a good test that the agent does not call a planned imbalance an SRM. Subsample for speed.

Download scripts cache to `data/` (gitignored). Never commit data.

---

## 5. Tools (MCP server)

Every tool returns a Pydantic model with a `result_id`, the inputs used, the method, and the numbers. All tools are read-only.

| Tool | Returns |
| --- | --- |
| `list_experiments()` | ids, names, status |
| `describe_experiment(experiment_id)` | arms, sample sizes, metrics with types, covariates, pre-registered plan |
| `check_srm(experiment_id)` | observed vs expected split, chi-square p-value, pass/fail at p < 0.001 |
| `power_analysis(metric_type, baseline, mde, alpha, power, ...)` | required n per arm, or the MDE for a given n |
| `analyze_metric(experiment_id, metric, method="auto", cuped_covariate=None, arms=None)` | effect (absolute and relative), CI, p-value, n per arm, method used, variance reduction if CUPED |
| `sequential_test(experiment_id, metric, looks, method="obrien_fleming" \| "msprt")` | boundaries per look, whether and when it stopped |
| `bayesian_ab(experiment_id, metric, prior=...)` | P(B > A), credible interval, expected loss |
| `adjust_pvalues(result_ids, method="holm" \| "bh")` | adjusted p-values per result |
| `segment_effects(experiment_id, metric, segment)` | effect per segment with adjusted p-values, flagged exploratory |
| `uplift_model(experiment_id, outcome, features, learner="t" \| "x")` | Qini coefficient, uplift by decile on a held-out split |
| `daily_effects(experiment_id, metric)` | effect by day, trend test (novelty check) |
| `calculate(operation, a, b)` | exact arithmetic for anything derived in prose |

`method="auto"` picks: two-proportion z-test for binary, Welch t-test for continuous, delta method for ratio metrics, bootstrap when asked.

**Resources:** `experiments://registry`, `experiments://{id}/plan`, `metrics://definitions`.
**Prompt:** `experiment_readout(experiment_id)`, the readout template and decision rules.

---

## 6. The phases

Build in order. Don't start a phase until the previous one's **Done when** is true.

### Phase 1: Scaffold, store and simulator

- Repo, `pyproject.toml`, ruff, pytest, `.env.example`, typer CLI.
- DuckDB store, registry schema, Pydantic models for experiments and results.
- `simulate.py` with every knob in section 4; `expc simulate` writes a set of experiments to the store.

**Done when:** `expc simulate --suite smoke` creates 20 experiments and `describe` prints their plans. **You learn:** data modeling for experiments, simulation design.

### Phase 2: Statistics engine, validated

- Pure functions in `src/expcopilot/stats/`, no LLM or MCP code.
- Unit tests compare every method against SciPy / statsmodels on fixed inputs.
- Simulation tests (marked `slow`): over thousands of simulated A/A and A/B tests,
  - 95% CIs cover the true effect ~95% of the time (per method)
  - A/A false-positive rate ~5%
  - naive daily peeking inflates false positives; the sequential test holds ~5%
  - CUPED shrinks CI width as covariate correlation rises
- Save these numbers to `results/validation.json`; they go in the README and on the website.

**Done when:** all validation checks pass and `results/validation.json` exists. **You learn:** the statistics, and how to prove they're right.

### Phase 3: MCP server

- Wrap the engine in `server.py` with FastMCP: tools, resources, prompt from section 5.
- Tests use FastMCP's in-memory client: every tool on a simulated experiment.
- Try it by hand with the MCP Inspector, then in Claude Desktop or Claude Code.

**Done when:** a Claude Desktop or Claude Code session can analyze a simulated experiment end to end using only the server's tools. **You learn:** MCP tool design and transport.

### Phase 4: Readout agent

- `agent.py`: an MCP client + function-calling LLM loop (cap the number of steps).
- Shared citation numbering: each tool result gets `[rN]`; the readout must cite them.
- Readout template: decision (ship / don't ship / extend / invalid), primary result, guardrails, validity checks, caveats, exploratory findings.
- Decision follows the experiment's pre-registered rule, not the agent's taste.
- Baseline mode `--no-tools`: same LLM, given only a summary table (counts and means per arm).

**Done when:** `expc readout <id>` prints a cited readout and the tool trace. **You learn:** agent loops over MCP, grounding.

### Phase 5: Evals

- `evals/scenarios.jsonl`: 60–100 simulated experiments covering every knob, each labelled with:
  - the **oracle decision** from a deterministic, correct analysis pipeline
  - the **true effect** sign and size
  - required checks (e.g. must detect SRM, must use the delta method)
- Split into dev and test; tune prompts on dev only.
- Metrics: decision accuracy vs oracle, agreement with the true effect, method-choice accuracy, SRM detection, number grounding (every number in the readout matches a tool result or a `calculate` call), false-ship rate, cost and latency per readout.
- LLM-as-judge only for the explanation quality, using a different model from the one answering.
- Run each config 3 times; report mean and range.

**Done when:** `expc eval` prints agent vs baseline on test, and the table is in the README. **You learn:** eval design with real ground truth.

### Phase 6: Real-data case studies

- Hillstrom: both email arms vs control, Holm-adjusted; CUPED with prior-year spend; segment effects by `history_segment` and `channel`.
- Criteo: subsample; planned 85/15 split must pass SRM; T- vs X-learner uplift, Qini on a held-out split.
- Write each up as a short notebook or markdown report with the agent's readout.

**Done when:** both case studies have a readout and the numbers for the website. **You learn:** messy real data, uplift modeling.

### Phase 7: Demo, CI and polish

- Streamlit demo: pick a scenario or case study, watch the agent's tool calls and readout.
- GitHub Actions: ruff, unit tests, fast simulation checks, smoke eval on a few scenarios (API key as a secret).
- README: diagram, the core rule, validation numbers, eval table, "what the evals caught", Claude Desktop config, limitations.

**Done when:** someone can clone, install, run the server in Claude Desktop, and reproduce the eval table.

---

## 7. Repo layout

```
experiment-copilot/
├── README.md
├── PROJECT_PLAN.md
├── CLAUDE.md
├── pyproject.toml
├── .env.example
├── data/                       # cached datasets + DuckDB file (gitignored)
├── src/expcopilot/
│   ├── models.py               # Pydantic: experiments, plans, tool results
│   ├── store.py                # DuckDB registry and unit tables
│   ├── simulate.py             # simulated experiments with known truth
│   ├── datasets/
│   │   ├── hillstrom.py
│   │   └── criteo.py
│   ├── stats/
│   │   ├── power.py
│   │   ├── srm.py
│   │   ├── frequentist.py      # z-test, Welch, bootstrap
│   │   ├── ratio.py            # delta method
│   │   ├── cuped.py
│   │   ├── sequential.py       # O'Brien-Fleming, mSPRT
│   │   ├── bayesian.py
│   │   ├── multiple.py         # Holm, BH
│   │   ├── hte.py              # segment effects, T/X-learner uplift, Qini
│   │   └── time_effects.py     # daily effects, novelty
│   ├── server.py               # FastMCP tools, resources, prompt
│   ├── agent.py                # MCP client + LLM loop, citations
│   ├── readout.py              # template and decision rules
│   ├── baseline.py             # no-tools LLM baseline
│   └── cli.py                  # expc simulate / serve / readout / eval
├── evals/
│   ├── scenarios.jsonl
│   ├── oracle.py               # deterministic correct analysis
│   ├── scoring.py
│   ├── run.py
│   └── results/
├── results/validation.json     # simulation validation numbers
├── app/streamlit_app.py
├── tests/
│   ├── test_stats_*.py         # vs SciPy / statsmodels
│   ├── test_validation.py      # slow simulation checks
│   └── test_server.py          # in-memory MCP client
└── .github/workflows/ci.yml
```

---

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| A subtle stats bug makes the whole agent wrong | Engine built and validated before any agent code (Phase 2) |
| The agent does math in prose | Number-grounding check in evals; `calculate` tool; prompt rule |
| Eval only tests what the simulator can produce | Real-data case studies; include messy scenarios (skew, outliers, bugs) |
| Eval noise between runs | 3 runs per config; report range |
| Criteo is large | Subsample; cache Parquet |
| Tools return too much data into context | Tools return summaries and ids only; cap list lengths |

## 9. Out of scope

Network interference and marketplace effects, long-term holdouts, observational (non-randomized) causal inference, a write-capable server (no creating or stopping experiments).

## 10. References

- MineThatData email challenge (Hillstrom): https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
- Criteo Uplift Prediction Dataset: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- FastMCP: https://github.com/PrefectHQ/fastmcp
- Model Context Protocol: https://modelcontextprotocol.io
