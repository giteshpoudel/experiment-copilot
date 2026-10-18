# CLAUDE.md — Experimentation Copilot

Read `PROJECT_PLAN.md` first. It is the source of truth for scope, tools, phases and repo layout.

## The core rule

Numbers come from tools. The agent chooses methods and explains. Every number in a readout cites the tool result it came from. Never let LLM-written prose compute or round a number that a tool did not return.

## How to work

- Build phases in order (PROJECT_PLAN.md section 6). Do not start a phase until the previous phase's "Done when" is true; say so when it is.
- At the start of each phase, write a short plan for that phase and wait for my go-ahead before coding.
- The statistics engine (`src/expcopilot/stats/`) is pure Python: no LLM, MCP or I/O code in it.
- Every stats function gets a unit test against SciPy or statsmodels on fixed inputs before it is used anywhere else.
- When a simulation check fails, investigate the statistics before changing the test's tolerance. Tell me if you change a tolerance, and why.
- Keep a running `NOTES.md` of bugs found and design decisions; the README's "what the evals caught" section is written from it.

## MCP tool design

- Tools take experiment ids and metric names, never raw data arrays.
- Every tool input and output is a Pydantic model. Outputs include `result_id`, the inputs used and the method name.
- Tools are read-only.
- Keep outputs small: summaries, not rows. Cap list lengths.
- Docstrings are the tool descriptions the model sees: say when to use the tool and when not to.

## Evals

- Tune prompts on the dev split only. Report on test.
- Run each config 3 times and report mean and range.
- The judge model must differ from the answering model.
- Never edit `evals/scenarios.jsonl` labels to make a run pass; if a label looks wrong, flag it to me.

## Conventions

- Python 3.11+, type hints everywhere, ruff for lint and format, pytest.
- Mark slow simulation tests with `@pytest.mark.slow`.
- Config and secrets in `.env` (gitignored); keep `.env.example` current.
- Never commit anything under `data/`.
- Pin `fastmcp` to one major version.

## Commands

```
pip install -e ".[dev]"
pytest -m "not slow"          # fast tests
pytest -m slow                # simulation validation
expc simulate --suite smoke   # create simulated experiments
expc serve                    # run the MCP server (stdio)
expc readout <experiment_id>  # agent readout with tool trace
expc readout <id> --no-tools  # baseline
expc eval --split dev         # eval suite
```
