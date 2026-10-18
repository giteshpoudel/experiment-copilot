# Experimentation Copilot

An MCP server that gives any AI client a toolkit for analyzing A/B tests, plus an agent that writes
cited ship / don't-ship readouts, measured against simulated experiments with known true effects.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for scope and phases. Work in progress: Phase 1 (scaffold,
store and simulator).

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
expc simulate --suite smoke
expc list
expc describe smoke-01
pytest -m "not slow"
```
