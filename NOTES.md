# Notes: bugs found and design decisions

Running log for the README's "what the evals caught" section. Newest last.

## Phase 1: scaffold, store, simulator (2026-10-18)

### Decisions

- **Decision rule is structured, not prose** (`models.DecisionRule`). Order: invalid if
  SRM p < 0.001 → ship if the primary CI excludes zero on the good side and every guardrail CI
  stays within its non-inferiority margin → extend if inconclusive and n < planned n → don't
  ship. Structured so the Phase 5 oracle can apply it mechanically.
- **MDE is relative** (0.10 = 10%); the plan also stores the expected control `baseline`.
- **Planned sample is `planned_n_total`**, not per arm, so unequal splits (Criteo's 85/15) need
  no special case. The "extend" rule compares observed total units to it.
- **Units in Parquet per experiment; the registry in DuckDB.** Queries use
  `read_parquet(path)` directly instead of registered views, which would hard-code absolute
  paths into the database file.
- **Ground truth lives in a separate `truth.duckdb`.** The MCP server will only get a `Store`,
  so it cannot read answers. `expc describe --truth` prints them for humans.
- **Experiment ids and names are neutral** (`smoke-07`, "bundle-discounts"). An early plan had
  ids like `sim-smoke-07-ratio-srm`, which would hand the agent the answer. Knob tags are in
  the truth store only; a test checks names and descriptions for leaking words.
- **Simulated metrics:** conversion (binary), revenue (zero-inflated lognormal), sessions,
  revenue_per_session (ratio), bounce (binary, lower is better; the default guardrail).
- **Effects are exact relative changes in population means**, including with segment
  heterogeneity (multipliers are rescaled to a share-weighted mean of 1).
- **Novelty decays additively toward the steady state:**
  `effect(d) = steady + (initial - steady) * 0.5**(d/h)`. The first draft multiplied the
  steady-state effect by a decaying factor, which made novelty impossible when the true effect
  is zero, the most important case. The truth stores the steady-state effect; smoke-16 has a
  true effect of 0, but its 14-day average looks like about +10%.
- **Daily rows** exist only for experiments with `has_daily`. All units start on day 0, and
  each converter's purchase is placed on one day with weights `1 + effect(d)`, which makes the
  daily treatment effect exactly `effect(d)`.

### Bugs caught

- **CUPED knob had almost no effect.** Conversion was tied to the engagement latent while
  pre_revenue came from the spend latent, so corr(pre_revenue, revenue) was 0.04 at rho=0.5 and
  0.08 at rho=0.9 (revenue is about 90% zeros). Fix: conversion and order value both load on
  the spend latent. Their correlation means E[revenue] needs a closed form for the scaling
  factor k (exponential tilting: E[e^{σv}·1{u>t}] = e^{σ²/2}·Φ(cσ − t), with c = rho²).
  Realized correlation is now 0.00 / 0.35 / 0.79 at rho 0 / 0.5 / 0.9, and a 1M-unit check
  shows every effect unbiased.
- **Consequence:** the revenue baseline now depends on rho (8.25 at rho 0.5, 17.4 at rho 0.9),
  because buyers are the high spenders. `baseline()` accounts for it.

### Tolerance changes

- `test_realized_effects_match_truth[smoke-07]` (revenue +40%) failed at 200k units:
  +0.328. Investigation: across 20 seeds the estimate averaged +0.412 with sd 0.023, and a 1M
  run gave +0.399, so there is no bias. The original ±0.06 was only about 2.6 sd for skewed
  revenue. Changed to 1M units and ±0.045 (about 4 sd). The other cases are unchanged.

### Open

- `planned_n_total` values in the smoke suite are hand-computed approximations to 80% power.
  Recompute them with `stats.power` in Phase 2, especially for the revenue plans, whose variance
  depends on rho.
