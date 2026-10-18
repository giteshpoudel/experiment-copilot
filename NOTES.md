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

- ~~`planned_n_total` values in the smoke suite are hand-computed approximations to 80% power.~~
  Closed in Phase 2: recomputed with `stats.power` (see below).

## Phase 2: statistics engine, validated (2026-10-18)

### Decisions

- **Engine returns frozen dataclasses** (`EffectEstimate`, `SRMResult`, `SequentialResult`,
  ...). The Phase 3 server wraps them in Pydantic `ToolResult`s with result ids. This keeps
  `stats/` free of I/O and fast inside validation loops.
- **Two-proportion z-test uses the unpooled (Wald) se for both the p-value and the CI**, so
  "CI excludes zero" and "p < alpha" always agree. The decision rule is written in terms of
  the CI. The pooled-se z-test can disagree with its own Wald CI near the boundary.
- **CUPED theta is the pooled within-arm slope.** The adjusted difference then equals the OLS
  ANCOVA treatment coefficient exactly (tested to 1e-10); the se agrees with HC2 to about 1%.
- **Relative CIs use the delta method** on the ratio of independent arm means.
- **Group sequential uses two-sided Lan-DeMets O'Brien-Fleming spending**, with boundaries from
  a numerical recursion on a 1,201-point Simpson grid. Any look schedule works.
- **mSPRT's tau is the plan's absolute MDE.**
- **Uplift (T-/X-learners, Qini) moves to Phase 6** with Criteo. Phase 2 has segment effects
  with Holm/BH adjustment plus Cochran's Q for heterogeneity.
- **Validation pass bands are the target ± 3 binomial SEs** for the reps run (±1.46 pp at
  2,000 reps), so they widen automatically for quick runs.
- **Peeking validation uses the planned information fractions** k/14 (uniform enrolment) so
  the boundaries are computed once. The Phase 3 tool will use the realized fractions.

### Bugs caught

- **Wrong two-sided spending function** (caught by the published-values test). I first used
  the one-sided OBF form 2 − 2Φ(z_{α/2}/√t) with the total α. It also spends exactly α (the
  SciPy multivariate-normal check passed), but it front-loads: the first of five equal looks
  got a boundary of 4.38 instead of the standard 4.88. The correct two-sided form is
  4 − 4Φ(z_{α/4}/√t), which now matches 4.877 / 3.357 / 2.680 / 2.290 / 2.031. Lesson: the
  total-α check alone could not catch it.
- **Infinite boundaries broke the recursion grid.** With 14 daily looks, look 1 spends about
  1e-16 of α, so its boundary is infinite. Such looks now never stop, and the grid is truncated
  at |z| = 8.
- **pytest collected statsmodels' `test_proportions_2indep`** as a test because it was imported
  under its own name. It is now imported under an alias.
- **numpy bools broke `json.dumps`** in the validation report. Values are cast to Python types.
- **Smoke-suite planned n was off by up to 70%** (D-07). Recomputed from a 400k-unit pilot with
  `stats.power`: revenue needs 33,414 total at rho 0.5 (was 37,000) and 21,916 at rho 0.9
  (was 37,000); revenue_per_session needs 33,896 (was 40,000). Plans and n_total now use the
  power-derived n rounded up to the next 1,000, and a slow test recomputes them.

### Tolerance changes

- `test_bootstrap_close_to_scipy`: my first tolerance (5% of the se) assumed the Monte Carlo
  error of a 2.5% quantile at B=4,000 was about 2% of the se. It is actually
  sqrt(q(1-q)/B)/φ(z_q) ≈ 4%, and two independent bootstraps differ by about 6%. The observed
  gap of 7.7% was 1.3 sd. Changed to B=20,000 with a tolerance of 0.11 se (about 4 sd).
  This is a test-design error, not an engine bug.

### Validation results (2,000 reps, n=10,000 per experiment)

- CI coverage, nominal 95%: 94.1% to 94.8% for z, Welch, delta method and CUPED. The
  bootstrap covers 94.0% (300 reps, n=4,000, B=1,000).
- A/A false-positive rate: 4.5% to 4.9% (bootstrap 4.3%).
- Peeking with 14 daily looks: naive 22.3%, OBF 4.75%, mSPRT 0.35%.
- CUPED CI width relative to Welch: 1.00 / 0.935 / 0.607 at rho 0 / 0.5 / 0.9, matching
  sqrt(1 − r²) for the realized r.

### Open

- **Relative CIs slightly under-cover.** Every method's relative CI covered 94.1% to 94.35%,
  below its absolute CI, though inside the band. The delta-method ratio interval is mildly
  anti-conservative at n=10k with skewed revenue. Consider Fieller or a log-ratio interval if
  smaller experiments matter.
- **mSPRT is very conservative here** (0.35% vs 5%), which will cost power. The tau choice
  could be tuned, but tuning it on the validation results themselves would be circular.
