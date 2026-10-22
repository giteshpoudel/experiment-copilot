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

## Phase 3: MCP server (2026-10-19)

### Decisions

- **fastmcp pinned `>=4.1,<5`.** The plan (and my Phase 1 note) assumed v2 or v3, but 4.1.1 was
  current. The APIs (decorator tools, URI-template resources, in-memory `Client`, structured
  output from Pydantic return types) were confirmed with a smoke test before planning.
- **Three layers:** `analysis.py` (load data, pick the method, call `stats/`), `results.py`
  (Pydantic outputs), `server.py` (thin FastMCP wrappers plus docstrings). The Phase 5 oracle
  can call `Analyzer` directly and run exactly the agent's code path.
- **Numbers are rounded to 6 significant figures, and every result has a quotable `summary`**
  (e.g. `+11.6% (95% CI +4.18% to +19%), p = 0.0012`), so the agent never formats or converts
  numbers itself.
- **Deterministic result ids**: `r-` + 8 hex characters of sha1(tool, inputs, data version).
  Comparisons are `<id>.<arm>`. `adjust_pvalues` resolves ids from the same server process.
- **Guardrail verdicts are computed by the tool** (worst-case CI bound vs margin), not by the
  agent.
- **Probabilities are never shown as a flat "100%"**: anything above 99.99% displays as
  "> 99.99%".
- **The store opens read-only for the server**: no schema creation and no writes (guarded
  with PermissionError). The server data directory never contains `truth.duckdb`.
- **Sequential tests on non-daily data** use units enrolled by day d, with final outcomes
  treated as observed; the tool reports this in `assumptions`. Information fraction =
  enrolled / max(planned n, final n).

### Bugs caught

- **Tool descriptions did not all say when to use the tool** (`check_srm`, `describe_experiment`,
  `segment_effects`). A test now requires "Use ..." in every tool description.

### Observed

- **The novelty experiment (smoke-16) "stops" early under O'Brien-Fleming** on day 3: the early
  spike really is in the data. Only `daily_effects` shows that the lift fades (day 1 +48.8%,
  day 14 +6.6%, trend p = 0.011). This is a good eval scenario: a correct agent must run
  daily_effects and flag the stop as novelty-driven.

## Phase 4: readout agent (2026-10-19)

### Decisions

- **DeepSeek through the `openai` SDK** (the user has a Claude Pro subscription but no API
  credits). Agent model `deepseek-v4-pro` (V4-Pro-0813); judge `deepseek-flash`
  (V4.1-Flash-1017), so the judge is a different model. The `llm.py` interface is narrow, so a
  Claude adapter can be added later. Every result records the model the API reports, because
  DeepSeek remaps aliases silently.
- **Thinking mode on**, `reasoning_effort=high`. `reasoning_content` is sent back on every
  later request in a tool-using turn (DeepSeek returns 400 otherwise). Temperature is ignored
  in thinking mode, so runs vary; that is why Phase 5 runs each config 3 times.
- **Manual loop** with caps of 12 turns and 30 tool calls. When a cap is hit, the model gets a
  final `tool_choice="none"` turn and the readout is marked truncated. Tool errors and invalid
  JSON arguments go back to the model as `ERROR: ...` tool messages.
- **Short `[rN]` labels in readouts**, mapped in the trace to the server's stable
  `r-xxxxxxxx` ids. One template (`readout.py`) serves both the MCP prompt (cites result ids)
  and the agent (cites labels).
- **Cost accounting** uses DeepSeek's cache-hit and cache-miss token counts and the
  peak/off-peak price table.

### Observed in the first live runs (deepseek-v4-pro)

| Run | Decision | Turns / tool calls | Tokens (cache hits) | Cost | Time |
| --- | --- | --- | --- | --- | --- |
| smoke-15 agent | INVALID (SRM) | 3 / 2 | 12.4k (7.9k) | $0.0049 | 12 s |
| smoke-03 agent | DON'T SHIP (bounce guardrail failed) | 4 / 4 | 18.2k (13.1k) | $0.0093 | 28 s |
| smoke-03 baseline | SHIP | 1 / 0 | 0.7k in, 11k out | $0.0222 | 84 s |

- **The baseline did statistics in prose and made a unit error.** It computed its own SEs,
  CIs, z-test and SRM χ² (the core rule forbids this), then compared bounce's *absolute* CI
  upper bound (0.0206) with the plan's *relative* 5% margin (about 0.0148 here). Under the
  plan's rule the guardrail fails, so its SHIP was wrong. This is a "what the evals caught"
  example: tools return the guardrail verdict, so the agent never has to handle margin units.
- **The guardrail rule was underpowered (D-19, now settled; see below).** Smoke-03's true bounce effect is 0,
  yet the non-inferiority check failed: the relative CI half-width is about 3.5% at 15k per
  arm, so with a 5% margin a harmless change fails about 20% of the time. The agent applied
  the rule correctly. The rule itself needs a decision before the Phase 5 oracle labels
  scenarios.

### Decision: guardrails fail only on statistically significant harm (D-19, user's choice)

- Replaced the non-inferiority rule ("the CI's bad-side bound must stay inside a 5% relative
  margin") with **"fail only if the CI at the plan's alpha lies entirely on the harmful side
  of zero"**. `GuardrailSpec` no longer has a margin; `DecisionRule.guardrail_check` is
  `no_significant_harm`.
- Why: with a 5% margin at 15k per arm, a change with zero true harm failed about 20% of the
  time (smoke-03). The new rule blocks a harmless change about 2.5% of the time (one tail at
  alpha 0.05).
- Trade-off, accepted: an underpowered guardrail can now miss a small real harm, because
  "not significant" is not "no harm". Readouts should say so when a guardrail is inconclusive
  with a worse-looking point estimate. Phase 5 should include a scenario with a small real
  guardrail harm to measure this.
- Effect on smoke-03: bounce +3.45% (CI −0.124% to +7.03%) is now a pass, so its decision under
  the rule is SHIP. The saved Phase 4 readout (`results/readouts/smoke-03-agent.json`) predates
  this change and shows DON'T SHIP.

## Phase 5: evals (2026-10-20)

### Setup

- 80 scenarios (`evals/scenarios.jsonl`), 40 dev / 40 test, stratified by knob. Labels come
  from a deterministic oracle that applies the pre-registered rule using the same validated
  `Analyzer`, plus truth labels from the simulator. A slow test checks that the labels
  regenerate identically from the seeds; `eval-build` refuses to overwrite them.
- Rule clarifications made before any eval run (so the oracle is unambiguous): the plan
  pre-registers the CUPED covariate (`primary_covariate`); with planned looks, the primary is
  significant only if the O'Brien-Fleming boundary was crossed; with several arms, an arm
  qualifies on its Holm-adjusted p and the best qualifying arm with passing guardrails ships.
- Oracle vs truth disagree on 10/80 for known reasons: 5 power misses (true effect at or
  below the MDE, not significant), 3 novelty false ships, 1 small guardrail harm missed by the
  significant-harm rule, 1 small effect.
- Configs: `agent` (deepseek-v4-pro), `baseline` (same model, plan plus per-arm means only),
  `agent-flash` (deepseek-flash). 3 runs each. Judge: flash for the Pro configs, Pro for the
  Flash config.

### Dev split (40 scenarios x 3 configs x 3 runs, off-peak, $4.37, 28 min)

- Agent (V4-Pro): 100% decision accuracy vs oracle, 100% required checks, 99% grounded, judge
  4.7, $0.0066 and 34 s per readout. Flash agent: also 100% / 100%, 97% grounded, $0.0016, 19 s.
  No prompt changes were made after dev, so no dev rerun.
- Baseline: 71% (68–75%) decision accuracy, 49% grounded, 5% unparseable decisions, $0.0197 and
  102 s per readout. Its misses are mostly "don't ship" where the oracle ships, concentrated
  in the CUPED revenue scenarios: without covariate data it cannot use CUPED, so its own
  arithmetic is underpowered.
- The agent's only false ships were the two dev novelty scenarios (e052, e053), as designed:
  the rule says ship. It ran `daily_effects` and discussed novelty in every one.

### Found during dev review

- **The daily trend test is underpowered for mild novelty (open).** In e053 (initial +30%,
  half-life 3 days) the test reports p = 0.29, "no significant downward trend", with about 100
  conversions per arm per day. The agent quoted that faithfully; the judge, which knows the
  scenario has novelty, scored its caveats 2/5. It's the guardrail lesson again: "not
  significant" is not "no effect". Options: say in the tool output that the test has low power
  at this volume; compare first-half vs second-half effects; or fit a decay curve. Not changed
  mid-eval, so dev and test stay comparable.
- **Grounding misses are mostly harmless format conversions** in the agent ("50/50 split" from
  shares of 0.5, "80% power" from 0.8). The scorer is deliberately strict; these are left as
  ungrounded.

### Test split (40 scenarios x 3 configs x 3 runs, off-peak, $4.35, 26 min)

| Metric | agent (V4-Pro) | agent-flash | baseline |
| --- | --- | --- | --- |
| Decision accuracy vs oracle | 98% (98-100%) | 100% | 73% (68-78%) |
| Agreement with true effect | 86% (85-88%) | 88% | 62% (57-65%) |
| False-ship rate | 12% | 12% | 12% |
| Required checks | 100% | 100% | n/a |
| SRM detected | 100% | 100% | 89% (67-100%) |
| Grounded numbers | 99% (98-99%) | 96% | 47% (45-48%) |
| Unparseable decision | 1% (0-2%) | 0% | 3% (0-5%) |
| Judge (1-5) | 4.7 | 4.8 | 4.2 |
| Cost / latency | $0.0072 / 33 s | $0.0016 / 18 s | $0.0191 / 95 s |

- The oracle's own truth agreement on test is 88%, with a 12% false-ship rate: that is the ceiling
  for any rule-following agent. The agents' false ships are exactly the oracle's (e054 novelty,
  e061 small guardrail harm).
- **Bug found on test (open): relative and absolute CIs can disagree.** e042: absolute CI 8.07e-05
  to 0.0139 (p = 0.047, direction "better"), relative delta-method CI −0.157% to +14.1%. The
  `summary` shows the relative CI next to the verdict "better", so the tool contradicts itself.
  Agent runs: ship (correct), don't ship (trusted the relative CI), and no decision (hit
  max_tokens deliberating). Ties to the Phase 2 open item that relative CIs slightly under-cover.
  Proposed fix: a Fieller interval for the ratio. At θ = 1 its test statistic is exactly the
  unpooled Wald z for the difference, so "relative CI excludes 0" ⇔ "absolute test rejects", and
  coverage improves. It was found on the test split, so any rerun after the fix must be
  disclosed as such.
- One agent tool call passed an unexpected argument; the server's validation error went back to
  the model, which recovered.

### Fix: Fieller relative CIs (2026-10-22)

- Replaced the delta-method relative CI with a Fieller interval (`fieller_relative_ci`) in every
  `from_arm_means` estimator. At ratio = 1 its statistic is the unpooled Wald z for the
  difference, so "relative CI excludes 0" ⇔ "absolute test rejects", including with Welch's t.
  `rel_se` is still the delta-method se, for reference.
- Tests: endpoints match scipy root-finding to 1e-10; on 5,000 near-boundary cases the new
  interval never disagrees with the test (the old one disagreed 1,248 times); unbounded when
  the control mean is not distinguishable from 0.
- `expc validate`: relative coverage rose from 94.1–94.35% to 94.25–94.9%; absolute coverage
  unchanged. Closes the Phase 2 open item on relative-CI under-coverage.
- Eval labels are unchanged (the oracle never used the relative CI; the slow reproduce test
  passes).
- **Disclosure:** found by inspecting a test-split failure (e042). The README keeps the test
  numbers as measured before the fix. Only dev was rerun afterwards (agent configs only; the
  baseline never sees tool output): agent 100% / 100% (decision accuracy / required checks),
  99% grounded, $0.0062 and 26 s per readout; Flash 100% / 100%, 97%, $0.0015. No regression.
  Rerun cost: $1.81.

## Phase 6: real-data case studies (2026-10-22)

### Decisions

- **Hillstrom plan (retrospective, written before the analysis):** primary spend with CUPED on
  `history`; secondaries visit and conversion; Holm across the two e-mail arms; no guardrails.
  The raw CSV (3.96 MB, minethatdata.com) is pinned by SHA-256 and never committed.
- **Criteo:** a seeded DuckDB reservoir sample of 1M users; planned shares 85/15, so the SRM
  check must pass. Criteo's own link returned 404, so the user downloads the file by hand. The
  loader warns when the file's SHA-256 differs from the value published by libuplift, and the
  data is CC BY-NC-SA.
- **Uplift engine** (`stats/uplift.py`): T- and X-learners on HistGradientBoosting; Qini curve
  and coefficient match scikit-uplift to 1e-10 (a test-only dependency); evaluation on a
  held-out 30%. Exposed as the exploratory `uplift_model` tool (12 tools now).

### Findings on Hillstrom

- The agent ships the men's e-mail: spend +117% (95% CI +64% to +197%), Holm p < 0.0001; the
  women's e-mail +64.7%, p = 0.0012. $0.0118, 4 tool calls. The baseline said DON'T SHIP because
  it could not build a CI from means alone.
- **CUPED reduces variance by under 1% here** (spend: 0.04% with `history`; best case visit with
  `recency`: 0.66%, a CI 0.997× as wide). Two-week spend is sparse and barely predicted by
  past-year spend. This is the honest value for the kit's "[R]%" row.
- The women's e-mail lifts spend mainly for new customers (+197% vs +12.2%, Cochran's Q
  p = 0.018). Only it is worth targeting: visit Qini 0.039 (T) / 0.041 (X) vs about 0 for the
  men's e-mail.

### Bugs and test changes

- **Large percentages printed in scientific notation** (`+1.14e+03%`) on real data. `fmt_pct`
  now prints values of 1,000% and above as whole numbers.
- **Uplift test expectation (my error, not a tolerance relaxed to pass):** I first required Qini
  > 0.1 on the synthetic data, but Qini is normalized by a "perfect" ranking built from realized
  outcomes, so even the true-effect score only reaches 0.121 there. The test now requires the
  learners to reach 40% of that ceiling (they reach about 55–60%), and the random-score check
  was tightened from |Qini| < 0.1 to < 0.03.
- DuckDB deprecated `fetch_arrow_table()`; switched to `to_arrow_table()`.

### Open

- The Criteo case study runs when `data/raw/criteo-research-uplift-v2.1.csv.gz` is present:
  `expc load criteo`, then `python scripts/case_study.py criteo`, plus a readout.

## Phase 7: demo, CI, polish (2026-10-22)

- **Streamlit demo** (`app/streamlit_app.py`): replay mode over saved runs needs no key or
  data, so it can be hosted free (`app/requirements.txt` lists only Streamlit). It shows the
  cited readout beside each tool call and the no-tools baseline, plus the eval table and the
  validation checks. Live mode appears only with `LLM_API_KEY` and a local store. AppTest
  renders every saved run.
- **CI:** lint and fast tests on every push and PR; slow validation weekly or on demand; a paid
  smoke eval only by hand, which needs the `LLM_API_KEY` repository secret (the user adds it).
- **Fresh-clone check:** a clean clone with a new venv passes lint and 157 tests (2 skipped:
  the raw-file check and the live API test). The data regenerates, the eval labels come out
  identical to the committed file, and the MCP server serves 12 tools over stdio.
- **Found while testing the demo:** Streamlit's AppTest applies a selectbox `format_func`
  twice when selecting by index, so the selector now shows plain ids with a caption.

### Criteo case study (2026-10-22)

- The user downloaded `criteo-research-uplift-v2.1.csv.gz` (296 MB) from the
  `jszymon/uplift_sklearn_data` GitHub mirror after Criteo's own link returned 404. Its SHA-256
  matches the value libuplift publishes (`2716e1bf…`). The loader took 12 s to sample 1M users.
- **SRM passes on the planned 85/15 split** (observed 15.01% / 84.99%, p = 0.71). The agent ships:
  visit +30.2% (95% CI +26.8% to +33.9%); conversion +38.4%. 4 tool calls, $0.0065, 24 s.
- **The baseline hit max_tokens (16,000) while reasoning and returned no text** ($0.0321, 125 s).
  Reported as is; it was not rerun with a larger budget, so it stays comparable with the eval.
- **Uplift, held-out 30% (149,941 users):** T-learner visit Qini 0.0517 (top decile +5.66 points
  vs +1.21 overall; deciles 3–9 ≈ 0); X-learner 0.0316. Conversion: T 0.0898, X −0.0159. The
  T-learner wins despite the usual advice that the X-learner is better for unbalanced arms; at a
  0.3% conversion rate the X-learner's second stage fits very noisy imputed effects.
- **Caveat: the ranking isn't monotone.** The bottom decile also has large real uplift (+2.96
  points, T, visit). Extreme feature values get extreme predictions in both directions. The
  report shows deciles next to the Qini for this reason.
- Phase 6 "Done when" is now met: both case studies have a readout and website numbers.
