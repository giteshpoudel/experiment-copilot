# Case study: the Hillstrom e-mail test

**Both e-mails raised two-week spend, and the agent ships the men's e-mail, the larger winner
after Holm adjustment. The interesting part is what doesn't hold up on real data: CUPED with
prior-year spend barely helps, and the women's e-mail works mainly on new customers.**

Data: Kevin Hillstrom's [MineThatData e-mail test](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
(2008). 64,000 customers who bought in the past year were randomized three ways: a men's
merchandise e-mail, a women's merchandise e-mail, or no e-mail. Outcomes were visit,
conversion and spend over the next two weeks.

The plan below is **retrospective**, written for this case study before the analysis ran:
primary metric spend; secondaries visit and conversion; Holm across the two e-mail arms;
CUPED with prior-year spend (`history`); no guardrails.

Every number here comes from the tools: the agent's readout cites them, and the exploratory
numbers come from [`scripts/case_study.py`](../scripts/case_study.py), which calls the same
analysis code ([`results/case_studies/hillstrom.json`](../results/case_studies/hillstrom.json)).

## The agent's readout

`expc readout hillstrom`, DeepSeek V4-Pro: 5 turns, 4 tool calls, $0.0118, 38 s
([full JSON](../results/readouts/hillstrom-agent.json)).

```
## Decision: SHIP
Ship mens_email (vs no_email).

## Primary result
- mens_email vs no_email: +117% (95% CI +64% to +197%), p < 0.0001;
  absolute +0.7674 (0.654 -> 1.421); better [r3]. Holm-adjusted p < 0.0001 [r4].
- womens_email vs no_email: +64.7% (95% CI +22.2% to +127%), p = 0.0012;
  absolute +0.4227 (0.6536 -> 1.076); better [r3]. Holm-adjusted p = 0.0012 [r4].

## Validity checks
- SRM: passed. Observed split no_email 33.29% / mens_email 33.29% / womens_email 33.42%;
  chi2 = 0.2025, p = 0.9 [r2].

Tool trace:
  [r1] describe_experiment(experiment_id=hillstrom)                                  r-a75f0d36
  [r2] check_srm(experiment_id=hillstrom)                                            r-631d5954
  [r3] analyze_metric(experiment_id=hillstrom, metric=spend, cuped_covariate=history)  r-d42f178c
  [r4] adjust_pvalues(result_ids=[r-d42f178c.mens_email, r-d42f178c.womens_email])    r-bbcefd03
```

The no-tools baseline (same model, given the plan plus counts and means per arm) said
**DON'T SHIP**: without variances it could not build a confidence interval for spend, so it
could not apply the pre-registered rule. That is the right call for what it was given, and it
shows the gap: the tools are what make the plan executable. It also took twice as long and cost
more ($0.0196, 74 s).

## What the data say beyond the decision

**CUPED barely helps here.** CUPED narrows a confidence interval in proportion to how well
the pre-period covariate predicts the outcome. Two-week spend is mostly zeros (about 1% of
customers buy), and past-year spend hardly predicts it:

| Outcome | Covariate | Variance reduction (men's arm) | CI width vs no CUPED |
| --- | --- | --- | --- |
| spend | history | 0.04% | 0.9998× |
| visit | history | 0.44% | 0.9978× |
| visit | recency | 0.66% | 0.9968× |
| conversion | history | 0.09% | 0.9996× |

The simulation suite shows what CUPED does when the covariate is informative: 0.935× the CI
width at ρ = 0.5 and 0.607× at ρ = 0.9. Hillstrom is a reminder to check the correlation before
promising a speed-up.

**The women's e-mail works mainly on new customers** (exploratory, Holm-adjusted across
levels):

- Spend, women's e-mail, by `newbie`: Cochran's Q p = 0.018. New customers +197% (+85.3% to
  +442%), p < 0.0001; returning customers +12.2% (−24.9% to +69.3%), p = 0.57.
- The men's e-mail shows no such split (Cochran's Q p = 0.19): +258% for new customers and
  +61.8% for returning ones, both significant.
- By prior-year spend band, the women's e-mail lifts spend most in the lowest bands ($0–$100:
  +121%; $100–$200: +108%), with Cochran's Q p = 0.042. Bands above $750 have too few buyers
  for a bounded interval.

**Targeting helps for one e-mail, not the other** (uplift models on visit, trained on 70% of
customers and evaluated on the held-out 30%):

| E-mail | Qini, T-learner | Qini, X-learner | Visit uplift: top decile vs bottom decile (T) |
| --- | --- | --- | --- |
| Women's | 0.0387 | 0.0405 | +8.43 vs +1.89 points |
| Men's | −0.0078 | −0.0102 | +9.15 vs +9.08 points |

The men's e-mail lifts visits about equally for everyone, so a model can't rank anyone
higher. The women's e-mail has a responsive group, consistent with the new-customer split
above.

## Takeaways

1. **Decision:** ship the men's e-mail. Both e-mails beat no e-mail on spend after Holm; the
   men's has the larger lift.
2. **If both are sent,** the women's e-mail is the one worth targeting (at new and low-spend
   customers), since its effect is concentrated there.
3. **Don't count on CUPED for this kind of metric.** With sparse two-week outcomes and a weak
   pre-period covariate, it shortens nothing; budget the sample size as if it weren't there.

## Reproduce

```bash
expc load hillstrom                  # downloads the CSV, verifies its SHA-256
expc readout hillstrom               # agent readout (needs LLM_API_KEY)
python scripts/case_study.py hillstrom
```
