# Case study: Criteo ad-targeting uplift

**Ad targeting lifts site visits +30.2%, and the agent ships it. It correctly treats the planned
85/15 split as valid, not as a sample-ratio mismatch. Most of the lift sits in a small group of
users: an uplift model's top decile gains +5.7 points against +1.2 overall. On this data the
simple T-learner beat the X-learner.**

Data: the [Criteo Uplift Prediction dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
v2.1 (CC BY-NC-SA 4.0). About 14 million users were randomly targeted by advertising
(treatment, 85% by design) or held out (control, 15%). Outcomes are visit and conversion, and
there are 12 anonymized pre-treatment features. This study uses a seeded random sample of
1,000,000 users (DuckDB reservoir sample, seed 42).

The plan is **retrospective**, written for this case study before the analysis ran: primary
metric visit, secondary conversion, planned shares 85/15, no guardrails. Every number below
comes from the tools: the agent's readout cites them, and the rest come from
[`scripts/case_study.py`](../scripts/case_study.py)
([`results/case_studies/criteo.json`](../results/case_studies/criteo.json)).

## The agent's readout

`expc readout criteo`, DeepSeek V4-Pro: 4 tool calls, $0.0065, 24 s
([full JSON](../results/readouts/criteo-agent.json)).

```
## Decision: SHIP
Ship the treatment arm.

## Primary result
- Treatment vs control: +30.2% (95% CI +26.8% to +33.9%), p < 0.0001;
  absolute +0.01128 (0.0373 -> 0.04858); better [r3]

## Validity checks
- SRM: passed. Observed split control 15.01% / treatment 84.99%; chi2 = 0.1387, p = 0.71 [r2].

## Exploratory findings
- Conversion: +38.4% (95% CI +24.1% to +55.9%), p < 0.0001;
  absolute +0.0008574 (0.002231 -> 0.003089); better [r4].
```

The SRM check is the trap this dataset sets: an 85/15 split looks wildly unbalanced, but it's
what the plan intended, and the check compares against the planned shares.

The no-tools baseline (same model, given only the plan and per-arm means) **spent its entire
16,000-token output budget reasoning and never wrote a readout**: no decision, $0.0321, 125 s.

## Who responds: uplift models

Uplift models were trained on 70% of the sample (350,059 users) and evaluated on the held-out
30% (149,941 users):

| Outcome | Learner | Qini | Top-decile uplift | Bottom-decile uplift | Overall |
| --- | --- | --- | --- | --- | --- |
| visit | T | 0.0517 | +5.66 points | +2.96 points | +1.21 points |
| visit | X | 0.0316 | +5.11 points | +3.87 points | +1.21 points |
| conversion | T | 0.0898 | +0.38 points | +0.02 points | +0.10 points |
| conversion | X | −0.0159 | +0.34 points | +0.46 points | +0.10 points |

What this shows:

- **The lift is concentrated.** With the T-learner on visits, deciles 3–9 show close to zero
  uplift. Most of the ad's effect comes from roughly the top fifth of users the model ranks.
- **The T-learner beat the X-learner** on both outcomes, even though the X-learner is usually
  recommended when arms are this unbalanced. With conversions at 0.3%, the X-learner's
  second-stage models fit very noisy imputed effects, and its conversion ranking is worse than
  random.
- **The ranking is not monotone.** The bottom decile also has a large real uplift (+2.96 points
  for the T-learner on visits). Users with extreme feature values get extreme predictions in
  both directions, but respond strongly either way. A positive Qini doesn't mean the model is
  right at the tails, which is why the deciles are reported next to it.

## Takeaways

1. **Decision:** ship the ad targeting. The visit lift is large and precise, and the SRM check
   correctly passes the planned 85/15 split.
2. **For targeting,** prefer the T-learner on this data and read its decile table, not just its
   Qini: the top decile is worth targeting, and the bottom decile is not the place to cut.
3. **Rare outcomes need care.** Every conversion model is fragile at a 0.3% base rate. Visit is
   the better outcome to target on.

## Reproduce

```bash
# Criteo's own link was returning 404 (October 2026); place the file by hand:
#   data/raw/criteo-research-uplift-v2.1.csv.gz
expc load criteo                     # seeded 1M-user sample; warns if the SHA-256 differs
expc readout criteo                  # agent readout (needs LLM_API_KEY)
python scripts/case_study.py criteo
```
