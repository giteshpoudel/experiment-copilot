"""Compute a case study's numbers through the same Analyzer the MCP tools use.

Usage: python scripts/case_study.py hillstrom|criteo  ->  results/case_studies/<id>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from expcopilot.analysis import Analyzer
from expcopilot.store import Store


def width(iv) -> float:
    return iv.high - iv.low


def hillstrom(an: Analyzer) -> dict:
    out: dict = {"describe": an.describe_experiment("hillstrom").summary,
                 "srm": an.check_srm("hillstrom").summary, "cuped": {}, "segments": {},
                 "uplift": {}}
    for metric in ("spend", "visit", "conversion"):
        for cov in ("history", "recency"):
            plain = an.analyze_metric("hillstrom", metric)
            adj = an.analyze_metric("hillstrom", metric, cuped_covariate=cov)
            out["cuped"][f"{metric}|{cov}"] = {
                arm.arm: {
                    "variance_reduction": c.variance_reduction,
                    "ci_width_ratio": width(c.abs_ci) / width(arm.abs_ci),
                    "summary": c.summary,
                }
                for arm, c in zip(plain.comparisons, adj.comparisons, strict=True)
            }
    for metric in ("spend", "visit"):
        for arm in ("mens_email", "womens_email"):
            for seg in ("history_segment", "channel", "zip_code", "newbie"):
                out["segments"][f"{metric}|{arm}|{seg}"] = an.segment_effects(
                    "hillstrom", metric, seg, arm=arm).summary
    for arm in ("mens_email", "womens_email"):
        for learner in ("t", "x"):
            r = an.uplift_model("hillstrom", "visit", learner=learner, arm=arm)
            out["uplift"][f"visit|{arm}|{learner}"] = {
                "qini": r.qini_coefficient, "summary": r.summary,
                "deciles": [d.model_dump() for d in r.deciles]}
    return out


def criteo(an: Analyzer) -> dict:
    out: dict = {"describe": an.describe_experiment("criteo").summary,
                 "srm": an.check_srm("criteo").summary,
                 "visit": [c.summary for c in an.analyze_metric("criteo", "visit").comparisons],
                 "conversion": [c.summary for c in
                                an.analyze_metric("criteo", "conversion").comparisons],
                 "uplift": {}}
    for outcome in ("visit", "conversion"):
        for learner in ("t", "x"):
            r = an.uplift_model("criteo", outcome, learner=learner)
            out["uplift"][f"{outcome}|{learner}"] = {
                "qini": r.qini_coefficient, "n_train": r.n_train, "n_test": r.n_test,
                "summary": r.summary, "deciles": [d.model_dump() for d in r.deciles]}
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    an = Analyzer(Store("data", read_only=True))
    data = {"hillstrom": hillstrom, "criteo": criteo}[name](an)
    path = Path("results/case_studies") / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"wrote {path}")
