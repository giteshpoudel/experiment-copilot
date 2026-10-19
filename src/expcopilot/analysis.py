"""Analysis service: loads experiment data from the store, picks methods, calls `stats/`.

This is everything the MCP tools do, minus the MCP wrapping, so tests and the Phase 5 oracle
can run the same code path as the agent. It never opens the truth store.

Result ids are deterministic (`r-` + 8 hex chars of a hash over tool, inputs and the
experiment's data version), so asking the same question twice gives the same id. Comparisons
inside a result get `<result_id>.<arm>`. Ids are remembered per `Analyzer` (one per server
process) so `adjust_pvalues` can find the p-values they refer to.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow.parquet as pq

from expcopilot.models import Experiment, MetricDef, MetricType
from expcopilot.results import (
    MAX_LIST,
    AdjustedEntry,
    AdjustPValuesResult,
    AnalyzeMetricResult,
    ArmInfo,
    BayesianResult,
    CalculateResult,
    Comparison,
    DailyEffectsResult,
    DayEffect,
    DescribeExperimentResult,
    ExperimentRow,
    GuardrailCheck,
    Interval,
    ListExperimentsResult,
    Look,
    MetricInfo,
    PowerResult,
    SegmentEffectsResult,
    SegmentLevel,
    SequentialResult,
    SRMArm,
    SRMCheckResult,
    fmt_num,
    fmt_p,
    fmt_pct,
    fmt_prob,
    fmt_signed,
    sig,
)
from expcopilot.stats import bayesian, hte, power, sequential, time_effects
from expcopilot.stats.cuped import cuped
from expcopilot.stats.frequentist import (
    EffectEstimate,
    bootstrap,
    two_proportion_ztest,
    welch_ttest,
)
from expcopilot.stats.multiple import adjust_pvalues as _adjust
from expcopilot.stats.ratio import delta_method, ratio_and_variance
from expcopilot.stats.srm import srm_test
from expcopilot.store import ExperimentNotFoundError, Store

Method = Literal["auto", "z_test", "welch", "delta", "bootstrap"]
SequentialMethod = Literal["obrien_fleming", "msprt"]
Operation = Literal["add", "subtract", "multiply", "divide", "pct_change"]

DEFAULT_METHOD = {
    MetricType.BINARY: "z_test",
    MetricType.CONTINUOUS: "welch",
    MetricType.RATIO: "delta",
}
METHOD_NAMES = {
    "z_test": "two_proportion_z",
    "welch": "welch_t",
    "delta": "delta_method",
    "bootstrap": "bootstrap_percentile",
}
BOOTSTRAP_RESAMPLES = 2000
MAX_DAYS = 60


class AnalysisError(ValueError):
    """A problem with the request (unknown id, metric, method...). Message is user-facing."""


def _choices(items: list[str], cap: int = 20) -> str:
    shown = ", ".join(items[:cap])
    return shown + (f", ... ({len(items)} total)" if len(items) > cap else "")


def _iv(lo: float, hi: float) -> Interval:
    return Interval(low=sig(lo), high=sig(hi))


@dataclass
class Loaded:
    experiment: Experiment
    version: str
    arm: np.ndarray
    cols: dict[str, np.ndarray]
    daily: dict[str, np.ndarray] | None


@dataclass(frozen=True)
class _Registered:
    label: str
    p_value: float
    alpha: float


class Analyzer:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._cache: dict[str, Loaded] = {}
        self._comparisons: dict[str, _Registered] = {}
        self._groups: dict[str, list[str]] = {}

    # --- plumbing ---

    def load(self, experiment_id: str) -> Loaded:
        try:
            version = self.store.data_version(experiment_id)
        except ExperimentNotFoundError:
            ids = [r.experiment_id for r in self.store.list()]
            raise AnalysisError(
                f"no experiment {experiment_id!r}; known ids: {_choices(ids)}"
            ) from None
        hit = self._cache.get(experiment_id)
        if hit is not None and hit.version == version:
            return hit
        exp = self.store.get(experiment_id)
        table = pq.read_table(self.store.units_path(experiment_id))
        cols = {name: table.column(name).to_numpy() for name in table.column_names}
        daily = None
        dpath = self.store.daily_path(experiment_id)
        if dpath is not None:
            dt = pq.read_table(dpath)
            daily = {name: dt.column(name).to_numpy() for name in dt.column_names}
        loaded = Loaded(exp, version, cols["arm"].astype(str), cols, daily)
        self._cache[experiment_id] = loaded
        return loaded

    @staticmethod
    def _rid(tool: str, inputs: dict[str, object], version: str = "") -> str:
        blob = json.dumps([tool, inputs, version], sort_keys=True, default=str)
        return "r-" + hashlib.sha1(blob.encode()).hexdigest()[:8]

    @staticmethod
    def _metric(exp: Experiment, name: str) -> MetricDef:
        try:
            return exp.metric(name)
        except KeyError:
            raise AnalysisError(
                f"experiment {exp.experiment_id!r} has no metric {name!r}; "
                f"metrics: {_choices([m.name for m in exp.metrics])}"
            ) from None

    @staticmethod
    def _treatment_arms(exp: Experiment, arms: list[str] | None) -> list[str]:
        treatments = [a.name for a in exp.arms if not a.is_control]
        if not arms:
            return treatments
        bad = [a for a in arms if a not in treatments]
        if bad:
            raise AnalysisError(
                f"unknown or control arm(s) {bad}; treatment arms: {_choices(treatments)}"
            )
        return list(dict.fromkeys(arms))

    def _one_arm(self, exp: Experiment, arm: str | None) -> str:
        return self._treatment_arms(exp, [arm] if arm else None)[0]

    @staticmethod
    def _role(exp: Experiment, metric: str) -> str:
        p = exp.plan
        if metric == p.primary_metric:
            return "primary"
        if metric in p.secondary_metrics:
            return "secondary"
        if any(g.metric == metric for g in p.guardrails):
            return "guardrail"
        return "unplanned"

    def _estimate(
        self,
        loaded: Loaded,
        m: MetricDef,
        c: np.ndarray,
        t: np.ndarray,
        method: str = "auto",
        covariate: str | None = None,
        alpha: float = 0.05,
        cols: dict[str, np.ndarray] | None = None,
    ) -> tuple[EffectEstimate, str]:
        """Estimate treatment vs control for the units selected by boolean masks c and t."""
        cols = cols if cols is not None else loaded.cols
        chosen = DEFAULT_METHOD[m.type] if method == "auto" else method
        if m.type is MetricType.RATIO:
            if covariate:
                raise AnalysisError("CUPED is not supported for ratio metrics; omit the covariate")
            if chosen != "delta":
                raise AnalysisError(f"ratio metric {m.name!r} supports method 'delta' (or 'auto')")
            num, den = cols[m.numerator].astype(float), cols[m.denominator].astype(float)
            return delta_method(num[c], den[c], num[t], den[t], alpha), "delta_method"
        y = cols[m.column].astype(float)
        if covariate:
            if covariate not in loaded.experiment.covariates:
                raise AnalysisError(
                    f"unknown covariate {covariate!r}; covariates: "
                    f"{_choices(loaded.experiment.covariates)}"
                )
            if chosen not in (DEFAULT_METHOD[m.type], "welch"):
                raise AnalysisError("CUPED runs with method 'auto' or 'welch' only")
            x = cols[covariate].astype(float)
            return cuped(y[c], x[c], y[t], x[t], alpha).estimate, "cuped_welch"
        if chosen == "z_test":
            if m.type is not MetricType.BINARY:
                raise AnalysisError("method 'z_test' is for binary metrics; use 'welch'")
            return two_proportion_ztest(
                int(y[c].sum()), int(c.sum()), int(y[t].sum()), int(t.sum()), alpha
            ), "two_proportion_z"
        if chosen == "welch":
            return welch_ttest(y[c], y[t], alpha), "welch_t"
        if chosen == "bootstrap":
            est = bootstrap(y[c], y[t], alpha, n_resamples=BOOTSTRAP_RESAMPLES, seed=0)
            return est, "bootstrap_percentile"
        raise AnalysisError(f"method {method!r} does not apply to {m.type} metric {m.name!r}")

    @staticmethod
    def _direction(est: EffectEstimate, higher_is_better: bool) -> str:
        lo, hi = est.abs_ci
        if lo > 0:
            return "better" if higher_is_better else "worse"
        if hi < 0:
            return "worse" if higher_is_better else "better"
        return "inconclusive"

    @staticmethod
    def _ci_label(alpha: float) -> str:
        return f"{(1 - alpha) * 100:g}% CI"

    def _comparison_summary(
        self, arm: str, control: str, est: EffectEstimate, direction: str
    ) -> str:
        return (
            f"{arm} vs {control}: {fmt_pct(est.rel_effect)} "
            f"({self._ci_label(est.alpha)} {fmt_pct(est.rel_ci[0])} to {fmt_pct(est.rel_ci[1])}), "
            f"{fmt_p(est.p_value)}; absolute {fmt_signed(est.abs_effect)} "
            f"({fmt_num(est.mean_control)} -> {fmt_num(est.mean_treatment)}); {direction}"
        )

    # --- tools ---

    def list_experiments(self) -> ListExperimentsResult:
        rows = self.store.list()
        shown = rows[:MAX_LIST]
        return ListExperimentsResult(
            result_id=self._rid("list_experiments", {}, str(len(rows))),
            method="registry_listing",
            inputs={},
            experiments=[
                ExperimentRow(
                    experiment_id=r.experiment_id, name=r.name, status=r.status, n_units=r.n_units
                )
                for r in shown
            ],
            total=len(rows),
            truncated=len(rows) > len(shown),
            summary=f"{len(rows)} experiments"
            + (f", first {len(shown)} shown" if len(rows) > len(shown) else ""),
        )

    def describe_experiment(self, experiment_id: str) -> DescribeExperimentResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        counts = self.store.arm_counts(experiment_id)
        p = exp.plan
        guard = (
            ", ".join(
                f"{g.metric} (margin {fmt_pct(g.margin, signed=False)})" for g in p.guardrails
            )
            or "none"
        )
        summary = (
            f"{exp.experiment_id} '{exp.name}': {len(exp.arms)} arms, "
            f"{sum(counts.values()):,} units. Primary {p.primary_metric} "
            f"(MDE {fmt_pct(p.mde, signed=False)} relative, alpha {p.alpha:g}); "
            f"guardrails: {guard}; planned n {p.planned_n_total:,}."
        )
        return DescribeExperimentResult(
            result_id=self._rid(
                "describe_experiment", {"experiment_id": experiment_id}, loaded.version
            ),
            method="registry_lookup",
            inputs={"experiment_id": experiment_id},
            experiment_id=exp.experiment_id,
            name=exp.name,
            description=exp.description,
            status=exp.status.value,
            unit=exp.unit,
            start_date=exp.start_date.isoformat(),
            end_date=exp.end_date.isoformat(),
            has_daily=exp.has_daily,
            arms=[
                ArmInfo(
                    name=a.name,
                    is_control=a.is_control,
                    expected_share=a.expected_share,
                    observed_n=counts[a.name],
                )
                for a in exp.arms
            ],
            n_total=sum(counts.values()),
            metrics=[
                MetricInfo(
                    name=m.name,
                    type=m.type.value,
                    higher_is_better=m.higher_is_better,
                    columns=m.columns,
                    default_method=METHOD_NAMES[DEFAULT_METHOD[m.type]],
                    description=m.description,
                )
                for m in exp.metrics
            ],
            covariates=exp.covariates,
            segments=exp.segments,
            plan=p,
            decision_rule_text=p.decision_rule.describe(),
            summary=summary,
        )

    def check_srm(self, experiment_id: str) -> SRMCheckResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        counts = self.store.arm_counts(experiment_id)
        alpha = exp.plan.decision_rule.srm_alpha
        res = srm_test(
            [counts[a.name] for a in exp.arms], [a.expected_share for a in exp.arms], alpha
        )
        arms = [
            SRMArm(
                arm=a.name,
                expected_share=a.expected_share,
                observed_share=sig(res.observed_shares[i]),
                observed=res.observed[i],
                expected=sig(res.expected[i]),
            )
            for i, a in enumerate(exp.arms)
        ]
        split = " / ".join(f"{a.arm} {fmt_pct(a.observed_share, 4, signed=False)}" for a in arms)
        verdict = "passed" if res.passed else f"FAILED (p < {alpha:g}): results are not trustworthy"
        return SRMCheckResult(
            result_id=self._rid("check_srm", {"experiment_id": experiment_id}, loaded.version),
            method=res.method,
            inputs={"experiment_id": experiment_id},
            experiment_id=experiment_id,
            arms=arms,
            chi2=sig(res.chi2),
            df=res.df,
            p_value=sig(res.p_value),
            alpha=alpha,
            passed=res.passed,
            summary=f"Observed split {split}; chi2 = {fmt_num(res.chi2)}, "
            f"{fmt_p(res.p_value)}; SRM check {verdict}.",
        )

    def power_analysis(
        self,
        metric_type: Literal["binary", "continuous", "ratio"],
        baseline: float,
        mde: float | None = None,
        alpha: float = 0.05,
        power_: float = 0.8,
        sd: float | None = None,
        n_per_arm: int | None = None,
        ratio: float = 1.0,
    ) -> PowerResult:
        inputs = {
            "metric_type": metric_type,
            "baseline": baseline,
            "mde": mde,
            "alpha": alpha,
            "power": power_,
            "sd": sd,
            "n_per_arm": n_per_arm,
            "ratio": ratio,
        }
        if (mde is None) == (n_per_arm is None):
            raise AnalysisError("give exactly one of mde (to get n) or n_per_arm (to get the MDE)")
        if baseline <= 0:
            raise AnalysisError("baseline must be positive")
        kw = {"alpha": alpha, "power": power_, "ratio": ratio}
        try:
            if metric_type == "binary":
                if n_per_arm is not None:
                    mde = power.mde_binary(baseline, n_per_arm, **kw)
                    n = n_per_arm
                else:
                    n = power.n_per_arm_binary(baseline, mde, **kw)
                mde_abs = baseline * mde
            else:
                if sd is None or sd <= 0:
                    raise AnalysisError(
                        f"{metric_type} power needs sd > 0 (sd of the metric "
                        "per unit; for ratios, of the linearized metric)"
                    )
                if n_per_arm is not None:
                    mde_abs = power.mde_continuous(sd, n_per_arm, **kw)
                    mde, n = mde_abs / baseline, n_per_arm
                else:
                    mde_abs = baseline * mde
                    n = power.n_per_arm_continuous(sd, mde_abs, **kw)
        except ValueError as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(str(e)) from None
        n_t = math.ceil(n * ratio)
        what = (
            f"needs {n:,} control + {n_t:,} treatment units to detect"
            if n_per_arm is None
            else f"with {n:,} control units can detect"
        )
        return PowerResult(
            result_id=self._rid("power_analysis", inputs),
            method=f"normal_approximation_{'binary' if metric_type == 'binary' else 'continuous'}",
            inputs=inputs,
            metric_type=metric_type,
            baseline=baseline,
            alpha=alpha,
            power=power_,
            ratio=ratio,
            n_per_arm_control=n,
            n_per_arm_treatment=n_t,
            mde_relative=sig(mde),
            mde_absolute=sig(mde_abs),
            summary=f"{metric_type} metric, baseline {fmt_num(baseline)}: {what} a "
            f"{fmt_pct(mde, signed=False)} relative ({fmt_num(mde_abs)} absolute) change "
            f"at alpha {alpha:g}, power {power_:g}.",
        )

    def analyze_metric(
        self,
        experiment_id: str,
        metric: str,
        method: Method = "auto",
        cuped_covariate: str | None = None,
        arms: list[str] | None = None,
    ) -> AnalyzeMetricResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        m = self._metric(exp, metric)
        treatments = self._treatment_arms(exp, arms)
        alpha = exp.plan.alpha
        inputs = {
            "experiment_id": experiment_id,
            "metric": metric,
            "method": method,
            "cuped_covariate": cuped_covariate,
            "arms": treatments,
        }
        rid = self._rid("analyze_metric", inputs, loaded.version)
        control = exp.control.name
        c = loaded.arm == control
        role = self._role(exp, metric)
        guard = next((g for g in exp.plan.guardrails if g.metric == metric), None)
        comparisons, method_used = [], ""
        for arm in treatments:
            est, method_used = self._estimate(
                loaded, m, c, loaded.arm == arm, method, cuped_covariate, alpha
            )
            direction = self._direction(est, m.higher_is_better)
            guardrail = None
            summary = self._comparison_summary(arm, control, est, direction)
            if guard is not None:
                worst = est.rel_ci[0] if m.higher_is_better else est.rel_ci[1]
                passed = worst > -guard.margin if m.higher_is_better else worst < guard.margin
                guardrail = GuardrailCheck(
                    margin=guard.margin, worst_case_relative=sig(worst), passed=bool(passed)
                )
                summary += (
                    f"; guardrail {'passed' if passed else 'FAILED'} "
                    f"(worst case {fmt_pct(worst)} vs margin "
                    f"{fmt_pct(guard.margin, signed=False)})"
                )
            if est.variance_reduction is not None:
                summary += (
                    f"; CUPED variance reduction {fmt_pct(est.variance_reduction, signed=False)}"
                )
            cid = f"{rid}.{arm}"
            self._comparisons[cid] = _Registered(
                f"{experiment_id} {metric}: {arm} vs {control}", est.p_value, alpha
            )
            comparisons.append(
                Comparison(
                    result_id=cid,
                    arm=arm,
                    control=control,
                    n_control=est.n_control,
                    n_treatment=est.n_treatment,
                    mean_control=sig(est.mean_control),
                    mean_treatment=sig(est.mean_treatment),
                    abs_effect=sig(est.abs_effect),
                    abs_ci=_iv(*est.abs_ci),
                    rel_effect=sig(est.rel_effect),
                    rel_ci=_iv(*est.rel_ci),
                    p_value=sig(est.p_value),
                    significant=bool(est.p_value < alpha),
                    direction=direction,
                    variance_reduction=sig(est.variance_reduction),
                    guardrail=guardrail,
                    summary=summary,
                )
            )
        self._groups[rid] = [c_.result_id for c_ in comparisons]
        notes = []
        if role == "unplanned":
            notes.append("This metric is not in the pre-registered plan: treat it as exploratory.")
        if len(treatments) > 1:
            notes.append(
                "Several treatment arms: adjust p-values with adjust_pvalues "
                f"(the plan specifies {exp.plan.multiple_testing})."
            )
        if m.type is MetricType.RATIO:
            notes.append("Ratio metric: delta method on per-unit numerator and denominator sums.")
        return AnalyzeMetricResult(
            result_id=rid,
            method=method_used,
            inputs=inputs,
            experiment_id=experiment_id,
            metric=metric,
            metric_type=m.type.value,
            role=role,
            alpha=alpha,
            cuped_covariate=cuped_covariate,
            comparisons=comparisons,
            notes=notes,
        )

    def adjust_pvalues(
        self, result_ids: list[str], method: Literal["holm", "bh"] = "holm"
    ) -> AdjustPValuesResult:
        if not result_ids:
            raise AnalysisError("give at least one result id from analyze_metric")
        if len(result_ids) > MAX_LIST:
            raise AnalysisError(f"at most {MAX_LIST} result ids")
        ids: list[str] = []
        for rid in result_ids:
            if rid in self._groups:
                ids.extend(self._groups[rid])
            elif rid in self._comparisons:
                ids.append(rid)
            else:
                known = list(self._groups) + list(self._comparisons)
                raise AnalysisError(
                    f"unknown result id {rid!r}. Use ids returned by analyze_metric in this "
                    f"session: {_choices(known) or 'none yet'}"
                )
        ids = list(dict.fromkeys(ids))
        regs = [self._comparisons[i] for i in ids]
        alpha = min(r.alpha for r in regs)
        adj = _adjust([r.p_value for r in regs], method)
        entries = [
            AdjustedEntry(
                result_id=i,
                label=r.label,
                p_value=sig(r.p_value),
                p_adjusted=sig(float(a)),
                significant_adjusted=bool(a < alpha),
            )
            for i, r, a in zip(ids, regs, adj, strict=True)
        ]
        sig_n = sum(e.significant_adjusted for e in entries)
        lines = "; ".join(f"{e.label}: {fmt_p(e.p_adjusted)} adjusted" for e in entries[:10])
        return AdjustPValuesResult(
            result_id=self._rid("adjust_pvalues", {"ids": ids, "method": method}),
            method=method,
            inputs={"result_ids": result_ids, "method": method},
            alpha=alpha,
            entries=entries,
            summary=f"{method.upper() if method == 'bh' else 'Holm'} over {len(entries)} "
            f"comparisons: {sig_n} significant at alpha {alpha:g}. {lines}",
        )

    def sequential_test(
        self,
        experiment_id: str,
        metric: str,
        looks: list[int] | None = None,
        method: SequentialMethod = "obrien_fleming",
        arm: str | None = None,
    ) -> SequentialResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        m = self._metric(exp, metric)
        arm = self._one_arm(exp, arm)
        days = (exp.end_date - exp.start_date).days + 1
        looks = sorted(set(looks or exp.plan.planned_looks))
        if not looks:
            raise AnalysisError(
                f"the plan has no interim looks; pass looks as a list of days, e.g. [7, {days}]"
            )
        if looks[0] < 1 or looks[-1] > days:
            raise AnalysisError(f"looks must be days between 1 and {days}")
        inputs = {
            "experiment_id": experiment_id,
            "metric": metric,
            "looks": looks,
            "method": method,
            "arm": arm,
        }
        control = exp.control.name
        alpha = exp.plan.alpha
        assumptions: list[str] = []
        per_look: list[tuple[int, float, EffectEstimate]] = []

        if exp.has_daily:
            if loaded.daily is None or m.type is MetricType.RATIO or m.column not in loaded.daily:
                raise AnalysisError(
                    f"daily data has no {metric!r}; daily metrics: "
                    f"{_choices(self._daily_metrics(loaded))}"
                )
            matrix = self._daily_matrix(loaded, m.column, days)
            cum = np.cumsum(matrix, axis=1)
            assumptions.append(
                "All units start on day 1; each look uses outcomes accumulated through that day."
            )
            for d in looks:
                y = cum[:, d - 1]
                if m.type is MetricType.BINARY:
                    y = (y > 0).astype(float)
                cols = {m.column: y}
                est, _ = self._estimate(
                    loaded, m, loaded.arm == control, loaded.arm == arm, cols=cols, alpha=alpha
                )
                per_look.append((d, d / days, est))
        else:
            enrolled_by = loaded.cols["assign_day"]
            final_n = int((enrolled_by < looks[-1]).sum())
            denom = max(exp.plan.planned_n_total, final_n)
            assumptions.append(
                "A look on day d uses units assigned before the end of day d, "
                "with their final outcomes treated as already observed."
            )
            assumptions.append(
                f"Information fraction = units enrolled / {denom:,} "
                "(the larger of planned and final enrolment)."
            )
            last_frac = 0.0
            for d in looks:
                mask = enrolled_by < d
                frac = float(mask.sum()) / denom
                if frac <= last_frac:
                    continue
                last_frac = frac
                est, _ = self._estimate(
                    loaded,
                    m,
                    mask & (loaded.arm == control),
                    mask & (loaded.arm == arm),
                    alpha=alpha,
                )
                per_look.append((d, frac, est))

        fractions = [f for _, f, _ in per_look]
        ests = [e for _, _, e in per_look]
        if method == "obrien_fleming":
            z = [e.abs_effect / e.abs_se if e.abs_se > 0 else 0.0 for e in ests]
            res = sequential.group_sequential_test(z, fractions, alpha)
            stats_, bounds = list(res.statistics), list(res.boundaries)
            method_name = res.method
        else:
            tau = exp.plan.mde * abs(ests[-1].mean_control)
            res = sequential.msprt(
                [e.abs_effect for e in ests],
                [e.abs_se**2 for e in ests],
                tau=tau,
                alpha=alpha,
                info_fractions=fractions,
            )
            stats_, bounds = list(res.p_values or []), list(res.boundaries)
            method_name = res.method
            assumptions.append(f"mSPRT mixing sd tau = plan MDE x control mean = {fmt_num(tau)}.")
        look_rows = []
        for i, (d, frac, e) in enumerate(per_look):
            crossed = res.stopped_at is not None and i >= res.stopped_at
            look_rows.append(
                Look(
                    day=d,
                    information_fraction=sig(frac),
                    n_control=e.n_control,
                    n_treatment=e.n_treatment,
                    statistic=sig(stats_[i]),
                    boundary=sig(bounds[i]),
                    crossed=bool(crossed),
                )
            )
        stopped_day = per_look[res.stopped_at][0] if res.stopped_at is not None else None
        if stopped_day is None:
            verdict = (
                f"never crossed the boundary in {len(per_look)} looks; no early stop "
                f"(last look day {per_look[-1][0]})"
            )
        else:
            verdict = f"crossed the boundary on day {stopped_day}: stop and reject H0"
        return SequentialResult(
            result_id=self._rid("sequential_test", inputs, loaded.version),
            method=method_name,
            inputs=inputs,
            experiment_id=experiment_id,
            metric=metric,
            arm=arm,
            alpha=alpha,
            looks=look_rows,
            stopped=stopped_day is not None,
            stopped_at_day=stopped_day,
            assumptions=assumptions,
            summary=f"{method_name} on {metric} ({arm} vs {control}): {verdict}.",
        )

    @staticmethod
    def _daily_metrics(loaded: Loaded) -> list[str]:
        cols = loaded.daily or {}
        return [m.name for m in loaded.experiment.metrics if m.column and m.column in cols]

    def _daily_matrix(self, loaded: Loaded, column: str, days: int) -> np.ndarray:
        """(n_units, days) matrix of a daily column, rows aligned with the unit table."""
        d = loaded.daily
        assert d is not None
        unit_ids = loaded.cols["unit_id"]
        order = np.argsort(unit_ids)
        pos = np.searchsorted(unit_ids[order], d["unit_id"])
        rows = order[pos]
        out = np.zeros((unit_ids.size, days))
        np.add.at(out, (rows, d["day"].astype(int)), d[column].astype(float))
        return out

    def bayesian_ab(
        self,
        experiment_id: str,
        metric: str,
        arm: str | None = None,
        prior: list[float] | None = None,
    ) -> BayesianResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        m = self._metric(exp, metric)
        arm = self._one_arm(exp, arm)
        inputs = {"experiment_id": experiment_id, "metric": metric, "arm": arm, "prior": prior}
        c, t = loaded.arm == exp.control.name, loaded.arm == arm
        credible = 1 - exp.plan.alpha
        if m.type is MetricType.BINARY:
            if prior is not None and (len(prior) != 2 or min(prior) <= 0):
                raise AnalysisError("prior must be [a, b] with a, b > 0 (Beta prior)")
            y = loaded.cols[m.column].astype(float)
            res = bayesian.beta_binomial(
                int(y[c].sum()),
                int(c.sum()),
                int(y[t].sum()),
                int(t.sum()),
                prior=tuple(prior) if prior else (1.0, 1.0),
                credible=credible,
                higher_is_better=m.higher_is_better,
            )
        else:
            if prior is not None:
                raise AnalysisError(
                    "priors are only supported for binary metrics (Beta prior); "
                    "continuous and ratio metrics use a flat prior"
                )
            if m.type is MetricType.RATIO:
                num, den = (loaded.cols[k].astype(float) for k in (m.numerator, m.denominator))
                mc, vc = ratio_and_variance(num[c], den[c])
                mt, vt = ratio_and_variance(num[t], den[t])
            else:
                y = loaded.cols[m.column].astype(float)
                mc, vc = y[c].mean(), y[c].var(ddof=1) / c.sum()
                mt, vt = y[t].mean(), y[t].var(ddof=1) / t.sum()
            res = bayesian.normal_posterior(
                mc, math.sqrt(vc), mt, math.sqrt(vt), credible, m.higher_is_better
            )
        lvl = f"{credible * 100:g}%"
        return BayesianResult(
            result_id=self._rid("bayesian_ab", inputs, loaded.version),
            method=res.method,
            inputs=inputs,
            experiment_id=experiment_id,
            metric=metric,
            arm=arm,
            prob_treatment_better=sig(res.prob_treatment_better),
            expected_loss_ship=sig(res.expected_loss_ship),
            expected_loss_keep=sig(res.expected_loss_keep),
            abs_credible_interval=_iv(*res.abs_credible_interval),
            rel_credible_interval=_iv(*res.rel_credible_interval),
            credible_level=credible,
            prior=list(res.prior) if res.prior else None,
            summary=f"P({arm} better than {exp.control.name}) = "
            f"{fmt_prob(res.prob_treatment_better)}; {lvl} credible "
            f"interval for the relative effect {fmt_pct(res.rel_credible_interval[0])} to "
            f"{fmt_pct(res.rel_credible_interval[1])}; expected loss if shipped "
            f"{fmt_num(res.expected_loss_ship)} ({metric} units).",
        )

    def segment_effects(
        self,
        experiment_id: str,
        metric: str,
        segment: str,
        arm: str | None = None,
        adjust: Literal["holm", "bh"] = "holm",
    ) -> SegmentEffectsResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        m = self._metric(exp, metric)
        if segment not in exp.segments:
            raise AnalysisError(f"unknown segment {segment!r}; segments: {_choices(exp.segments)}")
        arm = self._one_arm(exp, arm)
        inputs = {
            "experiment_id": experiment_id,
            "metric": metric,
            "segment": segment,
            "arm": arm,
            "adjust": adjust,
        }
        control = exp.control.name
        sel = np.flatnonzero((loaded.arm == control) | (loaded.arm == arm))
        n = loaded.arm.size

        def est(cm: np.ndarray, tm: np.ndarray) -> EffectEstimate:
            fc, ft = np.zeros(n, bool), np.zeros(n, bool)
            fc[sel[cm]], ft[sel[tm]] = True, True
            return self._estimate(loaded, m, fc, ft, alpha=exp.plan.alpha)[0]

        try:
            res = hte.segment_effects(
                segment, loaded.cols[segment][sel], loaded.arm[sel] == arm, est, adjust
            )
        except ValueError as e:
            raise AnalysisError(str(e)) from None
        levels = [
            SegmentLevel(
                level=s.level,
                n_control=s.estimate.n_control,
                n_treatment=s.estimate.n_treatment,
                rel_effect=sig(s.estimate.rel_effect),
                rel_ci=_iv(*s.estimate.rel_ci),
                abs_effect=sig(s.estimate.abs_effect),
                p_value=sig(s.estimate.p_value),
                p_adjusted=sig(s.p_adjusted),
                summary=f"{segment}={s.level}: {fmt_pct(s.estimate.rel_effect)} "
                f"({fmt_pct(s.estimate.rel_ci[0])} to {fmt_pct(s.estimate.rel_ci[1])}), "
                f"{fmt_p(s.p_adjusted)} adjusted",
            )
            for s in res.effects
        ]
        het = (
            "effects differ across levels"
            if res.heterogeneity_p < 0.05
            else "no evidence the effect differs across levels"
        )
        return SegmentEffectsResult(
            result_id=self._rid("segment_effects", inputs, loaded.version),
            method=f"per_segment_{METHOD_NAMES[DEFAULT_METHOD[m.type]]}_{adjust}_adjusted",
            inputs=inputs,
            experiment_id=experiment_id,
            metric=metric,
            segment=segment,
            arm=arm,
            adjust_method=adjust,
            levels=levels,
            skipped_levels=list(res.skipped_levels),
            heterogeneity_q=sig(res.heterogeneity_q),
            heterogeneity_p=sig(res.heterogeneity_p),
            summary=f"Exploratory, not part of the pre-registered decision. Cochran's Q "
            f"{fmt_p(res.heterogeneity_p)}: {het}. " + "; ".join(lv.summary for lv in levels),
        )

    def daily_effects(
        self, experiment_id: str, metric: str, arm: str | None = None
    ) -> DailyEffectsResult:
        loaded = self.load(experiment_id)
        exp = loaded.experiment
        m = self._metric(exp, metric)
        if not exp.has_daily or loaded.daily is None:
            raise AnalysisError(
                f"experiment {experiment_id!r} has no daily data; use "
                "analyze_metric for the overall effect"
            )
        if m.type is MetricType.RATIO or m.column not in loaded.daily:
            daily_metrics = self._daily_metrics(loaded)
            raise AnalysisError(
                f"no daily data for {metric!r}; daily metrics: {_choices(daily_metrics)}"
            )
        arm = self._one_arm(exp, arm)
        inputs = {"experiment_id": experiment_id, "metric": metric, "arm": arm}
        days = min((exp.end_date - exp.start_date).days + 1, MAX_DAYS)
        matrix = self._daily_matrix(loaded, m.column, days)
        c, t = loaded.arm == exp.control.name, loaded.arm == arm
        ests = []
        for d in range(days):
            est, _ = self._estimate(
                loaded, m, c, t, cols={m.column: matrix[:, d]}, alpha=exp.plan.alpha
            )
            ests.append(est)
        trend = time_effects.daily_effects(list(range(1, days + 1)), ests, exp.plan.alpha)
        rows = [
            DayEffect(
                day=d + 1,
                rel_effect=sig(e.rel_effect),
                rel_ci=_iv(*e.rel_ci),
                p_value=sig(e.p_value),
            )
            for d, e in enumerate(ests)
        ]
        pattern = (
            "the effect shrinks over time (novelty pattern)"
            if trend.decaying
            else "no significant downward trend"
        )
        return DailyEffectsResult(
            result_id=self._rid("daily_effects", inputs, loaded.version),
            method=trend.method,
            inputs=inputs,
            experiment_id=experiment_id,
            metric=metric,
            arm=arm,
            days=rows,
            slope_per_day=sig(trend.slope_per_day),
            slope_p_value=sig(trend.slope_p_value),
            decaying=trend.decaying,
            summary=f"{metric} by day ({arm} vs {exp.control.name}): day 1 "
            f"{fmt_pct(ests[0].rel_effect)}, day {days} {fmt_pct(ests[-1].rel_effect)}; "
            f"trend {fmt_signed(trend.slope_per_day * 100, 3)} percentage points of lift per day, "
            f"{fmt_p(trend.slope_p_value)}: {pattern}.",
        )

    def calculate(self, operation: Operation, a: float, b: float) -> CalculateResult:
        inputs = {"operation": operation, "a": a, "b": b}
        if operation == "add":
            value = a + b
        elif operation == "subtract":
            value = a - b
        elif operation == "multiply":
            value = a * b
        elif operation in ("divide", "pct_change"):
            if (b if operation == "divide" else a) == 0:
                raise AnalysisError("division by zero")
            value = a / b if operation == "divide" else (b - a) / a
        else:
            raise AnalysisError(f"unknown operation {operation!r}")
        shown = fmt_pct(value) if operation == "pct_change" else fmt_num(value, 8)
        return CalculateResult(
            result_id=self._rid("calculate", inputs),
            method=f"exact_{operation}",
            inputs=inputs,
            operation=operation,
            a=a,
            b=b,
            value=sig(value, 12),
            summary=f"{operation}(a={fmt_num(a, 8)}, b={fmt_num(b, 8)}) = {shown}",
        )
