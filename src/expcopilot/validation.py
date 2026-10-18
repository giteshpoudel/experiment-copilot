"""Simulation validation of the statistics engine (Phase 2).

Runs thousands of simulated A/A and A/B experiments through the real simulator and checks:

- 95% CI coverage of the true effect, per method, absolute and relative
- A/A false-positive rate at alpha = 0.05, per method
- naive daily peeking inflates false positives; OBF group sequential and mSPRT hold ~5%
- CUPED shrinks the CI by sqrt(1 - r^2) as covariate correlation r rises

Pass bands are the target +/- 3 binomial standard errors for the number of reps run, so they
widen automatically for quick runs. Results go to results/validation.json for the README.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from expcopilot.models import EffectSpec, SimulationSpec
from expcopilot.simulate import DEFAULT_SEGMENTS, TWO_ARMS, _plan, baseline, generate
from expcopilot.stats.cuped import cuped
from expcopilot.stats.frequentist import (
    EffectEstimate,
    bootstrap,
    two_proportion_ztest,
    welch_ttest,
)
from expcopilot.stats.ratio import delta_method
from expcopilot.stats.sequential import group_sequential_bounds, msprt

ALPHA = 0.05
DAYS = 14
AB_LIFT = {"conversion": 0.10, "revenue": 0.15}  # sessions flat, so revenue_per_session +15%


@dataclass
class Check:
    name: str
    method: str
    metric: str
    kind: str  # coverage | false_positive_rate | inflation | cuped_width
    target: float
    band: tuple[float, float]
    observed: float
    reps: int
    observed_ci: tuple[float, float] | None = None
    note: str = ""
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.observed = float(self.observed)
        self.passed = bool(self.band[0] <= self.observed <= self.band[1])


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def _band(target: float, reps: int, k: float = 3.0) -> tuple[float, float]:
    se = math.sqrt(target * (1 - target) / reps)
    return max(0.0, target - k * se), min(1.0, target + k * se)


def _spec(seed: int, n: int, lift: dict[str, float], rho: float = 0.5) -> SimulationSpec:
    return SimulationSpec(
        experiment_id="validation",
        name="validation",
        seed=seed,
        n_total=n,
        arms=TWO_ARMS,
        days=DAYS,
        segments=DEFAULT_SEGMENTS,
        covariate_rho=rho,
        effects=[EffectSpec(metric=m, arm="treatment", relative=v) for m, v in lift.items()],
        plan=_plan(),
    )


@dataclass(frozen=True)
class Arms:
    c: np.ndarray
    t: np.ndarray


def _arrays(spec: SimulationSpec) -> tuple[dict[str, np.ndarray], np.ndarray]:
    units = generate(spec).units
    cols = {
        c: units.column(c).to_numpy()
        for c in ("converted", "revenue", "sessions", "pre_revenue", "assign_day")
    }
    return cols, units.column("arm").to_numpy() == "treatment"


Estimator = Callable[[dict[str, np.ndarray], np.ndarray], EffectEstimate]


def _z(cols: dict[str, np.ndarray], t: np.ndarray) -> EffectEstimate:
    y = cols["converted"]
    return two_proportion_ztest(int(y[~t].sum()), int((~t).sum()), int(y[t].sum()), int(t.sum()))


def _welch(cols: dict[str, np.ndarray], t: np.ndarray) -> EffectEstimate:
    return welch_ttest(cols["revenue"][~t], cols["revenue"][t])


def _delta(cols: dict[str, np.ndarray], t: np.ndarray) -> EffectEstimate:
    r, s = cols["revenue"], cols["sessions"]
    return delta_method(r[~t], s[~t], r[t], s[t])


def _cuped(cols: dict[str, np.ndarray], t: np.ndarray) -> EffectEstimate:
    y, x = cols["revenue"], cols["pre_revenue"]
    return cuped(y[~t], x[~t], y[t], x[t]).estimate


METHODS: dict[str, tuple[str, Estimator]] = {
    "two_proportion_z": ("conversion", _z),
    "welch_t": ("revenue", _welch),
    "delta_method": ("revenue_per_session", _delta),
    "cuped_welch": ("revenue", _cuped),
}


def _covers(ci: tuple[float, float], truth: float) -> bool:
    return ci[0] <= truth <= ci[1]


def run_coverage_and_fpr(reps: int, n: int, seed: int) -> list[Check]:
    hits = {m: {"abs": 0, "rel": 0, "fp": 0} for m in METHODS}
    template = _spec(0, n, AB_LIFT)
    true_rel = {"conversion": 0.10, "revenue": 0.15, "revenue_per_session": 0.15}
    true_abs = {m: baseline(template, m) * r for m, r in true_rel.items()}
    for i in range(reps):
        ab_cols, ab_t = _arrays(_spec(seed + i, n, AB_LIFT))
        aa_cols, aa_t = _arrays(_spec(seed + 10**6 + i, n, {}))
        for name, (metric, est) in METHODS.items():
            ab = est(ab_cols, ab_t)
            hits[name]["abs"] += _covers(ab.abs_ci, true_abs[metric])
            hits[name]["rel"] += _covers(ab.rel_ci, true_rel[metric])
            hits[name]["fp"] += est(aa_cols, aa_t).p_value < ALPHA
    checks = []
    for name, (metric, _) in METHODS.items():
        for scale in ("abs", "rel"):
            k = hits[name][scale]
            checks.append(
                Check(
                    f"coverage_{scale}_{name}",
                    name,
                    metric,
                    "coverage",
                    0.95,
                    _band(0.95, reps),
                    k / reps,
                    reps,
                    _wilson(k, reps),
                    f"n={n}, true lift {true_rel[metric]:+.0%}",
                )
            )
        k = hits[name]["fp"]
        checks.append(
            Check(
                f"aa_fpr_{name}",
                name,
                metric,
                "false_positive_rate",
                ALPHA,
                _band(ALPHA, reps),
                k / reps,
                reps,
                _wilson(k, reps),
                f"A/A, n={n}",
            )
        )
    return checks


def run_bootstrap(reps: int, n: int, seed: int, resamples: int = 1000) -> list[Check]:
    template = _spec(0, n, AB_LIFT)
    truth = baseline(template, "revenue") * AB_LIFT["revenue"]
    cover = fp = 0
    for i in range(reps):
        cols, t = _arrays(_spec(seed + i, n, AB_LIFT))
        y = cols["revenue"]
        cover += _covers(bootstrap(y[~t], y[t], n_resamples=resamples, seed=i).abs_ci, truth)
        cols, t = _arrays(_spec(seed + 10**6 + i, n, {}))
        y = cols["revenue"]
        fp += bootstrap(y[~t], y[t], n_resamples=resamples, seed=i).p_value < ALPHA
    note = f"n={n}, B={resamples}"
    return [
        Check(
            "coverage_abs_bootstrap",
            "bootstrap_percentile",
            "revenue",
            "coverage",
            0.95,
            _band(0.95, reps),
            cover / reps,
            reps,
            _wilson(cover, reps),
            note,
        ),
        Check(
            "aa_fpr_bootstrap",
            "bootstrap_percentile",
            "revenue",
            "false_positive_rate",
            ALPHA,
            _band(ALPHA, reps),
            fp / reps,
            reps,
            _wilson(fp, reps),
            note,
        ),
    ]


def run_peeking(reps: int, n: int, seed: int) -> list[Check]:
    """A/A experiments looked at after every day of staggered enrolment (14 looks)."""
    looks = np.arange(1, DAYS + 1)
    fractions = looks / DAYS  # planned information fractions under uniform enrolment
    bounds = group_sequential_bounds(fractions, ALPHA)
    tau = baseline(_spec(0, n, {}), "conversion") * 0.10  # the plan's absolute MDE
    naive = obf = ms = 0
    for i in range(reps):
        cols, t = _arrays(_spec(seed + 2 * 10**6 + i, n, {}))
        z, effects, variances = [], [], []
        for day in looks:
            m = cols["assign_day"] < day
            sub = {"converted": cols["converted"][m]}
            est = _z(sub, t[m])
            z.append(est.abs_effect / est.abs_se)
            effects.append(est.abs_effect)
            variances.append(est.abs_se**2)
        z_arr = np.abs(np.array(z))
        naive += bool(np.any(z_arr >= 1.959964))
        obf += bool(np.any(z_arr >= bounds))
        ms += msprt(effects, variances, tau=tau, alpha=ALPHA).rejected
    note = f"A/A, n={n}, {DAYS} daily looks"
    return [
        Check(
            "peeking_naive_inflates",
            "naive_peeking",
            "conversion",
            "inflation",
            ALPHA,
            (2 * ALPHA, 1.0),
            naive / reps,
            reps,
            _wilson(naive, reps),
            note + "; passes if FPR >= 2x alpha",
        ),
        Check(
            "peeking_fpr_obf",
            "group_sequential_obf",
            "conversion",
            "false_positive_rate",
            ALPHA,
            (0.0, _band(ALPHA, reps)[1]),
            obf / reps,
            reps,
            _wilson(obf, reps),
            note,
        ),
        Check(
            "peeking_fpr_msprt",
            "msprt",
            "conversion",
            "false_positive_rate",
            ALPHA,
            (0.0, _band(ALPHA, reps)[1]),
            ms / reps,
            reps,
            _wilson(ms, reps),
            note + f", tau={tau:.4f}",
        ),
    ]


def run_cuped_width(
    reps: int, n: int, seed: int, rhos: tuple[float, ...] = (0.0, 0.5, 0.9)
) -> list[Check]:
    checks = []
    for j, rho in enumerate(rhos):
        ratios, corrs = [], []
        for i in range(reps):
            cols, t = _arrays(_spec(seed + 3 * 10**6 + j * reps + i, n, AB_LIFT, rho))
            y, x = cols["revenue"], cols["pre_revenue"]
            res = cuped(y[~t], x[~t], y[t], x[t])
            ratios.append(res.estimate.abs_se / welch_ttest(y[~t], y[t]).abs_se)
            corrs.append(res.covariate_correlation)
        r = float(np.mean(corrs))
        expected = math.sqrt(1 - r**2)
        observed = float(np.mean(ratios))
        checks.append(
            Check(
                f"cuped_width_rho_{rho}",
                "cuped_welch",
                "revenue",
                "cuped_width",
                expected,
                (expected * 0.95, expected * 1.05),
                observed,
                reps,
                None,
                f"CI width ratio CUPED/Welch; mean realized r={r:.3f}; passes within 5%",
            )
        )
    widths = [c.observed for c in checks]
    mono = all(a > b for a, b in itertools.pairwise(widths))
    checks.append(
        Check(
            "cuped_width_monotone",
            "cuped_welch",
            "revenue",
            "cuped_width",
            1.0,
            (1.0, 1.0),
            float(mono),
            reps,
            None,
            "width ratio strictly falls as rho rises",
        )
    )
    return checks


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_all(
    reps: int = 2000,
    seed: int = 20261018,
    n: int = 10_000,
    bootstrap_reps: int | None = None,
    cuped_reps: int | None = None,
) -> dict:
    t0 = time.perf_counter()
    checks = [
        *run_coverage_and_fpr(reps, n, seed),
        *run_bootstrap(bootstrap_reps or max(reps * 3 // 20, 50), 4_000, seed),
        *run_peeking(reps, n, seed),
        *run_cuped_width(cuped_reps or max(reps // 10, 20), n, seed),
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": seed,
        "reps": reps,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "all_passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
