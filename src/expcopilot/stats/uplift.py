"""Uplift modelling: who responds to the treatment, not just whether it works on average.

- T-learner: separate outcome models for treated and control; uplift = mu_1(x) - mu_0(x).
- X-learner (Kunzel et al. 2019): impute individual effects with the other arm's model, fit
  models to those, and blend them with the propensity e(x) (constant here: randomized).
- Qini curve and coefficient (Radcliffe 2007), matching scikit-uplift's qini_auc_score:
  the area between the model's Qini curve and the random diagonal, divided by the same area
  for the perfect ranking. 0 = no better than random, 1 = perfect, negative = worse.

Models are scikit-learn HistGradientBoosting, which handles mixed-scale numeric features.
Evaluation is always on a held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


@dataclass(frozen=True)
class DecileRow:
    decile: int  # 1 = highest predicted uplift
    n_treatment: int
    n_control: int
    rate_treatment: float
    rate_control: float
    uplift: float  # observed rate difference in this decile


@dataclass(frozen=True)
class UpliftResult:
    learner: str
    n_train: int
    n_test: int
    qini_coefficient: float
    mean_predicted_uplift: float
    observed_ate_test: float
    deciles: tuple[DecileRow, ...]


def _model(binary: bool, seed: int):
    kw = {
        "max_iter": 200,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "random_state": seed,
        "early_stopping": False,
    }
    return HistGradientBoostingClassifier(**kw) if binary else HistGradientBoostingRegressor(**kw)


def _predict(model, x: np.ndarray, binary: bool) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if binary else model.predict(x)


def t_learner(x: np.ndarray, y: np.ndarray, t: np.ndarray, binary: bool, seed: int = 0):
    m1 = _model(binary, seed).fit(x[t == 1], y[t == 1])
    m0 = _model(binary, seed).fit(x[t == 0], y[t == 0])
    return lambda xn: _predict(m1, xn, binary) - _predict(m0, xn, binary)


def x_learner(x: np.ndarray, y: np.ndarray, t: np.ndarray, binary: bool, seed: int = 0):
    m1 = _model(binary, seed).fit(x[t == 1], y[t == 1])
    m0 = _model(binary, seed).fit(x[t == 0], y[t == 0])
    d1 = y[t == 1] - _predict(m0, x[t == 1], binary)  # treated: observed - imputed control
    d0 = _predict(m1, x[t == 0], binary) - y[t == 0]  # control: imputed treated - observed
    tau1 = _model(False, seed).fit(x[t == 1], d1)
    tau0 = _model(False, seed).fit(x[t == 0], d0)
    e = float(t.mean())  # randomized: the propensity is the treated share
    return lambda xn: e * tau0.predict(xn) + (1 - e) * tau1.predict(xn)


def qini_curve(y: np.ndarray, score: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental responses when targeting units in descending score order."""
    order = np.argsort(score, kind="mergesort")[::-1]
    y, t, score = y[order], t[order], score[order]
    y_t = np.cumsum(np.where(t == 1, y, 0))
    y_c = np.cumsum(np.where(t == 0, y, 0))
    n_t = np.cumsum(t)
    n_all = np.arange(1, y.size + 1)
    n_c = n_all - n_t
    # Evaluate only at the last index of each run of tied scores.
    ends = np.r_[np.flatnonzero(np.diff(score)), y.size - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c[ends] > 0, n_t[ends] / n_c[ends], 0.0)
    values = y_t[ends] - y_c[ends] * ratio
    return np.r_[0, n_all[ends]], np.r_[0.0, values]


def qini_coefficient(y: np.ndarray, score: np.ndarray, t: np.ndarray) -> float:
    y, score, t = (np.asarray(a, float) for a in (y, score, t))
    x_m, q_m = qini_curve(y, score, t)
    perfect = y * t - y * (1 - t)  # treated responders first, control responders last
    x_p, q_p = qini_curve(y, perfect, t)
    base = np.trapezoid([0.0, q_p[-1]], [0.0, x_p[-1]])
    return float((np.trapezoid(q_m, x_m) - base) / (np.trapezoid(q_p, x_p) - base))


def by_decile(y: np.ndarray, score: np.ndarray, t: np.ndarray, bins: int = 10) -> list[DecileRow]:
    order = np.argsort(score, kind="mergesort")[::-1]
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), start=1):
        yt, yc = y[idx][t[idx] == 1], y[idx][t[idx] == 0]
        rt = float(yt.mean()) if yt.size else float("nan")
        rc = float(yc.mean()) if yc.size else float("nan")
        rows.append(DecileRow(i, int(yt.size), int(yc.size), rt, rc, rt - rc))
    return rows


def fit_and_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    learner: str = "t",
    test_share: float = 0.3,
    seed: int = 0,
) -> UpliftResult:
    """Fit on a random (1 - test_share) of units, evaluate Qini and deciles on the rest."""
    x, y, t = np.asarray(x, float), np.asarray(y, float), np.asarray(t, int)
    if x.ndim != 2 or len({x.shape[0], y.size, t.size}) != 1:
        raise ValueError("x must be 2-d with one row per unit; y and t one value per unit")
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("t must contain both treated (1) and control (0) units")
    binary = set(np.unique(y)) <= {0.0, 1.0}
    rng = np.random.default_rng(seed)
    test = rng.random(t.size) < test_share
    fit = {"t": t_learner, "x": x_learner}[learner]
    predict = fit(x[~test], y[~test], t[~test], binary, seed)
    score = predict(x[test])
    yt, tt = y[test], t[test]
    return UpliftResult(
        learner=f"{learner}_learner_hist_gradient_boosting",
        n_train=int((~test).sum()),
        n_test=int(test.sum()),
        qini_coefficient=qini_coefficient(yt, score, tt) if binary else float("nan"),
        mean_predicted_uplift=float(score.mean()),
        observed_ate_test=float(yt[tt == 1].mean() - yt[tt == 0].mean()),
        deciles=tuple(by_decile(yt, score, tt)),
    )
