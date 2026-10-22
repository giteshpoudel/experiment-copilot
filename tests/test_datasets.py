"""Dataset loaders on small fixtures in the real files' formats (CI has no raw data), plus a
check against the real Hillstrom file when it is present locally."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from expcopilot.analysis import Analyzer
from expcopilot.datasets import DatasetError, criteo, hillstrom
from expcopilot.store import Store

REAL_HILLSTROM = Path(__file__).parents[1] / "data" / "raw" / hillstrom.FILENAME


def _hillstrom_csv(path: Path, n: int = 3_000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arms = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n)
    lines = [
        "recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,"
        "visit,conversion,spend"
    ]
    for a in arms:
        visit = int(rng.random() < (0.10 if a == "No E-Mail" else 0.17))
        conv = int(visit and rng.random() < 0.08)
        spend = round(float(rng.gamma(2, 50)), 2) if conv else 0
        lines.append(
            f"{rng.integers(1, 13)},1) $0 - $100,{rng.gamma(2, 100):.2f},1,0,"
            f"Urban,{rng.integers(0, 2)},Web,{a},{visit},{conv},{spend}"
        )
    path.write_text("\n".join(lines) + "\n")


def test_hillstrom_units_and_plan(tmp_path: Path) -> None:
    csv = tmp_path / "h.csv"
    _hillstrom_csv(csv)
    units = hillstrom.units_table(csv)
    exp = hillstrom.experiment(units.num_rows)
    assert set(units.column("arm").to_pylist()) == {"no_email", "mens_email", "womens_email"}
    assert exp.plan.primary_metric == "spend" and exp.plan.primary_covariate == "history"
    assert exp.control.name == "no_email"
    store = Store(tmp_path / "data")
    store.save(exp, units)
    an = Analyzer(Store(tmp_path / "data", read_only=True))
    assert an.check_srm("hillstrom").passed
    res = an.analyze_metric("hillstrom", "spend", cuped_covariate="history")
    assert len(res.comparisons) == 2 and res.method == "cuped_welch"


def test_hillstrom_rejects_unknown_arms(tmp_path: Path) -> None:
    csv = tmp_path / "h.csv"
    _hillstrom_csv(csv, n=10)
    csv.write_text(csv.read_text().replace("No E-Mail", "Postcard"))
    with pytest.raises(DatasetError, match="unexpected arms"):
        hillstrom.units_table(csv)


@pytest.mark.skipif(not REAL_HILLSTROM.exists(), reason="raw Hillstrom file not downloaded")
def test_real_hillstrom_file() -> None:
    hillstrom.verify(REAL_HILLSTROM, hillstrom.SHA256)
    units = hillstrom.units_table(REAL_HILLSTROM)
    assert units.num_rows == 64_000
    arms = units.column("arm").to_numpy()
    counts = {a: int((arms == a).sum()) for a in ("no_email", "mens_email", "womens_email")}
    assert counts == {"no_email": 21_306, "mens_email": 21_307, "womens_email": 21_387}


def _criteo_gz(path: Path, n: int = 4_000, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    header = ",".join([*criteo.FEATURES, "treatment", "conversion", "visit", "exposure"])
    rows = [header]
    for _ in range(n):
        f = rng.normal(size=12)
        t = int(rng.random() < 0.85)
        visit = int(rng.random() < 0.04 + 0.02 * t * (f[0] > 0))
        conv = int(visit and rng.random() < 0.05)
        rows.append(",".join([*(f"{v:.4f}" for v in f), str(t), str(conv), str(visit), str(t)]))
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(rows) + "\n")


def test_criteo_load_sample_and_planned_imbalance(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    _criteo_gz(raw / criteo.FILENAME)
    store = Store(tmp_path / "data")
    exp, notes = criteo.load(store, tmp_path / "data", sample=3_000)
    assert notes and "differs from the published reference" in notes[0]  # fixture, not real
    an = Analyzer(Store(tmp_path / "data", read_only=True))
    assert an.describe_experiment("criteo").n_total == 3_000
    srm = an.check_srm("criteo")
    assert srm.passed, srm.summary  # 85/15 is planned, not a mismatch
    up = an.uplift_model("criteo", "visit", learner="x")
    assert up.n_train + up.n_test == 3_000 and len(up.deciles) == 10
    assert exp.covariates == criteo.FEATURES


def test_criteo_missing_file_explains_where_to_put_it(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"criteo-research-uplift-v2\.1\.csv\.gz"):
        criteo.load(Store(tmp_path / "d"), tmp_path / "d")
