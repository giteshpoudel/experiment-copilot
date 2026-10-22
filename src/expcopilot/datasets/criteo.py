"""Criteo Uplift Prediction dataset v2.1 (~14M users), subsampled for speed.

Users were randomly assigned to be targeted by advertising (treatment, ~85% by design) or
not (control, ~15%). Outcomes: visit and conversion. Features f0-f11 are anonymized
pre-treatment features. The planned 85/15 split is NOT a sample-ratio mismatch: the
experiment's planned shares say so, and the SRM check must pass.

Place the file at data/raw/criteo-research-uplift-v2.1.csv.gz (Criteo's own link was returning
404 in October 2026, so it is fetched by hand). License: CC BY-NC-SA 4.0.
Source: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from expcopilot.datasets import DatasetError, verify
from expcopilot.models import (
    Arm,
    Experiment,
    MetricDef,
    MetricType,
    PreRegisteredPlan,
    Source,
    Status,
)
from expcopilot.store import Store

FILENAME = "criteo-research-uplift-v2.1.csv.gz"
# Published by the libuplift package for this file; checked with a warning, not trusted alone.
SHA256_REFERENCE = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
EXPERIMENT_ID = "criteo"
FEATURES = [f"f{i}" for i in range(12)]
TREATED_SHARE = 0.85
DEFAULT_SAMPLE = 1_000_000
SAMPLE_SEED = 42


def path(data_dir: Path) -> Path:
    return data_dir / "raw" / FILENAME


def experiment(n: int, sample_note: str) -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        name="criteo-ad-uplift",
        description=(
            "Criteo uplift dataset v2.1: users randomly targeted by advertising (treatment) or "
            f"held out (control) at a planned 85/15 split. {sample_note} The plan is "
            "retrospective, written for this case study; dates are placeholders."
        ),
        status=Status.COMPLETE,
        source=Source.CRITEO,
        unit="user",
        start_date=date(2018, 1, 1),
        end_date=date(2018, 1, 14),
        arms=[
            Arm(name="control", expected_share=1 - TREATED_SHARE, is_control=True),
            Arm(name="treatment", expected_share=TREATED_SHARE),
        ],
        metrics=[
            MetricDef(
                name="visit",
                type=MetricType.BINARY,
                column="visit",
                description="User visited the advertiser's site",
            ),
            MetricDef(
                name="conversion",
                type=MetricType.BINARY,
                column="conversion",
                description="User converted on the advertiser's site",
            ),
        ],
        covariates=FEATURES,
        segments=[],
        plan=PreRegisteredPlan(
            primary_metric="visit",
            secondary_metrics=["conversion"],
            guardrails=[],
            mde=0.02,
            alpha=0.05,
            multiple_testing="holm",
            planned_n_total=n,
        ),
    )


def sample_table(csv_gz: Path, sample: int | None = DEFAULT_SAMPLE) -> pa.Table:
    """Read the CSV with DuckDB and take a seeded reservoir sample of rows."""
    q = f"SELECT * FROM read_csv_auto('{csv_gz.as_posix()}', header=true)"
    if sample:
        q += f" USING SAMPLE reservoir({int(sample)} ROWS) REPEATABLE ({SAMPLE_SEED})"
    with duckdb.connect() as con:
        t = con.execute(q).to_arrow_table()
    missing = [
        c for c in [*FEATURES, "treatment", "visit", "conversion"] if c not in t.column_names
    ]
    if missing:
        raise DatasetError(f"{csv_gz} is missing columns {missing}")
    n = t.num_rows
    treat = t.column("treatment").to_numpy().astype(int)
    return pa.table(
        {
            "unit_id": np.arange(1, n + 1, dtype=np.int64),
            "arm": np.where(treat == 1, "treatment", "control"),
            "assigned_at": np.full(n, np.datetime64("2018-01-01T00:00:00", "s")),
            **{f: t.column(f).cast(pa.float64()) for f in FEATURES},
            "visit": t.column("visit").cast(pa.int8()),
            "conversion": t.column("conversion").cast(pa.int8()),
        }
    )


def load(
    store: Store,
    data_dir: Path,
    sample: int | None = DEFAULT_SAMPLE,
    overwrite: bool = False,
    check_digest: bool = True,
) -> tuple[Experiment, list[str]]:
    src = path(data_dir)
    if not src.exists():
        raise DatasetError(
            f"{src} not found. Download criteo-research-uplift-v2.1.csv.gz from a source you "
            "trust (see https://ailab.criteo.com/criteo-uplift-prediction-dataset/) into "
            f"{src.parent}/"
        )
    notes: list[str] = []
    if check_digest:
        digest = verify(src, None)
        if digest != SHA256_REFERENCE:
            notes.append(
                f"SHA-256 {digest} differs from the published reference "
                f"{SHA256_REFERENCE}; check the file's source."
            )
    units = sample_table(src, sample)
    note = (
        f"This is a seeded random sample of {units.num_rows:,} users."
        if sample
        else f"All {units.num_rows:,} users."
    )
    exp = experiment(units.num_rows, note)
    store.save(exp, units, overwrite=overwrite)
    return exp, notes
