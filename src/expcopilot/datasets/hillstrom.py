"""Kevin Hillstrom's MineThatData e-mail test (2008): 64,000 customers, three arms.

Customers who bought in the last 12 months were randomized to a men's merchandise e-mail, a
women's merchandise e-mail, or no e-mail, and tracked for two weeks: visit, conversion, spend.
`history` (dollars spent in the past year) is the pre-period covariate for CUPED.

Source: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
"""

from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

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

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
FILENAME = "hillstrom.csv"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
EXPERIMENT_ID = "hillstrom"
ARMS = {"No E-Mail": "no_email", "Mens E-Mail": "mens_email", "Womens E-Mail": "womens_email"}
SEGMENTS = ["history_segment", "channel", "zip_code", "newbie", "mens", "womens"]
COVARIATES = ["history", "recency"]


def path(data_dir: Path) -> Path:
    return data_dir / "raw" / FILENAME


def download(data_dir: Path) -> Path:
    target = path(data_dir)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, target)
    verify(target, SHA256)
    return target


def experiment(counts_total: int) -> Experiment:
    return Experiment(
        experiment_id=EXPERIMENT_ID,
        name="hillstrom-email-test",
        description=(
            "MineThatData e-mail test (Kevin Hillstrom, 2008): customers randomized to a men's "
            "e-mail, a women's e-mail or no e-mail, outcomes tracked for two weeks. The plan is "
            "retrospective, written for this case study; dates are approximate."
        ),
        status=Status.COMPLETE,
        source=Source.HILLSTROM,
        unit="customer",
        start_date=date(2008, 3, 1),
        end_date=date(2008, 3, 14),
        arms=[
            Arm(name="no_email", expected_share=1 / 3, is_control=True),
            Arm(name="mens_email", expected_share=1 / 3),
            Arm(name="womens_email", expected_share=1 / 3),
        ],
        metrics=[
            MetricDef(
                name="spend",
                type=MetricType.CONTINUOUS,
                column="spend",
                description="Dollars spent in the two weeks after the e-mail",
            ),
            MetricDef(
                name="visit",
                type=MetricType.BINARY,
                column="visit",
                description="Visited the website in the two weeks after the e-mail",
            ),
            MetricDef(
                name="conversion",
                type=MetricType.BINARY,
                column="conversion",
                description="Purchased in the two weeks after the e-mail",
            ),
        ],
        covariates=COVARIATES,
        segments=SEGMENTS,
        plan=PreRegisteredPlan(
            primary_metric="spend",
            secondary_metrics=["visit", "conversion"],
            guardrails=[],
            mde=0.15,
            alpha=0.05,
            multiple_testing="holm",
            planned_n_total=counts_total,
            primary_covariate="history",
        ),
    )


def units_table(csv_path: Path) -> pa.Table:
    t = pacsv.read_csv(csv_path)
    segment = t.column("segment").to_pylist()
    unknown = set(segment) - set(ARMS)
    if unknown:
        raise DatasetError(f"unexpected arms in {csv_path}: {sorted(unknown)}")
    n = t.num_rows
    start = np.datetime64("2008-03-01T00:00:00", "s")
    return pa.table(
        {
            "unit_id": np.arange(1, n + 1, dtype=np.int64),
            "arm": [ARMS[s] for s in segment],
            "assigned_at": np.full(n, start),
            "history_segment": t.column("history_segment").cast(pa.string()),
            "channel": t.column("channel").cast(pa.string()),
            "zip_code": t.column("zip_code").cast(pa.string()),
            "newbie": t.column("newbie").cast(pa.string()),
            "mens": t.column("mens").cast(pa.string()),
            "womens": t.column("womens").cast(pa.string()),
            "history": t.column("history").cast(pa.float64()),
            "recency": t.column("recency").cast(pa.float64()),
            "spend": t.column("spend").cast(pa.float64()),
            "visit": t.column("visit").cast(pa.int8()),
            "conversion": t.column("conversion").cast(pa.int8()),
        }
    )


def load(store: Store, data_dir: Path, overwrite: bool = False) -> Experiment:
    csv_path = download(data_dir)
    units = units_table(csv_path)
    exp = experiment(units.num_rows)
    store.save(exp, units, overwrite=overwrite)
    return exp
