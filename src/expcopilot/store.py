"""DuckDB registry over per-experiment Parquet files, plus a separate ground-truth store.

Layout under the data directory:

    expc.duckdb          experiments registry (what tools and the agent may read)
    truth.duckdb         simulation ground truth (evals only; the MCP server never opens it)
    units/<id>.parquet   one row per unit
    daily/<id>.parquet   one row per unit per day (only experiments with has_daily)
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from expcopilot.models import Experiment, ExperimentSummary, SimulationTruth

DEFAULT_DATA_DIR = "data"
DAILY_COLUMNS = ("unit_id", "arm", "day")

REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    source        VARCHAR NOT NULL,
    unit          VARCHAR NOT NULL,
    start_date    DATE NOT NULL,
    end_date      DATE NOT NULL,
    n_units       BIGINT NOT NULL,
    units_path    VARCHAR NOT NULL,   -- relative to the data directory
    daily_path    VARCHAR,
    definition    VARCHAR NOT NULL,   -- Experiment model as JSON
    created_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""

TRUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS truth (
    experiment_id VARCHAR PRIMARY KEY,
    suite         VARCHAR NOT NULL,
    definition    VARCHAR NOT NULL,   -- SimulationTruth model as JSON
    created_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""


class ExperimentNotFoundError(KeyError):
    pass


class ExperimentExistsError(ValueError):
    pass


def default_data_dir() -> Path:
    return Path(os.environ.get("EXPC_DATA_DIR", DEFAULT_DATA_DIR))


def _sql_str(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


class Store:
    """Experiment registry and unit data. Opens a short-lived connection per call."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.db_path = self.data_dir / "expc.duckdb"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "units").mkdir(exist_ok=True)
        (self.data_dir / "daily").mkdir(exist_ok=True)
        with self.connect() as con:
            con.execute(REGISTRY_SCHEMA)

    @contextmanager
    def connect(self, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(self.db_path), read_only=read_only)
        try:
            yield con
        finally:
            con.close()

    # --- writes ---

    def save(
        self,
        experiment: Experiment,
        units: pa.Table,
        daily: pa.Table | None = None,
        overwrite: bool = False,
    ) -> None:
        eid = experiment.experiment_id
        self._check_units(experiment, units, daily)
        if self.exists(eid):
            if not overwrite:
                raise ExperimentExistsError(f"experiment {eid!r} already exists")
            self.delete(eid)

        units_rel = Path("units") / f"{eid}.parquet"
        pq.write_table(units, self.data_dir / units_rel)
        daily_rel = None
        if daily is not None:
            daily_rel = Path("daily") / f"{eid}.parquet"
            pq.write_table(daily, self.data_dir / daily_rel)

        with self.connect() as con:
            con.execute(
                """
                INSERT INTO experiments (experiment_id, name, status, source, unit, start_date,
                    end_date, n_units, units_path, daily_path, definition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    eid,
                    experiment.name,
                    experiment.status.value,
                    experiment.source.value,
                    experiment.unit,
                    experiment.start_date,
                    experiment.end_date,
                    units.num_rows,
                    str(units_rel),
                    str(daily_rel) if daily_rel else None,
                    experiment.model_dump_json(),
                ],
            )

    def delete(self, experiment_id: str) -> None:
        row = self._row(experiment_id, "units_path, daily_path")
        for rel in row:
            if rel:
                (self.data_dir / rel).unlink(missing_ok=True)
        with self.connect() as con:
            con.execute("DELETE FROM experiments WHERE experiment_id = ?", [experiment_id])

    @staticmethod
    def _check_units(experiment: Experiment, units: pa.Table, daily: pa.Table | None) -> None:
        missing = [c for c in experiment.unit_columns if c not in units.column_names]
        if missing:
            raise ValueError(f"unit table is missing columns: {missing}")
        arms = {a.name for a in experiment.arms}
        seen = set(units.column("arm").unique().to_pylist())
        if not seen <= arms:
            raise ValueError(f"unit table has unknown arms: {sorted(seen - arms)}")
        if daily is None:
            if experiment.has_daily:
                raise ValueError("experiment has_daily but no daily table was given")
            return
        if not experiment.has_daily:
            raise ValueError("daily table given but experiment.has_daily is False")
        missing = [c for c in DAILY_COLUMNS if c not in daily.column_names]
        if missing:
            raise ValueError(f"daily table is missing columns: {missing}")

    # --- reads ---

    def exists(self, experiment_id: str) -> bool:
        with self.connect(read_only=True) as con:
            hit = con.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?", [experiment_id]
            ).fetchone()
        return hit is not None

    def get(self, experiment_id: str) -> Experiment:
        (definition,) = self._row(experiment_id, "definition")
        return Experiment.model_validate_json(definition)

    def list(self) -> list[ExperimentSummary]:
        with self.connect(read_only=True) as con:
            rows = con.execute(
                "SELECT experiment_id, name, status, source, n_units FROM experiments "
                "ORDER BY experiment_id"
            ).fetchall()
        return [
            ExperimentSummary(experiment_id=r[0], name=r[1], status=r[2], source=r[3], n_units=r[4])
            for r in rows
        ]

    def units_path(self, experiment_id: str) -> Path:
        (rel,) = self._row(experiment_id, "units_path")
        return self.data_dir / rel

    def daily_path(self, experiment_id: str) -> Path | None:
        (rel,) = self._row(experiment_id, "daily_path")
        return self.data_dir / rel if rel else None

    def arm_counts(self, experiment_id: str) -> dict[str, int]:
        """Observed units per arm, in the experiment's arm order."""
        experiment = self.get(experiment_id)
        path = self.units_path(experiment_id)
        with self.connect(read_only=True) as con:
            rows = con.execute(
                f"SELECT arm, count(*) FROM read_parquet({_sql_str(path)}) GROUP BY arm"
            ).fetchall()
        counts = dict(rows)
        return {a.name: int(counts.get(a.name, 0)) for a in experiment.arms}

    def _row(self, experiment_id: str, columns: str) -> tuple:
        with self.connect(read_only=True) as con:
            row = con.execute(
                f"SELECT {columns} FROM experiments WHERE experiment_id = ?", [experiment_id]
            ).fetchone()
        if row is None:
            raise ExperimentNotFoundError(experiment_id)
        return row


class TruthStore:
    """Ground truth for simulated experiments, in its own DuckDB file.

    Kept apart from the registry so the MCP server (which only gets a `Store`) cannot read the
    answers, even by accident. Only the simulator, the eval oracle and `expc describe --truth`
    use it.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "truth.duckdb"
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(TRUTH_SCHEMA)

    def save(self, truth: SimulationTruth) -> None:
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(
                "INSERT OR REPLACE INTO truth (experiment_id, suite, definition) VALUES (?, ?, ?)",
                [truth.experiment_id, truth.suite, truth.model_dump_json()],
            )

    def get(self, experiment_id: str) -> SimulationTruth:
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            row = con.execute(
                "SELECT definition FROM truth WHERE experiment_id = ?", [experiment_id]
            ).fetchone()
        if row is None:
            raise ExperimentNotFoundError(experiment_id)
        return SimulationTruth.model_validate_json(row[0])

    def delete(self, experiment_id: str) -> None:
        with duckdb.connect(str(self.db_path)) as con:
            con.execute("DELETE FROM truth WHERE experiment_id = ?", [experiment_id])
