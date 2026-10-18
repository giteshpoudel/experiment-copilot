from __future__ import annotations

import pyarrow as pa
import pytest

from expcopilot.simulate import generate, smoke_suite
from expcopilot.store import ExperimentExistsError, ExperimentNotFoundError, Store, TruthStore


@pytest.fixture(scope="module")
def sim():
    spec = smoke_suite()[0].model_copy(update={"n_total": 2_000})
    return generate(spec, suite="test")


def test_save_get_list_round_trip(store: Store, sim) -> None:
    store.save(sim.experiment, sim.units)
    assert store.exists("smoke-01")
    assert store.get("smoke-01") == sim.experiment
    [row] = store.list()
    assert row.experiment_id == "smoke-01"
    assert row.n_units == sim.units.num_rows
    counts = store.arm_counts("smoke-01")
    assert list(counts) == ["control", "treatment"]
    assert sum(counts.values()) == sim.units.num_rows


def test_save_refuses_duplicates_unless_overwrite(store: Store, sim) -> None:
    store.save(sim.experiment, sim.units)
    with pytest.raises(ExperimentExistsError):
        store.save(sim.experiment, sim.units)
    store.save(sim.experiment, sim.units.slice(0, 100), overwrite=True)
    assert store.list()[0].n_units == 100


def test_missing_columns_are_rejected(store: Store, sim) -> None:
    units = sim.units.drop_columns(["converted"])
    with pytest.raises(ValueError, match="missing columns"):
        store.save(sim.experiment, units)


def test_unknown_arm_is_rejected(store: Store, sim) -> None:
    arm = pa.array(["control"] * (sim.units.num_rows - 1) + ["ghost"])
    units = sim.units.set_column(sim.units.column_names.index("arm"), "arm", arm)
    with pytest.raises(ValueError, match="unknown arms"):
        store.save(sim.experiment, units)


def test_missing_experiment_raises(store: Store) -> None:
    with pytest.raises(ExperimentNotFoundError):
        store.get("nope")


def test_truth_lives_in_its_own_file(store: Store, truth_store: TruthStore, sim) -> None:
    store.save(sim.experiment, sim.units)
    truth_store.save(sim.truth)
    assert truth_store.db_path != store.db_path
    assert truth_store.get("smoke-01") == sim.truth
    with store.connect(read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert tables == {"experiments"}
