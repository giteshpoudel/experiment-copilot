from __future__ import annotations

from pathlib import Path

import pytest

from expcopilot.store import Store, TruthStore


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def store(data_dir: Path) -> Store:
    return Store(data_dir)


@pytest.fixture
def truth_store(data_dir: Path) -> TruthStore:
    return TruthStore(data_dir)
