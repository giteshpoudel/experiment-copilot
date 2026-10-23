"""The Streamlit demo renders in replay mode without an API key or a data store."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")


def test_replay_page_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Experimentation Copilot"
    assert len(at.radio[0].options) == 2  # no live mode without a key
    assert any("Readout" in h.value for h in at.subheader)


def test_every_saved_run_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    for i, option in enumerate(at.selectbox[0].options):
        at.selectbox[0].select_index(i).run()
        assert not at.exception, option


def test_evals_page_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio[0].set_value("Evals and validation").run()
    assert not at.exception
    assert any("held-out test" in h.value for h in at.subheader)


def test_runs_open_on_hillstrom_in_curated_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    options = list(at.selectbox[0].options)
    assert options[0] == "Hillstrom e-mail test (real data)"
    assert options[:3] == [
        "Hillstrom e-mail test (real data)",
        "Criteo ad targeting (real data)",
        "Traffic split went wrong (simulated)",
    ]
    assert at.selectbox[0].value == options[0]
