"""Simulation validation of the statistics engine. The full run is slow (~40 s)."""

from __future__ import annotations

import json

import pytest

from expcopilot.validation import run_all, write


def test_quick_run_produces_every_check(tmp_path) -> None:
    report = run_all(reps=40, bootstrap_reps=20, cuped_reps=10)
    names = {c["name"] for c in report["checks"]}
    for method in ("two_proportion_z", "welch_t", "delta_method", "cuped_welch"):
        assert {f"coverage_abs_{method}", f"coverage_rel_{method}", f"aa_fpr_{method}"} <= names
    assert {"peeking_naive_inflates", "peeking_fpr_obf", "peeking_fpr_msprt"} <= names
    path = tmp_path / "v.json"
    write(report, path)
    assert json.loads(path.read_text())["reps"] == 40


@pytest.mark.slow
def test_full_validation_passes() -> None:
    report = run_all(reps=2000)
    failed = [
        f"{c['name']}: {c['observed']:.4f} not in {c['band']}"
        for c in report["checks"]
        if not c["passed"]
    ]
    assert not failed, failed
