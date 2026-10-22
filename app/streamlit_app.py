"""Experimentation Copilot demo.

Replay mode (default, no API key): saved agent runs from results/readouts/, showing the
question, every tool call with its result id and summary, and the cited readout, next to the
no-tools baseline where one exists. Also shows the eval table and statistics validation.

Live mode (only when LLM_API_KEY is set and a local store exists): run the agent on any
experiment in the store.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
READOUTS = ROOT / "results" / "readouts"
EVAL_RESULTS = ROOT / "evals" / "results"
VALIDATION = ROOT / "results" / "validation.json"

DECISION_STYLE = {
    "ship": ("SHIP", "green"),
    "dont_ship": ("DON'T SHIP", "orange"),
    "extend": ("EXTEND", "blue"),
    "invalid": ("INVALID", "red"),
    None: ("NO DECISION", "gray"),
}
SCENARIO_NOTES = {
    "smoke-15": "Traffic split drifted from 50/50: the sample-ratio check must stop the analysis.",
    "smoke-16": "A redesign with a novelty spike that fades over two weeks; daily data available.",
    "smoke-19": "Four arms, so p-values need a multiple-testing adjustment before shipping.",
    "smoke-10": "A ratio metric (revenue per session) that needs the delta method.",
    "smoke-03": "A one-page checkout test with a conversion lift.",
    "hillstrom": "Real data: Kevin Hillstrom's 64,000-customer e-mail test, three arms.",
}


@st.cache_data
def load_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(READOUTS.glob("*.json")):
        data = json.loads(path.read_text())
        runs[path.stem] = data
    return runs


def decision_badge(decision: str | None) -> str:
    label, color = DECISION_STYLE.get(decision, DECISION_STYLE[None])
    return f":{color}-background[**{label}**]"


def linkify_citations(text: str) -> str:
    """Bold the [rN] citation labels so readers can match them to the trace, and demote the
    readout's own headings so they sit under the page's section titles."""
    text = re.sub(r"^(#{1,3}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", text, flags=re.M)
    return re.sub(r"\[(r\d+(?:\s*,\s*r\d+)*)\]", r"**[\1]**", text)


def show_trace(run: dict) -> None:
    for t in run["trace"]:
        label = t["label"] or "error"
        args = (
            t["arguments"]
            if isinstance(t["arguments"], str)
            else ", ".join(f"{k}={v}" for k, v in t["arguments"].items() if v is not None)
        )
        with st.expander(f"**[{label}]** `{t['tool']}({args})`", expanded=False):
            st.caption(
                f"result_id `{t['result_id']}` · method `{t.get('method')}` · "
                f"{t['latency_s']:.2f} s"
            )
            st.write(t["summary"])


def show_usage(run: dict) -> None:
    u = run["usage"]
    cols = st.columns(4)
    cols[0].metric("Tool calls", run["tool_calls"])
    cols[1].metric("Cost", f"${u['cost_usd']:.4f}")
    cols[2].metric("Latency", f"{run['latency_s']:.0f} s")
    cols[3].metric("Cached prompt", f"{u['cache_hit_tokens'] / max(u['prompt_tokens'], 1):.0%}")


def replay_page() -> None:
    runs = load_runs()
    agents = {k[: -len("-agent")]: v for k, v in runs.items() if k.endswith("-agent")}
    if not agents:
        st.warning("No saved runs in results/readouts/.")
        return
    eid = st.selectbox("Saved run", list(agents))
    if eid in SCENARIO_NOTES:
        st.caption(SCENARIO_NOTES[eid])
    run = agents[eid]
    st.markdown(f"**Question:** Should we ship `{eid}`?  {decision_badge(run['decision'])}")
    st.caption(f"Model {run['model']} · {run['turns']} turns")
    show_usage(run)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Readout")
        st.markdown(linkify_citations(run["text"]))
    with right:
        st.subheader("Tool calls")
        show_trace(run)
        baseline = runs.get(f"{eid}-baseline")
        if baseline:
            st.subheader("Same model, no tools")
            st.markdown(f"Decision: {decision_badge(baseline['decision'])}")
            st.caption(
                f"${baseline['usage']['cost_usd']:.4f} · "
                f"{baseline['latency_s']:.0f} s · computes its own statistics"
            )
            with st.expander("Baseline readout"):
                st.markdown(baseline["text"])


def evals_page() -> None:
    tests = sorted(EVAL_RESULTS.glob("*-test/table.md"))
    st.subheader("Evals on the held-out test split")
    if tests:
        st.markdown(tests[-1].read_text())
        st.caption(
            "40 simulated experiments with known effects x 3 runs per configuration; "
            "mean with the range across runs."
        )
    else:
        st.info("No eval results found.")
    if VALIDATION.exists():
        v = json.loads(VALIDATION.read_text())
        st.subheader("Statistics validation")
        rows = [
            {
                "check": c["name"],
                "observed": round(c["observed"], 4),
                "band": f"{c['band'][0]:.3f} to {c['band'][1]:.3f}",
                "passed": "yes" if c["passed"] else "NO",
            }
            for c in v["checks"]
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(f"{v['reps']} simulated experiments per check, commit {v['git_commit']}.")


def live_page() -> None:
    import asyncio

    from expcopilot.agent import run_agent, server_transport
    from expcopilot.llm import from_env
    from expcopilot.store import Store

    store = Store(ROOT / "data", read_only=True)
    ids = [r.experiment_id for r in store.list()]
    eid = st.selectbox("Experiment", ids)
    if st.button("Run the agent", type="primary"):
        with st.spinner("The agent is calling tools..."):
            run = asyncio.run(run_agent(eid, from_env("agent"), server_transport(store.data_dir)))
        data = run.to_dict()
        st.markdown(decision_badge(data["decision"]))
        show_usage(data)
        st.markdown(linkify_citations(data["text"]))
        show_trace(data)


def live_available() -> bool:
    return bool(os.environ.get("LLM_API_KEY")) and (ROOT / "data" / "expc.duckdb").exists()


def main() -> None:
    st.set_page_config(page_title="Experimentation Copilot", layout="wide")
    st.title("Experimentation Copilot")
    st.write(
        "An AI analyst for A/B tests that never does the math itself: every number in a "
        "readout comes from a validated statistics tool, cited by label."
    )
    pages = ["Replay saved runs", "Evals and validation"]
    if live_available():
        pages.append("Run live")
    page = st.radio("View", pages, horizontal=True, label_visibility="collapsed")
    if page == pages[0]:
        replay_page()
    elif page == pages[1]:
        evals_page()
    else:
        live_page()


main()
