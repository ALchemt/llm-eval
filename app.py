"""Streamlit dashboard for LLM eval runs.

Reads `runs/scores_*.csv` + `runs/raw_*.jsonl`, shows:
- per-run × suite summary table
- accuracy delta between two selected runs
- per-prompt diff (where run A and run B disagree)

Run locally: `streamlit run app.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.metrics import RUNS_DIR, build_summary, diff_runs, load_all_scores

st.set_page_config(page_title="LLM Eval Framework", page_icon=":bar_chart:", layout="wide")

st.title("LLM Evaluation Framework")
st.caption(
    "Regression eval over curated test suites × LLM configs. "
    "Manual rubric + LLM-as-judge + reproducible reports. "
    "[Source & spec on GitHub](https://github.com/ALchemt/llm-eval)"
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        - **Stack:** HF Inference Providers (Qwen 2.5 72B) · pandas · Streamlit
        - **Suites:** factuality, instruction_following (seed)
        - **Judge:** Qwen-as-judge with JSON verdict + rubric
        - **Reproducibility:** raw responses persisted in `runs/`
        """
    )
    st.markdown("---")
    st.caption("Portfolio project by Andrey Ovsyannikov")


summary = build_summary()

if summary.empty:
    st.warning(
        "No score files yet. Run in the repo:\n\n"
        "```\npython -m src.runner --dry-run\npython -m src.judge --dry-run\n```"
    )
    st.stop()

st.markdown("## Per-run × suite summary")
st.dataframe(
    summary.style.format(
        {
            "accuracy": "{:.1%}",
            "p50_latency_ms": "{:.0f}",
            "p95_latency_ms": "{:.0f}",
            "est_cost_usd": "${:.4f}",
            "agreement_vs_human": "{:.1%}",
        }
    ),
    use_container_width=True,
)

run_ids = sorted(summary["run_id"].unique())

if len(run_ids) >= 2:
    st.markdown("## Compare two runs")
    col1, col2 = st.columns(2)
    run_a = col1.selectbox("Run A", run_ids, index=0)
    run_b = col2.selectbox("Run B", run_ids, index=1)

    if run_a == run_b:
        st.info("Pick two different runs to see diffs.")
    else:
        delta = diff_runs(summary, run_a, run_b)
        if not delta.empty:
            st.markdown(f"### Accuracy delta per suite — `{run_b}` minus `{run_a}`")
            st.dataframe(
                delta.style.format({run_a: "{:.1%}", run_b: "{:.1%}", "delta": "{:+.1%}"}),
                use_container_width=True,
            )

        scores = load_all_scores()
        sub = scores[scores["run_id"].isin([run_a, run_b])]
        pivot = sub.pivot_table(
            index=["suite", "prompt_id"], columns="run_id", values="passed"
        )
        if run_a in pivot.columns and run_b in pivot.columns:
            disagree = pivot[pivot[run_a] != pivot[run_b]].reset_index()
            st.markdown(f"### Prompts where `{run_a}` and `{run_b}` disagree ({len(disagree)})")
            if disagree.empty:
                st.success("No disagreement on current samples.")
            else:
                st.dataframe(disagree, use_container_width=True)

st.markdown("## Raw samples viewer")
sel_run = st.selectbox("Run", run_ids, key="raw_viewer")
raw_path = RUNS_DIR / f"raw_{sel_run}.jsonl"
if raw_path.exists():
    with raw_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)[
        ["suite", "prompt_id", "prompt", "expected", "response", "latency_ms", "mock"]
    ]
    st.dataframe(df, use_container_width=True, height=400)
else:
    st.info(f"No raw file for {sel_run}.")
