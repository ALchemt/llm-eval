"""Produce a markdown report and summary CSV from per-run scores.

Reads everything in `runs/scores_*.csv`, aggregates via metrics.build_summary(),
writes `runs/summary_<ts>.csv` + `runs/report_<ts>.md`.

Usage:
    python -m src.report
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from src.metrics import RUNS_DIR, build_summary, diff_runs

ROOT = Path(__file__).resolve().parent.parent


def _to_md_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Minimal markdown table formatter — avoids tabulate dependency."""
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:{float_fmt}}" if pd.notna(v) else "NaN")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def render_md(summary) -> str:
    if summary.empty:
        return "# LLM Eval Report\n\nNo score files found. Run runner + judge first.\n"

    lines = [
        "# LLM Eval Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Per-run × suite summary",
        "",
        _to_md_table(summary),
        "",
    ]

    run_ids = sorted(summary["run_id"].unique())
    if len(run_ids) >= 2:
        a, b = run_ids[0], run_ids[1]
        delta = diff_runs(summary, a, b)
        if not delta.empty:
            lines += [
                f"## Accuracy delta: `{b}` vs `{a}`",
                "",
                _to_md_table(delta),
                "",
            ]

    lines += [
        "## Notes",
        "",
        "- `accuracy` = share of samples passing the rubric (exact/contains/judge).",
        "- `agreement_vs_human` = share of samples where judge verdict matches `human_score` "
        "in the suite file. NaN means no human labels yet.",
        "- `est_cost_usd` uses public pricing snapshot in `src/metrics.py`.",
        "- For scaffold / `--dry-run` outputs, responses come from a mock LLM and",
        "  accuracy numbers are not meaningful — the pipeline is exercised end-to-end only.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = build_summary()
    ts = int(time.time())

    summary_csv = RUNS_DIR / f"summary_{ts}.csv"
    report_md = RUNS_DIR / f"report_{ts}.md"

    if summary.empty:
        print("No scores found in runs/. Run runner + judge first.")
        return 1

    summary.to_csv(summary_csv, index=False)
    report_md.write_text(render_md(summary))
    print(f"Summary: {summary_csv.relative_to(ROOT)}")
    print(f"Report:  {report_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
