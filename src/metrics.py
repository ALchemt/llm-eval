"""Aggregate per-run scores into a summary table.

Metrics per (run_id, suite):
    accuracy       — share of passed samples
    n              — number of samples
    p50_latency_ms — median end-to-end latency
    p95_latency_ms — 95th percentile latency
    input_tokens   — sum of input tokens across samples
    output_tokens  — sum of output tokens across samples
    est_cost_usd   — rough cost from tokens (see PRICING table)
    agreement      — share of samples where judge verdict matches human_score
                     (NaN if no human labels yet — they are optional)

Call `build_summary()` from report.py / app.py to avoid duplicating logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

# Rough public pricing at time of scaffold. Kept here (not in configs.yaml)
# because it ties to model identity, not to a run config.
PRICING_USD_PER_MTOK = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
}


def _cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING_USD_PER_MTOK.get(model)
    if not p:
        return 0.0
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000


def load_all_scores() -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS_DIR.glob("scores_*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _model_for_run(run_id: str) -> str:
    """Recover model name from raw_*.jsonl (first line)."""
    import json

    raw = RUNS_DIR / f"raw_{run_id}.jsonl"
    if not raw.exists():
        return ""
    with raw.open() as f:
        first = f.readline()
    if not first.strip():
        return ""
    return json.loads(first).get("model", "")


def _agreement(group: pd.DataFrame) -> float:
    labeled = group.dropna(subset=["human_score"])
    if labeled.empty:
        return float("nan")
    judge_pass = labeled["passed"].astype(bool)
    human_pass = labeled["human_score"].astype(float) >= 1.0
    return float((judge_pass == human_pass).mean())


def build_summary() -> pd.DataFrame:
    scores = load_all_scores()
    if scores.empty:
        return scores

    groups = []
    for (run_id, suite), g in scores.groupby(["run_id", "suite"]):
        model = _model_for_run(run_id)
        groups.append(
            {
                "run_id": run_id,
                "suite": suite,
                "n": len(g),
                "accuracy": float(g["passed"].mean()),
                "p50_latency_ms": float(np.percentile(g["latency_ms"], 50)),
                "p95_latency_ms": float(np.percentile(g["latency_ms"], 95)),
                "input_tokens": int(g["input_tokens"].sum()),
                "output_tokens": int(g["output_tokens"].sum()),
                "est_cost_usd": _cost_usd(
                    model, int(g["input_tokens"].sum()), int(g["output_tokens"].sum())
                ),
                "agreement_vs_human": _agreement(g),
            }
        )
    return pd.DataFrame(groups).sort_values(["run_id", "suite"]).reset_index(drop=True)


def diff_runs(summary: pd.DataFrame, run_a: str, run_b: str) -> pd.DataFrame:
    """Per-suite accuracy delta between two runs: run_b - run_a."""
    pivot = summary.pivot_table(index="suite", columns="run_id", values="accuracy")
    if run_a not in pivot.columns or run_b not in pivot.columns:
        return pd.DataFrame()
    out = pivot[[run_a, run_b]].copy()
    out["delta"] = out[run_b] - out[run_a]
    return out.reset_index()
