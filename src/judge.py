"""Score raw responses using three rubric modes.

Rubric modes per prompt:
    exact    — normalized string equality with `expected`
    contains — all tokens in `expected_contains` appear in response (case-insensitive)
    judge    — LLM-as-judge returns JSON {pass: bool, reason: str}

Output: `runs/scores_<run_id>.csv` — one row per sample with pass/fail + reason.

Usage:
    python -m src.judge                 # judge all raw_*.jsonl in runs/
    python -m src.judge --dry-run       # mock judge (simple heuristics)
    python -m src.judge --runs baseline # limit to a single run id
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.runner import PROVIDER_ENDPOINTS, SUITES_DIR, load_suite

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CONFIGS_PATH = ROOT / "configs.yaml"

load_dotenv(ROOT / ".env")


JUDGE_SYSTEM = """You grade whether a model's RESPONSE satisfies the PROMPT and the EXPECTED answer.

Return ONLY a single JSON object on one line:
{"pass": true|false, "reason": "<one short sentence>"}

Rules:
- If the response is factually correct and meets any format constraints in the prompt, pass=true.
- If it contradicts the expected answer, hallucinates, or breaks the format, pass=false.
- Be strict about format constraints (exact counts, JSON validity, casing).
- Do not output anything besides the JSON object."""


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def judge_exact(response: str, expected: str) -> tuple[bool, str]:
    ok = normalize(response) == normalize(expected)
    return ok, "exact match" if ok else f"expected '{expected}', got '{response[:80]}'"


def judge_contains(response: str, tokens: list[str]) -> tuple[bool, str]:
    missing = [t for t in tokens if t.lower() not in response.lower()]
    if not missing:
        return True, f"contains all of {tokens}"
    return False, f"missing tokens: {missing}"


def judge_mock(prompt: str, expected: str, response: str) -> tuple[bool, str]:
    """Cheap heuristic for --dry-run: pass if response is nonempty and
    doesn't look like an obvious failure. Good enough to exercise the pipeline."""
    if not response.strip():
        return False, "empty response"
    if "[MOCK" in response:
        return True, "mock pipeline ok"
    return True, "heuristic pass"


def judge_llm(prompt: str, expected: str, response: str, cfg: dict) -> tuple[bool, str]:
    from openai import OpenAI

    endpoint = PROVIDER_ENDPOINTS.get(cfg["provider"])
    if endpoint is None:
        raise RuntimeError(f"Unknown judge provider '{cfg['provider']}'")
    api_key = os.getenv(endpoint["env_key"])
    if not api_key:
        raise RuntimeError(f"{endpoint['env_key']} not set. Use --dry-run for offline scaffold.")

    client = OpenAI(api_key=api_key, base_url=endpoint["base_url"])
    user_msg = (
        f"PROMPT:\n{prompt}\n\n"
        f"EXPECTED:\n{expected}\n\n"
        f"RESPONSE:\n{response}\n\n"
        "Return the JSON verdict now."
    )
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 256),
    )
    raw = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False, f"judge parse error: {raw[:120]}"
    try:
        verdict = json.loads(match.group(0))
        return bool(verdict.get("pass")), str(verdict.get("reason", ""))[:200]
    except json.JSONDecodeError:
        return False, f"judge parse error: {raw[:120]}"


def build_prompt_index() -> dict[tuple[str, str], dict]:
    """Look up suite prompt metadata by (suite, id) — needed for expected_contains."""
    idx: dict[tuple[str, str], dict] = {}
    for path in SUITES_DIR.glob("*.jsonl"):
        for p in load_suite(path.stem):
            idx[(path.stem, p["id"])] = p
    return idx


def score_run(run_id: str, cfg: dict, dry_run: bool) -> Path:
    raw_path = RUNS_DIR / f"raw_{run_id}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw file for run '{run_id}': {raw_path}")

    prompt_idx = build_prompt_index()
    rows = []
    with raw_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            meta = prompt_idx.get((s["suite"], s["prompt_id"]), {})
            rubric = s.get("rubric", "judge")

            if rubric == "exact":
                passed, reason = judge_exact(s["response"], s["expected"])
            elif rubric == "contains":
                tokens = meta.get("expected_contains", [])
                passed, reason = judge_contains(s["response"], tokens)
            elif dry_run:
                passed, reason = judge_mock(s["prompt"], s["expected"], s["response"])
            else:
                passed, reason = judge_llm(s["prompt"], s["expected"], s["response"], cfg)

            rows.append(
                {
                    "run_id": s["run_id"],
                    "suite": s["suite"],
                    "prompt_id": s["prompt_id"],
                    "rubric": rubric,
                    "passed": passed,
                    "reason": reason,
                    "latency_ms": s["latency_ms"],
                    "input_tokens": s["input_tokens"],
                    "output_tokens": s["output_tokens"],
                    "human_score": meta.get("human_score"),
                }
            )

    df = pd.DataFrame(rows)
    out = RUNS_DIR / f"scores_{run_id}.csv"
    df.to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runs", nargs="*")
    args = ap.parse_args()

    with CONFIGS_PATH.open() as f:
        cfg = yaml.safe_load(f)
    judge_cfg = cfg["judge"]

    raw_files = sorted(RUNS_DIR.glob("raw_*.jsonl"))
    run_ids = [p.stem.replace("raw_", "") for p in raw_files]
    if args.runs:
        run_ids = [r for r in run_ids if r in args.runs]

    if not run_ids:
        print("No raw files to judge. Run `python -m src.runner` first.")
        return 1

    for rid in run_ids:
        out = score_run(rid, judge_cfg, args.dry_run)
        df = pd.read_csv(out)
        acc = df["passed"].mean()
        print(f"{rid}: {int(df['passed'].sum())}/{len(df)} passed ({acc:.1%}) → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
