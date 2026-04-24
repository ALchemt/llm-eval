"""Run test suites against a set of LLM configs.

For each run config in configs.yaml and each prompt across the specified
suites, call the LLM and append the raw response as a jsonl line into
`runs/raw_<run_id>.jsonl`.

Usage:
    python -m src.runner                 # real LLM calls (needs HF_TOKEN)
    python -m src.runner --dry-run       # deterministic mock LLM (no token)
    python -m src.runner --runs baseline # limit to a single run id
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SUITES_DIR = ROOT / "suites"
RUNS_DIR = ROOT / "runs"
CONFIGS_PATH = ROOT / "configs.yaml"

load_dotenv(ROOT / ".env")


@dataclass
class Sample:
    run_id: str
    suite: str
    prompt_id: str
    prompt: str
    expected: str
    rubric: str
    response: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model: str
    temperature: float
    seed: int
    mock: bool


def load_configs() -> dict:
    with CONFIGS_PATH.open() as f:
        return yaml.safe_load(f)


def load_suite(suite_name: str) -> list[dict]:
    path = SUITES_DIR / f"{suite_name}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def discover_suites() -> list[str]:
    return sorted(p.stem for p in SUITES_DIR.glob("*.jsonl"))


def mock_call(prompt: str, system_prompt: str, temperature: float) -> tuple[str, int, int]:
    """Deterministic echo response for offline scaffold runs."""
    text = f"[MOCK T={temperature}] Echo of: {prompt[:80]}"
    return text, len(prompt.split()), len(text.split())


PROVIDER_ENDPOINTS = {
    "openai": {"base_url": None, "env_key": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY"},
    "together": {"base_url": "https://api.together.xyz/v1", "env_key": "TOGETHER_API_KEY"},
}


def live_call(
    prompt: str,
    system_prompt: str,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    """OpenAI-compatible chat completion. Works against OpenAI, OpenRouter,
    Groq, Together, etc. — all expose the same Chat Completions schema."""
    from openai import OpenAI

    cfg = PROVIDER_ENDPOINTS.get(provider)
    if cfg is None:
        raise RuntimeError(f"Unknown provider '{provider}'. Add it to PROVIDER_ENDPOINTS.")

    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise RuntimeError(f"{cfg['env_key']} not set. Copy .env.example to .env or pass --dry-run.")

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return (
        text,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


def run_one(
    run_cfg: dict,
    suite_name: str,
    prompts: list[dict],
    dry_run: bool,
    seed: int,
) -> list[Sample]:
    samples: list[Sample] = []
    for p in prompts:
        t0 = time.time()
        if dry_run:
            text, in_tok, out_tok = mock_call(
                p["prompt"], run_cfg["system_prompt"], run_cfg["temperature"]
            )
        else:
            text, in_tok, out_tok = live_call(
                p["prompt"],
                run_cfg["system_prompt"],
                run_cfg["model"],
                run_cfg["provider"],
                run_cfg["temperature"],
                run_cfg.get("max_tokens", 512),
            )
        latency_ms = int((time.time() - t0) * 1000)
        samples.append(
            Sample(
                run_id=run_cfg["id"],
                suite=suite_name,
                prompt_id=p["id"],
                prompt=p["prompt"],
                expected=p.get("expected", ""),
                rubric=p.get("rubric", "judge"),
                response=text,
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                model=run_cfg["model"],
                temperature=run_cfg["temperature"],
                seed=seed,
                mock=dry_run,
            )
        )
    return samples


def write_raw(run_id: str, samples: list[Sample]) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"raw_{run_id}.jsonl"
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="use mock LLM, no HF calls")
    ap.add_argument("--runs", nargs="*", help="limit to these run ids (default: all)")
    ap.add_argument("--suites", nargs="*", help="limit to these suites (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_configs()
    run_cfgs = cfg["runs"]
    if args.runs:
        run_cfgs = [r for r in run_cfgs if r["id"] in args.runs]
        if not run_cfgs:
            print(f"No runs match {args.runs}. Available: {[r['id'] for r in cfg['runs']]}")
            return 1

    suite_names = args.suites or discover_suites()
    if not suite_names:
        print("No suites found in suites/*.jsonl")
        return 1

    for run_cfg in run_cfgs:
        all_samples: list[Sample] = []
        for suite_name in suite_names:
            prompts = load_suite(suite_name)
            samples = run_one(run_cfg, suite_name, prompts, args.dry_run, args.seed)
            all_samples.extend(samples)
        out = write_raw(run_cfg["id"], all_samples)
        mode = "DRY" if args.dry_run else "LIVE"
        print(f"[{mode}] {run_cfg['id']}: wrote {len(all_samples)} samples to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
