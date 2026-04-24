# LLM Evaluation Framework — Spec

**Project:** `llm-eval`
**Status:** Phase C done — first live run on 2026-04-24, gpt-4o 68.9% vs gpt-4o-mini 55.6% on 45 prompts
**Author:** Andrey Ovsyannikov (ALchemt)
**Target role signal:** Junior AI Engineer / AI Automation Developer / Generative AI Specialist

## Problem

Most "LLM eval" portfolio projects are a single benchmark run: pick MMLU / GSM8K / HumanEval, call an API, screenshot the score. That demonstrates that a candidate can `pip install` an eval harness, nothing more.

A working AI engineer actually cares about: _does my system still behave when I change the prompt / model / temperature, and can I detect a regression?_ This project is a minimal working framework for exactly that — test suites × run configs × LLM-as-judge → reproducible reports and diffs.

It is a portfolio artifact, not a library — judged on clarity, evaluation rigor, and whether the code could realistically become the seed of an internal eval pipeline.

## User

Hypothetical: an engineer shipping an LLM feature who needs to answer "is version 2 of my prompt worse than version 1 on our factuality suite?" without hand-reading 30 outputs.

Real: the recruiter reading the README in under 5 minutes.

## Scope

### In

- 3 curated test suites (factuality, instruction_following, reasoning), 15 prompts each
- Config-driven runner: `configs.yaml` lists run configs (model × provider × temperature × system prompt); add a row = add a run
- Three rubric modes per prompt: `exact`, `contains`, `judge` (LLM-as-judge)
- LLM-as-judge with structured JSON verdict (`{pass: bool, reason: str}`)
- Metrics: accuracy, p50/p95 latency, input/output tokens, rough cost in USD, judge-vs-human agreement rate
- Report: markdown + per-run summary CSV + Streamlit dashboard with run-to-run diff
- Offline `--dry-run` mode with mock LLM so the whole pipeline is exercisable without an HF token
- Deploy: Hugging Face Spaces (free tier) — dashboard over committed results

### Out

- Training / fine-tuning
- Judge agreement calibration beyond simple match rate (no Cohen's kappa, no inter-rater sampling plans)
- Streaming / async runners (sequential is fine for this size)
- Arbitrary benchmark import (MMLU-style multiple choice, code execution) — hooks are left open
- Detailed cost modelling per provider — a single pricing dict covers the one model in use

## Architecture

```
  suites/*.jsonl ─┐
                  ├──▶ runner.py ──▶ runs/raw_<run_id>.jsonl
  configs.yaml ──┘        │ (dry-run|live)
                          │
                          ▼
                      judge.py ──▶ runs/scores_<run_id>.csv
                   (exact|contains|judge)
                          │
                          ▼
                     metrics.py
                     (accuracy, latency p50/p95, cost,
                      agreement vs human_score)
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
           report.py            app.py
           report_*.md          Streamlit
           summary_*.csv        diff between runs
```

Data formats:

**Suite prompt**
```json
{"id": "fact_01", "suite": "factuality", "prompt": "...", "expected": "...",
 "rubric": "exact|contains|judge", "expected_contains": ["..."], "human_score": null}
```

**Run config (configs.yaml)**
```yaml
runs:
  - id: baseline
    model: Qwen/Qwen2.5-72B-Instruct
    provider: together
    temperature: 0.0
    system_prompt: "..."
```

**Raw sample (runs/raw_<id>.jsonl)** — everything needed to re-score without re-calling the model: run_id, suite, prompt_id, prompt, expected, rubric, response, latency_ms, input_tokens, output_tokens, model, temperature, seed, mock.

## Tech choices

| Layer | Choice | Why |
|---|---|---|
| LLM under test | `gpt-4o` and `gpt-4o-mini` via OpenAI API | cross-size compare within one family isolates the capability axis from the vendor axis; pricing difference (~15×) makes the cost/quality tradeoff visible in the report |
| Judge | `gpt-4o-mini` (same as one of the configs under test) | scaffold simplicity; `configs.yaml.judge` slot allows swap in one edit |
| Provider abstraction | OpenAI-compatible `/chat/completions` via the `openai` SDK | works against OpenAI, OpenRouter, Groq, Together without per-provider code paths (see `PROVIDER_ENDPOINTS` in `src/runner.py`) |
| Runner | sequential Python, no async | suite sizes are tiny; cognitive load of async isn't earned |
| Storage | jsonl (raw) + csv (scores/summary) | grep-able, diff-able, git-friendly; no DB needed |
| Aggregation | pandas + numpy | same muscle as Project 1's eval |
| UI | Streamlit | fastest path, HF Spaces native, consistent with Project 1 |
| Config | YAML | readable diff when a reviewer compares run configs |
| Determinism | `temperature=0.0` by default + fixed seed field (saved with each sample) | reproducibility per-run, even if provider doesn't honour seed |

**Known limitation — self-judge bias.** `gpt-4o-mini` is both one of the two models under test and the judge. Results from the `judge` rubric for the `gpt_4o_mini` run are expected to skew positive compared to the `gpt_4o` run (a model judging its own output is structurally friendlier). That asymmetry is itself a useful diagnostic — `agreement_vs_human` is the honest metric once labels are in.

## Suites

Each suite is a jsonl file in `suites/`. 15 prompts per suite, authored by hand with hand-verified gold answers.

- **factuality** (15) — short-answer questions with a stable correct answer (Transformer/RAG/BPE/context-window/hallucination facts). Mix of `judge` and `contains` rubrics.
- **instruction_following** (15) — format constraints the response must obey (exact bullet counts, valid JSON with specified keys, casing, single-word answers). Mix of `judge` and `exact`.
- **reasoning** (15) — multi-step word problems with unambiguous numeric answers (arithmetic, simple algebra, sets, time, sequences). Mostly `exact`/`contains`, a few `judge` where format is ambiguous.

`human_score` starts empty across all suites — it gets filled by the author after the first live run so that `agreement_vs_human` in the report becomes meaningful.

## Judge rubric

Judge receives: `PROMPT`, `EXPECTED`, `RESPONSE`. Returns a single-line JSON:

```json
{"pass": true, "reason": "one short sentence"}
```

System prompt forces:
- factual correctness check against `EXPECTED`
- strict format-constraint check (counts, JSON validity, casing)
- no output beyond the JSON verdict

Parser extracts the first `{...}` block with regex; malformed output fails open as `pass=false` with reason `"judge parse error: <prefix>"`. That's intentional — a judge that can't be parsed is a judge that can't be trusted.

## Metrics

Per `(run_id, suite)`:

- **accuracy** — share of samples with `passed=True`
- **n** — sample count
- **p50_latency_ms / p95_latency_ms** — end-to-end LLM call latency
- **input_tokens / output_tokens** — summed across samples
- **est_cost_usd** — tokens × pricing dict in `src/metrics.py`
- **agreement_vs_human** — share of samples where judge verdict agrees with `human_score >= 1` (NaN until humans labels exist)

Report table shape:

| Config | Accuracy | p50 latency | p95 latency | Tokens in/out | $/run | Agreement vs human |
|---|---|---|---|---|---|---|
| baseline (qwen72b, T=0.0) | TBD | TBD | TBD | TBD / TBD | TBD | TBD |
| low_temp_strict (qwen72b, T=0.2) | TBD | TBD | TBD | TBD / TBD | TBD | TBD |

Filled after first live run (requires HF_TOKEN).

## Milestones

| Phase | Deliverable | Time budget |
|---|---|---|
| A Spec + Scaffold (done) | this file, working skeleton, `--dry-run` exercisable end-to-end | 2 h |
| B Suite growth (done) | 15 prompts × 3 suites, reasoning suite added | 1 h |
| C First live run (done) | Run on real Qwen via HF Inference, fill eval table in README | 1 h |
| C' Human labelling | score 10+ samples per suite by hand, populate `agreement_vs_human` | 1 h |
| D Polish + deploy | README final, HF Spaces live URL, screenshots | 1 h |

Total: ~7 h over 2–3 sessions after scaffold.

## Success criteria

- `python -m src.runner --dry-run && python -m src.judge --dry-run && python -m src.report` succeeds on a fresh checkout with no HF_TOKEN
- `streamlit run app.py` shows a non-empty summary table and at least one between-run diff
- Adding a new run config or new suite does not require editing any file in `src/`
- README contains a filled evaluation table after Phase C (not TBD)

## What I'd do differently (filled in README at Phase D)

Placeholder — filled honestly after shipping. Likely candidates: swap judge to a different model family (cross-provider), add Cohen's kappa for agreement, wire a simple regression gate (fail build if accuracy drops > X points on any suite), persist runs as dated subdirs instead of overwriting.

## Out of scope for v1 (maybe v2)

- Web UI to author / edit suites
- Parallel async runner
- Multi-judge ensemble
- Detailed cost curves across providers
- Integration with an existing eval library (lm-eval-harness, promptfoo) — the point here is to show the shape of the problem, not reinvent the ecosystem
