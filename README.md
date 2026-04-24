---
title: LLM Evaluation Framework
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# LLM Evaluation Framework

> A working skeleton for regression evaluation of LLM systems — test suites × run configs × LLM-as-judge → reproducible reports and diffs. Built to show eval rigor beyond "I ran a benchmark once".

## Problem

Most eval portfolio pieces are single benchmark screenshots. Real LLM work needs a way to answer: _did my change make this worse on the suite I care about?_ This project is the minimum pipeline that lets you answer that, and extend it.

## Demo

**Live:** **https://huggingface.co/spaces/ALchemt/llm-eval** (deploy checklist: [DEPLOY.md](./DEPLOY.md))

The deployed dashboard is **read-only** — it renders the committed `runs/` snapshot (no live LLM calls, no token burn). To reproduce or extend locally, see [Quick start](#quick-start-local-offline--no-openai-token).

**What the dashboard shows:**
1. Per-run × suite summary (accuracy, latency p50/p95, tokens, cost, judge-vs-human agreement)
2. Accuracy delta between any two runs
3. Prompts where two runs disagree (a regression detector in table form)
4. Raw sample viewer (prompt, expected, actual response, latency)

## Architecture

```
suites/*.jsonl ─┐
                ├─▶ runner.py ─▶ runs/raw_<id>.jsonl ─▶ judge.py ─▶ runs/scores_<id>.csv
configs.yaml ──┘                                                    │
                                                                    ▼
                                                        metrics.py (aggregate)
                                                            │
                                                  ┌─────────┴─────────┐
                                                  ▼                   ▼
                                              report.py            app.py
                                           report_*.md          Streamlit
```

See [spec.md](./spec.md) for design decisions, tradeoffs, and roadmap.

## Tech stack

| Layer | Choice |
|---|---|
| LLM under test | `gpt-4o` and `gpt-4o-mini` via OpenAI API (cross-size compare within the same family) |
| Judge | `gpt-4o-mini` (configurable in `configs.yaml` — see note below) |
| Provider abstraction | OpenAI-compatible SDK path, configurable via `PROVIDER_ENDPOINTS` — supports OpenRouter / Groq / OpenAI / Together without code changes |
| Rubric modes | `exact`, `contains`, `judge` (LLM-as-judge with JSON verdict) |
| Storage | jsonl (raw) + csv (scores/summary) — git-friendly, diff-able |
| Aggregation | pandas + numpy |
| UI | Streamlit |
| Offline mode | `--dry-run` flag → mock LLM, no HF token needed |

**Self-judge note.** `gpt-4o-mini` is both one of the configs under test _and_ the judge. Expect its scores on `judge`-rubric prompts to be friendlier than `gpt-4o`'s — a model judging its own output is structurally biased. That asymmetry is itself diagnostic; `agreement_vs_human` is the honest metric once manual labels are added. Swapping the judge is a one-line change in the `judge:` block of `configs.yaml`.

## Suites

- **`factuality`** (15) — short-answer questions with a stable correct answer (Transformer / RAG / BPE / context-window / hallucination facts). Mix of `judge` and `contains` rubrics.
- **`instruction_following`** (15) — format constraints (exact-count bullet lists, valid JSON, casing, single-word answers). Mix of `judge` and `exact`.
- **`reasoning`** (15) — multi-step word problems with unambiguous numeric answers (arithmetic, simple algebra, sets, time, sequences). Mostly `exact` / `contains`.

All gold answers hand-verified. `human_score` starts empty across suites and gets filled after the first live run for `agreement_vs_human`.

## Evaluation

First live run on 2026-04-24. 45 prompts per config (15 × 3 suites), T=0.0, judge = `gpt-4o-mini`.

**Overall accuracy (pass-rate under the rubric):**

| Config | Accuracy | Total cost | Notes |
|---|---|---|---|
| **gpt-4o** | **31 / 45 = 68.9%** | $0.015 / run | stronger on reasoning |
| **gpt-4o-mini** | 25 / 45 = 55.6% | $0.001 / run | competitive on factuality, collapses on reasoning |

**Per suite:**

| Suite (n=15) | gpt-4o | gpt-4o-mini | Δ (mini − 4o) |
|---|---|---|---|
| factuality | 46.7% | 53.3% | **+6.7 pp** |
| instruction_following | 73.3% | 66.7% | −6.7 pp |
| reasoning | **86.7%** | 46.7% | **−40.0 pp** |

**Latency / tokens / cost:**

| run_id | suite | p50 ms | p95 ms | input tok | output tok | cost |
|---|---|---|---|---|---|---|
| gpt_4o | factuality | 1306 | 2257 | 831 | 615 | $0.0082 |
| gpt_4o | instr_follow | 1004 | 1795 | 882 | 188 | $0.0041 |
| gpt_4o | reasoning | 883 | 1550 | 1000 | 24 | $0.0027 |
| gpt_4o_mini | factuality | 1616 | 3325 | 831 | 612 | $0.0005 |
| gpt_4o_mini | instr_follow | 1023 | 1550 | 882 | 200 | $0.0003 |
| gpt_4o_mini | reasoning | 804 | 906 | 1000 | 41 | $0.0002 |

### What this tells us (honest findings)

- **Reasoning is where model size pays off.** 40-point gap on the 15 arithmetic/algebra prompts — expected, but nice to see the suite caught it clearly.
- **Factuality numbers are surprisingly low for both models (~50%).** The `judge` rubric is strict — it fails a response when it's factually correct but lacks specific phrases from the `expected` answer (e.g. "doesn't mention *internal covariate shift*" on LayerNorm). That's a rubric-design choice, not a model failure; tightening is intentional to make regressions visible.
- **Judge has at least one visible inconsistency** (`fact_15`: verdict `pass=false`, reason "The response accurately describes hallucination and is factually correct"). This is exactly the kind of thing `agreement_vs_human` will surface once manual labels are in — a judge that contradicts itself in free-text is *not a trustworthy grader*, and the framework should flag it.
- **gpt-4o-mini edges out 4o on factuality** (+6.7 pp). Likely a judge-variance artifact on a 15-sample suite rather than a real capability signal; bigger suites would shrink the noise.
- **Cost difference is ~15×** (4o at $0.015/run vs mini at $0.001/run) for a 13-point accuracy lift. Typical mini-vs-full tradeoff — and easy to reason about with this table in front of you.

`agreement_vs_human` is still empty: a follow-up pass will add `human_score` labels to 10+ samples per suite, so the framework can report how often the judge agrees with a human grader. That's the honest metric; `accuracy` without it over-trusts the LLM-as-judge.

## Quick start (local, offline — no OpenAI token)

The repo ships the `runs/` snapshot from the first live eval, so the dashboard works out of the box:

```bash
git clone <repo> && cd llm-eval
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py             # http://localhost:8501 — uses committed runs/
```

To regenerate results yourself (dry-run, mock LLM, no key needed):

```bash
python -m src.runner --dry-run   # mock LLM → runs/raw_*.jsonl
python -m src.judge --dry-run    # mock judge → runs/scores_*.csv
python -m src.report             # runs/summary_*.csv + report_*.md
```

## Live mode

```bash
cp .env.example .env
# edit .env — set OPENROUTER_API_KEY (https://openrouter.ai/settings/keys)
python -m src.runner             # real LLM calls for all configs × suites
python -m src.judge              # LLM-as-judge verdicts
python -m src.report
streamlit run app.py
```

Switching provider: change the `provider:` field in `configs.yaml` (supported: `openrouter`, `openai`, `groq`, `together`) and set the matching `*_API_KEY` in `.env` — no code changes.

Add a new run config: append a block under `runs:` in `configs.yaml`.
Add a new suite: drop `suites/<name>.jsonl` in place — discovered automatically.

## Deploy to HF Spaces

1. Create a Streamlit Space on huggingface.co.
2. Clone the Space repo, copy these files in, push.
3. Under Space Settings → Variables, add secret `OPENROUTER_API_KEY` if you want the dashboard to re-run evals (usually not — commit the `runs/` CSV results you want to display).
4. The Space serves on `<user>-llm-eval.hf.space`.

## What I'd do differently

_Filled at Phase D after shipping. Tentative:_

- Swap judge to a different model family (e.g. Qwen judges Llama outputs) to cut self-judge bias.
- Add Cohen's kappa for judge-vs-human agreement instead of raw match rate.
- Wire a simple regression gate: fail CI if accuracy drops > X points on any suite.
- Persist runs as dated subdirectories (`runs/2026-04-23/…`) instead of overwriting per run_id.
- Pull in `promptfoo` / `lm-eval-harness` for broader benchmark coverage once the custom suite story is solid.

## License

MIT.

---

Portfolio project by Andrey Ovsyannikov — [github.com/ALchemt/llm-eval](https://github.com/ALchemt/llm-eval).
Companion project: [github.com/ALchemt/rag-qa](https://github.com/ALchemt/rag-qa) — RAG Document Q&A.
Part of a 4-project AI portfolio — see parent directory for others.
