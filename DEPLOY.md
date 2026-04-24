# Deploy Checklist — LLM Eval Framework на HF Spaces

> ~5 минут. Быстрее чем rag-qa: dashboard статичный, читает closed-loop `runs/` результаты из репо, live LLM-вызовов на Space нет.

## Что деплоим

`app.py` — Streamlit dashboard, который читает `runs/raw_*.jsonl` + `runs/scores_*.csv` (они закоммичены) и показывает:
- per-run × suite summary
- accuracy delta между двумя runs
- список промптов где runs разошлись
- raw sample viewer

Никаких LLM-вызовов при рендере — только pandas + Streamlit. Cold start <10с, дёшево.

## 1. HuggingFace аккаунт

Уже есть с rag-qa deploy. Пропустить шаг.

## 2. HF_TOKEN

Тот же что для Project 1, scope **Write**. Если потерял — https://huggingface.co/settings/tokens.

## 3. Создать Space

- https://huggingface.co/new-space
- Name: `llm-eval`
- SDK: **Streamlit** (выбирать из списка; если в UI только Gradio/Static/Docker — выбери **Gradio** или **Blank**, Hugging Face соберёт по `sdk: streamlit` в README YAML)
- Hardware: **CPU basic (free)**
- Visibility: **Public**
- Create.

## 4. Секреты (необязательно)

Dashboard на Space работает без ключей — читает committed results.

Опционально можно добавить `OPENAI_API_KEY` как secret, если захочешь раскомментить "re-run" кнопку в `app.py` (сейчас её нет — deliberately, чтобы посетители Space не жгли твои $).

## 5. Push код

Из `ai-portfolio/llm-eval/`:

```bash
git init
git remote add space https://huggingface.co/spaces/<username>/llm-eval
git add .
git commit -m "Initial deploy: dashboard snapshot of first live eval"
git push space main
```

Если просит auth — логин `<username>`, пароль — HF_TOKEN.

## 6. Ждать билд

- Вкладка **App** → логи.
- Первый билд: **~1 минута** (pandas + streamlit + pyyaml; без torch/chromadb).
- URL: `<username>-llm-eval.hf.space`.

## 7. После успешного деплоя

Скинь URL, добавлю:
1. В `README.md` Project 2 (Demo section — сейчас там placeholder).
2. В `ai-portfolio/README.md` (когда появится).
3. В LinkedIn пост.

## Если сломалось

- **"No module named 'openai'"** — лишний импорт из `runner.py` при статичном показе. Не должно всплывать, dashboard `app.py` openai не импортирует; если всё же — допиши try/except вокруг импорта в `app.py`.
- **"No score files found"** — значит `runs/scores_*.csv` не попали в push. Проверь `.gitignore` локально: только `summary_*.csv` и `report_*.md` должны быть в нём.
- **Streamlit deprecation warnings в логах** — не критично, приложение работает.

## Стоимость

$0. Dashboard read-only.

## Что обновлять, когда появятся новые результаты

После очередного live-прогона:
1. `python -m src.runner && python -m src.judge && python -m src.report`
2. Обновлённые `runs/raw_*.jsonl` + `runs/scores_*.csv` → `git add runs/` → commit → `git push space main`.
3. HF пересобирает за минуту, URL тот же.
