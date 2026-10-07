# CDAZZDEV-MLE-Poshitha-Malagala

Senior Machine Learning Engineer assessment for CDAZZDEV. Task 1 includes a configurable daily financial data pipeline, numerical indicators, normalized news, deterministic momentum, Groq sentiment/recommendation and a compact research brief.

| Component | Status |
| --- | --- |
| Task 1A — Financial data pipeline | Implemented |
| Task 1B — LLM sentiment and recommendation | Implemented |
| Task 1 bonus — Research brief | Implemented |
| Task 2A — Synthetic compliance dataset | Implemented; live generation pending |
| Task 2B/2C — Student training/evaluation | Pending |
| Task 3 — Agentic workflows | Implemented |

## Architecture

```text
shared/                         configuration and logging
task1_financial/
  data_pipeline.py              ingestion, validation and summary
  indicators.py                pandas/numpy formulas
  momentum.py                  explicit directional votes
  news.py                      defensive provider normalization
  models.py                    dataclasses and domain exceptions
  llm.py / llm_models.py        Groq adapter and Pydantic schemas
  prompts.py                   separate system/user prompts
  sentiment.py                 per-headline calls and aggregation
  recommendation.py            latest indicator evidence and retries
  technical_facts.py           deterministic indicator relationships
  json_utils.py                strict JSON conversion
  report.py                    Markdown, styled HTML and PNG chart
  tests/                       deterministic pytest suite
  task1_financial.ipynb         executable local/Colab demonstration
task2_genai/                    synthetic policies, batched teacher generation, validation and chat splits
task3_agentic/                  LangGraph agents, tools, typed handoffs, memory and tracing
```

## Installation

Python 3.11+ is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running tests

```bash
python -m pytest -q
python -m compileall -q shared task1_financial task2_genai task3_agentic
```

Tests use synthetic prices and mocked Yahoo/Groq clients; they require neither network access nor API keys. Report tests write to temporary directories. Live requests are separate smoke checks, not part of the suite.

Phase 1 validation (2026-10-06): 27 offline tests passed on Python 3.12. Import checks, Python 3.11 syntax parsing, notebook schema and code-cell syntax checks passed. The live NVDA smoke request was attempted but failed TLS certificate verification in this environment. No live results or notebook outputs are claimed; rerun the notebook in Colab to verify current market data.

Phase 2 validation (2026-10-07): all 73 offline tests passed, including the original 27. Import/compile, Python 3.11 syntax, notebook schema/cell syntax, Ruff lint/format and Git whitespace checks passed. Yahoo NVDA smoke still failed TLS certificate verification; Groq smoke was skipped because its key/model configuration was unavailable. No live report artifacts were generated. Report generation was verified using temporary test fixtures only, with no prepopulated notebook outputs.

Phase 3 validation (2026-10-07): all 105 tests passed, including the original 73. Real LangGraph execution with mocked providers demonstrated different tool orders, observe/replan recovery, the mandatory critique, follow-ups without data calls and zero-call persistent cache hits. Import/compile, Python 3.11 syntax, both notebook schemas/cell syntax, lint/format and whitespace checks passed. Live DuckDuckGo search returned five results; Yahoo still failed TLS certificate verification. Groq/agent live smoke was skipped because key/model settings were unavailable. Two actual tool-smoke records were written to the ignored Task 3 JSONL trace; no live agent report or notebook results were fabricated.

## Running the notebook

Run `jupyter notebook task1_financial/task1_financial.ipynb` from the root. In Colab, open the notebook from your public GitHub repository. In the setup cell, paste your actual repository URL if it cannot locate an existing checkout; the cell clones into the current runtime directory and installs `requirements.txt`. Use an updated checkout/fresh runtime and run all cells in order. NVDA and a dynamic `2y` period are defaults. Saved outputs must reflect actual execution; no market or LLM results are fabricated.

For Task 1B, set `GROQ_API_KEY` and `GROQ_MODEL` in the runtime environment or Colab Secrets and grant notebook access. The setup cell also supports hidden key input. Select a Groq model supporting JSON object mode; no model identifier is hardcoded. `.env.example` shows variable names, but `.env` files are not automatically loaded. Every retrieved headline is analyzed, with successful/failed coverage visible. The recommendation combines latest indicators, compact derived relationships and aggregate sentiment; no historical OHLCV table is sent. The notebook demonstrates invalid-output rejection offline and controlled invalid-ticker failure through the production pipeline. Response validation and transient transport retries are bounded; provider usage limits apply.

Reports generated from actual pipeline results are saved under `task1_financial/outputs/` (ignored by Git). Download the HTML together with its PNG for the relative chart link. If Yahoo data is unavailable no report is generated; unavailable LLM analysis is explicitly marked rather than replaced with invented sentiment or a recommendation.

Task 1 final quality validation (2026-10-07): 145 offline tests passed, with three new relationship tests and all 142 existing tests preserved. Task 1 lint/format, imports/compile, Python 3.11 syntax, notebook schema/cell syntax, rejection/robustness demos and Git whitespace checks passed. Production news returned ten real RSS headlines after Yahoo TLS failure. Live price/invalid-ticker requests failed in a controlled manner; Groq credentials were unavailable, so run the notebook in Colab for actual ten-headline sentiment, recommendation and report evidence. No live outputs or report artifacts were fabricated. Task 2 and Task 3 were not modified in this pass.

See [Task 1 documentation](task1_financial/README.md) for formulas and limitations. Review [AI citations](CITATIONS.md) before submission. Reflection headings are intentionally left for the candidate.

## Task 3: agentic research

Open `jupyter notebook task3_agentic/task3_agentic.ipynb` locally or load it from GitHub in Colab. Configure the same Groq environment/Colab Secrets variables. Task 3 uses LangGraph `StateGraph` with validated JSON decisions through the existing Groq adapter. Free search uses DDGS with its DuckDuckGo backend, without a search API key.

The single agent chooses among five tools. The two-agent graph restricts Agent A to price/volatility/sentiment and Agent B to news/search, with a mandatory typed brief → review → clarification → report path. Tool order within stages is selected by the LLM from observations, including failures. Reports contain financial-health and sentiment summaries, exactly three evidence-linked 90-day risks and one quantitatively supported hedge concept.

The notebook exposes sanitized messages/decisions/observations/handoffs, a follow-up without data-tool calls, and a saved first run followed by a new-session cache hit. Reports are cached in `task3_agentic/memory/TICKER_UTC-DATE.json` with separate single/multi slots. Tools append to `task3_agentic/logs/agent_trace.jsonl`, intentionally committed for submission; temporary logs and cache files remain ignored. `use_cache=False` reruns the graph and clears session snapshots. News uses Yahoo first with recent no-key RSS fallback. Identical failed tool calls are blocked; reports require quantitative and qualitative evidence. Groq transient transport failures use bounded backoff without consuming planning steps. Reliability validation: 128 tests passed; a real RSS smoke returned ten NVDA headlines, while live agent completion needs Colab Groq credentials. See [Task 3 documentation](task3_agentic/README.md) for validation steps and limitations.

## Task 2A: policy-grounded compliance dataset

Open [the single Task 2 notebook](task2_genai/task2_genai.ipynb) in Colab or locally.
Task 2A requests 160 fictional internal-policy examples in batches of five from
`openai/gpt-oss-120b` via Groq, resumes missing IDs from raw checkpoints, validates and globally deduplicates them, displays
diversity, and produces seeded 80/10/10 chat splits only after retaining at least
120 clean examples. The planned student is `Qwen/Qwen2.5-3B-Instruct`; no model is
loaded or trained in this phase. Task 2B/2C remain pending.

Use environment variables/Colab Secrets for Groq; default free-tier pacing is
65 seconds between requests with a 2,500-token completion budget. Raw responses, clean records, audit summaries and
protected split manifests are saved under `task2_genai/data/` only from actual
generation. No dataset or notebook outputs are fabricated. Download the real data
and executed notebook after generation and manual grounding review. These fictional
policy outputs are not legal advice. See [Task 2 documentation](task2_genai/README.md)
for generation settings, artifact formats and test-set protection.

## Security and secrets

Task 1A requires no credentials; Task 1B reads Groq configuration from environment variables. Never commit API keys, tokens, `.env` files or environment-specific paths. `.env.example` contains empty credential/model variables. LLM failure logs contain safe error categories and attempt counts, excluding keys, raw errors and response bodies. TLS verification stays enabled. Do not place secrets in notebook cells or outputs.
