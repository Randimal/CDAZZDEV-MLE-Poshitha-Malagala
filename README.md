# CDAZZDEV-MLE-Poshitha-Malagala

Senior Machine Learning Engineer assessment for CDAZZDEV. Task 1 includes a configurable daily financial data pipeline, numerical indicators, normalized news, deterministic momentum, Groq sentiment/recommendation and a compact research brief.

| Component | Status |
| --- | --- |
| Task 1A — Financial data pipeline | Implemented |
| Task 1B — LLM sentiment and recommendation | Implemented |
| Task 1 bonus — Research brief | Implemented |
| Task 2 | Pending |
| Task 3 | Pending |

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
  json_utils.py                strict JSON conversion
  report.py                    Markdown, styled HTML and PNG chart
  tests/                       deterministic pytest suite
  task1_financial.ipynb         executable local/Colab demonstration
task2_genai/README.md           placeholder
task3_agentic/README.md         placeholder
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
python -m compileall -q shared task1_financial
```

Tests use synthetic prices and mocked Yahoo/Groq clients; they require neither network access nor API keys. Report tests write to temporary directories. Live requests are separate smoke checks, not part of the suite.

Phase 1 validation (2026-10-06): 27 offline tests passed on Python 3.12. Import checks, Python 3.11 syntax parsing, notebook schema and code-cell syntax checks passed. The live NVDA smoke request was attempted but failed TLS certificate verification in this environment. No live results or notebook outputs are claimed; rerun the notebook in Colab to verify current market data.

Phase 2 validation (2026-10-07): all 73 offline tests passed, including the original 27. Import/compile, Python 3.11 syntax, notebook schema/cell syntax, Ruff lint/format and Git whitespace checks passed. Yahoo NVDA smoke still failed TLS certificate verification; Groq smoke was skipped because its key/model configuration was unavailable. No live report artifacts were generated. Report generation was verified using temporary test fixtures only, with no prepopulated notebook outputs.

## Running the notebook

Run `jupyter notebook task1_financial/task1_financial.ipynb` from the root. In Colab, open the notebook from your public GitHub repository. In the setup cell, paste your actual repository URL if it cannot locate an existing checkout; the cell clones into the current runtime directory and installs `requirements.txt`. Run all cells in order. NVDA and a dynamic `2y` period are defaults. Outputs are deliberately unexecuted in the committed notebook.

For Task 1B, set `GROQ_API_KEY` and `GROQ_MODEL` in the runtime environment or Colab Secrets and grant notebook access. The appended setup cell also supports hidden key input. Select a Groq model supporting JSON object mode; no model identifier is hardcoded. `.env.example` shows variable names, but `.env` files are not automatically loaded. There is one LLM request per available headline and at most three recommendation attempts; provider usage limits apply.

Reports generated from actual pipeline results are saved under `task1_financial/outputs/` (ignored by Git). Download the HTML together with its PNG for the relative chart link. If Yahoo data is unavailable no report is generated; unavailable LLM analysis is explicitly marked rather than replaced with invented sentiment or a recommendation.

See [Task 1 documentation](task1_financial/README.md) for formulas and limitations. Review [AI citations](CITATIONS.md) before submission. Reflection headings are intentionally left for the candidate.

## Security and secrets

Task 1A requires no credentials; Task 1B reads Groq configuration from environment variables. Never commit API keys, tokens, `.env` files or environment-specific paths. `.env.example` contains empty credential/model variables. LLM failure logs contain attempt counts only, excluding keys, raw errors and response bodies. TLS verification stays enabled. Do not place secrets in notebook cells or outputs.
