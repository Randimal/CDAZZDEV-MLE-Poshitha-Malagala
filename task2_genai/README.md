# Task 2A — Policy-Grounded Financial Compliance Triage

Prepare an auditable synthetic dataset for a generative assistant that receives an
internal policy excerpt and scenario and returns `category`, `risk_level`,
`recommended_action` and `rationale`. The policies are fictional internal controls,
independent of jurisdiction-specific law. Outputs are not legal advice.
Task 2A contains no model loading, training, fine-tuning or evaluation implementation.

## Models and architecture

The teacher is **openai/gpt-oss-120b via Groq**; `GROQ_MODEL` is configurable, with
that assessment model as the default. The planned student is
**Qwen/Qwen2.5-3B-Instruct**, a distinct model. Its tokenizer/chat template and weights
are reserved for Task 2B; no training dependencies are added now.

| File | Responsibility |
| --- | --- |
| `policies.py` | Fourteen fictional policy topics with three explicit risk/action rules and two equivalent excerpt styles |
| `teacher_system_prompt.txt`, `prompts.py` | Full auditable teacher prompt, batch assignments and student system message |
| `schemas.py` | Strict Pydantic examples, taxonomy consistency and generation settings |
| `client.py` | Mockable Task 2-only Groq transport, safe errors, bounded retries and optional pacing |
| `generation.py` | Balanced assignments, JSON parsing, exact policy/ID matching and per-record rejection audit |
| `dataset.py` | Global deduplication, word-length analysis, seeded splits and chat JSONL export |
| `task2_genai.ipynb` | One executable local/Colab dataset preparation notebook |
| `tests/` | Offline validation, failure, duplicate, leakage, split and formatting tests |

Tasks 1 and 3 are reused neither as mutable dependencies nor as copied business
logic; their implementation remains frozen.

## Generation and validation

Default configuration requests **160 candidates in sixteen batches of ten** rather
than one call per example. Batch sizes 10–20 are supported; use a candidate count
divisible by the batch size to avoid a smaller final batch. Prefer ten on Groq Free.
The plan balances fourteen topics (11–12 assignments each) and rotates low/medium/
high risk. Hints vary anonymous customer types, amount bands, 25–95-word scenarios,
ambiguity and policy style. The teacher receives five recent accepted scenarios
to discourage repeated templates. Target risk is a generation hint, not part of
the student input. Transaction size alone never overrides policy triggers.

The production teacher system prompt is saved in full, displayed in the notebook
and SHA-256 hashed in raw batch provenance. Each assignment supplies its exact
policy excerpt. Safe JSON parsing accepts JSON objects or one complete JSON fence;
it rejects malformed/non-finite JSON without `eval()` or substring extraction.
Each record must pass Pydantic risk/text/length/topic-category checks, preserve its
assigned policy, category and ID, match the assigned risk, and have a unique ID.
Invalid records are rejected without changing their target; valid siblings survive.
Missing records and failed batches are counted, never replaced with local examples.

Provider retries are independent of record validation: at most three requests,
including at most one text-mode JSON fallback for a provider `400/json_validate_failed`.
Only timeout, connection, rate-limit and server failures get transport backoff.
Retry-After is respected; a wait longer than the 120-second retry budget fails the
batch instead of retrying early. Deterministic parse/schema errors are not retried.
The SDK's automatic retries are disabled. Logs contain only safe categories,
numeric status, batch/attempt numbers and counts, excluding raw responses/keys.

Colab enables optional **60-second minimum request spacing**, 4,500 completion tokens
and supported low reasoning effort for GPT-OSS. This limits, but cannot guarantee
avoidance of, account-wide quota failures. Allow roughly 16 minutes plus response/
retry time for the default generation. Higher-tier users can disable pacing; larger
batches may need a larger completion budget and sufficient TPM allowance. Configure
`GROQ_API_KEY` in environment/Colab Secrets; `.env` is ignored and is not auto-loaded.
TLS verification stays enabled.

## Deduplication, diversity and split

Global exact matching normalizes Unicode, case, punctuation and whitespace in
scenario text, even across policies. Near duplicates use standard-library token
`SequenceMatcher` similarity with `autojunk=False`, removing scores **>= 0.88** and
retaining the first example. This threshold is a conservative lexical heuristic,
not a semantic guarantee; removal IDs, similarities and comparison statistics are
saved for review. No embeddings or external technical frameworks are needed.

The notebook displays measured topic/category/risk counts, topic-by-risk coverage,
input and scenario word-length summaries/histograms, and duplicate statistics using
pandas/matplotlib. Word lengths are transparent proxies; Qwen token lengths will be
measured in Task 2B. At least **120 cleaned examples** are required before splitting.
If fewer remain, actual partial raw/clean data is retained and split generation is
blocked. Review failures/duplicates and generate another real run in a new directory;
do not invent replacements or claim the minimum has been achieved.

Use fixed seed **42** and rounded **80/10/10** counts (e.g. 160 -> 128/16/16;
120 -> 96/12/12). Prefer topic+risk stratification when both holdouts can represent
all strata; otherwise topic, then risk, then seeded shuffling. Proportional
largest-remainder allocation meets exact global sizes. Stratification is approximate
when cells are sparse; inspect the recorded strategy and split coverage. Duplicate
IDs/scenarios are rejected at splitting, and near-duplicate filtering precedes it.

Chat JSONL contains `id` metadata and `messages` with system/user/assistant roles.
The user contains only policy and scenario; the assistant contains the four target
fields as JSON. Do not pass metadata to the model. Later apply the student tokenizer's
chat template rather than manually constructing model control tokens.

## Running and artifacts

From the repository root with Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
jupyter notebook task2_genai/task2_genai.ipynb
```

In Colab run the single notebook in order, granting Secrets access. Setup can clone
your actual public repository URL. Configuration can be changed before generation;
leave the teacher as `openai/gpt-oss-120b` for the assessment. Only live generation
produces submission datasets; tests use isolated fixtures and never populate `data/`.

Default output directory: `task2_genai/data/`. Artifacts:

- `raw_teacher_examples.jsonl`: one raw batch envelope per line, including assignments,
  model, prompt hash, exact response and safe error category; checkpointed per batch.
- `cleaned_examples.jsonl`: validated/deduplicated full examples.
- `analysis_summary.json`: configuration, actual validation/rejection and deduplication audit.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: chat-format splits.
- `split_manifest.json`: counts, strata, IDs, seed and SHA-256 artifact hashes.

Completed saved clean data is reused without teacher calls. An interrupted raw-only
run is protected rather than overwritten; inspect its responses before starting a
new run directory. Existing splits are never overwritten. Keep the test set and its
manifest hash **untouched until Task 2C**, and do not use test answers during training,
prompt selection or tuning. Download real artifacts and the executed notebook from
Colab after reviewing their content; no example artifacts or notebook outputs are
prefilled by implementation.

## Limitations and next phases

Structural validation cannot verify semantic grounding, label correctness or the
absence of every form of PII. Review a stratified sample and all flagged records
before training/submission. Lexical similarity misses paraphrases and may remove
legitimate similar cases. Teacher biases and this small taxonomy limit real-world
generality; the dataset is not a production compliance tool or legal guidance.
Quotas and malformed responses may reduce retained size. Verify all topics, risks
and ambiguity/length patterns are meaningfully represented in the actual output.

Task 2B will load the distinct student, establish a baseline and train it; Task 2C
will evaluate on the frozen held-out test set. Neither is implemented. Review the
student's Qwen research license before use; see the official sources in CITATIONS.
