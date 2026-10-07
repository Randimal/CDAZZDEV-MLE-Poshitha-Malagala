# Task 3 — Agentic Financial Research

## Objective and architecture

Answer the assessment's current financial-health/market-sentiment question with three evidence-supported share-price risks over 90 days and one data-driven hedge concept. Task 2 remains unimplemented.

| Module | Responsibility |
| --- | --- |
| `schemas.py` / `state.py` | Pydantic decisions, evidence-linked metrics, reports and handoffs; TypedDict graph state |
| `tools.py` | Five callable implementations, shared Task 1 services, argument validation and enforced role permissions |
| `runtime.py` | Shared bounded LangGraph decision/tool/observation loop and evidence validation |
| `synthesis.py` | Single-agent final synthesis with a compact digest, local validation and one targeted repair |
| `demonstrations.py` | Opt-in, one-failure executor for the labelled notebook replan demonstration |
| `single_agent.py` | Single researcher with persistent cache guard |
| `multi_agent.py` | Mandatory analyst/review/clarification/writer graph |
| `prompts.py` | Separate agent, tool-description and follow-up instructions |
| `memory.py` | State-only follow-ups and versioned atomic ticker/date JSON persistence |
| `tracing.py` | Redacted full session events and truncated tool JSONL logging |
| `task3_agentic.ipynb` | Executable Colab demonstrations, initially without outputs |

The application uses LangGraph `StateGraph` directly. The single research planner returns a Pydantic ResearchDecision with a concise rationale and either tool arguments or a readiness signal; it never writes a report. A separate final_synthesis node produces and validates ResearchReport. The multi-agent stages retain their existing AgentAction and typed stage-output contracts. The existing frozen Task 1 CompletionClient, safe JSON/fenced-JSON parser and Groq configuration/SDK are reused.

```mermaid
flowchart LR
  Decide -->|LLM selects tool| Tool
  Tool --> Observe
  Observe -->|result or failure| Decide
  Decide -->|LLM signals readiness| Coverage
  Coverage -->|missing quantitative or qualitative evidence| Decide
  Coverage -->|sufficient| FinalSynthesis
  FinalSynthesis -->|valid report or controlled failure| End
```

## Autonomous selection and observe/replan

No tool order is encoded in graph edges. Each decision sees allowed tool schemas, successful/failed observations, available verified headlines, structured handoffs, evidence IDs and remaining budget. Failed tool name/arguments are normalized (defaults, ticker case and JSON ordering) in state; identical failed calls are blocked before dispatch. Changed arguments or another source remain allowed. Invalid planning actions or premature readiness replan; the multi-agent stage contracts retain bounded output correction. Each stage has a default budget of 12 planning responses (configurable 1–30). Groq retries transient errors internally with bounded 1/2/4-second backoff and Retry-After, independent of planning steps. Injected legacy clients have a separate two-failure bound with a pause. SDK retries remain disabled to prevent nested retries. Budget exhaustion returns `report=None` with an explicit error.

For the single researcher, readiness passes the coverage guard to a separate LangGraph `final_synthesis` node. It receives summary/latest indicators, compact news/search/sentiment evidence and preserved IDs; the full report schema is sent here, not in each planner request. One LLM-generated report is parsed safely and validated with Pydantic plus the existing evidence/language checks. Parse/schema/grounding failures get at most ONE targeted repair using the same evidence. This node cannot dispatch tools or return to planning. Groq JSON-object mode stays first; only a provider `json_validate_failed` rejection permits the repair in text mode requesting JSON, with the same mandatory local validation. Other exhausted transport errors stop. No fallback report is fabricated.

Reports require successful price/volatility evidence AND news/search evidence, as well as a quantitative hedge reference. The writer's validated analyst handoff supplies quantitative coverage. Sentiment alone does not replace a news/search observation. Coverage status is supplied to the planner. These are evidence-coverage checks, not a fixed sequence. The three risks must cite known successful observations or permitted handoff IDs. Numeric handoff fields must exactly match retrieved values and canonical paths. Structural/reference checks cannot verify all qualitative LLM claims.

## Five tools and reuse

All agent/notebook invocations go through `ToolExecutor.invoke(role, tool_name, arguments)`. It returns a typed ToolObservation envelope with evidence_id, output, success/error and cache_hit. FinancialTools provides the callable implementations; the dispatcher is the permission/logging boundary.

| Tool | Structured output |
| --- | --- |
| `get_price_data(ticker, period)` | OHLCV/indicators, dated rows, latest values, adjusted-price basis and summary through Task 1 run_pipeline. Period: 2y/5y/10y/max. News is explicitly excluded. |
| `get_news(ticker, n)` | Task 1 title/publisher/published_at/url records. Yahoo primary, recent no-key Google News RSS fallback for shortfalls, deduplicated across sources. Requests at least ten upstream; returns up to n (1–50). Source selection is logged. |
| `calculate_volatility(ticker, window)` | Annualized historical volatility as a fraction, observation count, date and formula; session Task 1 history is reused. |
| `llm_sentiment(headlines)` | Task 1 validated per-headline results and deterministic aggregate, including failures. Only caller-supplied or actually retrieved headline titles are accepted. |
| `web_search(query)` | Up to five title/url/snippet results from free DDGS with backend=duckduckgo; no search API key. |

Volatility uses simple returns `r[t] = Close[t]/Close[t-1]-1`, without filling gaps. For the latest window consecutive valid returns (2–252):

`annualized_volatility = sample_std(returns, ddof=1) * sqrt(252)`.

One additional close is required. 252 is the explicit trading-days-per-year convention. Adjusted Close follows Task 1. A fraction of 0.20 means 20%; this is historical volatility, not a 90-day forecast or loss probability. No options chain is retrieved, so strikes, premiums, deltas and liquidity must not be invented.

## Agent roles and mandatory critique

| Role | Only permitted tools |
| --- | --- |
| Single researcher | All five |
| Agent A — Data Analyst | get_price_data, calculate_volatility, llm_sentiment |
| Agent B — Research Writer | get_news, web_search |

Permissions are enforced before tool lookup or cache access, not only in prompts. B can consume A's quantitative handoff but cannot invoke price/volatility tools. Task 1's backward-compatible include_news=False option avoids hidden news fetching through A's price tool. Sentiment can use seed headlines or headlines retrieved by B and retained as structured records in the trusted registry; A cannot fabricate or directly fetch news.

Every uncached multi-agent run executes:

1. A autonomously gathers allowed evidence and produces a Pydantic QuantitativeBrief.
2. B receives the Pydantic QuantitativeBrief (the DataBrief), reviews it and gathers news/search.
3. B produces one ClarificationRequest: a specific question, one missing requested metric and a material reason. Code rejects a request for an already supplied non-null brief metric or one made before qualitative research. The prompt encourages sentiment analysis of B's actual retrieved headlines when useful and missing; it does not fix the metric or question wording. Only A has llm_sentiment, so this is a complementary handoff rather than redundant data access.
4. A receives the typed request plus previous state, selects allowed tools or existing evidence, and returns a ClarificationResponse for the exact question/metric. Unavailable data is acknowledged.
5. B receives both handoffs and validates against WriterResearchReport: clarification_used is a required non-null/non-empty string copied verbatim from A's answer. Available quantitative clarification must also be cited in the final analysis. Citations alone cannot replace the required field. A field-specific safe rejection feeds the next planner response, with at most three failed writer finish attempts and no transport retry for output validation. Accepted reports use the canonical ResearchReport model for stable cache round trips.

The stage order is fixed because the critique is mandatory. Tool order inside each stage remains autonomous. No manual interruption is needed after the initial query. Structured_handoff, critique_request and clarification_response events expose execution, and positive/negative tests verify actual incorporation.

## Short-term and persistent memory

Live state retains observations, reports and typed handoffs. `answer_followup(run, question, client)` has no reachable tool executor: it asks the LLM to answer from memory only and validates references. It may make an LLM call but cannot refetch Yahoo/volatility/search. The notebook asserts unchanged tool counters and JSONL byte size.

Full price history remains in session state; planner context contains zero OHLCV rows, retaining summary, latest indicators, row count, date and price basis. News context retains title/publisher/date, search snippets are capped at 400 characters with source hosts, sentiment context retains its aggregate, and full source URLs/provider output remain in live observations. The sentiment title registry omits duplicated metadata and is supplied only to roles allowed sentiment. Persisted observations use the same compact view. Evidence IDs and canonical numeric paths are preserved. Questions outside supplied memory must be treated as unavailable.

Planner JSON uses compact separators. Schema presentation titles/descriptions/defaults are removed while field names, required lists, references, enums and bounds remain. Action/tool contracts remain supplied on each stateless planning call; single-agent report contracts appear only in synthesis. Multi-agent output contracts remain stage-specific. Traces expose prompt_chars as a size diagnostic, not a measured token count. Provider failures expose only safe category, HTTP status and allowlisted error codes; raw response bodies and credentials are excluded.

The report safeguard conservatively rejects unsupported balance-sheet strength, solvency, cash-flow health or earnings-quality conclusions. Current tools do not retrieve audited accounts, so reports should describe market/technical condition and explicitly state those fundamental assessments are unavailable. This is a small sentence-pattern check, not comprehensive NLP moderation or semantic verification.

Historical volatility remains annualized. If scaling to a 90-day one-standard-deviation risk estimate, the tool supplies `horizon_volatility = annualized_volatility * sqrt(90 / 252)`. For 37.34% annualized volatility this is approximately 22.31%. This explicitly assumes **90 trading days** with constant/independent return variance; it is not a calendar-day conversion, forecast or guaranteed loss range. Common numeric 90-day move wording is checked against retrieved volatility, allowing rounding to two decimal percentage points. A hedge remains a concept: optimal strike, premium/cost and execution feasibility require option-chain/implied-volatility data.

Completed runs are atomically saved to `task3_agentic/memory/TICKER_UTC-DATE.json`, with a version and separate single/multi slots. A cached single report cannot bypass the two-agent critique. Schema-invalid, corrupt or incompatible caches are misses; incomplete runs are not saved. Same ticker/date/workflow cache loads occur before any tool/LLM call and emit a persistent_cache_hit event. Old trace events are saved history, not re-executed calls.

The date partitions execution cache, not historical as-of price retrieval. It defaults to UTC today; provider observations retain their dates. Query/model/prompt are not separate cache-key dimensions: this cache is intended for the canonical assessment question. Use follow-ups for additional questions or use_cache=False to rerun; this also clears provider/tool session snapshots. A new UTC cache date clears snapshots automatically. Local JSON is not a concurrent multi-process database or tamper-proof evidence archive.

## Observability and security

Every dispatcher invocation, including failures, denied calls and session cache hits, appends to `task3_agentic/logs/agent_trace.jsonl`. Fields include UTC timestamp, role, tool_name, input_arguments, output, duration_ms, success and cache_hit. Output is truncated after redaction to at most 200 characters; the snippet need not be complete JSON.

ResearchRun.trace retains full sanitized model-facing messages, decisions, observations/replans and handoffs. The notebook displays both an event table and expanded JSON. Price observations use the same compact view sent to the model. Decision explanations are recorded, not private internal model reasoning. Tool-log storage failure stops unobservable execution explicitly; ordinary provider failures replan.

Redaction removes secret-named fields, configured secret values, bearer credentials and common token patterns from traces/context/cache. Raw provider exception bodies are not retained by Task 3. API keys use existing SDK environment configuration. `.env`, temporary logs and memory are ignored; the required `logs/agent_trace.jsonl` is intentionally committed. TLS verification stays enabled. Never put arbitrary credentials in questions or source text: unrecognized opaque secrets cannot be guaranteed identifiable.

## Running and verification

The notebook includes a clearly labelled **Controlled failure injection for fallback demonstration**. ControlledFailureExecutor temporarily substitutes a failing dependency for the first allowed tool actually selected; the real dispatcher records one failed observation and the LLM decides its next action. It is an opt-in, single-threaded showcase, not a production outage or fixed replacement plan. Its cache directory is separate from the main workflows. Real external failures may also occur; only actual trace/output can establish live recovery.

The permission demo dispatches A→web_search and B→get_price_data, displaying failed PermissionError observations without provider access. Follow-up output explicitly shows tool_calls_before/after and zero difference. A new NoCallsClient session for the second same-day cache run displays cache_hit=True, new_tool_calls=0, new_llm_calls=0. Main first runs use use_cache=False, which is the refresh equivalent, so cached output cannot replace the first-run demonstrations.

Install root requirements using Python 3.11+. Configure GROQ_API_KEY/GROQ_MODEL securely and open task3_agentic.ipynb locally or in Colab. The notebook uses GroqClient(max_completion_tokens=1800); Task 1's default remains 700.

```python
from task1_financial.llm import GroqClient
from task3_agentic.tools import FinancialTools, ToolExecutor
from task3_agentic.runtime import AgentRuntime
from task3_agentic.multi_agent import TwoAgentResearch
from task3_agentic.memory import answer_followup

client = GroqClient(max_completion_tokens=1800)
runtime = AgentRuntime(client, ToolExecutor(FinancialTools("NVDA", client)))
run = TwoAgentResearch(runtime).run()
if run.report is not None:
    final_report = run.report.model_dump(mode="json")
    followup = answer_followup(run, "What evidence supports the hedge?", client)
```

Run `python -m pytest -q` from the root. Tests mock Groq/Yahoo/search while executing the real LangGraph nodes and critique path. They verify role denials, failure recovery, numerical grounding, mandatory clarification incorporation, session reuse, refresh and persistent caches. Live checks are separate from the suite.

Validation on 2026-10-07: 105 tests passed, preserving all original 73. Import/compile, Python 3.11 syntax, notebook schema/cell syntax, lint/format and whitespace checks passed. Live search returned five results and its JSONL record was inspected; Yahoo failed TLS verification, and Groq/agent live smoke was skipped due to unavailable key/model settings. Autonomous/critique/memory behavior was exercised with mocks, not claimed as a live LLM result.

Reliability validation (2026-10-07): 128 offline tests passed, including all original 105. The live no-key RSS smoke returned ten recent real NVDA headlines with normal TLS verification. Groq credentials were unavailable locally, so a successful live single/two-agent report remains to be verified in Colab.

Final-quality validation (2026-10-07): **165 tests passed**, preserving all prior 154 tests and adding 11 focused regressions. Existing single-agent mock fixtures now explicitly separate readiness from the subsequent synthesis response. Task 1, shared modules and Task 2 have no changes. Task 3 Ruff lint/format, import/compile/Python 3.11 syntax, notebook schema/cell syntax and Git whitespace checks passed. Changed notebook demonstration cells were executed with mocked services without saving outputs. Live connectivity returned ten real headlines and five search results; Yahoo price history failed local TLS verification, which was not bypassed. Groq settings were unavailable, so no successful live agent/notebook run is claimed.

Use the updated checkout and a fresh Colab runtime. Run code cells **2, 4, 6, 8, 9, 11, 13, 15, 17, 19 and 21** in order (numbers include Markdown cells): setup/configuration, five-tool smoke, single research/trace/report, injected failure, multi-agent critique/report, permissions, follow-up, cache hit, and JSONL sample. All 22 cells can also be run with Run all. Save the real executed notebook and updated sanitized trace for submission; the local notebook has no fabricated execution outputs.

In Colab, use the updated checkout and restart the runtime to avoid stale imports. Load Groq settings through userdata/hidden prompts and run all cells. Inspect the real decision → tool → observation → replan → next-action cycle, A's brief, B's critique and A's response. Then inspect the follow-up/cache assertions and JSONL contents. Saved outputs must come from actual execution; do not claim live agent performance from mocked tests alone. If the safe category is rate_limit, wait for quota reset; invalid_request requires checking model/configuration/input rather than repeated immediate attempts.

## Limitations

Yahoo TLS/rate limits, search availability and missing Groq settings can prevent research. JSON object mode is model-dependent and decision budgets can be exhausted. Search snippets are not independently verified documents and may be stale, biased or incomplete. Prompt injection cannot be eliminated by instructions/role restrictions alone. OHLCV/PE/headlines cannot establish comprehensive balance-sheet health; filings/fundamentals need independent review. Historical volatility depends on the window/stationarity assumptions, and confidence/sentiment/hedge language are not calibrated trading signals. Numerical/reference grounding does not validate causal risk statements. Same-day caches can be stale. Review actual sources, traces, hedge feasibility and notebook outputs before submission.

References: [LangGraph StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph), [DDGS API](https://pypi.org/project/ddgs/) and the existing [Groq citations](../CITATIONS.md).
