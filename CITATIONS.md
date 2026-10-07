# Citations

## Phase 1 implementation

AI-ASSISTED: OpenAI Codex
Date: 2026-10-06

Purpose:
- repository scaffolding
- Task 1A architecture
- financial data pipeline implementation
- indicator implementation assistance
- unit test generation
- executable notebook generation
- documentation assistance

Prompt:
Implement only Phase 1 of the CDAZZDEV Senior Machine Learning Engineer assessment: a clean Python 3.11+ modular repository, shared configuration/logging, a yfinance daily OHLCV pipeline with at least two years requested and sufficient SMA200 history, pandas/numpy SMA50/SMA200, Wilder RSI14, MACD12/26/9 and Bollinger20/2 formulas, defensive parsing of at least ten requested news headlines, price-derived summary and YTD return, explicit deterministic multi-indicator momentum, robust exceptions/logging, meaningful offline pytest tests, an executable NVDA Colab notebook with displays/checks/chart, documentation and citation. Keep Tasks 1B, 2 and 3 unimplemented, secrets out of code, and reflection headings empty. Execute tests/import checks and attempt a separate live smoke test where possible; never fabricate notebook results.

All AI-generated code must be reviewed, tested and understood by the candidate before submission. This entry attributes implementation assistance to AI and does not claim that the candidate personally authored AI-generated code.

## API references consulted

- [yfinance daily history API](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html): explicit daily history and adjustment parameters.
- [yfinance get_news API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_news.html): count parameter for requesting headlines.

## Phase 2 implementation

AI-ASSISTED: OpenAI Codex
Date: 2026-10-06

Purpose:
- Groq LLM integration
- Pydantic schemas and validation
- prompt engineering and separation
- deterministic sentiment aggregation and failure accounting
- mocked unit tests and validation
- Markdown/HTML research report and chart generation
- extending the existing Task 1 notebook and documentation
- Task 1A formula, YTD baseline and adjusted-price audit

Prompt:
Continue the existing assessment repository and implement only Task 1B and the Task 1 bonus. Use Groq environment configuration and mockable calls; validate every headline sentiment and technical BUY/HOLD/SELL response with Pydantic; isolate headline failures, explicitly count them and never invent fallbacks. Aggregate confidence-weighted direction over successes with documented thresholds. Separate prompts, use supplied financial evidence only, combine indicator interactions and convert pandas/numpy missing values to strict JSON null. Generate a compact Markdown/HTML brief and matplotlib chart from real inputs, append secure Colab-compatible notebook cells without fabricated outputs, add offline tests, audit existing Task 1A behavior without unnecessary refactoring, update documentation, preserve Phase 1 citations, and run validation/smoke tests only when configuration allows. Do not implement Tasks 2 or 3 or weaken TLS verification.

All AI-generated code must be reviewed, tested and understood by the candidate before submission. AI-assisted authorship is disclosed; this does not assert that the candidate personally authored generated code.

References:
- [Groq structured outputs](https://console.groq.com/docs/structured-outputs): JSON object mode and the need for application schema validation.
- [Groq chat API](https://console.groq.com/docs/api-reference): chat messages, model selection and JSON response format.

## Phase 3 implementation

AI-ASSISTED: OpenAI Codex
Date: 2026-10-07

Purpose:
- LangGraph single research agent and autonomous tool routing
- restricted analyst/writer agents and structured handoffs
- mandatory critique request/response loop
- Task 1 price/news/sentiment/Groq reuse and historical volatility
- free web search integration
- session follow-ups, persistent JSON cache and redacted tool observability
- mocked tests, Colab notebook and documentation

Prompt:
Implement only Task 3 of the existing CDAZZDEV assessment. Use LangGraph and reuse Task 1 services. Provide price, news, annualized historical volatility, structured LLM sentiment and free search tools; let the model choose tools based on state/observations rather than a fixed sequence. Produce a validated financial-health/sentiment report with three evidence-supported 90-day risks and one data-driven hedge concept. Enforce analyst and writer tool restrictions, pass a structured quantitative brief, execute one specific critique request and analyst clarification, and require the writer to incorporate the response. Demonstrate state-only follow-ups and ticker/date persistent caching, log every tool invocation with redaction/truncated output/timing/success, add offline mocked tests and an executable notebook with visible traces, update documentation and preserve earlier citations. Leave Task 2 and personal reflection content untouched; never weaken TLS verification or fabricate notebook results.

All AI-generated code must be reviewed, tested and understood by the candidate before submission. Generated implementation assistance is attributed to AI, not represented as the candidate's independent authorship.

References:
- [LangGraph StateGraph API](https://reference.langchain.com/python/langgraph/graph/state/StateGraph): typed state, nodes and conditional routing.
- [DDGS package/API](https://pypi.org/project/ddgs/): free structured search with the DuckDuckGo backend.

## Task 3 live reliability fixes

AI-ASSISTED: OpenAI Codex
Date: 2026-10-07

Purpose: shared Yahoo news shortfall/RSS fallback; normalized failed-tool-call tracking; quantitative/qualitative completion guards; safe Groq error categories and bounded transport backoff; mocked regression tests; notebook/documentation corrections.

Prompt: Fix only the observed Task 3 live reliability failures without redesigning the agent architecture or implementing Task 2. Preserve Yahoo as primary and supplement real news with no-key RSS, block duplicate identical failed calls while retaining autonomous selection, require grounded evidence coverage, handle transient Groq failures inside transport with bounded exponential backoff/Retry-After and safe logging, preserve planning budgets, and update executable Colab demonstrations without fabricated outputs.

All AI-generated code must be reviewed, tested and understood by the candidate before submission.

Reference: [Groq rate-limit headers](https://console.groq.com/docs/rate-limits) and [provider error codes](https://console.groq.com/docs/errors). Google News RSS is a free best-effort search feed, not a guaranteed market-data service.

## Task 3 writer and prompt-size regression fixes

AI-ASSISTED: OpenAI Codex
Date: 2026-10-07

Purpose: explicit writer clarification schema and targeted validation feedback; compact planner evidence/schema payloads; safe provider status/code diagnostics; conservative fundamental-claim and hedge-math safeguards; focused mocked regression tests; existing notebook/documentation updates.

Prompt: Preserve the Task 3 architecture and leave Task 2 untouched. Fix the live writer's null clarification_used loop without relaxing critique incorporation; reduce planner token pressure while retaining evidence IDs and autonomous selection; bound deterministic writer corrections separately from provider retries; reject unsupported financial-health/optimal-option claims and distinguish annualized volatility from 90-trading-day historical risk scaling. Add focused regressions and executable notebook checks without fabricated outputs.

All AI-generated code must be reviewed, tested and understood by the candidate before submission.
