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
