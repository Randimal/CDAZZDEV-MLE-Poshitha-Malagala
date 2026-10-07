# Task 1 — Financial Pipeline, LLM Analysis and Research Brief

## Objective and architecture

Fetch at least two years of daily OHLCV for a configurable ticker (NVDA by default), compute technical indicators ourselves, normalize recent news and return a price-derived summary with deterministic momentum. Task 1B adds validated Groq headline sentiment and technical recommendation; the bonus produces a compact research brief. Task 2 remains pending; [Task 3](../task3_agentic/README.md) reuses these services in agentic workflows.

`shared/config.py` holds immutable configuration and named indicator constants. `data_pipeline.py` orchestrates provider calls and validates history. `indicators.py`, `news.py` and `momentum.py` contain independent transformations; `technical_facts.py` describes relationships among the latest computed values. Dataclasses describe the result and news records; domain exceptions distinguish essential data failures. Tests inject a mocked client instead of contacting Yahoo.

Task 3 adds two optional, backward-compatible entry-point settings: `run_pipeline(include_news=False)` omits news for its restricted price tool, and `GroqClient(max_completion_tokens=1800)` permits report-sized output. Existing defaults remain news enabled and 700 completion tokens; indicator formulas and summary behavior are unchanged.

## Indicator formulas

All formulas use adjusted daily Close; OHLC are requested with `auto_adjust=True` explicitly. A trading-day window counts observations, not calendar days. No technical analysis packages are used.

- **SMA50/SMA200:** arithmetic mean of the most recent 50/200 closes, respectively. `min_periods` equals the window, so warm-up rows remain NaN.
- **RSI14:** delta = Close[t] - Close[t-1]; gain = max(delta, 0), loss = max(-delta, 0). Seed average gain and loss with the arithmetic means of the first 14 changes (15 closes). Thereafter average[t] = (13*average[t-1] + observation[t])/14, implemented by `ewm(alpha=1/14, adjust=False)` on the seeded series. RSI = 100 - 100/(1 + average_gain/average_loss). Only gains yields 100; only losses yields 0; no gains and no losses yields 50 by an explicit neutral convention. Missing prices restart the seed rather than imply zero changes.
- **MACD12/26/9:** EMA alpha = 2/(span+1); EMA[t] = alpha*Close[t] + (1-alpha)*EMA[t-1], seeded with the first close (`adjust=False`). The fast and slow EMAs are hidden until 12/26 observations. MACD = fast EMA - slow EMA; signal = EMA9 of defined MACD values; histogram = MACD - signal. Signal first appears at observation 34. This seed differs from software that starts EMAs with a span-length SMA.
- **Bollinger20/2:** middle = SMA20; upper/lower = middle +/- 2*rolling standard deviation. Population standard deviation (`ddof=0`) is intentional. All 20 observations must be present.

Missing numeric values stay NaN; prices are never forward-filled. Indicator helpers retain gaps, reset EMA state at gaps and require fresh rolling windows. Configurable settings are available through `IndicatorConfig`; standard output column names denote the assessment defaults.

## Momentum methodology

Each of five comparisons contributes +1, -1 or 0: Close vs SMA50, SMA50 vs SMA200, RSI vs its mathematical midpoint 50, MACD vs signal, and Close vs Bollinger middle. Equal or unavailable comparisons abstain. Net score >=3 means BULLISH, <=-3 means BEARISH; otherwise NEUTRAL. Three is an absolute majority of five possible votes, requiring agreement despite contrary votes. Reasons and score accompany the final summary signal.

This combines short/long trend, oscillator direction and EMA momentum. Votes are correlated (particularly price comparisons), so the score is a transparent heuristic, not a probability, optimized strategy or trading recommendation. RSI extremes are not automatically treated as reversals without supporting evidence.

## Summary semantics

`current_price` is the latest available adjusted daily close, not an intraday quote. Trailing 52-week extrema are calculated from adjusted High/Low in a calendar 52-week window ending today in the exchange timezone. `pe_ratio` is finite `trailingPE` if available, else null. Calendar YTD return is 100*(latest close / last valid close before January 1 - 1). This handles holidays/weekends without assuming January 1 is a trading day. Missing prior-year baseline or current-year prices yields null. Tests can supply an explicit `as_of` date.

## Running and expected outputs

Install root requirements, run `python -m pytest -q`, then execute `task1_financial.ipynb` locally or in Colab. Programmatic use:

```python
from shared.config import PipelineConfig
from task1_financial.data_pipeline import run_pipeline
from task1_financial.models import PipelineError

try:
    result = run_pipeline(PipelineConfig(ticker="NVDA", period="2y"))
except PipelineError as exc:
    # An application can show a useful message or retry later.
    result = None
```

A successful result contains the OHLCV/indicator DataFrame, normalized news dataclasses, momentum details and the seven-field summary dictionary. The notebook displays recent data, indicators, summary, available headlines, sanity checks and a price/SMA chart. It catches domain errors and does not fabricate outputs.

## Error handling and limitations

Empty/provider-failed history raises `DataFetchError`. Missing columns, invalid dates or insufficient latest valid history raises `InvalidHistoryError`. Dates are sorted and duplicates keep the final observation. Invalid OHLCV observations (nonfinite, nonpositive prices, negative volume or inconsistent bounds) become entirely missing, with a warning. The latest 200 rows must all be valid. A two-year period is requested, but newly listed tickers may have less actual coverage; the strict minimum enforced is sufficient SMA200 history.

Metadata errors degrade to null PE. News uses yfinance first, then supplements shortfalls with no-key Google News RSS search for the ticker/stock over the last seven days. RSS records retain real publisher/date/links; undated or stale records are skipped. Titles are deduplicated by case/normalized whitespace before the RSS quota is applied and across sources, and source/fallback selection is logged. Source failures log safe warnings and an unresolved shortfall remains explicit; no headlines are invented. Search matches still need relevance/source review. The notebook calls this production flow and requests ten headlines, rather than implementing a separate fetcher.

Yahoo data may be delayed, revised, sparse or rate limited, and news may not be exclusively ticker-specific. Adjusted historical prices reflect corporate actions; reported 52-week prices can differ from unadjusted vendor fields. This is a daily analysis demonstration, not production market data or a backtest. No retry/cache system is included. Dependencies have compatible version ranges rather than a fully locked environment. Indicators and momentum must be reviewed by the candidate before submission.

Provider API references: [daily history](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html) and [get_news](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_news.html).

## Phase 1 audit and price-data choice

The Phase 2 audit found no concrete correctness issue requiring a Task 1A behavior change. SMA windows, Wilder's arithmetic RSI seed/recurrence, EMA `adjust=False`/warm-up conventions, population Bollinger deviation and the pre-January-1 YTD baseline align with the documented definitions. Existing deterministic numerical tests remain the audit evidence.

Yahoo is explicitly called with `auto_adjust=True`. OHLC are adjusted for corporate actions, which avoids split discontinuities distorting indicators and keeps price comparisons consistent. Unadjusted Yahoo OHLC would represent historical nominal trading prices and produce different 52-week extremes and returns. The chosen adjusted YTD is an adjusted-price return that reflects dividend adjustments; it is not a strictly unadjusted capital-gain return or an independently reconstructed total-return index. PE still comes from provider metadata and is not recomputed from adjusted prices. Latest Close is a daily observation rather than a live quote. No Phase 1 formula or price behavior was changed.

## Task 1B architecture and schemas

`llm.py` exposes a small mockable `CompletionClient` protocol and a Groq SDK adapter. Both `GROQ_API_KEY` and `GROQ_MODEL` are required environment variables; the notebook can load Colab userdata or hidden key input. No dotenv auto-loader is used. Pick a model supporting Groq JSON object mode; availability is a runtime provider choice. JSON mode does not guarantee the schema, so Pydantic validation remains mandatory. See [Groq structured outputs](https://console.groq.com/docs/structured-outputs) and [chat API](https://console.groq.com/docs/api-reference).

`llm_models.py` defines strict schemas with forbidden extra fields:

- `HeadlineSentiment`: exact supplied headline; positive/negative/neutral; finite confidence in [0,1]; nonempty brief reason (up to 500 characters).
- `TechnicalRecommendation`: BUY/HOLD/SELL with 3–5 complete sentences (up to 900 characters). Sentence validation uses a documented punctuation/whitespace heuristic; it is a structural check, not proof of reasoning quality.

`prompts.py` separates system and JSON user prompts. It requires supplied evidence only, treats headline text as untrusted data and forbids invented facts/crossovers. Recommendation reasoning should combine trend, RSI, MACD and band positioning, weigh conflicting evidence and acknowledge uncertainty. A single observation cannot prove a historical crossover. `json_utils.py` recursively converts pandas/numpy scalars, containers and timestamps; nonfinite or missing values become JSON null and serialization rejects NaN.

## Sentiment aggregation and failure policy

`sentiment.py` submits one logical analysis for every available headline; transient transport retries are bounded inside the Groq client. Parsing accepts plain JSON or a complete JSON fence, rejects malformed JSON/nonstandard numeric constants, requires every field and checks that the returned headline matches. API or validation failures log safe categories/status codes and attempt counts, are counted explicitly and never assigned substitute sentiment. Later headlines still run. The notebook displays all successful rows and coverage/failure counts, explicitly showing whether ten successful real-headline analyses were achieved.

For each successful result: positive contributes +confidence, negative contributes -confidence, neutral contributes 0. The overall score is the sum divided by **all successful results**, including neutral results, excluding failures. Counts include positive, negative, neutral, successful and failed results. Score >= `POSITIVE_THRESHOLD = 0.20` is positive; <= `NEGATIVE_THRESHOLD = -0.20` is negative; the interior is neutral. These symmetric thresholds are an explicit heuristic deadband for weak/canceling evidence, not statistically calibrated limits. With zero successes, score is null and label is **unavailable**, including an empty batch. Coverage/failures accompany the score so missing evidence remains visible.

The retained formula is `sum(polarity * confidence) / successful_count`, not division by total confidence. Low-confidence predictions contribute smaller directional magnitudes, damping their influence. Failed analyses do not enter the denominator and are never treated as neutral; neutral is a successful explicit prediction.

`recommendation.py` supplies current adjusted daily price, SMA50/SMA200, RSI14, MACD/signal/histogram, all three Bollinger bands, deterministic momentum, derived technical facts and optional aggregate sentiment. No historical OHLCV table is sent. The prompt explicitly asks for confirmation and contradiction: price/average ordering and MACD may agree, while RSI extremes or a band stretch may reduce conviction. It allows up to three bounded response-validation attempts. Groq transport independently permits four attempts for rate limits/timeouts/connections, with 1/2/4-second backoff and Retry-After respected. A requested wait over 60 seconds stops gracefully rather than retrying too early. Deterministic provider errors stop immediately; exhausted transport is not retried by the validation layer. SDK retries remain disabled to avoid nested transport retries; requests have a 20-second timeout. Logs contain safe categories/status and allowlisted codes only. Exhaustion returns `None`, never a fabricated HOLD. No sentiment score alone mechanically determines the LLM recommendation. There is no claim of semantic correctness or calibrated confidence.

## Derived technical relationships

`build_technical_facts(latest, momentum_signal=...)` consumes existing computed values, without recomputing indicators or introducing technical-analysis libraries:

- Signed percentage distance from each SMA is `100 * (Close / SMA - 1)`: positive means above, negative below. Nonpositive SMA denominators or invalid prices yield null.
- SMA50 vs SMA200 is above/below/equal. Trend structure is bullish for `Close > SMA50 > SMA200`, bearish for the reverse, and neutral for other fully available orderings. Missing inputs yield null rather than an invented neutral reading.
- RSI regimes: <=30 oversold; (30,45) weak; [45,55] neutral; (55,70) strong; >=70 overbought. 30/70 are conventional extremes; the five-point band around 50 is a documented descriptive policy. These are not reversal forecasts.
- MACD vs signal is above/below/equal; its existing histogram is positive/negative/zero. A latest comparison does not establish a crossover.
- Bollinger position is below_lower, lower_half, middle, upper_half or above_upper. Equality to the middle is explicit; equality to outer bands belongs to its corresponding half. Missing/inconsistently ordered bands yield null.
- The existing deterministic momentum signal is included unchanged. Conflicting relationships remain visible, rather than being collapsed into another trading score.

The notebook displays the exact production facts used by the recommendation. It also includes one offline invalid-enum demonstration using TechnicalRecommendation and catches its ValidationError, plus a real-provider invalid-ticker demonstration using run_pipeline and PipelineError. Neither example relaxes validation or fabricates financial output. Existing tests already numerically verify Wilder's RSI arithmetic seed/recurrence, as well as SMA50/SMA200, RSI rising/falling/flat edge cases, MACD recurrence, Bollinger deviation and missing-state resets; no redundant reference test was added.

## Bonus report and running Phase 2

After obtaining `result` from Task 1A:

```python
from task1_financial.llm import GroqClient
from task1_financial.sentiment import analyze_headlines
from task1_financial.recommendation import get_recommendation
from task1_financial.report import generate_report

client = GroqClient()  # GROQ_API_KEY and GROQ_MODEL must be set.
batch = analyze_headlines(result.ticker, result.news, client)
recommendation = get_recommendation(result, client, batch.aggregate)
artifacts = generate_report(result, batch.aggregate, recommendation)
```

The single Task 1 notebook presents retrieval/indicators, real headlines, summary/momentum, chart, secure Groq setup, all-headline sentiment/coverage, derived facts, validated recommendation, rejection/robustness demos and report generation. Saved outputs must come from actual execution. The report includes Company Snapshot, Technical Outlook, News Sentiment, Recommendation and Risk Disclaimer; ticker, adjusted price, 52-week high/low, PE, YTD, deterministic momentum, recommendation with its validated 3–5 sentences, aggregate coverage and the first three provider headlines are included. Missing PE/analysis is clearly marked unavailable. Company facts beyond the supplied data are not invented.

`report.py` saves timestamped `.md`, styled `.html` and `.png` files under `task1_financial/outputs/`. The Agg chart renderer needs no desktop GUI. The HTML uses A4 print styles and compact content bounds for a one-page brief; exact browser pagination depends on font/rendering/print settings and should be checked before exporting. Download HTML and PNG together. Untrusted headline/reasoning text is escaped; filename ticker characters are sanitized. No sample report is generated when live price data is unavailable. Synthetic report fixtures are confined to pytest temporary directories.

`.env` and generated outputs are ignored. Credentials never enter prompts, artifacts or application failure logs. No TLS verification bypass is used. Remaining limitations include provider model/JSON-mode availability, rate limits, prompt injection and semantic LLM errors despite structural validation, incomplete news coverage, heuristic sentiment thresholds and correlated momentum indicators. Review actual notebook results and report layout before submission.

Final quality-pass validation (2026-10-07): 145 offline tests passed (the actual 142-test repository baseline plus three focused relationship tests). Existing recommendation/RSS tests were strengthened; numerical indicator tests and report architecture were retained. Task 1 import/compile, Python 3.11 syntax, Ruff lint/format, notebook schema/cell syntax and Git whitespace checks passed. The schema-rejection and mocked empty-provider robustness cells executed successfully. Live production news returned ten real RSS-fallback headlines after Yahoo failed TLS verification; live NVDA and invalid-ticker requests produced controlled DataFetchError. Groq settings were unavailable, so ten successful live sentiments, a live recommendation and real report artifacts still require Colab execution. The invalid-ticker live failure does not distinguish an invalid symbol from that TLS outage. Task 2, Task 3 and personal reflection content were unchanged.
