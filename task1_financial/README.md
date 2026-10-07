# Task 1 — Financial Pipeline, LLM Analysis and Research Brief

## Objective and architecture

Fetch at least two years of daily OHLCV for a configurable ticker (NVDA by default), compute technical indicators ourselves, normalize recent news and return a price-derived summary with deterministic momentum. Task 1B adds validated Groq headline sentiment and technical recommendation; the bonus produces a compact research brief. Tasks 2 and 3 remain pending.

`shared/config.py` holds immutable configuration and named indicator constants. `data_pipeline.py` orchestrates provider calls and validates history. `indicators.py`, `news.py` and `momentum.py` contain independent transformations. Dataclasses describe the result and news records; domain exceptions distinguish essential data failures. Tests inject a mocked client instead of contacting Yahoo.

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

Metadata and news errors log warnings and degrade to null PE/empty news. The parser accepts legacy top-level and modern nested news, skips unusable titles, deduplicates titles and normalizes parseable timestamps. Fewer than ten usable headlines logs a warning. No alternate source or invented headlines are substituted.

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

`sentiment.py` makes exactly one call for each available headline. Parsing accepts plain JSON or a complete JSON fence, rejects malformed JSON/nonstandard numeric constants, requires every field and checks that the returned headline matches. API or validation failures are logged using attempt counts only, counted explicitly and never assigned substitute sentiment. Later headlines still run.

For each successful result: positive contributes +confidence, negative contributes -confidence, neutral contributes 0. The overall score is the sum divided by **all successful results**, including neutral results, excluding failures. Counts include positive, negative, neutral, successful and failed results. Score >= `POSITIVE_THRESHOLD = 0.20` is positive; <= `NEGATIVE_THRESHOLD = -0.20` is negative; the interior is neutral. These symmetric thresholds are an explicit heuristic deadband for weak/canceling evidence, not statistically calibrated limits. With zero successes, score is null and label is **unavailable**, including an empty batch. Coverage/failures accompany the score so missing evidence remains visible.

`recommendation.py` supplies current adjusted daily price, SMA50/SMA200, RSI14, MACD/signal/histogram, all three Bollinger bands, deterministic momentum and optional aggregate sentiment. It allows up to three API/validation attempts. SDK retries are disabled (`max_retries=0`) to keep the total bounded; requests have a 20-second timeout. Exhaustion returns `None`, never a fabricated HOLD. No sentiment score alone mechanically determines the LLM recommendation. There is no claim of semantic correctness or calibrated confidence.

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

The existing notebook adds secure setup, sentiment/result tables, aggregate display, validated recommendation reasoning and report generation. No outputs are prepopulated. The report includes Company Snapshot, Technical Outlook, News Sentiment, Recommendation and Risk Disclaimer; ticker, adjusted price, 52-week high/low, PE, YTD, deterministic momentum, recommendation, aggregate coverage and the first three provider headlines are included. Missing PE/analysis is clearly marked unavailable. Company facts beyond the supplied data are not invented.

`report.py` saves timestamped `.md`, styled `.html` and `.png` files under `task1_financial/outputs/`. The Agg chart renderer needs no desktop GUI. The HTML uses A4 print styles and compact content bounds for a one-page brief; exact browser pagination depends on font/rendering/print settings and should be checked before exporting. Download HTML and PNG together. Untrusted headline/reasoning text is escaped; filename ticker characters are sanitized. No sample report is generated when live price data is unavailable. Synthetic report fixtures are confined to pytest temporary directories.

`.env` and generated outputs are ignored. Credentials never enter prompts, artifacts or application failure logs. No TLS verification bypass is used. Remaining limitations include provider model/JSON-mode availability, rate limits, prompt injection and semantic LLM errors despite structural validation, incomplete news coverage, heuristic sentiment thresholds and correlated momentum indicators. Review actual notebook results and report layout before submission.
