"""Compact Markdown/HTML equity brief and a real-input matplotlib chart."""

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from task1_financial.llm_models import SentimentAggregate, TechnicalRecommendation
from task1_financial.models import PipelineResult

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DISCLAIMER = (
    "Educational research only; not personalized investment advice. "
    "Adjusted daily prices may be delayed or revised. News coverage is incomplete; "
    "LLM reasoning can be wrong and confidence is not calibrated. "
    "Review original sources and independent evidence before making decisions."
)


@dataclass(frozen=True)
class ReportArtifacts:
    markdown: Path
    html: Path
    chart: Path


def _number(value: object, suffix: str = "") -> str:
    return "Unavailable" if value is None else f"{float(value):,.2f}{suffix}"


def _compact(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _markdown_text(text: str) -> str:
    # Escape untrusted titles/reasoning, including embedded HTML and links.
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", html.escape(text))


def generate_report(
    result: PipelineResult,
    sentiment: SentimentAggregate | None,
    recommendation: TechnicalRecommendation | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ReportArtifacts:
    """Write a print-styled one-page brief; missing analysis says unavailable.

    Callers supply actual pipeline results; no demonstration data is created.
    Test fixtures use a temporary directory, never the default output folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "_", result.ticker)[:40] or "equity"
    stem = f"{safe_ticker}_{stamp}"
    artifacts = ReportArtifacts(
        output_dir / f"{stem}.md",
        output_dir / f"{stem}.html",
        output_dir / f"{stem}.png",
    )
    fig = Figure(figsize=(9, 2.4))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    try:
        for column in ("Close", "SMA_50", "SMA_200"):
            ax.plot(result.data.index, result.data[column], label=column)
        ax.legend(loc="best")
        ax.set(
            title=f"{result.ticker}: adjusted daily prices",
            xlabel="Date",
            ylabel="Quote currency",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(artifacts.chart, dpi=140)
    finally:
        fig.clear()
    summary = result.summary
    snapshot = (
        f"Ticker: {result.ticker} | Latest adjusted close: {_number(summary['current_price'])} | "
        f"52-week high/low: {_number(summary['52_week_high'])} / {_number(summary['52_week_low'])} | "
        f"Trailing PE: {_number(summary['pe_ratio'])} | YTD: {_number(summary['ytd_return_pct'], '%')}"
    )
    outlook = (
        f"Deterministic momentum: {result.momentum.signal} (score {result.momentum.score:+d}/5). "
        + " ".join(result.momentum.reasons)
    )
    news = "Aggregate sentiment unavailable."
    if sentiment is not None:
        news = (
            f"Aggregate: {sentiment.overall_label}; score: {_number(sentiment.overall_score)}. "
            f"Positive/negative/neutral: {sentiment.positive_count}/{sentiment.negative_count}/{sentiment.neutral_count}. "
            f"Successful: {sentiment.successful_count}; failed: {sentiment.failed_count}."
        )
    recommendation_text = (
        "LLM recommendation unavailable; no BUY/HOLD/SELL was substituted."
    )
    if recommendation is not None:
        recommendation_text = f"{recommendation.signal}: {recommendation.reasoning}"
    headlines = [_compact(item.title) for item in result.news[:3]]
    provenance = f"Price observation: {result.data.index[-1].isoformat()}; generated UTC: {stamp}."
    sections = [
        ("Company Snapshot", snapshot),
        ("Technical Outlook", outlook),
        ("News Sentiment", news),
        ("Recommendation", recommendation_text),
        ("Risk Disclaimer", DISCLAIMER),
    ]
    markdown = [
        f"# {_markdown_text(result.ticker)} — Equity Research Brief",
        provenance,
    ]
    html_sections: list[str] = []
    for title, body in sections:
        markdown.extend([f"## {title}", _markdown_text(body)])
        section = f"<section><h2>{title}</h2><p>{html.escape(body)}</p>"
        if title == "Technical Outlook":
            markdown.append(
                f"![Adjusted Close, SMA50 and SMA200]({artifacts.chart.name})"
            )
            section += f'<img src="{artifacts.chart.name}" alt="Adjusted Close, SMA50 and SMA200">'
        if title == "News Sentiment":
            markdown.extend(f"- {_markdown_text(headline)}" for headline in headlines)
            if not headlines:
                markdown.append("No usable headlines available.")
            section += (
                "<ul>"
                + "".join(f"<li>{html.escape(headline)}</li>" for headline in headlines)
                + "</ul>"
            )
            if not headlines:
                section += "<p>No usable headlines available.</p>"
        html_sections.append(section + "</section>")
    styles = """
    @page { size: A4; margin: 12mm; }
    * { box-sizing: border-box; }
    body { font: 13px/1.35 Arial, sans-serif; color: #243247; background: #edf2f7; margin: 0; }
    main { max-width: 800px; margin: 24px auto; padding: 24px; background: white; border-top: 6px solid #176b87; }
    h1 { margin: 0 0 4px; font-size: 24px; color: #14536b; }
    h2 { font-size: 15px; color: #14536b; margin: 12px 0 4px; }
    p { margin: 4px 0; } ul { margin: 4px 0; padding-left: 20px; }
    img { width: 100%; height: 180px; object-fit: contain; }
    small { color: #566478; } section { break-inside: avoid; }
    @media print { body { background: white; font-size: 9pt; } main { margin: 0; padding: 0; max-width: none; }
      h1 { font-size: 18pt; } h2 { margin-top: 8px; font-size: 11pt; } img { height: 145px; } }
    """
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(result.ticker)} Research Brief</title><style>{styles}</style>"
        f"</head><body><main><h1>{html.escape(result.ticker)} — Equity Research Brief</h1>"
        f"<small>{html.escape(provenance)}</small>{''.join(html_sections)}</main></body></html>"
    )
    artifacts.markdown.write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    artifacts.html.write_text(document, encoding="utf-8")
    return artifacts
