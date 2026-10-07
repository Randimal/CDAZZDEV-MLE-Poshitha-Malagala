"""Single autonomous research agent with an explicit persistent cache guard."""

from datetime import date, datetime, timezone

from task3_agentic.memory import PersistentMemory
from task3_agentic.prompts import QUESTION
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import ResearchReport, ResearchRun
from task3_agentic.tracing import event


class SingleResearchAgent:
    def __init__(
        self, runtime: AgentRuntime, memory: PersistentMemory | None = None
    ) -> None:
        self.runtime = runtime
        self.memory = memory or PersistentMemory()

    def run(
        self,
        query: str | None = None,
        *,
        as_of: date | None = None,
        use_cache: bool = True,
    ) -> ResearchRun:
        ticker = self.runtime.executor.tools.ticker
        as_of = as_of or datetime.now(timezone.utc).date()
        if use_cache:
            cached = self.memory.load(ticker, as_of, "single")
            if cached:
                return cached
        self.runtime.executor.start_session(as_of, refresh=not use_cache)
        result = self.runtime.run(
            query or QUESTION.format(ticker=ticker),
            role="researcher",
            stage="single_research",
            output_schema=ResearchReport,
            as_of=as_of,
        )
        run = ResearchRun(
            ticker=ticker,
            as_of=as_of,
            workflow="single",
            report=result.output,
            observations=result.observations,
            trace=result.trace,
            error=result.error,
        )
        if run.report:
            run.trace.append(
                event(
                    "memory",
                    "persistent_cache_save",
                    {"workflow": "single", "as_of": as_of},
                )
            )
            try:
                self.memory.save(run)
            except OSError:
                run.trace.append(
                    event(
                        "memory",
                        "cache_write_failed",
                        {"reason": "storage unavailable"},
                    )
                )
        return run
