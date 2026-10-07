import json
from datetime import timedelta
from pathlib import Path

import pytest

from task3_agentic.memory import PersistentMemory
from task3_agentic.multi_agent import TwoAgentResearch
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import ResearchRun
from task3_agentic.single_agent import SingleResearchAgent
from task3_agentic.tests.conftest import AS_OF, ScriptedClient
from task3_agentic.tests.test_workflows import multi_responses, single_responses
from task3_agentic.tools import ToolExecutor


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_cache_save_load_and_second_run_avoids_all_calls(
    executor: ToolExecutor, tmp_path: Path, mode: str
) -> None:
    memory = PersistentMemory(tmp_path / "memory")
    responses = single_responses() if mode == "single" else multi_responses(executor)[0]
    client = ScriptedClient(responses)
    workflow = SingleResearchAgent if mode == "single" else TwoAgentResearch
    agent = workflow(AgentRuntime(client, executor), memory)
    first = agent.run(as_of=AS_OF)
    assert first.report and not first.cached
    path = memory.path("NVDA", AS_OF)
    assert path.name == "NVDA_2026-10-07.json" and path.exists()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["workflows"][mode]["report"]["ticker"] == "NVDA"
    calls, decisions = executor.counter, len(client.requests)
    second = agent.run(as_of=AS_OF)
    assert second.cached and second.report == first.report
    assert executor.counter == calls and len(client.requests) == decisions
    assert second.trace[-1].event == "persistent_cache_hit"
    assert memory.load("NVDA", AS_OF + timedelta(days=1), mode) is None
    assert memory.load("NVDA", AS_OF, "other") is None


def test_force_refresh_executes_workflow(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    memory = PersistentMemory(tmp_path)
    client = ScriptedClient(single_responses())
    assert (
        SingleResearchAgent(AgentRuntime(client, executor), memory)
        .run(as_of=AS_OF)
        .report
    )
    # Fresh tool-observation IDs continue across the session.
    responses = single_responses()
    responses[-1]["output"]["top_three_risks"] = [
        {**item, "evidence_ids": ["obs-6"]}
        for item in responses[-1]["output"]["top_three_risks"]
    ]
    responses[-1]["output"]["hedge_strategy_recommendation"]["evidence_ids"] = ["obs-4"]
    refreshed_client = ScriptedClient(responses)
    refreshed = SingleResearchAgent(
        AgentRuntime(refreshed_client, executor), memory
    ).run(as_of=AS_OF, use_cache=False)
    assert (
        refreshed.report
        and not refreshed.cached
        and len(refreshed_client.requests) == 4
    )


def test_corrupt_cache_and_incomplete_run(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    memory = PersistentMemory(tmp_path)
    memory.path("NVDA", AS_OF).write_text('{"bad":', encoding="utf-8")
    assert memory.load("NVDA", AS_OF, "single") is None
    with pytest.raises(ValueError):
        memory.save(ResearchRun(ticker="NVDA", as_of=AS_OF, workflow="single"))
    with pytest.raises(ValueError):
        memory.path("../../NVDA", AS_OF)


def test_cache_preserves_separate_workflow_slots(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    memory = PersistentMemory(tmp_path)
    single_client = ScriptedClient(single_responses())
    single = SingleResearchAgent(AgentRuntime(single_client, executor), memory).run(
        as_of=AS_OF
    )
    new_executor = ToolExecutor(executor.tools, executor.tracer)
    multi_client = ScriptedClient(multi_responses(new_executor)[0])
    multi = TwoAgentResearch(AgentRuntime(multi_client, new_executor), memory).run(
        as_of=AS_OF
    )
    assert single.report and multi.report
    body = json.loads(memory.path("NVDA", AS_OF).read_text(encoding="utf-8"))
    assert set(body["workflows"]) == {"single", "multi"}
    assert memory.load("NVDA", AS_OF, "single").report == single.report
    assert memory.load("NVDA", AS_OF, "multi").report == multi.report


def test_new_date_clears_session_snapshots(executor: ToolExecutor) -> None:
    executor.start_session(AS_OF)
    assert executor.invoke("analyst", "get_price_data", {"ticker": "NVDA"}).success
    executor.start_session(AS_OF + timedelta(days=1))
    second = executor.invoke("analyst", "get_price_data", {"ticker": "NVDA"})
    assert second.success and not second.cache_hit
    assert executor.tools.pipeline.call_count == 2
