"""Explicit opt-in failure injection for a single-session notebook demonstration."""

from typing import Any

from task3_agentic.schemas import Role, ToolObservation
from task3_agentic.tools import ROLE_TOOLS, ToolExecutor


class ControlledFailureExecutor(ToolExecutor):
    """Fail the first selected allowed tool once, using the real dispatcher.

    This demonstration executor is intentionally single-threaded. Ordinary
    ToolExecutor/FinancialTools never inject failures. The LLM chooses both
    the initial action and the alternative after observing the failure.
    """

    injected_tool: str | None = None

    def invoke(
        self, role: Role, name: str, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self.injected_tool is not None or name not in ROLE_TOOLS[role]:
            return super().invoke(role, name, arguments)
        original = getattr(self.tools, name)

        def fail_once(**kwargs: Any) -> dict[str, Any]:
            self.injected_tool = name
            raise RuntimeError(
                "Controlled failure injection for fallback demonstration"
            )

        setattr(self.tools, name, fail_once)
        try:
            return super().invoke(role, name, arguments)
        finally:
            setattr(self.tools, name, original)
