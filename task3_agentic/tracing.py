"""Redacted tool JSONL logging and full in-session event tracing."""

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task1_financial.json_utils import json_safe
from task3_agentic.schemas import TraceEvent

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / "logs" / "agent_trace.jsonl"
OUTPUT_LIMIT = 200
_WRITE_LOCK = threading.Lock()
_SECRET_FIELD = re.compile(r"key|token|secret|password|authorization", re.I)


def redact(value: Any) -> Any:
    """Remove secret fields, configured secret values and common token patterns."""
    value = json_safe(value)
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_FIELD.search(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for name, secret in os.environ.items():
            if secret and _SECRET_FIELD.search(name):
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?:gsk_|sk-|hf_)[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        return re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", value)
    return value


def event(role: str, name: str, content: dict[str, Any]) -> TraceEvent:
    return TraceEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        role=role,
        event=name,
        content=redact(content),
    )


class ToolTracer:
    def __init__(self, path: Path = DEFAULT_TRACE_PATH) -> None:
        self.path = Path(path)

    def record(
        self,
        *,
        role: str,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
        duration_ms: float,
        success: bool,
        cache_hit: bool = False,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "tool_name": tool_name,
            "input_arguments": redact(arguments),
            "output": json.dumps(redact(output), ensure_ascii=False, allow_nan=False)[
                :OUTPUT_LIMIT
            ],
            "duration_ms": round(duration_ms, 3),
            "success": success,
            "cache_hit": cache_hit,
        }
        with _WRITE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
                )
