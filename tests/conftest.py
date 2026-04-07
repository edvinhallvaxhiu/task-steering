"""Shared fixtures and mock objects for task-steering tests."""

import pytest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

# All tests require langchain >= 1.0.0
pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0 — pip install langchain>=1.0.0",
)

from langchain.messages import ToolMessage
from langchain.tools import tool

from task_steering import Task, TaskMiddleware, TaskSteeringMiddleware


# ── Mock tools ──────────────────────────────────────────────


@tool
def tool_a(items: list[str]) -> str:
    """Tool A — adds items."""
    return f"Added {len(items)} items."


@tool
def tool_b(names: list[str]) -> str:
    """Tool B — processes names."""
    return f"Processed {len(names)} names."


@tool
def tool_c(query: str) -> str:
    """Tool C — searches."""
    return f"Results for: {query}"


@tool
def global_read() -> str:
    """Read current state (global tool)."""
    return "Current state: ..."


# ── Mock request/response objects ───────────────────────────
# Lightweight stand-ins for ModelRequest / ToolCallRequest so
# tests don't depend on their exact constructor signatures.


@dataclass
class MockSystemMessage:
    content: Any

    @property
    def content_blocks(self) -> list:
        if isinstance(self.content, str):
            return [{"type": "text", "text": self.content}]
        if isinstance(self.content, list):
            return list(self.content)
        return []


@dataclass
class MockModelRequest:
    state: dict
    system_message: Any
    tools: list
    runtime: Any = None
    messages: list = field(default_factory=list)

    def override(self, **kwargs):
        return MockModelRequest(
            state=kwargs.get("state", self.state),
            system_message=kwargs.get("system_message", self.system_message),
            tools=kwargs.get("tools", self.tools),
            runtime=kwargs.get("runtime", self.runtime),
            messages=kwargs.get("messages", self.messages),
        )


@dataclass
class MockToolCallRequest:
    tool_call: dict
    state: dict

    def override(self, **kwargs):
        return MockToolCallRequest(
            tool_call=kwargs.get("tool_call", self.tool_call),
            state=kwargs.get("state", self.state),
        )


# ── Reusable task middleware ────────────────────────────────


class RejectCompletionMiddleware(TaskMiddleware):
    """Always rejects task completion."""

    def __init__(self, reason: str = "Not ready yet."):
        super().__init__()
        self.reason = reason

    def validate_completion(self, state) -> str | None:
        return self.reason


class AllowCompletionMiddleware(TaskMiddleware):
    """Always allows task completion."""

    def validate_completion(self, state) -> str | None:
        return None


class ToolGateMiddleware(TaskMiddleware):
    """Blocks a specific tool until a state condition is met."""

    def __init__(self, gate_tool: str, state_key: str, min_value: int):
        super().__init__()
        self.gate_tool = gate_tool
        self.state_key = state_key
        self.min_value = min_value

    def validate_completion(self, state) -> str | None:
        val = state.get(self.state_key, 0)
        if val < self.min_value:
            return f"{self.state_key} is {val}, need >= {self.min_value}."
        return None

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] == self.gate_tool:
            val = request.state.get(self.state_key, 0)
            if val < self.min_value:
                return ToolMessage(
                    content=(
                        f"Cannot use {self.gate_tool}: "
                        f"{self.state_key}={val}, need >= {self.min_value}."
                    ),
                    tool_call_id=request.tool_call["id"],
                )
        return handler(request)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def three_tasks():
    return [
        Task(name="step_1", instruction="Do step 1.", tools=[tool_a]),
        Task(name="step_2", instruction="Do step 2.", tools=[tool_b]),
        Task(name="step_3", instruction="Do step 3.", tools=[tool_c]),
    ]


@pytest.fixture
def middleware(three_tasks):
    return TaskSteeringMiddleware(tasks=three_tasks, global_tools=[global_read])
