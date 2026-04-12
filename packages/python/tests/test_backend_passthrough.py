"""Tests for backend tools passthrough."""

import pytest
from unittest.mock import MagicMock

pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0",
)

from langchain_task_steering import Task, TaskSteeringMiddleware
from tests.conftest import (
    MockModelRequest,
    MockSystemMessage,
    make_mock_tool,
    tool_a,
    tool_b,
    global_read,
)


# ════════════════════════════════════════════════════════════
# Init
# ════════════════════════════════════════════════════════════


class TestBackendToolsInit:
    def test_default_passthrough_disabled(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw._backend_tools_passthrough is False

    def test_default_backend_tools_frozenset(self):
        expected = {
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
            "write_todos",
            "task",
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }
        assert TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS == frozenset(expected)
        assert len(TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS) == 14

    def test_custom_backend_tools_override(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        custom = {"my_tool", "other_tool"}
        mw = TaskSteeringMiddleware(
            tasks=tasks,
            backend_tools_passthrough=True,
            backend_tools=custom,
        )
        assert mw.get_backend_tools() == frozenset(custom)

    def test_passthrough_works(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(
            tasks=tasks,
            backend_tools_passthrough=True,
        )
        assert mw._backend_tools_passthrough is True
        allowed = mw._allowed_tool_names(mw._ctx, "a")
        assert "read_file" in allowed


# ════════════════════════════════════════════════════════════
# Scoping
# ════════════════════════════════════════════════════════════


class TestBackendToolsScoping:
    def _make_middleware(self, passthrough=True, backend_tools=None):
        tasks = [
            Task(name="step_1", instruction="Do 1.", tools=[tool_a]),
            Task(name="step_2", instruction="Do 2.", tools=[tool_b]),
        ]
        return TaskSteeringMiddleware(
            tasks=tasks,
            global_tools=[global_read],
            backend_tools_passthrough=passthrough,
            backend_tools=backend_tools,
        )

    def test_passthrough_adds_tools_to_allowed(self):
        mw = self._make_middleware(passthrough=True)
        allowed = mw._allowed_tool_names(mw._ctx, "step_1")
        assert "read_file" in allowed
        assert "write_file" in allowed
        assert "execute" in allowed
        assert "tool_a" in allowed
        assert "global_read" in allowed

    def test_passthrough_disabled_does_not_add(self):
        mw = self._make_middleware(passthrough=False)
        allowed = mw._allowed_tool_names(mw._ctx, "step_1")
        assert "read_file" not in allowed
        assert "write_file" not in allowed

    def test_passthrough_combines_with_task_tools(self):
        mw = self._make_middleware(passthrough=True)
        allowed = mw._allowed_tool_names(mw._ctx, "step_2")
        assert "tool_b" in allowed
        assert "tool_a" not in allowed  # belongs to step_1
        assert "ls" in allowed
        assert "glob" in allowed

    def test_passthrough_with_no_active_task(self):
        mw = self._make_middleware(passthrough=True)
        allowed = mw._allowed_tool_names(mw._ctx, None)
        assert "read_file" in allowed
        assert "ls" in allowed
        assert "tool_a" not in allowed
        assert "tool_b" not in allowed

    def test_custom_backend_tools_used(self):
        mw = self._make_middleware(
            passthrough=True,
            backend_tools={"custom_tool"},
        )
        allowed = mw._allowed_tool_names(mw._ctx, "step_1")
        assert "custom_tool" in allowed
        assert "read_file" not in allowed  # not in custom set

    def test_wrap_model_call_filters_tools(self):
        mw = self._make_middleware(passthrough=True)
        backend_tools = [make_mock_tool(n) for n in ["read_file", "ls", "write_file"]]
        state = {
            "messages": [],
            "task_statuses": {"step_1": "in_progress", "step_2": "pending"},
        }
        req = MockModelRequest(
            state=state,
            system_message=MockSystemMessage("System"),
            tools=[tool_a, tool_b, global_read, *backend_tools],
        )
        captured = {}
        mw.wrap_model_call(req, lambda r: captured.update(req=r) or MagicMock())
        scoped_names = {t.name for t in captured["req"].tools}
        assert "tool_a" in scoped_names
        assert "global_read" in scoped_names
        assert "read_file" in scoped_names
        assert "ls" in scoped_names
        assert "write_file" in scoped_names
        assert "tool_b" not in scoped_names


# ════════════════════════════════════════════════════════════
# get_backend_tools
# ════════════════════════════════════════════════════════════


class TestGetBackendTools:
    def test_returns_defaults(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw.get_backend_tools() == TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS

    def test_returns_custom(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        custom = {"x", "y"}
        mw = TaskSteeringMiddleware(
            tasks=tasks,
            backend_tools=custom,
        )
        assert mw.get_backend_tools() == frozenset(custom)
