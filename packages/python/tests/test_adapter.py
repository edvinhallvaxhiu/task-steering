"""Tests for AgentMiddlewareAdapter."""

import typing

import pytest
from unittest.mock import MagicMock

pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0",
)

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import ToolMessage
from langchain.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired

from langchain_task_steering import (
    AgentMiddlewareAdapter,
    Task,
    TaskSteeringMiddleware,
)
from tests.conftest import (
    MockModelRequest,
    MockSystemMessage,
    MockToolCallRequest,
    tool_a,
)


# ── Inner middleware stubs ───────────────────────────────


class NoOpMiddleware(AgentMiddleware):
    """Inner middleware that overrides nothing."""

    pass


class ModelInterceptor(AgentMiddleware):
    """Inner middleware that only overrides wrap_model_call."""

    def __init__(self):
        super().__init__()
        self.called = False

    def wrap_model_call(self, request, handler):
        self.called = True
        return handler(request)


class ToolInterceptor(AgentMiddleware):
    """Inner middleware that only overrides wrap_tool_call."""

    def __init__(self):
        super().__init__()
        self.called = False

    def wrap_tool_call(self, request, handler):
        self.called = True
        return handler(request)


class BothInterceptor(AgentMiddleware):
    """Inner middleware that overrides both hooks."""

    def __init__(self):
        super().__init__()
        self.model_called = False
        self.tool_called = False

    def wrap_model_call(self, request, handler):
        self.model_called = True
        return handler(request)

    def wrap_tool_call(self, request, handler):
        self.tool_called = True
        return handler(request)


class AsyncModelInterceptor(AgentMiddleware):
    """Inner middleware that only overrides awrap_model_call."""

    def __init__(self):
        super().__init__()
        self.called = False

    async def awrap_model_call(self, request, handler):
        self.called = True
        return await handler(request)


class AsyncBothInterceptor(AgentMiddleware):
    """Inner middleware that overrides both async hooks."""

    def __init__(self):
        super().__init__()
        self.model_called = False
        self.tool_called = False

    async def awrap_model_call(self, request, handler):
        self.model_called = True
        return await handler(request)

    async def awrap_tool_call(self, request, handler):
        self.tool_called = True
        return await handler(request)


class ToolContributor(AgentMiddleware):
    """Inner middleware that contributes tools."""

    def __init__(self):
        super().__init__()
        self.tools = [tool_a]


class SchemaMiddleware(AgentMiddleware):
    """Inner middleware with a custom state_schema."""

    class MyState(AgentState):
        custom_field: NotRequired[int]

    state_schema = MyState


# ════════════════════════════════════════════════════════════
# Construction
# ════════════════════════════════════════════════════════════


class TestAdapterInit:
    def test_no_op_inner_still_has_hooks_bound(self):
        """Adapter always binds hook methods (they pass through to handler)."""
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())
        # All wrap hooks are bound as instance attributes
        assert hasattr(adapter, "wrap_model_call")
        assert hasattr(adapter, "wrap_tool_call")

    def test_model_only_inner_delegates_model(self):
        inner = ModelInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        handler = MagicMock(return_value="result")
        adapter.wrap_model_call(MagicMock(), handler)
        assert inner.called is True

    def test_tool_only_inner_delegates_tool(self):
        inner = ToolInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        handler = MagicMock(return_value="result")
        adapter.wrap_tool_call(MagicMock(), handler)
        assert inner.called is True

    def test_both_inner_delegates_both(self):
        inner = BothInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        adapter.wrap_model_call(MagicMock(), MagicMock(return_value="r"))
        adapter.wrap_tool_call(MagicMock(), MagicMock(return_value="r"))
        assert inner.model_called is True
        assert inner.tool_called is True

    def test_async_inner_delegates_async(self):
        inner = AsyncBothInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        assert hasattr(adapter, "awrap_model_call")
        assert hasattr(adapter, "awrap_tool_call")

    def test_exposes_inner_tools(self):
        adapter = AgentMiddlewareAdapter(ToolContributor())
        names = {t.name for t in adapter.tools}
        assert "tool_a" in names

    def test_no_tools_when_inner_has_none(self):
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())
        assert adapter.tools == []

    def test_forwards_state_schema(self):
        adapter = AgentMiddlewareAdapter(SchemaMiddleware())
        hints = typing.get_type_hints(adapter.state_schema, include_extras=True)
        assert "custom_field" in hints

    def test_no_state_schema_when_inner_has_none(self):
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())
        # Should have the default TaskMiddleware schema, not crash
        assert adapter.state_schema is not None


# ════════════════════════════════════════════════════════════
# Sync hook delegation
# ════════════════════════════════════════════════════════════


class TestSyncDelegation:
    def test_no_override_passes_through_model(self):
        """wrap_model_call should call handler directly when inner has no override."""
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())
        handler = MagicMock(return_value="result")
        result = adapter.wrap_model_call(MagicMock(), handler)
        handler.assert_called_once()
        assert result == "result"

    def test_no_override_passes_through_tool(self):
        """wrap_tool_call should call handler directly when inner has no override."""
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())
        handler = MagicMock(return_value="result")
        result = adapter.wrap_tool_call(MagicMock(), handler)
        handler.assert_called_once()
        assert result == "result"

    def test_delegates_model_call(self):
        inner = ModelInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        handler = MagicMock(return_value="result")
        adapter.wrap_model_call(MagicMock(), handler)
        assert inner.called is True
        handler.assert_called_once()

    def test_delegates_tool_call(self):
        inner = ToolInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        handler = MagicMock(return_value="result")
        adapter.wrap_tool_call(MagicMock(), handler)
        assert inner.called is True
        handler.assert_called_once()

    def test_model_only_inner_does_not_crash_on_tool_call(self):
        """If inner only overrides wrap_model_call, wrap_tool_call should not crash."""
        inner = ModelInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        handler = MagicMock(return_value="result")
        result = adapter.wrap_tool_call(MagicMock(), handler)
        assert result == "result"


# ════════════════════════════════════════════════════════════
# Async hook delegation
# ════════════════════════════════════════════════════════════


class TestAsyncDelegation:
    @pytest.mark.asyncio
    async def test_no_override_passes_through_model(self):
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())

        async def handler(r):
            return "result"

        result = await adapter.awrap_model_call(MagicMock(), handler)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_no_override_passes_through_tool(self):
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())

        async def handler(r):
            return "result"

        result = await adapter.awrap_tool_call(MagicMock(), handler)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_delegates_async_model_call(self):
        inner = AsyncModelInterceptor()
        adapter = AgentMiddlewareAdapter(inner)

        async def handler(r):
            return "result"

        await adapter.awrap_model_call(MagicMock(), handler)
        assert inner.called is True

    @pytest.mark.asyncio
    async def test_delegates_async_both(self):
        inner = AsyncBothInterceptor()
        adapter = AgentMiddlewareAdapter(inner)

        async def handler(r):
            return "result"

        await adapter.awrap_model_call(MagicMock(), handler)
        await adapter.awrap_tool_call(MagicMock(), handler)
        assert inner.model_called is True
        assert inner.tool_called is True

    @pytest.mark.asyncio
    async def test_sync_only_inner_passes_through_async(self):
        """If inner only overrides sync hooks, async should pass through."""
        inner = ModelInterceptor()
        adapter = AgentMiddlewareAdapter(inner)

        async def handler(r):
            return "result"

        result = await adapter.awrap_model_call(MagicMock(), handler)
        assert result == "result"
        assert inner.called is False  # sync hook not invoked in async path


# ════════════════════════════════════════════════════════════
# Integration with TaskSteeringMiddleware
# ════════════════════════════════════════════════════════════


class TestAdapterIntegration:
    def test_adapter_tools_are_scoped(self):
        """Adapter's inner tools should be available when the task is active."""

        @tool
        def inner_tool(x: str) -> str:
            """Inner tool."""
            return x

        class ToolMw(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.tools = [inner_tool]

        adapter = AgentMiddlewareAdapter(ToolMw())
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=adapter)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        names = mw._allowed_tool_names(mw._ctx, "a")
        assert "inner_tool" in names
        assert "tool_a" in names

    def test_adapter_state_schema_merged(self):
        """Adapter's forwarded state_schema should be merged into the middleware."""
        adapter = AgentMiddlewareAdapter(SchemaMiddleware())
        tasks = [Task(name="a", instruction="A", tools=[], middleware=adapter)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        hints = typing.get_type_hints(mw.state_schema, include_extras=True)
        assert "custom_field" in hints
        assert "task_statuses" in hints

    def test_adapter_delegates_in_wrap_model_call(self):
        """TaskSteeringMiddleware should delegate to adapter's wrap_model_call."""
        inner = ModelInterceptor()
        adapter = AgentMiddlewareAdapter(inner)
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=adapter)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )

        mw.wrap_model_call(request, lambda r: MagicMock())
        assert inner.called is True

    def test_noop_adapter_does_not_crash_in_pipeline(self):
        """Adapter wrapping a no-op inner should work in the full pipeline."""
        adapter = AgentMiddlewareAdapter(NoOpMiddleware())
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=adapter)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )

        handler = MagicMock()
        mw.wrap_model_call(request, handler)
        handler.assert_called_once()

        tool_request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress"}},
        )

        expected = ToolMessage(content="ok", tool_call_id="call-1")
        handler = MagicMock(return_value=expected)
        result = mw.wrap_tool_call(tool_request, handler)
        handler.assert_called_once()
        assert result == expected
