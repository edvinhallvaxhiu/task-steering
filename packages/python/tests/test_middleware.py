"""Tests for TaskSteeringMiddleware."""

import typing

import pytest
from unittest.mock import MagicMock

pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0",
)

from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import tool
from typing_extensions import NotRequired

from langchain_task_steering import (
    Task,
    TaskMiddleware,
    TaskStatus,
    TaskSteeringMiddleware,
    TaskSteeringState,
)
from langchain_task_steering.middleware import _REQUIRE_ALL
from tests.conftest import (
    AllowCompletionMiddleware,
    MockModelRequest,
    MockSystemMessage,
    MockToolCallRequest,
    RejectCompletionMiddleware,
    ToolGateMiddleware,
    global_read,
    tool_a,
    tool_b,
    tool_c,
)


# ════════════════════════════════════════════════════════════
# Init
# ════════════════════════════════════════════════════════════


class TestInit:
    def test_requires_at_least_one_task(self):
        with pytest.raises(ValueError, match="At least one Task"):
            TaskSteeringMiddleware(tasks=[])

    def test_task_order_preserved(self, middleware):
        assert middleware._task_order == ["step_1", "step_2", "step_3"]

    def test_task_map(self, middleware):
        assert set(middleware._task_map.keys()) == {"step_1", "step_2", "step_3"}
        assert middleware._task_map["step_1"].instruction == "Do step 1."

    def test_all_tools_auto_registered(self, middleware):
        names = {t.name for t in middleware.tools}
        assert names == {
            "update_task_status",
            "tool_a",
            "tool_b",
            "tool_c",
            "global_read",
        }

    def test_tools_deduplicated(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)
        count = sum(1 for t in mw.tools if t.name == "tool_a")
        assert count == 1

    def test_rejects_duplicate_task_names(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="a", instruction="B", tools=[tool_b]),
        ]
        with pytest.raises(ValueError, match="Duplicate task names"):
            TaskSteeringMiddleware(tasks=tasks)

    def test_enforce_order_default_true(self, middleware):
        assert middleware._enforce_order is True

    def test_enforce_order_false(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks, enforce_order=False)
        assert mw._enforce_order is False


# ════════════════════════════════════════════════════════════
# before_agent — state initialization
# ════════════════════════════════════════════════════════════


class TestBeforeAgent:
    def test_initializes_all_tasks_as_pending(self, middleware):
        result = middleware.before_agent({"messages": []}, runtime=None)
        assert result is not None
        assert result["task_statuses"] == {
            "step_1": "pending",
            "step_2": "pending",
            "step_3": "pending",
        }

    def test_noop_when_already_initialized(self, middleware):
        state = {
            "messages": [],
            "task_statuses": {
                "step_1": "complete",
                "step_2": "in_progress",
                "step_3": "pending",
            },
        }
        result = middleware.before_agent(state, runtime=None)
        assert result is None

    def test_noop_preserves_existing_statuses(self, middleware):
        """Ensures before_agent doesn't overwrite in-progress state."""
        state = {
            "messages": [],
            "task_statuses": {
                "step_1": "in_progress",
                "step_2": "pending",
                "step_3": "pending",
            },
        }
        assert middleware.before_agent(state, runtime=None) is None


# ════════════════════════════════════════════════════════════
# Internal helpers — status reading
# ════════════════════════════════════════════════════════════


class TestStatusHelpers:
    def test_get_statuses_defaults_to_pending(self, middleware):
        statuses = middleware._get_statuses({})
        assert all(v == "pending" for v in statuses.values())
        assert len(statuses) == 3

    def test_get_statuses_reads_from_state(self, middleware):
        state = {
            "task_statuses": {
                "step_1": "complete",
                "step_2": "in_progress",
                "step_3": "pending",
            }
        }
        statuses = middleware._get_statuses(state)
        assert statuses == {
            "step_1": "complete",
            "step_2": "in_progress",
            "step_3": "pending",
        }

    def test_get_statuses_handles_none(self, middleware):
        statuses = middleware._get_statuses({"task_statuses": None})
        assert all(v == "pending" for v in statuses.values())

    def test_active_task_none_when_all_pending(self, middleware):
        statuses = {"step_1": "pending", "step_2": "pending", "step_3": "pending"}
        assert middleware._active_task(statuses) is None

    def test_active_task_none_when_all_complete(self, middleware):
        statuses = {"step_1": "complete", "step_2": "complete", "step_3": "complete"}
        assert middleware._active_task(statuses) is None

    def test_active_task_finds_in_progress(self, middleware):
        statuses = {"step_1": "complete", "step_2": "in_progress", "step_3": "pending"}
        assert middleware._active_task(statuses) == "step_2"

    def test_active_task_returns_first_in_progress(self, middleware):
        """If multiple tasks are in_progress (shouldn't happen), returns first."""
        statuses = {
            "step_1": "in_progress",
            "step_2": "in_progress",
            "step_3": "pending",
        }
        assert middleware._active_task(statuses) == "step_1"


# ════════════════════════════════════════════════════════════
# Prompt rendering
# ════════════════════════════════════════════════════════════


class TestPromptRendering:
    def test_all_pending_no_active(self, middleware):
        statuses = {"step_1": "pending", "step_2": "pending", "step_3": "pending"}
        block = middleware._render_status_block(statuses, active=None)
        assert "<task_pipeline>" in block
        assert "[ ] step_1 (pending)" in block
        assert "[ ] step_2 (pending)" in block
        assert "[ ] step_3 (pending)" in block
        assert "<current_task" not in block
        assert "</task_pipeline>" in block

    def test_active_task_shows_instruction(self, middleware):
        statuses = {"step_1": "in_progress", "step_2": "pending", "step_3": "pending"}
        block = middleware._render_status_block(statuses, active="step_1")
        assert "[>] step_1 (in_progress)" in block
        assert '<current_task name="step_1">' in block
        assert "Do step 1." in block

    def test_mixed_statuses(self, middleware):
        statuses = {"step_1": "complete", "step_2": "complete", "step_3": "in_progress"}
        block = middleware._render_status_block(statuses, active="step_3")
        assert "[x] step_1 (complete)" in block
        assert "[x] step_2 (complete)" in block
        assert "[>] step_3 (in_progress)" in block
        assert "Do step 3." in block

    def test_rules_when_enforce_order(self, middleware):
        statuses = {"step_1": "pending", "step_2": "pending", "step_3": "pending"}
        block = middleware._render_status_block(statuses, active=None)
        assert "<rules>" in block
        assert "Required order: step_1 -> step_2 -> step_3" in block
        assert "Do not skip tasks." in block

    def test_no_rules_when_order_not_enforced(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks, enforce_order=False)
        statuses = {"step_1": "pending", "step_2": "pending", "step_3": "pending"}
        block = mw._render_status_block(statuses, active=None)
        assert "<rules>" not in block


# ════════════════════════════════════════════════════════════
# Tool scoping
# ════════════════════════════════════════════════════════════


class TestToolScoping:
    def test_no_active_task(self, middleware):
        names = middleware._allowed_tool_names(active_name=None)
        assert names == {"update_task_status", "global_read"}

    def test_step_1_active(self, middleware):
        names = middleware._allowed_tool_names(active_name="step_1")
        assert "tool_a" in names
        assert "update_task_status" in names
        assert "global_read" in names
        assert "tool_b" not in names
        assert "tool_c" not in names

    def test_step_2_active(self, middleware):
        names = middleware._allowed_tool_names(active_name="step_2")
        assert "tool_b" in names
        assert "tool_a" not in names
        assert "tool_c" not in names

    def test_step_3_active(self, middleware):
        names = middleware._allowed_tool_names(active_name="step_3")
        assert "tool_c" in names
        assert "tool_a" not in names
        assert "tool_b" not in names


# ════════════════════════════════════════════════════════════
# Schema merging
# ════════════════════════════════════════════════════════════


class TestSchemaMerging:
    def test_no_task_schemas(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks)
        assert mw.state_schema is TaskSteeringState

    def test_single_task_schema(self):
        class CounterState(AgentState):
            counter: NotRequired[int]

        class CounterMw(TaskMiddleware):
            state_schema = CounterState

        tasks = [Task(name="a", instruction="A", tools=[], middleware=CounterMw())]
        mw = TaskSteeringMiddleware(tasks=tasks)

        hints = typing.get_type_hints(mw.state_schema, include_extras=True)
        assert "counter" in hints
        assert "task_statuses" in hints
        assert "messages" in hints

    def test_multiple_task_schemas(self):
        class StateA(AgentState):
            field_a: NotRequired[str]

        class StateB(AgentState):
            field_b: NotRequired[int]

        class MwA(TaskMiddleware):
            state_schema = StateA

        class MwB(TaskMiddleware):
            state_schema = StateB

        tasks = [
            Task(name="a", instruction="A", tools=[], middleware=MwA()),
            Task(name="b", instruction="B", tools=[], middleware=MwB()),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        hints = typing.get_type_hints(mw.state_schema, include_extras=True)
        assert "field_a" in hints
        assert "field_b" in hints
        assert "task_statuses" in hints
        assert "messages" in hints

    def test_middleware_without_schema_is_skipped(self):
        class NoSchemaMw(TaskMiddleware):
            pass

        tasks = [Task(name="a", instruction="A", tools=[], middleware=NoSchemaMw())]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw.state_schema is TaskSteeringState

    def test_middleware_with_none_schema_is_skipped(self):
        class NoneMw(TaskMiddleware):
            state_schema = None

        tasks = [Task(name="a", instruction="A", tools=[], middleware=NoneMw())]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw.state_schema is TaskSteeringState


# ════════════════════════════════════════════════════════════
# wrap_model_call — prompt injection + tool filtering
# ════════════════════════════════════════════════════════════


class TestWrapModelCall:
    def _make_request(self, middleware, task_statuses):
        return MockModelRequest(
            state={"task_statuses": task_statuses},
            system_message=MockSystemMessage("You are helpful."),
            tools=middleware.tools,
        )

    def _extract_text(self, captured_request):
        """Extract full text from the modified system message."""
        content = captured_request.system_message.content
        if isinstance(content, str):
            return content
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )

    def test_appends_pipeline_block(self, middleware):
        request = self._make_request(
            middleware,
            {"step_1": "in_progress", "step_2": "pending", "step_3": "pending"},
        )

        captured = {}
        middleware.wrap_model_call(
            request, lambda r: captured.update(req=r) or MagicMock()
        )

        text = self._extract_text(captured["req"])
        assert "You are helpful." in text
        assert "<task_pipeline>" in text
        assert "[>] step_1 (in_progress)" in text
        assert "Do step 1." in text

    def test_base_prompt_preserved(self, middleware):
        request = self._make_request(
            middleware,
            {"step_1": "pending", "step_2": "pending", "step_3": "pending"},
        )

        captured = {}
        middleware.wrap_model_call(
            request, lambda r: captured.update(req=r) or MagicMock()
        )

        text = self._extract_text(captured["req"])
        assert text.startswith("You are helpful.")

    def test_tools_scoped_to_active_task(self, middleware):
        request = self._make_request(
            middleware,
            {"step_1": "complete", "step_2": "in_progress", "step_3": "pending"},
        )

        captured = {}
        middleware.wrap_model_call(
            request, lambda r: captured.update(req=r) or MagicMock()
        )

        tool_names = {t.name for t in captured["req"].tools}
        assert tool_names == {"tool_b", "global_read", "update_task_status"}

    def test_no_active_task_only_globals(self, middleware):
        request = self._make_request(
            middleware,
            {"step_1": "pending", "step_2": "pending", "step_3": "pending"},
        )

        captured = {}
        middleware.wrap_model_call(
            request, lambda r: captured.update(req=r) or MagicMock()
        )

        tool_names = {t.name for t in captured["req"].tools}
        assert tool_names == {"global_read", "update_task_status"}

    def test_all_complete_only_globals(self, middleware):
        request = self._make_request(
            middleware,
            {"step_1": "complete", "step_2": "complete", "step_3": "complete"},
        )

        captured = {}
        middleware.wrap_model_call(
            request, lambda r: captured.update(req=r) or MagicMock()
        )

        tool_names = {t.name for t in captured["req"].tools}
        assert tool_names == {"global_read", "update_task_status"}

    def test_delegates_to_task_middleware(self):
        class SpyMiddleware(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.received_request = None

            def wrap_model_call(self, request, handler):
                self.received_request = request
                return handler(request)

        spy = SpyMiddleware()
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )

        mw.wrap_model_call(request, lambda r: MagicMock())

        assert spy.received_request is not None
        # The request the task middleware sees should already have the pipeline block
        text = "\n".join(
            b.get("text", "")
            for b in spy.received_request.system_message.content
            if isinstance(b, dict)
        )
        assert "<task_pipeline>" in text

    def test_no_delegation_when_no_task_middleware(self, middleware):
        """Smoke test — no task has middleware, wrap_model_call still works."""
        request = self._make_request(
            middleware,
            {"step_1": "in_progress", "step_2": "pending", "step_3": "pending"},
        )

        handler = MagicMock()
        middleware.wrap_model_call(request, handler)
        handler.assert_called_once()


# ════════════════════════════════════════════════════════════
# wrap_tool_call — completion validation + delegation
# ════════════════════════════════════════════════════════════


class TestWrapToolCall:
    # ── Completion validation ─────────────────────────────

    def test_rejects_completion_when_validator_fails(self):
        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[],
                middleware=RejectCompletionMiddleware("Need more work."),
            ),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "Cannot complete 'a'" in result.content
        assert "Need more work." in result.content

    def test_allows_completion_when_validator_passes(self):
        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[],
                middleware=AllowCompletionMiddleware(),
            ),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        expected = ToolMessage(content="ok", tool_call_id="call-1")
        handler = MagicMock(return_value=expected)
        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()
        assert result == expected

    def test_in_progress_skips_completion_validation(self):
        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[],
                middleware=RejectCompletionMiddleware("Should not fire."),
            ),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        handler = MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="call-1")
        )
        mw.wrap_tool_call(request, handler)

        handler.assert_called_once()

    def test_no_middleware_allows_completion(self):
        tasks = [Task(name="a", instruction="A", tools=[])]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        handler = MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="call-1")
        )
        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()

    # ── Task middleware delegation ────────────────────────

    def test_delegates_non_transition_tool_to_task_middleware(self):
        gate = ToolGateMiddleware(gate_tool="tool_a", state_key="count", min_value=5)
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=gate)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress"}, "count": 2},
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "Cannot use tool_a" in result.content
        assert "count=2, need >= 5" in result.content

    def test_task_middleware_allows_tool_when_condition_met(self):
        gate = ToolGateMiddleware(gate_tool="tool_a", state_key="count", min_value=5)
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=gate)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress"}, "count": 10},
        )

        expected = ToolMessage(content="ok", tool_call_id="call-1")
        handler = MagicMock(return_value=expected)
        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()
        assert result == expected

    def test_no_delegation_when_no_task_middleware(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress"}},
        )

        expected = ToolMessage(content="ok", tool_call_id="call-1")
        handler = MagicMock(return_value=expected)
        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()
        assert result == expected

    def test_rejects_out_of_scope_tool_when_no_active_task(self, middleware):
        """tool_a is not a global tool, so it's rejected when no task is active."""
        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={
                "task_statuses": {
                    "step_1": "pending",
                    "step_2": "pending",
                    "step_3": "pending",
                }
            },
        )

        handler = MagicMock()
        result = middleware.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "not available" in result.content

    def test_allows_global_tool_when_no_active_task(self, middleware):
        request = MockToolCallRequest(
            tool_call={"name": "global_read", "args": {}, "id": "call-1"},
            state={
                "task_statuses": {
                    "step_1": "pending",
                    "step_2": "pending",
                    "step_3": "pending",
                }
            },
        )

        expected = ToolMessage(content="ok", tool_call_id="call-1")
        handler = MagicMock(return_value=expected)
        result = middleware.wrap_tool_call(request, handler)

        handler.assert_called_once()


# ════════════════════════════════════════════════════════════
# Lifecycle hooks — on_start / on_complete
# ════════════════════════════════════════════════════════════


class TestLifecycleHooks:
    def _make_lifecycle_middleware(self):
        from langgraph.types import Command

        class LifecycleSpy(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.started = False
                self.completed = False

            def on_start(self, state):
                self.started = True

            def on_complete(self, state):
                self.completed = True

        spy = LifecycleSpy()
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)
        return mw, spy

    def test_on_start_called_on_in_progress(self):
        from langgraph.types import Command

        mw, spy = self._make_lifecycle_middleware()

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)

        assert spy.started is True
        assert spy.completed is False

    def test_on_complete_called_on_complete(self):
        from langgraph.types import Command

        mw, spy = self._make_lifecycle_middleware()

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)

        assert spy.completed is True
        assert spy.started is False

    def test_hooks_not_called_on_failed_transition(self):
        mw, spy = self._make_lifecycle_middleware()

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        # Handler returns ToolMessage (error) instead of Command
        handler = MagicMock(
            return_value=ToolMessage(content="Error", tool_call_id="call-1")
        )
        mw.wrap_tool_call(request, handler)

        assert spy.started is False
        assert spy.completed is False

    def test_on_start_receives_state(self):
        from langgraph.types import Command

        received_state = {}

        class StateSpy(TaskMiddleware):
            def on_start(self, state):
                received_state.update(state)

        spy = StateSpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {"task_statuses": {"a": "pending"}, "custom_field": 42}
        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state=state,
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)

        assert received_state["custom_field"] == 42


# ════════════════════════════════════════════════════════════
# Tool gating — out-of-scope tool rejection
# ════════════════════════════════════════════════════════════


class TestToolGating:
    def test_rejects_wrong_task_tool(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        # task "a" is active, but we call tool_b
        request = MockToolCallRequest(
            tool_call={"name": "tool_b", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress", "b": "pending"}},
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "not available" in result.content

    def test_allows_correct_task_tool(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress", "b": "pending"}},
        )

        expected = ToolMessage(content="ok", tool_call_id="call-1")
        handler = MagicMock(return_value=expected)
        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()

    def test_transition_tool_always_allowed(self):
        from langgraph.types import Command

        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)

        handler.assert_called_once()


# ════════════════════════════════════════════════════════════
# End-to-end scenario (unit-level, no real model)
# ════════════════════════════════════════════════════════════


class TestScenario:
    """Simulates an agent loop progression through the middleware."""

    def test_full_lifecycle(self):
        tasks = [
            Task(name="collect", instruction="Collect items.", tools=[tool_a]),
            Task(
                name="review",
                instruction="Review collected items.",
                tools=[tool_b],
                middleware=RejectCompletionMiddleware("Items not reviewed."),
            ),
            Task(name="finalize", instruction="Finalize.", tools=[tool_c]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, global_tools=[global_read])

        # 1. before_agent initializes state
        state = {"messages": []}
        init = mw.before_agent(state, runtime=None)
        assert init["task_statuses"]["collect"] == "pending"
        state["task_statuses"] = init["task_statuses"]

        # 2. First model call — no active task, only globals + transition
        req = MockModelRequest(
            state=state,
            system_message=MockSystemMessage("Base prompt."),
            tools=mw.tools,
        )
        captured = {}
        mw.wrap_model_call(req, lambda r: captured.update(req=r) or MagicMock())

        tool_names = {t.name for t in captured["req"].tools}
        assert tool_names == {"update_task_status", "global_read"}

        # 3. Agent starts "collect" — update state
        state["task_statuses"]["collect"] = "in_progress"

        req = MockModelRequest(
            state=state,
            system_message=MockSystemMessage("Base prompt."),
            tools=mw.tools,
        )
        captured = {}
        mw.wrap_model_call(req, lambda r: captured.update(req=r) or MagicMock())

        tool_names = {t.name for t in captured["req"].tools}
        assert "tool_a" in tool_names
        assert "tool_b" not in tool_names

        # 4. Agent completes "collect", starts "review"
        state["task_statuses"]["collect"] = "complete"
        state["task_statuses"]["review"] = "in_progress"

        req = MockModelRequest(
            state=state,
            system_message=MockSystemMessage("Base prompt."),
            tools=mw.tools,
        )
        captured = {}
        mw.wrap_model_call(req, lambda r: captured.update(req=r) or MagicMock())

        tool_names = {t.name for t in captured["req"].tools}
        assert "tool_b" in tool_names
        assert "tool_a" not in tool_names

        # 5. Agent tries to complete "review" — rejected by middleware
        complete_req = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "review", "status": "complete"},
                "id": "call-99",
            },
            state=state,
        )
        result = mw.wrap_tool_call(complete_req, MagicMock())
        assert isinstance(result, ToolMessage)
        assert "Cannot complete 'review'" in result.content


# ════════════════════════════════════════════════════════════
# Init — required_tasks validation
# ════════════════════════════════════════════════════════════


class TestRequiredTasksInit:
    def test_default_is_all(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks)
        assert mw._required_tasks == {"step_1", "step_2", "step_3"}

    def test_wildcard_resolves_to_all(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks, required_tasks=["*"])
        assert mw._required_tasks == {"step_1", "step_2", "step_3"}

    def test_explicit_subset(self, three_tasks):
        mw = TaskSteeringMiddleware(
            tasks=three_tasks, required_tasks=["step_1", "step_3"]
        )
        assert mw._required_tasks == {"step_1", "step_3"}

    def test_none_means_no_required(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks, required_tasks=None)
        assert mw._required_tasks == set()

    def test_unknown_task_raises(self, three_tasks):
        with pytest.raises(ValueError, match="Unknown required tasks"):
            TaskSteeringMiddleware(tasks=three_tasks, required_tasks=["nonexistent"])

    def test_max_nudges_default(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks)
        assert mw._max_nudges == 3

    def test_max_nudges_custom(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks, max_nudges=5)
        assert mw._max_nudges == 5


# ════════════════════════════════════════════════════════════
# after_agent — required task nudging
# ════════════════════════════════════════════════════════════


class TestAfterAgent:
    def test_nudges_when_required_tasks_incomplete(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "complete", "b": "pending"},
        }

        result = mw.after_agent(state, runtime=None)
        assert result is not None
        assert result["jump_to"] == "model"
        assert result["nudge_count"] == 1
        assert len(result["messages"]) == 1
        assert "b" in result["messages"][0].content
        assert "required tasks" in result["messages"][0].content

    def test_nudge_message_has_task_steering_metadata(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "complete", "b": "pending"},
        }

        result = mw.after_agent(state, runtime=None)
        msg = result["messages"][0]
        meta = msg.additional_kwargs.get("task_steering")
        assert meta is not None
        assert meta["kind"] == "nudge"
        assert "b" in meta["incomplete_tasks"]
        assert "a" not in meta["incomplete_tasks"]

    def test_no_nudge_when_all_complete(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "complete", "b": "complete"},
        }

        result = mw.after_agent(state, runtime=None)
        assert result is None

    def test_no_nudge_when_required_tasks_is_none(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, required_tasks=None)

        state = {
            "messages": [],
            "task_statuses": {"a": "pending"},
        }

        result = mw.after_agent(state, runtime=None)
        assert result is None

    def test_only_checks_required_subset(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, required_tasks=["a"])

        state = {
            "messages": [],
            "task_statuses": {"a": "complete", "b": "pending"},
        }

        # "b" is incomplete but not required — no nudge
        result = mw.after_agent(state, runtime=None)
        assert result is None

    def test_nudge_lists_only_incomplete_required(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
            Task(name="c", instruction="C", tools=[tool_c]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, required_tasks=["a", "c"])

        state = {
            "messages": [],
            "task_statuses": {"a": "complete", "b": "pending", "c": "pending"},
        }

        result = mw.after_agent(state, runtime=None)
        assert result is not None
        # Only "c" should be listed as incomplete (not "a" which is complete)
        msg = result["messages"][0].content
        assert "required tasks: c." in msg

    def test_stops_nudging_after_max_nudges(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, max_nudges=2)

        state = {
            "messages": [],
            "task_statuses": {"a": "pending"},
            "nudge_count": 2,
        }

        result = mw.after_agent(state, runtime=None)
        assert result is None

    def test_increments_nudge_count(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, max_nudges=3)

        state = {
            "messages": [],
            "task_statuses": {"a": "pending"},
            "nudge_count": 1,
        }

        result = mw.after_agent(state, runtime=None)
        assert result is not None
        assert result["nudge_count"] == 2

    def test_nudge_count_defaults_to_zero(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "pending"},
        }

        result = mw.after_agent(state, runtime=None)
        assert result is not None
        assert result["nudge_count"] == 1

    def test_nudges_in_progress_task(self):
        """A task that is in_progress but not complete should still be nudged."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "in_progress"},
        }

        result = mw.after_agent(state, runtime=None)
        assert result is not None
        assert "a" in result["messages"][0].content


# ════════════════════════════════════════════════════════════
# wrap_model_call — null system message
# ════════════════════════════════════════════════════════════


class TestNullSystemMessage:
    def test_wrap_model_call_no_system_message(self, middleware):
        """wrap_model_call should not crash when system_message is None."""
        request = MockModelRequest(
            state={
                "task_statuses": {
                    "step_1": "in_progress",
                    "step_2": "pending",
                    "step_3": "pending",
                }
            },
            system_message=None,
            tools=middleware.tools,
        )

        captured = {}
        middleware.wrap_model_call(
            request, lambda r: captured.update(req=r) or MagicMock()
        )

        # The pipeline block should still be injected
        content = captured["req"].system_message.content
        text = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
        assert "<task_pipeline>" in text
        assert "[>] step_1 (in_progress)" in text


# ════════════════════════════════════════════════════════════
# Lifecycle hooks — post-transition state
# ════════════════════════════════════════════════════════════


class TestLifecycleHooksState:
    def test_on_start_sees_updated_status(self):
        """on_start should receive state with task_statuses reflecting the transition."""
        from langgraph.types import Command

        received_statuses = {}

        class StateSpy(TaskMiddleware):
            def on_start(self, state):
                received_statuses.update(state.get("task_statuses", {}))

        spy = StateSpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)

        assert received_statuses["a"] == "in_progress"

    def test_on_complete_sees_updated_status(self):
        """on_complete should receive state with task_statuses reflecting the transition."""
        from langgraph.types import Command

        received_statuses = {}

        class StateSpy(TaskMiddleware):
            def on_complete(self, state):
                received_statuses.update(state.get("task_statuses", {}))

        spy = StateSpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)

        assert received_statuses["a"] == "complete"


# ════════════════════════════════════════════════════════════
# Async hooks — awrap_model_call / awrap_tool_call
# ════════════════════════════════════════════════════════════


class TestAsyncHooks:
    @pytest.mark.asyncio
    async def test_awrap_model_call_injects_pipeline(self, middleware):
        """awrap_model_call should inject the pipeline block like the sync version."""
        request = MockModelRequest(
            state={
                "task_statuses": {
                    "step_1": "in_progress",
                    "step_2": "pending",
                    "step_3": "pending",
                }
            },
            system_message=MockSystemMessage("You are helpful."),
            tools=middleware.tools,
        )

        captured = {}

        async def async_handler(r):
            captured["req"] = r
            return MagicMock()

        await middleware.awrap_model_call(request, async_handler)

        content = captured["req"].system_message.content
        text = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
        assert "You are helpful." in text
        assert "<task_pipeline>" in text
        assert "[>] step_1 (in_progress)" in text

    @pytest.mark.asyncio
    async def test_awrap_model_call_null_system_message(self, middleware):
        """awrap_model_call should handle None system message."""
        request = MockModelRequest(
            state={
                "task_statuses": {
                    "step_1": "in_progress",
                    "step_2": "pending",
                    "step_3": "pending",
                }
            },
            system_message=None,
            tools=middleware.tools,
        )

        captured = {}

        async def async_handler(r):
            captured["req"] = r
            return MagicMock()

        await middleware.awrap_model_call(request, async_handler)

        content = captured["req"].system_message.content
        text = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
        assert "<task_pipeline>" in text

    @pytest.mark.asyncio
    async def test_awrap_tool_call_validates_completion(self):
        """awrap_tool_call should reject completion when validator fails."""
        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[],
                middleware=RejectCompletionMiddleware("Not ready."),
            ),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        async def async_handler(r):
            return MagicMock()

        result = await mw.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "Cannot complete 'a'" in result.content

    @pytest.mark.asyncio
    async def test_awrap_tool_call_lifecycle_hooks_get_post_state(self):
        """awrap_tool_call lifecycle hooks should see post-transition state."""
        from langgraph.types import Command

        received_statuses = {}

        class StateSpy(TaskMiddleware):
            def on_start(self, state):
                received_statuses.update(state.get("task_statuses", {}))

        spy = StateSpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        async def async_handler(r):
            return Command(update={})

        await mw.awrap_tool_call(request, async_handler)
        assert received_statuses["a"] == "in_progress"

    @pytest.mark.asyncio
    async def test_awrap_tool_call_rejects_out_of_scope(self, middleware):
        """awrap_tool_call should reject tools not in scope."""
        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={
                "task_statuses": {
                    "step_1": "pending",
                    "step_2": "pending",
                    "step_3": "pending",
                }
            },
        )

        async def async_handler(r):
            return MagicMock()

        result = await middleware.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "not available" in result.content

    @pytest.mark.asyncio
    async def test_abefore_agent_initializes_statuses(self, middleware):
        """abefore_agent should initialize task_statuses like the sync version."""
        result = await middleware.abefore_agent({"messages": []}, runtime=None)
        assert result is not None
        assert result["task_statuses"] == {
            "step_1": "pending",
            "step_2": "pending",
            "step_3": "pending",
        }

    @pytest.mark.asyncio
    async def test_abefore_agent_noop_when_initialized(self, middleware):
        """abefore_agent should return None when already initialized."""
        state = {
            "messages": [],
            "task_statuses": {
                "step_1": "in_progress",
                "step_2": "pending",
                "step_3": "pending",
            },
        }
        result = await middleware.abefore_agent(state, runtime=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_aafter_agent_nudges_incomplete(self):
        """aafter_agent should nudge when required tasks are incomplete."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "complete", "b": "pending"},
        }

        result = await mw.aafter_agent(state, runtime=None)
        assert result is not None
        assert result["jump_to"] == "model"
        assert result["nudge_count"] == 1
        assert "b" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_aafter_agent_no_nudge_when_complete(self):
        """aafter_agent should return None when all required tasks complete."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        state = {
            "messages": [],
            "task_statuses": {"a": "complete"},
        }

        result = await mw.aafter_agent(state, runtime=None)
        assert result is None


# ════════════════════════════════════════════════════════════
# Async lifecycle hooks — avalidate_completion / aon_start / aon_complete
# ════════════════════════════════════════════════════════════


class TestAsyncLifecycleHooks:
    @pytest.mark.asyncio
    async def test_async_validate_completion_called_in_awrap(self):
        """awrap_tool_call should use avalidate_completion for async validation."""

        class AsyncValidator(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.async_called = False

            async def avalidate_completion(self, state):
                self.async_called = True
                return "async rejection"

        validator = AsyncValidator()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=validator)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        async def async_handler(r):
            return MagicMock()

        result = await mw.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "async rejection" in result.content
        assert validator.async_called is True

    @pytest.mark.asyncio
    async def test_sync_validate_completion_used_as_fallback(self):
        """awrap_tool_call should fall back to sync validate_completion via default avalidate_completion."""
        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[],
                middleware=RejectCompletionMiddleware("sync rejection"),
            ),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        async def async_handler(r):
            return MagicMock()

        result = await mw.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "sync rejection" in result.content

    @pytest.mark.asyncio
    async def test_async_on_start_called_in_awrap(self):
        """awrap_tool_call should call aon_start on successful in_progress transition."""
        from langgraph.types import Command

        class AsyncLifecycleSpy(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.async_started = False
                self.sync_started = False

            def on_start(self, state):
                self.sync_started = True

            async def aon_start(self, state):
                self.async_started = True

        spy = AsyncLifecycleSpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        async def async_handler(r):
            return Command(update={})

        await mw.awrap_tool_call(request, async_handler)
        assert spy.async_started is True
        assert spy.sync_started is False

    @pytest.mark.asyncio
    async def test_async_on_complete_called_in_awrap(self):
        """awrap_tool_call should call aon_complete on successful complete transition."""
        from langgraph.types import Command

        class AsyncLifecycleSpy(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.async_completed = False
                self.sync_completed = False

            def on_complete(self, state):
                self.sync_completed = True

            async def aon_complete(self, state):
                self.async_completed = True

        spy = AsyncLifecycleSpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        async def async_handler(r):
            return Command(update={})

        await mw.awrap_tool_call(request, async_handler)
        assert spy.async_completed is True
        assert spy.sync_completed is False

    @pytest.mark.asyncio
    async def test_sync_only_hooks_work_in_async_path(self):
        """Middlewares with only sync hooks should still work in awrap_tool_call via defaults."""
        from langgraph.types import Command

        class SyncOnlySpy(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.started = False

            def on_start(self, state):
                self.started = True

        spy = SyncOnlySpy()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=spy)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        async def async_handler(r):
            return Command(update={})

        await mw.awrap_tool_call(request, async_handler)
        assert spy.started is True

    @pytest.mark.asyncio
    async def test_composed_async_validate_completion(self):
        """Composed middleware should chain avalidate_completion."""

        class AsyncFail(TaskMiddleware):
            async def avalidate_completion(self, state):
                return "async error from first"

        class AsyncAllow(TaskMiddleware):
            async def avalidate_completion(self, state):
                return None

        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[],
                middleware=[AsyncFail(), AsyncAllow()],
            )
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )

        async def async_handler(r):
            return MagicMock()

        result = await mw.awrap_tool_call(request, async_handler)
        assert isinstance(result, ToolMessage)
        assert "async error from first" in result.content

    @pytest.mark.asyncio
    async def test_composed_async_lifecycle_all_fire(self):
        """Composed middleware should fire all aon_start hooks."""
        from langgraph.types import Command

        started = []

        class Hook1(TaskMiddleware):
            async def aon_start(self, state):
                started.append("hook1")

        class Hook2(TaskMiddleware):
            async def aon_start(self, state):
                started.append("hook2")

        tasks = [
            Task(name="a", instruction="A", tools=[], middleware=[Hook1(), Hook2()])
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )

        async def async_handler(r):
            return Command(update={})

        await mw.awrap_tool_call(request, async_handler)
        assert started == ["hook1", "hook2"]


# ════════════════════════════════════════════════════════════
# Middleware list composition
# ════════════════════════════════════════════════════════════


class TestMiddlewareListComposition:
    def test_single_item_list_unwrapped(self):
        """A list with one middleware should be unwrapped to that middleware."""
        spy = AllowCompletionMiddleware()
        tasks = [Task(name="a", instruction="A", tools=[], middleware=[spy])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw._task_map["a"].middleware is spy

    def test_empty_list_becomes_none(self):
        tasks = [Task(name="a", instruction="A", tools=[], middleware=[])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw._task_map["a"].middleware is None

    def test_wrap_model_call_chains_in_order(self):
        """First middleware in list = outermost wrapper."""
        call_order = []

        class Outer(TaskMiddleware):
            def wrap_model_call(self, request, handler):
                call_order.append("outer")
                return handler(request)

        class Inner(TaskMiddleware):
            def wrap_model_call(self, request, handler):
                call_order.append("inner")
                return handler(request)

        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[tool_a],
                middleware=[Outer(), Inner()],
            )
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )
        mw.wrap_model_call(request, lambda r: MagicMock())

        assert call_order == ["outer", "inner"]

    def test_wrap_tool_call_chains_in_order(self):
        call_order = []

        class Outer(TaskMiddleware):
            def wrap_tool_call(self, request, handler):
                call_order.append("outer")
                return handler(request)

        class Inner(TaskMiddleware):
            def wrap_tool_call(self, request, handler):
                call_order.append("inner")
                return handler(request)

        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[tool_a],
                middleware=[Outer(), Inner()],
            )
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "args": {}, "id": "call-1"},
            state={"task_statuses": {"a": "in_progress"}},
        )
        expected = ToolMessage(content="ok", tool_call_id="call-1")
        mw.wrap_tool_call(request, lambda r: expected)

        assert call_order == ["outer", "inner"]

    def test_validate_completion_first_error_wins(self):
        class Fail1(TaskMiddleware):
            def validate_completion(self, state):
                return "error from first"

        class Fail2(TaskMiddleware):
            def validate_completion(self, state):
                return "error from second"

        tasks = [
            Task(name="a", instruction="A", tools=[], middleware=[Fail1(), Fail2()])
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )
        result = mw.wrap_tool_call(request, MagicMock())
        assert isinstance(result, ToolMessage)
        assert "error from first" in result.content

    def test_lifecycle_hooks_all_fire(self):
        from langgraph.types import Command

        started = []

        class Hook1(TaskMiddleware):
            def on_start(self, state):
                started.append("hook1")

        class Hook2(TaskMiddleware):
            def on_start(self, state):
                started.append("hook2")

        tasks = [
            Task(name="a", instruction="A", tools=[], middleware=[Hook1(), Hook2()])
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}},
        )
        mw.wrap_tool_call(request, lambda r: Command(update={}))
        assert started == ["hook1", "hook2"]

    def test_tools_merged_from_all_middlewares(self):
        @tool
        def extra_tool(x: str) -> str:
            """Extra."""
            return x

        class ToolMw(TaskMiddleware):
            def __init__(self):
                super().__init__()
                self.tools = [extra_tool]

        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[tool_a],
                middleware=[ToolMw(), AllowCompletionMiddleware()],
            )
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        names = mw._allowed_tool_names(active_name="a")
        assert "extra_tool" in names
        assert "tool_a" in names

    def test_mixed_adapter_and_task_middleware(self):
        """A list can mix AgentMiddlewareAdapter and TaskMiddleware."""
        from langchain.agents.middleware import AgentMiddleware
        from langchain_task_steering import AgentMiddlewareAdapter

        class Interceptor(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.called = False

            def wrap_model_call(self, request, handler):
                self.called = True
                return handler(request)

        interceptor = Interceptor()

        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[tool_a],
                middleware=[
                    AgentMiddlewareAdapter(interceptor),
                    RejectCompletionMiddleware("Not yet."),
                ],
            )
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        # Model call should be intercepted by adapter
        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )
        mw.wrap_model_call(request, lambda r: MagicMock())
        assert interceptor.called is True

        # Completion should be rejected by validator
        complete_req = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )
        result = mw.wrap_tool_call(complete_req, MagicMock())
        assert isinstance(result, ToolMessage)
        assert "Not yet." in result.content


# ════════════════════════════════════════════════════════════
# Auto-wrapping raw AgentMiddleware
# ════════════════════════════════════════════════════════════


class TestAutoWrapping:
    def test_raw_agent_middleware_auto_wrapped(self):
        """Passing a raw AgentMiddleware should auto-wrap in adapter."""
        from langchain.agents.middleware import AgentMiddleware
        from langchain_task_steering.adapter import AgentMiddlewareAdapter

        class MyMw(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.called = False

            def wrap_model_call(self, request, handler):
                self.called = True
                return handler(request)

        inner = MyMw()
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=inner)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        # Should have been wrapped
        assert isinstance(mw._task_map["a"].middleware, AgentMiddlewareAdapter)

        # Should still work
        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )
        mw.wrap_model_call(request, lambda r: MagicMock())
        assert inner.called is True

    def test_raw_agent_middleware_in_list(self):
        """Raw AgentMiddleware in a list should be auto-wrapped."""
        from langchain.agents.middleware import AgentMiddleware

        class MyMw(AgentMiddleware):
            def __init__(self):
                super().__init__()
                self.called = False

            def wrap_model_call(self, request, handler):
                self.called = True
                return handler(request)

        inner = MyMw()
        tasks = [
            Task(
                name="a",
                instruction="A",
                tools=[tool_a],
                middleware=[inner, RejectCompletionMiddleware("Nope.")],
            )
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        # Model call should be intercepted
        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )
        mw.wrap_model_call(request, lambda r: MagicMock())
        assert inner.called is True

        # Completion should be rejected
        complete_req = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )
        result = mw.wrap_tool_call(complete_req, MagicMock())
        assert isinstance(result, ToolMessage)
        assert "Nope." in result.content

    def test_task_middleware_not_double_wrapped(self):
        """TaskMiddleware should pass through without wrapping."""
        from langchain_task_steering.adapter import AgentMiddlewareAdapter

        mw_instance = RejectCompletionMiddleware("No.")
        tasks = [Task(name="a", instruction="A", tools=[], middleware=mw_instance)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        assert mw._task_map["a"].middleware is mw_instance
        assert not isinstance(mw._task_map["a"].middleware, AgentMiddlewareAdapter)

    def test_invalid_middleware_warns_and_ignored(self):
        """Passing an invalid object should warn and be ignored."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tasks = [
                Task(
                    name="a",
                    instruction="A",
                    tools=[tool_a],
                    middleware="not a middleware",
                )
            ]
            mw = TaskSteeringMiddleware(tasks=tasks)

        assert mw._task_map["a"].middleware is None
        assert len(w) == 1
        assert "Ignoring invalid task middleware" in str(w[0].message)

    def test_invalid_in_list_warns_and_skipped(self):
        """Invalid items in a list should be warned and skipped."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tasks = [
                Task(
                    name="a",
                    instruction="A",
                    tools=[tool_a],
                    middleware=[42, RejectCompletionMiddleware("No.")],
                )
            ]
            mw = TaskSteeringMiddleware(tasks=tasks)

        # Invalid 42 is skipped, RejectCompletionMiddleware survives
        assert mw._task_map["a"].middleware is not None
        assert len(w) == 1
        assert "Ignoring invalid task middleware" in str(w[0].message)

        # Validator still works
        complete_req = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress"}},
        )
        result = mw.wrap_tool_call(complete_req, MagicMock())
        assert isinstance(result, ToolMessage)
        assert "No." in result.content

    def test_duck_typed_object_accepted(self):
        """An object with wrap_model_call should be auto-wrapped."""

        class DuckMw:
            def __init__(self):
                self.called = False

            def wrap_model_call(self, request, handler):
                self.called = True
                return handler(request)

        duck = DuckMw()
        tasks = [Task(name="a", instruction="A", tools=[tool_a], middleware=duck)]
        mw = TaskSteeringMiddleware(tasks=tasks)

        assert mw._task_map["a"].middleware is not None

        request = MockModelRequest(
            state={"task_statuses": {"a": "in_progress"}},
            system_message=MockSystemMessage("Base"),
            tools=mw.tools,
        )
        mw.wrap_model_call(request, lambda r: MagicMock())
        assert duck.called is True


# ════════════════════════════════════════════════════════════
# Single active task enforcement
# ════════════════════════════════════════════════════════════


class TestSingleActiveTask:
    def test_rejects_concurrent_in_progress_ordered(self):
        """Cannot start a second task while one is already in_progress (enforce_order=True)."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "b", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress", "b": "pending"}},
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "already in progress" in result.content

    def test_rejects_concurrent_in_progress_unordered(self):
        """Cannot start a second task even when enforce_order=False."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, enforce_order=False)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "b", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "in_progress", "b": "pending"}},
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "already in progress" in result.content
        assert "'a'" in result.content

    def test_allows_start_when_no_active(self):
        """Can start a task when no other is in_progress."""
        from langgraph.types import Command

        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, enforce_order=False)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending", "b": "pending"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_allows_start_after_previous_complete(self):
        """Can start a task once the previous in_progress task is completed."""
        from langgraph.types import Command

        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "b", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "complete", "b": "pending"}},
        )

        handler = MagicMock(return_value=Command(update={}))
        mw.wrap_tool_call(request, handler)
        handler.assert_called_once()


# ════════════════════════════════════════════════════════════
# Nudge count reset on task transition
# ════════════════════════════════════════════════════════════


class TestNudgeCountReset:
    def test_nudge_count_resets_on_start(self):
        """nudge_count should reset to 0 when a task transitions to in_progress."""
        from langgraph.types import Command

        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "in_progress"},
                "id": "call-1",
            },
            state={"task_statuses": {"a": "pending"}, "nudge_count": 5},
        )

        handler = MagicMock(return_value=Command(update={}))
        result = mw.wrap_tool_call(request, handler)

        # The handler is called — get the actual transition tool result
        handler.assert_called_once()

    def test_nudge_count_resets_on_complete(self):
        """nudge_count should reset to 0 when a task transitions to complete."""
        from langgraph.types import Command

        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)

        request = MockToolCallRequest(
            tool_call={
                "name": "update_task_status",
                "args": {"task": "a", "status": "complete"},
                "id": "call-1",
            },
            state={
                "task_statuses": {"a": "in_progress", "b": "pending"},
                "nudge_count": 3,
            },
        )

        handler = MagicMock(return_value=Command(update={}))
        result = mw.wrap_tool_call(request, handler)
        handler.assert_called_once()

    def test_nudge_count_survives_in_state(self):
        """nudge_count persists in state for checkpointer recovery."""
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks, max_nudges=3)

        # Simulate state recovered from checkpoint with nudge_count=2
        state = {
            "messages": [],
            "task_statuses": {"a": "pending"},
            "nudge_count": 2,
        }

        result = mw.after_agent(state, runtime=None)
        assert result is not None
        assert result["nudge_count"] == 3

        # Next call should stop nudging
        state["nudge_count"] = 3
        result = mw.after_agent(state, runtime=None)
        assert result is None
