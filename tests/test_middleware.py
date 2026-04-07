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

from task_steering import (
    Task,
    TaskMiddleware,
    TaskStatus,
    TaskSteeringMiddleware,
    TaskSteeringState,
)
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
