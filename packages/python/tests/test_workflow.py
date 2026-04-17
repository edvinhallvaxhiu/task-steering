"""Tests for workflow mode (dynamic pipeline activation/deactivation)."""

import pytest
from unittest.mock import MagicMock

pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0",
)

from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langgraph.types import Command

from langchain_task_steering import (
    Task,
    TaskMiddleware,
    TaskStatus,
    TaskSteeringMiddleware,
    Workflow,
    WorkflowSteeringMiddleware,
)
from langchain_task_steering.middleware import (
    _ACTIVATE_TOOL_NAME,
    _DEACTIVATE_TOOL_NAME,
    _TRANSITION_TOOL_NAME,
)
from tests.conftest import (
    MockModelRequest,
    MockSystemMessage,
    MockToolCallRequest,
    RejectCompletionMiddleware,
    make_mock_tool,
    tool_a,
    tool_b,
    tool_c,
    global_read,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_runtime(state, tool_call_id="call_123"):
    """Create a minimal mock ToolRuntime."""
    from langchain.tools import ToolRuntime

    rt = ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda x: None,
        tool_call_id=tool_call_id,
        store=None,
    )
    return rt


def _invoke_tool(tool_obj, state, **kwargs):
    """Invoke a @tool function with a ToolRuntime."""
    rt = _make_runtime(state)
    return tool_obj.invoke({**kwargs, "runtime": rt})


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def onboarding_tasks():
    return [
        Task(name="collect_info", instruction="Collect user info.", tools=[tool_a]),
        Task(name="verify", instruction="Verify identity.", tools=[tool_b]),
    ]


@pytest.fixture
def support_tasks():
    return [
        Task(name="diagnose", instruction="Diagnose the issue.", tools=[tool_c]),
    ]


@pytest.fixture
def two_workflows(onboarding_tasks, support_tasks):
    return [
        Workflow(
            name="onboarding",
            description="Onboard a new user",
            tasks=onboarding_tasks,
            global_tools=[global_read],
        ),
        Workflow(
            name="support",
            description="Handle a support request",
            tasks=support_tasks,
        ),
    ]


@pytest.fixture
def wf_middleware(two_workflows):
    return WorkflowSteeringMiddleware(workflows=two_workflows)


# ════════════════════════════════════════════════════════════
# Init validation
# ════════════════════════════════════════════════════════════


class TestWorkflowInit:
    def test_requires_at_least_one_task(self):
        with pytest.raises(ValueError, match="At least one Task"):
            TaskSteeringMiddleware(tasks=[])

    def test_requires_at_least_one_workflow(self):
        with pytest.raises(ValueError, match="At least one Workflow"):
            WorkflowSteeringMiddleware(workflows=[])

    def test_rejects_duplicate_workflow_names(self, onboarding_tasks):
        wf = Workflow(
            name="dup",
            description="A",
            tasks=onboarding_tasks,
        )
        with pytest.raises(ValueError, match="Duplicate workflow names"):
            WorkflowSteeringMiddleware(workflows=[wf, wf])

    def test_rejects_workflow_with_no_tasks(self):
        wf = Workflow(name="empty", description="Empty", tasks=[])
        with pytest.raises(ValueError, match="has no tasks"):
            WorkflowSteeringMiddleware(workflows=[wf])

    def test_rejects_duplicate_task_names_within_workflow(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a]),
            Task(name="a", instruction="B", tools=[tool_b]),
        ]
        wf = Workflow(name="wf", description="WF", tasks=tasks)
        with pytest.raises(ValueError, match="Duplicate task names"):
            WorkflowSteeringMiddleware(workflows=[wf])

    def test_rejects_unknown_required_tasks(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        wf = Workflow(
            name="wf",
            description="WF",
            tasks=tasks,
            required_tasks=["nonexistent"],
        )
        with pytest.raises(ValueError, match="Unknown required tasks"):
            WorkflowSteeringMiddleware(workflows=[wf])

    def test_is_workflow_instance(self, wf_middleware):
        assert isinstance(wf_middleware, WorkflowSteeringMiddleware)

    def test_task_mode_is_separate_class(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks)
        assert isinstance(mw, TaskSteeringMiddleware)
        assert not isinstance(mw, WorkflowSteeringMiddleware)

    def test_all_tools_registered(self, wf_middleware):
        names = {t.name for t in wf_middleware.tools}
        assert _ACTIVATE_TOOL_NAME in names
        assert _DEACTIVATE_TOOL_NAME in names
        assert _TRANSITION_TOOL_NAME in names
        assert "tool_a" in names
        assert "tool_b" in names
        assert "tool_c" in names
        assert "global_read" in names

    def test_tools_deduplicated(self, two_workflows):
        mw = WorkflowSteeringMiddleware(workflows=two_workflows)
        count = sum(1 for t in mw.tools if t.name == "global_read")
        assert count == 1


# ════════════════════════════════════════════════════════════
# before_agent
# ════════════════════════════════════════════════════════════


class TestWorkflowBeforeAgent:
    def test_noop_in_workflow_mode(self, wf_middleware):
        result = wf_middleware.before_agent({"messages": []}, runtime=None)
        assert result is None

    def test_noop_even_without_task_statuses(self, wf_middleware):
        """Workflow mode doesn't init task_statuses — that's done by activate."""
        result = wf_middleware.before_agent(
            {"messages": [], "task_statuses": None},
            runtime=None,
        )
        assert result is None


# ════════════════════════════════════════════════════════════
# activate_workflow tool
# ════════════════════════════════════════════════════════════


class TestActivateWorkflow:
    def test_happy_path(self, wf_middleware):
        state = {"active_workflow": None, "messages": []}
        result = _invoke_tool(
            wf_middleware._activate_tool,
            state,
            workflow="onboarding",
        )
        assert isinstance(result, Command)
        assert result.update["active_workflow"] == "onboarding"
        assert result.update["task_statuses"] == {
            "collect_info": "pending",
            "verify": "pending",
        }

    def test_unknown_workflow(self, wf_middleware):
        state = {"active_workflow": None}
        result = _invoke_tool(
            wf_middleware._activate_tool,
            state,
            workflow="nonexistent",
        )
        assert isinstance(result, str)
        assert "Unknown workflow" in result

    def test_already_active(self, wf_middleware):
        state = {"active_workflow": "support"}
        result = _invoke_tool(
            wf_middleware._activate_tool,
            state,
            workflow="onboarding",
        )
        assert isinstance(result, str)
        assert "already active" in result

    def test_initializes_nudge_count(self, wf_middleware):
        state = {"active_workflow": None, "messages": []}
        result = _invoke_tool(
            wf_middleware._activate_tool,
            state,
            workflow="onboarding",
        )
        assert result.update["nudge_count"] == 0


# ════════════════════════════════════════════════════════════
# deactivate_workflow tool
# ════════════════════════════════════════════════════════════


class TestDeactivateWorkflow:
    def test_happy_path(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "complete",
                "verify": "complete",
            },
        }
        result = _invoke_tool(wf_middleware._deactivate_tool, state)
        assert isinstance(result, Command)
        assert result.update["active_workflow"] is None
        assert result.update["task_statuses"] == {}

    def test_no_active_workflow(self, wf_middleware):
        state = {"active_workflow": None}
        result = _invoke_tool(wf_middleware._deactivate_tool, state)
        assert isinstance(result, str)
        assert "No workflow" in result

    def test_blocked_when_task_in_progress(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "in_progress",
                "verify": "pending",
            },
        }
        result = _invoke_tool(wf_middleware._deactivate_tool, state)
        assert isinstance(result, str)
        assert "Cannot deactivate" in result

    def test_allowed_when_allow_deactivate_in_progress(self, onboarding_tasks):
        wf = Workflow(
            name="flex",
            description="Flexible workflow",
            tasks=onboarding_tasks,
            allow_deactivate_in_progress=True,
        )
        mw = WorkflowSteeringMiddleware(workflows=[wf])
        state = {
            "active_workflow": "flex",
            "task_statuses": {
                "collect_info": "in_progress",
                "verify": "pending",
            },
        }
        result = _invoke_tool(mw._deactivate_tool, state)
        assert isinstance(result, Command)
        assert result.update["active_workflow"] is None


# ════════════════════════════════════════════════════════════
# Workflow transition tool (update_task_status)
# ════════════════════════════════════════════════════════════


class TestWorkflowTransition:
    def test_no_workflow_active(self, wf_middleware):
        state = {"active_workflow": None}
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="in_progress",
        )
        assert isinstance(result, str)
        assert "No workflow is active" in result

    def test_invalid_task_for_workflow(self, wf_middleware):
        state = {
            "active_workflow": "support",
            "task_statuses": {"diagnose": "pending"},
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="in_progress",
        )
        assert isinstance(result, str)
        assert "Invalid task" in result

    def test_happy_path_start(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "pending",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="in_progress",
        )
        assert isinstance(result, Command)
        assert result.update["task_statuses"]["collect_info"] == "in_progress"

    def test_happy_path_complete(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "in_progress",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="complete",
        )
        assert isinstance(result, Command)
        assert result.update["task_statuses"]["collect_info"] == "complete"

    def test_enforce_order(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "pending",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="verify",
            status="in_progress",
        )
        assert isinstance(result, str)
        assert "not complete yet" in result

    def test_no_enforce_order(self, onboarding_tasks):
        wf = Workflow(
            name="flex",
            description="Flexible",
            tasks=onboarding_tasks,
            enforce_order=False,
        )
        mw = WorkflowSteeringMiddleware(workflows=[wf])
        state = {
            "active_workflow": "flex",
            "task_statuses": {
                "collect_info": "pending",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            mw._workflow_transition_tool,
            state,
            task="verify",
            status="in_progress",
        )
        assert isinstance(result, Command)
        assert result.update["task_statuses"]["verify"] == "in_progress"

    def test_invalid_status(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "pending",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="invalid",
        )
        assert isinstance(result, str)
        assert "Invalid status" in result

    def test_already_complete(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "complete",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="complete",
        )
        assert isinstance(result, str)
        assert "already complete" in result

    def test_invalid_transition(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "pending",
                "verify": "pending",
            },
        }
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="complete",
        )
        assert isinstance(result, str)
        assert "Cannot transition" in result


# ════════════════════════════════════════════════════════════
# Tool scoping (wrap_model_call)
# ════════════════════════════════════════════════════════════


class TestWorkflowToolScoping:
    def _make_request(self, state, tools):
        return MockModelRequest(
            state=state,
            system_message=MockSystemMessage("Base prompt."),
            tools=tools,
        )

    def test_no_workflow_transparent(self, wf_middleware):
        """When no workflow active, all non-workflow tools pass through."""
        external_tool = make_mock_tool("external_search")
        all_tools = [*wf_middleware.tools, external_tool]
        request = self._make_request(
            {"active_workflow": None, "messages": []},
            all_tools,
        )
        modified = wf_middleware._build_catalog_request(request)
        active = None
        tool_names = {t.name for t in modified.tools}
        # activate_workflow is injected, external tool passes through
        assert _ACTIVATE_TOOL_NAME in tool_names
        assert "external_search" in tool_names
        # Workflow-specific tools should NOT be present
        assert _DEACTIVATE_TOOL_NAME not in tool_names
        assert _TRANSITION_TOOL_NAME not in tool_names
        assert active is None

    def test_workflow_active_no_task(self, wf_middleware):
        """Workflow active but no task in_progress — global tools + transition + deactivate."""
        request = self._make_request(
            {
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "pending",
                    "verify": "pending",
                },
                "messages": [],
            },
            wf_middleware.tools,
        )
        ctx = wf_middleware._workflow_ctxs["onboarding"]
        modified, active = wf_middleware._prepare_model_request(request, ctx)
        tool_names = {t.name for t in modified.tools}
        assert _TRANSITION_TOOL_NAME in tool_names
        assert _DEACTIVATE_TOOL_NAME in tool_names
        assert "global_read" in tool_names
        # Task-specific tools should NOT be present
        assert "tool_a" not in tool_names
        assert "tool_b" not in tool_names
        # activate should NOT be present when a workflow is active
        assert _ACTIVATE_TOOL_NAME not in tool_names
        assert active is None

    def test_workflow_active_with_task(self, wf_middleware):
        """Workflow active + task in_progress — task tools + global + transition + deactivate."""
        request = self._make_request(
            {
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
                "messages": [],
            },
            wf_middleware.tools,
        )
        ctx = wf_middleware._workflow_ctxs["onboarding"]
        modified, active = wf_middleware._prepare_model_request(request, ctx)
        tool_names = {t.name for t in modified.tools}
        assert _TRANSITION_TOOL_NAME in tool_names
        assert _DEACTIVATE_TOOL_NAME in tool_names
        assert "global_read" in tool_names
        assert "tool_a" in tool_names  # collect_info task's tool
        assert "tool_b" not in tool_names  # verify task's tool
        assert active == "collect_info"


# ════════════════════════════════════════════════════════════
# Prompt rendering
# ════════════════════════════════════════════════════════════


class TestWorkflowPromptRendering:
    def test_catalog_view(self, wf_middleware):
        block = wf_middleware._render_catalog()
        assert "<available_workflows>" in block
        assert 'workflow name="onboarding"' in block
        assert "Onboard a new user" in block
        assert "Tasks: collect_info, verify" in block
        assert 'workflow name="support"' in block
        assert "Handle a support request" in block
        assert "activate_workflow" in block

    def test_active_workflow_pipeline_view(self, wf_middleware):
        ctx = wf_middleware._workflow_ctxs["onboarding"]
        statuses = {"collect_info": "in_progress", "verify": "pending"}
        block = wf_middleware._render_status_block(
            ctx,
            statuses,
            "collect_info",
        )
        assert '<task_pipeline workflow="onboarding">' in block
        assert "[>] collect_info (in_progress)" in block
        assert "[ ] verify (pending)" in block
        assert '<current_task name="collect_info">' in block
        assert "Collect user info." in block

    def test_catalog_injected_when_no_workflow(self, wf_middleware):
        """_prepare_model_request should inject catalog prompt when no workflow active."""
        request = MockModelRequest(
            state={"active_workflow": None, "messages": []},
            system_message=MockSystemMessage("Base."),
            tools=wf_middleware.tools,
        )
        modified = wf_middleware._build_catalog_request(request)
        content = modified.system_message.content
        text_blocks = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        full = "\n".join(text_blocks)
        assert "<available_workflows>" in full

    def test_pipeline_injected_when_workflow_active(self, wf_middleware):
        request = MockModelRequest(
            state={
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
                "messages": [],
            },
            system_message=MockSystemMessage("Base."),
            tools=wf_middleware.tools,
        )
        ctx = wf_middleware._workflow_ctxs["onboarding"]
        modified, _ = wf_middleware._prepare_model_request(request, ctx)
        content = modified.system_message.content
        text_blocks = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        full = "\n".join(text_blocks)
        assert "<task_pipeline" in full
        assert "collect_info" in full


# ════════════════════════════════════════════════════════════
# wrap_tool_call — workflow mode
# ════════════════════════════════════════════════════════════


class TestWorkflowWrapToolCall:
    def test_activate_passes_through(self, wf_middleware):
        request = MockToolCallRequest(
            tool_call={
                "name": _ACTIVATE_TOOL_NAME,
                "id": "c1",
                "args": {"workflow": "onboarding"},
            },
            state={"active_workflow": None},
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "handled")
        assert result == "handled"

    def test_deactivate_passes_through(self, wf_middleware):
        request = MockToolCallRequest(
            tool_call={"name": _DEACTIVATE_TOOL_NAME, "id": "c1", "args": {}},
            state={"active_workflow": "onboarding", "task_statuses": {}},
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "handled")
        assert result == "handled"

    def test_no_workflow_transparent(self, wf_middleware):
        """Any tool passes through when no workflow active."""
        request = MockToolCallRequest(
            tool_call={"name": "some_random_tool", "id": "c1", "args": {}},
            state={"active_workflow": None},
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "passed")
        assert result == "passed"

    def test_tool_gated_when_workflow_active(self, wf_middleware):
        """tool_c is support's tool, not onboarding's — should be gated."""
        request = MockToolCallRequest(
            tool_call={"name": "tool_c", "id": "c1", "args": {}},
            state={
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
            },
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "should_not_reach")
        assert isinstance(result, ToolMessage)
        assert "not available" in result.content

    def test_tool_allowed_for_active_task(self, wf_middleware):
        """tool_a is collect_info's tool — should pass through."""
        request = MockToolCallRequest(
            tool_call={"name": "tool_a", "id": "c1", "args": {}},
            state={
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
            },
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "passed")
        assert result == "passed"

    def test_global_tool_allowed(self, wf_middleware):
        """global_read is a global tool for onboarding — should pass through."""
        request = MockToolCallRequest(
            tool_call={"name": "global_read", "id": "c1", "args": {}},
            state={
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
            },
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "passed")
        assert result == "passed"

    def test_transition_validates_and_fires_hooks(self, wf_middleware):
        """Transition tool call should go through validation and handler."""
        cmd = Command(
            update={
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
                "nudge_count": 0,
                "messages": [
                    ToolMessage(
                        "Task 'collect_info' -> in_progress.", tool_call_id="c1"
                    )
                ],
            }
        )
        request = MockToolCallRequest(
            tool_call={
                "name": _TRANSITION_TOOL_NAME,
                "id": "c1",
                "args": {"task": "collect_info", "status": "in_progress"},
            },
            state={
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "pending",
                    "verify": "pending",
                },
            },
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: cmd)
        assert isinstance(result, Command)

    def test_transition_validation_rejects_double_start(self, wf_middleware):
        """Can't start a task when another is in_progress."""
        request = MockToolCallRequest(
            tool_call={
                "name": _TRANSITION_TOOL_NAME,
                "id": "c1",
                "args": {"task": "verify", "status": "in_progress"},
            },
            state={
                "active_workflow": "onboarding",
                "task_statuses": {
                    "collect_info": "in_progress",
                    "verify": "pending",
                },
            },
        )
        result = wf_middleware.wrap_tool_call(request, lambda r: "unreachable")
        assert isinstance(result, ToolMessage)
        assert "already in progress" in result.content


# ════════════════════════════════════════════════════════════
# Validation via TaskMiddleware in workflow mode
# ════════════════════════════════════════════════════════════


class TestWorkflowValidation:
    def test_validate_completion_rejects(self):
        tasks = [
            Task(
                name="gated",
                instruction="Gated task",
                tools=[tool_a],
                middleware=RejectCompletionMiddleware("Not ready."),
            ),
        ]
        wf = Workflow(name="wf", description="WF", tasks=tasks)
        mw = WorkflowSteeringMiddleware(workflows=[wf])

        request = MockToolCallRequest(
            tool_call={
                "name": _TRANSITION_TOOL_NAME,
                "id": "c1",
                "args": {"task": "gated", "status": "complete"},
            },
            state={
                "active_workflow": "wf",
                "task_statuses": {"gated": "in_progress"},
            },
        )
        result = mw.wrap_tool_call(request, lambda r: "unreachable")
        assert isinstance(result, ToolMessage)
        assert "Not ready." in result.content


# ════════════════════════════════════════════════════════════
# after_agent — workflow mode
# ════════════════════════════════════════════════════════════


class TestWorkflowAfterAgent:
    def test_no_workflow_noop(self, wf_middleware):
        state = {"active_workflow": None, "messages": []}
        result = wf_middleware.after_agent(state, runtime=None)
        assert result is None

    def test_noop_when_all_required_complete(self, wf_middleware):
        """No auto-deactivation — agent must call deactivate_workflow itself."""
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "complete",
                "verify": "complete",
            },
            "messages": [],
        }
        result = wf_middleware.after_agent(state, runtime=None)
        assert result is None

    def test_nudge_when_incomplete(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "complete",
                "verify": "pending",
            },
            "nudge_count": 0,
            "messages": [],
        }
        result = wf_middleware.after_agent(state, runtime=None)
        assert result is not None
        assert result["jump_to"] == "model"
        assert result["nudge_count"] == 1
        assert any(isinstance(m, HumanMessage) for m in result["messages"])

    def test_stops_nudging_after_max(self, wf_middleware):
        state = {
            "active_workflow": "onboarding",
            "task_statuses": {
                "collect_info": "complete",
                "verify": "pending",
            },
            "nudge_count": 3,
            "messages": [],
        }
        result = wf_middleware.after_agent(state, runtime=None)
        assert result is None

    def test_no_required_tasks_no_nudge(self, onboarding_tasks):
        wf = Workflow(
            name="opt",
            description="Optional",
            tasks=onboarding_tasks,
            required_tasks=None,
        )
        mw = WorkflowSteeringMiddleware(workflows=[wf])
        state = {
            "active_workflow": "opt",
            "task_statuses": {
                "collect_info": "pending",
                "verify": "pending",
            },
            "nudge_count": 0,
            "messages": [],
        }
        result = mw.after_agent(state, runtime=None)
        assert result is None

    def test_partial_required_tasks_noop_when_met(self, onboarding_tasks):
        """Only collect_info is required — completing it means no nudge."""
        wf = Workflow(
            name="partial",
            description="Partial",
            tasks=onboarding_tasks,
            required_tasks=["collect_info"],
        )
        mw = WorkflowSteeringMiddleware(workflows=[wf])
        state = {
            "active_workflow": "partial",
            "task_statuses": {
                "collect_info": "complete",
                "verify": "pending",
            },
            "messages": [],
        }
        result = mw.after_agent(state, runtime=None)
        assert result is None


# ════════════════════════════════════════════════════════════
# Lifecycle hooks (on_start / on_complete)
# ════════════════════════════════════════════════════════════


class TestWorkflowLifecycleHooks:
    def test_on_start_fires(self):
        class TrackingMiddleware(TaskMiddleware):
            started = False

            def on_start(self, state):
                TrackingMiddleware.started = True
                return None

        TrackingMiddleware.started = False
        tasks = [
            Task(
                name="tracked",
                instruction="Tracked",
                tools=[tool_a],
                middleware=TrackingMiddleware(),
            ),
        ]
        wf = Workflow(name="wf", description="WF", tasks=tasks)
        mw = WorkflowSteeringMiddleware(workflows=[wf])

        cmd = Command(
            update={
                "task_statuses": {"tracked": "in_progress"},
                "nudge_count": 0,
                "messages": [ToolMessage("ok", tool_call_id="c1")],
            }
        )
        request = MockToolCallRequest(
            tool_call={
                "name": _TRANSITION_TOOL_NAME,
                "id": "c1",
                "args": {"task": "tracked", "status": "in_progress"},
            },
            state={
                "active_workflow": "wf",
                "task_statuses": {"tracked": "pending"},
                "messages": [],
            },
        )
        mw.wrap_tool_call(request, lambda r: cmd)
        assert TrackingMiddleware.started is True

    def test_on_complete_fires(self):
        class TrackingMiddleware(TaskMiddleware):
            completed = False

            def on_complete(self, state):
                TrackingMiddleware.completed = True
                return None

        TrackingMiddleware.completed = False
        tasks = [
            Task(
                name="tracked",
                instruction="Tracked",
                tools=[tool_a],
                middleware=TrackingMiddleware(),
            ),
        ]
        wf = Workflow(name="wf", description="WF", tasks=tasks)
        mw = WorkflowSteeringMiddleware(workflows=[wf])

        cmd = Command(
            update={
                "task_statuses": {"tracked": "complete"},
                "nudge_count": 0,
                "messages": [ToolMessage("ok", tool_call_id="c1")],
            }
        )
        request = MockToolCallRequest(
            tool_call={
                "name": _TRANSITION_TOOL_NAME,
                "id": "c1",
                "args": {"task": "tracked", "status": "complete"},
            },
            state={
                "active_workflow": "wf",
                "task_statuses": {"tracked": "in_progress"},
                "messages": [],
            },
        )
        mw.wrap_tool_call(request, lambda r: cmd)
        assert TrackingMiddleware.completed is True


# ════════════════════════════════════════════════════════════
# Backend tools passthrough in workflow mode
# ════════════════════════════════════════════════════════════


class TestWorkflowBackendPassthrough:
    def test_backend_tools_available_when_enabled(self, onboarding_tasks):
        wf = Workflow(
            name="wf",
            description="WF",
            tasks=onboarding_tasks,
        )
        mw = WorkflowSteeringMiddleware(
            workflows=[wf],
            backend_tools_passthrough=True,
        )
        ctx = mw._workflow_ctxs["wf"]
        allowed = mw._allowed_tool_names(ctx, "collect_info")
        assert "read_file" in allowed
        assert "ls" in allowed

    def test_backend_tools_not_available_by_default(self, wf_middleware):
        ctx = wf_middleware._workflow_ctxs["onboarding"]
        allowed = wf_middleware._allowed_tool_names(ctx, "collect_info")
        assert "read_file" not in allowed


# ════════════════════════════════════════════════════════════
# End-to-end scenario
# ════════════════════════════════════════════════════════════


class TestWorkflowEndToEnd:
    """Simulates: activate A -> complete tasks -> auto-deactivate -> activate B."""

    def test_full_lifecycle(self, wf_middleware):
        # 1. Start with no workflow active
        state = {"active_workflow": None, "messages": [], "nudge_count": 0}
        result = wf_middleware.before_agent(state, runtime=None)
        assert result is None  # No init in workflow mode

        # 2. Activate onboarding
        result = _invoke_tool(
            wf_middleware._activate_tool,
            state,
            workflow="onboarding",
        )
        assert isinstance(result, Command)
        state.update(result.update)
        assert state["active_workflow"] == "onboarding"

        # 3. Start collect_info
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="in_progress",
        )
        assert isinstance(result, Command)
        state.update(result.update)

        # 4. Complete collect_info
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="collect_info",
            status="complete",
        )
        assert isinstance(result, Command)
        state.update(result.update)

        # 5. Start and complete verify
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="verify",
            status="in_progress",
        )
        state.update(result.update)
        result = _invoke_tool(
            wf_middleware._workflow_transition_tool,
            state,
            task="verify",
            status="complete",
        )
        state.update(result.update)

        # 6. after_agent should NOT auto-deactivate
        result = wf_middleware.after_agent(state, runtime=None)
        assert result is None

        # 7. Explicitly deactivate
        result = _invoke_tool(wf_middleware._deactivate_tool, state)
        assert isinstance(result, Command)
        state.update(result.update)
        assert state["active_workflow"] is None

        # 8. Now activate support
        result = _invoke_tool(
            wf_middleware._activate_tool,
            state,
            workflow="support",
        )
        assert isinstance(result, Command)
        state.update(result.update)
        assert state["active_workflow"] == "support"
        assert state["task_statuses"] == {"diagnose": "pending"}


# ════════════════════════════════════════════════════════════
# Backward compatibility
# ════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Task-mode (legacy) still works exactly as before."""

    def test_task_mode_still_works(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks)

        result = mw.before_agent({"messages": []}, runtime=None)
        assert result is not None
        assert "task_statuses" in result
        assert set(result["task_statuses"]) == {"step_1", "step_2", "step_3"}

    def test_task_mode_tools_registered(self, three_tasks):
        mw = TaskSteeringMiddleware(tasks=three_tasks)
        names = {t.name for t in mw.tools}
        assert _TRANSITION_TOOL_NAME in names
        assert _ACTIVATE_TOOL_NAME not in names
        assert _DEACTIVATE_TOOL_NAME not in names


# ════════════════════════════════════════════════════════════
# Catalog view via wrap_model_call (sync + async)
# ════════════════════════════════════════════════════════════


class TestWorkflowCatalogModelCall:
    def _build_request(self, state, tools):
        return MockModelRequest(
            state=state,
            system_message=MockSystemMessage("Base."),
            tools=tools,
        )

    def test_wrap_model_call_injects_catalog_when_no_workflow(self, wf_middleware):
        """With no active workflow, wrap_model_call should route through catalog."""
        seen: dict = {}

        def handler(request):
            seen["tools"] = {t.name for t in request.tools}
            seen["system"] = request.system_message
            return MagicMock()

        request = self._build_request(
            {"active_workflow": None, "messages": []},
            wf_middleware.tools,
        )
        wf_middleware.wrap_model_call(request, handler)

        # Catalog scoping: activate is present, deactivate/transition are not
        assert _ACTIVATE_TOOL_NAME in seen["tools"]
        assert _DEACTIVATE_TOOL_NAME not in seen["tools"]
        assert _TRANSITION_TOOL_NAME not in seen["tools"]

    @pytest.mark.asyncio
    async def test_awrap_model_call_injects_catalog_when_no_workflow(
        self, wf_middleware
    ):
        """Async version of catalog injection."""
        seen: dict = {}

        async def async_handler(request):
            seen["tools"] = {t.name for t in request.tools}
            return MagicMock()

        request = self._build_request(
            {"active_workflow": None, "messages": []},
            wf_middleware.tools,
        )
        await wf_middleware.awrap_model_call(request, async_handler)

        assert _ACTIVATE_TOOL_NAME in seen["tools"]
        assert _DEACTIVATE_TOOL_NAME not in seen["tools"]
        assert _TRANSITION_TOOL_NAME not in seen["tools"]
