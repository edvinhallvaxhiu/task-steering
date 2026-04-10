"""
Runnable demo: simulates a full agent loop through the middleware.

Run with:
    python examples/demo.py
"""

from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Any

from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.types import Command

from langchain_task_steering import (
    Task,
    TaskMiddleware,
    TaskSteeringMiddleware,
    AgentMiddlewareAdapter,
)
from langchain_task_steering.types import TaskStatus

# ── Colors ──────────────────────────────────────────────────

BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def log(prefix: str, color: str, msg: str):
    print(f"  {color}{prefix}{RESET} {msg}")


def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── Mock request objects ────────────────────────────────────


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


# ── Tools ───────────────────────────────────────────────────


@tool
def gather_requirements(topic: str) -> str:
    """Gather requirements for a given topic."""
    return f"Requirements for '{topic}': must be fast, secure, and scalable."


@tool
def write_design(requirements: str) -> str:
    """Write a design document based on requirements."""
    return f"Design document created based on: {requirements}"


@tool
def review_design(design: str) -> str:
    """Review a design document and provide feedback."""
    return f"Review complete. Design looks good: {design}"


@tool
def search_docs(query: str) -> str:
    """Search documentation (available in all tasks)."""
    return f"Results for: {query}"


@tool
def audit_log(entry: str) -> str:
    """Log an audit entry (contributed by adapter)."""
    return f"Audit logged: {entry}"


# ── Agent-level middleware to wrap via adapter ──────────────

from langchain.agents.middleware import AgentMiddleware


class AuditMiddleware(AgentMiddleware):
    """Simulates an existing agent middleware that intercepts calls and contributes tools."""

    def __init__(self):
        super().__init__()
        self.tools = [audit_log]
        self.model_call_count = 0
        self.tool_call_count = 0

    def wrap_model_call(self, request, handler):
        self.model_call_count += 1
        log(
            "ADAPTER",
            YELLOW,
            f"AuditMiddleware.wrap_model_call() — call #{self.model_call_count}",
        )
        return handler(request)

    def wrap_tool_call(self, request, handler):
        self.tool_call_count += 1
        tool_name = request.tool_call["name"]
        log(
            "ADAPTER",
            YELLOW,
            f"AuditMiddleware.wrap_tool_call({tool_name}) — call #{self.tool_call_count}",
        )
        return handler(request)


class NoOpMiddleware(AgentMiddleware):
    """A no-op middleware — verifies the adapter gracefully handles no overrides."""

    pass


# ── Task middleware with validation ─────────────────────────


class DesignMiddleware(TaskMiddleware):
    def validate_completion(self, state) -> str | None:
        if not state.get("design_written"):
            return "You must call write_design before completing this task."
        return None

    def on_start(self, state):
        statuses = state.get("task_statuses", {})
        log("LIFECYCLE", YELLOW, f"DesignMiddleware.on_start() — statuses: {statuses}")

    def on_complete(self, state):
        statuses = state.get("task_statuses", {})
        log(
            "LIFECYCLE",
            YELLOW,
            f"DesignMiddleware.on_complete() — statuses: {statuses}",
        )


# ── Build the middleware ────────────────────────────────────

audit_mw = AuditMiddleware()

tasks = [
    Task(
        name="requirements",
        instruction="Gather the requirements for a login page.",
        tools=[gather_requirements],
        middleware=AgentMiddlewareAdapter(NoOpMiddleware()),
    ),
    Task(
        name="design",
        instruction="Write a design document based on the gathered requirements.",
        tools=[write_design],
        middleware=DesignMiddleware(),
    ),
    Task(
        name="review",
        instruction="Review the design document and provide final feedback.",
        tools=[review_design],
        middleware=AgentMiddlewareAdapter(audit_mw),
    ),
]

mw = TaskSteeringMiddleware(tasks=tasks, global_tools=[search_docs])

print(f"\n{GREEN}langchain-task-steering Python Demo{RESET}")
print(f"{DIM}Simulating an agent loop through all middleware hooks{RESET}")
print(f"{DIM}Including AgentMiddlewareAdapter forwarding{RESET}\n")

print("Registered tools:", ", ".join(t.name for t in mw.tools))


# ── State ───────────────────────────────────────────────────

state: dict[str, Any] = {"messages": []}


def extract_text(request):
    content = request.system_message.content
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


# ════════════════════════════════════════════════════════════
# Step 1: before_agent — initialize state
# ════════════════════════════════════════════════════════════

header("Step 1: before_agent — initialize state")

init = mw.before_agent(state, runtime=None)
if init:
    state.update(init)
    log("STATE", BLUE, f"task_statuses = {state['task_statuses']}")

# ════════════════════════════════════════════════════════════
# Step 2: wrap_model_call — no active task yet
# ════════════════════════════════════════════════════════════

header("Step 2: wrap_model_call — no active task yet")

captured = {}
mw.wrap_model_call(
    MockModelRequest(
        state=state, system_message=MockSystemMessage("You are a PM."), tools=mw.tools
    ),
    lambda r: captured.update(req=r) or MagicMock(),
)

tool_names = {t.name for t in captured["req"].tools}
log("TOOLS", BLUE, f"Model sees: {', '.join(sorted(tool_names))}")

prompt = extract_text(captured["req"])
log("PROMPT", DIM, "System prompt includes:")
for line in prompt.split("\n"):
    print(f"    {DIM}{line}{RESET}")

# ════════════════════════════════════════════════════════════
# Step 3: Start "requirements" (NoOpMiddleware adapter)
# ════════════════════════════════════════════════════════════

header('Step 3: Start "requirements" (NoOpMiddleware adapter — should pass through)')

state["task_statuses"]["requirements"] = "in_progress"
log("OK", GREEN, "requirements -> in_progress")
log("STATE", BLUE, f"task_statuses = {state['task_statuses']}")

# ════════════════════════════════════════════════════════════
# Step 4: wrap_model_call — "requirements" active with NoOp adapter
# ════════════════════════════════════════════════════════════

header('Step 4: wrap_model_call — "requirements" active (NoOp adapter passes through)')

captured = {}
mw.wrap_model_call(
    MockModelRequest(
        state=state, system_message=MockSystemMessage("Base."), tools=mw.tools
    ),
    lambda r: captured.update(req=r) or MagicMock(),
)

tool_names = {t.name for t in captured["req"].tools}
log("TOOLS", BLUE, f"Model sees: {', '.join(sorted(tool_names))}")
log("OK", GREEN, "No crash — NoOp adapter correctly passed through to handler")

# ════════════════════════════════════════════════════════════
# Step 5: Try wrong-task tool (rejected)
# ════════════════════════════════════════════════════════════

header("Step 5: Try to call write_design (wrong task — should be rejected)")

result = mw.wrap_tool_call(
    MockToolCallRequest(
        tool_call={"name": "write_design", "args": {}, "id": "call-2"},
        state=state,
    ),
    MagicMock(),
)
if isinstance(result, ToolMessage):
    log("REJECTED", RED, result.content)

# ════════════════════════════════════════════════════════════
# Step 6: Try to skip ahead (rejected)
# ════════════════════════════════════════════════════════════

header('Step 6: Try to start "design" before completing "requirements"')

skip_req = MockToolCallRequest(
    tool_call={
        "name": "update_task_status",
        "args": {"task": "design", "status": "in_progress"},
        "id": "call-3",
    },
    state={
        **state,
        "task_statuses": {**state["task_statuses"], "requirements": "pending"},
    },
)
skip_handler = MagicMock(
    return_value="Cannot start 'design': 'requirements' is not complete yet."
)
result = mw.wrap_tool_call(skip_req, skip_handler)
log(
    "REJECTED",
    RED,
    str(result) if isinstance(result, str) else skip_handler.return_value,
)

# ════════════════════════════════════════════════════════════
# Step 7: Complete "requirements"
# ════════════════════════════════════════════════════════════

header('Step 7: Complete "requirements"')

state["task_statuses"]["requirements"] = "complete"
log("OK", GREEN, "requirements -> complete")

# ════════════════════════════════════════════════════════════
# Step 8: Start "design" (lifecycle hooks fire)
# ════════════════════════════════════════════════════════════

header('Step 8: Start "design" (lifecycle hooks fire with post-transition state)')

start_design_req = MockToolCallRequest(
    tool_call={
        "name": "update_task_status",
        "args": {"task": "design", "status": "in_progress"},
        "id": "call-5",
    },
    state=state,
)
result = mw.wrap_tool_call(
    start_design_req,
    lambda r: Command(
        update={"task_statuses": {**state["task_statuses"], "design": "in_progress"}}
    ),
)
if isinstance(result, Command):
    state["task_statuses"]["design"] = "in_progress"
    log("OK", GREEN, "design -> in_progress")
    log("STATE", BLUE, f"task_statuses = {state['task_statuses']}")

# ════════════════════════════════════════════════════════════
# Step 9: Try to complete "design" without writing (rejected)
# ════════════════════════════════════════════════════════════

header('Step 9: Try to complete "design" (validation should reject)')

complete_req = MockToolCallRequest(
    tool_call={
        "name": "update_task_status",
        "args": {"task": "design", "status": "complete"},
        "id": "call-6",
    },
    state=state,
)
result = mw.wrap_tool_call(complete_req, MagicMock())
if isinstance(result, ToolMessage):
    log("REJECTED", RED, result.content)

# ════════════════════════════════════════════════════════════
# Step 10: Fix state and complete "design"
# ════════════════════════════════════════════════════════════

header('Step 10: Set design_written=True, then complete "design" (on_complete fires)')

state["design_written"] = True
log("STATE", BLUE, "Set design_written = True")

complete_req2 = MockToolCallRequest(
    tool_call={
        "name": "update_task_status",
        "args": {"task": "design", "status": "complete"},
        "id": "call-7",
    },
    state=state,
)
result = mw.wrap_tool_call(
    complete_req2,
    lambda r: Command(
        update={"task_statuses": {**state["task_statuses"], "design": "complete"}}
    ),
)
if isinstance(result, Command):
    state["task_statuses"]["design"] = "complete"
    log("OK", GREEN, "design -> complete")

# ════════════════════════════════════════════════════════════
# Step 11: Start "review" (AuditMiddleware adapter)
# ════════════════════════════════════════════════════════════

header('Step 11: Start "review" (AuditMiddleware adapter — hooks forwarded)')

start_review_req = MockToolCallRequest(
    tool_call={
        "name": "update_task_status",
        "args": {"task": "review", "status": "in_progress"},
        "id": "call-8",
    },
    state=state,
)
result = mw.wrap_tool_call(
    start_review_req,
    lambda r: Command(
        update={"task_statuses": {**state["task_statuses"], "review": "in_progress"}}
    ),
)
if isinstance(result, Command):
    state["task_statuses"]["review"] = "in_progress"
    log("OK", GREEN, "review -> in_progress")
    log("STATE", BLUE, f"task_statuses = {state['task_statuses']}")

# ════════════════════════════════════════════════════════════
# Step 12: wrap_model_call — "review" active, adapter intercepts
# ════════════════════════════════════════════════════════════

header('Step 12: wrap_model_call — "review" active (AuditMiddleware intercepts)')

captured = {}
mw.wrap_model_call(
    MockModelRequest(
        state=state, system_message=MockSystemMessage("Base."), tools=mw.tools
    ),
    lambda r: captured.update(req=r) or MagicMock(),
)

tool_names = {t.name for t in captured["req"].tools}
log("TOOLS", BLUE, f"Model sees: {', '.join(sorted(tool_names))}")
log("INFO", BLUE, f"audit_log tool is scoped: {'audit_log' in tool_names}")

# ════════════════════════════════════════════════════════════
# Step 13: Use review_design — adapter intercepts wrap_tool_call
# ════════════════════════════════════════════════════════════

header("Step 13: Call review_design — AuditMiddleware.wrap_tool_call intercepts")

review_tool_req = MockToolCallRequest(
    tool_call={
        "name": "review_design",
        "args": {"design": "login page v1"},
        "id": "call-9",
    },
    state=state,
)
expected = ToolMessage(content="Review looks good.", tool_call_id="call-9")
result = mw.wrap_tool_call(review_tool_req, lambda r: expected)
if isinstance(result, ToolMessage):
    log("RESULT", GREEN, result.content)

# ════════════════════════════════════════════════════════════
# Step 14: Use audit_log (adapter-contributed tool)
# ════════════════════════════════════════════════════════════

header("Step 14: Call audit_log — tool contributed by adapter")

audit_tool_req = MockToolCallRequest(
    tool_call={
        "name": "audit_log",
        "args": {"entry": "design reviewed"},
        "id": "call-10",
    },
    state=state,
)
expected = ToolMessage(content="Audit entry logged.", tool_call_id="call-10")
result = mw.wrap_tool_call(audit_tool_req, lambda r: expected)
if isinstance(result, ToolMessage):
    log("RESULT", GREEN, result.content)

# ════════════════════════════════════════════════════════════
# Step 15: Complete "review"
# ════════════════════════════════════════════════════════════

header('Step 15: Complete "review"')

state["task_statuses"]["review"] = "complete"
log("OK", GREEN, "review -> complete")
log("STATE", BLUE, f"task_statuses = {state['task_statuses']}")

# ════════════════════════════════════════════════════════════
# Step 16: after_agent — all tasks complete
# ════════════════════════════════════════════════════════════

header("Step 16: after_agent — check if agent can exit")

nudge = mw.after_agent(state, runtime=None)
if nudge is None:
    log("OK", GREEN, "All required tasks complete — agent can exit.")
else:
    log("NUDGE", YELLOW, str(nudge))

# ════════════════════════════════════════════════════════════
# Step 17: Adapter call summary
# ════════════════════════════════════════════════════════════

header("Step 17: AuditMiddleware adapter call summary")

log(
    "INFO",
    BLUE,
    f"AuditMiddleware.wrap_model_call invoked {audit_mw.model_call_count} time(s)",
)
log(
    "INFO",
    BLUE,
    f"AuditMiddleware.wrap_tool_call invoked {audit_mw.tool_call_count} time(s)",
)

# ════════════════════════════════════════════════════════════
# Bonus: null system message
# ════════════════════════════════════════════════════════════

header("Bonus: wrap_model_call with null system message — no crash")

captured = {}
mw.wrap_model_call(
    MockModelRequest(state=state, system_message=None, tools=mw.tools),
    lambda r: captured.update(req=r) or MagicMock(),
)
content = captured["req"].system_message.content
has_pipeline = any(
    "<task_pipeline>" in b.get("text", "") for b in content if isinstance(b, dict)
)
log("OK", GREEN, f"Pipeline block injected: {has_pipeline}")

# ════════════════════════════════════════════════════════════
# Bonus: afterAgent nudge with incomplete tasks
# ════════════════════════════════════════════════════════════

header("Bonus: after_agent nudge with incomplete tasks")

incomplete = {
    "messages": [],
    "task_statuses": {
        "requirements": "complete",
        "design": "in_progress",
        "review": "pending",
    },
}
nudge_result = mw.after_agent(incomplete, runtime=None)
if nudge_result:
    log("NUDGE", YELLOW, f"jump_to: {nudge_result['jump_to']}")
    log("NUDGE", YELLOW, f"nudge_count: {nudge_result['nudge_count']}")
    log("NUDGE", YELLOW, f"message: {nudge_result['messages'][0].content}")

print(f"\n{GREEN}Demo complete!{RESET}\n")
