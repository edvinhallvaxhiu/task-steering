# Task-Scoped Middleware

Each task can have a `TaskMiddleware` that activates only when the task is `IN_PROGRESS`. This enables mid-task enforcement, not just completion gating.

## Example

```python
from langchain.messages import ToolMessage
from langchain_task_steering import Task, TaskMiddleware, TaskSteeringMiddleware


class ThreatsMiddleware(TaskMiddleware):
    """Block gap_analysis until enough threats exist."""

    def __init__(self, min_threats: int = 25):
        super().__init__()
        self.min_threats = min_threats

    def validate_completion(self, state) -> str | None:
        threats = state.get("threats", [])
        if len(threats) < self.min_threats:
            return f"Only {len(threats)} threats — need at least {self.min_threats}."
        return None

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] == "gap_analysis":
            threats = request.state.get("threats", [])
            if len(threats) < self.min_threats:
                return ToolMessage(
                    content=f"Cannot run gap_analysis: {len(threats)}/{self.min_threats} threats.",
                    tool_call_id=request.tool_call["id"],
                )
        return handler(request)


pipeline = TaskSteeringMiddleware(
    tasks=[
        Task(name="assets", instruction="...", tools=[create_assets]),
        Task(
            name="threats",
            instruction="Identify STRIDE threats for each asset.",
            tools=[create_threats, gap_analysis],
            middleware=ThreatsMiddleware(min_threats=25),
        ),
    ],
)
```

## TaskMiddleware hooks

| Method | When it runs | Purpose |
|---|---|---|
| `validate_completion(state)` | Before `complete` transition | Return error string to reject, `None` to allow |
| `avalidate_completion(state)` | Async version (used by `awrap_tool_call`) | Default delegates to sync `validate_completion` |
| `on_start(state)` | After successful `in_progress` transition | Side effects (logging, state init) |
| `aon_start(state)` | Async version | Default delegates to sync `on_start` |
| `on_complete(state)` | After successful `complete` transition | Side effects (trail capture, cleanup) |
| `aon_complete(state)` | Async version | Default delegates to sync `on_complete` |
| `wrap_tool_call(request, handler)` | On every tool call during this task | Mid-task tool gating / modification |
| `wrap_model_call(request, handler)` | On every model call during this task | Extra prompt injection / request modification |
| `state_schema` | At middleware init | Merge custom state fields into the agent's state |
| `tools` *(property)* | At middleware construction | Extra tools to register and scope to this task |

## Using community middleware at task scope

Standard `AgentMiddleware` instances can be passed directly to a task — they're auto-wrapped in `AgentMiddlewareAdapter`:

```python
from langchain.agents.middleware import SummarizationMiddleware
from langchain_task_steering import Task, TaskSteeringMiddleware

pipeline = TaskSteeringMiddleware(
    tasks=[
        Task(
            name="research",
            instruction="Research the topic thoroughly.",
            tools=[search_tool],
            middleware=SummarizationMiddleware(),  # auto-wrapped
        ),
    ],
)
```

The adapter forwards `wrap_model_call`, `wrap_tool_call` (and their async counterparts), `tools`, and `state_schema` from the inner middleware. Agent-level hooks (`before_agent`, `after_agent`) are not forwarded. Invalid middleware objects are warned and skipped.

Wrap-style hooks are discovered dynamically from `AgentMiddleware` at import time, so new hooks added by future LangChain versions are picked up automatically.

## Middleware composition

Tasks accept a list of middleware, composed like LangChain's `create_agent(middleware=[...])`:

```python
Task(
    name="research",
    instruction="Research the topic thoroughly.",
    tools=[search_tool],
    middleware=[
        SummarizationMiddleware(),   # auto-wrapped, outermost hook wrapper
        ResearchValidator(),         # TaskMiddleware with validate_completion
    ],
)
```

Composition semantics:
- **Wrap-style hooks** (`wrap_model_call`, `wrap_tool_call`): first = outermost wrapper.
- **`validate_completion`**: all validators run; first error wins.
- **`on_start` / `on_complete`**: all fire in order.
- **`tools`**: merged from all middleware, deduplicated.

## Persistent state

Task middleware can declare a `state_schema` to persist custom fields across interrupts:

```python
from langchain.agents import AgentState
from typing_extensions import NotRequired


class ThreatsState(AgentState):
    gap_analysis_uses: NotRequired[int]


class ThreatsMiddleware(TaskMiddleware):
    state_schema = ThreatsState
    # ...
```

`TaskSteeringMiddleware` (and `WorkflowSteeringMiddleware`) automatically merge all task middleware schemas into their own `state_schema`, so the fields survive checkpointing and interrupts.
