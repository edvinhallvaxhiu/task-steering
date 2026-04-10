# langchain-task-steering

Implicit state-machine middleware for [LangChain v1](https://python.langchain.com/) agents. Define ordered task pipelines with per-task tool scoping, dynamic prompt injection, and composable validation — all as a drop-in `AgentMiddleware`.

Also available for [TypeScript/JavaScript](https://www.npmjs.com/package/langchain-task-steering).

```
PENDING ──> IN_PROGRESS ──> COMPLETE
```

The model drives its own transitions by calling `update_task_status`. The middleware enforces ordering, scopes tools, injects the active task's instruction into the system prompt, and gates completion via pluggable validators.

## When to use this

| Scenario | task-steering | LangGraph explicit workflows |
|---|:---:|:---:|
| Linear task pipeline (A then B then C) | **Best fit** | Verbose — one node + edges per task |
| Per-task tool scoping | **Built-in** | Manual — separate tool lists per node |
| Dynamic tasks from config / DB | **Easy** — tasks are data | Hard — graph is compiled at build time |
| Branching / parallel execution | Not supported | **Built-in** — edges + `Send()` |
| Per-task human-in-the-loop interrupts | Not supported | **Built-in** — `interrupt()` per node |
| Complex orchestration with retries / cycles | Not supported | **Built-in** — conditional edges |
| Composition with other middleware | **Native** — it's an `AgentMiddleware` | N/A — different abstraction |
| Debuggability in LangGraph Studio | Opaque — single agent node | **Clear** — each node visible in traces |

**Rule of thumb:** If your tasks are sequential and tool-scoped, use task-steering. If you need branching, parallelism, or per-task interrupts, use explicit LangGraph workflows.

## Install

```bash
pip install langchain-task-steering
```

For development:

```bash
git clone https://github.com/edvinhallvaxhiu/langchain-task-steering
cd langchain-task-steering/packages/python
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- `langchain >= 1.0.0`
- `langgraph >= 0.4.0`

## Quick start

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_task_steering import TaskSteeringMiddleware, Task


@tool
def add_items(items: list[str]) -> str:
    """Add items to the inventory."""
    return f"Added {len(items)} items."


@tool
def categorize(categories: dict[str, list[str]]) -> str:
    """Assign items to categories."""
    return f"Categorized into {len(categories)} groups."


pipeline = TaskSteeringMiddleware(
    tasks=[
        Task(
            name="collect",
            instruction="Collect all relevant items from the user's input.",
            tools=[add_items],
        ),
        Task(
            name="categorize",
            instruction="Organize the collected items into categories.",
            tools=[categorize],
        ),
    ],
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[pipeline],
    system_prompt="You are an inventory assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "I have apples, bolts, and milk."}]}
)
```

The agent automatically receives an `update_task_status` tool and sees a task pipeline block in its system prompt. It must complete `collect` before starting `categorize`.

## How it works

### What the model sees

Every model call, the middleware appends a status block to the system prompt:

```xml
<task_pipeline>
  [x] collect (complete)
  [>] categorize (in_progress)

  <current_task name="categorize">
    Organize the collected items into categories.
  </current_task>

  <rules>
    Required order: collect -> categorize
    Use update_task_status to advance. Do not skip tasks.
  </rules>
</task_pipeline>
```

Only the active task's tools (plus globals and `update_task_status`) are visible to the model.

### Middleware hooks

| Hook | Behavior |
|---|---|
| `before_agent` | Initializes `task_statuses` in state on first invocation. |
| `wrap_model_call` | Appends task status board + active task instruction to system prompt. Filters tools to only the active task's tools + globals + `update_task_status`. Delegates to task-scoped middleware if present. |
| `wrap_tool_call` | Intercepts `update_task_status` — runs `validate_completion` on the task's scoped middleware before allowing completion. Rejects out-of-scope tool calls. Delegates other tool calls to the active task's scoped middleware. |
| `after_agent` | Checks if required tasks are complete. If not, nudges the agent with a `HumanMessage` and jumps back to the model (up to `max_nudges` times). |
| `tools` | Auto-registers all task tools + globals + `update_task_status` with the agent. |

### Task lifecycle

```
PENDING ──> IN_PROGRESS ──> COMPLETE
```

- The agent drives transitions by calling `update_task_status(task, status)`.
- Transitions are enforced: `pending -> in_progress -> complete` only.
- When `enforce_order=True`, a task cannot start until all preceding tasks are complete.
- On `complete`, the task's `middleware.validate_completion(state)` runs first — rejection returns an error to the agent without completing the transition.

## Task-scoped middleware

Each task can have a `TaskMiddleware` that activates only when the task is `IN_PROGRESS`. This enables mid-task enforcement, not just completion gating.

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

### TaskMiddleware hooks

| Method | When it runs | Purpose |
|---|---|---|
| `validate_completion(state)` | Before `complete` transition | Return error string to reject, `None` to allow |
| `on_start(state)` | After successful `in_progress` transition | Side effects (logging, state init) |
| `on_complete(state)` | After successful `complete` transition | Side effects (trail capture, cleanup) |
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

## Async support

All middleware hooks have async counterparts (`awrap_model_call`, `awrap_tool_call`, `abefore_agent`, `aafter_agent`). Agents using `astream()` or `ainvoke()` are fully supported.

### Persistent state for task middleware

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

`TaskSteeringMiddleware` automatically merges all task middleware schemas into its own `state_schema`, so the fields survive checkpointing and interrupts.

## Required tasks

By default, all tasks are required — if the agent tries to exit without completing them, the middleware nudges it back with a `HumanMessage` listing the incomplete tasks.

```python
# All tasks required (default)
pipeline = TaskSteeringMiddleware(tasks=tasks)

# Only specific tasks required
pipeline = TaskSteeringMiddleware(tasks=tasks, required_tasks=["collect", "review"])

# No tasks required (agent can exit at any time)
pipeline = TaskSteeringMiddleware(tasks=tasks, required_tasks=None)

# Custom nudge limit (default is 3)
pipeline = TaskSteeringMiddleware(tasks=tasks, max_nudges=5)
```

The nudge mechanism uses the `after_agent` hook with `jump_to: "model"` to re-enter the agent loop. After `max_nudges` attempts, the agent is allowed to exit regardless.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `tasks` | *(required)* | Ordered list of `Task` definitions. |
| `global_tools` | `[]` | Tools available in every task. |
| `enforce_order` | `True` | Require tasks to be completed in definition order. |
| `required_tasks` | `["*"]` | Tasks that must be completed before the agent can exit. `["*"]` = all, `None` = none, or a list of task names. |
| `max_nudges` | `3` | Max times the agent is nudged to complete required tasks before being allowed to exit. |

### Task fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique identifier (used in prompts and state). |
| `instruction` | yes | Injected into system prompt when this task is active. |
| `tools` | yes | Tools visible when this task is `IN_PROGRESS`. |
| `middleware` | no | Scoped middleware — a `TaskMiddleware`, `AgentMiddleware` (auto-wrapped), or a list of them. Only active during this task. |

## Composability

`TaskSteeringMiddleware` is a standard `AgentMiddleware`. It composes with other LangChain v1 middleware:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[
        SummarizationMiddleware(
            model="anthropic:claude-haiku-4-5-20251001",
            trigger={"tokens": 8000},
        ),
        pipeline,
    ],
)
```

## Development

```bash
cd packages/python
pip install -e ".[dev]"
pytest
pytest --cov=langchain_task_steering
```

## License

MIT — see [LICENSE](../../LICENSE).
