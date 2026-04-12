# Task Mode

Task mode is the default way to use `TaskSteeringMiddleware`. You pass a list of `Task` definitions and the middleware enforces an ordered pipeline from the first model call.

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
| `before_agent` | Initializes `task_statuses` in state. |
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
| `global_skills` | `None` | Skill names available regardless of active task. |
| `backend_tools_passthrough` | `False` | Whitelist known backend tools through the tool filter. |
| `backend_tools` | `None` | Override `DEFAULT_BACKEND_TOOLS`. `None` uses the built-in set. |
| `model` | `None` | Default chat model for `TaskSummarization(mode="summarize")`. |

### Task fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique identifier (used in prompts and state). |
| `instruction` | yes | Injected into system prompt when this task is active. |
| `tools` | yes | Tools visible when this task is `IN_PROGRESS`. |
| `middleware` | no | Scoped middleware — a `TaskMiddleware`, `AgentMiddleware` (auto-wrapped), or a list of them. Only active during this task. |
| `skills` | no | Skill names available when this task is `IN_PROGRESS`. Skill metadata comes from state (loaded by `SkillsMiddleware`). |
| `summarize` | no | Post-completion summarization config. See [Task summarization](summarization.md). |

## Async support

All middleware hooks have async counterparts (`awrap_model_call`, `awrap_tool_call`, `abefore_agent`, `aafter_agent`). Agents using `astream()` or `ainvoke()` are fully supported.

## Composability

`TaskSteeringMiddleware` is a standard `AgentMiddleware`. It composes with other middleware:

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
