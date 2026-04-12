# Workflow Mode

For agents that handle mixed workloads — freeform conversation plus structured workflows — use `WorkflowSteeringMiddleware`. The agent sees a catalog of available workflows and activates one on demand. When no workflow is active, the middleware is fully transparent (no tool filtering or prompt injection).

## Quick start

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_task_steering import (
    Task, Workflow, WorkflowSteeringMiddleware,
)


@tool
def ask_user(question: str) -> str:
    """Ask the user a question and wait for their response."""
    return interrupt({"question": question})


@tool
def register_user(username: str, email: str) -> str:
    """Register a new user."""
    return f"User '{username}' registered."


@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


onboarding = Workflow(
    name="user_onboarding",
    description="Register and onboard a new user account.",
    tasks=[
        Task(
            name="collect_info",
            instruction="Use ask_user to collect the user's username and email.",
            tools=[],
        ),
        Task(
            name="register",
            instruction="Register the user with the collected information.",
            tools=[register_user],
        ),
    ],
    global_tools=[ask_user],
    enforce_order=True,
)

research = Workflow(
    name="research",
    description="Research a topic and provide a summary.",
    tasks=[
        Task(name="search", instruction="Search for info.", tools=[search_web]),
        Task(name="summarize", instruction="Summarize findings.", tools=[]),
    ],
)

middleware = WorkflowSteeringMiddleware(workflows=[onboarding, research])

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[middleware],
    system_prompt="You are a helpful assistant.",
    checkpointer=MemorySaver(),
)
```

## What the agent sees

### No workflow active — catalog view

The middleware is transparent. All tools from `create_agent` pass through, plus `activate_workflow` is injected:

```xml
<available_workflows>
  <workflow name="user_onboarding">
    Register and onboard a new user account.
    Tasks: collect_info, register
  </workflow>
  <workflow name="research">
    Research a topic and provide a summary.
    Tasks: search, summarize
  </workflow>

  Use activate_workflow to start a workflow when needed.
</available_workflows>
```

### Workflow active — pipeline view

After `activate_workflow("user_onboarding")`, the agent sees the standard pipeline prompt scoped to that workflow:

```xml
<task_pipeline workflow="user_onboarding">
  [>] collect_info (in_progress)
  [ ] register (pending)

  <current_task name="collect_info">
    Use ask_user to collect the user's username and email.
  </current_task>

  <rules>
    Required order: collect_info -> register
    Use update_task_status to advance. Do not skip tasks.
  </rules>
</task_pipeline>
```

## Agent-driven tools

| Tool | When available | Description |
|---|---|---|
| `activate_workflow(workflow)` | No workflow active | Sets `active_workflow`, initializes `task_statuses` |
| `deactivate_workflow()` | Workflow active | Clears state. Blocked if a task is `in_progress` unless `allow_deactivate_in_progress=True` |
| `update_task_status(task, status)` | Workflow active | Transitions within the active workflow |

## Tool scoping

| State | Available tools |
|---|---|
| No workflow active | **All tools** from `create_agent` + `activate_workflow` (transparent) |
| Workflow active, no task active | `deactivate_workflow` + `update_task_status` + workflow's `global_tools` |
| Workflow active, task active | Above + active task's `tools` |

## Human-in-the-loop with `interrupt()`

For tasks that need user input (e.g., collecting information), use LangGraph's `interrupt()` inside a tool. The agent pauses, the caller collects the response, and resumes:

```python
from langgraph.types import Command, interrupt

@tool
def ask_user(question: str) -> str:
    """Ask the user a question and wait for their response."""
    return interrupt({"question": question})

# In your run loop:
for chunk in agent.stream(inputs, config, stream_mode="messages"):
    display(chunk)

# Check for interrupt
snapshot = agent.get_state(config)
if snapshot.next:  # agent is paused
    question = snapshot.tasks[0].interrupts[0].value["question"]
    response = input(question)
    # Resume the agent
    for chunk in agent.stream(Command(resume=response), config, stream_mode="messages"):
        display(chunk)
```

## Deactivation

The agent must explicitly call `deactivate_workflow()` to exit a workflow and return to freeform mode. By default, deactivation is blocked while a task is `in_progress`:

```python
Workflow(
    name="flexible",
    tasks=[...],
    allow_deactivate_in_progress=True,  # allow bailing mid-task
)
```

## Feature parity with task mode

Tasks inside workflows support all the same features as `TaskSteeringMiddleware`:

- **[Task summarization](summarization.md)** — both `replace` and `summarize` modes
- **[Task-scoped middleware](task-middleware.md)** — `validate_completion`, `wrap_tool_call`, `wrap_model_call`, lifecycle hooks, `state_schema`, and composition
- **[Task-scoped skills](skills.md)** — `Task(skills=...)` and `Workflow(global_skills=...)`
- **[Backend tools passthrough](backend-passthrough.md)** — `backend_tools_passthrough=True`

## Workflow vs task mode

| | `TaskSteeringMiddleware` | `WorkflowSteeringMiddleware` |
|---|---|---|
| Steering starts | Immediately | When agent calls `activate_workflow` |
| Tool filtering | Always on | Only when a workflow is active |
| `global_tools` | On the middleware | On each `Workflow` |
| Multiple pipelines | No | Yes — agent picks which one |
| Exit workflow | Automatic when done | Agent calls `deactivate_workflow` |
| `enforce_order`, `required_tasks` | On the middleware | On each `Workflow` |

## Workflow fields

| Field | Default | Description |
|---|---|---|
| `name` | *(required)* | Unique workflow identifier. |
| `description` | *(required)* | Shown in catalog so the agent can decide which to activate. |
| `tasks` | *(required)* | Ordered list of `Task` definitions. |
| `global_tools` | `[]` | Tools available across all tasks in this workflow. |
| `global_skills` | `None` | Skill names available across all tasks. |
| `enforce_order` | `True` | Tasks must be completed in order within this workflow. |
| `required_tasks` | `["*"]` | Tasks that must complete before the agent can exit. |
| `allow_deactivate_in_progress` | `False` | Allow deactivation while a task is `in_progress`. |

## WorkflowSteeringMiddleware configuration

| Parameter | Default | Description |
|---|---|---|
| `workflows` | *(required)* | List of `Workflow` definitions. |
| `max_nudges` | `3` | Max nudges for incomplete required tasks. |
| `backend_tools_passthrough` | `False` | Whitelist backend tools through the filter. |
| `backend_tools` | `None` | Override `DEFAULT_BACKEND_TOOLS`. |
| `model` | `None` | Default model for `TaskSummarization(mode="summarize")`. |
