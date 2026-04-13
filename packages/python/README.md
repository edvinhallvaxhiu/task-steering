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
| Multiple workflows, agent-driven activation | **Built-in** — `WorkflowSteeringMiddleware` | Manual — routing logic + subgraphs |
| Human-in-the-loop within tasks | **Built-in** — `interrupt()` in tools | **Built-in** — `interrupt()` per node |
| Branching / parallel execution | Not supported | **Built-in** — edges + `Send()` |
| Complex orchestration with retries / cycles | Not supported | **Built-in** — conditional edges |
| Composition with other middleware | **Native** — it's an `AgentMiddleware` | N/A — different abstraction |

**Rule of thumb:** If your tasks are sequential and tool-scoped, use task-steering. If you need agent-driven workflow selection with mixed freeform + structured work, use `WorkflowSteeringMiddleware`. If you need branching, parallelism, or per-node graph control, use explicit LangGraph workflows.

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

## Workflow mode

For agents that handle mixed workloads — freeform conversation plus structured workflows — use `WorkflowSteeringMiddleware`:

```python
from langchain_task_steering import Workflow, WorkflowSteeringMiddleware

middleware = WorkflowSteeringMiddleware(
    workflows=[
        Workflow(
            name="onboarding",
            description="Onboard a new user",
            tasks=[
                Task(name="collect_info", instruction="Collect user details.", tools=[...]),
                Task(name="register", instruction="Register the account.", tools=[...]),
            ],
            global_tools=[ask_user],
        ),
        Workflow(
            name="support",
            description="Handle a support request",
            tasks=[...],
        ),
    ],
)
```

The agent starts in freeform mode with its full toolset. When a request matches a workflow, it calls `activate_workflow("onboarding")` to enter the structured pipeline. Tool scoping, prompt injection, and task ordering kick in only while a workflow is active.

See [Workflow Mode](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/workflow-mode.md) for full documentation.

## Documentation

| Topic | Description |
|---|---|
| [Task Mode](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/task-mode.md) | Task lifecycle, hooks, tool scoping, required tasks, configuration |
| [Workflow Mode](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/workflow-mode.md) | Dynamic workflow activation, catalog, human-in-the-loop, deactivation |
| [Task Middleware](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/task-middleware.md) | TaskMiddleware hooks, validation, composition, persistent state |
| [Summarization](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/summarization.md) | Post-completion message compression (replace and summarize modes) |
| [Skills](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/skills.md) | Task-scoped skills from SKILL.md files |
| [Backend Passthrough](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/docs/backend-passthrough.md) | Whitelisting backend tools through the filter |

## Development

```bash
cd packages/python
pip install -e ".[dev]"
pytest
pytest --cov=langchain_task_steering
```

## License

MIT — see [LICENSE](../../LICENSE).
