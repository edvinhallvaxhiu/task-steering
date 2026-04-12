# langchain-task-steering

Implicit state-machine middleware for LangChain agents. Define ordered task pipelines with per-task tool scoping, per-task skill scoping, dynamic prompt injection, and composable validation.

Available for both **Python** and **TypeScript**.

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

### Python

```bash
pip install langchain-task-steering
```

**Requirements:** Python >= 3.10, `langchain >= 1.0.0`, `langgraph >= 0.4.0`

### TypeScript / JavaScript

```bash
npm install langchain-task-steering
```

## Quick start

### Python

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

### TypeScript

```typescript
import { TaskSteeringMiddleware, type Task, type ToolLike } from "langchain-task-steering";

const addItems: ToolLike = {
  name: "add_items",
  description: "Add items to the inventory.",
};

const categorize: ToolLike = {
  name: "categorize",
  description: "Assign items to categories.",
};

const pipeline = new TaskSteeringMiddleware({
  tasks: [
    {
      name: "collect",
      instruction: "Collect all relevant items from the user's input.",
      tools: [addItems],
    },
    {
      name: "categorize",
      instruction: "Organize the collected items into categories.",
      tools: [categorize],
    },
  ],
});

// pipeline.tools           — all tools to register with the agent
// pipeline.beforeAgent()   — call on first invocation to init state
// pipeline.wrapModelCall() — wrap model calls for prompt injection + tool scoping
// pipeline.wrapToolCall()  — wrap tool calls for validation + delegation
// pipeline.afterAgent()    — call after agent to check required tasks
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
| `beforeAgent` | Initializes `taskStatuses` in state. |
| `wrapModelCall` | Appends task status board + active task instruction to system prompt. Filters tools to only the active task's tools + globals + `update_task_status`. Delegates to task-scoped middleware if present. |
| `wrapToolCall` | Intercepts `update_task_status` — runs `validateCompletion` on the task's scoped middleware before allowing completion. Rejects out-of-scope tool calls. Delegates other tool calls to the active task's scoped middleware. |
| `afterAgent` | Checks if required tasks are complete. If not, nudges the agent back (up to `maxNudges` times). |
| `tools` | Auto-registers all task tools + globals + `update_task_status` with the agent. |

### Task lifecycle

```
PENDING ──> IN_PROGRESS ──> COMPLETE
```

- The agent drives transitions by calling `update_task_status(task, status)`.
- Transitions are enforced: `pending -> in_progress -> complete` only.
- When `enforceOrder` is true, a task cannot start until all preceding tasks are complete.
- On `complete`, the task's `middleware.validateCompletion(state)` runs first — rejection returns an error to the agent without completing the transition.

## Task summarization

When a task completes, its intermediate messages (tool calls, tool results, reasoning) can be compressed to save context window space. Two modes are available:

- **`replace`** — removes all task messages, injects a static string you provide into the transition `ToolMessage`.
- **`summarize`** — calls an LLM to produce a summary from the task messages, injects it into the transition `ToolMessage`. Only `AIMessage`/`ToolMessage` objects are removed; `HumanMessage` objects are preserved.

### Python

```python
from langchain_task_steering import Task, TaskSteeringMiddleware, TaskSummarization

model = ChatBedrockConverse(model="anthropic.claude-sonnet-4-6-v1", region_name="us-east-1")

pipeline = TaskSteeringMiddleware(
    tasks=[
        Task(
            name="research",
            instruction="Research the topic.",
            tools=[search],
            summarize=TaskSummarization(
                mode="summarize",
                # model is optional — falls back to middleware's model
                # model=custom_model,
                # prompt is optional — overrides the default HumanMessage
                # prompt="Summarize in bullet points.",
            ),
        ),
        Task(
            name="write",
            instruction="Write the report.",
            tools=[write],
            summarize=TaskSummarization(mode="replace", content="Research complete."),
        ),
    ],
    model=model,  # default model for summarize mode
)
```

### TypeScript

```typescript
const pipeline = new TaskSteeringMiddleware({
  tasks: [
    {
      name: 'research',
      instruction: 'Research the topic.',
      tools: [searchTool],
      summarize: { mode: 'summarize' },
    },
    {
      name: 'write',
      instruction: 'Write the report.',
      tools: [writeTool],
      summarize: { mode: 'replace', content: 'Research complete.' },
    },
  ],
  model: chatModel, // default model for summarize mode
})
```

### How it works

1. When a task transitions to `in_progress`, the middleware records the current message index.
2. When the task transitions to `complete`, messages between the start index and the completion are processed:
   - **Replace**: all task messages are removed via `RemoveMessage`.
   - **Summarize**: `AIMessage`/`ToolMessage` objects are removed; the LLM receives a `SystemMessage` (with task name + instruction), the flattened task messages (tool metadata stripped to plain text), and a `HumanMessage` instruction.
3. The summary is injected into the transition `ToolMessage` (e.g., `Task 'research' -> complete.\n\nTask summary:\n...`).
4. By default, the text content of the complete-transition `AIMessage` is also stripped (`trimCompleteMessage: true`), since it's redundant once the summary exists.

### Model resolution

The `model` for `summarize` mode is resolved in order: `TaskSummarization.model` > `TaskSteeringMiddleware.model`. If neither is set, summarization is skipped with a warning.

### TaskSummarization fields

| Field | Default | Description |
|---|---|---|
| `mode` | `"replace"` | `"replace"` or `"summarize"`. |
| `content` | — | Replacement text for `replace` mode (required). |
| `model` | `None` | Chat model for `summarize` mode. Falls back to middleware `model`. |
| `prompt` | `None` | Custom `HumanMessage` content for the summarizer. |
| `trimCompleteMessage` | `true` | Strip text from the complete-transition `AIMessage`. |

## Task-scoped middleware

Each task can have a `TaskMiddleware` that activates only when the task is `IN_PROGRESS`. This enables mid-task enforcement, not just completion gating.

### Python

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
```

### TypeScript

```typescript
import { TaskMiddleware, type ToolCallRequest, type ToolCallHandler, type ToolMessageResult, type CommandResult } from "langchain-task-steering";

class ThreatsMiddleware extends TaskMiddleware {
  constructor(private minThreats: number = 25) {
    super();
  }

  validateCompletion(state: Record<string, unknown>): string | null {
    const threats = (state.threats as unknown[]) ?? [];
    if (threats.length < this.minThreats) {
      return `Only ${threats.length} threats — need at least ${this.minThreats}.`;
    }
    return null;
  }

  wrapToolCall(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult {
    if (request.toolCall.name === "gap_analysis") {
      const threats = (request.state.threats as unknown[]) ?? [];
      if (threats.length < this.minThreats) {
        return {
          content: `Cannot run gap_analysis: ${threats.length}/${this.minThreats} threats.`,
          toolCallId: request.toolCall.id,
        };
      }
    }
    return handler(request);
  }
}
```

### TaskMiddleware hooks

| Method | When it runs | Purpose |
|---|---|---|
| `validateCompletion(state)` | Before `complete` transition | Return error string to reject, `null` to allow |
| `onStart(state)` | After successful `in_progress` transition | Side effects (logging, state init) |
| `onComplete(state)` | After successful `complete` transition | Side effects (trail capture, cleanup) |
| `wrapToolCall(request, handler)` | On every tool call during this task | Mid-task tool gating / modification |
| `wrapModelCall(request, handler)` | On every model call during this task | Extra prompt injection / request modification |
| `tools` *(property)* | At middleware construction | Extra tools to register and scope to this task |

## Using community middleware at task scope

Standard `AgentMiddleware` instances can be passed directly to a task — they're auto-wrapped in `AgentMiddlewareAdapter`. No import needed:

### Python

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
        Task(
            name="write",
            instruction="Write the final report.",
            tools=[write_tool],
        ),
    ],
)
```

### TypeScript

```typescript
import { TaskSteeringMiddleware } from "langchain-task-steering";

const pipeline = new TaskSteeringMiddleware({
  tasks: [
    {
      name: "research",
      instruction: "Research the topic thoroughly.",
      tools: [searchTool],
      middleware: summarizationMiddleware, // auto-wrapped
    },
  ],
});
```

The adapter forwards `wrapModelCall`, `wrapToolCall`, `tools`, and `state_schema` from the inner middleware. Agent-level hooks (`beforeAgent`, `afterAgent`) are not forwarded — use `onStart` / `onComplete` for task lifecycle events. Invalid middleware objects are warned and skipped.

## Middleware composition

Tasks accept a list of middleware, composed like LangChain's `create_agent(middleware=[...])`:

```python
from langchain.agents.middleware import SummarizationMiddleware
from langchain_task_steering import Task

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

```typescript
{
  name: "research",
  instruction: "Research the topic thoroughly.",
  tools: [searchTool],
  middleware: [summarizationMw, new ResearchValidator()],
}
```

Composition semantics:
- **Wrap-style hooks** (`wrapModelCall`, `wrapToolCall`): first = outermost wrapper.
- **`validateCompletion`**: all validators run; first error wins.
- **`onStart` / `onComplete`**: all fire in order.
- **`tools`**: merged from all middleware, deduplicated.

## Async support

All middleware hooks have async counterparts (`awrap_model_call`, `awrap_tool_call`, `abefore_agent`, `aafter_agent` in Python). Agents using `astream()` or `ainvoke()` are fully supported. The `AgentMiddlewareAdapter` also forwards async hooks from the inner middleware.

## Task-scoped skills

Skills are prompt-injected capabilities loaded from `SKILL.md` files. When configured, skills are scoped per task — just like tools.

`SkillsMiddleware` (in `create_deep_agent`) loads all skills into state. `TaskSteeringMiddleware` filters them per task:

```python
agent = create_deep_agent(
    backend=my_backend,
    skills=["/skills/user/", "/skills/project/"],
    middleware=[
        TaskSteeringMiddleware(
            tasks=[
                Task(name="research", instruction="Research the topic.",
                     tools=[search], skills=["web-research"]),
                Task(name="write", instruction="Write the report.",
                     tools=[write], skills=["report-writing"]),
            ],
            global_skills=["general-formatting"],
        ),
    ],
)
```

When skills are active, the model sees an `<available_skills>` section in the status block listing the skill name, description, and `SKILL.md` path. `read_file` and `ls` are auto-whitelisted for any task that has skills (its own or via `globalSkills`) so the model can read skill files.

## Backend tools passthrough

When the middleware is used alongside other middleware that contribute tools (e.g., filesystem, subagent), those tools get filtered out by tool scoping. Backend tools passthrough lets known backend tools pass through automatically.

```python
pipeline = TaskSteeringMiddleware(tasks=[...], backend_tools_passthrough=True)
```

```typescript
const pipeline = new TaskSteeringMiddleware({ tasks: [...], backendToolsPassthrough: true })
```

`TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS` contains 14 known tool names (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `write_todos`, `task`, `start_async_task`, etc.). Override with `backendTools`/`backend_tools`. Inspect at runtime with `getBackendTools()`/`get_backend_tools()`.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `tasks` | *(required)* | Ordered list of `Task` definitions. |
| `globalTools` | `[]` | Tools available in every task. |
| `enforceOrder` | `true` | Require tasks to be completed in definition order. |
| `requiredTasks` | `["*"]` | Tasks that must be completed before the agent can exit. `["*"]` = all, `null` = none, or a list of task names. |
| `maxNudges` | `3` | Max times the agent is nudged to complete required tasks before being allowed to exit. |
| `globalSkills` | `[]` | Skill names available regardless of active task. |
| `backendToolsPassthrough` | `false` | Whitelist known backend tools through the tool filter. |
| `backendTools` | `null` | Override `DEFAULT_BACKEND_TOOLS`. `null` uses the built-in set. |
| `model` | `None` | Default chat model for `TaskSummarization(mode="summarize")`. |

### Task fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique identifier (used in prompts and state). |
| `instruction` | yes | Injected into system prompt when this task is active. |
| `tools` | yes | Tools visible when this task is `IN_PROGRESS`. |
| `middleware` | no | Scoped middleware — a `TaskMiddleware`, `AgentMiddleware` (auto-wrapped), or a list of them. Only active during this task. |
| `skills` | no | Skill names available when this task is `IN_PROGRESS`. |
| `summarize` | no | Post-completion summarization config. See [Task summarization](#task-summarization). |

## Development

### Python

```bash
cd packages/python
pip install -e ".[dev]"
pytest
pytest --cov=langchain_task_steering
```

### TypeScript

```bash
cd packages/typescript
npm install
npm test
npm run build
```

## Project structure

```
langchain-task-steering/
  packages/
    python/
      src/langchain_task_steering/
        __init__.py          # Public exports
        types.py             # Task, TaskMiddleware, TaskStatus, SkillMetadata
        middleware.py        # TaskSteeringMiddleware + composition
        adapter.py           # AgentMiddlewareAdapter
        _hooks.py            # Dynamic hook discovery from AgentMiddleware
        _skills.py           # Skill loading utilities (YAML parsing, backend I/O)
      tests/
        conftest.py          # Fixtures and mock objects
        test_middleware.py    # Core middleware tests
        test_skills.py       # Skill loading/parsing tests
        test_task_skills.py  # Task-scoped skills integration tests
        test_backend_passthrough.py  # Backend tools passthrough tests
      examples/
        simple_agent.py      # End-to-end example with Bedrock
      pyproject.toml
    typescript/
      src/
        index.ts             # Public exports
        types.ts             # Task, TaskMiddleware, TaskStatus, SkillMetadata
        middleware.ts         # TaskSteeringMiddleware implementation
        adapter.ts           # AgentMiddlewareAdapter
        skills.ts            # Skill loading utilities (frontmatter parsing, backend I/O)
      tests/
        middleware.test.ts   # Core middleware tests
        skills.test.ts       # Skill loading/parsing tests
        task-skills.test.ts  # Task-scoped skills integration tests
        backend-passthrough.test.ts  # Backend tools passthrough tests
      examples/
        simple-agent.ts      # Example usage
      package.json
      tsconfig.json
  .github/
    workflows/
      publish.yml            # PyPI + npm publish on release
  LICENSE
  README.md
```

## License

MIT — see [LICENSE](LICENSE).
