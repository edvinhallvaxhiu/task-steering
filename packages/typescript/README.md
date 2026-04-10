# langchain-task-steering

Implicit state-machine middleware for LangChain.js agents. Define ordered task pipelines with per-task tool scoping, dynamic prompt injection, and composable validation.

Also available for [Python](https://pypi.org/project/langchain-task-steering/).

```
PENDING ──> IN_PROGRESS ──> COMPLETE
```

The model drives its own transitions by calling `update_task_status`. The middleware enforces ordering, scopes tools, injects the active task's instruction into the system prompt, and gates completion via pluggable validators.

## When to use this

| Scenario                                    |       task-steering       |      LangGraph explicit workflows      |
| ------------------------------------------- | :-----------------------: | :------------------------------------: |
| Linear task pipeline (A then B then C)      |       **Best fit**        |  Verbose — one node + edges per task   |
| Per-task tool scoping                       |       **Built-in**        | Manual — separate tool lists per node  |
| Dynamic tasks from config / DB              | **Easy** — tasks are data | Hard — graph is compiled at build time |
| Branching / parallel execution              |       Not supported       |    **Built-in** — edges + `Send()`     |
| Per-task human-in-the-loop interrupts       |       Not supported       | **Built-in** — `interrupt()` per node  |
| Complex orchestration with retries / cycles |       Not supported       |    **Built-in** — conditional edges    |

**Rule of thumb:** If your tasks are sequential and tool-scoped, use task-steering. If you need branching, parallelism, or per-task interrupts, use explicit LangGraph workflows.

## Install

```bash
npm i langchain-task-steering
```

Zero runtime dependencies. Works with any LangChain.js model provider (`@langchain/anthropic`, `@langchain/aws`, `@langchain/openai`, etc.).

## Quick start

```typescript
import { TaskSteeringMiddleware, type Task, type ToolLike } from 'langchain-task-steering'

const addItems: ToolLike = {
  name: 'add_items',
  description: 'Add items to the inventory.',
}

const categorize: ToolLike = {
  name: 'categorize',
  description: 'Assign items to categories.',
}

const pipeline = new TaskSteeringMiddleware({
  tasks: [
    {
      name: 'collect',
      instruction: "Collect all relevant items from the user's input.",
      tools: [addItems],
    },
    {
      name: 'categorize',
      instruction: 'Organize the collected items into categories.',
      tools: [categorize],
    },
  ],
})
```

The middleware exposes hooks you integrate into your agent loop:

```typescript
pipeline.tools // all tools to register with the agent
pipeline.beforeAgent() // call on first invocation to init state
pipeline.wrapModelCall() // wrap model calls for prompt injection + tool scoping
pipeline.wrapToolCall() // wrap tool calls for validation + delegation
pipeline.afterAgent() // call after agent to check required tasks
```

See [`examples/simple-agent.ts`](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/packages/typescript/examples/simple-agent.ts) for a full end-to-end example with Bedrock.

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

| Hook                                         | Behavior                                                                                                                                                                                                                    |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `beforeAgent(state)`                         | Initializes `taskStatuses` in state. When skills are configured, loads skill metadata from the backend and stores it in state.                                                                                              |
| `wrapModelCall(request, handler)`            | Appends task status board + active task instruction to system prompt. Filters tools to only the active task's tools + globals + `update_task_status`. Delegates to task-scoped middleware if present.                       |
| `wrapToolCall(request, handler)`             | Intercepts `update_task_status` — runs `validateCompletion` on the task's scoped middleware before allowing completion. Rejects out-of-scope tool calls. Delegates other tool calls to the active task's scoped middleware. |
| `afterAgent(state)`                          | Checks if required tasks are complete. If not, returns a nudge message (up to `maxNudges` times).                                                                                                                           |
| `executeTransition(args, state, toolCallId)` | Handles the `update_task_status` tool call — validates transitions and ordering, returns state updates.                                                                                                                     |
| `tools`                                      | Auto-registers all task tools + globals + `update_task_status`.                                                                                                                                                             |

### Task lifecycle

```
PENDING ──> IN_PROGRESS ──> COMPLETE
```

- The agent drives transitions by calling `update_task_status(task, status)`.
- Transitions are enforced: `pending -> in_progress -> complete` only.
- When `enforceOrder` is true, a task cannot start until all preceding tasks are complete.
- On `complete`, the task's `middleware.validateCompletion(state)` runs first — rejection returns an error to the agent without completing the transition.

## Task-scoped middleware

Each task can have a `TaskMiddleware` that activates only when the task is `IN_PROGRESS`. This enables mid-task enforcement, not just completion gating.

```typescript
import {
  TaskMiddleware,
  type ToolCallRequest,
  type ToolCallHandler,
  type ToolMessageResult,
  type CommandResult,
} from 'langchain-task-steering'

class ThreatsMiddleware extends TaskMiddleware {
  constructor(private minThreats: number = 25) {
    super()
  }

  validateCompletion(state: Record<string, unknown>): string | null {
    const threats = (state.threats as unknown[]) ?? []
    if (threats.length < this.minThreats) {
      return `Only ${threats.length} threats — need at least ${this.minThreats}.`
    }
    return null
  }

  wrapToolCall(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult {
    if (request.toolCall.name === 'gap_analysis') {
      const threats = (request.state.threats as unknown[]) ?? []
      if (threats.length < this.minThreats) {
        return {
          content: `Cannot run gap_analysis: ${threats.length}/${this.minThreats} threats.`,
          toolCallId: request.toolCall.id,
        }
      }
    }
    return handler(request)
  }
}
```

### TaskMiddleware hooks

| Method                            | When it runs                              | Purpose                                        |
| --------------------------------- | ----------------------------------------- | ---------------------------------------------- |
| `validateCompletion(state)`       | Before `complete` transition              | Return error string to reject, `null` to allow |
| `aValidateCompletion(state)`      | Async version (used by `awrapToolCall`)   | Default delegates to sync `validateCompletion` |
| `onStart(state)`                  | After successful `in_progress` transition | Side effects (logging, state init)             |
| `aOnStart(state)`                 | Async version (used by `awrapToolCall`)   | Default delegates to sync `onStart`            |
| `onComplete(state)`               | After successful `complete` transition    | Side effects (trail capture, cleanup)          |
| `aOnComplete(state)`              | Async version (used by `awrapToolCall`)   | Default delegates to sync `onComplete`         |
| `wrapToolCall(request, handler)`  | On every tool call during this task       | Mid-task tool gating / modification            |
| `wrapModelCall(request, handler)` | On every model call during this task      | Extra prompt injection / request modification  |
| `tools` _(property)_              | At middleware construction                | Extra tools to register and scope to this task |

## Using community middleware at task scope

Any object with `wrapModelCall` or `wrapToolCall` can be passed directly as task middleware — it's auto-wrapped in `AgentMiddlewareAdapter`:

```typescript
import { TaskSteeringMiddleware } from 'langchain-task-steering'

const pipeline = new TaskSteeringMiddleware({
  tasks: [
    {
      name: 'research',
      instruction: 'Research the topic thoroughly.',
      tools: [searchTool],
      middleware: summarizationMiddleware, // auto-wrapped
    },
  ],
})
```

The adapter forwards `wrapModelCall`, `wrapToolCall`, and `tools` from the inner middleware. Agent-level hooks (`beforeAgent`, `afterAgent`) are not forwarded. Invalid middleware objects are warned and skipped.

## Middleware composition

Tasks accept a list of middleware, composed like LangChain's `create_agent(middleware=[...])`:

```typescript
{
  name: 'research',
  instruction: 'Research the topic thoroughly.',
  tools: [searchTool],
  middleware: [summarizationMw, new ResearchValidator()],
}
```

Composition semantics:

- **Wrap-style hooks** (`wrapModelCall`, `wrapToolCall`): first = outermost wrapper.
- **`validateCompletion`**: all validators run; first error wins.
- **`onStart` / `onComplete`**: all fire in order.
- **`tools`**: merged from all middleware, deduplicated.

## Task-scoped skills

Skills are prompt-injected capabilities loaded from `SKILL.md` files. When configured, skills are scoped per task — just like tools.

`SkillsMiddleware` (in `create_deep_agent`) loads all skills into state. `TaskSteeringMiddleware` filters them per task — no `backend` or `skillSources` needed:

```typescript
createDeepAgent({
  backend: myBackend,
  skills: ['/skills/user/', '/skills/project/'],
  middleware: [
    new TaskSteeringMiddleware({
      tasks: [
        {
          name: 'research',
          instruction: 'Research the topic.',
          tools: [searchTool],
          skills: ['web-research', 'citation-format'],
        },
        {
          name: 'write_report',
          instruction: 'Write the report.',
          tools: [writeTool],
          skills: ['report-writing'],
        },
      ],
      globalSkills: ['general-formatting'],
    }),
  ],
})
```

### How it works

When skills are active, the model sees them in the status block:

```xml
<task_pipeline>
  [x] research (complete)
  [>] write_report (in_progress)

  <current_task name="write_report">
    Write the report.
  </current_task>

  <available_skills>
    - report-writing: Templates and structure for technical reports. Path: /skills/project/report-writing/SKILL.md
    - general-formatting: Standard formatting guidelines. Path: /skills/user/general-formatting/SKILL.md
  </available_skills>

  <rules>
    Required order: research -> write_report
    Use update_task_status to advance. Do not skip tasks.
    To use a skill, read its SKILL.md file for full instructions.
  </rules>
</task_pipeline>
```

When skills are active, `read_file` and `ls` are auto-whitelisted in the tool filter so the model can read `SKILL.md` files regardless of which task is active.

## Backend tools passthrough

When the middleware is used alongside other middleware that contribute tools (e.g., filesystem, subagent), those tools get filtered out by tool scoping unless explicitly added to `globalTools` or a task's `tools`. Backend tools passthrough lets known backend tools pass through the filter automatically.

```typescript
const pipeline = new TaskSteeringMiddleware({
  tasks: [...],
  backendToolsPassthrough: true, // whitelist known backend tools
})

// Inspect the whitelist
TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS
// → Set { 'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep',
//         'execute', 'write_todos', 'task', 'start_async_task', ... }

// Override the whitelist
new TaskSteeringMiddleware({
  tasks: [...],
  backendToolsPassthrough: true,
  backendTools: new Set(['read_file', 'write_file', 'my_custom_tool']),
})

// Inspect at runtime
pipeline.getBackendTools() // → the effective whitelist
```

No `backend` is required for passthrough — it just whitelists tool names in the filter.

## Configuration

```typescript
const pipeline = new TaskSteeringMiddleware({
  tasks: [...],                    // required — ordered Task list
  globalTools: [],                 // tools available in every task
  enforceOrder: true,              // require tasks in definition order
  requiredTasks: ['*'],            // ['*'] = all, null = none, or list of names
  maxNudges: 3,                    // max nudge attempts before allowing exit
  globalSkills: [],                // skill names available in all tasks
  backendToolsPassthrough: false,  // whitelist known backend tools
  backendTools: null,              // override DEFAULT_BACKEND_TOOLS
})
```

### Task fields

| Field         | Required | Description                                                                                                                      |
| ------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | yes      | Unique identifier (used in prompts and state).                                                                                   |
| `instruction` | yes      | Injected into system prompt when this task is active.                                                                            |
| `tools`       | yes      | Tools visible when this task is `IN_PROGRESS`.                                                                                   |
| `middleware`  | no       | Scoped middleware — a `TaskMiddleware`, agent middleware object (auto-wrapped), or a list of them. Only active during this task. |
| `skills`      | no       | Skill names available when this task is `IN_PROGRESS`. Requires `backend` + `skillSources` on the middleware.                    |

## Agent integration

The middleware provides hooks — you wire them into your agent loop. Here's the pattern with `@langchain/aws`:

```typescript
import { ChatBedrockConverse } from '@langchain/aws'
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'

const model = new ChatBedrockConverse({
  model: 'us.anthropic.claude-sonnet-4-6',
  region: 'us-east-1',
})

// 1. Init state
const state = { messages: [] }
Object.assign(state, pipeline.beforeAgent(state))

// 2. Before each model call — get scoped tools + injected prompt
let modified
pipeline.wrapModelCall(request, (req) => {
  modified = req
  return {}
})
const response = await model.bindTools(scopedTools).invoke(messages)

// 3. For update_task_status calls — route through middleware
const result = pipeline.wrapToolCall(toolCallReq, (r) =>
  pipeline.executeTransition(r.toolCall.args, r.state, r.toolCall.id)
)

// 4. When agent stops — check required tasks
const nudge = pipeline.afterAgent(state)
if (nudge) {
  /* add nudge message, continue loop */
}
```

See [`examples/simple-agent.ts`](https://github.com/edvinhallvaxhiu/langchain-task-steering/blob/main/packages/typescript/examples/simple-agent.ts) for the complete working example.

## Development

```bash
cd packages/typescript
npm install
npm test
npm run build
```

## License

MIT — see [LICENSE](../../LICENSE).
