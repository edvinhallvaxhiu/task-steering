# langchain-task-steering

Implicit state-machine middleware for LangChain.js agents. Define ordered task pipelines with per-task tool scoping, dynamic prompt injection, and composable validation.

Also available for [Python](https://pypi.org/project/langchain-task-steering/).

```
PENDING ──> IN_PROGRESS ──> COMPLETE
```

The model drives its own transitions by calling `update_task_status`. The middleware enforces ordering, scopes tools, injects the active task's instruction into the system prompt, and gates completion via pluggable validators.

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

## Agent integration

The middleware provides hooks — you wire them into your agent loop:

```typescript
import { ChatBedrockConverse } from '@langchain/aws'

const model = new ChatBedrockConverse({
  model: 'us.anthropic.claude-sonnet-4-6',
  region: 'us-east-1',
})

// 1. Init state
const state = { messages: [] }
Object.assign(state, pipeline.beforeAgent(state))

// 2. Before each model call — get scoped tools + injected prompt
pipeline.wrapModelCall(request, (req) => {
  // req has filtered tools + injected system prompt
  return model.bindTools(req.tools).invoke(req.messages)
})

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

## Documentation

Detailed documentation is shared across both Python and TypeScript packages. The concepts and behavior are identical — only the API surface differs by language.

| Topic                                                    | Description                                                           |
| -------------------------------------------------------- | --------------------------------------------------------------------- |
| [Task Mode](../../docs/task-mode.md)                     | Task lifecycle, hooks, tool scoping, required tasks, configuration    |
| [Workflow Mode](../../docs/workflow-mode.md)             | Dynamic workflow activation, catalog, human-in-the-loop, deactivation |
| [Task Middleware](../../docs/task-middleware.md)         | TaskMiddleware hooks, validation, composition, persistent state       |
| [Summarization](../../docs/summarization.md)             | Post-completion message compression (replace and summarize modes)     |
| [Skills](../../docs/skills.md)                           | Task-scoped skills from SKILL.md files                                |
| [Backend Passthrough](../../docs/backend-passthrough.md) | Whitelisting backend tools through the filter                         |

> **Note:** Code examples in the docs are primarily in Python. The TypeScript API mirrors the same design with camelCase naming (`enforceOrder`, `globalTools`, `validateCompletion`, etc.).

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
  model: chatModel,                // default model for TaskSummarization
})
```

### Task fields

| Field         | Required | Description                                                                                |
| ------------- | -------- | ------------------------------------------------------------------------------------------ |
| `name`        | yes      | Unique identifier (used in prompts and state).                                             |
| `instruction` | yes      | Injected into system prompt when this task is active.                                      |
| `tools`       | yes      | Tools visible when this task is `IN_PROGRESS`.                                             |
| `middleware`  | no       | Scoped middleware — a `TaskMiddleware`, agent middleware object (auto-wrapped), or a list. |
| `skills`      | no       | Skill names available when this task is `IN_PROGRESS`.                                     |
| `summarize`   | no       | Post-completion summarization config. See [Summarization](../../docs/summarization.md).    |

## Development

```bash
cd packages/typescript
npm install
npm test              # vitest run
npm run test:watch    # vitest in watch mode
npm run build         # tsup (CJS + ESM + .d.ts)
npm run lint          # tsc --noEmit
npm run format        # prettier --write
```

## License

MIT — see [LICENSE](../../LICENSE).
