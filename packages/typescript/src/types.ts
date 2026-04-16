/**
 * Public types for langchain-task-steering.
 */

/**
 * Lifecycle states for a task.
 */
export enum TaskStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETE = 'complete',
  ABORTED = 'aborted',
}

/**
 * Signal from `onComplete` to abort all remaining tasks.
 *
 * Return an `AbortAll` instance from {@link TaskMiddleware.onComplete} (or
 * `aOnComplete`) to indicate that, after this task completes, all remaining
 * `pending` / `in_progress` tasks should be marked `aborted`.
 *
 * The task that returned `AbortAll` itself still completes successfully —
 * `onComplete` inspects the post-transition state and decides the downstream
 * consequence. The middleware enforces the business rule; the agent doesn't
 * have to reason about whether to stop.
 *
 * In workflow mode, `AbortAll` also deactivates the active workflow, so the
 * agent returns to catalog mode and can activate another workflow or chat
 * freely.
 *
 * The `reason` is surfaced to the agent in the `update_task_status` tool
 * response so it knows what happened and can respond naturally.
 */
export class AbortAll {
  readonly reason: string
  /** Nominal-type marker so `instanceof` works across bundlers. */
  readonly __abortAll: true = true
  constructor(reason: string) {
    this.reason = reason
  }
}

export function isAbortAll(value: unknown): value is AbortAll {
  return (
    value instanceof AbortAll ||
    (typeof value === 'object' && value !== null && (value as AbortAll).__abortAll === true)
  )
}

/**
 * Configuration for post-completion message summarization.
 *
 * Attached to a Task via `Task.summarize`. When the task transitions to
 * `complete`, the middleware replaces task messages according to the chosen mode.
 *
 * Modes:
 *
 * `"replace"` — Removes **all** messages produced during the task and inserts
 * a single summary whose content is `content`. Use this when you already know
 * the summary (e.g. a static acknowledgment).
 *
 * `"summarize"` — Feeds the task messages to an LLM and replaces every
 * AI/Tool message produced during the task with the LLM's summary.
 * HumanMessages are preserved.
 */
export interface TaskSummarization {
  mode: 'replace' | 'summarize'
  /** Replacement text for `"replace"` mode (required when mode is `"replace"`). */
  content?: string
  /**
   * Chat model for `"summarize"` mode. Any object with
   * `invoke(messages)` and optionally `ainvoke(messages)` returning
   * `{ content: string }`. Falls back to `TaskSteeringMiddleware(model=...)`.
   */
  model?: unknown
  /** Custom prompt appended after the task messages when calling the summarization model. */
  prompt?: string
  /**
   * If true (default), strip the text content from the complete-transition
   * AIMessage, keeping only its tool_calls. The text is redundant once the
   * summary is in the ToolMessage.
   */
  trimCompleteMessage?: boolean
}

/**
 * Validate a TaskSummarization config.
 * Throws if mode is `"replace"` and `content` is not provided.
 */
export function validateTaskSummarization(cfg: TaskSummarization): void {
  if (cfg.mode === 'replace' && (cfg.content == null || cfg.content === undefined)) {
    throw new Error("TaskSummarization(mode='replace') requires 'content'.")
  }
}

/**
 * State shape for task-steering middleware.
 */
export interface TaskSteeringState {
  messages: unknown[]
  taskStatuses?: Record<string, string>
  taskMessageStarts?: Record<string, number>
  nudgeCount?: number
  skillsMetadata?: SkillMetadata[]
  activeWorkflow?: string | null
  [key: string]: unknown
}

// ── Middleware API interfaces ─────────────────────────────
// These mirror the LangChain v1 Python middleware API.

export interface ContentBlock {
  type: string
  text?: string
}

export interface SystemMessageLike {
  content: string | ContentBlock[]
}

export function getContentBlocks(msg: SystemMessageLike): ContentBlock[] {
  if (typeof msg.content === 'string') {
    return [{ type: 'text', text: msg.content }]
  }
  return [...(msg.content as ContentBlock[])]
}

export interface ModelRequest {
  state: Record<string, unknown>
  systemMessage: SystemMessageLike
  tools: ToolLike[]
  modelSettings?: Record<string, unknown>
  override(overrides: {
    systemMessage?: SystemMessageLike
    tools?: ToolLike[]
    modelSettings?: Record<string, unknown>
  }): ModelRequest
}

export type ModelResponse = unknown

export interface ToolCallInfo {
  name: string
  args: Record<string, unknown>
  id: string
}

export interface ToolCallRequest {
  toolCall: ToolCallInfo
  state: Record<string, unknown>
}

export interface ToolLike {
  name: string
  description?: string
  [key: string]: unknown
}

// ── Result types ──────────────────────────────────────────

export interface ToolMessageResult {
  content: string
  toolCallId: string
}

export interface CommandResult {
  update: Record<string, unknown>
}

// ── Handlers ──────────────────────────────────────────────

export type ToolCallHandler = (request: ToolCallRequest) => ToolMessageResult | CommandResult

export type AsyncToolCallHandler = (
  request: ToolCallRequest
) => Promise<ToolMessageResult | CommandResult>

export type ModelCallHandler = (request: ModelRequest) => ModelResponse

export type AsyncModelCallHandler = (request: ModelRequest) => Promise<ModelResponse>

// ── Task Middleware ───────────────────────────────────────

/**
 * Base class for task-scoped middleware.
 *
 * Subclass this to add:
 * - Mid-task enforcement via `wrapToolCall`
 * - Extra prompt injection via `wrapModelCall`
 * - Completion validation via `validateCompletion`
 * - Lifecycle hooks via `onStart` / `onComplete`
 *
 * Hooks are only active when the owning task is IN_PROGRESS.
 */
export class TaskMiddleware {
  stateSchema?: Record<string, unknown>

  validateCompletion(_state: Record<string, unknown>): string | null {
    return null
  }

  /**
   * Called after the task transitions to in_progress.
   *
   * Optionally return a record of state updates to merge into the
   * transition Command. If `messages` appears in both the returned
   * record and the existing Command update, the arrays are **appended**
   * (not overwritten) so the transition ToolMessage is preserved.
   * Return `undefined` / `void` (default) for no state changes.
   *
   * Note: `state` contains the *projected* post-transition
   * `taskStatuses` but all other fields reflect the pre-transition
   * snapshot (the Command has not been applied to the graph yet).
   */
  onStart(_state: Record<string, unknown>): Record<string, unknown> | void {}

  /**
   * Called after the task transitions to complete (after validation).
   *
   * Return value:
   *
   * - `undefined` / `void` (default): no state changes.
   * - `Record<string, unknown>`: state updates to merge into the transition
   *   Command. If `messages` appears in both the returned record and the
   *   existing Command update, the arrays are **appended** (not overwritten)
   *   so the transition ToolMessage is preserved.
   * - {@link AbortAll}: abort every remaining `pending` / `in_progress`
   *   task. In workflow mode, the active workflow is also deactivated.
   *   The `reason` is surfaced to the agent in the tool-response message.
   *
   * Note: `state` contains the *projected* post-transition `taskStatuses`
   * but all other fields reflect the pre-transition snapshot (the Command
   * has not been applied to the graph yet).
   */
  onComplete(_state: Record<string, unknown>): Record<string, unknown> | AbortAll | void {}

  /**
   * Async version of `validateCompletion`.
   * Override for validation requiring async I/O. Default delegates to sync.
   */
  async aValidateCompletion(state: Record<string, unknown>): Promise<string | null> {
    return this.validateCompletion(state)
  }

  /**
   * Async version of `onStart`. Default delegates to sync.
   */
  async aOnStart(state: Record<string, unknown>): Promise<Record<string, unknown> | void> {
    return this.onStart(state)
  }

  /**
   * Async version of `onComplete`. Default delegates to sync.
   * May return {@link AbortAll} to abort remaining tasks.
   */
  async aOnComplete(
    state: Record<string, unknown>
  ): Promise<Record<string, unknown> | AbortAll | void> {
    return this.onComplete(state)
  }

  wrapToolCall?(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult

  wrapModelCall?(request: ModelRequest, handler: ModelCallHandler): ModelResponse
}

// ── Skill metadata ──────────────────────────────────────

/**
 * Metadata parsed from a SKILL.md frontmatter.
 *
 * Compatible with deepagents' `SkillMetadata` — the two are
 * interchangeable via structural subtyping.
 */
export interface SkillMetadata {
  name: string
  description: string
  path: string
  license?: string | null
  compatibility?: string | null
  metadata?: Record<string, string>
  allowedTools?: string[]
}

// ── Task definition ──────────────────────────────────────

/**
 * A single task in an ordered pipeline.
 */
/**
 * Anything with optional wrapModelCall/wrapToolCall that can serve as middleware.
 * Raw agent-level middleware objects are auto-wrapped in AgentMiddlewareAdapter.
 */
export type TaskMiddlewareInput =
  | TaskMiddleware
  | { wrapModelCall?: unknown; wrapToolCall?: unknown; tools?: ToolLike[] }

export interface Task {
  name: string
  instruction: string
  tools: ToolLike[]
  middleware?: TaskMiddlewareInput | TaskMiddlewareInput[]
  /** Skill names available when this task is IN_PROGRESS. */
  skills?: string[]
  /** Optional post-completion summarization config. */
  summarize?: TaskSummarization
  /**
   * Per-task overrides for `ModelRequest.modelSettings`, applied only while
   * this task is `IN_PROGRESS`. Shallow-merged on top of any settings
   * already present on the request (task keys win). Forwarded as kwargs to
   * the chat model's invoke call, so any kwarg the provider accepts is
   * valid (e.g. `reasoning_effort` for OpenAI, `thinking` for Anthropic,
   * `additional_model_request_fields` for Bedrock Converse). Provider-
   * nested configs are replaced, not deep-merged — restate the full block
   * if you need to change a leaf.
   */
  modelSettings?: Record<string, unknown>
}

/**
 * A named, self-describing wrapper around a task list.
 *
 * The agent sees a catalog of available workflows and activates one on
 * demand via the `activate_workflow` tool.
 */
export interface Workflow {
  /** Unique workflow identifier. */
  name: string
  /** Shown in the catalog view so the agent can decide which workflow to activate. */
  description: string
  /** Ordered list of Task definitions for this workflow. */
  tasks: Task[]
  /** Tools available across all tasks when this workflow is active. */
  globalTools?: ToolLike[]
  /** Skill names available across all tasks when this workflow is active. */
  globalSkills?: string[]
  /** If true (default), tasks must be completed in the order they are defined. */
  enforceOrder?: boolean
  /**
   * Task names that must be completed before the workflow can be considered done.
   * Defaults to all tasks (`['*']`). Pass `null` for no required tasks.
   */
  requiredTasks?: readonly string[] | null
  /**
   * If true, the agent can deactivate this workflow even while a task is in progress.
   * Default false blocks deactivation until the active task is completed.
   */
  allowDeactivateInProgress?: boolean
}
