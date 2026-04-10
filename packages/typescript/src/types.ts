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
}

/**
 * State shape for task-steering middleware.
 */
export interface TaskSteeringState {
  messages: unknown[]
  taskStatuses?: Record<string, string>
  nudgeCount?: number
  skillsMetadata?: SkillMetadata[]
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
  override(overrides: { systemMessage?: SystemMessageLike; tools?: ToolLike[] }): ModelRequest
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
   * Note: `state` contains the *projected* post-transition
   * `taskStatuses` but all other fields reflect the pre-transition
   * snapshot (the Command has not been applied to the graph yet).
   */
  onStart(_state: Record<string, unknown>): void {}

  /**
   * Called after the task transitions to complete (after validation).
   *
   * Note: `state` contains the *projected* post-transition
   * `taskStatuses` but all other fields reflect the pre-transition
   * snapshot (the Command has not been applied to the graph yet).
   */
  onComplete(_state: Record<string, unknown>): void {}

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
  async aOnStart(state: Record<string, unknown>): Promise<void> {
    this.onStart(state)
  }

  /**
   * Async version of `onComplete`. Default delegates to sync.
   */
  async aOnComplete(state: Record<string, unknown>): Promise<void> {
    this.onComplete(state)
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
}
