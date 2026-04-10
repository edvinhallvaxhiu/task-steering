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

export type ModelCallHandler = (request: ModelRequest) => ModelResponse

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

  onStart(_state: Record<string, unknown>): void {}

  onComplete(_state: Record<string, unknown>): void {}

  wrapToolCall?(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult

  wrapModelCall?(request: ModelRequest, handler: ModelCallHandler): ModelResponse
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
}
