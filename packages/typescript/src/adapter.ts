/**
 * Adapter to use an AgentMiddleware-shaped object as a TaskMiddleware.
 */

import {
  TaskMiddleware,
  type ModelRequest,
  type ModelResponse,
  type ModelCallHandler,
  type ToolCallRequest,
  type ToolCallHandler,
  type ToolMessageResult,
  type CommandResult,
  type ToolLike,
} from './types.js'

/**
 * Shape of an agent-level middleware that the adapter can wrap.
 * Only the hooks that overlap with TaskMiddleware are required.
 */
export interface AgentMiddlewareLike {
  wrapModelCall?(request: ModelRequest, handler: ModelCallHandler): ModelResponse
  wrapToolCall?(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult
  tools?: ToolLike[]
}

/**
 * Wraps a standard agent-level middleware so it can be used at task scope.
 *
 * Forwards the hooks that overlap between the two interfaces:
 *
 * - `wrapModelCall` → delegates to the inner middleware
 * - `wrapToolCall`  → delegates to the inner middleware
 * - `tools`         → exposes the inner middleware's tools (merged
 *   with the task's own tools by `TaskSteeringMiddleware`)
 *
 * Agent-level hooks (`beforeAgent`, `afterAgent`) have no task-scope
 * equivalent and are **not** forwarded — use `onStart` / `onComplete`
 * for task lifecycle events.
 *
 * @example
 * ```ts
 * import { AgentMiddlewareAdapter } from 'langchain-task-steering'
 *
 * const task: Task = {
 *   name: 'research',
 *   instruction: '...',
 *   tools: [searchTool],
 *   middleware: new AgentMiddlewareAdapter(new SummarizationMiddleware()),
 * }
 * ```
 */
export class AgentMiddlewareAdapter extends TaskMiddleware {
  private readonly _inner: AgentMiddlewareLike

  /** Tools contributed by the inner middleware. */
  readonly tools: ToolLike[]

  constructor(inner: AgentMiddlewareLike) {
    super()
    this._inner = inner
    this.tools = inner.tools ? [...inner.tools] : []

    // Wire up optional hooks only if the inner middleware implements them
    if (inner.wrapModelCall) {
      this.wrapModelCall = (request: ModelRequest, handler: ModelCallHandler): ModelResponse => {
        return inner.wrapModelCall!(request, handler)
      }
    }

    if (inner.wrapToolCall) {
      this.wrapToolCall = (
        request: ToolCallRequest,
        handler: ToolCallHandler
      ): ToolMessageResult | CommandResult => {
        return inner.wrapToolCall!(request, handler)
      }
    }
  }
}
