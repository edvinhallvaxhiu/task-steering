/**
 * TaskSteeringMiddleware — implicit state machine for LangChain agents.
 */

import {
  TaskStatus,
  TaskMiddleware,
  getContentBlocks,
  type Task,
  type TaskMiddlewareInput,
  type TaskSteeringState,
  type ToolLike,
  type ModelRequest,
  type ModelResponse,
  type ModelCallHandler,
  type ToolCallRequest,
  type ToolCallHandler,
  type ToolMessageResult,
  type CommandResult,
  type ContentBlock,
} from './types.js'
import { AgentMiddlewareAdapter } from './adapter.js'

const TRANSITION_TOOL_NAME = 'update_task_status'
const REQUIRE_ALL = ['*']

export interface TaskSteeringMiddlewareConfig {
  tasks: Task[]
  globalTools?: ToolLike[]
  enforceOrder?: boolean
  requiredTasks?: string[] | null
  maxNudges?: number
}

/**
 * Implicit state machine middleware for ordered task execution.
 *
 * Provides:
 * - Ordered task pipeline with enforced transitions
 *   (pending -> in_progress -> complete).
 * - Per-task tool scoping — only the active task's tools are visible
 *   to the model.
 * - Dynamic system prompt injection — task status board and active
 *   task instruction appended before every model call.
 * - Completion validation via task-scoped middleware
 *   (`TaskMiddleware.validateCompletion`).
 * - Mid-task enforcement via task-scoped middleware hooks
 *   (`wrapToolCall`, `wrapModelCall`).
 */
export class TaskSteeringMiddleware {
  private readonly _tasks: Task[]
  private readonly _taskOrder: string[]
  private readonly _taskMap: Map<string, Task>
  private readonly _globalTools: ToolLike[]
  private readonly _enforceOrder: boolean
  private readonly _maxNudges: number
  private readonly _requiredTasks: Set<string>
  private readonly _transitionTool: ToolLike

  /** All tools registered by this middleware. */
  readonly tools: ToolLike[]

  constructor(config: TaskSteeringMiddlewareConfig) {
    const {
      tasks,
      globalTools = [],
      enforceOrder = true,
      requiredTasks = REQUIRE_ALL,
      maxNudges = 3,
    } = config

    if (tasks.length === 0) {
      throw new Error('At least one Task is required.')
    }

    const names = tasks.map((t) => t.name)
    const dupes = names.filter((n, i) => names.indexOf(n) !== i)
    if (dupes.length > 0) {
      const uniqueDupes = [...new Set(dupes)]
      throw new Error(`Duplicate task names: ${uniqueDupes.join(', ')}`)
    }

    // Normalize middleware: auto-wrap raw objects, compose lists
    for (const task of tasks) {
      task.middleware = normalizeMiddleware(task.middleware)
    }

    this._tasks = tasks
    this._taskOrder = tasks.map((t) => t.name)
    this._taskMap = new Map(tasks.map((t) => [t.name, t]))
    this._globalTools = globalTools
    this._enforceOrder = enforceOrder
    this._maxNudges = maxNudges

    // Resolve required tasks
    if (requiredTasks !== null && requiredTasks.includes('*')) {
      this._requiredTasks = new Set(names)
    } else if (requiredTasks !== null) {
      const unknown = requiredTasks.filter((n) => !names.includes(n))
      if (unknown.length > 0) {
        throw new Error(`Unknown required tasks: ${unknown.join(', ')}`)
      }
      this._requiredTasks = new Set(requiredTasks)
    } else {
      this._requiredTasks = new Set()
    }

    this._transitionTool = this._buildTransitionTool()

    // Auto-register all tools (deduplicated), including tools
    // contributed by task middleware adapters.
    const seen = new Set<string>()
    const allTools: ToolLike[] = []
    const candidates: ToolLike[] = [
      this._transitionTool,
      ...this._globalTools,
      ...tasks.flatMap((t) => t.tools),
      ...tasks.flatMap((t) => {
        const mwTools = (t.middleware as { tools?: ToolLike[] })?.tools
        return mwTools ?? []
      }),
    ]
    for (const t of candidates) {
      if (!seen.has(t.name)) {
        seen.add(t.name)
        allTools.push(t)
      }
    }
    this.tools = allTools
  }

  // ── Node-style hooks ──────────────────────────────────

  /**
   * Initialize task_statuses on first invocation.
   * Returns state updates, or null if already initialized.
   */
  beforeAgent(state: TaskSteeringState): Record<string, unknown> | null {
    if (state.taskStatuses == null) {
      const statuses: Record<string, string> = {}
      for (const t of this._tasks) {
        statuses[t.name] = TaskStatus.PENDING
      }
      return { taskStatuses: statuses }
    }
    return null
  }

  /**
   * Nudge the agent back if required tasks are incomplete.
   * Returns state updates with jumpTo, or null.
   */
  afterAgent(
    state: TaskSteeringState
  ): { jumpTo: string; nudgeCount: number; messages: unknown[] } | null {
    if (this._requiredTasks.size === 0) return null

    const statuses = this._getStatuses(state)
    const incomplete = this._taskOrder.filter(
      (name) => this._requiredTasks.has(name) && statuses[name] !== TaskStatus.COMPLETE
    )

    if (incomplete.length === 0) return null

    const nudgeCount = (state.nudgeCount as number) ?? 0
    if (nudgeCount >= this._maxNudges) return null

    const taskList = incomplete.join(', ')
    return {
      jumpTo: 'model',
      nudgeCount: nudgeCount + 1,
      messages: [
        {
          role: 'human',
          content: `You have not completed the following required tasks: ${taskList}. Please continue.`,
        },
      ],
    }
  }

  // ── Wrap-style hooks ──────────────────────────────────

  /**
   * Inject task pipeline prompt and scope tools per active task.
   */
  wrapModelCall(request: ModelRequest, handler: ModelCallHandler): ModelResponse {
    const statuses = this._getStatuses(request.state as TaskSteeringState)
    const activeName = this._activeTask(statuses)

    // 1. Append task pipeline block to system prompt
    const block = this._renderStatusBlock(statuses, activeName)
    const existingBlocks = request.systemMessage ? getContentBlocks(request.systemMessage) : []
    const newContent: ContentBlock[] = [...existingBlocks, { type: 'text', text: block }]

    // 2. Scope tools to active task + globals + transition tool
    const allowedNames = this._allowedToolNames(activeName)
    const scoped = request.tools.filter((t) => allowedNames.has(t.name))

    const modified = request.override({
      systemMessage: { content: newContent },
      tools: scoped,
    })

    // 3. Delegate to task-scoped middleware if present
    const taskMw = this._getTaskMiddleware(activeName)
    if (taskMw?.wrapModelCall) {
      return taskMw.wrapModelCall(modified, handler)
    }

    return handler(modified)
  }

  /**
   * Gate completion transitions, enforce tool scoping, and delegate.
   */
  wrapToolCall(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult {
    const statuses = this._getStatuses(request.state as TaskSteeringState)
    const activeName = this._activeTask(statuses)

    // Intercept update_task_status for completion validation + lifecycle
    if (request.toolCall.name === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as {
        task: string
        status: string
      }
      const taskName = args.task
      const target = args.status

      if (target === TaskStatus.COMPLETE && this._taskMap.has(taskName)) {
        const taskMw = this._getTaskMiddleware(taskName)
        if (taskMw && overridesMethod(taskMw, 'validateCompletion')) {
          const error = taskMw.validateCompletion(request.state)
          if (error) {
            return {
              content: `Cannot complete '${taskName}': ${error}. Address the issues then try again.`,
              toolCallId: request.toolCall.id,
            }
          }
        }
      }

      const result = handler(request)

      // Fire lifecycle hooks on successful transition.
      // Build post-transition state so hooks see the updated
      // taskStatuses (the Command hasn't been applied yet).
      if (this._isCommand(result) && this._taskMap.has(taskName)) {
        const taskMw = this._getTaskMiddleware(taskName)
        if (taskMw) {
          const updatedStatuses = { ...statuses, [taskName]: target }
          const postState = { ...request.state, taskStatuses: updatedStatuses }
          if (target === TaskStatus.IN_PROGRESS) {
            taskMw.onStart(postState)
          } else if (target === TaskStatus.COMPLETE) {
            taskMw.onComplete(postState)
          }
        }
      }

      return result
    }

    // Gate: reject tool calls not in scope for the active task
    const allowed = this._allowedToolNames(activeName)
    if (!allowed.has(request.toolCall.name)) {
      return {
        content: `Tool '${request.toolCall.name}' is not available for the current task.`,
        toolCallId: request.toolCall.id,
      }
    }

    // Delegate all other tool calls to active task's middleware
    const taskMw = this._getTaskMiddleware(activeName)
    if (taskMw?.wrapToolCall) {
      return taskMw.wrapToolCall(request, handler)
    }

    return handler(request)
  }

  /**
   * Execute the transition tool logic.
   * Use this as the handler for update_task_status calls.
   */
  executeTransition(
    args: { task: string; status: string },
    state: Record<string, unknown>,
    toolCallId: string
  ): ToolMessageResult | CommandResult {
    const { task, status } = args

    if (!this._taskOrder.includes(task)) {
      return {
        content: `Invalid task '${task}'. Must be one of: ${this._taskOrder.join(', ')}`,
        toolCallId,
      }
    }

    if (status !== TaskStatus.IN_PROGRESS && status !== TaskStatus.COMPLETE) {
      return {
        content: `Invalid status '${status}'. Must be 'in_progress' or 'complete'.`,
        toolCallId,
      }
    }

    const statuses: Record<string, string> = {
      ...((state.taskStatuses as Record<string, string>) ?? {}),
    }
    for (const t of this._taskOrder) {
      if (!(t in statuses)) statuses[t] = TaskStatus.PENDING
    }

    const current = statuses[task]

    // Enforce valid transitions: pending -> in_progress -> complete
    const validNext: Record<string, string> = {
      [TaskStatus.PENDING]: TaskStatus.IN_PROGRESS,
      [TaskStatus.IN_PROGRESS]: TaskStatus.COMPLETE,
    }

    const expected = validNext[current]
    if (expected !== status) {
      return {
        content: `Cannot transition '${task}' from '${current}' to '${status}'. Expected next: '${expected ?? 'N/A'}'.`,
        toolCallId,
      }
    }

    // Enforce ordering: all prior tasks must be complete
    if (this._enforceOrder && status === TaskStatus.IN_PROGRESS) {
      const idx = this._taskOrder.indexOf(task)
      for (let i = 0; i < idx; i++) {
        const prev = this._taskOrder[i]
        if (statuses[prev] !== TaskStatus.COMPLETE) {
          return {
            content: `Cannot start '${task}': '${prev}' is not complete yet. Order: ${this._taskOrder.join(' -> ')}.`,
            toolCallId,
          }
        }
      }
    }

    statuses[task] = status

    const display = Object.entries(statuses)
      .map(([k, v]) => `  ${k}: ${v}`)
      .join('\n')

    return {
      update: {
        taskStatuses: statuses,
        messages: [
          {
            role: 'tool',
            content: `Task '${task}' -> ${status}.\n\n${display}`,
            toolCallId,
          },
        ],
      },
    }
  }

  // ── Internal helpers ──────────────────────────────────

  private _getStatuses(state: Record<string, unknown>): Record<string, string> {
    const raw = (state.taskStatuses as Record<string, string>) ?? {}
    const result: Record<string, string> = {}
    for (const t of this._tasks) {
      result[t.name] = raw[t.name] ?? TaskStatus.PENDING
    }
    return result
  }

  private _activeTask(statuses: Record<string, string>): string | null {
    for (const name of this._taskOrder) {
      if (statuses[name] === TaskStatus.IN_PROGRESS) return name
    }
    return null
  }

  private _getTaskMiddleware(taskName: string | null): TaskMiddleware | undefined {
    if (!taskName) return undefined
    return this._taskMap.get(taskName)?.middleware
  }

  /** @internal */
  _allowedToolNames(activeName: string | null): Set<string> {
    const names = new Set<string>([TRANSITION_TOOL_NAME])
    for (const t of this._globalTools) names.add(t.name)
    if (activeName) {
      const task = this._taskMap.get(activeName)
      if (task) {
        for (const t of task.tools) names.add(t.name)
        // Include tools contributed by task middleware (e.g. adapters)
        const mwTools = (task.middleware as { tools?: ToolLike[] })?.tools
        if (mwTools) {
          for (const t of mwTools) names.add(t.name)
        }
      }
    }
    return names
  }

  /** @internal */
  _renderStatusBlock(statuses: Record<string, string>, active: string | null): string {
    const icons: Record<string, string> = {
      [TaskStatus.PENDING]: '[ ]',
      [TaskStatus.IN_PROGRESS]: '[>]',
      [TaskStatus.COMPLETE]: '[x]',
    }

    const lines: string[] = ['\n<task_pipeline>']
    for (const t of this._tasks) {
      const s = statuses[t.name]
      lines.push(`  ${icons[s]} ${t.name} (${s})`)
    }

    if (active) {
      const task = this._taskMap.get(active)!
      lines.push(`\n  <current_task name="${active}">`)
      lines.push(`    ${task.instruction}`)
      lines.push('  </current_task>')
    }

    if (this._enforceOrder) {
      const orderStr = this._taskOrder.join(' -> ')
      lines.push('\n  <rules>')
      lines.push(`    Required order: ${orderStr}`)
      lines.push('    Use update_task_status to advance. Do not skip tasks.')
      lines.push('  </rules>')
    }

    lines.push('</task_pipeline>')
    return lines.join('\n')
  }

  private _buildTransitionTool(): ToolLike {
    const taskNamesHint = this._taskOrder.map((n) => `'${n}'`).join(', ')

    return {
      name: TRANSITION_TOOL_NAME,
      description:
        "Transition a task to 'in_progress' or 'complete'. " +
        'Must be called ALONE — never in parallel with other tools. ' +
        `Tasks must follow the defined order. Task names: ${taskNamesHint}`,
    }
  }

  private _isCommand(result: ToolMessageResult | CommandResult): result is CommandResult {
    return 'update' in result
  }
}

// ── Middleware composition ─────────────────────────────────

function overridesMethod(mw: TaskMiddleware, method: keyof TaskMiddleware): boolean {
  return (
    (mw as Record<string, unknown>)[method] !== undefined &&
    (mw as Record<string, unknown>)[method] !==
      (TaskMiddleware.prototype as Record<string, unknown>)[method]
  )
}

class ComposedTaskMiddleware extends TaskMiddleware {
  private readonly _middlewares: TaskMiddleware[]
  readonly tools: ToolLike[]

  constructor(middlewares: TaskMiddleware[]) {
    super()
    this._middlewares = middlewares

    // Merge tools (deduplicated)
    const seen = new Set<string>()
    const merged: ToolLike[] = []
    for (const mw of middlewares) {
      for (const t of (mw as { tools?: ToolLike[] }).tools ?? []) {
        if (!seen.has(t.name)) {
          seen.add(t.name)
          merged.push(t)
        }
      }
    }
    this.tools = merged

    // Wire up wrapModelCall chain (first = outermost)
    const modelMws = middlewares.filter((mw) => mw.wrapModelCall)
    if (modelMws.length > 0) {
      this.wrapModelCall = (request: ModelRequest, handler: ModelCallHandler): ModelResponse => {
        let chain = handler
        for (let i = modelMws.length - 1; i >= 0; i--) {
          const outer = modelMws[i]
          const inner = chain
          chain = (r) => outer.wrapModelCall!(r, inner)
        }
        return chain(request)
      }
    }

    // Wire up wrapToolCall chain
    const toolMws = middlewares.filter((mw) => mw.wrapToolCall)
    if (toolMws.length > 0) {
      this.wrapToolCall = (
        request: ToolCallRequest,
        handler: ToolCallHandler
      ): ToolMessageResult | CommandResult => {
        let chain: ToolCallHandler = handler
        for (let i = toolMws.length - 1; i >= 0; i--) {
          const outer = toolMws[i]
          const inner = chain
          chain = (r) => outer.wrapToolCall!(r, inner)
        }
        return chain(request)
      }
    }
  }

  validateCompletion(state: Record<string, unknown>): string | null {
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'validateCompletion')) {
        const error = mw.validateCompletion(state)
        if (error) return error
      }
    }
    return null
  }

  onStart(state: Record<string, unknown>): void {
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'onStart')) {
        mw.onStart(state)
      }
    }
  }

  onComplete(state: Record<string, unknown>): void {
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'onComplete')) {
        mw.onComplete(state)
      }
    }
  }
}

function isValidMiddleware(mw: unknown): boolean {
  if (mw instanceof TaskMiddleware) return true
  if (typeof mw !== 'object' || mw === null) return false
  const obj = mw as Record<string, unknown>
  // Duck-type: has at least one wrap-style hook
  return typeof obj.wrapModelCall === 'function' || typeof obj.wrapToolCall === 'function'
}

function coerceMiddleware(mw: TaskMiddlewareInput): TaskMiddleware | undefined {
  if (mw instanceof TaskMiddleware) return mw
  if (isValidMiddleware(mw)) {
    return new AgentMiddlewareAdapter(mw as any)
  }
  console.warn(
    `[langchain-task-steering] Ignoring invalid task middleware of type ` +
      `${(mw as any)?.constructor?.name ?? typeof mw}. Expected TaskMiddleware ` +
      `or an object with wrap-style hooks (e.g. wrapModelCall).`
  )
  return undefined
}

function normalizeMiddleware(
  middleware: TaskMiddlewareInput | TaskMiddlewareInput[] | undefined
): TaskMiddleware | undefined {
  if (middleware == null) return undefined
  if (Array.isArray(middleware)) {
    const valid = middleware.map(coerceMiddleware).filter((m): m is TaskMiddleware => m != null)
    if (valid.length === 0) return undefined
    if (valid.length === 1) return valid[0]
    return new ComposedTaskMiddleware(valid)
  }
  return coerceMiddleware(middleware)
}
