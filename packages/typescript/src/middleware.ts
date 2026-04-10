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
  type SkillMetadata,
  type ToolLike,
  type ModelRequest,
  type ModelResponse,
  type ModelCallHandler,
  type AsyncModelCallHandler,
  type ToolCallRequest,
  type ToolCallHandler,
  type AsyncToolCallHandler,
  type ToolMessageResult,
  type CommandResult,
  type ContentBlock,
} from './types.js'
import { AgentMiddlewareAdapter } from './adapter.js'

const TRANSITION_TOOL_NAME = 'update_task_status'
const REQUIRE_ALL = ['*'] as const

export interface TaskSteeringMiddlewareConfig {
  tasks: Task[]
  globalTools?: ToolLike[]
  enforceOrder?: boolean
  requiredTasks?: readonly string[] | null
  maxNudges?: number
  /** When true, known backend tools pass through the tool filter on all tasks. */
  backendToolsPassthrough?: boolean
  /** Override the default backend tools whitelist. `null`/`undefined` uses DEFAULT_BACKEND_TOOLS. */
  backendTools?: ReadonlySet<string> | null
  /** Skill names available regardless of active task. */
  globalSkills?: string[]
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
  /** Known backend tool names — whitelisted when `backendToolsPassthrough` is enabled. */
  static readonly DEFAULT_BACKEND_TOOLS: ReadonlySet<string> = new Set([
    // FilesystemMiddleware
    'ls',
    'read_file',
    'write_file',
    'edit_file',
    'glob',
    'grep',
    'execute',
    // TodoListMiddleware
    'write_todos',
    // SubAgentMiddleware
    'task',
    // AsyncSubAgentMiddleware
    'start_async_task',
    'check_async_task',
    'update_async_task',
    'cancel_async_task',
    'list_async_tasks',
  ])

  private readonly _tasks: Task[]
  private readonly _taskOrder: string[]
  private readonly _taskMap: Map<string, Task>
  private readonly _globalTools: ToolLike[]
  private readonly _enforceOrder: boolean
  private readonly _maxNudges: number
  private readonly _requiredTasks: Set<string>
  private readonly _transitionTool: ToolLike

  // Backend tools passthrough
  private _backendToolsPassthrough: boolean
  private readonly _backendTools: ReadonlySet<string>

  // Task-scoped skills
  private readonly _globalSkills: string[]
  private readonly _skillsActive: boolean
  private readonly _skillRequiredTools: ReadonlySet<string>

  /** All tools registered by this middleware. */
  readonly tools: ToolLike[]

  constructor(config: TaskSteeringMiddlewareConfig) {
    const {
      tasks,
      globalTools = [],
      enforceOrder = true,
      requiredTasks = REQUIRE_ALL,
      maxNudges = 3,
      backendToolsPassthrough = false,
      backendTools = null,
      globalSkills = [],
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

    // Normalize middleware on shallow copies to avoid mutating the caller's objects
    const tasksCopy = tasks.map((t) => ({ ...t }))
    for (const task of tasksCopy) {
      task.middleware = normalizeMiddleware(task.middleware)
      task.tools = [...task.tools]
    }

    this._tasks = tasksCopy
    this._taskOrder = tasksCopy.map((t) => t.name)
    this._taskMap = new Map(tasksCopy.map((t) => [t.name, t]))
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

    // ── Backend tools passthrough ────────────────────────────
    this._backendTools =
      backendTools != null ? new Set(backendTools) : TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS
    this._backendToolsPassthrough = backendToolsPassthrough

    // ── Task-scoped skills ───────────────────────────────────
    // Skills are active if any task defines skills or globalSkills are set.
    // Skill metadata comes from state (e.g. loaded by SkillsMiddleware in
    // create_deep_agent). This middleware only filters, never loads.
    this._globalSkills = [...globalSkills]
    this._skillsActive =
      tasksCopy.some((t) => t.skills && t.skills.length > 0) || this._globalSkills.length > 0
    this._skillRequiredTools = this._skillsActive ? new Set(['read_file', 'ls']) : new Set()

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
   * Initialize task_statuses and load skills on first invocation.
   * Returns state updates, or null if already initialized.
   */
  beforeAgent(state: TaskSteeringState): Record<string, unknown> | null {
    const updates: Record<string, unknown> = {}

    if (state.taskStatuses == null) {
      const statuses: Record<string, string> = {}
      for (const t of this._tasks) {
        statuses[t.name] = TaskStatus.PENDING
      }
      updates.taskStatuses = statuses
    }

    return Object.keys(updates).length > 0 ? updates : null
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
          additional_kwargs: {
            task_steering: { kind: 'nudge', incomplete_tasks: incomplete },
          },
        },
      ],
    }
  }

  // ── Shared request preparation ─────────────────────────

  /**
   * Build the modified model request with pipeline prompt and scoped tools.
   * Shared between sync and async wrapModelCall.
   */
  private _prepareModelRequest(request: ModelRequest): {
    modified: ModelRequest
    activeName: string | null
  } {
    const statuses = this._getStatuses(request.state as TaskSteeringState)
    const activeName = this._activeTask(statuses)

    const block = this._renderStatusBlock(statuses, activeName, request.state)
    let existingBlocks = request.systemMessage ? getContentBlocks(request.systemMessage) : []

    // Strip SkillsMiddleware's global prompt injection — we replace it with
    // per-task scoped skills in the pipeline block.
    if (this._skillsActive) {
      existingBlocks = existingBlocks.filter(
        (b) => !(b.type === 'text' && b.text?.includes('## Skills System'))
      )
    }

    const newContent: ContentBlock[] = [...existingBlocks, { type: 'text', text: block }]

    const allowedNames = this._allowedToolNames(activeName)
    const scoped = request.tools.filter((t) => allowedNames.has(t.name))

    const modified = request.override({
      systemMessage: { content: newContent },
      tools: scoped,
    })

    return { modified, activeName }
  }

  /**
   * Pre-handler validation: reject starting a task when another is in progress.
   * Returns rejection message or null.
   */
  private _rejectConcurrentStart(
    request: ToolCallRequest,
    statuses: Record<string, string>,
    taskName: string,
    target: string
  ): ToolMessageResult | null {
    if (target !== TaskStatus.IN_PROGRESS) return null
    const active = this._taskOrder.filter((name) => statuses[name] === TaskStatus.IN_PROGRESS)
    if (active.length > 0) {
      return {
        content: `Cannot start '${taskName}': '${active[0]}' is already in progress. Complete it first.`,
        toolCallId: request.toolCall.id,
      }
    }
    return null
  }

  /** Reject tool calls not in scope for the active task. */
  private _gateTool(request: ToolCallRequest, activeName: string | null): ToolMessageResult | null {
    const allowed = this._allowedToolNames(activeName)
    if (!allowed.has(request.toolCall.name)) {
      return {
        content: `Tool '${request.toolCall.name}' is not available for the current task.`,
        toolCallId: request.toolCall.id,
      }
    }
    return null
  }

  /**
   * Fire sync lifecycle hooks after a successful transition.
   * Build post-transition state so hooks see the updated taskStatuses.
   */
  private _fireLifecycleHooks(
    request: ToolCallRequest,
    result: ToolMessageResult | CommandResult,
    statuses: Record<string, string>,
    taskName: string,
    target: string
  ): void {
    if (!this._isCommand(result) || !this._taskMap.has(taskName)) return
    const taskMw = this._getTaskMiddleware(taskName)
    if (!taskMw) return
    const updatedStatuses = { ...statuses, [taskName]: target }
    const postState = { ...request.state, taskStatuses: updatedStatuses }
    if (target === TaskStatus.IN_PROGRESS) {
      taskMw.onStart(postState)
    } else if (target === TaskStatus.COMPLETE) {
      taskMw.onComplete(postState)
    }
  }

  // ── Wrap-style hooks ──────────────────────────────────

  /**
   * Inject task pipeline prompt and scope tools per active task.
   */
  wrapModelCall(request: ModelRequest, handler: ModelCallHandler): ModelResponse {
    const { modified, activeName } = this._prepareModelRequest(request)

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

    if (request.toolCall.name === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as { task: string; status: string }
      const { task: taskName, status: target } = args

      const concurrentReject = this._rejectConcurrentStart(request, statuses, taskName, target)
      if (concurrentReject) return concurrentReject

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
      this._fireLifecycleHooks(request, result, statuses, taskName, target)
      return result
    }

    const gate = this._gateTool(request, activeName)
    if (gate) return gate

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
    if (current === TaskStatus.COMPLETE) {
      return {
        content: `Task '${task}' is already complete.`,
        toolCallId,
      }
    }

    const validNext: Record<string, string> = {
      [TaskStatus.PENDING]: TaskStatus.IN_PROGRESS,
      [TaskStatus.IN_PROGRESS]: TaskStatus.COMPLETE,
    }

    const expected = validNext[current]
    if (expected !== status) {
      return {
        content: `Cannot transition '${task}' from '${current}' to '${status}'. Expected next: '${expected}'.`,
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
        nudgeCount: 0,
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
    // After normalization in the constructor, middleware is always TaskMiddleware | undefined
    return this._taskMap.get(taskName)?.middleware as TaskMiddleware | undefined
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

    // Backend tools passthrough
    if (this._backendToolsPassthrough) {
      for (const t of this._backendTools) names.add(t)
    }

    // Skills auto-whitelist
    if (this._skillsActive) {
      for (const t of this._skillRequiredTools) names.add(t)
    }

    return names
  }

  /** Return skill names visible for the given active task. */
  private _allowedSkillNames(activeName: string | null): Set<string> {
    const names = new Set(this._globalSkills)
    if (activeName) {
      const task = this._taskMap.get(activeName)
      if (task?.skills) {
        for (const s of task.skills) names.add(s)
      }
    }
    return names
  }

  /** Return the effective backend tools whitelist. */
  getBackendTools(): ReadonlySet<string> {
    return this._backendTools
  }

  /** @internal */
  _renderStatusBlock(
    statuses: Record<string, string>,
    active: string | null,
    state?: Record<string, unknown>
  ): string {
    const icons: Record<string, string> = {
      [TaskStatus.PENDING]: '[ ]',
      [TaskStatus.IN_PROGRESS]: '[>]',
      [TaskStatus.COMPLETE]: '[x]',
    }

    const lines: string[] = ['\n<task_pipeline>']
    for (const t of this._tasks) {
      const s = statuses[t.name]
      lines.push(`  ${icons[s] ?? '[?]'} ${t.name} (${s})`)
    }

    if (active) {
      const task = this._taskMap.get(active)!
      lines.push(`\n  <current_task name="${active}">`)
      lines.push(`    ${task.instruction}`)
      lines.push('  </current_task>')
    }

    // ── Skill rendering ──────────────────────────────────────
    let hasVisibleSkills = false
    if (this._skillsActive && state != null) {
      const allSkills = (state.skillsMetadata as SkillMetadata[] | undefined) ?? []
      const allowedNames = this._allowedSkillNames(active)
      const visibleSkills = allSkills.filter((s) => allowedNames.has(s.name))
      if (visibleSkills.length > 0) {
        hasVisibleSkills = true
        lines.push('\n  <available_skills>')
        for (const skill of visibleSkills) {
          const desc = skill.description || 'No description.'
          lines.push(`    - ${skill.name}: ${desc} Path: ${skill.path}`)
        }
        lines.push('  </available_skills>')
      }
    }

    if (this._enforceOrder || hasVisibleSkills) {
      lines.push('\n  <rules>')
      if (this._enforceOrder) {
        const orderStr = this._taskOrder.join(' -> ')
        lines.push(`    Required order: ${orderStr}`)
      }
      lines.push('    Use update_task_status to advance. Do not skip tasks.')
      if (hasVisibleSkills) {
        lines.push('    To use a skill, read its SKILL.md file for full instructions.')
      }
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

  // ── Async wrap-style hooks ─────────────────────────────

  /**
   * Async version of wrapModelCall.
   */
  async awrapModelCall(
    request: ModelRequest,
    handler: AsyncModelCallHandler
  ): Promise<ModelResponse> {
    const { modified, activeName } = this._prepareModelRequest(request)

    const taskMw = this._getTaskMiddleware(activeName)
    if (taskMw?.wrapModelCall) {
      return taskMw.wrapModelCall(modified, handler as unknown as ModelCallHandler)
    }

    return handler(modified)
  }

  /**
   * Async version of wrapToolCall.
   * Uses aValidateCompletion, aOnStart, aOnComplete for async lifecycle.
   */
  async awrapToolCall(
    request: ToolCallRequest,
    handler: AsyncToolCallHandler
  ): Promise<ToolMessageResult | CommandResult> {
    const statuses = this._getStatuses(request.state as TaskSteeringState)
    const activeName = this._activeTask(statuses)

    if (request.toolCall.name === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as { task: string; status: string }
      const { task: taskName, status: target } = args

      const concurrentReject = this._rejectConcurrentStart(request, statuses, taskName, target)
      if (concurrentReject) return concurrentReject

      // Async completion validation
      if (target === TaskStatus.COMPLETE && this._taskMap.has(taskName)) {
        const taskMw = this._getTaskMiddleware(taskName)
        if (
          taskMw &&
          (overridesMethod(taskMw, 'aValidateCompletion') ||
            overridesMethod(taskMw, 'validateCompletion'))
        ) {
          const error = await taskMw.aValidateCompletion(request.state)
          if (error) {
            return {
              content: `Cannot complete '${taskName}': ${error}. Address the issues then try again.`,
              toolCallId: request.toolCall.id,
            }
          }
        }
      }

      const result = await handler(request)

      // Async lifecycle hooks
      if (this._isCommand(result) && this._taskMap.has(taskName)) {
        const taskMw = this._getTaskMiddleware(taskName)
        if (taskMw) {
          const updatedStatuses = { ...statuses, [taskName]: target }
          const postState = { ...request.state, taskStatuses: updatedStatuses }
          if (target === TaskStatus.IN_PROGRESS) {
            await taskMw.aOnStart(postState)
          } else if (target === TaskStatus.COMPLETE) {
            await taskMw.aOnComplete(postState)
          }
        }
      }

      return result
    }

    const gate = this._gateTool(request, activeName)
    if (gate) return gate

    const taskMw = this._getTaskMiddleware(activeName)
    if (taskMw?.wrapToolCall) {
      return taskMw.wrapToolCall(request, handler as unknown as ToolCallHandler)
    }

    return handler(request)
  }
}

// ── Middleware composition ─────────────────────────────────

function overridesMethod(mw: TaskMiddleware, method: keyof TaskMiddleware): boolean {
  return (
    (mw as unknown as Record<string, unknown>)[method] !== undefined &&
    (mw as unknown as Record<string, unknown>)[method] !==
      (TaskMiddleware.prototype as unknown as Record<string, unknown>)[method]
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

  async aValidateCompletion(state: Record<string, unknown>): Promise<string | null> {
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'aValidateCompletion') || overridesMethod(mw, 'validateCompletion')) {
        const error = await mw.aValidateCompletion(state)
        if (error) return error
      }
    }
    return null
  }

  async aOnStart(state: Record<string, unknown>): Promise<void> {
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'aOnStart') || overridesMethod(mw, 'onStart')) {
        await mw.aOnStart(state)
      }
    }
  }

  async aOnComplete(state: Record<string, unknown>): Promise<void> {
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'aOnComplete') || overridesMethod(mw, 'onComplete')) {
        await mw.aOnComplete(state)
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
