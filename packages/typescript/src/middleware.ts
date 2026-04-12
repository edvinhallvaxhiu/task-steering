/**
 * TaskSteeringMiddleware — implicit state machine for LangChain agents.
 */

import {
  TaskStatus,
  TaskMiddleware,
  getContentBlocks,
  validateTaskSummarization,
  type Task,
  type TaskMiddlewareInput,
  type TaskSteeringState,
  type TaskSummarization,
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
  /**
   * Default chat model for `TaskSummarization(mode="summarize")`.
   * Used when a task's `summarize.model` is not set. Any object with
   * `invoke(messages)` / `ainvoke(messages)` returning `{ content: string }`.
   */
  model?: unknown
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

  // Summarization model fallback
  private readonly _model: unknown

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
      model = null,
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

    // Validate summarization configs
    for (const task of tasksCopy) {
      if (task.summarize) {
        validateTaskSummarization(task.summarize)
      }
    }

    this._tasks = tasksCopy
    this._taskOrder = tasksCopy.map((t) => t.name)
    this._taskMap = new Map(tasksCopy.map((t) => [t.name, t]))
    this._globalTools = globalTools
    this._enforceOrder = enforceOrder
    this._maxNudges = maxNudges
    this._model = model

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
   * Initialize taskStatuses on first invocation.
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

    const allowedNames = this._allowedToolNames(activeName, request.state)
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
    const allowed = this._allowedToolNames(activeName, request.state)
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
   * Merges any returned state updates into the CommandResult.
   * Also handles task start recording and summarization on completion.
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
    if (taskMw) {
      const updatedStatuses = { ...statuses, [taskName]: target }
      const postState = { ...request.state, taskStatuses: updatedStatuses }

      let updates: Record<string, unknown> | void = undefined
      if (target === TaskStatus.IN_PROGRESS) {
        updates = taskMw.onStart(postState)
      } else if (target === TaskStatus.COMPLETE) {
        updates = taskMw.onComplete(postState)
      }

      if (updates) {
        const merged = { ...result.update, ...updates }
        if ('messages' in updates && 'messages' in result.update) {
          merged.messages = [
            ...(result.update.messages as unknown[]),
            ...(updates.messages as unknown[]),
          ]
        }
        result.update = merged
      }
    }

    if (target === TaskStatus.IN_PROGRESS) {
      this._recordTaskStart(result as CommandResult, request.state, taskName)
    } else if (target === TaskStatus.COMPLETE) {
      this._applySummarization(result as CommandResult, request.state, taskName)
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

  // ── Summarization ──────────────────────────────────────

  /**
   * Inject `taskMessageStarts[taskName]` into the Command update.
   * Mutates `result.update` in place.
   */
  private _recordTaskStart(
    result: CommandResult,
    state: Record<string, unknown>,
    taskName: string
  ): void {
    const task = this._taskMap.get(taskName)
    if (!task?.summarize) return

    const messages = (state.messages as unknown[]) ?? []
    const startIndex = messages.length + 1
    const update = result.update
    const starts: Record<string, number> = {
      ...((state.taskMessageStarts as Record<string, number>) ?? {}),
    }
    starts[taskName] = startIndex
    update.taskMessageStarts = starts
  }

  /**
   * Shared setup for sync/async summarization.
   *
   * Returns `{ task, cfg, taskMessages, removeOps, model }` or `null` if
   * summarization should be skipped.
   */
  private _prepareSummarization(
    state: Record<string, unknown>,
    taskName: string
  ): {
    task: Task
    cfg: TaskSummarization
    taskMessages: unknown[]
    removeOps: unknown[]
    model: unknown
  } | null {
    const task = this._taskMap.get(taskName)
    if (!task?.summarize) return null

    const messages = (state.messages as unknown[]) ?? []
    const starts = (state.taskMessageStarts as Record<string, number>) ?? {}
    const startIndex = starts[taskName]
    if (startIndex == null) return null

    // Exclude the complete-transition AIMessage (last element)
    const taskMessages = messages.slice(startIndex, messages.length - 1)
    if (taskMessages.length === 0) return null

    const cfg = task.summarize
    const model = cfg.model ?? this._model

    if (cfg.mode === 'replace') {
      const removeOps = taskMessages
        .filter((m) => msgId(m) != null)
        .map((m) => ({ id: msgId(m)!, _remove: true }))
      return { task, cfg, taskMessages, removeOps, model }
    } else {
      if (model == null) {
        console.warn(
          `[langchain-task-steering] Skipping summarization for task '${taskName}': ` +
            `no model configured. Set model on TaskSummarization or TaskSteeringMiddleware.`
        )
        return null
      }
      // Only remove AI/Tool messages (preserve Human messages)
      const removeOps = taskMessages
        .filter((m) => {
          const role = msgRole(m)
          return (role === 'ai' || role === 'tool') && msgId(m) != null
        })
        .map((m) => ({ id: msgId(m)!, _remove: true }))
      return { task, cfg, taskMessages, removeOps, model }
    }
  }

  /**
   * Inject remove ops and rewrite the transition ToolMessage with the summary.
   * Mutates `result.update` in place.
   */
  private _finalizeSummarization(
    result: CommandResult,
    state: Record<string, unknown>,
    taskName: string,
    cfg: TaskSummarization,
    removeOps: unknown[],
    summary: string
  ): void {
    const update = result.update
    const existingMsgs = [...((update.messages as unknown[]) ?? [])]

    // Rewrite the transition ToolMessage to include the summary
    for (let i = 0; i < existingMsgs.length; i++) {
      const msg = existingMsgs[i] as Record<string, unknown>
      if (msg.role === 'tool') {
        existingMsgs[i] = {
          ...msg,
          content: `${msg.content}\n\nTask summary:\n${summary}`,
        }
        break
      }
    }

    // Strip text from the complete-transition AIMessage, keeping only tool_calls
    const trimOps: unknown[] = []
    const trimComplete = cfg.trimCompleteMessage !== false // default true
    if (trimComplete) {
      const messages = (state.messages as unknown[]) ?? []
      if (messages.length > 0) {
        const completeAi = messages[messages.length - 1] as Record<string, unknown>
        if (completeAi.role === 'ai' && msgId(completeAi) != null) {
          trimOps.push({
            role: 'ai',
            content: '',
            id: msgId(completeAi),
            tool_calls: (completeAi.tool_calls as unknown[]) ?? [],
          })
        }
      }
    }

    update.messages = [...removeOps, ...trimOps, ...existingMsgs]

    // Clean up start index
    const starts: Record<string, number> = {
      ...((state.taskMessageStarts as Record<string, number>) ?? {}),
    }
    delete starts[taskName]
    update.taskMessageStarts = starts
  }

  /**
   * Sync summarization: replace task messages with a summary.
   * Mutates `result.update` in place.
   */
  private _applySummarization(
    result: CommandResult,
    state: Record<string, unknown>,
    taskName: string
  ): void {
    const prep = this._prepareSummarization(state, taskName)
    if (!prep) return

    const { task, cfg, taskMessages, removeOps, model } = prep

    let summary: string
    if (cfg.mode === 'replace') {
      summary = cfg.content!
    } else {
      const invokeMessages = TaskSteeringMiddleware._buildSummaryMessages(task, cfg, taskMessages)
      const response = (model as { invoke(msgs: unknown[]): { content: string } }).invoke(
        invokeMessages
      )
      summary = response.content
    }

    this._finalizeSummarization(result, state, taskName, cfg, removeOps, summary)
  }

  /**
   * Async summarization: replace task messages with a summary.
   * Mutates `result.update` in place.
   */
  private async _aapplySummarization(
    result: CommandResult,
    state: Record<string, unknown>,
    taskName: string
  ): Promise<void> {
    const prep = this._prepareSummarization(state, taskName)
    if (!prep) return

    const { task, cfg, taskMessages, removeOps, model } = prep

    let summary: string
    if (cfg.mode === 'replace') {
      summary = cfg.content!
    } else {
      const invokeMessages = TaskSteeringMiddleware._buildSummaryMessages(task, cfg, taskMessages)
      const response = await (
        model as { ainvoke(msgs: unknown[]): Promise<{ content: string }> }
      ).ainvoke(invokeMessages)
      summary = response.content
    }

    this._finalizeSummarization(result, state, taskName, cfg, removeOps, summary)
  }

  /**
   * Convert task messages to plain text for the summarization LLM.
   * Strips tool_calls / tool_call_id metadata so providers don't warn.
   */
  static _flattenForSummary(taskMessages: unknown[]): Array<{ role: string; content: string }> {
    const flat: Array<{ role: string; content: string }> = []
    for (const m of taskMessages) {
      const msg = m as Record<string, unknown>
      let content = msg.content as string | unknown[] | undefined
      let text: string

      if (Array.isArray(content)) {
        text = (content as Array<Record<string, unknown>>)
          .filter((b) => b.type === 'text')
          .map((b) => (b.text as string) ?? '')
          .join('\n')
      } else {
        text = String(content ?? '')
      }

      const role = msgRole(msg)
      if (role === 'ai') {
        // Include tool call names/args as text
        const toolCalls = (msg.tool_calls as Array<Record<string, unknown>>) ?? []
        for (const tc of toolCalls) {
          const name = (tc.name as string) ?? '?'
          const args = tc.args ?? {}
          text += `\n[called ${name}(${JSON.stringify(args)})]`
        }
        if (text.trim()) {
          flat.push({ role: 'ai', content: text.trim() })
        }
      } else if (role === 'tool') {
        const name = (msg.name as string) ?? 'tool'
        flat.push({ role: 'human', content: `[${name} result]: ${text}` })
      } else if (text.trim()) {
        flat.push({ role: 'human', content: text.trim() })
      }
    }
    return flat
  }

  /**
   * Build the full message list for the summarization LLM call.
   */
  static _buildSummaryMessages(
    task: Task,
    cfg: TaskSummarization,
    taskMessages: unknown[]
  ): Array<{ role: string; content: string }> {
    const system = {
      role: 'system',
      content:
        'You are summarizing a completed agent task.\n\n' +
        `Task name: ${task.name}\n` +
        `Task instruction: ${task.instruction}`,
    }
    const human = {
      role: 'human',
      content: cfg.prompt ?? 'Provide a concise summary of what was accomplished.',
    }
    const flat = TaskSteeringMiddleware._flattenForSummary(taskMessages)
    return [system, ...flat, human]
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
  _allowedToolNames(activeName: string | null, state?: Record<string, unknown>): Set<string> {
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
      const allowedSkills = this._allowedSkillNames(activeName)
      if (allowedSkills.size > 0) {
        for (const t of this._skillRequiredTools) names.add(t)
        // Whitelist tools declared by visible skills (allowedTools frontmatter)
        if (state != null) {
          const allSkills = (state.skillsMetadata as SkillMetadata[] | undefined) ?? []
          for (const skill of allSkills) {
            if (allowedSkills.has(skill.name) && skill.allowedTools) {
              for (const toolName of skill.allowedTools) names.add(toolName)
            }
          }
        }
      }
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
      const availableNames = new Set(allSkills.map((s) => s.name))
      const visibleSkills = allSkills.filter((s) => allowedNames.has(s.name))

      const missing = [...allowedNames].filter((n) => !availableNames.has(n))
      if (missing.length > 0) {
        console.warn(
          `[langchain-task-steering] Skill(s) ${missing.sort().join(', ')} referenced by ` +
            `task/global config but not found in skillsMetadata state. Check skill names ` +
            `and ensure skills are loaded (e.g. via SkillsMiddleware).`
        )
      }

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

    if (hasVisibleSkills) {
      lines.push('')
      lines.push('  <skill_usage>')
      lines.push('    **How to Use Skills (Progressive Disclosure):**')
      lines.push('')
      lines.push(
        '    Skills follow a progressive disclosure pattern - you see' +
          ' their name and description above, but only read full' +
          ' instructions when needed:'
      )
      lines.push('')
      lines.push(
        '    1. **Recognize when a skill applies**: Check if the' +
          " user's task matches a skill's description"
      )
      lines.push(
        "    2. **Read the skill's full instructions**: Use the path" +
          ' shown in the skill list above'
      )
      lines.push(
        "    3. **Follow the skill's instructions**: SKILL.md" +
          ' contains step-by-step workflows, best practices, and' +
          ' examples'
      )
      lines.push(
        '    4. **Access supporting files**: Skills may include' +
          ' helper scripts, configs, or reference docs - use absolute' +
          ' paths'
      )
      lines.push('')
      lines.push('    **When to Use Skills:**')
      lines.push("    - User's request matches a skill's domain")
      lines.push('    - You need specialized knowledge or structured workflows')
      lines.push('    - A skill provides proven patterns for complex tasks')
      lines.push('')
      lines.push('    **Executing Skill Scripts:**')
      lines.push(
        '    Skills may contain Python scripts or other executable' +
          ' files. Always use absolute paths from the skill list.'
      )
      lines.push('  </skill_usage>')
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

      // Async lifecycle hooks — merge returned updates into the Command
      if (this._isCommand(result) && this._taskMap.has(taskName)) {
        const taskMw = this._getTaskMiddleware(taskName)
        if (taskMw) {
          const updatedStatuses = { ...statuses, [taskName]: target }
          const postState = { ...request.state, taskStatuses: updatedStatuses }
          let updates: Record<string, unknown> | void = undefined
          if (target === TaskStatus.IN_PROGRESS) {
            updates = await taskMw.aOnStart(postState)
          } else if (target === TaskStatus.COMPLETE) {
            updates = await taskMw.aOnComplete(postState)
          }
          if (updates) {
            const merged = { ...result.update, ...updates }
            if ('messages' in updates && 'messages' in result.update) {
              merged.messages = [
                ...(result.update.messages as unknown[]),
                ...(updates.messages as unknown[]),
              ]
            }
            result.update = merged
          }
        }

        if (target === TaskStatus.IN_PROGRESS) {
          this._recordTaskStart(result, request.state, taskName)
        } else if (target === TaskStatus.COMPLETE) {
          await this._aapplySummarization(result, request.state, taskName)
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

// ── Message helpers ──────────────────────────────────────────

function msgId(m: unknown): string | undefined {
  return (m as Record<string, unknown>)?.id as string | undefined
}

function msgRole(m: unknown): string | undefined {
  return (m as Record<string, unknown>)?.role as string | undefined
}

// ── Middleware composition ─────────────────────────────────

function overridesMethod(mw: TaskMiddleware, method: keyof TaskMiddleware): boolean {
  return (
    (mw as unknown as Record<string, unknown>)[method] !== undefined &&
    (mw as unknown as Record<string, unknown>)[method] !==
      (TaskMiddleware.prototype as unknown as Record<string, unknown>)[method]
  )
}

/**
 * Merge lifecycle hook return values, appending `messages` arrays.
 */
function mergeHookUpdates(
  base: Record<string, unknown> | void,
  updates: Record<string, unknown> | void
): Record<string, unknown> | void {
  if (!updates) return base
  if (!base) return { ...updates }
  const merged = { ...base, ...updates }
  if ('messages' in base && 'messages' in updates) {
    merged.messages = [...(base.messages as unknown[]), ...(updates.messages as unknown[])]
  }
  return merged
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

  onStart(state: Record<string, unknown>): Record<string, unknown> | void {
    let merged: Record<string, unknown> | void = undefined
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'onStart')) {
        const updates = mw.onStart(state)
        merged = mergeHookUpdates(merged, updates)
      }
    }
    return merged
  }

  onComplete(state: Record<string, unknown>): Record<string, unknown> | void {
    let merged: Record<string, unknown> | void = undefined
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'onComplete')) {
        const updates = mw.onComplete(state)
        merged = mergeHookUpdates(merged, updates)
      }
    }
    return merged
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

  async aOnStart(state: Record<string, unknown>): Promise<Record<string, unknown> | void> {
    let merged: Record<string, unknown> | void = undefined
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'aOnStart') || overridesMethod(mw, 'onStart')) {
        const updates = await mw.aOnStart(state)
        merged = mergeHookUpdates(merged, updates)
      }
    }
    return merged
  }

  async aOnComplete(state: Record<string, unknown>): Promise<Record<string, unknown> | void> {
    let merged: Record<string, unknown> | void = undefined
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'aOnComplete') || overridesMethod(mw, 'onComplete')) {
        const updates = await mw.aOnComplete(state)
        merged = mergeHookUpdates(merged, updates)
      }
    }
    return merged
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
