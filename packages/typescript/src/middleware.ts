/**
 * TaskSteeringMiddleware — implicit state machine for LangChain agents.
 */

import {
  AbortAll,
  isAbortAll,
  TaskStatus,
  TaskMiddleware,
  getContentBlocks,
  validateTaskSummarization,
  type Task,
  type TaskMiddlewareInput,
  type TaskSteeringState,
  type TaskSummarization,
  type SkillMetadata,
  type Workflow,
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
const ACTIVATE_TOOL_NAME = 'activate_workflow'
const DEACTIVATE_TOOL_NAME = 'deactivate_workflow'
const REQUIRE_ALL = ['*'] as const

/** Statuses that count as "task is done" — neither blocks ordering nor needs nudging. */
const TERMINAL_STATUSES: ReadonlySet<string> = new Set([TaskStatus.COMPLETE, TaskStatus.ABORTED])
const isTerminal = (s: string): boolean => TERMINAL_STATUSES.has(s)

/** Status icons rendered in the `<task_pipeline>` block. Module-scope so renderStatusBlock doesn't rebuild it per model call. */
const STATUS_ICONS: Readonly<Record<string, string>> = {
  [TaskStatus.PENDING]: '[ ]',
  [TaskStatus.IN_PROGRESS]: '[>]',
  [TaskStatus.COMPLETE]: '[x]',
  [TaskStatus.ABORTED]: '[-]',
}

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

  /** @internal */ readonly _ctx: PipelineContext
  private readonly _maxNudges: number
  private readonly _transitionTool: ToolLike

  // Summarization model fallback
  private readonly _model: unknown

  // Backend tools passthrough
  private _backendToolsPassthrough: boolean
  private readonly _backendTools: ReadonlySet<string>

  /**
   * Skill names already warned about — keeps the per-render warning
   * from spamming logs every model call.
   */
  private readonly _warnedMissingSkills: Set<string> = new Set()

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

    const resolvedRequiredTasks = resolveRequiredTasks(requiredTasks, names)

    this._ctx = buildPipelineContext(tasksCopy, globalTools, enforceOrder, resolvedRequiredTasks, [
      ...globalSkills,
    ])
    this._maxNudges = maxNudges
    this._model = model

    // ── Backend tools passthrough ────────────────────────────
    this._backendTools =
      backendTools != null ? new Set(backendTools) : TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS
    this._backendToolsPassthrough = backendToolsPassthrough

    this._transitionTool = this._buildTransitionTool()

    // Auto-register all tools (deduplicated), including tools
    // contributed by task middleware adapters.
    const seen = new Set<string>()
    const allTools: ToolLike[] = []
    const candidates: ToolLike[] = [
      this._transitionTool,
      ...this._ctx.globalTools,
      ...this._ctx.tasks.flatMap((t) => t.tools),
      ...this._ctx.tasks.flatMap((t) => {
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
      for (const t of this._ctx.tasks) {
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
    if (this._ctx.requiredTasks.size === 0) return null

    const statuses = getStatuses(this._ctx, state)
    const incomplete = this._ctx.taskOrder.filter(
      (name) => this._ctx.requiredTasks.has(name) && !isTerminal(statuses[name])
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
    const statuses = getStatuses(this._ctx, request.state as TaskSteeringState)
    const activeName = getActiveTask(this._ctx, statuses)

    const block = renderStatusBlock(
      this._ctx,
      statuses,
      activeName,
      request.state,
      undefined,
      this._warnedMissingSkills
    )
    let existingBlocks = request.systemMessage ? getContentBlocks(request.systemMessage) : []

    // Strip SkillsMiddleware's global prompt injection — we replace it with
    // per-task scoped skills in the pipeline block.
    if (this._ctx.skillsActive) {
      existingBlocks = existingBlocks.filter(
        (b) => !(b.type === 'text' && b.text?.includes('## Skills System'))
      )
    }

    const newContent: ContentBlock[] = [...existingBlocks, { type: 'text', text: block }]

    const allowedNames = getAllowedToolNames(
      this._ctx,
      activeName,
      new Set(),
      this._backendToolsPassthrough,
      this._backendTools,
      request.state
    )
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
    const active = this._ctx.taskOrder.filter((name) => statuses[name] === TaskStatus.IN_PROGRESS)
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
    const allowed = getAllowedToolNames(
      this._ctx,
      activeName,
      new Set(),
      this._backendToolsPassthrough,
      this._backendTools,
      request.state
    )
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
    if (!this._isCommand(result) || !this._ctx.taskMap.has(taskName)) return

    // Aborted is user-driven: no lifecycle hooks, no summarization.
    if (target === TaskStatus.ABORTED) return

    const taskMw = getTaskMiddleware(this._ctx, taskName)
    if (taskMw) {
      const updatedStatuses = { ...statuses, [taskName]: target }
      const postState = { ...request.state, taskStatuses: updatedStatuses }

      let updates: Record<string, unknown> | AbortAll | void = undefined
      if (target === TaskStatus.IN_PROGRESS) {
        updates = taskMw.onStart(postState)
      } else if (target === TaskStatus.COMPLETE) {
        updates = taskMw.onComplete(postState)
      }

      if (isAbortAll(updates)) {
        applyAbortAll(result, updates, this._ctx)
        return
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

    const taskMw = getTaskMiddleware(this._ctx, activeName)
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
    const statuses = getStatuses(this._ctx, request.state as TaskSteeringState)
    const activeName = getActiveTask(this._ctx, statuses)

    if (request.toolCall.name === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as { task: string; status: string }
      const { task: taskName, status: target } = args

      const concurrentReject = this._rejectConcurrentStart(request, statuses, taskName, target)
      if (concurrentReject) return concurrentReject

      if (target === TaskStatus.COMPLETE && this._ctx.taskMap.has(taskName)) {
        const taskMw = getTaskMiddleware(this._ctx, taskName)
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

    const taskMw = getTaskMiddleware(this._ctx, activeName)
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
    return executeTaskTransition(
      args.task,
      args.status,
      this._ctx.taskOrder,
      this._ctx.enforceOrder,
      this._ctx.requiredTasks,
      state,
      toolCallId
    )
  }

  // ── Summarization ──────────────────────────────────────

  /**
   * Inject `taskMessageStarts[taskName]` into the Command update.
   * Mutates `result.update` in place.
   *
   * Always records (not just when summarization is configured) so the
   * abort-commitment check can detect whether any tool calls were made
   * during the task's in_progress window.
   */
  private _recordTaskStart(
    result: CommandResult,
    state: Record<string, unknown>,
    taskName: string
  ): void {
    recordTaskStart(result, state, taskName)
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
    const task = this._ctx.taskMap.get(taskName)
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
      const response = (
        model as { invoke(msgs: unknown[]): { content: string | unknown[] } }
      ).invoke(invokeMessages)
      summary = TaskSteeringMiddleware._extractResponseText(response.content)
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
        model as { ainvoke(msgs: unknown[]): Promise<{ content: string | unknown[] }> }
      ).ainvoke(invokeMessages)
      summary = TaskSteeringMiddleware._extractResponseText(response.content)
    }

    this._finalizeSummarization(result, state, taskName, cfg, removeOps, summary)
  }

  /**
   * Extract plain text from a model response content.
   * Handles extended-thinking models that return an array of content blocks
   * (e.g. reasoning_content + text blocks) instead of a plain string.
   */
  static _extractResponseText(content: string | unknown[]): string {
    if (typeof content === 'string') return content
    if (Array.isArray(content)) {
      return (content as Array<Record<string, unknown>>)
        .filter((b) => b.type === 'text')
        .map((b) => b.text as string)
        .join('\n')
    }
    return String(content)
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

  /** Return the effective backend tools whitelist. */
  getBackendTools(): ReadonlySet<string> {
    return this._backendTools
  }

  private _buildTransitionTool(): ToolLike {
    const taskNamesHint = this._ctx.taskOrder.map((n) => `'${n}'`).join(', ')

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

    const taskMw = getTaskMiddleware(this._ctx, activeName)
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
    const statuses = getStatuses(this._ctx, request.state as TaskSteeringState)
    const activeName = getActiveTask(this._ctx, statuses)

    if (request.toolCall.name === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as { task: string; status: string }
      const { task: taskName, status: target } = args

      const concurrentReject = this._rejectConcurrentStart(request, statuses, taskName, target)
      if (concurrentReject) return concurrentReject

      // Async completion validation
      if (target === TaskStatus.COMPLETE && this._ctx.taskMap.has(taskName)) {
        const taskMw = getTaskMiddleware(this._ctx, taskName)
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
      if (this._isCommand(result) && this._ctx.taskMap.has(taskName)) {
        // Aborted is user-driven: no lifecycle hooks, no summarization.
        if (target === TaskStatus.ABORTED) return result

        const taskMw = getTaskMiddleware(this._ctx, taskName)
        if (taskMw) {
          const updatedStatuses = { ...statuses, [taskName]: target }
          const postState = { ...request.state, taskStatuses: updatedStatuses }
          let updates: Record<string, unknown> | AbortAll | void = undefined
          if (target === TaskStatus.IN_PROGRESS) {
            updates = await taskMw.aOnStart(postState)
          } else if (target === TaskStatus.COMPLETE) {
            updates = await taskMw.aOnComplete(postState)
          }
          if (isAbortAll(updates)) {
            applyAbortAll(result, updates, this._ctx)
            return result
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

    const taskMw = getTaskMiddleware(this._ctx, activeName)
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

/**
 * Record `taskMessageStarts[taskName]` on the CommandResult.
 *
 * Always records (not just when summarization is configured) so the
 * abort-commitment check can detect tool calls made during the task.
 * Mutates `result.update` in place.
 *
 * `startIndex` points past every message already in `state` plus every
 * message the transition CommandResult will append (the transition tool
 * message and any messages returned by `onStart`), so the first "task
 * message" lands at this index.
 */
function recordTaskStart(
  result: CommandResult,
  state: Record<string, unknown>,
  taskName: string
): void {
  const messages = (state.messages as unknown[]) ?? []
  const update = result.update
  const pendingMsgs = (update.messages as unknown[] | undefined) ?? []
  const startIndex = messages.length + pendingMsgs.length
  const starts: Record<string, number> = {
    ...((state.taskMessageStarts as Record<string, number>) ?? {}),
  }
  starts[taskName] = startIndex
  update.taskMessageStarts = starts
}

/**
 * Apply an {@link AbortAll} signal from `onComplete`.
 *
 * The task that triggered the signal remains marked `complete` — `AbortAll`
 * is a downstream decision about *remaining* tasks, not a rollback. Any
 * still-`pending` or `in_progress` task is marked `aborted`, the transition
 * ToolMessage is extended with the abort reason, and in workflow mode the
 * active workflow is deactivated.
 *
 * Mutates `result.update` in place.
 */
function applyAbortAll(result: CommandResult, abort: AbortAll, ctx: PipelineContext): void {
  const update = result.update
  const statuses: Record<string, string> = {
    ...((update.taskStatuses as Record<string, string>) ?? {}),
  }

  const abortedNames: string[] = []
  for (const name of ctx.taskOrder) {
    if (statuses[name] === TaskStatus.PENDING || statuses[name] === TaskStatus.IN_PROGRESS) {
      statuses[name] = TaskStatus.ABORTED
      abortedNames.push(name)
    }
  }
  update.taskStatuses = statuses

  const workflowMode = ctx.label != null
  if (workflowMode) {
    update.activeWorkflow = null
    update.taskMessageStarts = {}
    update.nudgeCount = 0
  }

  const noteParts: string[] = [`All remaining tasks aborted: ${abort.reason}`]
  if (abortedNames.length > 0) {
    noteParts.push(`Aborted: ${abortedNames.join(', ')}`)
  }
  if (workflowMode) {
    noteParts.push(`Workflow '${ctx.label}' deactivated.`)
  }
  const note = '\n\n' + noteParts.join('\n')

  const existingMsgs = [...((update.messages as unknown[]) ?? [])]
  let rewrote = false
  for (let i = 0; i < existingMsgs.length; i++) {
    if (msgRole(existingMsgs[i]) === 'tool') {
      const msg = existingMsgs[i] as Record<string, unknown>
      existingMsgs[i] = { ...msg, content: `${msg.content}${note}` }
      rewrote = true
      break
    }
  }
  if (!rewrote) {
    // Invariant: the transition CommandResult (built by executeTaskTransition)
    // always carries a tool message. Reaching here means the upstream contract
    // changed — fail loudly rather than silently dropping the abort note.
    throw new Error('AbortAll: transition CommandResult had no tool message to extend.')
  }
  update.messages = existingMsgs
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

  onComplete(state: Record<string, unknown>): Record<string, unknown> | AbortAll | void {
    let merged: Record<string, unknown> | void = undefined
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'onComplete')) {
        const updates = mw.onComplete(state)
        if (isAbortAll(updates)) return updates
        merged = mergeHookUpdates(merged, updates as Record<string, unknown> | void)
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

  async aOnComplete(
    state: Record<string, unknown>
  ): Promise<Record<string, unknown> | AbortAll | void> {
    let merged: Record<string, unknown> | void = undefined
    for (const mw of this._middlewares) {
      if (overridesMethod(mw, 'aOnComplete') || overridesMethod(mw, 'onComplete')) {
        const updates = await mw.aOnComplete(state)
        if (isAbortAll(updates)) return updates
        merged = mergeHookUpdates(merged, updates as Record<string, unknown> | void)
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

// ── Shared helpers for deduplication ──────────────────────

function dedupTools(candidates: ToolLike[]): ToolLike[] {
  const seen = new Set<string>()
  const out: ToolLike[] = []
  for (const t of candidates) {
    if (!seen.has(t.name)) {
      seen.add(t.name)
      out.push(t)
    }
  }
  return out
}

function resolveRequiredTasks(
  requiredTasks: readonly string[] | null | undefined,
  allNames: string[],
  contextLabel?: string
): Set<string> {
  if (requiredTasks === undefined || (requiredTasks !== null && requiredTasks.includes('*'))) {
    return new Set(allNames)
  }
  if (requiredTasks !== null) {
    const unknown = requiredTasks.filter((n) => !allNames.includes(n))
    if (unknown.length > 0) {
      const prefix = contextLabel ? ` in ${contextLabel}` : ''
      throw new Error(`Unknown required tasks${prefix}: ${unknown.join(', ')}`)
    }
    return new Set(requiredTasks)
  }
  return new Set()
}

// ── Pipeline context (shared between task & workflow modes) ──

interface PipelineContext {
  tasks: Task[]
  taskOrder: string[]
  taskMap: Map<string, Task>
  globalTools: ToolLike[]
  enforceOrder: boolean
  requiredTasks: Set<string>
  globalSkills: string[]
  skillsActive: boolean
  skillRequiredTools: ReadonlySet<string>
  label: string | null
}

function buildPipelineContext(
  tasks: Task[],
  globalTools: ToolLike[],
  enforceOrder: boolean,
  requiredTasks: Set<string>,
  globalSkills: string[],
  label: string | null = null
): PipelineContext {
  const skillsActive = tasks.some((t) => t.skills && t.skills.length > 0) || globalSkills.length > 0
  return {
    tasks,
    taskOrder: tasks.map((t) => t.name),
    taskMap: new Map(tasks.map((t) => [t.name, t])),
    globalTools,
    enforceOrder,
    requiredTasks,
    globalSkills,
    skillsActive,
    skillRequiredTools: skillsActive ? new Set(['read_file', 'ls']) : new Set(),
    label,
  }
}

function validateAndNormalizeTasks(tasks: Task[]): Task[] {
  const names = tasks.map((t) => t.name)
  const dupes = names.filter((n, i) => names.indexOf(n) !== i)
  if (dupes.length > 0) {
    throw new Error(`Duplicate task names: ${[...new Set(dupes)].join(', ')}`)
  }
  const copies = tasks.map((t) => ({ ...t }))
  for (const task of copies) {
    task.middleware = normalizeMiddleware(task.middleware)
    task.tools = [...task.tools]
    if (task.summarize) validateTaskSummarization(task.summarize)
  }
  return copies
}

// ── Shared rendering helpers ──────────────────────────────

function renderStatusBlock(
  ctx: PipelineContext,
  statuses: Record<string, string>,
  active: string | null,
  state?: Record<string, unknown>,
  backendToolsPassthrough?: boolean,
  warnedMissingSkills?: Set<string>
): string {
  const lines: string[] =
    ctx.label != null ? [`\n<task_pipeline workflow="${ctx.label}">`] : ['\n<task_pipeline>']

  let hasOptional = false
  for (const t of ctx.tasks) {
    const s = statuses[t.name] ?? TaskStatus.PENDING
    let optionalTag = ''
    if (!ctx.requiredTasks.has(t.name)) {
      hasOptional = true
      optionalTag = ' [optional]'
    }
    lines.push(`  ${STATUS_ICONS[s] ?? '[?]'} ${t.name} (${s})${optionalTag}`)
  }

  if (active) {
    const task = ctx.taskMap.get(active)!
    lines.push(`\n  <current_task name="${active}">`)
    lines.push(`    ${task.instruction}`)
    if (!ctx.requiredTasks.has(active)) {
      lines.push('')
      lines.push("    This task is optional. You may set it to 'in_progress' to review,")
      lines.push("    then abort it (update_task_status with status='aborted') if it's not")
      lines.push('    needed. Once you call any tool for this task, you are committed to')
      lines.push('    completing it.')
    }
    lines.push('  </current_task>')
  }

  // ── Skill rendering ──────────────────────────────────────
  let hasVisibleSkills = false
  if (ctx.skillsActive && state != null) {
    const allSkills = (state.skillsMetadata as SkillMetadata[] | undefined) ?? []
    const allowedNames = getAllowedSkillNames(ctx, active)
    const visibleSkills = allSkills.filter((s) => allowedNames.has(s.name))

    // Only task mode (no label) produces missing skill warnings
    if (ctx.label == null) {
      const availableNames = new Set(allSkills.map((s) => s.name))
      const missing = [...allowedNames].filter((n) => !availableNames.has(n))
      const newMissing = warnedMissingSkills
        ? missing.filter((n) => !warnedMissingSkills.has(n))
        : missing
      if (newMissing.length > 0) {
        console.warn(
          `[langchain-task-steering] Skill(s) ${newMissing.sort().join(', ')} referenced by ` +
            `task/global config but not found in skillsMetadata state. Check skill names ` +
            `and ensure skills are loaded (e.g. via SkillsMiddleware).`
        )
        if (warnedMissingSkills) {
          for (const n of newMissing) warnedMissingSkills.add(n)
        }
      }
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

  if (ctx.enforceOrder || hasVisibleSkills || hasOptional) {
    lines.push('\n  <rules>')
    if (ctx.enforceOrder) {
      const orderStr = ctx.taskOrder.join(' -> ')
      lines.push(`    Required order: ${orderStr}`)
    }
    lines.push('    Use update_task_status to advance. Do not skip tasks.')
    if (hasOptional) {
      lines.push('    Tasks marked [optional] can be aborted before their first tool')
      lines.push("    call (update_task_status with status='aborted'). After the first")
      lines.push('    tool call you are committed to completing the task.')
    }
    if (hasVisibleSkills) {
      lines.push('    To use a skill, read its SKILL.md file for full instructions.')
    }
    lines.push('  </rules>')
  }

  // Only task mode (no label) produces the verbose skill_usage block
  if (hasVisibleSkills && ctx.label == null) {
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

function getStatuses(ctx: PipelineContext, state: Record<string, unknown>): Record<string, string> {
  const raw = (state.taskStatuses as Record<string, string>) ?? {}
  const result: Record<string, string> = {}
  for (const t of ctx.tasks) {
    result[t.name] = raw[t.name] ?? TaskStatus.PENDING
  }
  return result
}

function getActiveTask(ctx: PipelineContext, statuses: Record<string, string>): string | null {
  for (const name of ctx.taskOrder) {
    if (statuses[name] === TaskStatus.IN_PROGRESS) return name
  }
  return null
}

function getTaskMiddleware(
  ctx: PipelineContext,
  taskName: string | null
): TaskMiddleware | undefined {
  if (!taskName) return undefined
  return ctx.taskMap.get(taskName)?.middleware as TaskMiddleware | undefined
}

function getAllowedSkillNames(ctx: PipelineContext, activeName: string | null): Set<string> {
  const names = new Set(ctx.globalSkills)
  if (activeName) {
    const task = ctx.taskMap.get(activeName)
    if (task?.skills) {
      for (const s of task.skills) names.add(s)
    }
  }
  return names
}

function getAllowedToolNames(
  ctx: PipelineContext,
  activeName: string | null,
  extraToolNames: Set<string>,
  backendToolsPassthrough: boolean,
  backendTools: ReadonlySet<string>,
  state?: Record<string, unknown>
): Set<string> {
  const names = new Set<string>([TRANSITION_TOOL_NAME])
  for (const n of extraToolNames) names.add(n)
  for (const t of ctx.globalTools) names.add(t.name)
  if (activeName) {
    const task = ctx.taskMap.get(activeName)
    if (task) {
      for (const t of task.tools) names.add(t.name)
      const mwTools = (task.middleware as { tools?: ToolLike[] })?.tools
      if (mwTools) {
        for (const t of mwTools) names.add(t.name)
      }
    }
  }
  if (backendToolsPassthrough) {
    for (const t of backendTools) names.add(t)
  }
  if (ctx.skillsActive) {
    const allowedSkills = getAllowedSkillNames(ctx, activeName)
    if (allowedSkills.size > 0) {
      for (const t of ctx.skillRequiredTools) names.add(t)
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

/**
 * Validate and execute a task status transition.
 * Shared by task-mode and workflow-mode transition executors.
 *
 * Supports three target statuses:
 * - `in_progress` — start a task. Blocked by required preceding tasks
 *   that are not `complete`/`aborted`. Optional preceding tasks that
 *   are still `pending` are skipped (so a never-started optional task
 *   never blocks a required task behind it).
 * - `complete` — complete an `in_progress` task.
 * - `aborted` — abort an `in_progress` *optional* task, provided no
 *   tool calls have been made for that task yet. Required tasks cannot
 *   be aborted by the agent (use `AbortAll` from `onComplete` for
 *   programmatic abort).
 */
function executeTaskTransition(
  task: string,
  status: string,
  taskOrder: string[],
  enforceOrder: boolean,
  requiredTasks: Set<string>,
  state: Record<string, unknown>,
  toolCallId: string,
  contextLabel?: string
): ToolMessageResult | CommandResult {
  if (!taskOrder.includes(task)) {
    const suffix = contextLabel ? ` for ${contextLabel}` : ''
    return {
      content: `Invalid task '${task}'${suffix}. Must be one of: ${taskOrder.join(', ')}`,
      toolCallId,
    }
  }

  if (
    status !== TaskStatus.IN_PROGRESS &&
    status !== TaskStatus.COMPLETE &&
    status !== TaskStatus.ABORTED
  ) {
    return {
      content: `Invalid status '${status}'. Must be 'in_progress', 'complete', or 'aborted'.`,
      toolCallId,
    }
  }

  const statuses: Record<string, string> = {
    ...((state.taskStatuses as Record<string, string>) ?? {}),
  }
  for (const t of taskOrder) {
    if (!(t in statuses)) statuses[t] = TaskStatus.PENDING
  }

  const current = statuses[task]

  // ── Abort transition ─────────────────────────────────
  if (status === TaskStatus.ABORTED) {
    if (requiredTasks.has(task)) {
      return {
        content: `Cannot abort '${task}': task is required. Complete it instead.`,
        toolCallId,
      }
    }
    if (current === TaskStatus.PENDING) {
      return {
        content: `Cannot abort '${task}': task hasn't started. Set it to 'in_progress' first to review, or leave it pending.`,
        toolCallId,
      }
    }
    if (current !== TaskStatus.IN_PROGRESS) {
      return {
        content: `Task '${task}' is already ${current}.`,
        toolCallId,
      }
    }

    // Commitment check: any tool message since the task went in_progress?
    const starts = (state.taskMessageStarts as Record<string, number>) ?? {}
    const startIndex = starts[task]
    if (startIndex != null) {
      const messages = (state.messages as unknown[]) ?? []
      for (let i = startIndex; i < messages.length; i++) {
        if (msgRole(messages[i]) === 'tool') {
          return {
            content: `Cannot abort '${task}': tools already executed — task has made state changes. Complete it instead.`,
            toolCallId,
          }
        }
      }
    }

    statuses[task] = TaskStatus.ABORTED

    const newStarts: Record<string, number> = {
      ...((state.taskMessageStarts as Record<string, number>) ?? {}),
    }
    delete newStarts[task]

    const display = Object.entries(statuses)
      .map(([k, v]) => `  ${k}: ${v}`)
      .join('\n')

    return {
      update: {
        taskStatuses: statuses,
        taskMessageStarts: newStarts,
        nudgeCount: 0,
        messages: [
          {
            role: 'tool',
            content: `Task '${task}' -> aborted.\n\n${display}`,
            toolCallId,
          },
        ],
      },
    }
  }

  // ── Pending → in_progress → complete ─────────────────
  if (isTerminal(current)) {
    return {
      content: `Task '${task}' is already ${current}.`,
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

  if (enforceOrder && status === TaskStatus.IN_PROGRESS) {
    const idx = taskOrder.indexOf(task)
    for (let i = 0; i < idx; i++) {
      const prev = taskOrder[i]
      // Optional, never-started tasks don't block (dead-end fix).
      if (!requiredTasks.has(prev) && statuses[prev] === TaskStatus.PENDING) {
        continue
      }
      if (!isTerminal(statuses[prev])) {
        return {
          content: `Cannot start '${task}': '${prev}' is not complete yet. Order: ${taskOrder.join(' -> ')}.`,
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

// ════════════════════════════════════════════════════════════════
// WorkflowSteeringMiddleware — workflow mode (dynamic workflows)
// ════════════════════════════════════════════════════════════════

export interface WorkflowSteeringMiddlewareConfig {
  workflows: Workflow[]
  maxNudges?: number
  /** When true, known backend tools pass through the tool filter on all tasks. */
  backendToolsPassthrough?: boolean
  /** Override the default backend tools whitelist. */
  backendTools?: ReadonlySet<string> | null
  /**
   * Default chat model for `TaskSummarization(mode="summarize")`.
   */
  model?: unknown
}

/**
 * Workflow-mode middleware for dynamic pipeline activation/deactivation.
 *
 * The agent sees a catalog of available workflows and activates one on
 * demand via the `activate_workflow` tool. When no workflow is active
 * the middleware is transparent (no tool filtering or prompt injection).
 */
export class WorkflowSteeringMiddleware {
  static readonly DEFAULT_BACKEND_TOOLS: ReadonlySet<string> =
    TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS

  private readonly _workflows: Workflow[]
  private readonly _workflowMap: Map<string, Workflow>
  /** @internal */ readonly _workflowCtxs: Map<string, PipelineContext>
  private readonly _maxNudges: number
  private readonly _model: unknown
  private readonly _backendToolsPassthrough: boolean
  private readonly _backendTools: ReadonlySet<string>

  /** Cached catalog text (invariant). */
  private readonly _catalogText: string
  /** All workflow tool names (for filtering in catalog mode). */
  private readonly _workflowToolNames: Set<string>

  /** All tools registered by this middleware. */
  readonly tools: ToolLike[]

  constructor(config: WorkflowSteeringMiddlewareConfig) {
    const {
      workflows,
      maxNudges = 3,
      backendToolsPassthrough = false,
      backendTools = null,
      model = null,
    } = config

    if (workflows.length === 0) {
      throw new Error('At least one Workflow is required.')
    }

    // Check duplicate workflow names
    const wfNames = workflows.map((w) => w.name)
    const wfDupes = wfNames.filter((n, i) => wfNames.indexOf(n) !== i)
    if (wfDupes.length > 0) {
      throw new Error(`Duplicate workflow names: ${[...new Set(wfDupes)].join(', ')}`)
    }

    // Validate and normalize each workflow
    this._workflows = []
    for (const wf of workflows) {
      if (!wf.tasks || wf.tasks.length === 0) {
        throw new Error(
          `Workflow '${wf.name}' has no tasks. Each workflow must have at least one Task.`
        )
      }
      const normalized: Workflow = {
        ...wf,
        tasks: validateAndNormalizeTasks(wf.tasks),
        globalTools: wf.globalTools ? [...wf.globalTools] : [],
        globalSkills: wf.globalSkills ? [...wf.globalSkills] : undefined,
        enforceOrder: wf.enforceOrder !== false,
        allowDeactivateInProgress: wf.allowDeactivateInProgress === true,
      }
      this._workflows.push(normalized)
    }

    this._workflowMap = new Map(this._workflows.map((w) => [w.name, w]))
    this._maxNudges = maxNudges
    this._model = model
    this._backendTools =
      backendTools != null
        ? new Set(backendTools)
        : WorkflowSteeringMiddleware.DEFAULT_BACKEND_TOOLS
    this._backendToolsPassthrough = backendToolsPassthrough

    // Build pipeline context per workflow
    this._workflowCtxs = new Map()
    for (const wf of this._workflows) {
      const taskNames = wf.tasks.map((t) => t.name)
      this._workflowCtxs.set(
        wf.name,
        buildPipelineContext(
          wf.tasks,
          wf.globalTools ?? [],
          wf.enforceOrder !== false,
          resolveRequiredTasks(wf.requiredTasks, taskNames, `workflow '${wf.name}'`),
          [...(wf.globalSkills ?? [])],
          wf.name
        )
      )
    }

    // Build management tools
    const activateTool = this._buildActivateTool()
    const deactivateTool = this._buildDeactivateTool()
    const transitionTool = this._buildTransitionTool()

    // Register all tools (deduplicated)
    this.tools = dedupTools([
      activateTool,
      deactivateTool,
      transitionTool,
      ...this._workflows.flatMap((wf) => wf.globalTools ?? []),
      ...this._workflows.flatMap((wf) => wf.tasks.flatMap((t) => t.tools)),
      ...this._workflows.flatMap((wf) =>
        wf.tasks.flatMap((t) => {
          const mwTools = (t.middleware as { tools?: ToolLike[] })?.tools
          return mwTools ?? []
        })
      ),
    ])

    this._workflowToolNames = new Set(this.tools.map((t) => t.name))
    this._catalogText = this._renderCatalog()
  }

  // ── Node-style hooks ──────────────────────────────────

  /** Workflow mode: nothing to do here (activate tool inits state). */
  beforeAgent(_state: TaskSteeringState): Record<string, unknown> | null {
    return null
  }

  /**
   * Nudge the agent back if required tasks are incomplete within the active workflow.
   */
  afterAgent(
    state: TaskSteeringState
  ): { jumpTo: string; nudgeCount: number; messages: unknown[] } | null {
    const ctx = this._getPipelineCtx(state)
    if (ctx == null) return null

    if (ctx.requiredTasks.size === 0) return null

    const statuses = getStatuses(ctx, state)
    const incomplete = ctx.taskOrder.filter(
      (name) => ctx.requiredTasks.has(name) && !isTerminal(statuses[name])
    )

    if (incomplete.length === 0) return null

    const nudgeCount = (state.nudgeCount as number) ?? 0
    if (nudgeCount >= this._maxNudges) return null

    const wfName = state.activeWorkflow ?? 'unknown'
    const taskList = incomplete.join(', ')
    return {
      jumpTo: 'model',
      nudgeCount: nudgeCount + 1,
      messages: [
        {
          role: 'human',
          content: `You have not completed the following required tasks in workflow '${wfName}': ${taskList}. Please continue.`,
          additional_kwargs: {
            task_steering: { kind: 'nudge', incomplete_tasks: incomplete },
          },
        },
      ],
    }
  }

  // ── Wrap-style hooks ──────────────────────────────────

  wrapModelCall(request: ModelRequest, handler: ModelCallHandler): ModelResponse {
    const ctx = this._getPipelineCtx(request.state)
    if (ctx == null) {
      // No workflow active — inject catalog, pass external tools through
      return handler(this._buildCatalogRequest(request))
    }

    const { modified, activeName } = this._prepareModelRequest(request, ctx)

    const taskMw = getTaskMiddleware(ctx, activeName)
    if (taskMw?.wrapModelCall) {
      return taskMw.wrapModelCall(modified, handler)
    }

    return handler(modified)
  }

  wrapToolCall(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult {
    const toolName = request.toolCall.name

    // Management tools always pass straight through to handler
    if (toolName === ACTIVATE_TOOL_NAME || toolName === DEACTIVATE_TOOL_NAME) {
      return handler(request)
    }

    const ctx = this._getPipelineCtx(request.state)
    if (ctx == null) {
      // No workflow active — transparent
      return handler(request)
    }

    const statuses = getStatuses(ctx, request.state)
    const activeName = getActiveTask(ctx, statuses)

    if (toolName === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as { task: string; status: string }
      const { task: taskName, status: target } = args

      // Reject concurrent start
      if (target === TaskStatus.IN_PROGRESS) {
        const active = ctx.taskOrder.filter((n) => statuses[n] === TaskStatus.IN_PROGRESS)
        if (active.length > 0) {
          return {
            content: `Cannot start '${taskName}': '${active[0]}' is already in progress. Complete it first.`,
            toolCallId: request.toolCall.id,
          }
        }
      }

      // Validate completion
      if (target === TaskStatus.COMPLETE && ctx.taskMap.has(taskName)) {
        const taskMw = getTaskMiddleware(ctx, taskName)
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
      this._fireLifecycleHooks(request, result, statuses, ctx, taskName, target)
      return result
    }

    // Gate tool
    const allowed = getAllowedToolNames(
      ctx,
      activeName,
      new Set([DEACTIVATE_TOOL_NAME]),
      this._backendToolsPassthrough,
      this._backendTools,
      request.state
    )
    if (!allowed.has(request.toolCall.name)) {
      return {
        content: `Tool '${request.toolCall.name}' is not available for the current task.`,
        toolCallId: request.toolCall.id,
      }
    }

    const taskMw = getTaskMiddleware(ctx, activeName)
    if (taskMw?.wrapToolCall) {
      return taskMw.wrapToolCall(request, handler)
    }

    return handler(request)
  }

  // ── Async wrap-style hooks ─────────────────────────────

  async awrapModelCall(
    request: ModelRequest,
    handler: AsyncModelCallHandler
  ): Promise<ModelResponse> {
    const ctx = this._getPipelineCtx(request.state)
    if (ctx == null) {
      return handler(this._buildCatalogRequest(request))
    }

    const { modified, activeName } = this._prepareModelRequest(request, ctx)

    const taskMw = getTaskMiddleware(ctx, activeName)
    if (taskMw?.wrapModelCall) {
      return taskMw.wrapModelCall(modified, handler as unknown as ModelCallHandler)
    }

    return handler(modified)
  }

  async awrapToolCall(
    request: ToolCallRequest,
    handler: AsyncToolCallHandler
  ): Promise<ToolMessageResult | CommandResult> {
    const toolName = request.toolCall.name

    if (toolName === ACTIVATE_TOOL_NAME || toolName === DEACTIVATE_TOOL_NAME) {
      return handler(request)
    }

    const ctx = this._getPipelineCtx(request.state)
    if (ctx == null) {
      return handler(request)
    }

    const statuses = getStatuses(ctx, request.state)
    const activeName = getActiveTask(ctx, statuses)

    if (toolName === TRANSITION_TOOL_NAME) {
      const args = request.toolCall.args as { task: string; status: string }
      const { task: taskName, status: target } = args

      if (target === TaskStatus.IN_PROGRESS) {
        const active = ctx.taskOrder.filter((n) => statuses[n] === TaskStatus.IN_PROGRESS)
        if (active.length > 0) {
          return {
            content: `Cannot start '${taskName}': '${active[0]}' is already in progress. Complete it first.`,
            toolCallId: request.toolCall.id,
          }
        }
      }

      if (target === TaskStatus.COMPLETE && ctx.taskMap.has(taskName)) {
        const taskMw = getTaskMiddleware(ctx, taskName)
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
      if (this._isCommand(result) && ctx.taskMap.has(taskName)) {
        // Aborted is user-driven: no lifecycle hooks, no summarization.
        if (target === TaskStatus.ABORTED) return result

        const taskMw = getTaskMiddleware(ctx, taskName)
        if (taskMw) {
          const updatedStatuses = { ...statuses, [taskName]: target }
          const postState = { ...request.state, taskStatuses: updatedStatuses }
          let updates: Record<string, unknown> | AbortAll | void = undefined
          if (target === TaskStatus.IN_PROGRESS) {
            updates = await taskMw.aOnStart(postState)
          } else if (target === TaskStatus.COMPLETE) {
            updates = await taskMw.aOnComplete(postState)
          }
          if (isAbortAll(updates)) {
            applyAbortAll(result, updates, ctx)
            return result
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
          recordTaskStart(result, request.state, taskName)
        }
      }

      return result
    }

    const allowed = getAllowedToolNames(
      ctx,
      activeName,
      new Set([DEACTIVATE_TOOL_NAME]),
      this._backendToolsPassthrough,
      this._backendTools,
      request.state
    )
    if (!allowed.has(request.toolCall.name)) {
      return {
        content: `Tool '${request.toolCall.name}' is not available for the current task.`,
        toolCallId: request.toolCall.id,
      }
    }

    const taskMw = getTaskMiddleware(ctx, activeName)
    if (taskMw?.wrapToolCall) {
      return taskMw.wrapToolCall(request, handler as unknown as ToolCallHandler)
    }

    return handler(request)
  }

  // ── Public transition executors ────────────────────────

  /**
   * Execute the activate_workflow tool logic.
   */
  executeActivate(
    args: { workflow: string },
    state: Record<string, unknown>,
    toolCallId: string
  ): ToolMessageResult | CommandResult {
    const { workflow } = args

    if (!this._workflowMap.has(workflow)) {
      return {
        content: `Unknown workflow '${workflow}'. Available: ${[...this._workflowMap.keys()].join(', ')}`,
        toolCallId,
      }
    }

    const current = state.activeWorkflow as string | null | undefined
    if (current != null) {
      return {
        content: `Workflow '${current}' is already active. Deactivate it first with deactivate_workflow.`,
        toolCallId,
      }
    }

    const wf = this._workflowMap.get(workflow)!
    const statuses: Record<string, string> = {}
    for (const t of wf.tasks) {
      statuses[t.name] = TaskStatus.PENDING
    }

    const display = Object.entries(statuses)
      .map(([k, v]) => `  ${k}: ${v}`)
      .join('\n')

    return {
      update: {
        activeWorkflow: workflow,
        taskStatuses: statuses,
        nudgeCount: 0,
        messages: [
          {
            role: 'tool',
            content: `Workflow '${workflow}' activated.\n\n${display}`,
            toolCallId,
          },
        ],
      },
    }
  }

  /**
   * Execute the deactivate_workflow tool logic.
   */
  executeDeactivate(
    state: Record<string, unknown>,
    toolCallId: string
  ): ToolMessageResult | CommandResult {
    const current = state.activeWorkflow as string | null | undefined
    if (current == null) {
      return {
        content: 'No workflow is currently active.',
        toolCallId,
      }
    }

    const wf = this._workflowMap.get(current)

    // Check deactivation policy
    if (wf && !wf.allowDeactivateInProgress) {
      const statuses = (state.taskStatuses as Record<string, string>) ?? {}
      const active = Object.entries(statuses)
        .filter(([, s]) => s === TaskStatus.IN_PROGRESS)
        .map(([name]) => name)
      if (active.length > 0) {
        return {
          content: `Cannot deactivate: task '${active[0]}' is in progress. Complete or skip it first.`,
          toolCallId,
        }
      }
    }

    return {
      update: {
        activeWorkflow: null,
        taskStatuses: {},
        taskMessageStarts: {},
        nudgeCount: 0,
        messages: [
          {
            role: 'tool',
            content: `Workflow '${current}' deactivated.`,
            toolCallId,
          },
        ],
      },
    }
  }

  /**
   * Execute the update_task_status tool logic for workflow mode.
   */
  executeTransition(
    args: { task: string; status: string },
    state: Record<string, unknown>,
    toolCallId: string
  ): ToolMessageResult | CommandResult {
    const wfName = state.activeWorkflow as string | null | undefined
    if (wfName == null) {
      return {
        content: 'No workflow is active. Activate a workflow first.',
        toolCallId,
      }
    }

    const wf = this._workflowMap.get(wfName)
    const ctx = this._workflowCtxs.get(wfName)
    if (!wf || !ctx) {
      return {
        content: `Active workflow '${wfName}' not found.`,
        toolCallId,
      }
    }

    return executeTaskTransition(
      args.task,
      args.status,
      ctx.taskOrder,
      wf.enforceOrder !== false,
      ctx.requiredTasks,
      state,
      toolCallId,
      `workflow '${wfName}'`
    )
  }

  // ── Internal helpers ─────────────────────────��────────

  private _getPipelineCtx(state: Record<string, unknown>): PipelineContext | null {
    const wfName = state.activeWorkflow as string | null | undefined
    if (wfName == null) return null
    return this._workflowCtxs.get(wfName) ?? null
  }

  private _renderCatalog(): string {
    const lines = ['\n<available_workflows>']
    for (const wf of this._workflows) {
      const taskNames = wf.tasks.map((t) => t.name).join(', ')
      lines.push(`  <workflow name="${wf.name}">`)
      lines.push(`    ${wf.description}`)
      lines.push(`    Tasks: ${taskNames}`)
      lines.push('  </workflow>')
    }
    lines.push('')
    lines.push('  Use activate_workflow to start a workflow when needed.')
    lines.push('</available_workflows>')
    return lines.join('\n')
  }

  private _buildCatalogRequest(request: ModelRequest): ModelRequest {
    const existingBlocks = request.systemMessage ? getContentBlocks(request.systemMessage) : []
    const newContent: ContentBlock[] = [
      ...existingBlocks,
      { type: 'text', text: this._catalogText },
    ]

    // In catalog mode: show activate_workflow + all external (non-workflow) tools
    const allowed = new Set<string>([ACTIVATE_TOOL_NAME])
    const scoped: ToolLike[] = []
    for (const t of request.tools) {
      if (t.name === ACTIVATE_TOOL_NAME || !this._workflowToolNames.has(t.name)) {
        if (!allowed.has(t.name)) {
          allowed.add(t.name)
        }
        scoped.push(t)
      }
    }
    // Ensure activate_workflow is always present
    if (!scoped.some((t) => t.name === ACTIVATE_TOOL_NAME)) {
      const activateTool = this.tools.find((t) => t.name === ACTIVATE_TOOL_NAME)
      if (activateTool) scoped.unshift(activateTool)
    }

    return request.override({
      systemMessage: { content: newContent },
      tools: scoped,
    })
  }

  /** @internal */
  _prepareModelRequest(
    request: ModelRequest,
    ctx: PipelineContext
  ): { modified: ModelRequest; activeName: string | null } {
    const statuses = getStatuses(ctx, request.state)
    const activeName = getActiveTask(ctx, statuses)

    const block = renderStatusBlock(ctx, statuses, activeName, request.state)
    let existingBlocks = request.systemMessage ? getContentBlocks(request.systemMessage) : []

    if (ctx.skillsActive) {
      existingBlocks = existingBlocks.filter(
        (b) => !(b.type === 'text' && b.text?.includes('## Skills System'))
      )
    }

    const newContent: ContentBlock[] = [...existingBlocks, { type: 'text', text: block }]

    const allowedNames = getAllowedToolNames(
      ctx,
      activeName,
      new Set([DEACTIVATE_TOOL_NAME]),
      this._backendToolsPassthrough,
      this._backendTools,
      request.state
    )
    const scoped = request.tools.filter((t) => allowedNames.has(t.name))

    const modified = request.override({
      systemMessage: { content: newContent },
      tools: scoped,
    })

    return { modified, activeName }
  }

  private _fireLifecycleHooks(
    request: ToolCallRequest,
    result: ToolMessageResult | CommandResult,
    statuses: Record<string, string>,
    ctx: PipelineContext,
    taskName: string,
    target: string
  ): void {
    if (!this._isCommand(result) || !ctx.taskMap.has(taskName)) return

    // Aborted is user-driven: no lifecycle hooks, no summarization.
    if (target === TaskStatus.ABORTED) return

    const taskMw = getTaskMiddleware(ctx, taskName)
    if (taskMw) {
      const updatedStatuses = { ...statuses, [taskName]: target }
      const postState = { ...request.state, taskStatuses: updatedStatuses }

      let updates: Record<string, unknown> | AbortAll | void = undefined
      if (target === TaskStatus.IN_PROGRESS) {
        updates = taskMw.onStart(postState)
      } else if (target === TaskStatus.COMPLETE) {
        updates = taskMw.onComplete(postState)
      }

      if (isAbortAll(updates)) {
        applyAbortAll(result, updates, ctx)
        return
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
      recordTaskStart(result, request.state, taskName)
    }
  }

  private _isCommand(result: ToolMessageResult | CommandResult): result is CommandResult {
    return 'update' in result
  }

  private _buildActivateTool(): ToolLike {
    const wfNamesHint = [...this._workflowMap.keys()].map((n) => `'${n}'`).join(', ')
    return {
      name: ACTIVATE_TOOL_NAME,
      description:
        'Activate a workflow to start working on a structured task pipeline. ' +
        `Only one workflow can be active at a time. Workflow names: ${wfNamesHint}`,
    }
  }

  private _buildDeactivateTool(): ToolLike {
    return {
      name: DEACTIVATE_TOOL_NAME,
      description:
        'Deactivate the current workflow, clearing all task state. ' +
        'May be blocked if a task is in progress.',
    }
  }

  private _buildTransitionTool(): ToolLike {
    return {
      name: TRANSITION_TOOL_NAME,
      description:
        "Transition a task to 'in_progress' or 'complete'. " +
        'Must be called ALONE — never in parallel with other tools. ' +
        'Tasks must follow the defined order within the active workflow.',
    }
  }

  /** Return the effective backend tools whitelist. */
  getBackendTools(): ReadonlySet<string> {
    return this._backendTools
  }
}

export {
  ACTIVATE_TOOL_NAME as _ACTIVATE_TOOL_NAME,
  DEACTIVATE_TOOL_NAME as _DEACTIVATE_TOOL_NAME,
  TRANSITION_TOOL_NAME as _TRANSITION_TOOL_NAME,
  // Shared helpers (for tests)
  getStatuses as _getStatuses,
  getActiveTask as _getActiveTask,
  getAllowedToolNames as _getAllowedToolNames,
  getAllowedSkillNames as _getAllowedSkillNames,
  renderStatusBlock as _renderStatusBlock,
  executeTaskTransition as _executeTaskTransition,
}

export type { PipelineContext as _PipelineContext }
