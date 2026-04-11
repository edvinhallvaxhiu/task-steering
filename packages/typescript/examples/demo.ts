/**
 * Runnable demo: simulates a full agent loop through the middleware.
 *
 * Run with:
 *   npx tsx examples/demo.ts
 */

import {
  TaskSteeringMiddleware,
  TaskMiddleware,
  AgentMiddlewareAdapter,
  TaskStatus,
  getContentBlocks,
  type AgentMiddlewareLike,
  type Task,
  type ToolLike,
  type ModelRequest,
  type ToolCallRequest,
  type ToolMessageResult,
  type CommandResult,
  type ContentBlock,
  type ModelCallHandler,
  type ToolCallHandler,
} from '../src/index.js'

// ── Helpers ─────────────────────────────────────────────────

const BLUE = '\x1b[34m'
const GREEN = '\x1b[32m'
const RED = '\x1b[31m'
const YELLOW = '\x1b[33m'
const DIM = '\x1b[2m'
const RESET = '\x1b[0m'

function log(prefix: string, color: string, msg: string) {
  console.log(`  ${color}${prefix}${RESET} ${msg}`)
}

function header(title: string) {
  console.log(`\n${'='.repeat(60)}`)
  console.log(`  ${title}`)
  console.log('='.repeat(60))
}

function isCommand(r: ToolMessageResult | CommandResult): r is CommandResult {
  return 'update' in r
}

/** Simulate a model request to see what prompt/tools the model would receive. */
function mockModelRequest(state: Record<string, unknown>, tools: ToolLike[]): ModelRequest {
  return {
    state,
    systemMessage: { content: 'You are a helpful project manager.' },
    tools,
    override(overrides) {
      return {
        state,
        systemMessage: overrides.systemMessage ?? this.systemMessage,
        tools: overrides.tools ?? this.tools,
        override: this.override,
      }
    },
  }
}

// ── Tools ───────────────────────────────────────────────────

const gatherRequirements: ToolLike = {
  name: 'gather_requirements',
  description: 'Gather requirements for a topic.',
}
const writeDesign: ToolLike = {
  name: 'write_design',
  description: 'Write a design document.',
}
const reviewDesign: ToolLike = {
  name: 'review_design',
  description: 'Review a design document.',
}
const globalSearch: ToolLike = {
  name: 'search_docs',
  description: 'Search documentation (available in all tasks).',
}

// ── Adapter-contributed tools ───────────────────────────────

const auditLog: ToolLike = {
  name: 'audit_log',
  description: 'Log an audit entry (contributed by adapter).',
}

// ── Agent-level middleware to wrap via adapter ──────────────

/**
 * Simulates an existing agent middleware that:
 * - Contributes a tool (audit_log)
 * - Intercepts model calls (e.g. to inject extra context)
 * - Intercepts tool calls (e.g. to log usage)
 */
class AuditMiddleware implements AgentMiddlewareLike {
  tools = [auditLog]
  modelCallCount = 0
  toolCallCount = 0

  wrapModelCall(request: ModelRequest, handler: ModelCallHandler) {
    this.modelCallCount++
    log('ADAPTER', YELLOW, `AuditMiddleware.wrapModelCall() — call #${this.modelCallCount}`)
    return handler(request)
  }

  wrapToolCall(request: ToolCallRequest, handler: ToolCallHandler) {
    this.toolCallCount++
    log(
      'ADAPTER',
      YELLOW,
      `AuditMiddleware.wrapToolCall(${request.toolCall.name}) — call #${this.toolCallCount}`
    )
    return handler(request)
  }
}

/**
 * A no-op middleware — verifies that the adapter gracefully
 * handles an inner middleware that implements no hooks.
 */
class NoOpMiddleware implements AgentMiddlewareLike {}

// ── Task middleware with validation ─────────────────────────

class DesignMiddleware extends TaskMiddleware {
  validateCompletion(state: Record<string, unknown>): string | null {
    const designWritten = state.designWritten as boolean | undefined
    if (!designWritten) {
      return 'You must call write_design before completing this task.'
    }
    return null
  }

  onStart(state: Record<string, unknown>): void {
    const statuses = state.taskStatuses as Record<string, string> | undefined
    log('LIFECYCLE', YELLOW, `DesignMiddleware.onStart() — statuses: ${JSON.stringify(statuses)}`)
  }

  onComplete(state: Record<string, unknown>): void {
    const statuses = state.taskStatuses as Record<string, string> | undefined
    log(
      'LIFECYCLE',
      YELLOW,
      `DesignMiddleware.onComplete() — statuses: ${JSON.stringify(statuses)}`
    )
  }
}

// ── Build the middleware ────────────────────────────────────

const auditMw = new AuditMiddleware()

const tasks: Task[] = [
  {
    name: 'requirements',
    instruction: 'Gather the requirements for a login page.',
    tools: [gatherRequirements],
    middleware: new AgentMiddlewareAdapter(new NoOpMiddleware()),
  },
  {
    name: 'design',
    instruction: 'Write a design document based on the gathered requirements.',
    tools: [writeDesign],
    middleware: new DesignMiddleware(),
  },
  {
    name: 'review',
    instruction: 'Review the design document and provide final feedback.',
    tools: [reviewDesign],
    middleware: new AgentMiddlewareAdapter(auditMw),
  },
]

const mw = new TaskSteeringMiddleware({
  tasks,
  globalTools: [globalSearch],
})

console.log(`\n${GREEN}langchain-task-steering TypeScript Demo${RESET}`)
console.log(`${DIM}Simulating an agent loop through all middleware hooks${RESET}`)
console.log(`${DIM}Including AgentMiddlewareAdapter forwarding${RESET}\n`)

console.log('Registered tools:', mw.tools.map((t) => t.name).join(', '))

// ── State ───────────────────────────────────────────────────

const state: Record<string, unknown> = { messages: [] }

// ════════════════════════════════════════════════════════════
// Step 1: beforeAgent — initialize state
// ════════════════════════════════════════════════════════════

header('Step 1: beforeAgent — initialize state')

const init = mw.beforeAgent(state as any)
if (init) {
  Object.assign(state, init)
  log('STATE', BLUE, `taskStatuses = ${JSON.stringify(state.taskStatuses)}`)
}

// ════════════════════════════════════════════════════════════
// Step 2: wrapModelCall — see what model receives (no active task)
// ════════════════════════════════════════════════════════════

header('Step 2: wrapModelCall — no active task yet')

let captured: ModelRequest | null = null
mw.wrapModelCall(mockModelRequest(state, mw.tools), (r) => {
  captured = r
  return {}
})

const visibleTools = captured!.tools.map((t) => t.name)
log('TOOLS', BLUE, `Model sees: ${visibleTools.join(', ')}`)

const promptText = (captured!.systemMessage.content as ContentBlock[])
  .filter((b) => b.type === 'text')
  .map((b) => b.text)
  .join('\n')
log('PROMPT', DIM, 'System prompt includes:')
console.log(
  promptText
    .split('\n')
    .map((l) => `    ${DIM}${l}${RESET}`)
    .join('\n')
)

// ════════════════════════════════════════════════════════════
// Step 3: Start "requirements" (has NoOpMiddleware adapter)
// ════════════════════════════════════════════════════════════

header('Step 3: Start "requirements" (NoOpMiddleware adapter — should pass through)')

let result = mw.executeTransition({ task: 'requirements', status: 'in_progress' }, state, 'call-1')
if (isCommand(result)) {
  Object.assign(state, result.update)
  log('OK', GREEN, `requirements -> in_progress`)
  log('STATE', BLUE, `taskStatuses = ${JSON.stringify(state.taskStatuses)}`)
}

// ════════════════════════════════════════════════════════════
// Step 4: wrapModelCall — "requirements" active with NoOp adapter
// ════════════════════════════════════════════════════════════

header('Step 4: wrapModelCall — "requirements" active (NoOp adapter delegates to handler)')

captured = null
mw.wrapModelCall(mockModelRequest(state, mw.tools), (r) => {
  captured = r
  return {}
})
log('TOOLS', BLUE, `Model sees: ${captured!.tools.map((t) => t.name).join(', ')}`)
log('OK', GREEN, 'No crash — NoOp adapter correctly passed through to handler')

// ════════════════════════════════════════════════════════════
// Step 5: Try to use a tool from another task (should be rejected)
// ════════════════════════════════════════════════════════════

header('Step 5: Try to call write_design (wrong task — should be rejected)')

const badToolReq: ToolCallRequest = {
  toolCall: { name: 'write_design', args: {}, id: 'call-2' },
  state,
}
const badResult = mw.wrapToolCall(badToolReq, () => ({
  content: 'should not reach',
  toolCallId: 'call-2',
}))
if (!isCommand(badResult)) {
  log('REJECTED', RED, badResult.content)
}

// ════════════════════════════════════════════════════════════
// Step 6: Try to skip ahead (should be rejected)
// ════════════════════════════════════════════════════════════

header('Step 6: Try to start design before completing requirements')

result = mw.executeTransition({ task: 'design', status: 'in_progress' }, state, 'call-3')
if (!isCommand(result)) {
  log('REJECTED', RED, result.content)
}

// ════════════════════════════════════════════════════════════
// Step 7: Complete "requirements", start "design"
// ════════════════════════════════════════════════════════════

header('Step 7: Complete "requirements"')

result = mw.executeTransition({ task: 'requirements', status: 'complete' }, state, 'call-4')
if (isCommand(result)) {
  Object.assign(state, result.update)
  log('OK', GREEN, 'requirements -> complete')
}

header('Step 8: Start "design" (lifecycle hooks fire with post-transition state)')

// Use wrapToolCall so lifecycle hooks fire
const startDesignReq: ToolCallRequest = {
  toolCall: {
    name: 'update_task_status',
    args: { task: 'design', status: 'in_progress' },
    id: 'call-5',
  },
  state,
}
result = mw.wrapToolCall(startDesignReq, (req) =>
  mw.executeTransition(
    req.toolCall.args as { task: string; status: string },
    req.state,
    req.toolCall.id
  )
)
if (isCommand(result)) {
  Object.assign(state, result.update)
  log('OK', GREEN, 'design -> in_progress')
  log('STATE', BLUE, `taskStatuses = ${JSON.stringify(state.taskStatuses)}`)
}

// ════════════════════════════════════════════════════════════
// Step 9: Try to complete "design" without writing (validation rejects)
// ════════════════════════════════════════════════════════════

header('Step 9: Try to complete "design" (validation should reject)')

const completeDesignReq: ToolCallRequest = {
  toolCall: {
    name: 'update_task_status',
    args: { task: 'design', status: 'complete' },
    id: 'call-6',
  },
  state,
}
result = mw.wrapToolCall(completeDesignReq, (req) =>
  mw.executeTransition(
    req.toolCall.args as { task: string; status: string },
    req.state,
    req.toolCall.id
  )
)
if (!isCommand(result)) {
  log('REJECTED', RED, result.content)
}

// ════════════════════════════════════════════════════════════
// Step 10: Fix state and complete "design"
// ════════════════════════════════════════════════════════════

header('Step 10: Set designWritten=true, then complete "design" (onComplete fires)')

state.designWritten = true
log('STATE', BLUE, 'Set designWritten = true')

const completeDesignReq2: ToolCallRequest = {
  toolCall: {
    name: 'update_task_status',
    args: { task: 'design', status: 'complete' },
    id: 'call-7',
  },
  state,
}
result = mw.wrapToolCall(completeDesignReq2, (req) =>
  mw.executeTransition(
    req.toolCall.args as { task: string; status: string },
    req.state,
    req.toolCall.id
  )
)
if (isCommand(result)) {
  Object.assign(state, result.update)
  log('OK', GREEN, 'design -> complete')
}

// ════════════════════════════════════════════════════════════
// Step 11: Start "review" — has AuditMiddleware adapter
// ════════════════════════════════════════════════════════════

header('Step 11: Start "review" (AuditMiddleware adapter — hooks forwarded)')

const startReviewReq: ToolCallRequest = {
  toolCall: {
    name: 'update_task_status',
    args: { task: 'review', status: 'in_progress' },
    id: 'call-8',
  },
  state,
}
result = mw.wrapToolCall(startReviewReq, (req) =>
  mw.executeTransition(
    req.toolCall.args as { task: string; status: string },
    req.state,
    req.toolCall.id
  )
)
if (isCommand(result)) {
  Object.assign(state, result.update)
  log('OK', GREEN, 'review -> in_progress')
  log('STATE', BLUE, `taskStatuses = ${JSON.stringify(state.taskStatuses)}`)
}

// ════════════════════════════════════════════════════════════
// Step 12: wrapModelCall with "review" active — adapter intercepts
// ════════════════════════════════════════════════════════════

header('Step 12: wrapModelCall — "review" active (AuditMiddleware intercepts)')

captured = null
mw.wrapModelCall(mockModelRequest(state, mw.tools), (r) => {
  captured = r
  return {}
})
log('TOOLS', BLUE, `Model sees: ${captured!.tools.map((t) => t.name).join(', ')}`)
log(
  'INFO',
  BLUE,
  `audit_log tool is scoped: ${captured!.tools.some((t) => t.name === 'audit_log')}`
)

// ════════════════════════════════════════════════════════════
// Step 13: Use review_design tool — adapter intercepts wrapToolCall
// ════════════════════════════════════════════════════════════

header('Step 13: Call review_design — AuditMiddleware.wrapToolCall intercepts')

const reviewToolReq: ToolCallRequest = {
  toolCall: { name: 'review_design', args: { design: 'login page v1' }, id: 'call-9' },
  state,
}
const reviewResult = mw.wrapToolCall(reviewToolReq, () => ({
  content: 'Review looks good.',
  toolCallId: 'call-9',
}))
if (!isCommand(reviewResult)) {
  log('RESULT', GREEN, reviewResult.content)
}

// ════════════════════════════════════════════════════════════
// Step 14: Use audit_log (adapter-contributed tool)
// ════════════════════════════════════════════════════════════

header('Step 14: Call audit_log — tool contributed by adapter')

const auditToolReq: ToolCallRequest = {
  toolCall: { name: 'audit_log', args: { entry: 'design reviewed' }, id: 'call-10' },
  state,
}
const auditResult = mw.wrapToolCall(auditToolReq, () => ({
  content: 'Audit entry logged.',
  toolCallId: 'call-10',
}))
if (!isCommand(auditResult)) {
  log('RESULT', GREEN, auditResult.content)
}

// ════════════════════════════════════════════════════════════
// Step 15: Complete "review"
// ════════════════════════════════════════════════════════════

header('Step 15: Complete "review"')

result = mw.executeTransition({ task: 'review', status: 'complete' }, state, 'call-11')
if (isCommand(result)) {
  Object.assign(state, result.update)
  log('OK', GREEN, 'review -> complete')
}

log('STATE', BLUE, `taskStatuses = ${JSON.stringify(state.taskStatuses)}`)

// ════════════════════════════════════════════════════════════
// Step 16: afterAgent — all tasks complete, no nudge needed
// ════════════════════════════════════════════════════════════

header('Step 16: afterAgent — check if agent can exit')

const nudge = mw.afterAgent(state as any)
if (nudge === null) {
  log('OK', GREEN, 'All required tasks complete — agent can exit.')
} else {
  log('NUDGE', YELLOW, JSON.stringify(nudge))
}

// ════════════════════════════════════════════════════════════
// Step 17: Adapter call summary
// ════════════════════════════════════════════════════════════

header('Step 17: AuditMiddleware adapter call summary')

log('INFO', BLUE, `AuditMiddleware.wrapModelCall invoked ${auditMw.modelCallCount} time(s)`)
log('INFO', BLUE, `AuditMiddleware.wrapToolCall invoked ${auditMw.toolCallCount} time(s)`)

// ════════════════════════════════════════════════════════════
// Bonus: afterAgent nudge with incomplete tasks
// ════════════════════════════════════════════════════════════

header('Bonus: afterAgent nudge with incomplete tasks')

const incompleteState = {
  messages: [],
  taskStatuses: {
    requirements: 'complete',
    design: 'in_progress',
    review: 'pending',
  },
}
const nudgeResult = mw.afterAgent(incompleteState)
if (nudgeResult) {
  log('NUDGE', YELLOW, `jumpTo: ${nudgeResult.jumpTo}`)
  log('NUDGE', YELLOW, `nudgeCount: ${nudgeResult.nudgeCount}`)
  log('NUDGE', YELLOW, `message: ${(nudgeResult.messages[0] as any).content}`)
}

// ════════════════════════════════════════════════════════════
// Bonus: wrapModelCall with null system message
// ════════════════════════════════════════════════════════════

header('Bonus: wrapModelCall with null system message — no crash')

const nullSysMsgReq: ModelRequest = {
  state,
  systemMessage: null as unknown as any,
  tools: mw.tools,
  override(overrides) {
    return {
      state,
      systemMessage: overrides.systemMessage ?? this.systemMessage,
      tools: overrides.tools ?? this.tools,
      override: this.override,
    }
  },
}
let nullCaptured: ModelRequest | null = null
mw.wrapModelCall(nullSysMsgReq, (r) => {
  nullCaptured = r
  return {}
})
const nullBlocks = getContentBlocks(nullCaptured!.systemMessage)
log(
  'OK',
  GREEN,
  `Pipeline block injected: ${nullBlocks.some((b) => b.text?.includes('<task_pipeline>'))}`
)

console.log(`\n${GREEN}Demo complete!${RESET}\n`)
