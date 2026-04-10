import { describe, it, expect, vi } from 'vitest'
import {
  TaskSteeringMiddleware,
  TaskStatus,
  TaskMiddleware,
  getContentBlocks,
  type Task,
  type ModelRequest,
  type ToolCallRequest,
  type ToolLike,
  type ToolMessageResult,
  type CommandResult,
  type SystemMessageLike,
  type ToolCallHandler,
  type ModelCallHandler,
} from '../src/index.js'

// ── Mock tools ──────────────────────────────────────────

const toolA: ToolLike = { name: 'tool_a', description: 'Tool A' }
const toolB: ToolLike = { name: 'tool_b', description: 'Tool B' }
const toolC: ToolLike = { name: 'tool_c', description: 'Tool C' }
const globalRead: ToolLike = {
  name: 'global_read',
  description: 'Global read tool',
}

// ── Mock request helpers ────────────────────────────────

function mockModelRequest(opts: {
  state: Record<string, unknown>
  systemMessage: SystemMessageLike
  tools: ToolLike[]
}): ModelRequest {
  return {
    ...opts,
    override(overrides) {
      return mockModelRequest({
        state: opts.state,
        systemMessage: overrides.systemMessage ?? opts.systemMessage,
        tools: overrides.tools ?? opts.tools,
      })
    },
  }
}

function mockToolCallRequest(opts: {
  toolCall: { name: string; args: Record<string, unknown>; id: string }
  state: Record<string, unknown>
}): ToolCallRequest {
  return opts
}

// ── Reusable middleware subclasses ───────────────────────

class RejectCompletionMiddleware extends TaskMiddleware {
  constructor(private reason: string = 'Not ready yet.') {
    super()
  }
  validateCompletion(_state: Record<string, unknown>): string | null {
    return this.reason
  }
}

class AllowCompletionMiddleware extends TaskMiddleware {
  validateCompletion(_state: Record<string, unknown>): string | null {
    return null
  }
}

class ToolGateMiddleware extends TaskMiddleware {
  constructor(
    private gateTool: string,
    private stateKey: string,
    private minValue: number
  ) {
    super()
  }

  validateCompletion(state: Record<string, unknown>): string | null {
    const val = (state[this.stateKey] as number) ?? 0
    if (val < this.minValue) {
      return `${this.stateKey} is ${val}, need >= ${this.minValue}.`
    }
    return null
  }

  wrapToolCall(
    request: ToolCallRequest,
    handler: ToolCallHandler
  ): ToolMessageResult | CommandResult {
    if (request.toolCall.name === this.gateTool) {
      const val = (request.state[this.stateKey] as number) ?? 0
      if (val < this.minValue) {
        return {
          content: `Cannot use ${this.gateTool}: ${this.stateKey}=${val}, need >= ${this.minValue}.`,
          toolCallId: request.toolCall.id,
        }
      }
    }
    return handler(request)
  }
}

// ── Fixtures ────────────────────────────────────────────

function threeTasks(): Task[] {
  return [
    { name: 'step_1', instruction: 'Do step 1.', tools: [toolA] },
    { name: 'step_2', instruction: 'Do step 2.', tools: [toolB] },
    { name: 'step_3', instruction: 'Do step 3.', tools: [toolC] },
  ]
}

function createMiddleware(): TaskSteeringMiddleware {
  return new TaskSteeringMiddleware({
    tasks: threeTasks(),
    globalTools: [globalRead],
  })
}

function extractText(request: ModelRequest): string {
  const content = request.systemMessage.content
  if (typeof content === 'string') return content
  return (content as Array<{ type: string; text?: string }>)
    .filter((b) => b.type === 'text')
    .map((b) => b.text ?? '')
    .join('\n')
}

// ════════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════════

describe('Init', () => {
  it('requires at least one task', () => {
    expect(() => new TaskSteeringMiddleware({ tasks: [] })).toThrow('At least one Task')
  })

  it('preserves task order', () => {
    const mw = createMiddleware()
    expect((mw as any)._taskOrder).toEqual(['step_1', 'step_2', 'step_3'])
  })

  it('builds task map', () => {
    const mw = createMiddleware()
    const map = (mw as any)._taskMap as Map<string, Task>
    expect([...map.keys()]).toEqual(['step_1', 'step_2', 'step_3'])
    expect(map.get('step_1')!.instruction).toBe('Do step 1.')
  })

  it('auto-registers all tools', () => {
    const mw = createMiddleware()
    const names = new Set(mw.tools.map((t) => t.name))
    expect(names).toEqual(
      new Set(['update_task_status', 'tool_a', 'tool_b', 'tool_c', 'global_read'])
    )
  })

  it('deduplicates tools', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [
        { name: 'a', instruction: 'A', tools: [toolA] },
        { name: 'b', instruction: 'B', tools: [toolA] },
      ],
    })
    const count = mw.tools.filter((t) => t.name === 'tool_a').length
    expect(count).toBe(1)
  })

  it('rejects duplicate task names', () => {
    expect(
      () =>
        new TaskSteeringMiddleware({
          tasks: [
            { name: 'a', instruction: 'A', tools: [toolA] },
            { name: 'a', instruction: 'B', tools: [toolB] },
          ],
        })
    ).toThrow('Duplicate task names')
  })

  it('enforceOrder defaults to true', () => {
    const mw = createMiddleware()
    expect((mw as any)._enforceOrder).toBe(true)
  })

  it('enforceOrder can be set to false', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      enforceOrder: false,
    })
    expect((mw as any)._enforceOrder).toBe(false)
  })
})

// ════════════════════════════════════════════════════════════
// beforeAgent — state initialization
// ════════════════════════════════════════════════════════════

describe('beforeAgent', () => {
  it('initializes all tasks as pending', () => {
    const mw = createMiddleware()
    const result = mw.beforeAgent({ messages: [] })
    expect(result).not.toBeNull()
    expect(result!.taskStatuses).toEqual({
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    })
  })

  it('noop when already initialized', () => {
    const mw = createMiddleware()
    const state = {
      messages: [],
      taskStatuses: {
        step_1: 'complete',
        step_2: 'in_progress',
        step_3: 'pending',
      },
    }
    expect(mw.beforeAgent(state)).toBeNull()
  })

  it('preserves existing statuses', () => {
    const mw = createMiddleware()
    const state = {
      messages: [],
      taskStatuses: {
        step_1: 'in_progress',
        step_2: 'pending',
        step_3: 'pending',
      },
    }
    expect(mw.beforeAgent(state)).toBeNull()
  })
})

// ════════════════════════════════════════════════════════════
// Internal helpers — status reading
// ════════════════════════════════════════════════════════════

describe('Status helpers', () => {
  it('defaults to pending when no state', () => {
    const mw = createMiddleware()
    const statuses = (mw as any)._getStatuses({})
    expect(Object.values(statuses).every((v: string) => v === 'pending')).toBe(true)
    expect(Object.keys(statuses).length).toBe(3)
  })

  it('reads from state', () => {
    const mw = createMiddleware()
    const statuses = (mw as any)._getStatuses({
      taskStatuses: {
        step_1: 'complete',
        step_2: 'in_progress',
        step_3: 'pending',
      },
    })
    expect(statuses).toEqual({
      step_1: 'complete',
      step_2: 'in_progress',
      step_3: 'pending',
    })
  })

  it('handles null taskStatuses', () => {
    const mw = createMiddleware()
    const statuses = (mw as any)._getStatuses({ taskStatuses: null })
    expect(Object.values(statuses).every((v: string) => v === 'pending')).toBe(true)
  })

  it('activeTask returns null when all pending', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    }
    expect((mw as any)._activeTask(statuses)).toBeNull()
  })

  it('activeTask returns null when all complete', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'complete',
      step_2: 'complete',
      step_3: 'complete',
    }
    expect((mw as any)._activeTask(statuses)).toBeNull()
  })

  it('activeTask finds in_progress', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'complete',
      step_2: 'in_progress',
      step_3: 'pending',
    }
    expect((mw as any)._activeTask(statuses)).toBe('step_2')
  })

  it('activeTask returns first in_progress', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'in_progress',
      step_2: 'in_progress',
      step_3: 'pending',
    }
    expect((mw as any)._activeTask(statuses)).toBe('step_1')
  })
})

// ════════════════════════════════════════════════════════════
// Prompt rendering
// ════════════════════════════════════════════════════════════

describe('Prompt rendering', () => {
  it('renders all pending with no active', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    }
    const block = mw._renderStatusBlock(statuses, null)
    expect(block).toContain('<task_pipeline>')
    expect(block).toContain('[ ] step_1 (pending)')
    expect(block).toContain('[ ] step_2 (pending)')
    expect(block).toContain('[ ] step_3 (pending)')
    expect(block).not.toContain('<current_task')
    expect(block).toContain('</task_pipeline>')
  })

  it('shows active task instruction', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'in_progress',
      step_2: 'pending',
      step_3: 'pending',
    }
    const block = mw._renderStatusBlock(statuses, 'step_1')
    expect(block).toContain('[>] step_1 (in_progress)')
    expect(block).toContain('<current_task name="step_1">')
    expect(block).toContain('Do step 1.')
  })

  it('renders mixed statuses', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'complete',
      step_2: 'complete',
      step_3: 'in_progress',
    }
    const block = mw._renderStatusBlock(statuses, 'step_3')
    expect(block).toContain('[x] step_1 (complete)')
    expect(block).toContain('[x] step_2 (complete)')
    expect(block).toContain('[>] step_3 (in_progress)')
    expect(block).toContain('Do step 3.')
  })

  it('shows rules when enforceOrder is true', () => {
    const mw = createMiddleware()
    const statuses = {
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    }
    const block = mw._renderStatusBlock(statuses, null)
    expect(block).toContain('<rules>')
    expect(block).toContain('Required order: step_1 -> step_2 -> step_3')
    expect(block).toContain('Do not skip tasks.')
  })

  it('no rules when enforceOrder is false', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      enforceOrder: false,
    })
    const statuses = {
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    }
    const block = mw._renderStatusBlock(statuses, null)
    expect(block).not.toContain('<rules>')
  })
})

// ════════════════════════════════════════════════════════════
// Tool scoping
// ════════════════════════════════════════════════════════════

describe('Tool scoping', () => {
  it('no active task returns globals + transition', () => {
    const mw = createMiddleware()
    const names = mw._allowedToolNames(null)
    expect(names).toEqual(new Set(['update_task_status', 'global_read']))
  })

  it('step_1 active', () => {
    const mw = createMiddleware()
    const names = mw._allowedToolNames('step_1')
    expect(names.has('tool_a')).toBe(true)
    expect(names.has('update_task_status')).toBe(true)
    expect(names.has('global_read')).toBe(true)
    expect(names.has('tool_b')).toBe(false)
    expect(names.has('tool_c')).toBe(false)
  })

  it('step_2 active', () => {
    const mw = createMiddleware()
    const names = mw._allowedToolNames('step_2')
    expect(names.has('tool_b')).toBe(true)
    expect(names.has('tool_a')).toBe(false)
    expect(names.has('tool_c')).toBe(false)
  })

  it('step_3 active', () => {
    const mw = createMiddleware()
    const names = mw._allowedToolNames('step_3')
    expect(names.has('tool_c')).toBe(true)
    expect(names.has('tool_a')).toBe(false)
    expect(names.has('tool_b')).toBe(false)
  })
})

// ════════════════════════════════════════════════════════════
// wrapModelCall — prompt injection + tool filtering
// ════════════════════════════════════════════════════════════

describe('wrapModelCall', () => {
  function makeRequest(
    mw: TaskSteeringMiddleware,
    taskStatuses: Record<string, string>
  ): ModelRequest {
    return mockModelRequest({
      state: { taskStatuses },
      systemMessage: { content: 'You are helpful.' },
      tools: mw.tools,
    })
  }

  it('appends pipeline block', () => {
    const mw = createMiddleware()
    const request = makeRequest(mw, {
      step_1: 'in_progress',
      step_2: 'pending',
      step_3: 'pending',
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (r) => {
      captured = r
      return {}
    })

    const text = extractText(captured!)
    expect(text).toContain('You are helpful.')
    expect(text).toContain('<task_pipeline>')
    expect(text).toContain('[>] step_1 (in_progress)')
    expect(text).toContain('Do step 1.')
  })

  it('preserves base prompt', () => {
    const mw = createMiddleware()
    const request = makeRequest(mw, {
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (r) => {
      captured = r
      return {}
    })

    const text = extractText(captured!)
    expect(text.startsWith('You are helpful.')).toBe(true)
  })

  it('scopes tools to active task', () => {
    const mw = createMiddleware()
    const request = makeRequest(mw, {
      step_1: 'complete',
      step_2: 'in_progress',
      step_3: 'pending',
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (r) => {
      captured = r
      return {}
    })

    const toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames).toEqual(new Set(['tool_b', 'global_read', 'update_task_status']))
  })

  it('no active task shows only globals', () => {
    const mw = createMiddleware()
    const request = makeRequest(mw, {
      step_1: 'pending',
      step_2: 'pending',
      step_3: 'pending',
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (r) => {
      captured = r
      return {}
    })

    const toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames).toEqual(new Set(['global_read', 'update_task_status']))
  })

  it('all complete shows only globals', () => {
    const mw = createMiddleware()
    const request = makeRequest(mw, {
      step_1: 'complete',
      step_2: 'complete',
      step_3: 'complete',
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (r) => {
      captured = r
      return {}
    })

    const toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames).toEqual(new Set(['global_read', 'update_task_status']))
  })

  it('delegates to task middleware', () => {
    class SpyMiddleware extends TaskMiddleware {
      receivedRequest: ModelRequest | null = null
      wrapModelCall(request: ModelRequest, handler: ModelCallHandler): unknown {
        this.receivedRequest = request
        return handler(request)
      }
    }

    const spy = new SpyMiddleware()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: spy }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: { content: 'Base' },
      tools: mw.tools,
    })

    mw.wrapModelCall(request, () => ({}))

    expect(spy.receivedRequest).not.toBeNull()
    const text = extractText(spy.receivedRequest!)
    expect(text).toContain('<task_pipeline>')
  })

  it('works without task middleware', () => {
    const mw = createMiddleware()
    const request = makeRequest(mw, {
      step_1: 'in_progress',
      step_2: 'pending',
      step_3: 'pending',
    })

    const handler = vi.fn(() => ({}))
    mw.wrapModelCall(request, handler)
    expect(handler).toHaveBeenCalledOnce()
  })
})

// ════════════════════════════════════════════════════════════
// wrapToolCall — completion validation + delegation
// ════════════════════════════════════════════════════════════

describe('wrapToolCall', () => {
  it('rejects completion when validator fails', () => {
    const tasks: Task[] = [
      {
        name: 'a',
        instruction: 'A',
        tools: [],
        middleware: new RejectCompletionMiddleware('Need more work.'),
      },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })

    const handler = vi.fn()
    const result = mw.wrapToolCall(request, handler)

    expect(handler).not.toHaveBeenCalled()
    expect('content' in result && 'toolCallId' in result).toBe(true)
    expect((result as ToolMessageResult).content).toContain("Cannot complete 'a'")
    expect((result as ToolMessageResult).content).toContain('Need more work.')
  })

  it('allows completion when validator passes', () => {
    const tasks: Task[] = [
      {
        name: 'a',
        instruction: 'A',
        tools: [],
        middleware: new AllowCompletionMiddleware(),
      },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })

    const expected: ToolMessageResult = {
      content: 'ok',
      toolCallId: 'call-1',
    }
    const handler = vi.fn(() => expected)
    const result = mw.wrapToolCall(request, handler)

    expect(handler).toHaveBeenCalledOnce()
    expect(result).toBe(expected)
  })

  it('in_progress skips completion validation', () => {
    const tasks: Task[] = [
      {
        name: 'a',
        instruction: 'A',
        tools: [],
        middleware: new RejectCompletionMiddleware('Should not fire.'),
      },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'pending' } },
    })

    const handler = vi.fn(() => ({
      content: 'ok',
      toolCallId: 'call-1',
    }))
    mw.wrapToolCall(request, handler)
    expect(handler).toHaveBeenCalledOnce()
  })

  it('no middleware allows completion', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [] }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })

    const handler = vi.fn(() => ({
      content: 'ok',
      toolCallId: 'call-1',
    }))
    mw.wrapToolCall(request, handler)
    expect(handler).toHaveBeenCalledOnce()
  })

  it('delegates non-transition tool to task middleware', () => {
    const gate = new ToolGateMiddleware('tool_a', 'count', 5)
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: gate }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: { name: 'tool_a', args: {}, id: 'call-1' },
      state: { taskStatuses: { a: 'in_progress' }, count: 2 },
    })

    const handler = vi.fn()
    const result = mw.wrapToolCall(request, handler)

    expect(handler).not.toHaveBeenCalled()
    expect((result as ToolMessageResult).content).toContain('Cannot use tool_a')
    expect((result as ToolMessageResult).content).toContain('count=2, need >= 5')
  })

  it('task middleware allows tool when condition met', () => {
    const gate = new ToolGateMiddleware('tool_a', 'count', 5)
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: gate }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: { name: 'tool_a', args: {}, id: 'call-1' },
      state: { taskStatuses: { a: 'in_progress' }, count: 10 },
    })

    const expected: ToolMessageResult = {
      content: 'ok',
      toolCallId: 'call-1',
    }
    const handler = vi.fn(() => expected)
    const result = mw.wrapToolCall(request, handler)

    expect(handler).toHaveBeenCalledOnce()
    expect(result).toBe(expected)
  })

  it('rejects out-of-scope tool when no active task', () => {
    const mw = createMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: 'tool_a', args: {}, id: 'call-1' },
      state: {
        taskStatuses: {
          step_1: 'pending',
          step_2: 'pending',
          step_3: 'pending',
        },
      },
    })

    const handler = vi.fn()
    const result = mw.wrapToolCall(request, handler)

    expect(handler).not.toHaveBeenCalled()
    expect((result as ToolMessageResult).content).toContain('not available')
  })

  it('allows global tool when no active task', () => {
    const mw = createMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: 'global_read', args: {}, id: 'call-1' },
      state: {
        taskStatuses: {
          step_1: 'pending',
          step_2: 'pending',
          step_3: 'pending',
        },
      },
    })

    const expected: ToolMessageResult = {
      content: 'ok',
      toolCallId: 'call-1',
    }
    const handler = vi.fn(() => expected)
    mw.wrapToolCall(request, handler)
    expect(handler).toHaveBeenCalledOnce()
  })
})

// ════════════════════════════════════════════════════════════
// Lifecycle hooks — onStart / onComplete
// ════════════════════════════════════════════════════════════

describe('Lifecycle hooks', () => {
  function makeLifecycleMiddleware() {
    class LifecycleSpy extends TaskMiddleware {
      started = false
      completed = false

      onStart(_state: Record<string, unknown>): void {
        this.started = true
      }

      onComplete(_state: Record<string, unknown>): void {
        this.completed = true
      }
    }

    const spy = new LifecycleSpy()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: spy }]
    const mw = new TaskSteeringMiddleware({ tasks })
    return { mw, spy }
  }

  it('onStart called on in_progress transition', () => {
    const { mw, spy } = makeLifecycleMiddleware()
    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'pending' } },
    })

    const handler = vi.fn(() => ({ update: {} }))
    mw.wrapToolCall(request, handler)

    expect(spy.started).toBe(true)
    expect(spy.completed).toBe(false)
  })

  it('onComplete called on complete transition', () => {
    const { mw, spy } = makeLifecycleMiddleware()
    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })

    const handler = vi.fn(() => ({ update: {} }))
    mw.wrapToolCall(request, handler)

    expect(spy.completed).toBe(true)
    expect(spy.started).toBe(false)
  })

  it('hooks not called on failed transition', () => {
    const { mw, spy } = makeLifecycleMiddleware()
    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'pending' } },
    })

    // Handler returns ToolMessageResult (not Command) — indicates error
    const handler = vi.fn(() => ({
      content: 'Error',
      toolCallId: 'call-1',
    }))
    mw.wrapToolCall(request, handler)

    expect(spy.started).toBe(false)
    expect(spy.completed).toBe(false)
  })

  it('onStart receives state', () => {
    const receivedState: Record<string, unknown> = {}

    class StateSpy extends TaskMiddleware {
      onStart(state: Record<string, unknown>): void {
        Object.assign(receivedState, state)
      }
    }

    const spy = new StateSpy()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [], middleware: spy }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const state = { taskStatuses: { a: 'pending' }, customField: 42 }
    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state,
    })

    const handler = vi.fn(() => ({ update: {} }))
    mw.wrapToolCall(request, handler)

    expect(receivedState.customField).toBe(42)
  })
})

// ════════════════════════════════════════════════════════════
// Tool gating — out-of-scope tool rejection
// ════════════════════════════════════════════════════════════

describe('Tool gating', () => {
  it('rejects wrong task tool', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'b', instruction: 'B', tools: [toolB] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: { name: 'tool_b', args: {}, id: 'call-1' },
      state: { taskStatuses: { a: 'in_progress', b: 'pending' } },
    })

    const handler = vi.fn()
    const result = mw.wrapToolCall(request, handler)

    expect(handler).not.toHaveBeenCalled()
    expect((result as ToolMessageResult).content).toContain('not available')
  })

  it('allows correct task tool', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'b', instruction: 'B', tools: [toolB] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: { name: 'tool_a', args: {}, id: 'call-1' },
      state: { taskStatuses: { a: 'in_progress', b: 'pending' } },
    })

    const expected: ToolMessageResult = {
      content: 'ok',
      toolCallId: 'call-1',
    }
    const handler = vi.fn(() => expected)
    mw.wrapToolCall(request, handler)
    expect(handler).toHaveBeenCalledOnce()
  })

  it('transition tool always allowed', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'pending' } },
    })

    const handler = vi.fn(() => ({ update: {} }))
    mw.wrapToolCall(request, handler)
    expect(handler).toHaveBeenCalledOnce()
  })
})

// ════════════════════════════════════════════════════════════
// executeTransition — transition tool logic
// ════════════════════════════════════════════════════════════

describe('executeTransition', () => {
  it('rejects invalid task name', () => {
    const mw = createMiddleware()
    const result = mw.executeTransition(
      { task: 'nonexistent', status: 'in_progress' },
      { taskStatuses: {} },
      'call-1'
    )
    expect('content' in result).toBe(true)
    expect((result as ToolMessageResult).content).toContain('Invalid task')
  })

  it('rejects invalid status', () => {
    const mw = createMiddleware()
    const result = mw.executeTransition(
      { task: 'step_1', status: 'invalid' },
      { taskStatuses: {} },
      'call-1'
    )
    expect((result as ToolMessageResult).content).toContain('Invalid status')
  })

  it('rejects invalid transition', () => {
    const mw = createMiddleware()
    const result = mw.executeTransition(
      { task: 'step_1', status: 'complete' },
      { taskStatuses: { step_1: 'pending' } },
      'call-1'
    )
    expect((result as ToolMessageResult).content).toContain('Cannot transition')
  })

  it('allows valid pending -> in_progress', () => {
    const mw = createMiddleware()
    const result = mw.executeTransition(
      { task: 'step_1', status: 'in_progress' },
      { taskStatuses: { step_1: 'pending', step_2: 'pending', step_3: 'pending' } },
      'call-1'
    )
    expect('update' in result).toBe(true)
    const cmd = result as CommandResult
    expect((cmd.update.taskStatuses as Record<string, string>).step_1).toBe('in_progress')
  })

  it('allows valid in_progress -> complete', () => {
    const mw = createMiddleware()
    const result = mw.executeTransition(
      { task: 'step_1', status: 'complete' },
      { taskStatuses: { step_1: 'in_progress', step_2: 'pending', step_3: 'pending' } },
      'call-1'
    )
    expect('update' in result).toBe(true)
    const cmd = result as CommandResult
    expect((cmd.update.taskStatuses as Record<string, string>).step_1).toBe('complete')
  })

  it('enforces ordering', () => {
    const mw = createMiddleware()
    const result = mw.executeTransition(
      { task: 'step_2', status: 'in_progress' },
      { taskStatuses: { step_1: 'pending', step_2: 'pending', step_3: 'pending' } },
      'call-1'
    )
    expect((result as ToolMessageResult).content).toContain("Cannot start 'step_2'")
    expect((result as ToolMessageResult).content).toContain("'step_1' is not complete")
  })

  it('allows out-of-order when enforceOrder is false', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      enforceOrder: false,
    })
    const result = mw.executeTransition(
      { task: 'step_2', status: 'in_progress' },
      { taskStatuses: { step_1: 'pending', step_2: 'pending', step_3: 'pending' } },
      'call-1'
    )
    expect('update' in result).toBe(true)
  })
})

// ════════════════════════════════════════════════════════════
// Required tasks init
// ════════════════════════════════════════════════════════════

describe('Required tasks init', () => {
  it('default is all', () => {
    const mw = new TaskSteeringMiddleware({ tasks: threeTasks() })
    expect((mw as any)._requiredTasks).toEqual(new Set(['step_1', 'step_2', 'step_3']))
  })

  it('wildcard resolves to all', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      requiredTasks: ['*'],
    })
    expect((mw as any)._requiredTasks).toEqual(new Set(['step_1', 'step_2', 'step_3']))
  })

  it('explicit subset', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      requiredTasks: ['step_1', 'step_3'],
    })
    expect((mw as any)._requiredTasks).toEqual(new Set(['step_1', 'step_3']))
  })

  it('null means no required', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      requiredTasks: null,
    })
    expect((mw as any)._requiredTasks).toEqual(new Set())
  })

  it('unknown task raises', () => {
    expect(
      () =>
        new TaskSteeringMiddleware({
          tasks: threeTasks(),
          requiredTasks: ['nonexistent'],
        })
    ).toThrow('Unknown required tasks')
  })

  it('maxNudges defaults to 3', () => {
    const mw = new TaskSteeringMiddleware({ tasks: threeTasks() })
    expect((mw as any)._maxNudges).toBe(3)
  })

  it('maxNudges custom', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: threeTasks(),
      maxNudges: 5,
    })
    expect((mw as any)._maxNudges).toBe(5)
  })
})

// ════════════════════════════════════════════════════════════
// afterAgent — required task nudging
// ════════════════════════════════════════════════════════════

describe('afterAgent', () => {
  it('nudges when required tasks incomplete', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'b', instruction: 'B', tools: [toolB] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const state = {
      messages: [],
      taskStatuses: { a: 'complete', b: 'pending' },
    }

    const result = mw.afterAgent(state)
    expect(result).not.toBeNull()
    expect(result!.jumpTo).toBe('model')
    expect(result!.nudgeCount).toBe(1)
    expect(result!.messages.length).toBe(1)
    expect((result!.messages[0] as any).content).toContain('b')
    expect((result!.messages[0] as any).content).toContain('required tasks')
  })

  it('no nudge when all complete', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'b', instruction: 'B', tools: [toolB] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const state = {
      messages: [],
      taskStatuses: { a: 'complete', b: 'complete' },
    }

    expect(mw.afterAgent(state)).toBeNull()
  })

  it('no nudge when requiredTasks is null', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({
      tasks,
      requiredTasks: null,
    })

    const state = {
      messages: [],
      taskStatuses: { a: 'pending' },
    }

    expect(mw.afterAgent(state)).toBeNull()
  })

  it('only checks required subset', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'b', instruction: 'B', tools: [toolB] },
    ]
    const mw = new TaskSteeringMiddleware({
      tasks,
      requiredTasks: ['a'],
    })

    const state = {
      messages: [],
      taskStatuses: { a: 'complete', b: 'pending' },
    }

    // b is incomplete but not required
    expect(mw.afterAgent(state)).toBeNull()
  })

  it('nudge lists only incomplete required', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'b', instruction: 'B', tools: [toolB] },
      { name: 'c', instruction: 'C', tools: [toolC] },
    ]
    const mw = new TaskSteeringMiddleware({
      tasks,
      requiredTasks: ['a', 'c'],
    })

    const state = {
      messages: [],
      taskStatuses: { a: 'complete', b: 'pending', c: 'pending' },
    }

    const result = mw.afterAgent(state)
    expect(result).not.toBeNull()
    const msg = (result!.messages[0] as any).content
    expect(msg).toContain('required tasks: c.')
  })

  it('stops nudging after maxNudges', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({ tasks, maxNudges: 2 })

    const state = {
      messages: [],
      taskStatuses: { a: 'pending' },
      nudgeCount: 2,
    }

    expect(mw.afterAgent(state)).toBeNull()
  })

  it('increments nudgeCount', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({ tasks, maxNudges: 3 })

    const state = {
      messages: [],
      taskStatuses: { a: 'pending' },
      nudgeCount: 1,
    }

    const result = mw.afterAgent(state)
    expect(result).not.toBeNull()
    expect(result!.nudgeCount).toBe(2)
  })

  it('nudgeCount defaults to zero', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const state = {
      messages: [],
      taskStatuses: { a: 'pending' },
    }

    const result = mw.afterAgent(state)
    expect(result).not.toBeNull()
    expect(result!.nudgeCount).toBe(1)
  })

  it('nudges in_progress task', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const state = {
      messages: [],
      taskStatuses: { a: 'in_progress' },
    }

    const result = mw.afterAgent(state)
    expect(result).not.toBeNull()
    expect((result!.messages[0] as any).content).toContain('a')
  })
})

// ════════════════════════════════════════════════════════════
// End-to-end scenario (unit-level, no real model)
// ════════════════════════════════════════════════════════════

describe('Scenario', () => {
  it('full lifecycle', () => {
    const tasks: Task[] = [
      { name: 'collect', instruction: 'Collect items.', tools: [toolA] },
      {
        name: 'review',
        instruction: 'Review collected items.',
        tools: [toolB],
        middleware: new RejectCompletionMiddleware('Items not reviewed.'),
      },
      { name: 'finalize', instruction: 'Finalize.', tools: [toolC] },
    ]
    const mw = new TaskSteeringMiddleware({
      tasks,
      globalTools: [globalRead],
    })

    // 1. beforeAgent initializes state
    const state: Record<string, unknown> = { messages: [] }
    const init = mw.beforeAgent(state as any)
    expect((init!.taskStatuses as Record<string, string>).collect).toBe('pending')
    state.taskStatuses = init!.taskStatuses

    // 2. First model call — no active task, only globals + transition
    const req1 = mockModelRequest({
      state,
      systemMessage: { content: 'Base prompt.' },
      tools: mw.tools,
    })
    let captured: ModelRequest | null = null
    mw.wrapModelCall(req1, (r) => {
      captured = r
      return {}
    })
    let toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames).toEqual(new Set(['update_task_status', 'global_read']))

    // 3. Agent starts "collect" — update state
    ;(state.taskStatuses as Record<string, string>).collect = 'in_progress'

    const req2 = mockModelRequest({
      state,
      systemMessage: { content: 'Base prompt.' },
      tools: mw.tools,
    })
    captured = null
    mw.wrapModelCall(req2, (r) => {
      captured = r
      return {}
    })
    toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames.has('tool_a')).toBe(true)
    expect(toolNames.has('tool_b')).toBe(false)

    // 4. Agent completes "collect", starts "review"
    ;(state.taskStatuses as Record<string, string>).collect = 'complete'
    ;(state.taskStatuses as Record<string, string>).review = 'in_progress'

    const req3 = mockModelRequest({
      state,
      systemMessage: { content: 'Base prompt.' },
      tools: mw.tools,
    })
    captured = null
    mw.wrapModelCall(req3, (r) => {
      captured = r
      return {}
    })
    toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames.has('tool_b')).toBe(true)
    expect(toolNames.has('tool_a')).toBe(false)

    // 5. Agent tries to complete "review" — rejected
    const completeReq = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'review', status: 'complete' },
        id: 'call-99',
      },
      state,
    })
    const result = mw.wrapToolCall(completeReq, vi.fn())
    expect((result as ToolMessageResult).content).toContain("Cannot complete 'review'")
  })
})

// ════════════════════════════════════════════════════════════
// Null system message
// ════════════════════════════════════════════════════════════

describe('wrapModelCall — null system message', () => {
  it('should not crash when systemMessage is null', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: null as unknown as SystemMessageLike,
      tools: mw.tools,
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (r) => {
      captured = r
      return {}
    })

    const blocks = getContentBlocks(captured!.systemMessage)
    const text = blocks.map((b) => b.text ?? '').join('\n')
    expect(text).toContain('<task_pipeline>')
    expect(text).toContain('[>] a (in_progress)')
  })
})

// ════════════════════════════════════════════════════════════
// Lifecycle hooks — post-transition state
// ════════════════════════════════════════════════════════════

describe('lifecycle hooks — post-transition state', () => {
  it('onStart sees updated taskStatuses', () => {
    let receivedStatuses: Record<string, string> = {}

    class StateSpy extends TaskMiddleware {
      onStart(state: Record<string, unknown>): void {
        receivedStatuses = { ...(state.taskStatuses as Record<string, string>) }
      }
    }

    const spy = new StateSpy()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [], middleware: spy }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'pending' } },
    })

    const handler: ToolCallHandler = () => ({ update: { taskStatuses: { a: 'in_progress' } } })
    mw.wrapToolCall(request, handler)

    expect(receivedStatuses.a).toBe('in_progress')
  })

  it('onComplete sees updated taskStatuses', () => {
    let receivedStatuses: Record<string, string> = {}

    class StateSpy extends TaskMiddleware {
      onComplete(state: Record<string, unknown>): void {
        receivedStatuses = { ...(state.taskStatuses as Record<string, string>) }
      }
    }

    const spy = new StateSpy()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [], middleware: spy }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })

    const handler: ToolCallHandler = () => ({ update: { taskStatuses: { a: 'complete' } } })
    mw.wrapToolCall(request, handler)

    expect(receivedStatuses.a).toBe('complete')
  })
})

// ════════════════════════════════════════════════════════════
// Middleware list composition
// ════════════════════════════════════════════════════════════

describe('middleware list composition', () => {
  it('single-item list is unwrapped', () => {
    const spy = new TaskMiddleware()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [], middleware: [spy] }]
    const mw = new TaskSteeringMiddleware({ tasks })
    // Should be the same instance, not a composed wrapper
    expect((mw as any)._taskMap.get('a').middleware).toBe(spy)
  })

  it('empty list becomes undefined', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [], middleware: [] }]
    const mw = new TaskSteeringMiddleware({ tasks })
    expect((mw as any)._taskMap.get('a').middleware).toBeUndefined()
  })

  it('wrapModelCall chains in order (first = outermost)', () => {
    const callOrder: string[] = []

    class Outer extends TaskMiddleware {
      wrapModelCall(request: ModelRequest, handler: ModelCallHandler) {
        callOrder.push('outer')
        return handler(request)
      }
    }

    class Inner extends TaskMiddleware {
      wrapModelCall(request: ModelRequest, handler: ModelCallHandler) {
        callOrder.push('inner')
        return handler(request)
      }
    }

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA], middleware: [new Outer(), new Inner()] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: { content: 'Base' },
      tools: mw.tools,
    })
    mw.wrapModelCall(request, () => ({}))

    expect(callOrder).toEqual(['outer', 'inner'])
  })

  it('wrapToolCall chains in order', () => {
    const callOrder: string[] = []

    class Outer extends TaskMiddleware {
      wrapToolCall(request: ToolCallRequest, handler: ToolCallHandler) {
        callOrder.push('outer')
        return handler(request)
      }
    }

    class Inner extends TaskMiddleware {
      wrapToolCall(request: ToolCallRequest, handler: ToolCallHandler) {
        callOrder.push('inner')
        return handler(request)
      }
    }

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA], middleware: [new Outer(), new Inner()] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: { name: 'tool_a', args: {}, id: 'call-1' },
      state: { taskStatuses: { a: 'in_progress' } },
    })
    mw.wrapToolCall(request, () => ({ content: 'ok', toolCallId: 'call-1' }))

    expect(callOrder).toEqual(['outer', 'inner'])
  })

  it('validateCompletion — first error wins', () => {
    class Fail1 extends TaskMiddleware {
      validateCompletion() {
        return 'error from first'
      }
    }

    class Fail2 extends TaskMiddleware {
      validateCompletion() {
        return 'error from second'
      }
    }

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [], middleware: [new Fail1(), new Fail2()] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })
    const result = mw.wrapToolCall(request, vi.fn())
    expect((result as ToolMessageResult).content).toContain('error from first')
  })

  it('lifecycle hooks all fire', () => {
    const started: string[] = []

    class Hook1 extends TaskMiddleware {
      onStart() {
        started.push('hook1')
      }
    }

    class Hook2 extends TaskMiddleware {
      onStart() {
        started.push('hook2')
      }
    }

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [], middleware: [new Hook1(), new Hook2()] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'in_progress' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'pending' } },
    })
    const handler: ToolCallHandler = () => ({ update: { taskStatuses: { a: 'in_progress' } } })
    mw.wrapToolCall(request, handler)

    expect(started).toEqual(['hook1', 'hook2'])
  })

  it('tools merged from all middlewares', () => {
    const extraTool: ToolLike = { name: 'extra_tool', description: 'Extra' }

    class ToolMw extends TaskMiddleware {
      tools = [extraTool]
    }

    const tasks: Task[] = [
      {
        name: 'a',
        instruction: 'A',
        tools: [toolA],
        middleware: [new ToolMw(), new TaskMiddleware()],
      },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    const names = mw._allowedToolNames('a')
    expect(names.has('extra_tool')).toBe(true)
    expect(names.has('tool_a')).toBe(true)
  })
})

// ════════════════════════════════════════════════════════════
// Auto-wrapping raw agent middleware
// ════════════════════════════════════════════════════════════

describe('auto-wrapping raw agent middleware', () => {
  it('raw object with wrapModelCall is auto-wrapped', () => {
    let called = false
    const rawMw = {
      wrapModelCall(request: ModelRequest, handler: ModelCallHandler) {
        called = true
        return handler(request)
      },
    }

    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: rawMw }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: { content: 'Base' },
      tools: mw.tools,
    })
    mw.wrapModelCall(request, () => ({}))
    expect(called).toBe(true)
  })

  it('raw object in a list is auto-wrapped', () => {
    let called = false
    const rawMw = {
      wrapModelCall(request: ModelRequest, handler: ModelCallHandler) {
        called = true
        return handler(request)
      },
    }

    class Validator extends TaskMiddleware {
      validateCompletion() {
        return 'Nope.'
      }
    }

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA], middleware: [rawMw, new Validator()] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    // Model call intercepted
    const request = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: { content: 'Base' },
      tools: mw.tools,
    })
    mw.wrapModelCall(request, () => ({}))
    expect(called).toBe(true)

    // Completion rejected
    const completeReq = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })
    const result = mw.wrapToolCall(completeReq, vi.fn())
    expect((result as ToolMessageResult).content).toContain('Nope.')
  })

  it('TaskMiddleware instance is not double-wrapped', () => {
    const validator = new TaskMiddleware()
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [], middleware: validator }]
    const mw = new TaskSteeringMiddleware({ tasks })
    expect((mw as any)._taskMap.get('a').middleware).toBe(validator)
  })

  it('invalid middleware warns and is ignored', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA], middleware: 'not a middleware' as any },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    expect((mw as any)._taskMap.get('a').middleware).toBeUndefined()
    expect(warnSpy).toHaveBeenCalledOnce()
    expect(warnSpy.mock.calls[0][0]).toContain('Ignoring invalid task middleware')

    warnSpy.mockRestore()
  })

  it('invalid items in list warn and are skipped', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    class Validator extends TaskMiddleware {
      validateCompletion() {
        return 'Nope.'
      }
    }

    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA], middleware: [42 as any, new Validator()] },
    ]
    const mw = new TaskSteeringMiddleware({ tasks })

    expect((mw as any)._taskMap.get('a').middleware).toBeDefined()
    expect(warnSpy).toHaveBeenCalledOnce()

    // Validator still works
    const request = mockToolCallRequest({
      toolCall: {
        name: 'update_task_status',
        args: { task: 'a', status: 'complete' },
        id: 'call-1',
      },
      state: { taskStatuses: { a: 'in_progress' } },
    })
    const result = mw.wrapToolCall(request, vi.fn())
    expect((result as ToolMessageResult).content).toContain('Nope.')

    warnSpy.mockRestore()
  })
})
