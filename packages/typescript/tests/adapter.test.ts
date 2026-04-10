import { describe, it, expect, vi } from 'vitest'
import {
  AgentMiddlewareAdapter,
  TaskSteeringMiddleware,
  type AgentMiddlewareLike,
  type Task,
  type ModelRequest,
  type ToolCallRequest,
  type ToolLike,
  type ToolMessageResult,
  type SystemMessageLike,
  type ToolCallHandler,
  type ModelCallHandler,
} from '../src/index.js'

// ── Mock tools ──────────────────────────────────────────

const toolA: ToolLike = { name: 'tool_a', description: 'Tool A' }
const innerTool: ToolLike = { name: 'inner_tool', description: 'Inner tool' }

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

// ════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════

describe('AgentMiddlewareAdapter — construction', () => {
  it('exposes inner tools', () => {
    const inner: AgentMiddlewareLike = { tools: [innerTool] }
    const adapter = new AgentMiddlewareAdapter(inner)
    expect(adapter.tools.map((t) => t.name)).toContain('inner_tool')
  })

  it('empty tools when inner has none', () => {
    const inner: AgentMiddlewareLike = {}
    const adapter = new AgentMiddlewareAdapter(inner)
    expect(adapter.tools).toEqual([])
  })

  it('wires wrapModelCall only when present on inner', () => {
    const inner: AgentMiddlewareLike = {
      wrapModelCall: vi.fn((_req, handler) => handler(_req)),
    }
    const adapter = new AgentMiddlewareAdapter(inner)
    expect(adapter.wrapModelCall).toBeDefined()
    // wrapToolCall should NOT be wired
    expect(adapter.wrapToolCall).toBeUndefined()
  })

  it('wires wrapToolCall only when present on inner', () => {
    const inner: AgentMiddlewareLike = {
      wrapToolCall: vi.fn((_req, handler) => handler(_req)),
    }
    const adapter = new AgentMiddlewareAdapter(inner)
    expect(adapter.wrapToolCall).toBeDefined()
    expect(adapter.wrapModelCall).toBeUndefined()
  })

  it('wires both hooks when present', () => {
    const inner: AgentMiddlewareLike = {
      wrapModelCall: vi.fn((_req, handler) => handler(_req)),
      wrapToolCall: vi.fn((_req, handler) => handler(_req)),
    }
    const adapter = new AgentMiddlewareAdapter(inner)
    expect(adapter.wrapModelCall).toBeDefined()
    expect(adapter.wrapToolCall).toBeDefined()
  })

  it('no hooks wired when inner has none', () => {
    const inner: AgentMiddlewareLike = {}
    const adapter = new AgentMiddlewareAdapter(inner)
    expect(adapter.wrapModelCall).toBeUndefined()
    expect(adapter.wrapToolCall).toBeUndefined()
  })
})

// ════════════════════════════════════════════════════════════
// Hook delegation
// ════════════════════════════════════════════════════════════

describe('AgentMiddlewareAdapter — delegation', () => {
  it('delegates wrapModelCall to inner', () => {
    const innerFn = vi.fn((_req: ModelRequest, handler: ModelCallHandler) => handler(_req))
    const inner: AgentMiddlewareLike = { wrapModelCall: innerFn }
    const adapter = new AgentMiddlewareAdapter(inner)

    const handler = vi.fn(() => 'result')
    adapter.wrapModelCall!(
      mockModelRequest({
        state: {},
        systemMessage: { content: 'test' },
        tools: [],
      }),
      handler
    )

    expect(innerFn).toHaveBeenCalledOnce()
    expect(handler).toHaveBeenCalledOnce()
  })

  it('delegates wrapToolCall to inner', () => {
    const innerFn = vi.fn((_req: ToolCallRequest, handler: ToolCallHandler) => handler(_req))
    const inner: AgentMiddlewareLike = { wrapToolCall: innerFn }
    const adapter = new AgentMiddlewareAdapter(inner)

    const handler: ToolCallHandler = vi.fn(() => ({ content: 'ok', toolCallId: 'call-1' }))
    adapter.wrapToolCall!(
      mockToolCallRequest({
        toolCall: { name: 'test', args: {}, id: 'call-1' },
        state: {},
      }),
      handler
    )

    expect(innerFn).toHaveBeenCalledOnce()
    expect(handler).toHaveBeenCalledOnce()
  })
})

// ════════════════════════════════════════════════════════════
// Integration with TaskSteeringMiddleware
// ════════════════════════════════════════════════════════════

describe('AgentMiddlewareAdapter — integration', () => {
  it('adapter tools are scoped when task is active', () => {
    const inner: AgentMiddlewareLike = { tools: [innerTool] }
    const adapter = new AgentMiddlewareAdapter(inner)
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: adapter }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const names = mw._allowedToolNames('a')
    expect(names.has('inner_tool')).toBe(true)
    expect(names.has('tool_a')).toBe(true)
  })

  it('adapter delegates wrapModelCall in pipeline', () => {
    const innerFn = vi.fn((_req: ModelRequest, handler: ModelCallHandler) => handler(_req))
    const inner: AgentMiddlewareLike = { wrapModelCall: innerFn }
    const adapter = new AgentMiddlewareAdapter(inner)
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: adapter }]
    const mw = new TaskSteeringMiddleware({ tasks })

    const request = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: { content: 'Base' },
      tools: mw.tools,
    })

    mw.wrapModelCall(request, () => ({}))
    expect(innerFn).toHaveBeenCalledOnce()
  })

  it('no-op adapter does not crash in pipeline', () => {
    const inner: AgentMiddlewareLike = {}
    const adapter = new AgentMiddlewareAdapter(inner)
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA], middleware: adapter }]
    const mw = new TaskSteeringMiddleware({ tasks })

    // wrap_model_call
    const modelReq = mockModelRequest({
      state: { taskStatuses: { a: 'in_progress' } },
      systemMessage: { content: 'Base' },
      tools: mw.tools,
    })
    const modelHandler = vi.fn(() => ({}))
    mw.wrapModelCall(modelReq, modelHandler)
    expect(modelHandler).toHaveBeenCalledOnce()

    // wrap_tool_call
    const toolReq = mockToolCallRequest({
      toolCall: { name: 'tool_a', args: {}, id: 'call-1' },
      state: { taskStatuses: { a: 'in_progress' } },
    })
    const toolHandler: ToolCallHandler = vi.fn(() => ({ content: 'ok', toolCallId: 'call-1' }))
    const result = mw.wrapToolCall(toolReq, toolHandler)
    expect(toolHandler).toHaveBeenCalledOnce()
    expect((result as ToolMessageResult).content).toBe('ok')
  })
})
