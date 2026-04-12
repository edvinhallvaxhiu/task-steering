import { describe, it, expect, vi } from 'vitest'
import {
  TaskSteeringMiddleware,
  _getAllowedToolNames,
  type Task,
  type ToolLike,
  type ModelRequest,
  type SystemMessageLike,
} from '../src/index.js'

function mwAllowedToolNames(
  mw: TaskSteeringMiddleware,
  activeName: string | null,
  state?: Record<string, unknown>
): Set<string> {
  return _getAllowedToolNames(
    mw._ctx,
    activeName,
    new Set(),
    (mw as any)._backendToolsPassthrough as boolean,
    (mw as any)._backendTools as ReadonlySet<string>,
    state
  )
}

// ── Mock tools ──────────────────────────────────────────────

const toolA: ToolLike = { name: 'tool_a', description: 'Tool A' }
const toolB: ToolLike = { name: 'tool_b', description: 'Tool B' }
const globalRead: ToolLike = { name: 'global_read', description: 'Global read' }

function makeMockTool(name: string): ToolLike {
  return { name, description: `Mock ${name}` }
}

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

const mockBackend = {} // minimal backend object

// ════════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════════

describe('Backend tools init', () => {
  it('default passthrough is disabled', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
    })
    expect((mw as any)._backendToolsPassthrough).toBe(false)
  })

  it('DEFAULT_BACKEND_TOOLS has expected entries', () => {
    const expected = new Set([
      'ls',
      'read_file',
      'write_file',
      'edit_file',
      'glob',
      'grep',
      'execute',
      'write_todos',
      'task',
      'start_async_task',
      'check_async_task',
      'update_async_task',
      'cancel_async_task',
      'list_async_tasks',
    ])
    expect(TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS).toEqual(expected)
    expect(TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS.size).toBe(14)
  })

  it('custom backendTools override defaults', () => {
    const custom = new Set(['my_tool', 'other_tool'])
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
      backend: mockBackend,
      backendToolsPassthrough: true,
      backendTools: custom,
    })
    expect(mw.getBackendTools()).toEqual(custom)
  })

  it('passthrough works without backend', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
      backendToolsPassthrough: true,
    })
    expect((mw as any)._backendToolsPassthrough).toBe(true)
    const allowed = mwAllowedToolNames(mw, 'a')
    expect(allowed.has('read_file')).toBe(true)
  })

  it('passthrough with backend', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
      backend: mockBackend,
      backendToolsPassthrough: true,
    })
    expect((mw as any)._backendToolsPassthrough).toBe(true)
  })
})

// ════════════════════════════════════════════════════════════
// Scoping
// ════════════════════════════════════════════════════════════

describe('Backend tools scoping', () => {
  function makeMiddleware(
    passthrough = true,
    backend: unknown = mockBackend,
    backendTools?: ReadonlySet<string>
  ) {
    return new TaskSteeringMiddleware({
      tasks: [
        { name: 'step_1', instruction: 'Do 1.', tools: [toolA] },
        { name: 'step_2', instruction: 'Do 2.', tools: [toolB] },
      ],
      globalTools: [globalRead],
      backend,
      backendToolsPassthrough: passthrough,
      backendTools: backendTools ?? null,
    })
  }

  it('passthrough adds backend tools to allowed', () => {
    const mw = makeMiddleware(true)
    const allowed = mwAllowedToolNames(mw, 'step_1')
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('write_file')).toBe(true)
    expect(allowed.has('execute')).toBe(true)
    expect(allowed.has('tool_a')).toBe(true)
    expect(allowed.has('global_read')).toBe(true)
  })

  it('passthrough disabled does not add backend tools', () => {
    const mw = makeMiddleware(false)
    const allowed = mwAllowedToolNames(mw, 'step_1')
    expect(allowed.has('read_file')).toBe(false)
    expect(allowed.has('write_file')).toBe(false)
  })

  it('passthrough combines with task tools', () => {
    const mw = makeMiddleware(true)
    const allowed = mwAllowedToolNames(mw, 'step_2')
    expect(allowed.has('tool_b')).toBe(true)
    expect(allowed.has('tool_a')).toBe(false) // belongs to step_1
    expect(allowed.has('ls')).toBe(true)
    expect(allowed.has('glob')).toBe(true)
  })

  it('passthrough works with no active task', () => {
    const mw = makeMiddleware(true)
    const allowed = mwAllowedToolNames(mw, null)
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
    expect(allowed.has('tool_a')).toBe(false)
    expect(allowed.has('tool_b')).toBe(false)
  })

  it('custom backendTools used instead of defaults', () => {
    const mw = makeMiddleware(true, mockBackend, new Set(['custom_tool']))
    const allowed = mwAllowedToolNames(mw, 'step_1')
    expect(allowed.has('custom_tool')).toBe(true)
    expect(allowed.has('read_file')).toBe(false) // not in custom set
  })

  it('wrapModelCall filters tools correctly with passthrough', () => {
    const mw = makeMiddleware(true)
    const backendTools = [makeMockTool('read_file'), makeMockTool('ls'), makeMockTool('write_file')]
    const state = {
      messages: [],
      taskStatuses: { step_1: 'in_progress', step_2: 'pending' },
    }
    const req = mockModelRequest({
      state,
      systemMessage: { content: 'System' },
      tools: [toolA, toolB, globalRead, ...backendTools],
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(req, (r) => {
      captured = r
      return {}
    })

    const scopedNames = new Set(captured!.tools.map((t) => t.name))
    expect(scopedNames.has('tool_a')).toBe(true)
    expect(scopedNames.has('global_read')).toBe(true)
    expect(scopedNames.has('read_file')).toBe(true)
    expect(scopedNames.has('ls')).toBe(true)
    expect(scopedNames.has('write_file')).toBe(true)
    expect(scopedNames.has('tool_b')).toBe(false) // step_2 tools filtered
  })
})

// ════════════════════════════════════════════════════════════
// getBackendTools
// ════════════════════════════════════════════════════════════

describe('getBackendTools', () => {
  it('returns defaults when no override', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
    })
    expect(mw.getBackendTools()).toEqual(TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS)
  })

  it('returns custom when overridden', () => {
    const custom = new Set(['x', 'y'])
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
      backend: mockBackend,
      backendTools: custom,
    })
    expect(mw.getBackendTools()).toEqual(custom)
  })
})
