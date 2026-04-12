/**
 * Tests for workflow mode (dynamic pipeline activation/deactivation).
 */
import { describe, it, expect } from 'vitest'
import {
  WorkflowSteeringMiddleware,
  TaskSteeringMiddleware,
  TaskStatus,
  TaskMiddleware,
  getContentBlocks,
  _ACTIVATE_TOOL_NAME,
  _DEACTIVATE_TOOL_NAME,
  _TRANSITION_TOOL_NAME,
  _getAllowedToolNames,
  _renderStatusBlock,
  type Workflow,
  type Task,
  type ModelRequest,
  type ToolCallRequest,
  type ToolLike,
  type ToolMessageResult,
  type CommandResult,
  type SystemMessageLike,
  type ModelCallHandler,
  type ToolCallHandler,
} from '../src/index.js'

// ── Mock tools ──────────────────────────────────────────

const toolA: ToolLike = { name: 'tool_a', description: 'Tool A' }
const toolB: ToolLike = { name: 'tool_b', description: 'Tool B' }
const toolC: ToolLike = { name: 'tool_c', description: 'Tool C' }
const globalRead: ToolLike = { name: 'global_read', description: 'Global read tool' }

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

// ── Fixtures ─────────────────────────────────────────────

function makeOnboardingTasks(): Task[] {
  return [
    { name: 'collect_info', instruction: 'Collect user info.', tools: [toolA] },
    { name: 'verify', instruction: 'Verify identity.', tools: [toolB] },
  ]
}

function makeSupportTasks(): Task[] {
  return [{ name: 'diagnose', instruction: 'Diagnose the issue.', tools: [toolC] }]
}

function makeTwoWorkflows(): Workflow[] {
  return [
    {
      name: 'onboarding',
      description: 'Onboard a new user',
      tasks: makeOnboardingTasks(),
      globalTools: [globalRead],
    },
    {
      name: 'support',
      description: 'Handle a support request',
      tasks: makeSupportTasks(),
    },
  ]
}

function makeWfMiddleware(): WorkflowSteeringMiddleware {
  return new WorkflowSteeringMiddleware({ workflows: makeTwoWorkflows() })
}

function makeThreeTasks(): Task[] {
  return [
    { name: 'step_1', instruction: 'Step one.', tools: [toolA] },
    { name: 'step_2', instruction: 'Step two.', tools: [toolB] },
    { name: 'step_3', instruction: 'Step three.', tools: [toolC] },
  ]
}

// ── Helper to check result types ────────────────────────

function isCommand(result: ToolMessageResult | CommandResult): result is CommandResult {
  return 'update' in result
}

function isToolMessage(result: ToolMessageResult | CommandResult): result is ToolMessageResult {
  return 'content' in result && !('update' in result)
}

// ════════════════════════════════════════════════════════════
// Init validation
// ════════════════════════════════════════════════════════════

describe('WorkflowInit', () => {
  it('requires at least one workflow', () => {
    expect(() => new WorkflowSteeringMiddleware({ workflows: [] })).toThrow('At least one Workflow')
  })

  it('rejects duplicate workflow names', () => {
    const wf: Workflow = {
      name: 'dup',
      description: 'A',
      tasks: makeOnboardingTasks(),
    }
    expect(() => new WorkflowSteeringMiddleware({ workflows: [wf, wf] })).toThrow(
      'Duplicate workflow names'
    )
  })

  it('rejects workflow with no tasks', () => {
    const wf: Workflow = { name: 'empty', description: 'Empty', tasks: [] }
    expect(() => new WorkflowSteeringMiddleware({ workflows: [wf] })).toThrow('has no tasks')
  })

  it('rejects duplicate task names within workflow', () => {
    const tasks: Task[] = [
      { name: 'a', instruction: 'A', tools: [toolA] },
      { name: 'a', instruction: 'B', tools: [toolB] },
    ]
    const wf: Workflow = { name: 'wf', description: 'WF', tasks }
    expect(() => new WorkflowSteeringMiddleware({ workflows: [wf] })).toThrow(
      'Duplicate task names'
    )
  })

  it('rejects unknown required tasks', () => {
    const tasks: Task[] = [{ name: 'a', instruction: 'A', tools: [toolA] }]
    const wf: Workflow = {
      name: 'wf',
      description: 'WF',
      tasks,
      requiredTasks: ['nonexistent'],
    }
    expect(() => new WorkflowSteeringMiddleware({ workflows: [wf] })).toThrow(
      'Unknown required tasks'
    )
  })

  it('is WorkflowSteeringMiddleware instance', () => {
    const mw = makeWfMiddleware()
    expect(mw).toBeInstanceOf(WorkflowSteeringMiddleware)
  })

  it('task mode is separate class', () => {
    const mw = new TaskSteeringMiddleware({ tasks: makeThreeTasks() })
    expect(mw).toBeInstanceOf(TaskSteeringMiddleware)
    expect(mw).not.toBeInstanceOf(WorkflowSteeringMiddleware)
  })

  it('all tools registered', () => {
    const mw = makeWfMiddleware()
    const names = new Set(mw.tools.map((t) => t.name))
    expect(names.has(_ACTIVATE_TOOL_NAME)).toBe(true)
    expect(names.has(_DEACTIVATE_TOOL_NAME)).toBe(true)
    expect(names.has(_TRANSITION_TOOL_NAME)).toBe(true)
    expect(names.has('tool_a')).toBe(true)
    expect(names.has('tool_b')).toBe(true)
    expect(names.has('tool_c')).toBe(true)
    expect(names.has('global_read')).toBe(true)
  })

  it('tools deduplicated', () => {
    const mw = makeWfMiddleware()
    const count = mw.tools.filter((t) => t.name === 'global_read').length
    expect(count).toBe(1)
  })
})

// ════════════════════════════════════════════════════════════
// beforeAgent
// ════════════════════════════════════════════════════════════

describe('WorkflowBeforeAgent', () => {
  it('is a noop in workflow mode', () => {
    const mw = makeWfMiddleware()
    const result = mw.beforeAgent({ messages: [] })
    expect(result).toBeNull()
  })

  it('is a noop even without taskStatuses', () => {
    const mw = makeWfMiddleware()
    const result = mw.beforeAgent({ messages: [], taskStatuses: undefined })
    expect(result).toBeNull()
  })
})

// ════════════════════════════════════════════════════════════
// activate_workflow (executeActivate)
// ════════════════════════════════════════════════════════════

describe('ActivateWorkflow', () => {
  it('happy path', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeActivate(
      { workflow: 'onboarding' },
      { activeWorkflow: null, messages: [] },
      'call_123'
    )
    expect(isCommand(result)).toBe(true)
    if (isCommand(result)) {
      expect(result.update.activeWorkflow).toBe('onboarding')
      expect(result.update.taskStatuses).toEqual({
        collect_info: 'pending',
        verify: 'pending',
      })
    }
  })

  it('unknown workflow', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeActivate(
      { workflow: 'nonexistent' },
      { activeWorkflow: null },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('Unknown workflow')
    }
  })

  it('already active', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeActivate(
      { workflow: 'onboarding' },
      { activeWorkflow: 'support' },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('already active')
    }
  })

  it('initializes nudge count', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeActivate(
      { workflow: 'onboarding' },
      { activeWorkflow: null, messages: [] },
      'call_123'
    )
    if (isCommand(result)) {
      expect(result.update.nudgeCount).toBe(0)
    }
  })
})

// ════════════════════════════════════════════════════════════
// deactivate_workflow (executeDeactivate)
// ════════════════════════════════════════════════════════════

describe('DeactivateWorkflow', () => {
  it('happy path', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeDeactivate(
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'complete', verify: 'complete' },
      },
      'call_123'
    )
    expect(isCommand(result)).toBe(true)
    if (isCommand(result)) {
      expect(result.update.activeWorkflow).toBeNull()
      expect(result.update.taskStatuses).toEqual({})
    }
  })

  it('no active workflow', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeDeactivate({ activeWorkflow: null }, 'call_123')
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('No workflow')
    }
  })

  it('blocked when task in progress', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeDeactivate(
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('Cannot deactivate')
    }
  })

  it('allowed when allowDeactivateInProgress', () => {
    const wf: Workflow = {
      name: 'flex',
      description: 'Flexible workflow',
      tasks: makeOnboardingTasks(),
      allowDeactivateInProgress: true,
    }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })
    const result = mw.executeDeactivate(
      {
        activeWorkflow: 'flex',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
      'call_123'
    )
    expect(isCommand(result)).toBe(true)
    if (isCommand(result)) {
      expect(result.update.activeWorkflow).toBeNull()
    }
  })
})

// ════════════════════════════════════════════════════════════
// Workflow transition tool (executeTransition)
// ════════════════════════════════════════════════════════════

describe('WorkflowTransition', () => {
  it('no workflow active', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'in_progress' },
      { activeWorkflow: null },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('No workflow is active')
    }
  })

  it('invalid task for workflow', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'in_progress' },
      { activeWorkflow: 'support', taskStatuses: { diagnose: 'pending' } },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('Invalid task')
    }
  })

  it('happy path start', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'in_progress' },
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
      },
      'call_123'
    )
    expect(isCommand(result)).toBe(true)
    if (isCommand(result)) {
      expect((result.update.taskStatuses as Record<string, string>).collect_info).toBe(
        'in_progress'
      )
    }
  })

  it('happy path complete', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'complete' },
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
      'call_123'
    )
    expect(isCommand(result)).toBe(true)
    if (isCommand(result)) {
      expect((result.update.taskStatuses as Record<string, string>).collect_info).toBe('complete')
    }
  })

  it('enforce order', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'verify', status: 'in_progress' },
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
      },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('not complete yet')
    }
  })

  it('no enforce order', () => {
    const wf: Workflow = {
      name: 'flex',
      description: 'Flexible',
      tasks: makeOnboardingTasks(),
      enforceOrder: false,
    }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })
    const result = mw.executeTransition(
      { task: 'verify', status: 'in_progress' },
      {
        activeWorkflow: 'flex',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
      },
      'call_123'
    )
    expect(isCommand(result)).toBe(true)
    if (isCommand(result)) {
      expect((result.update.taskStatuses as Record<string, string>).verify).toBe('in_progress')
    }
  })

  it('invalid status', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'invalid' },
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
      },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('Invalid status')
    }
  })

  it('already complete', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'complete' },
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'complete', verify: 'pending' },
      },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('already complete')
    }
  })

  it('invalid transition (pending to complete)', () => {
    const mw = makeWfMiddleware()
    const result = mw.executeTransition(
      { task: 'collect_info', status: 'complete' },
      {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
      },
      'call_123'
    )
    expect(isToolMessage(result)).toBe(true)
    if (isToolMessage(result)) {
      expect(result.content).toContain('Cannot transition')
    }
  })
})

// ════════════════════════════════════════════════════════════
// Tool scoping (wrapModelCall)
// ════════════════════════════════════════════════════════════

describe('WorkflowToolScoping', () => {
  it('no workflow — transparent with catalog', () => {
    const mw = makeWfMiddleware()
    const externalTool: ToolLike = { name: 'external_search', description: 'ext' }
    const allTools = [...mw.tools, externalTool]
    const request = mockModelRequest({
      state: { activeWorkflow: null, messages: [] },
      systemMessage: { content: 'Base prompt.' },
      tools: allTools,
    })
    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (req) => {
      captured = req
      return 'ok'
    })
    expect(captured).not.toBeNull()
    const toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames.has(_ACTIVATE_TOOL_NAME)).toBe(true)
    expect(toolNames.has('external_search')).toBe(true)
    // Workflow-specific tools should NOT be present
    expect(toolNames.has(_DEACTIVATE_TOOL_NAME)).toBe(false)
    expect(toolNames.has(_TRANSITION_TOOL_NAME)).toBe(false)
  })

  it('workflow active but no task in_progress', () => {
    const mw = makeWfMiddleware()
    const request = mockModelRequest({
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
        messages: [],
      },
      systemMessage: { content: 'Base prompt.' },
      tools: mw.tools,
    })
    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (req) => {
      captured = req
      return 'ok'
    })
    const toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames.has(_TRANSITION_TOOL_NAME)).toBe(true)
    expect(toolNames.has(_DEACTIVATE_TOOL_NAME)).toBe(true)
    expect(toolNames.has('global_read')).toBe(true)
    // Task-specific tools should NOT be present
    expect(toolNames.has('tool_a')).toBe(false)
    expect(toolNames.has('tool_b')).toBe(false)
    // activate should NOT be present when a workflow is active
    expect(toolNames.has(_ACTIVATE_TOOL_NAME)).toBe(false)
  })

  it('workflow active with task in_progress', () => {
    const mw = makeWfMiddleware()
    const request = mockModelRequest({
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
        messages: [],
      },
      systemMessage: { content: 'Base prompt.' },
      tools: mw.tools,
    })
    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (req) => {
      captured = req
      return 'ok'
    })
    const toolNames = new Set(captured!.tools.map((t) => t.name))
    expect(toolNames.has(_TRANSITION_TOOL_NAME)).toBe(true)
    expect(toolNames.has(_DEACTIVATE_TOOL_NAME)).toBe(true)
    expect(toolNames.has('global_read')).toBe(true)
    expect(toolNames.has('tool_a')).toBe(true) // collect_info's tool
    expect(toolNames.has('tool_b')).toBe(false) // verify's tool
  })
})

// ════════════════════════════════════════════════════════════
// Prompt rendering
// ════════════════════════════════════════════════════════════

describe('WorkflowPromptRendering', () => {
  it('catalog view', () => {
    const mw = makeWfMiddleware()
    const block = (mw as any)._catalogText as string
    expect(block).toContain('<available_workflows>')
    expect(block).toContain('workflow name="onboarding"')
    expect(block).toContain('Onboard a new user')
    expect(block).toContain('Tasks: collect_info, verify')
    expect(block).toContain('workflow name="support"')
    expect(block).toContain('Handle a support request')
    expect(block).toContain('activate_workflow')
  })

  it('active workflow pipeline view', () => {
    const mw = makeWfMiddleware()
    const ctx = mw._workflowCtxs.get('onboarding')!
    const statuses = { collect_info: 'in_progress', verify: 'pending' }
    const block = _renderStatusBlock(ctx, statuses, 'collect_info')
    expect(block).toContain('<task_pipeline workflow="onboarding">')
    expect(block).toContain('[>] collect_info (in_progress)')
    expect(block).toContain('[ ] verify (pending)')
    expect(block).toContain('<current_task name="collect_info">')
    expect(block).toContain('Collect user info.')
  })

  it('catalog injected when no workflow active', () => {
    const mw = makeWfMiddleware()
    const request = mockModelRequest({
      state: { activeWorkflow: null, messages: [] },
      systemMessage: { content: 'Base.' },
      tools: mw.tools,
    })
    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (req) => {
      captured = req
      return 'ok'
    })
    const content = captured!.systemMessage.content as Array<{ type: string; text?: string }>
    const textBlocks = content.filter((b) => b.type === 'text').map((b) => b.text ?? '')
    const full = textBlocks.join('\n')
    expect(full).toContain('<available_workflows>')
  })

  it('pipeline injected when workflow active', () => {
    const mw = makeWfMiddleware()
    const request = mockModelRequest({
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
        messages: [],
      },
      systemMessage: { content: 'Base.' },
      tools: mw.tools,
    })
    let captured: ModelRequest | null = null
    mw.wrapModelCall(request, (req) => {
      captured = req
      return 'ok'
    })
    const content = captured!.systemMessage.content as Array<{ type: string; text?: string }>
    const textBlocks = content.filter((b) => b.type === 'text').map((b) => b.text ?? '')
    const full = textBlocks.join('\n')
    expect(full).toContain('<task_pipeline')
    expect(full).toContain('collect_info')
  })
})

// ════════════════════════════════════════════════════════════
// wrapToolCall — workflow mode
// ════════════════════════════════════════════════════════════

describe('WorkflowWrapToolCall', () => {
  it('activate passes through', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: _ACTIVATE_TOOL_NAME, id: 'c1', args: { workflow: 'onboarding' } },
      state: { activeWorkflow: null },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'handled', toolCallId: 'c1' }))
    expect((result as ToolMessageResult).content).toBe('handled')
  })

  it('deactivate passes through', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: _DEACTIVATE_TOOL_NAME, id: 'c1', args: {} },
      state: { activeWorkflow: 'onboarding', taskStatuses: {} },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'handled', toolCallId: 'c1' }))
    expect((result as ToolMessageResult).content).toBe('handled')
  })

  it('no workflow — transparent', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: 'some_random_tool', id: 'c1', args: {} },
      state: { activeWorkflow: null },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'passed', toolCallId: 'c1' }))
    expect((result as ToolMessageResult).content).toBe('passed')
  })

  it('tool gated when workflow active', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: 'tool_c', id: 'c1', args: {} },
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
    })
    const result = mw.wrapToolCall(request, () => ({
      content: 'should_not_reach',
      toolCallId: 'c1',
    }))
    expect(isToolMessage(result)).toBe(true)
    expect((result as ToolMessageResult).content).toContain('not available')
  })

  it('tool allowed for active task', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: 'tool_a', id: 'c1', args: {} },
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'passed', toolCallId: 'c1' }))
    expect((result as ToolMessageResult).content).toBe('passed')
  })

  it('global tool allowed', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: { name: 'global_read', id: 'c1', args: {} },
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'passed', toolCallId: 'c1' }))
    expect((result as ToolMessageResult).content).toBe('passed')
  })

  it('transition validates and fires hooks', () => {
    const mw = makeWfMiddleware()
    const cmd: CommandResult = {
      update: {
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
        nudgeCount: 0,
        messages: [
          { role: 'tool', content: "Task 'collect_info' -> in_progress.", toolCallId: 'c1' },
        ],
      },
    }
    const request = mockToolCallRequest({
      toolCall: {
        name: _TRANSITION_TOOL_NAME,
        id: 'c1',
        args: { task: 'collect_info', status: 'in_progress' },
      },
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'pending', verify: 'pending' },
      },
    })
    const result = mw.wrapToolCall(request, () => cmd)
    expect(isCommand(result)).toBe(true)
  })

  it('transition validation rejects double start', () => {
    const mw = makeWfMiddleware()
    const request = mockToolCallRequest({
      toolCall: {
        name: _TRANSITION_TOOL_NAME,
        id: 'c1',
        args: { task: 'verify', status: 'in_progress' },
      },
      state: {
        activeWorkflow: 'onboarding',
        taskStatuses: { collect_info: 'in_progress', verify: 'pending' },
      },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'unreachable', toolCallId: 'c1' }))
    expect(isToolMessage(result)).toBe(true)
    expect((result as ToolMessageResult).content).toContain('already in progress')
  })
})

// ════════════════════════════════════════════════════════════
// Validation via TaskMiddleware in workflow mode
// ════════════════════════════════════════════════════════════

describe('WorkflowValidation', () => {
  it('validate completion rejects', () => {
    const tasks: Task[] = [
      {
        name: 'gated',
        instruction: 'Gated task',
        tools: [toolA],
        middleware: new RejectCompletionMiddleware('Not ready.'),
      },
    ]
    const wf: Workflow = { name: 'wf', description: 'WF', tasks }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })

    const request = mockToolCallRequest({
      toolCall: {
        name: _TRANSITION_TOOL_NAME,
        id: 'c1',
        args: { task: 'gated', status: 'complete' },
      },
      state: {
        activeWorkflow: 'wf',
        taskStatuses: { gated: 'in_progress' },
      },
    })
    const result = mw.wrapToolCall(request, () => ({ content: 'unreachable', toolCallId: 'c1' }))
    expect(isToolMessage(result)).toBe(true)
    expect((result as ToolMessageResult).content).toContain('Not ready.')
  })
})

// ════════════════════════════════════════════════════════════
// afterAgent — workflow mode
// ════════════════════════════════════════════════════════════

describe('WorkflowAfterAgent', () => {
  it('no workflow — noop', () => {
    const mw = makeWfMiddleware()
    const result = mw.afterAgent({ activeWorkflow: null, messages: [] })
    expect(result).toBeNull()
  })

  it('noop when all required complete', () => {
    const mw = makeWfMiddleware()
    const result = mw.afterAgent({
      activeWorkflow: 'onboarding',
      taskStatuses: { collect_info: 'complete', verify: 'complete' },
      messages: [],
    })
    expect(result).toBeNull()
  })

  it('nudge when incomplete', () => {
    const mw = makeWfMiddleware()
    const result = mw.afterAgent({
      activeWorkflow: 'onboarding',
      taskStatuses: { collect_info: 'complete', verify: 'pending' },
      nudgeCount: 0,
      messages: [],
    })
    expect(result).not.toBeNull()
    expect(result!.jumpTo).toBe('model')
    expect(result!.nudgeCount).toBe(1)
    expect(result!.messages.length).toBeGreaterThan(0)
  })

  it('stops nudging after max', () => {
    const mw = makeWfMiddleware()
    const result = mw.afterAgent({
      activeWorkflow: 'onboarding',
      taskStatuses: { collect_info: 'complete', verify: 'pending' },
      nudgeCount: 3,
      messages: [],
    })
    expect(result).toBeNull()
  })

  it('no required tasks — no nudge', () => {
    const wf: Workflow = {
      name: 'opt',
      description: 'Optional',
      tasks: makeOnboardingTasks(),
      requiredTasks: null,
    }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })
    const result = mw.afterAgent({
      activeWorkflow: 'opt',
      taskStatuses: { collect_info: 'pending', verify: 'pending' },
      nudgeCount: 0,
      messages: [],
    })
    expect(result).toBeNull()
  })

  it('partial required tasks — noop when met', () => {
    const wf: Workflow = {
      name: 'partial',
      description: 'Partial',
      tasks: makeOnboardingTasks(),
      requiredTasks: ['collect_info'],
    }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })
    const result = mw.afterAgent({
      activeWorkflow: 'partial',
      taskStatuses: { collect_info: 'complete', verify: 'pending' },
      messages: [],
    })
    expect(result).toBeNull()
  })
})

// ════════════════════════════════════════════════════════════
// Lifecycle hooks (onStart / onComplete)
// ════════════════════════════════════════════════════════════

describe('WorkflowLifecycleHooks', () => {
  it('onStart fires', () => {
    let started = false

    class TrackingMiddleware extends TaskMiddleware {
      onStart(_state: Record<string, unknown>) {
        started = true
      }
    }

    const tasks: Task[] = [
      {
        name: 'tracked',
        instruction: 'Tracked',
        tools: [toolA],
        middleware: new TrackingMiddleware(),
      },
    ]
    const wf: Workflow = { name: 'wf', description: 'WF', tasks }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })

    const cmd: CommandResult = {
      update: {
        taskStatuses: { tracked: 'in_progress' },
        nudgeCount: 0,
        messages: [{ role: 'tool', content: 'ok', toolCallId: 'c1' }],
      },
    }
    const request = mockToolCallRequest({
      toolCall: {
        name: _TRANSITION_TOOL_NAME,
        id: 'c1',
        args: { task: 'tracked', status: 'in_progress' },
      },
      state: {
        activeWorkflow: 'wf',
        taskStatuses: { tracked: 'pending' },
        messages: [],
      },
    })
    mw.wrapToolCall(request, () => cmd)
    expect(started).toBe(true)
  })

  it('onComplete fires', () => {
    let completed = false

    class TrackingMiddleware extends TaskMiddleware {
      onComplete(_state: Record<string, unknown>) {
        completed = true
      }
    }

    const tasks: Task[] = [
      {
        name: 'tracked',
        instruction: 'Tracked',
        tools: [toolA],
        middleware: new TrackingMiddleware(),
      },
    ]
    const wf: Workflow = { name: 'wf', description: 'WF', tasks }
    const mw = new WorkflowSteeringMiddleware({ workflows: [wf] })

    const cmd: CommandResult = {
      update: {
        taskStatuses: { tracked: 'complete' },
        nudgeCount: 0,
        messages: [{ role: 'tool', content: 'ok', toolCallId: 'c1' }],
      },
    }
    const request = mockToolCallRequest({
      toolCall: {
        name: _TRANSITION_TOOL_NAME,
        id: 'c1',
        args: { task: 'tracked', status: 'complete' },
      },
      state: {
        activeWorkflow: 'wf',
        taskStatuses: { tracked: 'in_progress' },
        messages: [],
      },
    })
    mw.wrapToolCall(request, () => cmd)
    expect(completed).toBe(true)
  })
})

// ════════════════════════════════════════════════════════════
// Backend tools passthrough in workflow mode
// ════════════════════════════════════════════════════════════

describe('WorkflowBackendPassthrough', () => {
  it('backend tools available when enabled', () => {
    const wf: Workflow = {
      name: 'wf',
      description: 'WF',
      tasks: makeOnboardingTasks(),
    }
    const mw = new WorkflowSteeringMiddleware({
      workflows: [wf],
      backendToolsPassthrough: true,
    })
    const ctx = mw._workflowCtxs.get('wf')!
    const allowed = _getAllowedToolNames(
      ctx,
      'collect_info',
      new Set([_DEACTIVATE_TOOL_NAME]),
      true,
      WorkflowSteeringMiddleware.DEFAULT_BACKEND_TOOLS
    )
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
  })

  it('backend tools not available by default', () => {
    const mw = makeWfMiddleware()
    const ctx = mw._workflowCtxs.get('onboarding')!
    const allowed = _getAllowedToolNames(
      ctx,
      'collect_info',
      new Set([_DEACTIVATE_TOOL_NAME]),
      false,
      WorkflowSteeringMiddleware.DEFAULT_BACKEND_TOOLS
    )
    expect(allowed.has('read_file')).toBe(false)
  })
})

// ════════════════════════════════════════════════════════════
// End-to-end scenario
// ════════════════════════════════════════════════════════════

describe('WorkflowEndToEnd', () => {
  it('full lifecycle: activate -> complete tasks -> deactivate -> activate another', () => {
    const mw = makeWfMiddleware()

    // 1. Start with no workflow active
    const state: Record<string, unknown> = {
      activeWorkflow: null,
      messages: [],
      nudgeCount: 0,
    }
    const initResult = mw.beforeAgent(state as any)
    expect(initResult).toBeNull() // No init in workflow mode

    // 2. Activate onboarding
    let result = mw.executeActivate({ workflow: 'onboarding' }, state, 'c1')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)
    expect(state.activeWorkflow).toBe('onboarding')

    // 3. Start collect_info
    result = mw.executeTransition({ task: 'collect_info', status: 'in_progress' }, state, 'c2')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)

    // 4. Complete collect_info
    result = mw.executeTransition({ task: 'collect_info', status: 'complete' }, state, 'c3')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)

    // 5. Start and complete verify
    result = mw.executeTransition({ task: 'verify', status: 'in_progress' }, state, 'c4')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)
    result = mw.executeTransition({ task: 'verify', status: 'complete' }, state, 'c5')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)

    // 6. afterAgent should NOT auto-deactivate
    const afterResult = mw.afterAgent(state as any)
    expect(afterResult).toBeNull()

    // 7. Explicitly deactivate
    result = mw.executeDeactivate(state, 'c6')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)
    expect(state.activeWorkflow).toBeNull()

    // 8. Now activate support
    result = mw.executeActivate({ workflow: 'support' }, state, 'c7')
    expect(isCommand(result)).toBe(true)
    Object.assign(state, (result as CommandResult).update)
    expect(state.activeWorkflow).toBe('support')
    expect(state.taskStatuses).toEqual({ diagnose: 'pending' })
  })
})

// ════════════════════════════════════════════════════════════
// Backward compatibility
// ════════════════════════════════════════════════════════════

describe('BackwardCompatibility', () => {
  it('task mode still works', () => {
    const mw = new TaskSteeringMiddleware({ tasks: makeThreeTasks() })
    const result = mw.beforeAgent({ messages: [] })
    expect(result).not.toBeNull()
    expect(result!.taskStatuses).toBeDefined()
    expect(new Set(Object.keys(result!.taskStatuses as Record<string, string>))).toEqual(
      new Set(['step_1', 'step_2', 'step_3'])
    )
  })

  it('task mode tools registered', () => {
    const mw = new TaskSteeringMiddleware({ tasks: makeThreeTasks() })
    const names = new Set(mw.tools.map((t) => t.name))
    expect(names.has(_TRANSITION_TOOL_NAME)).toBe(true)
    expect(names.has(_ACTIVATE_TOOL_NAME)).toBe(false)
    expect(names.has(_DEACTIVATE_TOOL_NAME)).toBe(false)
  })
})
