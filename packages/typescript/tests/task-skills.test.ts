import { describe, it, expect, vi } from 'vitest'
import {
  TaskSteeringMiddleware,
  type Task,
  type ToolLike,
  type ModelRequest,
  type SystemMessageLike,
  type SkillMetadata,
} from '../src/index.js'

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

// ── Mock backend helpers ───────────────────────────────────

interface MockDownloadResponse {
  content?: string | Uint8Array | null
  error?: string | null
}

function makeBackendWithSkills(...skillDefs: Array<[string, string]>) {
  const entries: Array<{ path: string; is_dir: boolean }> = []
  const downloads: Record<string, MockDownloadResponse> = {}

  for (const [name, desc] of skillDefs) {
    const dirPath = `/skills/${name}`
    const mdPath = `${dirPath}/SKILL.md`
    entries.push({ path: dirPath, is_dir: true })
    downloads[mdPath] = {
      content: `---\nname: ${name}\ndescription: ${desc}\n---\n`,
    }
  }

  return {
    ls(path: string) {
      if (path === '/skills/') return { entries }
      return { entries: [] }
    },
    downloadFiles(paths: string[]) {
      return paths.map((p) => downloads[p] ?? { error: 'not found' })
    },
  }
}

const mockBackend = {} // minimal backend (no ls/downloadFiles needed for init-only tests)

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

describe('Skills init', () => {
  it('task skills activates without backend', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    expect((mw as any)._skillsActive).toBe(true)
  })

  it('no skills configured is inert', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
    })
    expect((mw as any)._skillsActive).toBe(false)
  })

  it('globalSkills activates without backend', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
      globalSkills: ['gs'],
    })
    expect((mw as any)._skillsActive).toBe(true)
  })

  it('task skills field defaults to undefined', () => {
    const task: Task = { name: 'a', instruction: 'A', tools: [toolA] }
    expect(task.skills).toBeUndefined()
  })
})

// ════════════════════════════════════════════════════════════
// Skill Loading (beforeAgent)
// ════════════════════════════════════════════════════════════

describe('Skill state handling', () => {
  it('uses skillsMetadata from state (loaded by SkillsMiddleware)', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    const state = {
      messages: [],
      taskStatuses: { a: 'pending' },
      skillsMetadata: [{ name: 's1', description: 'Already loaded', path: '/x' }],
    }
    const result = mw.beforeAgent(state)
    expect(result).toBeNull()
  })

  it('beforeAgent only inits taskStatuses when no skills in state', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    const result = mw.beforeAgent({ messages: [] })
    expect(result).not.toBeNull()
    expect(result!.taskStatuses).toBeDefined()
    expect(result!.skillsMetadata).toBeUndefined()
  })
})

// ════════════════════════════════════════════════════════════
// Skill Scoping
// ════════════════════════════════════════════════════════════

describe('Skill scoping', () => {
  it('no active task returns only global skills', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
      globalSkills: ['gs'],
    })
    const allowed = (mw as any)._allowedSkillNames(null) as Set<string>
    expect(allowed).toEqual(new Set(['gs']))
  })

  it('active task returns task skills + global', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1', 's2'] }],
      globalSkills: ['gs'],
    })
    const allowed = (mw as any)._allowedSkillNames('a') as Set<string>
    expect(allowed).toEqual(new Set(['gs', 's1', 's2']))
  })

  it('task without skills returns only global', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [
        { name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] },
        { name: 'b', instruction: 'B', tools: [toolB] },
      ],
      globalSkills: ['gs'],
    })
    const allowed = (mw as any)._allowedSkillNames('b') as Set<string>
    expect(allowed).toEqual(new Set(['gs']))
  })
})

// ════════════════════════════════════════════════════════════
// Skill Rendering
// ════════════════════════════════════════════════════════════

describe('Skill rendering', () => {
  function makeMiddleware() {
    return new TaskSteeringMiddleware({
      tasks: [
        { name: 'research', instruction: 'Do research.', tools: [toolA], skills: ['web-research'] },
        { name: 'write', instruction: 'Write report.', tools: [toolB] },
      ],
      globalSkills: ['formatting'],
    })
  }

  function stateWithSkills() {
    return {
      messages: [],
      taskStatuses: { research: 'in_progress', write: 'pending' },
      skillsMetadata: [
        {
          name: 'web-research',
          description: 'Search the web.',
          path: '/skills/web-research/SKILL.md',
        },
        {
          name: 'formatting',
          description: 'Format documents.',
          path: '/skills/formatting/SKILL.md',
        },
        { name: 'other', description: 'Other skill.', path: '/skills/other/SKILL.md' },
      ],
    }
  }

  it('renders available_skills section', () => {
    const mw = makeMiddleware()
    const state = stateWithSkills()
    const statuses = (mw as any)._getStatuses(state) as Record<string, string>
    const block = mw._renderStatusBlock(statuses, 'research', state)
    expect(block).toContain('<available_skills>')
    expect(block).toContain('web-research')
    expect(block).toContain('formatting')
    // "other" skill should not appear in available_skills
    // (the word "other" appears in skill_usage boilerplate, so check specifically)
    expect(block).not.toContain('other:')
    expect(block).not.toContain('/skills/other/')
  })

  it('no skills section when feature inactive', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
    })
    const block = mw._renderStatusBlock({ a: 'in_progress' }, 'a', {})
    expect(block).not.toContain('<available_skills>')
  })

  it('skills filtered to active task', () => {
    const mw = makeMiddleware()
    const state = stateWithSkills()
    state.taskStatuses = { research: 'complete', write: 'in_progress' }
    const statuses = (mw as any)._getStatuses(state) as Record<string, string>
    const block = mw._renderStatusBlock(statuses, 'write', state)
    expect(block).toContain('<available_skills>')
    expect(block).toContain('formatting')
    expect(block).not.toContain('web-research')
  })

  it('skill instruction appears in rules', () => {
    const mw = makeMiddleware()
    const state = stateWithSkills()
    const statuses = (mw as any)._getStatuses(state) as Record<string, string>
    const block = mw._renderStatusBlock(statuses, 'research', state)
    expect(block).toContain('read its SKILL.md file')
  })

  it('no skills section without state', () => {
    const mw = makeMiddleware()
    const block = mw._renderStatusBlock({ research: 'in_progress', write: 'pending' }, 'research')
    expect(block).not.toContain('<available_skills>')
  })

  it('warns on missing skill names', () => {
    const mw = makeMiddleware()
    const state = {
      messages: [],
      taskStatuses: { research: 'in_progress', write: 'pending' },
      skillsMetadata: [
        // "web-research" and "formatting" are referenced but only "formatting" exists
        {
          name: 'formatting',
          description: 'Format documents.',
          path: '/skills/formatting/SKILL.md',
        },
      ],
    }
    const statuses = (mw as any)._getStatuses(state) as Record<string, string>
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    mw._renderStatusBlock(statuses, 'research', state)
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('web-research'))
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('not found in skillsMetadata'))
    warnSpy.mockRestore()
  })

  it('no warning when all skills present', () => {
    const mw = makeMiddleware()
    const state = stateWithSkills()
    const statuses = (mw as any)._getStatuses(state) as Record<string, string>
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    mw._renderStatusBlock(statuses, 'research', state)
    expect(warnSpy).not.toHaveBeenCalled()
    warnSpy.mockRestore()
  })
})

// ════════════════════════════════════════════════════════════
// Tool Auto-Whitelist
// ════════════════════════════════════════════════════════════

describe('Skill tool auto-whitelist', () => {
  it('read_file and ls auto-whitelisted when skills active', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    const allowed = mw._allowedToolNames('a')
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
  })

  it('no auto-whitelist when no skills configured', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
    })
    const allowed = mw._allowedToolNames('a')
    expect(allowed.has('read_file')).toBe(false)
    expect(allowed.has('ls')).toBe(false)
  })

  it('no auto-whitelist for task without skills', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [
        { name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] },
        { name: 'b', instruction: 'B', tools: [toolB] },
      ],
    })
    const allowed = mw._allowedToolNames('b')
    expect(allowed.has('read_file')).toBe(false)
    expect(allowed.has('ls')).toBe(false)
  })

  it('auto-whitelist via global skills for task without own skills', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [
        { name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] },
        { name: 'b', instruction: 'B', tools: [toolB] },
      ],
      globalSkills: ['gs'],
    })
    const allowed = mw._allowedToolNames('b')
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
  })

  it('auto-whitelist independent of passthrough', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
      backendToolsPassthrough: false,
    })
    const allowed = mw._allowedToolNames('a')
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
    expect(allowed.has('write_file')).toBe(false)
    expect(allowed.has('execute')).toBe(false)
  })

  it('passthrough and skills combined', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
      backendToolsPassthrough: true,
    })
    const allowed = mw._allowedToolNames('a')
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
    expect(allowed.has('write_file')).toBe(true)
    expect(allowed.has('execute')).toBe(true)
  })

  it('skill allowedTools whitelisted when state provided', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    const state = {
      skillsMetadata: [
        {
          name: 's1',
          description: 'Skill 1',
          path: '/skills/s1/SKILL.md',
          allowedTools: ['web_search', 'scrape_url'],
        },
      ],
    }
    const allowed = mw._allowedToolNames('a', state)
    expect(allowed.has('web_search')).toBe(true)
    expect(allowed.has('scrape_url')).toBe(true)
  })

  it('skill allowedTools scoped to active task', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [
        { name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] },
        { name: 'b', instruction: 'B', tools: [toolB], skills: ['s2'] },
      ],
    })
    const state = {
      skillsMetadata: [
        {
          name: 's1',
          description: 'Skill 1',
          path: '/skills/s1/SKILL.md',
          allowedTools: ['web_search'],
        },
        {
          name: 's2',
          description: 'Skill 2',
          path: '/skills/s2/SKILL.md',
          allowedTools: ['code_exec'],
        },
      ],
    }
    const allowedA = mw._allowedToolNames('a', state)
    expect(allowedA.has('web_search')).toBe(true)
    expect(allowedA.has('code_exec')).toBe(false)

    const allowedB = mw._allowedToolNames('b', state)
    expect(allowedB.has('code_exec')).toBe(true)
    expect(allowedB.has('web_search')).toBe(false)
  })

  it('skill allowedTools includes global skills', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA] }],
      globalSkills: ['gs'],
    })
    const state = {
      skillsMetadata: [
        {
          name: 'gs',
          description: 'Global',
          path: '/skills/gs/SKILL.md',
          allowedTools: ['format_doc'],
        },
      ],
    }
    const allowed = mw._allowedToolNames('a', state)
    expect(allowed.has('format_doc')).toBe(true)
  })

  it('skill allowedTools without state is noop', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    const allowed = mw._allowedToolNames('a')
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('web_search')).toBe(false)
  })

  it('skill without allowedTools field does not break', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'A', tools: [toolA], skills: ['s1'] }],
    })
    const state = {
      skillsMetadata: [{ name: 's1', description: 'Skill 1', path: '/skills/s1/SKILL.md' }],
    }
    const allowed = mw._allowedToolNames('a', state)
    expect(allowed.has('read_file')).toBe(true)
    expect(allowed.has('ls')).toBe(true)
  })
})

// ════════════════════════════════════════════════════════════
// End-to-End
// ════════════════════════════════════════════════════════════

describe('Skills end-to-end', () => {
  it('skills from state scoped per task in wrapModelCall', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [
        {
          name: 'research',
          instruction: 'Do research.',
          tools: [toolA],
          skills: ['research-skill'],
        },
        { name: 'write', instruction: 'Write report.', tools: [toolB], skills: ['writing-skill'] },
      ],
      globalTools: [globalRead],
      globalSkills: ['global-skill'],
    })

    // Simulate state with skills already loaded (by SkillsMiddleware)
    const state: Record<string, unknown> = {
      messages: [],
      taskStatuses: { research: 'in_progress', write: 'pending' },
      skillsMetadata: [
        {
          name: 'research-skill',
          description: 'Research things',
          path: '/skills/research-skill/SKILL.md',
        },
        {
          name: 'writing-skill',
          description: 'Write things',
          path: '/skills/writing-skill/SKILL.md',
        },
        {
          name: 'global-skill',
          description: 'Always available',
          path: '/skills/global-skill/SKILL.md',
        },
      ],
    }

    const readFileTool = makeMockTool('read_file')
    const lsTool = makeMockTool('ls')
    const writeFileTool = makeMockTool('write_file')

    const req = mockModelRequest({
      state,
      systemMessage: { content: 'System' },
      tools: [toolA, toolB, globalRead, readFileTool, lsTool, writeFileTool],
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(req, (r) => {
      captured = r
      return {}
    })

    const scopedNames = new Set(captured!.tools.map((t) => t.name))
    // Research task tools + global + auto-whitelisted
    expect(scopedNames.has('tool_a')).toBe(true)
    expect(scopedNames.has('global_read')).toBe(true)
    expect(scopedNames.has('read_file')).toBe(true)
    expect(scopedNames.has('ls')).toBe(true)
    // Not in scope
    expect(scopedNames.has('tool_b')).toBe(false)
    expect(scopedNames.has('write_file')).toBe(false)

    // Check skills rendered in prompt
    const promptText = extractText(captured!)
    expect(promptText).toContain('research-skill')
    expect(promptText).toContain('global-skill')
    expect(promptText).not.toContain('writing-skill')
  })

  it('strips SkillsMiddleware prompt injection', () => {
    const mw = new TaskSteeringMiddleware({
      tasks: [{ name: 'a', instruction: 'Do A.', tools: [toolA], skills: ['s1'] }],
      globalSkills: ['gs'],
    })

    const state: Record<string, unknown> = {
      messages: [],
      taskStatuses: { a: 'in_progress' },
      skillsMetadata: [
        { name: 's1', description: 'Skill 1', path: '/skills/s1/SKILL.md' },
        { name: 'gs', description: 'Global', path: '/skills/gs/SKILL.md' },
      ],
    }

    // Simulate SkillsMiddleware having already injected its prompt
    const skillsMiddlewareBlock =
      '## Skills System\n\nYou have access to a skills library...\n\n- s1: Skill 1\n- gs: Global'

    const req = mockModelRequest({
      state,
      systemMessage: {
        content: [
          { type: 'text', text: 'Base system prompt.' },
          { type: 'text', text: skillsMiddlewareBlock },
        ],
      },
      tools: mw.tools,
    })

    let captured: ModelRequest | null = null
    mw.wrapModelCall(req, (r) => {
      captured = r
      return {}
    })

    const promptText = extractText(captured!)
    // SkillsMiddleware's section stripped
    expect(promptText).not.toContain('## Skills System')
    // Base prompt preserved
    expect(promptText).toContain('Base system prompt.')
    // Our scoped skills rendered
    expect(promptText).toContain('<available_skills>')
    expect(promptText).toContain('s1')
  })
})
