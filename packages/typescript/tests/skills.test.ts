import { describe, it, expect, vi } from 'vitest'
import { parseSkillFrontmatter, loadSkillsFromBackend } from '../src/skills.js'

// ── parseSkillFrontmatter ──────────────────────────────────

const VALID_SKILL_MD = `\
---
name: web-research
description: Search the web for information.
license: MIT
compatibility: python>=3.10
allowed-tools: search browse
metadata:
  category: research
---

# Web Research

Use this skill to search the web.
`

describe('parseSkillFrontmatter', () => {
  it('parses valid frontmatter with all fields', () => {
    const result = parseSkillFrontmatter(VALID_SKILL_MD, '/skills/web-research/SKILL.md')
    expect(result).not.toBeNull()
    expect(result!.name).toBe('web-research')
    expect(result!.description).toBe('Search the web for information.')
    expect(result!.path).toBe('/skills/web-research/SKILL.md')
    expect(result!.license).toBe('MIT')
    expect(result!.compatibility).toBe('python>=3.10')
    expect(result!.allowedTools).toEqual(['search', 'browse'])
    expect(result!.metadata).toEqual({ category: 'research' })
  })

  it('parses minimal frontmatter', () => {
    const content = '---\nname: minimal\ndescription: A minimal skill.\n---\n\nBody.'
    const result = parseSkillFrontmatter(content, '/skills/minimal/SKILL.md')
    expect(result).not.toBeNull()
    expect(result!.name).toBe('minimal')
    expect(result!.description).toBe('A minimal skill.')
    expect(result!.license).toBeUndefined()
    expect(result!.allowedTools).toBeUndefined()
  })

  it('returns null for missing frontmatter', () => {
    const result = parseSkillFrontmatter('No frontmatter here.', '/bad/SKILL.md')
    expect(result).toBeNull()
  })

  it('returns null for missing name', () => {
    const content = '---\ndescription: No name.\n---\n'
    const result = parseSkillFrontmatter(content, '/bad/SKILL.md')
    expect(result).toBeNull()
  })

  it('returns null for missing description', () => {
    const content = '---\nname: no-desc\n---\n'
    const result = parseSkillFrontmatter(content, '/bad/SKILL.md')
    expect(result).toBeNull()
  })

  it('returns null for non-mapping frontmatter', () => {
    const content = '---\n- list\n- not dict\n---\n'
    const result = parseSkillFrontmatter(content, '/bad/SKILL.md')
    expect(result).toBeNull()
  })

  it('returns null for oversized content', () => {
    const content = '---\nname: big\ndescription: Big.\n---\n' + 'x'.repeat(11 * 1024 * 1024)
    const result = parseSkillFrontmatter(content, '/big/SKILL.md')
    expect(result).toBeNull()
  })

  it('handles allowed-tools as YAML list', () => {
    const content = '---\nname: t\ndescription: d\nallowed-tools:\n  - a\n  - b\n---\n'
    const result = parseSkillFrontmatter(content, '/t/SKILL.md')
    expect(result).not.toBeNull()
    expect(result!.allowedTools).toEqual(['a', 'b'])
  })
})

// ── Mock backend helpers ───────────────────────────────────

interface MockDownloadResponse {
  content?: string | Uint8Array | null
  error?: string | null
}

function makeMockBackend(
  lsResults: Record<string, Array<{ path: string; is_dir: boolean }>>,
  downloadResponses: Record<string, MockDownloadResponse>
) {
  return {
    ls(path: string) {
      return { entries: lsResults[path] ?? [] }
    },
    downloadFiles(paths: string[]) {
      return paths.map((p) => downloadResponses[p] ?? { error: 'not found' })
    },
  }
}

function makeBackendWithSkills(...skillDefs: Array<[string, string]>) {
  const entries: Array<{ path: string; is_dir: boolean }> = []
  const downloads: Record<string, MockDownloadResponse> = {}

  for (const [name, desc] of skillDefs) {
    const dirPath = `/skills/${name}`
    const mdPath = `${dirPath}/SKILL.md`
    entries.push({ path: dirPath, is_dir: true })
    const content = new TextEncoder().encode(`---\nname: ${name}\ndescription: ${desc}\n---\n`)
    downloads[mdPath] = { content }
  }

  return makeMockBackend({ '/skills/': entries }, downloads)
}

// ── loadSkillsFromBackend ──────────────────────────────────

describe('loadSkillsFromBackend', () => {
  it('loads from single source', () => {
    const backend = makeMockBackend(
      {
        '/skills/': [
          { path: '/skills/research', is_dir: true },
          { path: '/skills/writing', is_dir: true },
        ],
      },
      {
        '/skills/research/SKILL.md': {
          content: '---\nname: research\ndescription: Research skill.\n---\n',
        },
        '/skills/writing/SKILL.md': {
          content: '---\nname: writing\ndescription: Writing skill.\n---\n',
        },
      }
    )

    const result = loadSkillsFromBackend(backend, ['/skills/'])
    expect(result).toHaveLength(2)
    const names = new Set(result.map((s) => s.name))
    expect(names).toEqual(new Set(['research', 'writing']))
  })

  it('loads from multiple sources', () => {
    const backend = makeMockBackend(
      {
        '/a/': [{ path: '/a/s1', is_dir: true }],
        '/b/': [{ path: '/b/s2', is_dir: true }],
      },
      {
        '/a/s1/SKILL.md': { content: '---\nname: s1\ndescription: Skill 1.\n---\n' },
        '/b/s2/SKILL.md': { content: '---\nname: s2\ndescription: Skill 2.\n---\n' },
      }
    )
    const result = loadSkillsFromBackend(backend, ['/a/', '/b/'])
    expect(result).toHaveLength(2)
  })

  it('skips non-directory entries', () => {
    const backend = makeMockBackend(
      {
        '/skills/': [
          { path: '/skills/readme.md', is_dir: false },
          { path: '/skills/research', is_dir: true },
        ],
      },
      {
        '/skills/research/SKILL.md': {
          content: '---\nname: research\ndescription: R.\n---\n',
        },
      }
    )
    const result = loadSkillsFromBackend(backend, ['/skills/'])
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('research')
  })

  it('handles download error', () => {
    const backend = makeMockBackend(
      {
        '/skills/': [
          { path: '/skills/good', is_dir: true },
          { path: '/skills/bad', is_dir: true },
        ],
      },
      {
        '/skills/good/SKILL.md': {
          content: '---\nname: good\ndescription: Good.\n---\n',
        },
        '/skills/bad/SKILL.md': { error: 'not found' },
      }
    )
    const result = loadSkillsFromBackend(backend, ['/skills/'])
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('good')
  })

  it('handles empty source dir', () => {
    const backend = makeMockBackend({ '/empty/': [] }, {})
    const result = loadSkillsFromBackend(backend, ['/empty/'])
    expect(result).toEqual([])
  })

  it('handles ls failure gracefully', () => {
    const backend = {
      ls(_path: string) {
        throw new Error('connection failed')
      },
      downloadFiles(_paths: string[]) {
        return []
      },
    }
    const result = loadSkillsFromBackend(backend, ['/skills/'])
    expect(result).toEqual([])
  })

  it('last source wins on name conflict', () => {
    const backend = makeMockBackend(
      {
        '/a/': [{ path: '/a/x', is_dir: true }],
        '/b/': [{ path: '/b/x', is_dir: true }],
      },
      {
        '/a/x/SKILL.md': { content: '---\nname: x\ndescription: From A.\n---\n' },
        '/b/x/SKILL.md': { content: '---\nname: x\ndescription: From B.\n---\n' },
      }
    )
    const result = loadSkillsFromBackend(backend, ['/a/', '/b/'])
    expect(result).toHaveLength(1)
    expect(result[0].description).toBe('From B.')
  })

  it('handles null content', () => {
    const backend = makeMockBackend(
      { '/s/': [{ path: '/s/x', is_dir: true }] },
      { '/s/x/SKILL.md': { content: null } }
    )
    const result = loadSkillsFromBackend(backend, ['/s/'])
    expect(result).toEqual([])
  })

  it('supports snake_case download_files method', () => {
    const backend = {
      ls(path: string) {
        return { entries: [{ path: '/skills/s1', is_dir: true }] }
      },
      download_files(paths: string[]) {
        return paths.map(() => ({
          content: '---\nname: s1\ndescription: Skill.\n---\n',
        }))
      },
    }
    const result = loadSkillsFromBackend(backend, ['/skills/'])
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('s1')
  })

  it('supports Uint8Array content', () => {
    const content = new TextEncoder().encode(
      '---\nname: binary\ndescription: Binary content.\n---\n'
    )
    const backend = makeMockBackend(
      { '/s/': [{ path: '/s/b', is_dir: true }] },
      { '/s/b/SKILL.md': { content } }
    )
    const result = loadSkillsFromBackend(backend, ['/s/'])
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('binary')
  })
})
