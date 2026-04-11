/**
 * Skill loading utilities for task-scoped skills.
 *
 * Handles reading SKILL.md files from a backend and parsing their
 * YAML frontmatter into {@link SkillMetadata} objects.  Isolated from
 * `middleware.ts` to keep I/O and parsing separate from orchestration.
 */

import type { SkillMetadata } from './types.js'

const FRONTMATTER_RE = /^---\s*\n([\s\S]*?)\n---\s*\n/
const MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024 // 10 MB

// ── Simple YAML frontmatter parser ─────────────────────────
// Handles the limited subset used by SKILL.md files:
// - Top-level scalar key-value pairs
// - Block sequences (- item)
// - One-level nested mappings (key:\n  k: v)

/** @internal */
export function parseSimpleYaml(text: string): Record<string, unknown> | null {
  const result: Record<string, unknown> = {}
  const lines = text.split('\n')
  let i = 0

  // Reject non-mapping top-level (e.g. starts with "- ")
  const firstNonEmpty = lines.find((l) => l.trim() !== '' && !l.trim().startsWith('#'))
  if (firstNonEmpty && /^\s*-\s/.test(firstNonEmpty)) {
    return null
  }

  while (i < lines.length) {
    const line = lines[i]

    // Skip empty lines and comments
    if (line.trim() === '' || line.trim().startsWith('#')) {
      i++
      continue
    }

    // Top-level key: value
    const kvMatch = line.match(/^([A-Za-z0-9_-]+)\s*:\s*(.*)/)
    if (!kvMatch) {
      // Unrecognised line at top level — likely malformed YAML
      i++
      continue
    }

    const key = kvMatch[1]
    const inlineValue = kvMatch[2].trim()

    if (inlineValue) {
      result[key] = inlineValue
      i++
    } else {
      // Look ahead for block content (list or nested mapping)
      i++
      const items: string[] = []
      const mapping: Record<string, string> = {}
      let isList = false
      let isMapping = false

      while (i < lines.length) {
        const nextLine = lines[i]
        if (nextLine.trim() === '' || nextLine.trim().startsWith('#')) {
          i++
          continue
        }

        // Not indented → back to top level
        if (!/^\s/.test(nextLine)) break

        const listItemMatch = nextLine.match(/^\s+- (.+)/)
        const nestedKvMatch = nextLine.match(/^\s+([A-Za-z0-9_-]+)\s*:\s*(.+)/)

        if (listItemMatch) {
          isList = true
          items.push(listItemMatch[1].trim())
          i++
        } else if (nestedKvMatch) {
          isMapping = true
          mapping[nestedKvMatch[1]] = nestedKvMatch[2].trim()
          i++
        } else {
          break
        }
      }

      if (isList) {
        result[key] = items
      } else if (isMapping) {
        result[key] = mapping
      }
      // If neither, key has no value — omit from result
    }
  }

  return Object.keys(result).length > 0 ? result : null
}

// ── Frontmatter parsing ────────────────────────────────────

/**
 * Parse YAML frontmatter from a SKILL.md file's content.
 *
 * Returns a `SkillMetadata` object on success, or `null` if the
 * content is invalid or missing required fields.
 */
export function parseSkillFrontmatter(content: string, path: string): SkillMetadata | null {
  if (content.length > MAX_SKILL_FILE_SIZE) {
    console.warn(`[langchain-task-steering] Skipping ${path}: file exceeds 10 MB size limit`)
    return null
  }

  const match = content.match(FRONTMATTER_RE)
  if (!match) {
    console.warn(`[langchain-task-steering] Skipping ${path}: no valid YAML frontmatter found`)
    return null
  }

  const data = parseSimpleYaml(match[1])
  if (!data) {
    console.warn(`[langchain-task-steering] Skipping ${path}: frontmatter is not a mapping`)
    return null
  }

  const name = String(data.name ?? '').trim()
  const description = String(data.description ?? '').trim()

  if (!name || !description) {
    console.warn(
      `[langchain-task-steering] Skipping ${path}: missing required 'name' or 'description'`
    )
    return null
  }

  const result: SkillMetadata = { name, description, path }

  const rawTools = data['allowed-tools']
  if (typeof rawTools === 'string') {
    result.allowedTools = rawTools.split(/\s+/)
  } else if (Array.isArray(rawTools)) {
    result.allowedTools = rawTools.map(String)
  }

  const license = String(data.license ?? '').trim() || null
  if (license) result.license = license

  const compatibility = String(data.compatibility ?? '').trim() || null
  if (compatibility) result.compatibility = compatibility

  const rawMetadata = data.metadata
  if (typeof rawMetadata === 'object' && rawMetadata !== null && !Array.isArray(rawMetadata)) {
    const metadataDict: Record<string, string> = {}
    for (const [k, v] of Object.entries(rawMetadata)) {
      metadataDict[String(k)] = String(v)
    }
    if (Object.keys(metadataDict).length > 0) {
      result.metadata = metadataDict
    }
  }

  return result
}

// ── Backend loading ────────────────────────────────────────

/**
 * Load skill metadata from backend source paths.
 *
 * For each source path, lists directories, constructs SKILL.md paths,
 * batch-downloads them, and parses frontmatter.  Mirrors the loading
 * flow in deepagents' `SkillsMiddleware`.
 *
 * The backend is duck-typed — it must provide:
 * - `ls(path)` → `{ entries: [...] }` or `[...]`
 * - `downloadFiles(paths)` or `download_files(paths)` → `[{ content, error }, ...]`
 */
export function loadSkillsFromBackend(backend: unknown, sourcePaths: string[]): SkillMetadata[] {
  const allSkills = new Map<string, SkillMetadata>()
  const b = backend as Record<string, unknown>

  for (const sourcePath of sourcePaths) {
    let lsResult: unknown
    try {
      lsResult = (b.ls as (p: string) => unknown)(sourcePath)
    } catch (exc) {
      console.warn(`[langchain-task-steering] Failed to list skill source '${sourcePath}': ${exc}`)
      continue
    }

    let entries: unknown[]
    const lsObj = lsResult as Record<string, unknown> | unknown[] | null
    if (Array.isArray(lsObj)) {
      entries = lsObj
    } else if (lsObj && typeof lsObj === 'object' && 'entries' in lsObj) {
      entries = (
        Array.isArray((lsObj as Record<string, unknown>).entries)
          ? (lsObj as Record<string, unknown>).entries
          : []
      ) as unknown[]
    } else {
      entries = []
    }

    const skillDirs: string[] = []
    for (const entry of entries) {
      if (typeof entry !== 'object' || entry === null) continue
      const e = entry as Record<string, unknown>
      const isDir = e.is_dir ?? e.isDir ?? false
      if (!isDir) continue
      const pathVal = e.path as string | undefined
      if (pathVal) skillDirs.push(pathVal)
    }

    if (skillDirs.length === 0) continue

    const mdPaths: Array<[string, string]> = skillDirs.map((dirPath) => {
      const mdPath = dirPath.endsWith('/') ? `${dirPath}SKILL.md` : `${dirPath}/SKILL.md`
      return [dirPath, mdPath]
    })

    const pathsToDownload = mdPaths.map(([, md]) => md)
    const downloadFn = (b.downloadFiles ?? b.download_files) as
      | ((paths: string[]) => unknown[])
      | undefined

    if (!downloadFn) {
      console.warn(
        `[langchain-task-steering] Backend does not have downloadFiles or download_files method`
      )
      continue
    }

    let responses: unknown[]
    try {
      responses = downloadFn.call(backend, pathsToDownload)
    } catch (exc) {
      console.warn(
        `[langchain-task-steering] Failed to download SKILL.md files from '${sourcePath}': ${exc}`
      )
      continue
    }

    for (let i = 0; i < mdPaths.length; i++) {
      const [, mdPath] = mdPaths[i]
      const response = responses[i] as Record<string, unknown> | null | undefined

      if (!response) continue
      if (response.error) continue

      const contentRaw = response.content
      if (contentRaw == null) continue

      let content: string
      if (typeof contentRaw === 'string') {
        content = contentRaw
      } else if (contentRaw instanceof Uint8Array) {
        content = new TextDecoder().decode(contentRaw)
      } else {
        content = String(contentRaw)
      }

      const skill = parseSkillFrontmatter(content, mdPath)
      if (skill) {
        // Last source wins on name conflict
        allSkills.set(skill.name, skill)
      }
    }
  }

  return Array.from(allSkills.values())
}
