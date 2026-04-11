"""Skill loading utilities for task-scoped skills.

Handles reading SKILL.md files from a backend and parsing their
YAML frontmatter into ``SkillMetadata`` dicts.  Isolated from
``middleware.py`` to keep I/O and parsing separate from orchestration.
"""

import logging
import re
from pathlib import PurePosixPath
from typing import Any

from .types import SkillMetadata

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


_yaml_module = None


def _lazy_yaml():
    """Lazy-import pyyaml with a helpful error if not installed."""
    global _yaml_module
    if _yaml_module is not None:
        return _yaml_module
    try:
        import yaml

        _yaml_module = yaml
        return yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required for skill loading. "
            "Install it with: pip install pyyaml  "
            "(or: pip install langchain-task-steering[skills])"
        ) from None


def parse_skill_frontmatter(content: str, path: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from a SKILL.md file's content.

    Returns a ``SkillMetadata`` dict on success, or ``None`` if the
    content is invalid or missing required fields.
    """
    if len(content) > _MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: file exceeds 10 MB size limit", path)
        return None

    match = _FRONTMATTER_RE.match(content)
    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter found", path)
        return None

    yaml = _lazy_yaml()
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Skipping %s: frontmatter is not a mapping", path)
        return None

    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()

    if not name or not description:
        logger.warning("Skipping %s: missing required 'name' or 'description'", path)
        return None

    # Optional fields
    raw_tools = data.get("allowed-tools")
    if isinstance(raw_tools, str):
        allowed_tools = raw_tools.split()
    elif isinstance(raw_tools, list):
        allowed_tools = [str(t) for t in raw_tools]
    else:
        allowed_tools = []

    license_str = str(data.get("license", "")).strip() or None
    compat_str = str(data.get("compatibility", "")).strip() or None

    raw_metadata = data.get("metadata", {})
    metadata_dict: dict[str, str] = {}
    if isinstance(raw_metadata, dict):
        metadata_dict = {str(k): str(v) for k, v in raw_metadata.items()}

    result: SkillMetadata = {
        "name": name,
        "description": description,
        "path": path,
    }
    if license_str is not None:
        result["license"] = license_str
    if compat_str is not None:
        result["compatibility"] = compat_str
    if metadata_dict:
        result["metadata"] = metadata_dict
    if allowed_tools:
        result["allowed_tools"] = allowed_tools

    return result


def load_skills_from_backend(
    backend: Any, source_paths: list[str]
) -> list[SkillMetadata]:
    """Load skill metadata from backend source paths.

    For each source path, lists directories, constructs SKILL.md paths,
    batch-downloads them, and parses frontmatter.  Mirrors the loading
    flow in deepagents' ``SkillsMiddleware`` but uses our own types.
    """
    all_skills: dict[str, SkillMetadata] = {}

    for source_path in source_paths:
        try:
            ls_result = backend.ls(source_path)
        except Exception as exc:
            logger.warning("Failed to list skill source '%s': %s", source_path, exc)
            continue

        # LsResult has .entries, but duck-type for flexibility
        entries = getattr(ls_result, "entries", None)
        if entries is None:
            entries = ls_result if isinstance(ls_result, list) else []

        # Collect subdirectories (duck-type: dict or object with .is_dir/.path)
        skill_dirs: list[str] = []
        for entry in entries or []:
            is_dir = (
                entry.get("is_dir")
                if isinstance(entry, dict)
                else getattr(entry, "is_dir", False)
            )
            if not is_dir:
                continue
            path_val = (
                entry["path"]
                if isinstance(entry, dict)
                else getattr(entry, "path", None)
            )
            if path_val:
                skill_dirs.append(path_val)

        if not skill_dirs:
            continue

        # Build SKILL.md paths
        md_paths: list[tuple[str, str]] = []
        for dir_path in skill_dirs:
            md_path = str(PurePosixPath(dir_path) / "SKILL.md")
            md_paths.append((dir_path, md_path))

        # Batch download
        paths_to_download = [md for _, md in md_paths]
        try:
            responses = backend.download_files(paths_to_download)
        except Exception as exc:
            logger.warning(
                "Failed to download SKILL.md files from '%s': %s",
                source_path,
                exc,
            )
            continue

        # Parse each response
        for (dir_path, md_path), response in zip(md_paths, responses, strict=True):
            error = getattr(response, "error", None)
            if error:
                continue

            content_bytes = getattr(response, "content", None)
            if content_bytes is None:
                continue

            try:
                content = (
                    content_bytes.decode("utf-8")
                    if isinstance(content_bytes, bytes)
                    else str(content_bytes)
                )
            except UnicodeDecodeError as exc:
                logger.warning("Error decoding %s: %s", md_path, exc)
                continue

            skill = parse_skill_frontmatter(content, md_path)
            if skill:
                # Last source wins on name conflict
                all_skills[skill["name"]] = skill

    return list(all_skills.values())
