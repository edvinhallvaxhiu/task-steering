"""Tests for _skills.py — skill loading utilities."""

import pytest

pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0",
)

yaml = pytest.importorskip("yaml", reason="pyyaml required for skill tests")

from langchain_task_steering._skills import (
    parse_skill_frontmatter,
    load_skills_from_backend,
)
from tests.conftest import MockBackend, MockDownloadResponse, MockLsResult


# ════════════════════════════════════════════════════════════
# parse_skill_frontmatter
# ════════════════════════════════════════════════════════════


VALID_SKILL_MD = """\
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
"""


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        result = parse_skill_frontmatter(
            VALID_SKILL_MD, "/skills/web-research/SKILL.md"
        )
        assert result is not None
        assert result["name"] == "web-research"
        assert result["description"] == "Search the web for information."
        assert result["path"] == "/skills/web-research/SKILL.md"
        assert result["license"] == "MIT"
        assert result["compatibility"] == "python>=3.10"
        assert result["allowed_tools"] == ["search", "browse"]
        assert result["metadata"] == {"category": "research"}

    def test_minimal_frontmatter(self):
        content = "---\nname: minimal\ndescription: A minimal skill.\n---\n\nBody."
        result = parse_skill_frontmatter(content, "/skills/minimal/SKILL.md")
        assert result is not None
        assert result["name"] == "minimal"
        assert result["description"] == "A minimal skill."
        assert "license" not in result
        assert "allowed_tools" not in result

    def test_missing_frontmatter(self):
        result = parse_skill_frontmatter("No frontmatter here.", "/bad/SKILL.md")
        assert result is None

    def test_missing_name(self):
        content = "---\ndescription: No name.\n---\n"
        result = parse_skill_frontmatter(content, "/bad/SKILL.md")
        assert result is None

    def test_missing_description(self):
        content = "---\nname: no-desc\n---\n"
        result = parse_skill_frontmatter(content, "/bad/SKILL.md")
        assert result is None

    def test_malformed_yaml(self):
        content = "---\n: invalid: yaml: [[\n---\n"
        result = parse_skill_frontmatter(content, "/bad/SKILL.md")
        assert result is None

    def test_non_dict_frontmatter(self):
        content = "---\n- list\n- not dict\n---\n"
        result = parse_skill_frontmatter(content, "/bad/SKILL.md")
        assert result is None

    def test_oversized_content(self):
        content = "---\nname: big\ndescription: Big.\n---\n" + "x" * (11 * 1024 * 1024)
        result = parse_skill_frontmatter(content, "/big/SKILL.md")
        assert result is None

    def test_allowed_tools_as_list(self):
        content = "---\nname: t\ndescription: d\nallowed-tools:\n  - a\n  - b\n---\n"
        result = parse_skill_frontmatter(content, "/t/SKILL.md")
        assert result is not None
        assert result["allowed_tools"] == ["a", "b"]


# ════════════════════════════════════════════════════════════
# load_skills_from_backend
# ════════════════════════════════════════════════════════════


class TestLoadSkills:
    def test_loads_from_single_source(self):
        backend = MockBackend(
            ls_results={
                "/skills/": [
                    {"path": "/skills/research", "is_dir": True},
                    {"path": "/skills/writing", "is_dir": True},
                ],
            },
            download_responses={
                "/skills/research/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: research\ndescription: Research skill.\n---\n"
                ),
                "/skills/writing/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: writing\ndescription: Writing skill.\n---\n"
                ),
            },
        )
        result = load_skills_from_backend(backend, ["/skills/"])
        assert len(result) == 2
        names = {s["name"] for s in result}
        assert names == {"research", "writing"}

    def test_loads_from_multiple_sources(self):
        backend = MockBackend(
            ls_results={
                "/a/": [{"path": "/a/s1", "is_dir": True}],
                "/b/": [{"path": "/b/s2", "is_dir": True}],
            },
            download_responses={
                "/a/s1/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: s1\ndescription: Skill 1.\n---\n"
                ),
                "/b/s2/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: s2\ndescription: Skill 2.\n---\n"
                ),
            },
        )
        result = load_skills_from_backend(backend, ["/a/", "/b/"])
        assert len(result) == 2

    def test_skips_non_directory_entries(self):
        backend = MockBackend(
            ls_results={
                "/skills/": [
                    {"path": "/skills/readme.md", "is_dir": False},
                    {"path": "/skills/research", "is_dir": True},
                ],
            },
            download_responses={
                "/skills/research/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: research\ndescription: R.\n---\n"
                ),
            },
        )
        result = load_skills_from_backend(backend, ["/skills/"])
        assert len(result) == 1
        assert result[0]["name"] == "research"

    def test_handles_download_error(self):
        backend = MockBackend(
            ls_results={
                "/skills/": [
                    {"path": "/skills/good", "is_dir": True},
                    {"path": "/skills/bad", "is_dir": True},
                ],
            },
            download_responses={
                "/skills/good/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: good\ndescription: Good.\n---\n"
                ),
                "/skills/bad/SKILL.md": MockDownloadResponse(error="not found"),
            },
        )
        result = load_skills_from_backend(backend, ["/skills/"])
        assert len(result) == 1
        assert result[0]["name"] == "good"

    def test_empty_source_dir(self):
        backend = MockBackend(ls_results={"/empty/": []})
        result = load_skills_from_backend(backend, ["/empty/"])
        assert result == []

    def test_ls_failure_graceful(self):
        class FailingBackend:
            def ls(self, path):
                raise RuntimeError("connection failed")

            def download_files(self, paths):
                return []

        result = load_skills_from_backend(FailingBackend(), ["/skills/"])
        assert result == []

    def test_last_source_wins_on_name_conflict(self):
        backend = MockBackend(
            ls_results={
                "/a/": [{"path": "/a/x", "is_dir": True}],
                "/b/": [{"path": "/b/x", "is_dir": True}],
            },
            download_responses={
                "/a/x/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: x\ndescription: From A.\n---\n"
                ),
                "/b/x/SKILL.md": MockDownloadResponse(
                    content=b"---\nname: x\ndescription: From B.\n---\n"
                ),
            },
        )
        result = load_skills_from_backend(backend, ["/a/", "/b/"])
        assert len(result) == 1
        assert result[0]["description"] == "From B."

    def test_handles_none_content(self):
        backend = MockBackend(
            ls_results={"/s/": [{"path": "/s/x", "is_dir": True}]},
            download_responses={
                "/s/x/SKILL.md": MockDownloadResponse(content=None),
            },
        )
        result = load_skills_from_backend(backend, ["/s/"])
        assert result == []

    def test_download_files_exception_logs_and_skips(self, caplog):
        """A backend that raises from download_files is logged and that source is skipped."""
        import logging

        class RaisingDownloadBackend:
            def ls(self, path):
                return MockLsResult([{"path": "/s/x", "is_dir": True}])

            def download_files(self, paths):
                raise RuntimeError("network down")

        with caplog.at_level(logging.WARNING, logger="langchain_task_steering._skills"):
            result = load_skills_from_backend(RaisingDownloadBackend(), ["/s/"])

        assert result == []
        assert any("Failed to download SKILL.md" in rec.message for rec in caplog.records)

    def test_handles_unicode_decode_error(self, caplog):
        """A non-utf8 SKILL.md body is logged and skipped without crashing."""
        import logging

        # Invalid UTF-8 byte sequence (lone continuation byte).
        bad_bytes = b"\x80\xff\xfe"
        backend = MockBackend(
            ls_results={"/s/": [{"path": "/s/x", "is_dir": True}]},
            download_responses={
                "/s/x/SKILL.md": MockDownloadResponse(content=bad_bytes),
            },
        )

        with caplog.at_level(logging.WARNING, logger="langchain_task_steering._skills"):
            result = load_skills_from_backend(backend, ["/s/"])

        assert result == []
        assert any("Error decoding" in rec.message for rec in caplog.records)
