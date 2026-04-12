"""Tests for task-scoped skills integration."""

import logging
import pytest
from unittest.mock import MagicMock

pytest.importorskip(
    "langchain.agents.middleware",
    reason="Requires langchain >= 1.0.0",
)

from langchain_task_steering import Task, TaskSteeringMiddleware
from tests.conftest import (
    MockModelRequest,
    MockSystemMessage,
    make_backend_with_skills,
    make_mock_tool,
    tool_a,
    tool_b,
    global_read,
)


# ════════════════════════════════════════════════════════════
# Init
# ════════════════════════════════════════════════════════════


class TestSkillsInit:
    def test_task_skills_activates_without_backend(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw._ctx.skills_active is True

    def test_no_skills_configured_inert(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        assert mw._ctx.skills_active is False

    def test_global_skills_activates_without_backend(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])
        assert mw._ctx.skills_active is True

    def test_task_skills_field_default_none(self):
        task = Task(name="a", instruction="A", tools=[tool_a])
        assert task.skills is None


# ════════════════════════════════════════════════════════════
# Skill Loading (before_agent)
# ════════════════════════════════════════════════════════════


class TestSkillStateHandling:
    def test_uses_skills_metadata_from_state(self):
        """When SkillsMiddleware already loaded skills, use them as-is."""
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        state = {
            "messages": [],
            "task_statuses": {"a": "pending"},
            "skills_metadata": [
                {"name": "s1", "description": "Already loaded", "path": "/x"}
            ],
        }
        result = mw.before_agent(state, runtime=None)
        assert result is None  # no updates needed

    def test_before_agent_only_inits_task_statuses(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        result = mw.before_agent({"messages": []}, runtime=None)
        assert result is not None
        assert "task_statuses" in result
        assert "skills_metadata" not in result


# ════════════════════════════════════════════════════════════
# Skill Scoping
# ════════════════════════════════════════════════════════════


class TestSkillScoping:
    def test_allowed_skill_names_no_active(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])
        allowed = mw._allowed_skill_names(mw._ctx, None)
        assert allowed == {"gs"}

    def test_allowed_skill_names_with_active(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1", "s2"])]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])
        allowed = mw._allowed_skill_names(mw._ctx, "a")
        assert allowed == {"gs", "s1", "s2"}

    def test_allowed_skill_names_task_without_skills(self):
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a], skills=["s1"]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])
        allowed = mw._allowed_skill_names(mw._ctx, "b")
        assert allowed == {"gs"}


# ════════════════════════════════════════════════════════════
# Skill Rendering
# ════════════════════════════════════════════════════════════


class TestSkillRendering:
    def _make_middleware(self):
        tasks = [
            Task(
                name="research",
                instruction="Do research.",
                tools=[tool_a],
                skills=["web-research"],
            ),
            Task(name="write", instruction="Write report.", tools=[tool_b]),
        ]
        return TaskSteeringMiddleware(
            tasks=tasks,
            global_skills=["formatting"],
        )

    def _state_with_skills(self):
        return {
            "messages": [],
            "task_statuses": {"research": "in_progress", "write": "pending"},
            "skills_metadata": [
                {
                    "name": "web-research",
                    "description": "Search the web.",
                    "path": "/skills/web-research/SKILL.md",
                },
                {
                    "name": "formatting",
                    "description": "Format documents.",
                    "path": "/skills/formatting/SKILL.md",
                },
                {
                    "name": "other",
                    "description": "Other skill.",
                    "path": "/skills/other/SKILL.md",
                },
            ],
        }

    def test_renders_available_skills_section(self):
        mw = self._make_middleware()
        state = self._state_with_skills()
        statuses = mw._get_statuses(mw._ctx, state)
        block = mw._render_status_block(mw._ctx, statuses, "research", state=state)
        assert "<available_skills>" in block
        assert "web-research" in block
        assert "formatting" in block
        # "other" skill should not appear in available_skills
        assert "other:" not in block
        assert "/skills/other/" not in block

    def test_no_skills_section_when_inactive(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        statuses = {"a": "in_progress"}
        block = mw._render_status_block(mw._ctx, statuses, "a", state={})
        assert "<available_skills>" not in block

    def test_skills_filtered_to_active_task(self):
        mw = self._make_middleware()
        state = self._state_with_skills()
        state["task_statuses"] = {"research": "complete", "write": "in_progress"}
        statuses = mw._get_statuses(mw._ctx, state)
        block = mw._render_status_block(mw._ctx, statuses, "write", state=state)
        assert "<available_skills>" in block
        # write task has no skills, only global_skills
        assert "formatting" in block
        assert "web-research" not in block

    def test_skill_instruction_in_rules(self):
        mw = self._make_middleware()
        state = self._state_with_skills()
        statuses = mw._get_statuses(mw._ctx, state)
        block = mw._render_status_block(mw._ctx, statuses, "research", state=state)
        assert "read its SKILL.md file" in block

    def test_no_skills_section_without_state(self):
        mw = self._make_middleware()
        statuses = {"research": "in_progress", "write": "pending"}
        block = mw._render_status_block(mw._ctx, statuses, "research")
        assert "<available_skills>" not in block

    def test_warns_on_missing_skill_names(self, caplog):
        mw = self._make_middleware()
        state = {
            "messages": [],
            "task_statuses": {"research": "in_progress", "write": "pending"},
            "skills_metadata": [
                # "web-research" and "formatting" are referenced but only "formatting" exists
                {
                    "name": "formatting",
                    "description": "Format documents.",
                    "path": "/skills/formatting/SKILL.md",
                },
            ],
        }
        statuses = mw._get_statuses(mw._ctx, state)
        with caplog.at_level(
            logging.WARNING, logger="langchain_task_steering.middleware"
        ):
            mw._render_status_block(mw._ctx, statuses, "research", state=state)
        assert "web-research" in caplog.text
        assert "not found in skills_metadata" in caplog.text

    def test_no_warning_when_all_skills_present(self, caplog):
        mw = self._make_middleware()
        state = self._state_with_skills()
        statuses = mw._get_statuses(mw._ctx, state)
        with caplog.at_level(
            logging.WARNING, logger="langchain_task_steering.middleware"
        ):
            mw._render_status_block(mw._ctx, statuses, "research", state=state)
        assert "not found in skills_metadata" not in caplog.text


# ════════════════════════════════════════════════════════════
# Tool Auto-Whitelist
# ════════════════════════════════════════════════════════════


class TestSkillToolAutoWhitelist:
    def test_read_file_and_ls_auto_whitelisted(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        allowed = mw._allowed_tool_names(mw._ctx, "a")
        assert "read_file" in allowed
        assert "ls" in allowed

    def test_no_auto_whitelist_when_no_skills(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        allowed = mw._allowed_tool_names(mw._ctx, "a")
        assert "read_file" not in allowed
        assert "ls" not in allowed

    def test_no_auto_whitelist_for_task_without_skills(self):
        """Task with no skills should not get read_file/ls even if another task has skills."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a], skills=["s1"]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)
        allowed = mw._allowed_tool_names(mw._ctx, "b")
        assert "read_file" not in allowed
        assert "ls" not in allowed

    def test_auto_whitelist_via_global_skills(self):
        """Task without own skills still gets read_file/ls when global_skills are set."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a], skills=["s1"]),
            Task(name="b", instruction="B", tools=[tool_b]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])
        allowed = mw._allowed_tool_names(mw._ctx, "b")
        assert "read_file" in allowed
        assert "ls" in allowed

    def test_auto_whitelist_independent_of_passthrough(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(
            tasks=tasks,
            backend_tools_passthrough=False,
        )
        allowed = mw._allowed_tool_names(mw._ctx, "a")
        assert "read_file" in allowed
        assert "ls" in allowed
        assert "write_file" not in allowed
        assert "execute" not in allowed

    def test_passthrough_and_skills_combined(self):
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(
            tasks=tasks,
            backend_tools_passthrough=True,
        )
        allowed = mw._allowed_tool_names(mw._ctx, "a")
        assert "read_file" in allowed
        assert "ls" in allowed
        assert "write_file" in allowed
        assert "execute" in allowed

    def test_skill_allowed_tools_whitelisted(self):
        """Tools declared in a skill's allowed_tools frontmatter are whitelisted."""
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        state = {
            "skills_metadata": [
                {
                    "name": "s1",
                    "description": "Skill 1",
                    "path": "/skills/s1/SKILL.md",
                    "allowed_tools": ["web_search", "scrape_url"],
                },
            ],
        }
        allowed = mw._allowed_tool_names(mw._ctx, "a", state=state)
        assert "web_search" in allowed
        assert "scrape_url" in allowed

    def test_skill_allowed_tools_scoped_to_active_task(self):
        """allowed_tools from skills not in scope are excluded."""
        tasks = [
            Task(name="a", instruction="A", tools=[tool_a], skills=["s1"]),
            Task(name="b", instruction="B", tools=[tool_b], skills=["s2"]),
        ]
        mw = TaskSteeringMiddleware(tasks=tasks)
        state = {
            "skills_metadata": [
                {
                    "name": "s1",
                    "description": "Skill 1",
                    "path": "/skills/s1/SKILL.md",
                    "allowed_tools": ["web_search"],
                },
                {
                    "name": "s2",
                    "description": "Skill 2",
                    "path": "/skills/s2/SKILL.md",
                    "allowed_tools": ["code_exec"],
                },
            ],
        }
        allowed_a = mw._allowed_tool_names(mw._ctx, "a", state=state)
        assert "web_search" in allowed_a
        assert "code_exec" not in allowed_a

        allowed_b = mw._allowed_tool_names(mw._ctx, "b", state=state)
        assert "code_exec" in allowed_b
        assert "web_search" not in allowed_b

    def test_skill_allowed_tools_includes_global_skills(self):
        """allowed_tools from global skills are whitelisted on any task."""
        tasks = [Task(name="a", instruction="A", tools=[tool_a])]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])
        state = {
            "skills_metadata": [
                {
                    "name": "gs",
                    "description": "Global",
                    "path": "/skills/gs/SKILL.md",
                    "allowed_tools": ["format_doc"],
                },
            ],
        }
        allowed = mw._allowed_tool_names(mw._ctx, "a", state=state)
        assert "format_doc" in allowed

    def test_skill_allowed_tools_without_state_is_noop(self):
        """Without state, allowed_tools cannot be resolved — no error."""
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        allowed = mw._allowed_tool_names(mw._ctx, "a")
        # Only the hardcoded read_file/ls, not skill-specific tools
        assert "read_file" in allowed
        assert "web_search" not in allowed

    def test_skill_without_allowed_tools_field(self):
        """Skills that don't declare allowed_tools don't break anything."""
        tasks = [Task(name="a", instruction="A", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks)
        state = {
            "skills_metadata": [
                {"name": "s1", "description": "Skill 1", "path": "/skills/s1/SKILL.md"},
            ],
        }
        allowed = mw._allowed_tool_names(mw._ctx, "a", state=state)
        assert "read_file" in allowed
        assert "ls" in allowed


# ════════════════════════════════════════════════════════════
# End-to-End
# ════════════════════════════════════════════════════════════


class TestSkillsEndToEnd:
    def test_skills_from_state_scoped_per_task(self):
        """Skills from state scoped per task, tools auto-whitelisted."""
        tasks = [
            Task(
                name="research",
                instruction="Do research.",
                tools=[tool_a],
                skills=["research-skill"],
            ),
            Task(
                name="write",
                instruction="Write report.",
                tools=[tool_b],
                skills=["writing-skill"],
            ),
        ]
        mw = TaskSteeringMiddleware(
            tasks=tasks,
            global_tools=[global_read],
            global_skills=["global-skill"],
        )

        # Simulate state with skills already loaded (by SkillsMiddleware)
        state = {
            "messages": [],
            "task_statuses": {"research": "in_progress", "write": "pending"},
            "skills_metadata": [
                {
                    "name": "research-skill",
                    "description": "Research things",
                    "path": "/skills/research-skill/SKILL.md",
                },
                {
                    "name": "writing-skill",
                    "description": "Write things",
                    "path": "/skills/writing-skill/SKILL.md",
                },
                {
                    "name": "global-skill",
                    "description": "Always available",
                    "path": "/skills/global-skill/SKILL.md",
                },
            ],
        }

        read_file_tool = make_mock_tool("read_file")
        ls_tool = make_mock_tool("ls")
        write_file_tool = make_mock_tool("write_file")

        req = MockModelRequest(
            state=state,
            system_message=MockSystemMessage("System"),
            tools=[
                tool_a,
                tool_b,
                global_read,
                read_file_tool,
                ls_tool,
                write_file_tool,
            ],
        )

        captured = {}
        mw.wrap_model_call(req, lambda r: captured.update(req=r) or MagicMock())

        scoped_names = {t.name for t in captured["req"].tools}
        # Research task tools + global + auto-whitelisted
        assert "tool_a" in scoped_names
        assert "global_read" in scoped_names
        assert "read_file" in scoped_names
        assert "ls" in scoped_names
        # Not in scope
        assert "tool_b" not in scoped_names
        assert "write_file" not in scoped_names

        # Check skills rendered in prompt
        prompt_text = str(captured["req"].system_message.content)
        assert "research-skill" in prompt_text
        assert "global-skill" in prompt_text
        assert "writing-skill" not in prompt_text

    def test_strips_skills_middleware_prompt_injection(self):
        """SkillsMiddleware's ## Skills System section is stripped."""
        tasks = [Task(name="a", instruction="Do A.", tools=[tool_a], skills=["s1"])]
        mw = TaskSteeringMiddleware(tasks=tasks, global_skills=["gs"])

        skills_mw_block = (
            "## Skills System\n\nYou have access to a skills library...\n\n"
            "- s1: Skill 1\n- gs: Global"
        )

        state = {
            "messages": [],
            "task_statuses": {"a": "in_progress"},
            "skills_metadata": [
                {"name": "s1", "description": "Skill 1", "path": "/skills/s1/SKILL.md"},
                {"name": "gs", "description": "Global", "path": "/skills/gs/SKILL.md"},
            ],
        }

        req = MockModelRequest(
            state=state,
            system_message=MockSystemMessage(
                [
                    {"type": "text", "text": "Base system prompt."},
                    {"type": "text", "text": skills_mw_block},
                ]
            ),
            tools=[tool_a],
        )

        captured = {}
        mw.wrap_model_call(req, lambda r: captured.update(req=r) or MagicMock())

        prompt_text = str(captured["req"].system_message.content)
        # SkillsMiddleware's section stripped
        assert "## Skills System" not in prompt_text
        # Base prompt preserved
        assert "Base system prompt." in prompt_text
        # Our scoped skills rendered
        assert "<available_skills>" in prompt_text
        assert "s1" in prompt_text
