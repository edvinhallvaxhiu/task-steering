"""TaskSteeringMiddleware — implicit state machine for LangChain v1 agents."""

import copy
import logging
import typing
import warnings
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware import hook_config
from langchain.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool, ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict

from ._hooks import WRAP_HOOK_PAIRS, overrides_base
from .types import (
    AbortAll,
    Task,
    TaskMiddleware,
    TaskStatus,
    TaskSteeringState,
    TaskSummarization,
    Workflow,
    _REQUIRE_ALL,
)

logger = logging.getLogger(__name__)

_TRANSITION_TOOL_NAME = "update_task_status"
_ACTIVATE_TOOL_NAME = "activate_workflow"
_DEACTIVATE_TOOL_NAME = "deactivate_workflow"


def _overrides_task(middleware: TaskMiddleware, method_name: str) -> bool:
    """Check if a TaskMiddleware subclass overrides a TaskMiddleware hook."""
    return getattr(type(middleware), method_name) is not getattr(
        TaskMiddleware, method_name, None
    )


def _merge_hook_updates(
    base: dict[str, Any] | None, updates: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Merge lifecycle hook return values, appending ``messages`` lists."""
    if not updates:
        return base
    if not base:
        return dict(updates)
    merged = {**base, **updates}
    if "messages" in base and "messages" in updates:
        merged["messages"] = base["messages"] + updates["messages"]
    return merged


class _ComposedTaskMiddleware(TaskMiddleware):
    """Chains multiple ``TaskMiddleware`` instances into one.

    Wrap-style hooks are discovered dynamically from ``AgentMiddleware``
    at import time via ``WRAP_HOOK_PAIRS``, so new hooks added by
    LangChain are automatically chained — no manual updates needed.

    Composition semantics (matching LangChain's agent middleware list):
    - Wrap-style hooks: first = outermost wrapper, chain inward.
    - ``validate_completion``: all validators run; first error wins.
    - ``on_start`` / ``on_complete``: all fire in order.
    - ``tools``: merged (deduplicated by name).
    - ``state_schema``: last non-default schema wins.
    """

    def __init__(self, middlewares: list[TaskMiddleware]) -> None:
        super().__init__()
        self._middlewares = middlewares

        # Merge tools (deduplicated)
        seen: set[str] = set()
        merged_tools: list = []
        for mw in middlewares:
            for t in getattr(mw, "tools", None) or []:
                if t.name not in seen:
                    seen.add(t.name)
                    merged_tools.append(t)
        self.tools = merged_tools

        # Forward last non-default state_schema
        for mw in reversed(middlewares):
            schema = getattr(mw, "state_schema", None)
            if schema is not None:
                self.state_schema = schema
                break

        # Dynamically bind chaining methods for all discovered wrap hooks
        for sync_name, async_name in WRAP_HOOK_PAIRS:
            self._bind_sync_chain(sync_name)
            if async_name:
                self._bind_async_chain(async_name)

    def _bind_sync_chain(self, method_name: str) -> None:
        middlewares = self._middlewares

        def chained(request, handler, _name=method_name):
            chain = handler
            for mw in reversed(middlewares):
                if overrides_base(mw, _name):
                    outer, inner = mw, chain
                    chain = lambda r, _o=outer, _i=inner: getattr(_o, _name)(r, _i)
            return chain(request)

        setattr(self, method_name, chained)

    def _bind_async_chain(self, method_name: str) -> None:
        middlewares = self._middlewares

        async def chained(request, handler, _name=method_name):
            chain = handler
            for mw in reversed(middlewares):
                if overrides_base(mw, _name):
                    outer, inner = mw, chain

                    async def make_chain(r, _o=outer, _i=inner):
                        return await getattr(_o, _name)(r, _i)

                    chain = make_chain
            return await chain(request)

        setattr(self, method_name, chained)

    def validate_completion(self, state):
        for mw in self._middlewares:
            if _overrides_task(mw, "validate_completion"):
                error = mw.validate_completion(state)
                if error:
                    return error
        return None

    def on_start(self, state):
        merged = None
        for mw in self._middlewares:
            if _overrides_task(mw, "on_start"):
                updates = mw.on_start(state)
                merged = _merge_hook_updates(merged, updates)
        return merged

    def on_complete(self, state):
        merged = None
        for mw in self._middlewares:
            if _overrides_task(mw, "on_complete"):
                updates = mw.on_complete(state)
                if isinstance(updates, AbortAll):
                    return updates
                merged = _merge_hook_updates(merged, updates)
        return merged

    async def avalidate_completion(self, state):
        for mw in self._middlewares:
            if _overrides_task(mw, "avalidate_completion") or _overrides_task(
                mw, "validate_completion"
            ):
                error = await mw.avalidate_completion(state)
                if error:
                    return error
        return None

    async def aon_start(self, state):
        merged = None
        for mw in self._middlewares:
            if _overrides_task(mw, "aon_start") or _overrides_task(mw, "on_start"):
                updates = await mw.aon_start(state)
                merged = _merge_hook_updates(merged, updates)
        return merged

    async def aon_complete(self, state):
        merged = None
        for mw in self._middlewares:
            if _overrides_task(mw, "aon_complete") or _overrides_task(
                mw, "on_complete"
            ):
                updates = await mw.aon_complete(state)
                if isinstance(updates, AbortAll):
                    return updates
                merged = _merge_hook_updates(merged, updates)
        return merged


def _is_valid_middleware(mw: Any) -> bool:
    """Check if an object is a valid middleware (TaskMiddleware or AgentMiddleware)."""
    if isinstance(mw, (TaskMiddleware, AgentMiddleware)):
        return True
    # Duck-type check: has at least one wrap-style hook callable
    for sync_name, async_name in WRAP_HOOK_PAIRS:
        if callable(getattr(mw, sync_name, None)):
            return True
        if async_name and callable(getattr(mw, async_name, None)):
            return True
    return False


def _coerce_middleware(mw: Any) -> TaskMiddleware | None:
    """Coerce a middleware to TaskMiddleware, or None with a warning."""
    if isinstance(mw, TaskMiddleware):
        return mw
    if isinstance(mw, AgentMiddleware):
        from .adapter import AgentMiddlewareAdapter

        return AgentMiddlewareAdapter(mw)
    if _is_valid_middleware(mw):
        from .adapter import AgentMiddlewareAdapter

        return AgentMiddlewareAdapter(mw)
    warnings.warn(
        f"Ignoring invalid task middleware of type {type(mw).__name__!r}. "
        f"Expected TaskMiddleware, AgentMiddleware, or an object with "
        f"wrap-style hooks (e.g. wrap_model_call).",
        stacklevel=3,
    )
    return None


def _normalize_middleware(
    middleware: TaskMiddleware | list | AgentMiddleware | None,
) -> TaskMiddleware | None:
    """Normalize any middleware input into a single TaskMiddleware or None."""
    if middleware is None:
        return None
    if isinstance(middleware, list):
        valid = [
            m for m in (_coerce_middleware(x) for x in middleware) if m is not None
        ]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        return _ComposedTaskMiddleware(valid)
    return _coerce_middleware(middleware)


from dataclasses import dataclass, field as dataclass_field

_STATUS_ICONS: dict[str, str] = {
    TaskStatus.PENDING.value: "[ ]",
    TaskStatus.IN_PROGRESS.value: "[>]",
    TaskStatus.COMPLETE.value: "[x]",
    TaskStatus.ABORTED.value: "[-]",
}

# Statuses that count as "task is done" — neither blocks ordering nor needs
# nudging to complete. Promoted to module scope so the same definition is
# shared by every transition / nudge / ordering check.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    (TaskStatus.COMPLETE.value, TaskStatus.ABORTED.value)
)


def _find_dupes(names: list[str]) -> set[str]:
    """Return duplicate entries in *names*."""
    seen: set[str] = set()
    return {n for n in names if n in seen or seen.add(n)}  # type: ignore[func-returns-value]


def _resolve_required_tasks(
    required_tasks: list[str] | tuple[str, ...] | None,
    all_task_names: set[str],
    context_label: str = "",
) -> set[str]:
    """Resolve the *required_tasks* sentinel/list into a concrete set."""
    if required_tasks is _REQUIRE_ALL:
        return set(all_task_names)
    if required_tasks is not None and "*" in required_tasks:
        return set(all_task_names)
    if required_tasks is not None:
        unknown = set(required_tasks) - all_task_names
        if unknown:
            prefix = f" in {context_label}" if context_label else ""
            raise ValueError(f"Unknown required tasks{prefix}: {unknown}")
        return set(required_tasks)
    return set()


def _validate_and_normalize_tasks(tasks: list[Task]) -> list[Task]:
    """Check for duplicate names, shallow-copy, normalize middleware."""
    names = [t.name for t in tasks]
    dupes = _find_dupes(names)
    if dupes:
        raise ValueError(f"Duplicate task names: {dupes}")
    tasks = [copy.copy(t) for t in tasks]
    for task in tasks:
        task.middleware = _normalize_middleware(task.middleware)
        task.tools = list(task.tools)
    return tasks


def _dedup_tools(tool_iter) -> list:
    """Deduplicate tools by name, preserving order."""
    seen: set[str] = set()
    out: list = []
    for t in tool_iter:
        if t.name not in seen:
            seen.add(t.name)
            out.append(t)
    return out


@dataclass
class _PipelineContext:
    """Resolved pipeline data passed to shared base-class methods."""

    tasks: list  # list[Task]
    task_order: list[str]
    task_map: dict  # dict[str, Task]
    global_tools: list
    enforce_order: bool
    required_tasks: set[str]
    global_skills: list[str]
    skills_active: bool
    skill_required_tools: frozenset[str]
    label: str | None = None  # Workflow name for prompt tag, or None


def _build_pipeline_context(
    tasks: list[Task],
    global_tools: list,
    enforce_order: bool,
    required_tasks: set[str],
    global_skills: list[str] | None,
    label: str | None = None,
) -> _PipelineContext:
    """Build a fully resolved ``_PipelineContext``."""
    gs = list(global_skills or [])
    skills_active = bool(any(t.skills for t in tasks) or gs)
    skill_required_tools: frozenset[str] = (
        frozenset({"read_file", "ls"}) if skills_active else frozenset()
    )
    return _PipelineContext(
        tasks=tasks,
        task_order=[t.name for t in tasks],
        task_map={t.name: t for t in tasks},
        global_tools=list(global_tools),
        enforce_order=enforce_order,
        required_tasks=required_tasks,
        global_skills=gs,
        skills_active=skills_active,
        skill_required_tools=skill_required_tools,
        label=label,
    )


class _TaskSteeringBase(AgentMiddleware[TaskSteeringState]):
    """Base class with all shared pipeline logic.

    Subclasses must implement ``_get_pipeline_ctx`` and ``before_agent``.
    """

    DEFAULT_BACKEND_TOOLS: frozenset[str] = frozenset(
        {
            # FilesystemMiddleware
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
            # TodoListMiddleware
            "write_todos",
            # SubAgentMiddleware
            "task",
            # AsyncSubAgentMiddleware
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }
    )

    state_schema = TaskSteeringState

    # Tool names that always pass through wrap_tool_call to handler
    # (e.g. activate/deactivate in workflow mode).
    _management_tool_names: set[str] = set()

    def __init__(
        self,
        *,
        max_nudges: int = 3,
        backend_tools_passthrough: bool = False,
        backend_tools: set[str] | None = None,
        model: Any = None,
    ) -> None:
        super().__init__()
        self._max_nudges = max_nudges
        self._model = model
        self._backend_tools = (
            frozenset(backend_tools)
            if backend_tools is not None
            else self.DEFAULT_BACKEND_TOOLS
        )
        self._backend_tools_passthrough = backend_tools_passthrough
        # Skill names already warned about — keeps the per-render warning
        # from spamming logs every model call.
        self._warned_missing_skills: set[str] = set()

    # ── Abstract-ish methods for subclasses ──────────────────

    def _get_pipeline_ctx(self, state: dict) -> _PipelineContext | None:
        """Return the pipeline context for the current state, or None."""
        raise NotImplementedError

    def _extra_allowed_tool_names(self) -> frozenset[str]:
        """Extra tool names to add to allowed set when pipeline is active."""
        return frozenset()

    def _on_no_pipeline_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Called by wrap_model_call when _get_pipeline_ctx returns None."""
        return handler(request)

    def _on_no_pipeline_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Called by wrap_tool_call when _get_pipeline_ctx returns None."""
        return handler(request)

    async def _aon_no_pipeline_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of _on_no_pipeline_model_call."""
        return await handler(request)

    async def _aon_no_pipeline_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of _on_no_pipeline_tool_call."""
        return await handler(request)

    def _build_nudge_message(self, incomplete: list[str], state: dict) -> HumanMessage:
        """Build the nudge HumanMessage for after_agent."""
        task_list = ", ".join(incomplete)
        return HumanMessage(
            content=(
                f"You have not completed the following required tasks: "
                f"{task_list}. Please continue."
            ),
            additional_kwargs={
                "task_steering": {"kind": "nudge", "incomplete_tasks": incomplete},
            },
        )

    # ── Shared helpers (take ctx param) ─────────────────────

    @staticmethod
    def _get_statuses(ctx: _PipelineContext, state: dict) -> dict[str, str]:
        raw = state.get("task_statuses") or {}
        return {t.name: raw.get(t.name, TaskStatus.PENDING.value) for t in ctx.tasks}

    @staticmethod
    def _active_task(ctx: _PipelineContext, statuses: dict[str, str]) -> str | None:
        for name in ctx.task_order:
            if statuses.get(name) == TaskStatus.IN_PROGRESS.value:
                return name
        return None

    @staticmethod
    def _overrides(middleware: AgentMiddleware, method_name: str) -> bool:
        """Check if a middleware subclass actually overrides a hook method."""
        return overrides_base(middleware, method_name)

    @staticmethod
    def _get_task_middleware(
        ctx: _PipelineContext, task_name: str | None
    ) -> TaskMiddleware | None:
        if task_name is None:
            return None
        task = ctx.task_map.get(task_name)
        return task.middleware if task else None

    def _allowed_tool_names(
        self, ctx: _PipelineContext, active_name: str | None, state: dict | None = None
    ) -> set[str]:
        names = {_TRANSITION_TOOL_NAME}
        names.update(self._extra_allowed_tool_names())
        names.update(t.name for t in ctx.global_tools)
        if active_name:
            task = ctx.task_map[active_name]
            names.update(t.name for t in task.tools)
            # Include tools contributed by task middleware (e.g. adapters)
            if task.middleware and hasattr(task.middleware, "tools"):
                names.update(t.name for t in (task.middleware.tools or []))
        if self._backend_tools_passthrough:
            names.update(self._backend_tools)
        if ctx.skills_active:
            allowed_skills = self._allowed_skill_names(ctx, active_name)
            if allowed_skills:
                names.update(ctx.skill_required_tools)
                # Whitelist tools declared by visible skills (allowed_tools frontmatter)
                if state is not None:
                    for skill in state.get("skills_metadata") or []:
                        if skill["name"] in allowed_skills:
                            names.update(skill.get("allowed_tools") or [])
        return names

    @staticmethod
    def _allowed_skill_names(
        ctx: _PipelineContext, active_name: str | None
    ) -> set[str]:
        """Return skill names visible for the given active task."""
        names = set(ctx.global_skills)
        if active_name:
            task = ctx.task_map[active_name]
            if task.skills:
                names.update(task.skills)
        return names

    def get_backend_tools(self) -> frozenset[str]:
        """Return the effective backend tools whitelist."""
        return self._backend_tools

    @staticmethod
    def _merge_state_schemas(all_tasks: list) -> type:
        """Merge ``TaskSteeringState`` with state schemas from task middlewares."""
        base_fields = set(typing.get_type_hints(TaskSteeringState, include_extras=True))
        task_schemas: list[type] = []

        for task in all_tasks:
            schema = (
                getattr(task.middleware, "state_schema", None)
                if task.middleware
                else None
            )
            if schema is None:
                continue
            schema_fields = set(typing.get_type_hints(schema, include_extras=True))
            if schema_fields - base_fields:
                task_schemas.append(schema)

        if not task_schemas:
            return TaskSteeringState

        merged: dict[str, Any] = {}
        for schema in [TaskSteeringState, *task_schemas]:
            merged.update(typing.get_type_hints(schema, include_extras=True))

        return TypedDict("TaskSteeringState", merged)  # type: ignore[call-overload]

    # ── Shared render/validate/lifecycle helpers ─────────────

    def _render_status_block(
        self,
        ctx: _PipelineContext,
        statuses: dict[str, str],
        active: str | None,
        state: dict | None = None,
    ) -> str:
        """Render the task pipeline block.

        When ``ctx.label`` is set, produces ``<task_pipeline workflow="name">``.
        When ``ctx.label`` is None, produces ``<task_pipeline>``.
        """
        if ctx.label is not None:
            lines = [f'\n<task_pipeline workflow="{ctx.label}">']
        else:
            lines = ["\n<task_pipeline>"]

        has_optional = False
        for t in ctx.tasks:
            s = statuses.get(t.name, TaskStatus.PENDING.value)
            optional_tag = ""
            if t.name not in ctx.required_tasks:
                has_optional = True
                optional_tag = " [optional]"
            lines.append(
                f"  {_STATUS_ICONS.get(s, '[?]')} {t.name} ({s}){optional_tag}"
            )

        if active and active in ctx.task_map:
            lines.append(f'\n  <current_task name="{active}">')
            lines.append(f"    {ctx.task_map[active].instruction}")
            if active not in ctx.required_tasks:
                lines.append("")
                lines.append(
                    "    This task is optional. You may set it to 'in_progress' to review,"
                )
                lines.append(
                    "    then abort it (update_task_status with status='aborted') if it's not"
                )
                lines.append(
                    "    needed. Once you call any tool for this task, you are committed to"
                )
                lines.append("    completing it.")
            lines.append("  </current_task>")

        # ── Skill rendering ──────────────────────────────────────
        has_visible_skills = False
        if ctx.skills_active and state is not None:
            all_skills = state.get("skills_metadata") or []
            allowed_names = self._allowed_skill_names(ctx, active)
            visible_skills = [s for s in all_skills if s["name"] in allowed_names]

            # Only task mode (no label) produces missing skill warnings
            if ctx.label is None:
                available_names = {s["name"] for s in all_skills}
                missing = allowed_names - available_names
                new_missing = missing - self._warned_missing_skills
                if new_missing:
                    logger.warning(
                        "Skill(s) %s referenced by task/global config but not found "
                        "in skills_metadata state. Check skill names and ensure "
                        "skills are loaded (e.g. via SkillsMiddleware).",
                        ", ".join(sorted(new_missing)),
                    )
                    self._warned_missing_skills.update(new_missing)

            if visible_skills:
                has_visible_skills = True
                lines.append("\n  <available_skills>")
                for skill in visible_skills:
                    desc = skill.get("description", "No description.")
                    lines.append(f"    - {skill['name']}: {desc} Path: {skill['path']}")
                lines.append("  </available_skills>")

        if ctx.enforce_order or has_visible_skills or has_optional:
            lines.append("\n  <rules>")
            if ctx.enforce_order:
                order_str = " -> ".join(ctx.task_order)
                lines.append(f"    Required order: {order_str}")
            lines.append("    Use update_task_status to advance. Do not skip tasks.")
            if has_optional:
                lines.append(
                    "    Tasks marked [optional] can be aborted before their first tool"
                )
                lines.append(
                    "    call (update_task_status with status='aborted'). After the first"
                )
                lines.append("    tool call you are committed to completing the task.")
            if has_visible_skills:
                lines.append(
                    "    To use a skill, read its SKILL.md file for full instructions."
                )
            lines.append("  </rules>")

        # Only task mode (no label) produces the verbose skill_usage block
        if has_visible_skills and ctx.label is None:
            lines.append("")
            lines.append("  <skill_usage>")
            lines.append("    **How to Use Skills (Progressive Disclosure):**")
            lines.append("")
            lines.append(
                "    Skills follow a progressive disclosure pattern - you see"
                " their name and description above, but only read full"
                " instructions when needed:"
            )
            lines.append("")
            lines.append(
                "    1. **Recognize when a skill applies**: Check if the"
                " user's task matches a skill's description"
            )
            lines.append(
                "    2. **Read the skill's full instructions**: Use the path"
                " shown in the skill list above"
            )
            lines.append(
                "    3. **Follow the skill's instructions**: SKILL.md"
                " contains step-by-step workflows, best practices, and"
                " examples"
            )
            lines.append(
                "    4. **Access supporting files**: Skills may include"
                " helper scripts, configs, or reference docs - use absolute"
                " paths"
            )
            lines.append("")
            lines.append("    **When to Use Skills:**")
            lines.append("    - User's request matches a skill's domain")
            lines.append("    - You need specialized knowledge or structured workflows")
            lines.append("    - A skill provides proven patterns for complex tasks")
            lines.append("")
            lines.append("    **Executing Skill Scripts:**")
            lines.append(
                "    Skills may contain Python scripts or other executable"
                " files. Always use absolute paths from the skill list."
            )
            lines.append("  </skill_usage>")

        lines.append("</task_pipeline>")
        return "\n".join(lines)

    def _validate_transition(
        self,
        request: ToolCallRequest,
        statuses: dict[str, str],
        ctx: _PipelineContext,
    ) -> ToolMessage | None:
        """Pre-handler validation for transitions. Returns a rejection ToolMessage or None."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if target == TaskStatus.IN_PROGRESS.value:
            already_active = self._active_task(ctx, statuses)
            if already_active:
                return ToolMessage(
                    content=(
                        f"Cannot start '{task_name}': '{already_active}' is already "
                        f"in progress. Complete it first."
                    ),
                    tool_call_id=request.tool_call["id"],
                )

        if target == TaskStatus.COMPLETE.value and task_name in ctx.task_map:
            task_mw = self._get_task_middleware(ctx, task_name)
            if task_mw and _overrides_task(task_mw, "validate_completion"):
                error = task_mw.validate_completion(request.state)
                if error:
                    return ToolMessage(
                        content=(
                            f"Cannot complete '{task_name}': {error}. "
                            f"Address the issues then try again."
                        ),
                        tool_call_id=request.tool_call["id"],
                    )
        return None

    async def _avalidate_transition(
        self,
        request: ToolCallRequest,
        statuses: dict[str, str],
        ctx: _PipelineContext,
    ) -> ToolMessage | None:
        """Async pre-handler validation for transitions."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if target == TaskStatus.IN_PROGRESS.value:
            already_active = self._active_task(ctx, statuses)
            if already_active:
                return ToolMessage(
                    content=(
                        f"Cannot start '{task_name}': '{already_active}' is already "
                        f"in progress. Complete it first."
                    ),
                    tool_call_id=request.tool_call["id"],
                )

        if target == TaskStatus.COMPLETE.value and task_name in ctx.task_map:
            task_mw = self._get_task_middleware(ctx, task_name)
            if task_mw and (
                _overrides_task(task_mw, "avalidate_completion")
                or _overrides_task(task_mw, "validate_completion")
            ):
                error = await task_mw.avalidate_completion(request.state)
                if error:
                    return ToolMessage(
                        content=(
                            f"Cannot complete '{task_name}': {error}. "
                            f"Address the issues then try again."
                        ),
                        tool_call_id=request.tool_call["id"],
                    )
        return None

    def _fire_lifecycle_hooks(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
        statuses: dict[str, str],
        ctx: _PipelineContext,
    ) -> ToolMessage | Command:
        """Fire on_start/on_complete and merge returned updates into the Command."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if not isinstance(result, Command) or task_name not in ctx.task_map:
            return result

        # Aborted is a user-driven transition: no lifecycle hooks fire and
        # no summarization runs (the task never produced meaningful output).
        if target == TaskStatus.ABORTED.value:
            return result

        task_mw = self._get_task_middleware(ctx, task_name)

        if task_mw:
            updated_statuses = {**statuses, task_name: target}
            post_state = {**request.state, "task_statuses": updated_statuses}

            if target == TaskStatus.IN_PROGRESS.value:
                updates = task_mw.on_start(post_state)
            elif target == TaskStatus.COMPLETE.value:
                updates = task_mw.on_complete(post_state)
            else:
                updates = None

            if isinstance(updates, AbortAll):
                return self._apply_abort_all(result, task_name, updates, ctx)

            if updates:
                merged = _merge_hook_updates(dict(result.update), updates)
                result = Command(update=merged)

        if target == TaskStatus.IN_PROGRESS.value:
            result = self._record_task_start(result, request.state, task_name, ctx)
        elif target == TaskStatus.COMPLETE.value:
            result = self._apply_summarization(result, request.state, task_name, ctx)

        return result

    async def _afire_lifecycle_hooks(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
        statuses: dict[str, str],
        ctx: _PipelineContext,
    ) -> ToolMessage | Command:
        """Async version of _fire_lifecycle_hooks."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if not isinstance(result, Command) or task_name not in ctx.task_map:
            return result

        if target == TaskStatus.ABORTED.value:
            return result

        task_mw = self._get_task_middleware(ctx, task_name)

        if task_mw:
            updated_statuses = {**statuses, task_name: target}
            post_state = {**request.state, "task_statuses": updated_statuses}

            if target == TaskStatus.IN_PROGRESS.value:
                updates = await task_mw.aon_start(post_state)
            elif target == TaskStatus.COMPLETE.value:
                updates = await task_mw.aon_complete(post_state)
            else:
                updates = None

            if isinstance(updates, AbortAll):
                return self._apply_abort_all(result, task_name, updates, ctx)

            if updates:
                merged = _merge_hook_updates(dict(result.update), updates)
                result = Command(update=merged)

        if target == TaskStatus.IN_PROGRESS.value:
            result = self._record_task_start(result, request.state, task_name, ctx)
        elif target == TaskStatus.COMPLETE.value:
            result = await self._aapply_summarization(
                result,
                request.state,
                task_name,
                ctx,
            )

        return result

    @staticmethod
    def _apply_abort_all(
        result: Command,
        completed_task: str,
        abort: AbortAll,
        ctx: _PipelineContext,
    ) -> Command:
        """Apply an :class:`AbortAll` signal from ``on_complete``.

        The task that triggered the signal (``completed_task``) remains
        marked ``complete`` — ``AbortAll`` is a downstream decision about
        *remaining* tasks, not a rollback.  Any still-``pending`` or
        ``in_progress`` task is marked ``aborted``, the transition
        ToolMessage is extended with the abort reason, and in workflow
        mode the active workflow is deactivated.
        """
        update = dict(result.update) if result.update else {}
        statuses = dict(update.get("task_statuses") or {})

        aborted_names: list[str] = []
        for name in ctx.task_order:
            if statuses.get(name) in (
                TaskStatus.PENDING.value,
                TaskStatus.IN_PROGRESS.value,
            ):
                statuses[name] = TaskStatus.ABORTED.value
                aborted_names.append(name)
        update["task_statuses"] = statuses

        workflow_mode = ctx.label is not None
        if workflow_mode:
            update["active_workflow"] = None
            update["task_message_starts"] = {}
            update["nudge_count"] = 0

        # Extend the transition ToolMessage with the abort reason.
        existing_msgs = list(update.get("messages", []))
        note_parts = [
            f"All remaining tasks aborted: {abort.reason}",
        ]
        if aborted_names:
            note_parts.append(f"Aborted: {', '.join(aborted_names)}")
        if workflow_mode:
            note_parts.append(f"Workflow '{ctx.label}' deactivated.")
        note = "\n\n" + "\n".join(note_parts)

        for i, msg in enumerate(existing_msgs):
            if isinstance(msg, ToolMessage):
                existing_msgs[i] = msg.model_copy(
                    update={"content": f"{msg.content}{note}"}
                )
                break
        else:
            # Invariant: the transition Command (built by
            # _execute_task_transition) always carries a ToolMessage. If we
            # ever reach here, the upstream contract has changed and the
            # abort note would silently fall on the floor — fail loudly.
            raise RuntimeError(
                "AbortAll: transition Command had no ToolMessage to extend."
            )

        update["messages"] = existing_msgs
        return Command(update=update)

    def _gate_tool(
        self,
        request: ToolCallRequest,
        active_name: str | None,
        ctx: _PipelineContext,
        state: dict,
    ) -> ToolMessage | None:
        """Reject tool calls not in scope for the active task."""
        allowed = self._allowed_tool_names(ctx, active_name, state=state)
        if request.tool_call["name"] not in allowed:
            return ToolMessage(
                content=(
                    f"Tool '{request.tool_call['name']}' is not available "
                    f"for the current task."
                ),
                tool_call_id=request.tool_call["id"],
            )
        return None

    def _prepare_model_request(
        self, request: ModelRequest, ctx: _PipelineContext
    ) -> tuple[ModelRequest, str | None]:
        """Build the modified model request with pipeline prompt and scoped tools."""
        statuses = self._get_statuses(ctx, request.state)
        active_name = self._active_task(ctx, statuses)

        block = self._render_status_block(
            ctx, statuses, active_name, state=request.state
        )
        existing = (
            list(request.system_message.content_blocks)
            if request.system_message is not None
            else []
        )

        # Strip SkillsMiddleware's global prompt injection — we replace it
        # with per-task scoped skills in the pipeline block.
        if ctx.skills_active:
            existing = [
                b
                for b in existing
                if not (
                    isinstance(b, dict)
                    and b.get("type") == "text"
                    and "## Skills System" in (b.get("text") or "")
                )
            ]

        new_content = existing + [{"type": "text", "text": block}]

        allowed_names = self._allowed_tool_names(ctx, active_name, state=request.state)
        scoped = [t for t in request.tools if t.name in allowed_names]

        overrides: dict[str, Any] = {
            "system_message": SystemMessage(content=new_content),
            "tools": scoped,
        }

        active_task = ctx.task_map.get(active_name) if active_name else None
        if active_task is not None and active_task.model_settings:
            overrides["model_settings"] = {
                **(request.model_settings or {}),
                **active_task.model_settings,
            }

        modified = request.override(**overrides)
        return modified, active_name

    # ── Summarization helpers ───────────────────────────────

    @staticmethod
    def _record_task_start(
        result: Command, state: dict, task_name: str, ctx: _PipelineContext
    ) -> Command:
        """Inject ``task_message_starts[task_name]`` into the Command update.

        Always records (not just when summarization is configured) so the
        abort-commitment check can detect whether any tool calls were made
        during the task's in_progress window.

        ``start_index`` points past every message already in ``state`` plus
        every message the transition ``Command`` will append (the
        transition ``ToolMessage`` and any messages returned by
        ``on_start``), so the first "task message" lands at this index.
        """
        update = dict(result.update) if result.update else {}
        pending_msgs = update.get("messages") or []
        start_index = len(state.get("messages", [])) + len(pending_msgs)
        starts = dict(state.get("task_message_starts") or {})
        starts[task_name] = start_index
        update["task_message_starts"] = starts
        return Command(update=update)

    def _prepare_summarization(
        self, state: dict, task_name: str, ctx: _PipelineContext
    ) -> tuple[Task, TaskSummarization, list, list, Any] | None:
        """Shared setup for sync/async summarization.

        Returns ``(task, cfg, task_messages, remove_ops, model)``
        or ``None`` if summarization should be skipped.

        Task messages are those *between* the in_progress transition and
        the complete-transition AIMessage (which stays untouched).  The
        summary text is injected into the transition ``ToolMessage``
        instead of creating a replacement ``AIMessage``.
        """
        task = ctx.task_map[task_name]
        if task.summarize is None:
            return None

        messages = state.get("messages", [])
        starts = state.get("task_message_starts") or {}
        start_index = starts.get(task_name)
        if start_index is None:
            return None

        # Exclude the complete-transition AIMessage (last element).
        task_messages = messages[start_index : len(messages) - 1]
        if not task_messages:
            return None

        cfg = task.summarize
        model = cfg.model or self._model

        if cfg.mode == "replace":
            remove_ops = [
                RemoveMessage(id=m.id) for m in task_messages if getattr(m, "id", None)
            ]
        else:
            if model is None:
                logger.warning(
                    "Skipping summarization for task '%s': no model configured. "
                    "Set model on TaskSummarization or TaskSteeringMiddleware.",
                    task_name,
                )
                return None
            remove_ops = [
                RemoveMessage(id=m.id)
                for m in task_messages
                if isinstance(m, (AIMessage, ToolMessage)) and getattr(m, "id", None)
            ]

        return task, cfg, task_messages, remove_ops, model

    @staticmethod
    def _finalize_summarization(
        result: Command,
        state: dict,
        task_name: str,
        cfg: TaskSummarization,
        remove_ops: list,
        summary: str,
    ) -> Command:
        """Inject remove ops and rewrite the transition ToolMessage with the summary."""
        update = dict(result.update) if result.update else {}
        existing_msgs = list(update.get("messages", []))

        for i, msg in enumerate(existing_msgs):
            if isinstance(msg, ToolMessage):
                existing_msgs[i] = msg.model_copy(
                    update={"content": f"{msg.content}\n\nTask summary:\n{summary}"}
                )
                break

        # Strip text from the complete-transition AIMessage, keeping only tool_calls
        trim_ops: list = []
        if cfg.trim_complete_message:
            messages = state.get("messages", [])
            if messages:
                complete_ai = messages[-1]
                if isinstance(complete_ai, AIMessage) and getattr(
                    complete_ai, "id", None
                ):
                    trim_ops = [
                        AIMessage(
                            content="",
                            id=complete_ai.id,
                            tool_calls=getattr(complete_ai, "tool_calls", []),
                        )
                    ]

        update["messages"] = [*remove_ops, *trim_ops, *existing_msgs]

        starts = dict(state.get("task_message_starts") or {})
        starts.pop(task_name, None)
        update["task_message_starts"] = starts

        return Command(update=update)

    def _apply_summarization(
        self, result: Command, state: dict, task_name: str, ctx: _PipelineContext
    ) -> Command:
        """Replace task messages with a summary injected into the transition ToolMessage."""
        prep = self._prepare_summarization(state, task_name, ctx)
        if prep is None:
            return result

        task, cfg, task_messages, remove_ops, model = prep

        if cfg.mode == "replace":
            summary = cfg.content
        else:
            invoke_msgs = self._build_summary_messages(task, cfg, task_messages)
            response = model.invoke(invoke_msgs)
            summary = _TaskSteeringBase._extract_response_text(response.content)

        return self._finalize_summarization(
            result, state, task_name, cfg, remove_ops, summary
        )

    async def _aapply_summarization(
        self, result: Command, state: dict, task_name: str, ctx: _PipelineContext
    ) -> Command:
        """Async version of ``_apply_summarization``."""
        prep = self._prepare_summarization(state, task_name, ctx)
        if prep is None:
            return result

        task, cfg, task_messages, remove_ops, model = prep

        if cfg.mode == "replace":
            summary = cfg.content
        else:
            invoke_msgs = self._build_summary_messages(task, cfg, task_messages)
            response = await model.ainvoke(invoke_msgs)
            summary = _TaskSteeringBase._extract_response_text(response.content)

        return self._finalize_summarization(
            result, state, task_name, cfg, remove_ops, summary
        )

    @staticmethod
    def _extract_response_text(content: "str | list") -> str:
        """Extract plain text from a model response content.

        Handles extended-thinking models that return a list of content blocks
        (e.g. ``reasoning_content`` + ``text`` blocks) instead of a plain string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                b["text"]
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(content)

    @staticmethod
    def _flatten_for_summary(task_messages: list) -> list[HumanMessage | AIMessage]:
        """Convert task messages to plain text for the summarization LLM.

        Strips tool_calls / tool_call_id metadata so providers like Bedrock
        don't warn about tool_use blocks without toolConfig.
        """
        flat: list = []
        for m in task_messages:
            content = getattr(m, "content", "")
            if isinstance(content, list):
                text = "\n".join(
                    b["text"]
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)

            if isinstance(m, AIMessage):
                # Include tool call names/args as text
                for tc in getattr(m, "tool_calls", []):
                    text += f"\n[called {tc.get('name', '?')}({tc.get('args', {})})]"
                if text.strip():
                    flat.append(AIMessage(content=text.strip()))
            elif isinstance(m, ToolMessage):
                name = getattr(m, "name", None) or "tool"
                flat.append(HumanMessage(content=f"[{name} result]: {text}"))
            elif text.strip():
                flat.append(HumanMessage(content=text.strip()))
        return flat

    @staticmethod
    def _build_summary_messages(
        task: Task, cfg: TaskSummarization, task_messages: list
    ) -> list:
        """Build the full message list for the summarization LLM call."""
        system = SystemMessage(
            content=(
                "You are summarizing a completed agent task.\n\n"
                f"Task name: {task.name}\n"
                f"Task instruction: {task.instruction}"
            )
        )
        human = HumanMessage(
            content=cfg.prompt
            if cfg.prompt is not None
            else "Provide a concise summary of what was accomplished."
        )
        flat = _TaskSteeringBase._flatten_for_summary(task_messages)
        return [system, *flat, human]

    # ── Shared transition logic ──────────────────────────────

    @staticmethod
    def _execute_task_transition(
        task: str,
        status: str,
        task_order: list[str],
        enforce_order: bool,
        required_tasks: set[str],
        state: dict,
        tool_call_id: str,
        context_label: str = "",
    ) -> Command | str:
        """Validate and execute a task status transition.

        Shared by task-mode and workflow-mode transition tools.
        Returns a ``Command`` on success or an error string on failure.

        Supports three target statuses:

        - ``in_progress`` — start a task.  Blocked by required preceding
          tasks that are not ``complete``/``aborted``.  Optional preceding
          tasks that are still ``pending`` are skipped (so a never-started
          optional task never blocks a required task behind it).
        - ``complete`` — complete an ``in_progress`` task.
        - ``aborted`` — abort an ``in_progress`` *optional* task, provided
          no tool calls have been made for that task yet.  Required tasks
          cannot be aborted by the agent (use :class:`AbortAll` from
          ``on_complete`` for programmatic abort).
        """
        if task not in task_order:
            suffix = f" for {context_label}" if context_label else ""
            return (
                f"Invalid task '{task}'{suffix}. "
                f"Must be one of: {', '.join(task_order)}"
            )

        if status not in (
            TaskStatus.IN_PROGRESS.value,
            TaskStatus.COMPLETE.value,
            TaskStatus.ABORTED.value,
        ):
            return (
                f"Invalid status '{status}'. "
                f"Must be 'in_progress', 'complete', or 'aborted'."
            )

        statuses = dict(state.get("task_statuses") or {})
        for t in task_order:
            statuses.setdefault(t, TaskStatus.PENDING.value)

        current = statuses[task]

        # ── Abort transition ─────────────────────────────────
        if status == TaskStatus.ABORTED.value:
            if task in required_tasks:
                return f"Cannot abort '{task}': task is required. Complete it instead."
            if current == TaskStatus.PENDING.value:
                return (
                    f"Cannot abort '{task}': task hasn't started. "
                    f"Set it to 'in_progress' first to review, or leave it pending."
                )
            if current != TaskStatus.IN_PROGRESS.value:
                return f"Task '{task}' is already {current}."

            # Commitment check: any ToolMessage since the task went in_progress?
            starts = state.get("task_message_starts") or {}
            start_index = starts.get(task)
            if start_index is not None:
                messages = state.get("messages", [])
                for msg in messages[start_index:]:
                    if isinstance(msg, ToolMessage):
                        return (
                            f"Cannot abort '{task}': tools already executed — "
                            f"task has made state changes. Complete it instead."
                        )

            statuses[task] = TaskStatus.ABORTED.value

            starts = dict(state.get("task_message_starts") or {})
            starts.pop(task, None)

            display = "\n".join(f"  {k}: {v}" for k, v in statuses.items())
            return Command(
                update={
                    "task_statuses": statuses,
                    "task_message_starts": starts,
                    "nudge_count": 0,
                    "messages": [
                        ToolMessage(
                            f"Task '{task}' -> aborted.\n\n{display}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        # ── Pending → in_progress → complete ─────────────────
        if current in _TERMINAL_STATUSES:
            return f"Task '{task}' is already {current}."

        valid_next = {
            TaskStatus.PENDING.value: TaskStatus.IN_PROGRESS.value,
            TaskStatus.IN_PROGRESS.value: TaskStatus.COMPLETE.value,
        }
        expected = valid_next.get(current)
        if expected != status:
            return (
                f"Cannot transition '{task}' from '{current}' to "
                f"'{status}'. Expected next: '{expected}'."
            )

        if enforce_order and status == TaskStatus.IN_PROGRESS.value:
            idx = task_order.index(task)
            for prev in task_order[:idx]:
                # Optional, never-started tasks don't block (dead-end fix).
                if (
                    prev not in required_tasks
                    and statuses[prev] == TaskStatus.PENDING.value
                ):
                    continue
                if statuses[prev] not in _TERMINAL_STATUSES:
                    return (
                        f"Cannot start '{task}': '{prev}' is not "
                        f"complete yet. "
                        f"Order: {' -> '.join(task_order)}."
                    )

        statuses[task] = status

        display = "\n".join(f"  {k}: {v}" for k, v in statuses.items())
        return Command(
            update={
                "task_statuses": statuses,
                "nudge_count": 0,
                "messages": [
                    ToolMessage(
                        f"Task '{task}' -> {status}.\n\n{display}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # ── Shared wrap-style hooks ──────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject task pipeline prompt and scope tools per active task."""
        ctx = self._get_pipeline_ctx(request.state)
        if ctx is None:
            return self._on_no_pipeline_model_call(request, handler)

        modified, active_name = self._prepare_model_request(request, ctx)

        task_mw = self._get_task_middleware(ctx, active_name)
        if task_mw and self._overrides(task_mw, "wrap_model_call"):
            return task_mw.wrap_model_call(modified, handler)

        return handler(modified)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Gate completion transitions, enforce tool scoping, and delegate."""
        tool_name = request.tool_call["name"]

        # Management tools always pass straight through to handler
        if tool_name in self._management_tool_names:
            return handler(request)

        ctx = self._get_pipeline_ctx(request.state)
        if ctx is None:
            return self._on_no_pipeline_tool_call(request, handler)

        statuses = self._get_statuses(ctx, request.state)
        active_name = self._active_task(ctx, statuses)

        if tool_name == _TRANSITION_TOOL_NAME:
            rejection = self._validate_transition(request, statuses, ctx)
            if rejection:
                return rejection

            result = handler(request)
            return self._fire_lifecycle_hooks(request, result, statuses, ctx)

        gate = self._gate_tool(request, active_name, ctx, request.state)
        if gate:
            return gate

        task_mw = self._get_task_middleware(ctx, active_name)
        if task_mw and self._overrides(task_mw, "wrap_tool_call"):
            return task_mw.wrap_tool_call(request, handler)

        return handler(request)

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Nudge the agent back to the model if required tasks are incomplete."""
        ctx = self._get_pipeline_ctx(state)
        if ctx is None:
            return None

        if not ctx.required_tasks:
            return None

        statuses = self._get_statuses(ctx, state)
        incomplete = [
            name
            for name in ctx.task_order
            if name in ctx.required_tasks
            and statuses.get(name) not in _TERMINAL_STATUSES
        ]

        if not incomplete:
            return None

        nudge_count = state.get("nudge_count", 0)
        if nudge_count >= self._max_nudges:
            return None

        nudge_msg = self._build_nudge_message(incomplete, state)

        return {
            "jump_to": "model",
            "nudge_count": nudge_count + 1,
            "messages": [nudge_msg],
        }

    # ── Async wrap-style hooks ──────────────────────────────

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        ctx = self._get_pipeline_ctx(request.state)
        if ctx is None:
            return await self._aon_no_pipeline_model_call(request, handler)

        modified, active_name = self._prepare_model_request(request, ctx)

        task_mw = self._get_task_middleware(ctx, active_name)
        if task_mw and self._overrides(task_mw, "awrap_model_call"):
            return await task_mw.awrap_model_call(modified, handler)

        return await handler(modified)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call."""
        tool_name = request.tool_call["name"]

        # Management tools always pass straight through to handler
        if tool_name in self._management_tool_names:
            return await handler(request)

        ctx = self._get_pipeline_ctx(request.state)
        if ctx is None:
            return await self._aon_no_pipeline_tool_call(request, handler)

        statuses = self._get_statuses(ctx, request.state)
        active_name = self._active_task(ctx, statuses)

        if tool_name == _TRANSITION_TOOL_NAME:
            rejection = await self._avalidate_transition(request, statuses, ctx)
            if rejection:
                return rejection

            result = await handler(request)
            return await self._afire_lifecycle_hooks(request, result, statuses, ctx)

        gate = self._gate_tool(request, active_name, ctx, request.state)
        if gate:
            return gate

        task_mw = self._get_task_middleware(ctx, active_name)
        if task_mw and self._overrides(task_mw, "awrap_tool_call"):
            return await task_mw.awrap_tool_call(request, handler)

        return await handler(request)

    # ── Async node-style hooks ──────────────────────────────

    async def abefore_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Async version of before_agent."""
        return self.before_agent(state, runtime)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Async version of after_agent."""
        return self.after_agent(state, runtime)


# ════════════════════════════════════════════════════════════════
# TaskSteeringMiddleware — task mode (direct tasks= param)
# ════════════════════════════════════════════════════════════════


class TaskSteeringMiddleware(_TaskSteeringBase):
    """Implicit state machine middleware for ordered task execution.

    Provides:
    - Ordered task pipeline with enforced transitions
      (pending -> in_progress -> complete).
    - Per-task tool scoping — only the active task's tools are visible
      to the model.
    - Dynamic system prompt injection — task status board and active
      task instruction appended before every model call.
    - Completion validation via task-scoped middleware
      (``TaskMiddleware.validate_completion``).
    - Mid-task enforcement via task-scoped middleware hooks
      (``wrap_tool_call``, ``wrap_model_call``).
    - Per-task skill scoping — skills from ``SKILL.md`` files are
      filtered to the active task (requires a backend).
    - Backend tools passthrough — known backend tools can be
      whitelisted to pass through the tool filter.

    Args:
        tasks: Ordered list of :class:`Task` definitions.
        global_tools: Tools available regardless of which task is active.
        enforce_order: If ``True`` (default), tasks must be completed in
            the order they are defined.
        backend_tools_passthrough: If ``True``, known backend tools
            pass through the tool filter on all tasks.
        backend_tools: Override the default backend tools whitelist.
            ``None`` uses :attr:`DEFAULT_BACKEND_TOOLS`.
        global_skills: Skill names available regardless of active task.
        model: Default chat model for ``TaskSummarization(mode="summarize")``.
            Used when a task's ``summarize.model`` is ``None``.  Typically
            the same model passed to ``create_agent``.
    """

    def __init__(
        self,
        tasks: list[Task],
        *,
        global_tools: list | None = None,
        enforce_order: bool = True,
        required_tasks: list[str] | tuple[str, ...] | None = _REQUIRE_ALL,
        max_nudges: int = 3,
        backend_tools_passthrough: bool = False,
        backend_tools: set[str] | None = None,
        global_skills: list[str] | None = None,
        model: Any = None,
    ) -> None:
        super().__init__(
            max_nudges=max_nudges,
            backend_tools_passthrough=backend_tools_passthrough,
            backend_tools=backend_tools,
            model=model,
        )

        if not tasks:
            raise ValueError("At least one Task is required.")

        tasks = _validate_and_normalize_tasks(tasks)
        all_names = {t.name for t in tasks}

        self._ctx = _build_pipeline_context(
            tasks=tasks,
            global_tools=global_tools or [],
            enforce_order=enforce_order,
            required_tasks=_resolve_required_tasks(required_tasks, all_names),
            global_skills=global_skills,
        )

        self._transition_tool = self._build_transition_tool()
        self.state_schema = self._merge_state_schemas(tasks)

        self.tools = _dedup_tools(
            [
                self._transition_tool,
                *self._ctx.global_tools,
                *(tool_obj for task in tasks for tool_obj in task.tools),
                *(
                    tool_obj
                    for task in tasks
                    if task.middleware and hasattr(task.middleware, "tools")
                    for tool_obj in (task.middleware.tools or [])
                ),
            ]
        )

    # ── Pipeline context ────────────────────────────────────

    def _get_pipeline_ctx(self, state: dict) -> _PipelineContext | None:
        return self._ctx

    # ── Node-style hooks ────────────────────────────────────

    def before_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Initialize task_statuses on first invocation."""
        updates: dict[str, Any] = {}

        if state.get("task_statuses") is None:
            updates["task_statuses"] = {
                t.name: TaskStatus.PENDING.value for t in self._ctx.tasks
            }

        return updates or None

    # ── Tool builder ────────────────────────────────────────

    def _build_transition_tool(self):
        """Build the update_task_status tool with closure over pipeline config."""
        task_order = self._ctx.task_order
        enforce_order = self._ctx.enforce_order
        required_tasks = self._ctx.required_tasks
        task_names_hint = ", ".join(f"'{n}'" for n in task_order)
        execute = _TaskSteeringBase._execute_task_transition

        @tool(
            name_or_callable=_TRANSITION_TOOL_NAME,
            description=(
                "Transition a task to 'in_progress', 'complete', or 'aborted'. "
                "Must be called ALONE — never in parallel with other tools. "
                "Tasks must follow the defined order. "
                "'aborted' is only valid for optional tasks that are in_progress "
                "and have made no tool calls yet."
            ),
        )
        def update_task_status(
            task: Annotated[str, f"Task name: {task_names_hint}"],
            status: Annotated[
                str, "New status: 'in_progress', 'complete', or 'aborted'"
            ],
            runtime: ToolRuntime,
        ) -> Command | str:
            """Set task status with enforced transition and ordering rules."""
            return execute(
                task,
                status,
                task_order,
                enforce_order,
                required_tasks,
                runtime.state,
                runtime.tool_call_id,
            )

        return update_task_status


# ════════════════════════════════════════════════════════════════
# WorkflowSteeringMiddleware — workflow mode (dynamic workflows)
# ════════════════════════════════════════════════════════════════


class WorkflowSteeringMiddleware(_TaskSteeringBase):
    """Workflow-mode middleware for dynamic pipeline activation/deactivation.

    The agent sees a catalog of available workflows and activates one on
    demand via the ``activate_workflow`` tool.  When no workflow is active
    the middleware is transparent (no tool filtering or prompt injection).

    Args:
        workflows: List of :class:`Workflow` definitions.
        max_nudges: Maximum number of nudge messages before giving up.
        backend_tools_passthrough: If ``True``, known backend tools
            pass through the tool filter on all tasks.
        backend_tools: Override the default backend tools whitelist.
            ``None`` uses :attr:`DEFAULT_BACKEND_TOOLS`.
        model: Default chat model for ``TaskSummarization(mode="summarize")``.
    """

    _management_tool_names = {_ACTIVATE_TOOL_NAME, _DEACTIVATE_TOOL_NAME}

    def __init__(
        self,
        workflows: list[Workflow],
        *,
        max_nudges: int = 3,
        backend_tools_passthrough: bool = False,
        backend_tools: set[str] | None = None,
        model: Any = None,
    ) -> None:
        super().__init__(
            max_nudges=max_nudges,
            backend_tools_passthrough=backend_tools_passthrough,
            backend_tools=backend_tools,
            model=model,
        )

        if not workflows:
            raise ValueError("At least one Workflow is required.")

        wf_names = [w.name for w in workflows]
        dupes = _find_dupes(wf_names)
        if dupes:
            raise ValueError(f"Duplicate workflow names: {dupes}")

        self._workflows: list[Workflow] = []
        for wf in workflows:
            if not wf.tasks:
                raise ValueError(
                    f"Workflow '{wf.name}' has no tasks. "
                    f"Each workflow must have at least one Task."
                )
            wf = copy.copy(wf)
            wf.tasks = _validate_and_normalize_tasks(wf.tasks)
            self._workflows.append(wf)

        self._workflow_map: dict[str, Workflow] = {w.name: w for w in self._workflows}

        self._workflow_ctxs: dict[str, _PipelineContext] = {}
        for wf in self._workflows:
            all_names = {t.name for t in wf.tasks}
            self._workflow_ctxs[wf.name] = _build_pipeline_context(
                tasks=wf.tasks,
                global_tools=wf.global_tools,
                enforce_order=wf.enforce_order,
                required_tasks=_resolve_required_tasks(
                    wf.required_tasks,
                    all_names,
                    context_label=f"workflow '{wf.name}'",
                ),
                global_skills=wf.global_skills,
                label=wf.name,
            )

        self._activate_tool = self._build_activate_tool()
        self._deactivate_tool = self._build_deactivate_tool()
        self._workflow_transition_tool = self._build_workflow_transition_tool()

        all_wf_tasks = [t for wf in self._workflows for t in wf.tasks]
        self.state_schema = self._merge_state_schemas(all_wf_tasks)

        self.tools = _dedup_tools(
            [
                self._activate_tool,
                self._deactivate_tool,
                self._workflow_transition_tool,
                *(tool_obj for wf in self._workflows for tool_obj in wf.global_tools),
                *(
                    tool_obj
                    for wf in self._workflows
                    for task in wf.tasks
                    for tool_obj in task.tools
                ),
                *(
                    tool_obj
                    for wf in self._workflows
                    for task in wf.tasks
                    if task.middleware and hasattr(task.middleware, "tools")
                    for tool_obj in (task.middleware.tools or [])
                ),
            ]
        )

        # Cache invariants for hot-path use.
        self._workflow_tool_names: frozenset[str] = frozenset(
            t.name for t in self.tools
        )
        self._catalog_text: str = self._render_catalog()

    # ── Pipeline context ────────────────────────────────────

    def _get_pipeline_ctx(self, state: dict) -> _PipelineContext | None:
        wf_name = state.get("active_workflow")
        if wf_name is None:
            return None
        return self._workflow_ctxs.get(wf_name)

    _extra_allowed: frozenset[str] = frozenset({_DEACTIVATE_TOOL_NAME})

    def _extra_allowed_tool_names(self) -> frozenset[str]:
        return self._extra_allowed

    # ── No-pipeline overrides (catalog mode) ────────────────

    def _build_catalog_request(self, request: ModelRequest) -> ModelRequest:
        """Build a model request with catalog prompt and transparent tool pass-through."""
        existing = (
            list(request.system_message.content_blocks)
            if request.system_message is not None
            else []
        )

        new_content = existing + [{"type": "text", "text": self._catalog_text}]

        allowed = {_ACTIVATE_TOOL_NAME}
        scoped = [t for t in request.tools if t.name in allowed]
        for t in request.tools:
            if t.name not in self._workflow_tool_names and t.name not in allowed:
                scoped.append(t)
                allowed.add(t.name)

        return request.override(
            system_message=SystemMessage(content=new_content),
            tools=scoped,
        )

    def _on_no_pipeline_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._build_catalog_request(request))

    async def _aon_no_pipeline_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._build_catalog_request(request))

    # ── Nudge message override ──────────────────────────────

    def _build_nudge_message(self, incomplete: list[str], state: dict) -> HumanMessage:
        """Workflow mode nudge includes the workflow name."""
        wf_name = state.get("active_workflow", "unknown")
        task_list = ", ".join(incomplete)
        return HumanMessage(
            content=(
                f"You have not completed the following required tasks "
                f"in workflow '{wf_name}': {task_list}. Please continue."
            ),
            additional_kwargs={
                "task_steering": {"kind": "nudge", "incomplete_tasks": incomplete},
            },
        )

    # ── Node-style hooks ────────────────────────────────────

    def before_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Workflow mode: nothing to do here (activate tool inits state)."""
        return None

    # ── Catalog rendering ───────────────────────────────────

    def _render_catalog(self) -> str:
        """Render the workflow catalog for when no workflow is active."""
        lines = ["\n<available_workflows>"]
        for wf in self._workflows:
            task_names = ", ".join(t.name for t in wf.tasks)
            lines.append(f'  <workflow name="{wf.name}">')
            lines.append(f"    {wf.description}")
            lines.append(f"    Tasks: {task_names}")
            lines.append("  </workflow>")
        lines.append("")
        lines.append("  Use activate_workflow to start a workflow when needed.")
        lines.append("</available_workflows>")
        return "\n".join(lines)

    # ── Workflow tool builders ──────────────────────────────

    def _build_activate_tool(self):
        """Build the activate_workflow tool for workflow mode."""
        workflow_map = self._workflow_map

        wf_names_hint = ", ".join(f"'{w}'" for w in workflow_map)

        @tool(
            name_or_callable=_ACTIVATE_TOOL_NAME,
            description=(
                "Activate a workflow to start working on a structured task pipeline. "
                "Only one workflow can be active at a time."
            ),
        )
        def activate_workflow(
            workflow: Annotated[str, f"Workflow name: {wf_names_hint}"],
            runtime: ToolRuntime,
        ) -> Command | str:
            """Activate a workflow by name."""
            if workflow not in workflow_map:
                return (
                    f"Unknown workflow '{workflow}'. "
                    f"Available: {', '.join(workflow_map)}"
                )

            current = runtime.state.get("active_workflow")
            if current is not None:
                return (
                    f"Workflow '{current}' is already active. "
                    f"Deactivate it first with deactivate_workflow."
                )

            wf = workflow_map[workflow]
            statuses = {t.name: TaskStatus.PENDING.value for t in wf.tasks}

            return Command(
                update={
                    "active_workflow": workflow,
                    "task_statuses": statuses,
                    "nudge_count": 0,
                    "messages": [
                        ToolMessage(
                            f"Workflow '{workflow}' activated.\n\n"
                            + "\n".join(f"  {k}: {v}" for k, v in statuses.items()),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        return activate_workflow

    def _build_deactivate_tool(self):
        """Build the deactivate_workflow tool for workflow mode."""
        workflow_map = self._workflow_map

        @tool(
            name_or_callable=_DEACTIVATE_TOOL_NAME,
            description=(
                "Deactivate the current workflow, clearing all task state. "
                "May be blocked if a task is in progress."
            ),
        )
        def deactivate_workflow(
            runtime: ToolRuntime,
        ) -> Command | str:
            """Deactivate the current workflow."""
            current = runtime.state.get("active_workflow")
            if current is None:
                return "No workflow is currently active."

            wf = workflow_map.get(current)

            # Check deactivation policy
            if wf and not wf.allow_deactivate_in_progress:
                statuses = runtime.state.get("task_statuses") or {}
                active = [
                    name
                    for name, s in statuses.items()
                    if s == TaskStatus.IN_PROGRESS.value
                ]
                if active:
                    return (
                        f"Cannot deactivate: task '{active[0]}' is in progress. "
                        f"Complete or skip it first."
                    )

            return Command(
                update={
                    "active_workflow": None,
                    "task_statuses": {},
                    "task_message_starts": {},
                    "nudge_count": 0,
                    "messages": [
                        ToolMessage(
                            f"Workflow '{current}' deactivated.",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        return deactivate_workflow

    def _build_workflow_transition_tool(self):
        """Build update_task_status for workflow mode.

        Reads ``active_workflow`` from state to determine task order
        and ``enforce_order`` dynamically, then delegates to the shared
        ``_execute_task_transition`` logic.
        """
        workflow_map = self._workflow_map
        workflow_ctxs = self._workflow_ctxs
        execute = _TaskSteeringBase._execute_task_transition

        @tool(
            name_or_callable=_TRANSITION_TOOL_NAME,
            description=(
                "Transition a task to 'in_progress', 'complete', or 'aborted'. "
                "Must be called ALONE — never in parallel with other tools. "
                "Tasks must follow the defined order within the active workflow. "
                "'aborted' is only valid for optional tasks that are in_progress "
                "and have made no tool calls yet."
            ),
        )
        def update_task_status(
            task: Annotated[str, "Task name within the active workflow"],
            status: Annotated[
                str, "New status: 'in_progress', 'complete', or 'aborted'"
            ],
            runtime: ToolRuntime,
        ) -> Command | str:
            """Set task status within the active workflow."""
            wf_name = runtime.state.get("active_workflow")
            if wf_name is None:
                return "No workflow is active. Activate a workflow first."

            wf = workflow_map.get(wf_name)
            ctx = workflow_ctxs.get(wf_name)
            if wf is None or ctx is None:
                return f"Active workflow '{wf_name}' not found."

            return execute(
                task,
                status,
                ctx.task_order,
                wf.enforce_order,
                ctx.required_tasks,
                runtime.state,
                runtime.tool_call_id,
                context_label=f"workflow '{wf_name}'",
            )

        return update_task_status
