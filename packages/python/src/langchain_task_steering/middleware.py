"""TaskSteeringMiddleware — implicit state machine for LangChain v1 agents."""

import copy
import typing
import warnings
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware import hook_config
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict

from ._hooks import WRAP_HOOK_PAIRS, overrides_base
from .types import Task, TaskMiddleware, TaskStatus, TaskSteeringState

_TRANSITION_TOOL_NAME = "update_task_status"
_REQUIRE_ALL = ("*",)


def _overrides_task(middleware: TaskMiddleware, method_name: str) -> bool:
    """Check if a TaskMiddleware subclass overrides a TaskMiddleware hook."""
    return getattr(type(middleware), method_name) is not getattr(
        TaskMiddleware, method_name, None
    )


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
        for mw in self._middlewares:
            if _overrides_task(mw, "on_start"):
                mw.on_start(state)

    def on_complete(self, state):
        for mw in self._middlewares:
            if _overrides_task(mw, "on_complete"):
                mw.on_complete(state)

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
        for mw in self._middlewares:
            if _overrides_task(mw, "aon_start") or _overrides_task(mw, "on_start"):
                await mw.aon_start(state)

    async def aon_complete(self, state):
        for mw in self._middlewares:
            if _overrides_task(mw, "aon_complete") or _overrides_task(
                mw, "on_complete"
            ):
                await mw.aon_complete(state)


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


class TaskSteeringMiddleware(AgentMiddleware[TaskSteeringState]):
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

    def __init__(
        self,
        tasks: list[Task],
        global_tools: list | None = None,
        enforce_order: bool = True,
        required_tasks: list[str] | None = _REQUIRE_ALL,
        max_nudges: int = 3,
        backend_tools_passthrough: bool = False,
        backend_tools: set[str] | None = None,
        global_skills: list[str] | None = None,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("At least one Task is required.")

        names = [t.name for t in tasks]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate task names: {dupes}")

        # Normalize middleware on shallow copies to avoid mutating the caller's objects
        tasks = [copy.copy(t) for t in tasks]
        for task in tasks:
            task.middleware = _normalize_middleware(task.middleware)
            task.tools = list(task.tools)

        self._tasks = tasks
        self._task_order: list[str] = [t.name for t in tasks]
        self._task_map: dict[str, Task] = {t.name: t for t in tasks}
        self._global_tools: list = global_tools or []
        self._enforce_order = enforce_order
        self._max_nudges = max_nudges

        # ── Backend tools passthrough ────────────────────────────
        self._backend_tools = (
            frozenset(backend_tools)
            if backend_tools is not None
            else self.DEFAULT_BACKEND_TOOLS
        )
        self._backend_tools_passthrough = backend_tools_passthrough

        # ── Task-scoped skills ───────────────────────────────────
        # Skills are active if any task defines skills or global_skills are set.
        # Skill metadata comes from state (e.g. loaded by SkillsMiddleware in
        # create_deep_agent). This middleware only filters, never loads.
        self._global_skills: list[str] = list(global_skills or [])
        self._skills_active = bool(any(t.skills for t in tasks) or self._global_skills)
        self._skill_required_tools: frozenset[str] = (
            frozenset({"read_file", "ls"}) if self._skills_active else frozenset()
        )

        # Resolve required_tasks
        if required_tasks is _REQUIRE_ALL:
            self._required_tasks: set[str] = {t.name for t in tasks}
        elif required_tasks is not None and "*" in required_tasks:
            self._required_tasks = {t.name for t in tasks}
        elif required_tasks is not None:
            unknown = set(required_tasks) - set(names)
            if unknown:
                raise ValueError(f"Unknown required tasks: {unknown}")
            self._required_tasks = set(required_tasks)
        else:
            self._required_tasks = set()

        self._transition_tool = self._build_transition_tool()

        # Merge state schemas from task middlewares so their fields
        # persist in the agent's state graph and survive interrupts.
        self.state_schema = self._merge_state_schemas()

        # Auto-register all tools (transition + global + every task's tools
        # + tools contributed by task middleware adapters).
        # The agent receives these via the middleware ``tools`` attribute so
        # users don't have to duplicate them in ``create_agent(tools=...)``.
        seen: set[str] = set()
        all_tools: list = []
        for t in [
            self._transition_tool,
            *self._global_tools,
            *(tool for task in self._tasks for tool in task.tools),
            *(
                tool
                for task in self._tasks
                if task.middleware and hasattr(task.middleware, "tools")
                for tool in (task.middleware.tools or [])
            ),
        ]:
            if t.name not in seen:
                seen.add(t.name)
                all_tools.append(t)
        self.tools = all_tools

    # ── Node-style hooks ──────────────────────────────────────

    def before_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Initialize task_statuses and load skills on first invocation."""
        updates: dict[str, Any] = {}

        if state.get("task_statuses") is None:
            updates["task_statuses"] = {
                t.name: TaskStatus.PENDING.value for t in self._tasks
            }

        return updates or None

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Nudge the agent back to the model if required tasks are incomplete."""
        if not self._required_tasks:
            return None

        statuses = self._get_statuses(state)
        incomplete = [
            name
            for name in self._task_order
            if name in self._required_tasks
            and statuses[name] != TaskStatus.COMPLETE.value
        ]

        if not incomplete:
            return None

        nudge_count = state.get("nudge_count", 0)
        if nudge_count >= self._max_nudges:
            return None

        task_list = ", ".join(incomplete)
        nudge_msg = HumanMessage(
            content=(
                f"You have not completed the following required tasks: "
                f"{task_list}. Please continue."
            ),
            additional_kwargs={
                "task_steering": {"kind": "nudge", "incomplete_tasks": incomplete},
            },
        )

        return {
            "jump_to": "model",
            "nudge_count": nudge_count + 1,
            "messages": [nudge_msg],
        }

    # ── Async node-style hooks ───────────────────────────────

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

    # ── Wrap-style hooks ──────────────────────────────────────

    def _prepare_model_request(
        self, request: ModelRequest
    ) -> tuple[ModelRequest, str | None]:
        """Build the modified model request with pipeline prompt and scoped tools."""
        statuses = self._get_statuses(request.state)
        active_name = self._active_task(statuses)

        block = self._render_status_block(statuses, active_name, state=request.state)
        existing = (
            list(request.system_message.content_blocks)
            if request.system_message is not None
            else []
        )

        # Strip SkillsMiddleware's global prompt injection — we replace it
        # with per-task scoped skills in the pipeline block.
        if self._skills_active:
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

        allowed_names = self._allowed_tool_names(active_name)
        scoped = [t for t in request.tools if t.name in allowed_names]

        modified = request.override(
            system_message=SystemMessage(content=new_content),
            tools=scoped,
        )
        return modified, active_name

    def _validate_transition(
        self, request: ToolCallRequest, statuses: dict[str, str]
    ) -> ToolMessage | None:
        """Pre-handler validation for transitions. Returns a rejection ToolMessage or None."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if target == TaskStatus.IN_PROGRESS.value:
            already_active = self._active_task(statuses)
            if already_active:
                return ToolMessage(
                    content=(
                        f"Cannot start '{task_name}': '{already_active}' is already "
                        f"in progress. Complete it first."
                    ),
                    tool_call_id=request.tool_call["id"],
                )

        if target == TaskStatus.COMPLETE.value and task_name in self._task_map:
            task_mw = self._get_task_middleware(task_name)
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

    def _fire_lifecycle_hooks(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
        statuses: dict[str, str],
    ) -> None:
        """Fire on_start/on_complete if the transition succeeded."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if not isinstance(result, Command) or task_name not in self._task_map:
            return

        task_mw = self._get_task_middleware(task_name)
        if not task_mw:
            return

        # Build post-transition state so hooks see the updated task_statuses
        # (the Command hasn't been applied yet).
        updated_statuses = {**statuses, task_name: target}
        post_state = {**request.state, "task_statuses": updated_statuses}

        if target == TaskStatus.IN_PROGRESS.value:
            task_mw.on_start(post_state)
        elif target == TaskStatus.COMPLETE.value:
            task_mw.on_complete(post_state)

    async def _avalidate_transition(
        self, request: ToolCallRequest, statuses: dict[str, str]
    ) -> ToolMessage | None:
        """Async pre-handler validation for transitions."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if target == TaskStatus.IN_PROGRESS.value:
            already_active = self._active_task(statuses)
            if already_active:
                return ToolMessage(
                    content=(
                        f"Cannot start '{task_name}': '{already_active}' is already "
                        f"in progress. Complete it first."
                    ),
                    tool_call_id=request.tool_call["id"],
                )

        if target == TaskStatus.COMPLETE.value and task_name in self._task_map:
            task_mw = self._get_task_middleware(task_name)
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

    async def _afire_lifecycle_hooks(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
        statuses: dict[str, str],
    ) -> None:
        """Async version of _fire_lifecycle_hooks."""
        args = request.tool_call["args"]
        task_name = args.get("task")
        target = args.get("status")

        if not isinstance(result, Command) or task_name not in self._task_map:
            return

        task_mw = self._get_task_middleware(task_name)
        if not task_mw:
            return

        updated_statuses = {**statuses, task_name: target}
        post_state = {**request.state, "task_statuses": updated_statuses}

        if target == TaskStatus.IN_PROGRESS.value:
            await task_mw.aon_start(post_state)
        elif target == TaskStatus.COMPLETE.value:
            await task_mw.aon_complete(post_state)

    def _gate_tool(
        self, request: ToolCallRequest, active_name: str | None
    ) -> ToolMessage | None:
        """Reject tool calls not in scope for the active task."""
        allowed = self._allowed_tool_names(active_name)
        if request.tool_call["name"] not in allowed:
            return ToolMessage(
                content=(
                    f"Tool '{request.tool_call['name']}' is not available "
                    f"for the current task."
                ),
                tool_call_id=request.tool_call["id"],
            )
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject task pipeline prompt and scope tools per active task."""
        modified, active_name = self._prepare_model_request(request)

        task_mw = self._get_task_middleware(active_name)
        if task_mw and self._overrides(task_mw, "wrap_model_call"):
            return task_mw.wrap_model_call(modified, handler)

        return handler(modified)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Gate completion transitions, enforce tool scoping, and delegate."""
        statuses = self._get_statuses(request.state)
        active_name = self._active_task(statuses)

        if request.tool_call["name"] == _TRANSITION_TOOL_NAME:
            rejection = self._validate_transition(request, statuses)
            if rejection:
                return rejection

            result = handler(request)
            self._fire_lifecycle_hooks(request, result, statuses)
            return result

        gate = self._gate_tool(request, active_name)
        if gate:
            return gate

        task_mw = self._get_task_middleware(active_name)
        if task_mw and self._overrides(task_mw, "wrap_tool_call"):
            return task_mw.wrap_tool_call(request, handler)

        return handler(request)

    # ── Async wrap-style hooks ───────────────────────────────

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        modified, active_name = self._prepare_model_request(request)

        task_mw = self._get_task_middleware(active_name)
        if task_mw and self._overrides(task_mw, "awrap_model_call"):
            return await task_mw.awrap_model_call(modified, handler)

        return await handler(modified)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call."""
        statuses = self._get_statuses(request.state)
        active_name = self._active_task(statuses)

        if request.tool_call["name"] == _TRANSITION_TOOL_NAME:
            rejection = await self._avalidate_transition(request, statuses)
            if rejection:
                return rejection

            result = await handler(request)
            await self._afire_lifecycle_hooks(request, result, statuses)
            return result

        gate = self._gate_tool(request, active_name)
        if gate:
            return gate

        task_mw = self._get_task_middleware(active_name)
        if task_mw and self._overrides(task_mw, "awrap_tool_call"):
            return await task_mw.awrap_tool_call(request, handler)

        return await handler(request)

    # ── Internal helpers ──────────────────────────────────────

    def _merge_state_schemas(self) -> type:
        """Merge ``TaskSteeringState`` with state schemas from task middlewares.

        Task-scoped middlewares are composed internally — they aren't
        registered directly with ``create_agent``, so the agent won't
        see their ``state_schema``.  This method collects every task
        middleware's schema, merges the type hints (preserving
        ``Annotated`` reducers and ``NotRequired`` markers), and
        produces a single ``TypedDict`` that ``TaskSteeringMiddleware``
        exposes as its own ``state_schema``.
        """
        # Collect schemas that actually add fields beyond the base.
        # AgentMiddleware.state_schema defaults to _DefaultAgentState
        # (not None), so we compare field sets to detect real extensions.
        base_fields = set(typing.get_type_hints(TaskSteeringState, include_extras=True))
        task_schemas: list[type] = []
        for task in self._tasks:
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

        # Collect all type hints, preserving Annotated/NotRequired wrappers.
        # Later schemas override earlier ones on key conflict (last wins).
        merged: dict[str, Any] = {}
        for schema in [TaskSteeringState, *task_schemas]:
            merged.update(typing.get_type_hints(schema, include_extras=True))

        return TypedDict("TaskSteeringState", merged)  # type: ignore[call-overload]

    def _get_statuses(self, state: dict) -> dict[str, str]:
        raw = state.get("task_statuses") or {}
        return {t.name: raw.get(t.name, TaskStatus.PENDING.value) for t in self._tasks}

    def _active_task(self, statuses: dict[str, str]) -> str | None:
        for name in self._task_order:
            if statuses[name] == TaskStatus.IN_PROGRESS.value:
                return name
        return None

    @staticmethod
    def _overrides(middleware: AgentMiddleware, method_name: str) -> bool:
        """Check if a middleware subclass actually overrides a hook method."""
        return overrides_base(middleware, method_name)

    def _get_task_middleware(self, task_name: str | None) -> TaskMiddleware | None:
        if task_name is None:
            return None
        task = self._task_map.get(task_name)
        return task.middleware if task else None

    def _allowed_tool_names(self, active_name: str | None) -> set[str]:
        names = {_TRANSITION_TOOL_NAME}
        names.update(t.name for t in self._global_tools)
        if active_name:
            task = self._task_map[active_name]
            names.update(t.name for t in task.tools)
            # Include tools contributed by task middleware (e.g. adapters)
            if task.middleware and hasattr(task.middleware, "tools"):
                names.update(t.name for t in (task.middleware.tools or []))
        if self._backend_tools_passthrough:
            names.update(self._backend_tools)
        if self._skills_active:
            names.update(self._skill_required_tools)
        return names

    def _allowed_skill_names(self, active_name: str | None) -> set[str]:
        """Return skill names visible for the given active task."""
        names = set(self._global_skills)
        if active_name:
            task = self._task_map[active_name]
            if task.skills:
                names.update(task.skills)
        return names

    def get_backend_tools(self) -> frozenset[str]:
        """Return the effective backend tools whitelist."""
        return self._backend_tools

    def _render_status_block(
        self,
        statuses: dict[str, str],
        active: str | None,
        state: dict | None = None,
    ) -> str:
        icons = {
            TaskStatus.PENDING.value: "[ ]",
            TaskStatus.IN_PROGRESS.value: "[>]",
            TaskStatus.COMPLETE.value: "[x]",
        }

        lines = ["\n<task_pipeline>"]
        for t in self._tasks:
            s = statuses[t.name]
            lines.append(f"  {icons.get(s, '[?]')} {t.name} ({s})")

        if active:
            lines.append(f'\n  <current_task name="{active}">')
            lines.append(f"    {self._task_map[active].instruction}")
            lines.append("  </current_task>")

        # ── Skill rendering ──────────────────────────────────────
        has_visible_skills = False
        if self._skills_active and state is not None:
            all_skills = state.get("skills_metadata") or []
            allowed_names = self._allowed_skill_names(active)
            visible_skills = [s for s in all_skills if s["name"] in allowed_names]
            if visible_skills:
                has_visible_skills = True
                lines.append("\n  <available_skills>")
                for skill in visible_skills:
                    desc = skill.get("description", "No description.")
                    lines.append(f"    - {skill['name']}: {desc} Path: {skill['path']}")
                lines.append("  </available_skills>")

        if self._enforce_order or has_visible_skills:
            lines.append("\n  <rules>")
            if self._enforce_order:
                order_str = " -> ".join(self._task_order)
                lines.append(f"    Required order: {order_str}")
            lines.append("    Use update_task_status to advance. Do not skip tasks.")
            if has_visible_skills:
                lines.append(
                    "    To use a skill, read its SKILL.md file for full instructions."
                )
            lines.append("  </rules>")

        lines.append("</task_pipeline>")
        return "\n".join(lines)

    def _build_transition_tool(self):
        """Build the update_task_status tool with closure over pipeline config."""
        task_order = self._task_order
        enforce_order = self._enforce_order
        task_names_hint = ", ".join(f"'{n}'" for n in task_order)

        @tool(
            name_or_callable=_TRANSITION_TOOL_NAME,
            description=(
                "Transition a task to 'in_progress' or 'complete'. "
                "Must be called ALONE — never in parallel with other tools. "
                "Tasks must follow the defined order."
            ),
        )
        def update_task_status(
            task: Annotated[str, f"Task name: {task_names_hint}"],
            status: Annotated[str, "New status: 'in_progress' or 'complete'"],
            runtime: ToolRuntime,
        ) -> Command | str:
            """Set task status with enforced transition and ordering rules."""
            if task not in task_order:
                return f"Invalid task '{task}'. Must be one of: {', '.join(task_order)}"

            if status not in (
                TaskStatus.IN_PROGRESS.value,
                TaskStatus.COMPLETE.value,
            ):
                return (
                    f"Invalid status '{status}'. Must be 'in_progress' or 'complete'."
                )

            statuses = dict(runtime.state.get("task_statuses") or {})
            for t in task_order:
                statuses.setdefault(t, TaskStatus.PENDING.value)

            current = statuses[task]

            # Enforce valid transitions: pending -> in_progress -> complete
            if current == TaskStatus.COMPLETE.value:
                return f"Task '{task}' is already complete."

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

            # Enforce ordering: all prior tasks must be complete
            if enforce_order and status == TaskStatus.IN_PROGRESS.value:
                idx = task_order.index(task)
                for prev in task_order[:idx]:
                    if statuses[prev] != TaskStatus.COMPLETE.value:
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
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        return update_task_status
