"""TaskSteeringMiddleware — implicit state machine for LangChain v1 agents."""

import typing
from collections.abc import Callable
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict

from .types import Task, TaskMiddleware, TaskStatus, TaskSteeringState

_TRANSITION_TOOL_NAME = "update_task_status"


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

    Args:
        tasks: Ordered list of :class:`Task` definitions.
        global_tools: Tools available regardless of which task is active.
        enforce_order: If ``True`` (default), tasks must be completed in
            the order they are defined.
    """

    state_schema = TaskSteeringState

    def __init__(
        self,
        tasks: list[Task],
        global_tools: list | None = None,
        enforce_order: bool = True,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("At least one Task is required.")

        names = [t.name for t in tasks]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate task names: {dupes}")

        self._tasks = tasks
        self._task_order: list[str] = [t.name for t in tasks]
        self._task_map: dict[str, Task] = {t.name: t for t in tasks}
        self._global_tools: list = global_tools or []
        self._enforce_order = enforce_order

        self._transition_tool = self._build_transition_tool()

        # Merge state schemas from task middlewares so their fields
        # persist in the agent's state graph and survive interrupts.
        self.state_schema = self._merge_state_schemas()

        # Auto-register all tools (transition + global + every task's tools).
        # The agent receives these via the middleware ``tools`` attribute so
        # users don't have to duplicate them in ``create_agent(tools=...)``.
        seen: set[str] = set()
        all_tools: list = []
        for t in [
            self._transition_tool,
            *self._global_tools,
            *(tool for task in self._tasks for tool in task.tools),
        ]:
            if t.name not in seen:
                seen.add(t.name)
                all_tools.append(t)
        self.tools = all_tools

    # ── Node-style hooks ──────────────────────────────────────

    def before_agent(
        self, state: TaskSteeringState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Initialize task_statuses on first invocation."""
        if state.get("task_statuses") is None:
            return {
                "task_statuses": {t.name: TaskStatus.PENDING.value for t in self._tasks}
            }
        return None

    # ── Wrap-style hooks ──────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject task pipeline prompt and scope tools per active task."""
        statuses = self._get_statuses(request.state)
        active_name = self._active_task(statuses)

        # 1. Append task pipeline block to system prompt
        block = self._render_status_block(statuses, active_name)
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": block}
        ]

        # 2. Scope tools to active task + globals + transition tool
        allowed_names = self._allowed_tool_names(active_name)
        scoped = [t for t in request.tools if t.name in allowed_names]

        modified = request.override(
            system_message=SystemMessage(content=new_content),
            tools=scoped,
        )

        # 3. Delegate to task-scoped middleware if present
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

        # Intercept update_task_status for completion validation + lifecycle
        if request.tool_call["name"] == _TRANSITION_TOOL_NAME:
            args = request.tool_call["args"]
            task_name = args.get("task")
            target = args.get("status")

            if target == TaskStatus.COMPLETE.value and task_name in self._task_map:
                task_mw = self._get_task_middleware(task_name)
                if task_mw and self._overrides_task(task_mw, "validate_completion"):
                    error = task_mw.validate_completion(request.state)
                    if error:
                        return ToolMessage(
                            content=(
                                f"Cannot complete '{task_name}': {error}. "
                                f"Address the issues then try again."
                            ),
                            tool_call_id=request.tool_call["id"],
                        )

            result = handler(request)

            # Fire lifecycle hooks on successful transition
            if isinstance(result, Command) and task_name in self._task_map:
                task_mw = self._get_task_middleware(task_name)
                if task_mw:
                    if target == TaskStatus.IN_PROGRESS.value:
                        task_mw.on_start(request.state)
                    elif target == TaskStatus.COMPLETE.value:
                        task_mw.on_complete(request.state)

            return result

        # Gate: reject tool calls not in scope for the active task
        allowed = self._allowed_tool_names(active_name)
        if request.tool_call["name"] not in allowed:
            return ToolMessage(
                content=(
                    f"Tool '{request.tool_call['name']}' is not available "
                    f"for the current task."
                ),
                tool_call_id=request.tool_call["id"],
            )

        # Delegate all other tool calls to active task's middleware
        task_mw = self._get_task_middleware(active_name)
        if task_mw and self._overrides(task_mw, "wrap_tool_call"):
            return task_mw.wrap_tool_call(request, handler)

        return handler(request)

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
        """Check if a middleware subclass actually overrides a hook method.

        ``AgentMiddleware`` defines all hooks (raising ``NotImplementedError``),
        so ``hasattr`` is always ``True``.  This checks the MRO instead.
        """
        return getattr(type(middleware), method_name) is not getattr(
            AgentMiddleware, method_name, None
        )

    @staticmethod
    def _overrides_task(middleware: TaskMiddleware, method_name: str) -> bool:
        """Check if a TaskMiddleware subclass overrides a TaskMiddleware hook."""
        return getattr(type(middleware), method_name) is not getattr(
            TaskMiddleware, method_name, None
        )

    def _get_task_middleware(self, task_name: str | None) -> TaskMiddleware | None:
        if task_name is None:
            return None
        task = self._task_map.get(task_name)
        return task.middleware if task else None

    def _allowed_tool_names(self, active_name: str | None) -> set[str]:
        names = {_TRANSITION_TOOL_NAME}
        names.update(t.name for t in self._global_tools)
        if active_name:
            names.update(t.name for t in self._task_map[active_name].tools)
        return names

    def _render_status_block(self, statuses: dict[str, str], active: str | None) -> str:
        icons = {
            TaskStatus.PENDING.value: "[ ]",
            TaskStatus.IN_PROGRESS.value: "[>]",
            TaskStatus.COMPLETE.value: "[x]",
        }

        lines = ["\n<task_pipeline>"]
        for t in self._tasks:
            s = statuses[t.name]
            lines.append(f"  {icons[s]} {t.name} ({s})")

        if active:
            lines.append(f'\n  <current_task name="{active}">')
            lines.append(f"    {self._task_map[active].instruction}")
            lines.append("  </current_task>")

        if self._enforce_order:
            order_str = " -> ".join(self._task_order)
            lines.append("\n  <rules>")
            lines.append(f"    Required order: {order_str}")
            lines.append("    Use update_task_status to advance. Do not skip tasks.")
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
            valid_next = {
                TaskStatus.PENDING.value: TaskStatus.IN_PROGRESS.value,
                TaskStatus.IN_PROGRESS.value: TaskStatus.COMPLETE.value,
            }
            expected = valid_next.get(current)
            if expected != status:
                return (
                    f"Cannot transition '{task}' from '{current}' to "
                    f"'{status}'. Expected next: '{expected or 'N/A'}'."
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
                    "messages": [
                        ToolMessage(
                            f"Task '{task}' -> {status}.\n\n{display}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        return update_task_status
