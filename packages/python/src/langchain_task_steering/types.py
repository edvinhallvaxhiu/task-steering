"""Public types for task-steering."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing_extensions import NotRequired, TypedDict


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class SkillMetadata(TypedDict):
    """Metadata parsed from a SKILL.md frontmatter.

    Compatible with deepagents' ``SkillMetadata`` — the two are
    interchangeable via structural subtyping.
    """

    name: str
    description: str
    path: str
    license: NotRequired[str | None]
    compatibility: NotRequired[str | None]
    metadata: NotRequired[dict[str, str]]
    allowed_tools: NotRequired[list[str]]


class TaskSteeringState(AgentState):
    """Extends AgentState with task tracking managed by the middleware."""

    task_statuses: NotRequired[dict[str, str]]
    task_message_starts: NotRequired[dict[str, int]]
    nudge_count: NotRequired[int]
    skills_metadata: NotRequired[list[SkillMetadata]]


class TaskMiddleware(AgentMiddleware):
    """Base class for task-scoped middleware.

    Subclass this to add:
    - Mid-task enforcement via ``wrap_tool_call``
    - Extra prompt injection via ``wrap_model_call``
    - Completion validation via ``validate_completion``
    - Persistent state via ``state_schema`` (survives interrupts)

    The middleware hooks are only active when the owning task is IN_PROGRESS.

    To persist custom state across interrupts, set ``state_schema`` to a
    ``TypedDict`` that extends ``AgentState``.  ``TaskSteeringMiddleware``
    merges all task middleware schemas into its own so the fields are part
    of the agent's checkpointed state::

        class ThreatsState(AgentState):
            gap_analysis_uses: NotRequired[int]

        class ThreatsMiddleware(TaskMiddleware):
            state_schema = ThreatsState
            ...
    """

    def validate_completion(self, state: dict[str, Any]) -> str | None:
        """Called when the task attempts to transition to complete.

        Return an error message string to reject the transition,
        or ``None`` to allow it.
        """
        return None

    def on_start(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Called after the task transitions to in_progress.

        Use for side effects like logging, external state updates, or
        recording the message index for later trail capture.

        Optionally return a ``dict`` of state updates to merge into the
        transition ``Command``.  If ``"messages"`` appears in both the
        returned dict and the existing ``Command.update``, the lists are
        **appended** (not overwritten) so the transition ``ToolMessage``
        is preserved.  Return ``None`` (default) for no state changes.

        Note: ``state`` contains the *projected* post-transition
        ``task_statuses`` but all other fields reflect the pre-transition
        snapshot (the ``Command`` has not been applied to the graph yet).
        """
        return None

    def on_complete(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Called after the task transitions to complete (after validation).

        Use for side effects like reasoning trail capture or
        external state updates.

        Optionally return a ``dict`` of state updates to merge into the
        transition ``Command``.  If ``"messages"`` appears in both the
        returned dict and the existing ``Command.update``, the lists are
        **appended** (not overwritten) so the transition ``ToolMessage``
        is preserved.  Return ``None`` (default) for no state changes.

        Note: ``state`` contains the *projected* post-transition
        ``task_statuses`` but all other fields reflect the pre-transition
        snapshot (the ``Command`` has not been applied to the graph yet).
        """
        return None

    async def avalidate_completion(self, state: dict[str, Any]) -> str | None:
        """Async version of ``validate_completion``.

        Override this for validation that requires async I/O (e.g. calling
        an external service). The default delegates to the sync version.
        """
        return self.validate_completion(state)

    async def aon_start(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Async version of ``on_start``.

        Override this for start hooks that require async I/O.
        The default delegates to the sync version.
        """
        return self.on_start(state)

    async def aon_complete(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Async version of ``on_complete``.

        Override this for completion hooks that require async I/O.
        The default delegates to the sync version.
        """
        return self.on_complete(state)


@dataclass
class TaskSummarization:
    """Configuration for post-completion message summarization.

    Attached to a :class:`Task` via ``Task(summarize=...)``.  When the task
    transitions to ``complete``, the middleware replaces task messages
    according to the chosen mode.

    Modes:

    ``"replace"``
        Removes **all** messages produced during the task and inserts a
        single ``AIMessage`` whose content is *content*.  Use this when
        you already know the summary (e.g. a static acknowledgment).

    ``"summarize"``
        Feeds the task messages to an LLM and replaces every
        ``AIMessage`` / ``ToolMessage`` produced during the task with the
        LLM's summary (``HumanMessage`` objects are preserved).  The LLM
        receives the task instruction in its system prompt for context.

    Args:
        mode: ``"replace"`` or ``"summarize"``.
        content: Replacement text for ``"replace"`` mode (required).
        model: Chat model for ``"summarize"`` mode.  Any LangChain
            ``BaseChatModel``.  Optional — if ``None``, falls back to
            ``TaskSteeringMiddleware(model=...)`` (the same model used
            by ``create_agent`` / ``create_deep_agent``).
        prompt: Custom ``HumanMessage`` content appended after the task
            messages when calling the summarization model.  Use this to
            give explicit instructions on *how* to summarize (e.g.
            ``"List every tool call and its result."``).  If ``None``, a
            default instruction is used.
        trim_complete_message: If ``True`` (default), strip the text
            content from the complete-transition ``AIMessage``, keeping
            only its ``tool_calls``.  The text is redundant once the
            summary is in the ``ToolMessage``.
    """

    mode: Literal["replace", "summarize"] = "replace"
    content: str | None = None
    model: Any = None  # BaseChatModel
    prompt: str | None = None
    trim_complete_message: bool = True

    def __post_init__(self) -> None:
        if self.mode == "replace" and self.content is None:
            raise ValueError("TaskSummarization(mode='replace') requires 'content'.")


@dataclass
class Task:
    """A single task in an ordered pipeline.

    Args:
        name: Unique task identifier (used in prompts and state).
        instruction: Injected into the system prompt when this task is active.
        tools: LangChain tools available when this task is IN_PROGRESS.
        middleware: Optional scoped middleware, active only when this task
            is IN_PROGRESS. Can be a single ``TaskMiddleware``, a list of
            them (composed in order, first = outermost), or ``None``.
            Each can implement any ``AgentMiddleware`` hook plus
            ``validate_completion`` for completion gating.
        summarize: Optional post-completion summarization config.
            See :class:`TaskSummarization`.
    """

    name: str
    instruction: str
    tools: list
    middleware: "TaskMiddleware | AgentMiddleware | list[TaskMiddleware | AgentMiddleware] | None" = None
    skills: list[str] | None = None
    summarize: "TaskSummarization | None" = None
