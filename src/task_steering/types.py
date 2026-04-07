"""Public types for task-steering."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing_extensions import NotRequired


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class TaskSteeringState(AgentState):
    """Extends AgentState with task tracking managed by the middleware."""

    task_statuses: NotRequired[dict[str, str]]


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

    def on_start(self, state: dict[str, Any]) -> None:
        """Called after the task transitions to in_progress.

        Use for side effects like logging, external state updates, or
        recording the message index for later trail capture.
        """

    def on_complete(self, state: dict[str, Any]) -> None:
        """Called after the task transitions to complete (after validation).

        Use for side effects like reasoning trail capture or
        external state updates.
        """


@dataclass
class Task:
    """A single task in an ordered pipeline.

    Args:
        name: Unique task identifier (used in prompts and state).
        instruction: Injected into the system prompt when this task is active.
        tools: LangChain tools available when this task is IN_PROGRESS.
        middleware: Optional scoped middleware, active only when this task
            is IN_PROGRESS. Can implement any ``AgentMiddleware`` hook plus
            ``validate_completion`` for completion gating.
    """

    name: str
    instruction: str
    tools: list
    middleware: "TaskMiddleware | None" = None
