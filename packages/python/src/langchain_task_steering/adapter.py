"""Adapter to use an AgentMiddleware as a TaskMiddleware."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware

from ._hooks import WRAP_HOOK_PAIRS, overrides_base
from .types import TaskMiddleware


class AgentMiddlewareAdapter(TaskMiddleware):
    """Wraps a standard ``AgentMiddleware`` so it can be used at task scope.

    Wrap-style hooks (``wrap_*`` / ``awrap_*``) are discovered dynamically
    from ``AgentMiddleware`` at import time.  Only hooks the inner middleware
    actually overrides are forwarded; the rest fall through to the handler.

    This means new hooks added to ``AgentMiddleware`` in future LangChain
    versions are picked up automatically — no code changes required here.

    Additionally forwards:

    - ``tools``        → inner middleware's tools (scoped by ``TaskSteeringMiddleware``)
    - ``state_schema`` → inner middleware's state schema (merged into agent state)

    Agent-level hooks (``before_agent``, ``after_agent``) are **not** forwarded
    — use ``on_start`` / ``on_complete`` for task lifecycle events.

    Example::

        from langchain.agents.middleware import SummarizationMiddleware
        from langchain_task_steering import Task, AgentMiddlewareAdapter

        Task(
            name="research",
            instruction="...",
            tools=[search_tool],
            middleware=AgentMiddlewareAdapter(SummarizationMiddleware()),
        )
    """

    def __init__(self, inner: AgentMiddleware) -> None:
        super().__init__()
        self._inner = inner

        inner_tools = getattr(inner, "tools", None)
        self.tools: list = list(inner_tools) if inner_tools else []

        inner_schema = getattr(inner, "state_schema", None)
        if inner_schema is not None:
            self.state_schema = inner_schema

        # Dynamically bind forwarding methods for every discovered
        # wrap-style hook the inner middleware actually overrides.
        for sync_name, async_name in WRAP_HOOK_PAIRS:
            self._bind_sync_hook(sync_name)
            if async_name:
                self._bind_async_hook(async_name)

    def _bind_sync_hook(self, method_name: str) -> None:
        inner = self._inner
        has_override = overrides_base(inner, method_name)

        def forwarder(request, handler, _m=method_name, _has=has_override):
            if _has:
                return getattr(inner, _m)(request, handler)
            return handler(request)

        setattr(self, method_name, forwarder)

    def _bind_async_hook(self, method_name: str) -> None:
        inner = self._inner
        has_override = overrides_base(inner, method_name)

        async def forwarder(request, handler, _m=method_name, _has=has_override):
            if _has:
                return await getattr(inner, _m)(request, handler)
            return await handler(request)

        setattr(self, method_name, forwarder)
