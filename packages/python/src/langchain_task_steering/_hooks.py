"""Dynamic hook discovery for AgentMiddleware.

Inspects the ``AgentMiddleware`` base class at import time to find all
wrap-style hooks (``wrap_*`` / ``awrap_*``).  This lets the adapter,
composed middleware, and steering middleware pick up new hooks
automatically when LangChain adds them — no manual enumeration needed.
"""

import inspect
from langchain.agents.middleware import AgentMiddleware


def _discover_wrap_hooks() -> list[tuple[str, str | None]]:
    """Return (sync_name, async_name | None) for every wrap-style hook."""
    members = set(dir(AgentMiddleware))
    pairs: list[tuple[str, str | None]] = []

    for name in sorted(members):
        if name.startswith("_"):
            continue
        attr = getattr(AgentMiddleware, name, None)
        if attr is None or not callable(attr):
            continue
        if name.startswith("a") and name[1:] in members:
            continue

        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            continue

        params = set(sig.parameters.keys()) - {"self"}
        if "request" in params and "handler" in params:
            async_name = "a" + name
            has_async = async_name in members and callable(
                getattr(AgentMiddleware, async_name, None)
            )
            pairs.append((name, async_name if has_async else None))

    return pairs


# Discovered once at import time.
WRAP_HOOK_PAIRS: list[tuple[str, str | None]] = _discover_wrap_hooks()


def overrides_base(middleware, method_name: str) -> bool:
    """Check if a middleware overrides a method relative to AgentMiddleware.

    Checks both instance attributes (dynamically bound methods) and the
    class MRO. Handles duck-typed objects that don't inherit AgentMiddleware.
    """
    # Instance attribute (dynamic binding) takes priority
    if method_name in getattr(middleware, "__dict__", {}):
        return True
    # Class-level override
    _sentinel = object()
    class_method = getattr(type(middleware), method_name, _sentinel)
    if class_method is _sentinel:
        return False
    return class_method is not getattr(AgentMiddleware, method_name, None)
