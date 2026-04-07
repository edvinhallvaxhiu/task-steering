"""Simple example: TaskSteeringMiddleware with create_agent and Bedrock."""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

from task_steering import Task, TaskMiddleware, TaskSteeringMiddleware


# ── Tools ────────────────────────────────────────────────────


@tool
def gather_requirements(topic: str) -> str:
    """Gather requirements for a given topic."""
    return f"Requirements for '{topic}': must be fast, secure, and scalable."


@tool
def write_design(requirements: str) -> str:
    """Write a design document based on requirements."""
    return f"Design document created based on: {requirements}"


@tool
def review_design(design: str) -> str:
    """Review a design document and provide feedback."""
    return f"Review complete. Design looks good: {design}"


# ── Optional task middleware ─────────────────────────────────


class DesignMiddleware(TaskMiddleware):
    """Require the design tool to be called before completing."""

    def validate_completion(self, state) -> str | None:
        messages = state.get("messages", [])
        used_design = any(getattr(m, "name", None) == "write_design" for m in messages)
        if not used_design:
            return "You must call write_design before completing this task."
        return None


# ── Tasks ────────────────────────────────────────────────────

tasks = [
    Task(
        name="requirements",
        instruction="Gather the requirements for a login page.",
        tools=[gather_requirements],
    ),
    Task(
        name="design",
        instruction="Write a design document based on the gathered requirements.",
        tools=[write_design],
        middleware=DesignMiddleware(),
    ),
    Task(
        name="review",
        instruction="Review the design document and provide final feedback.",
        tools=[review_design],
    ),
]

# ── Build agent ──────────────────────────────────────────────

middleware = TaskSteeringMiddleware(tasks=tasks)

model = ChatBedrockConverse(
    model="global.anthropic.claude-opus-4-6-v1",
    region_name="us-east-1",
)

agent = create_agent(
    model=model,
    middleware=[middleware],
    system_prompt="You are a helpful software architect. Complete each task in order.",
)

# ── Run ──────────────────────────────────────────────────────


def print_message(msg):
    role = getattr(msg, "type", "unknown")
    name = getattr(msg, "name", None)
    content = getattr(msg, "content", str(msg))

    label = f"[{role}]" if not name else f"[{role} | {name}]"

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(f"{label} {block['text']}")
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                print(f"{label} -> {block['name']}({block.get('input', {})})")
    elif isinstance(content, str):
        print(f"{label} {content}")


if __name__ == "__main__":
    inputs = {"messages": [{"role": "user", "content": "Let's design a login page."}]}

    print("=== Streaming Updates ===\n")
    for update in agent.stream(inputs, {"recursion_limit": 50}, stream_mode="updates"):
        for node_name, state_update in update.items():
            messages = state_update.get("messages", [])
            for msg in messages:
                print_message(msg)
            # Print task status changes
            statuses = state_update.get("task_statuses")
            if statuses:
                print(f"  [statuses] {statuses}")
            print()

    print("=== Done ===")
