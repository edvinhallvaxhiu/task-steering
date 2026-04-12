"""Example: TaskSummarization with both replace and summarize modes.

Demonstrates how completed tasks can compress their message trails:
  - Task 1 (requirements): uses ``mode="summarize"`` -- an LLM distills
    the full tool-call conversation into a single AIMessage.
  - Task 2 (design): uses ``mode="replace"`` -- all task messages are
    swapped for a static AIMessage you define upfront.
  - Task 3 (review): no summarization -- messages are kept as-is.

Run:
    python examples/summarization_agent.py
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

from langchain_task_steering import (
    Task,
    TaskMiddleware,
    TaskSteeringMiddleware,
    TaskSummarization,
)


# -- Tools ------------------------------------------------------------


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


# -- Model -------------------------------------------------------------

model = ChatBedrockConverse(
    model="global.anthropic.claude-opus-4-6-v1",
    region_name="us-east-1",
)


# -- Optional task middleware ------------------------------------------


class DesignMiddleware(TaskMiddleware):
    """Require the design tool to be called before completing."""

    def validate_completion(self, state) -> str | None:
        messages = state.get("messages", [])
        used_design = any(getattr(m, "name", None) == "write_design" for m in messages)
        if not used_design:
            return "You must call write_design before completing this task."
        return None


# -- Tasks -------------------------------------------------------------

tasks = [
    # Mode "summarize" -- LLM reads the task's messages and produces a
    # concise summary.  The same model used by the agent is reused here,
    # but you can pass any BaseChatModel.
    Task(
        name="requirements",
        instruction="Gather the requirements for a login page.",
        tools=[gather_requirements],
        summarize=TaskSummarization(
            mode="summarize",
            # Optional: override the default HumanMessage instruction
            # prompt="Summarize in bullet points.",
        ),
    ),
    # Mode "replace" -- all task messages are replaced with a fixed string.
    # Useful when you already know the outcome or want a brief marker.
    Task(
        name="design",
        instruction="Write a design document based on the gathered requirements.",
        tools=[write_design],
        middleware=DesignMiddleware(),
        summarize=TaskSummarization(
            mode="replace",
            content="Design document has been written and saved.",
        ),
    ),
    # No summarization -- messages are preserved in full.
    Task(
        name="review",
        instruction="Review the design document and provide final feedback.",
        tools=[review_design],
    ),
]

# -- Build agent -------------------------------------------------------

middleware = TaskSteeringMiddleware(tasks=tasks, model=model)

agent = create_agent(
    model=model,
    middleware=[middleware],
    system_prompt="You are a helpful software architect. Complete each task in order.",
)

# -- Run ---------------------------------------------------------------


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
    final_state = None
    for update in agent.stream(inputs, {"recursion_limit": 50}, stream_mode="values"):
        final_state = update
        messages = update.get("messages", [])
        if messages:
            print_message(messages[-1])
            statuses = update.get("task_statuses")
            if statuses:
                print(f"  [statuses] {statuses}")
            print()

    # Print the full final message list to see the effect of summarization
    print("\n=== Final Messages ===\n")
    if final_state:
        for i, msg in enumerate(final_state.get("messages", [])):
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)

            # Extract text from block-style content
            if isinstance(content, list):
                text_parts = [
                    b["text"]
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                preview = " ".join(text_parts)[:140]
            elif isinstance(content, str):
                preview = content[:140]
            else:
                preview = str(content)[:140]

            suffix = ""
            if tool_calls:
                names = ", ".join(tc.get("name", "?") for tc in tool_calls)
                suffix = f"  -> {names}"
            print(f"  [{i:2d}] {role:8s}: {preview}{suffix}")
