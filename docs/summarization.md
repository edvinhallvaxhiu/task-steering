# Task Summarization

When a task completes, its intermediate messages (tool calls, tool results, reasoning) can be compressed to save context window space. Two modes are available.

## Modes

- **`replace`** — removes all task messages, injects a static string into the transition `ToolMessage`.
- **`summarize`** — calls an LLM to produce a summary, injects it into the transition `ToolMessage`. Only `AIMessage`/`ToolMessage` objects are removed; `HumanMessage` objects are preserved.

## Usage

```python
from langchain_task_steering import Task, TaskSteeringMiddleware, TaskSummarization

model = ChatBedrockConverse(model="anthropic.claude-sonnet-4-6-v1", region_name="us-east-1")

pipeline = TaskSteeringMiddleware(
    tasks=[
        Task(
            name="research",
            instruction="Research the topic.",
            tools=[search],
            summarize=TaskSummarization(
                mode="summarize",
                # model is optional — falls back to middleware's model
                # prompt is optional — overrides the default HumanMessage
                # prompt="Summarize in bullet points.",
            ),
        ),
        Task(
            name="write",
            instruction="Write the report.",
            tools=[write],
            summarize=TaskSummarization(mode="replace", content="Research complete."),
        ),
    ],
    model=model,  # default model for summarize mode
)
```

Summarization works identically for tasks inside workflows:

```python
from langchain_task_steering import Workflow, WorkflowSteeringMiddleware

workflow = Workflow(
    name="research_pipeline",
    description="Research and write a report.",
    tasks=[
        Task(
            name="search",
            instruction="Search for info.",
            tools=[search_web],
            summarize=TaskSummarization(mode="summarize"),
        ),
        Task(
            name="write",
            instruction="Write the report.",
            tools=[write_doc],
            summarize=TaskSummarization(mode="replace", content="Research done."),
        ),
    ],
)

middleware = WorkflowSteeringMiddleware(workflows=[workflow], model=model)
```

## How it works

1. When a task transitions to `in_progress`, the middleware records the current message index in `task_message_starts`.
2. When the task transitions to `complete`, messages between the start index and the completion are processed:
   - **Replace**: all task messages are removed via `RemoveMessage`.
   - **Summarize**: `AIMessage`/`ToolMessage` objects are removed; the LLM receives a `SystemMessage` (with task name + instruction), the flattened task messages (tool metadata stripped to plain text), and a `HumanMessage` instruction.
3. The summary is injected into the transition `ToolMessage` (e.g., `Task 'research' -> complete.\n\nTask summary:\n...`).
4. By default, the text content of the complete-transition `AIMessage` is also stripped (`trim_complete_message=True`), since it's redundant once the summary exists.

The `model` for `summarize` mode is resolved in order: `TaskSummarization.model` > middleware `model` param. If neither is set, summarization is skipped with a warning.

## TaskSummarization fields

| Field | Default | Description |
|---|---|---|
| `mode` | `"replace"` | `"replace"` or `"summarize"`. |
| `content` | `None` | Replacement text for `replace` mode (required). |
| `model` | `None` | Chat model for `summarize` mode. Falls back to middleware `model`. |
| `prompt` | `None` | Custom `HumanMessage` content for the summarizer. |
| `trim_complete_message` | `True` | Strip text from the complete-transition `AIMessage`. |
