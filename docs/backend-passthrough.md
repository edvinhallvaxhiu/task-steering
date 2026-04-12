# Backend Tools Passthrough

When the middleware is used inside `create_deep_agent`, other middleware (e.g., `FilesystemMiddleware`, `SubAgentMiddleware`) contribute tools that get filtered out by tool scoping unless explicitly added to `global_tools` or a task's `tools`. Backend tools passthrough lets known backend tools pass through the filter automatically.

## Usage

```python
pipeline = TaskSteeringMiddleware(
    tasks=[...],
    backend_tools_passthrough=True,
)
```

Works the same for workflow mode:

```python
middleware = WorkflowSteeringMiddleware(
    workflows=[...],
    backend_tools_passthrough=True,
)
```

## Default whitelist

```python
TaskSteeringMiddleware.DEFAULT_BACKEND_TOOLS
# frozenset({'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep',
#            'execute', 'write_todos', 'task', 'start_async_task',
#            'check_async_task', 'update_async_task', 'cancel_async_task',
#            'list_async_tasks'})
```

## Custom whitelist

```python
TaskSteeringMiddleware(
    tasks=[...],
    backend_tools_passthrough=True,
    backend_tools={"read_file", "write_file", "my_custom_tool"},
)
```

## Inspect at runtime

```python
pipeline.get_backend_tools()  # returns the effective whitelist (frozenset)
```

No `backend` is required for passthrough — it just whitelists tool names in the filter.
