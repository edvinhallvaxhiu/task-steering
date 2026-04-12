# Task-Scoped Skills

Skills are prompt-injected capabilities loaded from `SKILL.md` files. When configured, skills are scoped per task — just like tools.

`SkillsMiddleware` (in `create_deep_agent`) loads all skills into state. `TaskSteeringMiddleware` and `WorkflowSteeringMiddleware` filter them per task.

## Setup

```python
agent = create_deep_agent(
    backend=my_backend,
    skills=["/skills/user/", "/skills/project/"],
    middleware=[
        TaskSteeringMiddleware(
            tasks=[
                Task(name="research", instruction="Research the topic.",
                     tools=[search], skills=["web-research", "citation-format"]),
                Task(name="write_report", instruction="Write the report.",
                     tools=[write], skills=["report-writing"]),
            ],
            global_skills=["general-formatting"],
        ),
    ],
)
```

For workflow mode, skills live on the `Workflow`:

```python
Workflow(
    name="research",
    description="Research pipeline",
    tasks=[
        Task(name="search", instruction="...", tools=[search], skills=["web-research"]),
        Task(name="write", instruction="...", tools=[write]),
    ],
    global_skills=["general-formatting"],
)
```

## What the model sees

When skills are active, the model sees them in the status block:

```xml
<task_pipeline>
  [x] research (complete)
  [>] write_report (in_progress)

  <current_task name="write_report">
    Write the report.
  </current_task>

  <available_skills>
    - report-writing: Templates and structure for technical reports. Path: /skills/project/report-writing/SKILL.md
    - general-formatting: Standard formatting guidelines. Path: /skills/user/general-formatting/SKILL.md
  </available_skills>

  <rules>
    Required order: research -> write_report
    Use update_task_status to advance. Do not skip tasks.
    To use a skill, read its SKILL.md file for full instructions.
  </rules>
</task_pipeline>
```

## Auto-whitelisted tools

When skills are active, `read_file` and `ls` are auto-whitelisted in the tool filter for any task that has skills (its own or via `global_skills`) so the model can read `SKILL.md` files.

Skills can also declare `allowed_tools` in their frontmatter — these are auto-whitelisted when the skill is visible.
