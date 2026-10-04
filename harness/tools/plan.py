from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from harness.types.state import LabState, Plan


@tool
def create_plan():
    async def execute(task_name: str, steps: list[str]) -> str:
        """Create (or replace) the experiment plan for a task: an ordered checklist the
        technician must complete before take_measurement will return a trustworthy result.

        Args:
            task_name: Name of the task or skill this plan applies to.
            steps: Ordered list of step descriptions the technician must perform and check off.
        """
        if not steps:
            raise ToolError("A plan needs at least one step.")

        lab_state = store_as(LabState)
        lab_state.task_plans[task_name] = Plan(steps=list(steps), completed=[False] * len(steps))

        return f"Created plan for '{task_name}' with {len(steps)} step(s)."

    return execute


@tool
def complete_step():
    async def execute(task_name: str, step_number: int) -> str:
        """Check off a step of a task's experiment plan once it has actually been performed.

        Args:
            task_name: Name of the task or skill.
            step_number: 1-based index of the step to mark complete, as listed in the plan.
        """
        lab_state = store_as(LabState)

        if task_name not in lab_state.task_plans:
            raise ToolError(f"No plan exists for task '{task_name}'. Ask the scientist to create one.")

        plan = lab_state.task_plans[task_name]

        if not 1 <= step_number <= len(plan.steps):
            raise ToolError(f"step_number must be between 1 and {len(plan.steps)}.")

        plan.completed[step_number - 1] = True
        lab_state.task_plans[task_name] = plan

        remaining = plan.remaining
        return (f"Checked off step {step_number} ('{plan.steps[step_number - 1]}') for '{task_name}'."
                + (f" Remaining: {remaining}" if remaining else " Plan finished."))

    return execute


@tool
def view_plan():
    async def execute(task_name: str) -> str:
        """View a task's experiment plan and which steps are checked off.

        Args:
            task_name: Name of the task or skill.
        """
        lab_state = store_as(LabState)

        if task_name not in lab_state.task_plans:
            return f"No plan exists for task '{task_name}'."

        return f"Plan for '{task_name}' {lab_state.task_plans[task_name].to_text()}"

    return execute
