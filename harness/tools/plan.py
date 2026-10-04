from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.tools.lab_backend import existing_backend
from harness.types.state import LabState, Plan


@tool
def create_plan():
    async def execute(task_name: str, steps: list[str], params: dict[str, float],
                      parent: str = "root", reasoning: str = "",
                      allow_repeat: bool = False) -> str:
        """Create (or replace) the plan for one experiment: the condition to test, where it belongs
        in the reasoning graph, and the ordered checklist the technician must complete before
        take_measurement will return a trustworthy result.

        Args:
            task_name: Name of the task this plan applies to.
            steps: Ordered list of physical steps the technician must perform and check off.
            params: The condition to test -- a value for every parameter, e.g. {"ph": 3.5, ...}. Snapped to the nearest feasible condition when measured.
            parent: Id of the graph node this experiment follows from ("root" or an experiment id like "E3").
            reasoning: Why this experiment follows from the parent node.
            allow_repeat: Set True only to deliberately replicate a condition that has already been measured.
        """
        if not steps:
            raise ToolError("A plan needs at least one step.")

        lab_state = store_as(LabState)
        env, g = get_env(lab_state.env), lab_state.experiment_graph

        try:
            cond = env.condition(env.index(params))
        except ValueError as e:
            raise ToolError(str(e))
        if parent not in g.nodes:
            raise ToolError(f"Unknown parent node '{parent}'.")
        if g.nodes[parent].closed:
            raise ToolError(f"Branch '{parent}' is closed and cannot be extended.")

        # Requested settings snap to the nearest feasible condition, so a hand-adjusted condition
        # can land back on one already measured and silently waste budget.
        if not allow_repeat:
            dup = next((nd for nd in g.experiments if nd.params == cond), None)
            if dup is not None:
                raise ToolError(
                    f"That condition snaps to {cond}, which {dup.id} already measured "
                    f"(result {dup.result}). Pick a different condition -- the bayes_opt_suggest "
                    f"params are guaranteed to be unmeasured -- or pass allow_repeat=True to "
                    f"replicate it on purpose."
                )

        lab_state.task_plans[task_name] = Plan(
            steps=list(steps), completed=[False] * len(steps),
            params=params, parent=parent, reasoning=reasoning,
        )
        # A new plan supersedes whatever failed under the old one, so an action that can never
        # succeed (an unreachable well, say) cannot deadlock the run.
        if (backend := existing_backend()) is not None:
            backend.clear_failures()

        return (f"Created plan for '{task_name}' with {len(steps)} step(s), condition "
                f"{cond}, following {parent}.")

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
