"""A stand-in for the lab technician that does no lab work and calls no model.

Every plan the scientist writes is treated as perfectly executed: all steps are checked off and
the measurement is taken immediately. Use it to test the scientist/BO loop -- condition choice,
reasoning graph, branch closing, submission -- in isolation, without the manipulation agent, its
token cost, or any physical failure. The experiment budget is still consumed exactly as normal.
"""

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.tool import ToolError
from inspect_ai.util import store_as

from harness.tools.take_measurement import measure
from harness.types.state import LabState


@solver
def auto_technician():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        notes = []

        for task_name, plan in list(lab_state.task_plans.items()):
            if plan.finished:                      # already executed and measured
                continue
            plan.completed = [True] * len(plan.steps)
            lab_state.task_plans[task_name] = plan
            try:
                node = measure(task_name)
            except ToolError as e:
                notes.append(f"'{task_name}' could not be measured: {e}")
                continue
            notes.append(f"{node.id}: {task_name} at {node.params} -> {node.result:.4g}")

        used = len(lab_state.experiment_graph.experiments)
        state.messages.append(ChatMessageUser(content=(
            "TECHNICIAN (automated -- every plan is executed exactly as written).\n"
            + ("\n".join(notes) if notes else "No outstanding plan to execute.")
            + f"\nBudget: {used}/{lab_state.budget} experiments used."
        )))

        return state

    return solve
