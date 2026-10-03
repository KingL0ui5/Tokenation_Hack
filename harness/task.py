from inspect_ai import task, Task
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.util import store_as

from .solvers import scientist_solver, technician_solver
from .state import LabState

@solver
def init_lab_state(budget: int = 30):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        s = store_as(LabState)
        s.budget = budget
        return state
    return solve

@task
def autonomous_lab_task():
    plan_sequence = []
    for _ in range(5):
        plan_sequence.append(scientist_solver())
        plan_sequence.append(technician_solver())

    return Task(
        dataset=[{"input": "Synthesize Compound X"}],
        setup=init_lab_state(),
        plan=plan_sequence,
        # TODO: Add appropriate scorer once environment is fully defined
    )
