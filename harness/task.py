from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.scorer import lab_scorer
from harness.solvers import scientist_solver, technician_solver
from harness.types.state import LabState

@solver
def init_lab_state(budget: int = 30, env: str = "upo_abts", seed: int = 0):

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        lab_state.budget = budget
        lab_state.env = env
        lab_state.seed = seed

        return state
    
    return solve

@task
def autonomous_lab_task(env: str = "upo_abts", budget: int = 30, seed: int = 0,
                        tolerance: float = 0.0, graph_dir: str = "logs/graphs"):
    plan_sequence = []
    
    for _ in range(5):
        plan_sequence.append(scientist_solver())
        plan_sequence.append(technician_solver())

    return Task(
        dataset=[Sample(id=env, input=get_env(env).prompt(budget), metadata={"env": env, "seed": seed})],
        setup=init_lab_state(budget, env, seed),
        plan=plan_sequence,
        scorer=lab_scorer(graph_dir, tolerance),
    )
