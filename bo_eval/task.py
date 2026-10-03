from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from bo_eval.env import get_env
from bo_eval.scorer import bo_scorer
from bo_eval.solvers import SOLVERS, init_bo


@task
def bo_eval(
    env: str = "upo_abts",
    solver: str = "react",
    budget: int = 30,
    seed: int = 0,
    tolerance: float = 0.0,
    graph_dir: str = "logs/graphs",
):
    return Task(
        dataset=[Sample(id=env, input=get_env(env).prompt(budget), metadata={"env": env})],
        setup=init_bo(budget, seed),
        solver=SOLVERS[solver](),
        scorer=bo_scorer(tolerance, graph_dir),
    )
