import json

from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.types.state import LabState


def submit_params(task_name: str, params: dict, require_closed: bool = False) -> dict:
    s = store_as(LabState)
    env = get_env(s.env)
    try:
        cond = env.condition(env.index(params))
    except ValueError as e:
        raise ToolError(str(e))
    if require_closed:
        if task_name not in s.graphs:
            raise ToolError(f"Task graph '{task_name}' does not exist.")
        g = s.graphs[task_name]
        dangling = [n for n in g.open_leaves() if g.nodes[n].params != cond]
        if dangling:
            raise ToolError(
                f"Unexplained open branches: {dangling}. Call close_branch on each with the reason "
                "it was not continued, then submit again."
            )
    s.submission = cond
    return cond


@tool
def submit():
    async def execute(task_name: str, params: dict[str, float]) -> str:
        """Submit the configuration you believe is optimal for a task. Ends the episode. Every other unextended experiment must first be closed with close_branch.

        Args:
            task_name: Name of the task
            params: Value for every parameter.
        """
        return json.dumps(submit_params(task_name, params, require_closed=True))

    return execute
