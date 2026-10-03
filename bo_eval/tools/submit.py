import json

from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState


def submit_params(params: dict, require_closed: bool = False) -> dict:
    s = store_as(BOState)
    env = get_env(s.env)
    try:
        cond = env.condition(env.index(params))
    except ValueError as e:
        raise ToolError(str(e))
    if require_closed:
        dangling = [n for n in s.graph.open_leaves() if s.graph.nodes[n].params != cond]
        if dangling:
            raise ToolError(
                f"Unexplained open branches: {dangling}. Call close_branch on each with the reason "
                "it was not continued, then submit again."
            )
    s.submission = cond
    return cond


@tool
def submit():
    async def execute(params: dict[str, float]) -> str:
        """Submit the configuration you believe is optimal. Ends the episode. Every other unextended experiment must first be closed with close_branch.

        Args:
            params: Value for every parameter.
        """
        return json.dumps(submit_params(params, require_closed=True))

    return execute
