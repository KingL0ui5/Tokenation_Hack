import json

from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState


def submit_params(params: dict) -> dict:
    s = store_as(BOState)
    env = get_env(s.env)
    try:
        s.submission = env.condition(env.index(params))
    except ValueError as e:
        raise ToolError(str(e))
    return s.submission


@tool
def submit():
    async def execute(params: dict[str, float]) -> str:
        """Submit the parameter configuration you believe is optimal. Ends the episode.

        Args:
            params: Value for every parameter.
        """
        return json.dumps(submit_params(params))

    return execute
