import json

import numpy as np
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState, Edge, Node


def run(params: dict, parent: str = "root", reasoning: str = "") -> Node:
    s = store_as(BOState)
    env, g = get_env(s.env), s.graph
    if parent not in g.nodes:
        raise ToolError(f"Unknown parent node '{parent}'.")
    if g.nodes[parent].closed:
        raise ToolError(f"Branch '{parent}' is closed and cannot be extended.")
    n = len(g.experiments)
    if n >= s.budget:
        raise ToolError("Experiment budget exhausted. Submit your answer.")
    try:
        i = env.index(params)
    except ValueError as e:
        raise ToolError(str(e))
    node = Node(id=f"E{n + 1}", params=env.condition(i), result=env.sample(i, np.random.default_rng([s.seed, n])))
    g.nodes[node.id] = node
    g.edges.append(Edge(source=parent, target=node.id, reasoning=reasoning))
    s.graph = g
    return node


@tool
def run_experiment():
    async def execute(params: dict[str, float], parent: str = "root", reasoning: str = "") -> str:
        """Run one experiment (consumes one unit of budget) and add it to the reasoning graph.

        Args:
            params: Value for every parameter, e.g. {"ph": 3.5, ...}. Snapped to the nearest feasible condition.
            parent: Id of the graph node this experiment follows from ("root" or an experiment id like "E3").
            reasoning: Why this experiment follows from the parent node.
        """
        node = run(params, parent, reasoning)
        s = store_as(BOState)
        return json.dumps(
            {"id": node.id, "params": node.params, "result": node.result,
             "remaining_budget": s.budget - len(s.graph.experiments)}
        )

    return execute
