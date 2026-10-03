import json

import numpy as np
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.state import LabState, Edge, Node, ReasoningGraph


def run(task_name: str, params: dict, parent: str = "root", reasoning: str = "") -> Node:
    s = store_as(LabState)
    if task_name not in s.graphs:
        s.graphs[task_name] = ReasoningGraph()
        
    env, g = get_env(s.env), s.graphs[task_name]
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
    s.graphs[task_name] = g
    return node


@tool
def run_experiment():
    async def execute(task_name: str, params: dict[str, float], parent: str = "root", reasoning: str = "") -> str:
        """Run one experiment for a specific task and add it to the reasoning graph.

        Args:
            task_name: Name of the task or skill being executed (e.g., 'pick_up_test_tube')
            params: Value for every parameter, e.g. {"ph": 3.5, ...}. Snapped to the nearest feasible condition.
            parent: Id of the graph node this experiment follows from ("root" or an experiment id like "E3").
            reasoning: Why this experiment follows from the parent node.
        """
        node = run(task_name, params, parent, reasoning)
        s = store_as(LabState)
        return json.dumps(
            {"id": node.id, "params": node.params, "result": node.result,
             "remaining_budget": s.budget - len(s.graphs[task_name].experiments)}
        )

    return execute
