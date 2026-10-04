"""The technician's only way to turn a finished task into a number.

A measurement is a real experiment: it consumes budget and is recorded as a node in the
task's reasoning graph. It is only trustworthy once every step of the scientist's plan for
that task has been checked off -- taking one early still costs budget but yields an INVALID
reading with no value attached.
"""

import json

import numpy as np
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.types.state import Edge, LabState, Node, ReasoningGraph


def measure(task_name: str, params: dict, parent: str = "root", reasoning: str = "") -> Node:
    s = store_as(LabState)
    if task_name not in s.task_plans:
        raise ToolError(f"No plan exists for task '{task_name}'. Ask the scientist to create one.")
    plan = s.task_plans[task_name]
    env = get_env(s.env)
    g = s.task_graphs.get(task_name, ReasoningGraph())
    if parent not in g.nodes:
        raise ToolError(f"Unknown parent node '{parent}' in task '{task_name}'.")
    if g.nodes[parent].closed:
        raise ToolError(f"Branch '{parent}' is closed and cannot be extended.")
    n = sum(len(graph.experiments) for graph in s.task_graphs.values())
    if n >= s.budget:
        raise ToolError("Experiment budget exhausted. Submit your answer.")
    try:
        i = env.index(params)
    except ValueError as e:
        raise ToolError(str(e))

    valid = plan.finished
    node = Node(
        id=f"E{n + 1}",
        params=env.condition(i),
        result=env.sample(i, np.random.default_rng([s.seed, n])) if valid else None,
        valid=valid,
    )
    g.nodes[node.id] = node
    g.edges.append(Edge(source=parent, target=node.id, reasoning=reasoning))
    s.task_graphs[task_name] = g
    return node


@tool
def take_measurement():
    async def execute(task_name: str, params: dict[str, float], parent: str = "root", reasoning: str = "") -> str:
        """Measure the result of a finished task (consumes one unit of budget) and add it to the
        task's reasoning graph. The reading is only valid if every step of the task's plan has
        been checked off with complete_step; otherwise it returns an INVALID result with no value.

        Args:
            task_name: Name of the task or skill being measured (must already have a plan).
            params: Value for every parameter, e.g. {"ph": 3.5, ...}. Snapped to the nearest feasible condition.
            parent: Id of the graph node this measurement follows from ("root" or a node id like "E3").
            reasoning: Why this measurement follows from the parent node.
        """
        node = measure(task_name, params, parent, reasoning)
        s = store_as(LabState)
        out = {
            "id": node.id,
            "task_name": task_name,
            "params": node.params,
            "valid": node.valid,
            "result": node.result,
            "remaining_budget": s.budget - sum(len(g.experiments) for g in s.task_graphs.values()),
        }
        if not node.valid:
            out["error"] = (
                f"INVALID MEASUREMENT: the plan for '{task_name}' is not complete, so no reading was "
                f"obtained. Finish the remaining steps, check them off with complete_step, then measure again."
            )
            out["remaining_steps"] = s.task_plans[task_name].remaining
        return json.dumps(out)

    return execute
