"""The technician's only way to turn a finished plan into a number.

A measurement is a real experiment: it consumes budget and becomes a node of the single
reasoning graph, linked to the parent node with the reasoning the scientist gave when it
created the plan -- the technician chooses nothing here, it only reports that the plan is done.
The reading is trustworthy only once every step of that plan has been checked off; measuring
early still costs budget but the measurement fails, leaving a node with no value attached.
"""

import json

import numpy as np
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.tools.lab_backend import existing_backend
from harness.types.state import Edge, LabState, Node


def measure(task_name: str) -> Node:
    s = store_as(LabState)
    if task_name not in s.task_plans:
        raise ToolError(f"No plan exists for task '{task_name}'. Ask the scientist to create one.")
    plan = s.task_plans[task_name]

    # An action that failed (collision, unreachable site, tip fault) did not happen, so the well
    # is not in the planned state. Refuse to measure -- and spend no budget -- until it is redone.
    backend = existing_backend()
    if backend and backend.unresolved_failures:
        pending = [f"{f['tool']}({', '.join(f'{k}={v!r}' for k, v in f['args'].items())}): {f['reason']}"
                   for f in backend.unresolved_failures.values()]
        raise ToolError(
            "Cannot measure: these actions failed and have not been repeated successfully, so the "
            "bench is not in the planned state. No budget was spent. Redo them, or ask the "
            "scientist to re-plan.\n  " + "\n  ".join(pending)
        )
    env, g = get_env(s.env), s.experiment_graph
    if plan.parent not in g.nodes:
        raise ToolError(f"Plan '{task_name}' names an unknown parent node '{plan.parent}'.")
    if g.nodes[plan.parent].closed:
        raise ToolError(f"Branch '{plan.parent}' is closed and cannot be extended.")
    n = len(g.experiments)
    if n >= s.budget:
        raise ToolError("Experiment budget exhausted. The scientist must submit an answer.")
    try:
        i = env.index(plan.params)
    except ValueError as e:
        raise ToolError(f"Plan '{task_name}' does not state a usable condition: {e}")

    valid = plan.finished
    node = Node(
        id=f"E{n + 1}",
        params=env.condition(i),
        result=env.sample(i, np.random.default_rng([s.seed, n])) if valid else None,
        valid=valid,
    )
    g.nodes[node.id] = node
    g.edges.append(Edge(source=plan.parent, target=node.id, reasoning=plan.reasoning))
    s.experiment_graph = g
    return node


@tool
def take_measurement():
    async def execute(task_name: str) -> str:
        """Measure the result of a finished plan (consumes one unit of budget) and add it to the
        reasoning graph. The condition, parent node and reasoning all come from the plan itself.
        The reading is only valid if every step of the plan has been checked off with
        complete_step; otherwise the measurement fails and returns no value.

        Args:
            task_name: Name of the task whose plan you have just finished executing.
        """
        node = measure(task_name)
        s = store_as(LabState)
        out = {
            "id": node.id,
            "task_name": task_name,
            "params": node.params,
            "valid": node.valid,
            "result": node.result,
            "remaining_budget": s.budget - len(s.experiment_graph.experiments),
        }
        if not node.valid:
            out["error"] = (
                f"MEASUREMENT FAILED: the plan for '{task_name}' is not complete, so no reading was "
                f"obtained. Finish the remaining steps, check them off with complete_step, then measure again."
            )
            out["remaining_steps"] = s.task_plans[task_name].remaining
        return json.dumps(out)

    return execute
