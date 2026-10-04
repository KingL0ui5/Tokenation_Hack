"""Technician tools for acting on lab failures: see a collision, then redo the plan.

- check_collisions reads the collisions LabBackend logged (`backend.log("collision", ...)`, which
  Lok's held-object collision checks feed) and reports what was carried and what it hit.
- redo_plan unchecks every step of a plan so the technician starts the attempt over, instead of
  measuring a spoiled one. It costs no budget; the redo is logged as a "redo" lab event.

Read-only on the simulation: nothing here moves the robot or changes the scene.

Usage:
    tools = technician_tools()          # per-sample backend, as lab_tools()
    tools = technician_tools(backend)   # or bind an explicit LabBackend
"""

from __future__ import annotations

import json

from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import store_as

from harness.tools.lab_backend import LabBackend, current_backend
from harness.types.state import LabState


def technician_tools(backend: LabBackend | None = None) -> list[Tool]:

    def B() -> LabBackend:
        return backend if backend is not None else current_backend()

    @tool
    def check_collisions() -> Tool:
        async def execute(since_min: float = 0.0) -> str:
            """Report collisions the robot detected: what it was carrying and what it hit.
            A collision spoils the current attempt -- call redo_plan and redo it.

            Args:
                since_min: Only report collisions at or after this lab time (minutes), e.g. the
                    lab_time_min when you started the current plan.
            """
            b = B()
            hits = [{"t_min": e["t_min"], "carried": e.get("carried"), "hit": e.get("other")}
                    for e in b.incidents if e["t_min"] >= since_min]
            return json.dumps({"lab_time_min": round(b.clock_min, 2),
                               "collision": bool(hits), "collisions": hits})
        return execute

    @tool
    def redo_plan() -> Tool:
        async def execute(task_name: str, reason: str) -> str:
            """Start a plan's attempt over after something went wrong (e.g. a collision): every
            step is unchecked so you redo them from step 1, then measure. Costs no budget --
            use it instead of measuring a spoiled attempt.

            Args:
                task_name: The plan to redo.
                reason: What went wrong.
            """
            s = store_as(LabState)
            if task_name not in s.task_plans:
                raise ToolError(f"No plan exists for task '{task_name}'.")
            plan = s.task_plans[task_name]
            plan.completed = [False] * len(plan.steps)
            s.task_plans[task_name] = plan
            b = B()
            b.log("redo", task=task_name, reason=reason)
            return json.dumps({"task_name": task_name, "lab_time_min": round(b.clock_min, 2),
                               "steps_to_redo": plan.steps})
        return execute

    return [check_collisions(), redo_plan()]


__all__ = ["technician_tools"]
