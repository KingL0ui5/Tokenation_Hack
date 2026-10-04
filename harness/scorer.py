"""Exports each task's experiment plan (the checklist of physical steps the scientist gave
the technician, and how much of it got checked off) so a run can be inspected after the fact.

There is no measured "result" anywhere in this harness -- the simulation only models physical
asset positions, not chemistry -- so there is nothing to build a results graph from; this
reports on plan/procedure completion instead."""
import json
from pathlib import Path

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import store_as

from harness.types.state import LabState

METRICS = {k: [mean(), stderr()] for k in ("n_plans", "n_plans_finished", "n_steps_completed")}


@scorer(metrics=METRICS)
def lab_scorer(graph_dir: str | None = "logs/graphs"):
    """n_plans: tasks the scientist planned. n_plans_finished: of those, how many the
    technician fully checked off. n_steps_completed: total checklist steps completed."""

    async def score(state: TaskState, target: Target) -> Score:
        lab_state = store_as(LabState)
        plans = lab_state.task_plans
        n_finished = sum(1 for p in plans.values() if p.finished)
        n_steps = sum(sum(p.completed) for p in plans.values())

        if graph_dir:
            out = Path(graph_dir) / f"{state.sample_id}_epoch{state.epoch}_plans"
            out.parent.mkdir(parents=True, exist_ok=True)
            text = "\n\n".join(f"## {name}\n{plan.to_text()}" for name, plan in plans.items()) or "(no plans created)"
            out.with_suffix(".md").write_text(text + "\n")
            out.with_suffix(".json").write_text(
                json.dumps({k: v.model_dump() for k, v in plans.items()}, indent=2)
            )

        return Score(
            value={"n_plans": len(plans), "n_plans_finished": n_finished, "n_steps_completed": n_steps},
            explanation="\n\n".join(f"### {name}\n{plan.to_text()}" for name, plan in plans.items()),
            metadata={"task_plans": {k: v.model_dump() for k, v in plans.items()}},
        )

    return score
