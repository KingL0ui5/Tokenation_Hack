"""Scores a run the way `bo_eval.scorer` does -- regret against the environment's true optimum,
from the single reasoning graph the agents built -- and exports that graph as mermaid + JSON.

What this harness adds is the plan layer: a measurement is only valid if the technician finished
the scientist's checklist first, so invalid (reading-less) experiments are reported alongside.
"""
import json
from pathlib import Path

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.types.state import LabState

METRICS = {k: [mean(), stderr()]
           for k in ("found_optimal", "n_experiments", "regret", "n_invalid", "n_plans_finished")}


@scorer(metrics=METRICS)
def lab_scorer(graph_dir: str | None = "logs/graphs", tolerance: float = 0.0):
    """found_optimal: submitted (else best measured) condition is within `tolerance` relative regret
    of the true optimum. n_experiments: experiments run (budget spent). regret: relative gap in true
    mean to the optimum. n_invalid: experiments measured on an incomplete plan, so without a reading.
    n_plans_finished: plans the technician fully checked off."""

    async def score(state: TaskState, target: Target) -> Score:
        s = store_as(LabState)
        env, g = get_env(s.env), s.experiment_graph
        exps, measured = g.experiments, g.measured
        plans = s.task_plans

        answer = s.submission
        if answer is None and measured:
            pick = max if env.goal == "maximize" else min
            answer = pick(measured, key=lambda e: e.result).params
        best = env.true_value(env.optimum)
        regret = 1.0 if answer is None else abs(best - env.true_value(env.index(answer))) / abs(best)
        hits = [e.id for e in measured if env.index(e.params) == env.optimum]  # ty: ignore[invalid-argument-type]

        mermaid = g.to_mermaid()
        if graph_dir:
            out = Path(graph_dir) / f"{state.sample_id}_epoch{state.epoch}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".md").write_text(
                f"```mermaid\n{mermaid}\n```\n\n```\n{g.to_text()}\n```\n\n"
                + "\n\n".join(f"## {name}\n{plan.to_text()}" for name, plan in plans.items())
                + "\n"
            )
            out.with_suffix(".json").write_text(g.model_dump_json(indent=2))

        return Score(
            value={
                "found_optimal": float(regret <= tolerance),
                "n_experiments": len(exps),
                "regret": regret,
                "n_invalid": len(exps) - len(measured),
                "n_plans_finished": sum(1 for p in plans.values() if p.finished),
            },
            answer=json.dumps(answer),
            explanation=f"```mermaid\n{mermaid}\n```",
            metadata={
                "optimum": env.condition(env.optimum),
                "optimum_value": best,
                "first_experiment_at_optimum": hits[0] if hits else None,
                "graph": g.model_dump(),
                "task_plans": {k: v.model_dump() for k, v in plans.items()},
            },
        )

    return score
