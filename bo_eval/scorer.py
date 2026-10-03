import json
from pathlib import Path

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState

METRICS = {k: [mean(), stderr()] for k in ("found_optimal", "n_experiments", "regret")}


@scorer(metrics=METRICS)
def bo_scorer(tolerance: float = 0.0, graph_dir: str | None = "logs/graphs"):
    """found_optimal: submitted (else best observed) condition is within `tolerance` relative regret of the true optimum.
    n_experiments: experiments run. regret: relative gap in true mean to the optimum."""

    async def score(state: TaskState, target: Target) -> Score:
        s = store_as(BOState)
        env, g = get_env(s.env), s.graph
        exps = g.experiments
        answer = s.submission
        if answer is None and exps:
            pick = max if env.goal == "maximize" else min
            answer = pick(exps, key=lambda e: e.result).params
        best = env.true_value(env.optimum)
        regret = 1.0 if answer is None else abs(best - env.true_value(env.index(answer))) / abs(best)
        hits = [e.id for e in exps if env.index(e.params) == env.optimum]

        mermaid = g.to_mermaid()
        if graph_dir:
            out = Path(graph_dir) / f"{state.sample_id}_epoch{state.epoch}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".md").write_text(f"```mermaid\n{mermaid}\n```\n\n```\n{g.to_text()}\n```\n")
            out.with_suffix(".json").write_text(g.model_dump_json(indent=2))

        return Score(
            value={"found_optimal": float(regret <= tolerance), "n_experiments": len(exps), "regret": regret},
            answer=json.dumps(answer),
            explanation=f"```mermaid\n{mermaid}\n```",
            metadata={
                "optimum": env.condition(env.optimum),
                "optimum_value": best,
                "first_experiment_at_optimum": hits[0] if hits else None,
                "graph": g.model_dump(),
            },
        )

    return score
