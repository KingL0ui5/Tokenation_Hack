"""Solver registry. Add new architectures to SOLVERS (or pass any solver via `inspect eval --solver`)."""

import numpy as np
from inspect_ai.agent import AgentSubmit, as_solver, react
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store_as

from bo_eval.env import get_env
from bo_eval.state import BOState
from bo_eval.tools import add_reasoning, bayes_opt_suggest, close_branch, run_experiment, submit, view_graph
from bo_eval.tools.bayes_opt import suggest
from bo_eval.tools.experiment import run
from bo_eval.tools.submit import submit_params

INSTRUCTIONS = """You are an autonomous experimentalist searching an experimental space for the optimal configuration.
Every experiment is a node in a reasoning graph; link it to the node it follows from (its parent) with a short
reasoning label. When the evidence shows a branch cannot contain the optimum, close it with close_branch;
closed branches cannot be extended. Before submitting, every experiment you did not continue from must be
closed with a reason explaining why it was not pursued. Keep the number of experiments small. When confident,
call submit()."""


@solver
def init_bo(budget: int, seed: int = 0):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        s = store_as(BOState)
        s.env, s.budget, s.seed = state.metadata["env"], budget, seed * 1000 + state.epoch
        return state

    return solve


def llm_agent(bo: bool = True, graph: bool = True):
    tools = [run_experiment()]
    tools += [bayes_opt_suggest()] if bo else []
    tools += [add_reasoning(), close_branch(), view_graph()] if graph else []
    agent = react(
        prompt=INSTRUCTIONS,
        tools=tools,
        submit=AgentSubmit(tool=submit(), answer_only=True),
    )
    return as_solver(agent)


@solver
def bo_baseline(n_init: int = 3):
    """No LLM: random initial design, then GP-EI until the budget is spent; submit the best observed."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        s = store_as(BOState)
        env = get_env(s.env)
        init = np.random.default_rng(s.seed).choice(len(env.df), size=n_init, replace=False)
        for k in range(s.budget):
            sug = [{"params": env.condition(init[k])}] if k < n_init else suggest(1)
            if not sug:
                break
            run(sug[0]["params"], reasoning="initial design" if k < n_init else "BO suggestion")
        _submit_best()
        return state

    return solve


@solver
def random_baseline():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        s = store_as(BOState)
        env = get_env(s.env)
        rng = np.random.default_rng(s.seed)
        for i in rng.permutation(len(env.df))[: s.budget]:
            run(env.condition(i), reasoning="random")
        _submit_best()
        return state

    return solve


def _submit_best():
    s = store_as(BOState)
    exps = s.graph.experiments
    if exps:
        pick = max if get_env(s.env).goal == "maximize" else min
        submit_params(pick(exps, key=lambda e: e.result).params)  # ty: ignore[invalid-argument-type]


SOLVERS = {
    "react": lambda: llm_agent(bo=True, graph=True),
    "react_no_bo": lambda: llm_agent(bo=False, graph=True),
    "react_no_graph": lambda: llm_agent(bo=True, graph=False),
    "bo": bo_baseline,
    "random": random_baseline,
}
