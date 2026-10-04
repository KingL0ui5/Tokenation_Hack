import json

import numpy as np
from inspect_ai.tool import tool
from inspect_ai.util import store_as
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bo_eval.env import get_env
from harness.types.state import LabState


def suggest(n: int = 1, xi: float = 0.01, avoid_closed: bool = True) -> list[dict]:
    """GP + expected improvement over all feasible, not-yet-run conditions. Only valid readings
    inform the surrogate: a measurement taken on an incomplete plan carries no value."""
    lab_state = store_as(LabState)
    env, exps = get_env(lab_state.env), lab_state.experiment_graph.measured
    C = env.encode(env.X)
    run_idx = [env.index(e.params) for e in exps]
    mask = np.ones(len(C), bool)
    mask[run_idx] = False
    mask0 = mask.copy()      # not-yet-run, before any closed-branch filtering

    # The incumbent is where the next experiment should normally hang in the reasoning graph.
    incumbent = lab_state.experiment_graph.best(env.goal)
    parent = incumbent.id if incumbent else "root"

    if len(exps) < 2:
        rng = np.random.default_rng([lab_state.seed, len(exps), 1])
        pick = rng.choice(np.flatnonzero(mask), size=n, replace=False)

        return [{"params": env.condition(i), "reason": "random initial design",
                 "suggested_parent": parent} for i in pick]

    Xo = C[run_idx]
    sign = 1.0 if env.goal == "maximize" else -1.0
    y = sign * np.array([e.result for e in exps])
    kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(C.shape[1])) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2).fit(Xo, y)
    mu, sd = gp.predict(C, return_std=True)
    z = (mu - y.max() - xi) / np.maximum(sd, 1e-9)
    ei = (mu - y.max() - xi) * norm.cdf(z) + sd * norm.pdf(z)

    if avoid_closed:
        nearest = ((C[:, None, :] - Xo[None]) ** 2).sum(-1).argmin(1)
        mask &= ~np.array([exps[j].closed for j in nearest])
    order = [i for i in np.argsort(-ei) if mask[i]][:n]
    if not order:   # every candidate sits next to a closed branch; better a suggestion than none
        order = [i for i in np.argsort(-ei) if mask0[i]][:n]
    return [
        {"params": env.condition(i), "predicted_mean": float(sign * mu[i]),
         "predicted_sd": float(sd[i]), "expected_improvement": float(ei[i]),
         "suggested_parent": parent,
         "incumbent": {"node": parent, "result": float(incumbent.result)} if incumbent else None}
        for i in order
    ]


@tool
def bayes_opt_suggest():
    async def execute(n: int = 1, avoid_closed: bool = True) -> str:
        """Suggest the next experiment(s) by Bayesian optimisation (GP + expected improvement) over all results so far. Does not consume budget.

        Args:
            n: Number of suggestions to return.
            avoid_closed: Skip candidates whose nearest observed experiment lies on a closed branch.
        """
        return json.dumps(suggest(n, avoid_closed=avoid_closed))

    return execute
