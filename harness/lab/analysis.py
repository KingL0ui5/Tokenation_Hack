"""Readout analysis and Bayesian optimisation, as specified in the protocol.

Everything here sees only what the plate reader reports, never the hidden
enzyme model.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from . import config as C

# ---------------------------------------------------------------- readout

def rate_from_reads(times, absorbance, min_points: int = 3, r2_min: float = 0.98) -> dict:
    """Initial rate (A/min) from the longest linear prefix of a progress curve.

    A curve that is non-linear over every window of >= min_points reads, or
    whose reads leave the detector's linear range, is rate-indeterminate.
    """
    t = np.asarray(times, float)
    a = np.asarray(absorbance, float)
    in_range = a < C.DETECTOR_LINEAR_MAX_A
    best = None
    for n in range(len(t), min_points - 1, -1):
        if not in_range[:n].all():
            continue
        slope, intercept = np.polyfit(t[:n], a[:n], 1)
        pred = slope * t[:n] + intercept
        ss_res = float(((a[:n] - pred) ** 2).sum())
        ss_tot = float(((a[:n] - a[:n].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        # A flat line (blank) is linear by definition even if r2 is noisy.
        if r2 >= r2_min or abs(slope) * (t[n - 1] - t[0]) < 0.01:
            best = {"slope_A_per_min": float(slope), "points_used": n, "r2": round(r2, 4)}
            break
    if best is None:
        return {"slope_A_per_min": None, "rate_indeterminate": True,
                "reason": "non-linear or out of detector range across all windows"}
    best["rate_indeterminate"] = False
    return best


def slope_to_rate(slope_a_per_min: float, enzyme_ug: float) -> float:
    """A/min -> uM pNP / min / ug enzyme."""
    mm_per_min = slope_a_per_min / (C.PNP_EPSILON_405 * C.PATH_LENGTH_CM)
    return mm_per_min * 1000.0 / max(enzyme_ug, 1e-12)


def apply_outlier_rule(values: list[float | None], cv_max: float = 0.30) -> dict:
    """Borkowski's rule: if CV of the triplicate > 30%, drop the value farthest
    from the other two and flag it."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return {"mean": vals[0] if vals else None, "sd": None, "cv": None, "kept": vals,
                "dropped": None, "flagged": False}
    arr = np.array(vals)
    mean = arr.mean()
    cv = float(arr.std(ddof=1) / abs(mean)) if abs(mean) > 1e-12 else float("inf")
    dropped = None
    if cv > cv_max and len(arr) >= 3:
        dist = [abs(arr[i] - np.delete(arr, i).mean()) for i in range(len(arr))]
        k = int(np.argmax(dist))
        dropped = float(arr[k])
        arr = np.delete(arr, k)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "cv": round(cv, 4), "kept": arr.round(5).tolist(), "dropped": dropped,
            "flagged": dropped is not None}


def z_prime(pos: list[float], neg: list[float]) -> float | None:
    if len(pos) < 2 or len(neg) < 2:
        return None
    mp, mn = np.mean(pos), np.mean(neg)
    if abs(mp - mn) < 1e-12:
        return float("-inf")
    return float(1 - 3 * (np.std(pos, ddof=1) + np.std(neg, ddof=1)) / abs(mp - mn))


def linear_r2(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# --------------------------------------------------------------- encoding

def encode(cond: dict) -> np.ndarray:
    """One-hot buffer, ordinal pH/levels/temperature scaled to [0, 1]."""
    buf = [1.0 if cond["buffer"] == b else 0.0 for b in C.BUFFERS]
    ph = (cond["pH"] - 7.0) / 3.0
    lv = [C.LEVELS[cond[a]] for a in C.ADDITIVES]
    t = (cond["temperature"] - 25.0) / 20.0
    return np.array(buf + [ph] + lv + [t])


def _grid_matrix():
    if not hasattr(_grid_matrix, "cache"):
        conds = list(C.iter_grid())
        _grid_matrix.cache = (conds, np.stack([encode(c) for c in conds]))
    return _grid_matrix.cache


# --------------------------------------------------------------------- GP

def fit_gp(conditions: list[dict], yields: list[float]):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    X = np.stack([encode(c) for c in conditions])
    y = np.asarray(yields, float)
    dims = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.ones(dims), length_scale_bounds=(0.05, 20.0), nu=1.5)
              + WhiteKernel(0.02, (1e-4, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
                                  random_state=0)
    with warnings.catch_warnings():
        # Length scales pinned at a bound are expected for irrelevant variables (ARD).
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def cross_validated_r2(conditions: list[dict], yields: list[float], folds: int = 5) -> float | None:
    n = len(yields)
    if n < 8:
        return None
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    preds = np.zeros(n)
    y = np.asarray(yields, float)
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        gp = fit_gp([conditions[i] for i in train], y[train])
        preds[test] = gp.predict(np.stack([encode(conditions[i]) for i in test]))
    return float(1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def describe_gp(gp, conditions, yields) -> dict:
    names = [f"buffer={b}" for b in C.BUFFERS] + ["pH", *C.ADDITIVES, "temperature"]
    params = gp.kernel_.get_params()
    ls = np.atleast_1d(params["k1__k2__length_scale"])
    conds, Xg = _grid_matrix()
    mu, sd = gp.predict(Xg, return_std=True)
    k = int(np.argmax(mu))
    return {
        "n_observations": len(yields),
        "kernel": str(gp.kernel_),
        "length_scales": {n: round(float(v), 3) for n, v in zip(names, ls)},
        "noise_variance": round(float(params["k2__noise_level"]), 5),
        "log_marginal_likelihood": round(float(gp.log_marginal_likelihood_value_), 3),
        "predicted_optimum": {"condition": conds[k], "mean_yield": round(float(mu[k]), 4),
                              "sd": round(float(sd[k]), 4)},
        "best_observed": {"condition": conditions[int(np.argmax(yields))],
                          "yield": round(float(np.max(yields)), 4)},
    }


def suggest_ucb(gp, tested: list[dict], n: int, exploitation: float = 1.0,
                exploration: float = math.sqrt(2), excluded: dict | None = None) -> list[dict]:
    """Batch UCB with a Kriging-believer update between picks.

    `excluded` maps a variable to values that must not be proposed, e.g.
    {"buffer": ["PBS"]}.
    """
    conds, Xg = _grid_matrix()
    mask = np.ones(len(conds), bool)
    tested_keys = {C.condition_key(c) for c in tested}
    for i, c in enumerate(conds):
        if C.condition_key(c) in tested_keys:
            mask[i] = False
        elif excluded and any(c[v] in vals for v, vals in excluded.items()):
            mask[i] = False
    picks = []
    X_obs = gp.X_train_.copy()
    y_obs = gp.y_train_ * gp._y_train_std + gp._y_train_mean
    model = gp
    for _ in range(n):
        mu, sd = model.predict(Xg, return_std=True)
        score = exploitation * mu + exploration * sd
        score[~mask] = -np.inf
        k = int(np.argmax(score))
        if not np.isfinite(score[k]):
            break
        picks.append({"condition": conds[k], "predicted_mean": round(float(mu[k]), 4),
                      "predicted_sd": round(float(sd[k]), 4), "ucb": round(float(score[k]), 4)})
        mask[k] = False
        # Kriging believer: pretend we observed the mean, refit without re-optimising.
        X_obs = np.vstack([X_obs, Xg[k]])
        y_obs = np.append(y_obs, mu[k])
        from sklearn.base import clone
        model = clone(gp).set_params(kernel=gp.kernel_, optimizer=None)
        model.fit(X_obs, y_obs)
    return picks


def mutual_information(conditions: list[dict], yields: list[float]) -> dict:
    from sklearn.feature_selection import mutual_info_regression

    X = np.array([[list(C.BUFFERS).index(c["buffer"]), c["pH"],
                   *[C.LEVELS[c[a]] for a in C.ADDITIVES], c["temperature"]] for c in conditions])
    discrete = [True] * X.shape[1]
    mi = mutual_info_regression(X, np.asarray(yields), discrete_features=discrete, random_state=0)
    names = ["buffer", "pH", *C.ADDITIVES, "temperature"]
    return dict(sorted(((n, round(float(v), 4)) for n, v in zip(names, mi)),
                       key=lambda kv: -kv[1]))
