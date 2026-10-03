"""Black-box experiment environments backed by measured data."""

from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class TabularEnv:
    """Each run snaps to the nearest measured condition and returns a draw from N(mean, sd)."""

    name: str
    description: str
    df: pd.DataFrame
    params: list[str]
    mean_col: str
    sd_col: str
    goal: str = "maximize"

    @cached_property
    def X(self) -> np.ndarray:
        return self.df[self.params].to_numpy(float)

    def encode(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.X.min(0), self.X.max(0)
        return (X - lo) / np.where(hi > lo, hi - lo, 1.0)

    def index(self, params: dict) -> int:
        missing = set(self.params) - set(params)
        if missing:
            raise ValueError(f"Missing parameters: {sorted(missing)}")
        x = np.array([[float(params[p]) for p in self.params]])
        return int(np.argmin(((self.encode(self.X) - self.encode(x)) ** 2).sum(1)))

    def condition(self, i: int) -> dict:
        return {p: float(self.df[p].iloc[i]) for p in self.params}

    def sample(self, i: int, rng: np.random.Generator) -> float:
        row = self.df.iloc[i]
        return float(max(0.0, rng.normal(row[self.mean_col], row[self.sd_col])))

    def true_value(self, i: int) -> float:
        return float(self.df[self.mean_col].iloc[i])

    @cached_property
    def optimum(self) -> int:
        y = self.df[self.mean_col].to_numpy()
        return int(np.argmax(y) if self.goal == "maximize" else np.argmin(y))

    def prompt(self, budget: int) -> str:
        levels = "\n".join(
            f"- {p}: {sorted(self.df[p].unique().tolist())}" for p in self.params
        )
        return (
            f"Goal: {self.goal} {self.description}.\n"
            f"Parameters and their available levels:\n{levels}\n"
            f"Only {len(self.df)} combinations are feasible; requested settings are snapped to the "
            f"nearest feasible condition. Measurements are noisy.\n"
            f"Experiment budget: {budget}. You are scored on finding the true optimum "
            f"using as few experiments as possible."
        )


def _upo_abts() -> TabularEnv:
    params = ["ph", "salt_conc", "cosubstrate_conc", "organic_solvent_conc", "temperature"]
    df = pd.read_csv(DATA / "upo_abts.csv")
    df["rate_var"] = df["rate_sd"] ** 2
    df = df.groupby(params, as_index=False)[["rate_mean", "rate_var"]].mean()
    df["rate_sd"] = np.sqrt(df.pop("rate_var"))
    return TabularEnv(
        name="upo_abts",
        description="the mean specific rate [U/mg] of unspecific peroxygenase (UPO) oxidising ABTS",
        df=df,
        params=params,
        mean_col="rate_mean",
        sd_col="rate_sd",
    )


ENVS = {"upo_abts": _upo_abts}


@cache
def get_env(name: str) -> TabularEnv:
    return ENVS[name]()
