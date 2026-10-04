"""Prompt context shared by both agents: the experiment space they are searching and what
the simulated lab physically contains. Without this neither agent knows the parameter names
it must report, nor the site names it must address, and the technician stalls asking for them.
"""

from inspect_ai.model import ChatMessage

from bo_eval.env import get_env
from harness.tools.lab_backend import current_backend


def experiment_space(env: str, budget: int) -> str:
    """Goal, parameters, feasible levels and budget for the environment under study."""
    return get_env(env).prompt(budget)


def lab_inventory() -> str:
    """The sites the pipette can actually reach, taken from the scene's own contract."""
    b = current_backend()
    return (
        "Sites in this lab (use these exact names):\n"
        f"- wells: {', '.join(w.removeprefix('well_') for w in b.wells)}\n"
        f"- reagent stock tubes: {', '.join(b.reagents)}\n"
        "Volumes are intended only: no liquid, volume or composition is tracked, so a tool call "
        "reports only whether the pipette motion succeeded."
    )


def briefed(messages: list[ChatMessage], marker: str) -> bool:
    """True once a role's standing briefing is in the transcript, so it is sent once and not
    re-appended every round."""
    return any(marker in m.text for m in messages if m.role == "system")
