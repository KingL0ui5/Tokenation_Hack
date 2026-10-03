"""Backend for the lab tools: drives Lok's lab_sim scene (MuJoCo rigid-body physics only).
Nothing in the scene has a free joint, so no container can be grasped, knocked over or
otherwise relocated -- the only thing genuinely simulated is the held pipette's nozzle
travelling to, and descending into, named sites (`lab_sim.robot.skills.PipetteSkills`).

There is no liquid, no volume, no composition and no instrument, so none of that is tracked
or reported: a tool only ever succeeds or fails based on real motion outcomes (unreachable,
excess tilt, collision). A spill from a knocked-over container is a planned future feature
(needs free joints on the vessels) and is not faked here.

Truth ledger (`LedgerEntry`): intended / observed for every tool call.
The agent never sees the ledger or the seed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mujoco

TIP_CHANGE_MIN = 5 / 60


@dataclass
class LedgerEntry:
    t_lab: float
    tool: str
    intended: dict
    observed: dict
    events: list[str] = field(default_factory=list)


class LabBackend:
    """One simulated lab session. Create one per sample."""

    APPROACH_CLEAR = 0.04      # hover this far above an opening before descending
    ENTER_DEPTH = 0.055        # slow vertical descent, nozzle enters the opening

    def __init__(self, seed: int = 0):
        logging.getLogger("mink").setLevel(logging.ERROR)
        from lab_sim.robot.skills import PipetteSkills
        from lab_sim.scenes.build_lab import load_model, scene_contract  # Lok's scene + contract

        self.model = load_model()
        self.contract = scene_contract(self.model)              # scene publishes its own contract
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.wells = list(self.contract.wells)
        self.reagents = list(self.contract.reagents)
        # Every site name the agent can address as a container or station.
        self.sites = (set(self.contract.wells) | set(self.contract.reagents.values())
                      | {v["site"] for v in self.contract.waste_bins.values()}
                      | set(self.contract.stations.values()))

        self.skills = PipetteSkills(self.model, self.data, self.contract.obstacles, safe_z=0.22)

        self.clock_min = 0.0
        self.events: list[dict] = []
        self.ledger: list[LedgerEntry] = []
        self.incidents: list[dict] = []        # collisions, logged for later use

    def log(self, kind: str, **detail) -> dict:
        ev = {"t_min": round(self.clock_min, 3), "kind": kind, **detail}
        self.events.append(ev)
        if kind == "collision":
            self.incidents.append(ev)
        return ev

    def pipette(self, source: str, dest: str) -> dict:
        """Aspirate from `source`, dispense over `dest`, with the held pipette (nozzle IK via
        `skills.py`). Reports only real motion outcomes; no liquid is tracked."""
        sk = self.skills
        sk.set_active_point("nozzle")
        moves = []
        self.clock_min += TIP_CHANGE_MIN
        for site in (source, dest):
            r = sk.travel_to(site, self.APPROACH_CLEAR)
            moves.append(("travel", site, r))
            if not r.ok:
                return {"ok": False, "reason": f"{site}: {r.reason}", "moves": moves}
            r = sk.descend(self.ENTER_DEPTH)
            moves.append(("descend", site, r))
            sk.ascend()
            if not r.ok:
                return {"ok": False, "reason": f"{site}: {r.reason}", "moves": moves}
        return {"ok": True, "moves": moves}

    def mix(self, container: str, cycles: int) -> dict:
        """Pipette up and down inside `container`. Reports only real motion outcomes."""
        sk = self.skills
        sk.set_active_point("nozzle")
        r = sk.travel_to(container, self.APPROACH_CLEAR)
        if r.ok:
            r = sk.descend(0.05)
        for _ in range(max(1, cycles)):
            if not r.ok:
                break
            sk.descend(-0.015)
            r = sk.descend(0.015)
        sk.ascend()
        return {"ok": r.ok, "reason": None if r.ok else r.reason}


# Per-sample backends, so tools can be created without passing a backend (e.g. in a
# solver's tool list) and each Inspect sample still gets its own lab.
_BACKENDS: dict[tuple, LabBackend] = {}


def current_backend(**kwargs) -> LabBackend:
    key: tuple = ("default",)
    seed = kwargs.pop("seed", None)
    try:
        from inspect_ai.solver._task_state import sample_state

        state = sample_state()
        if state is not None:
            key = (state.sample_id, state.epoch)
            if seed is None:
                seed = int(state.metadata.get("seed", 0)) if state.metadata else 0
    except Exception:
        pass
    if key not in _BACKENDS:
        _BACKENDS[key] = LabBackend(seed=seed or 0, **kwargs)
    return _BACKENDS[key]
