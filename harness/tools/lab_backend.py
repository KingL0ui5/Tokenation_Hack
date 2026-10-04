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
from lab_sim.robot.skills import PipetteSkills
from lab_sim.scenes.build_lab import load_model, scene_contract

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
        self._incident_cursor = 0              # how many of skills.incidents we've drained

    def log(self, kind: str, **detail) -> dict:
        ev = {"t_min": round(self.clock_min, 3), "kind": kind, **detail}
        self.events.append(ev)
        if kind == "collision":
            self.incidents.append(ev)
        return ev

    def _drain_incidents(self) -> list[dict]:
        """Pull any new held-object incidents the skill layer recorded (robot/skills.py's
        PipetteSkills._record_incidents) since the last check, and log each as a collision
        event -- the same ledger/incidents channel a real collision would use."""
        new = self.skills.incidents[self._incident_cursor:]
        self._incident_cursor = len(self.skills.incidents)
        for inc in new:
            self.log("collision", carried=inc["carried"], other=inc["other"], dist=inc["dist"])
        return new

    def pipette(self, source: str, dest: str) -> dict:
        """Aspirate from `source`, dispense over `dest`, with a fresh disposable tip (nozzle IK
        via `skills.py`): mount a tip, visit both sites, eject the tip. Reports only real motion
        outcomes; no liquid is tracked."""
        sk = self.skills
        if sk.has_tip:                       # leftover from a prior failed transfer
            sk.eject_tip()
        tip = sk.pick_up_tip()
        moves = [("pick_up_tip", None, tip)]
        if not tip.ok:                        # empty box, or a "tip not seated" fault
            return {"ok": False, "reason": f"tip: {tip.reason}", "moves": moves,
                    "tip_slot": None, "touched": []}
        tip_slot = sk._cur_tip_slot
        self.clock_min += TIP_CHANGE_MIN
        for site in (source, dest):
            r = sk.travel_to(site, self.APPROACH_CLEAR)
            moves.append(("travel", site, r))
            bad = self._drain_incidents()
            if bad:
                return {"ok": False, "reason": f"collision: {bad[0]['carried']} vs {bad[0]['other']}",
                        "moves": moves, "tip_slot": tip_slot, "touched": list(sk.tip_contacts),
                        "incidents": bad}
            if not r.ok:
                return {"ok": False, "reason": f"{site}: {r.reason}", "moves": moves,
                        "tip_slot": tip_slot, "touched": list(sk.tip_contacts)}
            r = sk.descend(self.ENTER_DEPTH)
            moves.append(("descend", site, r))
            sk.ascend()
            bad = self._drain_incidents()
            if bad:
                return {"ok": False, "reason": f"collision: {bad[0]['carried']} vs {bad[0]['other']}",
                        "moves": moves, "tip_slot": tip_slot, "touched": list(sk.tip_contacts),
                        "incidents": bad}
            if not r.ok:
                return {"ok": False, "reason": f"{site}: {r.reason}", "moves": moves,
                        "tip_slot": tip_slot, "touched": list(sk.tip_contacts)}
            sk.note_tip_contact(site)
        touched = list(sk.tip_contacts)
        moves.append(("eject_tip", None, sk.eject_tip()))
        return {"ok": True, "moves": moves, "tip_slot": tip_slot, "touched": touched}

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
        bad = self._drain_incidents()
        if bad:
            return {"ok": False, "reason": f"collision: {bad[0]['carried']} vs {bad[0]['other']}",
                    "incidents": bad}
        return {"ok": r.ok, "reason": None if r.ok else r.reason}

    def move_tube(self, reagent: str, dest_site: str, grip_gap: float = 0.022) -> dict:
        """Carry a reagent's tube by the gripper to `dest_site` (e.g. a spare rack hole). Parks
        the pipette first if it's mounted, and re-mounts it afterward. `grip_gap` narrows the
        jaws before threading down between packed tubes, so the open gripper doesn't bump a
        neighbour on the way in (the approach only, not the final grasp width -- grasp() picks
        its own gap from the tube's own radius)."""
        sk = self.skills
        moves = []
        source_site = reagent if reagent.startswith("reagent_") else f"reagent_{reagent}"
        grip_site = f"tube_grip_{reagent[len('reagent_'):] if reagent.startswith('reagent_') else reagent}"
        was_held = sk.held
        if was_held:
            moves.append(("put_down_pipette", None, sk.put_down_pipette()))

        def fail(reason: str) -> dict:
            bad = self._drain_incidents()
            return {"ok": False, "reason": reason, "moves": moves, "incidents": bad}

        sk.close_gripper(grip_gap)
        r = sk.travel_to(grip_site, clearance=0.04)
        moves.append(("travel", grip_site, r))
        bad = self._drain_incidents()
        if bad:
            return fail(f"collision: {bad[0]['carried']} vs {bad[0]['other']}")
        if not r.ok:
            return fail(f"{grip_site}: {r.reason}")
        r = sk.descend(0.04)
        moves.append(("descend", grip_site, r))
        bad = self._drain_incidents()
        if bad:
            return fail(f"collision: {bad[0]['carried']} vs {bad[0]['other']}")
        if not r.ok:
            return fail(f"{grip_site}: {r.reason}")

        g = sk.grasp(reagent)
        moves.append(("grasp", reagent, g))
        if not g.ok:
            return fail(g.reason)

        p = sk.place(reagent, dest_site)
        moves.append(("place", dest_site, p))
        bad = self._drain_incidents()
        if bad:
            return fail(f"collision: {bad[0]['carried']} vs {bad[0]['other']}")
        if not p.ok:
            return fail(f"{dest_site}: {p.reason}")

        if was_held:
            moves.append(("pick_up_pipette", None, sk.pick_up_pipette()))
        return {"ok": True, "moves": moves, "source": source_site, "dest": dest_site}


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
