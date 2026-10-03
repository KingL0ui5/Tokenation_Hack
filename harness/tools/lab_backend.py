"""Backend for the lab tools: drives Lok's lab_sim scene and tracks liquid positions only
(no chemistry, no instrument -- the simulation models physical asset positions and nothing
else). Follows `lab_sim/virtual-lab-plan.md`:

- Layer 0-1 (scene + motion): the scene comes from `lab_sim.scenes.build_lab.load_model()`,
  unchanged. Motion is mink IK on `attachment_site` with collision avoidance (as in
  `lab_sim/robot/ik_smoke_test.py`), tracked by the Panda's position servos. The
  controller adds gravity feed-forward each step, as the real Panda controller does;
  without it the servos sag several mm, more than a 7 mm well tolerates.
- Container state (plan's `Container`): every well, tube and reservoir in the scene, in
  two copies: `actual` (what physically happened, hidden) and `nominal` (what was
  commanded, the agent's own bookkeeping).
- Where liquid goes: decided by the tip's real position at dispense. Inside the target
  -> target; inside a neighbouring well -> the neighbour; otherwise a spill.
- Truth ledger (plan's `LedgerEntry`): intended / actual / observed for every tool call.

The agent never sees `actual`, the ledger or the seeds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

import mink
import mujoco
import numpy as np

EE_SITE = "attachment_site"
ARM_JOINTS = 7
SAFE_Z = 0.20               # tip travel height (clears reader/incubator tops at 0.10 m)
PLATE_HOVER = 0.08          # hover between wells; keeps the 20 cm-wide hand above the incubator
APPROACH_STEP = 0.025       # descend via a waypoint this far above the target
PARK = (0.38, 0.14, 0.30)   # where the arm waits, clear of the plate_top camera
TIP_CAPACITY_UL = 1000.0
LAB_SECONDS_PER_MOVE = 1.5

# Vessel geometry from lab_sim/scenes/build_lab.py (radius, height in metres).
VESSELS = {"well": (0.007, 0.012), "tube": (0.006, 0.040), "reservoir": (0.025, 0.050)}
RESERVOIR_START_UL = 20_000.0


@dataclass
class Container:
    id: str                         # "well_B3", "rack_2", "reservoir_enzyme"
    kind: str                       # well | tube | reservoir
    capacity_ul: float
    volume_ul: float = 0.0
    composition: dict[str, float] = field(default_factory=dict)  # source reagent -> uL
    slot: str = ""                  # scene site name
    mixed: bool = True

    def add(self, composition: dict[str, float], volume: float, t_min: float) -> None:
        if volume <= 0:
            return
        total = sum(composition.values()) or 1.0
        for k, v in composition.items():
            self.composition[k] = self.composition.get(k, 0.0) + volume * v / total
        self.volume_ul += volume
        self.mixed = self.volume_ul == volume  # adding to existing liquid leaves it unmixed

    def remove(self, volume: float) -> dict[str, float]:
        volume = min(volume, self.volume_ul)
        if self.volume_ul <= 0:
            return {}
        frac = volume / self.volume_ul
        taken = {k: v * frac for k, v in self.composition.items()}
        for k in self.composition:
            self.composition[k] -= taken[k]
        self.volume_ul -= volume
        if self.volume_ul < 1e-6:
            self.volume_ul, self.composition = 0.0, {}
        return taken

    def summary(self) -> dict:
        d = asdict(self)
        d["volume_ul"] = round(self.volume_ul, 2)
        d["composition"] = {k: round(v, 2) for k, v in self.composition.items() if v > 1e-6}
        return d


@dataclass
class LedgerEntry:
    t_lab: float
    tool: str
    intended: dict
    actual: dict
    observed: dict
    events: list[str] = field(default_factory=list)


class LabBackend:
    """One simulated lab session. Create one per sample."""

    def __init__(self, seed: int = 0, budget_wells: int = 24,
                 reagent_budget_ul: dict[str, float] | None = None,
                 plate_offset_mm: tuple[float, float] = (0.0, 0.0),
                 pipette_bias: float | None = None):
        logging.getLogger("mink").setLevel(logging.ERROR)
        from lab_sim.scenes.build_lab import load_model, scene_contract  # Lok's scene + contract

        streams = np.random.SeedSequence(seed).spawn(2)
        self.rng = {k: np.random.default_rng(s) for k, s in zip(["reset", "command"], streams)}
        self.model = load_model()
        self.contract = scene_contract(self.model)              # scene publishes its own contract
        self.vessels = {t: (v.radius_m, v.height_m) for t, v in self.contract.vessels.items()}
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.ee = self.model.site(EE_SITE).id
        self.down = self.data.site_xmat[self.ee].reshape(3, 3).copy()
        self._init_ik()

        # short well labels ("B3") for the default single plate
        self.wells = sorted((s[len("well_"):] for s in self.contract.wells if s.startswith("well_")),
                            key=lambda w: (w[0], int(w[1:])))
        # Where the robot believes each well is; a non-zero offset mis-seats the plate.
        self._believed = {w: self.site_pos(f"well_{w}") for w in self.wells}
        self.plate_offset = np.array([*plate_offset_mm, 0.0]) / 1000.0

        # containers: wells + reagent stock tubes (the reagent sources), all from the contract
        well_cap = self.contract.vessels["well"].capacity_ul
        tube_cap = self.contract.vessels["tube"].capacity_ul
        self.actual: dict[str, Container] = {
            f"well_{w}": Container(f"well_{w}", "well", well_cap, slot=f"well_{w}") for w in self.wells}
        for reagent, site in self.contract.reagents.items():
            c = Container(site, "tube", tube_cap, slot=site)
            c.add({reagent: 1.0}, REAGENT_START_UL, 0.0)
            self.actual[site] = c
        self.nominal = {k: Container(**{**asdict(v), "composition": dict(v.composition)})
                        for k, v in self.actual.items()}
        self.reagents = list(self.contract.reagents)

        r = self.rng["reset"]
        self.pipette_bias = float(r.normal(0, 0.01)) if pipette_bias is None else pipette_bias
        self.budget = {"wells": budget_wells,
                       "reagent_ul": dict(reagent_budget_ul or {k: REAGENT_START_UL for k in self.reagents})}
        self.wells_used: set[str] = set()
        self.reagent_used_ul: dict[str, float] = {}

        self.clock_min = 0.0
        self.events: list[dict] = []
        self.ledger: list[LedgerEntry] = []
        self.incidents: list[dict] = []        # spills/collisions logged for later use; not yet acted on
        self.liquid_geom = {w: self.model.geom(f"liquid_well_{w}").id for w in self.wells}
        self._liquid_base = {w: self.model.geom_pos[g].copy() for w, g in self.liquid_geom.items()}

        # Skills for the held-pipette motion path (IK targets the pipette nozzle, not the EE).
        from lab_sim.robot.skills import PipetteSkills
        self.skills = PipetteSkills(self.model, self.data, self.contract.obstacles, safe_z=0.22)

    # ================================================================ geometry
    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(name).id].copy()

    def actual_pos(self, container_id: str) -> np.ndarray:
        """Where a container really is (wells move with a mis-seated plate)."""
        p = self.site_pos(container_id)
        return p + self.plate_offset if container_id.startswith("well_") else p

    def believed_pos(self, container_id: str) -> np.ndarray:
        if container_id.startswith("well_"):
            return self._believed[container_id[5:]].copy()
        return self.site_pos(container_id)

    # ================================================================== motion
    def _init_ik(self) -> None:
        m = self.model
        self.cfg = mink.Configuration(m)
        self.frame_task = mink.FrameTask(EE_SITE, "site", position_cost=1.0, orientation_cost=1.0,
                                         lm_damping=1.0)
        self.posture_task = mink.PostureTask(m, cost=1e-2)
        self.posture_task.set_target(self.data.qpos.copy())
        robot = mink.get_subtree_geom_ids(m, m.body("link0").id)
        obstacles = [m.geom(n).id for n in self.contract.obstacles]
        self.limits = [mink.ConfigurationLimit(m), mink.CollisionAvoidanceLimit(
            m, geom_pairs=[(robot, obstacles)], minimum_distance_from_collisions=0.005,
            collision_detection_distance=0.05)]
        self.arm_geoms, self.obstacle_geoms = set(robot), set(obstacles)

    def _step(self) -> None:
        self.data.qfrc_applied[:ARM_JOINTS] = self.data.qfrc_bias[:ARM_JOINTS]  # gravity feed-forward
        mujoco.mj_step(self.model, self.data)

    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.ee].copy()

    def solve_ik(self, target) -> tuple[np.ndarray, float]:
        self.cfg.update(self.data.qpos)
        self.frame_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.down), np.asarray(target, float)))
        err = np.inf
        for _ in range(400):
            vel = mink.solve_ik(self.cfg, [self.frame_task, self.posture_task], 0.02, "daqp",
                                limits=self.limits)
            self.cfg.integrate_inplace(vel, 0.02)
            e = self.frame_task.compute_error(self.cfg)
            err = float(np.linalg.norm(e[:3]))
            if err < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
        return self.cfg.q[:ARM_JOINTS].copy(), err

    def move_to(self, target, duration_s: float = 0.25, settle_s: float = 0.05) -> dict:
        target = np.asarray(target, float)
        q_goal, ik_err = self.solve_ik(target)
        if ik_err > 2e-3:
            self.log("ik_failure", target=target.round(4).tolist(), residual_mm=round(ik_err * 1e3, 2))
            return {"ok": False, "reason": "unreachable"}
        dt = self.model.opt.timestep
        q0 = self.data.ctrl[:ARM_JOINTS].copy()
        n = max(1, int(duration_s / dt))
        hits: set[str] = set()
        for k in range(n):
            s = (k + 1) / n
            self.data.ctrl[:ARM_JOINTS] = q0 + (3 * s**2 - 2 * s**3) * (q_goal - q0)
            self._step()
            hits |= self._contacts()
        for k in range(int(1.0 / dt)):
            self._step()
            hits |= self._contacts()
            if k * dt >= settle_s and np.abs(self.data.qvel[:ARM_JOINTS]).max() < 2e-3:
                break
        self.clock_min += LAB_SECONDS_PER_MOVE / 60
        if hits:
            self.log("collision", obstacles=sorted(hits))
        return {"ok": True, "tracking_error_mm": round(float(np.linalg.norm(self.tip() - target)) * 1e3, 3)}

    def _contacts(self) -> set[str]:
        hits = set()
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if c.dist < -1e-4 and ((g1 in self.arm_geoms and g2 in self.obstacle_geoms)
                                   or (g2 in self.arm_geoms and g1 in self.obstacle_geoms)):
                hits.add(self.model.geom(g2 if g1 in self.arm_geoms else g1).name)
        return hits

    def travel(self, target, hover: float = SAFE_Z) -> dict:
        target = np.asarray(target, float)
        p = self.tip()
        if p[2] < hover - 1e-3:
            r = self.move_to([p[0], p[1], hover], duration_s=0.15)
            if not r["ok"]:
                return r
        for waypoint in ([target[0], target[1], hover],
                         [target[0], target[1], target[2] + APPROACH_STEP]):
            if waypoint[2] > target[2] + 1e-6:
                r = self.move_to(waypoint)
                if not r["ok"]:
                    return r
        return self.move_to(target, duration_s=0.15, settle_s=0.2)

    def park(self) -> None:
        p = self.tip()
        self.move_to([p[0], p[1], SAFE_Z], duration_s=0.15)
        self.move_to(PARK)

    # ================================================================ liquids
    def log(self, kind: str, **detail) -> dict:
        ev = {"t_min": round(self.clock_min, 3), "kind": kind, **detail}
        self.events.append(ev)
        if kind in ("spill", "wrong_well", "collision", "ik_failure"):
            self.incidents.append(ev)
        return ev

    def _pipetting_error(self, volume: float) -> float:
        r = self.rng["command"]
        return max(0.0, volume * (1 + self.pipette_bias + r.normal(0, 0.01)) + r.normal(0, 0.15))

    def _resolve_landing(self, container_id: str, tip_xy=None) -> str | None:
        """Where liquid released over `container_id` really goes. `tip_xy` defaults to the
        hand EE (old motion path); the skills-based pipette passes the pipette nozzle xy."""
        tip = self.tip()[:2] if tip_xy is None else np.asarray(tip_xy)[:2]
        radius = self.vessels[self.actual[container_id].kind][0]
        if container_id.startswith("well_"):
            d = {f"well_{w}": float(np.linalg.norm(tip - self.actual_pos(f"well_{w}")[:2])) for w in self.wells}
            near = min(d, key=d.get)
            return near if d[near] <= radius else None
        return container_id if np.linalg.norm(tip - self.actual_pos(container_id)[:2]) <= radius else None

    APPROACH_CLEAR = 0.04      # hover this far above an opening before descending
    ENTER_DEPTH = 0.055        # slow vertical descent, nozzle enters the opening

    def pipette(self, source: str, dest: str, volume: float) -> dict:
        """Held-pipette path: aspirate from source, dispense over dest, via robot/skills.py
        (IK targets the pipette nozzle). Repeats for >1 tip volume. Returns per-move tip
        positions/errors in `moves` and liquid events in `events`; never raises."""
        sk = self.skills
        sk.set_active_point("nozzle")
        actual_dest, delivered, events, moves = dest, 0.0, [], []
        self.clock_min += 5 / 60   # tip change
        remaining = volume
        while remaining > 1e-9:
            chunk = min(remaining, TIP_CAPACITY_UL)
            remaining -= chunk

            # --- aspirate from source ---
            r = sk.travel_to(source, self.APPROACH_CLEAR); moves.append(("travel", source, r))
            if not r.ok:
                return {"ok": False, "reason": f"source: {r.reason}", "delivered_ul": delivered, "moves": moves}
            r = sk.descend(self.ENTER_DEPTH); moves.append(("descend", source, r))
            if not r.ok:
                sk.ascend()
                return {"ok": False, "reason": f"source descend: {r.reason}", "delivered_ul": delivered, "moves": moves}
            if self._resolve_landing(source, sk.tip()[:2]) != source:
                self.log("aspirate_miss", source=source)
                sk.ascend()
                return {"ok": False, "reason": f"tip not inside {source}", "delivered_ul": delivered, "moves": moves}
            taken = self.actual[source].remove(self._pipetting_error(chunk))
            moves.append(("ascend", source, sk.ascend()))

            # --- dispense over dest ---
            r = sk.travel_to(dest, self.APPROACH_CLEAR); moves.append(("travel", dest, r))
            if not r.ok:
                self.log("spill", container=dest, volume_ul=round(sum(taken.values()), 2), reason=r.reason)
                return {"ok": False, "reason": f"dest: {r.reason}", "delivered_ul": delivered, "moves": moves}
            r = sk.descend(self.ENTER_DEPTH); moves.append(("descend", dest, r))
            landed = self._resolve_landing(dest, sk.tip()[:2]) if r.ok else None
            amount = sum(taken.values())
            if landed is None:
                events.append(self.log("spill", container=dest, volume_ul=round(amount, 2)))
            else:
                if landed != dest:
                    events.append(self.log("wrong_well", intended=dest, actual=landed, volume_ul=round(amount, 2)))
                self.actual[landed].add(taken, amount, self.clock_min)
                self._show_level(landed)
                actual_dest, delivered = landed, delivered + amount
            moves.append(("ascend", dest, sk.ascend()))
        return {"ok": True, "actual_dest": actual_dest, "delivered_ul": delivered,
                "events": events, "moves": moves}

    # ================================================================ visuals
    def _show_level(self, container_id: str) -> None:
        if not container_id.startswith("well_"):
            return
        w = container_id[5:]
        c = self.actual[container_id]
        half = max(1e-4, min(1.0, c.volume_ul / c.capacity_ul) * self.vessels["well"][1] * 0.45)
        g = self.liquid_geom[w]
        self.model.geom_size[g][1] = half
        self.model.geom_pos[g][2] = self._liquid_base[w][2] + half


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
