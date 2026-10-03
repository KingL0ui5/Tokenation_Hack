"""Physical layer on Lok's lab_sim scene (lab_sim/scenes/lab.xml).

This replaces harness's own MuJoCo scene. It loads Lok's lab bench unchanged and
adds only what the enzyme protocol needs and the scene does not yet have:

- gravity compensation on the arm links (without it the position servos sag
  4-11 mm, more than the tip can afford over a 7 mm-radius well);
- a stock rack for reagents with no reservoir in the scene (protocol buffers,
  salts, glycerol, product standard, water). Reagents that the scene already has
  a reservoir for (`reservoir_enzyme`, `reservoir_substrate`, ...) use it.

Control follows lab_sim's plan and IK smoke test: mink differential IK on
`attachment_site` with a posture task, joint limits and collision avoidance
against the bench obstacles; the solution is tracked by the Panda's position
servos in full physics. The pipette tip is taken to be `attachment_site`
(lab_sim has no pipette-grasp primitive yet).

Where liquid goes follows lab_sim's noise-layer plan: the tip's actual position
at dispense decides it. Inside the target well -> target; inside a neighbouring
well -> the neighbour; otherwise a spill. Every dispense is written to a ledger
of intended vs actual (the truth ledger the agent never sees).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import mink
import mujoco
import numpy as np

from . import config as C

LAB_SIM = Path(__file__).resolve().parents[2] / "lab_sim"
SCENE = LAB_SIM / "scenes" / "lab.xml"
EE_SITE = "attachment_site"
ARM_JOINTS = 7
ARM_BODIES = ["link1", "link2", "link3", "link4", "link5", "link6", "link7",
              "hand", "left_finger", "right_finger"]
OBSTACLES = ["bench", "plate_collision", "incubator", "plate_reader", "tube_rack",
             "pipette_holder", "waste_bin", "stock_rack"]
SAFE_Z = 0.20              # travel height of the tip (clears reader/incubator tops at 0.10 m)
PLATE_HOVER = 0.08         # hover between wells; keeps the 20 cm-wide hand above the incubator (top 0.10 m)
LAB_SECONDS_PER_MOVE = 1.5
APPROACH_STEP = 0.025      # final descent waypoint above a target
PARK = (0.38, 0.14, 0.30)  # where home() parks the tip
WELL_RADIUS = 0.007        # lab_sim build_lab.WELL_R
WELL_HEIGHT = 0.012        # lab_sim build_lab.WELL_H
RESERVOIR_RADIUS = 0.025   # lab_sim build_lab.RES_R
STOCK_TUBE_RADIUS = 0.008
STOCK_ROWS_Y = (-0.150, -0.195)   # free strip between the plate and lab_sim's reservoir row
STOCK_X0, STOCK_DX, STOCK_PER_ROW = 0.345, 0.022, 12   # ends at x=0.59, clear of the incubator

# Protocol reagent -> lab_sim reservoir, where the scene already has one.
SCENE_RESERVOIRS = {"enzyme": "reservoir_enzyme", "substrate": "reservoir_substrate"}


@dataclass
class Event:
    t_min: float
    kind: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"t_min": round(self.t_min, 3), "kind": self.kind, **self.detail}


@dataclass
class PipetteState:
    reagent: str | None = None
    volume_ul: float = 0.0
    tip_fresh: bool = True


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


class LabSimWorld:
    def __init__(self, seed: int = 0, plate_offset_mm: tuple[float, float] = (0.0, 0.0),
                 reagents: list[str] | None = None):
        logging.getLogger("mink").setLevel(logging.ERROR)
        from lab_sim.scenes.build_lab import ensure_assets_link  # Lok's helper: mesh path symlink

        ensure_assets_link()
        self.rng = np.random.default_rng(seed)
        self.spec = mujoco.MjSpec.from_file(str(SCENE))
        for name in ARM_BODIES:
            self.spec.body(name).gravcomp = 1.0

        self.wells = sorted((s.name[len("well_"):] for s in self.spec.sites if s.name.startswith("well_")),
                            key=lambda w: (w[0], int(w[1:])))
        self.rows = sorted({w[0] for w in self.wells})
        self.cols = sorted({int(w[1:]) for w in self.wells})
        self.edge_wells = {w for w in self.wells if w[0] in (self.rows[0], self.rows[-1])
                           or int(w[1:]) in (self.cols[0], self.cols[-1])}
        # Where the robot believes each well is (the scene as designed).
        self._believed = {w: np.array(self.spec.site(f"well_{w}").pos, float) for w in self.wells}
        self._offset_plate(np.array([*plate_offset_mm, 0.0]) / 1000.0)
        self.reservoir_index = self._add_stock_rack(reagents or C.reagent_names())

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.ee = self.model.site(EE_SITE).id
        self.down = self.data.site_xmat[self.ee].reshape(3, 3).copy()
        self._init_ik()
        self.arm_geoms = set(self._robot_geoms)
        self.obstacle_geoms = set(self._obstacle_geoms)
        self.liquid = {w: self.model.geom(f"liquid_well_{w}").id for w in self.wells}
        self._liquid_base = {w: self.model.geom_pos[g].copy() for w, g in self.liquid.items()}
        self._liquid_rgba = self.model.geom_rgba[next(iter(self.liquid.values()))].copy()
        self.well_volume = {w: 0.0 for w in self.wells}

        self.pipette = PipetteState()
        self.events: list[Event] = []
        self.ledger: list[dict] = []
        self.clock_min = 0.0
        self.pipette_bias = float(self.rng.normal(0.0, 0.005))
        self.plate_number = 1

    # ------------------------------------------------------------ scene edits
    def _offset_plate(self, offset: np.ndarray) -> None:
        """Mis-seat the plate (fault injection): move its geoms and sites, not the robot's belief."""
        if not offset.any():
            return
        for g in self.spec.geoms:
            if g.name.startswith(("plate_", "vessel_well_", "liquid_well_")):
                g.pos = np.array(g.pos) + offset
        for s in self.spec.sites:
            if s.name.startswith("well_"):
                s.pos = np.array(s.pos) + offset

    def _add_stock_rack(self, reagents: list[str]) -> dict[str, str]:
        """Map every reagent to a site; add tubes for those the scene lacks."""
        index = {}
        missing = []
        for r in reagents:
            if r in SCENE_RESERVOIRS and any(s.name == SCENE_RESERVOIRS[r] for s in self.spec.sites):
                index[r] = SCENE_RESERVOIRS[r]
            else:
                missing.append(r)
        if len(missing) > STOCK_PER_ROW * len(STOCK_ROWS_Y):
            raise ValueError(f"{len(missing)} reagents need stock tubes; rack holds "
                             f"{STOCK_PER_ROW * len(STOCK_ROWS_Y)}")
        world = self.spec.worldbody
        rack_h = 0.02
        xs = STOCK_X0 + STOCK_DX * (STOCK_PER_ROW - 1) / 2
        ys = sum(STOCK_ROWS_Y) / 2
        world.add_geom(name="stock_rack", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[STOCK_DX * STOCK_PER_ROW / 2, 0.04, rack_h / 2], pos=[xs, ys, rack_h / 2],
                       rgba=[0.45, 0.45, 0.5, 1])
        for k, r in enumerate(missing):
            row, col = divmod(k, STOCK_PER_ROW)
            x, y = STOCK_X0 + col * STOCK_DX, STOCK_ROWS_Y[row]
            name = f"stock_{_slug(r)}"
            world.add_geom(name=f"vessel_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[STOCK_TUBE_RADIUS, 0.02, 0], pos=[x, y, rack_h + 0.02],
                           rgba=[0.9, 0.95, 1.0, 0.35], contype=0, conaffinity=0, group=1)
            world.add_site(name=name, pos=[x, y, rack_h + 0.04 + 0.01], size=[0.003, 0, 0],
                           rgba=[1, 0, 0, 0.5], group=4)
            index[r] = name
        return index

    def _init_ik(self) -> None:
        m = self.model
        self.cfg = mink.Configuration(m)
        self.frame_task = mink.FrameTask(EE_SITE, "site", position_cost=1.0, orientation_cost=1.0,
                                         lm_damping=1.0)
        self.posture_task = mink.PostureTask(m, cost=1e-2)
        self.posture_task.set_target(self.data.qpos.copy())
        self._robot_geoms = mink.get_subtree_geom_ids(m, m.body("link0").id)
        self._obstacle_geoms = [m.geom(n).id for n in OBSTACLES] + [
            m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("collide_reservoir")]
        self.limits = [
            mink.ConfigurationLimit(m),
            mink.CollisionAvoidanceLimit(m, geom_pairs=[(self._robot_geoms, self._obstacle_geoms)],
                                         minimum_distance_from_collisions=0.005,
                                         collision_detection_distance=0.05),
        ]

    # ------------------------------------------------------------- geometry
    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(name).id].copy()

    def nominal_well_pos(self, well: str) -> np.ndarray:
        return self._believed[well].copy()

    def actual_well_pos(self, well: str) -> np.ndarray:
        return self.site_pos(f"well_{well}")

    def reservoir_pos(self, reagent: str) -> np.ndarray:
        return self.site_pos(self.reservoir_index[reagent])

    def layout(self) -> dict:
        return {
            "plate": {"format": f"{len(self.rows)}x{len(self.cols)} ({len(self.wells)} wells), rows "
                                f"{self.rows[0]}-{self.rows[-1]} along x, columns 1-{self.cols[-1]} along y",
                      "wells_m": {w: self.nominal_well_pos(w).round(4).tolist() for w in self.wells},
                      "edge_wells": sorted(self.edge_wells), "well_radius_m": WELL_RADIUS,
                      "current_plate_number": self.plate_number},
            "reagent_sources": {r: {"site": s, "pos_m": self.reservoir_pos(r).round(4).tolist()}
                                for r, s in self.reservoir_index.items()},
            "stations": {n: self.site_pos(n).round(4).tolist()
                         for n in ["station_reader", "station_incubator", "station_bench", "waste", "pipette_grip"]},
            "safe_travel_z_m": SAFE_Z,
            "cameras": self.cameras,
        }

    @property
    def cameras(self) -> list[str]:
        return [self.model.camera(i).name for i in range(self.model.ncam)]

    # --------------------------------------------------------------- events
    def log(self, kind: str, **detail) -> Event:
        ev = Event(self.clock_min, kind, detail)
        self.events.append(ev)
        return ev

    # -------------------------------------------------------------- control
    def tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.site_xpos[self.ee].copy(),
                self.data.site_xmat[self.ee].reshape(3, 3)[:, 2].copy())

    def solve_ik(self, target: np.ndarray, iters: int = 400) -> tuple[np.ndarray, float]:
        self.cfg.update(self.data.qpos)
        self.frame_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.down), np.asarray(target, float)))
        err = np.inf
        for _ in range(iters):
            vel = mink.solve_ik(self.cfg, [self.frame_task, self.posture_task], 0.02, "daqp",
                                limits=self.limits)
            self.cfg.integrate_inplace(vel, 0.02)
            e = self.frame_task.compute_error(self.cfg)
            err = float(np.linalg.norm(e[:3]))
            if err < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
        return self.cfg.q[:ARM_JOINTS].copy(), err

    def move_tip(self, target, duration_s: float = 0.25, settle_s: float = 0.05) -> dict:
        target = np.asarray(target, float)
        q_goal, ik_err = self.solve_ik(target)
        if ik_err > 2e-3:
            self.log("ik_failure", target=target.round(4).tolist(), residual_mm=round(ik_err * 1e3, 2))
            return {"ok": False, "reason": "unreachable", "ik_residual_mm": round(ik_err * 1e3, 2)}
        dt = self.model.opt.timestep
        q_start = self.data.ctrl[:ARM_JOINTS].copy()
        n_move = max(1, int(duration_s / dt))
        hits: set[str] = set()
        for k in range(n_move):
            s = (k + 1) / n_move
            self.data.ctrl[:ARM_JOINTS] = q_start + (3 * s**2 - 2 * s**3) * (q_goal - q_start)
            mujoco.mj_step(self.model, self.data)
            hits |= self._contacts()
        for k in range(int(1.0 / dt)):
            mujoco.mj_step(self.model, self.data)
            hits |= self._contacts()
            if k * dt >= settle_s and np.abs(self.data.qvel[:ARM_JOINTS]).max() < 2e-3:
                break
        self.clock_min += LAB_SECONDS_PER_MOVE / 60.0
        pos, axis = self.tip_pose()
        if hits:
            self.log("collision", geoms=sorted(hits), target=target.round(4).tolist())
        return {"ok": True, "tip_pos": pos.round(5).tolist(),
                "tracking_error_mm": round(float(np.linalg.norm(pos - target)) * 1e3, 3),
                "tool_axis": axis.round(4).tolist(), "collisions": sorted(hits)}

    def _contacts(self) -> set[str]:
        hits = set()
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if c.dist < -1e-4 and ((g1 in self.arm_geoms and g2 in self.obstacle_geoms)
                                   or (g2 in self.arm_geoms and g1 in self.obstacle_geoms)):
                other = g2 if g1 in self.arm_geoms else g1
                hits.add(self.model.geom(other).name)
        return hits

    def travel(self, target, hover: float | None = None) -> dict:
        target = np.asarray(target, float)
        z_hover = SAFE_Z if hover is None else hover
        pos, _ = self.tip_pose()
        if pos[2] < z_hover - 1e-3:
            r = self.move_tip([pos[0], pos[1], z_hover], duration_s=0.15)
            if not r["ok"]:
                return r
        r = self.move_tip([target[0], target[1], z_hover])
        if not r["ok"]:
            return r
        # Descend in short steps: one long step near obstacles stalls mink's collision-avoidance limit.
        if z_hover - target[2] > APPROACH_STEP:
            r = self.move_tip([target[0], target[1], target[2] + APPROACH_STEP], duration_s=0.15)
            if not r["ok"]:
                return r
        return self.move_tip(target, duration_s=0.15, settle_s=0.2)

    def set_gripper(self, open_: bool) -> dict:
        self.data.ctrl[ARM_JOINTS] = 255.0 if open_ else 0.0
        for _ in range(int(0.3 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        return {"gripper_open": open_, "finger_gap_m": round(float(self.data.qpos[7] + self.data.qpos[8]), 4)}

    def home(self) -> dict:
        pos, _ = self.tip_pose()
        self.move_tip([pos[0], pos[1], SAFE_Z], duration_s=0.15)
        return self.move_tip(PARK)  # off to the side, so the plate_top camera sees the plate

    # -------------------------------------------------------------- pipette
    def _volume_error(self, volume_ul: float) -> float:
        actual = volume_ul * (1 + self.pipette_bias + self.rng.normal(0, 0.01)) + self.rng.normal(0, 0.15)
        return max(0.0, actual)

    def change_tip(self) -> None:
        self.pipette = PipetteState()
        self.clock_min += 5 / 60.0
        self.log("tip_change")

    def aspirate(self, reagent: str, volume_ul: float) -> dict:
        if reagent not in self.reservoir_index:
            return {"ok": False, "reason": f"no reagent source for {reagent!r}"}
        p = self.pipette
        if p.reagent not in (None, reagent) and p.volume_ul > 0:
            return {"ok": False, "reason": f"tip holds {p.volume_ul:.1f} uL {p.reagent}; dispense or change tip first"}
        if p.reagent not in (None, reagent) and not p.tip_fresh:
            self.log("cross_contamination_risk", previous=p.reagent, now=reagent)
        if p.volume_ul + volume_ul > C.TIP_CAPACITY_UL + 1e-6:
            return {"ok": False, "reason": "exceeds tip capacity"}
        site = self.reservoir_pos(reagent)
        r = self.travel(site)
        if not r["ok"]:
            return r
        radius = RESERVOIR_RADIUS if self.reservoir_index[reagent].startswith("reservoir_") else STOCK_TUBE_RADIUS
        xy_err = float(np.linalg.norm(self.tip_pose()[0][:2] - site[:2]))
        if xy_err > radius:
            self.log("aspirate_miss", reagent=reagent, xy_error_mm=round(xy_err * 1e3, 2))
            return {"ok": False, "reason": "tip not over the reagent source", "xy_error_mm": xy_err * 1e3}
        p.reagent, p.tip_fresh = reagent, False
        p.volume_ul += volume_ul
        return {"ok": True, "reagent": reagent, "tip_volume_ul": round(p.volume_ul, 2)}

    def dispense(self, well: str, volume_ul: float, hover: float | None = None) -> dict:
        """Dispense where the robot believes `well` is. Returns where the liquid actually went."""
        p = self.pipette
        if p.volume_ul + 1e-9 < volume_ul:
            return {"ok": False, "reason": f"tip holds only {p.volume_ul:.1f} uL"}
        r = self.travel(self.nominal_well_pos(well), hover=hover)
        if not r["ok"]:
            return r
        tip_xy = self.tip_pose()[0][:2]
        dists = {w: float(np.linalg.norm(tip_xy - self.actual_well_pos(w)[:2])) for w in self.wells}
        landed = min(dists, key=dists.get)
        p.volume_ul = max(0.0, p.volume_ul - volume_ul)
        if p.volume_ul < 1e-6:  # float residue from many partial dispenses
            p.volume_ul = 0.0
        delivered = self._volume_error(volume_ul)
        if dists[landed] > WELL_RADIUS:
            landed = None
            self.log("spill", well=well, reagent=p.reagent, volume_ul=round(delivered, 2),
                     xy_error_mm=round(dists[well] * 1e3, 2))
        elif landed != well:
            self.log("wrong_well", intended=well, actual=landed, reagent=p.reagent,
                     volume_ul=round(delivered, 2))
        if landed:
            self._fill(landed, delivered)
        self.ledger.append({"t_min": round(self.clock_min, 3), "tool": "dispense", "plate": self.plate_number,
                            "intended": {"well": well, "reagent": p.reagent, "volume_ul": volume_ul},
                            "actual": {"well": landed, "volume_ul": round(delivered, 3) if landed else 0.0},
                            "tip_error_mm": round(dists[well] * 1e3, 3)})
        return {"ok": True, "intended_well": well, "actual_well": landed, "spilled": landed is None,
                "delivered_ul": delivered if landed else 0.0, "reagent": p.reagent,
                "xy_error_mm": round(dists[well] * 1e3, 3)}

    # --------------------------------------------------------------- visuals
    def _fill(self, well: str, volume_ul: float) -> None:
        self.well_volume[well] += volume_ul
        frac = min(1.0, self.well_volume[well] / C.WELL_VOLUME_UL)
        g = self.liquid[well]
        half = max(1e-4, frac * WELL_HEIGHT * 0.9 / 2)
        self.model.geom_size[g][1] = half
        self.model.geom_pos[g][2] = self._liquid_base[well][2] + half

    def show_absorbance(self, well: str, absorbance: float) -> None:
        """Colour a well by its last A405 read (yellow p-nitrophenolate)."""
        a = float(np.clip(absorbance / 1.5, 0, 1))
        self.model.geom_rgba[self.liquid[well]] = [1.0, 1.0, 1.0 - 0.85 * a, 0.9]

    def swap_plate(self) -> None:
        """Replace the plate with a fresh one (next load of a multi-plate batch)."""
        for w, g in self.liquid.items():
            self.model.geom_size[g][1] = 1e-4
            self.model.geom_pos[g] = self._liquid_base[w]
            self.model.geom_rgba[g] = self._liquid_rgba
            self.well_volume[w] = 0.0
        self.plate_number += 1
        self.clock_min += 1.0
        self.log("plate_swap", plate=self.plate_number)

    # ------------------------------------------------------------------ read
    def robot_state(self) -> dict:
        pos, axis = self.tip_pose()
        return {
            "joint_positions_rad": self.data.qpos[:ARM_JOINTS].round(4).tolist(),
            "joint_velocities": self.data.qvel[:ARM_JOINTS].round(4).tolist(),
            "joint_torques_Nm": self.data.actuator_force[:ARM_JOINTS].round(3).tolist(),
            "tip_position_m": pos.round(5).tolist(),
            "tool_axis": axis.round(4).tolist(),
            "finger_gap_m": round(float(self.data.qpos[7] + self.data.qpos[8]), 4),
            "pipette": {"reagent": self.pipette.reagent, "volume_ul": round(self.pipette.volume_ul, 2),
                        "tip_fresh": self.pipette.tip_fresh},
            "current_plate_number": self.plate_number,
            "sim_time_s": round(float(self.data.time), 3),
            "lab_clock_min": round(self.clock_min, 2),
        }

    def render_png(self, camera: str = "front", width: int = 640, height: int = 480) -> bytes:
        import io

        from PIL import Image

        with mujoco.Renderer(self.model, height, width) as r:
            r.update_scene(self.data, camera=camera)
            img = r.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()
