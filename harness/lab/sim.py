"""MuJoCo world: Franka Panda with a single-channel pipette over a 96-well plate.

Motion uses classical control only: damped-least-squares IK to a pose with the
tip pointing down, a joint-space trajectory to the IK solution tracked by the
Panda's position actuators, and a settle phase. Everything the robot does is
logged as events (moves, aspirations, dispenses, spills, collisions).

Liquid is tracked in software, not simulated as fluid. The physics decides
*where* the tip actually ends up; the pipette model decides how much liquid
goes where, given that position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import config as C

PANDA_SCENE = Path(__file__).resolve().parents[2] / "lab_sim" / "models" / "franka_emika_panda" / "scene.xml"
PIPETTE_LENGTH = 0.20
ARM_JOINTS = 7
DOWN = np.array([0.0, 0.0, -1.0])
WELL_RADIUS = 0.0034
RESERVOIR_RADIUS = 0.009
LAB_SECONDS_PER_MOVE = 1.5


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


class LabWorld:
    """Physical layer. Positions are metres in the world frame."""

    def __init__(self, seed: int = 0, plate_offset_mm: tuple[float, float] = (0.0, 0.0)):
        self.rng = np.random.default_rng(seed)
        self.spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
        # Where the plate really is, versus where the robot believes it is.
        # A non-zero offset models a mis-seated plate (a calibration fault).
        self.plate_offset = np.array([*plate_offset_mm, 0.0]) / 1000.0
        self._build_deck()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.tip = self.model.site("pipette_tip").id
        self.arm_bodies = {self.model.body(n).id for n in
                           ["link5", "link6", "link7", "hand", "left_finger", "right_finger"]}
        self.deck_bodies = {self.model.body(n).id for n in ["plate", "rack"]}
        self.pipette = PipetteState()
        self.events: list[Event] = []
        self.clock_min = 0.0
        # Small per-instrument calibration bias, fixed for the session.
        self.pipette_bias = float(self.rng.normal(0.0, 0.005))

    # ---------------------------------------------------------------- deck
    def _build_deck(self) -> None:
        # Gravity compensation, as the real Panda controller does; without it the
        # position servos droop 5-7 mm at the tip.
        for name in ["link1", "link2", "link3", "link4", "link5", "link6", "link7",
                     "hand", "left_finger", "right_finger"]:
            self.spec.body(name).gravcomp = 1.0
        hand = self.spec.body("hand")
        tip = hand.add_site(name="pipette_tip", pos=[0, 0, PIPETTE_LENGTH], size=[0.002, 0, 0])
        tip.rgba = [1, 0.2, 0.2, 1]
        hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, PIPETTE_LENGTH / 2, 0],
                      pos=[0, 0, PIPETTE_LENGTH / 2 + 0.02], rgba=[0.9, 0.9, 0.9, 1],
                      contype=0, conaffinity=0)

        world = self.spec.worldbody
        px, py, pz = C.PLATE_CENTER
        actual = np.array(C.PLATE_CENTER) + self.plate_offset
        plate = world.add_body(name="plate", pos=list(actual[:2]) + [pz - 0.0075])
        plate.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.043, 0.064, 0.0075],
                       rgba=[0.85, 0.85, 0.95, 1])
        for i, r in enumerate(C.ROWS):
            for j in C.COLS:
                local = self._well_local(i, j)
                plate.add_site(name=f"well_{r}{j}", pos=[local[0], local[1], 0.0075],
                               size=[WELL_RADIUS, 0, 0], rgba=[0.3, 0.3, 0.8, 0.6])

        rx, ry, rz = C.RACK_CENTER
        rack = world.add_body(name="rack", pos=[rx, ry, rz - 0.02])
        names = C.reagent_names()
        ncols = 8
        nrows = int(np.ceil(len(names) / ncols))
        rack.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[nrows * C.RACK_PITCH / 2, ncols * C.RACK_PITCH / 2, 0.02],
                      rgba=[0.6, 0.6, 0.6, 1])
        self.reservoir_index = {}
        for k, name in enumerate(names):
            i, j = divmod(k, ncols)
            lx = (i - (nrows - 1) / 2) * C.RACK_PITCH
            ly = (j - (ncols - 1) / 2) * C.RACK_PITCH
            rack.add_site(name=f"res_{k}", pos=[lx, ly, 0.02], size=[RESERVOIR_RADIUS, 0, 0],
                          rgba=[0.2, 0.7, 0.3, 0.7])
            self.reservoir_index[name] = k

        world.add_camera(name="overview", pos=[1.3, 0.0, 0.9], xyaxes=[0, 1, 0, -0.55, 0, 0.85])
        world.add_camera(name="deck_top", pos=[0.5, 0.07, 0.75], xyaxes=[0, 1, 0, -1, 0, 0])

    @staticmethod
    def _well_local(i: int, j: int) -> np.ndarray:
        # Rows along x, columns along y.
        return np.array([(i - 3.5) * C.WELL_PITCH, (j - 6.5) * C.WELL_PITCH])

    def nominal_well_pos(self, well: str) -> np.ndarray:
        i = C.ROWS.index(well[0])
        j = int(well[1:])
        xy = np.array(C.PLATE_CENTER[:2]) + self._well_local(i, j)
        return np.array([*xy, C.PLATE_CENTER[2]])

    def actual_well_pos(self, well: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(f"well_{well}").id].copy()

    def reservoir_pos(self, reagent: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(f"res_{self.reservoir_index[reagent]}").id].copy()

    # -------------------------------------------------------------- events
    def log(self, kind: str, **detail) -> Event:
        ev = Event(self.clock_min, kind, detail)
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------- control
    def tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.site_xpos[self.tip].copy(),
                self.data.site_xmat[self.tip].reshape(3, 3)[:, 2].copy())

    def solve_ik(self, target: np.ndarray, iters: int = 200) -> tuple[np.ndarray, float]:
        """Damped least squares on position + tool-down orientation."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        lo, hi = self.model.jnt_range[:ARM_JOINTS].T
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        err = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p = d.site_xpos[self.tip]
            z = d.site_xmat[self.tip].reshape(3, 3)[:, 2]
            e = np.r_[target - p, np.cross(z, DOWN)]
            err = float(np.linalg.norm(e[:3]))
            if err < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, d, jp, jr, self.tip)
            J = np.r_[jp, jr][:, :ARM_JOINTS]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d.qpos[:ARM_JOINTS] = np.clip(d.qpos[:ARM_JOINTS] + dq, lo, hi)
        return d.qpos[:ARM_JOINTS].copy(), err

    def move_tip(self, target, duration_s: float = 0.25, settle_s: float = 0.05) -> dict:
        """Move the pipette tip to a world position. Returns the achieved pose."""
        target = np.asarray(target, dtype=float)
        q_goal, ik_err = self.solve_ik(target)
        if ik_err > 1e-3:
            self.log("ik_failure", target=target.round(4).tolist(), residual_mm=round(ik_err * 1e3, 2))
            return {"ok": False, "reason": "unreachable", "ik_residual_mm": ik_err * 1e3}
        q_start = self.data.ctrl[:ARM_JOINTS].copy()
        dt = self.model.opt.timestep
        n_move = max(1, int(duration_s / dt))
        collided = set()
        for k in range(n_move):
            s = (k + 1) / n_move
            s = 3 * s**2 - 2 * s**3  # smoothstep
            self.data.ctrl[:ARM_JOINTS] = q_start + s * (q_goal - q_start)
            mujoco.mj_step(self.model, self.data)
            collided |= self._arm_deck_contacts()
        # Settle for at least settle_s, then until the joints have stopped (capped at 1 s).
        for k in range(int(1.0 / dt)):
            mujoco.mj_step(self.model, self.data)
            collided |= self._arm_deck_contacts()
            if k * dt >= settle_s and np.abs(self.data.qvel[:ARM_JOINTS]).max() < 2e-3:
                break
        self.clock_min += LAB_SECONDS_PER_MOVE / 60.0
        pos, axis = self.tip_pose()
        tracking = float(np.linalg.norm(pos - target))
        if collided:
            self.log("collision", bodies=sorted(collided), target=target.round(4).tolist())
        return {"ok": True, "tip_pos": pos.round(5).tolist(), "tracking_error_mm": round(tracking * 1e3, 3),
                "tool_axis": axis.round(4).tolist(), "collisions": sorted(collided)}

    def _arm_deck_contacts(self) -> set[str]:
        hits = set()
        for c in self.data.contact[: self.data.ncon]:
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1 in self.arm_bodies and b2 in self.deck_bodies) or (b2 in self.arm_bodies and b1 in self.deck_bodies):
                hits.add(f"{self.model.body(b1).name}<->{self.model.body(b2).name}")
        return hits

    def travel(self, target, hover: float | None = None) -> dict:
        """Lift to a safe height, translate, then descend to target."""
        target = np.asarray(target, dtype=float)
        z_hover = C.SAFE_Z if hover is None else hover
        pos, _ = self.tip_pose()
        if pos[2] < z_hover - 1e-3:
            r = self.move_tip([pos[0], pos[1], z_hover], duration_s=0.15)
            if not r["ok"]:
                return r
        r = self.move_tip([target[0], target[1], z_hover])
        if not r["ok"]:
            return r
        # Only the final approach needs to settle fully (sub-0.1 mm).
        return self.move_tip(target, duration_s=0.15, settle_s=0.2)

    def set_gripper(self, open_: bool) -> dict:
        self.data.ctrl[ARM_JOINTS] = 255.0 if open_ else 0.0
        for _ in range(int(0.3 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        width = float(self.data.qpos[7] + self.data.qpos[8])
        return {"gripper_open": open_, "finger_gap_m": round(width, 4)}

    def home(self) -> dict:
        pos, _ = self.tip_pose()
        self.move_tip([pos[0], pos[1], C.SAFE_Z], duration_s=0.15)
        return self.move_tip([0.50, 0.05, C.SAFE_Z])

    # ------------------------------------------------------------- pipette
    def _volume_error(self, volume_ul: float) -> float:
        # 1% CV relative + 0.15 uL absolute, plus the session calibration bias.
        actual = volume_ul * (1 + self.pipette_bias + self.rng.normal(0, 0.01)) + self.rng.normal(0, 0.15)
        return max(0.0, actual)

    def change_tip(self) -> None:
        self.pipette = PipetteState()
        self.clock_min += 5 / 60.0
        self.log("tip_change")

    def aspirate(self, reagent: str, volume_ul: float) -> dict:
        if reagent not in self.reservoir_index:
            return {"ok": False, "reason": f"no reservoir for {reagent!r}"}
        p = self.pipette
        if p.reagent not in (None, reagent) and p.volume_ul > 0:
            return {"ok": False, "reason": f"tip holds {p.volume_ul:.1f} uL {p.reagent}; dispense or change tip first"}
        if p.reagent not in (None, reagent) and not p.tip_fresh:
            self.log("cross_contamination_risk", previous=p.reagent, now=reagent)
        if p.volume_ul + volume_ul > C.TIP_CAPACITY_UL:
            return {"ok": False, "reason": "exceeds tip capacity"}
        target = self.reservoir_pos(reagent) + [0, 0, -0.01]
        r = self.travel(target)
        if not r["ok"]:
            return r
        xy_err = float(np.linalg.norm(self.tip_pose()[0][:2] - target[:2]))
        if xy_err > RESERVOIR_RADIUS:
            self.log("aspirate_miss", reagent=reagent, xy_error_mm=round(xy_err * 1e3, 2))
            return {"ok": False, "reason": "tip not inside reservoir", "xy_error_mm": xy_err * 1e3}
        p.reagent = reagent
        p.volume_ul += volume_ul
        p.tip_fresh = False
        return {"ok": True, "reagent": reagent, "tip_volume_ul": round(p.volume_ul, 2)}

    def dispense(self, well: str, volume_ul: float, hover: float | None = None) -> dict:
        """Dispense into a well. Returns the volume that actually landed in it."""
        p = self.pipette
        if p.volume_ul + 1e-9 < volume_ul:
            return {"ok": False, "reason": f"tip holds only {p.volume_ul:.1f} uL"}
        believed = self.nominal_well_pos(well) + [0, 0, C.WORK_Z_OFFSET]
        r = self.travel(believed, hover=hover)
        if not r["ok"]:
            return r
        tip_xy = self.tip_pose()[0][:2]
        xy_err = float(np.linalg.norm(tip_xy - self.actual_well_pos(well)[:2]))
        p.volume_ul -= volume_ul
        delivered = self._volume_error(volume_ul)
        if xy_err > WELL_RADIUS:
            self.log("spill", well=well, reagent=p.reagent, volume_ul=round(delivered, 2),
                     xy_error_mm=round(xy_err * 1e3, 2))
            return {"ok": True, "spilled": True, "delivered_ul": 0.0, "reagent": p.reagent,
                    "xy_error_mm": round(xy_err * 1e3, 3)}
        return {"ok": True, "spilled": False, "delivered_ul": delivered, "reagent": p.reagent,
                "xy_error_mm": round(xy_err * 1e3, 3)}

    # ---------------------------------------------------------------- read
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
            "sim_time_s": round(float(self.data.time), 3),
            "lab_clock_min": round(self.clock_min, 2),
        }

    def render_png(self, camera: str = "overview", width: int = 640, height: int = 480) -> bytes:
        import io

        from PIL import Image

        with mujoco.Renderer(self.model, height, width) as r:
            r.update_scene(self.data, camera=camera)
            img = r.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()
