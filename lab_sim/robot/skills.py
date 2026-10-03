"""Reusable pipette-motion skills for the held pipette.

The robot carries the pipette rigidly (see scenes.build_lab._mount_pipette), so IK targets a
point ON the pipette (pipette_nozzle, or pipette_tip_end when a disposable tip is attached),
not the hand's attachment_site. Each skill:

- keeps the pipette within `max_tilt_deg` of vertical (a 20 cm lever magnifies wrist error);
- travels high enough that the nozzle clears the tall 15 mL tubes, and makes the final
  approach a slow straight vertical descent;
- returns a MoveResult (ok, reason, actual tip position, error from target, tilt) and never
  raises.

These operate on a compiled model + data; the backend's pipette() path calls them instead of
moving inline. Collision avoidance uses the pipette shaft capsule automatically (it's in the
hand subtree); the last few cm of tip has no collider, so it can enter openings.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from scenes.build_lab import PIPETTE_PARTS as _PIPETTE_PARTS

ARM = 7


def _has_site(model, name: str) -> bool:
    return any(model.site(i).name == name for i in range(model.nsite))
_POINTS = {"nozzle": "pipette_nozzle", "tip_end": "pipette_tip_end", "hand": "attachment_site"}
GRIP_HALF_M = 0.00914     # finger half-opening that grips the pipette handle


@dataclass
class MoveResult:
    ok: bool
    reason: str | None
    tip_pos: list          # actual active-tip world position after the move
    error_m: float = 0.0   # distance from the commanded target
    tilt_deg: float = 0.0  # pipette deviation from vertical


class PipetteSkills:
    def __init__(self, model, data, obstacles, safe_z: float = 0.22, max_tilt_deg: float = 2.0,
                 active: str = "pipette_nozzle"):
        self.model, self.data = model, data
        self.safe_z, self.max_tilt = safe_z, max_tilt_deg
        # vertical reference orientation per targetable point, captured at the (home) pose
        self._down_by_site = {s: data.site_xmat[model.site(s).id].reshape(3, 3).copy()
                              for s in _POINTS.values() if _has_site(model, s)}
        self._set_site(active)
        self.down = self._down_by_site[active]

        # pipette-swap state: hand-mounted pipette (on body "pipette") vs the stand pipette
        self.held = True
        self._drop_qpos = None
        self.gripper_act = model.actuator("actuator8").id
        pid = model.body("pipette").id
        self.mounted_shaft = model.geom("pipette_shaft").id
        self.mounted_visual = [g for g in range(model.ngeom)
                               if model.geom_bodyid[g] == pid and g != self.mounted_shaft]
        self.stand_shaft = model.geom("stand_pipette_shaft").id
        self.stand_visual = [model.geom(f"stand_pipette_{p}").id for p in _PIPETTE_PARTS]
        self.cfg = mink.Configuration(model)
        self.posture = mink.PostureTask(model, cost=1e-2)
        self.posture.set_target(data.qpos.copy())
        robot = mink.get_subtree_geom_ids(model, model.body("link0").id)
        obs = [model.geom(n).id for n in obstacles]
        self.limits = [mink.ConfigurationLimit(model),
                       mink.CollisionAvoidanceLimit(model, geom_pairs=[(robot, obs)],
                                                    minimum_distance_from_collisions=0.005,
                                                    collision_detection_distance=0.05)]
        self._make_task()

    # -------------------------------------------------------------- internals
    def _set_site(self, site: str) -> None:
        self.active = site
        self.site_id = self.model.site(site).id
        self.down = self._down_by_site[site]

    def _make_task(self) -> None:
        self.frame_task = mink.FrameTask(self.active, "site", position_cost=1.0,
                                         orientation_cost=1.0, lm_damping=1.0)

    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def _tilt_deg(self) -> float:
        z = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]   # active-site approach axis
        return float(np.degrees(np.arccos(np.clip(abs(z @ np.array([0, 0, -1.0])), 0, 1))))

    def _result(self, ok: bool, reason, target=None) -> MoveResult:
        tp = self.tip()
        err = float(np.linalg.norm(tp - target)) if target is not None else 0.0
        tilt = self._tilt_deg()
        if ok and tilt > self.max_tilt:
            ok, reason = False, f"tilt {tilt:.1f} deg exceeds {self.max_tilt} deg from vertical"
        return MoveResult(ok, reason, tp.round(4).tolist(), round(err, 4), round(tilt, 2))

    def _step(self) -> None:
        self.data.qfrc_applied[:ARM] = self.data.qfrc_bias[:ARM]   # gravity feed-forward
        mujoco.mj_step(self.model, self.data)

    def _solve(self, target):
        self.cfg.update(self.data.qpos)
        self.frame_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.down), np.asarray(target, float)))
        err = np.inf
        for _ in range(400):
            try:
                vel = mink.solve_ik(self.cfg, [self.frame_task, self.posture], 0.02, "daqp",
                                    limits=self.limits)
            except Exception:
                return None, np.inf        # infeasible QP (e.g. out of reach) -> unreachable
            self.cfg.integrate_inplace(vel, 0.02)
            e = self.frame_task.compute_error(self.cfg)
            err = float(np.linalg.norm(e[:3]))
            if err < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
        return self.cfg.q[:ARM].copy(), err

    def _goto(self, target, duration: float):
        q, err = self._solve(target)
        if q is None or err > 3e-3:
            return False, "unreachable"
        dt = self.model.opt.timestep
        q0 = self.data.ctrl[:ARM].copy()
        for k in range(max(1, int(duration / dt))):
            s = (k + 1) / max(1, int(duration / dt))
            self.data.ctrl[:ARM] = q0 + (3 * s**2 - 2 * s**3) * (q - q0)
            self._step()
        for k in range(int(0.6 / dt)):               # settle
            self._step()
            if k * dt > 0.05 and np.abs(self.data.qvel[:ARM]).max() < 2e-3:
                break
        return True, None

    # ---------------------------------------------------------------- skills
    def set_active_point(self, name: str) -> MoveResult:
        """Switch which point on the pipette IK targets ("nozzle" or "tip_end")."""
        site = _POINTS.get(name, name)
        if site not in (self.model.site(i).name for i in range(self.model.nsite)):
            return MoveResult(False, f"unknown point {name!r}", self.tip().round(4).tolist())
        self._set_site(site)
        self._make_task()
        return self._result(True, None)

    def tip_error(self, site: str) -> MoveResult:
        """No motion: the active tip's current position and its error from `site`."""
        target = self.data.site_xpos[self.model.site(site).id].copy()
        return self._result(True, None, target)

    def travel_to(self, site: str, clearance: float = 0.04) -> MoveResult:
        """Bring the active tip to `clearance` above `site`, via a safe travel height that
        clears the tall tubes. Vertical throughout."""
        target = self.data.site_xpos[self.model.site(site).id].copy() + np.array([0, 0, clearance])
        tp = self.tip()
        if tp[2] < self.safe_z - 1e-3:
            ok, r = self._goto([tp[0], tp[1], self.safe_z], 0.25)
            if not ok:
                return self._result(False, r, [tp[0], tp[1], self.safe_z])
        ok, r = self._goto([target[0], target[1], max(self.safe_z, target[2])], 0.4)
        if not ok:
            return self._result(False, r, target)
        ok, r = self._goto(target, 0.3)              # down to the hover point
        return self._result(ok, r, target)

    def descend(self, depth: float, duration: float = 0.8) -> MoveResult:
        """Slow straight-down descent of the active tip by `depth` metres."""
        target = self.tip() - np.array([0, 0, depth])
        ok, r = self._goto(target, duration)
        return self._result(ok, r, target)

    def ascend(self, duration: float = 0.3) -> MoveResult:
        """Straight-up ascent of the active tip back to the safe travel height."""
        tp = self.tip()
        target = np.array([tp[0], tp[1], self.safe_z])
        ok, r = self._goto(target, duration)
        return self._result(ok, r, target)

    # --------------------------------------------------------------- gripper
    def open_gripper(self) -> MoveResult:
        self.data.ctrl[self.gripper_act] = 255.0                  # fully open
        for _ in range(int(0.3 / self.model.opt.timestep)):
            self._step()
        return self._result(True, "gripper open")

    def close_gripper(self, width: float = 2 * GRIP_HALF_M) -> MoveResult:
        self.data.ctrl[self.gripper_act] = float(np.clip(width / 0.04 * 255, 0, 255))
        for _ in range(int(0.3 / self.model.opt.timestep)):
            self._step()
        return self._result(True, f"gripper to {width * 1000:.0f} mm")

    # --------------------------------------------------- pipette put-down / pick-up
    def _show(self, visual_ids, shaft_id, visible: bool) -> None:
        a = 1.0 if visible else 0.0
        for g in visual_ids:
            self.model.geom_rgba[g][3] = a
        self.model.geom_contype[shaft_id] = 1 if visible else 0
        self.model.geom_conaffinity[shaft_id] = 1 if visible else 0

    def _copy_mounted_to_stand(self) -> None:
        """Place the stand pipette exactly where the held pipette is now (seamless swap)."""
        def copy(src, dst):
            self.model.geom_pos[dst] = self.data.geom_xpos[src].copy()
            q = np.zeros(4); mujoco.mju_mat2Quat(q, self.data.geom_xmat[src]); self.model.geom_quat[dst] = q
        for s, d in zip(self.mounted_visual, self.stand_visual):
            copy(s, d)
        copy(self.mounted_shaft, self.stand_shaft)
        mujoco.mj_forward(self.model, self.data)

    def _goto_qpos(self, q, duration: float = 0.5) -> None:
        dt = self.model.opt.timestep
        q0 = self.data.ctrl[:ARM].copy()
        n = max(1, int(duration / dt))
        for k in range(n):
            s = (k + 1) / n
            self.data.ctrl[:ARM] = q0 + (3 * s**2 - 2 * s**3) * (q[:ARM] - q0)
            self._step()
        for k in range(int(0.5 / dt)):
            self._step()
            if k * dt > 0.05 and np.abs(self.data.qvel[:ARM]).max() < 2e-3:
                break

    def put_down_pipette(self) -> MoveResult:
        """Lower the held pipette into the stand, swap to the stand pipette, free the gripper."""
        self.set_active_point("nozzle")
        r = self.travel_to("pipette_stand", clearance=0.03)
        if r.ok:
            r = self.descend(0.03)
        if not r.ok:
            return r
        self._drop_qpos = self.data.qpos.copy()
        self._copy_mounted_to_stand()
        self._show(self.mounted_visual, self.mounted_shaft, False)
        self._show(self.stand_visual, self.stand_shaft, True)
        self.open_gripper()
        self.held = False
        self.set_active_point("hand")           # gripper free -> IK targets attachment_site
        self.ascend()
        return self._result(True, "pipette placed in stand")

    def pick_up_pipette(self) -> MoveResult:
        """Return to the stand, swap back to the held pipette, close the gripper on it."""
        if self._drop_qpos is None:
            return MoveResult(False, "no pipette in the stand", self.tip().round(4).tolist())
        self.set_active_point("hand")           # approach empty-handed
        self._goto_qpos(self._drop_qpos)        # re-dock to the exact drop pose (seamless)
        self._show(self.stand_visual, self.stand_shaft, False)
        self._show(self.mounted_visual, self.mounted_shaft, True)
        self.close_gripper()
        self.held = True
        self.set_active_point("nozzle")         # IK targets the nozzle while held
        self.ascend()
        return self._result(True, "pipette picked up")
