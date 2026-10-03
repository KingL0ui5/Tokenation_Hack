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

ARM = 7
_POINTS = {"nozzle": "pipette_nozzle", "tip_end": "pipette_tip_end"}


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
        self._set_site(active)
        # vertical reference = the active site's orientation at the current (home) pose
        self.down = data.site_xmat[self.site_id].reshape(3, 3).copy()
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
