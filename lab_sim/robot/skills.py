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
        # arm joint qpos / dof / actuator indices (free joints shift them off 0..6)
        self.arm_qadr = np.array([model.jnt_qposadr[model.joint(f"joint{i + 1}").id] for i in range(ARM)])
        self.arm_dof = np.array([model.jnt_dofadr[model.joint(f"joint{i + 1}").id] for i in range(ARM)])
        self.arm_act = np.array([model.actuator(f"actuator{i + 1}").id for i in range(ARM)])
        pid = model.body("pipette").id
        self.mounted_shaft = model.geom("pipette_shaft").id
        self.mounted_visual = [g for g in range(model.ngeom)
                               if model.geom_bodyid[g] == pid and g != self.mounted_shaft]
        self.stand_shaft = model.geom("stand_pipette_shaft").id
        self.stand_visual = [model.geom(f"stand_pipette_{p}").id for p in _PIPETTE_PARTS]

        # grasp / weld state
        self.held_object = None
        self.welds = {model.equality(e).name[len("weld_"):]: e
                      for e in range(model.neq) if (model.equality(e).name or "").startswith("weld_")}
        self.worldwelds = {model.equality(e).name[len("worldweld_"):]: e
                           for e in range(model.neq) if (model.equality(e).name or "").startswith("worldweld_")}
        self.free_qadr, self.free_dofadr = {}, {}
        for b in self.welds:
            jadr = model.body_jntadr[model.body(b).id]
            self.free_qadr[b], self.free_dofadr[b] = model.jnt_qposadr[jadr], model.jnt_dofadr[jadr]
        self.left_pads = {g for g in range(model.ngeom)
                          if model.body(model.geom_bodyid[g]).name == "left_finger"}
        self.right_pads = {g for g in range(model.ngeom)
                           if model.body(model.geom_bodyid[g]).name == "right_finger"}
        self.cfg = mink.Configuration(model)
        self.home_qpos = model.key_qpos[0].copy()      # arm neutral, for reseeding a stuck IK solve
        self.posture = mink.PostureTask(model, cost=1e-2)
        self.posture.set_target(data.qpos.copy())
        self._robot_geoms = mink.get_subtree_geom_ids(model, model.body("link0").id)
        self._obs_geoms = [model.geom(n).id for n in obstacles]
        self.limits = [mink.ConfigurationLimit(model), self._collision_limit()]
        self._make_task()
        for b in self.worldwelds:            # hold every tube/plate rigidly in its slot so it
            self._world_weld(b, True)        # can't be knocked; grasp's caller frees its target,
        mujoco.mj_forward(model, self.data)  # and place() re-welds it in the new slot.
        self._init_tips()
        self._init_incidents()

    # ------------------------------------------------------- held-object incidents
    def _init_incidents(self) -> None:
        """Collider lookup for every graspable body, and incident-tracking state. Incident
        checks are scoped to what the robot actually carries (held() docstring): the mounted
        pipette/tip (always, while mounted) and a grasped tube/plate (while welded to the hand)."""
        self._grasp_colliders = {b: self._collider_for_body(b) for b in self.welds}
        self._grasp_collider_ids = set(self._grasp_colliders.values())
        self._geom_names = [self.model.geom(g).name for g in range(self.model.ngeom)]
        self.incidents: list[dict] = []          # unexpected contacts while carrying something
        self._seen_incidents: set[tuple] = set()  # dedup key: (carried geom name, other geom name)
        self._placing = False                    # True only during place()'s final descent
        self._held_liftoff_exempt: set[int] = set()  # what the held object rested against at grasp
        m = self.model
        # Static scene geoms a placed object must not pass through: everything on the world body
        # that's visible or collidable (rack meshes are visual-only, so contacts can't see them).
        self._static_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0
                              and self._geom_names[g] != "floor"
                              and (m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_rgba[g][3] > 0)]

    def _ray_static(self, g: int, pnt, vec=(0.0, 0.0, -1.0)) -> float:
        """Distance along `vec` from `pnt` to static geom `g` (-1 if missed)."""
        pnt, vec = np.asarray(pnt, float), np.asarray(vec, float)
        if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            return mujoco.mj_rayMesh(self.model, self.data, g, pnt, vec)
        return mujoco.mju_rayGeom(self.data.geom_xpos[g], self.data.geom_xmat[g],
                                  self.model.geom_size[g], pnt, vec, self.model.geom_type[g])

    def _seat_z(self, body: str, target_z: float) -> float:
        """Base height for `body` seated at a slot site. Slot sites follow the tube convention
        (the site marks where the seated tube's own opening site sits), so the seat is the site
        minus the tube's base-to-opening-site offset. Other objects sit on the bench."""
        if body.startswith("tubebody_"):
            return target_z - self.model.site_pos[self.model.site("reagent_" + body[len("tubebody_"):]).id][2]
        return 0.0

    def _static_penetration(self, body: str, n_rays: int = 24) -> str | None:
        """Name of a static geom the body's (upright cylinder) collider passes through, or None.
        Vertical rays just inside its wall from top to base: any hit above the base means rack
        material (a solid plate, a wall) occupies the object's footprint."""
        coll = self._grasp_colliders[body]
        r, half = self.model.geom_size[coll][0] * 0.97, self.model.geom_size[coll][1]
        c = self.data.geom_xpos[coll]
        top, bottom = c[2] + half, c[2] - half
        pts = [(c[0], c[1])] + [(c[0] + r * np.cos(a), c[1] + r * np.sin(a))
                                for a in np.linspace(0, 2 * np.pi, n_rays, endpoint=False)]
        for g in self._static_geoms:
            if np.linalg.norm(self.data.geom_xpos[g][:2] - c[:2]) > self.model.geom_rbound[g] + r:
                continue
            for x, y in pts:
                d = self._ray_static(g, [x, y, top])
                if d >= 0 and top - d > bottom + 1e-3:
                    return self._geom_names[g]
        return None

    def _collider_for_body(self, body: str) -> int:
        if body.startswith("tubebody_"):
            return self.model.geom(f"collide_tube_{body[len('tubebody_'):]}").id
        return self.model.geom(f"{body}_collision").id   # plate_collision / p1_plate_collision

    def _carried_geoms(self) -> set[int]:
        """Everything the robot is physically carrying right now."""
        g = set()
        if self.held:
            g.add(self.mounted_shaft)
            if self.has_tip:
                g.add(self.tip_mounted_coll)
        if self.held_object:
            g.add(self._grasp_colliders[self.held_object])
        return g

    def _record_incidents(self) -> None:
        """Scan real contacts for anything carried touching something it shouldn't. Expected
        contacts are exempted: grip pads vs the object they're holding, and (only during
        place()'s final descent) the held object vs the destination structure. A sibling
        tube/plate is NEVER exempted, even while placing -- that's the brush-a-neighbour bug."""
        carried = self._carried_geoms()
        if not carried:
            return
        held_coll = self._grasp_colliders.get(self.held_object)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= -1e-4 or (c.geom1 in carried) == (c.geom2 in carried):
                continue                                  # not penetrating, or both/neither carried
            carried_g, other_g = (c.geom1, c.geom2) if c.geom1 in carried else (c.geom2, c.geom1)
            other_name = self._geom_names[other_g]
            other_body = self.model.body(self.model.geom_bodyid[other_g]).name
            is_sibling = other_g in self._grasp_collider_ids and other_g != held_coll
            if not is_sibling:
                if carried_g == held_coll and other_body in ("hand", "left_finger", "right_finger"):
                    continue                              # grip pads holding the object
                if self._placing or other_g in self._held_liftoff_exempt:
                    continue                              # expected contact with the slot/bench
                                                           # it's arriving at or departing from
            key = (self._geom_names[carried_g], other_name)
            if key not in self._seen_incidents:
                self._seen_incidents.add(key)
                self.incidents.append({"carried": self._geom_names[carried_g], "other": other_name,
                                       "dist": round(float(c.dist), 5)})

    def _held_lowest_z_offset(self) -> float:
        """How far below the active IK point the held object's actual lowest point currently
        is. Zero for the pipette/tip (the active point already IS their lowest point); for a
        grasped tube, the live gap between the hand (active point while carrying) and the
        tube body's origin (its base)."""
        if self.held_object:
            bid = self.model.body(self.held_object).id
            return max(0.0, self.tip()[2] - self.data.xpos[bid][2])
        return 0.0

    # -------------------------------------------------------------- internals
    def _collision_limit(self) -> "mink.CollisionAvoidanceLimit":
        """Robot<->obstacle avoidance. mink filters pairs by contype/conaffinity at build time,
        so the mounted-tip collider is only included once it's activated (pick_up_tip rebuilds).
        A grasped tube/plate's collider is added to the robot side while held (grasp/release
        rebuild), so it also keeps clear of the static scene (racks, bench, stations, bins)."""
        held = [self._grasp_colliders[self.held_object]] if getattr(self, "held_object", None) else []
        return mink.CollisionAvoidanceLimit(
            self.model, geom_pairs=[(self._robot_geoms + held, self._obs_geoms)],
            minimum_distance_from_collisions=0.005, collision_detection_distance=0.05)

    def _rebuild_collision_limit(self) -> None:
        self.limits[1] = self._collision_limit()

    def _init_tips(self) -> None:
        """Discover the tip-box slots (tip_00.. / tip_site_00..), the bin tips, and the mounted
        tip, and initialise tip-inventory state. Tips are tracked here in code, not in physics."""
        m = self.model
        geom_names = {m.geom(g).name: g for g in range(m.ngeom)}
        site_names = {m.site(i).name for i in range(m.nsite)}
        n = 0
        while f"tip_{n:02d}" in geom_names and f"tip_site_{n:02d}" in site_names:
            n += 1
        self.tip_slots = list(range(n))                      # slot indices 0..n-1
        self.tip_geoms = [geom_names[f"tip_{i:02d}"] for i in self.tip_slots]
        self.tip_sites = [f"tip_site_{i:02d}" for i in self.tip_slots]
        self.used_slots: set[int] = set()
        self.bin_tip_geoms = [geom_names[nm] for i in range(1000)
                              if (nm := f"bintip_{i}") in geom_names]
        self.bin_tips_shown = 0
        self.tip_mounted_vis = geom_names.get("tip_mounted_visual")
        self.tip_mounted_coll = geom_names.get("tip_mounted_collision")
        self.has_tip = False          # a seated disposable tip is on the nozzle
        self.tip_seated = False
        self.tip_contacts: list[str] = []     # sites the current tip has touched (reagents/wells)
        self.tip_history: list[dict] = []     # finished tips: {"slot", "contacts", "seated"}
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
        self.data.qfrc_applied[self.arm_dof] = self.data.qfrc_bias[self.arm_dof]  # gravity feed-forward
        mujoco.mj_step(self.model, self.data)
        self._record_incidents()

    def _solve(self, target):
        self.frame_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.down), np.asarray(target, float)))
        # First from the current pose; if it stalls in a local minimum (e.g. a stretched-out
        # low pose after a deep descend), retry seeded from the arm's neutral home config.
        best_q, best_err = None, np.inf
        for seed in (self.data.qpos, self._home_seed()):
            self.cfg.update(seed)
            err = np.inf
            for _ in range(400):
                try:
                    vel = mink.solve_ik(self.cfg, [self.frame_task, self.posture], 0.02, "daqp",
                                        limits=self.limits)
                except Exception:
                    err = np.inf           # infeasible QP (e.g. out of reach)
                    break
                self.cfg.integrate_inplace(vel, 0.02)
                e = self.frame_task.compute_error(self.cfg)
                err = float(np.linalg.norm(e[:3]))
                if err < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                    break
            if err < best_err:
                best_q, best_err = self.cfg.q[self.arm_qadr].copy(), err
            if best_err <= 3e-3:
                break
        return best_q, best_err

    def _home_seed(self):
        """Current qpos but with the arm joints reset to their neutral home values."""
        seed = self.data.qpos.copy()
        seed[self.arm_qadr] = self.home_qpos[self.arm_qadr]
        return seed

    def _goto(self, target, duration: float):
        q, err = self._solve(target)
        if q is None or err > 3e-3:
            return False, "unreachable"
        dt = self.model.opt.timestep
        q0 = self.data.ctrl[self.arm_act].copy()
        n = max(1, int(duration / dt))
        for k in range(n):
            s = (k + 1) / n
            self.data.ctrl[self.arm_act] = q0 + (3 * s**2 - 2 * s**3) * (q - q0)
            self._step()
        for k in range(int(0.6 / dt)):               # settle
            self._step()
            if k * dt > 0.05 and np.abs(self.data.qvel[self.arm_dof]).max() < 2e-3:
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

    def _safe_z(self) -> float:
        """Travel height for the active point such that whatever is actually held (pipette
        with/without tip, or a grasped tube) clears the tall tubes by its own lowest point,
        not a fixed value."""
        return self.safe_z + self._held_lowest_z_offset()

    def travel_to(self, site: str, clearance: float = 0.04) -> MoveResult:
        """Bring the active tip to `clearance` above `site`, via a safe travel height (based on
        the held object's actual lowest point) that clears the tall tubes: lift fully clear at
        the current (x, y), THEN move sideways at that height, THEN descend vertically onto the
        target -- never a combined diagonal move that could graze a neighbour."""
        target = self.data.site_xpos[self.model.site(site).id].copy() + np.array([0, 0, clearance])
        safe = self._safe_z()
        tp = self.tip()
        if tp[2] < safe - 1e-3:
            ok, r = self._goto([tp[0], tp[1], safe], 0.25)
            if not ok:
                return self._result(False, r, [tp[0], tp[1], safe])
        ok, r = self._goto([target[0], target[1], safe], 0.4)   # sideways, at the safe height
        if not ok:
            return self._result(False, r, target)
        ok, r = self._goto(target, 0.3)              # straight down to the hover point
        return self._result(ok, r, target)

    def descend(self, depth: float, duration: float = 0.8) -> MoveResult:
        """Slow straight-down descent of the active tip by `depth` metres."""
        target = self.tip() - np.array([0, 0, depth])
        ok, r = self._goto(target, duration)
        return self._result(ok, r, target)

    def ascend(self, duration: float = 0.3) -> MoveResult:
        """Straight-up ascent of the active tip back to the safe travel height."""
        tp = self.tip()
        target = np.array([tp[0], tp[1], self._safe_z()])
        ok, r = self._goto(target, duration)
        return self._result(ok, r, target)

    # --------------------------------------------------------- disposable tips
    def _show_mounted_tip(self, visible: bool) -> None:
        """Toggle the nozzle-mounted tip's visibility AND its collider; rebuild avoidance so the
        tip is included in obstacle avoidance only while mounted (mink filters by contype)."""
        self.model.geom_rgba[self.tip_mounted_vis][3] = 1.0 if visible else 0.0
        self.model.geom_contype[self.tip_mounted_coll] = 1 if visible else 0
        self.model.geom_conaffinity[self.tip_mounted_coll] = 1 if visible else 0
        self._rebuild_collision_limit()

    def tip_status(self) -> dict:
        """Tip inventory for the tool layer; a clear result even when the box is empty."""
        remaining = len(self.tip_slots) - len(self.used_slots)
        return {"tips_remaining": remaining, "tips_total": len(self.tip_slots),
                "used_slots": sorted(self.used_slots), "box_empty": remaining == 0,
                "has_tip": self.has_tip, "tip_seated": self.tip_seated,
                "current_tip_contacts": list(self.tip_contacts), "tips_in_bin": self.bin_tips_shown}

    def note_tip_contact(self, site: str) -> None:
        """Record that the mounted tip has touched a container (carry-over tracking)."""
        if self.has_tip and site not in self.tip_contacts:
            self.tip_contacts.append(site)

    def pick_up_tip(self, seat: bool = True, press: float = 0.004) -> MoveResult:
        """Mount the next unused tip: nozzle over the next slot, descend onto the tip top with a
        short downward press, hide that tip in the box and show a tip on the nozzle, then switch
        the active point to the tip end (so IK and travel height follow the whole tip) and extend
        the no-collision region over the tip. `seat=False` injects a 'tip not seated' fault: the
        motion happens but no tip appears and aspiration will fail. Empty box -> clear failure."""
        if self.has_tip:
            return MoveResult(False, "already holding a tip; eject first", self.tip().round(4).tolist())
        nxt = next((i for i in self.tip_slots if i not in self.used_slots
                    and self.model.geom_rgba[self.tip_geoms[i]][3] > 0), None)
        if nxt is None:
            return MoveResult(False, "tip box empty", self.tip().round(4).tolist())
        self.set_active_point("nozzle")
        r = self.travel_to(self.tip_sites[nxt], clearance=0.03)
        if not r.ok:
            return r
        r = self.descend(0.03 + press)                 # onto the tip top, then a short press
        if not r.ok:
            return r
        if not seat:                                   # fault: tip stays in the box, none mounts
            self.has_tip, self.tip_seated = False, False
            self.ascend()
            return MoveResult(False, f"tip not seated (fault) at slot {nxt}", self.tip().round(4).tolist())
        self.used_slots.add(nxt)
        self.model.geom_rgba[self.tip_geoms[nxt]][3] = 0.0   # the picked tip leaves the box
        self._show_mounted_tip(True)
        self.has_tip, self.tip_seated, self.tip_contacts = True, True, []
        self._cur_tip_slot = nxt
        self.set_active_point("tip_end")               # IK + travel height now follow the tip end
        self.ascend()
        return self._result(True, f"tip mounted from slot {nxt}")

    def eject_tip(self) -> MoveResult:
        """Discard the mounted tip into the solid-waste bin: travel over it, hide the mounted tip
        (and its collider), reveal one pre-placed tip in the bin, switch the active point back to
        the nozzle. Records the tip's contact history for carry-over tracking."""
        if not self.has_tip:
            return MoveResult(False, "no tip mounted", self.tip().round(4).tolist())
        r = self.travel_to("waste_solid", clearance=0.06)
        if not r.ok:
            return r
        self._show_mounted_tip(False)
        if self.bin_tips_shown < len(self.bin_tip_geoms):
            self.model.geom_rgba[self.bin_tip_geoms[self.bin_tips_shown]][3] = 1.0
            self.bin_tips_shown += 1
        self.tip_history.append({"slot": getattr(self, "_cur_tip_slot", None),
                                 "contacts": list(self.tip_contacts), "seated": True})
        self.has_tip, self.tip_seated, self.tip_contacts = False, False, []
        self.set_active_point("nozzle")
        self.ascend()
        return self._result(True, "tip ejected into solid waste")

    # --------------------------------------------------------------- gripper
    def open_gripper(self) -> MoveResult:
        self.data.ctrl[self.gripper_act] = 255.0                  # fully open
        for _ in range(int(0.3 / self.model.opt.timestep)):
            self._step()
        return self._result(True, "gripper open")

    def close_gripper(self, gap: float = 2 * GRIP_HALF_M) -> MoveResult:
        """`gap` = total opening between the pads (each finger travels gap/2)."""
        self.data.ctrl[self.gripper_act] = float(np.clip((gap / 2) / 0.04 * 255, 0, 255))
        for _ in range(int(0.3 / self.model.opt.timestep)):
            self._step()
        return self._result(True, f"gripper to {gap * 1000:.0f} mm gap")

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
        q_arm = q[self.arm_qadr]                       # target arm angles from the stored config
        q0 = self.data.ctrl[self.arm_act].copy()
        n = max(1, int(duration / dt))
        for k in range(n):
            s = (k + 1) / n
            self.data.ctrl[self.arm_act] = q0 + (3 * s**2 - 2 * s**3) * (q_arm - q0)
            self._step()
        for k in range(int(0.5 / dt)):
            self._step()
            if k * dt > 0.05 and np.abs(self.data.qvel[self.arm_dof]).max() < 2e-3:
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

    # ------------------------------------------------------------ grasp / weld
    @staticmethod
    def _obj_body(obj: str) -> str:
        if obj.startswith(("tubebody_", "p")) or obj == "plate":
            return obj
        if obj.startswith("reagent_"):
            return "tubebody_" + obj[len("reagent_"):]
        return "tubebody_" + obj

    def _both_pads_touch(self, geom_id: int) -> bool:
        left = right = False
        for c in self.data.contact[:self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if geom_id in pair:
                left = left or bool(pair & self.left_pads)
                right = right or bool(pair & self.right_pads)
        return left and right

    def _set_weld_relpose(self, eq_id: int, body: str) -> None:
        h, t = self.model.body("hand").id, self.model.body(body).id
        hp, hm = self.data.xpos[h], self.data.xmat[h].reshape(3, 3)
        tp, tm = self.data.xpos[t], self.data.xmat[t].reshape(3, 3)
        relq = np.zeros(4); mujoco.mju_mat2Quat(relq, (hm.T @ tm).flatten())
        self.model.eq_data[eq_id][:3] = 0.0                 # anchor
        self.model.eq_data[eq_id][3:6] = hm.T @ (tp - hp)   # relative position (body in hand frame)
        self.model.eq_data[eq_id][6:10] = relq              # relative orientation
        self.model.eq_data[eq_id][10] = 1.0                 # torquescale

    def _world_weld(self, body: str, active: bool) -> None:
        """Weld/unweld a free body to the world at its current pose (holds it in its slot)."""
        e = self.worldwelds[body]
        if active:
            bid = self.model.body(body).id
            q = np.zeros(4); mujoco.mju_mat2Quat(q, self.data.xmat[bid].flatten())
            self.model.eq_data[e][:3] = 0.0
            self.model.eq_data[e][3:6] = self.data.xpos[bid]
            self.model.eq_data[e][6:10] = q
            self.model.eq_data[e][10] = 1.0
        self.data.eq_active[e] = 1 if active else 0

    def grasp(self, obj: str, grip_gap: float | None = None) -> MoveResult:
        """Free `obj` from its slot, close onto it, require both finger pads to touch it, then
        weld it to the hand at the current relative pose. Grip gap defaults to the object's
        diameter minus a small squeeze, from its collider radius."""
        body = self._obj_body(obj)
        if body not in self.welds:
            return MoveResult(False, f"{obj!r} is not graspable", self.tip().round(4).tolist())
        coll = self._collider_for_body(body)
        gap = grip_gap if grip_gap is not None else max(0.004, 2 * self.model.geom_size[coll][0] - 0.004)
        self._world_weld(body, False)                    # free it from its slot
        mujoco.mj_forward(self.model, self.data)
        self.close_gripper(gap)
        if not self._both_pads_touch(coll):
            self._world_weld(body, True)                 # grasp failed -> re-seat it in its slot
            return MoveResult(False, "both finger pads not in contact with the object",
                              self.tip().round(4).tolist())
        self._set_weld_relpose(self.welds[body], body)
        self.data.eq_active[self.welds[body]] = 1
        self.held_object = body
        # whatever it was resting against (its slot/the bench) stays exempt for this hold, same
        # as the destination structure is exempt while placing -- only a SIBLING tube is never OK
        self._held_liftoff_exempt = {
            (c.geom2 if c.geom1 == coll else c.geom1) for c in self.data.contact[:self.data.ncon]
            if c.dist < -1e-4 and coll in (c.geom1, c.geom2)
            and (c.geom2 if c.geom1 == coll else c.geom1) not in self._grasp_collider_ids}
        self._rebuild_collision_limit()        # held object now avoids the static scene too
        return self._result(True, f"grasped {body}")

    def release(self, obj: str | None = None) -> MoveResult:
        body = self._obj_body(obj) if obj else self.held_object
        if body and body in self.welds:
            self.data.eq_active[self.welds[body]] = 0
        self.open_gripper()
        self.held_object = None
        self._held_liftoff_exempt = set()
        self._rebuild_collision_limit()
        return self._result(True, f"released {body}")

    def place(self, obj: str, target_site: str, tol: float = 0.004) -> MoveResult:
        """Carry the held object over `target_site`, lower to just above the slot, check the
        horizontal error is within `tol` BEFORE releasing; then SNAP it kinematically to the
        exact slot pose (upright, seated, zero velocity) and weld it to the world. Fails
        (keeps holding) if misaligned."""
        body = self._obj_body(obj)
        if self.held_object != body:
            return MoveResult(False, f"not holding {body}", self.tip().round(4).tolist())
        tgt = self.data.site_xpos[self.model.site(target_site).id].copy()
        bid = self.model.body(body).id
        seat_z = self._seat_z(body, tgt[2])                      # the slot's floor (rack hole/bench)
        r = self.travel_to(target_site, clearance=0.06)
        if not r.ok:
            return r
        self._placing = True                                    # allow touching the destination
        try:                                                     # structure on this final descent
            r = self.descend(self.data.xpos[bid][2] - seat_z - 0.02)   # ~2 cm above the slot floor
        finally:
            self._placing = False
        if not r.ok:
            return r
        err = float(np.linalg.norm(self.data.xpos[bid][:2] - tgt[:2]))
        if err > tol:
            return MoveResult(False, f"{err * 1000:.1f} mm off target before release (> {tol * 1000:.0f} mm)",
                              self.tip().round(4).tolist(), round(err, 4))
        # SNAP kinematically to the exact slot pose (upright, seated, zero velocity)
        qa, da = self.free_qadr[body], self.free_dofadr[body]
        self.data.qpos[qa:qa + 3] = [tgt[0], tgt[1], seat_z]
        self.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        self.data.qvel[da:da + 6] = 0
        self.data.eq_active[self.welds[body]] = 0               # drop the hand weld
        self.held_object = None
        self._held_liftoff_exempt = set()
        self._rebuild_collision_limit()
        mujoco.mj_forward(self.model, self.data)
        self._world_weld(body, True)                            # hold it in the slot
        self.open_gripper()
        mujoco.mj_forward(self.model, self.data)
        p = self.data.xpos[bid]
        err2 = float(np.linalg.norm(p[:2] - tgt[:2]))
        tilt = float(np.degrees(np.arccos(np.clip(
            self.data.xmat[bid].reshape(3, 3)[:, 2] @ np.array([0, 0, 1.]), -1, 1))))
        seated = err2 <= tol and tilt <= 2.0 and abs(p[2] - seat_z) < 0.002
        reason = None if seated else f"not seated: {err2 * 1000:.1f} mm off, tilt {tilt:.1f} deg, z {p[2] * 1000:.0f} mm"
        hit = self._static_penetration(body)                    # the destination exemption hid
        if seated and hit:                                      # descent contacts; check the result
            seated, reason = False, f"placed {body} penetrates {hit}"
        return MoveResult(seated, reason, p.round(4).tolist(), round(err2, 4), round(tilt, 2))

    def break_weld(self, obj: str | None = None) -> MoveResult:
        """Deactivate a grasp weld WITHOUT opening the gripper -- the noise layer's slip event."""
        body = self._obj_body(obj) if obj else self.held_object
        if body and body in self.welds:
            self.data.eq_active[self.welds[body]] = 0
        if body == self.held_object:
            self.held_object = None
            self._held_liftoff_exempt = set()
            self._rebuild_collision_limit()
        return self._result(True, f"weld broken: {body}")

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
