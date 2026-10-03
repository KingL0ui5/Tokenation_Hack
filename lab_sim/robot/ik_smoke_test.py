"""IK reachability smoke test for the lab scene.

Drives the Panda's `attachment_site` to every named slot with a gripper-straight-down
pose, using mink differential IK (posture task + joint limits). Reports per-site
position/orientation error and whether the position *servo* (actuator control, not just
the IK solve) actually reaches the target.

Run from repo root:  python -m robot.ik_smoke_test
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import mink

from scenes.build_lab import load_model

logging.disable(logging.WARNING)  # mink warns that the finger joints sit at their limit; harmless

SOLVER = "daqp"
POS_TOL = 2e-3          # 2 mm
ORI_TOL = np.deg2rad(2)  # 2 deg
IK_STEPS, IK_DT = 400, 0.01
SERVO_STEPS = 600
STANDOFF = 0.03         # hover this far above a slot; the `approach` primitive's pre-grasp offset

# Every slot a primitive will target, grouped so the report is readable.
TARGETS = {
    "plate (near/far corners + centre)": ["well_A1", "well_A6", "well_D1", "well_D6", "well_B3"],
    "reagent tubes (tall 15mL; reach check)": [
        "reagent_dea", "reagent_tris", "reagent_glycine", "reagent_phosphate", "reagent_pnpp",
        "reagent_mgcl2", "reagent_zncl2", "reagent_nacl", "reagent_glycerol", "reagent_water",
        "reagent_naoh", "reagent_pnp_standard"],
    "stations / tools": ["station_reader", "station_incubator", "tip_box"],
    "cold block": ["reagent_enzyme"],
    "waste bins": ["waste_aqueous", "waste_corrosive", "waste_solid"],
    "human zone (expected out of reach)": ["human_zone"],
}


def site_xpos(model, data, name):
    return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()  # ty: ignore[unresolved-attribute]


def main() -> int:
    model = load_model()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]

    ee = "pipette_nozzle"   # the robot holds the pipette; reach is measured at the nozzle tip
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee)  # ty: ignore[unresolved-attribute]

    # Gripper-straight-down orientation = the EE orientation at the home keyframe.
    mujoco.mj_resetDataKeyframe(model, data, 0)  # ty: ignore[unresolved-attribute]
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]
    down_R = data.site_xmat[ee_id].reshape(3, 3).copy()
    home_q = data.qpos.copy()

    configuration = mink.Configuration(model)
    frame_task = mink.FrameTask(ee, "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
    posture_task = mink.PostureTask(model, cost=1e-2)

    # Collision avoidance: keep every robot geom off the fixed lab obstacles, so IK picks a
    # non-colliding arm config instead of the first kinematically-valid one.
    robot_geoms = mink.get_subtree_geom_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0"))  # ty: ignore[unresolved-attribute]
    # Note: the pipette and the reagent tubes are grasp/aspirate *targets*, not fixed
    # obstacles, so they're excluded here (the reagent racks themselves are obstacles).
    obstacles = ["bench", "plate_collision", "incubator", "plate_reader", "cold_block",
                 "bin_aqueous", "bin_corrosive", "bin_solid",
                 "collide_rackA", "collide_rackB", "collide_tipbox"]
    obstacle_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in obstacles]  # ty: ignore[unresolved-attribute]
    collision_limit = mink.CollisionAvoidanceLimit(
        model, geom_pairs=[(robot_geoms, obstacle_geoms)],
        minimum_distance_from_collisions=0.005, collision_detection_distance=0.05)
    limits = [mink.ConfigurationLimit(model), collision_limit]

    arm_act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]  # ty: ignore[unresolved-attribute]
    robot_set, obstacle_set = set(robot_geoms), set(obstacle_geoms)

    def robot_obstacle_contacts():
        hits = []
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            if data.contact[c].dist < -1e-4 and (
                    (g1 in robot_set and g2 in obstacle_set) or (g2 in robot_set and g1 in obstacle_set)):
                other = g2 if g1 in robot_set else g1
                hits.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))  # ty: ignore[unresolved-attribute]
        return hits

    # reach = IK converged AND no robot/obstacle penetration. Gravity sag (steady-state droop
    # of the kp-only position actuators, mostly vertical) is reported but is NOT a failure.
    print(f"{'slot':20} {'ik_err(mm)':>10} {'ori(deg)':>9} {'IK':>3} "
          f"{'sag_h(mm)':>9} {'sag_v(mm)':>9} {'collide':>9} {'reach':>6}")
    print("-" * 82)

    n_ok = n_fail = 0
    for group, slots in TARGETS.items():
        print(f"# {group}")
        for slot in slots:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, slot) < 0:  # ty: ignore[unresolved-attribute]
                print(f"  {slot:18} {'MISSING SITE':>54}   FAIL")
                n_fail += 1
                continue
            # Target pose: slot position, gripper pointing down.
            configuration.update(home_q)
            posture_task.set_target(home_q)
            tgt_xyz = site_xpos(model, data, slot) + np.array([0, 0, STANDOFF])
            target = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(down_R), tgt_xyz)
            frame_task.set_target(target)

            # --- IK solve --- (mink raises if the QP is infeasible, e.g. an out-of-reach
            # target with collision limits; treat that as "unreachable" and report it.)
            infeasible = False
            for _ in range(IK_STEPS):
                try:
                    vel = mink.solve_ik(configuration, [frame_task, posture_task], IK_DT, SOLVER, limits=limits)
                except AssertionError:
                    infeasible = True
                    break
                configuration.integrate_inplace(vel, IK_DT)
                err = frame_task.compute_error(configuration)
                if np.linalg.norm(err[:3]) < POS_TOL and np.linalg.norm(err[3:]) < ORI_TOL:
                    break
            pos_err = np.linalg.norm(err[:3])
            ori_err = np.linalg.norm(err[3:])
            ik_ok = (not infeasible) and pos_err < POS_TOL and ori_err < ORI_TOL

            # --- servo check: closed-loop, exactly as a primitive runs (skipped if IK failed) ---
            sag_h = sag_v = float("nan")
            collisions = []
            if ik_ok:
                data.qpos[:] = home_q
                data.qvel[:] = 0
                servo = mink.Configuration(model, data.qpos)
                for _ in range(SERVO_STEPS):
                    servo.update(data.qpos)
                    try:
                        vel = mink.solve_ik(servo, [frame_task, posture_task], IK_DT, SOLVER, limits=limits)
                    except AssertionError:
                        break
                    servo.integrate_inplace(vel, IK_DT)
                    data.ctrl[arm_act] = servo.q[:7]
                    mujoco.mj_step(model, data)  # ty: ignore[unresolved-attribute]
                delta = site_xpos(model, data, ee) - (site_xpos(model, data, slot) + np.array([0, 0, STANDOFF]))
                sag_h, sag_v = np.linalg.norm(delta[:2]), abs(delta[2])
                collisions = robot_obstacle_contacts()

            reach = "OK" if (ik_ok and not collisions) else "FAIL"
            n_ok += reach == "OK"
            n_fail += reach == "FAIL"
            coll_str = collisions[0] if collisions else "-"
            print(f"  {slot:18} {pos_err*1e3:10.2f} {np.rad2deg(ori_err):9.2f} "
                  f"{'y' if ik_ok else 'n':>3} {sag_h*1e3:9.1f} {sag_v*1e3:9.1f} {coll_str:>9} {reach:>6}")

    print("-" * 70)
    print(f"reachable: {n_ok}  |  unreachable/failed: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
