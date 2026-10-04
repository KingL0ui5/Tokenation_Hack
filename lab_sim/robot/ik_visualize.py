"""Render the Panda touring lab slots under mink IK, to an MP4 you can scrub.

Headless (off-screen) so it works over SSH / on the render server. For a live
interactive window on macOS instead, run:

    mjpython -m robot.ik_viewer_live

Run from repo root:  python -m robot.ik_visualize   ->   experiments/ik_tour.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mujoco
import numpy as np
import mink

from lab_sim.scenes.build_lab import load_model

logging.disable(logging.WARNING)

OUT = "experiments/ik_tour.mp4"
CAM = "front"
W, H, FPS = 960, 720, 30
STANDOFF = 0.03
STEPS_PER_TARGET = 240
RENDER_EVERY = 4

# A tour that sweeps the whole bench so reachability is visible end to end.
TOUR = ["pipette_grip", "reservoir_enzyme", "well_B3", "well_D1", "well_A6",
        "reservoir_stop", "station_reader", "rack_1", "station_incubator", "waste"]


def site_xpos(model, data, name):
    return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()  # ty: ignore[unresolved-attribute]


def main() -> int:
    model = mujoco.MjModel.from_xml_path(XML)  # ty: ignore[unresolved-attribute]
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    ee = "attachment_site"
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee)  # ty: ignore[unresolved-attribute]

    mujoco.mj_resetDataKeyframe(model, data, 0)  # ty: ignore[unresolved-attribute]
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]
    down_R = data.site_xmat[ee_id].reshape(3, 3).copy()

    frame_task = mink.FrameTask(ee, "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
    posture_task = mink.PostureTask(model, cost=1e-2)
    posture_task.set_target(data.qpos.copy())
    robot_geoms = mink.get_subtree_geom_ids(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0"))  # ty: ignore[unresolved-attribute]
    obstacles = ["bench", "plate_collision", "incubator", "plate_reader", "tube_rack",
                 "pipette_holder", "waste_bin", "collide_reservoir_buffer", "collide_reservoir_enzyme",
                 "collide_reservoir_substrate", "collide_reservoir_inhibitor", "collide_reservoir_stop"]
    obstacle_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in obstacles]  # ty: ignore[unresolved-attribute]
    limits = [mink.ConfigurationLimit(model),
              mink.CollisionAvoidanceLimit(model, geom_pairs=[(robot_geoms, obstacle_geoms)],
                                           minimum_distance_from_collisions=0.005,
                                           collision_detection_distance=0.05)]
    arm_act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]  # ty: ignore[unresolved-attribute]

    # Show the target sites in the render.
    scene_opt = mujoco.MjvOption()  # ty: ignore[unresolved-attribute]
    scene_opt.sitegroup[4] = 1

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))  # ty: ignore[unresolved-attribute]
    renderer = mujoco.Renderer(model, height=H, width=W)
    servo = mink.Configuration(model, data.qpos)

    for slot in TOUR:
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(down_R), site_xpos(model, data, slot) + np.array([0, 0, STANDOFF]))
        frame_task.set_target(target)
        for i in range(STEPS_PER_TARGET):
            servo.update(data.qpos)
            vel = mink.solve_ik(servo, [frame_task, posture_task], 0.01, "daqp", limits=limits)
            servo.integrate_inplace(vel, 0.01)
            data.ctrl[arm_act] = servo.q[:7]
            mujoco.mj_step(model, data)  # ty: ignore[unresolved-attribute]
            if i % RENDER_EVERY == 0:
                renderer.update_scene(data, camera=CAM, scene_option=scene_opt)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"-> {slot}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)

    writer.release()
    renderer.close()
    print(f"wrote {OUT}  ({len(TOUR)} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
