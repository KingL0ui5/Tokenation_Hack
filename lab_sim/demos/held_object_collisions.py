"""Verify held-object collision checking end to end and record an MP4 to watch.

Three scenes, matching the acceptance tests in tests/test_held_object_collisions.py:
  (a) carry a tube from rack A to the spare hole in rack B -- zero incidents.
  (b) the mounted pipette, with a disposable tip, travelling between rack A, rack B and a
      well -- zero incidents.
  (c) a deliberate collision: the same carried tube driven sideways through a neighbour's
      slot (skipping the safe lift) -- caught and reported via PipetteSkills.incidents, the
      same channel harness/tools/lab_backend.py drains to fail a tool.

Run from lab_sim/:  python -m demos.held_object_collisions -> experiments/held_object_collisions.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mujoco

from scenes.build_lab import load_model, scene_contract
from robot.skills import PipetteSkills

logging.disable(logging.WARNING)

OUT = "experiments/held_object_collisions.mp4"
CAM = "racks"
W, H, FPS = 960, 720, 30
RENDER_EVERY = 4
MAX_STEPS = 60_000       # hard cap on physics steps (~2 min sim) so the demo can never hang


def main() -> int:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    contract = scene_contract(model)
    sk = PipetteSkills(model, data, contract.obstacles, safe_z=0.22)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.sitegroup[4] = 1
    label = {"text": ""}

    def render():
        renderer.update_scene(data, camera=CAM, scene_option=opt)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, label["text"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        n = len(sk.incidents)
        colour = (120, 220, 120) if n == 0 else (80, 80, 255)
        cv2.putText(frame, f"incidents: {n}", (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    colour, 2, cv2.LINE_AA)
        writer.write(frame)

    orig_step = sk._step
    counter = {"n": 0}

    def step_and_render():
        if counter["n"] >= MAX_STEPS:
            raise RuntimeError(f"step limit ({MAX_STEPS}) exceeded")
        orig_step()
        counter["n"] += 1
        if counter["n"] % RENDER_EVERY == 0:
            render()
    sk._step = step_and_render

    def hold(seconds=0.4):
        for _ in range(int(seconds / model.opt.timestep)):
            step_and_render()

    def show(tag, res):
        flag = "OK " if res.ok else "FAIL"
        print(f"  {flag} {tag:34} err={res.error_m*1000:6.1f}mm tilt={res.tilt_deg:.2f}deg"
              + (f"  ({res.reason})" if res.reason else ""))
        return res.ok

    label["text"] = "start"; hold(0.3)

    # ---- (a) carry a tube: rack A -> spare hole in rack B -------------------------------
    label["text"] = "(a) carry tube: rack A -> spare hole, rack B"
    show("put_down_pipette", sk.put_down_pipette()); hold(0.2)
    show("close_gripper (narrow, threading between tubes)", sk.close_gripper(0.022))
    show("travel_to(tube_grip_dea)", sk.travel_to("tube_grip_dea", clearance=0.04))
    show("descend onto dea", sk.descend(0.04))
    show("grasp(dea)", sk.grasp("dea"))
    show("place(dea, spare_hole_b)", sk.place("dea", "spare_hole_b"))
    show("pick_up_pipette", sk.pick_up_pipette()); hold(0.3)
    print(f"  incidents after (a): {sk.incidents}")
    assert sk.incidents == [], "carry should be clean"

    # ---- (b) pipette + tip travelling between racks -------------------------------------
    label["text"] = "(b) pipette+tip travelling: rack A -> rack B -> well"
    show("pick_up_tip", sk.pick_up_tip()); hold(0.2)
    show("travel_to(reagent_tris)  [rack A]", sk.travel_to("reagent_tris", clearance=0.04))
    show("travel_to(reagent_zncl2) [rack B]", sk.travel_to("reagent_zncl2", clearance=0.04))
    show("travel_to(well_A1)       [plate]", sk.travel_to("well_A1", clearance=0.04))
    show("eject_tip", sk.eject_tip()); hold(0.3)
    print(f"  incidents after (b): {sk.incidents}")
    assert sk.incidents == [], "tip travel should be clean"

    # ---- (c) deliberate collision: ram a held tube through a neighbour's slot -----------
    label["text"] = "(c) deliberate collision: glycine rammed through tris's slot"
    show("put_down_pipette", sk.put_down_pipette()); hold(0.2)
    show("close_gripper (narrow)", sk.close_gripper(0.022))
    show("travel_to(tube_grip_glycine)", sk.travel_to("tube_grip_glycine", clearance=0.04))
    show("descend onto glycine", sk.descend(0.04))
    show("grasp(glycine)", sk.grasp("glycine"))
    sk.ascend()
    hold(0.2)
    before = len(sk.incidents)
    label["text"] = "(c) skipping the safe lift -- ramming sideways, low"
    tris_id = model.body("tubebody_tris").id
    target_z = data.xpos[tris_id][2] + 0.05 + sk._held_lowest_z_offset()
    sk._goto([data.xpos[tris_id][0], data.xpos[tris_id][1], target_z], 0.8)
    hold(0.4)
    new = sk.incidents[before:]
    print(f"  deliberate collision caught: {new}")
    assert new, "the rammed tube must be caught as an incident"
    label["text"] = f"CAUGHT: {new[0]['carried']} vs {new[0]['other']}"
    hold(0.6)

    writer.release()
    renderer.close()
    print(f"total incidents recorded: {len(sk.incidents)} (expected: 1, from scene (c) only)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
