"""Verify held-object collision checking end to end and record an MP4 to watch.

Three scenes, matching the acceptance tests in tests/test_held_object_collisions.py:
  (a) carry a tube from rack A to the spare hole in rack B -- zero incidents.
  (b) the mounted pipette, with a disposable tip, travelling between rack A, rack B and a
      well, entering each vessel to its pipetting depth -- zero incidents.
  (c) a deliberate collision: the same carried tube driven sideways through a neighbour's
      slot (skipping the safe lift) -- caught and reported via PipetteSkills.incidents, the
      same channel harness/tools/lab_backend.py drains to fail a tool.

Run from lab_sim/:  python -m demos.held_object_collisions -> experiments/held_object_collisions.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np

from lab_sim.demos.grid import GridRecorder
from lab_sim.scenes.build_lab import load_model, scene_contract
from lab_sim.robot.skills import PipetteSkills

logging.disable(logging.WARNING)

OUT = "experiments/held_object_collisions.mp4"
RENDER_EVERY = 8      # 4 views per frame are costly; 8 steps @ 15 fps keeps the same playback speed
MAX_STEPS = 60_000       # hard cap on physics steps (~2 min sim) so the demo can never hang


def main() -> int:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    contract = scene_contract(model)
    sk = PipetteSkills(model, data, contract.obstacles, safe_z=0.22)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)

    def closeup_target():          # the carried tube while grasped, else the active pipette point
        if sk.held_object:
            return data.xpos[model.body(sk.held_object).id] + np.array([0, 0, 0.06])
        return sk.tip()
    rec = GridRecorder(model, data, OUT, closeup_target, fps=15)
    label = {"text": ""}

    def render():
        n = len(sk.incidents)
        rec.frame(label["text"], f"incidents: {n}", (120, 220, 120) if n == 0 else (80, 80, 255))

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
    label["text"] = "(b) pipette+tip entering: rack A -> rack B -> well"
    show("pick_up_tip", sk.pick_up_tip()); hold(0.2)
    for site, where in (("reagent_tris", "rack A"), ("reagent_zncl2", "rack B"), ("well_A1", "plate")):
        show(f"travel_to({site}) [{where}]", sk.travel_to(site, clearance=0.04))
        show(f"enter_vessel({site})", sk.enter_vessel(site, contract)); hold(0.3)
        show("ascend", sk.ascend())
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

    rec.close()
    print(f"total incidents recorded: {len(sk.incidents)} (expected: 1, from scene (c) only)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
