"""Verify held-object collision checking end to end and record an MP4 to watch.

Three scenes, matching the acceptance tests in tests/test_held_object_collisions.py:
  (a) carry a tube from rack A to the spare hole in rack B -- zero incidents.
  (b) two LabBackend.pipette() transfers (rack A -> well, rack B -> well) with ONE disposable
      tip that stays on between them (main's keep-the-tip behaviour), entering each vessel to
      its pipetting depth -- zero incidents.
  (c) a deliberate collision: the same carried tube driven sideways through a neighbour's
      slot (skipping the safe lift) -- caught and reported via PipetteSkills.incidents, the
      same channel harness/tools/lab_backend.py drains to fail a tool.

Run from lab_sim/:  python -m demos.held_object_collisions [--quick | --full]
                    -> experiments/held_object_collisions.mp4 (see demos/render.py for the modes)
"""

from __future__ import annotations

import logging

import numpy as np

from harness.tools.lab_backend import LabBackend
from lab_sim.demos.render import Recorder, mode_from_argv

logging.disable(logging.WARNING)

OUT = "experiments/held_object_collisions.mp4"
MAX_STEPS = 60_000       # hard cap on physics steps (~2 min sim) so the demo can never hang


def main() -> int:
    mode = mode_from_argv()
    backend = LabBackend()
    model, data, sk = backend.model, backend.data, backend.skills

    def closeup_target():          # the carried tube while grasped, else the active pipette point
        if sk.held_object:
            return data.xpos[model.body(sk.held_object).id] + np.array([0, 0, 0.06])
        return sk.tip()
    label = {"text": ""}

    def overlay():
        n = len(sk.incidents)
        tip = f"  tip: slot {sk._cur_tip_slot}" if sk.has_tip else ""
        return label["text"], f"incidents: {n}{tip}", (120, 220, 120) if n == 0 else (80, 80, 255)
    rec = Recorder(model, data, OUT, mode, closeup_target, overlay)

    orig_step = sk._step
    counter = {"n": 0}

    def step_and_render():
        if counter["n"] >= MAX_STEPS:
            raise RuntimeError(f"step limit ({MAX_STEPS}) exceeded")
        orig_step()
        counter["n"] += 1
        rec.step()
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
    slots = []
    for src, dst, where in (("reagent_tris", "well_A1", "rack A"), ("reagent_zncl2", "well_A2", "rack B")):
        label["text"] = f"(b) pipette {src} [{where}] -> {dst}, tip kept on"
        r = backend.pipette(src, dst)
        slots.append(r["tip_slot"])
        print(f"  {'OK ' if r['ok'] else 'FAIL'} pipette({src} -> {dst})  tip_slot={r['tip_slot']}"
              f"  touched={r['touched']}" + (f"  ({r['reason']})" if not r["ok"] else ""))
        assert r["ok"], r.get("reason")
        hold(0.3)
    assert slots[0] == slots[1] and sk.has_tip, "the tip should stay on between transfers"
    label["text"] = "(b) done: eject the (carried-over) tip"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
