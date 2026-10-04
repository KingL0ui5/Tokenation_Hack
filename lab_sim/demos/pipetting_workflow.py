"""Video 1 -- clean pipetting workflow (BRIEF.md): the robot does the job with zero collisions and
accurate placement. Hard-coded: calls the skills directly, in the brief's order, so the mix uses
the mounted tip (the backend's mix()/pipette() are not involved):

  1. pick up the pipette from its stand      5. mix in that well with the same tip
  2. pick up a tip                           6. eject the tip into solid waste
  3. aspirate from a reagent tube            7. pick up a fresh tip
  4. dispense into well_A1

Tip error at each aspirate/dispense is measured against the vessel axis at its pipetting depth
(PipetteSkills.vessel_depth_z). Mix cycles between just above the well floor and the well
opening (vessel height from scene_contract()), not mix()'s fixed 5 cm, which would hit the plate.
Travel plays fast; tip seating and every dispense play at 1x.

Run from lab_sim/:  python -m demos.pipetting_workflow [--quick | --full]
                    -> experiments/pipetting_workflow.mp4
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np

from harness.tools.lab_backend import LabBackend
from lab_sim.demos.render import GREEN, RED, Recorder, mode_from_argv

logging.disable(logging.WARNING)

OUT = "experiments/pipetting_workflow.mp4"
TARGET_S = 35
MAX_STEPS = 60_000
SOURCE, DEST = "reagent_water", "well_A1"
MIX_CYCLES = 3
FAST, MID = 1.5, 1.25       # playback speeds for travel / secondary moves (key moments: 1x)


def main() -> int:
    mode = mode_from_argv()
    backend = LabBackend()
    model, data, sk, contract = backend.model, backend.data, backend.skills, backend.contract
    model.cam_fovy[model.camera("side").id] = 60.0      # whole arm in a 16:9 frame (runtime only)

    sk.put_down_pipette()                                # setup (not recorded): pipette in its stand

    st = {"top": "", "bottom": []}

    def overlay():
        n = len(sk.incidents)
        return {"top": st["top"], "bottom": st["bottom"],
                "corner": (f"collisions: {n}", GREEN if n == 0 else RED)}
    rec = Recorder(model, data, OUT, mode, overlay=overlay, views=("side",),
                   sizes={"quick": (640, 360), "full": (1920, 1080)}, show_sites=False,
                   target_s=TARGET_S)
    hold = rec.attach(sk, MAX_STEPS)

    def run(label, res):
        print(f"  {'OK ' if res.ok else 'FAIL'} {label:34} err={res.error_m * 1000:5.1f} mm"
              + (f"  ({res.reason})" if res.reason else ""))
        assert res.ok, f"{label}: {res.reason}"
        return res

    def placement(site, verb):
        """Tip error vs the vessel axis at its pipetting depth, shown at 1x."""
        sp = data.site_xpos[model.site(site).id]
        target = np.array([sp[0], sp[1], sk.vessel_depth_z(site, contract)])
        err = float(np.linalg.norm(sk.tip() - target)) * 1000
        line = (f"{verb} {site} - tip error {err:.1f} mm " + ("OK" if err < 2 else "HIGH"),
                GREEN if err < 2 else RED)
        st["bottom"] = st["bottom"][-1:] + [line]
        print(f"  {line[0]}")
        return err

    def step(n, text):
        st["top"] = f"{n}. {text}"

    rec.speed(1); step(1, "Pick up the pipette from its stand"); hold(1.5)
    rec.speed(MID); run("pick_up_pipette", sk.pick_up_pipette())

    step(2, "Pick up a tip from the tip box")
    rec.speed(FAST); run("pick_up_tip (travel)", sk.travel_to(sk.tip_sites[0], clearance=0.03))
    rec.speed(1)                                          # tip seating at 1x
    with already_there(sk):                               # pick_up_tip re-issues the (done) travel
        run("pick_up_tip", sk.pick_up_tip())
    hold(1.0)

    step(3, f"Aspirate from {SOURCE} (tip enters the tube)")
    rec.speed(FAST); run(f"travel_to({SOURCE})", sk.travel_to(SOURCE, clearance=0.04))
    rec.speed(1); rec.cut("racks")
    run(f"enter_vessel({SOURCE})", sk.enter_vessel(SOURCE, contract))
    sk.note_tip_contact(SOURCE)
    placement(SOURCE, "aspirate"); hold(2.5)
    rec.speed(MID); run("ascend", sk.ascend()); rec.cut("side")

    step(4, f"Dispense into {DEST}")
    rec.speed(FAST); run(f"travel_to({DEST})", sk.travel_to(DEST, clearance=0.04))
    rec.speed(1)
    run(f"enter_vessel({DEST})", sk.enter_vessel(DEST, contract))
    sk.note_tip_contact(DEST)
    placement(DEST, "dispense"); hold(3.0)

    step(5, f"Mix in {DEST} with the same tip ({MIX_CYCLES} cycles, floor <-> opening)")
    floor = sk.vessel_depth_z(DEST, contract)
    g = model.geom(contract.liquid_geom(DEST)).id
    opening = data.geom_xpos[g][2] - model.geom_size[g][1] + contract.vessels["well"].height_m
    stroke = opening - floor
    rec.speed(1)
    for _ in range(MIX_CYCLES):
        run("mix up", sk.descend(-stroke, duration=0.4))
        run("mix down", sk.descend(stroke, duration=0.4))
    hold(1.0)
    rec.speed(MID); run("ascend", sk.ascend())

    step(6, "Eject the tip into solid waste")
    rec.speed(FAST); run("eject_tip", sk.eject_tip())
    rec.speed(1); hold(1.2)
    rec.speed(FAST)

    step(7, "Pick up a fresh tip")
    nxt = min(set(sk.tip_slots) - sk.used_slots)
    run("pick_up_tip (travel)", sk.travel_to(sk.tip_sites[nxt], clearance=0.03))
    rec.speed(1)
    with already_there(sk):
        run("pick_up_tip", sk.pick_up_tip())
    st["bottom"] = st["bottom"] + [(f"done - {len(sk.incidents)} collisions", GREEN)]
    hold(3.0)

    rec.close()
    print(f"incidents: {sk.incidents}")
    assert not sk.incidents, "the clean workflow must have zero collisions"
    return 0


@contextmanager
def already_there(sk):
    """pick_up_tip() starts with travel_to(slot); the demo already made that travel at a fast
    playback speed, so skip the repeat and record only the tip seating at 1x."""
    sk.travel_to = lambda site, clearance=0.04: sk._result(True, None)
    try:
        yield
    finally:
        del sk.travel_to


if __name__ == "__main__":
    raise SystemExit(main())
