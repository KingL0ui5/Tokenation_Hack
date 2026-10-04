"""Video 3 -- knock-over -> halt + human alert (BRIEF.md). Hard-coded, deterministic:

  1. the robot carries tube zncl2 out of rack B (pipette parked in its stand);
  2. INJECTED FAULT: the enzyme tube's world weld is released (unsecured) and the carry is
     driven low, straight through it;
  3. the per-step contact check flags the collision -> the tube flashes red, the arm stops
     (PipetteSkills.halt_on_incident, 100 ms stop reaction time) and the tube topples;
  4. the backend's latched safety stop engages; a simulated camera check compares the bench
     with the scene map and names what it sees ("tube near tube_enzyme's slot knocked over
     (tilt 90 deg)"); the robot retreats home;
  5. banner: EXPERIMENT HALTED - collision with tube_enzyme + the camera check. Human help requested.
  6. the next dispense (then a mix and a tip change) are attempted and REFUSED by the backend, with the refusal
     text and the refused-attempt count on screen.

Carry/retreat play fast; contact, topple, banner and refusals at 1x.

Run from lab_sim/:  python -m demos.knock_over [--quick | --full]  ->  experiments/knock_over.mp4
"""

from __future__ import annotations

import logging
import textwrap

import mujoco
import numpy as np

from harness.tools.lab_backend import LabBackend
from lab_sim.demos.render import AMBER, GREEN, RED, WHITE, Recorder, mode_from_argv

logging.disable(logging.WARNING)

OUT = "experiments/knock_over.mp4"
TARGET_S = 45
MAX_STEPS = 60_000
CARRIED, VICTIM = "zncl2", "enzyme"
FAST = 1.25
STOP_REACTION_S = 0.10     # deterministic delay between detection and the arm holding still
CARRY_Z = 0.09             # carried tube's base height for the low pass (victim spans 0.03-0.15)


def main() -> int:
    mode = mode_from_argv()
    backend = LabBackend()
    model, data, sk = backend.model, backend.data, backend.skills
    sk.halt_latency_s = STOP_REACTION_S
    model.cam_fovy[model.camera("side").id] = 60.0      # whole arm in a 16:9 frame (runtime only)
    victim_body = f"tubebody_{VICTIM}"
    victim_coll = f"collide_tube_{VICTIM}"
    victim_geoms = [model.geom(f"tube_{VICTIM}").id, model.geom(f"liquid_reagent_{VICTIM}").id]

    st = {"top": "", "bottom": [], "banner": None, "flashed": False}

    def overlay():
        n = len(sk.incidents)
        status = backend.safety_stop
        corner = (f"SAFETY STOP latched | refused: {backend.refused_attempts}", RED) if status else \
                 (f"collisions: {n}", GREEN if n == 0 else RED)
        return {"top": st["top"], "bottom": st["bottom"], "banner": st["banner"], "corner": corner}
    rec = Recorder(model, data, OUT, mode, overlay=overlay, views=("side",),
                   sizes={"quick": (640, 360), "full": (1920, 1080)}, show_sites=False,
                   target_s=TARGET_S)
    hold = rec.attach(sk, MAX_STEPS)

    inner = sk._step

    def step_and_flash():                                 # flash the hit tube red on contact
        inner()
        if not st["flashed"] and any(i["other"] == victim_coll for i in sk.incidents):
            st["flashed"] = True
            for g in victim_geoms:
                model.geom_matid[g] = -1
                model.geom_rgba[g] = [1.0, 0.1, 0.1, 1.0]
    sk._step = step_and_flash

    def run(label, res):
        print(f"  {'OK ' if res.ok else 'FAIL'} {label:30}" + (f"  ({res.reason})" if res.reason else ""))
        return res

    # 1. carry a tube out of rack B --------------------------------------------------------
    st["top"] = f"Carrying tube {CARRIED} out of rack B"
    hold(1.5)
    rec.speed(FAST)
    run("put_down_pipette", sk.put_down_pipette())
    sk.close_gripper(0.022)
    run(f"travel_to(tube_grip_{CARRIED})", sk.travel_to(f"tube_grip_{CARRIED}", clearance=0.04))
    run("descend", sk.descend(0.04))
    run(f"grasp({CARRIED})", sk.grasp(CARRIED))
    run("ascend", sk.ascend())

    # 2. injected fault: unsecured tube in the carry path --------------------------------
    rec.speed(1)
    st["top"] = f"INJECTED FAULT: tube_{VICTIM} unsecured, carry path driven through it"
    st["bottom"] = [("injected fault - deterministic, for this demo", AMBER)]
    sk._world_weld(victim_body, False)
    mujoco.mj_forward(model, data)
    hold(3.0)
    e = data.xpos[model.body(victim_body).id].copy()
    off = sk._held_lowest_z_offset()
    rec.speed(FAST)
    run("travel (high)", _goto(sk, [e[0], e[1] - 0.09, sk._safe_z()], 0.5))
    rec.speed(1)
    run("travel (low)", _goto(sk, [e[0], e[1] - 0.09, CARRY_Z + off], 0.5))
    r = run("carry through", _goto(sk, [e[0], e[1] + 0.07, CARRY_Z + off], 0.6))

    # 3. detection + stop ----------------------------------------------------------------
    status = backend.safety_status()
    assert status["halted"] and not r.ok, (status, r.reason)
    col = status["collision"]
    st["top"] = "DETECTED: collision - motion stopped"
    st["bottom"] = [("injected fault - deterministic, for this demo", AMBER),
                    (f"{r.reason}", RED),
                    (f"safety stop latched: collision with {col['other']}", RED)]
    hold(4.0)                                             # the tube topples at 1x
    tilt = _tilt(data, model, victim_body)
    print(f"  {VICTIM} tilt after the hit: {tilt:.0f} deg")
    assert tilt > 45, "the unsecured tube should visibly topple"
    cam = _deg(backend.safety_status()["collision"]["camera_check"])
    print(f"  camera check: {cam}")
    st["top"] = "Camera check (simulated): bench vs scene map"
    st["bottom"] = st["bottom"][1:] + [(f"camera check (simulated): {cam}", AMBER)]
    hold(3.0)

    # 4. retreat home ----------------------------------------------------------------------
    rec.speed(FAST)
    st["top"] = "Safety response: retreat to home"
    sk.halt_on_incident = False                           # the retreat IS the safety response
    run("ascend", sk.ascend())
    sk._goto_qpos(sk.home_qpos, duration=1.0)

    # 5. halt banner -----------------------------------------------------------------------
    rec.speed(1)
    st["top"] = "EXPERIMENT HALTED"
    st["banner"] = (f"EXPERIMENT HALTED - collision with {col['other']}.\n"
                    f"camera check (simulated): {cam}\nHuman help requested.", RED)
    hold(5.0)

    # 6. next step attempted -> refused ------------------------------------------------------
    st["banner"] = None
    for tool, call in (("dispense water -> well_A1", lambda: backend.pipette("reagent_water", "well_A1")),
                       ("mix well_A1", lambda: backend.mix("well_A1", 3)),
                       ("change tip", backend.change_tip)):
        st["top"] = f"Next step attempted: {tool}"
        out = call()
        print(f"  {tool}: {out}")
        assert out.get("refused") and not out["ok"]
        lines = textwrap.wrap(_deg(out["reason"]), 70)
        st["bottom"] = [(f"{tool}: REFUSED", RED)] + [(ln, WHITE) for ln in lines]
        hold(5.0)
    st["top"] = "Halted until a human clears the bench and resets the safety stop"
    hold(3.0)

    rec.close()
    print(f"refused attempts: {backend.refused_attempts}; incidents: {sk.incidents}")
    return 0


class _R:
    def __init__(self, ok, reason):
        self.ok, self.reason = ok, reason


def _goto(sk, target, duration):
    """A direct, unsafe-path move (the injected fault skips travel_to's lift-clear shape)."""
    return _R(*sk._goto(np.asarray(target, float), duration))


def _deg(text: str) -> str:
    return text.replace("°", " deg")           # OpenCV's Hershey font has no degree sign


def _tilt(data, model, body) -> float:
    z = data.xmat[model.body(body).id].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))


if __name__ == "__main__":
    raise SystemExit(main())
