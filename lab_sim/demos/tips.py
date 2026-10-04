"""Verify the disposable-tip skills end to end and record an MP4 to watch.

Sequence: pick_up_tip -> aspirate from a reagent tube -> dispense into a well -> eject_tip,
then a final pick_up_tip(seat=False) to show the 'tip not seated' fault. Frames are captured
by wrapping PipetteSkills._step so the whole motion is visible (not just end poses).

Run from lab_sim/:  python -m demos.tips   ->   experiments/tips.mp4 (gitignored; regenerate anytime)
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco

from demos.grid import GridRecorder
from scenes.build_lab import TIP_LEN, load_model, scene_contract
from robot.skills import PipetteSkills

logging.disable(logging.WARNING)

OUT = "experiments/tips.mp4"
RENDER_EVERY = 8      # 4 views per frame are costly; 8 steps @ 15 fps keeps the same playback speed
MAX_STEPS = 40_000       # hard cap on physics steps so the demo can never hang


def main() -> int:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    contract = scene_contract(model)
    sk = PipetteSkills(model, data, contract.obstacles, safe_z=0.22)

    # tip length -> 200 uL tip -> report the volume it corresponds to
    tip_len_mm = TIP_LEN * 1000
    print(f"tip length = {tip_len_mm:.0f} mm  (AutoBio tip_200ul -> 200 uL capacity)")
    print(f"pipette_tip_end at nozzle - {tip_len_mm:.0f} mm; tip box has {len(sk.tip_slots)} slots")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    rec = GridRecorder(model, data, OUT, sk.tip, fps=15)      # close-up follows the active tip point
    label = {"text": ""}

    def render():
        st = sk.tip_status()
        rec.frame(label["text"], f"tips left: {st['tips_remaining']}  has_tip: {st['has_tip']}")

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

    def hold(seconds=0.5):
        for _ in range(int(seconds / model.opt.timestep)):
            step_and_render()

    def show(tag, res):
        flag = "OK " if res.ok else "FAIL"
        print(f"  {flag} {tag:30} err={res.error_m*1000:6.1f}mm tilt={res.tilt_deg:.2f}deg"
              + (f"  ({res.reason})" if res.reason else ""))

    label["text"] = "start"; hold(0.5)

    label["text"] = "pick_up_tip()"
    show("pick_up_tip", sk.pick_up_tip()); hold(0.3)

    label["text"] = "aspirate: reagent_water"
    show("travel_to(reagent_water)", sk.travel_to("reagent_water", clearance=0.04))
    show("enter_vessel(reagent_water)", sk.enter_vessel("reagent_water", contract))
    sk.note_tip_contact("reagent_water")
    hold(0.3)
    show("ascend", sk.ascend())

    label["text"] = "dispense: well_B3"
    show("travel_to(well_B3)", sk.travel_to("well_B3", clearance=0.04))
    show("enter_vessel(well_B3)", sk.enter_vessel("well_B3", contract))
    sk.note_tip_contact("well_B3")
    hold(0.3)
    show("ascend", sk.ascend())

    label["text"] = "eject_tip()"
    show("eject_tip", sk.eject_tip()); hold(0.3)

    label["text"] = "fault: pick_up_tip(seat=False)"
    show("pick_up_tip(seat=False)", sk.pick_up_tip(seat=False)); hold(0.4)

    rec.close()
    print("tip_status:", sk.tip_status())
    print("tip_history:", sk.tip_history)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
