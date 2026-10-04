"""Standalone test for robot/skills.py (no backend, no tools).

Loads the scene, mounts are already baked into load_model(), then exercises each skill and
prints its MoveResult (ok, actual tip position, error from target, tilt). Verifies the pipette
stays within tolerance of vertical and that the nozzle reaches into a reagent tube and a well.

Run from lab_sim/:  python -m robot.skills_test
"""

from __future__ import annotations

import mujoco

from lab_sim.scenes.build_lab import load_model, scene_contract
from lab_sim.robot.skills import PipetteSkills


def main() -> int:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    contract = scene_contract(model)
    sk = PipetteSkills(model, data, contract.obstacles, safe_z=0.22)

    def show(label, res):
        flag = "OK " if res.ok else "FAIL"
        print(f"  {flag} {label:28} tip={res.tip_pos} err={res.error_m*1000:6.1f}mm tilt={res.tilt_deg:.2f}deg"
              + (f"  ({res.reason})" if res.reason else ""))
        return res.ok

    ok = True
    print("active point:")
    ok &= show("set_active_point(nozzle)", sk.set_active_point("nozzle"))

    print("tip_error (no motion):")
    for site in ("well_B3", "reagent_water", "reagent_enzyme"):
        show(f"tip_error({site})", sk.tip_error(site))

    print("aspirate cycle at a reagent tube (reagent_water):")
    ok &= show("travel_to(reagent_water)", sk.travel_to("reagent_water", clearance=0.04))
    ok &= show("descend(0.055)", sk.descend(0.055))
    ok &= show("ascend()", sk.ascend())

    print("dispense cycle at a well (well_B3):")
    ok &= show("travel_to(well_B3)", sk.travel_to("well_B3", clearance=0.04))
    ok &= show("descend(0.055)", sk.descend(0.055))
    ok &= show("ascend()", sk.ascend())

    print("switch active point to the disposable-tip end:")
    ok &= show("set_active_point(tip_end)", sk.set_active_point("tip_end"))
    show("tip_error(well_B3) from tip_end", sk.tip_error("well_B3"))

    print("\n" + ("ALL SKILLS OK" if ok else "SOME SKILLS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
