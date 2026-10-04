"""Live interactive viewer of the composed lab (bench + Panda) via load_model().

macOS needs mjpython for the GUI loop:

    mjpython -m scenes.view

Drag to orbit, scroll to zoom. Shows the static scene (no motion); for the IK tour use
`mjpython -m robot.ik_viewer_live`.
"""

from __future__ import annotations

import mujoco
import mujoco.viewer

from lab_sim.scenes.build_lab import load_model


def main() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # home pose
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.sitegroup[4] = 1               # show named slot/EE sites
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
