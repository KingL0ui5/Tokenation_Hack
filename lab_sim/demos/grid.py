"""2x2 multi-view video recorder shared by the demos: overview | front / side | close-up.

The overview is a free camera on the bench centre, pulled back until the whole bench (and the
arm) is in frame; front/side are the scene's named cameras (scenes/build_lab.py); the close-up
is a free camera that follows `closeup_target()` (e.g. the active pipette tip, or a carried
tube) each frame. Each view is 640x480 (within lab.xml's 1280x960 offscreen buffer).
"""

from __future__ import annotations

from typing import Callable

import cv2
import mujoco
import numpy as np

from lab_sim.scenes.build_lab import BENCH_CENTER

TILE_W, TILE_H = 640, 480        # 4:3 -- the aspect the overview camera is fitted to
VIEWS = ("overview", "front", "side", "close-up")


class GridRecorder:
    def __init__(self, model, data, out: str, closeup_target: Callable[[], np.ndarray],
                 fps: int = 30, closeup_distance: float = 0.35):
        self.model, self.data, self.target = model, data, closeup_target
        self.writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                      (2 * TILE_W, 2 * TILE_H))
        self.renderer = mujoco.Renderer(model, height=TILE_H, width=TILE_W)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[4] = 1
        self.overview = self._free_cam([*BENCH_CENTER, -0.1], 2.4, 135.0, -35.0)
        self.closeup = self._free_cam([0, 0, 0], closeup_distance, 200.0, -30.0)

    @staticmethod
    def _free_cam(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        return cam

    def _tile(self, view: str) -> np.ndarray:
        if view == "close-up":
            self.closeup.lookat[:] = self.target()
        cam = {"overview": self.overview, "close-up": self.closeup}.get(view, view)
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, view, (12, TILE_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def frame(self, label: str = "", footer: str = "", footer_colour=(200, 255, 200)) -> None:
        t = [self._tile(v) for v in VIEWS]
        img = np.vstack([np.hstack(t[:2]), np.hstack(t[2:])])
        cv2.line(img, (TILE_W, 0), (TILE_W, 2 * TILE_H), (40, 40, 40), 2)
        cv2.line(img, (0, TILE_H), (2 * TILE_W, TILE_H), (40, 40, 40), 2)
        cv2.putText(img, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if footer:
            cv2.putText(img, footer, (TILE_W + 16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        footer_colour, 2, cv2.LINE_AA)
        self.writer.write(img)

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()
