"""Shared video recorder for every demo in lab_sim/demos/.

Frames are captured by SIMULATED time (FPS frames per simulated second, so playback is
real-time), not per physics step -- call `step()` after every physics step and it renders only
when the next frame is due.

Two modes, chosen on the demo's command line (`mode_from_argv()`):
  --quick (default): the overview camera only, at 480x360. For checking a change.
  --full:            the 2x2 grid overview | front / side | close-up, 640x480 per view. Only for
                     final demo videos (4 renders per frame).
The overview is a free camera on the bench centre, pulled back until the whole bench (and the
arm) is in frame; front/side are the scene's named cameras (scenes/build_lab.py); the close-up is
a free camera that follows `closeup_target()` (e.g. the active pipette tip, or a carried tube).
`close()` prints the render time. Tests never render.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import cv2
import mujoco
import numpy as np

from lab_sim.scenes.build_lab import BENCH_CENTER

FPS = 30
MODES = {"quick": ((480, 360), ("overview",)),
         "full": ((640, 480), ("overview", "front", "side", "close-up"))}


def mode_from_argv(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", dest="mode", action="store_const", const="quick",
                   help="overview only, 480x360 (default)")
    g.add_argument("--full", dest="mode", action="store_const", const="full",
                   help="2x2 grid, 640x480 per view -- final demo videos only")
    return ap.parse_args(argv).mode or "quick"


def _free_cam(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


class Recorder:
    def __init__(self, model, data, out: str, mode: str = "quick",
                 closeup_target: Callable[[], np.ndarray] | None = None,
                 overlay: Callable[[], tuple[str, str, tuple]] | None = None,
                 closeup_distance: float = 0.35):
        """`overlay()` -> (label, footer, footer BGR colour), drawn on every frame."""
        (self.w, self.h), self.views = MODES[mode]
        self.mode, self.model, self.data = mode, model, data
        self.target, self.overlay = closeup_target, overlay or (lambda: ("", "", (0, 0, 0)))
        cols = 2 if len(self.views) > 1 else 1
        rows = (len(self.views) + cols - 1) // cols
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                      (cols * self.w, rows * self.h))
        self.renderer = mujoco.Renderer(model, height=self.h, width=self.w)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[4] = 1
        self.cams = {"overview": _free_cam([*BENCH_CENTER, -0.1], 2.4, 135.0, -35.0),
                     "close-up": _free_cam([0, 0, 0], closeup_distance, 200.0, -30.0)}
        self.next_t = data.time
        self.frames, self.render_s, self.t0 = 0, 0.0, time.perf_counter()

    def step(self) -> None:
        """Call after each physics step; renders a frame only when one is due in sim time."""
        if self.data.time + 1e-9 >= self.next_t:
            self.next_t += 1.0 / FPS
            self.frame()

    def _tile(self, view: str) -> np.ndarray:
        if view == "close-up" and self.target is not None:
            self.cams["close-up"].lookat[:] = self.target()
        self.renderer.update_scene(self.data, camera=self.cams.get(view, view), scene_option=self.opt)
        img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        if len(self.views) > 1:
            cv2.putText(img, view, (12, self.h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def frame(self) -> None:
        t = time.perf_counter()
        tiles = [self._tile(v) for v in self.views]
        if len(tiles) == 1:
            img = tiles[0]
        else:
            img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            cv2.line(img, (self.w, 0), (self.w, 2 * self.h), (40, 40, 40), 2)
            cv2.line(img, (0, self.h), (2 * self.w, self.h), (40, 40, 40), 2)
        label, footer, colour = self.overlay()
        scale = 0.85 if len(tiles) > 1 else 0.5
        cv2.putText(img, label, (12, int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2 if len(tiles) > 1 else 1, cv2.LINE_AA)
        if footer:
            x, y = (self.w + 16, 34) if len(tiles) > 1 else (12, self.h - 12)
            cv2.putText(img, footer, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.9, colour,
                        2 if len(tiles) > 1 else 1, cv2.LINE_AA)
        self.writer.write(img)
        self.frames += 1
        self.render_s += time.perf_counter() - t

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()
        total = time.perf_counter() - self.t0
        print(f"wrote {self.out}  [{self.mode}: {self.frames} frames @ {FPS} fps sim time]  "
              f"render {self.render_s:.1f} s of {total:.1f} s total")
