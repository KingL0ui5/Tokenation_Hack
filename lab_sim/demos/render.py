"""Shared video recorder for every demo in lab_sim/demos/.

Frames are captured by SIMULATED time, never by shortcutting the simulation: at playback speed
S (`speed(S)`, default 1) one frame is captured every S/FPS simulated seconds, and the output is
always FPS (30) fps -- so S=1 plays in real time and S=6 plays six times faster. Mark travel
segments fast and key moments `speed(1)`; a small "6x" label shows whenever S > 1. Call `step()`
after every physics step; it renders only when the next frame is due.

Modes, chosen on the demo's command line (`mode_from_argv()`):
  --quick (default): low resolution, for iterating.
  --full:            the final take. Legacy demos: the 2x2 grid overview | front / side |
                     close-up at 640x480 per view. Demos that pass `views=`/`sizes=` choose their
                     own (e.g. the side camera at 1920x1080).
The overview is a free camera on the bench centre, pulled back until the whole bench (and the
arm) is in frame; front/side/racks are the scene's named cameras (scenes/build_lab.py); the
close-up is a free camera that follows `closeup_target()`. `close()` prints the render time and
the video's length. Tests never render.
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
GREEN, RED, WHITE, AMBER = (120, 220, 120), (70, 70, 255), (255, 255, 255), (60, 190, 255)


def mode_from_argv(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", dest="mode", action="store_const", const="quick",
                   help="low resolution, for iterating (default)")
    g.add_argument("--full", dest="mode", action="store_const", const="full",
                   help="final take at full resolution")
    return ap.parse_args(argv).mode or "quick"


def _free_cam(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def _text(img, s, org, scale, colour, thick, anchor="left"):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = {"left": org[0], "right": org[0] - tw, "centre": org[0] - tw // 2}[anchor]
    cv2.rectangle(img, (x - 8, org[1] - th - 8), (x + tw + 8, org[1] + 10), (20, 20, 20), -1)
    cv2.putText(img, s, (x, org[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


class Recorder:
    def __init__(self, model, data, out: str, mode: str = "quick",
                 closeup_target: Callable[[], np.ndarray] | None = None,
                 overlay: Callable[[], tuple | dict] | None = None,
                 closeup_distance: float = 0.35, views: tuple[str, ...] | None = None,
                 sizes: dict[str, tuple[int, int]] | None = None, show_sites: bool = True,
                 target_s: float | None = None):
        """`overlay()` returns either (label, footer, footer BGR colour) or a dict with optional
        "top" (str), "bottom" (list of (str, colour) lines), "corner" ((str, colour), top-right
        under the speed label) and "banner" ((str, colour), large, centred)."""
        (self.w, self.h), self.views = MODES[mode]
        if sizes:
            self.w, self.h = sizes[mode]
        if views:
            self.views = views
        self.mode, self.model, self.data, self.target_s = mode, model, data, target_s
        self.target, self.overlay = closeup_target, overlay or (lambda: {})
        cols = 2 if len(self.views) > 1 else 1
        rows = (len(self.views) + cols - 1) // cols
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                      (cols * self.w, rows * self.h))
        self.renderer = mujoco.Renderer(model, height=self.h, width=self.w)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[4] = 1 if show_sites else 0
        self.cams = {"overview": _free_cam([*BENCH_CENTER, -0.1], 2.4, 135.0, -35.0),
                     "close-up": _free_cam([0, 0, 0], closeup_distance, 200.0, -30.0)}
        self.s = 1.0                    # current playback speed
        self.next_t = data.time
        self.frames, self.render_s, self.t0 = 0, 0.0, time.perf_counter()

    # ------------------------------------------------------------- capture control
    def speed(self, s: float) -> None:
        """Playback speed from now on: one frame per s/FPS simulated seconds."""
        self.s = float(s)
        self.next_t = min(self.next_t, self.data.time + self.s / FPS)

    def cut(self, camera: str) -> None:
        """Switch a single-view recorder to another camera (e.g. "racks" for a close-up)."""
        self.views = (camera,)

    def step(self) -> None:
        """Call after each physics step; renders a frame only when one is due in sim time."""
        if self.data.time + 1e-9 >= self.next_t:
            self.next_t += self.s / FPS
            self.frame()

    # ------------------------------------------------------------------ drawing
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
        ov = self.overlay()
        if isinstance(ov, tuple):                     # legacy (label, footer, colour)
            label, footer, colour = ov
            ov = {"top": label, "bottom": [(footer, colour)] if footer else []}
        W, H = img.shape[1], img.shape[0]
        k = W / 1280                                  # scale text with the frame width
        sc, th = 0.75 * k, max(1, round(2 * k))
        pad = int(24 * k)
        if ov.get("top"):
            _text(img, ov["top"], (pad, int(44 * k)), sc, WHITE, th)
        right_y = int(44 * k)
        if self.s > 1:
            _text(img, f"{self.s:g}x", (W - pad, right_y), sc, AMBER, th, "right")
            right_y += int(44 * k)
        if ov.get("corner"):
            s, c = ov["corner"]
            _text(img, s, (W - pad, right_y), sc, c, th, "right")
        for i, (s, c) in enumerate(reversed(ov.get("bottom", []))):
            _text(img, s, (pad, H - pad - int(i * 44 * k)), sc, c, th)
        if ov.get("banner"):
            s, c = ov["banner"]
            bs = 1.0 * k
            for j, line in enumerate(s.split("\n")):
                _text(img, line, (W // 2, H // 2 + int(j * 56 * k)), bs, c, max(2, th), "centre")
        self.writer.write(img)
        self.frames += 1
        self.render_s += time.perf_counter() - t

    def attach(self, skills, max_steps: int) -> Callable[[float], None]:
        """Record every physics step `skills` takes (wraps PipetteSkills._step), with a hard
        step cap so a demo can never hang. Returns hold(seconds): step the sim in place."""
        orig, n = skills._step, [0]

        def step():
            if n[0] >= max_steps:
                raise RuntimeError(f"step limit ({max_steps}) exceeded")
            orig()
            n[0] += 1
            self.step()
        skills._step = step

        def hold(seconds: float) -> None:
            for _ in range(int(seconds / self.model.opt.timestep)):
                step()
        return hold

    def hold_frames(self, seconds: float) -> None:
        """Repeat the current frame for `seconds` of video (a title/banner card) without
        stepping the simulation."""
        for _ in range(int(seconds * FPS)):
            self.frame()

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()
        total = time.perf_counter() - self.t0
        length = self.frames / FPS
        tgt = f" (target {self.target_s:.0f} s)" if self.target_s else ""
        print(f"wrote {self.out}  [{self.mode}: {self.frames} frames @ {FPS} fps]  "
              f"video length {length:.1f} s{tgt}  render {self.render_s:.1f} s of {total:.1f} s total")
